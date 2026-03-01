#!/usr/bin/env python3
"""
MPPI-based drone circle/lemniscate trajectory tracking with MuJoCo.

Architecture (cascade control):
  Outer loop (MPPI, 50 Hz) → desired [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
  Inner loop (PID, 200 Hz) → MuJoCo [body_thrust, x_moment, y_moment, z_moment]

Usage:
    python run_mppi.py [--radius 1.0] [--height 1.0] [--omega 0.5]
                       [--duration 20] [--render] [--save PATH]
                       [--n-samples 256] [--temperature 0.05]
"""

import argparse
import time
import numpy as np

from drone_mpc.drone_env import DroneEnv, quat_to_euler
from drone_mpc.mppi_controller import MPPIController
from drone_mpc.inner_loop import CascadeController
from drone_mpc.trajectory import CircleTrajectory, LemniscateTrajectory
from drone_mpc.visualization import plot_tracking_results


def main():
    parser = argparse.ArgumentParser(description="MPPI Drone Trajectory Tracking")
    parser.add_argument("--radius",      type=float, default=1.0,   help="Trajectory radius (m)")
    parser.add_argument("--height",      type=float, default=1.0,   help="Flight height (m)")
    parser.add_argument("--omega",       type=float, default=0.5,   help="Angular speed (rad/s)")
    parser.add_argument("--duration",    type=float, default=20.0,  help="Simulation duration (s)")
    parser.add_argument("--render",      action="store_true",       help="Enable MuJoCo viewer")
    parser.add_argument("--save",        type=str,   default=None,  help="Save plot to file")
    parser.add_argument("--trajectory",  type=str,   default="circle",
                        choices=["circle", "lemniscate"])
    parser.add_argument("--n-samples",   type=int,   default=256,   help="MPPI sample count K")
    parser.add_argument("--horizon",     type=int,   default=30,    help="MPPI horizon steps")
    parser.add_argument("--temperature", type=float, default=0.05,  help="MPPI temperature λ")
    parser.add_argument("--dt-ctrl",     type=float, default=0.02,  help="Outer-loop period (s)")
    parser.add_argument("--dt-sim",      type=float, default=0.005, help="Simulation timestep (s)")
    args = parser.parse_args()

    dt_sim  = args.dt_sim
    dt_ctrl = args.dt_ctrl
    sim_steps_per_ctrl = max(1, round(dt_ctrl / dt_sim))

    print("=" * 60)
    print("  MPPI Drone Trajectory Tracking  (cascade control)")
    print("=" * 60)
    print(f"  Trajectory : {args.trajectory}  r={args.radius}m  h={args.height}m  ω={args.omega}rad/s")
    print(f"  Duration   : {args.duration}s")
    print(f"  MPPI       : K={args.n_samples}  N={args.horizon}  λ={args.temperature}")
    print(f"  outer dt={dt_ctrl}s   sim dt={dt_sim}s")
    print("=" * 60)

    # ---- Environment -------------------------------------------------------
    env = DroneEnv(dt=dt_sim, render=args.render)

    # ---- Trajectory --------------------------------------------------------
    if args.trajectory == "circle":
        traj = CircleTrajectory(radius=args.radius, omega=args.omega,
                                center=(0.0, 0.0), height=args.height)
    else:
        traj = LemniscateTrajectory(scale=args.radius, omega=args.omega,
                                    center=(0.0, 0.0), height=args.height)

    # ---- Controllers -------------------------------------------------------
    Q  = np.diag([200., 200., 300., 0.5, 0.5, 0.5])
    Qf = np.diag([500., 500., 600., 5., 5., 5.])
    R  = np.diag([0.5, 5.0, 5.0])
    outer = MPPIController(dt=dt_ctrl, horizon=args.horizon,
                           n_samples=args.n_samples,
                           temperature=args.temperature,
                           mass=0.027, gravity=9.81,
                           Q=Q, R=R, Q_terminal=Qf, max_tilt_deg=10.0)
    inner = CascadeController(dt_inner=dt_sim)

    # ---- Initialise --------------------------------------------------------
    ref0  = traj.get_reference(0.0)
    state = env.reset(pos=ref0[:3])
    inner.reset()
    outer.reset()

    if args.render:
        env.launch_viewer()

    # ---- Logging buffers ---------------------------------------------------
    log_times, log_pos, log_ref, log_ctrl, log_compute = [], [], [], [], []

    # ---- Main loop ---------------------------------------------------------
    outer_cmd     = np.array([env.HOVER_THRUST, 0.0, 0.0, 0.0])
    n_steps       = int(args.duration / dt_sim)
    outer_counter = 0

    print("\nRunning simulation …")
    wall_start = time.time()

    for step in range(n_steps):
        t_sim = step * dt_sim

        # ---------- outer loop (MPPI) every sim_steps_per_ctrl steps ---------
        if outer_counter == 0:
            ref_seq = traj.get_reference_sequence(t_sim, outer.N, outer.dt)

            t0 = time.time()
            outer_cmd = outer.compute_control(state, ref_seq)
            t_solve   = time.time() - t0

            ref_now = traj.get_reference(t_sim)
            pos     = state[:3]
            log_times.append(t_sim)
            log_pos.append(pos.copy())
            log_ref.append(ref_now[:3].copy())
            log_ctrl.append(outer_cmd.copy())
            log_compute.append(t_solve)

            if step % (50 * sim_steps_per_ctrl) == 0:
                info = outer.get_info()
                err  = np.linalg.norm(pos - ref_now[:3])
                print(f"  t={t_sim:6.2f}s  pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                      f"  err={err:.4f}m  cost={info['best_cost']:.1f}"
                      f"  solve={t_solve*1e3:.1f}ms")

        outer_counter = (outer_counter + 1) % sim_steps_per_ctrl

        # ---------- inner loop (PID attitude) every simulation step ----------
        euler = quat_to_euler(state[3:7])
        z     = state[2]
        vz    = state[9]
        ctrl  = inner.step(outer_cmd, euler, z, vz, ang_vel=state[10:13])

        # ---------- step MuJoCo ---------------------------------------------
        state = env.step(ctrl)

        # Wall-clock pacing when rendering
        if args.render and env._viewer is not None:
            elapsed_wall = time.time() - wall_start
            elapsed_sim  = (step + 1) * dt_sim
            if elapsed_sim > elapsed_wall:
                time.sleep(elapsed_sim - elapsed_wall)

    # ---- Statistics --------------------------------------------------------
    times      = np.array(log_times)
    actual_pos = np.array(log_pos)
    ref_pos    = np.array(log_ref)
    controls   = np.array(log_ctrl)
    comp_times = np.array(log_compute)

    errors = np.linalg.norm(actual_pos - ref_pos, axis=1)
    rmse   = np.sqrt(np.mean(errors ** 2))

    print("\n" + "=" * 60)
    print(f"  RMSE         : {rmse:.4f} m")
    print(f"  Max error    : {np.max(errors):.4f} m")
    print(f"  Avg MPPI time: {np.mean(comp_times)*1e3:.1f} ms")
    print("=" * 60)

    plot_tracking_results(times, actual_pos, ref_pos, controls,
                          title_prefix="MPPI", save_path=args.save)
    env.close()

    return dict(times=times, actual_pos=actual_pos, ref_pos=ref_pos,
                controls=controls, compute_times=comp_times)


if __name__ == "__main__":
    main()
