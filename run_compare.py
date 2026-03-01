#!/usr/bin/env python3
"""
Compare MPC and MPPI controllers on the same trajectory (cascade control).

Usage:
    python run_compare.py [--radius 1.0] [--duration 20] [--save comparison.png]
"""

import argparse
import time
import numpy as np

from drone_mpc.drone_env import DroneEnv, quat_to_euler
from drone_mpc.mpc_controller import MPCController
from drone_mpc.mppi_controller import MPPIController
from drone_mpc.inner_loop import CascadeController
from drone_mpc.trajectory import CircleTrajectory, LemniscateTrajectory
from drone_mpc.visualization import compare_controllers, plot_tracking_results


def run_controller(env, outer, traj, duration, dt_ctrl, dt_sim, name="Controller"):
    """Run a cascade (outer + inner) controller and collect results."""
    inner = CascadeController(dt_inner=dt_sim)
    sim_steps_per_ctrl = max(1, round(dt_ctrl / dt_sim))

    ref0  = traj.get_reference(0.0)
    state = env.reset(pos=ref0[:3])
    inner.reset()
    if hasattr(outer, 'reset'):
        outer.reset()

    log_times, log_pos, log_ref, log_ctrl, log_compute = [], [], [], [], []

    outer_cmd     = np.array([env.HOVER_THRUST, 0.0, 0.0, 0.0])
    n_steps       = int(duration / dt_sim)
    outer_counter = 0

    print(f"\n  Running {name} …")

    for step in range(n_steps):
        t_sim = step * dt_sim

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

            if step % (100 * sim_steps_per_ctrl) == 0:
                err = np.linalg.norm(pos - ref_now[:3])
                print(f"    t={t_sim:6.2f}s  err={err:.4f}m  solve={t_solve*1e3:.1f}ms")

        outer_counter = (outer_counter + 1) % sim_steps_per_ctrl

        euler = quat_to_euler(state[3:7])
        ctrl  = inner.step(outer_cmd, euler, state[2], state[9], ang_vel=state[10:13])
        state = env.step(ctrl)

    times      = np.array(log_times)
    actual_pos = np.array(log_pos)
    ref_pos    = np.array(log_ref)
    controls   = np.array(log_ctrl)
    comp_times = np.array(log_compute)

    errors = np.linalg.norm(actual_pos - ref_pos, axis=1)
    rmse   = np.sqrt(np.mean(errors ** 2))
    print(f"    {name}: RMSE={rmse:.4f}m  Max={np.max(errors):.4f}m"
          f"  AvgSolve={np.mean(comp_times)*1e3:.1f}ms")

    return dict(times=times, actual_pos=actual_pos, ref_pos=ref_pos,
                controls=controls, compute_times=comp_times)


def main():
    parser = argparse.ArgumentParser(description="Compare MPC and MPPI")
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius (m)")
    parser.add_argument("--height", type=float, default=1.0, help="Flight height (m)")
    parser.add_argument("--omega", type=float, default=0.5, help="Angular speed (rad/s)")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument("--save", type=str, default=None, help="Save comparison plot")
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["circle", "lemniscate"])
    parser.add_argument("--dt-ctrl", type=float, default=0.02, help="Control timestep (s)")
    parser.add_argument("--dt-sim", type=float, default=0.005, help="Simulation timestep (s)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MPC vs MPPI Comparison")
    print("=" * 60)

    # Create environment
    env = DroneEnv(dt=args.dt_sim, render=False)

    # Create trajectory
    if args.trajectory == "circle":
        traj = CircleTrajectory(
            radius=args.radius, omega=args.omega,
            center=(0.0, 0.0), height=args.height,
        )
    else:
        traj = LemniscateTrajectory(
            scale=args.radius, omega=args.omega,
            center=(0.0, 0.0), height=args.height,
        )

    # --- Run MPC ---
    mpc = MPCController(dt=args.dt_ctrl, horizon=25, mass=0.027, gravity=9.81)
    mpc_results = run_controller(
        env, mpc, traj, args.duration, args.dt_ctrl, args.dt_sim, name="MPC"
    )

    # --- Run MPPI ---
    mppi = MPPIController(
        dt=args.dt_ctrl, horizon=30, n_samples=256,
        temperature=0.05, mass=0.027, gravity=9.81,
    )
    mppi_results = run_controller(
        env, mppi, traj, args.duration, args.dt_ctrl, args.dt_sim, name="MPPI"
    )

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("  Comparison Summary")
    print("-" * 60)

    for name, res in [("MPC", mpc_results), ("MPPI", mppi_results)]:
        errors = np.linalg.norm(res["actual_pos"] - res["ref_pos"], axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        max_err = np.max(errors)
        avg_ct = np.mean(res["compute_times"]) * 1000
        print(f"  {name:5s}: RMSE={rmse:.4f}m | Max Error={max_err:.4f}m | "
              f"Avg Compute={avg_ct:.1f}ms")

    print("=" * 60)

    # Plot comparison
    compare_controllers(
        {"MPC": mpc_results, "MPPI": mppi_results},
        save_path=args.save,
    )

    # Also plot individual results
    for name, res in [("MPC", mpc_results), ("MPPI", mppi_results)]:
        save_path_i = None
        if args.save:
            save_path_i = args.save.replace(".", f"_{name.lower()}.")
        plot_tracking_results(
            res["times"], res["actual_pos"], res["ref_pos"], res["controls"],
            title_prefix=name,
            save_path=save_path_i,
        )

    env.close()


if __name__ == "__main__":
    main()
