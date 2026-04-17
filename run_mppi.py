#!/usr/bin/env python3
"""
MPPI-based drone circle/lemniscate trajectory tracking with MuJoCo.

Architecture (cascade control):
  Outer loop (MPPI, 50 Hz) → desired [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
  Inner loop (PID, 200 Hz) → MuJoCo [body_thrust, x_moment, y_moment, z_moment]

Usage:
    python run_mppi.py [--radius 1.0] [--height 1.0] [--omega 0.5]
                       [--duration 20] [--render] [--save PATH]
                       [--save-gif out.gif]
                       [--n-samples 512] [--temperature 0.05] [--smoothing-alpha 0.05]
"""

import argparse
import atexit
import os
import time
import tempfile
import numpy as np
import mujoco

from drone_mpc.drone_env import DroneEnv, quat_to_euler
from drone_mpc.mppi_controller import MPPIController
from drone_mpc.inner_loop import CascadeController
from drone_mpc.trajectory import CircleTrajectory, LemniscateTrajectory
from drone_mpc.visualization import plot_tracking_results
from drone_mpc.mppi_risk import MLPRisk, AnalyticProximityRisk, default_analytic_risk_for_walls_scene


_TEMP_SCENE_FILES = set()


def _cleanup_temp_scene_files():
    for path in list(_TEMP_SCENE_FILES):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_scene_files)


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "drone", "bitcraze_crazyflie_2")


def _make_custom_walls_scene(args: argparse.Namespace) -> str:
    """Create a temporary scene XML with configurable wall spacing."""
    wall_half_gap = float(getattr(args, "wall_half_gap", 2.65))
    wall_half_thickness = float(getattr(args, "wall_half_thickness", 0.08))
    scene_dir = _models_dir()
    cf2_xml = os.path.join(scene_dir, "cf2.xml")

    fd, tmp_path = tempfile.mkstemp(prefix="scene_with_walls_", suffix=".xml", dir=scene_dir)
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"""<mujoco model=\"CF2 scene with walls\">
    <include file=\"{cf2_xml}\"/>

    <statistic center=\"0 0 0.1\" extent=\"0.2\" meansize=\".05\"/>

    <visual>
        <headlight diffuse=\"0.6 0.6 0.6\" ambient=\"0.3 0.3 0.3\" specular=\"0 0 0\"/>
        <rgba haze=\"0.15 0.25 0.35 1\"/>
        <global azimuth=\"-20\" elevation=\"-20\" ellipsoidinertia=\"true\"/>
    </visual>

    <asset>
        <texture type=\"skybox\" builtin=\"gradient\" rgb1=\"0.3 0.5 0.7\" rgb2=\"0 0 0\" width=\"512\" height=\"3072\"/>
        <texture type=\"2d\" name=\"groundplane\" builtin=\"checker\" mark=\"edge\" rgb1=\"0.2 0.3 0.4\" rgb2=\"0.1 0.2 0.3\"
            markrgb=\"0.8 0.8 0.8\" width=\"300\" height=\"300\"/>
        <material name=\"groundplane\" texture=\"groundplane\" texuniform=\"true\" texrepeat=\"5 5\" reflectance=\"0.2\"/>
    </asset>

    <worldbody>
        <light pos=\"0 0 1.5\" dir=\"0 0 -1\" directional=\"true\"/>
        <geom name=\"floor\" size=\"0 0 0.05\" type=\"plane\" material=\"groundplane\"/>
        <geom name=\"wall_pos_x\" type=\"box\" size=\"{wall_half_thickness:.4f} 5 2.5\" pos=\"{wall_half_gap:.4f} 0 1.25\"
              rgba=\"0.85 0.35 0.35 0.4\" contype=\"1\" conaffinity=\"1\"/>
        <geom name=\"wall_neg_x\" type=\"box\" size=\"{wall_half_thickness:.4f} 5 2.5\" pos=\"{-wall_half_gap:.4f} 0 1.25\"
              rgba=\"0.85 0.35 0.35 0.4\" contype=\"1\" conaffinity=\"1\"/>
    </worldbody>
</mujoco>
""")
    _TEMP_SCENE_FILES.add(tmp_path)
    return tmp_path


def _resolve_scene_path(args: argparse.Namespace):
    if getattr(args, "scene", "default") == "with_walls":
        if (
            abs(float(getattr(args, "wall_half_gap", 2.65)) - 2.65) > 1e-9
            or abs(float(getattr(args, "wall_half_thickness", 0.08)) - 0.08) > 1e-9
        ):
            return _make_custom_walls_scene(args)
        return os.path.join(_models_dir(), "scene_with_walls.xml")
    return None


def _build_risk(args: argparse.Namespace):
    """Returns (risk_model_or_None, risk_weight)."""
    mode = getattr(args, "risk", "none")
    w = float(getattr(args, "risk_weight", 0.0))
    if mode == "none":
        return None, 0.0
    if mode == "analytic":
        if getattr(args, "scene", "default") == "with_walls":
            wall_half_gap = float(getattr(args, "wall_half_gap", 2.65))
            wall_half_thickness = float(getattr(args, "wall_half_thickness", 0.08))
            inner_boundary = wall_half_gap - wall_half_thickness
            # Keep the same 0.12 m inset used in the default risk config.
            x_bound = max(0.2, inner_boundary - 0.12)
            return AnalyticProximityRisk(
                x_bounds=(-x_bound, x_bound),
                y_bounds=(-10.0, 10.0),
                z_bounds=(0.08, 3.0),
                margin=0.35,
            ), w
        return default_analytic_risk_for_walls_scene(), w
    if mode == "learned":
        path = getattr(args, "risk_weights", None)
        if not path:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_mpc", "risk_mlp_default.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Learned risk weights not found: {path}\n"
                "Run: python scripts/train_mppi_risk_mlp.py --out drone_mpc/risk_mlp_default.npz"
            )
        return MLPRisk(path), w
    raise ValueError(f"Unknown --risk {mode}")


def main():
    parser = argparse.ArgumentParser(description="MPPI Drone Trajectory Tracking")
    parser.add_argument("--radius",      type=float, default=1.0,   help="Trajectory radius (m)")
    parser.add_argument("--height",      type=float, default=1.0,   help="Flight height (m)")
    parser.add_argument("--omega",       type=float, default=0.5,   help="Angular speed (rad/s)")
    parser.add_argument("--duration",    type=float, default=20.0,  help="Simulation duration (s)")
    parser.add_argument("--loop-render", action="store_true",
                        help="Keep rendering continuously until viewer window is closed")
    parser.add_argument("--render",      action="store_true",       help="Enable MuJoCo viewer")
    parser.add_argument("--save",        type=str,   default=None,  help="Save plot to file")
    parser.add_argument("--save-gif",    type=str,   default=None,  help="Save simulation animation to GIF")
    parser.add_argument("--gif-fps",     type=int,   default=30,    help="GIF frame rate")
    parser.add_argument("--gif-width",   type=int,   default=640,   help="GIF frame width")
    parser.add_argument("--gif-height",  type=int,   default=360,   help="GIF frame height")
    parser.add_argument("--gif-stride",  type=int,   default=4,     help="Capture every N sim steps for GIF")
    parser.add_argument("--trajectory",  type=str,   default="circle",
                        choices=["circle", "lemniscate"])
    parser.add_argument("--n-samples",   type=int,   default=512,   help="MPPI sample count K (REPORT §3.5 default)")
    parser.add_argument("--horizon",     type=int,   default=30,    help="MPPI horizon steps")
    parser.add_argument("--temperature", type=float, default=0.05,  help="MPPI temperature λ")
    parser.add_argument("--smoothing-alpha", type=float, default=0.05,
                        help="Temporal smoothing on U_nominal (REPORT §3.5 default 0.05)")
    parser.add_argument("--dt-ctrl",     type=float, default=0.02,  help="Outer-loop period (s)")
    parser.add_argument("--dt-sim",      type=float, default=0.005, help="Simulation timestep (s)")
    parser.add_argument("--seed",        type=int,   default=42,    help="RNG seed for MPPI sampling")
    parser.add_argument(
        "--scene",
        type=str,
        default="default",
        choices=["default", "with_walls"],
        help="MuJoCo scene: default floor-only or with vertical walls (match analytic risk)",
    )
    parser.add_argument(
        "--risk",
        type=str,
        default="none",
        choices=["none", "analytic", "learned"],
        help="Add proximity risk to MPPI cost (REPORT §4.2.3)",
    )
    parser.add_argument(
        "--risk-weight",
        type=float,
        default=0.35,
        help="Multiplier on risk cost (use lower if tracking degrades)",
    )
    parser.add_argument(
        "--risk-weights",
        type=str,
        default=None,
        help="Path to risk_mlp.npz for --risk learned",
    )
    parser.add_argument(
        "--wall-half-gap",
        type=float,
        default=2.65,
        help="Only for --scene with_walls: wall center at x=±value (smaller = narrower corridor)",
    )
    parser.add_argument(
        "--wall-half-thickness",
        type=float,
        default=0.08,
        help="Only for --scene with_walls: wall half-thickness in x (m)",
    )
    args = parser.parse_args()

    dt_sim  = args.dt_sim
    dt_ctrl = args.dt_ctrl
    sim_steps_per_ctrl = max(1, round(dt_ctrl / dt_sim))

    print("=" * 60)
    print("  MPPI Drone Trajectory Tracking  (cascade control)")
    print("=" * 60)
    print(f"  Trajectory : {args.trajectory}  r={args.radius}m  h={args.height}m  ω={args.omega}rad/s")
    print(f"  Duration   : {args.duration}s")
    if args.loop_render:
        print("  Loop mode  : ON (run until viewer is closed)")
    print(f"  MPPI       : K={args.n_samples}  N={args.horizon}  λ={args.temperature}  "
          f"α_smooth={args.smoothing_alpha}")
    print(f"  outer dt={dt_ctrl}s   sim dt={dt_sim}s  seed={args.seed}")
    print(f"  scene={args.scene}   risk={args.risk}  risk_w={args.risk_weight if args.risk != 'none' else 0.0}")
    if args.scene == "with_walls":
        corridor_inner = 2.0 * (args.wall_half_gap - args.wall_half_thickness)
        print(f"  walls      : center=±{args.wall_half_gap:.2f}m  thickness={2*args.wall_half_thickness:.2f}m  inner_gap≈{corridor_inner:.2f}m")
    print("=" * 60)

    risk_model, rw = _build_risk(args)

    # ---- Environment -------------------------------------------------------
    env = DroneEnv(dt=dt_sim, render=args.render, model_path=_resolve_scene_path(args))

    # ---- Trajectory --------------------------------------------------------
    if args.trajectory == "circle":
        traj = CircleTrajectory(radius=args.radius, omega=args.omega,
                                center=(0.0, 0.0), height=args.height)
    else:
        traj = LemniscateTrajectory(scale=args.radius, omega=args.omega,
                                    center=(0.0, 0.0), height=args.height)

    # ---- Controllers -------------------------------------------------------
    # Q: [x, y, z, vx, vy, vz]  — increase vz (was 0.5) to dampen Z oscillation
    Q  = np.diag([200., 200., 300., 0.5, 0.5, 5.0])
    Qf = np.diag([500., 500., 600., 5.,  5.,  20.])
    # R: [thrust, roll, pitch]  — increase attitude penalty to reduce control noise
    R  = np.diag([1.0, 20.0, 20.0])
    outer = MPPIController(
        dt=dt_ctrl,
        horizon=args.horizon,
        n_samples=args.n_samples,
        temperature=args.temperature,
        mass=env.MASS,
        gravity=9.81,
        Q=Q,
        R=R,
        Q_terminal=Qf,
        max_tilt_deg=10.0,
        smoothing_alpha=args.smoothing_alpha,
        seed=args.seed,
        risk_model=risk_model,
        risk_weight=rw,
    )
    inner = CascadeController(dt_inner=dt_sim)

    # ---- Initialise --------------------------------------------------------
    ref0  = traj.get_reference(0.0)
    state = env.reset(pos=ref0[:3])
    inner.reset()
    outer.reset()

    if args.render:
        env.launch_viewer(track_drone=True, cam_azimuth=45.0,
                          cam_elevation=-30.0, cam_distance=5.0)

    gif_renderer = None
    gif_writer = None
    if args.save_gif:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "GIF export requires imageio. Install with: pip install imageio pillow"
            ) from exc

        gif_path = os.path.abspath(args.save_gif)
        gif_dir = os.path.dirname(gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)

        gif_renderer = mujoco.Renderer(env.model, height=args.gif_height, width=args.gif_width)
        gif_writer = imageio.get_writer(gif_path, mode="I", fps=args.gif_fps, loop=0)
        print(f"  GIF output : {gif_path}  ({args.gif_width}x{args.gif_height}, fps={args.gif_fps}, stride={args.gif_stride})")

    # ---- Logging buffers ---------------------------------------------------
    log_times, log_pos, log_ref, log_ctrl, log_compute = [], [], [], [], []

    # ---- Main loop ---------------------------------------------------------
    outer_cmd     = np.array([env.HOVER_THRUST, 0.0, 0.0, 0.0])
    n_steps       = int(args.duration / dt_sim)
    outer_counter = 0
    step = 0

    print("\nRunning simulation …")
    wall_start = time.time()

    while step < n_steps or args.loop_render:
        if args.loop_render and args.render and env._viewer is not None:
            # Stop loop when user closes the MuJoCo window.
            if hasattr(env._viewer, "is_running") and not env._viewer.is_running():
                print("\nViewer closed by user. Exiting loop mode.")
                break

        t_sim = step * dt_sim

        # ---------- outer loop (MPPI) every sim_steps_per_ctrl steps ---------
        if outer_counter == 0:
            ref_seq = traj.get_reference_sequence(t_sim, outer.N, outer.dt)

            t0 = time.time()
            outer_cmd = outer.compute_control(state, ref_seq)
            t_solve   = time.time() - t0

            ref_now = traj.get_reference(t_sim)
            pos     = state[:3]
            if not args.loop_render:
                log_times.append(t_sim)
                log_pos.append(pos.copy())
                log_ref.append(ref_now[:3].copy())
                log_ctrl.append(outer_cmd.copy())
                log_compute.append(t_solve)

            # Append current position to flight trail (render mode)
            env.add_trail_point(pos)

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

        if gif_renderer is not None and gif_writer is not None and (step % max(1, args.gif_stride) == 0):
            gif_renderer.update_scene(env.data, camera="track")
            gif_writer.append_data(gif_renderer.render())

        # Wall-clock pacing when rendering
        if args.render and env._viewer is not None:
            elapsed_wall = time.time() - wall_start
            elapsed_sim  = (step + 1) * dt_sim
            if elapsed_sim > elapsed_wall:
                time.sleep(elapsed_sim - elapsed_wall)

        step += 1

    # ---- Statistics --------------------------------------------------------
    if len(log_times) == 0:
        print("\nNo summary statistics collected (loop-render mode).")
        if gif_writer is not None:
            gif_writer.close()
        if gif_renderer is not None:
            gif_renderer.close()
        if args.save_gif:
            print(f"  GIF saved    : {os.path.abspath(args.save_gif)}")
        env.close()
        return dict(times=np.array([]), actual_pos=np.array([]), ref_pos=np.array([]),
                    controls=np.array([]), compute_times=np.array([]))

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

    if gif_writer is not None:
        gif_writer.close()
    if gif_renderer is not None:
        gif_renderer.close()
    if args.save_gif:
        print(f"  GIF saved    : {os.path.abspath(args.save_gif)}")

    if args.loop_render:
        env.close()
        return dict(times=np.array([]), actual_pos=np.array([]), ref_pos=np.array([]),
                    controls=np.array([]), compute_times=np.array([]))

    try:
        plot_tracking_results(times, actual_pos, ref_pos, controls,
                              title_prefix="MPPI", save_path=args.save)
    except RuntimeError as exc:
        # macOS backend can fail under mjpython/worker-thread contexts; retry headless.
        msg = str(exc)
        if "FigureManager" in msg or "MacOS backend" in msg:
            print("  Plot warning : GUI backend unavailable, retrying with Agg backend.")
            import matplotlib.pyplot as plt

            plt.switch_backend("Agg")
            plot_tracking_results(times, actual_pos, ref_pos, controls,
                                  title_prefix="MPPI", save_path=args.save)
        else:
            raise
    env.close()

    return dict(times=times, actual_pos=actual_pos, ref_pos=ref_pos,
                controls=controls, compute_times=comp_times)


if __name__ == "__main__":
    main()
