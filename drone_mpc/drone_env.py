"""
MuJoCo Drone Simulation Environment.

Wraps the Crazyflie 2 MJCF model with a Gym-like interface
for MPC/MPPI control. Provides state extraction, control application,
and visualization support.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Any
from pathlib import Path


class DroneEnv:
    """
    MuJoCo-based quadrotor environment for trajectory tracking.

    The drone model uses 4 actuators:
        - body_thrust: total vertical thrust [0, 0.35]
        - x_moment: roll torque [-1, 1]
        - y_moment: pitch torque [-1, 1]
        - z_moment: yaw torque [-1, 1]

    State vector (13):
        [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
         position   quaternion       linear_vel    angular_vel

    Control vector (4):
        [thrust, roll_moment, pitch_moment, yaw_moment]
    """

    # Physical constants for Crazyflie 2
    # Note: 4 prop bodies add 4×0.1g = 0.4g; total model mass = 27.4g
    MASS = 0.0274  # kg  (27g body + 0.4g prop geoms)
    GRAVITY = 9.81  # m/s^2
    HOVER_THRUST = MASS * GRAVITY  # ≈ 0.2688 N

    def __init__(
        self,
        model_path: Optional[str] = None,
        dt: float = 0.005,
        render: bool = False,
    ):
        """
        Args:
            model_path: Path to the MuJoCo XML model. Defaults to bundled scene.xml
            dt: Simulation timestep (seconds)
            render: Whether to enable real-time rendering
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "drone", "bitcraze_crazyflie_2", "scene.xml",
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        # Main flight actuators (indices 0-3): thrust + 3 moments
        # Prop spin actuators (indices 4-7): visual-only velocity servos
        self.n_ctrl = 4   # external interface still uses 4 commands
        self.ctrl_limits = np.array([
            [self.model.actuator_ctrlrange[i, 0], self.model.actuator_ctrlrange[i, 1]]
            for i in range(4)   # only flight actuators
        ])

        # Prop-spin scaling: at hover thrust T_hover → target ~200 rad/s
        # omega = PROP_SCALE * thrust_N  (linear approximation)
        self._prop_scale = 200.0 / self.HOVER_THRUST   # ≈ 755 rad/s per N

        # Rendering
        self._render = render
        self._viewer = None

        self.reset()

    def reset(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset the environment to initial state.

        Args:
            pos: Optional initial position [x, y, z]

        Returns:
            state: Initial state vector (13,)
        """
        mujoco.mj_resetData(self.model, self.data)

        if pos is not None:
            self.data.qpos[0:3] = pos
        else:
            # Default hover position
            self.data.qpos[0:3] = [0.0, 0.0, 1.0]

        # Identity quaternion (upright)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Zero velocity
        self.data.qvel[:] = 0.0

        # Apply hover thrust
        self.data.ctrl[0] = self.HOVER_THRUST

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Extract the full drone state.

        Returns:
            state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13,)
        """
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[0:3].copy()
        angvel = self.data.qvel[3:6].copy()
        return np.concatenate([pos, quat, vel, angvel])

    def get_position(self) -> np.ndarray:
        """Get drone position [x, y, z]."""
        return self.data.qpos[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get drone linear velocity [vx, vy, vz]."""
        return self.data.qvel[0:3].copy()

    def get_quaternion(self) -> np.ndarray:
        """Get drone orientation quaternion [qw, qx, qy, qz]."""
        return self.data.qpos[3:7].copy()

    def get_euler(self) -> np.ndarray:
        """Get drone orientation as Euler angles [roll, pitch, yaw] in radians."""
        quat = self.get_quaternion()
        return quat_to_euler(quat)

    def get_angular_velocity(self) -> np.ndarray:
        """Get drone angular velocity [wx, wy, wz]."""
        return self.data.qvel[3:6].copy()

    def step(self, ctrl: np.ndarray) -> np.ndarray:
        """
        Apply control and step the simulation.

        Args:
            ctrl: Control vector [thrust, roll_moment, pitch_moment, yaw_moment] (4,)
                  Propeller spin speeds are derived automatically from thrust.

        Returns:
            state: New state vector (13,)
        """
        # Clip flight actuators
        ctrl_clipped = np.clip(ctrl, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        self.data.ctrl[0:4] = ctrl_clipped

        # Drive prop spin speed proportional to thrust by directly setting joint velocity.
        # This avoids numerical instability from torque-based velocity servos on tiny-inertia joints.
        # qvel indices: 0-2 = body translational, 3-5 = body rotational, 6-9 = 4 prop hinges
        if self.model.nv >= 10:
            omega = float(ctrl_clipped[0]) * self._prop_scale
            self.data.qvel[6] =  omega   # prop1 CCW
            self.data.qvel[7] =  omega   # prop2 CCW
            self.data.qvel[8] = -omega   # prop3 CW
            self.data.qvel[9] = -omega   # prop4 CW
            self.data.ctrl[4:8] = 0.0    # zero motor torque (speed is imposed directly)

        mujoco.mj_step(self.model, self.data)

        if self._render and self._viewer is not None:
            self._viewer.sync()

        return self.get_state()

    def step_multiple(self, ctrl: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Apply control for multiple simulation steps.

        Args:
            ctrl: Control vector (4,)
            n_steps: Number of simulation steps

        Returns:
            state: Final state vector (13,)
        """
        ctrl_clipped = np.clip(ctrl, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        self.data.ctrl[0:4] = ctrl_clipped
        if self.model.nv >= 10:
            omega = float(ctrl_clipped[0]) * self._prop_scale
            self.data.qvel[6] =  omega
            self.data.qvel[7] =  omega
            self.data.qvel[8] = -omega
            self.data.qvel[9] = -omega
            self.data.ctrl[4:8] = 0.0

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        if self._render and self._viewer is not None:
            self._viewer.sync()

        return self.get_state()

    def get_sim_state(self) -> Dict[str, np.ndarray]:
        """Get full simulator state for rollback."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": self.data.time,
        }

    def set_sim_state(self, state: Dict[str, np.ndarray]):
        """Restore full simulator state."""
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        self.data.time = state["time"]
        mujoco.mj_forward(self.model, self.data)

    def launch_viewer(self,
                      track_drone: bool = True,
                      cam_azimuth: float = 45.0,
                      cam_elevation: float = -30.0,
                      cam_distance: float = 1.0):
        """Launch the MuJoCo interactive viewer with optional drone-tracking camera.

        Args:
            track_drone:   If True, camera follows the drone (side-above view).
            cam_azimuth:   Camera azimuth angle (degrees). 45° = side-diagonal view.
            cam_elevation: Camera elevation angle (degrees). -30° = looking slightly down.
            cam_distance:  Distance from the tracked body (meters).
        """
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                    show_left_ui=False,
                                                    show_right_ui=False)
        self._render = True

        # ---- Trail geom buffer (stored in viewer's user_scn) ----
        self._trail_pos: list = []   # list of np.array([x,y,z]) waypoints
        self._trail_max = 2000       # keep last N segments (avoid unbounded growth)

        # ---- Camera: track drone body ----
        if track_drone:
            drone_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "cf2"
            )
            with self._viewer.lock():
                cam = self._viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = drone_body_id
                cam.azimuth   = cam_azimuth
                cam.elevation = cam_elevation
                cam.distance  = cam_distance

    def _update_trail(self, pos: np.ndarray):
        """Append a new position to the trail and redraw all line segments in user_scn.

        Called every outer-loop step while rendering.

        Args:
            pos: Current drone position [x, y, z]
        """
        if self._viewer is None or self._viewer.user_scn is None:
            return

        self._trail_pos.append(pos.copy())
        # Keep only the last _trail_max+1 points (= _trail_max segments)
        if len(self._trail_pos) > self._trail_max + 1:
            self._trail_pos = self._trail_pos[-(self._trail_max + 1):]

        scn = self._viewer.user_scn
        n_seg = len(self._trail_pos) - 1
        if n_seg <= 0:
            return

        # Cap segments to scene capacity
        n_seg = min(n_seg, scn.maxgeom)
        # Use geoms[0..n_seg-1] for trail
        for i in range(n_seg):
            g = scn.geoms[i]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),  # red, slightly transparent
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                3.0,                      # line width in pixels
                self._trail_pos[i],
                self._trail_pos[i + 1],
            )
        scn.ngeom = n_seg

    def add_trail_point(self, pos: np.ndarray):
        """Public interface: append a position to the flight trail (render mode only).

        Args:
            pos: Drone position [x, y, z] at this instant.
        """
        if self._render and self._viewer is not None:
            self._update_trail(pos)

    def close(self):
        """Close the viewer."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self.data.time


# ===================== Utility Functions =====================


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
    Uses ZYX convention (yaw-pitch-roll).
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def rotation_matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return R
