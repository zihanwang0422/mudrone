"""
Inner-loop PD attitude controller for Crazyflie 2 in MuJoCo.

Architecture:
  Outer loop (MPC/MPPI) → [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
  Inner loop (PD)        → [body_thrust, x_moment_ctrl, y_moment_ctrl, z_moment_ctrl]

CF2 model actuators (cf2.xml, gear=0.001):
  ctrl[0]  body_thrust  [0, 0.35] N
  ctrl[1]  x_moment     [-1, 1]   → actual moment = ctrl*0.001 N·m (roll)
  ctrl[2]  y_moment     [-1, 1]   → actual moment = ctrl*0.001 N·m (pitch)
  ctrl[3]  z_moment     [-1, 1]   → actual moment = ctrl*0.001 N·m (yaw)

PD uses angular-rate (gyro) feedback for the D term to avoid noise amplification.

Closed-loop tuning (gear=0.001, Ixx=Iyy=2.4e-5, Izz=3.2e-5):
  Roll/Pitch: ωn=25 rad/s, ζ=0.9
    kp = Ixx*ωn²/gear = 15.0
    kd = 2*ζ*Ixx*ωn/gear = 1.08
  Yaw: ωn=20 rad/s, ζ=1.0
    kp = Izz*ωn²/gear = 12.9
    kd = 2*ζ*Izz*ωn/gear = 1.29
"""

import numpy as np


class AttitudePD:
    """
    Inner-loop PD converting outer-loop commands to MuJoCo actuator inputs.

    Uses angular-rate (gyro) feedback for D term.
    """

    MASS = 0.027
    G    = 9.81
    IXX  = 2.3951e-5
    IYY  = 2.3951e-5
    IZZ  = 3.2347e-5
    GEAR = 0.001   # magnitude of gear (cf2.xml uses gear=-0.001, so sign is handled below)

    # Empirically tuned for gear=0.001 magnitude, dt=5ms
    # ctrl = -(KP*(angle_err) - KD*rate)  (minus sign due to gear=-0.001 in XML)
    KP_ROLL  = 5.0
    KD_ROLL  = 0.15
    KI_ROLL  = 0.3
    KP_PITCH = 5.0
    KD_PITCH = 0.15
    KI_PITCH = 0.3
    KP_YAW   = 4.0
    KD_YAW   = 0.12
    KI_RP    = 0.3

    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self._yaw_sp       = 0.0
        self._roll_integ   = 0.0
        self._pitch_integ  = 0.0

    def reset(self):
        self._yaw_sp      = 0.0
        self._roll_integ  = 0.0
        self._pitch_integ = 0.0

    def compute(
        self,
        outer_cmd: np.ndarray,
        euler: np.ndarray,
        z: float        = 0.0,
        vz: float       = 0.0,
        ang_vel: np.ndarray = None,
    ) -> np.ndarray:
        """
        Args:
            outer_cmd : [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
            euler     : [roll, pitch, yaw] rad
            z, vz     : altitude / rate (kept for API compat, thrust is feedforward)
            ang_vel   : body angular rates [p, q, r] rad/s from MuJoCo gyro
        Returns:
            ctrl : [body_thrust, x_moment_ctrl, y_moment_ctrl, z_moment_ctrl]
        """
        thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd = outer_cmd
        body_thrust = float(np.clip(thrust_cmd, 0.0, 0.35))

        p, q, r = (0., 0., 0.) if ang_vel is None else (
            float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]))

        # ---- roll ----
        roll_err = roll_cmd - euler[0]
        self._roll_integ = float(np.clip(
            self._roll_integ + roll_err * self.dt, -0.3, 0.3))
        # minus sign: gear="...  -0.001 ..." → ctrl>0 gives negative roll
        x_moment = float(np.clip(
            -(self.KP_ROLL * roll_err - self.KD_ROLL * p) - self.KI_ROLL * self._roll_integ,
            -1., 1.))

        # ---- pitch ----
        pitch_err = pitch_cmd - euler[1]
        self._pitch_integ = float(np.clip(
            self._pitch_integ + pitch_err * self.dt, -0.3, 0.3))
        y_moment = float(np.clip(
            -(self.KP_PITCH * pitch_err - self.KD_PITCH * q) - self.KI_PITCH * self._pitch_integ,
            -1., 1.))

        # ---- yaw ----
        self._yaw_sp += yaw_rate_cmd * self.dt
        yaw_err = (self._yaw_sp - euler[2] + np.pi) % (2 * np.pi) - np.pi
        z_moment = float(np.clip(
            -(self.KP_YAW * yaw_err - self.KD_YAW * r),
            -1., 1.))

        return np.array([body_thrust, x_moment, y_moment, z_moment])


# Backwards-compatible aliases
AttitudePID = AttitudePD


class CascadeController:
    """Outer-loop (MPC/MPPI) + inner-loop (AttitudePD) cascade."""

    def __init__(self, dt_inner: float = 0.005):
        self.inner = AttitudePD(dt=dt_inner)

    def reset(self):
        self.inner.reset()

    def step(
        self,
        outer_cmd: np.ndarray,
        euler: np.ndarray,
        z: float        = 0.0,
        vz: float       = 0.0,
        ang_vel: np.ndarray = None,
    ) -> np.ndarray:
        return self.inner.compute(outer_cmd, euler, z, vz, ang_vel)
