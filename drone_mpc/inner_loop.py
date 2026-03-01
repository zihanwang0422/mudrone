"""
Inner-loop PD attitude controller for Crazyflie 2 in MuJoCo.

Architecture:
  Outer loop (MPC/MPPI) → [thrust_N, roll_world_rad, pitch_world_rad, yaw_rate_rads]
  Inner loop (PD)        → [body_thrust, x_moment_ctrl, y_moment_ctrl, z_moment_ctrl]

Coordinate-frame note
---------------------
The MPC/MPPI outer loop works in the WORLD frame:
  ax =  (T/m)*sin(pitch_world)
  ay = -(T/m)*sin(roll_world)
These "world-frame tilt angles" must be projected onto the BODY frame before
the attitude PD can track them, otherwise any non-zero yaw causes cross-axis
coupling and uncontrolled oscillation.

Conversion (yaw=ψ is the current heading):
  roll_body  =  cos(ψ)*roll_world + sin(ψ)*pitch_world
  pitch_body = -sin(ψ)*roll_world + cos(ψ)*pitch_world

CF2 model actuators (cf2.xml, gear=-0.001):
  ctrl[0]  body_thrust  [0, 0.35] N
  ctrl[1]  x_moment     [-1, 1]   → actual moment = ctrl*(-0.001) N·m  (roll)
  ctrl[2]  y_moment     [-1, 1]   → actual moment = ctrl*(-0.001) N·m  (pitch)
  ctrl[3]  z_moment     [-1, 1]   → actual moment = ctrl*(-0.001) N·m  (yaw)

PD uses angular-rate (gyro) feedback for the D term to avoid noise amplification.
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

    # Tuned gains (gear=−0.001, Ixx≈2.4e-5 kg·m²):
    # Target bandwidth ~60 rad/s (≈10 Hz) so the inner loop settles in <100 ms
    # and can cleanly track MPC commands at 50 Hz outer-loop rate.
    #
    # With gear magnitude g=0.001:
    #   ωn = sqrt(kp * g / Ixx) = sqrt(15 * 0.001 / 2.4e-5) ≈ 25 rad/s  ← too slow
    # Use kp=25, kd=0.5 → ωn≈32 rad/s, ζ≈0.8 → ~100ms settle time
    # ctrl = -(KP*err - KD*rate) - KI*integ  (minus due to gear=-0.001 in XML)
    KP_ROLL  = 25.0
    KD_ROLL  = 0.5
    KI_ROLL  = 0.5
    KP_PITCH = 25.0
    KD_PITCH = 0.5
    KI_PITCH = 0.5
    KP_YAW   = 10.0
    KD_YAW   = 0.3
    KI_RP    = 0.5

    # First-order low-pass on tilt commands: τ=0.06s → smooths MPC jitter
    # while keeping step response rise time ≈ 0.1s
    CMD_TAU  = 0.06   # seconds

    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self._yaw_sp       = 0.0
        self._roll_integ   = 0.0
        self._pitch_integ  = 0.0
        # Low-pass filtered tilt commands (world frame)
        self._roll_filt    = 0.0
        self._pitch_filt   = 0.0

    def reset(self):
        self._yaw_sp       = 0.0
        self._roll_integ   = 0.0
        self._pitch_integ  = 0.0
        self._roll_filt    = 0.0
        self._pitch_filt   = 0.0

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
            outer_cmd : [thrust_N, roll_world_rad, pitch_world_rad, yaw_rate_rads]
                        roll/pitch are given in the WORLD frame by MPC/MPPI.
            euler     : [roll, pitch, yaw] rad  (body frame, from MuJoCo)
            z, vz     : altitude / rate (kept for API compat, thrust is feedforward)
            ang_vel   : body angular rates [p, q, r] rad/s from MuJoCo gyro
        Returns:
            ctrl : [body_thrust, x_moment_ctrl, y_moment_ctrl, z_moment_ctrl]
        """
        thrust_cmd, roll_world, pitch_world, yaw_rate_cmd = outer_cmd
        body_thrust = float(np.clip(thrust_cmd, 0.0, 0.35))

        p, q, r = (0., 0., 0.) if ang_vel is None else (
            float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]))

        # --- Low-pass filter on world-frame tilt commands -------------------
        # Smooths rapid MPC command reversals so the inner loop can keep up.
        alpha = self.dt / (self.CMD_TAU + self.dt)   # ≈ 0.077 at dt=5ms, τ=60ms
        self._roll_filt  += alpha * (roll_world  - self._roll_filt)
        self._pitch_filt += alpha * (pitch_world - self._pitch_filt)

        # --- World→Body frame rotation for tilt commands -------------------
        # MPC outputs roll/pitch in world frame (yaw=0 assumption).
        # Project onto current body heading ψ so inner loop tracks correctly
        # regardless of actual yaw.
        #   roll_body  =  cos(ψ)*roll_world + sin(ψ)*pitch_world
        #   pitch_body = -sin(ψ)*roll_world + cos(ψ)*pitch_world
        yaw = euler[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        roll_cmd  =  cy * self._roll_filt + sy * self._pitch_filt
        pitch_cmd = -sy * self._roll_filt + cy * self._pitch_filt

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
