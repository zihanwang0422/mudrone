"""
MPPI Controller for Crazyflie 2 trajectory tracking.

Design (v2 — aligned with MPC redesign):
  State  (6): [x, y, z, vx, vy, vz]
  Control(3): [thrust_N, roll_rad, pitch_rad]
              yaw held at 0 by inner-loop PID

  Dynamics (yaw=0):
    ax =  (T/m)*sin(pitch)
    ay = -(T/m)*sin(roll)
    az =  (T/m)*cos(roll)*cos(pitch) - g

  Output: [thrust, roll, pitch, 0.0]  → AttitudePID
"""

import numpy as np
from typing import Optional, Dict, Any


class MPPIController:
    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 30,
        n_samples: int = 512,
        temperature: float = 0.05,
        noise_sigma: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_terminal: Optional[np.ndarray] = None,
        mass: float = 0.027,
        gravity: float = 9.81,
        max_thrust: float = 0.35,
        max_tilt_deg: float = 10.0,
        smoothing_alpha: float = 0.05,
        seed: int = 42,
    ):
        self.dt = dt
        self.N = horizon
        self.K = n_samples
        self.lam = temperature
        self.mass = mass
        self.gravity = gravity
        self.n_state = 6
        self.n_ctrl  = 3
        self.smoothing_alpha = smoothing_alpha
        self.hover_thrust = mass * gravity

        if noise_sigma is None:
            noise_sigma = np.array([0.04, 0.08, 0.08])
        self.noise_sigma = noise_sigma

        if Q is None:
            Q = np.diag([200., 200., 300., 0.5, 0.5, 0.5])
        if R is None:
            R = np.diag([0.5, 5.0, 5.0])
        if Q_terminal is None:
            Q_terminal = Q * 5.0

        self.Q          = Q
        self.R          = R
        self.Q_terminal = Q_terminal

        max_tilt = np.deg2rad(max_tilt_deg)
        self.u_min   = np.array([0.0,         -max_tilt, -max_tilt])
        self.u_max   = np.array([max_thrust,   max_tilt,  max_tilt])
        self.u_hover = np.array([self.hover_thrust, 0.0, 0.0])

        self.U_nominal = np.tile(self.u_hover, (self.N, 1)).astype(float)
        self.rng = np.random.default_rng(seed)
        self._best_cost = np.inf
        self._mean_cost = np.inf

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised: x (K,6), u (K,3) → dx (K,6)"""
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :];  u = u[np.newaxis, :]

        vx, vy, vz     = x[:,3], x[:,4], x[:,5]
        T, roll, pitch = u[:,0], u[:,1], u[:,2]

        ax =  (T / self.mass) * np.sin(pitch)
        ay = -(T / self.mass) * np.sin(roll)
        az =  (T / self.mass) * np.cos(roll) * np.cos(pitch) - self.gravity

        dx = np.column_stack([vx, vy, vz, ax, ay, az])
        return dx[0] if single else dx

    def _dynamics_step(self, x, u):
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5*self.dt*k1, u)
        k3 = self._dynamics(x + 0.5*self.dt*k2, u)
        k4 = self._dynamics(x + self.dt*k3,      u)
        return x + (self.dt/6.)*(k1 + 2*k2 + 2*k3 + k4)

    def _rollout(self, x0: np.ndarray, U_samples: np.ndarray, refs: np.ndarray) -> np.ndarray:
        K = U_samples.shape[0]
        costs = np.zeros(K)
        x = np.tile(x0, (K, 1))

        for t in range(self.N):
            x_ref = refs[t]
            err   = x - x_ref
            costs += np.einsum('ki,ij,kj->k', err,  self.Q, err)
            u_err = U_samples[:, t, :] - self.u_hover
            costs += np.einsum('ki,ij,kj->k', u_err, self.R, u_err)
            x = self._dynamics_step(x, U_samples[:, t, :])
            bad = np.any(~np.isfinite(x), axis=1)
            costs[bad] = 1e10

        x_ref_f = refs[-1]
        err_f   = x - x_ref_f
        costs  += np.einsum('ki,ij,kj->k', err_f, self.Q_terminal, err_f)
        return costs

    def compute_control(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        x0 = self._to_6d(state)

        ref6 = np.zeros((self.N, 6))
        cols = min(6, reference.shape[1])
        rows = min(self.N, reference.shape[0])
        ref6[:rows, :cols] = reference[:rows, :cols]
        # pad remaining rows with the last reference
        if rows < self.N:
            ref6[rows:] = ref6[rows-1]

        noise    = self.rng.normal(0, 1, (self.K, self.N, self.n_ctrl)) * self.noise_sigma
        U_samp   = np.clip(self.U_nominal[np.newaxis] + noise, self.u_min, self.u_max)
        costs    = self._rollout(x0, U_samp, ref6)

        cost_min = np.min(costs)
        w = np.exp(-(costs - cost_min) / self.lam)
        w_sum = w.sum()
        if w_sum < 1e-10:
            w = np.ones(self.K) / self.K
        else:
            w /= w_sum

        self.U_nominal += np.sum(w[:, np.newaxis, np.newaxis] * noise, axis=0)
        if self.smoothing_alpha > 0:
            for t in range(1, self.N):
                self.U_nominal[t] = (self.smoothing_alpha * self.U_nominal[t-1]
                                     + (1-self.smoothing_alpha) * self.U_nominal[t])
        self.U_nominal = np.clip(self.U_nominal, self.u_min, self.u_max)

        ctrl = self.U_nominal[0].copy()
        self.U_nominal = np.roll(self.U_nominal, -1, axis=0)
        self.U_nominal[-1] = self.u_hover

        self._best_cost = float(cost_min)
        self._mean_cost = float(np.mean(costs))

        thrust = float(np.clip(ctrl[0], 0.0, 0.35))
        roll   = float(np.clip(ctrl[1], self.u_min[1], self.u_max[1]))
        pitch  = float(np.clip(ctrl[2], self.u_min[2], self.u_max[2]))
        return np.array([thrust, roll, pitch, 0.0])

    def _to_6d(self, state: np.ndarray) -> np.ndarray:
        if len(state) == 13:
            return np.concatenate([state[0:3], state[7:10]])
        if len(state) == 9:
            return np.concatenate([state[0:3], state[3:6]])
        if len(state) == 6:
            return state.copy()
        raise ValueError(f"Unsupported state dim {len(state)}")

    def reset(self):
        self.U_nominal = np.tile(self.u_hover, (self.N, 1)).astype(float)
        self._best_cost = np.inf
        self._mean_cost = np.inf

    def get_info(self) -> Dict[str, Any]:
        return {"type": "MPPI", "horizon": self.N, "n_samples": self.K,
                "temperature": self.lam, "best_cost": self._best_cost,
                "mean_cost": self._mean_cost, "dt": self.dt}
