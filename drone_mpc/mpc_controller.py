"""
MPC Controller for Crazyflie 2 trajectory tracking.

Design (v2 — fixed after drift failure):
  State  (6): [x, y, z, vx, vy, vz]   — pure translational
  Control(3): [thrust_N, roll_rad, pitch_rad]
              yaw is held at 0 by inner-loop PID (not an MPC DoF)

  Dynamics (yaw=0):
    ax =  (T/m)*sin(pitch)
    ay = -(T/m)*sin(roll)
    az =  (T/m)*cos(roll)*cos(pitch) - g
"""

import numpy as np
import casadi as ca
from typing import Optional, Dict, Any


class MPCController:
    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 20,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_terminal: Optional[np.ndarray] = None,
        mass: float = 0.027,
        gravity: float = 9.81,
        max_thrust: float = 0.35,
        max_tilt_deg: float = 10.0,   # inner-loop responds in ~0.2s at ±10°
        verbose: bool = False,
    ):
        self.dt      = dt
        self.N       = horizon
        self.mass    = mass
        self.gravity = gravity
        self.n_state = 6
        self.n_ctrl  = 3
        self.verbose = verbose
        self.hover_thrust = mass * gravity

        if Q is None:
            Q = np.diag([100., 100., 200., 10., 10., 20.])
        if R is None:
            R = np.diag([0.5, 5.0, 5.0])
        if Q_terminal is None:
            Q_terminal = Q * 5.0

        self.Q  = Q
        self.R  = R
        self.Qf = Q_terminal

        max_tilt = np.deg2rad(max_tilt_deg)
        self.u_min   = np.array([0.0,        -max_tilt, -max_tilt])
        self.u_max   = np.array([max_thrust,  max_tilt,  max_tilt])
        self.u_hover = np.array([self.hover_thrust, 0.0, 0.0])

        self._build_solver()
        self._prev_u: Optional[np.ndarray] = None

    def _dynamics(self, x, u):
        vx, vy, vz     = x[3], x[4], x[5]
        T, roll, pitch = u[0], u[1], u[2]
        ax =  (T / self.mass) * ca.sin(pitch)
        ay = -(T / self.mass) * ca.sin(roll)
        az =  (T / self.mass) * ca.cos(roll) * ca.cos(pitch) - self.gravity
        return ca.vertcat(vx, vy, vz, ax, ay, az)

    def _build_solver(self):
        nx, nu, N = self.n_state, self.n_ctrl, self.N
        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)
        P = ca.SX.sym("P", nx + N * nx)

        Q  = ca.DM(self.Q)
        R  = ca.DM(self.R)
        Qf = ca.DM(self.Qf)
        uh = ca.DM(self.u_hover)

        cost = ca.SX(0.0)
        g, lbg, ubg = [], [], []

        g   += [X[:, 0] - P[:nx]];  lbg += [0.]*nx;  ubg += [0.]*nx

        for k in range(N):
            x_ref = P[nx + k*nx : nx + (k+1)*nx]
            cost += ca.mtimes([(X[:,k]-x_ref).T,  Q,  X[:,k]-x_ref])
            cost += ca.mtimes([(U[:,k]-uh).T,      R,  U[:,k]-uh])
            xk, uk = X[:,k], U[:,k]
            f1 = self._dynamics(xk, uk)
            f2 = self._dynamics(xk + self.dt/2*f1, uk)
            f3 = self._dynamics(xk + self.dt/2*f2, uk)
            f4 = self._dynamics(xk + self.dt*f3,   uk)
            xn = xk + (self.dt/6.)*(f1+2*f2+2*f3+f4)
            g   += [X[:,k+1] - xn];  lbg += [0.]*nx;  ubg += [0.]*nx

        x_ref_f = P[nx+(N-1)*nx : nx+N*nx]
        cost += ca.mtimes([(X[:,N]-x_ref_f).T, Qf, X[:,N]-x_ref_f])

        opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
        G        = ca.vertcat(*g)

        n_vars = (N+1)*nx + N*nu
        lbx = np.full(n_vars, -np.inf);  ubx = np.full(n_vars, np.inf)
        for k in range(N+1):
            i = k*nx
            lbx[i:i+3]   = -100.; ubx[i:i+3]   = 100.
            lbx[i+3:i+6] = -15.;  ubx[i+3:i+6] = 15.
        cs = (N+1)*nx
        for k in range(N):
            i = cs + k*nu
            lbx[i:i+nu] = self.u_min;  ubx[i:i+nu] = self.u_max

        self._lbx = lbx;  self._ubx = ubx

        nlp  = {"x": opt_vars, "f": cost, "g": G, "p": P}
        opts = {
            "ipopt.print_level":           3 if self.verbose else 0,
            "ipopt.max_iter":              200,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol":                   1e-4,
            "ipopt.acceptable_tol":        5e-3,
            "ipopt.mu_strategy":           "adaptive",
            "print_time":                  1 if self.verbose else 0,
        }
        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self._nx = nx;  self._nu = nu;  self._n_vars = n_vars

    def compute_control(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        x_mpc = self._to_6d(state)
        ref6  = np.zeros((self.N, 6))
        cols  = min(6, reference.shape[1])
        ref6[:, :cols] = reference[:, :cols]
        p = np.concatenate([x_mpc, ref6.flatten()])

        n_vars = self._n_vars
        x0 = np.zeros(n_vars)
        cs = (self.N+1)*self._nx
        if self._prev_u is not None:
            for k in range(self.N-1):
                x0[cs+k*self._nu : cs+(k+1)*self._nu] = self._prev_u[k+1]
            x0[cs+(self.N-1)*self._nu:] = self._prev_u[-1]
        else:
            for k in range(self.N):
                x0[cs+k*self._nu : cs+(k+1)*self._nu] = self.u_hover
        for k in range(self.N+1):
            x0[k*self._nx:(k+1)*self._nx] = x_mpc

        sol   = self._solver(x0=x0, lbx=self._lbx, ubx=self._ubx,
                             lbg=0., ubg=0., p=p)
        opt_x = np.array(sol["x"]).flatten()

        u_seq = []
        for k in range(self.N):
            u_seq.append(opt_x[cs+k*self._nu : cs+(k+1)*self._nu].copy())
        self._prev_u = np.array(u_seq)

        u0     = self._prev_u[0]
        thrust = float(np.clip(u0[0], 0.0, 0.35))
        roll   = float(np.clip(u0[1], self.u_min[1], self.u_max[1]))
        pitch  = float(np.clip(u0[2], self.u_min[2], self.u_max[2]))
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
        self._prev_u = None

    def get_info(self) -> Dict[str, Any]:
        return {"type": "MPC", "horizon": self.N, "dt": self.dt,
                "n_state": self.n_state, "n_ctrl": self.n_ctrl}
