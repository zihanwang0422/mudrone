"""
Proximity risk models for MPPI cost augmentation (REPORT §4.2.3).

- AnalyticProximityRisk: axis-aligned safe box; penalty when near boundary.
- MLPRisk: 2-layer ReLU MLP in pure NumPy; weights from train_mppi_risk_mlp.py (.npz).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class RiskModel(ABC):
    """Per-step additive risk cost for MPPI rollouts (vectorised over K samples)."""

    @abstractmethod
    def step_cost(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: state batch (K, 6) — [x,y,z,vx,vy,vz]
        Returns:
            cost increment per sample (K,), non-negative
        """
        raise NotImplementedError


class AnalyticProximityRisk(RiskModel):
    """
    Safe region = axis-aligned box [x_lo,x_hi]×[y_lo,y_hi]×[z_lo,z_hi].
    Inside the box: d = distance to nearest face; penalty = relu(margin - d)^2.
    Outside: large constant penalty (soft barrier).
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        z_bounds: Tuple[float, float],
        margin: float = 0.35,
        outside_penalty: float = 1e4,
    ):
        self.x_lo, self.x_hi = x_bounds
        self.y_lo, self.y_hi = y_bounds
        self.z_lo, self.z_hi = z_bounds
        self.margin = margin
        self.outside_penalty = outside_penalty

    def step_cost(self, x: np.ndarray) -> np.ndarray:
        p = x[:, :3]
        # Per-face distances (positive when inside along that face's inward direction)
        d_faces = np.stack(
            [
                p[:, 0] - self.x_lo,
                self.x_hi - p[:, 0],
                p[:, 1] - self.y_lo,
                self.y_hi - p[:, 1],
                p[:, 2] - self.z_lo,
                self.z_hi - p[:, 2],
            ],
            axis=1,
        )
        inside = np.all(d_faces > 0.0, axis=1)
        d_nearest = np.min(d_faces, axis=1)
        barrier = np.maximum(0.0, self.margin - np.maximum(d_nearest, 0.0)) ** 2
        cost = np.where(inside, barrier, self.outside_penalty + np.abs(np.minimum(d_nearest, 0.0)))
        return cost.astype(float)


def mlp_risk_forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """x: (K, in_dim) -> (K,) linear last layer; non-negative enforced in MLPRisk."""
    h = np.maximum(0.0, x @ w1 + b1)
    out = (h @ w2).ravel() + b2.ravel()
    return out


class MLPRisk(RiskModel):
    """
    2-layer MLP: in_dim -> hidden -> 1, ReLU on hidden, softplus-like ReLU on output.
    Expects normalized xyz; normalization params stored in npz.
    """

    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w1 = np.asarray(data["w1"], dtype=float)
        self.b1 = np.asarray(data["b1"], dtype=float).ravel()
        self.w2 = np.asarray(data["w2"], dtype=float)
        self.b2 = np.asarray(data["b2"], dtype=float).ravel()
        self.xyz_mean = np.asarray(data.get("xyz_mean", np.zeros(3)), dtype=float).ravel()
        self.xyz_std = np.asarray(data.get("xyz_std", np.ones(3)), dtype=float).ravel()
        self.xyz_std = np.where(self.xyz_std < 1e-6, 1.0, self.xyz_std)
        self.y_scale = float(np.asarray(data.get("y_scale", 1.0)))

    def step_cost(self, x: np.ndarray) -> np.ndarray:
        p = (x[:, :3] - self.xyz_mean) / self.xyz_std
        z = mlp_risk_forward(p, self.w1, self.b1, self.w2, self.b2)
        return np.maximum(0.0, z * self.y_scale)


def default_analytic_risk_for_walls_scene() -> AnalyticProximityRisk:
    """
    Matches scene_with_walls.xml: box walls centered at x=±2.65, half-thickness 0.08
    (inner free-space boundary ≈ ±2.57 m). Safe box inset for r=1 m circle at z≈1.
    """
    return AnalyticProximityRisk(
        x_bounds=(-2.45, 2.45),
        y_bounds=(-10.0, 10.0),
        z_bounds=(0.08, 3.0),
        margin=0.35,
    )
