#!/usr/bin/env python3
"""
Train a tiny NumPy MLP to mimic AnalyticProximityRisk (teacher) for MPPI cost augmentation.

Usage:
  python scripts/train_mppi_risk_mlp.py --out drone_mpc/risk_mlp_default.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

# Allow running from repo root (import mppi_risk without loading drone_env / mujoco)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "mppi_risk", os.path.join(_ROOT, "drone_mpc", "mppi_risk.py")
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
default_analytic_risk_for_walls_scene = _MOD.default_analytic_risk_for_walls_scene


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def forward(X: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    """X (N,3), returns pred (N,) linear output (non-neg enforced at inference via ReLU)."""
    z1 = X @ w1 + b1
    h = _relu(z1)
    z2 = (h @ w2).ravel() + float(b2.squeeze())
    pred = z2
    return pred, (z1, h, z2)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 64,
    lr: float = 5e-3,
    epochs: int = 800,
    batch: int = 256,
    seed: int = 0,
) -> tuple:
    rng = np.random.default_rng(seed)
    n, in_dim = X.shape
    w1 = rng.standard_normal((in_dim, hidden)) * 0.1
    b1 = np.zeros(hidden)
    w2 = rng.standard_normal((hidden, 1)) * 0.1
    b2 = np.zeros(1)

    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            sel = idx[s : s + batch]
            xb = X[sel]
            yb = y[sel]
            nb = len(sel)
            pred, (z1, h, z2) = forward(xb, w1, b1, w2, b2)
            err = pred - yb
            # mean MSE gradient: dL/dpred = 2*err/nb * relu'(z2)
            # Linear output: d_relu on output removed
            d_z2 = (2.0 / nb) * err
            d_w2 = h.T @ d_z2[:, None]
            db2 = np.sum(d_z2)
            d_h = d_z2[:, None] * w2.T
            d_z1 = d_h * (z1 > 0).astype(float)
            d_w1 = xb.T @ d_z1
            db1 = np.sum(d_z1, axis=0)

            w1 -= lr * d_w1
            b1 -= lr * db1
            w2 -= lr * d_w2
            b2 -= lr * db2

        if ep % 100 == 0:
            pred_all, _ = forward(X, w1, b1, w2, b2)
            print(f"epoch {ep:4d}  mse={np.mean((pred_all - y) ** 2):.6f}")

    return w1, b1, w2, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="drone_mpc/risk_mlp_default.npz")
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    teacher = default_analytic_risk_for_walls_scene()

    # Sample positions in a box around the flight envelope (wider than safe box)
    lo = np.array([-3.5, -4.0, 0.0])
    hi = np.array([3.5, 4.0, 2.5])
    P = rng.uniform(lo, hi, size=(args.n_samples, 3))
    X6 = np.concatenate([P, np.zeros((args.n_samples, 3))], axis=1)
    y = teacher.step_cost(X6)
    # Cap outside-wall penalties so labels stay in a range the small MLP can fit
    y_fit = np.minimum(y, 500.0)
    y_scale = float(np.std(y_fit) + 1e-6)
    yn = y_fit / y_scale

    mean = P.mean(axis=0)
    std = P.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (P - mean) / std

    w1, b1, w2, b2 = train_mlp(Xn, yn, epochs=args.epochs, seed=args.seed)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        xyz_mean=mean,
        xyz_std=std,
        y_scale=y_scale,
    )
    pred, _ = forward(Xn, w1, b1, w2, b2)
    pred_clipped = np.maximum(0.0, pred)
    print(
        f"Saved {out_path}  mse_normalized={np.mean((pred_clipped - yn) ** 2):.6f}  "
        f"mae_phys={np.mean(np.abs(pred_clipped * y_scale - y_fit)):.4f}"
    )


if __name__ == "__main__":
    main()
