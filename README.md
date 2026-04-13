# 🚁 mudrone

Quadrotor (Crazyflie 2) trajectory tracking in MuJoCo physics simulation, implementing two Model Predictive Control approaches: **MPC** (CasADi nonlinear optimization) and **MPPI** (sampling-based Path Integral control).

| MPC — Circle | MPC — Figure-8 |
|:---:|:---:|
| ![MPC Circle](media/mpc_circle.gif) | ![MPC Figure-8](media/mpc_eight.gif) |

| MPPI — Circle | MPPI — Figure-8 |
|:---:|:---:|
| ![MPPI Circle](media/mppi_circle.gif) | ![MPPI Figure-8](media/mppi_eight.gif) |

<!-- ---

## ✨ Features

- 🎯 **Dual MPC backends** — gradient-based IPOPT solver (MPC) and gradient-free sampling (MPPI) -->


---

## 📁 Project Structure

```
mobile_mujoco/
├── models/
│   └── drone/
│       └── bitcraze_crazyflie_2/
│           ├── cf2.xml          # Crazyflie 2 MJCF model (with spinning props)
│           ├── scene.xml        # Simulation scene (ground + lighting)
│           └── assets/          # 3D mesh files
├── drone_mpc/
│   ├── drone_env.py             # MuJoCo environment wrapper
│   ├── mpc_controller.py        # MPC controller (CasADi + IPOPT)
│   ├── mppi_controller.py       # MPPI controller (sampling-based)
│   ├── mppi_risk.py             # Analytic / learned proximity risk (§4.2.3)
│   ├── inner_loop.py            # Attitude PD inner loop (200 Hz)
│   ├── trajectory.py            # Trajectory generators
│   └── visualization.py        # Plotting utilities
├── docs/
│   └── MPPI_GUIDE.md            # MPPI 参数、风险场与复现实验
├── run_mpc.py                   # Run MPC tracking
├── run_mppi.py                  # Run MPPI tracking
├── run_compare.py               # MPC vs MPPI comparison
├── scripts/
│   └── train_mppi_risk_mlp.py   # Train learned risk MLP (NumPy)
└── environment.yml              # Conda environment
```

---

## 🔧 Installation

**1. Create Conda environment**
```bash
conda create -n mobile_mujoco python=3.10 -y
conda activate mobile_mujoco
```

**2. Install dependencies**
```bash
conda env create -f environment.yml
conda activate mobile_mujoco
```

---

## 🚀 Usage

**MPPI 参数、风险场与复现实验：** 见 [docs/MPPI_GUIDE.md](docs/MPPI_GUIDE.md)。

### ▶️ MPC Trajectory Tracking

```bash
# Basic run (no viewer, generates result plots)
python run_mpc.py

# With real-time MuJoCo viewer
python run_mpc.py --render

# Custom parameters
python run_mpc.py --radius 1.5 --height 1.2 --omega 0.3 --duration 30 --render

# Figure-8 trajectory
python run_mpc.py --trajectory lemniscate --radius 2.0

# Save result plot
python run_mpc.py --save results/mpc_circle.png
```

### ▶️ MPPI Trajectory Tracking

```bash
# Basic run
python run_mppi.py

# With viewer
python run_mppi.py --render

# Tune MPPI parameters
python run_mppi.py --n-samples 512 --temperature 0.02 --horizon 40

# Save result
python run_mppi.py --save results/mppi_circle.png
```

### ⚖️ MPC vs MPPI Comparison

```bash
python run_compare.py
python run_compare.py --radius 1.0 --omega 0.5 --duration 30
python run_compare.py --save results/comparison.png
```

### 🎛️ CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--radius` | `1.0` | Trajectory radius (m) |
| `--height` | `1.0` | Flight altitude (m) |
| `--omega` | `0.5` | Angular speed (rad/s) |
| `--duration` | `20.0` | Simulation duration (s) |
| `--trajectory` | `circle` | `circle` \| `lemniscate` |
| `--render` | `False` | Enable MuJoCo real-time viewer |
| `--save` | `None` | Save plot to file path |
| `--horizon` | `25/30` | MPC/MPPI prediction horizon steps |
| `--n-samples` | `512` | MPPI sample count K (REPORT §3.5) |
| `--temperature` | `0.05` | MPPI temperature λ (lower = greedier) |
| `--smoothing-alpha` | `0.05` | MPPI nominal-control smoothing (REPORT §3.5) |
| `--seed` | `42` | MPPI RNG seed |
| `--risk` | `none` | `none` \| `analytic` \| `learned` (§4.2.3) |
| `--scene` | `default` | `default` \| `with_walls` (match analytic walls) |

---


## 📝 TODO

- [ ] Add warmup and smoothing fixes for MPC/MPPI startup.
- [ ] Improve MPPI performance by optionally integrating JAX (vectorized sampling / GPU).
- [ ] Add f1tenth env.


