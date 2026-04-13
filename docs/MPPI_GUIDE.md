# MPPI 实验与训练指南（mudrone）

本文说明如何配置 **MPPI** 与 `REPORT.md` 对齐、如何运行解析/学习型邻近风险（§4.2.3），以及如何复现实验指标。完整命令以仓库 [`run_mppi.py`](../run_mppi.py) 为准。

## 1. 概念区分

- **基线 MPPI**（采样 + 指数权重 + 名义控制更新）**没有梯度训练**。需要调节的是超参数：样本数 \(K\)、温度 \(\lambda\)、扰动标准差 `noise_sigma`、时间平滑 `smoothing_alpha`、代价矩阵 \(Q,R\) 与外环步长 `dt_ctrl`。
- **学习型风险场**：小型 MLP 在 **离线** 用解析风险（教师）做监督，导出 `risk_mlp_default.npz`；在线仅在 NumPy 前向，与 CasADi/IPOPT 无关。

## 2. 环境

```bash
cd mudrone
conda env create -f environment.yml   # 若尚未创建
conda activate mobile_mujoco
```

依赖：Python 3.10、NumPy、MuJoCo ≥3、CasADi（仅 MPC）、Matplotlib。

## 3. 与 `REPORT.md` §3.5 对齐的参数

| 参数 | 报告示例 | `MPPIController` / `run_mppi.py` 默认 |
|------|-----------|----------------------------------------|
| Horizon \(N\) | 30 步 (0.6 s @ 0.02 s) | `--horizon 30` |
| 外环周期 | 0.02 s (50 Hz) | `--dt-ctrl 0.02` |
| 样本数 \(K\) | 512 | `--n-samples 512` |
| 温度 \(\lambda\) | 0.05 | `--temperature 0.05` |
| 推力/倾角噪声 | 0.02 N / 0.03 rad | 类内 `noise_sigma` 默认 |
| 时间平滑 | 0.05 | `--smoothing-alpha 0.05` |
| 随机种子 | （实验固定） | `--seed 42` |

若需与旧行为接近，可显式传入：`--n-samples 256 --smoothing-alpha 0.5`（不推荐与 §3.5 表混写）。

## 4. 基线 MPPI：命令与产出

```bash
# 最短复现（圆轨迹，报告默认超参）
python run_mppi.py

# 8 字轨迹
python run_mppi.py --trajectory lemniscate --radius 1.5

# 保存误差/控制曲线图
mkdir -p results
python run_mppi.py --save results/mppi_circle.png

# 固定种子复现 RMSE / 求解时间
python run_mppi.py --seed 123 --duration 25 --save results/mppi_seed123.png
```

终端会打印 **RMSE**、**最大误差**、**平均 MPPI 求解时间**。调参提示：

- **\(K\) 增大**：统计更稳、单步更慢。
- **\(\lambda\) 过小**：权重过于尖锐；过大：接近均匀加权。
- **`smoothing_alpha` 增大**：控制序列更顺、可能更钝。

## 5. MPC vs MPPI

```bash
python run_compare.py
python run_compare.py --radius 1.0 --omega 0.5 --duration 30 --seed 42
python run_compare.py --save results/comparison.png
```

MPC 使用 `horizon=25`、MPPI `horizon=30`（与各自脚本默认一致）。对比表中的 **AvgSolve** 为外环单次 `compute_control` 墙钟时间均值。

## 6. 解析邻近风险 + 可选带墙场景

解析风险在轴对齐安全箱内对「距墙面距离」施加障碍型代价，参数与 [`scene_with_walls.xml`](../models/drone/bitcraze_crazyflie_2/scene_with_walls.xml) 几何一致（约 \(x=\pm 2.65\,\mathrm{m}\) 竖直薄墙）。

```bash
# 软约束（代价），默认地板场景
python run_mppi.py --risk analytic --risk-weight 0.35

# 同时在 MuJoCo 中加载物理墙（可能碰撞）
python run_mppi.py --scene with_walls --risk analytic --risk-weight 0.35 --render
```

`--risk-weight` 过大易导致只躲墙、跟踪变差，可从 **0.15–0.5** 扫描。

## 7. 学习型风险 MLP

**训练**（不依赖 MuJoCo，仅 NumPy）：

```bash
python scripts/train_mppi_risk_mlp.py --out drone_mpc/risk_mlp_default.npz --epochs 600
```

产出：`drone_mpc/risk_mlp_default.npz`（权重与 `xyz_mean/std`、`y_scale`）。

**推理**：

```bash
python run_mppi.py --risk learned --risk-weights drone_mpc/risk_mlp_default.npz --risk-weight 0.35
```

若缺省路径无文件，会提示先运行训练脚本。

## 8. 复现检查清单

1. 记录完整命令行（含 `--seed`、`--horizon`、`--n-samples`、`--smoothing-alpha`、`--risk*`）。
2. 固定 `--duration` ≥ 15–20 s 使圆轨迹进入周期稳态再读 RMSE。
3. 保存图到 `results/` 或 `mini-project/report/figures/` 并在报告中引用同一命令。

## 9. 故障排除

| 现象 | 建议 |
|------|------|
| 单步求解 >20 ms | 减小 `K` 或 `horizon` |
| 轨迹发散 | 检查高度初值、适当增大 \(Q_z\) 或减小 `--omega` |
| 加风险后 RMSE 变差 | 减小 `--risk-weight` |
| `learned` 报缺文件 | 先运行 `scripts/train_mppi_risk_mlp.py` |
