# Mobile MuJoCo — 无人机 MPC / MPPI 轨迹跟踪

基于 MuJoCo 物理仿真引擎的无人机（Crazyflie 2）轨迹跟踪控制项目，实现了 **MPC**（基于 CasADi 非线性优化）和 **MPPI**（采样 Path Integral）两种模型预测控制方法。

## 📦 项目结构

```
mobile_mujoco/
├── models/
│   └── drone/
│       └── bitcraze_crazyflie_2/    # Crazyflie 2 MJCF 无人机模型
│           ├── cf2.xml              # 无人机主体模型
│           ├── scene.xml            # 仿真场景（含地面、光照）
│           └── assets/              # 3D 网格文件
├── drone_mpc/
│   ├── __init__.py                  # 模块入口
│   ├── drone_env.py                 # MuJoCo 仿真环境封装
│   ├── mpc_controller.py            # MPC 控制器（CasADi + IPOPT）
│   ├── mppi_controller.py           # MPPI 控制器（采样优化）
│   ├── trajectory.py                # 轨迹生成器（圆形/八字形/螺旋）
│   └── visualization.py            # 可视化工具
├── run_mpc.py                       # 运行 MPC 轨迹跟踪
├── run_mppi.py                      # 运行 MPPI 轨迹跟踪
├── run_compare.py                   # MPC vs MPPI 对比实验
├── environment.yml                  # Conda 环境配置
├── mujoco_mpc/                      # Google DeepMind MuJoCo MPC 参考代码
└── README.md
```

## 🔧 环境安装

### 1. 创建 Conda 环境

```bash
conda create -n mobile_mujoco python=3.10 -y
conda activate mobile_mujoco
```

### 2. 安装依赖

```bash
pip install mujoco numpy scipy matplotlib casadi
```

或者通过 `environment.yml` 一键安装：

```bash
conda env create -f environment.yml
conda activate mobile_mujoco
```

### 3. 验证安装

```bash
python -c "import mujoco; import casadi; print('MuJoCo version:', mujoco.__version__); print('All dependencies OK')"
```

## 🚀 快速开始

### 运行 MPC 圆形轨迹跟踪

```bash
cd mobile_mujoco
conda activate mobile_mujoco

# 基础运行（无可视化窗口，生成结果图）
python run_mpc.py

# 带 MuJoCo 实时可视化
python run_mpc.py --render

# 自定义参数
python run_mpc.py --radius 1.5 --height 1.2 --omega 0.3 --duration 30 --render

# 八字形轨迹
python run_mpc.py --trajectory lemniscate --radius 2.0

# 保存结果图片
python run_mpc.py --save results/mpc_circle.png
```

### 运行 MPPI 圆形轨迹跟踪

```bash
# 基础运行
python run_mppi.py

# 带可视化
python run_mppi.py --render

# 调整 MPPI 参数
python run_mppi.py --n-samples 512 --temperature 0.02 --horizon 40

# 保存结果
python run_mppi.py --save results/mppi_circle.png
```

### MPC vs MPPI 对比实验

```bash
# 运行对比
python run_compare.py

# 自定义参数对比
python run_compare.py --radius 1.0 --omega 0.5 --duration 30

# 保存对比图
python run_compare.py --save results/comparison.png
```

## 🎮 命令行参数

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--radius` | 1.0 | 轨迹半径 (m) |
| `--height` | 1.0 | 飞行高度 (m) |
| `--omega` | 0.5 | 角速度 (rad/s) |
| `--duration` | 20.0 | 仿真时长 (s) |
| `--trajectory` | circle | 轨迹类型: `circle` / `lemniscate` |
| `--render` | False | 开启 MuJoCo 实时可视化 |
| `--save` | None | 保存结果图片路径 |
| `--dt-ctrl` | 0.02 | 控制周期 (s), 即 50 Hz |
| `--dt-sim` | 0.005 | 仿真步长 (s), 即 200 Hz |

### MPC 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | 25 | 预测时域步数 |

### MPPI 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | 30 | 预测时域步数 |
| `--n-samples` | 256 | 采样轨迹数量 |
| `--temperature` | 0.05 | 温度参数 λ（越小越贪心） |

## 📐 算法原理

### MPC (Model Predictive Control)

基于 CasADi 的非线性 MPC，使用 IPOPT 求解器：

$$\min_{u_0, \dots, u_{N-1}} \sum_{k=0}^{N-1} \left[ \|x_k - x_k^{ref}\|_Q^2 + \|u_k - u_{hover}\|_R^2 \right] + \|x_N - x_N^{ref}\|_{Q_f}^2$$

$$\text{s.t.} \quad x_{k+1} = f(x_k, u_k), \quad u_{min} \le u_k \le u_{max}$$

- **状态向量** (9D): $[x, y, z, v_x, v_y, v_z, \phi, \theta, \psi]$（位置 + 速度 + 姿态角）
- **控制向量** (4D): $[T, \tau_\phi, \tau_\theta, \tau_\psi]$（推力 + 力矩指令）
- **动力学**: 简化四旋翼模型，RK4 积分
- **求解器**: IPOPT（内点法），支持 warm start

### MPPI (Model Predictive Path Integral)

基于采样的路径积分控制：

$$u_k^* = u_k^{nom} + \sum_{i=1}^{K} w_i \cdot \delta u_k^{(i)}$$

$$w_i = \frac{\exp(-S_i / \lambda)}{\sum_j \exp(-S_j / \lambda)}$$

- **采样**: $K$ 条控制扰动序列 $\delta u \sim \mathcal{N}(0, \Sigma)$
- **前向仿真**: 对每条采样轨迹进行 RK4 积分
- **加权平均**: 基于指数代价权重更新标称控制序列
- **优点**: 无需梯度计算，天然处理非凸代价，易并行化

### 无人机动力学模型

使用 Crazyflie 2 微型四旋翼 MJCF 模型：
- **质量**: 27g
- **执行器**: 推力 $[0, 0.35]$ N + 三轴力矩 $[-1, 1]$
- **传感器**: IMU（陀螺仪 + 加速度计 + 姿态四元数）
- **悬停推力**: $T_{hover} = mg \approx 0.265$ N

## 📊 输出结果

每次运行会输出：
1. **3D 轨迹图**: 参考 vs 实际轨迹三维可视化
2. **位置跟踪图**: X/Y/Z 各轴随时间变化
3. **跟踪误差图**: 位置误差范数随时间变化 + RMSE
4. **控制输入图**: 推力和力矩指令随时间变化

对比模式还会生成并排比较图。

## 🏗️ 代码模块说明

### `drone_mpc/drone_env.py` — 仿真环境

- `DroneEnv`: MuJoCo 仿真环境封装
  - `reset(pos)`: 重置到指定位置
  - `step(ctrl)`: 施加控制、步进仿真
  - `get_state()`: 获取 13D 状态 (pos + quat + vel + angvel)
  - `launch_viewer()`: 启动 MuJoCo 交互可视化

### `drone_mpc/mpc_controller.py` — MPC 控制器

- `MPCController`: CasADi 非线性 MPC
  - `compute_control(state, reference)`: 求解最优控制
  - 支持 warm start（利用上一步解初始化）
  - 9D 简化状态 + RK4 动力学

### `drone_mpc/mppi_controller.py` — MPPI 控制器

- `MPPIController`: 采样路径积分控制
  - `compute_control(state, reference)`: MPPI 采样优化
  - 批量向量化前向仿真
  - 自适应控制序列平滑

### `drone_mpc/trajectory.py` — 轨迹生成

- `CircleTrajectory`: 水平圆形轨迹
- `LemniscateTrajectory`: 八字形轨迹
- `HelixTrajectory`: 螺旋上升轨迹

## 🔗 参考资料

- [MuJoCo](https://mujoco.org/) — 物理仿真引擎
- [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) — Google DeepMind MPC 框架
- [Crazyflie 2](https://github.com/google-deepmind/mujoco_menagerie) — 无人机 MJCF 模型
- [CasADi](https://web.casadi.org/) — 非线性优化框架
- Williams et al., *"Information Theoretic MPC for Model-Based Reinforcement Learning"*, ICRA 2017 — MPPI 原始论文

## 📝 License

MIT License
