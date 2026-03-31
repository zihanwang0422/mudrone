# 无人机 MPC 轨迹跟踪控制系统

> 基于 MuJoCo 仿真环境，使用模型预测控制（MPC）与 CasADi 优化框架，实现 Crazyflie 2 四旋翼无人机的高精度轨迹跟踪。

---

## 目录

- [项目概览](#项目概览)
- [系统架构](#系统架构)
- [无人机动力学模型](#无人机动力学模型)
- [MPC 控制原理与公式](#mpc-控制原理与公式)
- [CasADi 优化框架详解](#casadi-优化框架详解)
- [内环姿态控制](#内环姿态控制)
- [轨迹生成器](#轨迹生成器)
- [MPPI 随机控制对比](#mppi-随机控制对比)
- [快速上手](#快速上手)
- [简历摘要](#简历摘要)

---

## 项目概览

本项目实现了一套完整的**级联控制（Cascade Control）**框架：

```
外环 MPC（50 Hz）──→ [推力, 横滚角, 俯仰角]
                              │
内环 PD 姿态控制（200 Hz）──→ MuJoCo 仿真执行器
```

支持两种轨迹类型：
- **圆形轨迹**（Circle）
- **∞字形轨迹**（Lemniscate / Figure-8）

---

## 系统架构

```
run_mpc.py
├── DroneEnv         # MuJoCo 仿真环境，封装 Crazyflie 2 MJCF 模型
├── MPCController    # 外环 MPC（CasADi + IPOPT 求解器）
├── CascadeController / AttitudePD   # 内环 PD 姿态控制
└── CircleTrajectory / LemniscateTrajectory   # 参考轨迹生成
```

---

## 无人机动力学模型

### 状态与控制量定义

| 量   | 维度 | 内容                    |
|------|------|-------------------------|
| 状态 $\mathbf{x}$ | 6    | $[x,\, y,\, z,\, v_x,\, v_y,\, v_z]$    |
| 控制 $\mathbf{u}$ | 3    | $[T,\, \phi,\, \theta]$（推力 N、横滚角 rad、俯仰角 rad）|

> Yaw（偏航角）由内环 PID 独立保持为 0，不作为 MPC 优化自由度。

### 连续时间动力学

在偏航角 $\psi = 0$ 的假设下，无人机平动加速度为：

$$
\dot{x} = v_x, \quad \dot{y} = v_y, \quad \dot{z} = v_z
$$

$$
a_x = \frac{T}{m} \sin\theta, \quad
a_y = -\frac{T}{m} \sin\phi, \quad
a_z = \frac{T}{m} \cos\phi \cos\theta - g
$$

代码中的动力学函数（CasADi 符号表达）：

```python
# drone_mpc/mpc_controller.py
def _dynamics(self, x, u):
    vx, vy, vz     = x[3], x[4], x[5]
    T, roll, pitch = u[0], u[1], u[2]
    ax =  (T / self.mass) * ca.sin(pitch)
    ay = -(T / self.mass) * ca.sin(roll)
    az =  (T / self.mass) * ca.cos(roll) * ca.cos(pitch) - self.gravity
    return ca.vertcat(vx, vy, vz, ax, ay, az)
```

---

## MPC 控制原理与公式

### 滚动时域优化（Receding Horizon Optimization）

MPC 在每个控制周期 $t_k$ 求解一个有限时域的最优控制问题，仅执行第一步控制量 $\mathbf{u}_0^*$，然后滚动推进：

$$
\min_{\mathbf{X}, \mathbf{U}} \quad J = \underbrace{\sum_{k=0}^{N-1} \left[ (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})^\top Q (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}) + (\mathbf{u}_k - \mathbf{u}_h)^\top R (\mathbf{u}_k - \mathbf{u}_h) + \Delta\mathbf{u}_k^\top S \,\Delta\mathbf{u}_k \right]}_{\text{阶段代价}} + \underbrace{(\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})^\top Q_f (\mathbf{x}_N - \mathbf{x}_N^{\text{ref}})}_{\text{终端代价}}
$$

**subject to：**

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k), \quad k = 0, \ldots, N-1
$$

$$
\mathbf{x}_0 = \mathbf{x}_{\text{current}}, \quad \mathbf{u}_{\min} \le \mathbf{u}_k \le \mathbf{u}_{\max}
$$

### 代价矩阵含义

| 矩阵 | 作用 |
|------|------|
| $Q$ | 状态跟踪精度权重（$z$ 方向权重更高，保证高度精度） |
| $R$ | 控制量消耗代价（$\phi, \theta$ 权重大，限制倾斜角幅度） |
| $Q_f$ | 终端状态权重（$= 5Q$，拉升预测末端精度） |
| $S$ | 控制增量惩罚 $\Delta\mathbf{u}_k = \mathbf{u}_k - \mathbf{u}_{k-1}$，使控制平滑 |

```python
Q  = np.diag([200., 200., 300., 0.5, 0.5, 5.0])   # [x, y, z, vx, vy, vz]
Qf = np.diag([500., 500., 600., 5.,  5.,  20.])
R  = np.diag([0.5, 5.0, 5.0])                       # [T, roll, pitch]
S_delta = np.diag([2.0, 80.0, 80.0])               # 控制平滑
```

### 离散化：四阶 Runge-Kutta

连续动力学 $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ 使用 RK4 积分离散化，保证精度：

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

其中 $k_1 = f(\mathbf{x}_k, \mathbf{u}_k)$，$k_2 = f(\mathbf{x}_k + \frac{\Delta t}{2} k_1, \mathbf{u}_k)$，依此类推。

```python
# drone_mpc/mpc_controller.py
f1 = self._dynamics(xk, uk)
f2 = self._dynamics(xk + self.dt/2*f1, uk)
f3 = self._dynamics(xk + self.dt/2*f2, uk)
f4 = self._dynamics(xk + self.dt*f3,   uk)
xn = xk + (self.dt/6.)*(f1 + 2*f2 + 2*f3 + f4)
```

---

### 圆形轨迹跟踪：从参考生成到控制输出的完整流程

#### 1. 参考轨迹的解析表达

圆形轨迹在 $t$ 时刻的参考状态由解析公式直接给出（无需数值积分）：

$$
\mathbf{x}^{\text{ref}}(t) = \begin{bmatrix}
r\cos(\omega t) \\
r\sin(\omega t) \\
h \\
-r\omega\sin(\omega t) \\
r\omega\cos(\omega t) \\
0
\end{bmatrix}
$$

**关键：速度分量同样被纳入参考**（不只是位置），这使得 MPC 的 $Q$ 矩阵可以同时惩罚位置误差和速度误差，从而在曲线段提前"预判"需要的向心速度。

```python
# drone_mpc/trajectory.py
def get_reference(self, t: float) -> np.ndarray:
    x  = cx + r * np.cos(w * t)
    y  = cy + r * np.sin(w * t)
    vx = -r * w * np.sin(w * t)   # 位置对 t 求导
    vy =  r * w * np.cos(w * t)
    return np.array([x, y, z, vx, vy, vz])
```

#### 2. 预测域内的参考序列展开

每次 MPC 求解前，将未来 $N$ 步（预测域 $N\Delta t = 0.5\,\text{s}$）的参考状态一次性展开：

```python
# run_mpc.py：每个外环周期（20 ms）调用一次
ref_seq = traj.get_reference_sequence(t_sim, N=outer.N, dt=outer.dt)
# → shape (25, 6)，覆盖未来 0.5 s 的圆弧段

outer_cmd = outer.compute_control(state, ref_seq)
```

在 MPC 参数向量 $\mathbf{p}$ 中布局如下：

$$
\mathbf{p} = \underbrace{\mathbf{x}_0}_{6} \;\Big|\; \underbrace{\mathbf{x}_1^{\text{ref}}, \mathbf{x}_2^{\text{ref}}, \ldots, \mathbf{x}_N^{\text{ref}}}_{N \times 6} \;\Big|\; \underbrace{\mathbf{u}_{\text{prev}}}_{3}
$$

这样 IPOPT 求解器在优化时，每一步 $k$ 都有对应的参考点 $\mathbf{x}_k^{\text{ref}}$，而不是只追末端目标点。

> **注意区分**：$\mathbf{x}^{\text{ref}}_{1:N}$ 是喂给求解器的"目标"（参数），$\mathbf{X}_{1:N}$ 是求解器优化出来的"预测状态"（决策变量），两者不相等，差值正是代价函数惩罚的对象。

#### 3. $u^*$ 是决策变量，不是从参考反推的

**很重要的概念**：MPC 的控制量 $\mathbf{u}_0^*$ **不是**由参考轨迹直接公式算出来的，而是 IPOPT 把整条预测轨迹上的所有 $\mathbf{u}_{0:N-1}$ 和 $\mathbf{x}_{0:N}$ 作为决策变量，一起求解的结果。

IPOPT 的决策变量向量结构：

$$
\text{opt\_vars} = \underbrace{[\mathbf{x}_0,\, \mathbf{x}_1,\, \ldots,\, \mathbf{x}_N]}_{\text{预测状态，}(N+1)\times 6} \;\Big\|\; \underbrace{[\mathbf{u}_0,\, \mathbf{u}_1,\, \ldots,\, \mathbf{u}_{N-1}]}_{\text{控制序列，}N\times 3}
$$

这 $(N+1)\times 6 + N\times 3 = 186$ 个数全部是**未知数**，由 IPOPT 一次性求解，约束是动力学等式（RK4 积分链把 $\mathbf{X}$ 和 $\mathbf{U}$ 耦合）：

$$
\mathbf{x}_{k+1} = f_{\text{RK4}}(\mathbf{x}_k,\, \mathbf{u}_k), \quad k = 0,\ldots,N-1
$$

| 量 | 类型 | 每周期是否变化 |
|---|---|---|
| `opt_vars`（$\mathbf{X}, \mathbf{U}$） | **决策变量**，IPOPT 求解输出 | 是 |
| `P`（$\mathbf{x}_0^{\text{cur}},\, \mathbf{x}^{\text{ref}}_{1:N},\, \mathbf{u}_{\text{prev}}$） | **参数**，外部喂入 | 是 |

直觉上，$\mathbf{u}_0^*$ 的选择满足三重权衡：

```
① x_1 = f(x0, u_0) 尽量接近 x_1_ref        （近端精度，Q 矩阵）
② x_1, u_1 能让后续 x_2 ... x_N 也跟得上    （全局最优性，终端 Qf）
③ u_0 与 u_prev 差距不大                     （控制平滑，S 矩阵）
```

在圆形轨迹中，MPC 把未来 $N=25$ 步圆弧参考点喂入，求解器发现"要让整段弧代价最小，现在就需要产生向心加速度"，因此 $\mathbf{u}_0^* = [T^*, \phi^*, \theta^*]$ 中的 $\theta^*$ 会有一个微小俯仰倾角指向圆心——**这是优化的结果，不是事先规定的**。

#### 4. 向心加速度如何转化为倾斜角

圆周运动需要持续的向心加速度，大小为：

$$
a_c = r\omega^2 \quad (\text{指向圆心})
$$

以飞机在 $y$ 轴正方向（即 $\theta = \omega t = 0$ 处）飞行为例，向心加速度方向为 $-x$，需要：

$$
a_x = -r\omega^2 = \frac{T}{m}\sin\theta_{\text{pitch}} \implies \theta_{\text{pitch}} \approx -\frac{mr\omega^2}{T}
$$

以本项目参数（$r=1\,\text{m}$，$\omega=0.5\,\text{rad/s}$，$m=0.027\,\text{kg}$，$T \approx mg$）：

$$
|\theta_{\text{pitch}}| \approx \frac{r\omega^2}{g} = \frac{1 \times 0.25}{9.81} \approx 0.026\,\text{rad} \approx 1.5°
$$

这远小于 MPC 的倾斜角约束 $\pm 10°$，因此 MPC 的约束不会在正常圆形轨迹中激活，优化解有充足的可行域。

#### 5. 滚动时域"向前看"示意

```
圆形轨迹（俯视）：
                    ★ ref_k+4
               ★ ref_k+3
          ★ ref_k+2        ← 预测域 N=25 步，覆盖 0.5s 弧段
     ★ ref_k+1
● x_current ──→ 求解最优 U* ──→ 执行 u_0* ──→ 下一周期重新求解

  ●：当前位置（MPC 的 x0）
  ★：未来各步参考点（从 CircleTrajectory 计算）
  MPC 看到前方弧度约 ω×0.5 = 0.25 rad，提前调整姿态
```

**"向前看"的收益**：在圆弧转弯处，MPC 提前 0.5 s 感知到需要转向，开始缓慢 倾斜机体，而不是等到偏差累积后再纠正，因此跟踪误差远小于纯反馈控制。

#### 6. 热启动（Warm Start）加速求解

圆形轨迹飞行时，前后两个控制周期的最优解非常接近（轨迹变化缓慢）。项目利用这一特性将上一步最优控制序列向前平移一步作为初值：

```python
# 热启动：上一步 U*[1:] 作为本步初值，最后一步重复末尾值
if self._prev_u is not None:
    for k in range(self.N - 1):
        x0[cs + k*nu : cs + (k+1)*nu] = self._prev_u[k + 1]
    x0[cs + (self.N-1)*nu:] = self._prev_u[-1]
```

效果：IPOPT 从接近最优的初值出发，迭代次数从 ~50 次降至 ~5 次，单步求解时间约 **5–15 ms**（50 Hz 外环预算 20 ms 内完成）。

#### 7. 圆形轨迹跟踪的控制约束分析

最大圆速度受 MPC 倾斜角约束限制：

$$
\omega_{\max} = \sqrt{\frac{g \cdot \tan(\phi_{\max})}{r}}
$$

以 $\phi_{\max} = 10°$、$r = 1\,\text{m}$：

$$
\omega_{\max} = \sqrt{\frac{9.81 \times \tan(10°)}{1}} \approx \sqrt{1.73} \approx 1.3\,\text{rad/s}
$$

项目默认 $\omega = 0.5\,\text{rad/s}$，留有约 2.6 倍的裕量，保证 MPC 解始终在约束可行域内。

---

## CasADi 优化框架详解

### 为什么使用 CasADi？

[CasADi](https://web.casadi.org/) 是专为数值优化设计的计算框架，核心优势：

1. **自动微分（Automatic Differentiation）**：自动计算 Jacobian / Hessian，无需手推梯度
2. **符号计算图**：先用 `ca.SX.sym` 搭建符号表达式树，再编译为高效 C 代码
3. **无缝对接求解器**：支持 IPOPT（非线性规划）、qpOASES（二次规划）等工业级求解器

### CasADi 使用流程（以本项目为例）

#### Step 1：定义符号变量

```python
import casadi as ca

nx, nu, N = 6, 3, 20   # 状态维、控制维、预测步数

X = ca.SX.sym("X", nx, N + 1)   # 状态序列 (6, 21)
U = ca.SX.sym("U", nu, N)        # 控制序列 (3, 20)
P = ca.SX.sym("P", nx + N*nx + nu)  # 参数向量：初始状态 + 参考轨迹 + 上一步控制
```

#### Step 2：构建目标函数（符号表达式）

```python
cost = ca.SX(0.0)
Q  = ca.DM(np.diag([200., 200., 300., 0.5, 0.5, 5.0]))
R  = ca.DM(np.diag([0.5, 5.0, 5.0]))
S  = ca.DM(np.diag([2.0, 80.0, 80.0]))
uh = ca.DM([mass * 9.81, 0.0, 0.0])   # 悬停控制量

u_prev = P[nx + N*nx : nx + N*nx + nu]
for k in range(N):
    x_ref = P[nx + k*nx : nx + (k+1)*nx]
    # 状态跟踪代价
    cost += ca.mtimes([(X[:,k] - x_ref).T, Q, X[:,k] - x_ref])
    # 控制代价（相对悬停点）
    cost += ca.mtimes([(U[:,k] - uh).T, R, U[:,k] - uh])
    # 控制平滑代价
    du = U[:,k] - (U[:,k-1] if k > 0 else u_prev)
    cost += ca.mtimes([du.T, S, du])
```

#### Step 3：添加约束（等式 + 不等式）

```python
g, lbg, ubg = [], [], []

# 初始状态约束
g   += [X[:, 0] - P[:nx]]
lbg += [0.] * nx;  ubg += [0.] * nx

# 动力学约束（RK4 积分）
for k in range(N):
    xn = rk4_step(X[:,k], U[:,k])
    g   += [X[:, k+1] - xn]
    lbg += [0.] * nx;  ubg += [0.] * nx
```

#### Step 4：定义 NLP 并创建 IPOPT 求解器

```python
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
G        = ca.vertcat(*g)

nlp = {"x": opt_vars, "f": cost, "g": G, "p": P}
opts = {
    "ipopt.print_level":           0,       # 关闭输出
    "ipopt.max_iter":              200,
    "ipopt.warm_start_init_point": "yes",   # 热启动（用上一步解作初值）
    "ipopt.tol":                   1e-4,
    "ipopt.mu_strategy":           "adaptive",
}
solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
```

#### Step 5：在线求解（每控制周期调用）

```python
# 参数向量：当前状态 + 参考轨迹 + 上一步控制
p = np.concatenate([x_current, ref_sequence.flatten(), u_prev])

sol = solver(x0=x_warm,          # 热启动初值（上一步最优解平移）
             lbx=lbx, ubx=ubx,   # 变量上下界（控制约束）
             lbg=0., ubg=0.,      # 等式约束
             p=p)

opt_x  = np.array(sol["x"]).flatten()
u_opt  = opt_x[(N+1)*nx : (N+1)*nx + nu]  # 取第一步控制量
```

### CasADi 核心 API 速查

| API | 用途 |
|-----|------|
| `ca.SX.sym(name, n, m)` | 创建 $n \times m$ 符号矩阵 |
| `ca.DM(array)` | 将 numpy 数组转为 CasADi 数值矩阵 |
| `ca.mtimes([A, B, C])` | 矩阵连乘 $ABC$ |
| `ca.vertcat(*args)` | 垂直拼接向量/矩阵 |
| `ca.nlpsol(name, solver, nlp, opts)` | 创建 NLP 求解器实例 |
| `ca.sin / ca.cos` | 符号三角函数（支持自动微分） |

---

## 内环姿态控制

外环 MPC 输出世界坐标系下的期望倾斜角，内环 PD 将其转换为机体力矩控制指令。

**坐标系变换（偏航角解耦）：**

$$
\begin{bmatrix} \phi_\text{body} \\ \theta_\text{body} \end{bmatrix}
= \begin{bmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{bmatrix}
\begin{bmatrix} \phi_\text{world} \\ \theta_\text{world} \end{bmatrix}
$$

**PD 控制律：**

$$
\tau_\phi = K_p^{\phi}(\phi_\text{ref} - \phi) - K_d^{\phi} \dot{\phi}, \quad
\tau_\theta = K_p^{\theta}(\theta_\text{ref} - \theta) - K_d^{\theta} \dot{\theta}
$$

内环带宽约 32 rad/s（~5 Hz），保证其响应速度远快于外环 MPC（50 Hz）。

---

## 轨迹生成器

### 圆形轨迹

$$
x(t) = r\cos(\omega t), \quad y(t) = r\sin(\omega t), \quad z(t) = h
$$

$$
v_x(t) = -r\omega\sin(\omega t), \quad v_y(t) = r\omega\cos(\omega t)
$$

```python
traj = CircleTrajectory(radius=1.0, omega=0.5, height=1.0)
ref  = traj.get_reference(t)        # → [x, y, z, vx, vy, vz]
refs = traj.get_reference_sequence(t, N=25, dt=0.02)  # → (N, 6) MPC 预测域参考
```

### ∞字形轨迹（Lemniscate）

基于 Bernoulli 双纽线参数化：

$$
x(\theta) = \frac{a\cos\theta}{1 + \sin^2\theta}, \quad y(\theta) = \frac{a\sin\theta\cos\theta}{1 + \sin^2\theta}, \quad \theta = \omega t
$$

支持软启动（warmup）：在 $t \in [0, t_w]$ 内速度参考从 0 线性爬升，避免初始冲击。

---

## MPPI 随机控制对比

除 MPC 外，项目还实现了 **MPPI（Model Predictive Path Integral）** 控制器作为基准对比：

| 特性 | MPC（IPOPT） | MPPI |
|------|-------------|------|
| 求解方式 | 梯度优化（确定性） | 蒙特卡洛采样（随机性） |
| 计算量 | 高（每步求解 NLP） | 可并行（GPU/向量化） |
| 约束处理 | 硬约束（精确） | 软约束（代价惩罚） |
| 跟踪精度 | 高 | 中等 |
| 样本数 | N/A | K = 512 |

```python
mppi = MPPIController(horizon=30, n_samples=512, temperature=0.05)
u    = mppi.compute_control(state, reference)
```

---

## 快速上手

### 环境安装

```bash
conda env create -f environment.yml
conda activate mudrone
```

### 运行 MPC 轨迹跟踪

```bash
# 圆形轨迹
python run_mpc.py --trajectory circle --radius 1.0 --omega 0.5 --duration 20 --render

# ∞字形轨迹
python run_mpc.py --trajectory lemniscate --radius 1.5 --omega 0.4 --duration 30

# 自定义 MPC 预测域
python run_mpc.py --horizon 30 --dt-ctrl 0.02 --dt-sim 0.005

# 保存可视化图像
python run_mpc.py --save results/mpc_circle.png
```

### 运行 MPPI 控制器

```bash
python run_mppi.py --trajectory lemniscate --render
```

### 对比两种控制器

```bash
python run_compare.py
```
