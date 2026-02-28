# 优化理论

优化理论是机器学习的核心数学基础。从线性回归的最小二乘解，到深度学习的随机梯度下降，再到支持向量机的凸优化，优化方法贯穿机器学习的方方面面。理解优化理论，能让你更好地理解模型是如何学习的，以及如何加速训练过程。

## 优化问题概述

### 基本形式

📌 **优化问题**的一般形式是寻找使目标函数取得极值的参数：

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

其中 $f(\mathbf{x})$ 是目标函数（损失函数），$\mathbf{x}$ 是决策变量（模型参数）。

### 约束优化

📌 **约束优化问题**在无约束优化的基础上增加了约束条件：

$$
\begin{aligned}
\min_{\mathbf{x}} & \quad f(\mathbf{x}) \\
\text{s.t.} & \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \quad \text{（不等式约束）} \\
& \quad h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \quad \text{（等式约束）}
\end{aligned}
$$

### 全局最优与局部最优

📌 **全局最优**：$\mathbf{x}^*$ 是全局最优解，如果对所有可行 $\mathbf{x}$，有 $f(\mathbf{x}^*) \leq f(\mathbf{x})$。

📌 **局部最优**：$\mathbf{x}^*$ 是局部最优解，如果存在邻域 $N$，使得对所有 $\mathbf{x} \in N$，有 $f(\mathbf{x}^*) \leq f(\mathbf{x})$。

⚠️ **重要**：对于非凸优化问题，局部最优不一定是全局最优，这是深度学习优化的核心挑战之一。

## 凸优化基础

### 凸集

📌 **凸集**：集合 $C$ 是凸集，如果对于任意 $\mathbf{x}, \mathbf{y} \in C$ 和 $\theta \in [0,1]$：

$$\theta \mathbf{x} + (1-\theta) \mathbf{y} \in C$$

💡 **直观理解**：集合中任意两点的连线仍然在集合内部。

```python
import numpy as np
import matplotlib.pyplot as plt

# 凸集 vs 非凸集可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 凸集示例：圆
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 100)
ax1.fill(np.cos(theta), np.sin(theta), alpha=0.3, color='blue')
ax1.plot([0.5, -0.5], [0.3, -0.3], 'ro-', linewidth=2, markersize=10)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('凸集：圆\n（任意两点连线在集合内）')
ax1.grid(True, alpha=0.3)

# 非凸集示例：月牙形
ax2 = axes[1]
# 外圆
theta = np.linspace(0, 2*np.pi, 100)
ax2.fill(np.cos(theta), np.sin(theta), alpha=0.3, color='red')
# 内圆（挖空）
ax2.fill(0.5*np.cos(theta)+0.5, 0.5*np.sin(theta), color='white')
ax2.plot([0.8, -0.8], [0.2, -0.2], 'bo-', linewidth=2, markersize=10)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.set_title('非凸集：月牙形\n（存在两点连线不在集合内）')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 凸函数

📌 **凸函数**：函数 $f$ 是凸函数，如果其定义域是凸集，且对于任意 $\mathbf{x}, \mathbf{y}$ 和 $\theta \in [0,1]$：

$$f(\theta \mathbf{x} + (1-\theta) \mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta) f(\mathbf{y})$$

💡 **直观理解**：函数图像上任意两点的连线在函数图像之上。

**判断凸函数的方法**：
1. 定义法（Jensen 不等式）
2. 一阶条件：$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})$
3. 二阶条件：Hessian 矩阵半正定（$\nabla^2 f(\mathbf{x}) \succeq 0$）

```python
# 凸函数 vs 非凸函数可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.linspace(-3, 3, 100)

# 凸函数：f(x) = x^2
ax1 = axes[0]
f_convex = x**2
ax1.plot(x, f_convex, 'b-', linewidth=2, label='f(x) = x²')
# 任意两点连线
ax1.plot([-2, 2], [4, 4], 'r--', linewidth=2, label='弦')
ax1.fill_between(x, f_convex, 4, where=(x >= -2) & (x <= 2), alpha=0.2, color='blue')
ax1.scatter([-2, 2], [4, 4], c='red', s=100, zorder=5)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('凸函数：弦在函数图像之上')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 非凸函数：f(x) = x^4 - 2x^2
ax2 = axes[1]
f_nonconvex = x**4 - 2*x**2
ax2.plot(x, f_nonconvex, 'b-', linewidth=2, label='f(x) = x⁴ - 2x²')
# 两点连线穿过函数图像
ax2.plot([-2, 1.5], [f_nonconvex[0], f_nonconvex[-1]], 'r--', linewidth=2, label='弦')
ax2.scatter([-2, 1.5], [f_nonconvex[0], f_nonconvex[-1]], c='red', s=100, zorder=5)
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('非凸函数：存在多个局部最优')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 凸优化的重要性

📌 **核心优势**：凸优化问题的**局部最优解就是全局最优解**。

这意味着：
- 可以高效求解
- 解有理论保证
- 是许多机器学习算法的理论基础

```python
# 凸优化问题的解的唯一性
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 5))

# 凸函数 3D
ax1 = fig.add_subplot(121, projection='3d')
x1 = np.linspace(-3, 3, 50)
x2 = np.linspace(-3, 3, 50)
X1, X2 = np.meshgrid(x1, x2)
Z_convex = X1**2 + X2**2

ax1.plot_surface(X1, X2, Z_convex, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_zlabel('f(x)')
ax1.set_title('凸函数：唯一全局最优')

# 非凸函数 3D
ax2 = fig.add_subplot(122, projection='3d')
Z_nonconvex = X1**2 + X2**2 - 2*np.exp(-(X1**2 + X2**2)/2)

ax2.plot_surface(X1, X2, Z_nonconvex, cmap='plasma', alpha=0.8)
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.set_zlabel('f(x)')
ax2.set_title('非凸函数：多个局部最优')

plt.tight_layout()
plt.show()
```

## 无约束优化方法

### 梯度下降

📌 **梯度下降**是最基本的优化算法，沿负梯度方向迭代更新：

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha \nabla f(\mathbf{x}^{(k)})$$

其中 $\alpha$ 是学习率（步长）。

💡 **几何意义**：梯度指向函数增长最快的方向，负梯度是最陡下降方向。

```python
def gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=1000, tol=1e-6):
    """
    梯度下降算法
    
    参数:
        f: 目标函数
        grad_f: 梯度函数
        x0: 初始点
        alpha: 学习率
        max_iter: 最大迭代次数
        tol: 收敛阈值
    
    返回:
        x: 最优解
        history: 迭代历史
    """
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - alpha * gradient
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            print(f"在第 {i+1} 次迭代后收敛")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# 示例：最小化二次函数
f = lambda x: x[0]**2 + 2*x[1]**2
grad_f = lambda x: np.array([2*x[0], 4*x[1]])

x0 = np.array([3.0, 3.0])
x_opt, history = gradient_descent(f, grad_f, x0, alpha=0.1)

print(f"最优解: x* = {x_opt}")
print(f"最优值: f(x*) = {f(x_opt)}")

# 可视化优化路径
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + 2*X2**2

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.7)
plt.colorbar(label='f(x)')
plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=1.5, markersize=4, label='优化路径')
plt.scatter(x0[0], x0[1], c='green', s=100, marker='o', label='起点')
plt.scatter(x_opt[0], x_opt[1], c='red', s=100, marker='*', label='终点')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('梯度下降优化路径')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

### 学习率的影响

⚠️ 学习率是梯度下降中最重要的超参数：

- **学习率过大**：可能震荡或发散
- **学习率过小**：收敛速度太慢
- **理想学习率**：快速且稳定地收敛

```python
# 不同学习率的影响
f = lambda x: x**2
grad_f = lambda x: 2*x

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (alpha, title) in enumerate([
    (1.1, '学习率过大：震荡'),
    (0.01, '学习率过小：收敛慢'),
    (0.5, '合适的学习率')
]):
    ax = axes[idx]
    x0 = 2.0
    x = x0
    history = [x]
    
    for _ in range(20):
        x = x - alpha * grad_f(x)
        history.append(x)
    
    # 绘制函数
    x_range = np.linspace(-2.5, 2.5, 100)
    ax.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x)')
    
    # 绘制迭代点
    for i in range(len(history) - 1):
        ax.annotate('', xy=(history[i+1], f(history[i+1])), 
                   xytext=(history[i], f(history[i])),
                   arrowprops=dict(arrowstyle='->', color='red'))
    ax.scatter(history, [f(x) for x in history], c='red', s=30, zorder=5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{title} (α={alpha})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 5)

plt.tight_layout()
plt.show()
```

### 动量法

📌 **动量法**通过累积历史梯度来加速收敛，减少震荡：

$$
\begin{aligned}
\mathbf{v}^{(k+1)} &= \beta \mathbf{v}^{(k)} + (1-\beta) \nabla f(\mathbf{x}^{(k)}) \\
\mathbf{x}^{(k+1)} &= \mathbf{x}^{(k)} - \alpha \mathbf{v}^{(k+1)}
\end{aligned}
$$

其中 $\beta$ 是动量系数（通常取 0.9）。

```python
def gradient_descent_with_momentum(f, grad_f, x0, alpha=0.1, beta=0.9, 
                                    max_iter=100, tol=1e-6):
    """带动量的梯度下降"""
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        v = beta * v + (1 - beta) * gradient
        x_new = x - alpha * v
        
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# 对比：普通梯度下降 vs 动量法
# 使用病态条件函数（Rosenbrock 函数的简化版）
f = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
grad_f = lambda x: np.array([
    -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
    200*(x[1] - x[0]**2)
])

x0 = np.array([-1.0, 1.0])

# 普通 GD
x_gd, history_gd = gradient_descent(f, grad_f, x0, alpha=0.001, max_iter=500)

# 动量法
x_momentum, history_momentum = gradient_descent_with_momentum(
    f, grad_f, x0, alpha=0.001, beta=0.9, max_iter=500
)

# 可视化
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 100*(X2 - X1**2)**2 + (1 - X1)**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, history, title in [
    (axes[0], history_gd, '普通梯度下降'),
    (axes[1], history_momentum, '动量法')
]:
    ax.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.7)
    ax.plot(history[:, 0], history[:, 1], 'r.-', linewidth=1, markersize=3)
    ax.scatter(x0[0], x0[1], c='green', s=100, marker='o', label='起点')
    ax.scatter(1, 1, c='red', s=100, marker='*', label='最优解')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'{title}\n迭代次数: {len(history)}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Adam 优化器

📌 **Adam**（Adaptive Moment Estimation）结合了动量法和自适应学习率：

$$
\begin{aligned}
\mathbf{m}^{(k+1)} &= \beta_1 \mathbf{m}^{(k)} + (1-\beta_1) \nabla f(\mathbf{x}^{(k)}) \\
\mathbf{v}^{(k+1)} &= \beta_2 \mathbf{v}^{(k)} + (1-\beta_2) (\nabla f(\mathbf{x}^{(k)}))^2 \\
\hat{\mathbf{m}} &= \frac{\mathbf{m}^{(k+1)}}{1-\beta_1^{k+1}} \quad \text{（偏差修正）} \\
\hat{\mathbf{v}} &= \frac{\mathbf{v}^{(k+1)}}{1-\beta_2^{k+1}} \\
\mathbf{x}^{(k+1)} &= \mathbf{x}^{(k)} - \alpha \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}
\end{aligned}
$$

默认参数：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$

```python
def adam(f, grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, 
         eps=1e-8, max_iter=500, tol=1e-6):
    """Adam 优化器"""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]
    
    for t in range(1, max_iter + 1):
        g = grad_f(x)
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x_new = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# 测试 Adam
x_adam, history_adam = adam(f, grad_f, x0, alpha=0.1)

print(f"Adam 最优解: x* = {x_adam}")
print(f"Adam 迭代次数: {len(history_adam)}")

# 可视化 Adam 路径
plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.7)
plt.plot(history_adam[:, 0], history_adam[:, 1], 'r.-', linewidth=1.5, markersize=3)
plt.scatter(x0[0], x0[1], c='green', s=100, marker='o', label='起点')
plt.scatter(1, 1, c='red', s=100, marker='*', label='最优解')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title(f'Adam 优化路径 (迭代次数: {len(history_adam)})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 约束优化

### 拉格朗日乘子法

📌 **拉格朗日乘子法**是求解等式约束优化问题的经典方法。

对于问题：
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad h(\mathbf{x}) = 0$$

构造拉格朗日函数：
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$$

最优解满足：
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) + \lambda \nabla h(\mathbf{x}) = 0$$

```python
# 拉格朗日乘子法示例
# 问题：min x² + y²  s.t. x + y = 1

# 解析解
# L = x² + y² + λ(x + y - 1)
# ∂L/∂x = 2x + λ = 0  →  x = -λ/2
# ∂L/∂y = 2y + λ = 0  →  y = -λ/2
# 约束: x + y = 1  →  -λ = 1  →  λ = -1
# 最优解: x = 0.5, y = 0.5

print("拉格朗日乘子法示例")
print("问题: min x² + y²  s.t. x + y = 1")
print("=" * 40)

# 解析解
x_opt, y_opt = 0.5, 0.5
print(f"解析最优解: x* = {x_opt}, y* = {y_opt}")
print(f"最优值: f(x*) = {x_opt**2 + y_opt**2}")

# 可视化
fig, ax = plt.subplots(figsize=(8, 8))

# 目标函数等高线
x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='f(x,y)')

# 约束条件
ax.plot([0, 1], [1, 0], 'r-', linewidth=3, label='约束: x + y = 1')

# 最优解
ax.scatter(x_opt, y_opt, c='red', s=200, marker='*', zorder=5, label='最优解')

# 从最优解出发的梯度
grad_f = np.array([2*x_opt, 2*y_opt])  # ∇f
grad_h = np.array([1, 1])              # ∇h
ax.arrow(x_opt, y_opt, -grad_f[0]*0.2, -grad_f[1]*0.2, 
        head_width=0.05, fc='blue', ec='blue', label='∇f')
ax.arrow(x_opt, y_opt, grad_h[0]*0.2, grad_h[1]*0.2, 
        head_width=0.05, fc='green', ec='green', label='∇h')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('拉格朗日乘子法\n最优解处 ∇f 与 ∇h 平行')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.show()
```

### KKT 条件

📌 **KKT 条件**是拉格朗日乘子法的推广，适用于同时有等式和不等式约束的问题。

对于问题：
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \ h_j(\mathbf{x}) = 0
$$

KKT 条件包括：

| 条件 | 公式 | 含义 |
|------|------|------|
| 原始可行性 | $g_i(\mathbf{x}^*) \leq 0, h_j(\mathbf{x}^*) = 0$ | 解满足约束 |
| 对偶可行性 | $\mu_i \geq 0$ | 乘子非负 |
| 互补松弛 | $\mu_i g_i(\mathbf{x}^*) = 0$ | 不等式约束要么激活，要么乘子为 0 |
| 梯度条件 | $\nabla f + \sum \mu_i \nabla g_i + \sum \lambda_j \nabla h_j = 0$ | 梯度平衡 |

## 机器学习中的应用

### 线性回归

📌 **线性回归**的优化问题：

$$\min_{\mathbf{w}} \frac{1}{2} \|X\mathbf{w} - \mathbf{y}\|^2$$

有**闭式解**：$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$

```python
# 线性回归：闭式解 vs 梯度下降
np.random.seed(42)

# 生成数据
n_samples, n_features = 100, 3
X = np.random.randn(n_samples, n_features)
true_w = np.array([1.5, -2.0, 1.0])
y = X @ true_w + np.random.randn(n_samples) * 0.5

print("=" * 50)
print("线性回归求解")
print("=" * 50)

# 闭式解
w_closed = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"闭式解: {w_closed}")

# 梯度下降
def linear_regression_gd(X, y, alpha=0.01, max_iter=1000):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    
    for _ in range(max_iter):
        gradient = X.T @ (X @ w - y)
        w = w - alpha * gradient
    
    return w

w_gd = linear_regression_gd(X, y)
print(f"梯度下降解: {w_gd}")
print(f"真实参数: {true_w}")
print(f"\n闭式解误差: {np.linalg.norm(w_closed - true_w):.4f}")
print(f"梯度下降误差: {np.linalg.norm(w_gd - true_w):.4f}")
```

### 岭回归（L2 正则化）

📌 **岭回归**添加 L2 正则化项：

$$\min_{\mathbf{w}} \frac{1}{2} \|X\mathbf{w} - \mathbf{y}\|^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

闭式解：$\mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$

```python
# 岭回归：正则化的作用
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

# 生成病态数据
np.random.seed(42)
n_samples, n_features = 50, 100  # 特征数 > 样本数
X = np.random.randn(n_samples, n_features)
true_w = np.zeros(n_features)
true_w[:5] = [1, 2, -1, 0.5, -0.5]  # 只有前 5 个特征有用
y = X @ true_w + np.random.randn(n_samples) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("=" * 50)
print("岭回归 vs 普通线性回归")
print("=" * 50)

# 普通线性回归（可能过拟合）
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"普通线性回归训练 R²: {lr.score(X_train, y_train):.4f}")
print(f"普通线性回归测试 R²: {lr.score(X_test, y_test):.4f}")

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归训练 R²: {ridge.score(X_train, y_train):.4f}")
print(f"岭回归测试 R²: {ridge.score(X_test, y_test):.4f}")

# 可视化正则化效果
alphas = np.logspace(-3, 3, 50)
train_scores = []
test_scores = []
coef_norms = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    coef_norms.append(np.linalg.norm(ridge.coef_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# R² 曲线
axes[0].semilogx(alphas, train_scores, 'b-', label='训练集')
axes[0].semilogx(alphas, test_scores, 'r-', label='测试集')
axes[0].set_xlabel('正则化系数 α')
axes[0].set_ylabel('R²')
axes[0].set_title('岭回归：正则化强度对性能的影响')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 系数范数
axes[1].semilogx(alphas, coef_norms, 'g-')
axes[1].set_xlabel('正则化系数 α')
axes[1].set_ylabel('||w||₂')
axes[1].set_title('岭回归：正则化对系数的影响')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 正则化系数太大 → 欠拟合；太小 → 过拟合")
```

### 支持向量机（SVM）

📌 **支持向量机**的优化问题是一个典型的凸优化问题：

原始问题：
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)
y = 2 * y - 1  # 转换为 -1, 1

print("=" * 50)
print("支持向量机")
print("=" * 50)

# 训练 SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# 获取支持向量
support_vectors = X[svm.support_]
print(f"支持向量数量: {len(support_vectors)}")

# 可视化
plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='类别 +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='s', label='类别 -1')

# 标记支持向量
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           s=200, facecolors='none', edgecolors='green', linewidth=2, 
           label='支持向量')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间隔
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
          linestyles=['--', '-', '--'])

plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('支持向量机：最大间隔分类')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 支持向量是离决策边界最近的点，它们决定了分类超平面")
```

## 优化算法总结

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 批量梯度下降 | 收敛稳定 | 大数据集慢 | 小规模数据 |
| 随机梯度下降（SGD） | 速度快、可在线学习 | 噪声大、需调学习率 | 大规模数据 |
| 动量法 | 加速收敛、减少震荡 | 需要调参 | 病态问题 |
| Adam | 自适应学习率、收敛快 | 可能错过局部最优 | 深度学习首选 |

## 常见问题与注意事项

1. **学习率调度**：训练过程中调整学习率（如学习率衰减、余弦退火）可以提升性能
2. **批量大小**：大批量更稳定但内存占用大；小批量有正则化效果但噪声大
3. **初始化**：好的初始化（如 Xavier、He 初始化）可以加速收敛
4. **早停**：在验证误差开始上升时停止训练，防止过拟合
5. **梯度裁剪**：防止梯度爆炸，特别是在 RNN 训练中

## 参考资料

- Stephen Boyd & Lieven Vandenberghe, *Convex Optimization*
- Sebastian Ruder, *An overview of gradient descent optimization algorithms*
- Andrew Ng, *Machine Learning* 课程中的优化部分
