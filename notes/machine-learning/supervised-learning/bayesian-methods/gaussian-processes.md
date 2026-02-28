# 高斯过程

高斯过程（Gaussian Process, GP）是一种强大的**贝叶斯非参数**方法，它不限定具体的参数形式，而是直接在函数空间上进行推断。高斯过程能够为预测提供完整的不确定性估计，特别适合小样本学习、主动学习和贝叶斯优化等场景。

## 基本概念

### 什么是高斯过程？

📌 **定义**：高斯过程是随机变量的集合，其中任意有限个随机变量的联合分布都是多元高斯分布。

数学上，一个高斯过程定义为：

$$ f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) $$

其中：
- $m(\mathbf{x})$：**均值函数**，描述函数的平均行为
- $k(\mathbf{x}, \mathbf{x}')$：**协方差函数（核函数）**，描述不同输入点之间的相关性

### 与参数模型的对比

| 特性 | 参数模型 | 高斯过程 |
|------|----------|----------|
| 模型复杂度 | 固定参数数量 | 随数据量增长 |
| 先验假设 | 参数的先验分布 | 函数的先验分布 |
| 预测不确定性 | 需要额外方法估计 | 天然提供 |
| 过拟合风险 | 可能过拟合 | 自动正则化 |
| 适用场景 | 大样本数据 | 小样本、需要不确定性估计 |

### 核心思想

💡 高斯过程将**函数**视为随机变量。给定有限的观测数据，我们关心的是函数在所有可能输入点上的分布。通过核函数定义函数值之间的相关性，可以在观测点附近获得精确预测，在远离观测点处保持不确定性。

## 数学原理

### 多元高斯分布回顾

对于 $n$ 维随机向量 $\mathbf{f} = [f_1, \ldots, f_n]^T$，其多元高斯分布为：

$$ \mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K}) $$

其中：
- $\boldsymbol{\mu} \in \mathbb{R}^n$：均值向量
- $\mathbf{K} \in \mathbb{R}^{n \times n}$：协方差矩阵，$K_{ij} = \text{Cov}(f_i, f_j)$

**多元高斯分布的重要性质**：
- **边缘分布**：任何子集的边缘分布仍为高斯
- **条件分布**：给定部分变量，剩余变量的条件分布仍为高斯

### 高斯过程回归

假设我们有训练数据 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$，其中：

$$ y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2) $$

#### 联合分布

训练目标值 $\mathbf{y}$ 和测试点预测值 $\mathbf{f}_*$ 的联合分布为：

$$ \begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu} \\ \boldsymbol{\mu}_* \end{bmatrix}, \begin{bmatrix} \mathbf{K} + \sigma_n^2\mathbf{I} & \mathbf{K}_* \\ \mathbf{K}_*^T & \mathbf{K}_{**} \end{bmatrix} \right) $$

其中：
- $\mathbf{K} = k(\mathbf{X}, \mathbf{X})$：训练数据间的核矩阵
- $\mathbf{K}_* = k(\mathbf{X}, \mathbf{X}_*)$：训练与测试数据间的核矩阵
- $\mathbf{K}_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$：测试数据间的核矩阵

#### 预测分布

根据多元高斯分布的条件分布公式，预测分布为：

$$ \mathbf{f}_* | \mathbf{X}, \mathbf{y}, \mathbf{X}_* \sim \mathcal{N}(\boldsymbol{\bar{f}}_*, \text{cov}(\mathbf{f}_*)) $$

**预测均值**：

$$ \boldsymbol{\bar{f}}_* = \mathbf{K}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{y} $$

**预测协方差**：

$$ \text{cov}(\mathbf{f}_*) = \mathbf{K}_{**} - \mathbf{K}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{K}_* $$

### 边缘似然

高斯过程的边缘似然（证据）为：

$$ \log p(\mathbf{y}|\mathbf{X}, \theta) = -\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{n}{2}\log 2\pi $$

其中 $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$，$\theta$ 是核函数的超参数。

边缘似然由三部分组成：
1. **数据拟合项**：$-\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y}$，衡量模型与数据的匹配程度
2. **复杂度惩罚项**：$-\frac{1}{2}\log|\mathbf{K}_y|$，惩罚过于复杂的模型
3. **归一化常数**：$-\frac{n}{2}\log 2\pi$

## 核函数选择

核函数（协方差函数）是高斯过程的核心，它决定了函数的平滑性、周期性等性质。

### 常用核函数

#### 平方指数核（RBF/高斯核）

$$ k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2l^2}\right) $$

参数：
- $\sigma_f^2$：信号方差，控制函数的振幅
- $l$：长度尺度，控制函数的平滑程度

特点：无限可微，产生非常平滑的函数

#### 马顿核（Matérn Kernel）

$$ k_{\nu}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}r}{l}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{l}\right) $$

其中 $r = \|\mathbf{x} - \mathbf{x}'\|$，$K_\nu$ 是修正贝塞尔函数。

常用变体：
- $\nu = 1/2$：指数核，产生的函数连续但不可微
- $\nu = 3/2$：一次可微
- $\nu = 5/2$：二次可微
- $\nu \to \infty$：收敛到RBF核

#### 周期核

$$ k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|\mathbf{x} - \mathbf{x}'|/p)}{l^2}\right) $$

参数 $p$ 控制周期，适合具有周期性的数据。

#### 线性核

$$ k(\mathbf{x}, \mathbf{x}') = \sigma_b^2 + \sigma_v^2 (\mathbf{x} - \mathbf{c})(\mathbf{x}' - \mathbf{c}) $$

适合线性趋势的数据。

### 核函数组合

核函数可以通过加法和乘法组合：

$$ \begin{aligned} k_1 + k_2 &: \text{独立过程的叠加} \\ k_1 \times k_2 &: \text{两个过程的乘积} \end{aligned} $$

**常用组合**：
- 线性核 + RBF核：线性趋势 + 平滑波动
- RBF核 × 周期核：局部周期性
- RBF核 + RBF核（不同长度尺度）：多尺度结构

## 代码示例

### 基本高斯过程回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error

# 设置随机种子
np.random.seed(42)

# 生成示例数据
def true_function(x):
    """真实函数"""
    return np.sin(2 * x) + 0.5 * np.cos(3 * x)

# 生成训练数据
n_train = 25
X_train = np.random.uniform(0, 5, n_train).reshape(-1, 1)
y_train = true_function(X_train.flatten()) + 0.1 * np.random.randn(n_train)

# 生成测试数据
X_test = np.linspace(-0.5, 5.5, 200).reshape(-1, 1)
y_true = true_function(X_test.flatten())

print("=" * 50)
print("高斯过程回归基础示例")
print("=" * 50)

# 定义核函数
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# 训练高斯过程
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)
gp.fit(X_train, y_train)

print(f"初始核函数: {kernel}")
print(f"优化后核函数: {gp.kernel_}")
print(f"对数边缘似然: {gp.log_marginal_likelihood_value_:.3f}")

# 预测
y_pred, y_std = gp.predict(X_test, return_std=True)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, c='red', s=50, zorder=5, label='训练数据')
plt.plot(X_test, y_true, 'g--', label='真实函数', alpha=0.8)
plt.plot(X_test, y_pred, 'b-', label='GP预测均值', linewidth=2)
plt.fill_between(X_test.flatten(), 
                 y_pred - 2*y_std, y_pred + 2*y_std,
                 alpha=0.3, color='blue', label='95%置信区间')
plt.xlabel('x')
plt.ylabel('y')
plt.title('高斯过程回归')
plt.legend()
plt.grid(True, alpha=0.3)

# 放大不确定性区域
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, c='red', s=50, zorder=5, label='训练数据')
plt.plot(X_test, y_pred, 'b-', label='GP预测均值', linewidth=2)
plt.fill_between(X_test.flatten(), 
                 y_pred - 2*y_std, y_pred + 2*y_std,
                 alpha=0.3, color='blue', label='95%置信区间')
# 标注训练数据范围
plt.axvline(x=X_train.min(), color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=X_train.max(), color='gray', linestyle=':', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('不确定性区域（训练范围外置信区间变宽）')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 不同核函数比较

```python
print("=" * 50)
print("不同核函数比较")
print("=" * 50)

# 定义不同的核函数
kernels = {
    'RBF': ConstantKernel(1.0) * RBF(length_scale=1.0),
    'Matern 3/2': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
    'Matern 5/2': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
    'RBF + WhiteKernel': ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, kernel) in zip(axes.flat, kernels.items()):
    gp_temp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gp_temp.fit(X_train, y_train)
    
    y_pred_temp, y_std_temp = gp_temp.predict(X_test, return_std=True)
    
    ax.scatter(X_train, y_train, c='red', s=30, zorder=5)
    ax.plot(X_test, y_true, 'g--', alpha=0.8)
    ax.plot(X_test, y_pred_temp, 'b-', linewidth=2)
    ax.fill_between(X_test.flatten(), 
                    y_pred_temp - 2*y_std_temp, 
                    y_pred_temp + 2*y_std_temp, 
                    alpha=0.3, color='blue')
    
    mse = mean_squared_error(y_true, y_pred_temp)
    ax.set_title(f'{name}\nMSE: {mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 后验采样

```python
print("=" * 50)
print("从后验分布采样函数")
print("=" * 50)

plt.figure(figsize=(10, 6))

# 使用优化后的GP进行采样
n_samples = 10
y_samples = gp.sample_y(X_test, n_samples=n_samples, random_state=42)

plt.scatter(X_train, y_train, c='red', s=50, zorder=5, label='训练数据')
plt.plot(X_test, y_true, 'g--', label='真实函数', alpha=0.8, linewidth=2)

for i in range(n_samples):
    alpha = 0.8 if i == 0 else 0.2
    label = '后验样本' if i == 0 else None
    plt.plot(X_test, y_samples[:, i], 'purple', alpha=alpha, linewidth=1, label=label)

plt.plot(X_test, y_pred, 'b-', label='预测均值', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('高斯过程后验采样')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 高斯过程分类

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_classification

print("=" * 50)
print("高斯过程分类")
print("=" * 50)

# 生成分类数据
X_clf, y_clf = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# 训练高斯过程分类器
gpc = GaussianProcessClassifier(kernel=RBF(1.0), random_state=42)
gpc.fit(X_clf, y_clf)

print(f"核函数: {gpc.kernel_}")
print(f"对数边缘似然: {gpc.log_marginal_likelihood_value_:.3f}")

# 创建网格用于可视化
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 预测概率
Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 决策边界
ax1 = axes[0]
contour1 = ax1.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.8, cmap='RdBu')
ax1.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='RdBu', edgecolors='k')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
ax1.set_title('高斯过程分类 - 预测概率')
plt.colorbar(contour1, ax=ax1, label='P(y=1)')

# 预测不确定性（熵）
ax2 = axes[1]
entropy = -Z * np.log(Z + 1e-10) - (1-Z) * np.log(1-Z + 1e-10)
contour2 = ax2.contourf(xx, yy, entropy, alpha=0.8, cmap='hot')
ax2.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='RdBu', edgecolors='k')
ax2.set_xlabel('特征1')
ax2.set_ylabel('特征2')
ax2.set_title('预测不确定性（熵）')
plt.colorbar(contour2, ax=ax2, label='熵')

plt.tight_layout()
plt.show()
```

### 贝叶斯优化应用

```python
from scipy.stats import norm
from scipy.optimize import minimize

print("=" * 50)
print("贝叶斯优化示例")
print("=" * 50)

# 定义目标函数（黑盒函数）
def objective(x):
    """目标函数：我们想找到最小值"""
    return np.sin(3 * x) + x**2 - 0.7 * x + np.sin(x) * np.cos(2 * x)

# 初始观测点
X_obs = np.array([[0.1], [1.0], [2.0], [3.5]])
y_obs = objective(X_obs).flatten()

def expected_improvement(X, gp, y_best, xi=0.01):
    """
    期望改进（Expected Improvement）采集函数
    
    EI(x) = E[max(y_best - f(x), 0)]
    """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    
    with np.errstate(divide='warn'):
        imp = y_best - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
    
    return ei

def upper_confidence_bound(X, gp, beta=2.0):
    """
    上置信界（Upper Confidence Bound）采集函数
    
    UCB(x) = mu(x) + beta * sigma(x)
    """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + beta * sigma

# 贝叶斯优化循环
n_iterations = 8
X_plot = np.linspace(0, 4, 200).reshape(-1, 1)
y_true_plot = objective(X_plot).flatten()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_iterations):
    # 训练高斯过程
    gp_bo = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
        n_restarts_optimizer=3,
        random_state=42
    )
    gp_bo.fit(X_obs, y_obs)
    
    # 预测
    y_pred, y_std = gp_bo.predict(X_plot, return_std=True)
    
    # 计算采集函数
    y_best = np.min(y_obs)
    ei = expected_improvement(X_plot, gp_bo, y_best)
    
    # 选择下一个评估点
    next_x = X_plot[np.argmax(ei)]
    next_y = objective(next_x)
    
    # 可视化
    ax = axes[i]
    
    # 真实函数和预测
    ax.plot(X_plot, y_true_plot, 'g--', label='真实函数', alpha=0.6)
    ax.plot(X_plot, y_pred, 'b-', label='GP预测')
    ax.fill_between(X_plot.flatten(), y_pred - y_std, y_pred + y_std, 
                    alpha=0.2, color='blue')
    
    # 观测点
    ax.scatter(X_obs, y_obs, c='red', s=50, zorder=5, label='观测点')
    
    # 下一个评估点
    ax.axvline(x=next_x[0], color='orange', linestyle='--', label='下一个点')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'迭代 {i+1}')
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)
    
    # 更新观测点
    X_obs = np.vstack([X_obs, next_x.reshape(1, -1)])
    y_obs = np.append(y_obs, next_y)

plt.tight_layout()
plt.show()

# 最终结果
best_idx = np.argmin(y_obs)
print(f"\n最优解: x = {X_obs[best_idx][0]:.4f}, y = {y_obs[best_idx]:.4f}")

# 真实最优
from scipy.optimize import minimize_scalar
result = minimize_scalar(objective, bounds=(0, 4), method='bounded')
print(f"真实最优: x = {result.x:.4f}, y = {result.fun:.4f}")
```

## 计算复杂度与扩展方法

### 计算复杂度

高斯过程的主要计算瓶颈：

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 训练 | $O(n^3)$ | 矩阵求逆/Cholesky分解 |
| 预测均值 | $O(n)$ | 向量-向量乘法 |
| 预测方差 | $O(n^2)$ | 向量-矩阵-向量乘法 |
| 存储 | $O(n^2)$ | 核矩阵 |

对于 $n > 10,000$ 的数据，标准GP变得不实用。

### 稀疏近似方法

**诱导点方法**（Sparse GP）：选择 $m \ll n$ 个诱导点，将复杂度降至 $O(nm^2)$

```python
# 使用 GPy 或 GPflow 实现稀疏高斯过程
# 这里展示概念

print("稀疏高斯过程概念:")
print("- 选择 m 个诱导点 Z = {z_1, ..., z_m}")
print("- 近似完整核矩阵 K ≈ Q = K_{nm} K_{mm}^{-1} K_{mn}")
print("- 复杂度从 O(n³) 降至 O(nm²)")
```

### 其他扩展

- **变分推断**：处理非高斯似然
- **深度高斯过程**：多层非线性变换
- **多任务高斯过程**：同时学习多个相关任务
- **在线学习**：增量更新模型

## 实践建议

### 核函数选择

| 数据特点 | 推荐核函数 |
|----------|------------|
| 平滑函数 | RBF核 |
| 非平滑函数 | Matern核（小ν值） |
| 周期性数据 | 周期核 或 RBF×周期核 |
| 线性趋势 | 线性核 + RBF核 |
| 多尺度结构 | 多个RBF核相加 |

### 超参数优化

1. **最大化边缘似然**：最常用方法
2. **交叉验证**：更鲁棒但计算量大
3. **贝叶斯方法**：对超参数也做贝叶斯推断

### 注意事项

⚠️ **数值稳定性**：
- 核矩阵可能病态，添加噪声项或使用 Cholesky 分解
- 对角加载（jitter）：$\mathbf{K} \to \mathbf{K} + \epsilon\mathbf{I}$

⚠️ **标准化**：
- 输入特征标准化有助于核函数参数的学习
- 输出标准化使超参数初始化更容易

⚠️ **核函数组合**：
- 复杂模式可能需要核函数组合
- 避免过度复杂化核函数

---

[上一节：EM算法](./em-algorithm.md) | [返回贝叶斯方法目录](./index.md)
