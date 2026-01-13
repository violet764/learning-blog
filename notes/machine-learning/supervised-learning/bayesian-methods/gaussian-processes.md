# 高斯过程

## 1. 算法概述

高斯过程是贝叶斯非参数方法，为函数空间上的分布提供了一种灵活的建模框架。它能够对函数进行不确定性量化，特别适合小样本数据和需要不确定性估计的场景。

### 1.1 基本概念

**定义**：高斯过程是随机变量的集合，其中任意有限个随机变量的联合分布都是高斯分布。

**数学表示**：
$$ f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) $$

其中：
- $m(\mathbf{x})$：均值函数
- $k(\mathbf{x}, \mathbf{x}')$：协方差函数（核函数）

### 1.2 核心思想

高斯过程将函数视为随机变量，通过核函数定义函数之间的相似性，从而实现对函数的贝叶斯推断。

## 2. 数学原理

### 2.1 多元高斯分布

设$\mathbf{f} = [f(\mathbf{x}_1), \dots, f(\mathbf{x}_n)]^T$服从多元高斯分布：
$$ \mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K}) $$

其中：
- $\boldsymbol{\mu} = [m(\mathbf{x}_1), \dots, m(\mathbf{x}_n)]^T$
- $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$

### 2.2 预测分布

给定训练数据$\mathbf{X}, \mathbf{y}$，新点$\mathbf{x}_*$的预测分布为：
$$ f_* | \mathbf{X}, \mathbf{y}, \mathbf{x}_* \sim \mathcal{N}(\bar{f}_*, \mathbb{V}[f_*]) $$

其中：
$$ \bar{f}_* = \mathbf{k}_*^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{y} $$
$$ \mathbb{V}[f_*] = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_* $$

### 2.3 边缘似然

高斯过程的边缘似然（证据）为：
$$ \log p(\mathbf{y}|\mathbf{X}) = -\frac{1}{2}\mathbf{y}^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K} + \sigma_n^2\mathbf{I}| - \frac{n}{2}\log 2\pi $$

## 3. 核函数选择

### 3.1 常用核函数

**平方指数核（RBF核）**：
$$ k_{SE}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2l^2}\right) $$

**马顿核**：
$$ k_{Matern}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|\mathbf{x} - \mathbf{x}'\|}{l}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|\mathbf{x} - \mathbf{x}'\|}{l}\right) $$

**周期核**：
$$ k_{Per}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi\|\mathbf{x} - \mathbf{x}'\|/p)}{l^2}\right) $$

### 3.2 核函数组合

核函数可以线性组合或相乘来构建更复杂的协方差结构：
- $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
- $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') \times k_2(\mathbf{x}, \mathbf{x}')$

## 4. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# 1. 基本高斯过程回归
print("=== 基本高斯过程回归 ===")

# 生成示例数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(100)

# 添加噪声
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义核函数
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# 训练高斯过程
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(X_train, y_train)

print("优化后的核函数:", gp.kernel_)

# 预测
y_pred, y_std = gp.predict(X, return_std=True)

# 可视化结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, c='red', label='训练数据')
plt.plot(X, np.sin(X), 'g-', label='真实函数', alpha=0.7)
plt.plot(X, y_pred, 'b-', label='GP预测')
plt.fill_between(X.flatten(), y_pred - 2*y_std, y_pred + 2*y_std, 
                alpha=0.3, color='blue', label='95%置信区间')
plt.xlabel('x')
plt.ylabel('y')
plt.title('高斯过程回归')
plt.legend()
plt.grid(True)

# 2. 不同核函数比较
print("\n=== 不同核函数比较 ===")

kernels = {
    'RBF核': RBF(length_scale=1.0),
    'Matern 3/2': Matern(length_scale=1.0, nu=1.5),
    'Matern 5/2': Matern(length_scale=1.0, nu=2.5)
}

for i, (name, kernel) in enumerate(kernels.items(), 2):
    gp_temp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp_temp.fit(X_train, y_train)
    
    y_pred_temp, y_std_temp = gp_temp.predict(X, return_std=True)
    
    plt.subplot(2, 2, i)
    plt.scatter(X_train, y_train, c='red', alpha=0.5)
    plt.plot(X, y_pred_temp, 'b-', label=f'{name}预测')
    plt.fill_between(X.flatten(), y_pred_temp - 2*y_std_temp, 
                    y_pred_temp + 2*y_std_temp, alpha=0.3, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{name}高斯过程')
    plt.grid(True)

plt.tight_layout()
plt.show()

# 3. 超参数优化效果
print("\n=== 超参数优化效果 ===")

# 比较优化前后的核函数
kernel_initial = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp_initial = GaussianProcessRegressor(kernel=kernel_initial, optimizer=None, random_state=42)
gp_initial.fit(X_train, y_train)

print("初始核函数:", kernel_initial)
print("优化后核函数:", gp.kernel_)

# 计算优化前后的性能
y_pred_initial, y_std_initial = gp_initial.predict(X_test, return_std=True)
y_pred_optimized, y_std_optimized = gp.predict(X_test, return_std=True)

mse_initial = mean_squared_error(y_test, y_pred_initial)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)

print(f"初始核函数MSE: {mse_initial:.4f}")
print(f"优化后核函数MSE: {mse_optimized:.4f}")

# 4. 不确定性量化
print("\n=== 不确定性量化 ===")

# 生成新的测试点
X_new = np.linspace(-2, 12, 200).reshape(-1, 1)
y_pred_new, y_std_new = gp.predict(X_new, return_std=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, c='red', label='训练数据', alpha=0.7)
plt.plot(X_new, y_pred_new, 'b-', label='预测均值')
plt.fill_between(X_new.flatten(), y_pred_new - 2*y_std_new, 
                y_pred_new + 2*y_std_new, alpha=0.3, color='blue', label='95%置信区间')
plt.xlabel('x')
plt.ylabel('y')
plt.title('高斯过程外推预测')
plt.legend()
plt.grid(True)

# 采样来自后验分布的样本
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, c='red', label='训练数据', alpha=0.7)
plt.plot(X_new, y_pred_new, 'b-', label='预测均值')

# 从后验分布采样
for i in range(5):
    y_sample = gp.sample_y(X_new, random_state=42+i)
    plt.plot(X_new, y_sample, '--', alpha=0.7, label=f'样本{i+1}' if i < 1 else "")

plt.xlabel('x')
plt.ylabel('y')
plt.title('后验分布采样')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. 多维高斯过程
print("\n=== 多维高斯过程 ===")

# 生成二维数据
X_2d = np.random.rand(100, 2)
y_2d = np.sin(X_2d[:, 0]) + np.cos(X_2d[:, 1]) + 0.1 * np.random.randn(100)

# 训练二维高斯过程
kernel_2d = RBF(length_scale=[1.0, 1.0])
gp_2d = GaussianProcessRegressor(kernel=kernel_2d, random_state=42)
gp_2d.fit(X_2d, y_2d)

print("二维核函数:", gp_2d.kernel_)

# 创建网格进行预测
x1 = np.linspace(0, 1, 20)
x2 = np.linspace(0, 1, 20)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

Y_pred, Y_std = gp_2d.predict(X_grid, return_std=True)
Y_pred = Y_pred.reshape(20, 20)
Y_std = Y_std.reshape(20, 20)

# 可视化二维预测
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# 真实函数
ax1 = fig.add_subplot(131, projection='3d')
Y_true = (np.sin(X1) + np.cos(X2))
surf1 = ax1.plot_surface(X1, X2, Y_true, cmap='viridis', alpha=0.8)
ax1.set_title('真实函数')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

# 预测均值
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.8)
ax2.scatter(X_2d[:, 0], X_2d[:, 1], y_2d, c='red', alpha=0.6)
ax2.set_title('高斯过程预测')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

# 预测不确定性
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X1, X2, Y_std, cmap='plasma', alpha=0.8)
ax3.set_title('预测标准差')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('标准差')

plt.tight_layout()
plt.show()

# 6. 分类问题的高斯过程
print("\n=== 高斯过程分类 ===")

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

# 生成分类数据
X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1,
                                  random_state=42)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42)

# 训练高斯过程分类器
gpc = GaussianProcessClassifier(kernel=RBF(1.0), random_state=42)
gpc.fit(X_train_clf, y_train_clf)

# 预测
y_pred_clf = gpc.predict(X_test_clf)
y_prob_clf = gpc.predict_proba(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"高斯过程分类准确率: {accuracy:.4f}")

# 可视化决策边界
plt.figure(figsize=(12, 5))

# 创建网格
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

# 预测网格点
Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 1)
contour = plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_clf[:, 0], X_train_clf[:, 1], c=y_train_clf, 
           edgecolors='k', cmap='viridis')
plt.colorbar(contour)
plt.title('高斯过程分类决策边界')
plt.xlabel('特征1')
plt.ylabel('特征2')

# 预测不确定性
Z_std = np.std(gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1)
Z_std = Z_std.reshape(xx.shape)

plt.subplot(1, 2, 2)
contour_std = plt.contourf(xx, yy, Z_std, alpha=0.8)
plt.scatter(X_train_clf[:, 0], X_train_clf[:, 1], c=y_train_clf, 
           edgecolors='k', cmap='viridis')
plt.colorbar(contour_std)
plt.title('预测不确定性')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.tight_layout()
plt.show()

# 7. 贝叶斯优化应用
print("\n=== 贝叶斯优化应用 ===")

# 简单的贝叶斯优化演示
def objective_function(x):
    """目标函数：寻找最小值"""
    return (x - 2)**2 + 10 * np.sin(x) + 10

# 初始观测点
X_obs = np.array([[0.0], [1.0], [3.0], [4.0]])
y_obs = objective_function(X_obs).reshape(-1, 1)

# 采集函数：期望改进（Expected Improvement）
def expected_improvement(X, gp, best_y):
    """期望改进采集函数"""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1)
    
    with np.errstate(divide='warn'):
        improvement = best_y - mu
        Z = improvement / sigma
        ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

# 贝叶斯优化循环
n_iterations = 10
best_y = np.min(y_obs)

for i in range(n_iterations):
    # 训练高斯过程
    gp_bo = GaussianProcessRegressor(kernel=RBF(1.0), random_state=42)
    gp_bo.fit(X_obs, y_obs)
    
    # 生成候选点
    X_candidates = np.linspace(0, 5, 100).reshape(-1, 1)
    
    # 计算期望改进
    ei = expected_improvement(X_candidates, gp_bo, best_y)
    
    # 选择下一个点
    next_x = X_candidates[np.argmax(ei)]
    next_y = objective_function(next_x)
    
    # 更新观测点
    X_obs = np.vstack([X_obs, next_x])
    y_obs = np.vstack([y_obs, next_y])
    
    # 更新最佳值
    if next_y < best_y:
        best_y = next_y
    
    print(f"迭代 {i+1}: x={next_x[0]:.3f}, y={next_y[0]:.3f}, 最佳y={best_y[0]:.3f}")

print(f"\n找到的最小值: y = {best_y[0]:.3f}")
print(f"对应的x值: x = {X_obs[np.argmin(y_obs)][0]:.3f}")

# 可视化贝叶斯优化过程
plt.figure(figsize=(12, 4))

# 真实函数
x_plot = np.linspace(0, 5, 100)
y_true = objective_function(x_plot)

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, 'g-', label='真实函数', alpha=0.7)
plt.scatter(X_obs[:-n_iterations], y_obs[:-n_iterations], c='blue', 
           label='初始点', alpha=0.7)
plt.scatter(X_obs[-n_iterations:], y_obs[-n_iterations:], c='red', 
           label='贝叶斯优化点', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('贝叶斯优化过程')
plt.legend()
plt.grid(True)

# 最终的高斯过程拟合
plt.subplot(1, 2, 2)
mu_final, sigma_final = gp_bo.predict(x_plot.reshape(-1, 1), return_std=True)

plt.plot(x_plot, y_true, 'g-', label='真实函数', alpha=0.7)
plt.plot(x_plot, mu_final, 'b-', label='GP预测')
plt.fill_between(x_plot, mu_final - 2*sigma_final, mu_final + 2*sigma_final, 
                alpha=0.3, color='blue', label='95%置信区间')
plt.scatter(X_obs, y_obs, c='red', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('最终高斯过程拟合')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 5. 高级特性与优化

### 5.1 稀疏高斯过程

对于大规模数据，使用诱导点方法减少计算复杂度：
- **完全独立训练条件（FITC）**
- **变分自由能（VFE）**
- **随机变分推断（SVI）**

### 5.2 多任务高斯过程

同时学习多个相关任务：
$$ \mathbf{f} \sim \mathcal{GP}(\mathbf{0}, \mathbf{K} \otimes \mathbf{B}) $$
其中$\mathbf{B}$是任务间的协方差矩阵。

### 5.3 非平稳高斯过程

使用非平稳核函数处理变化的相关性结构。

## 6. 实践建议

### 6.1 核函数选择

- **平滑函数**：使用RBF核
- **非平滑函数**：使用马顿核
- **周期函数**：使用周期核
- **线性趋势**：使用线性核

### 6.2 超参数优化

- 使用边缘似然最大化
- 考虑使用贝叶斯优化
- 注意过拟合问题

### 6.3 计算效率

- 对于n>1000的数据，考虑稀疏近似
- 使用Cholesky分解的数值稳定性
- 利用矩阵结构加速计算

## 7. 理论深入

### 7.1 再生核希尔伯特空间

高斯过程与RKHS理论密切相关，核函数定义了RKHS中的内积。

### 7.2 普遍逼近性

在适当条件下，高斯过程可以逼近任意连续函数。

### 7.3 一致性理论

高斯过程在样本数趋于无穷时具有一致性。

---

[下一节：EM算法](./em-algorithm.md)