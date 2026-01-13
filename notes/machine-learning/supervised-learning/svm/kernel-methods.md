# 核方法理论

## 1. 核方法基本概念

### 1.1 特征映射与核技巧

核方法的核心思想是通过非线性映射将数据从原始空间映射到高维特征空间，使得在原始空间中非线性可分的问题在特征空间中变得线性可分。

**数学定义：** 设$\phi: \mathcal{X} \to \mathcal{F}$是从输入空间$\mathcal{X}$到特征空间$\mathcal{F}$的映射，核函数定义为：
$$ K(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle_{\mathcal{F}} $$

### 1.2 核技巧的优势

核技巧允许我们在高维特征空间中工作，而无需显式计算特征映射$\phi(\mathbf{x})$，只需通过核函数计算内积。这避免了"维度灾难"问题。

## 2. 正定核理论

### 2.1 Mercer定理

**定理（Mercer）：** 对称函数$K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$是正定核的充分必要条件是：对于任意有限点集$\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subset \mathcal{X}$，对应的Gram矩阵：
$$ K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j) $$
是半正定矩阵。

### 2.2 再生核希尔伯特空间（RKHS）

**定义：** 希尔伯特空间$\mathcal{H}$称为关于核函数K的再生核希尔伯特空间，如果满足：
1. 对所有$\mathbf{x} \in \mathcal{X}$，$K(\cdot, \mathbf{x}) \in \mathcal{H}$
2. 再生性质：$\langle f, K(\cdot, \mathbf{x}) \rangle_{\mathcal{H}} = f(\mathbf{x})$，对所有$f \in \mathcal{H}$

## 3. 常用核函数

### 3.1 线性核
$$ K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z} $$

### 3.2 多项式核
$$ K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T\mathbf{z} + r)^d $$
其中$\gamma > 0$，$r \geq 0$，$d \in \mathbb{N}$

### 3.3 高斯核（RBF核）
$$ K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right) $$
其中$\gamma > 0$

### 3.4 Sigmoid核
$$ K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T\mathbf{z} + r) $$

## 4. 核函数的性质与构造

### 4.1 核函数的闭包性质

如果$K_1$和$K_2$是核函数，则以下函数也是核函数：
1. $K(\mathbf{x}, \mathbf{z}) = cK_1(\mathbf{x}, \mathbf{z})$，$c > 0$
2. $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$
3. $K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z})K_2(\mathbf{x}, \mathbf{z})$
4. $K(\mathbf{x}, \mathbf{z}) = f(\mathbf{x})f(\mathbf{z})$，$f: \mathcal{X} \to \mathbb{R}$

### 4.2 字符串核

用于文本数据的核函数，衡量两个字符串的相似度：
$$ K(s, t) = \sum_{u \in \Sigma^*} \phi_u(s)\phi_u(t) $$
其中$\phi_u(s)$是子串u在字符串s中出现的次数。

### 4.3 图核

用于图结构数据的核函数，比较两个图的相似性。

## 5. 核方法的数学基础

### 5.1 表示定理

**定理：** 对于任意损失函数L和单调递增正则化函数$\Omega$，优化问题：
$$ \min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i)) + \Omega(\|f\|_{\mathcal{H}}) $$
的解具有形式：
$$ f(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x}) $$

### 5.2 核主成分分析（KPCA）

KPCA是PCA的核化版本，在特征空间中进行主成分分析：

1. 计算核矩阵$K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$
2. 中心化核矩阵：$\tilde{K} = K - 1_nK - K1_n + 1_nK1_n$
3. 求解特征值问题：$\tilde{K}\boldsymbol{\alpha} = \lambda\boldsymbol{\alpha}$
4. 投影：$\phi(\mathbf{x})^T\mathbf{v}_k = \sum_{i=1}^n \alpha_i^k K(\mathbf{x}_i, \mathbf{x})$

## 6. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. 核函数可视化
print("=== 核函数可视化 ===")

# 定义一维数据点
x = np.linspace(-3, 3, 100).reshape(-1, 1)
z = np.array([0]).reshape(-1, 1)  # 参考点

# 计算不同核函数的值
linear_vals = linear_kernel(x, z).flatten()
poly_vals = polynomial_kernel(x, z, degree=3, gamma=0.1, coef0=1).flatten()
rbf_vals = rbf_kernel(x, z, gamma=0.5).flatten()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, linear_vals)
plt.title('线性核函数')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, poly_vals)
plt.title('三次多项式核函数')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, rbf_vals)
plt.title('高斯核函数 (γ=0.5)')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. 核方法在非线性分类中的应用
print("\n=== 核方法在非线性分类中的应用 ===")

# 生成非线性可分数据
X, y = make_circles(n_samples=400, noise=0.05, factor=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用不同核函数的SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_names = ['线性核', '多项式核', '高斯核', 'Sigmoid核']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    # 训练SVM
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, gamma='scale', random_state=42)
    else:
        svm = SVC(kernel=kernel, gamma='scale', random_state=42)
    
    svm.fit(X_scaled, y)
    
    # 创建网格用于绘制决策边界
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolors='k')
    axes[i].set_title(f'{name} SVM')
    axes[i].set_xlabel('特征1')
    axes[i].set_ylabel('特征2')

plt.tight_layout()
plt.show()

# 3. 核主成分分析（KPCA）
print("\n=== 核主成分分析（KPCA） ===")

# 生成瑞士卷数据
from sklearn.datasets import make_swiss_roll
X_swiss, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# 应用不同核函数的KPCA
kpca_linear = KernelPCA(kernel='linear', n_components=2)
kpca_rbf = KernelPCA(kernel='rbf', gamma=0.04, n_components=2)
kpca_poly = KernelPCA(kernel='poly', degree=3, gamma=0.1, n_components=2)

X_kpca_linear = kpca_linear.fit_transform(X_swiss)
X_kpca_rbf = kpca_rbf.fit_transform(X_swiss)
X_kpca_poly = kpca_poly.fit_transform(X_swiss)

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据
ax = axes[0, 0]
scatter = ax.scatter(X_swiss[:, 0], X_swiss[:, 2], c=color, cmap='viridis')
ax.set_title('原始数据（3D投影）')
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.colorbar(scatter, ax=ax)

# 线性核KPCA
ax = axes[0, 1]
scatter = ax.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=color, cmap='viridis')
ax.set_title('线性核KPCA')
ax.set_xlabel('第一主成分')
ax.set_ylabel('第二主成分')
plt.colorbar(scatter, ax=ax)

# 高斯核KPCA
ax = axes[1, 0]
scatter = ax.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=color, cmap='viridis')
ax.set_title('高斯核KPCA')
ax.set_xlabel('第一主成分')
ax.set_ylabel('第二主成分')
plt.colorbar(scatter, ax=ax)

# 多项式核KPCA
ax = axes[1, 1]
scatter = ax.scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=color, cmap='viridis')
ax.set_title('多项式核KPCA')
ax.set_xlabel('第一主成分')
ax.set_ylabel('第二主成分')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# 4. 自定义核函数
print("\n=== 自定义核函数 ===")

def custom_rbf_kernel(X, Y, gamma=0.1):
    """自定义RBF核函数实现"""
    # 计算平方欧氏距离矩阵
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    
    # 应用RBF核
    K = np.exp(-gamma * distances)
    return K

# 测试自定义核函数
X_test = np.random.randn(5, 3)
Y_test = np.random.randn(3, 3)

custom_kernel = custom_rbf_kernel(X_test, Y_test, gamma=0.1)
sklearn_kernel = rbf_kernel(X_test, Y_test, gamma=0.1)

print("自定义核函数矩阵:")
print(custom_kernel)
print("\nsklearn核函数矩阵:")
print(sklearn_kernel)
print(f"\n两个实现是否一致: {np.allclose(custom_kernel, sklearn_kernel)}")

# 5. 核函数参数调优
print("\n=== 核函数参数调优 ===")

# 使用月牙形数据
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_moons_scaled = scaler.fit_transform(X_moons)

# 测试不同gamma值对RBF核的影响
gamma_values = [0.1, 0.5, 1, 5, 10]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, gamma in enumerate(gamma_values):
    svm = SVC(kernel='rbf', gamma=gamma, random_state=42)
    svm.fit(X_moons_scaled, y_moons)
    
    # 创建网格
    x_min, x_max = X_moons_scaled[:, 0].min() - 0.5, X_moons_scaled[:, 0].max() + 0.5
    y_min, y_max = X_moons_scaled[:, 1].min() - 0.5, X_moons_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_moons, 
                   cmap='viridis', edgecolors='k')
    axes[i].set_title(f'RBF核, γ={gamma}')
    axes[i].set_xlabel('特征1')
    axes[i].set_ylabel('特征2')

# 最后一个子图显示支持向量
axes[5].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_moons, 
               cmap='viridis', edgecolors='k')
support_vectors = svm.support_vectors_
axes[5].scatter(support_vectors[:, 0], support_vectors[:, 1], 
               s=100, facecolors='none', edgecolors='red', linewidths=2)
axes[5].set_title('支持向量（红色圆圈）')
axes[5].set_xlabel('特征1')
axes[5].set_ylabel('特征2')

plt.tight_layout()
plt.show()

# 6. 核矩阵的特征分析
print("\n=== 核矩阵的特征分析 ===")

# 计算不同核函数的核矩阵
X_small = X_moons_scaled[:10]  # 使用前10个样本进行分析

kernels_to_analyze = {
    '线性核': linear_kernel(X_small, X_small),
    '多项式核': polynomial_kernel(X_small, X_small, degree=2),
    '高斯核': rbf_kernel(X_small, X_small, gamma=1.0)
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (name, kernel_matrix) in enumerate(kernels_to_analyze.items()):
    # 绘制核矩阵热图
    ax = axes[0, i]
    sns.heatmap(kernel_matrix, ax=ax, cmap='viridis', annot=True, fmt='.2f')
    ax.set_title(f'{name}矩阵')
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(kernel_matrix)
    eigenvalues_real = np.real(eigenvalues)
    
    ax = axes[1, i]
    ax.bar(range(1, len(eigenvalues_real)+1), eigenvalues_real)
    ax.set_title(f'{name}特征值分布')
    ax.set_xlabel('特征值序号')
    ax.set_ylabel('特征值大小')
    ax.grid(True)

plt.tight_layout()
plt.show()

# 7. 多核学习简介
print("\n=== 多核学习简介 ===")

# 简单的多核学习：核函数线性组合
def multiple_kernel_learning(X, Y, kernels, weights):
    """多核学习：核函数的线性组合"""
    combined_kernel = np.zeros((X.shape[0], Y.shape[0]))
    
    for kernel_func, weight in zip(kernels, weights):
        combined_kernel += weight * kernel_func(X, Y)
    
    return combined_kernel

# 定义多个核函数
kernel_functions = [
    lambda X, Y: linear_kernel(X, Y),
    lambda X, Y: rbf_kernel(X, Y, gamma=0.1),
    lambda X, Y: polynomial_kernel(X, Y, degree=2)
]

weights = [0.3, 0.5, 0.2]  # 核函数权重

# 计算多核
multi_kernel = multiple_kernel_learning(X_test, Y_test, kernel_functions, weights)
print("多核学习矩阵形状:", multi_kernel.shape)
print("多核学习矩阵:\n", multi_kernel)
```

## 7. 核方法的理论性质

### 7.1 通用性定理

**定理：** 对于连续核函数K，如果对应的RKHS是稠密的，则K是通用核。

### 7.2 一致性理论

核方法在适当条件下具有一致性，即当样本数趋于无穷时，估计值收敛到真实值。

### 7.3 学习速率

核方法的泛化误差界通常为$O(1/\sqrt{n})$，其中n是样本数。

## 8. 实践建议

### 8.1 核函数选择

- **线性可分数据**：使用线性核
- **适度非线性数据**：使用多项式核
- **高度非线性数据**：使用高斯核
- **文本数据**：考虑字符串核
- **图数据**：使用图核

### 8.2 参数调优

- 使用交叉验证选择最优参数
- 对于高斯核，γ值过小会导致欠拟合，过大会导致过拟合
- 对于多项式核，需要选择合适的次数和系数

### 8.3 计算效率

- 核方法的计算复杂度通常为$O(n^3)$
- 对于大规模数据，考虑使用近似方法或在线学习
- 使用缓存技术加速核矩阵计算

---

[下一节：支持向量机实现细节](../svm/svm-implementation.md)