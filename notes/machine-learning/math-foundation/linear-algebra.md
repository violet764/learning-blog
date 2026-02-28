# 线性代数与矩阵论

线性代数是机器学习的数学基石。从数据表示（向量、矩阵）到模型求解（特征分解、奇异值分解），再到深度学习的核心计算（张量运算），线性代数无处不在。掌握线性代数不仅能帮助你理解算法原理，还能让你写出更高效的代码。

本文将从基础概念出发，逐步深入到矩阵分解和机器学习中的实际应用。

## 向量空间与线性变换

### 向量空间

📌 **向量空间**是一个满足特定运算规则的集合，其中的元素（向量）可以进行加法和数乘运算。

**形式化定义**：设 $V$ 是一个非空集合，$F$ 是一个数域（通常是实数域 $\mathbb{R}$ 或复数域 $\mathbb{C}$）。如果 $V$ 上定义了加法和数乘运算，且满足以下 8 条公理，则称 $V$ 是 $F$ 上的向量空间：

| 公理 | 描述 |
|------|------|
| 加法交换律 | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 加法结合律 | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| 零向量存在 | $\exists \mathbf{0} \in V, \forall \mathbf{v}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 负向量存在 | $\forall \mathbf{v}, \exists -\mathbf{v}: \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| 数乘单位元 | $1 \cdot \mathbf{v} = \mathbf{v}$ |
| 数乘结合律 | $(ab)\mathbf{v} = a(b\mathbf{v})$ |
| 分配律1 | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| 分配律2 | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |

💡 **直观理解**：向量空间就像一个"容器"，里面的元素可以自由地相加和缩放，结果仍然在这个容器内。常见的例子包括：
- $\mathbb{R}^n$：$n$ 维实数向量空间
- 多项式空间
- 函数空间

### 基与维度

📌 **线性无关**：向量组 $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$ 线性无关，当且仅当方程 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0}$ 只有零解。

📌 **基**：向量空间的基是一组线性无关的向量，空间中任意向量都可以唯一地表示为这组向量的线性组合。

📌 **维度**：向量空间的维度等于其基中向量的个数，记为 $\dim(V)$。

```python
import numpy as np

# 判断向量组是否线性无关
def is_linearly_independent(vectors):
    """
    通过计算矩阵的秩来判断向量组是否线性无关
    vectors: 向量列表，每个向量是一维数组
    """
    # 将向量组成矩阵（每列是一个向量）
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    return rank == len(vectors)

# 示例
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
v4 = np.array([1, 1, 0])  # v4 = v1 + v2，与前面的向量线性相关

print("v1, v2, v3 线性无关:", is_linearly_independent([v1, v2, v3]))
print("v1, v2, v3, v4 线性无关:", is_linearly_independent([v1, v2, v3, v4]))
```

### 线性变换

📌 **线性变换**是保持向量加法和数乘运算的映射。设 $T: V \to W$ 是从向量空间 $V$ 到 $W$ 的映射，若满足：
- $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
- $T(c\mathbf{v}) = cT(\mathbf{v})$

则称 $T$ 是线性变换。

💡 **矩阵表示**：有限维空间中的线性变换可以用矩阵表示。给定 $V$ 的基 $\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$ 和 $W$ 的基 $\{\mathbf{f}_1, \dots, \mathbf{f}_m\}$，线性变换 $T$ 的矩阵表示为：

$$A = [T(\mathbf{e}_1) \ T(\mathbf{e}_2) \ \cdots \ T(\mathbf{e}_n)]$$

```python
import matplotlib.pyplot as plt

# 线性变换可视化
def visualize_linear_transform(T, title="线性变换"):
    """可视化线性变换对单位正方形的影响"""
    # 单位正方形的四个顶点
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
    ]).T
    
    # 应用变换
    transformed = T @ square
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 原始图形
    axes[0].plot(square[0], square[1], 'b-', linewidth=2)
    axes[0].fill(square[0], square[1], alpha=0.3)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('原始图形')
    
    # 变换后图形
    axes[1].plot(transformed[0], transformed[1], 'r-', linewidth=2)
    axes[1].fill(transformed[0], transformed[1], alpha=0.3, color='red')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'变换后: {title}')
    
    plt.tight_layout()
    plt.show()

# 示例变换
T_rotation = np.array([[0, -1], [1, 0]])      # 旋转90度
T_stretch = np.array([[2, 0], [0, 1]])        # x方向拉伸
T_shear = np.array([[1, 0.5], [0, 1]])        # 剪切

visualize_linear_transform(T_rotation, "旋转90°")
visualize_linear_transform(T_stretch, "拉伸")
visualize_linear_transform(T_shear, "剪切")
```

## 矩阵分解

矩阵分解是线性代数中最重要的工具之一，它将复杂的矩阵分解为简单矩阵的乘积，从而简化计算和分析。

### 特征值分解

📌 **特征值与特征向量**：对于 $n \times n$ 矩阵 $A$，如果存在标量 $\lambda$ 和非零向量 $\mathbf{v}$ 使得：

$$A\mathbf{v} = \lambda\mathbf{v}$$

则称 $\lambda$ 是 $A$ 的特征值，$\mathbf{v}$ 是对应的特征向量。

💡 **几何意义**：特征向量是线性变换下"方向不变"的向量，特征值表示该方向上的伸缩比例。

📌 **谱定理**：如果 $A$ 是对称矩阵（$A = A^T$），则存在正交矩阵 $Q$ 和对角矩阵 $\Lambda$ 使得：

$$A = Q\Lambda Q^T$$

其中 $\Lambda$ 的对角线元素是 $A$ 的特征值，$Q$ 的列向量是相应的特征向量。

```python
# 特征值分解示例
A = np.array([[4, 2], [2, 3]])  # 对称矩阵

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)

# 验证 A = QΛQ^T
Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ Q.T

print("\n重构矩阵:\n", np.real_if_close(A_reconstructed))
print("重构误差:", np.linalg.norm(A - A_reconstructed))

# 验证特征值方程
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lhs = A @ v
    rhs = eigenvalues[i] * v
    print(f"\n特征值 {eigenvalues[i]:.4f} 验证:")
    print(f"  Av = {lhs}")
    print(f"  λv = {rhs}")
```

### 奇异值分解（SVD）

📌 **奇异值分解**是特征值分解的推广，适用于任意形状的矩阵。

**定理**：任何 $m \times n$ 实矩阵 $A$ 都可以分解为：

$$A = U\Sigma V^T$$

其中：
- $U$ 是 $m \times m$ 正交矩阵，列向量称为**左奇异向量**
- $V$ 是 $n \times n$ 正交矩阵，列向量称为**右奇异向量**
- $\Sigma$ 是 $m \times n$ 对角矩阵，对角线元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 称为**奇异值**

```python
# SVD 分解示例
np.random.seed(42)
A = np.random.randn(4, 3)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("原矩阵形状:", A.shape)
print("U 形状:", U.shape)
print("奇异值:", S)
print("V^T 形状:", Vt.shape)

# 验证分解
A_reconstructed = U @ np.diag(S) @ Vt
print("\n重构误差:", np.linalg.norm(A - A_reconstructed))

# SVD 的几何意义：将变换分解为旋转-缩放-旋转
print("\nSVD 的几何意义:")
print("  V^T: 第一旋转（输入空间）")
print("  Σ:  缩放（沿坐标轴）")
print("  U:  第二旋转（输出空间）")
```

### SVD 在降维中的应用

SVD 最重要的应用之一是数据压缩和降维。保留前 $k$ 个最大的奇异值，可以得到矩阵的最佳低秩近似：

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

```python
from sklearn.datasets import load_digits

# 使用手写数字数据集演示 SVD 降维
digits = load_digits()
X = digits.data  # 1797 个样本，每个 64 维（8x8 图像）

# SVD 分解
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# 计算累积解释方差
explained_variance_ratio = (S ** 2) / (S ** 2).sum()
cumulative_variance = np.cumsum(explained_variance_ratio)

# 找到解释 90% 方差需要的成分数
k_90 = np.argmax(cumulative_variance >= 0.9) + 1
print(f"解释 90% 方差需要 {k_90} 个成分")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(S) + 1), cumulative_variance, 'b-')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% 方差')
plt.axvline(x=k_90, color='r', linestyle='--')
plt.xlabel('成分数量')
plt.ylabel('累积解释方差')
plt.title('SVD 降维效果')
plt.legend()
plt.grid(True, alpha=0.3)

# 使用不同数量的成分重构图像
plt.subplot(1, 2, 2)
image_idx = 0
original = X[image_idx].reshape(8, 8)
k_values = [5, 10, 20, 64]

for i, k in enumerate(k_values):
    X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    reconstructed = X_reconstructed[image_idx].reshape(8, 8)
    
    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'k={k}')
    plt.axis('off')

plt.suptitle('SVD 图像重构')
plt.tight_layout()
plt.show()
```

## 协方差矩阵与 PCA

### 协方差矩阵

📌 **协方差矩阵**描述了多维数据中各维度之间的线性相关程度。对于随机向量 $\mathbf{X} = (X_1, \dots, X_p)^T$，协方差矩阵定义为：

$$\Sigma = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

**性质**：
- 对称性：$\Sigma = \Sigma^T$
- 半正定性：对任意向量 $\mathbf{a}$，有 $\mathbf{a}^T\Sigma\mathbf{a} \geq 0$
- 对角线元素是各维度的方差
- 非对角线元素是两两维度间的协方差

```python
# 协方差矩阵计算
np.random.seed(42)

# 生成相关数据
n_samples = 1000
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # 强正相关
X = np.random.multivariate_normal(mean, cov, n_samples)

# 计算样本协方差矩阵
sample_cov = np.cov(X.T)
print("理论协方差矩阵:\n", np.array(cov))
print("\n样本协方差矩阵:\n", sample_cov)

# 可视化
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('二维相关数据')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# 可视化协方差椭圆
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)

# 计算特征向量（主方向）
eigenvalues, eigenvectors = np.linalg.eig(sample_cov)
for i in range(2):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    plt.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.1, fc='r', ec='r')

plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('协方差椭圆与主方向')
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 主成分分析（PCA）

📌 **主成分分析**是一种无监督降维方法，通过线性变换将数据投影到方差最大的方向上。

**核心思想**：找到一组正交基，使得数据在这组基上的投影方差最大化。这等价于对协方差矩阵进行特征值分解。

**算法步骤**：
1. 数据中心化：$\tilde{X} = X - \bar{X}$
2. 计算协方差矩阵：$\Sigma = \frac{1}{n-1}\tilde{X}^T\tilde{X}$
3. 特征值分解：$\Sigma = V\Lambda V^T$
4. 选择前 $k$ 个最大特征值对应的特征向量作为主成分

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 使用鸢尾花数据集演示 PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("原始维度:", X.shape[1])
print("降维后维度:", X_pca.shape[1])
print("\n各主成分解释方差比:", pca.explained_variance_ratio_)
print("总解释方差:", sum(pca.explained_variance_ratio_))

# 可视化
plt.figure(figsize=(12, 5))

# 原始数据（取前两个特征）
plt.subplot(1, 2, 1)
for i, name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=name, alpha=0.7)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('原始数据（前两个特征）')
plt.legend()
plt.grid(True, alpha=0.3)

# PCA 降维后
plt.subplot(1, 2, 2)
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name, alpha=0.7)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('PCA 降维结果')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 主成分载荷
plt.figure(figsize=(8, 6))
components = pca.components_
plt.imshow(components.T, cmap='coolwarm', aspect='auto')
plt.xticks([0, 1], ['PC1', 'PC2'])
plt.yticks(range(len(feature_names)), feature_names)
plt.colorbar(label='载荷')
plt.title('主成分载荷矩阵')
plt.tight_layout()
plt.show()
```

## 马氏距离

📌 **马氏距离**考虑了特征之间的相关性，是一种尺度无关的距离度量。

对于均值为 $\boldsymbol{\mu}$、协方差矩阵为 $\Sigma$ 的分布，点 $\mathbf{x}$ 的马氏距离为：

$$d_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

💡 **与欧氏距离的区别**：
- 欧氏距离假设各维度独立且方差相同
- 马氏距离考虑了维度间的相关性和不同的方差

```python
from scipy.spatial.distance import mahalanobis

# 马氏距离 vs 欧氏距离
np.random.seed(42)

# 生成椭圆形分布数据（有相关性）
mean = [0, 0]
cov = [[1, 0.9], [0.9, 1]]
X = np.random.multivariate_normal(mean, cov, 500)

# 测试点
test_points = np.array([
    [2, 2],      # 沿主轴方向
    [2, -0.5],   # 垂直主轴方向
    [0, 2],      # 另一个方向
])

# 计算协方差矩阵的逆
cov_inv = np.linalg.inv(cov)

# 计算距离
print("点 \t\t 欧氏距离 \t 马氏距离")
print("-" * 50)
for point in test_points:
    euclidean = np.linalg.norm(point - mean)
    mahal = mahalanobis(point, mean, cov_inv)
    print(f"{point} \t {euclidean:.3f} \t {mahal:.3f}")

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='数据点')
colors = ['red', 'green', 'blue']
for i, point in enumerate(test_points):
    plt.scatter(point[0], point[1], c=colors[i], s=100, marker='*', 
                label=f'点{i+1}', edgecolors='black')

# 绘制马氏距离等高线
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X_grid)
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        Z[j, i] = mahalanobis([x_grid[i], y_grid[j]], mean, cov_inv)

plt.contour(X_grid, Y_grid, Z, levels=[1, 2, 3], colors='orange', 
            linestyles='--', alpha=0.7)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('马氏距离等高线（橙色虚线）')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

## 矩阵微积分

在机器学习中，经常需要对向量和矩阵进行求导。

### 常用求导公式

| 函数 $f$ | 导数 $\frac{\partial f}{\partial \mathbf{x}}$ |
|----------|-----------------------------------------------|
| $\mathbf{a}^T\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ |
| $\|A\mathbf{x} - \mathbf{b}\|^2$ | $2A^T(A\mathbf{x} - \mathbf{b})$ |

### 矩阵求导示例

```python
# 数值验证矩阵求导公式
def numerical_gradient(f, x, eps=1e-7):
    """数值梯度计算"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# 示例：f(x) = x^T A x
np.random.seed(42)
n = 5
A = np.random.randn(n, n)
A = A + A.T  # 确保对称

x = np.random.randn(n)

# 定义函数
f = lambda x: x @ A @ x

# 解析梯度：对于对称矩阵 A，梯度为 2Ax
analytical_grad = 2 * A @ x

# 数值梯度
numerical_grad = numerical_gradient(f, x)

print("解析梯度:", analytical_grad)
print("数值梯度:", numerical_grad)
print("误差:", np.linalg.norm(analytical_grad - numerical_grad))
```

## 数值稳定性

⚠️ 在实际计算中，数值稳定性是一个重要问题。

### 条件数

📌 **条件数**衡量矩阵对输入扰动的敏感性：

$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

条件数越大，矩阵越"病态"，计算结果越不稳定。

```python
# 条件数示例
# 良态矩阵
A_good = np.array([[1, 0], [0, 1]])
print("良态矩阵条件数:", np.linalg.cond(A_good))

# 病态矩阵（近似奇异）
A_bad = np.array([[1, 1], [1, 1.0001]])
print("病态矩阵条件数:", np.linalg.cond(A_bad))

# 演示病态问题
b = np.array([1, 1])
x_good = np.linalg.solve(A_good, b)
x_bad = np.linalg.solve(A_bad, b)

print("\n良态矩阵解:", x_good)
print("病态矩阵解:", x_bad)

# 添加小扰动
b_perturbed = b + np.array([0, 0.001])
x_bad_perturbed = np.linalg.solve(A_bad, b_perturbed)
print("扰动后病态矩阵解:", x_bad_perturbed)
print("解的变化:", np.linalg.norm(x_bad_perturbed - x_bad))
```

### 正则化技巧

处理病态问题的常用方法：

```python
# 岭回归：解决病态最小二乘问题
def ridge_regression(X, y, alpha=1.0):
    """岭回归：添加正则化项"""
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)

# 普通最小二乘（可能不稳定）
def ordinary_least_squares(X, y):
    """普通最小二乘"""
    return np.linalg.solve(X.T @ X, X.T @ y)

# 示例
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# X.T @ X 可能是病态的
print("X^T X 条件数:", np.linalg.cond(X.T @ X))

# 比较
w_ols = ordinary_least_squares(X, y)
w_ridge = ridge_regression(X, y, alpha=0.1)

print("\nOLS 解范数:", np.linalg.norm(w_ols))
print("Ridge 解范数:", np.linalg.norm(w_ridge))
```

## 常见问题与注意事项

1. **维度匹配**：矩阵乘法时注意维度是否匹配，使用 `A @ B` 而不是 `A * B`（逐元素乘法）
2. **数值精度**：比较浮点数时使用容差，而不是直接 `==`
3. **奇异矩阵**：求逆前检查条件数，或使用伪逆 `np.linalg.pinv`
4. **内存效率**：大矩阵运算注意内存，可以考虑分块计算

## 参考资料

- Gilbert Strang, *Linear Algebra and Its Applications*
- 3Blue1Brown, *Essence of Linear Algebra*（视频系列）
- sklearn 文档中关于 PCA 和 SVD 的说明
