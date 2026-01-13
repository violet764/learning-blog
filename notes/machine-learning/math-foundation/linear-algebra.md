# 线性代数与矩阵论

## 1. 向量空间理论

### 1.1 向量空间基本概念

**定义：** 设V是一个非空集合，F是一个数域（通常为实数域R或复数域C），如果在V上定义了加法和数乘运算，且满足以下8条公理，则称V是F上的向量空间：

1. **加法交换律：** $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
2. **加法结合律：** $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
3. **零向量存在：** 存在$\mathbf{0} \in V$，使得$\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. **负向量存在：** 对每个$\mathbf{v} \in V$，存在$-\mathbf{v}$使得$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
5. **数乘单位元：** $1 \cdot \mathbf{v} = \mathbf{v}$
6. **数乘结合律：** $(ab)\mathbf{v} = a(b\mathbf{v})$
7. **分配律1：** $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$
8. **分配律2：** $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$

### 1.2 基与维度

**线性无关：** 向量组$\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$称为线性无关，如果方程
$$ c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0} $$
只有零解$c_1 = c_2 = \cdots = c_n = 0$。

**基：** 向量空间V的一个基是V中线性无关的向量组，且V中每个向量都可以唯一表示为基向量的线性组合。

**维度：** 向量空间V的维度是其任意一个基包含的向量个数，记为dim(V)。

### 1.3 线性变换

**定义：** 设V和W是F上的向量空间，映射$T: V \to W$称为线性变换，如果满足：
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{v}) = cT(\mathbf{v})$

**矩阵表示：** 有限维向量空间中的线性变换可以用矩阵表示。设$\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$是V的基，$\{\mathbf{f}_1, \dots, \mathbf{f}_m\}$是W的基，则T的矩阵表示为：
$$ A = [T(\mathbf{e}_1) \ T(\mathbf{e}_2) \ \cdots \ T(\mathbf{e}_n)] $$

## 2. 矩阵运算与分解

### 2.1 特征值分解

**特征值与特征向量：** 设A是n×n矩阵，如果存在标量λ和非零向量$\mathbf{v}$使得：
$$ A\mathbf{v} = \lambda\mathbf{v} $$
则称λ是A的特征值，$\mathbf{v}$是对应的特征向量。

**特征多项式：** $p(\lambda) = \det(A - \lambda I)$

**谱定理：** 如果A是对称矩阵（$A = A^T$），则存在正交矩阵Q和对角矩阵Λ使得：
$$ A = Q\Lambda Q^T $$
其中Λ的对角线元素是A的特征值，Q的列向量是相应的特征向量。

### 2.2 奇异值分解（SVD）

**定理：** 任何m×n实矩阵A都可以分解为：
$$ A = U\Sigma V^T $$
其中：
- U是m×m正交矩阵（左奇异向量）
- V是n×n正交矩阵（右奇异向量）
- Σ是m×n对角矩阵，对角线元素是奇异值$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

**数学推导：**
考虑矩阵$A^TA$，这是一个对称半正定矩阵，可以进行特征值分解：
$$ A^TA = V\Lambda V^T $$
其中Λ的对角线元素是特征值$\lambda_i$，令$\sigma_i = \sqrt{\lambda_i}$，则：
$$ \Sigma = \begin{bmatrix} \text{diag}(\sigma_1, \dots, \sigma_r) & 0 \\ 0 & 0 \end{bmatrix} $$
$$ U = [\mathbf{u}_1, \dots, \mathbf{u}_m], \quad \mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i $$

### 2.3 矩阵范数与条件数

**Frobenius范数：** $\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2}$

**谱范数：** $\|A\|_2 = \max_{\mathbf{x} \neq 0} \frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_1$

**条件数：** $\kappa(A) = \|A\| \cdot \|A^{-1}\|$

## 3. 机器学习中的应用

### 3.1 主成分分析（PCA）

PCA的数学基础是协方差矩阵的特征值分解。设数据矩阵X已中心化，协方差矩阵为：
$$ \Sigma = \frac{1}{n-1}X^TX $$

对Σ进行特征值分解：
$$ \Sigma = V\Lambda V^T $$

选择前k个最大特征值对应的特征向量作为主成分方向，投影矩阵为：
$$ W = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k] $$

降维后的数据：
$$ Z = XW $$

### 3.2 协方差矩阵的性质

**定义：** 对于随机向量$\mathbf{X} = (X_1, \dots, X_p)^T$，协方差矩阵为：
$$ \Sigma = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T] $$

**性质：**
1. 对称性：$\Sigma = \Sigma^T$
2. 半正定性：对任意向量$\mathbf{a}$，$\mathbf{a}^T\Sigma\mathbf{a} \geq 0$
3. 对角线元素是方差：$\Sigma_{ii} = \text{Var}(X_i)$
4. 非对角线元素是协方差：$\Sigma_{ij} = \text{Cov}(X_i, X_j)$

### 3.3 马氏距离

**定义：** 对于均值向量$\boldsymbol{\mu}$和协方差矩阵Σ的多元正态分布，点$\mathbf{x}$到$\boldsymbol{\mu}$的马氏距离为：
$$ d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})} $$

**性质：**
1. 考虑了特征的相关性
2. 对线性变换具有不变性
3. 在多元正态分布中，等马氏距离的曲面是椭球面

## 4. 数值线性代数

### 4.1 矩阵求逆的数值方法

**LU分解：** $A = LU$，其中L是下三角矩阵，U是上三角矩阵。

**Cholesky分解：** 对于对称正定矩阵A，存在下三角矩阵L使得$A = LL^T$。

### 4.2 最小二乘问题的数值解

对于超定方程组$A\mathbf{x} = \mathbf{b}$，最小二乘解为：
$$ \min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 $$

**正规方程：** $A^TA\mathbf{x} = A^T\mathbf{b}$

**使用SVD的解法：** $\mathbf{x} = V\Sigma^+U^T\mathbf{b}$，其中$\Sigma^+$是Σ的伪逆。

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig, cholesky, lu
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 1. 特征值分解示例
print("=== 特征值分解 ===")
A = np.array([[4, 2], [2, 3]])  # 对称矩阵
eigenvalues, eigenvectors = eig(A)
print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)

# 验证特征值分解
A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
print("重构误差:", np.linalg.norm(A - A_reconstructed))

# 2. 奇异值分解示例
print("\n=== 奇异值分解 ===")
B = np.random.randn(5, 3)
U, S, Vt = svd(B, full_matrices=False)
print("奇异值:", S)
print("左奇异向量形状:", U.shape)
print("右奇异向量形状:", Vt.shape)

# 验证SVD
B_reconstructed = U @ np.diag(S) @ Vt
print("SVD重构误差:", np.linalg.norm(B - B_reconstructed))

# 3. PCA应用示例
print("\n=== PCA降维 ===")
X, y = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)

# 数据标准化
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_standardized.T)
print("协方差矩阵形状:", cov_matrix.shape)

# 特征值分解
eigvals, eigvecs = eig(cov_matrix)
# 按特征值大小排序
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("前5个特征值:", eigvals[:5])
print("解释方差比:", eigvals[:5] / eigvals.sum())

# 使用sklearn的PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
print("sklearn PCA解释方差比:", pca.explained_variance_ratio_)

# 4. 马氏距离计算
print("\n=== 马氏距离 ===")
# 生成多元正态分布数据
np.random.seed(42)
mu = np.array([0, 0])
sigma = np.array([[2, 1], [1, 2]])  # 协方差矩阵
X_mvn = np.random.multivariate_normal(mu, sigma, 100)

# 计算样本均值和协方差
sample_mu = X_mvn.mean(axis=0)
sample_sigma = np.cov(X_mvn.T)

# 计算马氏距离
test_point = np.array([1, 1])
diff = test_point - sample_mu
mahalanobis_dist = np.sqrt(diff.T @ np.linalg.inv(sample_sigma) @ diff)
print("马氏距离:", mahalanobis_dist)

# 5. 矩阵分解方法比较
print("\n=== 矩阵分解方法比较 ===")
C = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

# LU分解
P, L, U = lu(C)
print("LU分解验证误差:", np.linalg.norm(P @ L @ U - C))

# Cholesky分解（要求矩阵对称正定）
if np.all(np.linalg.eigvals(C) > 0):
    L_chol = cholesky(C)
    print("Cholesky分解验证误差:", np.linalg.norm(L_chol @ L_chol.T - C))

# 6. 条件数分析
print("\n=== 矩阵条件数分析 ===")
D = np.array([[1, 2], [2, 4.0001]])  # 近乎奇异的矩阵
cond_number = np.linalg.cond(D)
print("矩阵条件数:", cond_number)
print("矩阵是否病态:", "是" if cond_number > 1000 else "否")

# 7. 线性变换可视化
print("\n=== 线性变换可视化 ===")
# 定义线性变换矩阵
T = np.array([[2, 1], [1, 2]])

# 原始网格点
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X_grid, Y_grid = np.meshgrid(x, y)
points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

# 应用线性变换
transformed_points = (T @ points.T).T

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
plt.title('原始点集')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.6)
plt.title('线性变换后的点集')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

# 8. 特征向量的几何意义
print("\n=== 特征向量的几何意义 ===")
# 绘制特征向量方向
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制单位圆
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
ax.plot(circle_x, circle_y, 'b-', label='单位圆')

# 应用线性变换
ellipse_points = T @ np.vstack([circle_x, circle_y])
ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'r-', label='变换后的椭圆')

# 绘制特征向量方向
for i in range(2):
    eigenvector = eigenvectors[:, i]
    # 归一化
    eigenvector = eigenvector / np.linalg.norm(eigenvector)
    ax.arrow(0, 0, eigenvector[0], eigenvector[1], head_width=0.1, 
             head_length=0.1, fc='g', ec='g', label=f'特征向量{i+1}')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('线性变换与特征向量')
plt.show()
```

## 6. 重要定理与性质

### 6.1 秩-零化度定理

对于线性变换$T: V \to W$，有：
$$ \text{dim}(V) = \text{rank}(T) + \text{nullity}(T) $$
其中rank(T)是像空间的维度，nullity(T)是核空间的维度。

### 6.2 谱定理推广

对于正规矩阵（满足$AA^H = A^HA$），可以进行酉对角化：
$$ A = U\Lambda U^H $$
其中U是酉矩阵，Λ是对角矩阵。

### 6.3 矩阵微积分

**标量对向量求导：**
$$ \frac{\partial \mathbf{a}^T\mathbf{x}}{\partial \mathbf{x}} = \mathbf{a} $$

**二次型求导：**
$$ \frac{\partial \mathbf{x}^TA\mathbf{x}}{\partial \mathbf{x}} = (A + A^T)\mathbf{x} $$

## 7. 应用实例

### 7.1 图像压缩

使用SVD进行图像压缩：
$$ A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T $$
其中k远小于min(m,n)，可以实现高效的图像压缩。

### 7.2 推荐系统

协同过滤算法中的矩阵分解：
$$ R \approx UV^T $$
其中R是用户-物品评分矩阵，U和V是低维因子矩阵。

### 7.3 自然语言处理

词嵌入模型如Word2Vec和GloVe都基于矩阵分解思想，将词-上下文共现矩阵分解为低维表示。

## 8. 数值稳定性考虑

### 8.1 病态问题

当矩阵条件数很大时，小的输入误差会导致大的输出误差。解决方法包括：
- 正则化方法
- 使用更稳定的算法
- 提高计算精度

### 8.2 算法选择

- **小规模稠密矩阵**：直接使用特征值分解
- **大规模稀疏矩阵**：使用迭代方法如Lanczos算法
- **最小二乘问题**：优先使用SVD而非正规方程

---

[下一节：概率论与统计学](../math-foundation/probability-statistics.md)