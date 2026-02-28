# 核方法详解

核方法（Kernel Methods）是机器学习中的一个重要理论框架，它通过**核技巧（Kernel Trick）**将线性方法扩展到非线性场景。核方法的核心思想是：**在高维特征空间中进行线性运算，但无需显式计算高维映射**。

核方法不仅适用于 SVM，还可以应用于 PCA、逻辑回归、K-means 等多种算法，是一种通用的数学工具。

## 从线性到非线性

### 问题的提出

考虑一个简单的二分类问题：数据在原始空间中是**非线性可分**的，但如果将数据映射到更高维的空间，就可能变得线性可分。

**经典例子**：异或问题（XOR）

在二维空间中，异或问题无法用线性分类器解决。但如果添加一个新特征 $x_3 = x_1 \cdot x_2$，在三维空间中就可以用超平面分离。

### 特征映射

定义映射函数：

$$
\phi: \mathcal{X} \rightarrow \mathcal{F}
$$

其中 $\mathcal{X}$ 是原始输入空间，$\mathcal{F}$ 是高维特征空间。

例如，对于二维输入 $\mathbf{x} = (x_1, x_2)^T$，二次多项式映射为：

$$
\phi(\mathbf{x}) = (1, \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1 x_2, x_1^2, x_2^2)^T
$$

原始二维空间映射到了六维空间！

### 维度灾难

直接计算 $\phi(\mathbf{x})$ 存在严重问题：
- 特征空间维度可能极高（甚至无限维）
- 计算和存储开销巨大

**核技巧**提供了解决方案：很多算法只需要计算样本间的**内积** $\langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$，而内积可以通过核函数直接计算，无需显式映射。

## 核函数

### 定义

**核函数**是一个函数 $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$，满足：

$$
K(\mathbf{x}, \mathbf{z}) = \langle\phi(\mathbf{x}), \phi(\mathbf{z})\rangle_{\mathcal{F}}
$$

即核函数计算的是两个样本在特征空间中的内积。

### Mercer 定理

**定理**：对称函数 $K$ 是有效核函数的充要条件是，对于任意有限点集 $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$，对应的 **Gram 矩阵**（核矩阵）是半正定的：

$$
\mathbf{K} = \begin{bmatrix}
K(\mathbf{x}_1, \mathbf{x}_1) & \cdots & K(\mathbf{x}_1, \mathbf{x}_n) \\
\vdots & \ddots & \vdots \\
K(\mathbf{x}_n, \mathbf{x}_1) & \cdots & K(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix} \succeq 0
$$

这个定理告诉我们：**只要核矩阵半正定，就存在某个特征空间使该核函数对应于内积运算**。

### 再生核希尔伯特空间（RKHS）

对于核函数 $K$，存在唯一的希尔伯特空间 $\mathcal{H}$ 满足：
1. 对任意 $\mathbf{x}$，$K(\cdot, \mathbf{x}) \in \mathcal{H}$
2. **再生性质**：$\langle f, K(\cdot, \mathbf{x})\rangle_{\mathcal{H}} = f(\mathbf{x})$

这意味着 RKHS 中的函数可以通过核函数"再生"。

## 常用核函数

### 线性核

$$
K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}
$$

- **特点**：不进行映射，保持原始空间
- **适用场景**：线性可分数据、高维稀疏数据（如文本）
- **优势**：计算高效，可解释性强

### 多项式核

$$
K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d
$$

- **参数**：
  - $d$：多项式次数
  - $\gamma$：缩放系数
  - $r$：常数项（coef0）
- **特点**：映射到所有 $d$ 次多项式特征组成的空间
- **适用场景**：特征间存在多项式关系

**直观理解**：$d=2$ 时，原始 $d$ 维特征扩展到 $O(d^2)$ 维。

### 高斯核（RBF 核）

$$
K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma\|\mathbf{x} - \mathbf{z}\|^2\right)
$$

- **参数**：$\gamma > 0$，控制核的"宽度"
- **等价形式**：$\gamma = \frac{1}{2\sigma^2}$，其中 $\sigma$ 是带宽参数
- **特点**：
  - 映射到无限维空间
  - 局部性强：距离近的点核值大，距离远的点核值趋近于 0
- **适用场景**：最常用的核函数，适合大多数非线性问题

**$\gamma$ 的影响**：
- 大 $\gamma$：核函数"尖锐"，决策边界复杂，易过拟合
- 小 $\gamma$：核函数"平滑"，决策边界简单，易欠拟合

### Sigmoid 核

$$
K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)
$$

- **特点**：源于神经网络，相当于两层神经元的激活
- **注意**：不满足 Mercer 条件（某些参数下不是正定核），实践中仍可使用
- **适用场景**：与神经网络对比研究

### 核函数对比

| 核函数 | 特征空间维度 | 参数 | 适用场景 |
|--------|------------|------|---------|
| 线性核 | 原始维度 | 无 | 线性可分、高维稀疏 |
| 多项式核 | $O(d^D)$ | $d, \gamma, r$ | 特征有多项式关系 |
| RBF 核 | 无限维 | $\gamma$ | 通用，最推荐 |
| Sigmoid 核 | 取决于参数 | $\gamma, r$ | 神经网络对比 |

## 核函数的构造

核函数具有良好的数学性质，可以通过组合构造新的核函数。

### 闭包性质

如果 $K_1$ 和 $K_2$ 是有效核函数，则以下也是有效核函数：

1. **正数乘法**：$K(\mathbf{x}, \mathbf{z}) = c \cdot K_1(\mathbf{x}, \mathbf{z})$，其中 $c > 0$

2. **加法**：$K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) + K_2(\mathbf{x}, \mathbf{z})$

3. **乘法**：$K(\mathbf{x}, \mathbf{z}) = K_1(\mathbf{x}, \mathbf{z}) \cdot K_2(\mathbf{x}, \mathbf{z})$

4. **函数乘法**：$K(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) f(\mathbf{z})$，其中 $f$ 是任意实函数

5. **指数**：$K(\mathbf{x}, \mathbf{z}) = \exp(K_1(\mathbf{x}, \mathbf{z}))$

这些性质使得我们可以灵活构造适合特定问题的核函数。

### 多核学习

将多个核函数线性组合：

$$
K(\mathbf{x}, \mathbf{z}) = \sum_{m=1}^M \beta_m K_m(\mathbf{x}, \mathbf{z})
$$

其中 $\beta_m \geq 0$ 是权重，$\sum_m \beta_m = 1$。

多核学习可以自动学习不同核函数的组合权重，适用于多源异构数据。

## 表示定理

**定理**：对于正则化优化问题

$$
\min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i)) + \lambda \|f\|_{\mathcal{H}}^2
$$

其最优解具有形式：

$$
f^*(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x})
$$

**重要意义**：
- 核方法的解可以表示为训练样本核函数的线性组合
- 无需显式计算 $\phi(\mathbf{x})$
- 将无限维问题转化为有限维（样本数量）问题

## 核方法的应用

### 核主成分分析（KPCA）

将 PCA 扩展到非线性场景，在特征空间中进行主成分分析：

1. 计算核矩阵 $\mathbf{K}_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$
2. 中心化：$\tilde{\mathbf{K}} = \mathbf{K} - \mathbf{1}_n\mathbf{K} - \mathbf{K}\mathbf{1}_n + \mathbf{1}_n\mathbf{K}\mathbf{1}_n$
3. 求解特征值问题：$\tilde{\mathbf{K}}\boldsymbol{\alpha} = \lambda\boldsymbol{\alpha}$
4. 投影：$f_k(\mathbf{x}) = \sum_{i=1}^n \alpha_i^k K(\mathbf{x}_i, \mathbf{x})$

### 核岭回归

结合岭回归与核方法：

$$
\min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{w}^T\phi(\mathbf{x}_i))^2 + \lambda\|\mathbf{w}\|^2
$$

通过表示定理，解为：

$$
f(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x})
$$

其中 $\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$。

### 核逻辑回归

将逻辑回归扩展到非线性：

$$
P(y=1|\mathbf{x}) = \sigma\left(\sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x})\right)
$$

## 代码示例

### 核函数可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

# 定义参考点
x_ref = np.array([[0, 0]])

# 创建网格
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]

# 计算不同核函数值
kernels = {
    '线性核': linear_kernel(X_grid, x_ref),
    '多项式核 (d=2)': polynomial_kernel(X_grid, x_ref, degree=2, gamma=0.5, coef0=1),
    '多项式核 (d=3)': polynomial_kernel(X_grid, x_ref, degree=3, gamma=0.5, coef0=1),
    'RBF 核 (γ=0.5)': rbf_kernel(X_grid, x_ref, gamma=0.5),
    'RBF 核 (γ=2)': rbf_kernel(X_grid, x_ref, gamma=2),
    'RBF 核 (γ=0.1)': rbf_kernel(X_grid, x_ref, gamma=0.1)
}

# 绘制热图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, K_val) in zip(axes.ravel(), kernels.items()):
    K_plot = K_val.reshape(xx.shape)
    im = ax.contourf(xx, yy, K_plot, levels=20, cmap='viridis')
    ax.set_title(name)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.plot(0, 0, 'r*', markersize=15, label='参考点')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### KPCA 降维示例

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

# 生成瑞士卷数据
X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# 不同核函数的 KPCA
kpca_methods = {
    '线性核': KernelPCA(n_components=2, kernel='linear'),
    'RBF 核': KernelPCA(n_components=2, kernel='rbf', gamma=0.01),
    '多项式核': KernelPCA(n_components=2, kernel='poly', degree=3, gamma=0.1)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 原始数据投影
ax = axes[0, 0]
ax.scatter(X[:, 0], X[:, 2], c=color, cmap='viridis', s=10)
ax.set_title('原始数据 (X-Z 平面)')
ax.set_xlabel('X')
ax.set_ylabel('Z')

# KPCA 结果
for ax, (name, kpca) in zip(axes.ravel()[1:], kpca_methods.items()):
    X_kpca = kpca.fit_transform(X)
    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap='viridis', s=10)
    ax.set_title(f'KPCA - {name}')
    ax.set_xlabel('第一主成分')
    ax.set_ylabel('第二主成分')

plt.tight_layout()
plt.show()
```

### 核函数对分类边界的影响

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 生成月牙形数据
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X = StandardScaler().fit_transform(X)

# 不同 gamma 值的 RBF 核
gamma_values = [0.1, 0.5, 1, 5, 10, 50]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, gamma in zip(axes.ravel(), gamma_values):
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X, y)
    
    # 决策边界
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), 
                         np.linspace(-2.5, 2.5, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdBu')
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
    
    # 标记支持向量
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=80, facecolors='none', edgecolors='g', linewidths=1.5)
    
    ax.set_title(f'γ = {gamma}, SV 数 = {len(svm.support_vectors_)}')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

plt.tight_layout()
plt.show()
```

### 自定义核函数

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomKernelSVM(BaseEstimator, ClassifierMixin):
    """使用自定义核函数的 SVM"""
    
    def __init__(self, kernel_func, C=1.0):
        self.kernel_func = kernel_func
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None
    
    def fit(self, X, y):
        n = X.shape[0]
        
        # 计算核矩阵
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel_func(X[i], X[j])
        
        # 使用简化 SMO（实际应用中使用 sklearn 或 libsvm）
        from sklearn.svm import SVC
        # 通过预计算核矩阵使用 SVC
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K, y)
        
        self.X_train = X
        return self
    
    def decision_function(self, X):
        n_train = self.X_train.shape[0]
        n_test = X.shape[0]
        
        K_test = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_test[i, j] = self.kernel_func(X[i], self.X_train[j])
        
        return self.svm.decision_function(K_test)
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# 自定义核函数：拉普拉斯核
def laplacian_kernel(x, z, sigma=1.0):
    """拉普拉斯核: K(x,z) = exp(-||x-z||_1 / sigma)"""
    return np.exp(-np.sum(np.abs(x - z)) / sigma)

# 使用自定义核
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = 2 * y - 1  # 转换为 -1, +1
X = StandardScaler().fit_transform(X)

custom_svm = CustomKernelSVM(kernel_func=lambda x, z: laplacian_kernel(x, z, sigma=0.5))
custom_svm.fit(X, y)

print(f"自定义拉普拉斯核 SVM 训练完成")
```

### 多核学习示例

```python
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

def combined_kernel(X, Y, weights=[0.5, 0.5], gamma=0.1, degree=3):
    """组合多个核函数"""
    K_rbf = rbf_kernel(X, Y, gamma=gamma)
    K_poly = polynomial_kernel(X, Y, degree=degree, gamma=0.1)
    return weights[0] * K_rbf + weights[1] * K_poly

# 使用预计算核矩阵
X, y = make_classification(n_samples=200, n_features=10, 
                           n_informative=5, random_state=42)
X = StandardScaler().fit_transform(X)

# 计算组合核矩阵
K_combined = combined_kernel(X, X, weights=[0.6, 0.4])

# 使用预计算核训练 SVM
svm = SVC(kernel='precomputed', C=1)
svm.fit(K_combined, y)

print("多核学习 SVM 训练完成")
print(f"支持向量数量: {len(svm.support_vectors_)}")
```

## 实践建议

### 核函数选择策略

1. **优先尝试 RBF 核**：在大多数情况下表现良好
2. **高维稀疏数据用线性核**：如文本分类，样本数远大于特征数
3. **已知特征关系用多项式核**：如果知道特征间存在多项式关系
4. **对比实验**：用交叉验证比较不同核函数的效果

### 参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")
```

### 大规模数据处理

当样本量很大时，核方法计算复杂度 $O(n^2)$ 或 $O(n^3)$ 成为瓶颈：

1. **核近似方法**：
   - Nystroem 方法：对核矩阵低秩近似
   - Random Fourier Features：对平移不变核的随机特征映射

```python
from sklearn.kernel_approximation import Nystroem, RBFSampler

# Nystroem 近似
nystroem = Nystroem(kernel='rbf', gamma=0.1, n_components=100)
X_transformed = nystroem.fit_transform(X)

# Random Fourier Features
rbf_sampler = RBFSampler(gamma=0.1, n_components=100)
X_features = rbf_sampler.fit_transform(X)

# 然后使用线性模型
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge')
clf.fit(X_transformed, y)
```

2. **使用线性 SVM**：LinearSVC 对于大规模数据更高效

## 注意事项

⚠️ **数据标准化至关重要**：核函数基于距离计算，特征尺度会影响结果。

⚠️ **RBF 核的 gamma 参数敏感**：需要仔细调优，建议使用对数尺度搜索。

⚠️ **核矩阵的存储开销**：对于大规模数据，核矩阵需要 $O(n^2)$ 内存。

⚠️ **避免过拟合**：RBF 核的强大拟合能力可能导致过拟合，注意正则化。

## 参考资料

- Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*
- Hofmann, T., et al. (2008). Kernel methods in machine learning. *Annals of Statistics*
