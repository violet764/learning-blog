# 支持向量机理论

支持向量机（Support Vector Machine, SVM）是一种基于**结构风险最小化**原理的监督学习算法。它通过寻找**最优超平面**来实现分类或回归任务，核心思想是最大化不同类别之间的**间隔（Margin）**。

SVM 的独特之处在于：它只关注"最难分类"的样本点（支持向量），而忽略那些容易分类的样本。这种"关注边界"的特性使得 SVM 在高维空间和小样本场景中表现出色。

## 基本概念

### 线性分类器与超平面

对于二分类问题，给定训练样本 $D = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$，其中 $\mathbf{x}_i \in \mathbb{R}^d$，$y_i \in \{-1, +1\}$。

**超平面**的定义：

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

其中 $\mathbf{w}$ 是法向量（决定超平面方向），$b$ 是偏置项（决定超平面位置）。

超平面将特征空间分为两部分：
- $\mathbf{w}^T \mathbf{x} + b > 0$ → 预测为正类（$y = +1$）
- $\mathbf{w}^T \mathbf{x} + b < 0$ → 预测为负类（$y = -1$）

### 间隔的概念

**函数间隔（Functional Margin）**：

$$
\hat{\gamma}_i = y_i(\mathbf{w}^T \mathbf{x}_i + b)
$$

函数间隔衡量样本被正确分类的"置信度"。当 $\hat{\gamma}_i > 0$ 时，样本被正确分类。

**几何间隔（Geometric Margin）**：

$$
\gamma_i = \frac{y_i(\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|}
$$

几何间隔是样本到超平面的**实际距离**。对于同一个超平面，等比例缩放 $\mathbf{w}$ 和 $b$ 会改变函数间隔，但几何间隔保持不变。

### 支持向量

**支持向量**是距离超平面最近的那些样本点，它们满足：

$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1
$$

支持向量决定了最优超平面的位置，其他样本点不影响决策边界。这正是 SVM 的核心特点：**只有支持向量起作用**。

## 线性可分 SVM（硬间隔）

### 优化目标

对于线性可分数据，目标是找到一个超平面，使得**间隔最大化**：

$$
\max_{\mathbf{w}, b} \gamma = \max_{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|}
$$

等价于：

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2
$$

**约束条件**（所有样本被正确分类）：

$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n
$$

### 对偶问题推导

引入拉格朗日乘子 $\alpha_i \geq 0$，构建拉格朗日函数：

$$
L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]
$$

对 $\mathbf{w}$ 和 $b$ 求偏导并令其为零：

$$
\frac{\partial L}{\partial \mathbf{w}} = 0 \Rightarrow \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i
$$

$$
\frac{\partial L}{\partial b} = 0 \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0
$$

将这两个条件代入拉格朗日函数，得到**对偶问题**：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$

**约束条件**：

$$
\sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

### KKT 条件

最优解必须满足 KKT 条件：

$$
\begin{cases}
\alpha_i \geq 0 \\
y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1 \geq 0 \\
\alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1] = 0
\end{cases}
$$

📌 **关键洞察**：根据 KKT 条件，只有当 $y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$ 时，$\alpha_i > 0$。这意味着**只有支持向量对应的 $\alpha_i$ 非零**，其他样本的 $\alpha_i = 0$。

### 决策函数

训练完成后，决策函数为：

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b\right)
$$

由于只有支持向量的 $\alpha_i \neq 0$，实际只需要计算：

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b\right)
$$

## 软间隔 SVM

现实数据往往是**线性不可分**的，或者存在噪声。硬间隔 SVM 要求所有样本都被正确分类，这可能导致过拟合。软间隔 SVM 通过引入**松弛变量**来解决这个问题。

### 数学建模

引入松弛变量 $\xi_i \geq 0$，允许部分样本被错误分类：

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i
$$

**约束条件**：

$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

其中 $C > 0$ 是惩罚参数：
- $C$ 越大，对误分类的惩罚越重（趋向硬间隔）
- $C$ 越小，允许更多误分类（更宽容）

### 对偶问题

软间隔的对偶问题与硬间隔非常相似：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$

**约束条件**变为：

$$
0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

### 支持向量的类型

软间隔情况下，支持向量分为三类：

| 条件 | 类型 | 含义 |
|------|------|------|
| $\alpha_i = 0$ | 非支持向量 | 正确分类，不在边界上 |
| $0 < \alpha_i < C$ | 边界支持向量 | 恰好在间隔边界上，$\xi_i = 0$ |
| $\alpha_i = C$ | 误分类/边界内 | 可能被误分类或在间隔内，$\xi_i > 0$ |

## 核方法简介

当数据本质上非线性可分时，可以通过**特征映射**将数据投影到高维空间：

$$
\phi: \mathcal{X} \rightarrow \mathcal{F}
$$

在高维空间 $\mathcal{F}$ 中，数据可能变得线性可分。但直接计算 $\phi(\mathbf{x})$ 往往维度极高甚至无限维。

**核技巧**的核心思想：用核函数 $K(\mathbf{x}_i, \mathbf{x}_j) = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$ 替代内积运算，无需显式计算 $\phi(\mathbf{x})$。

对偶问题变为：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

决策函数变为：

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

> 详细的核方法理论请参阅 [核方法详解](./kernel-methods.md)

## 代码示例

### 线性 SVM 可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 生成线性可分数据
X, y = make_blobs(n_samples=100, centers=2, 
                  cluster_std=0.8, random_state=42)
y = 2 * y - 1  # 转换为 -1, +1

# 训练线性 SVM
svm = SVC(kernel='linear', C=1000)  # 大 C 值近似硬间隔
svm.fit(X, y)

# 获取超平面参数
w = svm.coef_[0]
b = svm.intercept_[0]

# 绘制决策边界
plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='正类')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='red', label='负类')

# 绘制支持向量
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k', linewidths=2,
            label='支持向量')

# 绘制超平面和间隔边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)

# 决策边界: w[0]*x + w[1]*y + b = 0 → y = -(w[0]*x + b)/w[1]
yy = -(w[0] * xx + b) / w[1]

# 间隔边界: y = -(w[0]*x + b ± 1)/w[1]
margin = 1 / np.linalg.norm(w)
yy_up = yy + margin * np.sqrt(1 + (w[0]/w[1])**2)
yy_down = yy - margin * np.sqrt(1 + (w[0]/w[1])**2)

plt.plot(xx, yy, 'k-', linewidth=2, label='决策边界')
plt.plot(xx, yy_up, 'k--', linewidth=1, label='间隔边界')
plt.plot(xx, yy_down, 'k--', linewidth=1)

plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title(f'线性 SVM (间隔 = {margin:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"支持向量数量: {len(svm.support_vectors_)}")
print(f"间隔大小: {2 * margin:.4f}")
```

### 软间隔效果演示

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 生成带噪声的数据
X, y = make_classification(n_samples=200, n_features=2, 
                          n_redundant=0, n_informative=2,
                          n_clusters_per_class=1, 
                          class_sep=0.8, random_state=42)
y = 2 * y - 1

X = StandardScaler().fit_transform(X)

# 不同 C 值的效果
C_values = [0.1, 1, 10, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, C in zip(axes.ravel(), C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)
    
    # 绘制决策边界
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1])
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', alpha=0.6)
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='red', alpha=0.6)
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=80, facecolors='none', edgecolors='k', linewidths=1.5)
    
    margin = 1 / np.linalg.norm(svm.coef_[0])
    ax.set_title(f'C = {C}, 支持向量数 = {len(svm.support_vectors_)}, 间隔 = {2*margin:.3f}')

plt.tight_layout()
plt.show()
```

### 非线性分类与核函数

```python
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

# 生成非线性数据
datasets = [
    ("月牙形", make_moons(n_samples=300, noise=0.15, random_state=42)),
    ("环形", make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42))
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row, (name, (X, y)) in enumerate(datasets):
    X = StandardScaler().fit_transform(X)
    
    kernels = [
        ('线性核', SVC(kernel='linear', C=1)),
        ('多项式核', SVC(kernel='poly', degree=3, C=1)),
        ('RBF核', SVC(kernel='rbf', C=1, gamma=1))
    ]
    
    for col, (kernel_name, model) in enumerate(kernels):
        model.fit(X, y)
        
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax = axes[row, col]
        ax.contourf(xx, yy, Z, levels=20, alpha=0.5, cmap='RdBu')
        ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.7, edgecolors='k')
        
        ax.set_title(f'{name} - {kernel_name}')
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')

plt.tight_layout()
plt.show()
```

## 常见问题与注意事项

### ⚠️ 为什么必须标准化数据？

SVM 基于距离计算，如果特征尺度差异很大，大尺度特征会主导距离计算。**务必在使用 SVM 前标准化数据**。

```python
from sklearn.preprocessing import StandardScaler

# 正确做法
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm.fit(X_scaled, y)

# 预测时也需要标准化
X_test_scaled = scaler.transform(X_test)
predictions = svm.predict(X_test_scaled)
```

### ⚠️ 如何选择核函数？

| 数据特点 | 推荐核函数 | 说明 |
|---------|-----------|------|
| 线性可分/高维稀疏 | 线性核 | 计算效率高，可解释性好 |
| 中等复杂度 | 多项式核 | 需要调参 degree |
| 复杂边界 | RBF 核 | 最常用，适应性强 |
| 文本数据 | 线性核 | 文本通常高维稀疏 |

**实践经验**：先尝试线性核和 RBF 核，RBF 通常是最安全的选择。

### ⚠️ C 和 gamma 参数如何影响模型？

- **C（惩罚参数）**：
  - 大 C：低偏差高方差，可能过拟合
  - 小 C：高偏差低方差，可能欠拟合

- **gamma（RBF 核参数）**：
  - 大 gamma：决策边界复杂，可能过拟合
  - 小 gamma：决策边界平滑，可能欠拟合

```python
from sklearn.model_selection import GridSearchCV

# 网格搜索最佳参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")
```

### ⚠️ 大规模数据怎么办？

当样本量很大时，核 SVM 训练很慢（时间复杂度 $O(n^3)$）。解决方案：

1. **使用 LinearSVC**：专门优化的线性 SVM
2. **使用 SGDClassifier**：随机梯度下降，适合流式数据
3. **核近似方法**：如 Nystroem 方法或 RBFSampler

```python
from sklearn.linear_model import SGDClassifier

# 大规模数据使用 SGD
sgd_svm = SGDClassifier(loss='hinge', penalty='l2', 
                        max_iter=1000, random_state=42)
sgd_svm.fit(X_large, y_large)
```

## 参考资料

- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 7
- scikit-learn SVM 文档: https://scikit-learn.org/stable/modules/svm.html
