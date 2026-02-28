# SVM 实现细节

理解 SVM 的实现细节对于实际应用至关重要。本文深入介绍 SVM 的优化算法、高效实现技巧以及工程实践要点。

## 优化问题回顾

SVM 的对偶问题为：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

约束条件：

$$
0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

这是一个**二次规划（QP）问题**，理论上可以用通用 QP 求解器，但效率不高。专门针对 SVM 设计的算法更加高效。

## SMO 算法

**序列最小优化（Sequential Minimal Optimization, SMO）** 是最流行的 SVM 训练算法，由 John Platt 于 1998 年提出。

### 核心思想

SMO 将大优化问题分解为多个**两个变量**的小优化问题：

1. 选择两个拉格朗日乘子 $\alpha_i$ 和 $\alpha_j$
2. 固定其他参数，只优化这两个变量
3. 解析求解这个小问题（有闭式解）
4. 重复直到收敛

### 为什么选择两个变量？

约束条件 $\sum_i \alpha_i y_i = 0$ 意味着：
- 如果只优化一个变量 $\alpha_i$，根据约束 $\alpha_i y_i = -\sum_{j \neq i} \alpha_j y_j$，该变量实际上被固定
- 至少需要优化两个变量才能保持约束成立

### 两变量优化问题

固定 $\alpha_3, \ldots, \alpha_n$，优化 $\alpha_1$ 和 $\alpha_2$：

$$
\min_{\alpha_1, \alpha_2} \frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 + y_1 y_2 K_{12}\alpha_1\alpha_2 - (\alpha_1 + \alpha_2) + \text{常数}
$$

约束：
- $0 \leq \alpha_1, \alpha_2 \leq C$
- $\alpha_1 y_1 + \alpha_2 y_2 = \zeta$（常数，由其他 $\alpha$ 决定）

### 闭式解

定义：
- $\eta = 2K_{12} - K_{11} - K_{22}$（二阶导数）
- $E_i = f(\mathbf{x}_i) - y_i$（预测误差）

更新公式：

$$
\alpha_2^{new} = \alpha_2^{old} - \frac{y_2(E_1 - E_2)}{\eta}
$$

然后裁剪到可行域 $[L, H]$：

$$
\alpha_2^{new,clipped} = \begin{cases}
H & \text{if } \alpha_2^{new} > H \\
\alpha_2^{new} & \text{if } L \leq \alpha_2^{new} \leq H \\
L & \text{if } \alpha_2^{new} < L
\end{cases}
$$

其中边界 $L$ 和 $H$ 由约束条件确定：

当 $y_1 \neq y_2$ 时：
$$L = \max(0, \alpha_2^{old} - \alpha_1^{old}), \quad H = \min(C, C + \alpha_2^{old} - \alpha_1^{old})$$

当 $y_1 = y_2$ 时：
$$L = \max(0, \alpha_1^{old} + \alpha_2^{old} - C), \quad H = \min(C, \alpha_1^{old} + \alpha_2^{old})$$

最后更新 $\alpha_1$：

$$
\alpha_1^{new} = \alpha_1^{old} + y_1 y_2(\alpha_2^{old} - \alpha_2^{new,clipped})
$$

### 变量选择启发式

SMO 的效率很大程度上取决于如何选择要优化的变量对。

**外层循环（选择第一个变量 $\alpha_1$）**：
1. 优先选择违反 KKT 条件的样本
2. 检查条件：$0 < \alpha_i < C$ 且 $y_i E_i \neq 0$，或 $\alpha_i = 0$ 且 $y_i E_i < 0$，或 $\alpha_i = C$ 且 $y_i E_i > 0$

**内层循环（选择第二个变量 $\alpha_2$）**：
1. 选择使 $|E_1 - E_2|$ 最大的样本（最大步长）
2. 如果效果不好，遍历所有支持向量
3. 如果仍不好，随机选择

### 阈值 $b$ 的更新

每次更新后，重新计算阈值：

$$
b = \begin{cases}
b_1 = E_1 + y_1(\alpha_1^{new} - \alpha_1^{old})K_{11} + y_2(\alpha_2^{new} - \alpha_2^{old})K_{21} + b^{old} & \text{if } 0 < \alpha_1^{new} < C \\
b_2 = E_2 + y_1(\alpha_1^{new} - \alpha_1^{old})K_{12} + y_2(\alpha_2^{new} - \alpha_2^{old})K_{22} + b^{old} & \text{if } 0 < \alpha_2^{new} < C \\
(b_1 + b_2)/2 & \text{otherwise}
\end{cases}
$$

## 简化 SMO 实现

```python
import numpy as np

class SimplifiedSMO:
    """简化的 SMO 算法实现"""
    
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.X = X
        self.y = y
        
        # 预计算核矩阵（线性核）
        self.K = X @ X.T
        
        # SMO 主循环
        for iteration in range(self.max_iter):
            num_changed = 0
            
            for i in range(n_samples):
                # 计算 Ei
                Ei = self._decision_function(i) - y[i]
                
                # 检查是否违反 KKT 条件
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alpha[i] > 0):
                    
                    # 选择第二个变量
                    j = self._select_j(i, Ei, n_samples)
                    Ej = self._decision_function(j) - y[j]
                    
                    # 保存旧值
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    
                    # 计算边界 L 和 H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算 eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    
                    if eta >= 0:
                        continue
                    
                    # 更新 alpha_j
                    self.alpha[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                    
                    # 裁剪
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新 alpha_i
                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新 b
                    b1 = Ei + y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] + \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j] + self.b
                    b2 = Ej + y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] + \
                         y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j] + self.b
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed += 1
            
            if num_changed == 0:
                break
        
        # 提取支持向量
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.alpha_sv = self.alpha[sv_mask]
        
        return self
    
    def _decision_function(self, i):
        """计算样本 i 的决策函数值"""
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b
    
    def _select_j(self, i, Ei, n_samples):
        """选择第二个变量"""
        max_delta_E = 0
        j = i
        
        # 简化：随机选择（完整实现应选择 |Ei - Ej| 最大的 j）
        candidates = [k for k in range(n_samples) if k != i]
        if len(candidates) > 0:
            j = np.random.choice(candidates)
        
        return j
    
    def predict(self, X):
        """预测"""
        K_test = X @ self.support_vectors_.T
        decision = np.sum(self.alpha_sv * self.support_labels * K_test, axis=1) + self.b
        return np.sign(decision)

# 测试
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = 2 * y - 1  # 转换为 -1, +1
X = StandardScaler().fit_transform(X)

svm = SimplifiedSMO(C=1.0)
svm.fit(X, y)

print(f"支持向量数量: {len(svm.support_vectors_)}")
print(f"训练准确率: {np.mean(svm.predict(X) == y):.2%}")
```

## 线性 SVM 的优化算法

对于线性核 SVM，可以利用问题的特殊结构设计更高效的算法。

### 坐标下降法

直接优化原始问题：

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))
$$

**对偶坐标下降（DCD）**：逐个更新 $\alpha_i$

$$
\alpha_i^{new} = \min\left(\max\left(\alpha_i - \frac{\nabla_i f}{Q_{ii}}, 0\right), C\right)
$$

其中 $Q_{ii} = \mathbf{x}_i^T \mathbf{x}_i$，$\nabla_i f = y_i(\mathbf{w}^T\mathbf{x}_i) - 1 + b y_i$

### Pegasos 算法

**Primal Estimated sub-GrAdient SOlver for SVM**：使用随机梯度下降

$$
\mathbf{w}_{t+1} = (1 - \eta_t \lambda)\mathbf{w}_t + \eta_t y_t \mathbf{x}_t \cdot \mathbb{I}[y_t \mathbf{w}_t^T \mathbf{x}_t < 1]
$$

**算法步骤**：
1. 选择随机样本 $(x_t, y_t)$
2. 如果 $y_t \mathbf{w}^T \mathbf{x}_t < 1$，更新 $\mathbf{w}$
3. 投影到球面：$\mathbf{w} \leftarrow \min(1, \frac{1/\sqrt{\lambda}}{\|\mathbf{w}\|})\mathbf{w}$

```python
class PegasosSVM:
    """Pegasos 算法实现"""
    
    def __init__(self, lam=0.01, T=1000):
        self.lam = lam
        self.T = T
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        for t in range(1, self.T + 1):
            # 学习率
            eta = 1.0 / (self.lam * t)
            
            # 随机选择样本
            i = np.random.randint(n_samples)
            
            # 检查是否需要更新
            if y[i] * np.dot(self.w, X[i]) < 1:
                self.w = (1 - eta * self.lam) * self.w + eta * y[i] * X[i]
            else:
                self.w = (1 - eta * self.lam) * self.w
        
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w)

# 测试
svm = PegasosSVM(lam=0.01, T=5000)
svm.fit(X, y)
print(f"Pegasos 训练准确率: {np.mean(svm.predict(X) == y):.2%}")
```

### LibLinear

**LibLinear** 是大规模线性分类的高效库：

| 求解器 | 问题类型 | 适用场景 |
|--------|---------|---------|
| L2-regularized L2-loss | 原始问题 | 一般用途 |
| L2-regularized L1-loss | 对偶问题 | 稀疏数据 |
| L1-regularized L2-loss | 原始问题 | 特征选择 |
| Dual coordinate descent | 对偶问题 | 样本数较少 |

```python
from sklearn.svm import LinearSVC

# 不同求解器比较
solvers = {
    'liblinear (dual)': LinearSVC(dual=True, random_state=42),
    'liblinear (primal)': LinearSVC(dual=False, random_state=42),
    'sag': LinearSVC(loss='squared_hinge', dual=False, 
                     solver='sag', random_state=42),
    'saga': LinearSVC(loss='squared_hinge', dual=False, 
                      solver='saga', random_state=42)
}

import time
for name, model in solvers.items():
    start = time.time()
    model.fit(X, y)
    elapsed = time.time() - start
    acc = model.score(X, y)
    print(f"{name}: 准确率={acc:.3f}, 时间={elapsed:.4f}s")
```

## 多类 SVM

SVM 本质是二分类器，扩展到多类有两种策略。

### 一对多（One-vs-Rest, OvR）

训练 $K$ 个二分类器，第 $k$ 个分类器区分类别 $k$ 与其他所有类别：

$$
f_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + b_k
$$

预测：$\hat{y} = \arg\max_k f_k(\mathbf{x})$

**优点**：分类器数量少（$K$ 个）
**缺点**：类别不平衡问题

### 一对一（One-vs-One, OvO）

训练 $\binom{K}{2} = \frac{K(K-1)}{2}$ 个二分类器，每对类别一个：

**优点**：每个分类器训练数据少，速度快
**缺点**：分类器数量多，预测时需投票

### 直接多类 SVM（Crammer-Singer）

$$
\min_{\mathbf{w}_1,\ldots,\mathbf{w}_K} \frac{1}{2}\sum_{k=1}^K \|\mathbf{w}_k\|^2 + C\sum_{i=1}^n \xi_i
$$

约束：$\mathbf{w}_{y_i}^T \mathbf{x}_i - \mathbf{w}_k^T \mathbf{x}_i \geq 1 - \xi_i$，对 $\forall k \neq y_i$

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris

# 加载多类数据
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

# 比较不同策略
strategies = {
    'OvR': SVC(decision_function_shape='ovr', kernel='linear'),
    'OvO': SVC(decision_function_shape='ovo', kernel='linear'),
    'Crammer-Singer': LinearSVC(multi_class='crammer_singer')
}

for name, model in strategies.items():
    model.fit(X, y)
    acc = model.score(X, y)
    print(f"{name}: 准确率 = {acc:.3f}")
```

## 实现优化技巧

### 核缓存

对于核 SVM，核矩阵计算和存储是主要瓶颈：

```python
# sklearn 的缓存机制
from sklearn.svm import SVC

# cache_size 参数控制缓存大小（MB）
svm = SVC(kernel='rbf', cache_size=1000)  # 1GB 缓存
```

### 收缩（Shrinking）

**收缩启发式**：某些样本在优化过程中不太可能成为支持向量，可以暂时从优化中移除：

```python
# sklearn 默认启用收缩
svm = SVC(kernel='rbf', shrinking=True)
```

收缩通常能减少 20-50% 的训练时间。

### 并行化

对于多类问题和网格搜索，可以并行计算：

```python
from sklearn.model_selection import GridSearchCV

# 并行网格搜索
grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)  # 使用所有 CPU
grid.fit(X, y)
```

### 核近似

对于大规模数据，使用核近似方法：

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# 核近似 + 线性模型
pipeline = Pipeline([
    ('nystroem', Nystroem(kernel='rbf', n_components=100)),
    ('sgd', SGDClassifier(loss='hinge'))
])

pipeline.fit(X_train, y_train)
```

## 代码示例

### 不同实现对比

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# 生成不同规模的数据
sizes = [100, 500, 1000, 2000, 5000]
results = []

for n in sizes:
    X, y = make_classification(n_samples=n, n_features=20, 
                               n_informative=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # 测试不同实现
    models = {
        'LinearSVC': LinearSVC(dual='auto', random_state=42),
        'SVC(linear)': SVC(kernel='linear', random_state=42),
        'SVC(rbf)': SVC(kernel='rbf', random_state=42),
        'SGD': SGDClassifier(loss='hinge', random_state=42)
    }
    
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        acc = model.score(X_test, y_test)
        
        results.append({
            'n_samples': n,
            'model': name,
            'accuracy': acc,
            'time': train_time
        })

# 可视化
import pandas as pd
df = pd.DataFrame(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for model in df['model'].unique():
    subset = df[df['model'] == model]
    ax1.plot(subset['n_samples'], subset['time'], 'o-', label=model)
    ax2.plot(subset['n_samples'], subset['accuracy'], 'o-', label=model)

ax1.set_xlabel('样本数量')
ax1.set_ylabel('训练时间 (秒)')
ax1.set_title('训练时间 vs 数据规模')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True)

ax2.set_xlabel('样本数量')
ax2.set_ylabel('准确率')
ax2.set_title('准确率 vs 数据规模')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### 超参数调优可视化

```python
from sklearn.model_selection import validation_curve

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X = StandardScaler().fit_transform(X)

# C 参数的影响
param_range = np.logspace(-3, 3, 20)
train_scores, test_scores = validation_curve(
    SVC(kernel='rbf', gamma=0.1), X, y,
    param_name='C', param_range=param_range,
    cv=5, scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, 'o-', label='训练分数')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.semilogx(param_range, test_mean, 'o-', label='验证分数')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2)

plt.xlabel('C 参数')
plt.ylabel('准确率')
plt.title('验证曲线：C 参数对模型性能的影响')
plt.legend()
plt.grid(True)
plt.show()
```

### 学习曲线分析

```python
from sklearn.model_selection import learning_curve

# 不同核函数的学习曲线
kernels = ['linear', 'rbf']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, kernel in zip(axes, kernels):
    train_sizes, train_scores, test_scores = learning_curve(
        SVC(kernel=kernel, C=1), X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', label='训练分数')
    ax.plot(train_sizes, test_mean, 'o-', label='验证分数')
    ax.set_xlabel('训练样本数')
    ax.set_ylabel('准确率')
    ax.set_title(f'{kernel} 核学习曲线')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## 工程实践要点

### 数据预处理

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 最佳实践：将预处理和模型封装为 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale'))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 类别不平衡处理

```python
# 使用 class_weight 参数
svm = SVC(kernel='rbf', class_weight='balanced')  # 自动平衡权重

# 或手动设置权重
svm = SVC(kernel='rbf', class_weight={0: 1, 1: 5})  # 类别 1 权重更高
```

### 概率输出

SVM 本身不输出概率，但可以通过 Platt Scaling 或 Isotonic Regression 估计：

```python
# 启用概率估计（会显著增加训练时间）
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

# 获取概率
prob = svm.predict_proba(X_test)
```

### 在线学习

对于流式数据或增量更新：

```python
from sklearn.linear_model import SGDClassifier

# 使用 SGD 实现"在线"SVM
svm_online = SGDClassifier(loss='hinge', penalty='l2')

# 增量训练
for batch_X, batch_y in data_stream:
    svm_online.partial_fit(batch_X, batch_y, classes=[0, 1])
```

## 性能对比总结

| 方法 | 时间复杂度 | 适用场景 | 优势 |
|------|-----------|---------|------|
| SMO | $O(n^2 \sim n^3)$ | 通用 | 核方法支持 |
| 坐标下降 | $O(n \cdot d)$ | 线性核 | 高效 |
| Pegasos | $O(T \cdot d)$ | 大规模线性 | 线性时间 |
| 核近似 | $O(n \cdot m)$ | 大规模非线性 | $m$ 为近似维度 |

## 注意事项

⚠️ **线性核首选 LinearSVC**：对于线性核，LinearSVC 比 SVC(kernel='linear') 快很多。

⚠️ **大数据用 SGD**：当样本数超过 10000 时，考虑使用 SGDClassifier。

⚠️ **概率估计开销大**：probability=True 会显著增加训练时间，非必要不启用。

⚠️ **内存管理**：核 SVM 需要存储 $O(n^2)$ 的核矩阵（或缓存），注意内存限制。

## 参考资料

- Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
- Joachims, T. (2006). Training Linear SVMs in Linear Time
- Shalev-Shwartz, S., et al. (2011). Pegasos: Primal Estimated sub-GrAdient SOlver for SVM
- LibLinear: https://www.csie.ntu.edu.tw/~cjlin/liblinear/
