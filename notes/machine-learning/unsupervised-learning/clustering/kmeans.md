# 聚类算法

## 1. 聚类概述

聚类是将数据集划分为若干个组（簇）的过程，使得同一组内的数据对象彼此相似，而不同组的数据对象不相似。

### 1.1 聚类的基本概念

- **簇（Cluster）**：相似对象的集合
- **质心（Centroid）**：簇的中心点
- **距离度量**：衡量对象相似性的方法
- **聚类准则**：评价聚类质量的指标

## 2. K均值聚类数学原理详解

### 2.1 算法概述

K-means是一种基于划分的聚类算法，其目标是将n个数据点划分为k个簇，使得每个数据点都属于离其最近的质心所在的簇，同时最小化簇内平方误差和（SSE）。

### 2.2 数学定义

**问题形式化：**
给定数据集 $X = \{x_1, x_2, \dots, x_n\}$，其中 $x_i \in \mathbb{R}^d$，K-means的目标是找到k个簇 $C = \{C_1, C_2, \dots, C_k\}$，使得：

1. $\bigcup_{i=1}^k C_i = X$（所有数据点都被分配）
2. $C_i \cap C_j = \emptyset$ 对于 $i \neq j$（簇之间互斥）
3. 最小化目标函数：

$$J = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$

其中 $\mu_i$ 是簇 $C_i$ 的质心：

$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

### 2.3 算法步骤的数学推导

#### 2.3.1 初始化阶段

**目标**：选择k个初始质心 $\mu_1^{(0)}, \mu_2^{(0)}, \dots, \mu_k^{(0)}$

**常用方法**：
1. **随机初始化**：从数据集中随机选择k个点
2. **K-means++**：基于概率的优化初始化

**K-means++初始化公式**：
- 第一个质心随机选择：$\mu_1 = x_r$，其中 $r \sim \text{Uniform}(1, n)$
- 后续质心选择概率：
  $$P(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^n D(x_j)^2}$$
  其中 $D(x_i)$ 是 $x_i$ 到最近已选质心的距离

#### 2.3.2 分配阶段（E步）

对于每个数据点 $x_j$，分配到最近的质心所在的簇：

$$C_i^{(t)} = \{x_j : \|x_j - \mu_i^{(t)}\|^2 \leq \|x_j - \mu_l^{(t)}\|^2 \ \forall l \neq i\}$$

**数学证明**：该分配策略确保目标函数J在给定质心时最小化。

**证明**：
假设我们将 $x_j$ 分配给簇 $C_i$，其对目标函数的贡献为 $\|x_j - \mu_i\|^2$。如果将其重新分配给另一个簇 $C_l$，贡献变为 $\|x_j - \mu_l\|^2$。由于 $\|x_j - \mu_i\|^2 \leq \|x_j - \mu_l\|^2$，所以当前分配是最优的。

#### 2.3.3 更新阶段（M步）

重新计算每个簇的质心：

$$\mu_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{x \in C_i^{(t)}} x$$

**数学证明**：该更新策略确保目标函数J在给定簇分配时最小化。

**证明**：
考虑簇 $C_i$ 对目标函数的贡献：
$$J_i = \sum_{x \in C_i} \|x - \mu\|^2$$

对 $\mu$ 求导并令导数为零：
$$\frac{\partial J_i}{\partial \mu} = -2\sum_{x \in C_i} (x - \mu) = 0$$

解得：
$$\sum_{x \in C_i} x = |C_i| \mu \Rightarrow \mu = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

二阶导数验证：
$$\frac{\partial^2 J_i}{\partial \mu^2} = 2|C_i| > 0$$

因此，均值确实是极小值点。

### 2.4 收敛性证明

#### 2.4.1 目标函数单调递减

K-means算法在每次迭代中都会减少目标函数J的值：

**定理**：$J^{(t+1)} \leq J^{(t)}$，等号成立当且仅当算法收敛。

**证明**：
1. **分配阶段**：在固定质心的情况下，重新分配数据点到最近质心不会增加J
2. **更新阶段**：在固定簇分配的情况下，重新计算质心为均值会减少J

因此，每次完整的迭代（E步+M步）都会减少J或保持不变。

#### 2.4.2 有限收敛性

由于数据点数量有限，可能的簇分配方式也是有限的。目标函数J有下界（$J \geq 0$），且每次迭代严格递减（除非已收敛），因此算法必然在有限步内收敛。

### 2.5 目标函数的数学性质

#### 2.5.1 目标函数分解

目标函数可以重写为：

$$J = \sum_{i=1}^k \frac{1}{2|C_i|} \sum_{x \in C_i} \sum_{y \in C_i} \|x - y\|^2$$

**证明**：
$$\sum_{x \in C_i} \|x - \mu_i\|^2 = \frac{1}{2|C_i|} \sum_{x \in C_i} \sum_{y \in C_i} \|x - y\|^2$$

这显示了K-means实际上在最小化簇内点对距离的平方和。

#### 2.5.2 方差分解定理

总方差可以分解为簇间方差和簇内方差：

$$\text{总方差} = \text{簇间方差} + \text{簇内方差}$$

更精确地：

$$\sum_{j=1}^n \|x_j - \bar{x}\|^2 = \sum_{i=1}^k |C_i| \|\mu_i - \bar{x}\|^2 + \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$

其中 $\bar{x}$ 是整体均值。

### 2.6 算法复杂度分析

#### 2.6.1 时间复杂度

每次迭代的时间复杂度为 $O(nkd)$，其中：
- n：数据点数量
- k：簇数量
- d：数据维度

总复杂度为 $O(nkdi)$，其中i是迭代次数。

#### 2.6.2 空间复杂度

空间复杂度为 $O((n + k)d)$，主要用于存储数据和质心。

### 2.7 数学优化技巧

#### 2.7.1 距离计算优化

使用距离平方而不是欧氏距离可以避免开方运算：

$$\|x - y\|^2 = (x - y)^T(x - y) = x^Tx - 2x^Ty + y^Ty$$

对于固定质心，可以预计算 $\mu_i^T\mu_i$ 来加速距离计算。

#### 2.7.2 质心更新优化

当数据点从一个簇移动到另一个簇时，可以增量更新质心：

如果点 $x$ 从簇 $C_i$ 移动到 $C_j$：

$$\mu_i^{new} = \frac{|C_i|\mu_i^{old} - x}{|C_i| - 1}$$
$$\mu_j^{new} = \frac{|C_j|\mu_j^{old} + x}{|C_j| + 1}$$

## 3. 层次聚类

### 3.1 算法分类

#### 3.1.1 凝聚层次聚类（自底向上）
- 开始时每个点作为一个簇
- 逐步合并最相似的簇
- 最终形成一棵聚类树

#### 3.1.2 分裂层次聚类（自顶向下）
- 开始时所有点在一个簇中
- 逐步分裂为更小的簇

### 3.2 簇间距离度量

- **单链接（Single Linkage）：** $d(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$
- **全链接（Complete Linkage）：** $d(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$
- **平均链接（Average Linkage）：** $d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$
- **质心链接（Centroid Linkage）：** $d(C_i, C_j) = d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)$

## 4. DBSCAN密度聚类

### 4.1 核心概念

- **ε-邻域：** 以点为中心，半径为ε的圆形区域
- **核心点：** ε-邻域内至少包含MinPts个点的点
- **边界点：** 在核心点的ε-邻域内，但不是核心点的点
- **噪声点：** 既不是核心点也不是边界点的点

### 4.2 算法步骤

1. 标记所有点为核心点、边界点或噪声点
2. 删除噪声点
3. 为核心点之间距离在ε范围内的点添加边
4. 每个连通分量形成一个簇

### 4.3 数学公式

**密度可达性：**
如果存在点序列$p_1, p_2, \dots, p_n$，其中$p_1 = q$，$p_n = p$，且$p_{i+1}$在$p_i$的ε-邻域内，则称p从q密度可达。

**密度连接：**
如果存在点o，使得p和q都从o密度可达，则称p和q密度连接。

## 5. 高斯混合模型（GMM）

### 5.1 概率模型

假设数据由K个高斯分布混合生成：
$$ p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) $$

其中：
- $\pi_k$：第k个高斯分布的混合系数，$\sum_{k=1}^K \pi_k = 1$
- $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$：第k个高斯分布

### 5.2 EM算法求解

#### E步：计算后验概率
$$ \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} $$

#### M步：更新参数
$$ \boldsymbol{\mu}_k^{new} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n $$
$$ \boldsymbol{\Sigma}_k^{new} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k^{new})(\mathbf{x}_n - \boldsymbol{\mu}_k^{new})^T $$
$$ \pi_k^{new} = \frac{N_k}{N} $$

其中$N_k = \sum_{n=1}^N \gamma(z_{nk})$

## 6. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns

# 生成不同类型的测试数据
print("=== 聚类算法比较 ===")

# 1. 球形数据（适合K均值）
X_spherical, y_spherical = make_blobs(n_samples=300, centers=4, 
                                     cluster_std=0.60, random_state=42)

# 2. 非线性数据（适合DBSCAN）
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

# 3. 环形数据（适合层次聚类）
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                   factor=0.5, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_spherical_scaled = scaler.fit_transform(X_spherical)
X_moons_scaled = scaler.fit_transform(X_moons)
X_circles_scaled = scaler.fit_transform(X_circles)

datasets = [
    (X_spherical_scaled, y_spherical, "球形数据"),
    (X_moons_scaled, y_moons, "月牙形数据"),
    (X_circles_scaled, y_circles, "环形数据")
]

algorithms = {
    "K均值": KMeans(n_clusters=4, random_state=42),
    "层次聚类": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
    "高斯混合模型": GaussianMixture(n_components=4, random_state=42)
}

# 可视化聚类结果
fig, axes = plt.subplots(len(datasets), len(algorithms) + 1, 
                        figsize=(20, 15))

for i, (X, y_true, title) in enumerate(datasets):
    # 原始数据
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    axes[i, 0].set_title(f"{title} - 真实标签")
    
    for j, (algo_name, algo) in enumerate(algorithms.items()):
        # 预测聚类标签
        if algo_name == "高斯混合模型":
            y_pred = algo.fit_predict(X)
        else:
            y_pred = algo.fit_predict(X)
        
        # 计算评价指标
        if len(np.unique(y_pred)) > 1:  # 确保有多个簇
            silhouette = silhouette_score(X, y_pred)
        else:
            silhouette = -1
        
        # 绘制聚类结果
        axes[i, j+1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
        axes[i, j+1].set_title(f"{algo_name}\n轮廓系数: {silhouette:.3f}")

plt.tight_layout()
plt.show()

# K均值详细示例
print("\n=== K均值聚类详细分析 ===")
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# 寻找最佳K值
k_range = range(2, 10)
sse = []  # 平方误差和
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, y_pred))

# 肘部法则图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(k_range, sse, 'bo-')
plt.xlabel('K值')
plt.ylabel('SSE')
plt.title('肘部法则')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('K值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法')

plt.tight_layout()
plt.show()

# DBSCAN参数调优示例
print("\n=== DBSCAN参数调优 ===")
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

eps_values = [0.1, 0.2, 0.3, 0.4]
min_samples_values = [3, 5, 10]

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), 
                        figsize=(15, 12))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        
        n_clusters = len(np.unique(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1)
        
        axes[i, j].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
        axes[i, j].set_title(f"eps={eps}, min_samples={min_samples}\n簇数: {n_clusters}, 噪声点: {n_noise}")

plt.tight_layout()
plt.show()

# 高斯混合模型概率聚类
print("\n=== 高斯混合模型概率聚类 ===")
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
y_pred = gmm.predict(X)
probabilities = gmm.predict_proba(X)

# 可视化概率分布
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('真实标签')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('预测标签')

plt.subplot(1, 3, 3)
for k in range(3):
    plt.scatter(X[:, 0], X[:, 1], c=probabilities[:, k], 
               cmap='Reds', alpha=0.6, label=f'簇{k}概率')
plt.title('属于每个簇的概率')
plt.legend()

plt.tight_layout()
plt.show()

print("高斯混合模型参数:")
print("均值:", gmm.means_)
print("协方差:", gmm.covariances_)
print("混合系数:", gmm.weights_)
```

## 7. 聚类评价指标

### 7.1 内部评价指标

**轮廓系数（Silhouette Coefficient）：**
$$ s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} $$
其中$a(i)$是i到同簇其他点的平均距离，$b(i)$是i到其他簇的最小平均距离。

**Calinski-Harabasz指数：**
$$ CH = \frac{\text{tr}(B_k)/(k-1)}{\text{tr}(W_k)/(n-k)} $$
其中$B_k$是簇间离散矩阵，$W_k$是簇内离散矩阵。

**Davies-Bouldin指数：**
$$ DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left\{\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right\} $$
其中$\sigma_i$是簇i内点到质心的平均距离。

### 7.2 外部评价指标

**调整兰德指数（Adjusted Rand Index）：**
$$ ARI = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]} $$

**互信息（Mutual Information）：**
$$ MI(U,V) = \sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i \cap V_j|}{N} \log\left(\frac{N|U_i \cap V_j|}{|U_i||V_j|}\right) $$

## 8. 应用场景

### 8.1 客户细分
- 基于消费行为对客户进行分群
- 制定个性化营销策略

### 8.2 图像分割
- 将图像像素聚类为不同区域
- 用于目标检测和图像分析

### 8.3 异常检测
- 识别不符合任何簇模式的异常点
- 用于欺诈检测和系统监控

### 8.4 文档聚类
- 将相似文档自动分组
- 用于信息检索和主题发现

## 9. 优缺点分析

### 9.1 K均值
**优点：**
- 计算效率高
- 对球形簇效果好
- 易于理解和实现

**缺点：**
- 需要预先指定K值
- 对初始质心敏感
- 对非球形簇效果差

### 9.2 层次聚类
**优点：**
- 不需要指定簇数
- 可以得到层次化的聚类结果
- 对数据分布没有强假设

**缺点：**
- 计算复杂度高$O(n^3)$
- 对噪声敏感
- 合并/分裂决策不可逆

### 9.3 DBSCAN
**优点：**
- 能发现任意形状的簇
- 对噪声鲁棒
- 不需要指定簇数

**缺点：**
- 对参数敏感
- 高维数据效果差
- 对密度变化大的数据效果差

### 9.4 高斯混合模型
**优点：**
- 提供概率聚类
- 对椭球形簇效果好
- 可以处理不同大小的簇

**缺点：**
- 对初始值敏感
- 可能收敛到局部最优
- 需要指定成分数

## 10. 实践建议

1. **数据预处理**：标准化数据，处理异常值
2. **算法选择**：根据数据特征和业务需求选择合适算法
3. **参数调优**：使用网格搜索和评价指标优化参数
4. **结果验证**：结合业务知识验证聚类结果的合理性
5. **迭代优化**：根据反馈不断调整聚类策略

---

[下一节：降维技术](./dimensionality-reduction.md)