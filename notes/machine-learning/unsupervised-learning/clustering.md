# 聚类分析

聚类分析是将数据集划分为若干个组（簇）的过程，使得**同一组内的数据对象彼此相似**，而**不同组的数据对象不相似**。它是无监督学习中最核心的任务之一。

## 核心概念

### 问题定义

给定数据集 $X = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$，聚类目标是将 $n$ 个数据点划分为 $k$ 个簇 $C = \{C_1, C_2, \dots, C_k\}$，满足：

1. **完备性**：$\bigcup_{i=1}^k C_i = X$
2. **互斥性**：$C_i \cap C_j = \emptyset$，对于 $i \neq j$
3. **相似性**：簇内相似性最大，簇间相似性最小

### 距离度量

聚类的基础是距离（相似度）度量：

| 距离类型 | 公式 | 特点 |
|---------|------|------|
| 欧氏距离 | $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_i (x_i - y_i)^2}$ | 最常用，球形簇 |
| 曼哈顿距离 | $d(\mathbf{x}, \mathbf{y}) = \sum_i \|x_i - y_i\|$ | 鲁棒性强 |
| 余弦相似度 | $\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$ | 文本、高维数据 |
| 马氏距离 | $d_M = \sqrt{(\mathbf{x}-\mathbf{y})^T \Sigma^{-1} (\mathbf{x}-\mathbf{y})}$ | 考虑特征相关性 |

---

## K-means 聚类

K-means 是最经典的**基于划分**的聚类算法，通过迭代优化将数据分配到最近的质心所在簇。

### 算法原理

**目标函数**：最小化簇内平方误差和（SSE）

$$J = \sum_{i=1}^k \sum_{\mathbf{x} \in C_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

其中 $\boldsymbol{\mu}_i$ 是簇 $C_i$ 的质心：

$$\boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x} \in C_i} \mathbf{x}$$

### 算法步骤

1. **初始化**：随机选择 $k$ 个数据点作为初始质心
2. **分配阶段（E步）**：将每个数据点分配到最近的质心
   $$C_i^{(t)} = \{\mathbf{x}_j : \|\mathbf{x}_j - \boldsymbol{\mu}_i^{(t)}\|^2 \leq \|\mathbf{x}_j - \boldsymbol{\mu}_l^{(t)}\|^2, \forall l \neq i\}$$
3. **更新阶段（M步）**：重新计算每个簇的质心
   $$\boldsymbol{\mu}_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{\mathbf{x} \in C_i^{(t)}} \mathbf{x}$$
4. **迭代**：重复步骤 2-3 直到收敛

### 收敛性证明

**定理**：K-means 算法在有限步内收敛。

**证明要点**：
- 分配阶段：固定质心，最优分配使 $J$ 不增
- 更新阶段：固定分配，质心取均值使 $J$ 最小
- $J$ 有下界（$J \geq 0$）且单调递减
- 可能的分配方式有限，故必收敛

### K-means++ 初始化

标准 K-means 对初始质心敏感，K-means++ 通过概率选择改善：

$$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{j=1}^n D(\mathbf{x}_j)^2}$$

其中 $D(\mathbf{x}_i)$ 是 $\mathbf{x}_i$ 到最近已选质心的距离。

### 代码实现

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class MyKMeans:
    """K-means 手动实现"""
    
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # K-means++ 初始化
        self.centroids = self._kmeans_plusplus_init(X)
        
        for _ in range(self.max_iters):
            # 分配阶段：计算每个点到质心的距离
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # 更新阶段：重新计算质心
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    new_centroids[k] = X[self.labels == k].mean(axis=0)
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        
        return self
    
    def _kmeans_plusplus_init(self, X):
        """K-means++ 初始化"""
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # 计算每个点到最近质心的距离
            distances = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
            # 按距离平方的概率选择下一个质心
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(n_samples, p=probs)])
        
        return np.array(centroids)
    
    def _compute_distances(self, X):
        """计算所有点到质心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k, centroid in enumerate(self.centroids):
            distances[:, k] = np.sum((X - centroid)**2, axis=1)
        return distances
    
    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)


# 演示：肘部法则选择最佳 K 值
def elbow_method(X, k_range=range(2, 11)):
    """使用肘部法则选择最佳簇数"""
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 4))
    plt.plot(k_range, sse, 'bo-')
    plt.xlabel('簇数量 K')
    plt.ylabel('SSE (簇内平方误差和)')
    plt.title('肘部法则选择最佳 K 值')
    plt.grid(True)
    plt.show()
    
    return sse


# 生成测试数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 使用自定义实现
my_kmeans = MyKMeans(n_clusters=4)
my_kmeans.fit(X)
labels = my_kmeans.predict(X)

print(f"质心位置:\n{my_kmeans.centroids}")
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 算法简单、易于理解 | 需要预先指定 K 值 |
| 计算效率高 $O(nkd)$ | 对初始值敏感 |
| 对球形簇效果好 | 对非球形簇效果差 |
| 可扩展性好 | 对噪声和异常值敏感 |

---

## 层次聚类

层次聚类通过构建**树状结构（ dendrogram ）**来组织数据，可以展示不同层次的聚类结果。

### 算法类型

#### 凝聚式（自底向上）

1. 每个点初始为一个簇
2. 逐步合并最相似的簇
3. 直到所有点合并为一个簇

#### 分裂式（自顶向下）

1. 所有点初始在一个簇中
2. 递归分裂为更小的簇
3. 直到每个点为一个簇

### 簇间距离度量

合并决策取决于簇间距离的定义：

| 方法 | 公式 | 特点 |
|------|------|------|
| 单链接 | $d_{\min}(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$ | 能发现任意形状，但链式效应 |
| 全链接 | $d_{\max}(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$ | 紧凑簇，对异常值敏感 |
| 平均链接 | $d_{\text{avg}}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$ | 平衡单链接和全链接 |
| Ward 方法 | 合并使 SSE 增加最小的两个簇 | 倾向于生成大小相近的簇 |

### 距离更新公式（Lance-Williams）

合并 $C_i, C_j$ 为 $C_k$ 后，到其他簇 $C_l$ 的距离：

$$d(C_k, C_l) = \alpha_i d(C_i, C_l) + \alpha_j d(C_j, C_l) + \beta d(C_i, C_j) + \gamma |d(C_i, C_l) - d(C_j, C_l)|$$

不同方法对应不同的系数 $(\alpha_i, \alpha_j, \beta, \gamma)$。

### 代码实现

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def hierarchical_clustering_demo(X, method='ward'):
    """层次聚类演示"""
    
    # 计算链接矩阵
    Z = linkage(X, method=method)
    
    # 绘制树状图
    plt.figure(figsize=(12, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
    plt.title(f'层次聚类树状图 ({method} 方法)')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.tight_layout()
    plt.show()
    
    # 获取指定簇数的分割
    labels = fcluster(Z, t=4, criterion='maxclust')
    
    return Z, labels


# 不同链接方法比较
def compare_linkage_methods(X):
    """比较不同的链接方法"""
    methods = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for ax, method in zip(axes.ravel(), methods):
        Z = linkage(X, method=method)
        dendrogram(Z, ax=ax, leaf_rotation=90)
        ax.set_title(f'{method} 链接')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('距离')
    
    plt.tight_layout()
    plt.show()


# 运行演示
Z, labels = hierarchical_clustering_demo(X)
compare_linkage_methods(X[:100])  # 使用部分数据避免过于密集
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 不需要预先指定簇数 | 时间复杂度高 $O(n^3)$ 或 $O(n^2 \log n)$ |
| 可生成层次化结构 | 空间复杂度高 $O(n^2)$ |
| 可通过树状图直观展示 | 合并/分裂决策不可逆 |
| 能发现任意形状的簇 | 对噪声敏感 |

---

## DBSCAN 密度聚类

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是基于密度的聚类算法，能发现**任意形状的簇**并识别**噪声点**。

### 核心概念

**ε-邻域**：以点 $p$ 为中心，半径为 $\varepsilon$ 的区域
$$N_\varepsilon(p) = \{q \in D : d(p, q) \leq \varepsilon\}$$

**点类型**：
- **核心点**：$|N_\varepsilon(p)| \geq \text{MinPts}$
- **边界点**：在核心点的邻域内，但自身不是核心点
- **噪声点**：既非核心点也非边界点

**密度关系**：
- **直接密度可达**：$q \in N_\varepsilon(p)$ 且 $p$ 是核心点
- **密度可达**：存在点链 $p_1, \dots, p_n$ 使得 $p_{i+1}$ 从 $p_i$ 直接密度可达
- **密度相连**：存在点 $o$ 使得 $p$ 和 $q$ 都从 $o$ 密度可达

### 算法步骤

1. 标记所有点为核心点、边界点或噪声点
2. 删除噪声点
3. 为距离在 $\varepsilon$ 内的核心点之间添加边
4. 每个连通分量形成一个簇

### 参数选择：k-距离图

计算每个点到第 $k$ 个最近邻的距离，按降序排列，选择拐点作为 $\varepsilon$：

```python
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, k=5):
    """使用 k-距离图选择最佳 eps"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
    # 取第 k 个最近邻的距离
    k_distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(k_distances)
    plt.xlabel('数据点（按距离排序）')
    plt.ylabel(f'{k}-NN 距离')
    plt.title('k-距离图（拐点为最佳 eps）')
    plt.grid(True)
    plt.show()
    
    return k_distances
```

### 代码实现

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

def dbscan_demo():
    """DBSCAN 演示：处理非球形数据"""
    
    # 生成月牙形数据
    X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCAN 聚类
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    # 统计结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"簇数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制不同簇
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, col in zip(unique_labels, colors):
        if label == -1:
            # 噪声点用黑色
            col = 'black'
            marker = 'x'
        else:
            marker = 'o'
        
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[col], marker=marker, 
                   label=f'簇 {label}' if label != -1 else '噪声', alpha=0.7)
    
    plt.title(f'DBSCAN 聚类结果 (eps=0.2, min_samples=5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return labels

# 运行演示
dbscan_labels = dbscan_demo()
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 能发现任意形状的簇 | 对参数 $(\varepsilon, \text{MinPts})$ 敏感 |
| 自动确定簇数量 | 高维数据效果差（维度灾难） |
| 能识别噪声点 | 密度不均匀时效果差 |
| 只需两个参数 | 参数选择依赖经验 |

---

## 高斯混合模型（GMM）

GMM 假设数据由**多个高斯分布混合生成**，是一种**概率聚类**方法。

### 模型定义

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中：
- $\pi_k$：第 $k$ 个成分的混合系数，$\sum_k \pi_k = 1$
- $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$：高斯分布概率密度

### EM 算法求解

**E步**：计算后验概率（责任度）
$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M步**：更新参数
$$\boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_n \gamma_{nk} \mathbf{x}_n}{\sum_n \gamma_{nk}}$$

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{\sum_n \gamma_{nk} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T}{\sum_n \gamma_{nk}}$$

$$\pi_k^{\text{new}} = \frac{\sum_n \gamma_{nk}}{N}$$

### 代码实现

```python
from sklearn.mixture import GaussianMixture

def gmm_demo(X):
    """GMM 演示：概率聚类"""
    
    # 拟合 GMM
    gmm = GaussianMixture(n_components=4, random_state=42)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    
    print(f"混合系数: {gmm.weights_}")
    print(f"对数似然: {gmm.lower_bound_}")
    
    # 可视化：按概率大小调整点的大小
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title('GMM 硬聚类结果')
    
    plt.subplot(1, 2, 2)
    # 每个点的最大概率决定大小
    max_probs = probs.max(axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=max_probs*100, cmap='viridis', alpha=0.7)
    plt.title('GMM 软聚类结果（大小表示置信度）')
    
    plt.tight_layout()
    plt.show()
    
    return gmm

# 运行演示
gmm_model = gmm_demo(X)
```

### GMM vs K-means

| 特性 | K-means | GMM |
|------|---------|-----|
| 聚类类型 | 硬聚类 | 软聚类（概率） |
| 簇形状 | 球形 | 椭球形（任意协方差） |
| 簇大小 | 假设相等 | 可不同 |
| 计算复杂度 | 低 | 较高 |
| 适用场景 | 大规模、球形簇 | 需要概率解释 |

---

## 聚类评估

### 内部指标（无需真实标签）

**轮廓系数（Silhouette Coefficient）**
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

其中 $a(i)$ 是点 $i$ 到同簇其他点的平均距离，$b(i)$ 是到最近其他簇的平均距离。

**取值范围**：$[-1, 1]$，越大越好

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    """评估聚类结果"""
    metrics = {
        '轮廓系数': silhouette_score(X, labels),
        'Calinski-Harabasz 指数': calinski_harabasz_score(X, labels),
        'Davies-Bouldin 指数': davies_bouldin_score(X, labels)
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

# 评估 K-means 结果
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
evaluate_clustering(X, labels)
```

### 外部指标（需要真实标签）

- **调整兰德指数（ARI）**：衡量聚类结果与真实标签的一致性
- **标准化互信息（NMI）**：基于信息论的评估

---

## 算法选择指南

```
数据特点                      推荐算法
──────────────────────────────────────────────
大规模、球形簇                K-means
任意形状、含噪声              DBSCAN
需要层次结构                  层次聚类
需要概率解释                  GMM
簇大小差异大                  GMM
需要自动确定簇数              DBSCAN / 层次聚类
```

---

## 小结

| 算法 | 时间复杂度 | 是否需要指定 K | 簇形状 | 噪声处理 |
|------|-----------|---------------|--------|---------|
| K-means | $O(nkd)$ | 是 | 球形 | 差 |
| 层次聚类 | $O(n^2 \log n)$ | 否（可切分） | 任意 | 差 |
| DBSCAN | $O(n \log n)$ | 否 | 任意 | 好 |
| GMM | $O(nkd^2)$ | 是 | 椭球形 | 中 |

---

**下一节**：[降维技术](./dimensionality-reduction.md)
