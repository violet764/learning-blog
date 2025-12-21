# 无监督学习算法详解

无监督学习是机器学习的一个重要分支，其目标是从无标签数据中发现隐藏的模式和结构。与监督学习不同，无监督学习不需要预先标记的训练数据。

## 一、无监督学习概述

### 1.1 什么是无监督学习
无监督学习是指从无标签数据中学习数据的内在结构和模式。主要任务包括：
- **聚类（Clustering）**：将数据分组到相似的簇中
- **降维（Dimensionality Reduction）**：减少数据特征维度
- **异常检测（Anomaly Detection）**：识别异常数据点
- **关联规则学习（Association Rule Learning）**：发现数据项之间的关系

### 1.2 无监督学习的应用场景
- 客户细分和市场分析
- 图像和文档聚类
- 数据压缩和可视化
- 推荐系统
- 异常检测

## 二、聚类算法

聚类是将相似的数据点分组到同一簇中的过程，目标是使同一簇内的数据点尽可能相似，不同簇的数据点尽可能不同。

### 2.1 K-Means聚类

#### 算法原理
K-Means是最常用的聚类算法之一，通过迭代优化将数据划分为K个簇。

**算法步骤**：
1. 随机选择K个初始质心
2. 将每个数据点分配到最近的质心所在的簇
3. 重新计算每个簇的质心（均值）
4. 重复步骤2-3直到质心不再变化或达到最大迭代次数

**目标函数**（SSE，平方误差和）：
$$
J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
$$
其中$C_i$是第i个簇，$\mu_i$是第i个簇的质心。

#### Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansClustering:
    """K-Means聚类实现"""
    
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def _initialize_centroids(self, X):
        """随机初始化质心"""
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X, centroids):
        """分配数据点到最近的质心"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """更新质心位置"""
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids
    
    def fit(self, X):
        """训练K-Means模型"""
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # 分配簇标签
            old_labels = self.labels if self.labels is not None else None
            self.labels = self._assign_clusters(X, self.centroids)
            
            # 检查收敛
            if old_labels is not None and np.array_equal(old_labels, self.labels):
                break
            
            # 更新质心
            new_centroids = self._update_centroids(X, self.labels)
            
            # 处理空簇
            for i in range(self.k):
                if np.isnan(new_centroids[i]).any():
                    # 重新初始化空簇的质心
                    new_centroids[i] = X[np.random.choice(len(X))]
            
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """预测新数据的簇标签"""
        if self.centroids is None:
            raise ValueError("Model not fitted yet")
        return self._assign_clusters(X, self.centroids)
    
    def inertia_(self, X):
        """计算SSE（平方误差和）"""
        if self.labels is None:
            self.labels = self.predict(X)
        sse = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - self.centroids[i])**2)
        return sse

# K-Means演示示例
def demonstrate_kmeans():
    """K-Means聚类演示"""
    
    # 生成测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=0.60)
    
    # 使用自定义K-Means
    kmeans = KMeansClustering(k=4, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始数据
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
    plt.title('原始数据（真实标签）')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # K-Means聚类结果
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c='red', marker='X', s=200, label='质心')
    plt.title('K-Means聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    
    # 肘部法则确定K值
    plt.subplot(1, 3, 3)
    k_values = range(1, 10)
    inertias = []
    
    for k in k_values:
        kmeans_temp = KMeansClustering(k=k, random_state=42)
        kmeans_temp.fit(X)
        inertias.append(kmeans_temp.inertia_(X))
    
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('K值')
    plt.ylabel('SSE（平方误差和）')
    plt.title('肘部法则')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算轮廓系数
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, y_pred)
    print(f"轮廓系数: {silhouette_avg:.4f}")
    
    return kmeans

if __name__ == "__main__":
    demonstrate_kmeans()
```

#### 优缺点分析
**优点**：
- 简单高效，时间复杂度O(nkI)
- 适用于球形簇
- 可扩展性好

**缺点**：
- 需要预先指定K值
- 对初始质心敏感
- 对非球形簇效果差
- 对异常值敏感

### 2.2 层次聚类（Hierarchical Clustering）

#### 算法原理
层次聚类通过构建树状结构（树状图）来表示数据的层次关系。有两种主要方法：

**1. 凝聚层次聚类（Agglomerative）**：自底向上，每个数据点开始是一个簇，逐步合并
**2. 分裂层次聚类（Divisive）**：自顶向下，所有数据点开始在一个簇，逐步分裂

**常用链接准则**：
- **单链接（Single Linkage）**：簇间最小距离
- **全链接（Complete Linkage）**：簇间最大距离
- **平均链接（Average Linkage）**：簇间平均距离
- **沃德链接（Ward Linkage）**：最小化簇内方差增加

#### Python实现

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class HierarchicalClustering:
    """层次聚类实现"""
    
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.labels = None
    
    def fit(self, X, n_clusters=None, distance_threshold=None):
        """训练层次聚类模型"""
        # 计算链接矩阵
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        # 根据条件分配簇标签
        if n_clusters is not None:
            self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        elif distance_threshold is not None:
            self.labels = fcluster(self.linkage_matrix, distance_threshold, criterion='distance')
        else:
            # 默认使用肘部法则确定簇数
            self.labels = self._auto_determine_clusters()
        
        return self
    
    def _auto_determine_clusters(self):
        """自动确定最佳簇数"""
        # 计算合并距离的变化
        last = self.linkage_matrix[-10:, 2]
        acceleration = np.diff(last, 2)
        acceleration_rev = acceleration[::-1]
        
        # 找到最大的加速度变化点
        idx = np.where(acceleration_rev > np.mean(acceleration_rev))[0]
        if len(idx) > 0:
            n_clusters = idx[0] + 2
        else:
            n_clusters = 2
        
        return fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
    
    def plot_dendrogram(self, truncate_mode=None, p=30, **kwargs):
        """绘制树状图"""
        if self.linkage_matrix is None:
            raise ValueError("Model not fitted yet")
        
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix, truncate_mode=truncate_mode, p=p, **kwargs)
        plt.title('层次聚类树状图')
        plt.xlabel('样本索引')
        plt.ylabel('距离')
        plt.show()

# 层次聚类演示
def demonstrate_hierarchical():
    """层次聚类演示"""
    
    # 生成测试数据
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=50, centers=3, n_features=2, 
                     random_state=42, cluster_std=0.60)
    
    # 使用不同链接方法的层次聚类
    methods = ['single', 'complete', 'average', 'ward']
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        plt.subplot(2, 2, i+1)
        
        # 执行层次聚类
        hc = HierarchicalClustering(method=method)
        hc.fit(X, n_clusters=3)
        
        # 绘制聚类结果
        plt.scatter(X[:, 0], X[:, 1], c=hc.labels, cmap='viridis', s=50, alpha=0.7)
        plt.title(f'{method.title()}链接方法')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制详细的树状图
    hc_ward = HierarchicalClustering(method='ward')
    hc_ward.fit(X)
    hc_ward.plot_dendrogram(truncate_mode='lastp', p=12)
    
    return hc_ward
```

### 2.3 DBSCAN聚类

#### 算法原理
DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是基于密度的聚类算法。

**核心概念**：
- **核心点**：在ε半径内至少有minPts个点的点
- **边界点**：在核心点的ε邻域内，但自身不是核心点
- **噪声点**：既不是核心点也不是边界点

**算法步骤**：
1. 随机选择一个未访问的点
2. 如果该点是核心点，创建新簇并扩展簇
3. 如果该点是噪声点，标记为噪声
4. 重复直到所有点都被访问

#### Python实现

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class DBSCANClustering:
    """DBSCAN聚类实现"""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def _find_neighbors(self, X, point_idx):
        """找到ε邻域内的邻居"""
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, neighbors, cluster_id, visited, clustered):
        """扩展簇"""
        i = 0
        while i < len(neighbors):
            point_idx = neighbors[i]
            
            if not visited[point_idx]:
                visited[point_idx] = True
                new_neighbors = self._find_neighbors(X, point_idx)
                
                if len(new_neighbors) >= self.min_samples:
                    # 点为核心点，扩展邻居列表
                    neighbors = np.append(neighbors, new_neighbors)
            
            if clustered[point_idx] == -1:  # 未分配簇
                clustered[point_idx] = cluster_id
            
            i += 1
    
    def fit(self, X):
        """训练DBSCAN模型"""
        n_samples = X.shape[0]
        visited = np.zeros(n_samples, dtype=bool)
        clustered = np.full(n_samples, -1)  # -1表示未分配簇
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self._find_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                # 标记为噪声点（簇ID为-1）
                clustered[point_idx] = -1
            else:
                # 创建新簇
                clustered[point_idx] = cluster_id
                self._expand_cluster(X, neighbors, cluster_id, visited, clustered)
                cluster_id += 1
        
        self.labels = clustered
        return self
    
    def predict(self, X_new):
        """预测新数据的簇标签（简化实现）"""
        # 在实际应用中，需要更复杂的预测方法
        # 这里简化处理，返回-1（噪声点）
        return np.full(X_new.shape[0], -1)

# DBSCAN演示
def demonstrate_dbscan():
    """DBSCAN聚类演示"""
    
    # 生成包含噪声和不同密度的数据
    from sklearn.datasets import make_moons, make_blobs
    
    # 月牙形数据+噪声
    X1, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X2, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.1, random_state=42)
    X2 = X2 + [2.5, 0]
    
    # 添加噪声点
    noise = np.random.rand(50, 2) * 4 - 2
    X = np.vstack([X1, X2, noise])
    
    # 使用不同参数的DBSCAN
    params = [
        (0.1, 5),   # eps太小
        (0.2, 5),   # 合适的参数
        (0.3, 5),   # eps太大
        (0.2, 10)   # min_samples太大
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, (eps, min_samples) in enumerate(params):
        plt.subplot(2, 2, i+1)
        
        dbscan = DBSCANClustering(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        
        # 可视化结果
        unique_labels = set(dbscan.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点用黑色表示
                col = 'black'
            
            class_member_mask = (dbscan.labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.7)
        
        plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
    
    plt.tight_layout()
    plt.show()
    
    return dbscan
```

### 2.4 聚类算法比较

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| K-Means | 简单高效，可扩展性好 | 需要指定K值，对初始值敏感 | 球形簇，大数据集 |
| 层次聚类 | 不需要指定簇数，可解释性强 | 时间复杂度高，对参数敏感 | 小数据集，需要层次结构 |
| DBSCAN | 能发现任意形状簇，抗噪声 | 对参数敏感，高维数据效果差 | 任意形状簇，含噪声数据 |
| 高斯混合模型 | 概率模型，软聚类 | 计算复杂，可能陷入局部最优 | 概率分布数据 |

### 2.5 聚类评估指标

#### 内部评估指标
- **轮廓系数（Silhouette Score）**：衡量簇内紧密度和簇间分离度
- **Calinski-Harabasz指数**：簇间方差与簇内方差的比值
- **Davies-Bouldin指数**：簇间距离与簇内直径的比值

#### 外部评估指标（需要真实标签）
- **调整兰德指数（ARI）**
- **归一化互信息（NMI）**
- **同质性、完整性和V测度**

```python
def evaluate_clustering(X, labels_true, labels_pred):
    """评估聚类结果"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    print("=== 聚类评估结果 ===")
    
    # 内部评估指标
    print(f"轮廓系数: {silhouette_score(X, labels_pred):.4f}")
    print(f"Calinski-Harabasz指数: {calinski_harabasz_score(X, labels_pred):.4f}")
    print(f"Davies-Bouldin指数: {davies_bouldin_score(X, labels_pred):.4f}")
    
    if labels_true is not None:
        # 外部评估指标
        print(f"调整兰德指数: {adjusted_rand_score(labels_true, labels_pred):.4f}")
        print(f"归一化互信息: {normalized_mutual_info_score(labels_true, labels_pred):.4f}")
```

## 三、降维算法

降维是将高维数据转换为低维表示的过程，旨在保留数据的主要结构和信息。降维的主要目的是：
- **数据可视化**：将高维数据降至2D或3D便于可视化
- **特征提取**：减少特征数量，提高模型效率
- **噪声去除**：保留主要信息，去除噪声
- **数据压缩**：减少存储和计算需求

### 3.1 主成分分析（PCA）

#### 3.1.1 算法原理
PCA通过线性变换将高维数据投影到低维空间，保留最大方差的方向。其数学基础是特征值分解。

**数学推导**：
1. **数据标准化**：将数据标准化为零均值和单位方差
2. **计算协方差矩阵**：
   $$\Sigma = \frac{1}{n-1} X^T X$$
3. **特征值分解**：求解特征值和特征向量
   $$\Sigma v = \lambda v$$
4. **选择主成分**：按特征值大小排序，选择前k个特征向量
5. **数据投影**：将数据投影到主成分空间
   $$X_{reduced} = X \cdot V_k$$

**方差解释率**：
$$\text{解释方差比例} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

#### 3.1.2 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class PCA:
    """主成分分析实现"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        """训练PCA模型"""
        # 数据标准化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值大小排序（降序）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择主成分数量
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        # 存储主成分
        self.components = eigenvectors[:, :self.n_components]
        
        # 计算解释方差
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """将数据投影到主成分空间"""
        if self.components is None:
            raise ValueError("Model not fitted yet")
        
        X_centered = X - self.mean_
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """训练并转换数据"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced):
        """将降维数据还原到原始空间"""
        if self.components is None:
            raise ValueError("Model not fitted yet")
        
        return X_reduced @ self.components.T + self.mean_

# PCA演示示例
def demonstrate_pca():
    """PCA降维演示"""
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用自定义PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始数据的前两个特征
    plt.subplot(1, 3, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('原始数据（前两个特征）')
    
    # PCA降维结果
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('第一主成分 (PC1)')
    plt.ylabel('第二主成分 (PC2)')
    plt.title('PCA降维结果')
    
    # 方差解释率
    plt.subplot(1, 3, 3)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.bar(range(1, len(cumulative_variance)+1), pca_full.explained_variance_ratio_, 
            alpha=0.7, label='单个主成分')
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 
             'ro-', label='累积解释方差')
    plt.xlabel('主成分数量')
    plt.ylabel('解释方差比例')
    plt.title('方差解释率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印解释方差信息
    print("=== PCA结果分析 ===")
    print(f"前2个主成分解释方差比例: {cumulative_variance[1]:.4f}")
    print(f"各主成分解释方差: {pca.explained_variance_ratio_}")
    
    return pca, X_pca

if __name__ == "__main__":
    demonstrate_pca()
```

#### 3.1.3 优缺点分析
**优点**：
- 数学基础坚实，理论完备
- 计算效率高，适合大数据集
- 能有效去除数据相关性
- 提供方差解释率指导选择主成分数量

**缺点**：
- 线性假设，无法处理非线性关系
- 对异常值敏感
- 主成分难以解释（线性组合）
- 需要数据标准化

### 3.2 t-SNE（t分布随机邻域嵌入）

#### 3.2.1 算法原理
t-SNE是一种非线性降维方法，特别适合高维数据的可视化。其核心思想是在高维和低维空间保持数据点之间的相似性关系。

**算法步骤**：
1. **高维空间相似度计算**：使用高斯分布计算条件概率
   $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k\neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
   $$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

2. **低维空间相似度计算**：使用t分布计算概率
   $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k\neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

3. **优化目标**：最小化KL散度
   $$KL(P\|Q) = \sum_i \sum_j p_{ij} \log\frac{p_{ij}}{q_{ij}}$$

4. **梯度下降优化**：使用动量法更新低维表示

#### 3.2.2 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

class TSNE_Manual:
    """简化版t-SNE实现（用于教学理解）"""
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding_ = None
    
    def _binary_search_perplexity(self, distances, perplexity, tol=1e-5, max_iter=50):
        """二分搜索找到合适的方差参数"""
        n = distances.shape[0]
        target_entropy = np.log(perplexity)
        
        # 初始化方差
        beta = np.ones(n)
        
        for i in range(n):
            beta_min = -np.inf
            beta_max = np.inf
            
            # 计算当前概率分布
            Di = distances[i, np.concatenate([np.r_[0:i], np.r_[i+1:n]])]
            
            for _ in range(max_iter):
                # 计算概率
                P = np.exp(-Di * beta[i])
                sum_P = np.sum(P)
                
                if sum_P == 0:
                    break
                
                P = P / sum_P
                entropy = -np.sum(P * np.log(P + 1e-8))
                
                # 调整方差
                entropy_diff = entropy - target_entropy
                
                if np.abs(entropy_diff) < tol:
                    break
                
                if entropy_diff > 0:
                    beta_min = beta[i]
                    if beta_max == np.inf:
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + beta_max) / 2
                else:
                    beta_max = beta[i]
                    if beta_min == -np.inf:
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + beta_min) / 2
        
        return beta
    
    def _compute_p_matrix(self, X):
        """计算高维空间的相似度矩阵P"""
        n = X.shape[0]
        
        # 计算距离矩阵
        distances = squareform(pdist(X, 'euclidean'))
        
        # 二分搜索找到合适的方差
        beta = self._binary_search_perplexity(distances, self.perplexity)
        
        # 计算条件概率
        P = np.zeros((n, n))
        
        for i in range(n):
            # 排除自身
            indices = np.concatenate([np.r_[0:i], np.r_[i+1:n]])
            Di = distances[i, indices]
            
            # 计算概率
            P_i = np.exp(-Di * beta[i])
            P_i = P_i / np.sum(P_i)
            
            P[i, indices] = P_i
        
        # 对称化
        P = (P + P.T) / (2 * n)
        
        return P
    
    def fit_transform(self, X):
        """训练并转换数据"""
        n = X.shape[0]
        
        # 计算高维相似度矩阵
        P = self._compute_p_matrix(X)
        
        # 初始化低维表示
        Y = np.random.randn(n, self.n_components) * 1e-4
        
        # 梯度下降优化
        gains = np.ones((n, self.n_components))
        momentum = 0.5
        final_momentum = 0.8
        
        for iteration in range(self.n_iter):
            # 计算低维相似度矩阵Q
            distances_low = squareform(pdist(Y, 'euclidean'))
            Q = 1.0 / (1.0 + distances_low**2)
            np.fill_diagonal(Q, 0)
            Q = Q / np.sum(Q)
            
            # 计算梯度
            PQ = P - Q
            grad = np.zeros((n, self.n_components))
            
            for i in range(n):
                diff = Y[i] - Y
                grad[i] = 4 * np.sum((PQ[i, :, np.newaxis] * Q[i, :, np.newaxis] * diff), axis=0)
            
            # 更新低维表示
            gains = (gains + 0.2) * ((grad > 0) != (grad < 0)) + (gains * 0.8) * ((grad > 0) == (grad < 0))
            gains = np.clip(gains, 0.01, np.inf)
            
            grad = grad * gains
            
            if iteration < 250:
                momentum = 0.5
            else:
                momentum = final_momentum
            
            Y = Y - self.learning_rate * grad + momentum * (Y - Y_prev if iteration > 0 else 0)
            Y_prev = Y.copy()
            
            # 中心化
            Y = Y - np.mean(Y, axis=0)
            
            if iteration % 100 == 0:
                cost = np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))
                print(f"迭代 {iteration}, 损失: {cost:.4f}")
        
        self.embedding_ = Y
        return Y

# t-SNE演示示例
def demonstrate_tsne():
    """t-SNE降维演示"""
    
    # 加载手写数字数据集
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # 使用sklearn的t-SNE（更稳定）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # t-SNE结果
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE降维结果（手写数字）')
    
    # 与PCA比较
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA降维结果（手写数字）')
    
    plt.tight_layout()
    plt.show()
    
    # 分析不同perplexity参数的影响
    perplexities = [5, 30, 50, 100]
    
    plt.figure(figsize=(15, 10))
    for i, perplexity in enumerate(perplexities):
        plt.subplot(2, 2, i+1)
        
        tsne_temp = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne_temp = tsne_temp.fit_transform(X)
        
        plt.scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
        plt.title(f't-SNE (perplexity={perplexity})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    return X_tsne, X_pca

if __name__ == "__main__":
    demonstrate_tsne()
```

#### 3.2.3 优缺点分析
**优点**：
- 能有效保留局部和全局结构
- 对非线性关系处理效果好
- 可视化效果优秀
- 能发现复杂的数据模式

**缺点**：
- 计算复杂度高（O(n²)）
- 结果对参数敏感（perplexity）
- 每次运行结果可能不同
- 不适合大数据集
- 无法用于新数据的预测

### 3.3 UMAP（统一流形逼近与投影）

#### 3.3.1 算法原理
UMAP是一种基于流形学习和拓扑数据分析的降维方法，结合了t-SNE的优点并改进了其计算效率。

**核心思想**：
1. **构建模糊拓扑表示**：在高维空间构建加权图
2. **优化低维表示**：最小化高维和低维拓扑结构之间的交叉熵
3. **使用黎曼几何**：基于流形假设进行数据建模

**主要改进**：
- 计算复杂度O(n log n)，适合大数据集
- 更好的全局结构保持
- 可处理新数据（有transform方法）

#### 3.3.2 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from umap import UMAP

def demonstrate_umap():
    """UMAP降维演示"""
    
    # 生成瑞士卷数据
    X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    
    # 使用UMAP
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = umap_model.fit_transform(X)
    
    # 与t-SNE比较
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始3D数据
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(1, 3, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=20, alpha=0.7)
    ax.set_title('原始数据（3D瑞士卷）')
    
    # UMAP结果
    plt.subplot(1, 3, 2)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=color, cmap='viridis', s=20, alpha=0.7)
    plt.title('UMAP降维结果')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # t-SNE结果
    plt.subplot(1, 3, 3)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis', s=20, alpha=0.7)
    plt.title('t-SNE降维结果')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()
    
    # 分析不同参数的影响
    n_neighbors_list = [5, 15, 50]
    min_dist_list = [0.1, 0.5, 0.99]
    
    plt.figure(figsize=(15, 10))
    
    for i, n_neighbors in enumerate(n_neighbors_list):
        for j, min_dist in enumerate(min_dist_list):
            plt.subplot(3, 3, i*3 + j + 1)
            
            umap_temp = UMAP(n_components=2, random_state=42, 
                           n_neighbors=n_neighbors, min_dist=min_dist)
            X_umap_temp = umap_temp.fit_transform(X)
            
            plt.scatter(X_umap_temp[:, 0], X_umap_temp[:, 1], c=color, 
                       cmap='viridis', s=10, alpha=0.7)
            plt.title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return X_umap, X_tsne

# UMAP参数说明
class UMAPParameters:
    """UMAP关键参数说明"""
    
    def __init__(self):
        self.parameters = {
            'n_neighbors': {
                'description': '局部邻域大小',
                'effect': '值越小，保留更多局部结构；值越大，保留更多全局结构',
                'typical_range': [5, 50],
                'default': 15
            },
            'min_dist': {
                'description': '低维空间中点之间的最小距离',
                'effect': '值越小，点越聚集；值越大，点越分散',
                'typical_range': [0.0, 1.0],
                'default': 0.1
            },
            'n_components': {
                'description': '降维后的维度',
                'effect': '通常设为2或3用于可视化',
                'typical_range': [2, 100],
                'default': 2
            },
            'metric': {
                'description': '距离度量方法',
                'effect': '影响数据的拓扑结构构建',
                'options': ['euclidean', 'cosine', 'manhattan'],
                'default': 'euclidean'
            }
        }
    
    def print_parameters(self):
        """打印参数说明"""
        print("=== UMAP关键参数说明 ===")
        for param, info in self.parameters.items():
            print(f"\n{param}:")
            print(f"  描述: {info['description']}")
            print(f"  影响: {info['effect']}")
            if 'typical_range' in info:
                print(f"  典型范围: {info['typical_range']}")
            if 'options' in info:
                print(f"  可选值: {info['options']}")
            print(f"  默认值: {info['default']}")

if __name__ == "__main__":
    demonstrate_umap()
    
    # 打印参数说明
    param_guide = UMAPParameters()
    param_guide.print_parameters()
```

### 3.4 其他降维方法

#### 3.4.1 线性判别分析（LDA）
- **有监督降维**：利用类别标签信息
- **目标**：最大化类间距离，最小化类内距离
- **适用场景**：分类问题中的特征降维

#### 3.4.2 等距映射（Isomap）
- **基于流形学习**：保持测地距离
- **步骤**：构建邻域图 → 计算最短路径 → MDS降维
- **优点**：能处理非线性流形

#### 3.4.3 局部线性嵌入（LLE）
- **局部线性假设**：每个点由其邻居线性重构
- **步骤**：找邻居 → 计算重构权重 → 优化低维表示
- **特点**：保持局部几何结构

### 3.5 降维算法比较与选择指南

| 算法 | 类型 | 时间复杂度 | 优点 | 缺点 | 适用场景 |
|------|------|------------|------|------|----------|
| PCA | 线性 | O(p³ + np²) | 理论完备，计算高效 | 线性假设，无法处理非线性 | 大数据集，线性关系明显 |
| t-SNE | 非线性 | O(n²) | 可视化效果好，保留局部结构 | 计算慢，参数敏感 | 小数据集可视化 |
| UMAP | 非线性 | O(n log n) | 计算效率高，全局局部平衡 | 参数调优复杂 | 大数据集，复杂流形 |
| LDA | 有监督线性 | O(np²) | 利用类别信息，分类效果好 | 需要标签，线性假设 | 分类问题特征提取 |
| Isomap | 非线性 | O(n³) | 保持测地距离，流形学习 | 计算复杂，对噪声敏感 | 流形数据可视化 |

**选择指南**：
1. **数据量大小**：大数据集选PCA/UMAP，小数据集可选t-SNE
2. **线性/非线性**：线性关系用PCA，非线性用t-SNE/UMAP
3. **是否有标签**：有标签可考虑LDA
4. **计算资源**：资源有限选PCA，充足可尝试复杂方法
5. **可视化需求**：可视化优先t-SNE/UMAP

### 3.6 降维评估指标

#### 内部评估指标
- **重建误差**：衡量降维后信息损失
- **信任度（Trustworthiness）**：保留的局部结构比例
- **连续性（Continuity）**：保留的全局结构比例

#### 外部评估指标（需要真实标签）
- **分类准确率**：用降维特征训练分类器的性能
- **聚类质量**：降维后聚类结果与真实标签的相似度

```python
def evaluate_dimensionality_reduction(X_original, X_reduced, y_true=None):
    """评估降维效果"""
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    
    print("=== 降维效果评估 ===")
    
    # 计算重建误差（适用于PCA）
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=X_reduced.shape[1])
        X_reconstructed = pca.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X_original - X_reconstructed) ** 2)
        print(f"重建误差: {reconstruction_error:.4f}")
    except:
        print("重建误差: 不适用于此方法")
    
    # 计算信任度和连续性
    n_neighbors = min(10, X_original.shape[0] // 10)
    
    # 高维空间的k近邻
    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X_original)
    distances_high, indices_high = nbrs_high.kneighbors(X_original)
    
    # 低维空间的k近邻
    nbrs_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_reduced)
    distances_low, indices_low = nbrs_low.kneighbors(X_reduced)
    
    # 信任度：高维邻居在低维中也是邻居的比例
    trustworthiness = 0
    for i in range(len(X_original)):
        high_dim_neighbors = set(indices_high[i][1:])  # 排除自身
        low_dim_neighbors = set(indices_low[i][1:n_neighbors+1])
        
        # 计算交集
        intersection = high_dim_neighbors.intersection(low_dim_neighbors)
        trustworthiness += len(intersection) / len(high_dim_neighbors)
    
    trustworthiness /= len(X_original)
    print(f"信任度: {trustworthiness:.4f}")
    
    if y_true is not None:
        # 使用降维特征进行分类（如果有标签）
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(random_state=42)
        scores_original = cross_val_score(clf, X_original, y_true, cv=5)
        scores_reduced = cross_val_score(clf, X_reduced, y_true, cv=5)
        
        print(f"原始特征分类准确率: {scores_original.mean():.4f} (±{scores_original.std():.4f})")
        print(f"降维特征分类准确率: {scores_reduced.mean():.4f} (±{scores_reduced.std():.4f})")
```

## 四、实际应用案例

### 4.1 客户细分
使用聚类算法对客户进行分组，实现精准营销。

### 4.2 图像分割
将图像像素聚类，实现图像分割和对象识别。

### 4.3 异常检测
使用聚类识别异常数据点。

## 总结

无监督学习是发现数据内在结构的重要工具。通过本文的学习，您可以：

1. **掌握主要聚类算法**：K-Means、层次聚类、DBSCAN的原理和实现
2. **理解算法优缺点**：不同算法的适用场景和限制
3. **学会评估聚类结果**：使用合适的评估指标
4. **应用于实际问题**：客户细分、图像分析等场景

无监督学习在实际应用中具有广泛的价值，是数据挖掘和模式识别的重要基础。