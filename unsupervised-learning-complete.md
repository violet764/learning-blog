# 无监督学习完全指南：聚类与降维算法

## 概述

无监督学习是机器学习的重要分支，主要目标是从未标记数据中发现隐藏的结构和模式。本指南将全面介绍两大核心无监督学习技术：**聚类分析**和**降维技术**。

## 第一部分：聚类算法

### 1.1 聚类分析基础

聚类分析是将数据集中的对象分成多个组（簇）的过程，使得：
- 同一簇内的对象相似度较高
- 不同簇间的对象相似度较低

### 1.2 主要聚类算法分类

#### 1.2.1 基于划分的方法

**K-means 聚类算法**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成示例数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-means 实现
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        # 1. 随机初始化质心
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # 2. 分配点到最近的质心
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # 3. 更新质心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
            
            # 4. 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
    
    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

# 使用示例
kmeans = CustomKMeans(n_clusters=4)
kmeans.fit(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='red')
plt.title('K-means 聚类结果')
plt.show()
```

#### 1.2.2 基于层次的方法

**凝聚层次聚类**

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# 生成数据
X, y = make_blobs(n_samples=20, centers=3, random_state=42)

# 层次聚类实现
class HierarchicalClustering:
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
    
    def fit(self, X, n_clusters=None, threshold=None):
        # 计算链接矩阵
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        if n_clusters is not None:
            self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        elif threshold is not None:
            self.labels = fcluster(self.linkage_matrix, threshold, criterion='distance')
        
        return self.labels
    
    def plot_dendrogram(self):
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title('层次聚类树状图')
        plt.xlabel('样本索引')
        plt.ylabel('距离')
        plt.show()

# 使用示例
hc = HierarchicalClustering()
labels = hc.fit(X, n_clusters=3)
hc.plot_dendrogram()
```

#### 1.2.3 基于密度的方法

**DBSCAN 算法**

```python
from sklearn.neighbors import NearestNeighbors

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 计算邻域
        neighbors = NearestNeighbors(radius=self.eps)
        neighbors.fit(X)
        neighborhoods = neighbors.radius_neighbors(X, return_distance=False)
        
        # 初始化标签
        self.labels = np.full(n_samples, -1)  # -1 表示噪声
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:  # 已经处理过
                continue
            
            # 检查是否为核心点
            if len(neighborhoods[i]) >= self.min_samples:
                # 扩展簇
                self._expand_cluster(i, neighborhoods, cluster_id)
                cluster_id += 1
        
        return self.labels
    
    def _expand_cluster(self, point_idx, neighborhoods, cluster_id):
        """扩展簇"""
        seeds = set(neighborhoods[point_idx])
        self.labels[point_idx] = cluster_id
        
        while seeds:
            current_point = seeds.pop()
            
            if self.labels[current_point] == -1:  # 噪声点
                self.labels[current_point] = cluster_id
            elif self.labels[current_point] != -1:  # 已经属于其他簇
                continue
            
            # 检查邻域
            if len(neighborhoods[current_point]) >= self.min_samples:
                # 添加新的邻域点
                seeds.update(neighborhoods[current_point])

# 使用示例
from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

dbscan = CustomDBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN 聚类结果')
plt.show()
```

### 1.3 聚类评估指标

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ClusteringEvaluator:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
    
    def evaluate(self):
        metrics = {}
        
        # 轮廓系数
        metrics['silhouette_score'] = silhouette_score(self.X, self.labels)
        
        # Calinski-Harabasz 指数
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(self.X, self.labels)
        
        # Davies-Bouldin 指数
        metrics['davies_bouldin_score'] = davies_bouldin_score(self.X, self.labels)
        
        return metrics
    
    def compare_algorithms(self, algorithms):
        """比较不同聚类算法"""
        results = {}
        
        for name, (algorithm, params) in algorithms.items():
            model = algorithm(**params)
            labels = model.fit_predict(self.X)
            
            results[name] = self.evaluate()
            results[name]['n_clusters'] = len(np.unique(labels))
        
        return pd.DataFrame(results).T

# 使用示例
evaluator = ClusteringEvaluator(X, kmeans.labels)
metrics = evaluator.evaluate()
print("聚类评估指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

## 第二部分：降维算法

### 2.1 降维技术基础

降维的主要目标：
- 减少特征数量，降低计算复杂度
- 消除冗余特征和噪声
- 可视化高维数据
- 提高模型性能

### 2.2 线性降维方法

#### 2.2.1 主成分分析 (PCA)

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CustomPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_scaled.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 排序特征值和特征向量
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择主成分
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            # 自动选择保留95%方差的成分
            total_variance = np.sum(eigenvalues)
            explained_variance_ratio = eigenvalues / total_variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            self.n_components = np.argmax(cumulative_variance >= 0.95) + 1
            self.components = eigenvectors[:, :self.n_components]
        
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled.dot(self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# 使用示例
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data

# 自定义PCA
pca = CustomPCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

print(f"解释方差比例: {pca.explained_variance_ratio}")
print(f"累计解释方差: {np.sum(pca.explained_variance_ratio):.3f}")

# 可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.title('PCA 降维结果')
plt.colorbar()
plt.show()
```

#### 2.2.2 线性判别分析 (LDA)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class CustomLDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 计算类内散度矩阵 Sw
        Sw = np.zeros((n_features, n_features))
        overall_mean = np.mean(X, axis=0)
        
        for c in np.unique(y):
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            Sw += (X_c - mean_c).T.dot(X_c - mean_c)
        
        # 计算类间散度矩阵 Sb
        Sb = np.zeros((n_features, n_features))
        for c in np.unique(y):
            n_c = np.sum(y == c)
            mean_c = np.mean(X[y == c], axis=0)
            Sb += n_c * (mean_c - overall_mean).reshape(-1, 1).dot(
                (mean_c - overall_mean).reshape(1, -1))
        
        # 求解广义特征值问题
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        
        # 排序特征向量
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择成分
        max_components = min(n_classes - 1, n_features)
        if self.n_components is None:
            self.n_components = max_components
        else:
            self.n_components = min(self.n_components, max_components)
        
        self.scalings = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        return X.dot(self.scalings)

# 使用示例
lda = CustomLDA(n_components=2)
X_lda = lda.fit_transform(iris.data, iris.target)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('LDA 成分 1')
plt.ylabel('LDA 成分 2')
plt.title('LDA 降维结果')
plt.colorbar()
plt.show()
```

### 2.3 非线性降维方法

#### 2.3.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)

```python
from sklearn.manifold import TSNE

class CustomTSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    
    def _compute_pairwise_affinities(self, X):
        """计算高维空间中的相似度"""
        n_samples = X.shape[0]
        
        # 计算平方欧氏距离
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        
        # 计算条件概率 p(j|i)
        P = np.zeros((n_samples, n_samples))
        beta = np.ones(n_samples)
        
        for i in range(n_samples):
            # 二分搜索找到合适的beta值
            beta_min = -np.inf
            beta_max = np.inf
            
            # 计算当前beta下的概率
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))]
            H, thisP = self._Hbeta(Di, beta[i])
            
            # 二分搜索调整beta
            Hdiff = H - np.log(self.perplexity)
            tries = 0
            while np.abs(Hdiff) > 1e-5 and tries < 50:
                if Hdiff > 0:
                    beta_min = beta[i]
                    if beta_max == np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + beta_max) / 2
                else:
                    beta_max = beta[i]
                    if beta_min == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + beta_min) / 2
                
                H, thisP = self._Hbeta(Di, beta[i])
                Hdiff = H - np.log(self.perplexity)
                tries += 1
            
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = thisP
        
        return P
    
    def _Hbeta(self, D, beta):
        """计算香农熵和概率分布"""
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P
    
    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # 初始化低维表示
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        
        # 计算高维相似度
        P = self._compute_pairwise_affinities(X)
        P = P + P.T
        P = P / np.sum(P)
        P = np.maximum(P, 1e-12)
        
        # 梯度下降优化
        gains = np.ones((n_samples, self.n_components))
        y_incs = np.zeros((n_samples, self.n_components))
        
        for iter in range(self.n_iter):
            # 计算低维相似度
            sum_Y = np.sum(np.square(Y), axis=1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
            
            # 计算梯度
            PQ = P - Q
            
            for i in range(n_samples):
                dY = np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T
                dY = dY * (Y[i, :] - Y)
                y_incs[i, :] = np.sum(dY, axis=0)
            
            # 更新低维表示
            gains = (gains + 0.2) * ((y_incs > 0) != (y_incs > 0))
            gains = np.maximum(gains, 0.01)
            Y += self.learning_rate * gains * y_incs
            Y = Y - np.tile(np.mean(Y, axis=0), (n_samples, 1))
        
        return Y

# 使用示例
tsne = CustomTSNE(perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(iris.data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE 降维结果')
plt.colorbar()
plt.show()
```

#### 2.3.2 UMAP (Uniform Manifold Approximation and Projection)

```python
from umap import UMAP

# UMAP 使用示例
umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(iris.data)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP 降维结果')
plt.colorbar()
plt.show()
```

### 2.4 降维算法比较

```python
class DimensionalityReductionComparison:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = {}
    
    def compare_methods(self, methods):
        """比较不同降维方法"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, method) in enumerate(methods.items()):
            # 应用降维
            if name == 'LDA':
                X_reduced = method.fit_transform(self.X, self.y)
            else:
                X_reduced = method.fit_transform(self.X)
            
            # 可视化
            scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=self.y, cmap='viridis', alpha=0.7)
            axes[i].set_title(f'{name} 降维')
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')
            
            # 保存结果
            self.results[name] = X_reduced
        
        plt.tight_layout()
        plt.show()
        
        return self.results

# 比较不同降维方法
methods = {
    'PCA': PCA(n_components=2),
    'LDA': LinearDiscriminantAnalysis(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42),
    'UMAP': UMAP(n_components=2, random_state=42)
}

comparison = DimensionalityReductionComparison(iris.data, iris.target)
results = comparison.compare_methods(methods)
```

## 第三部分：聚类与降维结合应用

### 3.1 高维数据聚类

```python
class HighDimClustering:
    def __init__(self, n_clusters=3, reduction_method='pca', n_components=2):
        self.n_clusters = n_clusters
        self.reduction_method = reduction_method
        self.n_components = n_components
    
    def fit_predict(self, X):
        # 第一步：降维
        if self.reduction_method == 'pca':
            reducer = PCA(n_components=self.n_components)
        elif self.reduction_method == 'tsne':
            reducer = TSNE(n_components=self.n_components, random_state=42)
        elif self.reduction_method == 'umap':
            reducer = UMAP(n_components=self.n_components, random_state=42)
        else:
            raise ValueError("不支持的降维方法")
        
        X_reduced = reducer.fit_transform(X)
        
        # 第二步：聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        
        self.reducer = reducer
        self.kmeans = kmeans
        self.X_reduced = X_reduced
        
        return labels
    
    def visualize(self, labels, true_labels=None):
        """可视化聚类结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 降维结果
        if true_labels is not None:
            scatter1 = ax1.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], 
                                 c=true_labels, cmap='viridis', alpha=0.7)
            ax1.set_title('真实标签')
        else:
            scatter1 = ax1.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], 
                                 cmap='viridis', alpha=0.7)
            ax1.set_title('降维结果')
        
        # 聚类结果
        scatter2 = ax2.scatter(self.X_reduced[:, 0], self.X_reduced[:, 1], 
                             c=labels, cmap='viridis', alpha=0.7)
        ax2.scatter(self.kmeans.cluster_centers_[:, 0], 
                   self.kmeans.cluster_centers_[:, 1], 
                   marker='x', s=200, linewidths=3, color='red')
        ax2.set_title('聚类结果')
        
        plt.tight_layout()
        plt.show()

# 使用示例
from sklearn.datasets import load_digits
digits = load_digits()
X_digits = digits.data

# 高维数据聚类
clusterer = HighDimClustering(n_clusters=10, reduction_method='pca', n_components=2)
labels = clusterer.fit_predict(X_digits)
clusterer.visualize(labels, digits.target)
```

### 3.2 异常检测应用

```python
class AnomalyDetection:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
    
    def pca_anomaly_detection(self, X):
        """基于PCA的异常检测"""
        # 应用PCA
        pca = PCA()
        X_pca = pca.fit_transform(X)
        
        # 计算重构误差
        X_reconstructed = pca.inverse_transform(X_pca)
        reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # 确定异常阈值
        threshold = np.percentile(reconstruction_error, 100 * (1 - self.contamination))
        anomalies = reconstruction_error > threshold
        
        return anomalies, reconstruction_error
    
    def clustering_anomaly_detection(self, X, n_clusters=5):
        """基于聚类的异常检测"""
        # 应用K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # 计算点到质心的距离
        distances = np.min(np.linalg.norm(X[:, np.newaxis] - kmeans.cluster_centers_, axis=2), axis=1)
        
        # 确定异常阈值
        threshold = np.percentile(distances, 100 * (1 - self.contamination))
        anomalies = distances > threshold
        
        return anomalies, distances

# 使用示例
ad = AnomalyDetection(contamination=0.05)
anomalies_pca, scores_pca = ad.pca_anomaly_detection(iris.data)
anomalies_cluster, scores_cluster = ad.clustering_anomaly_detection(iris.data)

print(f"PCA检测到的异常点数量: {np.sum(anomalies_pca)}")
print(f"聚类检测到的异常点数量: {np.sum(anomalies_cluster)}")
```

## 第四部分：性能优化与最佳实践

### 4.1 大数据集处理

```python
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

class BigDataProcessor:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
    
    def incremental_pca(self, X, n_components=2):
        """增量PCA处理大数据集"""
        ipca = IncrementalPCA(n_components=n_components, batch_size=self.batch_size)
        X_reduced = ipca.fit_transform(X)
        return X_reduced, ipca
    
    def minibatch_kmeans(self, X, n_clusters=5):
        """MiniBatch K-means处理大数据集"""
        mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=self.batch_size, random_state=42)
        labels = mbk.fit_predict(X)
        return labels, mbk
    
    def process_large_dataset(self, X, n_clusters=5, n_components=2):
        """完整的大数据处理流程"""
        # 增量降维
        X_reduced, ipca = self.incremental_pca(X, n_components)
        
        # MiniBatch聚类
        labels, mbk = self.minibatch_kmeans(X_reduced, n_clusters)
        
        return {
            'reduced_data': X_reduced,
            'labels': labels,
            'pca_model': ipca,
            'kmeans_model': mbk
        }

# 使用示例
# 假设X_large是一个大数据集
# processor = BigDataProcessor(batch_size=1000)
# result = processor.process_large_dataset(X_large)
```

### 4.2 参数调优指南

```python
from sklearn.model_selection import ParameterGrid

class ParameterOptimizer:
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def optimize_kmeans(self, k_range=range(2, 11)):
        """优化K-means参数"""
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X)
            
            # 计算评估指标
            silhouette = silhouette_score(self.X, labels)
            calinski = calinski_harabasz_score(self.X, labels)
            
            results.append({
                'k': k,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'inertia': kmeans.inertia_
            })
        
        return pd.DataFrame(results)
    
    def optimize_pca(self, variance_thresholds=[0.8, 0.9, 0.95, 0.99]):
        """优化PCA参数"""
        results = []
        
        for threshold in variance_thresholds:
            pca = PCA(n_components=threshold)
            X_reduced = pca.fit_transform(self.X)
            
            results.append({
                'variance_threshold': threshold,
                'n_components': pca.n_components_,
                'explained_variance': np.sum(pca.explained_variance_ratio_)
            })
        
        return pd.DataFrame(results)

# 使用示例
optimizer = ParameterOptimizer(iris.data, iris.target)
kmeans_results = optimizer.optimize_kmeans()
pca_results = optimizer.optimize_pca()

print("K-means参数优化结果:")
print(kmeans_results)
print("\nPCA参数优化结果:")
print(pca_results)
```

## 第五部分：高级聚类算法

### 5.1 谱聚类 (Spectral Clustering)

```python
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

class CustomSpectralClustering:
    def __init__(self, n_clusters=3, gamma=1.0, affinity='rbf'):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
    
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 1. 构建相似度矩阵
        if self.affinity == 'rbf':
            W = rbf_kernel(X, gamma=self.gamma)
        elif self.affinity == 'nearest_neighbors':
            # 构建k近邻图
            from sklearn.neighbors import kneighbors_graph
            W = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
            W = 0.5 * (W + W.T)  # 确保对称
        
        # 2. 计算度矩阵
        D = np.diag(np.sum(W, axis=1))
        
        # 3. 计算拉普拉斯矩阵
        L = D - W
        
        # 4. 归一化拉普拉斯矩阵
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D)))
        L_normalized = D_sqrt_inv.dot(L).dot(D_sqrt_inv)
        
        # 5. 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(L_normalized)
        
        # 6. 选择前k个特征向量
        indices = np.argsort(eigenvalues)[:self.n_clusters]
        eigenvectors = eigenvectors[:, indices]
        
        # 7. 对特征向量进行K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(eigenvectors)
        
        return self

# 使用示例
from sklearn.datasets import make_circles
X_circles, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=0)

spectral = CustomSpectralClustering(n_clusters=2, gamma=10.0)
labels = spectral.fit(X_circles).labels

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=labels, cmap='viridis')
plt.title('谱聚类结果 - 环形数据')
plt.show()
```

### 5.2 均值漂移聚类 (Mean Shift Clustering)

```python
from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity

class CustomMeanShift:
    def __init__(self, bandwidth=1.0, max_iters=100):
        self.bandwidth = bandwidth
        self.max_iters = max_iters
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化每个点作为质心
        centroids = X.copy()
        
        for _ in range(self.max_iters):
            new_centroids = np.zeros_like(centroids)
            
            for i, centroid in enumerate(centroids):
                # 计算当前质心的邻域点
                distances = np.linalg.norm(X - centroid, axis=1)
                neighbors_mask = distances < self.bandwidth
                
                if np.sum(neighbors_mask) > 0:
                    # 计算邻域点的均值
                    neighbors = X[neighbors_mask]
                    new_centroids[i] = np.mean(neighbors, axis=0)
                else:
                    new_centroids[i] = centroid
            
            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # 合并相近的质心
        unique_centroids = []
        labels = np.zeros(n_samples, dtype=int)
        
        for i, centroid in enumerate(centroids):
            assigned = False
            for j, unique_centroid in enumerate(unique_centroids):
                if np.linalg.norm(centroid - unique_centroid) < self.bandwidth:
                    labels[i] = j
                    assigned = True
                    break
            
            if not assigned:
                labels[i] = len(unique_centroids)
                unique_centroids.append(centroid)
        
        self.labels = labels
        self.cluster_centers_ = np.array(unique_centroids)
        
        return self

# 使用示例
meanshift = CustomMeanShift(bandwidth=0.8)
labels = meanshift.fit(X).labels

print(f"发现的簇数量: {len(np.unique(labels))}")
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], 
           marker='x', s=200, linewidths=3, color='red')
plt.title('均值漂移聚类结果')
plt.show()
```

### 5.3 仿射传播聚类 (Affinity Propagation)

```python
from sklearn.cluster import AffinityPropagation

class CustomAffinityPropagation:
    def __init__(self, damping=0.5, max_iters=200, convergence_iter=15):
        self.damping = damping
        self.max_iters = max_iters
        self.convergence_iter = convergence_iter
    
    def fit(self, S):
        """S是相似度矩阵"""
        n_samples = S.shape[0]
        
        # 初始化责任矩阵和可用性矩阵
        R = np.zeros((n_samples, n_samples))
        A = np.zeros((n_samples, n_samples))
        
        # 迭代更新
        for it in range(self.max_iters):
            # 保存旧的R和A用于收敛检查
            R_old = R.copy()
            A_old = A.copy()
            
            # 更新责任矩阵R
            AS = A + S
            for i in range(n_samples):
                for k in range(n_samples):
                    max_val = -np.inf
                    for k2 in range(n_samples):
                        if k2 != k:
                            max_val = max(max_val, AS[i, k2])
                    R[i, k] = S[i, k] - max_val
            
            # 阻尼更新
            R = (1 - self.damping) * R + self.damping * R_old
            
            # 更新可用性矩阵A
            for i in range(n_samples):
                for k in range(n_samples):
                    if i == k:
                        # 对角线元素
                        A[k, k] = np.sum(np.maximum(R[:, k], 0)) - R[k, k]
                    else:
                        # 非对角线元素
                        A[i, k] = min(0, R[k, k] + np.sum(np.maximum(R[:, k], 0)) - 
                                    max(0, R[i, k]) - max(0, R[k, k]))
            
            # 阻尼更新
            A = (1 - self.damping) * A + self.damping * A_old
            
            # 检查收敛
            if it > self.convergence_iter and \
               np.allclose(R, R_old, atol=1e-6) and \
               np.allclose(A, A_old, atol=1e-6):
                break
        
        # 确定代表点
        exemplars = np.argmax(A + R, axis=1)
        
        # 分配标签
        unique_exemplars = np.unique(exemplars)
        labels = np.zeros(n_samples, dtype=int)
        for i, exemplar in enumerate(unique_exemplars):
            labels[exemplars == exemplar] = i
        
        self.labels = labels
        self.cluster_centers_indices_ = unique_exemplars
        
        return self

# 使用示例
from sklearn.metrics.pairwise import euclidean_distances

# 计算负的欧氏距离作为相似度
S = -euclidean_distances(X, squared=True)

affinity = CustomAffinityPropagation(damping=0.9)
labels = affinity.fit(S).labels

print(f"发现的簇数量: {len(np.unique(labels))}")
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('仿射传播聚类结果')
plt.show()
```

## 第六部分：流形学习与非线性降维

### 6.1 局部线性嵌入 (LLE)

```python
from sklearn.manifold import LocallyLinearEmbedding

class CustomLLE:
    def __init__(self, n_components=2, n_neighbors=10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
    
    def fit_transform(self, X):
        n_samples, n_features = X.shape
        
        # 1. 寻找每个点的k近邻
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        neighbors.fit(X)
        indices = neighbors.kneighbors(X, return_distance=False)[:, 1:]
        
        # 2. 计算重构权重
        W = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # 获取邻域点
            Xi = X[indices[i]]
            
            # 中心化
            Xi_centered = Xi - X[i]
            
            # 计算局部协方差矩阵
            C = Xi_centered.dot(Xi_centered.T)
            
            # 添加正则化项避免奇异矩阵
            C += np.eye(self.n_neighbors) * 1e-3 * np.trace(C)
            
            # 求解权重
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w /= np.sum(w)
            
            W[i, indices[i]] = w
        
        # 3. 计算嵌入
        M = np.eye(n_samples) - W - W.T + W.T.dot(W)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # 选择最小的非零特征值对应的特征向量
        indices = np.argsort(eigenvalues)[1:self.n_components+1]
        Y = eigenvectors[:, indices]
        
        return Y

# 使用示例
lle = CustomLLE(n_neighbors=12)
X_lle = lle.fit_transform(iris.data)

plt.scatter(X_lle[:, 0], X_lle[:, 1], c=iris.target, cmap='viridis')
plt.title('LLE 降维结果')
plt.colorbar()
plt.show()
```

### 6.2 等距映射 (Isomap)

```python
from sklearn.manifold import Isomap

class CustomIsomap:
    def __init__(self, n_components=2, n_neighbors=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
    
    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # 1. 构建k近邻图
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
        neighbors.fit(X)
        
        # 2. 计算最短路径距离
        from scipy.sparse.csgraph import shortest_path
        from scipy.sparse import csr_matrix
        
        # 构建邻接矩阵
        distances, indices = neighbors.kneighbors(X)
        
        row = np.repeat(np.arange(n_samples), self.n_neighbors)
        col = indices.ravel()
        data = distances.ravel()
        
        graph = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))
        
        # 计算最短路径距离矩阵
        dist_matrix = shortest_path(graph, directed=False)
        
        # 3. 多维缩放
        from sklearn.manifold import MDS
        mds = MDS(n_components=self.n_components, dissimilarity='precomputed', random_state=42)
        Y = mds.fit_transform(dist_matrix)
        
        return Y

# 使用示例
isomap = CustomIsomap(n_neighbors=10)
X_isomap = isomap.fit_transform(iris.data)

plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=iris.target, cmap='viridis')
plt.title('Isomap 降维结果')
plt.colorbar()
plt.show()
```

### 6.3 拉普拉斯特征映射 (Laplacian Eigenmaps)

```python
class LaplacianEigenmaps:
    def __init__(self, n_components=2, n_neighbors=10, gamma=1.0):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.gamma = gamma
    
    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # 1. 构建相似度矩阵
        from sklearn.neighbors import kneighbors_graph
        W = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='connectivity', include_self=True)
        W = 0.5 * (W + W.T)  # 确保对称
        
        # 转换为密集矩阵并应用热核
        W = W.toarray()
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        W = np.exp(-distances**2 / (2 * self.gamma**2)) * (W > 0)
        
        # 2. 计算度矩阵和拉普拉斯矩阵
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        # 3. 计算广义特征值问题
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 4. 选择最小的非零特征值对应的特征向量
        indices = np.argsort(eigenvalues)[1:self.n_components+1]
        Y = eigenvectors[:, indices]
        
        return Y

# 使用示例
le = LaplacianEigenmaps(n_neighbors=15, gamma=0.5)
X_le = le.fit_transform(iris.data)

plt.scatter(X_le[:, 0], X_le[:, 1], c=iris.target, cmap='viridis')
plt.title('拉普拉斯特征映射结果')
plt.colorbar()
plt.show()
```

## 第七部分：聚类集成方法

### 7.1 基于共识的聚类集成

```python
class ConsensusClustering:
    def __init__(self, base_estimators, consensus_method='voting'):
        self.base_estimators = base_estimators
        self.consensus_method = consensus_method
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # 获取所有基础聚类结果
        all_labels = []
        for estimator in self.base_estimators:
            labels = estimator.fit_predict(X)
            all_labels.append(labels)
        
        all_labels = np.array(all_labels)
        
        if self.consensus_method == 'voting':
            # 简单投票
            final_labels = []
            for i in range(n_samples):
                votes = all_labels[:, i]
                final_labels.append(np.bincount(votes).argmax())
            
            return np.array(final_labels)
        
        elif self.consensus_method == 'coassociation':
            # 共现矩阵方法
            coassociation_matrix = np.zeros((n_samples, n_samples))
            
            for labels in all_labels:
                for i in range(n_samples):
                    for j in range(n_samples):
                        if labels[i] == labels[j]:
                            coassociation_matrix[i, j] += 1
            
            coassociation_matrix /= len(all_labels)
            
            # 使用层次聚类
            from scipy.cluster.hierarchy import linkage, fcluster
            linkage_matrix = linkage(1 - coassociation_matrix, method='average')
            n_clusters = len(np.unique(all_labels[0]))
            final_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            return final_labels - 1

# 使用示例
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# 创建基础聚类器
estimators = [
    KMeans(n_clusters=3, random_state=0),
    AgglomerativeClustering(n_clusters=3),
    DBSCAN(eps=0.5, min_samples=5)
]

consensus = ConsensusClustering(estimators, consensus_method='voting')
labels_ensemble = consensus.fit_predict(iris.data)

print(f"集成聚类发现的簇数量: {len(np.unique(labels_ensemble))}")
```

### 7.2 基于图割的聚类集成

```python
class GraphBasedClusteringEnsemble:
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # 构建共识图
        consensus_graph = np.zeros((n_samples, n_samples))
        
        for estimator in self.base_estimators:
            labels = estimator.fit_predict(X)
            
            # 构建邻接矩阵
            adjacency = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        adjacency[i, j] = 1
            
            consensus_graph += adjacency
        
        consensus_graph /= len(self.base_estimators)
        
        # 应用谱聚类
        spectral = CustomSpectralClustering(n_clusters=3)
        labels = spectral.fit(consensus_graph).labels
        
        return labels

# 使用示例
graph_ensemble = GraphBasedClusteringEnsemble(estimators)
labels_graph = graph_ensemble.fit_predict(iris.data)

print(f"图基集成聚类发现的簇数量: {len(np.unique(labels_graph))}")
```

## 第八部分：性能基准测试

### 8.1 聚类算法性能比较

```python
import time
from sklearn.datasets import make_blobs, make_moons, make_circles

class ClusteringBenchmark:
    def __init__(self):
        self.datasets = {}
        self.algorithms = {}
    
    def generate_datasets(self):
        """生成测试数据集"""
        n_samples = 1000
        
        # 球形数据
        X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=3, 
                                     cluster_std=0.60, random_state=0)
        
        # 半月形数据
        X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
        
        # 环形数据
        X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.05, 
                                           factor=0.5, random_state=0)
        
        self.datasets = {
            'blobs': (X_blobs, y_blobs),
            'moons': (X_moons, y_moons),
            'circles': (X_circles, y_circles)
        }
    
    def register_algorithms(self):
        """注册测试算法"""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
        
        self.algorithms = {
            'KMeans': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=3),
            'Spectral': SpectralClustering(n_clusters=3, random_state=42)
        }
    
    def benchmark(self):
        """运行性能测试"""
        results = []
        
        for dataset_name, (X, y_true) in self.datasets.items():
            for algo_name, algorithm in self.algorithms.items():
                # 计时
                start_time = time.time()
                
                try:
                    labels = algorithm.fit_predict(X)
                    exec_time = time.time() - start_time
                    
                    # 计算评估指标
                    from sklearn.metrics import silhouette_score, adjusted_rand_score
                    
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(X, labels)
                    else:
                        silhouette = -1
                    
                    ari = adjusted_rand_score(y_true, labels)
                    
                    n_clusters = len(np.unique(labels))
                    
                    results.append({
                        'dataset': dataset_name,
                        'algorithm': algo_name,
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette,
                        'adjusted_rand_score': ari,
                        'execution_time': exec_time
                    })
                except Exception as e:
                    print(f"算法 {algo_name} 在数据集 {dataset_name} 上失败: {e}")
        
        return pd.DataFrame(results)

# 运行基准测试
benchmark = ClusteringBenchmark()
benchmark.generate_datasets()
benchmark.register_algorithms()
results = benchmark.benchmark()

print("聚类算法性能比较:")
print(results.pivot(index='algorithm', columns='dataset', values='silhouette_score'))
```

### 8.2 降维算法性能比较

```python
class DimensionalityReductionBenchmark:
    def __init__(self):
        self.datasets = {}
        self.methods = {}
    
    def generate_datasets(self):
        """生成测试数据集"""
        from sklearn.datasets import load_digits, fetch_olivetti_faces
        
        # 手写数字数据集
        digits = load_digits()
        X_digits = digits.data
        y_digits = digits.target
        
        # 人脸数据集
        faces = fetch_olivetti_faces()
        X_faces = faces.data
        y_faces = faces.target
        
        self.datasets = {
            'digits': (X_digits, y_digits),
            'faces': (X_faces, y_faces)
        }
    
    def register_methods(self):
        """注册降维方法"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
        
        self.methods = {
            'PCA': PCA(n_components=2),
            't-SNE': TSNE(n_components=2, random_state=42),
            'Isomap': Isomap(n_components=2),
            'LLE': LocallyLinearEmbedding(n_components=2)
        }
    
    def benchmark(self):
        """运行性能测试"""
        results = []
        
        for dataset_name, (X, y_true) in self.datasets.items():
            for method_name, method in self.methods.items():
                # 计时
                start_time = time.time()
                
                try:
                    X_reduced = method.fit_transform(X)
                    exec_time = time.time() - start_time
                    
                    # 计算评估指标
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.model_selection import cross_val_score
                    
                    # 使用KNN分类器评估降维效果
                    knn = KNeighborsClassifier(n_neighbors=3)
                    accuracy = cross_val_score(knn, X_reduced, y_true, cv=5).mean()
                    
                    results.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'accuracy': accuracy,
                        'execution_time': exec_time,
                        'n_components': X_reduced.shape[1]
                    })
                except Exception as e:
                    print(f"方法 {method_name} 在数据集 {dataset_name} 上失败: {e}")
        
        return pd.DataFrame(results)

# 运行基准测试
dr_benchmark = DimensionalityReductionBenchmark()
dr_benchmark.generate_datasets()
dr_benchmark.register_methods()
dr_results = dr_benchmark.benchmark()

print("降维算法性能比较:")
print(dr_results.pivot(index='method', columns='dataset', values='accuracy'))
```

## 总结

本指南全面介绍了无监督学习中的两大核心技术：聚类分析和降维技术。通过详细的算法原理、代码实现和实际应用案例，帮助读者深入理解这些技术并在实际项目中有效应用。

### 关键要点：

1. **聚类算法选择**：根据数据特性和业务需求选择合适的聚类方法
2. **降维技术应用**：合理使用降维技术处理高维数据和可视化
3. **参数调优**：通过系统的方法优化算法参数
4. **性能优化**：针对大数据集采用增量学习和并行处理
5. **结果评估**：使用多种指标综合评估聚类和降维效果

### 进一步学习方向：

1. 深度聚类方法
2. 自编码器在降维中的应用
3. 流数据聚类技术
4. 多视图聚类方法
5. 可解释性聚类分析

## 第九部分：生产部署与最佳实践

### 9.1 生产环境部署指南

#### 9.1.1 模型序列化与持久化

```python
import pickle
import joblib
from sklearn.pipeline import Pipeline

class ProductionClusteringSystem:
    def __init__(self):
        self.pipeline = None
        self.scaler = None
        self.model = None
    
    def build_pipeline(self, X, algorithm='kmeans', n_clusters=3):
        """构建完整的预处理和聚类管道"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 选择聚类算法
        if algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        # 训练模型
        self.model.fit(X_scaled)
        
        # 构建管道
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('cluster', self.model)
        ])
    
    def save_model(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'model_info': {
                    'algorithm': type(self.model).__name__,
                    'n_clusters': len(np.unique(self.model.labels_)) if hasattr(self.model, 'labels_') else None
                }
            }, f)
    
    def load_model(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.pipeline = saved_data['pipeline']
            self.model = self.pipeline.named_steps['cluster']
    
    def predict(self, X):
        """预测新数据"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """预测概率（如果支持）"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # 对于不支持概率的算法，返回硬标签
            labels = self.predict(X)
            n_clusters = len(np.unique(labels))
            proba = np.eye(n_clusters)[labels]
            return proba

# 使用示例
production_system = ProductionClusteringSystem()
production_system.build_pipeline(iris.data, algorithm='kmeans', n_clusters=3)

# 保存模型
production_system.save_model('clustering_model.pkl')

# 加载模型
new_system = ProductionClusteringSystem()
new_system.load_model('clustering_model.pkl')

# 预测新数据
new_labels = new_system.predict(iris.data)
```

#### 9.1.2 API服务部署

```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

class ClusteringAPI:
    def __init__(self, model_path):
        self.model = ProductionClusteringSystem()
        self.model.load_model(model_path)
    
    def predict_batch(self, X):
        """批量预测"""
        labels = self.model.predict(X)
        
        # 计算每个簇的统计信息
        results = []
        for label in np.unique(labels):
            cluster_data = X[labels == label]
            results.append({
                'cluster_id': int(label),
                'size': len(cluster_data),
                'centroid': cluster_data.mean(axis=0).tolist(),
                'std': cluster_data.std(axis=0).tolist()
            })
        
        return {
            'labels': labels.tolist(),
            'clusters': results,
            'n_clusters': len(np.unique(labels))
        }

# 初始化API
api = ClusteringAPI('clustering_model.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': str(np.datetime64('now'))})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 验证输入数据
        if 'data' not in data:
            return jsonify({'error': 'Missing data field'}), 400
        
        X = np.array(data['data'])
        
        # 验证数据形状
        if len(X.shape) != 2:
            return jsonify({'error': 'Data must be 2D array'}), 400
        
        # 预测
        result = api.predict_batch(X)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cluster_info', methods=['GET'])
def cluster_info():
    """获取聚类模型信息"""
    model_info = {
        'algorithm': type(api.model.model).__name__,
        'n_features': api.model.model.n_features_in_ if hasattr(api.model.model, 'n_features_in_') else 'unknown',
        'n_clusters': len(np.unique(api.model.model.labels_)) if hasattr(api.model.model, 'labels_') else 'unknown'
    }
    return jsonify(model_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 9.2 监控与维护

#### 9.2.1 模型性能监控

```python
import logging
from datetime import datetime

class ClusteringMonitor:
    def __init__(self, model, threshold=0.1):
        self.model = model
        self.threshold = threshold
        self.performance_history = []
        self.logger = logging.getLogger('ClusteringMonitor')
    
    def monitor_performance(self, X, y_true=None):
        """监控模型性能"""
        # 预测
        labels = self.model.predict(X)
        
        # 计算性能指标
        from sklearn.metrics import silhouette_score
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        # 检测性能下降
        if len(self.performance_history) > 0:
            last_score = self.performance_history[-1]['silhouette_score']
            if abs(silhouette - last_score) > self.threshold:
                self.logger.warning(f"性能显著变化: {last_score:.3f} -> {silhouette:.3f}")
        
        # 记录性能
        performance_record = {
            'timestamp': datetime.now(),
            'silhouette_score': silhouette,
            'n_clusters': len(np.unique(labels)),
            'n_samples': len(X)
        }
        
        self.performance_history.append(performance_record)
        
        return performance_record
    
    def detect_concept_drift(self, window_size=10):
        """检测概念漂移"""
        if len(self.performance_history) < window_size:
            return False
        
        recent_scores = [record['silhouette_score'] for record in self.performance_history[-window_size:]]
        older_scores = [record['silhouette_score'] for record in self.performance_history[-2*window_size:-window_size]]
        
        if len(older_scores) < window_size:
            return False
        
        # 使用t检验检测显著差异
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(recent_scores, older_scores)
        
        if p_value < 0.05:
            self.logger.warning(f"检测到概念漂移 (p-value: {p_value:.4f})")
            return True
        
        return False
    
    def generate_report(self):
        """生成监控报告"""
        if not self.performance_history:
            return "无性能数据"
        
        recent_scores = [record['silhouette_score'] for record in self.performance_history[-10:]]
        
        report = {
            'current_performance': self.performance_history[-1],
            'avg_performance_last_10': np.mean(recent_scores),
            'performance_trend': '稳定' if np.std(recent_scores) < 0.05 else '波动',
            'concept_drift_detected': self.detect_concept_drift()
        }
        
        return report

# 使用示例
monitor = ClusteringMonitor(production_system)
performance = monitor.monitor_performance(iris.data)
report = monitor.generate_report()

print("监控报告:")
print(report)
```

#### 9.2.2 自动重训练机制

```python
class AutoRetrainingSystem:
    def __init__(self, model, retrain_interval=24, performance_threshold=0.1):
        self.model = model
        self.retrain_interval = retrain_interval  # 小时
        self.performance_threshold = performance_threshold
        self.last_retrain_time = datetime.now()
        self.monitor = ClusteringMonitor(model)
    
    def should_retrain(self, X):
        """判断是否需要重训练"""
        current_time = datetime.now()
        hours_since_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
        
        # 检查时间间隔
        if hours_since_retrain >= self.retrain_interval:
            return True
        
        # 检查性能下降
        performance = self.monitor.monitor_performance(X)
        if performance['silhouette_score'] < self.performance_threshold:
            return True
        
        # 检查概念漂移
        if self.monitor.detect_concept_drift():
            return True
        
        return False
    
    def retrain_model(self, X, algorithm='kmeans', n_clusters=None):
        """重训练模型"""
        if n_clusters is None:
            # 自动确定最佳簇数量
            from sklearn.metrics import silhouette_score
            
            best_score = -1
            best_k = 2
            
            for k in range(2, min(10, len(X))):
                temp_model = ProductionClusteringSystem()
                temp_model.build_pipeline(X, algorithm=algorithm, n_clusters=k)
                labels = temp_model.predict(X)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            n_clusters = best_k
        
        # 重训练
        self.model.build_pipeline(X, algorithm=algorithm, n_clusters=n_clusters)
        self.last_retrain_time = datetime.now()
        
        return n_clusters

# 使用示例
auto_trainer = AutoRetrainingSystem(production_system)

if auto_trainer.should_retrain(iris.data):
    print("检测到需要重训练")
    optimal_k = auto_trainer.retrain_model(iris.data)
    print(f"自动选择的最佳簇数量: {optimal_k}")
```

### 9.3 故障排除与最佳实践

#### 9.3.1 常见问题解决方案

```python
class ClusteringTroubleshooter:
    def __init__(self):
        self.common_issues = {
            'all_points_one_cluster': "尝试调整参数或使用不同算法",
            'too_many_clusters': "减少簇数量或调整密度参数",
            'poor_silhouette_score': "检查数据质量，尝试特征工程",
            'high_memory_usage': "使用增量学习或采样方法",
            'slow_convergence': "调整学习率或使用更快的算法"
        }
    
    def diagnose_issue(self, X, labels, algorithm_name):
        """诊断聚类问题"""
        issues = []
        
        n_clusters = len(np.unique(labels))
        n_samples = len(X)
        
        # 检查簇数量问题
        if n_clusters == 1:
            issues.append('all_points_one_cluster')
        elif n_clusters > n_samples * 0.1:  # 簇数量超过样本数的10%
            issues.append('too_many_clusters')
        
        # 检查性能问题
        from sklearn.metrics import silhouette_score
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            if silhouette < 0.2:
                issues.append('poor_silhouette_score')
        
        # 检查算法特定问题
        if algorithm_name == 'DBSCAN' and n_clusters == 0:
            issues.append('all_points_noise')
        
        return issues
    
    def suggest_solutions(self, issues):
        """提供解决方案建议"""
        solutions = {}
        
        for issue in issues:
            if issue in self.common_issues:
                solutions[issue] = self.common_issues[issue]
            else:
                solutions[issue] = "需要进一步分析"
        
        return solutions
    
    def optimize_parameters(self, X, algorithm_name):
        """参数优化建议"""
        if algorithm_name == 'KMeans':
            return self._optimize_kmeans(X)
        elif algorithm_name == 'DBSCAN':
            return self._optimize_dbscan(X)
        else:
            return {}
    
    def _optimize_kmeans(self, X):
        """KMeans参数优化"""
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_score = -1
        
        for k in range(2, min(11, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        return {'n_clusters': best_k}
    
    def _optimize_dbscan(self, X):
        """DBSCAN参数优化"""
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        distances = np.sort(distances[:, 4], axis=0)
        
        # 使用肘部法则确定eps
        from kneed import KneeLocator
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        
        optimal_eps = distances[kneedle.knee] if kneedle.knee else distances[int(len(distances)*0.02)]
        
        return {'eps': optimal_eps, 'min_samples': 5}

# 使用示例
troubleshooter = ClusteringTroubleshooter()

# 诊断问题
labels = production_system.predict(iris.data)
issues = troubleshooter.diagnose_issue(iris.data, labels, 'KMeans')

if issues:
    print("检测到的问题:", issues)
    solutions = troubleshooter.suggest_solutions(issues)
    print("解决方案建议:", solutions)
    
    # 参数优化建议
    optimized_params = troubleshooter.optimize_parameters(iris.data, 'KMeans')
    print("优化参数建议:", optimized_params)
```

#### 9.3.2 最佳实践检查清单

```python
class BestPracticesChecklist:
    def __init__(self):
        self.checklist = [
            {
                'item': '数据预处理',
                'checks': [
                    '数据已标准化/归一化',
                    '处理了缺失值',
                    '处理了异常值',
                    '特征工程已完成'
                ]
            },
            {
                'item': '算法选择',
                'checks': [
                    '算法适合数据类型',
                    '考虑了数据规模',
                    '评估了多种算法',
                    '选择了最合适的算法'
                ]
            },
            {
                'item': '参数调优',
                'checks': [
                    '使用了交叉验证',
                    '参数范围合理',
                    '评估了多个指标',
                    '选择了最优参数'
                ]
            },
            {
                'item': '结果验证',
                'checks': [
                    '使用了内部评估指标',
                    '使用了外部评估指标（如果有标签）',
                    '进行了可视化检查',
                    '业务验证通过'
                ]
            },
            {
                'item': '部署准备',
                'checks': [
                    '模型已序列化',
                    '有监控机制',
                    '有重训练计划',
                    '有故障恢复方案'
                ]
            }
        ]
    
    def run_checklist(self, project_data):
        """运行检查清单"""
        results = {}
        
        for category in self.checklist:
            category_name = category['item']
            results[category_name] = {}
            
            for check in category['checks']:
                # 这里可以根据具体项目数据实现检查逻辑
                results[category_name][check] = '待检查'  # 实际项目中应实现具体检查逻辑
        
        return results
    
    def generate_report(self, results):
        """生成检查报告"""
        report = "最佳实践检查报告\n" + "="*50 + "\n"
        
        total_checks = 0
        passed_checks = 0
        
        for category, checks in results.items():
            report += f"\n{category}:\n"
            
            for check, status in checks.items():
                total_checks += 1
                if status == '通过':
                    passed_checks += 1
                    report += f"  ✓ {check}\n"
                else:
                    report += f"  ✗ {check} - {status}\n"
        
        score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        report += f"\n总体得分: {score:.1f}% ({passed_checks}/{total_checks})"
        
        return report

# 使用示例
checklist = BestPracticesChecklist()
results = checklist.run_checklist({})  # 传入项目数据
report = checklist.generate_report(results)
print(report)
```

## 第十部分：快速参考与总结

### 10.1 算法选择快速指南

| 数据类型 | 推荐算法 | 关键参数 | 适用场景 |
|---------|---------|----------|----------|
| 球形簇，大小相近 | K-means | n_clusters | 大数据集，客户细分 |
| 任意形状簇 | DBSCAN | eps, min_samples | 异常检测，图像分割 |
| 层次结构数据 | 层次聚类 | n_clusters, linkage | 小数据集，生物信息学 |
| 高维数据 | 谱聚类 | n_clusters, gamma | 复杂形状，图数据 |
| 流数据 | MiniBatch K-means | batch_size | 实时处理，大数据 |
| 概率模型 | GMM | n_components | 混合分布，软聚类 |

### 10.2 降维方法选择指南

| 数据特性 | 推荐方法 | 关键参数 | 主要优势 |
|---------|---------|----------|----------|
| 线性关系 | PCA | n_components | 计算效率高，可解释性强 |
| 分类任务 | LDA | n_components | 最大化类间分离 |
| 非线性结构 | t-SNE | perplexity | 可视化效果好 |
| 大规模数据 | UMAP | n_neighbors | 速度快，质量高 |
| 流形学习 | LLE | n_neighbors | 保持局部结构 |
| 图数据 | 拉普拉斯特征映射 | n_neighbors | 基于图理论 |

### 10.3 性能优化技巧总结

1. **数据预处理**
   - 标准化数值特征
   - 处理缺失值和异常值
   - 特征选择减少维度

2. **算法优化**
   - 使用增量学习处理大数据
   - 并行计算加速训练
   - 缓存中间结果

3. **参数调优**
   - 网格搜索找到最优参数
   - 交叉验证评估泛化能力
   - 早停法避免过拟合

4. **部署优化**
   - 模型序列化减少加载时间
   - 批量预测提高吞吐量
   - 监控系统实时检测问题

### 10.4 常见陷阱与避免方法

1. **维度灾难**
   - 问题：高维数据导致距离计算失效
   - 解决：使用降维技术或专门的高维算法

2. **初始值敏感性**
   - 问题：K-means等算法对初始值敏感
   - 解决：多次运行取最优结果或使用K-means++

3. **簇数量选择**
   - 问题：难以确定最佳簇数量
   - 解决：使用肘部法则、轮廓系数等方法

4. **数据尺度问题**
   - 问题：不同尺度的特征影响距离计算
   - 解决：标准化或归一化数据

### 10.5 未来发展趋势

1. **深度聚类**：结合深度学习的聚类方法
2. **可解释性聚类**：提供可解释的聚类结果
3. **在线聚类**：实时处理流数据
4. **多模态聚类**：处理多种类型的数据
5. **联邦聚类**：保护隐私的分布式聚类

## 结论

本指南全面涵盖了无监督学习中聚类和降维两大核心技术的理论原理、算法实现、实践应用和部署运维。通过系统学习这些内容，读者可以：

1. **深入理解算法原理**：掌握各种聚类和降维算法的数学基础
2. **熟练实现算法**：能够用Python实现主要算法
3. **有效应用技术**：根据实际问题选择合适的算法和方法
4. **优化系统性能**：掌握性能优化和故障排除技巧
5. **部署生产系统**：了解如何将算法部署到生产环境

无监督学习是一个快速发展的领域，新的算法和技术不断涌现。建议读者保持学习，关注最新研究进展，并在实际项目中不断实践和总结经验。

### 实际应用建议：

1. **数据预处理**：确保数据质量，进行适当的标准化和清洗
2. **算法选择**：根据数据规模、特征数量和业务目标选择合适算法
3. **参数调优**：使用交叉验证和网格搜索优化参数
4. **结果验证**：结合业务知识和统计指标验证结果合理性
5. **持续监控**：在生产环境中持续监控模型性能

希望本指南能为您的无监督学习之旅提供有力的支持！