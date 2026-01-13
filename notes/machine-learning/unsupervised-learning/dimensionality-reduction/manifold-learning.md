# 流形学习

## 概述

非线性降维技术，假设数据位于低维流形上。能够发现数据的内在几何结构。

## 核心思想

流形学习假设高维数据实际上位于一个低维流形上，通过保持局部几何关系来实现降维。

## 主要算法

### 等距映射（ISOMAP）

#### 算法步骤
1. **构建邻域图**：使用k近邻或ε邻域确定每个点的邻域
2. **计算测地距离**：使用Dijkstra算法计算图中点的最短路径距离
3. **多维缩放**：将测地距离矩阵映射到低维空间

#### 数学原理
测地距离矩阵$D_G$，通过MDS求解：
$B = -\frac{1}{2}HD_GH$
其中$H = I - \frac{1}{n}11^T$是中心化矩阵。

### 局部线性嵌入（LLE）

#### 算法步骤
1. **寻找近邻**：为每个点找到k个最近邻
2. **重构权重**：最小化局部重构误差
3. **低维嵌入**：保持重构权重不变的降维

#### 数学原理
重构权重优化：
$\min_W \sum_i \|x_i - \sum_j W_{ij}x_j\|^2$
低维嵌入优化：
$\min_Y \sum_i \|y_i - \sum_j W_{ij}y_j\|^2$

### t-SNE（t-分布随机邻域嵌入）

#### 算法步骤
1. **计算相似度**：高维空间使用高斯分布，低维空间使用t分布
2. **最小化KL散度**：优化高低维空间分布的相似性

#### 数学原理
高维相似度：
$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k≠i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$

低维相似度：
$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k≠l}(1 + \|y_k - y_l\|^2)^{-1}}$

目标函数（KL散度）：
$C = KL(P\|Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$

## Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.decomposition import PCA

def manifold_learning_demo():
    """
    流形学习算法演示
    """
    # 生成瑞士卷数据集
    n_samples = 1000
    X, color = datasets.make_swiss_roll(n_samples, random_state=42)
    
    # 使用不同流形学习算法
    methods = [
        ('PCA', PCA(n_components=2)),
        ('Isomap', manifold.Isomap(n_components=2)),
        ('LLE', manifold.LocallyLinearEmbedding(n_components=2)),
        ('t-SNE', manifold.TSNE(n_components=2, random_state=42))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, method) in enumerate(methods):
        # 计算降维结果
        Y = method.fit_transform(X)
        
        # 可视化
        scatter = axes[i].scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        axes[i].set_title(f'{name} Projection')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def isomap_implementation(X, n_neighbors=10, n_components=2):
    """
    手动实现ISOMAP算法
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import shortest_path
    
    n_samples = X.shape[0]
    
    # 1. 构建邻域图
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 2. 构建距离矩阵
    graph = np.full((n_samples, n_samples), np.inf)
    for i in range(n_samples):
        for j, idx in enumerate(indices[i]):
            if i != idx:
                graph[i, idx] = distances[i, j]
    
    # 3. 计算测地距离（最短路径）
    geodesic_distances = shortest_path(graph, directed=False)
    
    # 4. 多维缩放（MDS）
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(geodesic_distances)
    
    return embedding

def lle_implementation(X, n_neighbors=10, n_components=2):
    """
    手动实现LLE算法
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples, n_features = X.shape
    
    # 1. 寻找近邻
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 删除自身（第一个最近邻）
    indices = indices[:, 1:]
    
    # 2. 计算重构权重
    W = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        # 获取近邻
        neighbors = indices[i]
        
        # 计算局部协方差矩阵
        Z = X[neighbors] - X[i]
        C = Z @ Z.T
        
        # 添加正则化项避免奇异矩阵
        C += np.eye(n_neighbors) * 1e-3 * np.trace(C)
        
        # 求解权重（最小二乘）
        w = np.linalg.solve(C, np.ones(n_neighbors))
        w /= np.sum(w)
        
        # 存储权重
        W[i, neighbors] = w
    
    # 3. 计算低维嵌入
    M = np.eye(n_samples) - W - W.T + W.T @ W
    
    # 求解特征值问题
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # 选择最小的非零特征值对应的特征向量
    embedding = eigenvectors[:, 1:n_components + 1]
    
    return embedding

# 示例使用
if __name__ == "__main__":
    # 生成S曲线数据
    X, color = datasets.make_s_curve(300, random_state=42)
    
    # 比较不同算法
    print("比较流形学习算法效果...")
    
    # 手动实现ISOMAP
    iso_embedding = isomap_implementation(X)
    
    # 手动实现LLE
    lle_embedding = lle_implementation(X)
    
    # 可视化比较
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(iso_embedding[:, 0], iso_embedding[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.set_title('Manual ISOMAP Implementation')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    
    ax2.scatter(lle_embedding[:, 0], lle_embedding[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title('Manual LLE Implementation')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.show()
    
    # 完整演示
    manifold_learning_demo()
```

## 数学基础

### 流形定义

流形是局部类似于欧几里得空间的拓扑空间。对于每个点p∈M，存在邻域U⊂M同胚于Rⁿ。

### 测地距离

流形上两点之间的最短路径长度，比欧氏距离更能反映数据的真实结构。

### 局部线性性

流形在局部可以近似为线性空间，这是LLE算法的基础。

## 算法比较

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| ISOMAP | 保持全局几何结构 | 计算复杂度高 | 具有明确流形结构的数据 |
| LLE | 计算相对简单 | 对参数敏感 | 局部线性结构明显的数据 |
| t-SNE | 可视化效果好 | 计算复杂，结果不稳定 | 高维数据可视化 |

## 参数选择

### 邻域大小（k）
- 太小：无法捕捉流形结构
- 太大：破坏局部几何关系
- 通常选择5-20之间

### 学习率（t-SNE）
- 控制优化过程的步长
- 通常使用默认值或自动调整

## 应用场景
- 高维数据可视化
- 图像处理
- 生物信息学
- 自然语言处理

## 局限性
- 对噪声敏感
- 参数选择困难
- 计算复杂度高
- 缺乏理论保证