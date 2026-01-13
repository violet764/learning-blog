# DBSCAN密度聚类

## 概述

基于密度的聚类算法，能发现任意形状的簇，对噪声鲁棒。不需要预先指定簇数。

## 核心概念

### 邻域定义
- **ε-邻域**：以点p为中心，半径为ε的圆形区域
- **直接密度可达**：点q在点p的ε-邻域内，且p是核心点
- **密度可达**：存在点链p₁,p₂,...,pₙ，使得每个点都是前一个点的直接密度可达
- **密度相连**：存在点o，使得p和q都从o密度可达

### 点类型
- **核心点**：ε-邻域内至少包含MinPts个点
- **边界点**：在核心点的ε-邻域内，但自身不是核心点
- **噪声点**：既非核心点也非边界点

## 算法步骤

1. 标记所有点为核心点、边界点或噪声点
2. 删除噪声点
3. 为核心点之间距离在ε范围内的点添加边
4. 每个连通分量形成一个簇

## Python实现

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    DBSCAN聚类实现
    """
    # 创建DBSCAN模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # 拟合数据
    labels = dbscan.fit_predict(X)
    
    # 统计结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f'簇数量: {n_clusters}')
    print(f'噪声点数量: {n_noise}')
    
    # 可视化结果
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(10, 8))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用黑色
            col = [0, 0, 0, 1]
        
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title(f'DBSCAN聚类结果 (eps={eps}, min_samples={min_samples})')
    plt.show()
    
    return labels

def find_optimal_eps(X, k=5):
    """
    使用k距离图确定最优eps参数
    """
    # 计算每个点到第k个最近邻的距离
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # 按距离排序
    distances = np.sort(distances[:, k-1], axis=0)
    
    # 绘制k距离图
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'k距离图 (k={k})')
    plt.xlabel('数据点索引')
    plt.ylabel(f'第{k}个最近邻距离')
    plt.grid(True)
    plt.show()
    
    return distances

# 示例使用
if __name__ == "__main__":
    # 生成示例数据（包含噪声和不同密度的簇）
    np.random.seed(42)
    
    # 第一个密集簇
    cluster1 = np.random.randn(100, 2) * 0.1 + [2, 2]
    
    # 第二个稀疏簇
    cluster2 = np.random.randn(50, 2) * 0.3 + [-2, -2]
    
    # 噪声点
    noise = np.random.randn(20, 2) * 2
    
    X = np.vstack([cluster1, cluster2, noise])
    
    # 确定最优eps参数
    distances = find_optimal_eps(X, k=5)
    
    # 执行DBSCAN聚类
    labels = dbscan_clustering(X, eps=0.3, min_samples=5)
```

## 数学分析

### 密度定义

对于点p，其ε-邻域定义为：

$N_ε(p) = \{q \in D \mid d(p,q) ≤ ε\}$

其中d(p,q)是点p和q之间的距离。

### 核心点判定

点p是核心点当且仅当：

$|N_ε(p)| ≥ MinPts$

### 密度可达性

点p密度可达于点q，如果存在点链p₁,p₂,...,pₙ，使得：
- p₁ = p, pₙ = q
- pᵢ₊₁ ∈ N_ε(pᵢ)
- pᵢ是核心点（对于i=1到n-1）

## 参数选择方法

### ε参数选择
- **k距离图法**：计算每个点到第k个最近邻的距离，选择拐点处的距离值
- **经验法则**：通常选择数据维度+1作为MinPts

### MinPts参数选择
- **经验值**：通常设置为2×数据维度
- **领域知识**：根据具体应用调整

## 复杂度分析

- **时间复杂度**：O(n log n)（使用空间索引），O(n²)（朴素的）
- **空间复杂度**：O(n)

## 优点
- 能发现任意形状的簇
- 对噪声鲁棒
- 不需要指定簇数
- 能识别噪声点

## 缺点
- 对参数敏感
- 处理不同密度的簇效果差
- 高维数据效果不佳

## 应用场景
- 异常检测
- 图像分割
- 空间数据聚类
- 网络入侵检测