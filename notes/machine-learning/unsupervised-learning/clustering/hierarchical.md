# 层次聚类

## 概述

通过构建树状结构（树状图）来组织数据，形成不同层次的聚类。

## 算法类型

### 聚合式（自底向上）
- 每个点初始为一个簇
- 逐步合并最相似的簇
- 直到所有点合并为一个簇

### 分裂式（自顶向下）
- 所有点初始为一个簇
- 递归地分裂簇
- 直到每个点为一个簇

## 距离度量

### 簇间距离
- **单链接**：最近邻距离
- **全链接**：最远邻距离
- **平均链接**：平均距离
- **质心法**：质心间距离

## 算法步骤

1. 计算所有点对的距离
2. 将每个点视为一个簇
3. 找到距离最近的两个簇并合并
4. 更新距离矩阵
5. 重复步骤3-4直到所有点合并

## Python实现

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def hierarchical_clustering(X, method='ward', metric='euclidean'):
    """
    层次聚类实现
    """
    # 计算距离矩阵
    distance_matrix = pdist(X, metric=metric)
    
    # 层次聚类
    Z = linkage(distance_matrix, method=method)
    
    # 绘制树状图
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
    return Z

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    # 执行层次聚类
    linkage_matrix = hierarchical_clustering(X)
    
    # 获取不同层次的分割
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
    print(f"聚类结果: {np.unique(clusters)}")
```

## 数学分析

### 距离更新公式

对于合并后的新簇Cₖ = Cᵢ ∪ Cⱼ，与其他簇Cₗ的距离：

**单链接**：$d(C_k, C_l) = min(d(C_i, C_l), d(C_j, C_l))$

**全链接**：$d(C_k, C_l) = max(d(C_i, C_l), d(C_j, C_l))$

**平均链接**：$d(C_k, C_l) = \frac{|C_i|d(C_i, C_l) + |C_j|d(C_j, C_l)}{|C_i| + |C_j|}$

**质心法**：$d(C_k, C_l) = \|\mathbf{c}_k - \mathbf{c}_l\|$

其中$\mathbf{c}_k$是簇Cₖ的质心。

## 复杂度分析

- **时间复杂度**：O(n³) 或 O(n² log n)（优化后）
- **空间复杂度**：O(n²)

## 优点
- 不需要预先指定簇数
- 可视化效果好（树状图）
- 可以发现任意形状的簇

## 缺点
- 时间复杂度高
- 对噪声敏感
- 合并/分裂决策不可逆

## 应用场景
- 生物信息学（基因表达分析）
- 文档聚类
- 市场细分
- 图像分割