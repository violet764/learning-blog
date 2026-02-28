# 高级数据结构

在机器学习中，高效的**空间数据结构**对于加速查询操作至关重要。本章介绍 KD 树及其相关数据结构，它们在最近邻搜索、范围查询等任务中有着广泛应用。

## 核心问题

考虑以下常见场景：

- **最近邻搜索**：给定查询点，找到数据集中距离最近的点
- **K近邻搜索**：找到距离最近的 K 个点
- **范围查询**：找到位于某区域内的所有点

**朴素方法**：遍历所有点，时间复杂度 $O(n)$

**目标**：构建索引结构，将查询复杂度降到 $O(\log n)$

---

## KD 树

KD 树（k-dimensional tree）是一种用于组织 k 维空间中点的**空间分割数据结构**。

### 核心思想

通过递归地将空间划分为超矩形区域，每个区域包含大致相同数量的点，从而实现高效查询。

### 树结构

- 每个节点代表一个 k 维空间中的点
- 每个非叶节点生成一个**分割超平面**，将空间划分为两个子空间
- 分割超平面**垂直于坐标轴**

### 构建算法

**算法步骤**：

1. 选择分割维度 $d$（通常选择方差最大的维度）
2. 选择分割值 $v$（通常选择该维度的中位数）
3. 创建节点，存储分割点和分割维度
4. 递归构建左子树（$x_d < v$ 的点）和右子树（$x_d \geq v$ 的点）

### 维度选择策略

| 策略 | 方法 | 特点 |
|------|------|------|
| 轮换法 | 第 $i$ 层使用维度 $d = i \mod k$ | 简单，均匀分割 |
| 方差最大法 | 选择方差最大的维度 | 分割效果更好 |
| 空间最大法 | 选择空间跨度最大的维度 | 平衡空间分割 |

### 代码实现

```python
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class KDNode:
    """KD 树节点"""
    def __init__(self, point, dimension, left=None, right=None):
        self.point = point          # 数据点
        self.dimension = dimension  # 分割维度
        self.left = left            # 左子树
        self.right = right          # 右子树


class KDTree:
    """KD 树实现"""
    
    def __init__(self, points):
        self.points = np.array(points)
        self.k = self.points.shape[1]  # 维度
        self.root = self._build_tree(self.points, depth=0)
    
    def _build_tree(self, points, depth):
        """递归构建 KD 树"""
        if len(points) == 0:
            return None
        
        # 选择分割维度（轮换法）
        dimension = depth % self.k
        
        # 按当前维度排序并选择中位数
        sorted_indices = np.argsort(points[:, dimension])
        median_idx = len(points) // 2
        
        # 创建节点
        node = KDNode(
            point=points[sorted_indices[median_idx]],
            dimension=dimension
        )
        
        # 递归构建子树
        node.left = self._build_tree(
            points[sorted_indices[:median_idx]], depth + 1
        )
        node.right = self._build_tree(
            points[sorted_indices[median_idx + 1:]], depth + 1
        )
        
        return node
    
    def nearest_neighbor(self, query_point):
        """最近邻搜索"""
        self.best_distance = float('inf')
        self.best_point = None
        
        def search(node, query, depth):
            if node is None:
                return
            
            # 计算当前节点距离
            dist = distance.euclidean(query, node.point)
            if dist < self.best_distance:
                self.best_distance = dist
                self.best_point = node.point
            
            # 确定搜索方向
            dimension = node.dimension
            if query[dimension] < node.point[dimension]:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left
            
            # 搜索近侧子树
            search(near, query, depth + 1)
            
            # 检查是否需要搜索远侧子树（剪枝）
            # 如果查询点到分割超平面的距离小于当前最优距离
            if abs(query[dimension] - node.point[dimension]) < self.best_distance:
                search(far, query, depth + 1)
        
        search(self.root, query_point, 0)
        return self.best_point, self.best_distance
    
    def k_nearest_neighbors(self, query_point, k):
        """K 近邻搜索"""
        import heapq
        
        # 最大堆存储 K 个最近邻（Python 只有最小堆，存负距离）
        self.knn_heap = []
        
        def search(node, query, depth):
            if node is None:
                return
            
            dist = distance.euclidean(query, node.point)
            
            if len(self.knn_heap) < k:
                heapq.heappush(self.knn_heap, (-dist, tuple(node.point)))
            elif dist < -self.knn_heap[0][0]:
                heapq.heappop(self.knn_heap)
                heapq.heappush(self.knn_heap, (-dist, tuple(node.point)))
            
            dimension = node.dimension
            if query[dimension] < node.point[dimension]:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left
            
            search(near, query, depth + 1)
            
            # 当前堆中最大距离
            max_dist = -self.knn_heap[0][0] if len(self.knn_heap) == k else float('inf')
            
            if abs(query[dimension] - node.point[dimension]) < max_dist:
                search(far, query, depth + 1)
        
        search(self.root, query_point, 0)
        
        # 返回排序结果
        results = sorted([(-d, p) for d, p in self.knn_heap])
        return [(np.array(p), d) for d, p in results]
    
    def range_search(self, query_range):
        """范围查询"""
        results = []
        
        def search(node, depth):
            if node is None:
                return
            
            # 检查当前节点是否在范围内
            in_range = all(
                query_range[i][0] <= node.point[i] <= query_range[i][1]
                for i in range(self.k)
            )
            
            if in_range:
                results.append(node.point)
            
            dimension = node.dimension
            
            # 递归搜索子树
            if query_range[dimension][0] <= node.point[dimension]:
                search(node.left, depth + 1)
            if node.point[dimension] <= query_range[dimension][1]:
                search(node.right, depth + 1)
        
        search(self.root, 0)
        return np.array(results) if results else None


# 可视化 KD 树分割
def visualize_kdtree_2d(points):
    """可视化 2D KD 树的分割结构"""
    
    kdtree = KDTree(points)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, alpha=0.6, label='数据点')
    
    # 递归绘制分割线
    def draw_splits(node, xmin, xmax, ymin, ymax, depth=0):
        if node is None:
            return
        
        if depth % 2 == 0:  # 垂直分割
            plt.axvline(x=node.point[0], color='red', alpha=0.3, linestyle='--')
            draw_splits(node.left, xmin, node.point[0], ymin, ymax, depth + 1)
            draw_splits(node.right, node.point[0], xmax, ymin, ymax, depth + 1)
        else:  # 水平分割
            plt.axhline(y=node.point[1], color='green', alpha=0.3, linestyle='--')
            draw_splits(node.left, xmin, xmax, ymin, node.point[1], depth + 1)
            draw_splits(node.right, xmin, xmax, node.point[1], ymax, depth + 1)
    
    margin = 0.5
    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    
    draw_splits(kdtree.root, xmin, xmax, ymin, ymax)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.title('KD 树分割结构（红: 垂直分割, 绿: 水平分割）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return kdtree


# 演示
np.random.seed(42)
points_2d = np.random.randn(50, 2)
kdtree_demo = visualize_kdtree_2d(points_2d)

# 最近邻搜索
query = np.array([0.5, 0.5])
nearest, dist = kdtree_demo.nearest_neighbor(query)
print(f"查询点: {query}")
print(f"最近邻: {nearest}, 距离: {dist:.4f}")

# K 近邻
knn_results = kdtree_demo.k_nearest_neighbors(query, k=5)
print(f"\n5 个最近邻:")
for point, dist in knn_results:
    print(f"  {point}, 距离: {dist:.4f}")
```

### 最近邻搜索的剪枝原理

**关键洞察**：如果查询点到分割超平面的距离已经大于当前最优距离，则远侧子树一定没有更近的点。

设：
- 查询点 $\mathbf{q}$
- 当前节点分割维度 $d$，分割值 $v$
- 当前最优距离 $r$

如果 $|q_d - v| \geq r$，则远侧子树可以剪枝。

### 复杂度分析

| 操作 | 平均情况 | 最坏情况 |
|------|---------|---------|
| 构建 | $O(n \log n)$ | $O(n^2)$ |
| 最近邻查询 | $O(\log n)$ | $O(n)$ |
| 范围查询 | $O(n^{1-1/k} + m)$ | $O(n)$ |

其中 $k$ 是维度，$m$ 是结果数量。

### 高维问题（维度灾难）

当维度 $k$ 很大时：
- 分割超平面的区分度下降
- 剪枝效果减弱
- 查询复杂度趋近 $O(n)$

**经验法则**：KD 树在 $k \leq 20$ 时效果好

---

## 球树（Ball Tree）

球树是 KD 树的改进，使用**超球面**而非超平面分割空间。

### 核心思想

- 每个节点对应一个超球面，包含一组点
- 球面分割与坐标轴无关，对高维数据更有效

### 优势

- 对高维数据效果更好
- 支持任意距离度量

### 代码示例

```python
from sklearn.neighbors import BallTree

def ball_tree_demo():
    """球树演示"""
    
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    
    # 构建球树
    ball_tree = BallTree(X, leaf_size=40)
    
    # 查询
    query = np.random.randn(1, 10)
    dist, idx = ball_tree.query(query, k=5)
    
    print(f"查询点: {query[0]}")
    print(f"最近邻索引: {idx[0]}")
    print(f"距离: {dist[0]}")
    
    return ball_tree

ball_tree_model = ball_tree_demo()
```

---

## 数据结构比较

### KD 树 vs 球树 vs 暴力搜索

```python
from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
import time

def benchmark_structures():
    """性能对比"""
    
    # 不同规模和维度
    configs = [
        (1000, 5),
        (1000, 10),
        (10000, 5),
        (10000, 10),
        (10000, 20),
    ]
    
    results = []
    
    for n_samples, n_features in configs:
        print(f"\n数据规模: {n_samples}, 维度: {n_features}")
        
        X = np.random.randn(n_samples, n_features)
        query = np.random.randn(1, n_features)
        
        # KD 树
        start = time.time()
        kdtree = KDTree(X)
        kdtree.query(query, k=5)
        kd_time = time.time() - start
        
        # 球树
        start = time.time()
        balltree = BallTree(X)
        balltree.query(query, k=5)
        ball_time = time.time() - start
        
        # 暴力搜索
        start = time.time()
        distances = np.linalg.norm(X - query, axis=1)
        np.argsort(distances)[:5]
        brute_time = time.time() - start
        
        print(f"  KD 树: {kd_time*1000:.2f}ms")
        print(f"  球树: {ball_time*1000:.2f}ms")
        print(f"  暴力搜索: {brute_time*1000:.2f}ms")
        
        results.append({
            'n_samples': n_samples,
            'n_features': n_features,
            'kd_tree': kd_time,
            'ball_tree': ball_time,
            'brute': brute_time
        })
    
    return results

benchmark_results = benchmark_structures()
```

### 选择指南

| 场景 | 推荐方法 |
|------|----------|
| 低维 ($d \leq 20$)、中等规模 | KD 树 |
| 高维 ($d > 20$) | 球树 |
| 小数据集 ($n < 1000$) | 暴力搜索 |
| 需要精确最近邻 | KD 树 / 球树 |
| 需要近似最近邻 | LSH（局部敏感哈希） |

---

## 应用场景

### K 近邻算法加速

```python
from sklearn.neighbors import KNeighborsClassifier

def knn_with_kdtree():
    """KNN 使用 KD 树加速"""
    
    from sklearn.datasets import make_classification
    
    # 生成数据
    X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
    
    # 使用 KD 树的 KNN
    knn_kdtree = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    
    # 使用暴力搜索的 KNN
    knn_brute = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    
    # 训练时间对比
    import time
    
    start = time.time()
    knn_kdtree.fit(X, y)
    kd_fit_time = time.time() - start
    
    start = time.time()
    knn_brute.fit(X, y)
    brute_fit_time = time.time() - start
    
    # 预测时间对比
    X_test = np.random.randn(100, 10)
    
    start = time.time()
    knn_kdtree.predict(X_test)
    kd_pred_time = time.time() - start
    
    start = time.time()
    knn_brute.predict(X_test)
    brute_pred_time = time.time() - start
    
    print(f"训练时间 - KD树: {kd_fit_time*1000:.2f}ms, 暴力: {brute_fit_time*1000:.2f}ms")
    print(f"预测时间 - KD树: {kd_pred_time*1000:.2f}ms, 暴力: {brute_pred_time*1000:.2f}ms")

knn_with_kdtree()
```

### 其他应用

- **图像检索**：在特征空间中搜索相似图像
- **地理信息系统**：最近设施查询
- **推荐系统**：基于用户/物品相似度推荐
- **异常检测**：基于距离的异常检测方法

---

## 小结

| 数据结构 | 分割方式 | 适用维度 | 构建复杂度 | 查询复杂度 |
|----------|---------|---------|-----------|-----------|
| KD 树 | 超平面 | $d \leq 20$ | $O(n \log n)$ | $O(\log n)$ |
| 球树 | 超球面 | 任意 | $O(n \log n)$ | $O(\log n)$ |
| 暴力搜索 | 无 | 任意 | $O(1)$ | $O(n)$ |

**关键要点**：

1. KD 树通过坐标轴分割加速空间查询
2. 剪枝原理是避免不必要的搜索
3. 高维数据需要考虑维度灾难
4. 选择合适的数据结构取决于数据规模和维度

---

**上一节**：[关联规则与异常检测](./association-anomaly.md)
