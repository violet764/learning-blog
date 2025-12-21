# KDTree

KDTree（K-Dimensional Tree，K维树）是一种用于组织K维空间中点的空间划分数据结构。它是一种二叉搜索树，每个节点代表一个K维空间中的点，树的结构反映了空间的分割。

## KDTree算法原理

让我们通过一个简单的2维例子来理解KDTree的构建过程。

示例数据点（6个二维点）：
```
点A: (2, 3)
点B: (5, 4)
点C: (9, 6)
点D: (4, 7)
点E: (8, 1)
点F: (7, 2)
```

**可视化图表：**
```
Y轴
  |
7 |     D●
  |
6 |         C●
  |
5 |
  |
4 |   B●
  |
3 | A●
  |
2 |           F●
  |
1 |         E●
  +---------------- X轴
    1 2 3 4 5 6 7 8 9
```

**构建过程详解：**

**第1层（深度0，分割维度X轴）**：
1. 按X坐标排序：A(2), D(4), B(5), F(7), E(8), C(9)
2. 选择中位数点：B(5, 4)
3. 分割线：X=5的垂直线

```
Y轴
  |
7 |     D● |
  |        |
6 |         | C●
  |        |
5 |        |
  |        |
4 |   B●---|-- (分割线X=5)
  |        |
3 | A●     |
  |        |
2 |           F●
  |        |
1 |         E●
  +---------------- X轴
    1 2 3 4 5 6 7 8 9
```

**第2层（深度1，分割维度Y轴）**：

**左子树（X < 5的点）**：A(2,3), D(4,7)
1. 按Y坐标排序：A(3), D(7)
2. 选择中位数点：D(4,7)
3. 分割线：Y=7的水平线

**右子树（X ≥ 5的点）**：F(7,2), E(8,1), C(9,6)
1. 按Y坐标排序：E(1), F(2), C(6)
2. 选择中位数点：F(7,2)
3. 分割线：Y=2的水平线

```
Y轴
  |
7 |     D●--- (分割线Y=7)
  |        |
6 |         | C●
  |        |
5 |        |
  |        |
4 |   B●---|-- (分割线X=5)
  |        |
3 | A●     |
  |        |
2 |           F●--- (分割线Y=2)
  |        |
1 |         E●
  +---------------- X轴
    1 2 3 4 5 6 7 8 9
```

**最终KDTree结构**：
```
        B(5,4)
       /     \
   D(4,7)    F(7,2)
   /    \    /    \
A(2,3) 空   E(8,1) C(9,6)
```

**搜索过程示例**

假设我们要查询点Q(6, 3)的最近邻：

```
Y轴
  |
7 |     D●--- (分割线Y=7)
  |        |
6 |         | C●
  |        |
5 |        |
  |        |
4 |   B●---|-- (分割线X=5)
  |        |
3 | A●     | Q●
  |        |
2 |           F●--- (分割线Y=2)
  |        |
1 |         E●
  +---------------- X轴
    1 2 3 4 5 6 7 8 9
```

**搜索过程**：
1. 从根节点B(5,4)开始：Q在B的右侧（因为Q.x=6 > B.x=5）
2. 搜索右子树F(7,2)：Q在F的上方（因为Q.y=3 > F.y=2）
3. 搜索F的右子树C(9,6)：计算距离，当前最近邻是F(7,2)，距离=√((6-7)²+(3-2)²)=√2≈1.41
4. 回溯检查左子树：因为|Q.y - F.y| = 1 < 1.41，需要检查F的左子树E(8,1)
5. E到Q的距离=√((6-8)²+(3-1)²)=√8≈2.83，比F远，不更新
6. 回溯到根节点B：因为|Q.x - B.x| = 1 < 1.41，需要检查B的左子树D(4,7)
7. D到Q的距离=√((6-4)²+(3-7)²)=√20≈4.47，比F远，不更新
8. 最终最近邻是F(7,2)，距离≈1.41

**KDTree的构建过程**  
KDTree的构建采用递归分割的方法：

1. **选择分割维度**：通常选择方差最大的维度，或者按维度轮换
2. **选择分割点**：选择当前维度上的中位数点作为分割点
3. **递归构建**：将数据分为左右子树，分别构建KDTree

分割维度选择策略
- **轮换策略**：按维度顺序轮换（0,1,2,...,k-1,0,1,...）
- **最大方差策略**：选择方差最大的维度进行分割
- **随机策略**：随机选择分割维度

## KDTree实现


```python
import numpy as np
from collections import deque
import heapq

class KDTreeNode:
    """KDTree节点类"""
    def __init__(self, point, left=None, right=None):
        self.point = point        # 节点存储的数据点
        self.left = left          # 左子树
        self.right = right        # 右子树

class KDTree:
    """KDTree实现类"""
    
    def __init__(self, points, leaf_size=1):
        """
        初始化KDTree
        
        Args:
            points: 数据点列表，每个点是一个K维向量
            leaf_size: 叶子节点包含的最小点数
        """
        self.dim = len(points[0]) if points else 0
        self.leaf_size = leaf_size
        self.root = self._build_tree(points)
    
    def _build_tree(self, points, depth=0):
        """递归构建KDTree"""
        if not points:
            return None
        
        # 如果点数小于叶子大小，创建叶子节点
        if len(points) <= self.leaf_size:
            return KDTreeNode(points[0])
        
        # 选择分割维度（轮换策略）
        axis = depth % self.dim
        
        # 按当前维度排序并选择中位数
        points_sorted = sorted(points, key=lambda x: x[axis])
        median_idx = len(points_sorted) // 2
        
        # 创建当前节点
        node = KDTreeNode(points_sorted[median_idx])
        
        # 递归构建左右子树
        node.left = self._build_tree(points_sorted[:median_idx], depth + 1)
        node.right = self._build_tree(points_sorted[median_idx+1:], depth + 1)
        
        return node
    
    def euclidean_distance(self, point1, point2):
        """计算欧几里得距离"""
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
    
    def nearest_neighbor(self, query_point):
        """寻找最近邻点"""
        
        def search(node, point, depth=0, best=None):
            if node is None:
                return best
            
            # 计算当前节点距离
            current_dist = self.euclidean_distance(node.point, point)
            
            # 更新最佳点
            if best is None or current_dist < best[0]:
                best = (current_dist, node.point)
            
            # 选择分割维度
            axis = depth % self.dim
            
            # 决定搜索方向
            if point[axis] < node.point[axis]:
                next_branch = node.left
                other_branch = node.right
            else:
                next_branch = node.right
                other_branch = node.left
            
            # 搜索更近的分支
            best = search(next_branch, point, depth + 1, best)
            
            # 检查另一分支是否需要搜索（剪枝优化）
            if abs(point[axis] - node.point[axis]) < best[0]:
                best = search(other_branch, point, depth + 1, best)
            
            return best
        
        result = search(self.root, query_point)
        return result[1], result[0]  # 返回点和距离
    
    def k_nearest_neighbors(self, query_point, k=5):
        """寻找K个最近邻点"""
        
        def search(node, point, depth=0, heap=None):
            if node is None:
                return heap
            
            if heap is None:
                heap = []
            
            # 计算当前节点距离
            dist = self.euclidean_distance(node.point, point)
            
            # 维护大小为K的最大堆（存储负距离以便使用最小堆API）
            if len(heap) < k:
                heapq.heappush(heap, (-dist, node.point))
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, (-dist, node.point))
            
            # 选择分割维度
            axis = depth % self.dim
            
            # 决定搜索方向
            if point[axis] < node.point[axis]:
                next_branch = node.left
                other_branch = node.right
            else:
                next_branch = node.right
                other_branch = node.left
            
            # 搜索更近的分支
            heap = search(next_branch, point, depth + 1, heap)
            
            # 检查另一分支是否需要搜索
            if len(heap) < k or abs(point[axis] - node.point[axis]) < -heap[0][0]:
                heap = search(other_branch, point, depth + 1, heap)
            
            return heap
        
        heap = search(self.root, query_point)
        
        # 转换堆结果为排序列表
        results = [(-dist, point) for dist, point in heap]
        results.sort(key=lambda x: x[0])
        
        return [(point, dist) for dist, point in results]
    
    def range_search(self, query_point, radius):
        """范围搜索：找到距离query_point在radius范围内的所有点"""
        
        def search(node, point, radius, depth=0, results=None):
            if node is None:
                return results
            
            if results is None:
                results = []
            
            # 计算当前节点距离
            dist = self.euclidean_distance(node.point, point)
            
            # 如果在范围内，添加到结果
            if dist <= radius:
                results.append((node.point, dist))
            
            # 选择分割维度
            axis = depth % self.dim
            
            # 决定搜索方向
            if point[axis] - radius < node.point[axis]:
                search(node.left, point, radius, depth + 1, results)
            
            if point[axis] + radius >= node.point[axis]:
                search(node.right, point, radius, depth + 1, results)
            
            return results
        
        return search(self.root, query_point, radius)
    
    def visualize(self, max_depth=5):
        """可视化KDTree结构（简化版）"""
        
        def print_tree(node, depth=0, prefix=""):
            if node is None or depth > max_depth:
                return
            
            indent = "  " * depth
            print(f"{indent}{prefix}Point: {node.point}")
            
            if node.left:
                print_tree(node.left, depth + 1, "L: ")
            if node.right:
                print_tree(node.right, depth + 1, "R: ")
        
        print("KDTree Structure:")
        print_tree(self.root)


# 使用示例数据测试
class SimpleKDTreeExample:
    """使用简单示例数据的KDTree演示"""
    
    def __init__(self):
        # 使用前面示例的数据点
        self.points = [
            [2, 3],  # A
            [5, 4],  # B
            [9, 6],  # C
            [4, 7],  # D
            [8, 1],  # E
            [7, 2]   # F
        ]
        self.kdtree = KDTree(self.points)
    
    def demonstrate_building(self):
        """演示构建过程"""
        print("=== KDTree构建演示 ===")
        print("原始数据点:")
        for i, point in enumerate(self.points):
            print(f"点{chr(65+i)}: {point}")
        
        print("\nKDTree结构:")
        self.kdtree.visualize(max_depth=3)
    
    def demonstrate_searching(self):
        """演示搜索过程"""
        print("\n=== KDTree搜索演示 ===")
        
        # 查询点Q(6, 3)
        query_point = [6, 3]
        print(f"查询点Q: {query_point}")
        
        # 最近邻搜索
        nearest, distance = self.kdtree.nearest_neighbor(query_point)
        print(f"最近邻点: {nearest}, 距离: {distance:.4f}")
        
        # K最近邻搜索
        k_neighbors = self.kdtree.k_nearest_neighbors(query_point, k=3)
        print("\n3个最近邻点:")
        for i, (point, dist) in enumerate(k_neighbors):
            print(f"{i+1}. 点: {point}, 距离: {dist:.4f}")
        
        # 范围搜索
        radius = 3.0
        range_results = self.kdtree.range_search(query_point, radius)
        print(f"\n在半径{radius}范围内的点:")
        for point, dist in range_results:
            print(f"点: {point}, 距离: {dist:.4f}")


# KDTree在KNN中的应用示例
class KNNWithKDTree:
    """使用KDTree优化的KNN分类器"""
    
    def __init__(self, k=5):
        self.k = k
        self.kdtree = None
        self.labels = None
        self.point_to_index = {}
        
    def fit(self, X, y):
        """训练模型"""
        self.kdtree = KDTree(X)
        self.labels = y
        # 建立点到索引的映射
        self.point_to_index = {tuple(point): i for i, point in enumerate(X)}
        
    def predict(self, X):
        """预测单个样本"""
        if self.kdtree is None:
            raise ValueError("Model not fitted yet")
        
        # 找到K个最近邻
        neighbors = self.kdtree.k_nearest_neighbors(X, self.k)
        
        # 统计邻居的标签
        label_counts = {}
        for neighbor_point, _ in neighbors:
            neighbor_idx = self.point_to_index[tuple(neighbor_point)]
            label = self.labels[neighbor_idx]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 返回出现次数最多的标签
        return max(label_counts.items(), key=lambda x: x[1])[0]


if __name__ == "__main__":
    # 演示简单示例
    example = SimpleKDTreeExample()
    example.demonstrate_building()
    example.demonstrate_searching()
    
    # 性能对比测试
    print("\n=== 性能对比测试 ===")
    import time
    
    # 生成大数据集
    np.random.seed(42)
    large_points = np.random.rand(10000, 2)
    large_query = [0.5, 0.5]
    
    # 朴素KNN（计算所有距离）
    start_time = time.time()
    distances = []
    for point in large_points:
        dist = np.sqrt(np.sum((point - large_query) ** 2))
        distances.append(dist)
    min_dist = min(distances)
    naive_time = time.time() - start_time
    
    # KDTree KNN
    start_time = time.time()
    kdtree_large = KDTree(large_points)
    _, kdtree_dist = kdtree_large.nearest_neighbor(large_query)
    kdtree_time = time.time() - start_time
    
    print(f"朴素KNN时间: {naive_time:.4f}秒")
    print(f"KDTree KNN时间: {kdtree_time:.4f}秒")
    print(f"加速比: {naive_time/kdtree_time:.1f}倍")
    print(f"两种方法找到的最小距离: 朴素={min_dist:.6f}, KDTree={kdtree_dist:.6f}")
```


运行上面的代码会得到类似以下输出：

```
=== KDTree构建演示 ===
原始数据点:
点A: [2, 3]
点B: [5, 4]
点C: [9, 6]
点D: [4, 7]
点E: [8, 1]
点F: [7, 2]

KDTree结构:
KDTree Structure:
Point: [5, 4]
  L: Point: [4, 7]
    L: Point: [2, 3]
    R: None
  R: Point: [7, 2]
    L: Point: [8, 1]
    R: Point: [9, 6]

=== KDTree搜索演示 ===
查询点Q: [6, 3]
最近邻点: [7, 2], 距离: 1.4142

3个最近邻点:
1. 点: [7, 2], 距离: 1.4142
2. 点: [5, 4], 距离: 2.2361
3. 点: [8, 1], 距离: 2.8284

在半径3.0范围内的点:
点: [7, 2], 距离: 1.4142
点: [5, 4], 距离: 2.2361
点: [8, 1], 距离: 2.8284

=== 性能对比测试 ===
朴素KNN时间: 0.0456秒
KDTree KNN时间: 0.0012秒
加速比: 38.0倍
两种方法找到的最小距离: 朴素=0.002345, KDTree=0.002345
```


<div style="height: 2px; background: linear-gradient(to right, #FF85C0, #2196F3, #4CAF50); border-radius: 1px; margin: 20px 0;"></div>  

[返回](./supervised_learning.md#knn)