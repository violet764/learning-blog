# 最小生成树

最小生成树（Minimum Spanning Tree, MST）是图论中的重要概念，它是在保证连通性的前提下，使边的总权重最小的树结构。最小生成树在网络设计、聚类分析等领域有广泛应用。

## 基本概念

### 生成树定义

**生成树（Spanning Tree）** 是连通图的一个子图，满足以下条件：
- 包含图中所有顶点
- 是一棵树（无环、连通）
- 边数为顶点数减一（|E| = |V| - 1）

📌 **理解要点**：
- 生成树是连接所有顶点的"最精简"方式
- 一个图可能有多个不同的生成树
- 非连通图不存在生成树（每个连通分量有生成森林）

```
原图：              生成树示例：
    A --- B             A --- B
    |  /  |             |     |
    | /   |             |     |
    C --- D             C --- D

保留所有4个顶点，3条边（无环）
```

### 最小生成树（MST）

**最小生成树** 是所有生成树中边权之和最小的那棵树。

**数学定义**：给定加权连通图 G = (V, E, w)，求生成树 T = (V, E')，使得：
$$\sum_{e \in E'} w(e) \text{ 最小}$$

💡 **关键性质**：
1. **唯一性**：如果所有边权都不相同，MST 唯一
2. **切分定理**：任意切分，切分两端的最小权边必属于某个 MST
3. **环性质**：在环上删除最大权边，不影响 MST 的存在性

### 应用场景

| 应用领域 | 具体场景 |
|---------|---------|
| 网络设计 | 通信网络、电力网络、管道铺设 |
| 交通规划 | 公路网、铁路网的最低成本建设 |
| 聚类分析 | 单链接聚类的合并过程 |
| 图像处理 | 图像分割、区域合并 |
| VLSI设计 | 电路布线优化 |

---

## Prim 算法

### 算法原理

Prim 算法采用**贪心策略**，从一个顶点开始，逐步扩展生成树：

1. 从任意顶点开始，将其加入 MST
2. 在所有连接 MST 内外顶点的边中，选择权值最小的边
3. 将该边及新顶点加入 MST
4. 重复步骤 2-3，直到所有顶点都加入

📌 **核心思想**：每次选择"跨越边界"的最小边，保证局部最优导致全局最优。

```
初始状态：          选择最小边：         继续扩展：
    A(已选)             A --- B             A --- B
    |                   |                   |  \
    | x                 | ✓                 |   \
    C --- D             C --- D             C(已选) D

选择 A-C 边（假设权重最小）
```

### 优先队列实现

#### C++ 实现

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

/**
 * Prim 算法实现（优先队列版本）
 * 时间复杂度：O(E log V)
 * 空间复杂度：O(V + E)
 */

// 边结构：目标顶点和权重
struct Edge {
    int to;      // 目标顶点
    int weight;  // 边权重
    
    Edge(int t, int w) : to(t), weight(w) {}
    
    // 优先队列默认是大顶堆，需要反转比较器
    bool operator>(const Edge& other) const {
        return weight > other.weight;
    }
};

// Prim 算法主函数
int prim(int n, vector<vector<Edge>>& graph, int start = 0) {
    // n: 顶点数量
    // graph: 邻接表表示的图
    // start: 起始顶点
    
    vector<bool> visited(n, false);  // 标记顶点是否已加入MST
    vector<int> minDist(n, INT_MAX); // 记录每个顶点到MST的最小距离
    vector<int> parent(n, -1);       // 记录MST中每个顶点的父节点
    
    // 小顶堆：存储 (权重, 顶点)
    priority_queue<Edge, vector<Edge>, greater<Edge>> pq;
    
    // 从起始顶点开始
    pq.push(Edge(start, 0));
    minDist[start] = 0;
    
    int totalWeight = 0;  // MST的总权重
    int edgeCount = 0;    // 已选边数
    
    while (!pq.empty() && edgeCount < n) {
        // 取出当前最小权边对应的顶点
        Edge curr = pq.top();
        pq.pop();
        
        int u = curr.to;
        
        // 如果顶点已访问，跳过（避免重复处理）
        if (visited[u]) continue;
        
        // 将顶点加入MST
        visited[u] = true;
        totalWeight += curr.weight;
        edgeCount++;
        
        // 遍历所有邻边，更新最小距离
        for (const Edge& e : graph[u]) {
            int v = e.to;
            int w = e.weight;
            
            // 如果v未访问，且发现更小的边
            if (!visited[v] && w < minDist[v]) {
                minDist[v] = w;
                parent[v] = u;
                pq.push(Edge(v, w));
            }
        }
    }
    
    // 打印MST结构
    cout << "最小生成树边：" << endl;
    for (int i = 1; i < n; i++) {
        if (parent[i] != -1) {
            cout << parent[i] << " -- " << i << endl;
        }
    }
    
    return totalWeight;
}

int main() {
    // 示例：5个顶点的图
    int n = 5;
    vector<vector<Edge>> graph(n);
    
    // 添加边（无向图，添加两次）
    auto addEdge = [&](int u, int v, int w) {
        graph[u].push_back(Edge(v, w));
        graph[v].push_back(Edge(u, w));
    };
    
    addEdge(0, 1, 2);  // A-B, 权重2
    addEdge(0, 2, 1);  // A-C, 权重1
    addEdge(1, 2, 3);  // B-C, 权重3
    addEdge(1, 3, 4);  // B-D, 权重4
    addEdge(2, 4, 5);  // C-E, 权重5
    addEdge(3, 4, 1);  // D-E, 权重1
    
    int mstWeight = prim(n, graph, 0);
    cout << "最小生成树总权重: " << mstWeight << endl;
    
    return 0;
}
```

#### Python 实现

```python
import heapq
from typing import List, Tuple

def prim(n: int, graph: List[List[Tuple[int, int]]], start: int = 0) -> int:
    """
    Prim 算法实现（优先队列版本）
    
    参数：
        n: 顶点数量
        graph: 邻接表，graph[u] = [(v, weight), ...]
        start: 起始顶点
    
    返回：
        最小生成树的总权重
    """
    visited = [False] * n      # 标记顶点是否已加入MST
    min_dist = [float('inf')] * n  # 每个顶点到MST的最小距离
    parent = [-1] * n          # 记录MST中每个顶点的父节点
    
    # 小顶堆：(权重, 顶点)
    heap = [(0, start)]
    min_dist[start] = 0
    
    total_weight = 0
    edge_count = 0
    
    while heap and edge_count < n:
        # 取出当前最小权边
        weight, u = heapq.heappop(heap)
        
        # 如果顶点已访问，跳过
        if visited[u]:
            continue
        
        # 将顶点加入MST
        visited[u] = True
        total_weight += weight
        edge_count += 1
        
        # 遍历所有邻边
        for v, w in graph[u]:
            if not visited[v] and w < min_dist[v]:
                min_dist[v] = w
                parent[v] = u
                heapq.heappush(heap, (w, v))
    
    # 打印MST结构
    print("最小生成树边：")
    for i in range(1, n):
        if parent[i] != -1:
            print(f"{parent[i]} -- {i}")
    
    return total_weight


def main():
    # 示例：5个顶点的图
    n = 5
    graph = [[] for _ in range(n)]
    
    # 添加边（无向图，添加两次）
    def add_edge(u: int, v: int, w: int):
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    add_edge(0, 1, 2)  # A-B, 权重2
    add_edge(0, 2, 1)  # A-C, 权重1
    add_edge(1, 2, 3)  # B-C, 权重3
    add_edge(1, 3, 4)  # B-D, 权重4
    add_edge(2, 4, 5)  # C-E, 权重5
    add_edge(3, 4, 1)  # D-E, 权重1
    
    mst_weight = prim(n, graph, 0)
    print(f"最小生成树总权重: {mst_weight}")


if __name__ == "__main__":
    main()
```

### 时间复杂度分析

| 实现方式 | 时间复杂度 | 适用场景 |
|---------|-----------|---------|
| 邻接矩阵 + 暴力查找 | O(V²) | 稠密图 |
| 邻接表 + 二叉堆 | O(E log V) | 稀疏图 |
| 邻接表 + 斐波那契堆 | O(E + V log V) | 理论最优 |

⚠️ **注意**：优先队列版本中，每条边最多入堆一次，但可能存在重复的顶点（不同权重），所以实际复杂度为 O(E log E) = O(E log V)（因为 E ≤ V²）。

---

## Kruskal 算法

### 算法原理

Kruskal 算法同样采用**贪心策略**，但方式不同：

1. 将所有边按权重从小到大排序
2. 从最小的边开始，依次考虑每条边
3. 如果这条边连接的是两个不同的连通分量，则加入 MST
4. 使用并查集判断和合并连通分量
5. 重复直到有 |V|-1 条边

📌 **核心思想**：始终选择全局最小且不形成环的边，利用并查集高效判断连通性。

```
边排序后：E(1), A-C(1), A-B(2), B-C(3), B-D(4), C-E(5)

步骤1：选 D-E(1)，不形成环 ✓
步骤2：选 A-C(1)，不形成环 ✓
步骤3：选 A-B(2)，不形成环 ✓
已选3条边，MST完成！

结果：A-C, A-B, D-E，总权重=1+1+2=4
```

### 边排序 + 并查集实现

#### C++ 实现

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/**
 * Kruskal 算法实现（并查集版本）
 * 时间复杂度：O(E log E) = O(E log V)
 * 空间复杂度：O(V + E)
 */

// 边结构
struct Edge {
    int from;    // 起点
    int to;      // 终点
    int weight;  // 权重
    
    Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
    
    // 按权重升序排序
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// 并查集类
class UnionFind {
private:
    vector<int> parent;
    vector<int> rank;
    
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    // 路径压缩查找
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 路径压缩
        }
        return parent[x];
    }
    
    // 按秩合并
    bool unite(int x, int y) {
        int px = find(x);
        int py = find(y);
        
        if (px == py) return false;  // 已在同一集合
        
        // 按秩合并：将小树挂到大树下
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// Kruskal 算法主函数
int kruskal(int n, vector<Edge>& edges) {
    // n: 顶点数量
    // edges: 所有边的列表
    
    // 按权重升序排序
    sort(edges.begin(), edges.end());
    
    UnionFind uf(n);
    vector<Edge> mstEdges;  // 存储MST中的边
    int totalWeight = 0;
    
    for (const Edge& e : edges) {
        // 如果这条边连接两个不同的连通分量
        if (uf.unite(e.from, e.to)) {
            mstEdges.push_back(e);
            totalWeight += e.weight;
            
            // MST已有 n-1 条边，提前结束
            if (mstEdges.size() == n - 1) {
                break;
            }
        }
    }
    
    // 检查图是否连通
    if (mstEdges.size() != n - 1) {
        cout << "图不连通，无法构成生成树！" << endl;
        return -1;
    }
    
    // 打印MST结构
    cout << "最小生成树边：" << endl;
    for (const Edge& e : mstEdges) {
        cout << e.from << " -- " << e.to << " (权重: " << e.weight << ")" << endl;
    }
    
    return totalWeight;
}

int main() {
    int n = 5;  // 5个顶点
    vector<Edge> edges;
    
    // 添加边
    edges.push_back(Edge(0, 1, 2));  // A-B, 权重2
    edges.push_back(Edge(0, 2, 1));  // A-C, 权重1
    edges.push_back(Edge(1, 2, 3));  // B-C, 权重3
    edges.push_back(Edge(1, 3, 4));  // B-D, 权重4
    edges.push_back(Edge(2, 4, 5));  // C-E, 权重5
    edges.push_back(Edge(3, 4, 1));  // D-E, 权重1
    
    int mstWeight = kruskal(n, edges);
    cout << "最小生成树总权重: " << mstWeight << endl;
    
    return 0;
}
```

#### Python 实现

```python
from typing import List, Tuple

class UnionFind:
    """并查集实现（路径压缩 + 按秩合并）"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """路径压缩查找"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def unite(self, x: int, y: int) -> bool:
        """
        合并两个集合
        返回：True 表示合并成功，False 表示已在同一集合
        """
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """判断两个元素是否在同一集合"""
        return self.find(x) == self.find(y)


def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    Kruskal 算法实现
    
    参数：
        n: 顶点数量
        edges: 边列表，每个元素为 (u, v, weight)
    
    返回：
        最小生成树的总权重，如果不连通返回 -1
    """
    # 按权重升序排序
    edges = sorted(edges, key=lambda e: e[2])
    
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0
    
    for u, v, w in edges:
        # 如果这条边连接两个不同的连通分量
        if uf.unite(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            
            # MST已有 n-1 条边
            if len(mst_edges) == n - 1:
                break
    
    # 检查图是否连通
    if len(mst_edges) != n - 1:
        print("图不连通，无法构成生成树！")
        return -1
    
    # 打印MST结构
    print("最小生成树边：")
    for u, v, w in mst_edges:
        print(f"{u} -- {v} (权重: {w})")
    
    return total_weight


def main():
    n = 5  # 5个顶点
    edges = [
        (0, 1, 2),  # A-B, 权重2
        (0, 2, 1),  # A-C, 权重1
        (1, 2, 3),  # B-C, 权重3
        (1, 3, 4),  # B-D, 权重4
        (2, 4, 5),  # C-E, 权重5
        (3, 4, 1),  # D-E, 权重1
    ]
    
    mst_weight = kruskal(n, edges)
    print(f"最小生成树总权重: {mst_weight}")


if __name__ == "__main__":
    main()
```

### 时间复杂度分析

| 步骤 | 时间复杂度 |
|-----|-----------|
| 边排序 | O(E log E) |
| 并查集操作 | O(E α(V)) ≈ O(E) |
| **总计** | O(E log E) = O(E log V) |

💡 **说明**：
- α(V) 是反阿克曼函数，增长极慢，实际可视为常数
- E log E = E log(V²) = 2E log V = O(E log V)

---

## 算法对比

### 稠密图 vs 稀疏图

```
稠密图：E ≈ V²        稀疏图：E ≈ V
```

| 特性 | Prim | Kruskal |
|-----|------|---------|
| **策略** | 从顶点扩展 | 从边扩展 |
| **数据结构** | 优先队列 | 并查集 |
| **时间复杂度** | O(E log V) | O(E log E) |
| **空间复杂度** | O(V + E) | O(V + E) |
| **适合稠密图** | ✅ 优（可优化到O(V²)） | ⚠️ 较差 |
| **适合稀疏图** | ✅ 良好 | ✅ 优秀 |
| **实现难度** | 中等 | 简单 |
| **求MST形态** | 连续扩展 | 离散合并 |

### 选择建议

| 场景 | 推荐算法 |
|-----|---------|
| 邻接矩阵存储 + 稠密图 | Prim（暴力版 O(V²)） |
| 邻接表存储 + 稀疏图 | Kruskal 或 Prim（堆优化） |
| 需要按边处理 | Kruskal |
| 需要知道MST的生成过程 | Prim（逐步扩展） |
| 图的连通分量已知 | Kruskal |

---

## 典型应用

### 网络设计

**问题**：设计连接 n 个城市的最小成本通信网络。

```python
def network_design(cities: int, connections: List[Tuple[int, int, int]]) -> int:
    """
    最小成本网络设计
    
    参数：
        cities: 城市数量
        connections: 可选连接 (城市1, 城市2, 成本)
    
    返回：
        最小总成本，如果无法连通所有城市返回 -1
    """
    return kruskal(cities, connections)


# 示例：5个城市
cities = 5
connections = [
    (0, 1, 100),  # 城市0-1，成本100万
    (0, 2, 50),   # 城市0-2，成本50万
    (1, 2, 80),   # 城市1-2，成本80万
    (1, 3, 90),   # 城市1-3，成本90万
    (2, 4, 60),   # 城市2-4，成本60万
    (3, 4, 70),   # 城市3-4，成本70万
]

min_cost = network_design(cities, connections)
print(f"最小建设成本: {min_cost}万")
```

### 聚类分析

**问题**：使用 MST 进行层次聚类（单链接聚类）。

```python
def mst_clustering(points: List[List[float]], k: int) -> List[int]:
    """
    基于 MST 的聚类算法
    
    思路：
    1. 构建完全图，边权为点间距离
    2. 求 MST
    3. 删除 k-1 条最大边，得到 k 个聚类
    
    参数：
        points: 数据点坐标列表
        k: 聚类数量
    
    返回：
        每个点的聚类标签
    """
    from itertools import combinations
    import math
    
    n = len(points)
    
    # 计算所有点对之间的距离
    edges = []
    for i, j in combinations(range(n), 2):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(points[i], points[j])))
        edges.append((i, j, dist))
    
    # 按 MST 顺序选边（这里简化处理）
    edges.sort(key=lambda e: e[2])
    
    uf = UnionFind(n)
    mst_edges = []
    
    for u, v, w in edges:
        if uf.unite(u, v):
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break
    
    # 删除最大的 k-1 条边
    mst_edges.sort(key=lambda e: e[2], reverse=True)
    uf = UnionFind(n)  # 重新初始化
    
    # 加入前 n-k 条边（跳过最大的 k-1 条）
    for u, v, w in sorted(mst_edges[k-1:], key=lambda e: e[2]):
        uf.unite(u, v)
    
    # 为每个点分配聚类标签
    cluster_map = {}
    labels = []
    current_label = 0
    
    for i in range(n):
        root = uf.find(i)
        if root not in cluster_map:
            cluster_map[root] = current_label
            current_label += 1
        labels.append(cluster_map[root])
    
    return labels


# 示例
points = [
    [0, 0], [1, 1],   # 类别1：左下
    [10, 10], [11, 11],  # 类别2：右上
    [5, 5]  # 类别3：中间
]
labels = mst_clustering(points, k=3)
print(f"聚类标签: {labels}")
```

---

## 常见问题

### 1. MST 是否唯一？

**不唯一**。当存在多条相同权重的边时，可能有多个不同的 MST，但它们的总权重相同。

```
例子：等边三角形
    A
   /|\
  1 1 1
 /  |  \
B---1---C

可以选择任意两条边，共3种MST，但权重都是2
```

### 2. 如何求次小生成树？

**方法**：枚举 MST 中的每条边，删除后重新计算。

```python
def second_mst(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """求次小生成树"""
    edges_sorted = sorted(edges, key=lambda e: e[2])
    
    # 先求 MST
    uf = UnionFind(n)
    mst_edges = []
    
    for e in edges_sorted:
        if uf.unite(e[0], e[1]):
            mst_edges.append(e)
    
    # 枚举删除 MST 中的每条边
    second_weight = float('inf')
    
    for skip_edge in mst_edges:
        uf = UnionFind(n)
        weight = 0
        count = 0
        
        for e in edges_sorted:
            if e == skip_edge:
                continue
            if uf.unite(e[0], e[1]):
                weight += e[2]
                count += 1
        
        if count == n - 1:  # 找到有效生成树
            second_weight = min(second_weight, weight)
    
    return second_weight
```

### 3. 有负权边怎么办？

✅ **Prim 和 Kruskal 都支持负权边**，算法逻辑不变。只需注意：
- 边排序时负边会优先被选择
- 总权重可能为负

### 4. 图不连通怎么办？

- 返回生成森林（每个连通分量一个 MST）
- 或者提前检测连通性，报错处理

---

## 总结

| 算法 | 核心思想 | 适用场景 |
|-----|---------|---------|
| **Prim** | 从顶点扩展，每次选最小切分边 | 稠密图、邻接矩阵存储 |
| **Kruskal** | 从边扩展，选最小非环边 | 稀疏图、按边处理 |

💡 **选择口诀**：
> 稠密图用 Prim，稀疏图用 Kruskal  
> 需要边处理就 Kruskal，需要渐进扩展就 Prim
