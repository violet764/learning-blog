# 并查集（Disjoint Set Union）

并查集是一种用于管理元素分组的数据结构，支持高效的**合并（Union）**和**查找（Find）**操作。它特别适合处理**连通性问题**，在图论算法中有着广泛应用。

<UnionFindAnimation />

## 基本概念

### 什么是并查集？

并查集（Disjoint Set Union，简称 DSU，也称 Union-Find）是一种树形数据结构，用于处理一些**不相交集合**的合并与查询问题。

📌 **核心思想**：
- 每个集合用一棵树表示
- 树的根节点作为集合的代表元素
- 同一集合内的元素有相同的根节点

### 基本操作

| 操作 | 描述 | 时间复杂度（优化后） |
|------|------|---------------------|
| Find(x) | 查找元素 x 所属集合的代表元素（根节点） | O(α(n)) ≈ O(1) |
| Union(x, y) | 合并元素 x 和 y 所在的两个集合 | O(α(n)) ≈ O(1) |
| Connected(x, y) | 判断 x 和 y 是否在同一集合 | O(α(n)) ≈ O(1) |

### 应用场景

并查集特别适合处理以下问题：

1. **连通性问题**：判断图中两点是否连通
2. **动态连通性**：动态添加边并查询连通性
3. **最小生成树**：Kruskal 算法中的环检测
4. **社交网络**：朋友圈、群组关系
5. **图像处理**：连通区域标记（如岛屿计数）

### 树形结构示意

```
初始状态：每个元素独立成一个集合
┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ 0 │ │ 1 │ │ 2 │ │ 3 │ │ 4 │  parent[i] = i
└───┘ └───┘ └───┘ └───┘ └───┘

合并后形成树形结构：
      0
     / \
    1   2     →  parent[1]=0, parent[2]=0
       / \
      3   4     →  parent[3]=2, parent[4]=2

查找元素4的根：4 → 2 → 0，根节点是0
```

---

## 优化技术

并查集的性能关键在于两个优化：**路径压缩**和**按秩合并**。单独使用任何一个都能大幅提升性能，两者结合则能达到接近 O(1) 的时间复杂度。

### 路径压缩（Path Compression）

**原理**：在 Find 操作时，将访问路径上的所有节点直接连接到根节点。

```
压缩前：              压缩后：
    0                     0
    |                    /|\
    1                   1 2 3
    |        →          
    2
    |
    3

查找节点3时，将路径上的节点1、2、3都直接连到根节点0
```

**实现方式**：

```cpp
// C++ 递归实现（推荐）
int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);  // 递归找根并更新父节点
    }
    return parent[x];
}

// C++ 迭代实现
int find(int x) {
    int root = x;
    while (parent[root] != root) {
        root = parent[root];  // 找到根
    }
    // 路径压缩：将路径上所有节点直接连到根
    while (parent[x] != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}
```

```python
# Python 递归实现
def find(self, x: int) -> int:
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # 路径压缩
    return self.parent[x]

# Python 迭代实现
def find(self, x: int) -> int:
    # 找到根节点
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    # 路径压缩
    while self.parent[x] != root:
        next_node = self.parent[x]
        self.parent[x] = root
        x = next_node
    return root
```

### 按秩合并（Union by Rank）

**原理**：合并时，将较矮的树挂到较高的树下，避免树退化成链表。

```
按秩合并示例：

情况1：rank[A] < rank[B]
    A          B           B
   / \        /|\    →    /|\
  ...       .....       ..... A
                            /\
                           ...

情况2：rank[A] == rank[B]
    A          B           A（或B，任选一个）
   / \        /|\    →    / | \
  ...       .....       ....  B
                             /|\
                            .....
注意：选为根的树 rank 加 1
```

```cpp
// C++ 实现
void unionSets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    
    if (rootX == rootY) return;  // 已在同一集合
    
    // 按秩合并：小树挂到大树下
    if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
    } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
    } else {
        parent[rootY] = rootX;
        rank[rootX]++;  // 高度相同，合并后高度+1
    }
}
```

```python
# Python 实现
def union(self, x: int, y: int) -> None:
    root_x = self.find(x)
    root_y = self.find(y)
    
    if root_x == root_y:
        return  # 已在同一集合
    
    # 按秩合并
    if self.rank[root_x] < self.rank[root_y]:
        self.parent[root_x] = root_y
    elif self.rank[root_x] > self.rank[root_y]:
        self.parent[root_y] = root_x
    else:
        self.parent[root_y] = root_x
        self.rank[root_x] += 1
```

### 时间复杂度分析

使用路径压缩和按秩合并后，单次操作的时间复杂度为 **O(α(n))**，其中 α 是**阿克曼函数**的反函数。

💡 **阿克曼函数**：增长极快的函数，其反函数 α(n) 增长极慢。

| n | α(n) |
|---|------|
| ≤ 1 | 0 |
| ≤ 4 | 1 |
| ≤ 16 | 2 |
| ≤ 65536 | 3 |
| ≤ 2^65536 | 4 |

📌 **实际意义**：对于所有实际应用，α(n) ≤ 4，可以近似看作 **O(1)** 常数时间。

---

## 实现模板

### 基础并查集

最常用的标准实现，包含路径压缩和按秩合并。

```cpp
// C++ 基础并查集
class UnionFind {
private:
    vector<int> parent;  // 父节点数组
    vector<int> rank;    // 秩（树的高度）
    int count;           // 连通分量数量

public:
    // 构造函数：初始化 n 个独立集合
    UnionFind(int n) : parent(n), rank(n, 0), count(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;  // 初始时每个元素的父节点是自己
        }
    }
    
    // 查找根节点（带路径压缩）
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    // 合并两个集合
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return;  // 已在同一集合
        
        // 按秩合并
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        count--;  // 连通分量减少
    }
    
    // 判断是否连通
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
    
    // 获取连通分量数量
    int getCount() const {
        return count;
    }
};
```

```python
# Python 基础并查集
class UnionFind:
    def __init__(self, n: int):
        """初始化 n 个独立集合"""
        self.parent = list(range(n))  # parent[i] = i
        self.rank = [0] * n           # 初始高度为 0
        self.count = n                # 连通分量数量
    
    def find(self, x: int) -> int:
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        """合并两个集合"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
    
    def connected(self, x: int, y: int) -> bool:
        """判断是否连通"""
        return self.find(x) == self.find(y)
    
    def get_count(self) -> int:
        """获取连通分量数量"""
        return self.count
```

### 带权并查集

适用于需要维护节点间关系的问题，如距离、相对位置等。

```cpp
// C++ 带权并查集
class WeightedUnionFind {
private:
    vector<int> parent;
    vector<int> rank;
    vector<long long> weight;  // weight[i] 表示 i 到 parent[i] 的权值
    
public:
    WeightedUnionFind(int n) : parent(n), rank(n, 0), weight(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    // 查找根节点，同时更新权值
    int find(int x) {
        if (parent[x] != x) {
            int root = find(parent[x]);
            weight[x] += weight[parent[x]];  // 累加权值
            parent[x] = root;
        }
        return parent[x];
    }
    
    // 获取 x 到根的权值
    long long getWeight(int x) {
        find(x);  // 确保路径压缩
        return weight[x];
    }
    
    // 合并：x 到 y 的权值为 w
    // 即 weight(x → y) = w
    bool unite(int x, int y, long long w) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) {
            // 检查一致性
            return getWeight(y) - getWeight(x) == w;
        }
        
        // 按秩合并
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
            // 计算 rootX 到 rootY 的权值
            // weight[x] + weight[rootX] = w + weight[y]
            weight[rootX] = w + getWeight(y) - getWeight(x);
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
            weight[rootY] = getWeight(x) - getWeight(y) - w;
        } else {
            parent[rootY] = rootX;
            weight[rootY] = getWeight(x) - getWeight(y) - w;
            rank[rootX]++;
        }
        return true;
    }
    
    // 查询 x 到 y 的权值
    long long query(int x, int y) {
        if (find(x) != find(y)) return LLONG_MAX;  // 不连通
        return getWeight(y) - getWeight(x);
    }
};
```

```python
# Python 带权并查集
class WeightedUnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # weight[i] = i 到 parent[i] 的权值
    
    def find(self, x: int) -> int:
        """查找根节点，同时更新权值"""
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]
    
    def get_weight(self, x: int) -> int:
        """获取 x 到根的权值"""
        self.find(x)
        return self.weight[x]
    
    def union(self, x: int, y: int, w: int) -> bool:
        """
        合并：x 到 y 的权值为 w
        即 weight(x → y) = w
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            # 检查一致性
            return self.get_weight(y) - self.get_weight(x) == w
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = w + self.get_weight(y) - self.get_weight(x)
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.get_weight(x) - self.get_weight(y) - w
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.get_weight(x) - self.get_weight(y) - w
            self.rank[root_x] += 1
        
        return True
    
    def query(self, x: int, y: int) -> int:
        """查询 x 到 y 的权值"""
        if self.find(x) != self.find(y):
            return None  # 不连通
        return self.get_weight(y) - self.get_weight(x)
```

### 可持久化并查集

支持查询历史版本的并查集，通常使用**可持久化数组**实现。

```cpp
// C++ 可持久化并查集（简化版，使用主席树思想）
// 适用场景：需要回退到历史状态的并查集操作

#include <vector>
#include <memory>
using namespace std;

class PersistentUnionFind {
private:
    struct Node {
        int parent, rank;
        Node(int p = 0, int r = 0) : parent(p), rank(r) {}
    };
    
    vector<vector<pair<int, Node>>> history;  // history[version] = {节点, 值}
    vector<Node> current;
    int n;
    
public:
    PersistentUnionFind(int n) : n(n), current(n) {
        for (int i = 0; i < n; i++) {
            current[i] = Node(i, 0);
        }
    }
    
    // 保存当前版本
    int save() {
        int version = history.size();
        history.push_back({});
        return version;
    }
    
    // 回退到指定版本
    void rollback(int version) {
        for (auto& [node, value] : history[version]) {
            current[node] = value;
        }
    }
    
    // 查找（不使用路径压缩，因为会影响可持久化）
    int find(int x) {
        while (current[x].parent != x) {
            x = current[x].parent;
        }
        return x;
    }
    
    // 合并（记录修改）
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) return false;
        
        // 记录修改
        history.back().push_back({rootX, current[rootX]});
        history.back().push_back({rootY, current[rootY]});
        
        // 按秩合并
        if (current[rootX].rank < current[rootY].rank) {
            current[rootX].parent = rootY;
        } else if (current[rootX].rank > current[rootY].rank) {
            current[rootY].parent = rootX;
        } else {
            current[rootY].parent = rootX;
            current[rootX].rank++;
        }
        return true;
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

---

## 典型应用

### 岛屿数量（LeetCode 200）

**问题描述**：给定一个由 '1'（陆地）和 '0'（水）组成的网格，计算岛屿数量。

**思路**：将相邻的陆地合并成一个集合，最终统计根节点数量。

```cpp
// C++ 实现
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty()) return 0;
        
        int m = grid.size(), n = grid[0].size();
        UnionFind uf(m * n);
        int waterCount = 0;
        
        // 方向数组
        int dx[] = {0, 1};
        int dy[] = {1, 0};
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0') {
                    waterCount++;
                    continue;
                }
                // 只需向右和向下合并，避免重复
                for (int k = 0; k < 2; k++) {
                    int ni = i + dx[k];
                    int nj = j + dy[k];
                    if (ni < m && nj < n && grid[ni][nj] == '1') {
                        uf.unite(i * n + j, ni * n + nj);
                    }
                }
            }
        }
        
        return uf.getCount() - waterCount;
    }
};
```

```python
# Python 实现
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        uf = UnionFind(m * n)
        water_count = 0
        
        # 只需向右和向下合并
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0':
                    water_count += 1
                    continue
                
                # 向右合并
                if j + 1 < n and grid[i][j + 1] == '1':
                    uf.union(i * n + j, i * n + j + 1)
                # 向下合并
                if i + 1 < m and grid[i + 1][j] == '1':
                    uf.union(i * n + j, (i + 1) * n + j)
        
        return uf.get_count() - water_count
```

### 朋友圈问题（LeetCode 547）

**问题描述**：n 个学生，给出他们之间的好友关系矩阵，求朋友圈数量。

```cpp
// C++ 实现
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        UnionFind uf(n);
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    uf.unite(i, j);
                }
            }
        }
        
        return uf.getCount();
    }
};
```

```python
# Python 实现
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        uf = UnionFind(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        
        return uf.get_count()
```

### 连通性判断

动态维护连通性，支持添加边和查询。

```cpp
// C++ 实现：动态连通性
class DynamicConnectivity {
private:
    UnionFind uf;
    
public:
    DynamicConnectivity(int n) : uf(n) {}
    
    // 添加边
    void addEdge(int u, int v) {
        uf.unite(u, v);
    }
    
    // 查询连通性
    bool query(int u, int v) {
        return uf.connected(u, v);
    }
    
    // 获取连通分量数量
    int getComponentCount() {
        return uf.getCount();
    }
};
```

### Kruskal 最小生成树

并查集用于检测添加边是否会形成环。

```cpp
// C++ Kruskal 算法
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

int kruskal(int n, vector<Edge>& edges) {
    UnionFind uf(n);
    sort(edges.begin(), edges.end());
    
    int mstWeight = 0;
    int edgeCount = 0;
    
    for (const Edge& e : edges) {
        if (!uf.connected(e.u, e.v)) {
            uf.unite(e.u, e.v);
            mstWeight += e.weight;
            edgeCount++;
            
            if (edgeCount == n - 1) break;  // MST 有 n-1 条边
        }
    }
    
    // 判断图是否连通
    if (edgeCount != n - 1) {
        return -1;  // 图不连通，无法构建 MST
    }
    
    return mstWeight;
}
```

```python
# Python Kruskal 算法
def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    edges: List of (u, v, weight)
    返回最小生成树总权重，如果不连通返回 -1
    """
    uf = UnionFind(n)
    edges.sort(key=lambda e: e[2])  # 按权重排序
    
    mst_weight = 0
    edge_count = 0
    
    for u, v, w in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst_weight += w
            edge_count += 1
            
            if edge_count == n - 1:
                break
    
    return mst_weight if edge_count == n - 1 else -1
```

---

## 常见问题与注意事项

### ⚠️ 初始化问题

确保初始化时每个元素的父节点指向自己：

```cpp
// 正确 ✓
for (int i = 0; i < n; i++) {
    parent[i] = i;
}

// 错误 ✗（未初始化，可能导致无限循环）
// parent 数组初始值不确定
```

### ⚠️ Find 操作的递归深度

虽然路径压缩后深度很小，但在某些极端情况下（不使用路径压缩），递归可能栈溢出：

```cpp
// 安全做法：使用迭代版本
int find(int x) {
    int root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    // 路径压缩
    while (parent[x] != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}
```

### ⚠️ 按秩合并 vs 按大小合并

两种策略都可以，按大小合并实现更简单：

```cpp
// 按大小合并
vector<int> size(n, 1);  // 记录集合大小

void unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX == rootY) return;
    
    // 小树挂到大树下
    if (size[rootX] < size[rootY]) {
        parent[rootX] = rootY;
        size[rootY] += size[rootX];
    } else {
        parent[rootY] = rootX;
        size[rootX] += size[rootY];
    }
}
```

### 💡 选择合适的优化策略

| 场景 | 推荐优化 | 原因 |
|------|---------|------|
| 一般问题 | 路径压缩 + 按秩合并 | 最优时间复杂度 |
| 需要撤销操作 | 只用按秩合并 | 路径压缩破坏了历史信息 |
| 带权问题 | 带权并查集 + 路径压缩 | 维护节点间关系 |
| 可持久化 | 不用路径压缩 | 保持历史版本可访问 |

---

## 参考资料

- [Union-Find - Wikipedia](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- LeetCode 并查集专题
- 《算法导论》第21章
