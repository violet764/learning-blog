# 最短路径算法

最短路径问题是图论中最经典的问题之一，旨在寻找图中两个顶点之间"代价最小"的路径。这类问题在实际中有着广泛的应用，如地图导航、网络路由、物流规划等。

本文将系统介绍几种经典的最短路径算法，从单源最短路径到多源最短路径，从无负权边到负权边处理，帮助你建立完整的知识体系。

## 问题概述

### 问题分类

最短路径问题可以从不同维度进行分类：

| 分类维度 | 类型 | 描述 | 代表算法 |
|---------|------|------|----------|
| **源点数量** | 单源最短路径 | 一个起点到所有其他点 | Dijkstra, Bellman-Ford, SPFA |
| | 多源最短路径 | 任意两点间最短路径 | Floyd |
| **边的权重** | 无权图 | 每条边权重相同 | BFS |
| | 有权图（非负） | 边有权重但无负权 | Dijkstra |
| | 有权图（含负权） | 存在负权边 | Bellman-Ford, SPFA |

### 形式化定义

给定带权图 $G = (V, E)$，每条边 $(u, v)$ 有权重 $w(u, v)$：

- **路径权重**：路径 $p = (v_0, v_1, ..., v_k)$ 的权重为：
$$w(p) = \sum_{i=1}^{k} w(v_{i-1}, v_i)$$

- **最短路径**：从 $u$ 到 $v$ 的最短路径是权重最小的路径，记为 $\delta(u, v)$

### 最优子结构

最短路径具有**最优子结构**性质：如果 $p = (v_0, v_1, ..., v_k)$ 是从 $v_0$ 到 $v_k$ 的最短路径，那么 $p$ 的任意子路径也是相应两点间的最短路径。

::: tip 三角不等式
对于任意顶点 $u, v, x$，有：
$$\delta(u, v) \leq \delta(u, x) + \delta(x, v)$$
:::

---

## Dijkstra 算法

Dijkstra 算法是求解**非负权图单源最短路径**的经典算法，由荷兰计算机科学家 Edsger W. Dijkstra 于 1956 年提出。

### 算法原理

Dijkstra 算法采用**贪心策略**：每次从未确定最短路径的顶点中，选择距离起点最近的顶点，然后对其所有出边进行**松弛操作**。

**松弛操作（Relaxation）**：尝试通过顶点 $u$ 改进到顶点 $v$ 的最短距离：
$$\text{if } d[u] + w(u, v) < d[v]: \quad d[v] = d[u] + w(u, v)$$

### 算法步骤

```mermaid
graph TD
    A[初始化: d[start] = 0, 其他 d = ∞] --> B[将所有顶点加入候选集]
    B --> C{候选集为空?}
    C -->|否| D[选择候选集中 d 值最小的顶点 u]
    D --> E[将 u 移出候选集, 标记为已确定]
    E --> F[对 u 的每个邻居 v 进行松弛]
    F --> C
    C -->|是| G[算法结束]
```

### 动画演示

<DijkstraAnimation />

### 代码实现

#### Python 实现（优先队列优化）

```python
import heapq
from typing import List, Tuple

def dijkstra(n: int, edges: List[Tuple[int, int, int]], start: int) -> List[int]:
    """
    Dijkstra 算法求单源最短路径（优先队列优化）
    
    :param n: 顶点数量
    :param edges: 边列表，每个元素为 (u, v, weight)
    :param start: 起点
    :return: 起点到各顶点的最短距离，-1 表示不可达
    """
    # 构建邻接表
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # 初始化距离数组
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    
    # 优先队列：(距离, 顶点)
    pq = [(0, start)]
    
    # 已确定的顶点集合
    confirmed = [False] * n
    
    while pq:
        d, u = heapq.heappop(pq)
        
        # 如果该顶点已确定，跳过
        if confirmed[u]:
            continue
        
        # 标记为已确定
        confirmed[u] = True
        
        # 松弛邻居
        for v, w in graph[u]:
            if not confirmed[v] and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    # 将 INF 转换为 -1
    return [-1 if d == INF else d for d in dist]


def dijkstra_with_path(n: int, edges: List[Tuple[int, int, int]], 
                       start: int, end: int) -> Tuple[int, List[int]]:
    """
    Dijkstra 算法求最短路径并返回路径
    
    :return: (最短距离, 路径列表)，不可达返回 (-1, [])
    """
    # 构建邻接表
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    parent = [-1] * n  # 记录前驱节点
    
    pq = [(0, start)]
    confirmed = [False] * n
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if confirmed[u]:
            continue
        confirmed[u] = True
        
        # 提前终止：找到终点
        if u == end:
            break
        
        for v, w in graph[u]:
            if not confirmed[v] and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))
    
    # 重建路径
    if dist[end] == INF:
        return -1, []
    
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    
    return int(dist[end]), path


# 使用示例
if __name__ == "__main__":
    n = 5
    edges = [
        (0, 1, 4), (0, 2, 2),
        (1, 2, 1), (1, 3, 5),
        (2, 1, 1), (2, 3, 8), (2, 4, 10),
        (3, 4, 2)
    ]
    
    # 计算从顶点 0 到所有点的最短距离
    distances = dijkstra(n, edges, 0)
    print(f"从顶点 0 到各点的最短距离: {distances}")
    # 输出: [0, 3, 2, 8, 10]
    
    # 求从 0 到 4 的最短路径
    distance, path = dijkstra_with_path(n, edges, 0, 4)
    print(f"从 0 到 4 的最短距离: {distance}, 路径: {path}")
    # 输出: 最短距离: 10, 路径: [0, 2, 1, 3, 4]
```

#### C++ 实现

```cpp
#include <vector>
#include <queue>
#include <limits>
using namespace std;

// Dijkstra 算法（优先队列优化）
vector<long long> dijkstra(int n, const vector<vector<pair<int, int>>>& graph, int start) {
    /*
     * Dijkstra 算法求单源最短路径
     * graph[u] = {(v, weight), ...} 邻接表
     * 返回 start 到各顶点的最短距离
     */
    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    dist[start] = 0;
    
    // 优先队列：{距离, 顶点}，小根堆
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    pq.push({0, start});
    
    // 已确定的顶点
    vector<bool> confirmed(n, false);
    
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        // 已确定则跳过
        if (confirmed[u]) continue;
        confirmed[u] = true;
        
        // 松弛邻居
        for (auto& [v, w] : graph[u]) {
            if (!confirmed[v] && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    
    return dist;
}

// 带路径记录的 Dijkstra
pair<long long, vector<int>> dijkstraWithPath(
    int n, 
    const vector<vector<pair<int, int>>>& graph, 
    int start, 
    int end
) {
    /*
     * 返回 {最短距离, 路径}
     */
    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);
    dist[start] = 0;
    
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    pq.push({0, start});
    
    vector<bool> confirmed(n, false);
    
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        if (confirmed[u]) continue;
        confirmed[u] = true;
        
        if (u == end) break;  // 提前终止
        
        for (auto& [v, w] : graph[u]) {
            if (!confirmed[v] && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    
    // 重建路径
    if (dist[end] == INF) {
        return {-1, {}};
    }
    
    vector<int> path;
    for (int cur = end; cur != -1; cur = parent[cur]) {
        path.push_back(cur);
    }
    reverse(path.begin(), path.end());
    
    return {dist[end], path};
}

// 使用示例
int main() {
    int n = 5;
    vector<vector<pair<int, int>>> graph(n);
    
    // 添加边
    auto addEdge = [&](int u, int v, int w) {
        graph[u].push_back({v, w});
    };
    
    addEdge(0, 1, 4); addEdge(0, 2, 2);
    addEdge(1, 2, 1); addEdge(1, 3, 5);
    addEdge(2, 1, 1); addEdge(2, 3, 8); addEdge(2, 4, 10);
    addEdge(3, 4, 2);
    
    // 计算最短距离
    auto dist = dijkstra(n, graph, 0);
    for (int i = 0; i < n; i++) {
        cout << "dist[0][" << i << "] = " << dist[i] << endl;
    }
    
    // 求最短路径
    auto [distance, path] = dijkstraWithPath(n, graph, 0, 4);
    cout << "最短距离: " << distance << endl;
    cout << "路径: ";
    for (int v : path) cout << v << " ";
    cout << endl;
    
    return 0;
}
```

### 时间复杂度分析

| 实现方式 | 时间复杂度 | 空间复杂度 |
|---------|-----------|-----------|
| 朴素实现（邻接矩阵） | O(V²) | O(V²) |
| 优先队列 + 邻接表 | O((V + E) log V) | O(V + E) |
| 斐波那契堆（理论最优） | O(E + V log V) | O(V + E) |

::: warning 注意事项
1. **Dijkstra 不能处理负权边**：贪心策略在负权边下可能得到错误结果
2. **优先队列中的重复顶点**：使用 `confirmed` 数组避免重复处理
3. **大图优化**：使用 `visited` 数组而非从优先队列中删除元素
:::

### 典型应用

- **地图导航**：计算最短行车路线
- **网络路由**：数据包的最优转发路径
- **社交网络**：计算用户间的"距离"

---

## Bellman-Ford 算法

Bellman-Ford 算法可以处理**含有负权边**的图，并能检测负权环的存在。

### 算法原理

Bellman-Ford 算法基于**动态规划**思想：对所有边进行 V-1 轮松弛操作，每轮松弛保证至少确定一个顶点的最短距离。

**核心观察**：从起点到任意顶点的最短路径最多包含 V-1 条边（否则存在环）。

### 算法步骤

```mermaid
graph TD
    A[初始化: d[start] = 0, 其他 d = ∞] --> B[第 i = 1 轮松弛]
    B --> C[遍历所有边 u → v]
    C --> D{d[u] + w < d[v]?}
    D -->|是| E[更新 d[v] = d[u] + w]
    D -->|否| F[跳过]
    E --> G{还有边未处理?}
    F --> G
    G -->|是| C
    G -->|否| H{i < V-1?}
    H -->|是| B
    H -->|否| I[检测负环: 再松弛一轮]
```

### 代码实现

#### Python 实现

```python
from typing import List, Tuple

def bellman_ford(n: int, edges: List[Tuple[int, int, int]], 
                  start: int) -> Tuple[List[int], bool]:
    """
    Bellman-Ford 算法求单源最短路径
    
    :param n: 顶点数量
    :param edges: 边列表，每个元素为 (u, v, weight)
    :param start: 起点
    :return: (距离数组, 是否存在负环)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    
    # 进行 V-1 轮松弛
    for i in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        
        # 优化：本轮无更新则提前结束
        if not updated:
            break
    
    # 检测负权环：再进行一轮松弛
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
    
    return dist, has_negative_cycle


def bellman_ford_with_path(n: int, edges: List[Tuple[int, int, int]], 
                           start: int, end: int) -> Tuple[int, List[int], bool]:
    """
    Bellman-Ford 算法并返回路径
    
    :return: (最短距离, 路径, 是否存在负环)
    """
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    parent = [-1] * n
    
    # V-1 轮松弛
    for i in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break
    
    # 检测负环
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
    
    if has_negative_cycle or dist[end] == INF:
        return -1, [], has_negative_cycle
    
    # 重建路径
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    
    return int(dist[end]), path, False


# 使用示例
if __name__ == "__main__":
    n = 4
    edges = [
        (0, 1, 1),
        (1, 2, -1),
        (2, 3, -1),
        (3, 1, -1)  # 负环: 1 -> 2 -> 3 -> 1, 总权重 -3
    ]
    
    dist, has_cycle = bellman_ford(n, edges, 0)
    print(f"是否存在负环: {has_cycle}")
    print(f"距离数组: {dist}")
```

#### C++ 实现

```cpp
#include <vector>
#include <tuple>
using namespace std;

const long long INF = LLONG_MAX;

// Bellman-Ford 算法
pair<vector<long long>, bool> bellmanFord(
    int n, 
    const vector<tuple<int, int, int>>& edges, 
    int start
) {
    /*
     * 返回 {距离数组, 是否存在负环}
     */
    vector<long long> dist(n, INF);
    dist[start] = 0;
    
    // V-1 轮松弛
    for (int i = 0; i < n - 1; i++) {
        bool updated = false;
        for (auto& [u, v, w] : edges) {
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                updated = true;
            }
        }
        if (!updated) break;  // 提前终止
    }
    
    // 检测负环
    bool hasNegativeCycle = false;
    for (auto& [u, v, w] : edges) {
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            hasNegativeCycle = true;
            break;
        }
    }
    
    return {dist, hasNegativeCycle};
}

// 链式前向星优化版本
namespace Optimized {
    const int MAXN = 10005;
    const int MAXM = 50005;
    
    int head[MAXN], to[MAXM], nxt[MAXM], weight[MAXM], cnt;
    long long dist[MAXN];
    
    void init(int n) {
        for (int i = 0; i <= n; i++) head[i] = -1;
        cnt = 0;
    }
    
    void addEdge(int u, int v, int w) {
        to[cnt] = v;
        weight[cnt] = w;
        nxt[cnt] = head[u];
        head[u] = cnt++;
    }
    
    bool bellmanFord(int n, int start) {
        /*
         * 返回 true 表示存在负环
         */
        for (int i = 0; i <= n; i++) dist[i] = INF;
        dist[start] = 0;
        
        for (int i = 0; i < n - 1; i++) {
            bool updated = false;
            for (int u = 0; u < n; u++) {
                if (dist[u] == INF) continue;
                for (int e = head[u]; e != -1; e = nxt[e]) {
                    int v = to[e];
                    int w = weight[e];
                    if (dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                        updated = true;
                    }
                }
            }
            if (!updated) break;
        }
        
        // 检测负环
        for (int u = 0; u < n; u++) {
            if (dist[u] == INF) continue;
            for (int e = head[u]; e != -1; e = nxt[e]) {
                int v = to[e];
                int w = weight[e];
                if (dist[u] + w < dist[v]) {
                    return true;  // 存在负环
                }
            }
        }
        
        return false;
    }
}
```

### 负权环检测

**负权环**是指环上所有边权重之和为负的环。存在负权环时，最短路径无意义（可以无限减小）。

检测方法：进行第 V 轮松弛，如果仍能更新距离，则存在负权环。

```python
def find_negative_cycle(n: int, edges: List[Tuple[int, int, int]]) -> List[int]:
    """
    检测并返回负权环
    
    :return: 负权环上的顶点列表，无负环返回空列表
    """
    INF = float('inf')
    dist = [0] * n  # 初始化为 0，可以检测任意位置的负环
    parent = [-1] * n
    
    # V 轮松弛
    last_updated = -1
    for i in range(n):
        last_updated = -1
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                last_updated = v
    
    if last_updated == -1:
        return []  # 无负环
    
    # 回溯找到负环
    # 从 last_updated 开始回溯 V 步，确保进入环内
    for i in range(n):
        last_updated = parent[last_updated]
    
    # 收集环上的顶点
    cycle = []
    v = last_updated
    while True:
        cycle.append(v)
        v = parent[v];
        if v == last_updated:
            break
    cycle.reverse()
    
    return cycle
```

### 时间复杂度

- **时间复杂度**：O(V × E)
- **空间复杂度**：O(V)

虽然 Bellman-Ford 比Dijkstra慢，但能处理负权边和检测负环。

---

## SPFA 算法

SPFA（Shortest Path Faster Algorithm）是 Bellman-Ford 的队列优化版本，在大多数情况下效率更高。

### 算法原理

SPFA 的核心思想：只有被松弛成功的顶点才可能松弛其邻居，因此将这些顶点加入队列等待处理。

### 代码实现

#### Python 实现

```python
from collections import deque
from typing import List, Tuple

def spfa(n: int, edges: List[Tuple[int, int, int]], 
         start: int) -> Tuple[List[int], bool]:
    """
    SPFA 算法求单源最短路径
    
    :param n: 顶点数量
    :param edges: 边列表
    :param start: 起点
    :return: (距离数组, 是否存在负环)
    """
    # 构建邻接表
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    
    # 队列优化
    in_queue = [False] * n
    queue = deque([start])
    in_queue[start] = True
    
    # 记录入队次数，用于检测负环
    count = [0] * n
    count[start] = 1
    
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        for v, w in graph[u]:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    
                    # 入队次数 >= V，说明存在负环
                    if count[v] >= n:
                        return dist, True
    
    return dist, False


def spfa_slf_lll(n: int, edges: List[Tuple[int, int, int]], 
                  start: int) -> Tuple[List[int], bool]:
    """
    SPFA 的 SLF + LLL 优化版本
    - SLF (Small Label First): 小距离优先
    - LLL (Large Label Last): 大距离延后
    """
    from collections import deque
    
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0
    
    in_queue = [False] * n
    queue = deque([start])
    in_queue[start] = True
    count = [0] * n
    count[start] = 1
    
    total = 0  # 队列中距离总和，用于 LLL 优化
    
    while queue:
        # LLL 优化：如果队首距离 > 平均值，移到队尾
        avg = total / len(queue) if queue else 0
        while queue and dist[queue[0]] > avg:
            u = queue.popleft()
            queue.append(u)
        
        u = queue.popleft()
        in_queue[u] = False
        total -= dist[u]
        
        for v, w in graph[u]:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                
                if not in_queue[v]:
                    # SLF 优化：小距离插入队首
                    if queue and dist[v] < dist[queue[0]]:
                        queue.appendleft(v)
                    else:
                        queue.append(v)
                    
                    in_queue[v] = True
                    total += dist[v]
                    count[v] += 1
                    
                    if count[v] >= n:
                        return dist, True
    
    return dist, False


# 使用示例
if __name__ == "__main__":
    n = 4
    edges = [
        (0, 1, 2),
        (0, 2, 4),
        (1, 2, 1),
        (1, 3, 7),
        (2, 3, 3)
    ]
    
    dist, has_cycle = spfa(n, edges, 0)
    print(f"距离数组: {dist}")  # [0, 2, 3, 6]
    print(f"是否存在负环: {has_cycle}")
```

#### C++ 实现

```cpp
#include <vector>
#include <queue>
using namespace std;

// SPFA 算法
pair<vector<long long>, bool> spfa(
    int n, 
    const vector<vector<pair<int, int>>>& graph, 
    int start
) {
    /*
     * 返回 {距离数组, 是否存在负环}
     */
    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    dist[start] = 0;
    
    vector<bool> inQueue(n, false);
    vector<int> count(n, 0);  // 入队次数
    
    queue<int> q;
    q.push(start);
    inQueue[start] = true;
    count[start] = 1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inQueue[u] = false;
        
        for (auto& [v, w] : graph[u]) {
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                
                if (!inQueue[v]) {
                    q.push(v);
                    inQueue[v] = true;
                    count[v]++;
                    
                    // 检测负环
                    if (count[v] >= n) {
                        return {dist, true};
                    }
                }
            }
        }
    }
    
    return {dist, false};
}

// 链式前向星版本
namespace SPFA {
    const int MAXN = 10005;
    const int MAXM = 50005;
    
    int head[MAXN], to[MAXM], nxt[MAXM], weight[MAXM], cnt;
    long long dist[MAXN];
    bool inQueue[MAXN];
    int count[MAXN];
    
    void init(int n) {
        for (int i = 0; i <= n; i++) {
            head[i] = -1;
            inQueue[i] = false;
            count[i] = 0;
        }
        cnt = 0;
    }
    
    void addEdge(int u, int v, int w) {
        to[cnt] = v;
        weight[cnt] = w;
        nxt[cnt] = head[u];
        head[u] = cnt++;
    }
    
    bool spfa(int n, int start) {
        /*
         * 返回 true 表示存在负环
         */
        for (int i = 0; i <= n; i++) dist[i] = LLONG_MAX;
        dist[start] = 0;
        
        queue<int> q;
        q.push(start);
        inQueue[start] = true;
        count[start] = 1;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            inQueue[u] = false;
            
            for (int e = head[u]; e != -1; e = nxt[e]) {
                int v = to[e];
                int w = weight[e];
                
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    
                    if (!inQueue[v]) {
                        q.push(v);
                        inQueue[v] = true;
                        count[v]++;
                        
                        if (count[v] >= n) {
                            return true;  // 存在负环
                        }
                    }
                }
            }
        }
        
        return false;
    }
}
```

### 时间复杂度

| 情况 | 时间复杂度 |
|------|-----------|
| 平均情况 | O(kE)，k 是一个小常数 |
| 最坏情况 | O(VE)，退化为 Bellman-Ford |

### 适用场景

- ✅ 图中含有负权边
- ✅ 图比较稀疏
- ✅ 需要检测负环
- ❌ 可能被特殊构造的数据卡到最坏情况（慎用于竞赛）

::: warning 竞赛注意
SPFA 在某些特殊数据下会退化，竞赛中如果确定无负权边，优先使用 Dijkstra。
:::

---

## Floyd 算法

Floyd 算法（Floyd-Warshall）用于求解**多源最短路径**问题，可以一次计算任意两点间的最短距离。

### 算法原理

Floyd 算法基于**动态规划**思想：

设 $d_{k}[i][j]$ 表示只允许经过顶点 $\{1, 2, ..., k\}$ 时，从 $i$ 到 $j$ 的最短距离。

**状态转移方程**：
$$d_{k}[i][j] = \min(d_{k-1}[i][j], d_{k-1}[i][k] + d_{k-1}[k][j])$$

即：要么不经过 $k$，要么经过 $k$（此时路径为 $i \to k \to j$）。

### 动画演示

<FloydAnimation />

### 代码实现

#### Python 实现

```python
from typing import List

def floyd(n: int, edges: List[tuple]) -> List[List[int]]:
    """
    Floyd 算法求多源最短路径
    
    :param n: 顶点数量
    :param edges: 边列表，每个元素为 (u, v, weight)
    :return: dist[i][j] 表示从 i 到 j 的最短距离
    """
    INF = float('inf')
    
    # 初始化距离矩阵
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0  # 自己到自己的距离为 0
    
    # 读入边
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # 处理重边
        # 如果是无向图，添加: dist[v][u] = min(dist[v][u], w)
    
    # Floyd 核心代码：三重循环
    for k in range(n):          # 枚举中间点
        for i in range(n):      # 枚举起点
            for j in range(n):  # 枚举终点
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist


def floyd_with_path(n: int, edges: List[tuple]) -> tuple:
    """
    Floyd 算法并记录路径
    
    :return: (距离矩阵, next_node)
             next_node[i][j] 表示从 i 到 j 最短路径上的下一个节点
    """
    INF = float('inf')
    
    dist = [[INF] * n for _ in range(n)]
    next_node = [[-1] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
        next_node[i][i] = i
    
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
            next_node[u][v] = v
    
    # Floyd 算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
    
    return dist, next_node


def get_path(next_node: List[List[int]], i: int, j: int) -> List[int]:
    """
    根据路径记录矩阵重建路径
    """
    if next_node[i][j] == -1:
        return []
    
    path = [i]
    while i != j:
        i = next_node[i][j]
        path.append(i)
    
    return path


def floyd_detect_negative_cycle(n: int, edges: List[tuple]) -> tuple:
    """
    Floyd 算法检测负环
    
    :return: (距离矩阵, 是否存在负环)
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    
    # Floyd
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # 检测负环：如果 dist[i][i] < 0，则存在经过 i 的负环
    has_negative_cycle = any(dist[i][i] < 0 for i in range(n))
    
    return dist, has_negative_cycle


# 使用示例
if __name__ == "__main__":
    n = 4
    edges = [
        (0, 1, 4),
        (0, 2, 2),
        (1, 2, 1),
        (1, 3, 5),
        (2, 3, 8)
    ]
    
    dist = floyd(n, edges)
    
    print("最短距离矩阵:")
    print("    0    1    2    3")
    for i in range(n):
        print(f"{i}: ", end="")
        for j in range(n):
            if dist[i][j] == float('inf'):
                print("INF  ", end="")
            else:
                print(f"{dist[i][j]:<5}", end="")
        print()
```

#### C++ 实现

```cpp
#include <vector>
#include <iostream>
using namespace std;

// Floyd 算法
vector<vector<long long>> floyd(int n, const vector<vector<pair<int, int>>>& graph) {
    /*
     * graph[u] = {(v, weight), ...}
     * 返回距离矩阵
     */
    const long long INF = LLONG_MAX;
    vector<vector<long long>> dist(n, vector<long long>(n, INF));
    
    // 初始化
    for (int i = 0; i < n; i++) {
        dist[i][i] = 0;
    }
    
    // 读入边
    for (int u = 0; u < n; u++) {
        for (auto& [v, w] : graph[u]) {
            dist[u][v] = min(dist[u][v], (long long)w);
        }
    }
    
    // Floyd 核心代码
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    
    return dist;
}

// 邻接矩阵版本的 Floyd（竞赛常用）
void floyd_matrix(int n, long long dist[][100], int INF_VAL) {
    /*
     * 直接在 dist 矩阵上操作
     * INF_VAL 是无穷大的值（如 0x3f3f3f3f3f3f3f3f）
     */
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INF_VAL && dist[k][j] != INF_VAL) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

// 带路径记录的 Floyd
vector<vector<int>> floydWithPath(
    int n, 
    vector<vector<long long>>& dist
) {
    /*
     * 返回 next_node 矩阵
     * next_node[i][j] 表示 i 到 j 路径上的下一个节点
     */
    vector<vector<int>> nextNode(n, vector<int>(n, -1));
    
    // 初始化 next_node
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i][j] < LLONG_MAX) {
                nextNode[i][j] = j;
            }
        }
        nextNode[i][i] = i;
    }
    
    // Floyd
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != LLONG_MAX && dist[k][j] != LLONG_MAX) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        nextNode[i][j] = nextNode[i][k];
                    }
                }
            }
        }
    }
    
    return nextNode;
}

// 重建路径
vector<int> getPath(const vector<vector<int>>& nextNode, int i, int j) {
    if (nextNode[i][j] == -1) return {};
    
    vector<int> path = {i};
    while (i != j) {
        i = nextNode[i][j];
        path.push_back(i);
    }
    return path;
}

// 检测负环
bool hasNegativeCycle(int n, const vector<vector<long long>>& dist) {
    for (int i = 0; i < n; i++) {
        if (dist[i][i] < 0) return true;
    }
    return false;
}
```

### 时间复杂度

- **时间复杂度**：O(V³)
- **空间复杂度**：O(V²)

### 适用场景

- ✅ 需要求任意两点间最短距离
- ✅ 图的顶点数较少（V < 500）
- ✅ 需要检测负环
- ✅ 图的边可能动态变化

---

## 算法对比

### 综合对比表

| 算法 | 问题类型 | 时间复杂度 | 空间复杂度 | 负权边 | 负环检测 |
|------|---------|-----------|-----------|--------|---------|
| BFS | 单源无权 | O(V + E) | O(V) | ❌ | ❌ |
| Dijkstra | 单源非负权 | O((V+E)logV) | O(V + E) | ❌ | ❌ |
| Bellman-Ford | 单源有负权 | O(V × E) | O(V) | ✅ | ✅ |
| SPFA | 单源有负权 | O(kE)~O(VE) | O(V) | ✅ | ✅ |
| Floyd | 多源 | O(V³) | O(V²) | ✅ | ✅ |

### 选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                      最短路径算法选择                         │
├─────────────────────────────────────────────────────────────┤
│  需要求所有点对最短路径？                                     │
│  ├── 是 → 顶点数 < 500？                                     │
│  │         ├── 是 → Floyd 算法                               │
│  │         └── 否 → 对每个点运行 Dijkstra                     │
│  │                                                           │
│  └── 否 → 单源最短路径                                       │
│           ├── 无权图 → BFS                                   │
│           ├── 非负权图 → Dijkstra                            │
│           └── 有负权边                                       │
│                      ├── 需要稳定复杂度 → Bellman-Ford       │
│                      └── 期望更快 → SPFA                     │
└─────────────────────────────────────────────────────────────┘
```

### 竞赛技巧

::: tip 常用优化
1. **Dijkstra**：使用 `visited` 数组而非从优先队列删除
2. **SPFA**：SLF（小距离优先）+ LLL（大距离延后）优化
3. **Floyd**：注意 `dist[i][k] + dist[k][j]` 的溢出问题
4. **通用**：使用链式前向星存储图
:::

---

## 练习建议

### 基础题目

| 算法 | LeetCode | 难度 |
|------|----------|------|
| Dijkstra | 743. 网络延迟时间 | 中等 |
| Dijkstra | 1514. 概率最大的路径 | 中等 |
| Dijkstra | 1631. 最小体力消耗路径 | 中等 |
| Bellman-Ford | 787. K 站中转内最便宜的航班 | 中等 |
| Floyd | 1334. 阈值距离内邻居最少的城市 | 中等 |

### 进阶题目

| 主题 | LeetCode | 难度 |
|------|----------|------|
| 最短路径 + 状态压缩 | 847. 访问所有节点的最短路径 | 困难 |
| 最短路径 + 贪心 | 778. 水位上升的泳池中游泳 | 困难 |
| 最短路径变种 | 2642. 设计可以求最短路径的图类 | 困难 |

---

## 小结

本节介绍了最短路径问题的核心算法：

1. **Dijkstra**：非负权图的单源最短路径首选，优先队列优化后效率高
2. **Bellman-Ford**：可处理负权边，能检测负环，时间复杂度较高
3. **SPFA**：Bellman-Ford 的队列优化，平均效率更好但可能退化
4. **Floyd**：多源最短路径的经典算法，简单但时间复杂度高

选择合适算法的关键：**问题类型（单源/多源）+ 边权特性（无权/非负/含负）**。
