# 高级搜索算法

高级搜索算法是在基础搜索（DFS、BFS）之上的优化与扩展，通过引入**启发式信息**、**迭代加深**、**双向搜索**等技术，大幅提升搜索效率。这些算法在路径规划、游戏AI、约束满足问题等领域有广泛应用。

📌 **核心思想**

基础搜索的问题在于"盲目"——没有目标导向地遍历状态空间。高级搜索的核心是让搜索"更聪明"：
- **启发式搜索**：利用问题特性引导搜索方向
- **迭代加深**：在深度优先和广度优先间取得平衡
- **双向搜索**：从两端同时搜索，指数级减少搜索空间

---

## A* 算法

### 启发式搜索原理

A*（A-Star）算法是最著名的启发式搜索算法，它结合了 Dijkstra 算法的最优性保证和贪婪最佳优先搜索的高效性。

<AStarSearch />

**核心公式**：

$$f(n) = g(n) + h(n)$$

其中：
- $g(n)$：从起点到节点 $n$ 的**实际代价**（已知）
- $h(n)$：从节点 $n$ 到终点的**启发式估计代价**（预测）
- $f(n)$：经过节点 $n$ 的估计总代价

💡 **A* 的优势**

| 特性 | Dijkstra | 贪婪最佳优先 | A* |
|------|----------|--------------|-----|
| 最优性保证 | ✅ 保证 | ❌ 不保证 | ✅ 保证（h可采纳） |
| 搜索效率 | 较慢（盲目） | 快但可能绕路 | 高效且最优 |
| 适用场景 | 无权图最短路 | 快速近似 | 路径规划、游戏AI |

### 估价函数设计

估价函数 $h(n)$ 的设计直接影响 A* 的效率和正确性。

📌 **可采纳性（Admissibility）**

如果 $h(n)$ **永远不高估**实际代价（即 $h(n) \leq h^*(n)$，$h^*$ 是真实代价），则称 $h$ 是**可采纳的**。可采纳的启发函数保证 A* 找到最优解。

📌 **一致性（Consistency）**

对于所有节点 $n$ 和其后继 $n'$，满足：

$$h(n) \leq c(n, n') + h(n')$$

一致性保证了：
- A* 不会重复扩展已关闭的节点
- 第一个到达目标的路径就是最优路径

**常用启发函数**：

| 场景 | 启发函数 | 公式 | 可采纳性 |
|------|----------|------|----------|
| 网格（四方向） | 曼哈顿距离 | $\|x_1-x_2\| + \|y_1-y_2\|$ | ✅ |
| 网格（八方向） | 切比雪夫距离 | $\max(\|x_1-x_2\|, \|y_1-y_2\|)$ | ✅ |
| 任意两点 | 欧几里得距离 | $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ | ✅ |
| 八数码 | 错位距离 | 错位格子数量 | ✅ |
| 八数码 | 曼哈顿距离 | 每个格子到目标位置距离之和 | ✅ |

### A* 算法实现

::: code-group
```cpp [C++]
#include <bits/stdc++.h>
using namespace std;

/**
 * A* 算法实现 - 网格最短路径
 * 
 * 核心数据结构：
 * - openList：优先队列，按 f 值排序
 * - closedList：已访问节点集合
 * - gScore：实际代价
 * - fScore：估计总代价
 */

// 节点结构
struct Node {
    int x, y;           // 坐标
    int g, h, f;        // 代价：g=实际, h=启发, f=总估计
    Node* parent;       // 父节点指针
    
    Node(int x, int y) : x(x), y(y), g(INT_MAX), h(0), f(INT_MAX), parent(nullptr) {}
};

// 比较器（优先队列需要最小堆）
struct CompareNode {
    bool operator()(const Node* a, const Node* b) const {
        return a->f > b->f;  // f 值小的优先
    }
};

class AStar {
private:
    int rows, cols;
    vector<vector<int>> grid;      // 0=空地, 1=障碍
    vector<vector<Node*>> nodes;   // 节点矩阵
    
    // 四个方向
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    
public:
    AStar(vector<vector<int>>& grid) : grid(grid) {
        rows = grid.size();
        cols = grid[0].size();
        // 初始化节点矩阵
        nodes.resize(rows, vector<Node*>(cols, nullptr));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                nodes[i][j] = new Node(i, j);
            }
        }
    }
    
    ~AStar() {
        for (auto& row : nodes) {
            for (auto node : row) {
                delete node;
            }
        }
    }
    
    // 曼哈顿距离启发函数
    int heuristic(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }
    
    /**
     * A* 搜索主函数
     * @param startX, startY 起点坐标
     * @param endX, endY 终点坐标
     * @return 最短路径，若不存在返回空
     */
    vector<pair<int, int>> search(int startX, int startY, int endX, int endY) {
        // 检查起点终点有效性
        if (grid[startX][startY] == 1 || grid[endX][endY] == 1) {
            return {};
        }
        
        // 优先队列（开放列表）
        priority_queue<Node*, vector<Node*>, CompareNode> openList;
        // 已访问集合（关闭列表）
        set<pair<int, int>> closedList;
        
        // 初始化起点
        Node* start = nodes[startX][startY];
        start->g = 0;
        start->h = heuristic(startX, startY, endX, endY);
        start->f = start->g + start->h;
        openList.push(start);
        
        while (!openList.empty()) {
            // 取出 f 值最小的节点
            Node* current = openList.top();
            openList.pop();
            
            // 到达终点
            if (current->x == endX && current->y == endY) {
                return reconstructPath(current);
            }
            
            // 已处理过则跳过
            if (closedList.count({current->x, current->y})) {
                continue;
            }
            closedList.insert({current->x, current->y});
            
            // 探索四个方向
            for (int i = 0; i < 4; i++) {
                int nx = current->x + dx[i];
                int ny = current->y + dy[i];
                
                // 边界检查
                if (nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
                // 障碍物检查
                if (grid[nx][ny] == 1) continue;
                // 已访问检查
                if (closedList.count({nx, ny})) continue;
                
                // 计算新的 g 值
                int newG = current->g + 1;  // 移动代价为 1
                
                Node* neighbor = nodes[nx][ny];
                
                // 找到更优路径
                if (newG < neighbor->g) {
                    neighbor->g = newG;
                    neighbor->h = heuristic(nx, ny, endX, endY);
                    neighbor->f = neighbor->g + neighbor->h;
                    neighbor->parent = current;
                    openList.push(neighbor);
                }
            }
        }
        
        return {};  // 无路径
    }
    
private:
    // 回溯构建路径
    vector<pair<int, int>> reconstructPath(Node* end) {
        vector<pair<int, int>> path;
        Node* current = end;
        while (current != nullptr) {
            path.push_back({current->x, current->y});
            current = current->parent;
        }
        reverse(path.begin(), path.end());
        return path;
    }
};

int main() {
    // 示例：网格搜索
    vector<vector<int>> grid = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    
    AStar astar(grid);
    auto path = astar.search(0, 0, 4, 4);
    
    if (path.empty()) {
        cout << "无法到达终点" << endl;
    } else {
        cout << "最短路径（长度 " << path.size() - 1 << "）：" << endl;
        for (auto& [x, y] : path) {
            cout << "(" << x << "," << y << ") ";
        }
        cout << endl;
    }
    
    return 0;
}
```

```python [Python]
import heapq
from typing import List, Tuple, Optional, Set

class AStar:
    """
    A* 算法实现 - 网格最短路径
    
    核心数据结构：
    - open_list：最小堆，按 f 值排序
    - closed_set：已访问节点集合
    - g_score：实际代价字典
    - f_score：估计总代价字典
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        初始化 A* 搜索器
        
        Args:
            grid: 网格地图，0=空地，1=障碍
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 四个方向
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        曼哈顿距离启发函数
        
        Args:
            (x1, y1): 当前节点坐标
            (x2, y2): 目标节点坐标
        
        Returns:
            启发式估计距离
        """
        return abs(x1 - x2) + abs(y1 - y2)
    
    def search(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* 搜索主函数
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
        
        Returns:
            最短路径列表，若不存在返回 None
        """
        start_x, start_y = start
        end_x, end_y = end
        
        # 检查起点终点有效性
        if self.grid[start_x][start_y] == 1 or self.grid[end_x][end_y] == 1:
            return None
        
        # 开放列表：最小堆 (f, g, x, y)
        # g 用于打破平局，保证先扩展 g 值小的（更接近最优）
        open_list = []
        # 已访问集合
        closed_set: Set[Tuple[int, int]] = set()
        # 代价记录
        g_score = {start: 0}
        f_score = {start: self.heuristic(start_x, start_y, end_x, end_y)}
        # 父节点记录（用于回溯路径）
        parent = {}
        
        # 起点入堆
        heapq.heappush(open_list, (f_score[start], 0, start_x, start_y))
        
        while open_list:
            # 取出 f 值最小的节点
            _, _, current_x, current_y = heapq.heappop(open_list)
            current = (current_x, current_y)
            
            # 已处理过则跳过
            if current in closed_set:
                continue
            
            # 到达终点
            if current == end:
                return self._reconstruct_path(parent, current)
            
            closed_set.add(current)
            
            # 探索四个方向
            for dx, dy in self.directions:
                nx, ny = current_x + dx, current_y + dy
                neighbor = (nx, ny)
                
                # 边界检查
                if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                    continue
                # 障碍物检查
                if self.grid[nx][ny] == 1:
                    continue
                # 已访问检查
                if neighbor in closed_set:
                    continue
                
                # 计算新的 g 值
                new_g = g_score[current] + 1
                
                # 找到更优路径
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    f_score[neighbor] = new_g + self.heuristic(nx, ny, end_x, end_y)
                    parent[neighbor] = current
                    heapq.heappush(open_list, (f_score[neighbor], new_g, nx, ny))
        
        return None  # 无路径
    
    def _reconstruct_path(self, parent: dict, end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """回溯构建路径"""
        path = [end]
        current = end
        while current in parent:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path


# 使用示例
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    astar = AStar(grid)
    path = astar.search((0, 0), (4, 4))
    
    if path is None:
        print("无法到达终点")
    else:
        print(f"最短路径（长度 {len(path) - 1}）：")
        print(" -> ".join(map(str, path)))
```
:::

### 八数码问题

八数码问题是 A* 算法的经典应用。3×3 的棋盘上有 1-8 八个数字和一个空格，目标是将任意初始状态移动到目标状态。

::: code-group
```cpp [C++]
#include <bits/stdc++.h>
using namespace std;

/**
 * 八数码问题 - A* 求解
 * 
 * 状态表示：字符串 "123456780" 表示目标状态
 * 空格用 '0' 表示
 */

// 目标状态
const string GOAL = "123456780";

// 计算启发函数（曼哈顿距离）
int heuristic(const string& state) {
    int distance = 0;
    for (int i = 0; i < 9; i++) {
        if (state[i] != '0') {
            int val = state[i] - '1';  // 数字 1-8 对应位置 0-7
            int target_row = val / 3;
            int target_col = val % 3;
            int current_row = i / 3;
            int current_col = i % 3;
            distance += abs(current_row - target_row) + abs(current_col - target_col);
        }
    }
    return distance;
}

// 获取空格位置
int findZero(const string& state) {
    return state.find('0');
}

// 移动方向
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

/**
 * A* 求解八数码
 * @param start 初始状态
 * @return 最少移动步数，无解返回 -1
 */
int solve8Puzzle(string start) {
    if (start == GOAL) return 0;
    
    // 优先队列：(f, g, state)
    priority_queue<tuple<int, int, string>, 
                   vector<tuple<int, int, string>>, 
                   greater<tuple<int, int, string>>> pq;
    
    // 已访问集合
    unordered_map<string, int> visited;  // state -> g 值
    
    // 初始化
    int h = heuristic(start);
    pq.push({h, 0, start});
    visited[start] = 0;
    
    while (!pq.empty()) {
        auto [f, g, state] = pq.top();
        pq.pop();
        
        // 找到目标
        if (state == GOAL) return g;
        
        // 剪枝：已有更优路径
        if (g > visited[state]) continue;
        
        // 找到空格位置
        int pos = findZero(state);
        int row = pos / 3, col = pos % 3;
        
        // 尝试四个方向移动
        for (int i = 0; i < 4; i++) {
            int nrow = row + dx[i];
            int ncol = col + dy[i];
            
            if (nrow >= 0 && nrow < 3 && ncol >= 0 && ncol < 3) {
                int npos = nrow * 3 + ncol;
                
                // 交换空格和数字
                string new_state = state;
                swap(new_state[pos], new_state[npos]);
                
                int new_g = g + 1;
                
                // 新状态或更优路径
                if (visited.find(new_state) == visited.end() || new_g < visited[new_state]) {
                    visited[new_state] = new_g;
                    int new_h = heuristic(new_state);
                    pq.push({new_g + new_h, new_g, new_state});
                }
            }
        }
    }
    
    return -1;  // 无解
}

// 判断是否有解
bool isSolvable(string state) {
    int inversions = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (state[i] != '0' && state[j] != '0' && state[i] > state[j]) {
                inversions++;
            }
        }
    }
    return inversions % 2 == 0;  // 逆序对数为偶数才有解
}

int main() {
    string start;
    cout << "输入初始状态（9个数字，空格用0表示）：" << endl;
    cin >> start;
    
    if (start.length() != 9) {
        cout << "输入格式错误！" << endl;
        return 1;
    }
    
    if (!isSolvable(start)) {
        cout << "该状态无解！" << endl;
        return 0;
    }
    
    int steps = solve8Puzzle(start);
    cout << "最少移动步数：" << steps << endl;
    
    return 0;
}
```

```python [Python]
import heapq
from typing import Optional

def solve_8_puzzle(start: str) -> Optional[int]:
    """
    A* 求解八数码问题
    
    Args:
        start: 初始状态，如 "123456780"
    
    Returns:
        最少移动步数，无解返回 None
    """
    GOAL = "123456780"
    
    if start == GOAL:
        return 0
    
    # 移动方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(state: str) -> int:
        """计算曼哈顿距离启发函数"""
        distance = 0
        for i, ch in enumerate(state):
            if ch != '0':
                val = int(ch) - 1  # 数字 1-8 对应位置 0-7
                target_row, target_col = val // 3, val % 3
                current_row, current_col = i // 3, i % 3
                distance += abs(current_row - target_row) + abs(current_col - target_col)
        return distance
    
    def get_neighbors(state: str) -> list:
        """获取所有可能的下一步状态"""
        pos = state.index('0')
        row, col = pos // 3, pos % 3
        neighbors = []
        
        for dr, dc in directions:
            nrow, ncol = row + dr, col + dc
            if 0 <= nrow < 3 and 0 <= ncol < 3:
                npos = nrow * 3 + ncol
                # 交换空格和数字
                state_list = list(state)
                state_list[pos], state_list[npos] = state_list[npos], state_list[pos]
                neighbors.append(''.join(state_list))
        
        return neighbors
    
    # A* 搜索
    # 优先队列：(f, g, state)
    open_list = [(heuristic(start), 0, start)]
    visited = {start: 0}
    
    while open_list:
        f, g, state = heapq.heappop(open_list)
        
        # 找到目标
        if state == GOAL:
            return g
        
        # 剪枝：已有更优路径
        if g > visited.get(state, float('inf')):
            continue
        
        for neighbor in get_neighbors(state):
            new_g = g + 1
            
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(open_list, (new_f, new_g, neighbor))
    
    return None  # 无解


def is_solvable(state: str) -> bool:
    """判断八数码是否有解"""
    inversions = 0
    for i in range(9):
        for j in range(i + 1, 9):
            if state[i] != '0' and state[j] != '0' and state[i] > state[j]:
                inversions += 1
    return inversions % 2 == 0


# 使用示例
if __name__ == "__main__":
    start = input("输入初始状态（9个数字，空格用0表示）：")
    
    if len(start) != 9:
        print("输入格式错误！")
    elif not is_solvable(start):
        print("该状态无解！")
    else:
        steps = solve_8_puzzle(start)
        print(f"最少移动步数：{steps}")
```
:::

💡 **八数码问题的可解性**

并非所有八数码状态都能达到目标状态。判断方法：
- 计算状态的**逆序对数**（不包括空格）
- 逆序对数为**偶数**则有解，**奇数**则无解

---

## 迭代加深搜索（IDS）

### 算法原理

迭代加深搜索（Iterative Deepening Search，IDS）结合了 DFS 的空间效率和 BFS 的完整性。

📌 **核心思想**

```
深度限制 d = 0, 1, 2, 3, ...
每次执行深度限制为 d 的 DFS
如果找到解则返回，否则 d++
```

**为什么"重复"搜索效率不低？**

虽然看起来重复搜索了很多节点，但树形结构的特点使得：
- 第 $k$ 层的节点数约为 $b^k$（$b$ 是分支因子）
- 前面所有层的节点数之和约为 $\frac{b^k - 1}{b - 1} \approx \frac{b^k}{b - 1}$
- 重复搜索的代价约为 $\frac{1}{b-1}$，当 $b$ 较大时可忽略

### 与 DFS/BFS 的对比

| 特性 | DFS | BFS | IDS |
|------|-----|-----|-----|
| 时间复杂度 | $O(b^d)$ | $O(b^d)$ | $O(b^d)$ |
| 空间复杂度 | $O(bd)$ | $O(b^d)$ | $O(bd)$ |
| 最优性 | ❌ | ✅ | ✅（代价一致） |
| 完备性 | ❌（可能无限深） | ✅ | ✅ |
| 适用场景 | 深度有限 | 内存充足 | 内存受限 |

💡 **IDS 的优势**：
- 像 BFS 一样保证找到最短解
- 像 DFS 一样节省内存
- 适合搜索深度未知的问题

### IDA* 算法

IDA*（Iterative Deepening A*）将迭代加深与 A* 结合，是解决状态空间搜索问题的利器。

::: code-group
```cpp [C++]
#include <bits/stdc++.h>
using namespace std;

/**
 * IDA* 算法实现 - 八数码问题
 * 
 * 思想：用 f 值作为"阈值"进行迭代加深
 * - 每次深度受限 DFS，但限制是 f 值而非深度
 * - 如果某节点 f > 阈值，剪枝
 * - 本轮无解则增大阈值，重新搜索
 */

const string GOAL = "123456780";
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

// 曼哈顿距离启发函数
int heuristic(const string& state) {
    int distance = 0;
    for (int i = 0; i < 9; i++) {
        if (state[i] != '0') {
            int val = state[i] - '1';
            int target_row = val / 3, target_col = val % 3;
            int current_row = i / 3, current_col = i % 3;
            distance += abs(current_row - target_row) + abs(current_col - target_col);
        }
    }
    return distance;
}

/**
 * 深度受限 DFS
 * @param state 当前状态
 * @param g 已走步数
 * @param limit f 值阈值
 * @param last_move 上一步移动方向（避免来回走）
 * @return 找到解返回步数，否则返回超过阈值的最小 f 值
 */
int dfs_ida(string& state, int g, int limit, int last_move, int& min_exceed) {
    int h = heuristic(state);
    int f = g + h;
    
    // 超过阈值，记录最小超限值
    if (f > limit) {
        min_exceed = min(min_exceed, f);
        return -1;
    }
    
    // 找到目标
    if (h == 0) return g;
    
    int pos = state.find('0');
    int row = pos / 3, col = pos % 3;
    
    for (int i = 0; i < 4; i++) {
        // 避免来回移动
        if (last_move != -1 && (i ^ 1) == last_move) continue;
        
        int nrow = row + dx[i];
        int ncol = col + dy[i];
        
        if (nrow >= 0 && nrow < 3 && ncol >= 0 && ncol < 3) {
            int npos = nrow * 3 + ncol;
            
            // 交换
            swap(state[pos], state[npos]);
            
            int result = dfs_ida(state, g + 1, limit, i, min_exceed);
            
            if (result != -1) {
                return result;  // 找到解
            }
            
            // 回溯
            swap(state[pos], state[npos]);
        }
    }
    
    return -1;
}

/**
 * IDA* 主函数
 */
int idaStar(string start) {
    if (start == GOAL) return 0;
    
    int limit = heuristic(start);
    
    while (limit < 100) {  // 防止无限循环
        int min_exceed = INT_MAX;
        string state = start;
        
        int result = dfs_ida(state, 0, limit, -1, min_exceed);
        
        if (result != -1) {
            return result;
        }
        
        // 更新阈值为最小的超限值
        limit = min_exceed;
    }
    
    return -1;
}

int main() {
    string start;
    cout << "输入初始状态：" << endl;
    cin >> start;
    
    int steps = idaStar(start);
    if (steps != -1) {
        cout << "最少步数：" << steps << endl;
    } else {
        cout << "无解或搜索超时" << endl;
    }
    
    return 0;
}
```

```python [Python]
from typing import Optional, Tuple

def ida_star(start: str) -> Optional[int]:
    """
    IDA* 算法求解八数码问题
    
    思想：用 f 值作为"阈值"进行迭代加深
    - 每次深度受限 DFS，但限制是 f 值而非深度
    - 如果某节点 f > 阈值，剪枝
    - 本轮无解则增大阈值，重新搜索
    """
    GOAL = "123456780"
    
    if start == GOAL:
        return 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}  # 相反方向
    
    def heuristic(state: str) -> int:
        """曼哈顿距离"""
        distance = 0
        for i, ch in enumerate(state):
            if ch != '0':
                val = int(ch) - 1
                target_row, target_col = val // 3, val % 3
                current_row, current_col = i // 3, i % 3
                distance += abs(current_row - target_row) + abs(current_col - target_col)
        return distance
    
    def dfs(state: str, g: int, limit: int, last_move: int) -> Tuple[Optional[int], int]:
        """
        深度受限 DFS
        
        Returns:
            (结果, 最小超限值)
            结果：找到解返回步数，否则返回 None
        """
        h = heuristic(state)
        f = g + h
        
        if f > limit:
            return None, f
        
        if h == 0:
            return g, f
        
        pos = state.index('0')
        row, col = pos // 3, pos % 3
        min_exceed = float('inf')
        
        for i, (dr, dc) in enumerate(directions):
            # 避免来回移动
            if last_move != -1 and i == opposite[last_move]:
                continue
            
            nrow, ncol = row + dr, col + dc
            if 0 <= nrow < 3 and 0 <= ncol < 3:
                npos = nrow * 3 + ncol
                
                # 交换
                state_list = list(state)
                state_list[pos], state_list[npos] = state_list[npos], state_list[pos]
                new_state = ''.join(state_list)
                
                result, exceed = dfs(new_state, g + 1, limit, i)
                
                if result is not None:
                    return result, exceed
                
                min_exceed = min(min_exceed, exceed)
        
        return None, min_exceed
    
    # IDA* 主循环
    limit = heuristic(start)
    
    while limit < 100:
        result, exceed = dfs(start, 0, limit, -1)
        
        if result is not None:
            return result
        
        limit = exceed
    
    return None


# 使用示例
if __name__ == "__main__":
    start = input("输入初始状态：")
    steps = ida_star(start)
    
    if steps is not None:
        print(f"最少步数：{steps}")
    else:
        print("无解或搜索超时")
```
:::

---

## 双向搜索深化

### 应用场景

双向搜索在 BFS 笔记中已介绍基础概念，这里深化应用场景和技巧。

📌 **适用条件**：
1. **已知起点和终点**
2. **状态转移可逆**（从 A 能到 B，则从 B 也能到 A）
3. **分支因子适中**（太大或太小优化效果有限）

📌 **经典应用场景**：

| 问题类型 | 示例 | 特点 |
|----------|------|------|
| 最短路径 | 单词接龙 | 状态空间大，解较短 |
| 状态转换 | 数字变换 | 两端状态明确 |
| 游戏搜索 | 走迷宫 | 可从两端同时扩展 |

### 双向搜索优化技巧

::: code-group
```cpp [C++]
/**
 * 双向 BFS 优化技巧
 * 
 * 核心优化点：
 * 1. 始终扩展较小的集合
 * 2. 使用哈希表加速查找
 * 3. 提前终止检测
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * 双向 BFS - 通用模板
 */
class BidirectionalBFS {
public:
    // 状态转移函数类型
    using GetNeighbors = function<vector<string>(const string&)>;
    
    /**
     * 双向 BFS 搜索
     * @param start 起点
     * @param end 终点
     * @param getNeighbors 获取邻居状态的函数
     * @return 最短路径长度
     */
    static int search(const string& start, const string& end,
                      GetNeighbors getNeighbors) {
        if (start == end) return 0;
        
        // 双端队列和访问标记
        unordered_set<string> beginSet{start};
        unordered_set<string> endSet{end};
        unordered_set<string> visited{start, end};
        
        int level = 0;
        
        while (!beginSet.empty() && !endSet.empty()) {
            // 优化1：始终扩展较小的集合
            if (beginSet.size() > endSet.size()) {
                swap(beginSet, endSet);
            }
            
            unordered_set<string> nextSet;
            
            for (const string& word : beginSet) {
                // 获取所有邻居
                for (const string& neighbor : getNeighbors(word)) {
                    // 优化2：在另一端找到则返回
                    if (endSet.count(neighbor)) {
                        return level + 1;
                    }
                    
                    // 未访问过
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        nextSet.insert(neighbor);
                    }
                }
            }
            
            beginSet = nextSet;
            level++;
        }
        
        return -1;  // 无路径
    }
};

/**
 * 示例：单词接龙
 */
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    
    if (!wordSet.count(endWord)) return 0;
    
    auto getNeighbors = [&](const string& word) {
        vector<string> neighbors;
        for (int i = 0; i < word.size(); i++) {
            char original = word[i];
            for (char c = 'a'; c <= 'z'; c++) {
                if (c == original) continue;
                string newWord = word;
                newWord[i] = c;
                if (wordSet.count(newWord)) {
                    neighbors.push_back(newWord);
                }
            }
        }
        return neighbors;
    };
    
    return BidirectionalBFS::search(beginWord, endWord, getNeighbors);
}

/**
 * 双向 BFS + 路径重建
 */
vector<string> findPath(string beginWord, string endWord, 
                        vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (!wordSet.count(endWord)) return {};
    
    if (beginWord == endWord) return {beginWord};
    
    // 记录每个状态的父节点（双向）
    unordered_map<string, string> parentFromBegin;
    unordered_map<string, string> parentFromEnd;
    
    unordered_set<string> beginSet{beginWord};
    unordered_set<string> endSet{endWord};
    
    parentFromBegin[beginWord] = "";
    parentFromEnd[endWord] = "";
    
    bool found = false;
    string meetPoint;
    
    while (!beginSet.empty() && !endSet.empty() && !found) {
        if (beginSet.size() > endSet.size()) {
            swap(beginSet, endSet);
            swap(parentFromBegin, parentFromEnd);
        }
        
        unordered_set<string> nextSet;
        
        for (const string& word : beginSet) {
            for (int i = 0; i < word.size(); i++) {
                char original = word[i];
                for (char c = 'a'; c <= 'z'; c++) {
                    if (c == original) continue;
                    string newWord = word;
                    newWord[i] = c;
                    
                    if (!wordSet.count(newWord)) continue;
                    
                    // 在另一端找到
                    if (parentFromEnd.count(newWord)) {
                        found = true;
                        meetPoint = newWord;
                        parentFromBegin[newWord] = word;
                        break;
                    }
                    
                    if (!parentFromBegin.count(newWord)) {
                        parentFromBegin[newWord] = word;
                        nextSet.insert(newWord);
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        
        beginSet = nextSet;
    }
    
    if (!found) return {};
    
    // 重建路径
    vector<string> path;
    
    // 从相遇点向起点回溯
    string curr = meetPoint;
    while (!curr.empty()) {
        path.push_back(curr);
        curr = parentFromBegin[curr];
    }
    reverse(path.begin(), path.end());
    
    // 从相遇点向终点回溯（跳过相遇点）
    curr = parentFromEnd[meetPoint];
    while (!curr.empty()) {
        path.push_back(curr);
        curr = parentFromEnd[curr];
    }
    
    return path;
}
```

```python [Python]
from typing import List, Set, Optional, Callable

def bidirectional_bfs(
    start: str,
    end: str,
    get_neighbors: Callable[[str], List[str]]
) -> Optional[int]:
    """
    双向 BFS 通用模板
    
    Args:
        start: 起点状态
        end: 终点状态
        get_neighbors: 获取邻居状态的函数
    
    Returns:
        最短路径长度，无解返回 None
    """
    if start == end:
        return 0
    
    # 双端集合
    begin_set = {start}
    end_set = {end}
    visited = {start, end}
    
    level = 0
    
    while begin_set and end_set:
        # 优化1：始终扩展较小的集合
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        next_set = set()
        
        for word in begin_set:
            for neighbor in get_neighbors(word):
                # 优化2：在另一端找到则返回
                if neighbor in end_set:
                    return level + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_set.add(neighbor)
        
        begin_set = next_set
        level += 1
    
    return None


def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    示例：单词接龙
    """
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    def get_neighbors(word: str) -> List[str]:
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == word[i]:
                    continue
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set:
                    neighbors.append(new_word)
        return neighbors
    
    result = bidirectional_bfs(beginWord, endWord, get_neighbors)
    return result if result is not None else 0


def find_path(beginWord: str, endWord: str, wordList: List[str]) -> Optional[List[str]]:
    """
    双向 BFS + 路径重建
    """
    word_set = set(wordList)
    if endWord not in word_set:
        return None
    
    if beginWord == endWord:
        return [beginWord]
    
    # 记录父节点（双向）
    parent_from_begin = {beginWord: None}
    parent_from_end = {endWord: None}
    
    begin_set = {beginWord}
    end_set = {endWord}
    
    found = False
    meet_point = None
    
    while begin_set and end_set and not found:
        # 始终扩展较小的集合
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
            parent_from_begin, parent_from_end = parent_from_end, parent_from_begin
        
        next_set = set()
        
        for word in begin_set:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    
                    new_word = word[:i] + c + word[i+1:]
                    
                    if new_word not in word_set:
                        continue
                    
                    # 在另一端找到
                    if new_word in parent_from_end:
                        found = True
                        meet_point = new_word
                        parent_from_begin[new_word] = word
                        break
                    
                    if new_word not in parent_from_begin:
                        parent_from_begin[new_word] = word
                        next_set.add(new_word)
                
                if found:
                    break
            if found:
                break
        
        begin_set = next_set
    
    if not found:
        return None
    
    # 重建路径
    path = []
    
    # 从相遇点向起点回溯
    curr = meet_point
    while curr is not None:
        path.append(curr)
        curr = parent_from_begin[curr]
    path.reverse()
    
    # 从相遇点向终点回溯
    curr = parent_from_end[meet_point]
    while curr is not None:
        path.append(curr)
        curr = parent_from_end[curr]
    
    return path
```
:::

---

## 搜索优化技巧

### 状态压缩

当状态可以用整数或位运算表示时，状态压缩能大幅减少内存和提高速度。

::: code-group
```cpp [C++]
/**
 * 状态压缩技巧
 * 
 * 适用场景：
 * 1. 状态元素数量有限（如网格坐标、少量物品）
 * 2. 状态可以用整数唯一表示
 * 3. 状态转移涉及位操作
 */

// 技巧1：坐标压缩
// 将 (x, y) 压缩为 x * cols + y
int encode(int x, int y, int cols) {
    return x * cols + y;
}
pair<int, int> decode(int code, int cols) {
    return {code / cols, code % cols};
}

// 技巧2：钥匙状态压缩
// 最多 6 把钥匙，用 6 位二进制表示
int addKey(int state, int key) {
    return state | (1 << key);  // 添加钥匙
}
bool hasKey(int state, int key) {
    return state & (1 << key);  // 检查是否有钥匙
}
int keyCount(int state) {
    return __builtin_popcount(state);  // 钥匙数量
}

// 技巧3：访问数组优化
// 用一维数组代替哈希表
vector<bool> visited(rows * cols * (1 << 6), false);
// 状态: x * cols * KEY_STATES + y * KEY_STATES + keys
int encodeState(int x, int y, int keys, int cols) {
    return x * cols * 64 + y * 64 + keys;
}

/**
 * 示例：迷宫问题（带钥匙）- 状态压缩优化
 */
int shortestPathAllKeys(vector<string>& grid) {
    int m = grid.size(), n = grid[0].size();
    
    int startX, startY, totalKeys = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '@') {
                startX = i; startY = j;
            } else if (islower(grid[i][j])) {
                totalKeys++;
            }
        }
    }
    
    int targetKeys = (1 << totalKeys) - 1;
    int keyStates = 1 << totalKeys;
    
    // 三维访问数组：[x][y][keys]
    vector<vector<vector<bool>>> visited(m, 
        vector<vector<bool>>(n, vector<bool>(keyStates, false)));
    
    // BFS: (x, y, keys)
    queue<tuple<int, int, int>> q;
    q.push({startX, startY, 0});
    visited[startX][startY][0] = true;
    
    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    int steps = 0;
    
    while (!q.empty()) {
        int size = q.size();
        while (size--) {
            auto [x, y, keys] = q.front();
            q.pop();
            
            if (keys == targetKeys) return steps;
            
            for (auto& [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                
                char cell = grid[nx][ny];
                if (cell == '#') continue;
                
                // 检查锁
                if (isupper(cell) && !(keys & (1 << (cell - 'A')))) continue;
                
                int newKeys = keys;
                if (islower(cell)) {
                    newKeys |= (1 << (cell - 'a'));
                }
                
                if (!visited[nx][ny][newKeys]) {
                    visited[nx][ny][newKeys] = true;
                    q.push({nx, ny, newKeys});
                }
            }
        }
        steps++;
    }
    
    return -1;
}
```

```python [Python]
from collections import deque
from typing import List, Tuple

def shortest_path_all_keys(grid: List[str]) -> int:
    """
    状态压缩示例：迷宫问题（带钥匙）
    
    状态压缩技巧：
    - 钥匙状态用位掩码表示
    - 访问数组用三维列表代替哈希表
    """
    m, n = len(grid), len(grid[0])
    
    # 找起点和钥匙数量
    start_x = start_y = total_keys = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '@':
                start_x, start_y = i, j
            elif grid[i][j].islower():
                total_keys += 1
    
    target_keys = (1 << total_keys) - 1
    key_states = 1 << total_keys
    
    # 三维访问数组：visited[x][y][keys]
    visited = [[[False] * key_states for _ in range(n)] for _ in range(m)]
    
    # BFS: (x, y, keys)
    queue = deque([(start_x, start_y, 0)])
    visited[start_x][start_y][0] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    steps = 0
    
    while queue:
        for _ in range(len(queue)):
            x, y, keys = queue.popleft()
            
            if keys == target_keys:
                return steps
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < m and 0 <= ny < n):
                    continue
                
                cell = grid[nx][ny]
                if cell == '#':
                    continue
                
                # 检查锁
                if cell.isupper() and not (keys & (1 << (ord(cell) - ord('A')))):
                    continue
                
                new_keys = keys
                if cell.islower():
                    new_keys |= (1 << (ord(cell) - ord('a')))
                
                if not visited[nx][ny][new_keys]:
                    visited[nx][ny][new_keys] = True
                    queue.append((nx, ny, new_keys))
        
        steps += 1
    
    return -1
```
:::

### 剪枝策略汇总

| 剪枝类型 | 描述 | 适用场景 |
|----------|------|----------|
| **可行性剪枝** | 当前状态无法到达目标 | 约束满足问题 |
| **最优性剪枝** | 当前代价已超最优解 | 最优化问题 |
| **重复状态剪枝** | 已处理过相同状态 | 所有搜索问题 |
| **对称性剪枝** | 利用对称性减少分支 | 排列组合问题 |
| **预估剪枝** | 启发函数估计无法成功 | A*/IDA* |

### 启发函数设计原则

📌 **设计好启发函数的关键**：

1. **可采纳性**：$h(n) \leq h^*(n)$，永不高估
2. **一致性**：满足三角不等式
3. **计算效率**：启发函数要快速计算
4. **信息量**：$h(n)$ 越接近 $h^*(n)$，搜索越高效

💡 **常见启发函数选择**：

```
场景：网格寻路
- 四方向移动 → 曼哈顿距离
- 八方向移动 → 切比雪夫距离
- 任意方向 → 欧几里得距离

场景：组合优化
- TSP问题 → MST下界、最近邻估计
- 背包问题 → 松弛问题的解

场景：游戏状态
- 八数码 → 曼哈顿距离、错位距离
- 国际象棋 → 棋盘评估函数
```

---

## 题型总结

### 📊 高级搜索题型分类

| 题型 | 推荐算法 | 特点 | 代表题目 |
|------|----------|------|----------|
| 路径规划 | A* | 状态空间明确、有目标 | 网格最短路、游戏AI |
| 状态空间搜索 | IDA* | 深度未知、内存受限 | 八数码、十五数码 |
| 最短变换 | 双向BFS | 起点终点明确 | 单词接龙 |
| 带状态搜索 | BFS+状态压缩 | 状态维度增加 | 钥匙与锁、迷宫II |
| 组合优化 | 启发式搜索+剪枝 | 解空间大 | TSP、背包 |

### 🎯 算法选择决策树

```
问题有明确的起点和终点吗？
├─ 是 → 双向 BFS 可能有效
│   └─ 状态空间很大？→ 双向 BFS
│
└─ 否或单目标 → 
    │
    ├─ 能设计好的启发函数？
    │   ├─ 是 → A*（内存够）或 IDA*（内存受限）
    │   └─ 否 → BFS（求最短路）或 DFS（求存在性）
    │
    └─ 状态有额外维度（钥匙、时间等）？
        └─ 是 → 状态压缩 + BFS/DFS
```

### ⚠️ 常见错误

1. **A* 启发函数不可采纳**
   - 后果：可能找不到最优解
   - 解决：确保 $h(n)$ 永不高估实际代价

2. **IDA* 忘记回溯状态**
   - 后果：状态被错误修改
   - 解决：DFS 返回前恢复状态

3. **双向 BFS 没有优化集合选择**
   - 后果：优化效果减半
   - 解决：每次选择较小的集合扩展

4. **状态压缩范围不足**
   - 后果：数组越界或状态冲突
   - 解决：正确计算状态空间大小

### 🔧 调试技巧

```cpp
// 调试 A*：打印 open list 状态
void debugOpenList(priority_queue<Node*>& pq) {
    cout << "Open List: ";
    auto temp = pq;
    while (!temp.empty()) {
        auto node = temp.top(); temp.pop();
        cout << "(" << node->x << "," << node->y 
             << " f=" << node->f << ") ";
    }
    cout << endl;
}

// 调试 IDA*：打印每层阈值
cout << "Current limit: " << limit << ", min_exceed: " << min_exceed << endl;
```

---

## 总结

高级搜索算法是基础搜索的升级版，核心思想是让搜索"更有方向感"：

| 算法 | 核心优化 | 适用场景 |
|------|----------|----------|
| **A*** | 启发函数引导搜索 | 路径规划、游戏AI |
| **IDS/IDA*** | 迭代加深平衡深度与广度 | 内存受限、深度未知 |
| **双向BFS** | 两端同时扩展 | 起点终点明确 |

掌握这些算法的关键是理解**何时使用**和**如何设计启发函数**。在实际问题中，往往需要结合具体场景选择合适的算法和优化策略。
