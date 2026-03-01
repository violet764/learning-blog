# 广度优先搜索（BFS）

广度优先搜索（Breadth-First Search，BFS）是一种用于遍历或搜索图、树等数据结构的算法。与深度优先搜索（DFS）的"一条路走到黑"不同，BFS 采用"层层推进"的策略，先访问离起点最近的所有节点，再访问稍远的节点，依次类推。

这种特性使得 BFS 成为**寻找无权图最短路径**的最佳选择，也是解决迷宫问题、连通性问题的核心工具。

<BFSAnimation />

## BFS 核心思想

### 广度优先的本质

BFS 的核心思想可以概括为"**水波扩散**"——就像向平静的湖面投入一颗石子，水波从中心向四周均匀扩散。

```
        第3层    ● ● ● ● ●
        第2层      ● ● ●
        第1层        ●        ← 起点
        第2层      ● ● ●
        第3层    ● ● ● ● ●
```

📌 **关键特性**：
- **按层遍历**：先访问距离起点为 1 的所有节点，再访问距离为 2 的节点...
- **最短路径保证**：在无权图中，BFS 第一次访问到目标节点时，必然是最短路径
- **公平探索**：不会像 DFS 那样深入某一条路径而忽略其他可能

### 队列的使用

BFS 的实现离不开**队列（Queue）**这一数据结构。队列的"先进先出"（FIFO）特性完美契合 BFS 的"先遇到先处理"的需求。

```
队列工作流程：
┌─────────────────────────────────────────────┐
│  入队顺序: A → B → C → D → E               │
│  出队顺序: A → B → C → D → E  (FIFO)       │
└─────────────────────────────────────────────┘

BFS 执行过程：
1. 起点入队
2. 队首元素出队，将其所有未访问邻居入队
3. 重复步骤 2，直到队列为空或找到目标
```

💡 **为什么不用栈？**
- 栈是 LIFO（后进先出），会导致算法优先深入最后发现的路径
- 队列确保先发现的节点先被处理，保证"按层"的顺序

### 时间与空间复杂度

| 复杂度 | 分析 |
|--------|------|
| **时间复杂度** | $O(V + E)$，其中 $V$ 是顶点数，$E$ 是边数 |
| **空间复杂度** | $O(V)$，队列中最多存储所有顶点 |

---

## BFS 模板

### 基础模板

以下是 BFS 的标准实现模板，适用于图遍历和最短路径问题：

::: code-group
```cpp [C++]
#include <bits/stdc++.h>
using namespace std;

/**
 * BFS 基础模板 - 图遍历/最短路径
 * @param start 起点
 * @param target 目标点（可选）
 * @param graph 图的邻接表表示
 * @return 从起点到目标的最短距离，找不到返回 -1
 */
int bfs(int start, int target, unordered_map<int, vector<int>>& graph) {
    // 队列存储待访问节点
    queue<int> q;
    // 访问标记，同时记录距离
    unordered_map<int, int> dist;
    
    // 初始化
    q.push(start);
    dist[start] = 0;
    
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        
        // 找到目标
        if (curr == target) {
            return dist[curr];
        }
        
        // 遍历所有邻居
        for (int next : graph[curr]) {
            // 未访问过的节点
            if (dist.find(next) == dist.end()) {
                dist[next] = dist[curr] + 1;  // 距离+1
                q.push(next);
            }
        }
    }
    
    return -1;  // 无法到达目标
}

/**
 * BFS 模板 - 返回完整路径
 */
vector<int> bfs_path(int start, int target, 
                     unordered_map<int, vector<int>>& graph) {
    queue<int> q;
    unordered_map<int, int> parent;  // 记录父节点，用于回溯路径
    
    q.push(start);
    parent[start] = -1;  // 起点无父节点
    
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        
        if (curr == target) {
            // 回溯构建路径
            vector<int> path;
            for (int node = target; node != -1; node = parent[node]) {
                path.push_back(node);
            }
            reverse(path.begin(), path.end());
            return path;
        }
        
        for (int next : graph[curr]) {
            if (parent.find(next) == parent.end()) {
                parent[next] = curr;
                q.push(next);
            }
        }
    }
    
    return {};  // 无法到达
}
```

```python [Python]
from collections import deque
from typing import Dict, List, Optional

def bfs(start: int, target: int, graph: Dict[int, List[int]]) -> int:
    """
    BFS 基础模板 - 图遍历/最短路径
    
    Args:
        start: 起点
        target: 目标点
        graph: 图的邻接表表示
    
    Returns:
        从起点到目标的最短距离，找不到返回 -1
    """
    # 双端队列，popleft() 时间复杂度 O(1)
    queue = deque([start])
    # 访问标记，同时记录距离
    dist = {start: 0}
    
    while queue:
        curr = queue.popleft()
        
        # 找到目标
        if curr == target:
            return dist[curr]
        
        # 遍历所有邻居
        for next_node in graph.get(curr, []):
            if next_node not in dist:
                dist[next_node] = dist[curr] + 1
                queue.append(next_node)
    
    return -1  # 无法到达目标


def bfs_path(start: int, target: int, 
             graph: Dict[int, List[int]]) -> List[int]:
    """
    BFS 模板 - 返回完整路径
    
    Returns:
        从起点到目标的路径列表，找不到返回空列表
    """
    queue = deque([start])
    parent = {start: None}  # 记录父节点
    
    while queue:
        curr = queue.popleft()
        
        if curr == target:
            # 回溯构建路径
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]  # 反转路径
        
        for next_node in graph.get(curr, []):
            if next_node not in parent:
                parent[next_node] = curr
                queue.append(next_node)
    
    return []  # 无法到达
```
:::

### 多源 BFS

多源 BFS 从多个起点同时开始搜索，常用于求解"离最近源点的距离"问题。

::: code-group
```cpp [C++]
/**
 * 多源 BFS - 计算每个格子到最近源点的距离
 * 典型应用：01矩阵、腐烂的橘子、地图分析
 * 
 * @param grid 输入网格，sources 为源点集合
 * @return 每个位置到最近源点的距离矩阵
 */
vector<vector<int>> multiSourceBFS(vector<vector<int>>& grid,
                                    vector<pair<int,int>>& sources) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<int>> dist(m, vector<int>(n, -1));
    queue<pair<int,int>> q;
    
    // 所有源点同时入队
    for (auto& [r, c] : sources) {
        q.push({r, c});
        dist[r][c] = 0;
    }
    
    // 四个方向
    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        
        for (auto& [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            // 边界检查 + 未访问检查
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && dist[nr][nc] == -1) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    
    return dist;
}

// 示例：01矩阵 - 找到每个 1 到最近 0 的距离
vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    vector<pair<int,int>> sources;
    
    // 收集所有 0 作为源点
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (mat[i][j] == 0) {
                sources.push_back({i, j});
            }
        }
    }
    
    return multiSourceBFS(mat, sources);
}
```

```python [Python]
from collections import deque
from typing import List, Tuple

def multi_source_bfs(grid: List[List[int]], 
                     sources: List[Tuple[int, int]]) -> List[List[int]]:
    """
    多源 BFS - 计算每个格子到最近源点的距离
    
    Args:
        grid: 输入网格
        sources: 源点坐标列表
    
    Returns:
        每个位置到最近源点的距离矩阵
    """
    if not grid or not grid[0]:
        return []
    
    m, n = len(grid), len(grid[0])
    dist = [[-1] * n for _ in range(m)]
    queue = deque()
    
    # 所有源点同时入队
    for r, c in sources:
        queue.append((r, c))
        dist[r][c] = 0
    
    # 四个方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # 边界检查 + 未访问检查
            if 0 <= nr < m and 0 <= nc < n and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    
    return dist


def update_matrix(mat: List[List[int]]) -> List[List[int]]:
    """
    示例：01矩阵 - 找到每个 1 到最近 0 的距离
    """
    m, n = len(mat), len(mat[0])
    sources = [(i, j) for i in range(m) for j in range(n) if mat[i][j] == 0]
    return multi_source_bfs(mat, sources)
```
:::

### 带状态的 BFS

当问题需要记录额外状态时（如钥匙、方向、时间等），需要将状态纳入访问标记。

::: code-group
```cpp [C++]
/**
 * 带状态的 BFS - 迷宫问题（带钥匙）
 * 状态 = (位置, 持有的钥匙集合)
 * 
 * @param grid 迷宫网格，'a'-'f' 是钥匙，'A'-'F' 是锁
 * @return 获取所有钥匙的最少步数
 */
int shortestPathAllKeys(vector<string>& grid) {
    int m = grid.size(), n = grid[0].size();
    
    // 找起点和钥匙数量
    int start_r, start_c, total_keys = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '@') {
                start_r = i; start_c = j;
            } else if (islower(grid[i][j])) {
                total_keys++;
            }
        }
    }
    
    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    int target_keys = (1 << total_keys) - 1;  // 所有钥匙都收集
    
    // 状态: (row, col, keys_bitmask)
    // keys_bitmask 用位表示持有哪些钥匙
    queue<tuple<int,int,int>> q;
    set<tuple<int,int,int>> visited;
    
    q.push({start_r, start_c, 0});
    visited.insert({start_r, start_c, 0});
    int steps = 0;
    
    while (!q.empty()) {
        int size = q.size();
        while (size--) {
            auto [r, c, keys] = q.front();
            q.pop();
            
            if (keys == target_keys) return steps;
            
            for (auto& [dr, dc] : dirs) {
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= m || nc < 0 || nc >= n) continue;
                
                char cell = grid[nr][nc];
                // 墙壁
                if (cell == '#') continue;
                // 锁但没有钥匙
                if (isupper(cell) && !(keys & (1 << (cell - 'A')))) continue;
                
                int new_keys = keys;
                // 捡起钥匙
                if (islower(cell)) {
                    new_keys |= (1 << (cell - 'a'));
                }
                
                auto state = make_tuple(nr, nc, new_keys);
                if (visited.find(state) == visited.end()) {
                    visited.insert(state);
                    q.push(state);
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
    带状态的 BFS - 迷宫问题（带钥匙）
    状态 = (位置, 持有的钥匙集合)
    
    Args:
        grid: 迷宫网格，'a'-'f' 是钥匙，'A'-'F' 是锁
    
    Returns:
        获取所有钥匙的最少步数
    """
    m, n = len(grid), len(grid[0])
    
    # 找起点和钥匙数量
    start_r = start_c = total_keys = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '@':
                start_r, start_c = i, j
            elif grid[i][j].islower():
                total_keys += 1
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    target_keys = (1 << total_keys) - 1
    
    # 状态: (row, col, keys_bitmask)
    queue = deque([(start_r, start_c, 0)])
    visited = {(start_r, start_c, 0)}
    steps = 0
    
    while queue:
        for _ in range(len(queue)):
            r, c, keys = queue.popleft()
            
            if keys == target_keys:
                return steps
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < m and 0 <= nc < n):
                    continue
                
                cell = grid[nr][nc]
                # 墙壁
                if cell == '#':
                    continue
                # 锁但没有钥匙
                if cell.isupper() and not (keys & (1 << (ord(cell) - ord('A')))):
                    continue
                
                new_keys = keys
                # 捡起钥匙
                if cell.islower():
                    new_keys |= (1 << (ord(cell) - ord('a')))
                
                state = (nr, nc, new_keys)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        
        steps += 1
    
    return -1
```
:::

---

## 经典应用

### 层序遍历

二叉树的层序遍历是 BFS 的典型应用，按层输出节点值。

::: code-group
```cpp [C++]
/**
 * 二叉树的层序遍历
 * @return 按层分组的节点值
 */
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};
    
    vector<vector<int>> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int level_size = q.size();  // 当前层的节点数
        vector<int> level;
        
        // 处理当前层的所有节点
        for (int i = 0; i < level_size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(level);
    }
    
    return result;
}
```

```python [Python]
from collections import deque
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    二叉树的层序遍历
    
    Returns:
        按层分组的节点值
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # 当前层的节点数
        level = []
        
        # 处理当前层的所有节点
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```
:::

### 最短路径（无权图）

BFS 天然适合寻找无权图的最短路径。

::: code-group
```cpp [C++]
/**
 * 无权图最短路径
 * 单词接龙问题：每次改变一个字母，求最短变换序列长度
 */
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) return 0;
    
    queue<string> q;
    unordered_set<string> visited;
    
    q.push(beginWord);
    visited.insert(beginWord);
    int level = 1;  // 起点算第 1 层
    
    while (!q.empty()) {
        int size = q.size();
        
        for (int i = 0; i < size; i++) {
            string word = q.front();
            q.pop();
            
            if (word == endWord) return level;
            
            // 尝试改变每个位置的字母
            for (int j = 0; j < word.size(); j++) {
                char original = word[j];
                for (char c = 'a'; c <= 'z'; c++) {
                    if (c == original) continue;
                    word[j] = c;
                    
                    if (wordSet.count(word) && !visited.count(word)) {
                        visited.insert(word);
                        q.push(word);
                    }
                }
                word[j] = original;  // 恢复
            }
        }
        level++;
    }
    
    return 0;
}
```

```python [Python]
from collections import deque
from typing import List

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    无权图最短路径 - 单词接龙
    
    Returns:
        最短变换序列长度，无法变换返回 0
    """
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    queue = deque([beginWord])
    visited = {beginWord}
    level = 1  # 起点算第 1 层
    
    while queue:
        for _ in range(len(queue)):
            word = queue.popleft()
            
            if word == endWord:
                return level
            
            # 尝试改变每个位置的字母
            for j in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[j]:
                        continue
                    new_word = word[:j] + c + word[j+1:]
                    
                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append(new_word)
        
        level += 1
    
    return 0
```
:::

### 迷宫问题

::: code-group
```cpp [C++]
/**
 * 迷宫最短路径
 * @param maze 迷宫，0 表示空地，1 表示墙
 * @param start 起点
 * @param destination 终点
 * @return 最少步数
 */
int shortestPathBinaryMaze(vector<vector<int>>& maze,
                           vector<int>& start,
                           vector<int>& destination) {
    int m = maze.size(), n = maze[0].size();
    if (maze[start[0]][start[1]] == 1 || 
        maze[destination[0]][destination[1]] == 1) {
        return -1;
    }
    
    // 8 个方向（包括对角线）
    int dirs[8][2] = {{-1,0}, {1,0}, {0,-1}, {0,1},
                      {-1,-1}, {-1,1}, {1,-1}, {1,1}};
    
    queue<pair<int,int>> q;
    vector<vector<int>> dist(m, vector<int>(n, -1));
    
    q.push({start[0], start[1]});
    dist[start[0]][start[1]] = 1;
    
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        
        if (r == destination[0] && c == destination[1]) {
            return dist[r][c];
        }
        
        for (auto& [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < m && nc >= 0 && nc < n 
                && maze[nr][nc] == 0 && dist[nr][nc] == -1) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    
    return -1;
}
```

```python [Python]
from collections import deque
from typing import List

def shortest_path_binary_maze(maze: List[List[int]], 
                              start: List[int],
                              destination: List[int]) -> int:
    """
    迷宫最短路径
    
    Args:
        maze: 迷宫，0 表示空地，1 表示墙
        start: 起点 [row, col]
        destination: 终点 [row, col]
    
    Returns:
        最少步数，无法到达返回 -1
    """
    m, n = len(maze), len(maze[0])
    
    if maze[start[0]][start[1]] == 1 or maze[destination[0]][destination[1]] == 1:
        return -1
    
    # 8 个方向（包括对角线）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    queue = deque([(start[0], start[1])])
    dist = [[-1] * n for _ in range(m)]
    dist[start[0]][start[1]] = 1
    
    while queue:
        r, c = queue.popleft()
        
        if [r, c] == destination:
            return dist[r][c]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < m and 0 <= nc < n 
                and maze[nr][nc] == 0 and dist[nr][nc] == -1):
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    
    return -1
```
:::

### 岛屿问题

::: code-group
```cpp [C++]
/**
 * 岛屿数量 - 使用 BFS 统计连通分量
 */
int numIslands(vector<vector<char>>& grid) {
    if (grid.empty()) return 0;
    
    int m = grid.size(), n = grid[0].size();
    int count = 0;
    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                count++;
                // BFS 淹没整个岛屿
                queue<pair<int,int>> q;
                q.push({i, j});
                grid[i][j] = '0';  // 标记为已访问
                
                while (!q.empty()) {
                    auto [r, c] = q.front();
                    q.pop();
                    
                    for (auto& [dr, dc] : dirs) {
                        int nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < m && nc >= 0 && nc < n 
                            && grid[nr][nc] == '1') {
                            grid[nr][nc] = '0';
                            q.push({nr, nc});
                        }
                    }
                }
            }
        }
    }
    
    return count;
}

/**
 * 岛屿周长 - BFS 计算每个陆地块的贡献
 */
int islandPerimeter(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    int perimeter = 0;
    int dirs[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                for (auto& [dr, dc] : dirs) {
                    int ni = i + dr, nj = j + dc;
                    // 边界或水域贡献 1
                    if (ni < 0 || ni >= m || nj < 0 || nj >= n || grid[ni][nj] == 0) {
                        perimeter++;
                    }
                }
            }
        }
    }
    
    return perimeter;
}
```

```python [Python]
from collections import deque
from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    岛屿数量 - 使用 BFS 统计连通分量
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                # BFS 淹没整个岛屿
                queue = deque([(i, j)])
                grid[i][j] = '0'  # 标记为已访问
                
                while queue:
                    r, c = queue.popleft()
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < m and 0 <= nc < n 
                            and grid[nr][nc] == '1'):
                            grid[nr][nc] = '0'
                            queue.append((nr, nc))
    
    return count


def island_perimeter(grid: List[List[int]]) -> int:
    """
    岛屿周长 - 计算每个陆地块的贡献
    
    每个陆地块贡献的边数 = 4 - 相邻陆地块数量
    """
    m, n = len(grid), len(grid[0])
    perimeter = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                for dr, dc in directions:
                    ni, nj = i + dr, j + dc
                    # 边界或水域贡献 1
                    if ni < 0 or ni >= m or nj < 0 or nj >= n or grid[ni][nj] == 0:
                        perimeter += 1
    
    return perimeter
```
:::

---

## 双向 BFS

### 原理与优化

双向 BFS 从起点和终点**同时向中间搜索**，当两个搜索相遇时找到答案。

<BidirectionalSearch />

**为什么更快？**

单向 BFS 搜索的节点数约为 $b^d$（b 是分支因子，d 是深度）

双向 BFS 搜索的节点数约为 $2 \times b^{d/2}$

当 $b=10, d=10$ 时：
- 单向：$10^{10} = 10,000,000,000$ 次操作
- 双向：$2 \times 10^5 = 200,000$ 次操作

**优化效果**：搜索空间从指数级降为开方级别！

### 双向 BFS 实现

::: code-group
```cpp [C++]
/**
 * 双向 BFS - 单词接龙
 * 从起点和终点同时搜索，相遇时得到答案
 */
int ladderLengthBidirectional(string beginWord, string endWord, 
                               vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (!wordSet.count(endWord)) return 0;
    
    // 双端搜索集合
    unordered_set<string> beginSet{beginWord};
    unordered_set<string> endSet{endWord};
    int level = 1;
    
    while (!beginSet.empty() && !endSet.empty()) {
        // 始终从较小的集合扩展（优化）
        if (beginSet.size() > endSet.size()) {
            swap(beginSet, endSet);
        }
        
        unordered_set<string> nextSet;
        
        for (string word : beginSet) {
            // 尝试每个位置的变换
            for (int j = 0; j < word.size(); j++) {
                char original = word[j];
                for (char c = 'a'; c <= 'z'; c++) {
                    word[j] = c;
                    
                    // 在另一端找到，说明相遇
                    if (endSet.count(word)) {
                        return level + 1;
                    }
                    
                    // 有效单词，加入下一层
                    if (wordSet.count(word)) {
                        nextSet.insert(word);
                        wordSet.erase(word);  // 避免重复访问
                    }
                }
                word[j] = original;
            }
        }
        
        beginSet = nextSet;
        level++;
    }
    
    return 0;
}
```

```python [Python]
from typing import List, Set

def ladder_length_bidirectional(beginWord: str, endWord: str, 
                                 wordList: List[str]) -> int:
    """
    双向 BFS - 单词接龙
    从起点和终点同时搜索，相遇时得到答案
    """
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    # 双端搜索集合
    begin_set = {beginWord}
    end_set = {endWord}
    level = 1
    
    while begin_set and end_set:
        # 始终从较小的集合扩展（优化）
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        next_set = set()
        
        for word in begin_set:
            # 尝试每个位置的变换
            for j in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:j] + c + word[j+1:]
                    
                    # 在另一端找到，说明相遇
                    if new_word in end_set:
                        return level + 1
                    
                    # 有效单词，加入下一层
                    if new_word in word_set:
                        next_set.add(new_word)
                        word_set.remove(new_word)  # 避免重复访问
        
        begin_set = next_set
        level += 1
    
    return 0
```
:::

### 双向 BFS 的适用场景

| 特征 | 是否适合双向 BFS |
|------|-----------------|
| 已知起点和终点 | ✅ 非常适合 |
| 只知道起点，终点不明确 | ❌ 不适合 |
| 状态空间大但解较短 | ✅ 效果显著 |
| 分支因子较大 | ✅ 优化明显 |

---

## 题型总结

### 📊 BFS 题型分类

| 题型 | 特点 | 代表题目 |
|------|------|----------|
| **基础遍历** | 图/树的层序遍历 | 二叉树层序遍历 |
| **最短路径** | 无权图求最短路 | 单词接龙、迷宫 |
| **连通性** | 统计连通分量 | 岛屿数量、朋友圈 |
| **多源 BFS** | 多起点同时扩散 | 01矩阵、腐烂的橘子 |
| **状态 BFS** | 状态空间搜索 | 钥匙与锁、滑块谜题 |
| **双向 BFS** | 两端向中间搜 | 单词接龙 II |

### 🎯 BFS vs DFS 选择

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 求最短路径 | BFS | BFS 保证首次到达即最短 |
| 搜索所有解 | DFS | 更容易回溯和剪枝 |
| 判断连通性 | 均可 | BFS 更直观 |
| 拓扑排序 | BFS | Kahn 算法天然适合 |
| 空间受限 | DFS | 栈空间通常更小 |

### ⚠️ 常见错误

1. **忘记标记已访问**
   - 导致无限循环或重复访问
   - 解决：入队时立即标记，而不是出队时

2. **队列使用不当**
   - Python 中用 `list.pop(0)` 是 $O(n)$
   - 解决：使用 `collections.deque`

3. **层序遍历层数计算错误**
   - 在循环内部增加层数导致计数过多
   - 解决：用 `for _ in range(len(queue))` 控制一层的处理

4. **双向 BFS 没有优化集合选择**
   - 总是固定从一端扩展
   - 解决：每次选择较小的集合扩展

### 🔧 调试技巧

```python
# 调试时打印每层状态
while queue:
    level_size = len(queue)
    print(f"Level {level}: {[node for node in queue]}")  # 调试
    
    for _ in range(level_size):
        # ... 处理逻辑
```

---

## 总结

BFS 是一种强大而优雅的搜索算法，其核心优势在于：

1. **最短路径保证**：在无权图中首次到达即是最短
2. **层次分明**：天然适合按层处理的问题
3. **状态空间可控**：适用于明确状态空间的问题

掌握 BFS 的关键是理解**队列的作用**和**访问标记的时机**，同时灵活运用**多源 BFS** 和**双向 BFS** 等变体来优化特定问题。
