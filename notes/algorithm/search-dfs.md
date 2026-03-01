# 深度优先搜索 (DFS)

深度优先搜索（Depth-First Search，DFS）是一种用于遍历或搜索树/图结构的算法。它从起点开始，沿着一条路径尽可能深入，直到无法继续前进时才回溯，探索其他路径。

📌 **DFS的本质**

想象你在迷宫中探险：你选择一条路一直走到底，走不通就退回来换另一条路。这就是DFS的精髓——**"一条路走到黑，碰壁再回头"**。DFS不保证找到最短路径，但可以找到所有可能的解。

<DFSAnimation />

## 核心思想

### 深度优先的本质

DFS的核心思想可以用一句话概括：**优先深入，后回溯**。

💡 **为什么优先深入？**
- 深入探索可能更快找到某个解（但不是最短解）
- 空间效率高，只需存储当前路径上的节点
- 适合求解"是否存在解"或"所有解"的问题

**DFS vs BFS 对比**

| 特性 | DFS | BFS |
|------|-----|-----|
| 数据结构 | 栈（递归调用栈或显式栈） | 队列 |
| 遍历策略 | 深入优先，回溯 | 逐层扩展 |
| 空间复杂度 | O(d)，d为深度 | O(b^d)，b为分支因子 |
| 最短路径 | 不保证 | 保证（无权图） |
| 适用场景 | 所有解、解的存在性、路径问题 | 最短路径、层次遍历 |

### 递归与栈的关系

📌 **关键理解：递归 = 隐式栈**

DFS有两种实现方式：
1. **递归实现**：利用函数调用栈自动管理状态
2. **迭代实现**：显式使用栈数据结构

```
递归调用过程：
dfs(状态A)
  └─> dfs(状态B)     // 压栈
        └─> dfs(状态C)  // 压栈
              └─> 回溯   // 弹栈
        └─> dfs(状态D)  // 继续探索
```

💡 **递归的本质**：每次递归调用都会将当前状态"压入"调用栈，当递归返回时，状态自动"弹出"恢复。这正是DFS需要的"前进-回溯"机制。

---

## DFS模板

### 递归模板

递归是DFS最自然的实现方式，代码简洁但需要注意栈溢出问题。

::: code-group
```cpp
// C++ 递归DFS模板
void dfs(状态 state) {
    // 1. 终止条件
    if (满足目标条件) {
        处理目标状态;
        return;
    }
    
    // 2. 剪枝条件（可选，优化效率）
    if (不可能达到目标) {
        return;
    }
    
    // 3. 遍历所有可能的下一步
    for (选择 in 所有可能的选择) {
        if (选择合法) {
            做出选择;           // 修改状态
            dfs(新状态);        // 递归探索
            撤销选择;           // 回溯，恢复状态
        }
    }
}
```

```python
# Python 递归DFS模板
def dfs(state):
    # 1. 终止条件
    if 满足目标条件:
        处理目标状态
        return
    
    # 2. 剪枝条件（可选，优化效率）
    if 不可能达到目标:
        return
    
    # 3. 遍历所有可能的下一步
    for choice in 所有可能的选择:
        if 选择合法:
            做出选择           # 修改状态
            dfs(新状态)        # 递归探索
            撤销选择           # 回溯，恢复状态
```
:::

### 迭代模板（显式栈）

当递归深度过大时，可以使用显式栈避免栈溢出。

::: code-group
```cpp
// C++ 迭代DFS模板
#include <stack>
void dfsIterative(状态 start) {
    stack<状态> stk;
    stk.push(start);
    
    while (!stk.empty()) {
        状态 current = stk.top();
        stk.pop();
        
        if (visited[current]) continue;  // 避免重复访问
        visited[current] = true;
        
        if (满足目标条件) {
            处理目标状态;
            return;
        }
        
        // 注意：栈是后进先出，所以需要反向添加
        for (选择 in 所有可能的选择) {
            if (选择合法 && !visited[新状态]) {
                stk.push(新状态);
            }
        }
    }
}
```

```python
# Python 迭代DFS模板
def dfs_iterative(start):
    stack = [start]
    visited = set()
    
    while stack:
        current = stack.pop()
        
        if current in visited:
            continue
        visited.add(current)
        
        if 满足目标条件:
            处理目标状态
            return
        
        # 注意：栈是后进先出，所以需要反向添加
        for choice in 所有可能的选择:
            if 选择合法 and 新状态 not in visited:
                stack.append(新状态)
```
:::

### 回溯框架

回溯法是DFS的一种特殊形式，用于在解空间中搜索所有满足条件的解。

::: code-group
```cpp
// C++ 回溯框架
vector<类型> path;       // 当前路径
vector<vector<类型>> result;  // 存储所有解

void backtrack(状态 state) {
    // 终止条件：找到一个完整解
    if (满足结束条件) {
        result.push_back(path);  // 保存解
        return;
    }
    
    // 遍历所有可能的选择
    for (选择 in 选择列表) {
        // 剪枝：跳过不合法的选择
        if (!isValid(选择)) continue;
        
        // 做选择
        path.push_back(选择);
        
        // 递归
        backtrack(新状态);
        
        // 撤销选择（回溯）
        path.pop_back();
    }
}
```

```python
# Python 回溯框架
path = []       # 当前路径
result = []     # 存储所有解

def backtrack(state):
    # 终止条件：找到一个完整解
    if 满足结束条件:
        result.append(path[:])  # 保存解的副本
        return
    
    # 遍历所有可能的选择
    for choice in 选择列表:
        # 剪枝：跳过不合法的选择
        if not is_valid(choice):
            continue
        
        # 做选择
        path.append(choice)
        
        # 递归
        backtrack(新状态)
        
        # 撤销选择（回溯）
        path.pop()
```
:::

💡 **回溯三要素**
1. **路径**：已经做出的选择
2. **选择列表**：当前可以做的选择
3. **结束条件**：达到决策树底层，无法再做选择

---

## 经典应用

### 全排列问题

**问题描述**：给定n个不同的元素，输出所有可能的排列方式。

例如，[1,2,3]的全排列有6种：
```
[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
```

💡 **搜索树分析**
```
                [开始]
               /  |  \
             1    2    3     ← 第1位选择
            /\   /\   /\
          2  3  1  3  1  2   ← 第2位选择
          |  |  |  |  |  |
          3  2  3  1  2  1   ← 第3位选择（唯一）
```

**实现代码**

::: code-group
```cpp
#include <iostream>
#include <vector>
using namespace std;

int n;
vector<int> path;           // 当前排列
vector<bool> used;          // 标记元素是否已使用
vector<vector<int>> result; // 存储所有排列

void backtrack() {
    // 终止条件：排列长度等于n
    if (path.size() == n) {
        result.push_back(path);
        return;
    }
    
    // 遍历所有可能的数字
    for (int i = 1; i <= n; i++) {
        if (used[i]) continue;  // 剪枝：已使用的数字跳过
        
        // 做选择
        path.push_back(i);
        used[i] = true;
        
        // 递归
        backtrack();
        
        // 撤销选择
        path.pop_back();
        used[i] = false;
    }
}

int main() {
    cin >> n;
    used.resize(n + 1, false);
    backtrack();
    
    // 输出所有排列
    for (const auto& perm : result) {
        for (int num : perm) {
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}
```

```python
def permute(n):
    """生成1到n的所有全排列"""
    path = []           # 当前排列
    used = [False] * (n + 1)  # 标记元素是否已使用
    result = []         # 存储所有排列
    
    def backtrack():
        # 终止条件：排列长度等于n
        if len(path) == n:
            result.append(path[:])  # 保存副本
            return
        
        # 遍历所有可能的数字
        for i in range(1, n + 1):
            if used[i]:
                continue  # 剪枝：已使用的数字跳过
            
            # 做选择
            path.append(i)
            used[i] = True
            
            # 递归
            backtrack()
            
            # 撤销选择
            path.pop()
            used[i] = False
    
    backtrack()
    return result

# 使用示例
n = int(input())
result = permute(n)
for perm in result:
    print(' '.join(map(str, perm)))
```
:::

### 组合问题

**问题描述**：给定n和k，从1到n中选出k个数的所有组合。

例如，n=4, k=2的组合有：
```
[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
```

💡 **剪枝技巧**
为了避免重复（如[1,2]和[2,1]），我们规定选择必须按非递减顺序，即下一个选择必须大于当前选择。

**实现代码**

::: code-group
```cpp
#include <iostream>
#include <vector>
using namespace std;

int n, k;
vector<int> path;           // 当前组合
vector<vector<int>> result; // 存储所有组合

void backtrack(int start) {
    // 终止条件：组合长度等于k
    if (path.size() == k) {
        result.push_back(path);
        return;
    }
    
    // 遍历选择，从start开始避免重复
    // 剪枝：还需要选择 k - path.size() 个数
    // 所以 i 最多到 n - (k - path.size()) + 1
    for (int i = start; i <= n - (k - path.size()) + 1; i++) {
        path.push_back(i);
        backtrack(i + 1);  // 从i+1开始，保证递增
        path.pop_back();
    }
}

int main() {
    cin >> n >> k;
    backtrack(1);
    
    for (const auto& comb : result) {
        for (int num : comb) {
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}
```

```python
def combine(n, k):
    """从1到n中选出k个数的所有组合"""
    path = []
    result = []
    
    def backtrack(start):
        # 终止条件：组合长度等于k
        if len(path) == k:
            result.append(path[:])
            return
        
        # 遍历选择，从start开始避免重复
        # 剪枝：i最多到 n - (k - len(path)) + 1
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1)  # 从i+1开始，保证递增
            path.pop()
    
    backtrack(1)
    return result

# 使用示例
n, k = map(int, input().split())
result = combine(n, k)
for comb in result:
    print(' '.join(map(str, comb)))
```
:::

### 子集问题

**问题描述**：给定一个不含重复元素的数组，返回所有可能的子集（幂集）。

例如，[1,2,3]的子集有：
```
[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]
```

💡 **子集问题 vs 组合问题**
- 组合问题：只收集叶子节点
- 子集问题：收集所有节点

**实现代码**

::: code-group
```cpp
#include <iostream>
#include <vector>
using namespace std;

vector<int> nums;
vector<int> path;
vector<vector<int>> result;

void backtrack(int start) {
    // 子集问题：每个节点都是解
    result.push_back(path);
    
    // 遍历选择
    for (int i = start; i < nums.size(); i++) {
        path.push_back(nums[i]);
        backtrack(i + 1);
        path.pop_back();
    }
}

int main() {
    int n;
    cin >> n;
    nums.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    
    backtrack(0);
    
    for (const auto& subset : result) {
        cout << "[";
        for (int i = 0; i < subset.size(); i++) {
            cout << subset[i];
            if (i < subset.size() - 1) cout << ",";
        }
        cout << "]" << endl;
    }
    return 0;
}
```

```python
def subsets(nums):
    """生成所有子集"""
    path = []
    result = []
    
    def backtrack(start):
        # 子集问题：每个节点都是解
        result.append(path[:])
        
        # 遍历选择
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()
    
    backtrack(0)
    return result

# 使用示例
nums = list(map(int, input().split()))
result = subsets(nums)
for subset in result:
    print(subset)
```
:::

### N皇后问题

**问题描述**：在n×n的棋盘上放置n个皇后，使其不能互相攻击（同行、同列、同对角线上不能有两个皇后）。

💡 **关键约束**
- 每行只能有一个皇后
- 每列只能有一个皇后
- 每条对角线上只能有一个皇后

**对角线判断技巧**：
- 主对角线（↘）：row - col 值相同
- 副对角线（↙）：row + col 值相同

**实现代码**

::: code-group
```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int n;
vector<string> board;       // 棋盘
vector<vector<string>> result;

bool isValid(int row, int col) {
    // 检查列
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') return false;
    }
    
    // 检查左上方对角线
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') return false;
    }
    
    // 检查右上方对角线
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q') return false;
    }
    
    return true;
}

void backtrack(int row) {
    // 终止条件：处理完所有行
    if (row == n) {
        result.push_back(board);
        return;
    }
    
    // 尝试在当前行的每一列放置皇后
    for (int col = 0; col < n; col++) {
        if (!isValid(row, col)) continue;  // 剪枝
        
        // 做选择
        board[row][col] = 'Q';
        
        // 递归
        backtrack(row + 1);
        
        // 撤销选择
        board[row][col] = '.';
    }
}

int main() {
    cin >> n;
    
    // 初始化棋盘
    board.resize(n, string(n, '.'));
    
    backtrack(0);
    
    cout << "共有 " << result.size() << " 种解法" << endl;
    for (const auto& solution : result) {
        for (const string& row : solution) {
            cout << row << endl;
        }
        cout << endl;
    }
    
    return 0;
}
```

```python
def solve_n_queens(n):
    """求解N皇后问题"""
    board = [['.' for _ in range(n)] for _ in range(n)]
    result = []
    
    def is_valid(row, col):
        # 检查列
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # 检查左上方对角线
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # 检查右上方对角线
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        # 终止条件：处理完所有行
        if row == n:
            result([''.join(row) for row in board])
            return
        
        # 尝试在当前行的每一列放置皇后
        for col in range(n):
            if not is_valid(row, col):
                continue  # 剪枝
            
            # 做选择
            board[row][col] = 'Q'
            
            # 递归
            backtrack(row + 1)
            
            # 撤销选择
            board[row][col] = '.'
    
    backtrack(0)
    return result

# 使用示例
n = int(input())
solutions = solve_n_queens(n)
print(f"共有 {len(solutions)} 种解法")
for solution in solutions:
    for row in solution:
        print(row)
    print()
```
:::

---

## 剪枝优化

剪枝是DFS优化的核心技术，通过提前排除不可能的分支，大幅减少搜索空间。

### 可行性剪枝

当当前状态已经无法达到目标时，提前终止搜索。

```cpp
// 示例：数独问题中的可行性剪枝
bool isValid(vector<vector<char>>& board, int row, int col, char num) {
    // 检查行
    for (int j = 0; j < 9; j++) {
        if (board[row][j] == num) return false;
    }
    // 检查列
    for (int i = 0; i < 9; i++) {
        if (board[i][col] == num) return false;
    }
    // 检查3x3宫格
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) {
        for (int j = startCol; j < startCol + 3; j++) {
            if (board[i][j] == num) return false;
        }
    }
    return true;
}

void solveSudoku(vector<vector<char>>& board, int row, int col) {
    if (row == 9) { /* 找到解 */ return; }
    
    int nextRow = (col == 8) ? row + 1 : row;
    int nextCol = (col == 8) ? 0 : col + 1;
    
    if (board[row][col] != '.') {
        solveSudoku(board, nextRow, nextCol);
        return;
    }
    
    for (char num = '1'; num <= '9'; num++) {
        if (!isValid(board, row, col, num)) continue;  // 可行性剪枝
        board[row][col] = num;
        solveSudoku(board, nextRow, nextCol);
        board[row][col] = '.';
    }
}
```

### 最优性剪枝

当当前代价已经超过已知最优解时，提前终止搜索。

```cpp
// 示例：旅行商问题中的最优性剪枝
int minCost = INT_MAX;

void dfs(int current, int visited, int cost, int n, vector<vector<int>>& graph) {
    // 最优性剪枝：当前代价已经超过已知最优解
    if (cost >= minCost) {
        return;
    }
    
    // 所有城市都访问过
    if (visited == (1 << n) - 1) {
        minCost = min(minCost, cost + graph[current][0]);
        return;
    }
    
    for (int next = 0; next < n; next++) {
        if (!(visited & (1 << next))) {
            dfs(next, visited | (1 << next), cost + graph[current][next], n, graph);
        }
    }
}
```

### 记忆化搜索

将已经计算过的子问题结果缓存起来，避免重复计算。

::: code-group
```cpp
// 示例：记忆化搜索求解斐波那契数列
#include <vector>
#include <unordered_map>
using namespace std;

vector<int> memo;

int fib(int n) {
    if (n <= 1) return n;
    
    // 查表
    if (memo[n] != -1) return memo[n];
    
    // 计算并存储
    memo[n] = fib(n - 1) + fib(n - 2);
    return memo[n];
}

int main() {
    int n = 30;
    memo.resize(n + 1, -1);
    cout << fib(n) << endl;
    return 0;
}
```

```python
# 示例：记忆化搜索求解斐波那契数列
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# 或手动实现
memo = {}
def fib_manual(n):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_manual(n - 1) + fib_manual(n - 2)
    return memo[n]

print(fib(30))
```
:::

---

## 题型总结

### 树的DFS

树的DFS是最基础的应用，用于遍历树结构。

::: code-group
```cpp
// 二叉树的DFS遍历
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 前序遍历：根 -> 左 -> 右
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);   // 访问根
    preorder(root->left, result);  // 遍历左子树
    preorder(root->right, result); // 遍历右子树
}

// 中序遍历：左 -> 根 -> 右
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);   // 遍历左子树
    result.push_back(root->val);   // 访问根
    inorder(root->right, result);  // 遍历右子树
}

// 后序遍历：左 -> 右 -> 根
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);  // 遍历左子树
    postorder(root->right, result); // 遍历右子树
    result.push_back(root->val);    // 访问根
}

// 计算树的深度
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

```python
# 二叉树的DFS遍历
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 前序遍历：根 -> 左 -> 右
def preorder(root, result):
    if not root:
        return
    result.append(root.val)    # 访问根
    preorder(root.left, result)  # 遍历左子树
    preorder(root.right, result) # 遍历右子树

# 中序遍历：左 -> 根 -> 右
def inorder(root, result):
    if not root:
        return
    inorder(root.left, result)   # 遍历左子树
    result.append(root.val)      # 访问根
    inorder(root.right, result)  # 遍历右子树

# 后序遍历：左 -> 右 -> 根
def postorder(root, result):
    if not root:
        return
    postorder(root.left, result)   # 遍历左子树
    postorder(root.right, result)  # 遍历右子树
    result.append(root.val)        # 访问根

# 计算树的深度
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```
:::

### 图的DFS

图的DFS需要注意标记已访问节点，避免重复访问和死循环。

::: code-group
```cpp
#include <vector>
using namespace std;

vector<vector<int>> graph;
vector<bool> visited;

// 基本DFS遍历
void dfs(int node) {
    visited[node] = true;
    cout << "访问节点: " << node << endl;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor);
        }
    }
}

// 检测图中是否有环
bool hasCycle(int node, int parent) {
    visited[node] = true;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            if (hasCycle(neighbor, node)) {
                return true;
            }
        } else if (neighbor != parent) {
            // 访问到的不是父节点，说明有环
            return true;
        }
    }
    return false;
}

// 拓扑排序（DFS版本）
vector<int> topoOrder;
void topoSort(int node, vector<bool>& inStack) {
    visited[node] = true;
    inStack[node] = true;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            topoSort(neighbor, inStack);
        }
    }
    
    inStack[node] = false;
    topoOrder.push_back(node);  // 后序遍历的逆序
}
```

```python
# 图的DFS遍历
from collections import defaultdict

graph = defaultdict(list)
visited = set()

# 基本DFS遍历
def dfs(node):
    visited.add(node)
    print(f"访问节点: {node}")
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor)

# 检测图中是否有环（无向图）
def has_cycle(node, parent):
    visited.add(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            if has_cycle(neighbor, node):
                return True
        elif neighbor != parent:
            # 访问到的不是父节点，说明有环
            return True
    return False

# 拓扑排序（DFS版本）
topo_order = []
def topo_sort(node, in_stack):
    visited.add(node)
    in_stack.add(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            topo_sort(neighbor, in_stack)
    
    in_stack.remove(node)
    topo_order.append(node)  # 后序遍历的逆序
```
:::

### 网格DFS

网格DFS常用于岛屿问题、迷宫问题等二维网格场景。

::: code-group
```cpp
#include <vector>
using namespace std;

// 四个方向：上、下、左、右
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

// 岛屿数量问题
void dfsIsland(vector<vector<char>>& grid, int x, int y) {
    int m = grid.size();
    int n = grid[0].size();
    
    // 边界检查
    if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != '1') {
        return;
    }
    
    // 标记已访问
    grid[x][y] = '2';  // 或使用visited数组
    
    // 四个方向DFS
    for (int i = 0; i < 4; i++) {
        dfsIsland(grid, x + dx[i], y + dy[i]);
    }
}

int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    int m = grid.size();
    int n = grid[0].size();
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                dfsIsland(grid, i, j);
                count++;
            }
        }
    }
    return count;
}

// 计算岛屿面积
int dfsArea(vector<vector<int>>& grid, int x, int y) {
    int m = grid.size();
    int n = grid[0].size();
    
    if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != 1) {
        return 0;
    }
    
    grid[x][y] = 0;  // 标记已访问
    
    int area = 1;  // 当前格子
    for (int i = 0; i < 4; i++) {
        area += dfsArea(grid, x + dx[i], y + dy[i]);
    }
    return area;
}
```

```python
# 网格DFS
# 四个方向：上、下、左、右
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 岛屿数量问题
def dfs_island(grid, x, y):
    m, n = len(grid), len(grid[0])
    
    # 边界检查
    if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] != '1':
        return
    
    # 标记已访问
    grid[x][y] = '2'  # 或使用visited集合
    
    # 四个方向DFS
    for dx, dy in DIRECTIONS:
        dfs_island(grid, x + dx, y + dy)

def num_islands(grid):
    if not grid:
        return 0
    
    count = 0
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs_island(grid, i, j)
                count += 1
    
    return count

# 计算岛屿面积
def dfs_area(grid, x, y):
    m, n = len(grid), len(grid[0])
    
    if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] != 1:
        return 0
    
    grid[x][y] = 0  # 标记已访问
    
    area = 1  # 当前格子
    for dx, dy in DIRECTIONS:
        area += dfs_area(grid, x + dx, y + dy)
    return area

# 使用集合记录访问过的位置
def dfs_with_visited(grid, x, y, visited):
    m, n = len(grid), len(grid[0])
    
    if (x, y) in visited:
        return 0
    if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] == '0':
        return 0
    
    visited.add((x, y))
    area = 1
    
    for dx, dy in DIRECTIONS:
        area += dfs_with_visited(grid, x + dx, y + dy, visited)
    
    return area
```
:::

---

## 注意事项

### ⚠️ 栈溢出问题

递归DFS可能因深度过大导致栈溢出。解决方案：
1. 使用迭代实现（显式栈）
2. 增加栈空间限制（编译器选项）
3. 使用尾递归优化（如果语言支持）

### ⚠️ 重复访问问题

在图和网格DFS中，必须正确标记已访问节点：
- 无向图：需要记录父节点避免回溯
- 有向图：需要区分"正在访问"和"已访问完成"
- 网格：修改原数组或使用visited集合

### ⚠️ 状态恢复问题

回溯法中，撤销选择是关键：
- 路径数组：push后必须pop
- 访问标记：设置后必须重置
- 原地修改：修改后必须恢复

---

## 参考资料

- [LeetCode回溯算法总结](https://leetcode.cn/problems/tag-list/backtracking/)
- [OI Wiki - 搜索](https://oi-wiki.org/search/)
