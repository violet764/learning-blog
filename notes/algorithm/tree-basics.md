# 树基础：二叉树与堆

树是一种非线性数据结构，由节点和边组成，具有层次关系。树结构在计算机科学中应用广泛，如文件系统、数据库索引、表达式解析等。本文将系统介绍二叉树和堆这两个核心数据结构。

## 基本概念

### 树的术语

| 术语 | 定义 |
|------|------|
| **节点 (Node)** | 树的基本单元，包含数据和指向子节点的指针 |
| **根节点 (Root)** | 树的唯一入口节点，没有父节点 |
| **叶子节点 (Leaf)** | 没有子节点的节点 |
| **内部节点** | 至少有一个子节点的非叶子节点 |
| **父节点/子节点** | 相邻节点间的层次关系 |
| **兄弟节点** | 具有相同父节点的节点 |
| **深度 (Depth)** | 从根节点到该节点的边数（根节点深度为0） |
| **高度 (Height)** | 从该节点到最远叶子节点的边数（叶子节点高度为0） |
| **树的度** | 树中节点的最大子节点数 |
| **子树** | 以某节点为根的局部树结构 |

### 树的高度与深度示意图

```
        A          ← 深度0，高度3（整树高度）
       / \
      B   C        ← 深度1，高度分别为1和2
     / \   \
    D   E   F      ← 深度2，高度分别为0、0、1
             \
              G    ← 深度3，高度0（叶子节点）
```

---

## 二叉树基础

二叉树是每个节点最多有两个子节点的树结构，是最重要、应用最广泛的树形结构。

### 二叉树的类型

```
📌 满二叉树：每个节点要么是叶子，要么有两个子节点
📌 完全二叉树：除最后一层外，每层节点数达到最大，最后一层从左到右填充
📌 完美二叉树：所有内部节点都有两个子节点，所有叶子在同一层
📌 斜树：所有节点都只有左子节点或只有右子节点
```

### 二叉树的存储方式

#### 1. 链式存储（推荐）

```cpp
// C++ 二叉树节点定义
struct TreeNode {
    int val;           // 节点值
    TreeNode* left;    // 左子节点指针
    TreeNode* right;   // 右子节点指针
    
    // 构造函数
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) 
        : val(x), left(left), right(right) {}
};
```

```python
# Python 二叉树节点定义
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val      # 节点值
        self.left = left    # 左子节点
        self.right = right  # 右子节点
```

#### 2. 数组存储（适用于完全二叉树）

对于完全二叉树，可以用数组存储，利用索引关系表示父子关系：

```
数组索引：  0  1  2  3  4  5  6
节点值：   [A, B, C, D, E, F, G]

父子关系（根节点索引从0开始）：
- 左子节点：parent * 2 + 1
- 右子节点：parent * 2 + 2
- 父节点：  (child - 1) // 2
```

```
        A(0)
       /    \
     B(1)   C(2)
    /  \    /  \
  D(3) E(4) F(5) G(6)
```

---

## 二叉树的遍历

遍历是二叉树最基本的操作，按照访问根节点的时机分为四种方式。

### 遍历方式对比

| 遍历方式 | 访问顺序 | 典型应用 |
|---------|---------|---------|
| **前序遍历** | 根 → 左 → 右 | 复制树、表达式前缀表示 |
| **中序遍历** | 左 → 根 → 右 | BST有序输出、表达式中缀表示 |
| **后序遍历** | 左 → 右 → 根 | 删除树、计算目录大小、表达式后缀表示 |
| **层序遍历** | 逐层从左到右 | 按层处理、找最短路径 |

### 遍历示意图

```
        1
       / \
      2   3
     / \
    4   5

前序遍历：1 → 2 → 4 → 5 → 3
中序遍历：4 → 2 → 5 → 1 → 3
后序遍历：4 → 5 → 2 → 3 → 1
层序遍历：1 → 2 → 3 → 4 → 5
```

### 递归遍历实现

```cpp
// C++ 递归遍历
class Solution {
public:
    // 前序遍历：根 → 左 → 右
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> result;
        preorder(root, result);
        return result;
    }
    
    void preorder(TreeNode* node, vector<int>& result) {
        if (node == nullptr) return;
        result.push_back(node->val);  // 1. 访问根节点
        preorder(node->left, result); // 2. 遍历左子树
        preorder(node->right, result);// 3. 遍历右子树
    }
    
    // 中序遍历：左 → 根 → 右
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        inorder(root, result);
        return result;
    }
    
    void inorder(TreeNode* node, vector<int>& result) {
        if (node == nullptr) return;
        inorder(node->left, result);  // 1. 遍历左子树
        result.push_back(node->val);  // 2. 访问根节点
        inorder(node->right, result); // 3. 遍历右子树
    }
    
    // 后序遍历：左 → 右 → 根
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> result;
        postorder(root, result);
        return result;
    }
    
    void postorder(TreeNode* node, vector<int>& result) {
        if (node == nullptr) return;
        postorder(node->left, result);  // 1. 遍历左子树
        postorder(node->right, result); // 2. 遍历右子树
        result.push_back(node->val);    // 3. 访问根节点
    }
};
```

```python
# Python 递归遍历
class Solution:
    # 前序遍历：根 → 左 → 右
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        self.preorder(root, result)
        return result
    
    def preorder(self, node: TreeNode, result: list):
        if not node:
            return
        result.append(node.val)      # 1. 访问根节点
        self.preorder(node.left, result)  # 2. 遍历左子树
        self.preorder(node.right, result) # 3. 遍历右子树
    
    # 中序遍历：左 → 根 → 右
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        self.inorder(root, result)
        return result
    
    def inorder(self, node: TreeNode, result: list):
        if not node:
            return
        self.inorder(node.left, result)   # 1. 遍历左子树
        result.append(node.val)           # 2. 访问根节点
        self.inorder(node.right, result)  # 3. 遍历右子树
    
    # 后序遍历：左 → 右 → 根
    def postorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        self.postorder(root, result)
        return result
    
    def postorder(self, node: TreeNode, result: list):
        if not node:
            return
        self.postorder(node.left, result)   # 1. 遍历左子树
        self.postorder(node.right, result)  # 2. 遍历右子树
        result.append(node.val)             # 3. 访问根节点
```

### 迭代遍历实现

💡 **核心思想**：使用栈模拟递归调用过程

```cpp
// C++ 迭代遍历
class Solution {
public:
    // 前序遍历（迭代）
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> result;
        if (root == nullptr) return result;
        
        stack<TreeNode*> stk;
        stk.push(root);
        
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            result.push_back(node->val);  // 访问根节点
            
            // 先压右子节点，再压左子节点（栈是后进先出）
            if (node->right) stk.push(node->right);
            if (node->left) stk.push(node->left);
        }
        return result;
    }
    
    // 中序遍历（迭代）
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> stk;
        TreeNode* cur = root;
        
        while (cur != nullptr || !stk.empty()) {
            // 一路向左，将路径上的节点压栈
            while (cur != nullptr) {
                stk.push(cur);
                cur = cur->left;
            }
            // 弹出栈顶节点并访问
            cur = stk.top();
            stk.pop();
            result.push_back(cur->val);
            // 转向右子树
            cur = cur->right;
        }
        return result;
    }
    
    // 后序遍历（迭代）- 修改前序遍历顺序
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> result;
        if (root == nullptr) return result;
        
        stack<TreeNode*> stk;
        stk.push(root);
        
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            result.push_back(node->val);
            
            // 前序是"根左右"，后序是"左右根"
            // 修改压栈顺序得到"根右左"，最后反转得到"左右根"
            if (node->left) stk.push(node->left);
            if (node->right) stk.push(node->right);
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```

```python
# Python 迭代遍历
class Solution:
    # 前序遍历（迭代）
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        if not root:
            return result
        
        stack = [root]
        while stack:
            node = stack.pop()
            result.append(node.val)  # 访问根节点
            
            # 先压右子节点，再压左子节点（栈是后进先出）
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result
    
    # 中序遍历（迭代）
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        stack = []
        cur = root
        
        while cur or stack:
            # 一路向左，将路径上的节点压栈
            while cur:
                stack.append(cur)
                cur = cur.left
            # 弹出栈顶节点并访问
            cur = stack.pop()
            result.append(cur.val)
            # 转向右子树
            cur = cur.right
        return result
    
    # 后序遍历（迭代）- 修改前序遍历顺序
    def postorderTraversal(self, root: TreeNode) -> list[int]:
        result = []
        if not root:
            return result
        
        stack = [root]
        while stack:
            node = stack.pop()
            result.append(node.val)
            
            # 前序是"根左右"，后序是"左右根"
            # 修改压栈顺序得到"根右左"，最后反转得到"左右根"
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        
        result.reverse()
        return result
```

### 层序遍历（BFS）

```cpp
// C++ 层序遍历
#include <queue>

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();  // 当前层的节点数
        vector<int> levelNodes;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            levelNodes.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(levelNodes);
    }
    return result;
}
```

```python
# Python 层序遍历
from collections import deque

def levelOrder(root: TreeNode) -> list[list[int]]:
    result = []
    if not root:
        return result
    
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # 当前层的节点数
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result
```

---

## 二叉搜索树 (BST)

二叉搜索树是一种特殊的二叉树，具有有序性质，支持高效的查找、插入和删除操作。

### BST 性质

```
📌 对于每个节点：
   - 左子树所有节点值 < 当前节点值
   - 右子树所有节点值 > 当前节点值
   - 中序遍历得到有序序列
```

### BST 示意图

```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13

中序遍历：1 → 3 → 4 → 6 → 7 → 8 → 10 → 13 → 14（有序！）
```

### BST 基本操作

```cpp
// C++ BST 操作
class BST {
private:
    TreeNode* root;
    
    // 递归查找
    TreeNode* search(TreeNode* node, int val) {
        if (node == nullptr || node->val == val) {
            return node;
        }
        if (val < node->val) {
            return search(node->left, val);  // 在左子树查找
        } else {
            return search(node->right, val); // 在右子树查找
        }
    }
    
    // 递归插入
    TreeNode* insert(TreeNode* node, int val) {
        if (node == nullptr) {
            return new TreeNode(val);  // 找到插入位置
        }
        if (val < node->val) {
            node->left = insert(node->left, val);
        } else if (val > node->val) {
            node->right = insert(node->right, val);
        }
        return node;  // 值已存在，不重复插入
    }
    
    // 找到最小节点（中序后继）
    TreeNode* findMin(TreeNode* node) {
        while (node->left != nullptr) {
            node = node->left;
        }
        return node;
    }
    
    // 递归删除
    TreeNode* remove(TreeNode* node, int val) {
        if (node == nullptr) return nullptr;
        
        if (val < node->val) {
            node->left = remove(node->left, val);
        } else if (val > node->val) {
            node->right = remove(node->right, val);
        } else {
            // 找到要删除的节点
            // 情况1&2：只有一个子节点或无子节点
            if (node->left == nullptr) {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            } else if (node->right == nullptr) {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }
            // 情况3：有两个子节点，用中序后继替换
            TreeNode* successor = findMin(node->right);
            node->val = successor->val;
            node->right = remove(node->right, successor->val);
        }
        return node;
    }
    
public:
    BST() : root(nullptr) {}
    
    void insert(int val) { root = insert(root, val); }
    void remove(int val) { root = remove(root, val); }
    bool search(int val) { return search(root, val) != nullptr; }
};
```

```python
# Python BST 操作
class BST:
    def __init__(self):
        self.root = None
    
    # 递归查找
    def _search(self, node: TreeNode, val: int) -> TreeNode:
        if not node or node.val == val:
            return node
        if val < node.val:
            return self._search(node.left, val)   # 在左子树查找
        else:
            return self._search(node.right, val)  # 在右子树查找
    
    # 递归插入
    def _insert(self, node: TreeNode, val: int) -> TreeNode:
        if not node:
            return TreeNode(val)  # 找到插入位置
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node  # 值已存在，不重复插入
    
    # 找到最小节点（中序后继）
    def _find_min(self, node: TreeNode) -> TreeNode:
        while node.left:
            node = node.left
        return node
    
    # 递归删除
    def _remove(self, node: TreeNode, val: int) -> TreeNode:
        if not node:
            return None
        
        if val < node.val:
            node.left = self._remove(node.left, val)
        elif val > node.val:
            node.right = self._remove(node.right, val)
        else:
            # 找到要删除的节点
            # 情况1&2：只有一个子节点或无子节点
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            # 情况3：有两个子节点，用中序后继替换
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._remove(node.right, successor.val)
        return node
    
    def insert(self, val: int):
        self.root = self._insert(self.root, val)
    
    def remove(self, val: int):
        self.root = self._remove(self.root, val)
    
    def search(self, val: int) -> bool:
        return self._search(self.root, val) is not None
```

### BST 时间复杂度

| 操作 | 平均情况 | 最坏情况 |
|------|---------|---------|
| 查找 | O(log n) | O(n) |
| 插入 | O(log n) | O(n) |
| 删除 | O(log n) | O(n) |

⚠️ **注意**：最坏情况发生在BST退化为链表时。为避免此问题，需要使用平衡二叉树（如AVL树、红黑树）。

---

## 平衡二叉树

平衡二叉树通过约束树的高度，保证操作的时间复杂度为 O(log n)。

### 常见平衡二叉树

| 类型 | 平衡条件 | 特点 |
|------|---------|------|
| **AVL树** | 任意节点左右子树高度差 ≤ 1 | 严格平衡，查找效率高 |
| **红黑树** | 满足红黑性质的BST | 近似平衡，插入删除效率高 |
| **B树/B+树** | 所有叶子节点在同一层 | 适合磁盘存储，数据库索引 |

### AVL树旋转操作

```
左旋（解决右重）：
    y                x
   / \              / \
  T1  x    →       y  T3
     / \          / \
    T2 T3        T1 T2

右旋（解决左重）：
      y                x
     / \              / \
    x  T3    →       T1  y
   / \                  / \
  T1 T2                T2 T3
```

---

## 堆 (Heap)

堆是一种特殊的完全二叉树，满足堆序性质。堆是实现优先队列的核心数据结构。

### 堆的性质

```
📌 大根堆：每个节点的值 ≥ 其子节点的值（根节点最大）
📌 小根堆：每个节点的值 ≤ 其子节点的值（根节点最小）
📌 堆是完全二叉树，通常用数组存储
```

### 大根堆示意图

```
        16          ← 最大值在根节点
       /  \
      14   10
     / \   / \
    8   7 9   3
   / \
  2   4

数组表示：[16, 14, 10, 8, 7, 9, 3, 2, 4]
```

### 堆的数组存储

```cpp
// 索引关系（根节点索引从0开始）
int parent(int i) { return (i - 1) / 2; }      // 父节点索引
int left(int i) { return 2 * i + 1; }          // 左子节点索引
int right(int i) { return 2 * i + 2; }         // 右子节点索引
```

### 堆的基本操作

```cpp
// C++ 大根堆实现
#include <vector>
#include <algorithm>

class MaxHeap {
private:
    std::vector<int> heap;
    
    // 上浮操作：将节点上浮到正确位置
    void siftUp(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[idx] <= heap[parent]) break;
            std::swap(heap[idx], heap[parent]);
            idx = parent;
        }
    }
    
    // 下沉操作：将节点下沉到正确位置
    void siftDown(int idx) {
        int n = heap.size();
        while (true) {
            int largest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            
            // 找出父节点和两个子节点中的最大值
            if (left < n && heap[left] > heap[largest]) {
                largest = left;
            }
            if (right < n && heap[right] > heap[largest]) {
                largest = right;
            }
            
            if (largest == idx) break;  // 已经满足堆性质
            std::swap(heap[idx], heap[largest]);
            idx = largest;
        }
    }
    
public:
    // 插入元素：添加到末尾，然后上浮
    void push(int val) {
        heap.push_back(val);
        siftUp(heap.size() - 1);
    }
    
    // 弹出堆顶：交换堆顶和末尾元素，删除末尾，然后下沉
    int pop() {
        if (heap.empty()) throw std::runtime_error("Heap is empty");
        int top = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) {
            siftDown(0);
        }
        return top;
    }
    
    // 获取堆顶元素
    int top() const {
        if (heap.empty()) throw std::runtime_error("Heap is empty");
        return heap[0];
    }
    
    // 建堆（从无序数组）
    void buildHeap(std::vector<int>& arr) {
        heap = arr;
        // 从最后一个非叶子节点开始，依次下沉
        for (int i = heap.size() / 2 - 1; i >= 0; i--) {
            siftDown(i);
        }
    }
    
    bool empty() const { return heap.empty(); }
    int size() const { return heap.size(); }
};
```

```python
# Python 小根堆实现
class MinHeap:
    def __init__(self):
        self.heap = []
    
    # 上浮操作：将节点上浮到正确位置
    def _sift_up(self, idx: int):
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx] >= self.heap[parent]:
                break
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            idx = parent
    
    # 下沉操作：将节点下沉到正确位置
    def _sift_down(self, idx: int):
        n = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            # 找出父节点和两个子节点中的最小值
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest == idx:
                break  # 已经满足堆性质
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            idx = smallest
    
    # 插入元素：添加到末尾，然后上浮
    def push(self, val: int):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)
    
    # 弹出堆顶：交换堆顶和末尾元素，删除末尾，然后下沉
    def pop(self) -> int:
        if not self.heap:
            raise IndexError("Heap is empty")
        top = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return top
    
    # 获取堆顶元素
    def top(self) -> int:
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    # 建堆（从无序数组）
    def build_heap(self, arr: list):
        self.heap = arr[:]
        # 从最后一个非叶子节点开始，依次下沉
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)
    
    def empty(self) -> bool:
        return len(self.heap) == 0
    
    def size(self) -> int:
        return len(self.heap)
```

### 堆排序

```cpp
// C++ 堆排序
void heapSort(vector<int>& arr) {
    int n = arr.size();
    
    // 建堆（大根堆）
    for (int i = n / 2 - 1; i >= 0; i--) {
        // 下沉操作
        auto siftDown = [&](int idx, int end) {
            while (true) {
                int largest = idx;
                int left = 2 * idx + 1;
                int right = 2 * idx + 2;
                
                if (left < end && arr[left] > arr[largest]) largest = left;
                if (right < end && arr[right] > arr[largest]) largest = right;
                
                if (largest == idx) break;
                swap(arr[idx], arr[largest]);
                idx = largest;
            }
        };
        siftDown(i, n);
    }
    
    // 排序：依次取出堆顶
    auto siftDown = [&](int idx, int end) {
        while (true) {
            int largest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            
            if (left < end && arr[left] > arr[largest]) largest = left;
            if (right < end && arr[right] > arr[largest]) largest = right;
            
            if (largest == idx) break;
            swap(arr[idx], arr[largest]);
            idx = largest;
        }
    };
    
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);  // 将堆顶移到末尾
        siftDown(0, i);        // 重新调整堆
    }
}
```

```python
# Python 堆排序
def heap_sort(arr: list) -> list:
    n = len(arr)
    
    def sift_down(idx: int, end: int):
        """下沉操作"""
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < end and arr[left] > arr[largest]:
                largest = left
            if right < end and arr[right] > arr[largest]:
                largest = right
            
            if largest == idx:
                break
            arr[idx], arr[largest] = arr[largest], arr[idx]
            idx = largest
    
    # 建堆（大根堆）
    for i in range(n // 2 - 1, -1, -1):
        sift_down(i, n)
    
    # 排序：依次取出堆顶
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 将堆顶移到末尾
        sift_down(0, i)                   # 重新调整堆
    
    return arr
```

### STL/C++ 标准库堆操作

```cpp
#include <vector>
#include <queue>
#include <algorithm>

// 方法1：使用 priority_queue
std::priority_queue<int> maxHeap;           // 大根堆
std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;  // 小根堆

maxHeap.push(3);
maxHeap.push(1);
maxHeap.push(4);
int top = maxHeap.top();   // 4
maxHeap.pop();

// 方法2：使用 vector + make_heap
std::vector<int> v = {3, 1, 4, 1, 5, 9};
std::make_heap(v.begin(), v.end());         // 建大根堆
std::push_heap(v.begin(), v.end());         // 插入后调整
std::pop_heap(v.begin(), v.end());          // 弹出前调整
v.pop_back();                               // 删除末尾元素
std::sort_heap(v.begin(), v.end());         // 堆排序
```

```python
# Python heapq 模块（小根堆）
import heapq

arr = [3, 1, 4, 1, 5, 9]
heapq.heapify(arr)          # 建堆（原地修改）
heapq.heappush(arr, 2)      # 插入元素
top = heapq.heappop(arr)    # 弹出最小值

# 大根堆技巧：取负数
max_heap = [-x for x in [3, 1, 4]]
heapq.heapify(max_heap)
top = -heapq.heappop(max_heap)  # 得到最大值
```

---

## 典型 LeetCode 题型

### 1. 二叉树的最大深度（LeetCode 104）

```cpp
// C++ 递归解法
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}

// C++ BFS 解法
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    queue<TreeNode*> q;
    q.push(root);
    int depth = 0;
    
    while (!q.empty()) {
        int levelSize = q.size();
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        depth++;
    }
    return depth;
}
```

```python
# Python 递归解法
def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

# Python BFS 解法
from collections import deque

def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        depth += 1
    
    return depth
```

### 2. 翻转二叉树（LeetCode 226）

```cpp
// C++ 递归解法
TreeNode* invertTree(TreeNode* root) {
    if (root == nullptr) return nullptr;
    
    // 交换左右子树
    swap(root->left, root->right);
    
    // 递归处理子树
    invertTree(root->left);
    invertTree(root->right);
    
    return root;
}
```

```python
# Python 递归解法
def invertTree(root: TreeNode) -> TreeNode:
    if not root:
        return None
    
    # 交换左右子树
    root.left, root.right = root.right, root.left
    
    # 递归处理子树
    invertTree(root.left)
    invertTree(root.right)
    
    return root
```

### 3. 验证二叉搜索树（LeetCode 98）

```cpp
// C++ 中序遍历法（BST中序遍历是有序的）
bool isValidBST(TreeNode* root) {
    stack<TreeNode*> stk;
    TreeNode* cur = root;
    long long prev = LLONG_MIN;  // 记录前一个节点的值
    
    while (cur != nullptr || !stk.empty()) {
        while (cur != nullptr) {
            stk.push(cur);
            cur = cur->left;
        }
        cur = stk.top();
        stk.pop();
        
        // 检查是否有序
        if (cur->val <= prev) return false;
        prev = cur->val;
        
        cur = cur->right;
    }
    return true;
}

// C++ 递归法（传递有效范围）
bool isValidBST(TreeNode* root) {
    return isValidBST(root, LLONG_MIN, LLONG_MAX);
}

bool isValidBST(TreeNode* node, long long minVal, long long maxVal) {
    if (node == nullptr) return true;
    
    // 当前节点值必须在 (minVal, maxVal) 范围内
    if (node->val <= minVal || node->val >= maxVal) {
        return false;
    }
    
    // 递归检查子树，更新范围
    return isValidBST(node->left, minVal, node->val) &&
           isValidBST(node->right, node->val, maxVal);
}
```

```python
# Python 中序遍历法
def isValidBST(root: TreeNode) -> bool:
    stack = []
    cur = root
    prev = float('-inf')  # 记录前一个节点的值
    
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        
        # 检查是否有序
        if cur.val <= prev:
            return False
        prev = cur.val
        
        cur = cur.right
    
    return True

# Python 递归法（传递有效范围）
def isValidBST(root: TreeNode) -> bool:
    def helper(node: TreeNode, min_val: float, max_val: float) -> bool:
        if not node:
            return True
        
        # 当前节点值必须在 (min_val, max_val) 范围内
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # 递归检查子树，更新范围
        return helper(node.left, min_val, node.val) and \
               helper(node.right, node.val, max_val)
    
    return helper(root, float('-inf'), float('inf'))
```

### 4. 二叉树的最近公共祖先（LeetCode 236）

```cpp
// C++ 递归解法
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    // 终止条件：找到p或q，或到达空节点
    if (root == nullptr || root == p || root == q) {
        return root;
    }
    
    // 在左右子树中查找
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    
    // 根据查找结果判断
    if (left != nullptr && right != nullptr) {
        return root;  // p和q分别在左右子树，当前节点就是LCA
    }
    if (left != nullptr) {
        return left;  // p和q都在左子树
    }
    return right;     // p和q都在右子树
}
```

```python
# Python 递归解法
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # 终止条件：找到p或q，或到达空节点
    if not root or root == p or root == q:
        return root
    
    # 在左右子树中查找
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    # 根据查找结果判断
    if left and right:
        return root   # p和q分别在左右子树，当前节点就是LCA
    if left:
        return left   # p和q都在左子树
    return right      # p和q都在右子树
```

### 5. Top K 问题（堆的应用）

```cpp
// C++ 找数组中第K大的元素（LeetCode 215）
int findKthLargest(vector<int>& nums, int k) {
    // 方法1：小根堆，维护K个最大元素
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();  // 弹出最小的
        }
    }
    
    return minHeap.top();  // 堆顶就是第K大的元素
}

// 方法2：快速选择算法（平均O(n)）
int findKthLargest(vector<int>& nums, int k) {
    int target = nums.size() - k;  // 第K大 = 升序排序后的第(n-k)个
    
    auto partition = [&](int left, int right) {
        int pivot = nums[right];
        int i = left;
        for (int j = left; j < right; j++) {
            if (nums[j] <= pivot) {
                swap(nums[i], nums[j]);
                i++;
            }
        }
        swap(nums[i], nums[right]);
        return i;
    };
    
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        int pos = partition(left, right);
        if (pos == target) {
            return nums[pos];
        } else if (pos < target) {
            left = pos + 1;
        } else {
            right = pos - 1;
        }
    }
    return nums[left];
}
```

```python
# Python 找数组中第K大的元素
import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    # 方法1：小根堆，维护K个最大元素
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # 弹出最小的
    return min_heap[0]  # 堆顶就是第K大的元素

# 方法2：大根堆（取负数）
def findKthLargest(nums: list[int], k: int) -> int:
    max_heap = [-x for x in nums]
    heapq.heapify(max_heap)
    for _ in range(k - 1):
        heapq.heappop(max_heap)
    return -heapq.heappop(max_heap)

# 方法3：快速选择算法
import random

def findKthLargest(nums: list[int], k: int) -> int:
    target = len(nums) - k  # 第K大 = 升序排序后的第(n-k)个
    
    def partition(left: int, right: int) -> int:
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    left, right = 0, len(nums) - 1
    while left < right:
        pos = partition(left, right)
        if pos == target:
            return nums[pos]
        elif pos < target:
            left = pos + 1
        else:
            right = pos - 1
    return nums[left]
```

---

## 常见问题与注意事项

### 1. 递归 vs 迭代的选择

| 方面 | 递归 | 迭代 |
|------|------|------|
| **代码简洁性** | ✅ 简洁直观 | ❌ 相对复杂 |
| **空间效率** | ❌ 栈空间 O(h) | ✅ 可优化到 O(1) |
| **栈溢出风险** | ❌ 深树可能溢出 | ✅ 无溢出风险 |
| **调试难度** | ❌ 较难调试 | ✅ 容易调试 |

### 2. 二叉树问题的常见思路

```
📌 递归思维：将问题分解为子树问题
📌 自底向上：后序遍历，先处理子树，再合并结果
📌 自顶向下：前序遍历，传递参数到子树
📌 BFS思维：层序遍历处理按层相关问题
```

### 3. 堆的使用场景

| 场景 | 推荐堆类型 |
|------|-----------|
| 找前K大元素 | 小根堆（大小为K） |
| 找前K小元素 | 大根堆（大小为K） |
| 合并K个有序链表 | 小根堆 |
| 实时获取中位数 | 大根堆 + 小根堆 |
| 任务调度 | 小根堆（按优先级） |

---

## 复杂度总结

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 二叉树遍历 | O(n) | O(h) 递归栈 |
| BST 查找/插入/删除 | O(h) 平均，O(n) 最坏 | O(h) 递归栈 |
| 堆 插入/删除 | O(log n) | O(1) |
| 堆排序 | O(n log n) | O(1) |
| 建堆 | O(n) | O(1) |

> **h** 为树的高度，**n** 为节点数量
