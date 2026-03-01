# 高级树结构：线段树、树状数组、字典树

高级树结构是在基础树结构之上发展出的高效数据结构，专门用于解决特定的算法问题。本文将介绍三种经典的高级树结构：线段树、树状数组和字典树，它们分别擅长处理区间查询、前缀计算和字符串检索问题。

<SegmentTreeAnimation />

---

## 线段树 (Segment Tree)

线段树是一种用于解决**区间问题**的高级数据结构，支持高效的区间查询和单点/区间更新操作。

### 核心思想

线段树将数组区间递归地划分为子区间，每个节点存储一个区间的信息（如区间和、最值等）：

```
📌 每个节点代表一个区间 [l, r]
📌 叶子节点代表单个元素
📌 非叶子节点的区间 = 左子节点区间 ∪ 右子节点区间
📌 树的高度约为 O(log n)
```

### 线段树结构示意图

```
原数组: [1, 3, 5, 7, 2, 4, 6, 8]

线段树（存储区间和）：

                [0-7]=36
               /        \
         [0-3]=16      [4-7]=20
         /    \         /    \
    [0-1]=4  [2-3]=12 [4-5]=6 [6-7]=14
    /   \    /   \    /   \   /   \
  [0]=1 [1]=3 [2]=5 [3]=7 [4]=2 [5]=4 [6]=6 [7]=8
```

### 存储方式

线段树通常使用数组存储，类似堆的存储方式：

```cpp
// C++ 线段树存储
// 对于 n 个元素的数组，线段树需要 4n 的空间
vector<int> tree(4 * n);  // 安全的上界

// 节点索引关系（根节点索引从 0 开始）
int leftChild(int i) { return 2 * i + 1; }   // 左子节点
int rightChild(int i) { return 2 * i + 2; }  // 右子节点
int parent(int i) { return (i - 1) / 2; }    // 父节点
```

```python
# Python 线段树存储
# 对于 n 个元素的数组，线段树需要 4n 的空间
tree = [0] * (4 * n)

# 节点索引关系（根节点索引从 0 开始）
def left_child(i): return 2 * i + 1
def right_child(i): return 2 * i + 2
def parent(i): return (i - 1) // 2
```

### 建树操作

时间复杂度：O(n)

```cpp
// C++ 建树
void build(vector<int>& arr, vector<int>& tree, int node, int l, int r) {
    if (l == r) {
        // 叶子节点：直接存储数组元素
        tree[node] = arr[l];
        return;
    }
    
    int mid = l + (r - l) / 2;
    int leftNode = 2 * node + 1;
    int rightNode = 2 * node + 2;
    
    // 递归构建左右子树
    build(arr, tree, leftNode, l, mid);
    build(arr, tree, rightNode, mid + 1, r);
    
    // 合并左右子树的结果
    tree[node] = tree[leftNode] + tree[rightNode];  // 区间和
    // tree[node] = max(tree[leftNode], tree[rightNode]);  // 区间最大值
    // tree[node] = min(tree[leftNode], tree[rightNode]);  // 区间最小值
}

// 使用示例
int n = arr.size();
vector<int> tree(4 * n);
build(arr, tree, 0, 0, n - 1);
```

```python
# Python 建树
def build(arr: list, tree: list, node: int, l: int, r: int):
    if l == r:
        # 叶子节点：直接存储数组元素
        tree[node] = arr[l]
        return
    
    mid = l + (r - l) // 2
    left_node = 2 * node + 1
    right_node = 2 * node + 2
    
    # 递归构建左右子树
    build(arr, tree, left_node, l, mid)
    build(arr, tree, right_node, mid + 1, r)
    
    # 合并左右子树的结果
    tree[node] = tree[left_node] + tree[right_node]  # 区间和

# 使用示例
n = len(arr)
tree = [0] * (4 * n)
build(arr, tree, 0, 0, n - 1)
```

### 区间查询操作

查询区间 [ql, qr] 的信息，时间复杂度：O(log n)

```cpp
// C++ 区间查询
int query(vector<int>& tree, int node, int l, int r, int ql, int qr) {
    // 情况1：查询区间与当前节点区间无交集
    if (qr < l || ql > r) {
        return 0;  // 对于求和，返回0（单位元）
        // return INT_MIN;  // 对于求最大值，返回负无穷
        // return INT_MAX;  // 对于求最小值，返回正无穷
    }
    
    // 情况2：当前节点区间完全在查询区间内
    if (ql <= l && r <= qr) {
        return tree[node];
    }
    
    // 情况3：部分重叠，递归查询左右子树
    int mid = l + (r - l) / 2;
    int leftNode = 2 * node + 1;
    int rightNode = 2 * node + 2;
    
    int leftSum = query(tree, leftNode, l, mid, ql, qr);
    int rightSum = query(tree, rightNode, mid + 1, r, ql, qr);
    
    return leftSum + rightSum;
}

// 使用示例
int result = query(tree, 0, 0, n - 1, 2, 5);  // 查询区间 [2, 5] 的和
```

```python
# Python 区间查询
def query(tree: list, node: int, l: int, r: int, ql: int, qr: int) -> int:
    # 情况1：查询区间与当前节点区间无交集
    if qr < l or ql > r:
        return 0  # 对于求和，返回0（单位元）
    
    # 情况2：当前节点区间完全在查询区间内
    if ql <= l and r <= qr:
        return tree[node]
    
    # 情况3：部分重叠，递归查询左右子树
    mid = l + (r - l) // 2
    left_node = 2 * node + 1
    right_node = 2 * node + 2
    
    left_sum = query(tree, left_node, l, mid, ql, qr)
    right_sum = query(tree, right_node, mid + 1, r, ql, qr)
    
    return left_sum + right_sum

# 使用示例
result = query(tree, 0, 0, n - 1, 2, 5)  # 查询区间 [2, 5] 的和
```

### 单点更新操作

更新某个位置的值，时间复杂度：O(log n)

```cpp
// C++ 单点更新
void update(vector<int>& tree, int node, int l, int r, int idx, int val) {
    // 找到叶子节点
    if (l == r) {
        tree[node] = val;
        return;
    }
    
    int mid = l + (r - l) / 2;
    int leftNode = 2 * node + 1;
    int rightNode = 2 * node + 2;
    
    // 根据位置决定更新哪个子树
    if (idx <= mid) {
        update(tree, leftNode, l, mid, idx, val);
    } else {
        update(tree, rightNode, mid + 1, r, idx, val);
    }
    
    // 更新当前节点的值
    tree[node] = tree[leftNode] + tree[rightNode];
}

// 使用示例
update(tree, 0, 0, n - 1, 3, 10);  // 将 arr[3] 更新为 10
```

```python
# Python 单点更新
def update(tree: list, node: int, l: int, r: int, idx: int, val: int):
    # 找到叶子节点
    if l == r:
        tree[node] = val
        return
    
    mid = l + (r - l) // 2
    left_node = 2 * node + 1
    right_node = 2 * node + 2
    
    # 根据位置决定更新哪个子树
    if idx <= mid:
        update(tree, left_node, l, mid, idx, val)
    else:
        update(tree, right_node, mid + 1, r, idx, val)
    
    # 更新当前节点的值
    tree[node] = tree[left_node] + tree[right_node]

# 使用示例
update(tree, 0, 0, n - 1, 3, 10)  # 将 arr[3] 更新为 10
```

### 懒标记 (Lazy Propagation)

懒标记是线段树的核心优化技术，用于实现高效的**区间更新**操作。

#### 核心思想

```
📌 区间更新时，不立即更新所有叶子节点
📌 而是在父节点上打一个"标记"，表示该区间需要更新
📌 当需要访问子节点时，才将标记下推（push down）
📌 这样可以将 O(n) 的区间更新优化为 O(log n)
```

#### 懒标记示意图

```
更新区间 [0, 3] 全部加 5：

            [0-7] (lazy=0)
           /      \
    [0-3]+5 ←打标记  [4-7]
    (lazy=5)
    /    \
  ...    ...

访问 [0-1] 时，将懒标记下推：
    [0-3] 的 lazy=5 下推到 [0-1] 和 [2-3]
```

#### 带懒标记的线段树实现

```cpp
// C++ 带懒标记的线段树
class LazySegmentTree {
private:
    vector<int> tree;   // 线段树数组
    vector<int> lazy;   // 懒标记数组
    int n;
    
    // 下推懒标记
    void pushDown(int node, int l, int r) {
        if (lazy[node] == 0) return;  // 无标记
        
        int mid = l + (r - l) / 2;
        int leftNode = 2 * node + 1;
        int rightNode = 2 * node + 2;
        
        // 将标记下推到子节点
        lazy[leftNode] += lazy[node];
        lazy[rightNode] += lazy[node];
        
        // 更新子节点的值
        tree[leftNode] += lazy[node] * (mid - l + 1);
        tree[rightNode] += lazy[node] * (r - mid);
        
        // 清除当前节点的标记
        lazy[node] = 0;
    }
    
public:
    LazySegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        lazy.resize(4 * n, 0);
        build(arr, 0, 0, n - 1);
    }
    
    void build(vector<int>& arr, int node, int l, int r) {
        if (l == r) {
            tree[node] = arr[l];
            return;
        }
        int mid = l + (r - l) / 2;
        build(arr, 2 * node + 1, l, mid);
        build(arr, 2 * node + 2, mid + 1, r);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    // 区间更新：[ul, ur] 区间每个元素加 val
    void updateRange(int node, int l, int r, int ul, int ur, int val) {
        // 无交集
        if (ur < l || ul > r) return;
        
        // 完全包含
        if (ul <= l && r <= ur) {
            tree[node] += val * (r - l + 1);  // 更新当前节点
            lazy[node] += val;                 // 打标记
            return;
        }
        
        // 部分重叠：先下推标记，再递归更新
        pushDown(node, l, r);
        int mid = l + (r - l) / 2;
        updateRange(2 * node + 1, l, mid, ul, ur, val);
        updateRange(2 * node + 2, mid + 1, r, ul, ur, val);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    // 区间查询
    int queryRange(int node, int l, int r, int ql, int qr) {
        if (qr < l || ql > r) return 0;
        if (ql <= l && r <= qr) return tree[node];
        
        pushDown(node, l, r);  // 查询前下推标记
        int mid = l + (r - l) / 2;
        return queryRange(2 * node + 1, l, mid, ql, qr) +
               queryRange(2 * node + 2, mid + 1, r, ql, qr);
    }
};
```

```python
# Python 带懒标记的线段树
class LazySegmentTree:
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr: list, node: int, l: int, r: int):
        if l == r:
            self.tree[node] = arr[l]
            return
        mid = l + (r - l) // 2
        self._build(arr, 2 * node + 1, l, mid)
        self._build(arr, 2 * node + 2, mid + 1, r)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    # 下推懒标记
    def _push_down(self, node: int, l: int, r: int):
        if self.lazy[node] == 0:
            return
        
        mid = l + (r - l) // 2
        left_node = 2 * node + 1
        right_node = 2 * node + 2
        
        # 将标记下推到子节点
        self.lazy[left_node] += self.lazy[node]
        self.lazy[right_node] += self.lazy[node]
        
        # 更新子节点的值
        self.tree[left_node] += self.lazy[node] * (mid - l + 1)
        self.tree[right_node] += self.lazy[node] * (r - mid)
        
        # 清除当前节点的标记
        self.lazy[node] = 0
    
    # 区间更新：[ul, ur] 区间每个元素加 val
    def update_range(self, ul: int, ur: int, val: int):
        self._update_range(0, 0, self.n - 1, ul, ur, val)
    
    def _update_range(self, node: int, l: int, r: int, ul: int, ur: int, val: int):
        # 无交集
        if ur < l or ul > r:
            return
        
        # 完全包含
        if ul <= l and r <= ur:
            self.tree[node] += val * (r - l + 1)  # 更新当前节点
            self.lazy[node] += val                 # 打标记
            return
        
        # 部分重叠：先下推标记，再递归更新
        self._push_down(node, l, r)
        mid = l + (r - l) // 2
        self._update_range(2 * node + 1, l, mid, ul, ur, val)
        self._update_range(2 * node + 2, mid + 1, r, ul, ur, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    # 区间查询
    def query_range(self, ql: int, qr: int) -> int:
        return self._query_range(0, 0, self.n - 1, ql, qr)
    
    def _query_range(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if qr < l or ql > r:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        
        self._push_down(node, l, r)  # 查询前下推标记
        mid = l + (r - l) // 2
        return self._query_range(2 * node + 1, l, mid, ql, qr) + \
               self._query_range(2 * node + 2, mid + 1, r, ql, qr)
```

### 线段树的应用场景

| 应用 | 存储信息 | 合并操作 |
|------|---------|---------|
| 区间求和 | 区间和 | 左右子树和相加 |
| 区间最值 | 区间最大/最小值 | 取左右子树最值 |
| 区间 GCD | 区间最大公约数 | gcd(左, 右) |
| 区间异或 | 区间异或值 | 左右异或值异或 |
| 区间众数 | 区间众数及出现次数 | 复杂合并逻辑 |

---

## 树状数组 (Binary Indexed Tree)

树状数组（也称 Fenwick Tree）是一种比线段树更简洁的数据结构，用于高效计算**前缀和**。

<BITAnimation />

### lowbit 原理

树状数组的核心是 `lowbit` 运算：取出一个数二进制表示中最低位的 1。

```cpp
// C++ lowbit 计算
int lowbit(int x) {
    return x & (-x);  // 利用补码特性
}

// 原理说明：
// x 的二进制：  000...101100
// -x 的二进制： 111...010100 (补码)
// x & (-x)：   000...000100 ← 最低位的1
```

```python
# Python lowbit 计算
def lowbit(x: int) -> int:
    return x & (-x)

# 示例：
# lowbit(6) = lowbit(0b110) = 0b010 = 2
# lowbit(8) = lowbit(0b1000) = 0b1000 = 8
```

### 树状数组结构

树状数组将原数组重新组织，每个节点 `t[i]` 存储一段区间的和：

```
📌 t[i] 存储区间 [i - lowbit(i) + 1, i] 的和
📌 区间长度 = lowbit(i)
📌 父节点：i + lowbit(i)
📌 前驱节点：i - lowbit(i)
```

### 树状数组示意图

```
原数组 a[]: [_, 1, 3, 5, 7, 2, 4, 6, 8]  （下标从1开始）

树状数组 t[]:
t[1] = a[1] = 1              （管理区间 [1,1]，长度 = lowbit(1) = 1）
t[2] = a[1] + a[2] = 4       （管理区间 [1,2]，长度 = lowbit(2) = 2）
t[3] = a[3] = 5              （管理区间 [3,3]，长度 = lowbit(3) = 1）
t[4] = a[1]+a[2]+a[3]+a[4] = 16 （管理区间 [1,4]，长度 = lowbit(4) = 4）
...

索引关系图：
        t[8]
       /    \
    t[4]    t[6]
   /   \    /   \
t[2] t[3] t[5] t[7]
 /
t[1]
```

### 单点更新操作

在位置 `i` 加上增量 `delta`，时间复杂度：O(log n)

```cpp
// C++ 单点更新
void update(vector<int>& t, int i, int delta) {
    int n = t.size() - 1;
    while (i <= n) {
        t[i] += delta;
        i += lowbit(i);  // 跳到父节点
    }
}

// 使用示例
update(t, 3, 5);  // a[3] += 5
```

```python
# Python 单点更新
def update(t: list, i: int, delta: int):
    n = len(t) - 1
    while i <= n:
        t[i] += delta
        i += lowbit(i)  # 跳到父节点

# 使用示例
update(t, 3, 5)  # a[3] += 5
```

### 前缀查询操作

查询前缀和 `sum[1..i]`，时间复杂度：O(log n)

```cpp
// C++ 前缀查询
int query(vector<int>& t, int i) {
    int sum = 0;
    while (i > 0) {
        sum += t[i];
        i -= lowbit(i);  // 跳到前驱节点
    }
    return sum;
}

// 区间查询 [l, r] = sum[1..r] - sum[1..l-1]
int rangeQuery(vector<int>& t, int l, int r) {
    return query(t, r) - query(t, l - 1);
}

// 使用示例
int prefixSum = query(t, 5);         // sum[1..5]
int rangeSum = rangeQuery(t, 2, 5);  // sum[2..5]
```

```python
# Python 前缀查询
def query(t: list, i: int) -> int:
    total = 0
    while i > 0:
        total += t[i]
        i -= lowbit(i)  # 跳到前驱节点
    return total

# 区间查询 [l, r] = sum[1..r] - sum[1..l-1]
def range_query(t: list, l: int, r: int) -> int:
    return query(t, r) - query(t, l - 1)

# 使用示例
prefix_sum = query(t, 5)          # sum[1..5]
range_sum = range_query(t, 2, 5)  # sum[2..5]
```

### 完整实现

```cpp
// C++ 树状数组完整实现
class BIT {
private:
    vector<int> t;
    int n;
    
    int lowbit(int x) { return x & (-x); }
    
public:
    BIT(int size) : n(size), t(size + 1, 0) {}
    
    // 从数组初始化
    BIT(vector<int>& arr) : n(arr.size() - 1), t(arr.size(), 0) {
        for (int i = 1; i <= n; i++) {
            update(i, arr[i]);
        }
    }
    
    // 单点更新：a[i] += delta
    void update(int i, int delta) {
        while (i <= n) {
            t[i] += delta;
            i += lowbit(i);
        }
    }
    
    // 前缀查询：sum[1..i]
    int query(int i) {
        int sum = 0;
        while (i > 0) {
            sum += t[i];
            i -= lowbit(i);
        }
        return sum;
    }
    
    // 区间查询：sum[l..r]
    int rangeQuery(int l, int r) {
        return query(r) - query(l - 1);
    }
};
```

```python
# Python 树状数组完整实现
class BIT:
    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)
    
    @staticmethod
    def lowbit(x: int) -> int:
        return x & (-x)
    
    # 从数组初始化
    @classmethod
    def from_array(cls, arr: list) -> 'BIT':
        bit = cls(len(arr) - 1)
        for i in range(1, len(arr)):
            bit.update(i, arr[i])
        return bit
    
    # 单点更新：a[i] += delta
    def update(self, i: int, delta: int):
        while i <= self.n:
            self.t[i] += delta
            i += self.lowbit(i)
    
    # 前缀查询：sum[1..i]
    def query(self, i: int) -> int:
        total = 0
        while i > 0:
            total += self.t[i]
            i -= self.lowbit(i)
        return total
    
    # 区间查询：sum[l..r]
    def range_query(self, l: int, r: int) -> int:
        return self.query(r) - self.query(l - 1)
```

### 线段树 vs 树状数组

| 特性 | 线段树 | 树状数组 |
|------|--------|---------|
| **空间复杂度** | O(4n) | O(n) |
| **代码复杂度** | 较复杂 | 简洁 |
| **单点更新** | O(log n) | O(log n) |
| **区间查询** | O(log n) | O(log n) |
| **区间更新** | ✅ 支持（懒标记） | ❌ 不直接支持 |
| **区间最值** | ✅ 支持 | ❌ 不支持 |
| **适用场景** | 通用区间问题 | 前缀和问题 |

💡 **选择建议**：如果只需要处理前缀和/区间和问题，优先使用树状数组；如果需要区间更新或区间最值，使用线段树。

---

## 字典树 (Trie)

字典树（Trie，也称前缀树）是一种专门用于字符串检索的树形数据结构。

<TrieAnimation />

### 核心思想

```
📌 根节点不包含字符，其他节点包含一个字符
📌 从根到某节点的路径表示一个字符串
📌 用 is_end 标记表示单词结尾
📌 查找效率只与字符串长度有关，与字典大小无关
```

### 字典树结构示意图

```
存储单词：["apple", "app", "apply", "ban", "banana"]

           (root)
          /      \
        a         b
        |         |
        p         a
        |         |
    ★   p         n ★
       / \        |
      l   ★       a
      |           |
      e           n
      |           |
    ★   y         a ★
```

> 标记 ★ 的节点表示单词结尾

### 节点定义

```cpp
// C++ Trie 节点定义
struct TrieNode {
    unordered_map<char, TrieNode*> children;  // 子节点映射
    bool is_end;                               // 是否为单词结尾
    
    TrieNode() : is_end(false) {}
};
```

```python
# Python Trie 节点定义
class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点映射
        self.is_end = False  # 是否为单词结尾
```

### 基本操作实现

```cpp
// C++ Trie 完整实现
class Trie {
private:
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    // 插入单词
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();  // 创建新节点
            }
            node = node->children[c];  // 移动到子节点
        }
        node->is_end = true;  // 标记单词结尾
    }
    
    // 搜索单词（完全匹配）
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;  // 字符不存在
            }
            node = node->children[c];
        }
        return node->is_end;  // 必须是单词结尾
    }
    
    // 判断是否存在以 prefix 为前缀的单词
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return true;  // 只要路径存在即可
    }
    
    // 删除单词（可选实现）
    bool remove(string word) {
        return removeHelper(root, word, 0);
    }
    
private:
    bool removeHelper(TrieNode* node, string& word, int idx) {
        if (idx == word.size()) {
            if (!node->is_end) return false;  // 单词不存在
            node->is_end = false;
            return node->children.empty();  // 是否可以删除此节点
        }
        
        char c = word[idx];
        if (node->children.find(c) == node->children.end()) {
            return false;  // 单词不存在
        }
        
        bool shouldDelete = removeHelper(node->children[c], word, idx + 1);
        
        if (shouldDelete) {
            delete node->children[c];
            node->children.erase(c);
            return node->children.empty() && !node->is_end;
        }
        return false;
    }
};
```

```python
# Python Trie 完整实现
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    # 插入单词
    def insert(self, word: str):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()  # 创建新节点
            node = node.children[c]  # 移动到子节点
        node.is_end = True  # 标记单词结尾
    
    # 搜索单词（完全匹配）
    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children:
                return False  # 字符不存在
            node = node.children[c]
        return node.is_end  # 必须是单词结尾
    
    # 判断是否存在以 prefix 为前缀的单词
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True  # 只要路径存在即可
    
    # 删除单词
    def remove(self, word: str) -> bool:
        def _remove(node: TrieNode, word: str, idx: int) -> bool:
            if idx == len(word):
                if not node.is_end:
                    return False  # 单词不存在
                node.is_end = False
                return len(node.children) == 0  # 是否可以删除此节点
            
            c = word[idx]
            if c not in node.children:
                return False  # 单词不存在
            
            should_delete = _remove(node.children[c], word, idx + 1)
            
            if should_delete:
                del node.children[c]
                return len(node.children) == 0 and not node.is_end
            return False
        
        return _remove(self.root, word, 0)
```

### 字典树的应用场景

| 应用场景 | 说明 |
|---------|------|
| **自动补全** | 输入前缀，快速找到所有可能的补全词 |
| **拼写检查** | 判断单词是否在字典中 |
| **IP 路由** | 最长前缀匹配 |
| **词频统计** | 每个节点存储经过该节点的单词数量 |
| **字符串排序** | 先序遍历字典树得到有序序列 |

### 时间与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 插入 | O(m) | m 为字符串长度 |
| 查找 | O(m) | m 为字符串长度 |
| 前缀匹配 | O(m) | m 为前缀长度 |
| 空间 | O(n × m) | n 为单词数，m 为平均长度 |

---

## 典型题型

### 题型一：区域和检索（线段树/树状数组）

**LeetCode 307. 区域和检索 - 数组可修改**

给定一个数组，支持两种操作：
1. 更新某个位置的值
2. 查询某个区间的和

```cpp
// C++ 解法（使用树状数组）
class NumArray {
private:
    vector<int> nums;
    vector<int> tree;
    int n;
    
    int lowbit(int x) { return x & (-x); }
    
    void add(int i, int delta) {
        while (i <= n) {
            tree[i] += delta;
            i += lowbit(i);
        }
    }
    
    int prefixSum(int i) {
        int sum = 0;
        while (i > 0) {
            sum += tree[i];
            i -= lowbit(i);
        }
        return sum;
    }
    
public:
    NumArray(vector<int>& nums) {
        this->n = nums.size();
        this->nums = nums;
        this->tree.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            add(i + 1, nums[i]);  // 树状数组下标从1开始
        }
    }
    
    void update(int index, int val) {
        int delta = val - nums[index];
        nums[index] = val;
        add(index + 1, delta);
    }
    
    int sumRange(int left, int right) {
        return prefixSum(right + 1) - prefixSum(left);
    }
};
```

```python
# Python 解法（使用线段树）
class NumArray:
    def __init__(self, nums: list):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums: list, node: int, l: int, r: int):
        if l == r:
            self.tree[node] = nums[l]
            return
        mid = l + (r - l) // 2
        self._build(nums, 2 * node + 1, l, mid)
        self._build(nums, 2 * node + 2, mid + 1, r)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def _update(self, node: int, l: int, r: int, idx: int, val: int):
        if l == r:
            self.tree[node] = val
            return
        mid = l + (r - l) // 2
        if idx <= mid:
            self._update(2 * node + 1, l, mid, idx, val)
        else:
            self._update(2 * node + 2, mid + 1, r, idx, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def _query(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if qr < l or ql > r:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = l + (r - l) // 2
        return self._query(2 * node + 1, l, mid, ql, qr) + \
               self._query(2 * node + 2, mid + 1, r, ql, qr)
    
    def update(self, index: int, val: int):
        self._update(0, 0, self.n - 1, index, val)
    
    def sumRange(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n - 1, left, right)
```

### 题型二：数组逆序对（树状数组）

**LeetCode 剑指 Offer 51. 数组中的逆序对**

计算数组中逆序对的数量。

💡 **思路**：从右向左遍历，用树状数组统计每个数右边有多少个比它小的数。

```cpp
// C++ 解法
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        // 离散化：将数值映射到 1~n 的范围
        vector<int> sorted = nums;
        sort(sorted.begin(), sorted.end());
        sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
        
        int n = sorted.size();
        vector<int> tree(n + 1, 0);
        
        auto lowbit = [](int x) { return x & (-x); };
        
        auto update = [&](int i) {
            while (i <= n) {
                tree[i]++;
                i += lowbit(i);
            }
        };
        
        auto query = [&](int i) {
            int sum = 0;
            while (i > 0) {
                sum += tree[i];
                i -= lowbit(i);
            }
            return sum;
        };
        
        long long ans = 0;
        // 从右向左遍历
        for (int i = nums.size() - 1; i >= 0; i--) {
            // 离散化后的值（1-based）
            int idx = lower_bound(sorted.begin(), sorted.end(), nums[i]) - sorted.begin() + 1;
            // 查询比当前数小的数的个数
            ans += query(idx - 1);
            // 更新树状数组
            update(idx);
        }
        
        return ans;
    }
};
```

```python
# Python 解法
class Solution:
    def reversePairs(self, nums: list) -> int:
        # 离散化
        sorted_nums = sorted(set(nums))
        rank = {v: i + 1 for i, v in enumerate(sorted_nums)}
        
        n = len(sorted_nums)
        tree = [0] * (n + 1)
        
        def lowbit(x):
            return x & (-x)
        
        def update(i):
            while i <= n:
                tree[i] += 1
                i += lowbit(i)
        
        def query(i):
            total = 0
            while i > 0:
                total += tree[i]
                i -= lowbit(i)
            return total
        
        ans = 0
        # 从右向左遍历
        for num in reversed(nums):
            r = rank[num]
            # 查询比当前数小的数的个数
            ans += query(r - 1)
            # 更新树状数组
            update(r)
        
        return ans
```

### 题型三：单词搜索（字典树）

**LeetCode 79. 单词搜索 + LeetCode 212. 单词搜索 II**

在二维网格中搜索单词。

```cpp
// C++ 解法（使用字典树优化）
class Solution {
private:
    struct TrieNode {
        TrieNode* children[26];
        string word;  // 在结尾节点存储完整单词
        
        TrieNode() {
            for (int i = 0; i < 26; i++) children[i] = nullptr;
            word = "";
        }
    };
    
    TrieNode* buildTrie(vector<string>& words) {
        TrieNode* root = new TrieNode();
        for (string& word : words) {
            TrieNode* node = root;
            for (char c : word) {
                int idx = c - 'a';
                if (node->children[idx] == nullptr) {
                    node->children[idx] = new TrieNode();
                }
                node = node->children[idx];
            }
            node->word = word;
        }
        return root;
    }
    
    vector<string> result;
    int m, n;
    
    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node) {
        char c = board[i][j];
        if (c == '#' || node->children[c - 'a'] == nullptr) return;
        
        node = node->children[c - 'a'];
        if (!node->word.empty()) {
            result.push_back(node->word);
            node->word = "";  // 避免重复添加
        }
        
        board[i][j] = '#';  // 标记已访问
        
        // 四个方向搜索
        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (auto& d : dirs) {
            int ni = i + d[0], nj = j + d[1];
            if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                dfs(board, ni, nj, node);
            }
        }
        
        board[i][j] = c;  // 恢复
    }
    
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode* root = buildTrie(words);
        m = board.size();
        n = board[0].size();
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, i, j, root);
            }
        }
        
        return result;
    }
};
```

```python
# Python 解法（使用字典树优化）
class Solution:
    def findWords(self, board: list, words: list) -> list:
        # 构建字典树
        root = {}
        for word in words:
            node = root
            for c in word:
                node = node.setdefault(c, {})
            node['#'] = word  # 在结尾存储完整单词
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i: int, j: int, node: dict):
            c = board[i][j]
            if c not in node:
                return
            
            curr = node[c]
            if '#' in curr:
                result.append(curr['#'])
                del curr['#']  # 避免重复添加
            
            board[i][j] = '$'  # 标记已访问
            
            # 四个方向搜索
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '$':
                    dfs(ni, nj, curr)
            
            board[i][j] = c  # 恢复
            
            # 清理空节点
            if not curr:
                del node[c]
        
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        
        return result
```

---

## 常见问题与注意事项

### 1. 数据结构选择指南

| 问题类型 | 推荐数据结构 | 原因 |
|---------|-------------|------|
| 区间求和 + 单点更新 | 树状数组 | 代码简洁，效率高 |
| 区间求和 + 区间更新 | 线段树（懒标记） | 树状数组不支持 |
| 区间最值 | 线段树 | 树状数组不支持 |
| 字符串前缀匹配 | 字典树 | 专门优化字符串检索 |
| 动态区间第K大 | 权值线段树 / 树套树 | 需要额外处理 |

### 2. 线段树注意事项

```
⚠️ 空间：数组大小至少开 4n
⚠️ 边界：注意 l, r, mid 的计算，避免死循环
⚠️ 懒标记：注意 pushDown 的时机和顺序
⚠️ 合并：不同问题需要不同的合并函数
```

### 3. 树状数组注意事项

```
⚠️ 下标：通常从 1 开始，注意与原数组下标的转换
⚠️ 初始化：需要先构建树，不能直接赋值
⚠️ 区间更新：需要使用差分技巧或树状数组套树状数组
⚠️ 离散化：处理大数值时需要先离散化
```

### 4. 字典树注意事项

```
⚠️ 空间：节点数量 = 所有字符串长度之和
⚠️ 删除：删除时要判断节点是否可以被删除
⚠️ 字符集：可以用数组或哈希表存储子节点
⚠️ 内存：大量字符串时注意内存使用
```

---

## 总结

| 数据结构 | 核心操作 | 时间复杂度 | 典型应用 |
|---------|---------|-----------|---------|
| **线段树** | 建树、查询、更新 | O(n), O(log n), O(log n) | 区间求和、区间最值、区间更新 |
| **树状数组** | 更新、前缀查询 | O(log n), O(log n) | 前缀和、逆序对、第K大 |
| **字典树** | 插入、查找、前缀匹配 | O(m), O(m), O(m) | 字符串检索、自动补全、词频统计 |

选择合适的数据结构是解决问题的关键，理解每种数据结构的特点和适用场景，才能在算法竞赛和实际开发中游刃有余。
