# 线性数据结构

线性数据结构是最基础、最常用的数据结构类型，其特点是数据元素之间存在**一对一**的线性关系。线性结构中的元素按顺序排列，每个元素（除首尾外）都有且只有一个前驱和一个后继。

📌 **为什么线性结构如此重要？**
- 是所有数据结构的基础，树、图等复杂结构都可以通过线性结构组合或扩展而来
- 是算法实现的基本工具，绝大多数算法都需要借助线性结构存储和操作数据
- 理解线性结构是学习高级数据结构的前提

---

## 数组 (Array)

### 存储原理与特性

**定义**：数组是一种**连续内存**存储的线性数据结构，所有元素类型相同，可通过下标在 O(1) 时间内直接访问。

**内存布局示意**：
```
内存地址:    1000    1004    1008    1012    1016
            ┌───────┬───────┬───────┬───────┬───────┐
索引:       │   0   │   1   │   2   │   3   │   4   │
            │  arr  │  arr  │  arr  │  arr  │  arr  │
            │  [0]  │  [1]  │  [2]  │  [3]  │  [4]  │
            └───────┴───────┴───────┴───────┴───────┘
```

**核心特性**：
| 特性 | 说明 |
|------|------|
| 连续存储 | 元素在内存中紧密相邻，无间隙 |
| 随机访问 | 通过基地址 + 偏移量直接计算地址 |
| 类型统一 | 所有元素占用相同大小的内存空间 |
| 缓存友好 | 连续内存访问对 CPU 缓存极其友好 |

💡 **随机访问的原理**
```
arr[i] 的内存地址 = 基地址 + i × 单个元素大小

例如：int arr[5] 存储在地址 1000 开始处
- arr[0] 地址 = 1000 + 0 × 4 = 1000
- arr[3] 地址 = 1000 + 3 × 4 = 1012
```

### 时间复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|------------|------|
| 访问元素 | O(1) | 直接地址计算 |
| 修改元素 | O(1) | 直接地址计算 |
| 查找元素 | O(n) | 需要遍历 |
| 插入元素 | O(n) | 需要移动后续元素 |
| 删除元素 | O(n) | 需要移动后续元素 |

### 动态数组扩容机制

静态数组创建后大小固定，而动态数组（如 C++ 的 `vector`、Python 的 `list`）通过**扩容机制**实现动态增长。

**扩容策略**：

```
初始容量: 4
插入顺序: 1, 2, 3, 4, 5, 6, 7, 8, 9

插入 1-4: [1, 2, 3, 4]  容量=4, 大小=4
插入 5:   触发扩容 → 容量变为 8
         [1, 2, 3, 4, 5, _, _, _]  容量=8, 大小=5
插入 6-8: [1, 2, 3, 4, 5, 6, 7, 8]  容量=8, 大小=8
插入 9:   触发扩容 → 容量变为 16
         [1, 2, 3, 4, 5, 6, 7, 8, 9, _, _, _, _, _, _, _]
```

**均摊分析**：
- 扩容时间复杂度为 O(n)，但分摊到每次插入操作，**均摊时间复杂度为 O(1)**
- 原理：每 n 次插入才触发一次扩容，扩容代价 n，均摊后每次 O(1)

::: code-group
```cpp
#include <iostream>
#include <vector>
using namespace std;

// 动态数组扩容演示
void demonstrateDynamicArray() {
    vector<int> arr;
    
    cout << "初始状态: 容量=" << arr.capacity() << ", 大小=" << arr.size() << endl;
    
    for (int i = 1; i <= 20; i++) {
        arr.push_back(i);
        // 当容量变化时输出
        if (arr.capacity() != (size_t)(i <= 1 ? 1 : arr.capacity())) {
            cout << "插入 " << i << " 后: 容量=" << arr.capacity() 
                 << ", 大小=" << arr.size() << endl;
        }
    }
}

// 自定义动态数组实现
template<typename T>
class DynamicArray {
private:
    T* data;
    int capacity;
    int size;
    
    void resize(int newCapacity) {
        T* newData = new T[newCapacity];
        for (int i = 0; i < size; i++) {
            newData[i] = data[i];
        }
        delete[] data;
        data = newData;
        capacity = newCapacity;
    }
    
public:
    DynamicArray() : capacity(4), size(0) {
        data = new T[capacity];
    }
    
    ~DynamicArray() {
        delete[] data;
    }
    
    void push_back(const T& value) {
        if (size == capacity) {
            resize(capacity * 2);  // 2倍扩容策略
        }
        data[size++] = value;
    }
    
    T& operator[](int index) {
        return data[index];
    }
    
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
};
```

```python
class DynamicArray:
    """动态数组实现"""
    
    def __init__(self):
        self.capacity = 4  # 初始容量
        self.size = 0      # 当前元素数量
        self.data = [None] * self.capacity
    
    def _resize(self, new_capacity):
        """扩容到新容量"""
        new_data = [None] * new_capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_capacity
    
    def append(self, value):
        """添加元素到末尾"""
        if self.size == self.capacity:
            self._resize(self.capacity * 2)  # 2倍扩容
        self.data[self.size] = value
        self.size += 1
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.data[:self.size])


# 演示扩容过程
def demonstrate_capacity_growth():
    arr = []
    capacities = []
    
    for i in range(1, 21):
        arr.append(i)
        # Python list 没有直接获取容量的方法
        # 但可以通过 __sizeof__() 估算
        capacities.append((i, arr.__sizeof__()))
    
    print("Python list 扩容过程中的内存大小变化:")
    for i, size in capacities[::4]:  # 每4个打印一次
        print(f"  元素数量={i}, 内存大小={size}")


if __name__ == "__main__":
    demonstrate_capacity_growth()
```
:::

### 典型应用场景

| 场景 | 说明 | 示例 |
|------|------|------|
| 随机访问频繁 | O(1) 访问是最大优势 | 二分查找、前缀和 |
| 数据量固定 | 避免动态扩容开销 | 图的邻接矩阵 |
| 缓存敏感场景 | 连续内存提高缓存命中率 | 数值计算、图像处理 |
| 多维数据处理 | 天然支持多维扩展 | 矩阵、张量运算 |

### 典型 LeetCode 题型

| 题目 | 难度 | 核心技巧 |
|------|------|----------|
| [两数之和](https://leetcode.cn/problems/two-sum/) | 简单 | 哈希表优化查找 |
| [移动零](https://leetcode.cn/problems/move-zeroes/) | 简单 | 双指针 |
| [合并区间](https://leetcode.cn/problems/merge-intervals/) | 中等 | 排序 + 贪心 |
| [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | 困难 | 单调队列 |

---

## 链表 (Linked List)

### 基本概念

**定义**：链表是一种通过**指针连接**的线性数据结构，元素（节点）在内存中可以**不连续存储**。

**节点结构**：
```
单链表节点:
┌─────────┬─────────┐
│  data   │  next   │
│ (数据域) │ (指针域) │
└─────────┴─────────┘
     │
     ▼
  指向下一个节点

双链表节点:
┌─────────┬─────────┬─────────┐
│  prev   │  data   │  next   │
└─────────┴─────────┴─────────┘
```

### 链表的类型

| 类型 | 结构 | 特点 | 适用场景 |
|------|------|------|----------|
| 单链表 | `data + next` | 只能单向遍历 | LRU 缓存、链式栈 |
| 双链表 | `prev + data + next` | 可双向遍历 | LRU 缓存、浏览器历史 |
| 循环链表 | 尾节点指向头节点 | 形成环状 | 约瑟夫问题、轮转调度 |

**链表 vs 数组**：
| 特性 | 数组 | 链表 |
|------|------|------|
| 内存布局 | 连续 | 离散 |
| 随机访问 | ✅ O(1) | ❌ O(n) |
| 头部插入 | ❌ O(n) | ✅ O(1) |
| 内存开销 | 仅数据 | 数据 + 指针 |
| 缓存性能 | 优秀 | 较差 |

### 基本实现

::: code-group
```cpp
#include <iostream>
using namespace std;

// 单链表节点
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// 双链表节点
struct DoublyListNode {
    int val;
    DoublyListNode* prev;
    DoublyListNode* next;
    DoublyListNode(int x) : val(x), prev(nullptr), next(nullptr) {}
};

// 链表基本操作
class LinkedList {
public:
    // 头插法
    static ListNode* insertAtHead(ListNode* head, int val) {
        ListNode* newNode = new ListNode(val);
        newNode->next = head;
        return newNode;
    }
    
    // 尾插法
    static ListNode* insertAtTail(ListNode* head, int val) {
        ListNode* newNode = new ListNode(val);
        if (!head) return newNode;
        
        ListNode* curr = head;
        while (curr->next) {
            curr = curr->next;
        }
        curr->next = newNode;
        return head;
    }
    
    // 在指定位置插入
    static ListNode* insert(ListNode* head, int val, int pos) {
        if (pos == 0) return insertAtHead(head, val);
        
        ListNode* curr = head;
        for (int i = 0; i < pos - 1 && curr; i++) {
            curr = curr->next;
        }
        if (curr) {
            ListNode* newNode = new ListNode(val);
            newNode->next = curr->next;
            curr->next = newNode;
        }
        return head;
    }
    
    // 删除指定位置的节点
    static ListNode* remove(ListNode* head, int pos) {
        if (!head) return nullptr;
        
        if (pos == 0) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
            return head;
        }
        
        ListNode* curr = head;
        for (int i = 0; i < pos - 1 && curr->next; i++) {
            curr = curr->next;
        }
        if (curr->next) {
            ListNode* temp = curr->next;
            curr->next = temp->next;
            delete temp;
        }
        return head;
    }
    
    // 查找元素
    static int find(ListNode* head, int val) {
        ListNode* curr = head;
        int pos = 0;
        while (curr) {
            if (curr->val == val) return pos;
            curr = curr->next;
            pos++;
        }
        return -1;  // 未找到
    }
};
```

```python
from typing import Optional, List

class ListNode:
    """单链表节点"""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"


class DoublyListNode:
    """双链表节点"""
    def __init__(self, val: int = 0):
        self.val = val
        self.prev = None
        self.next = None


class LinkedList:
    """链表基本操作"""
    
    @staticmethod
    def create_from_list(arr: List[int]) -> Optional[ListNode]:
        """从数组创建链表"""
        if not arr:
            return None
        head = ListNode(arr[0])
        curr = head
        for val in arr[1:]:
            curr.next = ListNode(val)
            curr = curr.next
        return head
    
    @staticmethod
    def to_list(head: Optional[ListNode]) -> List[int]:
        """链表转数组"""
        result = []
        curr = head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result
    
    @staticmethod
    def insert_at_head(head: Optional[ListNode], val: int) -> ListNode:
        """头插法"""
        new_node = ListNode(val)
        new_node.next = head
        return new_node
    
    @staticmethod
    def insert_at_tail(head: Optional[ListNode], val: int) -> ListNode:
        """尾插法"""
        new_node = ListNode(val)
        if not head:
            return new_node
        curr = head
        while curr.next:
            curr = curr.next
        curr.next = new_node
        return head
    
    @staticmethod
    def delete_by_value(head: Optional[ListNode], val: int) -> Optional[ListNode]:
        """按值删除节点"""
        # 使用虚拟头节点简化操作
        dummy = ListNode(0, head)
        curr = dummy
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return dummy.next


# 测试
if __name__ == "__main__":
    # 创建链表
    arr = [1, 2, 3, 4, 5]
    head = LinkedList.create_from_list(arr)
    print("原链表:", LinkedList.to_list(head))
    
    # 头插
    head = LinkedList.insert_at_head(head, 0)
    print("头插0:", LinkedList.to_list(head))
    
    # 尾插
    head = LinkedList.insert_at_tail(head, 6)
    print("尾插6:", LinkedList.to_list(head))
```
:::

### 经典操作：反转链表

::: code-group
```cpp
// 迭代法反转链表
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr) {
        ListNode* nextTemp = curr->next;  // 保存下一个节点
        curr->next = prev;                 // 反转指针
        prev = curr;                       // 前移 prev
        curr = nextTemp;                   // 前移 curr
    }
    
    return prev;
}

// 递归法反转链表
ListNode* reverseListRecursive(ListNode* head) {
    if (!head || !head->next) return head;
    
    ListNode* newHead = reverseListRecursive(head->next);
    head->next->next = head;  // 反转指针
    head->next = nullptr;      // 断开原指针
    
    return newHead;
}
```

```python
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """迭代法反转链表"""
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next  # 保存下一个节点
        curr.next = prev       # 反转指针
        prev = curr            # 前移 prev
        curr = next_temp       # 前移 curr
    
    return prev


def reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """递归法反转链表"""
    if not head or not head.next:
        return head
    
    new_head = reverse_list_recursive(head.next)
    head.next.next = head  # 反转指针
    head.next = None       # 断开原指针
    
    return new_head
```
:::

### 经典技巧：快慢指针

快慢指针是链表问题的核心技巧，通过两个不同速度的指针解决多种问题。

**应用场景**：
1. 找链表中点
2. 检测链表是否有环
3. 找环的入口节点
4. 删除倒数第 N 个节点

::: code-group
```cpp
// 找链表中点
ListNode* findMiddle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast && fast->next) {
        slow = slow->next;          // 慢指针走一步
        fast = fast->next->next;    // 快指针走两步
    }
    
    return slow;  // 慢指针指向中点
}

// 检测链表是否有环
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) return true;  // 相遇则有环
    }
    
    return false;
}

// 找环的入口节点
ListNode* detectCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    // 第一次相遇
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) break;
    }
    
    if (!fast || !fast->next) return nullptr;  // 无环
    
    // 从头和相遇点同时出发，再次相遇即为入口
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow;
}

// 删除倒数第N个节点
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    
    ListNode* fast = dummy;
    ListNode* slow = dummy;
    
    // 快指针先走 n 步
    for (int i = 0; i < n; i++) {
        fast = fast->next;
    }
    
    // 同时移动直到快指针到达末尾
    while (fast->next) {
        slow = slow->next;
        fast = fast->next;
    }
    
    // 删除节点
    ListNode* toDelete = slow->next;
    slow->next = slow->next->next;
    delete toDelete;
    
    return dummy->next;
}
```

```python
def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """找链表中点"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def has_cycle(head: Optional[ListNode]) -> bool:
    """检测链表是否有环"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """找环的入口节点"""
    slow = fast = head
    
    # 第一次相遇
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 无环
    
    # 从头和相遇点同时出发
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """删除倒数第N个节点"""
    dummy = ListNode(0, head)
    fast = slow = dummy
    
    # 快指针先走 n 步
    for _ in range(n):
        fast = fast.next
    
    # 同时移动
    while fast.next:
        slow = slow.next
        fast = fast.next
    
    # 删除节点
    slow.next = slow.next.next
    return dummy.next
```
:::

### 经典技巧：合并链表

::: code-group
```cpp
// 合并两个有序链表
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            tail->next = l1;
            l1 = l1->next;
        } else {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;
    }
    
    tail->next = l1 ? l1 : l2;  // 连接剩余部分
    return dummy.next;
}
```

```python
def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """合并两个有序链表"""
    dummy = ListNode(0)
    tail = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    
    tail.next = l1 if l1 else l2
    return dummy.next
```
:::

### 典型 LeetCode 题型

| 题目 | 难度 | 核心技巧 |
|------|------|----------|
| [反转链表](https://leetcode.cn/problems/reverse-linked-list/) | 简单 | 双指针、递归 |
| [合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/) | 简单 | 双指针 |
| [环形链表](https://leetcode.cn/problems/linked-list-cycle/) | 简单 | 快慢指针 |
| [删除链表倒数第N个节点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) | 中等 | 快慢指针 |
| [链表排序](https://leetcode.cn/problems/sort-list/) | 中等 | 归并排序 |
| [合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) | 困难 | 优先队列、分治 |

---

## 栈 (Stack)

### LIFO 特性

**定义**：栈是一种**后进先出 (LIFO, Last In First Out)** 的线性数据结构，只允许在栈顶进行插入和删除操作。

```
栈的操作示意：

入栈 push:        出栈 pop:
    ┌───┐              ┌───┐ ← 被移除
    │ C │ ← 栈顶       │ C │
    ├───┤              ├───┤
    │ B │              │ B │
    ├───┤              ├───┤
    │ A │ ← 栈底       │ A │
    └───┘              └───┘
```

**核心特性**：
| 特性 | 说明 |
|------|------|
| 单端操作 | 所有操作仅在栈顶进行 |
| 顺序访问 | 只能按 LIFO 顺序访问元素 |
| 受限访问 | 无法直接访问栈底或中间元素 |

### 栈的实现

::: code-group
```cpp
#include <stack>
#include <vector>
#include <stdexcept>
using namespace std;

// 数组实现栈
class ArrayStack {
private:
    vector<int> data;
    
public:
    void push(int x) {
        data.push_back(x);
    }
    
    int pop() {
        if (empty()) throw runtime_error("Stack is empty");
        int top = data.back();
        data.pop_back();
        return top;
    }
    
    int top() {
        if (empty()) throw runtime_error("Stack is empty");
        return data.back();
    }
    
    bool empty() {
        return data.empty();
    }
    
    int size() {
        return data.size();
    }
};

// 链表实现栈
struct StackNode {
    int val;
    StackNode* next;
    StackNode(int x) : val(x), next(nullptr) {}
};

class LinkedStack {
private:
    StackNode* topNode;
    int stackSize;
    
public:
    LinkedStack() : topNode(nullptr), stackSize(0) {}
    
    void push(int x) {
        StackNode* newNode = new StackNode(x);
        newNode->next = topNode;
        topNode = newNode;
        stackSize++;
    }
    
    int pop() {
        if (empty()) throw runtime_error("Stack is empty");
        int val = topNode->val;
        StackNode* temp = topNode;
        topNode = topNode->next;
        delete temp;
        stackSize--;
        return val;
    }
    
    int top() {
        if (empty()) throw runtime_error("Stack is empty");
        return topNode->val;
    }
    
    bool empty() {
        return topNode == nullptr;
    }
    
    int size() {
        return stackSize;
    }
};

// STL 栈使用
void stlStackDemo() {
    stack<int> st;
    st.push(1);
    st.push(2);
    st.push(3);
    
    cout << "栈顶: " << st.top() << endl;  // 3
    st.pop();
    cout << "栈顶: " << st.top() << endl;  // 2
    cout << "大小: " << st.size() << endl; // 2
}
```

```python
class ArrayStack:
    """数组实现栈"""
    
    def __init__(self):
        self.data = []
    
    def push(self, x: int) -> None:
        self.data.append(x)
    
    def pop(self) -> int:
        if self.empty():
            raise Exception("Stack is empty")
        return self.data.pop()
    
    def top(self) -> int:
        if self.empty():
            raise Exception("Stack is empty")
        return self.data[-1]
    
    def empty(self) -> bool:
        return len(self.data) == 0
    
    def size(self) -> int:
        return len(self.data)


class LinkedStack:
    """链表实现栈"""
    
    class Node:
        def __init__(self, val: int):
            self.val = val
            self.next = None
    
    def __init__(self):
        self.top_node = None
        self.stack_size = 0
    
    def push(self, x: int) -> None:
        new_node = self.Node(x)
        new_node.next = self.top_node
        self.top_node = new_node
        self.stack_size += 1
    
    def pop(self) -> int:
        if self.empty():
            raise Exception("Stack is empty")
        val = self.top_node.val
        self.top_node = self.top_node.next
        self.stack_size -= 1
        return val
    
    def top(self) -> int:
        if self.empty():
            raise Exception("Stack is empty")
        return self.top_node.val
    
    def empty(self) -> bool:
        return self.top_node is None


# Python 内置列表模拟栈
stack = []
stack.append(1)    # push
stack.append(2)
stack.append(3)
print(stack[-1])   # top: 3
stack.pop()        # pop
print(stack[-1])   # top: 2
```
:::

### 单调栈 ⭐

**定义**：单调栈是一种特殊的栈，栈内元素保持**单调递增**或**单调递减**的顺序。

**核心思想**：
- 单调递增栈：栈底到栈顶元素递增（用于找下一个更小元素）
- 单调递减栈：栈底到栈顶元素递减（用于找下一个更大元素）

**应用场景**：
- 下一个更大/更小元素
- 柱状图最大矩形
- 接雨水问题

::: code-group
```cpp
#include <vector>
#include <stack>
using namespace std;

// 单调递减栈：找下一个更大元素
vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);  // 默认 -1 表示不存在
    stack<int> st;  // 存储索引，保持栈内元素递减
    
    for (int i = 0; i < n; i++) {
        // 当前元素比栈顶大，说明找到了栈顶元素的下一个更大元素
        while (!st.empty() && nums[i] > nums[st.top()]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}
// 示例: [2, 1, 2, 4, 3] → [4, 2, 4, -1, -1]

// 单调递增栈：找下一个更小元素
vector<int> nextSmallerElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;  // 存储索引，保持栈内元素递增
    
    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[i] < nums[st.top()]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}

// 经典题：柱状图中最大矩形
int largestRectangleArea(vector<int>& heights) {
    heights.push_back(0);  // 添加哨兵，确保最后能清空栈
    stack<int> st;
    int maxArea = 0;
    
    for (int i = 0; i < heights.size(); i++) {
        while (!st.empty() && heights[i] < heights[st.top()]) {
            int h = heights[st.top()];
            st.pop();
            int w = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        st.push(i);
    }
    
    return maxArea;
}

// 经典题：接雨水
int trap(vector<int>& height) {
    stack<int> st;
    int water = 0;
    
    for (int i = 0; i < height.size(); i++) {
        while (!st.empty() && height[i] > height[st.top()]) {
            int mid = st.top();
            st.pop();
            
            if (!st.empty()) {
                int h = min(height[st.top()], height[i]) - height[mid];
                int w = i - st.top() - 1;
                water += h * w;
            }
        }
        st.push(i);
    }
    
    return water;
}
```

```python
from typing import List

def next_greater_element(nums: List[int]) -> List[int]:
    """单调递减栈：找下一个更大元素"""
    n = len(nums)
    result = [-1] * n
    stack = []  # 存储索引，保持栈内元素递减
    
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result
# 示例: [2, 1, 2, 4, 3] → [4, 2, 4, -1, -1]


def next_smaller_element(nums: List[int]) -> List[int]:
    """单调递增栈：找下一个更小元素"""
    n = len(nums)
    result = [-1] * n
    stack = []  # 存储索引，保持栈内元素递增
    
    for i in range(n):
        while stack and nums[i] < nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result


def largest_rectangle_area(heights: List[int]) -> int:
    """柱状图中最大矩形"""
    heights.append(0)  # 添加哨兵
    stack = []
    max_area = 0
    
    for i, h in enumerate(heights):
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    heights.pop()  # 恢复原数组
    return max_area


def trap(height: List[int]) -> int:
    """接雨水"""
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            mid = stack.pop()
            if stack:
                h_min = min(height[stack[-1]], h)
                water += (h_min - height[mid]) * (i - stack[-1] - 1)
        stack.append(i)
    
    return water


# 测试
if __name__ == "__main__":
    print("下一个更大元素:", next_greater_element([2, 1, 2, 4, 3]))
    print("最大矩形:", largest_rectangle_area([2, 1, 5, 6, 2, 3]))
```
:::

### 经典应用：括号匹配

::: code-group
```cpp
// 括号匹配
bool isValid(string s) {
    stack<char> st;
    unordered_map<char, char> mapping = {
        {')', '('},
        {']', '['},
        {'}', '{'}
    };
    
    for (char c : s) {
        if (mapping.count(c)) {  // 右括号
            if (st.empty() || st.top() != mapping[c]) {
                return false;
            }
            st.pop();
        } else {  // 左括号
            st.push(c);
        }
    }
    
    return st.empty();
}

// 表达式求值（逆波兰表达式）
int evalRPN(vector<string>& tokens) {
    stack<int> st;
    
    for (const string& token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();
            if (token == "+") st.push(a + b);
            else if (token == "-") st.push(a - b);
            else if (token == "*") st.push(a * b);
            else st.push(a / b);
        } else {
            st.push(stoi(token));
        }
    }
    
    return st.top();
}
```

```python
def is_valid(s: str) -> bool:
    """括号匹配"""
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    
    for c in s:
        if c in mapping:  # 右括号
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:  # 左括号
            stack.append(c)
    
    return len(stack) == 0


def eval_rpn(tokens: List[str]) -> int:
    """逆波兰表达式求值"""
    stack = []
    operators = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)  # 整数除法
    }
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            stack.append(operators[token](a, b))
        else:
            stack.append(int(token))
    
    return stack[0]
```
:::

### 典型 LeetCode 题型

| 题目 | 难度 | 核心技巧 |
|------|------|----------|
| [有效的括号](https://leetcode.cn/problems/valid-parentheses/) | 简单 | 栈匹配 |
| [最小栈](https://leetcode.cn/problems/min-stack/) | 中等 | 辅助栈 |
| [每日温度](https://leetcode.cn/problems/daily-temperatures/) | 中等 | 单调栈 |
| [柱状图中最大矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/) | 困难 | 单调栈 |
| [接雨水](https://leetcode.cn/problems/trapping-rain-water/) | 困难 | 单调栈/双指针 |

---

## 队列 (Queue)

### FIFO 特性

**定义**：队列是一种**先进先出 (FIFO, First In First Out)** 的线性数据结构，在队尾插入元素，在队头删除元素。

```
队列操作示意：

入队 enqueue:     出队 dequeue:
                    ┌───┐ ← 被移除
┌───┬───┬───┐      │ A │
│ A │ B │ C │  →   ├───┤
└───┴───┴───┘      │ B │
 ↑       ↑         ├───┤
队头    队尾        │ C │
                    └───┘
```

### 队列的实现

::: code-group
```cpp
#include <queue>
#include <deque>
#include <vector>
using namespace std;

// 数组实现循环队列
class CircularQueue {
private:
    vector<int> data;
    int front;
    int rear;
    int capacity;
    
public:
    CircularQueue(int k) : front(0), rear(0), capacity(k + 1) {
        data.resize(capacity);
    }
    
    bool enqueue(int value) {
        if (isFull()) return false;
        data[rear] = value;
        rear = (rear + 1) % capacity;
        return true;
    }
    
    bool dequeue() {
        if (isEmpty()) return false;
        front = (front + 1) % capacity;
        return true;
    }
    
    int Front() {
        return isEmpty() ? -1 : data[front];
    }
    
    int Rear() {
        return isEmpty() ? -1 : data[(rear - 1 + capacity) % capacity];
    }
    
    bool isEmpty() {
        return front == rear;
    }
    
    bool isFull() {
        return (rear + 1) % capacity == front;
    }
};

// STL 队列使用
void stlQueueDemo() {
    queue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);
    
    cout << "队头: " << q.front() << endl;  // 1
    cout << "队尾: " << q.back() << endl;   // 3
    q.pop();
    cout << "队头: " << q.front() << endl;  // 2
}

// STL 双端队列
void stlDequeDemo() {
    deque<int> dq;
    dq.push_back(1);   // 尾部插入
    dq.push_front(0);  // 头部插入
    dq.pop_back();     // 尾部删除
    dq.pop_front();    // 头部删除
    
    // 支持随机访问
    cout << dq[0] << endl;
}
```

```python
from collections import deque
from typing import Optional, List

class CircularQueue:
    """循环队列实现"""
    
    def __init__(self, k: int):
        self.capacity = k + 1  # 多留一个空位区分满和空
        self.data = [0] * self.capacity
        self.front = 0
        self.rear = 0
    
    def enqueue(self, value: int) -> bool:
        if self.is_full():
            return False
        self.data[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        return True
    
    def dequeue(self) -> bool:
        if self.is_empty():
            return False
        self.front = (self.front + 1) % self.capacity
        return True
    
    def front(self) -> int:
        return -1 if self.is_empty() else self.data[self.front]
    
    def rear(self) -> int:
        if self.is_empty():
            return -1
        return self.data[(self.rear - 1 + self.capacity) % self.capacity]
    
    def is_empty(self) -> bool:
        return self.front == self.rear
    
    def is_full(self) -> bool:
        return (self.rear + 1) % self.capacity == self.front


# Python 双端队列
def deque_demo():
    dq = deque()
    dq.append(1)       # 右端添加
    dq.appendleft(0)   # 左端添加
    dq.pop()           # 右端删除
    dq.popleft()       # 左端删除
    
    # 支持索引访问
    print(dq[0])
```
:::

### 双端队列 (Deque)

双端队列允许在两端进行插入和删除操作，比普通队列更灵活。

| 操作 | 普通队列 | 双端队列 |
|------|----------|----------|
| 队头插入 | ❌ | ✅ |
| 队头删除 | ✅ | ✅ |
| 队尾插入 | ✅ | ✅ |
| 队尾删除 | ❌ | ✅ |

### 单调队列 ⭐

**定义**：单调队列是一种特殊的双端队列，队列内元素保持单调性，常用于解决**滑动窗口最值**问题。

**核心思想**：
- 维护一个单调队列，存储可能是窗口最值的元素索引
- 新元素入队时，移除不可能成为最值的元素

::: code-group
```cpp
#include <vector>
#include <deque>
using namespace std;

// 滑动窗口最大值
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> result;
    deque<int> dq;  // 存储索引，保持递减
    
    for (int i = 0; i < nums.size(); i++) {
        // 移除超出窗口的元素
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        // 维护单调递减：移除比当前元素小的元素
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
        
        // 窗口形成后，队头就是最大值
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    
    return result;
}
// 示例: [1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]

// 滑动窗口最小值
vector<int> minSlidingWindow(vector<int>& nums, int k) {
    vector<int> result;
    deque<int> dq;  // 存储索引，保持递增
    
    for (int i = 0; i < nums.size(); i++) {
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        while (!dq.empty() && nums[dq.back()] > nums[i]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
        
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    
    return result;
}
```

```python
from collections import deque
from typing import List

def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """滑动窗口最大值"""
    result = []
    dq = deque()  # 存储索引，保持递减
    
    for i, num in enumerate(nums):
        # 移除超出窗口的元素
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # 维护单调递减
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # 窗口形成后记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
# 示例: [1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]


def min_sliding_window(nums: List[int], k: int) -> List[int]:
    """滑动窗口最小值"""
    result = []
    dq = deque()  # 存储索引，保持递增
    
    for i, num in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        while dq and nums[dq[-1]] > num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# 测试
if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print("滑动窗口最大值:", max_sliding_window(nums, k))
    print("滑动窗口最小值:", min_sliding_window(nums, k))
```
:::

### 典型 LeetCode 题型

| 题目 | 难度 | 核心技巧 |
|------|------|----------|
| [用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/) | 简单 | 双队列模拟 |
| [设计循环队列](https://leetcode.cn/problems/design-circular-queue/) | 中等 | 循环数组 |
| [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | 困难 | 单调队列 |
| [和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/) | 中等 | 前缀和 + 哈希 |

---

## 线性结构对比总结

### 数据结构特性对比

| 特性 | 数组 | 链表 | 栈 | 队列 |
|------|------|------|------|------|
| 访问原则 | 随机访问 | 顺序访问 | LIFO | FIFO |
| 随机访问 | ✅ O(1) | ❌ O(n) | ❌ 仅栈顶 | ❌ 仅队头队尾 |
| 头部插入 | ❌ O(n) | ✅ O(1) | ❌ 不支持 | ✅ O(1)* |
| 尾部插入 | ✅ 均摊O(1) | ✅ O(1) | ✅ O(1) | ✅ O(1) |
| 中间插入 | ❌ O(n) | ✅ O(1)* | ❌ 不支持 | ❌ 不支持 |
| 内存布局 | 连续 | 离散 | 连续/离散 | 连续/离散 |
| 缓存友好 | ✅ 优秀 | ❌ 较差 | 视实现 | 视实现 |
| 内存开销 | 小 | 大(指针) | 小 | 小 |

*注：链表中间插入需要 O(n) 查找位置；队列头部插入需要双端队列

### 时间复杂度总览

| 操作 | 数组 | 链表 | 栈 | 队列 |
|------|------|------|------|------|
| 访问元素 | O(1) | O(n) | O(1)* | O(1)* |
| 查找元素 | O(n) | O(n) | O(n) | O(n) |
| 插入元素 | O(n) | O(1) | O(1) | O(1) |
| 删除元素 | O(n) | O(1) | O(1) | O(1) |

*注：栈和队列只能访问特定位置的元素

### 选择指南

| 需求场景 | 推荐结构 | 理由 |
|----------|----------|------|
| 频繁随机访问 | 数组 | O(1) 访问效率 |
| 频繁头部插入删除 | 链表 | O(1) 操作效率 |
| 需要回溯操作 | 栈 | LIFO 特性天然支持 |
| 任务调度/排队 | 队列 | FIFO 公平处理 |
| 滑动窗口问题 | 单调队列 | 高效维护窗口最值 |
| 下一个更大/更小元素 | 单调栈 | 线性时间复杂度 |
| 内存受限 | 数组 | 无指针开销 |
| 缓存敏感场景 | 数组 | 连续内存友好 |

### 单调栈 vs 单调队列

| 特性 | 单调栈 | 单调队列 |
|------|--------|----------|
| 数据结构 | 栈 | 双端队列 |
| 适用问题 | 找下一个更大/更小元素 | 滑动窗口最值 |
| 维护方向 | 单向（从后往前） | 双向（滑动窗口） |
| 典型题目 | 每日温度、最大矩形 | 滑动窗口最大值 |

---

## 总结

线性数据结构是算法学习的基石：

1. **数组**：随机访问效率高，适合读多写少的场景
2. **链表**：插入删除效率高，适合写多读少的场景
3. **栈**：LIFO 特性，适合需要"撤销"或"回溯"的场景
4. **队列**：FIFO 特性，适合需要"公平调度"的场景
5. **单调栈**：解决"下一个更大/更小元素"类问题
6. **单调队列**：解决"滑动窗口最值"类问题

💡 **学习建议**：
- 熟练掌握基本操作的时间复杂度
- 理解各种结构的适用场景
- 多练习经典 LeetCode 题目
- 注意边界条件处理（空栈、空队列等）
