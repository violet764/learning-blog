# C++ STL容器详解

## STL概述与设计理念

**STL（Standard Template Library）** 是C++标准库的核心组成部分，提供了一套高效、通用的数据结构和算法。STL基于模板技术实现，遵循泛型编程思想，将算法与数据结构解耦，实现"一次编写，处处使用"的设计目标。

### STL三大组件
1. **容器（Containers）**：存储和管理数据的通用数据结构
2. **算法（Algorithms）**：对容器中数据进行操作的通用算法
3. **迭代器（Iterators）**：连接容器和算法的桥梁

### 设计原则
- **泛型编程**：模板技术实现类型无关性
- **效率优先**：零开销抽象原则
- **接口统一**：一致的访问模式和操作方式
- **异常安全**：提供不同级别的异常安全保证

## 序列容器（Sequence Containers）

### vector：动态数组

**数据结构与内存管理**：vector在堆上维护连续的动态数组，通过三个指针管理内存：起始指针、结束指针和容量指针。采用几何增长策略（通常1.5或2倍）实现动态扩容。

```cpp
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

// vector内部实现原理（简化）
// template<typename T>
// class vector {
// private:
//     T* begin_;      // 数据起始位置
//     T* end_;        // 最后一个元素之后
//     T* capacity_;   // 分配内存的末尾
// public:
//     size_t size() const { return end_ - begin_; }
//     size_t capacity() const { return capacity_ - begin_; }
// };

void vector_detailed_demo() {
    cout << "=== vector动态数组详细演示 ===" << endl;
    
    // 1. 构造与初始化
    vector<int> v1;                     // 空vector
    vector<int> v2(5, 10);              // 5个元素，每个值为10
    vector<int> v3 = {1, 2, 3, 4, 5};   // 初始化列表
    vector<int> v4(v3.begin(), v3.end()); // 迭代器范围
    
    // 2. 容量管理
    cout << "初始容量: " << v3.capacity() << endl;
    v3.reserve(20);  // 预留容量，避免多次重新分配
    cout << "预留后容量: " << v3.capacity() << endl;
    
    // 3. 元素访问
    cout << "第一个元素: " << v3[0] << " (operator[])" << endl;
    cout << "第二个元素: " << v3.at(1) << " (at()带边界检查)" << endl;
    cout << "最后一个元素: " << v3.back() << endl;
    cout << "第一个元素: " << v3.front() << endl;
    
    // 4. 修改操作
    v3.push_back(6);                    // 末尾插入
    v3.insert(v3.begin() + 2, 99);      // 指定位置插入
    v3.pop_back();                      // 删除末尾
    v3.erase(v3.begin() + 1);           // 删除指定位置
    
    // 5. 性能演示
    vector<int> performance_vec;
    performance_vec.reserve(100000);    // 预分配提升性能
    
    for (int i = 0; i < 100000; i++) {
        performance_vec.push_back(i);   // 在预留空间内操作
    }
    
    // 6. 内存管理
    cout << "大小: " << performance_vec.size() << endl;
    cout << "容量: " << performance_vec.capacity() << endl;
    
    performance_vec.shrink_to_fit();    // 缩减容量到实际大小
    cout << "收缩后容量: " << performance_vec.capacity() << endl;
    
    // 7. 算法应用
    sort(v3.begin(), v3.end());         // 排序
    auto it = find(v3.begin(), v3.end(), 99); // 查找
    if (it != v3.end()) {
        cout << "找到元素99" << endl;
    }
}

// vector性能特征
/*
时间复杂度分析：
- 随机访问：O(1)
- 末尾插入/删除：平均O(1)，最坏O(n)（重新分配）
- 中间插入/删除：O(n)
- 查找：O(n)

内存特征：
- 连续内存布局，缓存友好
- 重新分配时复制所有元素
- 容量通常大于实际大小
*/
```

**vector适用场景**：
- 需要频繁随机访问元素
- 主要在末尾进行插入删除操作
- 对内存连续性要求高的场景
- 需要与C接口交互（数据在连续内存中）

### deque：双端队列

**数据结构特性**：deque采用分段连续存储，由多个固定大小的数组块组成，支持两端高效插入删除。

```cpp
#include <deque>
#include <iostream>
using namespace std;

void deque_detailed_demo() {
    cout << "\\n=== deque双端队列详细演示 ===" << endl;
    
    deque<int> dq;
    
    // 1. 两端操作
    dq.push_front(1);   // 前端插入
    dq.push_back(2);    // 后端插入
    dq.push_front(0);   
    dq.push_back(3);
    
    cout << "前端: " << dq.front() << endl;  // 0
    cout << "后端: " << dq.back() << endl;   // 3
    
    // 2. 随机访问
    for (size_t i = 0; i < dq.size(); i++) {
        cout << dq[i] << " ";  // O(1)随机访问
    }
    cout << endl;
    
    // 3. 中间操作（性能较差）
    dq.insert(dq.begin() + 2, 99);  // O(n)操作
    
    // 4. 删除操作
    dq.pop_front();     // 删除前端
    dq.pop_back();      // 删除后端
    
    // 5. 容量特性
    cout << "大小: " << dq.size() << endl;
    
    // deque没有capacity()方法，因为内存是分段的
}

// deque内部结构
/*
deque采用"分段连续"存储：
- 由多个固定大小的数组块组成
- 每个块大小通常为512字节或系统页大小
- 维护一个中央控制数组（map）来管理各个块

性能特征：
- 两端插入删除：O(1)
- 随机访问：O(1)
- 中间插入删除：O(n)
- 内存非完全连续，但近似连续
*/
```

**deque适用场景**：
- 需要在两端频繁插入删除
- 需要随机访问但两端操作比vector频繁
- 作为队列和栈的底层容器

### list：双向链表

**链表结构特性**：list是双向链表实现，每个节点包含数据和前后指针，支持任意位置高效插入删除。

```cpp
#include <list>
#include <iostream>
#include <algorithm>
using namespace std;

void list_detailed_demo() {
    cout << "\\n=== list双向链表详细演示 ===" << endl;
    
    list<int> lst = {1, 2, 3, 4, 5};
    
    // 1. 任意位置插入删除
    auto it = lst.begin();
    advance(it, 2);             // 移动到第三个位置
    lst.insert(it, 99);         // O(1)插入
    
    it = lst.begin();
    advance(it, 3);
    lst.erase(it);              // O(1)删除
    
    // 2. 顺序访问（不支持随机访问）
    cout << "链表内容: ";
    for (auto num : lst) {      // 范围for循环
        cout << num << " ";
    }
    cout << endl;
    
    // 3. list特有操作
    lst.sort();                 // 成员函数排序
    lst.unique();               // 去除连续重复元素
    
    list<int> lst2 = {6, 7, 8};
    lst.splice(lst.end(), lst2); // 拼接操作
    
    lst.remove(99);             // 删除所有值为99的元素
    
    // 4. 大小操作
    cout << "大小: " << lst.size() << endl;
    cout << "是否为空: " << (lst.empty() ? "是" : "否") << endl;
}

// list性能特征
/*
链表节点结构：
struct Node {
    T data;
    Node* prev;
    Node* next;
};

时间复杂度：
- 任意位置插入删除：O(1)
- 查找：O(n)
- 排序：O(n log n)（成员函数）
- 不支持随机访问

内存特征：
- 非连续内存，每个节点单独分配
- 额外存储前后指针（每个节点额外16-32字节）
- 缓存不友好
*/
```

**list适用场景**：
- 需要频繁在任意位置插入删除
- 不需要随机访问
- 内存分配频繁但大小不固定的场景

### forward_list：单向链表

**轻量级链表**：forward_list是单向链表，只保存指向下一个节点的指针，比list更节省内存。

```cpp
#include <forward_list>
#include <iostream>
using namespace std;

void forward_list_demo() {
    cout << "\\n=== forward_list单向链表演示 ===" << endl;
    
    forward_list<int> flist = {1, 2, 3, 4, 5};
    
    // 1. 只能单向遍历
    cout << "链表内容: ";
    for (auto it = flist.begin(); it != flist.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // 2. 插入删除（只能操作当前节点之后）
    auto it = flist.begin();
    flist.insert_after(it, 99);  // 在第一个元素后插入
    
    it = flist.begin();
    advance(it, 2);
    flist.erase_after(it);       // 删除指定位置后的元素
    
    // 3. 没有size()方法（为了性能）
    cout << "是否为空: " << (flist.empty() ? "是" : "否") << endl;
}
```

**forward_list适用场景**：
- 内存极度敏感的应用
- 只需要单向遍历
- 插入删除主要在链表前端

### array：固定大小数组

**编译时固定大小**：array是C风格数组的封装，大小在编译时确定，提供STL接口和边界检查。

```cpp
#include <array>
#include <iostream>
using namespace std;

void array_demo() {
    cout << "\\n=== array固定大小数组演示 ===" << endl;
    
    array<int, 5> arr = {1, 2, 3, 4, 5};
    
    // 1. 固定大小，编译时确定
    cout << "大小: " << arr.size() << endl;  // 5（编译时常量）
    
    // 2. 安全访问
    cout << "第一个元素: " << arr[0] << endl;        // 无检查
    cout << "第二个元素: " << arr.at(1) << endl;     // 边界检查
    
    // 3. 迭代器支持
    cout << "数组内容: ";
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // 4. 编译时大小检查
    static_assert(arr.size() == 5, "数组大小必须是5");
}
```

**array适用场景**：
- 大小固定的数组
- 需要STL接口但不要动态大小的场景
- 栈上分配的小型数组

## 关联容器（Associative Containers）

### set/multiset：有序集合

**红黑树实现**：set基于红黑树实现，元素自动排序且唯一（multiset允许重复）。

```cpp
#include <set>
#include <iostream>
#include <algorithm>
using namespace std;

void set_detailed_demo() {
    cout << "\\n=== set/multiset有序集合详细演示 ===" << endl;
    
    // 1. set自动排序和去重
    set<int> unique_nums = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "set内容（自动排序去重）: ";
    for (int num : unique_nums) {
        cout << num << " ";  // 1 2 3 4 5 6 9
    }
    cout << endl;
    
    // 2. multiset允许重复
    multiset<int> multi_nums = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "multiset内容（允许重复）: ";
    for (int num : multi_nums) {
        cout << num << " ";  // 1 1 2 3 4 5 6 9
    }
    cout << endl;
    
    // 3. 查找操作
    auto found = unique_nums.find(5);
    if (found != unique_nums.end()) {
        cout << "找到元素5" << endl;
    }
    
    // 4. 范围查询
    auto lower = unique_nums.lower_bound(3);  // 第一个>=3的元素
    auto upper = unique_nums.upper_bound(6);  // 第一个>6的元素
    cout << "范围[3,6]的元素: ";
    for (auto it = lower; it != upper; ++it) {
        cout << *it << " ";  // 3 4 5 6
    }
    cout << endl;
    
    // 5. 自定义比较函数
    struct CaseInsensitiveCompare {
        bool operator()(const string& a, const string& b) const {
            return lexicographical_compare(
                a.begin(), a.end(), b.begin(), b.end(),
                [](char c1, char c2) { return tolower(c1) < tolower(c2); }
            );
        }
    };
    
    set<string, CaseInsensitiveCompare> case_insensitive_set;
    case_insensitive_set.insert("Apple");
    case_insensitive_set.insert("apple");  // 不会重复插入
    
    // 6. 集合操作
    set<int> set1 = {1, 2, 3, 4, 5};
    set<int> set2 = {4, 5, 6, 7, 8};
    
    set<int> union_set;
    set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
              inserter(union_set, union_set.begin()));
    
    cout << "并集: ";
    for (int num : union_set) {
        cout << num << " ";  // 1 2 3 4 5 6 7 8
    }
    cout << endl;
}

// set性能特征
/*
红黑树特性：
- 自平衡二叉搜索树
- 树高最多为2*log(n)，保证操作效率
- 插入删除时需要重新平衡

时间复杂度：
- 查找：O(log n)
- 插入：O(log n)
- 删除：O(log n)
- 遍历：O(n)

内存特征：
- 每个节点需要额外存储颜色和指针信息
- 内存非连续，缓存不友好
*/
```

**set适用场景**：
- 需要有序且唯一的元素集合
- 频繁查找操作
- 需要范围查询

### map/multimap：键值对映射

**关联数组**：map存储键值对，按键排序，提供快速查找。

```cpp
#include <map>
#include <iostream>
#include <string>
using namespace std;

void map_detailed_demo() {
    cout << "\\n=== map/multimap键值映射详细演示 ===" << endl;
    
    // 1. map基本操作
    map<string, int> age_map;
    age_map["Alice"] = 25;      // 插入键值对
    age_map["Bob"] = 30;
    age_map["Charlie"] = 35;
    
    // 2. 遍历（按键排序）
    cout << "年龄映射: " << endl;
    for (const auto& pair : age_map) {
        cout << pair.first << ": " << pair.second << endl;
        // 按字典序输出：Alice, Bob, Charlie
    }
    
    // 3. 查找操作
    auto it = age_map.find("Alice");
    if (it != age_map.end()) {
        cout << "Alice的年龄: " << it->second << endl;
    }
    
    // 4. 安全访问
    cout << "Bob的年龄: " << age_map["Bob"] << endl;        // 运算符[]
    // cout << age_map["David"] << endl;  // 插入新键值对（可能不是期望的）
    
    auto david_it = age_map.find("David");
    if (david_it == age_map.end()) {
        cout << "David不在映射中" << endl;
    }
    
    // 5. multimap允许重复键
    multimap<string, int> multi_map;
    multi_map.insert({"Alice", 25});
    multi_map.insert({"Alice", 26});  // 允许重复键
    
    cout << "Alice的所有年龄: ";
    auto range = multi_map.equal_range("Alice");
    for (auto it = range.first; it != range.second; ++it) {
        cout << it->second << " ";  // 25 26
    }
    cout << endl;
    
    // 6. 自定义比较函数
    struct StringLengthCompare {
        bool operator()(const string& a, const string& b) const {
            return a.length() < b.length();  // 按长度排序
        }
    };
    
    map<string, int, StringLengthCompare> length_map;
    length_map["hi"] = 1;
    length_map["hello"] = 2;
    length_map["world"] = 3;
    
    cout << "按长度排序的映射: " << endl;
    for (const auto& pair : length_map) {
        cout << pair.first << " (长度" << pair.first.length() << "): " 
             << pair.second << endl;
    }
}

// map内部结构
/*
map节点结构：
struct Node {
    pair<Key, Value> data;
    Node* left;
    Node* right;
    Node* parent;
    bool color;  // 红黑树颜色
};

性能特征：
- 查找：O(log n)
- 插入：O(log n)
- 删除：O(log n)
- 内存开销较大（每个节点额外信息多）
*/
```

**map适用场景**：
- 键值对映射，需要按键排序
- 频繁按键查找
- 需要有序遍历键

## 无序容器（Unordered Containers）

### unordered_set/unordered_multiset：哈希集合

**哈希表实现**：基于哈希表，提供平均O(1)的查找性能。

```cpp
#include <unordered_set>
#include <iostream>
#include <string>
using namespace std;

// 自定义哈希函数
struct StringHash {
    size_t operator()(const string& s) const {
        // 简单的FNV-1a哈希算法
        size_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<size_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

// 自定义相等比较
struct StringEqual {
    bool operator()(const string& a, const string& b) const {
        return a.length() == b.length() && 
               equal(a.begin(), a.end(), b.begin(),
                     [](char c1, char c2) { return tolower(c1) == tolower(c2); });
    }
};

void unordered_set_demo() {
    cout << "\\n=== unordered_set哈希集合详细演示 ===" << endl;
    
    // 1. 基本使用
    unordered_set<int> hash_set = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "哈希集合（无序）: ";
    for (int num : hash_set) {
        cout << num << " ";  // 无序输出
    }
    cout << endl;
    
    // 2. 性能优化
    unordered_set<int> optimized_set;
    optimized_set.reserve(100);         // 预分配桶
    optimized_set.max_load_factor(0.7f); // 设置最大负载因子
    
    for (int i = 0; i < 100; i++) {
        optimized_set.insert(i);
    }
    
    // 3. 哈希表参数
    cout << "桶数量: " << optimized_set.bucket_count() << endl;
    cout << "负载因子: " << optimized_set.load_factor() << endl;
    cout << "最大负载因子: " << optimized_set.max_load_factor() << endl;
    
    // 4. 自定义哈希函数
    unordered_set<string, StringHash, StringEqual> custom_set;
    custom_set.insert("Apple");
    custom_set.insert("apple");  // 不区分大小写，不会重复
    
    cout << "自定义哈希集合大小: " << custom_set.size() << endl;  // 1
}

// 哈希表性能特征
/*
哈希冲突处理：
- 链表法：同一桶内的元素用链表连接
- 开放定址法：线性探测、二次探测等

时间复杂度：
- 平均情况：O(1)
- 最坏情况：O(n)（所有元素哈希到同一桶）

负载因子控制：
- 负载因子 = 元素数量 / 桶数量
- 负载因子过高时重新哈希（重新分配桶）
*/
```

### unordered_map/unordered_multimap：哈希映射

**哈希表键值对**：基于哈希表的键值对存储，提供快速查找。

```cpp
#include <unordered_map>
#include <iostream>
#include <string>
using namespace std;

void unordered_map_demo() {
    cout << "\\n=== unordered_map哈希映射详细演示 ===" << endl;
    
    unordered_map<string, int> word_count;
    
    // 1. 插入和计数
    word_count["hello"]++;      // 插入新键或增加值
    word_count["world"]++;
    word_count["hello"]++;      // 增加值
    
    // 2. 遍历（无序）
    cout << "单词统计: " << endl;
    for (const auto& pair : word_count) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // 3. 查找和更新
    auto it = word_count.find("hello");
    if (it != word_count.end()) {
        it->second = 5;  // 更新值
    }
    
    // 4. 桶信息
    cout << "桶数量: " << word_count.bucket_count() << endl;
    cout << "负载因子: " << word_count.load_factor() << endl;
    
    // 5. 性能优化
    unordered_map<string, int> optimized_map;
    optimized_map.reserve(1000);
    optimized_map.max_load_factor(0.75f);
}
```

**无序容器适用场景**：
- 需要快速查找，不关心顺序
- 键的类型有良好的哈希函数
- 不需要有序遍历

## 容器适配器（Container Adapters）

### stack：栈

**后进先出**：基于deque、vector或list实现的栈适配器。

```cpp
#include <stack>
#include <iostream>
using namespace std;

void stack_demo() {
    cout << "\\n=== stack栈适配器演示 ===" << endl;
    
    stack<int> stk;
    
    // 压栈操作
    stk.push(1);
    stk.push(2);
    stk.push(3);
    
    // 访问栈顶
    cout << "栈顶元素: " << stk.top() << endl;  // 3
    
    // 弹出栈顶
    stk.pop();
    cout << "弹出后栈顶: " << stk.top() << endl;  // 2
    
    // 栈大小
    cout << "栈大小: " << stk.size() << endl;
}
```

### queue：队列

**先进先出**：基于deque或list实现的队列适配器。

```cpp
#include <queue>
#include <iostream>
using namespace std;

void queue_demo() {
    cout << "\\n=== queue队列适配器演示 ===" << endl;
    
    queue<int> q;
    
    // 入队
    q.push(1);
    q.push(2);
    q.push(3);
    
    // 访问队首队尾
    cout << "队首: " << q.front() << endl;  // 1
    cout << "队尾: " << q.back() << endl;   // 3
    
    // 出队
    q.pop();
    cout << "出队后队首: " << q.front() << endl;  // 2
}
```

### priority_queue：优先队列

**堆实现**：基于vector实现的二叉堆，提供优先级队列功能。

```cpp
#include <queue>
#include <iostream>
#include <vector>
#include <functional>
using namespace std;

void priority_queue_demo() {
    cout << "\\n=== priority_queue优先队列演示 ===" << endl;
    
    // 最大堆（默认）
    priority_queue<int> max_heap;
    max_heap.push(3);
    max_heap.push(1);
    max_heap.push(4);
    max_heap.push(1);
    max_heap.push(5);
    
    cout << "最大堆（降序输出）: ";
    while (!max_heap.empty()) {
        cout << max_heap.top() << " ";  // 5 4 3 1 1
        max_heap.pop();
    }
    cout << endl;
    
    // 最小堆
    priority_queue<int, vector<int>, greater<int>> min_heap;
    min_heap.push(3);
    min_heap.push(1);
    min_heap.push(4);
    min_heap.push(1);
    min_heap.push(5);
    
    cout << "最小堆（升序输出）: ";
    while (!min_heap.empty()) {
        cout << min_heap.top() << " ";  // 1 1 3 4 5
        min_heap.pop();
    }
    cout << endl;
}
```

## 容器选择指南

### 性能比较表

| 容器类型 | 随机访问 | 查找 | 插入删除 | 内存 | 适用场景 |
|----------|----------|------|-----------|------|----------|
| **vector** | O(1) | O(n) | 末尾O(1)，中间O(n) | 连续 | 频繁随机访问 |
| **deque** | O(1) | O(n) | 两端O(1)，中间O(n) | 分段连续 | 两端操作频繁 |
| **list** | O(n) | O(n) | O(1) | 非连续 | 频繁任意位置操作 |
| **set/map** | - | O(log n) | O(log n) | 非连续 | 需要有序集合/映射 |
| **unordered_set/map** | - | O(1)平均 | O(1)平均 | 非连续 | 快速查找，不关心顺序 |

### 选择原则

1. **需要随机访问**：vector、deque、array
2. **频繁插入删除**：list、forward_list
3. **需要有序**：set、map
4. **快速查找**：unordered_set、unordered_map
5. **内存敏感**：array、forward_list
6. **需要特定数据结构**：stack、queue、priority_queue

### 最佳实践

```cpp
// 1. 预分配空间提升性能
vector<int> vec;
vec.reserve(1000);  // 避免多次重新分配

// 2. 使用emplace避免不必要的拷贝
vector<pair<string, int>> data;
data.emplace_back("Alice", 25);  // 原地构造

// 3. 利用移动语义
vector<string> move_demo;
string large_string = "很大的字符串";
move_demo.push_back(std::move(large_string));  // 移动而非拷贝

// 4. 选择合适的迭代器
vector<int> numbers = {1, 2, 3, 4, 5};
for (auto it = numbers.cbegin(); it != numbers.cend(); ++it) {
    // 只读访问使用const迭代器
}

// 5. 利用算法提升效率
sort(numbers.begin(), numbers.end());
auto found = lower_bound(numbers.begin(), numbers.end(), 3);
```

## 总结

STL容器提供了丰富的数据结构选择，每种容器都有其特定的性能特征和适用场景。理解各容器的内部实现原理和性能特征，能够帮助我们在实际开发中做出合适的选择，编写出高效、可维护的C++代码。

关键要点：
- **vector**：连续内存，随机访问高效
- **list**：任意位置插入删除高效
- **set/map**：有序集合，查找高效
- **unordered_set/map**：哈希表，平均O(1)查找
- **容器适配器**：提供特定数据结构接口

通过合理选择和使用STL容器，可以显著提升C++程序的性能和可维护性。