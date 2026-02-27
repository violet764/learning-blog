# C++ STL 容器详解

STL（Standard Template Library）是 C++ 标准库的核心组成部分，提供了一套高效、通用的数据结构和算法。STL 基于模板技术实现，遵循泛型编程思想，将算法与数据结构解耦。

## STL 核心组件

| 组件 | 说明 |
|------|------|
| **容器（Containers）** | 存储和管理数据的通用数据结构 |
| **迭代器（Iterators）** | 连接容器和算法的桥梁，提供统一的访问接口 |
| **算法（Algorithms）** | 对容器中数据进行操作的通用算法 |
| **函数对象（Functors）** | 重载 `operator()` 的类，可像函数一样调用 |
| **适配器（Adapters）** | 修改容器或函数对象接口的组件 |
| **分配器（Allocators）** | 自定义内存管理的组件 |

## 序列容器

序列容器按线性顺序存储元素，提供对元素的顺序访问。

### vector：动态数组

`vector` 在堆上维护连续的动态数组，采用几何增长策略（通常 1.5 或 2 倍）实现动态扩容。

```cpp
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    // 构造方式
    std::vector<int> v1;                     // 空 vector
    std::vector<int> v2(5, 10);              // 5个元素，每个值为10
    std::vector<int> v3 = {1, 2, 3, 4, 5};   // 初始化列表
    
    // 容量管理
    v3.reserve(20);          // 预留容量，避免多次重新分配
    std::cout << "容量: " << v3.capacity() << std::endl;
    v3.shrink_to_fit();      // 缩减容量到实际大小
    
    // 元素访问
    std::cout << v3[0] << std::endl;         // 无边界检查
    std::cout << v3.at(1) << std::endl;      // 带边界检查
    std::cout << v3.front() << std::endl;    // 首元素
    std::cout << v3.back() << std::endl;     // 尾元素
    
    // 修改操作
    v3.push_back(6);         // 末尾插入
    v3.pop_back();           // 删除末尾
    v3.insert(v3.begin() + 2, 99);  // 指定位置插入
    v3.erase(v3.begin() + 1);       // 删除指定位置
    
    // 性能优化：预分配 + emplace
    std::vector<std::pair<std::string, int>> data;
    data.reserve(100);
    data.emplace_back("Alice", 25);  // 原地构造，避免临时对象
    
    return 0;
}
```

**性能特征**：
- 随机访问：O(1)
- 末尾插入/删除：平均 O(1)，扩容时 O(n)
- 中间插入/删除：O(n)
- 内存连续，缓存友好

### deque：双端队列

`deque` 采用分段连续存储，由多个固定大小的数组块组成，支持两端高效操作。

```cpp
#include <deque>
#include <iostream>

int main() {
    std::deque<int> dq;
    
    // 两端操作（都是 O(1)）
    dq.push_front(1);
    dq.push_back(2);
    dq.push_front(0);
    
    std::cout << "前端: " << dq.front() << std::endl;  // 0
    std::cout << "后端: " << dq.back() << std::endl;   // 2
    
    // 随机访问
    for (size_t i = 0; i < dq.size(); i++) {
        std::cout << dq[i] << " ";
    }
    
    dq.pop_front();
    dq.pop_back();
    
    return 0;
}
```

**性能特征**：
- 两端插入/删除：O(1)
- 随机访问：O(1)（比 vector 略慢）
- 中间插入/删除：O(n)

### list 与 forward_list：链表

```cpp
#include <list>
#include <forward_list>
#include <iostream>

int main() {
    // list：双向链表
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    auto it = lst.begin();
    std::advance(it, 2);       // 移动到第三个位置
    lst.insert(it, 99);        // O(1) 插入
    lst.erase(it);             // O(1) 删除
    
    // list 特有操作
    lst.sort();                // 成员函数排序
    lst.unique();              // 去除连续重复元素
    lst.remove(99);            // 删除所有值为99的元素
    
    // forward_list：单向链表，更节省内存
    std::forward_list<int> flist = {1, 2, 3, 4, 5};
    flist.insert_after(flist.begin(), 99);  // 在第一个元素后插入
    flist.erase_after(flist.begin());       // 删除第一个元素后的元素
    
    return 0;
}
```

**性能特征**：
| 容器 | 插入/删除 | 随机访问 | 内存开销 |
|------|-----------|----------|----------|
| list | O(1) 任意位置 | O(n) 不支持 | 每节点 2 指针 |
| forward_list | O(1) 指定位置后 | O(n) 不支持 | 每节点 1 指针 |

### array：固定大小数组

```cpp
#include <array>
#include <iostream>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    
    std::cout << "大小: " << arr.size() << std::endl;  // 编译时常量
    std::cout << arr.at(1) << std::endl;               // 带边界检查
    
    // 编译时检查
    static_assert(arr.size() == 5, "数组大小必须是5");
    
    // 支持 STL 算法
    std::sort(arr.begin(), arr.end());
    
    return 0;
}
```

## 关联容器

关联容器基于红黑树实现，元素按键自动排序，提供 O(log n) 的查找性能。

### set 与 multiset：有序集合

```cpp
#include <set>
#include <iostream>
#include <algorithm>

int main() {
    // set：元素唯一且有序
    std::set<int> s = {3, 1, 4, 1, 5, 9, 2, 6};  // 自动去重排序
    
    // 插入与查找
    s.insert(10);
    auto it = s.find(5);          // O(log n)
    
    // 范围查询
    auto lower = s.lower_bound(3);  // 第一个 >= 3 的元素
    auto upper = s.upper_bound(6);  // 第一个 > 6 的元素
    
    std::cout << "范围 [3,6]: ";
    for (auto iter = lower; iter != upper; ++iter) {
        std::cout << *iter << " ";  // 3 4 5 6
    }
    std::cout << std::endl;
    
    // multiset：允许重复元素
    std::multiset<int> ms = {1, 1, 2, 2, 2, 3};
    auto range = ms.equal_range(2);  // 所有等于2的元素范围
    
    return 0;
}
```

### map 与 multimap：键值映射

```cpp
#include <map>
#include <iostream>
#include <string>

int main() {
    std::map<std::string, int> age;
    
    // 插入
    age["Alice"] = 25;
    age.insert({"Bob", 30});
    age.emplace("Charlie", 35);
    
    // 访问（按键排序）
    for (const auto& [name, a] : age) {  // C++17 结构化绑定
        std::cout << name << ": " << a << std::endl;
    }
    
    // 查找
    if (auto it = age.find("Alice"); it != age.end()) {
        std::cout << "Alice 的年龄: " << it->second << std::endl;
    }
    
    // 安全访问：at() 会抛出异常，[] 会插入默认值
    try {
        std::cout << age.at("David") << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "David 不存在" << std::endl;
    }
    
    return 0;
}
```

**性能特征**：
- 查找/插入/删除：O(log n)
- 元素自动按键排序
- 基于红黑树实现

## 无序容器

无序容器基于哈希表实现，提供平均 O(1) 的查找性能。

### unordered_set 与 unordered_map

```cpp
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <string>

int main() {
    // unordered_set
    std::unordered_set<int> us = {3, 1, 4, 1, 5, 9};
    
    // 性能优化
    us.reserve(100);               // 预分配桶
    us.max_load_factor(0.7f);      // 设置最大负载因子
    
    std::cout << "桶数量: " << us.bucket_count() << std::endl;
    std::cout << "负载因子: " << us.load_factor() << std::endl;
    
    // unordered_map
    std::unordered_map<std::string, int> word_count;
    word_count["hello"]++;
    word_count["world"]++;
    
    for (const auto& [word, count] : word_count) {
        std::cout << word << ": " << count << std::endl;
    }
    
    return 0;
}
```

### 自定义哈希函数

```cpp
#include <unordered_map>
#include <string>

// 自定义哈希函数
struct StringHash {
    size_t operator()(const std::string& s) const {
        size_t hash = 0;
        for (char c : s) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};

// 自定义相等比较
struct StringEqual {
    bool operator()(const std::string& a, const std::string& b) const {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(a[i]) != std::tolower(b[i])) return false;
        }
        return true;
    }
};

// 使用自定义哈希和比较
std::unordered_map<std::string, int, StringHash, StringEqual> custom_map;
```

**性能特征**：
- 平均查找/插入/删除：O(1)
- 最坏情况：O(n)（哈希冲突严重时）
- 元素无序

## 容器适配器

容器适配器基于底层容器提供特定数据结构的接口。

### stack：栈（LIFO）

```cpp
#include <stack>
#include <vector>
#include <iostream>

int main() {
    // 默认基于 deque，可指定底层容器
    std::stack<int> s1;
    std::stack<int, std::vector<int>> s2;  // 基于 vector
    
    s1.push(1);
    s1.push(2);
    s1.push(3);
    
    std::cout << "栈顶: " << s1.top() << std::endl;  // 3
    s1.pop();
    std::cout << "栈大小: " << s1.size() << std::endl;
    
    return 0;
}
```

### queue：队列（FIFO）

```cpp
#include <queue>
#include <iostream>

int main() {
    std::queue<int> q;
    
    q.push(1);
    q.push(2);
    q.push(3);
    
    std::cout << "队首: " << q.front() << std::endl;  // 1
    std::cout << "队尾: " << q.back() << std::endl;   // 3
    q.pop();
    
    return 0;
}
```

### priority_queue：优先队列

```cpp
#include <queue>
#include <vector>
#include <functional>
#include <iostream>

int main() {
    // 最大堆（默认）
    std::priority_queue<int> max_heap;
    max_heap.push(3);
    max_heap.push(1);
    max_heap.push(4);
    
    std::cout << "最大堆栈顶: " << max_heap.top() << std::endl;  // 4
    
    // 最小堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
    min_heap.push(3);
    min_heap.push(1);
    min_heap.push(4);
    
    std::cout << "最小堆栈顶: " << min_heap.top() << std::endl;  // 1
    
    // 自定义比较
    auto cmp = [](int a, int b) { return a > b; };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> custom_heap(cmp);
    
    return 0;
}
```

## 迭代器

迭代器是连接容器和算法的桥梁，提供统一的元素访问接口。

### 迭代器类别

| 类别 | 支持操作 | 典型容器 |
|------|----------|----------|
| **输入迭代器** | 只读、单向、单遍 | `istream_iterator` |
| **输出迭代器** | 只写、单向、单遍 | `ostream_iterator` |
| **前向迭代器** | 读写、单向、多遍 | `forward_list` |
| **双向迭代器** | 读写、双向 | `list`, `set`, `map` |
| **随机访问迭代器** | 读写、随机访问 | `vector`, `deque`, `array` |

```cpp
#include <vector>
#include <list>
#include <iterator>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // 正向迭代器
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    
    // 反向迭代器
    for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
        std::cout << *rit << " ";
    }
    
    // 常量迭代器
    for (auto cit = vec.cbegin(); cit != vec.cend(); ++cit) {
        // *cit = 10;  // 错误：不能修改
    }
    
    // 迭代器适配器
    std::copy(vec.begin(), vec.end(),
              std::ostream_iterator<int>(std::cout, " "));
    
    return 0;
}
```

### 迭代器失效

迭代器失效是使用 STL 容器时需要特别注意的问题：

```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // 错误示例：插入时迭代器失效
    // auto it = vec.begin() + 2;
    // vec.push_back(6);  // 可能触发重新分配
    // *it;  // 危险！it 可能已失效
    
    // 正确做法：使用返回值或索引
    auto it = vec.insert(vec.begin() + 2, 99);  // 返回有效迭代器
    
    // 删除时的正确做法
    for (auto it = vec.begin(); it != vec.end(); ) {
        if (*it % 2 == 0) {
            it = vec.erase(it);  // erase 返回下一个有效迭代器
        } else {
            ++it;
        }
    }
    
    return 0;
}
```

**迭代器失效规则**：

| 操作 | vector/deque | list/forward_list | 关联容器 |
|------|--------------|-------------------|----------|
| 插入 | 可能全部失效 | 不失效 | 不失效 |
| 删除 | 被删元素及之后失效 | 只是被删元素失效 | 只是被删元素失效 |

### 自定义迭代器

可以为自定义容器实现迭代器，使其兼容 STL 算法：

```cpp
#include <iterator>
#include <algorithm>
#include <iostream>

// 自定义范围迭代器
class RangeIterator {
private:
    int current;
    int end;
    int step;
    
public:
    // 迭代器特性（必需）
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;
    
    RangeIterator(int start, int end_val, int step_val = 1)
        : current(start), end(end_val), step(step_val) {}
    
    // 前置递增
    RangeIterator& operator++() {
        current += step;
        return *this;
    }
    
    // 后置递增
    RangeIterator operator++(int) {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }
    
    // 解引用
    int operator*() const { return current; }
    
    // 相等比较
    bool operator==(const RangeIterator& other) const {
        return current == other.current;
    }
    
    bool operator!=(const RangeIterator& other) const {
        return !(*this == other);
    }
};

// 范围类（提供 begin/end）
class Range {
private:
    int start, end, step;
    
public:
    Range(int s, int e, int st = 1) : start(s), end(e), step(st) {}
    
    RangeIterator begin() const { return RangeIterator(start, end, step); }
    RangeIterator end() const { return RangeIterator(end, end, step); }
};

int main() {
    // 使用自定义迭代器
    Range r(1, 10, 2);
    for (int num : r) {
        std::cout << num << " ";  // 输出: 1 3 5 7 9
    }
    
    // 与 STL 算法兼容
    std::vector<int> result;
    std::copy(r.begin(), r.end(), std::back_inserter(result));
    
    return 0;
}
```

## 函数对象

函数对象（Functor）是重载了 `operator()` 的类，可以像函数一样调用。

### 基本用法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

// 自定义函数对象
struct Multiplier {
    int factor;
    Multiplier(int f) : factor(f) {}
    int operator()(int x) const { return x * factor; }
};

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    
    // 使用函数对象
    Multiplier triple(3);
    std::transform(nums.begin(), nums.end(), nums.begin(), triple);
    
    // 使用 lambda 表达式
    std::transform(nums.begin(), nums.end(), nums.begin(),
                   [](int x) { return x * 2; });
    
    // 标准库函数对象
    std::sort(nums.begin(), nums.end(), std::greater<int>());  // 降序
    
    return 0;
}
```

### 标准库函数对象

```cpp
#include <functional>
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5};
    
    // 算术运算
    std::plus<int> add;
    int sum = add(3, 4);  // 7
    
    // 比较运算
    std::greater<int> gt;
    bool result = gt(5, 3);  // true
    
    // 逻辑运算
    std::logical_and<bool> land;
    bool both = land(true, false);  // false
    
    // 绑定器
    auto multiply_by_2 = std::bind(std::multiplies<int>(), 
                                   std::placeholders::_1, 2);
    int result2 = multiply_by_2(5);  // 10
    
    return 0;
}
```

## 分配器

分配器是 STL 的内存管理组件，用于自定义容器的内存分配策略。

```cpp
#include <vector>
#include <memory>
#include <iostream>

// 自定义分配器示例（简化版）
template<typename T>
class TrackingAllocator {
public:
    using value_type = T;
    
    TrackingAllocator() = default;
    
    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>&) {}
    
    T* allocate(size_t n) {
        std::cout << "分配 " << n << " 个 " << sizeof(T) 
                  << " 字节的对象" << std::endl;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    
    void deallocate(T* p, size_t n) {
        std::cout << "释放 " << n << " 个对象" << std::endl;
        ::operator delete(p);
    }
    
    template<typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
};

int main() {
    // 使用自定义分配器
    std::vector<int, TrackingAllocator<int>> v;
    v.reserve(10);
    v.push_back(1);
    v.push_back(2);
    
    // 使用 PMR 分配器（C++17）
    // std::pmr::vector<int> pmr_vec;
    
    return 0;
}
```

## 容器选择指南

### 性能对比

| 容器 | 随机访问 | 查找 | 插入/删除 | 内存布局 |
|------|----------|------|-----------|----------|
| **vector** | O(1) | O(n) | 末尾 O(1)，中间 O(n) | 连续 |
| **deque** | O(1) | O(n) | 两端 O(1)，中间 O(n) | 分段连续 |
| **list** | - | O(n) | O(1) | 非连续 |
| **set/map** | - | O(log n) | O(log n) | 非连续 |
| **unordered_set/map** | - | O(1) 平均 | O(1) 平均 | 非连续 |

### 选择原则

```cpp
// 1. 需要随机访问 → vector, deque, array
std::vector<int> random_access;

// 2. 频繁两端操作 → deque
std::deque<int> both_ends;

// 3. 频繁任意位置插入删除 → list
std::list<int> frequent_insert;

// 4. 需要有序集合 → set, map
std::set<int> ordered_set;
std::map<std::string, int> ordered_map;

// 5. 快速查找，不关心顺序 → unordered_set, unordered_map
std::unordered_set<int> fast_lookup;
std::unordered_map<std::string, int> fast_map;

// 6. 内存敏感 → array, forward_list
std::array<int, 100> fixed_size;
std::forward_list<int> memory_efficient;

// 7. 特定数据结构 → 适配器
std::stack<int> lifo;
std::queue<int> fifo;
std::priority_queue<int> heap;
```

## 最佳实践

```cpp
#include <vector>
#include <string>
#include <algorithm>

void best_practices() {
    // 1. 预分配空间
    std::vector<int> v;
    v.reserve(1000);  // 避免多次重新分配
    
    // 2. 使用 emplace 原地构造
    std::vector<std::pair<std::string, int>> data;
    data.emplace_back("Alice", 25);  // 避免临时对象
    
    // 3. 利用移动语义
    std::string large = "很长的字符串";
    data.emplace_back(std::move(large), 30);  // 移动而非拷贝
    
    // 4. 使用 const 迭代器
    for (auto it = v.cbegin(); it != v.cend(); ++it) {
        // 只读访问
    }
    
    // 5. erase-remove 惯用法
    v.erase(std::remove_if(v.begin(), v.end(),
                           [](int x) { return x < 0; }),
            v.end());
    
    // 6. C++20: 使用 ranges（如果可用）
    // std::erase_if(v, [](int x) { return x < 0; });
}
```

## 参考资料

- [cppreference - Containers library](https://en.cppreference.com/w/cpp/container)
- 《Effective STL》- Scott Meyers
- 《C++ Primer》第9-11章