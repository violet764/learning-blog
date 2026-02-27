# C++ STL 标准模板库

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

---

## 容器

### 序列容器

序列容器按线性顺序存储元素，提供对元素的顺序访问。

#### vector：动态数组

```cpp
#include <vector>

int main() {
    // 构造方式
    std::vector<int> v1;                     // 空 vector
    std::vector<int> v2(5, 10);              // 5个元素，每个值为10
    std::vector<int> v3 = {1, 2, 3, 4, 5};   // 初始化列表
    
    // 容量管理
    v3.reserve(20);          // 预留容量
    v3.shrink_to_fit();      // 缩减容量到实际大小
    
    // 元素访问
    v3[0];         // 无边界检查
    v3.at(1);      // 带边界检查
    v3.front();    // 首元素
    v3.back();     // 尾元素
    
    // 修改操作
    v3.push_back(6);         // 末尾插入
    v3.pop_back();           // 删除末尾
    v3.insert(v3.begin() + 2, 99);  // 指定位置插入
    v3.erase(v3.begin() + 1);       // 删除指定位置
    
    // 性能优化：预分配 + emplace
    std::vector<std::pair<std::string, int>> data;
    data.reserve(100);
    data.emplace_back("Alice", 25);  // 原地构造
    
    return 0;
}
```

**性能特征**：随机访问 O(1)，末尾插入/删除平均 O(1)，中间插入/删除 O(n)

#### deque：双端队列

```cpp
#include <deque>

std::deque<int> dq;

// 两端操作（都是 O(1)）
dq.push_front(1);
dq.push_back(2);
dq.pop_front();
dq.pop_back();
```

#### list 与 forward_list：链表

```cpp
#include <list>
#include <forward_list>

// list：双向链表
std::list<int> lst = {1, 2, 3, 4, 5};
lst.insert(it, 99);   // O(1) 插入
lst.erase(it);        // O(1) 删除
lst.sort();           // 成员函数排序
lst.unique();         // 去除连续重复元素

// forward_list：单向链表，更节省内存
std::forward_list<int> flist = {1, 2, 3, 4, 5};
flist.insert_after(flist.begin(), 99);
```

#### array：固定大小数组

```cpp
#include <array>

std::array<int, 5> arr = {1, 2, 3, 4, 5};
arr.size();           // 编译时常量
arr.at(1);            // 带边界检查
```

### 关联容器

关联容器基于红黑树实现，元素按键自动排序，提供 O(log n) 的查找性能。

#### set 与 multiset：有序集合

```cpp
#include <set>

// set：元素唯一且有序
std::set<int> s = {3, 1, 4, 1, 5, 9, 2, 6};  // 自动去重排序

s.insert(10);
auto it = s.find(5);          // O(log n)

// 范围查询
auto lower = s.lower_bound(3);  // 第一个 >= 3 的元素
auto upper = s.upper_bound(6);  // 第一个 > 6 的元素

// multiset：允许重复元素
std::multiset<int> ms = {1, 1, 2, 2, 2, 3};
```

#### map 与 multimap：键值映射

```cpp
#include <map>

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
```

### 无序容器

无序容器基于哈希表实现，提供平均 O(1) 的查找性能。

```cpp
#include <unordered_set>
#include <unordered_map>

std::unordered_set<int> us = {3, 1, 4, 1, 5, 9};
us.reserve(100);               // 预分配桶
us.max_load_factor(0.7f);      // 设置最大负载因子

std::unordered_map<std::string, int> word_count;
word_count["hello"]++;
```

### 容器适配器

```cpp
#include <stack>
#include <queue>

// stack：栈（LIFO）
std::stack<int> s;
s.push(1);
s.top();    // 栈顶
s.pop();

// queue：队列（FIFO）
std::queue<int> q;
q.push(1);
q.front();  // 队首
q.back();   // 队尾
q.pop();

// priority_queue：优先队列
std::priority_queue<int> max_heap;  // 最大堆（默认）
std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;  // 最小堆
```

### 容器选择指南

| 容器 | 随机访问 | 查找 | 插入/删除 | 内存布局 |
|------|----------|------|-----------|----------|
| **vector** | O(1) | O(n) | 末尾 O(1)，中间 O(n) | 连续 |
| **deque** | O(1) | O(n) | 两端 O(1)，中间 O(n) | 分段连续 |
| **list** | - | O(n) | O(1) | 非连续 |
| **set/map** | - | O(log n) | O(log n) | 非连续 |
| **unordered_set/map** | - | O(1) 平均 | O(1) 平均 | 非连续 |

---

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
std::vector<int> vec = {1, 2, 3, 4, 5};

// 正向迭代器
for (auto it = vec.begin(); it != vec.end(); ++it) { }

// 反向迭代器
for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) { }

// 常量迭代器
for (auto cit = vec.cbegin(); cit != vec.cend(); ++cit) { }
```

### 迭代器失效规则

| 操作 | vector/deque | list/forward_list | 关联容器 |
|------|--------------|-------------------|----------|
| 插入 | 可能全部失效 | 不失效 | 不失效 |
| 删除 | 被删元素及之后失效 | 只是被删元素失效 | 只是被删元素失效 |

```cpp
// 删除时的正确做法
for (auto it = vec.begin(); it != vec.end(); ) {
    if (*it % 2 == 0) {
        it = vec.erase(it);  // erase 返回下一个有效迭代器
    } else {
        ++it;
    }
}
```

---

## 算法

### 头文件

```cpp
#include <algorithm>   // 大多数算法
#include <numeric>     // 数值算法
#include <execution>   // 并行执行策略（C++17）
```

### 非修改序列操作

```cpp
#include <algorithm>

std::vector<int> nums = {1, 2, 3, 4, 5};

// 查找
auto it = std::find(nums.begin(), nums.end(), 5);
auto even = std::find_if(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });

// 统计
int count = std::count(nums.begin(), nums.end(), 2);
int even_count = std::count_if(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });

// 检查
bool all_pos = std::all_of(nums.begin(), nums.end(), [](int n) { return n > 0; });
bool has_even = std::any_of(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });
bool no_zero = std::none_of(nums.begin(), nums.end(), [](int n) { return n == 0; });
```

### 修改序列操作

```cpp
// 复制
std::copy(src.begin(), src.end(), dst.begin());
std::copy_if(src.begin(), src.end(), std::back_inserter(dst), [](int n) { return n > 0; });

// 变换
std::transform(nums.begin(), nums.end(), squares.begin(), [](int n) { return n * n; });

// 替换
std::replace(nums.begin(), nums.end(), 2, 99);
std::replace_if(nums.begin(), nums.end(), [](int n) { return n < 0; }, 0);

// 填充
std::fill(v.begin(), v.end(), 42);
std::iota(v.begin(), v.end(), 1);  // 1, 2, 3, ...

// 删除（erase-remove 惯用法）
nums.erase(std::remove(nums.begin(), nums.end(), 2), nums.end());
nums.erase(std::remove_if(nums.begin(), nums.end(), [](int n) { return n < 0; }), nums.end());

// 去重
std::sort(nums.begin(), nums.end());
auto last = std::unique(nums.begin(), nums.end());
nums.erase(last, nums.end());

// 反转
std::reverse(nums.begin(), nums.end());
```

### 排序与相关操作

```cpp
// 排序
std::sort(nums.begin(), nums.end());                    // 升序
std::sort(nums.begin(), nums.end(), std::greater<int>()); // 降序
std::stable_sort(nums.begin(), nums.end());             // 稳定排序

// 部分排序
std::partial_sort(nums.begin(), nums.begin() + 3, nums.end());  // 前3个最小

// 第n小元素
std::nth_element(nums.begin(), nums.begin() + n, nums.end());

// 检查排序
bool sorted = std::is_sorted(nums.begin(), nums.end());
```

### 二分查找（要求有序序列）

```cpp
std::vector<int> sorted = {1, 2, 3, 4, 5, 5, 5, 6, 7, 8};

bool found = std::binary_search(sorted.begin(), sorted.end(), 5);

auto lower = std::lower_bound(sorted.begin(), sorted.end(), 5);  // 第一个 >= 5
auto upper = std::upper_bound(sorted.begin(), sorted.end(), 5);  // 第一个 > 5

auto range = std::equal_range(sorted.begin(), sorted.end(), 5);  // 等于5的范围
```

### 分区与排列

```cpp
// 分区
auto partition_point = std::partition(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });
std::stable_partition(nums.begin(), nums.end(), [](int n) { return n % 2 == 0; });

// 排列
std::next_permutation(s.begin(), s.end());  // 下一个排列
std::prev_permutation(s.begin(), s.end());  // 上一个排列
```

### 集合操作

```cpp
std::vector<int> a = {1, 2, 3, 4, 5};
std::vector<int> b = {4, 5, 6, 7, 8};

// 并集
std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// 交集
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// 差集
std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

// 检查子集
bool is_subset = std::includes(a.begin(), a.end(), subset.begin(), subset.end());
```

### 堆操作

```cpp
std::make_heap(nums.begin(), nums.end());   // 构建堆
std::push_heap(nums.begin(), nums.end());   // 插入元素
std::pop_heap(nums.begin(), nums.end());    // 弹出堆顶
std::sort_heap(nums.begin(), nums.end());   // 堆排序
```

### 数值算法

```cpp
#include <numeric>

// 累加
int sum = std::accumulate(nums.begin(), nums.end(), 0);
int product = std::accumulate(nums.begin(), nums.end(), 1, std::multiplies<int>());

// 内积
int dot = std::inner_product(a.begin(), a.end(), b.begin(), 0);

// 部分和
std::partial_sum(nums.begin(), nums.end(), std::back_inserter(partial));

// 相邻差
std::adjacent_difference(nums.begin(), nums.end(), std::back_inserter(diff));
```

### 最小/最大值

```cpp
int m = std::min(3, 5);
int M = std::max(3, 5);
auto [min_val, max_val] = std::minmax(3, 5);  // C++17

auto min_it = std::min_element(nums.begin(), nums.end());
auto max_it = std::max_element(nums.begin(), nums.end());

int clamped = std::clamp(10, 0, 5);  // C++17: 限制范围
```

---

## 函数对象

```cpp
#include <functional>

// 自定义函数对象
struct Multiplier {
    int factor;
    int operator()(int x) const { return x * factor; }
};

// 标准库函数对象
std::plus<int> add;           // 加法
std::greater<int> gt;         // 大于比较
std::logical_and<bool> land;  // 逻辑与

// 绑定器
auto multiply_by_2 = std::bind(std::multiplies<int>(), std::placeholders::_1, 2);
```

---

## 并行算法（C++17）

```cpp
#include <execution>

// 执行策略
// std::execution::seq      - 顺序执行
// std::execution::par      - 并行执行
// std::execution::par_unseq - 并行向量化

std::sort(std::execution::par, nums.begin(), nums.end());
int sum = std::reduce(std::execution::par, nums.begin(), nums.end(), 0);
```

---

## C++20 Ranges 库

```cpp
#include <ranges>

std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 视图组合（惰性求值）
auto result = nums 
    | std::views::filter([](int n) { return n % 2 == 0; })  // 过滤偶数
    | std::views::transform([](int n) { return n * n; });    // 平方

// 常用视图
auto first_five = nums | std::views::take(5);    // 取前5个
auto skip_three = nums | std::views::drop(3);    // 跳过前3个
auto reversed = nums | std::views::reverse;      // 反转

// ranges 算法
std::ranges::sort(nums);
auto found = std::ranges::find(nums, 5);
```

---

## 算法复杂度汇总

| 算法类别 | 典型算法 | 时间复杂度 |
|----------|----------|------------|
| **查找** | `find`, `count` | O(n) |
| **二分查找** | `binary_search`, `lower_bound` | O(log n) |
| **排序** | `sort` | O(n log n) |
| **稳定排序** | `stable_sort` | O(n log² n) |
| **第k小** | `nth_element` | O(n) 平均 |
| **堆操作** | `make_heap`, `push_heap`, `pop_heap` | O(n), O(log n), O(log n) |

---

## 最佳实践

```cpp
// 1. 预分配空间
std::vector<int> v;
v.reserve(1000);

// 2. 使用 emplace 原地构造
data.emplace_back("Alice", 25);

// 3. 利用移动语义
data.emplace_back(std::move(large_string), 30);

// 4. erase-remove 惯用法
v.erase(std::remove_if(v.begin(), v.end(), [](int x) { return x < 0; }), v.end());

// 5. 选择正确的容器
// 随机访问 → vector
// 两端操作 → deque
// 快速查找 → unordered_set/map
// 有序存储 → set/map

// 6. 大数据量使用并行算法（C++17）
std::sort(std::execution::par, data.begin(), data.end());
```
