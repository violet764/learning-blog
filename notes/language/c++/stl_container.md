# C++ STL容器详解

## STL概述与设计理念

**STL（Standard Template Library）** 是C++标准库的核心组成部分，提供了一套高效、通用的数据结构和算法。STL基于模板技术实现，遵循泛型编程思想，将算法与数据结构解耦，实现"一次编写，处处使用"的设计目标。

## 标准模板库（STL）核心

### 容器：vector、map、set、unordered_map

**vector（动态数组）：连续内存的动态扩展**

**内存管理原理：** vector在堆上维护连续的动态数组，通过三个指针管理：指向数据起始的指针、指向最后一个元素之后的指针、指向分配内存末尾的指针。当空间不足时，vector会以几何增长（通常是1.5或2倍）重新分配更大的内存块。

```cpp
#include <vector>
#include <algorithm>

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

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};
// 初始化时分配足够空间存储8个元素

// 添加元素：可能需要重新分配
numbers.push_back(7);  // 如果空间足够，直接构造在末尾
// 如果capacity() == size()，触发重新分配：
// 1. 分配新内存（通常是当前容量的1.5-2倍）
// 2. 将现有元素移动到新内存（拷贝或移动构造）
// 3. 在新位置构造新元素
// 4. 释放旧内存

numbers.insert(numbers.begin() + 2, 8);  // 在索引2处插入8
// 插入操作需要移动后续元素，时间复杂度O(n)

// 删除元素
numbers.pop_back();                      // 删除最后一个：O(1)
numbers.erase(numbers.begin() + 1);     // 删除索引1处的元素：需要移动后续元素

// 容量操作：理解内存分配策略
std::cout << "大小: " << numbers.size() << std::endl;      // 当前元素数量
std::cout << "容量: " << numbers.capacity() << std::endl;  // 当前分配的总空间

numbers.shrink_to_fit();                // 缩减容量到实际大小
// 请求释放未使用的内存，但实现可能选择忽略

// 性能优化技巧
numbers.reserve(100);  // 预分配空间，避免多次重新分配
for (int i = 0; i < 100; i++) {
    numbers.push_back(i);  // 在预留空间内操作，高效
}

// 对比Python的list（也是动态数组，但实现不同）
# Python: numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# Python列表也是动态数组，但采用不同的增长策略
# numbers.append(7)
# numbers.insert(2, 8)
# numbers.pop()
# del numbers[1]
```

**map（有序关联容器）：红黑树实现的键值对存储**

**数据结构原理：** std::map通常基于红黑树（一种自平衡二叉搜索树）实现。红黑树确保最坏情况下的查找、插入、删除操作都是O(log n)时间复杂度，同时保持元素按键排序。

```cpp
#include <map>

// map内部实现原理（基于红黑树）
// template<typename Key, typename Value>
// class map {
// private:
//     struct Node {
//         Key key;
//         Value value;
//         Node* left;
//         Node* right;
//         Node* parent;
//         bool color;  // 红黑树颜色
//     };
//     Node* root_;
// };

std::map<std::string, int> age_map;

// 插入元素：在红黑树中查找合适位置并插入
age_map["Alice"] = 25;  // 运算符[]：如果键不存在则插入，存在则修改
// 等价于：age_map.insert({"Alice", 25}).first->second = 25;

age_map["Bob"] = 30;
age_map.insert({"Charlie", 35});  // insert方法：返回pair<iterator, bool>

// 访问元素（自动排序）：中序遍历红黑树
for (const auto &pair : age_map) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}
// 输出按字典序排序：Alice, Bob, Charlie

// 查找元素：在红黑树中搜索
auto it = age_map.find("Alice");  // O(log n)时间复杂度
if (it != age_map.end()) {
    std::cout << "找到Alice: " << it->second << std::endl;
}

// 性能特性分析
// 插入：O(log n) - 需要重新平衡树
// 查找：O(log n) - 二分搜索
// 删除：O(log n) - 需要重新平衡
// 空间：每个节点需要额外指针和颜色信息

// 与unordered_map的对比
// map：有序，稳定性能，需要比较运算符
// unordered_map：无序，平均O(1)性能，需要哈希函数

// 自定义比较函数
struct CaseInsensitiveCompare {
    bool operator()(const std::string& a, const std::string& b) const {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); }
        );
    }
};

std::map<std::string, int, CaseInsensitiveCompare> case_insensitive_map;

// 对比Python的dict（哈希表实现，无序但平均O(1)）
# age_map = {"Alice": 25, "Bob": 30, "Charlie": 35}
# Python 3.7+ dict保持插入顺序，但本质是无序容器
# for key, value in age_map.items():
#     print(f"{key}: {value}")
```

**set（有序集合）：基于红黑树的唯一元素容器**

**设计原理：** std::set也是基于红黑树实现，但只存储键而不存储值。它保证元素唯一性和自动排序，提供高效的查找、插入和删除操作。

```cpp
#include <set>
#include <iterator>
#include <algorithm>

// set内部实现：与map类似，但节点只存储键值
// template<typename Key>
// class set {
// private:
//     struct Node {
//         Key key;
//         Node* left;
//         Node* right;
//         Node* parent;
//         bool color;
//     };
// };

std::set<int> unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6};
// 初始化时自动去重：重复的1只保留一个
// 自动排序：元素按升序排列

// 自动去重和排序：中序遍历红黑树
for (int num : unique_numbers) {
    std::cout << num << " ";  // 输出: 1 2 3 4 5 6 9（排序后）
}
std::cout << std::endl;

// 集合操作：基于有序序列的高效算法
std::set<int> set1 = {1, 2, 3, 4, 5};
std::set<int> set2 = {4, 5, 6, 7, 8};

// 并集：合并两个有序序列
std::set<int> union_set;
std::set_union(set1.begin(), set1.end(), 
               set2.begin(), set2.end(),
               std::inserter(union_set, union_set.begin()));
// 时间复杂度：O(n + m)，线性合并

// 其他集合操作
std::set<int> intersection_set;
std::set_intersection(set1.begin(), set1.end(),
                      set2.begin(), set2.end(),
                      std::inserter(intersection_set, intersection_set.begin()));
// 交集：{4, 5}

std::set<int> difference_set;
std::set_difference(set1.begin(), set1.end(),
                    set2.begin(), set2.end(),
                    std::inserter(difference_set, difference_set.begin()));
// 差集：set1 - set2 = {1, 2, 3}

// 成员函数版本的集合操作（更高效）
std::set<int> union_set2;
union_set2.insert(set1.begin(), set1.end());
union_set2.insert(set2.begin(), set2.end());
// 同样得到并集，但利用set的自动去重特性

// 性能优化：利用有序特性进行范围查询
auto lower = unique_numbers.lower_bound(3);  // 第一个>=3的元素
auto upper = unique_numbers.upper_bound(6);  // 第一个>6的元素
for (auto it = lower; it != upper; ++it) {
    std::cout << *it << " ";  // 输出3到6之间的元素
}

// 对比Python的set（基于哈希表，无序但平均O(1)）
# unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6}  # 自动去重
# Python set是无序的，但Python 3.7+保持插入顺序
# set1 = {1, 2, 3, 4, 5}
# set2 = {4, 5, 6, 7, 8}
# union_set = set1 | set2  # 并集操作
```

**unordered_map（哈希表）：平均O(1)访问的关联容器**

**哈希表原理：** unordered_map基于哈希表实现，通过哈希函数将键映射到桶(bucket)中。在理想情况下（良好哈希函数、适当负载因子），提供平均O(1)的插入、删除和查找性能。

```cpp
#include <unordered_map>

// unordered_map内部实现原理（简化）
// template<typename Key, typename Value>
// class unordered_map {
// private:
//     struct Node {
//         Key key;
//         Value value;
//         Node* next;  // 链表解决哈希冲突
//     };
//     std::vector<Node*> buckets_;  // 桶数组
//     size_t size_;                // 元素数量
//     float max_load_factor_;       // 最大负载因子
// };

std::unordered_map<std::string, int> word_count;

// 插入和计数：通过哈希函数定位桶
word_count["hello"]++;  // 1. 计算"hello"的哈希值
                         // 2. 哈希值 % 桶数量 = 桶索引
                         // 3. 在对应链表中查找/插入

word_count["world"]++;
word_count["hello"]++;  // 找到已有键，增加值

// 遍历（无序）：按照桶顺序遍历
for (const auto &pair : word_count) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}
// 输出顺序取决于哈希函数和桶分布，无保证

// 哈希表性能关键参数
std::cout << "负载因子: " << word_count.load_factor() << std::endl;
std::cout << "桶数量: " << word_count.bucket_count() << std::endl;
std::cout << "最大负载因子: " << word_count.max_load_factor() << std::endl;

// 性能优化：调整哈希表参数
word_count.reserve(100);  // 预留空间，减少重新哈希
word_count.max_load_factor(0.7f);  // 设置最大负载因子

// 自定义哈希函数（重要：避免哈希冲突）
struct StringHash {
    size_t operator()(const std::string& s) const {
        // 简单哈希函数示例（实际std::hash更复杂）
        size_t hash = 0;
        for (char c : s) {
            hash = hash * 31 + c;  // 常用质数乘法
        }
        return hash;
    }
};

struct StringEqual {
    bool operator()(const std::string& a, const std::string& b) const {
        return a == b;  // 相等性比较
    }
};

std::unordered_map<std::string, int, StringHash, StringEqual> custom_map;

// 哈希冲突处理：链表法（开放定址法在C++标准库中较少使用）
// 当多个键哈希到同一桶时，使用链表存储

// 性能分析
// 最佳情况：O(1) 所有操作
// 最坏情况：O(n) 所有键哈希到同一桶（哈希攻击）
// 平均情况：O(1) 假设良好哈希函数和适当负载因子

// 与map的对比选择
// 使用unordered_map当：需要快速查找，不关心顺序，有良好哈希函数
// 使用map当：需要有序遍历，性能稳定性重要，或键类型无良好哈希函数

// 对比Python的dict（也是哈希表实现，Python 3.6+优化了内存布局）
# word_count = {}
# word_count["hello"] = word_count.get("hello", 0) + 1
# Python dict使用更复杂的内存布局优化（紧凑字典）
```

### 迭代器：概念与使用

**迭代器基础：统一的容器访问接口**

**设计模式：** 迭代器模式提供了一种顺序访问聚合对象元素的方法，而不暴露其底层表示。C++迭代器是泛型编程的核心，将算法与容器解耦。

```cpp
#include <iterator>

std::vector<int> vec = {10, 20, 30, 40, 50};

// 迭代器本质：指针的抽象
// vector<int>::iterator 实际可能是 int* 的包装
// 但提供统一的接口，隐藏具体实现

// 使用迭代器遍历：与指针操作类似
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";  // 解引用获取值
}
std::cout << std::endl;

// 迭代器操作符重载原理
// class iterator {
// public:
//     T& operator*() { return *current_; }     // 解引用
//     iterator& operator++() { ++current_; return *this; }  // 前缀++
//     bool operator!=(const iterator& other) { return current_ != other.current_; }
// };

// 反向迭代器：适配器模式的应用
for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
    std::cout << *rit << " ";  // 50 40 30 20 10
    // 反向迭代器内部存储正向迭代器，但操作方向相反
}
std::cout << std::endl;

// 迭代器失效问题：重要概念
std::vector<int> numbers = {1, 2, 3, 4, 5};
auto it = numbers.begin() + 2;  // 指向3
numbers.push_back(6);          // 可能触发重新分配
// it可能失效！指向已释放的内存
// 解决方案：使用索引或重新获取迭代器

// 常量迭代器：只读访问
for (auto cit = vec.cbegin(); cit != vec.cend(); ++cit) {
    // *cit = 100;  // 错误：常量迭代器不能修改元素
    std::cout << *cit << " ";
}

// 迭代器适配器：功能扩展
std::vector<int> data = {1, 2, 3, 4, 5};
// 插入迭代器：将赋值操作转换为插入操作
std::copy(data.begin(), data.end(), 
          std::back_inserter(vec));  // 在vec末尾插入

// 流迭代器：连接容器与流
std::copy(vec.begin(), vec.end(),
          std::ostream_iterator<int>(std::cout, " "));

// 对比Python的迭代器协议
# Python迭代器基于__iter__和__next__方法
# for item in vec:  # 调用vec.__iter__()
#     print(item, end=" ")
# C++迭代器更接近指针，提供更细粒度的控制
```

**迭代器类别：层次化的能力模型**

**概念设计：** C++迭代器按照能力分为5个层次，每个层次提供特定的操作集合。算法根据需要的迭代器类别进行约束，实现编译时多态。

```cpp
// 迭代器类别层次（从弱到强）：

// 1. 输入迭代器：只读，单向，单遍扫描
// 典型应用：std::istream_iterator
// 支持操作：==, !=, ++, *, ->
// 只能单遍遍历，遍历后迭代器失效

template<typename InputIt>
void process_input(InputIt first, InputIt last) {
    // 只能读取，不能修改元素
    while (first != last) {
        std::cout << *first << " ";
        ++first;  // 只能向前移动
    }
}

// 2. 输出迭代器：只写，单向，单遍扫描  
// 典型应用：std::ostream_iterator, std::back_inserter
// 支持操作：++, *（仅用于赋值）
// 只能单遍写入，不能读取

template<typename OutputIt>
void generate_output(OutputIt dest, int count) {
    for (int i = 0; i < count; ++i) {
        *dest = i;    // 只能赋值，不能读取
        ++dest;
    }
}

// 3. 前向迭代器：可读写，单向，多遍扫描
// 典型容器：std::forward_list
// 支持操作：输入迭代器 + 输出迭代器的所有操作
// 可以多遍遍历，迭代器在遍历间保持有效

template<typename ForwardIt>
bool is_palindrome(ForwardIt first, ForwardIt last) {
    // 需要多遍扫描，验证回文
    auto mid = first;
    auto end = last;
    while (first != end && ++first != end) {
        ++mid;
        --end;
    }
    // 可以再次从头开始扫描
    return std::equal(first, mid, std::reverse_iterator(last));
}

// 4. 双向迭代器：可读写，双向移动
// 典型容器：std::list, std::set, std::map
// 支持操作：前向迭代器 + --（递减）
// 可以向前和向后移动

template<typename BidirIt>
void reverse_range(BidirIt first, BidirIt last) {
    while ((first != last) && (first != --last)) {
        std::iter_swap(first, last);  // 需要双向移动能力
        ++first;
    }
}

// 5. 随机访问迭代器：最强能力，直接索引访问
// 典型容器：std::vector, std::array, std::deque
// 支持操作：双向迭代器 + [], +, -, <, >等
// 可以在常数时间内跳到任意位置

template<typename RandomIt>
void quick_sort(RandomIt first, RandomIt last) {
    if (last - first > 1) {  // 随机访问迭代器支持减法
        auto pivot = first + (last - first) / 2;  // 直接计算中间位置
        // 快速排序算法需要随机访问能力
    }
}

// 迭代器类别标签：编译时类型识别
// 用于算法重载和优化
template<typename It>
void algorithm_impl(It first, It last, std::random_access_iterator_tag) {
    // 针对随机访问迭代器的优化版本
    // 可以使用[]操作符和指针算术
}

template<typename It>
void algorithm_impl(It first, It last, std::forward_iterator_tag) {
    // 针对前向迭代器的通用版本
    // 只能使用++操作符
}

// 迭代器特性：获取迭代器相关信息
template<typename It>
void print_iterator_info() {
    using iterator_category = typename std::iterator_traits<It>::iterator_category;
    using value_type = typename std::iterator_traits<It>::value_type;
    using difference_type = typename std::iterator_traits<It>::difference_type;
    
    std::cout << "值类型: " << typeid(value_type).name() << std::endl;
    // 根据迭代器类别选择最优算法
}
```

### 算法：sort、find、transform等

**常用算法示例：泛型编程的威力**

**设计哲学：** STL算法通过迭代器将算法与容器解耦，实现"一次编写，处处使用"。算法不关心容器的具体类型，只关心迭代器提供的接口。

```cpp
#include <algorithm>
#include <numeric>
#include <iterator>

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};

// 排序：std::sort - 快速排序的泛化实现
std::sort(numbers.begin(), numbers.end());
// 内部实现：使用introspective sort（快速排序+堆排序+插入排序）
// 时间复杂度：平均O(n log n)，最坏O(n log n)
// 要求：随机访问迭代器，元素可比较（提供<运算符）

// 稳定排序：保持相等元素的相对顺序
std::stable_sort(numbers.begin(), numbers.end());
// 时间复杂度：O(n log² n) 或 O(n log n)（有额外内存时）

// 部分排序：只排序前k个元素
std::partial_sort(numbers.begin(), numbers.begin() + 3, numbers.end());
// 前3个元素有序，其余无序但都在前3个之后

// 查找：std::find - 线性搜索
auto found = std::find(numbers.begin(), numbers.end(), 5);
if (found != numbers.end()) {
    std::cout << "找到5在位置: " << std::distance(numbers.begin(), found) << std::endl;
}
// 时间复杂度：O(n)，适用于无序序列

// 二分查找：要求序列已排序
auto binary_found = std::binary_search(numbers.begin(), numbers.end(), 5);
if (binary_found) {
    std::cout << "通过二分查找找到5" << std::endl;
}
// 时间复杂度：O(log n)，但要求序列有序

// 转换：std::transform - 类似函数式编程的map
std::vector<int> squared;
std::transform(numbers.begin(), numbers.end(), 
               std::back_inserter(squared),
               [](int x) { return x * x; });
// 原理：对每个元素应用一元函数，结果写入目标区间
// 可以处理不同容器类型：vector → list, array → vector等

// 原地转换：修改原序列
std::transform(numbers.begin(), numbers.end(), numbers.begin(),
               [](int x) { return x * 2; });

// 过滤：erase-remove惯用法（C++经典模式）
numbers.erase(std::remove_if(numbers.begin(), numbers.end(),
                            [](int x) { return x % 2 == 0; }),
              numbers.end());
// 原理：remove_if将满足条件的元素移动到末尾，返回新的逻辑结尾
// erase真正删除这些元素

// C++20的简化版本（如果可用）
// std::erase_if(numbers, [](int x) { return x % 2 == 0; });

// 累加：std::accumulate - 类似函数式编程的reduce
int total = std::accumulate(numbers.begin(), numbers.end(), 0);
// 原理：从初始值0开始，对每个元素应用累加操作
// 可以自定义操作：乘法、字符串连接等

double product = std::accumulate(numbers.begin(), numbers.end(), 1.0,
                                [](double a, int b) { return a * b; });

// 其他重要算法
std::vector<int> copy_numbers;
std::copy(numbers.begin(), numbers.end(), 
          std::back_inserter(copy_numbers));  // 复制

std::reverse(numbers.begin(), numbers.end());  // 反转

auto max_it = std::max_element(numbers.begin(), numbers.end());  // 最大值
auto min_it = std::min_element(numbers.begin(), numbers.end());  // 最小值

// 算法复杂度保证
// 非修改序列操作：O(n) - find, count, for_each等
// 修改序列操作：O(n) - copy, transform, replace等  
// 排序和相关操作：O(n log n) - sort, stable_sort等
// 数值算法：O(n) - accumulate, inner_product等

// 对比Python的函数式操作
# numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# numbers.sort()
# squared = list(map(lambda x: x*x, numbers))
# numbers = list(filter(lambda x: x % 2 != 0, numbers))
# total = sum(numbers)
# Python更函数式，C++ STL更注重性能和泛型
```

### lambda表达式（C++11）：匿名函数与闭包

**lambda语法：函数对象的语法糖**

**实现原理：** lambda表达式在编译时被转换为匿名函数对象（functor）。编译器根据捕获列表生成对应的类，重载函数调用运算符。

```cpp
// 基本lambda：无捕获的简单函数对象
auto square = [](int x) { return x * x; };
std::cout << square(5) << std::endl;  // 25

// 编译后等价于：
// class __lambda_1 {
// public:
//     auto operator()(int x) const { return x * x; }
// };
// __lambda_1 square;

// 捕获外部变量：创建闭包
int factor = 3;
auto multiply = [factor](int x) { return x * factor; };
// 等价于：
// class __lambda_2 {
// private:
//     int factor;  // 按值捕获的副本
// public:
//     __lambda_2(int f) : factor(f) {}
//     auto operator()(int x) const { return x * factor; }
// };
// __lambda_2 multiply(factor);

// 按引用捕获：共享外部变量
int counter = 0;
auto incrementer = [&counter]() { counter++; };
// 等价于：
// class __lambda_3 {
// private:
//     int& counter;  // 引用捕获
// public:
//     __lambda_3(int& c) : counter(c) {}
//     auto operator()() { counter++; }  // 非const，可以修改
// };

// mutable lambda：允许修改按值捕获的变量
int value = 10;
auto modifier = [value]() mutable { 
    // value是副本，可以修改但不影响外部
    value += 5; 
    return value; 
};

// 在算法中使用lambda：STL算法的完美搭档
std::vector<int> nums = {1, 2, 3, 4, 5};
std::for_each(nums.begin(), nums.end(), 
              [](int &x) { x *= 2; });
// nums变为: {2, 4, 6, 8, 10}

// 泛型lambda（C++14）：自动类型推导
auto generic_add = [](auto a, auto b) { return a + b; };
std::cout << generic_add(5, 3.14) << std::endl;  // 8.14

// 初始化捕获（C++14）：移动语义支持
std::unique_ptr<int> ptr = std::make_unique<int>(42);
auto lambda_with_move = [p = std::move(ptr)]() {
    return *p;
};

// constexpr lambda（C++17）：编译时计算
constexpr auto compile_time_square = [](int x) { return x * x; };
constexpr int result = compile_time_square(5);  // 编译时计算

// 立即调用lambda：一次性函数
int result = [](int a, int b) { return a + b; }(3, 4);  // 直接调用：7

// lambda与标准库算法的结合
std::vector<std::string> words = {"hello", "world", "cpp", "lambda"};

// 排序：按长度排序
std::sort(words.begin(), words.end(), 
          [](const std::string& a, const std::string& b) {
              return a.length() < b.length();
          });

// 查找：查找特定条件的元素
auto long_word = std::find_if(words.begin(), words.end(),
                             [](const std::string& s) {
                                 return s.length() > 5;
                             });

// 性能考虑：lambda vs 函数对象 vs 函数指针
// lambda：通常被内联，零开销抽象
// 函数对象：同样可内联，但需要显式定义类
// 函数指针：可能有间接调用开销

// 对比Python的lambda（真正的闭包，动态类型）
# square = lambda x: x * x
# factor = 3
# multiply = lambda x: x * factor  # Python闭包捕获引用
# nums = [1, 2, 3, 4, 5]
# nums = list(map(lambda x: x * 2, nums))
# Python lambda是真正的闭包，C++ lambda是编译时生成的函数对象
```

**捕获方式：**
- `[]`：不捕获任何变量
- `[=]`：按值捕获所有外部变量
- `[&]`：按引用捕获所有外部变量
- `[x, &y]`：按值捕获x，按引用捕获y
- `[this]`：捕获当前对象的this指针


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
