# C++高级特性与实战

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

## 错误处理

### 异常处理：try-catch-throw

**基本异常处理：栈展开与资源安全**

**异常机制原理：** C++异常通过栈展开(stack unwinding)实现错误传播。当异常抛出时，运行时系统沿着调用栈向上查找匹配的catch块，同时自动调用局部对象的析构函数，确保资源安全释放。

```cpp
#include <stdexcept>
#include <memory>

// 异常抛出：创建异常对象并开始栈展开
double safe_divide(double a, double b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
        // 1. 构造invalid_argument异常对象
        // 2. 开始栈展开：从当前函数开始向上
        // 3. 析构局部对象（RAII确保资源释放）
        // 4. 查找匹配的catch块
    }
    return a / b;
}

// 异常安全函数示例
class Resource {
    std::unique_ptr<int> data;
public:
    Resource() : data(std::make_unique<int>(42)) {}
    ~Resource() { std::cout << "资源释放" << std::endl; }
    
    void risky_operation() {
        if (rand() % 2 == 0) {
            throw std::runtime_error("操作失败");
        }
        // 即使抛出异常，unique_ptr也会自动释放
    }
};

int main() {
    try {
        Resource res;  // RAII：异常安全的基础
        res.risky_operation();
        
        double result = safe_divide(10, 0);
        std::cout << "结果: " << result << std::endl;
    }
    // 异常捕获：按类型匹配，从具体到一般
    catch (const std::invalid_argument &e) {
        // 最具体的异常类型先匹配
        std::cerr << "数学错误: " << e.what() << std::endl;
        // e.what()返回异常描述字符串
    }
    catch (const std::exception &e) {
        // 基类异常捕获更一般的错误
        std::cerr << "一般错误: " << e.what() << std::endl;
    }
    catch (...) {
        // 捕获所有异常（包括非std::exception派生的）
        std::cerr << "未知错误" << std::endl;
        // 通常用于日志记录或紧急清理
    }
    
    return 0;
}

// 异常性能考虑
// 正常执行路径：零开销（现代编译器优化）
// 抛出异常时：有运行时开销（栈展开、类型匹配）
// 适用于罕见错误，不应用于流程控制

// 异常安全保证级别
class ExceptionSafe {
    std::vector<int> data;
public:
    // 基本保证：不泄漏资源，对象有效
    void basic_guarantee() {
        auto backup = data;  // 备份状态
        try {
            // 可能抛出异常的操作
            data.push_back(42);
        } catch (...) {
            data = std::move(backup);  // 恢复状态
            throw;  // 重新抛出
        }
    }
    
    // 强保证：操作原子性（成功或完全回滚）
    void strong_guarantee() {
        std::vector<int> new_data = data;  // 操作副本
        new_data.push_back(42);           // 在副本上操作
        data = std::move(new_data);       // 原子性提交
    }
    
    // 不抛出保证：函数声明noexcept
    void no_throw() noexcept {
        // 简单操作，保证不抛出
    }
};

// 对比Python的异常处理（更轻量级，用于流程控制）
# def safe_divide(a, b):
#     if b == 0:
#         raise ValueError("除数不能为零")
#     return a / b
# 
# try:
#     result = safe_divide(10, 0)
# except ValueError as e:
#     print(f"数学错误: {e}")
# except Exception as e:
#     print(f"一般错误: {e}")
# Python异常更常用于流程控制，C++异常更强调资源安全
```

**自定义异常：**
```cpp
class FileNotFoundException : public std::runtime_error {
public:
    FileNotFoundException(const std::string &filename)
        : std::runtime_error("文件未找到: " + filename) {}
};

void read_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileNotFoundException(filename);
    }
    // 读取文件内容...
}
```

### 异常安全与noexcept

**异常安全保证级别：**
1. **基本保证**：不泄漏资源，对象处于有效状态
2. **强保证**：操作要么成功，要么回滚到原始状态
3. **不抛出保证**：操作绝不抛出异常

**noexcept说明符：**
```cpp
// 不抛出异常的函数
void no_throw_function() noexcept {
    // 这个函数保证不抛出异常
}

// 条件性noexcept
template<typename T>
void swap(T &a, T &b) noexcept(noexcept(a.swap(b))) {
    a.swap(b);
}

// 移动构造函数通常标记为noexcept
class Movable {
public:
    Movable(Movable &&other) noexcept {
        // 移动资源...
    }
};
```

## 现代C++特性

### 自动类型推导：auto关键字

**auto使用场景：编译时类型推导的强大工具**

**推导规则：** auto使用模板参数推导规则，根据初始化表达式推导类型。它不是动态类型，而是静态类型推导，在编译时确定具体类型。

```cpp
// auto推导规则：类似于模板参数推导
// auto x = expr; 等价于：template<typename T> void f(T x); f(expr);

// 基本类型推导
auto number = 42;                    // int（字面量类型）
auto name = std::string("Alice");    // std::string（构造函数返回类型）
auto scores = std::vector<int>{90, 85, 95};  // std::vector<int>（初始化列表）

// auto推导的细节
auto x1 = 5;              // int
auto x2 = 5.0;            // double
auto x3 = 5.0f;           // float
auto x4 = "hello";        // const char[6]（数组类型）
auto x5 = std::string("hello");  // std::string

// 引用和const的推导
auto y1 = x1;             // int（值类型，复制）
auto& y2 = x1;            // int&（引用）
const auto y3 = x1;       // const int
const auto& y4 = x1;      // const int&

auto&& y5 = x1;           // int&（左值引用）
auto&& y6 = 42;           // int&&（右值引用）

// 在迭代器中使用：避免冗长的类型名称
auto it = scores.begin();            // std::vector<int>::iterator
for (auto &score : scores) {         // int&，可以修改元素
    score += 5;
}

// 在泛型编程中的威力
template<typename Container>
auto get_first(const Container& c) -> decltype(*c.begin()) {
    return *c.begin();  // 返回类型由容器元素类型决定
}

// 函数返回类型推导（C++14）
auto add(int a, int b) {      // 自动推导返回类型
    return a + b;             // 推导为int
}

// 多返回语句需要相同类型
auto calculate(int x) {
    if (x > 0) {
        return x * 1.5;  // double
    } else {
        return x * 2;    // int → 错误！类型不一致
    }
}

// 尾置返回类型（C++11）：解决复杂返回类型
auto complex_function() -> decltype(some_expression) {
    return some_expression;
}

// auto在lambda表达式中的应用（C++14）
auto lambda = [](auto x, auto y) { return x + y; };  // 泛型lambda

// 结构化绑定（C++17）：结合auto解构对象
auto [min, max] = std::minmax({3, 1, 4, 1, 5});  // min和max自动推导

// 性能考虑：auto与移动语义
auto vec = get_large_vector();  // 可能触发拷贝或移动
// 如果get_large_vector()返回右值，auto会推导为值类型，可能移动

// 最佳实践：何时使用auto
// 推荐：迭代器、lambda、模板代码、复杂类型
// 谨慎：基本类型、需要明确类型的场景

// 对比Python的动态类型（运行时类型）
# number = 42          # 运行时确定类型
# name = "Alice"       # 可以随时改变类型
# scores = [90, 85, 95] # 动态类型，灵活但可能运行时错误
# C++的auto是编译时静态类型，Python是运行时动态类型
```

**decltype类型推导：**
```cpp
int x = 10;
decltype(x) y = 20;  // y的类型与x相同（int）

auto add(int a, int b) -> decltype(a + b) {
    return a + b;  // 返回类型由a+b的类型决定
}
```

### 移动语义与右值引用（C++11）：避免不必要的拷贝

**左值 vs 右值：值类别系统的核心概念**

**值类别层次：**
- **左值(lvalue)**：有标识符，可以取地址（变量、函数名等）
- **将亡值(xvalue)**：即将被移动的资源（std::move的结果等）
- **纯右值(prvalue)**：临时对象（字面量、函数返回的临时值等）

```cpp
#include <utility>

// 值类别示例
int x = 10;           // x是左值
int& lref = x;        // 左值引用
int&& rref = 42;      // 右值引用（绑定到临时对象）

// 移动语义实现：资源所有权的转移
class String {
private:
    char *data;
    size_t length;
    
public:
    // 移动构造函数：从右值"窃取"资源
    String(String &&other) noexcept 
        : data(other.data), length(other.length) {
        // 转移资源所有权
        other.data = nullptr;    // 使源对象处于有效但空的状态
        other.length = 0;
    }
    
    // 移动赋值运算符
    String &operator=(String &&other) noexcept {
        if (this != &other) {    // 自赋值检查
            delete[] data;       // 释放当前资源
            data = other.data;   // 转移新资源
            length = other.length;
            other.data = nullptr;  // 使源对象为空
            other.length = 0;
        }
        return *this;
    }
    
    // 拷贝构造函数（对比）
    String(const String &other) 
        : data(new char[other.length + 1]), length(other.length) {
        std::copy(other.data, other.data + length + 1, data);  // 深拷贝
    }
    
    ~String() {
        delete[] data;  // 空指针delete是安全的
    }
};

// 移动语义的应用场景
String create_string() {
    String temp("Hello");
    return temp;  // 返回值优化(RVO)或移动语义
}

String s1 = create_string();  // 可能触发移动构造或RVO

// std::move：将左值转换为右值引用
String s2("World");
String s3 = std::move(s2);  // 显式移动，s2变为空

// 移动语义在STL中的应用
std::vector<String> strings;
strings.push_back(String("Hello"));  // 移动而非拷贝

// 完美转发：保持值类别
template<typename T>
void forwarder(T&& arg) {  // 万能引用
    processor(std::forward<T>(arg));  // 完美转发
}

// 移动语义的性能优势
class HeavyObject {
    std::vector<int> large_data;
public:
    // 移动构造：只需转移指针，O(1)时间复杂度
    HeavyObject(HeavyObject&& other) noexcept 
        : large_data(std::move(other.large_data)) {}
    
    // 拷贝构造：需要复制所有数据，O(n)时间复杂度
    HeavyObject(const HeavyObject& other) 
        : large_data(other.large_data) {}
};

// 移动语义的注意事项
class Resource {
    int* ptr;
public:
    Resource(Resource&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;  // 必须置空源对象
    }
    
    // 错误示例：忘记置空源对象
    // Resource(Resource&& other) : ptr(other.ptr) {}
    // 会导致双重释放！
};

// noexcept的重要性：移动操作通常标记为noexcept
// 允许STL容器在重新分配时使用移动而非拷贝
// 例如：vector在扩容时会优先移动元素（如果移动操作是noexcept）

// 对比Python的引用计数（自动内存管理）
# Python使用引用计数自动管理内存，无需手动移动
# 但C++的移动语义提供了更精细的控制和更好的性能
```

**std::move：强制转换为右值引用**
```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination = std::move(source);  // 移动而非拷贝

// source现在为空
td::cout << "source大小: " << source.size() << std::endl;      // 0
std::cout << "destination大小: " << destination.size() << std::endl;  // 5
```

### 范围for循环与初始化列表

**范围for循环（C++11）：**
```cpp
std::vector<std::string> names = {"Alice", "Bob", "Charlie"};

// 只读访问
for (const auto &name : names) {
    std::cout << name << " ";
}
std::cout << std::endl;

// 可修改访问
for (auto &name : names) {
    name += " Smith";
}

// 对比Python
# for name in names:
#     print(name, end=" ")
```

**初始化列表（C++11）：**
```cpp
// 统一初始化语法
std::vector<int> v1{1, 2, 3, 4, 5};      // 初始化列表
std::vector<int> v2 = {1, 2, 3, 4, 5};   // 复制初始化

// 类成员初始化
class Point {
public:
    int x, y;
    
    Point(int x, int y) : x(x), y(y) {}
};

Point p1{10, 20};      // 直接初始化
Point p2 = {30, 40};   // 复制初始化

// std::initializer_list使用
void print_numbers(std::initializer_list<int> nums) {
    for (int num : nums) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

print_numbers({1, 2, 3, 4, 5});  // 传递初始化列表
```

## 实战应用与项目结构

### 头文件与源文件组织：C++模块化编程的基础

**头文件（.h/.hpp）：声明接口与编译防火墙**

**设计原则：** 头文件应该只包含声明，不包含实现细节。使用头文件保护机制防止重复包含，合理组织包含关系减少编译依赖。

```cpp
// math_utils.h
#ifndef MATH_UTILS_H        // 头文件保护：防止重复包含
#define MATH_UTILS_H

// 包含必要的外部头文件（避免传递依赖）
#include <vector>           // 直接使用的标准库
#include <string>           // 函数参数或返回类型需要

// 前向声明：减少编译依赖
class Database;            // 只需要指针或引用时使用前向声明

namespace math {           // 命名空间：防止命名冲突
    
    // 函数声明：只提供接口，隐藏实现
    double calculate_average(const std::vector<double> &numbers);
    
    // 内联函数：简单函数可以在头文件中定义
    inline double square(double x) { return x * x; }
    
    // 模板必须在头文件中定义（编译时需要实例化）
    template<typename T>
    T max(const T& a, const T& b) {
        return (a > b) ? a : b;
    }
    
    // 类声明：公开接口，隐藏私有实现
    class Statistics {
    private:
        // 使用Pimpl idiom隐藏实现细节
        class Impl;
        std::unique_ptr<Impl> pimpl_;  // 实现指针
        
        std::vector<double> data;      // 简单的私有成员
        
    public:
        // 构造函数/析构函数声明
        Statistics();
        ~Statistics();
        
        // 禁用拷贝（Rule of Five）
        Statistics(const Statistics&) = delete;
        Statistics& operator=(const Statistics&) = delete;
        
        // 允许移动
        Statistics(Statistics&&) noexcept;
        Statistics& operator=(Statistics&&) noexcept;
        
        // 公开接口
        void add_data(double value);
        double mean() const;
        double variance() const;
        
        // 静态成员声明
        static constexpr double PI = 3.141592653589793;
    };
    
    // 枚举声明
    enum class Operation {
        ADD, SUBTRACT, MULTIPLY, DIVIDE
    };
    
    // 类型别名（C++11）
    using DataVector = std::vector<double>;
    
} // namespace math

#endif // MATH_UTILS_H

// 头文件设计最佳实践：
// 1. 最小化包含：只包含直接需要的头文件
// 2. 使用前向声明：减少编译依赖
// 3. 内联简单函数：避免函数调用开销
// 4. 使用命名空间：组织代码，防止冲突
// 5. Pimpl模式：隐藏实现细节，减少编译时间
```

**源文件（.cpp）：实现功能**
```cpp
// math_utils.cpp
#include "math_utils.h"
#include <cmath>
#include <numeric>

namespace math {
    double calculate_average(const std::vector<double> &numbers) {
        if (numbers.empty()) return 0.0;
        double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
        return sum / numbers.size();
    }
    
    void Statistics::add_data(double value) {
        data.push_back(value);
    }
    
    double Statistics::mean() const {
        return calculate_average(data);
    }
}
```

**对比Python的模块：**
```python
# math_utils.py
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

class Statistics:
    def __init__(self):
        self.data = []
    
    def add_data(self, value):
        self.data.append(value)
    
    def mean(self):
        return calculate_average(self.data)
```

### CMake基础配置

**基本CMakeLists.txt：**
```cmake
# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称和语言
project(MyCppProject LANGUAGES CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
endif()

# 添加可执行文件
add_executable(main main.cpp math_utils.cpp)

# 添加库（如果有）
# add_library(math_utils STATIC math_utils.cpp)
# target_link_libraries(main math_utils)
```

### 常用设计模式在C++中的实现

**单例模式：**
```cpp
class Logger {
private:
    static Logger* instance;
    std::ofstream log_file;
    
    // 私有构造函数
    Logger() {
        log_file.open("app.log", std::ios::app);
    }
    
public:
    // 删除拷贝构造函数和赋值运算符
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // 获取单例实例
    static Logger& get_instance() {
        static Logger instance;  // C++11保证线程安全
        return instance;
    }
    
    void log(const std::string &message) {
        log_file << message << std::endl;
    }
};

// 使用单例
Logger::get_instance().log("应用程序启动");
```

**工厂模式：**
```cpp
class Shape {
public:
    virtual void draw() = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    void draw() override {
        std::cout << "绘制圆形" << std::endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() override {
        std::cout << "绘制矩形" << std::endl;
    }
};

class ShapeFactory {
public:
    static std::unique_ptr<Shape> create_shape(const std::string &type) {
        if (type == "circle") {
            return std::make_unique<Circle>();
        } else if (type == "rectangle") {
            return std::make_unique<Rectangle>();
        }
        return nullptr;
    }
};

// 使用工厂
auto shape = ShapeFactory::create_shape("circle");
if (shape) {
    shape->draw();
}
```

### 性能优化要点

**避免不必要的拷贝：**
```cpp
// 不良做法：不必要的拷贝
std::vector<int> process_data(std::vector<int> data) {  // 拷贝
    // 处理数据...
    return data;  // 可能再次拷贝
}

// 优化做法：使用引用和移动语义
std::vector<int> process_data(const std::vector<int> &data) {  // 引用，无拷贝
    std::vector<int> result = data;  // 只在需要时拷贝
    // 处理result...
    return result;  // 可能触发移动语义
}

// 最佳做法：原地修改
void process_data_inplace(std::vector<int> &data) {  // 引用，原地修改
    // 直接修改data...
}
```

**预分配内存：**
```cpp
std::vector<int> numbers;
numbers.reserve(1000);  // 预分配内存，避免多次重分配

for (int i = 0; i < 1000; i++) {
    numbers.push_back(i);  // 不会触发重分配
}
```

**使用emplace_back避免临时对象：**
```cpp
std::vector<std::string> names;

// 不良做法：创建临时string对象
names.push_back(std::string("Alice"));  // 创建临时对象然后移动

// 优化做法：直接构造
names.emplace_back("Alice");  // 在vector中直接构造，无临时对象
```

---

**全书总结：**

通过这三个文件的学习，您应该已经掌握了：

1. **C++基础语法**：从Python开发者角度理解C++的核心概念
2. **内存管理**：指针、引用、智能指针和RAII原则
3. **面向对象**：类、继承、多态和现代C++特性
4. **STL和高级特性**：容器、算法、lambda和移动语义
5. **实战应用**：项目组织、设计模式和性能优化

**关键转换思维：**
- Python的"动态灵活"转向C++的"静态安全"
- Python的"自动内存管理"转向C++的"精确控制"
- Python的"解释执行"转向C++的"编译优化"

**下一步学习建议：**
1. 实践小型项目，巩固所学知识
2. 学习C++模板元编程
3. 了解并发编程（多线程）
4. 探索C++20/23的新特性
5. 阅读优秀的C++开源项目代码

C++是一门强大而复杂的语言，持续实践和深入学习是掌握它的关键。