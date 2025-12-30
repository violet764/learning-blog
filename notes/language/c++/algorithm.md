# C++算法库（STL Algorithms）

C++标准模板库（STL）提供了丰富的算法组件，这些算法是现代C++编程的核心工具。它们遵循函数式编程范式，提供高效、类型安全的操作，能够处理各种容器和迭代器。

## 算法库概述

### 设计哲学与优势

**核心设计原则：**
- **泛型编程**：算法独立于数据类型和容器
- **迭代器抽象**：通过迭代器统一访问不同容器
- **函数对象支持**：可自定义比较和操作逻辑
- **无副作用**：大多数算法不修改原始数据

**主要优势：**
- 代码复用性高，避免重复造轮子
- 性能优化，经过严格测试和优化
- 类型安全，编译时错误检查
- 表达力强，代码简洁明了

## 常用算法分类详解

### 1. 非修改序列操作

这些算法不修改容器内容，只进行查询或计算。

#### 查找算法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include <iterator>

void demonstrate_find_algorithms() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. find: 查找特定元素
    auto it = std::find(numbers.begin(), numbers.end(), 5);
    if (it != numbers.end()) {
        std::cout << "找到元素5，位置：" << std::distance(numbers.begin(), it) << std::endl;
    }
    
    // 2. find_if: 使用条件查找
    auto even_it = std::find_if(numbers.begin(), numbers.end(), 
                               [](int n) { return n % 2 == 0; });
    if (even_it != numbers.end()) {
        std::cout << "第一个偶数：" << *even_it << std::endl;
    }
    
    // 3. find_if_not: 查找不满足条件的元素
    auto odd_it = std::find_if_not(numbers.begin(), numbers.end(),
                                 [](int n) { return n % 2 == 0; });
    if (odd_it != numbers.end()) {
        std::cout << "第一个奇数：" << *odd_it << std::endl;
    }
    
    // 4. adjacent_find: 查找相邻重复元素
    std::vector<int> duplicates = {1, 2, 2, 3, 4, 4, 5};
    auto adj_it = std::adjacent_find(duplicates.begin(), duplicates.end());
    if (adj_it != duplicates.end()) {
        std::cout << "相邻重复元素：" << *adj_it << std::endl;
    }
    
    // 5. search: 查找子序列
    std::vector<int> subseq = {3, 4, 5};
    auto sub_it = std::search(numbers.begin(), numbers.end(),
                             subseq.begin(), subseq.end());
    if (sub_it != numbers.end()) {
        std::cout << "找到子序列，起始位置：" 
                  << std::distance(numbers.begin(), sub_it) << std::endl;
    }
}
```

#### 统计与计数算法

```cpp
void demonstrate_count_algorithms() {
    std::vector<int> numbers = {1, 2, 3, 2, 4, 2, 5, 6, 2, 7};
    
    // 1. count: 统计特定元素出现次数
    int count_2 = std::count(numbers.begin(), numbers.end(), 2);
    std::cout << "数字2出现次数：" << count_2 << std::endl;
    
    // 2. count_if: 使用条件统计
    int even_count = std::count_if(numbers.begin(), numbers.end(),
                                  [](int n) { return n % 2 == 0; });
    std::cout << "偶数个数：" << even_count << std::endl;
    
    // 3. all_of: 检查所有元素是否满足条件
    bool all_positive = std::all_of(numbers.begin(), numbers.end(),
                                  [](int n) { return n > 0; });
    std::cout << "所有元素都是正数：" << std::boolalpha << all_positive << std::endl;
    
    // 4. any_of: 检查是否有元素满足条件
    bool has_negative = std::any_of(numbers.begin(), numbers.end(),
                                   [](int n) { return n < 0; });
    std::cout << "存在负数：" << has_negative << std::endl;
    
    // 5. none_of: 检查是否没有元素满足条件
    bool no_zero = std::none_of(numbers.begin(), numbers.end(),
                               [](int n) { return n == 0; });
    std::cout << "没有零元素：" << no_zero << std::endl;
}
```

### 2. 修改序列操作

这些算法会修改容器内容。

#### 复制与移动算法

```cpp
void demonstrate_copy_algorithms() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination(5);
    
    // 1. copy: 简单复制
    std::copy(source.begin(), source.end(), destination.begin());
    std::cout << "复制结果：";
    for (int n : destination) std::cout << n << " ";
    std::cout << std::endl;
    
    // 2. copy_if: 条件复制
    std::vector<int> even_numbers;
    std::copy_if(source.begin(), source.end(),
                 std::back_inserter(even_numbers),
                 [](int n) { return n % 2 == 0; });
    std::cout << "偶数复制结果：";
    for (int n : even_numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    // 3. copy_n: 复制指定数量元素
    std::vector<int> partial_copy(3);
    std::copy_n(source.begin(), 3, partial_copy.begin());
    std::cout << "部分复制结果：";
    for (int n : partial_copy) std::cout << n << " ";
    std::cout << std::endl;
    
    // 4. move: 移动语义（C++11）
    std::vector<std::string> strings = {"hello", "world", "cpp"};
    std::vector<std::string> moved_strings(3);
    std::move(strings.begin(), strings.end(), moved_strings.begin());
    
    std::cout << "移动后源字符串：";
    for (const auto& s : strings) std::cout << "\"" << s << "\" ";
    std::cout << std::endl;
    
    std::cout << "移动后目标字符串：";
    for (const auto& s : moved_strings) std::cout << "\"" << s << "\" ";
    std::cout << std::endl;
}
```

#### 变换与替换算法

```cpp
void demonstrate_transform_replace_algorithms() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 2, 7, 2};
    
    // 1. transform: 对每个元素应用函数
    std::vector<int> squared;
    std::transform(numbers.begin(), numbers.end(),
                   std::back_inserter(squared),
                   [](int n) { return n * n; });
    std::cout << "平方变换：";
    for (int n : squared) std::cout << n << " ";
    std::cout << std::endl;
    
    // 2. replace: 替换特定值
    std::vector<int> replaced = numbers;
    std::replace(replaced.begin(), replaced.end(), 2, 99);
    std::cout << "替换2为99：";
    for (int n : replaced) std::cout << n << " ";
    std::cout << std::endl;
    
    // 3. replace_if: 条件替换
    std::vector<int> condition_replaced = numbers;
    std::replace_if(condition_replaced.begin(), condition_replaced.end(),
                    [](int n) { return n % 2 == 0; }, 0);
    std::cout << "替换偶数为0：";
    for (int n : condition_replaced) std::cout << n << " ";
    std::cout << std::endl;
    
    // 4. replace_copy: 替换并复制到新容器
    std::vector<int> replace_copy_result;
    std::replace_copy(numbers.begin(), numbers.end(),
                      std::back_inserter(replace_copy_result),
                      2, 88);
    std::cout << "替换复制结果：";
    for (int n : replace_copy_result) std::cout << n << " ";
    std::cout << std::endl;
}
```

### 3. 排序与相关操作

#### 排序算法

```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <random>

void demonstrate_sorting_algorithms() {
    // 生成随机数据
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::vector<std::string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    
    // 1. sort: 默认排序（升序）
    std::vector<int> sorted_numbers = numbers;
    std::sort(sorted_numbers.begin(), sorted_numbers.end());
    std::cout << "升序排序：";
    for (int n : sorted_numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    // 2. 自定义比较函数排序
    std::vector<int> descending = numbers;
    std::sort(descending.begin(), descending.end(),
              [](int a, int b) { return a > b; });
    std::cout << "降序排序：";
    for (int n : descending) std::cout << n << " ";
    std::cout << std::endl;
    
    // 3. stable_sort: 稳定排序（保持相等元素的相对顺序）
    std::vector<std::pair<int, char>> pairs = {
        {3, 'a'}, {1, 'b'}, {2, 'c'}, {3, 'd'}, {1, 'e'}
    };
    
    std::stable_sort(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) {
                        return a.first < b.first;
                    });
    
    std::cout << "稳定排序结果：";
    for (const auto& p : pairs) {
        std::cout << "(" << p.first << "," << p.second << ") ";
    }
    std::cout << std::endl;
    
    // 4. partial_sort: 部分排序
    std::vector<int> partial = numbers;
    std::partial_sort(partial.begin(), partial.begin() + 3, partial.end());
    std::cout << "部分排序（前3个最小）：";
    for (int n : partial) std::cout << n << " ";
    std::cout << std::endl;
    
    // 5. nth_element: 找到第n小的元素
    std::vector<int> nth = numbers;
    auto middle = nth.begin() + nth.size() / 2;
    std::nth_element(nth.begin(), middle, nth.end());
    std::cout << "中位数：" << *middle << std::endl;
}
```

#### 二分查找算法（要求有序序列）

```cpp
void demonstrate_binary_search() {
    std::vector<int> sorted = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. binary_search: 检查元素是否存在
    bool found = std::binary_search(sorted.begin(), sorted.end(), 5);
    std::cout << "5是否存在：" << std::boolalpha << found << std::endl;
    
    // 2. lower_bound: 找到第一个不小于给定值的元素
    auto lower = std::lower_bound(sorted.begin(), sorted.end(), 4);
    if (lower != sorted.end()) {
        std::cout << "第一个不小于4的元素：" << *lower 
                  << "，位置：" << std::distance(sorted.begin(), lower) << std::endl;
    }
    
    // 3. upper_bound: 找到第一个大于给定值的元素
    auto upper = std::upper_bound(sorted.begin(), sorted.end(), 4);
    if (upper != sorted.end()) {
        std::cout << "第一个大于4的元素：" << *upper 
                  << "，位置：" << std::distance(sorted.begin(), upper) << std::endl;
    }
    
    // 4. equal_range: 返回等于给定值的范围
    auto range = std::equal_range(sorted.begin(), sorted.end(), 5);
    std::cout << "等于5的范围：[" 
              << std::distance(sorted.begin(), range.first) << ", "
              << std::distance(sorted.begin(), range.second) << ")" << std::endl;
}
```

### 4. 数值算法

```cpp
#include <numeric>
#include <vector>
#include <iostream>

void demonstrate_numeric_algorithms() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // 1. accumulate: 累加
    int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    std::cout << "累加和：" << sum << std::endl;
    
    // 自定义累加操作
    int product = std::accumulate(numbers.begin(), numbers.end(), 1,
                                 [](int a, int b) { return a * b; });
    std::cout << "累乘积：" << product << std::endl;
    
    // 2. inner_product: 内积计算
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    int dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0);
    std::cout << "向量内积：" << dot_product << std::endl;
    
    // 3. partial_sum: 部分和
    std::vector<int> partial_sums;
    std::partial_sum(numbers.begin(), numbers.end(),
                    std::back_inserter(partial_sums));
    std::cout << "部分和：";
    for (int n : partial_sums) std::cout << n << " ";
    std::cout << std::endl;
    
    // 4. adjacent_difference: 相邻差值
    std::vector<int> differences;
    std::adjacent_difference(numbers.begin(), numbers.end(),
                            std::back_inserter(differences));
    std::cout << "相邻差值：";
    for (int n : differences) std::cout << n << " ";
    std::cout << std::endl;
    
    // 5. iota: 生成递增序列（C++11）
    std::vector<int> sequence(10);
    std::iota(sequence.begin(), sequence.end(), 1);
    std::cout << "递增序列：";
    for (int n : sequence) std::cout << n << " ";
    std::cout << std::endl;
}
```

## 算法性能分析与复杂度

### 时间复杂度对比

| 算法类别 | 典型算法 | 平均时间复杂度 | 最坏情况 | 适用场景 |
|----------|----------|----------------|----------|----------|
| **查找算法** | `find`, `count` | O(n) | O(n) | 无序序列查找 |
| **排序算法** | `sort` | O(n log n) | O(n log n) | 通用排序 |
| **稳定排序** | `stable_sort` | O(n log n) | O(n log n) | 需要保持顺序 |
| **部分排序** | `partial_sort` | O(n log k) | O(n log k) | 只关心前k个 |
| **二分查找** | `binary_search` | O(log n) | O(log n) | 有序序列查找 |
| **堆操作** | `make_heap` | O(n) | O(n) | 优先级队列 |
| **集合操作** | `set_union` | O(n+m) | O(n+m) | 集合运算 |

### 空间复杂度分析

- **原地算法**：`sort`, `reverse`, `rotate` - O(1) 额外空间
- **需要额外空间**：`stable_sort` - O(n) 额外空间
- **复制算法**：`copy`, `transform` - O(n) 额外空间

## 实际工程应用场景

### 场景1：数据处理管道

```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <ranges>  // C++20

class DataProcessor {
private:
    std::vector<int> data;
    
public:
    void add_data(int value) {
        data.push_back(value);
    }
    
    // 数据处理管道：过滤 → 转换 → 聚合
    struct ProcessResult {
        double average;
        int max_value;
        int min_value;
        std::vector<int> processed_data;
    };
    
    ProcessResult process_data() {
        ProcessResult result;
        
        // 1. 过滤：移除异常值
        std::vector<int> filtered;
        std::copy_if(data.begin(), data.end(),
                    std::back_inserter(filtered),
                    [](int x) { return x > 0 && x < 1000; });
        
        // 2. 转换：数据标准化
        std::vector<int> normalized;
        std::transform(filtered.begin(), filtered.end(),
                      std::back_inserter(normalized),
                      [](int x) { return x * 10; });
        
        // 3. 排序
        std::sort(normalized.begin(), normalized.end());
        
        // 4. 聚合统计
        if (!normalized.empty()) {
            result.average = std::accumulate(normalized.begin(), 
                                           normalized.end(), 0.0) / normalized.size();
            result.max_value = *std::max_element(normalized.begin(), normalized.end());
            result.min_value = *std::min_element(normalized.begin(), normalized.end());
            result.processed_data = std::move(normalized);
        }
        
        return result;
    }
    
    // C++20 范围视图（更简洁的写法）
    auto get_even_numbers() {
        return data | std::views::filter([](int x) { return x % 2 == 0; })
                   | std::views::transform([](int x) { return x * 2; });
    }
};
```

### 场景2：缓存系统实现

```cpp
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <list>

template<typename Key, typename Value>
class LRUCache {
private:
    size_t capacity_;
    std::list<std::pair<Key, Value>> cache_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> cache_map_;
    
public:
    LRUCache(size_t capacity) : capacity_(capacity) {}
    
    Value* get(const Key& key) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return nullptr;
        }
        
        // 移动到链表头部（最近使用）
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return &(it->second->second);
    }
    
    void put(const Key& key, const Value& value) {
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // 键已存在，更新值并移动到头部
            it->second->second = value;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }
        
        // 检查容量，如果满了则移除最久未使用的
        if (cache_map_.size() >= capacity_) {
            auto last = cache_list_.back();
            cache_map_.erase(last.first);
            cache_list_.pop_back();
        }
        
        // 插入新元素到头部
        cache_list_.emplace_front(key, value);
        cache_map_[key] = cache_list_.begin();
    }
    
    // 获取所有键（按使用顺序）
    std::vector<Key> get_keys() const {
        std::vector<Key> keys;
        std::transform(cache_list_.begin(), cache_list_.end(),
                      std::back_inserter(keys),
                      [](const auto& pair) { return pair.first; });
        return keys;
    }
};
```

### 场景3：算法组合解决复杂问题

```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <set>

class ProblemSolver {
public:
    // 问题：找出两个向量的交集，去重并排序
    static std::vector<int> find_sorted_intersection(
        const std::vector<int>& vec1, 
        const std::vector<int>& vec2) {
        
        std::vector<int> result;
        
        // 1. 对两个向量排序
        std::vector<int> sorted1 = vec1;
        std::vector<int> sorted2 = vec2;
        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());
        
        // 2. 使用set_intersection求交集
        std::set_intersection(sorted1.begin(), sorted1.end(),
                             sorted2.begin(), sorted2.end(),
                             std::back_inserter(result));
        
        // 3. 使用unique去重
        auto last = std::unique(result.begin(), result.end());
        result.erase(last, result.end());
        
        return result;
    }
    
    // 问题：统计文本中单词频率，按频率降序排列
    static std::vector<std::pair<std::string, int>> 
    word_frequency_analysis(const std::vector<std::string>& words) {
        
        std::unordered_map<std::string, int> freq_map;
        
        // 1. 统计频率
        for (const auto& word : words) {
            ++freq_map[word];
        }
        
        // 2. 转换为向量便于排序
        std::vector<std::pair<std::string, int>> freq_vec;
        std::copy(freq_map.begin(), freq_map.end(),
                 std::back_inserter(freq_vec));
        
        // 3. 按频率降序排序
        std::sort(freq_vec.begin(), freq_vec.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        return freq_vec;
    }
};
```

## 最佳实践与性能优化

### 1. 选择合适的算法

```cpp
// 不好的做法：手动实现查找
bool manual_find(const std::vector<int>& vec, int target) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == target) return true;
    }
    return false;
}

// 好的做法：使用标准算法
bool good_find(const std::vector<int>& vec, int target) {
    return std::find(vec.begin(), vec.end(), target) != vec.end();
}
```

### 2. 避免不必要的拷贝

```cpp
// 不好的做法：多次拷贝
std::vector<int> process_data_bad(const std::vector<int>& data) {
    std::vector<int> temp = data;  // 第一次拷贝
    std::sort(temp.begin(), temp.end());
    
    std::vector<int> result;        // 第二次拷贝
    std::copy_if(temp.begin(), temp.end(),
                 std::back_inserter(result),
                 [](int x) { return x > 0; });
    return result;
}

// 好的做法：使用移动语义和视图
std::vector<int> process_data_good(std::vector<int> data) {  // 按值传递
    std::sort(data.begin(), data.end());  // 原地排序
    
    // 使用erase-remove惯用法，避免额外拷贝
    data.erase(std::remove_if(data.begin(), data.end(),
                            [](int x) { return x <= 0; }),
               data.end());
    
    return data;  // 移动返回
}
```

### 3. 利用并行算法（C++17）

```cpp
#include <execution>  // C++17 并行执行策略

void demonstrate_parallel_algorithms() {
    std::vector<int> large_data(1000000);
    std::iota(large_data.begin(), large_data.end(), 1);
    
    // 顺序执行
    auto start_seq = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::seq, large_data.begin(), large_data.end());
    auto end_seq = std::chrono::high_resolution_clock::now();
    
    // 并行执行
    auto start_par = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par, large_data.begin(), large_data.end());
    auto end_par = std::chrono::high_resolution_clock::now();
    
    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
    auto par_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);
    
    std::cout << "顺序执行时间：" << seq_time.count() << "ms" << std::endl;
    std::cout << "并行执行时间：" << par_time.count() << "ms" << std::endl;
    std::cout << "加速比：" << static_cast<double>(seq_time.count()) / par_time.count() << std::endl;
}
```

## 现代C++特性集成

### C++11/14/17/20 新特性

```cpp
#include <algorithm>
#include <vector>
#include <ranges>

void demonstrate_modern_features() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // C++11: lambda表达式
    auto even_count = std::count_if(data.begin(), data.end(),
                                   [](int x) { return x % 2 == 0; });
    
    // C++14: 泛型lambda
    auto generic_transform = [](auto container, auto func) {
        std::vector<decltype(func(container[0]))> result;
        std::transform(container.begin(), container.end(),
                      std::back_inserter(result), func);
        return result;
    };
    
    // C++17: 并行算法和结构化绑定
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](int& x) { x *= 2; });
    
    // C++20: 范围库和概念
    auto even_squares = data | std::views::filter([](int x) { return x % 2 == 0; })
                            | std::views::transform([](int x) { return x * x; });
    
    std::cout << "偶数平方：";
    for (int n : even_squares) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
}
```

## 总结

C++算法库是现代C++编程不可或缺的工具，它提供了：

1. **丰富的算法集合**：覆盖了从基本操作到复杂算法的各种需求
2. **卓越的性能**：经过严格优化，提供最佳的时间/空间复杂度
3. **类型安全**：编译时检查确保代码安全性
4. **现代特性支持**：完美集成lambda、移动语义、并行计算等特性
5. **高可维护性**：标准化的接口和清晰的语义

通过熟练掌握这些算法，开发者可以编写出更简洁、高效和可维护的C++代码，显著提升开发效率和程序性能。