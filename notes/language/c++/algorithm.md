# C++ STL 算法库

STL 算法库提供了丰富的通用算法，通过迭代器将算法与容器解耦，实现"一次编写，处处使用"的设计理念。

## 算法库概述

### 设计哲学

- **泛型编程**：算法独立于数据类型和容器
- **迭代器抽象**：通过迭代器统一访问不同容器
- **函数对象支持**：可自定义比较和操作逻辑
- **无副作用**：大多数算法不修改原始数据

### 头文件

```cpp
#include <algorithm>   // 大多数算法
#include <numeric>     // 数值算法
#include <execution>   // 并行执行策略（C++17）
```

## 非修改序列操作

### 查找算法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // find: 查找特定元素
    auto it = std::find(nums.begin(), nums.end(), 5);
    if (it != nums.end()) {
        std::cout << "找到5，位置: " << std::distance(nums.begin(), it) << std::endl;
    }
    
    // find_if: 查找满足条件的第一个元素
    auto even = std::find_if(nums.begin(), nums.end(), 
                             [](int n) { return n % 2 == 0; });
    if (even != nums.end()) {
        std::cout << "第一个偶数: " << *even << std::endl;  // 2
    }
    
    // find_if_not: 查找不满足条件的第一个元素
    auto not_even = std::find_if_not(nums.begin(), nums.end(),
                                     [](int n) { return n % 2 == 0; });
    
    // find_end: 查找最后一个子序列
    std::vector<int> sub = {3, 4};
    auto last_sub = std::find_end(nums.begin(), nums.end(), sub.begin(), sub.end());
    
    // find_first_of: 查找第一个匹配任意元素的元素
    std::vector<int> targets = {3, 7, 11};
    auto first_match = std::find_first_of(nums.begin(), nums.end(),
                                          targets.begin(), targets.end());
    
    // adjacent_find: 查找相邻重复元素
    std::vector<int> dup = {1, 2, 2, 3, 4, 4, 5};
    auto adj = std::adjacent_find(dup.begin(), dup.end());
    if (adj != dup.end()) {
        std::cout << "相邻重复: " << *adj << std::endl;  // 2
    }
    
    return 0;
}
```

### 统计与检查算法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>

int main() {
    std::vector<int> nums = {1, 2, 3, 2, 4, 2, 5};
    
    // count: 统计元素出现次数
    int count_2 = std::count(nums.begin(), nums.end(), 2);
    std::cout << "2出现次数: " << count_2 << std::endl;  // 3
    
    // count_if: 统计满足条件的元素个数
    int even_count = std::count_if(nums.begin(), nums.end(),
                                   [](int n) { return n % 2 == 0; });
    std::cout << "偶数个数: " << even_count << std::endl;
    
    // all_of: 检查是否所有元素都满足条件
    bool all_positive = std::all_of(nums.begin(), nums.end(),
                                    [](int n) { return n > 0; });
    std::cout << "所有元素都是正数: " << std::boolalpha << all_positive << std::endl;
    
    // any_of: 检查是否存在元素满足条件
    bool has_even = std::any_of(nums.begin(), nums.end(),
                                [](int n) { return n % 2 == 0; });
    
    // none_of: 检查是否所有元素都不满足条件
    bool no_zero = std::none_of(nums.begin(), nums.end(),
                                [](int n) { return n == 0; });
    
    return 0;
}
```

### 搜索算法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> text = {1, 2, 3, 4, 5, 3, 4, 5, 6};
    std::vector<int> pattern = {3, 4, 5};
    
    // search: 搜索子序列
    auto it = std::search(text.begin(), text.end(), pattern.begin(), pattern.end());
    if (it != text.end()) {
        std::cout << "找到子序列，位置: " << std::distance(text.begin(), it) << std::endl;
    }
    
    // search_n: 搜索连续n个相同元素
    std::vector<int> repeated = {1, 2, 2, 2, 3, 4};
    auto two_twos = std::search_n(repeated.begin(), repeated.end(), 3, 2);
    if (two_twos != repeated.end()) {
        std::cout << "找到3个连续的2" << std::endl;
    }
    
    return 0;
}
```

## 修改序列操作

### 复制与移动

```cpp
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>

int main() {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);
    
    // copy: 复制元素
    std::copy(src.begin(), src.end(), dst.begin());
    
    // copy_if: 条件复制
    std::vector<int> evens;
    std::copy_if(src.begin(), src.end(),
                 std::back_inserter(evens),
                 [](int n) { return n % 2 == 0; });
    
    // copy_n: 复制前n个元素
    std::vector<int> first_3(3);
    std::copy_n(src.begin(), 3, first_3.begin());
    
    // copy_backward: 向后复制（避免覆盖）
    std::vector<int> large = {1, 2, 3, 4, 5, 6, 7};
    std::copy_backward(large.begin(), large.begin() + 3, large.begin() + 5);
    
    // move: 移动元素
    std::vector<std::string> strings = {"hello", "world"};
    std::vector<std::string> moved(2);
    std::move(strings.begin(), strings.end(), moved.begin());
    // strings 中的元素现在是有效但未定义状态
    
    // move_backward: 向后移动
    std::move_backward(large.begin(), large.begin() + 3, large.begin() + 5);
    
    return 0;
}
```

### 变换与替换

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include <cctype>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    
    // transform: 对每个元素应用函数
    std::vector<int> squares;
    std::transform(nums.begin(), nums.end(),
                   std::back_inserter(squares),
                   [](int n) { return n * n; });
    // squares: 1, 4, 9, 16, 25
    
    // transform: 合并两个序列
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    std::vector<int> sum;
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(sum),
                   std::plus<int>());
    // sum: 5, 7, 9
    
    // replace: 替换所有特定值
    std::vector<int> replaced = {1, 2, 2, 3, 2};
    std::replace(replaced.begin(), replaced.end(), 2, 99);
    // replaced: 1, 99, 99, 3, 99
    
    // replace_if: 条件替换
    std::vector<int> cond_replace = {1, 2, 3, 4, 5, 6};
    std::replace_if(cond_replace.begin(), cond_replace.end(),
                    [](int n) { return n % 2 == 0; }, 0);
    // cond_replace: 1, 0, 3, 0, 5, 0
    
    // replace_copy: 替换并复制
    std::vector<int> replaced_copy;
    std::replace_copy(replaced.begin(), replaced.end(),
                      std::back_inserter(replaced_copy), 99, 0);
    
    return 0;
}
```

### 填充与生成

```cpp
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>

int main() {
    std::vector<int> v(10);
    
    // fill: 填充相同值
    std::fill(v.begin(), v.end(), 42);
    
    // fill_n: 填充前n个
    std::fill_n(v.begin(), 5, 0);
    
    // generate: 使用函数生成
    int counter = 0;
    std::generate(v.begin(), v.end(), [&counter]() { return counter++; });
    // v: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    // generate_n: 生成前n个
    std::generate_n(v.begin(), 5, []() { return 100; });
    
    // iota: 生成递增序列（C++11）
    std::iota(v.begin(), v.end(), 1);  // 1, 2, 3, ..., 10
    
    return 0;
}
```

### 删除与去重

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 2, 4, 2, 5};
    
    // remove: 移动元素，不删除，返回新结尾
    auto new_end = std::remove(nums.begin(), nums.end(), 2);
    // nums 变为: 1, 3, 4, 5, ?, ?, ?
    // new_end 指向第一个 "?" 位置
    
    // erase-remove 惯用法：真正删除
    nums.erase(new_end, nums.end());
    // nums 变为: 1, 3, 4, 5
    
    // remove_if: 条件删除
    std::vector<int> nums2 = {1, -2, 3, -4, 5};
    nums2.erase(std::remove_if(nums2.begin(), nums2.end(),
                               [](int n) { return n < 0; }),
                nums2.end());
    
    // unique: 去除连续重复元素
    std::vector<int> dup = {1, 1, 2, 2, 2, 3, 3, 4};
    auto last = std::unique(dup.begin(), dup.end());
    dup.erase(last, dup.end());
    // dup: 1, 2, 3, 4
    
    // unique 带条件
    std::vector<int> case_dup = {1, 1, 2, 2, 3};
    last = std::unique(case_dup.begin(), case_dup.end(),
                       [](int a, int b) { return a == b; });
    
    return 0;
}
```

### 反转与旋转

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    // reverse: 反转
    std::reverse(v.begin(), v.end());
    // v: 5, 4, 3, 2, 1
    
    // reverse_copy: 反转并复制
    std::vector<int> reversed(v.size());
    std::reverse_copy(v.begin(), v.end(), reversed.begin());
    
    // rotate: 旋转（中间元素成为第一个）
    std::vector<int> rot = {1, 2, 3, 4, 5};
    std::rotate(rot.begin(), rot.begin() + 2, rot.end());
    // rot: 3, 4, 5, 1, 2
    
    // rotate_copy: 旋转并复制
    std::vector<int> rotated(rot.size());
    std::rotate_copy(rot.begin(), rot.begin() + 2, rot.end(), rotated.begin());
    
    // shift_left / shift_right (C++23)
    // std::shift_left(v.begin(), v.end(), 2);
    
    return 0;
}
```

## 排序与相关操作

### 排序算法

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>

int main() {
    std::vector<int> nums = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // sort: 排序（默认升序）
    std::sort(nums.begin(), nums.end());
    // nums: 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    // 自定义比较（降序）
    std::sort(nums.begin(), nums.end(), std::greater<int>());
    // nums: 9, 8, 7, 6, 5, 4, 3, 2, 1
    
    // 使用 lambda 排序
    std::vector<std::string> words = {"apple", "pie", "a", "longer"};
    std::sort(words.begin(), words.end(),
              [](const std::string& a, const std::string& b) {
                  return a.length() < b.length();  // 按长度排序
              });
    
    // stable_sort: 稳定排序（保持相等元素的相对顺序）
    std::vector<std::pair<int, char>> pairs = {
        {3, 'a'}, {1, 'b'}, {2, 'c'}, {3, 'd'}
    };
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto& a, const auto& b) {
                         return a.first < b.first;
                     });
    // pairs: {1,b}, {2,c}, {3,a}, {3,d}  -- {3,a} 在 {3,d} 前面
    
    // partial_sort: 部分排序（只排序前k个）
    std::vector<int> partial = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::partial_sort(partial.begin(), partial.begin() + 3, partial.end());
    // partial: 1, 2, 3, ?, ?, ?, ?, ?, ?  （前3个最小，其余无序）
    
    // nth_element: 找第n小的元素
    std::vector<int> nth = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    auto mid = nth.begin() + nth.size() / 2;
    std::nth_element(nth.begin(), mid, nth.end());
    std::cout << "中位数: " << *mid << std::endl;  // 第5小的数
    
    // is_sorted: 检查是否已排序
    bool sorted = std::is_sorted(nums.begin(), nums.end());
    
    return 0;
}
```

### 二分查找（要求有序序列）

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> sorted = {1, 2, 3, 4, 5, 5, 5, 6, 7, 8};
    
    // binary_search: 检查元素是否存在
    bool found = std::binary_search(sorted.begin(), sorted.end(), 5);
    std::cout << "5存在: " << std::boolalpha << found << std::endl;
    
    // lower_bound: 第一个 >= value 的位置
    auto lower = std::lower_bound(sorted.begin(), sorted.end(), 5);
    std::cout << "第一个>=5的位置: " << std::distance(sorted.begin(), lower) << std::endl;  // 4
    
    // upper_bound: 第一个 > value 的位置
    auto upper = std::upper_bound(sorted.begin(), sorted.end(), 5);
    std::cout << "第一个>5的位置: " << std::distance(sorted.begin(), upper) << std::endl;   // 7
    
    // equal_range: 返回等于value的范围
    auto range = std::equal_range(sorted.begin(), sorted.end(), 5);
    std::cout << "等于5的元素个数: " << std::distance(range.first, range.second) << std::endl;  // 3
    
    // 在有序序列中插入元素（保持有序）
    std::vector<int> data = {1, 3, 5, 7, 9};
    int value = 4;
    auto insert_pos = std::lower_bound(data.begin(), data.end(), value);
    data.insert(insert_pos, value);
    // data: 1, 3, 4, 5, 7, 9
    
    return 0;
}
```

### 合并与分区

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    // merge: 合并两个有序序列
    std::vector<int> a = {1, 3, 5};
    std::vector<int> b = {2, 4, 6};
    std::vector<int> merged(a.size() + b.size());
    std::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin());
    // merged: 1, 2, 3, 4, 5, 6
    
    // inplace_merge: 原地合并
    std::vector<int> inplace = {1, 3, 5, 2, 4, 6};
    std::inplace_merge(inplace.begin(), inplace.begin() + 3, inplace.end());
    // inplace: 1, 2, 3, 4, 5, 6
    
    // partition: 分区（满足条件的在前）
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
    auto partition_point = std::partition(nums.begin(), nums.end(),
                                          [](int n) { return n % 2 == 0; });
    // nums: 偶数在前，奇数在后（顺序可能改变）
    std::cout << "分区点位置: " << std::distance(nums.begin(), partition_point) << std::endl;
    
    // stable_partition: 稳定分区（保持相对顺序）
    nums = {1, 2, 3, 4, 5, 6, 7, 8};
    partition_point = std::stable_partition(nums.begin(), nums.end(),
                                            [](int n) { return n % 2 == 0; });
    // nums: 2, 4, 6, 8, 1, 3, 5, 7
    
    // partition_copy: 分区并复制
    std::vector<int> evens, odds;
    std::partition_copy(nums.begin(), nums.end(),
                        std::back_inserter(evens),
                        std::back_inserter(odds),
                        [](int n) { return n % 2 == 0; });
    
    // is_partitioned: 检查是否已分区
    bool is_part = std::is_partitioned(nums.begin(), nums.end(),
                                       [](int n) { return n % 2 == 0; });
    
    return 0;
}
```

## 排列与组合

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>

int main() {
    // next_permutation: 下一个排列
    std::string s = "abc";
    do {
        std::cout << s << std::endl;
    } while (std::next_permutation(s.begin(), s.end()));
    // 输出: abc, acb, bac, bca, cab, cba
    
    // prev_permutation: 上一个排列
    s = "cba";
    do {
        std::cout << s << std::endl;
    } while (std::prev_permutation(s.begin(), s.end()));
    
    // is_permutation: 检查是否为排列
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {5, 4, 3, 2, 1};
    bool is_perm = std::is_permutation(v1.begin(), v1.end(), v2.begin());
    std::cout << "是排列: " << std::boolalpha << is_perm << std::endl;
    
    return 0;
}
```

## 集合操作

```cpp
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>

int main() {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {4, 5, 6, 7, 8};
    
    // 并集
    std::vector<int> union_set;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(union_set));
    // union_set: 1, 2, 3, 4, 5, 6, 7, 8
    
    // 交集
    std::vector<int> intersection;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(intersection));
    // intersection: 4, 5
    
    // 差集 (a - b)
    std::vector<int> difference;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(difference));
    // difference: 1, 2, 3
    
    // 对称差集 ((a - b) ∪ (b - a))
    std::vector<int> sym_diff;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(sym_diff));
    // sym_diff: 1, 2, 3, 6, 7, 8
    
    // includes: 检查子集
    std::vector<int> subset = {2, 3};
    bool is_subset = std::includes(a.begin(), a.end(),
                                   subset.begin(), subset.end());
    std::cout << "是子集: " << std::boolalpha << is_subset << std::endl;
    
    return 0;
}
```

## 堆操作

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> heap = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // make_heap: 构建堆
    std::make_heap(heap.begin(), heap.end());
    // heap 是最大堆
    
    // push_heap: 插入元素
    heap.push_back(10);
    std::push_heap(heap.begin(), heap.end());
    
    // pop_heap: 弹出堆顶
    std::pop_heap(heap.begin(), heap.end());  // 堆顶移到末尾
    int max_val = heap.back();
    heap.pop_back();
    std::cout << "最大值: " << max_val << std::endl;
    
    // sort_heap: 堆排序
    std::sort_heap(heap.begin(), heap.end());
    // heap 现在是有序的（升序）
    
    // is_heap: 检查是否为堆
    bool is_heap = std::is_heap(heap.begin(), heap.end());
    
    return 0;
}
```

## 数值算法

```cpp
#include <numeric>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    
    // accumulate: 累加
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    std::cout << "和: " << sum << std::endl;  // 15
    
    // 自定义操作
    int product = std::accumulate(nums.begin(), nums.end(), 1,
                                  std::multiplies<int>());
    std::cout << "积: " << product << std::endl;  // 120
    
    // inner_product: 内积
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    int dot = std::inner_product(a.begin(), a.end(), b.begin(), 0);
    std::cout << "内积: " << dot << std::endl;  // 1*4 + 2*5 + 3*6 = 32
    
    // partial_sum: 部分和
    std::vector<int> partial;
    std::partial_sum(nums.begin(), nums.end(), std::back_inserter(partial));
    // partial: 1, 3, 6, 10, 15
    
    // adjacent_difference: 相邻差
    std::vector<int> diff;
    std::adjacent_difference(nums.begin(), nums.end(), std::back_inserter(diff));
    // diff: 1, 1, 1, 1, 1
    
    // iota: 填充递增序列
    std::vector<int> seq(10);
    std::iota(seq.begin(), seq.end(), 0);
    // seq: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    // C++17: reduce（可并行）
    // int sum2 = std::reduce(std::execution::par, nums.begin(), nums.end(), 0);
    
    return 0;
}
```

## 最小/最大值操作

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    // min/max
    int m = std::min(3, 5);
    int M = std::max(3, 5);
    auto [min_val, max_val] = std::minmax(3, 5);  // C++17
    
    // 自定义比较
    std::string s1 = "hello", s2 = "world";
    const auto& shorter = std::min(s1, s2,
        [](const std::string& a, const std::string& b) {
            return a.length() < b.length();
        });
    
    std::vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // min_element/max_element
    auto min_it = std::min_element(nums.begin(), nums.end());
    auto max_it = std::max_element(nums.begin(), nums.end());
    auto [min_elem, max_elem] = std::minmax_element(nums.begin(), nums.end());  // C++17
    
    std::cout << "最小值: " << *min_it << std::endl;   // 1
    std::cout << "最大值: " << *max_it << std::endl;   // 9
    
    // clamp: 限制范围 (C++17)
    int clamped = std::clamp(10, 0, 5);  // 结果为 5
    
    return 0;
}
```

## 未初始化内存操作

```cpp
#include <memory>
#include <algorithm>
#include <iostream>

int main() {
    // uninitialized_fill
    int* p1 = static_cast<int*>(::operator new(5 * sizeof(int)));
    std::uninitialized_fill(p1, p1 + 5, 42);
    // p1[0..4] 都是 42
    
    // uninitialized_copy
    int src[] = {1, 2, 3, 4, 5};
    int* p2 = static_cast<int*>(::operator new(5 * sizeof(int)));
    std::uninitialized_copy(std::begin(src), std::end(src), p2);
    
    // 释放内存（需要显式调用析构函数）
    for (int i = 0; i < 5; ++i) {
        (p1 + i)->~int();
        (p2 + i)->~int();
    }
    ::operator delete(p1);
    ::operator delete(p2);
    
    // C++17: uninitialized_move
    // std::uninitialized_move(src, src + 5, p2);
    
    return 0;
}
```

## 并行算法（C++17）

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

int main() {
    std::vector<int> large(10'000'000);
    std::iota(large.begin(), large.end(), 1);
    
    // 执行策略
    // std::execution::seq   - 顺序执行
    // std::execution::par   - 并行执行
    // std::execution::par_unseq - 并行向量化
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 并行排序
    std::sort(std::execution::par, large.begin(), large.end());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "并行排序时间: " << duration.count() << "ms" << std::endl;
    
    // 支持并行的主要算法
    // sort, stable_sort, for_each, transform, copy, fill
    // reduce, transform_reduce, count, any_of, all_of
    
    // transform_reduce（并行map-reduce）
    int sum_of_squares = std::transform_reduce(
        std::execution::par,
        large.begin(), large.end(),
        0,
        std::plus<int>(),
        [](int x) { return x * x; }
    );
    
    return 0;
}
```

## C++20 Ranges 库

```cpp
#include <ranges>
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 视图组合（惰性求值）
    auto result = nums 
        | std::views::filter([](int n) { return n % 2 == 0; })  // 过滤偶数
        | std::views::transform([](int n) { return n * n; });    // 平方
    
    std::cout << "偶数平方: ";
    for (int n : result) {
        std::cout << n << " ";  // 4 16 36 64 100
    }
    std::cout << std::endl;
    
    // 更多视图
    auto first_five = nums | std::views::take(5);      // 取前5个
    auto skip_three = nums | std::views::drop(3);      // 跳过前3个
    auto reversed = nums | std::views::reverse;        // 反转
    
    // 无限序列
    auto evens = std::views::iota(0)                   // 0, 1, 2, ...
               | std::views::filter([](int n) { return n % 2 == 0; })
               | std::views::take(5);                  // 取前5个偶数
    
    // ranges 算法
    std::ranges::sort(nums);
    auto found = std::ranges::find(nums, 5);
    
    // C++23: views::enumerate, views::zip, views::chunk
    // for (auto [index, value] : nums | std::views::enumerate) { ... }
    
    return 0;
}
```

## 算法复杂度汇总

| 算法类别 | 典型算法 | 时间复杂度 |
|----------|----------|------------|
| **查找** | `find`, `count` | O(n) |
| **二分查找** | `binary_search`, `lower_bound` | O(log n) |
| **排序** | `sort` | O(n log n) |
| **稳定排序** | `stable_sort` | O(n log² n) |
| **部分排序** | `partial_sort` | O(n log k) |
| **第k小** | `nth_element` | O(n) 平均 |
| **堆操作** | `make_heap`, `push_heap`, `pop_heap` | O(n), O(log n), O(log n) |
| **集合操作** | `set_union`, `set_intersection` | O(n+m) |

## 最佳实践

```cpp
#include <algorithm>
#include <vector>
#include <execution>
#include <iostream>

void best_practices() {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // 1. 选择正确的算法
    // 排序后只需一次查找 → 先排序，再用二分查找
    std::sort(data.begin(), data.end());
    bool found = std::binary_search(data.begin(), data.end(), 5);
    
    // 2. 使用 erase-remove 惯用法
    data.erase(std::remove(data.begin(), data.end(), 1), data.end());
    
    // 3. 利用 lambda 表达式
    std::sort(data.begin(), data.end(), std::greater<int>());
    
    // 4. 大数据量使用并行算法（C++17）
    std::sort(std::execution::par, data.begin(), data.end());
    
    // 5. 使用 C++20 ranges（更简洁）
    // auto evens = data | std::views::filter([](int n) { return n % 2 == 0; });
    
    // 6. 避免不必要的拷贝
    std::vector<std::string> strings = {"hello", "world"};
    std::vector<std::string> upper;
    upper.reserve(strings.size());
    std::transform(strings.begin(), strings.end(),
                   std::back_inserter(upper),
                   [](const std::string& s) {
                       std::string result = s;
                       std::transform(result.begin(), result.end(),
                                     result.begin(), ::toupper);
                       return result;
                   });
}
```

## 参考资料

- [cppreference - Algorithms library](https://en.cppreference.com/w/cpp/algorithm)
- 《Effective STL》- Scott Meyers
- 《C++ Primer》第10章
- [C++20 Ranges](https://en.cppreference.com/w/cpp/ranges)