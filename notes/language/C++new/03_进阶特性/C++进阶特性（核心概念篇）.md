# C++进阶特性（核心概念篇）

## 概述
本文件整合了C++进阶特性中的核心概念模块，包括STL标准库、容器、迭代器与算法、Lambda表达式等关键特性，为深入学习C++高级编程打下坚实基础。

---

## 1. STL标准库

### 核心概念
- **STL组成**：容器、算法、迭代器、函数对象、适配器、分配器
- **设计原则**：通用性、效率、可扩展性
- **模板编程**：基于泛型编程思想

### 关键组件
```cpp
#include <vector>
#include <algorithm>
#include <functional>

// 常用STL组件
vector<int> vec = {1, 2, 3};
sort(vec.begin(), vec.end());
for_each(vec.begin(), vec.end(), [](int x) { cout << x; });
```

### 特性优势
- **类型安全**：编译时类型检查
- **性能优化**：内联展开，消除虚函数调用
- **代码复用**：一套算法适用于多种容器

---

## 2. 容器（vector, map, set等）

### 容器分类

#### 序列容器
- **vector**：动态数组，随机访问高效
- **list**：双向链表，插入删除高效
- **deque**：双端队列，首尾操作高效

#### 关联容器
- **set/multiset**：有序集合，基于红黑树
- **map/multimap**：键值对映射，快速查找

#### 无序容器（C++11）
- **unordered_set**：哈希集合，平均O(1)访问
- **unordered_map**：哈希映射，基于哈希表

### 容器选择指南
```cpp
// 根据需求选择合适容器
vector<int> v;          // 需要随机访问
list<int> l;            // 频繁插入删除
map<string, int> m;     // 需要键值映射
set<int> s;             // 需要唯一元素集合
```

### 性能对比
- **插入性能**：list最优，vector最差（中间插入）
- **查找性能**：map/set O(log n)，unordered_map O(1)
- **内存使用**：vector最紧凑，list额外指针开销

---

## 3. 迭代器与算法

### 迭代器类型

#### 输入迭代器
- 只能读取，单向移动
- 典型应用：istream_iterator

#### 输出迭代器
- 只能写入，单向移动
- 典型应用：ostream_iterator

#### 前向迭代器
- 可读写，单向移动
- 典型应用：forward_list

#### 双向迭代器
- 可读写，双向移动
- 典型应用：list, set, map

#### 随机访问迭代器
- 可读写，随机访问
- 典型应用：vector, deque, array

### STL算法分类

#### 非修改序列算法
```cpp
#include <algorithm>

vector<int> v = {1, 2, 3, 4, 5};

// 查找
auto it = find(v.begin(), v.end(), 3);

// 计数
int count = count_if(v.begin(), v.end(), 
                     [](int x) { return x % 2 == 0; });

// 判断
bool allEven = all_of(v.begin(), v.end(), 
                      [](int x) { return x % 2 == 0; });
```

#### 修改序列算法
```cpp
// 复制
vector<int> dest(5);
copy(v.begin(), v.end(), dest.begin());

// 变换
transform(v.begin(), v.end(), dest.begin(),
          [](int x) { return x * 2; });

// 替换
replace(v.begin(), v.end(), 3, 30);
```

#### 排序和搜索算法
```cpp
// 排序
sort(v.begin(), v.end());

// 二分查找
bool found = binary_search(v.begin(), v.end(), 3);

// 堆操作
make_heap(v.begin(), v.end());
pop_heap(v.begin(), v.end());
```

### 自定义算法示例
```cpp
template<typename InputIt, typename UnaryPredicate>
int my_count_if(InputIt first, InputIt last, UnaryPredicate p) {
    int count = 0;
    for (; first != last; ++first) {
        if (p(*first)) {
            ++count;
        }
    }
    return count;
}
```

---

## 4. Lambda表达式

### 基本语法
```cpp
// 基本形式
[](参数) -> 返回类型 { 函数体 }

// 示例
auto add = [](int a, int b) -> int { return a + b; };
auto print = [](int x) { cout << x; };
```

### 捕获模式

#### 值捕获
```cpp
int x = 10;
auto func = [x]() { return x + 1; };  // 捕获x的副本
```

#### 引用捕获
```cpp
int y = 20;
auto func = [&y]() { y += 1; };  // 捕获y的引用
```

#### 混合捕获
```cpp
int a = 1, b = 2;
auto func = [a, &b]() { return a + b; };  // a值捕获，b引用捕获
```

#### 隐式捕获
```cpp
int x = 10, y = 20;
// 值捕获所有外部变量
auto func1 = [=]() { return x + y; };  

// 引用捕获所有外部变量  
auto func2 = [&]() { x++; y++; };

// 混合隐式捕获
auto func3 = [=, &y]() { return x + y; };  // x值捕获，y引用捕获
auto func4 = [&, x]() { return x + y; };  // x值捕获，其他引用捕获
```

### Lambda与STL算法结合
```cpp
vector<int> numbers = {1, 2, 3, 4, 5};

// 过滤偶数
vector<int> evens;
copy_if(numbers.begin(), numbers.end(), back_inserter(evens),
        [](int n) { return n % 2 == 0; });

// 排序自定义规则
sort(numbers.begin(), numbers.end(),
     [](int a, int b) { return a > b; });  // 降序排序

// 累积计算
int sum = accumulate(numbers.begin(), numbers.end(), 0,
                    [](int total, int n) { return total + n; });
```

### 通用Lambda（C++14）
```cpp
// 自动推导参数类型
auto adder = [](auto a, auto b) { return a + b; };

cout << adder(1, 2) << endl;        // 3
cout << adder(1.5, 2.5) << endl;    // 4.0
cout << adder(string("hello"), string(" world")) << endl; // "hello world"
```

### 可变Lambda
```cpp
int counter = 0;

// mutable允许修改值捕获的变量
auto incrementer = [counter]() mutable {
    return ++counter;  // 修改副本，不影响外部counter
};

cout << incrementer() << endl;  // 1
cout << incrementer() << endl;  // 2
cout << "外部counter: " << counter << endl;  // 0（不变）
```

### Lambda表达式应用场景

#### 回调函数
```cpp
class Button {
    function<void()> onClick;
public:
    void setOnClick(function<void()> callback) {
        onClick = callback;
    }
    
    void click() {
        if (onClick) onClick();
    }
};

Button btn;
btn.setOnClick([]() {
    cout << "按钮被点击!" << endl;
});
```

#### 异步编程
```cpp
#include <future>

auto future = async(launch::async, []() {
    this_thread::sleep_for(chrono::seconds(1));
    return 42;
});

// 主线程继续执行其他任务
int result = future.get();  // 等待结果
```

#### 函数式编程风格
```cpp
vector<int> processNumbers(const vector<int>& nums) {
    vector<int> result;
    
    // 函数式编程链式操作
    transform(nums.begin(), nums.end(), back_inserter(result),
        [](int n) { return n * 2; });  // 映射：乘以2
    
    result.erase(remove_if(result.begin(), result.end(),
        [](int n) { return n % 3 == 0; }), result.end());  // 过滤：移除3的倍数
    
    sort(result.begin(), result.end(),
        [](int a, int b) { return a > b; });  // 排序：降序
    
    return result;
}
```

### Lambda表达式优势
- **简洁性**：减少函数定义代码
- **封装性**：捕获上下文变量
- **灵活性**：支持多种捕获模式
- **性能**：内联优化，消除函数调用开销

---

## 5. 右值引用与移动语义

### 核心概念
- **右值引用**：绑定到临时对象的引用类型
- **移动语义**：高效转移资源所有权
- **完美转发**：保持参数值类别进行转发

### 基本语法
```cpp
// 右值引用
类型&& 引用名 = 右值表达式;

// 移动构造和赋值
类名(类名&& other);
类名& operator=(类名&& other);

// 完美转发
template<typename T>
void func(T&& arg);
```

### 移动语义优势
- **性能提升**：避免不必要的深拷贝
- **资源管理**：明确所有权转移语义
- **标准库支持**：STL全面支持移动语义

---

## 总结

本核心概念篇涵盖了C++进阶特性的基础模块，为后续的综合应用和实践打下坚实基础。掌握这些核心概念后，可以更高效地使用C++进行复杂程序开发。