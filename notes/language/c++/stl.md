# C++高级特性与实战

## 1. 标准模板库（STL）核心

### 1.1 容器：vector、map、set、unordered_map

**vector（动态数组）：**
```cpp
#include <vector>
#include <algorithm>

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};

// 添加元素
numbers.push_back(7);
numbers.insert(numbers.begin() + 2, 8);  // 在索引2处插入8

// 删除元素
numbers.pop_back();                      // 删除最后一个
numbers.erase(numbers.begin() + 1);     // 删除索引1处的元素

// 容量操作
std::cout << "大小: " << numbers.size() << std::endl;
std::cout << "容量: " << numbers.capacity() << std::endl;
numbers.shrink_to_fit();                // 缩减容量到实际大小

// 对比Python的list
# Python: numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# numbers.append(7)
# numbers.insert(2, 8)
# numbers.pop()
# del numbers[1]
```

**map（有序关联容器）：**
```cpp
#include <map>

std::map<std::string, int> age_map;

// 插入元素
age_map["Alice"] = 25;
age_map["Bob"] = 30;
age_map.insert({"Charlie", 35});

// 访问元素（自动排序）
for (const auto &pair : age_map) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}

// 查找元素
auto it = age_map.find("Alice");
if (it != age_map.end()) {
    std::cout << "找到Alice: " << it->second << std::endl;
}

// 对比Python的dict
# age_map = {"Alice": 25, "Bob": 30, "Charlie": 35}
# for key, value in age_map.items():
#     print(f"{key}: {value}")
```

**set（有序集合）：**
```cpp
#include <set>

std::set<int> unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6};

// 自动去重和排序
for (int num : unique_numbers) {
    std::cout << num << " ";  // 输出: 1 2 3 4 5 6 9
}
std::cout << std::endl;

// 集合操作
std::set<int> set1 = {1, 2, 3, 4, 5};
std::set<int> set2 = {4, 5, 6, 7, 8};

// 并集
std::set<int> union_set;
std::set_union(set1.begin(), set1.end(), 
               set2.begin(), set2.end(),
               std::inserter(union_set, union_set.begin()));

// 对比Python的set
# unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6}  # 自动去重
# set1 = {1, 2, 3, 4, 5}
# set2 = {4, 5, 6, 7, 8}
# union_set = set1 | set2
```

**unordered_map（哈希表）：**
```cpp
#include <unordered_map>

std::unordered_map<std::string, int> word_count;

// 插入和计数
word_count["hello"]++;
word_count["world"]++;
word_count["hello"]++;

// 遍历（无序，但查找速度快）
for (const auto &pair : word_count) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}

// 对比Python的dict（Python的dict也是哈希表实现）
# word_count = {}
# word_count["hello"] = word_count.get("hello", 0) + 1
```

### 1.2 迭代器：概念与使用

**迭代器基础：**
```cpp
std::vector<int> vec = {10, 20, 30, 40, 50};

// 使用迭代器遍历
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}
std::cout << std::endl;

// 反向迭代器
for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
    std::cout << *rit << " ";  // 50 40 30 20 10
}
std::cout << std::endl;

// 对比Python的迭代器
# for item in vec:
#     print(item, end=" ")
# for item in reversed(vec):
#     print(item, end=" ")
```

**迭代器类别：**
- **输入迭代器**：只能读取，单向移动
- **输出迭代器**：只能写入，单向移动
- **前向迭代器**：可读写，单向移动
- **双向迭代器**：可读写，双向移动（list、set、map）
- **随机访问迭代器**：可读写，随机访问（vector、array）

### 1.3 算法：sort、find、transform等

**常用算法示例：**
```cpp
#include <algorithm>
#include <numeric>

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};

// 排序
std::sort(numbers.begin(), numbers.end());

// 查找
auto found = std::find(numbers.begin(), numbers.end(), 5);
if (found != numbers.end()) {
    std::cout << "找到5在位置: " << std::distance(numbers.begin(), found) << std::endl;
}

// 转换（类似Python的map）
std::vector<int> squared;
std::transform(numbers.begin(), numbers.end(), 
               std::back_inserter(squared),
               [](int x) { return x * x; });

// 过滤（类似Python的filter）
numbers.erase(std::remove_if(numbers.begin(), numbers.end(),
                            [](int x) { return x % 2 == 0; }),
              numbers.end());

// 累加（类似Python的sum）
int total = std::accumulate(numbers.begin(), numbers.end(), 0);

// 对比Python
# numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# numbers.sort()
# squared = list(map(lambda x: x*x, numbers))
# numbers = list(filter(lambda x: x % 2 != 0, numbers))
# total = sum(numbers)
```

### 1.4 lambda表达式（C++11）

**lambda语法：**
```cpp
// 基本lambda
auto square = [](int x) { return x * x; };
std::cout << square(5) << std::endl;  // 25

// 捕获外部变量
int factor = 3;
auto multiply = [factor](int x) { return x * factor; };

// 按引用捕获
int counter = 0;
auto incrementer = [&counter]() { counter++; };

// 在算法中使用lambda
std::vector<int> nums = {1, 2, 3, 4, 5};
std::for_each(nums.begin(), nums.end(), 
              [](int &x) { x *= 2; });

// 对比Python的lambda
# square = lambda x: x * x
# factor = 3
# multiply = lambda x: x * factor
# nums = [1, 2, 3, 4, 5]
# nums = list(map(lambda x: x * 2, nums))
```

**捕获方式：**
- `[]`：不捕获任何变量
- `[=]`：按值捕获所有外部变量
- `[&]`：按引用捕获所有外部变量
- `[x, &y]`：按值捕获x，按引用捕获y
- `[this]`：捕获当前对象的this指针

## 2. 错误处理

### 2.1 异常处理：try-catch-throw

**基本异常处理：**
```cpp
#include <stdexcept>

double safe_divide(double a, double b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
    }
    return a / b;
}

int main() {
    try {
        double result = safe_divide(10, 0);
        std::cout << "结果: " << result << std::endl;
    }
    catch (const std::invalid_argument &e) {
        std::cerr << "数学错误: " << e.what() << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "一般错误: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "未知错误" << std::endl;
    }
    
    return 0;
}

// 对比Python的try-except
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

### 2.2 异常安全与noexcept

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

## 3. 现代C++特性

### 3.1 自动类型推导：auto关键字

**auto使用场景：**
```cpp
// 简化复杂类型声明
auto number = 42;                    // int
auto name = std::string("Alice");    // std::string
auto scores = std::vector<int>{90, 85, 95};  // std::vector<int>

// 在迭代器中使用
auto it = scores.begin();            // std::vector<int>::iterator
for (auto &score : scores) {         // 引用，可以修改
    score += 5;
}

// 函数返回类型推导（C++14）
auto add(int a, int b) -> int {      // 尾置返回类型
    return a + b;
}

// 对比Python的动态类型
# number = 42
# name = "Alice"
# scores = [90, 85, 95]
```

**decltype类型推导：**
```cpp
int x = 10;
decltype(x) y = 20;  // y的类型与x相同（int）

auto add(int a, int b) -> decltype(a + b) {
    return a + b;  // 返回类型由a+b的类型决定
}
```

### 3.2 移动语义与右值引用（C++11）

**左值 vs 右值：**
- **左值**：有标识符，可以取地址的表达式
- **右值**：临时对象，即将被销毁的对象

**移动语义示例：**
```cpp
class String {
private:
    char *data;
    size_t length;
    
public:
    // 移动构造函数
    String(String &&other) noexcept 
        : data(other.data), length(other.length) {
        other.data = nullptr;    // 转移所有权
        other.length = 0;
    }
    
    // 移动赋值运算符
    String &operator=(String &&other) noexcept {
        if (this != &other) {
            delete[] data;       // 释放原有资源
            data = other.data;   // 转移资源
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
    
    ~String() {
        delete[] data;
    }
};

String create_string() {
    String temp("Hello");
    return temp;  // 触发移动语义（而非拷贝）
}

String s1 = create_string();  // 高效，使用移动构造函数
```

**std::move：强制转换为右值引用**
```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination = std::move(source);  // 移动而非拷贝

// source现在为空
td::cout << "source大小: " << source.size() << std::endl;      // 0
std::cout << "destination大小: " << destination.size() << std::endl;  // 5
```

### 3.3 范围for循环与初始化列表

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

## 4. 实战应用与项目结构

### 4.1 头文件与源文件组织

**头文件（.h/.hpp）：声明接口**
```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>

namespace math {
    // 函数声明
    double calculate_average(const std::vector<double> &numbers);
    double calculate_standard_deviation(const std::vector<double> &numbers);
    
    // 类声明
    class Statistics {
    private:
        std::vector<double> data;
        
    public:
        void add_data(double value);
        double mean() const;
        double variance() const;
    };
}

#endif // MATH_UTILS_H
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

### 4.2 CMake基础配置

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

### 4.3 常用设计模式在C++中的实现

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

### 4.4 性能优化要点

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