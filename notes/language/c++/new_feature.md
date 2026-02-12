## <span style="background-color: #ff6b6b; padding: 2px 4px; border-radius: 3px; color: white;">C++11新特性</span>

### 自动类型推导（auto）

C++11引入auto关键字，允许编译器自动推导变量类型：

```cpp
// 基本类型推导
auto x = 5;                    // int
auto y = 3.14;                 // double
auto name = "Hello";           // const char*

// 迭代器类型推导简化
std::vector<int> numbers = {1, 2, 3, 4, 5};
for (auto it = numbers.begin(); it != numbers.end(); ++it) {
    std::cout << *it << " ";
}

// 与复杂类型结合
auto result = std::make_pair("Alice", 95);  // std::pair<const char*, int>
```

### 范围for循环

提供更简洁的容器遍历语法：

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};

// 传统迭代方式
for (std::vector<int>::iterator it = numbers.begin(); it != numbers.end(); ++it) {
    std::cout << *it << " ";
}

// C++11范围for循环
for (const auto& number : numbers) {
    std::cout << number << " ";
}

// 修改元素
for (auto& number : numbers) {
    number *= 2;  // 修改每个元素
}
```

### Lambda表达式

C++11引入匿名函数，简化函数对象创建：

```cpp
#include <algorithm>
#include <vector>

std::vector<int> numbers = {1, 2, 3, 4, 5};

// 基本Lambda表达式
std::for_each(numbers.begin(), numbers.end(), [](int n) {
    std::cout << n << " ";
});

// 捕获外部变量
int threshold = 3;
numbers.erase(std::remove_if(numbers.begin(), numbers.end(), 
    [threshold](int n) { return n > threshold; }), numbers.end());

// 多种捕获方式
int a = 1, b = 2;
auto lambda1 = [a]() { return a; };           // 值捕获
auto lambda2 = [&b]() { return b++; };       // 引用捕获
auto lambda3 = [=]() { return a + b; };      // 全部值捕获
auto lambda4 = [&]() { return ++a + ++b; };  // 全部引用捕获
```

### 右值引用和移动语义

提高性能，避免不必要的拷贝：

```cpp
class String {
private:
    char* data;
    size_t length;
    
public:
    // 移动构造函数
    String(String&& other) noexcept 
        : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
    }
    
    // 移动赋值运算符
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
    
    // std::move使用
    String createString() {
        String temp("Hello");
        return std::move(temp);  // 触发移动语义
    }
};
```

### 智能指针

自动内存管理，避免内存泄漏：

```cpp
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created\n"; }
    ~Resource() { std::cout << "Resource destroyed\n"; }
    void use() { std::cout << "Using resource\n"; }
};

// unique_ptr：独占所有权
std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>();
ptr1->use();

// shared_ptr：共享所有权
std::shared_ptr<Resource> ptr2 = std::make_shared<Resource>();
std::shared_ptr<Resource> ptr3 = ptr2;  // 引用计数增加

// weak_ptr：避免循环引用
std::weak_ptr<Resource> weak = ptr2;
if (auto shared = weak.lock()) {
    shared->use();
}
```

### 初始化列表

统一的初始化语法：

```cpp
// 各种类型的统一初始化
std::vector<int> numbers = {1, 2, 3, 4, 5};
std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 88}};
int array[] = {1, 2, 3, 4, 5};

// 自定义类的初始化列表
class MyClass {
public:
    MyClass(std::initializer_list<int> list) {
        for (int value : list) {
            std::cout << value << " ";
        }
    }
};

MyClass obj = {1, 2, 3, 4, 5};  // 使用初始化列表
```

### 类型推导（decltype）

在编译时获取表达式的类型：

```cpp
int x = 10;
double y = 3.14;

// 基本类型推导
decltype(x) a = 20;           // int
decltype(y) b = 2.71;         // double
decltype(x + y) c = x + y;    // double

// 与函数返回类型结合
template<typename T, typename U>
auto add(T t, U u) -> decltype(t + u) {
    return t + u;
}

// 引用类型推导
int& getRef(int& x) { return x; }
decltype(getRef(x)) ref = x;  // int&
```

### nullptr关键字

类型安全的空指针：

```cpp
// 传统NULL的问题
void func(int* ptr) {}
void func(int value) {}

func(NULL);    // 歧义：可能调用func(int)
func(nullptr); // 明确调用func(int*)

// 使用示例
int* ptr = nullptr;
if (ptr == nullptr) {
    std::cout << "指针为空" << std::endl;
}
```

### 强类型枚举

避免传统枚举的命名冲突和隐式转换：

```cpp
// 传统枚举的问题
enum Color { RED, GREEN, BLUE };
enum TrafficLight { RED, YELLOW, GREEN };  // 冲突！

// C++11强类型枚举
enum class Color { RED, GREEN, BLUE };
enum class TrafficLight { RED, YELLOW, GREEN };  // 无冲突

Color color = Color::RED;
TrafficLight light = TrafficLight::GREEN;

// 需要显式转换
// int value = color;  // 错误：不能隐式转换
int value = static_cast<int>(color);  // 正确：显式转换
```

### constexpr关键字

编译时常量表达式：

```cpp
// 编译时常量函数
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fact5 = factorial(5);  // 编译时计算

// 编译时数组大小
constexpr int size = 10;
int array[size];  // 有效：size是编译时常量

// 编译时字符串长度
constexpr size_t str_len(const char* str) {
    return (*str == '\0') ? 0 : 1 + str_len(str + 1);
}
constexpr auto len = str_len("Hello");  // 5
```

### 可变参数模板

处理任意数量和类型的参数：

```cpp
// 基本可变参数模板
template<typename T>
void print(T t) {
    std::cout << t << std::endl;
}

template<typename T, typename... Args>
void print(T t, Args... args) {
    std::cout << t << " ";
    print(args...);
}

// 使用示例
print(1, 2.5, "Hello", 'A');  // 输出：1 2.5 Hello A

// 参数包展开
template<typename... Args>
void log_all(Args... args) {
    (std::cout << ... << args) << std::endl;  // C++17折叠表达式
}
```

### 标准库增强

C++11为STL添加了大量新功能：

```cpp
#include <thread>
#include <chrono>
#include <regex>
#include <random>

// 多线程支持
std::thread t([]() {
    std::cout << "Hello from thread!" << std::endl;
});
t.join();

// 时间库
auto start = std::chrono::high_resolution_clock::now();
// 执行一些操作
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

// 正则表达式
std::regex pattern(R"(\w+@\w+\.\w+)");
std::string email = "test@example.com";
if (std::regex_match(email, pattern)) {
    std::cout << "Valid email" << std::endl;
}

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, 6);
int dice_roll = dis(gen);  // 生成1-6的随机数
```

## <span style="background-color: #ff6b6b; padding: 2px 4px; border-radius: 3px; color: white;">C++14新特性</span>

### 函数返回类型推导

C++14允许函数返回类型自动推导，简化了函数定义：

```cpp
// C++11需要显式指定返回类型
auto add(int a, int b) -> decltype(a + b) {
    return a + b;
}

// C++14可以直接使用auto推导
auto add(int a, int b) {
    return a + b;  // 自动推导返回类型为int
}

// 模板函数也大大简化
template<typename T, typename U>
auto multiply(T t, U u) {
    return t * u;  // 自动推导返回类型
}
```

### 泛型Lambda表达式

C++14允许Lambda表达式使用auto作为参数类型，实现泛型功能：

```cpp
// 基本泛型Lambda
auto genericAdd = [](auto x, auto y) {
    return x + y;
};

std::cout << genericAdd(5, 3) << std::endl;                    // 8
std::cout << genericAdd(2.5, 1.5) << std::endl;                // 4.0
std::cout << genericAdd(std::string("Hello, "), std::string("world!")) << std::endl;

// 与STL算法结合使用
std::vector<int> numbers = {1, 2, 3, 4, 5};
std::for_each(numbers.begin(), numbers.end(), [](const auto& n) {
    std::cout << n << " ";
});
```

### Lambda捕获表达式

C++14允许在Lambda捕获列表中创建新变量并初始化：

```cpp
// 基本初始化捕获
auto lambda = [value = 42]() {
    return value;
};

// 移动捕获
auto moveLambda = [ptr = std::make_unique<int>(10)]() {
    return *ptr;
};

// 引用捕获并初始化
int external = 5;
auto refLambda = [&x = external]() {
    x *= 2;
    return x;
};
```

### 变量模板

C++14引入变量模板，允许定义可以参数化的变量：

```cpp
template<typename T>
constexpr T pi = T(3.1415926535897932385);

template<typename T>
T circularArea(T r) {
    return pi<T> * r * r;
}

// 使用
double area = circularArea(5.0);
std::cout << "Pi (float): " << pi<float> << std::endl;
std::cout << "Pi (double): " << pi<double> << std::endl;
```

### [[deprecated]]属性

C++14引入标准化的deprecated属性：

```cpp
[[deprecated("Use newFunction() instead")]]
void oldFunction() {
    // 旧实现
}

void newFunction() {
    // 新实现
}
```

### 二进制字面量和数字分隔符

C++14支持二进制字面量和数字分隔符，提高代码可读性：

```cpp
// 二进制字面量
int binary = 0b10101010;       // 十进制170
unsigned char flags = 0b1010'1010;  // 使用分隔符

// 数字分隔符
long long largeNumber = 1'000'000'000'000;  // 1万亿
double pi = 3.141'592'653'589'793'238'462;  // 更易读的pi值
```

### std::make_unique

C++14补充了std::make_unique用于创建unique_ptr：

```cpp
#include <memory>

// 基本用法
auto ptr = std::make_unique<int>(42);

// 创建对象并传递参数
auto obj = std::make_unique<MyClass>(arg1, arg2);

// 异常安全：比直接使用new更安全
auto r1 = std::make_unique<Resource>("r1");
auto r2 = std::make_unique<Resource>("r2");
```

## <span style="background-color: #4ecdc4; padding: 2px 4px; border-radius: 3px; color: white;">C++17新特性</span>

### 结构化绑定

C++17允许将结构体、类、数组或元组的成员直接绑定到多个变量：

```cpp
#include <tuple>
#include <map>
#include <string>

// 元组结构化绑定
auto getStudent() {
    return std::make_tuple(123, "Alice", 95.5);
}

auto [id, name, score] = getStudent();  // 结构化绑定

// 结构体结构化绑定
struct Point {
    double x, y;
};

Point p = {3.0, 4.0};
auto [x, y] = p;  // x = 3.0, y = 4.0

// map遍历优化
std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 88}};
for (const auto& [name, score] : scores) {
    std::cout << name << ": " << score << std::endl;
}
```

### if和switch语句初始化

C++17允许在if和switch语句中声明变量：

```cpp
#include <map>
#include <string>

std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 88}};

// if语句初始化
if (auto it = scores.find("Alice"); it != scores.end()) {
    std::cout << "Found: " << it->second << std::endl;
} else {
    std::cout << "Not found" << std::endl;
}

// switch语句初始化
switch (int value = getValue(); value) {
    case 1:
        std::cout << "Value is 1" << std::endl;
        break;
    case 2:
        std::cout << "Value is 2" << std::endl;
        break;
    default:
        std::cout << "Other value" << std::endl;
}
```

### 内联变量

C++17允许在头文件中定义内联变量：

```cpp
// 传统方式（需要.cpp文件定义）
// header.h
extern const int VERSION;
// header.cpp
const int VERSION = 1;

// C++17内联变量
// header.h
inline constexpr int VERSION = 1;  // 可以在头文件中定义
```

### constexpr Lambda表达式

C++17允许Lambda表达式在编译时求值：

```cpp
constexpr auto square = [](int n) { return n * n; };
constexpr int result = square(5);  // 编译时计算

// 在模板中使用
template<typename F>
constexpr auto apply(F f, int n) {
    return f(n);
}

constexpr int cubed = apply([](int x) { return x * x * x; }, 3);
```

### 类模板参数推导

C++17可以自动推导类模板参数：

```cpp
#include <vector>
#include <tuple>

// C++17之前需要显式指定类型
std::vector<int> v1 = {1, 2, 3};
std::pair<int, double> p1(1, 3.14);

// C++17自动推导
std::vector v2 = {1, 2, 3};        // 推导为vector<int>
std::pair p2(1, 3.14);             // 推导为pair<int, double>
std::tuple t3(1, 3.14, "hello");   // 推导为tuple<int, double, const char*>
```

### std::variant, std::optional, std::any

C++17引入新的类型安全容器：

```cpp
#include <variant>
#include <optional>
#include <any>

// std::variant：类型安全的联合体
std::variant<int, double, std::string> value;
value = "hello";
if (std::holds_alternative<std::string>(value)) {
    std::cout << "String: " << std::get<std::string>(value) << std::endl;
}

// std::optional：可选的返回值
std::optional<int> findValue(const std::vector<int>& vec, int target) {
    auto it = std::find(vec.begin(), vec.end(), target);
    if (it != vec.end()) {
        return *it;
    }
    return std::nullopt;  // 没有找到
}

std::any：任意类型的容器
std::any anything;
anything = 42;
anything = "hello";
anything = 3.14;
```

### std::string_view

C++17引入string_view，提供非拥有字符串视图：

```cpp
#include <string_view>

// 避免不必要的字符串拷贝
void processString(std::string_view sv) {
    std::cout << "Length: " << sv.length() << std::endl;
    std::cout << "Substring: " << sv.substr(0, 5) << std::endl;
}

// 可以接受多种字符串类型
processString("Hello World");          // C风格字符串
processString(std::string("Hello"));    // std::string
processString(std::string_view("Hi"));  // string_view
```

### 文件系统库

C++17提供标准文件系统操作：

```cpp
#include <filesystem>
namespace fs = std::filesystem;

// 检查文件状态
if (fs::exists("example.txt")) {
    std::cout << "文件大小: " << fs::file_size("example.txt") << " 字节" << std::endl;
}

// 遍历目录
for (const auto& entry : fs::directory_iterator(".")) {
    std::cout << (entry.is_directory() ? "[DIR] " : "[FILE] ") 
              << entry.path().filename() << std::endl;
}

// 创建目录和文件
fs::create_directory("new_dir");
fs::copy("source.txt", "new_dir/destination.txt");
```

## <span style="background-color: #ff9ff3; padding: 2px 4px; border-radius: 3px; color: white;">C++20新特性</span>

### 概念（Concepts）

C++20引入概念，为模板编程提供类型约束：

```cpp
#include <concepts>

// 定义概念
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// 使用概念约束模板
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// 更复杂的约束
template<typename T>
concept Container = requires(T t) {
    t.begin();
    t.end();
    typename T::value_type;
};

template<Container C>
void print_container(const C& container) {
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### 范围（Ranges）

C++20提供更强大的范围操作：

```cpp
#include <ranges>
#include <vector>
#include <algorithm>

std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 范围视图操作
auto result = numbers | std::views::filter([](int n) { return n % 2 == 0; })
                     | std::views::transform([](int n) { return n * n; })
                     | std::views::take(3);

for (int n : result) {
    std::cout << n << " ";  // 输出：4 16 36
}
```

### 协程（Coroutines）

C++20引入协程支持异步编程：

```cpp
#include <coroutine>
#include <iostream>

generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;  // 暂停并返回值
    }
}

// 使用协程
for (int i : range(1, 5)) {
    std::cout << i << " ";  // 输出：1 2 3 4
}
```

### 三路比较运算符（Spaceship Operator）

C++20简化比较操作：

```cpp
#include <compare>

class Point {
public:
    int x, y;
    
    // 自动生成所有比较运算符
    auto operator<=>(const Point&) const = default;
};

Point p1{1, 2}, p2{3, 4};
if (p1 < p2) {  // 自动可用
    std::cout << "p1小于p2" << std::endl;
}
```

### 模块（Modules）

C++20引入模块系统，改善编译时间：

```cpp
// math.ixx (模块接口文件)
export module math;

export int add(int a, int b) {
    return a + b;
}

export double pi = 3.14159;

// main.cpp
import math;  // 导入模块

int main() {
    std::cout << add(2, 3) << std::endl;  // 5
    return 0;
}
```

### constexpr的增强

C++20进一步扩展constexpr能力：

```cpp
// constexpr动态内存分配
constexpr auto create_array() {
    std::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    return vec;
}

constexpr auto arr = create_array();  // 编译时创建vector

// constexpr虚函数
struct Shape {
    virtual constexpr double area() const = 0;
};
```

## <span style="background-color: #1a936f; padding: 2px 4px; border-radius: 3px; color: white;">现代C++演进总结</span>

### 核心特性演进对比

| 版本 | 类型系统 | 标准库增强 | 编译时优化 | 语法简化 |
|------|----------|------------|------------|----------|
| **C++11** | auto、decltype、右值引用 | 智能指针、正则表达式、多线程 | constexpr、可变参数模板 | 范围for、lambda、初始化列表 |
| **C++14** | 函数返回类型推导 | make_unique、泛型lambda | 变量模板、constexpr增强 | 数字分隔符、二进制字面量 |
| **C++17** | 结构化绑定、optional/variant | 文件系统、string_view | if/switch初始化、内联变量 | 类模板参数推导 |
| **C++20** | 概念（Concepts） | 范围（Ranges）、协程 | 模块、增强constexpr | 三路比较运算符 |

### 语法演进趋势

**推导化趋势：** 现代C++鼓励编译器推断类型，减少冗余代码
```cpp
// 演进对比：从显式到隐式
std::vector<int>::iterator it = vec.begin();  // C++98
auto it = vec.begin();                         // C++11
std::vector v = {1, 2, 3};                     // C++17
```

**声明式编程：** 从命令式迭代转向声明式操作
```cpp
// 现代C++20风格：管道操作符
auto result = numbers | std::views::filter(even) | std::views::transform(square);
```

### 关键进步领域

1. **类型安全增强**
   - 概念（Concepts）提供编译时类型约束
   - optional/variant替代裸指针和联合体
   - 结构化绑定简化对象解构

2. **性能优化持续**
   - 移动语义减少不必要的拷贝
   - constexpr实现编译时计算
   - 模块系统改善编译时间

3. **开发效率提升**
   - 范围库提供声明式数据操作
   - 协程支持异步编程
   - 统一的初始化语法

### 实际应用代码示例

**现代C++20完整示例：**
```cpp
import std.core;
import std.ranges;

template<std::integral T>
consteval auto generate_fibonacci(int n) -> std::vector<T> {
    std::vector<T> fib(n);
    if (n > 0) fib[0] = 0;
    if (n > 1) fib[1] = 1;
    for (int i = 2; i < n; ++i) {
        fib[i] = fib[i-1] + fib[i-2];
    }
    return fib;
}

int main() {
    constexpr auto fibs = generate_fibonacci<int>(10);
    
    // 现代范围操作
    auto result = fibs | std::views::filter([](int n) { return n % 2 == 0; })
                       | std::views::transform([](int n) { return n * n; });
    
    for (int n : result) {
        std::cout << n << " ";
    }
    return 0;
}
```

### 学习路径建议

1. **基础阶段**（C++11）：掌握auto、lambda、智能指针、移动语义
2. **进阶阶段**（C++14/17）：学习结构化绑定、optional、文件系统
3. **高级阶段**（C++20）：深入概念、范围、协程、模块
4. **实践应用**：在项目中逐步引入现代特性，关注性能提升

**总结：** 现代C++通过持续的语法简化和性能优化，保持了在高性能计算和系统编程领域的竞争力。关键在于理解新特性背后的设计理念，并在合适的场景下应用它们。

