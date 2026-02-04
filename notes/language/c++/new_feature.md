

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

## 总结

C++14和C++17为现代C++开发带来了众多实用特性，显著提高了代码的可读性、安全性和开发效率。这些新特性与传统的C++语法相结合，使得C++在现代系统编程、高性能计算等领域继续保持强大的竞争力。


