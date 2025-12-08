# C++基础与核心语法

## 1. C++语言概述

### 1.1 C++与Python的主要差异

**核心差异对比表：**

| 特性 | C++ | Python |
|------|-----|--------|
| **执行方式** | 编译型，直接生成机器码 | 解释型，通过解释器执行 |
| **类型系统** | 静态强类型，编译时检查 | 动态弱类型，运行时检查 |
| **内存管理** | 手动/智能指针管理 | 自动垃圾回收 |
| **性能特点** | 运行速度快，内存控制精细 | 开发效率高，运行相对较慢 |
| **语法特点** | 语法严格，需要分号结束 | 语法简洁，依赖缩进 |

### 1.2 C++的编译型语言特性

**编译过程深度解析：**
```cpp
// 源代码文件: main.cpp
#include <iostream>

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}

// 编译命令: g++ -o main main.cpp
// 执行: ./main
```

**编译流程：**
1. **预处理**：处理 `#include`、`#define` 等指令
2. **编译**：将C++代码翻译成汇编代码
3. **汇编**：将汇编代码翻译成机器代码
4. **链接**：连接多个目标文件生成可执行文件

> **注意**：Python是解释执行，无需编译步骤；C++需要完整的编译流程才能运行

### 1.3 C++标准发展历程

**重要版本特性简介：**
- **C++11**：auto、lambda、智能指针、范围for循环
- **C++14**：泛型lambda、返回类型推导
- **C++17**：结构化绑定、std::optional、文件系统库
- **C++20**：概念(concepts)、协程(coroutines)、范围(ranges)

## 2. 开发环境与基础语法

### 2.1 快速开发环境配置

**VS Code + CMake 配置：**
```cpp
// CMakeLists.txt 示例
cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main main.cpp)
```

**对比Python环境：**
- Python：直接安装解释器即可运行
- C++：需要编译器(g++/clang) + 构建工具(CMake/Make)

### 2.2 第一个C++程序深度解析

```cpp
#include <iostream>  // 类似Python的import，但处理方式不同

// Python: print("Hello")
// C++: 需要包含头文件和使用命名空间
int main() {  // 程序入口，必须的函数
    std::cout << "Hello, C++!" << std::endl;  // 输出语句
    return 0;  // 返回状态码
}
```

**关键差异：**
- Python：脚本直接执行，从第一行开始
- C++：必须有 `main()` 函数作为入口
- Python：`print()` 是内置函数
- C++：`std::cout` 需要包含头文件

### 2.3 数据类型系统：静态类型 vs 动态类型

**C++数据类型声明：**
```cpp
// C++需要显式声明类型
int number = 42;           // 整数
double pi = 3.14159;       // 双精度浮点数
char letter = 'A';         // 字符
bool is_valid = true;      // 布尔值
std::string name = "Alice"; // 字符串

// 对比Python的动态类型
# Python: 变量类型动态确定
number = 42          # 整数
pi = 3.14159         # 浮点数
name = "Alice"       # 字符串
```

**类型系统优势对比：**
- **C++静态类型**：编译时检查，性能优化，减少运行时错误
- **Python动态类型**：开发灵活，代码简洁，但运行时可能出错

### 2.4 变量声明与初始化

```cpp
// C++变量声明与初始化
int x;              // 声明，未初始化（危险！）
int y = 10;         // 声明并初始化
int z{20};          // C++11统一初始化（推荐）

// 常量声明
const int MAX_SIZE = 100;      // 编译时常量
constexpr int BUFFER_SIZE = 64; // C++11编译时常量表达式

// 对比Python
# Python变量使用
x = 10              # 直接赋值，无需声明类型
MAX_SIZE = 100      # 常量（约定，非强制）
```

## 3. 运算符与流程控制

### 3.1 运算符详解

**算术运算符：**
```cpp
int a = 10, b = 3;
std::cout << a + b << std::endl;  // 13
std::cout << a / b << std::endl;  // 3（整数除法）
std::cout << a % b << std::endl;  // 1（取模）

// 对比Python的除法
# Python: 10 / 3 = 3.333... (浮点除法)
# Python: 10 // 3 = 3 (整数除法)
```

**关系与逻辑运算符：**
```cpp
bool result = (a > b) && (a != 0);  // 与运算
result = (a < 5) || (b > 0);        // 或运算
result = !(a == b);                 // 非运算
```

**位运算（Python中也存在）：**
```cpp
unsigned int flags = 0b1010;  // 二进制表示
flags = flags | 0b0001;       // 按位或
flags = flags & ~0b1000;      // 按位与+取反
```

### 3.2 条件语句

**if-else语句：**
```cpp
int score = 85;

if (score >= 90) {
    std::cout << "优秀" << std::endl;
} else if (score >= 80) {
    std::cout << "良好" << std::endl;
} else {
    std::cout << "需努力" << std::endl;
}

// 对比Python的if-elif-else结构
# if score >= 90:
#     print("优秀")
# elif score >= 80:
#     print("良好")
# else:
#     print("需努力")
```

**switch-case语句（C++特有）：**
```cpp
int day = 3;
switch (day) {
    case 1:
        std::cout << "Monday" << std::endl;
        break;  // 必须break，否则会"穿透"
    case 2:
        std::cout << "Tuesday" << std::endl;
        break;
    default:
        std::cout << "Other day" << std::endl;
}
```

> **注意**：Python没有switch语句，通常用if-elif或字典映射实现类似功能

### 3.3 循环结构

**for循环：**
```cpp
// 传统for循环
for (int i = 0; i < 5; i++) {
    std::cout << i << " ";
}
// 输出: 0 1 2 3 4

// 对比Python的for循环
# for i in range(5):
#     print(i, end=" ")
```

**范围for循环（C++11）：**
```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};

// 范围for循环（类似Python的for-in）
for (int num : numbers) {
    std::cout << num << " ";
}
// 输出: 1 2 3 4 5

// 对比Python
# for num in numbers:
#     print(num, end=" ")
```

**while循环：**
```cpp
int count = 0;
while (count < 5) {
    std::cout << count << " ";
    count++;
}
// 与Python的while语法相同
```

**do-while循环（C++特有）：**
```cpp
int input;
do {
    std::cout << "请输入正数: ";
    std::cin >> input;
} while (input <= 0);  // 至少执行一次
```

## 4. 函数与作用域

### 4.1 函数声明与定义

**函数基本结构：**
```cpp
// 函数声明（通常在头文件中）
int add(int a, int b);

// 函数定义
int add(int a, int b) {
    return a + b;
}

// 对比Python的def
# def add(a, b):
#     return a + b
```

**函数声明与定义分离：**
```cpp
// math_utils.h（头文件）
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int multiply(int x, int y);
double power(double base, int exponent);

#endif

// math_utils.cpp（源文件）
#include "math_utils.h"

int multiply(int x, int y) {
    return x * y;
}

double power(double base, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}
```

### 4.2 参数传递机制

**值传递（默认）：**
```cpp
void modifyValue(int x) {
    x = 100;  // 只修改副本，不影响原变量
}

int main() {
    int a = 10;
    modifyValue(a);
    std::cout << a << std::endl;  // 输出: 10
    return 0;
}
```

**引用传递：**
```cpp
void modifyReference(int &x) {
    x = 100;  // 修改原变量
}

int main() {
    int a = 10;
    modifyReference(a);
    std::cout << a << std::endl;  // 输出: 100
    return 0;
}
```

**指针传递：**
```cpp
void modifyPointer(int *x) {
    *x = 100;  // 通过指针修改原变量
}

int main() {
    int a = 10;
    modifyPointer(&a);
    std::cout << a << std::endl;  // 输出: 100
    return 0;
}
```

> **注意**：Python中所有参数传递都是"对象引用传递"，与C++的引用传递概念不同

### 4.3 函数重载（C++特有）

**函数重载示例：**
```cpp
// 同名函数，不同参数列表
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

std::string add(const std::string &a, const std::string &b) {
    return a + b;
}

// 使用重载函数
std::cout << add(5, 3) << std::endl;               // 8
std::cout << add(2.5, 3.7) << std::endl;           // 6.2
std::cout << add("Hello", " World") << std::endl;  // Hello World
```

**对比Python：** Python不支持函数重载，通常用默认参数或类型检查实现类似功能

### 4.4 内联函数与constexpr

**内联函数：**
```cpp
// 建议编译器将函数体直接插入调用处
inline int square(int x) {
    return x * x;
}

// 使用内联函数（可能被优化为直接计算）
int result = square(5);  // 可能被优化为: int result = 5 * 5;
```

**constexpr函数（C++11）：**
```cpp
// 编译时可计算的函数
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// 编译时计算
constexpr int fact_5 = factorial(5);  // 在编译时计算出120
```

**作用域规则：**
```cpp
int global_var = 100;  // 全局变量

void function() {
    int local_var = 50;  // 局部变量
    
    if (true) {
        int block_var = 10;  // 块作用域变量
        std::cout << block_var << std::endl;  // 可以访问
    }
    // std::cout << block_var << std::endl;  // 错误！超出作用域
}
```

> **关键理解**：C++的作用域规则比Python更严格，变量生命周期管理更重要

---

**总结要点：**
- C++的静态类型系统提供编译时安全性和性能优势
- 函数重载和内联函数是C++特有的重要特性
- 参数传递机制（值/引用/指针）需要仔细理解
- C++的编译模型与Python的解释执行有本质区别

下一章将深入探讨C++的内存管理和面向对象特性。