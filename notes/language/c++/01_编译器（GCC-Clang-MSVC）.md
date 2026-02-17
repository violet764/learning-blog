# 工具与环境

本章节介绍C++开发所需的工具与环境，包括编译器、调试器、构建系统等核心工具的使用。

---

# 1. 编译器（GCC/Clang/MSVC）

## 1.1 核心概念
- 定义：编译器是将C++源代码转换为可执行程序的工具
- 关键特性：代码优化、错误检查、平台兼容性、标准支持

## 1.2 语法规则与使用
- 基本语法：命令行参数、编译选项、链接选项
- 代码示例：
```cpp
// simple.cpp - 简单的C++程序示例
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int sum = 0;

    // C++11范围for循环
    for (int num : numbers) {
        sum += num;
    }

    std::cout << "数字总和: " << sum << std::endl;
    std::cout << "编译器信息测试" << std::endl;

    // 编译器特定宏检查
    #if defined(__GNUC__)
        std::cout << "使用GCC编译器，版本: " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
    #elif defined(__clang__)
        std::cout << "使用Clang编译器，版本: " << __clang_major__ << "." << __clang_minor__ << std::endl;
    #elif defined(_MSC_VER)
        std::cout << "使用MSVC编译器，版本: " << _MSC_VER << std::endl;
    #endif

    return 0;
}
```
- 注意事项：编译器差异、标准支持程度、平台特定功能

## 1.3 常见用法

### 编译和链接过程详解

#### 手动编译步骤
```bash
# 1. 编译源文件为目标文件
g++ -c main.cpp -o main.o
g++ -c math_operations.cpp -o math_operations.o
g++ -c globals.cpp -to globals.o

# 2. 链接目标文件为可执行文件
g++ main.o math_operations.o globals.o -o calculator

# 3. 运行程序
./calculator
```

#### 编译过程详解

**预处理阶段**：
```bash
g++ -E main.cpp -o main.i  # 查看预处理结果
```

**编译阶段**：
```bash
g++ -S main.cpp -o main.s  # 生成汇编代码
```

**汇编阶段**：
```bash
g++ -c main.s -o main.o    # 汇编为目标文件
```

#### 编译选项说明

| 选项 | 说明 | 示例 |
|------|------|------|
| `-c` | 只编译不链接 | `g++ -c file.cpp` |
| `-o` | 指定输出文件名 | `g++ -o program file.cpp` |
| `-I` | 添加头文件搜索路径 | `g++ -I./include file.cpp` |
| `-L` | 添加库文件搜索路径 | `g++ -L./lib file.cpp` |
| `-l` | 链接指定的库 | `g++ -lm file.cpp`（数学库） |
| `-std` | 指定C++标准 | `g++ -std=c++17 file.cpp` |
| `-Wall` | 开启所有警告 | `g++ -Wall file.cpp` |
| `-g` | 生成调试信息 | `g++ -g file.cpp` |
| `-O2` | 优化级别2 | `g++ -O2 file.cpp` |

### GCC编译命令
```bash
# 基本编译
g++ -o program simple.cpp

# 启用C++11标准
g++ -std=c++11 -o program simple.cpp

# 启用优化和调试信息
g++ -O2 -g -o program simple.cpp

# 显示所有警告
g++ -Wall -Wextra -o program simple.cpp

# 多文件编译
g++ -c file1.cpp -o file1.o
g++ -c file2.cpp -o file2.o
g++ file1.o file2.o -o program
```

### Clang编译命令
```bash
# 基本编译
clang++ -o program simple.cpp

# 启用最新C++标准
clang++ -std=c++17 -o program simple.cpp

# 静态分析
clang++ --analyze simple.cpp

# 生成LLVM IR
clang++ -S -emit-llvm simple.cpp -o simple.ll
```

### MSVC编译命令
```cmd
# 使用Visual Studio开发者命令提示符
cl /EHsc simple.cpp

# 启用C++17标准
cl /std:c++17 /EHsc simple.cpp

# 调试版本
cl /Zi /EHsc simple.cpp

# 发布版本优化
cl /O2 /EHsc simple.cpp
```

## 1.4 易错点/坑

### 错误示例1：使用编译器特定扩展
```cpp
int array[10] = {0};
// 某些编译器可能不支持某些初始化语法
```
- 原因：依赖于特定编译器的非标准扩展
- 修正方案：使用标准C++语法，避免编译器特定功能

### 错误示例2：错误的链接顺序
```bash
# 错误的链接顺序
g++ -o program -lmath main.cpp  # 数学库应该在对象文件之后
```
- 原因：链接器参数顺序很重要
- 修正方案：正确排序链接参数
```bash
g++ main.cpp -o program -lmath
```

## 1.5 拓展补充
- 关联知识点：预处理器、链接器、标准库、平台ABI
- 进阶延伸：交叉编译、编译器插件、自定义编译工具链、性能分析工具

## 1.6 多文件编程与头文件设计

### 为什么需要多文件编程？

**代码规模问题**：随着项目规模扩大，将所有代码放在单一文件中会变得难以维护和理解。

**模块化优势**：
- **代码组织**：将相关功能分解为独立模块
- **可重用性**：模块可以在不同项目中复用
- **协作开发**：多人可同时开发不同模块
- **编译效率**：只需重新编译修改过的文件

**示例项目结构**：
```
calculator/
├── calculator.cpp     # 主程序
├── addition.cpp       # 加法模块
├── subtraction.cpp    # 减法模块
├── multiplication.cpp # 乘法模块
├── division.cpp       # 除法模块
├── math_utils.h       # 公共头文件
└── Makefile          # 构建脚本
```

### 头文件设计原则

#### 包含保护机制
```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// 头文件内容...

#endif // MATH_UTILS_H
```

**现代 C++ 替代方案**：
```cpp
#pragma once  // 编译器扩展，更简洁但非标准
```

#### 头文件内容规范

**可以放在头文件中的内容**：
- 函数声明
- 类定义（包含成员函数声明）
- 模板定义
- 内联函数定义
- 常量定义
- 类型定义（typedef, using）

**避免放在头文件中的内容**：
- 普通函数定义（会导致重复定义）
- 全局变量定义（使用 extern 声明）
- 大型实现代码

#### 头文件组织示例
```cpp
// math_operations.h
#ifndef MATH_OPERATIONS_H
#define MATH_OPERATIONS_H

#include <iostream>
#include <string>

// 函数声明
int add(int a, int b);
double add(double a, double b);

// 类定义
class Calculator {
private:
    std::string name;

public:
    Calculator(const std::string& calcName);

    // 成员函数声明
    double multiply(double a, double b) const;
    double divide(double a, double b) const;

    // 内联函数定义（可以放在头文件中）
    const std::string& getName() const { return name; }
};

// 常量定义
constexpr double PI = 3.141592653589793;

// 类型别名
using OperationFunc = double(*)(double, double);

#endif // MATH_OPERATIONS_H
```

#### 源文件组织原则
```cpp
// math_operations.cpp
#include "math_operations.h"
#include <stdexcept>
#include <cmath>

// 函数定义
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

// 类成员函数定义
Calculator::Calculator(const std::string& calcName) : name(calcName) {}

double Calculator::multiply(double a, double b) const {
    return a * b;
}

double Calculator::divide(double a, double b) const {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
    }
    return a / b;
}
```

#### 全局变量管理
```cpp
// globals.cpp
#include "math_operations.h"

// 全局变量定义（只能在一个源文件中定义）
int globalCounter = 0;
```

在头文件中声明：
```cpp
// math_operations.h 中添加
extern int globalCounter;  // 声明，不是定义
```

### 最佳实践

1. **单一职责**：每个头文件只负责一个明确的功能
2. **包含最小化**：只包含必要的头文件
3. **前向声明**：使用前向声明减少头文件依赖
4. **接口清晰**：提供清晰的API文档

---

# 2. DEBUG调试技巧

## 2.1 核心概念
- 定义：调试是发现和修复程序错误的过程
- 关键特性：断点设置、变量监控、调用栈跟踪、内存检查

## 2.2 语法规则与使用
- 基本语法：调试器命令、断言宏，日志输出
- 代码示例：
```cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

class DebugExample {
private:
    std::vector<int> data;

public:
    void addNumber(int value) {
        // 使用断言检查前置条件
        assert(value >= 0);  // 确保值非负
        data.push_back(value);
    }

    int getNumberAt(int index) {
        // 边界检查
        if (index < 0 || index >= data.size()) {
            throw std::out_of_range("索引超出范围");
        }
        return data[index];
    }

    void printData() {
        std::cout << "数据内容: ";
        for (size_t i = 0; i < data.size(); i++) {
            std::cout << data[i] << " ";

            // 调试输出 - 只在调试模式下生效
            #ifdef DEBUG
            std::cout << "[索引: " << i << "] ";
            #endif
        }
        std::cout << std::endl;
    }

    // 内存泄漏检测示例
    void memoryLeakExample() {
        int* ptr = new int[100];  // 可能的内存泄漏

        // 使用智能指针避免内存泄漏
        auto smartPtr = std::make_unique<int[]>(100);
    }
};

// 自定义调试宏
#ifdef DEBUG
#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)  // 在发布版本中为空
#endif

int main() {
    DebugExample example;

    try {
        DEBUG_LOG("程序开始执行");

        example.addNumber(10);
        example.addNumber(20);
        example.addNumber(30);

        DEBUG_LOG("添加数字完成");

        example.printData();

        std::cout << "第二个数字: " << example.getNumberAt(1) << std::endl;

        example.memoryLeakExample();

        DEBUG_LOG("程序执行完成");
    }
    catch (const std::exception& e) {
        std::cerr << "捕获异常: " << e.what() << std::endl;
        #ifdef DEBUG
        // 在调试模式下显示更多信息
        std::cerr << "异常类型: " << typeid(e).name() << std::endl;
        #endif
    }

    return 0;
}
```
- 注意事项：调试信息安全性、性能影响、平台兼容性

## 2.3 常见用法

### GDB调试会话
```bash
# 编译带调试信息的程序
g++ -g -o debug_program debug_example.cpp

# 启动GDB
gdb debug_program

# GDB常用命令
(gdb) break main              # 在main函数设置断点
(gdb) run                     # 运行程序
(gdb) next                    # 单步执行
(gdb) step                    # 进入函数
(gdb) print variable          # 打印变量值
(gdb) backtrace              # 显示调用栈
(gdb) watch variable         # 监视变量变化
(gdb) continue               # 继续执行
(gdb) quit                   # 退出GDB
```

### Visual Studio调试技巧
```cpp
// 条件断点示例
void processData(const std::vector<int>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        // 在Visual Studio中设置条件断点: i == 5
        if (data[i] < 0) {
            std::cout << "发现负数: " << data[i] << std::endl;
        }
    }
}
```

### Valgrind内存检测
```bash
# 检测内存泄漏
valgrind --leak-check=full ./program

# 检测未初始化内存
valgrind --track-origins=yes ./program

# 检测线程错误
valgrind --tool=helgrind ./program
```

## 2.4 易错点/坑

### 错误示例1：调试代码留在发布版本中
```cpp
void processSensitiveData(const std::string& password) {
    #ifdef DEBUG
    std::cout << "密码: " << password << std::endl;  // 安全隐患！
    #endif
}
```
- 原因：敏感信息泄露风险
- 修正方案：使用安全的日志记录，避免输出敏感信息

### 错误示例2：断言滥用
```cpp
int divide(int a, int b) {
    assert(b != 0);  // 在发布版本中会被移除
    return a / b;    // 如果b==0，发布版本会崩溃
}
```
- 原因：断言只在调试模式下生效
- 修正方案：对关键检查使用运行时检查
```cpp
int divide(int a, int b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
    }
    return a / b;
}
```

## 2.5 拓展补充
- 关联知识点：异常处理、日志系统、单元测试、性能分析
- 进阶延伸：远程调试、核心转储分析、动态分析工具、代码覆盖率测试

---

# 3. CMake使用指南

## 1. 核心概念
- 定义：CMake是一个跨平台的构建系统生成器，用于管理C++项目的构建过程
- 关键特性：跨平台支持、依赖管理、模块化配置、自动化构建

## 2. 语法规则
- 基本语法：CMakeLists.txt文件结构、命令语法、变量使用
- 代码示例：

### 基础CMakeLists.txt示例
```cmake
# CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称和基本信息
project(MyProject 
    VERSION 1.0.0
    DESCRIPTION "一个示例C++项目"
    LANGUAGES CXX
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译器选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# 添加可执行文件
add_executable(my_app 
    src/main.cpp
    src/utils.cpp
    src/calculator.cpp
)

# 包含目录
target_include_directories(my_app 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# 链接库（如果有）
# target_link_libraries(my_app 
#     PRIVATE 
#         pthread
#         m
# )

# 安装规则（可选）
install(TARGETS my_app 
    RUNTIME DESTINATION bin
)

# 测试支持（如果启用）
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 对应的C++源代码示例
```cpp
// main.cpp - 主程序文件
#include <iostream>
#include "utils.h"
#include "calculator.h"

int main() {
    std::cout << "CMake项目示例" << std::endl;
    
    // 使用工具函数
    std::string message = "Hello, CMake!";
    printMessage(message);
    
    // 使用计算器
    Calculator calc;
    std::cout << "5 + 3 = " << calc.add(5, 3) << std::endl;
    std::cout << "10 - 4 = " << calc.subtract(10, 4) << std::endl;
    
    return 0;
}
```

```cpp
// utils.h - 工具头文件
#ifndef UTILS_H
#define UTILS_H

#include <string>

void printMessage(const std::string& message);

#endif
```

```cpp
// utils.cpp - 工具实现
#include "utils.h"
#include <iostream>

void printMessage(const std::string& message) {
    std::cout << "消息: " << message << std::endl;
}
```

```cpp
// calculator.h - 计算器头文件
#ifndef CALCULATOR_H
#define CALCULATOR_H

class Calculator {
public:
    int add(int a, int b);
    int subtract(int a, int b);
    int multiply(int a, int b);
    double divide(int a, int b);
};

#endif
```

```cpp
// calculator.cpp - 计算器实现
#include "calculator.h"
#include <stdexcept>

int Calculator::add(int a, int b) {
    return a + b;
}

int Calculator::subtract(int a, int b) {
    return a - b;
}

int Calculator::multiply(int a, int b) {
    return a * b;
}

double Calculator::divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error("除数不能为零");
    }
    return static_cast<double>(a) / b;
}
```
- 注意事项：平台兼容性、依赖管理、构建类型选择

## 3. 常见用法
- 场景1：多目录项目结构
```cmake
# 根目录CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyLargeProject)

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(docs)

# 设置全局变量
set(PROJECT_VERSION "1.0.0")
set(CMAKE_CXX_STANDARD 11)
```

```cmake
# src/CMakeLists.txt
# 创建库
add_library(mylib STATIC
    utils.cpp
    calculator.cpp
    network.cpp
)

# 创建可执行文件
add_executable(myapp main.cpp)

# 链接库
target_link_libraries(myapp mylib)

# 包含目录
target_include_directories(mylib PUBLIC include)
```

- 场景2：查找和使用外部库
```cmake
# 查找系统库
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# 使用外部库
target_link_libraries(myapp 
    PRIVATE 
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
)

# 自定义查找库
find_library(MYLIB_LIBRARY mylib)
find_path(MYLIB_INCLUDE_DIR mylib.h)

if(MYLIB_LIBRARY AND MYLIB_INCLUDE_DIR)
    target_link_libraries(myapp ${MYLIB_LIBRARY})
    target_include_directories(myapp PRIVATE ${MYLIB_INCLUDE_DIR})
endif()
```

## 4. 易错点/坑
- 错误示例：
```cmake
# 错误：忘记设置C++标准
project(MyProject)
add_executable(my_app main.cpp)
# 可能导致C++11特性无法使用
```
- 原因：未明确指定C++标准版本
- 修正方案：明确设置C++标准
```cmake
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

- 错误示例：
```cmake
# 错误：错误的包含目录设置
include_directories(include)  # 全局包含，影响所有目标
# 更好的方式：
target_include_directories(my_target PRIVATE include)
```
- 原因：全局包含可能导致命名冲突
- 修正方案：使用目标特定的包含目录

## 5. 拓展补充
- 关联知识点：Makefile、编译器工具链、依赖管理、持续集成
- 进阶延伸：CPack打包、CTest测试、ExternalProject外部项目、自定义命令

---

# 4. Makefile使用指南

## 4.1 核心概念
- 定义：Makefile是一个文本文件，描述了文件之间的依赖关系和构建规则，用于自动化编译过程
- 关键特性：增量编译、依赖管理、自动化构建、跨平台支持

## 4.2 语法规则与使用
- 基本语法：目标、依赖、命令；变量定义；模式匹配
- 代码示例：

### 基本Makefile语法
```makefile
# 注释以 # 开头

# 变量定义
CXX = g++
CXXFLAGS = -Wall -g -std=c++17
TARGET = calculator
SOURCES = main.cpp math_operations.cpp globals.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# 默认目标
all: $(TARGET)

# 链接目标
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJECTS) $(TARGET)

# 伪目标声明
.PHONY: all clean
```

### Makefile自动变量

| 自动变量 | 含义 | 示例 |
|----------|------|------|
| `$@` | 规则的目标文件名 | `$(TARGET)` |
| `$<` | 第一个依赖文件名 | `main.cpp` |
| `$^` | 所有依赖文件 | `main.o math_operations.o` |
| `$?` | 比目标新的依赖文件 | 更新的文件 |
| `$*` | 不包含扩展名的目标文件 | `main` |

## 4.3 常见用法

### 高级Makefile特性

#### 条件判断
```makefile
# 根据调试模式设置不同的编译选项
DEBUG ?= 0

ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -O0
else
    CXXFLAGS += -O2
endif
```

#### 函数使用
```makefile
# 获取目录下所有 .cpp 文件
SOURCES = $(wildcard src/*.cpp)

# 生成对应的 .o 文件列表
OBJECTS = $(patsubst src/%.cpp, build/%.o, $(SOURCES))

# 添加前缀
INCLUDE_DIRS = include lib/include
CXXFLAGS += $(addprefix -I, $(INCLUDE_DIRS))
```

#### 依赖关系生成
```makefile
# 自动生成依赖关系
DEPENDS = $(SOURCES:.cpp=.d)

%.d: %.cpp
	@$(CXX) -MM $(CXXFLAGS) $< > $@
	@sed -i 's/\($*\)\.o[ :]*/\1.o $@ : /g' $@

# 包含依赖文件
-include $(DEPENDS)
```

### 完整项目Makefile示例
```makefile
# 编译器设置
CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -g

# 项目设置
TARGET = calculator
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

# 源文件列表
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))
DEPENDS = $(OBJECTS:.o=.d)

# 包含路径
CXXFLAGS += -I$(INCLUDE_DIR)

# 默认目标
all: $(BUILD_DIR) $(TARGET)

# 创建构建目录
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 链接可执行文件
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)
	@echo "构建完成: $(TARGET)"

# 编译规则
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 依赖关系
-include $(DEPENDS)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp
	@$(CXX) -MM $(CXXFLAGS) $< | sed 's|\(.*\)\.o|$(BUILD_DIR)/\1.o $@|' > $@

# 测试目标
test: $(TARGET)
	@echo "运行测试..."
	@./$(TARGET)

# 清理目标
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo "清理完成"

# 安装目标
install: $(TARGET)
	@cp $(TARGET) /usr/local/bin/
	@echo "安装完成"

# 帮助信息
help:
	@echo "可用目标:"
	@echo "  all      - 构建项目（默认）"
	@echo "  clean    - 清理构建文件"
	@echo "  test     - 运行测试"
	@echo "  install  - 安装到系统"

.PHONY: all clean test install help
```

## 4.4 易错点/坑

### 错误示例1：重复定义
```
multiple definition of `function_name'
```
- 原因：函数定义放在了头文件中
- 修正方案：确保函数定义只在源文件中，头文件中只放声明

### 错误示例2：未定义引用
```
undefined reference to `function_name'
```
- 原因：检查是否链接了所有必要的目标文件

### 错误示例3：循环依赖
```
#include "A.h"  // A.h 包含 B.h
#include "B.h"  // B.h 包含 A.h
```
- 原因：头文件循环包含
- 修正方案：使用前向声明打破循环依赖

## 4.5 拓展补充
- 关联知识点：CMake、编译器工具链、依赖管理、持续集成
- 进阶延伸：Makefile模板、构建系统比较、分布式编译、预编译头文件

# 代码规范

## 1. 核心概念
- 定义：代码规范是保证代码可读性、可维护性和一致性的规则集合
- 关键特性：命名约定、格式标准、注释规范、架构原则

## 2. 语法规则
- 基本语法：遵循特定的编码风格和最佳实践
- 代码示例：

### 符合规范的C++代码示例
```cpp
// math_utils.h - 数学工具头文件
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>
#include <stdexcept>

namespace math {

/// @brief 数学工具类，提供基本数学运算
class MathUtils {
public:
    /// @brief 计算数字列表的平均值
    /// @param numbers 数字列表
    /// @return 平均值
    /// @throws std::invalid_argument 如果列表为空
    static double calculateAverage(const std::vector<double>& numbers);
    
    /// @brief 计算阶乘
    /// @param n 非负整数
    /// @return n的阶乘
    /// @throws std::invalid_argument 如果n为负数
    static long long factorial(int n);
    
    /// @brief 判断数字是否为素数
    /// @param number 要检查的数字
    /// @return 如果是素数返回true，否则返回false
    static bool isPrime(int number);
};

} // namespace math

#endif // MATH_UTILS_H
```

```cpp
// math_utils.cpp - 数学工具实现
#include "math_utils.h"
#include <cmath>
#include <algorithm>

namespace math {

double MathUtils::calculateAverage(const std::vector<double>& numbers) {
    if (numbers.empty()) {
        throw std::invalid_argument("数字列表不能为空");
    }
    
    double sum = 0.0;
    for (double number : numbers) {
        sum += number;
    }
    
    return sum / numbers.size();
}

long long MathUtils::factorial(int n) {
    if (n < 0) {
        throw std::invalid_argument("阶乘只能计算非负整数");
    }
    
    if (n == 0 || n == 1) {
        return 1;
    }
    
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    
    return result;
}

bool MathUtils::isPrime(int number) {
    if (number < 2) {
        return false;
    }
    
    if (number == 2) {
        return true;
    }
    
    if (number % 2 == 0) {
        return false;
    }
    
    int sqrt_num = static_cast<int>(std::sqrt(number));
    for (int i = 3; i <= sqrt_num; i += 2) {
        if (number % i == 0) {
            return false;
        }
    }
    
    return true;
}

} // namespace math
```

```cpp
// main.cpp - 主程序文件，展示规范用法
#include <iostream>
#include <vector>
#include "math_utils.h"

// 使用命名空间别名提高可读性
namespace mu = math;

/// @brief 演示数学工具类的使用
void demonstrateMathUtils() {
    std::vector<double> test_numbers = {1.5, 2.5, 3.5, 4.5, 5.5};
    
    try {
        double average = mu::MathUtils::calculateAverage(test_numbers);
        std::cout << "平均值: " << average << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "计算平均值错误: " << e.what() << std::endl;
    }
    
    // 测试素数判断
    std::vector<int> test_primes = {2, 3, 17, 25, 29};
    for (int num : test_primes) {
        bool is_prime = mu::MathUtils::isPrime(num);
        std::cout << num << " 是" << (is_prime ? "" : "不") << "素数" << std::endl;
    }
}

/// @brief 演示智能指针和RAII的使用
void demonstrateSmartPointers() {
    // 使用智能指针自动管理资源
    auto numbers = std::make_unique<std::vector<int>>();
    numbers->push_back(1);
    numbers->push_back(2);
    numbers->push_back(3);
    
    // 使用范围for循环
    for (const auto& num : *numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 智能指针自动释放内存
}

int main() {
    std::cout << "=== C++代码规范示例 ===" << std::endl;
    
    demonstrateMathUtils();
    demonstrateSmartPointers();
    
    return 0;
}
```
- 注意事项：一致性、可读性、可维护性、性能考虑

## 3. 常见用法
- 场景1：Google C++风格指南示例
```cpp
// 文件名：my_class.h
#ifndef MY_PROJECT_MY_CLASS_H_
#define MY_PROJECT_MY_CLASS_H_

#include <string>
#include <vector>

namespace my_project {

// 类名使用大驼峰命名法
class MyClass {
public:
    // 构造函数使用explicit避免隐式转换
    explicit MyClass(const std::string& name);
    
    // 方法名使用小驼峰命名法
    void doSomethingImportant();
    
    // 常量使用k前缀
    static constexpr int kDefaultSize = 100;
    
private:
    // 成员变量使用下划线后缀
    std::string name_;
    std::vector<int> data_;
};

} // namespace my_project

#endif // MY_PROJECT_MY_CLASS_H_
```

- 场景2：现代C++最佳实践
```cpp
#include <memory>
#include <algorithm>
#include <type_traits>

// 使用auto和类型推导
void modernCppExamples() {
    // 使用auto避免冗长的类型声明
    auto numbers = std::vector<int>{1, 2, 3, 4, 5};
    
    // 使用lambda表达式
    auto is_even = [](int n) { return n % 2 == 0; };
    
    // 使用算法和函数式编程
    auto even_numbers = std::vector<int>{};
    std::copy_if(numbers.begin(), numbers.end(), 
                std::back_inserter(even_numbers), is_even);
    
    // 使用结构化绑定（C++17）
    std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << std::endl;
    }
}

// 使用移动语义优化性能
class ResourceManager {
private:
    std::unique_ptr<int[]> data_;
    size_t size_;
    
public:
    // 移动构造函数
    ResourceManager(ResourceManager&& other) noexcept
        : data_(std::move(other.data_))
        , size_(other.size_) {
        other.size_ = 0;
    }
    
    // 移动赋值运算符
    ResourceManager& operator=(ResourceManager&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
};
```

## 4. 易错点/坑
- 错误示例：
```cpp
// 不符合规范的代码
class badclass{  // 缺少空格，类名不规范
    int x,y;     // 多个变量声明在一行
    void badmethod(){  // 括号位置不规范
    if(x==0){    // 缺少空格
    cout<<"error";  // 使用C风格输出
    }
    }
};
```
- 原因：格式混乱，命名不规范，使用过时特性
- 修正方案：遵循一致的代码风格

- 错误示例：
```cpp
// 内存管理错误
void memoryLeak() {
    int* ptr = new int[100];
    // 使用ptr...
    // 忘记delete[] ptr;  // 内存泄漏！
}
```
- 原因：手动内存管理容易出错
- 修正方案：使用智能指针
```cpp
void safeMemoryManagement() {
    auto ptr = std::make_unique<int[]>(100);
    // 自动管理内存
}
```

## 5. 拓展补充
- 关联知识点：设计模式、重构技术、代码审查、静态分析
- 进阶延伸：领域特定语言、代码生成工具、自动化代码格式化、代码质量度量



# 常见错误与坑

## 1. 核心概念
- 定义：整理C++编程中常见的错误类型和容易踩的坑
- 关键特性：编译错误、运行时错误、逻辑错误、内存错误

## 2. 语法规则
- 基本语法：错误识别、调试技巧、预防措施
- 代码示例：

### 常见错误示例集合
```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

class CommonErrors {
public:
    // 错误1：未初始化变量
    static void uninitializedVariable() {
        int x;  // 未初始化
        std::cout << "x = " << x << std::endl;  // 未定义行为
        
        // 修正：总是初始化变量
        int y = 0;
        std::cout << "y = " << y << std::endl;
    }
    
    // 错误2：数组越界
    static void arrayOutOfBounds() {
        int arr[5] = {1, 2, 3, 4, 5};
        // std::cout << arr[5] << std::endl;  // 越界访问
        
        // 修正：使用边界检查或标准容器
        std::vector<int> vec = {1, 2, 3, 4, 5};
        if (vec.size() > 5) {
            std::cout << vec[5] << std::endl;
        } else {
            std::cout << "索引越界" << std::endl;
        }
    }
    
    // 错误3：内存泄漏
    static void memoryLeakExample() {
        int* ptr = new int[100];  // 分配内存
        // 使用ptr...
        // 忘记 delete[] ptr;  // 内存泄漏！
        
        // 修正：使用智能指针
        auto smartPtr = std::make_unique<int[]>(100);
        // 自动释放内存
    }
    
    // 错误4：悬空指针
    static void danglingPointer() {
        int* ptr = nullptr;
        {
            int x = 10;
            ptr = &x;  // ptr指向局部变量x
        }  // x的生命周期结束
        // std::cout << *ptr << std::endl;  // 悬空指针，未定义行为
        
        // 修正：确保指针有效性
        int y = 20;
        ptr = &y;  // ptr指向有效的变量
        std::cout << *ptr << std::endl;
    }
    
    // 错误5：除零错误
    static void divisionByZero() {
        int a = 10, b = 0;
        // int result = a / b;  // 除零错误
        
        // 修正：检查除数
        if (b != 0) {
            int result = a / b;
            std::cout << "结果: " << result << std::endl;
        } else {
            std::cout << "错误：除数不能为零" << std::endl;
        }
    }
    
    // 错误6：字符串操作错误
    static void stringErrors() {
        char str[10] = "hello";
        // strcpy(str, "这是一个很长的字符串");  // 缓冲区溢出
        
        // 修正：使用安全的字符串函数或std::string
        std::string safeStr = "hello";
        safeStr = "这是一个很长的字符串";  // 自动处理内存
        std::cout << safeStr << std::endl;
    }
    
    // 错误7：类型转换错误
    static void typeConversionErrors() {
        double d = 3.14;
        // int* p = (int*)&d;  // 危险的类型转换
        // std::cout << *p << std::endl;  // 未定义行为
        
        // 修正：使用安全的类型转换
        int i = static_cast<int>(d);  // 安全转换
        std::cout << i << std::endl;
    }
    
    // 错误8：忘记返回值
    static int forgetReturnValue() {
        int x = 10;
        // 忘记 return x;  // 编译警告，运行时未定义行为
        
        // 修正：确保所有路径都有返回值
        return x;
    }
};

// 错误9：循环中的错误
void loopErrors() {
    // 无限循环
    // while (true) {
    //     // 忘记退出条件
    // }
    
    // 修正：明确的退出条件
    int count = 0;
    while (count < 10) {
        std::cout << count++ << " ";
    }
    std::cout << std::endl;
}

// 错误10：多线程竞争条件
#include <thread>
#include <mutex>

class ThreadSafeCounter {
private:
    int count = 0;
    std::mutex mtx;
    
public:
    // 错误：非线程安全
    void unsafeIncrement() {
        count++;  // 竞争条件
    }
    
    // 修正：使用互斥锁
    void safeIncrement() {
        std::lock_guard<std::mutex> lock(mtx);
        count++;
    }
    
    int getCount() {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
};

int main() {
    std::cout << "=== 常见错误与修正示例 ===" << std::endl;
    
    CommonErrors::uninitializedVariable();
    CommonErrors::arrayOutOfBounds();
    CommonErrors::memoryLeakExample();
    CommonErrors::danglingPointer();
    CommonErrors::divisionByZero();
    CommonErrors::stringErrors();
    CommonErrors::typeConversionErrors();
    std::cout << "返回值: " << CommonErrors::forgetReturnValue() << std::endl;
    
    loopErrors();
    
    // 测试线程安全
    ThreadSafeCounter counter;
    std::thread t1([&counter]() {
        for (int i = 0; i < 1000; i++) {
            counter.safeIncrement();
        }
    });
    
    std::thread t2([&counter]() {
        for (int i = 0; i < 1000; i++) {
            counter.safeIncrement();
        }
    });
    
    t1.join();
    t2.join();
    
    std::cout << "最终计数: " << counter.getCount() << std::endl;
    
    return 0;
}
```
- 注意事项：错误识别、调试工具使用、预防性编程

## 3. 常见用法
- 场景1：使用编译器警告和静态分析
```bash
# 启用所有警告
g++ -Wall -Wextra -Wpedantic -o program main.cpp

# 将警告视为错误
g++ -Werror -o program main.cpp

# 使用Clang静态分析器
clang++ --analyze main.cpp

# 使用Cppcheck进行静态分析
cppcheck --enable=all main.cpp
```

- 场景2：运行时错误检测工具
```bash
# 使用Valgrind检测内存错误
valgrind --tool=memcheck ./program

# 检测内存泄漏
valgrind --leak-check=full ./program

# 检测未初始化内存
valgrind --track-origins=yes ./program

# 使用AddressSanitizer
g++ -fsanitize=address -g -o program main.cpp
./program
```

## 4. 易错点/坑
- 错误示例：
```cpp
// 错误：混淆=和==
if (x = 5) {  // 应该是 if (x == 5)
    // 总是执行，因为赋值表达式返回5（真）
}
```
- 原因：运算符优先级和类型混淆
- 修正方案：使用明确的比较，启用编译器警告

- 错误示例：
```cpp
// 错误：虚函数析构函数
class Base {
public:
    ~Base() { }  // 应该是 virtual ~Base() { }
};

class Derived : public Base {
    std::vector<int> data;
public:
    ~Derived() { }  // 可能不会调用
};

Base* ptr = new Derived();
delete ptr;  // 只调用Base的析构函数，内存泄漏
```
- 原因：多态基类缺少虚析构函数
- 修正方案：为多态基类声明虚析构函数

## 5. 拓展补充
- 关联知识点：异常处理、调试技巧、代码审查、测试驱动开发
- 进阶延伸：性能分析、安全编程、代码质量工具、持续集成