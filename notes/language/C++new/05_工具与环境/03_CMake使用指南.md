# CMake使用指南

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