# 编译器（GCC/Clang/MSVC）

## 1. 核心概念
- 定义：编译器是将C++源代码转换为可执行程序的工具
- 关键特性：代码优化、错误检查、平台兼容性、标准支持

## 2. 语法规则
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

## 3. 常见用法
- 场景1：GCC编译命令
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

- 场景2：Clang编译命令
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

- 场景3：MSVC编译命令
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

## 4. 易错点/坑
- 错误示例：
```cpp
// 使用编译器特定扩展
int array[10] = {0};
// 某些编译器可能不支持某些初始化语法
```
- 原因：依赖于特定编译器的非标准扩展
- 修正方案：使用标准C++语法，避免编译器特定功能

- 错误示例：
```bash
# 错误的链接顺序
g++ -o program -lmath main.cpp  # 数学库应该在对象文件之后
```
- 原因：链接器参数顺序很重要
- 修正方案：正确排序链接参数
```bash
g++ main.cpp -o program -lmath
```

## 5. 拓展补充
- 关联知识点：预处理器、链接器、标准库、平台ABI
- 进阶延伸：交叉编译、编译器插件、自定义编译工具链、性能分析工具