# C++ 多文件管理与 Makefile 工程

## 课程目标
- 理解 C++ 多文件编程的必要性和优势
- 掌握头文件和源文件的组织方式
- 学会编写和使用 Makefile 自动化构建项目
- 掌握多文件项目的编译和链接过程
- 能够组织和管理中小型 C++ 项目

## <span style="background-color: #ce2183ff; padding: 2px 4px; border-radius: 3px; color: #333;">命名空间（Namespace）</span>

### 什么是命名空间？

命名空间是C++中用于组织代码、防止命名冲突的重要机制。它将相关的类、函数、变量等封装在一个逻辑分组中，避免不同库或模块中的同名标识符产生冲突。

### 命名空间的基本语法

#### 1. 定义命名空间
```cpp
namespace MyNamespace {
    // 在命名空间内定义变量、函数、类等
    int value = 42;
    
    void myFunction() {
        std::cout << "Hello from MyNamespace!" << std::endl;
    }
    
    class MyClass {
    public:
        void display() {
            std::cout << "MyClass in MyNamespace" << std::endl;
        }
    };
}

// 命名空间可以嵌套
namespace Outer {
    namespace Inner {
        void innerFunction() {
            std::cout << "Nested namespace function" << std::endl;
        }
    }
}
```

#### 2. 使用命名空间中的成员
```cpp
#include <iostream>

namespace Math {
    const double PI = 3.14159;
    
    double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    namespace Geometry {
        struct Point {
            double x, y;
        };
        
        double distance(Point p1, Point p2) {
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            return sqrt(dx * dx + dy * dy);
        }
    }
}

int main() {
    // 方式1：使用作用域解析运算符 ::
    std::cout << "PI = " << Math::PI << std::endl;
    std::cout << "Area = " << Math::circleArea(5.0) << std::endl;
    
    Math::Geometry::Point p1 = {0, 0};
    Math::Geometry::Point p2 = {3, 4};
    std::cout << "Distance = " << Math::Geometry::distance(p1, p2) << std::endl;
    
    // 方式2：使用using声明（引入特定成员）
    using Math::PI;
    using Math::circleArea;
    
    std::cout << "PI = " << PI << std::endl;  // 直接使用，无需Math::
    std::cout << "Area = " << circleArea(3.0) << std::endl;
    
    // 方式3：使用using namespace（引入整个命名空间）
    using namespace Math::Geometry;
    
    Point p3 = {1, 1};
    Point p4 = {4, 5};
    std::cout << "Distance = " << distance(p3, p4) << std::endl;
    
    return 0;
}
```

### 标准命名空间 `std`

C++标准库中的所有组件都定义在`std`命名空间中，这是为了避免与用户定义的标识符冲突。

```cpp
#include <iostream>
#include <string>
#include <vector>

// 良好的做法：在函数内部或局部使用using
void goodPractice() {
    // 局部使用using namespace
    using namespace std;
    
    string name = "Alice";
    vector<int> numbers = {1, 2, 3};
    cout << "Name: " << name << endl;
    
    // 或者使用using声明（更安全）
    using std::string;
    using std::vector;
    using std::cout;
    using std::endl;
    
    string greeting = "Hello";
    vector<double> prices = {1.99, 2.49, 0.99};
    cout << greeting << endl;
}

// 最佳实践：明确使用std::前缀
void bestPractice() {
    std::string message = "Best practice";
    std::vector<std::string> words = {"hello", "world"};
    
    for (const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}

// 避免的做法：全局使用using namespace std
// using namespace std;  // 不推荐在全局使用

int main() {
    goodPractice();
    bestPractice();
    return 0;
}
```

### 匿名命名空间

匿名命名空间用于定义只在当前文件内可见的标识符，类似于C中的`static`关键字。

```cpp
// file1.cpp
namespace {  // 匿名命名空间
    int internalVariable = 100;  // 只在当前文件内可见
    
    void internalFunction() {
        std::cout << "Internal function" << std::endl;
    }
}

void publicFunction() {
    internalFunction();  // 可以在同一文件内访问
    std::cout << "Internal variable: " << internalVariable << std::endl;
}

// file2.cpp（另一个文件）
namespace {
    int internalVariable = 200;  // 这是不同的变量，不会冲突
}

void anotherFunction() {
    // internalFunction();  // 错误：无法访问file1.cpp中的匿名命名空间
    std::cout << "Different internal variable: " << internalVariable << std::endl;
}
```

### 命名空间别名

当命名空间名称过长时，可以创建别名来简化代码。

```cpp
#include <iostream>

namespace VeryLongNamespaceName {
    void importantFunction() {
        std::cout << "Important function" << std::endl;
    }
}

// 创建命名空间别名
namespace VLN = VeryLongNamespaceName;

// 标准库别名示例
namespace fs = std::filesystem;  // C++17文件系统别名

int main() {
    // 使用别名
    VLN::importantFunction();  // 等价于 VeryLongNamespaceName::importantFunction()
    
    // 标准库别名使用
    // fs::path filePath = "example.txt";  // C++17特性
    
    return 0;
}
```

### 内联命名空间（C++11）

内联命名空间用于版本控制，允许新旧API共存。

```cpp
#include <iostream>

namespace Library {
    // 版本1的API
    namespace v1 {
        void process(int x) {
            std::cout << "v1 processing: " << x << std::endl;
        }
    }
    
    // 版本2的API（当前版本）
    inline namespace v2 {
        void process(int x) {
            std::cout << "v2 processing: " << x * 2 << std::endl;
        }
        
        void newFeature() {
            std::cout << "New feature in v2" << std::endl;
        }
    }
}

int main() {
    // 默认使用内联命名空间（v2）
    Library::process(10);  // 调用v2::process
    Library::newFeature(); // 调用v2::newFeature
    
    // 明确指定版本
    Library::v1::process(10);  // 调用v1::process
    Library::v2::process(10);  // 调用v2::process
    
    return 0;
}
```

### 命名空间的最佳实践

#### 1. 使用建议
```cpp
// ✅ 推荐做法：明确使用std::前缀
void goodCode() {
    std::vector<std::string> names;
    std::cout << "Good practice" << std::endl;
}

// ❌ 避免做法：全局使用using namespace
// using namespace std;  // 不推荐

void badCode() {
    vector<string> names;  // 可能产生命名冲突
    cout << "Bad practice" << endl;
}

// ✅ 局部使用using声明（安全）
void safeCode() {
    using std::vector;
    using std::string;
    using std::cout;
    using std::endl;
    
    vector<string> safeNames;
    cout << "Safe practice" << endl;
}
```

#### 2. 项目中的命名空间组织
```cpp
// 大型项目中的命名空间组织示例
namespace MyCompany {
    namespace ProjectName {
        namespace Core {
            class Database {
                // 核心数据库功能
            };
        }
        
        namespace UI {
            class Window {
                // 用户界面组件
            };
        }
        
        namespace Utils {
            // 工具函数和辅助类
            template<typename T>
            class Singleton {
                // 单例模式实现
            };
        }
    }
}

// 使用示例
void useNamespaces() {
    MyCompany::ProjectName::Core::Database db;
    MyCompany::ProjectName::UI::Window window;
    
    // 使用别名简化
    namespace MP = MyCompany::ProjectName;
    MP::Core::Database anotherDb;
}
```

### 命名空间与头文件

在头文件中使用命名空间时需要特别小心：

```cpp
// mylibrary.h
#ifndef MYLIBRARY_H
#define MYLIBRARY_H

#include <string>

// ✅ 在头文件中定义命名空间是安全的
namespace MyLibrary {
    class Calculator {
    public:
        static int add(int a, int b);
        static double divide(double a, double b);
    };
    
    const std::string VERSION = "1.0.0";
}

// ❌ 不要在头文件中使用using namespace
// using namespace std;  // 绝对避免！

// ✅ 可以在头文件中使用using声明（但要谨慎）
// using std::string;    // 谨慎使用，可能影响包含该头文件的所有文件

#endif

// mylibrary.cpp
#include "mylibrary.h"

namespace MyLibrary {
    int Calculator::add(int a, int b) {
        return a + b;
    }
    
    double Calculator::divide(double a, double b) {
        if (b == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return a / b;
    }
}

// main.cpp
#include "mylibrary.h"
#include <iostream>

int main() {
    // 使用命名空间中的类
    int result = MyLibrary::Calculator::add(10, 20);
    std::cout << "10 + 20 = " << result << std::endl;
    
    std::cout << "Library version: " << MyLibrary::VERSION << std::endl;
    
    return 0;
}
```

### 总结

命名空间是C++中管理代码组织和防止命名冲突的重要工具：

1. **基本用法**：使用`namespace`关键字定义，通过`::`访问成员
2. **标准库**：所有标准库组件都在`std`命名空间中
3. **最佳实践**：避免全局`using namespace`，优先使用明确的前缀
4. **高级特性**：匿名命名空间、内联命名空间、命名空间别名
5. **头文件规则**：在头文件中谨慎使用`using`声明

通过合理使用命名空间，可以编写出更加清晰、可维护和不易冲突的C++代码。




### 命名空间

命名空间是 C++ 中用于**组织代码**和**避免命名冲突**的机制。

#### 为什么需要命名空间？

```cpp
// 问题示例：命名冲突
#include <iostream>
using namespace std;

int count = 5;  // 全局变量

void count() {  // 函数名冲突！
    cout << "Hello";
}

int main() {
    cout << count;  // 错误：count 是变量还是函数？
}
```

#### 解决方案：命名空间

```cpp
#include <iostream>
namespace MySpace {
    int count = 5;
    void print() {
        std::cout << "Hello from MySpace";
    }
}

int main() {
    std::cout << MySpace::count;  // 明确指定命名空间
    MySpace::print();
}
```

#### 定义命名空间

**基本语法：**
```cpp
namespace namespace_name {
    // 变量、函数、类等声明
}
```

**示例：**
```cpp
namespace Math {
    const double PI = 3.14159;
    
    double add(double a, double b) {
        return a + b;
    }
    
    class Calculator {
    public:
        double multiply(double x, double y) {
            return x * y;
        }
    };
}
```

#### 使用命名空间

**方法 1：完全限定名（推荐）**
```cpp
#include <iostream>
int main() {
    std::cout << Math::PI;  // 明确指定命名空间
    Math::Calculator calc;
    std::cout << calc.multiply(3, 4);
}
```

**方法 2：using 声明（引入特定成员）**
```cpp
#include <iostream>
using Math::PI;  // 只引入 PI
using Math::Calculator;  // 只引入 Calculator 类

int main() {
    std::cout << PI;  // 直接使用
    Calculator calc;
    std::cout << calc.multiply(3, 4);
}
```

**方法 3：using 指令（引入整个命名空间）**
```cpp
#include <iostream>
using namespace Math;  // 引入所有成员

int main() {
    std::cout << PI;  // 直接使用
    Calculator calc;
    std::cout << calc.multiply(3, 4);
}
```

⚠️ **注意**：在头文件中避免使用 `using` 指令，防止污染全局命名空间

#### 命名空间嵌套

```cpp
namespace Outer {
    int value = 10;
    
    namespace Inner {
        int value = 20;
        void show() {
            std::cout << "Inner value: " << value;
        }
    }
}

int main() {
    std::cout << Outer::value;        // 10
    std::cout << Outer::Inner::value; // 20
    Outer::Inner::show();             // "Inner value: 20"
}
```

#### 命名空间别名

为长命名空间创建别名：
```cpp
namespace VeryLongNamespaceName {
    void func() { std::cout << "Hello!"; }
}

// 创建别名
namespace VLNN = VeryLongNamespaceName;

int main() {
    VLNN::func();  // 使用别名
}
```

#### 标准命名空间 `std`

C++ 标准库的所有内容都在 `std` 命名空间中：
```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::string name = "Alice";
    std::vector<int> scores = {90, 85, 88};
    std::cout << "Name: " << name;
}
```

**常见用法：**
```cpp
// 方法 1：完全限定名（推荐）
std::cout << "Hello";

// 方法 2：using 声明（常用）
using std::cout;
using std::endl;
cout << "Hello" << endl;

// 方法 3：using 指令（仅限小型程序）
using namespace std;  // 简单但可能引起冲突
```

#### 命名空间最佳实践

1. **避免全局 `using`**：在头文件中绝不使用 `using namespace std;`
2. **优先使用完全限定名**：明确代码来源，提高可读性
3. **合理使用 `using` 声明**：在函数内部引入常用名称
4. **组织代码**：将相关功能放在同一命名空间
5. **命名规范**：使用有意义的命名空间名称（如 `Company::Project::Module`）

> **编程提示**：在团队协作中，命名空间是组织代码的重要工具，就像给文件分类放入不同文件夹一样！

## 1. 多文件编程概述

### 1.1 为什么需要多文件编程？

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

### 1.2 头文件与源文件的角色分工

#### 头文件 (.h/.hpp)
- **声明接口**：函数、类、变量的声明
- **提供接口信息**：给其他文件使用
- **避免重复包含**：使用包含保护
- **不包含实现**：避免重复定义错误

#### 源文件 (.cpp)
- **包含头文件**：使用 `#include` 指令
- **实现功能**：定义函数、类、变量的具体实现
- **独立编译**：每个源文件独立编译为目标文件

## 2. 头文件设计原则

### 2.1 包含保护机制

**问题**：多次包含同一个头文件会导致重复定义错误

**解决方案**：使用预处理指令实现包含保护

```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// 头文件内容...

#endif // MATH_UTILS_H
```

**现代 C++ 替代方案**（C++23）：
```cpp
#pragma once  // 编译器扩展，更简洁但非标准
```

### 2.2 头文件内容规范

#### 可以放在头文件中的内容：
- 函数声明
- 类定义（包含成员函数声明）
- 模板定义
- 内联函数定义
- 常量定义
- 类型定义（typedef, using）

#### 避免放在头文件中的内容：
- 普通函数定义（会导致重复定义）
- 全局变量定义（使用 extern 声明）
- 大型实现代码

### 2.3 头文件组织示例

```cpp
// math_operations.h
#ifndef MATH_OPERATIONS_H
#define MATH_OPERATIONS_H

#include <iostream>
#include <string>

// 函数声明
int add(int a, int b);
double add(double a, double b);
int subtract(int a, int b);
double subtract(double a, double b);

// 类定义
class Calculator {
private:
    std::string name;
    
public:
    Calculator(const std::string& calcName);
    
    // 成员函数声明
    double multiply(double a, double b) const;
    double divide(double a, double b) const;
    void displayInfo() const;
    
    // 内联函数定义（可以放在头文件中）
    const std::string& getName() const { return name; }
};

// 常量定义
constexpr double PI = 3.141592653589793;
constexpr int MAX_OPERATIONS = 1000;

// 类型别名
using OperationFunc = double(*)(double, double);

#endif // MATH_OPERATIONS_H
```

## 3. 源文件实现

### 3.1 源文件组织原则

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

int subtract(int a, int b) {
    return a - b;
}

double subtract(double a, double b) {
    return a - b;
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

void Calculator::displayInfo() const {
    std::cout << "计算器名称: " << name << std::endl;
}
```

### 3.2 主程序文件

```cpp
// main.cpp
#include "math_operations.h"
#include <iostream>
#include <vector>

// 全局变量声明（在头文件中使用 extern）
extern int globalCounter;

// 函数原型
void demonstrateOperations();
void demonstrateCalculator();

int main() {
    std::cout << "=== C++ 多文件编程演示 ===" << std::endl;
    
    demonstrateOperations();
    demonstrateCalculator();
    
    std::cout << "全局计数器: " << globalCounter << std::endl;
    
    return 0;
}

void demonstrateOperations() {
    std::cout << "\n--- 基本运算演示 ---" << std::endl;
    
    std::cout << "整数加法: " << add(5, 3) << std::endl;
    std::cout << "浮点数加法: " << add(2.5, 3.7) << std::endl;
    std::cout << "减法: " << subtract(10, 4) << std::endl;
    std::cout << "PI 值: " << PI << std::endl;
}

void demonstrateCalculator() {
    std::cout << "\n--- 计算器类演示 ---" << std::endl;
    
    Calculator calc("科学计算器");
    calc.displayInfo();
    
    std::cout << "乘法: " << calc.multiply(6, 7) << std::endl;
    std::cout << "除法: " << calc.divide(15, 3) << std::endl;
    
    try {
        std::cout << "除以零测试: " << calc.divide(10, 0) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "捕获异常: " << e.what() << std::endl;
    }
}
```

### 3.3 全局变量管理

```cpp
// globals.cpp
#include "math_operations.h"

// 全局变量定义（只能在一个源文件中定义）
int globalCounter = 0;

// 其他全局功能...
```

在头文件中声明：
```cpp
// math_operations.h 中添加
extern int globalCounter;  // 声明，不是定义
```

## 4. 编译和链接过程

### 4.1 手动编译步骤

**1. 编译源文件为目标文件**：
```bash
g++ -c main.cpp -o main.o
g++ -c math_operations.cpp -o math_operations.o
g++ -c globals.cpp -o globals.o
```

**2. 链接目标文件为可执行文件**：
```bash
g++ main.o math_operations.o globals.o -o calculator
```

**3. 运行程序**：
```bash
./calculator
```

### 4.2 编译过程详解

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

### 4.3 编译选项说明

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

## 5. Makefile 自动化构建

### 5.1 Makefile 基本概念

**Makefile** 是一个文本文件，描述了：
- 文件之间的依赖关系
- 生成目标文件所需的命令
- 构建规则和顺序

### 5.2 基本 Makefile 语法

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

### 5.3 Makefile 自动变量

| 自动变量 | 含义 | 示例 |
|----------|------|------|
| `$@` | 规则的目标文件名 | `$(TARGET)` |
| `$<` | 第一个依赖文件名 | `main.cpp` |
| `$^` | 所有依赖文件 | `main.o math_operations.o` |
| `$?` | 比目标新的依赖文件 | 更新的文件 |
| `$*` | 不包含扩展名的目标文件 | `main` |

### 5.4 高级 Makefile 特性

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

## 6. 完整项目示例

### 6.1 项目目录结构

```
calculator_project/
├── include/
│   ├── math_operations.h
│   └── utils.h
├── src/
│   ├── main.cpp
│   ├── math_operations.cpp
│   ├── utils.cpp
│   └── globals.cpp
├── build/
├── lib/
├── tests/
│   └── test_math.cpp
├── Makefile
└── README.md
```

### 6.2 完整 Makefile 示例

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

### 6.3 测试文件

```cpp
// tests/test_math.cpp
#include "../include/math_operations.h"
#include <cassert>
#include <iostream>

void testAddition() {
    assert(add(2, 3) == 5);
    assert(add(-1, 1) == 0);
    std::cout << "加法测试通过" << std::endl;
}

void testCalculator() {
    Calculator calc("测试计算器");
    assert(calc.multiply(3, 4) == 12);
    assert(calc.divide(10, 2) == 5);
    std::cout << "计算器测试通过" << std::endl;
}

int main() {
    testAddition();
    testCalculator();
    std::cout << "所有测试通过!" << std::endl;
    return 0;
}
```

## 7. 现代构建工具简介

### 7.1 CMake

**CMakeLists.txt 示例**：
```cmake
cmake_minimum_required(VERSION 3.10)
project(Calculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(calculator
    src/main.cpp
    src/math_operations.cpp
    src/utils.cpp
)

# 包含目录
target_include_directories(calculator PRIVATE include)

# 编译选项
target_compile_options(calculator PRIVATE -Wall -Wextra)
```

### 7.2 Meson

**meson.build 示例**：
```txt
project('calculator', 'cpp',
        version: '1.0',
        default_options: ['cpp_std=c++17'])

src_files = [
    'src/main.cpp',
    'src/math_operations.cpp',
    'src/utils.cpp'
]

inc_dirs = include_directories('include')

executable('calculator',
           src_files,
           include_directories: inc_dirs,
           install: true)
```

## 8. 最佳实践与常见问题

### 8.1 头文件设计最佳实践

1. **单一职责**：每个头文件只负责一个明确的功能
2. **包含最小化**：只包含必要的头文件
3. **前向声明**：使用前向声明减少头文件依赖
4. **接口清晰**：提供清晰的API文档

### 8.2 编译优化技巧

1. **预编译头文件**：加速编译过程
2. **增量编译**：只编译修改过的文件
3. **并行编译**：使用 `make -j4` 加速构建
4. **分布式编译**：使用 distcc 或 icecc

### 8.3 常见错误与解决方案

**错误1：重复定义**
```
multiple definition of `function_name'
```
**解决方案**：确保函数定义只在源文件中，头文件中只放声明

**错误2：未定义引用**
```
undefined reference to `function_name'
```
**解决方案**：检查是否链接了所有必要的目标文件

**错误3：循环依赖**
```
#include "A.h"  // A.h 包含 B.h
#include "B.h"  // B.h 包含 A.h
```
**解决方案**：使用前向声明打破循环依赖

## 9. 总结

通过本课程的学习，您应该能够：

✅ **理解多文件编程的优势和组织原则**
✅ **掌握头文件和源文件的设计规范**  
✅ **熟练使用 Makefile 自动化构建项目**
✅ **理解编译和链接的完整过程**
✅ **能够组织和管理中小型 C++ 项目**

**进阶学习方向**：
- 学习 CMake、Meson 等现代构建系统
- 掌握大型项目的模块化设计
- 了解持续集成和自动化测试
- 探索包管理和依赖管理工具

多文件管理和构建系统是 C++ 工程化开发的基础，掌握这些技能将显著提升您的开发效率和代码质量。