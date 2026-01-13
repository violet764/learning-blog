# C++基础与核心语法

**C++语言概述**  
C++ 是一种静态类型的、编译式的、通用的、大小写敏感的、不规则的编程语言，支持过程化编程、面向对象编程和泛型编程。


**第一个C++程序**

```cpp
#include <iostream>  
using namespace std;

int main() {  
    std::cout << "Hello, World!" << std::endl;  
    return 0;  
}
```

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

## <span style="background-color: #26e6bcff; padding: 2px 4px; border-radius: 3px; color: #333;">数据类型</span>

C++ 原生支持的基础数据类型，分为数值型、布尔型、空类型，尺寸受编译器/平台影响（以下为64位系统典型值）。

### 基础数据类型 

**整数类型**
| 类型               | 典型尺寸 | 取值范围（典型）| 无符号标识 | 核心用途                     |
|--------------------|----------|-------------------------|------------|------------------------------|
| `char`             | 1字节    | -128 ~ 127              | `unsigned char` | 存储字符/ASCII码/小整数      |
| `short`            | 2字节    | -32768 ~ 32767          | `unsigned short` | 小范围整数，节省内存         |
| `int`              | 4字节    | $-2^{15}$ ~ $2^{15}-1$（2字节）/ $-2^{31}$ ~ $2^{31}-1$（4字节） | `unsigned int` | 最常用整数类型，性能最优     |
| `long`             | 8字节 | $-2^{31}$ ~ $2^{31}-1$（4字节）/ $-2^{63}$ ~ $2^{63}-1$（8字节） | `unsigned long` | 大范围整数                   |
| `long long`        | 8字节    | $-2^{63}$ ~ $2^{63}-1$            | `unsigned long long` | 超大范围整数|

整数类型说明：
- 无符号类型（`unsigned`）仅存储非负数，取值上限翻倍；
- 字面量后缀：`U`（无符号）、`L`（long）、`LL`（long long），如 `100LL`、`200U`；
- 用 `sizeof(类型)`（一个运算符） 获取当前平台实际尺寸，如 `sizeof(int)`。

**浮点类型**  
遵循 IEEE 754 标准，存在精度损失，默认字面量为 `double` 类型。

| 类型         | 典型尺寸 | 有效数字 | 取值范围               | 后缀 | 核心用途                     |
|--------------|----------|----------|------------------------|------|------------------------------|
| `float`      | 4字节    | 6~7位    | ±$10^{-38}$ ~ ±$10^{38}$         | `f`  | 单精度浮点，适合内存受限场景 |
| `double`     | 8字节    | 15~16位  | ±$10^{-308}$ ~ ±$10^{308}$       | -    | 双精度浮点，默认推荐使用     |
| `long double`| 8/16字节 | 18~19位  | ±$10^{-4932}$ ~ ±$10^{4932}$     | `L`  | 扩展精度浮点，极少使用       |

浮点类型注意事项：
- 避免直接用 `==` 比较浮点数，需判断差值小于极小值（如 `1e-6`）；
- 示例：`if (fabs(0.1 + 0.2 - 0.3) < 1e-6)`。

**布尔类型**
| 类型   | 典型尺寸 | 取值       | 核心用途       |
|--------|----------|------------|----------------|
| `bool` | 1字节    | `true`/`false` | 逻辑判断，语义清晰 |

说明：
- `true` 等价于 1，`false` 等价于 0；
- 非0值隐式转换为 `true`，0 转换为 `false`。

**空类型**
| 类型   | 核心用途                                                                 |
|--------|--------------------------------------------------------------------------|
| `void` | 1. 函数返回值：`void func()` 表示无返回值；<br>2. 通用指针：`void*` 指向任意类型（需强制转换） |

### 复合数据类型
基于基本类型扩展，实现更复杂的数据存储逻辑。

**指针（`*`）**
| 语法格式       | 核心特性                                                                 | 示例                     |
|----------------|--------------------------------------------------------------------------|--------------------------|
| `类型* 指针名` | 1. 存储变量内存地址；<br>2. 64位系统尺寸为8字节；<br>3. 需避免野指针/空指针解引用 | `int a=10; int* p=&a;` |

关键操作：
- `&变量名`：取地址；
- `*指针名`：解引用（访问指向的变量）；
- `nullptr`：空指针常量。

**引用（`&`）**
| 语法格式         | 核心特性                                                                 | 示例                     |
|------------------|--------------------------------------------------------------------------|--------------------------|
| `类型& 引用名 = 变量名` | 1. 变量的别名，必须初始化；<br>2. 不可更改指向；<br>3. 无额外内存开销     | `int a=10; int& ref=a;` |

**数组（`[]`）**
| 语法格式               | 核心特性                                                                 | 示例                     |
|------------------------|--------------------------------------------------------------------------|--------------------------|
| `类型 数组名[长度]`    | 1. 相同类型数据的连续集合；<br>2. 静态数组在栈上，动态数组（`new`）在堆上；<br>3. 下标从0开始 | `int arr[5]={1,2,3,4,5};` |

**字符串类型**
| 类型               | 核心特性                                                                 | 示例                     |
|--------------------|--------------------------------------------------------------------------|--------------------------|
| C风格字符串（`char[]/char*`） | 1. 以 `'\0'` 结尾；<br>2. 需手动管理内存                                 | `char str[]="hello";`    |
| C++风格字符串（`std::string`） | 1. 标准库封装；<br>2. 支持拼接/查找/替换等便捷操作；<br>3. 自动管理内存   | `string s="world"; s+="!";` |

**C风格字符串详细操作：**
```cpp
#include <cstring>

char greeting[20];
strcpy(greeting, "Hello");      // 复制字符串
strcat(greeting, " World");     // 连接字符串
cout << greeting << endl;       // 输出Hello World
```

**`std::string`详细操作：**
```cpp
#include <string>

string str1 = "Hello";
string str2 = "World";
string greeting = str1 + " " + str2;     // 字符串拼接
cout << "长度：" << greeting.length() << endl;     // 输出11
cout << "是否为空：" << greeting.empty() << endl;  // 输出0（false）

// 子字符串操作
string sub = greeting.substr(0, 5);    // 提取前5个字符
cout << "子字符串：" << sub << endl;    // 输出Hello

// 查找操作
size_t pos = greeting.find("World");
if (pos != string::npos) {
    cout << "World found at position: " << pos << endl;
} // 输出World found at position: 6


```


### 自定义数据类型
开发者基于业务需求封装的新类型。核心包括结构体（struct）、枚举（enum）、类（class）、联合体（union），还支持 typedef/using 重命名类型，是模块化、面向对象编程的基础。

**结构体（`struct`）**  
用于封装多个不同类型的变量（数据成员）  
语法格式：
```cpp
struct 结构体名 {
    // 数据成员（默认public）
    类型 成员名1;
    类型 成员名2;
    // 成员函数（C++扩展）
    返回值类型 函数名(参数列表) { /* 逻辑 */ }
}; // 末尾分号不可省略
```
示例：  
```cpp
struct Student {
    std::string name;
    int id;
    float score;
};

struct Student {
    string name;
    int age;
    float score;

    // 成员函数：打印信息
    void show() {
        cout << "姓名：" << name << "，年龄：" << age << "，成绩：" << score << endl;
    }

    // 构造函数（初始化成员）
    Student(string n, int a, float s) : name(n), age(a), score(s) {}
};
```

::: tip
在内存占用上存在字节对齐现象，字节对齐是一种内存布局优化，通过在数据间插入填充字节，保证数据起始地址满足对齐模数要求（默认等于数据类型大小），同时复合类型整体大小满足对齐模数整数倍，以此提升内存访问性能。
:::  
**枚举（`enum`）**

用于定义一组命名的整数常量，分为「普通枚举（enum）」和「强类型枚举（enum class）」，避免魔法数字。

**普通枚举：**
```cpp
// 定义枚举：颜色（默认值从0开始，依次+1）
enum Color {
    Red,    // 0
    Green,  // 1
    Blue    // 2
};

// 自定义初始值
enum Week {
    Mon = 1,
    Tue,    // 2
    Wed     // 3
};
```
**强枚举：**  
解决普通枚举的「作用域污染」和「类型不安全」问题

```cpp
// 强类型枚举：作用域限定，类型安全
enum class Direction {
    Left,
    Right,
    Up,
    Down
};
```

**联合体（`union`）**  

所有成员共享同一块内存空间（大小为最大成员的大小），用于节省内存，适合 “互斥使用的成员” 场景。
```cpp
// 定义联合体：存储不同类型，但同一时间仅能使用一个成员
union Data {
    int i;
    float f;
    char c;
}; // 大小为4字节（float/int的大小）

int main() {
    Data d;
    d.i = 10;
    cout << "int值：" << d.i << endl; // 10
    // 覆盖内存：此时i的值失效
    d.f = 3.14f;
    cout << "float值：" << d.f << endl; // 3.14
    cout << "覆盖后int值：" << d.i << endl; // 随机值（内存被覆盖）
    return 0;
}
```

## 常量与变量

### 变量：可修改的数据存储

**变量定义与特性：** 变量是程序中可以修改的数据存储单元，具有名称、类型和值三个基本属性。

```cpp
#include <iostream>
using namespace std;

int main() {
    // 变量定义与初始化
    int age = 25;                    // 定义并初始化
    double salary;                   // 定义未初始化（值不确定）
    salary = 5000.0;                 // 后续赋值
    
    // 不同类型变量的定义
    char grade = 'A';                // 字符变量
    bool is_active = true;           // 布尔变量
    string name = "Alice";           // 字符串变量
    
    // 变量作用域：局部变量 vs 全局变量
    int global_var = 100;            // 全局变量（文件作用域）
    
    {
        int local_var = 200;          // 块作用域局部变量
        cout << "局部变量: " << local_var << endl;
        cout << "全局变量: " << global_var << endl;
    }
    
    // cout << local_var << endl;     // 错误：local_var超出作用域
    
    // 变量修改
    age = 26;                        // 修改变量值
    age++;                           // 自增操作
    age += 5;                        // 复合赋值
    
    cout << "年龄: " << age << endl;
    cout << "工资: " << salary << endl;
    
    // 变量的内存地址
    cout << "age的地址: " << &age << endl;
    cout << "salary的地址: " << &salary << endl;
    
    return 0;
}
```

### 常量：不可修改的数据存储

**常量定义方式：** C++提供多种方式定义常量，确保数据在程序运行期间不被修改。

```cpp
#include <iostream>
using namespace std;

// 方式1：使用#define预处理指令（C风格，不推荐在现代C++中使用）
#define MAX_SIZE 100
#define PI 3.14159

// 方式2：使用const关键字（推荐）
const int MIN_AGE = 18;
const double GRAVITY = 9.8;

// 方式3：使用constexpr（C++11，编译时常量）
constexpr int ARRAY_SIZE = 50;
constexpr double E = 2.71828;

// 方式4：枚举常量
enum Color { RED, GREEN, BLUE };
enum class Status { OK = 0, ERROR = -1, PENDING = 1 };

int main() {
    // 常量的使用
    cout << "最大尺寸: " << MAX_SIZE << endl;
    cout << "最小年龄: " << MIN_AGE << endl;
    cout << "数组大小: " << ARRAY_SIZE << endl;
    
    // 编译时常量可以在编译时计算
    constexpr int square_size = ARRAY_SIZE * ARRAY_SIZE;
    cout << "平方大小: " << square_size << endl;
    
    // 枚举常量使用
    Color favorite_color = GREEN;
    Status current_status = Status::OK;
    
    cout << "最喜欢的颜色: " << favorite_color << endl;
    cout << "当前状态: " << static_cast<int>(current_status) << endl;
    
    // 常量的优势
    const int buffer_size = 1024;
    // buffer_size = 2048;  // 错误：常量不可修改
    
    // 数组大小必须使用常量
    int numbers[ARRAY_SIZE];  // 正确：ARRAY_SIZE是编译时常量
    // int dynamic_size = 100;
    // int arr[dynamic_size];  // 错误：动态大小不能用于数组声明
    
    // const vs constexpr
    int runtime_value;
    cout << "输入一个值: ";
    cin >> runtime_value;
    
    const int const_value = runtime_value;     // 运行时常量
    // constexpr int ce_value = runtime_value;  // 错误：constexpr必须是编译时常量
    
    constexpr int compile_time_value = 42;     // 编译时常量
    
    return 0;
}
```

### 常量指针与指针常量

**四种const指针组合：** const关键字的位置决定了指针的常量性质。

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10;
    int b = 20;
    
    cout << "=== 常量指针与指针常量详解 ===" << endl;
    
    // 1. 普通指针：可以修改指向和指向的值
    int* ptr1 = &a;
    cout << "1. 普通指针:" << endl;
    cout << "初始指向a: " << *ptr1 << endl;
    *ptr1 = 15;                    // 可以修改指向的值
    ptr1 = &b;                     // 可以修改指向
    cout << "修改后指向b: " << *ptr1 << endl;
    
    // 2. 指针常量（const在*右边）：指针本身是常量，指向不能改
    int* const ptr2 = &a;          // ptr2是指针常量
    cout << "\n2. 指针常量（指向不能改）:" << endl;
    cout << "指向a: " << *ptr2 << endl;
    *ptr2 = 25;                    // 可以修改指向的值
    // ptr2 = &b;                  // 错误：指针常量不能改变指向
    cout << "修改a的值后: " << *ptr2 << endl;
    
    // 3. 常量指针（const在*左边）：指向的值是常量，值不能改
    const int* ptr3 = &a;          // ptr3是常量指针
    cout << "\n3. 常量指针（值不能改）:" << endl;
    cout << "指向a: " << *ptr3 << endl;
    // *ptr3 = 30;                 // 错误：常量指针不能修改指向的值
    ptr3 = &b;                     // 可以修改指向
    cout << "指向b后: " << *ptr3 << endl;
    
    // 4. 指向常量的指针常量（双const）：指向和值都不能改
    const int* const ptr4 = &a;    // ptr4是指向常量的指针常量
    cout << "\n4. 指向常量的指针常量（指向和值都不能改）:" << endl;
    cout << "指向a: " << *ptr4 << endl;
    // *ptr4 = 40;                 // 错误：不能修改值
    // ptr4 = &b;                  // 错误：不能修改指向
    
    // 记忆技巧：const在*左边→值不变，const在*右边→指向不变
    
    cout << "\n=== 字符串常量指针示例 ===" << endl;
    
    // 字符串常量指针
    const char* message = "Hello, World!";
    cout << "消息: " << message << endl;
    // *message = 'h';             // 错误：字符串常量不可修改
    
    // 可以修改指向不同的字符串
    message = "Welcome to C++";
    cout << "新消息: " << message << endl;
    
    // 字符串指针常量
    char greeting[] = "Hello";
    char* const greeting_ptr = greeting;  // 指针常量
    cout << "问候: " << greeting_ptr << endl;
    greeting_ptr[0] = 'h';               // 可以修改字符串内容
    // greeting_ptr = "Hi";              // 错误：指针常量不能改变指向
    cout << "修改后问候: " << greeting_ptr << endl;
    
    cout << "\n=== 函数参数中的const指针 ===" << endl;
    
    // 函数声明示例
    void print_array(const int* arr, int size);      // 常量指针参数
    void modify_array(int* const arr, int size);     // 指针常量参数
    
    int numbers[] = {1, 2, 3, 4, 5};
    
    // 常量指针参数：保证不修改数组内容
    print_array(numbers, 5);
    
    // 指针常量参数：保证不改变指针指向
    modify_array(numbers, 5);
    
    return 0;
}

// 函数定义：使用常量指针参数（只读访问）
void print_array(const int* arr, int size) {
    cout << "数组内容: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
        // arr[i] = 0;  // 错误：常量指针参数不能修改数据
    }
    cout << endl;
}

// 函数定义：使用指针常量参数（保证不改变指向）
void modify_array(int* const arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;  // 可以修改数据
    }
    // arr = nullptr;  // 错误：指针常量参数不能改变指向
}
```

### `const`在函数中的应用

**`const`函数参数和返回值：** 使用`const`可以提高代码的安全性和可读性。

```cpp
#include <iostream>
#include <string>
using namespace std;

// 1. const引用参数：避免拷贝，保证不修改参数
void print_string(const string& str) {
    cout << "字符串: " << str << endl;
    // str[0] = 'X';  // 错误：const引用不能修改
}

// 2. const指针参数：保证不修改指向的数据
int find_max(const int* arr, int size) {
    if (size <= 0) return -1;
    
    int max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// 3. const返回值：返回常量，防止修改
const int& get_reference(const int& a, const int& b) {
    return (a > b) ? a : b;
}

// 4. 常量成员函数（在类中）
class Calculator {
private:
    mutable int call_count;  // mutable：即使在const函数中也可修改
    
public:
    Calculator() : call_count(0) {}
    
    // const成员函数：承诺不修改对象状态
    double calculate(double x, double y) const {
        // call_count++;  // 错误：const成员函数不能修改成员变量
        return x * y;
    }
    
    // 使用mutable成员
    int get_call_count() const {
        call_count++;  // 正确：mutable成员可以在const函数中修改
        return call_count;
    }
    
    // 非const成员函数
    void reset() {
        call_count = 0;
    }
};

int main() {
    // const引用参数使用
    string greeting = "Hello, C++";
    print_string(greeting);
    
    // const指针参数使用
    int numbers[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int max_val = find_max(numbers, 8);
    cout << "最大值: " << max_val << endl;
    
    // const返回值使用
    int x = 10, y = 20;
    const int& larger = get_reference(x, y);
    cout << "较大的值: " << larger << endl;
    // larger = 30;  // 错误：const引用不能修改
    
    // 常量成员函数使用
    Calculator calc;
    const Calculator const_calc;
    
    cout << "计算结果: " << calc.calculate(3.14, 2.71) << endl;
    cout << "常量对象调用: " << const_calc.calculate(1.5, 2.5) << endl;
    // const_calc.reset();  // 错误：常量对象只能调用const成员函数
    
    cout << "调用次数: " << calc.get_call_count() << endl;
    
    return 0;
}
```

**常量使用的最佳实践：**

1. **优先使用const**：默认将变量声明为const，除非需要修改
2. **使用const引用参数**：避免不必要的拷贝，提高性能
3. **const成员函数**：对于不修改对象状态的成员函数声明为const

通过合理使用常量，可以提高代码的安全性、可读性和性能。 


### 变量声明与初始化

**核心概念：** C++中的变量声明不仅确定了变量的类型，还决定了其在内存中的分配方式和生命周期。理解变量声明与初始化的区别对于编写安全高效的C++代码至关重要。

```cpp
// C++变量声明与初始化（内存管理角度）
int x;              // 声明：在栈上分配4字节内存，但内容未定义（危险！）
int y = 10;         // 声明并初始化：分配内存并设置初始值
int z{20};          // C++11统一初始化：避免窄化转换，更安全（推荐）

// 不同初始化方式的区别
int a = 3.14;       // 可能产生警告：窄化转换（3.14→3）
int b{3.14};        // 编译错误：阻止窄化转换，更安全

// 常量声明：编译时优化机会
const int MAX_SIZE = 100;      // 编译时常量：可能被内联优化
constexpr int BUFFER_SIZE = 64; // C++11编译时常量表达式：保证编译时计算

// 自动类型推导（C++11）
auto num = 42;                  // 推导为int
auto name = "Alice";            // 推导为const char*
auto list = {1, 2, 3};          // 推导为std::initializer_list<int>

// C++的变量生命周期管理
{
    int local_var = 50;  // 进入作用域时创建
    // 使用local_var...
}  // 离开作用域时自动销毁（栈内存回收）

// 未初始化变量的危险性
int uninitialized;
std::cout << uninitialized << std::endl;  // 未定义行为：可能输出任意值
```

## 运算符与流程控制

### 运算符

**算术运算符：类型转换与精度控制**

**重要概念：** C++的算术运算遵循严格的类型转换规则。不同类型的操作数运算时，会按照"类型提升"规则转换为更宽的类型。整数除法会截断小数部分，这与Python的浮点除法行为不同。

**位运算符：**
```cpp
unsigned int flags = 0b1010;  // 二进制表示
flags = flags | 0b0001;       // 按位或
flags = flags & ~0b1000;      // 按位与+取反
```

**赋值运算符：**
```cpp
int a = 10;
a += 5;  // 等价于 a = a + 5
a -= 3;  // 等价于 a = a - 3
a *= 2;  // 等价于 a = a × 2
a /= 4;  // 等价于 a = a ÷ 4
```

**关系与逻辑运算符：**
```cpp
int x = 10, y = 5;
bool isEqual = (x == y);      // 等于：false
bool notEqual = (x != y);     // 不等于：true
bool greater = (x > y);       // 大于：true
bool less = (x < y);          // 小于：false
bool greaterEqual = (x >= y); // 大于等于：true
bool lessEqual = (x <= y);    // 小于等于：false

bool andResult = (x > 0 && y > 0);  // 逻辑与：true（两者都为真）
bool orResult = (x > 0 || y < 0);   // 逻辑或：true（至少一个为真）
bool notResult = !(x > 0);          // 逻辑非：false（取反）
```

**短路求值特性：**
- `&&`运算符：如果第一个操作数为假，第二个操作数不会执行
- `||`运算符：如果第一个操作数为真，第二个操作数不会执行

```cpp
int a = 10, b = 3;
std::cout << a + b << std::endl;  // 13: int + int → int
std::cout << a / b << std::endl;  // 3: 整数除法，截断小数部分
std::cout << a % b << std::endl;  // 1: 取模运算（只能用于整数）

// 类型转换示例
double c = 10.0;
std::cout << c / b << std::endl;  // 3.333: double / int → double

// 强制类型转换
std::cout << static_cast<double>(a) / b << std::endl;  // 3.333

// 运算符优先级与结合性
int result = 2 + 3 * 4;     // 14: 乘法优先级高于加法
result = (2 + 3) * 4;       // 20: 括号改变优先级

// 对比Python的除法行为
# Python: 10 / 3 = 3.333... (浮点除法，更符合数学直觉)
# Python: 10 // 3 = 3 (整数除法，与C++相同)
# Python: 10.0 // 3 = 3.0 (浮点数整数除法)

// 浮点数精度问题（C++与Python都存在）
double d1 = 0.1 + 0.2;
std::cout << std::fixed << std::setprecision(20) << d1 << std::endl;
// 输出可能不是精确的0.3，这是浮点数表示的限制
```

**关系与逻辑运算符：**
```cpp
bool result = (a > b) && (a != 0);  // 与运算 and 
result = (a < 5) || (b > 0);        // 或运算 or 
result = !(a == b);                 // 非运算 not
```

**位运算（Python中也存在）：**
```cpp
unsigned int flags = 0b1010;  // 二进制表示
flags = flags | 0b0001;       // 按位或
flags = flags & ~0b1000;      // 按位与+取反
```
[位运算查看python中的位运算](../python/基础_变量.html#位运算符)
### 条件语句

**`if-else`语句：条件逻辑与代码优化**

**编程最佳实践：** C++的`if`语句在条件判断时，编译器会进行短路求值优化。合理的条件顺序可以显著提升性能。同时要注意作用域和变量生命周期问题。
语法：
```cpp
if (条件表达式1) {
    // 条件1为真时执行
} else if (条件表达式2) {
    // 条件1为假、条件2为真时执行
} else if (条件表达式3) {
    // 条件1/2为假、条件3为真时执行
}
// 可添加更多 else if 分支...
else {
    // 所有条件都为假时执行（可选）
}
```

示例：

```cpp
int score = 85;

// 基本if-else结构
if (score >= 90) {
    std::cout << "优秀" << std::endl;
} else if (score >= 80) {  // 只有在第一个条件为false时才检查
    std::cout << "良好" << std::endl;
} else {
    std::cout << "需努力" << std::endl;
}

// 短路求值优化示例
if (ptr != nullptr && ptr->isValid()) {
    // 如果ptr为nullptr，第二个条件不会执行，避免空指针访问
}

// 作用域问题：if语句内声明的变量只在块内有效
if (bool found = searchFunction()) {  // C++17: if with initializer
    // found变量在此块内有效
    std::cout << "Found: " << found << std::endl;
}
// found变量在此处不可访问

```

**`switch-case`语句（C++特有）：**
C++ 的 `switch-case` 语句是多分支选择结构，用于替代多个嵌套的 `if-else`，尤其适合 “变量等于多个离散值” 的分支判断场景，语法简洁且执行效率更高（基于跳转表实现）.<br>

语法：
```cpp
switch (表达式) {
    case 常量表达式1:
        // 分支1代码
        [break;] // 可选，跳出switch
    case 常量表达式2:
        // 分支2代码
        [break;] // 若省略，会触发 “case 穿透”（执行完当前 case 后，继续执行后续 case 代码）
    // 更多case分支...
    default: // 可选，所有case不匹配时执行
        // 默认分支代码
        [break;]
}
```
示例：
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

> **注意**：Python没有`switch`语句，通常用`if-elif`或字典映射实现类似功能

### 循环结构

**`for`循环：迭代控制与性能优化**

**底层机制：** C++的传统`for`循环实际上是一个语法糖，编译器会将其转换为基于条件判断的循环结构。循环变量在每次迭代时都会进行条件检查和增量操作，这种机制在底层被优化为高效的机器代码。
语法：
```cpp
for (初始化表达式; 条件表达式; 更新表达式) {
    // 循环体：条件为真时执行的代码
}
```
示例：
```cpp
// 传统for循环：初始化-条件-增量三部分
for (int i = 0; i < 5; i++) {  // 等价于：
    std::cout << i << " ";      // int i = 0;
}                               // while (i < 5) {
// 输出: 0 1 2 3 4              //     std::cout << i << " ";
                                //     i++;
                                // }

```

**范围for循环：**  

遍历数组 / 容器（无需手动控制索引）
```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};

// 范围for循环（类似Python的for-in）
for (int num : numbers) {
    std::cout << num << " ";
}
// 输出: 1 2 3 4 5

```

**while循环：**  

语法：
```cpp
while (条件表达式) {
    // 循环体：条件为真时执行
}
```
示例：

```cpp
int count = 0;
while (count < 5) {
    std::cout << count << " ";
    count++;
}
// 与Python的while语法相同
```

**do-while循环（C++特有）：**

语法：
```cpp
do {
    // 循环体：至少执行一次
} while (条件表达式); // 末尾必须加分号
```

示例：
```cpp
int input;
do {
    std::cout << "请输入正数: ";
    std::cin >> input;
} while (input <= 0);  // 至少执行一次
```


## IO流系统详解

### 概述
C++的IO流系统基于面向对象设计，提供了类型安全、可扩展的输入输出机制。整个系统构建在流（stream）概念之上，所有IO操作都通过流对象完成。

### IO流类层次结构
```
ios_base (流基类)
    ↓
ios (基础流功能)
    ↓
istream (输入流)        ostream (输出流)
    ↓                       ↓
ifstream (文件输入)     ofstream (文件输出)
istringstream (字符串输入) ostringstream (字符串输出)
cin (标准输入)          cout/cerr/clog (标准输出)
```

### 标准输入输出流

#### 1. `cout` - 标准输出流

**技术原理：** `cout`是`ostream`类的全局对象，连接到标准输出设备（通常是控制台）。

**格式化输出控制：**
```cpp
#include <iostream>
#include <iomanip>  // 格式化控制
using namespace std;

int main() {
    int num = 255;
    double pi = 3.1415926535;
    
    // 数值格式化
    cout << "十进制: " << num << endl;                    // 255
    cout << "十六进制: " << hex << num << endl;           // ff
    cout << "八进制: " << oct << num << endl;             // 377
    
    // 浮点数格式化
    cout << "默认精度: " << pi << endl;                   // 3.14159
    cout << "固定小数点: " << fixed << pi << endl;        // 3.141593
    cout << "科学计数法: " << scientific << pi << endl;   // 3.141593e+00
    cout << "精度控制: " << setprecision(4) << pi << endl; // 3.1416
    
    // 宽度和对齐
    cout << setw(10) << "Hello" << endl;                  // "     Hello"
    cout << setw(10) << left << "Hello" << endl;          // "Hello     "
    cout << setw(10) << right << "Hello" << endl;         // "     Hello"
    cout << setfill('*') << setw(10) << "Hi" << endl;     // "********Hi"
    
    // 重置格式
    cout << dec;  // 恢复十进制
    cout << defaultfloat;  // 恢复默认浮点格式
    
    return 0;
}
```

**缓冲区管理：**
```cpp
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    // 缓冲区行为对比
    cout << "无换行，可能延迟显示";
    this_thread::sleep_for(chrono::seconds(2));
    cout << " - 现在显示完整" << endl;
    
    cout << "立即显示: ";
    cout.flush();  // 强制刷新缓冲区
    this_thread::sleep_for(chrono::seconds(1));
    cout << "延迟内容" << endl;
    
    // endl vs \n
    cout << "使用\\n:" << "第一行\n第二行\n";
    cout << "使用endl:" << "第一行" << endl << "第二行" << endl;
    
    return 0;
}
```

#### 2. `cin` - 标准输入流

**错误处理机制：**
```cpp
#include <iostream>
#include <limits>
using namespace std;

void clear_input_buffer() {
    cin.clear();  // 清除错误状态
    cin.ignore(numeric_limits<streamsize>::max(), '\n');  // 清空缓冲区
}

int main() {
    int age;
    double salary;
    
    // 基本输入
    cout << "请输入年龄: ";
    while (!(cin >> age)) {
        cout << "输入错误，请重新输入年龄: ";
        clear_input_buffer();
    }
    
    cout << "请输入工资: ";
    while (!(cin >> salary)) {
        cout << "输入错误，请重新输入工资: ";
        clear_input_buffer();
    }
    
    // 流状态检查
    if (cin.fail()) {
        cerr << "输入操作失败!" << endl;
    }
    if (cin.eof()) {
        cerr << "到达文件末尾!" << endl;
    }
    
    cout << "年龄: " << age << ", 工资: " << salary << endl;
    
    return 0;
}
```

**高级输入技巧：**
```cpp
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

int main() {
    string line;
    
    // 读取整行并解析
    cout << "请输入多个数据（格式：姓名 年龄 工资）: ";
    getline(cin, line);
    
    istringstream iss(line);
    string name;
    int age;
    double salary;
    
    if (iss >> name >> age >> salary) {
        cout << "姓名: " << name << ", 年龄: " << age 
             << ", 工资: " << salary << endl;
    } else {
        cerr << "输入格式错误!" << endl;
    }
    
    // 字符级输入
    cout << "请输入一个字符: ";
    char ch = cin.get();  // 读取单个字符（包括空白符）
    cin.ignore();  // 消耗换行符
    cout << "字符: '" << ch << "'" << endl;
    
    // 查看下一个字符但不提取
    cout << "请输入一些文本: ";
    char next_char = cin.peek();
    cout << "下一个字符是: '" << next_char << "'" << endl;
    
    return 0;
}
```

### 字符串流（String Streams）

#### 3. `stringstream` 家族

**高级字符串处理：**
```cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

// 数据类型转换工具函数
template<typename T>
string toString(const T& value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

template<typename T>
T fromString(const string& str) {
    istringstream iss(str);
    T value;
    iss >> value;
    if (iss.fail()) {
        throw invalid_argument("转换失败: " + str);
    }
    return value;
}

int main() {
    // 复杂数据解析
    string complex_data = "Alice:25:95.5,Bob:30:88.2,Charlie:22:91.8";
    
    vector<tuple<string, int, double>> people;
    
    // 使用字符串流解析CSV格式数据
    istringstream data_stream(complex_data);
    string person_str;
    
    while (getline(data_stream, person_str, ',')) {
        istringstream person_stream(person_str);
        string name, age_str, salary_str;
        
        getline(person_stream, name, ':');
        getline(person_stream, age_str, ':');
        getline(person_stream, salary_str, ':');
        
        try {
            int age = fromString<int>(age_str);
            double salary = fromString<double>(salary_str);
            people.emplace_back(name, age, salary);
        } catch (const exception& e) {
            cerr << "解析错误: " << e.what() << endl;
        }
    }
    
    // 输出解析结果
    for (const auto& person : people) {
        cout << "姓名: " << get<0>(person) 
             << ", 年龄: " << get<1>(person)
             << ", 工资: " << get<2>(person) << endl;
    }
    
    // 构建复杂字符串
    ostringstream report;
    report << "=== 人员报告 ===" << endl;
    report << "总人数: " << people.size() << endl;
    
    double total_salary = 0;
    for (const auto& person : people) {
        total_salary += get<2>(person);
    }
    
    report << "平均工资: " << (total_salary / people.size()) << endl;
    report << "报告生成时间: " << __DATE__ << " " << __TIME__ << endl;
    
    cout << report.str() << endl;
    
    return 0;
}
```

### 文件流（File Streams）

#### 4. 文件输入输出

**文件操作最佳实践：**
```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
using namespace std;

class FileHandler {
private:
    string filename;
    
public:
    FileHandler(const string& fname) : filename(fname) {}
    
    // 安全写入文件
    bool writeToFile(const vector<string>& lines) {
        ofstream outfile(filename);
        if (!outfile) {
            cerr << "无法打开文件: " << filename << endl;
            return false;
        }
        
        for (const auto& line : lines) {
            outfile << line << '\n';
        }
        
        // 检查写入是否成功
        if (outfile.fail()) {
            cerr << "写入文件失败!" << endl;
            return false;
        }
        
        outfile.close();
        return true;
    }
    
    // 安全读取文件
    vector<string> readFromFile() {
        ifstream infile(filename);
        vector<string> lines;
        
        if (!infile) {
            throw runtime_error("无法打开文件: " + filename);
        }
        
        string line;
        while (getline(infile, line)) {
            lines.push_back(line);
        }
        
        // 检查读取状态
        if (!infile.eof() && infile.fail()) {
            throw runtime_error("读取文件时发生错误");
        }
        
        infile.close();
        return lines;
    }
    
    // 追加模式写入
    bool appendToFile(const string& content) {
        ofstream outfile(filename, ios::app);  // 追加模式
        if (!outfile) {
            return false;
        }
        
        outfile << content << endl;
        return !outfile.fail();
    }
    
    // 二进制文件操作
    bool writeBinary(const vector<int>& data) {
        ofstream outfile(filename, ios::binary);
        if (!outfile) return false;
        
        for (int value : data) {
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        
        return !outfile.fail();
    }
    
    vector<int> readBinary() {
        ifstream infile(filename, ios::binary);
        vector<int> data;
        
        if (!infile) return data;
        
        int value;
        while (infile.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            data.push_back(value);
        }
        
        return data;
    }
};

int main() {
    FileHandler handler("test.txt");
    
    // 文本文件操作
    vector<string> lines = {"第一行", "第二行", "第三行"};
    if (handler.writeToFile(lines)) {
        cout << "文件写入成功!" << endl;
    }
    
    try {
        auto read_lines = handler.readFromFile();
        cout << "读取到的内容:" << endl;
        for (const auto& line : read_lines) {
            cout << " - " << line << endl;
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    
    // 二进制文件操作
    vector<int> binary_data = {1, 2, 3, 4, 5};
    if (handler.writeBinary(binary_data)) {
        cout << "二进制文件写入成功!" << endl;
    }
    
    auto read_binary = handler.readBinary();
    cout << "读取的二进制数据: ";
    for (int val : read_binary) {
        cout << val << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 流操纵器（Stream Manipulators）

#### 5. 自定义流操纵器

**创建自定义格式化工具：**
```cpp
#include <iostream>
#include <iomanip>
#include <string>
using namespace std;

// 自定义流操纵器：颜色输出
ostream& red(ostream& os) {
    return os << "\033[31m";
}

ostream& green(ostream& os) {
    return os << "\033[32m";
}

ostream& blue(ostream& os) {
    return os << "\033[34m";
}

ostream& reset(ostream& os) {
    return os << "\033[0m";
}

// 自定义流操纵器：格式化输出
ostream& currency(ostream& os) {
    os << fixed << setprecision(2);
    return os << "$";
}

ostream& percentage(ostream& os) {
    os << fixed << setprecision(1);
    return os << "%";
}

// 带参数的流操纵器
class indent {
    int spaces;
public:
    indent(int s) : spaces(s) {}
    friend ostream& operator<<(ostream& os, const indent& ind);
};

ostream& operator<<(ostream& os, const indent& ind) {
    for (int i = 0; i < ind.spaces; ++i) {
        os << " ";
    }
    return os;
}

int main() {
    // 使用自定义流操纵器
    cout << red << "错误信息" << reset << endl;
    cout << green << "成功信息" << reset << endl;
    cout << blue << "提示信息" << reset << endl;
    
    double price = 19.99;
    double discount = 15.5;
    
    cout << "原价: " << currency << price << endl;
    cout << "折扣: " << percentage << discount << endl;
    
    // 使用带参数的流操纵器
    cout << indent(4) << "缩进4格" << endl;
    cout << indent(8) << "缩进8格" << endl;
    
    return 0;
}
```

### 错误处理与异常安全

#### 6. 流异常处理

**异常安全的IO操作：**
```cpp
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <memory>
using namespace std;

class SafeFileWriter {
private:
    string filename;
    
public:
    SafeFileWriter(const string& fname) : filename(fname) {}
    
    // RAII方式处理文件
    void writeWithRAII(const string& content) {
        ofstream file(filename);
        if (!file) {
            throw runtime_error("无法打开文件: " + filename);
        }
        
        file << content;
        
        if (file.fail()) {
            throw runtime_error("写入文件失败");
        }
        
        // 文件自动关闭（RAII）
    }
    
    // 使用智能指针管理文件流
    void writeWithSmartPtr(const vector<string>& lines) {
        auto file = make_unique<ofstream>(filename);
        if (!file->is_open()) {
            throw runtime_error("无法打开文件");
        }
        
        for (const auto& line : lines) {
            *file << line << endl;
            if (file->fail()) {
                throw runtime_error("写入失败");
            }
        }
    }
};

// 流状态检查工具
class StreamChecker {
public:
    static void checkStreamState(const ios& stream, const string& operation) {
        if (stream.bad()) {
            throw runtime_error(operation + ": 流已损坏");
        }
        if (stream.fail()) {
            throw runtime_error(operation + ": 操作失败");
        }
        if (stream.eof()) {
            cout << operation + ": 到达文件末尾" << endl;
        }
    }
    
    static void resetStream(ios& stream) {
        stream.clear();
        stream.ignore(numeric_limits<streamsize>::max(), '\n');
    }
};

int main() {
    try {
        SafeFileWriter writer("safe_test.txt");
        writer.writeWithRAII("这是安全写入的内容\n");
        
        vector<string> data = {"第一行", "第二行", "第三行"};
        writer.writeWithSmartPtr(data);
        
        cout << "文件操作成功!" << endl;
        
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    
    // 流状态检查示例
    ifstream test_file("nonexistent.txt");
    StreamChecker::checkStreamState(test_file, "打开文件");
    
    return 0;
}
```

### 性能优化技巧

#### 7. IO性能优化

**减少IO操作开销：**
```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
using namespace std;
using namespace chrono;

class IOBenchmark {
public:
    // 批量写入 vs 单次写入
    static void benchmarkWriteMethods() {
        const int iterations = 10000;
        
        // 方法1：频繁单次写入
        auto start1 = high_resolution_clock::now();
        {
            ofstream file1("method1.txt");
            for (int i = 0; i < iterations; ++i) {
                file1 << "Line " << i << endl;
            }
        }
        auto end1 = high_resolution_clock::now();
        
        // 方法2：批量写入
        auto start2 = high_resolution_clock::now();
        {
            ofstream file2("method2.txt");
            ostringstream buffer;
            for (int i = 0; i < iterations; ++i) {
                buffer << "Line " << i << endl;
            }
            file2 << buffer.str();
        }
        auto end2 = high_resolution_clock::now();
        
        auto duration1 = duration_cast<milliseconds>(end1 - start1);
        auto duration2 = duration_cast<milliseconds>(end2 - start2);
        
        cout << "单次写入时间: " << duration1.count() << "ms" << endl;
        cout << "批量写入时间: " << duration2.count() << "ms" << endl;
        cout << "性能提升: " << (duration1.count() - duration2.count()) << "ms" << endl;
    }
    
    // 缓冲区大小优化
    static void optimizeBufferSize() {
        const int data_size = 1000000;
        vector<int> data(data_size);
        
        // 默认缓冲区
        auto start1 = high_resolution_clock::now();
        {
            ofstream file("default_buffer.bin", ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()), 
                      data_size * sizeof(int));
        }
        auto end1 = high_resolution_clock::now();
        
        // 自定义大缓冲区
        auto start2 = high_resolution_clock::now();
        {
            ofstream file("large_buffer.bin", ios::binary);
            char buffer[8192];  // 8KB缓冲区
            file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
            file.write(reinterpret_cast<const char*>(data.data()), 
                      data_size * sizeof(int));
        }
        auto end2 = high_resolution_clock::now();
        
        cout << "默认缓冲区时间: " 
             << duration_cast<milliseconds>(end1 - start1).count() << "ms" << endl;
        cout << "大缓冲区时间: " 
             << duration_cast<milliseconds>(end2 - start2).count() << "ms" << endl;
    }
};

int main() {
    cout << "=== IO性能测试 ===" << endl;
    IOBenchmark::benchmarkWriteMethods();
    cout << endl;
    IOBenchmark::optimizeBufferSize();
    
    return 0;
}
```

### 现代C++ IO特性

#### 8. C++17/20新特性

**文件系统操作：**
```cpp
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;
using namespace std;

class ModernIO {
public:
    // 使用string_view提高效率
    static void processStringView(string_view sv) {
        cout << "处理字符串视图: " << sv << endl;
        cout << "长度: " << sv.length() << endl;
        cout << "子串: " << sv.substr(0, 5) << endl;
    }
    
    // 文件系统操作
    static void demonstrateFilesystem() {
        fs::path file_path = "modern_test.txt";
        
        // 检查文件状态
        if (fs::exists(file_path)) {
            cout << "文件存在，大小: " << fs::file_size(file_path) << " 字节" << endl;
            cout << "最后修改时间: " << fs::last_write_time(file_path).time_since_epoch().count() << endl;
        } else {
            cout << "文件不存在，创建新文件" << endl;
            ofstream file(file_path);
            file << "这是现代C++创建的文件" << endl;
        }
        
        // 遍历目录
        try {
            for (const auto& entry : fs::directory_iterator(".")) {
                cout << (entry.is_directory() ? "[DIR] " : "[FILE] ") 
                     << entry.path().filename() << endl;
            }
        } catch (const fs::filesystem_error& e) {
            cerr << "文件系统错误: " << e.what() << endl;
        }
    }
    
    // 格式化输出（C++20）
    static void demonstrateFormat() {
        int x = 42;
        double y = 3.14159;
        string name = "Alice";
        
        // C++20格式化（需要编译器支持）
        // cout << format("姓名: {}, 年龄: {}, 圆周率: {:.2f}", name, x, y) << endl;
        
        // 传统方式（兼容性更好）
        cout << "姓名: " << name << ", 年龄: " << x << ", 圆周率: " << fixed << setprecision(2) << y << endl;
    }
};

int main() {
    ModernIO::processStringView("Hello Modern C++");
    cout << endl;
    ModernIO::demonstrateFilesystem();
    cout << endl;
    ModernIO::demonstrateFormat();
    
    return 0;
}
```

### 关键总结与最佳实践

1. **流状态管理**：始终检查流状态，使用`clear()`和`ignore()`处理错误
2. **异常安全**：使用RAII和智能指针管理流资源
3. **性能优化**：减少IO操作次数，使用缓冲区批量处理
4. **类型安全**：优先使用C++流而非C风格IO函数
5. **现代特性**：利用`string_view`、文件系统等现代C++特性
6. **格式化控制**：熟练掌握流操纵器进行精确输出控制
7. **错误处理**：实现健壮的错误处理机制，避免程序崩溃

通过深入理解C++ IO流系统的技术原理和最佳实践，可以编写出高效、安全、可维护的输入输出代码。

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

// std::any：任意类型的容器
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
