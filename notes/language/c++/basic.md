# C++基础与核心语法

## C++语言概述

**编译流程：**
1. **预处理**：处理 `#include`、`#define` 等指令
2. **编译**：将C++代码翻译成汇编代码
3. **汇编**：将汇编代码翻译成机器代码
4. **链接**：连接多个目标文件生成可执行文件

> **注意**：Python是解释执行，无需编译步骤；C++需要完整的编译流程才能运行

## 第一个C++程序

```cpp
#include <iostream>  
using namespace std;

int main() {  
    std::cout << "Hello, World!" << std::endl;  
    return 0;  
}
```


## IO流

|组件|核心作用|依赖头文件|
|---|---|---|
|`cout`|标准输出流，向控制台打印数据|`<iostream>`|
|`cin`|标准输入流，从控制台读取数据|`<iostream>`|
|`stringstream`|字符串流，在内存中读写字符串（解析/拼接）|`<sstream>`|

**`cout`（标准输出）**

- **基础功能**：将数据（整数、字符串、浮点数等）输出到控制台，支持链式调用。

- **常用操作符/方法**：

    - `<<`：输出运算符，可连续拼接多个输出内容；

    - `endl`：输出换行符并刷新缓冲区；

    - `flush`：仅强制刷新缓冲区（无换行）。

- **示例**：

```C++
#include <iostream>
using namespace std;
int main() {
    int age = 20;
    double score = 98.5;
    // 链式输出，混合文本与变量
    cout << "年龄：" << age << "，成绩：" << score << endl;
    return 0;
}
```

**`cin`（标准输入）**

- **基础功能**：从控制台读取用户输入的内容，存储到变量中，同样支持链式读取。

- **核心注意点**：

    - `>>`：输入运算符，默认以**空格/换行符**为分隔符，读取时会跳过空白字符；

    - 读取字符串时，`cin >> str` 遇空格停止，需读取含空格的整行时用 `getline(cin, str)`；

    - `cin` 读取后缓冲区会残留换行符，需用 `cin.ignore()` 清理后再调用 `getline`。

- **示例**：

```C++
#include <iostream>
#include <string>
using namespace std;
int main() {
    int num;
    string name, desc;
    cout << "请输入编号和姓名（用空格分隔）: ";
    cin >> num >> name; // 读取整数和不含空格的字符串
    cin.ignore();       // 清理缓冲区的换行符
    cout << "请输入描述: ";
    getline(cin, desc); // 读取含空格的整行描述
    cout << "编号：" << num << "，姓名：" << name << "，描述：" << desc << endl;
    return 0;
}
```

**`stringstream`（字符串流）**

- **核心功能**：在内存中操作字符串，分为两种核心类型：

    - `istringstream`：将字符串解析为不同类型的数据（替代 `atoi`/`atof`，更安全）；

    - `ostringstream`：将不同类型数据拼接为字符串（替代 `sprintf`，避免缓冲区溢出）。

- **核心方法**：

    - `iss >> 变量`：从字符串流读取数据到变量；

    - `oss << 数据`：向字符串流写入数据；

    - `str()`：获取字符串流中的完整字符串。

- **示例**：

```C++
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
int main() {
    // 1. 解析字符串：拆分混合类型的字符串
    string data = "100 3.14 C++";
    istringstream iss(data);
    int num; double pi; string lang;
    iss >> num >> pi >> lang; // 分别解析为int、double、string
    
    // 2. 拼接字符串：组合不同类型数据
    ostringstream oss;
    oss << "数字：" << num << "，圆周率：" << pi << "，语言：" << lang;
    string result = oss.str(); // 获取拼接后的字符串
    
    cout << result << endl; // 输出：数字：100，圆周率：3.14，语言：C++
    return 0;
}
```

**关键总结**

1. `cout`/`cin` 聚焦控制台IO，需注意输入缓冲区的清理（尤其是 `getline` 前）；

2. `stringstream` 适合字符串与基础数据类型的互转，比C语言的字符数组操作更安全；

3. 所有IO流操作需包含对应头文件，`using namespace std` 可简化 `std::cout`/`std::cin` 等写法。

## <span style="background-color: #26e6bcff; padding: 2px 4px; border-radius: 3px; color: #333;">数据类型</span>

C++ 原生支持的基础数据类型，分为数值型、布尔型、空类型，尺寸受编译器/平台影响（以下为64位系统典型值）。

### 原生数据类型  

**整数类型**
| 类型               | 典型尺寸 | 取值范围（典型）| 无符号标识 | 核心用途                     |
|--------------------|----------|-------------------------|------------|------------------------------|
| `char`             | 1字节    | -128 ~ 127              | `unsigned char` | 存储字符/ASCII码/小整数      |
| `short`            | 2字节    | -32768 ~ 32767          | `unsigned short` | 小范围整数，节省内存         |
| `int`              | 4字节    | -2¹⁵ ~ 2¹⁵-1（2字节）/ -2³¹ ~ 2³¹-1（4字节） | `unsigned int` | 最常用整数类型，性能最优     |
| `long`             | 8字节 | -2³¹ ~ 2³¹-1（4字节）/ -2⁶³ ~ 2⁶³-1（8字节） | `unsigned long` | 大范围整数                   |
| `long long`        | 8字节    | -2⁶³ ~ 2⁶³-1            | `unsigned long long` | 超大范围整数|

整数类型说明：
- 无符号类型（`unsigned`）仅存储非负数，取值上限翻倍；
- 字面量后缀：`U`（无符号）、`L`（long）、`LL`（long long），如 `100LL`、`200U`；
- 用 `sizeof(类型)`（一个运算符） 获取当前平台实际尺寸，如 `sizeof(int)`。

**浮点类型**  
遵循 IEEE 754 标准，存在精度损失，默认字面量为 `double` 类型。

| 类型         | 典型尺寸 | 有效数字 | 取值范围               | 后缀 | 核心用途                     |
|--------------|----------|----------|------------------------|------|------------------------------|
| `float`      | 4字节    | 6~7位    | ±10⁻³⁸ ~ ±10³⁸         | `f`  | 单精度浮点，适合内存受限场景 |
| `double`     | 8字节    | 15~16位  | ±10⁻³⁰⁸ ~ ±10³⁰⁸       | -    | 双精度浮点，默认推荐使用     |
| `long double`| 8/16字节 | 18~19位  | ±10⁻⁴⁹³² ~ ±10⁴⁹³²     | `L`  | 扩展精度浮点，极少使用       |

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

### 复合/派生数据类型
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

**容器类型（标准库）**
| 类型               | 核心特性                                                                 | 示例                     |
|--------------------|--------------------------------------------------------------------------|--------------------------|
| `std::vector`      | 动态数组，自动扩容，随机访问高效                                         | `vector<int> vec={1,2,3};` |
| `std::map`         | 键值对集合，有序存储，基于红黑树实现                                     | `map<string,int> mp={{"age",20}};` |
| `std::set`         | 无序不重复集合，基于哈希表（`unordered_set`）或红黑树（`set`）| `set<int> s={1,2,3};`    |

**C++数据类型声明：**
```cpp
// C++需要显式声明类型
int number = 42;           // 整数
double pi = 3.14159;       // 双精度浮点数
char letter = 'A';         // 字符
char str[6] = {'H', 'e', 'l', 'l', 'o','\0} // 末尾必须添加'\0'
bool is_valid = true;      // 布尔值
std::string name = "Alice"; // 字符串

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
在内存占用上存在字节对齐现象，占用的内存为最大成员大小的整数倍。
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

// 对比Python的动态变量机制
# Python变量使用：名称绑定到对象
x = 10              # 创建整数对象10，x绑定到该对象
x = "hello"         # 创建字符串对象，x重新绑定（类型动态改变）
MAX_SIZE = 100      # 只是约定，实际可以修改（非真正常量）

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

## 函数与作用域

### 函数声明与定义

**函数基本结构：**
```cpp
// 函数声明（通常在头文件中）
int add(int a, int b);

// 函数定义
int add(int a, int b) {
    return a + b;
}

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

### 参数传递机制

**值传递（默认）：内存拷贝机制**

**底层原理：** 值传递会在函数调用时创建参数的完整副本。对于基本类型（如`int`、`double`），这通常很高效；但对于大型对象（如`std::vector`、自定义类），会产生不必要的内存拷贝开销。

```cpp
// 值传递：创建参数的完整副本
void modifyValue(int x) {  // x是a的副本，占用独立内存空间
    x = 100;  // 只修改副本，不影响原变量
    // 函数结束时，x的副本被销毁
}

int main() {
    int a = 10;           // a在栈上分配内存
    modifyValue(a);       // 调用时：将a的值拷贝给x
    std::cout << a << std::endl;  // 输出: 10（原值未改变）
    return 0;
}

// 大型对象的值传递（可能产生性能问题）
void processVector(std::vector<int> vec) {  // 拷贝整个vector
    // 处理vector...
}  // 函数结束时，vec的副本被销毁（可能触发析构函数）
```

**引用传递：别名机制与性能优化**

**底层原理：** 引用本质上是变量的别名，不占用额外内存空间。编译器在内部将引用实现为指针，但语法上更安全，避免了空指针和野指针问题。引用传递避免了不必要的拷贝，是C++中高效传递大型对象的首选方式。
语法：
```cpp
返回值类型 函数名(参数列表) {
    // 函数体：执行逻辑
    [return 返回值;] // 非void类型必须返回对应类型值
}
```


示例：
```cpp
// 引用传递：创建变量的别名（不拷贝数据）
void modifyReference(int &x) {  // x是a的引用（别名）
    x = 100;  // 通过引用直接修改原变量
    // x和a实际上是同一块内存的两个名称
}

int main() {
    int a = 10;
    modifyReference(a);       // 传递a的引用，无数据拷贝
    std::cout << a << std::endl;  // 输出: 100（原值被修改）
    return 0;
}

// 常量引用：只读访问，避免拷贝（最佳实践）
void readOnly(const std::vector<int> &vec) {  // 不拷贝，只读访问
    // 可以读取vec的内容，但不能修改
    std::cout << "Size: " << vec.size() << std::endl;
}

// 右值引用（C++11）：移动语义，避免不必要的拷贝
void takeOwnership(std::vector<int> &&vec) {  // 接收即将销毁的对象
    // 可以"窃取"vec的资源，避免拷贝
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

### 函数重载

**函数重载示例：编译时多态与类型安全**

**设计原理：** 函数重载是C++实现编译时多态的重要机制。编译器根据调用时的实参类型选择最匹配的函数版本，这提供了类型安全的接口，同时保持了代码的简洁性。

```cpp
// 同名函数，不同参数列表（函数签名不同）
int add(int a, int b) {           // 版本1：整数加法
    return a + b;
}

double add(double a, double b) {   // 版本2：浮点数加法
    return a + b;
}

std::string add(const std::string &a, const std::string &b) {  // 版本3：字符串连接
    return a + b;
}

// 重载解析过程：编译器根据实参类型选择最合适的版本
std::cout << add(5, 3) << std::endl;               // 调用版本1：int add(int, int)
std::cout << add(2.5, 3.7) << std::endl;           // 调用版本2：double add(double, double)
std::cout << add("Hello", " World") << std::endl;  // 调用版本3：string add(const string&, const string&)

// 类型转换与重载解析
std::cout << add(5, 3.14) << std::endl;  // 可能产生歧义：int+double
// 编译器需要决定：将5转换为double，还是将3.14转换为int

// 重载规则：返回值类型不同不能构成重载
// int process();        // ✅
// double process();     // ❌ 错误：仅返回值不同

// 默认参数与重载的交互
void print(int a, int b = 10);     // 版本A
void print(int a);                 // 版本B

print(5);  // 歧义：可以调用版本A（使用默认参数）或版本B
```

> **对比Python：** Python不支持函数重载，通常用默认参数或类型检查实现类似功能

### 内联函数

**内联函数：**
```cpp
// 建议编译器将函数体直接插入调用处
inline int square(int x) {
    return x * x;
}

// 使用内联函数（可能被优化为直接计算）
int result = square(5);  // 可能被优化为: int result = 5 * 5;
```

**constexpr函数：**
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


