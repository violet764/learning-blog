# C++基础与核心语法

## 语言概述

C++ 是一种静态类型的、编译式的、通用的、大小写敏感的编程语言，支持过程化编程、面向对象编程和泛型编程。

### 第一个C++程序

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```


## 数据类型与变量

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

位运算符直接操作整数的二进制位，在底层编程、性能优化和硬件操作中非常重要。

```cpp
unsigned int flags = 0b1010;  // 二进制表示
flags = flags | 0b0001;       // 按位或
flags = flags & ~0b1000;      // 按位与+取反
```
[位运算查看python中的位运算](../python/基础_变量.html#位运算符)



**位运算符概述**
```cpp
#include <iostream>
#include <bitset>
#include <iomanip>

void demonstrateBitOperations() {
    unsigned int a = 0b11001100;  // 204
    unsigned int b = 0b10101010;  // 170
    
    std::cout << "a = " << std::bitset<8>(a) << " (" << a << ")" << std::endl;
    std::cout << "b = " << std::bitset<8>(b) << " (" << b << ")" << std::endl;
    
    // 按位与 (&)
    unsigned int and_result = a & b;
    std::cout << "a & b = " << std::bitset<8>(and_result) << " (" << and_result << ")" << std::endl;
    
    // 按位或 (|)
    unsigned int or_result = a | b;
    std::cout << "a | b = " << std::bitset<8>(or_result) << " (" << or_result << ")" << std::endl;
    
    // 按位异或 (^)
    unsigned int xor_result = a ^ b;
    std::cout << "a ^ b = " << std::bitset<8>(xor_result) << " (" << xor_result << ")" << std::endl;
    
    // 按位取反 (~)
    unsigned int not_result = ~a;
    std::cout << "~a = " << std::bitset<8>(not_result) << " (" << not_result << ")" << std::endl;
    
    // 左移 (<<)
    unsigned int left_shift = a << 2;
    std::cout << "a << 2 = " << std::bitset<8>(left_shift) << " (" << left_shift << ")" << std::endl;
    
    // 右移 (>>)
    unsigned int right_shift = a >> 2;
    std::cout << "a >> 2 = " << std::bitset<8>(right_shift) << " (" << right_shift << ")" << std::endl;
}
```

**位运算实用技巧**

```cpp
#include <iostream>
#include <bitset>

class BitUtils {
public:
    // 检查特定位是否为1
    static bool isBitSet(unsigned int num, int pos) {
        return (num & (1 << pos)) != 0;
    }
    
    // 设置特定位为1
    static unsigned int setBit(unsigned int num, int pos) {
        return num | (1 << pos);
    }
    
    // 清除特定位（设置为0）
    static unsigned int clearBit(unsigned int num, int pos) {
        return num & ~(1 << pos);
    }
    
    // 切换特定位（0变1，1变0）
    static unsigned int toggleBit(unsigned int num, int pos) {
        return num ^ (1 << pos);
    }
    
    // 计算1的个数（汉明重量）
    static int countOnes(unsigned int num) {
        int count = 0;
        while (num) {
            count += num & 1;
            num >>= 1;
        }
        return count;
    }
    
    // 判断是否为2的幂
    static bool isPowerOfTwo(unsigned int num) {
        return num && !(num & (num - 1));
    }
    
    // 获取最低有效位（最右边的1）
    static unsigned int getLowestSetBit(unsigned int num) {
        return num & -num;
    }
    
    // 交换两个变量的值（不使用临时变量）
    static void swap(int& a, int& b) {
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    }
};

void demonstrateBitUtils() {
    unsigned int num = 0b10110110;  // 182
    
    std::cout << "原始数字: " << std::bitset<8>(num) << " (" << num << ")" << std::endl;
    
    // 检查第3位
    std::cout << "第3位是否为1: " << BitUtils::isBitSet(num, 3) << std::endl;
    
    // 设置第0位
    unsigned int set = BitUtils::setBit(num, 0);
    std::cout << "设置第0位后: " << std::bitset<8>(set) << " (" << set << ")" << std::endl;
    
    // 清除第2位
    unsigned int cleared = BitUtils::clearBit(num, 2);
    std::cout << "清除第2位后: " << std::bitset<8>(cleared) << " (" << cleared << ")" << std::endl;
    
    // 切换第4位
    unsigned int toggled = BitUtils::toggleBit(num, 4);
    std::cout << "切换第4位后: " << std::bitset<8>(toggled) << " (" << toggled << ")" << std::endl;
    
    // 计算1的个数
    std::cout << "1的个数: " << BitUtils::countOnes(num) << std::endl;
    
    // 判断是否为2的幂
    std::cout << "是否为2的幂: " << BitUtils::isPowerOfTwo(num) << std::endl;
    std::cout << "64是否为2的幂: " << BitUtils::isPowerOfTwo(64) << std::endl;
    
    // 交换变量
    int x = 10, y = 20;
    std::cout << "交换前: x=" << x << ", y=" << y << std::endl;
    BitUtils::swap(x, y);
    std::cout << "交换后: x=" << x << ", y=" << y << std::endl;
}
```



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


