## 函数与作用域
函数是一组一起执行一个任务的语句。每个 C++ 程序都至少有一个函数，即主函数 `main()` ，所有简单的程序都可以定义其他额外的函数。  

### 函数声明与定义

**函数基本结构：**
```cpp
// 函数定义形式一般如下
return_type function_name( parameter list )
{
   body of the function
}

// 函数声明（通常在头文件中）
int add(int a, int b);

// 函数定义
int add(int a, int b) {
    return a + b;
}

```

**函数声明与定义分离：**

`ifndef` 是 "`if not defined`" 的缩写，意为「检查某个宏是否未被定义」。这里检查 `MATH_UTILS_H` 这个宏是否从未被编译器定义过，若未定义，则执行后续代码（直到 `#endif`）；若已定义，则直接跳过后续代码（直到 `#endif`）。
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

**三种参数传递方式的对比分析：**

```cpp
#include <iostream>
#include <string>
using namespace std;

// 1. 值传递：创建实参的完整副本
void incrementByValue(int num) {
    num++;  // 只修改副本，不影响原变量
    cout << "函数内（值传递）：" << num << endl;
}

// 2. 引用传递：传递实参的别名
void incrementByReference(int &num) {
    num++;  // 直接修改原变量
    cout << "函数内（引用传递）：" << num << endl;
}

// 3. 指针传递：传递实参的地址
void incrementByPointer(int *num) {
    (*num)++;  // 通过指针修改原变量
    cout << "函数内（指针传递）：" << *num << endl;
}

// 复杂对象的值传递（性能问题演示）
void processLargeObject(string str) {
    cout << "处理字符串（值传递）：" << str << endl;
    // 这里会创建str的完整副本，对于大型字符串可能影响性能
}

// 复杂对象的常量引用传递（最佳实践）
void processLargeObjectEfficiently(const string &str) {
    cout << "处理字符串（引用传递）：" << str << endl;
    // 不创建副本，只读访问，性能最佳
}

int main() {
    int x = 10;
    
    cout << "初始值：" << x << endl;
    
    // 值传递演示
    incrementByValue(x);
    cout << "值传递后：" << x << endl;  // 仍然是10
    
    // 引用传递演示
    incrementByReference(x);
    cout << "引用传递后：" << x << endl;  // 变为11
    
    // 指针传递演示
    incrementByPointer(&x);
    cout << "指针传递后：" << x << endl;  // 变为12
    
    // 大型对象处理演示
    string largeString = "这是一个很长的字符串，用于演示不同传递方式的性能差异...";
    
    processLargeObject(largeString);           // 会产生内存拷贝
    processLargeObjectEfficiently(largeString); // 无拷贝，性能更好
    
    return 0;
}
```

**参数传递机制总结：**

| 传递方式 | 内存开销 | 是否修改原变量 | 适用场景 |
|----------|----------|----------------|----------|
| **值传递** | 创建完整副本 | 否 | 基本类型、小型对象、不需要修改原变量时 |
| **引用传递** | 仅传递别名 | 是 | 需要修改原变量、大型对象 |
| **常量引用** | 仅传递别名 | 否 | 只读访问大型对象，性能最佳 |
| **指针传递** | 传递地址值 | 是 | 需要修改原变量，C风格兼容 |


### 函数重载


**设计原理：** 函数重载是 C++ 特性，指同一作用域内同名、参数列表不同的函数，编译器自动匹配调用。根据参数个数、参数类型、参数顺序（类型不同时）来进行函数重载，返回值不参与区分。

```cpp
// 重载解析示例：编译器如何选择
void process(int x) { cout << "process(int): " << x << endl; }
void process(double x) { cout << "process(double): " << x << endl; }
void process(const string& x) { cout << "process(string): " << x << endl; }

// 调用时的重载解析
process(10);        // 精确匹配：process(int)
process(10.5);      // 精确匹配：process(double)  
process(10);        // 标准转换：int→double，但精确匹配优先
process("hello");   // 需要转换：const char*→string，用户定义转换
```

**默认参数与重载的交互：**
```cpp
// 计算矩形面积，默认宽度为1
int calculateArea(int length, int width = 1) {
    return length * width;
}

// 打印消息，默认次数为1
void printMessage(string message, int times = 1) {
    for (int i = 0; i < times; i++) {
        cout << message << endl;
    }
}

int main() {
    cout << "面积1：" << calculateArea(5) << endl;      // 使用默认宽度
    cout << "面积2：" << calculateArea(5, 3) << endl;    // 指定宽度
    
    printMessage("Hello");     // 使用默认次数
    printMessage("World", 3);  // 指定次数
    
    return 0;
}
```

**默认参数规则：**
- 默认参数必须从右向左定义
- 默认参数在函数声明中指定
- 调用时可以省略有默认值的参数

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



**函数重载的重要规则：**
1. **参数类型或数量不同**：函数名相同，但参数列表必须不同
2. **返回值类型不能作为重载依据**
3. **const成员函数与非const成员函数可以重载**
4. **默认参数可能引起重载歧义**

### 内联函数

内联函数是一种编译器优化手段，核心目的是消除函数调用的开销，提升程序运行效率，同时保留函数封装的优点。**本质是**编译器在编译阶段，将内联函数的函数体代码，直接 “嵌入”（替换）到每一处函数调用的位置，而不是像普通函数那样进行跳转到函数地址的调用操作。

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

**函数优化技术总结：**

1. **内联函数**：减少函数调用开销，适合小型、频繁调用的函数
2. **constexpr函数**：编译时计算，提高运行时性能
3. **编译器优化**：现代编译器会自动进行内联优化

### 递归函数

**设计原理：** 递归是一种将复杂问题分解为相似子问题的编程技术，通过函数自我调用来实现。递归包含两个关键要素：递归终止条件和递归步骤。

**阶乘计算：基础递归示例**
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

// 递归计算阶乘
int factorial(int n) {
    if (n == 0) {
        return 1;  // 递归基
    }
    return n * factorial(n - 1);  // 递归步骤
}

// 迭代计算阶乘（性能对比）
int factorial_iterative(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

void performance_comparison() {
    const int n = 20;
    const int iterations = 1000000;
    
    // 递归版本性能测试
    auto start = high_resolution_clock::now();
    int recursive_result = 0;
    for (int i = 0; i < iterations; i++) {
        recursive_result += factorial(n);
    }
    auto recursive_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    // 迭代版本性能测试
    start = high_resolution_clock::now();
    int iterative_result = 0;
    for (int i = 0; i < iterations; i++) {
        iterative_result += factorial_iterative(n);
    }
    auto iterative_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    cout << "递归版本耗时：" << recursive_time.count() << "微秒" << endl;
    cout << "迭代版本耗时：" << iterative_time.count() << "微秒" << endl;
    cout << "性能差异：" << (recursive_time.count() - iterative_time.count()) * 100.0 / iterative_time.count() << "%" << endl;
}

int main() {
    cout << "5! = " << factorial(5) << endl;
    performance_comparison();
    return 0;
}
```

**递归过程分析：**
```
factorial(5)
= 5 × factorial(4)
= 5 × 4 × factorial(3)
= 5 × 4 × 3 × factorial(2)
= 5 × 4 × 3 × 2 × factorial(1)
= 5 × 4 × 3 × 2 × 1 × factorial(0)
= 5 × 4 × 3 × 2 × 1 × 1
= 120
```

## 进阶特性

### 运算符重载

**运算符重载原理：** 运算符重载允许为自定义类型重新定义现有运算符的行为，使对象操作更加直观自然。编译器将运算符表达式转换为对应的函数调用。

**可以重载的运算符：**
- 算术运算符：`+`, `-`, `*`, `/`, `%`
- 关系运算符：`==`, `!=`, `<`, `>`, `<=`, `>=`
- 逻辑运算符：`&&`, `||`, `!`
- 赋值运算符：`=`, `+=`, `-=`, `*=`, `/=`
- 自增自减运算符：`++`, `--`
- 下标运算符：`[]`
- 函数调用运算符：`()`
- 流运算符：`<<`, `>>`

**不可重载的运算符：**
- `.` (成员访问运算符)
- `.*` (成员指针访问运算符)
- `::` (作用域解析运算符)
- `sizeof` (大小运算符)
- `typeid` (类型信息运算符)
- `?:` (条件运算符)
- `#` (预处理运算符)
- `##` (预处理连接运算符)

```cpp
#include <iostream>
#include <string>
using namespace std;

// 复数类：运算符重载的经典示例
class Complex {
private:
    double real, imag;
    
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // 成员函数形式的运算符重载
    // 加法运算符重载
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // 减法运算符重载
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // 乘法运算符重载
    Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    // 关系运算符重载
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    bool operator!=(const Complex& other) const {
        return !(*this == other);
    }
    
    // 前置自增运算符
    Complex& operator++() {
        ++real;
        return *this;
    }
    
    // 后置自增运算符（使用int参数区分）
    Complex operator++(int) {
        Complex temp = *this;
        ++real;
        return temp;
    }
    
    // 友元函数形式的运算符重载（流操作符必须使用友元）
    friend ostream& operator<<(ostream& os, const Complex& c);
    friend istream& operator>>(istream& is, Complex& c);
    
    // 友元函数：非成员函数形式的运算符重载
    friend Complex operator+(double real, const Complex& c);
};

// 流输出运算符重载（必须使用友元函数）
ostream& operator<<(ostream& os, const Complex& c) {
    os << c.real;
    if (c.imag >= 0) {
        os << " + " << c.imag << "i";
    } else {
        os << " - " << -c.imag << "i";
    }
    return os;
}

// 流输入运算符重载
istream& operator>>(istream& is, Complex& c) {
    cout << "输入实部: ";
    is >> c.real;
    cout << "输入虚部: ";
    is >> c.imag;
    return is;
}

// 非成员函数形式的运算符重载
Complex operator+(double real, const Complex& c) {
    return Complex(real + c.real, c.imag);
}

// 数组类：下标运算符重载示例
class SmartArray {
private:
    int* data;
    int size;
    
public:
    SmartArray(int size) : size(size) {
        data = new int[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0;
        }
    }
    
    // 拷贝构造函数（深拷贝）
    SmartArray(const SmartArray& other) : size(other.size) {
        data = new int[size];
        for (int i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }
    
    // 赋值运算符重载
    SmartArray& operator=(const SmartArray& other) {
        if (this != &other) {  // 防止自赋值
            delete[] data;     // 释放原有内存
            size = other.size;
            data = new int[size];
            for (int i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // 下标运算符重载（非常量版本）
    int& operator[](int index) {
        if (index < 0 || index >= size) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // 下标运算符重载（常量版本）
    const int& operator[](int index) const {
        if (index < 0 || index >= size) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // 函数调用运算符重载（仿函数）
    int operator()(int start, int end) const {
        if (start < 0 || end >= size || start > end) {
            throw invalid_argument("无效的范围");
        }
        int sum = 0;
        for (int i = start; i <= end; i++) {
            sum += data[i];
        }
        return sum;
    }
    
    ~SmartArray() {
        delete[] data;
    }
    
    int getSize() const { return size; }
};

int main() {
    // 复数运算符重载测试
    Complex c1(2, 3);
    Complex c2(4, 5);
    
    cout << "复数运算测试:" << endl;
    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    cout << "c1 + c2 = " << (c1 + c2) << endl;
    cout << "c1 - c2 = " << (c1 - c2) << endl;
    cout << "c1 * c2 = " << (c1 * c2) << endl;
    cout << "5 + c1 = " << (5 + c1) << endl;  // 非成员函数重载
    
    // 自增运算符测试
    Complex c3(1, 1);
    cout << "c3 = " << c3 << endl;
    cout << "++c3 = " << ++c3 << endl;      // 前置自增
    cout << "c3++ = " << c3++ << endl;      // 后置自增
    cout << "c3 = " << c3 << endl;
    
    // 数组下标运算符测试
    SmartArray arr(5);
    for (int i = 0; i < arr.getSize(); i++) {
        arr[i] = i * 10;  // 使用下标运算符赋值
    }
    
    cout << "\n数组内容: ";
    for (int i = 0; i < arr.getSize(); i++) {
        cout << arr[i] << " ";  // 使用下标运算符访问
    }
    cout << endl;
    
    // 函数调用运算符测试
    cout << "数组索引1到3的和: " << arr(1, 3) << endl;
    
    return 0;
}
```

**运算符重载的最佳实践：**

| 运算符类型 | 重载方式 | 返回值 | 注意事项 |
|------------|----------|--------|----------|
| **算术运算符** | 成员函数或友元 | 新对象 | 不修改原对象 |
| **关系运算符** | 成员函数 | bool | 实现逻辑一致性 |
| **赋值运算符** | 成员函数 | 引用 | 处理自赋值情况 |
| **流运算符** | 友元函数 | 流引用 | 必须使用友元 |
| **下标运算符** | 成员函数 | 引用 | 提供const版本 |

### 智能指针

**智能指针设计原理：** 智能指针是RAII（资源获取即初始化）原则的典型应用，通过对象生命周期自动管理动态内存，避免内存泄漏和悬空指针问题。

```cpp
#include <iostream>
#include <memory>
#include <vector>
using namespace std;

// 资源管理类示例
class Resource {
private:
    string name;
    
public:
    Resource(const string& n) : name(n) {
        cout << "创建资源: " << name << endl;
    }
    
    ~Resource() {
        cout << "释放资源: " << name << endl;
    }
    
    void use() const {
        cout << "使用资源: " << name << endl;
    }
    
    string getName() const { return name; }
};

// 演示循环引用问题
class Node {
public:
    string name;
    shared_ptr<Node> next;  // 可能导致循环引用
    weak_ptr<Node> prev;    // 使用weak_ptr避免循环引用
    
    Node(const string& n) : name(n) {
        cout << "创建节点: " << name << endl;
    }
    
    ~Node() {
        cout << "销毁节点: " << name << endl;
    }
};

// 自定义删除器示例
void custom_deleter(Resource* ptr) {
    cout << "自定义删除器释放资源: " << ptr->getName() << endl;
    delete ptr;
}

int main() {
    cout << "=== unique_ptr 独占所有权 ===" << endl;
    {
        // unique_ptr：独占式所有权，不能拷贝只能移动
        unique_ptr<Resource> ptr1 = make_unique<Resource>("资源A");
        ptr1->use();
        
        // 所有权转移
        unique_ptr<Resource> ptr2 = move(ptr1);
        if (!ptr1) {
            cout << "ptr1已失去所有权" << endl;
        }
        ptr2->use();
        
        // 自动释放：离开作用域时自动调用析构函数
    }
    
    cout << "\n=== shared_ptr 共享所有权 ===" << endl;
    {
        // shared_ptr：共享所有权，引用计数管理
        shared_ptr<Resource> ptr1 = make_shared<Resource>("共享资源B");
        
        {
            shared_ptr<Resource> ptr2 = ptr1;  // 引用计数+1
            shared_ptr<Resource> ptr3 = ptr1;  // 引用计数+1
            
            cout << "当前引用计数: " << ptr1.use_count() << endl;  // 输出3
            ptr2->use();
        }  // ptr2, ptr3离开作用域，引用计数-2
        
        cout << "当前引用计数: " << ptr1.use_count() << endl;  // 输出1
        ptr1->use();
        
        // 离开作用域时，引用计数为0，自动释放
    }
    
    cout << "\n=== weak_ptr 弱引用 ===" << endl;
    {
        // weak_ptr：观察shared_ptr但不增加引用计数
        shared_ptr<Resource> shared = make_shared<Resource>("弱引用资源C");
        weak_ptr<Resource> weak = shared;
        
        cout << "shared引用计数: " << shared.use_count() << endl;  // 输出1
        
        // 检查weak_ptr是否有效
        if (auto locked = weak.lock()) {
            cout << "资源有效: " << locked->getName() << endl;
            locked->use();
        } else {
            cout << "资源已释放" << endl;
        }
        
        // 释放shared_ptr
        shared.reset();
        
        if (auto locked = weak.lock()) {
            cout << "资源仍然有效" << endl;
        } else {
            cout << "资源已释放，weak_ptr失效" << endl;
        }
    }
    
    cout << "\n=== 循环引用问题演示 ===" << endl;
    {
        // 创建循环引用
        shared_ptr<Node> node1 = make_shared<Node>("节点1");
        shared_ptr<Node> node2 = make_shared<Node>("节点2");
        
        node1->next = node2;  // node2引用计数+1
        node2->next = node1;  // node1引用计数+1（循环引用！）
        
        cout << "node1引用计数: " << node1.use_count() << endl;  // 输出2
        cout << "node2引用计数: " << node2.use_count() << endl;  // 输出2
        
        // 正确做法：使用weak_ptr避免循环引用
        shared_ptr<Node> node3 = make_shared<Node>("节点3");
        shared_ptr<Node> node4 = make_shared<Node>("节点4");
        
        node3->next = node4;
        node4->prev = node3;  // 使用weak_ptr，不增加引用计数
        
        cout << "node3引用计数: " << node3.use_count() << endl;  // 输出1
        cout << "node4引用计数: " << node4.use_count() << endl;  // 输出1
    }  // 正常释放，无内存泄漏
    
    cout << "\n=== 自定义删除器 ===" << endl;
    {
        // 使用自定义删除器
        unique_ptr<Resource, decltype(&custom_deleter)> 
            ptr1(new Resource("自定义删除资源"), custom_deleter);
        
        shared_ptr<Resource> ptr2(
            new Resource("shared_ptr自定义删除"), 
            custom_deleter
        );
        
        ptr1->use();
        ptr2->use();
        
        // 自动调用自定义删除器
    }
    
    cout << "\n=== 数组智能指针 ===" << endl;
    {
        // unique_ptr支持数组
        unique_ptr<int[]> arr_ptr(new int[5]);
        for (int i = 0; i < 5; i++) {
            arr_ptr[i] = i * 10;  // 使用数组下标
        }
        
        cout << "数组内容: ";
        for (int i = 0; i < 5; i++) {
            cout << arr_ptr[i] << " ";
        }
        cout << endl;
        
        // shared_ptr需要自定义删除器处理数组
        shared_ptr<int> shared_arr(
            new int[5],
            [](int* p) { 
                cout << "释放数组内存" << endl; 
                delete[] p; 
            }
        );
    }
    
    return 0;
}
```

**智能指针使用指南：**

| 智能指针类型 | 适用场景 | 特点 | 注意事项 |
|--------------|----------|------|----------|
| **unique_ptr** | 独占所有权场景 | 零开销，性能最佳 | 不能拷贝，只能移动 |
| **shared_ptr** | 共享所有权场景 | 引用计数管理 | 注意循环引用问题 |
| **weak_ptr** | 观察shared_ptr | 不增加引用计数 | 需要检查有效性 |

**内存管理最佳实践：**
1. **优先使用智能指针**：避免手动new/delete
2. **选择合适的智能指针**：根据所有权需求选择
3. **使用make_shared/make_unique**：更安全高效
4. **注意循环引用**：使用weak_ptr解决
5. **自定义删除器**：处理特殊资源释放

通过运算符重载和智能指针的学习，可以编写出更加安全、直观和高效的C++代码，这是现代C++编程的重要基础。