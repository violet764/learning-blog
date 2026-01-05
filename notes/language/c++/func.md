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

## 数组与字符串

### 原生数组与`std::array`

**原生数组（C风格）：连续内存布局与性能优势**

**内存布局原理：** C风格数组在内存中是连续分配的固定大小块。这种连续布局使得CPU缓存预取更加高效，是现代处理器优化的重要基础。数组名实际上是首元素的指针，编译器在编译时就知道数组的大小和类型信息。

```cpp
// 静态数组声明：在栈上分配连续内存块
int numbers[5] = {1, 2, 3, 4, 5};  // 固定大小，编译时确定

// 内存布局：地址连续，元素类型相同
// numbers[0] -> 0x1000: 值1
// numbers[1] -> 0x1004: 值2（int通常4字节）
// numbers[2] -> 0x1008: 值3
// numbers[3] -> 0x100C: 值4
// numbers[4] -> 0x1010: 值5

// 访问元素：编译器转换为指针算术
std::cout << numbers[0] << std::endl;  // 等价于 *(numbers + 0)
std::cout << numbers[4] << std::endl;  // 等价于 *(numbers + 4)

// 计算数组长度：利用编译时类型信息
int length = sizeof(numbers) / sizeof(numbers[0]);  // 20 / 4 = 5
// sizeof(numbers)返回整个数组的大小（5*4=20字节）
// sizeof(numbers[0])返回单个元素的大小（4字节）

// 遍历数组：连续内存访问，缓存友好
for (int i = 0; i < length; i++) {
    std::cout << numbers[i] << " ";  // 顺序访问，预取高效
}
std::cout << std::endl;

// 安全性问题：无边界检查
// numbers[5] = 100;  // 未定义行为：越界访问
```

**`std::array`：类型安全的现代数组**

**设计理念：** `std::array`是对C风格数组的封装，提供了类型安全的接口而不牺牲性能。它在编译时知道大小，因此可以内联优化，同时提供了边界检查、迭代器支持等现代C++特性。

```cpp
#include <array>

// std::array声明：模板参数指定类型和大小
std::array<int, 5> arr = {1, 2, 3, 4, 5};
// 底层实现：内部包含一个C风格数组，但提供安全的接口

// 更安全的访问方式：编译时和运行时安全检查
std::cout << arr.at(0) << std::endl;    // 边界检查，越界抛出std::out_of_range
std::cout << arr.size() << std::endl;   // 编译时常量：5

// 与C风格数组的性能对比
std::cout << arr[0] << std::endl;       // 无检查，性能与C数组相同
std::cout << arr.at(0) << std::endl;    // 有检查，轻微性能开销

// 范围for循环支持：基于迭代器协议
for (int num : arr) {  // 等价于：for(auto it = arr.begin(); it != arr.end(); ++it)
    std::cout << num << " ";
}
std::cout << std::endl;

// 迭代器支持（STL兼容）
auto it = arr.begin();
while (it != arr.end()) {
    std::cout << *it << " ";
    ++it;
}

// 编译时大小检查（模板元编程）
static_assert(arr.size() == 5, "数组大小必须是5");
```

**对比Python的`list`：**
```python
# Python列表（动态数组）
numbers = [1, 2, 3, 4, 5]
print(numbers[0])      # 1
print(len(numbers))    # 5

# Python列表可以动态调整大小
numbers.append(6)      # 添加元素
numbers.pop()          # 移除元素
```

> **关键差异**：C++数组大小固定，Python列表动态调整；`std::array`提供类型安全和边界检查

### C风格字符串 vs `std::string`

**C风格字符串：以空字符结尾的字符数组**

**底层表示：** C风格字符串实际上是字符数组，以空字符`'\0'`作为结束标志。这种设计简单高效，但安全性较差，容易出现缓冲区溢出等问题。字符串长度需要运行时计算，每次操作都需要遍历整个字符串。

```cpp
#include <cstring>

// C风格字符串声明：字符数组 + 空终止符
char cstr1[] = "Hello";           // 编译器自动添加'\0'，实际大小6字节
char cstr2[10] = "World";         // 指定大小，剩余空间填充'\0'

// 内存布局分析：
// cstr1: ['H','e','l','l','o','\0']
// cstr2: ['W','o','r','l','d','\0','\0','\0','\0','\0']

// 字符串操作：基于指针算术的函数
std::cout << strlen(cstr1) << std::endl;  // 遍历直到'\0'，返回5
strcpy(cstr2, "C++");                     // 复制：['C','+','+','\0','d','\0',...]
strcat(cstr1, " World");                  // 连接：需要足够空间

// 安全性问题分析
char buffer[5] = "test";
// strcpy(buffer, "overflow");  // 缓冲区溢出：写入超出分配空间
// 后果：可能覆盖相邻内存，导致程序崩溃或安全漏洞

// 安全替代函数（C11）
strncpy(cstr2, "C++", sizeof(cstr2));      // 指定最大复制长度
strncat(cstr1, " World", remaining_space); // 限制连接长度

// 性能考虑：O(n)时间复杂度
// strlen()需要遍历整个字符串
// strcpy()需要遍历源字符串
// strcat()需要先找到目标字符串结尾
```

**`std::string`：动态内存管理的安全字符串**

**设计原理：** `std::string`是一个类模板，内部管理动态分配的字符数组。它实现了RAII原则，自动处理内存分配和释放，提供了丰富的成员函数和安全的操作接口。现代实现通常使用小字符串优化（SSO）来避免小型字符串的动态分配。

```cpp
#include <string>

// std::string构造：多种初始化方式
std::string str1 = "Hello";           // 从C字符串构造
std::string str2("World");            // 直接构造
std::string str3(10, 'x');            // 重复字符构造
std::string str4(str1);               // 拷贝构造

// 内部实现原理（简化）
// class string {
// private:
//     char *data;        // 动态分配的字符数组
//     size_t length;     // 当前长度
//     size_t capacity;   // 分配的空间大小
//     // 可能包含小字符串优化（SSO）的缓冲区
// };

// 安全的字符串操作：自动内存管理
std::cout << str1.length() << std::endl;  // O(1)复杂度，直接返回长度
str1 += " C++";                          // 自动检查空间，必要时重新分配
str1.append("!");                        // 成员函数，类型安全

// 查找和子字符串：丰富的算法支持
size_t pos = str1.find("C++");           // 线性搜索，返回位置或npos
if (pos != std::string::npos) {  // std::string::npos专门用于表示「查找操作未找到目标」的状态
    std::string sub = str1.substr(pos, 3);  // 创建子字符串拷贝
}

// 现代C++特性支持
std::string_view sv = str1;             // C++17：非拥有字符串视图
auto result = str1.starts_with("Hello"); // C++20：前缀检查

// 性能优化：预留空间减少重新分配
std::string large_str;
large_str.reserve(1000);                // 预留空间，避免多次分配
for (int i = 0; i < 1000; i++) {
    large_str += 'x';                   // 在预留空间内操作，高效
}

// 与C字符串互操作
const char* c_str = str1.c_str();       // 获取C风格字符串（只读）
char buffer[100];
str1.copy(buffer, sizeof(buffer));      // 安全拷贝到缓冲区
```

**对比Python的`str`：**
```python
# Python字符串操作
s = "Hello"
print(len(s))           # 5
s += " World"          # 连接
print(s.find("World")) # 6
print(s[6:11])         # World
```

> **注意**：C++的`std::string`与Python的`str`在使用上非常相似，都是安全的字符串类型

### 多维数组的内存布局

**二维数组内存布局：**
```cpp
// 二维数组（连续内存块）
int matrix[3][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

// 内存布局：1,2,3,4,5,6,7,8,9（连续存储）
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        std::cout << &matrix[i][j] << " ";  // 打印地址
    }
    std::cout << std::endl;
}
```

**动态多维数组：**
```cpp
// 使用vector的vector（不连续内存）
std::vector<std::vector<int>> dynamic_matrix(3, std::vector<int>(3));

// 每个内层vector是独立的内存块
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        dynamic_matrix[i][j] = i * 3 + j + 1;
    }
}
```

## 指针与引用

### 指针：内存地址的操作

**指针基础：内存地址的间接访问机制**

**底层原理：** 指针是存储内存地址的变量，提供了对内存的直接操作能力。在32位系统中指针大小为4字节，64位系统中为8字节。指针运算基于所指类型的大小进行，这是C++类型系统的重要特性。

```cpp
int number = 42;
int *ptr = &number;  // ptr存储number的内存地址

// 内存布局分析：
// number变量：在栈上分配4字节，存储值42
// ptr指针：在栈上分配8字节（64位），存储number的地址

std::cout << "变量值: " << number << std::endl;      // 42（直接访问）
std::cout << "变量地址: " << &number << std::endl;   // 如0x7ffe1234（实际地址）
std::cout << "指针值: " << ptr << std::endl;         // 与&number相同（存储的地址）
std::cout << "指针指向的值: " << *ptr << std::endl;  // 42（间接访问，解引用）

// 通过指针修改变量：间接写入内存
*ptr = 100;  // 等价于：在ptr存储的地址处写入100
std::cout << "修改后: " << number << std::endl;      // 100（原变量被修改）

// 指针类型的重要性
double d = 3.14;
double *d_ptr = &d;
// d_ptr + 1 会移动8字节（double大小）
// ptr + 1 会移动4字节（int大小）

// 空指针安全
int *null_ptr = nullptr;  // C++11空指针字面量
// *null_ptr = 10;        // 运行时错误：空指针解引用
if (null_ptr != nullptr) {
    *null_ptr = 10;       // 安全检查
}
```

**指针与数组：数组名的指针语义**

**语言特性：** 在大多数情况下，数组名会退化为指向其首元素的指针。这种设计使得数组和指针可以互换使用，但也导致了数组大小信息的丢失。指针算术基于类型大小进行自动调整。

```cpp
int arr[] = {10, 20, 30, 40, 50};
int *p = arr;  // 数组名退化为指针，指向首元素

// 等价关系分析：
// arr[i] 等价于 *(arr + i)
// &arr[i] 等价于 arr + i

// 通过指针访问数组：指针算术
for (int i = 0; i < 5; i++) {
    std::cout << *(p + i) << " ";  // 10 20 30 40 50
    // p + i 实际地址：p + i * sizeof(int)
}
std::cout << std::endl;

// 指针算术：基于类型的地址计算
p++;  // 地址增加sizeof(int)=4字节，指向arr[1]
std::cout << *p << std::endl;  // 20（arr[1]的值）

// 数组指针与指针数组的区别
int (*array_ptr)[5] = &arr;    // 指向整个数组的指针
int *pointer_array[5];         // 包含5个int指针的数组

// 多维数组的指针表示
int matrix[2][3] = {{1,2,3},{4,5,6}};
int (*row_ptr)[3] = matrix;    // 指向包含3个int的数组的指针
std::cout << (*row_ptr)[1] << std::endl;  // 2
std::cout << (*(row_ptr + 1))[1] << std::endl;  // 5

// 数组大小信息的丢失
void func(int arr[]) {  // 实际上接收的是int*
    // sizeof(arr)返回指针大小，不是数组大小
}
```

### 引用：安全的别名机制

**引用基础：编译时的名称绑定**

**实现原理：** 引用在编译时被实现为变量的别名，通常不占用额外的存储空间（可能被优化掉）。与指针不同，引用在生命周期内必须绑定到有效的对象，且不能重新绑定。这种设计提供了语法上的便利性和安全性。

```cpp
int original = 42;
int &ref = original;  // ref是original的引用（编译时别名绑定）

// 引用与指针的底层关系：
// 编译器通常将引用实现为自动解引用的指针
// 但语法上更安全，避免了空指针和悬空引用问题

std::cout << "原始值: " << original << std::endl;  // 42
std::cout << "引用值: " << ref << std::endl;       // 42（访问同一内存）

// 通过引用修改：直接操作原变量
ref = 100;  // 等价于 original = 100
std::cout << "修改后原始值: " << original << std::endl;  // 100

// 引用必须在声明时初始化（语言强制要求）
// int &invalid_ref;  // 编译错误：引用必须初始化

// 引用与const的关系
const int &const_ref = original;  // 只读引用，不能通过它修改原变量
// const_ref = 200;  // 错误：const引用不能修改

// 临时对象的生命周期延长
const int &temp_ref = 42;  // 临时对象42的生命周期被延长
// 普通引用不能绑定到临时对象：int &bad_ref = 42;  // 错误

// 引用与函数参数（重要应用）
void process(int &param) {  // 引用参数，避免拷贝
    param *= 2;  // 直接修改调用者的变量
}

int value = 10;
process(value);  // value被修改为20
```

**引用作为函数参数：避免拷贝的高效传递**

**性能优势：** 引用参数避免了大型对象的拷贝开销，同时保持了调用语法的简洁性。对于需要修改调用者数据的函数，引用参数是首选方案。

```cpp
// 引用参数：直接操作调用者的变量
void swap(int &a, int &b) {  // a和b是调用者变量的引用
    int temp = a;    // 读取a指向的内存
    a = b;           // 将b的值写入a指向的内存
    b = temp;        // 将temp写入b指向的内存
}

// 调用过程分析：
int x = 5, y = 10;
swap(x, y);  // 编译器传递x和y的引用，而不是拷贝值
// 函数内部直接操作x和y的内存
std::cout << "x = " << x << ", y = " << y << std::endl;  // x=10, y=5

// 与值传递的对比
void swap_by_value(int a, int b) {  // 接收值的拷贝
    int temp = a;
    a = b;
    b = temp;
    // 只修改了局部拷贝，不影响原变量
}

// 与指针传递的对比
void swap_by_pointer(int *a, int *b) {  // 接收指针
    int temp = *a;
    *a = *b;
    *b = temp;
}
swap_by_pointer(&x, &y);  // 需要取地址操作，语法不够直观

// 常量引用：只读访问，避免拷贝
void print_large_object(const std::vector<int> &vec) {
    // 可以读取vec，但不能修改
    for (const auto &item : vec) {
        std::cout << item << " ";
    }
}

// 右值引用（C++11）：移动语义
void take_ownership(std::vector<int> &&vec) {
    // vec是右值引用，可以"窃取"其资源
    std::vector<int> local = std::move(vec);  // 移动而非拷贝
}
```

### 指针vs引用：使用场景与区别

**关键差异对比表：**

| 特性 | 指针 | 引用 |
|------|------|------|
| **语法** | `int *p = &var;` | `int &r = var;` |
| **重新赋值** | 可以指向不同变量 | 一旦绑定，不能更改 |
| **空值** | 可以是`nullptr` | 必须绑定到有效对象 |
| **多级间接** | 支持多级指针 | 只有一级引用 |
| **内存占用** | 存储地址（通常4/8字节） | 通常是优化掉的（无额外内存） |

**使用场景建议：**
- **指针**：需要重新指向、可选参数、动态内存管理
- **引用**：函数参数、返回值、避免拷贝大对象

### 智能指针简介

**unique_ptr（独占所有权）：**
```cpp
#include <memory>

// 创建unique_ptr
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
std::unique_ptr<int[]> arr_ptr = std::make_unique<int[]>(5);

// 使用智能指针
std::cout << *ptr1 << std::endl;  // 42
arr_ptr[0] = 10;

// 所有权转移（不能复制）
std::unique_ptr<int> ptr2 = std::move(ptr1);  // ptr1变为nullptr
```

**shared_ptr（共享所有权）：**
```cpp
std::shared_ptr<int> shared1 = std::make_shared<int>(100);
std::shared_ptr<int> shared2 = shared1;  // 共享所有权

std::cout << "引用计数: " << shared1.use_count() << std::endl;  // 2
std::cout << *shared1 << std::endl;  // 100
std::cout << *shared2 << std::endl;  // 100
```

**对比Python的引用计数：**
```python
# Python使用引用计数自动管理内存
a = [1, 2, 3]  # 引用计数=1
b = a          # 引用计数=2
del a          # 引用计数=1
# 当引用计数为0时，内存自动回收
```

> **关键理解**：C++智能指针模拟了Python的自动内存管理，但提供了更精细的控制

## 动态内存管理

### new/delete操作符

**动态内存分配：运行时内存请求机制**

**底层机制：** `new`操作符在堆上分配内存并调用构造函数，`delete`操作符调用析构函数并释放内存。堆内存的生命周期由程序员显式控制，这提供了灵活性但也带来了内存泄漏的风险。

```cpp
// 动态分配单个对象：在堆上分配内存
int *dynamic_int = new int(42);  // 分配4字节，调用int构造函数
// 底层过程：
// 1. 调用operator new(sizeof(int))分配内存
// 2. 在分配的内存上调用int构造函数（对于基本类型是初始化）
// 3. 返回指向该内存的指针

std::cout << *dynamic_int << std::endl;  // 42（通过指针访问堆内存）

// 动态分配数组：连续内存块
int *dynamic_array = new int[5];  // 分配5*4=20字节连续内存
for (int i = 0; i < 5; i++) {
    dynamic_array[i] = i * 10;    // 通过指针算术访问数组元素
}

// 内存布局分析：
// dynamic_int -> 堆内存地址（如0x12345678），存储值42
// dynamic_array -> 堆内存地址（如0x12345690），存储[0,10,20,30,40]

// 必须手动释放内存：显式生命周期管理
delete dynamic_int;        // 释放单个对象：调用析构函数 + 释放内存
delete[] dynamic_array;    // 释放数组：调用每个元素的析构函数 + 释放整个数组

// 与malloc/free的区别（C风格）
int *c_style = (int*)malloc(sizeof(int));  // 只分配内存，不调用构造函数
free(c_style);                             // 只释放内存，不调用析构函数

// 异常安全考虑
void risky_function() {
    int *ptr = new int(100);
    // 如果这里抛出异常，ptr将泄漏
    delete ptr;  // 可能不会执行到
}

// 异常安全版本
void safe_function() {
    std::unique_ptr<int> ptr = std::make_unique<int>(100);
    // 即使抛出异常，unique_ptr也会自动释放内存
}
```

**对比Python的自动垃圾回收：**
```python
# Python自动管理内存
a = [1, 2, 3]  # 自动分配
# 不需要手动释放，垃圾回收器自动处理
del a          # 只是减少引用计数
```

### 内存泄漏与悬空指针

**内存泄漏示例：无法回收的堆内存**

**泄漏机制：** 当程序分配了堆内存但忘记释放，且失去了所有指向该内存的指针时，就发生了内存泄漏。操作系统无法回收这些内存，导致程序内存占用不断增加。

```cpp
void memory_leak() {
    int *leak = new int(100);  // 在堆上分配4字节
    // 忘记delete，内存泄漏！
    // leak指针在函数结束时被销毁（栈内存回收）
    // 但指向的堆内存（存储100）永远无法被访问或释放
    // 应该: delete leak;
}

// 更复杂的泄漏场景
class Resource {
    int *data;
public:
    Resource() : data(new int[1000]) {}  // 分配大量内存
    ~Resource() { delete[] data; }       // 必须定义析构函数
};

void complex_leak() {
    Resource *res = new Resource();
    // 如果忘记delete，不仅res指针泄漏，还有内部的1000个int
    // delete res;  // 必须调用以触发析构函数
}

// 循环引用导致的泄漏（shared_ptr场景）
struct Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;  // 循环引用，引用计数不会归零
};

// 检测工具：Valgrind, AddressSanitizer等可以检测内存泄漏
```

**悬空指针示例：指向无效内存的指针**

**悬空指针成因：** 当指针指向的内存已被释放或超出作用域，但指针本身仍然存在时，就形成了悬空指针。访问悬空指针会导致未定义行为，可能是崩溃或数据损坏。

```cpp
int *dangling_pointer() {
    int local_var = 50;        // 栈上分配局部变量
    return &local_var;        // 危险！返回局部变量的地址
    // 函数结束时，local_var的栈内存被回收
    // 返回的指针指向已失效的内存
}

int main() {
    int *ptr = dangling_pointer();  // ptr成为悬空指针
    // ptr指向已销毁的栈内存，行为未定义
    // *ptr = 100;  // 可能崩溃或写入随机内存
    return 0;
}

// 其他悬空指针场景
void double_delete() {
    int *ptr = new int(42);
    delete ptr;     // 第一次释放
    delete ptr;     // 错误！悬空指针再次释放
}

void use_after_free() {
    int *ptr = new int(100);
    delete ptr;     // 释放内存
    *ptr = 200;     // 错误！使用已释放的内存
}

// 智能指针自动避免悬空指针
void safe_pointer_usage() {
    std::unique_ptr<int> ptr = std::make_unique<int>(100);
    // 不需要手动delete
    // 当ptr离开作用域时，自动释放内存
    // 不会出现悬空指针问题
}

// 预防措施：释放后立即置空
void safe_delete(int *&ptr) {
    delete ptr;
    ptr = nullptr;  // 防止悬空指针
}
```

### RAII（资源获取即初始化）
**RAII示例：基于对象生命周期的自动资源管理**

**设计哲学：** RAII将资源的生命周期与对象的生命周期绑定。在构造函数中获取资源，在析构函数中释放资源。这种设计确保了异常安全，避免了资源泄漏。

```cpp
#include <fstream>
#include <memory>

// 文件资源自动管理：RAII的经典示例
void read_file_raii(const std::string &filename) {
    std::ifstream file(filename);  // 构造函数打开文件（获取资源）
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
        // 文件自动关闭：file对象析构时调用析构函数关闭文件
    }
    // 即使抛出异常，file的析构函数也会被调用，确保文件关闭
}

// RAII的内部实现原理（简化）
class FileRAII {
private:
    FILE *file_;
public:
    explicit FileRAII(const char *filename) : file_(fopen(filename, "r")) {
        if (!file_) throw std::runtime_error("无法打开文件");
    }
    
    ~FileRAII() {
        if (file_) fclose(file_);  // 析构函数确保资源释放
    }
    
    // 禁用拷贝（避免重复释放）
    FileRAII(const FileRAII&) = delete;
    FileRAII& operator=(const FileRAII&) = delete;
    
    // 允许移动（转移所有权）
    FileRAII(FileRAII&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
};

// 对比非RAII方式（容易忘记关闭）
void read_file_manual(const std::string &filename) {
    FILE *file = fopen(filename.c_str(), "r");
    if (file) {
        // 读取文件...
        // 如果这里抛出异常或提前返回，文件将不会关闭
        fclose(file);  // 必须手动关闭！容易忘记
    }
}

// RAII在现代C++中的应用
void modern_raii_examples() {
    // 1. 智能指针：内存管理
    auto ptr = std::make_unique<int>(42);
    
    // 2. 锁管理：避免死锁
    std::mutex mtx;
    {
        std::lock_guard<std::mutex> lock(mtx);  // 获取锁
        // 临界区代码
    } // 自动释放锁
    
    // 3. 自定义RAII类
    class ScopedTimer {
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
    public:
        ScopedTimer() : start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "耗时: " << duration.count() << "ms" << std::endl;
        }
    };
    
    {
        ScopedTimer timer;  // 开始计时
        // 执行需要计时的代码
    } // 自动输出耗时
}
```

> **RAII核心思想**：在构造函数中获取资源，在析构函数中释放资源
