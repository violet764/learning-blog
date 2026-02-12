# C++基础语法（综合应用篇）

## 第一章：数组与字符串

### 1.1 数组基础概念
- **数组**：相同类型元素的连续内存集合
- **下标**：从0开始的索引访问元素
- **多维数组**：数组的数组，如二维数组

### 1.2 数组声明与使用
```cpp
#include <iostream>
using namespace std;

int main() {
    // 一维数组
    int numbers[5] = {1, 2, 3, 4, 5};
    double scores[] = {85.5, 90.0, 78.5};  // 自动推断大小
    
    // 访问和修改元素
    cout << "第一个元素: " << numbers[0] << endl;
    numbers[0] = 10;
    
    // 计算数组大小
    int size = sizeof(numbers) / sizeof(numbers[0]);
    
    return 0;
}
```

### 1.3 数组遍历方法
```cpp
// 传统for循环遍历
int arr[] = {10, 20, 30, 40, 50};
int size = sizeof(arr) / sizeof(arr[0]);

cout << "数组元素: ";
for (int i = 0; i < size; i++) {
    cout << arr[i] << " ";
}
cout << endl;

// C++11范围for循环
cout << "范围for循环: ";
for (int num : arr) {
    cout << num << " ";
}
cout << endl;
```

### 1.4 多维数组
```cpp
// 二维数组：3行4列
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// 遍历二维数组
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        cout << matrix[i][j] << "\t";
    }
    cout << endl;
}
```

### 1.5 字符串处理

#### C风格字符串
```cpp
char greeting[20] = "Hello, World!";
char name[] = "Alice";

// 字符串长度
int len = strlen(greeting);

// 字符串比较
if (strcmp(greeting, "Hello") == 0) {
    cout << "字符串相等" << endl;
}

// 字符串连接
char full[50];
strcpy(full, greeting);
strcat(full, " ");
strcat(full, name);
```

#### C++字符串对象
```cpp
#include <string>
using namespace std;

string str1 = "Hello";
string str2 = "World";

// 字符串操作
string combined = str1 + " " + str2;
cout << "长度: " << combined.length() << endl;
cout << "子串: " << combined.substr(0, 5) << endl;

// 查找和替换
size_t pos = combined.find("World");
if (pos != string::npos) {
    combined.replace(pos, 5, "C++");
}
```

### 1.6 数组与字符串易错点

#### 数组越界
```cpp
int arr[5] = {1, 2, 3, 4, 5};
// cout << arr[5] << endl;  // 错误！越界访问
```

#### 字符串终止问题
```cpp
// 错误：没有空间存放'\0'
// char str[5] = {'H', 'e', 'l', 'l', 'o'};

// 正确：预留'\0'空间
char str[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
// 或使用字符串字面量
char str2[] = "Hello";
```

## 第二章：函数基础

### 2.1 函数基本概念
- **函数**：完成特定任务的独立代码块
- **参数**：函数接收的输入数据
- **返回值**：函数执行后返回的结果
- **函数原型**：函数声明，包括返回类型、函数名和参数列表

### 2.2 函数定义与调用
```cpp
#include <iostream>
using namespace std;

// 函数声明（原型）
int add(int a, int b);
void printMessage(string message);

double calculateAverage(double arr[], int size);

int main() {
    // 函数调用
    int result = add(5, 3);
    printMessage("函数调用示例");
    
    double scores[] = {85.5, 90.0, 78.5};
    double avg = calculateAverage(scores, 3);
    
    return 0;
}

// 函数定义
int add(int a, int b) {
    return a + b;
}

void printMessage(string message) {
    cout << "消息: " << message << endl;
}

double calculateAverage(double arr[], int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}
```

### 2.3 函数重载
```cpp
// 同名函数，参数不同
int max(int a, int b) {
    return (a > b) ? a : b;
}

double max(double a, double b) {
    return (a > b) ? a : b;
}

int max(int a, int b, int c) {
    return max(max(a, b), c);
}

// 编译器根据参数类型选择合适版本
cout << max(3, 5) << endl;          // 调用int版本
cout << max(3.5, 2.8) << endl;      // 调用double版本
cout << max(1, 2, 3) << endl;       // 调用三参数版本
```

### 2.4 默认参数
```cpp
void printInfo(string name, int age = 18, string city = "北京") {
    cout << "姓名: " << name << endl;
    cout << "年龄: " << age << endl;
    cout << "城市: " << city << endl;
    cout << "---" << endl;
}

// 不同调用方式
printInfo("张三");                  // 使用所有默认参数
printInfo("李四", 25);             // 部分使用默认参数
printInfo("王五", 30, "上海");     // 不使用默认参数
```

### 2.5 递归函数
```cpp
// 阶乘计算
int factorial(int n) {
    if (n <= 1) {
        return 1;  // 基本情况
    } else {
        return n * factorial(n - 1);  // 递归调用
    }
}

// 斐波那契数列
int fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

cout << "5的阶乘: " << factorial(5) << endl;
cout << "斐波那契第10项: " << fibonacci(10) << endl;
```

### 2.6 函数易错点

#### 函数未声明
```cpp
// 错误：函数定义在调用之后且未声明
int main() {
    int result = add(2, 3);  // 错误！
    return 0;
}

int add(int a, int b) {
    return a + b;
}

// 正确：先声明函数
int add(int a, int b);  // 函数声明
```

#### 参数传递误解
```cpp
// 值传递：不改变原变量
void swap(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
}

// 引用传递：改变原变量
void swapRef(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}
```

## 第三章：预处理指令

### 3.1 预处理基础
- **预处理指令**：编译前处理的特殊指令
- **宏定义**：文本替换机制
- **条件编译**：根据条件决定是否编译代码

### 3.2 常用预处理指令
```cpp
#include <iostream>
using namespace std;

// 宏定义
#define PI 3.14159
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SQUARE(x) ((x) * (x))

// 条件编译
#define DEBUG
#define VERSION 2

int main() {
    // 使用宏
    double area = PI * SQUARE(5.0);
    cout << "最大值: " << MAX(10, 20) << endl;
    
    // 条件编译
#ifdef DEBUG
    cout << "调试信息" << endl;
#endif
    
#if VERSION == 2
    cout << "版本2" << endl;
#endif
    
    return 0;
}
```

### 3.3 头文件保护
```cpp
// myheader.h
#ifndef MYHEADER_H     // 如果未定义
#define MYHEADER_H     // 则定义

// 头文件内容
class MyClass {
public:
    void doSomething();
};

#endif // MYHEADER_H
```

### 3.4 调试宏
```cpp
#define DEBUG_LEVEL 2

#if DEBUG_LEVEL >= 1
#define DEBUG_INFO(msg) cout << "[INFO] " << msg << endl
#else
#define DEBUG_INFO(msg)  // 空定义
#endif

#if DEBUG_LEVEL >= 2
#define DEBUG_DETAIL(msg) cout << "[DETAIL] " << msg << endl
#else
#define DEBUG_DETAIL(msg)
#endif

DEBUG_INFO("程序开始");
DEBUG_DETAIL("详细调试信息");
```

### 3.5 预处理易错点

#### 宏参数未加括号
```cpp
// 错误：宏展开优先级问题
#define SQUARE(x) x * x
int result = SQUARE(2 + 3);  // 展开为 2 + 3 * 2 + 3 = 11

// 正确：为参数和表达式加括号
#define SQUARE(x) ((x) * (x))
int result = SQUARE(2 + 3);  // 展开为 ((2 + 3) * (2 + 3)) = 25
```

#### 宏副作用
```cpp
// 错误：参数可能被多次求值
#define MAX(a, b) ((a) > (b) ? (a) : (b))
int x = 5, y = 10;
int result = MAX(++x, y);  // x可能被增加两次

// 正确：使用内联函数
inline int max(int a, int b) {
    return (a > b) ? a : b;
}
```

## 第四章：综合应用案例

### 4.1 学生成绩管理系统
```cpp
#include <iostream>
#include <string>
using namespace std;

// 常量定义
const int MAX_STUDENTS = 100;
const int MAX_SUBJECTS = 5;

// 函数声明
void inputScores(double scores[][MAX_SUBJECTS], int studentCount, int subjectCount);
void calculateAverages(double scores[][MAX_SUBJECTS], double averages[], int studentCount, int subjectCount);
void displayResults(double scores[][MAX_SUBJECTS], double averages[], int studentCount, int subjectCount);

int main() {
    double scores[MAX_STUDENTS][MAX_SUBJECTS];
    double averages[MAX_STUDENTS];
    int studentCount, subjectCount;
    
    cout << "请输入学生人数: ";
    cin >> studentCount;
    cout << "请输入科目数量: ";
    cin >> subjectCount;
    
    inputScores(scores, studentCount, subjectCount);
    calculateAverages(scores, averages, studentCount, subjectCount);
    displayResults(scores, averages, studentCount, subjectCount);
    
    return 0;
}

void inputScores(double scores[][MAX_SUBJECTS], int studentCount, int subjectCount) {
    for (int i = 0; i < studentCount; i++) {
        cout << "第" << i+1 << "个学生的成绩: " << endl;
        for (int j = 0; j < subjectCount; j++) {
            cout << "科目" << j+1 << ": ";
            cin >> scores[i][j];
        }
    }
}

void calculateAverages(double scores[][MAX_SUBJECTS], double averages[], int studentCount, int subjectCount) {
    for (int i = 0; i < studentCount; i++) {
        double sum = 0.0;
        for (int j = 0; j < subjectCount; j++) {
            sum += scores[i][j];
        }
        averages[i] = sum / subjectCount;
    }
}

void displayResults(double scores[][MAX_SUBJECTS], double averages[], int studentCount, int subjectCount) {
    cout << "\\n成绩报告:\\n" << endl;
    for (int i = 0; i < studentCount; i++) {
        cout << "学生" << i+1 << ": ";
        for (int j = 0; j < subjectCount; j++) {
            cout << scores[i][j] << " ";
        }
        cout << "平均分: " << averages[i] << endl;
    }
}
```

### 4.2 字符串处理工具
```cpp
#include <iostream>
#include <string>
#include <cstring>
using namespace std;

// 函数声明
void stringStats(const char* str);
void reverseString(char* str);
bool isPalindrome(const char* str);

int main() {
    char input[100];
    cout << "请输入一个字符串: ";
    cin.getline(input, 100);
    
    stringStats(input);
    
    if (isPalindrome(input)) {
        cout << "是回文字符串" << endl;
    } else {
        cout << "不是回文字符串" << endl;
    }
    
    reverseString(input);
    cout << "反转后: " << input << endl;
    
    return 0;
}

void stringStats(const char* str) {
    int length = strlen(str);
    int letters = 0, digits = 0, spaces = 0;
    
    for (int i = 0; i < length; i++) {
        if (isalpha(str[i])) letters++;
        else if (isdigit(str[i])) digits++;
        else if (isspace(str[i])) spaces++;
    }
    
    cout << "长度: " << length << endl;
    cout << "字母: " << letters << endl;
    cout << "数字: " << digits << endl;
    cout << "空格: " << spaces << endl;
}

void reverseString(char* str) {
    int length = strlen(str);
    for (int i = 0; i < length / 2; i++) {
        char temp = str[i];
        str[i] = str[length - 1 - i];
        str[length - 1 - i] = temp;
    }
}

bool isPalindrome(const char* str) {
    int length = strlen(str);
    for (int i = 0; i < length / 2; i++) {
        if (str[i] != str[length - 1 - i]) {
            return false;
        }
    }
    return true;
}
```

## 第五章：学习总结与进阶

### 5.1 核心知识点回顾
- **数组和字符串**：理解连续存储和随机访问特性
- **函数编程**：掌握模块化设计和代码复用
- **预处理技术**：了解编译前处理机制

### 5.2 最佳实践建议
1. **数组使用**：始终检查边界，避免越界访问
2. **字符串处理**：优先使用C++ string类，更安全便捷
3. **函数设计**：保持函数功能单一，参数合理
4. **宏使用**：谨慎使用宏，优先考虑内联函数

### 5.3 进阶学习方向
- **标准模板库(STL)**：vector、list、map等容器
- **面向对象编程**：类、对象、继承、多态
- **内存管理**：动态内存分配、智能指针
- **文件操作**：读写文件、数据持久化

通过这两篇综合学习，您已经掌握了C++基础语法的核心内容，可以开始学习更高级的编程概念和技术了。