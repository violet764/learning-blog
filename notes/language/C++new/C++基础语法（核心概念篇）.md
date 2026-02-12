# C++基础语法（核心概念篇）

## 第一章：变量与数据类型

### 1.1 核心概念
- **变量**：程序中用于存储数据的命名内存位置
- **数据类型**：定义变量可以存储的数据种类和大小
- **常量**：值不可改变的变量
- **内存管理**：变量类型决定内存分配大小

### 1.2 基本语法与示例
```cpp
#include <iostream>
using namespace std;

int main() {
    // 基本数据类型声明
    int age = 25;                    // 整型变量
    double salary = 5000.50;        // 双精度浮点型
    char grade = 'A';                // 字符型
    bool isActive = true;            // 布尔型
    
    // 常量声明
    const float PI = 3.14159f;       // 常量浮点数
    
    // 类型转换
    int num1 = 10;
    double num2 = num1;              // 隐式转换
    double price = 99.99;
    int intPrice = (int)price;       // 显式转换
    
    return 0;
}
```

### 1.3 常见易错点
- **未初始化变量**：变量包含随机值，应总是初始化
- **整数溢出**：超出数据类型表示范围，应使用更大的类型
- **类型转换错误**：整数除法会截断小数，应确保至少一个操作数是浮点数

## 第二章：运算符与表达式

### 2.1 运算符分类
- **算术运算符**：+、-、*、/、%
- **关系运算符**：>、<、==、!=、>=、<=
- **逻辑运算符**：&&、||、!
- **赋值运算符**：=、+=、-=、*=、/=
- **自增自减**：++、--（前缀和后缀）

### 2.2 优先级与结合性
```cpp
// 优先级示例：乘法高于加法
int result = 2 + 3 * 4;  // 结果为14，不是20

// 使用括号明确优先级
int clear = (2 + 3) * 4;  // 结果为20

// 条件运算符（三元运算符）
int score = 85;
string result = (score >= 60) ? "及格" : "不及格";
```

### 2.3 复合赋值与自增自减
```cpp
int count = 0;
count += 5;     // 等价于 count = count + 5
count++;        // 后缀自增
++count;        // 前缀自增

int i = 5;
cout << ++i;    // 输出6，i现在为6
cout << i++;    // 输出6，i现在为7
```

### 2.4 易错点分析
- **运算符优先级混淆**：使用括号明确计算顺序
- **整数溢出**：超出int范围时使用long long
- **浮点数精度问题**：使用容差比较而非精确相等

## 第三章：流程控制语句

### 3.1 分支结构

#### if-else语句
```cpp
int score = 85;
if (score >= 90) {
    cout << "优秀" << endl;
} else if (score >= 80) {
    cout << "良好" << endl;
} else {
    cout << "继续努力" << endl;
}
```

#### switch语句
```cpp
char grade = 'B';
switch (grade) {
    case 'A': cout << "优秀"; break;
    case 'B': cout << "良好"; break;
    case 'C': cout << "及格"; break;
    default: cout << "不及格"; break;
}
```

**注意**：switch语句必须使用break，否则会继续执行后续case

### 3.2 循环结构

#### for循环
```cpp
// 基本for循环
for (int i = 1; i <= 5; i++) {
    cout << i << " ";
}

// 嵌套循环（打印乘法表）
for (int i = 1; i <= 9; i++) {
    for (int j = 1; j <= i; j++) {
        cout << j << "×" << i << "=" << i*j << "\t";
    }
    cout << endl;
}
```

#### while和do-while循环
```cpp
// while循环
int i = 1;
while (i <= 3) {
    cout << i << " ";
    i++;
}

// do-while循环（至少执行一次）
int j = 1;
do {
    cout << j << " ";
    j++;
} while (j <= 3);
```

### 3.3 循环控制语句

#### break和continue
```cpp
// break：立即退出循环
for (int i = 1; i <= 10; i++) {
    if (i % 3 == 0) {
        cout << "找到第一个3的倍数: " << i << endl;
        break;
    }
}

// continue：跳过当前迭代
for (int i = 1; i <= 10; i++) {
    if (i % 3 == 0) {
        continue;  // 跳过3的倍数
    }
    cout << i << " ";
}
```

### 3.4 易错点与注意事项

#### 无限循环
```cpp
// 错误示例：缺少循环变量更新
int i = 0;
while (i < 5) {
    cout << i << " ";
    // 忘记 i++
}
```

#### 作用域问题
```cpp
// 错误：变量作用域仅限于代码块
if (true) {
    int x = 10;
}
// cout << x;  // 错误！x未定义

// 正确：在外部声明变量
int x;
if (true) {
    x = 10;
}
cout << x;  // 正确
```

## 第四章：综合应用示例

### 4.1 成绩评级系统
```cpp
#include <iostream>
using namespace std;

int main() {
    int score;
    cout << "请输入成绩: ";
    cin >> score;
    
    if (score < 0 || score > 100) {
        cout << "成绩无效!" << endl;
    } else {
        char grade;
        if (score >= 90) grade = 'A';
        else if (score >= 80) grade = 'B';
        else if (score >= 70) grade = 'C';
        else if (score >= 60) grade = 'D';
        else grade = 'F';
        
        cout << "等级: " << grade << endl;
    }
    
    return 0;
}
```

### 4.2 数字猜谜游戏
```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    srand(time(0));  // 设置随机种子
    int secret = rand() % 100 + 1;  // 1-100的随机数
    int guess, attempts = 0;
    
    cout << "猜数字游戏 (1-100)" << endl;
    
    do {
        cout << "请输入你的猜测: ";
        cin >> guess;
        attempts++;
        
        if (guess > secret) {
            cout << "太大了!" << endl;
        } else if (guess < secret) {
            cout << "太小了!" << endl;
        } else {
            cout << "恭喜! 你在" << attempts << "次尝试中猜对了!" << endl;
        }
    } while (guess != secret);
    
    return 0;
}
```

## 第五章：核心概念总结

### 5.1 数据类型要点
- 理解基本数据类型的范围和用途
- 掌握类型转换的时机和方法
- 注意变量的作用域和生命周期

### 5.2 运算符要点
- 牢记运算符优先级顺序
- 合理使用括号明确计算顺序
- 注意不同数据类型的运算规则

### 5.3 流程控制要点
- 根据场景选择合适的控制结构
- 确保循环有明确的终止条件
- 合理使用break和continue控制流程

### 5.4 最佳实践
1. **变量命名**：使用有意义的名称，遵循命名规范
2. **代码格式化**：保持一致的缩进和代码风格
3. **注释文档**：为复杂逻辑添加必要的注释
4. **错误处理**：考虑边界条件和异常情况

通过掌握这些核心概念，您已经建立了坚实的C++编程基础，可以开始学习更复杂的数据结构和算法了。