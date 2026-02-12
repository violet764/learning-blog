# C++核心特性（基础篇）

## 第一章：面向对象基础

### 1.1 核心概念
- **面向对象编程(OOP)**：以对象为核心的编程范式
- **对象**：具有状态和行为的实体
- **类**：创建对象的蓝图或模板
- **封装**：将数据和操作数据的方法捆绑在一起

### 1.2 基本语法与示例
```cpp
#include <iostream>
#include <string>
using namespace std;

class Person {
public:
    string name;
    int age;
    
    void introduce() {
        cout << "我叫" << name << ", 今年" << age << "岁。" << endl;
    }
    
    void setAge(int newAge) {
        if (newAge > 0 && newAge < 150) {
            age = newAge;
        }
    }
};

int main() {
    Person person1;
    person1.name = "张三";
    person1.setAge(25);
    person1.introduce();
    
    return 0;
}
```

### 1.3 易错点分析
- **忘记类定义分号**：类定义必须以分号结尾
- **直接访问私有成员**：私有成员只能通过公有方法访问
- **对象未初始化**：成员变量应提供默认值

## 第二章：类与对象（封装）

### 2.1 封装核心概念
- **封装**：将数据和对数据的操作捆绑在一起，并隐藏实现细节
- **访问控制**：通过public、protected、private控制成员可见性
- **数据隐藏**：保护数据不被外部直接访问和修改

### 2.2 完整封装示例
```cpp
class BankAccount {
private:
    string accountNumber;
    double balance;
    
public:
    BankAccount(string accNum, double initialBalance) {
        accountNumber = accNum;
        balance = initialBalance;
    }
    
    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    double getBalance() const {
        return balance;
    }
};
```

### 2.3 封装最佳实践
- 所有数据成员设为private
- 提供完整的公有接口
- 使用const成员函数提供只读访问
- 通过方法控制数据修改，确保业务规则

### 2.4 日期类完整封装示例
```cpp
class Date {
private:
    int year, month, day;
    
    bool isValidDate(int y, int m, int d) {
        if (y < 1900 || y > 2100) return false;
        if (m < 1 || m > 12) return false;
        
        int daysInMonth;
        if (m == 2) {
            bool isLeap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
            daysInMonth = isLeap ? 29 : 28;
        } else if (m == 4 || m == 6 || m == 9 || m == 11) {
            daysInMonth = 30;
        } else {
            daysInMonth = 31;
        }
        
        return d >= 1 && d <= daysInMonth;
    }
    
public:
    Date(int y, int m, int d) {
        if (isValidDate(y, m, d)) {
            year = y;
            month = m;
            day = d;
        } else {
            year = 2000; month = 1; day = 1;
        }
    }
    
    bool setDate(int y, int m, int d) {
        if (isValidDate(y, m, d)) {
            year = y; month = m; day = d;
            return true;
        }
        return false;
    }
    
    void display() const {
        cout << year << "年" << month << "月" << day << "日" << endl;
    }
};
```

## 第三章：构造与析构函数

### 3.1 核心概念
- **构造函数**：对象创建时自动调用，用于初始化
- **析构函数**：对象销毁时自动调用，用于清理资源
- **对象生命周期**：从构造到析构的完整过程

### 3.2 构造函数类型
```cpp
class Student {
private:
    char* name;
    int age;
    
public:
    // 默认构造函数
    Student() : name(new char[10]), age(0) {
        strcpy(name, "未知");
    }
    
    // 带参构造函数
    Student(const char* n, int a) : age(a) {
        name = new char[strlen(n) + 1];
        strcpy(name, n);
    }
    
    // 拷贝构造函数（深拷贝）
    Student(const Student& other) : age(other.age) {
        name = new char[strlen(other.name) + 1];
        strcpy(name, other.name);
    }
    
    // 析构函数
    ~Student() {
        delete[] name;
    }
};
```

### 3.3 初始化列表语法
```cpp
class Point {
private:
    int x, y;
    const int id;  // 常量成员必须使用初始化列表
    
public:
    Point(int xVal, int yVal, int idVal) : x(xVal), y(yVal), id(idVal) {
        // 构造函数体
    }
};
```

### 3.4 RAII模式（资源获取即初始化）
```cpp
class FileHandler {
private:
    FILE* file;
    
public:
    FileHandler(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) throw runtime_error("无法打开文件");
    }
    
    ~FileHandler() {
        if (file) fclose(file);
    }
    
    // 禁止拷贝（避免重复释放）
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};
```

### 3.5 构造与析构易错点

#### 浅拷贝问题
```cpp
// 错误：默认拷贝构造函数进行浅拷贝
class BadString {
    char* data;
public:
    BadString(const char* str) {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
    }
    
    ~BadString() { delete[] data; }  // 多个对象共享同一内存
};

// 正确：自定义深拷贝
class GoodString {
    char* data;
public:
    GoodString(const char* str) {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
    }
    
    GoodString(const GoodString& other) {  // 深拷贝
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
    }
    
    ~GoodString() { delete[] data; }
};
```

#### 内存泄漏问题
```cpp
// 错误：忘记释放资源
class MemoryLeak {
    int* array;
public:
    MemoryLeak(int size) {
        array = new int[size];  // 分配内存
    }
    // 缺少析构函数！
};

// 正确：在析构函数中释放
class NoMemoryLeak {
    int* array;
public:
    NoMemoryLeak(int size) { array = new int[size]; }
    ~NoMemoryLeak() { delete[] array; }  // 释放内存
};
```

## 第四章：综合应用案例

### 4.1 完整的银行账户系统
```cpp
#include <iostream>
#include <string>
#include <stdexcept>
using namespace std;

class BankAccount {
private:
    string accountNumber;
    string ownerName;
    double balance;
    
    bool isValidAmount(double amount) const {
        return amount > 0;
    }
    
public:
    BankAccount(const string& accNum, const string& name, double initialBalance = 0.0) 
        : accountNumber(accNum), ownerName(name) {
        if (initialBalance < 0) {
            throw invalid_argument("初始余额不能为负数");
        }
        balance = initialBalance;
    }
    
    bool deposit(double amount) {
        if (isValidAmount(amount)) {
            balance += amount;
            return true;
        }
        return false;
    }
    
    bool withdraw(double amount) {
        if (isValidAmount(amount) && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    bool transfer(BankAccount& toAccount, double amount) {
        if (withdraw(amount)) {
            return toAccount.deposit(amount);
        }
        return false;
    }
    
    double getBalance() const { return balance; }
    string getAccountNumber() const { return accountNumber; }
    string getOwnerName() const { return ownerName; }
    
    void displayInfo() const {
        cout << "账户号: " << accountNumber << endl;
        cout << "户主: " << ownerName << endl;
        cout << "余额: ￥" << balance << endl;
    }
};

class Bank {
private:
    static const int MAX_ACCOUNTS = 100;
    BankAccount* accounts[MAX_ACCOUNTS];
    int accountCount;
    
public:
    Bank() : accountCount(0) {}
    
    ~Bank() {
        for (int i = 0; i < accountCount; i++) {
            delete accounts[i];
        }
    }
    
    BankAccount* createAccount(const string& accNum, const string& name, double balance = 0.0) {
        if (accountCount < MAX_ACCOUNTS) {
            accounts[accountCount] = new BankAccount(accNum, name, balance);
            return accounts[accountCount++];
        }
        return nullptr;
    }
    
    BankAccount* findAccount(const string& accNum) {
        for (int i = 0; i < accountCount; i++) {
            if (accounts[i]->getAccountNumber() == accNum) {
                return accounts[i];
            }
        }
        return nullptr;
    }
    
    void displayAllAccounts() const {
        cout << "=== 银行账户列表 ===" << endl;
        for (int i = 0; i < accountCount; i++) {
            accounts[i]->displayInfo();
            cout << "---" << endl;
        }
    }
};

int main() {
    Bank bank;
    
    // 创建账户
    auto acc1 = bank.createAccount("1001", "张三", 1000.0);
    auto acc2 = bank.createAccount("1002", "李四", 500.0);
    
    // 账户操作
    acc1->deposit(200.0);
    acc1->withdraw(100.0);
    acc1->transfer(*acc2, 300.0);
    
    bank.displayAllAccounts();
    
    return 0;
}
```

### 4.2 学生成绩管理系统
```cpp
class Student {
private:
    string name;
    int studentId;
    double scores[5];  // 5门课程成绩
    int scoreCount;
    
    bool isValidScore(double score) const {
        return score >= 0 && score <= 100;
    }
    
public:
    Student(const string& n, int id) : name(n), studentId(id), scoreCount(0) {}
    
    bool addScore(double score) {
        if (scoreCount < 5 && isValidScore(score)) {
            scores[scoreCount++] = score;
            return true;
        }
        return false;
    }
    
    double getAverage() const {
        if (scoreCount == 0) return 0.0;
        
        double sum = 0.0;
        for (int i = 0; i < scoreCount; i++) {
            sum += scores[i];
        }
        return sum / scoreCount;
    }
    
    char getGrade() const {
        double avg = getAverage();
        if (avg >= 90) return 'A';
        else if (avg >= 80) return 'B';
        else if (avg >= 70) return 'C';
        else if (avg >= 60) return 'D';
        else return 'F';
    }
    
    void displayInfo() const {
        cout << "学号: " << studentId << endl;
        cout << "姓名: " << name << endl;
        cout << "成绩: ";
        for (int i = 0; i < scoreCount; i++) {
            cout << scores[i] << " ";
        }
        cout << endl;
        cout << "平均分: " << getAverage() << endl;
        cout << "等级: " << getGrade() << endl;
    }
};

class StudentManager {
private:
    static const int MAX_STUDENTS = 50;
    Student* students[MAX_STUDENTS];
    int studentCount;
    
public:
    StudentManager() : studentCount(0) {}
    
    ~StudentManager() {
        for (int i = 0; i < studentCount; i++) {
            delete students[i];
        }
    }
    
    bool addStudent(const string& name, int id) {
        if (studentCount < MAX_STUDENTS) {
            students[studentCount++] = new Student(name, id);
            return true;
        }
        return false;
    }
    
    Student* findStudent(int id) {
        for (int i = 0; i < studentCount; i++) {
            if (true) {  // 假设有getId方法
                return students[i];
            }
        }
        return nullptr;
    }
    
    void displayAllStudents() const {
        cout << "=== 学生信息 ===" << endl;
        for (int i = 0; i < studentCount; i++) {
            students[i]->displayInfo();
            cout << "---" << endl;
        }
    }
    
    void displayStatistics() const {
        if (studentCount == 0) {
            cout << "暂无学生数据" << endl;
            return;
        }
        
        int gradeCount[5] = {0};  // A,B,C,D,F
        for (int i = 0; i < studentCount; i++) {
            char grade = students[i]->getGrade();
            switch (grade) {
                case 'A': gradeCount[0]++; break;
                case 'B': gradeCount[1]++; break;
                case 'C': gradeCount[2]++; break;
                case 'D': gradeCount[3]++; break;
                case 'F': gradeCount[4]++; break;
            }
        }
        
        cout << "=== 成绩统计 ===" << endl;
        cout << "A等: " << gradeCount[0] << "人" << endl;
        cout << "B等: " << gradeCount[1] << "人" << endl;
        cout << "C等: " << gradeCount[2] << "人" << endl;
        cout << "D等: " << gradeCount[3] << "人" << endl;
        cout << "F等: " << gradeCount[4] << "人" << endl;
    }
};
```

## 第五章：学习总结

### 5.1 核心概念回顾
- **面向对象基础**：类、对象、封装的基本概念
- **访问控制**：public、protected、private的使用场景
- **构造与析构**：对象生命周期管理，RAII模式
- **封装原则**：数据保护，接口与实现分离

### 5.2 最佳实践
1. **类设计原则**：单一职责，高内聚低耦合
2. **封装技巧**：私有数据，公有接口，const成员函数
3. **资源管理**：RAII模式，避免内存泄漏
4. **异常安全**：构造函数中的异常处理

### 5.3 常见模式
- **工厂模式**：通过工厂方法创建对象
- **RAII模式**：资源获取即初始化
- **单例模式**：确保类只有一个实例
- **观察者模式**：对象间的消息通知机制

通过掌握这些基础核心特性，您已经建立了坚实的C++面向对象编程基础，可以开始学习更高级的特性如继承、多态和模板编程。