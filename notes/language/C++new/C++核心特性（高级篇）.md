# C++核心特性（高级篇）

## 第一章：继承与多态

### 1.1 继承核心概念
- **继承**：允许派生类基于基类创建，获得基类的属性和方法
- **多态**：同一接口在不同情况下表现出不同行为的能力
- **虚函数**：在基类中声明为virtual的函数，可以在派生类中重写

### 1.2 基本语法与示例
```cpp
#include <iostream>
#include <string>
using namespace std;

// 基类：形状
class Shape {
protected:
    string name;
    
public:
    Shape(const string& n) : name(n) {}
    
    // 虚函数
    virtual double area() const {
        return 0.0;
    }
    
    virtual void display() const {
        cout << "形状: " << name << endl;
    }
    
    // 虚析构函数（重要！）
    virtual ~Shape() {}
};

// 派生类：圆形
class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : Shape("圆形"), radius(r) {}
    
    // 重写虚函数
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void display() const override {
        Shape::display();
        cout << "半径: " << radius << ", 面积: " << area() << endl;
    }
};

// 派生类：矩形
class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : Shape("矩形"), width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
    
    void display() const override {
        Shape::display();
        cout << "宽度: " << width << ", 高度: " << height 
             << ", 面积: " << area() << endl;
    }
};

// 多态演示
void polymorphicDemo() {
    Shape* shapes[2];
    shapes[0] = new Circle(5.0);
    shapes[1] = new Rectangle(4.0, 6.0);
    
    for (int i = 0; i < 2; i++) {
        shapes[i]->display();  // 动态绑定
        cout << "面积: " << shapes[i]->area() << endl;
    }
    
    // 释放内存
    for (int i = 0; i < 2; i++) {
        delete shapes[i];
    }
}
```

### 1.3 抽象基类与纯虚函数
```cpp
// 抽象基类：动物
class Animal {
protected:
    string name;
    
public:
    Animal(const string& n) : name(n) {}
    
    // 纯虚函数：使类成为抽象类
    virtual void speak() const = 0;
    virtual void move() const = 0;
    
    virtual ~Animal() = default;
};

// 派生类：狗
class Dog : public Animal {
public:
    Dog(const string& n) : Animal(n) {}
    
    void speak() const override {
        cout << name << "说: 汪汪!" << endl;
    }
    
    void move() const override {
        cout << name << "用四条腿跑" << endl;
    }
};

// 派生类：鸟
class Bird : public Animal {
public:
    Bird(const string& n) : Animal(n) {}
    
    void speak() const override {
        cout << name << "说: 叽叽!" << endl;
    }
    
    void move() const override {
        cout << name << "用翅膀飞" << endl;
    }
};
```

### 1.4 接口类（纯抽象类）
```cpp
// 接口类：可打印
class Printable {
public:
    virtual void print() const = 0;
    virtual ~Printable() = default;
};

// 接口类：可序列化
class Serializable {
public:
    virtual string serialize() const = 0;
    virtual void deserialize(const string& data) = 0;
    virtual ~Serializable() = default;
};

// 实现多个接口的类
class Document : public Printable, public Serializable {
private:
    string content;
    
public:
    Document(const string& c) : content(c) {}
    
    void print() const override {
        cout << "打印文档: " << content << endl;
    }
    
    string serialize() const override {
        return "DOCUMENT:" + content;
    }
    
    void deserialize(const string& data) override {
        if (data.find("DOCUMENT:") == 0) {
            content = data.substr(9);
        }
    }
};
```

### 1.5 继承与多态易错点

#### 缺少虚析构函数
```cpp
class Base {
public:
    Base() {}
    ~Base() { cout << "Base析构" << endl; }  // 非虚析构函数！
};

class Derived : public Base {
public:
    Derived() {}
    ~Derived() { cout << "Derived析构" << endl; }
};

// 错误使用
Base* ptr = new Derived();
delete ptr;  // 只调用Base的析构函数，Derived资源泄漏

// 修正：使用虚析构函数
virtual ~Base() { cout << "Base析构" << endl; }
```

#### 对象切片（Object Slicing）
```cpp
class Base {
public:
    virtual void show() { cout << "Base" << endl; }
};

class Derived : public Base {
public:
    void show() override { cout << "Derived" << endl; }
};

// 错误：对象切片
Derived d;
Base b = d;  // Derived特有部分被切掉
b.show();    // 输出"Base"，不是"Derived"

// 修正：使用引用或指针
Base& b = d;  // 或 Base* b = &d;
b.show();     // 正确输出"Derived"
```

## 第二章：模板（泛型编程）

### 2.1 模板核心概念
- **模板**：允许编写与数据类型无关的通用代码
- **泛型编程**：编写可处理多种数据类型的通用算法
- **模板实例化**：编译器根据具体类型生成特定版本的代码

### 2.2 函数模板
```cpp
// 函数模板：交换两个值
template <typename T>
void swapValues(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// 函数模板：查找最大值
template <typename T>
T findMax(const T arr[], int size) {
    T maxVal = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
        }
    }
    return maxVal;
}

// 使用示例
int intArr[] = {3, 7, 2, 9, 1};
double doubleArr[] = {3.5, 7.2, 2.8, 9.1, 1.6};

cout << "整数最大值: " << findMax(intArr, 5) << endl;
cout << "浮点数最大值: " << findMax(doubleArr, 5) << endl;
```

### 2.3 类模板
```cpp
// 类模板：栈
template <typename T, int MAX_SIZE = 100>
class Stack {
private:
    T data[MAX_SIZE];
    int topIndex;
    
public:
    Stack() : topIndex(-1) {}
    
    bool push(const T& value) {
        if (topIndex < MAX_SIZE - 1) {
            data[++topIndex] = value;
            return true;
        }
        return false;
    }
    
    bool pop(T& value) {
        if (topIndex >= 0) {
            value = data[topIndex--];
            return true;
        }
        return false;
    }
    
    T top() const {
        if (topIndex >= 0) {
            return data[topIndex];
        }
        throw out_of_range("栈为空");
    }
    
    bool isEmpty() const { return topIndex == -1; }
    int size() const { return topIndex + 1; }
};

// 使用示例
Stack<int> intStack;
Stack<string> stringStack;
Stack<double, 5> smallStack;
```

### 2.4 模板特化
```cpp
// 通用模板
template <typename T>
class TypeInfo {
public:
    static string getName() {
        return "未知类型";
    }
};

// 模板特化
template <>
class TypeInfo<int> {
public:
    static string getName() {
        return "整数类型";
    }
};

template <>
class TypeInfo<double> {
public:
    static string getName() {
        return "双精度浮点数";
    }
};

template <>
class TypeInfo<string> {
public:
    static string getName() {
        return "字符串类型";
    }
};

// 使用特化
cout << TypeInfo<int>::getName() << endl;     // 整数类型
cout << TypeInfo<double>::getName() << endl;  // 双精度浮点数
```

### 2.5 模板易错点

#### 模板定义在源文件中
```cpp
// 错误：模板定义在.cpp文件中
// mytemplate.cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}

// 修正：模板定义必须放在头文件中
// mytemplate.h
template <typename T>
T add(T a, T b) {
    return a + b;
}
```

#### 不支持的类型操作
```cpp
template <typename T>
T getAverage(const T arr[], int size) {
    T sum = T();
    for (int i = 0; i < size; i++) {
        sum += arr[i];  // 要求T支持+=操作
    }
    return sum / size;  // 要求T支持/操作
}

class MyClass {
    // 没有定义+=和/运算符
};

// 编译错误！
MyClass arr[3];
getAverage(arr, 3);

// 修正：使用静态断言或概念约束
template <typename T>
T getAverage(const T arr[], int size) {
    static_assert(std::is_arithmetic_v<T>, "T必须是算术类型");
    T sum = T();
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}
```

## 第三章：命名空间与异常处理

### 3.1 命名空间核心概念
- **命名空间**：将代码组织到逻辑组，防止名称冲突
- **名称隔离**：不同命名空间中的同名标识符不会冲突
- **代码组织**：实现模块化设计

### 3.2 命名空间使用
```cpp
namespace MathUtils {
    const double PI = 3.14159;
    
    double calculateCircleArea(double radius) {
        if (radius < 0) {
            throw invalid_argument("半径不能为负数");
        }
        return PI * radius * radius;
    }
}

namespace StringUtils {
    string toUpperCase(const string& str) {
        string result = str;
        for (char& c : result) {
            c = toupper(c);
        }
        return result;
    }
}

// 嵌套命名空间
namespace MyApp {
    namespace Database {
        class Connection {
            // 数据库连接类
        };
    }
}

// 使用
cout << MathUtils::calculateCircleArea(5.0) << endl;
cout << StringUtils::toUpperCase("hello") << endl;
MyApp::Database::Connection db;
```

### 3.3 异常处理
```cpp
// 自定义异常类
class MyException : public exception {
    string message;
public:
    MyException(const string& msg) : message(msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class FileNotFoundException : public MyException {
public:
    FileNotFoundException(const string& filename) 
        : MyException("文件未找到: " + filename) {}
};

// 异常处理示例
void processFile(const string& filename) {
    try {
        ifstream file(filename);
        if (!file.is_open()) {
            throw FileNotFoundException(filename);
        }
        
        // 文件操作
        string line;
        while (getline(file, line)) {
            cout << line << endl;
        }
        
    } catch (const FileNotFoundException& e) {
        cerr << "文件异常: " << e.what() << endl;
        throw;  // 重新抛出
    } catch (const exception& e) {
        cerr << "标准异常: " << e.what() << endl;
    } catch (...) {
        cerr << "未知异常" << endl;
    }
}
```

### 3.4 RAII与异常安全
```cpp
class FileHandler {
private:
    FILE* file;
    
public:
    FileHandler(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) {
            throw runtime_error("无法打开文件");
        }
    }
    
    ~FileHandler() {
        if (file) fclose(file);
    }
    
    // 禁止拷贝
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    void write(const string& data) {
        if (fprintf(file, "%s\n", data.c_str()) < 0) {
            throw runtime_error("写入文件失败");
        }
    }
};

// 异常安全的函数
void safeFunction() {
    FileHandler file("data.txt", "w");  // RAII
    
    file.write("开始处理");
    
    // 可能抛出异常的操作
    throw runtime_error("处理过程中出错");
    
    file.write("处理完成");  // 不会执行
    // 但file的析构函数会自动调用，确保文件关闭
}
```

## 第四章：智能指针（C++11+）

### 4.1 智能指针核心概念
- **智能指针**：自动管理动态内存的模板类
- **RAII模式**：资源获取即初始化
- **所有权语义**：明确指针的所有权关系

### 4.2 unique_ptr：独占所有权
```cpp
#include <memory>

class MyClass {
    string name;
public:
    MyClass(const string& n) : name(n) {
        cout << "构造: " << name << endl;
    }
    
    ~MyClass() {
        cout << "析构: " << name << endl;
    }
    
    void doSomething() {
        cout << "操作: " << name << endl;
    }
};

void uniquePtrDemo() {
    // 创建unique_ptr
    unique_ptr<MyClass> ptr1 = make_unique<MyClass>("对象1");
    ptr1->doSomething();
    
    // 转移所有权
    unique_ptr<MyClass> ptr2 = move(ptr1);  // ptr1变为nullptr
    
    if (!ptr1) {
        cout << "ptr1所有权已转移" << endl;
    }
    
    ptr2->doSomething();
    
    // 自动释放：ptr2离开作用域时自动销毁
}
```

### 4.3 shared_ptr：共享所有权
```cpp
void sharedPtrDemo() {
    // 创建shared_ptr
    shared_ptr<MyClass> ptr1 = make_shared<MyClass>("共享对象");
    cout << "引用计数: " << ptr1.use_count() << endl;  // 1
    
    {
        // 共享所有权
        shared_ptr<MyClass> ptr2 = ptr1;
        cout << "引用计数: " << ptr1.use_count() << endl;  // 2
        
        ptr2->doSomething();
        
        // ptr2离开作用域，引用计数减1
    }
    
    cout << "引用计数: " << ptr1.use_count() << endl;  // 1
    ptr1->doSomething();
    
    // 当ptr1离开作用域，引用计数为0时，对象自动销毁
}
```

### 4.4 weak_ptr：弱引用
```cpp
struct Node {
    string name;
    shared_ptr<Node> next;
    weak_ptr<Node> prev;  // 使用weak_ptr避免循环引用
    
    Node(const string& n) : name(n) {
        cout << "Node构造: " << name << endl;
    }
    
    ~Node() {
        cout << "Node析构: " << name << endl;
    }
};

void weakPtrDemo() {
    // 创建双向链表节点
    shared_ptr<Node> node1 = make_shared<Node>("节点1");
    shared_ptr<Node> node2 = make_shared<Node>("节点2");
    
    // 建立连接
    node1->next = node2;
    node2->prev = node1;  // weak_ptr不增加引用计数
    
    cout << "node1引用计数: " << node1.use_count() << endl;  // 1
    cout << "node2引用计数: " << node2.use_count() << endl;  // 2
    
    // 检查weak_ptr是否有效
    if (auto sharedPrev = node2->prev.lock()) {
        cout << "通过weak_ptr访问: " << sharedPrev->name << endl;
    }
    
    // 节点自动释放，不会出现循环引用导致的内存泄漏
}
```

### 4.5 智能指针与STL容器
```cpp
void smartPtrWithContainers() {
    // vector存储unique_ptr（需要移动语义）
    vector<unique_ptr<MyClass>> objects;
    
    objects.push_back(make_unique<MyClass>("容器对象1"));
    objects.push_back(make_unique<MyClass>("容器对象2"));
    
    cout << "容器中有 " << objects.size() << " 个对象" << endl;
    
    // 遍历和访问
    for (const auto& obj : objects) {
        obj->doSomething();
    }
    
    // 容器析构时，所有unique_ptr会自动释放管理的对象
}
```

### 4.6 智能指针易错点

#### 循环引用问题
```cpp
struct BadNode {
    shared_ptr<BadNode> next;
    shared_ptr<BadNode> prev;  // 错误：shared_ptr循环引用！
    
    ~BadNode() { cout << "Node析构" << endl; }
};

void circularReference() {
    auto node1 = make_shared<BadNode>();
    auto node2 = make_shared<BadNode>();
    
    node1->next = node2;
    node2->prev = node1;  // 循环引用，内存泄漏！
    
    // 对象永远不会被释放
}

// 修正：使用weak_ptr
struct GoodNode {
    shared_ptr<GoodNode> next;
    weak_ptr<GoodNode> prev;  // 正确：使用weak_ptr
    
    ~GoodNode() { cout << "Node正常析构" << endl; }
};
```

#### 错误的所有权共享
```cpp
class BadDesign {
    shared_ptr<BadDesign> self;  // 循环引用！
public:
    void setSelf() {
        self = shared_ptr<BadDesign>(this);  // 严重错误！
    }
};

// 修正：使用enable_shared_from_this
class GoodDesign : public enable_shared_from_this<GoodDesign> {
    // 正确实现
};
```

## 第五章：综合应用案例

### 5.1 图形界面框架设计
```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

// 抽象基类：UI组件
class UIComponent {
protected:
    string name;
    int x, y, width, height;
    
public:
    UIComponent(const string& n, int xPos, int yPos, int w, int h)
        : name(n), x(xPos), y(yPos), width(w), height(h) {}
    
    virtual ~UIComponent() = default;
    
    // 纯虚函数：绘制组件
    virtual void draw() const = 0;
    
    // 虚函数：处理事件
    virtual void handleEvent(const string& event) {
        cout << name << " 处理事件: " << event << endl;
    }
    
    string getName() const { return name; }
    virtual string getType() const = 0;
};

// 具体组件：按钮
class Button : public UIComponent {
private:
    string label;
    
public:
    Button(const string& n, int x, int y, int w, int h, const string& lbl)
        : UIComponent(n, x, y, w, h), label(lbl) {}
    
    void draw() const override {
        cout << "绘制按钮[" << name << "]: " << label 
             << " 位置(" << x << "," << y << ") 大小(" 
             << width << "x" << height << ")" << endl;
    }
    
    void handleEvent(const string& event) override {
        if (event == "click") {
            cout << "按钮 " << name << " 被点击!" << endl;
        } else {
            UIComponent::handleEvent(event);
        }
    }
    
    string getType() const override { return "Button"; }
};

// 具体组件：文本框
class TextBox : public UIComponent {
private:
    string text;
    
public:
    TextBox(const string& n, int x, int y, int w, int h, const string& txt = "")
        : UIComponent(n, x, y, w, h), text(txt) {}
    
    void draw() const override {
        cout << "绘制文本框[" << name << "]: \"" << text 
             << "\" 位置(" << x << "," << y << ")" << endl;
    }
    
    void handleEvent(const string& event) override {
        if (event.find("text:") == 0) {
            text = event.substr(5);
            cout << "文本框 " << name << " 内容更新为: " << text << endl;
        } else {
            UIComponent::handleEvent(event);
        }
    }
    
    string getType() const override { return "TextBox"; }
    string getText() const { return text; }
};

// 容器组件：面板
class Panel : public UIComponent {
private:
    vector<shared_ptr<UIComponent>> children;
    
public:
    Panel(const string& n, int x, int y, int w, int h)
        : UIComponent(n, x, y, w, h) {}
    
    void addComponent(shared_ptr<UIComponent> component) {
        children.push_back(component);
    }
    
    void draw() const override {
        cout << "绘制面板[" << name << "] 包含 " << children.size() 
             << " 个子组件:" << endl;
        
        for (const auto& child : children) {
            cout << "  ";
            child->draw();
        }
    }
    
    void handleEvent(const string& event) override {
        cout << "面板 " << name << " 分发事件: " << event << endl;
        for (const auto& child : children) {
            child->handleEvent(event);
        }
    }
    
    string getType() const override { return "Panel"; }
    
    template<typename T>
    shared_ptr<T> findComponent(const string& compName) {
        for (const auto& child : children) {
            if (child->getName() == compName) {
                return dynamic_pointer_cast<T>(child);
            }
        }
        return nullptr;
    }
};

// 窗口管理器
class WindowManager {
private:
    vector<shared_ptr<UIComponent>> windows;
    
public:
    void addWindow(shared_ptr<UIComponent> window) {
        windows.push_back(window);
    }
    
    void drawAll() {
        cout << "=== 绘制所有窗口 ===" << endl;
        for (const auto& window : windows) {
            window->draw();
            cout << endl;
        }
    }
    
    void sendEventToAll(const string& event) {
        cout << "=== 发送事件到所有窗口 ===" << endl;
        for (const auto& window : windows) {
            window->handleEvent(event);
        }
    }
    
    template<typename T>
    shared_ptr<T> findWindow(const string& windowName) {
        for (const auto& window : windows) {
            if (window->getName() == windowName) {
                return dynamic_pointer_cast<T>(window);
            }
        }
        return nullptr;
    }
};

// 使用示例
void uiFrameworkDemo() {
    WindowManager wm;
    
    // 创建登录窗口
    auto loginPanel = make_shared<Panel>("登录面板", 100, 100, 300, 200);
    
    auto usernameBox = make_shared<TextBox>("用户名", 120, 120, 200, 30, "admin");
    auto passwordBox = make_shared<TextBox>("密码", 120, 160, 200, 30);
    auto loginButton = make_shared<Button>("登录按钮", 120, 200, 80, 30, "登录");
    
    loginPanel->addComponent(usernameBox);
    loginPanel->addComponent(passwordBox);
    loginPanel->addComponent(loginButton);
    
    wm.addWindow(loginPanel);
    
    // 绘制界面
    wm.drawAll();
    
    // 模拟用户交互
    wm.sendEventToAll("text:newuser");
    wm.sendEventToAll("click");
    
    // 动态查找组件
    if (auto button = loginPanel->findComponent<Button>("登录按钮")) {
        button->handleEvent("click");
    }
}

int main() {
    uiFrameworkDemo();
    return 0;
}
```

## 第六章：学习总结

### 6.1 高级特性回顾
- **继承与多态**：建立类层次结构，实现运行时多态
- **模板编程**：编写通用代码，支持多种数据类型
- **命名空间**：组织代码，防止名称冲突
- **异常处理**：优雅处理运行时错误
- **智能指针**：自动内存管理，避免资源泄漏

### 6.2 设计模式应用
- **工厂模式**：通过工厂方法创建对象
- **策略模式**：通过模板实现不同的算法策略
- **观察者模式**：事件处理机制
- **组合模式**：UI框架中的容器-组件关系

### 6.3 最佳实践
1. **继承设计**：优先使用组合而非继承
2. **模板约束**：使用概念或静态断言约束模板参数
3. **异常安全**：确保代码在异常发生时保持正确状态
4. **资源管理**：使用RAII和智能指针自动管理资源

### 6.4 进阶学习方向
- **C++17/20新特性**：概念、协程、模块等
- **模板元编程**：编译时计算和代码生成
- **并发编程**：多线程和异步操作
- **性能优化**：内存布局、缓存友好设计

通过掌握这些高级核心特性，您已经具备了开发复杂C++应用程序的能力，可以开始学习更专业的领域如游戏开发、系统编程或高性能计算。