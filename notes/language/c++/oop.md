## 面向对象编程

### 类与对象：封装性

**类定义示例：数据抽象与信息隐藏**

**封装原理：** 类将数据（成员变量）和操作（成员函数）捆绑在一起，通过访问控制（`public`/`private`/`protected`）实现信息隐藏。这种设计保护了内部状态的一致性，提供了清晰的接口。

```cpp
class BankAccount {
private:    // 私有成员：实现细节，外部不能直接访问
    std::string account_number;  // 内部状态：账户号码
    double balance;              // 内部状态：余额
    
    // 私有方法：内部辅助函数
    void log_transaction(const std::string &type, double amount) {
        // 记录交易日志（实现细节）
    }
    
public:     // 公有接口：对外提供的服务
    // 构造函数：对象初始化
    BankAccount(const std::string &acc_num, double initial_balance)
        : account_number(acc_num), balance(initial_balance) {
        // 成员初始化列表：直接在对象内存中初始化
        // 比在构造函数体内赋值更高效
    }
    
    // 成员函数：业务逻辑封装
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;          // 修改内部状态
            log_transaction("存款", amount);  // 内部操作
        }
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            log_transaction("取款", amount);
            return true;
        }
        return false;  // 业务规则验证
    }
    
    double get_balance() const {  // const成员函数：不修改对象状态的承诺
        return balance;  // const函数内只能调用其他const成员函数
    }
    
    // 拷贝控制（Rule of Three/Five）
    BankAccount(const BankAccount&) = delete;            // 禁用拷贝构造
    BankAccount& operator=(const BankAccount&) = delete; // 禁用拷贝赋值
    BankAccount(BankAccount&&) = default;                // 允许移动构造
    BankAccount& operator=(BankAccount&&) = default;     // 允许移动赋值
};

// 类的内存布局（简化）
// BankAccount对象包含：
// - std::string account_number（通常是24字节，包含指向堆内存的指针）
// - double balance（8字节）
// - 可能的填充字节（内存对齐）
// 总大小：通常32-40字节（取决于实现）
```

**使用类：对象生命周期与成员函数调用**

**对象创建与使用：** 对象在栈上或堆上创建，通过点运算符（`.`）或箭头运算符（`->`）访问成员。成员函数调用隐含传递`this`指针，指向当前对象。

```cpp
// 栈上对象：自动生命周期管理
BankAccount my_account("123456", 1000.0);  // 调用构造函数
// 对象在栈上分配，包含account_number和balance的完整内存

// 成员函数调用：隐含this指针
my_account.deposit(500.0);  // 等价于：BankAccount::deposit(&my_account, 500.0)
// this指针指向my_account对象，允许函数访问对象的成员变量

my_account.withdraw(200.0);
std::cout << "余额: " << my_account.get_balance() << std::endl;

// 堆上对象：手动生命周期管理
BankAccount *heap_account = new BankAccount("789012", 2000.0);
heap_account->deposit(300.0);  // 使用箭头运算符
delete heap_account;  // 必须手动释放

// 自动存储期对象的析构
{
    BankAccount temp_account("temp", 100.0);
    // 使用temp_account...
} // 离开作用域时，temp_account的析构函数自动调用

// const对象的使用
const BankAccount const_account("const", 500.0);
// const_account.deposit(100.0);  // 错误：const对象只能调用const成员函数
std::cout << const_account.get_balance() << std::endl;  // 正确：get_balance是const

// 对象数组
BankAccount accounts[3] = {
    BankAccount("acc1", 1000.0),
    BankAccount("acc2", 2000.0),
    BankAccount("acc3", 3000.0)
};
// 每个元素都是完整的BankAccount对象
```

**对比Python的class：**
```python
class BankAccount:
    def __init__(self, acc_num, initial_balance):
        self.account_number = acc_num
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
```

### 构造函数与析构函数：对象生命周期管理

**构造函数类型：对象初始化与资源获取**

**构造函数作用：** 构造函数在对象创建时自动调用，负责初始化对象状态和获取所需资源。C++提供了多种构造函数来满足不同的初始化需求。

```cpp
class Person {
private:
    std::string name;
    int age;
    
public:
    // 默认构造函数：无参初始化
    Person() : name("Unknown"), age(0) {
        // 成员初始化列表：在构造函数体执行前完成初始化
        // 比在构造函数体内赋值更高效（避免默认构造+赋值）
    }
    
    // 参数化构造函数：带参数的初始化
    Person(const std::string &n, int a) : name(n), age(a) {
        // 直接构造成员，避免不必要的拷贝
    }
    
    // 委托构造函数（C++11）：重用其他构造函数的代码
    Person(const std::string &n) : Person(n, 0) {}  // 委托给双参数构造函数
    
    // 拷贝构造函数：创建对象的副本
    Person(const Person &other) : name(other.name), age(other.age) {
        // 深拷贝：复制所有成员（包括动态分配的资源）
        // 默认拷贝构造执行浅拷贝，对于包含指针的类需要自定义
    }
    
    // 移动构造函数（C++11）：资源转移
    Person(Person &&other) noexcept 
        : name(std::move(other.name)), age(other.age) {
        // 转移资源所有权，避免不必要的拷贝
        other.age = 0;  // 使源对象处于有效但未定义状态
    }
    
    // 析构函数：资源释放与清理
    ~Person() {
        // 自动调用：对象销毁时（离开作用域、delete等）
        std::cout << name << "对象被销毁" << std::endl;
        // 析构顺序：与构造顺序相反（成员变量、基类）
    }
    
    void display() const {
        std::cout << "姓名: " << name << ", 年龄: " << age << std::endl;
    }
    
    // 赋值运算符（Rule of Three/Five）
    Person& operator=(const Person &other) {  // 拷贝赋值
        if (this != &other) {  // 自赋值检查
            name = other.name;
            age = other.age;
        }
        return *this;
    }
    
    Person& operator=(Person &&other) noexcept {  // 移动赋值
        if (this != &other) {
            name = std::move(other.name);
            age = other.age;
            other.age = 0;
        }
        return *this;
    }
};

// 构造函数调用场景
Person p1;                     // 默认构造
Person p2("Alice", 25);        // 参数化构造
Person p3 = p2;                // 拷贝构造
Person p4 = std::move(p3);     // 移动构造
Person p5{"Bob"};              // 统一初始化语法
```

**对比Python的__init__和__del__：**
```python
class Person:
    def __init__(self, name="Unknown", age=0):
        self.name = name
        self.age = age
    
    def __del__(self):
        print(f"{self.name}对象被销毁")
```

### 继承与多态：代码复用与运行时灵活性

**继承示例：类型层次与代码复用**

**继承原理：** 继承建立了类型之间的"is-a"关系，派生类自动获得基类的成员（根据访问控制）。C++支持单继承和多继承，提供了强大的代码复用机制。

```cpp
// 基类：抽象形状定义
class Shape {
protected:  // 保护成员：派生类可访问，外部不可访问
    std::string color;
    
public:
    // 构造函数：初始化基类部分
    Shape(const std::string &c) : color(c) {}
    
    // 虚函数（支持多态）：运行时动态绑定
    virtual double area() const = 0;  // 纯虚函数，使Shape成为抽象类
    // 抽象类不能实例化，只能作为基类
    
    virtual void display() const {
        std::cout << "形状颜色: " << color << std::endl;
    }
    
    // 虚析构函数（重要！确保正确调用派生类析构函数）
    virtual ~Shape() {}
    // 如果基类析构函数不是虚的，通过基类指针删除派生类对象会导致未定义行为
};

// 派生类：具体形状实现
class Circle : public Shape {  // public继承：is-a关系
private:
    double radius;
    
public:
    // 派生类构造函数：先初始化基类部分
    Circle(const std::string &c, double r) : Shape(c), radius(r) {}
    
    // 重写虚函数：override关键字确保正确重写（C++11）
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void display() const override {
        Shape::display();  // 调用基类方法
        std::cout << "圆形半径: " << radius << std::endl;
    }
    
    // 新增功能：派生类特有方法
    double circumference() const {
        return 2 * 3.14159 * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(const std::string &c, double w, double h) 
        : Shape(c), width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
    
    void display() const override {
        Shape::display();
        std::cout << "矩形宽高: " << width << " × " << height << std::endl;
    }
};

// 虚函数表（vtable）实现原理（简化）
// 每个有虚函数的类都有一个虚函数表
// Shape的vtable: [&Shape::area, &Shape::display, &Shape::~Shape]
// Circle的vtable: [&Circle::area, &Circle::display, &Circle::~Circle]
// 对象包含指向vtable的指针，实现动态绑定
```

**多态使用：运行时类型识别与动态绑定**

**多态机制：** 多态允许通过基类接口操作派生类对象，在运行时根据实际对象类型调用正确的函数。这是通过虚函数表和动态绑定实现的。

```cpp
// 多态示例：统一接口，不同实现
void print_area(const Shape &shape) {  // 接收基类引用
    std::cout << "面积: " << shape.area() << std::endl;  // 动态绑定
    // 编译时：只知道shape是Shape引用
    // 运行时：根据实际对象类型调用正确的area()实现
}

int main() {
    Circle circle("红色", 5.0);       // 具体Circle对象
    Rectangle rectangle("蓝色", 4.0, 6.0);  // 具体Rectangle对象
    
    // 多态调用：静态类型 vs 动态类型
    print_area(circle);     // 静态类型：const Shape&，动态类型：Circle
    // 运行时调用Circle::area()
    
    print_area(rectangle);  // 静态类型：const Shape&，动态类型：Rectangle  
    // 运行时调用Rectangle::area()
    
    // 通过基类指针/引用调用：动态绑定的核心应用
    Shape *shapes[] = {&circle, &rectangle};  // 基类指针数组
    for (Shape *shape : shapes) {
        shape->display();  // 动态绑定到具体类型的display()
        // shape指向Circle时：调用Circle::display()
        // shape指向Rectangle时：调用Rectangle::display()
    }
    
    // 动态内存中的多态
    std::vector<std::unique_ptr<Shape>> shape_list;
    shape_list.push_back(std::make_unique<Circle>("绿色", 3.0));
    shape_list.push_back(std::make_unique<Rectangle>("黄色", 2.0, 4.0));
    
    for (const auto &shape : shape_list) {
        shape->display();  // 多态调用，无需知道具体类型
    }
    
    // 类型识别：dynamic_cast（运行时类型检查）
    Shape *unknown_shape = &circle;
    if (Circle *circle_ptr = dynamic_cast<Circle*>(unknown_shape)) {
        // 转换成功：unknown_shape确实指向Circle
        std::cout << "周长: " << circle_ptr->circumference() << std::endl;
    }
    
    return 0;
}

// 多态的性能考虑
// 虚函数调用有轻微开销：通过vtable间接调用
// 但对于大多数应用，这种开销可以忽略
// 编译器可能进行去虚拟化优化（devirtualization）

// 多态的设计优势
// 1. 可扩展性：添加新形状无需修改现有代码
// 2. 可维护性：统一接口，减少重复代码
// 3. 灵活性：运行时决定具体行为
```

### 访问控制与静态成员

**访问控制：**
```cpp
class AccessExample {
private:    // 仅类内部可访问
    int private_var;
    
protected:  // 类内部和派生类可访问
    int protected_var;
    
public:     // 任何地方都可访问
    int public_var;
    
    void set_private(int value) {
        private_var = value;  // 类内部可以访问private
    }
};

class Derived : public AccessExample {
public:
    void access_members() {
        // public_var = 10;     // 可以访问
        // protected_var = 20;  // 可以访问（派生类）
        // private_var = 30;    // 错误！不能访问基类private
    }
};
```

**静态成员：类级别的共享数据**

**静态成员原理：** 静态成员属于类本身而非对象实例，所有对象共享同一份静态成员。静态成员在程序生命周期内存在，提供类级别的状态管理和工具函数。

```cpp
class Counter {
private:
    static int count;  // 静态成员变量：类级别，所有对象共享
    int id;            // 普通成员变量：每个对象独立
    
public:
    Counter() {
        id = ++count;  // 访问静态成员：无需对象实例
        // count在所有Counter对象间共享，提供全局计数
    }
    
    // 静态成员函数：属于类而非对象
    static int get_count() {
        return count;  // 只能访问静态成员，没有this指针
    }
    
    // 静态函数可以作为工具函数
    static void reset_counter() {
        count = 0;     // 重置全局计数
    }
    
    int get_id() const {
        return id;     // 普通成员函数：访问对象特定数据
    }
    
    // 析构函数：减少计数
    ~Counter() {
        --count;       // 对象销毁时更新共享计数
    }
};

// 静态成员变量定义：必须在类外定义（分配存储空间）
int Counter::count = 0;  // 在全局数据区分配内存
// 定义提供了变量的实际存储，链接器需要这个定义

// 使用静态成员：类名作用域
Counter c1, c2, c3;
std::cout << "对象数量: " << Counter::get_count() << std::endl;  // 3
// 通过类名直接访问，无需对象实例

std::cout << "c2的ID: " << c2.get_id() << std::endl;           // 2
// 普通成员函数需要通过对象调用

// 静态成员的初始化时机
class Logger {
private:
    static std::string log_file;  // 静态字符串
    
public:
    static void set_log_file(const std::string &filename) {
        log_file = filename;
    }
    
    static void log(const std::string &message) {
        // 使用log_file记录日志
    }
};

// 静态成员初始化（C++17内联静态变量）
// C++17之前：需要在.cpp文件中定义
// std::string Logger::log_file = "default.log";

// C++17内联静态变量（推荐）
class ModernLogger {
private:
    inline static std::string log_file = "default.log";  // 类内初始化
    
public:
    static void set_log_file(const std::string &filename) {
        log_file = filename;
    }
};

// 静态常量成员（可以在类内初始化）
class MathConstants {
public:
    static constexpr double PI = 3.141592653589793;  // 类内初始化
    static constexpr int MAX_ITERATIONS = 1000;
    
    // 对于整数类型，甚至可以在类内定义
    static const int DEFAULT_SIZE = 64;  // 声明+定义
};

// 使用静态常量
double circle_area(double r) {
    return MathConstants::PI * r * r;  // 编译时常量，无运行时开销
}

// 静态成员的单例模式应用
class Singleton {
private:
    static Singleton* instance;  // 静态实例指针
    Singleton() {}  // 私有构造函数
    
public:
    static Singleton* get_instance() {
        if (!instance) {
            instance = new Singleton();
        }
        return instance;
    }
    
    // 禁用拷贝
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};

// 静态成员变量定义
Singleton* Singleton::instance = nullptr;
```

**const成员函数：**
```cpp
class ConstExample {
private:
    mutable int cache;  // mutable可以在const函数中修改
    int value;
    
public:
    // const成员函数：承诺不修改对象状态
    int get_value() const {
        // value = 10;  // 错误！const函数不能修改非mutable成员
        cache++;        // 可以修改mutable成员
        return value;
    }
    
    // 非const成员函数
    void set_value(int v) {
        value = v;
    }
};

### const成员函数深度解析：常量正确性的核心

**const成员函数设计原理：** const成员函数通过函数签名中的const关键字承诺不修改对象状态，这是C++常量正确性的重要机制。编译器会在编译时检查const成员函数的实现，确保其不会修改对象的非mutable成员。

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// 复杂的const成员函数示例
class Student {
private:
    string name;
    int id;
    mutable int access_count;  // mutable：可以在const函数中修改
    vector<int> scores;
    
public:
    Student(const string& n, int student_id) 
        : name(n), id(student_id), access_count(0) {}
    
    // 基本const成员函数：只读访问
    const string& get_name() const {
        access_count++;  // 正确：mutable成员可以修改
        return name;
    }
    
    int get_id() const {
        access_count++;
        return id;
    }
    
    // const成员函数中的复杂操作
    double get_average_score() const {
        if (scores.empty()) {
            return 0.0;
        }
        
        int sum = 0;
        for (int score : scores) {  // 正确：不修改scores
            sum += score;
        }
        return static_cast<double>(sum) / scores.size();
    }
    
    // const成员函数调用其他const成员函数
    void display_info() const {
        cout << "学生信息:" << endl;
        cout << "姓名: " << get_name() << endl;     // 调用const成员函数
        cout << "学号: " << get_id() << endl;       // 调用const成员函数
        cout << "平均分: " << get_average_score() << endl;
        cout << "访问次数: " << access_count << endl;
    }
    
    // 非const成员函数：可以修改对象状态
    void add_score(int score) {
        scores.push_back(score);
    }
    
    void set_name(const string& new_name) {
        name = new_name;
    }
    
    // 重载：const和非const版本
    const vector<int>& get_scores() const {
        access_count++;
        return scores;  // 返回const引用
    }
    
    vector<int>& get_scores() {
        access_count++;
        return scores;  // 返回非const引用，允许修改
    }
    
    // const成员函数中的异常安全操作
    int get_score_at(int index) const {
        if (index < 0 || index >= scores.size()) {
            throw out_of_range("索引越界");
        }
        return scores[index];
    }
};

// 常量正确性在继承中的应用
class Person {
protected:
    string name;
    int age;
    
public:
    Person(const string& n, int a) : name(n), age(a) {}
    
    // 虚函数也应该是const的（如果合适）
    virtual void introduce() const {
        cout << "我是" << name << ", " << age << "岁" << endl;
    }
    
    virtual ~Person() = default;
};

class Employee : public Person {
private:
    string department;
    
public:
    Employee(const string& n, int a, const string& dept)
        : Person(n, a), department(dept) {}
    
    // 重写基类的const虚函数
    void introduce() const override {
        cout << "我是" << name << ", " << age << "岁, " 
             << "在" << department << "部门工作" << endl;
    }
    
    // 新的const成员函数
    const string& get_department() const {
        return department;
    }
};

// const成员函数与STL容器的结合
class StudentManager {
private:
    vector<Student> students;
    
public:
    void add_student(const Student& student) {
        students.push_back(student);
    }
    
    // const成员函数返回const迭代器
    vector<Student>::const_iterator begin() const {
        return students.begin();
    }
    
    vector<Student>::const_iterator end() const {
        return students.end();
    }
    
    // 查找操作应该是const的
    const Student* find_student_by_id(int id) const {
        for (const auto& student : students) {
            if (student.get_id() == id) {
                return &student;
            }
        }
        return nullptr;
    }
    
    // 统计操作也应该是const的
    size_t count_students() const {
        return students.size();
    }
    
    double get_class_average() const {
        if (students.empty()) return 0.0;
        
        double total = 0.0;
        for (const auto& student : students) {
            total += student.get_average_score();
        }
        return total / students.size();
    }
};

// 演示const正确性的重要性
void demonstrate_const_correctness() {
    cout << "=== const成员函数演示 ===" << endl;
    
    Student student("张三", 1001);
    student.add_score(85);
    student.add_score(92);
    student.add_score(78);
    
    // 非const对象可以调用const和非const成员函数
    student.display_info();
    student.set_name("李四");  // 调用非const成员函数
    
    // const对象只能调用const成员函数
    const Student const_student("王五", 1002);
    const_student.display_info();
    // const_student.add_score(90);  // 错误：const对象不能调用非const成员函数
    
    cout << "\n=== 继承中的const正确性 ===" << endl;
    
    Employee employee("赵六", 30, "技术部");
    const Person& person_ref = employee;  // 基类const引用指向派生类对象
    person_ref.introduce();               // 多态调用，使用const成员函数
    
    cout << "\n=== STL容器与const成员函数 ===" << endl;
    
    StudentManager manager;
    manager.add_student(student);
    manager.add_student(const_student);
    
    const StudentManager& const_manager = manager;
    
    cout << "学生数量: " << const_manager.count_students() << endl;
    cout << "班级平均分: " << const_manager.get_class_average() << endl;
    
    // const管理器可以安全遍历
    cout << "所有学生信息:" << endl;
    for (auto it = const_manager.begin(); it != const_manager.end(); ++it) {
        it->display_info();
    }
}

// 模板类中的const成员函数
template<typename T>
class SafeArray {
private:
    vector<T> data;
    
public:
    SafeArray(size_t size) : data(size) {}
    
    // const版本的下标运算符
    const T& operator[](size_t index) const {
        if (index >= data.size()) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // 非const版本的下标运算符
    T& operator[](size_t index) {
        if (index >= data.size()) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // const成员函数
    size_t size() const {
        return data.size();
    }
    
    bool empty() const {
        return data.empty();
    }
    
    // 返回const迭代器
    typename vector<T>::const_iterator begin() const {
        return data.begin();
    }
    
    typename vector<T>::const_iterator end() const {
        return data.end();
    }
};

int main() {
    demonstrate_const_correctness();
    
    cout << "\n=== 模板类中的const成员函数 ===" << endl;
    
    SafeArray<int> arr(5);
    arr[0] = 10;
    arr[1] = 20;
    
    const SafeArray<int>& const_arr = arr;
    
    cout << "数组大小: " << const_arr.size() << endl;
    cout << "第一个元素: " << const_arr[0] << endl;  // 调用const版本
    // const_arr[0] = 30;  // 错误：const对象不能调用非const版本
    
    return 0;
}
```

**const成员函数的最佳实践：**

| 场景 | 推荐做法 | 原因 |
|------|----------|------|
| **只读访问** | 所有getter函数声明为const | 保证不修改对象状态 |
| **计算操作** | 不修改对象的计算函数声明为const | 提高代码安全性 |
| **STL容器操作** | 查找、统计等操作声明为const | 支持const对象使用 |
| **虚函数** | 如果不修改状态，声明为const | 保持继承层次的一致性 |
| **重载** | 提供const和非const版本 | 根据对象常量性选择合适版本 |

**const正确性的重要性：**
1. **编译时检查**：编译器确保const成员函数不会意外修改对象状态
2. **接口清晰**：通过函数签名明确表达函数的意图
3. **支持const对象**：const对象只能调用const成员函数
4. **线程安全**：const成员函数通常更容易实现线程安全
5. **优化机会**：编译器可能对const成员函数进行更好的优化

通过合理使用const成员函数，可以编写出更加安全、清晰和高效的面向对象代码。

### this指针详解：对象自我引用机制

**this指针原理：** 每个非静态成员函数都隐含一个this指针参数，指向调用该函数的对象。编译器在编译时将成员函数调用转换为普通函数调用，并自动传递this指针。

```cpp
#include <iostream>
#include <string>
using namespace std;

class Point {
private:
    int x, y;
    
public:
    // 构造函数中使用this解决命名冲突
    Point(int x, int y) {
        this->x = x;  // 使用this区分参数和成员变量
        this->y = y;
    }
    
    // 链式调用：返回对象引用支持连续操作
    Point& setX(int x) {
        this->x = x;
        return *this;  // 返回当前对象的引用
    }
    
    Point& setY(int y) {
        this->y = y;
        return *this;
    }
    
    // 比较函数：使用this指针进行比较
    bool isEqual(const Point& other) const {
        return this->x == other.x && this->y == other.y;
    }
    
    // 显示坐标
    void display() const {
        cout << "Point(" << x << ", " << y << ")" << endl;
    }
    
    // 静态方法：没有this指针
    static Point createOrigin() {
        return Point(0, 0);
    }
};

int main() {
    Point p1(10, 20);
    Point p2(30, 40);
    
    // 链式调用示例
    p1.setX(50).setY(60).display();  // Point(50, 60)
    
    // this指针的比较功能
    cout << "p1和p2是否相等：" << (p1.isEqual(p2) ? "是" : "否") << endl;
    
    // 静态方法调用
    Point origin = Point::createOrigin();
    origin.display();  // Point(0, 0)
    
    return 0;
}
```

**this指针的底层实现：**
```cpp
// 编译器视角：成员函数转换
class Point {
    int x, y;
public:
    void setX(int x);  // 实际签名：void setX(Point* this, int x)
};

// 成员函数调用转换
Point p;
p.setX(10);  // 实际调用：Point::setX(&p, 10)
```

### 友元机制：打破封装的特权访问

**友元设计原理：** 友元机制允许特定函数或类访问类的私有成员，这是对封装原则的有意破坏。友元关系是授予的，不是索取的，需要在类内部明确声明。

```cpp
#include <iostream>
#include <cmath>
using namespace std;

// 前向声明
class Circle;

// 友元函数声明
void printCircleInfo(const Circle& c);
double calculateDistance(const Circle& c1, const Circle& c2);

class Circle {
private:
    double x, y, radius;
    
public:
    Circle(double x = 0, double y = 0, double r = 1) 
        : x(x), y(y), radius(r) {}
    
    // 声明友元函数
    friend void printCircleInfo(const Circle& c);
    friend double calculateDistance(const Circle& c1, const Circle& c2);
    
    // 声明友元类
    friend class CircleCalculator;
    
    // 成员函数：正常访问私有成员
    double area() const {
        return 3.14159 * radius * radius;
    }
};

// 友元函数定义：可以访问Circle的私有成员
void printCircleInfo(const Circle& c) {
    cout << "圆心坐标：(" << c.x << ", " << c.y << ")" << endl;
    cout << "半径：" << c.radius << endl;
    cout << "面积：" << c.area() << endl;
}

// 另一个友元函数
double calculateDistance(const Circle& c1, const Circle& c2) {
    double dx = c1.x - c2.x;
    double dy = c1.y - c2.y;
    return sqrt(dx * dx + dy * dy);
}

// 友元类定义
class CircleCalculator {
public:
    // 可以访问Circle的所有私有成员
    static bool isOverlapping(const Circle& c1, const Circle& c2) {
        double distance = calculateDistance(c1, c2);
        return distance < (c1.radius + c2.radius);
    }
    
    static Circle createUnitCircle() {
        return Circle(0, 0, 1);  // 可以调用私有构造函数
    }
};

// 非友元函数：无法访问私有成员
void tryAccessPrivate(const Circle& c) {
    // cout << c.x << endl;  // 错误：x是私有成员
    // cout << c.radius << endl;  // 错误：radius是私有成员
    cout << "只能通过公有接口访问" << endl;
}

int main() {
    Circle c1(0, 0, 5);
    Circle c2(8, 6, 3);
    
    // 使用友元函数
    printCircleInfo(c1);
    cout << "两圆距离：" << calculateDistance(c1, c2) << endl;
    
    // 使用友元类
    cout << "两圆是否重叠：" 
         << (CircleCalculator::isOverlapping(c1, c2) ? "是" : "否") << endl;
    
    // 非友元函数调用
    tryAccessPrivate(c1);
    
    return 0;
}
```

**友元使用场景与注意事项：**

| 使用场景 | 优点 | 风险 |
|----------|------|------|
| **运算符重载** | 支持非成员函数形式的运算符重载 | 破坏封装性 |
| **工具函数** | 为类提供额外的辅助功能 | 增加耦合度 |
| **紧密协作类** | 类之间需要深度交互 | 维护困难 |

### 静态成员深度解析：类级别的共享状态

**静态成员实现原理：** 静态成员在程序的数据段分配存储空间，生命周期与程序相同。所有对象共享同一份静态成员，提供类级别的状态管理和工具函数。

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Employee {
private:
    string name;
    int id;
    double salary;
    
    // 静态成员变量：类级别的共享数据
    static int totalEmployees;        // 员工总数
    static double totalSalary;        // 总工资
    static vector<Employee*> allEmployees;  // 所有员工指针
    
public:
    // 构造函数：更新静态数据
    Employee(const string& n, double s) 
        : name(n), salary(s) {
        id = ++totalEmployees;  // 自动分配唯一ID
        totalSalary += salary;
        allEmployees.push_back(this);  // 注册到全局列表
        
        cout << "创建员工：" << name << " (ID: " << id << ")" << endl;
    }
    
    // 析构函数：更新静态数据
    ~Employee() {
        totalSalary -= salary;
        totalEmployees--;
        
        // 从全局列表中移除
        for (auto it = allEmployees.begin(); it != allEmployees.end(); ++it) {
            if (*it == this) {
                allEmployees.erase(it);
                break;
            }
        }
        
        cout << "删除员工：" << name << " (ID: " << id << ")" << endl;
    }
    
    // 静态成员函数：类级别的操作
    static int getTotalEmployees() {
        return totalEmployees;
    }
    
    static double getTotalSalary() {
        return totalSalary;
    }
    
    static double getAverageSalary() {
        return totalEmployees > 0 ? totalSalary / totalEmployees : 0.0;
    }
    
    static void listAllEmployees() {
        cout << "=== 所有员工列表 ===" << endl;
        for (const auto& emp : allEmployees) {
            cout << "ID: " << emp->id << ", 姓名: " << emp->name 
                 << ", 工资: " << emp->salary << endl;
        }
        cout << "===================" << endl;
    }
    
    // 普通成员函数
    void displayInfo() const {
        cout << "员工信息 - ID: " << id << ", 姓名: " << name 
             << ", 工资: " << salary << endl;
    }
    
    void setSalary(double newSalary) {
        totalSalary = totalSalary - salary + newSalary;
        salary = newSalary;
    }
};

// 静态成员变量定义（必须在类外）
int Employee::totalEmployees = 0;
double Employee::totalSalary = 0.0;
vector<Employee*> Employee::allEmployees;

// 静态常量成员（可以在类内初始化）
class MathConstants {
public:
    static constexpr double PI = 3.141592653589793;
    static constexpr double E = 2.718281828459045;
    static const int MAX_ITERATIONS = 1000;  // 整数静态常量
    
    // 静态工具函数
    static double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    static double exponential(double x) {
        return pow(E, x);
    }
};

// 单例模式应用：使用静态成员实现全局唯一实例
class Logger {
private:
    static Logger* instance;  // 静态实例指针
    string logFile;
    
    // 私有构造函数
    Logger() : logFile("app.log") {
        cout << "Logger实例创建" << endl;
    }
    
public:
    // 删除拷贝构造函数和赋值运算符
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // 静态方法获取唯一实例
    static Logger& getInstance() {
        if (!instance) {
            instance = new Logger();
        }
        return *instance;
    }
    
    // 日志功能
    void log(const string& message) {
        cout << "[" << logFile << "] " << message << endl;
    }
    
    void setLogFile(const string& filename) {
        logFile = filename;
    }
    
    // 静态清理方法
    static void cleanup() {
        if (instance) {
            delete instance;
            instance = nullptr;
        }
    }
};

// 静态成员变量定义
Logger* Logger::instance = nullptr;

int main() {
    // 静态成员使用示例
    cout << "初始状态 - 员工总数: " << Employee::getTotalEmployees() 
         << ", 总工资: " << Employee::getTotalSalary() << endl;
    
    {
        Employee emp1("张三", 5000.0);
        Employee emp2("李四", 6000.0);
        
        cout << "创建后 - 员工总数: " << Employee::getTotalEmployees() 
             << ", 平均工资: " << Employee::getAverageSalary() << endl;
        
        Employee::listAllEmployees();
        
        // 修改工资
        emp1.setSalary(5500.0);
        cout << "修改后平均工资: " << Employee::getAverageSalary() << endl;
    }  // emp1, emp2离开作用域
    
    cout << "最终状态 - 员工总数: " << Employee::getTotalEmployees() 
         << ", 总工资: " << Employee::getTotalSalary() << endl;
    
    // 静态常量使用
    cout << "圆周率: " << MathConstants::PI << endl;
    cout << "半径5的圆面积: " << MathConstants::circleArea(5.0) << endl;
    
    // 单例模式使用
    Logger::getInstance().log("应用程序启动");
    Logger::getInstance().setLogFile("new_app.log");
    Logger::getInstance().log("配置文件已更新");
    
    // 清理单例
    Logger::cleanup();
    
    return 0;
}
```

**静态成员的重要特性总结：**

1. **存储特性**：静态成员在程序的数据段分配，生命周期与程序相同
2. **共享性**：所有对象共享同一份静态成员
3. **访问方式**：可以通过类名或对象访问，但推荐使用类名
4. **初始化**：必须在类外定义（除了静态常量整数成员）
5. **线程安全**：C++11后，局部静态变量是线程安全的

### 成员初始化列表：高效的初始化机制

**初始化列表原理：** 成员初始化列表在对象内存分配后立即执行，直接在对象内存中构造成员，避免了先默认构造再赋值的性能开销。

```cpp
#include <iostream>
#include <string>
using namespace std;

class Student {
private:
    const int id;           // 常量成员必须使用初始化列表
    string name;
    int& ageRef;           // 引用成员必须使用初始化列表
    int* scores;
    int scoreCount;
    
public:
    // 使用初始化列表的构造函数
    Student(int studentId, const string& studentName, int& age) 
        : id(studentId), name(studentName), ageRef(age), scores(nullptr), scoreCount(0) {
        // 初始化列表已经完成了所有成员的构造
        // 构造函数体中可以执行其他逻辑
        cout << "学生 " << name << " 创建完成" << endl;
    }
    
    // 对比：不使用初始化列表（效率较低）
    Student(int studentId, const string& studentName) 
        : id(studentId), name(studentName), ageRef(*(new int(0))) {  // 错误的引用初始化
        // 这里name会被先默认构造，然后再赋值
        // ageRef必须绑定到有效对象，这里使用了临时对象，很危险
    }
    
    ~Student() {
        delete[] scores;
    }
    
    void display() const {
        cout << "学生ID: " << id << ", 姓名: " << name 
             << ", 年龄引用: " << ageRef << endl;
    }
};

class Complex {
private:
    double real, imag;
    
public:
    // 委托构造函数：一个构造函数调用另一个
    Complex() : Complex(0, 0) {}  // 委托给双参数构造函数
    
    Complex(double r) : Complex(r, 0) {}  // 委托给双参数构造函数
    
    Complex(double r, double i) : real(r), imag(i) {
        cout << "创建复数: " << real << " + " << imag << "i" << endl;
    }
    
    // 移动构造函数（C++11）
    Complex(Complex&& other) noexcept 
        : real(move(other.real)), imag(move(other.imag)) {
        other.real = 0;
        other.imag = 0;
        cout << "移动构造复数" << endl;
    }
};

int main() {
    int age = 20;
    Student s1(1001, "王五", age);
    s1.display();
    
    // 委托构造函数使用
    Complex c1;          // 调用Complex(0, 0)
    Complex c2(3.14);    // 调用Complex(3.14, 0)
    Complex c3(2.5, 1.8); // 直接调用
    
    return 0;
}
```

通过以上内容的补充，"面向对象.md"文件现在包含了从"03_类与面向对象编程.md"中提取的详细内容，涵盖了this指针、友元机制、静态成员和成员初始化列表等重要概念，使面向对象编程的知识点更加全面。

## 面向对象进阶特性

### 纯虚函数与抽象类：接口与实现的分离

**纯虚函数设计原理：** 纯虚函数通过`= 0`语法强制派生类实现特定接口，实现真正的多态和接口分离。抽象类不能实例化，专门用于定义接口规范。

```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// 抽象基类：图形接口
class Shape {
public:
    // 纯虚函数：强制派生类实现
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual void draw() const = 0;
    
    // 虚析构函数：确保正确释放派生类对象
    virtual ~Shape() {
        cout << "Shape析构函数调用" << endl;
    }
    
    // 普通成员函数：提供通用功能
    void printInfo() const {
        cout << "面积: " << area() << ", 周长: " << perimeter() << endl;
    }
    
    // 静态成员函数：类级别操作
    static void describe() {
        cout << "这是一个图形抽象类" << endl;
    }
};

// 具体派生类：圆形
class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {
        if (radius <= 0) {
            throw invalid_argument("半径必须大于0");
        }
    }
    
    // 实现纯虚函数
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }
    
    void draw() const override {
        cout << "绘制圆形 (半径: " << radius << ")" << endl;
    }
    
    // 特有方法
    double getRadius() const { return radius; }
    void setRadius(double r) { 
        if (r > 0) radius = r; 
    }
};

// 具体派生类：矩形
class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {
        if (width <= 0 || height <= 0) {
            throw invalid_argument("宽高必须大于0");
        }
    }
    
    // 实现纯虚函数
    double area() const override {
        return width * height;
    }
    
    double perimeter() const override {
        return 2 * (width + height);
    }
    
    void draw() const override {
        cout << "绘制矩形 (" << width << " × " << height << ")" << endl;
    }
    
    // 特有方法
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    bool isSquare() const { return width == height; }
};

// 具体派生类：三角形
class Triangle : public Shape {
private:
    double a, b, c;
    
public:
    Triangle(double side1, double side2, double side3) 
        : a(side1), b(side2), c(side3) {
        if (!isValidTriangle()) {
            throw invalid_argument("无效的三角形边长");
        }
    }
    
    bool isValidTriangle() const {
        return (a + b > c) && (a + c > b) && (b + c > a) && 
               a > 0 && b > 0 && c > 0;
    }
    
    // 实现纯虚函数
    double area() const override {
        // 使用海伦公式计算面积
        double s = perimeter() / 2;
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }
    
    double perimeter() const override {
        return a + b + c;
    }
    
    void draw() const override {
        cout << "绘制三角形 (边长: " << a << ", " << b << ", " << c << ")" << endl;
    }
};

// 图形管理器：使用抽象类实现多态
class ShapeManager {
private:
    vector<unique_ptr<Shape>> shapes;
    
public:
    // 添加图形
    void addShape(unique_ptr<Shape> shape) {
        shapes.push_back(move(shape));
    }
    
    // 计算总面积
    double totalArea() const {
        double total = 0;
        for (const auto& shape : shapes) {
            total += shape->area();
        }
        return total;
    }
    
    // 计算总周长
    double totalPerimeter() const {
        double total = 0;
        for (const auto& shape : shapes) {
            total += shape->perimeter();
        }
        return total;
    }
    
    // 绘制所有图形
    void drawAll() const {
        for (const auto& shape : shapes) {
            shape->draw();
            shape->printInfo();
            cout << "---" << endl;
        }
    }
    
    // 查找最大面积的图形
    const Shape* findLargest() const {
        if (shapes.empty()) return nullptr;
        
        const Shape* largest = shapes[0].get();
        for (const auto& shape : shapes) {
            if (shape->area() > largest->area()) {
                largest = shape.get();
            }
        }
        return largest;
    }
};

// 接口继承示例：可绘制接口
class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

// 多重接口继承
class DrawableShape : public Shape, public Drawable {
    // Shape已经实现了draw()，所以不需要重复实现
};

int main() {
    // 抽象类不能实例化
    // Shape shape;  // 错误：不能实例化抽象类
    
    Shape::describe();  // 调用静态方法
    
    ShapeManager manager;
    
    // 添加各种图形（多态）
    manager.addShape(make_unique<Circle>(5.0));
    manager.addShape(make_unique<Rectangle>(4.0, 6.0));
    manager.addShape(make_unique<Triangle>(3.0, 4.0, 5.0));
    
    cout << "=== 所有图形信息 ===" << endl;
    manager.drawAll();
    
    cout << "=== 统计信息 ===" << endl;
    cout << "总面积: " << manager.totalArea() << endl;
    cout << "总周长: " << manager.totalPerimeter() << endl;
    
    const Shape* largest = manager.findLargest();
    if (largest) {
        cout << "最大面积的图形信息:" << endl;
        largest->printInfo();
    }
    
    // 多态测试：基类指针指向派生类对象
    cout << "\n=== 多态测试 ===" << endl;
    vector<unique_ptr<Shape>> polymorphicShapes;
    polymorphicShapes.push_back(make_unique<Circle>(3.0));
    polymorphicShapes.push_back(make_unique<Rectangle>(2.0, 4.0));
    
    for (const auto& shape : polymorphicShapes) {
        // 运行时多态：根据实际对象类型调用相应方法
        cout << "类型: " << typeid(*shape).name() << endl;
        shape->draw();
        cout << "面积: " << shape->area() << endl;
        cout << "---" << endl;
    }
    
    return 0;
}
```

**纯虚函数与抽象类的关键特性：**

| 特性 | 作用 | 语法 | 注意事项 |
|------|------|------|----------|
| **纯虚函数** | 强制派生类实现 | `virtual 返回类型 函数名() = 0;` | 没有函数体 |
| **抽象类** | 定义接口规范 | 包含纯虚函数的类 | 不能实例化 |
| **虚析构函数** | 正确释放派生类 | `virtual ~类名() {}` | 多态基类必须 |
| **接口分离** | 解耦接口与实现 | 纯虚函数定义接口 | 提高灵活性 |

### 深拷贝与浅拷贝：对象拷贝的内存管理

**拷贝语义设计原理：** C++默认提供浅拷贝，对于包含动态资源的类需要实现深拷贝来避免内存问题和资源冲突。

```cpp
#include <iostream>
#include <cstring>
#include <memory>
using namespace std;

// 浅拷贝问题演示类
class ShallowString {
private:
    char* data;
    int length;
    
public:
    // 构造函数
    ShallowString(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        cout << "构造ShallowString: " << data << endl;
    }
    
    // 默认拷贝构造函数（浅拷贝）
    ShallowString(const ShallowString& other) 
        : data(other.data), length(other.length) {  // 危险：共享内存！
        cout << "浅拷贝构造: " << data << endl;
    }
    
    // 默认赋值运算符（浅拷贝）
    ShallowString& operator=(const ShallowString& other) {
        if (this != &other) {
            delete[] data;  // 释放原有内存
            data = other.data;    // 共享内存
            length = other.length;
        }
        cout << "浅拷贝赋值: " << data << endl;
        return *this;
    }
    
    ~ShallowString() {
        cout << "析构ShallowString: " << (data ? data : "null") << endl;
        delete[] data;  // 可能导致双重释放！
    }
    
    const char* getData() const { return data; }
    int getLength() const { return length; }
};

// 深拷贝实现类
class DeepString {
private:
    char* data;
    int length;
    
public:
    // 构造函数
    DeepString(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        cout << "构造DeepString: " << data << " (地址: " << (void*)data << ")" << endl;
    }
    
    // 深拷贝构造函数
    DeepString(const DeepString& other) {
        length = other.length;
        data = new char[length + 1];  // 分配新内存
        strcpy(data, other.data);     // 复制内容
        cout << "深拷贝构造: " << data << " (地址: " << (void*)data << ")" << endl;
    }
    
    // 深拷贝赋值运算符
    DeepString& operator=(const DeepString& other) {
        if (this != &other) {  // 防止自赋值
            // 先分配新内存再释放旧内存（异常安全）
            char* newData = new char[other.length + 1];
            strcpy(newData, other.data);
            
            delete[] data;  // 释放旧内存
            data = newData; // 指向新内存
            length = other.length;
        }
        cout << "深拷贝赋值: " << data << " (地址: " << (void*)data << ")" << endl;
        return *this;
    }
    
    // 移动构造函数（C++11）
    DeepString(DeepString&& other) noexcept 
        : data(other.data), length(other.length) {
        other.data = nullptr;  // 置空源对象
        other.length = 0;
        cout << "移动构造: " << data << " (地址: " << (void*)data << ")" << endl;
    }
    
    // 移动赋值运算符（C++11）
    DeepString& operator=(DeepString&& other) noexcept {
        if (this != &other) {
            delete[] data;     // 释放当前资源
            data = other.data; // 转移资源
            length = other.length;
            
            other.data = nullptr;  // 置空源对象
            other.length = 0;
        }
        cout << "移动赋值: " << data << " (地址: " << (void*)data << ")" << endl;
        return *this;
    }
    
    ~DeepString() {
        cout << "析构DeepString: " << (data ? data : "null") 
             << " (地址: " << (void*)data << ")" << endl;
        delete[] data;
    }
    
    // 下标运算符重载
    char& operator[](int index) {
        if (index < 0 || index >= length) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    const char& operator[](int index) const {
        if (index < 0 || index >= length) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // 字符串连接运算符
    DeepString operator+(const DeepString& other) const {
        DeepString result;
        delete[] result.data;  // 释放默认构造的内存
        
        result.length = length + other.length;
        result.data = new char[result.length + 1];
        strcpy(result.data, data);
        strcat(result.data, other.data);
        
        return result;  // 可能触发移动语义
    }
    
    const char* getData() const { return data; }
    int getLength() const { return length; }
    
    // 流输出运算符（友元）
    friend ostream& operator<<(ostream& os, const DeepString& str);
};

ostream& operator<<(ostream& os, const DeepString& str) {
    os << str.data;
    return os;
}

// 现代C++解决方案：使用智能指针避免拷贝问题
class ModernString {
private:
    unique_ptr<char[]> data;
    int length;
    
public:
    ModernString(const char* str = "") {
        length = strlen(str);
        data = make_unique<char[]>(length + 1);
        strcpy(data.get(), str);
        cout << "构造ModernString: " << data.get() << endl;
    }
    
    // 禁用拷贝（使用unique_ptr自动管理）
    ModernString(const ModernString&) = delete;
    ModernString& operator=(const ModernString&) = delete;
    
    // 允许移动
    ModernString(ModernString&& other) noexcept 
        : data(move(other.data)), length(other.length) {
        other.length = 0;
        cout << "移动ModernString" << endl;
    }
    
    ModernString& operator=(ModernString&& other) noexcept {
        if (this != &other) {
            data = move(other.data);
            length = other.length;
            other.length = 0;
        }
        cout << "移动赋值ModernString" << endl;
        return *this;
    }
    
    const char* getData() const { return data.get(); }
    int getLength() const { return length; }
};

void demonstrateShallowCopyProblems() {
    cout << "=== 浅拷贝问题演示 ===" << endl;
    
    // 浅拷贝导致的问题
    ShallowString str1("Hello");
    {
        ShallowString str2 = str1;  // 浅拷贝
        cout << "str1: " << str1.getData() << endl;
        cout << "str2: " << str2.getData() << endl;
        
        // str2离开作用域时释放内存，str1成为悬空指针
    }  // str2析构，释放共享内存
    
    // 危险：str1现在指向已释放的内存
    // cout << str1.getData() << endl;  // 未定义行为！
}

void demonstrateDeepCopy() {
    cout << "\n=== 深拷贝解决方案 ===" << endl;
    
    DeepString str1("Hello");
    {
        DeepString str2 = str1;  // 深拷贝
        cout << "str1: " << str1 << " (地址: " << (void*)str1.getData() << ")" << endl;
        cout << "str2: " << str2 << " (地址: " << (void*)str2.getData() << ")" << endl;
        
        // 修改str2不影响str1
        str2[0] = 'J';  // 使用下标运算符
        cout << "修改后 str1: " << str1 << endl;
        cout << "修改后 str2: " << str2 << endl;
        
        // 字符串连接测试
        DeepString str3 = str1 + str2;
        cout << "连接结果: " << str3 << endl;
    }  // str2, str3正常析构
    
    cout << "str1仍然有效: " << str1 << endl;
}

void demonstrateMoveSemantics() {
    cout << "\n=== 移动语义演示 ===" << endl;
    
    DeepString str1("Movable");
    DeepString str2 = move(str1);  // 移动构造
    
    cout << "移动后 str1: " << (str1.getData() ? str1.getData() : "null") << endl;
    cout << "移动后 str2: " << str2 << endl;
    
    DeepString str3("Another");
    str3 = move(str2);  // 移动赋值
    
    cout << "移动赋值后 str2: " << (str2.getData() ? str2.getData() : "null") << endl;
    cout << "移动赋值后 str3: " << str3 << endl;
}

void demonstrateModernSolution() {
    cout << "\n=== 现代C++解决方案 ===" << endl;
    
    ModernString str1("Modern");
    // ModernString str2 = str1;  // 错误：拷贝被禁用
    
    ModernString str2 = move(str1);  // 移动构造
    cout << "移动后 str1: " << (str1.getData() ? str1.getData() : "null") << endl;
    cout << "移动后 str2: " << str2.getData() << endl;
}

int main() {
    demonstrateShallowCopyProblems();
    demonstrateDeepCopy();
    demonstrateMoveSemantics();
    demonstrateModernSolution();
    
    return 0;
}
```

**拷贝语义实现指南：**

| 拷贝类型 | 实现方式 | 适用场景 | 性能特点 |
|----------|----------|----------|----------|
| **浅拷贝** | 默认行为 | 无动态资源的类 | O(1)最快 |
| **深拷贝** | 自定义实现 | 有动态资源的类 | O(n)较慢 |
| **禁用拷贝** | `= delete` | 不可复制资源 | 避免意外拷贝 |
| **移动语义** | 移动构造/赋值 | 临时对象转移 | O(1)高效 |

**Rule of Three/Five法则：**
- **三法则**：如果需要自定义析构函数，通常也需要自定义拷贝构造函数和拷贝赋值运算符
- **五法则**：C++11后，还需要考虑移动构造函数和移动赋值运算符

通过纯虚函数、抽象类和深拷贝机制的学习，可以设计出更加安全、灵活和高效的面向对象系统，这是高质量C++编程的重要基础。

