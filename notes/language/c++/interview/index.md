# C++ 核心面试题

> 高频面试问题，每题包含题目、答案、重点追问

---

## 一、基础语法

### Q1：指针和引用的区别？

**答案：**

| 特性 | 指针 | 引用 |
|------|------|------|
| 本质 | 存储地址的变量 | 变量的别名 |
| 初始化 | 可以不初始化 | 必须初始化 |
| 重新绑定 | 可以指向其他地址 | 绑定后不可变 |
| 空值 | 可以为 nullptr | 不存在"空引用" |
| sizeof | 指针本身大小（8字节） | 引用对象大小 |
| 运算 | 支持算术运算 | 不支持 |

```cpp
int a = 10;
int* p = &a;    // 指针
int& r = a;     // 引用

// 使用场景
void func(const string& s);  // 参数传递用引用
void setCallback(Callback* cb);  // 可能为空用指针
```

**追问：什么是野指针？如何避免？**
> 野指针指向已释放或未知的内存。避免方法：
> 1. 初始化为 nullptr
> 2. delete 后置空
> 3. 使用智能指针

---

### Q2：const 关键字有哪些用法？

**答案：**

```cpp
// 1. 常量
const int MAX = 100;

// 2. 指针（重点）
const int* p1 = &val;   // 指向常量，值不可改
int* const p2 = &val;   // 常量指针，指向不可改
const int* const p3 = &val;  // 都不可改

// 记忆：const 在 * 左边修饰值，在 * 右边修饰指针

// 3. 引用
const int& ref = 10;  // 可绑定临时值

// 4. 成员函数
int getValue() const;  // 不能修改成员变量

// 5. 函数参数
void func(const string& s);  // 避免拷贝，保证不修改
```

**追问：const 和 #define 的区别？**
> const 有类型检查、遵循作用域、可调试；#define 是纯文本替换，无类型检查。

---

### Q3：static 关键字有哪些作用？

**答案：**

```cpp
// 1. 静态局部变量：只初始化一次，生命周期程序全程
void counter() {
    static int count = 0;
    count++;
}

// 2. 静态全局变量/函数：仅本文件可见
static int global_var = 100;
static void helper() {}

// 3. 静态成员变量：所有实例共享
class MyClass {
    static int count;  // 声明
};
int MyClass::count = 0;  // 定义（类外）

// 4. 静态成员函数：无 this 指针，只能访问静态成员
class Utility {
    static void helper();  // 无 this，非虚
};
```

---

### Q4：define 和 inline 的区别？

**答案：**

| 特性 | define | inline |
|------|--------|--------|
| 处理时机 | 预处理 | 编译 |
| 类型检查 | 无 | 有 |
| 调试 | 不可调试 | 可调试 |
| 作用域 | 全局替换 | 遵循作用域 |
| 副作用 | 可能有（如 MAX(a++, b)） | 无 |

```cpp
#define MAX(a, b) ((a) > (b) ? (a) : (b))
inline int max(int a, int b) { return a > b ? a : b; }
```

---

### Q5：sizeof 和 strlen 的区别？

**答案：**

| 特性 | sizeof | strlen |
|------|--------|--------|
| 性质 | 运算符 | 函数 |
| 计算时机 | 编译时 | 运行时 |
| 参数 | 类型/变量 | 字符串指针 |
| 结果 | 字节数 | 字符串长度（不含 \0） |

```cpp
char str[] = "hello";
sizeof(str);   // 6（包含 \0）
strlen(str);   // 5

char* p = str;
sizeof(p);     // 8（指针大小）
strlen(p);     // 5
```

---

### Q6：volatile 关键字的作用？

**答案：**

告诉编译器该变量可能被意外修改，不要优化对该变量的访问。

```cpp
volatile int flag = 0;

// 使用场景：
// 1. 硬件寄存器
// 2. 多线程共享变量（但不够，需配合 atomic）
// 3. 信号处理函数中的变量
```

---

### Q7：类型转换有哪些？推荐哪种？

**答案：**

| C 风格 | C++ 风格 | 用途 |
|--------|----------|------|
| (int)a | static\_cast\<int\>(a) | 基本类型转换 |
| - | dynamic\_cast\<Derived\*\>(base) | 多态向下转型（安全） |
| - | const\_cast\<int\*\>(p) | 去除 const |
| - | reinterpret\_cast\<int\*\>(p) | 底层重新解释（危险） |

```cpp
// 推荐 C++ 风格
double d = 3.14;
int i = static_cast<int>(d);

// dynamic_cast：多态安全转型
Base* base = new Derived();
Derived* derived = dynamic_cast<Derived*>(base);  // 失败返回 nullptr
```

---

## 二、面向对象

### Q8：虚函数的实现原理？

**答案：**

虚函数通过虚表（vtable）实现运行时多态：
- 每个含有虚函数的类有一个虚表，存储虚函数指针
- 对象内存中存储虚表指针（vptr）
- 调用时通过 vptr 查表，实现动态绑定

```cpp
class Base {
public:
    virtual void func1() {}
    virtual void func2() {}
private:
    int data_;  // 4 字节
};
// sizeof(Base) = 8(vptr) + 4(data) + 4(padding) = 16 字节
```

**追问：构造函数可以是虚函数吗？析构函数呢？**
> 构造函数**不能**是虚函数：构造时 vptr 还未初始化。
> 基类析构函数**必须**是虚函数：否则通过基类指针删除派生类对象时只会调用基类析构函数，导致资源泄漏。

---

### Q9：什么是多态？静态多态和动态多态的区别？

**答案：**

多态是指同一接口，不同实现。

| 特性 | 静态多态 | 动态多态 |
|------|----------|----------|
| 时机 | 编译时 | 运行时 |
| 实现 | 函数重载、模板 | 虚函数 |
| 效率 | 更高 | 略低（查表） |
| 灵活性 | 较低 | 更高 |

```cpp
// 静态多态：函数重载
void print(int x);
void print(double x);

// 静态多态：模板
template<typename T>
void process(T value);

// 动态多态：虚函数
class Base {
public:
    virtual void func() {}
};
```

---

### Q10：重载、重写、隐藏的区别？

**答案：**

| 特性 | 重载（Overload） | 重写（Override） | 隐藏（Hide） |
|------|------------------|------------------|--------------|
| 作用域 | 同一作用域 | 基类/派生类 | 基类/派生类 |
| 函数名 | 相同 | 相同 | 相同 |
| 参数 | 不同 | 相同 | 不同（或同但不虚） |
| virtual | 无关 | 必须 | 无关 |

```cpp
class Base {
public:
    void func(int x) {}      // 重载
    virtual void foo() {}    // 虚函数
    void bar(double x) {}    // 非虚
};

class Derived : public Base {
public:
    void func(int x) {}      // 隐藏 Base::func
    void foo() override {}   // 重写
    void bar(int x) {}       // 隐藏 Base::bar（不同参数）
};
```

---

### Q11：构造函数和析构函数的执行顺序？

**答案：**

```cpp
// 构造顺序：基类 → 成员对象 → 派生类
// 析构顺序：派生类 → 成员对象 → 基类（与构造相反）

class Member {
public:
    Member() { cout << "Member ctor\n"; }
    ~Member() { cout << "Member dtor\n"; }
};

class Base {
public:
    Base() { cout << "Base ctor\n"; }
    ~Base() { cout << "Base dtor\n"; }
};

class Derived : public Base {
    Member m;
public:
    Derived() { cout << "Derived ctor\n"; }
    ~Derived() { cout << "Derived dtor\n"; }
};

// 输出：
// Base ctor -> Member ctor -> Derived ctor
// Derived dtor -> Member dtor -> Base dtor
```

---

### Q12：什么是拷贝构造函数？什么时候调用？

**答案：**

```cpp
class MyClass {
public:
    MyClass(const MyClass& other);  // 拷贝构造
};

// 调用场景：
MyClass obj1;
MyClass obj2 = obj1;   // 1. 对象初始化
MyClass obj3(obj1);    // 2. 对象初始化
func(obj1);            // 3. 值传递参数
return obj1;           // 4. 值返回（可能被优化）

// 注意：MyClass obj2 = obj1 是初始化，不是赋值
// MyClass obj2; obj2 = obj1; // 这是赋值，调用 operator=
```

**追问：深拷贝和浅拷贝的区别？**
> 浅拷贝：只复制指针值，两个对象指向同一内存
> 深拷贝：复制指针指向的内容，两个对象独立
> 默认的拷贝构造是浅拷贝，有指针成员时需要自定义深拷贝

---

### Q13：什么是移动构造函数？为什么需要？

**答案：**

```cpp
class MyString {
public:
    // 拷贝构造：深拷贝
    MyString(const MyString& other) {
        data_ = new char[other.size_ + 1];
        strcpy(data_, other.data_);
    }
    
    // 移动构造：转移资源
    MyString(MyString&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;  // 源对象置空
        other.size_ = 0;
    }
};

// 使用场景
MyString s1 = createString();  // 移动而非拷贝
MyString s2 = std::move(s1);   // 显式移动
```

**追问：为什么移动构造要标记 noexcept？**
> STL 容器扩容时会检查：如果移动构造是 noexcept，使用移动；否则使用拷贝。因为移动失败可能导致数据丢失。

---

## 三、STL

### Q14：vector 的扩容机制？迭代器何时失效？

**答案：**

**扩容机制：**
- 容量不足时分配更大内存（GCC 2倍，MSVC 1.5倍）
- 移动/复制元素到新内存
- 释放旧内存

```cpp
vector<int> v;
v.reserve(1000);  // 预分配，避免多次扩容
```

**迭代器失效场景：**

| 操作 | 失效情况 |
|------|----------|
| push_back | 可能全部失效（扩容时） |
| insert | 插入点及之后失效 |
| erase | 被删元素及之后失效 |

```cpp
// 安全删除
for (auto it = v.begin(); it != v.end(); ) {
    if (*it % 2 == 0) {
        it = v.erase(it);  // 返回下一个有效迭代器
    } else {
        ++it;
    }
}

// 更好的写法
v.erase(remove_if(v.begin(), v.end(), [](int x){ return x % 2 == 0; }), v.end());
```

---

### Q15：map 和 unordered_map 的区别？

**答案：**

| 特性 | map | unordered_map |
|------|-----|---------------|
| 底层 | 红黑树 | 哈希表 |
| 有序 | 是 | 否 |
| 查找 | O(log n) | O(1) 平均 |
| 最坏 | O(log n) | O(n)（哈希冲突） |
| 内存 | 较小 | 较大 |

```cpp
// 需要有序遍历 → map
map<int, string> sorted_data;

// 只需快速查找 → unordered_map
unordered_map<string, int> freq;
```

---

### Q16：vector 和 list 的区别？

**答案：**

| 特性 | vector | list |
|------|--------|------|
| 底层 | 连续数组 | 双向链表 |
| 随机访问 | O(1) | O(n) |
| 插入删除 | O(n) | O(1)（已知位置） |
| 内存 | 连续 | 分散 |
| 迭代器失效 | 可能 | 仅被删元素 |

```cpp
// 选择：随机访问多 → vector
//       插入删除多 → list
//       实际中 vector 更常用（缓存友好）
```

---

### Q17：迭代器失效的场景有哪些？

**答案：**

| 容器 | 插入失效 | 删除失效 |
|------|----------|----------|
| vector | 扩容时全部 | 被删及之后 |
| deque | 两端可能，中间全部 | 被删及之后 |
| list/map | 不失效 | 仅被删元素 |
| unordered_map | rehash 时全部 | 仅被删元素 |

---

## 四、内存管理

### Q18：程序的内存布局是怎样的？

**答案：**

```
高地址 ┌─────────────┐
       │    栈区     │ ← 局部变量、函数调用
       │      ↓      │
       │      ↑      │
       │    堆区     │ ← new/malloc
       ├─────────────┤
       │ 全局/静态区 │ ← 全局变量、static 变量
       ├─────────────┤
       │   常量区    │ ← 字符串常量、const 全局
       ├─────────────┤
       │   代码区    │ ← 程序指令
低地址 └─────────────┘
```

---

### Q19：什么是 RAII？

**答案：**

RAII（Resource Acquisition Is Initialization）：资源获取即初始化。
- 构造函数获取资源
- 析构函数释放资源
- 利用对象生命周期自动管理资源

```cpp
class FileHandle {
    FILE* file_;
public:
    FileHandle(const char* path) : file_(fopen(path, "r")) {}
    ~FileHandle() { if (file_) fclose(file_); }
    // 禁止拷贝，允许移动
    FileHandle(const FileHandle&) = delete;
    FileHandle(FileHandle&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
};
```

**优势：** 异常安全、无需手动释放、代码简洁

---

### Q20：智能指针如何选择？

**答案：**

| 智能指针 | 所有权 | 使用场景 |
|----------|--------|----------|
| unique_ptr | 独占 | 默认选择 |
| shared_ptr | 共享 | 需要共享所有权 |
| weak_ptr | 不拥有 | 打破循环引用、观察者 |

```cpp
// unique_ptr：默认选择
auto p1 = make_unique<int>(10);

// shared_ptr：共享所有权
auto sp1 = make_shared<int>(20);
auto sp2 = sp1;  // 引用计数 +1

// weak_ptr：打破循环引用
struct Node {
    shared_ptr<Node> next;
    weak_ptr<Node> prev;  // 用 weak_ptr
};
```

**追问：shared_ptr 是线程安全的吗？**
> 引用计数操作是原子的，线程安全。但对象访问需要额外同步。

**追问：循环引用如何解决？**
> 用 weak_ptr 替代 shared_ptr，weak_ptr 不增加引用计数。

---

### Q21：new/delete 和 malloc/free 的区别？

**答案：**

| 特性 | new/delete | malloc/free |
|------|------------|-------------|
| 性质 | 运算符 | 函数 |
| 返回值 | 类型指针 | void* |
| 失败时 | 抛出异常 | 返回 NULL |
| 构造/析构 | 会调用 | 不会调用 |
| 可重载 | 是 | 否 |

```cpp
// new 做了两件事：1. 分配内存 2. 调用构造函数
// delete 做了两件事：1. 调用析构函数 2. 释放内存

// 定位 new：在已分配内存上构造对象
void* buffer = operator new(sizeof(MyClass));
MyClass* obj = new(buffer) MyClass();  // 定位 new
obj->~MyClass();  // 显式析构
operator delete(buffer);
```

---

### Q22：内存泄漏如何检测和避免？

**答案：**

**检测方法：**
1. 工具：Valgrind、AddressSanitizer、Visual Studio 内存检测
2. 重载 new/delete 记录分配信息

**避免方法：**
1. 使用 RAII 和智能指针
2. 成对使用 new/delete、malloc/free
3. 及时释放资源
4. 避免裸指针持有所有权

```cpp
// 开启 AddressSanitizer（编译选项）
// g++ -fsanitize=address -g main.cpp
```

---

## 五、并发编程

### Q23：如何避免死锁？

**答案：**

死锁四个必要条件：互斥、请求保持、不可剥夺、循环等待

**避免方法：**

```cpp
// 方法1：固定加锁顺序
// 所有线程按相同顺序加锁

// 方法2：std::lock 同时锁定多个锁
std::unique_lock<mutex> lock1(mtx1, std::defer_lock);
std::unique_lock<mutex> lock2(mtx2, std::defer_lock);
std::lock(lock1, lock2);  // 原子地锁定两个

// 方法3：C++17 scoped_lock
std::scoped_lock lock(mtx1, mtx2);  // 最简洁
```

---

### Q24：lock_guard 和 unique_lock 的区别？

**答案：**

| 特性 | lock_guard | unique_lock |
|------|------------|-------------|
| 灵活性 | 低 | 高 |
| 可延迟锁定 | 否 | 是（defer_lock） |
| 可提前解锁 | 否 | 是 |
| 可移动 | 否 | 是 |
| 开销 | 更小 | 略大 |

```cpp
// 简单场景用 lock_guard
{
    lock_guard<mutex> lock(mtx);
    // 临界区
}

// 需要灵活控制用 unique_lock
unique_lock<mutex> lock(mtx, defer_lock);
// 做一些准备工作...
lock.lock();  // 延迟锁定
// 临界区
lock.unlock();  // 提前解锁
```

---

### Q25：条件变量如何使用？

**答案：**

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>

std::mutex mtx;
std::condition_variable cv;
std::queue<int> dataQueue;

// 生产者
void produce(int value) {
    {
        lock_guard<mutex> lock(mtx);
        dataQueue.push(value);
    }
    cv.notify_one();  // 通知消费者
}

// 消费者
void consume() {
    unique_lock<mutex> lock(mtx);
    cv.wait(lock, []{ return !dataQueue.empty(); });  // 等待条件
    int value = dataQueue.front();
    dataQueue.pop();
    lock.unlock();
    // 处理 value
}

// 注意：wait 必须在循环中或使用谓词，防止虚假唤醒
```

---

### Q26：atomic 和 mutex 的区别？

**答案：**

| 特性 | atomic | mutex |
|------|--------|-------|
| 原理 | CPU 原子指令 | 锁机制 |
| 开销 | 小 | 较大 |
| 适用场景 | 简单操作 | 复杂操作 |
| 操作类型 | 单个变量 | 代码块 |

```cpp
atomic<int> counter(0);
counter++;  // 原子操作
counter.fetch_add(1, memory_order_seq_cst);

// 复杂操作仍需 mutex
mutex mtx;
vector<int> data;
{
    lock_guard<mutex> lock(mtx);
    data.push_back(1);  // 多步操作
    data.push_back(2);
}
```

---

## 六、现代 C++

### Q27：什么是右值引用？std::move 的作用？

**答案：**

```cpp
// 左值：有名字，可取地址
int x = 10;       // x 是左值
int& ref = x;     // 左值引用

// 右值：无名字，临时值
int&& rref = 10;  // 右值引用
int&& rref2 = x + 1;  // 绑定临时值

// std::move：将左值转换为右值引用
string s1 = "hello";
string s2 = std::move(s1);  // s1 被移动，之后为空
```

---

### Q28：Lambda 表达式的捕获方式？

**答案：**

```cpp
int a = 10, b = 20;

auto f1 = [a, b]() { return a + b; };    // 值捕获
auto f2 = [&a, &b]() { a++; b++; };      // 引用捕获
auto f3 = [=]() { return a + b; };       // 全部值捕获
auto f4 = [&]() { a++; b++; };           // 全部引用捕获
auto f5 = [=, &a]() { a++; return b; };  // 混合捕获

// mutable：允许修改值捕获的变量
auto f6 = [a]() mutable { return ++a; };
```

**注意：** 值捕获是只读的（除非加 mutable），引用捕获要注意生命周期

---

### Q29：nullptr 和 NULL 的区别？

**答案：**

```cpp
// NULL：可能是 0 或 (void*)0
#define NULL 0
// 或
#define NULL ((void*)0)

// 问题：函数重载时可能出错
void func(int);
void func(int*);
func(NULL);  // 歧义！可能调用 func(int)

// nullptr：类型为 std::nullptr_t
func(nullptr);  // 明确调用 func(int*)
```

**推荐：** 始终使用 nullptr

---

### Q30：auto 和 decltype 的区别？

**答案：**

```cpp
// auto：根据初始值推导类型
auto x = 10;        // int
auto& ref = x;      // int&
auto* ptr = &x;     // int*
const auto& cr = x; // const int&

// decltype：根据表达式推导类型
decltype(x) y = 20;       // int
decltype((x)) ref2 = x;   // int&（注意双括号）

// 返回类型后置
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}
```

---

## 七、高频代码题

### Q31：实现线程安全的单例模式

**答案：**

```cpp
class Singleton {
public:
    static Singleton& instance() {
        static Singleton inst;  // C++11 线程安全
        return inst;
    }
    
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    
private:
    Singleton() = default;
};
```

---

### Q32：实现 String 类

**答案：**

```cpp
class MyString {
public:
    MyString() : data_(new char[1]), size_(0) { data_[0] = '\0'; }
    
    MyString(const char* str) {
        size_ = strlen(str);
        data_ = new char[size_ + 1];
        strcpy(data_, str);
    }
    
    // 拷贝构造
    MyString(const MyString& other) {
        size_ = other.size_;
        data_ = new char[size_ + 1];
        strcpy(data_, other.data_);
    }
    
    // 移动构造
    MyString(MyString&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // 析构
    ~MyString() { delete[] data_; }
    
    // 拷贝赋值
    MyString& operator=(const MyString& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new char[size_ + 1];
            strcpy(data_, other.data_);
        }
        return *this;
    }
    
    // 移动赋值
    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    const char* c_str() const { return data_; }
    size_t size() const { return size_; }

private:
    char* data_;
    size_t size_;
};
```

---

### Q33：实现 unique_ptr

**答案：**

```cpp
template<typename T>
class UniquePtr {
public:
    explicit UniquePtr(T* p = nullptr) : ptr_(p) {}
    
    ~UniquePtr() { delete ptr_; }
    
    // 禁止拷贝
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;
    
    // 允许移动
    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    
    void reset(T* p = nullptr) {
        delete ptr_;
        ptr_ = p;
    }

private:
    T* ptr_;
};
```

---

### Q34：实现 strcpy 和 strlen

**答案：**

```cpp
// strlen
size_t my_strlen(const char* str) {
    size_t len = 0;
    while (*str++) len++;
    return len;
}

// strcpy
char* my_strcpy(char* dest, const char* src) {
    char* ret = dest;
    while ((*dest++ = *src++) != '\0');
    return ret;
}

// 安全版本
errno_t my_strcpy_s(char* dest, size_t size, const char* src) {
    if (!dest || !src) return EINVAL;
    if (size == 0) return ERANGE;
    
    size_t i;
    for (i = 0; i < size - 1 && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    dest[i] = '\0';
    return src[i] == '\0' ? 0 : ERANGE;
}
```

---

### Q35：实现memcpy（处理内存重叠）

**答案：**

```cpp
void* my_memcpy(void* dest, const void* src, size_t n) {
    char* d = (char*)dest;
    const char* s = (const char*)src;
    
    if (d > s && d < s + n) {
        // 有重叠，从后向前拷贝
        d += n;
        s += n;
        while (n--) *--d = *--s;
    } else {
        // 无重叠，从前向后拷贝
        while (n--) *d++ = *s++;
    }
    
    return dest;
}
```

---

## 📝 面试技巧

### 答题层次
1. **直接回答**：先给核心答案
2. **展开解释**：原理、机制
3. **代码示例**：关键代码
4. **注意事项**：边界情况

### 常见失误
- 忘记虚析构函数
- 迭代器失效未处理
- 智能指针循环引用
- 死锁场景
- 资源泄漏
