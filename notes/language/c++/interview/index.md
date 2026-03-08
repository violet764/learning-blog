# C++ 核心面试题

> 高频面试问题，每题包含题目、答案、重点追问

---

## 一、基础语法

### Q1：指针和引用的区别？

**一段式回答：** 指针是一个存储内存地址的变量，可以被重新赋值、可以为空、支持算术运算；而引用是变量的别名，必须在定义时初始化、绑定后不可更改、不存在空引用。实际使用中，参数传递优先用引用（避免拷贝），可能为空或需要重新指向时用指针。

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

**一段式回答：** const 主要有五种用法：定义常量（const int MAX = 100）、修饰指针（const int* p 表示指向的值不可变，int* const p 表示指针本身不可变）、修饰引用（const int& 可绑定临时值）、修饰成员函数（表示不修改成员变量）、修饰函数参数（避免拷贝同时保证不修改）。记忆技巧：const 在 * 左边修饰值，在 * 右边修饰指针。

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

**一段式回答：** static 有四种作用：修饰局部变量时，只初始化一次，生命周期延长到程序结束；修饰全局变量或函数时，限制其作用域仅在当前文件；修饰类成员变量时，所有实例共享同一份数据；修饰类成员函数时，没有 this 指针，只能访问静态成员。

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

**一段式回答：** define 是预处理阶段的文本替换，无类型检查、不可调试、可能有副作用（如 MAX(a++, b) 会执行两次）；inline 是编译阶段的内联函数，有类型检查、可调试、遵循作用域规则。inline 是更安全、更推荐的方式，但只是对编译器的建议，编译器可能会忽略。

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

**一段式回答：** sizeof 是编译时运算符，计算类型或变量占用的字节数，数组名表示整个数组大小（含 \0）；strlen 是运行时函数，计算字符串长度（不含 \0），需要遍历到 \0 为止。注意 sizeof(指针) 返回指针本身大小（8字节），而 strlen(指针) 返回字符串长度。

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

**一段式回答：** volatile 告诉编译器该变量可能被意外修改（如硬件寄存器、多线程共享、信号处理函数），禁止编译器对该变量进行优化，每次都从内存读取。注意：volatile 不保证线程安全，多线程场景应使用 atomic 或锁。

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

**一段式回答：** C++ 推荐使用四种新型转换：static_cast 用于基本类型转换和上行转型；dynamic_cast 用于多态类型的安全下行转型（失败返回 nullptr）；const_cast 用于去除 const 属性；reinterpret_cast 用于底层重新解释（危险，慎用）。相比 C 风格转换，C++ 转换更安全、意图明确、易于搜索。

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

**一段式回答：** 虚函数通过虚表和虚表指针实现运行时多态。每个含有虚函数的类都有一个虚表，存储虚函数的地址；每个对象内存中存储一个虚表指针指向类的虚表。调用虚函数时，通过 vptr 查表找到正确的函数地址，实现动态绑定。注意：构造函数不能是虚函数（构造时 vptr 还未初始化），基类析构函数必须是虚函数（确保正确调用派生类析构函数）。

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

**一段式回答：** 多态是指同一接口、不同实现的能力。静态多态发生在编译时，通过函数重载和模板实现，效率高但灵活性低；动态多态发生在运行时，通过虚函数实现，效率略低但灵活性高。选择时，性能敏感场景优先静态多态，需要运行时决定场景用动态多态。

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

**一段式回答：** 重载发生在同一作用域，函数名相同但参数不同，与 virtual 无关；重写发生在基类和派生类之间，函数签名完全相同且必须是虚函数，使用 override 关键字显式声明；隐藏发生在基类和派生类之间，派生类函数与基类函数同名但参数不同（或参数相同但非虚函数），基类函数被隐藏。区分关键：重载看参数，重写看 virtual，隐藏看作用域。

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

**一段式回答：** 构造顺序：基类构造函数 → 成员对象构造函数 → 派生类构造函数。析构顺序：派生类析构函数 → 成员对象析构函数 → 基类析构函数（与构造相反）。原因：派生类依赖基类和成员对象，必须先完成依赖对象的构造；析构时派生类先释放自己的资源，再依次释放依赖对象。

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

**一段式回答：** 拷贝构造函数用于通过已有对象初始化新对象，形式为 T(const T& other)。调用场景：对象初始化（MyClass obj2 = obj1）、值传递参数、值返回（可能被 RVO 优化）。注意区分初始化和赋值：MyClass obj2 = obj1 是拷贝构造，obj2 = obj1 是赋值运算符。默认拷贝构造是浅拷贝，有指针成员时需要自定义深拷贝。

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

**一段式回答：** 移动构造函数用于转移资源而非复制，形式为 T(T&& other) noexcept，将源对象的指针"偷"过来并将源对象置空。适用于处理临时对象和 std::move 后的对象，避免深拷贝的开销。标记 noexcept 是因为 STL 容器扩容时会检查：如果移动构造是 noexcept，使用移动；否则使用拷贝，因为移动失败可能导致数据丢失。

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

**一段式回答：** vector 扩容机制：容量不足时分配更大的内存（GCC 2倍，MSVC 1.5倍），移动元素到新内存，释放旧内存。可使用 reserve 预分配避免多次扩容。迭代器失效：push_back 可能导致全部失效（扩容时）；insert 导致插入点及之后失效；erase 导致被删元素及之后失效。安全删除方法：it = v.erase(it) 获取下一个有效迭代器。

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

**一段式回答：** map 底层是红黑树，元素有序存储，查找 O(log n)，内存占用较小；unordered_map 底层是哈希表，元素无序，查找 O(1) 平均但最坏 O(n)，内存占用较大。选择：需要有序遍历或范围查询用 map；只需要快速查找且不关心顺序用 unordered_map。

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

**一段式回答：** vector 是连续内存的动态数组，随机访问 O(1)，插入删除 O(n)，内存连续缓存友好；list 是双向链表，随机访问 O(n)，已知位置插入删除 O(1)，内存分散。实际开发中 vector 更常用，因为缓存友好性往往比算法复杂度更重要，只有频繁在中间插入删除时才考虑 list。

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

**一段式回答：** 序列容器：vector 插入可能全部失效（扩容时），删除导致被删及之后失效；deque 插入两端可能、中间全部失效，删除导致被删及之后失效；list/map 插入不失效，删除仅被删元素失效；unordered_map 的 rehash 会导致全部失效。关键原则：做任何修改容器的操作后，不要假设迭代器仍然有效，应该重新获取或使用返回值。

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

**一段式回答：** 程序内存从高地址到低地址依次为：栈区（局部变量、函数调用，向下增长）、堆区（new/malloc，向上增长）、全局/静态区（全局变量、static 变量）、常量区（字符串常量、const 全局变量）、代码区（程序指令）。栈空间有限且自动管理，堆空间大但需手动管理。

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

**一段式回答：** RAII（资源获取即初始化）是一种资源管理技术：在构造函数中获取资源，在析构函数中释放资源，利用对象生命周期自动管理资源。优势：异常安全（即使发生异常也会正确释放）、无需手动释放、代码简洁。C++ 智能指针、锁管理（lock_guard）都是 RAII 的典型应用。

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

**一段式回答：** unique_ptr 独占所有权，是默认选择，轻量高效；shared_ptr 共享所有权，引用计数管理，用于需要共享的场景；weak_ptr 不拥有所有权，用于打破循环引用和观察者模式。选择原则：优先 unique_ptr，确需共享时用 shared_ptr，有循环引用风险时用 weak_ptr。注意：shared_ptr 的引用计数是线程安全的，但对象访问需要额外同步。

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

**一段式回答：** new/delete 是 C++ 运算符，malloc/free 是 C 函数。主要区别：new 返回类型指针，malloc 返回 void*；new 失败抛出异常，malloc 返回 NULL；new/delete 会调用构造/析构函数，malloc/free 不会；new 可以重载，malloc 不能。推荐在 C++ 中使用 new/delete，保证对象的正确构造和析构。

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

**一段式回答：** 检测方法：工具检测（Valgrind、AddressSanitizer、Visual Studio 内存检测）、重载 new/delete 记录分配信息。避免方法：使用 RAII 和智能指针（最有效）、成对使用 new/delete 和 malloc/free、避免裸指针持有所有权、使用 STL 容器代替手动管理数组。核心原则：让对象的生命周期管理资源的生命周期。

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

**一段式回答：** 死锁需要四个条件：互斥、请求保持、不可剥夺、循环等待。避免方法：固定加锁顺序（所有线程按相同顺序加锁）、使用 std::lock 或 C++17 的 scoped_lock 同时锁定多个锁、使用 try_lock 避免无限等待、减少锁持有时间。推荐使用 scoped_lock，最简洁且自动处理多锁顺序。

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

**一段式回答：** lock_guard 简单轻量，构造时加锁，析构时解锁，适用于简单场景；unique_lock 更灵活，支持延迟锁定（defer_lock）、提前解锁、条件变量配合、移动语义，但开销略大。选择原则：简单场景用 lock_guard，需要灵活控制（如条件变量、延迟锁定）时用 unique_lock。

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

**一段式回答：** 条件变量用于线程间等待/通知机制，配合 unique_lock 使用。生产者：加锁后修改共享数据，调用 notify_one/notify_all 通知。消费者：在 unique_lock 上调用 wait，可带谓词避免虚假唤醒。注意：wait 必须在循环中或使用谓词，因为可能发生虚假唤醒；wait 会自动释放锁并阻塞，被唤醒后重新获取锁。

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

**一段式回答：** atomic 利用 CPU 原子指令实现，开销小，适用于单个变量的简单操作（如计数器、标志位）；mutex 是锁机制，开销较大，适用于需要保护复杂操作或多个变量的代码块。选择原则：简单的单变量操作用 atomic，复杂的临界区用 mutex。atomic 支持多种内存序，默认 memory_order_seq_cst 保证最强一致性。

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

**一段式回答：** 右值引用（T&&）是 C++11 引入的新类型，用于绑定到右值（临时对象），实现移动语义。左值是有名字、可取地址的对象；右值是无名字的临时值。std::move 将左值强制转换为右值引用，使编译器选择移动构造而非拷贝构造，适用于资源转移场景，转移后原对象处于有效但未定义状态，不应再使用。

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

**一段式回答：** Lambda 捕获方式：[a, b] 值捕获（只读，除非加 mutable）；[&a, &b] 引用捕获（注意生命周期）；[=] 全部值捕获；[&] 全部引用捕获；[=, &a] 混合捕获。选择原则：能值捕获就值捕获，引用捕获要确保被引用对象生命周期足够长。注意：值捕获是捕获时的值，不是实时值；mutable 允许修改值捕获的变量（修改的是副本）。

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

**一段式回答：** NULL 是宏定义，可能是 0 或 ((void*)0)，在函数重载时可能导致歧义（调用 func(int) 而非 func(int*)）；nullptr 是 C++11 引入的关键字，类型为 std::nullptr_t，可以隐式转换为任意指针类型，不会与整数类型混淆。推荐始终使用 nullptr，类型安全且无歧义。

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

**一段式回答：** auto 根据初始值推导类型，用于简化变量声明和迭代器类型；decltype 根据表达式推导类型，保留引用和 const 属性。区别：auto 会忽略顶层 const 和引用，decltype 保留；decltype((x)) 得到引用类型（注意双括号）。常见用法：auto 简化声明，decltype 用于返回类型推导和泛型编程。

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

**一段式回答：** C++11 后最简单的方法是使用静态局部变量（Meyers 单例）：在函数内定义 static 变量，C++11 保证其初始化是线程安全的。需删除拷贝构造和赋值运算符，私有化构造函数。这种方法利用了编译器的线程安全静态初始化机制，无需手动加锁，推荐作为首选方案。

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

**一段式回答：** String 类需要实现：默认构造、带参构造（const char*）、拷贝构造（深拷贝）、移动构造（转移资源）、析构（释放内存）、拷贝赋值运算符（处理自赋值、先删后拷）、移动赋值运算符。关键点：遵循三五法则（三/五/零原则），移动操作要将源对象置空，赋值要处理自赋值。

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

**一段式回答：** unique_ptr 需要实现：构造函数、析构函数（delete）、禁用拷贝构造和拷贝赋值、移动构造（转移所有权并置空源指针）、移动赋值运算符、解引用运算符（* 和 ->）、get()、release()、reset()。核心原则：独占所有权，不可拷贝，只能移动。

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

**一段式回答：** strlen：遍历字符串直到 \0，返回字符数。strcpy：逐字符复制直到遇到 \0，返回目标地址。注意：strcpy 不检查目标缓冲区大小，有缓冲区溢出风险；安全版本 strcpy_s 需要指定缓冲区大小。面试中要主动提安全问题和边界检查。

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

**一段式回答：** memcpy 的关键问题是内存重叠：当目标地址在源地址范围内时，从前向后拷贝会覆盖未拷贝的源数据。解决方案：判断是否有重叠（dest > src && dest < src + n），有重叠则从后向前拷贝，无重叠则从前向后拷贝。memmove 标准库已经处理了这个问题，面试时要说明原理。

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
