# Lambda表达式

## 1. 核心概念

### 定义
- **Lambda表达式**：C++11引入的匿名函数对象，用于创建简洁的函数式代码
- **闭包**：捕获外部变量的Lambda函数实例
- **函数对象**：可调用对象，Lambda表达式在编译时转换为函数对象

### 关键特性
- **简洁语法**：内联定义，避免单独的函数声明
- **变量捕获**：可以访问外部作用域的变量
- **类型推断**：编译器自动推断参数和返回类型
- **泛型支持**：C++14起支持auto参数和泛型Lambda

## 2. 语法规则

### 基本语法
```cpp
[捕获列表] (参数列表) -> 返回类型 { 函数体 }

// 简写形式（当返回类型可推断时）
[捕获列表] (参数列表) { 函数体 }

// 无参数简写
[捕获列表] { 函数体 }
```

### 代码示例
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;

// Lambda表达式基本用法演示
void lambdaBasicDemo() {
    cout << "=== Lambda表达式基本用法 ===" << endl;
    
    // 1. 最简单的Lambda：无参数，无捕获
    auto simpleLambda = [] { 
        cout << "Hello, Lambda!" << endl; 
    };
    simpleLambda();
    
    // 2. 带参数的Lambda
    auto add = [](int a, int b) {
        return a + b;
    };
    cout << "5 + 3 = " << add(5, 3) << endl;
    
    // 3. 显式指定返回类型
    auto divide = [](double a, double b) -> double {
        if (b == 0) return 0;
        return a / b;
    };
    cout << "10.0 / 3.0 = " << divide(10.0, 3.0) << endl;
    
    // 4. 在算法中使用Lambda
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // 排序：使用Lambda作为比较函数
    sort(numbers.begin(), numbers.end(), [](int a, int b) {
        return a > b;  // 降序排序
    });
    
    cout << "降序排序: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // 条件计数
    int countEven = count_if(numbers.begin(), numbers.end(), [](int x) {
        return x % 2 == 0;
    });
    cout << "偶数个数: " << countEven << endl;
}

// 变量捕获演示
void captureDemo() {
    cout << "\\n=== 变量捕获 ===" << endl;
    
    int x = 10;
    int y = 20;
    
    // 1. 值捕获 [=]
    auto valueCapture = [=] {  // 捕获所有外部变量（值方式）
        cout << "值捕获: x=" << x << ", y=" << y << endl;
        // x = 5;  // 错误：值捕获的变量是const
    };
    valueCapture();
    
    // 2. 引用捕获 [&]
    auto referenceCapture = [&] {  // 捕获所有外部变量（引用方式）
        cout << "引用捕获前: x=" << x << endl;
        x = 100;  // 修改外部变量
        cout << "引用捕获后: x=" << x << endl;
    };
    referenceCapture();
    cout << "外部x的值: " << x << endl;
    
    // 3. 混合捕获
    auto mixedCapture = [&x, y] {  // x引用捕获，y值捕获
        cout << "混合捕获: x=" << x << " (引用), y=" << y << " (值)" << endl;
        x = 200;  // 可以修改x（引用捕获）
        // y = 300;  // 错误：y是值捕获，const
    };
    mixedCapture();
    cout << "外部x的值: " << x << endl;
    
    // 4. 初始化捕获（C++14）
    auto initCapture = [z = x + y] {  // 捕获时初始化新变量
        cout << "初始化捕获: z=" << z << endl;
    };
    initCapture();
    
    // 5. mutable Lambda（允许修改值捕获的变量）
    auto mutableLambda = [x]() mutable {  // mutable关键字
        cout << "mutable Lambda修改前: x=" << x << endl;
        x = 999;  // 可以修改（副本）
        cout << "mutable Lambda修改后: x=" << x << endl;
    };
    mutableLambda();
    cout << "外部x的值（未改变）: " << x << endl;
}

// 泛型Lambda（C++14）
void genericLambdaDemo() {
    cout << "\\n=== 泛型Lambda（C++14） ===" << endl;
    
    // 1. auto参数
    auto genericAdd = [](auto a, auto b) {
        return a + b;
    };
    
    cout << "整数相加: " << genericAdd(5, 3) << endl;
    cout << "浮点数相加: " << genericAdd(2.5, 3.7) << endl;
    cout << "字符串连接: " << genericAdd(string("Hello, "), string("Lambda!")) << endl;
    
    // 2. 在算法中使用泛型Lambda
    vector<int> intVec = {1, 2, 3, 4, 5};
    vector<double> doubleVec = {1.1, 2.2, 3.3, 4.4, 5.5};
    
    // 通用变换函数
    auto transformGeneric = [](auto& container, auto func) {
        for (auto& element : container) {
            element = func(element);
        }
    };
    
    // 对整数向量平方
    transformGeneric(intVec, [](auto x) { return x * x; });
    cout << "整数平方: ";
    for (auto x : intVec) cout << x << " ";
    cout << endl;
    
    // 对浮点数向量取倒数
    transformGeneric(doubleVec, [](auto x) { return 1.0 / x; });
    cout << "浮点数倒数: ";
    for (auto x : doubleVec) cout << x << " ";
    cout << endl;
}

int main() {
    lambdaBasicDemo();
    captureDemo();
    genericLambdaDemo();
    
    return 0;
}
```

### 注意事项
- Lambda表达式在编译时转换为函数对象
- 捕获列表控制对外部变量的访问方式
- mutable关键字允许修改值捕获的变量
- 返回类型通常可以自动推断，复杂表达式需要显式指定

## 3. 常见用法

### 场景1：STL算法中的Lambda
```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

void stlAlgorithmsWithLambda() {
    cout << "=== STL算法中的Lambda应用 ===" << endl;
    
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. 过滤和转换管道
    vector<int> result;
    
    // 管道：过滤偶数 → 平方 → 过滤大于20的数
    copy_if(numbers.begin(), numbers.end(), back_inserter(result),
            [](int x) { return x % 2 == 0; });
    
    transform(result.begin(), result.end(), result.begin(),
             [](int x) { return x * x; });
    
    result.erase(remove_if(result.begin(), result.end(),
                          [](int x) { return x <= 20; }),
                result.end());
    
    cout << "过滤转换结果: ";
    for (int x : result) cout << x << " ";
    cout << endl;
    
    // 2. 复杂排序
    vector<string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    
    // 按长度排序，长度相同按字母顺序
    sort(words.begin(), words.end(),
         [](const string& a, const string& b) {
             if (a.length() != b.length()) {
                 return a.length() < b.length();
             }
             return a < b;
         });
    
    cout << "按长度和字母排序: ";
    for (const string& word : words) {
        cout << word << " ";
    }
    cout << endl;
    
    // 3. 累积操作
    vector<double> prices = {10.5, 20.3, 15.7, 8.9, 12.4};
    
    // 带折扣的累计价格
    double discount = 0.1;  // 10%折扣
    double total = accumulate(prices.begin(), prices.end(), 0.0,
                             [discount](double sum, double price) {
                                 return sum + price * (1 - discount);
                             });
    
    cout << "折扣后总价: " << total << endl;
    
    // 4. 查找和条件判断
    auto it = find_if(numbers.begin(), numbers.end(),
                     [](int x) { 
                         return x > 5 && x % 3 == 0; 
                     });
    
    if (it != numbers.end()) {
        cout << "找到第一个大于5且是3的倍数的数: " << *it << endl;
    }
    
    // 5. 分区操作
    auto boundary = partition(numbers.begin(), numbers.end(),
                             [](int x) { return x % 2 == 0; });
    
    cout << "分区后（偶数在前）: ";
    for (int x : numbers) cout << x << " ";
    cout << endl;
}
```

### 场景2：回调函数和事件处理
```cpp
#include <functional>
#include <vector>
#include <string>
#include <iostream>

// 事件处理器类
class EventHandler {
private:
    vector<function<void(const string&)>> listeners;
    
public:
    // 添加事件监听器
    void addListener(function<void(const string&)> listener) {
        listeners.push_back(listener);
    }
    
    // 触发事件
    void triggerEvent(const string& eventData) {
        cout << "触发事件，数据: " << eventData << endl;
        for (const auto& listener : listeners) {
            listener(eventData);
        }
    }
};

// 按钮类
class Button {
private:
    string label;
    vector<function<void()>> clickHandlers;
    
public:
    Button(const string& lbl) : label(lbl) {}
    
    // 添加点击处理器
    void onClick(function<void()> handler) {
        clickHandlers.push_back(handler);
    }
    
    // 模拟点击
    void click() {
        cout << "按钮 \"" << label << "\" 被点击" << endl;
        for (const auto& handler : clickHandlers) {
            handler();
        }
    }
    
    string getLabel() const { return label; }
};

// 定时器类
class Timer {
private:
    vector<function<void()>> timeoutHandlers;
    
public:
    void setTimeout(function<void()> handler, int delay) {
        // 模拟异步操作（实际中会使用线程）
        cout << "设置定时器，延迟 " << delay << " 毫秒" << endl;
        timeoutHandlers.push_back(handler);
        
        // 这里简化实现，直接调用（实际应异步执行）
        cout << "定时器到期" << endl;
        handler();
    }
};

void callbackDemo() {
    cout << "\\n=== 回调函数和事件处理 ===" << endl;
    
    // 1. 事件处理器演示
    EventHandler eventHandler;
    
    // 添加多个事件监听器（使用Lambda）
    eventHandler.addListener([](const string& data) {
        cout << "监听器1收到: " << data << endl;
    });
    
    eventHandler.addListener([](const string& data) {
        cout << "监听器2处理数据，长度: " << data.length() << endl;
    });
    
    int callCount = 0;
    eventHandler.addListener([&callCount](const string& data) {
        callCount++;
        cout << "监听器3统计，第" << callCount << "次调用" << endl;
    });
    
    // 触发事件
    eventHandler.triggerEvent("Hello, Event!");
    eventHandler.triggerEvent("Another event");
    
    // 2. 按钮点击处理
    Button btn("提交");
    
    string userName = "Alice";
    int clickCount = 0;
    
    btn.onClick([userName, &clickCount]() {
        clickCount++;
        cout << userName << " 点击了按钮，第" << clickCount << "次" << endl;
    });
    
    btn.onClick([]() {
        cout << "执行提交操作..." << endl;
    });
    
    // 模拟点击
    btn.click();
    btn.click();
    
    // 3. 定时器回调
    Timer timer;
    
    int counter = 0;
    timer.setTimeout([&counter]() {
        counter += 5;
        cout << "定时器回调执行，counter = " << counter << endl;
    }, 1000);
    
    timer.setTimeout([&counter]() {
        cout << "另一个定时器，当前counter: " << counter << endl;
    }, 2000);
}
```

### 场景3：函数组合和柯里化
```cpp
#include <functional>
#include <iostream>

void functionCompositionDemo() {
    cout << "\\n=== 函数组合和柯里化 ===" << endl;
    
    // 1. 函数组合
    auto compose = [](auto f, auto g) {
        return [f, g](auto x) { return f(g(x)); };
    };
    
    auto square = [](int x) { return x * x; };
    auto doubleVal = [](int x) { return x * 2; };
    auto increment = [](int x) { return x + 1; };
    
    // 组合函数：先平方，然后加倍，最后加1
    auto complexFunc = compose(increment, compose(doubleVal, square));
    
    cout << "复杂函数计算(3): " << complexFunc(3) << endl;  // (3² * 2) + 1 = 19
    
    // 2. 柯里化（Currying）
    auto curry = [](auto f) {
        return [f](auto a) {
            return [f, a](auto b) {
                return f(a, b);
            };
        };
    };
    
    auto add = [](int a, int b) { return a + b; };
    auto curriedAdd = curry(add);
    
    auto add5 = curriedAdd(5);  // 固定第一个参数为5
    cout << "柯里化加法(5 + 3): " << add5(3) << endl;
    cout << "柯里化加法(5 + 7): " << add5(7) << endl;
    
    // 3. 部分应用
    auto partial = [](auto f, auto... fixedArgs) {
        return [f, fixedArgs...](auto... remainingArgs) {
            return f(fixedArgs..., remainingArgs...);
        };
    };
    
    auto multiply = [](int a, int b, int c) { return a * b * c; };
    auto multiplyBy2And3 = partial(multiply, 2, 3);  // 固定前两个参数
    
    cout << "部分应用乘法(2*3*4): " << multiplyBy2And3(4) << endl;
    cout << "部分应用乘法(2*3*5): " << multiplyBy2And3(5) << endl;
    
    // 4. 记忆化（Memoization）
    auto memoize = [](auto func) {
        return [func, cache = map<int, int>()](int n) mutable -> int {
            if (cache.find(n) != cache.end()) {
                cout << "缓存命中: fib(" << n << ")" << endl;
                return cache[n];
            }
            
            int result = func(n);
            cache[n] = result;
            cout << "计算并缓存: fib(" << n << ") = " << result << endl;
            return result;
        };
    };
    
    // 斐波那契函数（使用记忆化）
    function<int(int)> fib;
    fib = [&fib](int n) -> int {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    };
    
    auto memoizedFib = memoize(fib);
    
    cout << "记忆化斐波那契数列:" << endl;
    cout << "fib(5) = " << memoizedFib(5) << endl;
    cout << "fib(3) = " << memoizedFib(3) << endl;  // 从缓存获取
    cout << "fib(6) = " << memoizedFib(6) << endl;
}
```

## 4. 易错点/坑

### 错误示例1：悬空引用捕获
```cpp
void danglingReference() {
    function<void()> callback;
    
    {
        int localVar = 42;
        
        // 错误：捕获局部变量的引用
        callback = [&localVar]() {
            cout << "局部变量: " << localVar << endl;  // 悬空引用！
        };
    }  // localVar离开作用域，被销毁
    
    callback();  // 未定义行为！
}
```
**原因**：局部变量离开作用域后被销毁，但Lambda仍持有其引用
**修正方案**：
```cpp
void safeReferenceCapture() {
    function<void()> callback;
    
    // 方案1：值捕获
    int localVar = 42;
    callback = [localVar]() {  // 值捕获，创建副本
        cout << "局部变量副本: " << localVar << endl;
    };
    
    // 方案2：延长变量生命周期
    shared_ptr<int> sharedVar = make_shared<int>(42);
    callback = [sharedVar]() {  // 捕获shared_ptr
        cout << "共享变量: " << *sharedVar << endl;
    };
    
    callback();  // 安全
}
```

### 错误示例2：错误的mutable使用
```cpp
void mutableMisuse() {
    int counter = 0;
    
    // 错误理解mutable的意图
    auto lambda = [counter]() mutable {
        counter++;  // 修改的是副本，不影响外部变量
        cout << "内部counter: " << counter << endl;
    };
    
    lambda();  // 输出: 内部counter: 1
    lambda();  // 输出: 内部counter: 2
    
    cout << "外部counter: " << counter << endl;  // 仍然是0
    
    // 如果意图是修改外部变量，应该使用引用捕获
    auto correctLambda = [&counter]() {  // 引用捕获
        counter++;
        cout << "内部counter: " << counter << endl;
    };
    
    correctLambda();  // 输出: 内部counter: 1
    correctLambda();  // 输出: 内部counter: 2
    cout << "外部counter: " << counter << endl;  // 现在是2
}
```
**原因**：mutable允许修改值捕获的变量副本，但不影响外部变量
**修正方案**：根据意图选择正确的捕获方式

### 错误示例3：性能陷阱
```cpp
void performanceTrap() {
    vector<int> largeData(1000000);
    
    // 低效：在循环中创建复杂的Lambda
    for (int i = 0; i < 1000; ++i) {
        // 每次迭代都创建新的Lambda（可能涉及内存分配）
        auto processor = [i, &largeData]() {
            transform(largeData.begin(), largeData.end(), largeData.begin(),
                     [i](int x) { return x + i; });
        };
        processor();
    }
    
    // 低效：捕获大型对象的值
    vector<int> largeObject(10000);
    
    auto heavyLambda = [largeObject]() {  // 值捕获大型对象，复制开销大
        // 使用largeObject...
    };
}
```
**原因**：不了解Lambda创建和捕获的开销
**修正方案**：
```cpp
void performanceOptimized() {
    vector<int> largeData(1000000);
    
    // 高效：在循环外创建Lambda
    auto processor = [&largeData](int i) {  // 参数化
        transform(largeData.begin(), largeData.end(), largeData.begin(),
                 [i](int x) { return x + i; });
    };
    
    for (int i = 0; i < 1000; ++i) {
        processor(i);  // 重复使用同一个Lambda
    }
    
    // 高效：引用捕获大型对象
    vector<int> largeObject(10000);
    
    auto lightLambda = [&largeObject]() {  // 引用捕获，无复制开销
        // 使用largeObject...
    };
    
    // 或者使用移动捕获（C++14）
    auto moveLambda = [data = move(largeObject)]() {  // 移动语义
        // 使用data...
    };
}
```

## 5. 拓展补充

### 关联知识点
- **std::function**：类型擦除的函数包装器
- **函数指针**：C风格回调机制
- **函数对象**：重载operator()的类
- **调用运算符**：operator()的重载规则

### 进阶延伸
- **Lambda表达式优化**：编译器如何优化Lambda
- **泛型Lambda（C++14）**：auto参数和模板Lambda
- **初始化捕获（C++14）**：在捕获时初始化变量
- **constexpr Lambda（C++17）**：编译时求值的Lambda
- **模板Lambda（C++20）**：显式模板参数列表

# 右值引用与移动语义

## 1. 核心概念

### 定义
- **右值引用**：C++11引入的引用类型，用于绑定到临时对象（右值）
- **移动语义**：将资源所有权从一个对象转移到另一个对象，避免不必要的拷贝
- **完美转发**：保持参数的值类别（左值/右值）进行转发

### 关键特性
- **性能优化**：避免深拷贝，提高大对象操作效率
- **资源管理**：明确资源所有权转移语义
- **标准库支持**：STL容器和算法全面支持移动语义
- **编译器优化**：自动识别可移动场景

## 2. 语法规则

### 基本语法
```cpp
// 右值引用声明
类型&& 引用名 = 右值表达式;

// 移动构造函数
类名(类名&& other);

// 移动赋值运算符
类名& operator=(类名&& other);

// 完美转发
template<typename T>
void func(T&& arg);  // 通用引用
```

### 代码示例
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <utility>  // std::move, std::forward
using namespace std;

// 演示移动语义的字符串类
class MyString {
private:
    char* data;
    size_t length;
    
public:
    // 构造函数
    MyString(const char* str = "") : data(nullptr), length(0) {
        if (str) {
            length = strlen(str);
            data = new char[length + 1];
            strcpy(data, str);
            cout << "构造: " << data << endl;
        }
    }
    
    // 拷贝构造函数（深拷贝）
    MyString(const MyString& other) : data(nullptr), length(0) {
        length = other.length;
        if (length > 0) {
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        cout << "拷贝构造: " << (data ? data : "空") << endl;
    }
    
    // 移动构造函数
    MyString(MyString&& other) noexcept : data(other.data), length(other.length) {
        other.data = nullptr;  // 置空原对象，防止双重释放
        other.length = 0;
        cout << "移动构造: " << (data ? data : "空") << endl;
    }
    
    // 拷贝赋值运算符
    MyString& operator=(const MyString& other) {
        if (this != &other) {
            delete[] data;  // 释放原有资源
            
            length = other.length;
            if (length > 0) {
                data = new char[length + 1];
                strcpy(data, other.data);
            } else {
                data = nullptr;
            }
        }
        cout << "拷贝赋值: " << (data ? data : "空") << endl;
        return *this;
    }
    
    // 移动赋值运算符
    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] data;  // 释放原有资源
            
            data = other.data;    // 接管资源
            length = other.length;
            
            other.data = nullptr;  // 置空原对象
            other.length = 0;
        }
        cout << "移动赋值: " << (data ? data : "空") << endl;
        return *this;
    }
    
    // 析构函数
    ~MyString() {
        delete[] data;
        cout << "析构" << endl;
    }
    
    const char* c_str() const { return data ? data : ""; }
    size_t size() const { return length; }
    
    friend ostream& operator<<(ostream& os, const MyString& str) {
        return os << (str.data ? str.data : "空");
    }
};

// 基本移动语义演示
void basicMoveSemantics() {
    cout << "=== 基本移动语义演示 ===" << endl;
    
    // 1. 普通构造
    MyString str1("Hello");
    
    // 2. 拷贝构造
    MyString str2 = str1;  // 调用拷贝构造函数
    
    // 3. 移动构造（使用std::move）
    MyString str3 = move(str1);  // 调用移动构造函数
    
    cout << "str1: " << str1 << endl;  // 空（资源已被移动）
    cout << "str2: " << str2 << endl;  // Hello
    cout << "str3: " << str3 << endl;  // Hello
    
    // 4. 移动赋值
    MyString str4("World");
    str4 = move(str3);  // 调用移动赋值运算符
    
    cout << "str3: " << str3 << endl;  // 空
    cout << "str4: " << str4 << endl;  // Hello
}

// 右值引用演示
void rvalueReferenceDemo() {
    cout << "\\n=== 右值引用演示 ===" << endl;
    
    MyString str("临时字符串");
    
    // 左值引用
    MyString& lref = str;  // 正确：绑定到左值
    // MyString& lref2 = MyString("临时");  // 错误：不能绑定右值到左值引用
    
    // 右值引用
    MyString&& rref = MyString("临时");  // 正确：绑定到右值
    // MyString&& rref2 = str;  // 错误：不能绑定左值到右值引用
    
    // 常量左值引用可以绑定右值
    const MyString& const_ref = MyString("临时");  // 正确
    
    cout << "右值引用内容: " << rref << endl;
}

// 移动语义在STL中的应用
void stlMoveSemantics() {
    cout << "\\n=== STL中的移动语义 ===" << endl;
    
    vector<MyString> strings;
    
    // 1. 插入临时对象（移动构造）
    cout << "插入临时对象:" << endl;
    strings.push_back(MyString("第一个"));  // 移动构造
    
    // 2. 插入具名对象（拷贝构造）
    cout << "插入具名对象:" << endl;
    MyString named("第二个");
    strings.push_back(named);  // 拷贝构造
    
    // 3. 显式移动具名对象
    cout << "移动具名对象:" << endl;
    strings.push_back(move(named));  // 移动构造
    
    cout << "named after move: " << named << endl;  // 空
    
    // 4. 向量扩容时的移动
    cout << "向量扩容:" << endl;
    strings.reserve(10);  // 触发重新分配，移动现有元素
    
    cout << "最终内容:" << endl;
    for (const auto& s : strings) {
        cout << s << endl;
    }
}

int main() {
    basicMoveSemantics();
    rvalueReferenceDemo();
    stlMoveSemantics();
    
    return 0;
}
```

### 注意事项
- 移动操作后，原对象处于有效但未定义状态
- 移动构造函数和赋值运算符应标记为noexcept
- std::move只是类型转换，不实际移动任何东西
- 完美转发需要模板和通用引用配合

## 3. 常见用法

### 场景1：完美转发实现
```cpp
#include <iostream>
#include <utility>

// 完美转发包装器
class Logger {
public:
    // 通用引用模板 - 完美转发的关键
    template<typename T>
    void log(T&& value) {
        // std::forward保持值类别
        process(std::forward<T>(value));
    }
    
private:
    // 处理左值版本
    void process(const string& value) {
        cout << "处理左值: " << value << endl;
    }
    
    // 处理右值版本
    void process(string&& value) {
        cout << "处理右值: " << value << endl;
        // 可以安全地移动value
    }
    
    // 处理整数版本
    void process(int value) {
        cout << "处理整数: " << value << endl;
    }
};

void perfectForwardingDemo() {
    cout << "=== 完美转发演示 ===" << endl;
    
    Logger logger;
    
    // 左值
    string lvalue = "左值字符串";
    logger.log(lvalue);  // 调用左值版本
    cout << "左值处理后: " << lvalue << endl;  // 仍然有效
    
    // 右值
    logger.log(string("右值字符串"));  // 调用右值版本
    
    // 字面量
    logger.log("字面量");  // 调用右值版本
    
    // 整数
    int x = 42;
    logger.log(x);       // 左值版本
    logger.log(123);     // 右值版本
}

// 工厂函数模板
template<typename T, typename... Args>
T create(Args&&... args) {
    // 完美转发所有参数
    return T(std::forward<Args>(args)...);
}

class Product {
    string name;
    int id;
public:
    Product(const string& n, int i) : name(n), id(i) {
        cout << "Product构造: " << name << "(" << id << ")" << endl;
    }
    
    Product(string&& n, int i) : name(std::move(n)), id(i) {
        cout << "Product移动构造: " << name << "(" << id << ")" << endl;
    }
};

void factoryDemo() {
    cout << "\\n工厂函数演示:" << endl;
    
    string name = "产品A";
    
    // 使用左值
    auto p1 = create<Product>(name, 1);  // 拷贝构造name
    
    // 使用右值
    auto p2 = create<Product>(string("产品B"), 2);  // 移动构造
    
    // 使用字面量
    auto p3 = create<Product>("产品C", 3);  // 移动构造
}
```

### 场景2：资源管理类
```cpp
#include <fstream>
#include <memory>

// RAII文件资源管理器
class FileResource {
private:
    unique_ptr<FILE, decltype(&fclose)> file;
    
public:
    FileResource(const char* filename, const char* mode)
        : file(fopen(filename, mode), &fclose) {
        if (!file) throw runtime_error("文件打开失败");
    }
    
    // 移动构造函数
    FileResource(FileResource&& other) noexcept
        : file(move(other.file)) {}
    
    // 移动赋值运算符
    FileResource& operator=(FileResource&& other) noexcept {
        if (this != &other) {
            file = move(other.file);
        }
        return *this;
    }
    
    void write(const string& content) {
        if (file) {
            fwrite(content.c_str(), 1, content.size(), file.get());
        }
    }
    
    // 禁用拷贝
    FileResource(const FileResource&) = delete;
    FileResource& operator=(const FileResource&) = delete;
};

void resourceManagementDemo() {
    cout << "\\n=== 资源管理类演示 ===" << endl;
    
    // 创建文件资源
    FileResource file1("test1.txt", "w");
    file1.write("Hello, World!");
    
    // 移动资源所有权
    FileResource file2 = move(file1);
    file2.write("\\nMoved resource");
    
    // file1不再拥有资源，但处于有效状态
}
```

### 场景3：STL容器优化
```cpp
void stlOptimizationDemo() {
    cout << "\\n=== STL容器优化 ===" << endl;
    
    vector<string> strings;
    
    // emplace_back直接构造（C++11）
    strings.emplace_back("直接构造");  // 避免临时对象
    
    // 移动语义优化
    string largeString(1000, 'x');
    strings.push_back(move(largeString));  // 移动而非拷贝
    
    // reserve预分配避免重新分配
    strings.reserve(100);
    
    // 移动语义在算法中的应用
    vector<string> source = {"a", "b", "c"};
    vector<string> destination;
    
    // 移动所有元素
    move(source.begin(), source.end(), back_inserter(destination));
    
    cout << "源容器大小: " << source.size() << endl;  // 3（但元素可能为空）
    cout << "目标容器大小: " << destination.size() << endl;  // 3
}
```

## 4. 易错点/坑

### 错误示例1：过度使用std::move
```cpp
void overuseMoveDemo() {
    string str = "hello";
    
    // 错误的过度使用
    string copy1 = move(str);  // 不必要，str还需要使用
    
    // 正确：仅在需要转移所有权时使用
    vector<string> vec;
    vec.push_back(move(str));  // 正确：str不再需要
    
    // 错误：对小对象使用move可能更慢
    int x = 42;
    int y = move(x);  // 对小整型使用move没有意义
}
```

### 错误示例2：移动后使用原对象
```cpp
void useAfterMoveDemo() {
    vector<int> data = {1, 2, 3};
    
    // 移动后使用原对象
    vector<int> newData = move(data);
    
    // 错误：data处于有效但未定义状态
    // cout << data.size() << endl;  // 未定义行为
    
    // 正确：重新赋值或避免使用
    data = {4, 5, 6};  // 重新赋值
    cout << data.size() << endl;  // 正确
}
```

### 错误示例3：错误的noexcept声明
```cpp
class BadMoveClass {
    vector<int> data;
public:
    // 错误：移动操作可能抛出异常
    BadMoveClass(BadMoveClass&& other) {
        data = move(other.data);  // vector移动可能抛出
    }
    
    // 正确：标记为noexcept
    BadMoveClass(BadMoveClass&& other) noexcept {
        data = move(other.data);
    }
};
```

### 错误示例4：忘记实现移动操作
```cpp
class NoMoveClass {
    int* data;
public:
    NoMoveClass() : data(new int(42)) {}
    ~NoMoveClass() { delete data; }
    
    // 只有拷贝操作，没有移动操作
    NoMoveClass(const NoMoveClass& other) : data(new int(*other.data)) {}
    
    // 错误：需要移动操作时仍使用拷贝
    // vector<NoMoveClass> vec;
    // vec.push_back(NoMoveClass());  // 调用拷贝构造函数
};
```

## 5. 拓展补充

### 关联知识点

#### 返回值优化（RVO）
```cpp
string createString() {
    return string("临时字符串");  // 可能触发RVO
}

void rvoDemo() {
    // RVO可能优化为直接构造
    string str = createString();
    
    // 强制移动（可能干扰RVO）
    string str2 = move(createString());  // 不推荐
}
```

#### 通用引用（Universal Reference）
```cpp
template<typename T>
void universalReferenceDemo(T&& param) {
    // T&& 可能是左值引用或右值引用
    if constexpr (is_lvalue_reference_v<T>) {
        cout << "左值引用" << endl;
    } else {
        cout << "右值引用" << endl;
    }
}
```

### 进阶延伸

#### 移动语义与异常安全
```cpp
class ExceptionSafeResource {
    unique_ptr<int> resource;
public:
    // 强异常安全保证
    void swap(ExceptionSafeResource& other) noexcept {
        using std::swap;
        swap(resource, other.resource);
    }
    
    ExceptionSafeResource& operator=(ExceptionSafeResource other) noexcept {
        swap(other);
        return *this;
    }
};
```

#### 自定义移动迭代器
```cpp
template<typename Iterator>
class move_iterator {
    Iterator current;
public:
    using value_type = typename iterator_traits<Iterator>::value_type;
    
    move_iterator(Iterator it) : current(it) {}
    
    auto operator*() -> decltype(move(*current)) {
        return move(*current);
    }
    
    // 其他迭代器操作...
};
```

#### 完美转发与可变参数模板
```cpp
template<typename... Args>
void perfectForwardingVariadic(Args&&... args) {
    // 完美转发所有参数
    someFunction(forward<Args>(args)...);
}

void variadicDemo() {
    string str = "hello";
    perfectForwardingVariadic(str, 42, "world");
}
```

### 性能测试与基准
```cpp
#include <chrono>

void performanceBenchmark() {
    const int SIZE = 1000000;
    
    auto start = chrono::high_resolution_clock::now();
    
    // 测试拷贝性能
    vector<vector<int>> copyVec;
    for (int i = 0; i < 100; ++i) {
        vector<int> temp(SIZE, i);
        copyVec.push_back(temp);  // 拷贝
    }
    
    auto copyTime = chrono::high_resolution_clock::now() - start;
    
    start = chrono::high_resolution_clock::now();
    
    // 测试移动性能
    vector<vector<int>> moveVec;
    for (int i = 0; i < 100; ++i) {
        vector<int> temp(SIZE, i);
        moveVec.push_back(move(temp));  // 移动
    }
    
    auto moveTime = chrono::high_resolution_clock::now() - start;
    
    cout << "拷贝时间: " 
         << chrono::duration_cast<chrono::milliseconds>(copyTime).count() 
         << "ms" << endl;
    cout << "移动时间: " 
         << chrono::duration_cast<chrono::milliseconds>(moveTime).count() 
         << "ms" << endl;
}
```

## 总结

右值引用与移动语义是现代C++性能优化的关键技术：

1. **理解值类别**：区分左值、右值、将亡值
2. **正确使用移动**：在适当场景使用std::move
3. **实现移动操作**：为资源管理类实现移动构造和赋值
4. **掌握完美转发**：保持参数值类别进行转发
5. **注意异常安全**：移动操作应标记为noexcept
6. **避免常见错误**：不要过度使用或移动后使用原对象

通过合理使用移动语义，可以显著提升大对象操作的性能，避免不必要的深拷贝，是现代C++高性能编程的必备技能。