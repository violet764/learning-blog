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