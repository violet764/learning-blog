# C++进阶特性（核心机制篇）

## 1. STL标准库

### 核心概念
- **STL组成**：容器、算法、迭代器、函数对象
- **泛型编程**：模板实现的通用数据结构
- **算法统一**：标准接口处理各种容器

### 关键语法
```cpp
#include <vector>
#include <algorithm>
#include <functional>

vector<int> vec = {1, 2, 3, 4, 5};
sort(vec.begin(), vec.end(), greater<int>());
```

### 实战示例
```cpp
// 综合STL使用示例
void stlComprehensiveDemo() {
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // 排序和查找
    sort(numbers.begin(), numbers.end());
    auto it = lower_bound(numbers.begin(), numbers.end(), 4);
    
    // 函数式编程
    vector<int> doubled;
    transform(numbers.begin(), numbers.end(), back_inserter(doubled),
             [](int x) { return x * 2; });
    
    // 条件删除
    numbers.erase(remove_if(numbers.begin(), numbers.end(),
                           [](int x) { return x % 2 == 0; }), 
                 numbers.end());
}
```

## 2. 容器详解

### 容器分类
1. **序列容器**：vector, deque, list, forward_list
2. **关联容器**：set, map, multiset, multimap  
3. **无序容器**：unordered_set, unordered_map

### 性能对比
```cpp
// 不同容器性能特点
void containerPerformance() {
    // vector: 随机访问快，尾部插入快
    vector<int> vec(1000);
    vec[500] = 42;  // O(1)
    
    // list: 任意位置插入删除快
    list<int> lst = {1, 2, 3};
    lst.insert(next(lst.begin()), 4);  // O(1)
    
    // map: 按键排序，查找O(log n)
    map<string, int> wordCount;
    wordCount["hello"] = 1;
    
    // unordered_map: 哈希表，平均O(1)查找
    unordered_map<string, int> fastWordCount;
    fastWordCount["world"] = 2;
}
```

## 3. 迭代器与算法

### 迭代器类别
- **输入迭代器**：只读前向访问
- **输出迭代器**：只写前向访问
- **前向迭代器**：读写前向访问
- **双向迭代器**：双向移动
- **随机访问迭代器**：任意位置访问

### 算法应用
```cpp
void algorithmPatterns() {
    vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // 非修改序列算法
    int count = count_if(data.begin(), data.end(), 
                        [](int x) { return x > 5; });
    
    // 修改序列算法
    vector<int> result;
    copy_if(data.begin(), data.end(), back_inserter(result),
            [](int x) { return x % 2 == 0; });
    
    // 排序相关算法
    partial_sort(data.begin(), data.begin() + 3, data.end());
    
    // 数值算法
    int sum = accumulate(data.begin(), data.end(), 0);
    
    // 自定义迭代器
    class StepIterator {
        int current;
        int step;
    public:
        // 迭代器接口实现...
    };
}
```

## 4. Lambda表达式

### 语法深度解析
```cpp
// Lambda表达式完整语法
[捕获列表] (参数列表) mutable noexcept -> 返回类型 { 函数体 }

// 实际应用示例
void lambdaAdvanced() {
    vector<int> numbers = {1, 2, 3, 4, 5};
    
    // 值捕获与引用捕获
    int threshold = 3;
    auto countAbove = [threshold](int x) { return x > threshold; };
    
    // mutable修改捕获值
    int counter = 0;
    auto incrementer = [counter]() mutable { return ++counter; };
    
    // 泛型Lambda (C++14)
    auto genericAdder = [](auto a, auto b) { return a + b; };
    
    // Lambda作为返回值
    auto createMultiplier = [](int factor) {
        return [factor](int x) { return x * factor; };
    };
    
    auto doubleIt = createMultiplier(2);
    cout << doubleIt(5) << endl;  // 输出10
}
```

### 函数式编程模式
```cpp
// 组合函数
template<typename F, typename G>
auto compose(F f, G g) {
    return [f, g](auto x) { return f(g(x)); };
}

// 柯里化
auto curryAdd = [](int a) {
    return [a](int b) { return a + b; };
};

void functionalPatterns() {
    // 函数组合
    auto square = [](int x) { return x * x; };
    auto increment = [](int x) { return x + 1; };
    auto squareThenIncrement = compose(increment, square);
    
    // 柯里化应用
    auto add5 = curryAdd(5);
    cout << add5(3) << endl;  // 输出8
}
```

## 5. 右值引用与移动语义

### 移动语义深度解析
```cpp
class AdvancedString {
    char* data;
    size_t length;
    
public:
    // 移动构造函数
    AdvancedString(AdvancedString&& other) noexcept 
        : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
    }
    
    // 移动赋值运算符
    AdvancedString& operator=(AdvancedString&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
    
    // 完美转发构造函数
    template<typename T>
    AdvancedString(T&& str) {
        // 通用引用处理左值/右值
    }
};

void advancedMoveSemantics() {
    // 返回值优化 (RVO)
    auto createString = []() -> AdvancedString {
        return AdvancedString("临时字符串");  // 可能触发RVO
    };
    
    AdvancedString s1 = createString();  // 移动构造或RVO
    
    // 移动语义在容器中的应用
    vector<AdvancedString> strings;
    strings.reserve(10);  // 预留空间避免重新分配
    
    AdvancedString temp("可移动对象");
    strings.push_back(move(temp));  // 移动而非拷贝
    
    // emplace_back直接构造
    strings.emplace_back("直接构造");  // 避免临时对象
}
```

### 完美转发实践
```cpp
template<typename T>
class Wrapper {
    T value;
    
public:
    // 通用引用构造函数
    template<typename U>
    Wrapper(U&& u) : value(forward<U>(u)) {}
    
    // 完美转发到成员函数
    template<typename... Args>
    void emplace(Args&&... args) {
        value = T(forward<Args>(args)...);
    }
};

void perfectForwardingDemo() {
    string name = "测试";
    
    // 左值转发
    Wrapper<string> w1(name);  // 拷贝构造
    
    // 右值转发  
    Wrapper<string> w2(string("临时"));  // 移动构造
    
    // 字面量转发
    Wrapper<string> w3("字面量");  // 移动构造
    
    // 可变参数转发
    Wrapper<vector<int>> w4;
    w4.emplace(1, 2, 3, 4, 5);  // 直接构造vector
}
```

## 6. 多线程编程核心

### 线程同步机制
```cpp
class ThreadSafeData {
private:
    mutable mutex mtx;
    condition_variable cond;
    queue<int> data;
    bool stopped = false;
    
public:
    void push(int value) {
        lock_guard<mutex> lock(mtx);
        data.push(value);
        cond.notify_one();
    }
    
    optional<int> pop() {
        unique_lock<mutex> lock(mtx);
        cond.wait(lock, [this] { 
            return !data.empty() || stopped; 
        });
        
        if (data.empty()) return nullopt;
        
        int value = data.front();
        data.pop();
        return value;
    }
    
    void stop() {
        lock_guard<mutex> lock(mtx);
        stopped = true;
        cond.notify_all();
    }
};

void advancedThreading() {
    ThreadSafeData tsd;
    
    // 生产者线程
    vector<thread> producers;
    for (int i = 0; i < 3; ++i) {
        producers.emplace_back([&tsd, i]() {
            for (int j = 0; j < 5; ++j) {
                tsd.push(i * 10 + j);
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        });
    }
    
    // 消费者线程
    vector<thread> consumers;
    for (int i = 0; i < 2; ++i) {
        consumers.emplace_back([&tsd, i]() {
            while (auto value = tsd.pop()) {
                cout << "消费者" << i << "处理: " << *value << endl;
            }
        });
    }
    
    // 等待生产完成
    for (auto& t : producers) t.join();
    
    // 停止消费者
    tsd.stop();
    for (auto& t : consumers) t.join();
}
```

### 异步编程模式
```cpp
class AsyncProcessor {
public:
    template<typename Func, typename... Args>
    auto asyncExecute(Func&& func, Args&&... args) {
        return async(launch::async, forward<Func>(func), forward<Args>(args)...);
    }
    
    // 批量异步执行
    template<typename Func, typename Container>
    auto asyncBatch(Container&& items, Func&& func) {
        vector<future<void>> futures;
        
        for (auto&& item : items) {
            futures.push_back(asyncExecute(func, forward<decltype(item)>(item)));
        }
        
        return futures;
    }
};

void asyncPatterns() {
    AsyncProcessor processor;
    
    vector<int> data = {1, 2, 3, 4, 5};
    
    // 批量异步处理
    auto futures = processor.asyncBatch(data, [](int x) {
        this_thread::sleep_for(chrono::milliseconds(100));
        cout << "处理: " << x * x << endl;
    });
    
    // 等待所有任务完成
    for (auto& f : futures) {
        f.get();
    }
}
```

## 7. 智能指针与内存管理

### 智能指针深度应用
```cpp
class ResourceManager {
private:
    shared_ptr<Resource> primary;
    weak_ptr<Resource> backup;
    
public:
    void setPrimary(shared_ptr<Resource> res) {
        primary = move(res);
    }
    
    void setBackup(shared_ptr<Resource> res) {
        backup = res;  // 弱引用，不增加计数
    }
    
    shared_ptr<Resource> getPrimary() {
        return primary;
    }
    
    shared_ptr<Resource> getBackup() {
        return backup.lock();  // 尝试获取强引用
    }
    
    // 自定义删除器工厂
    template<typename T>
    static shared_ptr<T> createWithLogger(const string& name) {
        return shared_ptr<T>(new T(name), [](T* ptr) {
            cout << "销毁: " << ptr->getName() << endl;
            delete ptr;
        });
    }
};

void advancedMemoryManagement() {
    // 循环引用问题
    class Node {
    public:
        shared_ptr<Node> next;
        weak_ptr<Node> prev;  // 使用weak_ptr打破循环引用
        
        ~Node() { cout << "节点销毁" << endl; }
    };
    
    auto node1 = make_shared<Node>();
    auto node2 = make_shared<Node>();
    
    node1->next = node2;
    node2->prev = node1;  // 弱引用，不会形成循环
    
    // 自定义内存分配器
    class PoolAllocator {
        // 内存池实现...
    };
    
    // 使用分配器的智能指针
    using PoolString = basic_string<char, char_traits<char>, PoolAllocator<char>>;
    // shared_ptr<PoolString> str = allocate_shared<PoolString>(PoolAllocator<char>());
}
```

### RAII模式扩展
```cpp
// 通用RAII包装器
template<typename Resource, typename Deleter>
class RAIIWrapper {
    Resource resource;
    Deleter deleter;
    
public:
    template<typename... Args>
    RAIIWrapper(Deleter d, Args&&... args) 
        : resource(forward<Args>(args)...), deleter(move(d)) {}
    
    ~RAIIWrapper() {
        deleter(resource);
    }
    
    Resource& get() { return resource; }
    const Resource& get() const { return resource; }
};

void raiiPatterns() {
    // 文件RAII
    auto fileDeleter = [](FILE* f) { if (f) fclose(f); };
    RAIIWrapper<FILE*, decltype(fileDeleter)> 
        file(fileDeleter, fopen("test.txt", "w"));
    
    if (file.get()) {
        fprintf(file.get(), "RAII模式测试");
    }
    
    // 互斥锁RAII（类似lock_guard）
    mutex mtx;
    auto lockDeleter = [&mtx](bool*) { mtx.unlock(); };
    {
        mtx.lock();
        RAIIWrapper<bool*, decltype(lockDeleter)> 
            lock(lockDeleter, new bool(true));
        
        // 临界区代码
        cout << "在锁保护下执行" << endl;
        
        // lock析构时自动解锁
    }
}
```