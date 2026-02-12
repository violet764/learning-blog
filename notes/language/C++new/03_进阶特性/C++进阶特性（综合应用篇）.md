# C++进阶特性（综合应用篇）

## 1. 设计模式与现代C++

### 工厂模式与智能指针
```cpp
class Product {
public:
    virtual ~Product() = default;
    virtual void use() = 0;
};

class ConcreteProductA : public Product {
public:
    void use() override { cout << "使用产品A" << endl; }
};

class ConcreteProductB : public Product {
public:
    void use() override { cout << "使用产品B" << endl; }
};

class ProductFactory {
public:
    enum class Type { A, B };
    
    static unique_ptr<Product> create(Type type) {
        switch (type) {
            case Type::A: return make_unique<ConcreteProductA>();
            case Type::B: return make_unique<ConcreteProductB>();
            default: return nullptr;
        }
    }
    
    // 模板工厂方法
    template<typename T, typename... Args>
    static unique_ptr<Product> create(Args&&... args) {
        return make_unique<T>(forward<Args>(args)...);
    }
};

void factoryPatternDemo() {
    auto productA = ProductFactory::create(ProductFactory::Type::A);
    auto productB = ProductFactory::create<ConcreteProductB>();
    
    productA->use();
    productB->use();
}
```

### 观察者模式与函数对象
```cpp
template<typename T>
class Observable {
    vector<function<void(const T&)>> observers;
    
public:
    void subscribe(function<void(const T&)> observer) {
        observers.push_back(move(observer));
    }
    
    void notify(const T& data) {
        for (auto& observer : observers) {
            observer(data);
        }
    }
};

class DataModel {
    Observable<string> observable;
    string data;
    
public:
    void setData(const string& newData) {
        data = newData;
        observable.notify(data);
    }
    
    void subscribe(function<void(const string&)> observer) {
        observable.subscribe(move(observer));
    }
};

void observerPatternDemo() {
    DataModel model;
    
    // Lambda观察者
    model.subscribe([](const string& data) {
        cout << "观察者1收到: " << data << endl;
    });
    
    model.subscribe([](const string& data) {
        cout << "观察者2收到: " << data << endl;
    });
    
    model.setData("新数据");
}
```

## 2. 性能优化实践

### 移动语义优化
```cpp
class OptimizedVector {
private:
    vector<int> data;
    
public:
    // 完美转发添加元素
    template<typename T>
    void add(T&& value) {
        data.emplace_back(forward<T>(value));
    }
    
    // 批量添加（移动优化）
    void addRange(vector<int>&& values) {
        if (data.empty()) {
            data = move(values);  // 直接移动
        } else {
            data.insert(data.end(), 
                       make_move_iterator(values.begin()), 
                       make_move_iterator(values.end()));
        }
    }
    
    // 返回移动优化
    vector<int> extract() && {
        return move(data);  // 右值引用版本
    }
    
    const vector<int>& get() const & {
        return data;  // 左值引用版本
    }
};

void optimizationDemo() {
    OptimizedVector ov;
    
    // 左值添加（拷贝）
    int x = 42;
    ov.add(x);
    
    // 右值添加（移动）
    ov.add(123);
    ov.add(string("test"));  // 需要转换构造函数
    
    // 批量移动
    vector<int> temp = {1, 2, 3, 4, 5};
    ov.addRange(move(temp));
    
    // 提取数据（移动）
    auto extracted = move(ov).extract();
}
```

### 缓存与内存池
```cpp
template<typename T>
class ObjectPool {
private:
    queue<unique_ptr<T>> pool;
    mutex mtx;
    
public:
    template<typename... Args>
    unique_ptr<T> acquire(Args&&... args) {
        lock_guard<mutex> lock(mtx);
        
        if (!pool.empty()) {
            auto obj = move(pool.front());
            pool.pop();
            return obj;
        }
        
        return make_unique<T>(forward<Args>(args)...);
    }
    
    void release(unique_ptr<T> obj) {
        lock_guard<mutex> lock(mtx);
        if (obj) {
            pool.push(move(obj));
        }
    }
    
    size_t size() const {
        lock_guard<mutex> lock(mtx);
        return pool.size();
    }
};

class ExpensiveObject {
    vector<int> data;
public:
    ExpensiveObject(size_t size = 1000) : data(size) {}
    void reset() { fill(data.begin(), data.end(), 0); }
};

void poolDemo() {
    ObjectPool<ExpensiveObject> pool;
    
    // 从池中获取对象
    auto obj1 = pool.acquire(1000);
    auto obj2 = pool.acquire(2000);
    
    // 使用后放回池中
    obj1->reset();
    pool.release(move(obj1));
    
    cout << "池中对象数量: " << pool.size() << endl;
}
```

## 3. 并发编程实战

### 线程安全数据结构
```cpp
template<typename K, typename V>
class ThreadSafeMap {
private:
    mutable shared_mutex mtx;  // C++17读写锁
    unordered_map<K, V> data;
    
public:
    // 读操作（共享锁）
    optional<V> get(const K& key) const {
        shared_lock<shared_mutex> lock(mtx);
        auto it = data.find(key);
        return it != data.end() ? make_optional(it->second) : nullopt;
    }
    
    // 写操作（独占锁）
    void set(const K& key, V value) {
        unique_lock<shared_mutex> lock(mtx);
        data[key] = move(value);
    }
    
    // 批量操作
    template<typename Func>
    void forEach(Func&& func) const {
        shared_lock<shared_mutex> lock(mtx);
        for (const auto& [key, value] : data) {
            func(key, value);
        }
    }
    
    // 原子更新
    bool updateIfExists(const K& key, function<void(V&)> updater) {
        unique_lock<shared_mutex> lock(mtx);
        auto it = data.find(key);
        if (it != data.end()) {
            updater(it->second);
            return true;
        }
        return false;
    }
};

void concurrentMapDemo() {
    ThreadSafeMap<string, int> scores;
    
    vector<thread> writers;
    for (int i = 0; i < 3; ++i) {
        writers.emplace_back([&scores, i]() {
            for (int j = 0; j < 100; ++j) {
                scores.set("player" + to_string(j), i * 100 + j);
            }
        });
    }
    
    vector<thread> readers;
    for (int i = 0; i < 2; ++i) {
        readers.emplace_back([&scores, i]() {
            for (int j = 0; j < 50; ++j) {
                if (auto score = scores.get("player" + to_string(j))) {
                    cout << "读者" << i << "读到: " << *score << endl;
                }
            }
        });
    }
    
    for (auto& t : writers) t.join();
    for (auto& t : readers) t.join();
}
```

### 异步任务编排
```cpp
class TaskScheduler {
    vector<future<void>> tasks;
    
public:
    template<typename Func>
    void submit(Func&& func) {
        tasks.push_back(async(launch::async, forward<Func>(func)));
    }
    
    void waitAll() {
        for (auto& task : tasks) {
            task.get();
        }
        tasks.clear();
    }
    
    // 链式任务
    template<typename Func>
    auto then(Func&& func) -> future<invoke_result_t<Func>> {
        return async(launch::async, [func = forward<Func>(func)]() {
            return func();
        });
    }
};

void taskOrchestration() {
    TaskScheduler scheduler;
    
    // 提交多个任务
    scheduler.submit([]() {
        this_thread::sleep_for(chrono::milliseconds(100));
        cout << "任务1完成" << endl;
    });
    
    scheduler.submit([]() {
        this_thread::sleep_for(chrono::milliseconds(200));
        cout << "任务2完成" << endl;
    });
    
    // 等待所有任务
    scheduler.waitAll();
    
    // 链式执行
    auto future = scheduler.then([]() -> int {
        return 42;
    }).then([](int value) -> string {
        return "结果: " + to_string(value);
    });
    
    cout << future.get() << endl;
}
```

## 4. 模板元编程应用

### 类型特征与SFINAE
```cpp
// 类型特征检查
template<typename T>
struct is_container {
private:
    template<typename U>
    static auto test(int) -> decltype(
        begin(declval<U>()), end(declval<U>()), true_type{});
    
    template<typename>
    static false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
constexpr bool is_container_v = is_container<T>::value;

// SFINAE应用
template<typename T>
enable_if_t<is_container_v<T>, void> printContainer(const T& container) {
    for (const auto& item : container) {
        cout << item << " ";
    }
    cout << endl;
}

template<typename T>
enable_if_t<!is_container_v<T>, void> printContainer(const T& value) {
    cout << "非容器: " << value << endl;
}

void templateDemo() {
    vector<int> vec = {1, 2, 3};
    printContainer(vec);  // 调用容器版本
    
    int x = 42;
    printContainer(x);    // 调用非容器版本
}
```

### 编译时计算
```cpp
// 编译时字符串哈希
constexpr size_t hashString(const char* str, size_t len) {
    size_t hash = 5381;
    for (size_t i = 0; i < len; ++i) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

// 字符串字面量操作器
template<size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) {
        copy_n(str, N, value);
    }
    
    char value[N];
    static constexpr size_t length = N - 1;
    
    constexpr size_t hash() const {
        return hashString(value, length);
    }
};

// 编译时映射
template<StringLiteral Key, typename Value>
struct KeyValuePair {
    static constexpr auto key = Key;
    using value_type = Value;
};

template<typename... Pairs>
struct ConstexprMap {};

void constexprDemo() {
    constexpr StringLiteral key1("hello");
    constexpr StringLiteral key2("world");
    
    using MyMap = ConstexprMap<
        KeyValuePair<key1, int>,
        KeyValuePair<key2, string>
    >;
    
    constexpr auto hash1 = key1.hash();
    constexpr auto hash2 = key2.hash();
    
    cout << "编译时哈希: " << hash1 << ", " << hash2 << endl;
}
```

## 5. 错误处理与异常安全

### 异常安全保证
```cpp
class Transaction {
    vector<function<void()>> rollbackActions;
    
public:
    template<typename Action, typename Rollback>
    void execute(Action&& action, Rollback&& rollback) {
        action();
        rollbackActions.emplace_back(forward<Rollback>(rollback));
    }
    
    void commit() {
        rollbackActions.clear();
    }
    
    ~Transaction() {
        if (!rollbackActions.empty()) {
            // 发生异常，执行回滚
            for (auto it = rollbackActions.rbegin(); it != rollbackActions.rend(); ++it) {
                try {
                    (*it)();
                } catch (...) {
                    // 记录日志，继续回滚其他操作
                }
            }
        }
    }
};

void exceptionSafeDemo() {
    vector<int> data = {1, 2, 3};
    
    try {
        Transaction trans;
        
        // 操作1：添加元素
        size_t oldSize = data.size();
        trans.execute(
            [&]() { data.push_back(4); },
            [&]() { data.resize(oldSize); }
        );
        
        // 操作2：可能抛出异常的操作
        trans.execute(
            [&]() { 
                if (data.size() > 10) throw runtime_error("太大");
                data.push_back(5); 
            },
            [&]() { data.pop_back(); }
        );
        
        trans.commit();  // 所有操作成功，提交
        
    } catch (const exception& e) {
        cout << "事务失败: " << e.what() << endl;
        // data自动回滚到初始状态
    }
    
    cout << "最终数据大小: " << data.size() << endl;
}
```

### 资源管理包装器
```cpp
// 通用资源管理器
template<typename Resource, typename Deleter>
class ScopedResource {
    Resource resource;
    Deleter deleter;
    bool owned = true;
    
public:
    template<typename... Args>
    ScopedResource(Deleter d, Args&&... args) 
        : resource(forward<Args>(args)...), deleter(move(d)) {}
    
    ~ScopedResource() {
        if (owned) {
            deleter(resource);
        }
    }
    
    // 禁止拷贝
    ScopedResource(const ScopedResource&) = delete;
    ScopedResource& operator=(const ScopedResource&) = delete;
    
    // 允许移动
    ScopedResource(ScopedResource&& other) 
        : resource(move(other.resource)), deleter(move(other.deleter)) {
        other.owned = false;
    }
    
    ScopedResource& operator=(ScopedResource&& other) {
        if (this != &other) {
            if (owned) deleter(resource);
            resource = move(other.resource);
            deleter = move(other.deleter);
            owned = true;
            other.owned = false;
        }
        return *this;
    }
    
    Resource& get() { return resource; }
    const Resource& get() const { return resource; }
    
    Resource release() {
        owned = false;
        return move(resource);
    }
};

void resourceManagementDemo() {
    // 文件资源
    auto fileDeleter = [](FILE* f) { 
        if (f) {
            fclose(f);
            cout << "文件已关闭" << endl;
        }
    };
    
    ScopedResource<FILE*, decltype(fileDeleter)> 
        file(fileDeleter, fopen("test.txt", "w"));
    
    if (file.get()) {
        fprintf(file.get(), "资源管理测试");
    }
    
    // 动态数组资源
    auto arrayDeleter = [](int* arr) { 
        delete[] arr; 
        cout << "数组已释放" << endl;
    };
    
    ScopedResource<int*, decltype(arrayDeleter)> 
        arr(arrayDeleter, new int[100]);
    
    // 数组使用...
    
    // 资源自动释放
}
```

## 6. 现代C++工程实践

### 模块化设计
```cpp
// 接口定义
class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const string& message) = 0;
    virtual void error(const string& message) = 0;
};

// 具体实现
class FileLogger : public ILogger {
    ofstream file;
public:
    FileLogger(const string& filename) : file(filename) {}
    
    void log(const string& message) override {
        file << "[INFO] " << message << endl;
    }
    
    void error(const string& message) override {
        file << "[ERROR] " << message << endl;
    }
};

// 工厂函数
unique_ptr<ILogger> createLogger(const string& type, const string& param) {
    if (type == "file") {
        return make_unique<FileLogger>(param);
    }
    // 其他类型...
    return nullptr;
}

// 依赖注入
class Application {
    unique_ptr<ILogger> logger;
    
public:
    Application(unique_ptr<ILogger> log) : logger(move(log)) {}
    
    void run() {
        logger->log("应用启动");
        // 业务逻辑...
        logger->log("应用结束");
    }
};

void modularDesignDemo() {
    auto logger = createLogger("file", "app.log");
    Application app(move(logger));
    app.run();
}
```

### 测试与调试支持
```cpp
// 编译时断言
template<typename T>
class Stack {
    static_assert(is_default_constructible_v<T>, 
                  "T必须可默认构造");
    // 实现...
};

// 调试工具
class DebugHelper {
public:
    template<typename T>
    static void printType() {
        cout << "类型: " << typeid(T).name() << endl;
    }
    
    template<typename Container>
    static void printContainerInfo(const Container& c) {
        cout << "大小: " << c.size() 
             << ", 容量: " << c.capacity() << endl;
    }
    
    // 性能测量
    template<typename Func>
    static auto measureTime(Func&& func) {
        auto start = chrono::high_resolution_clock::now();
        auto result = func();
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "执行时间: " << duration.count() << "微秒" << endl;
        
        return result;
    }
};

void debuggingDemo() {
    vector<int> data(1000);
    iota(data.begin(), data.end(), 1);
    
    DebugHelper::printContainerInfo(data);
    DebugHelper::printType<decltype(data)>();
    
    auto result = DebugHelper::measureTime([&data]() {
        return accumulate(data.begin(), data.end(), 0);
    });
    
    cout << "结果: " << result << endl;
}
```

这个综合应用篇涵盖了C++进阶特性的实际应用场景，包括设计模式、性能优化、并发编程、模板元编程、错误处理等高级主题，展示了如何将核心机制应用到实际工程中。