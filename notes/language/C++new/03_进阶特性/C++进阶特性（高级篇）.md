# C++进阶特性（高级篇）

## 第一部分：右值引用与移动语义

### 核心概念
- **右值引用（&&）**：绑定到临时对象的引用类型
- **移动语义**：高效转移资源所有权，避免不必要拷贝
- **完美转发**：保持参数值类别进行转发

### 移动构造函数和赋值运算符
```cpp
class MyString {
private:
    char* data;
    size_t length;
    
public:
    // 移动构造函数
    MyString(MyString&& other) noexcept 
        : data(other.data), length(other.length) {
        other.data = nullptr;  // 转移所有权
        other.length = 0;
    }
    
    // 移动赋值运算符
    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] data;      // 释放现有资源
            data = other.data;  // 接管新资源
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
};
```

### std::move和std::forward
```cpp
#include <utility>

void moveDemo() {
    MyString str1("Hello");
    
    // 显式移动
    MyString str2 = std::move(str1);  // 调用移动构造函数
    
    // 移动后原对象处于有效但未定义状态
}

// 完美转发模板
template<typename T>
void forwardDemo(T&& arg) {
    // std::forward保持参数的值类别
    process(std::forward<T>(arg));
}
```

## 第二部分：多线程编程

### 线程创建与管理
```cpp
#include <thread>
#include <iostream>

void threadFunction(int id) {
    std::cout << "线程 " << id << " 运行中" << std::endl;
}

void basicThreading() {
    std::thread t1(threadFunction, 1);
    std::thread t2(threadFunction, 2);
    
    t1.join();  // 等待线程完成
    t2.join();
}
```

### 同步机制：互斥锁
```cpp
#include <mutex>
#include <vector>

class ThreadSafeCounter {
private:
    std::mutex mtx;
    int value = 0;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);  // RAII锁
        ++value;
    }
    
    int get() const {
        std::lock_guard<std::mutex> lock(mtx);
        return value;
    }
};
```

### 条件变量与生产者-消费者模式
```cpp
#include <condition_variable>
#include <queue>

template<typename T>
class ThreadSafeQueue {
private:
    std::mutex mtx;
    std::queue<T> data;
    std::condition_variable cond;
    
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx);
        data.push(std::move(value));
        cond.notify_one();  // 通知等待的消费者
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        
        // 等待队列不为空
        cond.wait(lock, [this] { return !data.empty(); });
        
        T value = std::move(data.front());
        data.pop();
        return value;
    }
};
```

### 异步编程
```cpp
#include <future>
#include <chrono>

int heavyComputation() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 42;
}

void asyncDemo() {
    // 异步执行计算
    std::future<int> result = std::async(std::launch::async, heavyComputation);
    
    // 主线程继续其他工作
    std::cout << "等待计算结果..." << std::endl;
    
    // 获取结果（阻塞直到完成）
    int value = result.get();
    std::cout << "结果: " << value << std::endl;
}
```

## 第三部分：内存管理

### 智能指针
```cpp
#include <memory>

class Resource {
public:
    Resource() { std::cout << "资源创建" << std::endl; }
    ~Resource() { std::cout << "资源释放" << std::endl; }
    
    void use() { std::cout << "使用资源" << std::endl; }
};

void smartPointerDemo() {
    // unique_ptr：独占所有权
    auto unique = std::make_unique<Resource>();
    unique->use();
    
    // shared_ptr：共享所有权
    auto shared1 = std::make_shared<Resource>();
    auto shared2 = shared1;  // 共享所有权
    
    std::cout << "引用计数: " << shared1.use_count() << std::endl;
    
    // weak_ptr：观察但不拥有
    std::weak_ptr<Resource> weak = shared1;
    
    if (auto locked = weak.lock()) {
        locked->use();  // 资源存在时使用
    }
}
```

### RAII模式
```cpp
class FileHandle {
private:
    FILE* file;
    
public:
    FileHandle(const char* filename, const char* mode) 
        : file(fopen(filename, mode)) {
        if (!file) throw std::runtime_error("文件打开失败");
    }
    
    ~FileHandle() {
        if (file) fclose(file);
    }
    
    // 禁用拷贝（避免重复释放）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    
    // 允许移动
    FileHandle(FileHandle&& other) noexcept : file(other.file) {
        other.file = nullptr;
    }
    
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (file) fclose(file);
            file = other.file;
            other.file = nullptr;
        }
        return *this;
    }
    
    void write(const std::string& content) {
        if (file) fwrite(content.c_str(), 1, content.size(), file);
    }
};
```

### 自定义内存管理
```cpp
#include <new>

class MemoryPool {
private:
    struct Block {
        Block* next;
    };
    
    Block* freeList = nullptr;
    size_t blockSize;
    
public:
    MemoryPool(size_t size) : blockSize(size) {}
    
    void* allocate() {
        if (!freeList) {
            // 分配新块
            return ::operator new(blockSize);
        }
        
        void* block = freeList;
        freeList = freeList->next;
        return block;
    }
    
    void deallocate(void* block) {
        if (!block) return;
        
        Block* newBlock = static_cast<Block*>(block);
        newBlock->next = freeList;
        freeList = newBlock;
    }
    
    ~MemoryPool() {
        while (freeList) {
            Block* next = freeList->next;
            ::operator delete(freeList);
            freeList = next;
        }
    }
};
```

## 综合应用：高性能数据处理系统

```cpp
#include <thread>
#include <memory>
#include <vector>
#include <queue>
#include <future>
#include <algorithm>

class DataProcessor {
private:
    std::vector<std::thread> workers;
    std::queue<std::vector<int>> dataQueue;
    std::mutex queueMutex;
    std::condition_variable queueCond;
    bool stop = false;
    
public:
    DataProcessor(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] { workerThread(); });
        }
    }
    
    ~DataProcessor() {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            stop = true;
        }
        queueCond.notify_all();
        
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    void addData(std::vector<int> data) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            dataQueue.push(std::move(data));
        }
        queueCond.notify_one();
    }
    
private:
    void workerThread() {
        while (true) {
            std::vector<int> data;
            
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCond.wait(lock, [this] { 
                    return stop || !dataQueue.empty(); 
                });
                
                if (stop && dataQueue.empty()) return;
                
                data = std::move(dataQueue.front());
                dataQueue.pop();
            }
            
            // 处理数据（使用移动语义避免拷贝）
            processData(std::move(data));
        }
    }
    
    void processData(std::vector<int> data) {
        // 模拟数据处理
        std::sort(data.begin(), data.end());
        
        // 使用智能指针管理结果
        auto result = std::make_shared<std::vector<int>>(std::move(data));
        
        // 异步处理结果
        std::async(std::launch::async, [result] {
            std::cout << "处理完成，数据大小: " << result->size() << std::endl;
        });
    }
};

int main() {
    DataProcessor processor(4);  // 4个处理线程
    
    // 添加数据处理任务
    for (int i = 0; i < 10; ++i) {
        processor.addData({5, 2, 8, 1, 9, i});
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    return 0;
}
```

## 高级特性最佳实践

1. **移动语义**：对大对象使用移动而非拷贝
2. **线程安全**：使用RAII管理锁，避免死锁
3. **内存管理**：优先使用智能指针，避免手动管理
4. **异常安全**：确保异常发生时资源正确释放
5. **性能优化**：理解各特性的性能影响，合理选择