# C++异常处理完整学习笔记

## 📚 概述

C++异常处理机制是现代C++编程中重要的错误处理方式，它通过**栈展开(stack unwinding)** 和**RAII(资源获取即初始化)** 相结合，提供了比传统错误码更安全、更优雅的错误处理方案。

---

## 🔄 异常处理基本流程

### 异常处理三要素
1. **throw** - 抛出异常
2. **try** - 尝试执行可能抛出异常的代码块
3. **catch** - 捕获并处理异常

### 基本语法结构
```cpp
#include <iostream>
#include <stdexcept>

void riskyOperation(int value) {
    if (value < 0) {
        throw std::invalid_argument("值不能为负数");
    }
    if (value > 100) {
        throw std::out_of_range("值超出范围");
    }
    std::cout << "操作成功，值: " << value << std::endl;
}

int main() {
    try {
        riskyOperation(-5);
    } catch (const std::invalid_argument& e) {
        std::cerr << "参数错误: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "范围错误: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "未知错误: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "未知异常" << std::endl;
    }
    return 0;
}
```

---

## 🏗️ 标准异常类层次结构

### 异常类继承关系
```
std::exception
├── std::logic_error           // 逻辑错误
│   ├── std::invalid_argument  // 无效参数
│   ├── std::out_of_range      // 范围越界
│   ├── std::length_error      // 长度错误
│   └── std::domain_error      // 定义域错误
├── std::runtime_error         // 运行时错误
│   ├── std::overflow_error    // 溢出错误
│   ├── std::underflow_error   // 下溢错误
│   ├── std::range_error       // 范围错误
│   └── std::system_error      // 系统错误
├── std::bad_alloc             // 内存分配失败
├── std::bad_cast              // 类型转换失败
└── std::bad_typeid            // typeid操作失败
```

### 常用标准异常类
- **`std::invalid_argument`** - 函数接收到无效参数
- **`std::out_of_range`** - 访问超出有效范围
- **`std::runtime_error`** - 运行时检测到的错误
- **`std::bad_alloc`** - 内存分配失败

---

## 🛡️ 异常安全保证级别

### 1. 基本保证 (Basic Guarantee)
- **不泄漏资源**，对象处于有效状态
- 即使发生异常，资源也能正确释放

```cpp
class ResourceManager {
    std::unique_ptr<int> resource;
public:
    void basicGuarantee(int value) {
        auto temp = std::make_unique<int>(value);
        
        // 可能抛出异常的操作
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        
        // 提交操作（无异常）
        resource = std::move(temp);
    }
};
```

### 2. 强保证 (Strong Guarantee)
- **操作具有原子性** - 要么完全成功，要么完全失败
- 实现事务性操作

```cpp
void strongGuarantee(int value) {
    auto oldResource = std::move(resource);
    
    try {
        auto temp = std::make_unique<int>(value);
        
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        
        resource = std::move(temp);
    } catch (...) {
        // 回滚操作
        resource = std::move(oldResource);
        throw; // 重新抛出异常
    }
}
```

### 3. 不抛出保证 (No-throw Guarantee)
- **函数承诺不会抛出异常**
- 使用 `noexcept` 关键字声明

```cpp
int noThrowGuarantee() noexcept {
    return resource ? *resource : 0;
}
```

---

## 🔄 RAII与异常安全

### RAII原则 (Resource Acquisition Is Initialization)
- **资源获取即初始化**
- 利用对象生命周期管理资源
- 确保异常发生时资源能正确释放

```cpp
#include <fstream>
#include <memory>

class FileHandler {
private:
    std::unique_ptr<std::fstream> file;
    
public:
    FileHandler(const std::string& filename) {
        file = std::make_unique<std::fstream>(filename);
        if (!file->is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }
    }
    
    // 自动关闭文件（RAII）
    ~FileHandler() = default;
    
    void write(const std::string& data) {
        *file << data;
        if (file->fail()) {
            throw std::runtime_error("写入文件失败");
        }
    }
};

// 使用RAII确保资源安全
void safeFileOperation() {
    FileHandler handler("data.txt"); // 资源自动管理
    handler.write("Hello, RAII!");
    // 文件自动关闭，即使抛出异常
}
```

---

## 🎯 自定义异常类

### 创建自定义异常
```cpp
#include <exception>
#include <string>

class MyException : public std::exception {
private:
    std::string message;
    int errorCode;
    
public:
    MyException(const std::string& msg, int code = 0) 
        : message(msg), errorCode(code) {}
    
    // 重写what()方法
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    int getErrorCode() const {
        return errorCode;
    }
};

// 使用自定义异常
void processData(int data) {
    if (data < 0) {
        throw MyException("数据不能为负数", 1001);
    }
    // 正常处理...
}

// 捕获自定义异常
try {
    processData(-5);
} catch (const MyException& e) {
    std::cerr << "错误代码: " << e.getErrorCode() 
              << ", 消息: " << e.what() << std::endl;
}
```

---

## ⚡ 异常性能考虑

### 性能特点
- **正常执行路径**：零开销（现代编译器优化）
- **抛出异常时**：有运行时开销（栈展开、类型匹配）
- **适用于罕见错误**，不应用于流程控制

### 栈展开过程
```cpp
double safe_divide(double a, double b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
        // 1. 构造异常对象
        // 2. 开始栈展开：从当前函数向上
        // 3. 析构局部对象（RAII确保资源释放）
        // 4. 查找匹配的catch块
    }
    return a / b;
}
```

---

## 📋 异常处理最佳实践

### 1. 构造函数中的异常处理
```cpp
class Application {
private:
    std::unique_ptr<Database> db;
    
public:
    // 使用函数try块处理构造函数异常
    Application() try : db(std::make_unique<Database>()) {
        db->connect();
    } catch (const std::exception& e) {
        std::cerr << "应用初始化失败: " << e.what() << std::endl;
        throw; // 重新抛出
    }
};
```

### 2. 析构函数中的异常处理
```cpp
~Application() noexcept {
    try {
        // 清理资源
        if (db) {
            // 数据库断开连接等
        }
    } catch (...) {
        // 记录日志，但不抛出异常
        std::cerr << "析构函数中发生异常，已忽略" << std::endl;
    }
}
```

### 3. 智能指针管理资源
```cpp
void processData() {
    auto data = std::make_unique<std::vector<int>>();
    data->push_back(1);
    data->push_back(2);
    
    // 即使这里抛出异常，data也会自动释放
    if (data->size() > 10) {
        throw std::runtime_error("数据量过大");
    }
}
```

### 4. 异常层次化处理
```cpp
void handleComplexOperation() {
    try {
        Application app;
        app.processData();
    } catch (const std::runtime_error& e) {
        std::cerr << "运行时错误: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "未知异常" << std::endl;
    }
}
```

---

## 🆕 noexcept关键字

### noexcept说明符
```cpp
// 不抛出异常的函数
void no_throw_function() noexcept {
    // 这个函数保证不抛出异常
}

// 条件性noexcept
template<typename T>
void swap(T &a, T &b) noexcept(noexcept(a.swap(b))) {
    a.swap(b);
}

// 移动构造函数通常标记为noexcept
class Movable {
public:
    Movable(Movable &&other) noexcept {
        // 移动资源...
    }
};
```

### noexcept的优势
- **编译器优化**：编译器知道函数不会抛出异常，可以进行更多优化
- **代码清晰**：明确表达函数的行为意图
- **性能提升**：避免异常处理的开销

---

## 🔧 实用工具和技巧

### 1. 异常重新抛出
```cpp
try {
    // 可能抛出异常的代码
} catch (const std::exception& e) {
    // 记录日志
    std::cerr << "错误: " << e.what() << std::endl;
    
    // 重新抛出异常，让上层处理
    throw;
}
```

### 2. 嵌套异常处理
```cpp
try {
    try {
        // 内部操作
    } catch (const std::exception& inner) {
        // 处理内部异常，然后抛出外层异常
        throw std::runtime_error("外层操作失败: " + std::string(inner.what()));
    }
} catch (const std::exception& outer) {
    // 处理外层异常
    std::cerr << outer.what() << std::endl;
}
```

### 3. 异常安全的自定义容器
```cpp
template<typename T>
class SafeVector {
    std::vector<T> data;
    
public:
    void push_back(const T& value) {
        // 强异常保证实现
        std::vector<T> new_data = data;
        new_data.push_back(value);
        data = std::move(new_data);
    }
    
    T& at(size_t index) {
        if (index >= data.size()) {
            throw std::out_of_range("索引超出范围");
        }
        return data[index];
    }
};
```

---

## 📊 异常处理 vs 错误码

### 对比分析

| 特性 | 异常处理 | 错误码 |
|------|----------|--------|
| 错误传播 | 自动传播 | 手动检查 |
| 代码清晰度 | 高 | 低 |
| 性能 | 异常发生时开销大 | 恒定开销 |
| 资源安全 | RAII自动保证 | 需要手动管理 |
| 适用场景 | 罕见错误 | 频繁发生的错误 |

### 推荐使用场景
- **使用异常处理**：资源管理、构造函数失败、罕见错误
- **使用错误码**：频繁发生的错误、性能敏感路径

---

## 🎓 学习建议

1. **理解RAII原则**：这是C++异常安全的基础
2. **掌握标准异常类**：了解何时使用哪种标准异常
3. **实践异常安全保证**：从基本保证到强保证的逐步实现
4. **善用智能指针**：自动资源管理是异常安全的关键
5. **合理使用noexcept**：明确函数的行为意图

---

## 💡 总结

C++异常处理是一个强大但需要谨慎使用的工具。正确的异常处理能够：
- 提高代码的健壮性和可维护性
- 确保资源的安全管理
- 提供清晰的错误处理逻辑
- 支持复杂的错误传播机制

通过RAII、智能指针和适当的异常安全保证，可以构建出既安全又高效的C++应用程序。