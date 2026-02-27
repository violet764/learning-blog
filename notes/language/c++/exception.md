# C++ 异常处理

C++ 异常处理机制通过**栈展开（stack unwinding）**和 **RAII** 相结合，提供了比传统错误码更安全、更优雅的错误处理方案。

## 异常处理流程

### 三要素

| 关键字 | 作用 |
|--------|------|
| `throw` | 抛出异常 |
| `try` | 尝试执行可能抛出异常的代码块 |
| `catch` | 捕获并处理异常 |

### 基本语法

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

💡 **提示**：`catch(...)` 可以捕获所有类型的异常，但无法获取异常信息，通常作为最后的兜底处理。

## 标准异常类层次

### 继承关系

```
std::exception
├── std::logic_error           // 逻辑错误（可预防）
│   ├── std::invalid_argument  // 无效参数
│   ├── std::out_of_range      // 范围越界
│   ├── std::length_error      // 长度错误
│   └── std::domain_error      // 定义域错误
├── std::runtime_error         // 运行时错误（难以预防）
│   ├── std::overflow_error    // 溢出错误
│   ├── std::underflow_error   // 下溢错误
│   ├── std::range_error       // 范围错误
│   └── std::system_error      // 系统错误
├── std::bad_alloc             // 内存分配失败
├── std::bad_cast              // 类型转换失败
└── std::bad_typeid            // typeid 操作失败
```

### 常用异常类选择

| 异常类 | 使用场景 |
|--------|----------|
| `std::invalid_argument` | 函数接收到无效参数 |
| `std::out_of_range` | 访问超出有效范围（如数组越界） |
| `std::runtime_error` | 运行时检测到的错误 |
| `std::bad_alloc` | 内存分配失败 |

## 异常安全保证

异常安全是指当异常发生时，程序仍能保持正确状态。共有三个级别：

### 基本保证（Basic Guarantee）

不泄漏资源，对象处于有效状态：

```cpp
class ResourceManager {
    std::unique_ptr<int> resource;
public:
    void basicGuarantee(int value) {
        auto temp = std::make_unique<int>(value);
        
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        
        resource = std::move(temp);  // 提交操作（无异常）
    }
};
```

### 强保证（Strong Guarantee）

操作具有原子性——要么完全成功，要么完全失败：

```cpp
void strongGuarantee(int value) {
    auto oldResource = std::move(resource);  // 保存旧状态
    
    try {
        auto temp = std::make_unique<int>(value);
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        resource = std::move(temp);
    } catch (...) {
        resource = std::move(oldResource);  // 回滚
        throw;  // 重新抛出异常
    }
}
```

### 不抛出保证（No-throw Guarantee）

函数承诺不会抛出异常：

```cpp
int noThrowGuarantee() noexcept {
    return resource ? *resource : 0;
}
```

## RAII 与异常安全

**RAII（Resource Acquisition Is Initialization）** 利用对象生命周期管理资源，确保异常发生时资源能正确释放。

```cpp
#include <fstream>
#include <memory>

class FileHandler {
    std::unique_ptr<std::fstream> file;
    
public:
    FileHandler(const std::string& filename) {
        file = std::make_unique<std::fstream>(filename);
        if (!file->is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }
    }
    
    ~FileHandler() = default;  // 自动关闭文件
    
    void write(const std::string& data) {
        *file << data;
        if (file->fail()) {
            throw std::runtime_error("写入文件失败");
        }
    }
};

void safeFileOperation() {
    FileHandler handler("data.txt");  // RAII 自动管理
    handler.write("Hello, RAII!");
    // 函数结束时文件自动关闭，即使抛出异常
}
```

⚠️ **注意**：永远不要在析构函数中抛出异常！

## 自定义异常类

```cpp
#include <exception>
#include <string>

class MyException : public std::exception {
    std::string message;
    int errorCode;
    
public:
    MyException(const std::string& msg, int code = 0) 
        : message(msg), errorCode(code) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    int getErrorCode() const noexcept { return errorCode; }
};

// 使用示例
void processData(int data) {
    if (data < 0) {
        throw MyException("数据不能为负数", 1001);
    }
}

int main() {
    try {
        processData(-5);
    } catch (const MyException& e) {
        std::cerr << "错误代码: " << e.getErrorCode() 
                  << ", 消息: " << e.what() << std::endl;
    }
}
```

## noexcept 关键字

### 基本用法

```cpp
// 不抛出异常的函数
void no_throw_function() noexcept {
    // 保证不抛出异常
}

// 条件性 noexcept
template<typename T>
void swap(T& a, T& b) noexcept(noexcept(a.swap(b))) {
    a.swap(b);
}
```

### noexcept 的优势

- **编译器优化**：编译器知道函数不会抛出异常，可以进行更多优化
- **代码清晰**：明确表达函数的行为意图
- **STL 容器优化**：移动操作标记为 `noexcept` 时，容器会优先使用移动而非拷贝

```cpp
class Movable {
public:
    Movable(Movable&& other) noexcept {  // 移动构造通常标记 noexcept
        // 移动资源...
    }
    
    Movable& operator=(Movable&& other) noexcept {
        // 移动赋值...
        return *this;
    }
};
```

## 最佳实践

### 构造函数中使用函数 try 块

```cpp
class Application {
    std::unique_ptr<Database> db;
    
public:
    Application() try : db(std::make_unique<Database>()) {
        db->connect();
    } catch (const std::exception& e) {
        std::cerr << "应用初始化失败: " << e.what() << std::endl;
        throw;  // 重新抛出
    }
};
```

### 析构函数必须 noexcept

```cpp
~Application() noexcept {
    try {
        if (db) {
            // 清理资源
        }
    } catch (...) {
        // 记录日志，但不抛出异常
        std::cerr << "析构函数中发生异常，已忽略" << std::endl;
    }
}
```

### 异常重新抛出

```cpp
try {
    // 可能抛出异常的代码
} catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    throw;  // 重新抛出，让上层处理
}
```

### 异常层次化捕获

```cpp
void handleComplexOperation() {
    try {
        // 复杂操作...
    } catch (const std::invalid_argument& e) {
        // 处理特定异常
    } catch (const std::runtime_error& e) {
        // 处理运行时错误
    } catch (const std::exception& e) {
        // 处理标准异常
    } catch (...) {
        // 兜底处理
    }
}
```

## 异常处理 vs 错误码

| 特性 | 异常处理 | 错误码 |
|------|----------|--------|
| 错误传播 | 自动传播 | 手动检查 |
| 代码清晰度 | 高 | 低 |
| 性能 | 异常发生时开销大 | 恒定开销 |
| 资源安全 | RAII 自动保证 | 需手动管理 |
| 适用场景 | 罕见错误 | 频繁发生的错误 |

### 选择建议

- ✅ **使用异常**：资源管理、构造函数失败、罕见错误
- ✅ **使用错误码**：频繁发生的错误、性能敏感路径、跨语言接口

## 性能注意事项

- **正常执行路径**：几乎零开销（现代编译器优化）
- **抛出异常时**：有运行时开销（栈展开、类型匹配）
- **不要用异常做流程控制**：异常适用于异常情况，不应作为常规控制流

```cpp
// ❌ 错误用法：用异常做流程控制
int findIndex(const std::vector<int>& v, int target) {
    try {
        for (size_t i = 0; ; i++) {
            if (v.at(i) == target) return i;  // at() 可能抛出异常
        }
    } catch (const std::out_of_range&) {
        return -1;
    }
}

// ✅ 正确用法：用错误码处理预期情况
int findIndex(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == target) return i;
    }
    return -1;
}
```

## 总结

C++ 异常处理的核心要点：

1. 📌 **理解 RAII**：这是 C++ 异常安全的基础
2. 📌 **掌握标准异常类**：选择合适的异常类型
3. 📌 **提供异常安全保证**：至少保证基本安全
4. 📌 **善用智能指针**：自动资源管理是关键
5. 📌 **合理使用 noexcept**：明确函数的行为意图
