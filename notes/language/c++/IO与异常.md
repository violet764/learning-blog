# C++ 输入输出与异常处理

C++ 的 IO 流系统基于面向对象设计，提供了类型安全、可扩展的输入输出机制。异常处理机制通过栈展开和 RAII 相结合，提供了安全、优雅的错误处理方案。

---

## 标准输入输出流

### cout - 标准输出流

```cpp
#include <iostream>
#include <iomanip>
using namespace std;

void formatOutput() {
    int num = 255;
    double pi = 3.1415926535;
    
    // 进制转换
    cout << "十进制: " << num << endl;           // 255
    cout << "十六进制: " << hex << num << endl;  // ff
    cout << "八进制: " << oct << num << endl;    // 377
    
    // 浮点数格式化
    cout << "固定小数点: " << fixed << pi << endl;       // 3.141593
    cout << "科学计数法: " << scientific << pi << endl;  // 3.141593e+00
    cout << "精度控制: " << setprecision(4) << pi << endl;
    
    // 宽度和对齐
    cout << setw(10) << "Hello" << endl;          // "     Hello"
    cout << setw(10) << left << "Hello" << endl;  // "Hello     "
    cout << setfill('*') << setw(10) << "Hi" << endl;
    
    // 重置格式
    cout << dec << defaultfloat;
}
```

**常用格式化操纵器：**

| 操纵器 | 作用 |
|-------|------|
| `hex/oct/dec` | 进制转换 |
| `fixed/scientific` | 浮点格式 |
| `setprecision(n)` | 精度控制 |
| `setw(n)` | 字段宽度 |
| `left/right` | 对齐方式 |
| `setfill(c)` | 填充字符 |

### cin - 标准输入流

```cpp
#include <iostream>
#include <limits>
using namespace std;

void clearInputBuffer() {
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void inputDemo() {
    int age;
    
    // 安全输入
    cout << "请输入年龄: ";
    while (!(cin >> age)) {
        cout << "输入错误，请重新输入: ";
        clearInputBuffer();
    }
}
```

**流状态标志：**

| 标志 | 含义 | 检查方法 |
|-----|------|---------|
| `goodbit` | 正常状态 | `cin.good()` |
| `failbit` | 格式错误 | `cin.fail()` |
| `badbit` | 流损坏 | `cin.bad()` |
| `eofbit` | 文件结束 | `cin.eof()` |

---

## 字符串流

字符串流将字符串作为 IO 源，适合数据格式转换和解析。

```cpp
#include <sstream>
#include <string>

void stringStreamDemo() {
    // 字符串转数值
    istringstream iss("42 3.14");
    int i;
    double d;
    iss >> i >> d;  // i=42, d=3.14
    
    // 数值转字符串
    ostringstream oss;
    oss << "数值: " << 42 << ", 浮点: " << 3.14;
    string result = oss.str();  // "数值: 42, 浮点: 3.14"
}

// 泛型类型转换
template<typename T>
string toString(const T& value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

template<typename T>
T fromString(const string& str) {
    istringstream iss(str);
    T value;
    iss >> value;
    return value;
}
```

---

## 文件流

### 文本文件操作

```cpp
#include <fstream>
#include <vector>
#include <string>

// 写入文件
void writeTextFile(const string& filename, const vector<string>& lines) {
    ofstream file(filename);
    if (!file) return;
    
    for (const auto& line : lines) {
        file << line << '\n';
    }
}

// 读取文件
vector<string> readTextFile(const string& filename) {
    ifstream file(filename);
    vector<string> lines;
    
    if (!file) return lines;
    
    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

// 追加内容
void appendToFile(const string& filename, const string& content) {
    ofstream file(filename, ios::app);
    if (!file) return;
    file << content << endl;
}
```

**文件打开模式：**

| 模式 | 作用 |
|-----|------|
| `ios::in` | 读模式 |
| `ios::out` | 写模式 |
| `ios::app` | 追加模式 |
| `ios::ate` | 打开时定位到文件末尾 |
| `ios::trunc` | 截断文件 |
| `ios::binary` | 二进制模式 |

### 二进制文件操作

```cpp
// 写入二进制数据
void writeBinary(const string& filename, const vector<int>& data) {
    ofstream file(filename, ios::binary);
    if (!file) return;
    
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(int));
}

// 读取二进制数据
vector<int> readBinary(const string& filename) {
    ifstream file(filename, ios::binary);
    vector<int> data;
    
    if (!file) return data;
    
    // 获取文件大小
    file.seekg(0, ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, ios::beg);
    
    data.resize(fileSize / sizeof(int));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    return data;
}
```

---

## 异常处理基础

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

## 标准异常类层次

```
std::exception
├── std::logic_error           // 逻辑错误
│   ├── std::invalid_argument  // 无效参数
│   ├── std::out_of_range      // 范围越界
│   └── std::length_error      // 长度错误
├── std::runtime_error         // 运行时错误
│   ├── std::overflow_error    // 溢出错误
│   └── std::system_error      // 系统错误
├── std::bad_alloc             // 内存分配失败
└── std::bad_cast              // 类型转换失败
```

**常用异常类选择：**

| 异常类 | 使用场景 |
|--------|----------|
| `std::invalid_argument` | 函数接收到无效参数 |
| `std::out_of_range` | 访问超出有效范围 |
| `std::runtime_error` | 运行时检测到的错误 |
| `std::bad_alloc` | 内存分配失败 |

---

## 异常安全保证

### 基本保证

不泄漏资源，对象处于有效状态：

```cpp
class ResourceManager {
    std::unique_ptr<int> resource;
public:
    void basicGuarantee(int value) {
        auto temp = std::make_unique<int>(value);
        if (value < 0) throw std::invalid_argument("值不能为负数");
        resource = std::move(temp);
    }
};
```

### 强保证

操作具有原子性——要么完全成功，要么完全失败：

```cpp
void strongGuarantee(int value) {
    auto oldResource = std::move(resource);
    try {
        auto temp = std::make_unique<int>(value);
        if (value < 0) throw std::invalid_argument("值不能为负数");
        resource = std::move(temp);
    } catch (...) {
        resource = std::move(oldResource);
        throw;
    }
}
```

### 不抛出保证

```cpp
int noThrowGuarantee() noexcept {
    return resource ? *resource : 0;
}
```

---

## RAII 与异常安全

RAII 利用对象生命周期管理资源，确保异常发生时资源能正确释放。

```cpp
#include <fstream>
#include <memory>

class FileHandler {
    std::unique_ptr<std::fstream> file;
    
public:
    FileHandler(const std::string& filename) {
        file = std::make_unique<std::fstream>(filename);
        if (!file->is_open()) {
            throw std::runtime_error("无法打开文件");
        }
    }
    
    void write(const std::string& data) {
        *file << data;
    }
    
    ~FileHandler() = default;  // 自动关闭文件
};

void safeFileOperation() {
    FileHandler handler("data.txt");
    handler.write("Hello, RAII!");
    // 函数结束时文件自动关闭，即使抛出异常
}
```

⚠️ **注意**：永远不要在析构函数中抛出异常！

---

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
```

---

## noexcept 关键字

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

// 移动操作通常标记 noexcept
class Movable {
public:
    Movable(Movable&& other) noexcept { }
    Movable& operator=(Movable&& other) noexcept { return *this; }
};
```

---

## 现代 C++ 文件系统（C++17）

```cpp
#include <filesystem>

namespace fs = std::filesystem;

void filesystemDemo() {
    fs::path filePath = "test.txt";
    
    // 检查文件状态
    if (fs::exists(filePath)) {
        cout << "文件大小: " << fs::file_size(filePath) << " 字节" << endl;
    }
    
    // 创建目录
    fs::create_directory("new_dir");
    
    // 遍历目录
    for (const auto& entry : fs::directory_iterator(".")) {
        cout << (entry.is_directory() ? "[DIR] " : "[FILE] ") 
             << entry.path().filename() << endl;
    }
    
    // 复制/移动/删除文件
    fs::copy_file("src.txt", "dest.txt");
    fs::rename("old.txt", "new.txt");
    fs::remove("unwanted.txt");
}
```

---

## 常见错误与最佳实践

### ⚠️ 常见错误

**1. 不检查文件是否成功打开**
```cpp
// ❌ 错误
ofstream file("data.txt");
file << "重要数据";

// ✅ 正确
ofstream file("data.txt");
if (!file) {
    cerr << "文件打开失败!" << endl;
    return;
}
```

**2. getline 后混合使用 cin >>**
```cpp
int n;
cin >> n;
string line;
getline(cin, line);  // 读取到空行

// 解决：清空缓冲区
cin >> n;
cin.ignore(numeric_limits<streamsize>::max(), '\n');
getline(cin, line);
```

### ✅ 最佳实践

1. **始终检查流状态**：使用 `if (!file)` 检查
2. **使用 RAII**：让文件对象自动管理资源
3. **批量处理**：减少 IO 操作次数
4. **选择合适的模式**：文本用默认，二进制用 `ios::binary`
5. **异常安全**：至少保证基本安全
6. **善用智能指针**：自动资源管理是关键

---

## 参考资料

- [cppreference - IO Stream](https://en.cppreference.com/w/cpp/io)
- [cppreference - Exception](https://en.cppreference.com/w/cpp/error/exception)
- [cppreference - Filesystem](https://en.cppreference.com/w/cpp/filesystem)
