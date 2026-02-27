# C++ IO 流系统

C++ 的 IO 流系统基于面向对象设计，提供了类型安全、可扩展的输入输出机制。整个系统构建在流（stream）概念之上，所有 IO 操作都通过流对象完成。

## IO 流类层次结构

```
ios_base (流基类)
    ↓
ios (基础流功能)
    ↓
istream (输入流)              ostream (输出流)
    ↓                             ↓
ifstream (文件输入)           ofstream (文件输出)
istringstream (字符串输入)     ostringstream (字符串输出)
cin (标准输入)                cout/cerr/clog (标准输出)
```

---

## 标准输入输出流

### cout - 标准输出流

`cout` 是 `ostream` 类的全局对象，连接到标准输出设备。

#### 格式化输出

```cpp
#include <iostream>
#include <iomanip>  // 格式化控制
using namespace std;

void formatOutput() {
    int num = 255;
    double pi = 3.1415926535;
    
    // 数值格式化
    cout << "十进制: " << num << endl;              // 255
    cout << "十六进制: " << hex << num << endl;     // ff
    cout << "八进制: " << oct << num << endl;       // 377
    
    // 浮点数格式化
    cout << "固定小数点: " << fixed << pi << endl;        // 3.141593
    cout << "科学计数法: " << scientific << pi << endl;   // 3.141593e+00
    cout << "精度控制: " << setprecision(4) << pi << endl; // 3.1416
    
    // 宽度和对齐
    cout << setw(10) << "Hello" << endl;           // "     Hello"
    cout << setw(10) << left << "Hello" << endl;   // "Hello     "
    cout << setfill('*') << setw(10) << "Hi" << endl; // "********Hi"
    
    // 重置格式
    cout << dec << defaultfloat;
}
```

**常用格式化操纵器：**

| 操纵器 | 作用 | 示例 |
|-------|------|------|
| `hex/oct/dec` | 进制转换 | `cout << hex << 255;` |
| `fixed/scientific` | 浮点格式 | `cout << fixed << 3.14;` |
| `setprecision(n)` | 精度控制 | `setprecision(4)` |
| `setw(n)` | 字段宽度 | `setw(10)` |
| `left/right` | 对齐方式 | `left`, `right` |
| `setfill(c)` | 填充字符 | `setfill('*')` |

#### 缓冲区管理

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void bufferDemo() {
    // 缓冲区延迟显示
    cout << "无换行，可能延迟显示";
    this_thread::sleep_for(chrono::seconds(1));
    cout << " - 现在显示完整" << endl;
    
    // 强制刷新缓冲区
    cout << "立即显示: ";
    cout.flush();
    
    // endl vs \n
    cout << "使用\\n\n";      // 不刷新缓冲区
    cout << "使用endl" << endl;  // 刷新缓冲区
}
```

### cin - 标准输入流

`cin` 是 `istream` 类的全局对象，连接到标准输入设备。

#### 基本输入与错误处理

```cpp
#include <iostream>
#include <limits>
using namespace std;

void clearInputBuffer() {
    cin.clear();  // 清除错误状态
    cin.ignore(numeric_limits<streamsize>::max(), '\n');  // 清空缓冲区
}

void inputDemo() {
    int age;
    
    // 安全输入
    cout << "请输入年龄: ";
    while (!(cin >> age)) {
        cout << "输入错误，请重新输入: ";
        clearInputBuffer();
    }
    
    // 流状态检查
    if (cin.fail()) cerr << "输入操作失败!" << endl;
    if (cin.eof()) cerr << "到达文件末尾!" << endl;
}
```

**流状态标志：**

| 标志 | 含义 | 检查方法 |
|-----|------|---------|
| `goodbit` | 正常状态 | `cin.good()` |
| `failbit` | 格式错误 | `cin.fail()` |
| `badbit` | 流损坏 | `cin.bad()` |
| `eofbit` | 文件结束 | `cin.eof()` |

#### 高级输入技巧

```cpp
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

void advancedInput() {
    string line;
    
    // 读取整行
    cout << "请输入一行: ";
    getline(cin, line);
    
    // 读取单个字符（包括空白符）
    char ch = cin.get();
    cin.ignore();  // 消耗换行符
    
    // 查看下一个字符但不提取
    char next = cin.peek();
    
    // 使用字符串流解析输入
    istringstream iss("Alice 25 95.5");
    string name;
    int age;
    double score;
    iss >> name >> age >> score;
}
```

---

## 字符串流

字符串流将字符串作为 IO 源，适合数据格式转换和解析。

### 基本使用

```cpp
#include <sstream>
#include <string>
using namespace std;

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
```

### 类型转换工具

```cpp
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

// 使用示例
string s = toString(3.14);      // "3.14"
int n = fromString<int>("42");  // 42
```

### 复杂数据解析

```cpp
#include <vector>
#include <tuple>

void parseCSV() {
    string data = "Alice:25:95.5,Bob:30:88.2";
    vector<tuple<string, int, double>> people;
    
    istringstream dataStream(data);
    string personStr;
    
    while (getline(dataStream, personStr, ',')) {
        istringstream personStream(personStr);
        string name, ageStr, scoreStr;
        
        getline(personStream, name, ':');
        getline(personStream, ageStr, ':');
        getline(personStream, scoreStr, ':');
        
        people.emplace_back(name, fromString<int>(ageStr), 
                            fromString<double>(scoreStr));
    }
}
```

---

## 文件流

文件流用于文件的读写操作，包括文本文件和二进制文件。

### 文本文件操作

```cpp
#include <fstream>
#include <vector>
#include <string>

class TextFileHandler {
public:
    // 写入文件
    static bool write(const string& filename, const vector<string>& lines) {
        ofstream file(filename);
        if (!file) return false;
        
        for (const auto& line : lines) {
            file << line << '\n';
        }
        return !file.fail();
    }
    
    // 读取文件
    static vector<string> read(const string& filename) {
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
    static bool append(const string& filename, const string& content) {
        ofstream file(filename, ios::app);  // 追加模式
        if (!file) return false;
        file << content << endl;
        return !file.fail();
    }
};

// 使用示例
void textFileDemo() {
    vector<string> lines = {"第一行", "第二行", "第三行"};
    TextFileHandler::write("example.txt", lines);
    
    auto content = TextFileHandler::read("example.txt");
    for (const auto& line : content) {
        cout << line << endl;
    }
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
class BinaryFileHandler {
public:
    // 写入二进制数据
    static bool write(const string& filename, const vector<int>& data) {
        ofstream file(filename, ios::binary);
        if (!file) return false;
        
        file.write(reinterpret_cast<const char*>(data.data()), 
                   data.size() * sizeof(int));
        return !file.fail();
    }
    
    // 读取二进制数据
    static vector<int> read(const string& filename) {
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
};

// 使用示例
void binaryFileDemo() {
    vector<int> data = {1, 2, 3, 4, 5};
    BinaryFileHandler::write("data.bin", data);
    
    auto readData = BinaryFileHandler::read("data.bin");
    for (int val : readData) {
        cout << val << " ";
    }
}
```

---

## 自定义流操纵器

可以创建自定义的流操纵器来实现特定的格式化输出。

```cpp
#include <iostream>
#include <iomanip>
using namespace std;

// 无参数的流操纵器
ostream& red(ostream& os) { return os << "\033[31m"; }
ostream& green(ostream& os) { return os << "\033[32m"; }
ostream& reset(ostream& os) { return os << "\033[0m"; }

// 带参数的流操纵器
class indent {
    int spaces;
public:
    explicit indent(int s) : spaces(s) {}
    friend ostream& operator<<(ostream& os, const indent& ind) {
        for (int i = 0; i < ind.spaces; ++i) os << " ";
        return os;
    }
};

// 使用示例
void customManipulatorDemo() {
    cout << red << "错误信息" << reset << endl;
    cout << green << "成功信息" << reset << endl;
    cout << indent(4) << "缩进4格" << endl;
}
```

---

## 异常安全的 IO 操作

使用 RAII 原则确保文件资源的正确释放。

```cpp
#include <fstream>
#include <stdexcept>

class SafeFileWriter {
    string filename;
public:
    explicit SafeFileWriter(const string& fname) : filename(fname) {}
    
    void write(const string& content) {
        ofstream file(filename);  // RAII: 文件自动关闭
        if (!file) {
            throw runtime_error("无法打开文件: " + filename);
        }
        file << content;
        if (file.fail()) {
            throw runtime_error("写入文件失败");
        }
    }  // 文件在此自动关闭
};

// 使用示例
void safeFileDemo() {
    try {
        SafeFileWriter writer("safe_test.txt");
        writer.write("安全写入的内容");
        cout << "文件操作成功!" << endl;
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
}
```

---

## 性能优化

### 批量写入

```cpp
#include <sstream>

void performanceDemo() {
    const int iterations = 10000;
    
    // 方法1：频繁单次写入（慢）
    {
        ofstream file1("slow.txt");
        for (int i = 0; i < iterations; ++i) {
            file1 << "Line " << i << endl;  // 频繁IO
        }
    }
    
    // 方法2：批量写入（快）
    {
        ofstream file2("fast.txt");
        ostringstream buffer;
        for (int i = 0; i < iterations; ++i) {
            buffer << "Line " << i << '\n';  // 先写入内存
        }
        file2 << buffer.str();  // 一次性写入文件
    }
}
```

### 缓冲区大小优化

```cpp
void bufferSizeOptimization() {
    ofstream file("optimized.bin", ios::binary);
    
    // 设置更大的缓冲区
    char buffer[8192];  // 8KB 缓冲区
    file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
    
    // 写入操作会更高效
    // ...
}
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

## 常见问题与最佳实践

### ⚠️ 常见错误

**1. 不检查文件是否成功打开**
```cpp
// ❌ 错误
ofstream file("data.txt");
file << "重要数据";  // 如果打开失败，数据丢失

// ✅ 正确
ofstream file("data.txt");
if (!file) {
    cerr << "文件打开失败!" << endl;
    return;
}
file << "重要数据";
```

**2. 二进制文件大小计算错误**
```cpp
// ❌ 错误
int data[10];
file.read(reinterpret_cast<char*>(data), 10);  // 应该是 10 * sizeof(int)

// ✅ 正确
file.read(reinterpret_cast<char*>(data), 10 * sizeof(int));
```

**3. getline 后混合使用 cin >>**
```cpp
// 问题：cin >> 会留下换行符，影响后续 getline
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

1. **始终检查流状态**：使用 `if (!file)` 或 `file.fail()` 检查
2. **使用 RAII**：让文件对象自动管理资源
3. **批量处理**：减少 IO 操作次数
4. **选择合适的模式**：文本用默认，二进制用 `ios::binary`
5. **异常安全**：使用 try-catch 处理文件操作错误

---

## 参考资料

- [cppreference - IO Stream](https://en.cppreference.com/w/cpp/io)
- [cppreference - Filesystem](https://en.cppreference.com/w/cpp/filesystem)