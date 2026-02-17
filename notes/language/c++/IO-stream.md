## IO流系统详解

### 概述
C++的IO流系统基于面向对象设计，提供了类型安全、可扩展的输入输出机制。整个系统构建在流（stream）概念之上，所有IO操作都通过流对象完成。

### IO流类层次结构
```
ios_base (流基类)
    ↓
ios (基础流功能)
    ↓
istream (输入流)        ostream (输出流)
    ↓                       ↓
ifstream (文件输入)     ofstream (文件输出)
istringstream (字符串输入) ostringstream (字符串输出)
cin (标准输入)          cout/cerr/clog (标准输出)
```

### 标准输入输出流

#### 1. `cout` - 标准输出流

**技术原理：** `cout`是`ostream`类的全局对象，连接到标准输出设备（通常是控制台）。

**格式化输出控制：**
```cpp
#include <iostream>
#include <iomanip>  // 格式化控制
using namespace std;

int main() {
    int num = 255;
    double pi = 3.1415926535;
    
    // 数值格式化
    cout << "十进制: " << num << endl;                    // 255
    cout << "十六进制: " << hex << num << endl;           // ff
    cout << "八进制: " << oct << num << endl;             // 377
    
    // 浮点数格式化
    cout << "默认精度: " << pi << endl;                   // 3.14159
    cout << "固定小数点: " << fixed << pi << endl;        // 3.141593
    cout << "科学计数法: " << scientific << pi << endl;   // 3.141593e+00
    cout << "精度控制: " << setprecision(4) << pi << endl; // 3.1416
    
    // 宽度和对齐
    cout << setw(10) << "Hello" << endl;                  // "     Hello"
    cout << setw(10) << left << "Hello" << endl;          // "Hello     "
    cout << setw(10) << right << "Hello" << endl;         // "     Hello"
    cout << setfill('*') << setw(10) << "Hi" << endl;     // "********Hi"
    
    // 重置格式
    cout << dec;  // 恢复十进制
    cout << defaultfloat;  // 恢复默认浮点格式
    
    return 0;
}
```

**缓冲区管理：**
```cpp
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    // 缓冲区行为对比
    cout << "无换行，可能延迟显示";
    this_thread::sleep_for(chrono::seconds(2));
    cout << " - 现在显示完整" << endl;
    
    cout << "立即显示: ";
    cout.flush();  // 强制刷新缓冲区
    this_thread::sleep_for(chrono::seconds(1));
    cout << "延迟内容" << endl;
    
    // endl vs \n
    cout << "使用\\n:" << "第一行\n第二行\n";
    cout << "使用endl:" << "第一行" << endl << "第二行" << endl;
    
    return 0;
}
```

#### 2. `cin` - 标准输入流

**错误处理机制：**
```cpp
#include <iostream>
#include <limits>
using namespace std;

void clear_input_buffer() {
    cin.clear();  // 清除错误状态
    cin.ignore(numeric_limits<streamsize>::max(), '\n');  // 清空缓冲区
}

int main() {
    int age;
    double salary;
    
    // 基本输入
    cout << "请输入年龄: ";
    while (!(cin >> age)) {
        cout << "输入错误，请重新输入年龄: ";
        clear_input_buffer();
    }
    
    cout << "请输入工资: ";
    while (!(cin >> salary)) {
        cout << "输入错误，请重新输入工资: ";
        clear_input_buffer();
    }
    
    // 流状态检查
    if (cin.fail()) {
        cerr << "输入操作失败!" << endl;
    }
    if (cin.eof()) {
        cerr << "到达文件末尾!" << endl;
    }
    
    cout << "年龄: " << age << ", 工资: " << salary << endl;
    
    return 0;
}
```

**高级输入技巧：**
```cpp
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

int main() {
    string line;
    
    // 读取整行并解析
    cout << "请输入多个数据（格式：姓名 年龄 工资）: ";
    getline(cin, line);
    
    istringstream iss(line);
    string name;
    int age;
    double salary;
    
    if (iss >> name >> age >> salary) {
        cout << "姓名: " << name << ", 年龄: " << age 
             << ", 工资: " << salary << endl;
    } else {
        cerr << "输入格式错误!" << endl;
    }
    
    // 字符级输入
    cout << "请输入一个字符: ";
    char ch = cin.get();  // 读取单个字符（包括空白符）
    cin.ignore();  // 消耗换行符
    cout << "字符: '" << ch << "'" << endl;
    
    // 查看下一个字符但不提取
    cout << "请输入一些文本: ";
    char next_char = cin.peek();
    cout << "下一个字符是: '" << next_char << "'" << endl;
    
    return 0;
}
```

### 字符串流（String Streams）

#### 3. `stringstream` 家族

**高级字符串处理：**
```cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

// 数据类型转换工具函数
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
    if (iss.fail()) {
        throw invalid_argument("转换失败: " + str);
    }
    return value;
}

int main() {
    // 复杂数据解析
    string complex_data = "Alice:25:95.5,Bob:30:88.2,Charlie:22:91.8";
    
    vector<tuple<string, int, double>> people;
    
    // 使用字符串流解析CSV格式数据
    istringstream data_stream(complex_data);
    string person_str;
    
    while (getline(data_stream, person_str, ',')) {
        istringstream person_stream(person_str);
        string name, age_str, salary_str;
        
        getline(person_stream, name, ':');
        getline(person_stream, age_str, ':');
        getline(person_stream, salary_str, ':');
        
        try {
            int age = fromString<int>(age_str);
            double salary = fromString<double>(salary_str);
            people.emplace_back(name, age, salary);
        } catch (const exception& e) {
            cerr << "解析错误: " << e.what() << endl;
        }
    }
    
    // 输出解析结果
    for (const auto& person : people) {
        cout << "姓名: " << get<0>(person) 
             << ", 年龄: " << get<1>(person)
             << ", 工资: " << get<2>(person) << endl;
    }
    
    // 构建复杂字符串
    ostringstream report;
    report << "=== 人员报告 ===" << endl;
    report << "总人数: " << people.size() << endl;
    
    double total_salary = 0;
    for (const auto& person : people) {
        total_salary += get<2>(person);
    }
    
    report << "平均工资: " << (total_salary / people.size()) << endl;
    report << "报告生成时间: " << __DATE__ << " " << __TIME__ << endl;
    
    cout << report.str() << endl;
    
    return 0;
}
```

### 文件流（File Streams）

#### 4. 文件输入输出

**文件操作最佳实践：**
```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
using namespace std;

class FileHandler {
private:
    string filename;
    
public:
    FileHandler(const string& fname) : filename(fname) {}
    
    // 安全写入文件
    bool writeToFile(const vector<string>& lines) {
        ofstream outfile(filename);
        if (!outfile) {
            cerr << "无法打开文件: " << filename << endl;
            return false;
        }
        
        for (const auto& line : lines) {
            outfile << line << '\n';
        }
        
        // 检查写入是否成功
        if (outfile.fail()) {
            cerr << "写入文件失败!" << endl;
            return false;
        }
        
        outfile.close();
        return true;
    }
    
    // 安全读取文件
    vector<string> readFromFile() {
        ifstream infile(filename);
        vector<string> lines;
        
        if (!infile) {
            throw runtime_error("无法打开文件: " + filename);
        }
        
        string line;
        while (getline(infile, line)) {
            lines.push_back(line);
        }
        
        // 检查读取状态
        if (!infile.eof() && infile.fail()) {
            throw runtime_error("读取文件时发生错误");
        }
        
        infile.close();
        return lines;
    }
    
    // 追加模式写入
    bool appendToFile(const string& content) {
        ofstream outfile(filename, ios::app);  // 追加模式
        if (!outfile) {
            return false;
        }
        
        outfile << content << endl;
        return !outfile.fail();
    }
    
    // 二进制文件操作
    bool writeBinary(const vector<int>& data) {
        ofstream outfile(filename, ios::binary);
        if (!outfile) return false;
        
        for (int value : data) {
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        
        return !outfile.fail();
    }
    
    vector<int> readBinary() {
        ifstream infile(filename, ios::binary);
        vector<int> data;
        
        if (!infile) return data;
        
        int value;
        while (infile.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            data.push_back(value);
        }
        
        return data;
    }
};

int main() {
    FileHandler handler("test.txt");
    
    // 文本文件操作
    vector<string> lines = {"第一行", "第二行", "第三行"};
    if (handler.writeToFile(lines)) {
        cout << "文件写入成功!" << endl;
    }
    
    try {
        auto read_lines = handler.readFromFile();
        cout << "读取到的内容:" << endl;
        for (const auto& line : read_lines) {
            cout << " - " << line << endl;
        }
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    
    // 二进制文件操作
    vector<int> binary_data = {1, 2, 3, 4, 5};
    if (handler.writeBinary(binary_data)) {
        cout << "二进制文件写入成功!" << endl;
    }
    
    auto read_binary = handler.readBinary();
    cout << "读取的二进制数据: ";
    for (int val : read_binary) {
        cout << val << " ";
    }
    cout << endl;
    
    return 0;
}
```

### 流操纵器（Stream Manipulators）

#### 5. 自定义流操纵器

**创建自定义格式化工具：**
```cpp
#include <iostream>
#include <iomanip>
#include <string>
using namespace std;

// 自定义流操纵器：颜色输出
ostream& red(ostream& os) {
    return os << "\033[31m";
}

ostream& green(ostream& os) {
    return os << "\033[32m";
}

ostream& blue(ostream& os) {
    return os << "\033[34m";
}

ostream& reset(ostream& os) {
    return os << "\033[0m";
}

// 自定义流操纵器：格式化输出
ostream& currency(ostream& os) {
    os << fixed << setprecision(2);
    return os << "$";
}

ostream& percentage(ostream& os) {
    os << fixed << setprecision(1);
    return os << "%";
}

// 带参数的流操纵器
class indent {
    int spaces;
public:
    indent(int s) : spaces(s) {}
    friend ostream& operator<<(ostream& os, const indent& ind);
};

ostream& operator<<(ostream& os, const indent& ind) {
    for (int i = 0; i < ind.spaces; ++i) {
        os << " ";
    }
    return os;
}

int main() {
    // 使用自定义流操纵器
    cout << red << "错误信息" << reset << endl;
    cout << green << "成功信息" << reset << endl;
    cout << blue << "提示信息" << reset << endl;
    
    double price = 19.99;
    double discount = 15.5;
    
    cout << "原价: " << currency << price << endl;
    cout << "折扣: " << percentage << discount << endl;
    
    // 使用带参数的流操纵器
    cout << indent(4) << "缩进4格" << endl;
    cout << indent(8) << "缩进8格" << endl;
    
    return 0;
}
```

### 错误处理与异常安全

#### 6. 流异常处理

**异常安全的IO操作：**
```cpp
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <memory>
using namespace std;

class SafeFileWriter {
private:
    string filename;
    
public:
    SafeFileWriter(const string& fname) : filename(fname) {}
    
    // RAII方式处理文件
    void writeWithRAII(const string& content) {
        ofstream file(filename);
        if (!file) {
            throw runtime_error("无法打开文件: " + filename);
        }
        
        file << content;
        
        if (file.fail()) {
            throw runtime_error("写入文件失败");
        }
        
        // 文件自动关闭（RAII）
    }
    
    // 使用智能指针管理文件流
    void writeWithSmartPtr(const vector<string>& lines) {
        auto file = make_unique<ofstream>(filename);
        if (!file->is_open()) {
            throw runtime_error("无法打开文件");
        }
        
        for (const auto& line : lines) {
            *file << line << endl;
            if (file->fail()) {
                throw runtime_error("写入失败");
            }
        }
    }
};

// 流状态检查工具
class StreamChecker {
public:
    static void checkStreamState(const ios& stream, const string& operation) {
        if (stream.bad()) {
            throw runtime_error(operation + ": 流已损坏");
        }
        if (stream.fail()) {
            throw runtime_error(operation + ": 操作失败");
        }
        if (stream.eof()) {
            cout << operation + ": 到达文件末尾" << endl;
        }
    }
    
    static void resetStream(ios& stream) {
        stream.clear();
        stream.ignore(numeric_limits<streamsize>::max(), '\n');
    }
};

int main() {
    try {
        SafeFileWriter writer("safe_test.txt");
        writer.writeWithRAII("这是安全写入的内容\n");
        
        vector<string> data = {"第一行", "第二行", "第三行"};
        writer.writeWithSmartPtr(data);
        
        cout << "文件操作成功!" << endl;
        
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    
    // 流状态检查示例
    ifstream test_file("nonexistent.txt");
    StreamChecker::checkStreamState(test_file, "打开文件");
    
    return 0;
}
```

### 性能优化技巧

#### 7. IO性能优化

**减少IO操作开销：**
```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
using namespace std;
using namespace chrono;

class IOBenchmark {
public:
    // 批量写入 vs 单次写入
    static void benchmarkWriteMethods() {
        const int iterations = 10000;
        
        // 方法1：频繁单次写入
        auto start1 = high_resolution_clock::now();
        {
            ofstream file1("method1.txt");
            for (int i = 0; i < iterations; ++i) {
                file1 << "Line " << i << endl;
            }
        }
        auto end1 = high_resolution_clock::now();
        
        // 方法2：批量写入
        auto start2 = high_resolution_clock::now();
        {
            ofstream file2("method2.txt");
            ostringstream buffer;
            for (int i = 0; i < iterations; ++i) {
                buffer << "Line " << i << endl;
            }
            file2 << buffer.str();
        }
        auto end2 = high_resolution_clock::now();
        
        auto duration1 = duration_cast<milliseconds>(end1 - start1);
        auto duration2 = duration_cast<milliseconds>(end2 - start2);
        
        cout << "单次写入时间: " << duration1.count() << "ms" << endl;
        cout << "批量写入时间: " << duration2.count() << "ms" << endl;
        cout << "性能提升: " << (duration1.count() - duration2.count()) << "ms" << endl;
    }
    
    // 缓冲区大小优化
    static void optimizeBufferSize() {
        const int data_size = 1000000;
        vector<int> data(data_size);
        
        // 默认缓冲区
        auto start1 = high_resolution_clock::now();
        {
            ofstream file("default_buffer.bin", ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()), 
                      data_size * sizeof(int));
        }
        auto end1 = high_resolution_clock::now();
        
        // 自定义大缓冲区
        auto start2 = high_resolution_clock::now();
        {
            ofstream file("large_buffer.bin", ios::binary);
            char buffer[8192];  // 8KB缓冲区
            file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
            file.write(reinterpret_cast<const char*>(data.data()), 
                      data_size * sizeof(int));
        }
        auto end2 = high_resolution_clock::now();
        
        cout << "默认缓冲区时间: " 
             << duration_cast<milliseconds>(end1 - start1).count() << "ms" << endl;
        cout << "大缓冲区时间: " 
             << duration_cast<milliseconds>(end2 - start2).count() << "ms" << endl;
    }
};

int main() {
    cout << "=== IO性能测试 ===" << endl;
    IOBenchmark::benchmarkWriteMethods();
    cout << endl;
    IOBenchmark::optimizeBufferSize();
    
    return 0;
}
```

### 现代C++ IO特性

#### 8. C++17/20新特性

**文件系统操作：**
```cpp
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;
using namespace std;

class ModernIO {
public:
    // 使用string_view提高效率
    static void processStringView(string_view sv) {
        cout << "处理字符串视图: " << sv << endl;
        cout << "长度: " << sv.length() << endl;
        cout << "子串: " << sv.substr(0, 5) << endl;
    }
    
    // 文件系统操作
    static void demonstrateFilesystem() {
        fs::path file_path = "modern_test.txt";
        
        // 检查文件状态
        if (fs::exists(file_path)) {
            cout << "文件存在，大小: " << fs::file_size(file_path) << " 字节" << endl;
            cout << "最后修改时间: " << fs::last_write_time(file_path).time_since_epoch().count() << endl;
        } else {
            cout << "文件不存在，创建新文件" << endl;
            ofstream file(file_path);
            file << "这是现代C++创建的文件" << endl;
        }
        
        // 遍历目录
        try {
            for (const auto& entry : fs::directory_iterator(".")) {
                cout << (entry.is_directory() ? "[DIR] " : "[FILE] ") 
                     << entry.path().filename() << endl;
            }
        } catch (const fs::filesystem_error& e) {
            cerr << "文件系统错误: " << e.what() << endl;
        }
    }
    
    // 格式化输出（C++20）
    static void demonstrateFormat() {
        int x = 42;
        double y = 3.14159;
        string name = "Alice";
        
        // C++20格式化（需要编译器支持）
        // cout << format("姓名: {}, 年龄: {}, 圆周率: {:.2f}", name, x, y) << endl;
        
        // 传统方式（兼容性更好）
        cout << "姓名: " << name << ", 年龄: " << x << ", 圆周率: " << fixed << setprecision(2) << y << endl;
    }
};

int main() {
    ModernIO::processStringView("Hello Modern C++");
    cout << endl;
    ModernIO::demonstrateFilesystem();
    cout << endl;
    ModernIO::demonstrateFormat();
    
    return 0;
}
```

### 关键总结与最佳实践

1. **流状态管理**：始终检查流状态，使用`clear()`和`ignore()`处理错误
2. **异常安全**：使用RAII和智能指针管理流资源
3. **性能优化**：减少IO操作次数，使用缓冲区批量处理
4. **类型安全**：优先使用C++流而非C风格IO函数
5. **现代特性**：利用`string_view`、文件系统等现代C++特性
6. **格式化控制**：熟练掌握流操纵器进行精确输出控制
7. **错误处理**：实现健壮的错误处理机制，避免程序崩溃

通过深入理解C++ IO流系统的技术原理和最佳实践，可以编写出高效、安全、可维护的输入输出代码。

# 文件读写案例

## 1. 核心概念
- 定义：实现C++中文件的读取、写入、追加等基本操作
- 关键特性：文本文件操作、二进制文件操作、文件流状态管理

## 2. 语法规则
- 基本语法：使用fstream、ifstream、ofstream类进行文件操作
- 代码示例：
```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class FileHandler {
public:
    // 写入文本文件
    static void writeTextFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        if (!file) {
            std::cout << "文件创建失败: " << filename << std::endl;
            return;
        }
        
        file << content;
        file.close();
        std::cout << "文件写入成功: " << filename << std::endl;
    }
    
    // 读取文本文件
    static void readTextFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cout << "文件打开失败: " << filename << std::endl;
            return;
        }
        
        std::string line;
        std::cout << "文件内容: " << std::endl;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
        file.close();
    }
    
    // 追加内容到文件
    static void appendToFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename, std::ios::app);  // append模式
        if (!file) {
            std::cout << "文件打开失败: " << filename << std::endl;
            return;
        }
        
        file << content << std::endl;
        file.close();
        std::cout << "内容追加成功: " << filename << std::endl;
    }
    
    // 二进制文件写入
    static void writeBinaryFile(const std::string& filename, const std::vector<int>& data) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "二进制文件创建失败: " << filename << std::endl;
            return;
        }
        
        file.write(reinterpret_cast<const char*>(data.data()), 
                   data.size() * sizeof(int));
        file.close();
        std::cout << "二进制文件写入成功: " << filename << std::endl;
    }
    
    // 二进制文件读取
    static std::vector<int> readBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "二进制文件打开失败: " << filename << std::endl;
            return {};
        }
        
        // 获取文件大小
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<int> data(fileSize / sizeof(int));
        file.read(reinterpret_cast<char*>(data.data()), fileSize);
        file.close();
        
        std::cout << "二进制文件读取成功，读取了 " << data.size() << " 个整数" << std::endl;
        return data;
    }
    
    // 文件信息统计
    static void showFileInfo(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cout << "文件打开失败: " << filename << std::endl;
            return;
        }
        
        size_t fileSize = file.tellg();
        file.close();
        
        std::cout << "文件信息: " << filename << std::endl;
        std::cout << "文件大小: " << fileSize << " 字节" << std::endl;
        std::cout << "文件大小: " << fileSize / 1024.0 << " KB" << std::endl;
    }
};

int main() {
    // 文本文件操作示例
    std::cout << "=== 文本文件操作 ===" << std::endl;
    FileHandler::writeTextFile("example.txt", "这是第一行文本\n这是第二行文本\n这是第三行文本");
    FileHandler::readTextFile("example.txt");
    FileHandler::appendToFile("example.txt", "这是追加的内容");
    FileHandler::readTextFile("example.txt");
    
    // 二进制文件操作示例
    std::cout << "\n=== 二进制文件操作 ===" << std::endl;
    std::vector<int> binaryData = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    FileHandler::writeBinaryFile("binary.dat", binaryData);
    
    auto readData = FileHandler::readBinaryFile("binary.dat");
    std::cout << "读取的数据: ";
    for (int num : readData) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 文件信息统计
    std::cout << "\n=== 文件信息统计 ===" << std::endl;
    FileHandler::showFileInfo("example.txt");
    FileHandler::showFileInfo("binary.dat");
    
    return 0;
}
```
- 注意事项：文件打开模式、错误处理、资源释放

## 3. 常见用法
- 场景1：配置文件读写
```cpp
#include <map>

class ConfigFileHandler {
public:
    static std::map<std::string, std::string> readConfig(const std::string& filename) {
        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        
        if (!file) {
            std::cout << "配置文件打开失败: " << filename << std::endl;
            return config;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                config[key] = value;
            }
        }
        file.close();
        return config;
    }
    
    static void writeConfig(const std::string& filename, 
                           const std::map<std::string, std::string>& config) {
        std::ofstream file(filename);
        if (!file) {
            std::cout << "配置文件创建失败: " << filename << std::endl;
            return;
        }
        
        for (const auto& pair : config) {
            file << pair.first << "=" << pair.second << std::endl;
        }
        file.close();
    }
};
```

- 场景2：日志文件记录
```cpp
#include <chrono>
#include <iomanip>

class Logger {
private:
    std::string logFile;
    
public:
    Logger(const std::string& filename) : logFile(filename) {}
    
    void log(const std::string& message, const std::string& level = "INFO") {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::ofstream file(logFile, std::ios::app);
        if (!file) return;
        
        file << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        file << " [" << level << "] " << message << std::endl;
        file.close();
    }
};
```

## 4. 易错点/坑
- 错误示例：
```cpp
// 不检查文件是否成功打开
std::ofstream file("data.txt");
file << "重要数据";  // 如果文件打开失败，数据丢失
```
- 原因：缺乏错误检查机制
- 修正方案：始终检查文件流状态

- 错误示例：
```cpp
// 二进制文件读取时大小计算错误
int data[10];
file.read(reinterpret_cast<char*>(data), 10);  // 错误：应该是sizeof(int)*10
```
- 原因：字节数计算错误
- 修正方案：使用sizeof计算正确字节数

## 5. 拓展补充
- 关联知识点：流操作、异常处理、内存管理、字符串处理
- 进阶延伸：XML/JSON解析、数据库文件操作、网络文件传输、压缩文件处理