# DEBUG调试技巧

## 1. 核心概念
- 定义：调试是发现和修复程序错误的过程
- 关键特性：断点设置、变量监控、调用栈跟踪、内存检查

## 2. 语法规则
- 基本语法：调试器命令、断言宏、日志输出
- 代码示例：
```cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

class DebugExample {
private:
    std::vector<int> data;
    
public:
    void addNumber(int value) {
        // 使用断言检查前置条件
        assert(value >= 0);  // 确保值非负
        data.push_back(value);
    }
    
    int getNumberAt(int index) {
        // 边界检查
        if (index < 0 || index >= data.size()) {
            throw std::out_of_range("索引超出范围");
        }
        return data[index];
    }
    
    void printData() {
        std::cout << "数据内容: ";
        for (size_t i = 0; i < data.size(); i++) {
            std::cout << data[i] << " ";
            
            // 调试输出 - 只在调试模式下生效
            #ifdef DEBUG
            std::cout << "[索引: " << i << "] ";
            #endif
        }
        std::cout << std::endl;
    }
    
    // 内存泄漏检测示例
    void memoryLeakExample() {
        int* ptr = new int[100];  // 可能的内存泄漏
        // 忘记delete[] ptr;
        
        // 使用智能指针避免内存泄漏
        auto smartPtr = std::make_unique<int[]>(100);
    }
};

// 自定义调试宏
#ifdef DEBUG
#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)  // 在发布版本中为空
#endif

int main() {
    DebugExample example;
    
    try {
        DEBUG_LOG("程序开始执行");
        
        example.addNumber(10);
        example.addNumber(20);
        example.addNumber(30);
        
        DEBUG_LOG("添加数字完成");
        
        example.printData();
        
        std::cout << "第二个数字: " << example.getNumberAt(1) << std::endl;
        
        // 测试边界条件
        // example.getNumberAt(5);  // 这会抛出异常
        
        example.memoryLeakExample();
        
        DEBUG_LOG("程序执行完成");
    }
    catch (const std::exception& e) {
        std::cerr << "捕获异常: " << e.what() << std::endl;
        #ifdef DEBUG
        // 在调试模式下显示更多信息
        std::cerr << "异常类型: " << typeid(e).name() << std::endl;
        #endif
    }
    
    return 0;
}
```
- 注意事项：调试信息安全性、性能影响、平台兼容性

## 3. 常见用法
- 场景1：GDB调试会话
```bash
# 编译带调试信息的程序
g++ -g -o debug_program debug_example.cpp

# 启动GDB
gdb debug_program

# GDB常用命令
(gdb) break main              # 在main函数设置断点
(gdb) run                     # 运行程序
(gdb) next                    # 单步执行
(gdb) step                    # 进入函数
(gdb) print variable          # 打印变量值
(gdb) backtrace              # 显示调用栈
(gdb) watch variable         # 监视变量变化
(gdb) continue               # 继续执行
(gdb) quit                   # 退出GDB
```

- 场景2：Visual Studio调试技巧
```cpp
// 条件断点示例
void processData(const std::vector<int>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        // 在Visual Studio中设置条件断点: i == 5
        if (data[i] < 0) {
            std::cout << "发现负数: " << data[i] << std::endl;
        }
    }
}

// 立即窗口调试
void debugImmediateWindow() {
    int x = 10;
    int y = 20;
    
    // 在VS立即窗口中可以输入:
    // ? x + y    // 计算表达式
    // x = 30     // 修改变量值
}
```

- 场景3：Valgrind内存检测
```bash
# 检测内存泄漏
valgrind --leak-check=full ./program

# 检测未初始化内存
valgrind --track-origins=yes ./program

# 检测线程错误
valgrind --tool=helgrind ./program
```

## 4. 易错点/坑
- 错误示例：
```cpp
// 调试代码留在发布版本中
void processSensitiveData(const std::string& password) {
    #ifdef DEBUG
    std::cout << "密码: " << password << std::endl;  // 安全隐患！
    #endif
    // ... 处理逻辑
}
```
- 原因：敏感信息泄露风险
- 修正方案：使用安全的日志记录，避免输出敏感信息

- 错误示例：
```cpp
// 断言滥用
int divide(int a, int b) {
    assert(b != 0);  // 在发布版本中会被移除
    return a / b;    // 如果b==0，发布版本会崩溃
}
```
- 原因：断言只在调试模式下生效
- 修正方案：对关键检查使用运行时检查
```cpp
int divide(int a, int b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
    }
    return a / b;
}
```

## 5. 拓展补充
- 关联知识点：异常处理、日志系统、单元测试、性能分析
- 进阶延伸：远程调试、核心转储分析、动态分析工具、代码覆盖率测试