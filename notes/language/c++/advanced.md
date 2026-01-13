# C++高级特性与实战

## 模板编程基础

### 模板的概念与泛型编程

**模板设计原理：** 模板是C++实现泛型编程的核心机制，允许编写可处理多种数据类型的代码而无需为每种类型重复编写。编译器在编译时根据实际类型参数生成具体的代码实例，实现类型安全的代码复用。

```cpp
#include <iostream>
#include <string>
using namespace std;

// 1. 函数模板：通用算法实现
template<typename T>
T getMax(T a, T b) {
    return (a > b) ? a : b;
}

// 模板实例化过程（编译器视角）：
// 调用getMax(5, 10) → 编译器生成int版本：
// int getMax(int a, int b) { return (a > b) ? a : b; }
// 调用getMax(3.14, 2.71) → 编译器生成double版本

// 2. 类模板：通用数据结构
template<typename T>
class MyArray {
private:
    T* data;
    int size;
    
public:
    // 构造函数：动态分配内存
    MyArray(int size) : size(size) {
        data = new T[size];
    }
    
    // 析构函数：释放资源
    ~MyArray() {
        delete[] data;
    }
    
    // 拷贝构造函数（深拷贝）
    MyArray(const MyArray& other) : size(other.size) {
        data = new T[size];
        for (int i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }
    
    // 移动构造函数（C++11）
    MyArray(MyArray&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    // 下标运算符重载
    T& operator[](int index) {
        if (index < 0 || index >= size) {
            throw out_of_range("索引越界");
        }
        return data[index];
    }
    
    // 获取大小
    int getSize() const { return size; }
    
    // 显示数组内容
    void display() const {
        cout << "[";
        for (int i = 0; i < size; i++) {
            cout << data[i];
            if (i < size - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
};

// 3. 成员函数模板：更灵活的类设计
template<typename ContainerType>
class DataProcessor {
private:
    ContainerType data;
    
public:
    // 构造函数：接受任意容器类型
    DataProcessor(const ContainerType& container) : data(container) {}
    
    // 成员函数模板：处理任意类型的元素
    template<typename ElementType>
    void processWith(ElementType modifier) {
        for (auto& item : data) {
            item = modifier(item);  // 应用修改函数
        }
    }
    
    // 显示处理结果
    void showResults() const {
        cout << "处理结果: ";
        for (const auto& item : data) {
            cout << item << " ";
        }
        cout << endl;
    }
};

// 4. 模板特化：为特定类型提供特殊实现
// 通用模板版本
template<typename T>
class TypeInfo {
public:
    static string getName() {
        return "未知类型";
    }
};

// 模板特化：为int类型提供特殊实现
template<>
class TypeInfo<int> {
public:
    static string getName() {
        return "整数类型 (int)";
    }
};

// 模板特化：为double类型
template<>
class TypeInfo<double> {
public:
    static string getName() {
        return "浮点数类型 (double)";
    }
};

// 模板特化：为string类型
template<>
class TypeInfo<string> {
public:
    static string getName() {
        return "字符串类型 (string)";
    }
};

// 5. 可变参数模板（C++11+）：处理任意数量的参数
template<typename... Args>
void printAll(Args... args) {
    // 使用折叠表达式（C++17）展开参数包
    (cout << ... << args) << endl;
}

// 递归方式处理参数包（C++11/14风格）
template<typename T>
void printRecursive(T first) {
    cout << first << endl;
}

template<typename T, typename... Rest>
void printRecursive(T first, Rest... rest) {
    cout << first << " ";
    printRecursive(rest...);
}

int main() {
    // 函数模板使用
    cout << "最大值测试:" << endl;
    cout << "max(5, 10) = " << getMax(5, 10) << endl;
    cout << "max(3.14, 2.71) = " << getMax(3.14, 2.71) << endl;
    cout << "max('a', 'z') = " << getMax('a', 'z') << endl;
    
    // 类模板使用
    cout << "\n自定义数组测试:" << endl;
    MyArray<int> intArray(5);
    for (int i = 0; i < 5; i++) {
        intArray[i] = i * 10;
    }
    intArray.display();
    
    MyArray<string> strArray(3);
    strArray[0] = "Hello";
    strArray[1] = "World";
    strArray[2] = "C++";
    strArray.display();
    
    // 成员函数模板使用
    cout << "\n数据处理器测试:" << endl;
    vector<int> numbers = {1, 2, 3, 4, 5};
    DataProcessor<vector<int>> processor(numbers);
    
    // 使用lambda表达式作为修改器
    processor.processWith([](int x) { return x * 2; });
    processor.showResults();
    
    // 模板特化使用
    cout << "\n类型信息测试:" << endl;
    cout << "int类型: " << TypeInfo<int>::getName() << endl;
    cout << "double类型: " << TypeInfo<double>::getName() << endl;
    cout << "string类型: " << TypeInfo<string>::getName() << endl;
    cout << "char类型: " << TypeInfo<char>::getName() << endl;  // 使用通用版本
    
    // 可变参数模板使用
    cout << "\n可变参数模板测试:" << endl;
    printAll(1, 2, 3, "hello", 4.5);  // 输出: 123hello4.5
    printRecursive(1, 2, 3, "hello", 4.5);  // 输出: 1 2 3 hello 4.5
    
    return 0;
}
```

**模板编程的关键特性：**

| 特性 | 作用 | 示例 |
|------|------|------|
| **类型参数化** | 代码可处理任意类型 | `template<typename T>` |
| **编译时实例化** | 根据实际类型生成代码 | 调用时自动推断类型 |
| **类型安全** | 编译时类型检查 | 避免运行时类型错误 |
| **零运行时开销** | 编译时完成所有工作 | 与手写代码性能相同 |

**模板元编程：编译时计算**

```cpp
#include <iostream>
using namespace std;

// 编译时计算阶乘
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

// 模板特化：终止条件
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// 编译时计算斐波那契数列
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

// 编译时判断素数
template<int N, int D>
struct IsPrimeHelper {
    static constexpr bool value = (N % D != 0) && IsPrimeHelper<N, D - 1>::value;
};

template<int N>
struct IsPrimeHelper<N, 1> {
    static constexpr bool value = true;
};

template<int N>
struct IsPrime {
    static constexpr bool value = IsPrimeHelper<N, N / 2>::value;
};

// 特化：1不是素数
template<>
struct IsPrime<1> {
    static constexpr bool value = false;
};

int main() {
    // 编译时计算结果（在编译时已知）
    cout << "编译时计算:" << endl;
    cout << "5! = " << Factorial<5>::value << endl;        // 120
    cout << "斐波那契第10项 = " << Fibonacci<10>::value << endl;  // 55
    cout << "17是素数: " << (IsPrime<17>::value ? "是" : "否") << endl;
    cout << "15是素数: " << (IsPrime<15>::value ? "是" : "否") << endl;
    
    // 这些值在编译时就已经计算完成
    constexpr int fact_5 = Factorial<5>::value;  // 编译时常量
    constexpr int fib_10 = Fibonacci<10>::value;
    
    return 0;
}
```

**模板编程的最佳实践：**

1. **接口设计**：设计通用的模板接口，支持多种类型
2. **约束检查**：使用SFINAE或C++20概念约束模板参数
3. **性能优化**：利用编译时计算减少运行时开销
4. **错误处理**：提供清晰的编译错误信息
5. **代码组织**：合理组织模板代码，避免编译时间过长

模板编程是C++泛型编程的基础，STL标准库正是建立在模板技术之上。通过掌握模板编程，可以编写出更加通用、高效和类型安全的C++代码。

## 模板高级特性与STL深度应用

### 模板特化与偏特化：为特定类型定制行为

**模板特化原理：** 模板特化允许为特定的类型参数提供特殊的实现，实现编译时多态和类型特定的优化。

```cpp
#include <iostream>
#include <type_traits>
#include <vector>
#include <string>
using namespace std;

// 通用模板版本
template<typename T>
class TypeTraits {
public:
    static const char* name() {
        return "unknown type";
    }
    
    static bool is_numeric() {
        return false;
    }
    
    static T zero() {
        return T();
    }
};

// 模板特化：为int类型
template<>
class TypeTraits<int> {
public:
    static const char* name() {
        return "int";
    }
    
    static bool is_numeric() {
        return true;
    }
    
    static int zero() {
        return 0;
    }
};

// 模板特化：为double类型
template<>
class TypeTraits<double> {
public:
    static const char* name() {
        return "double";
    }
    
    static bool is_numeric() {
        return true;
    }
    
    static double zero() {
        return 0.0;
    }
};

// 模板特化：为string类型
template<>
class TypeTraits<string> {
public:
    static const char* name() {
        return "string";
    }
    
    static bool is_numeric() {
        return false;
    }
    
    static string zero() {
        return "";
    }
};

// 偏特化：指针类型的通用特化
template<typename T>
class TypeTraits<T*> {
public:
    static const char* name() {
        return "pointer";
    }
    
    static bool is_numeric() {
        return false;
    }
    
    static T* zero() {
        return nullptr;
    }
};

// 函数模板特化示例
template<typename T>
void process(const T& value) {
    cout << "通用处理: " << value << endl;
}

// 函数模板特化：为string类型
template<>
void process<string>(const string& value) {
    cout << "字符串处理: '" << value << "' (长度: " << value.length() << ")" << endl;
}

// 函数模板特化：为vector类型
template<typename T>
void process<vector<T>>(const vector<T>& vec) {
    cout << "向量处理: [";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << vec[i];
        if (i < vec.size() - 1) cout << ", ";
    }
    cout << "] (大小: " << vec.size() << ")" << endl;
}

// 使用SFINAE（替换失败不是错误）实现条件编译
template<typename T>
typename enable_if<is_arithmetic<T>::value, T>::type
safe_divide(T a, T b) {
    if (b == 0) throw invalid_argument("除数不能为零");
    return a / b;
}

template<typename T>
typename enable_if<!is_arithmetic<T>::value, T>::type
safe_divide(T a, T b) {
    throw invalid_argument("非算术类型不支持除法");
}

// 可变参数模板与完美转发
template<typename... Args>
void log_all(Args&&... args) {
    // 使用折叠表达式（C++17）
    (cout << ... << forward<Args>(args)) << endl;
}

template<typename T, typename... Rest>
void print_with_separator(const T& first, const Rest&... rest) {
    cout << first;
    // 递归展开参数包
    ((cout << ", " << rest), ...);
    cout << endl;
}

// 模板元编程：编译时计算
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

// 编译时字符串处理
template<char... Chars>
struct CharSequence {
    static constexpr char value[] = {Chars..., '\0'};
    static constexpr size_t size = sizeof...(Chars);
};

// 使用用户定义字面量创建编译时字符串
template<typename T, T... Chars>
constexpr CharSequence<Chars...> operator""_cs() {
    return {};
}

int main() {
    cout << "=== 模板特化演示 ===" << endl;
    
    // 类型特性测试
    cout << "int类型: " << TypeTraits<int>::name() 
         << ", 数值类型: " << (TypeTraits<int>::is_numeric() ? "是" : "否") 
         << ", 零值: " << TypeTraits<int>::zero() << endl;
    
    cout << "string类型: " << TypeTraits<string>::name() 
         << ", 数值类型: " << (TypeTraits<string>::is_numeric() ? "是" : "否") 
         << ", 零值: '" << TypeTraits<string>::zero() << "'" << endl;
    
    int* ptr = nullptr;
    cout << "int*类型: " << TypeTraits<decltype(ptr)>::name() << endl;
    
    // 函数模板特化测试
    process(42);
    process(3.14);
    process(string("Hello"));
    process(vector<int>{1, 2, 3, 4, 5});
    
    // SFINAE测试
    cout << "10 / 2 = " << safe_divide(10, 2) << endl;
    // cout << safe_divide(string("a"), string("b")) << endl;  // 编译错误
    
    // 可变参数模板测试
    log_all("日志", ": ", "值=", 42, ", 时间=", 3.14);
    print_with_separator(1, 2, 3, "a", "b", "c");
    
    // 模板元编程测试
    cout << "斐波那契数列:" << endl;
    cout << "F(0) = " << Fibonacci<0>::value << endl;
    cout << "F(1) = " << Fibonacci<1>::value << endl;
    cout << "F(5) = " << Fibonacci<5>::value << endl;
    cout << "F(10) = " << Fibonacci<10>::value << endl;
    
    // 编译时常量
    constexpr auto seq = "hello"_cs;
    cout << "编译时字符串: " << decltype(seq)::value 
         << " (长度: " << decltype(seq)::size << ")" << endl;
    
    return 0;
}
```

### STL容器深度解析与实战应用

**STL容器分类与特性对比：**

| 容器类型 | 数据结构 | 访问方式 | 插入/删除 | 内存 | 适用场景 |
|----------|----------|----------|-----------|------|----------|
| **vector** | 动态数组 | 随机访问O(1) | 末尾O(1)，中间O(n) | 连续 | 频繁随机访问 |
| **deque** | 双端队列 | 随机访问O(1) | 两端O(1)，中间O(n) | 分段连续 | 两端操作频繁 |
| **list** | 双向链表 | 顺序访问O(n) | 任意位置O(1) | 非连续 | 频繁插入删除 |
| **forward_list** | 单向链表 | 顺序访问O(n) | 任意位置O(1) | 非连续 | 内存敏感场景 |
| **array** | 静态数组 | 随机访问O(1) | 不支持 | 连续 | 固定大小数组 |
| **set/multiset** | 红黑树 | 查找O(log n) | 插入删除O(log n) | 非连续 | 有序唯一元素 |
| **map/multimap** | 红黑树 | 查找O(log n) | 插入删除O(log n) | 非连续 | 键值对映射 |
| **unordered_set** | 哈希表 | 平均O(1) | 平均O(1) | 非连续 | 快速查找去重 |
| **unordered_map** | 哈希表 | 平均O(1) | 平均O(1) | 非连续 | 快速键值查找 |

```cpp
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
using namespace std;
using namespace chrono;

// 性能测试工具类
class Timer {
private:
    time_point<high_resolution_clock> start_time;
    string operation_name;
    
public:
    Timer(const string& name) : operation_name(name) {
        start_time = high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        cout << operation_name << " 耗时: " << duration.count() << "微秒" << endl;
    }
};

// 自定义比较函数（用于set/map）
struct CaseInsensitiveCompare {
    bool operator()(const string& a, const string& b) const {
        return lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return tolower(c1) < tolower(c2); }
        );
    }
};

// 自定义哈希函数（用于unordered容器）
struct StringHash {
    size_t operator()(const string& s) const {
        // 简单的FNV-1a哈希算法
        size_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<size_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

// 自定义相等比较（用于unordered容器）
struct StringEqual {
    bool operator()(const string& a, const string& b) const {
        if (a.length() != b.length()) return false;
        for (size_t i = 0; i < a.length(); i++) {
            if (tolower(a[i]) != tolower(b[i])) return false;
        }
        return true;
    }
};

void demonstrateVector() {
    cout << "=== vector动态数组 ===" << endl;
    
    vector<int> vec;
    
    {
        Timer timer("vector插入10000个元素");
        for (int i = 0; i < 10000; i++) {
            vec.push_back(i);
        }
    }
    
    cout << "大小: " << vec.size() << ", 容量: " << vec.capacity() << endl;
    
    {
        Timer timer("vector随机访问");
        int sum = 0;
        for (int i = 0; i < 10000; i++) {
            sum += vec[i];  // O(1)随机访问
        }
        cout << "总和: " << sum << endl;
    }
    
    {
        Timer timer("vector中间插入");
        vec.insert(vec.begin() + 5000, -1);  // O(n)操作
    }
    
    // 容量管理
    vec.shrink_to_fit();
    cout << "收缩后容量: " << vec.capacity() << endl;
    
    vec.reserve(20000);
    cout << "预留后容量: " << vec.capacity() << endl;
}

void demonstrateDeque() {
    cout << "\n=== deque双端队列 ===" << endl;
    
    deque<int> dq;
    
    {
        Timer timer("deque两端插入");
        for (int i = 0; i < 5000; i++) {
            dq.push_front(i);   // 前端插入
            dq.push_back(i);    // 后端插入
        }
    }
    
    cout << "大小: " << dq.size() << endl;
    cout << "前端: " << dq.front() << ", 后端: " << dq.back() << endl;
    
    {
        Timer timer("deque随机访问");
        int sum = 0;
        for (int i = 0; i < dq.size(); i++) {
            sum += dq[i];  // O(1)随机访问
        }
        cout << "总和: " << sum << endl;
    }
}

void demonstrateList() {
    cout << "\n=== list双向链表 ===" << endl;
    
    list<int> lst;
    
    {
        Timer timer("list插入10000个元素");
        for (int i = 0; i < 10000; i++) {
            lst.push_back(i);
        }
    }
    
    {
        Timer timer("list中间插入");
        auto it = lst.begin();
        advance(it, 5000);  // 移动到中间位置
        lst.insert(it, -1);  // O(1)插入
    }
    
    {
        Timer timer("list顺序访问");
        int sum = 0;
        for (auto it = lst.begin(); it != lst.end(); ++it) {
            sum += *it;  // O(n)顺序访问
        }
        cout << "总和: " << sum << endl;
    }
    
    // list特有的操作
    lst.sort();  // 成员函数排序
    lst.unique();  // 去重
    
    list<int> lst2 = {100, 200, 300};
    lst.splice(lst.begin(), lst2);  // 拼接操作
    
    cout << "拼接后大小: " << lst.size() << endl;
}

void demonstrateAssociativeContainers() {
    cout << "\n=== 关联容器(set/map) ===" << endl;
    
    // set自动排序和去重
    set<int> unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "set内容: ";
    for (int num : unique_numbers) {
        cout << num << " ";  // 自动排序: 1 2 3 4 5 6 9
    }
    cout << endl;
    
    // map键值对存储
    map<string, int> word_count;
    word_count["apple"] = 3;
    word_count["banana"] = 2;
    word_count["cherry"] = 5;
    
    cout << "单词统计:" << endl;
    for (const auto& pair : word_count) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // 自定义比较函数的set
    set<string, CaseInsensitiveCompare> case_insensitive_set;
    case_insensitive_set.insert("Apple");
    case_insensitive_set.insert("apple");  // 不会重复插入
    case_insensitive_set.insert("BANANA");
    
    cout << "不区分大小写set: ";
    for (const auto& s : case_insensitive_set) {
        cout << s << " ";
    }
    cout << endl;
}

void demonstrateUnorderedContainers() {
    cout << "\n=== 无序容器(unordered_set/map) ===" << endl;
    
    // 默认哈希容器
    unordered_set<int> hash_set = {3, 1, 4, 1, 5, 9, 2, 6};
    cout << "默认哈希set: ";
    for (int num : hash_set) {
        cout << num << " ";  // 无序输出
    }
    cout << endl;
    
    // 自定义哈希函数的容器
    unordered_set<string, StringHash, StringEqual> custom_hash_set;
    custom_hash_set.insert("Apple");
    custom_hash_set.insert("apple");  // 不区分大小写，不会重复
    custom_hash_set.insert("BANANA");
    
    cout << "自定义哈希set: ";
    for (const auto& s : custom_hash_set) {
        cout << s << " ";
    }
    cout << endl;
    
    // 哈希表性能参数
    cout << "桶数量: " << custom_hash_set.bucket_count() << endl;
    cout << "负载因子: " << custom_hash_set.load_factor() << endl;
    cout << "最大负载因子: " << custom_hash_set.max_load_factor() << endl;
    
    // 性能优化：预分配桶
    unordered_set<int> optimized_set;
    optimized_set.reserve(1000);  // 预分配桶
    optimized_set.max_load_factor(0.7f);  // 设置最大负载因子
}

void demonstrateContainerAlgorithms() {
    cout << "\n=== 容器算法应用 ===" << endl;
    
    vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    
    // 排序算法
    sort(numbers.begin(), numbers.end());
    cout << "排序后: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // 查找算法
    auto found = find(numbers.begin(), numbers.end(), 5);
    if (found != numbers.end()) {
        cout << "找到5，位置: " << distance(numbers.begin(), found) << endl;
    }
    
    // 计数算法
    int count_5 = count(numbers.begin(), numbers.end(), 5);
    cout << "5出现次数: " << count_5 << endl;
    
    // 去重算法
    auto last = unique(numbers.begin(), numbers.end());
    numbers.erase(last, numbers.end());
    cout << "去重后: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // 变换算法
    vector<int> squared;
    transform(numbers.begin(), numbers.end(), back_inserter(squared),
              [](int x) { return x * x; });
    cout << "平方后: ";
    for (int num : squared) {
        cout << num << " ";
    }
    cout << endl;
    
    // 累加算法
    int total = accumulate(numbers.begin(), numbers.end(), 0);
    cout << "总和: " << total << endl;
}

int main() {
    demonstrateVector();
    demonstrateDeque();
    demonstrateList();
    demonstrateAssociativeContainers();
    demonstrateUnorderedContainers();
    demonstrateContainerAlgorithms();
    
    return 0;
}
```

## 标准模板库（STL）核心

### 容器：vector、map、set、unordered_map

**vector（动态数组）：连续内存的动态扩展**

**内存管理原理：** vector在堆上维护连续的动态数组，通过三个指针管理：指向数据起始的指针、指向最后一个元素之后的指针、指向分配内存末尾的指针。当空间不足时，vector会以几何增长（通常是1.5或2倍）重新分配更大的内存块。

```cpp
#include <vector>
#include <algorithm>

// vector内部实现原理（简化）
// template<typename T>
// class vector {
// private:
//     T* begin_;      // 数据起始位置
//     T* end_;        // 最后一个元素之后
//     T* capacity_;   // 分配内存的末尾
// public:
//     size_t size() const { return end_ - begin_; }
//     size_t capacity() const { return capacity_ - begin_; }
// };

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};
// 初始化时分配足够空间存储8个元素

// 添加元素：可能需要重新分配
numbers.push_back(7);  // 如果空间足够，直接构造在末尾
// 如果capacity() == size()，触发重新分配：
// 1. 分配新内存（通常是当前容量的1.5-2倍）
// 2. 将现有元素移动到新内存（拷贝或移动构造）
// 3. 在新位置构造新元素
// 4. 释放旧内存

numbers.insert(numbers.begin() + 2, 8);  // 在索引2处插入8
// 插入操作需要移动后续元素，时间复杂度O(n)

// 删除元素
numbers.pop_back();                      // 删除最后一个：O(1)
numbers.erase(numbers.begin() + 1);     // 删除索引1处的元素：需要移动后续元素

// 容量操作：理解内存分配策略
std::cout << "大小: " << numbers.size() << std::endl;      // 当前元素数量
std::cout << "容量: " << numbers.capacity() << std::endl;  // 当前分配的总空间

numbers.shrink_to_fit();                // 缩减容量到实际大小
// 请求释放未使用的内存，但实现可能选择忽略

// 性能优化技巧
numbers.reserve(100);  // 预分配空间，避免多次重新分配
for (int i = 0; i < 100; i++) {
    numbers.push_back(i);  // 在预留空间内操作，高效
}

// 对比Python的list（也是动态数组，但实现不同）
# Python: numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# Python列表也是动态数组，但采用不同的增长策略
# numbers.append(7)
# numbers.insert(2, 8)
# numbers.pop()
# del numbers[1]
```

**map（有序关联容器）：红黑树实现的键值对存储**

**数据结构原理：** std::map通常基于红黑树（一种自平衡二叉搜索树）实现。红黑树确保最坏情况下的查找、插入、删除操作都是O(log n)时间复杂度，同时保持元素按键排序。

```cpp
#include <map>

// map内部实现原理（基于红黑树）
// template<typename Key, typename Value>
// class map {
// private:
//     struct Node {
//         Key key;
//         Value value;
//         Node* left;
//         Node* right;
//         Node* parent;
//         bool color;  // 红黑树颜色
//     };
//     Node* root_;
// };

std::map<std::string, int> age_map;

// 插入元素：在红黑树中查找合适位置并插入
age_map["Alice"] = 25;  // 运算符[]：如果键不存在则插入，存在则修改
// 等价于：age_map.insert({"Alice", 25}).first->second = 25;

age_map["Bob"] = 30;
age_map.insert({"Charlie", 35});  // insert方法：返回pair<iterator, bool>

// 访问元素（自动排序）：中序遍历红黑树
for (const auto &pair : age_map) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}
// 输出按字典序排序：Alice, Bob, Charlie

// 查找元素：在红黑树中搜索
auto it = age_map.find("Alice");  // O(log n)时间复杂度
if (it != age_map.end()) {
    std::cout << "找到Alice: " << it->second << std::endl;
}

// 性能特性分析
// 插入：O(log n) - 需要重新平衡树
// 查找：O(log n) - 二分搜索
// 删除：O(log n) - 需要重新平衡
// 空间：每个节点需要额外指针和颜色信息

// 与unordered_map的对比
// map：有序，稳定性能，需要比较运算符
// unordered_map：无序，平均O(1)性能，需要哈希函数

// 自定义比较函数
struct CaseInsensitiveCompare {
    bool operator()(const std::string& a, const std::string& b) const {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); }
        );
    }
};

std::map<std::string, int, CaseInsensitiveCompare> case_insensitive_map;

// 对比Python的dict（哈希表实现，无序但平均O(1)）
# age_map = {"Alice": 25, "Bob": 30, "Charlie": 35}
# Python 3.7+ dict保持插入顺序，但本质是无序容器
# for key, value in age_map.items():
#     print(f"{key}: {value}")
```

**set（有序集合）：基于红黑树的唯一元素容器**

**设计原理：** std::set也是基于红黑树实现，但只存储键而不存储值。它保证元素唯一性和自动排序，提供高效的查找、插入和删除操作。

```cpp
#include <set>
#include <iterator>
#include <algorithm>

// set内部实现：与map类似，但节点只存储键值
// template<typename Key>
// class set {
// private:
//     struct Node {
//         Key key;
//         Node* left;
//         Node* right;
//         Node* parent;
//         bool color;
//     };
// };

std::set<int> unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6};
// 初始化时自动去重：重复的1只保留一个
// 自动排序：元素按升序排列

// 自动去重和排序：中序遍历红黑树
for (int num : unique_numbers) {
    std::cout << num << " ";  // 输出: 1 2 3 4 5 6 9（排序后）
}
std::cout << std::endl;

// 集合操作：基于有序序列的高效算法
std::set<int> set1 = {1, 2, 3, 4, 5};
std::set<int> set2 = {4, 5, 6, 7, 8};

// 并集：合并两个有序序列
std::set<int> union_set;
std::set_union(set1.begin(), set1.end(), 
               set2.begin(), set2.end(),
               std::inserter(union_set, union_set.begin()));
// 时间复杂度：O(n + m)，线性合并

// 其他集合操作
std::set<int> intersection_set;
std::set_intersection(set1.begin(), set1.end(),
                      set2.begin(), set2.end(),
                      std::inserter(intersection_set, intersection_set.begin()));
// 交集：{4, 5}

std::set<int> difference_set;
std::set_difference(set1.begin(), set1.end(),
                    set2.begin(), set2.end(),
                    std::inserter(difference_set, difference_set.begin()));
// 差集：set1 - set2 = {1, 2, 3}

// 成员函数版本的集合操作（更高效）
std::set<int> union_set2;
union_set2.insert(set1.begin(), set1.end());
union_set2.insert(set2.begin(), set2.end());
// 同样得到并集，但利用set的自动去重特性

// 性能优化：利用有序特性进行范围查询
auto lower = unique_numbers.lower_bound(3);  // 第一个>=3的元素
auto upper = unique_numbers.upper_bound(6);  // 第一个>6的元素
for (auto it = lower; it != upper; ++it) {
    std::cout << *it << " ";  // 输出3到6之间的元素
}

// 对比Python的set（基于哈希表，无序但平均O(1)）
# unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6}  # 自动去重
# Python set是无序的，但Python 3.7+保持插入顺序
# set1 = {1, 2, 3, 4, 5}
# set2 = {4, 5, 6, 7, 8}
# union_set = set1 | set2  # 并集操作
```

**unordered_map（哈希表）：平均O(1)访问的关联容器**

**哈希表原理：** unordered_map基于哈希表实现，通过哈希函数将键映射到桶(bucket)中。在理想情况下（良好哈希函数、适当负载因子），提供平均O(1)的插入、删除和查找性能。

```cpp
#include <unordered_map>

// unordered_map内部实现原理（简化）
// template<typename Key, typename Value>
// class unordered_map {
// private:
//     struct Node {
//         Key key;
//         Value value;
//         Node* next;  // 链表解决哈希冲突
//     };
//     std::vector<Node*> buckets_;  // 桶数组
//     size_t size_;                // 元素数量
//     float max_load_factor_;       // 最大负载因子
// };

std::unordered_map<std::string, int> word_count;

// 插入和计数：通过哈希函数定位桶
word_count["hello"]++;  // 1. 计算"hello"的哈希值
                         // 2. 哈希值 % 桶数量 = 桶索引
                         // 3. 在对应链表中查找/插入

word_count["world"]++;
word_count["hello"]++;  // 找到已有键，增加值

// 遍历（无序）：按照桶顺序遍历
for (const auto &pair : word_count) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}
// 输出顺序取决于哈希函数和桶分布，无保证

// 哈希表性能关键参数
std::cout << "负载因子: " << word_count.load_factor() << std::endl;
std::cout << "桶数量: " << word_count.bucket_count() << std::endl;
std::cout << "最大负载因子: " << word_count.max_load_factor() << std::endl;

// 性能优化：调整哈希表参数
word_count.reserve(100);  // 预留空间，减少重新哈希
word_count.max_load_factor(0.7f);  // 设置最大负载因子

// 自定义哈希函数（重要：避免哈希冲突）
struct StringHash {
    size_t operator()(const std::string& s) const {
        // 简单哈希函数示例（实际std::hash更复杂）
        size_t hash = 0;
        for (char c : s) {
            hash = hash * 31 + c;  // 常用质数乘法
        }
        return hash;
    }
};

struct StringEqual {
    bool operator()(const std::string& a, const std::string& b) const {
        return a == b;  // 相等性比较
    }
};

std::unordered_map<std::string, int, StringHash, StringEqual> custom_map;

// 哈希冲突处理：链表法（开放定址法在C++标准库中较少使用）
// 当多个键哈希到同一桶时，使用链表存储

// 性能分析
// 最佳情况：O(1) 所有操作
// 最坏情况：O(n) 所有键哈希到同一桶（哈希攻击）
// 平均情况：O(1) 假设良好哈希函数和适当负载因子

// 与map的对比选择
// 使用unordered_map当：需要快速查找，不关心顺序，有良好哈希函数
// 使用map当：需要有序遍历，性能稳定性重要，或键类型无良好哈希函数

// 对比Python的dict（也是哈希表实现，Python 3.6+优化了内存布局）
# word_count = {}
# word_count["hello"] = word_count.get("hello", 0) + 1
# Python dict使用更复杂的内存布局优化（紧凑字典）
```

### 迭代器：概念与使用

**迭代器基础：统一的容器访问接口**

**设计模式：** 迭代器模式提供了一种顺序访问聚合对象元素的方法，而不暴露其底层表示。C++迭代器是泛型编程的核心，将算法与容器解耦。

```cpp
#include <iterator>

std::vector<int> vec = {10, 20, 30, 40, 50};

// 迭代器本质：指针的抽象
// vector<int>::iterator 实际可能是 int* 的包装
// 但提供统一的接口，隐藏具体实现

// 使用迭代器遍历：与指针操作类似
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";  // 解引用获取值
}
std::cout << std::endl;

// 迭代器操作符重载原理
// class iterator {
// public:
//     T& operator*() { return *current_; }     // 解引用
//     iterator& operator++() { ++current_; return *this; }  // 前缀++
//     bool operator!=(const iterator& other) { return current_ != other.current_; }
// };

// 反向迭代器：适配器模式的应用
for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
    std::cout << *rit << " ";  // 50 40 30 20 10
    // 反向迭代器内部存储正向迭代器，但操作方向相反
}
std::cout << std::endl;

// 迭代器失效问题：重要概念
std::vector<int> numbers = {1, 2, 3, 4, 5};
auto it = numbers.begin() + 2;  // 指向3
numbers.push_back(6);          // 可能触发重新分配
// it可能失效！指向已释放的内存
// 解决方案：使用索引或重新获取迭代器

// 常量迭代器：只读访问
for (auto cit = vec.cbegin(); cit != vec.cend(); ++cit) {
    // *cit = 100;  // 错误：常量迭代器不能修改元素
    std::cout << *cit << " ";
}

// 迭代器适配器：功能扩展
std::vector<int> data = {1, 2, 3, 4, 5};
// 插入迭代器：将赋值操作转换为插入操作
std::copy(data.begin(), data.end(), 
          std::back_inserter(vec));  // 在vec末尾插入

// 流迭代器：连接容器与流
std::copy(vec.begin(), vec.end(),
          std::ostream_iterator<int>(std::cout, " "));

// 对比Python的迭代器协议
# Python迭代器基于__iter__和__next__方法
# for item in vec:  # 调用vec.__iter__()
#     print(item, end=" ")
# C++迭代器更接近指针，提供更细粒度的控制
```

**迭代器类别：层次化的能力模型**

**概念设计：** C++迭代器按照能力分为5个层次，每个层次提供特定的操作集合。算法根据需要的迭代器类别进行约束，实现编译时多态。

```cpp
// 迭代器类别层次（从弱到强）：

// 1. 输入迭代器：只读，单向，单遍扫描
// 典型应用：std::istream_iterator
// 支持操作：==, !=, ++, *, ->
// 只能单遍遍历，遍历后迭代器失效

template<typename InputIt>
void process_input(InputIt first, InputIt last) {
    // 只能读取，不能修改元素
    while (first != last) {
        std::cout << *first << " ";
        ++first;  // 只能向前移动
    }
}

// 2. 输出迭代器：只写，单向，单遍扫描  
// 典型应用：std::ostream_iterator, std::back_inserter
// 支持操作：++, *（仅用于赋值）
// 只能单遍写入，不能读取

template<typename OutputIt>
void generate_output(OutputIt dest, int count) {
    for (int i = 0; i < count; ++i) {
        *dest = i;    // 只能赋值，不能读取
        ++dest;
    }
}

// 3. 前向迭代器：可读写，单向，多遍扫描
// 典型容器：std::forward_list
// 支持操作：输入迭代器 + 输出迭代器的所有操作
// 可以多遍遍历，迭代器在遍历间保持有效

template<typename ForwardIt>
bool is_palindrome(ForwardIt first, ForwardIt last) {
    // 需要多遍扫描，验证回文
    auto mid = first;
    auto end = last;
    while (first != end && ++first != end) {
        ++mid;
        --end;
    }
    // 可以再次从头开始扫描
    return std::equal(first, mid, std::reverse_iterator(last));
}

// 4. 双向迭代器：可读写，双向移动
// 典型容器：std::list, std::set, std::map
// 支持操作：前向迭代器 + --（递减）
// 可以向前和向后移动

template<typename BidirIt>
void reverse_range(BidirIt first, BidirIt last) {
    while ((first != last) && (first != --last)) {
        std::iter_swap(first, last);  // 需要双向移动能力
        ++first;
    }
}

// 5. 随机访问迭代器：最强能力，直接索引访问
// 典型容器：std::vector, std::array, std::deque
// 支持操作：双向迭代器 + [], +, -, <, >等
// 可以在常数时间内跳到任意位置

template<typename RandomIt>
void quick_sort(RandomIt first, RandomIt last) {
    if (last - first > 1) {  // 随机访问迭代器支持减法
        auto pivot = first + (last - first) / 2;  // 直接计算中间位置
        // 快速排序算法需要随机访问能力
    }
}

// 迭代器类别标签：编译时类型识别
// 用于算法重载和优化
template<typename It>
void algorithm_impl(It first, It last, std::random_access_iterator_tag) {
    // 针对随机访问迭代器的优化版本
    // 可以使用[]操作符和指针算术
}

template<typename It>
void algorithm_impl(It first, It last, std::forward_iterator_tag) {
    // 针对前向迭代器的通用版本
    // 只能使用++操作符
}

// 迭代器特性：获取迭代器相关信息
template<typename It>
void print_iterator_info() {
    using iterator_category = typename std::iterator_traits<It>::iterator_category;
    using value_type = typename std::iterator_traits<It>::value_type;
    using difference_type = typename std::iterator_traits<It>::difference_type;
    
    std::cout << "值类型: " << typeid(value_type).name() << std::endl;
    // 根据迭代器类别选择最优算法
}
```

### 算法：sort、find、transform等

**常用算法示例：泛型编程的威力**

**设计哲学：** STL算法通过迭代器将算法与容器解耦，实现"一次编写，处处使用"。算法不关心容器的具体类型，只关心迭代器提供的接口。

```cpp
#include <algorithm>
#include <numeric>
#include <iterator>

std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};

// 排序：std::sort - 快速排序的泛化实现
std::sort(numbers.begin(), numbers.end());
// 内部实现：使用introspective sort（快速排序+堆排序+插入排序）
// 时间复杂度：平均O(n log n)，最坏O(n log n)
// 要求：随机访问迭代器，元素可比较（提供<运算符）

// 稳定排序：保持相等元素的相对顺序
std::stable_sort(numbers.begin(), numbers.end());
// 时间复杂度：O(n log² n) 或 O(n log n)（有额外内存时）

// 部分排序：只排序前k个元素
std::partial_sort(numbers.begin(), numbers.begin() + 3, numbers.end());
// 前3个元素有序，其余无序但都在前3个之后

// 查找：std::find - 线性搜索
auto found = std::find(numbers.begin(), numbers.end(), 5);
if (found != numbers.end()) {
    std::cout << "找到5在位置: " << std::distance(numbers.begin(), found) << std::endl;
}
// 时间复杂度：O(n)，适用于无序序列

// 二分查找：要求序列已排序
auto binary_found = std::binary_search(numbers.begin(), numbers.end(), 5);
if (binary_found) {
    std::cout << "通过二分查找找到5" << std::endl;
}
// 时间复杂度：O(log n)，但要求序列有序

// 转换：std::transform - 类似函数式编程的map
std::vector<int> squared;
std::transform(numbers.begin(), numbers.end(), 
               std::back_inserter(squared),
               [](int x) { return x * x; });
// 原理：对每个元素应用一元函数，结果写入目标区间
// 可以处理不同容器类型：vector → list, array → vector等

// 原地转换：修改原序列
std::transform(numbers.begin(), numbers.end(), numbers.begin(),
               [](int x) { return x * 2; });

// 过滤：erase-remove惯用法（C++经典模式）
numbers.erase(std::remove_if(numbers.begin(), numbers.end(),
                            [](int x) { return x % 2 == 0; }),
              numbers.end());
// 原理：remove_if将满足条件的元素移动到末尾，返回新的逻辑结尾
// erase真正删除这些元素

// C++20的简化版本（如果可用）
// std::erase_if(numbers, [](int x) { return x % 2 == 0; });

// 累加：std::accumulate - 类似函数式编程的reduce
int total = std::accumulate(numbers.begin(), numbers.end(), 0);
// 原理：从初始值0开始，对每个元素应用累加操作
// 可以自定义操作：乘法、字符串连接等

double product = std::accumulate(numbers.begin(), numbers.end(), 1.0,
                                [](double a, int b) { return a * b; });

// 其他重要算法
std::vector<int> copy_numbers;
std::copy(numbers.begin(), numbers.end(), 
          std::back_inserter(copy_numbers));  // 复制

std::reverse(numbers.begin(), numbers.end());  // 反转

auto max_it = std::max_element(numbers.begin(), numbers.end());  // 最大值
auto min_it = std::min_element(numbers.begin(), numbers.end());  // 最小值

// 算法复杂度保证
// 非修改序列操作：O(n) - find, count, for_each等
// 修改序列操作：O(n) - copy, transform, replace等  
// 排序和相关操作：O(n log n) - sort, stable_sort等
// 数值算法：O(n) - accumulate, inner_product等

// 对比Python的函数式操作
# numbers = [3, 1, 4, 1, 5, 9, 2, 6]
# numbers.sort()
# squared = list(map(lambda x: x*x, numbers))
# numbers = list(filter(lambda x: x % 2 != 0, numbers))
# total = sum(numbers)
# Python更函数式，C++ STL更注重性能和泛型
```

### lambda表达式（C++11）：匿名函数与闭包

**lambda语法：函数对象的语法糖**

**实现原理：** lambda表达式在编译时被转换为匿名函数对象（functor）。编译器根据捕获列表生成对应的类，重载函数调用运算符。

```cpp
// 基本lambda：无捕获的简单函数对象
auto square = [](int x) { return x * x; };
std::cout << square(5) << std::endl;  // 25

// 编译后等价于：
// class __lambda_1 {
// public:
//     auto operator()(int x) const { return x * x; }
// };
// __lambda_1 square;

// 捕获外部变量：创建闭包
int factor = 3;
auto multiply = [factor](int x) { return x * factor; };
// 等价于：
// class __lambda_2 {
// private:
//     int factor;  // 按值捕获的副本
// public:
//     __lambda_2(int f) : factor(f) {}
//     auto operator()(int x) const { return x * factor; }
// };
// __lambda_2 multiply(factor);

// 按引用捕获：共享外部变量
int counter = 0;
auto incrementer = [&counter]() { counter++; };
// 等价于：
// class __lambda_3 {
// private:
//     int& counter;  // 引用捕获
// public:
//     __lambda_3(int& c) : counter(c) {}
//     auto operator()() { counter++; }  // 非const，可以修改
// };

// mutable lambda：允许修改按值捕获的变量
int value = 10;
auto modifier = [value]() mutable { 
    // value是副本，可以修改但不影响外部
    value += 5; 
    return value; 
};

// 在算法中使用lambda：STL算法的完美搭档
std::vector<int> nums = {1, 2, 3, 4, 5};
std::for_each(nums.begin(), nums.end(), 
              [](int &x) { x *= 2; });
// nums变为: {2, 4, 6, 8, 10}

// 泛型lambda（C++14）：自动类型推导
auto generic_add = [](auto a, auto b) { return a + b; };
std::cout << generic_add(5, 3.14) << std::endl;  // 8.14

// 初始化捕获（C++14）：移动语义支持
std::unique_ptr<int> ptr = std::make_unique<int>(42);
auto lambda_with_move = [p = std::move(ptr)]() {
    return *p;
};

// constexpr lambda（C++17）：编译时计算
constexpr auto compile_time_square = [](int x) { return x * x; };
constexpr int result = compile_time_square(5);  // 编译时计算

// 立即调用lambda：一次性函数
int result = [](int a, int b) { return a + b; }(3, 4);  // 直接调用：7

// lambda与标准库算法的结合
std::vector<std::string> words = {"hello", "world", "cpp", "lambda"};

// 排序：按长度排序
std::sort(words.begin(), words.end(), 
          [](const std::string& a, const std::string& b) {
              return a.length() < b.length();
          });

// 查找：查找特定条件的元素
auto long_word = std::find_if(words.begin(), words.end(),
                             [](const std::string& s) {
                                 return s.length() > 5;
                             });

// 性能考虑：lambda vs 函数对象 vs 函数指针
// lambda：通常被内联，零开销抽象
// 函数对象：同样可内联，但需要显式定义类
// 函数指针：可能有间接调用开销

// 对比Python的lambda（真正的闭包，动态类型）
# square = lambda x: x * x
# factor = 3
# multiply = lambda x: x * factor  # Python闭包捕获引用
# nums = [1, 2, 3, 4, 5]
# nums = list(map(lambda x: x * 2, nums))
# Python lambda是真正的闭包，C++ lambda是编译时生成的函数对象
```

**捕获方式：**
- `[]`：不捕获任何变量
- `[=]`：按值捕获所有外部变量
- `[&]`：按引用捕获所有外部变量
- `[x, &y]`：按值捕获x，按引用捕获y
- `[this]`：捕获当前对象的this指针

## 错误处理

### 异常处理：try-catch-throw

**基本异常处理：栈展开与资源安全**

**异常机制原理：** C++异常通过栈展开(stack unwinding)实现错误传播。当异常抛出时，运行时系统沿着调用栈向上查找匹配的catch块，同时自动调用局部对象的析构函数，确保资源安全释放。

```cpp
#include <stdexcept>
#include <memory>

// 异常抛出：创建异常对象并开始栈展开
double safe_divide(double a, double b) {
    if (b == 0) {
        throw std::invalid_argument("除数不能为零");
        // 1. 构造invalid_argument异常对象
        // 2. 开始栈展开：从当前函数开始向上
        // 3. 析构局部对象（RAII确保资源释放）
        // 4. 查找匹配的catch块
    }
    return a / b;
}

// 异常安全函数示例
class Resource {
    std::unique_ptr<int> data;
public:
    Resource() : data(std::make_unique<int>(42)) {}
    ~Resource() { std::cout << "资源释放" << std::endl; }
    
    void risky_operation() {
        if (rand() % 2 == 0) {
            throw std::runtime_error("操作失败");
        }
        // 即使抛出异常，unique_ptr也会自动释放
    }
};

int main() {
    try {
        Resource res;  // RAII：异常安全的基础
        res.risky_operation();
        
        double result = safe_divide(10, 0);
        std::cout << "结果: " << result << std::endl;
    }
    // 异常捕获：按类型匹配，从具体到一般
    catch (const std::invalid_argument &e) {
        // 最具体的异常类型先匹配
        std::cerr << "数学错误: " << e.what() << std::endl;
        // e.what()返回异常描述字符串
    }
    catch (const std::exception &e) {
        // 基类异常捕获更一般的错误
        std::cerr << "一般错误: " << e.what() << std::endl;
    }
    catch (...) {
        // 捕获所有异常（包括非std::exception派生的）
        std::cerr << "未知错误" << std::endl;
        // 通常用于日志记录或紧急清理
    }
    
    return 0;
}

// 异常性能考虑
// 正常执行路径：零开销（现代编译器优化）
// 抛出异常时：有运行时开销（栈展开、类型匹配）
// 适用于罕见错误，不应用于流程控制

// 异常安全保证级别
class ExceptionSafe {
    std::vector<int> data;
public:
    // 基本保证：不泄漏资源，对象有效
    void basic_guarantee() {
        auto backup = data;  // 备份状态
        try {
            // 可能抛出异常的操作
            data.push_back(42);
        } catch (...) {
            data = std::move(backup);  // 恢复状态
            throw;  // 重新抛出
        }
    }
    
    // 强保证：操作原子性（成功或完全回滚）
    void strong_guarantee() {
        std::vector<int> new_data = data;  // 操作副本
        new_data.push_back(42);           // 在副本上操作
        data = std::move(new_data);       // 原子性提交
    }
    
    // 不抛出保证：函数声明noexcept
    void no_throw() noexcept {
        // 简单操作，保证不抛出
    }
};

// 对比Python的异常处理（更轻量级，用于流程控制）
# def safe_divide(a, b):
#     if b == 0:
#         raise ValueError("除数不能为零")
#     return a / b
# 
# try:
#     result = safe_divide(10, 0)
# except ValueError as e:
#     print(f"数学错误: {e}")
# except Exception as e:
#     print(f"一般错误: {e}")
# Python异常更常用于流程控制，C++异常更强调资源安全
```

**自定义异常：**
```cpp
class FileNotFoundException : public std::runtime_error {
public:
    FileNotFoundException(const std::string &filename)
        : std::runtime_error("文件未找到: " + filename) {}
};

void read_file(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw FileNotFoundException(filename);
    }
    // 读取文件内容...
}
```

### 异常安全与noexcept

**异常安全保证级别：**
1. **基本保证**：不泄漏资源，对象处于有效状态
2. **强保证**：操作要么成功，要么回滚到原始状态
3. **不抛出保证**：操作绝不抛出异常

**noexcept说明符：**
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

## 现代C++特性

### 自动类型推导：auto关键字

**auto使用场景：编译时类型推导的强大工具**

**推导规则：** auto使用模板参数推导规则，根据初始化表达式推导类型。它不是动态类型，而是静态类型推导，在编译时确定具体类型。

```cpp
// auto推导规则：类似于模板参数推导
// auto x = expr; 等价于：template<typename T> void f(T x); f(expr);

// 基本类型推导
auto number = 42;                    // int（字面量类型）
auto name = std::string("Alice");    // std::string（构造函数返回类型）
auto scores = std::vector<int>{90, 85, 95};  // std::vector<int>（初始化列表）

// auto推导的细节
auto x1 = 5;              // int
auto x2 = 5.0;            // double
auto x3 = 5.0f;           // float
auto x4 = "hello";        // const char[6]（数组类型）
auto x5 = std::string("hello");  // std::string

// 引用和const的推导
auto y1 = x1;             // int（值类型，复制）
auto& y2 = x1;            // int&（引用）
const auto y3 = x1;       // const int
const auto& y4 = x1;      // const int&

auto&& y5 = x1;           // int&（左值引用）
auto&& y6 = 42;           // int&&（右值引用）

// 在迭代器中使用：避免冗长的类型名称
auto it = scores.begin();            // std::vector<int>::iterator
for (auto &score : scores) {         // int&，可以修改元素
    score += 5;
}

// 在泛型编程中的威力
template<typename Container>
auto get_first(const Container& c) -> decltype(*c.begin()) {
    return *c.begin();  // 返回类型由容器元素类型决定
}

// 函数返回类型推导（C++14）
auto add(int a, int b) {      // 自动推导返回类型
    return a + b;             // 推导为int
}

// 多返回语句需要相同类型
auto calculate(int x) {
    if (x > 0) {
        return x * 1.5;  // double
    } else {
        return x * 2;    // int → 错误！类型不一致
    }
}

// 尾置返回类型（C++11）：解决复杂返回类型
auto complex_function() -> decltype(some_expression) {
    return some_expression;
}

// auto在lambda表达式中的应用（C++14）
auto lambda = [](auto x, auto y) { return x + y; };  // 泛型lambda

// 结构化绑定（C++17）：结合auto解构对象
auto [min, max] = std::minmax({3, 1, 4, 1, 5});  // min和max自动推导

// 性能考虑：auto与移动语义
auto vec = get_large_vector();  // 可能触发拷贝或移动
// 如果get_large_vector()返回右值，auto会推导为值类型，可能移动

// 最佳实践：何时使用auto
// 推荐：迭代器、lambda、模板代码、复杂类型
// 谨慎：基本类型、需要明确类型的场景

// 对比Python的动态类型（运行时类型）
# number = 42          # 运行时确定类型
# name = "Alice"       # 可以随时改变类型
# scores = [90, 85, 95] # 动态类型，灵活但可能运行时错误
# C++的auto是编译时静态类型，Python是运行时动态类型
```

**decltype类型推导：**
```cpp
int x = 10;
decltype(x) y = 20;  // y的类型与x相同（int）

auto add(int a, int b) -> decltype(a + b) {
    return a + b;  // 返回类型由a+b的类型决定
}
```

### 移动语义与右值引用（C++11）：避免不必要的拷贝

**左值 vs 右值：值类别系统的核心概念**

**值类别层次：**
- **左值(lvalue)**：有标识符，可以取地址（变量、函数名等）
- **将亡值(xvalue)**：即将被移动的资源（std::move的结果等）
- **纯右值(prvalue)**：临时对象（字面量、函数返回的临时值等）

```cpp
#include <utility>

// 值类别示例
int x = 10;           // x是左值
int& lref = x;        // 左值引用
int&& rref = 42;      // 右值引用（绑定到临时对象）

// 移动语义实现：资源所有权的转移
class String {
private:
    char *data;
    size_t length;
    
public:
    // 移动构造函数：从右值"窃取"资源
    String(String &&other) noexcept 
        : data(other.data), length(other.length) {
        // 转移资源所有权
        other.data = nullptr;    // 使源对象处于有效但空的状态
        other.length = 0;
    }
    
    // 移动赋值运算符
    String &operator=(String &&other) noexcept {
        if (this != &other) {    // 自赋值检查
            delete[] data;       // 释放当前资源
            data = other.data;   // 转移新资源
            length = other.length;
            other.data = nullptr;  // 使源对象为空
            other.length = 0;
        }
        return *this;
    }
    
    // 拷贝构造函数（对比）
    String(const String &other) 
        : data(new char[other.length + 1]), length(other.length) {
        std::copy(other.data, other.data + length + 1, data);  // 深拷贝
    }
    
    ~String() {
        delete[] data;  // 空指针delete是安全的
    }
};

// 移动语义的应用场景
String create_string() {
    String temp("Hello");
    return temp;  // 返回值优化(RVO)或移动语义
}

String s1 = create_string();  // 可能触发移动构造或RVO

// std::move：将左值转换为右值引用
String s2("World");
String s3 = std::move(s2);  // 显式移动，s2变为空

// 移动语义在STL中的应用
std::vector<String> strings;
strings.push_back(String("Hello"));  // 移动而非拷贝

// 完美转发：保持值类别
template<typename T>
void forwarder(T&& arg) {  // 万能引用
    processor(std::forward<T>(arg));  // 完美转发
}

// 移动语义的性能优势
class HeavyObject {
    std::vector<int> large_data;
public:
    // 移动构造：只需转移指针，O(1)时间复杂度
    HeavyObject(HeavyObject&& other) noexcept 
        : large_data(std::move(other.large_data)) {}
    
    // 拷贝构造：需要复制所有数据，O(n)时间复杂度
    HeavyObject(const HeavyObject& other) 
        : large_data(other.large_data) {}
};

// 移动语义的注意事项
class Resource {
    int* ptr;
public:
    Resource(Resource&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;  // 必须置空源对象
    }
    
    // 错误示例：忘记置空源对象
    // Resource(Resource&& other) : ptr(other.ptr) {}
    // 会导致双重释放！
};

// noexcept的重要性：移动操作通常标记为noexcept
// 允许STL容器在重新分配时使用移动而非拷贝
// 例如：vector在扩容时会优先移动元素（如果移动操作是noexcept）

// 对比Python的引用计数（自动内存管理）
# Python使用引用计数自动管理内存，无需手动移动
# 但C++的移动语义提供了更精细的控制和更好的性能
```

**std::move：强制转换为右值引用**
```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination = std::move(source);  // 移动而非拷贝

// source现在为空
td::cout << "source大小: " << source.size() << std::endl;      // 0
std::cout << "destination大小: " << destination.size() << std::endl;  // 5
```

### 范围for循环与初始化列表

**范围for循环（C++11）：**
```cpp
std::vector<std::string> names = {"Alice", "Bob", "Charlie"};

// 只读访问
for (const auto &name : names) {
    std::cout << name << " ";
}
std::cout << std::endl;

// 可修改访问
for (auto &name : names) {
    name += " Smith";
}

// 对比Python
# for name in names:
#     print(name, end=" ")
```

**初始化列表（C++11）：**
```cpp
// 统一初始化语法
std::vector<int> v1{1, 2, 3, 4, 5};      // 初始化列表
std::vector<int> v2 = {1, 2, 3, 4, 5};   // 复制初始化

// 类成员初始化
class Point {
public:
    int x, y;
    
    Point(int x, int y) : x(x), y(y) {}
};

Point p1{10, 20};      // 直接初始化
Point p2 = {30, 40};   // 复制初始化

// std::initializer_list使用
void print_numbers(std::initializer_list<int> nums) {
    for (int num : nums) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

print_numbers({1, 2, 3, 4, 5});  // 传递初始化列表
```

## 实战应用与项目结构

### 头文件与源文件组织：C++模块化编程的基础

**头文件（.h/.hpp）：声明接口与编译防火墙**

**设计原则：** 头文件应该只包含声明，不包含实现细节。使用头文件保护机制防止重复包含，合理组织包含关系减少编译依赖。

```cpp
// math_utils.h
#ifndef MATH_UTILS_H        // 头文件保护：防止重复包含
#define MATH_UTILS_H

// 包含必要的外部头文件（避免传递依赖）
#include <vector>           // 直接使用的标准库
#include <string>           // 函数参数或返回类型需要

// 前向声明：减少编译依赖
class Database;            // 只需要指针或引用时使用前向声明

namespace math {           // 命名空间：防止命名冲突
    
    // 函数声明：只提供接口，隐藏实现
    double calculate_average(const std::vector<double> &numbers);
    
    // 内联函数：简单函数可以在头文件中定义
    inline double square(double x) { return x * x; }
    
    // 模板必须在头文件中定义（编译时需要实例化）
    template<typename T>
    T max(const T& a, const T& b) {
        return (a > b) ? a : b;
    }
    
    // 类声明：公开接口，隐藏私有实现
    class Statistics {
    private:
        // 使用Pimpl idiom隐藏实现细节
        class Impl;
        std::unique_ptr<Impl> pimpl_;  // 实现指针
        
        std::vector<double> data;      // 简单的私有成员
        
    public:
        // 构造函数/析构函数声明
        Statistics();
        ~Statistics();
        
        // 禁用拷贝（Rule of Five）
        Statistics(const Statistics&) = delete;
        Statistics& operator=(const Statistics&) = delete;
        
        // 允许移动
        Statistics(Statistics&&) noexcept;
        Statistics& operator=(Statistics&&) noexcept;
        
        // 公开接口
        void add_data(double value);
        double mean() const;
        double variance() const;
        
        // 静态成员声明
        static constexpr double PI = 3.141592653589793;
    };
    
    // 枚举声明
    enum class Operation {
        ADD, SUBTRACT, MULTIPLY, DIVIDE
    };
    
    // 类型别名（C++11）
    using DataVector = std::vector<double>;
    
} // namespace math

#endif // MATH_UTILS_H

// 头文件设计最佳实践：
// 1. 最小化包含：只包含直接需要的头文件
// 2. 使用前向声明：减少编译依赖
// 3. 内联简单函数：避免函数调用开销
// 4. 使用命名空间：组织代码，防止冲突
// 5. Pimpl模式：隐藏实现细节，减少编译时间
```

**源文件（.cpp）：实现功能**
```cpp
// math_utils.cpp
#include "math_utils.h"
#include <cmath>
#include <numeric>

namespace math {
    double calculate_average(const std::vector<double> &numbers) {
        if (numbers.empty()) return 0.0;
        double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
        return sum / numbers.size();
    }
    
    void Statistics::add_data(double value) {
        data.push_back(value);
    }
    
    double Statistics::mean() const {
        return calculate_average(data);
    }
}
```

**对比Python的模块：**
```python
# math_utils.py
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

class Statistics:
    def __init__(self):
        self.data = []
    
    def add_data(self, value):
        self.data.append(value)
    
    def mean(self):
        return calculate_average(self.data)
```

### CMake基础配置

**基本CMakeLists.txt：**
```cmake
# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称和语言
project(MyCppProject LANGUAGES CXX)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
endif()

# 添加可执行文件
add_executable(main main.cpp math_utils.cpp)

# 添加库（如果有）
# add_library(math_utils STATIC math_utils.cpp)
# target_link_libraries(main math_utils)
```

### 常用设计模式在C++中的实现

**单例模式：**
```cpp
class Logger {
private:
    static Logger* instance;
    std::ofstream log_file;
    
    // 私有构造函数
    Logger() {
        log_file.open("app.log", std::ios::app);
    }
    
public:
    // 删除拷贝构造函数和赋值运算符
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // 获取单例实例
    static Logger& get_instance() {
        static Logger instance;  // C++11保证线程安全
        return instance;
    }
    
    void log(const std::string &message) {
        log_file << message << std::endl;
    }
};

// 使用单例
Logger::get_instance().log("应用程序启动");
```

**工厂模式：**
```cpp
class Shape {
public:
    virtual void draw() = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    void draw() override {
        std::cout << "绘制圆形" << std::endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() override {
        std::cout << "绘制矩形" << std::endl;
    }
};

class ShapeFactory {
public:
    static std::unique_ptr<Shape> create_shape(const std::string &type) {
        if (type == "circle") {
            return std::make_unique<Circle>();
        } else if (type == "rectangle") {
            return std::make_unique<Rectangle>();
        }
        return nullptr;
    }
};

// 使用工厂
auto shape = ShapeFactory::create_shape("circle");
if (shape) {
    shape->draw();
}
```

### 性能优化要点

**避免不必要的拷贝：**
```cpp
// 不良做法：不必要的拷贝
std::vector<int> process_data(std::vector<int> data) {  // 拷贝
    // 处理数据...
    return data;  // 可能再次拷贝
}

// 优化做法：使用引用和移动语义
std::vector<int> process_data(const std::vector<int> &data) {  // 引用，无拷贝
    std::vector<int> result = data;  // 只在需要时拷贝
    // 处理result...
    return result;  // 可能触发移动语义
}

// 最佳做法：原地修改
void process_data_inplace(std::vector<int> &data) {  // 引用，原地修改
    // 直接修改data...
}
```

**预分配内存：**
```cpp
std::vector<int> numbers;
numbers.reserve(1000);  // 预分配内存，避免多次重分配

for (int i = 0; i < 1000; i++) {
    numbers.push_back(i);  // 不会触发重分配
}
```

**使用emplace_back避免临时对象：**
```cpp
std::vector<std::string> names;

// 不良做法：创建临时string对象
names.push_back(std::string("Alice"));  // 创建临时对象然后移动

// 优化做法：直接构造
names.emplace_back("Alice");  // 在vector中直接构造，无临时对象
```

---

**全书总结：**

通过这三个文件的学习，您应该已经掌握了：

1. **C++基础语法**：从Python开发者角度理解C++的核心概念
2. **内存管理**：指针、引用、智能指针和RAII原则
3. **面向对象**：类、继承、多态和现代C++特性
4. **STL和高级特性**：容器、算法、lambda和移动语义
5. **实战应用**：项目组织、设计模式和性能优化

**关键转换思维：**
- Python的"动态灵活"转向C++的"静态安全"
- Python的"自动内存管理"转向C++的"精确控制"
- Python的"解释执行"转向C++的"编译优化"

**下一步学习建议：**
1. 实践小型项目，巩固所学知识
2. 学习C++模板元编程
3. 了解并发编程（多线程）
4. 探索C++20/23的新特性
5. 阅读优秀的C++开源项目代码

C++是一门强大而复杂的语言，持续实践和深入学习是掌握它的关键。

## const在模板与STL中的高级应用

### const与模板类型推导

**模板类型推导中的const规则：** 在模板编程中，const关键字会影响类型推导结果，理解这些规则对于编写正确的模板代码至关重要。

```cpp
#include <iostream>
#include <type_traits>
#include <vector>
using namespace std;

// 模板类型推导示例
template<typename T>
void deduce_type(const T& param) {
    cout << "T类型: " << typeid(T).name() << endl;
    cout << "参数类型: " << typeid(param).name() << endl;
    cout << "是否const: " << is_const<decltype(param)>::value << endl;
    cout << "---" << endl;
}

// 万能引用模板（C++11）
template<typename T>
void perfect_forwarding(T&& param) {
    cout << "万能引用推导:" << endl;
    cout << "T类型: " << typeid(T).name() << endl;
    cout << "参数类型: " << typeid(param).name() << endl;
    
    // 使用std::forward保持值类别
    process_value(forward<T>(param));
    cout << "---" << endl;
}

void process_value(int& x) {
    cout << "左值引用处理: " << x << endl;
}

void process_value(const int& x) {
    cout << "常量左值引用处理: " << x << endl;
}

void process_value(int&& x) {
    cout << "右值引用处理: " << x << endl;
}

// constexpr模板：编译时计算
template<typename T>
constexpr T pi = T(3.14159265358979323846);

template<int N>
constexpr int factorial() {
    static_assert(N >= 0, "阶乘参数必须非负");
    if constexpr (N == 0) {
        return 1;
    } else {
        return N * factorial<N - 1>();
    }
}

// const在模板特化中的应用
template<typename T>
struct TypeInfo {
    static const char* name() { return "unknown"; }
    static constexpr bool is_const_type = false;
};

// 模板特化：const类型
template<typename T>
struct TypeInfo<const T> {
    static const char* name() { 
        static string name = string("const ") + TypeInfo<T>::name();
        return name.c_str();
    }
    static constexpr bool is_const_type = true;
};

// 模板特化：指针类型
template<typename T>
struct TypeInfo<T*> {
    static const char* name() { 
        static string name = string(TypeInfo<T>::name()) + "*";
        return name.c_str();
    }
    static constexpr bool is_const_type = false;
};

// const在可变参数模板中的应用
template<typename... Args>
void log_const_types(const Args&... args) {
    cout << "参数类型信息:" << endl;
    ((cout << TypeInfo<Args>::name() << " (const: " 
          << TypeInfo<Args>::is_const_type << ")\n"), ...);
}

int main() {
    cout << "=== 模板类型推导与const ===" << endl;
    
    int x = 10;
    const int cx = 20;
    const int& crx = x;
    
    deduce_type(x);     // T = int, param = const int&
    deduce_type(cx);    // T = int, param = const int&
    deduce_type(crx);   // T = int, param = const int&
    deduce_type(30);    // T = int, param = const int&
    
    cout << "=== 完美转发与const ===" << endl;
    
    perfect_forwarding(x);     // 左值
    perfect_forwarding(cx);    // const左值
    perfect_forwarding(40);    // 右值
    
    cout << "=== 编译时常量模板 ===" << endl;
    
    constexpr double pi_double = pi<double>;
    constexpr float pi_float = pi<float>;
    constexpr int fact_5 = factorial<5>();
    
    cout << "π (double): " << pi_double << endl;
    cout << "π (float): " << pi_float << endl;
    cout << "5! = " << fact_5 << endl;
    
    cout << "=== 类型信息模板 ===" << endl;
    
    cout << "int类型: " << TypeInfo<int>::name() << endl;
    cout << "const int类型: " << TypeInfo<const int>::name() << endl;
    cout << "int*类型: " << TypeInfo<int*>::name() << endl;
    cout << "const int*类型: " << TypeInfo<const int*>::name() << endl;
    cout << "int* const类型: " << TypeInfo<int* const>::name() << endl;
    
    log_const_types(x, cx, crx, 50);
    
    return 0;
}
```

### const在STL容器中的应用

**STL容器的const正确性：** STL容器提供了完整的const支持，包括const迭代器、const引用和const成员函数。

```cpp
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>
using namespace std;

// const迭代器示例
void demonstrate_const_iterators() {
    cout << "=== const迭代器使用 ===" << endl;
    
    vector<int> numbers = {1, 2, 3, 4, 5};
    const vector<int>& const_numbers = numbers;
    
    // 普通迭代器（可修改）
    cout << "使用普通迭代器修改:" << endl;
    for (auto it = numbers.begin(); it != numbers.end(); ++it) {
        *it *= 2;  // 可以修改元素
        cout << *it << " ";
    }
    cout << endl;
    
    // const迭代器（只读）
    cout << "使用const迭代器只读访问:" << endl;
    for (auto it = const_numbers.cbegin(); it != const_numbers.cend(); ++it) {
        // *it = 10;  // 错误：const迭代器不能修改元素
        cout << *it << " ";
    }
    cout << endl;
    
    // 反向const迭代器
    cout << "反向const迭代器:" << endl;
    for (auto rit = const_numbers.crbegin(); rit != const_numbers.crend(); ++rit) {
        cout << *rit << " ";
    }
    cout << endl;
}

// const在关联容器中的应用
void demonstrate_const_associative_containers() {
    cout << "\n=== 关联容器的const使用 ===" << endl;
    
    map<string, int> student_scores = {
        {"Alice", 85},
        {"Bob", 92},
        {"Charlie", 78}
    };
    
    const map<string, int>& const_scores = student_scores;
    
    // const容器的查找操作
    auto it = const_scores.find("Alice");
    if (it != const_scores.end()) {
        cout << "找到Alice: " << it->second << endl;
        // it->second = 90;  // 错误：通过const迭代器不能修改值
    }
    
    // const容器的遍历
    cout << "所有学生成绩:" << endl;
    for (const auto& pair : const_scores) {
        cout << pair.first << ": " << pair.second << endl;
        // pair.second = 100;  // 错误：const引用不能修改
    }
    
    // set的const特性
    set<int> unique_numbers = {3, 1, 4, 1, 5, 9};
    const set<int>& const_set = unique_numbers;
    
    // set的元素本身就是const的
    auto set_it = const_set.find(4);
    if (set_it != const_set.end()) {
        cout << "找到4在set中" << endl;
        // *set_it = 10;  // 错误：set元素是const的
    }
}

// const在STL算法中的应用
void demonstrate_const_algorithms() {
    cout << "\n=== STL算法的const使用 ===" << endl;
    
    const vector<int> const_data = {1, 2, 3, 4, 5, 2, 3, 1};
    
    // 只读算法（接受const迭代器）
    int count_2 = count(const_data.cbegin(), const_data.cend(), 2);
    cout << "2出现的次数: " << count_2 << endl;
    
    auto found = find(const_data.cbegin(), const_data.cend(), 3);
    if (found != const_data.cend()) {
        cout << "找到3在位置: " << distance(const_data.cbegin(), found) << endl;
    }
    
    // 排序算法需要非const迭代器
    vector<int> mutable_data = const_data;
    sort(mutable_data.begin(), mutable_data.end());  // 正确
    // sort(const_data.begin(), const_data.end());    // 错误：const容器不能排序
    
    cout << "排序后: ";
    copy(mutable_data.cbegin(), mutable_data.cend(), 
         ostream_iterator<int>(cout, " "));
    cout << endl;
    
    // 变换算法：输入可以是const，输出必须是非const
    vector<int> squared;
    transform(const_data.cbegin(), const_data.cend(), 
              back_inserter(squared), [](int x) { return x * x; });
    
    cout << "平方结果: ";
    copy(squared.cbegin(), squared.cend(), 
         ostream_iterator<int>(cout, " "));
    cout << endl;
}

// 自定义const安全的容器包装器
template<typename Container>
class ConstSafeWrapper {
private:
    const Container& container;
    
public:
    ConstSafeWrapper(const Container& c) : container(c) {}
    
    // 只提供const接口
    auto begin() const { return container.cbegin(); }
    auto end() const { return container.cend(); }
    auto size() const { return container.size(); }
    auto empty() const { return container.empty(); }
    
    // 查找操作
    auto find(const typename Container::value_type& value) const {
        if constexpr (is_same_v<Container, set<typename Container::value_type>>) {
            return container.find(value);
        } else {
            return std::find(container.cbegin(), container.cend(), value);
        }
    }
    
    // 访问操作（只读）
    const auto& at(size_t index) const {
        if constexpr (is_same_v<Container, vector<typename Container::value_type>>) {
            return container.at(index);
        } else {
            throw logic_error("此容器不支持at操作");
        }
    }
};

int main() {
    demonstrate_const_iterators();
    demonstrate_const_associative_containers();
    demonstrate_const_algorithms();
    
    cout << "\n=== 自定义const安全包装器 ===" << endl;
    
    vector<string> words = {"apple", "banana", "cherry"};
    ConstSafeWrapper wrapper(words);
    
    cout << "包装器内容: ";
    for (const auto& word : wrapper) {
        cout << word << " ";
    }
    cout << endl;
    
    cout << "大小: " << wrapper.size() << endl;
    
    auto it = wrapper.find("banana");
    if (it != wrapper.end()) {
        cout << "找到banana" << endl;
    }
    
    return 0;
}
```

### const在函数对象和lambda中的使用

**const函数对象和lambda：** 在STL算法中使用的函数对象和lambda表达式也需要注意const正确性。

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;

// const函数对象
class ConstComparator {
private:
    int threshold;
    
public:
    ConstComparator(int t) : threshold(t) {}
    
    // const调用运算符
    bool operator()(int x) const {
        return x > threshold;
    }
    
    // 非const版本（如果需要修改状态）
    bool operator()(int x) {
        // 可以修改成员变量（如果需要）
        return x > threshold;
    }
};

// 有状态的函数对象
class Counter {
private:
    mutable int count;  // mutable允许在const函数中修改
    
public:
    Counter() : count(0) {}
    
    // const调用运算符，但可以修改mutable成员
    void operator()(int x) const {
        cout << "处理: " << x << endl;
        count++;  // 正确：mutable成员
    }
    
    int get_count() const { return count; }
};

void demonstrate_const_functors() {
    cout << "=== const函数对象使用 ===" << endl;
    
    vector<int> numbers = {1, 5, 3, 8, 2, 7, 4};
    
    // 使用const函数对象
    ConstComparator comp(4);
    int count = count_if(numbers.begin(), numbers.end(), comp);
    cout << "大于4的元素数量: " << count << endl;
    
    // 使用const函数对象引用
    const ConstComparator& const_comp = comp;
    count = count_if(numbers.begin(), numbers.end(), const_comp);
    cout << "使用const引用统计: " << count << endl;
    
    // 有状态的函数对象
    Counter counter;
    for_each(numbers.begin(), numbers.end(), ref(counter));
    cout << "处理元素数量: " << counter.get_count() << endl;
    
    cout << "\n=== const lambda表达式 ===" << endl;
    
    // 无捕获lambda默认是const的
    auto simple_lambda = [](int x) { return x * x; };
    cout << "平方计算: " << simple_lambda(5) << endl;
    
    // 按值捕获的lambda默认是const的
    int factor = 3;
    auto capture_by_value = [factor](int x) { 
        // factor = 5;  // 错误：按值捕获的lambda默认const
        return x * factor; 
    };
    
    // mutable lambda：允许修改按值捕获的变量
    auto mutable_lambda = [factor](int x) mutable {
        factor++;  // 正确：mutable lambda
        return x * factor;
    };
    cout << "mutable lambda: " << mutable_lambda(5) << endl;
    
    // const lambda与STL算法
    const vector<int> const_data = {1, 2, 3, 4, 5};
    
    // 在const上下文中使用lambda
    vector<int> result;
    transform(const_data.begin(), const_data.end(), 
              back_inserter(result),
              [](int x) { return x * 2; });  // const lambda
    
    cout << "变换结果: ";
    for (int x : result) {
        cout << x << " ";
    }
    cout << endl;
}

// const在std::function中的应用
void demonstrate_const_function() {
    cout << "\n=== std::function与const ===" << endl;
    
    // const std::function
    const function<int(int)> const_func = [](int x) { return x * x; };
    cout << "const function: " << const_func(6) << endl;
    
    // std::function可以包装const成员函数
    class Math {
    public:
        int square(int x) const { return x * x; }
        int cube(int x) const { return x * x * x; }
    };
    
    Math math;
    const Math const_math;
    
    // 绑定const成员函数
    function<int(int)> func1 = bind(&Math::square, &math, placeholders::_1);
    function<int(int)> func2 = bind(&Math::square, &const_math, placeholders::_1);
    
    cout << "绑定非const对象: " << func1(3) << endl;
    cout << "绑定const对象: " << func2(3) << endl;
}

int main() {
    demonstrate_const_functors();
    demonstrate_const_function();
    
    return 0;
}
```

**const在模板与STL中的关键总结：**

1. **类型安全**：const确保模板和STL代码的类型安全性
2. **接口清晰**：通过const明确表达函数的意图和行为
3. **性能优化**：const引用避免不必要的拷贝
4. **线程安全**：const操作通常更容易实现线程安全
5. **编译时检查**：编译器在编译时验证const正确性

通过合理使用const，可以编写出更加安全、高效和可维护的模板和STL代码。

## <span style="background-color: #ff9f43; padding: 2px 4px; border-radius: 3px; color: white;">C++异常处理</span>

### 为什么需要异常处理？

异常处理机制提供了比传统错误码更优雅的错误处理方式：

```cpp
#include <iostream>
#include <stdexcept>
#include <memory>

// 传统错误处理方式的局限性
int readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return -1; // 文件打开失败，返回错误码
    }
    // ... 读取文件内容 ...
    return 0; // 成功
}

// 异常处理方式：更清晰、更安全
void readFileWithException(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    // 正常处理文件内容
}
```

### 异常处理基本语法

```cpp
#include <iostream>
#include <stdexcept>

// 抛出异常
void riskyOperation(int value) {
    if (value < 0) {
        throw std::invalid_argument("值不能为负数");
    }
    if (value > 100) {
        throw std::out_of_range("值超出范围");
    }
    std::cout << "操作成功，值: " << value << std::endl;
}

// 捕获异常
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
```

### 标准异常类层次结构

```cpp
#include <exception>
#include <stdexcept>
#include <new>        // bad_alloc
#include <typeinfo>   // bad_cast

// 标准异常类继承关系
/*
std::exception
├── std::logic_error
│   ├── std::invalid_argument
│   ├── std::out_of_range
│   ├── std::length_error
│   └── std::domain_error
├── std::runtime_error
│   ├── std::overflow_error
│   ├── std::underflow_error
│   ├── std::range_error
│   └── std::system_error
├── std::bad_alloc
├── std::bad_cast
└── std::bad_typeid
*/

// 自定义异常类
class MyException : public std::exception {
private:
    std::string message;
    int errorCode;
    
public:
    MyException(const std::string& msg, int code = 0) 
        : message(msg), errorCode(code) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    int getErrorCode() const {
        return errorCode;
    }
};

// 使用自定义异常
try {
    throw MyException("自定义错误", 1001);
} catch (const MyException& e) {
    std::cerr << "错误代码: " << e.getErrorCode() << ", 消息: " << e.what() << std::endl;
}
```

### 异常安全保证

C++提供了三种异常安全级别：

```cpp
class ResourceManager {
private:
    std::unique_ptr<int> resource1;
    std::unique_ptr<double> resource2;
    
public:
    // 基本保证：不泄漏资源，对象处于有效状态
    void basicGuarantee(int value) {
        auto temp1 = std::make_unique<int>(value);
        auto temp2 = std::make_unique<double>(value * 1.5);
        
        // 可能抛出异常的操作
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        
        // 提交操作（无异常）
        resource1 = std::move(temp1);
        resource2 = std::move(temp2);
    }
    
    // 强保证：操作要么完全成功，要么完全失败（事务性）
    void strongGuarantee(int value) {
        auto old1 = std::move(resource1);
        auto old2 = std::move(resource2);
        
        try {
            auto temp1 = std::make_unique<int>(value);
            auto temp2 = std::make_unique<double>(value * 1.5);
            
            if (value < 0) {
                throw std::invalid_argument("值不能为负数");
            }
            
            resource1 = std::move(temp1);
            resource2 = std::move(temp2);
        } catch (...) {
            // 回滚操作
            resource1 = std::move(old1);
            resource2 = std::move(old2);
            throw; // 重新抛出异常
        }
    }
    
    // 不抛出保证：函数承诺不会抛出异常
    int noThrowGuarantee() noexcept {
        return resource1 ? *resource1 : 0;
    }
};
```

### RAII与异常安全

资源获取即初始化（RAII）是C++异常安全的关键：

```cpp
#include <fstream>
#include <memory>
#include <vector>

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

### 异常处理最佳实践

```cpp
#include <iostream>
#include <stdexcept>
#include <memory>

class Database {
public:
    void connect() {
        // 模拟数据库连接
        throw std::runtime_error("数据库连接失败");
    }
};

class Application {
private:
    std::unique_ptr<Database> db;
    
public:
    // 最佳实践1：在构造函数中抛出异常要小心
    Application() try : db(std::make_unique<Database>()) {
        db->connect();
    } catch (const std::exception& e) {
        std::cerr << "应用初始化失败: " << e.what() << std::endl;
        throw; // 重新抛出
    }
    
    // 最佳实践2：使用智能指针管理资源
    void processData() {
        auto data = std::make_unique<std::vector<int>>();
        data->push_back(1);
        data->push_back(2);
        
        // 即使这里抛出异常，data也会自动释放
        if (data->size() > 10) {
            throw std::runtime_error("数据量过大");
        }
    }
    
    // 最佳实践3：不要在析构函数中抛出异常
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
};

// 最佳实践4：异常层次化处理
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

## 迭代器与其应用

### 迭代器类型和层次

```cpp
#include <iterator>
#include <vector>
#include <list>
#include <set>
#include <iostream>

// 迭代器类别（C++20概念）
/*
输入迭代器 (Input Iterator)
├── 前向迭代器 (Forward Iterator)
    ├── 双向迭代器 (Bidirectional Iterator)
        └── 随机访问迭代器 (Random Access Iterator)
*/

void demonstrateIterators() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    // 随机访问迭代器（vector）
    auto vec_it = vec.begin();
    vec_it += 3;                    // 随机访问
    std::cout << *vec_it << std::endl; // 4
    
    // 双向迭代器（list）
    auto lst_it = lst.begin();
    ++lst_it;                       // 前向移动
    --lst_it;                       // 反向移动
    // lst_it += 3;                 // 错误：list不支持随机访问
    
    // 使用迭代器遍历
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 使用反向迭代器
    for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
        std::cout << *rit << " ";
    }
    std::cout << std::endl;
}
```

### 自定义迭代器

```cpp
#include <iterator>
#include <algorithm>
#include <iostream>

// 自定义范围迭代器
class RangeIterator {
private:
    int current;
    int end;
    int step;
    
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;
    
    RangeIterator(int start, int end_val, int step_val = 1)
        : current(start), end(end_val), step(step_val) {}
    
    // 前置递增
    RangeIterator& operator++() {
        current += step;
        return *this;
    }
    
    // 后置递增
    RangeIterator operator++(int) {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }
    
    // 解引用
    int operator*() const {
        return current;
    }
    
    // 相等比较
    bool operator==(const RangeIterator& other) const {
        return current == other.current;
    }
    
    bool operator!=(const RangeIterator& other) const {
        return !(*this == other);
    }
};

// 范围类
class Range {
private:
    int start, end, step;
    
public:
    Range(int s, int e, int st = 1) : start(s), end(e), step(st) {}
    
    RangeIterator begin() const {
        return RangeIterator(start, end, step);
    }
    
    RangeIterator end() const {
        return RangeIterator(end, end, step);
    }
};

// 使用自定义迭代器
void useCustomIterator() {
    Range r(1, 10, 2);
    for (int num : r) {
        std::cout << num << " "; // 输出: 1 3 5 7 9
    }
    std::cout << std::endl;
    
    // 与STL算法结合
    std::vector<int> result;
    std::copy(r.begin(), r.end(), std::back_inserter(result));
    
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```