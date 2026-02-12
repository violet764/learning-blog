# C++高级特性与实战

## <span style="background-color: #1a73e8; padding: 2px 4px; border-radius: 3px; color: white;">模板编程高级特性</span>

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




## <span style="background-color: #ce2183ff; padding: 2px 4px; border-radius: 3px; color: #333;">命名空间（Namespace）</span>

### 什么是命名空间？

命名空间是C++中用于组织代码、防止命名冲突的重要机制。它将相关的类、函数、变量等封装在一个逻辑分组中，避免不同库或模块中的同名标识符产生冲突。

### 命名空间的基本语法

#### 1. 定义命名空间
```cpp
namespace MyNamespace {
    // 在命名空间内定义变量、函数、类等
    int value = 42;
    
    void myFunction() {
        std::cout << "Hello from MyNamespace!" << std::endl;
    }
    
    class MyClass {
    public:
        void display() {
            std::cout << "MyClass in MyNamespace" << std::endl;
        }
    };
}

// 命名空间可以嵌套
namespace Outer {
    namespace Inner {
        void innerFunction() {
            std::cout << "Nested namespace function" << std::endl;
        }
    }
}
```

#### 2. 使用命名空间中的成员
```cpp
#include <iostream>

namespace Math {
    const double PI = 3.14159;
    
    double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    namespace Geometry {
        struct Point {
            double x, y;
        };
        
        double distance(Point p1, Point p2) {
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            return sqrt(dx * dx + dy * dy);
        }
    }
}

int main() {
    // 方式1：使用作用域解析运算符 ::
    std::cout << "PI = " << Math::PI << std::endl;
    std::cout << "Area = " << Math::circleArea(5.0) << std::endl;
    
    Math::Geometry::Point p1 = {0, 0};
    Math::Geometry::Point p2 = {3, 4};
    std::cout << "Distance = " << Math::Geometry::distance(p1, p2) << std::endl;
    
    // 方式2：使用using声明（引入特定成员）
    using Math::PI;
    using Math::circleArea;
    
    std::cout << "PI = " << PI << std::endl;  // 直接使用，无需Math::
    std::cout << "Area = " << circleArea(3.0) << std::endl;
    
    // 方式3：使用using namespace（引入整个命名空间）
    using namespace Math::Geometry;
    
    Point p3 = {1, 1};
    Point p4 = {4, 5};
    std::cout << "Distance = " << distance(p3, p4) << std::endl;
    
    return 0;
}
```

### 标准命名空间 `std`

C++标准库中的所有组件都定义在`std`命名空间中，这是为了避免与用户定义的标识符冲突。

```cpp
#include <iostream>
#include <string>
#include <vector>

// 良好的做法：在函数内部或局部使用using
void goodPractice() {
    // 局部使用using namespace
    using namespace std;
    
    string name = "Alice";
    vector<int> numbers = {1, 2, 3};
    cout << "Name: " << name << endl;
    
    // 或者使用using声明（更安全）
    using std::string;
    using std::vector;
    using std::cout;
    using std::endl;
    
    string greeting = "Hello";
    vector<double> prices = {1.99, 2.49, 0.99};
    cout << greeting << endl;
}

// 最佳实践：明确使用std::前缀
void bestPractice() {
    std::string message = "Best practice";
    std::vector<std::string> words = {"hello", "world"};
    
    for (const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}

// 避免的做法：全局使用using namespace std
// using namespace std;  // 不推荐在全局使用

int main() {
    goodPractice();
    bestPractice();
    return 0;
}
```

### 匿名命名空间

匿名命名空间用于定义只在当前文件内可见的标识符，类似于C中的`static`关键字。

```cpp
// file1.cpp
namespace {  // 匿名命名空间
    int internalVariable = 100;  // 只在当前文件内可见
    
    void internalFunction() {
        std::cout << "Internal function" << std::endl;
    }
}

void publicFunction() {
    internalFunction();  // 可以在同一文件内访问
    std::cout << "Internal variable: " << internalVariable << std::endl;
}

// file2.cpp（另一个文件）
namespace {
    int internalVariable = 200;  // 这是不同的变量，不会冲突
}

void anotherFunction() {
    // internalFunction();  // 错误：无法访问file1.cpp中的匿名命名空间
    std::cout << "Different internal variable: " << internalVariable << std::endl;
}
```

### 命名空间别名

当命名空间名称过长时，可以创建别名来简化代码。

```cpp
#include <iostream>

namespace VeryLongNamespaceName {
    void importantFunction() {
        std::cout << "Important function" << std::endl;
    }
}

// 创建命名空间别名
namespace VLN = VeryLongNamespaceName;

// 标准库别名示例
namespace fs = std::filesystem;  // C++17文件系统别名

int main() {
    // 使用别名
    VLN::importantFunction();  // 等价于 VeryLongNamespaceName::importantFunction()
    
    // 标准库别名使用
    // fs::path filePath = "example.txt";  // C++17特性
    
    return 0;
}
```

### 内联命名空间（C++11）

内联命名空间用于版本控制，允许新旧API共存。

```cpp
#include <iostream>

namespace Library {
    // 版本1的API
    namespace v1 {
        void process(int x) {
            std::cout << "v1 processing: " << x << std::endl;
        }
    }
    
    // 版本2的API（当前版本）
    inline namespace v2 {
        void process(int x) {
            std::cout << "v2 processing: " << x * 2 << std::endl;
        }
        
        void newFeature() {
            std::cout << "New feature in v2" << std::endl;
        }
    }
}

int main() {
    // 默认使用内联命名空间（v2）
    Library::process(10);  // 调用v2::process
    Library::newFeature(); // 调用v2::newFeature
    
    // 明确指定版本
    Library::v1::process(10);  // 调用v1::process
    Library::v2::process(10);  // 调用v2::process
    
    return 0;
}
```

### 命名空间的最佳实践

#### 1. 使用建议
```cpp
// ✅ 推荐做法：明确使用std::前缀
void goodCode() {
    std::vector<std::string> names;
    std::cout << "Good practice" << std::endl;
}

// ❌ 避免做法：全局使用using namespace
// using namespace std;  // 不推荐

void badCode() {
    vector<string> names;  // 可能产生命名冲突
    cout << "Bad practice" << endl;
}

// ✅ 局部使用using声明（安全）
void safeCode() {
    using std::vector;
    using std::string;
    using std::cout;
    using std::endl;
    
    vector<string> safeNames;
    cout << "Safe practice" << endl;
}
```

#### 2. 项目中的命名空间组织
```cpp
// 大型项目中的命名空间组织示例
namespace MyCompany {
    namespace ProjectName {
        namespace Core {
            class Database {
                // 核心数据库功能
            };
        }
        
        namespace UI {
            class Window {
                // 用户界面组件
            };
        }
        
        namespace Utils {
            // 工具函数和辅助类
            template<typename T>
            class Singleton {
                // 单例模式实现
            };
        }
    }
}

// 使用示例
void useNamespaces() {
    MyCompany::ProjectName::Core::Database db;
    MyCompany::ProjectName::UI::Window window;
    
    // 使用别名简化
    namespace MP = MyCompany::ProjectName;
    MP::Core::Database anotherDb;
}
```

### 命名空间与头文件

在头文件中使用命名空间时需要特别小心：

```cpp
// mylibrary.h
#ifndef MYLIBRARY_H
#define MYLIBRARY_H

#include <string>

// ✅ 在头文件中定义命名空间是安全的
namespace MyLibrary {
    class Calculator {
    public:
        static int add(int a, int b);
        static double divide(double a, double b);
    };
    
    const std::string VERSION = "1.0.0";
}

// ❌ 不要在头文件中使用using namespace
// using namespace std;  // 绝对避免！

// ✅ 可以在头文件中使用using声明（但要谨慎）
// using std::string;    // 谨慎使用，可能影响包含该头文件的所有文件

#endif

// mylibrary.cpp
#include "mylibrary.h"

namespace MyLibrary {
    int Calculator::add(int a, int b) {
        return a + b;
    }
    
    double Calculator::divide(double a, double b) {
        if (b == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return a / b;
    }
}

// main.cpp
#include "mylibrary.h"
#include <iostream>

int main() {
    // 使用命名空间中的类
    int result = MyLibrary::Calculator::add(10, 20);
    std::cout << "10 + 20 = " << result << std::endl;
    
    std::cout << "Library version: " << MyLibrary::VERSION << std::endl;
    
    return 0;
}
```

### 总结

命名空间是C++中管理代码组织和防止命名冲突的重要工具：

1. **基本用法**：使用`namespace`关键字定义，通过`::`访问成员
2. **标准库**：所有标准库组件都在`std`命名空间中
3. **最佳实践**：避免全局`using namespace`，优先使用明确的前缀
4. **高级特性**：匿名命名空间、内联命名空间、命名空间别名
5. **头文件规则**：在头文件中谨慎使用`using`声明

通过合理使用命名空间，可以编写出更加清晰、可维护和不易冲突的C++代码。

