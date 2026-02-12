# STL标准库

## 1. 核心概念

### 定义
- **STL（Standard Template Library）**：C++标准模板库，提供通用的数据结构和算法
- **容器**：存储和管理数据的对象集合
- **算法**：对容器中数据进行操作的函数
- **迭代器**：连接容器和算法的桥梁

### 关键特性
- **泛型编程**：模板技术实现类型无关的代码
- **组件复用**：容器、算法、迭代器可独立使用
- **性能优化**：经过高度优化的通用组件
- **标准化**：所有C++编译器都支持的跨平台库

## 2. 语法规则

### 基本语法
```cpp
#include <vector>    // 包含容器头文件
#include <algorithm> // 包含算法头文件
#include <iostream>  // 输入输出

// 使用std命名空间
using namespace std;

// 创建容器
vector<int> vec = {1, 2, 3, 4, 5};

// 使用算法
sort(vec.begin(), vec.end());

// 使用迭代器遍历
for (auto it = vec.begin(); it != vec.end(); ++it) {
    cout << *it << " ";
}
```

### 代码示例
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <functional>
using namespace std;

// STL三大组件演示：容器、算法、迭代器
void stlBasicDemo() {
    cout << "=== STL基本组件演示 ===" << endl;
    
    // 1. 容器：vector
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // 2. 算法：排序
    sort(numbers.begin(), numbers.end());
    
    // 3. 迭代器：遍历
    cout << "排序后的数字: ";
    for (auto it = numbers.begin(); it != numbers.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // 使用范围for循环（C++11）
    cout << "使用范围for循环: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
}

// 函数对象（仿函数）示例
class MultiplyBy {
private:
    int factor;
    
public:
    MultiplyBy(int f) : factor(f) {}
    
    int operator()(int x) const {
        return x * factor;
    }
};

void functionObjectDemo() {
    cout << "\\n=== 函数对象演示 ===" << endl;
    
    vector<int> nums = {1, 2, 3, 4, 5};
    
    // 使用函数对象
    MultiplyBy multiplyBy2(2);
    cout << "每个元素乘以2: ";
    for (int num : nums) {
        cout << multiplyBy2(num) << " ";
    }
    cout << endl;
    
    // 使用STL算法和函数对象
    vector<int> result(nums.size());
    transform(nums.begin(), nums.end(), result.begin(), MultiplyBy(3));
    
    cout << "使用transform乘以3: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;
}

// 适配器示例
void adapterDemo() {
    cout << "\\n=== 适配器演示 ===" << endl;
    
    vector<int> nums = {1, 2, 3, 4, 5};
    
    // 使用反向迭代器适配器
    cout << "反向遍历: ";
    for (auto it = nums.rbegin(); it != nums.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // 使用函数适配器（C++11后推荐使用lambda）
    vector<int> evenNums;
    copy_if(nums.begin(), nums.end(), back_inserter(evenNums), 
            [](int x) { return x % 2 == 0; });
    
    cout << "偶数: ";
    for (int num : evenNums) {
        cout << num << " ";
    }
    cout << endl;
}

int main() {
    stlBasicDemo();
    functionObjectDemo();
    adapterDemo();
    
    return 0;
}
```

### 注意事项
- STL组件都在std命名空间中
- 容器和算法通过迭代器连接
- 算法不直接操作容器，而是通过迭代器范围
- 大多数STL算法要求迭代器至少是前向迭代器

## 3. 常见用法

### 场景1：数据处理管道
```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>

void dataProcessingPipeline() {
    vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 数据处理管道：过滤 → 转换 → 聚合
    
    // 1. 过滤：保留偶数
    vector<int> filtered;
    copy_if(data.begin(), data.end(), back_inserter(filtered),
            [](int x) { return x % 2 == 0; });
    
    // 2. 转换：平方
    vector<int> transformed;
    transform(filtered.begin(), filtered.end(), back_inserter(transformed),
             [](int x) { return x * x; });
    
    // 3. 聚合：求和
    int sum = accumulate(transformed.begin(), transformed.end(), 0);
    
    cout << "原始数据: ";
    for (int x : data) cout << x << " ";
    cout << endl;
    
    cout << "过滤后(偶数): ";
    for (int x : filtered) cout << x << " ";
    cout << endl;
    
    cout << "转换后(平方): ";
    for (int x : transformed) cout << x << " ";
    cout << endl;
    
    cout << "聚合结果(求和): " << sum << endl;
}
```

### 场景2：自定义类型与STL
```cpp
#include <vector>
#include <algorithm>
#include <string>

class Person {
private:
    string name;
    int age;
    
public:
    Person(const string& n, int a) : name(n), age(a) {}
    
    // 用于排序的比较运算符
    bool operator<(const Person& other) const {
        return age < other.age;
    }
    
    // 用于查找的相等比较
    bool operator==(const string& n) const {
        return name == n;
    }
    
    string getName() const { return name; }
    int getAge() const { return age; }
    
    friend ostream& operator<<(ostream& os, const Person& p) {
        os << p.name << "(" << p.age << ")";
        return os;
    }
};

void customTypeWithSTL() {
    vector<Person> people = {
        Person("Alice", 25),
        Person("Bob", 30),
        Person("Charlie", 22),
        Person("Diana", 28)
    };
    
    // 使用STL算法处理自定义类型
    
    // 1. 排序（使用operator<）
    sort(people.begin(), people.end());
    cout << "按年龄排序: ";
    for (const auto& p : people) {
        cout << p << " ";
    }
    cout << endl;
    
    // 2. 查找（使用自定义比较函数）
    auto it = find(people.begin(), people.end(), "Bob");
    if (it != people.end()) {
        cout << "找到: " << *it << endl;
    }
    
    // 3. 条件查找
    it = find_if(people.begin(), people.end(),
                [](const Person& p) { return p.getAge() > 25; });
    if (it != people.end()) {
        cout << "第一个年龄大于25的人: " << *it << endl;
    }
    
    // 4. 计数
    int count = count_if(people.begin(), people.end(),
                        [](const Person& p) { return p.getAge() >= 25; });
    cout << "年龄>=25的人数: " << count << endl;
}
```

## 4. 易错点/坑

### 错误示例1：迭代器失效
```cpp
void iteratorInvalidation() {
    vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();
    
    // 在迭代过程中修改容器
    for (; it != vec.end(); ++it) {
        if (*it == 3) {
            vec.erase(it);  // 错误：迭代器失效！
        }
    }
}
```
**原因**：修改容器（如erase、insert）会使指向该容器的迭代器失效
**修正方案**：
```cpp
void safeIteratorUsage() {
    vector<int> vec = {1, 2, 3, 4, 5};
    
    // 正确方式：使用erase的返回值
    for (auto it = vec.begin(); it != vec.end(); ) {
        if (*it == 3) {
            it = vec.erase(it);  // erase返回下一个有效迭代器
        } else {
            ++it;
        }
    }
    
    // 或者使用remove-erase惯用法
    vec.erase(remove(vec.begin(), vec.end(), 3), vec.end());
}
```

### 错误示例2：错误的算法使用
```cpp
void wrongAlgorithmUsage() {
    list<int> lst = {1, 2, 3, 4, 5};
    
    // 错误：list不支持随机访问，但sort要求随机访问迭代器
    // sort(lst.begin(), lst.end());  // 编译错误
    
    vector<int> vec = {5, 3, 1, 4, 2};
    
    // 错误：binary_search要求有序序列
    bool found = binary_search(vec.begin(), vec.end(), 3);  // 未定义行为
}
```
**原因**：算法对迭代器类别有要求，使用前需了解算法前提条件
**修正方案**：
```cpp
void correctAlgorithmUsage() {
    list<int> lst = {1, 2, 3, 4, 5};
    
    // 正确：使用list的成员函数sort
    lst.sort();
    
    vector<int> vec = {5, 3, 1, 4, 2};
    
    // 正确：先排序再二分查找
    sort(vec.begin(), vec.end());
    bool found = binary_search(vec.begin(), vec.end(), 3);  // 正确
}
```

### 错误示例3：性能陷阱
```cpp
void performanceTrap() {
    vector<int> vec;
    
    // 低效：多次重新分配内存
    for (int i = 0; i < 1000000; ++i) {
        vec.push_back(i);  // 可能多次重新分配
    }
    
    // 低效：在vector前端插入
    for (int i = 0; i < 1000; ++i) {
        vec.insert(vec.begin(), i);  // O(n)操作
    }
}
```
**原因**：不了解容器特性和算法复杂度
**修正方案**：
```cpp
void performanceOptimized() {
    // 高效：预分配内存
    vector<int> vec;
    vec.reserve(1000000);  // 预分配
    
    for (int i = 0; i < 1000000; ++i) {
        vec.push_back(i);  // 无重新分配
    }
    
    // 高效：选择合适的容器
    deque<int> dq;  // 前端插入高效
    for (int i = 0; i < 1000; ++i) {
        dq.push_front(i);  // O(1)操作
    }
}
```

## 5. 拓展补充

### 关联知识点
- **迭代器类别**：输入、输出、前向、双向、随机访问迭代器
- **分配器（Allocator）**：内存管理组件
- **特征类（Traits）**：类型信息提取技术
- **SFINAE**：模板替换失败不是错误

### 进阶延伸
- **C++17新算法**：sample、clamp、gcd等
- **并行算法**：C++17引入的执行策略
- **范围库（Ranges）**：C++20引入的更简洁的算法接口
- **概念（Concepts）**：C++20的模板约束机制