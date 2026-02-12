# C++进阶特性（基础篇）

## 第一部分：STL标准库

### 核心概念
STL（Standard Template Library）是C++标准库的核心部分，提供通用数据结构和算法。

### 主要组件
- **容器**：存储数据的模板类
- **算法**：操作容器中数据的函数模板
- **迭代器**：连接容器和算法的桥梁

### 代码示例
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

void stlDemo() {
    std::vector<int> nums = {5, 2, 8, 1, 9};
    
    // 排序
    std::sort(nums.begin(), nums.end());
    
    // 查找
    auto it = std::find(nums.begin(), nums.end(), 8);
    
    // 遍历
    for (int num : nums) {
        std::cout << num << " ";
    }
}
```

## 第二部分：容器（vector, map, set等）

### 序列容器
- **vector**：动态数组，支持快速随机访问
- **list**：双向链表，插入删除高效
- **deque**：双端队列，两端操作高效

### 关联容器
- **map**：键值对集合，按键排序
- **set**：唯一元素集合，自动排序
- **unordered_map**：哈希表实现的map

### 代码示例
```cpp
#include <map>
#include <set>
#include <unordered_map>

void containerDemo() {
    // map使用
    std::map<std::string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    
    // set使用
    std::set<int> uniqueNumbers = {1, 2, 2, 3, 3, 3};
    
    // unordered_map使用
    std::unordered_map<std::string, int> fastLookup;
    fastLookup["key"] = 100;
}
```

## 第三部分：迭代器与算法

### 迭代器类型
- **输入迭代器**：只读，单向
- **输出迭代器**：只写，单向
- **前向迭代器**：读写，单向
- **双向迭代器**：读写，双向移动
- **随机访问迭代器**：读写，任意位置访问

### 常用算法
- **排序算法**：sort, stable_sort
- **查找算法**：find, binary_search
- **数值算法**：accumulate, inner_product

### 代码示例
```cpp
#include <algorithm>
#include <numeric>

void algorithmDemo() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    
    // 累加
    int sum = std::accumulate(data.begin(), data.end(), 0);
    
    // 变换
    std::vector<int> squared;
    std::transform(data.begin(), data.end(), 
                   std::back_inserter(squared),
                   [](int x) { return x * x; });
}
```

## 第四部分：Lambda表达式

### 基本语法
```cpp
[捕获列表](参数列表) -> 返回类型 { 函数体 }
```

### 捕获方式
- **值捕获**：[x] - 捕获x的副本
- **引用捕获**：[&x] - 捕获x的引用
- **隐式捕获**：[=] 或 [&]

### 代码示例
```cpp
#include <vector>
#include <algorithm>

void lambdaDemo() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // 使用lambda过滤偶数
    numbers.erase(
        std::remove_if(numbers.begin(), numbers.end(),
                      [](int n) { return n % 2 == 0; }),
        numbers.end()
    );
    
    // 带捕获的lambda
    int threshold = 3;
    auto count = std::count_if(numbers.begin(), numbers.end(),
                              [threshold](int n) { return n > threshold; });
}
```

## 综合应用示例

### 学生成绩管理系统
```cpp
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

struct Student {
    std::string name;
    int score;
    
    Student(std::string n, int s) : name(n), score(s) {}
    
    bool operator<(const Student& other) const {
        return score > other.score; // 按分数降序
    }
};

class GradeManager {
private:
    std::vector<Student> students;
    
public:
    void addStudent(const std::string& name, int score) {
        students.emplace_back(name, score);
    }
    
    void sortByScore() {
        std::sort(students.begin(), students.end());
    }
    
    void printTopN(int n) {
        auto top = students;
        std::sort(top.begin(), top.end());
        
        std::cout << "前" << n << "名学生：" << std::endl;
        for (int i = 0; i < std::min(n, (int)top.size()); ++i) {
            std::cout << i+1 << ". " << top[i].name 
                      << ": " << top[i].score << std::endl;
        }
    }
    
    double averageScore() {
        if (students.empty()) return 0.0;
        
        int sum = std::accumulate(students.begin(), students.end(), 0,
                                [](int total, const Student& s) {
                                    return total + s.score;
                                });
        return static_cast<double>(sum) / students.size();
    }
};

int main() {
    GradeManager manager;
    
    manager.addStudent("Alice", 95);
    manager.addStudent("Bob", 87);
    manager.addStudent("Charlie", 92);
    manager.addStudent("David", 78);
    
    manager.printTopN(3);
    std::cout << "平均分: " << manager.averageScore() << std::endl;
    
    return 0;
}
```

## 关键要点总结

1. **STL设计理念**：泛型编程，算法与数据分离
2. **容器选择原则**：根据访问模式选择合适容器
3. **迭代器一致性**：所有STL算法通过迭代器操作容器
4. **Lambda优势**：简洁的匿名函数，方便函数式编程
5. **性能考虑**：理解各容器和算法的时间复杂度