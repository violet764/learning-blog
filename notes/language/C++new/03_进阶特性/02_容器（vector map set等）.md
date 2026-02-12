# 容器（vector/map/set等）

## 1. 核心概念

### 定义
- **容器**：存储和管理数据元素的对象
- **序列容器**：元素按线性顺序排列（vector、list、deque等）
- **关联容器**：元素按键值对存储，支持快速查找（map、set等）
- **无序容器**：基于哈希表的容器（unordered_map、unordered_set等）

### 关键特性
- **内存管理**：自动管理元素存储空间
- **元素访问**：提供多种访问方式（下标、迭代器、成员函数）
- **动态大小**：大多数容器支持动态扩容
- **性能保证**：不同容器有不同的时间复杂度保证

## 2. 语法规则

### 基本语法
```cpp
#include <vector>    // 序列容器
#include <map>       // 关联容器
#include <set>       // 集合容器
#include <unordered_map> // 无序容器

// 容器声明和初始化
vector<int> vec = {1, 2, 3};           // 向量
map<string, int> dict = {{"a", 1}, {"b", 2}};  // 映射
set<int> uniqueNums = {1, 2, 2, 3};    // 集合（自动去重）
```

### 代码示例
```cpp
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
using namespace std;

// 序列容器：vector演示
void vectorDemo() {
    cout << "=== vector演示 ===" << endl;
    
    // 创建和初始化
    vector<int> vec = {1, 2, 3, 4, 5};
    
    // 添加元素
    vec.push_back(6);
    vec.insert(vec.begin() + 2, 99);  // 在位置2插入99
    
    // 访问元素
    cout << "第一个元素: " << vec.front() << endl;
    cout << "最后一个元素: " << vec.back() << endl;
    cout << "下标访问[2]: " << vec[2] << endl;
    cout << "at访问[3]: " << vec.at(3) << endl;
    
    // 遍历
    cout << "所有元素: ";
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
    
    // 容量信息
    cout << "大小: " << vec.size() << ", 容量: " << vec.capacity() << endl;
    
    // 删除元素
    vec.pop_back();
    vec.erase(vec.begin() + 1);
    
    cout << "删除后: ";
    for (int num : vec) {
        cout << num << " ";
    }
    cout << endl;
}

// 关联容器：map演示
void mapDemo() {
    cout << "\\n=== map演示 ===" << endl;
    
    // 创建映射
    map<string, int> scores = {
        {"Alice", 85},
        {"Bob", 92},
        {"Charlie", 78}
    };
    
    // 添加元素
    scores["Diana"] = 88;
    scores.insert({"Eve", 95});
    
    // 访问元素
    cout << "Alice的分数: " << scores["Alice"] << endl;
    cout << "Bob的分数: " << scores.at("Bob") << endl;
    
    // 检查是否存在
    if (scores.find("Frank") == scores.end()) {
        cout << "Frank不在映射中" << endl;
    }
    
    // 遍历映射
    cout << "所有学生成绩:" << endl;
    for (const auto& pair : scores) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // 删除元素
    scores.erase("Charlie");
    cout << "删除Charlie后的大小: " << scores.size() << endl;
}

// 集合容器：set演示
void setDemo() {
    cout << "\\n=== set演示 ===" << endl;
    
    // 创建集合（自动去重）
    set<int> numbers = {1, 2, 2, 3, 3, 3, 4, 5};
    
    cout << "集合元素（自动去重）: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // 添加元素
    numbers.insert(6);
    numbers.insert(3);  // 重复元素不会被插入
    
    // 查找元素
    auto it = numbers.find(4);
    if (it != numbers.end()) {
        cout << "找到4: " << *it << endl;
    }
    
    // 集合操作
    set<int> set1 = {1, 2, 3, 4, 5};
    set<int> set2 = {4, 5, 6, 7, 8};
    
    // 并集
    set<int> unionSet;
    set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
              inserter(unionSet, unionSet.begin()));
    
    cout << "并集: ";
    for (int num : unionSet) {
        cout << num << " ";
    }
    cout << endl;
    
    // 交集
    set<int> intersectSet;
    set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                     inserter(intersectSet, intersectSet.begin()));
    
    cout << "交集: ";
    for (int num : intersectSet) {
        cout << num << " ";
    }
    cout << endl;
}

// 无序容器：unordered_map演示
void unorderedMapDemo() {
    cout << "\\n=== unordered_map演示 ===" << endl;
    
    // 创建无序映射（哈希表）
    unordered_map<string, int> wordCount;
    
    // 统计单词频率
    vector<string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    
    for (const string& word : words) {
        wordCount[word]++;  // 自动插入并计数
    }
    
    // 输出统计结果
    cout << "单词频率统计:" << endl;
    for (const auto& pair : wordCount) {
        cout << pair.first << ": " << pair.second << "次" << endl;
    }
    
    // 哈希表性能信息
    cout << "桶数量: " << wordCount.bucket_count() << endl;
    cout << "负载因子: " << wordCount.load_factor() << endl;
}

int main() {
    vectorDemo();
    mapDemo();
    setDemo();
    unorderedMapDemo();
    
    return 0;
}
```

### 注意事项
- 选择合适的容器基于访问模式和使用场景
- 了解不同容器的时间复杂度特性
- 注意迭代器失效规则
- 关联容器要求元素可比较或可哈希

## 3. 常见用法

### 场景1：优先级队列（使用vector和heap算法）
```cpp
#include <vector>
#include <algorithm>
#include <queue>

void priorityQueueDemo() {
    cout << "=== 优先级队列实现 ===" << endl;
    
    // 方法1：使用vector和make_heap
    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // 构建最大堆
    make_heap(nums.begin(), nums.end());
    cout << "最大堆顶: " << nums.front() << endl;
    
    // 添加元素
    nums.push_back(8);
    push_heap(nums.begin(), nums.end());
    cout << "添加8后堆顶: " << nums.front() << endl;
    
    // 弹出最大元素
    pop_heap(nums.begin(), nums.end());
    nums.pop_back();
    cout << "弹出后堆顶: " << nums.front() << endl;
    
    // 方法2：使用priority_queue（更简单）
    priority_queue<int> pq;
    for (int num : {3, 1, 4, 1, 5, 9, 2, 6}) {
        pq.push(num);
    }
    
    cout << "priority_queue内容: ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl;
}
```

### 场景2：LRU缓存实现（使用list和unordered_map）
```cpp
#include <list>
#include <unordered_map>

template<typename K, typename V>
class LRUCache {
private:
    size_t capacity;
    list<pair<K, V>> cacheList;  // 存储键值对，最近使用的在头部
    unordered_map<K, typename list<pair<K, V>>::iterator> cacheMap;  // 键到迭代器的映射
    
public:
    LRUCache(size_t cap) : capacity(cap) {}
    
    V get(K key) {
        auto it = cacheMap.find(key);
        if (it == cacheMap.end()) {
            return V();  // 返回默认值
        }
        
        // 移动到链表头部（最近使用）
        cacheList.splice(cacheList.begin(), cacheList, it->second);
        return it->second->second;
    }
    
    void put(K key, V value) {
        auto it = cacheMap.find(key);
        if (it != cacheMap.end()) {
            // 键已存在，更新值并移动到头部
            it->second->second = value;
            cacheList.splice(cacheList.begin(), cacheList, it->second);
            return;
        }
        
        // 检查容量
        if (cacheList.size() >= capacity) {
            // 删除最久未使用的元素（链表尾部）
            auto last = cacheList.back();
            cacheMap.erase(last.first);
            cacheList.pop_back();
        }
        
        // 添加新元素到头部
        cacheList.emplace_front(key, value);
        cacheMap[key] = cacheList.begin();
    }
    
    void display() const {
        cout << "LRU缓存内容（最近→最久）: ";
        for (const auto& pair : cacheList) {
            cout << pair.first << "=" << pair.second << " ";
        }
        cout << endl;
    }
};

void lruCacheDemo() {
    cout << "\\n=== LRU缓存演示 ===" << endl;
    
    LRUCache<int, string> cache(3);
    
    cache.put(1, "Apple");
    cache.put(2, "Banana");
    cache.put(3, "Cherry");
    cache.display();  // 3=Cherry 2=Banana 1=Apple
    
    cache.get(2);  // 访问2，使其成为最近使用的
    cache.display();  // 2=Banana 3=Cherry 1=Apple
    
    cache.put(4, "Date");  // 容量已满，删除最久未使用的1
    cache.display();  // 4=Date 2=Banana 3=Cherry
}
```

### 场景3：多层索引数据结构
```cpp
#include <map>
#include <set>
#include <vector>
#include <string>

// 学生课程成绩管理系统
class StudentCourseSystem {
private:
    // 学生ID到信息的映射
    map<int, string> studentInfo;  // ID -> 姓名
    
    // 课程到学生的映射（按成绩排序）
    map<string, map<int, int>> courseStudents;  // 课程 -> (学生ID -> 成绩)
    
    // 学生到课程的映射
    map<int, map<string, int>> studentCourses;  // 学生ID -> (课程 -> 成绩)
    
public:
    void addStudent(int id, const string& name) {
        studentInfo[id] = name;
    }
    
    void addGrade(int studentId, const string& course, int grade) {
        // 更新课程-学生映射
        courseStudents[course][studentId] = grade;
        
        // 更新学生-课程映射
        studentCourses[studentId][course] = grade;
    }
    
    // 获取课程的所有学生（按成绩降序）
    vector<pair<string, int>> getCourseRanking(const string& course) const {
        vector<pair<string, int>> ranking;
        
        auto it = courseStudents.find(course);
        if (it != courseStudents.end()) {
            // 使用vector进行排序（map本身按键排序，不是按值）
            vector<pair<int, int>> temp(it->second.begin(), it->second.end());
            
            // 按成绩降序排序
            sort(temp.begin(), temp.end(), 
                [](const pair<int, int>& a, const pair<int, int>& b) {
                    return a.second > b.second;
                });
            
            // 转换为姓名和成绩
            for (const auto& p : temp) {
                ranking.emplace_back(studentInfo.at(p.first), p.second);
            }
        }
        
        return ranking;
    }
    
    // 获取学生的所有课程成绩
    map<string, int> getStudentGrades(int studentId) const {
        auto it = studentCourses.find(studentId);
        if (it != studentCourses.end()) {
            return it->second;
        }
        return {};
    }
    
    // 获取平均分最高的学生
    pair<string, double> getTopStudent() const {
        string topStudent;
        double maxAvg = -1.0;
        
        for (const auto& student : studentCourses) {
            double sum = 0.0;
            for (const auto& course : student.second) {
                sum += course.second;
            }
            double avg = sum / student.second.size();
            
            if (avg > maxAvg) {
                maxAvg = avg;
                topStudent = studentInfo.at(student.first);
            }
        }
        
        return {topStudent, maxAvg};
    }
};

void studentSystemDemo() {
    cout << "\\n=== 学生课程系统演示 ===" << endl;
    
    StudentCourseSystem system;
    
    // 添加学生
    system.addStudent(1, "Alice");
    system.addStudent(2, "Bob");
    system.addStudent(3, "Charlie");
    
    // 添加成绩
    system.addGrade(1, "Math", 85);
    system.addGrade(1, "English", 92);
    system.addGrade(2, "Math", 78);
    system.addGrade(2, "English", 88);
    system.addGrade(3, "Math", 95);
    system.addGrade(3, "English", 79);
    
    // 查询
    auto mathRanking = system.getCourseRanking("Math");
    cout << "数学成绩排名:" << endl;
    for (const auto& p : mathRanking) {
        cout << p.first << ": " << p.second << endl;
    }
    
    auto aliceGrades = system.getStudentGrades(1);
    cout << "\\nAlice的成绩:" << endl;
    for (const auto& p : aliceGrades) {
        cout << p.first << ": " << p.second << endl;
    }
    
    auto topStudent = system.getTopStudent();
    cout << "\\n平均分最高学生: " << topStudent.first 
         << " (平均分: " << topStudent.second << ")" << endl;
}
```

## 4. 易错点/坑

### 错误示例1：vector的指数级扩容
```cpp
void vectorGrowthProblem() {
    vector<int> vec;
    
    // 低效：多次重新分配
    for (int i = 0; i < 1000000; ++i) {
        vec.push_back(i);  // 可能触发多次重新分配
    }
    
    // 测量重新分配次数
    size_t reallocations = 0;
    size_t lastCapacity = 0;
    
    vector<int> testVec;
    for (int i = 0; i < 100; ++i) {
        if (testVec.capacity() != lastCapacity) {
            reallocations++;
            lastCapacity = testVec.capacity();
        }
        testVec.push_back(i);
    }
    
    cout << "100次push_back触发了 " << reallocations << " 次重新分配" << endl;
}
```
**原因**：vector采用指数级扩容策略，可能造成内存浪费
**修正方案**：
```cpp
void efficientVectorUsage() {
    // 方案1：预分配内存
    vector<int> vec;
    vec.reserve(1000000);  // 一次性分配足够空间
    
    for (int i = 0; i < 1000000; ++i) {
        vec.push_back(i);  // 无重新分配
    }
    
    // 方案2：使用已知大小构造
    vector<int> vec2(1000000);  // 直接构造指定大小的vector
    for (int i = 0; i < 1000000; ++i) {
        vec2[i] = i;  // 直接赋值，无重新分配
    }
}
```

### 错误示例2：map的[]操作符副作用
```cpp
void mapBracketProblem() {
    map<string, int> wordCount;
    
    // 统计单词，但有个bug
    vector<string> words = {"apple", "banana", "apple"};
    
    for (const string& word : words) {
        // 意图：只统计已存在的单词
        if (wordCount[word] > 0) {  // 错误！[]操作符会插入新元素
            wordCount[word]++;
        }
    }
    
    // 意外地插入了所有单词，即使计数为0
    cout << "错误统计结果:" << endl;
    for (const auto& p : wordCount) {
        cout << p.first << ": " << p.second << endl;
    }
}
```
**原因**：map的[]操作符在键不存在时会插入新元素（值为默认值）
**修正方案**：
```cpp
void correctMapUsage() {
    map<string, int> wordCount;
    vector<string> words = {"apple", "banana", "apple"};
    
    for (const string& word : words) {
        // 正确方式1：使用find
        auto it = wordCount.find(word);
        if (it != wordCount.end()) {
            it->second++;
        } else {
            wordCount[word] = 1;  // 明确插入
        }
        
        // 正确方式2：直接使用[]（更简洁）
        // wordCount[word]++;  // 如果允许插入新单词
    }
    
    cout << "正确统计结果:" << endl;
    for (const auto& p : wordCount) {
        cout << p.first << ": " << p.second << endl;
    }
}
```

### 错误示例3：自定义类型的比较问题
```cpp
struct Point {
    int x, y;
    
    // 错误：没有定义比较运算符
};

void customTypeProblem() {
    set<Point> points;  // 编译错误！Point没有operator<
    
    map<Point, string> pointNames;  // 同样错误
}
```
**原因**：关联容器要求元素类型可比较（定义operator<或提供比较函数）
**修正方案**：
```cpp
struct Point {
    int x, y;
    
    // 方案1：定义比较运算符
    bool operator<(const Point& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

// 方案2：使用自定义比较函数
struct PointCompare {
    bool operator()(const Point& a, const Point& b) const {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};

void correctCustomTypeUsage() {
    // 使用operator<
    set<Point> points1;
    points1.insert({1, 2});
    points1.insert({3, 4});
    
    // 使用自定义比较函数
    set<Point, PointCompare> points2;
    points2.insert({1, 2});
    points2.insert({3, 4});
    
    // 对于unordered容器，需要哈希函数
    struct PointHash {
        size_t operator()(const Point& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1);
        }
    };
    
    struct PointEqual {
        bool operator()(const Point& a, const Point& b) const {
            return a.x == b.x && a.y == b.y;
        }
    };
    
    unordered_set<Point, PointHash, PointEqual> points3;
    points3.insert({1, 2});
}
```

## 5. 拓展补充

### 关联知识点
- **分配器（Allocator）**：自定义内存管理
- **异常安全**：容器操作的异常保证级别
- **移动语义**：C++11引入的高效元素转移
- **分配器感知**：C++11的分配器传播特性

### 进阶延伸
- **容器适配器**：stack、queue、priority_queue
- **字符串视图**：C++17的string_view
- **多集容器**：multimap、multiset
- **内存池分配器**：提高小对象分配性能