# 哈希表与位集

哈希表是一种基于**键值对**的数据结构，通过哈希函数将键映射到存储位置，实现 **O(1) 平均时间复杂度**的查找、插入和删除操作。位集则是一种使用**位运算**来高效存储和操作集合的数据结构，特别适合处理大规模布尔数组或状态压缩问题。

## 哈希表

### 基本概念

哈希表（Hash Table），也叫散列表，核心思想是：

1. **哈希函数**：将任意大小的键映射到固定范围的索引
2. **数组存储**：使用数组作为底层存储结构
3. **冲突处理**：解决不同键映射到相同索引的问题

```
键(Key) → 哈希函数 → 索引(Index) → 数组位置 → 值(Value)
```

### 哈希函数设计

哈希函数的质量直接影响哈希表的性能。好的哈希函数应该：

- **计算快速**：能在常数时间内完成
- **分布均匀**：将键均匀分布在哈希空间
- **确定性**：相同输入产生相同输出

#### 常见哈希函数

**1. 整数哈希**

```cpp
// C++ 常用整数哈希方法
int hashInt(int key, int tableSize) {
    // 直接取余法（适用于键分布均匀）
    return key % tableSize;
}

// 更好的整数哈希（Mersenne数）
int hashIntBetter(int key, int tableSize) {
    // 使用质数作为表大小
    return (key % tableSize + tableSize) % tableSize;  // 处理负数
}

// 乘法哈希
int hashMultiply(int key, int tableSize) {
    const double A = 0.6180339887;  // 黄金分割比
    return static_cast<int>(tableSize * (key * A - static_cast<int>(key * A)));
}
```

```python
# Python 常用整数哈希方法
def hash_int(key: int, table_size: int) -> int:
    """直接取余法"""
    return key % table_size

def hash_int_better(key: int, table_size: int) -> int:
    """处理负数情况"""
    return (key % table_size + table_size) % table_size

def hash_multiply(key: int, table_size: int) -> int:
    """乘法哈希"""
    A = 0.6180339887  # 黄金分割比
    return int(table_size * (key * A - int(key * A)))
```

**2. 字符串哈希**

```cpp
// C++ 字符串哈希 - 多项式滚动哈希
size_t hashString(const string& s, size_t tableSize) {
    const size_t BASE = 31;  // 基数，通常取质数
    size_t hash = 0;
    for (char c : s) {
        hash = (hash * BASE + c) % tableSize;
    }
    return hash;
}

// 更安全的字符串哈希 - 双哈希
pair<size_t, size_t> doubleHash(const string& s) {
    const size_t BASE1 = 31, BASE2 = 137;
    const size_t MOD1 = 1e9 + 7, MOD2 = 1e9 + 9;
    size_t h1 = 0, h2 = 0;
    for (char c : s) {
        h1 = (h1 * BASE1 + c) % MOD1;
        h2 = (h2 * BASE2 + c) % MOD2;
    }
    return {h1, h2};
}
```

```python
# Python 字符串哈希 - 多项式滚动哈希
def hash_string(s: str, table_size: int) -> int:
    """多项式滚动哈希"""
    BASE = 31
    hash_val = 0
    for c in s:
        hash_val = (hash_val * BASE + ord(c)) % table_size
    return hash_val

def double_hash(s: str) -> tuple[int, int]:
    """双哈希，降低碰撞概率"""
    BASE1, BASE2 = 31, 137
    MOD1, MOD2 = 10**9 + 7, 10**9 + 9
    h1, h2 = 0, 0
    for c in s:
        h1 = (h1 * BASE1 + ord(c)) % MOD1
        h2 = (h2 * BASE2 + ord(c)) % MOD2
    return h1, h2
```

### 冲突处理方法

当不同键映射到相同索引时，需要处理冲突。

#### 链地址法（Separate Chaining）

每个数组位置存储一个链表，冲突元素追加到链表中。

```cpp
// C++ 链地址法哈希表实现
template<typename K, typename V>
class HashMap {
private:
    struct Node {
        K key;
        V value;
        Node* next;
        Node(K k, V v) : key(k), value(v), next(nullptr) {}
    };
    
    vector<Node*> table;
    size_t size_;
    size_t capacity;
    const double LOAD_FACTOR = 0.75;
    
    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % capacity;
    }
    
    void resize() {
        size_t oldCapacity = capacity;
        capacity *= 2;
        vector<Node*> newTable(capacity, nullptr);
        
        for (size_t i = 0; i < oldCapacity; i++) {
            Node* curr = table[i];
            while (curr) {
                Node* next = curr->next;
                size_t idx = hash(curr->key);
                curr->next = newTable[idx];
                newTable[idx] = curr;
                curr = next;
            }
        }
        table = move(newTable);
    }
    
public:
    HashMap(size_t cap = 16) : capacity(cap), size_(0) {
        table.resize(capacity, nullptr);
    }
    
    void put(const K& key, const V& value) {
        if ((double)size_ / capacity > LOAD_FACTOR) {
            resize();
        }
        
        size_t idx = hash(key);
        Node* curr = table[idx];
        
        // 查找是否已存在
        while (curr) {
            if (curr->key == key) {
                curr->value = value;
                return;
            }
            curr = curr->next;
        }
        
        // 插入新节点（头插法）
        Node* newNode = new Node(key, value);
        newNode->next = table[idx];
        table[idx] = newNode;
        size_++;
    }
    
    V* get(const K& key) {
        size_t idx = hash(key);
        Node* curr = table[idx];
        while (curr) {
            if (curr->key == key) {
                return &curr->value;
            }
            curr = curr->next;
        }
        return nullptr;
    }
    
    bool remove(const K& key) {
        size_t idx = hash(key);
        Node* curr = table[idx];
        Node* prev = nullptr;
        
        while (curr) {
            if (curr->key == key) {
                if (prev) prev->next = curr->next;
                else table[idx] = curr->next;
                delete curr;
                size_--;
                return true;
            }
            prev = curr;
            curr = curr->next;
        }
        return false;
    }
    
    size_t size() const { return size_; }
};
```

```python
# Python 链地址法哈希表实现
from typing import Generic, TypeVar, Optional, List, Tuple

K = TypeVar('K')
V = TypeVar('V')

class HashMap(Generic[K, V]):
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.size = 0
        self.table: List[List[Tuple[K, V]]] = [[] for _ in range(capacity)]
        self.LOAD_FACTOR = 0.75
    
    def _hash(self, key: K) -> int:
        return hash(key) % self.capacity
    
    def _resize(self):
        """扩容"""
        old_table = self.table
        self.capacity *= 2
        self.table = [[] for _ in range(self.capacity)]
        self.size = 0
        
        for bucket in old_table:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key: K, value: V):
        """插入键值对"""
        if self.size / self.capacity > self.LOAD_FACTOR:
            self._resize()
        
        idx = self._hash(key)
        bucket = self.table[idx]
        
        # 检查是否已存在
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # 添加新键值对
        bucket.append((key, value))
        self.size += 1
    
    def get(self, key: K) -> Optional[V]:
        """获取值"""
        idx = self._hash(key)
        bucket = self.table[idx]
        for k, v in bucket:
            if k == key:
                return v
        return None
    
    def remove(self, key: K) -> bool:
        """删除键值对"""
        idx = self._hash(key)
        bucket = self.table[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        return False
    
    def __contains__(self, key: K) -> bool:
        return self.get(key) is not None
    
    def __len__(self) -> int:
        return self.size

# 使用示例
hmap = HashMap[str, int]()
hmap.put("apple", 1)
hmap.put("banana", 2)
print(hmap.get("apple"))   # 输出: 1
print("apple" in hmap)     # 输出: True
```

#### 开放地址法（Open Addressing）

冲突时探测下一个可用位置，包括：
- **线性探测**：依次查找下一个位置
- **平方探测**：按平方间隔探测
- **双重哈希**：使用第二个哈希函数确定步长

```cpp
// C++ 开放地址法哈希表（线性探测）
template<typename K, typename V>
class OpenAddressHashMap {
private:
    enum State { EMPTY, OCCUPIED, DELETED };
    
    struct Entry {
        K key;
        V value;
        State state;
        Entry() : state(EMPTY) {}
    };
    
    vector<Entry> table;
    size_t size_;
    size_t capacity;
    const double LOAD_FACTOR = 0.5;
    
    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % capacity;
    }
    
    size_t probe(size_t idx, size_t i) const {
        // 线性探测
        return (idx + i) % capacity;
    }
    
public:
    OpenAddressHashMap(size_t cap = 16) : capacity(cap), size_(0) {
        table.resize(capacity);
    }
    
    bool put(const K& key, const V& value) {
        if ((double)size_ / capacity > LOAD_FACTOR) return false;  // 需要扩容
        
        size_t idx = hash(key);
        size_t firstDeleted = capacity;
        
        for (size_t i = 0; i < capacity; i++) {
            size_t pos = probe(idx, i);
            
            if (table[pos].state == EMPTY) {
                size_t insertPos = (firstDeleted < capacity) ? firstDeleted : pos;
                table[insertPos].key = key;
                table[insertPos].value = value;
                table[insertPos].state = OCCUPIED;
                size_++;
                return true;
            }
            
            if (table[pos].state == DELETED && firstDeleted == capacity) {
                firstDeleted = pos;
            }
            
            if (table[pos].state == OCCUPIED && table[pos].key == key) {
                table[pos].value = value;
                return true;
            }
        }
        return false;
    }
    
    V* get(const K& key) {
        size_t idx = hash(key);
        for (size_t i = 0; i < capacity; i++) {
            size_t pos = probe(idx, i);
            if (table[pos].state == EMPTY) return nullptr;
            if (table[pos].state == OCCUPIED && table[pos].key == key) {
                return &table[pos].value;
            }
        }
        return nullptr;
    }
    
    bool remove(const K& key) {
        size_t idx = hash(key);
        for (size_t i = 0; i < capacity; i++) {
            size_t pos = probe(idx, i);
            if (table[pos].state == EMPTY) return false;
            if (table[pos].state == OCCUPIED && table[pos].key == key) {
                table[pos].state = DELETED;
                size_--;
                return true;
            }
        }
        return false;
    }
};
```

### 装载因子与扩容

📌 **装载因子** = 已存储元素数量 / 哈希表容量

| 冲突处理方法 | 推荐装载因子阈值 |
|------------|----------------|
| 链地址法 | 0.75 |
| 开放地址法 | 0.5 |

```cpp
// C++ 扩容策略示例
void resize() {
    size_t oldCapacity = capacity;
    capacity = capacity * 2;  // 通常翻倍
    
    // 选择新的容量为质数，可以减少冲突
    // capacity = nextPrime(capacity * 2);
    
    vector<Node*> newTable(capacity, nullptr);
    
    // 重新哈希所有元素
    for (size_t i = 0; i < oldCapacity; i++) {
        Node* curr = table[i];
        while (curr) {
            Node* next = curr->next;
            size_t newIdx = hash(curr->key);  // 使用新容量计算
            curr->next = newTable[newIdx];
            newTable[newIdx] = curr;
            curr = next;
        }
    }
    table = move(newTable);
}
```

### 时间复杂度分析

| 操作 | 平均情况 | 最坏情况 |
|------|---------|---------|
| 查找 | O(1) | O(n) |
| 插入 | O(1) | O(n) |
| 删除 | O(1) | O(n) |

⚠️ **最坏情况**：所有键都哈希到同一位置，退化为链表。

### C++ 标准库使用

#### unordered_map 和 unordered_set

```cpp
#include <iostream>
#include <unordered_map>
#include <unordered_set>
using namespace std;

int main() {
    // ========== unordered_map ==========
    unordered_map<string, int> scores;
    
    // 插入
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores.insert({"Charlie", 92});
    scores.emplace("David", 88);  // 更高效
    
    // 查找
    cout << scores["Alice"] << endl;           // 95
    cout << scores.at("Bob") << endl;          // 87（带边界检查）
    
    // 检查是否存在
    if (scores.count("Alice")) {               // count返回0或1
        cout << "Alice exists" << endl;
    }
    
    // 安全查找
    auto it = scores.find("Eve");
    if (it != scores.end()) {
        cout << it->second << endl;
    }
    
    // 遍历
    for (auto& [name, score] : scores) {
        cout << name << ": " << score << endl;
    }
    
    // 删除
    scores.erase("Bob");
    
    // ========== unordered_set ==========
    unordered_set<int> nums;
    
    nums.insert(1);
    nums.insert(2);
    nums.insert(3);
    nums.insert(1);  // 重复插入会被忽略
    
    cout << nums.size() << endl;  // 3
    
    // 检查存在
    if (nums.count(2)) {
        cout << "2 is in the set" << endl;
    }
    
    // 遍历
    for (int x : nums) {
        cout << x << " ";
    }
    
    return 0;
}
```

#### 自定义哈希函数

```cpp
// C++ 自定义类型的哈希函数
struct Person {
    string name;
    int age;
    
    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// 方法1：特化 std::hash
namespace std {
    template<>
    struct hash<Person> {
        size_t operator()(const Person& p) const {
            return hash<string>()(p.name) ^ (hash<int>()(p.age) << 1);
        }
    };
}

// 方法2：自定义哈希结构体传入模板参数
struct PersonHash {
    size_t operator()(const Person& p) const {
        size_t h1 = hash<string>{}(p.name);
        size_t h2 = hash<int>{}(p.age);
        return h1 ^ (h2 << 1);  // 组合哈希值
    }
};

// 使用方法2
unordered_map<Person, string, PersonHash> personMap;
```

### Python dict 和 set

```python
# ========== Python dict ==========
# 字典（基于哈希表实现）
scores = {}

# 插入
scores["Alice"] = 95
scores["Bob"] = 87
scores.update({"Charlie": 92, "David": 88})

# 查找
print(scores["Alice"])          # 95
print(scores.get("Eve", 0))     # 0（不存在时返回默认值）

# 检查存在
if "Alice" in scores:
    print("Alice exists")

# 安全访问
name = "Eve"
if name in scores:
    print(scores[name])

# 遍历
for name, score in scores.items():
    print(f"{name}: {score}")

for name in scores:  # 只遍历键
    print(name)

# 删除
del scores["Bob"]
removed = scores.pop("Charlie", None)  # 删除并返回值

# 字典推导式
squares = {x: x*x for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# ========== Python set ==========
# 集合（基于哈希表实现）
nums = set()

nums.add(1)
nums.add(2)
nums.add(3)
nums.add(1)  # 重复添加会被忽略

print(len(nums))  # 3

# 检查存在
if 2 in nums:
    print("2 is in the set")

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # 并集: {1, 2, 3, 4, 5, 6}
print(a & b)  # 交集: {3, 4}
print(a - b)  # 差集: {1, 2}
print(a ^ b)  # 对称差: {1, 2, 5, 6}

# 集合推导式
evens = {x for x in range(10) if x % 2 == 0}
print(evens)  # {0, 2, 4, 6, 8}
```

---

## 哈希表应用

### 两数之和

找出数组中两个数，使它们的和等于目标值。

```cpp
// C++ 两数之和
#include <vector>
#include <unordered_map>
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> numToIndex;  // 值 -> 索引
    
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        
        // 查找补数是否存在
        if (numToIndex.count(complement)) {
            return {numToIndex[complement], i};
        }
        
        numToIndex[nums[i]] = i;
    }
    
    return {};  // 未找到
}

// 使用示例
int main() {
    vector<int> nums = {2, 7, 11, 15};
    int target = 9;
    auto result = twoSum(nums, target);
    // result = {0, 1}，因为 nums[0] + nums[1] = 2 + 7 = 9
}
```

```python
# Python 两数之和
def two_sum(nums: list[int], target: int) -> list[int]:
    num_to_index = {}  # 值 -> 索引
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in num_to_index:
            return [num_to_index[complement], i]
        
        num_to_index[num] = i
    
    return []  # 未找到

# 使用示例
nums = [2, 7, 11, 15]
result = two_sum(nums, 9)
print(result)  # [0, 1]
```

### 字符统计

统计字符串中各字符出现的次数。

```cpp
// C++ 字符统计
#include <string>
#include <unordered_map>
#include <map>
using namespace std;

// 方法1：unordered_map（不保证顺序）
unordered_map<char, int> countChars(const string& s) {
    unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }
    return freq;
}

// 方法2：map（按字符排序）
map<char, int> countCharsSorted(const string& s) {
    map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }
    return freq;
}

// 方法3：仅统计小写字母（数组更快）
int* countLowercase(const string& s) {
    int* freq = new int[26]();
    for (char c : s) {
        if (c >= 'a' && c <= 'z') {
            freq[c - 'a']++;
        }
    }
    return freq;
}
```

```python
# Python 字符统计
from collections import Counter, defaultdict

# 方法1：使用 Counter（最简洁）
def count_chars(s: str) -> Counter:
    return Counter(s)

# 方法2：使用 defaultdict
def count_chars_defaultdict(s: str) -> dict:
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    return dict(freq)

# 方法3：手动实现
def count_chars_manual(s: str) -> dict:
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return freq

# 方法4：仅统计小写字母（更快）
def count_lowercase(s: str) -> list[int]:
    freq = [0] * 26
    for c in s:
        if 'a' <= c <= 'z':
            freq[ord(c) - ord('a')] += 1
    return freq

# 使用示例
s = "hello world"
print(Counter(s))
# Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# Counter 常用操作
cnt = Counter("abracadabra")
print(cnt.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]
```

### 去重

移除重复元素，保持原有顺序。

```cpp
// C++ 去重
#include <vector>
#include <unordered_set>
using namespace std;

// 保持原有顺序去重
vector<int> deduplicate(vector<int>& nums) {
    vector<int> result;
    unordered_set<int> seen;
    
    for (int num : nums) {
        if (seen.insert(num).second) {  // insert返回pair<iterator, bool>
            result.push_back(num);
        }
    }
    return result;
}

// 不需要保持顺序
vector<int> deduplicateUnordered(vector<int>& nums) {
    return vector<int>(unordered_set<int>(nums.begin(), nums.end()));
}
```

```python
# Python 去重
# 方法1：保持顺序（Python 3.7+ dict保持插入顺序）
def deduplicate(nums: list) -> list:
    return list(dict.fromkeys(nums))

# 方法2：使用集合（不保证顺序）
def deduplicate_unordered(nums: list) -> list:
    return list(set(nums))

# 方法3：手动实现保持顺序
def deduplicate_manual(nums: list) -> list:
    seen = set()
    result = []
    for num in nums:
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result

# 使用示例
nums = [1, 2, 2, 3, 1, 4, 3, 5]
print(deduplicate(nums))  # [1, 2, 3, 4, 5]
```

### LRU 缓存实现

LRU（Least Recently Used）缓存是一种常见的缓存淘汰策略。

```cpp
// C++ LRU 缓存实现
#include <list>
#include <unordered_map>
using namespace std;

class LRUCache {
private:
    int capacity;
    // 双向链表：存储键值对，最近使用的在头部
    list<pair<int, int>> cache;
    // 哈希表：键 -> 链表迭代器
    unordered_map<int, list<pair<int, int>>::iterator> map;
    
public:
    LRUCache(int cap) : capacity(cap) {}
    
    int get(int key) {
        auto it = map.find(key);
        if (it == map.end()) {
            return -1;
        }
        // 移动到链表头部
        cache.splice(cache.begin(), cache, it->second);
        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = map.find(key);
        
        if (it != map.end()) {
            // 更新已存在的键
            it->second->second = value;
            cache.splice(cache.begin(), cache, it->second);
            return;
        }
        
        // 检查容量
        if (cache.size() == capacity) {
            // 删除最久未使用的（链表尾部）
            int oldKey = cache.back().first;
            cache.pop_back();
            map.erase(oldKey);
        }
        
        // 插入新键值对
        cache.push_front({key, value});
        map[key] = cache.begin();
    }
};

// 使用示例
int main() {
    LRUCache cache(2);  // 容量为2
    
    cache.put(1, 1);
    cache.put(2, 2);
    cout << cache.get(1) << endl;  // 返回 1
    
    cache.put(3, 3);  // 淘汰 key=2
    cout << cache.get(2) << endl;  // 返回 -1（未找到）
    
    cache.put(4, 4);  // 淘汰 key=1
    cout << cache.get(1) << endl;  // 返回 -1
    cout << cache.get(3) << endl;  // 返回 3
    cout << cache.get(4) << endl;  // 返回 4
}
```

```python
# Python LRU 缓存实现
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 移动到末尾（表示最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 更新并移动到末尾
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # 检查容量
        if len(self.cache) > self.capacity:
            # 删除最久未使用的（开头）
            self.cache.popitem(last=False)

# Python 内置装饰器实现 LRU 缓存
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 使用示例
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # 淘汰 key=2
print(cache.get(2))  # -1
```

---

## 位运算基础

### 基本位运算

| 运算 | 符号 | 规则 | 示例 |
|------|------|------|------|
| 与 | `&` | 两位都为1时结果为1 | `5 & 3 = 1` |
| 或 | `\|` | 有一位为1时结果为1 | `5 \| 3 = 7` |
| 异或 | `^` | 两位不同时结果为1 | `5 ^ 3 = 6` |
| 非 | `~` | 按位取反 | `~5 = -6` |
| 左移 | `<<` | 左移n位，低位补0 | `5 << 1 = 10` |
| 右移 | `>>` | 右移n位，高位补符号位 | `5 >> 1 = 2` |

```cpp
// C++ 位运算示例
#include <iostream>
#include <bitset>
using namespace std;

int main() {
    int a = 5;  // 二进制: 0101
    int b = 3;  // 二进制: 0011
    
    cout << (a & b) << endl;   // 1  (0001)
    cout << (a | b) << endl;   // 7  (0111)
    cout << (a ^ b) << endl;   // 6  (0110)
    cout << (~a) << endl;      // -6 (按位取反，考虑补码)
    cout << (a << 1) << endl;  // 10 (1010)
    cout << (a >> 1) << endl;  // 2  (0010)
    
    // 使用 bitset 可视化
    bitset<8> x(5);
    cout << x << endl;  // 00000101
    
    return 0;
}
```

```python
# Python 位运算示例
a = 5  # 二进制: 0b0101
b = 3  # 二进制: 0b0011

print(a & b)    # 1  (0b0001)
print(a | b)    # 7  (0b0111)
print(a ^ b)    # 6  (0b0110)
print(~a)       # -6 (按位取反)
print(a << 1)   # 10 (0b1010)
print(a >> 1)   # 2  (0b0010)

# 二进制可视化
print(bin(a))   # 0b101
print(format(a, '08b'))  # 00000101
```

### 位运算技巧

```cpp
// C++ 位运算技巧
#include <iostream>
using namespace std;

// 1. 判断奇偶
bool isOdd(int n) {
    return n & 1;  // 最后一位为1则是奇数
}

// 2. 获取第i位（从0开始）
bool getBit(int n, int i) {
    return (n >> i) & 1;
}

// 3. 设置第i位为1
int setBit(int n, int i) {
    return n | (1 << i);
}

// 4. 清除第i位（设为0）
int clearBit(int n, int i) {
    return n & ~(1 << i);
}

// 5. 切换第i位
int toggleBit(int n, int i) {
    return n ^ (1 << i);
}

// 6. 清除最低位的1
int clearLowestBit(int n) {
    return n & (n - 1);
}

// 7. 获取最低位的1
int getLowestBit(int n) {
    return n & (-n);
}

// 8. 统计二进制中1的个数
int countBits(int n) {
    int count = 0;
    while (n) {
        n &= n - 1;  // 每次清除最低位的1
        count++;
    }
    return count;
}

// 9. 判断是否为2的幂
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// 10. 交换两个数（不使用临时变量）
void swapXOR(int& a, int& b) {
    a ^= b;
    b ^= a;
    a ^= b;
}

// 11. 快速乘除2
int multiply2(int n) { return n << 1; }
int divide2(int n) { return n >> 1; }

// 12. 取绝对值（不使用分支）
int absXOR(int n) {
    int mask = n >> 31;  // 负数全1，正数全0
    return (n ^ mask) - mask;
}

int main() {
    int n = 13;  // 二进制: 1101
    
    cout << isOdd(n) << endl;           // 1 (奇数)
    cout << getBit(n, 2) << endl;       // 1 (第2位)
    cout << setBit(n, 1) << endl;       // 15 (1111)
    cout << clearBit(n, 2) << endl;     // 9 (1001)
    cout << toggleBit(n, 0) << endl;    // 12 (1100)
    cout << countBits(n) << endl;       // 3
    cout << isPowerOfTwo(8) << endl;    // 1
    
    return 0;
}
```

```python
# Python 位运算技巧

# 1. 判断奇偶
def is_odd(n: int) -> bool:
    return bool(n & 1)

# 2. 获取第i位
def get_bit(n: int, i: int) -> bool:
    return bool((n >> i) & 1)

# 3. 设置第i位为1
def set_bit(n: int, i: int) -> int:
    return n | (1 << i)

# 4. 清除第i位
def clear_bit(n: int, i: int) -> int:
    return n & ~(1 << i)

# 5. 切换第i位
def toggle_bit(n: int, i: int) -> int:
    return n ^ (1 << i)

# 6. 清除最低位的1
def clear_lowest_bit(n: int) -> int:
    return n & (n - 1)

# 7. 获取最低位的1
def get_lowest_bit(n: int) -> int:
    return n & (-n)

# 8. 统计二进制中1的个数
def count_bits(n: int) -> int:
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# Python 内置方法
def count_bits_builtin(n: int) -> int:
    return bin(n).count('1')

# 9. 判断是否为2的幂
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

# 10. 交换两个数
def swap_xor(a: int, b: int) -> tuple[int, int]:
    a ^= b
    b ^= a
    a ^= b
    return a, b

# 使用示例
n = 13  # 二进制: 1101
print(is_odd(n))           # True
print(get_bit(n, 2))       # True
print(set_bit(n, 1))       # 15
print(clear_bit(n, 2))     # 9
print(count_bits(n))       # 3
print(is_power_of_two(8))  # True
```

---

## 位集 (Bitset)

### 位集原理

位集是一种紧凑存储布尔值的数据结构，每个元素只用 **1 bit** 存储：
- 存储密度高：1字节可存储8个布尔值
- 集合运算快：使用位运算批量操作
- 空间效率：比 bool 数组节省 8 倍空间

```
位集: [1, 0, 1, 1, 0, 0, 1, 0]
       ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
索引:   0  1  2  3  4  5  6  7
值:    0b10110010 = 178
```

### C++ bitset 使用

```cpp
// C++ bitset 使用
#include <iostream>
#include <bitset>
#include <string>
using namespace std;

int main() {
    // ========== 创建 bitset ==========
    bitset<8> b1;                    // 全0: 00000000
    bitset<8> b2(15);                // 整数初始化: 00001111
    bitset<8> b3(string("10101"));   // 字符串初始化: 00010101
    
    cout << b1 << endl;  // 00000000
    cout << b2 << endl;  // 00001111
    cout << b3 << endl;  // 00010101
    
    // ========== 访问元素 ==========
    cout << b2[0] << endl;      // 1 (最低位)
    cout << b2.test(3) << endl; // 1 (带边界检查)
    
    // ========== 修改元素 ==========
    b1.set(2);      // 设置第2位为1
    b1.set();       // 全部设为1
    b1.reset(2);    // 设置第2位为0
    b1.reset();     // 全部设为0
    b1.flip(2);     // 翻转第2位
    b1.flip();      // 全部翻转
    
    // ========== 查询操作 ==========
    cout << b2.count() << endl;    // 统计1的个数: 4
    cout << b2.size() << endl;     // 总位数: 8
    cout << b2.all() << endl;      // 是否全为1: false
    cout << b2.any() << endl;      // 是否有1: true
    cout << b2.none() << endl;     // 是否全为0: false
    
    // ========== 位运算 ==========
    bitset<8> a(string("1010"));
    bitset<8> b(string("1100"));
    
    cout << (a & b) << endl;   // 按位与: 1000
    cout << (a | b) << endl;   // 按位或: 1110
    cout << (a ^ b) << endl;   // 异或: 0110
    cout << (~a) << endl;      // 取反: 11110101
    cout << (a << 2) << endl;  // 左移: 100000
    cout << (a >> 1) << endl;  // 右移: 0101
    
    // ========== 转换 ==========
    cout << b2.to_ulong() << endl;    // 转为无符号长整型: 15
    cout << b2.to_string() << endl;   // 转为字符串: 00001111
    
    return 0;
}
```

### Python 位集实现

```python
# Python 位集实现
from typing import List, Set

class Bitset:
    """自定义位集类"""
    
    def __init__(self, size: int):
        self.size = size
        # 使用整数存储位集
        self.data = 0
    
    def set(self, i: int) -> None:
        """设置第i位为1"""
        if 0 <= i < self.size:
            self.data |= (1 << i)
    
    def reset(self, i: int) -> None:
        """设置第i位为0"""
        if 0 <= i < self.size:
            self.data &= ~(1 << i)
    
    def flip(self, i: int) -> None:
        """翻转第i位"""
        if 0 <= i < self.size:
            self.data ^= (1 << i)
    
    def test(self, i: int) -> bool:
        """检查第i位是否为1"""
        if 0 <= i < self.size:
            return bool(self.data & (1 << i))
        return False
    
    def count(self) -> int:
        """统计1的个数"""
        return bin(self.data).count('1')
    
    def all(self) -> bool:
        """检查是否全为1"""
        return self.count() == self.size
    
    def any(self) -> bool:
        """检查是否有1"""
        return self.data != 0
    
    def none(self) -> bool:
        """检查是否全为0"""
        return self.data == 0
    
    def to_set(self) -> Set[int]:
        """转换为集合（返回所有为1的位置）"""
        result = set()
        for i in range(self.size):
            if self.test(i):
                result.add(i)
        return result
    
    def __and__(self, other: 'Bitset') -> 'Bitset':
        """交集"""
        result = Bitset(self.size)
        result.data = self.data & other.data
        return result
    
    def __or__(self, other: 'Bitset') -> 'Bitset':
        """并集"""
        result = Bitset(self.size)
        result.data = self.data | other.data
        return result
    
    def __xor__(self, other: 'Bitset') -> 'Bitset':
        """对称差"""
        result = Bitset(self.size)
        result.data = self.data ^ other.data
        return result
    
    def __sub__(self, other: 'Bitset') -> 'Bitset':
        """差集"""
        result = Bitset(self.size)
        result.data = self.data & ~other.data
        return result
    
    def __str__(self) -> str:
        return format(self.data, f'0{self.size}b')
    
    def __repr__(self) -> str:
        return f"Bitset('{self}')"


# 使用示例
bs = Bitset(8)
bs.set(0)
bs.set(2)
bs.set(4)
print(bs)           # 10101
print(bs.count())   # 3
print(bs.test(2))   # True

# 集合运算
a = Bitset(8)
a.set(0); a.set(2); a.set(4)
b = Bitset(8)
b.set(2); b.set(3); b.set(5)

print(a & b)  # 00100 (交集)
print(a | b)  # 011101 (并集)
print(a - b)  # 10001 (差集)
```

### 集合运算

位集天然适合表示集合，位运算对应集合运算：

| 位运算 | 集合运算 | 含义 |
|--------|---------|------|
| `a & b` | 交集 | 同时在两个集合中的元素 |
| `a \| b` | 并集 | 在任一集合中的元素 |
| `a ^ b` | 对称差 | 只在一个集合中的元素 |
| `a & ~b` | 差集 | 在a但不在b中的元素 |
| `~a` | 补集 | 不在集合中的元素 |

```cpp
// C++ 集合运算示例
#include <iostream>
#include <bitset>
using namespace std;

int main() {
    // 使用位集表示集合 {0, 2, 4} 和 {1, 2, 3}
    bitset<8> a(0b00010101);  // {0, 2, 4}
    bitset<8> b(0b00001110);  // {1, 2, 3}
    
    cout << "集合A: " << a << endl;
    cout << "集合B: " << b << endl;
    
    // 交集: {2}
    bitset<8> intersection = a & b;
    cout << "交集: " << intersection << endl;  // 00000100
    
    // 并集: {0, 1, 2, 3, 4}
    bitset<8> unionSet = a | b;
    cout << "并集: " << unionSet << endl;  // 00011111
    
    // 对称差: {0, 1, 3, 4}
    bitset<8> symDiff = a ^ b;
    cout << "对称差: " << symDiff << endl;  // 00011011
    
    // 差集 A-B: {0, 4}
    bitset<8> diff = a & ~b;
    cout << "差集: " << diff << endl;  // 00010001
    
    // 判断子集: A 是否是 B 的子集
    bool isSubset = (a & b) == a;
    cout << "A是B的子集: " << isSubset << endl;  // false
    
    return 0;
}
```

```python
# Python 集合运算示例

class IntSet:
    """使用整数实现的集合"""
    
    def __init__(self, elements: list = None):
        self.data = 0
        if elements:
            for e in elements:
                self.add(e)
    
    def add(self, x: int) -> None:
        self.data |= (1 << x)
    
    def remove(self, x: int) -> None:
        self.data &= ~(1 << x)
    
    def contains(self, x: int) -> bool:
        return bool(self.data & (1 << x))
    
    def union(self, other: 'IntSet') -> 'IntSet':
        result = IntSet()
        result.data = self.data | other.data
        return result
    
    def intersection(self, other: 'IntSet') -> 'IntSet':
        result = IntSet()
        result.data = self.data & other.data
        return result
    
    def difference(self, other: 'IntSet') -> 'IntSet':
        result = IntSet()
        result.data = self.data & ~other.data
        return result
    
    def symmetric_difference(self, other: 'IntSet') -> 'IntSet':
        result = IntSet()
        result.data = self.data ^ other.data
        return result
    
    def is_subset(self, other: 'IntSet') -> bool:
        return (self.data & other.data) == self.data
    
    def elements(self) -> list:
        result = []
        x = self.data
        i = 0
        while x:
            if x & 1:
                result.append(i)
            x >>= 1
            i += 1
        return result
    
    def __str__(self) -> str:
        return "{" + ", ".join(map(str, self.elements())) + "}"


# 使用示例
a = IntSet([0, 2, 4])
b = IntSet([1, 2, 3])

print(f"集合A: {a}")  # {0, 2, 4}
print(f"集合B: {b}")  # {1, 2, 3}
print(f"交集: {a.intersection(b)}")  # {2}
print(f"并集: {a.union(b)}")  # {0, 1, 2, 3, 4}
print(f"差集A-B: {a.difference(b)}")  # {0, 4}
```

### 状态压缩应用

位集常用于**状态压缩DP**，将状态用二进制表示。

#### 示例：旅行商问题 (TSP)

```cpp
// C++ TSP 状态压缩
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

int tsp(const vector<vector<int>>& dist) {
    int n = dist.size();
    int VISITED_ALL = (1 << n) - 1;
    
    // dp[mask][i] = 已访问城市集合为mask，当前在城市i的最短路径
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX / 2));
    
    // 从城市0出发
    dp[1][0] = 0;  // mask=001，只访问了城市0
    
    // 枚举所有状态
    for (int mask = 1; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;  // 城市i未访问
            
            // 尝试从城市j转移到城市i
            for (int j = 0; j < n; j++) {
                if (mask & (1 << j)) {  // 城市j已访问
                    int prevMask = mask ^ (1 << i);
                    dp[mask][i] = min(dp[mask][i], dp[prevMask][j] + dist[j][i]);
                }
            }
        }
    }
    
    // 返回从任意城市回到起点的最短路径
    int result = INT_MAX;
    for (int i = 1; i < n; i++) {
        result = min(result, dp[VISITED_ALL][i] + dist[i][0]);
    }
    return result;
}

int main() {
    vector<vector<int>> dist = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };
    
    cout << "最短路径长度: " << tsp(dist) << endl;  // 80
    return 0;
}
```

```python
# Python TSP 状态压缩
def tsp(dist: list[list[int]]) -> int:
    n = len(dist)
    visited_all = (1 << n) - 1
    
    # dp[mask][i] = 已访问城市集合为mask，当前在城市i的最短路径
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 从城市0出发
    
    # 枚举所有状态
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue  # 城市i未访问
            
            # 尝试从城市j转移到城市i
            for j in range(n):
                if mask & (1 << j):  # 城市j已访问
                    prev_mask = mask ^ (1 << i)
                    dp[mask][i] = min(dp[mask][i], dp[prev_mask][j] + dist[j][i])
    
    # 返回从任意城市回到起点的最短路径
    result = float('inf')
    for i in range(1, n):
        result = min(result, dp[visited_all][i] + dist[i][0])
    
    return result

# 使用示例
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(f"最短路径长度: {tsp(dist)}")  # 80
```

#### 示例：N皇后问题

```cpp
// C++ N皇后位运算解法
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class NQueens {
private:
    int n;
    int count;
    vector<vector<string>> solutions;
    
    void solve(int row, int cols, int diag1, int diag2, vector<string>& board) {
        if (row == n) {
            solutions.push_back(board);
            count++;
            return;
        }
        
        // 计算可用位置
        int available = ((1 << n) - 1) & ~(cols | diag1 | diag2);
        
        while (available) {
            // 获取最低位的1
            int pos = available & (-available);
            int col = __builtin_ctz(pos);  // 计算位置
            
            // 放置皇后
            board[row][col] = 'Q';
            solve(row + 1, 
                  cols | pos, 
                  (diag1 | pos) << 1, 
                  (diag2 | pos) >> 1, 
                  board);
            board[row][col] = '.';
            
            // 清除最低位的1
            available &= available - 1;
        }
    }
    
public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        this->count = 0;
        vector<string> board(n, string(n, '.'));
        solve(0, 0, 0, 0, board);
        return solutions;
    }
    
    int getCount() { return count; }
};

int main() {
    NQueens solver;
    auto solutions = solver.solveNQueens(4);
    
    cout << "解的数量: " << solutions.size() << endl;
    for (auto& board : solutions) {
        for (auto& row : board) {
            cout << row << endl;
        }
        cout << endl;
    }
    return 0;
}
```

```python
# Python N皇后位运算解法
def solve_n_queens(n: int) -> list[list[str]]:
    solutions = []
    
    def solve(row: int, cols: int, diag1: int, diag2: int, board: list[str]):
        if row == n:
            solutions.append(board[:])
            return
        
        # 计算可用位置
        available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
        
        while available:
            # 获取最低位的1
            pos = available & (-available)
            col = pos.bit_length() - 1
            
            # 放置皇后
            new_board = board[:]
            new_row = list(new_board[row])
            new_row[col] = 'Q'
            new_board[row] = ''.join(new_row)
            
            solve(row + 1,
                  cols | pos,
                  (diag1 | pos) << 1,
                  (diag2 | pos) >> 1,
                  new_board)
            
            # 清除最低位的1
            available &= available - 1
    
    solve(0, 0, 0, 0, ['.' * n for _ in range(n)])
    return solutions

# 使用示例
solutions = solve_n_queens(4)
print(f"解的数量: {len(solutions)}")
for board in solutions:
    for row in board:
        print(row)
    print()
```

---

## 常见问题与注意事项

### 哈希表常见问题

1. **哈希碰撞攻击**：恶意构造的输入可能导致大量碰撞，使哈希表退化为链表
   - 解决：使用随机哈希种子或安全的哈希函数

2. **迭代器失效**：在 C++ 中，插入操作可能导致迭代器失效
   - 解决：避免在遍历时修改，或使用 `find` 返回的迭代器

3. **自定义类型的哈希函数**：确保相等的对象产生相同的哈希值

```cpp
// C++ 迭代器失效示例
unordered_map<int, int> m = {{1, 1}, {2, 2}};

// 错误：插入可能导致重哈希，迭代器失效
for (auto& [k, v] : m) {
    m[k + 10] = v;  // 危险！
}

// 正确：使用临时容器存储要插入的元素
unordered_map<int, int> temp;
for (auto& [k, v] : m) {
    temp[k + 10] = v;
}
m.insert(temp.begin(), temp.end());
```

### 位运算注意事项

1. **符号位处理**：右移时负数的行为在 C++ 中是**实现定义**的
   - 建议使用无符号类型进行位运算

2. **优先级问题**：位运算优先级较低，需要加括号
   ```cpp
   // 错误
   if (n & 1 == 1)  // 实际解析为 n & (1 == 1)
   
   // 正确
   if ((n & 1) == 1)
   ```

3. **移位越界**：移位数量超过类型位数是未定义行为
   ```cpp
   int x = 1;
   int y = x << 32;  // 未定义行为（int通常是32位）
   ```

### 性能优化建议

| 场景 | 建议 |
|------|------|
| 小规模整数集合 | 使用 bitset 或直接用整数 |
| 需要有序遍历 | 使用 map 而非 unordered_map |
| 高频查找 | 哈希表 O(1) 平均 |
| 内存受限 | bitset 比 bool 数组更省空间 |
| 状态压缩 | 位集或整数，配合位运算 |

---

## 参考资料

- [C++ Reference - unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map)
- [C++ Reference - bitset](https://en.cppreference.com/w/cpp/utility/bitset)
- [Python Documentation - dict](https://docs.python.org/3/library/stdtypes.html#dict)
- [算法导论 - 哈希表章节](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)
