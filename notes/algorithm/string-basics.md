# 字符串基础

字符串是由字符组成的序列，是编程中最基础的数据类型之一。字符串算法主要研究如何高效地处理文本数据，包括模式匹配、回文检测、子串查找等问题。

📌 **核心概念**
- **字符串**：由零个或多个字符组成的有限序列
- **子串**：字符串中连续的一段字符序列
- **前缀/后缀**：从开头/结尾开始的子串
- **模式匹配**：在文本串中查找模式串出现位置的过程

## 字符串的存储与表示

### 存储方式

字符串在计算机中有两种主要的存储方式：

| 存储方式 | 特点 | 适用场景 |
|---------|------|---------|
| 顺序存储 | 字符连续存放在内存中 | 大多数编程语言默认方式 |
| 链式存储 | 每个节点存储一个字符 | 频繁插入删除的场景 |

### 不同语言的字符串表示

::: code-group
```cpp
// C++ 字符串表示
#include <string>
using namespace std;

// C风格字符串（以'\0'结尾）
const char* cstr = "hello";  // 长度为5，但实际占用6字节

// C++ string类（推荐使用）
string s1 = "hello";          // 直接初始化
string s2(5, 'a');            // "aaaaa"
string s3 = s1 + " world";    // 拼接："hello world"
string s4 = s1.substr(0, 3);  // 子串："hel"

// 常用操作
int len = s1.length();        // 获取长度：5
char c = s1[0];               // 随机访问：'h'
s1.push_back('!');            // 追加字符
s1.append(" world");          // 追加字符串
```

```python
# Python 字符串表示
s1 = "hello"                  # 直接定义
s2 = 'world'                  # 单引号也可以
s3 = """多行
字符串"""                      # 多行字符串

# 字符串是不可变的！
s4 = s1 + " " + s2            # 拼接："hello world"
s5 = s1[0:3]                  # 切片："hel"
s6 = s1 * 3                   # 重复："hellohellohello"

# 常用操作
length = len(s1)              # 获取长度：5
c = s1[0]                     # 随机访问：'h'
s1.upper()                    # 转大写："HELLO"
s1.lower()                    # 转小写："hello"
", ".join(["a", "b", "c"])    # 连接："a, b, c"
```
:::

### 字符编码

📌 **常见编码方式**
- **ASCII**：7位编码，表示128个字符（英文、数字、控制字符）
- **UTF-8**：变长编码，兼容ASCII，可表示所有Unicode字符
- **GBK/GB2312**：中文编码标准

::: code-group
```cpp
// C++ 字符编码处理
#include <string>

// ASCII字符处理
char c = 'A';
int ascii_val = (int)c;       // 65
char upper = c + 32;          // 'a'（大小写转换）

// 判断字符类型
bool is_digit = isdigit(c);   // 是否数字
bool is_alpha = isalpha(c);   // 是否字母
bool is_lower = islower(c);   // 是否小写
bool is_upper = isupper(c);   // 是否大写

// 中文处理（UTF-8）
string chinese = "你好世界";
// 注意：中文字符在UTF-8中占3字节
// 需要特殊处理才能正确遍历
```

```python
# Python 字符编码处理
c = 'A'
ascii_val = ord(c)            # 65
char = chr(65)                # 'A'

# 判断字符类型
is_digit = c.isdigit()        # 是否数字
is_alpha = c.isalpha()        # 是否字母
is_lower = c.islower()        # 是否小写
is_upper = c.isupper()        # 是否大写

# 编码转换
s = "你好"
utf8_bytes = s.encode('utf-8')     # 编码为字节
decoded = utf8_bytes.decode('utf-8')  # 解码为字符串
```
:::

## 字符串的基本操作

### 遍历方式

::: code-group
```cpp
// C++ 字符串遍历
string s = "hello";

// 方式1：下标遍历
for (int i = 0; i < s.length(); i++) {
    cout << s[i] << " ";
}

// 方式2：范围for循环（C++11）
for (char c : s) {
    cout << c << " ";
}

// 方式3：迭代器
for (auto it = s.begin(); it != s.end(); it++) {
    cout << *it << " ";
}

// 方式4：范围for + 引用（可修改）
for (char& c : s) {
    c = toupper(c);  // 转换为大写
}
```

```python
# Python 字符串遍历
s = "hello"

# 方式1：直接遍历
for c in s:
    print(c, end=" ")

# 方式2：带索引遍历
for i, c in enumerate(s):
    print(f"s[{i}] = {c}")

# 方式3：索引遍历
for i in range(len(s)):
    print(s[i], end=" ")

# 方式4：列表推导式（生成新字符串）
s_upper = ''.join(c.upper() for c in s)  # "HELLO"
```
:::

### 常用操作函数

::: code-group
```cpp
// C++ 字符串常用操作
#include <string>
#include <algorithm>
using namespace std;

string s = "Hello World";

// 查找操作
size_t pos = s.find("World");     // 查找子串：6
size_t pos2 = s.find('o');        // 查找字符：4
size_t pos3 = s.rfind('o');       // 从后查找：7

// 截取操作
string sub = s.substr(0, 5);      // "Hello"
string sub2 = s.substr(6);        // "World"

// 插入删除
s.insert(5, " Beautiful");        // 在位置5插入
s.erase(5, 10);                   // 删除从位置5开始的10个字符
s.push_back('!');                 // 尾部追加字符
s.pop_back();                     // 删除最后一个字符

// 修改操作
reverse(s.begin(), s.end());      // 反转字符串
sort(s.begin(), s.end());         // 排序（按字典序）

// 比较操作
string a = "apple", b = "banana";
bool less = a < b;                // 按字典序比较：true
int cmp = a.compare(b);           // 返回-1, 0, 1
```

```python
# Python 字符串常用操作
s = "Hello World"

# 查找操作
pos = s.find("World")             # 查找子串：6
pos2 = s.find('o')                # 查找字符：4
pos3 = s.rfind('o')               # 从后查找：7
count = s.count('o')              # 统计出现次数：2

# 截取操作（切片）
sub = s[0:5]                      # "Hello"
sub2 = s[6:]                      # "World"
sub3 = s[-5:]                     # "World"（倒数）

# 分割和连接
parts = s.split(" ")              # ["Hello", "World"]
joined = "-".join(parts)          # "Hello-World"

# 修改操作（返回新字符串）
s_reversed = s[::-1]              # 反转："dlroW olleH"
s_upper = s.upper()               # 大写："HELLO WORLD"
s_lower = s.lower()               # 小写："hello world"
s_replace = s.replace("World", "Python")  # 替换

# 去除空白
s_strip = "  hello  ".strip()     # "hello"
s_lstrip = "  hello  ".lstrip()   # "hello  "
s_rstrip = "  hello  ".rstrip()   # "  hello"

# 比较操作
a, b = "apple", "banana"
less = a < b                      # 按字典序比较：True
```
:::

## 字符串哈希

字符串哈希是一种通过哈希函数将字符串映射为整数的技术，可以在 O(1) 时间内判断两个子串是否相等。

### 多项式滚动哈希

📌 **核心思想**
将字符串看作一个 base 进制的数，计算其对应的十进制值，并对某个大质数取模，得到哈希值。

**哈希公式**：
$$H(s) = \sum_{i=0}^{n-1} s[i] \times base^{n-1-i} \mod p$$

其中：
- $base$ 是基数（通常取一个较大的质数，如 131、137、911382629）
- $p$ 是模数（通常取大质数，如 $10^9+7$、$10^9+9$）

::: code-group
```cpp
// C++ 字符串哈希实现
#include <string>
#include <vector>
using namespace std;

class StringHash {
private:
    vector<long long> prefix;  // 前缀哈希值
    vector<long long> power;   // base的幂次
    long long base;
    long long mod;
    
public:
    // 构造函数：预处理字符串
    StringHash(const string& s, long long b = 131, long long m = 1e9 + 7) 
        : base(b), mod(m) {
        int n = s.length();
        prefix.resize(n + 1, 0);
        power.resize(n + 1, 1);
        
        // 计算前缀哈希和幂次
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = (prefix[i] * base + s[i]) % mod;
            power[i + 1] = (power[i] * base) % mod;
        }
    }
    
    // 获取子串 s[l..r-1] 的哈希值（左闭右开）
    long long getHash(int l, int r) {
        return (prefix[r] - prefix[l] * power[r - l] % mod + mod) % mod;
    }
    
    // 获取整个字符串的哈希值
    long long getFullHash() {
        return prefix[prefix.size() - 1];
    }
    
    // 判断两个子串是否相等
    bool isSame(int l1, int r1, int l2, int r2) {
        return getHash(l1, r1) == getHash(l2, r2);
    }
};

// 使用示例
int main() {
    string s = "ababab";
    StringHash sh(s);
    
    // 比较子串 "ab" (位置0-2) 和 "ab" (位置2-4)
    bool same = sh.isSame(0, 2, 2, 4);  // true
    cout << "子串相等: " << same << endl;
    
    return 0;
}
```

```python
# Python 字符串哈希实现
class StringHash:
    def __init__(self, s: str, base: int = 131, mod: int = 10**9 + 7):
        self.base = base
        self.mod = mod
        self.n = len(s)
        
        # 预处理前缀哈希和幂次
        self.prefix = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)
        
        for i in range(self.n):
            self.prefix[i + 1] = (self.prefix[i] * base + ord(s[i])) % self.mod
            self.power[i + 1] = (self.power[i] * base) % self.mod
    
    def get_hash(self, l: int, r: int) -> int:
        """获取子串 s[l:r] 的哈希值（左闭右开）"""
        return (self.prefix[r] - self.prefix[l] * self.power[r - l]) % self.mod
    
    def get_full_hash(self) -> int:
        """获取整个字符串的哈希值"""
        return self.prefix[self.n]
    
    def is_same(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """判断两个子串是否相等"""
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


# 使用示例
s = "ababab"
sh = StringHash(s)

# 比较子串 "ab" (位置0-2) 和 "ab" (位置2-4)
same = sh.is_same(0, 2, 2, 4)
print(f"子串相等: {same}")  # True
```
:::

### 双哈希防冲突

⚠️ **哈希冲突问题**
单个哈希函数存在冲突风险，使用双哈希可以大大降低冲突概率。

::: code-group
```cpp
// C++ 双哈希实现
class DoubleHash {
private:
    StringHash h1, h2;
    
public:
    DoubleHash(const string& s) 
        : h1(s, 131, 1000000007),    // 第一组参数
          h2(s, 137, 1000000009) {}  // 第二组参数（不同的base和mod）
    
    // 获取双哈希值
    pair<long long, long long> getHash(int l, int r) {
        return {h1.getHash(l, r), h2.getHash(l, r)};
    }
    
    // 判断两个子串是否相等
    bool isSame(int l1, int r1, int l2, int r2) {
        return getHash(l1, r1) == getHash(l2, r2);
    }
};

// 使用示例
bool checkSameSubstring(const string& s, int l1, int r1, int l2, int r2) {
    DoubleHash dh(s);
    return dh.isSame(l1, r1, l2, r2);
}
```

```python
# Python 双哈希实现
class DoubleHash:
    def __init__(self, s: str):
        # 使用两组不同的参数
        self.h1 = StringHash(s, base=131, mod=10**9 + 7)
        self.h2 = StringHash(s, base=137, mod=10**9 + 9)
    
    def get_hash(self, l: int, r: int) -> tuple:
        """获取双哈希值（元组）"""
        return (self.h1.get_hash(l, r), self.h2.get_hash(l, r))
    
    def is_same(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """判断两个子串是否相等"""
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)


# 使用示例
def check_same_substring(s: str, l1: int, r1: int, l2: int, r2: int) -> bool:
    dh = DoubleHash(s)
    return dh.is_same(l1, r1, l2, r2)
```
:::

### 快速子串比较

利用字符串哈希，可以实现 O(1) 时间的子串比较，这在很多字符串问题中非常有用。

::: code-group
```cpp
// 快速比较子串字典序（二分 + 哈希）
int compareSubstring(StringHash& sh, const string& s, 
                     int l1, int r1, int l2, int r2) {
    int len1 = r1 - l1, len2 = r2 - l2;
    int minLen = min(len1, len2);
    
    // 二分找最长公共前缀
    int lo = 0, hi = minLen;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (sh.getHash(l1, l1 + mid) == sh.getHash(l2, l2 + mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    
    // 根据公共前缀长度判断字典序
    if (lo == minLen) {
        return len1 - len2;  // 一个是另一个的前缀
    }
    return s[l1 + lo] - s[l2 + lo];  // 根据第一个不同字符判断
}
```

```python
def compare_substring(sh: StringHash, s: str, 
                      l1: int, r1: int, l2: int, r2: int) -> int:
    """比较两个子串的字典序，返回 -1, 0, 1"""
    len1, len2 = r1 - l1, r2 - l2
    min_len = min(len1, len2)
    
    # 二分找最长公共前缀
    lo, hi = 0, min_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sh.get_hash(l1, l1 + mid) == sh.get_hash(l2, l2 + mid):
            lo = mid
        else:
            hi = mid - 1
    
    # 根据公共前缀长度判断字典序
    if lo == min_len:
        return len1 - len2
    return ord(s[l1 + lo]) - ord(s[l2 + lo])
```
:::

## 字符串匹配算法

字符串匹配是在文本串 T 中查找模式串 P 出现位置的问题。这是字符串算法中最经典的问题之一。

### 朴素匹配算法

📌 **基本思想**
从文本串的每个位置开始，逐个字符比较是否与模式串匹配。

**时间复杂度**：最坏 O(n×m)，其中 n 是文本串长度，m 是模式串长度。

::: code-group
```cpp
// C++ 朴素字符串匹配
vector<int> naiveSearch(const string& text, const string& pattern) {
    int n = text.length(), m = pattern.length();
    vector<int> result;
    
    // 遍历所有可能的起始位置
    for (int i = 0; i <= n - m; i++) {
        bool match = true;
        
        // 逐字符比较
        for (int j = 0; j < m; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;  // 不匹配，跳出内层循环
            }
        }
        
        if (match) {
            result.push_back(i);  // 记录匹配位置
        }
    }
    
    return result;
}

// 使用示例
int main() {
    string text = "ABABABCABAB";
    string pattern = "ABAB";
    
    vector<int> positions = naiveSearch(text, pattern);
    // 输出: 0 2 7
    for (int pos : positions) {
        cout << pos << " ";
    }
    return 0;
}
```

```python
# Python 朴素字符串匹配
def naive_search(text: str, pattern: str) -> list:
    """返回所有匹配位置的列表"""
    n, m = len(text), len(pattern)
    result = []
    
    # 遍历所有可能的起始位置
    for i in range(n - m + 1):
        match = True
        
        # 逐字符比较
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        
        if match:
            result.append(i)
    
    return result


# 使用示例
text = "ABABABCABAB"
pattern = "ABAB"
positions = naive_search(text, pattern)
print(positions)  # [0, 2, 7]

# Python 内置方法
pos = text.find(pattern)           # 找第一个匹配位置
positions = []
start = 0
while True:
    pos = text.find(pattern, start)
    if pos == -1:
        break
    positions.append(pos)
    start = pos + 1
```
:::

### KMP算法

📌 **核心思想**
KMP（Knuth-Morris-Pratt）算法利用已匹配的信息，避免主串指针回溯，实现 O(n+m) 的线性时间复杂度。

💡 **关键概念：前缀函数（Next数组）**
前缀函数 `π[i]` 表示子串 `s[0..i]` 的最长相等前后缀的长度。

<KMPSearch />

#### 前缀函数计算

::: code-group
```cpp
// C++ 计算前缀函数（next数组）
vector<int> computePrefix(const string& pattern) {
    int m = pattern.length();
    vector<int> pi(m, 0);  // pi[0] = 0
    
    // j 表示当前最长相等前后缀的长度
    for (int i = 1, j = 0; i < m; i++) {
        // 不匹配时，j回溯到前一个位置
        while (j > 0 && pattern[i] != pattern[j]) {
            j = pi[j - 1];  // 关键：利用已计算的前缀函数
        }
        
        // 匹配时，j前进一步
        if (pattern[i] == pattern[j]) {
            j++;
        }
        
        pi[i] = j;
    }
    
    return pi;
}

// 示例：计算 "ABABCABAB" 的前缀函数
// pattern: A B A B C A B A B
// pi:      0 0 1 2 0 1 2 3 4
```

```python
# Python 计算前缀函数
def compute_prefix(pattern: str) -> list:
    """计算模式串的前缀函数（next数组）"""
    m = len(pattern)
    pi = [0] * m
    
    # j 表示当前最长相等前后缀的长度
    j = 0
    for i in range(1, m):
        # 不匹配时，j回溯到前一个位置
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]  # 关键：利用已计算的前缀函数
        
        # 匹配时，j前进一步
        if pattern[i] == pattern[j]:
            j += 1
        
        pi[i] = j
    
    return pi


# 示例：计算 "ABABCABAB" 的前缀函数
pattern = "ABABCABAB"
pi = compute_prefix(pattern)
print(pi)  # [0, 0, 1, 2, 0, 1, 2, 3, 4]
```
:::

#### KMP匹配实现

::: code-group
```cpp
// C++ KMP字符串匹配
#include <string>
#include <vector>
using namespace std;

class KMP {
private:
    string pattern;
    vector<int> pi;
    
public:
    KMP(const string& p) : pattern(p) {
        pi = computePrefix(pattern);
    }
    
    // 在文本串中查找所有匹配位置
    vector<int> search(const string& text) {
        int n = text.length(), m = pattern.length();
        vector<int> result;
        
        // j 表示当前匹配的长度
        for (int i = 0, j = 0; i < n; i++) {
            // 不匹配时，根据前缀函数回溯
            while (j > 0 && text[i] != pattern[j]) {
                j = pi[j - 1];
            }
            
            // 匹配时，j前进一步
            if (text[i] == pattern[j]) {
                j++;
            }
            
            // 完全匹配
            if (j == m) {
                result.push_back(i - m + 1);  // 记录匹配起始位置
                j = pi[j - 1];  // 继续搜索下一个匹配
            }
        }
        
        return result;
    }
    
    // 静态方法：一次性查找
    static vector<int> findAll(const string& text, const string& pattern) {
        KMP kmp(pattern);
        return kmp.search(text);
    }
};

// 使用示例
int main() {
    string text = "ABABABCABABABCABABC";
    string pattern = "ABABCABAB";
    
    KMP kmp(pattern);
    vector<int> positions = kmp.search(text);
    
    // 输出匹配位置
    for (int pos : positions) {
        cout << pos << " ";  // 输出: 2 10
    }
    
    return 0;
}
```

```python
# Python KMP字符串匹配
class KMP:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pi = self._compute_prefix(pattern)
    
    @staticmethod
    def _compute_prefix(pattern: str) -> list:
        """计算前缀函数"""
        m = len(pattern)
        pi = [0] * m
        
        j = 0
        for i in range(1, m):
            while j > 0 and pattern[i] != pattern[j]:
                j = pi[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            pi[i] = j
        
        return pi
    
    def search(self, text: str) -> list:
        """在文本串中查找所有匹配位置"""
        n, m = len(text), len(self.pattern)
        result = []
        
        j = 0
        for i in range(n):
            # 不匹配时，根据前缀函数回溯
            while j > 0 and text[i] != self.pattern[j]:
                j = self.pi[j - 1]
            
            # 匹配时，j前进一步
            if text[i] == self.pattern[j]:
                j += 1
            
            # 完全匹配
            if j == m:
                result.append(i - m + 1)  # 记录匹配起始位置
                j = self.pi[j - 1]  # 继续搜索下一个匹配
        
        return result
    
    @staticmethod
    def find_all(text: str, pattern: str) -> list:
        """静态方法：一次性查找"""
        kmp = KMP(pattern)
        return kmp.search(text)


# 使用示例
text = "ABABABCABABABCABABC"
pattern = "ABABCABAB"

kmp = KMP(pattern)
positions = kmp.search(text)
print(positions)  # [2, 10]
```
:::

#### 复杂度分析

| 阶段 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 预处理（计算前缀函数） | O(m) | O(m) |
| 匹配过程 | O(n) | - |
| **总计** | **O(n+m)** | **O(m)** |

### Z函数（扩展KMP）

📌 **定义**
Z函数 `z[i]` 表示字符串 `s` 与其后缀 `s[i:]` 的最长公共前缀长度。通常 `z[0]` 定义为 0 或 n。

::: code-group
```cpp
// C++ Z函数计算
vector<int> zFunction(const string& s) {
    int n = s.length();
    vector<int> z(n, 0);
    
    // [l, r] 是当前最右的 Z-box
    // Z-box: 一个区间 [i, i+z[i]-1]，其中 s[i..i+z[i]-1] = s[0..z[i]-1]
    for (int i = 1, l = 0, r = 0; i < n; i++) {
        // 如果 i 在当前 Z-box 内，可以利用之前计算的结果
        if (i <= r) {
            z[i] = min(r - i + 1, z[i - l]);
        }
        
        // 暴力扩展
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        
        // 更新 Z-box
        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
    
    return z;
}

// 使用 Z 函数进行字符串匹配
vector<int> zSearch(const string& text, const string& pattern) {
    // 将模式串和文本串连接，中间用特殊字符分隔
    string combined = pattern + "$" + text;
    vector<int> z = zFunction(combined);
    vector<int> result;
    
    int m = pattern.length();
    // 查找 z 值等于模式串长度的位置
    for (int i = m + 1; i < combined.length(); i++) {
        if (z[i] == m) {
            result.push_back(i - m - 1);  // 转换为原文本串中的位置
        }
    }
    
    return result;
}
```

```python
# Python Z函数计算
def z_function(s: str) -> list:
    """计算字符串的 Z 函数"""
    n = len(s)
    z = [0] * n
    
    # [l, r] 是当前最右的 Z-box
    l = r = 0
    for i in range(1, n):
        # 如果 i 在当前 Z-box 内，可以利用之前计算的结果
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        
        # 暴力扩展
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        # 更新 Z-box
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z


def z_search(text: str, pattern: str) -> list:
    """使用 Z 函数进行字符串匹配"""
    # 将模式串和文本串连接
    combined = pattern + "$" + text
    z = z_function(combined)
    
    m = len(pattern)
    result = []
    
    # 查找 z 值等于模式串长度的位置
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            result.append(i - m - 1)
    
    return result


# 使用示例
s = "aabcaabxaaz"
z = z_function(s)
print(z)  # [0, 1, 0, 0, 3, 1, 0, 0, 2, 1, 0]

text = "ababababc"
pattern = "abab"
positions = z_search(text, pattern)
print(positions)  # [0, 2, 4]
```
:::

**时间复杂度**：O(n+m)，其中 n 是文本串长度，m 是模式串长度。

## Manacher算法

Manacher算法是求解最长回文子串的经典算法，时间复杂度为 O(n)。

### 回文串问题

📌 **回文串定义**
正读和反读都相同的字符串称为回文串，如 "aba"、"abba"、"abcba"。

**回文串分类**：
- **奇数长度回文**：中心是一个字符，如 "aba"
- **偶数长度回文**：中心是两个字符之间，如 "abba"

💡 **统一处理技巧**
在字符之间插入特殊字符（如 `#`），将所有回文转化为奇数长度：
- "aba" → "#a#b#a#"
- "abba" → "#a#b#b#a#"

### 最长回文子串

::: code-group
```cpp
// C++ Manacher算法实现
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

class Manacher {
private:
    string s;           // 原字符串
    string t;           // 处理后的字符串（插入特殊字符）
    vector<int> p;      // p[i] 表示以 t[i] 为中心的最长回文半径
    
public:
    Manacher(const string& str) : s(str) {
        // 预处理：插入特殊字符
        t = "#";
        for (char c : s) {
            t += c;
            t += "#";
        }
        
        int n = t.length();
        p.resize(n, 0);
        
        // Manacher 核心算法
        // center: 当前最右回文的中心
        // right: 当前最右回文的右边界
        int center = 0, right = 0;
        
        for (int i = 0; i < n; i++) {
            // 利用对称性加速计算
            if (i < right) {
                int mirror = 2 * center - i;  // i 关于 center 的对称点
                p[i] = min(right - i, p[mirror]);
            }
            
            // 暴力扩展
            while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n && 
                   t[i - p[i] - 1] == t[i + p[i] + 1]) {
                p[i]++;
            }
            
            // 更新最右回文
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
        }
    }
    
    // 获取最长回文子串
    string getLongestPalindrome() {
        int maxLen = 0, center = 0;
        
        // 找最大半径
        for (int i = 0; i < t.length(); i++) {
            if (p[i] > maxLen) {
                maxLen = p[i];
                center = i;
            }
        }
        
        // 转换回原字符串的位置
        int start = (center - maxLen) / 2;
        return s.substr(start, maxLen);
    }
    
    // 获取最长回文子串长度
    int getLongestPalindromeLength() {
        return *max_element(p.begin(), p.end());
    }
    
    // 判断 s[l..r] 是否为回文
    bool isPalindrome(int l, int r) {
        // 转换到处理后字符串的位置
        int tl = 2 * l + 1;  // 左端点在 t 中的位置
        int tr = 2 * r + 1;  // 右端点在 t 中的位置
        int center = (tl + tr) / 2;  // 中心位置
        int radius = (tr - tl) / 2 + 1;  // 需要的半径
        return p[center] >= radius;
    }
    
    // 获取以位置 i 为中心的最长回文半径
    int getRadius(int i) {
        return p[2 * i + 1] / 2;
    }
};

// 使用示例
int main() {
    string s = "babad";
    Manacher manacher(s);
    
    cout << "最长回文子串: " << manacher.getLongestPalindrome() << endl;
    // 输出: "bab" 或 "aba"
    
    cout << "最长回文长度: " << manacher.getLongestPalindromeLength() << endl;
    // 输出: 3
    
    return 0;
}
```

```python
# Python Manacher算法实现
class Manacher:
    def __init__(self, s: str):
        self.s = s
        # 预处理：插入特殊字符
        self.t = '#'.join('^{}$'.format(s))  # 加入边界字符防止越界
        # 或者简单处理：self.t = '#' + '#'.join(s) + '#'
        
        n = len(self.t)
        self.p = [0] * n  # 回文半径数组
        
        # Manacher 核心算法
        center = right = 0
        
        for i in range(1, n - 1):  # 跳过边界字符
            # 利用对称性加速计算
            if i < right:
                mirror = 2 * center - i
                self.p[i] = min(right - i, self.p[mirror])
            
            # 暴力扩展
            while self.t[i - self.p[i] - 1] == self.t[i + self.p[i] + 1]:
                self.p[i] += 1
            
            # 更新最右回文
            if i + self.p[i] > right:
                center, right = i, i + self.p[i]
    
    def get_longest_palindrome(self) -> str:
        """获取最长回文子串"""
        max_len = max(self.p)
        center = self.p.index(max_len)
        
        # 转换回原字符串
        start = (center - max_len) // 2
        return self.s[start:start + max_len]
    
    def get_longest_palindrome_length(self) -> int:
        """获取最长回文子串长度"""
        return max(self.p)
    
    def is_palindrome(self, l: int, r: int) -> bool:
        """判断 s[l:r+1] 是否为回文"""
        # 转换到处理后字符串的位置
        center = l + r + 1  # 在插入特殊字符后的中心位置
        radius = r - l + 1
        return self.p[center] >= radius


# 使用示例
s = "babad"
manacher = Manacher(s)

print(f"最长回文子串: {manacher.get_longest_palindrome()}")
# 输出: "bab" 或 "aba"

print(f"最长回文长度: {manacher.get_longest_palindrome_length()}")
# 输出: 3
```
:::

### 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 预处理（插入特殊字符） | O(n) | O(n) |
| 计算回文半径数组 | O(n) | O(n) |
| 查询最长回文 | O(n) | - |
| 判断子串是否回文 | O(1) | - |

💡 **为什么是 O(n)？**
虽然有两层循环，但 `right` 最多增加 n 次，内层 while 循环的总次数被 `right` 的增加次数限制，因此总时间复杂度为 O(n)。

## 典型题目

### 实现 strStr()

**题目描述**：给定两个字符串 `haystack` 和 `needle`，返回 `needle` 在 `haystack` 中第一次出现的位置。如果不存在，返回 -1。

::: code-group
```cpp
// C++ 实现 strStr()（使用KMP）
class Solution {
public:
    int strStr(string haystack, string needle) {
        if (needle.empty()) return 0;
        if (haystack.empty() || haystack.length() < needle.length()) return -1;
        
        // 计算前缀函数
        int m = needle.length();
        vector<int> pi(m, 0);
        
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && needle[i] != needle[j]) {
                j = pi[j - 1];
            }
            if (needle[i] == needle[j]) j++;
            pi[i] = j;
        }
        
        // KMP匹配
        int n = haystack.length();
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && haystack[i] != needle[j]) {
                j = pi[j - 1];
            }
            if (haystack[i] == needle[j]) j++;
            
            if (j == m) {
                return i - m + 1;  // 找到匹配
            }
        }
        
        return -1;
    }
};
```

```python
# Python 实现 strStr()（使用KMP）
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        if not haystack or len(haystack) < len(needle):
            return -1
        
        # 计算前缀函数
        m = len(needle)
        pi = [0] * m
        
        j = 0
        for i in range(1, m):
            while j > 0 and needle[i] != needle[j]:
                j = pi[j - 1]
            if needle[i] == needle[j]:
                j += 1
            pi[i] = j
        
        # KMP匹配
        j = 0
        for i, c in enumerate(haystack):
            while j > 0 and c != needle[j]:
                j = pi[j - 1]
            if c == needle[j]:
                j += 1
            
            if j == m:
                return i - m + 1
        
        return -1


# Python 简单实现（使用内置方法）
def strStr_simple(haystack: str, needle: str) -> int:
    return haystack.find(needle)
```
:::

### 重复的子字符串

**题目描述**：给定一个字符串 `s`，判断它是否可以由它的一个子串重复多次构成。

💡 **解题思路**
利用前缀函数的性质：如果字符串可以由子串重复构成，那么最长相等前后缀的长度 `pi[n-1]` 满足：`n - pi[n-1]` 是重复子串的长度。

::: code-group
```cpp
// C++ 重复的子字符串
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.length();
        if (n <= 1) return false;
        
        // 计算前缀函数
        vector<int> pi(n, 0);
        for (int i = 1, j = 0; i < n; i++) {
            while (j > 0 && s[i] != s[j]) {
                j = pi[j - 1];
            }
            if (s[i] == s[j]) j++;
            pi[i] = j;
        }
        
        // 检查是否可由子串重复构成
        int lastPi = pi[n - 1];
        // lastPi 是最长相等前后缀的长度
        // 如果 lastPi > 0 且 n 能被 (n - lastPi) 整除，则可以
        return lastPi > 0 && n % (n - lastPi) == 0;
    }
};

// 另一种思路：字符串拼接
bool repeatedSubstringPattern_v2(string s) {
    string t = s + s;
    // 去掉首尾字符，避免匹配到原串本身
    t = t.substr(1, t.length() - 2);
    return t.find(s) != string::npos;
}
```

```python
# Python 重复的子字符串
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False
        
        # 计算前缀函数
        pi = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        
        # 检查是否可由子串重复构成
        last_pi = pi[-1]
        return last_pi > 0 and n % (n - last_pi) == 0


# 另一种思路：字符串拼接
def repeated_substring_pattern_v2(s: str) -> bool:
    t = (s + s)[1:-1]  # 去掉首尾字符
    return s in t
```
:::

### 最长回文子串

**题目描述**：给定一个字符串 `s`，找到其中最长的回文子串。

::: code-group
```cpp
// C++ 最长回文子串（Manacher算法）
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.length();
        if (n <= 1) return s;
        
        // 预处理
        string t = "#";
        for (char c : s) {
            t += c;
            t += "#";
        }
        
        int m = t.length();
        vector<int> p(m, 0);
        int center = 0, right = 0;
        int maxCenter = 0, maxLen = 0;
        
        for (int i = 0; i < m; i++) {
            if (i < right) {
                p[i] = min(right - i, p[2 * center - i]);
            }
            
            while (i - p[i] - 1 >= 0 && i + p[i] + 1 < m && 
                   t[i - p[i] - 1] == t[i + p[i] + 1]) {
                p[i]++;
            }
            
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
            
            if (p[i] > maxLen) {
                maxLen = p[i];
                maxCenter = i;
            }
        }
        
        // 转换回原字符串
        int start = (maxCenter - maxLen) / 2;
        return s.substr(start, maxLen);
    }
};

// 动态规划解法（更直观）
string longestPalindrome_dp(string s) {
    int n = s.length();
    if (n <= 1) return s;
    
    // dp[i][j] 表示 s[i..j] 是否为回文
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    
    int maxLen = 1, start = 0;
    
    // 所有长度为1的子串都是回文
    for (int i = 0; i < n; i++) {
        dp[i][i] = true;
    }
    
    // 检查长度为2的子串
    for (int i = 0; i < n - 1; i++) {
        if (s[i] == s[i + 1]) {
            dp[i][i + 1] = true;
            start = i;
            maxLen = 2;
        }
    }
    
    // 检查长度 >= 3 的子串
    for (int len = 3; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            
            if (s[i] == s[j] && dp[i + 1][j - 1]) {
                dp[i][j] = true;
                start = i;
                maxLen = len;
            }
        }
    }
    
    return s.substr(start, maxLen);
}
```

```python
# Python 最长回文子串（Manacher算法）
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        
        # 预处理
        t = '#' + '#'.join(s) + '#'
        
        n = len(t)
        p = [0] * n
        center = right = 0
        max_center = max_len = 0
        
        for i in range(n):
            if i < right:
                p[i] = min(right - i, p[2 * center - i])
            
            while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and \
                  t[i - p[i] - 1] == t[i + p[i] + 1]:
                p[i] += 1
            
            if i + p[i] > right:
                center, right = i, i + p[i]
            
            if p[i] > max_len:
                max_len = p[i]
                max_center = i
        
        # 转换回原字符串
        start = (max_center - max_len) // 2
        return s[start:start + max_len]


# 动态规划解法
def longest_palindrome_dp(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    
    dp = [[False] * n for _ in range(n)]
    
    max_len = 1
    start = 0
    
    # 所有长度为1的子串都是回文
    for i in range(n):
        dp[i][i] = True
    
    # 检查长度为2的子串
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # 检查长度 >= 3 的子串
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


# 中心扩展法（推荐）
def longest_palindrome_expand(s: str) -> str:
    def expand_around_center(left: int, right: int) -> str:
        """从中心向两边扩展，返回找到的回文串"""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    result = ""
    for i in range(len(s)):
        # 奇数长度回文
        odd = expand_around_center(i, i)
        # 偶数长度回文
        even = expand_around_center(i, i + 1)
        
        # 更新最长回文
        if len(odd) > len(result):
            result = odd
        if len(even) > len(result):
            result = even
    
    return result
```
:::

## 算法对比

| 算法 | 适用场景 | 时间复杂度 | 空间复杂度 | 特点 |
|------|---------|-----------|-----------|------|
| 朴素匹配 | 简单场景 | O(n×m) | O(1) | 实现简单 |
| KMP | 单模式匹配 | O(n+m) | O(m) | 线性时间，需预处理 |
| Z函数 | 模式匹配、周期判断 | O(n) | O(n) | 实现简单 |
| 字符串哈希 | 子串相等判断 | O(n)预处理，O(1)查询 | O(n) | 查询快，有冲突风险 |
| Manacher | 回文问题 | O(n) | O(n) | 线性求最长回文 |

## 常见问题与注意事项

⚠️ **边界情况**
- 空字符串处理
- 单字符字符串
- 模式串比文本串长
- 重复字符的字符串

💡 **优化技巧**
- 使用双哈希降低冲突概率
- KMP预处理时同时计算多个模式串
- Manacher可以预处理后O(1)判断任意子串是否回文

📌 **调试建议**
- 打印前缀函数/next数组验证正确性
- 注意字符串索引从0还是1开始
- 处理好字符编码问题（尤其是中文字符）
