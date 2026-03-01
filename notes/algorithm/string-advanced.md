# 高级字符串算法

高级字符串算法是在基础字符串匹配算法之上的进阶内容，主要解决多模式匹配、高效子串查询等问题。这些算法在实际工程中有广泛应用，如搜索引擎、敏感词过滤、DNA序列分析等。

📌 **核心内容**
- **字典树（Trie）深化**：前缀统计、词频统计等应用
- **AC自动机**：多模式匹配的利器
- **后缀数组**：强大的子串处理工具
- **算法对比与典型题目**

## 字典树（Trie）深化

### 结构回顾

字典树（Trie，又称前缀树）是一种树形数据结构，用于高效地存储和检索字符串集合。

<TrieAnimation />

📌 **核心特性**
- 根节点不包含字符，其他节点各包含一个字符
- 从根节点到某一节点的路径上的字符连接起来，就是该节点对应的字符串
- 所有节点的子节点字符各不相同

::: code-group
```cpp
// C++ 字典树基础实现
#include <string>
#include <vector>
using namespace std;

class Trie {
private:
    struct TrieNode {
        TrieNode* children[26];  // 26个小写字母
        bool isEnd;              // 是否是单词结尾
        int count;               // 经过此节点的单词数量
        
        TrieNode() : isEnd(false), count(0) {
            for (int i = 0; i < 26; i++) {
                children[i] = nullptr;
            }
        }
    };
    
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    // 插入单词
    void insert(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
            node->count++;  // 统计经过此节点的单词数
        }
        node->isEnd = true;
    }
    
    // 搜索单词是否存在
    bool search(const string& word) {
        TrieNode* node = findNode(word);
        return node != nullptr && node->isEnd;
    }
    
    // 判断是否存在以 prefix 为前缀的单词
    bool startsWith(const string& prefix) {
        return findNode(prefix) != nullptr;
    }
    
    // 查找节点
    TrieNode* findNode(const string& s) {
        TrieNode* node = root;
        for (char c : s) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                return nullptr;
            }
            node = node->children[idx];
        }
        return node;
    }
};
```

```python
# Python 字典树基础实现
class TrieNode:
    def __init__(self):
        self.children = {}  # 字典存储子节点
        self.is_end = False
        self.count = 0      # 经过此节点的单词数量


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """插入单词"""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.count += 1
        node.is_end = True
    
    def search(self, word: str) -> bool:
        """搜索单词是否存在"""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """判断是否存在以 prefix 为前缀的单词"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, s: str) -> TrieNode:
        """查找字符串对应的节点"""
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node
```
:::

### 前缀统计

📌 **问题**：给定一组单词，统计以某个前缀开头的单词数量。

::: code-group
```cpp
// C++ 前缀统计
class TrieWithCount : public Trie {
public:
    // 统计以 prefix 为前缀的单词数量
    int countPrefix(const string& prefix) {
        TrieNode* node = findNode(prefix);
        if (!node) return 0;
        
        // 方法1：遍历子树统计（如果只存了isEnd）
        // return countWords(node);
        
        // 方法2：直接返回count（如果每个节点存储了经过它的单词数）
        return node->count;
    }
    
private:
    // 遍历子树统计单词数量
    int countWords(TrieNode* node) {
        int cnt = node->isEnd ? 1 : 0;
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                cnt += countWords(node->children[i]);
            }
        }
        return cnt;
    }
};

// 使用示例
int main() {
    TrieWithCount trie;
    vector<string> words = {"apple", "app", "application", "apply", "banana"};
    
    for (const string& w : words) {
        trie.insert(w);
    }
    
    cout << "以 'app' 开头的单词数: " << trie.countPrefix("app") << endl;
    // 输出: 4 (apple, app, application, apply)
    
    return 0;
}
```

```python
# Python 前缀统计
class TrieWithCount(Trie):
    def count_prefix(self, prefix: str) -> int:
        """统计以 prefix 为前缀的单词数量"""
        node = self._find_node(prefix)
        if not node:
            return 0
        return node.count
    
    def count_words_from_node(self, node: TrieNode) -> int:
        """遍历子树统计单词数量（备用方法）"""
        cnt = 1 if node.is_end else 0
        for child in node.children.values():
            cnt += self.count_words_from_node(child)
        return cnt


# 使用示例
trie = TrieWithCount()
words = ["apple", "app", "application", "apply", "banana"]

for w in words:
    trie.insert(w)

print(f"以 'app' 开头的单词数: {trie.count_prefix('app')}")
# 输出: 4 (apple, app, application, apply)
```
:::

### 词频统计

📌 **问题**：统计每个单词出现的次数，支持动态更新。

::: code-group
```cpp
// C++ 词频统计字典树
class TrieFrequency {
private:
    struct TrieNode {
        TrieNode* children[26];
        int freq;      // 完整单词的词频
        int passCount; // 经过此节点的次数
        
        TrieNode() : freq(0), passCount(0) {
            for (int i = 0; i < 26; i++) {
                children[i] = nullptr;
            }
        }
    };
    
    TrieNode* root;
    
public:
    TrieFrequency() : root(new TrieNode()) {}
    
    // 插入单词（增加词频）
    void insert(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
            node->passCount++;
        }
        node->freq++;
    }
    
    // 获取单词词频
    int getFrequency(const string& word) {
        TrieNode* node = findNode(word);
        return node ? node->freq : 0;
    }
    
    // 删除单词（减少词频）
    void remove(const string& word) {
        TrieNode* node = findNode(word);
        if (!node || node->freq == 0) return;
        
        node->freq--;
        // 更新 passCount
        node = root;
        for (char c : word) {
            int idx = c - 'a';
            node = node->children[idx];
            node->passCount--;
        }
    }
    
    // 获取所有单词及其词频
    vector<pair<string, int>> getAllWords() {
        vector<pair<string, int>> result;
        collectWords(root, "", result);
        return result;
    }
    
private:
    TrieNode* findNode(const string& s) {
        TrieNode* node = root;
        for (char c : s) {
            int idx = c - 'a';
            if (!node->children[idx]) return nullptr;
            node = node->children[idx];
        }
        return node;
    }
    
    void collectWords(TrieNode* node, string path, vector<pair<string, int>>& result) {
        if (node->freq > 0) {
            result.push_back({path, node->freq});
        }
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                collectWords(node->children[i], path + char('a' + i), result);
            }
        }
    }
};
```

```python
# Python 词频统计字典树
class TrieFrequency:
    def __init__(self):
        self.root = {'freq': 0, 'pass_count': 0}
    
    def insert(self, word: str) -> None:
        """插入单词（增加词频）"""
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {'freq': 0, 'pass_count': 0}
            node = node[c]
            node['pass_count'] += 1
        node['freq'] += 1
    
    def get_frequency(self, word: str) -> int:
        """获取单词词频"""
        node = self._find_node(word)
        return node['freq'] if node else 0
    
    def remove(self, word: str) -> None:
        """删除单词（减少词频）"""
        node = self._find_node(word)
        if not node or node['freq'] == 0:
            return
        
        node['freq'] -= 1
        # 更新 pass_count
        node = self.root
        for c in word:
            node = node[c]
            node['pass_count'] -= 1
    
    def get_all_words(self) -> list:
        """获取所有单词及其词频"""
        result = []
        self._collect_words(self.root, "", result)
        return result
    
    def _find_node(self, s: str) -> dict:
        node = self.root
        for c in s:
            if c not in node:
                return None
            node = node[c]
        return node
    
    def _collect_words(self, node: dict, path: str, result: list) -> None:
        if node['freq'] > 0:
            result.append((path, node['freq']))
        for c in node:
            if c not in ('freq', 'pass_count'):
                self._collect_words(node[c], path + c, result)


# 使用示例
trie = TrieFrequency()
words = ["apple", "apple", "app", "banana", "apple", "app"]

for w in words:
    trie.insert(w)

print(f"'apple' 词频: {trie.get_frequency('apple')}")  # 3
print(f"'app' 词频: {trie.get_frequency('app')}")      # 2
print("所有单词:", trie.get_all_words())
```
:::

## AC自动机

AC自动机（Aho-Corasick Automaton）是一种高效的多模式匹配算法，可以在 O(n+m+z) 时间内完成文本匹配，其中 n 是文本长度，m 是所有模式串的总长度，z 是匹配结果数。

### 多模式匹配原理

📌 **核心思想**
AC自动机结合了字典树和KMP算法的思想：
1. 将所有模式串构建成一棵字典树
2. 为每个节点计算失配指针（fail指针）
3. 在文本串上进行一次扫描，找出所有匹配

💡 **与KMP的类比**
| KMP算法 | AC自动机 |
|---------|----------|
| 单模式匹配 | 多模式匹配 |
| next数组（前缀函数） | fail指针 |
| 模式串自匹配 | 字典树节点间的跳转 |

### 失配指针（fail指针）

📌 **fail指针定义**
对于字典树上的节点 u，其 fail 指针指向另一个节点 v，满足：
- v 是 u 的最长真后缀在字典树中对应的节点
- 如果没有这样的节点，则 fail 指向根节点

**作用**：当在当前节点失配时，跳转到 fail 指向的节点继续匹配，避免从头开始。

### 构建过程

::: code-group
```cpp
// C++ AC自动机实现
#include <string>
#include <vector>
#include <queue>
using namespace std;

class ACAutomaton {
private:
    struct ACNode {
        ACNode* children[26];  // 子节点
        ACNode* fail;          // 失配指针
        vector<int> output;    // 匹配的模式串编号（可能有多个模式串在此结束）
        
        ACNode() : fail(nullptr) {
            for (int i = 0; i < 26; i++) {
                children[i] = nullptr;
            }
        }
    };
    
    ACNode* root;
    vector<string> patterns;  // 存储模式串
    
public:
    ACAutomaton() {
        root = new ACNode();
    }
    
    // 添加模式串
    void addPattern(const string& pattern, int id = -1) {
        if (id == -1) id = patterns.size();
        patterns.push_back(pattern);
        
        ACNode* node = root;
        for (char c : pattern) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new ACNode();
            }
            node = node->children[idx];
        }
        node->output.push_back(id);  // 标记模式串结束
    }
    
    // 构建fail指针（BFS）
    void build() {
        queue<ACNode*> q;
        root->fail = root;  // 根节点的fail指向自己
        
        // 第一层节点的fail指向根节点
        for (int i = 0; i < 26; i++) {
            if (root->children[i]) {
                root->children[i]->fail = root;
                q.push(root->children[i]);
            } else {
                root->children[i] = root;  // 空指针指向根，简化代码
            }
        }
        
        // BFS构建fail指针
        while (!q.empty()) {
            ACNode* curr = q.front();
            q.pop();
            
            for (int i = 0; i < 26; i++) {
                ACNode* child = curr->children[i];
                if (child) {
                    // child的fail = curr的fail对应位置的子节点
                    child->fail = curr->fail->children[i];
                    
                    // 合并fail节点的output（重要！）
                    for (int id : child->fail->output) {
                        child->output.push_back(id);
                    }
                    
                    q.push(child);
                } else {
                    // 路径压缩：直接指向fail节点的对应子节点
                    curr->children[i] = curr->fail->children[i];
                }
            }
        }
    }
    
    // 在文本中查找所有匹配
    vector<pair<int, int>> search(const string& text) {
        vector<pair<int, int>> result;  // (结束位置, 模式串编号)
        
        ACNode* node = root;
        for (int i = 0; i < text.length(); i++) {
            int idx = text[i] - 'a';
            node = node->children[idx];  // 自动跳转（包括fail跳转）
            
            // 收集所有匹配
            for (int id : node->output) {
                int startPos = i - patterns[id].length() + 1;
                result.push_back({startPos, id});
            }
        }
        
        return result;
    }
    
    // 获取匹配的模式串
    string getPattern(int id) {
        return patterns[id];
    }
};

// 使用示例
int main() {
    ACAutomaton ac;
    
    // 添加敏感词
    ac.addPattern("he");      // id = 0
    ac.addPattern("she");     // id = 1
    ac.addPattern("his");     // id = 2
    ac.addPattern("hers");    // id = 3
    
    // 构建fail指针
    ac.build();
    
    // 在文本中搜索
    string text = "ushers";
    auto matches = ac.search(text);
    
    // 输出匹配结果
    for (auto& [pos, id] : matches) {
        cout << "位置 " << pos << ": " << ac.getPattern(id) << endl;
    }
    // 输出:
    // 位置 1: she
    // 位置 2: he
    // 位置 2: hers
    
    return 0;
}
```

```python
# Python AC自动机实现
from collections import deque

class ACNode:
    def __init__(self):
        self.children = {}      # 子节点字典
        self.fail = None        # 失配指针
        self.output = []        # 匹配的模式串编号


class ACAutomaton:
    def __init__(self):
        self.root = ACNode()
        self.patterns = []      # 存储模式串
    
    def add_pattern(self, pattern: str, id: int = -1) -> None:
        """添加模式串"""
        if id == -1:
            id = len(self.patterns)
        self.patterns.append(pattern)
        
        node = self.root
        for c in pattern:
            if c not in node.children:
                node.children[c] = ACNode()
            node = node.children[c]
        node.output.append(id)
    
    def build(self) -> None:
        """构建fail指针（BFS）"""
        q = deque()
        self.root.fail = self.root
        
        # 第一层节点的fail指向根节点
        for c, child in self.root.children.items():
            child.fail = self.root
            q.append(child)
        
        # BFS构建fail指针
        while q:
            curr = q.popleft()
            
            for c, child in curr.children.items():
                # 沿着fail链找到有c子节点的节点
                fail_node = curr.fail
                while c not in fail_node.children and fail_node != self.root:
                    fail_node = fail_node.fail
                
                # 设置child的fail指针
                if c in fail_node.children:
                    child.fail = fail_node.children[c]
                else:
                    child.fail = self.root
                
                # 合并fail节点的output
                child.output.extend(child.fail.output)
                
                q.append(child)
    
    def search(self, text: str) -> list:
        """在文本中查找所有匹配"""
        result = []  # [(起始位置, 模式串编号)]
        
        node = self.root
        for i, c in enumerate(text):
            # 沿着fail链找到有c子节点的节点
            while c not in node.children and node != self.root:
                node = node.fail
            
            if c in node.children:
                node = node.children[c]
            else:
                node = self.root
            
            # 收集所有匹配
            for pattern_id in node.output:
                start = i - len(self.patterns[pattern_id]) + 1
                result.append((start, pattern_id))
        
        return result
    
    def get_pattern(self, id: int) -> str:
        """获取模式串"""
        return self.patterns[id]


# 使用示例
ac = ACAutomaton()

# 添加敏感词
ac.add_pattern("he")       # id = 0
ac.add_pattern("she")      # id = 1
ac.add_pattern("his")      # id = 2
ac.add_pattern("hers")     # id = 3

# 构建fail指针
ac.build()

# 在文本中搜索
text = "ushers"
matches = ac.search(text)

# 输出匹配结果
for pos, pattern_id in matches:
    print(f"位置 {pos}: {ac.get_pattern(pattern_id)}")
# 输出:
# 位置 1: she
# 位置 2: he
# 位置 2: hers
```
:::

### 图解fail指针构建

```
模式串: "he", "she", "his", "hers"

字典树结构:
          root
         /  |  \
        h   s   ...
        |   |
        e   h
       /|   |
      r i   e
      | |   
      s s
      
fail指针:
- h.fail = root
- s.fail = root
- e(he下).fail = root
- h(she下).fail = h(root下)
- e(she下).fail = e(he下)
- i.fail = root
- s.fail = root
- r.fail = root
- s(hers下).fail = s(his下)
```

### 典型应用：敏感词过滤

::: code-group
```cpp
// C++ 敏感词过滤器
class SensitiveWordFilter {
private:
    ACAutomaton ac;
    bool built;
    
public:
    SensitiveWordFilter() : built(false) {}
    
    // 添加敏感词
    void addWord(const string& word) {
        ac.addPattern(word);
        built = false;  // 需要重新构建
    }
    
    // 批量添加敏感词
    void addWords(const vector<string>& words) {
        for (const string& w : words) {
            ac.addPattern(w);
        }
        built = false;
    }
    
    // 构建AC自动机
    void build() {
        if (!built) {
            ac.build();
            built = true;
        }
    }
    
    // 检测是否包含敏感词
    bool contains(const string& text) {
        build();
        return !ac.search(text).empty();
    }
    
    // 过滤敏感词（替换为***）
    string filter(const string& text, const string& mask = "***") {
        build();
        auto matches = ac.search(text);
        
        if (matches.empty()) return text;
        
        string result = text;
        // 从后向前替换，避免位置偏移
        sort(matches.begin(), matches.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (auto& [pos, id] : matches) {
            int len = ac.getPattern(id).length();
            result.replace(pos, len, mask);
        }
        
        return result;
    }
    
    // 获取所有敏感词及其位置
    vector<pair<string, int>> findAll(const string& text) {
        build();
        auto matches = ac.search(text);
        vector<pair<string, int>> result;
        
        for (auto& [pos, id] : matches) {
            result.push_back({ac.getPattern(id), pos});
        }
        
        return result;
    }
};

// 使用示例
int main() {
    SensitiveWordFilter filter;
    
    // 添加敏感词
    filter.addWords({"敏感词", "违禁词", "不良信息"});
    
    string text = "这是一条包含敏感词和违禁词的文本";
    
    // 过滤
    cout << "原文: " << text << endl;
    cout << "过滤后: " << filter.filter(text) << endl;
    // 输出: 这是一条包含***和***的文本
    
    // 检测
    if (filter.contains(text)) {
        cout << "检测到敏感词!" << endl;
    }
    
    return 0;
}
```

```python
# Python 敏感词过滤器
class SensitiveWordFilter:
    def __init__(self):
        self.ac = ACAutomaton()
        self.built = False
    
    def add_word(self, word: str) -> None:
        """添加敏感词"""
        self.ac.add_pattern(word)
        self.built = False
    
    def add_words(self, words: list) -> None:
        """批量添加敏感词"""
        for w in words:
            self.ac.add_pattern(w)
        self.built = False
    
    def build(self) -> None:
        """构建AC自动机"""
        if not self.built:
            self.ac.build()
            self.built = True
    
    def contains(self, text: str) -> bool:
        """检测是否包含敏感词"""
        self.build()
        return len(self.ac.search(text)) > 0
    
    def filter(self, text: str, mask: str = "***") -> str:
        """过滤敏感词"""
        self.build()
        matches = self.ac.search(text)
        
        if not matches:
            return text
        
        # 转为列表，按位置从后向前排序
        matches = sorted(matches, key=lambda x: x[0], reverse=True)
        
        result = list(text)
        for pos, pattern_id in matches:
            pattern = self.ac.get_pattern(pattern_id)
            end = pos + len(pattern)
            result[pos:end] = list(mask)
        
        return ''.join(result)
    
    def find_all(self, text: str) -> list:
        """获取所有敏感词及其位置"""
        self.build()
        matches = self.ac.search(text)
        return [(self.ac.get_pattern(id), pos) for pos, id in matches]


# 使用示例
filter = SensitiveWordFilter()

# 添加敏感词
filter.add_words(["敏感词", "违禁词", "不良信息"])

text = "这是一条包含敏感词和违禁词的文本"

# 过滤
print(f"原文: {text}")
print(f"过滤后: {filter.filter(text)}")
# 输出: 这是一条包含***和***的文本

# 检测
if filter.contains(text):
    print("检测到敏感词!")

# 查找所有
print("所有敏感词:", filter.find_all(text))
```
:::

## 后缀数组

后缀数组是一种强大的字符串数据结构，可以高效解决许多字符串问题，如最长公共子串、最长重复子串、字符串匹配等。

### 后缀数组定义

📌 **基本概念**
- **后缀**：字符串从某个位置开始到末尾的子串。后缀 `i` 表示从位置 `i` 开始的后缀 `s[i:]`
- **后缀数组 SA**：将所有后缀按字典序排序后，存储后缀起始位置的数组
- **排名数组 Rank**：`Rank[i]` 表示后缀 `i` 在排序后的排名（SA的逆数组）

**关系**：`SA[Rank[i]] = i`，`Rank[SA[i]] = i`

::: code-group
```cpp
// C++ 后缀数组朴素构建（用于理解）
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

// 朴素O(n^2 log n)构建方法
vector<int> buildSuffixArrayNaive(const string& s) {
    int n = s.length();
    vector<pair<string, int>> suffixes;
    
    // 生成所有后缀
    for (int i = 0; i < n; i++) {
        suffixes.push_back({s.substr(i), i});
    }
    
    // 按字典序排序
    sort(suffixes.begin(), suffixes.end());
    
    // 提取后缀数组
    vector<int> SA(n);
    for (int i = 0; i < n; i++) {
        SA[i] = suffixes[i].second;
    }
    
    return SA;
}

// 使用示例
int main() {
    string s = "banana";
    vector<int> sa = buildSuffixArrayNaive(s);
    
    // 输出后缀数组
    cout << "后缀数组 SA: ";
    for (int i : sa) cout << i << " ";
    cout << endl;
    // 输出: 5 3 1 0 4 2
    
    // 输出排序后的后缀
    cout << "排序后的后缀:" << endl;
    for (int i : sa) {
        cout << "SA[" << i << "] = " << s.substr(i) << endl;
    }
    // 输出:
    // SA[5] = a
    // SA[3] = ana
    // SA[1] = anana
    // SA[0] = banana
    // SA[4] = na
    // SA[2] = nana
    
    return 0;
}
```

```python
# Python 后缀数组朴素构建
def build_suffix_array_naive(s: str) -> list:
    """朴素O(n^2 log n)构建方法"""
    n = len(s)
    # 生成所有后缀及其起始位置
    suffixes = [(s[i:], i) for i in range(n)]
    
    # 按字典序排序
    suffixes.sort()
    
    # 提取后缀数组
    return [pos for _, pos in suffixes]


# 使用示例
s = "banana"
sa = build_suffix_array_naive(s)

print(f"后缀数组 SA: {sa}")
# 输出: [5, 3, 1, 0, 4, 2]

print("排序后的后缀:")
for i in sa:
    print(f"SA[{i}] = {s[i:]}")
# 输出:
# SA[5] = a
# SA[3] = ana
# SA[1] = anana
# SA[0] = banana
# SA[4] = na
# SA[2] = nana
```
:::

### SA与Rank数组

📌 **SA与Rank的转换关系**

```
字符串: "banana"

索引:     0  1  2  3  4  5
字符:     b  a  n  a  n  a

后缀:
  0: banana
  1: anana
  2: nana
  3: ana
  4: na
  5: a

排序后:
  排名  起始位置  后缀
  0     5        a
  1     3        ana
  2     1        anana
  3     0        banana
  4     4        na
  5     2        nana

SA = [5, 3, 1, 0, 4, 2]
Rank = [3, 2, 5, 1, 4, 0]
```

::: code-group
```cpp
// C++ 由SA构建Rank
vector<int> buildRankFromSA(const vector<int>& SA) {
    int n = SA.size();
    vector<int> Rank(n);
    
    for (int i = 0; i < n; i++) {
        Rank[SA[i]] = i;
    }
    
    return Rank;
}

// 由Rank构建SA
vector<int> buildSAFromRank(const vector<int>& Rank) {
    int n = Rank.size();
    vector<int> SA(n);
    
    for (int i = 0; i < n; i++) {
        SA[Rank[i]] = i;
    }
    
    return SA;
}
```

```python
# Python 由SA构建Rank
def build_rank_from_sa(sa: list) -> list:
    n = len(sa)
    rank = [0] * n
    for i, pos in enumerate(sa):
        rank[pos] = i
    return rank

# 由Rank构建SA
def build_sa_from_rank(rank: list) -> list:
    n = len(rank)
    sa = [0] * n
    for i, r in enumerate(rank):
        sa[r] = i
    return sa
```
:::

### 倍增算法

📌 **算法思想**
倍增算法通过多次排序，每次比较长度翻倍，最终得到完整的后缀数组。

**时间复杂度**：O(n log n)

::: code-group
```cpp
// C++ 倍增算法构建后缀数组
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

vector<int> buildSuffixArray(const string& s) {
    int n = s.length();
    if (n == 0) return {};
    if (n == 1) return {0};
    
    // 初始：按单个字符排序
    vector<int> SA(n), Rank(n), tmpRank(n);
    
    for (int i = 0; i < n; i++) {
        SA[i] = i;
        Rank[i] = s[i];  // 初始rank为字符的ASCII值
    }
    
    // 倍增：每次比较长度翻倍
    for (int k = 1; k < n; k *= 2) {
        // 排序关键字：第二关键字 + 第一关键字
        auto cmp = [&](int a, int b) {
            // 比较第一关键字
            if (Rank[a] != Rank[b]) {
                return Rank[a] < Rank[b];
            }
            // 比较第二关键字
            int ra = (a + k < n) ? Rank[a + k] : -1;
            int rb = (b + k < n) ? Rank[b + k] : -1;
            return ra < rb;
        };
        
        sort(SA.begin(), SA.end(), cmp);
        
        // 重新计算rank
        tmpRank[SA[0]] = 0;
        for (int i = 1; i < n; i++) {
            tmpRank[SA[i]] = tmpRank[SA[i - 1]];
            if (cmp(SA[i - 1], SA[i]) || cmp(SA[i], SA[i - 1])) {
                tmpRank[SA[i]]++;
            }
        }
        Rank = tmpRank;
        
        // 如果所有rank都不同，排序完成
        if (Rank[SA[n - 1]] == n - 1) break;
    }
    
    return SA;
}

// 更高效的倍增实现（使用基数排序）
vector<int> buildSuffixArrayFast(const string& s) {
    int n = s.length();
    vector<int> SA(n), Rank(n), tmpSA(n), tmpRank(n);
    
    // 初始化
    for (int i = 0; i < n; i++) {
        SA[i] = i;
        Rank[i] = s[i];
    }
    
    // 基数排序的计数数组
    auto radixSort = [&](int k) {
        int maxRank = max(256, n);
        vector<int> cnt(maxRank + 1, 0);
        
        // 统计第二关键字
        for (int i = 0; i < n; i++) {
            cnt[(i + k < n) ? Rank[i + k] + 1 : 1]++;
        }
        // 前缀和
        for (int i = 1; i <= maxRank; i++) {
            cnt[i] += cnt[i - 1];
        }
        // 按第二关键字排序到tmpSA
        for (int i = n - 1; i >= 0; i--) {
            int r = (SA[i] + k < n) ? Rank[SA[i] + k] + 1 : 0;
            tmpSA[--cnt[r]] = SA[i];
        }
        SA = tmpSA;
        
        // 按第一关键字排序
        cnt.assign(maxRank + 1, 0);
        for (int i = 0; i < n; i++) {
            cnt[Rank[SA[i]] + 1]++;
        }
        for (int i = 1; i <= maxRank; i++) {
            cnt[i] += cnt[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            tmpSA[--cnt[Rank[SA[i]] + 1]] = SA[i];
        }
        SA = tmpSA;
    };
    
    // 倍增过程
    for (int k = 1; k < n; k *= 2) {
        radixSort(k);
        
        // 重新计算rank
        tmpRank[SA[0]] = 0;
        for (int i = 1; i < n; i++) {
            tmpRank[SA[i]] = tmpRank[SA[i - 1]];
            bool same = (Rank[SA[i]] == Rank[SA[i - 1]]) &&
                        ((SA[i] + k >= n && SA[i - 1] + k >= n) ||
                         (SA[i] + k < n && SA[i - 1] + k < n && 
                          Rank[SA[i] + k] == Rank[SA[i - 1] + k]));
            if (!same) tmpRank[SA[i]]++;
        }
        Rank = tmpRank;
        
        if (Rank[SA[n - 1]] == n - 1) break;
    }
    
    return SA;
}
```

```python
# Python 倍增算法构建后缀数组
def build_suffix_array(s: str) -> list:
    """倍增算法构建后缀数组 O(n log n)"""
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # 初始：按单个字符排序
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp_rank = [0] * n
    
    k = 1
    while k < n:
        # 排序关键字：第二关键字 + 第一关键字
        def cmp_key(i):
            # (第一关键字, 第二关键字)
            return (rank[i], rank[i + k] if i + k < n else -1)
        
        sa.sort(key=cmp_key)
        
        # 重新计算rank
        tmp_rank[sa[0]] = 0
        for i in range(1, n):
            prev_key = cmp_key(sa[i - 1])
            curr_key = cmp_key(sa[i])
            tmp_rank[sa[i]] = tmp_rank[sa[i - 1]]
            if prev_key != curr_key:
                tmp_rank[sa[i]] += 1
        
        rank = tmp_rank[:]
        
        # 如果所有rank都不同，排序完成
        if rank[sa[n - 1]] == n - 1:
            break
        
        k *= 2
    
    return sa


# 使用示例
s = "banana"
sa = build_suffix_array(s)
print(f"后缀数组: {sa}")  # [5, 3, 1, 0, 4, 2]
```
:::

### Height数组

📌 **定义**
`Height[i]` 表示排名为 `i` 和 `i-1` 的两个相邻后缀的最长公共前缀长度。

**计算公式**：
$$Height[i] = LCP(SA[i], SA[i-1])$$

其中 LCP 表示最长公共前缀（Longest Common Prefix）。

💡 **重要性质**
- `Height[Rank[i]] >= Height[Rank[i-1]] - 1`
- 利用这个性质，可以在 O(n) 时间内计算 Height 数组

::: code-group
```cpp
// C++ 计算Height数组（O(n)算法）
#include <string>
#include <vector>
using namespace std;

// 需要先有SA和Rank数组
vector<int> buildHeightArray(const string& s, const vector<int>& SA, const vector<int>& Rank) {
    int n = s.length();
    vector<int> Height(n, 0);
    
    // 利用性质：Height[Rank[i]] >= Height[Rank[i-1]] - 1
    int k = 0;  // 当前LCP长度
    
    for (int i = 0; i < n; i++) {
        if (Rank[i] == 0) {
            k = 0;
            continue;
        }
        
        // j是排名比i前一位的后缀起始位置
        int j = SA[Rank[i] - 1];
        
        // 从上一个位置继续比较
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) {
            k++;
        }
        
        Height[Rank[i]] = k;
        
        // 下一个位置至少是 k - 1
        if (k > 0) k--;
    }
    
    return Height;
}

// 完整的后缀数组类
class SuffixArray {
private:
    string s;
    vector<int> SA;
    vector<int> Rank;
    vector<int> Height;
    
public:
    SuffixArray(const string& str) : s(str) {
        int n = s.length();
        
        // 构建SA
        SA = buildSuffixArray(s);
        
        // 构建Rank
        Rank.resize(n);
        for (int i = 0; i < n; i++) {
            Rank[SA[i]] = i;
        }
        
        // 构建Height
        Height = buildHeightArray(s, SA, Rank);
    }
    
    // 获取SA数组
    vector<int> getSA() const { return SA; }
    
    // 获取Rank数组
    vector<int> getRank() const { return Rank; }
    
    // 获取Height数组
    vector<int> getHeight() const { return Height; }
    
    // 计算任意两个后缀的LCP
    int lcp(int i, int j) {
        if (i == j) return s.length() - i;
        
        int ri = Rank[i], rj = Rank[j];
        if (ri > rj) swap(ri, rj);
        
        // RMQ问题，这里用简单方法
        int result = Height[ri + 1];
        for (int k = ri + 2; k <= rj; k++) {
            result = min(result, Height[k]);
        }
        return result;
    }
    
    // 比较两个子串的字典序
    int compare(int l1, int r1, int l2, int r2) {
        int len1 = r1 - l1 + 1, len2 = r2 - l2 + 1;
        int common = lcp(l1, l2);
        
        if (common >= len1 && common >= len2) {
            return len1 - len2;  // 一个是另一个的前缀
        }
        if (common >= len1) return -1;  // s[l1..r1] 是 s[l2..r2] 的前缀
        if (common >= len2) return 1;   // s[l2..r2] 是 s[l1..r1] 的前缀
        
        return Rank[l1] - Rank[l2];  // 根据排名判断
    }
};

// 使用示例
int main() {
    string s = "banana";
    SuffixArray sa(s);
    
    cout << "SA: ";
    for (int i : sa.getSA()) cout << i << " ";
    cout << endl;
    
    cout << "Height: ";
    for (int i : sa.getHeight()) cout << i << " ";
    cout << endl;
    
    // 计算LCP
    cout << "LCP(1, 3): " << sa.lcp(1, 3) << endl;  // 后缀"anana"和"ana"的LCP
    
    return 0;
}
```

```python
# Python 计算Height数组
def build_height_array(s: str, sa: list, rank: list) -> list:
    """计算Height数组 O(n)"""
    n = len(s)
    height = [0] * n
    k = 0
    
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        
        # j是排名比i前一位的后缀起始位置
        j = sa[rank[i] - 1]
        
        # 从上一个位置继续比较
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        
        height[rank[i]] = k
        
        if k > 0:
            k -= 1
    
    return height


# 完整的后缀数组类
class SuffixArray:
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        
        # 构建SA
        self.sa = build_suffix_array(s)
        
        # 构建Rank
        self.rank = [0] * self.n
        for i, pos in enumerate(self.sa):
            self.rank[pos] = i
        
        # 构建Height
        self.height = build_height_array(s, self.sa, self.rank)
    
    def get_sa(self) -> list:
        return self.sa
    
    def get_rank(self) -> list:
        return self.rank
    
    def get_height(self) -> list:
        return self.height
    
    def lcp(self, i: int, j: int) -> int:
        """计算任意两个后缀的LCP"""
        if i == j:
            return self.n - i
        
        ri, rj = self.rank[i], self.rank[j]
        if ri > rj:
            ri, rj = rj, ri
        
        # RMQ查询（简单方法，可用线段树/ST表优化）
        result = self.height[ri + 1]
        for k in range(ri + 2, rj + 1):
            result = min(result, self.height[k])
        return result


# 使用示例
s = "banana"
sa = SuffixArray(s)

print(f"SA: {sa.get_sa()}")
print(f"Rank: {sa.get_rank()}")
print(f"Height: {sa.get_height()}")

print(f"LCP(1, 3): {sa.lcp(1, 3)}")
```
:::

**Height数组示例**：
```
字符串: "banana"

排名  SA[i]  后缀      Height[i]
0     5      a         0
1     3      ana       1      (a 与 ana 的LCP = 1)
2     1      anana     3      (ana 与 anana 的LCP = 3)
3     0      banana    0      (anana 与 banana 的LCP = 0)
4     4      na        0      (banana 与 na 的LCP = 0)
5     2      nana      2      (na 与 nana 的LCP = 2)
```

### 应用：最长公共子串

📌 **问题**：给定两个字符串，求它们的最长公共子串。

💡 **思路**：将两个字符串用一个特殊字符连接，构建后缀数组，然后在 Height 数组中找相邻且来自不同字符串的最大值。

::: code-group
```cpp
// C++ 最长公共子串
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

string longestCommonSubstring(const string& s1, const string& s2) {
    // 用特殊字符连接两个字符串
    string combined = s1 + "#" + s2 + "$";
    int n1 = s1.length();
    int n = combined.length();
    
    // 构建后缀数组
    SuffixArray sa(combined);
    auto SA = sa.getSA();
    auto Height = sa.getHeight();
    
    // 找最大的Height，且相邻后缀来自不同字符串
    int maxLen = 0, pos = -1;
    
    for (int i = 1; i < n; i++) {
        // 判断两个相邻后缀是否来自不同字符串
        // SA[i-1] 在 s1 中当且仅当 SA[i-1] < n1
        // SA[i] 在 s1 中当且仅当 SA[i] < n1
        bool fromS1_prev = SA[i - 1] < n1;
        bool fromS1_curr = SA[i] < n1;
        
        // 来自不同字符串
        if (fromS1_prev != fromS1_curr) {
            if (Height[i] > maxLen) {
                maxLen = Height[i];
                pos = SA[i];
            }
        }
    }
    
    if (maxLen == 0) return "";
    return combined.substr(pos, maxLen);
}

// 使用示例
int main() {
    string s1 = "abcdefg";
    string s2 = "xbcdex";
    
    string lcs = longestCommonSubstring(s1, s2);
    cout << "最长公共子串: " << lcs << endl;  // 输出: "bcde"
    
    return 0;
}
```

```python
# Python 最长公共子串
def longest_common_substring(s1: str, s2: str) -> str:
    # 用特殊字符连接两个字符串
    combined = s1 + "#" + s2 + "$"
    n1 = len(s1)
    
    # 构建后缀数组
    sa = SuffixArray(combined)
    SA = sa.get_sa()
    Height = sa.get_height()
    
    # 找最大的Height，且相邻后缀来自不同字符串
    max_len = 0
    pos = -1
    
    for i in range(1, len(combined)):
        # 判断两个相邻后缀是否来自不同字符串
        from_s1_prev = SA[i - 1] < n1
        from_s1_curr = SA[i] < n1
        
        # 来自不同字符串
        if from_s1_prev != from_s1_curr:
            if Height[i] > max_len:
                max_len = Height[i]
                pos = SA[i]
    
    if max_len == 0:
        return ""
    return combined[pos:pos + max_len]


# 使用示例
s1 = "abcdefg"
s2 = "xbcdex"
lcs = longest_common_substring(s1, s2)
print(f"最长公共子串: {lcs}")  # 输出: "bcde"
```
:::

## 字符串算法对比

### 时间空间复杂度

| 算法 | 预处理时间 | 匹配时间 | 空间复杂度 | 适用场景 |
|------|-----------|---------|-----------|---------|
| 朴素匹配 | O(1) | O(n×m) | O(1) | 简单场景 |
| KMP | O(m) | O(n) | O(m) | 单模式匹配 |
| Z函数 | O(n+m) | - | O(n+m) | 单模式匹配 |
| 字典树 | O(总字符数) | O(模式串长度) | O(总字符数) | 前缀匹配、词频统计 |
| AC自动机 | O(总字符数) | O(n+z) | O(总字符数) | 多模式匹配 |
| 后缀数组 | O(n log n) | O(m log n) | O(n) | 子串查询、LCP |
| Manacher | O(n) | - | O(n) | 回文子串 |

### 适用场景分析

::: code-group
```cpp
// 场景选择指南
/*
 * 1. 单模式匹配
 *    - 模式串固定：KMP
 *    - 需要Z函数特性：Z函数
 * 
 * 2. 多模式匹配
 *    - 模式串集合固定：AC自动机
 *    - 需要前缀匹配：字典树
 * 
 * 3. 子串查询
 *    - 多次查询不同模式串：后缀数组
 *    - 查询固定模式串：KMP
 * 
 * 4. 回文问题
 *    - 最长回文子串：Manacher
 *    - 所有回文子串：Manacher + 枚举中心
 * 
 * 5. 字符串比较
 *    - 多次比较子串：后缀数组 + RMQ
 *    - 单次比较：直接比较
 */
```

```python
# 场景选择指南
"""
1. 单模式匹配
   - 模式串固定：KMP
   - 需要Z函数特性：Z函数

2. 多模式匹配
   - 模式串集合固定：AC自动机
   - 需要前缀匹配：字典树

3. 子串查询
   - 多次查询不同模式串：后缀数组
   - 查询固定模式串：KMP

4. 回文问题
   - 最长回文子串：Manacher
   - 所有回文子串：Manacher + 枚举中心

5. 字符串比较
   - 多次比较子串：后缀数组 + RMQ
   - 单次比较：直接比较
"""
```
:::

## 典型题目

### 单词搜索 II

**题目描述**：给定一个二维网格和一个单词列表，找出所有同时存在于网格和列表中的单词。单词必须按照字母顺序，通过相邻的单元格内的字母构成。

💡 **思路**：使用字典树存储单词列表，然后在网格中进行DFS搜索。

::: code-group
```cpp
// C++ 单词搜索 II
#include <vector>
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
private:
    struct TrieNode {
        TrieNode* children[26];
        string word;  // 如果是单词结尾，存储完整单词
        
        TrieNode() : word("") {
            for (int i = 0; i < 26; i++) {
                children[i] = nullptr;
            }
        }
    };
    
    TrieNode* root;
    vector<string> result;
    int rows, cols;
    
    // 构建字典树
    void buildTrie(const vector<string>& words) {
        root = new TrieNode();
        for (const string& word : words) {
            TrieNode* node = root;
            for (char c : word) {
                int idx = c - 'a';
                if (!node->children[idx]) {
                    node->children[idx] = new TrieNode();
                }
                node = node->children[idx];
            }
            node->word = word;  // 标记单词结尾
        }
    }
    
    // DFS搜索
    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node) {
        char c = board[i][j];
        
        // 边界检查
        if (c == '#' || !node->children[c - 'a']) {
            return;
        }
        
        node = node->children[c - 'a'];
        
        // 找到单词
        if (!node->word.empty()) {
            result.push_back(node->word);
            node->word = "";  // 避免重复添加
        }
        
        // 标记已访问
        board[i][j] = '#';
        
        // 四个方向搜索
        int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (auto& d : dirs) {
            int ni = i + d[0], nj = j + d[1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                dfs(board, ni, nj, node);
            }
        }
        
        // 恢复
        board[i][j] = c;
    }
    
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        if (board.empty() || words.empty()) return {};
        
        rows = board.size();
        cols = board[0].size();
        
        buildTrie(words);
        
        // 从每个位置开始DFS
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dfs(board, i, j, root);
            }
        }
        
        return result;
    }
};
```

```python
# Python 单词搜索 II
from typing import List

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words:
            return []
        
        # 构建字典树
        root = {}
        for word in words:
            node = root
            for c in word:
                node = node.setdefault(c, {})
            node['#'] = word  # 标记单词结尾
        
        result = []
        rows, cols = len(board), len(board[0])
        
        def dfs(i, j, node):
            c = board[i][j]
            
            # 边界检查
            if c == '#' or c not in node:
                return
            
            curr_node = node[c]
            
            # 找到单词
            if '#' in curr_node:
                result.append(curr_node['#'])
                del curr_node['#']  # 避免重复添加
            
            # 标记已访问
            board[i][j] = '#'
            
            # 四个方向搜索
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    dfs(ni, nj, curr_node)
            
            # 恢复
            board[i][j] = c
        
        # 从每个位置开始DFS
        for i in range(rows):
            for j in range(cols):
                dfs(i, j, root)
        
        return result
```
:::

### 多模式匹配

**题目描述**：给定一个文本串和多个模式串，找出所有模式串在文本中的所有出现位置。

::: code-group
```cpp
// C++ 多模式匹配
#include <string>
#include <vector>
#include <map>
using namespace std;

vector<vector<int>> multiPatternMatch(const string& text, const vector<string>& patterns) {
    ACAutomaton ac;
    
    // 添加所有模式串
    for (int i = 0; i < patterns.size(); i++) {
        ac.addPattern(patterns[i], i);
    }
    
    // 构建AC自动机
    ac.build();
    
    // 搜索
    auto matches = ac.search(text);
    
    // 按模式串分组
    vector<vector<int>> result(patterns.size());
    for (auto& [pos, id] : matches) {
        result[id].push_back(pos);
    }
    
    return result;
}

// 使用示例
int main() {
    string text = "ababababab";
    vector<string> patterns = {"ab", "aba", "bab"};
    
    auto result = multiPatternMatch(text, patterns);
    
    for (int i = 0; i < patterns.size(); i++) {
        cout << "模式串 \"" << patterns[i] << "\" 出现位置: ";
        for (int pos : result[i]) {
            cout << pos << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

```python
# Python 多模式匹配
def multi_pattern_match(text: str, patterns: list) -> dict:
    """返回每个模式串的所有出现位置"""
    ac = ACAutomaton()
    
    # 添加所有模式串
    for i, pattern in enumerate(patterns):
        ac.add_pattern(pattern, i)
    
    # 构建AC自动机
    ac.build()
    
    # 搜索
    matches = ac.search(text)
    
    # 按模式串分组
    result = {pattern: [] for pattern in patterns}
    for pos, pattern_id in matches:
        result[patterns[pattern_id]].append(pos)
    
    return result


# 使用示例
text = "ababababab"
patterns = ["ab", "aba", "bab"]

result = multi_pattern_match(text, patterns)
for pattern, positions in result.items():
    print(f"模式串 \"{pattern}\" 出现位置: {positions}")
```
:::

### 最长重复子串

**题目描述**：给定一个字符串，找出其中最长的重复子串（至少出现两次）。

💡 **思路**：使用后缀数组，最长重复子串一定是某个相邻后缀的最长公共前缀，即 Height 数组的最大值。

::: code-group
```cpp
// C++ 最长重复子串
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

string longestRepeatingSubstring(const string& s) {
    int n = s.length();
    if (n <= 1) return "";
    
    // 构建后缀数组
    SuffixArray sa(s);
    auto SA = sa.getSA();
    auto Height = sa.getHeight();
    
    // 找Height数组的最大值
    int maxLen = 0, maxIdx = -1;
    for (int i = 1; i < n; i++) {
        if (Height[i] > maxLen) {
            maxLen = Height[i];
            maxIdx = i;
        }
    }
    
    if (maxLen == 0) return "";
    
    // 返回最长重复子串
    return s.substr(SA[maxIdx], maxLen);
}

// 找所有最长重复子串
vector<string> allLongestRepeatingSubstrings(const string& s) {
    int n = s.length();
    if (n <= 1) return {};
    
    SuffixArray sa(s);
    auto SA = sa.getSA();
    auto Height = sa.getHeight();
    
    // 找最大值
    int maxLen = *max_element(Height.begin(), Height.end());
    if (maxLen == 0) return {};
    
    // 收集所有长度为maxLen的重复子串
    vector<string> result;
    unordered_set<string> seen;
    
    for (int i = 1; i < n; i++) {
        if (Height[i] == maxLen) {
            string sub = s.substr(SA[i], maxLen);
            if (seen.find(sub) == seen.end()) {
                result.push_back(sub);
                seen.insert(sub);
            }
        }
    }
    
    return result;
}

// 使用示例
int main() {
    string s = "banana";
    
    cout << "最长重复子串: " << longestRepeatingSubstring(s) << endl;
    // 输出: "ana" (或 "na")
    
    auto all = allLongestRepeatingSubstrings(s);
    cout << "所有最长重复子串: ";
    for (const string& sub : all) {
        cout << sub << " ";
    }
    cout << endl;
    
    return 0;
}
```

```python
# Python 最长重复子串
def longest_repeating_substring(s: str) -> str:
    """找最长重复子串"""
    n = len(s)
    if n <= 1:
        return ""
    
    # 构建后缀数组
    sa = SuffixArray(s)
    SA = sa.get_sa()
    Height = sa.get_height()
    
    # 找Height数组的最大值
    max_len = max(Height)
    if max_len == 0:
        return ""
    
    # 返回最长重复子串
    max_idx = Height.index(max_len)
    return s[SA[max_idx]:SA[max_idx] + max_len]


def all_longest_repeating_substrings(s: str) -> list:
    """找所有最长重复子串"""
    n = len(s)
    if n <= 1:
        return []
    
    sa = SuffixArray(s)
    SA = sa.get_sa()
    Height = sa.get_height()
    
    max_len = max(Height)
    if max_len == 0:
        return []
    
    result = []
    seen = set()
    
    for i in range(1, n):
        if Height[i] == max_len:
            sub = s[SA[i]:SA[i] + max_len]
            if sub not in seen:
                result.append(sub)
                seen.add(sub)
    
    return result


# 使用示例
s = "banana"
print(f"最长重复子串: {longest_repeating_substring(s)}")
# 输出: "ana"

print(f"所有最长重复子串: {all_longest_repeating_substrings(s)}")
```
:::

## 总结

📌 **算法选择要点**
1. **单模式匹配**：优先考虑KMP，简单场景可用朴素算法
2. **多模式匹配**：AC自动机是首选，预处理后查询高效
3. **前缀相关问题**：字典树最适合
4. **子串查询**：后缀数组功能强大，支持多种操作
5. **回文问题**：Manacher算法是最优解

💡 **学习建议**
- 理解每种算法的核心思想和适用场景
- 多练习典型题目，加深理解
- 注意算法之间的联系（如KMP和AC自动机）
- 关注代码实现的细节（如边界条件处理）
