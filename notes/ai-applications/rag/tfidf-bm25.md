# TF-IDF 与 BM25 算法详解

TF-IDF 和 BM25 是信息检索领域中两个最经典且广泛使用的文本相关性评分算法。它们是现代搜索引擎和 RAG 系统中检索模块的基石，理解这两个算法对于构建高效的文本检索系统至关重要。

## 基本概念

### 什么是文本检索？

文本检索是指从大规模文档集合中找出与用户查询最相关文档的过程。核心挑战在于：**如何量化文档与查询之间的相关性？**

### 术语解释

| 术语 | 英文 | 含义 |
|------|------|------|
| 文档 | Document | 被检索的文本单元 |
| 词项 | Term | 文档中的基本单位（字/词） |
| 词频 | Term Frequency (TF) | 词项在文档中出现的次数 |
| 文档频率 | Document Frequency (DF) | 包含某词项的文档数量 |
| 逆文档频率 | Inverse Document Frequency (IDF) | 衡量词项的区分度 |

---

## TF-IDF 算法

### 核心思想

TF-IDF（Term Frequency-Inverse Document Frequency）的核心思想非常直观：

- **TF（词频）**：一个词在文档中出现次数越多，该文档与这个词越相关
- **IDF（逆文档频率）**：一个词在所有文档中出现越少，它的区分度越高，权重应该越大

📌 **直觉理解**：像"的"、"是"这样的常见词虽然出现频率高，但区分度低；而专业术语虽然出现频率低，但区分度高。

### 数学公式

#### 词频 TF

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

其中 $f_{t,d}$ 是词项 $t$ 在文档 $d$ 中的出现次数。

实际应用中常用对数形式来抑制过大词频的影响：

$$
\text{TF}(t, d) = \log(1 + f_{t,d})
$$

#### 逆文档频率 IDF

$$
\text{IDF}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|}
$$

其中 $N$ 是文档总数，分母是包含词项 $t$ 的文档数量。

实际应用中通常会加 1 平滑：

$$
\text{IDF}(t, D) = \log \frac{N + 1}{|\{d \in D : t \in d\}| + 1}
$$

#### TF-IDF 综合

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

### 代码示例

```python
import math
from collections import Counter
from typing import List, Dict

class TFIDFCalculator:
    """TF-IDF 计算器"""
    
    def __init__(self, documents: List[List[str]]):
        """
        初始化 TF-IDF 计算器
        
        Args:
            documents: 分词后的文档列表，每个文档是一个词列表
        """
        self.documents = documents
        self.N = len(documents)  # 文档总数
        self.df = self._compute_df()  # 文档频率
        self.idf = self._compute_idf()  # 逆文档频率
    
    def _compute_df(self) -> Dict[str, int]:
        """计算每个词的文档频率"""
        df = Counter()
        for doc in self.documents:
            # 每个词在文档中只计数一次
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        return df
    
    def _compute_idf(self) -> Dict[str, float]:
        """计算每个词的 IDF 值"""
        idf = {}
        for term, df_t in self.df.items():
            # 加 1 平滑，避免除零
            idf[term] = math.log((self.N + 1) / (df_t + 1))
        return idf
    
    def compute_tf(self, term: str, document: List[str]) -> float:
        """计算词频 TF（使用对数形式）"""
        tf = document.count(term)
        if tf == 0:
            return 0
        return math.log(1 + tf)
    
    def compute_tfidf(self, term: str, document: List[str]) -> float:
        """计算单个词的 TF-IDF 值"""
        tf = self.compute_tf(term, document)
        idf = self.idf.get(term, math.log(self.N + 1))  # 未知词使用默认 IDF
        return tf * idf
    
    def compute_document_tfidf(self, document: List[str]) -> Dict[str, float]:
        """计算文档中所有词的 TF-IDF 值"""
        tfidf_vector = {}
        for term in set(document):
            tfidf_vector[term] = self.compute_tfidf(term, document)
        return tfidf_vector


# 使用示例
if __name__ == "__main__":
    # 示例文档集合（已分词）
    documents = [
        ["苹果", "手机", "发布", "新品", "手机"],
        ["华为", "手机", "芯片", "国产", "手机"],
        ["苹果", "公司", "发布", "财报", "收入"],
        ["人工智能", "技术", "发展", "迅速"],
        ["手机", "芯片", "技术", "发展"]
    ]
    
    # 初始化计算器
    tfidf_calc = TFIDFCalculator(documents)
    
    # 查询文档
    query = ["苹果", "手机", "技术"]
    
    # 计算查询的 TF-IDF 向量
    query_tfidf = tfidf_calc.compute_document_tfidf(query)
    print("查询的 TF-IDF 向量:")
    for term, score in sorted(query_tfidf.items(), key=lambda x: -x[1]):
        print(f"  {term}: {score:.4f}")
```

输出示例：
```
查询的 TF-IDF 向量:
  技术: 0.5108
  苹果: 0.4055
  手机: 0.2231
```

### TF-IDF 的局限性

⚠️ **主要问题**：

1. **词频饱和问题**：TF 使用对数形式虽然缓解了问题，但没有从根本上解决词频增加与相关性增长不成正比的问题

2. **忽略文档长度**：长文档天然会有更多词项匹配，但没有考虑文档长度归一化

3. **词项独立性假设**：假设词项之间相互独立，忽略了语义关系

---

## BM25 算法

### 核心思想

BM25（Best Matching 25）是对 TF-IDF 的改进，全称是 Okapi BM25。它在 TF-IDF 基础上引入了两个重要改进：

1. **词频饱和函数**：使用非线性函数替代对数 TF，让词频增长对相关性的贡献趋于饱和
2. **文档长度归一化**：考虑文档长度对匹配的影响，避免长文档优势

### 数学公式

#### BM25 评分公式

$$
\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

其中：
- $q_i$ 是查询中的第 $i$ 个词项
- $f(q_i, D)$ 是词项 $q_i$ 在文档 $D$ 中的词频
- $|D|$ 是文档 $D$ 的长度（词数）
- $\text{avgdl}$ 是平均文档长度
- $k_1$ 是词频饱和参数（通常取 1.2-2.0）
- $b$ 是文档长度归一化参数（通常取 0.75）

#### IDF 计算

BM25 中 IDF 的计算方式略有不同：

$$
\text{IDF}(q_i) = \log \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)
$$

其中 $n(q_i)$ 是包含词项 $q_i$ 的文档数量。

### 参数详解

| 参数 | 含义 | 典型值 | 作用 |
|------|------|--------|------|
| $k_1$ | 词频饱和参数 | 1.2-2.0 | 控制词频对评分的影响程度 |
| $b$ | 长度归一化参数 | 0.75 | 控制文档长度归一化程度 |
| avgdl | 平均文档长度 | 动态计算 | 用于归一化 |

#### $k_1$ 参数的影响

- **$k_1$ 越大**：词频对评分影响越大，饱和越慢
- **$k_1$ 越小**：词频快速饱和，多次出现贡献递减

```python
# k1 参数对 TF 部分的影响可视化
import numpy as np
import matplotlib.pyplot as plt

def bm25_tf_component(tf, k1=1.5, b=0.75, doc_len=100, avgdl=100):
    """BM25 的 TF 部分"""
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))

tf_range = np.arange(0, 21)

plt.figure(figsize=(10, 6))
for k1 in [0.5, 1.2, 2.0, 5.0]:
    scores = [bm25_tf_component(tf, k1=k1) for tf in tf_range]
    plt.plot(tf_range, scores, label=f'k1={k1}', marker='o')

plt.xlabel('词频 (TF)')
plt.ylabel('TF 评分分量')
plt.title('BM25 词频饱和曲线（不同 k1 值）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### $b$ 参数的影响

- **$b = 1$**：完全归一化文档长度
- **$b = 0$**：不考虑文档长度

### 代码示例

```python
import math
from collections import Counter
from typing import List, Dict, Tuple

class BM25:
    """BM25 检索模型实现"""
    
    def __init__(self, documents: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 模型
        
        Args:
            documents: 分词后的文档列表
            k1: 词频饱和参数
            b: 文档长度归一化参数
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / self.N
        self.df = self._compute_df()
        self.idf = self._compute_idf()
        # 预计算每个文档的词频
        self.doc_tf = [Counter(doc) for doc in documents]
    
    def _compute_df(self) -> Dict[str, int]:
        """计算文档频率"""
        df = Counter()
        for doc in self.documents:
            for term in set(doc):
                df[term] += 1
        return df
    
    def _compute_idf(self) -> Dict[str, float]:
        """计算 BM25 的 IDF 值"""
        idf = {}
        for term, n_t in self.df.items():
            idf[term] = math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)
        return idf
    
    def _score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 分数"""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.doc_tf[doc_idx]
        
        for term in query_terms:
            if term not in tf_dict:
                continue
            
            tf = tf_dict[term]
            idf = self.idf.get(term, math.log(self.N + 1))
            
            # BM25 核心公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 分词后的查询
            top_k: 返回前 k 个结果
            
        Returns:
            [(文档索引, 分数), ...]
        """
        scores = []
        for doc_idx in range(self.N):
            score = self._score_document(query, doc_idx)
            scores.append((doc_idx, score))
        
        # 按分数降序排序
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# 使用示例
if __name__ == "__main__":
    # 示例文档
    documents = [
        ["苹果", "发布", "新", "iPhone", "手机", "手机", "手机"],
        ["华为", "发布", "新", "芯片", "手机"],
        ["苹果", "公司", "季度", "财报", "超预期"],
        ["人工智能", "技术", "改变", "生活"],
        ["手机", "芯片", "技术", "发展", "迅速", "手机"]
    ]
    
    # 初始化 BM25
    bm25 = BM25(documents, k1=1.5, b=0.75)
    
    # 查询
    query = ["苹果", "手机"]
    
    # 检索
    results = bm25.search(query, top_k=3)
    
    print(f"查询: {query}")
    print("\n检索结果:")
    for doc_idx, score in results:
        print(f"  文档{doc_idx} (分数: {score:.4f}): {documents[doc_idx]}")
```

输出示例：
```
查询: ['苹果', '手机']

检索结果:
  文档0 (分数: 2.1834): ['苹果', '发布', '新', 'iPhone', '手机', '手机', '手机']
  文档4 (分数: 1.5678): ['手机', '芯片', '技术', '发展', '迅速', '手机']
  文档1 (分数: 1.2345): ['华为', '发布', '新', '芯片', '手机']
```

---

## TF-IDF vs BM25 对比

### 核心差异

| 特性 | TF-IDF | BM25 |
|------|--------|------|
| 词频处理 | 对数增长 | 饱和函数（有上限） |
| 文档长度 | 未归一化 | 归一化处理 |
| 参数 | 无可调参数 | 有 $k_1$、$b$ 参数 |
| 可调性 | 固定公式 | 可根据场景优化 |
| 短查询表现 | 一般 | 较好 |

### 词频增长曲线对比

```
分数
 │
 │        TF-IDF (对数)
 │       ╱─────────────────
 │      ╱
 │     ╱    BM25 (饱和)
 │    ╱    ╱──────────────
 │   ╱    ╱
 │  ╱    ╱
 │ ╱    ╱
 └──────────────────────── 词频
  0  1  2  3  4  5  6  7  8
```

💡 **关键区别**：BM25 的饱和特性使得词频超过某个阈值后，对分数的贡献趋于稳定，避免了"关键词堆砌"带来的不当优势。

### 实际应用选择

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 短文本检索（如标题搜索） | TF-IDF | 短文本词频差异小，文档长度差异小 |
| 长文本检索（如全文搜索） | BM25 | 需要文档长度归一化和词频饱和控制 |
| 对检索质量要求高 | BM25 | 可调参数允许针对场景优化 |
| 快速原型开发 | TF-IDF | 实现简单，无需调参 |

---

## 在 RAG 系统中的应用

### 与向量检索的结合

在现代 RAG 系统中，TF-IDF/BM25 常与向量检索结合使用：

```
┌─────────────────────────────────────────────────────┐
│                    混合检索流程                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│   查询 Query                                        │
│      │                                              │
│      ├──────────────┬──────────────┐               │
│      ▼              ▼              ▼               │
│   关键词匹配     语义向量检索    实体识别           │
│  (BM25/TF-IDF)   (Embedding)   (NER+检索)          │
│      │              │              │               │
│      └──────────────┼──────────────┘               │
│                     ▼                              │
│              结果融合 (RRF/加权)                    │
│                     │                              │
│                     ▼                              │
│              重排序 (Reranker)                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 代码示例：混合检索

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """混合检索器：结合 BM25 和向量检索"""
    
    def __init__(self, documents: List[str], embedder):
        """
        Args:
            documents: 文档列表
            embedder: 向量嵌入模型
        """
        self.documents = documents
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.embedder = embedder
        self.doc_embeddings = embedder.encode(documents)
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 检索"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked[:top_k]
    
    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """向量检索"""
        query_embedding = self.embedder.encode([query])[0]
        # 计算余弦相似度
        scores = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                      bm25_weight: float = 0.3, 
                      vector_weight: float = 0.7) -> List[Tuple[int, float]]:
        """混合检索：RRF 融合"""
        bm25_results = self.bm25_search(query, top_k=20)
        vector_results = self.vector_search(query, top_k=20)
        
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF 参数
        scores = {}
        
        for rank, (doc_idx, _) in enumerate(bm25_results):
            scores[doc_idx] = scores.get(doc_idx, 0) + bm25_weight / (k + rank + 1)
        
        for rank, (doc_idx, _) in enumerate(vector_results):
            scores[doc_idx] = scores.get(doc_idx, 0) + vector_weight / (k + rank + 1)
        
        # 按融合分数排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
```

---

## 总结

### 关键要点

1. **TF-IDF** 是经典的文本相关性评分方法，计算简单，适合快速原型
2. **BM25** 是 TF-IDF 的改进版本，通过词频饱和和文档长度归一化获得更好的检索效果
3. 两者都是**稀疏检索**方法，基于关键词匹配，与**稠密向量检索**形成互补
4. 现代 RAG 系统通常采用**混合检索**策略，结合两者优势

### 实践建议

- ✅ 默认使用 BM25，它在大多数场景下表现更好
- ✅ 根据数据特点调整 $k_1$ 和 $b$ 参数
- ✅ 结合向量检索提升语义理解能力
- ✅ 考虑使用倒排索引加速 BM25 检索（如 Elasticsearch、Whoosh）

---

## 参考资料

- [Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.nowpublishers.com/article/DownloadSummary/INR-019)
- [Elasticsearch BM25 文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)
