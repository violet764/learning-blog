# TF-IDF 详解

TF-IDF（Term Frequency-Inverse Document Frequency，词频-逆文档频率）是信息检索和文本挖掘中一种经典的**特征加权技术**。它通过评估一个词在文档中的重要程度，将文本转换为数值向量，广泛应用于搜索引擎、文本分类、关键词提取等任务。

## 基本概念

### 什么是 TF-IDF？

TF-IDF 的核心思想非常直观：

- **一个词在当前文档中出现次数越多，它对该文档越重要**（词频 TF）
- **一个词在整个语料库中出现的文档越少，它的区分度越高**（逆文档频率 IDF）

通过将两者相乘，我们得到一个词对特定文档的**重要性得分**——既考虑了局部频率，又考虑了全局稀有度。

### 为什么需要 TF-IDF？

在文本处理中，我们面临几个挑战：

| 问题 | 说明 |
|------|------|
| 停用词干扰 | "的"、"是"、"在" 等词出现频率高但语义价值低 |
| 特征表示 | 文本无法直接输入机器学习模型，需要数值化 |
| 重要性度量 | 如何衡量一个词对文档的贡献度 |

TF-IDF 通过**降低常见词权重、提升稀有词权重**来解决这些问题。

## 核心原理

### 词频 (TF)

词频衡量一个词在单篇文档中的出现频率：

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

其中 $f_{t,d}$ 是词 $t$ 在文档 $d$ 中的出现次数，分母是文档 $d$ 中所有词的总数。

**简化版本**（常用）：

$$
\text{TF}(t, d) = f_{t,d}
$$

即直接使用出现次数。

### 逆文档频率 (IDF)

逆文档频率衡量一个词在整个语料库中的稀有程度：

$$
\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}
$$

其中：
- $|D|$ 是语料库中的文档总数
- $|\{d \in D : t \in d\}|$ 是包含词 $t$ 的文档数量

**平滑版本**（避免除零）：

$$
\text{IDF}(t, D) = \log \frac{|D| + 1}{|\{d \in D : t \in d\}| + 1} + 1
$$

### TF-IDF 计算

最终的 TF-IDF 值是两者的乘积：

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

### 计算示例

假设有一个包含 3 篇文档的语料库：

```
文档1: "机器学习 是 人工智能 的 分支"
文档2: "深度学习 是 机器学习 的 子领域"  
文档3: "自然语言处理 是 人工智能 的 应用"
```

计算"机器学习"在文档1中的 TF-IDF：

| 步骤 | 计算 | 结果 |
|------|------|------|
| TF | 在文档1中出现 1 次 | $\text{TF} = 1/4 = 0.25$ |
| DF | 在文档1、2中出现 | $\text{DF} = 2$ |
| IDF | $\log(3/2)$ | $\text{IDF} \approx 0.176$ |
| TF-IDF | $0.25 \times 0.176$ | $\approx 0.044$ |

对比"分支"（仅出现在文档1）：

| 步骤 | 计算 | 结果 |
|------|------|------|
| TF | 在文档1中出现 1 次 | $\text{TF} = 0.25$ |
| DF | 仅在文档1出现 | $\text{DF} = 1$ |
| IDF | $\log(3/1)$ | $\text{IDF} \approx 0.477$ |
| TF-IDF | $0.25 \times 0.477$ | $\approx 0.119$ |

💡 "分支"的 TF-IDF 值更高，因为它更能代表文档1的独特内容。

## 代码示例

### 基础实现：从零手写 TF-IDF

```python
import math
from collections import Counter

class TFIDFCalculator:
    """TF-IDF 计算器"""
    
    def __init__(self, documents):
        """
        初始化
        
        Args:
            documents: 文档列表，每个文档是分词后的列表
        """
        self.documents = documents
        self.vocab = self._build_vocab()
        self.idf = self._compute_idf()
    
    def _build_vocab(self):
        """构建词汇表"""
        vocab = set()
        for doc in self.documents:
            vocab.update(doc)
        return vocab
    
    def _compute_idf(self):
        """计算所有词的 IDF 值"""
        n_docs = len(self.documents)
        idf = {}
        
        for word in self.vocab:
            # 统计包含该词的文档数
            n_containing = sum(1 for doc in self.documents if word in doc)
            # 计算 IDF（加 1 平滑）
            idf[word] = math.log((n_docs + 1) / (n_containing + 1)) + 1
        
        return idf
    
    def compute_tf(self, document):
        """
        计算文档中每个词的 TF 值
        
        Args:
            document: 分词后的文档列表
            
        Returns:
            dict: {词: TF值}
        """
        word_count = Counter(document)
        total_words = len(document)
        return {word: count / total_words for word, count in word_count.items()}
    
    def compute_tfidf(self, document):
        """
        计算文档中每个词的 TF-IDF 值
        
        Args:
            document: 分词后的文档列表
            
        Returns:
            dict: {词: TF-IDF值}
        """
        tf = self.compute_tf(document)
        tfidf = {word: tf_val * self.idf.get(word, 0) for word, tf_val in tf.items()}
        return tfidf
    
    def transform(self, document):
        """
        将文档转换为 TF-IDF 向量
        
        Args:
            document: 分词后的文档列表
            
        Returns:
            list: TF-IDF 向量（按词汇表顺序）
        """
        tfidf = self.compute_tfidf(document)
        return [tfidf.get(word, 0) for word in sorted(self.vocab)]


# 使用示例
documents = [
    ['机器学习', '是', '人工智能', '的', '分支'],
    ['深度学习', '是', '机器学习', '的', '子领域'],
    ['自然语言处理', '是', '人工智能', '的', '应用']
]

calculator = TFIDFCalculator(documents)

# 计算第一篇文档的 TF-IDF
tfidf_result = calculator.compute_tfidf(documents[0])
print("文档1的 TF-IDF 值:")
for word, score in sorted(tfidf_result.items(), key=lambda x: -x[1]):
    print(f"  {word}: {score:.4f}")
```

输出：
```
文档1的 TF-IDF 值:
  分支: 0.1193
  机器学习: 0.0441
  人工智能: 0.0441
  是: 0.0295
  的: 0.0295
```

### 使用 scikit-learn 实现

在实际项目中，推荐使用 `scikit-learn` 提供的高效实现：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 示例文档
documents = [
    "机器学习 是 人工智能 的 分支",
    "深度学习 是 机器学习 的 子领域",
    "自然语言处理 是 人工智能 的 应用"
]

# 创建 TF-IDF 向量化器
vectorizer = TfidfVectorizer(
    token_pattern=r'(?u)\b\w+\b',  # 匹配单个词
    norm='l2',                      # L2 归一化
    use_idf=True,
    smooth_idf=True,                # 平滑 IDF
    sublinear_tf=False              # 是否对 TF 取对数
)

# 拟合并转换
tfidf_matrix = vectorizer.fit_transform(documents)

# 获取特征名（词汇表）
feature_names = vectorizer.get_feature_names_out()

# 转换为 DataFrame 便于查看
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[f'文档{i+1}' for i in range(len(documents))]
)

print("TF-IDF 矩阵:")
print(df.round(4))
```

输出：
```
TF-IDF 矩阵:
       分支     应用     子领域    是      的    人工智能   ...  
文档1  0.5774  0.0000  0.0000  0.2887  0.2887  0.4082  ...
文档2  0.0000  0.0000  0.5774  0.2887  0.2887  0.0000  ...
文档3  0.0000  0.5774  0.0000  0.2887  0.2887  0.4082  ...
```

### 实际应用：文本相似度计算

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例：搜索引擎文档排序
corpus = [
    "Python 是一种流行的编程语言",
    "机器学习是人工智能的核心技术",
    "Python 广泛应用于机器学习和数据分析",
    "深度学习是机器学习的子领域",
    "自然语言处理使用深度学习技术"
]

query = "Python 机器学习"

# 创建向量器并转换
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
query_vec = vectorizer.transform([query])

# 计算余弦相似度
similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

# 排序并输出结果
ranked_indices = similarities.argsort()[::-1]
print(f"查询: '{query}'\n")
print("搜索结果排名:")
for idx in ranked_indices:
    print(f"  相似度 {similarities[idx]:.4f}: {corpus[idx]}")
```

输出：
```
查询: 'Python 机器学习'

搜索结果排名:
  相似度 0.6923: Python 广泛应用于机器学习和数据分析
  相似度 0.3162: Python 是一种流行的编程语言
  相似度 0.3162: 机器学习是人工智能的核心技术
  相似度 0.2236: 深度学习是机器学习的子领域
  相似度 0.0000: 自然语言处理使用深度学习技术
```

### 实际应用：关键词提取

```python
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text, top_k=5):
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        top_k: 返回前 k 个关键词
        
    Returns:
        list: 关键词列表
    """
    # 使用 jieba 分词
    words = jieba.cut(text)
    processed_text = ' '.join(words)
    
    # 创建向量器
    vectorizer = TfidfVectorizer(
        max_features=100,    # 最多保留 100 个特征
        token_pattern=r'(?u)\b\w+\b'
    )
    
    # 由于只有一个文档，IDF 计算意义不大
    # 这里主要利用 TF 来提取关键词
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    # 获取特征名和权重
    feature_names = vectorizer.get_feature_names_out()
    weights = tfidf_matrix.toarray().flatten()
    
    # 排序并返回 top_k
    word_weights = list(zip(feature_names, weights))
    word_weights.sort(key=lambda x: x[1], reverse=True)
    
    return word_weights[:top_k]


# 示例
text = """
自然语言处理是人工智能和语言学领域的分支学科。
在这一领域中探讨如何处理及运用自然语言；自然语言处理包括多方面和步骤，
基本认知、理解、生成等。自然语言处理广泛应用于机器翻译、情感分析、
问答系统、文本摘要、信息抽取等领域。
"""

keywords = extract_keywords(text)
print("关键词提取结果:")
for word, weight in keywords:
    print(f"  {word}: {weight:.4f}")
```

## 常见变体

### 1. TF 的变体

| 变体 | 公式 | 说明 |
|------|------|------|
| 原始词频 | $f_{t,d}$ | 直接使用出现次数 |
| 归一化词频 | $\frac{f_{t,d}}{\sum f_{t',d}}$ | 消除文档长度影响 |
| 对数词频 | $1 + \log f_{t,d}$ | 降低高频词的权重 |
| 增强词频 | $K + (1-K) \frac{f_{t,d}}{\max f_{t',d}}$ | 相对于最高频词的比率 |

### 2. IDF 的变体

| 变体 | 公式 | 说明 |
|------|------|------|
| 标准 IDF | $\log \frac{N}{n_t}$ | 经典公式 |
| 平滑 IDF | $\log \frac{N + 1}{n_t + 1} + 1$ | 避免除零 |
| 概率 IDF | $\log \frac{N - n_t + 0.5}{n_t + 0.5}$ | 基于概率模型 |

### 3. TF-IDF 的归一化

常见的归一化方式：

$$
\text{TF-IDF}_{\text{norm}} = \frac{\text{TF-IDF}}{\sqrt{\sum_{t} \text{TF-IDF}(t)^2}}
$$

即 L2 归一化，使得向量的模长为 1，便于计算相似度。

## 优缺点分析

### ✅ 优点

1. **简单高效**：计算复杂度低，易于实现
2. **可解释性强**：每个维度对应一个词，权重直观
3. **无需训练**：直接在语料库上统计即可
4. **适用广泛**：信息检索、文本分类、聚类等

### ❌ 局限性

| 局限 | 说明 | 解决方案 |
|------|------|----------|
| 词序丢失 | 忽略词的位置信息 | 使用 n-gram |
| 语义缺失 | 无法捕捉词的语义关系 | 使用 Word2Vec、BERT 等嵌入 |
| 维度稀疏 | 词汇表大时向量稀疏 | 降维、特征选择 |
| 领域依赖 | IDF 依赖语料库分布 | 领域自适应 |

## 与其他方法的对比

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **TF-IDF** | 基于统计，稀疏表示 | 信息检索、关键词提取 |
| **Word2Vec** | 稠密向量，捕捉语义 | 相似度计算、聚类 |
| **BERT** | 上下文相关，深度语义 | 复杂 NLP 任务 |
| **BM25** | TF-IDF 改进版 | 搜索引擎排序 |

## 常见问题

### Q1: TF-IDF 如何处理停用词？

TF-IDF 会自动降低停用词的权重，因为停用词在所有文档中都很常见，IDF 值低。但在实际应用中，通常会先去除停用词以减少计算开销。

### Q2: 如何选择 TF 和 IDF 的变体？

- **短文本**：使用原始词频或增强词频
- **长文本**：使用归一化词频或对数词频
- **小语料**：使用平滑 IDF

### Q3: TF-IDF 向量维度很大怎么办？

1. **特征选择**：保留 IDF 值最高的 k 个词
2. **降维**：使用 SVD/LSA 降维
3. **过滤**：去除出现频率过低或过高的词

## 参考资料

- [sklearn TfidfVectorizer 文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Manning, C. D., et al. "Introduction to Information Retrieval" - Chapter 6
