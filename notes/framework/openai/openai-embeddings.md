# Embeddings API

Embeddings API 用于将文本转换为向量表示，是语义搜索、相似度计算、推荐系统等应用的基础。本章介绍文本嵌入的原理、API 使用方法和实际应用。

---

## 📐 什么是文本嵌入

### 基本概念

文本嵌入（Text Embeddings）是将文本转换为高维向量（数组）的过程。这些向量捕捉了文本的语义信息，使得语义相似的文本在向量空间中距离更近。

```
文本                     嵌入向量
─────────────────────────────────────────
"猫是一种宠物"    →    [0.12, -0.45, 0.78, ...]
"狗是常见的宠物"  →    [0.15, -0.42, 0.75, ...]  (相似)
"今天天气不错"    →    [-0.32, 0.56, -0.21, ...]  (不同)
```

### 应用场景

| 场景 | 说明 |
|------|------|
| **语义搜索** | 根据含义而非关键词匹配搜索内容 |
| **相似度计算** | 计算文本之间的语义相似度 |
| **聚类分析** | 将相似文本分组 |
| **推荐系统** | 推荐相似内容 |
| **异常检测** | 识别与众不同的内容 |
| **RAG** | 检索增强生成的基础 |

---

## 🔧 API 基本使用

### 同步调用

```python
from openai import OpenAI

client = OpenAI()

# 单个文本嵌入
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="你好，世界"
)

# 获取嵌入向量
embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")  # 1536
print(f"前5个值: {embedding[:5]}")
```

### 批量处理

```python
from openai import OpenAI

client = OpenAI()

# 批量嵌入
texts = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络",
    "自然语言处理处理文本数据"
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# 获取所有嵌入向量
embeddings = [item.embedding for item in response.data]
print(f"处理了 {len(embeddings)} 个文本")
print(f"每个向量维度: {len(embeddings[0])}")
```

### 异步调用

```python
import asyncio
from openai import AsyncOpenAI

async def get_embeddings(texts):
    client = AsyncOpenAI()
    
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    return [item.embedding for item in response.data]

# 使用
texts = ["文本1", "文本2", "文本3"]
embeddings = asyncio.run(get_embeddings(texts))
```

---

## 📊 可用模型

### 模型对比

| 模型 | 维度 | 最大输入 | 价格 ($/1M tokens) | 特点 |
|------|------|----------|-------------------|------|
| `text-embedding-3-small` | 512/1536 | 8191 tokens | $0.02 | 性价比高 |
| `text-embedding-3-large` | 256/1024/3072 | 8191 tokens | $0.13 | 精度最高 |
| `text-embedding-ada-002` | 1536 | 8191 tokens | $0.10 | 旧版，不推荐 |

### 维度缩减

新模型支持指定输出维度：

```python
from openai import OpenAI

client = OpenAI()

# 使用较小维度降低存储成本
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="测试文本",
    dimensions=512  # 缩减到 512 维
)

print(f"向量维度: {len(response.data[0].embedding)}")  # 512
```

### 模型选择建议

| 场景 | 推荐模型 | 维度 | 说明 |
|------|----------|------|------|
| 大规模搜索 | text-embedding-3-small | 512 | 成本优先 |
| 语义相似度 | text-embedding-3-small | 1536 | 平衡选择 |
| 高精度需求 | text-embedding-3-large | 3072 | 精度优先 |
| RAG 应用 | text-embedding-3-small | 1536 | 推荐配置 |

---

## 📏 相似度计算

### 余弦相似度

最常用的相似度度量方法：

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

def get_embedding(text):
    """获取文本嵌入"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 示例
texts = [
    "猫是一种常见的宠物",
    "狗是人类的忠实伙伴",
    "今天天气很好",
    "机器学习是AI的分支"
]

# 获取嵌入
embeddings = [get_embedding(text) for text in texts]

# 计算相似度矩阵
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"'{text1[:10]}...' vs '{text2[:10]}...': {sim:.4f}")
```

### 相似度搜索

```python
import numpy as np
from openai import OpenAI
from typing import List, Tuple

client = OpenAI()

class SemanticSearch:
    """语义搜索引擎"""
    
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.documents = []
        self.embeddings = []
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入"""
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def index(self, documents: List[str]):
        """索引文档"""
        self.documents = documents
        self.embeddings = self._get_embeddings(documents)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """搜索相似文档"""
        # 获取查询嵌入
        query_embedding = self._get_embeddings([query])[0]
        
        # 计算所有文档的相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], sim))
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1, vec2):
        """余弦相似度"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 使用
search_engine = SemanticSearch()

documents = [
    "Python 是一种流行的编程语言",
    "机器学习是人工智能的核心技术",
    "深度学习使用多层神经网络",
    "自然语言处理用于理解和生成文本",
    "计算机视觉让机器能够看懂图像"
]

search_engine.index(documents)

results = search_engine.search("什么是AI技术？", top_k=3)
for doc, score in results:
    print(f"相似度: {score:.4f} - {doc}")
```

---

## 💾 向量存储

### 简单内存存储

```python
import json
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional

client = OpenAI()

class VectorStore:
    """简单的向量存储"""
    
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.vectors: List[Dict] = []
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """添加文本"""
        # 获取嵌入
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        for i, item in enumerate(response.data):
            self.vectors.append({
                "id": len(self.vectors),
                "text": texts[i],
                "embedding": item.embedding,
                "metadata": metadatas[i] if metadatas else {}
            })
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似文本"""
        # 获取查询嵌入
        response = client.embeddings.create(
            model=self.model,
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # 计算相似度
        results = []
        for item in self.vectors:
            embedding = np.array(item["embedding"])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            results.append({
                **item,
                "score": float(similarity)
            })
        
        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def save(self, filepath: str):
        """保存到文件"""
        # 转换 numpy 数组为列表
        data = []
        for item in self.vectors:
            data.append({
                "id": item["id"],
                "text": item["text"],
                "embedding": item["embedding"],
                "metadata": item["metadata"]
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vectors = data

# 使用
store = VectorStore()

# 添加文档
documents = [
    "Python 是解释型语言",
    "Java 是编译型语言",
    "JavaScript 用于前端开发"
]

store.add_texts(documents)

# 搜索
results = store.search("编程语言")
for r in results:
    print(f"分数: {r['score']:.4f} - {r['text']}")
```

### 集成 FAISS

```python
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Tuple

client = OpenAI()

class FAISSVectorStore:
    """使用 FAISS 的向量存储"""
    
    def __init__(self, dimension: int = 1536, model: str = "text-embedding-3-small"):
        self.dimension = dimension
        self.model = model
        self.index = faiss.IndexFlatIP(dimension)  # 内积（余弦相似度）
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量（用于余弦相似度）"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """添加文本"""
        # 获取嵌入
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
        embeddings = self._normalize(embeddings)
        
        # 添加到索引
        self.index.add(embeddings)
        
        # 存储文档
        self.documents.extend(texts)
        self.metadatas.extend(metadatas or [{}] * len(texts))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """搜索"""
        # 获取查询嵌入
        response = client.embeddings.create(
            model=self.model,
            input=query
        )
        
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        query_embedding = self._normalize(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # 有效索引
                results.append((
                    self.documents[idx],
                    float(scores[0][i]),
                    self.metadatas[idx]
                ))
        
        return results

# 使用
store = FAISSVectorStore()

documents = [
    "机器学习是从数据中学习",
    "深度学习使用神经网络",
    "自然语言处理处理文本"
]

store.add_texts(documents)

results = store.search("什么是AI？")
for text, score, meta in results:
    print(f"分数: {score:.4f} - {text}")
```

---

## 🎯 实践示例

### 示例1：文档问答系统

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

class DocumentQA:
    """基于嵌入的文档问答系统"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def _get_embeddings(self, texts):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def add_document(self, content: str, chunk_size: int = 500):
        """添加文档，自动分块"""
        # 简单分块
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])
        
        self.documents.extend(chunks)
        self.embeddings.extend(self._get_embeddings(chunks))
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """获取相关上下文"""
        query_emb = self._get_embeddings([query])[0]
        
        # 计算相似度
        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            score = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            scores.append((score, self.documents[i]))
        
        # 获取 top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        context = "\n\n".join([doc for _, doc in scores[:top_k]])
        
        return context
    
    def ask(self, question: str) -> str:
        """回答问题"""
        context = self.get_context(question)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "根据以下上下文回答问题。如果上下文没有相关信息，请说不知道。"
                },
                {
                    "role": "user",
                    "content": f"上下文：\n{context}\n\n问题：{question}"
                }
            ]
        )
        
        return response.choices[0].message.content

# 使用
qa = DocumentQA()

# 添加文档
qa.add_document("""
Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
Python 以其清晰的语法和代码可读性而闻名。
它支持多种编程范式，包括面向对象、命令式和函数式编程。
Python 广泛应用于 Web 开发、数据科学、人工智能等领域。
""")

# 提问
print(qa.ask("Python 是什么时候创建的？"))
print(qa.ask("Python 有什么特点？"))
```

### 示例2：文本聚类

```python
from openai import OpenAI
from sklearn.cluster import KMeans
import numpy as np

client = OpenAI()

def cluster_texts(texts: list, n_clusters: int = 3):
    """文本聚类"""
    # 获取嵌入
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    embeddings = np.array([item.embedding for item in response.data])
    
    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # 组织结果
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(texts[i])
    
    return clusters

# 使用
texts = [
    "Python 是一种编程语言",
    "JavaScript 用于网页开发",
    "机器学习是 AI 的核心技术",
    "深度学习使用神经网络",
    "Java 是面向对象语言",
    "自然语言处理处理文本数据",
    "C++ 性能很高",
    "计算机视觉处理图像"
]

clusters = cluster_texts(texts, n_clusters=3)

for cluster_id, items in clusters.items():
    print(f"\n聚类 {cluster_id}:")
    for item in items:
        print(f"  - {item}")
```

---

## 📋 API 参数速查

### embeddings.create()

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | str | 嵌入模型名称 |
| `input` | str/list | 输入文本或文本列表 |
| `dimensions` | int | 输出维度（可选） |
| `encoding_format` | str | 编码格式：`float` 或 `base64` |
| `user` | str | 用户标识 |

### 响应结构

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="测试"
)

# 响应属性
response.object     # "list"
response.data       # 嵌入数据列表
response.model      # 使用的模型
response.usage      # Token 使用情况

# 单个嵌入
response.data[0].object      # "embedding"
response.data[0].index       # 索引
response.data[0].embedding   # 嵌入向量
```

---

## 🔗 相关章节

- [Chat Completions](./openai-chat.md) - 对话 API
- [Function Calling](./openai-functions.md) - 工具调用
- [LangChain](../langchain/index.md) - RAG 应用开发
