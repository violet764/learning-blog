# 检索器模块（Retriever）

检索器是 RAG 系统的核心组件，负责从知识库中找到与查询最相关的文档片段。一个高效的检索器直接影响 RAG 系统的回答质量。本章将深入介绍检索器的三大核心要素：向量数据库、Embedding 模型和检索策略。

---

## 🎯 检索器的作用

### 检索器的定位

```
┌─────────────────────────────────────────────────────────────────────┐
│                       RAG 系统架构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   用户查询 ──────→ ┌─────────────────────────────────┐             │
│                   │         检索器 (Retriever)       │ ← 本章重点   │
│                   │  ┌─────────┐  ┌─────────────┐   │             │
│                   │  │ 查询编码 │→ │  向量检索    │   │             │
│                   │  └─────────┘  └─────────────┘   │             │
│                   └──────────────┬──────────────────┘             │
│                                  │                                  │
│                                  ▼                                  │
│                          检索到的文档                               │
│                                  │                                  │
│                                  ▼                                  │
│                   ┌─────────────────────────────────┐             │
│                   │        生成器 (Generator)        │             │
│                   │      Prompt 组装 + LLM 生成      │             │
│                   └─────────────────────────────────┘             │
│                                  │                                  │
│                                  ▼                                  │
│                             最终回答                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 检索器的核心目标

| 目标 | 描述 | 关键指标 |
|------|------|----------|
| **高召回率** | 尽可能找回所有相关文档 | Recall@K |
| **高精确率** | 返回的文档都应相关 | Precision@K |
| **低延迟** | 快速返回结果 | 查询延迟 (ms) |
| **可扩展** | 支持大规模数据 | 支持文档数量 |

---

## 📚 向量数据库

### 什么是向量数据库

向量数据库是专门用于存储和检索向量嵌入的数据库，支持高效的相似度搜索。

```
传统数据库 vs 向量数据库

传统数据库                    向量数据库
┌────────────────┐           ┌────────────────┐
│  结构化数据     │           │   向量数据      │
│  (表格、文本)   │           │  (Embedding)   │
└───────┬────────┘           └───────┬────────┘
        │                            │
        ▼                            ▼
┌────────────────┐           ┌────────────────┐
│  精确匹配查询   │           │  相似度搜索     │
│  WHERE name='x'│           │  找最近的向量   │
└────────────────┘           └────────────────┘
        │                            │
        ▼                            ▼
┌────────────────┐           ┌────────────────┐
│   完全匹配      │           │  语义相似匹配   │
│   精确结果      │           │  按相似度排序   │
└────────────────┘           └────────────────┘
```

### 向量相似度计算

常用的向量相似度计算方法：

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 - 最常用的相似度度量"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离 - 距离越小越相似"""
    return np.linalg.norm(a - b)

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """点积 - 简单高效（向量已归一化时等价于余弦相似度）"""
    return np.dot(a, b)

# 示例
vec_a = np.array([0.5, 0.3, 0.8])
vec_b = np.array([0.6, 0.4, 0.7])

print(f"余弦相似度: {cosine_similarity(vec_a, vec_b):.4f}")  # 0.9826
print(f"欧氏距离: {euclidean_distance(vec_a, vec_b):.4f}")    # 0.1732
print(f"点积: {dot_product(vec_a, vec_b):.4f}")               # 0.9400
```

### 主流向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Milvus** | 开源/自托管 | 高性能、分布式、功能丰富 | 生产级大规模应用 |
| **Pinecone** | 云服务 | 全托管、免运维 | 快速上线、中小规模 |
| **Chroma** | 开源/轻量 | 简单易用、嵌入式 | 开发测试、小规模 |
| **FAISS** | 库 | Meta 开源、纯向量搜索 | 本地部署、离线处理 |
| **Weaviate** | 开源 | 语义搜索、GraphQL | 语义理解场景 |
| **Qdrant** | 开源 | Rust 实现、高性能 | 高并发场景 |

### 实战：Chroma 快速入门

```python
# 安装: pip install chromadb

import chromadb
from chromadb.utils import embedding_functions

# 1. 创建客户端
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建集合（相当于数据库表）
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection(
    name="knowledge_base",
    embedding_function=embedding_function,
    metadata={"description": "技术文档知识库"}
)

# 3. 添加文档
documents = [
    "Python 是一种高级编程语言，由 Guido van Rossum 创建。",
    "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
    "Transformer 架构是现代大语言模型的基础。",
    "向量数据库用于存储和检索高维向量数据。"
]

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"category": "programming", "language": "python"},
        {"category": "ai", "field": "ml"},
        {"category": "ai", "field": "nlp"},
        {"category": "database", "type": "vector"}
    ]
)

# 4. 查询
results = collection.query(
    query_texts=["什么是大语言模型的基础架构？"],
    n_results=2
)

print("查询结果:")
for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  - {doc} [{metadata}]")

# 5. 带元数据过滤的查询
results_filtered = collection.query(
    query_texts=["告诉我关于 AI 的内容"],
    n_results=2,
    where={"category": "ai"}  # 只返回 AI 类别的文档
)

print("\n过滤查询结果:")
for doc in results_filtered["documents"][0]:
    print(f"  - {doc}")
```

### 实战：Milvus 生产部署

```python
# 安装: pip install pymilvus

from pymilvus import (
    connections, 
    Collection, 
    FieldSchema, 
    CollectionSchema, 
    DataType,
    utility
)

# 1. 连接 Milvus 服务器
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 2. 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200)
]

schema = CollectionSchema(
    fields=fields,
    description="技术文档知识库"
)

# 3. 创建集合
collection_name = "tech_documents"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# 4. 创建索引（用于加速检索）
index_params = {
    "metric_type": "COSINE",      # 相似度度量
    "index_type": "IVF_FLAT",     # 索引类型
    "params": {"nlist": 128}      # 聚类中心数量
}
collection.create_index(field_name="embedding", index_params=index_params)

# 5. 插入数据
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

documents = [
    "RAG 是检索增强生成技术，用于增强大语言模型的能力。",
    "向量数据库支持高效的相似度搜索。",
    "Embedding 模型将文本转换为稠密向量表示。"
]

embeddings = model.encode(documents)

data = [
    embeddings.tolist(),
    documents,
    ["doc1", "doc2", "doc3"]
]

collection.insert(data)
collection.flush()  # 确保数据持久化

# 6. 检索
collection.load()  # 加载到内存

query = "如何增强大语言模型的能力？"
query_embedding = model.encode([query])

search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

results = collection.search(
    data=query_embedding.tolist(),
    anns_field="embedding",
    param=search_params,
    limit=3,
    output_fields=["content", "source"]
)

print("检索结果:")
for hits in results:
    for hit in hits:
        print(f"  相似度: {hit.distance:.4f}")
        print(f"  内容: {hit.entity.get('content')}")
        print(f"  来源: {hit.entity.get('source')}")
        print()
```

### 向量索引类型选择

```
索引类型          特点                          适用场景
────────────────────────────────────────────────────────────────
FLAT             暴力搜索，精度最高              小规模数据 (<10万)
                 速度最慢                       建立基线

IVF_FLAT         倒排索引，平衡精度与速度        中等规模 (10-100万)
                 需要调节 nprobe               生产环境常用

IVF_PQ           乘积量化，内存占用小           大规模数据 (>1000万)
                 精度有所损失                   内存受限场景

HNSW             图索引，速度快                 实时性要求高
                 内存占用大                     推荐系统

SCANN            Google 开源，性能优秀          高性能要求场景
                 需要调参
```

---

## 🔤 Embedding 模型

### 什么是文本嵌入

文本嵌入（Text Embedding）是将离散的文本转换为连续的稠密向量表示的过程。

```
文本嵌入过程：

原始文本                    Embedding 模型                   向量表示
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│                 │        │                 │        │ [0.12, 0.85,    │
│  "机器学习是AI  │   ──→  │  Transformer    │   ──→  │  0.33, 0.67,    │
│   的一个子领域" │        │  编码器         │        │  ..., 0.21]     │
│                 │        │                 │        │ (768维向量)      │
└─────────────────┘        └─────────────────┘        └─────────────────┘

语义相似的文本 → 相似的向量表示
```

### 主流 Embedding 模型

| 模型 | 维度 | 语言 | 特点 |
|------|------|------|------|
| **OpenAI text-embedding-3-small** | 1536 | 多语言 | 性价比高，API 调用 |
| **OpenAI text-embedding-3-large** | 3072 | 多语言 | 最高质量，API 调用 |
| **BAAI/bge-large-zh-v1.5** | 1024 | 中文 | 开源中文最佳 |
| **BAAI/bge-m3** | 1024 | 多语言 | 多语言、多功能 |
| **E5-large-v2** | 1024 | 英文 | 英文表现优秀 |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | 英文 | 轻量级、速度快 |

### 实战：使用 OpenAI Embedding

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """获取文本的向量嵌入"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

# 计算语义相似度
texts = [
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的一个分支",
    "今天天气真好"
]

embeddings = [get_embedding(text) for text in texts]

# 计算相似度矩阵
print("语义相似度矩阵:")
print(f"文本 1 vs 文本 2: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"文本 1 vs 文本 3: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
print(f"文本 2 vs 文本 3: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

# 输出示例：
# 文本 1 vs 文本 2: 0.8542  (语义相关)
# 文本 1 vs 文本 3: 0.1234  (语义无关)
# 文本 2 vs 文本 3: 0.1098  (语义无关)
```

### 实战：使用开源中文 Embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载中文 Embedding 模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

def encode_texts(texts: list, normalize: bool = True) -> np.ndarray:
    """批量编码文本"""
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings

def encode_query(query: str) -> np.ndarray:
    """编码查询（添加指令前缀以提升检索效果）"""
    # BGE 模型推荐在查询前添加指令
    instruction = "为这个句子生成表示以用于检索相关文章："
    return model.encode(instruction + query, normalize_embeddings=True)

# 示例：文档检索
documents = [
    "Python 是一种解释型、面向对象的编程语言。",
    "JavaScript 是一种主要用于 Web 开发的脚本语言。",
    "机器学习算法可以从数据中学习模式和规律。",
    "深度学习使用多层神经网络进行特征学习。",
    "今天北京天气晴朗，气温 25 度。"
]

# 编码文档和查询
doc_embeddings = encode_texts(documents)
query = "什么是深度学习？"
query_embedding = encode_query(query)

# 检索最相关的文档
similarities = [
    (i, cosine_similarity(query_embedding, doc_emb))
    for i, doc_emb in enumerate(doc_embeddings)
]
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"查询: {query}")
print("\n检索结果:")
for idx, sim in similarities[:3]:
    print(f"  相似度: {sim:.4f} - {documents[idx]}")
```

### Embedding 模型选择指南

```
选择决策树：

是否需要本地部署？
├── 是 → 是否主要是中文？
│       ├── 是 → BGE-large-zh-v1.5 或 BGE-M3
│       └── 否 → 是否需要多语言？
│               ├── 是 → BGE-M3
│               └── 否 → E5-large-v2 或 all-MiniLM-L6-v2
│
└── 否 → 可以使用 API
        ├── 追求最高质量 → text-embedding-3-large
        └── 追求性价比 → text-embedding-3-small
```

---

## 🔍 检索策略

### 检索策略分类

```
检索策略
│
├── 稠密检索 (Dense Retrieval)
│   └── 向量相似度搜索（Embedding）
│
├── 稀疏检索 (Sparse Retrieval)
│   └── 关键词匹配（BM25）
│
└── 混合检索 (Hybrid Retrieval)
    └── 稠密 + 稀疏组合
```

### 1. 稠密检索（Dense Retrieval）

基于语义向量的检索方法：

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 创建向量存储
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

# 假设已有文档向量存储
vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# 稠密检索
def dense_retrieval(query: str, k: int = 5):
    """基于语义向量的检索"""
    results = vectorstore.similarity_search(
        query,
        k=k
    )
    return results

# 带分数的检索
def dense_retrieval_with_scores(query: str, k: int = 5):
    """返回相似度分数"""
    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )
    return results  # [(Document, score), ...]
```

**优点**：
- 语义理解能力强
- 能处理同义词、近义词
- 跨语言检索（多语言模型）

**缺点**：
- 对精确关键词匹配较弱
- 专业术语可能表现不佳
- 计算开销较大

### 2. 稀疏检索（Sparse Retrieval / BM25）

基于关键词的传统检索方法：

```python
from rank_bm25 import BM25Okapi
from typing import List
import jieba

class BM25Retriever:
    """基于 BM25 的稀疏检索器"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        # 中文分词
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def retrieve(self, query: str, k: int = 5) -> List[tuple]:
        """检索最相关的文档"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k
        top_k_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        return [(self.documents[i], scores[i]) for i in top_k_indices]

# 使用示例
documents = [
    "Python 是一种广泛使用的编程语言，由 Guido van Rossum 创建。",
    "机器学习是人工智能的核心技术，包括监督学习和无监督学习。",
    "深度学习是机器学习的子领域，使用神经网络进行学习。",
    "Transformer 架构是现代自然语言处理的基础。"
]

bm25_retriever = BM25Retriever(documents)

results = bm25_retriever.retrieve("Python 编程语言", k=2)
for doc, score in results:
    print(f"分数: {score:.4f}")
    print(f"文档: {doc}")
    print()
```

**优点**：
- 精确匹配关键词
- 计算效率高
- 对专业术语效果好

**缺点**：
- 无法理解语义
- 同义词问题
- 需要分词（中文）

### 3. 混合检索（Hybrid Retrieval）

结合稠密检索和稀疏检索的优势：

```python
from typing import List, Tuple
import numpy as np

class HybridRetriever:
    """混合检索器：稠密检索 + 稀疏检索"""
    
    def __init__(self, vectorstore, bm25_retriever, 
                 dense_weight: float = 0.5):
        self.vectorstore = vectorstore
        self.bm25 = bm25_retriever
        self.dense_weight = dense_weight  # 稠密检索权重
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple]:
        """
        混合检索
        Args:
            query: 查询文本
            k: 返回文档数量
        """
        # 1. 稠密检索
        dense_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 2. 稀疏检索
        sparse_results = self.bm25.retrieve(query, k=k*2)
        
        # 3. 分数归一化
        dense_scores = self._normalize_scores([r[1] for r in dense_results])
        sparse_scores = self._normalize_scores([r[1] for r in sparse_results])
        
        # 4. 创建文档到分数的映射
        doc_scores = {}
        
        for (doc, _), score in zip(dense_results, dense_scores):
            doc_content = doc.page_content
            if doc_content not in doc_scores:
                doc_scores[doc_content] = {"dense": 0, "sparse": 0, "doc": doc}
            doc_scores[doc_content]["dense"] = max(doc_scores[doc_content]["dense"], score)
        
        for (doc_content, _), score in zip(sparse_results, sparse_scores):
            if doc_content not in doc_scores:
                # 需要从原始文档创建 Document 对象
                from langchain.schema import Document
                doc_scores[doc_content] = {
                    "dense": 0, 
                    "sparse": 0, 
                    "doc": Document(page_content=doc_content)
                }
            doc_scores[doc_content]["sparse"] = max(doc_scores[doc_content]["sparse"], score)
        
        # 5. 计算加权分数
        combined_scores = []
        for doc_content, scores in doc_scores.items():
            final_score = (
                self.dense_weight * scores["dense"] + 
                (1 - self.dense_weight) * scores["sparse"]
            )
            combined_scores.append((scores["doc"], final_score))
        
        # 6. 排序返回
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-Max 归一化"""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]


# 使用示例
hybrid_retriever = HybridRetriever(
    vectorstore=vectorstore,
    bm25_retriever=bm25_retriever,
    dense_weight=0.6  # 60% 稠密检索，40% 稀疏检索
)

results = hybrid_retriever.retrieve("机器学习的核心技术", k=3)
for doc, score in results:
    print(f"混合分数: {score:.4f}")
    print(f"内容: {doc.page_content[:100]}...")
    print()
```

### LangChain 中的混合检索

```python
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever

# 创建向量检索器
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 组合成混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 向量检索 60%，BM25 40%
)

# 检索
results = ensemble_retriever.get_relevant_documents("机器学习算法")
for doc in results[:3]:
    print(f"- {doc.page_content[:80]}...")
```

### 检索策略选择指南

```
检索策略选择决策：

数据特点？
├── 专业术语多 → 优先稀疏检索或混合检索
├── 同义词多   → 优先稠密检索
└── 混合特点   → 混合检索
                  
性能要求？
├── 高精度     → 混合检索 + 重排序
├── 高速度     → 稀疏检索
└── 平衡       → 稠密检索
                  
语言？
├── 中文       └── 需要分词（BM25）+ 中文 Embedding
├── 英文       └── E5 / OpenAI Embedding
└── 多语言     └── BGE-M3 / OpenAI
```

---

## 📊 文档切分策略

### 为什么需要切分

```
原始文档可能很长：

┌─────────────────────────────────────────────────────────┐
│                    一份 50 页的技术文档                   │
│                                                         │
│  "RAG 是一种将信息检索与文本生成相结合的技术...          │
│   [大量内容]                                            │
│   ...向量数据库的选择需要考虑以下因素...                 │
│   [大量内容]                                            │
│   ...总结..."                                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ 切分
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Chunk 1 │ │  Chunk 2 │ │  Chunk 3 │ │  Chunk 4 │
│  RAG介绍 │ │ 向量数据库│ │ 检索策略  │ │ ...      │
│  500字   │ │  500字   │ │  500字   │ │          │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 切分策略

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter
)
from langchain.schema import Document

# 1. 递归字符切分（推荐）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # 每块最大字符数
    chunk_overlap=50,         # 块之间的重叠
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    length_function=len
)

# 2. 按 Markdown 标题切分
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]
)

# 3. 按 Token 数量切分
from langchain.text_splitter import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=300,           # 每个 chunk 的 token 数
    chunk_overlap=30
)

# 使用示例
long_text = """
# 机器学习概述

机器学习是人工智能的核心技术之一。它使计算机能够从数据中自动学习。

## 监督学习

监督学习是最常见的机器学习类型。它使用标记数据进行训练。

常见的监督学习算法包括：
- 线性回归
- 逻辑回归
- 决策树

## 无监督学习

无监督学习使用未标记的数据进行学习。
"""

# 递归切分
chunks = recursive_splitter.split_text(long_text)
print(f"递归切分结果：{len(chunks)} 个块")

# Markdown 标题切分
md_chunks = markdown_splitter.split_text(long_text)
print(f"\nMarkdown 切分结果：{len(md_chunks)} 个块")
for chunk in md_chunks:
    print(f"  - {chunk.metadata}")
```

### 切分参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **chunk_size** | 300-1000 | 太小：语义不完整；太大：检索精度下降 |
| **chunk_overlap** | chunk_size 的 10-20% | 保证上下文连贯 |
| **separators** | 按优先级排序 | 中文优先用句号、段落分隔 |

### 高级切分：语义切分

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

# 基于语义变化的切分
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 或 "gradient"
    breakpoint_threshold_amount=95
)

# 自动在语义变化点切分
chunks = semantic_splitter.split_text(long_text)
```

---

## 📈 检索效果评估

### 评估指标

```python
def evaluate_retrieval(retrieved_docs: list, relevant_docs: set, k: int = 5):
    """
    评估检索效果
    Args:
        retrieved_docs: 检索返回的文档列表
        relevant_docs: 相关文档的 ID 集合（人工标注）
        k: 评估前 k 个结果
    """
    retrieved_ids = [doc.metadata.get("id") for doc in retrieved_docs[:k]]
    
    # 命中数
    hits = len(set(retrieved_ids) & relevant_docs)
    
    # Precision@K
    precision = hits / k
    
    # Recall@K
    recall = hits / len(relevant_docs) if relevant_docs else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_docs:
            mrr = 1 / (i + 1)
            break
    
    # NDCG@K (简化版)
    dcg = sum(
        1 / np.log2(i + 2) if doc_id in relevant_docs else 0
        for i, doc_id in enumerate(retrieved_ids)
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_docs))))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        "MRR": mrr,
        f"NDCG@{k}": ndcg
    }

# 使用 RAGAS 评估框架
# pip install ragas

from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# 需要准备测试数据集
# dataset = {
#     "question": [...],
#     "contexts": [[...]],  # 检索到的文档
#     "ground_truth": [...]  # 标准答案
# }
# results = evaluate(dataset, metrics=[context_precision, context_recall])
```

---

## 🔗 相关内容

- [RAG 解决的问题](./rag-basics.md) - 了解为什么需要 RAG
- [生成器模块](./generator.md) - 学习如何处理检索到的文档
- [高级 RAG 技术](./advanced-rag.md) - 重排序、多跳检索等进阶内容
