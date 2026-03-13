# LlamaIndex RAG 应用构建

RAG（Retrieval-Augmented Generation，检索增强生成）是当前构建 LLM 应用的主流架构。本章将介绍如何使用 LlamaIndex 构建生产级 RAG 应用。

---

## 📌 RAG 架构概述

### 什么是 RAG？

RAG 通过检索相关文档来增强 LLM 的生成能力，解决 LLM 的知识边界问题：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG 架构全景                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐ │
│   │                        离线阶段：数据准备                          │ │
│   │                                                                  │ │
│   │   文档 → 分块 → 嵌入 → 向量存储                                   │ │
│   │    ↓      ↓       ↓       ↓                                      │ │
│   │  PDF    512字   Embed   Chroma                                   │ │
│   │  Word   chunk   Model   Milvus                                   │ │
│   │  Web                                         Pinecone             │ │
│   └──────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                    │
│   ┌──────────────────────────────────────────────────────────────────┐ │
│   │                        在线阶段：查询处理                          │ │
│   │                                                                  │ │
│   │   用户查询 → 查询嵌入 → 向量检索 → 重排序 → 上下文组装 → LLM生成   │ │
│   │      ↓          ↓          ↓         ↓          ↓         ↓      │ │
│   │   "问题"   [0.1,..]   Top-100    Top-5    Prompt + Context 回答   │ │
│   └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RAG vs 微调

| 对比维度 | RAG | 微调 |
|----------|-----|------|
| **知识更新** | 实时更新文档即可 | 需要重新训练 |
| **成本** | 较低 | 较高 |
| **可解释性** | 高，可追溯来源 | 低 |
| **适用场景** | 知识密集型任务 | 风格/行为定制 |
| **数据隐私** | 数据本地存储 | 数据进入模型 |

---

## 🚀 基础 RAG 实现

### 最简 RAG 应用

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 查询
response = query_engine.query("文档的主要内容是什么？")
print(response)
```

### 完整 RAG Pipeline

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import chromadb

# ==================== 配置 ====================
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ==================== 向量存储 ====================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==================== 索引构建 ====================
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# ==================== 重排序 ====================
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=5
)

# ==================== 查询引擎 ====================
query_engine = index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[reranker],
    streaming=True  # 启用流式输出
)

# ==================== 执行查询 ====================
response = query_engine.query("请总结文档的主要观点")

# 流式输出
for text in response.response_gen:
    print(text, end="", flush=True)
```

---

## 📄 文档处理与分块

### 文档分块策略

分块是 RAG 中最关键的设计决策之一：

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    MarkdownNodeParser
)
from llama_index.embeddings.openai import OpenAIEmbedding

# ========== 固定大小分块 ==========
splitter = SentenceSplitter(
    chunk_size=512,      # 块大小（token 数）
    chunk_overlap=50,    # 重叠大小
    paragraph_separator="\n\n"
)

# ========== 语义分块 ==========
# 根据语义相似度自动确定分割点
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding()
)

# ========== 层级分块 ==========
# 创建父子节点关系，支持递归检索
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 三层结构
)

# ========== Markdown 分块 ==========
# 保持 Markdown 结构完整性
md_parser = MarkdownNodeParser()
```

### 分块策略选择

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **固定大小** | 通用场景 | 简单可控 | 可能切断语义 |
| **语义分块** | 连续文档 | 保持语义完整 | 计算成本高 |
| **层级分块** | 长文档 | 支持多层次检索 | 复杂度高 |
| **结构分块** | Markdown/代码 | 保持结构 | 依赖格式 |

### 最佳实践

```python
# 推荐的通用配置
Settings.chunk_size = 512      # 适合大多数场景
Settings.chunk_overlap = 50    # 10-20% 的重叠

# 中文文档建议
Settings.chunk_size = 256      # 中文 token 较少
Settings.chunk_overlap = 30

# 技术文档建议
Settings.chunk_size = 1024     # 保持代码/配置完整
Settings.chunk_overlap = 100
```

---

## 🗄️ 向量数据库集成

### Chroma（开发推荐）

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# 持久化存储
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # 相似度度量
)

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 加载已存储的索引
index = VectorStoreIndex.from_vector_store(vector_store)
```

### Milvus（生产推荐）

```python
from pymilvus import MilvusClient
from llama_index.vector_stores.milvus import MilvusVectorStore

# 连接 Milvus
milvus_client = MilvusClient(uri="http://localhost:19530")

# 创建向量存储
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="documents",
    dim=1536,  # 嵌入维度
    overwrite=False  # 是否覆盖已有数据
)

# 构建索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

### Pinecone（云服务推荐）

```python
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 初始化 Pinecone
pc = Pinecone(api_key="your-api-key")
pinecone_index = pc.Index("documents")

# 创建向量存储
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace="default"
)

# 构建索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

### 向量数据库选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 开发测试 | Chroma | 轻量、易用、本地持久化 |
| 小规模生产 | Chroma/Qdrant | 单机部署足够 |
| 中大规模生产 | Milvus | 高性能、可扩展 |
| 云原生部署 | Pinecone | 全托管、免运维 |
| 数据敏感场景 | 本地 Milvus/Qdrant | 数据不出域 |

---

## 💬 对话式 RAG

### 基本对话

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建对话查询引擎
chat_engine = index.as_chat_engine()

# 多轮对话
response = chat_engine.chat("文档讲了什么？")
print(response)

response = chat_engine.chat("能详细解释第一点吗？")  # 有上下文记忆
print(response)
```

### 对话模式

```python
# condense_question 模式：将追问转换为独立问题
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    verbose=True
)

# condense_plus_context 模式：结合上下文重写问题
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context"
)

# react 模式：Agent 模式，可调用工具
chat_engine = index.as_chat_engine(
    chat_mode="react",
    verbose=True
)

# context 模式：简单检索 + 记忆
chat_engine = index.as_chat_engine(
    chat_mode="context"
)
```

### 流式对话

```python
# 启用流式输出
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    streaming=True
)

# 流式响应
response = chat_engine.stream_chat("文档的核心观点是什么？")

for text in response.response_gen:
    print(text, end="", flush=True)
```

### 对话记忆管理

```python
from llama_index.core.memory import ChatMemoryBuffer

# 创建带记忆限制的对话引擎
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory
)

# 重置对话
chat_engine.reset()
```

---

## 📊 RAG 评估

### 使用 TruLens 评估

```python
from trulens_eval import TruChain, Feedback, Tru
from llama_index.core import VectorStoreIndex

# 初始化 Tru
tru = Tru()

# 创建 RAG 应用
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 定义反馈函数
from trulens_eval.feedback.provider import OpenAI
provider = OpenAI()

f_relevance = Feedback(provider.relevance)
f_groundedness = Feedback(provider.groundedness_measure)

# 包装应用
tru_query_engine = TruChain(
    query_engine,
    feedbacks=[f_relevance, f_groundedness],
    tru=tru
)

# 执行查询并评估
response = tru_query_engine.query("查询内容")

# 查看评估结果
tru.run_dashboard()
```

### 使用 Ragas 评估

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": ["问题1", "问题2"],
    "answer": ["生成的答案1", "生成的答案2"],
    "contexts": [["检索上下文1"], ["检索上下文2"]],
    "ground_truth": ["真实答案1", "真实答案2"]
}

dataset = Dataset.from_dict(eval_data)

# 评估
results = evaluate(
    dataset,
    metrics=[
        context_precision,   # 检索精度
        context_recall,      # 检索召回
        faithfulness,        # 答案忠实度
        answer_relevancy     # 答案相关性
    ]
)

print(results)
```

### 评估指标说明

| 指标 | 说明 | 目标值 |
|------|------|--------|
| Context Precision | 检索内容中有多少是相关的 | > 0.8 |
| Context Recall | 相关内容有多少被检索到 | > 0.8 |
| Faithfulness | 答案是否基于检索内容 | > 0.9 |
| Answer Relevancy | 答案是否回答了问题 | > 0.8 |

---

## 🔧 高级 RAG 模式

### 多模态 RAG

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# 加载包含图像的文档
documents = SimpleDirectoryReader(
    "./data",
    exclude=["*.tmp"]
).load_data()

# 使用多模态 LLM
mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1000)

# 构建多模态索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(
    multi_modal_llm=mm_llm
)
```

### 多文档 RAG

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# 为每个文档源创建工具
tools = []

# 文档源1：技术文档
tech_index = VectorStoreIndex.from_documents(tech_docs)
tech_tool = QueryEngineTool(
    query_engine=tech_index.as_query_engine(),
    metadata=ToolMetadata(
        name="tech_docs",
        description="技术文档，包含 API 说明和代码示例"
    )
)
tools.append(tech_tool)

# 文档源2：产品文档
product_index = VectorStoreIndex.from_documents(product_docs)
product_tool = QueryEngineTool(
    query_engine=product_index.as_query_engine(),
    metadata=ToolMetadata(
        name="product_docs",
        description="产品文档，包含功能说明和使用指南"
    )
)
tools.append(product_tool)

# 创建 Agent 自动路由
agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)

# 执行查询
response = agent.chat("如何使用 API 创建用户？")
```

### 知识图谱 RAG

```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# 连接 Neo4j
graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j"
)

# 构建知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(graph_store=graph_store),
    max_triplets_per_chunk=10
)

# 查询
query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize"
)
response = query_engine.query("实体A和实体B是什么关系？")
```

---

## 💡 生产最佳实践

### 1. 文档预处理

```python
# 清理和标准化
def preprocess_document(text):
    # 移除多余空白
    text = " ".join(text.split())
    # 移除特殊字符
    text = text.replace("\x00", "")
    return text

# 批量处理
for doc in documents:
    doc.text = preprocess_document(doc.text)
```

### 2. 错误处理

```python
from llama_index.core import VectorStoreIndex

def safe_query(query_engine, query, max_retries=3):
    """安全的查询包装器"""
    for attempt in range(max_retries):
        try:
            response = query_engine.query(query)
            return response
        except Exception as e:
            print(f"查询失败 (尝试 {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return f"抱歉，查询出现问题：{str(e)}"
```

### 3. 缓存优化

```python
from llama_index.core import Settings
from llama_index.core.cache import Cache

# 启用嵌入缓存
Settings.embed_model.cache = Cache()

# 或使用 Redis 缓存
from llama_index.storage.cache.redis import RedisCache

redis_cache = RedisCache(redis_url="redis://localhost:6379")
Settings.embed_model.cache = redis_cache
```

### 4. 批量索引

```python
from llama_index.core import VectorStoreIndex, Document

# 批量插入优化
def batch_insert(index, documents, batch_size=100):
    """批量插入文档"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        for doc in batch:
            index.insert(doc)
        print(f"已插入 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
```

---

## 📚 参考资料

- [LlamaIndex RAG 教程](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)
- [Ragas 评估文档](https://docs.ragas.io/)
- [TruLens 文档](https://www.trulens.org/)

---

*下一章：[高级用法与优化](./llamaindex-advanced.md) →*
