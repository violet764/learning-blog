# LlamaIndex 索引类型与选择

索引是 LlamaIndex 的核心组件，决定了数据的组织方式和检索效率。选择合适的索引类型对 RAG 系统的性能至关重要。

---

## 📌 索引概述

索引的主要作用是将文本数据组织成可检索的结构。不同类型的索引适用于不同的场景：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LlamaIndex 索引类型对比                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   VectorStoreIndex          SummaryIndex          KeywordTableIndex    │
│   ┌───────────────┐         ┌───────────────┐     ┌───────────────┐   │
│   │    向量检索    │         │   全文遍历    │     │   关键词匹配   │   │
│   │   ┌───┐       │         │  ┌─┬─┬─┬─┬─┐  │     │  ┌─┐ ┌─┐ ┌─┐ │   │
│   │   │ Q │────→  │         │  │1│2│3│4│5│  │     │  │K│→│N│→│N│ │   │
│   │   └───┘  ┌───┐│         │  └─┴─┴─┴─┴─┘  │     │  └─┘ └─┘ └─┘ │   │
│   │         │ N ││         │       ↓        │     │      ↓        │   │
│   │         └───┘│         │    Summarize   │     │   Filter      │   │
│   └───────────────┘         └───────────────┘     └───────────────┘   │
│                                                                         │
│   适用: 大规模语义检索        适用: 小数据集精确         适用: 关键词精确  │
│                              完整检索                   匹配场景        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ VectorStoreIndex（向量索引）

向量索引是最常用、最强大的索引类型，通过向量相似度进行语义检索。

### 基本使用

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 执行查询
response = query_engine.query("什么是机器学习？")
```

### 工作原理

```
┌────────────────────────────────────────────────────────────────────┐
│                    VectorStoreIndex 工作流程                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. 文档分块          2. 向量化           3. 存储         4. 检索  │
│  ┌─────────┐        ┌─────────┐       ┌─────────┐     ┌─────────┐ │
│  │Document │        │  Node   │       │ Vector  │     │ Query   │ │
│  │  │      │   →    │  Text   │   →   │ Embed   │  →  │ Vector  │ │
│  │  ↓      │        │    ↓    │       │  Model  │     │    ↓    │ │
│  │ Nodes[] │        │ [0.1,..]│       │ Store   │     │ Similar │ │
│  └─────────┘        └─────────┘       └─────────┘     │ Search  │ │
│                                                       └─────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 向量数据库集成

#### Chroma

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# 初始化 Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 从存储加载
index = VectorStoreIndex.from_vector_store(vector_store)
```

#### Pinecone

```python
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 初始化 Pinecone
pc = Pinecone(api_key="your-api-key")
pinecone_index = pc.Index("your-index-name")

# 创建向量存储
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

#### Milvus

```python
from pymilvus import MilvusClient
from llama_index.vector_stores.milvus import MilvusVectorStore

# 初始化 Milvus
milvus_client = MilvusClient(uri="./milvus.db")

# 创建向量存储
vector_store = MilvusVectorStore(
    uri="./milvus.db",
    collection_name="documents",
    dim=1536  # 嵌入维度
)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
)
```

#### Weaviate

```python
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# 初始化 Weaviate
client = weaviate.connect_to_local()

# 创建向量存储
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="Documents"
)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
)
```

### 向量数据库对比

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Chroma** | 轻量级，易部署，支持持久化 | 开发测试、中小规模应用 |
| **Pinecone** | 全托管，高性能，可扩展 | 生产环境、大规模应用 |
| **Milvus** | 开源，高性能，支持分布式 | 自部署、大规模生产 |
| **Weaviate** | 开源，支持混合检索，GraphQL | 复杂检索需求 |
| **Qdrant** | 开源，高性能，Rust实现 | 高性能要求场景 |
| **FAISS** | Meta开源，本地运行，极高性能 | 本地部署、研究场景 |

---

## 📋 SummaryIndex（摘要索引）

SummaryIndex（原 ListIndex）遍历所有节点进行综合回答，适合小数据集或需要完整分析的场景。

### 基本使用

```python
from llama_index.core import SummaryIndex

# 构建摘要索引
index = SummaryIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询 - 会遍历所有节点
response = query_engine.query("总结所有文档的主要内容")
```

### 响应模式

```python
# 使用不同的响应模式
query_engine = index.as_query_engine(
    response_mode="tree_summarize"  # 层级总结
)

# 可选模式：
# - "compact": 默认，压缩所有内容后回答
# - "refine": 逐个优化答案
# - "tree_summarize": 层级总结，适合大量节点
# - "simple_summarize": 简单拼接总结
```

### 适用场景

| 场景 | 说明 |
|------|------|
| 文档总结 | 对所有内容进行综合总结 |
| 小数据集 | 数据量小，遍历成本低 |
| 完整性要求高 | 需要覆盖所有信息的场景 |

---

## 🔑 KeywordTableIndex（关键词索引）

通过关键词提取和匹配进行检索，适合关键词精确匹配场景。

### 基本使用

```python
from llama_index.core import KeywordTableIndex

# 构建关键词索引
index = KeywordTableIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query("Python 和机器学习的关系")
```

### 工作原理

```
查询: "Python 机器学习"
         │
         ↓
    关键词提取
  ["Python", "机器学习"]
         │
         ↓
    表格匹配查找
  ┌───────────────────┐
  │ Python → [N1, N3] │
  │ 机器学习 → [N2, N3]│
  └───────────────────┘
         │
         ↓
    合并相关节点
    [N1, N2, N3]
         │
         ↓
      LLM 生成回答
```

---

## 🕸️ KnowledgeGraphIndex（知识图谱索引）

基于知识图谱进行检索，适合需要理解实体关系的场景。

### 基本使用

```python
from llama_index.core import KnowledgeGraphIndex

# 构建知识图谱索引
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10  # 每个块最多提取的关系三元组数
)

# 创建查询引擎
query_engine = index.as_query_engine(
    include_text=True,  # 同时包含原始文本
    response_mode="tree_summarize"
)

# 执行查询
response = query_engine.query("张三和李四是什么关系？")
```

### 知识图谱存储

```python
# 使用 NebulaGraph 存储
from llama_index.graph_stores.nebula import NebulaGraphStore

graph_store = NebulaGraphStore(
    space_name="my_space",
    edge_types=["relationship"],
    rel_prop_names=["name"],
    tags=["entity"]
)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(graph_store=graph_store)
)
```

---

## 🔄 多文档代理索引

MultiDocumentAgent 适合处理多个文档，自动选择最相关的文档进行查询。

### 基本使用

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# 为每个文档创建索引和工具
tools = []
for i, doc in enumerate(documents):
    index = VectorStoreIndex.from_documents([doc])
    query_engine = index.as_query_engine()
    
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=f"doc_{i}",
            description=f"文档 {doc.metadata.get('file_name', i)}"
        )
    )
    tools.append(tool)

# 创建代理
agent = ReActAgent.from_tools(tools, llm=Settings.llm)

# 执行查询
response = agent.chat("哪个文档提到了人工智能？")
```

---

## 📊 索引选择指南

### 按场景选择

| 场景 | 推荐索引 | 原因 |
|------|----------|------|
| 语义相似检索 | VectorStoreIndex | 向量检索效果好 |
| 小数据集完整分析 | SummaryIndex | 遍历成本低 |
| 关键词精确匹配 | KeywordTableIndex | 关键词匹配准确 |
| 实体关系推理 | KnowledgeGraphIndex | 图谱适合关系查询 |
| 多文档智能路由 | MultiDocumentAgent | 自动选择相关文档 |

### 按数据规模选择

```
数据规模         推荐索引
────────────────────────────────
< 10 节点       SummaryIndex
10-100 节点     VectorStoreIndex（内存）
100-1000 节点   VectorStoreIndex + 向量库
> 1000 节点     VectorStoreIndex + 分布式向量库
```

### 性能对比

```python
import time

def benchmark_index(index_class, documents, query):
    """索引性能基准测试"""
    start = time.time()
    
    # 构建索引
    index = index_class.from_documents(documents)
    build_time = time.time() - start
    
    # 查询
    start = time.time()
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    query_time = time.time() - start
    
    return {
        "index_type": index_class.__name__,
        "build_time": build_time,
        "query_time": query_time
    }
```

---

## 💡 高级配置

### 自定义嵌入模型

```python
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# OpenAI 嵌入
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1536
)

# 本地嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5"
)
```

### 索引持久化

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引
index.storage_context.persist(persist_dir="./index_storage")

# 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)
```

### 增量更新

```python
# 向索引添加新文档
from llama_index.core import Document

new_doc = Document(text="新的文档内容...")
index.insert(new_doc)

# 批量插入
for doc in new_documents:
    index.insert(doc)
```

---

## 📚 参考资料

- [LlamaIndex 索引文档](https://docs.llamaindex.ai/en/stable/module_guides/indexing/)
- [向量数据库集成](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)
- [知识图谱索引](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphIndexDemo/)

---

*下一章：[检索策略](./llamaindex-retrieval.md) →*
