# LlamaIndex 基础概念与快速开始

LlamaIndex 是构建 RAG（检索增强生成）应用的核心框架。本文将介绍其核心概念，并通过实际代码帮助你快速上手。

---

## 📌 核心概念

LlamaIndex 的设计围绕几个核心抽象展开，理解这些概念是掌握框架的关键：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LlamaIndex 核心概念流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐ │
│   │ Documents  │ →  │   Nodes    │ →  │   Index    │ →  │  Query    │ │
│   │  原始文档   │    │   文本块    │    │   索引     │    │  Engine   │ │
│   └────────────┘    └────────────┘    └────────────┘    └───────────┘ │
│         ↓                 ↓                  ↓                ↓        │
│   加载各种格式        解析与分块          构建向量索引       执行查询     │
│   的数据文件          添加元数据          存储到向量库       生成回答     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Documents（文档）

**Document** 是 LlamaIndex 中最基础的数据单元，代表一个原始数据源：

```python
from llama_index.core import Document

# 创建文档
doc = Document(
    text="这是文档的文本内容...",
    metadata={"author": "张三", "date": "2024-01-01"}  # 可选元数据
)

print(doc.text)        # 文本内容
print(doc.metadata)    # 元数据字典
print(doc.doc_id)      # 唯一标识符
```

**Document 的特点**：

| 属性 | 说明 |
|------|------|
| `text` | 文档的文本内容 |
| `metadata` | 元数据字典，可用于过滤和追溯 |
| `doc_id` | 文档唯一标识，自动生成或手动指定 |
| `relationships` | 与其他文档/节点的关系 |

### Nodes（节点）

**Node** 是文档的一个片段，是索引和检索的基本单位：

```python
from llama_index.core import Node

# 创建节点
node = Node(
    text="这是一个文本块的内容...",
    metadata={"source": "doc_001", "page": 1}
)

print(node.text)           # 节点文本
print(node.node_id)        # 节点ID
print(node.metadata)       # 元数据
print(node.relationships)  # 与其他节点的关系
```

**Documents 与 Nodes 的关系**：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 将文档解析为节点
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

print(f"文档数: {len(documents)}")
print(f"节点数: {len(nodes)}")
```

### Index（索引）

**Index** 是组织和检索数据的核心组件，将 Nodes 以特定方式组织以便高效查询：

```python
from llama_index.core import VectorStoreIndex

# 从节点构建索引
index = VectorStoreIndex(nodes)

# 或直接从文档构建
index = VectorStoreIndex.from_documents(documents)
```

**主要索引类型**：

| 索引类型 | 适用场景 | 检索方式 |
|----------|----------|----------|
| `VectorStoreIndex` | 语义检索 | 向量相似度 |
| `SummaryIndex` | 小数据集、精确检索 | 遍历所有节点 |
| `KeywordTableIndex` | 关键词匹配 | 关键词提取与匹配 |
| `KnowledgeGraphIndex` | 实体关系推理 | 知识图谱查询 |

### Query Engine（查询引擎）

**Query Engine** 是执行查询的接口，封装了检索和生成的完整流程：

```python
# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query("什么是 RAG？")

print(response)  # 生成的回答
```

### Retriever（检索器）

**Retriever** 负责从索引中检索相关节点，是 Query Engine 的核心组件：

```python
# 获取检索器
retriever = index.as_retriever(similarity_top_k=5)

# 执行检索
nodes = retriever.retrieve("什么是向量数据库？")

for node in nodes:
    print(f"分数: {node.score:.3f}")
    print(f"内容: {node.text[:100]}...")
```

---

## 🚀 快速开始

### 环境准备

```bash
# 创建虚拟环境
conda create -n llamaindex python=3.10
conda activate llamaindex

# 安装核心包
pip install llama-index

# 安装常用组件
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma
pip install llama-index-readers-file
```

### 配置 API Key

```python
import os

# 方式一：环境变量
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 方式二：代码中直接配置
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

llm = OpenAI(api_key="your-api-key", model="gpt-4")
embed_model = OpenAIEmbedding(api_key="your-api-key")
```

### 使用 Settings 全局配置

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 设置全局配置
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

### 第一个 RAG 应用

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 加载文档
# 创建 ./data 目录并放入一些 txt/pdf 文件
documents = SimpleDirectoryReader("./data").load_data()
print(f"加载了 {len(documents)} 个文档")

# 2. 构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 执行查询
response = query_engine.query("文档的主要内容是什么？")
print(response)
```

---

## 📂 数据加载器

LlamaIndex 提供了丰富的数据加载器（Readers），支持多种数据源：

### SimpleDirectoryReader

最常用的文件加载器，支持多种文件格式：

```python
from llama_index.core import SimpleDirectoryReader

# 加载目录下所有支持的文件
documents = SimpleDirectoryReader("./data").load_data()

# 指定特定文件类型
documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".txt", ".md"]
).load_data()

# 加载单个文件
documents = SimpleDirectoryReader(
    input_files=["./data/report.pdf"]
).load_data()

# 递归加载子目录
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,  # 递归子目录
    exclude=["*.tmp", "*.bak"]  # 排除文件
).load_data()
```

### 支持的文件格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 文本 | `.txt` | 纯文本文件 |
| Markdown | `.md` | Markdown 文档 |
| PDF | `.pdf` | PDF 文档（需要 `pypdf`） |
| Word | `.docx` | Word 文档（需要 `python-docx`） |
| CSV | `.csv` | 表格数据 |
| JSON | `.json` | JSON 数据 |
| HTML | `.html` | 网页文件 |

### 其他常用加载器

```python
# Web 页面加载器
from llama_index.readers.web import SimpleWebPageReader
reader = SimpleWebPageReader(html_to_text=True)
documents = reader.load_data(urls=["https://example.com/article"])

# 数据库加载器
from llama_index.readers.database import DatabaseReader
reader = DatabaseReader(sql_database=sql_database)
documents = reader.load_data(query="SELECT * FROM articles")

# Notion 加载器
from llama_index.readers.notion import NotionPageReader
reader = NotionPageReader(integration_token="your-token")
documents = reader.load_data(page_ids=["page-id-1", "page-id-2"])
```

---

## 🔧 节点解析器

节点解析器（Node Parser）负责将文档分割成适当大小的节点：

### SentenceSplitter

按句子分割，是最常用的解析器：

```python
from llama_index.core.node_parser import SentenceSplitter

# 创建解析器
parser = SentenceSplitter(
    chunk_size=512,      # 每个块的最大 token 数
    chunk_overlap=50,    # 相邻块之间的重叠 token 数
    paragraph_separator="\n\n"  # 段落分隔符
)

# 解析文档
nodes = parser.get_nodes_from_documents(documents)
```

### SemanticSplitter

基于语义相似度进行分割，保持语义完整性：

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# 需要嵌入模型
embed_model = OpenAIEmbedding()

parser = SemanticSplitterNodeParser(
    buffer_size=1,           # 句子缓冲区大小
    breakpoint_percentile_threshold=95,  # 断点阈值
    embed_model=embed_model
)

nodes = parser.get_nodes_from_documents(documents)
```

### 层级解析

结合多种解析器实现层级分割：

```python
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter
)

# 层级解析器：先按文档分割，再按句子分割
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 三级分割大小
)

nodes = hierarchical_parser.get_nodes_from_documents(documents)
```

---

## 💾 持久化存储

### 本地存储

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage

# 构建索引并保存
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")

# 加载已保存的索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### 向量数据库存储

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# 初始化 Chroma 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引（自动存储到 Chroma）
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 从 Chroma 加载索引
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)
```

---

## 🎯 查询引擎配置

### 基本配置

```python
# 默认配置
query_engine = index.as_query_engine()

# 自定义配置
query_engine = index.as_query_engine(
    similarity_top_k=5,       # 检索 top-k 个节点
    response_mode="compact",  # 响应模式
    streaming=False           # 是否流式输出
)
```

### 响应模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `compact` | 压缩所有节点后生成回答 | 默认模式，通用 |
| `refine` | 逐个节点迭代优化答案 | 需要精确答案 |
| `simple_summarize` | 简单总结所有节点 | 快速概览 |
| `tree_summarize` | 层级总结 | 大量节点 |
| `no_text` | 仅返回检索结果 | 仅需检索 |

```python
# 使用 refine 模式获得更精确的答案
query_engine = index.as_query_engine(
    response_mode="refine",
    similarity_top_k=5
)
```

### 流式输出

```python
# 启用流式输出
query_engine = index.as_query_engine(streaming=True)

# 执行查询
streaming_response = query_engine.query("文档的主要内容是什么？")

# 逐块输出
for text in streaming_response.response_gen:
    print(text, end="", flush=True)
```

---

## 💡 常见问题

### Q1: 如何处理中文文档？

```python
# 使用支持中文的嵌入模型
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large"  # 对中文支持更好
)

# 或使用本地中文嵌入模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5"
)
```

### Q2: 如何控制 Token 消耗？

```python
# 限制检索数量
query_engine = index.as_query_engine(similarity_top_k=3)

# 使用更小的块大小
Settings.chunk_size = 256
Settings.chunk_overlap = 20

# 使用更便宜的模型
Settings.llm = OpenAI(model="gpt-4o-mini")
```

### Q3: 如何查看检索到的源文档？

```python
response = query_engine.query("问题内容")

# 查看源节点
for i, node in enumerate(response.source_nodes):
    print(f"\n--- 源文档 {i+1} (相关度: {node.score:.3f}) ---")
    print(node.text[:200] + "...")
    print(f"来源: {node.metadata}")
```

---

## 📚 参考资料

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex 核心概念](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)
- [数据加载器列表](https://llamahub.ai/)

---

*下一章：[索引类型与选择](./llamaindex-index.md) →*
