# LlamaIndex 学习指南

LlamaIndex（原 GPT Index）是一个专注于构建 LLM 应用的数据框架，特别擅长于**检索增强生成（RAG）**场景。它提供了丰富的工具来连接自定义数据与大语言模型，让开发者能够快速构建基于私有数据的智能应用。

---

## 📌 什么是 LlamaIndex？

LlamaIndex 的核心使命是解决大语言模型的**知识边界问题**——LLM 虽然强大，但其知识受限于训练数据，无法直接访问私有数据或实时信息。LlamaIndex 通过以下方式解决这一问题：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LlamaIndex 核心架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│   │  数据连接器  │ →  │  索引构建   │ →  │  查询引擎   │               │
│   │ (Connectors)│    │  (Indexes)  │    │  (Engines)  │               │
│   └─────────────┘    └─────────────┘    └─────────────┘               │
│         ↓                  ↓                  ↓                        │
│   加载各种数据源      构建向量/关键词索引    执行智能检索查询             │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                       核心组件关系                               │  │
│   │                                                                 │  │
│   │   Documents → Nodes → Index → Retriever → QueryEngine → LLM    │  │
│   │      ↓         ↓        ↓          ↓            ↓              │  │
│   │   原始文档   文本块    向量存储    检索策略    查询响应          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心价值

| 特性 | 说明 |
|------|------|
| **数据连接** | 支持 PDF、数据库、API、网页等多种数据源 |
| **智能索引** | 向量索引、关键词索引、知识图谱索引等多种选择 |
| **灵活检索** | 可配置的检索策略，支持混合检索、重排序 |
| **查询优化** | 自动构建 Prompt，支持多轮对话、流式输出 |
| **易于扩展** | 模块化设计，可自定义每个组件 |

---

## 🗺️ 学习路径

```
阶段一：基础入门（1-2周）
├── 理解 RAG 基本概念
├── 安装与环境配置
├── 第一个 RAG 应用
└── 数据加载与节点解析

阶段二：索引与检索（2-3周）
├── 索引类型对比与选择
├── 向量数据库集成
├── 检索策略优化
└── 分块策略设计

阶段三：RAG 应用实战（2-3周）
├── 完整 RAG Pipeline 构建
├── 多模态 RAG
├── 对话式 RAG
└── 评估与优化

阶段四：高级应用（持续深入）
├── Agent 与工具调用
├── 多文档推理
├── 生产部署优化
└── 自定义组件开发
```

---

## 📚 模块导航

### 🚀 基础入门

::: tip 学习入口
**[→ 基础概念与快速开始](./llamaindex-basics.md)**
:::

从零开始，掌握 LlamaIndex 的核心概念和基本使用方法。内容包括：
- 核心概念：Documents、Nodes、Index、Query Engine
- 环境配置与安装
- 第一个 RAG 应用
- 数据加载器使用

### 📊 索引系统

::: tip 学习入口
**[→ 索引类型与选择](./llamaindex-index.md)**
:::

深入理解 LlamaIndex 的索引机制，学会根据场景选择合适的索引类型：
- VectorStoreIndex：向量检索
- SummaryIndex：全文检索
- KeywordTableIndex：关键词检索
- KnowledgeGraphIndex：知识图谱
- 向量数据库集成方案

### 🔍 检索策略

::: tip 学习入口
**[→ 检索策略](./llamaindex-retrieval.md)**
:::

掌握高级检索技术，提升 RAG 系统的召回质量和准确性：
- 基础检索 vs 高级检索
- 混合检索策略
- 重排序（Reranking）
- 检索参数调优

### 🤖 RAG 应用

::: tip 学习入口
**[→ RAG 应用构建](./llamaindex-rag.md)**
:::

构建生产级 RAG 应用的完整指南：
- 完整 Pipeline 设计
- 文档处理与分块
- 向量数据库选择
- 对话式 RAG
- RAG 评估方法

### ⚡ 高级用法

::: tip 学习入口
**[→ 高级用法与优化](./llamaindex-advanced.md)**
:::

进阶技术，打造高性能 RAG 系统：
- Agent 与工具调用
- 流式输出与异步
- 性能优化策略
- 生产部署最佳实践

---

## 🛠️ 快速开始

### 安装

```bash
# 基础安装
pip install llama-index

# 常用扩展
pip install llama-index-vector-stores-chroma  # Chroma 向量库
pip install llama-index-readers-file          # 文件读取
pip install llama-index-llms-openai           # OpenAI LLM
pip install llama-index-embeddings-openai     # OpenAI Embedding
```

### 最简示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 执行查询
response = query_engine.query("文档的主要观点是什么？")
print(response)
```

---

## 📋 前置知识

| 知识领域 | 要求程度 | 说明 |
|----------|----------|------|
| Python 编程 | 熟练 | 能够编写 Python 代码 |
| LLM 基础 | 了解 | 知道 Prompt、Token、Embedding 等概念 |
| 向量数据库 | 了解 | 知道向量检索的基本原理 |
| API 使用 | 基础 | 会使用 OpenAI 等 API |

---

## 🔗 相关资源

### 官方资源

| 资源 | 链接 |
|------|------|
| 官方文档 | https://docs.llamaindex.ai |
| GitHub 仓库 | https://github.com/run-llama/llama_index |
| 示例代码 | https://github.com/run-llama/llama_index/tree/main/docs/examples |

### 相关框架

| 框架 | 说明 |
|------|------|
| [LangChain](../langchain/index.md) | 更通用的 LLM 应用框架 |
| [Transformers](../transformers/index.md) | HuggingFace 模型库 |
| [vLLM](../vllm/index.md) | 高性能 LLM 推理引擎 |

---

*准备好开始你的 RAG 之旅了吗？从 [基础概念与快速开始](./llamaindex-basics.md) 开始吧！🚀*
