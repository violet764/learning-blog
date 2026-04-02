# RAG 检索增强生成

检索增强生成（Retrieval-Augmented Generation, RAG）是一种将信息检索与文本生成相结合的技术范式，通过外部知识库增强大语言模型的能力，有效解决了大模型的知识局限性和幻觉问题。

---

## 🎯 什么是 RAG

### 核心概念

RAG 的核心思想是：**在生成回答之前，先从外部知识库中检索相关信息，然后将检索到的信息作为上下文提供给 LLM，引导其生成更准确、更有依据的回答。**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 工作流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    用户查询                                                          │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐         ┌─────────────┐                           │
│  │   查询编码   │ ──────→ │   向量检索   │                           │
│  │   Embedding │         │  Retrieval  │                           │
│  └─────────────┘         └─────────────┘                           │
│       │                         │                                   │
│       │                         ▼                                   │
│       │                   ┌─────────────┐                          │
│       │                   │  外部知识库  │                          │
│       │                   │  Vector DB  │                          │
│       │                   └─────────────┘                          │
│       │                         │                                   │
│       │         检索到的相关文档                                     │
│       │                         │                                   │
│       ▼                         ▼                                   │
│  ┌─────────────────────────────────────┐                           │
│  │            Prompt 组装               │                           │
│  │   Query + Retrieved Context          │                           │
│  └─────────────────────────────────────┘                           │
│                         │                                           │
│                         ▼                                           │
│                  ┌─────────────┐                                    │
│                  │     LLM     │                                    │
│                  │   生成回答   │                                    │
│                  └─────────────┘                                    │
│                         │                                           │
│                         ▼                                           │
│                   最终回答输出                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 为什么需要 RAG

大语言模型虽然强大，但存在以下固有局限：

| 局限性 | 具体表现 | RAG 解决方案 |
|--------|----------|--------------|
| **知识截止** | 训练数据有时间截止点，无法获取最新信息 | 实时检索最新知识 |
| **幻觉问题** | 编造不存在的事实或错误信息 | 基于检索到的真实文档生成 |
| **领域知识缺失** | 对特定领域、私有数据了解不足 | 接入领域知识库 |
| **长尾知识** | 对罕见知识记忆不准确 | 检索补充低频知识 |
| **可解释性差** | 无法追溯答案来源 | 提供引用来源 |

---

## 📊 RAG vs 其他方案对比

### 与 Fine-tuning 对比

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识更新** | 实时更新知识库即可 | 需要重新训练 |
| **成本** | 较低，主要是检索开销 | 较高，需要 GPU 训练 |
| **适用场景** | 知识频繁变化、需要引用来源 | 特定任务风格、领域适应 |
| **可解释性** | 高，可追溯来源 | 低，黑盒模型 |
| **私有数据** | 数据不出本地，安全性高 | 数据需要进入模型 |
| **推理成本** | 略高（检索 + 生成） | 较低（仅生成） |

### 适用场景选择

```
                    知识更新频率
                        高
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        │     RAG        │      RAG       │
        │  (实时新闻)     │   (企业知识库)  │
        │                │                │
   低 ──┼────────────────┼────────────────┼── 高
        │                │                │    领域
        │  Fine-tuning   │   RAG + SFT    │    特异性
        │  (通用任务)     │   (专业领域)    │
        │                │                │
        └────────────────┼────────────────┘
                         │
                        低
```

---

## 📚 章节导航

| 章节 | 内容概要 | 难度 |
|------|----------|------|
| [RAG 解决的问题](./rag-basics.md) | 长尾知识、私有数据、时效性、可解释性四大痛点 | ⭐⭐ |
| [检索器模块](./retriever.md) | 向量数据库、Embedding 模型、检索策略详解 | ⭐⭐⭐ |
| [TF-IDF 与 BM25](./tfidf-bm25.md) | 经典文本检索算法，信息检索的基础 | ⭐⭐⭐ |
| [重排序技术](./reranking.md) | Cross-Encoder、ColBERT、混合检索优化 | ⭐⭐⭐ |
| [生成器模块](./generator.md) | Prompt 设计、上下文融合、回答生成 | ⭐⭐⭐ |
| [高级 RAG 技术](./advanced-rag.md) | 混合检索、重排序、多跳检索、GraphRAG | ⭐⭐⭐⭐ |
| [RAG vs SFT 对比](./rag-vs-sft.md) | 技术选型、成本分析、最佳实践 | ⭐⭐⭐ |

---

## 🔧 技术栈概览

### 核心组件

```
RAG 技术栈
├── 检索器 (Retriever)
│   ├── Embedding 模型
│   │   ├── OpenAI text-embedding-ada-002
│   │   ├── BGE / M3E (中文)
│   │   └── E5 / Instructor
│   ├── 向量数据库
│   │   ├── Milvus (生产级)
│   │   ├── Pinecone (云服务)
│   │   ├── Chroma (轻量级)
│   │   └── FAISS (Facebook)
│   └── 检索策略
│       ├── 稠密检索 (Dense)
│       ├── 稀疏检索 (Sparse/BM25)
│       └── 混合检索 (Hybrid)
│
├── 生成器 (Generator)
│   ├── LLM 选择
│   │   ├── GPT-4 / GPT-3.5
│   │   ├── Claude
│   │   └── 开源模型 (LLaMA, Qwen)
│   └── Prompt 工程
│       ├── 上下文组织
│       ├── 引用格式
│       └── 约束生成
│
└── 编排框架
    ├── LangChain
    ├── LlamaIndex
    └── Haystack
```

### 快速开始示例

```python
# 使用 LangChain 构建简单 RAG
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# 1. 加载文档
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 2. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 3. 创建 RAG 链
llm = ChatOpenAI(model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 4. 提问
response = qa_chain.run("什么是 RAG？")
print(response)
```

---

## 🗺️ 学习路径

### 入门路径（1-2 周）

```
Day 1-3: 理解 RAG 基本概念
├── 阅读 [RAG 解决的问题](./rag-basics.md)
├── 理解为什么需要外部知识增强
└── 了解 RAG 与 Fine-tuning 的区别

Day 4-7: 构建第一个 RAG 应用
├── 学习 [检索器模块](./retriever.md)
├── 使用 LangChain 构建简单 RAG
└── 实践：搭建文档问答系统

Day 8-14: 深入核心组件
├── 学习 [生成器模块](./generator.md)
├── 理解 Prompt 设计原则
└── 优化检索质量
```

### 进阶路径（2-4 周）

```
Week 1: 高级检索技术
├── 学习 [高级 RAG 技术](./advanced-rag.md)
├── 实现混合检索
└── 添加重排序模块

Week 2: 生产化部署
├── 向量数据库选型与优化
├── 检索性能调优
└── 监控与评估

Week 3-4: 项目实战
├── 企业知识库问答
├── 多模态 RAG
└── GraphRAG 探索
```

---

## 📖 前置知识

### 必备基础

| 知识领域 | 具体要求 | 推荐资源 |
|----------|----------|----------|
| **Python 编程** | 熟练使用 Python | [Python 基础](../language/python/index.md) |
| **LLM 基础** | 了解 Transformer、Embedding | [LLM 导览](../llm/index.md) |
| **向量运算** | 理解向量、相似度计算 | 线性代数基础 |

### 推荐先学内容

```
1. 大语言模型基础
   ├── Transformer 架构
   ├── Embedding 概念
   └── Prompt Engineering
   ↓
2. 信息检索基础
   ├── 文本相似度
   ├── TF-IDF / BM25
   └── 向量检索
   ↓
3. RAG 核心技术（本章）
```

---

## 🔗 相关资源

### 经典论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | RAG 开山之作 |
| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | 2020 | DPR 检索方法 |
| [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) | 2020 | 端到端检索增强预训练 |

### 开源项目

| 项目 | 描述 | 链接 |
|------|------|------|
| **LangChain** | LLM 应用开发框架 | [GitHub](https://github.com/langchain-ai/langchain) |
| **LlamaIndex** | 数据连接框架，专注 RAG | [GitHub](https://github.com/run-llama/llama_index) |
| **Haystack** | NLP 管道框架 | [GitHub](https://github.com/deepset-ai/haystack) |
| **RAGAS** | RAG 评估框架 | [GitHub](https://github.com/explodinggradients/ragas) |

### 向量数据库对比

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Milvus** | 开源、高性能、分布式 | 生产级大规模应用 |
| **Pinecone** | 全托管云服务 | 快速上线、免运维 |
| **Chroma** | 轻量级、易上手 | 开发测试、小规模应用 |
| **FAISS** | Meta 开源、纯向量搜索 | 本地部署、离线处理 |
| **Weaviate** | 语义搜索、GraphQL API | 语义理解场景 |

---

## 💡 学习建议

### 实践驱动

RAG 是一项实践性很强的技术，建议边学边做：

1. **从简单开始**：先用 LangChain 搭建一个文档问答系统
2. **逐步深入**：理解每个组件的作用和优化方法
3. **关注评估**：学会评估 RAG 系统的质量

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| "RAG 能解决所有问题" | RAG 适合知识密集型任务，不适用所有场景 |
| "向量检索就够了" | 混合检索（向量 + 关键词）往往效果更好 |
| "检索越多越好" | 过多无关上下文会干扰 LLM 生成 |
| "Embedding 模型越强越好" | 需要与业务场景匹配，中文场景选中文模型 |

---

*开始你的 RAG 学习之旅！建议从 [RAG 解决的问题](./rag-basics.md) 开始，理解为什么需要这项技术。*
