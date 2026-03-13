# LangChain 学习指南

LangChain 是一个强大的框架，专为开发大语言模型（LLM）驱动的应用程序而设计。它提供了一套完整的工具链，帮助开发者快速构建从原型到生产的 AI 应用，无论是简单的问答系统还是复杂的多智能体协作系统。

---

## 🎯 为什么选择 LangChain

### 核心优势

| 特性 | 描述 |
|------|------|
| **模块化设计** | 组件可自由组合，灵活构建各种应用 |
| **模型无关** | 支持 OpenAI、Claude、LLaMA 等多种模型 |
| **链式调用** | LCEL 语法让复杂流程简洁优雅 |
| **智能体框架** | 内置 Agent 能力，支持自主决策和工具调用 |
| **丰富生态** | 大量集成工具、模板和社区资源 |

### 适用场景

- 🤖 **对话系统**：智能客服、个人助理、聊天机器人
- 📚 **RAG 应用**：知识库问答、文档分析、智能搜索
- 🔧 **Agent 应用**：自动化任务、数据分析、代码生成
- 🔄 **工作流自动化**：多步骤任务编排、数据处理管道
- 🎯 **垂直领域应用**：法律助手、医疗问答、金融分析

---

## 📚 章节导航

### 核心内容

| 章节 | 内容概要 | 难度 |
|------|----------|------|
| [基础概念](./langchain-basics.md) | 安装配置、核心组件、快速上手 | ⭐ |
| [链与 LCEL](./langchain-chains.md) | LCEL 语法、链式调用、Runnable 接口 | ⭐⭐ |
| [智能体开发](./langchain-agents.md) | Agent 类型、ReAct 模式、自主决策 | ⭐⭐⭐ |
| [工具调用](./langchain-tools.md) | 自定义工具、内置工具、工具集成 | ⭐⭐ |
| [记忆系统](./langchain-memory.md) | 对话记忆、历史管理、状态持久化 | ⭐⭐ |

---

## 🗺️ 学习路径建议

### 路径一：快速上手（推荐初学者）

适合刚接触 LLM 应用开发的开发者。

```
Day 1: 环境搭建与基础
├── 安装 LangChain 和相关依赖
├── 配置 API Key
├── 理解 Prompt Template
└── 完成第一次模型调用

Day 2-3: LCEL 与链
├── 学习 LCEL 语法
├── 构建简单的处理链
└── 理解 Runnable 接口

Day 4-5: 实践项目
├── 构建一个简单的问答系统
├── 添加对话记忆功能
└── 尝试不同的 LLM 后端
```

### 路径二：RAG 应用开发

适合需要构建知识库问答系统的开发者。

```
重点掌握:
├── Document Loader 使用
├── Text Splitter 文本分割
├── Embedding 模型选择
├── Vector Store 向量存储
├── Retriever 检索器
└── RAG 链构建
```

### 路径三：Agent 开发

适合需要构建智能自主系统的开发者。

```
重点掌握:
├── Agent 类型与选择
├── ReAct 推理模式
├── 工具定义与注册
├── 多工具协作
└── 错误处理与重试机制
```

---

## 🛠️ 快速开始

### 安装

```bash
# 基础安装
pip install langchain

# 安装社区集成
pip install langchain-community

# 安装核心依赖
pip install langchain-core

# 安装 OpenAI 支持
pip install langchain-openai

# 完整安装
pip install langchain langchain-community langchain-core langchain-openai
```

### Hello World

创建一个最简单的 LangChain 应用：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 定义模型
model = ChatOpenAI(model="gpt-4o-mini")

# 2. 定义提示词模板
prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")

# 3. 定义输出解析器
parser = StrOutputParser()

# 4. 使用 LCEL 构建链
chain = prompt | model | parser

# 5. 调用链
result = chain.invoke({"topic": "程序员"})
print(result)
```

运行后会输出一个关于程序员的笑话。

---

## 🧩 核心架构概览

LangChain 的核心架构可以分为以下几个层次：

```
┌─────────────────────────────────────────────┐
│                  应用层                      │
│    (Chains, Agents, RAG Applications)       │
├─────────────────────────────────────────────┤
│                  编排层                      │
│         (LCEL, Runnable Interface)          │
├─────────────────────────────────────────────┤
│                  组件层                      │
│  (Prompts, Memory, Tools, Retrievers)       │
├─────────────────────────────────────────────┤
│                  模型层                      │
│    (Chat Models, LLMs, Embeddings)          │
└─────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 | 关键类/函数 |
|------|------|-------------|
| **Model I/O** | 模型输入输出管理 | `ChatOpenAI`, `PromptTemplate` |
| **Memory** | 对话历史管理 | `ConversationBufferMemory` |
| **Chains** | 多步骤任务编排 | `LLMChain`, LCEL |
| **Agents** | 自主决策执行 | `AgentExecutor` |
| **Tools** | 外部工具集成 | `Tool`, `@tool` |
| **Retrieval** | 知识检索 | `VectorStore`, `Retriever` |

---

## 📋 前置知识要求

### 必备基础

| 知识领域 | 具体要求 | 重要程度 |
|----------|----------|----------|
| **Python 编程** | 函数、类、装饰器、异步编程 | ⭐⭐⭐⭐⭐ |
| **API 调用** | HTTP 请求、JSON 处理 | ⭐⭐⭐⭐ |
| **LLM 基础** | Prompt 设计、Token 概念 | ⭐⭐⭐ |
| **向量数据库** | 基本概念（RAG 必需） | ⭐⭐⭐ |

### 推荐搭配

LangChain 通常与以下技术栈配合使用：

```
LangChain 应用
    │
    ├── 模型后端
    │   ├── OpenAI (GPT-4, GPT-4o)
    │   ├── Anthropic (Claude)
    │   ├── 本地模型 (Ollama, vLLM)
    │   └── 开源模型 (LLaMA, Qwen)
    │
    ├── 向量数据库
    │   ├── Chroma
    │   ├── FAISS
    │   ├── Pinecone
    │   └── Milvus
    │
    └── UI 层
        ├── Gradio
        ├── Streamlit
        └── FastAPI
```

---

## 🌟 版本说明

LangChain 在 2024 年进行了重大重构，本教程基于最新版本：

| 版本 | 说明 |
|------|------|
| `langchain-core` | 核心接口定义，包含 LCEL |
| `langchain-community` | 社区集成（向量库、工具等） |
| `langchain-openai` | OpenAI 模型集成 |
| `langchain-anthropic` | Anthropic 模型集成 |

> ⚠️ **注意**: 旧版 API（如 `LLMChain`）已被标记为废弃，建议使用 LCEL 语法。

---

## 💡 设计理念

LangChain 的设计遵循以下核心理念：

### 1. 组合优于继承

通过 LCEL 的管道操作符 `|`，可以灵活组合各种组件：

```python
chain = prompt | model | parser
```

### 2. 模型无关设计

同一套代码可以切换不同的模型后端：

```python
# OpenAI
model = ChatOpenAI(model="gpt-4o")

# 切换到 Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 切换到本地模型
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3")
```

### 3. 声明式编程

用声明式的方式描述数据流，而非命令式的控制流：

```python
# 声明式：描述"做什么"
chain = (
    {"topic": RunnablePassthrough()} 
    | prompt 
    | model 
    | parser
)

# 而非命令式：描述"怎么做"
def process(topic):
    formatted_prompt = prompt.format(topic=topic)
    response = model.invoke(formatted_prompt)
    return parser.parse(response)
```

---

## 📖 学习资源推荐

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 官方文档 | [python.langchain.com](https://python.langchain.com/) | 最权威的参考资料 |
| API 参考 | [api.python.langchain.com](https://api.python.langchain.com/) | 详细的 API 文档 |
| GitHub | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 源码和 Issues |
| LangSmith | [smith.langchain.com](https://smith.langchain.com/) | 调试和监控平台 |

### 推荐教程

| 教程 | 平台 | 特点 |
|------|------|------|
| [LangChain 官方教程](https://python.langchain.com/docs/tutorials/) | 官方 | 系统全面，实践导向 |
| [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook) | GitHub | 实战案例集合 |

---

## 🔗 相关章节

- [Transformers](../transformers/index.md) - Hugging Face 模型库使用
- [OpenAI SDK](../openai/index.md) - OpenAI API 调用
- [Gradio](../gradio/index.md) - 构建 Web 界面
- [vLLM](../vllm/index.md) - 高效推理引擎
- [LlamaIndex](../llamaindex/index.md) - 数据框架

---

*准备好开始构建你的第一个 LLM 应用了吗？从 [基础概念](./langchain-basics.md) 开始吧！🚀*
