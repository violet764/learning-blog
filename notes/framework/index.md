# AI 开发框架学习指南

本章介绍 AI 应用开发中常用的框架和工具，帮助你快速构建、部署和管理 AI 应用。从模型推理到 Web 界面，从 LLM 应用开发到高效部署，涵盖 AI 工程化的完整技术栈。

---

## 🎯 框架概览

### 技术栈全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI 应用开发技术栈                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      应用层                                  │   │
│  │   Web 界面（Gradio） · 聊天应用 · RAG 系统 · AI Agent       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     应用框架层                               │   │
│  │   LangChain · LlamaIndex · OpenAI SDK                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     模型层                                   │   │
│  │   Transformers · vLLM · 模型推理优化                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 框架对比

| 框架 | 类型 | 核心功能 | 适用场景 |
|------|------|----------|----------|
| **Gradio** | Web 界面 | 快速构建 AI Demo | 模型展示、原型验证 |
| **Transformers** | 模型库 | 预训练模型加载与推理 | 模型使用、微调 |
| **OpenAI SDK** | API 客户端 | 调用 OpenAI 服务 | GPT 应用开发 |
| **LangChain** | 应用框架 | LLM 应用编排 | Agent、RAG、复杂流程 |
| **LlamaIndex** | 数据框架 | 知识库构建与检索 | RAG 应用、文档问答 |
| **vLLM** | 推理引擎 | 高效 LLM 推理 | 生产部署、高并发 |

---

## 📚 模块导航

### Gradio - Web 界面构建

Gradio 让你用几行 Python 代码就能创建漂亮的 Web 界面，快速展示 AI 模型的能力。

::: tip 学习入口
**[→ 进入 Gradio 学习指南](./gradio/index.md)**
:::

**核心内容**：

| 章节 | 内容 | 难度 |
|------|------|------|
| [基础组件与布局](./gradio/gradio-basics.md) | 输入输出组件、布局容器、样式定制 | ⭐⭐ |
| [Interface 快速构建](./gradio/gradio-interface.md) | Interface API、事件处理、多输入输出 | ⭐⭐ |
| [Blocks 灵活布局](./gradio/gradio-blocks.md) | Blocks API、复杂布局、状态管理 | ⭐⭐⭐ |
| [高级特性与部署](./gradio/gradio-advanced.md) | 流式输出、HF Spaces 部署 | ⭐⭐⭐ |

---

### Transformers - Hugging Face 模型库

Hugging Face Transformers 是最流行的预训练模型库，支持加载、使用和微调数千种模型。

**核心能力**：
- 🤖 支持 200+ 预训练模型
- 🔧 统一的 API 接口
- 📦 Pipeline 快速推理
- 🎯 微调与训练支持

---

### OpenAI SDK - API 调用

OpenAI 官方 SDK，用于调用 GPT-4、DALL-E、Whisper 等服务。

**核心能力**：
- 💬 Chat Completions API
- 🎨 图像生成 API
- 🎵 语音识别与合成
- 🔧 Function Calling

---

### LangChain - LLM 应用框架

LangChain 是开发 LLM 应用的首选框架，提供了构建复杂 AI 应用所需的各种组件。

**核心能力**：
- 🔗 Chain 编排
- 🛠️ Tool 工具调用
- 📝 Prompt 模板管理
- 🧠 Memory 记忆系统
- 🤖 Agent 智能体

---

### LlamaIndex - 知识库框架

LlamaIndex 专注于数据连接和检索增强，是构建 RAG 应用的利器。

**核心能力**：
- 📚 文档加载与解析
- 🔍 向量索引与检索
- 🗂️ 知识图谱构建
- 🔗 多数据源连接

---

### vLLM - 高效推理引擎

vLLM 是高性能 LLM 推理引擎，专为生产环境设计。

**核心能力**：
- ⚡ PagedAttention 内存优化
- 🚀 连续批处理
- 📈 高吞吐量
- 🔌 兼容 OpenAI API

---

## 🗺️ 学习路径建议

### 路径一：快速上手（初学者）

```
Week 1: Gradio 基础
├── 创建第一个 Web 界面
├── 掌握常用组件
└── 部署到 Hugging Face Spaces

Week 2: Transformers 入门
├── 使用 Pipeline 快速推理
├── 加载预训练模型
└── 处理不同类型的输入输出

Week 3: OpenAI SDK
├── 调用 Chat API
├── 实现简单对话应用
└── 理解 Function Calling
```

### 路径二：应用开发（工程师）

```
重点掌握:
├── LangChain 核心组件
├── RAG 系统构建（LlamaIndex）
├── Agent 开发
└── 生产部署（vLLM + Gradio）
```

### 路径三：模型部署（运维）

```
重点关注:
├── vLLM 推理优化
├── Docker 容器化
├── 高可用架构
└── 监控与日志
```

---

## 🛠️ 环境准备

### 基础安装

```bash
# 创建环境
conda create -n ai-app python=3.10
conda activate ai-app

# 安装核心框架
pip install gradio                  # Web 界面
pip install transformers torch      # 模型推理
pip install openai                  # OpenAI SDK
pip install langchain langchain-openai  # LangChain
pip install llama-index             # LlamaIndex
pip install vllm                    # vLLM（需要 GPU）
```

### 可选依赖

```bash
# 图像处理
pip install pillow diffusers

# 向量数据库
pip install chromadb faiss-cpu

# 文档处理
pip install pypdf unstructured
```

---

## 📖 学习资源

### 官方文档

| 框架 | 文档地址 |
|------|----------|
| Gradio | [gradio.app/docs](https://www.gradio.app/docs) |
| Transformers | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| OpenAI | [platform.openai.com/docs](https://platform.openai.com/docs) |
| LangChain | [python.langchain.com/docs](https://python.langchain.com/docs) |
| LlamaIndex | [docs.llamaindex.ai](https://docs.llamaindex.ai) |
| vLLM | [vllm.readthedocs.io](https://vllm.readthedocs.io) |

---

## 🔗 相关章节

- [AI 大模型](../ai-model/index.md) - 理解模型原理
- [深度学习](../deep-learning/index.md) - PyTorch 基础
- [Python 编程](../language/python/index.md) - Python 语法

---

*选择你感兴趣的框架，开始构建你的 AI 应用吧！建议从 [Gradio](./gradio/index.md) 开始，快速看到你的模型在 Web 上运行。🚀*
