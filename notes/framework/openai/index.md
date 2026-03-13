# OpenAI SDK 学习指南

OpenAI Python SDK 是与 OpenAI API 交互的官方 Python 库，提供了一套简洁优雅的接口来调用 GPT 系列、DALL-E、Whisper 等模型。v1.x 版本进行了全面重构，采用异步优先设计，性能更优、类型提示更完善。

---

## 🎯 为什么学习 OpenAI SDK

### 核心优势

| 特性 | 描述 |
|------|------|
| **官方支持** | OpenAI 官方维护，API 更新及时同步 |
| **类型完善** | 完整的类型提示，IDE 自动补全友好 |
| **异步优先** | 原生支持 async/await，高并发场景性能优异 |
| **流式处理** | 优雅的流式响应 API，提升用户体验 |
| **功能全面** | 覆盖 Chat、Embeddings、Images、Audio 等全部 API |

### 适用场景

- 💬 **对话系统**：智能客服、聊天机器人、虚拟助手
- 📝 **内容生成**：文案写作、代码生成、摘要提取
- 🔍 **语义搜索**：文本嵌入、相似度计算、向量检索
- 🛠️ **工具调用**：Function Calling、结构化输出
- 🤖 **AI Agent**：构建具有工具使用能力的智能代理

---

## 📚 章节导航

### 核心内容

| 章节 | 内容概要 | 难度 |
|------|----------|------|
| [基础 API 调用](./openai-basics.md) | 安装配置、认证方式、错误处理 | ⭐ |
| [Chat Completions](./openai-chat.md) | 参数详解、多轮对话、系统提示 | ⭐⭐ |
| [Embeddings API](./openai-embeddings.md) | 文本嵌入、相似度计算、向量存储 | ⭐⭐ |
| [Function Calling](./openai-functions.md) | 函数定义、参数解析、工具调用 | ⭐⭐⭐ |
| [Assistants API](./openai-assistants.md) | 助手创建、线程管理、文件处理 | ⭐⭐⭐ |
| [流式响应处理](./openai-streaming.md) | 异步流式、回调处理、最佳实践 | ⭐⭐ |

---

## 🗺️ 学习路径建议

### 路径一：快速上手（推荐初学者）

适合刚开始接触 OpenAI API 的开发者。

```
Day 1: 环境搭建与基础调用
├── 安装 openai 库
├── 获取 API Key 并配置
├── 发送第一个 Chat 请求
└── 理解请求与响应结构

Day 2-3: Chat API 深入
├── 理解 messages 结构
├── 掌握 temperature、max_tokens 等参数
├── 实现多轮对话
└── 使用系统提示词

Day 4-5: 实践项目
├── 构建一个简单的聊天机器人
├── 添加上下文记忆
└── 优化响应质量
```

### 路径二：高级功能

适合需要使用高级功能的开发者。

```
重点掌握:
├── Function Calling 工具调用
├── Embeddings 向量嵌入
├── 流式响应处理
└── 异步编程模式
```

### 路径三：企业应用

适合构建生产级应用的开发者。

```
重点掌握:
├── Assistants API 状态管理
├── 文件上传与处理
├── 错误处理与重试策略
├── 成本控制与监控
└── 安全最佳实践
```

---

## 🛠️ 快速开始

### 安装

```bash
# 安装最新版本
pip install openai

# 安装特定版本
pip install openai==1.12.0

# 安装开发版本
pip install git+https://github.com/openai/openai-python.git
```

### Hello World：第一次调用

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key="your-api-key")

# 发送请求
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
)

# 获取回复
print(response.choices[0].message.content)
```

### 异步调用

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="your-api-key")
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## 🧩 核心架构概览

OpenAI SDK v1.x 的架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                   OpenAI SDK 架构                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │              Client 客户端层                      │    │
│  │     OpenAI / AsyncOpenAI / AzureOpenAI          │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │               API 资源层                         │    │
│  │  chat / embeddings / images / audio / files     │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │              响应处理层                          │    │
│  │   Pydantic Models / Stream / Pagination         │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │              底层传输层                          │    │
│  │      httpx (同步) / httpx (异步)                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 | 关键类 |
|------|------|--------|
| **Client** | API 客户端 | `OpenAI`, `AsyncOpenAI` |
| **Chat** | 对话补全 | `client.chat.completions` |
| **Embeddings** | 文本嵌入 | `client.embeddings` |
| **Images** | 图像生成 | `client.images` |
| **Audio** | 语音处理 | `client.audio` |
| **Assistants** | 助手 API | `client.beta.assistants` |

---

## 📋 可用模型

### GPT 系列

| 模型 | 上下文长度 | 特点 | 适用场景 |
|------|-----------|------|----------|
| `gpt-4o` | 128K | 最新旗舰，多模态 | 复杂任务、视觉理解 |
| `gpt-4o-mini` | 128K | 性价比高 | 日常对话、简单任务 |
| `gpt-4-turbo` | 128K | 性能强劲 | 高质量输出 |
| `gpt-3.5-turbo` | 16K | 成本低 | 简单任务、批量处理 |

### Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| `text-embedding-3-small` | 512/1536 | 成本低，速度快 |
| `text-embedding-3-large` | 256/1024/3072 | 精度高，存储大 |
| `text-embedding-ada-002` | 1536 | 旧版本，不推荐 |

### 其他模型

| 类型 | 模型 | 用途 |
|------|------|------|
| 图像生成 | `dall-e-3` | 文生图 |
| 语音识别 | `whisper-1` | 语音转文字 |
| 语音合成 | `tts-1` | 文字转语音 |
| 模态转换 | `gpt-4o-audio-preview` | 音频理解 |

---

## 🔑 API Key 管理

### 安全配置方式

```python
# 方式一：环境变量（推荐）
import os
from openai import OpenAI

# 设置环境变量
# export OPENAI_API_KEY='your-api-key'

client = OpenAI()  # 自动读取 OPENAI_API_KEY

# 方式二：代码中设置（不推荐用于生产）
client = OpenAI(api_key="your-api-key")

# 方式三：从文件读取
with open(".openai_key") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)
```

### 使用 .env 文件

```bash
# .env 文件
OPENAI_API_KEY=sk-xxxx
OPENAI_ORG_ID=org-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

```python
# Python 代码
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 加载 .env 文件
client = OpenAI()
```

---

## 💡 v1.x vs v0.x 迁移

如果你使用过旧版本，以下是主要变化：

### 客户端初始化

```python
# v0.x (旧版)
import openai
openai.api_key = "your-key"
response = openai.ChatCompletion.create(...)

# v1.x (新版)
from openai import OpenAI
client = OpenAI(api_key="your-key")
response = client.chat.completions.create(...)
```

### 响应访问

```python
# v0.x
content = response['choices'][0]['message']['content']

# v1.x (属性访问)
content = response.choices[0].message.content
```

### 流式处理

```python
# v0.x
for chunk in response:
    delta = chunk['choices'][0]['delta']
    ...

# v1.x
for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="")
```

---

## 📖 学习资源推荐

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| API 文档 | [platform.openai.com/docs](https://platform.openai.com/docs) | 最权威的参考资料 |
| 官方示例 | [github.com/openai/openai-cookbook](https://github.com/openai/openai-cookbook) | 丰富的代码示例 |
| API 参考 | [platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference) | 详细 API 文档 |
| 定价 | [openai.com/pricing](https://openai.com/pricing) | 模型定价信息 |

### 推荐实践

| 项目 | 说明 |
|------|------|
| 成本监控 | 使用 `response.usage` 追踪 token 消耗 |
| 错误处理 | 捕获 `openai.APIError` 等异常 |
| 重试机制 | 使用 `tenacity` 库实现自动重试 |
| 流式输出 | 长文本生成时使用流式提升体验 |

---

## 🔗 相关章节

- [LangChain](../langchain/index.md) - LLM 应用开发框架
- [Transformers](../transformers/index.md) - 本地模型部署
- [vLLM](../vllm/index.md) - 高效推理引擎
- [LlamaIndex](../llamaindex/index.md) - 数据框架
- [Gradio](../gradio/index.md) - 构建 Web 界面

---

*准备好开始使用 OpenAI API 了吗？从 [基础 API 调用](./openai-basics.md) 开始吧！🚀*
