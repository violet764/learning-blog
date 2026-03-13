# LangChain 基础概念

LangChain 的核心是一套模块化的组件体系，理解这些基础概念是构建复杂 LLM 应用的基石。本章将介绍 LangChain 的安装配置、核心组件以及基本使用方法。

---

## 📦 安装与配置

### 安装方式

LangChain 采用模块化设计，可以根据需要选择性安装：

```bash
# 核心包（必须）
pip install langchain-core

# 主包（包含常用功能）
pip install langchain

# 社区集成（向量库、工具等）
pip install langchain-community

# 模型提供商（按需安装）
pip install langchain-openai      # OpenAI
pip install langchain-anthropic   # Anthropic
pip install langchain-google-genai  # Google
pip install langchain-ollama      # Ollama 本地模型

# 向量数据库（按需安装）
pip install langchain-chroma      # Chroma
pip install langchain-pinecone    # Pinecone
```

### 环境配置

通过环境变量配置 API Key：

```python
# 方式一：使用 python-dotenv
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载

# 方式二：直接设置环境变量
import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["ANTHROPIC_API_KEY"] = "sk-xxx"

# 方式三：在代码中直接传入
from langchain_openai import ChatOpenAI
model = ChatOpenAI(api_key="sk-xxx")
```

推荐使用 `.env` 文件管理敏感信息：

```bash
# .env 文件内容
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=xxx  # LangSmith 追踪
```

---

## 🧩 核心组件概览

LangChain 的架构围绕六个核心组件展开：

```
┌─────────────────────────────────────────────────────────┐
│                     LangChain 架构                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │  Model  │  │ Prompt  │  │ Output  │  │ Memory  │    │
│  │   I/O   │  │Template │  │ Parser  │  │         │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       └────────────┼────────────┼────────────┘          │
│                    │            │                       │
│              ┌─────┴────────────┴─────┐                 │
│              │         LCEL           │                 │
│              │   (编排与组合层)        │                 │
│              └───────────┬────────────┘                 │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │        Agents         │                  │
│              │    (智能体与工具)      │                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

| 组件 | 职责 | 关键类 |
|------|------|--------|
| **Model I/O** | 与 LLM 交互 | `ChatOpenAI`, `ChatAnthropic` |
| **Prompt Template** | 提示词模板化 | `ChatPromptTemplate`, `PromptTemplate` |
| **Output Parser** | 解析模型输出 | `StrOutputParser`, `JsonOutputParser` |
| **Memory** | 管理对话历史 | `ConversationBufferMemory` |
| **LCEL** | 组件编排组合 | `Runnable` 接口 |
| **Agents** | 自主决策执行 | `create_tool_calling_agent` |

---

## 🤖 Model I/O - 模型交互

### Chat Models vs LLMs

LangChain 区分两种模型类型：

```python
# Chat Models：对话模型（推荐）
# 输入：消息列表，输出：消息
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(model="gpt-4o")

# LLMs：文本补全模型（旧版）
# 输入：字符串，输出：字符串
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct")
```

> 💡 **推荐使用 Chat Models**：现代 LLM 主要是对话模型，功能更强大，支持多模态。

### 初始化模型

```python
from langchain_openai import ChatOpenAI

# 基础初始化
model = ChatOpenAI(model="gpt-4o-mini")

# 完整配置
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,           # 创造性程度 (0-2)
    max_tokens=1024,           # 最大输出长度
    timeout=30,                # 请求超时
    max_retries=2,             # 重试次数
    api_key="sk-xxx",          # API Key
    base_url="https://xxx",    # 自定义端点
)
```

### 消息类型

Chat Models 使用消息对象进行交互：

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="你是一个有帮助的AI助手。"),
    HumanMessage(content="什么是量子计算？"),
    AIMessage(content="量子计算是利用量子力学原理..."),
    HumanMessage(content="能举个例子吗？")
]

response = model.invoke(messages)
print(response.content)
```

### 模型切换

LangChain 的模型无关设计让切换模型非常简单：

```python
# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-pro")

# 本地模型 (Ollama)
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.2")

# 自定义端点 (如 vLLM)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="my-model",
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)
```

---

## 📝 Prompt Template - 提示词模板

提示词模板是将变量注入到提示词中的工具，让提示词可复用、可参数化。

### ChatPromptTemplate

最常用的提示词模板，支持多消息类型：

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式一：from_template（单条人类消息）
prompt = ChatPromptTemplate.from_template(
    "给我讲一个关于{topic}的{style}笑话"
)

# 方式二：from_messages（多条消息）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用{style}风格回答问题。"),
    ("human", "{question}")
])

# 格式化提示词
formatted = prompt.format(
    role="资深程序员",
    style="幽默",
    question="什么是Bug？"
)
print(formatted)
```

### 消息占位符

当需要动态插入消息列表时，使用 `MessagesPlaceholder`：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 使用时传入历史消息
formatted = prompt.format(
    chat_history=[
        ("human", "你好"),
        ("ai", "你好！有什么我可以帮你的吗？")
    ],
    input="今天的天气怎么样？"
)
```

### PromptTemplate（纯文本）

用于文本补全模型的模板：

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """请将以下文本翻译成{language}：

文本：{text}

翻译："""
)

formatted = template.format(
    language="日语",
    text="你好，世界"
)
```

### Few-shot 示例

通过少量示例引导模型输出：

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# 定义示例
examples = [
    {"input": "开心", "output": "😊 快乐"},
    {"input": "伤心", "output": "😢 悲伤"},
    {"input": "生气", "output": "😠 愤怒"},
]

# 示例模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Few-shot 提示
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 最终提示
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "将情绪词转换为emoji和同义词"),
    few_shot_prompt,
    ("human", "{input}")
])
```

---

## 🔍 Output Parser - 输出解析

输出解析器将模型输出的字符串转换为结构化数据。

### StrOutputParser

最简单的解析器，返回原始字符串：

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# 直接返回模型的 content
result = parser.parse(ai_message)  # 返回字符串
```

### JsonOutputParser

解析 JSON 格式输出：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 定义数据结构
class Person(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    hobbies: list[str] = Field(description="兴趣爱好")

# 创建解析器
parser = JsonOutputParser(pydantic_object=Person)

# 获取格式化指令，添加到提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "提取人物信息。\n{format_instructions}"),
    ("human", "{text}")
])

# 使用
chain = prompt | model | parser
result = chain.invoke({
    "text": "张三今年25岁，喜欢编程和篮球",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# {'name': '张三', 'age': 25, 'hobbies': ['编程', '篮球']}
```

### PydanticOutputParser

使用 Pydantic 进行严格类型验证：

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, validator

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('价格必须为正数')
        return v

parser = PydanticOutputParser(pydantic_object=Product)

# 解析时会进行验证
result = parser.parse('{"name": "手机", "price": 999.99, "in_stock": true}')
# 返回 Product 对象，可直接访问属性
print(result.name)  # 手机
```

### CommaSeparatedListOutputParser

解析逗号分隔的列表：

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template(
    "列出5个{topic}相关的关键词。\n{format_instructions}"
)

chain = prompt | model | parser
result = chain.invoke({
    "topic": "Python",
    "format_instructions": parser.get_format_instructions()
})

print(result)  # ['列表', '推导式', '装饰器', '生成器', '迭代器']
```

---

## 🔄 基本调用模式

### 直接调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 同步调用
response = model.invoke([HumanMessage(content="你好")])
print(response.content)

# 批量调用
responses = model.batch([
    [HumanMessage(content="什么是AI？")],
    [HumanMessage(content="什么是ML？")]
])

# 流式输出
for chunk in model.stream([HumanMessage(content="讲个故事")]):
    print(chunk.content, end="", flush=True)
```

### 异步调用

```python
import asyncio

async def main():
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # 异步调用
    response = await model.ainvoke([HumanMessage(content="你好")])
    
    # 异步流式
    async for chunk in model.astream([HumanMessage(content="讲个故事")]):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

---

## 🎯 实践示例：构建简单问答系统

下面是一个完整的示例，展示如何组合上述组件：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from typing import List

class SimpleQA:
    """简单的问答系统"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # 初始化模型
        self.model = ChatOpenAI(model=model_name, temperature=0.7)
        
        # 定义提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的AI助手，请用中文回答问题。"),
            ("human", "{question}")
        ])
        
        # 构建链
        self.chain = self.prompt | self.model | StrOutputParser()
        
        # 对话历史
        self.history: List[tuple] = []
    
    def ask(self, question: str) -> str:
        """提问并获取回答"""
        response = self.chain.invoke({"question": question})
        self.history.append((question, response))
        return response
    
    def stream_ask(self, question: str):
        """流式输出回答"""
        for chunk in self.chain.stream({"question": question}):
            yield chunk
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []

# 使用示例
if __name__ == "__main__":
    qa = SimpleQA()
    
    # 同步问答
    answer = qa.ask("什么是机器学习？")
    print(f"回答: {answer}\n")
    
    # 流式问答
    print("流式回答: ", end="")
    for chunk in qa.stream_ask("用一句话解释深度学习"):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 查看历史
    print(f"对话历史: {len(qa.history)} 轮")
```

---

## 📋 组件速查表

### 模型初始化

```python
# OpenAI
ChatOpenAI(model="gpt-4o", temperature=0.7)

# Anthropic
ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 本地模型
ChatOllama(model="llama3.2")
```

### 提示词模板

```python
# 单消息模板
ChatPromptTemplate.from_template("问题：{question}")

# 多消息模板
ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    ("human", "{question}")
])

# 带历史记录
ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{question}")
])
```

### 输出解析器

| 解析器 | 用途 | 输出类型 |
|--------|------|----------|
| `StrOutputParser` | 原始字符串 | `str` |
| `JsonOutputParser` | JSON 对象 | `dict` |
| `PydanticOutputParser` | 类型验证 | Pydantic 模型 |
| `CommaSeparatedListOutputParser` | 列表 | `list[str]` |

---

## 🔗 相关章节

- [链与 LCEL](./langchain-chains.md) - 学习如何使用 LCEL 组合组件
- [智能体开发](./langchain-agents.md) - 构建能自主决策的 Agent
- [工具调用](./langchain-tools.md) - 让模型调用外部工具
- [记忆系统](./langchain-memory.md) - 管理对话上下文
