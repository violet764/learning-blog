# 链与 LCEL

LCEL（LangChain Expression Language）是 LangChain 的核心编排语言，它提供了一种声明式的方式来组合各种组件。通过管道操作符 `|`，LCEL 让复杂的处理流程变得简洁优雅。

---

## 🌟 LCEL 核心概念

### 什么是 LCEL

LCEL 是一种声明式语言，用于将 LangChain 组件串联成处理链。它的核心思想是：

```
输入 → 组件1 → 组件2 → 组件3 → 输出
```

使用管道操作符 `|` 连接组件：

```python
chain = prompt | model | parser
```

### 为什么使用 LCEL

| 优势 | 说明 |
|------|------|
| **简洁性** | 用声明式替代命令式，代码更清晰 |
| **类型安全** | 自动验证输入输出类型 |
| **统一接口** | 所有组件都实现 Runnable 接口 |
| **功能完整** | 支持同步/异步、流式、批量处理 |
| **可追踪** | 自动集成 LangSmith 追踪 |

---

## 🔄 Runnable 接口

所有 LCEL 组件都实现 `Runnable` 接口，提供统一的方法：

### 核心方法

```python
from langchain_core.runnables import RunnableLambda

# 创建一个简单的 Runnable
def double(x: int) -> int:
    return x * 2

runnable = RunnableLambda(double)

# 1. invoke - 单次调用
result = runnable.invoke(5)  # 10

# 2. batch - 批量调用
results = runnable.batch([1, 2, 3, 4])  # [2, 4, 6, 8]

# 3. stream - 流式调用
for chunk in runnable.stream(5):
    print(chunk)

# 4. 异步版本
result = await runnable.ainvoke(5)
results = await runnable.abatch([1, 2, 3])
async for chunk in runnable.astream(5):
    print(chunk)
```

### Runnable 方法一览

| 方法 | 同步版本 | 异步版本 | 用途 |
|------|----------|----------|------|
| 单次调用 | `invoke()` | `ainvoke()` | 处理单个输入 |
| 批量调用 | `batch()` | `abatch()` | 处理多个输入 |
| 流式调用 | `stream()` | `astream()` | 流式输出结果 |
| 转换操作 | `map()` | - | 转换数据格式 |

---

## 🔗 基础链构建

### 简单链

最基础的链由三个组件组成：提示词 → 模型 → 解析器

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 定义组件
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
parser = StrOutputParser()

# 2. 构建链
chain = prompt | model | parser

# 3. 调用链
result = chain.invoke({"topic": "程序员"})
print(result)
```

### 链的执行过程

```
{"topic": "程序员"}
       ↓
    [prompt]  →  格式化提示词
       ↓
"讲一个关于程序员的笑话"
       ↓
    [model]   →  调用 LLM
       ↓
   AI Message  →  模型响应
       ↓
   [parser]   →  提取内容
       ↓
"为什么程序员喜欢用深色主题？因为Bug都藏不住了！"
```

---

## 🎛️ Runnable 原语

LCEL 提供了多种原语来组合和操作 Runnable。

### RunnablePassthrough - 数据传递

将输入原样传递或作为字典的一部分：

```python
from langchain_core.runnables import RunnablePassthrough

# 原样传递
chain = RunnablePassthrough()
chain.invoke("hello")  # "hello"

# 构建字典
chain = RunnablePassthrough.assign(
    upper=lambda x: x["input"].upper(),
    length=lambda x: len(x["input"])
)
chain.invoke({"input": "hello"})
# {"input": "hello", "upper": "HELLO", "length": 5}
```

### RunnableParallel - 并行执行

同时执行多个 Runnable，结果合并为字典：

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 定义多个处理分支
def process_text(x):
    return x["text"].upper()

def count_words(x):
    return len(x["text"].split())

chain = RunnableParallel(
    upper=RunnableLambda(process_text),
    word_count=RunnableLambda(count_words),
    original=RunnablePassthrough()
)

result = chain.invoke({"text": "hello world"})
# {"upper": "HELLO WORLD", "word_count": 2, "original": {"text": "hello world"}}
```

### RunnableLambda - 自定义函数

将普通 Python 函数包装为 Runnable：

```python
from langchain_core.runnables import RunnableLambda

def format_output(text: str) -> str:
    return f"📋 结果：{text}"

chain = prompt | model | parser | RunnableLambda(format_output)
```

### RunnableBranch - 条件分支

根据条件选择不同的处理路径：

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

def is_short(text):
    return len(text) < 10

# 分支处理
chain = RunnableBranch(
    (is_short, RunnableLambda(lambda x: f"短文本：{x}")),
    (lambda x: len(x) < 50, RunnableLambda(lambda x: f"中等文本：{x}")),
    RunnableLambda(lambda x: f"长文本摘要：{x[:50]}...")  # 默认分支
)

chain.invoke("你好")        # 短文本：你好
chain.invoke("这是一段中等长度的文本")  # 中等文本：...
chain.invoke("这是一段很长的文本" * 20)  # 长文本摘要：...
```

---

## 📊 实用链模式

### 模式一：多步骤处理链

将复杂任务拆分为多个步骤：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOpenAI(model="gpt-4o-mini")

# 步骤1：翻译
translate_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译成英文：\n{text}"
)
translate_chain = translate_prompt | model | StrOutputParser()

# 步骤2：总结
summarize_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容：\n{text}"
)
summarize_chain = summarize_prompt | model | StrOutputParser()

# 组合链
full_chain = (
    {"text": RunnablePassthrough()}
    | RunnablePassthrough.assign(english=translate_chain)
    | RunnablePassthrough.assign(summary=lambda x: summarize_chain.invoke(x["english"]))
)

result = full_chain.invoke("机器学习是人工智能的一个分支，它使计算机能够从数据中学习。")
# {"text": "...", "english": "...", "summary": "..."}
```

### 模式二：带预处理和后处理的链

```python
from langchain_core.runnables import RunnableLambda

# 预处理
def preprocess(text):
    return text.strip().replace("\n", " ")

# 后处理
def postprocess(text):
    return text.replace("AI", "人工智能")

chain = (
    RunnableLambda(preprocess)
    | prompt
    | model
    | parser
    | RunnableLambda(postprocess)
)
```

### 模式三：多模型对比链

使用多个模型处理同一输入：

```python
from langchain_openai import ChatOpenAI

# 不同配置的模型
fast_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
smart_model = ChatOpenAI(model="gpt-4o", temperature=0.3)

prompt = ChatPromptTemplate.from_template("分析以下文本的情感：\n{text}")

# 并行对比
compare_chain = RunnableParallel(
    fast=prompt | fast_model | StrOutputParser(),
    smart=prompt | smart_model | StrOutputParser()
)

result = compare_chain.invoke({"text": "今天天气真好，心情很愉快！"})
# {"fast": "...", "smart": "..."}
```

---

## 🔧 高级用法

### 流式输出

LCEL 链天然支持流式输出：

```python
# 同步流式
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# 异步流式
async def stream_response():
    async for chunk in chain.astream({"topic": "AI"}):
        print(chunk, end="", flush=True)
```

### 批量处理

高效处理多个输入：

```python
inputs = [
    {"topic": "程序员"},
    {"topic": "设计师"},
    {"topic": "产品经理"}
]

results = chain.batch(inputs)
# ["笑话1...", "笑话2...", "笑话3..."]
```

### 重试机制

使用 `RunnableRetry` 添加重试逻辑：

```python
from langchain_core.runnables import RunnableRetry

# 添加重试
chain_with_retry = (prompt | model | parser).with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1000
)
```

### 超时控制

```python
# 设置超时
chain_with_timeout = (prompt | model | parser).with_timeout(30)  # 30秒
```

### 配置回退

当主模型失败时使用备用模型：

```python
from langchain_anthropic import ChatAnthropic

primary_model = ChatOpenAI(model="gpt-4o")
fallback_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

chain = (
    prompt | primary_model | parser
).with_fallbacks([prompt | fallback_model | parser])
```

---

## 🎯 实践示例：智能文本处理管道

下面是一个完整的文本处理管道示例：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 定义数据结构
class TextAnalysis(BaseModel):
    summary: str = Field(description="一句话摘要")
    keywords: List[str] = Field(description="关键词列表")
    sentiment: str = Field(description="情感：正面/负面/中性")
    topic: str = Field(description="主题分类")

# 初始化模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
json_parser = JsonOutputParser(pydantic_object=TextAnalysis)

# 分析链
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文本分析专家。{format_instructions}"),
    ("human", "分析以下文本：\n{text}")
])

analysis_chain = analysis_prompt | model | json_parser

# 翻译链
translate_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译成英文（如果是英文则翻译成中文）：\n{text}"
)
translate_chain = translate_prompt | model | StrOutputParser()

# 完整管道
pipeline = (
    RunnablePassthrough.assign(
        # 文本长度
        length=lambda x: len(x["text"]),
        # 并行执行分析和翻译
        analysis=analysis_chain,
        translation=translate_chain
    )
    # 后处理
    | RunnablePassthrough.assign(
        word_count=lambda x: len(x["text"].split()),
        reading_time=lambda x: f"{len(x['text'].split()) // 200} 分钟"
    )
)

# 使用示例
result = pipeline.invoke({
    "text": "人工智能正在改变我们的生活方式，从智能手机到自动驾驶，AI无处不在。",
    "format_instructions": json_parser.get_format_instructions()
})

print(result)
# {
#     "text": "...",
#     "length": 35,
#     "analysis": {
#         "summary": "AI正在改变生活方式",
#         "keywords": ["人工智能", "智能手机", "自动驾驶"],
#         "sentiment": "正面",
#         "topic": "科技"
#     },
#     "translation": "Artificial intelligence is changing...",
#     "word_count": 20,
#     "reading_time": "0 分钟"
# }
```

---

## 📋 LCEL 速查表

### 基本操作

| 操作 | 语法 | 说明 |
|------|------|------|
| 串联 | `a \| b` | a 的输出作为 b 的输入 |
| 并行 | `RunnableParallel(a=x, b=y)` | 同时执行多个 Runnable |
| 分配 | `RunnablePassthrough.assign(x=...)` | 向输出添加新字段 |

### Runnable 原语

| 原语 | 用途 | 示例 |
|------|------|------|
| `RunnablePassthrough` | 传递输入 | `RunnablePassthrough()` |
| `RunnableParallel` | 并行执行 | `RunnableParallel(a=..., b=...)` |
| `RunnableLambda` | 自定义函数 | `RunnableLambda(lambda x: x.upper())` |
| `RunnableBranch` | 条件分支 | `RunnableBranch((cond, chain), default)` |

### 链配置方法

```python
# 重试
chain.with_retry(stop_after_attempt=3)

# 超时
chain.with_timeout(30)

# 回退
chain.with_fallbacks([fallback_chain])

# 配置
chain.with_config({"tags": ["production"], "max_concurrency": 5})
```

---

## 🔗 相关章节

- [基础概念](./langchain-basics.md) - 理解核心组件
- [智能体开发](./langchain-agents.md) - 构建自主决策的 Agent
- [工具调用](./langchain-tools.md) - 扩展模型能力
