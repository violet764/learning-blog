# 流式响应处理

流式响应（Streaming）允许在生成过程中逐步接收输出，而不是等待完整响应。这对于长文本生成和实时交互场景非常重要。本章介绍同步/异步流式处理、回调机制和最佳实践。

---

## 🌊 什么是流式响应

### 基本概念

传统 API 调用会等待完整响应后才返回，而流式响应在生成过程中逐步返回增量内容：

```
传统模式:
请求 ──────────────────────> 完整响应
      (等待 10 秒)           "这是一段完整的回复..."

流式模式:
请求 ─> "这" ─> "是一" ─> "段逐" ─> "步生" ─> "成的" ─> "回复"
      (立即开始)   (持续输出)
```

### 应用场景

| 场景 | 优势 |
|------|------|
| **聊天应用** | 用户立即看到回复，体验更好 |
| **长文本生成** | 不必等待完整响应 |
| **实时展示** | 逐字显示，类似打字效果 |
| **进度反馈** | 用户知道系统正在工作 |

---

## 📤 基本流式调用

### 同步流式

```python
from openai import OpenAI

client = OpenAI()

# 创建流式请求
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True  # 启用流式
)

# 遍历流式响应
for chunk in stream:
    # chunk 是一个 ChatCompletionChunk 对象
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 换行
```

### 异步流式

```python
import asyncio
from openai import AsyncOpenAI

async def stream_chat():
    client = AsyncOpenAI()
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "讲一个短故事"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print()

# 运行
asyncio.run(stream_chat())
```

### 使用上下文管理器

```python
from openai import OpenAI

client = OpenAI()

# 推荐方式：使用 with 语句
with client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
) as stream:
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

print()
```

---

## 📦 响应结构

### Chunk 结构

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in stream:
    print(f"ID: {chunk.id}")
    print(f"创建时间: {chunk.created}")
    print(f"模型: {chunk.model}")
    
    choice = chunk.choices[0]
    print(f"索引: {choice.index}")
    print(f"结束原因: {choice.finish_reason}")
    
    delta = choice.delta
    print(f"角色: {delta.role}")  # 第一个 chunk 有
    print(f"内容: {delta.content}")  # 增量内容
    
    print("---")
```

### 响应阶段

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用三句话介绍 Python"}],
    stream=True
)

full_content = ""

for chunk in stream:
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason
    
    # 内容增量
    if delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)
    
    # 完成标记
    if finish_reason:
        print(f"\n[完成: {finish_reason}]")

print(f"\n完整内容:\n{full_content}")
```

---

## 🎯 实用模式

### 模式1：收集完整响应

```python
from openai import OpenAI
from typing import Generator

client = OpenAI()

def stream_and_collect(messages: list, model: str = "gpt-4o-mini") -> tuple:
    """
    流式输出并收集完整响应
    返回: (完整内容, 使用情况)
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}  # 包含使用信息
    )
    
    full_content = ""
    usage = None
    
    for chunk in stream:
        # 收集内容
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            print(content, end="", flush=True)
        
        # 最后一个 chunk 包含使用信息
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = chunk.usage
    
    print()
    
    return full_content, usage

# 使用
content, usage = stream_and_collect([
    {"role": "user", "content": "介绍 Python 的特点"}
])

print(f"\nToken 使用: {usage}")
```

### 模式2：回调处理

```python
from openai import OpenAI
from typing import Callable, Optional

client = OpenAI()

def stream_with_callback(
    messages: list,
    on_content: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[str], None]] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    带回调的流式处理
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    full_content = ""
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            
            # 内容回调
            if on_content:
                on_content(content)
    
    # 完成回调
    if on_complete:
        on_complete(full_content)
    
    return full_content

# 使用示例
def print_content(text: str):
    print(text, end="", flush=True)

def save_result(text: str):
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n[已保存到 output.txt]")

result = stream_with_callback(
    [{"role": "user", "content": "写一篇短文"}],
    on_content=print_content,
    on_complete=save_result
)
```

### 模式3：异步批量流式

```python
import asyncio
from openai import AsyncOpenAI
from typing import List, AsyncGenerator

async def stream_single(
    client: AsyncOpenAI,
    prompt: str,
    index: int
) -> tuple:
    """流式处理单个请求"""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    full_content = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
    
    return index, full_content

async def batch_stream(prompts: List[str]) -> List[str]:
    """并发流式处理多个请求"""
    client = AsyncOpenAI()
    
    tasks = [
        stream_single(client, prompt, i)
        for i, prompt in enumerate(prompts)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 按原始顺序排序
    results.sort(key=lambda x: x[0])
    
    return [content for _, content in results]

# 使用
async def main():
    prompts = [
        "什么是机器学习？",
        "什么是深度学习？",
        "什么是自然语言处理？"
    ]
    
    results = await batch_stream(prompts)
    
    for i, result in enumerate(results):
        print(f"\n问题 {i+1} 的回答:")
        print(result)

asyncio.run(main())
```

---

## 🔄 流式与函数调用

### 流式函数调用

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=tools,
    stream=True
)

# 流式处理工具调用
tool_calls = {}  # 收集工具调用参数

for chunk in stream:
    delta = chunk.choices[0].delta
    
    # 处理内容
    if delta.content:
        print(delta.content, end="", flush=True)
    
    # 处理工具调用
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tc.id,
                    "name": "",
                    "arguments": ""
                }
            
            if tc.function:
                if tc.function.name:
                    tool_calls[idx]["name"] = tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments

print()

# 解析工具调用
for idx, tc in tool_calls.items():
    print(f"工具调用 {idx}:")
    print(f"  ID: {tc['id']}")
    print(f"  函数: {tc['name']}")
    print(f"  参数: {tc['arguments']}")
    
    args = json.loads(tc['arguments'])
    print(f"  解析后: {args}")
```

---

## 🎨 高级应用

### 打字机效果

```python
from openai import OpenAI
import time
import random

client = OpenAI()

def typewriter_stream(messages: list, delay_range: tuple = (0.01, 0.05)):
    """模拟打字机效果的流式输出"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            char = chunk.choices[0].delta.content
            print(char, end="", flush=True)
            
            # 随机延迟
            delay = random.uniform(*delay_range)
            time.sleep(delay)
    
    print()

# 使用
typewriter_stream([
    {"role": "user", "content": "你好，请自我介绍"}
])
```

### Web 框架集成（FastAPI）

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI()

async def generate_stream(prompt: str):
    """生成 SSE 流"""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # SSE 格式
            yield f"data: {json.dumps({'content': content})}\n\n"
    
    # 发送结束信号
    yield "data: [DONE]\n\n"

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream"
    )
```

### 进度显示

```python
from openai import OpenAI
import sys

client = OpenAI()

def stream_with_progress(messages: list):
    """带进度指示的流式输出"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    char_count = 0
    progress_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    progress_idx = 0
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            
            # 清除进度指示器
            if char_count == 0:
                sys.stdout.write("\r" + " " * 2 + "\r")
            
            print(content, end="", flush=True)
            char_count += len(content)
        
        # 显示进度（当没有内容时）
        elif chunk.choices[0].finish_reason is None:
            sys.stdout.write(f"\r{progress_chars[progress_idx]} ")
            sys.stdout.flush()
            progress_idx = (progress_idx + 1) % len(progress_chars)
    
    print(f"\n\n共 {char_count} 个字符")

# 使用
stream_with_progress([
    {"role": "user", "content": "写一篇 200 字的短文"}
])
```

---

## ⚠️ 注意事项

### 错误处理

```python
from openai import OpenAI, APIError, APIConnectionError

client = OpenAI()

def safe_stream(messages: list):
    """带错误处理的流式调用"""
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except APIConnectionError as e:
        yield f"\n[连接错误: {e}]"
    
    except APIError as e:
        yield f"\n[API 错误: {e}]"
    
    except Exception as e:
        yield f"\n[未知错误: {e}]"

# 使用
for content in safe_stream([{"role": "user", "content": "你好"}]):
    print(content, end="", flush=True)
```

### 超时控制

```python
from openai import OpenAI
import httpx

# 设置超时
client = OpenAI(
    timeout=httpx.Timeout(60.0, read=30.0)
)

# 或者使用 with 语句自动超时
with client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    timeout=30.0
) as stream:
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

### 取消流式

```python
from openai import OpenAI

client = OpenAI()

def stream_with_cancel(messages: list, max_chars: int = 100):
    """可取消的流式输出"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    char_count = 0
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            char_count += len(content)
            
            if char_count > max_chars:
                print("\n[已达到最大字符数，停止输出]")
                break
            
            print(content, end="", flush=True)

# 使用
stream_with_cancel(
    [{"role": "user", "content": "写一篇长文"}],
    max_chars=50
)
```

---

## 📋 最佳实践

### ✅ 推荐做法

```python
# 1. 使用上下文管理器
with client.chat.completions.create(..., stream=True) as stream:
    for chunk in stream:
        ...

# 2. 异步场景使用 AsyncOpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI()
async for chunk in await client.chat.completions.create(..., stream=True):
    ...

# 3. 处理 None 内容
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="")

# 4. 设置合理的超时
client = OpenAI(timeout=httpx.Timeout(60.0))
```

### ❌ 避免的做法

```python
# 1. 不要忘记处理 None
for chunk in stream:
    print(chunk.choices[0].delta.content)  # 可能报错

# 2. 不要在循环中创建客户端
for prompt in prompts:
    client = OpenAI()  # ❌ 效率低
    ...

# 3. 不要忽略错误
for chunk in stream:  # ❌ 没有错误处理
    ...
```

---

## 🔗 相关章节

- [Chat Completions](./openai-chat.md) - 对话 API
- [基础 API 调用](./openai-basics.md) - 错误处理
- [Function Calling](./openai-functions.md) - 工具调用
