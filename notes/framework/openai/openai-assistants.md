# Assistants API

Assistants API 是 OpenAI 提供的有状态 API，专门用于构建具有持久上下文、文件处理能力和工具调用功能的 AI 助手。本章介绍助手的创建、线程管理、文件处理和运行控制。

---

## 🎯 什么是 Assistants API

### 核心概念

Assistants API 提供了三个核心概念：

| 概念 | 说明 |
|------|------|
| **Assistant** | 助手，配置模型、指令和工具 |
| **Thread** | 线程，存储对话历史和上下文 |
| **Run** | 运行，执行助手处理消息的过程 |

```
┌─────────────────────────────────────────────────────┐
│                  Assistants 架构                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌─────────────┐     ┌─────────────┐              │
│   │  Assistant  │────▶│   Thread    │              │
│   │  (助手)     │     │  (线程)     │              │
│   │             │     │             │              │
│   │ - 模型      │     │ - Messages  │              │
│   │ - 指令      │     │ - 上下文    │              │
│   │ - 工具      │     │             │              │
│   │ - 文件      │     └──────┬──────┘              │
│   └─────────────┘            │                      │
│                              ▼                      │
│                      ┌─────────────┐                │
│                      │    Run      │                │
│                      │  (运行)     │                │
│                      │             │                │
│                      │ - 执行状态  │                │
│                      │ - 步骤追踪  │                │
│                      └─────────────┘                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 与 Chat API 的区别

| 特性 | Chat API | Assistants API |
|------|----------|----------------|
| 状态管理 | 无状态，需自行管理历史 | 有状态，自动管理线程 |
| 文件处理 | 需自行处理 | 内置文件上传和解析 |
| 工具调用 | 单次请求 | 支持多轮工具调用 |
| 代码执行 | 无 | 内置 Code Interpreter |
| 向量搜索 | 无 | 内置 File Search |

---

## 🔧 创建助手

### 基本创建

```python
from openai import OpenAI

client = OpenAI()

# 创建基本助手
assistant = client.beta.assistants.create(
    name="Python 学习助手",
    instructions="你是一个专业的 Python 编程老师，擅长解释编程概念并提供示例代码。",
    model="gpt-4o-mini"
)

print(f"助手 ID: {assistant.id}")
print(f"助手名称: {assistant.name}")
```

### 带工具的助手

```python
from openai import OpenAI

client = OpenAI()

# 创建带 Code Interpreter 的助手
assistant = client.beta.assistants.create(
    name="数据分析助手",
    instructions="""
你是一个数据分析专家。使用 Code Interpreter 来：
1. 分析用户上传的数据文件
2. 执行数据处理和计算
3. 生成可视化图表
""",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}]
)

# 创建带 File Search 的助手
assistant = client.beta.assistants.create(
    name="文档问答助手",
    instructions="你是一个文档问答助手。根据上传的文档回答用户问题。",
    model="gpt-4o-mini",
    tools=[{"type": "file_search"}]
)

# 创建带自定义函数的助手
assistant = client.beta.assistants.create(
    name="智能助手",
    instructions="你是一个智能助手，可以使用工具帮助用户。",
    model="gpt-4o-mini",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
)
```

### 助手属性

```python
# 查看助手信息
print(f"ID: {assistant.id}")
print(f"名称: {assistant.name}")
print(f"描述: {assistant.description}")
print(f"模型: {assistant.model}")
print(f"指令: {assistant.instructions}")
print(f"工具: {assistant.tools}")
print(f"元数据: {assistant.metadata}")

# 更新助手
assistant = client.beta.assistants.update(
    assistant.id,
    name="新名称",
    instructions="新的指令..."
)

# 删除助手
client.beta.assistants.delete(assistant.id)

# 列出所有助手
assistants = client.beta.assistants.list()
for asst in assistants:
    print(f"{asst.id}: {asst.name}")
```

---

## 💬 线程与消息

### 创建线程

```python
from openai import OpenAI

client = OpenAI()

# 创建空线程
thread = client.beta.threads.create()
print(f"线程 ID: {thread.id}")

# 创建带初始消息的线程
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "你好，我想学习 Python"
        }
    ]
)
```

### 添加消息

```python
# 添加用户消息
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="请介绍一下 Python 的列表推导式"
)

# 添加带文件的消息
# 先上传文件
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose='assistants'
)

# 添加带文件的消息
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="请分析这个数据文件",
    attachments=[
        {
            "file_id": file.id,
            "tools": [{"type": "code_interpreter"}]
        }
    ]
)
```

### 查看消息

```python
# 列出线程中的消息
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc"  # 按时间升序
)

for msg in messages:
    print(f"角色: {msg.role}")
    print(f"内容: {msg.content[0].text.value}")
    print(f"时间: {msg.created_at}")
    print("---")

# 获取单条消息
message = client.beta.threads.messages.retrieve(
    thread_id=thread.id,
    message_id=message.id
)
```

---

## ▶️ 运行助手

### 基本运行流程

```python
from openai import OpenAI
import time

client = OpenAI()

# 创建助手
assistant = client.beta.assistants.create(
    name="助手",
    instructions="你是一个有帮助的助手。",
    model="gpt-4o-mini"
)

# 创建线程并添加消息
thread = client.beta.threads.create(
    messages=[{"role": "user", "content": "什么是机器学习？"}]
)

# 创建运行
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

print(f"运行 ID: {run.id}")
print(f"状态: {run.status}")  # queued

# 等待运行完成
while run.status in ["queued", "in_progress", "cancelling"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(f"当前状态: {run.status}")

# 运行完成后获取回复
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages:
        if msg.role == "assistant":
            print(f"回复: {msg.content[0].text.value}")
```

### 运行状态

| 状态 | 说明 |
|------|------|
| `queued` | 排队中 |
| `in_progress` | 执行中 |
| `requires_action` | 需要执行工具 |
| `cancelling` | 取消中 |
| `cancelled` | 已取消 |
| `failed` | 失败 |
| `completed` | 完成 |
| `expired` | 过期 |

### 流式运行

```python
from openai import OpenAI

client = OpenAI()

# 创建流式运行
with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id
) as stream:
    for event in stream:
        # 处理不同类型的事件
        if event.event == "thread.message.delta":
            # 文本增量
            delta = event.data.delta.content[0].text.value
            print(delta, end="", flush=True)
        
        elif event.event == "thread.run.completed":
            print("\n[完成]")
        
        elif event.event == "thread.run.failed":
            print(f"\n[失败]: {event.data.last_error}")
```

---

## 🛠️ 工具调用处理

### 处理 requires_action

当助手需要调用工具时，运行状态会变为 `requires_action`：

```python
from openai import OpenAI
import json
import time

client = OpenAI()

# 定义工具函数
def get_weather(city: str) -> dict:
    # 模拟天气数据
    return {
        "city": city,
        "temperature": 25,
        "weather": "晴天"
    }

# 创建带工具的助手
assistant = client.beta.assistants.create(
    name="天气助手",
    instructions="你是一个天气助手，帮助用户查询天气。",
    model="gpt-4o-mini",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
)

# 创建线程和运行
thread = client.beta.threads.create(
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}]
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 处理工具调用
available_functions = {"get_weather": get_weather}

while True:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    
    if run.status == "requires_action":
        # 获取工具调用
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 执行函数
            result = available_functions[function_name](**function_args)
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result, ensure_ascii=False)
            })
        
        # 提交工具输出
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    
    elif run.status == "completed":
        break
    
    elif run.status in ["failed", "cancelled", "expired"]:
        print(f"运行失败: {run.status}")
        break
    
    time.sleep(1)

# 获取回复
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages:
    if msg.role == "assistant":
        print(msg.content[0].text.value)
```

---

## 📁 文件处理

### 上传文件

```python
from openai import OpenAI

client = OpenAI()

# 上传文件用于 Code Interpreter
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose='assistants'
)

print(f"文件 ID: {file.id}")
print(f"文件名: {file.filename}")

# 上传文件用于 File Search
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose='assistants'
)
```

### 助手关联文件

```python
# 创建带文件关联的助手
assistant = client.beta.assistants.create(
    name="文档助手",
    instructions="根据上传的文档回答问题。",
    model="gpt-4o-mini",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_stores": [{
                "file_ids": [file.id]
            }]
        }
    }
)
```

### 管理文件

```python
# 列出文件
files = client.files.list()
for f in files:
    print(f"{f.id}: {f.filename} ({f.purpose})")

# 获取文件信息
file = client.files.retrieve(file.id)

# 删除文件
client.files.delete(file.id)
```

---

## 🎯 完整示例

### 示例：数据分析助手

```python
from openai import OpenAI
import time

client = OpenAI()

class DataAnalysisAssistant:
    """数据分析助手"""
    
    def __init__(self):
        self.assistant = None
        self.thread = None
        self._setup_assistant()
    
    def _setup_assistant(self):
        """设置助手"""
        self.assistant = client.beta.assistants.create(
            name="数据分析专家",
            instructions="""
你是一个数据分析专家。你的任务是：
1. 分析用户上传的数据文件
2. 提供数据统计和洞察
3. 生成可视化图表
4. 回答关于数据的问题
""",
            model="gpt-4o-mini",
            tools=[{"type": "code_interpreter"}]
        )
        
        self.thread = client.beta.threads.create()
    
    def upload_file(self, filepath: str):
        """上传数据文件"""
        file = client.files.create(
            file=open(filepath, "rb"),
            purpose='assistants'
        )
        return file.id
    
    def analyze(self, filepath: str, question: str = "请分析这个数据"):
        """分析数据文件"""
        # 上传文件
        file_id = self.upload_file(filepath)
        
        # 添加带文件的消息
        client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=question,
            attachments=[
                {
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                }
            ]
        )
        
        # 运行助手
        run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )
        
        # 等待完成
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"运行失败: {run.status}")
            
            time.sleep(1)
        
        # 获取回复
        messages = client.beta.threads.messages.list(thread_id=self.thread.id)
        
        results = []
        for msg in messages:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        results.append(content.text.value)
                    elif content.type == "image_file":
                        # 获取生成的图片
                        image_file = content.image_file
                        results.append(f"[图片: {image_file.file_id}]")
        
        return results
    
    def chat(self, message: str):
        """继续对话"""
        client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message
        )
        
        run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )
        
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            
            if run.status == "completed":
                break
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"运行失败: {run.status}")
            
            time.sleep(1)
        
        messages = client.beta.threads.messages.list(thread_id=self.thread.id)
        
        for msg in messages:
            if msg.role == "assistant":
                return msg.content[0].text.value
        
        return None

# 使用
assistant = DataAnalysisAssistant()

# 分析数据
results = assistant.analyze("sales.csv", "请分析销售数据并生成趋势图")
for result in results:
    print(result)

# 继续对话
response = assistant.chat("哪个产品销售额最高？")
print(response)
```

---

## 📋 API 速查

### Assistants

```python
# 创建
assistant = client.beta.assistants.create(
    name="名称",
    instructions="指令",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}]
)

# 检索
assistant = client.beta.assistants.retrieve(assistant_id)

# 更新
assistant = client.beta.assistants.update(assistant_id, name="新名称")

# 删除
client.beta.assistants.delete(assistant_id)

# 列表
assistants = client.beta.assistants.list()
```

### Threads

```python
# 创建
thread = client.beta.threads.create()

# 检索
thread = client.beta.threads.retrieve(thread_id)

# 删除
client.beta.threads.delete(thread_id)
```

### Messages

```python
# 创建消息
message = client.beta.threads.messages.create(
    thread_id,
    role="user",
    content="内容"
)

# 列出消息
messages = client.beta.threads.messages.list(thread_id)

# 检索消息
message = client.beta.threads.messages.retrieve(thread_id, message_id)
```

### Runs

```python
# 创建运行
run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant_id
)

# 检索运行
run = client.beta.threads.runs.retrieve(thread_id, run_id)

# 取消运行
run = client.beta.threads.runs.cancel(thread_id, run_id)

# 提交工具输出
run = client.beta.threads.runs.submit_tool_outputs(
    thread_id,
    run_id,
    tool_outputs=[...]
)
```

---

## 🔗 相关章节

- [Chat Completions](./openai-chat.md) - 基础对话 API
- [Function Calling](./openai-functions.md) - 工具调用
- [LangChain](../langchain/index.md) - Agent 开发框架
