# 基础 API 调用

本章介绍 OpenAI Python SDK 的基础使用：安装配置、认证方式、错误处理和最佳实践。掌握这些基础知识是使用任何 OpenAI API 功能的前提。

---

## 📦 安装与环境配置

### 安装 SDK

```bash
# 安装最新版本
pip install openai

# 查看已安装版本
pip show openai

# 升级到最新版
pip install --upgrade openai
```

### 依赖说明

OpenAI SDK v1.x 的核心依赖：

```
openai
├── httpx        # HTTP 客户端（同步和异步）
├── pydantic     # 数据验证和模型
├── tqdm         # 进度条显示
├── typing-extensions  # 类型扩展
└── distro       # 系统信息
```

### 版本检查

```python
import openai

print(f"OpenAI SDK 版本: {openai.__version__}")
print(f"版本号: {openai.__version_info__}")
```

---

## 🔑 认证配置

### 方式一：环境变量（推荐）

最安全的方式，API Key 不会出现在代码中：

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_ORG_ID="org-your-org-id"  # 可选

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key"
$env:OPENAI_ORG_ID="org-your-org-id"
```

```python
from openai import OpenAI

# 自动读取环境变量
client = OpenAI()

# 等价于
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID")
)
```

### 方式二：使用 .env 文件

适合本地开发环境：

```bash
# 安装 python-dotenv
pip install python-dotenv
```

```bash
# .env 文件内容
OPENAI_API_KEY=sk-your-api-key
OPENAI_ORG_ID=org-your-org-id
OPENAI_BASE_URL=https://api.openai.com/v1
```

```python
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

# 初始化客户端
client = OpenAI()
```

### 方式三：代码中直接设置

⚠️ 不推荐用于生产环境或公开代码仓库：

```python
from openai import OpenAI

client = OpenAI(api_key="sk-your-api-key")
```

### 方式四：自定义 Base URL

适用于代理服务或 Azure OpenAI：

```python
from openai import OpenAI

# 使用代理服务
client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-proxy.com/v1"
)

# Azure OpenAI
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-azure-key",
    api_version="2024-02-15-preview",
    azure_endpoint="https://your-resource.openai.azure.com"
)
```

---

## 🚀 基本使用

### 同步客户端

```python
from openai import OpenAI

# 初始化
client = OpenAI()

# 发送请求
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "你好，介绍一下 Python"}
    ]
)

# 获取回复
print(response.choices[0].message.content)

# 查看 token 使用情况
print(f"提示词 tokens: {response.usage.prompt_tokens}")
print(f"完成 tokens: {response.usage.completion_tokens}")
print(f"总计 tokens: {response.usage.total_tokens}")
```

### 异步客户端

异步客户端适合高并发场景：

```python
import asyncio
from openai import AsyncOpenAI

async def chat():
    client = AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    return response.choices[0].message.content

# 运行
result = asyncio.run(chat())
print(result)
```

### 并发请求示例

```python
import asyncio
from openai import AsyncOpenAI

async def process_question(client, question):
    """处理单个问题"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return question, response.choices[0].message.content

async def batch_questions(questions):
    """批量并发处理问题"""
    client = AsyncOpenAI()
    
    tasks = [process_question(client, q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    return dict(results)

# 使用
questions = [
    "什么是机器学习？",
    "什么是深度学习？",
    "什么是自然语言处理？"
]

results = asyncio.run(batch_questions(questions))
for q, a in results.items():
    print(f"Q: {q}\nA: {a}\n")
```

---

## 📤 响应结构

### 响应对象

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}]
)

# 响应对象结构
print(type(response))  # <class 'openai.types.chat.ChatCompletion'>

# 访问属性
print(response.id)           # 响应 ID
print(response.object)       # "chat.completion"
print(response.created)      # 创建时间戳
print(response.model)        # 使用的模型

# 选择列表
print(response.choices)      # List[Choice]

# 第一个选择
choice = response.choices[0]
print(choice.index)          # 索引
print(choice.finish_reason)  # 结束原因: "stop", "length", etc.
print(choice.message)        # 消息对象

# 消息内容
message = choice.message
print(message.role)          # "assistant"
print(message.content)       # 回复内容

# Token 使用
print(response.usage)
# Usage(prompt_tokens=10, completion_tokens=50, total_tokens=60)
```

### 完整响应示例

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "说一个数字"}],
    max_tokens=10
)

# 格式化输出响应
def print_response(response):
    """格式化打印响应"""
    print("=" * 50)
    print(f"响应 ID: {response.id}")
    print(f"模型: {response.model}")
    print(f"创建时间: {response.created}")
    print("-" * 50)
    print("回复内容:")
    for i, choice in enumerate(response.choices):
        print(f"  [{i}] {choice.message.content}")
        print(f"  结束原因: {choice.finish_reason}")
    print("-" * 50)
    print("Token 使用:")
    print(f"  提示词: {response.usage.prompt_tokens}")
    print(f"  完成: {response.usage.completion_tokens}")
    print(f"  总计: {response.usage.total_tokens}")
    print("=" * 50)

print_response(response)
```

---

## ⚠️ 错误处理

### 异常类型

OpenAI SDK 定义了多种异常类型：

```python
from openai import (
    OpenAI,           # 主客户端
    APIError,         # API 错误基类
    APIConnectionError,   # 连接错误
    APITimeoutError,      # 超时错误
    AuthenticationError,  # 认证错误
    BadRequestError,      # 请求参数错误
    ConflictError,        # 冲突错误
    InternalServerError,  # 服务器错误
    NotFoundError,        # 资源不存在
    PermissionDeniedError,# 权限不足
    RateLimitError,       # 速率限制
    UnprocessableEntityError,  # 无法处理的实体
)
```

### 完整错误处理示例

```python
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import time

def safe_chat(client, messages, max_retries=3):
    """带错误处理的安全调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response
        
        except AuthenticationError as e:
            # 认证错误，不需要重试
            print(f"认证失败: {e}")
            raise
            
        except RateLimitError as e:
            # 速率限制，等待后重试
            wait_time = 2 ** attempt  # 指数退避
            print(f"触发速率限制，等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            continue
            
        except APIError as e:
            # API 错误
            print(f"API 错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
    
    raise Exception("超过最大重试次数")

# 使用
client = OpenAI()
try:
    response = safe_chat(
        client,
        [{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"请求失败: {e}")
```

### 使用 tenacity 重试库

```python
from openai import OpenAI, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置重试策略
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=lambda e: isinstance(e, (RateLimitError, APIError))
)
def chat_with_retry(client, messages):
    """自动重试的聊天函数"""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

# 使用
client = OpenAI()
response = chat_with_retry(client, [{"role": "user", "content": "你好"}])
```

---

## ⚙️ 客户端配置

### 常用配置参数

```python
from openai import OpenAI

client = OpenAI(
    # 基础配置
    api_key="your-api-key",
    organization="org-xxx",          # 组织 ID
    base_url="https://api.openai.com/v1",
    
    # 超时配置
    timeout=60.0,                    # 总超时时间（秒）
    max_retries=2,                   # 最大重试次数
    
    # HTTP 配置
    http_client=None,                # 自定义 HTTP 客户端
    
    # 默认请求头
    default_headers={
        "X-My-Header": "value"
    },
    
    # 默认查询参数
    default_query={
        "api-version": "2024-02-15-preview"
    }
)
```

### 自定义超时

```python
from openai import OpenAI
import httpx

# 创建自定义 HTTP 客户端
http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=5.0,      # 连接超时
        read=30.0,        # 读取超时
        write=10.0,       # 写入超时
        pool=5.0          # 连接池超时
    ),
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )
)

client = OpenAI(http_client=http_client)
```

### 异步客户端配置

```python
import httpx
from openai import AsyncOpenAI

# 自定义异步 HTTP 客户端
async_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_connections=100)
)

client = AsyncOpenAI(http_client=async_http_client)
```

---

## 🔧 高级用法

### 上下文管理器

```python
from openai import OpenAI

# 使用上下文管理器自动清理资源
with OpenAI() as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
# 自动关闭连接
```

### 自定义日志

```python
import logging
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

client = OpenAI()
# 现在会打印详细的请求/响应日志
```

### 请求 ID 追踪

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# 获取请求 ID 用于问题追踪
print(f"请求 ID: {response.id}")

# 获取响应头信息
print(f"请求时间: {response.created}")
```

---

## 📊 成本监控

### Token 计算

```python
from openai import OpenAI

client = OpenAI()

def chat_with_cost_tracking(messages, model="gpt-4o-mini"):
    """带成本追踪的聊天"""
    # 价格表（美元 / 1M tokens）
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    # 计算成本
    usage = response.usage
    cost_input = (usage.prompt_tokens / 1_000_000) * pricing[model]["input"]
    cost_output = (usage.completion_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = cost_input + cost_output
    
    print(f"模型: {model}")
    print(f"输入 tokens: {usage.prompt_tokens} (${cost_input:.6f})")
    print(f"输出 tokens: {usage.completion_tokens} (${cost_output:.6f})")
    print(f"总成本: ${total_cost:.6f}")
    
    return response

# 使用
response = chat_with_cost_tracking(
    [{"role": "user", "content": "解释一下量子计算"}]
)
```

---

## 📋 最佳实践

### ✅ 推荐做法

```python
# 1. 使用环境变量存储 API Key
import os
from openai import OpenAI

client = OpenAI()  # 自动读取 OPENAI_API_KEY

# 2. 使用异步客户端处理并发请求
from openai import AsyncOpenAI
async_client = AsyncOpenAI()

# 3. 设置合理的超时
client = OpenAI(timeout=60.0)

# 4. 实现重试机制
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def call_api():
    ...

# 5. 追踪 token 使用
response = client.chat.completions.create(...)
print(f"Tokens: {response.usage.total_tokens}")
```

### ❌ 避免的做法

```python
# 1. 不要在代码中硬编码 API Key
client = OpenAI(api_key="sk-xxx")  # ❌ 危险！

# 2. 不要忽略错误处理
response = client.chat.completions.create(...)  # ❌ 可能崩溃

# 3. 不要在循环中同步请求大量数据
for item in large_list:
    response = client.chat.completions.create(...)  # ❌ 太慢

# 4. 不要忘记关闭客户端
client = OpenAI()
# ... 使用后没有关闭连接
```

---

## 🔗 相关章节

- [Chat Completions](./openai-chat.md) - 对话 API 详解
- [流式响应处理](./openai-streaming.md) - 流式输出
- [Function Calling](./openai-functions.md) - 工具调用
