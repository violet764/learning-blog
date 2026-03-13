# API 服务器部署

vLLM 提供了与 OpenAI API 完全兼容的服务器，可以无缝替换 OpenAI 服务，支持 Chat Completions、Completions 和 Embeddings API。本章介绍如何部署和配置 vLLM API 服务器。

---

## 🚀 快速启动

### 使用命令行启动

```bash
# 基础启动
vllm serve Qwen/Qwen2.5-7B-Instruct

# 完整参数启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code
```

### 使用 Python 启动

```python
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs

# 配置引擎参数
engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-7B-Instruct",
    host="0.0.0.0",
    port=8000,
    dtype="auto",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    trust_remote_code=True,
)

# 启动服务器
run_server(engine_args)
```

### Docker 部署

```bash
# 使用官方镜像
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trust-remote-code

# 使用 docker-compose
# docker-compose.yml
version: '3'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --trust-remote-code
      --gpu-memory-utilization 0.9
```

---

## 🔌 OpenAI 兼容 API

### Chat Completions API

```python
from openai import OpenAI

# 连接到 vLLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要 API Key
)

# 发送对话请求
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是深度学习？"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### 流式输出

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 流式 Chat Completions
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True,
    temperature=0.7,
    max_tokens=200
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Completions API

```python
# 传统 Completions API（非 Chat）
response = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt="翻译成英文：人工智能正在改变世界",
    max_tokens=100,
    temperature=0.3
)

print(response.choices[0].text)
```

### Embeddings API

```python
# 获取文本嵌入向量
response = client.embeddings.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    input="这是一段需要嵌入的文本"
)

embedding = response.data[0].embedding
print(f"嵌入维度: {len(embedding)}")
print(f"前10维: {embedding[:10]}")
```

### 使用 cURL 测试

```bash
# Chat Completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "input": "测试文本"
  }'
```

---

## ⚙️ 启动参数详解

### 基础参数

```bash
vllm serve MODEL_NAME \
    # ===== 服务配置 =====
    --host 0.0.0.0              # 监听地址
    --port 8000                 # 监听端口
    --uvicorn-log-level info    # 日志级别
    
    # ===== 模型配置 =====
    --model Qwen/Qwen2.5-7B-Instruct  # 模型名称或路径
    --tokenizer TOKENIZER_PATH        # 分词器路径
    --revision main                   # 模型版本
    --trust-remote-code               # 信任远程代码
    --dtype auto                      # 数据类型
    
    # ===== 内存配置 =====
    --gpu-memory-utilization 0.9      # GPU 内存利用率
    --max-model-len 4096              # 最大序列长度
    --block-size 16                   # PagedAttention 块大小
    
    # ===== 量化配置 =====
    --quantization awq                # 量化方案
    --load-format auto                # 加载格式
```

### 性能参数

```bash
vllm serve MODEL_NAME \
    # ===== 批处理配置 =====
    --max-num-seqs 256              # 最大并发序列数
    --max-num-batched-tokens 8192   # 最大批处理 token 数
    
    # ===== 调度配置 =====
    --scheduler-policy priority     # 调度策略: priority, fcfs
    --max-seq-len-to-capture 8192   # CUDA graph 捕获长度
    
    # ===== 并行配置 =====
    --tensor-parallel-size 2        # 张量并行 GPU 数
    --pipeline-parallel-size 1      # 流水线并行数
    
    # ===== 缓存配置 =====
    --swap-space 4                  # CPU swap 空间 (GB)
    --enable-prefix-caching         # 启用前缀缓存
```

### API 扩展参数

```bash
vllm serve MODEL_NAME \
    # ===== API 配置 =====
    --api-key your-api-key          # 设置 API Key
    --enable-auto-tool-choice       # 启用工具调用
    --tool-call-parser parser_name  # 工具调用解析器
    --chat-template TEMPLATE_PATH   # 自定义聊天模板
    
    # ===== 响应配置 =====
    --max-log-len 100               # 日志中最大显示长度
    --disable-log-stats             # 禁用统计日志
    --disable-log-requests          # 禁用请求日志
```

### 完整启动示例

```bash
# 生产环境推荐配置
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --api-key sk-your-secret-key \
    --enable-auto-tool-choice \
    --uvicorn-log-level warning
```

---

## 🔧 多模型服务

### 单服务多模型

```bash
# 启动支持多模型的服务
vllm serve \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen-7b \
    --enable-auto-choice

# 同时加载多个模型（需要更多 GPU 内存）
# 方式一：启动多个服务实例
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 &
```

### 模型别名

```python
# 使用 served-model-name 设置别名
# 启动命令
# vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name my-model

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 使用别名调用
response = client.chat.completions.create(
    model="my-model",  # 使用别名
    messages=[{"role": "user", "content": "你好"}]
)
```

### 动态模型加载

```python
# 使用 LoRA 动态加载不同风格的模型
# 启动时启用 LoRA
# vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 使用基础模型
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)

# 使用 LoRA 适配器
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={
        "lora_name": "my-lora-adapter"
    }
)
```

---

## 🔄 流式响应处理

### Python 流式处理

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def stream_chat(messages, model="Qwen/Qwen2.5-7B-Instruct"):
    """流式聊天生成"""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=500
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content
        
        # 检查是否完成
        if chunk.choices[0].finish_reason:
            print(f"\n完成原因: {chunk.choices[0].finish_reason}")
    
    return full_response

# 使用
for text in stream_chat([{"role": "user", "content": "讲一个故事"}]):
    print(text, end="", flush=True)
```

### 异步流式处理

```python
import asyncio
from openai import AsyncOpenAI

async def async_stream_chat():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    
    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "解释一下量子计算"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

# 运行
asyncio.run(async_stream_chat())
```

### Web 框架集成 (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

@app.post("/chat/stream")
async def chat_stream(message: str):
    """SSE 流式响应"""
    
    async def generate():
        stream = await client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                data = json.dumps({"text": chunk.choices[0].delta.content})
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 🛡️ 安全与认证

### API Key 认证

```bash
# 启动时设置 API Key
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --api-key sk-your-secret-api-key
```

```python
from openai import OpenAI

# 客户端需要提供 API Key
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-secret-api-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 反向代理 (Nginx)

```nginx
# nginx.conf
upstream vllm_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    # HTTPS 配置
    # listen 443 ssl;
    # ssl_certificate /path/to/cert.pem;
    # ssl_certificate_key /path/to/key.pem;

    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 流式响应支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
    
    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    location /v1/chat/completions {
        limit_req zone=api_limit burst=20;
        proxy_pass http://vllm_backend;
    }
}
```

---

## 📊 监控与日志

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/health

# 获取模型列表
curl http://localhost:8000/v1/models
```

```python
import requests

def check_health():
    """健康检查"""
    try:
        response = requests.get("http://localhost:8000/health")
        return response.status_code == 200
    except:
        return False

def get_models():
    """获取可用模型"""
    response = requests.get("http://localhost:8000/v1/models")
    return response.json()
```

### Prometheus 监控

```bash
# 启动时启用 metrics
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-metrics \
    --metrics-port 8080
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8080']
```

### 日志配置

```bash
# 日志级别
vllm serve MODEL_NAME \
    --uvicorn-log-level debug \
    --disable-log-requests  # 禁用请求日志（减少日志量）
```

```python
# Python 日志配置
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 🔧 高级配置

### 自定义聊天模板

```bash
# 使用自定义聊天模板
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --chat-template /path/to/template.jinja
```

```jinja
{# template.jinja #}
{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
ASSISTANT: {{ message['content'] }}
{% elif message['role'] == 'system' %}
SYSTEM: {{ message['content'] }}
{% endif %}
{% endfor %}
ASSISTANT:
```

### 工具调用支持

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 发送带工具的请求
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"调用函数: {tool_call.function.name}")
        print(f"参数: {tool_call.function.arguments}")
```

---

## 📋 完整部署示例

### 生产环境配置

```bash
#!/bin/bash
# start_vllm.sh

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
GPU_MEMORY=0.85

vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --api-key ${VLLM_API_KEY:-sk-default-key} \
    --uvicorn-log-level warning \
    --disable-log-requests \
    --enable-metrics
```

### Systemd 服务

```ini
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM API Server
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/home/vllm
Environment="VLLM_API_KEY=your-api-key"
ExecStart=/home/vllm/start_vllm.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

---

## 🔗 相关章节

- [基础概念与安装](./vllm-basics.md) - 安装和基本使用
- [离线推理](./vllm-inference.md) - 批量推理详解
- [性能优化技巧](./vllm-optimization.md) - 深入优化方法
