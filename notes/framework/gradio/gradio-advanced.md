# Gradio 高级特性与部署

本章介绍 Gradio 的高级特性，包括流式输出、文件处理、认证安全等，以及如何将应用部署到 Hugging Face Spaces 或自有服务器。

---

## 📌 流式输出

流式输出对于大语言模型应用尤为重要，可以让用户实时看到生成的内容，提升用户体验。

### 文本流式输出

```python
import gradio as gr
import time

def stream_text(prompt):
    # 模拟流式生成
    response = "这是一段流式输出的文本，字符会逐个显示，模拟大语言模型的生成效果。"
    for char in response:
        time.sleep(0.03)
        yield char  # 使用 yield 实现流式输出

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="提示词")
    output_text = gr.Textbox(label="生成结果")
    btn = gr.Button("生成")
    
    btn.click(
        fn=stream_text,
        inputs=input_text,
        outputs=output_text
    )
```

### Chatbot 流式输出

```python
import gradio as gr
import time

def stream_chat(message, history):
    # 流式生成回复
    response = f"收到您的消息：{message}。这是一个模拟的流式回复，展示如何逐字显示内容。"
    
    partial_response = ""
    for char in response:
        partial_response += char
        time.sleep(0.02)
        yield history + [(message, partial_response)]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="输入消息...")
    
    msg.submit(
        fn=stream_chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
```

### 结合 LLM 的流式输出

```python
import gradio as gr
from openai import OpenAI

client = OpenAI()

def stream_llm(message, history):
    messages = [{"role": "system", "content": "你是一个友好的助手。"}]
    
    # 添加历史对话
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": message})
    
    # 流式调用 API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    
    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    
    msg.submit(
        fn=stream_llm,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
```

---

## 📁 文件处理

### 文件上传与下载

```python
import gradio as gr
import tempfile
import os

def process_file(file):
    if file is None:
        return None, "请上传文件"
    
    # 读取文件内容
    with open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 处理内容（示例：统计词数）
    word_count = len(content.split())
    
    # 创建输出文件
    output_path = tempfile.mktemp(suffix='.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"原文词数：{word_count}\n\n")
        f.write(content.upper())  # 转大写作为示例
    
    return output_path, f"处理完成，词数：{word_count}"

with gr.Blocks() as demo:
    gr.Markdown("### 文件处理工具")
    
    file_input = gr.File(label="上传文件", file_types=[".txt", ".md", ".csv"])
    
    with gr.Row():
        file_output = gr.File(label="下载处理结果")
        status = gr.Textbox(label="状态")
    
    process_btn = gr.Button("处理", variant="primary")
    
    process_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[file_output, status]
    )
```

### 批量文件处理

```python
import gradio as gr
import zipfile
import tempfile

def process_multiple_files(files):
    if not files:
        return None
    
    # 创建临时目录
    output_dir = tempfile.mkdtemp()
    
    for file in files:
        # 处理每个文件
        filename = os.path.basename(file.name)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        
        with open(file.name, 'r') as f:
            content = f.read()
        
        with open(output_path, 'w') as f:
            f.write(content.upper())
    
    # 打包成 zip
    zip_path = tempfile.mktemp(suffix='.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, filename), filename)
    
    return zip_path

with gr.Blocks() as demo:
    files_input = gr.File(
        label="上传多个文件",
        file_count="multiple",
        file_types=[".txt"]
    )
    files_output = gr.File(label="下载压缩包")
    
    btn = gr.Button("批量处理")
    btn.click(fn=process_multiple_files, inputs=files_input, outputs=files_output)
```

---

## 🔐 认证与安全

### 密码保护

```python
import gradio as gr

def greet(name):
    return f"你好，{name}！"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

# 添加用户名/密码认证
demo.launch(
    auth=("admin", "password123"),  # 单个用户
    # 或多个用户：
    # auth=[("user1", "pass1"), ("user2", "pass2")]
)
```

### 自定义认证函数

```python
import gradio as gr

def verify_auth(username, password):
    # 自定义验证逻辑
    valid_users = {
        "admin": "admin123",
        "user": "user123"
    }
    return username in valid_users and valid_users[username] == password

demo.launch(auth=verify_auth)
```

### 隐藏 API 信息

```python
demo.launch(
    show_api=False,  # 隐藏 API 文档
    share=False      # 不创建公开链接
)
```

---

## ⚡ 性能优化

### 队列配置

```python
import gradio as gr

demo = gr.Interface(...)

# 配置队列
demo.queue(
    max_size=20,                    # 最大排队数
    default_concurrency_limit=5,    # 默认并发数
    api_open=True                   # 是否开放 API
)

demo.launch()
```

### 并发限制

```python
import gradio as gr

with gr.Blocks() as demo:
    btn = gr.Button("处理")
    output = gr.Textbox()
    
    btn.click(
        fn=heavy_process,
        inputs=None,
        outputs=output,
        concurrency_limit=3,        # 限制并发数
        concurrency_id="heavy"      # 并发组标识
    )

# 为不同并发组设置不同限制
demo.queue(concurrency_count=10)
demo.launch()
```

### 缓存

```python
import gradio as gr
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_process(text):
    # 缓存处理结果
    return text.upper()

demo = gr.Interface(
    fn=cached_process,
    inputs="text",
    outputs="text"
)
```

---

## 🚀 部署到 Hugging Face Spaces

### 创建 Space

1. 访问 [huggingface.co/new-space](https://huggingface.co/new-space)
2. 选择 **Gradio** 作为 SDK
3. 命名你的 Space（如 `my-gradio-app`）

### 项目结构

```
my-gradio-app/
├── app.py           # 主应用文件（必须是这个名字）
├── requirements.txt # Python 依赖
├── README.md        # Space 说明
└── assets/          # 静态资源（可选）
    └── example.jpg
```

### requirements.txt 示例

```txt
gradio>=4.0.0
transformers
torch
accelerate
```

### README.md 配置

```yaml
---
title: My Gradio App
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 我的应用描述

这里是应用的详细说明...
```

### 推送到 Space

```bash
# 方法1：通过 Git
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
# 复制你的文件
git add .
git commit -m "Initial commit"
git push

# 方法2：使用 huggingface_hub 库
pip install huggingface_hub

from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id="YOUR_USERNAME/YOUR_SPACE_NAME",
    repo_type="space"
)
```

### 环境变量

在 Space 设置中添加 Secrets（敏感信息）：

```python
import os
api_key = os.environ.get("OPENAI_API_KEY")
```

---

## 🐳 Docker 部署

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]
```

### 启动脚本

在 `app.py` 中：

```python
import gradio as gr

demo = gr.Interface(...)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
```

### 构建和运行

```bash
# 构建镜像
docker build -t my-gradio-app .

# 运行容器
docker run -p 7860:7860 my-gradio-app

# 使用环境变量
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key my-gradio-app
```

---

## 🖥️ 服务器部署

### Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_websockets True;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Systemd 服务

创建 `/etc/systemd/system/gradio-app.service`：

```ini
[Unit]
Description=Gradio Application
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/app
ExecStart=/path/to/venv/bin/python app.py
Restart=always
RestartSec=10
Environment="OPENAI_API_KEY=your_key"

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl enable gradio-app
sudo systemctl start gradio-app
```

### SSL 配置

使用 Certbot 配置 HTTPS：

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 📊 监控与日志

### 日志记录

```python
import gradio as gr
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gradio.log'
)

def process(text):
    logging.info(f"Processing: {text[:50]}...")
    result = text.upper()
    logging.info(f"Result: {result[:50]}...")
    return result

demo = gr.Interface(fn=process, inputs="text", outputs="text")
demo.launch()
```

### 性能监控

```python
import gradio as gr
import time

def monitored_process(text):
    start_time = time.time()
    
    # 处理逻辑
    result = text.upper()
    
    elapsed = time.time() - start_time
    print(f"处理时间: {elapsed:.2f}s")
    
    return result
```

---

## 🔗 API 访问

Gradio 应用可以像 API 一样被调用：

### 获取 API 信息

```python
# 访问 /info 路由查看 API 文档
# http://localhost:7860/info
```

### 使用 Python 调用

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
result = client.predict(
    "Hello World",
    api_name="/predict"
)
print(result)
```

### 使用 cURL 调用

```bash
curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["Hello World"]}'
```

---

## 📋 部署检查清单

| 检查项 | 说明 |
|--------|------|
| ✅ `server_name="0.0.0.0"` | 允许外部访问 |
| ✅ 环境变量管理 | 敏感信息使用环境变量 |
| ✅ 队列配置 | 长时间任务启用队列 |
| ✅ 错误处理 | 添加 try-except 捕获异常 |
| ✅ 日志记录 | 记录关键操作和错误 |
| ✅ HTTPS | 生产环境使用 HTTPS |
| ✅ 认证保护 | 必要时添加认证 |

---

## 🔗 相关章节

- [基础组件与布局](./gradio-basics.md) - 组件和布局基础
- [Interface 快速构建](./gradio-interface.md) - 快速创建界面
- [Blocks 灵活布局](./gradio-blocks.md) - 复杂交互应用
- [Transformers](../transformers/index.md) - 模型集成
