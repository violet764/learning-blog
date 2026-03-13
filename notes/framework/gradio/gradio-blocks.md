# Gradio Blocks 灵活布局

`gr.Blocks` 是 Gradio 的底层 API，提供了比 `gr.Interface` 更高的灵活性。通过 Blocks，你可以完全自定义界面布局、实现复杂的组件交互、管理会话状态，构建专业级的 Web 应用。

---

## 📌 Blocks 基础

### 基本结构

```python
import gradio as gr

def greet(name):
    return f"你好，{name}！"

with gr.Blocks() as demo:
    gr.Markdown("# 问候应用")
    
    name = gr.Textbox(label="姓名")
    output = gr.Textbox(label="问候语")
    btn = gr.Button("发送")
    
    btn.click(fn=greet, inputs=name, outputs=output)

demo.launch()
```

### Blocks vs Interface

```python
# Interface 方式
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

# Blocks 方式（等价）
with gr.Blocks() as demo:
    name = gr.Textbox()
    output = gr.Textbox()
    name.submit(fn=greet, inputs=name, outputs=output)
```

---

## 🎨 页面元素

### 标题和描述

```python
with gr.Blocks() as demo:
    # 主标题
    gr.Markdown("# 我的应用")
    
    # 描述文字
    gr.Markdown("""
    这是一个功能强大的应用，支持：
    - 文本处理
    - 图像分析
    - 音频转录
    """)
    
    # 分隔线
    gr.HTML("<hr>")
```

### 组件标题

```python
with gr.Blocks() as demo:
    # 带标签的组件
    text = gr.Textbox(label="输入")
    
    # 不带标签
    text2 = gr.Textbox(show_label=False)
    
    # 自定义标题样式
    gr.Markdown("### 高级设置")
```

---

## 📐 复杂布局

### 响应式布局

使用 `scale` 控制组件的相对宽度：

```python
with gr.Blocks() as demo:
    with gr.Row():
        # 左侧占 1/3，右侧占 2/3
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="输入", lines=10)
            submit_btn = gr.Button("提交", variant="primary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="输出", lines=10)
```

### 多行布局

```python
with gr.Blocks() as demo:
    gr.Markdown("# 图像处理工具")
    
    # 第一行：输入
    with gr.Row():
        input_image = gr.Image(label="原始图像")
        mask_image = gr.Image(label="蒙版")
    
    # 第二行：控制面板
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["增强", "修复", "风格化"], label="处理模式")
            strength = gr.Slider(0, 1, value=0.5, label="强度")
        
        with gr.Column():
            process_btn = gr.Button("处理", variant="primary")
            clear_btn = gr.Button("清空")
    
    # 第三行：输出
    output_image = gr.Image(label="处理结果")
```

### 标签页布局

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("文本处理"):
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="处理结果")
        
        with gr.Tab("图像处理"):
            image_input = gr.Image(label="上传图像")
            image_output = gr.Image(label="处理结果")
        
        with gr.Tab("音频处理"):
            audio_input = gr.Audio(label="上传音频")
            audio_output = gr.Audio(label="处理结果")
```

---

## 🔗 事件处理

### 点击事件

```python
with gr.Blocks() as demo:
    name = gr.Textbox(label="姓名")
    output = gr.Textbox(label="结果")
    
    btn = gr.Button("点击我")
    btn.click(
        fn=greet,
        inputs=name,
        outputs=output
    )
```

### 多种触发方式

```python
with gr.Blocks() as demo:
    text = gr.Textbox(label="输入")
    output = gr.Textbox(label="输出")
    
    # 方式1：按钮点击
    btn = gr.Button("提交")
    btn.click(fn=process, inputs=text, outputs=output)
    
    # 方式2：回车提交
    text.submit(fn=process, inputs=text, outputs=output)
    
    # 方式3：内容变化时触发
    text.change(fn=process, inputs=text, outputs=output)
```

### 可用事件类型

| 组件 | 可用事件 |
|------|----------|
| `Textbox` | `change`, `input`, `submit` |
| `Button` | `click` |
| `Dropdown` | `change`, `input` |
| `Slider` | `change`, `release` |
| `Image` | `change`, `clear`, `upload`, `edit` |
| `Audio` | `change`, `clear`, `upload`, `play`, `pause`, `stop` |
| `Chatbot` | `change`, `select` |

---

## 🔄 组件交互

### 联动更新

一个组件的变化影响另一个组件：

```python
import gradio as gr

def update_text(choice):
    if choice == "选项A":
        return "你选择了 A"
    elif choice == "选项B":
        return "你选择了 B"
    return "请选择"

with gr.Blocks() as demo:
    dropdown = gr.Dropdown(["选项A", "选项B"], label="选择")
    text = gr.Textbox(label="结果")
    
    # 下拉选择变化时更新文本
    dropdown.change(
        fn=update_text,
        inputs=dropdown,
        outputs=text
    )
```

### 动态组件更新

根据条件改变组件的属性：

```python
import gradio as gr

def toggle_inputs(mode):
    if mode == "文本":
        return gr.Textbox(visible=True), gr.Image(visible=False)
    else:
        return gr.Textbox(visible=False), gr.Image(visible=True)

with gr.Blocks() as demo:
    mode = gr.Radio(["文本", "图像"], label="输入类型")
    
    text_input = gr.Textbox(label="文本输入", visible=True)
    image_input = gr.Image(label="图像输入", visible=False)
    
    mode.change(
        fn=toggle_inputs,
        inputs=mode,
        outputs=[text_input, image_input]
    )
```

### 更新组件属性

使用 `gr.update()` 动态更新组件：

```python
import gradio as gr

def update_dropdown(selected):
    if selected == "水果":
        return gr.Dropdown(choices=["苹果", "香蕉", "橙子"])
    elif selected == "蔬菜":
        return gr.Dropdown(choices=["白菜", "萝卜", "土豆"])
    return gr.Dropdown(choices=[])

with gr.Blocks() as demo:
    category = gr.Radio(["水果", "蔬菜"], label="类别")
    item = gr.Dropdown(label="项目")
    
    category.change(
        fn=update_dropdown,
        inputs=category,
        outputs=item
    )
```

---

## 🧠 状态管理

### 会话状态

使用 `gr.State` 保存会话状态：

```python
import gradio as gr

def counter(action, count):
    if action == "增加":
        count += 1
    elif action == "减少":
        count -= 1
    return count, f"当前计数：{count}"

with gr.Blocks() as demo:
    gr.Markdown("### 计数器")
    
    count_state = gr.State(value=0)  # 状态变量
    
    with gr.Row():
        add_btn = gr.Button("增加")
        sub_btn = gr.Button("减少")
    
    display = gr.Textbox(label="计数", value="当前计数：0")
    
    add_btn.click(
        fn=lambda c: counter("增加", c),
        inputs=[count_state],
        outputs=[count_state, display]
    )
    
    sub_btn.click(
        fn=lambda c: counter("减少", c),
        inputs=[count_state],
        outputs=[count_state, display]
    )
```

### 聊天历史

```python
import gradio as gr

def chat(message, history):
    # history 是对话历史列表
    history.append({"role": "user", "content": message})
    response = f"回复：{message}"
    history.append({"role": "assistant", "content": response})
    return history, ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="对话", type="messages")
    msg = gr.Textbox(label="输入", placeholder="输入消息...")
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
```

---

## 🤖 LLM 集成示例

### 聊天机器人

```python
import gradio as gr
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def respond(message, history):
    # 构建消息列表
    messages = [{"role": "system", "content": "你是一个友好的助手。"}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    # 调用 API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown("# AI 聊天助手")
    
    chatbot = gr.Chatbot(label="对话")
    msg = gr.Textbox(label="输入消息", placeholder="输入后按回车发送...")
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
```

### 带参数的聊天

```python
import gradio as gr

def chat_with_params(message, history, model, temperature, max_tokens):
    # 根据 model, temperature, max_tokens 进行处理
    response = f"[{model}] 回复：{message}"
    history.append((message, response))
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# 可配置的聊天机器人")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=400)
            msg = gr.Textbox(label="输入", placeholder="输入消息...")
        
        with gr.Column(scale=1):
            model = gr.Dropdown(
                ["gpt-3.5-turbo", "gpt-4", "claude-3"],
                label="模型",
                value="gpt-3.5-turbo"
            )
            temperature = gr.Slider(0, 2, value=0.7, label="温度")
            max_tokens = gr.Slider(100, 4000, value=1000, label="最大长度")
            clear_btn = gr.Button("清空对话")
    
    msg.submit(
        fn=chat_with_params,
        inputs=[msg, chatbot, model, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch()
```

### 流式输出

```python
import gradio as gr
import time

def stream_response(message, history):
    # 模拟流式响应
    response = ""
    for char in "这是一个流式输出的示例回复，字符会逐个显示。":
        response += char
        time.sleep(0.05)  # 模拟延迟
        yield history + [(message, response)]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    
    msg.submit(
        fn=stream_response,
        inputs=[msg, chatbot],
        outputs=chatbot
    )
```

---

## 🔲 高级组件

### Chatbot 组件

```python
import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        label="对话",
        height=400,
        show_copy_button=True,    # 显示复制按钮
        avatar_images=(None, None),  # 自定义头像
        bubble_full_width=False
    )
```

### Gallery 组件

```python
import gradio as gr

with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="图像画廊",
        show_label=True,
        columns=3,           # 每行显示3张
        rows=2,              # 显示2行
        height="auto",
        allow_preview=True   # 点击预览
    )
```

### DataFrame 组件

```python
import gradio as gr
import pandas as pd

def show_data():
    return pd.DataFrame({
        "姓名": ["张三", "李四", "王五"],
        "年龄": [25, 30, 28],
        "城市": ["北京", "上海", "广州"]
    })

with gr.Blocks() as demo:
    df = gr.DataFrame(label="数据表格", show_search="filter")
    btn = gr.Button("加载数据")
    btn.click(fn=show_data, outputs=df)
```

---

## 📋 完整示例：多功能工具箱

```python
import gradio as gr

# 文本处理函数
def process_text(text, operation):
    if operation == "大写":
        return text.upper()
    elif operation == "小写":
        return text.lower()
    elif operation == "反转":
        return text[::-1]
    return text

# 图像处理函数
def process_image(image, filter_type):
    # 这里可以实现图像滤镜
    return image

# 主界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ 多功能工具箱")
    
    with gr.Tabs():
        # 文本工具
        with gr.Tab("📝 文本工具"):
            with gr.Row():
                text_input = gr.Textbox(label="输入文本", lines=5)
                text_output = gr.Textbox(label="处理结果", lines=5)
            
            with gr.Row():
                text_operation = gr.Radio(
                    ["大写", "小写", "反转"],
                    label="操作"
                )
                text_btn = gr.Button("处理", variant="primary")
            
            text_btn.click(
                fn=process_text,
                inputs=[text_input, text_operation],
                outputs=text_output
            )
        
        # 图像工具
        with gr.Tab("🖼️ 图像工具"):
            with gr.Row():
                image_input = gr.Image(label="上传图像")
                image_output = gr.Image(label="处理结果")
            
            with gr.Row():
                filter_type = gr.Dropdown(
                    ["原图", "灰度", "模糊", "锐化"],
                    label="滤镜"
                )
                image_btn = gr.Button("应用", variant="primary")
            
            image_btn.click(
                fn=process_image,
                inputs=[image_input, filter_type],
                outputs=image_output
            )
        
        # 设置
        with gr.Tab("⚙️ 设置"):
            gr.Markdown("### 主题设置")
            theme_info = gr.Textbox(value="当前主题：Soft", label="主题信息")

demo.launch()
```

---

## 🔗 相关章节

- [基础组件与布局](./gradio-basics.md) - 了解各类组件
- [Interface 快速构建](./gradio-interface.md) - 快速创建简单界面
- [高级特性与部署](./gradio-advanced.md) - 流式输出、部署等高级功能
