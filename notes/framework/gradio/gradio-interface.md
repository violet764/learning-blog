# Gradio Interface 快速构建

`gr.Interface` 是 Gradio 最简单易用的 API，只需定义处理函数和输入输出类型，即可快速创建一个完整的 Web 界面。非常适合快速原型开发和简单应用场景。

---

## 📌 Interface 基础

### 最简示例

```python
import gradio as gr

def greet(name):
    return f"你好，{name}！"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()
```

### Interface 参数详解

```python
demo = gr.Interface(
    fn=process,              # 处理函数
    inputs="text",           # 输入组件（可以是字符串或组件对象）
    outputs="text",          # 输出组件
    title="应用标题",         # 界面标题
    description="应用描述",   # 描述文字
    article="更多信息...",    # 底部文章内容（支持 Markdown）
    examples=[["示例1"], ["示例2"]],  # 示例输入
    cache_examples=True,     # 缓存示例结果
    theme=gr.themes.Soft(),  # 主题
    allow_flagging="manual", # 允许标记（用于收集反馈）
    flagging_options=["好", "中", "差"]
)
```

---

## 📝 输入输出配置

### 字符串简写

Gradio 支持用字符串简写来指定组件类型：

```python
# 输入简写
inputs="text"      # Textbox
inputs="number"    # Number
inputs="image"     # Image
inputs="audio"     # Audio
inputs="video"     # Video
inputs="file"      # File
inputs="checkbox"  # Checkbox
inputs="dropdown"  # Dropdown（需要额外配置）

# 输出简写
outputs="text"     # Textbox
outputs="image"    # Image
outputs="audio"    # Audio
outputs="video"    # Video
outputs="label"    # Label（分类结果）
outputs="json"     # JSON
outputs="html"     # HTML
```

### 组件对象配置

使用组件对象可以获得更多配置选项：

```python
import gradio as gr

def analyze(text, language):
    return f"分析 {language} 文本：{text}"

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="输入文本", lines=5, placeholder="请输入..."),
        gr.Dropdown(choices=["中文", "英文", "日文"], label="语言")
    ],
    outputs=gr.Textbox(label="分析结果"),
    title="文本分析工具"
)
```

---

## 🔄 多输入输出

### 多输入示例

```python
import gradio as gr

def calculate(name, age, is_student):
    status = "学生" if is_student else "非学生"
    return f"{name}，{age}岁，{status}"

demo = gr.Interface(
    fn=calculate,
    inputs=[
        gr.Textbox(label="姓名"),
        gr.Number(label="年龄", minimum=0, maximum=150),
        gr.Checkbox(label="是否为学生")
    ],
    outputs="text"
)
```

### 多输出示例

```python
import gradio as gr

def analyze_image(image):
    # 模拟图像分析
    size = image.size if hasattr(image, 'size') else (0, 0)
    mode = image.mode if hasattr(image, 'mode') else "unknown"
    
    # 返回多个结果
    return (
        f"尺寸：{size[0]}x{size[1]}",  # 文本结果
        {"图像": 0.9, "非图像": 0.1},  # 分类标签
        image  # 处理后的图像
    )

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="图像信息"),
        gr.Label(label="分类结果"),
        gr.Image(label="原图预览")
    ]
)
```

---

## 📚 示例功能

### 添加示例

示例可以帮助用户快速了解如何使用应用：

```python
import gradio as gr

def translate(text, source_lang, target_lang):
    # 模拟翻译
    return f"[{source_lang}→{target_lang}] {text}"

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="输入文本"),
        gr.Dropdown(["中文", "英文", "日文"], label="源语言"),
        gr.Dropdown(["中文", "英文", "日文"], label="目标语言", value="英文")
    ],
    outputs=gr.Textbox(label="翻译结果"),
    examples=[
        ["你好世界", "中文", "英文"],
        ["Hello World", "英文", "中文"],
        ["こんにちは", "日文", "中文"],
    ]
)
```

### 从文件加载示例

```python
# examples.csv 内容：
# text,source,target
# 你好世界,中文,英文
# Hello World,英文,中文

demo = gr.Interface(
    fn=translate,
    inputs=[...],
    outputs=...,
    examples="examples.csv"  # 从 CSV 加载
)
```

---

## 🎯 实用场景示例

### 图像分类

```python
import gradio as gr
from transformers import pipeline

# 加载模型
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify_image(image):
    result = classifier(image)
    return {item["label"]: item["score"] for item in result}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="上传图片"),
    outputs=gr.Label(num_top_classes=5, label="分类结果"),
    title="图像分类",
    description="使用 ViT 模型对图像进行分类",
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"]
    ]
)

demo.launch()
```

### 文本情感分析

```python
import gradio as gr
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label_map = {"LABEL_0": "负面", "LABEL_1": "中性", "LABEL_2": "正面"}
    return {label_map[result["label"]]: result["score"]}

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="输入文本", lines=3, placeholder="输入要分析的文本..."),
    outputs=gr.Label(label="情感分析结果"),
    title="情感分析",
    examples=[
        ["今天天气真好，心情愉快！"],
        ["这个产品太差了，很失望。"],
        ["今天去上班，没什么特别的。"]
    ]
)
```

### 文本生成（接入 LLM）

```python
import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length, temperature):
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1
    )
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="提示词", lines=2),
        gr.Slider(10, 500, value=100, step=10, label="最大长度"),
        gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="温度")
    ],
    outputs=gr.Textbox(label="生成结果", lines=10),
    title="文本生成",
    description="使用 GPT-2 模型生成文本"
)
```

### 语音识别

```python
import gradio as gr
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe(audio):
    if audio is None:
        return "请上传或录制音频"
    result = asr(audio)
    return result["text"]

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=gr.Textbox(label="转录结果", lines=5),
    title="语音识别",
    description="使用 Whisper 模型将语音转换为文字"
)
```

---

## ⚙️ 高级配置

### 缓存示例结果

```python
demo = gr.Interface(
    fn=process,
    inputs=...,
    outputs=...,
    examples=[...],
    cache_examples=True,      # 缓存示例的计算结果
    cache_mode="lazy"         # lazy 或 eager
)
```

### 允许用户标记

收集用户反馈用于改进模型：

```python
demo = gr.Interface(
    fn=process,
    inputs=...,
    outputs=...,
    allow_flagging="manual",  # manual 或 auto
    flagging_options=["正确", "错误", "不确定"],
    flagging_dir="flagged"    # 保存目录
)
```

### 并发控制

```python
demo = gr.Interface(
    fn=process,
    inputs=...,
    outputs=...,
    concurrency_limit=5,      # 同时处理的最大请求数
    concurrency_id="default"
)
```

### 队列设置

```python
demo = gr.Interface(...)

# 启用队列（用于长时间处理）
demo.queue(
    max_size=20,              # 最大排队数
    default_concurrency_limit=5
)

demo.launch()
```

---

## 🚀 启动配置

### 基本启动

```python
demo.launch()  # 默认：本地运行，自动打开浏览器
```

### 自定义端口和地址

```python
demo.launch(
    server_name="0.0.0.0",    # 允许外部访问
    server_port=7860,         # 端口号
    share=False               # 是否创建公开链接
)
```

### 创建公开链接

```python
demo.launch(
    share=True,               # 创建一个公开的 xxxxx.gradio.live 链接
    share_server_address=None # 自定义分享服务器
)
```

### 调试模式

```python
demo.launch(
    debug=True,               # 启用调试模式
    show_error=True           # 显示详细错误信息
)
```

### 嵌入模式

用于将 Gradio 界面嵌入到其他网站：

```python
demo.launch(
    inline=True,              # 在 notebook 中内嵌显示
    height=800,               # 内嵌高度
    width="100%"              # 内嵌宽度
)
```

---

## 📋 Interface vs Blocks 对比

| 特性 | Interface | Blocks |
|------|-----------|--------|
| **学习曲线** | 简单 | 中等 |
| **代码量** | 少 | 较多 |
| **布局灵活性** | 固定 | 自由 |
| **组件交互** | 有限 | 完全支持 |
| **状态管理** | 不支持 | 支持 |
| **适用场景** | 简单应用、原型 | 复杂应用、生产环境 |

### 何时使用 Interface

- ✅ 单一功能，输入输出固定
- ✅ 快速原型验证
- ✅ 学术演示
- ✅ 学习和教学

### 何时迁移到 Blocks

- ❌ 需要复杂布局
- ❌ 需要组件间的动态交互
- ❌ 需要会话状态管理
- ❌ 需要多步骤处理流程

---

## 🔗 相关章节

- [基础组件与布局](./gradio-basics.md) - 详细了解各类组件
- [Blocks 灵活布局](./gradio-blocks.md) - 构建复杂交互界面
- [高级特性与部署](./gradio-advanced.md) - 生产部署指南
