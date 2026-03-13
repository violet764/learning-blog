# Gradio 基础组件与布局

Gradio 提供了丰富的组件库，涵盖文本、图像、音频、视频等多种数据类型。掌握这些组件及其布局方式，是构建美观实用的 Web 界面的基础。

---

## 📌 输入组件概览

Gradio 的输入组件用于接收用户数据，每种组件都针对特定的数据类型进行了优化。

### 组件分类

```
输入组件
├── 文本类
│   ├── Textbox（文本框）
│   ├── Number（数字）
│   └── Code（代码编辑器）
├── 选择类
│   ├── Dropdown（下拉选择）
│   ├── Checkbox（复选框）
│   ├── Radio（单选按钮）
│   └── Slider（滑块）
├── 媒体类
│   ├── Image（图像上传）
│   ├── Audio（音频上传/录制）
│   ├── Video（视频上传）
│   └── File（文件上传）
└── 特殊类
    ├── Dataframe（表格）
    ├── ColorPicker（颜色选择）
    └── DateTime（日期时间）
```

---

## 📝 文本类组件

### Textbox - 文本框

最基本的文本输入组件，支持单行和多行模式。

```python
import gradio as gr

# 单行文本
single_line = gr.Textbox(
    label="用户名",
    placeholder="请输入用户名",
    max_lines=1
)

# 多行文本
multi_line = gr.Textbox(
    label="输入内容",
    placeholder="请输入详细描述...",
    lines=5,          # 显示行数
    max_lines=10,     # 最大行数
    show_copy_button=True  # 显示复制按钮
)

# 密码输入
password = gr.Textbox(
    label="密码",
    type="password"   # 密码模式
)
```

### Number - 数字输入

专门用于数值输入，支持设置范围和步长。

```python
number_input = gr.Number(
    label="年龄",
    value=25,          # 默认值
    minimum=0,         # 最小值
    maximum=150,       # 最大值
    step=1,            # 步长
    precision=0        # 小数位数
)

# 浮点数输入
float_input = gr.Number(
    label="价格",
    value=99.99,
    precision=2
)
```

### Code - 代码编辑器

带有语法高亮的代码输入组件。

```python
code_input = gr.Code(
    label="Python 代码",
    language="python",     # 语言类型
    lines=10,              # 显示行数
    show_copy_button=True  # 复制按钮
)

# 支持的语言：python, javascript, html, css, sql, json 等
```

---

## 🎚️ 选择类组件

### Dropdown - 下拉选择

```python
dropdown = gr.Dropdown(
    label="选择模型",
    choices=["GPT-4", "Claude", "LLaMA", "Qwen"],
    value="GPT-4",           # 默认选中
    multiselect=False,       # 是否多选
    allow_custom_value=True  # 允许自定义输入
)

# 多选下拉
multi_dropdown = gr.Dropdown(
    label="选择功能",
    choices=["翻译", "摘要", "问答", "代码生成"],
    multiselect=True
)
```

### Radio - 单选按钮

```python
radio = gr.Radio(
    label="选择语言",
    choices=["中文", "英文", "日文"],
    value="中文",
    interactive=True
)
```

### Checkbox - 复选框

```python
# 单个复选框
checkbox = gr.Checkbox(
    label="同意用户协议",
    value=False
)

# 复选框组
checkbox_group = gr.CheckboxGroup(
    label="选择特性",
    choices=["流式输出", "历史记录", "多轮对话"],
    value=["流式输出"]
)
```

### Slider - 滑块

```python
slider = gr.Slider(
    label="温度参数",
    minimum=0,
    maximum=2,
    value=0.7,
    step=0.1,
    info="控制输出的随机性"
)

# 范围滑块
range_slider = gr.Slider(
    label="选择范围",
    minimum=0,
    maximum=100,
    value=[20, 80],  # 选择范围
    interactive=True
)
```

---

## 🖼️ 媒体类组件

### Image - 图像组件

图像上传和显示组件，支持多种输入输出格式。

```python
# 基础图像上传
image_input = gr.Image(
    label="上传图片",
    type="pil",           # 返回类型：pil, numpy, filepath
    height=300,           # 显示高度
    width=400             # 显示宽度
)

# 支持摄像头
webcam = gr.Image(
    label="拍照",
    sources=["webcam"],   # 来源：webcam, upload, clipboard
    type="pil"
)

# 图像编辑器
image_editor = gr.ImageEditor(
    label="编辑图片",
    type="pil",
    brush=gr.Brush(colors=["#ff0000", "#00ff00"])
)

# 输出图像
image_output = gr.Image(
    label="处理结果",
    type="pil",
    show_download_button=True
)
```

### Audio - 音频组件

```python
# 音频上传
audio_input = gr.Audio(
    label="上传音频",
    type="filepath",      # 返回类型：filepath, numpy
    sources=["upload", "microphone"]
)

# 仅麦克风录制
mic = gr.Audio(
    label="录音",
    sources=["microphone"],
    type="numpy"
)

# 音频输出
audio_output = gr.Audio(
    label="播放音频",
    autoplay=True
)
```

### Video - 视频组件

```python
video_input = gr.Video(
    label="上传视频",
    sources=["upload", "webcam"]
)

video_output = gr.Video(
    label="处理结果",
    autoplay=False
)
```

### File - 文件上传

```python
file_input = gr.File(
    label="上传文件",
    file_types=[".pdf", ".docx", ".txt"],  # 允许的文件类型
    file_count="multiple"   # single, multiple, directory
)
```

---

## 📊 输出组件

输出组件用于展示处理结果，与输入组件类似但侧重于展示。

### 常用输出组件

```python
# 文本输出
text_output = gr.Textbox(label="结果", lines=5)

# Markdown 输出（支持格式化）
markdown_output = gr.Markdown()

# JSON 输出
json_output = gr.JSON(label="结构化数据")

# 标签输出（分类结果）
label_output = gr.Label(label="分类结果", num_top_classes=3)

# 图表输出
plot_output = gr.Plot(label="数据可视化")

# HTML 输出
html_output = gr.HTML()
```

### Label - 分类标签

常用于展示分类模型的输出：

```python
import gradio as gr

def classify():
    return {
        "猫": 0.92,
        "狗": 0.05,
        "鸟": 0.03
    }

# 返回格式：{类别名: 概率}
label_output = gr.Label(
    label="预测结果",
    num_top_classes=3  # 显示前 N 个结果
)
```

---

## 📐 布局容器

Gradio 提供了多种布局容器来组织组件。

### Row - 行布局

将组件水平排列：

```python
with gr.Row():
    input_text = gr.Textbox(label="输入")
    output_text = gr.Textbox(label="输出")
```

### Column - 列布局

将组件垂直排列：

```python
with gr.Column():
    name = gr.Textbox(label="姓名")
    age = gr.Number(label="年龄")
    email = gr.Textbox(label="邮箱")
```

### 嵌套布局

行列布局可以嵌套使用：

```python
with gr.Row():
    with gr.Column(scale=1):   # scale 控制宽度比例
        input_image = gr.Image(label="输入图像")
    with gr.Column(scale=2):
        with gr.Row():
            output_image1 = gr.Image(label="输出1")
            output_image2 = gr.Image(label="输出2")
        result_text = gr.Textbox(label="分析结果")
```

### Tabs - 标签页

创建多标签界面：

```python
with gr.Tabs():
    with gr.Tab("图像分类"):
        gr.Image(label="上传图片")
        gr.Label(label="分类结果")
    
    with gr.Tab("目标检测"):
        gr.Image(label="上传图片")
        gr.Image(label="检测结果")
    
    with gr.Tab("图像分割"):
        gr.Image(label="上传图片")
        gr.Image(label="分割结果")
```

### Group - 组件分组

视觉上将相关组件分组：

```python
with gr.Group():
    gr.Textbox(label="标题")
    gr.Textbox(label="内容", lines=5)
```

### Accordion - 折叠面板

可折叠的内容区域：

```python
with gr.Accordion("高级选项", open=False):
    temperature = gr.Slider(0, 2, value=0.7, label="温度")
    max_tokens = gr.Slider(1, 4096, value=512, label="最大长度")
```

---

## 🎨 样式定制

### 主题选择

Gradio 内置多种主题：

```python
import gradio as gr

# 可用主题
# gr.themes.Default()
# gr.themes.Soft()
# gr.themes.Monochrome()
# gr.themes.Glass()
# gr.themes.Origin()

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    theme=gr.themes.Soft()  # 使用 Soft 主题
)
```

### 自定义主题

```python
theme = gr.themes.Soft(
    primary_hue="blue",      # 主色调
    secondary_hue="slate",   # 次要色调
    neutral_hue="gray",      # 中性色调
).set(
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white"
)

demo = gr.Interface(..., theme=theme)
```

### CSS 定制

```python
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    css="""
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-text {
        background-color: #f0f0f0;
    }
    """
)
```

---

## 💡 实用示例

### 完整的表单示例

```python
import gradio as gr

def process_form(name, age, gender, interests, bio, avatar):
    result = f"""
    ### 用户信息
    
    - **姓名**: {name}
    - **年龄**: {age}
    - **性别**: {gender}
    - **兴趣**: {', '.join(interests)}
    - **简介**: {bio}
    """
    return result, avatar

with gr.Blocks() as demo:
    gr.Markdown("# 用户注册表单")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="姓名", placeholder="请输入姓名")
            age = gr.Number(label="年龄", minimum=0, maximum=150)
            gender = gr.Radio(["男", "女", "其他"], label="性别")
            interests = gr.CheckboxGroup(
                ["编程", "阅读", "运动", "音乐", "旅行"],
                label="兴趣爱好"
            )
            bio = gr.Textbox(label="个人简介", lines=3)
            avatar = gr.Image(label="头像", type="pil")
        
        with gr.Column():
            output_text = gr.Markdown(label="注册信息")
            output_avatar = gr.Image(label="头像预览")
    
    submit = gr.Button("提交", variant="primary")
    submit.click(
        process_form,
        inputs=[name, age, gender, interests, bio, avatar],
        outputs=[output_text, output_avatar]
    )

demo.launch()
```

---

## 📋 组件速查表

| 组件 | 用途 | 常用参数 |
|------|------|----------|
| `Textbox` | 文本输入 | `lines`, `max_lines`, `placeholder` |
| `Number` | 数字输入 | `minimum`, `maximum`, `step` |
| `Dropdown` | 下拉选择 | `choices`, `multiselect` |
| `Radio` | 单选按钮 | `choices`, `value` |
| `Checkbox` | 复选框 | `value`, `label` |
| `Slider` | 滑块 | `minimum`, `maximum`, `step` |
| `Image` | 图像 | `type`, `sources`, `height` |
| `Audio` | 音频 | `sources`, `type` |
| `Video` | 视频 | `sources` |
| `File` | 文件 | `file_types`, `file_count` |
| `Markdown` | Markdown 显示 | `value` |
| `JSON` | JSON 显示 | - |
| `Label` | 分类结果 | `num_top_classes` |
| `Plot` | 图表 | - |

---

## 🔗 相关章节

- [Interface 快速构建](./gradio-interface.md) - 使用 Interface API 快速创建界面
- [Blocks 灵活布局](./gradio-blocks.md) - 使用 Blocks 构建复杂应用
- [高级特性与部署](./gradio-advanced.md) - 部署和高级功能
