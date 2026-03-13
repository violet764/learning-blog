# Pipeline 快速使用

Pipeline 是 Transformers 库最便捷的功能，它封装了预处理、模型推理和后处理的完整流程，让用户无需编写任何模型代码就能使用预训练模型进行推理。

---

## 🌟 Pipeline 简介

### 什么是 Pipeline

Pipeline 是一个端到端的推理管道，它自动处理：

```
原始输入 → 分词器编码 → 模型推理 → 结果解析 → 最终输出
```

### 基本用法

```python
from transformers import pipeline

# 创建 Pipeline（首次使用会自动下载模型）
classifier = pipeline("sentiment-analysis")

# 单条推理
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 批量推理
results = classifier([
    "This is a great day!",
    "I hate waiting in line."
])
print(results)
# [{'label': 'POSITIVE', 'score': 0.9999}, {'label': 'NEGATIVE', 'score': 0.9997}]
```

---

## 📋 内置 Pipeline 类型

Transformers 提供了多种开箱即用的 Pipeline：

### 文本分类

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("这个产品真的很棒！")

# 多标签分类
classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")
result = classifier("I'm feeling great today!")
```

### 命名实体识别（NER）

```python
# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
result = ner("张三在北京的清华大学学习人工智能。")
# [{'entity_group': 'PER', 'score': 0.99, 'word': '张三', 'start': 0, 'end': 2},
#  {'entity_group': 'LOC', 'score': 0.99, 'word': '北京', 'start': 3, 'end': 5},
#  {'entity_group': 'ORG', 'score': 0.99, 'word': '清华大学', 'start': 6, 'end': 10}]
```

### 问答系统

```python
# 抽取式问答
qa = pipeline("question-answering")
result = qa(
    question="什么是深度学习？",
    context="深度学习是机器学习的一个子领域，它使用神经网络来学习数据的表示。"
)
print(result)
# {'answer': '机器学习的一个子领域', 'score': 0.85, 'start': 5, 'end': 14}
```

### 文本生成

```python
# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=2
)
# [{'generated_text': 'Once upon a time, there lived a...'}, ...]
```

### 文本摘要

```python
# 文本摘要
summarizer = pipeline("summarization")
text = """
深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次表示。
深度学习在计算机视觉、自然语言处理、语音识别等领域取得了巨大成功。
"""
result = summarizer(text, max_length=30, min_length=10)
print(result[0]['summary_text'])
```

### 机器翻译

```python
# 英译中
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
# 你好，你好吗？

# 中译英
translator = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
result = translator("你好，世界！")
```

### 零样本分类

```python
# 零样本分类（无需训练数据）
classifier = pipeline("zero-shot-classification")
result = classifier(
    "这家餐厅的服务态度非常好",
    candidate_labels=["服务", "食物", "环境", "价格"]
)
print(result)
# {'sequence': '这家餐厅的服务态度非常好', 
#  'labels': ['服务', '环境', '食物', '价格'], 
#  'scores': [0.85, 0.08, 0.05, 0.02]}
```

---

## 🎛️ Pipeline 配置

### 指定模型

```python
from transformers import pipeline

# 使用默认模型
classifier = pipeline("sentiment-analysis")

# 指定模型名称
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# 使用中文情感分析模型
classifier = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)
```

### 设备配置

```python
import torch

# 使用 GPU
classifier = pipeline("sentiment-analysis", device=0)

# 使用多个 GPU（数据并行）
classifier = pipeline("sentiment-analysis", device_map="auto")

# 自动选择设备
classifier = pipeline("sentiment-analysis", device="cuda" if torch.cuda.is_available() else "cpu")
```

### 批处理配置

```python
# 配置批处理大小
classifier = pipeline(
    "sentiment-analysis",
    batch_size=8,  # 批量处理大小
    max_length=512,  # 最大序列长度
    truncation=True  # 超长截断
)

# 批量推理
texts = ["文本1", "文本2", "文本3", ...]  # 大量文本
results = classifier(texts)  # 自动分批处理
```

---

## 📊 Pipeline 类型一览表

| Pipeline 名称 | 任务类型 | 典型模型 | 输入示例 |
|---------------|----------|----------|----------|
| `sentiment-analysis` | 情感分析 | distilbert-sst2 | 文本字符串 |
| `text-classification` | 文本分类 | 通用分类模型 | 文本字符串 |
| `ner` | 命名实体识别 | bert-base-NER | 文本字符串 |
| `question-answering` | 问答系统 | distilbert-qa | question + context |
| `text-generation` | 文本生成 | gpt2 | 提示文本 |
| `summarization` | 文本摘要 | bart-large-cnn | 长文本 |
| `translation_xx_to_yy` | 机器翻译 | opus-mt 系列 | 文本字符串 |
| `zero-shot-classification` | 零样本分类 | bart-large-mnli | text + labels |
| `fill-mask` | 掩码填充 | bert-base | 带 [MASK] 的文本 |
| `feature-extraction` | 特征提取 | 通用模型 | 文本字符串 |

---

## 🔧 高级用法

### 自定义分词器参数

```python
classifier = pipeline("sentiment-analysis")

# 传入分词器参数
result = classifier(
    "这是一段很长的文本...",
    truncation=True,
    max_length=512,
    padding="max_length"
)
```

### 流式输出

```python
# 文本生成流式输出
generator = pipeline("text-generation", model="gpt2")

for output in generator(
    "Hello, I am",
    max_new_tokens=50,
    do_sample=True,
    return_full_text=False,
    stream=True  # 流式输出
):
    print(output['generated_text'], end="", flush=True)
```

### 异步推理

```python
from transformers import pipeline
import asyncio

async def async_inference():
    classifier = pipeline("sentiment-analysis")
    
    # 异步批量推理
    texts = ["文本1", "文本2", "文本3"]
    results = await classifier.ainvoke(texts)
    return results
```

### 返回 Tensor

```python
# 返回 PyTorch Tensor
pipe = pipeline("feature-extraction", return_tensors="pt")
features = pipe("Hello world")
print(features.shape)  # torch.Size([1, seq_len, hidden_size])
```

---

## 🎯 实践示例

### 示例1：构建评论分析器

```python
from transformers import pipeline

class ReviewAnalyzer:
    """评论情感分析器"""
    
    def __init__(self, model_name="lxyuan/distilbert-base-multilingual-cased-sentiments-student"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)
    
    def analyze(self, text):
        """分析单条评论"""
        result = self.classifier(text)[0]
        return {
            "text": text,
            "sentiment": result['label'],
            "confidence": round(result['score'], 4)
        }
    
    def batch_analyze(self, texts):
        """批量分析评论"""
        results = self.classifier(texts)
        return [
            {
                "text": text,
                "sentiment": r['label'],
                "confidence": round(r['score'], 4)
            }
            for text, r in zip(texts, results)
        ]

# 使用示例
analyzer = ReviewAnalyzer()

# 单条分析
print(analyzer.analyze("这个产品质量很好，发货也快！"))
# {'text': '...', 'sentiment': 'positive', 'confidence': 0.9987}

# 批量分析
reviews = [
    "非常满意，物超所值！",
    "质量一般，凑合能用",
    "太差了，再也不买了"
]
print(analyzer.batch_analyze(reviews))
```

### 示例2：智能问答助手

```python
from transformers import pipeline

class QAAssistant:
    """智能问答助手"""
    
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.qa = pipeline("question-answering", model=model_name)
        self.contexts = []
    
    def add_context(self, text):
        """添加知识库文本"""
        self.contexts.append(text)
    
    def ask(self, question):
        """提问"""
        if not self.contexts:
            return "请先添加知识库内容"
        
        # 在所有上下文中搜索最佳答案
        best_answer = None
        best_score = 0
        
        for context in self.contexts:
            result = self.qa(question=question, context=context)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result
        
        return {
            "question": question,
            "answer": best_answer['answer'],
            "confidence": round(best_answer['score'], 4)
        }

# 使用示例
assistant = QAAssistant()

# 添加知识库
assistant.add_context("""
Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。
Python 的设计哲学强调代码的可读性和简洁性。
Python 广泛应用于 Web 开发、数据科学、人工智能等领域。
""")

# 提问
print(assistant.ask("Python 是什么时候发布的？"))
# {'question': '...', 'answer': '1991 年', 'confidence': 0.95}
```

### 示例3：多语言翻译管道

```python
from transformers import pipeline

class MultiTranslator:
    """多语言翻译器"""
    
    def __init__(self):
        self.translators = {}
        self.model_map = {
            ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
            ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
            ("en", "ja"): "Helsinki-NLP/opus-mt-en-ja",
            ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
        }
    
    def translate(self, text, src_lang, tgt_lang):
        """翻译文本"""
        key = (src_lang, tgt_lang)
        
        if key not in self.model_map:
            raise ValueError(f"不支持 {src_lang} 到 {tgt_lang} 的翻译")
        
        if key not in self.translators:
            model = self.model_map[key]
            self.translators[key] = pipeline("translation", model=model)
        
        result = self.translators[key](text)
        return result[0]['translation_text']

# 使用示例
translator = MultiTranslator()

print(translator.translate("Hello, world!", "en", "zh"))  # 你好，世界！
print(translator.translate("你好，世界！", "zh", "en"))  # Hello, world!
```

---

## 📝 Pipeline 参数速查

### 通用参数

```python
pipeline(
    task,                    # 任务类型
    model=None,              # 模型名称或路径
    tokenizer=None,          # 分词器
    device=None,             # 设备 ID
    device_map=None,         # 设备映射（大模型）
    torch_dtype=None,        # 数据类型
    batch_size=None,         # 批处理大小
    **kwargs                 # 其他参数
)
```

### 文本生成参数

```python
generator(
    prompt,                  # 提示文本
    max_length=50,           # 最大长度
    max_new_tokens=50,       # 新生成token数
    min_length=10,           # 最小长度
    do_sample=True,          # 是否采样
    temperature=1.0,         # 温度
    top_k=50,                # Top-K 采样
    top_p=0.95,              # Top-P 采样
    num_return_sequences=1,  # 返回序列数
    num_beams=1,             # 束搜索
    repetition_penalty=1.0,  # 重复惩罚
)
```

### 摘要参数

```python
summarizer(
    text,
    max_length=130,          # 最大摘要长度
    min_length=30,           # 最小摘要长度
    do_sample=False,         # 是否采样
    length_penalty=2.0,      # 长度惩罚
    num_beams=4,             # 束搜索数量
)
```

---

## 🔗 相关章节

- [基础用法](./transformers-basics.md) - 深入理解模型和分词器
- [推理与部署](./transformers-inference.md) - 生产环境推理优化
- [模型训练与微调](./transformers-training.md) - 自定义模型训练
