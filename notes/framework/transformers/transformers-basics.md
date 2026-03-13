# 基础用法

本章介绍 Transformers 库的核心基础：模型加载、分词器使用、配置管理等。掌握这些基础概念是使用 Transformers 进行任何工作的前提。

---

## 📦 安装与环境配置

### 安装方式

```bash
# 基础安装
pip install transformers

# 推荐：安装完整工具链
pip install transformers datasets evaluate accelerate

# 安装 PyTorch 后端
pip install transformers torch

# 安装 TensorFlow 后端
pip install transformers tensorflow
```

### 版本检查

```python
import transformers
import torch

print(f"Transformers 版本: {transformers.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

### 离线模式配置

```python
# 设置离线模式（使用缓存模型）
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 设置镜像站点（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 或在加载时指定
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "bert-base-chinese",
    mirror="https://hf-mirror.com"
)
```

---

## 🧩 核心组件概览

Transformers 的核心组件及其关系：

```
┌─────────────────────────────────────────────────────────┐
│                    加载预训练模型                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐    ┌─────────────┐                   │
│   │ AutoConfig  │───▶│  配置信息    │                   │
│   └─────────────┘    └─────────────┘                   │
│          │                                             │
│          ▼                                             │
│   ┌─────────────┐    ┌─────────────┐                   │
│   │AutoTokenizer│───▶│  分词器      │                   │
│   └─────────────┘    └─────────────┘                   │
│          │                                             │
│          ▼                                             │
│   ┌─────────────┐    ┌─────────────┐                   │
│   │ AutoModel   │───▶│  模型权重    │                   │
│   └─────────────┘    └─────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| 组件 | 作用 | 自动类 |
|------|------|--------|
| **Config** | 存储模型超参数 | `AutoConfig` |
| **Tokenizer** | 文本编码与解码 | `AutoTokenizer` |
| **Model** | 神经网络模型 | `AutoModel` 系列 |

---

## 🔤 Tokenizer 分词器

分词器负责将原始文本转换为模型可理解的数字序列。

### 基本使用

```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 文本编码
text = "你好，世界！"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 872, 1962, 8024, 686, 1066, 8013, 102], 
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

# 解码回文本
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)  # [CLS] 你好，世界！ [SEP]
```

### 编码参数

```python
# 返回 PyTorch Tensor
inputs = tokenizer(text, return_tensors="pt")
print(inputs['input_ids'].shape)  # torch.Size([1, 8])

# 返回 TensorFlow Tensor
inputs = tokenizer(text, return_tensors="tf")

# 设置最大长度
inputs = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True
)

# 批量编码
texts = ["你好", "世界", "今天天气不错"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(inputs['input_ids'].shape)  # torch.Size([3, seq_len])
```

### 特殊标记

```python
# 查看特殊标记
print(tokenizer.cls_token)       # [CLS]
print(tokenizer.sep_token)       # [SEP]
print(tokenizer.pad_token)       # [PAD]
print(tokenizer.unk_token)       # [UNK]
print(tokenizer.mask_token)      # [MASK]

# 查看特殊标记 ID
print(tokenizer.cls_token_id)    # 101
print(tokenizer.sep_token_id)    # 102
```

### 分词与转换

```python
# 分词（文本 → Token）
tokens = tokenizer.tokenize("我爱自然语言处理")
print(tokens)  # ['我', '爱', '自', '然', '语', '言', '处', '理']

# 转换为 ID（Token → ID）
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)  # [2769, 4263, 5632, 4162, 6425, 6447, 1905, 4413]

# ID 转换为 Token
tokens_back = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens_back)
```

### 添加新词

```python
# 添加新词到词表
tokenizer.add_tokens(["新词1", "新词2"])

# 添加特殊标记
tokenizer.add_special_tokens({"additional_special_tokens": ["<NEW_TOKEN>"]})

# 注意：添加新词后需要调整模型 embedding 大小
model.resize_token_embeddings(len(tokenizer))
```

---

## 🤖 Model 模型

模型是神经网络的核心，负责进行前向计算。

### AutoModel 自动加载

```python
from transformers import AutoModel

# 加载预训练模型（仅 encoder，用于特征提取）
model = AutoModel.from_pretrained("bert-base-chinese")

# 查看模型结构
print(model.config)  # 模型配置
print(model)         # 模型结构
```

### 任务特定模型类

不同任务使用不同的 Model 类：

```python
from transformers import (
    AutoModelForSequenceClassification,  # 文本分类
    AutoModelForTokenClassification,      # 序列标注
    AutoModelForQuestionAnswering,        # 问答系统
    AutoModelForCausalLM,                 # 因果语言模型（GPT）
    AutoModelForMaskedLM,                 # 掩码语言模型（BERT）
    AutoModelForSeq2SeqLM,                # 序列到序列（T5, BART）
)

# 文本分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=3  # 分类数量
)

# 文本生成模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 翻译/摘要模型
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
```

### 模型推理

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 准备输入
text = "这个产品很好用"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1)
print(f"预测类别: {predicted_class.item()}")
```

### 模型输出

模型输出是一个 dataclass，包含多种信息：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-chinese")
inputs = tokenizer("你好", return_tensors="pt")

outputs = model(**inputs)

# 输出属性
print(outputs.last_hidden_state)      # 最后一层隐藏状态 [batch, seq_len, hidden_size]
print(outputs.pooler_output)          # [CLS] token 的池化输出 [batch, hidden_size]
print(outputs.hidden_states)          # 所有层隐藏状态（需要 output_hidden_states=True）
print(outputs.attentions)             # 注意力权重（需要 output_attentions=True）
```

---

## ⚙️ Config 配置

配置类存储模型的所有超参数。

### 加载与查看配置

```python
from transformers import AutoConfig

# 加载配置
config = AutoConfig.from_pretrained("bert-base-chinese")

# 查看关键参数
print(f"隐藏层大小: {config.hidden_size}")        # 768
print(f"注意力头数: {config.num_attention_heads}") # 12
print(f"隐藏层数: {config.num_hidden_layers}")    # 12
print(f"词表大小: {config.vocab_size}")           # 21128
print(f"最大位置编码: {config.max_position_embeddings}")  # 512

# 打印完整配置
print(config)
```

### 修改配置

```python
# 方式一：加载时修改
config = AutoConfig.from_pretrained(
    "bert-base-chinese",
    num_labels=5,           # 分类数量
    hidden_dropout_prob=0.2, # 增加 dropout
    attention_probs_dropout_prob=0.2
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    config=config
)

# 方式二：加载后修改
model.config.num_labels = 5
```

### 保存与加载配置

```python
# 保存配置
config.save_pretrained("./my-model")

# 从本地加载
config = AutoConfig.from_pretrained("./my-model")
```

---

## 💾 模型保存与加载

### 保存模型

```python
# 保存模型和分词器
model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")

# 保存后目录结构
# ./my-model/
# ├── config.json          # 模型配置
# ├── pytorch_model.bin    # 模型权重（旧格式）
# ├── model.safetensors    # 模型权重（新格式，推荐）
# └── vocab.txt            # 词表
```

### 加载模型

```python
# 从本地目录加载
model = AutoModel.from_pretrained("./my-model")
tokenizer = AutoTokenizer.from_pretrained("./my-model")

# 从 Hugging Face Hub 加载
model = AutoModel.from_pretrained("username/model-name")

# 加载特定版本
model = AutoModel.from_pretrained(
    "bert-base-chinese",
    revision="v1.0.0"  # 指定版本
)
```

### 仅加载权重

```python
# 加载预训练权重到自定义模型
model = MyCustomModel(config)
model.load_state_dict(
    torch.load("pytorch_model.bin"),
    strict=False  # 允许部分不匹配
)
```

---

## 🎯 实践示例

### 示例1：文本特征提取

```python
from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedding:
    """文本嵌入提取器"""
    
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embedding(self, text, pooling="cls"):
        """提取文本嵌入向量"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if pooling == "cls":
            # 使用 [CLS] token 的输出
            return outputs.last_hidden_state[:, 0, :].numpy()
        elif pooling == "mean":
            # 使用平均池化
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            return embedding.numpy()
    
    def similarity(self, text1, text2):
        """计算文本相似度"""
        import numpy as np
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # 余弦相似度
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity[0][0]

# 使用示例
embedder = TextEmbedding()

text1 = "今天天气很好"
text2 = "今天是个好天气"
text3 = "我喜欢吃苹果"

print(f"相似度 1-2: {embedder.similarity(text1, text2):.4f}")  # 较高
print(f"相似度 1-3: {embedder.similarity(text1, text3):.4f}")  # 较低
```

### 示例2：批量文本处理

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

class BatchProcessor:
    """批量文本处理器"""
    
    def __init__(self, model_name="bert-base-chinese", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
        self.model.eval()
    
    def predict_batch(self, texts, batch_size=32):
        """批量预测"""
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # 编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().tolist())
        
        return all_predictions

# 使用示例
processor = BatchProcessor()

texts = [
    "这个产品质量很好",
    "服务态度太差了",
    "物流很快，包装完好",
    # ... 更多文本
]

results = processor.predict_batch(texts)
print(results)
```

---

## 📋 Auto Classes 速查表

### AutoTokenizer

```python
# 加载
tokenizer = AutoTokenizer.from_pretrained("model-name")

# 编码
tokenizer(text)                           # 字典格式
tokenizer(text, return_tensors="pt")      # PyTorch Tensor
tokenizer(texts, padding=True, truncation=True)  # 批量

# 解码
tokenizer.decode(ids)
tokenizer.decode(ids, skip_special_tokens=True)

# 分词
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
```

### AutoModel 系列

| 模型类 | 任务 | 输出 |
|--------|------|------|
| `AutoModel` | 特征提取 | `last_hidden_state` |
| `AutoModelForSequenceClassification` | 文本分类 | `logits` |
| `AutoModelForTokenClassification` | 序列标注 | `logits` |
| `AutoModelForQuestionAnswering` | 问答 | `start_logits`, `end_logits` |
| `AutoModelForCausalLM` | 文本生成 | `logits` |
| `AutoModelForMaskedLM` | 掩码预测 | `logits` |
| `AutoModelForSeq2SeqLM` | 序列生成 | `logits` |

### 加载参数

```python
model = AutoModel.from_pretrained(
    "model-name",
    config=config,              # 自定义配置
    cache_dir="./cache",        # 缓存目录
    force_download=True,        # 强制重新下载
    local_files_only=True,      # 仅使用本地文件
    mirror="https://hf-mirror.com",  # 镜像站点
    torch_dtype=torch.float16,  # 半精度加载
    device_map="auto",          # 自动设备映射
)
```

---

## 🔗 相关章节

- [Pipeline 快速使用](./transformers-pipelines.md) - 零代码推理
- [模型训练与微调](./transformers-training.md) - 训练自己的模型
- [推理与部署](./transformers-inference.md) - 生产环境优化
