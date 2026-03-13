# 自定义模型与组件

当预训练模型无法满足需求时，Transformers 允许你创建自定义模型、分词器和配置。本章将介绍如何扩展和定制 Transformers 的核心组件。

---

## 🧩 自定义组件概览

Transformers 的自定义能力涵盖三个核心层面：

```
┌─────────────────────────────────────────────────────────┐
│                   自定义组件体系                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                 自定义 Config                     │    │
│  │          (模型超参数配置)                         │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │              自定义 Tokenizer                     │    │
│  │          (文本编码解码)                           │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │               自定义 Model                        │    │
│  │          (神经网络结构)                           │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ 自定义配置

### 继承 PretrainedConfig

```python
from transformers import PretrainedConfig

class CustomModelConfig(PretrainedConfig):
    """自定义模型配置"""
    
    model_type = "custom-model"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

# 使用自定义配置
config = CustomModelConfig(
    hidden_size=512,
    num_hidden_layers=6
)

# 保存配置
config.save_pretrained("./custom-model")

# 加载配置
config = CustomModelConfig.from_pretrained("./custom-model")
```

### 注册配置到 Auto

```python
from transformers import AutoConfig

# 注册配置类
AutoConfig.register("custom-model", CustomModelConfig)

# 现在可以使用 AutoConfig 加载
config = AutoConfig.from_pretrained("./custom-model")
```

---

## 🔤 自定义分词器

### 基于 PreTrainedTokenizer

```python
from transformers import PreTrainedTokenizer
from typing import List, Optional
import os

class CustomTokenizer(PreTrainedTokenizer):
    """自定义分词器"""
    
    vocab_files_names = {"vocab_file": "vocab.txt"}
    
    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        
        # 加载词表
        self.vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = i
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab
    
    def _tokenize(self, text):
        """分词逻辑"""
        # 简单的字符级分词
        return list(text)
    
    def _convert_token_to_id(self, token):
        """Token 转 ID"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        """ID 转 Token"""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens):
        """Tokens 转字符串"""
        return "".join(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """添加特殊标记"""
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]
        
        return output
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """获取特殊标记掩码"""
        if already_has_special_tokens:
            return [0] * len(token_ids_0)
        
        mask = [1] + [0] * len(token_ids_0) + [1]
        
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        
        return mask
```

### 使用 Fast Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 训练自定义分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=10000
)
tokenizer.pre_tokenizer = Whitespace()

# 训练
files = ["data/train.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("custom-tokenizer.json")

# 在 Transformers 中使用
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="custom-tokenizer.json",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    mask_token="[MASK]"
)
```

---

## 🤖 自定义模型

### 继承 PreTrainedModel

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_custom import CustomModelConfig

class CustomModel(PreTrainedModel):
    """自定义模型"""
    
    config_class = CustomModelConfig
    base_model_prefix = "custom"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Embedding 层
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # 输出层
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 初始化权重
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ):
        # 获取序列长度和批次大小
        batch_size, seq_length = input_ids.shape
        
        # 位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Token 类型编码
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        
        # 编码器
        if attention_mask is not None:
            # 转换注意力掩码格式
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        hidden_states = self.encoder(hidden_states.permute(1, 0, 2), src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        hidden_states = hidden_states.permute(1, 0, 2)
        
        # 池化输出
        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))
        
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output
        }
```

### 自定义分类模型

```python
from transformers import ModelOutput
from dataclasses import dataclass

@dataclass
class CustomClassifierOutput(ModelOutput):
    """分类模型输出"""
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: tuple = None
    attentions: tuple = None

class CustomModelForSequenceClassification(PreTrainedModel):
    """自定义序列分类模型"""
    
    config_class = CustomModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        
        # 基础模型
        self.model = CustomModel(config)
        
        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # 基础模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
        )
```

### 注册模型到 Auto

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# 注册基础模型
AutoModel.register(CustomModelConfig, CustomModel)

# 注册分类模型
AutoModelForSequenceClassification.register(CustomModelConfig, CustomModelForSequenceClassification)

# 现在可以使用 Auto 类加载
model = AutoModel.from_pretrained("./custom-model")
```

---

## 🎯 实践示例：自定义多头注意力

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """自定义多头注意力"""
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q, K, V 投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def transpose_for_scores(self, x):
        """重塑为多头格式"""
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq, head_size]
    
    def forward(self, hidden_states, attention_mask=None):
        # 计算 Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 计算上下文
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        batch_size, seq_length = context_layer.size()[:2]
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # 输出投影
        output = self.output(context_layer)
        
        return output, attention_probs
```

---

## 📁 完整项目结构

创建一个完整的自定义模型项目：

```
custom-transformers-model/
├── src/
│   └── custom_model/
│       ├── __init__.py
│       ├── configuration_custom.py    # 配置类
│       ├── modeling_custom.py         # 模型类
│       ├── tokenization_custom.py     # 分词器
│       └── tokenization_custom_fast.py # 快速分词器
├── custom-model/
│   ├── config.json                    # 模型配置
│   ├── pytorch_model.bin              # 模型权重
│   └── vocab.txt                      # 词表
├── setup.py                           # 安装脚本
└── README.md                          # 说明文档
```

### `__init__.py`

```python
from .configuration_custom import CustomModelConfig
from .modeling_custom import CustomModel, CustomModelForSequenceClassification
from .tokenization_custom import CustomTokenizer

__all__ = [
    "CustomModelConfig",
    "CustomModel",
    "CustomModelForSequenceClassification",
    "CustomTokenizer",
]
```

### `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="custom-transformers-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
```

---

## 📋 自定义组件检查清单

### Config 检查

- [ ] 继承 `PretrainedConfig`
- [ ] 设置 `model_type` 属性
- [ ] 所有参数在 `__init__` 中有默认值
- [ ] 调用 `super().__init__(**kwargs)`

### Model 检查

- [ ] 继承 `PreTrainedModel`
- [ ] 设置 `config_class` 属性
- [ ] 实现 `forward` 方法
- [ ] 调用 `self.post_init()` 初始化权重
- [ ] 实现 `get_input_embeddings` 和 `set_input_embeddings`

### Tokenizer 检查

- [ ] 继承 `PreTrainedTokenizer` 或使用 `PreTrainedTokenizerFast`
- [ ] 实现 `_tokenize` 方法
- [ ] 实现 `_convert_token_to_id` 和 `_convert_id_to_token`
- [ ] 实现 `vocab_size` 属性
- [ ] 处理特殊标记

---

## 🔗 相关章节

- [基础用法](./transformers-basics.md) - 模型与配置基础
- [模型训练与微调](./transformers-training.md) - 训练自定义模型
- [推理与部署](./transformers-inference.md) - 部署自定义模型
