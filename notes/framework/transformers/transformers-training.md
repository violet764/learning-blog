# 模型训练与微调

Transformers 提供了强大的训练工具链，包括 Trainer API、Accelerate 分布式训练库，以及配套的数据处理工具。本章将详细介绍如何使用这些工具进行模型训练和微调。

---

## 📚 训练工具概览

Transformers 训练生态由以下组件构成：

```
┌─────────────────────────────────────────────────────────┐
│                   训练工具生态                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Dataset   │  │   Evaluate  │  │   Trainer   │    │
│  │  数据处理   │  │   评估指标  │  │   训练循环  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │            │
│         └────────────────┼────────────────┘            │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │      Accelerate       │                  │
│              │   (分布式训练后端)     │                  │
│              └───────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| 工具 | 功能 | 安装 |
|------|------|------|
| **datasets** | 数据集加载与处理 | `pip install datasets` |
| **evaluate** | 评估指标计算 | `pip install evaluate` |
| **accelerate** | 分布式训练 | `pip install accelerate` |
| **transformers** | Trainer API | `pip install transformers` |

---

## 📊 Dataset 数据集

### 加载数据集

```python
from datasets import load_dataset, Dataset

# 从 Hugging Face Hub 加载
dataset = load_dataset("imdb")  # 电影评论情感分析
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 加载中文数据集
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 加载特定子集
dataset = load_dataset("glue", "mrpc")

# 从本地文件加载
dataset = load_dataset("csv", data_files="data.csv")
dataset = load_dataset("json", data_files="data.json")
dataset = load_dataset("text", data_files="data.txt")
```

### 创建自定义数据集

```python
from datasets import Dataset, DatasetDict

# 从字典创建
data = {
    "text": ["文本1", "文本2", "文本3"],
    "label": [0, 1, 0]
}
dataset = Dataset.from_dict(data)

# 从 Pandas DataFrame 创建
import pandas as pd
df = pd.DataFrame({"text": ["文本1", "文本2"], "label": [0, 1]})
dataset = Dataset.from_pandas(df)

# 创建训练/验证/测试集划分
dataset_dict = DatasetDict({
    "train": dataset.train_test_split(test_size=0.2, seed=42)["train"],
    "test": dataset.train_test_split(test_size=0.2, seed=42)["test"]
})
```

### 数据预处理

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 批量处理
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,  # 多进程
    remove_columns=["text"]  # 移除原始列
)

# 设置格式为 PyTorch Tensor
tokenized_dataset.set_format("torch")
```

### 数据筛选与变换

```python
# 筛选数据
filtered_dataset = dataset.filter(lambda x: len(x["text"]) > 10)

# 数据变换
def add_length(example):
    example["length"] = len(example["text"])
    return example

dataset = dataset.map(add_length)

# 数据打乱
shuffled_dataset = dataset.shuffle(seed=42)

# 采样
sampled_dataset = dataset.select(range(1000))  # 选择前1000条
```

---

## 🎯 Trainer API

Trainer 是 Transformers 提供的高级训练接口，封装了完整的训练循环。

### 基本训练流程

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 2. 加载模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 5. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 6. 开始训练
trainer.train()
```

### TrainingArguments 参数详解

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 基础配置
    output_dir="./output",              # 输出目录
    
    # 训练参数
    num_train_epochs=3,                 # 训练轮数
    per_device_train_batch_size=16,     # 训练批次大小
    per_device_eval_batch_size=32,      # 评估批次大小
    gradient_accumulation_steps=1,      # 梯度累积步数
    learning_rate=5e-5,                 # 学习率
    weight_decay=0.01,                  # 权重衰减
    adam_beta1=0.9,                     # Adam beta1
    adam_beta2=0.999,                   # Adam beta2
    adam_epsilon=1e-8,                  # Adam epsilon
    max_grad_norm=1.0,                  # 梯度裁剪
    
    # 学习率调度
    warmup_steps=0,                     # 预热步数
    warmup_ratio=0.0,                   # 预热比例
    lr_scheduler_type="linear",         # 调度器类型
    
    # 日志配置
    logging_dir="./logs",               # 日志目录
    logging_steps=100,                  # 日志记录频率
    logging_first_step=False,           # 是否记录第一步
    report_to=["tensorboard"],          # 报告工具
    
    # 评估配置
    evaluation_strategy="epoch",        # 评估策略: no/steps/epoch
    eval_steps=500,                     # 评估步数（strategy="steps"时）
    
    # 保存配置
    save_strategy="epoch",              # 保存策略: no/steps/epoch
    save_steps=500,                     # 保存步数
    save_total_limit=3,                 # 最多保存数量
    load_best_model_at_end=True,        # 结束时加载最佳模型
    metric_for_best_model="accuracy",   # 最佳模型指标
    
    # 混合精度
    fp16=False,                         # 是否使用 FP16
    bf16=False,                         # 是否使用 BF16
    
    # 分布式训练
    local_rank=-1,                      # 本地进程排名
    ddp_find_unused_parameters=False,   # DDP 参数
)
```

### 添加评估指标

```python
import numpy as np
import evaluate

# 加载评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 在 Trainer 中使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 使用多个指标
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "precision": precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
    }
```

---

## 🔧 自定义 Trainer

### 继承 Trainer 类

```python
from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    """自定义 Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失函数"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 自定义损失（如加权交叉熵）
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """自定义训练步骤"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        return loss.detach()
```

### 添加回调函数

```python
from transformers import TrainerCallback

class PrintCallback(TrainerCallback):
    """打印回调函数"""
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("训练开始！")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n=== Epoch {state.epoch} 开始 ===")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

# 使用回调
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[PrintCallback()]
)
```

---

## ⚡ Accelerate 分布式训练

Accelerate 提供了简化的分布式训练接口。

### 基本使用

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# 初始化 Accelerator
accelerator = Accelerator()

# 准备模型和数据
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 数据集处理
dataset = load_dataset("lansinuote/ChnSentiCorp")
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch")

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
eval_loader = DataLoader(tokenized_dataset["validation"], batch_size=32)

# 优化器和调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

# 使用 Accelerator 准备
model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, eval_loader, scheduler
)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 使用 Accelerator 配置文件

```bash
# 生成配置文件
accelerate config

# 启动分布式训练
accelerate launch train.py
```

---

## 🎯 实践示例：文本分类微调

下面是一个完整的文本分类微调示例：

```python
"""
完整的文本分类微调脚本
"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np

# 配置
MODEL_NAME = "bert-base-chinese"
OUTPUT_DIR = "./text-classification-model"
NUM_LABELS = 2

# 1. 加载数据集
print("加载数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 划分验证集
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
    "test": dataset["test"]
})

# 2. 加载模型和分词器
print("加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)

# 3. 数据预处理
print("预处理数据...")
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]
)

# 4. 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. 评估指标
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
    }

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    fp16=True,  # 混合精度训练
    gradient_accumulation_steps=2,
    report_to="tensorboard",
)

# 7. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. 训练
print("开始训练...")
trainer.train()

# 9. 评估
print("评估模型...")
eval_results = trainer.evaluate(tokenized_dataset["test"])
print(f"测试集结果: {eval_results}")

# 10. 保存模型
print("保存模型...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("训练完成！")
```

---

## 📋 训练参数速查表

### 常用训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_epochs` | 3 | 训练轮数 |
| `per_device_train_batch_size` | 8 | 训练批次大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `weight_decay` | 0 | 权重衰减 |
| `warmup_ratio` | 0 | 预热比例 |
| `gradient_accumulation_steps` | 1 | 梯度累积 |
| `fp16` | False | 混合精度 |

### 评估与保存

| 参数 | 说明 |
|------|------|
| `evaluation_strategy` | 评估策略：no/steps/epoch |
| `save_strategy` | 保存策略：no/steps/epoch |
| `save_total_limit` | 最多保存模型数 |
| `load_best_model_at_end` | 结束时加载最佳模型 |
| `metric_for_best_model` | 最佳模型指标 |

---

## 🔗 相关章节

- [基础用法](./transformers-basics.md) - 模型加载与分词器
- [推理与部署](./transformers-inference.md) - 模型推理优化
- [自定义组件](./transformers-custom.md) - 自定义模型结构
