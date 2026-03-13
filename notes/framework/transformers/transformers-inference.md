# 推理与部署

在生产环境中使用模型需要考虑推理效率、内存占用、部署方式等问题。本章将介绍如何优化推理性能，以及将模型部署到各种环境的方法。

---

## 🚀 推理基础

### 基本推理流程

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 准备输入
text = "这个产品很好用"
inputs = tokenizer(text, return_tensors="pt")

# 推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1)

print(f"预测类别: {predicted_class.item()}")
```

### 批量推理

```python
def batch_inference(texts, model, tokenizer, batch_size=32, device="cuda"):
    """批量推理"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 编码
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().tolist())
    
    return all_predictions
```

---

## 💻 设备管理

### GPU 推理

```python
import torch

# 检查 GPU 可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 将模型移到 GPU
model = model.to(device)

# 输入也需移到同一设备
inputs = {k: v.to(device) for k, v in inputs.items()}
```

### 多 GPU 推理

```python
# 方式一：DataParallel（简单但效率较低）
model = torch.nn.DataParallel(model)

# 方式二：DistributedDataParallel（推荐）
# 需要使用 torch.distributed.launch 启动

# 方式三：使用 device_map 自动分配
model = AutoModelForCausalLM.from_pretrained(
    "big-model",
    device_map="auto"  # 自动分配到可用 GPU
)

# 查看设备映射
print(model.hf_device_map)
```

### 设备映射策略

```python
from transformers import AutoModelForCausalLM

# 自动分配
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map="auto"
)

# 平衡分配
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map="balanced"
)

# 手动指定
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map={
        "encoder": 0,
        "decoder": 1,
    }
)
```

---

## ⚡ 性能优化

### 混合精度推理

```python
# 方式一：使用 torch.cuda.amp
with torch.cuda.amp.autocast():
    outputs = model(**inputs)

# 方式二：加载时指定 dtype
import torch
model = AutoModel.from_pretrained(
    "model-name",
    torch_dtype=torch.float16  # 或 torch.bfloat16
)

# 方式三：使用 .half() 转换
model = model.half()
```

### 模型量化

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=quantization_config,
    device_map="auto"
)

# 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    load_in_8bit=True,
    device_map="auto"
)
```

### Flash Attention

```python
# 启用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map="auto"
)
```

### KV Cache 优化

```python
# 文本生成时使用 KV Cache
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, I am", return_tensors="pt")

# 使用 past_key_values 加速连续生成
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    use_cache=True,  # 启用 KV Cache
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0]))
```

---

## 🔧 推理加速库

### 使用 BetterTransformer

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 转换为 BetterTransformer（使用 Flash Attention）
model = model.to_bettertransformer()

# 正常推理
outputs = model(**inputs)
```

### 使用 ONNX Runtime

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# 导出为 ONNX 格式
model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 或加载已导出的 ONNX 模型
model = ORTModelForSequenceClassification.from_pretrained(
    "./onnx-model",
    file_name="model.onnx"
)

# 推理
outputs = model(**inputs)
```

### 导出 ONNX 模型

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.exporters.onnx import main_export

# 导出模型
main_export(
    model_name_or_path="bert-base-chinese",
    output="./onnx-model",
    task="text-classification"
)

# 加载 ONNX 模型
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained("./onnx-model")
```

---

## 📦 模型导出与部署

### 导出为 TorchScript

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 创建示例输入
dummy_input = tokenizer("测试文本", return_tensors="pt")

# 追踪模型
traced_model = torch.jit.trace(model, (dummy_input['input_ids'], dummy_input['attention_mask']))

# 保存
traced_model.save("traced_model.pt")

# 加载
loaded_model = torch.jit.load("traced_model.pt")
```

### 导出为 ONNX

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 准备输入
dummy_input = tokenizer("测试文本", return_tensors="pt")

# 导出 ONNX
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state', 'pooler_output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
        'pooler_output': {0: 'batch_size'}
    },
    opset_version=14
)
```

### 使用 FastAPI 部署

```python
"""
FastAPI 推理服务
"""
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List

app = FastAPI()

# 全局变量存储模型
model = None
tokenizer = None
device = None

class InferenceRequest(BaseModel):
    texts: List[str]

class InferenceResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-model")
    model.to(device)
    model.eval()

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """推理接口"""
    # 编码
    inputs = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
    
    return InferenceResponse(
        predictions=predictions.cpu().tolist(),
        probabilities=probs.cpu().tolist()
    )

# 启动: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🎯 实践示例

### 示例1：高性能推理服务类

```python
"""
高性能推理服务封装
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Optional
from functools import lru_cache
import time

class InferenceService:
    """推理服务类"""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        batch_size: int = 32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(self.device)
        self.model.eval()
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """模型预热"""
        dummy_input = self.tokenizer("warmup", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**dummy_input)
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[dict]:
        """批量预测"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            for j, (pred, prob) in enumerate(zip(predictions, probs)):
                results.append({
                    "text": batch[j],
                    "prediction": pred.item(),
                    "confidence": prob[pred].item(),
                    "probabilities": prob.cpu().tolist()
                })
        
        return results

# 使用示例
service = InferenceService("./fine-tuned-model", batch_size=64)
results = service.predict(["文本1", "文本2", "文本3"])
```

### 示例2：流式文本生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def stream_generate(prompt: str, model_name: str = "gpt2"):
    """流式文本生成"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # 生成参数
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=100,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # 在单独线程中运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 流式输出
    for text in streamer:
        yield text
    
    thread.join()

# 使用示例
for text in stream_generate("从前有座山"):
    print(text, end="", flush=True)
```

---

## 📊 性能对比

### 不同优化方法的推理速度

| 优化方法 | 相对速度 | 内存占用 | 精度损失 |
|----------|----------|----------|----------|
| 原始 FP32 | 1x | 100% | 无 |
| FP16 | ~2x | ~50% | 极小 |
| BF16 | ~2x | ~50% | 极小 |
| INT8 量化 | ~2-3x | ~25% | 小 |
| INT4 量化 | ~3-4x | ~12.5% | 中等 |
| ONNX Runtime | ~1.5-2x | 相同 | 无 |
| Flash Attention 2 | ~2-3x | 相同 | 无 |

### 推理优化检查清单

```
□ 使用半精度 (FP16/BF16)
□ 启用 Flash Attention
□ 使用 KV Cache（文本生成）
□ 批量推理优化
□ 量化（如果可接受精度损失）
□ ONNX 导出（生产部署）
□ 模型预热
□ 合理的批处理大小
```

---

## 📋 推理参数速查

### 模型加载参数

```python
AutoModel.from_pretrained(
    "model-name",
    torch_dtype=torch.float16,     # 数据类型
    device_map="auto",             # 设备映射
    low_cpu_mem_usage=True,        # 低内存加载
    use_flash_attention_2=True,    # Flash Attention
    use_cache=True,                # KV Cache
)
```

### 生成参数

```python
model.generate(
    **inputs,
    max_new_tokens=100,            # 最大新 token 数
    min_new_tokens=10,             # 最小新 token 数
    do_sample=True,                # 是否采样
    temperature=0.7,               # 温度
    top_k=50,                      # Top-K
    top_p=0.9,                     # Top-P
    repetition_penalty=1.1,        # 重复惩罚
    num_beams=1,                   # 束搜索
    use_cache=True,                # KV Cache
    streamer=streamer,             # 流式输出
)
```

---

## 🔗 相关章节

- [基础用法](./transformers-basics.md) - 模型加载与分词器
- [Pipeline 快速使用](./transformers-pipelines.md) - 简单推理
- [模型训练与微调](./transformers-training.md) - 训练自定义模型
