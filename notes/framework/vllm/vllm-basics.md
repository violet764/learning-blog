# 基础概念与安装

本章介绍 vLLM 的核心概念、安装方法和基本使用。理解 PagedAttention 原理是掌握 vLLM 的关键，它能帮助你更好地进行性能调优。

---

## 🔬 核心概念

### 什么是 vLLM

vLLM 是 UC Berkeley 研发的开源大语言模型推理框架，核心创新是 **PagedAttention** 算法。它解决了传统 LLM 推理中的内存管理问题，实现了：

- **高吞吐量**：比 HuggingFace Transformers 快 10-20 倍
- **高内存效率**：内存利用率提升 2-4 倍
- **低延迟**：优化的 CUDA 内核实现

### PagedAttention 深入理解

#### 传统注意力机制的问题

在 Transformer 推理过程中，需要缓存每个 token 的 Key 和 Value（KV Cache）：

```
Self-Attention 计算：
                    Q₁   Q₂   Q₃   Q₄
                    ↓    ↓    ↓    ↓
K₁ K₂ K₃ K₄ ────►  Attention Matrix
                    ↑    ↑    ↑    ↑
V₁ V₂ V₃ V₄        V₁   V₂   V₃   V₄

KV Cache 随序列长度线性增长
```

传统实现的问题：

```
问题 1：内存预分配浪费
┌────────────────────────────────────────────┐
│ 预分配最大长度 4096 tokens                  │
│ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ 实际使用 512 tokens，浪费 87.5%             │
└────────────────────────────────────────────┘

问题 2：内存碎片
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Req 1  │ │ Req 2  │ │ Req 3  │ │ Req 4  │  不同长度
│ 1000   │ │ 500    │ │ 2000   │ │ 300    │  碎片严重
└────────┘ └────────┘ └────────┘ └────────┘

问题 3：无法共享相同前缀
┌────────────────────────────────────────────┐
│ Request 1: "请翻译以下文本：Hello World"    │
│ Request 2: "请翻译以下文本：Good Morning"   │
│            └─────────────────────┘         │
│              相同前缀，重复存储             │
└────────────────────────────────────────────┘
```

#### PagedAttention 的解决方案

借鉴操作系统的**虚拟内存管理**思想，将 KV Cache 划分为固定大小的**块（Block）**：

```
Block 结构：
┌────────────────────────────────────┐
│ Block (e.g., 16 tokens per block) │
├────────────────────────────────────┤
│ K: [k₁, k₂, ..., k₁₆]             │
│ V: [v₁, v₂, ..., v₁₆]             │
└────────────────────────────────────┘

虚拟内存映射：
┌─────────────────┐      ┌─────────────────┐
│ 逻辑视图         │      │ 物理内存池       │
│ (Logical Blocks)│      │ (Physical Blocks)│
├─────────────────┤      ├─────────────────┤
│ Request 1       │      │ Block Pool:     │
│ ┌───┐┌───┐┌───┐│      │ ┌───┐┌───┐┌───┐│
│ │ 0 ││ 3 ││ 7 ││─────►│ │ 0 ││ 1 ││ 2 ││
│ └───┘└───┘└───┘│      │ └───┘└───┘└───┘│
│                 │      │ ┌───┐┌───┐┌───┐│
│ Request 2       │      │ │ 3 ││ 4 ││ 5 ││
│ ┌───┐┌───┐      │─────►│ └───┘└───┘└───┘│
│ │ 1 ││ 4 │      │      │ ┌───┐┌───┐┌───┐│
│ └───┘└───┘      │      │ │ 6 ││ 7 ││ 8 ││
│                 │      │ └───┘└───┘└───┘│
└─────────────────┘      └─────────────────┘

特点：
- 非连续存储：块可以在物理内存中任意位置
- 按需分配：只分配实际需要的块
- 即时回收：请求结束立即释放块
```

#### 内存共享机制

PagedAttention 支持 **Copy-on-Write** 的内存共享：

```
示例：多轮对话共享前缀

Round 1: "请帮我写一首关于春天的诗"
         Block 0  Block 1  Block 2
         ┌────┐   ┌────┐   ┌────┐
         │ KV │───│ KV │───│ KV │
         └────┘   └────┘   └────┘

Round 2: "请帮我写一首关于春天的诗，要求押韵"
         Block 0  Block 1  Block 2  Block 3
         ┌────┐   ┌────┐   ┌────┐   ┌────┐
         │ KV │───│ KV │───│ KV │───│ KV │
         └────┘   └────┘   └────┘   └────┘
             ↑        ↑        ↑
             └────────┴────────┘
               共享相同的块（引用计数）

内存节省：前缀部分只需存储一次
```

### Continuous Batching

传统批处理需要等待所有请求完成才能处理下一批：

```
传统静态批处理：
时间 ──────────────────────────────────────►
Request 1: ████████████████████ (短任务，已完成)
Request 2: ████████████████████████████████ (长任务)
Request 3: ████████████████████████████████████████ (更长任务)

问题：Request 1 完成后 GPU 空闲等待
```

vLLM 的 Continuous Batching 实现动态调度：

```
Continuous Batching：
时间 ──────────────────────────────────────►
Request 1: ████████████████████ ✓
Request 2: ████████████████████████████████ ✓
Request 3: ████████████████████████████████████████ ✓
Request 4:                   ████████████████████ ✓
Request 5:                         ████████████████████ ✓

优势：
- 请求完成立即加入新请求
- GPU 利用率最大化
- 降低平均等待时间
```

---

## 📦 安装指南

### 环境要求

| 组件 | 要求 |
|------|------|
| **操作系统** | Linux (Ubuntu 20.04+ 推荐) |
| **Python** | 3.8 - 3.12 |
| **CUDA** | 11.8 / 12.1+ |
| **GPU** | NVIDIA GPU, 计算能力 7.0+ |

### 安装方式

#### 方式一：pip 安装（推荐）

```bash
# 基础安装
pip install vllm

# 安装特定版本
pip install vllm==0.4.0

# 指定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

#### 方式二：从源码安装

```bash
# 克隆仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装依赖
pip install -e .
```

#### 方式三：Docker 部署

```bash
# 拉取镜像
docker pull vllm/vllm-openai:latest

# 运行容器
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct
```

### 验证安装

```python
# 检查版本和 GPU 信息
import vllm
print(f"vLLM 版本: {vllm.__version__}")

# 检查 CUDA 可用性
from vllm import LLM
llm = LLM(model="facebook/opt-125m")  # 使用小模型测试
print("vLLM 安装成功！")
```

### 常见安装问题

#### CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi  # 驱动支持的 CUDA
nvcc --version  # 编译用的 CUDA

# 解决：安装匹配的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

#### 内存不足

```bash
# 错误信息
# RuntimeError: CUDA out of memory

# 解决：减少 GPU 内存占用
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.8,  # 限制 GPU 内存使用
    enforce_eager=True           # 禁用 CUDA graph
)
```

---

## 🚀 基本使用

### 离线推理

使用 `LLM` 类进行批量离线推理：

```python
from vllm import LLM, SamplingParams

# 1. 初始化模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,  # 信任远程代码
    dtype="auto",            # 自动选择精度
    gpu_memory_utilization=0.9,  # GPU 内存利用率
)

# 2. 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,     # 温度参数
    top_p=0.9,          # 核采样
    top_k=50,           # Top-K 采样
    max_tokens=256,     # 最大生成长度
    stop=["</s>", "\n\n"],  # 停止词
)

# 3. 准备提示词
prompts = [
    "请解释什么是机器学习",
    "用 Python 写一个快速排序算法",
    "翻译成英文：人工智能正在改变世界"
]

# 4. 批量推理
outputs = llm.generate(prompts, sampling_params)

# 5. 处理输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示词: {prompt}")
    print(f"生成: {generated_text}")
    print("-" * 50)
```

### 使用聊天模板

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 方式一：手动构建对话格式
prompts = [
    """<|im_start|>system
你是一个有帮助的AI助手。
<|im_end|>
<|im_start|>user
什么是深度学习？
<|im_end|>
<|im_start|>assistant
"""
]

# 方式二：使用 tokenizer 的聊天模板（推荐）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是深度学习？"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate([prompt], SamplingParams(max_tokens=256))
print(outputs[0].outputs[0].text)
```

### 流式输出

```python
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 使用异步引擎实现流式输出
from vllm import AsyncLLMEngine, AsyncLLMEngineArgs, SamplingParams
import asyncio

async def stream_generate():
    # 配置引擎参数
    engine_args = AsyncLLMEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
    )
    
    # 创建异步引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 生成请求
    prompt = "请写一首关于春天的诗"
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
    
    # 流式获取结果
    async for output in engine.generate(prompt, sampling_params, request_id="test"):
        if output.outputs:
            text = output.outputs[0].text
            print(text, end="", flush=True)

# 运行
asyncio.run(stream_generate())
```

---

## ⚙️ 模型加载配置

### 基础配置参数

```python
from vllm import LLM

llm = LLM(
    # 模型配置
    model="Qwen/Qwen2.5-7B-Instruct",  # 模型名称或路径
    tokenizer=None,                     # 分词器路径（默认同模型）
    tokenizer_mode="auto",              # 分词器模式
    trust_remote_code=True,             # 信任远程代码
    
    # 精度配置
    dtype="auto",                       # 数据类型: "auto", "float16", "bfloat16"
    quantization=None,                  # 量化方案: None, "awq", "gptq", "fp8"
    
    # 内存配置
    gpu_memory_utilization=0.9,         # GPU 内存利用率 (0-1)
    max_model_len=4096,                 # 最大序列长度
    enforce_eager=False,                # 是否禁用 CUDA graph
    
    # 并行配置
    tensor_parallel_size=1,             # 张量并行大小
    pipeline_parallel_size=1,           # 流水线并行大小
)
```

### 加载本地模型

```python
# 从本地路径加载
llm = LLM(
    model="/path/to/local/model",
    tokenizer="/path/to/local/tokenizer",  # 可选
    trust_remote_code=True
)

# 从 HuggingFace Hub 加载
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    revision="main",  # 可指定分支或 commit
    trust_remote_code=True
)
```

### 加载量化模型

```python
# AWQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16"
)

# GPTQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16"
)

# FP8 量化
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="fp8",
    enforce_eager=True  # FP8 需要禁用 CUDA graph
)
```

---

## 📊 性能监控

### 查看模型信息

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 获取模型配置
print(f"模型: {llm.llm_engine.model_config.model}")
print(f"最大长度: {llm.llm_engine.model_config.max_model_len}")
print(f"数据类型: {llm.llm_engine.model_config.dtype}")
```

### 监控 GPU 使用

```python
import torch
from vllm import LLM, SamplingParams

def generate_with_monitor():
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    
    # 监控 GPU 内存
    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 内存 - 已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")
    
    print("加载模型后:")
    print_gpu_memory()
    
    # 执行推理
    prompts = ["你好"] * 10
    sampling_params = SamplingParams(max_tokens=100)
    outputs = llm.generate(prompts, sampling_params)
    
    print("推理完成后:")
    print_gpu_memory()
    
    return outputs

outputs = generate_with_monitor()
```

---

## 📋 最佳实践

### ✅ 推荐做法

```python
# 1. 合理设置 GPU 内存利用率
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.85  # 留 15% 余量
)

# 2. 使用批量推理提高吞吐
prompts = [f"问题 {i}: ..." for i in range(100)]
outputs = llm.generate(prompts, sampling_params)  # 批量处理

# 3. 设置合理的最大长度
sampling_params = SamplingParams(
    max_tokens=512,  # 根据任务设置
    stop=["</s>", "```"]  # 设置停止词
)

# 4. 使用异步引擎处理并发请求
from vllm import AsyncLLMEngine
# 适合服务端部署
```

### ❌ 避免的做法

```python
# 1. 不要逐个处理请求（效率低）
for prompt in prompts:
    output = llm.generate([prompt], sampling_params)  # ❌ 低效
    
# 应该批量处理
outputs = llm.generate(prompts, sampling_params)  # ✅ 高效

# 2. 不要设置过大的 max_tokens
sampling_params = SamplingParams(max_tokens=32768)  # ❌ 浪费内存

# 3. 不要忽略内存限制
llm = LLM(model="70B-model", gpu_memory_utilization=0.99)  # ❌ 可能 OOM
```

---

## 🔗 相关章节

- [离线推理](./vllm-inference.md) - LLM 类和 SamplingParams 详解
- [API 服务器部署](./vllm-server.md) - OpenAI 兼容 API 服务
- [性能优化技巧](./vllm-optimization.md) - 深入优化方法
