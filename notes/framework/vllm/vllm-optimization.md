# 性能优化技巧

本章深入介绍 vLLM 的性能优化方法，包括 PagedAttention 内存管理、KV Cache 优化、量化技术、并行策略和生产环境最佳实践。

---

## 🧠 PagedAttention 深入优化

### Block 大小优化

Block 大小影响内存利用率和计算效率：

```python
from vllm import LLM

# 默认 block_size=16（推荐大多数场景）
llm_default = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    block_size=16  # 每个 block 存储 16 个 token 的 KV
)

# 短序列场景：使用更大的 block
llm_short = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    block_size=32  # 减少管理开销
)

# 长序列场景：使用更小的 block
llm_long = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    block_size=8   # 更精细的内存管理
)
```

**Block 大小选择建议：**

| 场景 | 推荐 Block 大小 | 原因 |
|------|-----------------|------|
| 短对话（< 512 tokens） | 32 | 减少 block 管理开销 |
| 中等长度（512-2048） | 16 | 平衡内存和性能 |
| 长文本（> 2048 tokens） | 8 | 提高内存利用率 |
| 变长序列混合 | 16 | 通用推荐 |

### 前缀缓存优化

启用前缀缓存可以复用相同前缀的 KV Cache：

```python
from vllm import LLM, SamplingParams

# 启用前缀缓存
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True,  # 开启前缀缓存
)

# 相同系统提示的多次对话
system_prompt = """你是一个专业的编程助手，请用中文回答问题。
回答要求：
1. 简洁明了
2. 包含代码示例
3. 解释关键概念"""

prompts = [
    f"{system_prompt}\n\n用户：什么是装饰器？",
    f"{system_prompt}\n\n用户：解释一下生成器？",
    f"{system_prompt}\n\n用户：什么是上下文管理器？",
]

# 前缀会被缓存，后续请求复用
params = SamplingParams(max_tokens=200)
outputs = llm.generate(prompts, params)

# 性能提升：第二个请求起，前缀计算被跳过
```

### Swap 空间配置

```python
# 配置 CPU swap 空间（当 GPU 内存不足时使用）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    swap_space=4,  # 4GB CPU swap 空间
    gpu_memory_utilization=0.9,
)

# 注意：swap 会降低性能，仅在内存不足时使用
```

---

## 📦 KV Cache 管理

### 内存利用率优化

```python
from vllm import LLM

# 精细控制 GPU 内存使用
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.85,  # 留 15% 给其他进程
    max_model_len=4096,            # 限制最大序列长度
    enforce_eager=False,           # 启用 CUDA graph（更快）
)

# 内存计算公式：
# KV Cache 内存 = 2 * num_layers * d_model * max_seq_len * dtype_size
# 示例：7B 模型，max_seq_len=4096, FP16
# ≈ 2 * 32 * 4096 * 4096 * 2 bytes ≈ 2GB (仅 KV Cache)
```

### 最大序列长度调优

```python
import torch
from vllm import LLM

def estimate_memory(model_name, max_len, dtype="float16"):
    """估算 GPU 内存需求"""
    # 粗略估算（仅参考）
    if "7B" in model_name:
        params = 7e9
    elif "13B" in model_name:
        params = 13e9
    elif "70B" in model_name:
        params = 70e9
    else:
        params = 7e9
    
    # 模型权重
    dtype_bytes = 2 if dtype == "float16" else 4
    model_memory = params * dtype_bytes / 1e9  # GB
    
    # KV Cache（粗略）
    kv_memory = max_len * 0.001 * params / 7e9  # 粗略估算
    
    print(f"模型权重: {model_memory:.1f} GB")
    print(f"KV Cache (max_len={max_len}): ~{kv_memory:.1f} GB")
    print(f"总计预估: {model_memory + kv_memory:.1f} GB")

# 使用
estimate_memory("Qwen2.5-7B", 4096)
# 输出:
# 模型权重: 14.0 GB
# KV Cache (max_len=4096): ~4.0 GB
# 总计预估: 18.0 GB
```

### 批处理大小优化

```python
from vllm import LLM, SamplingParams

# 配置批处理参数
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=256,              # 最大并发序列数
    max_num_batched_tokens=8192,   # 最大批处理 token 数
)

# 批处理大小选择：
# - 小批量：延迟低，吞吐量低
# - 大批量：吞吐量高，延迟高
# - 平衡点：根据硬件和业务需求调整
```

---

## 🔢 量化技术

### FP8 量化

```python
from vllm import LLM, SamplingParams

# FP8 量化（需要 Hopper GPU 或 Ada Lovelace）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="fp8",
    enforce_eager=True,  # FP8 需要禁用 CUDA graph
)

# 优点：精度损失小，内存节省约 50%
# 缺点：需要特定 GPU 架构
```

### AWQ 量化

```python
# 加载预量化的 AWQ 模型
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
)

# 优点：精度保持好，广泛支持
# 缺点：需要预先量化的模型
```

### GPTQ 量化

```python
# 加载预量化的 GPTQ 模型
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",
)

# 优点：压缩率高，社区支持广泛
# 缺点：推理速度可能稍慢
```

### 量化对比

| 量化方式 | 内存节省 | 精度损失 | 速度 | GPU 要求 |
|----------|----------|----------|------|----------|
| FP16 | 基准 | 无 | 快 | 通用 |
| FP8 | ~50% | 极小 | 更快 | Hopper/Ada |
| INT8 | ~50% | 小 | 快 | 通用 |
| INT4 (AWQ/GPTQ) | ~75% | 中等 | 中等 | 通用 |

### 量化选择建议

```python
# 根据场景选择量化方案

# 场景 1：追求精度
llm = LLM(model="...", dtype="float16")  # 无量化

# 场景 2：GPU 内存有限
llm = LLM(model="...", quantization="awq")  # INT4

# 场景 3：最新 GPU，追求性能
llm = LLM(model="...", quantization="fp8")  # FP8

# 场景 4：CPU 推理（vLLM 有限支持）
# 建议：使用 llama.cpp
```

---

## 🔀 并行策略

### 张量并行 (Tensor Parallelism)

```python
from vllm import LLM

# 单机多 GPU 张量并行
llm = LLM(
    model="Qwen/Qwen2.5-70B-Instruct",
    tensor_parallel_size=4,  # 使用 4 个 GPU
    gpu_memory_utilization=0.9,
)

# 启动命令
# vllm serve Qwen/Qwen2.5-70B-Instruct --tensor-parallel-size 4
```

**张量并行原理：**

```
单 GPU:
┌─────────────────────────────────────┐
│        完整模型权重                  │
│  [Layer 0] [Layer 1] ... [Layer N]  │
└─────────────────────────────────────┘

4 GPU 张量并行:
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │
│ 1/4 权重 │ │ 1/4 权重 │ │ 1/4 权重 │ │ 1/4 权重 │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                    All-Reduce
```

### 流水线并行 (Pipeline Parallelism)

```python
# 流水线并行（适合超大模型跨节点部署）
llm = LLM(
    model="Qwen/Qwen2.5-70B-Instruct",
    pipeline_parallel_size=2,  # 分两阶段
    tensor_parallel_size=2,    # 结合张量并行
)

# 总 GPU 数 = tensor_parallel_size * pipeline_parallel_size
```

### 分布式推理

```python
# 使用 Ray 后端进行分布式推理
llm = LLM(
    model="Qwen/Qwen2.5-70B-Instruct",
    tensor_parallel_size=4,
    distributed_executor_backend="ray",  # 使用 Ray
)

# 启动 Ray 集群
# ray start --head
# ray start --address=<head-node-address>
```

### 并行策略选择

| 模型大小 | GPU 配置 | 推荐策略 |
|----------|----------|----------|
| 7B | 1x A100 40GB | 单 GPU |
| 7B | 1x RTX 4090 | INT4 量化 |
| 13B | 1x A100 80GB | 单 GPU |
| 13B | 2x A100 40GB | TP=2 |
| 70B | 4x A100 80GB | TP=4 |
| 70B | 8x A100 40GB | TP=8 |
| 70B | 2 节点 4x A100 | TP=4, PP=2 |

---

## ⚡ CUDA Graph 优化

### CUDA Graph 原理

```python
from vllm import LLM

# CUDA Graph 可以减少内核启动开销
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=False,  # 启用 CUDA Graph（默认）
    max_seq_len_to_capture=8192,  # 捕获的最大序列长度
)

# CUDA Graph 限制：
# 1. 不支持动态形状（需要预设最大长度）
# 2. 某些量化方案不支持（如 FP8）
# 3. 初始化时间更长
```

### CUDA Graph 配置

```python
# 小批量、低延迟场景
llm_low_latency = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=False,
    max_seq_len_to_capture=2048,  # 较短的捕获长度
)

# 大批量、高吞吐场景
llm_high_throughput = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=False,
    max_seq_len_to_capture=8192,  # 较长的捕获长度
)

# 需要禁用 CUDA Graph 的场景
llm_no_graph = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,  # 禁用 CUDA Graph
    quantization="fp8",  # FP8 需要
)
```

---

## 📊 性能分析与监控

### 吞吐量测试

```python
import time
from vllm import LLM, SamplingParams

def benchmark_throughput(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    num_prompts=100,
    max_tokens=100,
):
    """测试吞吐量"""
    llm = LLM(model=model_name)
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    
    prompts = ["你好"] * num_prompts
    
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"总 token 数: {total_tokens}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"吞吐量: {throughput:.2f} tokens/s")
    print(f"请求/秒: {num_prompts / elapsed:.2f} req/s")
    
    return throughput

benchmark_throughput()
```

### 延迟测试

```python
import time
import statistics
from vllm import LLM, SamplingParams

def benchmark_latency(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    num_requests=50,
    max_tokens=100,
):
    """测试延迟"""
    llm = LLM(model=model_name)
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    
    latencies = []
    
    for i in range(num_requests):
        start = time.time()
        output = llm.generate(["你好"], params)
        latency = time.time() - start
        latencies.append(latency)
    
    print(f"平均延迟: {statistics.mean(latencies):.3f}s")
    print(f"P50 延迟: {statistics.median(latencies):.3f}s")
    print(f"P95 延迟: {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}s")
    print(f"P99 延迟: {sorted(latencies)[int(len(latencies) * 0.99)]:.3f}s")

benchmark_latency()
```

### GPU 利用率监控

```python
import subprocess
import time

def monitor_gpu(interval=1, duration=60):
    """监控 GPU 使用情况"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        for i, line in enumerate(result.stdout.strip().split('\n')):
            gpu_util, mem_used, mem_total = line.split(', ')
            print(f"GPU {i}: 利用率={gpu_util}%, 内存={mem_used}/{mem_total} MB")
        
        time.sleep(interval)

# 运行监控
monitor_gpu(interval=2, duration=30)
```

---

## 🏭 生产环境最佳实践

### 配置模板

```python
# config.py - 生产环境配置

from dataclasses import dataclass

@dataclass
class vLLMConfig:
    """vLLM 生产环境配置"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "auto"
    trust_remote_code: bool = True
    
    # 内存配置
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    block_size: int = 16
    
    # 批处理配置
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 8192
    
    # 并行配置
    tensor_parallel_size: int = 1
    
    # 优化配置
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 4096
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""

# 使用
config = vLLMConfig()

from vllm import LLM
llm = LLM(
    model=config.model_name,
    dtype=config.dtype,
    trust_remote_code=config.trust_remote_code,
    gpu_memory_utilization=config.gpu_memory_utilization,
    max_model_len=config.max_model_len,
    max_num_seqs=config.max_num_seqs,
    enable_prefix_caching=config.enable_prefix_caching,
)
```

### 请求队列管理

```python
import asyncio
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class RequestItem:
    prompt: str
    params: dict
    future: asyncio.Future
    arrival_time: float

class RequestQueue:
    """请求队列管理"""
    
    def __init__(self, max_concurrent: int = 64):
        self.max_concurrent = max_concurrent
        self.queue = asyncio.Queue()
        self.active_requests = 0
    
    async def submit(self, prompt: str, params: dict) -> str:
        """提交请求"""
        future = asyncio.Future()
        item = RequestItem(
            prompt=prompt,
            params=params,
            future=future,
            arrival_time=time.time()
        )
        await self.queue.put(item)
        return await future
    
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return self.queue.qsize()

# 使用
queue = RequestQueue(max_concurrent=64)
```

### 错误处理与重试

```python
import asyncio
from vllm import LLM, SamplingParams
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustLLM:
    """带错误处理的 LLM 封装"""
    
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def generate(self, prompts: list, params: SamplingParams):
        """带重试的生成"""
        try:
            return self.llm.generate(prompts, params)
        except Exception as e:
            print(f"生成失败: {e}")
            raise
    
    def generate_safe(self, prompts: list, params: SamplingParams):
        """安全生成（不抛异常）"""
        try:
            return self.llm.generate(prompts, params)
        except Exception as e:
            print(f"生成失败: {e}")
            return None

# 使用
robust_llm = RobustLLM("Qwen/Qwen2.5-7B-Instruct")
outputs = robust_llm.generate(
    ["你好"],
    SamplingParams(max_tokens=100)
)
```

### 资源清理

```python
import gc
import torch
from vllm import LLM

def cleanup_resources():
    """清理 GPU 资源"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU 资源已清理")

# 使用上下文管理器
class LLMContext:
    """LLM 上下文管理器"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.llm = None
    
    def __enter__(self):
        self.llm = LLM(model=self.model_name, **self.kwargs)
        return self.llm
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.llm
        cleanup_resources()

# 使用
with LLMContext("Qwen/Qwen2.5-7B-Instruct") as llm:
    outputs = llm.generate(["你好"], SamplingParams(max_tokens=100))
    print(outputs[0].outputs[0].text)
# 自动清理资源
```

---

## 📋 优化检查清单

### 内存优化

- [ ] 设置合适的 `gpu_memory_utilization`（推荐 0.85-0.9）
- [ ] 限制 `max_model_len` 到实际需要的长度
- [ ] 启用 `enable_prefix_caching` 复用前缀
- [ ] 考虑量化（INT8/INT4）减少内存占用

### 吞吐量优化

- [ ] 增大 `max_num_seqs` 提高并发
- [ ] 增大 `max_num_batched_tokens` 提高批处理效率
- [ ] 使用批量推理而非单个请求
- [ ] 启用 CUDA Graph（`enforce_eager=False`）

### 延迟优化

- [ ] 减小 `max_num_seqs` 降低排队时间
- [ ] 使用更快的 GPU 或更多 GPU 并行
- [ ] 启用前缀缓存避免重复计算
- [ ] 减小 `max_model_len` 加快初始化

### 稳定性优化

- [ ] 实现请求重试机制
- [ ] 配置健康检查和监控
- [ ] 设置合理的超时时间
- [ ] 准备资源清理机制

---

## 🔗 相关章节

- [基础概念与安装](./vllm-basics.md) - PagedAttention 原理
- [离线推理](./vllm-inference.md) - 批量推理详解
- [API 服务器部署](./vllm-server.md) - 服务部署指南
