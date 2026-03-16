# ZeRO 优化 (Zero Redundancy Optimizer)

ZeRO (Zero Redundancy Optimizer) 是 Microsoft DeepSpeed 提出的显存优化技术，通过消除数据并行中的冗余存储，大幅降低显存占用，使单机可以训练超大模型。

## 问题背景

### 数据并行的冗余存储

在标准数据并行（DDP）中，每个 GPU 都存储：

1. **模型参数** (Parameters)
2. **梯度** (Gradients)
3. **优化器状态** (Optimizer States)

这些存储完全相同，存在大量冗余！

```
┌─────────────────────────────────────────────────────────────────┐
│                   DDP 的冗余存储                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 0          GPU 1          GPU 2          GPU 3            │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐          │
│  │ P,G │        │ P,G │        │ P,G │        │ P,G │          │
│  │ OS  │        │ OS  │        │ OS  │        │ OS  │          │
│  └─────┘        └─────┘        └─────┘        └─────┘          │
│                                                                 │
│  P = Parameters, G = Gradients, OS = Optimizer States          │
│  每卡都存储完整副本，冗余 4 倍！                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 显存占用分析

以 Adam 优化器训练 $\Psi$ 参数的模型为例（FP32）：

| 存储项 | 计算公式 | 大小 |
|--------|---------|------|
| 模型参数 | $\Psi \times 4$ bytes | $4\Psi$ |
| 梯度 | $\Psi \times 4$ bytes | $4\Psi$ |
| 优化器状态 | $\Psi \times 8$ bytes (m, v) | $8\Psi$ |
| **总计** | | **$16\Psi$** |

::: tip 示例
**7B 模型**：$7 \times 10^9 \times 16 = 112$ GB 显存/GPU
**70B 模型**：$70 \times 10^9 \times 16 = 1.12$ TB 显存/GPU

单卡根本放不下！
:::

## ZeRO 三级优化

ZeRO 通过**分片存储**消除冗余，分为三个级别：

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZeRO 三级优化对比                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard DP:                                                   │
│  GPU 0: [P | G | OS]                                            │
│  GPU 1: [P | G | OS]  ← 所有 GPU 存储完整副本                   │
│  GPU 2: [P | G | OS]                                            │
│  GPU 3: [P | G | OS]                                            │
│                                                                 │
│  ZeRO-1 (分片优化器状态):                                        │
│  GPU 0: [P | G | OS₀]                                           │
│  GPU 1: [P | G | OS₁]  ← OS 分片                               │
│  GPU 2: [P | G | OS₂]                                           │
│  GPU 3: [P | G | OS₃]                                           │
│                                                                 │
│  ZeRO-2 (分片优化器状态 + 梯度):                                 │
│  GPU 0: [P | G₀ | OS₀]                                          │
│  GPU 1: [P | G₁ | OS₁]  ← OS + G 分片                          │
│  GPU 2: [P | G₂ | OS₂]                                          │
│  GPU 3: [P | G₃ | OS₃]                                          │
│                                                                 │
│  ZeRO-3 (分片所有):                                              │
│  GPU 0: [P₀ | G₀ | OS₀]                                         │
│  GPU 1: [P₁ | G₁ | OS₁]  ← 全部分片                            │
│  GPU 2: [P₂ | G₂ | OS₂]                                         │
│  GPU 3: [P₃ | G₃ | OS₃]                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 显存占用对比

| 级别 | 显存占用 | 通信量 |
|------|---------|--------|
| DDP | $16\Psi$ | $2\Psi$ |
| ZeRO-1 | $4\Psi + \frac{8\Psi}{N}$ | $2\Psi$ |
| ZeRO-2 | $4\Psi + \frac{12\Psi}{N}$ | $2\Psi$ |
| ZeRO-3 | $\frac{16\Psi}{N}$ | $3\Psi$ |

其中 $N$ 是 GPU 数量。

::: tip 计算示例
**7B 模型，64 GPU**：

- DDP: 112 GB/GPU（不可行）
- ZeRO-3: $16 \times 7 \text{GB} / 64 = 1.75$ GB/GPU ✅
:::

## ZeRO-1: 优化器状态分片

### 原理

只分片优化器状态，保持参数和梯度的完整副本。

```
前向传播: 
  每卡独立执行

反向传播:
  All-Reduce 同步梯度

参数更新:
  Reduce-Scatter 将梯度分片
  每卡只更新自己负责的参数
  All-Gather 同步更新后的参数
```

### 通信量

ZeRO-1 与 DDP 通信量相同：$2\Psi$

### 适用场景

- 模型参数能放入单卡
- 优化器状态是瓶颈
- 最小的代码改动

## ZeRO-2: 梯度 + 优化器状态分片

### 原理

进一步分片梯度，进一步降低显存。

```
反向传播:
  每卡计算梯度后，Reduce-Scatter 到对应卡
  每卡只存储自己负责参数的梯度

参数更新:
  每卡只更新自己负责的参数
  All-Gather 同步更新后的参数
```

### 通信量

ZeRO-2 与 DDP 通信量相同：$2\Psi$

### 适用场景

- 梯度存储是瓶颈
- 想要更多显存节省，但不接受通信增加

## ZeRO-3: 完全分片

### 原理

分片所有存储：参数、梯度、优化器状态。

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZeRO-3 执行流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  前向传播:                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. All-Gather 获取当前层参数                            │  │
│  │  2. 计算前向                                             │  │
│  │  3. 立即释放其他卡的参数副本                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  反向传播:                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. All-Gather 获取当前层参数                            │  │
│  │  2. 计算梯度                                             │  │
│  │  3. Reduce-Scatter 发送梯度到对应卡                      │  │
│  │  4. 释放参数副本                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  参数更新:                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  每卡只更新自己持有的参数分片                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 通信量

ZeRO-3 通信量增加：$3\Psi$

- 前向：$2\Psi$（All-Gather 参数）
- 反向：$2\Psi$（All-Gather 参数）
- 梯度同步：$\Psi$（Reduce-Scatter）

但通过 **通信优化**（预取、流水线），可以隐藏部分通信开销。

### 代码示例

```python
import torch
import torch.nn as nn
from deepspeed import deepspeed


class LargeModel(nn.Module):
    """大型模型示例"""
    def __init__(self, hidden_size=4096, num_layers=100):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


def train_zero3():
    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "zero_optimization": {
            "stage": 3,                    # ZeRO-3
            "overlap_comm": True,          # 通信与计算重叠
            "contiguous_gradients": True,  # 连续存储梯度
            "reduce_bucket_size": 5e8,     # Reduce bucket 大小
            "stage3_prefetch_bucket_size": 5e8,  # 预取 bucket 大小
            "stage3_param_persistence_threshold": 1e6,  # 小参数不分片
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
        }
    }
    
    # 初始化模型
    model = LargeModel()
    
    # 使用 DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 训练循环
    for epoch in range(10):
        for batch in dataloader:
            inputs, labels = batch
            
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            
            model_engine.backward(loss)
            model_engine.step()  # 包含梯度清零
    
    # 保存模型（ZeRO-3 需要特殊处理）
    # 见下文
```

### ZeRO-3 模型保存与加载

```python
def save_zero3_checkpoint(model_engine, save_dir, tag="checkpoint"):
    """保存 ZeRO-3 检查点"""
    
    # 使用 DeepSpeed 的保存方法
    # 每个进程保存自己的分片
    model_engine.save_checkpoint(save_dir, tag=tag)


def load_zero3_checkpoint(model, save_dir, tag="checkpoint"):
    """加载 ZeRO-3 检查点"""
    
    # 加载检查点
    _, _ = model_engine.load_checkpoint(
        save_dir,
        tag=tag,
        load_module_strict=True
    )
    
    return model_engine
```

### 获取完整模型（用于推理）

```python
def get_full_state_dict(model_engine):
    """从 ZeRO-3 分片中收集完整模型"""
    
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    # 方法1: 从检查点加载
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir="./checkpoint"
    )
    
    # 方法2: 实时收集（需要所有进程参与）
    # 需要在所有 GPU 上调用
    gathered_state_dict = {}
    for name, param in model_engine.named_parameters():
        # All-Gather 参数
        gathered_param = gather_parameter(param)
        gathered_state_dict[name] = gathered_param
    
    return gathered_state_dict
```

## ZeRO-Offload

ZeRO-Offload 将优化器状态和梯度卸载到 CPU 内存，进一步降低 GPU 显存。

### 原理

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZeRO-Offload 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     GPU                    CPU                  │
│                 ┌─────────┐            ┌─────────┐             │
│                 │   P     │            │   OS    │             │
│                 │  (分片) │            │  (分片) │             │
│                 │         │◄──────────►│    G    │             │
│                 │ Forward │   PCIe     │  (分片) │             │
│                 │ Backward│            │ Optim   │             │
│                 └─────────┘            └─────────┘             │
│                                                                 │
│  GPU: 存储参数分片，执行前向/反向                                │
│  CPU: 存储优化器状态和梯度，执行参数更新                         │
└─────────────────────────────────────────────────────────────────┘
```

### 配置示例

```python
ds_config = {
    "zero_optimization": {
        "stage": 2,              # Offload 通常与 ZeRO-2 配合
        "offload_optimizer": {
            "device": "cpu",     # 卸载到 CPU
            "pin_memory": True,  # 固定内存加速传输
        },
        "offload_param": {
            "device": "cpu",     # 参数也卸载到 CPU（可选）
        }
    }
}
```

### 显存占用

| 配置 | GPU 显存 | CPU 内存 |
|------|---------|---------|
| ZeRO-3 | $\frac{4\Psi}{N}$ | 0 |
| ZeRO-3 + Offload | $\frac{2\Psi}{N}$ | $4\Psi + \frac{8\Psi}{N}$ |

::: warning 注意
ZeRO-Offload 会增加 CPU-GPU 通信，训练速度会下降。
:::

## ZeRO-Infinity

ZeRO-Infinity 进一步扩展，支持 NVMe SSD 存储：

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",    # 卸载到 NVMe SSD
            "nvme_path": "/local_nvme",
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
        }
    }
}
```

**可训练模型规模**：理论上可达 **Trillion 参数**！

## 通信优化技术

### 1. 通信与计算重叠

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,  # 启用重叠
    }
}
```

### 2. 梯度桶化

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": 5e8,           # Reduce bucket 大小
        "stage3_prefetch_bucket_size": 5e8,  # 预取 bucket 大小
    }
}
```

### 3. 参数预取

在前向传播时，预取下一层的参数：

```
时间 ────────────────────────────────────────────────────────────▶

Layer 0: [Compute]──────────────────────────────────────────────
Layer 1: [Prefetch][Compute]─────────────────────────────────────
Layer 2:          [Prefetch][Compute]────────────────────────────
Layer 3:                   [Prefetch][Compute]───────────────────

通信与计算重叠！
```

## ZeRO vs FSDP

| 特性 | ZeRO-3 | PyTorch FSDP |
|------|--------|--------------|
| 显存优化 | 优秀 | 优秀 |
| 通信优化 | 更成熟 | 发展中 |
| CPU Offload | 支持 | 支持 |
| NVMe Offload | 支持 | 不支持 |
| 易用性 | 需要配置 | 原生支持 |
| 社区支持 | DeepSpeed | PyTorch 官方 |

## 最佳实践

### 选择 ZeRO 级别

```
模型能放入单卡?
  └─ 是 → DDP 或 ZeRO-1
  └─ 否 → 模型 + 优化器状态能放入?
            └─ 是 → ZeRO-2
            └─ 否 → ZeRO-3 或 ZeRO-3 + Offload
```

### 配置建议

```python
# 推荐配置（平衡显存与速度）
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",  # 自动调整
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    }
}
```

## 小结

| 级别 | 分片内容 | 显存占用 | 通信量 | 适用场景 |
|------|---------|---------|--------|---------|
| ZeRO-1 | 优化器状态 | $4\Psi + \frac{8\Psi}{N}$ | $2\Psi$ | 优化器瓶颈 |
| ZeRO-2 | + 梯度 | $4\Psi + \frac{12\Psi}{N}$ | $2\Psi$ | 梯度瓶颈 |
| ZeRO-3 | + 参数 | $\frac{16\Psi}{N}$ | $3\Psi$ | 大模型 |
| Offload | CPU 卸载 | 进一步降低 | 增加 | 超大模型 |

ZeRO 是大模型训练的关键技术，使单卡无法容纳的模型训练成为可能。
