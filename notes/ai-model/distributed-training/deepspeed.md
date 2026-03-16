# DeepSpeed 原理与实践

DeepSpeed 是 Microsoft 开源的深度学习优化库，提供 ZeRO、混合精度训练、梯度累积等功能，是大模型训练的核心工具。

## DeepSpeed 概述

### 核心特性

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSpeed 核心特性                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   ZeRO      │  │   混合精度   │  │  梯度累积   │             │
│  │  显存优化   │  │   FP16/BF16 │  │  大Batch    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  流水线并行  │  │  张量并行   │  │   Offload   │             │
│  │    PP       │  │    TP       │  │  CPU/NVMe   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  激活检查点  │  │  通信优化   │  │   推理加速   │             │
│  │  Activation │  │  Overlap    │  │  Kernel    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 安装

```bash
pip install deepspeed

# 验证安装
ds_report
```

## 快速开始

### 最简配置

```python
import torch
import torch.nn as nn
import deepspeed


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# DeepSpeed 配置字典
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    }
}

# 初始化
model = SimpleModel()
parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config=ds_config
)

# 训练循环
for batch in dataloader:
    inputs, labels = batch
    inputs = inputs.to(model_engine.device)
    
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)
    
    model_engine.backward(loss)
    model_engine.step()  # 自动处理梯度清零
```

### 启动脚本

```bash
# 单 GPU
deepspeed train.py --deepspeed_config ds_config.json

# 多 GPU
deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json

# 多节点
deepspeed --num_nodes=2 --num_gpus=4 \
    --hostfile=hostfile \
    train.py --deepspeed_config ds_config.json
```

## 配置文件详解

### 完整配置模板

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 2,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 10000,
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 500
        }
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": false
}
```

### 关键配置项说明

#### 批次大小配置

```json
{
    "train_batch_size": 64,           // 全局批次大小
    "train_micro_batch_size_per_gpu": 8,  // 每个 GPU 的微批次大小
    "gradient_accumulation_steps": 2  // 梯度累积步数
}
```

三者的关系：

$$\text{train\_batch\_size} = \text{micro\_batch\_size} \times \text{gradient\_accumulation} \times \text{num\_gpus}$$

#### 混合精度配置

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,              // 0 = 动态 loss scale
        "initial_scale_power": 16,    // 初始 scale = 2^16
        "loss_scale_window": 1000,    // 动态调整窗口
        "hysteresis": 2,              // 溢出容忍次数
        "min_loss_scale": 1           // 最小 scale
    }
}
```

::: tip FP16 vs BF16
- **FP16**：需要 loss scale 防止梯度下溢
- **BF16**：动态范围更大，通常不需要 loss scale
:::

#### ZeRO 配置

```json
{
    "zero_optimization": {
        "stage": 2,                   // ZeRO 级别: 1, 2, 3
        "offload_optimizer": {
            "device": "cpu",          // "cpu" 或 "none"
            "pin_memory": true        // 固定内存加速
        },
        "overlap_comm": true,         // 通信与计算重叠
        "contiguous_gradients": true, // 连续存储梯度
        "reduce_bucket_size": 5e8,    // Reduce bucket 大小
        "allgather_bucket_size": 5e8  // AllGather bucket 大小
    }
}
```

## 3D 并行配置

### DeepSpeed + Megatron-LM 风格

```python
# 启动命令
deepspeed --num_gpus=8 --num_nodes=2 \
    train.py \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 4 \
    --deepspeed_config ds_config.json
```

### 配置文件

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    
    "zero_optimization": {
        "stage": 1,  // 与 TP/PP 组合时通常用 ZeRO-1
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    
    "tensor_parallel": {
        "tp_size": 2
    },
    
    "pipeline_parallel": {
        "pp_size": 4,
        "num_micro_batches": 8
    },
    
    "fp16": {
        "enabled": true
    }
}
```

### 并行配置计算

```
总 GPU 数: 16 (2 节点 × 8 GPU)
TP = 2 (张量并行度)
PP = 4 (流水线并行度)
DP = 16 / (2 × 4) = 2 (数据并行度)

每卡批次: 256 / 2 / 4 / 2 = 16
```

## 检查点管理

### 保存检查点

```python
def save_checkpoint(model_engine, epoch, step, save_dir):
    """保存 DeepSpeed 检查点"""
    
    # DeepSpeed 会保存:
    # - 模型参数分片
    # - 优化器状态
    # - 学习率调度器状态
    
    save_path = f"{save_dir}/checkpoint-{epoch}-{step}"
    
    model_engine.save_checkpoint(
        save_dir=save_path,
        client_state={
            "epoch": epoch,
            "step": step,
            # 其他需要保存的状态
        }
    )
```

### 加载检查点

```python
def load_checkpoint(model_engine, load_dir, load_optimizer=True):
    """加载 DeepSpeed 检查点"""
    
    # 加载检查点
    _, client_state = model_engine.load_checkpoint(
        load_dir=load_dir,
        load_optimizer_states=load_optimizer,
        load_lr_scheduler_states=True,
        load_module_strict=True
    )
    
    return client_state


# 使用示例
model_engine, _, _, _ = deepspeed.initialize(...)
client_state = load_checkpoint(model_engine, "./checkpoint")

# 恢复训练
start_epoch = client_state.get("epoch", 0)
start_step = client_state.get("step", 0)
```

### ZeRO-3 完整模型导出

```python
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    convert_zero_checkpoint_to_fp32_state_dict
)

# 方法1: 从检查点导出完整模型
convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir="./checkpoint",
    output_file="model_fp32.pt"
)

# 方法2: 获取完整 state_dict
state_dict = get_fp32_state_dict_from_zero_checkpoint(
    checkpoint_dir="./checkpoint"
)
torch.save(state_dict, "model_fp32.pt")
```

## 激活检查点

激活检查点通过**重计算**减少激活内存：

```python
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

def custom_forward(x):
    """需要检查点的计算"""
    return expensive_computation(x)

# 使用激活检查点
output = checkpoint(custom_forward, x)
```

### 配置激活检查点

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 100,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

## 混合专家模型 (MoE)

DeepSpeed 支持混合专家模型训练：

```python
import deepspeed
from deepspeed.moe.layer import MoE

class MoEModel(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2):
        super().__init__()
        self.moe = MoE(
            hidden_size=hidden_size,
            expert=ExpertLayer(hidden_size),  # 专家网络
            num_experts=num_experts,
            k=top_k,  # Top-K 路由
        )
    
    def forward(self, x):
        output, gate_loss, _ = self.moe(x)
        return output, gate_loss
```

### MoE 配置

```json
{
    "train_batch_size": 64,
    "moe": {
        "enabled": true,
        "moed": {
            "ep_size": 8
        }
    }
}
```

## 性能调优

### 1. 通信优化

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto"
    }
}
```

### 2. 显存优化

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### 3. 吞吐量优化

```json
{
    "train_micro_batch_size_per_gpu": "auto",  // 自动调整
    "gradient_accumulation_steps": "auto",
    
    "gradient_clipping": 1.0,
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    }
}
```

## 监控与调试

### 训练监控

```python
# 在训练循环中
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
    
    # DeepSpeed 自动打印进度
    # 可以通过配置 steps_per_print 控制频率
```

### 性能分析

```json
{
    "wall_clock_breakdown": true,
    "flops_profiler": {
        "enabled": true,
        "profile_step": 100,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": true
    }
}
```

### 日志配置

```python
import deepspeed

# 启用详细日志
deepspeed.utils.logger.setLevel("DEBUG")
```

## 常见问题

### 1. CUDA Out of Memory

```python
# 解决方案1: 降低批次大小
"train_micro_batch_size_per_gpu": 4,

# 解决方案2: 增加梯度累积
"gradient_accumulation_steps": 4,

# 解决方案3: 升级 ZeRO 级别
"zero_optimization": {"stage": 3},

# 解决方案4: 启用 Offload
"offload_optimizer": {"device": "cpu"},
```

### 2. 通信超时

```python
# 增加 NCCL 超时时间
import os
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 分钟
```

### 3. Loss 为 NaN

```python
# 混合精度问题
"fp16": {
    "enabled": true,
    "loss_scale": 0,  // 使用动态 loss scale
    "initial_scale_power": 12,  // 降低初始 scale
}
```

## 完整训练示例

```python
import os
import torch
import torch.nn as nn
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # 初始化分布式
    deepspeed.init_distributed()
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 12
        },
        
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6
        },
        
        "gradient_clipping": 1.0,
        "steps_per_print": 100
    }
    
    # 初始化 DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 训练循环
    for epoch in range(10):
        for step, batch in enumerate(dataloader):
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
            inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}
            
            outputs = model_engine(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
        
        # 保存检查点
        model_engine.save_checkpoint(f"./checkpoint/epoch_{epoch}")


if __name__ == "__main__":
    main()
```

## 小结

| 功能 | 配置项 | 作用 |
|------|-------|------|
| ZeRO | `zero_optimization.stage` | 显存优化 |
| 混合精度 | `fp16.enabled` | 加速训练 |
| 梯度累积 | `gradient_accumulation_steps` | 等效大批次 |
| Offload | `offload_optimizer.device` | CPU 卸载 |
| 通信优化 | `overlap_comm` | 隐藏通信延迟 |
| 激活检查点 | `activation_checkpointing` | 减少激活内存 |

DeepSpeed 是大模型训练的核心工具，掌握其配置和调优对于高效训练至关重要。
