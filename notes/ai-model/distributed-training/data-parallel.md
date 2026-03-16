# 数据并行 (Data Parallelism)

数据并行是最简单、最常用的分布式训练策略。每个设备持有完整的模型副本，但处理不同的数据分片。

## 基本原理

### 核心思想

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据并行原理                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    数据集 D = {D₁, D₂, D₃, D₄}                                 │
│                                                                 │
│    GPU 0          GPU 1          GPU 2          GPU 3          │
│    ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐        │
│    │Model│        │Model│        │Model│        │Model│        │
│    │  ↓  │        │  ↓  │        │  ↓  │        │  ↓  │        │
│    │ D₁  │        │ D₂  │        │ D₃  │        │ D₄  │        │
│    │  ↓  │        │  ↓  │        │  ↓  │        │  ↓  │        │
│    │Grad₁│        │Grad₂│        │Grad₃│        │Grad₄│        │
│    └──┬──┘        └──┬──┘        └──┬──┘        └──┬──┘        │
│       │              │              │              │           │
│       └──────────────┴──────┬───────┴──────────────┘           │
│                             │                                   │
│                    All-Reduce (平均)                            │
│                             │                                   │
│                             ▼                                   │
│                    Avg(Grad₁,Grad₂,Grad₃,Grad₄)                │
│                             │                                   │
│              ┌──────────────┴──────────────┐                   │
│              ▼              ▼              ▼                    │
│           更新模型参数（所有 GPU 同步）                         │
└─────────────────────────────────────────────────────────────────┘
```

### 梯度同步公式

设有 $N$ 个 GPU，每个 GPU 计算的梯度为 $g_i$，同步后的梯度为：

$$\bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i$$

然后所有 GPU 使用相同的 $\bar{g}$ 更新参数，保证模型一致性。

## PyTorch DDP (DistributedDataParallel)

### DDP vs DP

PyTorch 提供两种数据并行实现：

| 特性 | DP (DataParallel) | DDP (DistributedDataParallel) |
|------|-------------------|------------------------------|
| 进程 | 单进程多线程 | 多进程 |
| 通信 | 单线程 GIL 限制 | 无 GIL 限制 |
| 性能 | 较差 | 更好 |
| 扩展性 | 单节点 | 多节点 |
| 推荐 | ❌ 不推荐 | ✅ 推荐 |

### DDP 工作流程

```
1. Forward Pass:
   每个 GPU 独立计算前向传播
   
2. Backward Pass:
   每个 GPU 独立计算梯度
   DDP 构造梯度桶 (bucket)
   
3. Gradient Synchronization:
   All-Reduce 同步梯度桶
   计算与通信重叠
   
4. Optimizer Step:
   所有 GPU 使用相同梯度更新参数
```

### DDP 完整代码示例

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def setup_distributed():
    """初始化分布式环境"""
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    """简单的 CNN 模型示例"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train():
    # 1. 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    
    # 2. 创建模型并移到 GPU
    model = SimpleModel().cuda()
    
    # 3. 用 DDP 包装模型
    # device_ids 必须指定，否则会有性能问题
    model = DDP(model, device_ids=[local_rank])
    
    # 4. 准备数据
    # 重要：使用 DistributedSampler 确保数据分片不重叠
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    
    # DistributedSampler: 每个 GPU 获取不同的数据分片
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # 每个 epoch 会自动 shuffle
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=64,          # 每个 GPU 的 batch size
        sampler=sampler,        # 使用分布式采样器
        num_workers=4,
        pin_memory=True
    )
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. 训练循环
    for epoch in range(10):
        # 重要：每个 epoch 设置 sampler 的 epoch
        # 确保每个 epoch 的 shuffle 不同
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if rank == 0:  # 只在主进程打印
            print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    
    # 7. 保存模型（只在主进程保存）
    if rank == 0:
        torch.save(model.module.state_dict(), 'model.pth')
    
    cleanup()


if __name__ == "__main__":
    train()
```

### 启动脚本

```bash
# 单节点 4 GPU
torchrun --nproc_per_node=4 train_ddp.py

# 多节点（2 节点，每节点 4 GPU）
# 节点 0（主节点）:
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr="10.0.0.1" --master_port=29500 \
    train_ddp.py

# 节点 1:
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=1 --master_addr="10.0.0.1" --master_port=29500 \
    train_ddp.py
```

## DDP 关键技术

### 梯度桶 (Gradient Bucketing)

DDP 将梯度按参数分组到桶中，实现计算与通信重叠：

```
┌─────────────────────────────────────────────────────────────────┐
│                     梯度桶机制                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  时间轴 ────────────────────────────────────────────────────▶  │
│                                                                 │
│  GPU 计算:  [Layer 4] [Layer 3] [Layer 2] [Layer 1]             │
│                    ↓         ↓         ↓         ↓              │
│  通信:            [Bucket 3] [Bucket 2] [Bucket 1]              │
│                              ↓         ↓         ↓              │
│                           AllReduce AllReduce AllReduce         │
│                                                                 │
│  计算梯度的同时，已经计算完的梯度桶开始通信                       │
└─────────────────────────────────────────────────────────────────┘
```

### 桶大小配置

```python
# 默认桶大小 25MB，可通过环境变量调整
# 较大的桶减少通信次数，但延迟通信开始时间
os.environ["NCCL_BUCKET_SIZE"] = "52428800"  # 50MB

model = DDP(
    model, 
    device_ids=[local_rank],
    bucket_cap_mb=25,  # 桶大小 (MB)
    find_unused_parameters=False,  # 是否检测未使用参数
)
```

### 梯度累积

DDP 支持梯度累积，实现更大的等效 batch size：

```python
# 累积 4 次再更新，等效 batch_size = 64 * 4 * world_size
accumulation_steps = 4

optimizer.zero_grad()

for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()
    
    output = model(data)
    loss = criterion(output, target) / accumulation_steps  # 重要：loss 缩放
    loss.backward()
    
    # 每 accumulation_steps 步更新一次
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## FSDP (Fully Sharded Data Parallel)

FSDP 是 DDP 的升级版，通过**分片存储**突破单卡显存限制。

### DDP vs FSDP

```
DDP (每卡存储完整模型):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Model   │  │  Model   │  │  Model   │  │  Model   │
│  Optim   │  │  Optim   │  │  Optim   │  │  Optim   │
│  Grad    │  │  Grad    │  │  Grad    │  │  Grad    │
│   ↓      │  │   ↓      │  │   ↓      │  │   ↓      │
│ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

FSDP (每卡只存储分片):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Shard 0  │  │ Shard 1  │  │ Shard 2  │  │ Shard 3  │
│   ↓      │  │   ↓      │  │   ↓      │  │   ↓      │
│ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### FSDP 分片策略

| 策略 | 说明 | 显存占用 |
|------|------|---------|
| FULL_SHARD | 分片：参数、梯度、优化器状态 | 最低 |
| SHARD_GRAD_OP | 分片：梯度、优化器状态 | 中等 |
| NO_SHARD | 不分片（类似 DDP） | 最高 |

### FSDP 代码示例

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class LargeModel(nn.Module):
    """大型模型示例"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(100)
        ])
        self.output = nn.Linear(4096, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


def train_fsdp():
    setup_distributed()
    
    model = LargeModel().cuda()
    
    # FSDP 配置
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
        auto_wrap_policy=size_based_auto_wrap_policy,   # 自动包装
        device_id=torch.cuda.current_device(),
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(10):
        model.train()
        # ... 数据加载和训练 ...
        
        # 保存模型
        if rank == 0:
            # FSDP 需要用专门的方法保存
            from torch.distributed.fsdp import StateDictType
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                state_dict = model.state_dict()
                torch.save(state_dict, 'model_fsdp.pth')
    
    cleanup()
```

### FSDP 混合精度训练

```python
from torch.distributed.fsdp import MixedPrecision

# 定义混合精度策略
mp_policy = MixedPrecision(
    param_dtype=torch.float16,      # 参数使用 FP16
    reduce_dtype=torch.float16,     # 梯度归约使用 FP16
    buffer_dtype=torch.float32,     # Buffer 使用 FP32
)

model = FSDP(
    model,
    mixed_precision=mp_policy,
)
```

### FSDP 激活检查点

进一步降低显存占用：

```python
from torch.distributed.fsdp import CPUOffload
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# 包装每层使用激活检查点
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(100):
            layer = nn.Linear(4096, 4096)
            # 使用激活检查点包装
            layer = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            self.layers.append(layer)
```

## 显存分析

### DDP 显存占用

$$M_{\text{DDP}} = M_{\text{model}} + M_{\text{grad}} + M_{\text{optim}} + M_{\text{activation}}$$

以 FP32 训练 Adam 优化器为例：

- 模型参数：$\Psi$ 个，占用 $4\Psi$ 字节
- 梯度：$\Psi$ 个，占用 $4\Psi$ 字节
- 优化器状态：$m$ 和 $v$ 两个动量，占用 $8\Psi$ 字节
- 激活值：与 batch size 和模型结构相关

**总计：$16\Psi$ 字节**（不含激活值）

### FSDP 显存占用

$$M_{\text{FSDP}} = \frac{16\Psi}{N} + M_{\text{activation}}$$

其中 $N$ 是 GPU 数量。

::: tip 示例计算
**175B 参数模型**：
- DDP 单卡显存：$175 \times 10^9 \times 16 = 2.8$ TB（不可行）
- FSDP (512 GPU)：$2.8 \text{ TB} / 512 \approx 5.5$ GB（可行）
:::

## 性能优化建议

### 1. 数据加载优化

```python
# 使用多进程加载，pin_memory 加速 GPU 传输
dataloader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=4,        # 根据 CPU 核心数调整
    pin_memory=True,      # 固定内存，加速传输
    prefetch_factor=2,    # 预取因子
    persistent_workers=True,  # 保持 worker 进程
)
```

### 2. 通信优化

```python
# 使用梯度压缩（适用于低带宽场景）
from torch.distributed.algorithms import Quantization

# 配置梯度压缩
ddp_state_dict = model.state_dict()
# ... 实现梯度压缩
```

### 3. 混合精度训练

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 小结

| 特性 | DDP | FSDP |
|------|-----|------|
| 显存占用 | 每卡完整模型 | 每卡分片 |
| 通信量 | All-Reduce 梯度 | All-Gather 参数 + Reduce-Scatter 梯度 |
| 适用模型 | 中小模型 | 大模型 |
| 实现复杂度 | 简单 | 中等 |
| PyTorch 版本 | 1.x 支持 | 1.12+ 推荐 |

**选择建议**：
- 模型能放入单卡 → DDP
- 模型无法放入单卡 → FSDP
- 追求极致性能 → 考虑 ZeRO + DeepSpeed
