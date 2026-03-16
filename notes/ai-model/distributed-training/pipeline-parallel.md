# 流水线并行 (Pipeline Parallelism)

流水线并行（Pipeline Parallelism, PP）将模型的**不同层**切分到不同设备上，形成流水线执行。相比张量并行，PP 通信量小，适合跨节点训练。

## 基本原理

### 模型切分

将模型按层切分到多个设备：

```
┌─────────────────────────────────────────────────────────────────┐
│                    流水线并行切分                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  完整模型:  [Layer 0] → [Layer 1] → ... → [Layer 23]           │
│                                                                 │
│  4 卡流水线切分:                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  GPU 0   │   │  GPU 1   │   │  GPU 2   │   │  GPU 3   │    │
│  │ Layers   │   │ Layers   │   │ Layers   │   │ Layers   │    │
│  │  0-5     │ → │  6-11    │ → │  12-17   │ → │  18-23   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       Stage 0        Stage 1       Stage 2        Stage 3      │
└─────────────────────────────────────────────────────────────────┘
```

### 朴素流水线的问题

最简单的流水线执行方式（朴素流水线）：

```
时间 ────────────────────────────────────────────────────────────▶

GPU 0: [F0]────────────────────────────────────────────────[F0']──
GPU 1: ─────[F1]─────────────────────────────────────[F1']────────
GPU 2: ───────────[F2]────────────────────────[F2']───────────────
GPU 3: ────────────────[F3]─────────────[F3']──────────────────────

F = Forward, F' = Forward (下一个 micro-batch)

问题：严重的"气泡"（GPU 空闲时间）！
```

## GPipe

GPipe 通过**微批次（Micro-batch）**技术减少气泡。

### 微批次切分

将一个 batch 分成多个 micro-batch：

```
原始 batch: B = 64
Micro-batch 数: m = 8
每个 micro-batch: b = 64/8 = 8
```

### GPipe 执行流程

```
时间 ──────────────────────────────────────────────────────────────▶

GPU 0: [F0][F1][F2][F3][F4][F5][F6][F7]──────────────────────────
GPU 1: ───[F0][F1][F2][F3][F4][F5][F6][F7]───────────────────────
GPU 2: ───────[F0][F1][F2][F3][F4][F5][F6][F7]───────────────────
GPU 3: ──────────[F0][F1][F2][F3][F4][F5][F6][F7][B0][B1]...[B7]─
                                                          ↑
                                               所有前向完成后开始反向

问题：反向传播时仍有较大气泡
```

### GPipe 气泡分析

设 micro-batch 数为 $m$，流水线深度为 $p$：

$$\text{Bubble Ratio} = \frac{p-1}{m + p - 1}$$

当 $m \gg p$ 时，气泡比例趋近于 0。

### GPipe 激活重计算

GPipe 的关键优化：**只保存 micro-batch 边界的激活，反向时重计算**。

```python
# 训练时只保存分割点激活
activations_to_save = []

for micro_batch in micro_batches:
    # 前向传播，只保存分割点
    activation = model_forward(micro_batch)
    if is_boundary:
        activations_to_save.append(activation)

# 反向传播时重计算中间激活
for activation in activations_to_save:
    # 从保存的激活重新计算
    recomputed = recompute_forward(activation)
    backward(recomputed)
```

### GPipe 显存优化

$$M_{\text{activation}} = \frac{m \times b \times L \times d}{p}$$

通过激活重计算，显存占用降低约 $p$ 倍。

## PipeDream

PipeDream 提出 **1F1B（One Forward One Backward）** 调度，进一步减少气泡。

### 1F1B 调度

交替执行前向和反向传播：

```
时间 ──────────────────────────────────────────────────────────────▶

GPU 0: [F0][F1][F2][F3][F4][B0][F5][B1][F6][B2][F7][B3][B4][B5][B6][B7]
GPU 1: ───[F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][B4][B5][B6][B7]─
GPU 2: ───────[F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][B4][B5][B6][B7]
GPU 3: ──────────[F0][F1][B0][F2][B1][F3][B2][F4][B3][B5][B4][B7][B6]

F = Forward, B = Backward
前向和反向交替执行，气泡更小
```

### 气泡对比

| 方法 | 气泡比例 |
|------|---------|
| 朴素流水线 | $(p-1)/p$ |
| GPipe | $(p-1)/(m+p-1)$ |
| 1F1B | $(p-1)/(m+p-1)$（但激活存储更少）|

### PipeDream 激活管理

PipeDream 使用 **激活存储优化**：

```python
# 稳定状态：只需存储 p 个 micro-batch 的激活
# 而非 GPipe 的 m 个

stored_activations = deque(maxlen=p)  # 最多存储 p 个

def step():
    # 执行一个前向
    act = forward()
    stored_activations.append(act)
    
    # 执行一个反向
    if len(stored_activations) >= p:
        oldest_act = stored_activations.popleft()
        backward(oldest_act)
```

## Interleaved Pipeline (交错流水线)

Megatron-LM 提出交错流水线，进一步减少气泡。

### 基本思想

每个设备负责**多个不连续的层块**：

```
标准流水线:
GPU 0: [Layer 0-5]
GPU 1: [Layer 6-11]
GPU 2: [Layer 12-17]
GPU 3: [Layer 18-23]

交错流水线 (每个 GPU 负责 2 个块):
GPU 0: [Layer 0-2] + [Layer 12-14]
GPU 1: [Layer 3-5] + [Layer 15-17]
GPU 2: [Layer 6-8] + [Layer 18-20]
GPU 3: [Layer 9-11] + [Layer 21-23]
```

### 交错流水线执行

```
时间 ──────────────────────────────────────────────────────────────▶

GPU 0: [F0][F0'][F1][F1'][B0][F2][F2'][B0'][B1][F3][F3'][B1'][B2]...
       ↑      ↑
    第一个块  第二个块

通信点更多，但气泡更小
```

### 气泡减少

$$\text{Bubble Ratio}_{\text{interleaved}} = \frac{p-1}{v \times (m + v \times (p-1))}$$

其中 $v$ 是每个设备的块数。

## PyTorch 实现

### 基础流水线并行

```python
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe


# 定义模型层
class LayerBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))


def create_pipeline_model():
    # 创建 4 层模型
    layers = nn.Sequential(
        LayerBlock(1024, 2048),  # Stage 0
        LayerBlock(2048, 2048),  # Stage 1
        LayerBlock(2048, 2048),  # Stage 2
        LayerBlock(2048, 10),    # Stage 3
    )
    
    # 将模型转换为流水线并行
    # chunks = micro-batch 数
    model = Pipe(layers, chunks=8)
    
    return model


def train_pipeline():
    import torch.distributed as dist
    import os
    
    # 初始化分布式
    dist.init_process_group(backend='gloo')  # 流水线可以用 gloo
    rank = int(os.environ.get("RANK", 0))
    
    model = create_pipeline_model()
    
    # 流水线会自动将层分配到不同设备
    # 每个 rank 负责一部分层
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        x = torch.randn(64, 1024)  # 输入
        y = torch.randint(0, 10, (64,))  # 标签
        
        optimizer.zero_grad()
        
        # 流水线前向传播
        output = model(x)
        
        # 只在最后一个 stage 计算损失
        if model.group.rank() == model.group.size() - 1:
            loss = criterion(output, y)
            loss.backward()
        
        optimizer.step()
```

### 使用 torchrun 启动

```bash
# 4 个 GPU 的流水线并行
torchrun --nproc_per_node=4 train_pipeline.py
```

## Megatron-LM 流水线实现

```python
import torch
import torch.nn as nn
from typing import List, Optional


class PipelineStage(nn.Module):
    """单个流水线阶段"""
    
    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
    
    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class PipelineParallel(nn.Module):
    """简化的流水线并行实现"""
    
    def __init__(
        self, 
        layers: nn.ModuleList, 
        num_stages: int,
        num_micro_batches: int,
        rank: int
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.rank = rank
        
        # 划分层到各个阶段
        layers_per_stage = len(layers) // num_stages
        start_idx = rank * layers_per_stage
        end_idx = start_idx + layers_per_stage
        
        self.stage = PipelineStage(
            layers[start_idx:end_idx],
            rank,
            num_stages
        )
        
        # 存储激活用于反向传播
        self.saved_activations = []
    
    def forward(self, x: torch.Tensor):
        """执行流水线前向传播"""
        outputs = []
        
        for mb in range(self.num_micro_batches):
            # 获取当前 micro-batch
            micro_input = x[mb::self.num_micro_batches]
            
            if self.rank == 0:
                # 第一个阶段：直接处理输入
                hidden = micro_input
            else:
                # 其他阶段：接收上一个阶段的输出
                hidden = self._recv_from_prev_stage()
            
            # 通过当前阶段的层
            output = self.stage(hidden)
            
            if self.rank == self.num_stages - 1:
                # 最后一个阶段：保存输出
                outputs.append(output)
            else:
                # 发送到下一个阶段
                self._send_to_next_stage(output)
            
            # 保存激活用于反向传播
            self.saved_activations.append(output.detach().requires_grad_(True))
        
        return outputs
    
    def _send_to_next_stage(self, tensor: torch.Tensor):
        """发送到下一个阶段"""
        import torch.distributed as dist
        dist.send(tensor.contiguous(), dst=self.rank + 1)
    
    def _recv_from_prev_stage(self) -> torch.Tensor:
        """从上一个阶段接收"""
        import torch.distributed as dist
        tensor = torch.empty_like(self.saved_activations[0])
        dist.recv(tensor, src=self.rank - 1)
        return tensor


def create_pipeline_model(
    num_layers: int,
    hidden_size: int,
    num_stages: int,
    num_micro_batches: int,
    rank: int
):
    """创建流水线并行模型"""
    
    # 创建所有层
    layers = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ) for _ in range(num_layers)
    ])
    
    # 创建流水线并行模型
    model = PipelineParallel(
        layers=layers,
        num_stages=num_stages,
        num_micro_batches=num_micro_batches,
        rank=rank
    )
    
    return model
```

## 通信分析

### 通信量

设隐藏维度为 $h$，batch size 为 $b$，流水线深度为 $p$：

$$\text{通信量} = \frac{b \times h}{p} \times 2 \times p = 2bh$$

与流水线深度 $p$ 无关！

### 与张量并行对比

| 策略 | 单层通信量 | 通信频率 | 适合场景 |
|------|-----------|---------|---------|
| 张量并行 | $O(bsh/t)$ | 每层 | 节点内（NVLink）|
| 流水线并行 | $O(bh)$ | 阶段间 | 跨节点 |

::: tip 关键洞察
流水线并行的通信量只与**激活大小**有关，与层数无关，非常适合深度网络！
:::

## 流水线并行的挑战

### 1. 气泡问题

- 增加微批次数量可减少气泡
- 但会增加内存占用（需存储更多激活）

### 2. 负载均衡

```
理想情况：每个阶段计算量相等

实际问题：
- Embedding 层计算轻
- Attention 计算重
- 需要精心划分层
```

### 3. 批归一化问题

- BN 统计量需要在所有 GPU 上同步
- 或改用 LayerNorm

## 小结

| 特性 | GPipe | PipeDream (1F1B) | Interleaved |
|------|-------|------------------|-------------|
| 调度 | Fill-Drain | 交替执行 | 交错执行 |
| 气泡 | 较大 | 较小 | 最小 |
| 激活存储 | 所有 micro-batch | 限制数量 | 限制数量 |
| 实现复杂度 | 简单 | 中等 | 复杂 |

**最佳实践**：

1. 流水线深度不宜过大（通常 4-8）
2. 微批次数量应为流水线深度的倍数
3. 与张量并行组合使用（TP 用于节点内，PP 用于跨节点）
4. 使用激活检查点减少内存占用
