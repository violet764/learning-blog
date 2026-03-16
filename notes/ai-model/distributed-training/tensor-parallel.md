# 张量并行 (Tensor Parallelism)

张量并行（Tensor Parallelism, TP）是一种**模型并行**策略，将模型单层的参数切分到多个设备上，实现层内并行计算。Megatron-LM 是最著名的实现。

## 核心思想

### 为什么需要张量并行？

数据并行的限制：
- 每个设备必须存储**完整的模型副本**
- 模型大小受限于单卡显存

张量并行解决方案：
- 将**单层参数切分**到多个设备
- 每个设备只存储部分参数
- 通过通信协作完成计算

```
┌─────────────────────────────────────────────────────────────────┐
│                    张量并行 vs 数据并行                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数据并行 - 每卡完整模型，数据分片:                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ [W₁,W₂] │  │ [W₁,W₂] │  │ [W₁,W₂] │  │ [W₁,W₂] │            │
│  │   ↓     │  │   ↓     │  │   ↓     │  │   ↓     │            │
│  │  D₁     │  │  D₂     │  │  D₃     │  │  D₄     │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                                                                 │
│  张量并行 - 参数分片，数据相同:                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │   W₁    │  │   W₂    │  │   W₃    │  │   W₄    │            │
│  │   ↓     │  │   ↓     │  │   ↓     │  │   ↓     │            │
│  │  D      │  │  D      │  │  D      │  │  D      │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 矩阵乘法的并行化

张量并行基于矩阵乘法的数学性质。对于线性变换 $Y = XW$：

### 行并行 (Row Parallel)

将权重 $W$ 按行切分，结果需要 All-Reduce：

$$Y = XW = X \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = XW_1 + XW_2$$

```
行并行:
     X         W₁          W₂
     │          │           │
     ├──────────┼───────────┤
     │          ↓           ↓
     │        Y₁=XW₁     Y₂=XW₂
     │          │           │
     │          └─────┬─────┘
     │                ↓
     │          All-Reduce
     │                ↓
     └──────────────▶ Y = Y₁ + Y₂
```

### 列并行 (Column Parallel)

将权重 $W$ 按列切分，输入需要复制：

$$Y = XW = X \begin{bmatrix} W_1 & W_2 \end{bmatrix} = \begin{bmatrix} XW_1 & XW_2 \end{bmatrix}$$

```
列并行:
     X ─────┬─────────────┐
            │             │
            ↓             ↓
          XW₁           XW₂
            │             │
            └─────┬───────┘
                  ↓
             All-Gather
                  ↓
              Y = [Y₁, Y₂]
```

## Megatron-LM 并行化方案

Megatron-LM 巧妙地组合行并行和列并行，减少通信次数。

### MLP 层并行化

标准 MLP：两个线性层 + 激活函数

$$Y = \text{GELU}(XW_1)W_2$$

Megatron 并行化方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLP 张量并行                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│        GPU 0                    GPU 1                           │
│    ┌──────────────┐        ┌──────────────┐                    │
│    │   W₁ᵃ        │        │   W₁ᵇ        │  列并行           │
│    │    ↓         │        │    ↓         │                    │
│ X ─┼─▶ XW₁ᵃ       │    ───┼─▶ XW₁ᵇ       │                    │
│    │    ↓         │        │    ↓         │                    │
│    │  GELU(XW₁ᵃ)  │        │  GELU(XW₁ᵇ)  │                    │
│    │    ↓         │        │    ↓         │                    │
│    │   W₂ᵃ        │        │   W₂ᵇ        │  行并行           │
│    │    ↓         │        │    ↓         │                    │
│    │ Yᵃ = G(XW₁ᵃ)W₂ᵃ      │ Yᵇ = G(XW₁ᵇ)W₂ᵇ                   │
│    └──────┬───────┘        └──────┬───────┘                    │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ↓                                         │
│                 All-Reduce                                      │
│                       ↓                                         │
│                   Y = Yᵃ + Yᵇ                                   │
└─────────────────────────────────────────────────────────────────┘
```

**关键洞察**：列并行的输出是分块的，直接作为行并行的输入，中间无需通信！

### 数学推导

设 2 个 GPU：

**第一层（列并行）**：
$$Y_1^{(a)} = \text{GELU}(XW_1^{(a)}), \quad Y_1^{(b)} = \text{GELU}(XW_1^{(b)})$$

**第二层（行并行）**：
$$Y_2^{(a)} = Y_1^{(a)}W_2^{(a)}, \quad Y_2^{(b)} = Y_1^{(b)}W_2^{(b)}$$

**最终结果**：
$$Y = Y_2^{(a)} + Y_2^{(b)} = \text{GELU}(XW_1)W_2$$

只需要一次 All-Reduce！

### Self-Attention 并行化

对于 Attention 层 $Q, K, V$ 投影：

```
┌─────────────────────────────────────────────────────────────────┐
│                  Attention 张量并行                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│        GPU 0                    GPU 1                           │
│    ┌──────────────┐        ┌──────────────┐                    │
│    │  Wqᵃ,Wkᵃ,Wvᵃ │        │  Wqᵇ,Wkᵇ,Wvᵇ │  列并行           │
│    │      ↓       │        │      ↓       │                    │
│    │ Qᵃ,Kᵃ,Vᵃ     │        │ Qᵇ,Kᵇ,Vᵇ     │                    │
│    │      ↓       │        │      ↓       │                    │
│    │  Attention   │        │  Attention   │  各自计算          │
│    │   head a     │        │   head b     │                    │
│    │      ↓       │        │      ↓       │                    │
│    │    Woᵃ       │        │    Woᵇ       │  行并行           │
│    │      ↓       │        │      ↓       │                    │
│    │    Outᵃ      │        │    Outᵇ      │                    │
│    └──────┬───────┘        └──────┬───────┘                    │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ↓                                         │
│                 All-Reduce                                      │
│                       ↓                                         │
│                   Out = Outᵃ + Outᵇ                             │
└─────────────────────────────────────────────────────────────────┘
```

## 代码实现

### 基础张量并行线性层

```python
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """列并行线性层：权重按列切分"""
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        
        # 每个设备只存储 1/world_size 的输出维度
        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size
        
        # 权重形状: [out_features/N, in_features]
        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.out_features_per_partition)
        )
    
    def forward(self, x):
        # x: [batch, in_features]
        # 输出: [batch, out_features/N]
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """行并行线性层：权重按行切分"""
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        
        # 每个设备只存储 1/world_size 的输入维度
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        
        # 权重形状: [out_features, in_features/N]
        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_partition)
        )
    
    def forward(self, x):
        # x: [batch, in_features/N]
        # 局部输出: [batch, out_features]
        output = torch.nn.functional.linear(x, self.weight)
        
        # All-Reduce 汇总所有设备的结果
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output
```

### Megatron 风格 MLP

```python
class TensorParallelMLP(nn.Module):
    """Megatron 风格的张量并行 MLP"""
    
    def __init__(self, hidden_size, intermediate_size, world_size):
        super().__init__()
        self.world_size = world_size
        
        # 第一个线性层：列并行（输出维度切分）
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size
        )
        
        # 第二个线性层：行并行（输入维度切分）
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, world_size
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: [batch, hidden_size]
        
        # 列并行：输出分块
        h = self.fc1(x)  # [batch, intermediate_size/N]
        h = self.activation(h)
        
        # 行并行：输入分块，输出 All-Reduce
        out = self.fc2(h)  # [batch, hidden_size]
        
        return out
```

### 张量并行 Self-Attention

```python
class TensorParallelAttention(nn.Module):
    """张量并行注意力层"""
    
    def __init__(self, hidden_size, num_heads, world_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.world_size = world_size
        
        # 每个设备处理 num_heads/N 个注意力头
        assert num_heads % world_size == 0
        self.num_heads_per_partition = num_heads // world_size
        
        # QKV 投影：列并行
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, world_size
        )
        
        # 输出投影：行并行
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, world_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        qkv = self.qkv_proj(x)  # [batch, seq, 3*hidden_size/N]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        
        # 转置: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 重塑: [batch, seq, hidden_size/N]
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)
        
        # 输出投影（行并行，自动 All-Reduce）
        out = self.out_proj(out)
        
        return out
```

## 通信分析

### 通信量计算

假设隐藏维度为 $h$，序列长度为 $s$，batch size 为 $b$，TP 度为 $t$：

| 操作 | 通信量 | 类型 |
|------|--------|------|
| MLP 前向 | $2bsh/t$ | All-Reduce |
| MLP 反向 | $2bsh/t$ | All-Reduce |
| Attention 前向 | $2bsh/t$ | All-Reduce |
| Attention 反向 | $2bsh/t$ | All-Reduce |

**每层总通信量**：$8bsh/t$

### 与数据并行对比

| 策略 | 通信量 | 通信时机 |
|------|--------|---------|
| 数据并行 | $2\Psi$（参数量） | 反向传播后 |
| 张量并行 | $8bsh/t$ | 每层 |

::: warning 关键区别
- **DP**：通信量与参数量成正比，通信频率低
- **TP**：通信量与激活大小成正比，通信频率高

TP 需要高带宽互联（如 NVLink），适合节点内！
:::

## 序列并行 (Sequence Parallelism)

Megatron-LM SP（Sequence Parallelism）将长序列切分到多个 GPU：

```python
class SequenceParallelAttention(nn.Module):
    """序列并行：将序列维度切分到多个 GPU"""
    
    def __init__(self, hidden_size, num_heads, world_size):
        super().__init__()
        # ... 初始化同上 ...
        self.world_size = world_size
    
    def forward(self, x):
        # x: [batch, seq/N, hidden_size]
        # 每个设备处理序列的一部分
        
        # Ring Attention 实现长序列处理
        # ...
```

## 与其他并行策略组合

### 3D 并行中的张量并行

```
TP 通常用于节点内（NVLink 高带宽）
PP 用于跨节点（通信量小）
DP 用于增加 batch size

示例：64 GPU 训练
- TP = 8（单节点内）
- PP = 2（跨节点流水线）
- DP = 4（数据并行）
```

### 通信组划分

```python
# 创建张量并行组
tp_group = dist.new_group(ranks=[0, 1, 2, 3])  # 节点内

# 创建流水线并行组
pp_group = dist.new_group(ranks=[0, 4], ranks=[1, 5], ...)

# 创建数据并行组
dp_group = dist.new_group(ranks=[0, 4, 8, 12], ...)
```

## 小结

| 特性 | 说明 |
|------|------|
| 分片方式 | 层内参数切分 |
| 适用层 | Linear、Attention |
| 通信模式 | All-Reduce |
| 通信频率 | 每层 |
| 硬件要求 | 高带宽互联（NVLink） |
| 扩展性 | 通常 ≤ 8（节点内） |

**核心优势**：
- 突破单卡显存限制
- 计算与通信可重叠
- 与数据并行、流水线并行正交

**适用场景**：
- 大模型单节点训练
- 与 PP、DP 组合成 3D 并行
