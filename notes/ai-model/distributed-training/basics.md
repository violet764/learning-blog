# 分布式训练基本概念

在深入学习各种并行策略之前，我们需要先理解分布式训练的基础概念，包括硬件拓扑、进程通信、集合通信操作等。

## 硬件拓扑

### 节点 (Node)

**节点**是指一台物理服务器，通常包含：
- 多个 GPU
- 多个 CPU 核心
- 系统内存 (RAM)
- 网络接口卡 (NIC)
- NVLink/NVSwitch（多 GPU 互联）

```
┌─────────────────────────────────────────────────────────────────┐
│                          Node (节点)                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │       │
│  │  A100    │  │  A100    │  │  A100    │  │  A100    │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴──────┬──────┴─────────────┘              │
│                            │                                   │
│                     NVLink/NVSwitch                           │
│                            │                                   │
│  ┌─────────────────────────┴─────────────────────────┐        │
│  │                    CPU + RAM                       │        │
│  └─────────────────────────┬─────────────────────────┘        │
│                            │                                   │
│                    Network Interface                          │
│                      (NIC / IB 网卡)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 进程与设备

在分布式训练中：

- **进程 (Process)**：一个独立的 Python 程序实例
- **Rank**：进程的全局唯一标识符
- **Local Rank**：进程在节点内的序号
- **World Size**：总进程数

```python
# 假设有 2 个节点，每个节点 4 个 GPU
# 总共 8 个进程，World Size = 8

# Rank 分配示例：
# Node 0: Rank 0, 1, 2, 3 (Local Rank: 0, 1, 2, 3)
# Node 1: Rank 4, 5, 6, 7 (Local Rank: 0, 1, 2, 3)
```

## 进程组 (Process Group)

**进程组**是一组协同工作的进程集合，用于限定通信范围。

### 默认进程组

PyTorch 初始化后会创建一个包含所有进程的默认组：

```python
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(
    backend='nccl',      # 通信后端
    init_method='env://', # 初始化方法
    world_size=8,        # 总进程数
    rank=rank            # 当前进程的 rank
)

# 默认组包含所有进程
default_group = dist.GroupMember.WORLD
```

### 自定义进程组

可以创建子组进行局部通信：

```python
import torch.distributed as dist

# 创建进程组：[0, 1, 2, 3] 和 [4, 5, 6, 7]
group1 = dist.new_group(ranks=[0, 1, 2, 3])
group2 = dist.new_group(ranks=[4, 5, 6, 7])

# 在特定组内通信
dist.all_reduce(tensor, group=group1)
```

### 通信后端 (Backend)

| 后端 | 适用场景 | 通信方式 |
|------|---------|---------|
| NCCL | GPU 间通信 | NVIDIA GPU 专用，性能最优 |
| Gloo | CPU/GPU 通信 | 通用后端，支持 CPU 集合通信 |
| MPI | 高性能计算 | 标准 MPI 接口 |

```python
# 选择后端
if torch.cuda.is_available():
    backend = 'nccl'  # GPU 训练首选
else:
    backend = 'gloo'  # CPU 训练
```

## 集合通信操作

集合通信（Collective Communication）是分布式训练的核心，所有进程共同参与的通信操作。

### 1. Broadcast (广播)

将一个进程的数据发送给所有进程。

```
Broadcast:
Rank 0: [A] ─────┬───▶ [A]
Rank 1: [?]      │    ▶ [A]
Rank 2: [?]      │    ▶ [A]
Rank 3: [?]      ┘    ▶ [A]
```

```python
import torch
import torch.distributed as dist

# 将 rank 0 的数据广播给所有进程
tensor = torch.zeros(4)
if dist.get_rank() == 0:
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

dist.broadcast(tensor, src=0)  # src=0 表示从 rank 0 广播
print(f"Rank {dist.get_rank()}: {tensor}")
# 所有进程输出: tensor([1., 2., 3., 4.])
```

### 2. Scatter (散射)

将数据分发给不同进程。

```
Scatter:
Rank 0: [A,B,C,D] ─┬─▶ [A]
Rank 1: [?]        ├──▶ [B]
Rank 2: [?]        ├──▶ [C]
Rank 3: [?]        └──▶ [D]
```

```python
# Scatter 操作
scatter_list = None
if dist.get_rank() == 0:
    # 将数据分成 4 份
    scatter_list = [torch.tensor([i]) for i in range(4)]

output = torch.zeros(1)
dist.scatter(output, scatter_list, src=0)
print(f"Rank {dist.get_rank()}: {output}")
# Rank 0: [0], Rank 1: [1], Rank 2: [2], Rank 3: [3]
```

### 3. Gather (收集)

收集所有进程的数据到一个进程。

```
Gather:
Rank 0: [A] ─────┐
Rank 1: [B] ─────┼───▶ [A,B,C,D] (Rank 0)
Rank 2: [C] ─────┤
Rank 3: [D] ─────┘
```

```python
# Gather 操作
input_tensor = torch.tensor([dist.get_rank()])
gather_list = None

if dist.get_rank() == 0:
    gather_list = [torch.zeros(1) for _ in range(4)]

dist.gather(input_tensor, gather_list, dst=0)

if dist.get_rank() == 0:
    print(f"Collected: {[t.item() for t in gather_list]}")
    # 输出: Collected: [0, 1, 2, 3]
```

### 4. All-Gather (全收集)

所有进程都获得所有数据。

```
All-Gather:
Rank 0: [A] ─────┐
Rank 1: [B] ─────┼───▶ [A,B,C,D] (所有进程)
Rank 2: [C] ─────┤
Rank 3: [D] ─────┘
```

```python
# All-Gather 操作
input_tensor = torch.tensor([dist.get_rank()])
output_list = [torch.zeros(1) for _ in range(4)]

dist.all_gather(output_list, input_tensor)

print(f"Rank {dist.get_rank()}: {[t.item() for t in output_list]}")
# 所有进程输出: [0, 1, 2, 3]
```

### 5. Reduce (归约)

对数据进行规约操作（求和、平均等）。

```
Reduce (SUM):
Rank 0: [A] ─────┐
Rank 1: [B] ─────┼───▶ [A+B+C+D] (Rank 0)
Rank 2: [C] ─────┤
Rank 3: [D] ─────┘
```

```python
# Reduce 操作 (求和)
input_tensor = torch.tensor([dist.get_rank() + 1], dtype=torch.float)
output = torch.zeros(1)

dist.reduce(input_tensor, dst=0, op=dist.ReduceOp.SUM)

if dist.get_rank() == 0:
    print(f"Reduced sum: {input_tensor.item()}")  # 1+2+3+4 = 10
```

### 6. All-Reduce (全归约)

所有进程都获得归约结果，是最常用的操作。

```
All-Reduce (SUM):
Rank 0: [A] ─────┐
Rank 1: [B] ─────┼───▶ [A+B+C+D] (所有进程)
Rank 2: [C] ─────┤
Rank 3: [D] ─────┘
```

```python
# All-Reduce 操作 (求和)
tensor = torch.tensor([dist.get_rank() + 1], dtype=torch.float)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"Rank {dist.get_rank()}: {tensor.item()}")  # 所有进程输出: 10
```

::: tip All-Reduce 是数据并行的核心
在数据并行中，每个 GPU 计算各自的梯度后，通过 All-Reduce 同步梯度平均值。
:::

### 7. Reduce-Scatter

先归约，再散射，常用于张量并行。

```
Reduce-Scatter (SUM):
Rank 0: [A1,A2,A3,A4] ──┐
Rank 1: [B1,B2,B3,B4] ──┼───▶ Reduce ──▶ Scatter
Rank 2: [C1,C2,C3,C4] ──┤
Rank 3: [D1,D2,D3,D4] ──┘

结果:
Rank 0: [A1+B1+C1+D1]
Rank 1: [A2+B2+C2+D2]
Rank 2: [A3+B3+C3+D3]
Rank 3: [A4+B4+C4+D4]
```

### 8. All-to-All

每个进程向所有其他进程发送不同的数据。

```
All-to-All:
        输入                      输出
Rank 0: [A0,A1,A2,A3] ────▶ [A0,B0,C0,D0]
Rank 1: [B0,B1,B2,B3] ────▶ [A1,B1,C1,D1]
Rank 2: [C0,C1,C2,C3] ────▶ [A2,B2,C2,D2]
Rank 3: [D0,D1,D2,D3] ────▶ [A3,B3,C3,D3]
```

## 点对点通信

除了集合通信，还有进程间的一对一通信：

```python
# Send / Recv
if dist.get_rank() == 0:
    tensor = torch.tensor([1.0, 2.0, 3.0])
    dist.send(tensor, dst=1)  # 发送给 rank 1
elif dist.get_rank() == 1:
    tensor = torch.zeros(3)
    dist.recv(tensor, src=0)  # 从 rank 0 接收
    print(f"Received: {tensor}")
```

## NCCL 通信环

NCCL 使用 **Ring All-Reduce** 实现高效的梯度同步。

### Ring All-Reduce 原理

假设有 4 个 GPU，数据分为 4 块：

```
Step 1: Scatter-Reduce (分块归约)
GPU0: [A0]→[A0+B0]→[A0+B0+C0]→[A0+B0+C0+D0]
GPU1: [B1]→[B1+C1]→[B1+C1+D1]→[A1+B1+C1+D1]
GPU2: [C2]→[C2+D2]→[A2+B2+C2+D2]→[A2+B2+C2+D2]
GPU3: [D3]→[A3+B3]→[A3+B3+C3]→[A3+B3+C3+D3]

Step 2: All-Gather (广播结果)
所有 GPU 最终都获得完整的归约结果
```

### 时间复杂度

设数据量为 $M$，进程数为 $N$：

- **朴素 All-Reduce**: $O(N \times M)$（单节点瓶颈）
- **Ring All-Reduce**: $O(2 \times \frac{N-1}{N} \times M)$（负载均衡）

当 $N$ 较大时，Ring All-Reduce 接近 $O(2M)$，与进程数无关！

## 通信带宽与延迟

不同互联方式的性能差异：

| 互联方式 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|---------|
| PCIe 4.0 x16 | 32 GB/s | ~1 μs | GPU-CPU 通信 |
| NVLink 3.0 | 50 GB/s/link | ~0.5 μs | GPU 间高速通信 |
| NVSwitch | 800 GB/s | ~0.5 μs | 多 GPU 全互联 |
| InfiniBand HDR | 200 Gb/s | ~1 μs | 节点间通信 |
| Ethernet 100GbE | 100 Gb/s | ~2-10 μs | 一般集群 |

::: warning 通信是分布式训练的瓶颈
- 张量并行：需要高带宽 GPU 互联（NVLink）
- 流水线并行：通信量小，适合跨节点
- 数据并行：All-Reduce 量大，受网络带宽限制
:::

## 分布式启动方式

### torchrun

PyTorch 推荐的分布式启动工具：

```bash
# 单节点 4 GPU
torchrun --nproc_per_node=4 train.py

# 多节点（节点 0 作为主节点）
# 节点 0:
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 --master_addr="10.0.0.1" --master_port=29500 \
    train.py

# 节点 1:
torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=1 --master_addr="10.0.0.1" --master_port=29500 \
    train.py
```

### 分布式初始化模板

```python
import os
import torch
import torch.distributed as dist

def setup_distributed():
    # 从环境变量获取分布式信息
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

# 使用示例
def main():
    rank, world_size, local_rank = setup_distributed()
    
    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}")
    
    # ... 训练代码 ...
    
    cleanup()

if __name__ == "__main__":
    main()
```

## 小结

| 概念 | 说明 |
|------|------|
| Node（节点） | 物理服务器，包含多个 GPU |
| Rank | 进程的全局标识符 |
| Local Rank | 进程在节点内的序号 |
| World Size | 总进程数 |
| Process Group | 协同通信的进程集合 |
| All-Reduce | 最常用的集合通信，同步梯度 |
| Ring All-Reduce | NCCL 高效实现方式 |

理解这些基本概念是学习分布式训练的基础，下一章我们将深入讲解数据并行策略。
