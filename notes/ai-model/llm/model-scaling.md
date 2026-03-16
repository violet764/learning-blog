# 模型规模估算

大语言模型的规模估算是模型设计与训练规划的核心环节。准确估算参数量、显存需求、计算量和训练时间，对于硬件选型、成本控制和项目规划至关重要。本文将从数学原理出发，系统性地介绍各项估算方法。

## 基本概念

### 核心指标概述

| 指标 | 符号 | 含义 | 典型单位 |
|------|------|------|----------|
| 参数量 | $N$ | 模型中可训练参数总数 | Billion (B) |
| 显存占用 | $M$ | 训练/推理时GPU内存需求 | GB |
| 计算量 | $C$ | 模型前向/反向传播的浮点运算数 | FLOPs |
| 吞吐量 | $\Phi$ | 每秒处理的token数 | tokens/s |

### 精度类型对比

| 精度 | 字节数 | 数值范围 | 适用场景 |
|------|--------|----------|----------|
| FP32 | 4 bytes | $\pm 3.4 \times 10^{38}$ | 优化器状态 |
| FP16 | 2 bytes | $\pm 6.5 \times 10^{4}$ | 训练主精度 |
| BF16 | 2 bytes | $\pm 3.4 \times 10^{38}$ | 大模型训练 |
| INT8 | 1 byte | $[-128, 127]$ | 量化推理 |

---

## 参数量计算

### Transformer整体架构

标准的Transformer解码器模型（如GPT系列）由以下组件构成：

```
Embedding层 → L × Transformer块 → LayerNorm → 输出层
```

其中每个Transformer块包含：
- 多头自注意力层 (Multi-Head Self-Attention)
- 前馈神经网络层 (Feed-Forward Network)
- 层归一化 (Layer Normalization)

### 符号定义

| 符号 | 含义 |
|------|------|
| $V$ | 词表大小 |
| $d_{model}$ | 隐藏维度 |
| $h$ | 注意力头数 |
| $d_{head}$ | 每个头的维度 ($d_{model}/h$) |
| $L$ | Transformer层数 |
| $d_{ff}$ | FFN中间层维度 |
| $l_{seq}$ | 序列长度 |

### Embedding层参数量

Embedding层包括词嵌入和位置嵌入：

$$
P_{embed} = V \times d_{model} + l_{max} \times d_{model}
$$

其中 $l_{max}$ 是最大序列长度。对于现代LLM：
- **绝对位置编码**：需要位置嵌入参数
- **RoPE/ALiBi**：无参数位置编码，$P_{embed} = V \times d_{model}$

**示例计算**（GPT-2）：
- $V = 50257$, $d_{model} = 768$, $l_{max} = 1024$
- $P_{embed} = 50257 \times 768 + 1024 \times 768 \approx 39.4M$

### Attention层参数量

#### 标准Attention

每个注意力层包含四个投影矩阵：

$$
\begin{aligned}
Q &= XW_Q, \quad W_Q \in \mathbb{R}^{d_{model} \times d_{model}} \\
K &= XW_K, \quad W_K \in \mathbb{R}^{d_{model} \times d_{model}} \\
V &= XW_V, \quad W_V \in \mathbb{R}^{d_{model} \times d_{model}} \\
O &= \text{Attn}(Q,K,V)W_O, \quad W_O \in \mathbb{R}^{d_{model} \times d_{model}}
\end{aligned}
$$

**参数量公式**：

$$
P_{attn} = 4 \times d_{model}^2 + 4 \times d_{model}
$$

其中 $4 \times d_{model}$ 是偏置项参数。

#### Multi-Query Attention (MQA)

MQA将所有查询头共享同一个Key和Value头：

$$
P_{attn}^{MQA} = d_{model}^2 + 2 \times d_{model} \times d_{head} + d_{model}^2 + 4 \times d_{model}
$$

简化后约为标准Attention的 $\frac{1}{h}$ 参数量用于KV投影。

#### Grouped-Query Attention (GQA)

GQA是MQA和MHA的折中方案，将查询头分成 $g$ 组，每组共享KV：

$$
P_{attn}^{GQA} \approx d_{model}^2 + 2g \times d_{model} \times d_{head} + d_{model}^2
$$

### FFN层参数量

标准FFN包含两个线性变换和一个激活函数：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

**参数量公式**：

$$
P_{ffn} = 2 \times d_{model} \times d_{ff} + d_{model} + d_{ff}
$$

通常 $d_{ff} = 4d_{model}$（标准Transformer）或 $d_{ff} = \frac{8}{3}d_{model}$（LLaMA风格）。

**LLaMA风格FFN**（SwiGLU激活）：

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_{gate}) \odot (xW_{up})W_{down}
$$

参数量：

$$
P_{ffn}^{SwiGLU} = 3 \times d_{model} \times d_{ff}
$$

### LayerNorm参数量

每个LayerNorm有两个可学习参数（缩放和偏移）：

$$
P_{ln} = 2 \times d_{model}
$$

每层有2个LayerNorm（Attention前和FFN前），加上最终输出层的LayerNorm。

### 总参数量公式

#### 标准Transformer（GPT风格）

$$
\begin{aligned}
P_{total} &= P_{embed} + L \times P_{layer} + P_{output} \\
P_{layer} &= P_{attn} + P_{ffn} + 2 \times P_{ln} \\
&= 4d^2 + 4d + 8d^2 + 2d + 4d \\
&= 12d^2 + 10d
\end{aligned}
$$

简化后：

$$
P_{total} \approx 12Ld^2 + Vd
$$

#### LLaMA风格模型

LLaMA使用RMSNorm（无偏置）、SwiGLU FFN、RoPE：

$$
P_{total} \approx L \times \left( 4d^2 + 3 \times d \times d_{ff} \right) + Vd
$$

当 $d_{ff} = \frac{8}{3}d$ 时：

$$
P_{total} \approx L \times 12d^2 + Vd
$$

### 不同模型参数量计算示例

| 模型 | $d_{model}$ | $L$ | $h$ | $d_{ff}$ | $V$ | 估算参数量 |
|------|-------------|-----|-----|----------|-----|-----------|
| GPT-2 Small | 768 | 12 | 12 | 3072 | 50257 | 124M |
| GPT-2 Medium | 1024 | 24 | 16 | 4096 | 50257 | 355M |
| LLaMA-7B | 4096 | 32 | 32 | 11008 | 32000 | 6.7B |
| LLaMA-13B | 5120 | 40 | 40 | 13824 | 32000 | 13.0B |
| LLaMA-70B | 8192 | 80 | 64 | 28672 | 32000 | 70.0B |

---

## 显存估算

### 显存组成部分

训练时GPU显存主要消耗在以下部分：

```
总显存 = 模型参数 + 梯度 + 优化器状态 + 激活值 + 临时缓存
```


### 模型参数显存占用

参数显存取决于存储精度：

$$
M_{params} = N \times \text{bytes\_per\_param}
$$

| 精度 | 每参数字节数 | 7B模型显存 |
|------|-------------|-----------|
| FP32 | 4 | 28 GB |
| FP16/BF16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| INT4 | 0.5 | 3.5 GB |

### 梯度显存占用

反向传播需要存储每个参数的梯度：

$$
M_{grads} = N \times \text{bytes\_per\_grad}
$$

通常梯度使用FP16/BF16存储（2 bytes），部分框架使用FP32主梯度。

### 优化器状态显存占用

#### Adam/AdamW优化器

Adam需要存储一阶动量 $m$ 和二阶动量 $v$：

$$
M_{optimizer}^{Adam} = 2 \times N \times 4 = 8N \text{ bytes}
$$

- $m$ (一阶动量): FP32, 4N bytes
- $v$ (二阶动量): FP32, 4N bytes

#### 混合精度训练

混合精度训练通常维护一份FP32参数副本：

$$
M_{optimizer}^{mixed} = N \times 4 + 8N = 12N \text{ bytes}
$$

包括：
- FP32参数副本: 4N bytes
- FP32梯度副本: 4N bytes（可选）
- 一阶动量: 4N bytes
- 二阶动量: 4N bytes

### 激活值显存占用

激活值（中间计算结果）在反向传播时需要保存。其显存占用与批量大小、序列长度相关。

#### 理论估算

每个Transformer层的激活值：

$$
M_{act}^{layer} \approx b \times l_{seq} \times d_{model} \times (34 + 5 \times \frac{d_{ff}}{d_{model}} + 4 \times \frac{l_{seq}}{d_{model}}) \times \text{bytes}
$$

其中 $b$ 是批量大小。

简化估算（忽略注意力缓存）：

$$
M_{act}^{layer} \approx b \times l_{seq} \times d_{model} \times 10 \times \text{bytes}
$$

总激活值：

$$
M_{act}^{total} = L \times M_{act}^{layer}
$$

#### 激活检查点（Activation Checkpointing）

通过重计算减少激活值存储：

$$
M_{act}^{checkpoint} \approx \sqrt{M_{act}^{total}}
$$

代价是增加约33%的计算量。

### 完整显存估算公式

#### FP32全精度训练

$$
M_{total}^{FP32} = 4N + 4N + 8N + M_{act} = 16N + M_{act}
$$

#### 混合精度训练（FP16/BF16）

$$
M_{total}^{mixed} = 2N + 2N + 12N + M_{act} = 16N + M_{act}
$$

#### 推理模式

$$
M_{inference} = N \times \text{bytes\_per\_param} + M_{kv\_cache}
$$

### KV Cache显存估算

推理时的KV缓存显存：

$$
M_{kv\_cache} = 2 \times L \times b \times l_{seq} \times d_{model} \times \text{bytes}
$$

其中因子2表示Key和Value两份缓存。

**示例**（LLaMA-7B, $b=1$, $l_{seq}=2048$, FP16）：

$$
M_{kv\_cache} = 2 \times 32 \times 1 \times 2048 \times 4096 \times 2 = 1 \text{ GB}
$$

### 不同模型显存需求参考

| 模型 | 参数量 | FP32训练 | 混合精度训练 | FP16推理 |
|------|--------|----------|-------------|----------|
| GPT-2 Small | 124M | ~4 GB | ~3 GB | ~0.5 GB |
| GPT-2 Medium | 355M | ~9 GB | ~7 GB | ~1 GB |
| LLaMA-7B | 7B | ~112 GB | ~28 GB | ~14 GB |
| LLaMA-13B | 13B | ~208 GB | ~52 GB | ~26 GB |
| LLaMA-70B | 70B | ~1.1 TB | ~280 GB | ~140 GB |

::: tip 显存优化策略
1. **混合精度训练**：使用FP16/BF16减少一半显存
2. **梯度累积**：减小批量大小，累积梯度
3. **激活检查点**：时间换空间
4. **ZeRO优化**：分片存储优化器状态
5. **模型并行**：将模型分布到多卡
:::

---

## FLOPs计算

### 矩阵乘法FLOPs

矩阵乘法 $(m \times k) \times (k \times n)$ 的FLOPs：

$$
\text{FLOPs} = 2mnk = 2 \times \text{output\_size} \times k
$$

每次乘加运算算作2个FLOP。

### 单层Transformer FLOPs

#### Attention层FLOPs

前向传播：

$$
\begin{aligned}
\text{QKV投影} &: 3 \times 2 \times b \times l \times d \times d = 6bld^2 \\
\text{QK}^T &: 2 \times b \times h \times l \times l \times d_{head} = 2bldl \\
\text{Softmax} &: \approx 3bhl^2 \text{ (可忽略)} \\
\text{Attention} \times V &: 2 \times b \times h \times l \times l \times d_{head} = 2bldl \\
\text{输出投影} &: 2 \times b \times l \times d \times d = 2bld^2
\end{aligned}
$$

忽略softmax和序列长度相关的项：

$$
\text{FLOPs}_{attn}^{fwd} \approx 8bld^2 + 4bl^2d
$$


#### FFN层FLOPs

对于 $d_{ff} = 4d$ 的标准FFN：

$$
\text{FLOPs}_{ffn}^{fwd} = 2 \times 2 \times b \times l \times d \times d_{ff} = 16bld^2
$$

#### 单层总FLOPs

$$
\text{FLOPs}_{layer}^{fwd} = 24bld^2 + 4bl^2d
$$

### 整体模型FLOPs

#### 前向传播

$$
\text{FLOPs}_{forward} = L \times (24bld^2 + 4bl^2d) + 2bld \times V
$$

简化后（忽略词表项）：

$$
\text{FLOPs}_{forward} \approx 24Lbld^2 \times \left(1 + \frac{l}{6d}\right)
$$

当 $l \ll d$ 时，可近似为：

$$
\text{FLOPs}_{forward} \approx 24Lbld^2 = 24Nbld
$$

因为 $N \approx 12Ld^2$。

#### 训练FLOPs（前向+反向）

反向传播的计算量约为前向传播的2倍：

$$
\text{FLOPs}_{training} \approx 3 \times \text{FLOPs}_{forward} = 72Nbld
$$

### 每Token计算量

**训练时每个Token的FLOPs**：

$$
\text{FLOPs/token}^{train} \approx 6N
$$

**推理时每个Token的FLOPs**：

$$
\text{FLOPs/token}^{inference} \approx 2N
$$

这个规律非常实用：训练一个Token约需要 $6N$ 次浮点运算。

### FLOPs计算示例

**LLaMA-7B训练1T tokens**：

$$
\text{Total FLOPs} = 6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22}
$$

使用A100（312 TFLOPS for BF16）需要：

$$
\text{GPU-hours} = \frac{4.2 \times 10^{22}}{312 \times 10^{12} \times 3600} \approx 37,300 \text{ GPU-hours}
$$

---

## 训练时间估算

### 理论计算时间

基于算力的理论训练时间：

$$
T_{theory} = \frac{\text{Total FLOPs}}{\text{GPU\_TFLOPS} \times 10^{12} \times \text{num\_gpus} \times \text{efficiency}}
$$

其中：
- **GPU\_TFLOPS**: GPU理论峰值算力
- **efficiency**: 实际效率（通常0.3-0.5）

### 常见GPU算力参考

| GPU | FP16/BF16 理论峰值 | 实际训练效率 | 实际算力 |
|-----|-------------------|-------------|----------|
| A100 80GB | 312 TFLOPS | 35-45% | ~120 TFLOPS |
| A100 40GB | 312 TFLOPS | 35-45% | ~120 TFLOPS |
| H100 | 989 TFLOPS | 40-50% | ~450 TFLOPS |
| V100 | 125 TFLOPS | 30-40% | ~45 TFLOPS |
| RTX 4090 | 330 TFLOPS | 25-35% | ~100 TFLOPS |

### 实际训练时间估算

考虑实际效率的训练时间：

$$
T_{actual} = \frac{6N \times D}{\Phi_{actual} \times \text{num\_gpus}}
$$

其中：
- $N$: 参数量
- $D$: 训练数据量（tokens）
- $\Phi_{actual}$: 每GPU实际吞吐量（tokens/s）

### 训练时间估算示例

**LLaMA-7B训练估算**：
- 参数量: 7B
- 数据量: 1T tokens
- 硬件: 256 × A100 80GB
- 实际吞吐量: ~4000 tokens/s/GPU

$$
T = \frac{6 \times 7 \times 10^9 \times 10^{12}}{4000 \times 256 \times 3600} \approx 11.4 \text{ 天}
$$

**LLaMA-70B训练估算**：
- 参数量: 70B
- 数据量: 1.4T tokens
- 硬件: 2048 × A100 80GB
- 实际吞吐量: ~2000 tokens/s/GPU（模型并行开销）

$$
T = \frac{6 \times 70 \times 10^9 \times 1.4 \times 10^{12}}{2000 \times 2048 \times 3600} \approx 39 \text{ 天}
$$

### Chinchilla最优训练

[Chinchilla论文](https://arxiv.org/abs/2203.15556)提出最优训练配比：

$$
D_{opt} = 20 \times N
$$

即训练数据量应为参数量的20倍，实现最佳的计算资源利用。

---

## 代码示例

### 参数量统计脚本

```python
import torch
import torch.nn as nn

def count_parameters(model: nn.Module, detailed: bool = False) -> dict:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        detailed: 是否返回详细统计
    
    Returns:
        参数统计字典
    """
    total_params = 0
    trainable_params = 0
    param_dict = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        if detailed:
            # 按模块分类统计
            module_name = name.split('.')[0]
            if module_name not in param_dict:
                param_dict[module_name] = 0
            param_dict[module_name] += num_params
    
    result = {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'total_M': total_params / 1e6,
        'total_B': total_params / 1e9,
    }
    
    if detailed:
        result['by_module'] = param_dict
    
    return result


def estimate_transformer_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int = None,
    tie_weights: bool = False
) -> dict:
    """
    估算Transformer模型参数量
    
    Args:
        vocab_size: 词表大小
        d_model: 隐藏维度
        num_layers: 层数
        num_heads: 注意力头数
        d_ff: FFN中间层维度（默认4*d_model）
        tie_weights: 是否共享embedding和输出层权重
    
    Returns:
        参数量估算字典
    """
    if d_ff is None:
        d_ff = 4 * d_model
    
    # Embedding层
    embed_params = vocab_size * d_model
    
    # 位置编码（假设使用可学习位置编码）
    # pos_params = max_seq_len * d_model  # 如果使用RoPE则为0
    
    # 每层参数
    # Attention: 4 * d^2 (Q,K,V,O投影) + 4 * d (偏置)
    attn_params = 4 * d_model * d_model + 4 * d_model
    
    # FFN: 2 * d * d_ff + d_ff + d (两层线性+偏置)
    ffn_params = 2 * d_model * d_ff + d_model + d_ff
    
    # LayerNorm: 2 * d * 2 (每层2个LN)
    ln_params = 4 * d_model
    
    # 单层总参数
    layer_params = attn_params + ffn_params + ln_params
    
    # 所有层
    total_layer_params = num_layers * layer_params
    
    # 输出层
    output_params = 0 if tie_weights else vocab_size * d_model
    
    # 总参数
    total_params = embed_params + total_layer_params + output_params
    
    return {
        'embedding': embed_params,
        'attention_per_layer': attn_params,
        'ffn_per_layer': ffn_params,
        'layer_norm_per_layer': ln_params,
        'all_layers': total_layer_params,
        'output': output_params,
        'total': total_params,
        'total_B': total_params / 1e9,
        'approximation': 12 * num_layers * d_model * d_model + vocab_size * d_model
    }


# 使用示例
if __name__ == "__main__":
    # 估算LLaMA-7B参数量
    params = estimate_transformer_params(
        vocab_size=32000,
        d_model=4096,
        num_layers=32,
        num_heads=32,
        d_ff=11008,
        tie_weights=False
    )
    
    print("LLaMA-7B 参数量估算:")
    print(f"  Embedding: {params['embedding']/1e6:.2f}M")
    print(f"  All Layers: {params['all_layers']/1e9:.2f}B")
    print(f"  Total: {params['total_B']:.2f}B")
    print(f"  简化公式估算: {params['approximation']/1e9:.2f}B")
```


### 显存估算工具

```python
def estimate_memory(
    num_params: float,  # 参数量（单位：B）
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    precision: str = 'bf16',
    mode: str = 'training',
    gradient_checkpointing: bool = False,
    optimizer: str = 'adamw'
) -> dict:
    """
    估算模型训练/推理显存需求
    
    Args:
        num_params: 参数量（单位：B）
        batch_size: 批量大小
        seq_len: 序列长度
        d_model: 隐藏维度
        num_layers: 层数
        precision: 精度类型 (fp32/fp16/bf16/int8)
        mode: 模式 (training/inference)
        gradient_checkpointing: 是否使用梯度检查点
        optimizer: 优化器类型 (adamw/adafactor)
    
    Returns:
        显存估算字典（单位：GB）
    """
    # 精度对应字节数
    bytes_map = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1}
    bp = bytes_map[precision]  # 每参数字节数
    
    N = num_params * 1e9  # 转换为实际参数数
    
    # 模型参数显存
    params_mem = N * bp / 1e9  # GB
    
    # 梯度显存（训练时）
    grads_mem = N * bp / 1e9 if mode == 'training' else 0
    
    # 优化器状态显存
    if mode == 'training':
        if optimizer == 'adamw':
            # AdamW: 2个FP32动量 + 1个FP32参数副本
            optimizer_mem = N * 12 / 1e9  # GB (FP32)
        elif optimizer == 'adafactor':
            # Adafactor: 更少的优化器状态
            optimizer_mem = N * 4 / 1e9
        else:
            optimizer_mem = N * 8 / 1e9
    else:
        optimizer_mem = 0
    
    # 激活值显存
    if mode == 'training':
        # 估算每层激活值（简化公式）
        activation_per_layer = batch_size * seq_len * d_model * 10 * 2  # bytes
        total_activation = num_layers * activation_per_layer / 1e9  # GB
        
        if gradient_checkpointing:
            # 激活检查点减少显存
            total_activation = (total_activation ** 0.5) * 1.5  # 近似
    else:
        total_activation = 0
    
    # KV Cache显存（推理时）
    if mode == 'inference':
        kv_cache = 2 * num_layers * batch_size * seq_len * d_model * bp / 1e9
    else:
        kv_cache = 0
    
    # 总显存
    total_mem = params_mem + grads_mem + optimizer_mem + total_activation + kv_cache
    
    # 添加10%的CUDA开销
    total_mem *= 1.1
    
    return {
        'params_gb': round(params_mem, 2),
        'grads_gb': round(grads_mem, 2),
        'optimizer_gb': round(optimizer_mem, 2),
        'activations_gb': round(total_activation, 2),
        'kv_cache_gb': round(kv_cache, 2),
        'total_gb': round(total_mem, 2),
        'recommended_gpu': get_gpu_recommendation(total_mem)
    }


def get_gpu_recommendation(memory_gb: float) -> str:
    """根据显存需求推荐GPU"""
    if memory_gb <= 24:
        return "RTX 3090/4090 (24GB)"
    elif memory_gb <= 40:
        return "A100 40GB"
    elif memory_gb <= 80:
        return "A100 80GB"
    elif memory_gb <= 160:
        return "2×A100 80GB (模型并行)"
    else:
        num_a100 = int(memory_gb / 70) + 1
        return f"{num_a100}×A100 80GB (模型并行)"


# 使用示例
if __name__ == "__main__":
    # 估算LLaMA-7B训练显存
    mem = estimate_memory(
        num_params=7,
        batch_size=1,
        seq_len=2048,
        d_model=4096,
        num_layers=32,
        precision='bf16',
        mode='training',
        gradient_checkpointing=True
    )
    
    print("LLaMA-7B 显存估算 (训练):")
    for k, v in mem.items():
        print(f"  {k}: {v}")
```

### FLOPs计算工具

```python
def estimate_flops(
    num_params: float,  # 参数量（单位：B）
    num_tokens: float,  # Token数量（单位：B）
    mode: str = 'training'
) -> dict:
    """
    估算模型计算量
    
    Args:
        num_params: 参数量（单位：B）
        num_tokens: Token数量（单位：B）
        mode: training 或 inference
    
    Returns:
        FLOPs估算字典
    """
    N = num_params * 1e9
    D = num_tokens * 1e9
    
    # 每Token FLOPs
    flops_per_token = 6 if mode == 'training' else 2
    
    # 总FLOPs
    total_flops = flops_per_token * N * D
    
    # 转换单位
    peta_flops = total_flops / 1e15
    exa_flops = total_flops / 1e18
    
    return {
        'total_flops': total_flops,
        'peta_flops': round(peta_flops, 2),
        'exa_flops': round(exa_flops, 4),
        'flops_per_token': flops_per_token
    }


def estimate_training_time(
    num_params: float,
    num_tokens: float,
    num_gpus: int,
    gpu_type: str = 'A100',
    gpu_efficiency: float = 0.4
) -> dict:
    """
    估算训练时间
    
    Args:
        num_params: 参数量（单位：B）
        num_tokens: Token数量（单位：B）
        num_gpus: GPU数量
        gpu_type: GPU类型
        gpu_efficiency: 实际效率（0-1）
    
    Returns:
        训练时间估算
    """
    # GPU理论算力 (TFLOPS for BF16)
    gpu_tflops = {
        'A100': 312,
        'H100': 989,
        'V100': 125,
        'RTX4090': 330
    }
    
    tflops = gpu_tflops.get(gpu_type, 312)
    actual_tflops = tflops * gpu_efficiency
    
    # 计算总FLOPs
    N = num_params * 1e9
    D = num_tokens * 1e9
    total_flops = 6 * N * D
    
    # 计算时间
    seconds = total_flops / (actual_tflops * 1e12 * num_gpus)
    hours = seconds / 3600
    days = hours / 24
    
    return {
        'total_flops': total_flops,
        'actual_tflops_per_gpu': round(actual_tflops, 1),
        'total_tflops': round(actual_tflops * num_gpus, 1),
        'hours': round(hours, 1),
        'days': round(days, 1),
        'gpu_hours': round(hours * num_gpus, 1)
    }


# 使用示例
if __name__ == "__main__":
    # 估算LLaMA-7B训练
    print("=== LLaMA-7B 训练估算 ===")
    
    flops = estimate_flops(num_params=7, num_tokens=1000, mode='training')
    print(f"计算量: {flops['exa_flops']} EFLOPs")
    
    time = estimate_training_time(
        num_params=7,
        num_tokens=1000,
        num_gpus=256,
        gpu_type='A100',
        gpu_efficiency=0.4
    )
    print(f"训练时间: {time['days']} 天 ({time['hours']} 小时)")
    print(f"GPU-hours: {time['gpu_hours']}")
```

### 综合估算工具

```python
class ModelScalingEstimator:
    """模型规模综合估算器"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int = None,
        max_seq_len: int = 2048
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_len = max_seq_len
        
        # 计算参数量
        self.num_params = self._compute_params()
    
    def _compute_params(self) -> float:
        """计算参数量（单位：B）"""
        embed = self.vocab_size * self.d_model
        attn = 4 * self.d_model ** 2
        ffn = 2 * self.d_model * self.d_ff
        ln = 4 * self.d_model
        layer = attn + ffn + ln
        
        total = embed + self.num_layers * layer + self.vocab_size * self.d_model
        return total / 1e9
    
    def estimate_training(
        self,
        batch_size: int,
        seq_len: int,
        num_tokens_billion: float,
        num_gpus: int = 1,
        gpu_type: str = 'A100',
        precision: str = 'bf16',
        gradient_checkpointing: bool = False
    ) -> dict:
        """综合训练估算"""
        
        # 显存估算
        memory = estimate_memory(
            num_params=self.num_params,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self.d_model,
            num_layers=self.num_layers,
            precision=precision,
            mode='training',
            gradient_checkpointing=gradient_checkpointing
        )
        
        # 计算量估算
        flops = estimate_flops(
            num_params=self.num_params,
            num_tokens=num_tokens_billion,
            mode='training'
        )
        
        # 时间估算
        time = estimate_training_time(
            num_params=self.num_params,
            num_tokens=num_tokens_billion,
            num_gpus=num_gpus,
            gpu_type=gpu_type
        )
        
        return {
            'model_params_B': round(self.num_params, 2),
            'memory': memory,
            'flops': flops,
            'time': time
        }
    
    def __repr__(self):
        return (
            f"ModelScalingEstimator(\n"
            f"  d_model={self.d_model}, layers={self.num_layers}, "
            f"heads={self.num_heads}\n"
            f"  d_ff={self.d_ff}, vocab={self.vocab_size}\n"
            f"  params={self.num_params:.2f}B\n"
            f")"


# 使用示例
if __name__ == "__main__":
    # 创建LLaMA-7B估算器
    estimator = ModelScalingEstimator(
        vocab_size=32000,
        d_model=4096,
        num_layers=32,
        num_heads=32,
        d_ff=11008
    )
    
    print(estimator)
    
    # 估算训练
    result = estimator.estimate_training(
        batch_size=1,
        seq_len=2048,
        num_tokens_billion=1000,
        num_gpus=256,
        gpu_type='A100',
        gradient_checkpointing=True
    )
    
    print("\n训练估算结果:")
    print(f"  参数量: {result['model_params_B']}B")
    print(f"  显存需求: {result['memory']['total_gb']} GB")
    print(f"  推荐GPU: {result['memory']['recommended_gpu']}")
    print(f"  计算量: {result['flops']['exa_flops']} EFLOPs")
    print(f"  训练时间: {result['time']['days']} 天")
```

---

## 常见问题

### Q1: 为什么实际训练显存比估算值大？

**原因**：
1. **CUDA内存碎片**：内存分配产生的碎片
2. **PyTorch缓存**：PyTorch预分配的缓存空间
3. **临时张量**：计算过程中产生的中间结果
4. **框架开销**：框架自身的内存占用

**解决方案**：在估算值基础上增加15-20%的安全余量。

### Q2: 混合精度训练真的能省一半显存吗？

**不完全正确**。混合精度训练主要节省：
- 模型参数：减半
- 梯度：减半
- 激活值：减半

但优化器状态（Adam动量）通常仍用FP32，这部分不变。总体节省约30-40%。

### Q3: 如何快速估算需要的GPU数量？

**经验公式**：

$$
\text{num\_gpus} = \lceil \frac{\text{训练总显存需求}}{\text{单卡显存} \times 0.7} \rceil
$$

其中0.7是安全系数，预留空间给框架开销。

### Q4: 参数量和模型能力的关系？

根据[Chinchilla定律](https://arxiv.org/abs/2203.15556)：
- 计算最优：$D \approx 20N$（数据量约为参数量20倍）
- 给定算力预算下，参数量和数据量应同步增长
- 过度参数化或数据不足都会导致计算浪费

---

## 参考资料

1. [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
2. [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
3. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
