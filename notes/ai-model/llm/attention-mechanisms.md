# 注意力机制

注意力机制（Attention Mechanism）是现代深度学习中最核心的创新之一，它使模型能够动态地关注输入序列的不同部分。从 Seq2Seq 模型中的萌芽，到 Transformer 中的自注意力，再到各种高效变体的涌现，注意力机制已成为大语言模型的基石。

---

## 章节概述

本章将系统介绍注意力机制的发展脉络和核心变体：

| 机制 | 核心思想 | 代表模型 | 主要优势 |
|------|----------|----------|----------|
| 缩放点积注意力 | Query-Key-Value 计算相关性 | Transformer | 计算高效、可并行 |
| 多头注意力 (MHA) | 多个子空间并行学习 | BERT、GPT | 捕获多样特征 |
| 多查询注意力 (MQA) | 共享 KV 减少内存 | PaLM | 推理速度快 |
| 分组查询注意力 (GQA) | 分组共享 KV | LLaMA2 | 平衡质量与速度 |
| 线性注意力 | 核函数近似 Softmax | Performer | $O(N)$ 复杂度 |
| 稀疏注意力 | 局部+全局稀疏模式 | Longformer | 支持长序列 |
| FlashAttention | IO 感知的分块计算 | GPT-4 | 内存高效 |
| 滑动窗口注意力 | 局部窗口限制 | Mistral | 高效建模局部 |

---

## 一、注意力机制概述

### 1.1 Seq2Seq 中的注意力起源

在早期的序列到序列（Seq2Seq）模型中，编码器将整个输入序列压缩成一个固定长度的上下文向量，解码器基于此生成输出。这种做法存在**信息瓶颈**问题：长序列的信息难以被完整压缩。

**注意力机制的引入**（Bahdanau et al., 2015）解决了这个问题：

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

其中：
- $c_i$ 是第 $i$ 个解码步的上下文向量
- $h_j$ 是编码器第 $j$ 个隐藏状态
- $\alpha_{ij}$ 是注意力权重，表示解码位置 $i$ 对编码位置 $j$ 的关注程度

**注意力权重的计算**：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ij} = a(s_{i-1}, h_j)
$$

其中 $a$ 是一个可学习的对齐函数（如前馈网络）。

### 1.2 注意力机制的核心思想

注意力机制的本质是**信息检索**：给定一个查询（Query），从一组键值对（Key-Value）中检索相关信息。

```
Query: 我想要什么信息？
Key:   信息的位置/标签
Value: 信息的内容
```

模型通过计算 Query 与 Key 的相似度，得到注意力权重，再对 Value 进行加权求和，得到最终输出。

---

## 二、缩放点积注意力

### 2.1 基本原理

缩放点积注意力（Scaled Dot-Product Attention）是 Transformer 中的核心组件，由 Vaswani et al. (2017) 提出。

给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵
- $d_k$：查询和键的维度
- $\sqrt{d_k}$：缩放因子

### 2.2 为什么需要缩放因子？

当 $d_k$ 较大时，$QK^T$ 的元素方差也会变大，导致 Softmax 输出趋于 one-hot 分布，梯度变得很小。

假设 $q$ 和 $k$ 的元素独立同分布，均值为 0，方差为 1：

$$
\text{Var}(q \cdot k) = d_k
$$

除以 $\sqrt{d_k}$ 后，方差归一化为 1：

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

### 2.3 计算复杂度分析

| 操作 | 复杂度 | 空间复杂度 |
|------|--------|------------|
| $QK^T$ | $O(n \cdot m \cdot d_k)$ | $O(n \cdot m)$ |
| Softmax | $O(n \cdot m)$ | $O(n \cdot m)$ |
| $\text{weights} \times V$ | $O(n \cdot m \cdot d_v)$ | $O(n \cdot d_v)$ |
| **总计** | $O(n \cdot m \cdot d)$ | $O(n \cdot m)$ |

对于自注意力（$n = m$），时间复杂度为 $O(n^2 d)$，空间复杂度为 $O(n^2)$。

### 2.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, num_heads, seq_len_q, d_k]
            key:   [batch_size, num_heads, seq_len_k, d_k]
            value: [batch_size, num_heads, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask:  [batch_size, 1, 1, seq_len_k] 或 [batch_size, 1, seq_len_q, seq_len_k]
        
        Returns:
            output: [batch_size, num_heads, seq_len_q, d_v]
            attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)
        
        # 计算注意力分数: [batch, heads, seq_q, seq_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（用于遮挡 padding 或实现因果注意力）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: [batch, heads, seq_q, d_v]
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


# 使用示例
def demo_scaled_dot_product():
    batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
    
    attention = ScaledDotProductAttention(dropout=0.0)
    
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # 创建因果掩码（下三角矩阵）
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output, weights = attention(query, key, value, mask)
    
    print(f"Query shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be 1): {weights[0, 0, 0].sum():.6f}")

if __name__ == "__main__":
    demo_scaled_dot_product()
```

---

## 三、多头注意力（MHA）

### 3.1 核心思想

多头注意力让模型能够同时关注不同位置的不同表示子空间。每个"头"学习不同的注意力模式，最后将结果拼接融合。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

其中：
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$
- $h$ 是头的数量

### 3.2 为什么需要多头？

单个注意力头可能会将所有信息"平均化"，难以捕获多样化的模式。多头注意力类似于卷积网络中的多通道：

- **头 1** 可能学习语法依赖关系
- **头 2** 可能学习语义关联
- **头 3** 可能学习位置信息
- ...

### 3.3 参数量分析

设 $d_{model} = 512$，$h = 8$，则 $d_k = d_v = 64$：

| 参数 | 形状 | 参数量 |
|------|------|--------|
| $W^Q$ | $512 \times 512$ | 262,144 |
| $W^K$ | $512 \times 512$ | 262,144 |
| $W^V$ | $512 \times 512$ | 262,144 |
| $W^O$ | $512 \times 512$ | 262,144 |
| **总计** | - | **1,048,576** |

### 3.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key:   [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model] (seq_len_k == seq_len_v)
            mask:  [batch_size, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 1. 线性投影
        Q = self.w_q(query)  # [batch, seq_q, d_model]
        K = self.w_k(key)    # [batch, seq_k, d_model]
        V = self.w_v(value)  # [batch, seq_v, d_model]
        
        # 2. 分割成多头: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # 3. 处理掩码维度
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_k]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_q, seq_k]
        
        # 4. 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 5. 拼接多头: [batch, heads, seq_q, d_v] -> [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 6. 输出投影
        output = self.w_o(attn_output)
        
        return output, attn_weights


# 使用示例
def demo_multi_head_attention():
    batch_size, seq_len, d_model, num_heads = 2, 16, 512, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 自注意力
    output, weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")

if __name__ == "__main__":
    demo_multi_head_attention()
```

---

## 四、多查询注意力（MQA）

### 4.1 动机：推理时的内存瓶颈

标准多头注意力在推理时需要缓存所有头的 Key 和 Value。对于长序列和大模型，这成为严重的内存瓶颈。

以 GPT-3 (175B) 为例，假设：
- 层数：96
- 头数：96
- 序列长度：2048
- 每个头的维度：128

KV Cache 大小（FP16）：
$$
2 \times 96 \times 96 \times 2048 \times 128 \times 2 \text{ bytes} \approx 9.6 \text{ GB}
$$

### 4.2 MQA 原理

多查询注意力（Multi-Query Attention，Shazeer et al., 2019）的核心思想：**所有查询头共享同一组 Key 和 Value**。

$$
\text{MQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)
$$

注意区别：
- MHA：每个头有独立的 $W_i^K$ 和 $W_i^V$
- MQA：所有头共享 $W^K$ 和 $W^V$

### 4.3 内存与速度优化

| 指标 | MHA | MQA | 提升 |
|------|-----|-----|------|
| KV Cache 大小 | $O(h \cdot n \cdot d)$ | $O(n \cdot d)$ | **h 倍** |
| 内存带宽需求 | 高 | 低 | 显著降低 |
| 推理速度 | 基准 | 更快 | 约 10-30% |

### 4.4 PyTorch 实现

```python
class MultiQueryAttention(nn.Module):
    """多查询注意力：所有查询头共享一组 KV"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query 仍然是多头投影
        self.w_q = nn.Linear(d_model, d_model)
        
        # Key 和 Value 只有单头（所有头共享）
        self.w_k = nn.Linear(d_model, self.d_k)  # 注意：输出维度是 d_k，不是 d_model
        self.w_v = nn.Linear(d_model, self.d_k)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Query 投影并分头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Key 和 Value 单头投影，然后扩展到多头维度
        K = self.w_k(key).unsqueeze(1)  # [batch, 1, seq_k, d_k]
        V = self.w_v(value).unsqueeze(1)  # [batch, 1, seq_v, d_k]
        
        # 广播到所有头
        K = K.expand(-1, self.num_heads, -1, -1)  # [batch, heads, seq_k, d_k]
        V = V.expand(-1, self.num_heads, -1, -1)  # [batch, heads, seq_v, d_k]
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
        
        attn_output, _ = self.attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output)


def compare_mha_mqa():
    """对比 MHA 和 MQA 的参数量和 KV cache"""
    d_model, num_heads = 512, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    
    print(f"MHA parameters: {mha_params:,}")
    print(f"MQA parameters: {mqa_params:,}")
    print(f"MQA parameter ratio: {mqa_params / mha_params:.2%}")
    
    # KV Cache 对比（假设序列长度 1024）
    seq_len = 1024
    mha_kv_cache = 2 * num_heads * seq_len * (d_model // num_heads)  # K + V
    mqa_kv_cache = 2 * 1 * seq_len * (d_model // num_heads)
    
    print(f"\nKV Cache (seq_len={seq_len}):")
    print(f"MHA: {mha_kv_cache * 2 / 1024 / 1024:.2f} MB (FP16)")
    print(f"MQA: {mqa_kv_cache * 2 / 1024 / 1024:.2f} MB (FP16)")
    print(f"Memory saving: {(1 - mqa_kv_cache / mha_kv_cache):.1%}")

if __name__ == "__main__":
    compare_mha_mqa()
```

---

## 五、分组查询注意力（GQA）

### 5.1 动机：平衡质量与速度

MQA 虽然大幅提升了推理速度，但由于过度共享 KV，可能导致模型质量下降。分组查询注意力（Grouped Query Attention，Ainslie et al., 2023）提供了一个折中方案。

### 5.2 GQA 原理

GQA 将查询头分成若干组，每组共享一组 Key-Value：

$$
\text{GQA}(Q, K, V) = \text{Concat}(\text{group}_1, ..., \text{group}_g)W^O
$$

其中：
- $g$ 是组数（$1 \leq g \leq h$）
- 当 $g = h$ 时，GQA 退化为 MHA
- 当 $g = 1$ 时，GQA 退化为 MQA

### 5.3 LLaMA 2 的使用

LLaMA 2 在 34B 和 70B 模型中使用 GQA（$g = 8$）：

| 模型 | 参数量 | 注意力类型 | 组数 |
|------|--------|------------|------|
| LLaMA 2 7B | 7B | MHA | - |
| LLaMA 2 13B | 13B | MHA | - |
| LLaMA 2 34B | 34B | GQA | 8 |
| LLaMA 2 70B | 70B | GQA | 8 |

实验表明，GQA 在保持接近 MHA 质量的同时，获得接近 MQA 的推理速度。

### 5.4 PyTorch 实现

```python
class GroupedQueryAttention(nn.Module):
    """分组查询注意力：查询头分组，每组共享一组 KV"""
    
    def __init__(self, d_model: int, num_heads: int, num_groups: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads
        
        # Query 投影（多头）
        self.w_q = nn.Linear(d_model, d_model)
        
        # Key 和 Value 投影（分组）
        self.w_k = nn.Linear(d_model, num_groups * self.d_k)
        self.w_v = nn.Linear(d_model, num_groups * self.d_k)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Query: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Key/Value: [batch, seq, num_groups * d_k] -> [batch, groups, seq, d_k]
        K = self.w_k(key).view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        
        # 将 KV 扩展到各组内的所有头
        # [batch, groups, seq, d_k] -> [batch, groups, heads_per_group, seq, d_k]
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        
        # 重排为 [batch, heads, seq, d_k]
        K = K.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
        
        attn_output, _ = self.attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output)


def compare_attention_mechanisms():
    """对比不同注意力机制"""
    d_model, num_heads = 512, 8
    seq_len = 1024
    batch_size = 2
    
    configs = [
        ("MHA", {"num_groups": num_heads}),  # GQA 退化为 MHA
        ("GQA-4", {"num_groups": 4}),
        ("GQA-2", {"num_groups": 2}),
        ("MQA", {"num_groups": 1}),  # GQA 退化为 MQA
    ]
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"{'Type':<10} {'Params':>12} {'KV Cache (MB)':>15}")
    print("-" * 40)
    
    for name, config in configs:
        attn = GroupedQueryAttention(d_model, num_heads, **config)
        params = sum(p.numel() for p in attn.parameters())
        kv_cache = 2 * config["num_groups"] * seq_len * (d_model // num_heads) * 2 / 1024 / 1024
        
        print(f"{name:<10} {params:>12,} {kv_cache:>15.2f}")

if __name__ == "__main__":
    compare_attention_mechanisms()
```

---

## 六、线性注意力

### 6.1 动机：二次复杂度问题

标准注意力的复杂度为 $O(n^2 d)$，对于长序列（如文档、代码）计算代价高昂。线性注意力通过核函数近似，将复杂度降至 $O(nd^2)$。

### 6.2 核心原理

回顾标准注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

展开为：

$$
o_i = \frac{\sum_{j=1}^{n} \text{sim}(q_i, k_j) v_j}{\sum_{j=1}^{n} \text{sim}(q_i, k_j)}
$$

其中 $\text{sim}(q, k) = \exp(q \cdot k / \sqrt{d_k})$。

**线性注意力的关键洞察**：使用核函数 $\phi$ 近似 softmax：

$$
\text{sim}(q, k) \approx \phi(q) \cdot \phi(k)
$$

则有：

$$
o_i = \frac{\phi(q_i)^T \sum_{j=1}^{n} \phi(k_j) v_j^T}{\phi(q_i)^T \sum_{j=1}^{n} \phi(k_j)}
$$

先计算 $\sum_{j=1}^{n} \phi(k_j) v_j^T$，复杂度从 $O(n^2)$ 降为 $O(nd^2)$！

### 6.3 Performer

Performer（Choromanski et al., 2021）使用随机特征映射：

$$
\phi(x) = \frac{1}{\sqrt{m}}[\exp(\omega_1^T x), ..., \exp(\omega_m^T x)]
$$

其中 $\omega_i$ 从适当分布中随机采样，保证：

$$
\mathbb{E}[\phi(x)^T \phi(y)] \approx \exp(x^T y)
$$

### 6.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class LinearAttention(nn.Module):
    """线性注意力：使用核函数近似 softmax"""
    
    def __init__(self, d_model: int, num_heads: int, feature_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_dim = feature_dim or self.d_k
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 随机特征矩阵
        self.register_buffer(
            'proj_matrix', 
            torch.randn(self.d_k, self.feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def feature_map(self, x):
        """随机特征映射：φ(x) = elu(x) + 1"""
        # 简单的elu+1特征映射
        return torch.nn.functional.elu(x) + 1
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影并分头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用特征映射
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # 线性注意力核心计算
        # 先计算 K^T V: [batch, heads, d_k, d_k]
        Kt_V = torch.matmul(K.transpose(-2, -1), V)
        
        # 计算 Q (K^T V): [batch, heads, seq_q, d_k]
        numerator = torch.matmul(Q, Kt_V)
        
        # 计算分母: sum(K)
        K_sum = K.sum(dim=-2, keepdim=True)  # [batch, heads, 1, d_k]
        denominator = torch.matmul(Q, K_sum.transpose(-2, -1))
        
        # 归一化
        output = numerator / (denominator + 1e-6)
        
        # 重排并输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output


def compare_complexity():
    """比较标准注意力和线性注意力的复杂度"""
    import time
    
    d_model, num_heads = 256, 8
    batch_size = 4
    
    mha = MultiHeadAttention(d_model, num_heads)
    linear_attn = LinearAttention(d_model, num_heads)
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"{'Seq Len':<10} {'MHA (ms)':>12} {'Linear (ms)':>12} {'Speedup':>10}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model)
        
        # MHA
        start = time.time()
        with torch.no_grad():
            _ = mha(x, x, x)
        mha_time = (time.time() - start) * 1000
        
        # Linear Attention
        start = time.time()
        with torch.no_grad():
            _ = linear_attn(x, x, x)
        linear_time = (time.time() - start) * 1000
        
        print(f"{seq_len:<10} {mha_time:>12.2f} {linear_time:>12.2f} {mha_time/linear_time:>10.2f}x")

if __name__ == "__main__":
    compare_complexity()
```

---

## 七、稀疏注意力

### 7.1 动机：全局注意力的冗余

在实际文本中，大多数 token 只与少数其他 token 有强关联。稀疏注意力通过限制每个 token 只关注部分位置，将复杂度降至近似线性。

### 7.2 稀疏注意力模式

常见的稀疏模式包括：

| 模式 | 描述 | 复杂度 |
|------|------|--------|
| 局部窗口 | 每个位置只关注附近 $w$ 个位置 | $O(nw)$ |
| 扩张窗口 | 跳跃式关注，类似扩张卷积 | $O(n \log n)$ |
| 全局 token | 少数 token 可关注全局 | $O(n \cdot g)$ |
| 随机注意力 | 随机选择关注位置 | $O(nr)$ |

### 7.3 Longformer

Longformer（Beltagy et al., 2020）结合局部和全局注意力：

$$
A_{ij} = \begin{cases}
1 & \text{if } |i - j| \leq w \text{ (局部)} \\
1 & \text{if } i \in G \text{ or } j \in G \text{ (全局)} \\
0 & \text{otherwise}
\end{cases}
$$

其中 $G$ 是全局 token 集合（如 [CLS] token）。

### 7.4 BigBird

BigBird（Zaheer et al., 2020) 在 Longformer 基础上增加随机注意力：

1. **局部窗口注意力**：滑动窗口大小 $w$
2. **全局注意力**：$g$ 个全局 token
3. **随机注意力**：每个位置随机关注 $r$ 个位置

理论证明：BigBird 能近似完整的自注意力，同时保持 $O(n)$ 复杂度。

### 7.5 PyTorch 实现（局部窗口注意力）

```python
class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力：每个位置只关注局部窗口"""
    
    def __init__(self, d_model: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_window_mask(self, seq_len, device):
        """创建滑动窗口掩码"""
        # 每个位置只关注 [i-w, i+w] 范围内的位置
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        return mask
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 创建窗口掩码
        window_mask = self.create_window_mask(seq_len, query.device)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        # 合并外部掩码（如 padding mask）
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            window_mask = window_mask * mask
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)


def demo_sparse_attention():
    """演示稀疏注意力的掩码模式"""
    import matplotlib.pyplot as plt
    
    seq_len = 64
    window_size = 8
    
    # 创建滑动窗口掩码
    attn = SlidingWindowAttention(256, 8, window_size)
    mask = attn.create_window_mask(seq_len, 'cpu')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='Blues')
    plt.title(f'Sliding Window Attention Mask (window_size={window_size})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Attention Allowed')
    plt.savefig('sparse_attention_mask.png', dpi=150, bbox_inches='tight')
    print("Mask visualization saved to 'sparse_attention_mask.png'")

if __name__ == "__main__":
    demo_sparse_attention()
```

---

## 八、FlashAttention

### 8.1 动机：内存访问瓶颈

标准注意力的主要瓶颈不是计算，而是**内存访问**。在 GPU 上：

- GPU 内存带宽：~1.5 TB/s (A100)
- GPU 计算能力：~312 TFLOPS (A100, FP16)

计算 $QK^T$ 需要将 $O(n^2)$ 的中间结果写入 HBM（高带宽内存），再读出进行 Softmax，这是巨大的 IO 开销。

### 8.2 FlashAttention 核心思想

FlashAttention（Dao et al., 2022）通过**分块计算**和**重计算**优化内存访问：

#### 分块计算（Tiling）

将 Q、K、V 分成小块，在 GPU 的 SRAM（片上内存）中完成注意力的完整计算：

```
for each block Q_i:
    for each block K_j, V_j:
        # 在 SRAM 中计算
        S_ij = Q_i @ K_j^T
        P_ij = softmax(S_ij)
        O_i += P_ij @ V_j
```

#### 在线 Softmax

使用增量计算避免两次遍历：

$$
m_{new} = \max(m_{old}, m_{current})
$$

$$
P_{new} = P_{old} \cdot e^{m_{old} - m_{new}} + P_{current} \cdot e^{m_{current} - m_{new}}
$$

### 8.3 内存优化效果

| 方法 | HBM 访问 | 内存占用 |
|------|----------|----------|
| 标准注意力 | $O(n^2)$ | $O(n^2)$ |
| FlashAttention | $O(n)$ | $O(n)$ |

### 8.4 PyTorch 使用示例

```python
# FlashAttention 已集成到 PyTorch 2.0+
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashMultiHeadAttention(nn.Module):
    """使用 PyTorch 内置 FlashAttention 的多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
    
    def forward(self, query, key, value, mask=None, is_causal=False):
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 使用 PyTorch 2.0+ 的 scaled_dot_product_attention
        # 自动选择 FlashAttention 或 Memory-Efficient Attention
        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.d_k)
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(output)


def benchmark_flash_attention():
    """对比标准注意力和 FlashAttention"""
    import time
    
    d_model, num_heads = 512, 8
    batch_size = 4
    
    standard_mha = MultiHeadAttention(d_model, num_heads)
    flash_mha = FlashMultiHeadAttention(d_model, num_heads)
    
    seq_lengths = [512, 1024, 2048, 4096]
    
    print(f"{'Seq Len':<10} {'Standard (ms)':>15} {'Flash (ms)':>15} {'Memory Saved':>15}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 标准注意力
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            _ = standard_mha(x, x, x)
        standard_time = (time.time() - start) * 1000
        
        # FlashAttention
        start = time.time()
        with torch.no_grad():
            _ = flash_mha(x, x, x)
        flash_time = (time.time() - start) * 1000
        
        # 内存节省（理论值）
        standard_memory = seq_len * seq_len * 4  # FP32
        flash_memory = seq_len * d_model * 4
        
        print(f"{seq_len:<10} {standard_time:>15.2f} {flash_time:>15.2f} {(1 - flash_memory/standard_memory):>15.1%}")

if __name__ == "__main__":
    benchmark_flash_attention()
```

---

## 九、滑动窗口注意力

### 9.1 原理

滑动窗口注意力限制每个位置只能关注固定大小的局部窗口：

$$
A_{ij} = \begin{cases}
\text{Attention}(q_i, k_j) & \text{if } |i - j| \leq w \\
0 & \text{otherwise}
\end{cases}
$$

### 9.2 Mistral 的使用

Mistral 7B 采用滑动窗口注意力（窗口大小 $w = 4096$），配合滚动缓冲区（Rolling Buffer Cache）：

| 特性 | 描述 |
|------|------|
| 窗口大小 | 4096 tokens |
| KV Cache | 固定大小，滚动更新 |
| 复杂度 | $O(n \cdot w)$ |

**信息传递**：虽然每个位置只能直接看到窗口内的内容，但通过多层堆叠，信息可以间接传递到更远的位置。第 $l$ 层的感受野大小为 $l \times w$。

### 9.3 与其他机制的结合

Mistral 还结合了 GQA（分组查询注意力），实现高效推理：

```python
class MistralAttention(nn.Module):
    """Mistral 风格的注意力：滑动窗口 + GQA"""
    
    def __init__(self, d_model: int, num_heads: int, num_groups: int, 
                 window_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_groups * self.d_k)
        self.w_v = nn.Linear(d_model, num_groups * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cache=None, cache_position=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        
        # 扩展 KV 到所有头
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        K = K.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        
        # 滑动窗口因果掩码
        # 位置 i 可以关注 [i-window_size, i] 范围
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # 添加窗口限制
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            causal_mask[i, :start] = float('-inf')
        
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 使用 FlashAttention（如果可用）
        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=causal_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(output)


def demo_mistral_attention():
    """演示 Mistral 注意力的掩码模式"""
    import matplotlib.pyplot as plt
    
    seq_len = 64
    window_size = 16
    
    # 创建滑动窗口因果掩码
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='Blues')
    plt.title(f'Mistral Sliding Window Causal Mask (window={window_size})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Attention Allowed')
    plt.savefig('mistral_attention_mask.png', dpi=150, bbox_inches='tight')
    print("Mask visualization saved to 'mistral_attention_mask.png'")

if __name__ == "__main__":
    demo_mistral_attention()
```

---

## 十、注意力机制对比与选择

### 10.1 综合对比表

| 机制 | 时间复杂度 | 空间复杂度 | KV Cache | 模型质量 | 推理速度 | 适用场景 |
|------|------------|------------|----------|----------|----------|----------|
| MHA | $O(n^2d)$ | $O(n^2)$ | 大 | ★★★★★ | 慢 | 训练、短序列 |
| MQA | $O(n^2d)$ | $O(n^2)$ | **小** | ★★★☆☆ | **快** | 推理优化 |
| GQA | $O(n^2d)$ | $O(n^2)$ | 中 | ★★★★☆ | 较快 | 平衡质量与速度 |
| 线性注意力 | $O(nd^2)$ | $O(nd)$ | - | ★★★☆☆ | 中 | 超长序列 |
| 稀疏注意力 | $O(nwd)$ | $O(nw)$ | - | ★★★★☆ | 较快 | 长文档 |
| FlashAttention | $O(n^2d)$ | $O(n)$ | 大 | ★★★★★ | 快 | 通用优化 |
| 滑动窗口 | $O(nwd)$ | $O(nw)$ | 小 | ★★★★☆ | 快 | 局部依赖 |

### 10.2 选择建议

```
是否需要处理超长序列（>16K tokens）？
├── 是 → 考虑线性注意力或稀疏注意力
└── 否 → 
    ├── 训练阶段 → FlashAttention + MHA
    └── 推理阶段 →
        ├── 追求极致速度 → MQA
        ├── 平衡质量与速度 → GQA（推荐）
        └── 需要局部建模 → 滑动窗口注意力
```

### 10.3 实践建议

1. **训练阶段**：
   - 使用 FlashAttention（PyTorch 2.0+ 自动启用）
   - 配合混合精度训练

2. **推理阶段**：
   - 优先选择支持 GQA 的模型（如 LLaMA 2）
   - 使用 KV Cache 优化
   - 考虑 PagedAttention（vLLM）进一步优化

3. **长序列场景**：
   - 文档处理：Longformer / BigBird
   - 代码生成：滑动窗口注意力
   - 流式处理：滑动窗口 + Rolling Cache

---

## 知识点关联

```
注意力机制发展脉络

Seq2Seq Attention (2015)
       │
       ▼
Transformer Self-Attention (2017)
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
   效率优化        结构优化        长序列优化
       │              │              │
       ▼              ▼              ▼
 FlashAttention    MQA/GQA      稀疏注意力
   (2022)         (2019/2023)   Longformer
                                  (2020)
                      │              │
                      ▼              ▼
                 滑动窗口        线性注意力
                  Mistral       Performer
                  (2023)         (2020)
```

---

## 核心考点

### 📌 必须掌握

1. **缩放点积注意力的计算过程**
   - 能手写公式并解释每一步
   - 理解缩放因子的作用

2. **多头注意力的原理**
   - 为什么需要多头
   - 参数量计算

3. **MHA / MQA / GQA 的区别**
   - KV Cache 大小的变化
   - 对模型质量的影响

### 💡 重点理解

1. **注意力复杂度分析**
   - 时间复杂度：$O(n^2d)$
   - 空间复杂度：$O(n^2)$
   - 如何优化到线性

2. **FlashAttention 的优化原理**
   - 分块计算
   - IO 感知
   - 内存节省

3. **稀疏注意力的稀疏模式**
   - 局部窗口
   - 全局 token
   - 随机注意力

### ⚠️ 常见误区

1. **MQA 不改变计算复杂度**
   - MQA 主要优化内存和带宽，不是计算量
   - 推理速度提升来自减少内存访问

2. **线性注意力有精度损失**
   - 核函数近似引入误差
   - 对于某些任务可能表现下降

3. **滑动窗口注意力的信息传递**
   - 单层只能看到局部
   - 多层堆叠可扩大感受野

---

## 学习建议

### 📚 推荐阅读顺序

1. **入门**：理解 Seq2Seq 注意力 → Transformer 自注意力
2. **进阶**：MHA 实现 → MQA/GQA 原理
3. **高级**：FlashAttention 论文 → 稀疏注意力变体
4. **实践**：PyTorch 实现 → 模型推理优化

### 🔗 经典论文

1. **Attention Is All You Need** (Vaswani et al., 2017) - Transformer 原文
2. **Fast Transformer Decoding** (Shazeer, 2019) - MQA
3. **GQA: Training Generalized Multi-Query Transformer** (Ainslie et al., 2023)
4. **FlashAttention** (Dao et al., 2022)
5. **Longformer** (Beltagy et al., 2020)
6. **Performer** (Choromanski et al., 2021)

### 💻 实践建议

1. 从零实现缩放点积注意力和多头注意力
2. 对比 MHA 和 MQA 的 KV Cache 大小
3. 使用 PyTorch Profiler 分析注意力计算的时间分布
4. 尝试将 FlashAttention 应用到自己的模型中
