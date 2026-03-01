# LLM 模型架构基础

大语言模型（LLM）的架构设计是决定模型性能、训练效率和推理速度的关键因素。从 Transformer 的提出到 GPT、LLaMA 等主流模型的演进，架构设计经历了多次重要革新。本章将系统介绍 LLM 的核心架构组件和主流模型的设计特点。

---

## 章节概述

本章涵盖 LLM 架构设计的核心知识点：

| 主题 | 核心内容 | 重要程度 |
|------|----------|----------|
| 架构概述 | Decoder-only 主流架构、架构选择原因 | ⭐⭐⭐⭐⭐ |
| 解码器层结构 | 完整层组成、各组件功能 | ⭐⭐⭐⭐⭐ |
| 层归一化位置 | Pre-Norm vs Post-Norm、梯度流影响 | ⭐⭐⭐⭐ |
| 前馈网络 FFN | 标准FFN、GLU变体、SwiGLU/GeGLU | ⭐⭐⭐⭐ |
| 残差连接 | 梯度流、跳跃连接设计 | ⭐⭐⭐ |
| 激活函数 | GELU、Swish 等选择依据 | ⭐⭐⭐ |
| 主流架构对比 | GPT、LLaMA、PaLM 架构特点 | ⭐⭐⭐⭐⭐ |
| 模型规模设计 | 层数、维度、头数的配比 | ⭐⭐⭐⭐ |

---

## 一、LLM 架构概述

### 1.1 三种主流 Transformer 架构

基于 Transformer 的模型架构主要分为三类：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Transformer 架构分类                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │  Encoder-only   │   │  Decoder-only   │   │ Encoder-Decoder │       │
│  │     (BERT)      │   │      (GPT)      │   │       (T5)      │       │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤       │
│  │  双向注意力      │   │  单向注意力      │   │  编码+解码       │       │
│  │  理解任务        │   │  生成任务        │   │  序列转换        │       │
│  │  NER、分类       │   │  文本生成        │   │  翻译、摘要      │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么 LLM 选择 Decoder-only？

主流大语言模型（GPT 系列、LLaMA、PaLM 等）均采用 Decoder-only 架构，原因如下：

**📌 因果注意力机制**

Decoder 使用因果掩码（Causal Mask），确保每个位置只能看到之前的 token：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中掩码矩阵 $M$：

$$
M_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

**💡 核心优势**

| 优势 | 说明 |
|------|------|
| **自回归生成** | 天然支持逐 token 生成，适合文本生成任务 |
| **参数效率** | 相比 Encoder-Decoder，相同参数下更深的网络 |
| **训练简洁** | 单向注意力计算更简单，无需编码器-解码器交叉注意力 |
| **涌现能力** | 大规模 Decoder-only 模型展现更强的涌现能力 |
| **KV Cache 友好** | 推理时只需缓存历史 KV，增量生成效率高 |

**⚠️ 双向注意力的代价**

Encoder 架构的双向注意力在生成任务中存在"信息泄露"问题：生成第 $i$ 个 token 时，如果模型能看到后续 token，会导致训练-推理不一致。

### 1.3 Decoder-only 架构总览

一个典型的 Decoder-only 模型由多层解码器堆叠而成：

```
输入 Token IDs
     ↓
┌─────────────────────────────────────────────┐
│  Token Embedding + Positional Embedding     │
└─────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────┐
│           Decoder Layer × N                  │
│  ┌───────────────────────────────────────┐  │
│  │  Layer Norm (Pre-Norm)                │  │
│  │  Masked Self-Attention                │  │
│  │  Residual Connection                  │  │
│  │  Layer Norm                           │  │
│  │  Feed-Forward Network (FFN)           │  │
│  │  Residual Connection                  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────┐
│        Final Layer Norm                     │
└─────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────┐
│        Output Linear (lm_head)              │
└─────────────────────────────────────────────┘
     ↓
    Logits
```

---

## 二、Transformer 解码器层结构

### 2.1 单层详细结构

每个 Decoder 层包含两个核心子层：

1. **掩码自注意力层（Masked Self-Attention）**
2. **前馈网络层（Feed-Forward Network）**

每层包含的组件：

```
输入 x
   │
   ├──→ LayerNorm → Self-Attention ──→ (+) ──→ x1 (残差连接)
   │                                    ↑
   └────────────────────────────────────┘
                                       │
   ┌───────────────────────────────────┘
   ↓
   x1
   │
   ├──→ LayerNorm → FFN ──────────→ (+) ──→ 输出 (残差连接)
   │                                ↑
   └────────────────────────────────┘
```

### 2.2 数据流与维度变化

假设输入序列长度为 $n$，隐藏维度为 $d$：

| 组件 | 输入维度 | 输出维度 | 参数量 |
|------|----------|----------|--------|
| Layer Norm | $(n, d)$ | $(n, d)$ | $2d$ |
| Self-Attention | $(n, d)$ | $(n, d)$ | $4d^2$ |
| FFN | $(n, d)$ | $(n, d)$ | $8d^2$ (扩展因子4) |
| **单层总计** | - | - | $\approx 12d^2$ |

### 2.3 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class DecoderLayer(nn.Module):
    """Transformer 解码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, norm_eps: float = 1e-5):
        super().__init__()
        
        # 自注意力组件
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化（Pre-Norm 风格）
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, is_causal=True):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attn_mask: 可选的注意力掩码
            is_causal: 是否使用因果掩码
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        # Pre-Norm + 自注意力 + 残差
        residual = x
        x = self.norm1(x)
        
        # 生成因果掩码
        if is_causal and attn_mask is None:
            seq_len = x.size(1)
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
        
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout(x)
        
        # Pre-Norm + FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer 模型"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_seq_len: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（可学习）
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 输出层（语言模型头）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（可选）
        # self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 词嵌入 + 位置嵌入
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, is_causal=True)
        
        # 最终归一化和输出
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits


# 使用示例
def demo_decoder_transformer():
    # 模型配置
    config = {
        'vocab_size': 32000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    
    model = DecoderOnlyTransformer(**config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 前向传播测试
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")

if __name__ == "__main__":
    demo_decoder_transformer()
```

---

## 三、层归一化位置（Pre-Norm vs Post-Norm）

### 3.1 两种归一化位置

层归一化（Layer Normalization）的位置对模型训练稳定性有重要影响：

**Post-Norm（原始 Transformer）**

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))
$$

```
x ─────────────────────→ (+) ─→ LayerNorm ─→ x'
│                        ↑
└→ SubLayer ─────────────┘
```

**Pre-Norm（GPT-2、LLaMA 等）**

$$
x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))
$$

```
x ─────────────────────→ (+) ─→ x'
│                        ↑
└→ LayerNorm → SubLayer ─┘
```

### 3.2 数学分析与梯度流

**Post-Norm 的梯度问题**

对于 $L$ 层的 Post-Norm 网络，梯度传递：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=1}^{L} \frac{\partial x_l}{\partial x_{l-1}}
$$

层归一化会引入复杂的梯度变换，深层网络中梯度容易消失或爆炸。

**Pre-Norm 的梯度优势**

Pre-Norm 中，存在一条"干净"的残差路径：

$$
x_L = x_0 + \sum_{l=1}^{L} f_l(\text{LN}(x_{l-1}))
$$

梯度可以直接通过恒等映射传递：

$$
\frac{\partial x_L}{\partial x_0} = I + \sum_{l=1}^{L} \frac{\partial f_l}{\partial x_0}
$$

**📌 核心结论**：Pre-Norm 使梯度可以直接通过残差连接"跳跃"传递，有效缓解深层网络的梯度消失问题。

### 3.3 对比总结

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| **原始论文** | Transformer (2017) | GPT-2 (2019) |
| **梯度流** | 经过 LayerNorm 调制 | 存在干净残差路径 |
| **训练稳定性** | 需要学习率预热 | 更稳定，无需预热 |
| **最终层输出** | 已归一化 | 需要 Final LayerNorm |
| **模型性能** | 略好（相同层数） | 相近 |
| **深层网络** | 训练困难 | 训练稳定 |

### 3.4 PyTorch 实现对比

```python
class PostNormDecoderLayer(nn.Module):
    """Post-Norm 风格的解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Self-Attention + Residual + Post-Norm
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + Residual + Post-Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class PreNormDecoderLayer(nn.Module):
    """Pre-Norm 风格的解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Pre-Norm + Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)
        
        # Pre-Norm + FFN + Residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x
```

### 3.5 LLaMA 的 RMSNorm

LLaMA 使用 RMSNorm（Root Mean Square Layer Normalization）替代 LayerNorm，计算更高效：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return (x / rms) * self.weight


def compare_norm_methods():
    """对比 LayerNorm 和 RMSNorm"""
    d_model = 512
    batch_size, seq_len = 2, 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    layer_norm = nn.LayerNorm(d_model)
    rms_norm = RMSNorm(d_model)
    
    # 参数量对比
    ln_params = sum(p.numel() for p in layer_norm.parameters())
    rms_params = sum(p.numel() for p in rms_norm.parameters())
    
    print(f"LayerNorm 参数量: {ln_params}")  # 2 * d_model (weight + bias)
    print(f"RMSNorm 参数量: {rms_params}")   # d_model (weight only)
    
    # 计算速度对比
    import time
    
    # LayerNorm
    start = time.time()
    for _ in range(1000):
        _ = layer_norm(x)
    ln_time = time.time() - start
    
    # RMSNorm
    start = time.time()
    for _ in range(1000):
        _ = rms_norm(x)
    rms_time = time.time() - start
    
    print(f"\nLayerNorm 时间: {ln_time:.4f}s")
    print(f"RMSNorm 时间: {rms_time:.4f}s")
    print(f"加速比: {ln_time / rms_time:.2f}x")

if __name__ == "__main__":
    compare_norm_methods()
```

---

## 四、前馈网络（FFN）设计

### 4.1 标准 FFN

原始 Transformer 中的 FFN 结构：

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

其中：
- $W_1 \in \mathbb{R}^{d \times 4d}$（扩展因子通常为 4）
- $W_2 \in \mathbb{R}^{4d \times d}$
- 激活函数通常为 GELU 或 ReLU

### 4.2 GLU 变体

门控线性单元（Gated Linear Unit, GLU）引入门控机制：

$$
\text{GLU}(x) = (xW_1) \otimes \sigma(xW_2)
$$

其中 $\otimes$ 是逐元素乘法，$\sigma$ 是 Sigmoid 函数。

**💡 GLU 的优势**：门控机制允许模型选择性地传递信息，增强表达能力。

### 4.3 SwiGLU（LLaMA 采用）

SwiGLU 结合了 Swish 激活和 GLU 结构：

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)
$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$。

**参数效率考虑**：由于有三个投影矩阵，通常将中间维度从 $4d$ 减小到 $\frac{8}{3}d \approx 2.67d$。

### 4.4 GeGLU

GeGLU 使用 GELU 作为门控激活：

$$
\text{GeGLU}(x) = \text{GELU}(xW_1) \otimes (xW_2)
$$

### 4.5 FFN 变体对比

| FFN 类型 | 公式 | 参数量 | 使用模型 |
|----------|------|--------|----------|
| 标准 FFN | $\text{GELU}(xW_1)W_2$ | $8d^2$ | BERT、GPT-2 |
| GLU | $(xW_1) \otimes \sigma(xW_2)W_3$ | $12d^2$ | - |
| SwiGLU | $\text{Swish}(xW_1) \otimes (xW_2)W_3$ | $\approx 8d^2$ | LLaMA、PaLM |
| GeGLU | $\text{GELU}(xW_1) \otimes (xW_2)W_3$ | $\approx 8d^2$ | T5 |

### 4.6 PyTorch 实现

```python
class StandardFFN(nn.Module):
    """标准 FFN"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(torch.nn.functional.gelu(self.w1(x))))


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN（LLaMA 风格）"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # w1 和 w3 用于门控结构
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        SwiGLU(x) = Swish(xW1) ⊗ (xW3) W2
        """
        # Swish(x @ w1) * (x @ w3)
        hidden = self._swish(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(hidden))
    
    @staticmethod
    def _swish(x):
        """Swish 激活函数: x * sigmoid(x)"""
        return x * torch.sigmoid(x)


class GeGLUFFN(nn.Module):
    """GeGLU FFN"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        GeGLU(x) = GELU(xW1) ⊗ (xW3) W2
        """
        hidden = torch.nn.functional.gelu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(hidden))


def compare_ffn_variants():
    """对比不同 FFN 变体"""
    d_model = 512
    d_ff = 2048  # 标准 FFN 使用 4d
    
    # SwiGLU 通常使用约 2.67d
    d_ff_swiglu = int(8 * d_model / 3)
    
    variants = {
        'Standard FFN': StandardFFN(d_model, d_ff),
        f'SwiGLU FFN (d_ff={d_ff_swiglu})': SwiGLUFFN(d_model, d_ff_swiglu),
        f'GeGLU FFN (d_ff={d_ff_swiglu})': GeGLUFFN(d_model, d_ff_swiglu),
    }
    
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"{'FFN 类型':<30} {'参数量':>15} {'输出形状':>20}")
    print("-" * 70)
    
    for name, ffn in variants.items():
        params = sum(p.numel() for p in ffn.parameters())
        output = ffn(x)
        print(f"{name:<30} {params:>15,} {str(output.shape):>20}")

if __name__ == "__main__":
    compare_ffn_variants()
```

---

## 五、残差连接与梯度流

### 5.1 残差连接的作用

残差连接（Residual Connection）是深度网络训练的关键技术：

$$
x_{l+1} = x_l + f_l(x_l)
$$

**📌 核心优势**：
1. **梯度直通**：梯度可以直接通过恒等映射传递
2. **信息保留**：原始信息可以无损传递到后续层
3. **学习简化**：网络只需学习残差（增量信息）

### 5.2 梯度流分析

考虑 $L$ 层网络：

$$
x_L = x_0 + \sum_{l=1}^{L} f_l(x_{l-1})
$$

梯度为：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left(I + \sum_{l=1}^{L} \frac{\partial f_l}{\partial x_0}\right)
$$

即使 $\frac{\partial f_l}{\partial x_0}$ 很小，恒等项 $I$ 保证梯度不会消失。

### 5.3 不同残差设计

**标准残差**

```
x ──────────────────→ (+) ─→ output
│                      ↑
└→ Layer → Layer ──────┘
```

**缩放残差（ReZero）**

$$
x_{l+1} = x_l + \alpha \cdot f_l(x_l)
$$

其中 $\alpha$ 是可学习参数，初始化为 0。

**DeepNet 缩放**

$$
x_{l+1} = x_l + \alpha \cdot f_l(\beta \cdot x_l)
$$

用于训练超深模型（如 BLOOM-176B）。

### 5.4 PyTorch 实现

```python
class ReZeroLayer(nn.Module):
    """ReZero 残差连接"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        # 可学习的残差缩放因子，初始化为 0
        self.res_scale_attn = nn.Parameter(torch.zeros(1))
        self.res_scale_ffn = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, attn_mask=None):
        # ReZero + Self-Attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.res_scale_attn * attn_out
        
        # ReZero + FFN
        ffn_out = self.ffn(x)
        x = x + self.res_scale_ffn * ffn_out
        
        return x


class ScaledResidualLayer(nn.Module):
    """带缩放的残差连接（用于超深网络）"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 attn_scale: float = 1.0, ffn_scale: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.attn_scale = attn_scale
        self.ffn_scale = ffn_scale
    
    def forward(self, x, attn_mask=None):
        # 缩放残差 + Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.attn_scale * attn_out
        
        # 缩放残差 + FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.ffn_scale * ffn_out
        
        return x
```

---

## 六、激活函数选择

### 6.1 常用激活函数

LLM 中常用的激活函数：

| 激活函数 | 公式 | 特点 | 使用模型 |
|----------|------|------|----------|
| **ReLU** | $\max(0, x)$ | 简单高效，存在"死神经元"问题 | 早期模型 |
| **GELU** | $x \cdot \Phi(x)$ | 平滑，适合 Transformer | BERT、GPT-2 |
| **Swish** | $x \cdot \sigma(x)$ | 平滑，负值有梯度 | LLaMA、PaLM |
| **SiLU** | $x \cdot \sigma(x)$ | Swish 的特例 | LLaMA |
| **GLU** | $(Wx) \otimes \sigma(Vx)$ | 门控结构 | 多种 LLM |

### 6.2 GELU 详解

Gaussian Error Linear Unit (GELU)：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

近似计算：

$$
\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
$$

**💡 为什么 GELU 适合 Transformer？**

1. **非单调性**：负值区域有非零输出，避免梯度完全消失
2. **平滑性**：处处可微，优化更稳定
3. **随机正则化**：可解释为随机 Dropout 的平滑版本

### 6.3 Swish/SiLU

Swish 激活函数：

$$
\text{Swish}(x) = x \cdot \sigma(\beta x)
$$

当 $\beta = 1$ 时，称为 SiLU（Sigmoid Linear Unit）：

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

### 6.4 激活函数可视化对比

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_activations():
    """可视化不同激活函数"""
    x = torch.linspace(-5, 5, 200)
    
    activations = {
        'ReLU': F.relu(x),
        'GELU': F.gelu(x),
        'Swish/SiLU': F.silu(x),  # SiLU = Swish(β=1)
        'Sigmoid': torch.sigmoid(x),
        'Tanh': torch.tanh(x),
    }
    
    plt.figure(figsize=(12, 8))
    
    for i, (name, y) in enumerate(activations.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(x.numpy(), y.numpy(), linewidth=2)
        plt.title(name)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('activations_comparison.png', dpi=150, bbox_inches='tight')
    print("激活函数对比图已保存至 'activations_comparison.png'")

if __name__ == "__main__":
    visualize_activations()
```

### 6.5 激活函数选择建议

| 场景 | 推荐激活函数 | 原因 |
|------|--------------|------|
| Transformer 编码器 | GELU | 平滑，训练稳定 |
| Transformer 解码器 | GELU 或 Swish | 主流选择 |
| FFN 中使用 GLU | Swish/SiLU | 与门控结构配合 |
| 推理优化场景 | ReLU | 可利用稀疏性加速 |
| 极深网络 | GELU + 残差缩放 | 训练稳定 |

---

## 七、主流 LLM 架构对比

### 7.1 GPT 系列架构

**GPT-1/2/3 架构特点**

| 特性 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| 层数 | 12 | 48 | 96 |
| 隐藏维度 | 768 | 1600 | 12288 |
| 注意力头数 | 12 | 25 | 96 |
| 参数量 | 117M | 1.5B | 175B |
| 位置编码 | 学习式 | 学习式 | 学习式 |
| 归一化 | LayerNorm | LayerNorm | LayerNorm |
| 归一化位置 | Pre-Norm | Pre-Norm | Pre-Norm |
| FFN 激活 | GELU | GELU | GELU |

**核心设计**：
- Pre-Norm + 学习式位置编码
- 标准 MHA 注意力
- GELU 激活的 FFN

### 7.2 LLaMA 架构

**LLaMA 的关键改进**

| 组件 | LLaMA 设计 | 优势 |
|------|------------|------|
| 归一化 | RMSNorm | 计算更高效 |
| 位置编码 | RoPE | 支持外推，相对位置感知 |
| 注意力 | GQA (LLaMA2+) | 推理效率高 |
| FFN | SwiGLU | 性能更好 |
| 偏置项 | 无偏置 | 参数效率 |

**LLaMA 模型配置**

| 模型 | 层数 | 隐藏维度 | 头数 | 参数量 |
|------|------|----------|------|--------|
| LLaMA-7B | 32 | 4096 | 32 | 7B |
| LLaMA-13B | 40 | 5120 | 40 | 13B |
| LLaMA-33B | 60 | 6656 | 52 | 33B |
| LLaMA-65B | 80 | 8192 | 64 | 65B |

### 7.3 PaLM 架构

**PaLM 的关键特点**

| 特性 | PaLM 设计 |
|------|-----------|
| 注意力 | MQA（多查询注意力） |
| FFN | SwiGLU |
| 归一化 | LayerNorm |
| 偏置项 | 无偏置 |
| 激活函数 | Swish |
| 并行训练 | 完全并行化 |

### 7.4 架构演进总结

```
Transformer (2017)
    │
    ├── Post-Norm + 学习式位置编码
    │
    ↓
GPT-2 (2019)
    │
    ├── Pre-Norm（训练稳定）
    │
    ↓
GPT-3 (2020)
    │
    ├── 规模扩大，涌现能力
    │
    ↓
PaLM (2022)
    │
    ├── MQA（推理加速）
    ├── SwiGLU FFN
    │
    ↓
LLaMA (2023)
    │
    ├── RMSNorm
    ├── RoPE 位置编码
    ├── SwiGLU FFN
    ├── GQA（LLaMA2）
    │
    ↓
LLaMA 3/各类开源模型...
```

### 7.5 架构实现对比

```python
class GPTBlock(nn.Module):
    """GPT 风格的 Decoder 层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Pre-LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 标准 MHA
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        # 标准 FFN with GELU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Pre-Norm + MHA + Residual
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), 
                                        attn_mask=attn_mask, need_weights=False)[0])
        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.ln2(x))
        return x


class LLaMABlock(nn.Module):
    """LLaMA 风格的 Decoder 层"""
    
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.1):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads  # GQA: num_kv_heads < num_heads
        
        # RMSNorm（无偏置）
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
        # GQA 注意力（简化实现）
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        
        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, d_ff)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Pre-Norm + GQA + Residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Pre-Norm + SwiGLU + Residual
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## 八、模型规模设计

### 8.1 缩放定律（Scaling Laws）

Kaplan et al. (2020) 发现模型性能与三个因素呈幂律关系：

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

其中：
- $N$：模型参数量
- $D$：训练数据量（token 数）
- $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$

**📌 核心结论**：
1. 参数量和数据量同等重要
2. 模型越大，越需要更多数据
3. 最优计算分配：$N \propto C^{0.73}$，$D \propto C^{0.27}$

### 8.2 模型参数配置

**层数与隐藏维度的关系**

经验法则：

$$
d \approx 128 \times \sqrt[3]{\frac{N}{12}}
$$

$$
L \approx \frac{N}{12 \times d^2}
$$

**注意力头数选择**

通常每个头的维度 $d_k = d / h$ 在 64-128 之间：

$$
h = \frac{d}{d_k}, \quad d_k \in [64, 128]
$$

### 8.3 主流模型配置对比

| 模型 | 参数量 | 层数 | 隐藏维度 | 头数 | $d_k$ | FFN维度 |
|------|--------|------|----------|------|-------|---------|
| GPT-2 Small | 117M | 12 | 768 | 12 | 64 | 3072 |
| GPT-2 Medium | 345M | 24 | 1024 | 16 | 64 | 4096 |
| GPT-2 Large | 762M | 36 | 1280 | 20 | 64 | 5120 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 64 | 6400 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 16384 |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 20480 |
| GPT-3 175B | 175B | 96 | 12288 | 96 | 128 | 49152 |
| LLaMA 7B | 7B | 32 | 4096 | 32 | 128 | 11008 |
| LLaMA 13B | 13B | 40 | 5120 | 40 | 128 | 13824 |
| LLaMA 65B | 65B | 80 | 8192 | 64 | 128 | 22016 |

### 8.4 参数量计算

对于一个 Decoder-only 模型：

**嵌入层**：
$$
P_{emb} = V \times d + L_{max} \times d
$$
其中 $V$ 是词表大小，$L_{max}$ 是最大序列长度。

**每层参数**：
$$
P_{layer} = P_{attn} + P_{ffn} + P_{norm}
$$

$$
P_{attn} = 4d^2 \quad (\text{Q, K, V, O 投影})
$$

$$
P_{ffn} = 2 \times d \times d_{ff} \quad (\text{标准 FFN})
$$

$$
P_{norm} = 4d \quad (\text{两个 LayerNorm})
$$

**总参数量**：
$$
P_{total} = P_{emb} + L \times P_{layer} + d \times V
$$

```python
def calculate_llm_params(vocab_size, d_model, num_layers, num_heads, d_ff, 
                          max_seq_len=2048, use_rope=True, use_swiglu=True):
    """
    计算 LLM 参数量
    
    Args:
        vocab_size: 词表大小
        d_model: 隐藏维度
        num_layers: 层数
        num_heads: 注意力头数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
        use_rope: 是否使用 RoPE（无需位置嵌入）
        use_swiglu: 是否使用 SwiGLU（三个投影矩阵）
    """
    params = {}
    
    # 词嵌入
    params['token_embedding'] = vocab_size * d_model
    
    # 位置编码（如果使用学习式）
    params['position_embedding'] = 0 if use_rope else max_seq_len * d_model
    
    # 每层参数
    # 注意力：Q, K, V, O 四个投影（无偏置）
    d_k = d_model // num_heads
    params['attention_per_layer'] = 4 * d_model * d_model
    
    # FFN
    if use_swiglu:
        # SwiGLU 有三个投影矩阵
        params['ffn_per_layer'] = 3 * d_model * d_ff
    else:
        # 标准 FFN
        params['ffn_per_layer'] = 2 * d_model * d_ff
    
    # LayerNorm (两个)
    params['norm_per_layer'] = 2 * 2 * d_model  # weight + bias
    
    # 单层总计
    params['per_layer'] = (params['attention_per_layer'] + 
                           params['ffn_per_layer'] + 
                           params['norm_per_layer'])
    
    # 所有层
    params['all_layers'] = params['per_layer'] * num_layers
    
    # 最终 LayerNorm
    params['final_norm'] = 2 * d_model
    
    # 输出层（通常与词嵌入共享权重）
    params['output_head'] = vocab_size * d_model
    
    # 总计
    params['total'] = (params['token_embedding'] + 
                       params['position_embedding'] +
                       params['all_layers'] +
                       params['final_norm'])
    
    # 打印结果
    print(f"\n{'组件':<25} {'参数量':>15,}")
    print("-" * 45)
    for key, value in params.items():
        if key != 'total':
            print(f"{key:<25} {value:>15,}")
    print("-" * 45)
    print(f"{'总计':<25} {params['total']:>15,}")
    print(f"{'约':<25} {params['total'] / 1e9:>12.2f} B")
    
    return params['total']


def demo_model_configs():
    """演示不同配置的参数量计算"""
    print("=" * 60)
    print("LLaMA 7B 配置")
    print("=" * 60)
    calculate_llm_params(
        vocab_size=32000,
        d_model=4096,
        num_layers=32,
        num_heads=32,
        d_ff=11008,
        use_rope=True,
        use_swiglu=True
    )
    
    print("\n" + "=" * 60)
    print("LLaMA 13B 配置")
    print("=" * 60)
    calculate_llm_params(
        vocab_size=32000,
        d_model=5120,
        num_layers=40,
        num_heads=40,
        d_ff=13824,
        use_rope=True,
        use_swiglu=True
    )
    
    print("\n" + "=" * 60)
    print("GPT-2 Medium 配置")
    print("=" * 60)
    calculate_llm_params(
        vocab_size=50257,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        use_rope=False,
        use_swiglu=False
    )

if __name__ == "__main__":
    demo_model_configs()
```

### 8.5 模型设计最佳实践

**📌 规模配置建议**

| 目标参数量 | 推荐层数 | 隐藏维度 | 头数 | FFN维度 |
|------------|----------|----------|------|---------|
| ~125M | 12 | 768 | 12 | 3072 |
| ~350M | 24 | 1024 | 16 | 4096 |
| ~760M | 24 | 1536 | 16 | 6144 |
| ~1.3B | 24 | 2048 | 32 | 8192 |
| ~2.7B | 32 | 2560 | 32 | 10240 |
| ~6.7B | 32 | 4096 | 32 | 16384 |
| ~13B | 40 | 5120 | 40 | 20480 |

**💡 设计原则**

1. **深度 vs 宽度**：现代 LLM 倾向于更深而非更宽
2. **头维度**：$d_k$ 通常在 64-128 之间，太小限制表达能力
3. **FFN 扩展比**：标准为 4x，SwiGLU 通常为 8/3x
4. **归一化选择**：Pre-Norm + RMSNorm 是现代标配
5. **位置编码**：RoPE 支持长度外推，适合生成任务

---

## 九、知识点关联

### 9.1 与其他章节的关系

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM 架构知识图谱                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [分词技术] ──→ [嵌入层] ──→ [注意力机制] ──→ [模型架构] ←─ [FFN]   │
│       │              │              │              │               │
│       ↓              ↓              ↓              ↓               │
│  Token IDs      词向量表示      序列建模能力    完整 LLM           │
│                                                                    │
│  [预训练原理] ←─────────────── [模型架构] ──────────→ [推理优化]   │
│       │                              │                  │          │
│       ↓                              ↓                  ↓          │
│  训练策略选择                   架构决定性能        KV Cache 设计    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 架构设计的决策链

```
任务需求（生成/理解）
       │
       ↓
┌──────────────────┐
│   架构类型选择    │  Decoder-only / Encoder-Decoder
└──────────────────┘
       │
       ↓
┌──────────────────┐
│   规模设计        │  参数量 → 层数、维度、头数
└──────────────────┘
       │
       ↓
┌──────────────────┐
│   组件选择        │  归一化、激活函数、FFN 类型
└──────────────────┘
       │
       ↓
┌──────────────────┐
│   注意力机制      │  MHA / GQA / MQA
└──────────────────┘
       │
       ↓
┌──────────────────┐
│   位置编码        │  学习式 / RoPE / ALiBi
└──────────────────┘
```

---

## 十、核心考点

### 10.1 概念理解题

**Q1: 为什么 LLM 普遍采用 Decoder-only 架构而非 Encoder-Decoder？**

<details>
<summary>点击查看答案</summary>

1. **生成任务适配**：Decoder 的因果注意力天然支持自回归生成
2. **参数效率**：相同参数量下可以构建更深的网络
3. **训练简洁**：无需复杂的编码器-解码器交叉注意力
4. **KV Cache 友好**：推理时增量缓存效率高
5. **涌现能力**：大规模 Decoder-only 模型展现出更强的涌现能力

</details>

**Q2: Pre-Norm 相比 Post-Norm 有什么优势？**

<details>
<summary>点击查看答案</summary>

1. **梯度流优化**：存在干净的残差路径，梯度可以直接跳跃传递
2. **训练稳定性**：无需学习率预热，深层网络训练更稳定
3. **实现简单**：不需要特殊初始化技巧
4. **权衡**：相同层数下性能略差，需要 Final LayerNorm

</details>

### 10.2 计算分析题

**Q3: 计算 LLaMA-7B 的参数量**

给定配置：
- 词表大小：32000
- 隐藏维度：4096
- 层数：32
- 注意力头数：32
- FFN 维度：11008
- 使用 RoPE、SwiGLU、RMSNorm

<details>
<summary>点击查看答案</summary>

```
词嵌入：32000 × 4096 = 131,072,000

每层：
- 注意力（无偏置）：4 × 4096 × 4096 = 67,108,864
- SwiGLU FFN：3 × 4096 × 11008 = 135,266,304
- RMSNorm（2个）：2 × 4096 = 8,192
- 单层总计：202,383,360

所有层：32 × 202,383,360 = 6,476,267,520

最终 RMSNorm：4096

总计：131,072,000 + 6,476,267,520 + 4096 ≈ 6.61B

（实际约 6.7B，差异来自词表大小等细节）
```

</details>

### 10.3 设计应用题

**Q4: 如果要设计一个 3B 参数的中文 LLM，你会如何选择配置？**

<details>
<summary>点击查看答案</summary>

参考配置：
- 词表大小：约 65000（中文需要更大词表）
- 隐藏维度：2560
- 层数：32
- 注意力头数：32（头维度 80）
- FFN 维度：10240（SwiGLU，约 4x）
- 位置编码：RoPE
- 归一化：RMSNorm
- 注意力：GQA（可选，提升推理效率）

设计理由：
1. 中文词表需要更大以覆盖常用字词
2. 层维度比约 1:4（深度与宽度平衡）
3. 采用现代架构组件（RoPE、SwiGLU、RMSNorm）

</details>

---

## 十一、学习建议

### 11.1 学习路径

```
阶段一：基础理解（1-2周）
├── 理解 Transformer 基本结构
├── 掌握 Decoder-only 架构特点
└── 阅读原始论文（Attention Is All You Need, GPT 系列）

阶段二：组件深入（2-3周）
├── 实现 Pre-Norm vs Post-Norm 对比
├── 实现不同 FFN 变体（标准、SwiGLU、GeGLU）
├── 理解 RMSNorm 原理
└── 动手搭建小型 GPT 模型

阶段三：架构设计（2-3周）
├── 研究主流模型配置（GPT、LLaMA、PaLM）
├── 理解 Scaling Laws
├── 实现完整的 LLaMA 风格模型
└── 进行配置消融实验

阶段四：前沿探索（持续）
├── 关注新架构创新（MoE、混合注意力）
├── 阅读最新论文
└── 参与开源项目实践
```

### 11.2 推荐实践项目

| 项目 | 难度 | 学习重点 |
|------|------|----------|
| 从零实现 GPT-2 Small | ⭐⭐⭐ | 完整架构实现 |
| 实现 LLaMA 风格模型 | ⭐⭐⭐⭐ | 现代架构组件 |
| 配置消融实验 | ⭐⭐⭐ | 理解各组件影响 |
| 小规模模型预训练 | ⭐⭐⭐⭐⭐ | 全流程实践 |

### 11.3 常见误区

| 误区 | 正确理解 |
|------|----------|
| "参数越多越好" | 架构设计、训练数据、对齐技术同样重要 |
| "Pre-Norm 一定更好" | Post-Norm 在浅层网络可能更好，需权衡 |
| "FFN 只是简单映射" | FFN 设计（SwiGLU 等）对性能影响显著 |
| "所有 LLM 架构相同" | 细节差异（归一化、激活函数等）带来性能差异 |

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA 架构详解
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 缩放定律
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE 位置编码

---

*下一章：[注意力机制详解](./attention-mechanisms.md) - 深入了解 MHA、MQA、GQA 等注意力变体*
