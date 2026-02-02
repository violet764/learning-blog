# Transformer Architecture and Self-Attention Mechanism

## 1. 生物学启发与历史背景

### 1.1 注意力机制的生物学基础
注意力机制受到人类视觉系统的启发。当人类观察复杂场景时，大脑会选择性关注重要区域，忽略不相关信息。这种"选择性注意力"机制在神经科学中被称为"瓶颈注意力"。

### 1.2 从RNN到Transformer的演进
传统RNN在处理长序列时面临梯度消失和计算效率低下的问题。Transformer通过并行计算和全局注意力机制解决了这些限制。

## 2. 自注意力机制详细数学推导

### 2.1 基本注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: 查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$: 键矩阵  
- $V \in \mathbb{R}^{m \times d_v}$: 值矩阵
- $d_k$: 键/查询维度，用于缩放防止softmax饱和

### 2.2 梯度推导

令 $A = \frac{QK^T}{\sqrt{d_k}}$，注意力权重 $W = \text{softmax}(A)$，输出 $O = WV$。

损失函数 $L$ 对 $O$ 的梯度为 $\frac{\partial L}{\partial O}$，则：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial O} V^T$$

$$\frac{\partial L}{\partial A_{ij}} = \sum_k \frac{\partial L}{\partial W_{ik}} \frac{\partial W_{ik}}{\partial A_{ij}}$$

其中softmax的梯度为：
$$\frac{\partial W_{ik}}{\partial A_{ij}} = W_{ik}(\delta_{kj} - W_{ij})$$

### 2.3 多头注意力数学形式化

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \ldots, head_h)W^O$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

## 3. Transformer架构详解

### 3.1 编码器层结构

每个编码器层包含：
1. **多头自注意力**：捕获序列内部依赖关系
2. **前馈神经网络**：两层全连接+ReLU激活
3. **残差连接**：缓解梯度消失
4. **层归一化**：稳定训练过程

### 3.2 位置编码

使用正弦和余弦函数编码位置信息：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

## 4. PyTorch完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权和
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def forward(self, src, src_mask=None):
        # 自注意力子层
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈子层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 
                                                   dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 
                                                        num_encoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        # Transformer编码器
        output = self.transformer_encoder(src, src_mask)
        output = self.fc_out(output)
        
        return output

# 使用示例
if __name__ == "__main__":
    model = Transformer(vocab_size=10000, d_model=512, nhead=8)
    src = torch.randint(0, 10000, (32, 50))  # batch_size=32, seq_len=50
    output = model(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
```

## 5. 训练技巧与优化

### 5.1 学习率调度
使用warmup策略：
$$lr = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$

### 5.2 梯度裁剪
防止梯度爆炸：
$$g \leftarrow g \cdot \min\left(1, \frac{\text{clip_value}}{\|g\|}\right)$$

## 6. 延伸学习

### 6.1 核心论文
1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Transformer原始论文
2. **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** - BERT模型
3. **[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)** - 代码注释版Transformer

### 6.2 开源项目
1. **[HuggingFace Transformers](https://github.com/huggingface/transformers)** - 最流行的Transformer库
2. **[Fairseq](https://github.com/facebookresearch/fairseq)** - Facebook的序列建模工具包
3. **[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)** - Google的Transformer实现

### 6.3 进阶阅读
1. **注意力机制变体**：相对位置编码、稀疏注意力、线性注意力
2. **高效Transformer**：Reformer、Linformer、Performer
3. **视觉Transformer**：ViT、DETR、Swin Transformer

## 7. 应用场景

### 7.1 自然语言处理
- 机器翻译
- 文本生成
- 问答系统
- 情感分析

### 7.2 计算机视觉
- 图像分类
- 目标检测
- 图像生成

### 7.3 多模态任务
- 视觉问答
- 图像描述生成
- 跨模态检索