# 嵌入层详解

## 章节概述

嵌入层（Embedding Layer）是大语言模型处理离散输入的入口，它将离散的 token ID 映射为连续的向量表示，是连接符号系统与神经网络表示的桥梁。本章从词嵌入的发展历程出发，深入解析 Word2Vec 的数学原理，重点讲解位置编码的设计思想与主流方案（正弦余弦编码、RoPE、ALiBi 等），帮助读者全面理解嵌入层在现代 LLM 中的核心作用。

---

## 一、词嵌入概述

### 1.1 从符号到向量

自然语言是由离散符号（词、字、子词）组成的序列，而神经网络只能处理连续的数值向量。词嵌入的核心任务是将离散的词映射到连续的向量空间，使语义相近的词在向量空间中距离相近。

**词嵌入的发展脉络：**

```
One-Hot 编码
    │ 问题：维度爆炸、无法表达语义相似性
    ↓
分布式表示 (Word2Vec, GloVe)
    │ 问题：静态嵌入，无法处理多义词
    ↓
上下文嵌入 (BERT, GPT)
    │ 同一词在不同上下文有不同表示
    ↓
现代 LLM 嵌入层
    │ 与位置编码结合，支持超长序列
```

### 1.2 One-Hot 编码

**基本思想**：每个词对应一个独热向量，向量长度等于词汇表大小。

$$
\text{OneHot}(w_i) = [0, 0, ..., 1, ..., 0, 0]
$$

其中第 $i$ 个位置为 1，其余为 0。

```python
import torch
import torch.nn as nn

def one_hot_encode(word_ids, vocab_size):
    """
    One-Hot 编码
    
    Args:
        word_ids: [batch_size, seq_len] 词ID序列
        vocab_size: 词汇表大小
    
    Returns:
        one_hot: [batch_size, seq_len, vocab_size]
    """
    return torch.nn.functional.one_hot(word_ids, num_classes=vocab_size).float()

# 示例
vocab_size = 10000
word_ids = torch.tensor([[1, 5, 100, 999]])
one_hot = one_hot_encode(word_ids, vocab_size)
print(f"词汇表大小: {vocab_size}")
print(f"One-Hot 向量形状: {one_hot.shape}")  # [1, 4, 10000]
print(f"非零元素数量: {one_hot.sum().item()}")  # 4
```

**One-Hot 编码的问题：**

| 问题 | 描述 | 影响 |
|------|------|------|
| 维度爆炸 | 向量维度 = 词汇表大小（通常 10k-100k+） | 存储和计算开销巨大 |
| 稀疏性 | 每个向量只有一个非零元素 | 参数利用率低 |
| 语义缺失 | 任意两向量正交，无法表达相似性 | $\vec{v}_{\text{cat}} \cdot \vec{v}_{\text{dog}} = 0$ |

### 1.3 分布式表示

**核心思想**：用低维稠密向量表示词，语义相似的词在向量空间中距离相近。

$$
\text{Embedding}(w_i) = \mathbf{e}_i \in \mathbb{R}^d
$$

其中 $d$ 是嵌入维度（通常 128-4096）。

```python
class EmbeddingLayer(nn.Module):
    """嵌入层：将词ID映射为稠密向量"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # 嵌入矩阵: [vocab_size, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 词ID序列
        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        return self.embedding(x)

# 示例
vocab_size, embed_dim = 10000, 512
embedding_layer = EmbeddingLayer(vocab_size, embed_dim)

word_ids = torch.tensor([[1, 5, 100, 999]])
embeddings = embedding_layer(word_ids)

print(f"嵌入矩阵形状: {embedding_layer.embedding.weight.shape}")  # [10000, 512]
print(f"输出形状: {embeddings.shape}")  # [1, 4, 512]
print(f"参数量: {vocab_size * embed_dim:,}")  # 5,120,000
```

**分布式表示的优势：**

- 📌 **低维稠密**：维度可控，通常 256-4096
- 📌 **语义编码**：相似词在向量空间中接近
- 📌 **泛化能力**：可捕获词之间的语义关系

### 1.4 静态嵌入 vs 上下文嵌入

| 特性 | 静态嵌入 (Word2Vec, GloVe) | 上下文嵌入 (BERT, GPT) |
|------|---------------------------|----------------------|
| 表示方式 | 每个词固定一个向量 | 根据上下文动态生成 |
| 多义词处理 | 无法区分 | 可区分不同含义 |
| 计算复杂度 | 低（查表） | 高（需要 Transformer） |
| 典型应用 | 词向量预训练、相似度计算 | 下游 NLP 任务 |

**多义词示例：**

```
"bank" 的不同含义：
├── 银行: "I went to the bank to deposit money"
└── 河岸: "The boat is by the river bank"

静态嵌入: "bank" → 固定向量 e_bank（混淆两种含义）
上下文嵌入: 根据上下文生成不同向量 e_bank_银行 ≠ e_bank_河岸
```

---

## 二、Word2Vec 原理

### 2.1 核心思想

Word2Vec（Mikolov et al., 2013）通过预测上下文词来学习词向量。其核心假设是：**出现在相似上下文中的词具有相似的含义**。

两种架构：

| 架构 | 目标 | 描述 |
|------|------|------|
| Skip-gram | 给定中心词预测上下文 | 更适合大数据集 |
| CBOW | 给定上下文预测中心词 | 训练更快 |

### 2.2 Skip-gram 模型

**目标**：给定中心词 $w_t$，最大化其上下文词 $\{w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c}\}$ 的出现概率。

$$
\max_{\theta} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t; \theta)
$$

**概率计算**：

$$
P(w_o | w_c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_c)}
$$

其中：
- $\mathbf{v}_c$：中心词嵌入（中心词矩阵 $V$ 中的行）
- $\mathbf{u}_o$：上下文词嵌入（上下文矩阵 $U$ 中的行）
- $V$：词汇表

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """Skip-gram Word2Vec 模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 初始化
        initrange = 0.5 / embed_dim
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, center_words, context_words):
        """
        Args:
            center_words: [batch_size] 中心词ID
            context_words: [batch_size] 上下文词ID
        
        Returns:
            score: [batch_size] 点积分数
        """
        center_embeds = self.center_embeddings(center_words)  # [batch, dim]
        context_embeds = self.context_embeddings(context_words)  # [batch, dim]
        
        # 计算点积
        score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch]
        return score
    
    def get_embedding(self, word_id):
        """获取词向量（通常使用中心词嵌入）"""
        return self.center_embeddings(word_id)


class NegativeSamplingLoss(nn.Module):
    """负采样损失函数"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_score, neg_scores):
        """
        Args:
            pos_score: [batch_size] 正样本分数
            neg_scores: [batch_size, num_neg] 负样本分数
        
        Returns:
            loss: 标量
        """
        # 正样本损失: -log(sigmoid(pos_score))
        pos_loss = -F.logsigmoid(pos_score)
        
        # 负样本损失: -log(sigmoid(-neg_score)) = -log(1 - sigmoid(neg_score))
        neg_loss = -F.logsigmoid(-neg_scores).sum(dim=1)
        
        return (pos_loss + neg_loss).mean()


def train_skipgram_demo():
    """Skip-gram 训练演示"""
    vocab_size, embed_dim = 1000, 100
    model = SkipGramModel(vocab_size, embed_dim)
    criterion = NegativeSamplingLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 模拟数据: (中心词, 正上下文词, 负样本词)
    batch_size = 32
    num_neg = 5
    
    for step in range(100):
        # 随机生成训练数据
        center = torch.randint(0, vocab_size, (batch_size,))
        pos_context = torch.randint(0, vocab_size, (batch_size,))
        neg_context = torch.randint(0, vocab_size, (batch_size, num_neg))
        
        # 正样本分数
        pos_score = model(center, pos_context)
        
        # 负样本分数
        center_expanded = center.unsqueeze(1).expand(-1, num_neg)
        neg_scores = model(center_expanded.reshape(-1), neg_context.reshape(-1))
        neg_scores = neg_scores.view(batch_size, num_neg)
        
        # 计算损失
        loss = criterion(pos_score, neg_scores)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_skipgram_demo()
```

### 2.3 CBOW 模型

**目标**：给定上下文词 $\{w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c}\}$，预测中心词 $w_t$。

$$
P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_t^T \bar{\mathbf{v}})}{\sum_{w \in V} \exp(\mathbf{u}_w^T \bar{\mathbf{v}})}
$$

其中 $\bar{\mathbf{v}} = \frac{1}{2c}\sum_{j=-c, j \neq 0}^{c} \mathbf{v}_{t+j}$ 是上下文词向量的平均。

```python
class CBOWModel(nn.Module):
    """CBOW Word2Vec 模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        initrange = 0.5 / embed_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, context_words):
        """
        Args:
            context_words: [batch_size, 2c] 上下文词ID
        
        Returns:
            logits: [batch_size, vocab_size] 预测logits
        """
        # 获取上下文嵌入并平均
        embeds = self.embeddings(context_words)  # [batch, 2c, dim]
        avg_embeds = embeds.mean(dim=1)  # [batch, dim]
        
        # 预测中心词
        logits = self.output_proj(avg_embeds)  # [batch, vocab_size]
        return logits
```

### 2.4 负采样

**动机**：Softmax 的分母需要遍历整个词汇表，计算代价高昂。负采样通过将多分类问题转化为二分类问题解决这一难题。

**目标函数**：

$$
\mathcal{L} = -\log \sigma(\mathbf{u}_o^T \mathbf{v}_c) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_c)
$$

其中：
- $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 sigmoid 函数
- $k$ 是负样本数量（通常 5-20）
- $P_n(w) \propto f(w)^{3/4}$ 是噪声分布（$f(w)$ 是词频）

**负采样策略**：

```python
import numpy as np

class NegativeSampler:
    """负采样器"""
    
    def __init__(self, word_counts: dict, num_neg: int, power: float = 0.75):
        """
        Args:
            word_counts: {word_id: count} 词频字典
            num_neg: 负样本数量
            power: 采样概率的幂次（通常为0.75）
        """
        self.num_neg = num_neg
        
        # 计算采样概率: P(w) ∝ count(w)^power
        vocab_size = len(word_counts)
        counts = np.zeros(vocab_size)
        for word_id, count in word_counts.items():
            counts[word_id] = count
        
        # 平滑处理
        probs = np.power(counts, power)
        probs = probs / probs.sum()
        
        self.probs = probs
        
        # 预计算大表用于高效采样（别名采样）
        self.table_size = int(1e7)
        self.table = np.random.choice(vocab_size, size=self.table_size, p=probs)
    
    def sample(self, batch_size: int):
        """采样负样本"""
        indices = np.random.randint(0, self.table_size, (batch_size, self.num_neg))
        return torch.from_numpy(self.table[indices])


def demonstrate_negative_sampling():
    """演示负采样的效果"""
    # 模拟词频
    word_counts = {i: (i + 1) ** 2 for i in range(1000)}  # 词频服从幂律分布
    
    sampler = NegativeSampler(word_counts, num_neg=5)
    
    # 采样
    neg_samples = sampler.sample(10)
    print(f"负样本形状: {neg_samples.shape}")  # [10, 5]
    
    # 观察采样分布
    all_samples = sampler.sample(10000).flatten()
    from collections import Counter
    sample_counts = Counter(all_samples.tolist())
    
    print("\n高频词采样概率更高:")
    for word_id in [0, 1, 2, 10, 100]:
        print(f"  词 {word_id}: 采样 {sample_counts.get(word_id, 0)} 次")

if __name__ == "__main__":
    demonstrate_negative_sampling()
```

### 2.5 Word2Vec 的语义性质

**词类比任务**：

$$
\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}
$$

```python
def word_analogy(embeddings, word_to_id, id_to_word, a, b, c, top_k=5):
    """
    词类比: a - b + c = ?
    
    寻找最接近 (a - b + c) 的词
    """
    vec_a = embeddings[word_to_id[a]]
    vec_b = embeddings[word_to_id[b]]
    vec_c = embeddings[word_to_id[c]]
    
    target = vec_a - vec_b + vec_c
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(
        embeddings.weight.data,
        target.unsqueeze(0)
    )
    
    # 排除输入词
    for word in [a, b, c]:
        similarities[word_to_id[word]] = -float('inf')
    
    # 返回最相似的词
    top_indices = similarities.topk(top_k).indices.tolist()
    return [id_to_word[i] for i in top_indices]
```

---

## 三、嵌入层的数学表示

### 3.1 嵌入矩阵

嵌入层的核心是一个查找表（Lookup Table）：

$$
\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}
$$

其中：
- $|\mathcal{V}|$ 是词汇表大小
- $d$ 是嵌入维度

给定词 ID 序列 $\mathbf{x} = [x_1, x_2, ..., x_n]$，嵌入层输出：

$$
\mathbf{H} = [\mathbf{E}[x_1], \mathbf{E}[x_2], ..., \mathbf{E}[x_n]] \in \mathbb{R}^{n \times d}
$$

### 3.2 与 One-Hot 的等价性

嵌入操作等价于 One-Hot 编码后乘以嵌入矩阵：

$$
\mathbf{E}[x_i] = \mathbf{one\_hot}(x_i) \cdot \mathbf{E}
$$

但实际实现中，嵌入层直接通过索引查找，避免构造稀疏的 One-Hot 向量。

```python
def demonstrate_equivalence():
    """演示嵌入与 One-Hot 的等价性"""
    vocab_size, embed_dim = 10, 4
    batch_size, seq_len = 2, 3
    
    # 创建嵌入层
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    # 输入词ID
    word_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 方法1: 直接嵌入查找
    embeds_direct = embedding(word_ids)
    
    # 方法2: One-Hot 后矩阵乘法
    one_hot = F.one_hot(word_ids, num_classes=vocab_size).float()  # [batch, seq, vocab]
    embeds_onehot = torch.matmul(one_hot, embedding.weight)  # [batch, seq, dim]
    
    # 验证等价性
    print(f"两种方法结果是否相同: {torch.allclose(embeds_direct, embeds_onehot)}")
    print(f"直接查找结果:\n{embeds_direct[0, 0]}")
    print(f"One-Hot方法结果:\n{embeds_onehot[0, 0]}")
```

### 3.3 参数量与内存分析

嵌入层的参数量：

$$
\text{Params} = |\mathcal{V}| \times d
$$

**主流模型的嵌入参数量：**

| 模型 | 词汇表大小 | 嵌入维度 | 嵌入参数量 |
|------|-----------|----------|-----------|
| BERT-base | 30,522 | 768 | 23.4M |
| GPT-2 | 50,257 | 768 | 38.6M |
| GPT-3 | ~100,000 | 12,288 | ~1.2B |
| LLaMA | 32,000 | 4,096 | 131M |

```python
def analyze_embedding_memory():
    """分析嵌入层内存占用"""
    configs = [
        ("BERT-base", 30522, 768),
        ("GPT-2", 50257, 768),
        ("GPT-3-175B", 100000, 12288),
        ("LLaMA-7B", 32000, 4096),
        ("LLaMA-65B", 32000, 8192),
    ]
    
    print(f"{'模型':<15} {'词汇表':>10} {'维度':>8} {'参数量':>12} {'内存(FP16)':>12}")
    print("-" * 60)
    
    for name, vocab_size, dim in configs:
        params = vocab_size * dim
        memory_mb = params * 2 / 1024 / 1024  # FP16 = 2 bytes
        print(f"{name:<15} {vocab_size:>10,} {dim:>8,} {params:>12,} {memory_mb:>10.1f} MB")

if __name__ == "__main__":
    analyze_embedding_memory()
```

### 3.4 权重绑定

为减少参数量，可将输入嵌入矩阵与输出层权重共享：

$$
\mathbf{W}_{\text{output}} = \mathbf{E}^T
$$

**优缺点分析：**

| 优点 | 缺点 |
|------|------|
| 减少约 $|\mathcal{V}| \times d$ 参数 | 可能限制模型表达能力 |
| 输入输出表示一致 | 预训练和微调可能需要解绑 |

```python
class TiedEmbedding(nn.Module):
    """权重绑定的嵌入层"""
    
    def __init__(self, vocab_size: int, embed_dim: int, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.tie_weights = tie_weights
        
        if tie_weights:
            # 输出层与嵌入层共享权重
            self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
            self.output_proj.weight = self.embedding.weight
        else:
            self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] 词ID
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        embeds = self.embedding(x)  # [batch, seq, dim]
        logits = self.output_proj(embeds)  # [batch, seq, vocab]
        return logits
```

---

## 四、位置编码

### 4.1 为什么需要位置编码？

Transformer 的自注意力机制是**置换不变的**（Permutation Invariant）：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

对于输入序列的任意置换 $\pi$：

$$
\text{Attention}(\pi(\mathbf{Q}), \pi(\mathbf{K}), \pi(\mathbf{V})) = \pi(\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))
$$

这意味着模型无法感知 token 的位置信息，需要额外的位置编码来注入位置信息。

**位置编码的核心要求：**

1. **唯一性**：每个位置有唯一的编码
2. **确定性**：编码方式固定，不依赖输入
3. **泛化性**：能外推到训练时未见过的长度
4. **可学习**：编码能够与模型协同优化

### 4.2 正弦余弦位置编码（Sinusoidal）

**原论文公式**（Vaswani et al., 2017）：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

其中：
- $pos$ 是位置索引
- $i$ 是维度索引
- $d$ 是嵌入维度

**设计原理：**

1. **不同频率**：不同维度使用不同频率，捕获不同粒度的位置信息
2. **周期性**：正弦余弦函数的周期性使模型能学习相对位置
3. **外推能力**：对于任意位置都能计算编码

```python
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母: 10000^(2i/d) = exp(2i * -log(10000) / d)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加 batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + pe: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_sinusoidal_pe():
    """可视化正弦余弦位置编码"""
    import matplotlib.pyplot as plt
    
    d_model, max_len = 64, 100
    pe_layer = SinusoidalPositionalEncoding(d_model, max_len)
    pe = pe_layer.pe[0].numpy()  # [max_len, d_model]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：位置编码热力图
    ax1 = axes[0]
    im = ax1.imshow(pe, aspect='auto', cmap='RdBu')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Sinusoidal Positional Encoding')
    plt.colorbar(im, ax=ax1)
    
    # 右图：不同维度的波形
    ax2 = axes[1]
    positions = range(max_len)
    dims = [0, 1, 10, 20, 30]
    for dim in dims:
        ax2.plot(positions, pe[:, dim], label=f'dim {dim}')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Position Encoding at Different Dimensions')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('sinusoidal_pe.png', dpi=150, bbox_inches='tight')
    print("可视化保存至 sinusoidal_pe.png")

if __name__ == "__main__":
    visualize_sinusoidal_pe()
```

**相对位置关系的数学证明：**

对于位置 $pos + k$ 和 $pos$，存在线性关系：

$$
PE_{pos+k} = \mathbf{M}_k \cdot PE_{pos}
$$

其中 $\mathbf{M}_k$ 是一个与位置无关的旋转矩阵：

$$
\mathbf{M}_k = \begin{bmatrix}
\cos(k\omega_0) & \sin(k\omega_0) & 0 & 0 & \cdots \\
-\sin(k\omega_0) & \cos(k\omega_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(k\omega_1) & \sin(k\omega_1) & \cdots \\
0 & 0 & -\sin(k\omega_1) & \cos(k\omega_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

这意味着模型可以通过学习线性变换来捕获相对位置关系。

### 4.3 学习式位置编码

**核心思想**：将位置编码作为可训练参数，让模型自己学习最优的位置表示。

```python
class LearnedPositionalEncoding(nn.Module):
    """学习式位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习的位置嵌入
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        x = x + self.position_embeddings(positions)
        return self.dropout(x)
```

**对比分析：**

| 特性 | 正弦余弦编码 | 学习式编码 |
|------|-------------|-----------|
| 参数量 | 0 | $O(max\_len \times d)$ |
| 外推能力 | ✅ 理论上无限 | ❌ 受限于 max_len |
| 灵活性 | 固定公式 | 可学习最优表示 |
| 适用场景 | 需要长序列外推 | 序列长度固定 |

**BERT 和 GPT 的选择：**
- BERT：学习式位置编码（序列长度固定 512）
- GPT：学习式位置编码（但支持位置裁剪和扩展）

---

## 五、旋转位置编码（RoPE）

### 5.1 动机

正弦余弦编码和学习式编码都是将位置信息**加到**词嵌入上，这种方式对相对位置的建模能力有限。旋转位置编码（Rotary Position Embedding, RoPE）将位置信息通过**旋转矩阵**注入到注意力计算中，能更好地建模相对位置关系。

### 5.2 数学推导

**核心思想**：将 token 表示看作复数，位置编码表现为旋转操作。

对于二维向量 $\mathbf{x} = (x_1, x_2)$，位置 $m$ 的编码定义为：

$$
f(\mathbf{x}, m) = \begin{pmatrix}
x_1 \cos(m\theta) - x_2 \sin(m\theta) \\
x_1 \sin(m\theta) + x_2 \cos(m\theta)
\end{pmatrix}
$$

等价于复数乘法：

$$
f(\mathbf{x}, m) = \mathbf{x} \cdot e^{im\theta}
$$

**关键性质**：两个 token 的点积只依赖于它们的**相对位置**：

$$
\langle f(\mathbf{x}_m, m), f(\mathbf{y}_n, n) \rangle = \text{Re}[\mathbf{x}_m \overline{\mathbf{y}_n} e^{i(n-m)\theta}]
$$

**扩展到高维**：将 $d$ 维向量分成 $d/2$ 组，每组独立应用旋转：

$$
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots \\
x_{d-1} \\
x_d
\end{pmatrix}
\rightarrow
\begin{pmatrix}
x_1 \cos(m\theta_1) - x_2 \sin(m\theta_1) \\
x_1 \sin(m\theta_1) + x_2 \cos(m\theta_1) \\
x_3 \cos(m\theta_2) - x_4 \sin(m\theta_2) \\
x_3 \sin(m\theta_2) + x_4 \cos(m\theta_2) \\
\vdots \\
x_{d-1} \cos(m\theta_{d/2}) - x_d \sin(m\theta_{d/2}) \\
x_{d-1} \sin(m\theta_{d/2}) + x_d \cos(m\theta_{d/2})
\end{pmatrix}
$$

其中频率 $\theta_i = 10000^{-2i/d}$。

### 5.3 RoPE 的矩阵形式

RoPE 可以表示为对角矩阵：

$$
\mathbf{R}_{\Theta,m}^d = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmmatrix}
$$

位置 $m$ 的 token 表示为：

$$
\mathbf{x}_m' = \mathbf{R}_{\Theta,m}^d \mathbf{x}_m
$$

### 5.4 PyTorch 实现

```python
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算频率: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算 cos 和 sin 缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]
        
        # 复制以匹配完整维度
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len: int):
        """
        Args:
            x: [batch, seq_len, num_heads, head_dim] 或 [batch, num_heads, seq_len, head_dim]
            seq_len: 序列长度
        
        Returns:
            旋转后的 x
        """
        # 确保缓存足够长
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x):
    """
    将输入分成两半，进行旋转操作
    
    输入: [..., x_1, x_2, x_3, x_4, ...]
    输出: [..., -x_2, x_1, -x_4, x_3, ...]
    """
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    对 Q 和 K 应用旋转位置编码
    
    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
    
    Returns:
        q_rotated, k_rotated
    """
    # 调整维度以匹配
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 应用旋转: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """使用 RoPE 的多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)


def demo_rope():
    """演示 RoPE 的效果"""
    d_model, num_heads, seq_len = 64, 4, 16
    batch_size = 2
    
    attention = RoPEAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = attention(x, is_causal=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证相对位置性质
    # 两个 token 的注意力分数应该只依赖于相对位置差
    print("\n验证相对位置性质:")
    q = torch.randn(1, 1, 1, d_model // num_heads)  # 单个 query
    
    rope = RotaryPositionalEmbedding(d_model // num_heads)
    cos, sin = rope(q, seq_len=4)
    
    print("位置编码的 cos 值:")
    print(cos[:4, :4])

if __name__ == "__main__":
    demo_rope()
```

### 5.5 RoPE 的优势

| 特性 | 描述 |
|------|------|
| **相对位置感知** | 点积只依赖相对位置差 $n - m$ |
| **远程衰减** | 随着相对距离增加，注意力自然衰减 |
| **外推能力** | 可处理比训练时更长的序列 |
| **无参数** | 不增加模型参数 |
| **计算高效** | 可预计算 cos/sin 缓存 |

**主流模型采用情况：**
- LLaMA / LLaMA 2 / LLaMA 3
- Mistral
- Falcon
- Qwen
- Baichuan

---

## 六、ALiBi 位置编码

### 6.1 核心思想

ALiBi（Attention with Linear Biases, Press et al., 2022）不修改输入嵌入，而是在注意力分数上添加与相对距离成比例的偏置：

$$
\text{Attention}(q_i, k_j) = q_i \cdot k_j - m \cdot |i - j|
$$

其中 $m$ 是每个注意力头的斜率（slope），通常按几何级数设定。

### 6.2 斜率设置

对于 $n$ 个注意力头，斜率设为：

$$
m_h = \frac{1}{2^{\frac{8h}{n}}}, \quad h = 1, 2, ..., n
$$

或者从 $\frac{1}{2^8} = \frac{1}{256}$ 开始的几何级数。

```python
def get_alibi_slopes(num_heads: int):
    """
    计算 ALiBi 的斜率
    
    Args:
        num_heads: 注意力头数量
    
    Returns:
        slopes: [num_heads] 斜率张量
    """
    # 方法1: 几何级数
    # m_h = 2^(-8h/n)
    slopes = 1.0 / (2 ** (torch.arange(1, num_heads + 1) * 8 / num_heads))
    return slopes


class ALiBiAttention(nn.Module):
    """使用 ALiBi 的多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 注册斜率
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # 预计算 ALiBi 偏置
        self._build_alibi_bias(max_seq_len)
    
    def _build_alibi_bias(self, seq_len: int):
        """预计算 ALiBi 偏置矩阵"""
        # 相对位置矩阵
        positions = torch.arange(seq_len)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        
        # 取绝对值
        relative_pos = relative_pos.abs()
        
        # 对每个头计算偏置: -slope * |i - j|
        # [num_heads, seq, seq]
        alibi_bias = -self.slopes.unsqueeze(1).unsqueeze(2) * relative_pos.unsqueeze(0)
        
        self.register_buffer('alibi_bias', alibi_bias)
    
    def forward(self, x, mask=None, is_causal: bool = False):
        batch_size, seq_len, _ = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加 ALiBi 偏置
        scores = scores + self.alibi_bias[:, :seq_len, :seq_len]
        
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)


def visualize_alibi():
    """可视化 ALiBi 偏置"""
    import matplotlib.pyplot as plt
    
    num_heads, seq_len = 8, 32
    
    # 获取斜率
    slopes = get_alibi_slopes(num_heads)
    print(f"ALiBi 斜率: {slopes.tolist()}")
    
    # 计算偏置矩阵
    positions = torch.arange(seq_len)
    relative_pos = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for h in range(num_heads):
        bias = -slopes[h] * relative_pos
        im = axes[h].imshow(bias.numpy(), cmap='RdBu_r', aspect='auto')
        axes[h].set_title(f'Head {h}, slope={slopes[h]:.4f}')
        axes[h].set_xlabel('Key Position')
        axes[h].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[h])
    
    plt.suptitle('ALiBi Bias Matrices for Different Attention Heads', fontsize=14)
    plt.tight_layout()
    plt.savefig('alibi_bias.png', dpi=150, bbox_inches='tight')
    print("可视化保存至 alibi_bias.png")

if __name__ == "__main__":
    visualize_alibi()
```

### 6.3 ALiBi 的特性

| 特性 | 描述 |
|------|------|
| **无位置嵌入参数** | 不增加模型参数 |
| **强外推能力** | 可处理比训练时长 10 倍以上的序列 |
| **线性偏置** | 相对距离越远，注意力越弱 |
| **多头多样性** | 不同头使用不同斜率，捕获不同范围的位置关系 |

**外推能力对比：**

| 方法 | 训练长度 | 推理长度 | 性能保持 |
|------|---------|---------|---------|
| 正弦余弦 | 512 | 1024 | 性能下降明显 |
| 学习式 | 512 | 512+ | 无法直接使用 |
| RoPE | 2048 | 4096+ | 性能较好 |
| ALiBi | 512 | 8192 | 性能保持良好 |

---

## 七、其他位置编码方案

### 7.1 相对位置编码（Relative PE）

**核心思想**：不编码绝对位置，而是编码 token 之间的相对距离。

Shaw et al. (2018) 在注意力计算中引入可学习的相对位置嵌入：

$$
e_{ij} = \frac{q_i k_j^T + q_i s_{ij}^T}{\sqrt{d_k}}
$$

其中 $s_{ij}$ 是位置 $i$ 和 $j$ 之间的相对位置嵌入。

```python
class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相对位置嵌入: [-max_rel, ..., 0, ..., max_rel]
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, d_model)
    
    def forward(self, length: int):
        """
        生成相对位置编码矩阵
        
        Args:
            length: 序列长度
        
        Returns:
            relative_embeddings: [length, length, d_model]
        """
        # 计算相对位置索引
        range_vec = torch.arange(length)
        relative_pos = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)  # [len, len]
        
        # 裁剪到有效范围
        relative_pos = relative_pos.clamp(-self.max_relative_position, self.max_relative_position)
        
        # 转换为嵌入索引
        relative_pos_indices = relative_pos + self.max_relative_position  # [0, 2*max_rel]
        
        # 获取嵌入
        return self.relative_position_embeddings(relative_pos_indices)
```

### 7.2 T5 偏置

T5 模型使用简化的相对位置编码：在注意力分数上添加可学习的标量偏置。

```python
class T5RelativePositionBias(nn.Module):
    """T5 风格的相对位置偏置"""
    
    def __init__(self, num_heads: int, relative_attention_num_buckets: int = 32, 
                 max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.max_distance = max_distance
        
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
    
    def _relative_position_bucket(self, relative_position):
        """
        将相对位置映射到桶索引
        """
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.max_distance
        
        # 转换为正数
        relative_buckets = 0
        n = -relative_position.min()
        relative_position = relative_position + n
        
        # 对数桶（用于大距离）
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = relative_position_if_large.clamp(max=num_buckets - 1)
        
        relative_buckets = torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(self, length: int):
        """
        生成 T5 风格的相对位置偏置
        
        Returns:
            bias: [num_heads, length, length]
        """
        # 计算相对位置
        range_vec = torch.arange(length, device=self.relative_attention_bias.weight.device)
        relative_pos = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # 映射到桶
        relative_pos_bucket = self._relative_position_bucket(relative_pos)
        
        # 获取偏置值
        bias = self.relative_attention_bias(relative_pos_bucket)  # [length, length, num_heads]
        bias = bias.permute(2, 0, 1)  # [num_heads, length, length]
        
        return bias
```

### 7.3 位置编码对比总结

| 方法 | 参数量 | 外推能力 | 相对位置 | 代表模型 |
|------|--------|---------|---------|---------|
| 正弦余弦 | 0 | 中等 | 隐式 | Transformer 原版 |
| 学习式 | $O(L \cdot d)$ | 差 | 无 | BERT, GPT-2 |
| RoPE | 0 | 好 | 显式 | LLaMA, Mistral |
| ALiBi | 0 | 极好 | 显式 | BLOOM, MPT |
| 相对 PE | $O(k \cdot d)$ | 好 | 显式 | Transformer-XL |
| T5 偏置 | $O(b \cdot h)$ | 好 | 显式 | T5 |

---

## 八、嵌入层在 LLM 中的应用

### 8.1 完整的嵌入层架构

现代 LLM 的嵌入层通常包含以下组件：

```python
class LLMEmbedding(nn.Module):
    """LLM 完整嵌入层"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        pos_encoding_type: str = 'rope',  # 'sinusoidal', 'learned', 'rope', 'alibi'
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding_type = pos_encoding_type
        
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding_type == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding_type == 'rope':
            self.pos_encoding = RotaryPositionalEmbedding(d_model, max_seq_len)
            self.dropout = nn.Dropout(dropout)
        elif pos_encoding_type == 'alibi':
            self.pos_encoding = None  # ALiBi 在注意力中应用
            self.dropout = nn.Dropout(dropout)
        
        # LayerNorm (某些模型使用)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, seq_len, d_model]
            pos_info: 位置编码信息（用于 RoPE 或 ALiBi）
        """
        # Token 嵌入
        token_embeds = self.token_embedding(input_ids)
        
        if self.pos_encoding_type in ['sinusoidal', 'learned']:
            # 直接相加
            embeds = self.pos_encoding(token_embeds)
            return embeds, None
        
        elif self.pos_encoding_type == 'rope':
            # 返回 token 嵌入和 RoPE 信息
            embeds = self.dropout(token_embeds)
            cos, sin = self.pos_encoding(token_embeds, input_ids.size(1))
            return embeds, (cos, sin)
        
        elif self.pos_encoding_type == 'alibi':
            # ALiBi 在注意力中应用
            embeds = self.dropout(token_embeds)
            return embeds, 'alibi'
```

### 8.2 嵌入层初始化策略

```python
def init_embedding_weights(module: nn.Module, init_std: float = 0.02):
    """
    初始化嵌入层权重
    
    常见策略:
    1. 正态分布初始化 (GPT 系列)
    2. Xavier 均匀初始化
    3. 从预训练词向量加载
    """
    if isinstance(module, nn.Embedding):
        # 正态分布初始化
        module.weight.data.normal_(mean=0.0, std=init_std)
        
        # 如果有 padding token，将其初始化为 0
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=init_std)
        if module.bias is not None:
            module.bias.data.zero_()
    
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

### 8.3 实际模型配置示例

```python
# LLaMA 嵌入层配置
LLAMA_CONFIG = {
    'vocab_size': 32000,
    'd_model': 4096,
    'max_seq_len': 2048,
    'pos_encoding_type': 'rope',
    'num_layers': 32,
    'num_heads': 32,
}

# Mistral 嵌入层配置
MISTRAL_CONFIG = {
    'vocab_size': 32000,
    'd_model': 4096,
    'max_seq_len': 32768,  # 滑动窗口注意力
    'pos_encoding_type': 'rope',
    'window_size': 4096,
}

# BERT 嵌入层配置
BERT_CONFIG = {
    'vocab_size': 30522,
    'd_model': 768,
    'max_seq_len': 512,
    'pos_encoding_type': 'learned',
    'num_layers': 12,
    'num_heads': 12,
}

def create_model_embedding(config: dict):
    """根据配置创建嵌入层"""
    return LLMEmbedding(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        max_seq_len=config['max_seq_len'],
        pos_encoding_type=config['pos_encoding_type'],
    )
```

### 8.4 内存优化技巧

```python
# 技巧1: 权重量化
class QuantizedEmbedding(nn.Module):
    """量化嵌入层（减少内存占用）"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # 使用 int8 存储权重
        self.weight = nn.Parameter(
            torch.zeros(vocab_size, embed_dim, dtype=torch.int8),
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.ones(vocab_size), requires_grad=False)
    
    def forward(self, x):
        # 反量化后返回
        return self.weight[x].float() * self.scale[x].unsqueeze(-1)


# 技巧2: 自适应嵌入（Adaptive Embedding）
class AdaptiveEmbedding(nn.Module):
    """
    自适应嵌入: 高频词用大维度，低频词用小维度
    
    用于降低大词汇表的参数量
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, cutoffs: list):
        super().__init__()
        self.cutoffs = cutoffs
        
        # 高频词使用完整维度
        self.emb_high = nn.Embedding(cutoffs[0], embed_dim)
        
        # 低频词使用较小维度，再投影
        self.emb_low_list = nn.ModuleList()
        self.proj_list = nn.ModuleList()
        
        for i in range(len(cutoffs)):
            if i == 0:
                continue
            self.emb_low_list.append(
                nn.Embedding(cutoffs[i] - cutoffs[i-1], embed_dim // 4)
            )
            self.proj_list.append(
                nn.Linear(embed_dim // 4, embed_dim, bias=False)
            )
    
    def forward(self, x):
        # 简化实现，实际需要根据 cutoffs 分组处理
        return self.emb_high(x.clamp(max=self.cutoffs[0]-1))
```

---

## 九、知识点关联

### 9.1 知识图谱

```
嵌入层 (Embedding)
    │
    ├──→ 词嵌入 (Token Embedding)
    │       ├── One-Hot → 分布式表示
    │       ├── Word2Vec (Skip-gram, CBOW, 负采样)
    │       └── 静态嵌入 vs 上下文嵌入
    │
    ├──→ 位置编码 (Positional Encoding)
    │       ├── 绝对位置编码
    │       │   ├── 正弦余弦编码 (Sinusoidal)
    │       │   └── 学习式编码 (Learned)
    │       │
    │       └── 相对位置编码
    │           ├── RoPE (旋转位置编码)
    │           ├── ALiBi (线性偏置)
    │           ├── 相对位置嵌入 (Relative PE)
    │           └── T5 偏置
    │
    └──→ 与其他模块的关联
            ├──→ 注意力机制 (位置信息注入方式)
            ├──→ 分词 (词汇表大小影响嵌入参数量)
            └──→ 模型架构 (LLM 的嵌入层设计)
```

### 9.2 与分词的关系

| 分词方式 | 词汇表大小 | 嵌入参数量 | 影响 |
|----------|-----------|-----------|------|
| 字符级 | ~256 | 极小 | 序列过长 |
| 子词级 (BPE) | ~30k-50k | 中等 | 平衡 |
| 词级 | ~100k+ | 巨大 | 参数冗余 |

### 9.3 与注意力机制的关系

不同位置编码方式对注意力的影响：

```
位置编码注入点:

输入嵌入阶段:
├── 正弦余弦编码: x = token_embed + pos_embed
├── 学习式编码: x = token_embed + learned_pos
└── 然后送入注意力层

注意力计算阶段:
├── RoPE: 对 Q, K 应用旋转变换
├── ALiBi: 在注意力分数上添加偏置
└── 相对 PE: 在 QK^T 上添加相对位置项
```

---

## 十、章节核心考点

### 10.1 概念理解

- 嵌入层的作用：将离散 token 映射为连续向量
- One-Hot 编码的问题：维度爆炸、稀疏、无语义
- Word2Vec 的核心假设：相似上下文的词语义相近
- 位置编码的必要性：注意力机制的置换不变性

### 10.2 算法原理

| 主题 | 核心公式/概念 |
|------|--------------|
| Skip-gram | $\max \sum \log P(w_{context} \| w_{center})$ |
| 负采样 | $\mathcal{L} = -\log \sigma(u_o^T v_c) - \sum \log \sigma(-u_{neg}^T v_c)$ |
| 正弦余弦编码 | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ |
| RoPE | $\mathbf{x}_m' = \mathbf{R}_{\Theta,m} \mathbf{x}_m$，点积只依赖相对位置 |
| ALiBi | $attention = q \cdot k - m \cdot \|i-j\|$ |

### 10.3 实践技能

- 实现 Skip-gram 和 CBOW 模型
- 实现正弦余弦位置编码
- 实现 RoPE 的旋转操作
- 选择合适的位置编码方案

### 10.4 数学基础

- Softmax 的梯度问题与负采样
- 正弦余弦编码的相对位置性质证明
- RoPE 的旋转矩阵推导
- ALiBi 的斜率设计

---

## 十一、学习建议

### 11.1 理论学习路径

1. **基础阶段**
   - 理解 One-Hot 编码及其局限
   - 掌握 Word2Vec 的 Skip-gram 和 CBOW
   - 理解负采样的动机和原理

2. **进阶阶段**
   - 深入理解正弦余弦位置编码的设计原理
   - 掌握 RoPE 的数学推导
   - 对比各种位置编码的优劣

3. **应用阶段**
   - 分析主流 LLM 的嵌入层设计
   - 理解不同位置编码的外推能力
   - 掌握嵌入层的优化技巧

### 11.2 实践项目建议

```python
# 项目1: 从零实现 Word2Vec
# - 在小型语料上训练 Skip-gram 模型
# - 实现负采样损失
# - 验证词类比任务

# 项目2: 对比位置编码效果
# - 在相同 Transformer 架构上测试不同位置编码
# - 评估外推能力（训练短序列，测试长序列）
# - 分析注意力模式差异

# 项目3: 实现完整的 LLM 嵌入层
# - 支持 Token 嵌入 + 多种位置编码
# - 实现权重绑定
# - 添加内存优化（量化、自适应嵌入）
```

### 11.3 推荐阅读

**经典论文：**
- Efficient Estimation of Word Representations in Vector Space (Word2Vec)
- Attention Is All You Need (正弦余弦编码)
- RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
- Train Short, Test Long: Attention with Linear Biases (ALiBi)

**源码阅读：**
- Hugging Face Transformers: `modeling_llama.py`
- LLaMA 官方实现
- FairSeq 位置编码实现

---

*通过本章学习，您将全面理解嵌入层的核心原理与实践要点，为深入理解大语言模型的输入表示打下坚实基础。*
