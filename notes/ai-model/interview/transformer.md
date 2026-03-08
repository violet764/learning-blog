# Transformer 架构面试题

Transformer 是现代大语言模型的基础架构，本章整理了 Transformer 相关的核心面试题目，涵盖 Attention 机制、位置编码、多头注意力、Encoder-Decoder 结构等核心主题。

---

## 一、Attention 机制

### Q1: 请解释 Self-Attention 的工作原理

**基础回答：**

Self-Attention 让序列中的每个位置都能关注到其他所有位置，计算序列元素之间的相关性。

**深入回答：**

**计算过程**：

$$
\begin{aligned}
\text{输入: } &X \in \mathbb{R}^{n \times d} \\
1.\ &\text{线性变换得到 } Q, K, V: \\
   &Q = XW_Q, \ K = XW_K, \ V = XW_V \\
2.\ &\text{计算注意力分数:} \\
   &\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
3.\ &\text{输出维度: } \mathbb{R}^{n \times d_v}
\end{aligned}
$$

**详细步骤**：

**Self-Attention 结构图**：

![Self-Attention 结构图](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E3%80%91LLM%E7%BB%93%E6%9E%84%E5%85%A8%E8%A7%86%E5%9B%BE.png)

```python
# 假设输入序列长度 n，模型维度 d
Q = X @ W_Q  # (n, d_k)
K = X @ W_K  # (n, d_k)
V = X @ W_V  # (n, d_v)

# 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # (n, n)

# Softmax 归一化
attention_weights = softmax(scores, dim=-1)  # (n, n)

# 加权求和
output = attention_weights @ V  # (n, d_v)
```

**追问：为什么除以 $\sqrt{d_k}$？**

1. **防止点积过大**：当 $d_k$ 很大时，点积结果会很大
2. **Softmax 梯度问题**：输入过大时，softmax 输出接近 one-hot，梯度接近 0
3. **方差分析**：假设 Q、K 元素独立同分布，均值为 0，方差为 1
   - 点积的方差 $= d_k$（因为 $d_k$ 个元素相加）
   - 除以 $\sqrt{d_k}$ 后，方差归一化为 1

**追问：Self-Attention 的计算复杂度是多少？**

- 时间复杂度：$O(n^2 d)$
- 空间复杂度：$O(n^2 + nd)$

其中 $n$ 是序列长度，$d$ 是模型维度。$n^2$ 来自注意力矩阵的计算和存储。

---

### Q2: Self-Attention 和 Cross-Attention 的区别？

**基础回答：**

Self-Attention 的 Q、K、V 来自同一输入序列，Cross-Attention 的 Q 来自一个序列，K、V 来自另一个序列。

**深入回答：**

| 类型 | Q 来源 | K、V 来源 | 应用场景 |
|------|--------|-----------|----------|
| Self-Attention | 序列自身 | 序列自身 | 编码器、解码器自注意力 |
| Cross-Attention | 解码器 | 编码器输出 | Encoder-Decoder 结构 |
| Masked Self-Attention | 序列自身 | 序列自身（带掩码） | 解码器，防止看到未来 |

**追问：在 Transformer Decoder 中，Cross-Attention 如何工作？**

```
Decoder 层结构:
┌─────────────────────────────────────┐
│  Masked Self-Attention (Q=K=V=X)    │  ← 解码器自身注意力
├─────────────────────────────────────┤
│  Cross-Attention (Q=解码器, K=V=编码器输出) │  ← 关注编码器
├─────────────────────────────────────┤
│  Feed Forward Network               │
└─────────────────────────────────────┘
```

Cross-Attention 让解码器能够"看到"编码器的输出，实现信息从编码器到解码器的传递。

---

### Q3: 为什么使用点积 Attention 而不是加法 Attention？

**基础回答：**

点积 Attention 计算更快，且有理论保证。加法 Attention 虽然在某些情况下效果可能更好，但计算效率较低。

**深入回答：**

**两种 Attention 公式**：

$
\begin{aligned}
\text{点积 Attention:} \quad &\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \\
\text{加法 Attention (Bahdanau):} \quad &\text{score}(h_i, s_j) = v^T \tanh(W_1 h_i + W_2 s_j)
\end{aligned}
$

**对比分析**：

| 方面 | 点积 Attention | 加法 Attention |
|------|----------------|----------------|
| 计算复杂度 | $O(nd^2)$ 利用矩阵乘法优化 | $O(n^2 d)$ 需要逐元素计算 |
| 硬件友好 | 矩阵乘法，GPU 优化好 | 逐元素操作，并行度低 |
| 效果 | 大部分情况下相当或更好 | 小数据集可能更好 |
| 可解释性 | 直接反映相似度 | 通过非线性变换 |

**追问：什么情况下加法 Attention 更好？**

当 $d_k$ 很小（如 $d_k < 10$）时，加法 Attention 可能更好。因为：
- 点积在低维空间表达能力有限
- 加法 Attention 通过非线性变换增加表达能力

但在实际应用中，$d_k$ 通常较大（64-512），点积 Attention 更优。

---

## 二、多头注意力

### Q4: 为什么需要多头注意力？

**基础回答：**

多头注意力让模型同时关注不同位置的不同表示子空间，捕获更丰富的信息。

**深入回答：**

**单头 vs 多头**：

```python
# 单头注意力
output = Attention(Q, K, V)  # 只能学习一种注意力模式

# 多头注意力
heads = [Attention(Q_i, K_i, V_i) for i in range(h)]
output = Concat(heads) @ W_O  # h 种不同的注意力模式
```

**追问：从直觉上如何理解多头？**

想象阅读一句话："The animal didn't cross the street because it was too tired"

当关注 "it" 时：
- 头 1 可能关注 "animal"（主语）
- 头 2 可能关注 "tired"（原因）
- 头 3 可能关注 "didn't"（否定）

每个头学习不同的"关注模式"，最后综合所有信息。

**追问：多头注意力的参数量是多少？**

假设模型维度 $d_{\text{model}}$，头数 $h$：

$$
\begin{aligned}
\text{每个头的维度: } &d_k = d_{\text{model}} / h \\
\text{参数量:} \\
&\text{- } W_Q: d_{\text{model}} \times d_k \times h = d_{\text{model}} \times d_{\text{model}} \\
&\text{- } W_K: d_{\text{model}} \times d_k \times h = d_{\text{model}} \times d_{\text{model}} \\
&\text{- } W_V: d_{\text{model}} \times d_k \times h = d_{\text{model}} \times d_{\text{model}} \\
&\text{- } W_O: d_{\text{model}} \times d_{\text{model}} \\
\text{总参数量: } &4 \times d_{\text{model}}^2
\end{aligned}
$$

注意：多头注意力与单头注意力的参数量相同！

---

### Q5: MHA、MQA、GQA 有什么区别？

**基础回答：**

- MHA（Multi-Head Attention）：每个头有独立的 Q、K、V
- MQA（Multi-Query Attention）：多个头共享 K、V
- GQA（Grouped Query Attention）：介于两者之间，分组共享 K、V

**深入回答：**

**对比图解**：

```
MHA (h=4):
Q: [Q₁, Q₂, Q₃, Q₄]  K: [K₁, K₂, K₃, K₄]  V: [V₁, V₂, V₃, V₄]
    每个头独立的 Q、K、V

MQA (h=4):
Q: [Q₁, Q₂, Q₃, Q₄]  K: [K]  V: [V]
    所有头共享一个 K、V

GQA (h=4, g=2):
Q: [Q₁, Q₂, Q₃, Q₄]  K: [K₁, K₁, K₂, K₂]  V: [V₁, V₁, V₂, V₂]
    每组头共享一个 K、V
```

**性能对比**：

| 方法 | KV Cache 大小 | 计算速度 | 模型质量 |
|------|---------------|----------|----------|
| MHA | 100% | 基准 | 最好 |
| MQA | $1/h$ | 最快 | 略有下降 |
| GQA | $g/h$ | 较快 | 接近 MHA |

**MHA/MQA/GQA 结构图**：

![MHA/MQA/GQA 结构图](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91MHA%E3%80%81GQA%E3%80%81MQA%E3%80%81MLA.png)

**追问：为什么 LLaMA 2/3 使用 GQA？**

1. **推理加速**：KV Cache 减少到原来的 $g/h$
2. **质量保持**：相比 MQA，GQA 能更好地保持模型质量
3. **平衡选择**：在速度和质量之间取得平衡

LLaMA 2 70B 使用 GQA-8（8 组），在保持接近 MHA 质量的同时，显著提升推理速度。

---

## 三、位置编码

### Q6: 为什么 Transformer 需要位置编码？

**基础回答：**

Self-Attention 是置换不变的（permutation invariant），无法区分序列中元素的位置信息。位置编码为模型提供位置信息。

**深入回答：**

**置换不变性证明**：

```python
# 对于序列 [x₁, x₂, x₃] 和 [x₂, x₁, x₃]
# Attention 的计算结果只是位置交换，内容相同

# 设 P 是置换矩阵
Attention(PX) = softmax(PXW_QW_K^TX^TP^T)PXW_V
             = P · softmax(XW_QW_K^TX^T)XW_V
             = P · Attention(X)
```

结果只是位置交换，说明模型无法感知绝对位置。

**追问：RNN 为什么不需要位置编码？**

RNN 按顺序处理输入，隐状态天然包含历史信息。第 t 步的隐状态 h_t 是前 t 个输入的函数，隐式编码了位置信息。

---

### Q7: 正弦位置编码的设计原理是什么？

**基础回答：**

正弦位置编码使用不同频率的正弦和余弦函数，为每个位置生成唯一的编码。

**深入回答：**

**编码公式**：

$$
\begin{aligned}
\text{PE}(\text{pos}, 2i) &= \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right) \\
\text{PE}(\text{pos}, 2i+1) &= \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

**追问：为什么使用不同频率？**

不同频率让模型能够学习相对位置关系：

$$
\begin{aligned}
\text{对于频率 } \omega_k &= 1/10000^{2k/d}: \\
\text{PE}(\text{pos} + k) &\text{ 和 PE}(\text{pos}) \text{ 的关系:} \\
\sin((\text{pos}+k)\omega) &= \sin(\text{pos} \cdot \omega)\cos(k \cdot \omega) + \cos(\text{pos} \cdot \omega)\sin(k \cdot \omega)
\end{aligned}
$$

这表示 $\text{PE}(\text{pos}+k)$ 可以表示为 $\text{PE}(\text{pos})$ 的线性函数，模型可以通过学习这个线性变换来理解相对位置。

**追问：正弦位置编码的优缺点？**

| 优点 | 缺点 |
|------|------|
| 可泛化到训练时未见过的长度 | 位置信息与内容相加，可能干扰 |
| 不同维度捕获不同粒度的位置 | 没有相对位置的直接建模 |
| 无需学习，不增加参数 | 长序列时高频分量区分度降低 |

---

### Q8: RoPE（旋转位置编码）的原理是什么？

**基础回答：**

RoPE 通过旋转向量的方式编码位置，将位置信息融入到注意力计算中。

**深入回答：**

**核心思想**：

对于二维向量，位置 $m$ 的旋转编码：

$
\begin{aligned}
\text{旋转矩阵:} \quad R(m, \theta) &= \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \\
\text{编码后:} \quad \tilde{q}_m &= R(m, \theta)q_m \\
\tilde{k}_n &= R(n, \theta)k_n
\end{aligned}
$

**关键性质**：注意力分数只依赖相对位置

$
\tilde{q}_m \cdot \tilde{k}_n = (R(m, \theta)q) \cdot (R(n, \theta)k) = q \cdot R(n-m, \theta)k \quad \text{只与 } n-m \text{ 有关}
$

**追问：RoPE 相比正弦编码的优势？**

| 特性 | 正弦编码 | RoPE |
|------|----------|------|
| 编码方式 | 加法 | 乘法（旋转） |
| 相对位置 | 隐式 | 显式 |
| 与注意力结合 | 独立 | 直接融入 |
| 长度外推 | 一般 | 更好 |

**RoPE 位置编码图示**：

![RoPE 位置编码](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91RoPE%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png)

**追问：LLaMA 中 RoPE 如何实现？**

```python
def apply_rotary_emb(x, cos, sin):
    # x: (batch, seq_len, n_heads, head_dim)
    # 将 head_dim 分成 head_dim//2 对，每对进行二维旋转
    
    x1, x2 = x[..., ::2], x[..., 1::2]  # 分成奇偶
    
    # 旋转
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rotated
```

---

### Q9: ALiBi 位置编码有什么特点？

**基础回答：**

ALiBi（Attention with Linear Biases）通过在注意力分数上添加线性偏置来编码位置。

**深入回答：**

**计算方式**：

$
\begin{aligned}
\text{标准 Attention:} \quad &\text{attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \\
\text{ALiBi:} \quad &\text{attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + m \cdot (-|i-j|)\right)V
\end{aligned}
$

其中 $m$ 是每个头不同的斜率。

**特点**：

1. **无需位置编码层**：直接在注意力计算中添加偏置
2. **线性衰减**：距离越远，惩罚越大
3. **长度外推能力强**：训练短序列，推理长序列

**追问：ALiBi 为什么外推能力强？**

正弦/RoPE 编码在训练长度外会"超出分布"，而 ALiBi：
- 不学习位置表示
- 直接使用相对距离的线性函数
- 推理时无论多长，都按同样规则计算

---

## 四、Transformer 架构

### Q10: Transformer Encoder 和 Decoder 的区别？

**基础回答：**

Encoder 使用双向自注意力，适合理解任务；Decoder 使用单向（masked）自注意力，适合生成任务。

**深入回答：**

**结构对比**：

```
Encoder 层:
┌────────────────────────────────┐
│  Multi-Head Self-Attention     │ ← 双向，可以看到整个序列
│  Add & Norm                    │
├────────────────────────────────┤
│  Feed Forward Network          │
│  Add & Norm                    │
└────────────────────────────────┘

Decoder 层:
┌────────────────────────────────┐
│  Masked Multi-Head Attention   │ ← 单向，只能看到当前位置之前
│  Add & Norm                    │
├────────────────────────────────┤
│  Cross-Attention               │ ← 关注 Encoder 输出
│  Add & Norm                    │
├────────────────────────────────┤
│  Feed Forward Network          │
│  Add & Norm                    │
└────────────────────────────────┘
```

**追问：为什么 GPT 只用 Decoder？**

GPT 是生成模型，只需要单向注意力：
- 训练：预测下一个 token
- 推理：自回归生成
- 不需要 Encoder（没有输入编码需求）

**追问：为什么 BERT 只用 Encoder？**

BERT 是理解模型，需要双向注意力：
- 任务：文本分类、NER、问答等
- 需要"看到"整个句子
- 不需要生成能力

---

### Q11: Pre-Norm 和 Post-Norm 的区别？

**基础回答：**

Post-Norm 是原始 Transformer 的设计（残差后归一化），Pre-Norm 是先归一化再进入子层（现代 LLM 常用）。

**深入回答：**

**结构对比**：

**Pre-Norm vs Post-Norm 图示**：

![Pre-Norm 与 Post-Norm](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91Pre-norm%E4%B8%8EPost-norm.png)

```python
# Post-Norm (原始 Transformer)
x = x + Sublayer(x)
x = LayerNorm(x)

# Pre-Norm (现代 LLM)
x = x + Sublayer(LayerNorm(x))
```

**追问：为什么现代 LLM 使用 Pre-Norm？**

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 训练稳定性 | 需要 warmup | 更稳定 |
| 梯度流动 | 经过 LayerNorm | 有直接路径 |
| 深层网络 | 难训练 | 更容易 |
| 最终输出 | 已经归一化 | 需要额外 LayerNorm |

**梯度分析**：

```python
# Pre-Norm 的梯度路径
∂L/∂x₁ = ∂L/∂xₙ  # 直连路径，梯度无损

# Post-Norm 的梯度路径
∂L/∂x₁ = ∂L/∂xₙ · ∏ LayerNorm_grad  # 梯度经过多个 LayerNorm
```

Pre-Norm 的直连路径让梯度更容易流向浅层，深层网络训练更稳定。

---

### Q12: Feed Forward Network 的作用是什么？

**基础回答：**

FFN 是两层全连接网络，中间有激活函数，用于增加模型的非线性表达能力。

**深入回答：**

**计算过程**：

$
\begin{aligned}
\text{标准 FFN:} \quad &\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \\
\text{带门控的 FFN (GLU 变体):} \quad &\text{FFN}(x) = (\text{Swish}(xW_1) \odot xV)W_2 \quad \text{(LLaMA 使用)}
\end{aligned}
$

**追问：FFN 在 Transformer 中的作用？**

1. **增加非线性**：Self-Attention 主要是线性操作（除了 softmax）
2. **知识存储**：研究表明 FFN 存储了大量事实知识
3. **升维降维**：通常升维 4 倍，增加表达能力

**追问：为什么 FFN 升维通常是 4 倍？**

$$
\begin{aligned}
\text{输入: } &d_{\text{model}} \\
\text{隐藏层: } &4 \times d_{\text{model}} \\
\text{输出: } &d_{\text{model}} \\
\text{参数量: } &2 \times d_{\text{model}} \times 4d_{\text{model}} = 8d_{\text{model}}^2
\end{aligned}
$$

这是 Transformer 中参数量最大的部分。4 倍是经验值，平衡了：
- 表达能力（越大越好）
- 参数量（越大越多）
- 计算量（越大越慢）

---

### Q13: Transformer 的参数量如何计算？

**基础回答：**

Transformer 的参数主要来自 Embedding、Attention、FFN 和 LayerNorm。

**深入回答：**

**以 GPT-3 175B 为例**（$d_{\text{model}} = 12288$, $n_{\text{layers}} = 96$, vocab_size $= 50257$）：

| 组件 | 计算公式 | 参数量 |
|------|----------|--------|
| Token Embedding | vocab $\times d_{\text{model}}$ | $50257 \times 12288 \approx 617\text{M}$ |
| Attention (每层) | $4 \times d_{\text{model}}^2$ | $4 \times 12288^2 \approx 604\text{M}$ |
| FFN (每层) | $2 \times d_{\text{model}} \times 4d_{\text{model}}$ | $8 \times 12288^2 \approx 1.2\text{B}$ |
| LayerNorm (每层) | $2 \times 2 \times d_{\text{model}}$ | 忽略不计 |
| 总参数 | embedding + layers $\times$ (attn + ffn) | $\approx 175\text{B}$ |

**简化估算公式**：

$$
\text{参数量} \approx 12 \times d_{\text{model}}^2 \times n_{\text{layers}} + \text{vocab} \times d_{\text{model}}
$$

**追问：给定参数量，如何估算模型维度和层数？**

经验法则（LLaMA 风格）：
- 层数 $n \approx \text{参数量}^{1/3} \times \text{常数}$
- $d_{\text{model}} \approx \text{参数量}^{1/3} \times \text{另一常数}$

例如 7B 模型：
- LLaMA 7B: $n_{\text{layers}}=32$, $d_{\text{model}}=4096$
- 估算: $32 \times 12 \times 4096^2 \approx 6.4\text{B}$（加上词表约 7B）

---

## 五、训练与优化

### Q14: Transformer 训练时如何处理变长序列？

**基础回答：**

使用 Padding 填充到统一长度，配合 Attention Mask 忽略 Padding 位置。

**深入回答：**

**处理方法**：

```python
# 1. Padding 到最大长度
sequences = pad_sequences(sequences, max_len, pad_token=0)

# 2. 创建 Attention Mask
attention_mask = (sequences != 0).long()  # 1 表示有效，0 表示 padding

# 3. 在 Attention 计算中应用 mask
scores = Q @ K.T / sqrt(d_k)
scores = scores.masked_fill(attention_mask == 0, -inf)  # padding 位置设为 -inf
attention_weights = softmax(scores)
```

**追问：如何优化变长序列的训练效率？**

1. **动态 Padding**：每个 batch pad 到该 batch 最长序列
2. **打包序列**：将多个短序列拼接，减少 padding
3. **FlashAttention**：优化内存访问，支持变长
4. **序列并行**：将长序列切分到多个 GPU

---

### Q15: 什么是因果掩码（Causal Mask）？

**基础回答：**

因果掩码确保 Decoder 在位置 i 只能看到位置 0 到 i-1 的信息，不能看到未来。

**深入回答：**

**掩码矩阵**：

```python
# 序列长度为 4 的因果掩码
causal_mask = [
    [0, -inf, -inf, -inf],  # 位置 0 只能看到自己
    [0,   0, -inf, -inf],   # 位置 1 能看到 0, 1
    [0,   0,   0, -inf],    # 位置 2 能看到 0, 1, 2
    [0,   0,   0,   0]      # 位置 3 能看到所有
]
```

**追问：为什么 GPT 训练时可以并行？**

虽然推理是自回归的，但训练时：
- 使用因果掩码，每个位置只能看到之前的信息
- 一次前向传播计算所有位置的预测
- 损失是所有位置损失的总和

这比 RNN 的串行训练高效得多。

---

### Q16: Transformer 推理如何加速？

**基础回答：**

常用方法包括 KV Cache、推测解码、量化等。

**深入回答：**

**KV Cache**：

```python
# 无 Cache：每次重新计算所有 K, V
for t in range(max_len):
    # 重新计算所有位置的 K, V
    K, V = compute_kv(tokens[:t+1])
    output = attention(Q_t, K, V)

# 有 Cache：只计算新位置的 K, V
cache = []
for t in range(max_len):
    # 只计算当前位置的 K, V
    K_t, V_t = compute_kv(tokens[t:t+1])
    cache.append((K_t, V_t))
    K, V = concat(cache)
    output = attention(Q_t, K, V)
```

**追问：KV Cache 的内存占用如何计算？**

$$
\begin{aligned}
\text{KV Cache 大小} = 2 \times n_{\text{layers}} \times \text{batch\_size} \times \text{seq\_len} \times n_{\text{heads}} \times \text{head\_dim} \times \text{bytes} \\
\text{例如 LLaMA 7B (batch=1, seq=2048):} \\
= 2 \times 32 \times 1 \times 2048 \times 32 \times 128 \times 2 \text{ (FP16)} = 1 \text{ GB}
\end{aligned}
$$

长序列时，KV Cache 是主要的内存瓶颈。

---

## 六、高级问题

### Q17: FlashAttention 的原理是什么？

**基础回答：**

FlashAttention 通过分块计算和内存重排，避免存储完整的注意力矩阵，显著减少显存占用。

**深入回答：**

**标准 Attention 的问题**：

需要 $O(n^2)$ 的显存存储注意力矩阵：
$
\begin{aligned}
\text{scores} &= Q @ K^T \quad (n, n) \\
\text{attention\_weights} &= \text{softmax}(\text{scores}) \quad (n, n) \text{ 需要存储} \\
\text{output} &= \text{attention\_weights} @ V
\end{aligned}
$

**FlashAttention 的优化**：

1. **分块计算**：将 Q、K、V 分成小块，逐块计算
2. **在线 Softmax**：使用数值稳定的增量式 softmax
3. **不存储中间结果**：只保留最终输出

```python
# FlashAttention 伪代码
for block_q in Q:
    output_block = 0
    for block_k, block_v in zip(K, V):
        # 计算 block 内的注意力
        scores_block = block_q @ block_k.T
        # 在线 softmax 更新
        output_block = online_softmax_update(output_block, scores_block, block_v)
    yield output_block
```

**效果**：
- 显存从 $O(n^2)$ 降到 $O(n)$
- 由于更好的内存局部性，速度也更快

---

### Q18: 什么是稀疏注意力？为什么需要它？

**基础回答：**

稀疏注意力减少每个位置关注的 token 数量，从 O(n²) 降低计算复杂度。

**深入回答：**

**几种稀疏模式**：

```
全注意力 (O(n²)):
████████████
████████████
████████████
████████████

局部注意力 (O(n)):
█░░░░░░░░░░░
███░░░░░░░░░
░████░░░░░░░
░░██████░░░░

稀疏注意力组合:
- 局部窗口 + 扩张 + 随机
- BigBird, Longformer, Sparse Transformer
```

**追问：为什么稀疏注意力没有在主流 LLM 中普及？**

1. **实现复杂**：需要特殊的 CUDA kernel
2. **硬件不友好**：稀疏操作难以高效并行
3. **长上下文解决方案**：RoPE 外推、线性 Attention 等替代方案
4. **FlashAttention**：大大缓解了长序列的显存问题

---

### Q19: Transformer 和 RNN 的本质区别？

**基础回答：**

Transformer 使用 Attention 并行处理整个序列，RNN 按顺序处理，逐个 token 更新状态。

**深入回答：**

| 特性 | RNN | Transformer |
|------|-----|-------------|
| **并行性** | 串行，无法并行 | 完全并行 |
| **长距离依赖** | 困难，信息逐层传递 | 直接，Attention 直连 |
| **计算复杂度** | $O(n)$ | $O(n^2)$ |
| **位置感知** | 隐式（顺序处理） | 显式（位置编码） |
| **训练效率** | 低（无法并行） | 高（可并行） |

**追问：为什么 Transformer 比 RNN 更适合 GPU？**

1. **矩阵运算**：Attention 是大规模矩阵乘法，GPU 优化好
2. **并行度**：所有位置同时计算，充分利用 GPU 并行能力
3. **内存访问**：连续内存访问模式，cache 命中率高

---

### Q20: 为什么 Transformer 能取代 RNN？

**基础回答：**

Transformer 在并行性、长距离建模能力和训练效率上都优于 RNN。

**深入回答：**

**关键突破**：

1. **并行训练**：RNN 必须逐 token 处理，Transformer 一次处理整个序列
2. **长距离依赖**：RNN 信息需要经过多个时间步传递，Transformer 直接连接
3. **梯度流动**：RNN 容易梯度消失，Transformer 有直连路径
4. **可扩展性**：Transformer 容易扩展到更大规模

**追问：RNN 还有什么应用场景？**

- 实时流处理（必须逐 token 处理）
- 资源受限设备（参数量小）
- 在线学习（无需存储完整序列）
- 某些时间序列预测任务

---

## 📝 总结

### 核心知识点

| 主题 | 核心要点 |
|------|----------|
| **Self-Attention** | QKV 计算、缩放因子、复杂度分析 |
| **多头注意力** | 子空间表示、MHA/MQA/GQA 对比 |
| **位置编码** | 正弦编码、RoPE、ALiBi 原理与对比 |
| **Encoder-Decoder** | 结构差异、适用场景、GPT/BERT 设计选择 |
| **Pre-Norm vs Post-Norm** | 训练稳定性、梯度流动 |
| **KV Cache** | 推理加速、内存占用计算 |
| **FlashAttention** | 分块计算、内存优化 |

### 常见追问方向

1. **公式推导**：能写出完整的 Attention 计算过程
2. **设计动机**：为什么这样设计？解决了什么问题？
3. **效率分析**：时间/空间复杂度、优化方法
4. **工程实践**：实际训练中的技巧和坑
5. **对比分析**：不同方法的优劣和适用场景

---

*[下一章：大模型核心原理面试题 →](./llm-advanced.md)*
