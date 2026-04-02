# 世界模型架构设计

本章深入探讨世界模型的核心架构组件，包括 VAE、RSSM、JEPA 和 Transformer-based 设计。

---

## 架构概览

世界模型的典型架构包含以下组件：

```
┌─────────────────────────────────────────────────────────────────┐
│                        世界模型架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   观测 o_t ──→ [编码器] ──→ 潜在状态 z_t ──┐                    │
│                                            │                    │
│   动作 a_t ──────────────────────────────→│                    │
│                                            ↓                    │
│                                   [转移模型] ──→ z_{t+1}        │
│                                            │                    │
│                                            ↓                    │
│   观测 o_t ←── [解码器] ←── 潜在状态 z_t ←─┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## VAE 架构

### 变分自编码器基础

VAE 是世界模型中常用的状态表示学习方法。

#### 核心思想

将观测编码为潜在分布，通过重参数化采样：

$$
z \sim q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x)^2)
$$

#### ELBO 目标

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

- **重构项**：确保潜在状态保留观测信息
- **KL项**：约束潜在分布接近先验

### VAE 在世界模型中的应用

```
观测 o_t ──→ 编码器 Encoder ──→ μ, σ ──→ 重参数化 ──→ z_t
                                                        │
潜在状态 z_t ──→ 解码器 Decoder ──→ 观测重构 ô_t ←───────┘
```

#### 编码器设计

```python
class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

#### 解码器设计

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.net(z)
```

---

## RSSM：循环状态空间模型

RSSM (Recurrent State-Space Model) 是 Dreamer 系列的核心架构，由 Danijar Hafner 等人提出。

### 核心设计

RSSM 结合了确定性状态和随机性状态：

$$
h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) \quad \text{(确定性)}
$$
$$
s_t \sim q(s_t | h_t, o_t) \quad \text{(随机性)}
$$

### 架构图

```
                        ┌──────────────────────────────────┐
                        │           RSSM 单步更新           │
                        └──────────────────────────────────┘

     a_{t-1}            h_{t-1}          s_{t-1}         o_t
        │                  │                │             │
        └────────┬─────────┘                │             │
                 │                          │             │
                 ↓                          │             │
        ┌────────────────┐                  │             │
        │   GRU/RNN      │◄─────────────────┘             │
        └────────────────┘                                │
                 │                                        │
                 ↓ h_t                                    │
        ┌────────────────┐                                │
        │   混合密度网络   │◄───────────────────────────────┘
        └────────────────┘
                 │
                 ↓ s_t (采样)
```

### RSSM 实现要点

```python
class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim):
        super().__init__()
        # 确定性状态转移
        self.gru = nn.GRUCell(hidden_dim + action_dim, hidden_dim)
        # 随机性状态参数
        self.fc_prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.fc_posterior = nn.Linear(hidden_dim + latent_dim, 2 * latent_dim)
    
    def forward(self, h_prev, s_prev, a_prev, z_obs=None):
        # 确定性更新
        h = self.gru(torch.cat([h_prev, s_prev, a_prev], dim=-1))
        
        # 先验分布
        prior_mu, prior_logvar = self.fc_prior(h).chunk(2, dim=-1)
        
        if z_obs is not None:
            # 后验分布（使用观测）
            posterior_mu, posterior_logvar = self.fc_posterior(
                torch.cat([h, z_obs], dim=-1)
            ).chunk(2, dim=-1)
            return h, posterior_mu, posterior_logvar
        else:
            return h, prior_mu, prior_logvar
```

### 确定性 vs 随机性状态

| 类型 | 特点 | 作用 |
|------|------|------|
| 确定性状态 $h$ | 完全由历史决定 | 保持长期记忆 |
| 随机性状态 $s$ | 服从概率分布 | 建模不确定性 |

---

## JEPA：联合嵌入预测架构

JEPA (Joint Embedding Predictive Architecture) 是 Yann LeCun 提出的架构，强调在表示空间而非像素空间预测。

### 核心思想

```
传统方法：预测下一个观测 (像素空间)
JEPA：预测下一个状态的表示 (表示空间)
```

### 架构设计

```
x_t ──→ [编码器 E] ──→ z_t ──┐
                            │
                            ├──→ [预测器 P] ──→ ŷ_{t+1}
                            │
x_{t+1} ──→ [编码器 E] ──→ z_{t+1} ──→ 作为监督信号
```

### 与 VAE 的对比

| 方面 | VAE | JEPA |
|------|-----|------|
| 预测空间 | 像素空间 | 表示空间 |
| 解码器 | 需要 | 不需要 |
| 计算开销 | 高（重构像素） | 低 |
| 信息瓶颈 | KL约束 | 预测损失 |

### JEPA 的优势

1. **避免像素级重构**：不浪费能力建模无关细节
2. **更好的抽象**：表示空间预测更关注语义
3. **自监督学习**：不需要标签，可大规模预训练

---

## Transformer 世界模型

### 概述

Transformer 架构也可用于构建世界模型，代表工作包括 IRIS、Trajectory Transformer 等。

### IRIS 架构

IRIS (Implicit Reward without Intermediate Supervision) 使用 Transformer 作为世界模型：

```
┌─────────────────────────────────────────────────────────────┐
│                    IRIS 架构                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  观测序列 o_1, o_2, ..., o_t                                │
│       ↓                                                     │
│  [VAE 编码器] → token 序列 z_1, z_2, ..., z_t               │
│       ↓                                                     │
│  [GPT-like Transformer] → 预测下一个 token                   │
│       ↓                                                     │
│  [VAE 解码器] → 重构观测                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Transformer vs RSSM

| 方面 | RSSM | Transformer |
|------|------|-------------|
| 序列建模 | RNN | 自注意力 |
| 并行性 | 串行 | 并行 |
| 长期依赖 | 可能遗忘 | 更好 |
| 计算复杂度 | O(T) | O(T²) |

### 实现

```python
class TransformerWorldModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(1000, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens):
        # tokens: [seq_len, batch_size]
        seq_len = tokens.size(0)
        
        # 嵌入
        tok_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=tokens.device))
        x = tok_emb + pos_emb
        
        # 因果注意力
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        x = self.transformer(x, x, tgt_mask=mask)
        
        # 预测下一个token
        logits = self.output(x)
        return logits
```

---

## 架构选择指南

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 视频游戏 | RSSM | 处理连续观测，想象规划 |
| 文本决策 | Transformer | 利用语言建模优势 |
| 机器人控制 | RSSM + VAE | 鲁棒的状态表示 |
| 大规模预训练 | JEPA | 不需要像素重构 |

---

## 小结

本章介绍了世界模型的核心架构：

1. **VAE**：提供状态表示学习的基础
2. **RSSM**：结合确定性和随机性状态，是 Dreamer 的核心
3. **JEPA**：在表示空间预测，避免像素级重构
4. **Transformer**：强大的序列建模能力

下一章我们将详细解析 Dreamer 系列模型。

---

## 参考资料

- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122)
- [Dream to Control (Dreamer v1, 2020)](https://arxiv.org/abs/1912.01603)
- [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [IRIS (2022)](https://arxiv.org/abs/2210.14301)
