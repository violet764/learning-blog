# 多模态生成模型

多模态生成模型（Multimodal Generation Models）是一类能够基于某种模态的输入（如文本、图像、音频）生成另一种模态内容（如图像、视频、音频）的深度学习模型。这些模型在创意设计、内容创作、虚拟现实等领域具有广泛应用。

## 章节概述

本章将系统介绍多模态生成模型的核心技术和代表性模型：

| 主题 | 核心技术 | 代表模型 | 应用场景 |
|------|----------|----------|----------|
| 文生图 | 扩散模型、Transformer | DALL-E、SDXL、Midjourney | 图像创作、设计 |
| 视频生成 | 时空扩散、DiT | Sora、Runway Gen-2 | 视频制作、动画 |
| 音频生成 | 扩散模型、Flow | AudioLDM、MusicGen | 音乐创作、音效 |
| 多模态编辑 | InstructPix2Pix | 图像编辑、内容修改 | 图像编辑、内容修改 |

---

## 多模态生成概述

### 什么是多模态生成

多模态生成指模型能够**跨越模态边界**进行内容创作：

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态生成任务谱系                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入模态              生成模态              任务名称        │
│   ┌──────┐             ┌──────┐                             │
│   │ 文本 │ ──────────▶ │ 图像 │      Text-to-Image          │
│   └──────┘             └──────┘                             │
│   ┌──────┐             ┌──────┐                             │
│   │ 文本 │ ──────────▶ │ 视频 │      Text-to-Video          │
│   └──────┘             └──────┘                             │
│   ┌──────┐             ┌──────┐                             │
│   │ 文本 │ ──────────▶ │ 音频 │      Text-to-Audio          │
│   └──────┘             └──────┘                             │
│   ┌──────┐             ┌──────┐                             │
│   │ 图像 │ ──────────▶ │ 文本 │      Image-to-Text (Caption)│
│   └──────┘             └──────┘                             │
│   ┌──────┐             ┌──────┐                             │
│   │ 图像 │ ──────────▶ │ 图像 │      Image-to-Image (编辑)  │
│   └──────┘             └──────┘                             │
│   ┌──────┐             ┌──────┐                             │
│   │ 图像 │ ──────────▶ │ 视频 │      Image-to-Video         │
│   └──────┘             └──────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心技术路线

多模态生成经历了多种技术范式的演进：

```
技术演进时间线：

2020 ─────────────────────────────────────────────────────▶
     │
     │  GAN时代：StyleGAN、BigGAN
     │  - 生成质量有限
     │  - 训练不稳定
     │  - 模式崩溃问题
     │
2021 ─────────────────────────────────────────────────────▶
     │
     │  VQ-VAE + Transformer：DALL-E、CogView
     │  - 两阶段生成：先编码后生成
     │  - 自回归方式生成
     │  - 生成速度较慢
     │
2022 ─────────────────────────────────────────────────────▶
     │
     │  扩散模型崛起：DALL-E 2、Stable Diffusion
     │  - 渐进去噪过程
     │  - 高质量生成
     │  - 可控性强
     │
2023-2024 ────────────────────────────────────────────────▶
     │
     │  Latent Diffusion + DiT：SDXL、Sora、DALL-E 3
     │  - 潜空间扩散，高效生成
     │  - Transformer架构增强
     │  - 多模态统一建模
     │
```

### 生成质量评估指标

评估多模态生成模型的质量需要多维度指标：

| 指标 | 全称 | 评估维度 | 适用场景 |
|------|------|----------|----------|
| FID | Fréchet Inception Distance | 生成质量与多样性 | 图像生成 |
| CLIP Score | CLIP Similarity Score | 文本-图像一致性 | 文生图 |
| IS | Inception Score | 生成质量 | 图像生成 |
| LPIPS | Learned Perceptual Image Patch Similarity | 感知相似度 | 图像编辑 |
| Precision & Recall | - | 生成质量与覆盖率 | 图像生成 |

**FID计算公式**：

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

其中 $\mu_r, \Sigma_r$ 是真实图像特征的均值和协方差，$\mu_g, \Sigma_g$ 是生成图像特征的统计量。FID越低越好。

**CLIP Score**：

$$
\text{CLIP Score} = \cos(f_I(I), f_T(T)) = \frac{f_I(I) \cdot f_T(T)}{\|f_I(I)\| \|f_T(T)\|}
$$

---

## DALL-E 系列

### DALL-E：离散变分自编码器 + Transformer

DALL-E（2021）是OpenAI提出的首个大规模文生图模型，采用**两阶段生成**策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    DALL-E 架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段一：dVAE（离散变分自编码器）                              │
│                                                             │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│   │  图像    │      │  Encoder │      │ Codebook │         │
│   │  x ∈ ℝ   │ ───▶ │  q(z|x)  │ ───▶ │  Z = {z} │         │
│   │  H×W×3   │      │          │      │  K × D   │         │
│   └──────────┘      └──────────┘      └────┬─────┘         │
│                                            │                │
│                                            ▼                │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│   │  重建    │ ◀─── │  Decoder │ ◀─── │ 离散编码  │         │
│   │  图像    │      │  p(x|z)  │      │ 索引序列  │         │
│   └──────────┘      └──────────┘      └──────────┘         │
│                                                             │
│  阶段二：Transformer自回归生成                                │
│                                                             │
│   ┌──────────┐      ┌──────────────────────────────┐       │
│   │  文本    │      │      Transformer             │       │
│   │  Token   │ ───▶ │   自回归生成图像token序列     │       │
│   │  序列    │      │                              │       │
│   └──────────┘      └──────────────┬───────────────┘       │
│                                     │                       │
│                                     ▼                       │
│                              ┌──────────┐                  │
│                              │  图像    │                  │
│                              │  Token   │                  │
│                              │  序列    │                  │
│                              └──────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**dVAE损失函数**：

$$
\mathcal{L}_{\text{dVAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{\text{KL}}(q(z|x) \| p(z))
$$

由于 $z$ 是离散的，使用**Gumbel-Softmax**松弛进行端到端训练：

$$
y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_{j=1}^{K} \exp((\log \pi_j + g_j) / \tau)}
$$

其中 $g_i \sim \text{Gumbel}(0, 1)$，$\tau$ 是温度参数。

### DALL-E 2：扩散模型 + CLIP先验

DALL-E 2（2022）引入扩散模型，显著提升生成质量：

```
┌─────────────────────────────────────────────────────────────┐
│                    DALL-E 2 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入文本                                                   │
│      │                                                      │
│      ▼                                                      │
│   ┌──────────────────────────────────────────────────┐     │
│   │              CLIP Text Encoder                   │     │
│   │              输出: 文本嵌入 t                      │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Prior Network                        │     │
│   │         从文本嵌入预测图像嵌入                      │     │
│   │              t → i (扩散模型)                      │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Decoder (扩散模型)                   │     │
│   │         从图像嵌入生成图像                         │     │
│   │              i → image                            │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Upsampler (超分辨率)                 │     │
│   │         64×64 → 256×256 → 1024×1024              │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Prior训练目标**：

Prior学习从文本嵌入 $t$ 到图像嵌入 $i$ 的映射，使用扩散模型：

$$
\mathcal{L}_{\text{prior}} = \mathbb{E}_{t, i, \epsilon \sim \mathcal{N}(0,1)} \left[ \| \epsilon - \epsilon_\theta(i_t, t) \|^2 \right]
$$

**解码器训练目标**：

$$
\mathcal{L}_{\text{decoder}} = \mathbb{E}_{x, i, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, i) \|^2 \right]
$$

### DALL-E 3：高质量标题重写

DALL-E 3（2023）的核心创新在于**训练数据质量提升**：

```
┌─────────────────────────────────────────────────────────────┐
│                    DALL-E 3 创新点                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题：原始数据集中的标题简短、不完整                          │
│                                                             │
│  解决方案：使用模型重写标题                                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  原始标题：                                          │   │
│  │  "一只猫"                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  重写后标题：                                        │   │
│  │  "一只毛茸茸的橘猫慵懒地躺在阳光照射的窗台上，        │   │
│  │   背景是模糊的城市天际线，温暖的午后光线，            │   │
│  │   摄影风格，高清细节"                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  效果：                                                     │
│  ✅ 更精确的文本-图像对应                                     │
│  ✅ 更好的复杂场景生成                                        │
│  ✅ 减少幻觉和错误                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Stable Diffusion XL

### Latent Diffusion Models 原理

Stable Diffusion XL (SDXL) 基于**潜空间扩散模型**，通过在低维潜空间进行扩散，大幅提升生成效率：

```
┌─────────────────────────────────────────────────────────────┐
│                Latent Diffusion 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   高分辨率图像 (512×512×3)                                   │
│         │                                                   │
│         ▼                                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │              VAE Encoder                          │     │
│   │         下采样 8× → 潜空间表示                     │     │
│   │         z ∈ ℝ^(64×64×4)                          │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Diffusion Process                    │     │
│   │              在潜空间进行扩散                       │     │
│   │         z_0 → z_T (加噪)                          │     │
│   │         z_T → z_0 (去噪)                          │     │
│   │                                                   │     │
│   │   条件输入: 文本嵌入 (CLIP)、时间步 t              │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              VAE Decoder                          │     │
│   │         上采样 8× → 重建图像                       │     │
│   │         x̂ ∈ ℝ^(512×512×3)                        │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
│   计算优势：潜空间维度仅为像素空间的 1/48                      │
│   - 像素空间: 512×512×3 = 786,432                          │
│   - 潜空间: 64×64×4 = 16,384                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 扩散过程数学描述

**前向扩散过程**（加噪）：

$$
q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) I)
$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，$\alpha_t = 1 - \beta_t$。

可以直接采样：

$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**反向去噪过程**：

$$
p_\theta(z_{t-1} | z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))
$$

均值参数化：

$$
\mu_\theta(z_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(z_t, t) \right)
$$

**训练目标**（简化版）：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{z_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]
$$

其中 $c$ 是条件信息（如文本嵌入）。

### SDXL架构改进

SDXL相比Stable Diffusion有以下改进：

```
┌─────────────────────────────────────────────────────────────┐
│                    SDXL 改进点                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 更大的U-Net骨干网络                                       │
│     - 参数量: 865M → 2.6B                                   │
│     - 更多注意力层和通道数                                    │
│                                                             │
│  2. 双文本编码器                                             │
│     ┌──────────────────────────────────────────────────┐   │
│     │  CLIP ViT-L + CLIP ViT-G                         │   │
│     │  拼接两种文本嵌入，增强语义理解                     │   │
│     └──────────────────────────────────────────────────┘   │
│                                                             │
│  3. 原生支持更高分辨率                                       │
│     - 训练分辨率: 1024×1024                                 │
│     - 无需额外超分辨率模块                                   │
│                                                             │
│  4. 条件注入改进                                             │
│     - OpenCLIP 全局嵌入                                     │
│     - CLIP 局部特征 (池化后)                                 │
│     - 原始分辨率和裁剪坐标                                   │
│                                                             │
│  5. Refiner模型                                             │
│     - 专门用于高分辨率细节增强                                │
│     - 在降噪后期阶段运行                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch实现示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class UNetConfig:
    """U-Net配置"""
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_heads: int = 8
    context_dim: int = 768  # CLIP嵌入维度


class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # 线性投影
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch] 时间步
        Returns:
            [batch, dim] 时间步嵌入
        """
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class CrossAttention(nn.Module):
    """交叉注意力层"""
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, query_dim] 查询特征
            context: [batch, context_len, context_dim] 上下文特征
        
        Returns:
            [batch, seq_len, query_dim]
        """
        if context is None:
            context = x
        
        # QKV投影
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 多头注意力
        batch_size = x.size(0)
        q = q.view(batch_size, -1, self.heads, x.size(-1) // self.heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, x.size(-1) // self.heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, x.size(-1) // self.heads).transpose(1, 2)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, x.size(-1))
        
        return self.to_out(out)


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 时间步嵌入投影
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        temb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, h, w]
            temb: [batch, temb_channels]
        """
        # 第一个卷积
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间步嵌入
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        # 第二个卷积
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """简化版U-Net用于扩散模型"""
    
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        
        # 时间步嵌入
        self.time_embed = TimestepEmbedding(config.model_channels)
        
        # 输入投影
        self.input_proj = nn.Conv2d(
            config.in_channels, 
            config.model_channels, 
            3, 
            padding=1
        )
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = [config.model_channels]
        ch = config.model_channels
        
        for mult in config.channel_mult:
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(ResBlock(ch, ch * mult, config.model_channels))
                ch = ch * mult
                channels.append(ch)
            if mult != config.channel_mult[-1]:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                channels.append(ch)
        
        # 中间块
        self.mid_block = nn.Sequential(
            ResBlock(ch, ch, config.model_channels),
            CrossAttention(ch, config.context_dim, config.num_heads),
            ResBlock(ch, ch, config.model_channels)
        )
        
        # 上采样路径（简化）
        self.up_blocks = nn.ModuleList()
        # ... 省略详细实现
        
        # 输出
        self.out = nn.Sequential(
            nn.GroupNorm(32, config.model_channels),
            nn.SiLU(),
            nn.Conv2d(config.model_channels, config.out_channels, 3, padding=1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, h, w] 噪声潜表示
            t: [batch] 时间步
            context: [batch, seq_len, context_dim] 文本条件
        
        Returns:
            [batch, out_channels, h, w] 预测的噪声
        """
        # 时间步嵌入
        temb = self.time_embed(t)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 下采样
        skips = [h]
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)
            skips.append(h)
        
        # 中间处理
        for layer in self.mid_block:
            if isinstance(layer, ResBlock):
                h = layer(h, temb)
            elif isinstance(layer, CrossAttention):
                h = h.flatten(2).transpose(1, 2)  # [B, H*W, C]
                h = layer(h, context)
                h = h.transpose(1, 2).reshape(h.size(0), -1, h.size(1), h.size(2))
        
        # 上采样（简化）
        # ...
        
        return self.out(h)


class GaussianDiffusion:
    """高斯扩散过程"""
    
    def __init__(
        self, 
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012
    ):
        self.num_timesteps = num_timesteps
        
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def q_sample(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散：从x_0采样x_t
        
        Args:
            x_0: [batch, ...] 原始数据
            t: [batch] 时间步
            noise: 可选的噪声
        
        Returns:
            x_t: 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 获取对应时间步的系数
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(
        self, 
        model: nn.Module,
        x_0: torch.Tensor, 
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            model: 去噪模型
            x_0: 原始数据
            t: 时间步
            context: 条件信息
            noise: 噪声
        
        Returns:
            loss: MSE损失
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 前向扩散
        x_t = self.q_sample(x_0, t, noise)
        
        # 预测噪声
        noise_pred = model(x_t, t, context)
        
        # MSE损失
        return F.mse_loss(noise_pred, noise)
    
    def p_sample(
        self, 
        model: nn.Module,
        x_t: torch.Tensor, 
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        反向采样一步
        
        Args:
            model: 去噪模型
            x_t: 当前时刻的噪声数据
            t: 时间步
            context: 条件信息
        
        Returns:
            x_{t-1}: 去噪一步后的数据
        """
        # 预测噪声
        noise_pred = model(x_t, t, context)
        
        # 获取系数
        alpha = self._extract(self.alphas, t, x_t.shape)
        alpha_cumprod = self._extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # 计算均值
        mean = (1 / torch.sqrt(alpha)) * (x_t - (1 - alpha) / sqrt_one_minus_alpha * noise_pred)
        
        # 添加噪声（t > 0时）
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            beta = self._extract(self.betas, t, x_t.shape)
            mean = mean + torch.sqrt(beta) * noise
        
        return mean
    
    @torch.no_grad()
    def p_sample_loop(
        self, 
        model: nn.Module,
        shape: tuple,
        context: Optional[torch.Tensor] = None,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        完整采样过程
        
        Args:
            model: 去噪模型
            shape: 输出形状
            context: 条件信息
            return_all_timesteps: 是否返回所有时刻的结果
        
        Returns:
            生成的样本
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if return_all_timesteps:
            xs = [x]
        
        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, context)
            
            if return_all_timesteps:
                xs.append(x)
        
        if return_all_timesteps:
            return torch.stack(xs)
        return x
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """从a中提取t对应位置的值并reshape"""
        b = a.to(t.device)[t]
        return b.view(b.size(0), *([1] * (len(shape) - 1)))


# 使用示例
if __name__ == "__main__":
    # 配置
    config = UNetConfig()
    
    # 创建模型
    model = SimpleUNet(config)
    diffusion = GaussianDiffusion(num_timesteps=1000)
    
    # 模拟训练
    batch_size = 4
    x_0 = torch.randn(batch_size, 4, 64, 64)  # 潜空间表示
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 768)  # 文本嵌入
    
    # 计算损失
    loss = diffusion.p_losses(model, x_0, t, context)
    print(f"训练损失: {loss.item():.4f}")
    
    # 采样生成
    # samples = diffusion.p_sample_loop(model, (1, 4, 64, 64), context[:1])
    # print(f"生成样本形状: {samples.shape}")
```

---

## Midjourney 原理

### 架构推测

虽然Midjourney未开源，但根据公开信息和技术分析，可以推测其核心技术：

```
┌─────────────────────────────────────────────────────────────┐
│                 Midjourney 技术架构（推测）                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心组件：                                                  │
│                                                             │
│  1. 高质量美学数据集                                         │
│     - 精心筛选的艺术作品                                     │
│     - 专业摄影师作品                                         │
│     - 高审美评分数据                                         │
│                                                             │
│  2. 改进的扩散模型架构                                       │
│     - 可能基于Latent Diffusion                              │
│     - 自定义的注意力机制                                     │
│     - 风格化的U-Net骨干                                      │
│                                                             │
│  3. 美学评分引导                                             │
│     ┌──────────────────────────────────────────────────┐   │
│     │  Classifier-Free Guidance + Aesthetic Guidance   │   │
│     │                                                   │   │
│     │  去噪方向 = ε_uncond + s_cfg × (ε_cond - ε_uncond)│   │
│     │           + s_aes × ∇_x aesthetic_score(x)       │   │
│     └──────────────────────────────────────────────────┘   │
│                                                             │
│  4. 高级提示词理解                                           │
│     - 参数解析（--ar, --style, --q等）                      │
│     - 风格嵌入                                               │
│     - 艺术家风格迁移                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Classifier-Free Guidance (CFG)

CFG是提升生成质量的关键技术：

$$
\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

其中：
- $\epsilon_\theta(z_t, t, c)$：条件噪声预测
- $\epsilon_\theta(z_t, t, \emptyset)$：无条件噪声预测
- $s$：引导强度（guidance scale）

**引导强度的效果**：

| CFG Scale | 效果 |
|-----------|------|
| 1.0 | 无引导，多样性高但质量低 |
| 7-8 | 平衡质量和多样性（推荐） |
| 15-20 | 强引导，更忠实于提示词但可能过度饱和 |
| >30 | 可能出现伪影和失真 |

### PyTorch实现CFG

```python
import torch
import torch.nn as nn
from typing import Optional, Callable

def classifier_free_guidance(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    null_context: torch.Tensor,
    guidance_scale: float = 7.5
) -> torch.Tensor:
    """
    Classifier-Free Guidance采样
    
    Args:
        model: 去噪模型
        x_t: 当前噪声状态
        t: 时间步
        context: 文本条件嵌入
        null_context: 无条件嵌入（空文本）
        guidance_scale: 引导强度
    
    Returns:
        引导后的噪声预测
    """
    # 条件预测
    noise_cond = model(x_t, t, context)
    
    # 无条件预测
    noise_uncond = model(x_t, t, null_context)
    
    # CFG组合
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    return noise_guided


class CFGWrapper(nn.Module):
    """CFG包装器，自动处理条件和无条件预测"""
    
    def __init__(
        self, 
        model: nn.Module, 
        guidance_scale: float = 7.5,
        context_dim: int = 768
    ):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        # 可学习的无条件嵌入
        self.null_embedding = nn.Parameter(torch.randn(1, 77, context_dim))
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        context: torch.Tensor,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        前向传播，自动应用CFG
        
        Args:
            x_t: [batch, channels, h, w] 噪声图像
            t: [batch] 时间步
            context: [batch, seq_len, dim] 文本嵌入
            guidance_scale: 可选的引导强度覆盖
        
        Returns:
            引导后的噪声预测
        """
        batch_size = x_t.size(0)
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        
        # 扩展输入以同时计算条件和无条件预测
        x_t_expanded = x_t.repeat(2, 1, 1, 1)
        t_expanded = t.repeat(2)
        
        # 条件和无条件上下文
        null_context = self.null_embedding.expand(batch_size, -1, -1)
        context_expanded = torch.cat([context, null_context], dim=0)
        
        # 模型预测
        noise_pred = self.model(x_t_expanded, t_expanded, context_expanded)
        
        # 分离条件和无条件预测
        noise_cond, noise_uncond = noise_pred.chunk(2, dim=0)
        
        # CFG
        return noise_uncond + scale * (noise_cond - noise_uncond)
```

---

## 视频生成模型

### Sora：时空扩散Transformer

Sora（2024）是OpenAI发布的文生视频模型，采用**DiT（Diffusion Transformer）**架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     Sora 架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心创新：时空Patch表示                                      │
│                                                             │
│   输入视频: [T, H, W, 3]                                     │
│         │                                                   │
│         ▼                                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │          Video Compression Network               │     │
│   │          将视频编码为潜空间表示                     │     │
│   │          [T', H', W', C'] 潜空间视频              │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │          Spatial-Temporal Patching               │     │
│   │          将潜空间视频切分为时空Patch               │     │
│   │                                                   │     │
│   │          ┌───┬───┬───┐                          │     │
│   │          │ t₁│ t₂│ t₃│  时间维度                │     │
│   │          ├───┼───┼───┤                          │     │
│   │          │ ■ │ ■ │ ■ │  每个■是一个时空Patch     │     │
│   │          │ ■ │ ■ │ ■ │  包含空间和时间信息       │     │
│   │          └───┴───┴───┘                          │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │          DiT (Diffusion Transformer)             │     │
│   │                                                   │     │
│   │    Patch Embedding → Transformer Blocks → Output │     │
│   │                                                   │     │
│   │    - 空间注意力：建模帧内关系                      │     │
│   │    - 时间注意力：建模帧间关系                      │     │
│   │    - 条件注入：文本、时间步                        │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │          Video Decoder                           │     │
│   │          潜空间视频 → 像素视频                     │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 时空Patch Embedding

将视频切分为时空Patch：

$$
\text{Patch}(V) = \{p_{i,j,k}\}_{i=1,j=1,k=1}^{T_p, H_p, W_p}
$$

其中每个Patch $p_{i,j,k} \in \mathbb{R}^{t \times h \times w \times c}$ 包含局部的时空信息。

**位置编码**：

$$
\text{PE}_{i,j,k} = \text{PE}_{\text{temporal}}(i) + \text{PE}_{\text{spatial}}^{(h)}(j) + \text{PE}_{\text{spatial}}^{(w)}(k)
$$

### DiT架构

DiT使用纯Transformer替代U-Net：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class PatchEmbedding3D(nn.Module):
    """3D时空Patch嵌入"""
    
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: tuple = (2, 16, 16),  # (temporal, height, width)
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time, height, width]
        
        Returns:
            [batch, num_patches, embed_dim]
        """
        # 3D卷积: [B, C, T, H, W] -> [B, embed_dim, T', H', W']
        x = self.proj(x)
        
        # 展平: [B, embed_dim, T', H', W'] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class TemporalAttention(nn.Module):
    """时间注意力：建模帧间关系"""
    
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, dim]
            T, H, W: 时间和空间维度
        
        Returns:
            [batch, num_patches, dim]
        """
        B, N, D = x.shape
        
        # 重塑为 [B, H*W, T, D]（空间位置分组，时间维度做注意力）
        x = x.view(B, T, H * W, D).transpose(1, 2)  # [B, H*W, T, D]
        x = x.reshape(B * H * W, T, D)
        
        # 计算注意力
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * H * W, T, self.heads, D // self.heads).transpose(1, 2), qkv)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B * H * W, T, D)
        
        # 恢复形状
        out = out.view(B, H * W, T, D).transpose(1, 2).reshape(B, N, D)
        
        return self.to_out(out)


class SpatialAttention(nn.Module):
    """空间注意力：建模帧内关系"""
    
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, dim]
        
        Returns:
            [batch, num_patches, dim]
        """
        B, N, D = x.shape
        
        # 重塑为 [B, T, H*W, D]（每帧独立做空间注意力）
        x = x.view(B, T, H * W, D)
        x = x.reshape(B * T, H * W, D)
        
        # 计算注意力
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * T, H * W, self.heads, D // self.heads).transpose(1, 2), qkv)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B * T, H * W, D)
        
        # 恢复形状
        out = out.view(B, T, H * W, D).reshape(B, N, D)
        
        return self.to_out(out)


class DiTBlock(nn.Module):
    """DiT Transformer块"""
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        mlp_ratio: float = 4.0,
        use_temporal: bool = True
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(dim, heads)
        
        self.use_temporal = use_temporal
        if use_temporal:
            self.norm_temporal = nn.LayerNorm(dim)
            self.temporal_attn = TemporalAttention(dim, heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        # 时间步调制（AdaLN）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim if use_temporal else 4 * dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        T: int, 
        H: int, 
        W: int,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, dim]
            T, H, W: 时空维度
            t_emb: [batch, dim] 时间步嵌入
        """
        # AdaLN调制参数
        if self.use_temporal:
            shift_msa, scale_msa, gate_msa, shift_temp, scale_temp, gate_temp = \
                self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation(t_emb).chunk(3, dim=-1)
        
        # 空间注意力
        x_norm = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.spatial_attn(x_norm, T, H, W)
        
        # 时间注意力
        if self.use_temporal:
            x_norm = self.norm_temporal(x) * (1 + scale_temp[:, None, :]) + shift_temp[:, None, :]
            x = x + gate_temp[:, None, :] * self.temporal_attn(x_norm, T, H, W)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class VideoDiT(nn.Module):
    """视频扩散Transformer"""
    
    def __init__(
        self,
        in_channels: int = 4,  # 潜空间通道数
        out_channels: int = 4,
        patch_size: tuple = (1, 2, 2),
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        context_dim: int = 768
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 10000, embed_dim) * 0.02)
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 条件嵌入
        self.context_embed = nn.Linear(context_dim, embed_dim)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, heads, mlp_ratio, use_temporal=True)
            for _ in range(depth)
        ])
        
        # 输出
        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        
        # 初始化
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time, height, width] 噪声视频潜表示
            t: [batch] 时间步
            context: [batch, seq_len, context_dim] 文本条件
        
        Returns:
            [batch, channels, time, height, width] 预测的噪声
        """
        B, C, T, H, W = x.shape
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        num_patches = x.size(1)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :num_patches, :]
        
        # 时间步嵌入
        t_emb = self._timestep_embedding(t)
        t_emb = self.time_embed(t_emb)
        
        # 条件嵌入
        if context is not None:
            t_emb = t_emb + self.context_embed(context.mean(dim=1))
        
        # Transformer块
        T_p = T // self.patch_size[0]
        H_p = H // self.patch_size[1]
        W_p = W // self.patch_size[2]
        
        for block in self.blocks:
            x = block(x, T_p, H_p, W_p, t_emb)
        
        # 输出
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # 重塑为视频形状
        x = x.view(B, T_p, H_p, W_p, -1)
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        return x
    
    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """正弦位置编码用于时间步"""
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = VideoDiT(
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),
        embed_dim=768,
        depth=12,
        heads=12
    )
    
    # 模拟输入
    batch_size = 2
    channels = 4
    time = 16  # 帧数
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, time, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 768)
    
    # 前向传播
    output = model(x, t, context)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

### Runway Gen-2

Runway Gen-2是另一个重要的视频生成模型，其特点：

```
┌─────────────────────────────────────────────────────────────┐
│                 Runway Gen-2 特点                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 多模态输入                                               │
│     - 文本 → 视频                                           │
│     - 图像 + 文本 → 视频                                     │
│     - 视频 + 文本 → 视频（视频编辑）                          │
│                                                             │
│  2. 时序一致性                                               │
│     - 递归生成：逐帧生成，保持帧间连续性                       │
│     - 光流引导：利用运动信息保证平滑过渡                       │
│                                                             │
│  3. 控制能力                                                 │
│     - 运动控制：控制相机运动、物体运动                        │
│     - 风格控制：参考图像风格迁移                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 音频生成

### AudioLDM

AudioLDM将扩散模型应用于音频生成：

```
┌─────────────────────────────────────────────────────────────┐
│                    AudioLDM 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入文本                                                   │
│      │                                                      │
│      ▼                                                      │
│   ┌──────────────────────────────────────────────────┐     │
│   │              CLAP Text Encoder                   │     │
│   │         (音频-文本对比学习模型)                    │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │         Latent Diffusion Model                   │     │
│   │                                                  │     │
│   │   潜空间: Mel频谱图 (Mel-Spectrogram)             │     │
│   │   - 更适合音频处理                                │     │
│   │   - 维度低于原始波形                              │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              VAE Decoder                         │     │
│   │         Mel频谱图 → 音频波形                      │     │
│   │         (HiFi-GAN vocoder)                       │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Mel频谱图**：

音频信号首先转换为Mel频谱图：

$$
\text{Mel}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$

### MusicGen

MusicGen（Meta, 2023）采用**单阶段Transformer**直接生成音频：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTokenizer(nn.Module):
    """音频分词器：将音频编码为离散token"""
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 512,
        num_tokens: int = 2048,
        sample_rate: int = 44100,
        hop_length: int = 512
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, padding=3),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ELU(),
        )
        
        # 量化码本
        self.codebook = nn.Embedding(num_tokens, hidden_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, in_channels, 7, padding=3),
        )
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        编码音频为token
        
        Args:
            audio: [batch, 1, samples]
        
        Returns:
            token_ids: [batch, num_frames]
        """
        # 编码
        z = self.encoder(audio)
        
        # 量化：找最近的码本向量
        z_flat = z.transpose(1, 2)  # [B, T, D]
        distances = torch.cdist(z_flat, self.codebook.weight.unsqueeze(0))
        token_ids = distances.argmin(dim=-1)
        
        return token_ids
    
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        解码token为音频
        
        Args:
            token_ids: [batch, num_frames]
        
        Returns:
            audio: [batch, 1, samples]
        """
        # 查找码本
        z = self.codebook(token_ids)  # [B, T, D]
        z = z.transpose(1, 2)  # [B, D, T]
        
        # 解码
        audio = self.decoder(z)
        
        return audio


class MusicGenTransformer(nn.Module):
    """MusicGen Transformer模型"""
    
    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 24,
        vocab_size: int = 2048,
        max_seq_len: int = 8192
    ):
        super().__init__()
        
        # 音频token嵌入
        self.audio_embed = nn.Embedding(vocab_size, audio_dim)
        
        # 文本条件嵌入
        self.text_proj = nn.Linear(text_dim, audio_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, audio_dim) * 0.02
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=audio_dim,
                nhead=num_heads,
                dim_feedforward=audio_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(audio_dim, vocab_size)
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        自回归生成音频token
        
        Args:
            audio_tokens: [batch, seq_len] 之前的音频token
            text_embedding: [batch, text_len, text_dim] 文本条件
            attention_mask: 注意力掩码
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # 音频嵌入
        x = self.audio_embed(audio_tokens)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # 文本条件投影
        memory = self.text_proj(text_embedding)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=attention_mask)
        
        # 输出logits
        return self.output(x)
    
    @torch.no_grad()
    def generate(
        self,
        text_embedding: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        生成音频token序列
        
        Args:
            text_embedding: 文本条件
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
        
        Returns:
            生成的音频token
        """
        batch_size = text_embedding.size(0)
        device = text_embedding.device
        
        # 初始化：开始token
        tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # 预测下一个token
            logits = self.forward(tokens, text_embedding)
            
            # 只取最后一个位置
            logits = logits[:, -1, :] / temperature
            
            # Top-K采样
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens[:, 1:]  # 移除开始token


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = MusicGenTransformer(
        audio_dim=512,
        text_dim=768,
        num_heads=16,
        num_layers=24,
        vocab_size=2048
    )
    
    # 模拟输入
    batch_size = 2
    audio_tokens = torch.randint(0, 2048, (batch_size, 100))
    text_embedding = torch.randn(batch_size, 77, 768)
    
    # 前向传播
    logits = model(audio_tokens, text_embedding)
    print(f"Logits形状: {logits.shape}")  # [2, 100, 2048]
    
    # 生成
    # generated = model.generate(text_embedding, max_length=512)
    # print(f"生成token数量: {generated.size(1)}")
```

---

## 多模态编辑技术

### InstructPix2Pix

InstructPix2Pix实现了**文本引导的图像编辑**：

```
┌─────────────────────────────────────────────────────────────┐
│                InstructPix2Pix 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入：                                                     │
│   ┌──────────┐     ┌──────────────────────────────────┐    │
│   │ 原始图像  │     │ 编辑指令                          │    │
│   │          │     │ "Turn the cat into a dog"         │    │
│   └────┬─────┘     └─────────────────┬────────────────┘    │
│        │                             │                      │
│        ▼                             ▼                      │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Conditional Diffusion Model         │     │
│   │                                                  │     │
│   │   条件输入：                                     │     │
│   │   - 原始图像（通过VAE编码）                       │     │
│   │   - 编辑指令文本嵌入                              │     │
│   │   - CFG引导                                      │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              编辑后的图像                         │     │
│   │         保持原图结构，只修改指定内容               │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**双条件CFG**：

InstructPix2Pix需要同时处理图像条件和文本条件：

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, \emptyset, \emptyset) + s_I \cdot (\epsilon_\theta(z_t, I, \emptyset) - \epsilon_\theta(z_t, \emptyset, \emptyset)) + s_T \cdot (\epsilon_\theta(z_t, I, T) - \epsilon_\theta(z_t, I, \emptyset))
$$

其中：
- $I$：原始图像条件
- $T$：编辑指令条件
- $s_I, s_T$：图像和文本的引导强度

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix图像编辑模型"""
    
    def __init__(
        self,
        unet: nn.Module,
        vae_encoder: nn.Module,
        vae_decoder: nn.Module,
        text_encoder: nn.Module,
        image_guidance_scale: float = 1.5,
        text_guidance_scale: float = 7.5
    ):
        super().__init__()
        self.unet = unet
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像到潜空间"""
        with torch.no_grad():
            latent = self.vae_encoder(image)
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """解码潜空间到图像"""
        with torch.no_grad():
            image = self.vae_decoder(latent)
        return image
    
    def encode_text(self, text: str, tokenizer) -> torch.Tensor:
        """编码文本"""
        with torch.no_grad():
            tokens = tokenizer(text, return_tensors="pt", padding=True)
            embedding = self.text_encoder(**tokens)
        return embedding
    
    def dual_cfg_denoise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        image_cond: torch.Tensor,
        text_cond: torch.Tensor,
        null_text_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        双条件CFG去噪
        
        Args:
            z_t: 当前噪声潜表示
            t: 时间步
            image_cond: 图像条件
            text_cond: 文本条件
            null_text_cond: 空文本条件
        
        Returns:
            引导后的噪声预测
        """
        # 三种预测
        # 1. 无条件
        noise_uncond = self.unet(z_t, t, null_text_cond, image_cond)
        
        # 2. 仅图像条件
        noise_image_cond = self.unet(z_t, t, null_text_cond, image_cond)
        
        # 3. 图像+文本条件
        noise_full_cond = self.unet(z_t, t, text_cond, image_cond)
        
        # 双CFG
        noise_guided = noise_uncond + \
            self.image_guidance_scale * (noise_image_cond - noise_uncond) + \
            self.text_guidance_scale * (noise_full_cond - noise_image_cond)
        
        return noise_guided
    
    @torch.no_grad()
    def edit(
        self,
        image: torch.Tensor,
        instruction: str,
        tokenizer,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        编辑图像
        
        Args:
            image: [batch, 3, H, W] 输入图像
            instruction: 编辑指令
            tokenizer: 文本分词器
            num_steps: 采样步数
        
        Returns:
            编辑后的图像
        """
        device = image.device
        batch_size = image.size(0)
        
        # 编码图像条件
        image_cond = self.encode_image(image)
        
        # 编码文本条件
        text_cond = self.encode_text(instruction, tokenizer).to(device)
        null_text_cond = self.encode_text("", tokenizer).to(device)
        
        # 初始化噪声
        latent = torch.randn_like(image_cond)
        
        # 扩散时间表
        betas = torch.linspace(0.00085, 0.012, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 逐步去噪
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 双CFG去噪
            noise_pred = self.dual_cfg_denoise(
                latent, t, image_cond, text_cond, null_text_cond
            )
            
            # 去噪步骤
            alpha = alphas[i]
            alpha_cumprod = alphas_cumprod[i]
            
            if i > 0:
                noise = torch.randn_like(latent)
                sigma = betas[i] ** 0.5
            else:
                noise = 0
                sigma = 0
            
            latent = (latent - (1 - alpha) / (1 - alpha_cumprod) ** 0.5 * noise_pred) / alpha ** 0.5
            latent = latent + sigma * noise
        
        # 解码
        edited_image = self.decode_latent(latent)
        
        return edited_image


class ImageBlendEditor:
    """图像混合编辑器"""
    
    def __init__(self, model: InstructPix2Pix):
        self.model = model
    
    def progressive_edit(
        self,
        image: torch.Tensor,
        instructions: list,
        tokenizer,
        blend_weight: float = 0.5
    ) -> torch.Tensor:
        """
        渐进式编辑：应用多个编辑指令
        
        Args:
            image: 输入图像
            instructions: 编辑指令列表
            tokenizer: 分词器
            blend_weight: 每次编辑的权重
        
        Returns:
            最终编辑结果
        """
        current_image = image
        
        for instruction in instructions:
            edited = self.model.edit(current_image, instruction, tokenizer)
            current_image = blend_weight * edited + (1 - blend_weight) * current_image
        
        return current_image
    
    def mask_edit(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        instruction: str,
        tokenizer
    ) -> torch.Tensor:
        """
        掩码编辑：只编辑掩码区域
        
        Args:
            image: 输入图像
            mask: [batch, 1, H, W] 编辑掩码（1=编辑，0=保持）
            instruction: 编辑指令
            tokenizer: 分词器
        
        Returns:
            编辑后的图像
        """
        # 编辑整张图像
        edited = self.model.edit(image, instruction, tokenizer)
        
        # 掩码混合
        result = mask * edited + (1 - mask) * image
        
        return result


# 使用示例
if __name__ == "__main__":
    # 假设已有预训练模型组件
    # unet, vae_encoder, vae_decoder, text_encoder = load_pretrained_models()
    
    # 创建InstructPix2Pix
    # editor = InstructPix2Pix(
    #     unet=unet,
    #     vae_encoder=vae_encoder,
    #     vae_decoder=vae_decoder,
    #     text_encoder=text_encoder,
    #     image_guidance_scale=1.5,
    #     text_guidance_scale=7.5
    # )
    
    # 加载图像
    # from PIL import Image
    # image = Image.open("input.jpg")
    # image_tensor = transform(image).unsqueeze(0)
    
    # 编辑
    # edited = editor.edit(
    #     image_tensor,
    #     instruction="Turn the day into night",
    #     tokenizer=tokenizer
    # )
    
    # 保存结果
    # save_image(edited, "edited.jpg")
    pass
```

---

## 知识点关联

```
┌─────────────────────────────────────────────────────────────┐
│                   多模态生成知识图谱                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      扩散模型                                │
│                    (核心生成范式)                            │
│                         │                                   │
│           ┌─────────────┼─────────────┐                    │
│           │             │             │                    │
│           ▼             ▼             ▼                    │
│      Latent         Pixel         DiT                    │
│     Diffusion     Diffusion    (Transformer)             │
│     (SD/SDXL)      (DALL-E2)                              │
│           │             │             │                    │
│           └─────────────┼─────────────┘                    │
│                         │                                   │
│                         ▼                                   │
│              ┌──────────────────────┐                     │
│              │    条件注入技术       │                     │
│              │  - CLIP文本编码       │                     │
│              │  - Cross-Attention    │                     │
│              │  - CFG引导           │                     │
│              └──────────┬───────────┘                     │
│                         │                                   │
│         ┌───────────────┼───────────────┐                 │
│         │               │               │                 │
│         ▼               ▼               ▼                 │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│    │图像生成 │    │视频生成 │    │音频生成 │            │
│    │ DALL-E  │    │  Sora   │    │AudioLDM │            │
│    │  SDXL   │    │ Runway  │    │MusicGen │            │
│    │Midjourney│   │         │    │         │            │
│    └─────────┘    └─────────┘    └─────────┘            │
│         │               │               │                 │
│         └───────────────┼───────────────┘                 │
│                         │                                   │
│                         ▼                                   │
│              ┌──────────────────────┐                     │
│              │    多模态编辑         │                     │
│              │  InstructPix2Pix      │                     │
│              │  图像/视频编辑         │                     │
│              └──────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心考点

### 理论考点

1. **扩散模型基础**
   - 前向扩散过程的数学描述
   - 反向去噪过程的参数化
   - 训练目标（简化损失 vs 完整ELBO）

2. **Latent Diffusion**
   - 为什么在潜空间进行扩散？（计算效率）
   - VAE的编码和解码过程
   - 潜空间维度与生成质量的权衡

3. **条件生成技术**
   - Classifier-Free Guidance原理和公式
   - 交叉注意力的条件注入方式
   - 多条件组合（如InstructPix2Pix的双CFG）

4. **视频生成挑战**
   - 时间一致性问题
   - 时空建模方法（时空注意力分离 vs 联合）
   - 计算复杂度优化

### 实践考点

1. **模型架构理解**
   - U-Net在扩散模型中的作用
   - DiT相比U-Net的优劣
   - 时间步嵌入的实现方式

2. **训练技巧**
   - 噪声调度（线性 vs 余弦）
   - 预测目标（噪声 vs 速度 vs 原始数据）
   - 条件丢弃策略（CFG训练）

3. **推理优化**
   - DDIM加速采样
   - DDPM vs DDIM采样对比
   - 批量处理与内存优化

### 公式记忆重点

| 公式名称 | 数学表达 | 含义 |
|----------|----------|------|
| 前向扩散 | $z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | 加噪过程 |
| 简化损失 | $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t)\|^2]$ | 噪声预测MSE |
| CFG | $\tilde{\epsilon} = \epsilon_{\emptyset} + s(\epsilon_c - \epsilon_{\emptyset})$ | 条件引导 |
| 双CFG | 添加图像条件引导项 | 多条件编辑 |

---

## 学习建议

### 📚 推荐学习路径

```
第一阶段：扩散模型基础（2周）
├── 理解DDPM原始论文
├── 掌握前向/反向过程数学
└── 实现简单的MNIST扩散模型

第二阶段：Latent Diffusion（2周）
├── 学习VAE原理
├── 理解潜空间扩散的优势
└── 运行Stable Diffusion推理

第三阶段：条件生成技术（2周）
├── 深入理解CFG
├── 学习交叉注意力条件注入
└── 实现文本条件扩散

第四阶段：高级应用（3周）
├── 视频生成模型（Sora架构）
├── 音频生成（AudioLDM/MusicGen）
└── 多模态编辑技术

第五阶段：实践项目（持续）
├── 微调Stable Diffusion
├── 实现自定义条件控制
└── 探索最新模型架构
```

### 🔧 实践建议

1. **从现成模型开始**
   - 先使用Hugging Face Diffusers库进行推理
   - 理解pipeline的工作流程
   - 尝试不同参数的效果

2. **逐步深入代码**
   - 阅读关键模块实现（UNet2DConditionModel）
   - 理解采样器的实现（DDIMScheduler）
   - 尝试修改和扩展

3. **关注最新进展**
   - Diffusers库更新
   - 新模型发布（SDXL Turbo、SD3等）
   - 社区训练方法（LoRA、DreamBooth）

### 📖 推荐资源

| 资源类型 | 名称 | 说明 |
|----------|------|------|
| 论文 | DDPM (Ho et al., 2020) | 扩散模型奠基作 |
| 论文 | Latent Diffusion (Rombach et al., 2022) | Stable Diffusion原理 |
| 论文 | DiT (Peebles & Xie, 2023) | Transformer扩散架构 |
| 代码库 | Hugging Face Diffusers | 主流扩散模型实现 |
| 教程 | fast.ai diffusion course | 实践导向教程 |
| 社区 | r/StableDiffusion | 社区动态和技巧 |

---

## 参考资料

1. **DALL-E系列**
   - DALL-E: Zero-Shot Text-to-Image Generation (Ramesh et al., 2021)
   - DALL-E 2: Hierarchical Text-Conditional Image Generation (Ramesh et al., 2022)
   - DALL-E 3 Technical Report (OpenAI, 2023)

2. **扩散模型基础**
   - DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
   - DDIM: Denoising Diffusion Implicit Models (Song et al., 2021)
   - Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)

3. **Stable Diffusion**
   - High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)
   - SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (Podell et al., 2023)

4. **视频生成**
   - Sora Technical Report (OpenAI, 2024)
   - Scalable Diffusion Models for Natural Video Generation (Ho et al., 2022)

5. **音频生成**
   - AudioLDM: Text-to-Audio Generation with Latent Diffusion Models (Liu et al., 2023)
   - MusicGen: Simple and Controllable Music Generation (Copet et al., 2023)
