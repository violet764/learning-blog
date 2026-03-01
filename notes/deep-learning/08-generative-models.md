# 生成模型

生成模型是深度学习中最具挑战性和创造性的领域之一。与判别模型不同，生成模型旨在学习数据的真实分布 $p(\mathbf{x})$，从而能够生成与真实数据相似的新样本。本章将系统介绍四类主流生成模型：变分自编码器（VAE）、生成对抗网络（GAN）、扩散模型（Diffusion）以及评估方法。

---

## 章节概述

### 📌 核心问题

| 问题 | 描述 |
|------|------|
| **密度估计** | 如何从有限样本中学习数据分布 $p(\mathbf{x})$？ |
| **采样生成** | 如何从学习到的分布中高效采样新样本？ |
| **潜在表示** | 如何学习数据的低维隐变量表示？ |
| **评估度量** | 如何量化生成样本的质量和多样性？ |

### 📌 生成模型谱系

```
生成模型
├── 显式密度模型
│   ├── 自回归模型 (PixelCNN, GPT)
│   └── 变分自编码器 (VAE)
├── 隐式密度模型
│   └── 生成对抗网络 (GAN)
└── 得分匹配模型
    └── 扩散模型 (DDPM, Stable Diffusion)
```

---

## 核心知识点详解

### 一、判别式模型 vs 生成式模型

#### 1.1 核心区别

**判别式模型**直接学习条件概率 $p(y|\mathbf{x})$，即给定输入 $\mathbf{x}$ 预测标签 $y$：

$$
\text{判别式模型}: \quad \hat{y} = \arg\max_y p(y|\mathbf{x})
$$

**生成式模型**学习联合概率 $p(\mathbf{x}, y)$ 或数据分布 $p(\mathbf{x})$：

$$
\text{生成式模型}: \quad p(\mathbf{x}, y) = p(y) \cdot p(\mathbf{x}|y)
$$

根据贝叶斯定理，两者关系为：

$$
p(y|\mathbf{x}) = \frac{p(\mathbf{x}|y) \cdot p(y)}{p(\mathbf{x})}
$$

#### 1.2 对比分析

| 维度 | 判别式模型 | 生成式模型 |
|------|-----------|-----------|
| **学习目标** | 决策边界 | 数据分布 |
| **概率建模** | $p(y\|\mathbf{x})$ | $p(\mathbf{x}, y)$ 或 $p(\mathbf{x})$ |
| **典型模型** | Logistic回归、SVM、神经网络 | 朴素贝叶斯、GMM、VAE、GAN |
| **分类性能** | 通常更优 | 通常次优 |
| **生成能力** | 无 | 有 |
| **数据效率** | 需要标签数据 | 可利用无标签数据 |
| **异常检测** | 困难 | 天然支持 |

#### 1.3 生成模型的优势

1. **数据生成**：能够创造新的样本
2. **无监督学习**：不需要标签即可学习
3. **密度估计**：可用于异常检测、数据压缩
4. **多任务迁移**：学习到的表示可迁移到下游任务

---

### 二、变分自编码器 (VAE)

#### 2.1 模型动机

VAE 由 Kingma 和 Welling 于 2013 年提出，核心思想是通过**变分推断**学习数据的潜在表示。

假设数据由隐变量 $\mathbf{z}$ 生成：

$$
p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}
$$

直接最大化边际似然 $p(\mathbf{x})$ 通常不可行，因为后验分布 $p(\mathbf{z}|\mathbf{x})$ 难以计算。

#### 2.2 变分下界 (ELBO) 推导

**核心思想**：用易处理的分布 $q_\phi(\mathbf{z}|\mathbf{x})$ 近似真实的后验 $p(\mathbf{z}|\mathbf{x})$。

从对数似然出发：

$$
\log p(\mathbf{x}) = \log p(\mathbf{x}) \int q_\phi(\mathbf{z}|\mathbf{x}) d\mathbf{z} = \int q_\phi(\mathbf{z}|\mathbf{x}) \log p(\mathbf{x}) d\mathbf{z}
$$

引入隐变量 $\mathbf{z}$：

$$
\log p(\mathbf{x}) = \int q_\phi(\mathbf{z}|\mathbf{x}) \log \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}
$$

引入推断网络 $q_\phi(\mathbf{z}|\mathbf{x})$：

$$
\log p(\mathbf{x}) = \int q_\phi(\mathbf{z}|\mathbf{x}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}
$$

展开得到：

$$
\log p(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi}[\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x})]}_{\text{ELBO } \mathcal{L}} + \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))}_{\text{KL散度} \geq 0}
$$

因此，**ELBO 是对数似然的下界**：

$$
\log p(\mathbf{x}) \geq \mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

**ELBO 的两项含义**：

- **重建项** $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$：希望从隐变量重建原始数据
- **正则项** $D_{KL}(q_\phi \| p)$：约束隐变量分布接近先验（通常为标准正态分布）

#### 2.3 重参数化技巧

直接对采样操作求梯度不可行。**重参数化技巧**将随机性转移到参数外部：

$$
\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))
$$

重参数化为：

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

这样梯度可以通过 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 反向传播：

$$
\frac{\partial \mathcal{L}}{\partial \phi} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \phi}
$$

#### 2.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """变分自编码器实现"""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # 编码器: x -> μ, log(σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差
        
        # 解码器: z -> x_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def encode(self, x):
        """编码到隐空间分布参数"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        z = mu + std * eps              # z = μ + σ·ε
        return z
    
    def decode(self, z):
        """从隐变量解码重建"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device='cpu'):
        """从先验分布采样新数据"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    VAE 损失函数
    
    Args:
        x: 原始输入
        x_recon: 重建输出
        mu: 隐变量均值
        logvar: 隐变量对数方差
        beta: KL 散度权重 (β-VAE)
    
    Returns:
        总损失 = 重建损失 + β × KL散度
    """
    # 重建损失 (二元交叉熵，适用于 [0,1] 范围数据)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL 散度: D_KL(q(z|x) || p(z))
    # 对于高斯分布有闭式解:
    # KL = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


# 训练示例
def train_vae(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.view(x.size(0), -1).to(device)  # 展平
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = vae_loss(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
```

#### 2.5 VAE 变体

| 变体 | 核心改进 | 应用场景 |
|------|---------|---------|
| **β-VAE** | 增加 KL 散度权重 $\beta > 1$ | 解纠缠表示学习 |
| **VQ-VAE** | 离散隐空间 + 向量量化 | 图像/音频生成 |
| **Conditional VAE** | 引入条件信息 $c$ | 条件生成任务 |
| **Hierarchical VAE** | 多层隐变量结构 | 高质量图像生成 |

---

### 三、生成对抗网络 (GAN)

#### 3.1 博弈论基础

GAN 由 Goodfellow 等人于 2014 年提出，将生成问题建模为**二人零和博弈**：

- **生成器 $G$**：学习将噪声 $\mathbf{z} \sim p_z$ 映射到数据空间，试图欺骗判别器
- **判别器 $D$**：学习区分真实数据 $\mathbf{x} \sim p_{data}$ 和生成数据 $G(\mathbf{z})$

**目标函数**：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]
$$

#### 3.2 理论分析

**最优判别器**：当 $G$ 固定时，最优判别器为：

$$
D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}
$$

**最优生成器**：当 $D = D^*$ 时，目标函数等价于 JS 散度：

$$
V(D^*, G) = 2 \cdot D_{JS}(p_{data} \| p_g) - \log 4
$$

因此，**最优生成器使得 $p_g = p_{data}$**，此时 $D^*(\mathbf{x}) = 0.5$。

#### 3.3 训练技巧

**问题**：原始 GAN 训练不稳定，易出现模式崩溃。

**常用技巧**：

1. **标签平滑**：真实标签用 $0.9$ 代替 $1.0$
2. **噪声注入**：在判别器输入添加噪声
3. **经验回放**：存储历史生成样本
4. **谱归一化**：约束判别器的 Lipschitz 常数
5. **TTUR (Two Time-Scale Update Rule)**：判别器学习率高于生成器

**改进损失函数**：

非饱和损失（生成器使用）：

$$
\mathcal{L}_G = -\mathbb{E}_{\mathbf{z} \sim p_z}[\log D(G(\mathbf{z}))]
$$

Wasserstein GAN (WGAN) 损失：

$$
\min_G \max_{D \in \mathcal{W}} \mathbb{E}_{\mathbf{x}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))]
$$

#### 3.4 模式崩溃问题

**现象**：生成器只产生有限种类的样本，无法覆盖真实数据分布的全部模式。

**原因分析**：

$$
\nabla_\theta \mathbb{E}_{\mathbf{z}}[\log(1 - D(G_\theta(\mathbf{z})))] \approx \sum_i \nabla_\theta \log(1 - D(G_\theta(\mathbf{z}_i)))
$$

当 $G$ 对多个 $\mathbf{z}_i$ 产生相似输出时，梯度指向同一方向，强化模式崩溃。

**解决方案**：

| 方法 | 原理 |
|------|------|
| **WGAN** | 使用 Wasserstein 距离，梯度更稳定 |
| **Unrolled GAN** | 生成器考虑判别器多步更新 |
| **Minibatch Discrimination** | 判别器同时处理多个样本 |
| **Feature Matching** | 匹配中间特征而非最终输出 |

#### 3.5 PyTorch 实现

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """DCGAN 生成器"""
    
    def __init__(self, latent_dim=100, channels=3, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 状态: (feature_maps*8, 4, 4)
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 状态: (feature_maps*4, 8, 8)
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 状态: (feature_maps*2, 16, 16)
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 状态: (feature_maps, 32, 32)
            
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (channels, 64, 64)
        )
    
    def forward(self, z):
        return self.main(z.view(z.size(0), z.size(1), 1, 1))


class Discriminator(nn.Module):
    """DCGAN 判别器"""
    
    def __init__(self, channels=3, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入: (channels, 64, 64)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)


def weights_init(m):
    """自定义权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(G, D, dataloader, optimizer_G, optimizer_D, epochs, device):
    """GAN 训练循环"""
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # 真实和假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== 训练判别器 ==========
            optimizer_D.zero_grad()
            
            # 真实图像的损失
            real_output = D(real_imgs)
            d_loss_real = criterion(real_output, real_labels)
            
            # 生成图像的损失
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = G(z)
            fake_output = D(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # ========== 训练生成器 ==========
            optimizer_G.zero_grad()
            
            fake_output = D(fake_imgs)
            g_loss = criterion(fake_output, real_labels)  # 希望判别器认为是真
            
            g_loss.backward()
            optimizer_G.step()
            
        print(f'Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
```

---

### 四、扩散模型 (Diffusion Models)

#### 4.1 基本思想

扩散模型受非平衡热力学启发，包含两个过程：

1. **前向扩散过程**：逐步向数据添加噪声，直至变成纯噪声
2. **反向去噪过程**：学习逐步去噪，从噪声恢复数据

#### 4.2 前向扩散过程

给定数据 $\mathbf{x}_0 \sim q(\mathbf{x})$，定义马尔可夫链：

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

其中 $\beta_t$ 是噪声调度参数，$t \in \{1, 2, ..., T\}$。

**闭式采样**：任意时刻 $t$ 的 $\mathbf{x}_t$ 可直接从 $\mathbf{x}_0$ 得到：

设 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，则：

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

采样公式：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

#### 4.3 反向去噪过程

真实后验 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 不可直接计算，但可证明当 $\beta_t$ 足够小时：

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

其中：

$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

**关键洞察**：由于 $\mathbf{x}_0$ 未知，我们用神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 预测噪声，从而估计 $\mathbf{x}_0$：

$$
\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
$$

#### 4.4 DDPM 训练目标

**简化损失函数**（DDPM 论文）：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$

其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$。

**完整 ELBO 推导**：

变分下界可以分解为：

$$
\mathcal{L} = \mathbb{E}_q \left[ \underbrace{D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} \right]
$$

- $L_T$：常数项（不参与训练）
- $L_{t-1}$：去噪步骤的 KL 散度
- $L_0$：重建项

#### 4.5 采样算法

**DDPM 采样**（去噪过程）：

```
输入: 噪声 x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, t)) + √β_t·z
输出: x_0
```

**DDIM 采样**（加速采样，无需马尔可夫）：

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{预测的 } \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
$$

#### 4.6 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步 t"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """时间步嵌入模块"""
    
    def __init__(self, time_dim, emb_dim):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, t):
        t = self.sinusoidal(t)
        return self.mlp(t)


class ResBlock(nn.Module):
    """残差块，带时间嵌入"""
    
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 加入时间嵌入
        h += self.time_mlp(t)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class UNet(nn.Module):
    """简化版 UNet 噪声预测网络"""
    
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = TimeEmbedding(time_dim, time_dim)
        
        # 编码器
        self.enc1 = ResBlock(in_channels, 64, time_dim)
        self.enc2 = ResBlock(64, 128, time_dim)
        self.enc3 = ResBlock(128, 256, time_dim)
        self.down = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = ResBlock(256, 512, time_dim)
        
        # 解码器
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ResBlock(512 + 256, 256, time_dim)
        self.dec2 = ResBlock(256 + 128, 128, time_dim)
        self.dec1 = ResBlock(128 + 64, 64, time_dim)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        # 时间嵌入
        t = self.time_mlp(t)
        
        # 编码器
        e1 = self.enc1(x, t)
        e2 = self.enc2(self.down(e1), t)
        e3 = self.enc3(self.down(e2), t)
        
        # 瓶颈层
        b = self.bottleneck(self.down(e3), t)
        
        # 解码器（带跳跃连接）
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1), t)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t)
        
        return self.out(d1)


class DDPM:
    """DDPM 扩散模型"""
    
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算常用项
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod.roll(1)) / (1 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_0, t, noise=None):
        """前向扩散：从 x_0 得到 x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 广播维度
        while len(sqrt_alpha.shape) < len(x_0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0, t, noise=None):
        """计算去噪损失"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        
        return F.mse_loss(noise, predicted_noise)
    
    def p_sample(self, x_t, t):
        """单步去噪"""
        beta_t = self.betas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        
        # 预测噪声
        predicted_noise = self.model(x_t, t.unsqueeze(0))
        
        # 计算均值
        model_mean = sqrt_recip_alpha * (
            x_t - beta_t / sqrt_one_minus_alpha * predicted_noise
        )
        
        # 添加噪声（t=0 时不加）
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            while len(variance.shape) < len(x_t.shape):
                variance = variance.unsqueeze(-1)
            return model_mean + torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(self, shape):
        """从噪声生成样本"""
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(shape).to(self.device)
        
        # 逐步去噪
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t], device=self.device).repeat(shape[0])
            x = self.p_sample(x, t_batch)
        
        return x
    
    def train_step(self, x_0, optimizer):
        """单步训练"""
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = x_0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        loss = self.p_losses(x_0, t)
        loss.backward()
        optimizer.step()
        
        return loss.item()
```

#### 4.7 重要扩展

| 模型 | 核心改进 | 应用 |
|------|---------|------|
| **DDPM** | 基础扩散框架 | 图像生成 |
| **DDIM** | 非马尔可夫采样，加速生成 | 快速生成 |
| **Score-based models** | 得分匹配 + SDE | 统一框架 |
| **Stable Diffusion** | 潜空间扩散 + 文本条件 | 文生图 |
| **DALL-E 2** | 先验模型 + 扩散解码器 | 文生图 |

---

### 五、生成模型评估指标

#### 5.1 Inception Score (IS)

**原理**：同时评估生成质量与多样性。

$$
IS = \exp\left( \mathbb{E}_{\mathbf{x} \sim p_g} \left[ D_{KL}(p(y|\mathbf{x}) \| p(y)) \right] \right)
$$

其中：
- $p(y|\mathbf{x})$：Inception 网络对生成图像的预测分布
- $p(y) = \mathbb{E}_{\mathbf{x}}[p(y|\mathbf{x})]$：边际分布

**解释**：
- 高质量图像 → $p(y|\mathbf{x})$ 低熵（高置信度预测）
- 高多样性 → $p(y)$ 高熵（均匀分布）
- IS 越高越好

**局限性**：
- 不考虑真实数据分布
- 对 Inception 模型依赖性强

#### 5.2 Fréchet Inception Distance (FID)

**原理**：比较生成数据与真实数据在特征空间的分布差异。

$$
FID = \| \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \|^2 + \text{Tr}(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2})
$$

其中 $(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$ 和 $(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$ 分别是真实数据和生成数据在 Inception 特征空间的均值和协方差。

**特点**：
- FID 越低越好
- 直接与真实数据比较
- 对模式崩溃敏感

#### 5.3 其他指标

| 指标 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Precision & Recall** | 生成质量 vs 多样性 | 可权衡分析 | 需要分布估计 |
| **IS (Inception Score)** | 质量与多样性的综合 | 计算简单 | 不与真实数据比较 |
| **LPIPS** | 感知相似度 | 符合人类感知 | 需要预训练网络 |
| **CLIP Score** | 文本-图像一致性 | 多模态评估 | 依赖 CLIP 模型 |

#### 5.4 PyTorch 实现

```python
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np

def calculate_inception_score(preds, splits=10):
    """
    计算 Inception Score
    
    Args:
        preds: (N, C) 模型预测概率分布
        splits: 分割数量
    """
    scores = []
    N = preds.shape[0]
    split_size = N // splits
    
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        
        # 计算 KL 散度
        py = part.mean(axis=0, keepdims=True)  # 边际分布
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = kl.sum(axis=1)
        
        scores.append(np.exp(kl.mean()))
    
    return np.mean(scores), np.std(scores)


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    计算 Fréchet Inception Distance
    
    Args:
        mu1, sigma1: 真实数据的均值和协方差
        mu2, sigma2: 生成数据的均值和协方差
    """
    # 平方项
    diff = mu1 - mu2
    
    # 计算协方差的平方根矩阵
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理数值不稳定
    if not np.isfinite(covmean).all():
        covmean = np.nan_to_num(covmean, nan=0.0, posinf=1e10, neginf=-1e10)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 处理复数情况
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 使用示例
class FIDCalculator:
    """FID 计算器"""
    
    def __init__(self, inception_model, device='cuda'):
        self.model = inception_model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def get_features(self, dataloader):
        """提取特征"""
        features = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            feat = self.model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    def calculate_statistics(self, features):
        """计算均值和协方差"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(self, real_loader, fake_loader):
        """计算 FID"""
        real_features = self.get_features(real_loader)
        fake_features = self.get_features(fake_loader)
        
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(fake_features)
        
        return calculate_fid(mu1, sigma1, mu2, sigma2)
```

---

## 知识点关联

```
┌─────────────────────────────────────────────────────────────────┐
│                        生成模型知识图谱                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌───────────┐                    ┌───────────┐               │
│    │   VAE     │                    │    GAN    │               │
│    │ 显式密度  │                    │ 隐式密度  │               │
│    └─────┬─────┘                    └─────┬─────┘               │
│          │                                │                     │
│          │ 变分推断                        │ 对抗训练             │
│          │                                │                     │
│          ▼                                ▼                     │
│    ┌───────────┐                    ┌───────────┐               │
│    │ ELBO优化  │                    │ 极小极大  │               │
│    │ 重参数化  │                    │ Nash均衡  │               │
│    └─────┬─────┘                    └─────┬─────┘               │
│          │                                │                     │
│          └────────────┬───────────────────┘                     │
│                       │                                         │
│                       ▼                                         │
│               ┌───────────────┐                                 │
│               │  Diffusion    │                                 │
│               │  得分匹配      │                                 │
│               └───────┬───────┘                                 │
│                       │                                         │
│          ┌────────────┼────────────┐                            │
│          │            │            │                            │
│          ▼            ▼            ▼                            │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│    │ 前向扩散 │ │ 反向去噪 │ │ 评估指标 │                      │
│    └──────────┘ └──────────┘ └──────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 模型演进关系

| 模型 | 关键创新 | 解决的问题 | 遗留问题 |
|------|---------|-----------|---------|
| **VAE** | 变分推断 + 重参数化 | 可训练的潜在表示 | 生成模糊 |
| **GAN** | 对抗训练 | 清晰生成 | 训练不稳定 |
| **Diffusion** | 渐进去噪 | 稳定训练 + 高质量 | 采样慢 |

### 核心数学联系

1. **VAE 与 Diffusion**：
   - VAE 可视为单步扩散
   - Hierarchical VAE 与 Diffusion 有深层联系

2. **GAN 与 Diffusion**：
   - GAN 判别器提供"真假"信号
   - Diffusion 的噪声预测器提供"方向"信号

3. **统一视角**：
   - Score-based models 统一了 Diffusion 和部分 GAN 理论
   - Flow-based models 与 VAE 有共同的 ELBO 目标

---

## 核心考点

### 🎯 理论推导

1. **VAE ELBO 推导**：从边际似然出发，推导变分下界
2. **重参数化技巧**：解释为什么需要以及如何实现
3. **GAN 纳什均衡**：证明最优判别器形式和 JS 散度关系
4. **DDPM 前向过程闭式解**：推导 $\mathbf{x}_t = f(\mathbf{x}_0, t, \boldsymbol{\epsilon})$

### 🎯 概念辨析

1. **判别式 vs 生成式模型**：目标、能力、典型应用
2. **显式 vs 隐式密度模型**：优缺点对比
3. **VAE 模糊 vs GAN 清晰**：原因分析（损失函数差异）
4. **模式崩溃**：定义、原因、解决方案

### 🎯 实践应用

1. **GAN 训练不稳定**：常见问题和解决技巧
2. **Diffusion 采样加速**：DDIM、DPM-Solver 原理
3. **评估指标选择**：IS vs FID 的适用场景
4. **条件生成**：条件 VAE、cGAN、Classifier-free Guidance

---

## 学习建议

### 📚 推荐学习路径

```
阶段一：基础概念 (1-2周)
├── 理解判别式与生成式模型区别
├── 掌握 VAE 原理与 ELBO 推导
└── 动手实现简单 VAE

阶段二：GAN 深入 (2-3周)
├── 理解博弈论基础与纳什均衡
├── 实现基础 GAN 和 DCGAN
├── 学习训练技巧（WGAN、谱归一化）
└── 理解模式崩溃问题

阶段三：扩散模型 (3-4周)
├── 掌握前向扩散过程的数学推导
├── 理解反向去噪过程
├── 实现 DDPM
└── 学习加速采样方法（DDIM）

阶段四：进阶拓展
├── Latent Diffusion Models
├── Score-based Generative Models
└── 最新论文跟踪
```

### 📖 推荐资源

**经典论文**：
- Kingma & Welling, "Auto-Encoding Variational Bayes" (VAE, 2013)
- Goodfellow et al., "Generative Adversarial Networks" (GAN, 2014)
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020)

**代码实践**：
- PyTorch 官方教程
- Hugging Face Diffusers 库

### ⚠️ 常见误区

1. **VAE 的 KL 散度项**：不是"越小越好"，需要与重建项平衡
2. **GAN 的判别器准确率**：50% 不代表训练成功，可能是生成器太强
3. **扩散模型的采样步数**：不是越多越好，存在最优值
4. **FID 数值**：绝对值意义有限，应关注相对比较

### 🔬 实验建议

1. **VAE 实验**：调整 $\beta$ 参数观察解纠缠效果
2. **GAN 实验**：对比原始 GAN 和 WGAN 的训练稳定性
3. **Diffusion 实验**：尝试不同的噪声调度（线性、余弦）
4. **评估实验**：计算生成数据集的 FID 曲线
