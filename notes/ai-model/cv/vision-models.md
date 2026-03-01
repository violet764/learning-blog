# 视觉模型演进

视觉模型的发展是深度学习领域最激动人心的篇章之一。从 AlexNet 开启深度学习时代，到 ResNet 的残差革命，再到 Vision Transformer 的跨界创新，每一次突破都推动着计算机视觉能力的飞跃。本章将系统梳理视觉骨干网络的演进历程，帮助读者理解不同架构的设计思想与适用场景。

```
┌─────────────────────────────────────────────────────────────────────┐
│                       视觉模型演进时间线                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  2012 ──► AlexNet        开启深度学习时代                           │
│              │                                                      │
│  2014 ──► VGG/GoogLeNet  更深更宽的网络                             │
│              │                                                      │
│  2015 ──► ResNet         残差连接，突破深度瓶颈                      │
│              │                                                      │
│  2017 ──► DenseNet       密集连接，特征复用                          │
│              │                                                      │
│  2018 ──► EfficientNet   复合缩放，效率最优                          │
│              │                                                      │
│  2020 ──► ViT            Transformer 进入视觉领域                    │
│              │                                                      │
│  2021 ──► Swin           层次化 Transformer                         │
│           ConvNeXt       现代化 CNN 反击                             │
│              │                                                      │
│  2021+ ─► MAE/BEiT/DINO  自监督预训练兴起                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## CNN 架构演进

卷积神经网络（CNN）通过局部感受野和权值共享，成为视觉任务的核心架构。从 AlexNet 到 EfficientNet，CNN 架构经历了多次革命性创新。

### ResNet：残差学习

#### 核心问题

随着网络深度增加，出现**退化问题（Degradation Problem）**：更深的网络反而表现更差。这不是过拟合，而是优化困难。

> 💡 **关键洞察**：如果浅层网络能训练得好，那么深层网络至少应该能学到恒等映射。但实践中深层网络很难学到恒等映射。

#### 残差连接

ResNet 引入**跳跃连接（Skip Connection）**，让网络学习残差映射：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

其中 $\mathcal{F}(\mathbf{x}, \{W_i\})$ 是残差映射，$\mathbf{x}$ 是恒等映射。

**为什么有效？**

1. **梯度直通**：反向传播时，梯度可以直接通过跳跃连接传到浅层

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)
$$

2. **易于学习恒等映射**：如果恒等映射是最优解，只需让 $\mathcal{F}(\mathbf{x}) = 0$ 即可

3. **集成效应**：残差网络可视为不同深度子网络的集成

#### ResNet 基本块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet基本块（用于ResNet-18/34）"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 如果维度不匹配，需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差连接
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块（用于ResNet-50/101/152）"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1降维 -> 3x3卷积 -> 1x1升维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet完整实现"""
    
    # 每个stage的块数量配置
    configs = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    
    def __init__(self, depth=50, num_classes=1000):
        super().__init__()
        self.depth = depth
        self.block = Bottleneck if depth >= 50 else BasicBlock
        self.expansion = self.block.expansion
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 四个stage
        self.in_channels = 64
        layers = self.configs[depth]
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """构建一个stage"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        layers = []
        layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.expansion
        
        for _ in range(1, num_blocks):
            layers.append(self.block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # 四个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        """提取多尺度特征（用于下游任务）"""
        features = []
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # 1/4
        x = self.layer2(x)
        features.append(x)  # 1/8
        x = self.layer3(x)
        features.append(x)  # 1/16
        x = self.layer4(x)
        features.append(x)  # 1/32
        
        return features


# 创建模型
def resnet50(num_classes=1000):
    return ResNet(depth=50, num_classes=num_classes)
```

### DenseNet：密集连接

#### 核心思想

DenseNet 采用**密集连接（Dense Connectivity）**：每一层的输入来自前面所有层的输出。

$$
\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{\ell-1}])
$$

其中 $[\cdot]$ 表示特征图的通道拼接。

#### 关键优势

| 特性 | 说明 |
|------|------|
| **特征复用** | 每层都能直接访问前面所有层的特征 |
| **参数高效** | 特征图通道数可以很小（增长率 k=12） |
| **梯度流动** | 密集连接提供了多条梯度传播路径 |
| **隐式深层监督** | 每层都能接收来自损失的直接监督 |

#### 增长率与过渡层

**增长率（Growth Rate）$k$**：每层产生 $k$ 个新特征图。第 $\ell$ 层的输入通道数为：

$$
k_0 + k \times (\ell - 1)
$$

**过渡层（Transition Layer）**：使用 $1 \times 1$ 卷积压缩通道数，防止参数爆炸。

```python
class DenseLayer(nn.Module):
    """DenseNet的单层"""
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        # BN-ReLU-Conv(1x1) -> BN-ReLU-Conv(3x3)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False)
        
    def forward(self, x):
        # BN-ReLU-Conv 顺序（pre-activation）
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return out


class DenseBlock(nn.Module):
    """DenseNet的密集块"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """过渡层：压缩特征图"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
        
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))


class DenseNet(nn.Module):
    """DenseNet实现"""
    configs = {
        121: (6, 12, 24, 16),  # 每个block的层数
        169: (6, 12, 32, 32),
        201: (6, 12, 48, 32),
        264: (6, 12, 64, 48)
    }
    
    def __init__(self, depth=121, growth_rate=32, num_classes=1000, compression=0.5):
        super().__init__()
        self.growth_rate = growth_rate
        
        # 初始卷积
        num_init_features = 64
        self.conv1 = nn.Conv2d(3, num_init_features, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Dense Blocks
        num_features = num_init_features
        block_configs = self.configs[depth]
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_configs):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate
            
            if i != len(block_configs) - 1:  # 最后一个block不需要过渡层
                transition = TransitionLayer(
                    num_features, 
                    int(num_features * compression)
                )
                self.transitions.append(transition)
                num_features = int(num_features * compression)
        
        # 分类头
        self.bn_final = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### EfficientNet：复合缩放

#### 核心问题

传统网络缩放通常只调整单一维度：
- **深度**：更多层数
- **宽度**：更多通道数
- **分辨率**：更大输入尺寸

但如何平衡这三个维度？

#### 复合缩放方法

EfficientNet 提出**复合缩放（Compound Scaling）**，使用一组固定系数同时缩放三个维度：

$$
\begin{aligned}
\text{depth} &: d = \alpha^\phi \\
\text{width} &: w = \beta^\phi \\
\text{resolution} &: r = \gamma^\phi \\
\text{s.t.} &: \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
& \alpha \geq 1, \beta \geq 1, \gamma \geq 1
\end{aligned}
$$

其中 $\phi$ 是复合系数，由用户指定；$\alpha, \beta, \gamma$ 通过网格搜索确定。

**为什么是 $\beta^2$ 和 $\gamma^2$？**

因为计算量与宽度平方和分辨率平方成正比：

$$
\text{FLOPs} \propto \text{depth} \times \text{width}^2 \times \text{resolution}^2
$$

#### MBConv 基础块

EfficientNet 使用**移动倒瓶颈卷积（MBConv）**作为基础块：

```
输入 → 1x1升维 → 3x3/5x5深度卷积 → SE注意力 → 1x1降维 → 输出
         ↑___________________________________________↓ (残差连接)
```

```python
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation注意力"""
    def __init__(self, in_channels, squeeze_ratio=0.25):
        super().__init__()
        squeezed = int(in_channels * squeeze_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, squeezed),
            nn.ReLU(inplace=True),
            nn.Linear(squeezed, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)
        scale = self.excitation(scale).view(B, C, 1, 1)
        return x * scale


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded = in_channels * expand_ratio
        
        layers = []
        # 升维（如果需要）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.SiLU(inplace=True)
            ])
        
        # 深度可分离卷积
        layers.extend([
            nn.Conv2d(expanded, expanded, kernel_size, stride, 
                      kernel_size // 2, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True)
        ])
        
        # SE注意力
        if se_ratio > 0:
            layers.append(SqueezeExcitation(expanded, se_ratio))
        
        # 降维
        layers.extend([
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class EfficientNet(nn.Module):
    """简化版EfficientNet"""
    # [expand_ratio, channels, layers, kernel_size, stride]
    # 对应不同的stage配置
    stage_configs = [
        [1, 16, 1, 3, 1],
        [6, 24, 2, 3, 2],
        [6, 40, 2, 5, 2],
        [6, 80, 3, 3, 2],
        [6, 112, 3, 5, 1],
        [6, 192, 4, 5, 2],
        [6, 320, 1, 3, 1],
    ]
    
    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        
        # 初始卷积
        out_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 构建各stage
        self.stages = nn.ModuleList()
        in_channels = out_channels
        
        for expand_ratio, channels, num_layers, kernel_size, stride in self.stage_configs:
            # 应用宽度缩放
            out_channels = int(channels * width_mult)
            # 应用深度缩放
            num_layers = int(num_layers * depth_mult)
            
            layers = []
            for i in range(num_layers):
                s = stride if i == 0 else 1
                layers.append(MBConv(
                    in_channels if i == 0 else out_channels,
                    out_channels, kernel_size, s, expand_ratio
                ))
            
            self.stages.append(nn.Sequential(*layers))
            in_channels = out_channels
        
        # 分类头
        final_channels = 1280
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def efficientnet_b0(num_classes=1000):
    """EfficientNet-B0 (φ=0)"""
    return EfficientNet(num_classes, width_mult=1.0, depth_mult=1.0)

def efficientnet_b4(num_classes=1000):
    """EfficientNet-B4 (φ=4)"""
    return EfficientNet(num_classes, width_mult=1.4, depth_mult=1.8)
```

### CNN 架构对比

| 模型 | 年份 | 核心创新 | 参数量 | ImageNet Top-1 |
|------|------|----------|--------|----------------|
| ResNet-50 | 2015 | 残差连接 | 25.6M | 76.2% |
| ResNet-152 | 2015 | 更深层残差 | 60.2M | 78.3% |
| DenseNet-121 | 2017 | 密集连接 | 8.0M | 74.9% |
| DenseNet-201 | 2017 | 更密集连接 | 20.0M | 77.3% |
| EfficientNet-B0 | 2019 | 复合缩放 | 5.3M | 77.1% |
| EfficientNet-B7 | 2019 | 大规模缩放 | 66.3M | 84.3% |

---

## Vision Transformer (ViT)

2020年，Google 提出 Vision Transformer（ViT），首次证明纯 Transformer 架构可以在图像分类任务上达到 SOTA，开启了视觉 Transformer 的研究热潮。

### 核心思想

将图像分割为固定大小的 Patch，将每个 Patch 视为一个"词"，然后输入标准 Transformer。

```
┌───────────────────────────────────────────────────────────────┐
│                    ViT 处理流程                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  输入图像 (224×224×3)                                         │
│         ↓                                                     │
│  Patch分割 (16×16) → 196个Patch                               │
│         ↓                                                     │
│  Patch Embedding (线性投影) → 196×768                         │
│         ↓                                                     │
│  添加 [CLS] Token + 位置编码                                  │
│         ↓                                                     │
│  Transformer Encoder × L层                                    │
│         ↓                                                     │
│  [CLS] Token → MLP Head → 类别预测                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Patch Embedding

将 $H \times W \times C$ 的图像划分为 $N = \frac{HW}{P^2}$ 个 Patch，每个 Patch 展平后通过线性层投影：

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \ldots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}
$$

其中：
- $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$ 是第 $i$ 个 Patch 展平后的向量
- $\mathbf{E} \in \mathbb{R}^{P^2 \cdot C \times D}$ 是投影矩阵
- $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ 是可学习的位置编码

### Multi-Head Self-Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

对于多头注意力：

$$
\text{MSA}(\mathbf{Z}) = [\text{head}_1; \ldots; \text{head}_h]\mathbf{W}^O
$$

其中 $\text{head}_i = \text{Attention}(\mathbf{Z}\mathbf{W}_i^Q, \mathbf{Z}\mathbf{W}_i^K, \mathbf{Z}\mathbf{W}_i^V)$

### Transformer Encoder

$$
\begin{aligned}
\mathbf{z}'_\ell &= \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1} \\
\mathbf{z}_\ell &= \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
\end{aligned}
$$

```python
class PatchEmbed(nn.Module):
    """图像Patch嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积实现Patch Embedding（更高效）
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """前馈网络"""
    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, 
                 qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop)
        
    def forward(self, x):
        # Pre-norm架构
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer完整实现"""
    
    # 不同规模的配置
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    }
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, variant='base', drop_rate=0.):
        super().__init__()
        config = self.configs[variant]
        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class Token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # 最终LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 分类（使用CLS Token）
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x
    
    def get_features(self, x):
        """提取特征（用于下游任务）"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]  # 返回CLS token特征


def vit_base(num_classes=1000):
    return VisionTransformer(variant='base', num_classes=num_classes)

def vit_large(num_classes=1000):
    return VisionTransformer(variant='large', num_classes=num_classes)
```

### ViT 的关键特性

| 特性 | 说明 |
|------|------|
| **全局感受野** | 每个 Patch 都能直接关注到所有其他 Patch |
| **位置编码** | 弥补 Transformer 缺乏位置信息的缺陷 |
| **CLS Token** | 类似 BERT，用于聚合全局信息 |
| **大规模预训练** | ViT 需要大量数据才能超越 CNN |

---

## Swin Transformer

ViT 虽然强大，但存在两个问题：
1. **计算复杂度**：全局自注意力的复杂度是 $O(N^2)$
2. **缺乏多尺度特征**：不利于检测、分割等下游任务

Swin Transformer 通过**层次化设计**和**滑动窗口注意力**解决了这些问题。

### 核心创新

#### 1. 层次化结构

```
Stage 1: [B, H/4 × W/4, C₁]   → 低分辨率，高语义
Stage 2: [B, H/8 × W/8, C₂]
Stage 3: [B, H/16 × W/16, C₃]
Stage 4: [B, H/32 × W/32, C₄] → 高分辨率，低语义
```

这类似 CNN 的特征金字塔，便于下游任务使用。

#### 2. 滑动窗口注意力

将注意力计算限制在局部窗口内：

$$
\text{Complexity}: O(N^2) \rightarrow O(M^2 \times \frac{N}{M^2}) = O(N)
$$

其中 $M$ 是窗口大小。

**移位窗口（Shifted Window）**：相邻层使用不同的窗口划分，实现跨窗口信息交换。

```
Window Partition (l层):          Shifted Window (l+1层):
┌─────┬─────┐                    ┌──┬────┬──┐
│  0  │  1  │                    │  │    │  │
├─────┼─────┤         →          ├──┼────┼──┤
│  2  │  3  │                    │  │    │  │
└─────┴─────┘                    └──┴────┴──┘
```

#### 3. 相对位置编码

使用相对位置偏置替代绝对位置编码：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V
$$

其中 $B \in \mathbb{R}^{M^2 \times M^2}$ 是可学习的相对位置偏置。

```python
class WindowAttention(nn.Module):
    """窗口自注意力"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 生成相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [num_windows*B, N, C], N = Wh * Ww
            mask: [num_windows, Wh*Ww, Wh*Ww] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 添加mask（用于移位窗口）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
            
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """将特征图划分为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口合并为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 如果窗口大于输入，调整为输入大小
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), 
            num_heads=num_heads, qkv_bias=qkv_bias
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_ratio)
        
        # 计算attention mask
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging(nn.Module):
    """Patch合并层（降采样）"""
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # 2x2合并
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer完整实现"""
    configs = {
        'tiny': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
        'small': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
        'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
        'large': {'embed_dim': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48]},
    }
    
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000,
                 variant='base', window_size=7, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        config = self.configs[variant]
        embed_dim = config['embed_dim']
        depths = config['depths']
        num_heads = config['num_heads']
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        
        # 计算各stage的分辨率
        patches_resolution = img_size // patch_size
        self.layers = nn.ModuleList()
        
        for i_layer in range(len(depths)):
            resolution = patches_resolution // (2 ** i_layer)
            dim = embed_dim * (2 ** i_layer)
            
            # 构建每个stage的blocks
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=(resolution, resolution),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio
                )
                for i in range(depths[i_layer])
            ])
            
            # Patch Merging（最后一个stage不需要）
            if i_layer < len(depths) - 1:
                downsample = PatchMerging((resolution, resolution), dim)
            else:
                downsample = None
            
            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))
        
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            for block in layer['blocks']:
                x = block(x)
            if layer['downsample'] is not None:
                x = layer['downsample'](x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        x = self.head(x)
        return x
    
    def get_features(self, x):
        """提取多尺度特征"""
        features = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            for block in layer['blocks']:
                x = block(x)
            features.append(x)
            if layer['downsample'] is not None:
                x = layer['downsample'](x)
        
        return features


def swin_base(num_classes=1000):
    return SwinTransformer(variant='base', num_classes=num_classes)
```

### Swin vs ViT 对比

| 特性 | ViT | Swin Transformer |
|------|-----|------------------|
| **结构** | 单一尺度 | 层次化 |
| **注意力** | 全局注意力 | 窗口注意力 |
| **计算复杂度** | $O(N^2)$ | $O(N)$ |
| **位置编码** | 绝对位置 | 相对位置 |
| **下游任务** | 需要适配 | 天然支持 |
| **适用场景** | 大规模分类 | 分类、检测、分割 |

---

## ConvNeXt：现代化 CNN

在 ViT 和 Swin 成功后，研究者开始反思：**CNN 真的落后了吗？**

ConvNeXt 通过借鉴 Transformer 的设计技巧，重新设计了纯 CNN 架构，证明了 CNN 依然具有强大竞争力。

### 设计思路

ConvNeXt 采用了"**向 Transformer 学习**"的策略：

```
ResNet → 改进 → ConvNeXt
          ↑
    借鉴 Transformer 设计
```

### 关键改进

| 改进项 | ResNet | ConvNeXt | 灵感来源 |
|--------|--------|----------|----------|
| **激活函数** | ReLU | GELU | Transformer |
| **归一化层** | BatchNorm | LayerNorm | Transformer |
| **块顺序** | Conv-BN-ReLU | LN-Conv-GELU (Pre-norm) | Transformer |
| **卷积核大小** | 3×3 | 7×7 (depthwise) | Swin 窗口大小 |
| **下采样** | stride=2 的卷积 | 独立的 2×2 卷积 | Swin Patch Merging |
| **通道比例** | Bottleneck | 倒瓶颈 (Inverted) | MobileNet |
| **位置编码** | 无 | 无（隐式编码） | 无需显式编码 |

### 核心结构

```python
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 7x7 深度可分离卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # 倒瓶颈结构：1x1升维 -> 1x1降维
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) \
            if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """支持channels_first格式的LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        
        if self.data_format == "channels_last":
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = (normalized_shape, 1, 1)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            return self.weight[:, None, None] * (x - mean) / torch.sqrt(var + self.eps) + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXt(nn.Module):
    """ConvNeXt完整实现"""
    configs = {
        'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]},
    }
    
    def __init__(self, in_channels=3, num_classes=1000, variant='base',
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        config = self.configs[variant]
        depths = config['depths']
        dims = config['dims']
        
        # Stem: 4x4 patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建各stage
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            # 下采样层（除了第一个stage）
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                )
                for j in range(depths[i])
            ])
            cur += depths[i]
            
            self.stages.append(nn.Sequential(downsample, blocks))
        
        # 分类头
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x.mean([-2, -1]))  # Global average pooling
        x = self.head(x)
        return x
    
    def get_features(self, x):
        """提取多尺度特征"""
        features = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def convnext_base(num_classes=1000):
    return ConvNeXt(variant='base', num_classes=num_classes)
```

### ConvNeXt 性能对比

| 模型 | 参数量 | FLOPs | ImageNet Top-1 |
|------|--------|-------|----------------|
| ResNet-50 | 25M | 4.1G | 78.2% |
| ConvNeXt-T | 28M | 4.5G | 82.1% |
| Swin-T | 28M | 4.5G | 81.3% |
| ConvNeXt-B | 89M | 15.4G | 83.8% |
| Swin-B | 88M | 15.4G | 83.5% |

---

## 混合架构

混合架构结合了 CNN 和 Transformer 的优势，在效率和性能之间取得平衡。

### 设计动机

| 架构 | 优势 | 劣势 |
|------|------|------|
| **CNN** | 计算高效、局部特征提取强 | 全局建模能力弱 |
| **Transformer** | 全局建模能力强 | 计算开销大、需大数据 |

**混合策略**：
1. **CNN + Transformer**：CNN 提取低级特征，Transformer 建模全局关系
2. **局部 + 全局**：局部卷积 + 全局注意力
3. **并行融合**：两个分支并行处理

### 典型模型

#### CoAtNet

**核心思想**：卷积和注意力的堆叠组合。

```
Stage 1-2: 纯卷积 (提取局部特征)
Stage 3-4: Transformer (全局建模)
```

#### ConvMixers

使用深度卷积替代 Transformer 的注意力：

```python
class ConvMixerBlock(nn.Module):
    """ConvMixer: 用深度卷积替代注意力"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, x):
        x = x + self.conv1(x)  # 类似残差连接
        x = self.conv2(x)      # 类似FFN
        return x
```

#### MobileViT

为移动端设计的混合架构：

```python
class MobileViTBlock(nn.Module):
    """MobileViT: 轻量级混合块"""
    def __init__(self, in_channels, dim, transformer_depth=2):
        super().__init__()
        # 局部特征提取
        self.local_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        
        # 展平并投影
        self.proj_in = nn.Conv2d(in_channels, dim, 1)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim*2, batch_first=True),
            num_layers=transformer_depth
        )
        
        # 投影回原始通道
        self.proj_out = nn.Conv2d(dim, in_channels, 1)
        
        # 融合
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        
    def forward(self, x):
        # 局部特征
        local_feat = self.local_conv(x)
        
        # 全局特征
        B, C, H, W = x.shape
        global_feat = self.proj_in(x)  # [B, dim, H, W]
        global_feat = global_feat.flatten(2).transpose(1, 2)  # [B, H*W, dim]
        global_feat = self.transformer(global_feat)
        global_feat = global_feat.transpose(1, 2).view(B, -1, H, W)
        global_feat = self.proj_out(global_feat)
        
        # 融合
        return self.fusion(torch.cat([local_feat, global_feat], dim=1))
```

---

## 预训练视觉模型

自监督预训练让视觉模型从海量无标注数据中学习通用表示。

### 掩码自编码器 (MAE)

#### 核心思想

随机遮挡图像的大部分区域（75%），让模型重建缺失部分。

```
┌─────────────────────────────────────────────────────────┐
│                    MAE 架构                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入图像 ──► Patch分割 ──► 随机遮挡75%                   │
│                               │                         │
│                               ▼                         │
│               可见Patch ──► ViT Encoder                  │
│                               │                         │
│                               ▼                         │
│    添加Mask Token ──► 完整序列 ──► 轻量Decoder           │
│                               │                         │
│                               ▼                         │
│                        重建原始图像                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 目标函数

重建损失使用均方误差：

$$
\mathcal{L} = \frac{1}{N_{mask}} \sum_{i \in \mathcal{M}} \| \hat{x}_i - x_i \|^2
$$

只在被遮挡的 Patch 上计算损失。

```python
class MAE(nn.Module):
    """Masked Autoencoder"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, encoder_depth=12, decoder_depth=8,
                 num_heads=12, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim // 2))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim // 2))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim // 2, num_heads // 2)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim // 2)
        
        # 重建头
        self.decoder_pred = nn.Linear(embed_dim // 2, patch_size ** 2 * in_channels)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
    def random_masking(self, x, mask_ratio):
        """随机遮挡"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # 随机打乱
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        
        # 保留的索引
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # 生成mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = x + self.pos_embed[:, 1:, :]
        
        # 随机遮挡
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 添加CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # 投影到decoder维度
        x = self.decoder_embed(x)
        
        # 添加mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # 添加位置编码
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # 预测
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # 移除CLS token
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """计算重建损失"""
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # 只在被遮挡的patch上计算损失
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def patchify(self, imgs):
        """将图像转换为patch"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(1, 2).flatten(2)
        return x
    
    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
```

### BEiT

BEiT (BERT pre-training of Image Transformers) 借鉴 BERT 的掩码语言模型思想。

**核心区别**：
- 不预测原始像素，而是预测离散的视觉词元（Visual Tokens）
- 使用 dVAE 将图像 Patch 编码为离散编码

$$
\mathcal{L} = -\sum_{i \in \mathcal{M}} \log p(z_i | x_{\backslash \mathcal{M}})
$$

其中 $z_i$ 是第 $i$ 个 Patch 的视觉词元。

### DINO / DINOv2

DINO (DIstillation with NO labels) 使用**自蒸馏**方法训练视觉模型。

**核心思想**：
- 两个网络：Teacher 和 Student
- 同一图像的两个不同增强视图
- Student 看全局视图，Teacher 看局部视图
- Teacher 参数是 Student 的指数移动平均

```python
class DINO(nn.Module):
    """DINO自蒸馏框架（简化版）"""
    def __init__(self, backbone, dim=256, num_prototypes=65536):
        super().__init__()
        self.student = backbone
        self.teacher = copy.deepcopy(backbone)
        
        # 投影头
        self.student_head = DINOHead(dim, num_prototypes)
        self.teacher_head = DINOHead(dim, num_prototypes)
        
        # 冻结teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
            
    def update_teacher(self, momentum=0.996):
        """指数移动平均更新teacher"""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data


class DINOHead(nn.Module):
    """DINO投影头"""
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x
```

### 预训练方法对比

| 方法 | 预训练目标 | 特点 | 下游迁移 |
|------|-----------|------|----------|
| **MAE** | 像素重建 | 高遮挡率，高效 | 分类、检测 |
| **BEiT** | 视觉词元预测 | 离散化，类BERT | 分类、分割 |
| **DINO** | 自蒸馏 | 无需掩码，学习语义 | 分类、分割 |
| **DINOv2** | 改进自蒸馏 | 更大数据，更好架构 | 全能视觉骨干 |
| **CLIP** | 图像-文本对齐 | 多模态，零样本 | 零样本迁移 |

---

## 视觉模型选择指南

### 根据任务选择

| 任务 | 推荐模型 | 原因 |
|------|----------|------|
| **图像分类** | ResNet/EfficientNet/ConvNeXt | 成熟稳定，部署友好 |
| **目标检测** | ResNet/Swin/ConvNeXt | 多尺度特征支持 |
| **语义分割** | Swin/ConvNeXt | 层次化特征，分辨率保持 |
| **医学图像** | ResNet/U-Net | 小数据友好，可解释 |
| **移动端部署** | EfficientNet/MobileViT | 参数少，计算高效 |

### 根据资源选择

| 计算资源 | 推荐模型 | 说明 |
|----------|----------|------|
| **CPU/边缘设备** | MobileNet/EfficientNet-B0 | 轻量高效 |
| **单GPU** | ResNet-50/ConvNeXt-T | 平衡性能与效率 |
| **多GPU** | Swin-B/ConvNeXt-B | 更强性能 |
| **大规模集群** | ViT-L/H, MAE预训练 | 追求极致性能 |

### 根据数据规模选择

| 数据规模 | 推荐策略 |
|----------|----------|
| **小数据（<10K）** | 使用预训练模型微调，选择 ResNet/ConvNeXt |
| **中等数据（10K-1M）** | 使用预训练模型微调，可尝试 ViT |
| **大数据（>1M）** | 可从头训练，ViT + MAE 预训练效果最佳 |

### 实用代码：使用 timm 加载预训练模型

```python
import timm

# 查看可用模型
print(timm.list_models('*resnet*'))

# 加载预训练模型
model = timm.create_model('resnet50', pretrained=True)

# 加载模型并修改分类头
model = timm.create_model(
    'convnext_base',
    pretrained=True,
    num_classes=10  # 自定义类别数
)

# 提取特征
features = model.forward_features(input_tensor)

# 常用模型推荐
recommended_models = {
    # 经典CNN
    'resnet50': '经典残差网络，稳定可靠',
    'resnet101': '更深层的ResNet',
    
    # 现代CNN
    'efficientnet_b0': '轻量高效',
    'efficientnet_b4': '平衡性能与效率',
    'convnext_base': '现代化CNN，SOTA性能',
    
    # Vision Transformer
    'vit_base_patch16_224': '标准ViT',
    'vit_large_patch16_224': '大型ViT',
    
    # Swin Transformer
    'swin_base_patch4_window7_224': '层次化Transformer',
    
    # 预训练模型
    'vit_base_patch16_224.mae': 'MAE预训练',
    'dinov2_base': 'DINOv2预训练',
}
```

---

## 知识点关联

```
┌─────────────────────────────────────────────────────────────────────┐
│                      视觉模型知识图谱                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CNN 演进线                        Transformer 演进线               │
│  ──────────                        ──────────────────               │
│       │                                   │                        │
│  AlexNet ─── VGG ─── ResNet ───┐    ViT ───┼── Swin                │
│                    │          │           │                        │
│                DenseNet       │           DeiT                     │
│                    │          │                                    │
│               EfficientNet    │        混合架构                     │
│                    │          │        ────────                    │
│                ConvNeXt ◄─────┴──────► CoAtNet                      │
│                              │           MobileViT                  │
│                              │                                    │
│                      预训练技术                                      │
│                      ──────────                                    │
│                              │                                    │
│                    MAE ◄─────┼─────► BEiT                         │
│                              │                                    │
│                    DINO ◄────┴─────► CLIP                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 与其他章节的关系

| 章节 | 关系 | 延伸阅读 |
|------|------|----------|
| **CNN 基础** | 视觉模型的基石 | [深度学习-CNN](../deep-learning/03-cnn.md) |
| **Transformer** | ViT 的核心机制 | [深度学习-Transformer](../deep-learning/05-transformer.md) |
| **目标检测** | 骨干网络应用 | [目标检测](./object-detection.md) |
| **多模态模型** | 视觉编码器 | [多模态模型](../multimodal/index.md) |
| **模型微调** | 预训练模型应用 | [微调与对齐](../finetuning-alignment.md) |

---

## 核心考点

### 理论考点

1. **ResNet 残差连接**
   - 为什么残差连接能解决退化问题？
   - 梯度如何通过跳跃连接传播？

2. **ViT vs CNN**
   - 归纳偏置（Inductive Bias）的差异
   - 计算复杂度对比
   - 数据需求差异

3. **Swin Transformer**
   - 滑动窗口注意力如何工作？
   - 相对位置编码的作用
   - 与 ViT 的复杂度对比

4. **预训练方法**
   - MAE 为什么使用高遮挡率？
   - DINO 的自蒸馏机制
   - 不同预训练方法的适用场景

### 实践考点

1. **模型选择**
   - 根据任务、数据、资源选择合适模型
   - 迁移学习策略

2. **代码实现**
   - 实现 ResNet BasicBlock
   - 实现 Patch Embedding
   - 实现窗口注意力

3. **调优技巧**
   - 学习率调度策略
   - 数据增强方法
   - 正则化技术

### 常见面试题

**Q1: 为什么 ResNet 能训练很深的网络？**

> 残差连接允许梯度直接流向前面的层，缓解了梯度消失问题。同时，如果恒等映射是最优解，残差块只需学习零映射即可。

**Q2: ViT 为什么需要大量数据训练？**

> ViT 缺乏 CNN 的归纳偏置（局部性、平移不变性），需要从数据中学习这些特性。在小数据集上，CNN 通常表现更好。

**Q3: Swin Transformer 如何降低计算复杂度？**

> 通过将全局注意力限制在局部窗口内，复杂度从 $O(N^2)$ 降为 $O(N)$。移位窗口机制允许跨窗口信息交换。

**Q4: ConvNeXt 为什么能超越 Swin？**

> ConvNeXt 借鉴了 Transformer 的设计技巧（大卷积核、LayerNorm、GELU 等），同时保持了 CNN 的计算效率。证明了 CNN 架构还有优化空间。

---

## 学习建议

### 学习路径

```
第1阶段：CNN 基础（2周）
├── 掌握卷积、池化、感受野等概念
├── 理解 ResNet 残差连接
└── 实践：用 timm 加载预训练模型

第2阶段：CNN 进阶（2周）
├── DenseNet 密集连接
├── EfficientNet 复合缩放
└── 实践：训练自定义数据集

第3阶段：Vision Transformer（3周）
├── 理解 Patch Embedding
├── 掌握多头注意力机制
├── 学习 Swin 的滑动窗口
└── 实践：实现 ViT 并训练

第4阶段：预训练方法（2周）
├── MAE 掩码建模
├── DINO 自蒸馏
└── 实践：使用预训练模型微调

第5阶段：实战应用（持续）
├── 根据场景选择模型
├── 调参与优化
└── 部署与加速
```

### 推荐资源

**必读论文**：
1. Deep Residual Learning (ResNet) - CVPR 2016
2. An Image is Worth 16x16 Words (ViT) - ICLR 2021
3. Swin Transformer - ICCV 2021
4. A ConvNet for the 2020s (ConvNeXt) - CVPR 2022
5. Masked Autoencoders (MAE) - CVPR 2022

**开源资源**：
- **timm**：PyTorch 图像模型库，包含数百个预训练模型
- **torchvision**：官方视觉库，经典模型实现
- **transformers**：HuggingFace，支持 ViT、Swin 等

### 实践建议

1. **从经典开始**：先理解 ResNet，再学习 Transformer
2. **动手实现**：亲自实现关键组件（残差块、Patch Embedding）
3. **使用预训练**：实践中优先使用预训练模型微调
4. **可视化特征**：理解模型学到了什么
5. **对比实验**：相同任务对比不同模型的效果

---

视觉模型的演进体现了深度学习领域的创新活力。从 CNN 到 Transformer，从监督学习到自监督预训练，每一步都在推动计算机视觉能力边界的扩展。理解这些模型的设计思想，掌握选择和使用的技巧，是成为优秀视觉算法工程师的必备素质。
