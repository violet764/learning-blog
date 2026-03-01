# Convolutional Neural Networks (CNN)

## 1. 生物学启发与视觉神经科学

### 1.1 视觉皮层的层次结构
CNN的设计灵感来源于大脑视觉皮层的层次化处理机制。Hubel和Wiesel的诺贝尔奖研究揭示了：
- **简单细胞**：响应特定方向的边缘（对应卷积核）
- **复杂细胞**：响应特定位置不变的特征（对应池化层）
- **超复杂细胞**：组合简单特征形成复杂模式（对应深层卷积）

### 1.2 感受野机制
视觉神经元只处理视网膜的局部区域（感受野），这种局部连接机制直接启发了卷积操作。

## 2. 数学基础与卷积操作

### 2.1 离散卷积公式

$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \cdot g[n-m]$$

在图像处理中，离散卷积表示为：

$$O(i,j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m,n)$$

其中$I$是输入图像，$K$是卷积核。

### 2.2 卷积层前向传播

设输入特征图$X \in \mathbb{R}^{H \times W \times C_{in}}$，卷积核$W \in \mathbb{R}^{K \times K \times C_{in} \times C_{out}}$，偏置$b \in \mathbb{R}^{C_{out}}$：

$$Y_{:,:,k} = \sum_{c=1}^{C_{in}} X_{:,:,c} * W_{:,:,c,k} + b_k$$

### 2.3 卷积层梯度推导

损失函数$L$对卷积核$W$的梯度：

$$\frac{\partial L}{\partial W_{m,n,c,k}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j,k}} \cdot X_{i+m, j+n, c}$$

这实际上是输入特征图与输出梯度的卷积操作。

### 2.4 池化层数学分析

**最大池化**：
$$Y_{i,j} = \max_{m,n \in \text{window}} X_{i\cdot s + m, j\cdot s + n}$$

**平均池化**：
$$Y_{i,j} = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i\cdot s + m, j\cdot s + n}$$

## 3. CNN架构演进与可视化分析

### 3.1 LeNet-5 (1998) - 里程碑架构
```
Input(32×32×1) → Conv(5×5,6) → Pool(2×2) → Conv(5×5,16) → Pool(2×2) → FC(120) → FC(84) → Output(10)
```

### 3.2 AlexNet (2012) - 深度学习复兴
```
Input(227×227×3) → Conv(11×11,96,s=4) → Pool(3×3,s=2) → Conv(5×5,256) → Pool(3×3,s=2) 
→ Conv(3×3,384) → Conv(3×3,384) → Conv(3×3,256) → Pool(3×3,s=2) → FC(4096) → FC(4096) → Output(1000)
```

### 3.3 VGGNet (2014) - 深度探索
```
VGG16: 13Conv + 3FC = 16层
VGG19: 16Conv + 3FC = 19层

特点：3×3小卷积核堆叠，感受野等效于大卷积核但参数更少
```

### 3.4 ResNet (2015) - 残差学习
```
基本残差块：
Input → Conv(3×3,64) → BN → ReLU → Conv(3×3,64) → BN → +Input → ReLU

残差连接解决梯度消失：F(x) + x
```

## 4. PyTorch完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class BasicConvBlock(nn.Module):
    """基础卷积块：Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = BasicConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 下采样残差连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)

class CustomCNN(nn.Module):
    """自定义CNN架构，包含多种现代技术"""
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = BasicConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 残差块堆叠
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 2个残差块
        self.layer2 = self._make_layer(64, 128, 2, stride=2) # 下采样
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化 + 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # 第一个块处理下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 剩余块保持相同维度
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        x = self.conv1(x)      # /2
        x = self.maxpool(x)    # /2
        
        x = self.layer1(x)     # 保持尺寸
        x = self.layer2(x)     # /2
        x = self.layer3(x)     # /2
        x = self.layer4(x)     # /2
        
        # 分类
        x = self.avgpool(x)    # 全局平均池化
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class FeatureVisualizer:
    """特征可视化工具"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        
        # 注册钩子捕获中间层输出
        self._register_hooks()
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # 为各层注册钩子
        self.model.conv1.register_forward_hook(get_activation('conv1'))
        self.model.layer1.register_forward_hook(get_activation('layer1'))
        self.model.layer2.register_forward_hook(get_activation('layer2'))
        self.model.layer3.register_forward_hook(get_activation('layer3'))
        self.model.layer4.register_forward_hook(get_activation('layer4'))
    
    def visualize_features(self, image_tensor, layer_name, num_filters=16):
        """可视化指定层的特征图"""
        if layer_name not in self.activations:
            print(f"层 {layer_name} 未找到")
            return
        
        features = self.activations[layer_name][0]  # 取第一个样本
        num_channels = min(features.size(0), num_filters)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_channels):
            feature_map = features[i].cpu().numpy()
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'通道 {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = CustomCNN(num_classes=10)
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2, 224×224 RGB图像
    output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 特征可视化示例
    visualizer = FeatureVisualizer(model)
    
    # 再次前向传播以捕获激活
    _ = model(dummy_input)
    
    print("\n可用层:", list(visualizer.activations.keys()))
    
    # 可视化第一层特征
    # visualizer.visualize_features(dummy_input, 'conv1')
```

## 5. 现代CNN架构技术

### 5.1 深度可分离卷积
将标准卷积分解为深度卷积和逐点卷积：
- **参数减少**：从$K^2 \times C_{in} \times C_{out}$到$K^2 \times C_{in} + C_{in} \times C_{out}$
- **计算效率**：MobileNet、Xception等架构的基础

### 5.2 空洞卷积（Dilated Convolution）
扩大感受野而不增加参数：
$$O(i,j) = \sum_{m,n} I(i + r \cdot m, j + r \cdot n) \cdot K(m,n)$$

### 5.3 注意力机制集成
- **SENet**：通道注意力
- **CBAM**：通道+空间注意力
- **Non-local**：自注意力机制

## 6. 训练技巧与优化

### 6.1 数据增强策略
```python
# PyTorch数据增强示例
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 6.2 学习率调度
- **StepLR**：固定步长衰减
- **CosineAnnealing**：余弦退火
- **OneCycleLR**：单周期策略

## 7. 延伸学习

### 7.1 核心论文
1. **[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)** - LeNet-5
2. **[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)** - AlexNet
3. **[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)** - VGGNet
4. **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)** - ResNet

### 7.2 开源项目
1. **[PyTorch Vision Models](https://github.com/pytorch/vision)** - 官方预训练模型
2. **[TensorFlow Models](https://github.com/tensorflow/models)** - TensorFlow模型库
3. **[MMDetection](https://github.com/open-mmlab/mmdetection)** - 目标检测工具箱

### 7.3 应用领域
1. **图像分类**：ImageNet挑战
2. **目标检测**：YOLO、Faster R-CNN
3. **语义分割**：U-Net、DeepLab
4. **风格迁移**：Neural Style Transfer
5. **超分辨率**：SRCNN、ESRGAN

## 8. 架构选择指南

### 8.1 根据任务复杂度选择
- **简单任务**：LeNet、小型VGG
- **中等任务**：ResNet-18、MobileNet
- **复杂任务**：ResNet-50/101、EfficientNet

### 8.2 根据资源约束选择
- **计算受限**：MobileNet、ShuffleNet
- **内存受限**：SqueezeNet、EfficientNet-B0
- **精度优先**：ResNet-152、Vision Transformer

### 8.3 迁移学习策略
- **特征提取**：冻结卷积层，只训练分类器
- **微调**：解冻部分层进行端到端训练
- **领域适应**：在目标领域数据上继续训练