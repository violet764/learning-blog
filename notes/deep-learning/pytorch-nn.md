# PyTorch神经网络模块详解

## 章节概述

`torch.nn`模块是PyTorch深度学习的核心，提供了构建神经网络所需的各种层、损失函数和工具。本章将深入讲解神经网络模块的各个组件，包括层定义、模型构建、参数管理和前向传播机制。

## 神经网络基础模块

### 基础层类型

#### 线性层（全连接层）
```python
import torch
import torch.nn as nn

print("=== 线性层（全连接层） ===")

# 创建线性层
linear = nn.Linear(in_features=10, out_features=5)
print(f"线性层权重形状: {linear.weight.shape}")  # [5, 10]
print(f"线性层偏置形状: {linear.bias.shape}")    # [5]

# 前向传播示例
input_tensor = torch.randn(32, 10)  # 批次大小32，特征维度10
output = linear(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output.shape}")  # [32, 5]

# 数学公式: output = input @ weight.T + bias
```

#### 卷积层
```python
print("\n=== 卷积层 ===")

# 1D卷积（用于序列数据）
conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
input_1d = torch.randn(8, 16, 100)  # [批次, 通道, 序列长度]
output_1d = conv1d(input_1d)
print(f"1D卷积输入形状: {input_1d.shape}")
print(f"1D卷积输出形状: {output_1d.shape}")

# 2D卷积（用于图像数据）
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
input_2d = torch.randn(4, 3, 224, 224)  # [批次, 通道, 高, 宽]
output_2d = conv2d(input_2d)
print(f"2D卷积输入形状: {input_2d.shape}")
print(f"2D卷积输出形状: {output_2d.shape}")

# 3D卷积（用于体积数据）
conv3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
input_3d = torch.randn(2, 1, 32, 32, 32)  # [批次, 通道, 深, 高, 宽]
output_3d = conv3d(input_3d)
print(f"3D卷积输入形状: {input_3d.shape}")
print(f"3D卷积输出形状: {output_3d.shape}")
```

#### 池化层
```python
print("\n=== 池化层 ===")

# 最大池化
max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
input_pool = torch.randn(1, 64, 32, 32)
output_max = max_pool_2d(input_pool)
print(f"最大池化输入: {input_pool.shape}")
print(f"最大池化输出: {output_max.shape}")

# 平均池化
avg_pool_2d = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg = avg_pool_2d(input_pool)
print(f"平均池化输出: {output_avg.shape}")

# 自适应池化（自动调整输出大小）
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
input_adaptive = torch.randn(1, 64, 32, 32)
output_adaptive = adaptive_pool(input_adaptive)
print(f"自适应池化输入: {input_adaptive.shape}")
print(f"自适应池化输出: {output_adaptive.shape}")
```

#### 循环神经网络层
```python
print("\n=== 循环神经网络层 ===")

# RNN层
rnn = nn.RNN(input_size=100, hidden_size=50, num_layers=2, batch_first=True)
input_rnn = torch.randn(16, 10, 100)  # [批次, 序列长度, 特征维度]
output_rnn, hidden_rnn = rnn(input_rnn)
print(f"RNN输入形状: {input_rnn.shape}")
print(f"RNN输出形状: {output_rnn.shape}")  # [16, 10, 50]
print(f"RNN隐藏状态形状: {hidden_rnn.shape}")  # [2, 16, 50]

# LSTM层
lstm = nn.LSTM(input_size=100, hidden_size=50, num_layers=2, batch_first=True)
output_lstm, (hidden_lstm, cell_lstm) = lstm(input_rnn)
print(f"LSTM输出形状: {output_lstm.shape}")
print(f"LSTM隐藏状态形状: {hidden_lstm.shape}")
print(f"LSTM细胞状态形状: {cell_lstm.shape}")

# GRU层
gru = nn.GRU(input_size=100, hidden_size=50, num_layers=2, batch_first=True)
output_gru, hidden_gru = gru(input_rnn)
print(f"GRU输出形状: {output_gru.shape}")
```

### 激活函数
```python
print("\n=== 激活函数 ===")

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU激活函数
relu = nn.ReLU()
print(f"ReLU输出: {relu(x)}")  # [0., 0., 0., 1., 2.]

# Sigmoid激活函数
sigmoid = nn.Sigmoid()
print(f"Sigmoid输出: {sigmoid(x)}")

# Tanh激活函数
tanh = nn.Tanh()
print(f"Tanh输出: {tanh(x)}")

# LeakyReLU（避免神经元死亡）
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
print(f"LeakyReLU输出: {leaky_relu(x)}")

# Softmax（用于多分类）
softmax = nn.Softmax(dim=0)
print(f"Softmax输出: {softmax(x)}")
print(f"Softmax总和: {softmax(x).sum()}")  # 应该为1.0
```

### 归一化层
```python
print("\n=== 归一化层 ===")

# 批归一化（Batch Normalization）
batchnorm = nn.BatchNorm2d(num_features=64)
input_bn = torch.randn(16, 64, 32, 32)
output_bn = batchnorm(input_bn)
print(f"批归一化输入: {input_bn.shape}")
print(f"批归一化输出: {output_bn.shape}")

# 层归一化（Layer Normalization）
layernorm = nn.LayerNorm(normalized_shape=[64, 32, 32])
output_ln = layernorm(input_bn)
print(f"层归一化输出: {output_ln.shape}")

# 实例归一化（Instance Normalization）
instancenorm = nn.InstanceNorm2d(num_features=64)
output_in = instancenorm(input_bn)
print(f"实例归一化输出: {output_in.shape}")

# 组归一化（Group Normalization）
groupnorm = nn.GroupNorm(num_groups=8, num_channels=64)
output_gn = groupnorm(input_bn)
print(f"组归一化输出: {output_gn.shape}")
```

### 丢弃层（Dropout）
```python
print("\n=== 丢弃层（Dropout） ===")

# Dropout层
dropout = nn.Dropout(p=0.5)
input_dropout = torch.randn(4, 10)

# 训练模式
dropout.train()
output_train = dropout(input_dropout)
print(f"训练模式Dropout输出: {output_train}")

# 评估模式
dropout.eval()
output_eval = dropout(input_dropout)
print(f"评估模式Dropout输出: {output_eval}")
print(f"评估模式下是否相等: {torch.allclose(input_dropout, output_eval)}")
```

## 自定义神经网络模块

### 基础模块定义
```python
print("\n=== 自定义神经网络模块 ===")

class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# 使用自定义模块
model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)
input_data = torch.randn(32, 784)  # 模拟MNIST数据
output = model(input_data)
print(f"MLP输入形状: {input_data.shape}")
print(f"MLP输出形状: {output.shape}")
```

### 复杂模块设计
```python
print("\n=== 复杂模块设计 ===")

class ResidualBlock(nn.Module):
    """残差块（ResNet中的基本单元）"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接（如果维度不匹配）
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        identity = self.skip_connection(identity)
        
        # 相加并激活
        out += identity
        out = self.relu(out)
        
        return out

# 使用残差块
res_block = ResidualBlock(64, 128, stride=2)
input_res = torch.randn(4, 64, 32, 32)
output_res = res_block(input_res)
print(f"残差块输入: {input_res.shape}")
print(f"残差块输出: {output_res.shape}")
```

### 模块组合与序列
```python
print("\n=== 模块组合 ===")

# 使用nn.Sequential组合模块
sequential_model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

input_seq = torch.randn(4, 3, 32, 32)
output_seq = sequential_model(input_seq)
print(f"序列模型输入: {input_seq.shape}")
print(f"序列模型输出: {output_seq.shape}")

# 按名称访问子模块
print(f"第一个卷积层: {sequential_model[0]}")
print(f"第二个卷积层权重形状: {sequential_model[3].weight.shape}")
```

## 模型参数管理

### 参数访问与修改
```python
print("\n=== 模型参数管理 ===")

# 访问所有参数
print("所有参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 获取参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数: {total_params}")
print(f"可训练参数数: {trainable_params}")

# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
print("参数初始化完成")

# 冻结特定层
for param in model.fc1.parameters():
    param.requires_grad = False

print(f"fc1层是否可训练: {not all(not p.requires_grad for p in model.fc1.parameters())}")
```

### 参数保存与加载
```python
print("\n=== 参数保存与加载 ===")

# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')
print("模型参数已保存")

# 加载模型参数
new_model = SimpleMLP(784, 128, 10)
new_model.load_state_dict(torch.load('model_weights.pth'))
new_model.eval()  # 设置为评估模式
print("模型参数已加载")

# 保存完整模型（包含架构）
torch.save(model, 'complete_model.pth')
loaded_model = torch.load('complete_model.pth')
print("完整模型已保存和加载")
```

## 损失函数详解

### 常用损失函数
```python
print("\n=== 损失函数 ===")

# 回归任务损失
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

# 分类任务损失
ce_loss = nn.CrossEntropyLoss()  # 多分类
bce_loss = nn.BCEWithLogitsLoss()  # 二分类（包含sigmoid）

# 使用示例
predictions = torch.randn(4, 10)  # 4个样本，10个类别
targets = torch.tensor([3, 5, 1, 7])  # 真实标签

# 多分类损失
ce_value = ce_loss(predictions, targets)
print(f"交叉熵损失: {ce_value:.4f}")

# 二分类示例
binary_pred = torch.randn(4, 1)
binary_target = torch.tensor([1., 0., 1., 0.]).unsqueeze(1)
bce_value = bce_loss(binary_pred, binary_target)
print(f"二分类交叉熵损失: {bce_value:.4f}")
```

### 自定义损失函数
```python
print("\n=== 自定义损失函数 ===")

class FocalLoss(nn.Module):
    """Focal Loss（用于处理类别不平衡）"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用自定义损失函数
focal_loss = FocalLoss()
focal_value = focal_loss(predictions, targets)
print(f"Focal Loss: {focal_value:.4f}")
```

## 实用工具函数

### 模型工具
```python
print("\n=== 实用工具函数 ===")

# 模型摘要
from torchsummary import summary
# summary(model, input_size=(3, 224, 224))  # 需要安装torchsummary

# 计算FLOPs（浮点运算次数）
def count_flops(model, input_size):
    """估算模型的FLOPs"""
    # 简化实现，实际可以使用thop等库
    input_tensor = torch.randn(*input_size)
    model.eval()
    
    # 这里可以使用更精确的FLOPs计算库
    return "需要安装专门的计算库"

# 梯度裁剪
def train_with_gradient_clipping(model, dataloader, optimizer, max_norm=1.0):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 前向传播和损失计算
        # ...
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()

print("工具函数定义完成")
```

## 实际应用示例

### 图像分类模型
```python
print("\n=== 图像分类模型示例 ===")

class CNNClassifier(nn.Module):
    """CNN图像分类器"""
    
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 使用示例
cnn_model = CNNClassifier(num_classes=10)
input_image = torch.randn(8, 3, 32, 32)  # CIFAR-10尺寸
output = cnn_model(input_image)
print(f"CNN分类器输入: {input_image.shape}")
print(f"CNN分类器输出: {output.shape}")
```

### 文本分类模型
```python
print("\n=== 文本分类模型示例 ===")

class TextClassifier(nn.Module):
    """LSTM文本分类器"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, emb_dim]
        
        # 打包序列以提高效率
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 处理双向LSTM的最终隐藏状态
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

# 使用示例（简化）
text_model = TextClassifier(
    vocab_size=10000, embedding_dim=100, hidden_dim=256, 
    output_dim=2, n_layers=2, bidirectional=True, dropout=0.5
)
print("文本分类器创建完成")
```

## 模型调试与可视化

### 模型结构可视化
```python
print("\n=== 模型调试 ===")

# 打印模型结构
print("模型结构:")
print(model)

# 检查中间层输出
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} 输出形状: {output.shape}")

# 注册钩子
hook_handle = model.fc1.register_forward_hook(hook_fn)

# 前向传播（触发钩子）
test_input = torch.randn(1, 784)
with torch.no_grad():
    model(test_input)

# 移除钩子
hook_handle.remove()

# 梯度检查
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name} 梯度范数: {grad_norm:.6f}")
        else:
            print(f"{name} 无梯度")

print("梯度检查完成")
```

## 总结

本章详细介绍了PyTorch神经网络模块的各个方面：

1. **基础层类型**：线性层、卷积层、池化层、RNN层等
2. **激活函数与归一化**：ReLU、Sigmoid、批归一化等
3. **自定义模块设计**：从简单MLP到复杂残差块
4. **参数管理**：初始化、冻结、保存加载
5. **损失函数**：内置损失和自定义损失
6. **实用工具**：模型摘要、梯度裁剪等
7. **实际应用**：图像分类和文本分类模型

掌握`torch.nn`模块是构建深度学习模型的关键，建议通过实际项目练习这些组件，加深理解。