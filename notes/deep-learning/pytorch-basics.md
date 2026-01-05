# PyTorch基础入门

## 章节概述

PyTorch是由Facebook开发的深度学习框架，以其**动态计算图、直观的Python接口和强大的GPU支持**而闻名。本章将带领您从零开始掌握PyTorch的基本概念、安装配置和核心特性。

## PyTorch核心特性

### 动态计算图（动态图）
- **运行时构建**：计算图在代码执行时动态构建
- **灵活调试**：可以像普通Python代码一样调试
- **条件控制**：支持if/for/while等Python控制流
- **直观理解**：更符合程序员的思维习惯

### Python优先设计
- **无缝集成**：与NumPy、SciPy等Python科学计算库完美配合
- **直观API**：API设计符合Python惯例，学习成本低
- **丰富生态**：可以方便地使用Python的各种工具和库

### 强大的GPU加速
- **CUDA支持**：自动利用NVIDIA GPU进行加速计算
- **设备管理**：简单的设备切换和内存管理
- **分布式训练**：支持多GPU和多机分布式训练

## 安装与环境配置

### 安装方法
```bash
# 使用conda安装（推荐）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 使用pip安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 环境验证
```python
import torch
import torchvision

print(f"PyTorch版本: {torch.__version__}")
print(f"TorchVision版本: {torchvision.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

### 设备管理
```python
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 多GPU设置
if torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 个GPU")
    # 数据并行（后面会详细介绍）
    # model = nn.DataParallel(model)
```

## PyTorch核心概念

### 张量（Tensor）
张量是PyTorch中最基本的数据结构，类似于NumPy的数组，但支持GPU加速和自动微分。

```python
import torch

# 张量基础
print("=== 张量基础 ===")

# 创建标量（0维张量）
scalar = torch.tensor(3.1415)
print(f"标量: {scalar}, 形状: {scalar.shape}")

# 创建向量（1维张量）
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"向量: {vector}, 形状: {vector.shape}")

# 创建矩阵（2维张量）
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"矩阵: {matrix}")
print(f"矩阵形状: {matrix.shape}")

# 创建3维张量
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D张量形状: {tensor_3d.shape}")
```

### 自动微分（Autograd）
PyTorch的自动微分系统可以自动计算梯度，这是深度学习训练的核心。

```python
print("\n=== 自动微分 ===")

# 启用梯度跟踪
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 前向传播
y = w * x + b
print(f"y = {y}")

# 反向传播（计算梯度）
y.backward()

# 查看梯度
print(f"dy/dx = {x.grad}")  # 应该是3.0 (w的值)
print(f"dy/dw = {w.grad}")  # 应该是2.0 (x的值)
print(f"dy/db = {b.grad}")  # 应该是1.0
```

### 计算图（Computational Graph）
PyTorch自动构建计算图来跟踪操作，用于梯度计算。

```python
print("\n=== 计算图 ===")

# 构建更复杂的计算图
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x**2 + y**3
out = torch.sin(z)

print(f"计算图: x → x² → x² + y³ → z → sin(z) → out")
print(f"中间结果: z = {z}, out = {out}")

# 反向传播
out.backward()

print(f"d(out)/dx = {x.grad}")
print(f"d(out)/dy = {y.grad}")
```

## PyTorch与NumPy的互操作性

PyTorch与NumPy可以无缝转换，方便利用现有的NumPy生态。

```python
import numpy as np

print("\n=== PyTorch与NumPy互操作 ===")

# NumPy数组转PyTorch张量
np_array = np.array([[1, 2], [3, 4]])
torch_tensor = torch.from_numpy(np_array)
print(f"NumPy数组: {np_array}")
print(f"PyTorch张量: {torch_tensor}")

# PyTorch张量转NumPy数组
torch_tensor_2 = torch.tensor([[5, 6], [7, 8]])
np_array_2 = torch_tensor_2.numpy()
print(f"PyTorch张量: {torch_tensor_2}")
print(f"NumPy数组: {np_array_2}")

# 注意：共享内存（修改一个会影响另一个）
print("\n=== 内存共享测试 ===")
torch_tensor[0, 0] = 100
print(f"修改后NumPy数组: {np_array}")  # 也会被修改
```

## 第一个PyTorch程序：线性回归

让我们用一个简单的线性回归例子来体验PyTorch的完整工作流程。

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("\n=== 线性回归示例 ===")

# 生成模拟数据
torch.manual_seed(42)  # 设置随机种子确保可重复性

# 真实参数
true_w = 2.0
true_b = 1.0

# 生成训练数据
X = torch.linspace(0, 10, 100).reshape(-1, 1)
Y = true_w * X + true_b + torch.randn(X.shape) * 0.5

print(f"数据形状: X {X.shape}, Y {Y.shape}")

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入1维，输出1维
    
    def forward(self, x):
        return self.linear(x)

# 创建模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

print(f"模型参数: {list(model.parameters())}")

# 训练模型
epochs = 100
losses = []

for epoch in range(epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, Y)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 查看训练后的参数
final_w = model.linear.weight.item()
final_b = model.linear.bias.item()
print(f"\n真实参数: w={true_w}, b={true_b}")
print(f"学习参数: w={final_w:.2f}, b={final_b:.2f}")

# 可视化结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), Y.numpy(), alpha=0.7, label='数据点')
plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', label='拟合直线')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('线性回归拟合')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.tight_layout()
plt.show()
```

## PyTorch项目结构最佳实践

### 典型的PyTorch项目结构
```
project/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── datasets.py         # 数据集定义
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── base_model.py       # 基础模型类
│   ├── cnn.py             # CNN模型
│   └── rnn.py             # RNN模型
├── utils/                  # 工具函数
│   ├── logger.py          # 日志工具
│   ├── metrics.py         # 评估指标
│   └── visualization.py   # 可视化工具
├── configs/               # 配置文件
│   └── default.yaml
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── inference.py          # 推理脚本
└── requirements.txt      # 依赖包
```

### 基础模型类模板
```python
import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):
    """基础模型类，提供常用功能"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("子类必须实现forward方法")
    
    def configure_optimizers(self, lr=0.001):
        """配置优化器"""
        return optim.Adam(self.parameters(), lr=lr)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.__dict__
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
```

## 常见问题与解决方案

### 问题1：CUDA内存不足
```python
# 解决方案：减少批量大小或使用梯度累积
batch_size = 32  # 减少批量大小

# 或者使用梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

### 问题2：梯度爆炸
```python
# 解决方案：梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 问题3：模型过拟合
```python
# 解决方案：添加正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2正则化
model = nn.Dropout(0.5)  # Dropout正则化
```

## 学习资源推荐

### 官方资源
- **PyTorch官方文档**：https://pytorch.org/docs/stable/index.html
- **PyTorch教程**：https://pytorch.org/tutorials/
- **PyTorch例子**：https://github.com/pytorch/examples

### 在线课程
- **PyTorch官方教程**：全面覆盖基础到高级主题
- **Fast.ai课程**：实践导向的深度学习课程
- **Coursera深度学习专项课程**：系统学习深度学习

### 书籍推荐
- 《深度学习入门之PyTorch》
- 《PyTorch深度学习实战》
- 《动手学深度学习》（PyTorch版）

## 总结

本章介绍了PyTorch的基础概念、安装配置和核心特性。关键要点包括：

1. **动态计算图**是PyTorch的核心优势，提供灵活的调试和控制
2. **张量**是基本数据结构，支持GPU加速和自动微分
3. **自动微分系统**自动计算梯度，简化反向传播
4. **与NumPy无缝集成**，便于利用现有生态
5. **完整的深度学习工作流程**：从数据准备到模型训练

在后续章节中，我们将深入探讨PyTorch的各个模块和高级特性。