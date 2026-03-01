# 神经网络基础

神经网络是深度学习的核心基础，从最简单的感知机到复杂的多层网络，理解其数学原理对于掌握深度学习至关重要。本章将系统介绍神经网络的数学基础、核心算法以及 PyTorch 实现。

## 章节概述

```
神经网络基础
├── 感知机模型
│   ├── 单层感知机数学定义
│   ├── 几何意义与决策边界
│   └── 感知机收敛定理
├── 多层感知机 (MLP)
│   ├── 网络结构设计
│   ├── 前向传播计算
│   └── 通用近似定理
├── 反向传播算法
│   ├── 链式法则推导
│   ├── 梯度计算详解
│   └── 计算图表示
├── 计算图与自动微分
│   ├── 静态图 vs 动态图
│   ├── 前向模式与反向模式
│   └── PyTorch 自动求导机制
├── 损失函数设计
│   ├── 回归损失 (MSE, MAE)
│   ├── 分类损失 (交叉熵)
│   └── 损失函数选择指南
└── 参数初始化策略
    ├── 随机初始化的问题
    ├── Xavier 初始化
    └── He 初始化
```

---

## 一、感知机模型

### 1.1 数学定义

感知机（Perceptron）是最简单的神经网络模型，由 Rosenblatt 于 1957 年提出。

**单层感知机模型**：

$$
f(x) = \text{sign}(w^T x + b) = \text{sign}\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

其中：
- $x = (x_1, x_2, \ldots, x_n)^T \in \mathbb{R}^n$ 为输入向量
- $w = (w_1, w_2, \ldots, w_n)^T \in \mathbb{R}^n$ 为权重向量
- $b \in \mathbb{R}$ 为偏置项
- $\text{sign}(\cdot)$ 为符号函数：$\text{sign}(z) = \begin{cases} +1, & z \geq 0 \\ -1, & z < 0 \end{cases}$

### 1.2 几何意义

📌 **决策超平面**：

感知机的决策边界是一个超平面：

$$
w^T x + b = 0
$$

**几何解释**：
- 超平面将输入空间划分为两个区域
- $w$ 是超平面的法向量，决定超平面的方向
- $b$ 决定超平面到原点的距离

**点到超平面的距离**：

样本点 $x_0$ 到决策超平面的距离为：

$$
d = \frac{|w^T x_0 + b|}{\|w\|}
$$

### 1.3 感知机学习算法

**目标**：找到一个能将正负样本分开的超平面。

**损失函数**（误分类点到超平面的总距离）：

$$
L(w, b) = -\sum_{x_i \in M} y_i (w^T x_i + b)
$$

其中 $M$ 是误分类样本集合。

**随机梯度下降更新规则**：

$$
w \leftarrow w + \eta y_i x_i
$$
$$
b \leftarrow b + \eta y_i
$$

其中 $\eta$ 为学习率。

### 1.4 感知机收敛定理

::: theorem 感知机收敛性
若训练数据集线性可分，则感知机算法在有限次迭代后收敛。
:::

**收敛条件**：
- 存在超平面能完全分开正负样本
- 存在 $\gamma > 0$，使得对所有样本 $y_i(w^T x_i + b) \geq \gamma$

**迭代次数上界**：

$$
T \leq \frac{R^2}{\gamma^2}
$$

其中 $R = \max \|x_i\|$ 为样本的最大半径。

### 1.5 PyTorch 实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """手动实现感知机"""
    
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
    
    def predict(self, x):
        """预测函数"""
        linear_output = np.dot(x, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
    
    def fit(self, X, y, max_epochs=1000):
        """训练感知机"""
        for epoch in range(max_epochs):
            misclassified = 0
            for xi, yi in zip(X, y):
                # 误分类条件: y_i * (w^T x_i + b) <= 0
                if yi * (np.dot(xi, self.weights) + self.bias) <= 0:
                    # 更新参数
                    self.weights += self.lr * yi * xi
                    self.bias += self.lr * yi
                    misclassified += 1
            
            # 无误分类样本时收敛
            if misclassified == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def decision_boundary(self, x1):
        """绘制决策边界（仅适用于2D）"""
        return -(self.weights[0] * x1 + self.bias) / self.weights[1]

# 生成线性可分数据
np.random.seed(42)
X_pos = np.random.randn(50, 2) + np.array([2, 2])
X_neg = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(50), -np.ones(50)])

# 训练感知机
perceptron = Perceptron(input_dim=2, lr=0.1)
perceptron.fit(X, y)

# 可视化决策边界
plt.figure(figsize=(10, 6))
plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', label='Class +1', alpha=0.6)
plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', label='Class -1', alpha=0.6)

x1_range = np.linspace(-4, 4, 100)
x2_boundary = perceptron.decision_boundary(x1_range)
plt.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
```

---

## 二、多层感知机 (MLP)

### 2.1 为什么需要多层网络

📌 **感知机的局限性**：单层感知机只能解决**线性可分**问题，无法处理 XOR 等非线性问题。

**XOR 问题示例**：

| $x_1$ | $x_2$ | XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR 问题**非线性可分**，无法用单一直线分开。

### 2.2 多层感知机结构

一个 $L$ 层的 MLP 可表示为：

$$
\begin{aligned}
h^{(1)} &= \sigma(W^{(1)} x + b^{(1)}) \\
h^{(2)} &= \sigma(W^{(2)} h^{(1)} + b^{(2)}) \\
&\vdots \\
y &= W^{(L)} h^{(L-1)} + b^{(L)}
\end{aligned}
$$

其中：
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ 为第 $l$ 层权重矩阵
- $b^{(l)} \in \mathbb{R}^{d_l}$ 为第 $l$ 层偏置向量
- $\sigma(\cdot)$ 为激活函数（非线性变换的关键）

### 2.3 前向传播详解

以一个三层 MLP 为例：

**网络结构**：输入层 → 隐藏层1 → 隐藏层2 → 输出层

$$
\begin{aligned}
z^{(1)} &= W^{(1)} x + b^{(1)} & \text{(隐藏层1线性变换)} \\
a^{(1)} &= \sigma(z^{(1)}) & \text{(隐藏层1激活)} \\
z^{(2)} &= W^{(2)} a^{(1)} + b^{(2)} & \text{(隐藏层2线性变换)} \\
a^{(2)} &= \sigma(z^{(2)}) & \text{(隐藏层2激活)} \\
z^{(3)} &= W^{(3)} a^{(2)} + b^{(3)} & \text{(输出层线性变换)} \\
\hat{y} &= f(z^{(3)}) & \text{(输出层激活)}
\end{aligned}
$$

**矩阵维度变化**（假设输入维度 $d_0=784$，隐藏层 $d_1=256, d_2=128$，输出 $d_3=10$）：

| 变量 | 形状 | 说明 |
|:---:|:---:|:---|
| $x$ | $(784, 1)$ | 输入向量 |
| $W^{(1)}$ | $(256, 784)$ | 第一层权重 |
| $z^{(1)}$ | $(256, 1)$ | 第一层线性输出 |
| $a^{(1)}$ | $(256, 1)$ | 第一层激活输出 |
| $W^{(2)}$ | $(128, 256)$ | 第二层权重 |
| $z^{(2)}$ | $(128, 1)$ | 第二层线性输出 |
| $a^{(2)}$ | $(128, 1)$ | 第二层激活输出 |
| $W^{(3)}$ | $(10, 128)$ | 输出层权重 |
| $\hat{y}$ | $(10, 1)$ | 最终输出 |

### 2.4 通用近似定理

::: theorem Universal Approximation Theorem
对于任意连续函数 $f: [0,1]^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在一个具有单隐藏层的神经网络，使得对于所有 $x \in [0,1]^n$：

$$
|g(x) - f(x)| < \epsilon
$$

其中 $g(x)$ 为神经网络的输出。
:::

**定理含义**：
- 💡 充分宽的单隐藏层网络可以逼近任意连续函数
- ⚠️ 定理只保证**存在性**，不说明如何找到这样的网络
- ⚠️ 深度网络通常比宽网络更高效

### 2.5 PyTorch 实现 MLP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """多层感知机实现"""
    
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, 
                 activation='relu'):
        super().__init__()
        
        # 选择激活函数
        self.activation = self._get_activation(activation)
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # 输出层（无激活函数）
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation(self, name):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 展平输入（如果需要）
        x = x.view(x.size(0), -1)
        return self.network(x)

# 创建模型实例
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# 打印模型结构
print(model)

# 测试前向传播
x = torch.randn(32, 1, 28, 28)  # 模拟MNIST输入
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 手动实现前向传播以理解计算过程
def manual_forward(x, model):
    """手动实现前向传播，展示中间结果"""
    x = x.view(x.size(0), -1)
    
    results = {'input': x}
    
    # 逐层计算
    for i, layer in enumerate(model.network):
        x = layer(x)
        layer_name = f'layer_{i}_{layer.__class__.__name__}'
        results[layer_name] = x
    
    return results

# 获取中间结果
intermediate = manual_forward(x[:1], model)
for name, tensor in intermediate.items():
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
```

---

## 三、反向传播算法

### 3.1 核心思想

反向传播（Backpropagation）是训练神经网络的核心算法，利用**链式法则**高效计算损失函数对所有参数的梯度。

**关键步骤**：
1. **前向传播**：计算各层激活值和最终输出
2. **计算损失**：根据损失函数计算误差
3. **反向传播**：从输出层向输入层逐层计算梯度
4. **参数更新**：使用梯度下降更新权重

### 3.2 链式法则回顾

**标量形式**：

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

**矩阵形式**（雅可比矩阵）：

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}
$$

### 3.3 完整推导过程

考虑一个三层网络，损失函数 $L$：

**输出层梯度**：

$$
\frac{\partial L}{\partial z^{(3)}} = \frac{\partial L}{\partial \hat{y}} \odot f'(z^{(3)})
$$

其中 $\odot$ 表示逐元素乘法，$f$ 为输出层激活函数。

**隐藏层梯度（反向传递）**：

$$
\frac{\partial L}{\partial a^{(2)}} = W^{(3)T} \frac{\partial L}{\partial z^{(3)}}
$$

$$
\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \odot \sigma'(z^{(2)})
$$

**权重梯度**：

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot a^{(l-1)T}
$$

$$
\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial z^{(l)}}
$$

### 3.4 详细推导示例

以单样本为例，使用交叉熵损失和 softmax 输出：

**损失函数**：

$$
L = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
$$

**Softmax 函数**：

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Softmax + 交叉熵的梯度**（简洁形式）：

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

这是 softmax 与交叉熵结合的优美性质——梯度形式非常简洁。

### 3.5 计算图表示

计算图将复杂的计算分解为基本操作节点：

```
      x (输入)
      │
      ▼
    [× W⁽¹⁾] ── z⁽¹⁾ ── [σ] ── a⁽¹⁾
                ↑              │
              [+ b⁽¹⁾]         │
                               ▼
                           [× W⁽²⁾] ── z⁽²⁾ ── [σ] ── a⁽²⁾
                                         ↑              │
                                       [+ b⁽²⁾]         │
                                                        ▼
                                                    [× W⁽³⁾] ── z⁽³⁾ ── [softmax] ── ŷ
                                                                ↑
                                                              [+ b⁽³⁾]
                                                                │
                              L (损失) ◄── [CrossEntropy] ◄────┘
```

### 3.6 PyTorch 反向传播演示

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """简单的MLP，便于观察反向传播"""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 创建模型和样本
model = SimpleMLP()
x = torch.tensor([[1.0, 2.0]], requires_grad=False)
y_true = torch.tensor([[1.0]])

# 注册hook来观察梯度
gradients = {}

def make_hook(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = {
            'grad_input': grad_input[0].clone() if grad_input[0] is not None else None,
            'grad_output': grad_output[0].clone()
        }
    return hook

model.fc1.register_full_backward_hook(make_hook('fc1'))
model.fc2.register_full_backward_hook(make_hook('fc2'))

# 前向传播
y_pred = model(x)
print(f"预测值: {y_pred.item():.4f}")

# 计算损失
loss = F.mse_loss(y_pred, y_true)
print(f"损失值: {loss.item():.4f}")

# 反向传播
loss.backward()

# 查看参数梯度
print("\n=== 参数梯度 ===")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}:")
        print(f"  参数值: {param.data}")
        print(f"  梯度值: {param.grad}")

# 手动计算验证梯度
print("\n=== 手动验证梯度计算 ===")

with torch.no_grad():
    # 手动前向传播
    z1 = model.fc1(x)
    a1 = torch.relu(z1)
    z2 = model.fc2(a1)
    
    print(f"z1 (第一层线性输出): {z1}")
    print(f"a1 (第一层激活输出): {a1}")
    print(f"z2 (输出层线性输出): {z2.item():.4f}")
    
    # 手动反向传播
    # 输出层梯度 (MSE loss)
    dL_dz2 = 2 * (z2 - y_true) / 1  # d(MSE)/d(z2)
    
    # 输出层权重梯度
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = dL_dz2.squeeze()
    
    print(f"\n手动计算的输出层权重梯度:\n{dL_dW2}")
    print(f"手动计算的输出层偏置梯度: {dL_db2.item():.4f}")
```

---

## 四、计算图与自动微分

### 4.1 静态图 vs 动态图

| 特性 | 静态计算图 (TensorFlow 1.x) | 动态计算图 (PyTorch) |
|:---|:---|:---|
| **构建时机** | 编译时构建 | 运行时构建 |
| **优化** | 可进行全局优化 | 优化受限 |
| **调试** | 难以调试 | 易于调试 |
| **灵活性** | 结构固定 | 可动态变化 |
| **执行模式** | 先定义后执行 | 立即执行 |

### 4.2 前向模式 vs 反向模式

**前向模式自动微分**：

$$
\dot{v}_i = \frac{\partial v_i}{\partial x}
$$

逐层向前计算导数，适用于**输入维度 < 输出维度**的情况。

**反向模式自动微分**：

$$
\bar{v}_i = \frac{\partial L}{\partial v_i}
$$

从损失函数反向传播，适用于**输入维度 > 输出维度**的情况（神经网络典型场景）。

### 4.3 PyTorch 自动求导机制

```python
import torch

# ===== 基础自动求导 =====
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
z.backward()

print(f"x: {x}")
print(f"y = x²: {y}")
print(f"z = Σy: {z.item()}")
print(f"∂z/∂x = 2x: {x.grad}")  # 应为 [4.0, 6.0]

# ===== 计算图可视化 =====
print("\n=== 计算图结构 ===")
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y + torch.sin(x)
z.backward()

print(f"z = x*y + sin(x) = {z.item():.4f}")
print(f"∂z/∂x = y + cos(x) = {x.grad.item():.4f}")  # 3 + cos(2) ≈ 2.58
print(f"∂z/∂y = x = {y.grad.item():.4f}")

# ===== 梯度累积与清零 =====
print("\n=== 梯度累积问题 ===")
w = torch.tensor([1.0], requires_grad=True)

for i in range(3):
    y = w ** 2
    y.backward()
    print(f"第 {i+1} 次反向传播后梯度: {w.grad}")

# 梯度清零
w.grad.zero_()
print(f"清零后梯度: {w.grad}")

# ===== 梯度禁用 =====
print("\n=== 禁用梯度计算 ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 方法1: torch.no_grad()
with torch.no_grad():
    y = x ** 2
    print(f"在 no_grad 上下文中，y.requires_grad: {y.requires_grad}")

# 方法2: .detach()
x_detached = x.detach()
y_detached = x_detached ** 2
print(f"detach 后，y.requires_grad: {y_detached.requires_grad}")

# ===== 保留计算图 =====
print("\n=== 保留计算图 =====")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# retain_graph=True 允许多次反向传播
y.backward(retain_graph=True)
print(f"第一次反向传播: x.grad = {x.grad}")
y.backward()
print(f"第二次反向传播: x.grad = {x.grad}")  # 梯度会累加

# ===== 自定义梯度函数 =====
print("\n=== 自定义梯度函数 ===")

class MyReLU(torch.autograd.Function):
    """自定义 ReLU 函数"""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# 使用自定义函数
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
z = y.sum()
z.backward()

print(f"输入: {x}")
print(f"MyReLU输出: {y}")
print(f"梯度: {x.grad}")
```

---

## 五、损失函数设计

### 5.1 回归损失函数

#### 均方误差 (MSE / L2 Loss)

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

**特点**：
- ✅ 数学性质好，处处可导
- ⚠️ 对异常值敏感
- 梯度：$\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y)$

#### 平均绝对误差 (MAE / L1 Loss)

$$
L_{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

**特点**：
- ✅ 对异常值鲁棒
- ⚠️ 在零点不可导
- 梯度：$\frac{\partial L}{\partial \hat{y}} = \frac{1}{N}\text{sign}(\hat{y} - y)$

#### Huber Loss

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2, & |y - \hat{y}| > \delta
\end{cases}
$$

**特点**：结合 MSE 和 MAE 的优点。

### 5.2 分类损失函数

#### 二元交叉熵 (Binary Cross-Entropy)

$$
L_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

#### 多类交叉熵 (Categorical Cross-Entropy)

$$
L_{CE} = -\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat{y}_{ik})
$$

对于单标签问题（$y$ 为 one-hot）：

$$
L_{CE} = -\log(\hat{y}_{c})
$$

其中 $c$ 为真实类别。

### 5.3 损失函数选择指南

| 任务类型 | 推荐损失函数 | 输出层激活 |
|:---|:---|:---|
| 二分类 | BCE Loss | Sigmoid |
| 多分类 | CrossEntropyLoss | Softmax（内置） |
| 多标签分类 | BCEWithLogitsLoss | Sigmoid |
| 回归（无异常值） | MSELoss | 无/线性 |
| 回归（有异常值） | HuberLoss/L1Loss | 无/线性 |

### 5.4 PyTorch 损失函数实践

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ===== 回归损失对比 =====
print("=== 回归损失对比 ===")

y_true = torch.tensor([1.0])
y_pred_normal = torch.tensor([1.1])  # 正常预测
y_pred_outlier = torch.tensor([5.0])  # 异常预测（有噪声）

loss_mse = nn.MSELoss()
loss_mae = nn.L1Loss()
loss_huber = nn.HuberLoss(delta=1.0)

print(f"正常预测 (y_pred=1.1):")
print(f"  MSE: {loss_mse(y_pred_normal, y_true).item():.4f}")
print(f"  MAE: {loss_mae(y_pred_normal, y_true).item():.4f}")
print(f"  Huber: {loss_huber(y_pred_normal, y_true).item():.4f}")

print(f"\n异常预测 (y_pred=5.0):")
print(f"  MSE: {loss_mse(y_pred_outlier, y_true).item():.4f}")  # 16.0
print(f"  MAE: {loss_mae(y_pred_outlier, y_true).item():.4f}")  # 4.0
print(f"  Huber: {loss_huber(y_pred_outlier, y_true).item():.4f}")  # 3.5

# ===== 损失函数可视化 =====
errors = np.linspace(-3, 3, 100)

mse_loss = errors ** 2
mae_loss = np.abs(errors)
huber_loss = np.where(np.abs(errors) <= 1, 
                       0.5 * errors ** 2, 
                       np.abs(errors) - 0.5)

plt.figure(figsize=(10, 6))
plt.plot(errors, mse_loss, label='MSE (L2)', linewidth=2)
plt.plot(errors, mae_loss, label='MAE (L1)', linewidth=2)
plt.plot(errors, huber_loss, label='Huber (δ=1)', linewidth=2)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Prediction Error (y - ŷ)')
plt.ylabel('Loss')
plt.title('Comparison of Regression Loss Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ===== 分类损失 =====
print("\n=== 分类损失详解 ===")

# 二分类
y_binary = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred_binary = torch.tensor([0.9, 0.1, 0.6, 0.3])

bce_loss = nn.BCELoss()
print(f"二分类 BCE Loss: {bce_loss(y_pred_binary, y_binary).item():.4f}")

# 多分类
num_classes = 5
y_multi = torch.tensor([2, 0, 4])  # 类别索引
logits = torch.randn(3, num_classes)

# CrossEntropyLoss 内置 softmax
ce_loss = nn.CrossEntropyLoss()
print(f"多分类 CrossEntropy Loss: {ce_loss(logits, y_multi).item():.4f}")

# 手动计算验证
print("\n手动计算交叉熵:")
softmax_out = F.softmax(logits, dim=1)
log_probs = torch.log(softmax_out)
selected_log_probs = log_probs[range(3), y_multi]
manual_ce = -selected_log_probs.mean()
print(f"手动计算结果: {manual_ce.item():.4f}")

# ===== 标签平滑 (Label Smoothing) =====
print("\n=== 标签平滑 ===")

# 标签平滑可以防止过拟合，提高泛化能力
ce_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

# 对比有无标签平滑
y_target = torch.tensor([1])  # 真实类别为1
logits_confident = torch.tensor([[0.0, 10.0, 0.0]])  # 非常自信的预测

loss_no_smooth = ce_loss(logits_confident, y_target)
loss_with_smooth = ce_loss_smooth(logits_confident, y_target)

print(f"无标签平滑的损失: {loss_no_smooth.item():.4f}")
print(f"有标签平滑的损失: {loss_with_smooth.item():.4f}")

# ===== 类别权重处理不平衡 =====
print("\n=== 类别权重处理不平衡 ===")

# 假设类别0有100个样本，类别1只有10个样本
class_weights = torch.tensor([0.1, 1.0])  # 给少数类更大权重
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

logits_imbalance = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
labels_imbalance = torch.tensor([0, 1])

loss_unweighted = nn.CrossEntropyLoss()(logits_imbalance, labels_imbalance)
loss_weighted = weighted_ce(logits_imbalance, labels_imbalance)

print(f"无权重损失: {loss_unweighted.item():.4f}")
print(f"有权重损失: {loss_weighted.item():.4f}")
```

---

## 六、参数初始化策略

### 6.1 为什么初始化很重要

**不恰当初始化的问题**：
- 📉 **权重过小**：激活值逐层衰减，梯度消失
- 📈 **权重过大**：激活值逐层放大，梯度爆炸
- 🔄 **权重相同**：对称性问题，神经元学习相同特征

### 6.2 常见初始化方法

#### 随机初始化（朴素方法）

$$
W \sim \mathcal{N}(0, \sigma^2) \quad \text{或} \quad W \sim \mathcal{U}(-a, a)
$$

**问题**：未考虑网络结构，可能导致梯度问题。

#### Xavier 初始化 (Glorot 初始化)

**适用于 Sigmoid、Tanh 等饱和激活函数**。

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

或正态分布：

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)
$$

**推导思路**：保持前向传播和反向传播中信号方差不变。

$$
\text{Var}(z) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

为使 $\text{Var}(z) = \text{Var}(x)$，需要 $\text{Var}(W) = \frac{1}{n_{in}}$。

#### He 初始化 (Kaiming 初始化)

**适用于 ReLU 系列激活函数**。

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$

**原理**：ReLU 将一半输入置零，需要额外的 $\sqrt{2}$ 因子来补偿。

### 6.3 初始化方法对比

| 初始化方法 | 适用激活函数 | 方差公式 |
|:---|:---|:---|
| Xavier Uniform | Sigmoid, Tanh | $\frac{6}{n_{in} + n_{out}}$ |
| Xavier Normal | Sigmoid, Tanh | $\frac{2}{n_{in} + n_{out}}$ |
| He Uniform | ReLU, LeakyReLU | $\frac{6}{n_{in}}$ |
| He Normal | ReLU, LeakyReLU | $\frac{2}{n_{in}}$ |

### 6.4 PyTorch 初始化实践

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np

# ===== 不同初始化方法 =====
def init_weights_demo():
    """演示不同初始化方法的效果"""
    
    # 创建一个大矩阵来观察初始化分布
    fan_in, fan_out = 784, 256
    
    # 1. 零初始化（不推荐）
    w_zeros = torch.zeros(fan_out, fan_in)
    
    # 2. 常数初始化
    w_constant = torch.full((fan_out, fan_in), 0.01)
    
    # 3. 随机正态分布
    w_normal = torch.randn(fan_out, fan_in)
    
    # 4. Xavier 初始化
    w_xavier_uniform = torch.empty(fan_out, fan_in)
    init.xavier_uniform_(w_xavier_uniform)
    
    w_xavier_normal = torch.empty(fan_out, fan_in)
    init.xavier_normal_(w_xavier_normal)
    
    # 5. He 初始化
    w_he_uniform = torch.empty(fan_out, fan_in)
    init.kaiming_uniform_(w_he_uniform, mode='fan_in', nonlinearity='relu')
    
    w_he_normal = torch.empty(fan_out, fan_in)
    init.kaiming_normal_(w_he_normal, mode='fan_in', nonlinearity='relu')
    
    # 统计信息
    methods = {
        'Zeros': w_zeros,
        'Constant(0.01)': w_constant,
        'Normal': w_normal,
        'Xavier Uniform': w_xavier_uniform,
        'Xavier Normal': w_xavier_normal,
        'He Uniform': w_he_uniform,
        'He Normal': w_he_normal
    }
    
    print("初始化方法统计:")
    print("-" * 60)
    for name, w in methods.items():
        print(f"{name:20s}: mean={w.mean():.6f}, std={w.std():.6f}, "
              f"min={w.min():.6f}, max={w.max():.6f}")
    
    return methods

methods = init_weights_demo()

# ===== 可视化权重分布 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

plot_methods = ['Normal', 'Xavier Uniform', 'Xavier Normal', 
                'He Uniform', 'He Normal', 'Constant(0.01)']

for ax, name in zip(axes, plot_methods):
    w = methods[name].flatten().numpy()
    ax.hist(w, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_title(f'{name}\nmean={w.mean():.4f}, std={w.std():.4f}')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

# ===== 激活值分布分析 =====
class DeepNetwork(nn.Module):
    """深度网络用于分析激活值分布"""
    
    def __init__(self, num_layers=10, hidden_dim=256, init_method='xavier'):
        super().__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # 应用初始化
        self._init_weights(init_method)
    
    def _init_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    init.xavier_normal_(m.weight)
                elif method == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'normal':
                    init.normal_(m.weight, std=0.01)
                
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward_with_activations(self, x):
        """返回各层激活值"""
        activations = [x]
        for layer in self.network:
            x = layer(x)
            activations.append(x)
        return activations

# 对比不同初始化方法的激活值分布
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, init_method in zip(axes, ['xavier', 'he', 'normal']):
    model = DeepNetwork(init_method=init_method)
    x = torch.randn(100, 256)
    activations = model.forward_with_activations(x)
    
    # 记录各层激活值的方差
    variances = [act.var().item() for act in activations]
    
    ax.plot(variances, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Variance')
    ax.set_title(f'{init_method.capitalize()} Initialization')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== 梯度分析 =====
def analyze_gradients():
    """分析不同初始化对梯度的影响"""
    
    input_dim, hidden_dim, output_dim = 784, 256, 10
    
    results = {}
    
    for init_name in ['xavier', 'he', 'normal']:
        # 创建网络
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init_name == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_name == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
        
        # 前向传播
        x = torch.randn(32, input_dim)
        y = torch.randint(0, output_dim, (32,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度统计
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        results[init_name] = grad_norms
    
    # 可视化
    plt.figure(figsize=(10, 6))
    x_labels = [f'Layer {i}' for i in range(len(results['xavier']))]
    x = np.arange(len(x_labels))
    width = 0.25
    
    for i, (name, norms) in enumerate(results.items()):
        plt.bar(x + i * width, norms, width, label=name.capitalize())
    
    plt.xlabel('Parameters')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms by Layer for Different Initializations')
    plt.xticks(x + width, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

analyze_gradients()

# ===== 正确的初始化实践 =====
print("\n=== 推荐的初始化实践 ===")

class ProperlyInitializedMLP(nn.Module):
    """正确初始化的MLP"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            
            # 根据激活函数选择初始化方法
            if activation in ['relu', 'leaky_relu', 'prelu']:
                init.kaiming_normal_(linear.weight, mode='fan_in', 
                                     nonlinearity=activation)
            else:  # sigmoid, tanh
                init.xavier_normal_(linear.weight)
            
            init.zeros_(linear.bias)
            
            layers.append(linear)
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # 输出层
        output_linear = nn.Linear(prev_dim, output_dim)
        init.xavier_normal_(output_linear.weight)
        init.zeros_(output_linear.bias)
        layers.append(output_linear)
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x):
        return self.network(x)

# 测试
model = ProperlyInitializedMLP(784, [256, 128], 10, activation='relu')
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 七、知识点关联

### 7.1 知识结构图

```
神经网络基础
     │
     ├─── 感知机 → 多层感知机
     │         │
     │         └──→ 非线性表达能力 (激活函数)
     │
     ├─── 前向传播 ──→ 损失函数计算
     │         │
     │         └──→ 计算图构建
     │
     ├─── 反向传播 ←── 链式法则
     │         │
     │         ├──→ 梯度计算
     │         │
     │         └──→ 参数更新 (优化器)
     │
     ├─── 损失函数
     │         │
     │         ├──→ 回归任务: MSE, MAE, Huber
     │         │
     │         └──→ 分类任务: 交叉熵
     │
     └─── 参数初始化
              │
              ├──→ Xavier (Sigmoid/Tanh)
              │
              └──→ He (ReLU系列)
```

### 7.2 与其他章节的联系

| 本章概念 | 关联章节 | 关联说明 |
|:---|:---|:---|
| 激活函数 | [激活函数详解](02-activations.md) | 激活函数的选择影响初始化策略 |
| 优化算法 | [优化算法详解](01-optimizers.md) | 梯度下降是参数更新的基础 |
| 正则化 | [正则化技术](04-regularization.md) | 初始化与正则化共同影响泛化能力 |
| CNN | [卷积神经网络](03-cnn.md) | CNN 是 MLP 的推广，共享权重 |
| RNN | [循环神经网络](04-rnn-family.md) | RNN 的反向传播需要处理时间维度 |

---

## 八、核心考点

### 8.1 理论考点

1. **感知机收敛性**
   - 感知机何时收敛？收敛条件是什么？
   - 为什么感知机不能解决 XOR 问题？

2. **通用近似定理**
   - 定理的内容和意义
   - 定理的局限性

3. **反向传播推导**
   - 能够手写链式法则推导
   - 理解梯度消失/爆炸的原因

4. **初始化选择**
   - 为什么零初始化不行？
   - Xavier 和 He 初始化的适用场景

### 8.2 实践考点

1. **PyTorch 实现**
   - 使用 `nn.Module` 定义网络
   - 正确使用 `requires_grad`
   - 梯度清零的时机

2. **调试技巧**
   - 检查梯度分布
   - 监控激活值方差
   - 使用 `torch.autograd.set_detect_anomaly(True)`

### 8.3 常见面试题

```python
# Q1: 为什么不能将权重初始化为零？
"""
答案：如果所有权重相同，则同一层的所有神经元将学习相同的特征，
    无论训练多久，它们都会保持相同。这称为对称性问题。
    解决方法：随机初始化，打破对称性。
"""

# Q2: 解释梯度消失和梯度爆炸
"""
梯度消失：反向传播时，梯度逐层乘以小于1的值，导致浅层梯度接近零。
梯度爆炸：梯度逐层乘以大于1的值，导致梯度变得极大。

解决方案：
- 使用 ReLU 激活函数（梯度消失问题较轻）
- 批归一化（Batch Normalization）
- 残差连接（ResNet）
- 梯度裁剪（Gradient Clipping）
- 合适的初始化
"""

# Q3: Xavier 和 He 初始化的区别？
"""
Xavier 初始化：
- 假设激活函数是线性的（或接近线性，如 Tanh、Sigmoid）
- 考虑输入和输出的 fan-in 和 fan-out
- 方差 = 2/(fan_in + fan_out)

He 初始化：
- 专为 ReLU 系列设计
- ReLU 将一半输入置零，需要额外补偿
- 方差 = 2/fan_in
"""
```

---

## 九、学习建议

### 9.1 学习路径

```
第一阶段：理论理解
├── 理解感知机的工作原理
├── 掌握多层网络的前向传播
└── 理解链式法则在反向传播中的应用

第二阶段：数学推导
├── 手动推导简单网络的梯度
├── 推导 softmax + 交叉熵的梯度
└── 理解 Xavier/He 初始化的推导

第三阶段：代码实践
├── 从零实现一个 MLP
├── 实现不同初始化方法
└── 观察不同损失函数的效果

第四阶段：深入理解
├── 分析激活值和梯度的分布
├── 理解为什么某些设计有效
└── 阅读经典论文
```

### 9.2 实践建议

1. **动手实现**
   - 不只用框架，尝试从零用 NumPy 实现 MLP
   - 手动实现反向传播，加深理解

2. **可视化调试**
   - 绘制损失曲线
   - 可视化权重分布
   - 监控各层激活值的统计量

3. **实验对比**
   - 对比不同初始化方法
   - 对比不同激活函数
   - 对比不同损失函数

### 9.3 推荐资源

**经典论文**：
- Rumelhart et al. (1986) - "Learning representations by back-propagating errors"
- Glorot & Bengio (2010) - "Understanding the difficulty of training deep feedforward neural networks"
- He et al. (2015) - "Delving Deep into Rectifiers"

**在线课程**：
- Stanford CS231n: Convolutional Neural Networks
- 3Blue1Brown: Neural Networks 系列视频

**书籍**：
- "Deep Learning" by Goodfellow, Bengio, Courville
- "神经网络与深度学习" by 邱锡鹏
