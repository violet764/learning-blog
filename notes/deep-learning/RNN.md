# 循环神经网络（RNN）与序列建模

## 章节概述
循环神经网络是专门用于处理**序列数据**的神经网络架构，在自然语言处理、语音识别、时间序列预测等领域有广泛应用。本章将系统讲解RNN的基本原理、变体模型和实际应用。

## 核心知识点分点详解

### 1. RNN基本结构与工作原理

#### 概念
RNN的核心思想是引入**循环连接**，使网络能够处理任意长度的序列数据，并在处理过程中**保持状态记忆**。

#### 原理
- **时序展开**：RNN在每个时间步共享相同的权重参数
- **隐藏状态**：网络维护一个隐藏状态向量，作为序列信息的压缩表示
- **数学公式**：
  $$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
  $$y_t = W_{hy}h_t + b_y$$
  
  其中：
  - $h_t$：当前时间步的隐藏状态
  - $x_t$：当前时间步的输入
  - $y_t$：当前时间步的输出
  - $W_{hh}, W_{xh}, W_{hy}$：权重矩阵

#### 实操要点
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 输入到隐藏层的权重
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 隐藏层到输出的权重
        self.h2o = nn.Linear(hidden_size, output_size)
        # 激活函数
        self.tanh = nn.Tanh()
    
    def forward(self, input_seq, hidden):
        """
        input_seq: 序列输入 [seq_len, batch_size, input_size]
        hidden: 初始隐藏状态 [batch_size, hidden_size]
        """
        outputs = []
        
        # 按时间步处理序列
        for i in range(input_seq.size(0)):
            # 拼接当前输入和上一时刻隐藏状态
            combined = torch.cat((input_seq[i], hidden), 1)
            
            # 更新隐藏状态
            hidden = self.tanh(self.i2h(combined))
            
            # 计算输出
            output = self.h2o(hidden)
            outputs.append(output)
        
        # 将所有时间步的输出堆叠
        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size)

# 使用示例
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
input_seq = torch.randn(8, 1, 10)  # 序列长度8，批次大小1，输入维度10
hidden = rnn.init_hidden(1)
outputs, final_hidden = rnn(input_seq, hidden)
print(f"输出形状: {outputs.shape}")  # [8, 1, 5]
```

### 2. 长短期记忆网络（LSTM）

#### 概念
LSTM是RNN的重要变体，通过**门控机制**解决传统RNN的**梯度消失/爆炸**问题，能够学习长期依赖关系。

#### 原理
**LSTM单元结构**：
1. **遗忘门（Forget Gate）**：决定丢弃哪些信息
2. **输入门（Input Gate）**：决定更新哪些新信息
3. **输出门（Output Gate）**：决定输出哪些信息

**数学公式**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

#### 实操要点
```python
class LSTMCell(nn.Module):
    """手动实现LSTM单元"""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 四个门控的线性变换
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_io = nn.Linear(input_size, hidden_size)
        
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden_state):
        """
        x: 当前输入 [batch_size, input_size]
        hidden_state: (h_prev, c_prev) 上一时刻隐藏状态和细胞状态
        """
        h_prev, c_prev = hidden_state
        
        # 输入门
        i_t = self.sigmoid(self.W_ii(x) + self.W_hi(h_prev))
        # 遗忘门
        f_t = self.sigmoid(self.W_if(x) + self.W_hf(h_prev))
        # 细胞状态候选值
        g_t = self.tanh(self.W_ig(x) + self.W_hg(h_prev))
        # 输出门
        o_t = self.sigmoid(self.W_io(x) + self.W_ho(h_prev))
        
        # 更新细胞状态
        c_t = f_t * c_prev + i_t * g_t
        # 更新隐藏状态
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t

# 使用PyTorch内置LSTM
lstm = nn.LSTM(input_size=100, hidden_size=50, num_layers=2, batch_first=True)
input_data = torch.randn(32, 10, 100)  # [batch_size, seq_len, input_size]
output, (h_n, c_n) = lstm(input_data)
print(f"LSTM输出形状: {output.shape}")  # [32, 10, 50]
print(f"最终隐藏状态: {h_n.shape}")     # [2, 32, 50]
```

### 3. 门控循环单元（GRU）

#### 概念
GRU是LSTM的简化版本，通过**减少门控数量**降低计算复杂度，同时保持较好的性能。

#### 原理
**GRU单元结构**：
1. **重置门（Reset Gate）**：控制历史信息的遗忘程度
2. **更新门（Update Gate）**：控制新旧信息的混合比例

**数学公式**：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

#### 实操要点
```python
# 使用PyTorch内置GRU
gru = nn.GRU(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
input_seq = torch.randn(16, 20, 64)  # [batch_size, seq_len, input_size]
output, h_n = gru(input_seq)
print(f"GRU输出形状: {output.shape}")  # [16, 20, 32]
print(f"最终隐藏状态: {h_n.shape}")     # [2, 16, 32]

# GRU与LSTM对比
print("\nGRU vs LSTM对比:")
print("GRU优势: 参数更少，训练更快")
print("LSTM优势: 长期记忆能力更强")
print("选择建议: 对于大多数任务，GRU是更好的选择")
```

### 4. 双向RNN与深度RNN

#### 双向RNN（Bi-RNN）
**概念**：同时使用正向和反向两个RNN，捕捉前后文信息

```python
# 双向LSTM示例
bi_lstm = nn.LSTM(
    input_size=100, 
    hidden_size=50, 
    num_layers=2, 
    batch_first=True,
    bidirectional=True  # 启用双向
)

input_data = torch.randn(32, 10, 100)
output, (h_n, c_n) = bi_lstm(input_data)
print(f"双向LSTM输出形状: {output.shape}")  # [32, 10, 100] - 正向+反向拼接
print(f"隐藏状态形状: {h_n.shape}")          # [4, 32, 50] - 2层×2方向
```

#### 深度RNN
**概念**：堆叠多个RNN层，增加网络深度和表达能力

```python
# 深度双向LSTM
deep_bi_lstm = nn.LSTM(
    input_size=100,
    hidden_size=64,
    num_layers=4,           # 4层深度
    batch_first=True,
    bidirectional=True,
    dropout=0.2             # 层间dropout
)
```

### 5. RNN在实际应用中的技巧

#### 序列填充与打包
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 处理变长序列的示例
def process_variable_length_sequences():
    # 原始序列（不同长度）
    sequences = [
        torch.tensor([1, 2, 3]),          # 长度3
        torch.tensor([4, 5]),             # 长度2
        torch.tensor([6, 7, 8, 9])        # 长度4
    ]
    
    # 填充到相同长度
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    print(f"填充后序列: {padded_sequences}")
    
    # 序列实际长度
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # 打包序列（提高计算效率）
    packed_input = pack_padded_sequence(
        padded_sequences, 
        lengths, 
        batch_first=True, 
        enforce_sorted=False
    )
    
    # 通过RNN处理
    lstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True)
    packed_output, (h_n, c_n) = lstm(packed_input)
    
    # 解包输出
    output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
    print(f"解包后输出形状: {output.shape}")
```

#### 梯度裁剪
```python
# 防止梯度爆炸的梯度裁剪
def train_rnn_with_gradient_clipping(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 前向传播
        output = model(batch.input)
        loss = criterion(output, batch.target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
```

## 知识点间关联逻辑

1. **RNN→LSTM→GRU**构成序列模型的演进路径
2. **门控机制**是解决长期依赖问题的关键创新
3. **双向处理**扩展了序列信息的利用范围
4. **深度堆叠**提升了模型的表征能力
5. **序列填充与打包**解决了实际应用中的变长序列问题

## 章节核心考点汇总

### 必考知识点
1. **RNN前向传播计算**：隐藏状态更新公式
2. **LSTM三个门控的作用**：遗忘门、输入门、输出门功能
3. **GRU与LSTM的区别**：门控数量、计算复杂度对比
4. **双向RNN原理**：正向和反向信息的融合方式
5. **梯度消失/爆炸问题**：RNN训练中的主要挑战

### 高频考点
1. LSTM细胞状态更新的数学推导
2. GRU重置门和更新门的实际作用
3. 序列填充与打包的技术细节
4. RNN在自然语言处理中的应用
5. 注意力机制与RNN的结合

## 学习建议 / 后续延伸方向

### 学习建议
1. **从简单到复杂**：先理解基本RNN，再学习LSTM/GRU
2. **代码实现**：手动实现RNN单元有助于深入理解
3. **可视化理解**：绘制RNN的时序展开图
4. **实际问题**：尝试用RNN解决文本分类、时间序列预测等任务

### 延伸方向
1. **注意力机制**：Transformer架构的基础
2. **序列到序列模型**：机器翻译、文本摘要应用
3. **记忆网络**：Neural Turing Machines, Memory Networks
4. **图神经网络**：处理图结构数据的RNN变体
5. **神经微分方程**：连续时间版本的RNN

### 实战项目建议
1. **文本情感分析**：使用RNN/LSTM分析电影评论情感
2. **时间序列预测**：股票价格、天气数据预测
3. **机器翻译**：实现简单的seq2seq翻译模型
4. **文本生成**：使用RNN生成诗歌、小说片段

**关键提示**：RNN是处理序列数据的基石，虽然现在Transformer在某些任务上表现更好，但理解RNN的原理对于掌握深度学习序列建模至关重要。建议在学习过程中重点关注门控机制的设计思想和数学原理。