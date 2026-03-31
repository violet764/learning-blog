# 条件随机场 (CRF) 详解

条件随机场（Conditional Random Field，CRF）是一种**判别式概率图模型**，广泛应用于序列标注任务，如命名实体识别（NER）、词性标注（POS）、分词等。相比于生成式模型（如 HMM），CRF 能够灵活地融合丰富的特征，且避免了标签偏置问题。

## 基本概念

### 什么是 CRF？

CRF 是一种**判别式**的无向图模型，直接建模条件概率 $P(Y|X)$，即在给定观测序列 $X$ 的条件下，预测标签序列 $Y$ 的概率。

> 📌 **核心思想**：给定观测序列，寻找最可能的标签序列。与 HMM 不同，CRF 不关心 $X$ 如何生成，只关心如何根据 $X$ 预测 $Y$。

### 生成式 vs 判别式模型

| 特性 | 生成式模型（HMM） | 判别式模型（CRF） |
|------|------------------|------------------|
| 建模对象 | 联合概率 $P(X, Y)$ | 条件概率 $P(Y\|X)$ |
| 模型复杂度 | 参数较少 | 参数较多，特征灵活 |
| 特征使用 | 受限（独立性假设） | 灵活（可使用任意特征） |
| 推理效率 | 相对高效 | 稍慢，但可优化 |
| 适用场景 | 数据少、特征简单 | 特征丰富、追求精度 |

### 为什么需要 CRF？

在序列标注任务中，我们面临以下挑战：

| 问题 | 说明 |
|------|------|
| 标签依赖 | 相邻标签之间有强依赖关系（如"B-LOC"后通常是"I-LOC"） |
| 特征融合 | 需要结合上下文、词性、字形等多维特征 |
| 标签偏置 | 简单的判别模型（如 MEMM）存在标签偏置问题 |

CRF 通过**全局归一化**解决了标签偏置问题，同时支持灵活的特征设计。

## 核心原理

### 概率图结构

CRF 最常用的是**线性链结构**（Linear Chain CRF），其图结构如下：

```
观测序列:  X1 ─── X2 ─── X3 ─── ... ─── Xn
           │      │      │              │
标签序列:  Y1 ─── Y2 ─── Y3 ─── ... ─── Yn
```

- 观测变量 $X = (x_1, x_2, ..., x_n)$：输入序列
- 标签变量 $Y = (y_1, y_2, ..., y_n)$：待预测的标签序列
- 标签之间形成马尔可夫链，每个标签依赖于对应的观测

### 条件概率定义

给定观测序列 $X$，标签序列 $Y$ 的条件概率为：

$$
P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{i=1}^{n}\sum_{k} \lambda_k f_k(y_{i-1}, y_i, X, i) + \sum_{i=1}^{n}\sum_{l} \mu_l g_l(y_i, X, i)\right)
$$

其中：
- $Z(X)$：归一化因子（配分函数），确保概率和为 1
- $f_k$：转移特征函数，刻画相邻标签之间的关系
- $g_l$：状态特征函数，刻画标签与观测之间的关系
- $\lambda_k, \mu_l$：模型参数（权重）

### 归一化因子

$$
Z(X) = \sum_{Y} \exp\left(\sum_{i,k} \lambda_k f_k(y_{i-1}, y_i, X, i) + \sum_{i,l} \mu_l g_l(y_i, X, i)\right)
$$

💡 $Z(X)$ 是对所有可能的标签序列求和，这正是 CRF 避免标签偏置的关键——**全局归一化**。

### 特征函数

特征函数是 CRF 的核心，分为两类：

**1. 转移特征（边特征）**

$$
f_k(y_{i-1}, y_i, X, i)
$$

描述从标签 $y_{i-1}$ 转移到 $y_i$ 的特征。

**2. 状态特征（节点特征）**

$$
g_l(y_i, X, i)
$$

描述在位置 $i$，标签 $y_i$ 与观测 $X$ 的关系。

**特征函数示例（命名实体识别）**：

```python
# 转移特征：B-PER 后面通常是 I-PER
def f_transition(prev_tag, curr_tag, X, i):
    if prev_tag == "B-PER" and curr_tag == "I-PER":
        return 1
    return 0

# 状态特征：当前词是大写字母开头，可能是人名
def f_state(tag, X, i):
    if tag == "B-PER" and X[i][0].isupper():
        return 1
    return 0

# 状态特征：当前词前面是"在"，可能是地点
def f_state_location(tag, X, i):
    if tag in ["B-LOC", "I-LOC"] and i > 0 and X[i-1] == "在":
        return 1
    return 0
```

### 简化表示

在实际计算中，通常将转移特征和状态特征统一表示：

$$
P(Y|X) = \frac{1}{Z(X)} \exp\left(\sum_{i=1}^{n} \sum_{j} w_j F_j(y_{i-1}, y_i, X, i)\right)
$$

其中 $F_j$ 统一表示所有特征函数，$w_j$ 为对应权重。

## 参数学习

### 极大似然估计

CRF 的训练目标是最大化对数似然：

$$
\mathcal{L}(w) = \sum_{m=1}^{M} \log P(Y^{(m)}|X^{(m)}; w)
$$

展开后：

$$
\mathcal{L}(w) = \sum_{m} \left[ \sum_{i,j} w_j F_j(y^{(m)}_{i-1}, y^{(m)}_i, X^{(m)}, i) - \log Z(X^{(m)}) \right]
$$

### 梯度计算

对参数 $w_j$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \sum_{m} \left[ F_j(Y^{(m)}, X^{(m)}) - \mathbb{E}_{P(Y|X^{(m)})}[F_j(Y, X^{(m)})] \right]
$$

直观理解：
- 第一项：真实标签序列中特征 $F_j$ 的**经验期望**
- 第二项：模型分布下特征 $F_j$ 的**模型期望**
- 训练过程就是让两者尽可能接近

### 优化算法

由于 CRF 的目标函数是凸函数，常用优化方法包括：

| 方法 | 特点 |
|------|------|
| 梯度下降（GD/SGD） | 简单，但收敛慢 |
| L-BFGS | 拟牛顿法，收敛快 |
| 共轭梯度法 | 介于两者之间 |

## 推理算法

### 问题定义

给定训练好的 CRF 模型和观测序列 $X$，推理任务有两种：

1. **序列预测**：找到最可能的标签序列
   $$Y^* = \arg\max_Y P(Y|X)$$

2. **边缘概率**：计算某个位置标签的概率
   $$P(y_i|X)$$

### 维特比算法（Viterbi）

用于求解最可能的标签序列，采用动态规划思想。

**核心思想**：定义 $\delta_i(y)$ 为到达位置 $i$ 且标签为 $y$ 的最大概率路径的得分。

**递推公式**：

$$
\delta_i(y) = \max_{y'} \left[ \delta_{i-1}(y') + \text{score}(y', y, X, i) \right]
$$

**算法步骤**：

1. **初始化**：$\delta_1(y) = \text{score}(\text{START}, y, X, 1)$
2. **递推**：依次计算每个位置的 $\delta_i(y)$，记录前驱节点
3. **回溯**：从终点开始，根据记录的前驱节点找到最优路径

### 前向-后向算法

用于计算边缘概率 $P(y_i|X)$，需要计算归一化因子 $Z(X)$。

**前向变量**：

$$
\alpha_i(y) = \sum_{y_1,...,y_{i-1}} \exp\left(\sum_{t=1}^{i} \text{score}(y_{t-1}, y_t, X, t)\right) \cdot \mathbb{I}(y_i = y)
$$

**后向变量**：

$$
\beta_i(y) = \sum_{y_{i+1},...,y_n} \exp\left(\sum_{t=i+1}^{n} \text{score}(y_{t-1}, y_t, X, t)\right) \cdot \mathbb{I}(y_i = y)
$$

**边缘概率**：

$$
P(y_i = y | X) = \frac{\alpha_i(y) \cdot \beta_i(y)}{Z(X)}
$$

## 代码示例

### 基础实现：从零手写 Linear Chain CRF

```python
import numpy as np
from collections import defaultdict

class SimpleCRF:
    """简单的线性链 CRF 实现"""
    
    def __init__(self, num_tags):
        """
        初始化 CRF
        
        Args:
            num_tags: 标签数量
        """
        self.num_tags = num_tags
        # 转移矩阵：transit[i][j] 表示从标签 i 转移到 j 的得分
        self.transition = np.random.randn(num_tags + 1, num_tags) * 0.01
        # +1 是为了处理起始状态
        
    def _forward(self, emission):
        """
        前向算法计算归一化因子
        
        Args:
            emission: 发射得分矩阵 [seq_len, num_tags]
            
        Returns:
            log_Z: 归一化因子的对数
            alpha: 前向变量
        """
        seq_len = len(emission)
        alpha = np.full((seq_len, self.num_tags), -np.inf)
        
        # 初始化：从起始状态转移
        alpha[0] = self.transition[-1] + emission[0]
        
        # 递推
        for t in range(1, seq_len):
            for j in range(self.num_tags):
                # log-sum-exp 技巧避免数值下溢
                scores = alpha[t-1] + self.transition[:, j] + emission[t, j]
                alpha[t, j] = self._logsumexp(scores)
        
        log_Z = self._logsumexp(alpha[-1])
        return log_Z, alpha
    
    def _backward(self, emission):
        """
        后向算法
        
        Args:
            emission: 发射得分矩阵 [seq_len, num_tags]
            
        Returns:
            beta: 后向变量
        """
        seq_len = len(emission)
        beta = np.full((seq_len, self.num_tags), -np.inf)
        
        # 初始化
        beta[-1] = 0  # log(1) = 0
        
        # 递推（从后往前）
        for t in range(seq_len - 2, -1, -1):
            for i in range(self.num_tags):
                scores = self.transition[i] + emission[t+1] + beta[t+1]
                beta[t, i] = self._logsumexp(scores)
        
        return beta
    
    def _logsumexp(self, x):
        """数值稳定的 log-sum-exp"""
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))
    
    def _viterbi(self, emission):
        """
        维特比算法解码
        
        Args:
            emission: 发射得分矩阵 [seq_len, num_tags]
            
        Returns:
            best_path: 最优标签序列
        """
        seq_len = len(emission)
        viterbi = np.full((seq_len, self.num_tags), -np.inf)
        backpointer = np.zeros((seq_len, self.num_tags), dtype=int)
        
        # 初始化
        viterbi[0] = self.transition[-1] + emission[0]
        
        # 递推
        for t in range(1, seq_len):
            for j in range(self.num_tags):
                scores = viterbi[t-1] + self.transition[:, j]
                best_prev = np.argmax(scores)
                viterbi[t, j] = scores[best_prev] + emission[t, j]
                backpointer[t, j] = best_prev
        
        # 回溯
        best_path = [np.argmax(viterbi[-1])]
        for t in range(seq_len - 1, 0, -1):
            best_path.append(backpointer[t, best_path[-1]])
        best_path.reverse()
        
        return best_path
    
    def neg_log_likelihood(self, X, y, emission_fn):
        """
        计算负对数似然（损失函数）
        
        Args:
            X: 观测序列
            y: 标签序列
            emission_fn: 计算发射得分的函数
        """
        emission = emission_fn(X)  # [seq_len, num_tags]
        seq_len = len(emission)
        
        # 计算真实路径得分
        score = self.transition[-1, y[0]] + emission[0, y[0]]
        for t in range(1, seq_len):
            score += self.transition[y[t-1], y[t]] + emission[t, y[t]]
        
        # 计算归一化因子
        log_Z, _ = self._forward(emission)
        
        return log_Z - score  # 负对数似然
    
    def decode(self, X, emission_fn):
        """
        预测最可能的标签序列
        
        Args:
            X: 观测序列
            emission_fn: 计算发射得分的函数
            
        Returns:
            best_path: 预测的标签序列
        """
        emission = emission_fn(X)
        return self._viterbi(emission)


# 使用示例
np.random.seed(42)

# 假设有 3 个标签：B-PER, I-PER, O
num_tags = 3
crf = SimpleCRF(num_tags)

# 模拟发射得分（实际中由神经网络等计算）
def mock_emission(X):
    seq_len = len(X)
    return np.random.randn(seq_len, num_tags)

# 观测序列
X = ["张三", "在", "北京", "工作"]
y = [0, 2, 1, 2]  # B-PER, O, I-LOC?, O（假设）

# 计算损失
loss = crf.neg_log_likelihood(X, y, mock_emission)
print(f"负对数似然: {loss:.4f}")

# 预测
pred = crf.decode(X, mock_emission)
print(f"预测标签序列: {pred}")
```

### 使用 sklearn-crfsuite

在实际项目中，推荐使用成熟的 CRF 库：

```python
# 安装: pip install sklearn-crfsuite

import sklearn_crfsuite
from sklearn_crfsuite import metrics

# 示例：命名实体识别
# 准备训练数据（每个词用特征字典表示）
train_sents = [
    [{'word': '张三', 'pos': 'NR', 'shape': 'Xx'}, 
     {'word': '在', 'pos': 'P', 'shape': 'x'}, 
     {'word': '北京', 'pos': 'NS', 'shape': 'Xx'}, 
     {'word': '工作', 'pos': 'V', 'shape': 'xx'}],
    # 更多句子...
]

train_labels = [
    ['B-PER', 'O', 'B-LOC', 'O'],
    # 更多标签...
]


def word2features(sent, i):
    """提取单词特征"""
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word['word'].lower(),
        'word[-3:]': word['word'][-3:],
        'word[-2:]': word['word'][-2:],
        'word.isupper()': word['word'].isupper(),
        'word.istitle()': word['word'].istitle(),
        'word.isdigit()': word['word'].isdigit(),
        'pos': word['pos'],
        'pos[:2]': word['pos'][:2],
    }
    
    # 前一个词的特征
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1['word'].lower(),
            '-1:pos': word1['pos'],
            '-1:word.istitle()': word1['word'].istitle(),
        })
    else:
        features['BOS'] = True  # 句首标记
    
    # 后一个词的特征
    if i < len(sent) - 1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1['word'].lower(),
            '+1:pos': word1['pos'],
            '+1:word.istitle()': word1['word'].istitle(),
        })
    else:
        features['EOS'] = True  # 句尾标记
    
    return features


def sent2features(sent):
    """将句子转换为特征列表"""
    return [word2features(sent, i) for i in range(len(sent))]


# 准备特征
X_train = [sent2features(s) for s in train_sents]
y_train = train_labels

# 创建并训练 CRF 模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,   # L1 正则化
    c2=0.1,   # L2 正则化
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

# 预测
y_pred = crf.predict(X_train)
print(f"预测结果: {y_pred[0]}")

# 查看模型学到的转移特征
print("\n转移特征权重（Top 5）:")
for (attr1, attr2), weight in crf.transition_features_.most_common(5):
    print(f"  {attr1} -> {attr2}: {weight:.4f}")
```

### 使用 PyTorch CRF 层

在现代深度学习中，CRF 常与 BiLSTM 等神经网络结合：

```python
# 安装: pip install pytorch-crf

import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    """BiLSTM + CRF 用于序列标注"""
    
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True)
        
    def forward(self, x, tags=None, mask=None):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            tags: 标签序列 [batch_size, seq_len]（训练时需要）
            mask: 有效位置掩码 [batch_size, seq_len]
        """
        # 嵌入
        embeds = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # BiLSTM 编码
        lstm_out, _ = self.lstm(embeds)  # [batch, seq_len, hidden_dim]
        
        # 映射到标签空间
        emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, tag_size]
        
        if tags is not None:
            # 训练：返回负对数似然
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:
            # 推理：返回最优路径
            return self.crf.decode(emissions, mask=mask)


# 使用示例
vocab_size = 1000
tag_size = 5  # B-PER, I-PER, B-LOC, I-LOC, O

model = BiLSTM_CRF(vocab_size, tag_size)

# 模拟输入
batch_size, seq_len = 2, 10
x = torch.randint(0, vocab_size, (batch_size, seq_len))
tags = torch.randint(0, tag_size, (batch_size, seq_len))
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

# 训练模式
loss = model(x, tags, mask)
print(f"训练损失: {loss.item():.4f}")

# 预测模式
model.eval()
with torch.no_grad():
    pred_tags = model(x, mask=mask)
    print(f"预测标签: {pred_tags[0]}")
```

## 与 HMM 和 MEMM 的对比

### 三种模型对比

| 特性 | HMM | MEMM | CRF |
|------|-----|------|-----|
| 类型 | 生成式 | 判别式 | 判别式 |
| 建模对象 | $P(X,Y)$ | $P(y_i\|y_{i-1},X)$ | $P(Y\|X)$ |
| 归一化 | 全局 | 局部 | 全局 |
| 标签偏置 | 无 | 有 | 无 |
| 特征灵活性 | 低 | 中 | 高 |
| 计算复杂度 | 低 | 中 | 高 |

### 标签偏置问题

MEMM 使用**局部归一化**：

$$
P(y_i|y_{i-1}, X) = \frac{\exp(\text{score}(y_{i-1}, y_i, X, i))}{\sum_{y'} \exp(\text{score}(y_{i-1}, y', X, i))}
$$

这导致：转移选项少的标签会获得更高的概率，因为分母更小。

CRF 使用**全局归一化**，分母对所有可能的完整序列求和，避免了这个问题。

```
MEMM 标签偏置示例：

标签 A: 可以转移到 A, B, C (3 个选项)
标签 B: 只能转移到 B (1 个选项)

即使 A->A 的得分很高，但由于 A 有 3 个转移选项，
局部归一化后概率可能反而比 B->B 低。
```

## 优缺点分析

### ✅ 优点

1. **全局最优**：全局归一化避免标签偏置
2. **特征灵活**：可以使用任意复杂的特征函数
3. **判别建模**：直接优化预测目标，效果通常更好
4. **理论完善**：凸优化，保证全局最优解

### ❌ 局限性

| 局限 | 说明 | 解决方案 |
|------|------|----------|
| 计算复杂 | 训练和推理需要计算全局归一化 | 使用 GPU 加速、近似算法 |
| 特征工程 | 需要人工设计特征 | 结合深度学习自动提取特征 |
| 长序列 | 维特比算法复杂度随序列长度增长 | 分段处理、近似解码 |

## 常见问题

### Q1: CRF 为什么能解决标签偏置？

CRF 使用全局归一化，归一化因子 $Z(X)$ 考虑了所有可能的标签序列，因此每个转移决策都是在全局上下文中做出的，不会因为局部转移选项数量而偏置。

### Q2: CRF 与深度学习如何结合？

常见的组合是 **BiLSTM-CRF**：
- BiLSTM 负责提取丰富的上下文特征（自动学习）
- CRF 层负责建模标签之间的依赖关系

这种组合既利用了深度学习的特征自动提取能力，又保留了 CRF 处理标签依赖的优势。

### Q3: 如何处理 OOV（未登录词）？

1. 使用字符级特征（如字符嵌入）
2. 使用字形特征（如大写、数字、标点等模式）
3. 结合预训练语言模型（如 BERT）的子词表示

### Q4: CRF 的训练为什么慢？

CRF 训练需要计算全局归一化因子 $Z(X)$，这需要使用前向-后向算法遍历所有可能的标签组合。对于 $n$ 个位置、$k$ 个标签，计算复杂度为 $O(nk^2)$。

## 参考资料

- Lafferty, J., et al. "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data" (ICML 2001)
- Sutton, C., & McCallum, A. "An Introduction to Conditional Random Fields" (Foundations and Trends in Machine Learning, 2012)
- [sklearn-crfsuite 文档](https://sklearn-crfsuite.readthedocs.io/)
- [PyTorch-CRF 文档](https://pytorch-crf.readthedocs.io/)
