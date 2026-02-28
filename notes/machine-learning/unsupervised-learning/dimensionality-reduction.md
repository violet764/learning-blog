# 降维技术

降维是将高维数据映射到低维空间的技术，目标是**保留数据的主要信息**，同时**减少计算复杂度**和**去除噪声**。它是无监督学习中的核心任务之一。

## 核心概念

### 为什么需要降维？

| 问题 | 解决方案 |
|------|----------|
| 维度灾难 | 高维空间数据稀疏，距离度量失效 |
| 计算复杂度 | 特征多时训练和预测慢 |
| 可视化困难 | 人只能理解 2D/3D 数据 |
| 冗余特征 | 相关特征携带重复信息 |
| 噪声干扰 | 不重要特征增加噪声 |

### 降维的数学框架

**目标**：找到映射 $f: \mathbb{R}^d \to \mathbb{R}^k$，其中 $k \ll d$

**信息保留准则**：
- 方差最大化 → PCA
- 类别区分最大化 → LDA
- 局部结构保持 → 流形学习
- 重构误差最小化 → 自编码器

### 方法分类

```
降维方法
├── 线性方法
│   ├── PCA（主成分分析）
│   ├── LDA（线性判别分析）
│   └── 因子分析
└── 非线性方法
    ├── 流形学习（Isomap, LLE, t-SNE）
    └── 自编码器
```

---

## 主成分分析（PCA）

PCA 是最经典的线性降维方法，通过**正交变换**将原始特征转换为一组**线性不相关的主成分**。

### 数学推导

#### 最大方差视角

**目标**：找到投影方向 $\mathbf{w}$ 使得投影后方差最大

$$\max_{\mathbf{w}} \text{Var}(\mathbf{X}\mathbf{w}) = \max_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w}$$

约束：$\|\mathbf{w}\| = 1$

**求解**：使用拉格朗日乘子法

$$L(\mathbf{w}, \lambda) = \mathbf{w}^T \Sigma \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{w} - 1)$$

求导得：$\Sigma \mathbf{w} = \lambda \mathbf{w}$

这正是**特征值问题**！最大方差等于最大特征值 $\lambda_1$，对应特征向量 $\mathbf{w}_1$ 即为第一主成分。

#### 最小重构误差视角

将数据投影到 $k$ 维子空间后重构：

$$\hat{\mathbf{X}} = \mathbf{X} \mathbf{W} \mathbf{W}^T$$

最小化重构误差：

$$\min_{\mathbf{W}} \|\mathbf{X} - \hat{\mathbf{X}}\|_F^2 = \min_{\mathbf{W}} \|\mathbf{X} - \mathbf{X}\mathbf{W}\mathbf{W}^T\|_F^2$$

两种视角等价，都得到协方差矩阵的特征分解。

### 算法步骤

1. **数据标准化**：$\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{X}}$（中心化）
2. **计算协方差矩阵**：$\Sigma = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$
3. **特征分解**：$\Sigma = \mathbf{W} \Lambda \mathbf{W}^T$
4. **选择主成分**：取前 $k$ 个最大特征值对应的特征向量
5. **投影**：$\mathbf{Z} = \mathbf{X} \mathbf{W}_k$

### 方差解释率

$$\text{解释方差比例}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

$$\text{累计解释方差} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

通常选择累计解释方差达到 85%-95% 的 $k$ 值。

### PCA 与 SVD 的关系

对于数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$：

$$\mathbf{X} = \mathbf{U} \mathbf{S} \mathbf{V}^T$$

- 主成分：$\mathbf{W} = \mathbf{V}$
- 投影数据：$\mathbf{Z} = \mathbf{U} \mathbf{S}$
- 特征值：$\lambda_i = \sigma_i^2 / (n-1)$

SVD 计算更稳定，是实际实现的首选。

### 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class MyPCA:
    """PCA 手动实现"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD 分解
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 特征值
        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        
        # 主成分
        self.components_ = Vt
        
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        
        return self
    
    def transform(self, X):
        return (X - self.mean_) @ self.components_.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_


# 演示：手写数字降维
def pca_demo():
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 标准化
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"原始维度: {X.shape[1]}")
    print(f"降维后维度: {X_pca.shape[1]}")
    print(f"解释方差比: {pca.explained_variance_ratio_}")
    print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('PCA 降维结果')
    
    # 累计解释方差
    plt.subplot(1, 2, 2)
    pca_full = PCA().fit(X_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(cumsum, 'b-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比')
    plt.title('选择主成分数量')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pca

pca_result = pca_demo()
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 计算效率高 | 只能发现线性结构 |
| 有明确的数学解释 | 对异常值敏感 |
| 能去除数据相关性 | 方差最大不一定有意义 |
| 可解释性强（主成分贡献率） | 假设主成分正交 |

---

## 线性判别分析（LDA）

LDA 是**监督学习**的降维方法，利用类别信息找到**类间差异最大、类内差异最小**的投影方向。

### 数学推导

#### 散度矩阵

**类间散度矩阵**：
$$\mathbf{S}_B = \sum_{c=1}^C n_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$$

**类内散度矩阵**：
$$\mathbf{S}_W = \sum_{c=1}^C \sum_{\mathbf{x} \in D_c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^T$$

其中 $\boldsymbol{\mu}_c$ 是第 $c$ 类的均值向量，$\boldsymbol{\mu}$ 是总体均值。

#### 优化目标

$$\max_{\mathbf{w}} J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

**求解**：广义特征值问题

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

等价于：$(\mathbf{S}_W^{-1} \mathbf{S}_B) \mathbf{w} = \lambda \mathbf{w}$

### 维度限制

LDA 最多只能降到 $C-1$ 维（$C$ 为类别数），因为 $\mathbf{S}_B$ 的秩最多为 $C-1$。

### 代码实现

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

def lda_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # LDA 降维
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    print(f"原始维度: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")
    print(f"最大降维维度: {len(np.unique(y)) - 1}")
    print(f"解释方差比: {lda.explained_variance_ratio_}")
    
    # 可视化
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(iris.target_names):
        mask = y == i
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=name, alpha=0.7)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA 降维结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return lda

lda_result = lda_demo()
```

### PCA vs LDA

| 特性 | PCA | LDA |
|------|-----|-----|
| 类型 | 无监督 | 有监督 |
| 目标 | 最大化方差 | 最大化类间/类内比 |
| 最大维度 | 特征数 | 类别数-1 |
| 适用场景 | 数据压缩、去噪 | 分类预处理 |
| 信息利用 | 仅特征 | 特征+标签 |

---

## 流形学习

流形学习假设高维数据实际上位于一个**低维流形**上，通过保持局部几何关系实现非线性降维。

### 流形假设

高维数据 $\mathbf{X} \subset \mathbb{R}^D$ 实际上采样自低维流形 $\mathcal{M} \subset \mathbb{R}^d$，其中 $d \ll D$。

### 等距映射（Isomap）

**核心思想**：保持数据点之间的**测地距离**而非欧氏距离。

**算法步骤**：
1. 构建 k-近邻图
2. 计算图上最短路径（测地距离）
3. 应用 MDS（多维缩放）

$$\text{测地距离} \approx \text{图上最短路径}$$

### 局部线性嵌入（LLE）

**核心思想**：每个点可以由其邻居**线性重构**，保持重构权重不变。

**算法步骤**：
1. 找到每个点的 k 个最近邻
2. 计算重构权重：$\min_W \sum_i \|\mathbf{x}_i - \sum_j W_{ij} \mathbf{x}_j\|^2$
3. 保持权重求解低维嵌入：$\min_Y \sum_i \|\mathbf{y}_i - \sum_j W_{ij} \mathbf{y}_j\|^2$

### t-SNE

**核心思想**：使高维和低维空间的**概率分布相似**。

**高维相似度**（高斯分布）：
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

**低维相似度**（t 分布）：
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**优化目标**：最小化 KL 散度
$$C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**为什么用 t 分布？**：t 分布比高斯分布有更重的尾部，能缓解"拥挤问题"。

### 代码实现

```python
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.datasets import make_swiss_roll, make_s_curve
import matplotlib.pyplot as plt

def manifold_learning_demo():
    """流形学习演示"""
    
    # 生成瑞士卷数据
    X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    
    # 不同降维方法
    methods = {
        'PCA': PCA(n_components=2),
        'Isomap': Isomap(n_components=2, n_neighbors=10),
        'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10),
        't-SNE': TSNE(n_components=2, random_state=42, perplexity=30)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for ax, (name, method) in zip(axes, methods.items()):
        X_transformed = method.fit_transform(X)
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                           c=color, cmap=plt.cm.Spectral, alpha=0.6)
        ax.set_title(f'{name}')
        ax.set_xlabel('维度 1')
        ax.set_ylabel('维度 2')
        plt.colorbar(scatter, ax=ax, label='原始位置')
    
    plt.suptitle('流形学习方法比较（瑞士卷数据）', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return methods

manifold_methods = manifold_learning_demo()
```

### t-SNE 参数调优

```python
def tsne_parameter_demo():
    """t-SNE 参数调优演示"""
    
    X, y = load_digits(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    # 不同 perplexity 值
    perplexities = [5, 30, 50, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for ax, perp in zip(axes, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
        ax.set_title(f't-SNE (perplexity={perp})')
        ax.set_xlabel('维度 1')
        ax.set_ylabel('维度 2')
    
    plt.suptitle('t-SNE perplexity 参数影响', fontsize=14)
    plt.tight_layout()
    plt.show()

tsne_parameter_demo()
```

### 流形学习方法比较

| 方法 | 保持的信息 | 优点 | 缺点 |
|------|-----------|------|------|
| Isomap | 全局测地距离 | 保持全局结构 | 计算慢，对噪声敏感 |
| LLE | 局部线性关系 | 计算较快 | 对参数敏感 |
| t-SNE | 局部概率分布 | 可视化效果最好 | 计算慢，不能变换新数据 |

---

## 自编码器

自编码器是**神经网络**实现的降维方法，通过**学习恒等映射**来提取数据特征。

### 网络结构

```
输入层    编码器    瓶颈层    解码器    输出层
  x   →  f(x)  →   h    →  g(h)  →   x̂
  d        →        k         →        d
  
目标：min ||x - x̂||²
```

- **编码器**：$\mathbf{h} = f(\mathbf{x}) = \sigma(\mathbf{W} \mathbf{x} + \mathbf{b})$
- **解码器**：$\hat{\mathbf{x}} = g(\mathbf{h}) = \sigma(\mathbf{W}' \mathbf{h} + \mathbf{b}')$
- **瓶颈层**：维度 $k < d$，迫使网络学习压缩表示

### 变体

| 类型 | 特点 | 应用 |
|------|------|------|
| 稀疏自编码器 | 加入稀疏约束 | 特征学习 |
| 去噪自编码器 | 输入加噪声 | 鲁棒表示 |
| 变分自编码器 (VAE) | 学习概率分布 | 生成模型 |
| 卷积自编码器 | 卷积层编码 | 图像处理 |

### 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    """简单自编码器"""
    
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_autoencoder(X, encoding_dim=2, epochs=50):
    """训练自编码器"""
    
    # 数据准备
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型初始化
    model = Autoencoder(X.shape[1], encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            decoded, _ = model(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    # 提取编码
    model.eval()
    with torch.no_grad():
        _, encoded = model(X_tensor)
    
    return model, encoded.numpy(), losses


# 使用 sklearn 简化演示
from sklearn.neural_network import MLPRegressor

def sklearn_autoencoder_demo(X, encoding_dim=2):
    """使用 sklearn 实现简单自编码器"""
    
    # 编码器：输入 -> 瓶颈
    encoder = MLPRegressor(
        hidden_layer_sizes=(64, encoding_dim),
        activation='relu',
        max_iter=200,
        random_state=42
    )
    
    # 训练（目标是重构自身）
    encoder.fit(X, X)
    
    # 提取中间层表示
    # 这里简化处理，实际需要访问隐藏层
    print("自编码器训练完成")
    
    return encoder

# 演示
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
autoencoder, encoded, losses = train_autoencoder(X, encoding_dim=2, epochs=30)

# 可视化结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(encoded[:, 0], encoded[:, 1], c=digits.target, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('编码维度 1')
plt.ylabel('编码维度 2')
plt.title('自编码器降维结果')

plt.tight_layout()
plt.show()
```

---

## 降维方法比较与选择

### 综合对比

| 方法 | 类型 | 时间复杂度 | 保持信息 | 新数据变换 | 可解释性 |
|------|------|-----------|---------|-----------|---------|
| PCA | 线性 | $O(d^2 n + d^3)$ | 全局方差 | ✓ | 高 |
| LDA | 有监督线性 | $O(d^2 n)$ | 类别区分 | ✓ | 高 |
| Isomap | 非线性 | $O(n^2 \log n)$ | 测地距离 | ✗ | 中 |
| LLE | 非线性 | $O(dn^2)$ | 局部线性 | ✗ | 低 |
| t-SNE | 非线性 | $O(n^2)$ | 局部概率 | ✗ | 低 |
| 自编码器 | 非线性 | 取决于网络 | 重构误差 | ✓ | 低 |

### 选择指南

```
任务需求                      推荐方法
──────────────────────────────────────────────
数据压缩、去噪                PCA
分类预处理                    LDA
数据可视化                    t-SNE / UMAP
非线性结构发现                流形学习
大规模数据                    PCA / 随机投影
需要变换新数据                PCA / LDA / 自编码器
```

### 实践建议

1. **预处理**：标准化数据（PCA、LDA 对尺度敏感）
2. **探索性分析**：先用 PCA 看数据线性可分性
3. **可视化**：t-SNE 可视化效果最好，但注意参数
4. **下游任务**：根据任务选择（分类用 LDA，聚类用 PCA）
5. **维度选择**：肘部法则或累计方差比

---

## 小结

降维是高维数据处理的核心技术：

- **PCA**：线性降维首选，计算高效，可解释性强
- **LDA**：有监督降维，分类任务的理想预处理
- **流形学习**：处理非线性结构，可视化效果好
- **自编码器**：神经网络方法，可学习复杂映射

选择方法时需考虑：数据特点、任务需求、计算资源、可解释性要求。

---

**上一节**：[聚类分析](./clustering.md)  
**下一节**：[关联规则与异常检测](./association-anomaly.md)
