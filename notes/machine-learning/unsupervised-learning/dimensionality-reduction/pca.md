# 降维技术

## 1. 降维概述

降维是通过数学变换将高维数据映射到低维空间的技术，旨在保留数据的主要结构和信息，同时减少计算复杂度和噪声影响。

### 1.1 降维的目的

- **数据可视化**：将高维数据投影到2D或3D空间
- **特征提取**：发现数据的内在结构
- **去除噪声**：减少不相关特征的影响
- **加速计算**：降低模型训练和预测的时间复杂度

### 1.2 降维方法分类

#### 线性降维
- 主成分分析（PCA）
- 线性判别分析（LDA）
- 因子分析（FA）

#### 非线性降维
- t-SNE
- 自编码器
- 等距映射（Isomap）
- 局部线性嵌入（LLE）

## 2. 主成分分析（PCA）数学原理详解

### 2.1 算法概述

主成分分析（Principal Component Analysis, PCA）是一种经典的线性降维方法，其核心思想是通过正交变换将原始特征转换为一组线性不相关的主成分，这些主成分按照方差大小排序，保留数据的主要信息。

### 2.2 数学基础

#### 2.2.1 方差与协方差

**方差**：衡量单个变量的离散程度
$$\text{Var}(X) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$$

**协方差**：衡量两个变量的线性相关性
$$\text{Cov}(X,Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$

**协方差矩阵**：对于d维数据，协方差矩阵为：
$$\Sigma = \begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1,X_2) & \cdots & \text{Cov}(X_1,X_d) \\
\text{Cov}(X_2,X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2,X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_d,X_1) & \text{Cov}(X_d,X_2) & \cdots & \text{Var}(X_d)
\end{bmatrix}$$

### 2.3 PCA的数学推导

#### 2.3.1 问题形式化

给定数据中心化后的数据矩阵 $X \in \mathbb{R}^{n \times d}$（每行一个样本，每列一个特征，且已中心化），PCA的目标是找到一组正交基 $W = [w_1, w_2, \dots, w_d] \in \mathbb{R}^{d \times d}$，使得投影后的数据：
$$Z = XW$$
满足：

1. 主成分之间不相关：$\text{Cov}(Z_i, Z_j) = 0$ 对于 $i \neq j$
2. 主成分按方差递减排序：$\text{Var}(Z_1) \geq \text{Var}(Z_2) \geq \cdots \geq \text{Var}(Z_d)$

#### 2.3.2 最大方差视角

**第一主成分**：寻找单位向量 $w_1$ 使得投影方差最大：
$$\max_{w_1} \text{Var}(Xw_1) = \max_{w_1} w_1^T \Sigma w_1$$
约束条件：$w_1^T w_1 = 1$

使用拉格朗日乘子法：
$$L(w_1, \lambda_1) = w_1^T \Sigma w_1 - \lambda_1(w_1^T w_1 - 1)$$

求导并令为零：
$$\frac{\partial L}{\partial w_1} = 2\Sigma w_1 - 2\lambda_1 w_1 = 0$$
$$\Rightarrow \Sigma w_1 = \lambda_1 w_1$$

这正是特征值问题！最大方差等于最大特征值 $\lambda_1$。

**第k主成分**：在已找到前k-1个主成分的基础上，寻找与它们正交的单位向量 $w_k$ 使得方差最大：
$$\max_{w_k} w_k^T \Sigma w_k$$
约束条件：$w_k^T w_k = 1$，且 $w_k^T w_j = 0$ 对于 $j = 1, \dots, k-1$

通过类似推导可得 $\Sigma w_k = \lambda_k w_k$。

#### 2.3.3 最小重建误差视角

PCA也可以从最小化重建误差的角度推导。将数据投影到k维子空间后重建：
$$\hat{X} = XWW^T$$

重建误差：
$$\|X - \hat{X}\|_F^2 = \|X - XWW^T\|_F^2$$

最小化重建误差等价于最大化投影方差。

### 2.4 PCA的数学性质

#### 2.4.1 特征值分解与奇异值分解

**特征值分解**：对于协方差矩阵 $\Sigma$
$$\Sigma = W\Lambda W^T$$
其中 $W$ 是正交矩阵，$\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$ 是特征值对角矩阵。

**奇异值分解（SVD）**：对于数据矩阵 $X$
$$X = U\Sigma V^T$$
其中 $U$ 和 $V$ 是正交矩阵，$\Sigma$ 是奇异值矩阵。

PCA可以通过SVD直接计算：
- 主成分：$W = V$
- 投影数据：$Z = U\Sigma$
- 特征值：$\lambda_i = \sigma_i^2/(n-1)$

#### 2.4.2 方差解释率

**单个主成分的解释方差**：
$$\text{解释方差比例} = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

**累计解释方差**：
$$\text{累计解释方差} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

#### 2.4.3 主成分的不相关性

投影后的数据协方差矩阵为对角矩阵：
$$\text{Cov}(Z) = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$$

这证明了主成分之间确实不相关。

### 2.3 算法步骤

1. 数据标准化（中心化）
2. 计算协方差矩阵$\mathbf{X}^T\mathbf{X}$
3. 计算协方差矩阵的特征值和特征向量
4. 按特征值大小降序排列特征向量
5. 选择前k个特征向量构成投影矩阵
6. 将数据投影到低维空间

## 3. 线性判别分析（LDA）

### 3.1 算法原理

LDA是监督学习的降维方法，旨在找到投影方向，使得类间方差最大，类内方差最小。

### 3.2 数学推导

#### 3.2.1 类间散度矩阵

$$ \mathbf{S}_B = \sum_{c=1}^C n_c(\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T $$

其中：
- $n_c$：第c类的样本数
- $\boldsymbol{\mu}_c$：第c类的均值向量
- $\boldsymbol{\mu}$：总体均值向量

#### 3.2.2 类内散度矩阵

$$ \mathbf{S}_W = \sum_{c=1}^C \sum_{\mathbf{x} \in D_c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^T $$

#### 3.2.3 优化目标

$$ \max_{\mathbf{w}} J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}} $$

解为广义特征值问题：
$$ \mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w} $$

## 4. t-SNE（t分布随机邻域嵌入）

### 4.1 算法原理

t-SNE是一种非线性降维方法，特别适合高维数据的可视化。它通过保留数据点之间的局部相似性来映射到低维空间。

### 4.2 数学推导

#### 4.2.1 高维空间相似度

使用高斯分布计算条件概率：
$$ p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)} $$

对称化：
$$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n} $$

#### 4.2.2 低维空间相似度

使用t分布计算：
$$ q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}} $$

#### 4.2.3 优化目标

最小化KL散度：
$$ C = \sum_i KL(P_i\|Q_i) = \sum_i \sum_j p_{ij} \log\frac{p_{ij}}{q_{ij}} $$

## 5. 自编码器

### 5.1 网络结构

自编码器由编码器和解码器组成：
- **编码器**：$\mathbf{h} = f(\mathbf{x}) = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$
- **解码器**：$\mathbf{\hat{x}} = g(\mathbf{h}) = \sigma(\mathbf{W}'\mathbf{h} + \mathbf{b}')$

### 5.2 损失函数

$$ L(\mathbf{x}, \mathbf{\hat{x}}) = \|\mathbf{x} - \mathbf{\hat{x}}\|^2 $$

### 5.3 变体

- **稀疏自编码器**：加入稀疏约束
- **去噪自编码器**：对输入加入噪声
- **变分自编码器**：学习数据的概率分布

## 6. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

# 加载数据集
print("=== 降维技术比较 ===")

# 1. 手写数字数据集（用于PCA和t-SNE）
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# 2. 鸢尾花数据集（用于LDA）
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 3. 瑞士卷数据集（用于非线性降维）
X_swiss, y_swiss = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits)
X_iris_scaled = scaler.fit_transform(X_iris)
X_swiss_scaled = scaler.fit_transform(X_swiss)

# PCA降维
print("\n=== PCA主成分分析 ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits_scaled)

print("解释方差比:", pca.explained_variance_ratio_)
print("累计解释方差:", np.cumsum(pca.explained_variance_ratio_))

# 可视化PCA结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('PCA降维结果')

# 解释方差比图
plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('PCA累计解释方差')
plt.grid(True)

plt.tight_layout()
plt.show()

# LDA降维
print("\n=== LDA线性判别分析 ===")
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_iris_scaled, y_iris)

print("类间散度矩阵特征值:", lda.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_iris, cmap='viridis', alpha=0.7)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA降维结果')
plt.colorbar()
plt.show()

# t-SNE降维
print("\n=== t-SNE非线性降维 ===")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE降维结果')

# 比较PCA和t-SNE
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA降维结果（对比）')

plt.tight_layout()
plt.show()

# 自编码器实现
print("\n=== 自编码器降维 ===")

class SimpleAutoencoder:
    def __init__(self, encoding_dim=2):
        self.encoding_dim = encoding_dim
        self.encoder = None
        self.decoder = None
    
    def fit(self, X, epochs=100, learning_rate=0.01):
        n_features = X.shape[1]
        
        # 简单的自编码器实现
        self.encoder = MLPRegressor(hidden_layer_sizes=(64, 32, self.encoding_dim), 
                                   max_iter=epochs, learning_rate_init=learning_rate)
        self.decoder = MLPRegressor(hidden_layer_sizes=(32, 64, n_features), 
                                   max_iter=epochs, learning_rate_init=learning_rate)
        
        # 训练编码器
        encoded = self.encoder.fit(X, X).predict(X)
        
        # 训练解码器
        self.decoder.fit(encoded, X)
        
        return self
    
    def transform(self, X):
        return self.encoder.predict(X)
    
    def inverse_transform(self, encoded):
        return self.decoder.predict(encoded)

# 使用自编码器
autoencoder = SimpleAutoencoder(encoding_dim=2)
autoencoder.fit(X_digits_scaled, epochs=50)
X_autoencoded = autoencoder.transform(X_digits_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_autoencoded[:, 0], X_autoencoded[:, 1], c=y_digits, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('自编码器维度1')
plt.ylabel('自编码器维度2')
plt.title('自编码器降维结果')
plt.show()

# 多种降维方法比较
print("\n=== 多种降维方法比较 ===")

methods = {
    'PCA': PCA(n_components=2),
    'LDA': LinearDiscriminantAnalysis(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42),
    'Autoencoder': SimpleAutoencoder(encoding_dim=2)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, method) in enumerate(methods.items()):
    if name == 'LDA':
        X_reduced = method.fit_transform(X_iris_scaled, y_iris)
        y_plot = y_iris
    elif name == 'Autoencoder':
        method.fit(X_digits_scaled, epochs=50)
        X_reduced = method.transform(X_digits_scaled)
        y_plot = y_digits
    else:
        X_reduced = method.fit_transform(X_digits_scaled)
        y_plot = y_digits
    
    scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_plot, 
                             cmap='tab10', alpha=0.6)
    axes[i].set_title(f'{name}降维')
    axes[i].set_xlabel(f'{name} 1')
    axes[i].set_ylabel(f'{name} 2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# 降维在分类任务中的应用
print("\n=== 降维在分类任务中的应用 ===")
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 原始高维数据
svc_original = SVC()
scores_original = cross_val_score(svc_original, X_digits_scaled, y_digits, cv=5)

# PCA降维后的数据
pca_10 = PCA(n_components=10)
X_pca_10 = pca_10.fit_transform(X_digits_scaled)
svc_pca = SVC()
scores_pca = cross_val_score(svc_pca, X_pca_10, y_digits, cv=5)

print(f"原始数据分类准确率: {scores_original.mean():.3f} ± {scores_original.std():.3f}")
print(f"PCA降维后分类准确率: {scores_pca.mean():.3f} ± {scores_pca.std():.3f}")

# 特征重要性分析
print("\n=== PCA特征重要性分析 ===")
# 计算每个原始特征对主成分的贡献
feature_importance = np.abs(pca.components_).sum(axis=0)

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('原始特征索引')
plt.ylabel('特征重要性（对主成分的贡献）')
plt.title('PCA特征重要性分析')
plt.show()
```

## 7. 降维评价指标

### 7.1 重构误差

对于自编码器等重构型方法：
$$ \text{Reconstruction Error} = \frac{1}{n} \sum_{i=1}^n \|\mathbf{x}_i - \mathbf{\hat{x}}_i\|^2 $$

### 7.2 可解释方差比

对于PCA：
$$ \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j} $$

### 7.3 最近邻保持度

衡量降维后邻居关系的保持程度：
$$ \text{Neighborhood Preservation} = \frac{1}{n} \sum_{i=1}^n \frac{|N_i^{high} \cap N_i^{low}|}{|N_i^{high}|} $$

## 8. 应用场景

### 8.1 数据可视化
- 高维数据的2D/3D可视化
- 聚类结果的直观展示

### 8.2 特征工程
- 去除冗余特征
- 提取有意义的潜在特征

### 8.3 图像处理
- 图像压缩和去噪
- 人脸识别特征提取

### 8.4 文本分析
- 文档主题建模
- 词向量降维

## 9. 优缺点分析

### 9.1 PCA
**优点：**
- 计算效率高
- 有明确的数学解释
- 能去除数据相关性

**缺点：**
- 只能发现线性结构
- 对异常值敏感
- 方差最大的方向不一定最有意义

### 9.2 LDA
**优点：**
- 充分利用类别信息
- 能最大化类间区分度
- 对分类任务特别有效

**缺点：**
- 需要类别标签
- 对数据分布有假设
- 最多只能降到C-1维（C为类别数）

### 9.3 t-SNE
**优点：**
- 能发现非线性结构
- 可视化效果优秀
- 对局部结构保持好

**缺点：**
- 计算复杂度高
- 结果对参数敏感
- 不能用于新样本的变换

### 9.4 自编码器
**优点：**
- 能学习复杂的非线性映射
- 可以处理各种类型的数据
- 有强大的表示学习能力

**缺点：**
- 训练时间长
- 需要大量数据
- 结果难以解释

## 10. 实践建议

1. **数据预处理**：必须进行标准化处理
2. **方法选择**：根据数据特性和任务目标选择合适方法
3. **维度选择**：使用肘部法则或累计方差比确定最佳维度
4. **结果验证**：结合下游任务验证降维效果
5. **参数调优**：对t-SNE等参数敏感的方法进行参数优化

---

[下一节：模型评估与特征工程](../model-evaluation-features/index.md)