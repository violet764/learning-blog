# 特征变换

特征变换是通过数学函数将原始特征映射到新的特征空间，旨在改善数据分布、增强模型表达能力、处理非线性关系。

## 核心概念

特征变换是特征工程中的核心技术，通过数学映射改变特征的分布或表达形式。

### 变换类型

根据变换是否保持数据的相对顺序，可以分为以下几类：

| 类型 | 描述 | 示例 |
|------|------|------|
| **单调变换** | 保持数据的相对大小顺序不变 | 对数、平方根、倒数 |
| **非线性变换** | 可能改变数据的相对顺序，引入新的特征关系 | 多项式、核方法 |
| **离散化** | 将连续值映射到离散区间 | 分箱处理 |

### 特征空间映射

特征变换的本质是将原始特征空间映射到一个新的特征空间，使数据在新空间中更适合模型学习：

$$\phi: \mathcal{X} \rightarrow \mathbb{R}^d$$

这种映射可以：
- 使数据分布更接近正态分布
- 将非线性关系转化为线性关系
- 降低噪声影响
- 创造更丰富的特征表示

---

## 一、数值特征变换

数值特征变换主要用于改善数据的统计分布特性，使其更适合模型学习。常见的应用场景包括处理偏态分布、稳定方差、处理极端值等。

### 1. 对数变换

对数变换是最常用的数值变换方法之一，特别适合处理右偏分布的数据。

**公式**：$x' = \log(x + c)$

**为什么有效**：
- 对数函数对大值压缩程度大，对小值压缩程度小
- 可以将乘法关系转化为加法关系
- 有助于稳定方差（当方差与均值相关时）

**适用场景**：
- 右偏分布数据（如收入、房价）
- 方差与均值相关的数据（如计数数据）
- 具有乘法关系的数据（如复利增长）

```python
import numpy as np
from scipy import stats

def log_transform(X, base='natural'):
    """
    对数变换
    
    参数:
        X: 输入数据（必须为正数）
        base: 对数底数类型
            - 'natural': 自然对数(ln)
            - 'log10': 以10为底
            - 'log2': 以2为底
    
    返回:
        变换后的数据
    
    注意: 加1是为了避免log(0)的问题
    """
    if base == 'natural':
        return np.log(X + 1)      # 自然对数 ln(x+1)
    elif base == 'log10':
        return np.log10(X + 1)    # 常用对数 lg(x+1)
    elif base == 'log2':
        return np.log2(X + 1)     # 二进制对数 log2(x+1)
```

**效果对比**：

| 指标 | 原始数据 | 对数变换后 |
|------|----------|------------|
| 偏度 | 高（右偏） | 降低 |
| 峰度 | 高 | 接近正态 |
| 方差 | 大 | 稳定 |

### 2. 幂变换

幂变换是一族更通用的变换方法，通过对数似然估计自动寻找最优的变换参数λ，使变换后的数据尽可能接近正态分布。

**Box-Cox 变换**（仅适用于正数）：

$$x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log(x) & \lambda = 0 \end{cases}$$

**λ参数的含义**：
- λ = 1：无变换（原始数据）
- λ = 0：对数变换
- λ = 0.5：平方根变换
- λ = -1：倒数变换

**Yeo-Johnson 变换**（适用于任意实数）：

Yeo-Johnson是Box-Cox的扩展版本，可以处理零值和负数，实际应用中更加灵活。

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox变换：只适用于正数数据
# 通过最大似然估计自动寻找最优λ参数
pt_boxcox = PowerTransformer(method='box-cox')
X_boxcox = pt_boxcox.fit_transform(X_positive)

# Yeo-Johnson变换：支持零和负数
# 是Box-Cox的扩展版本，适用范围更广
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
X_yeojohnson = pt_yeojohnson.fit_transform(X)

# 查看最优λ值：λ决定了变换的形式
# λ=1时无变换，λ=0时为对数变换
print(f"最优λ: {pt_yeojohnson.lambdas_}")
```

### 3. 分位数变换

分位数变换通过将数据的分位数映射到目标分布的分位数，强制改变数据的分布形状。这是一种非常强力的变换，可以将任意分布转换为正态分布或均匀分布。

**核心思想**：
- 计算原始数据每个值的分位数（百分位排名）
- 将该分位数对应的值替换为目标分布中相同分位数的值

**优点**：对异常值非常鲁棒，可以处理多峰分布
**缺点**：可能扭曲数据的原始结构，过度变换

```python
from sklearn.preprocessing import QuantileTransformer

# 映射到正态分布：强制数据服从高斯分布
# 对于偏态分布或异常值多的数据效果显著
qt_normal = QuantileTransformer(output_distribution='normal')
X_normal = qt_normal.fit_transform(X)

# 映射到均匀分布：将数据映射到[0,1]均匀分布
# 使数据在取值范围内分布更加均匀
qt_uniform = QuantileTransformer(output_distribution='uniform')
X_uniform = qt_uniform.fit_transform(X)
```

### 变换方法对比

选择合适的数值变换方法需要考虑数据特性和模型需求：

| 方法 | 适用数据 | 优点 | 缺点 |
|------|----------|------|------|
| 对数变换 | 正数、右偏分布 | 简单直观、可解释性强 | 只能处理正数 |
| Box-Cox | 正数 | 自动优化λ、效果稳定 | 要求正数 |
| Yeo-Johnson | 任意实数 | 适用范围最广 | 计算稍复杂、结果略难解释 |
| 分位数变换 | 任意数据 | 强制正态化、对异常值鲁棒 | 可能过度变换、破坏原始结构 |

---

## 二、多项式特征

多项式特征通过生成原始特征的高次项和交互项，使线性模型能够捕捉非线性关系。这是一种显式的特征空间扩展方法。

### 生成原理

将原始特征扩展为多项式组合。例如，对于两个特征 $x_1, x_2$，二次多项式扩展为：

$$\phi(x_1, x_2) = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$$

这样，原本只能学习线性关系的模型，现在可以学习二次多项式关系：
$$y = w_0 + w_1x_1 + w_2x_2 + w_3x_1^2 + w_4x_1x_2 + w_5x_2^2$$

### 代码实现

```python
from sklearn.preprocessing import PolynomialFeatures

# 生成2次多项式特征
# degree=2: 最高次数为2
# include_bias=False: 不添加常数项1（避免线性模型的共线性问题）
# interaction_only=False: 包含平方项和交互项
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)

# 查看生成的特征名称
# 例如输入['x1', 'x2']，输出为['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
feature_names = poly.get_feature_names_out(['x1', 'x2'])
print(feature_names)
```

### 参数说明

理解这些参数对于正确使用多项式特征至关重要：

| 参数 | 说明 | 效果 |
|------|------|------|
| `degree=2` | 多项式最高次数 | 次数越高，模型复杂度越高，过拟合风险越大 |
| `include_bias=False` | 是否添加常数项1 | 设为False避免线性模型的共线性问题 |
| `interaction_only=True` | 仅生成交互项 | 减少特征数量，避免特征自身的高次项 |

**特征数量增长**：对于n个特征，degree=d的多项式扩展将产生 $C(n+d, d)$ 个特征，特征数量随次数呈组合爆炸增长。

### 交互特征

有时我们只关心特征之间的交互效应，而不需要单个特征的高次项。例如，"面积 × 房间数" 可能比 "面积²" 更有意义。

```python
# 仅生成交互特征（不生成平方项）
# interaction_only=True: 只生成交互项，如x1*x2，不生成x1^2
# 可以减少特征数量，避免特征自身的高次项
poly_interact = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interact = poly_interact.fit_transform(X)
# 输入[x1, x2]，输出: [x1, x2, x1*x2]
```

---

## 三、离散化与分箱

离散化（Discretization）是将连续特征转换为离散特征的过程。这可以降低噪声影响、处理非线性关系，并使某些模型（如决策树）更高效。

### 分箱方法对比

不同的分箱策略适用于不同的数据分布和分析目的：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 等宽分箱 | 每个区间宽度相等 | 简单直观、易于理解 | 对异常值敏感，可能导致某些箱样本过少 |
| 等频分箱 | 每个区间样本数相等 | 各箱样本均衡 | 边界可能不平滑，相邻值可能被分到不同箱 |
| K-means分箱 | 使用聚类确定边界 | 自动适应数据分布 | 计算量大，需要预设箱数 |

### 代码实现

```python
from sklearn.preprocessing import KBinsDiscretizer

# 等宽分箱：每个区间的宽度相等
# encode='ordinal': 返回箱号（0,1,2,...）
discretizer_uniform = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_binned_uniform = discretizer_uniform.fit_transform(X)

# 等频分箱：每个区间包含大致相等的样本数
# 适合偏态分布，避免某些箱样本过少
discretizer_quantile = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned_quantile = discretizer_quantile.fit_transform(X)

# K-means分箱：使用K-means聚类确定分箱边界
# 自动适应数据分布，但计算量较大
discretizer_kmeans = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
X_binned_kmeans = discretizer_kmeans.fit_transform(X)

# 查看分箱边界：每个特征的分箱切分点
print(discretizer_uniform.bin_edges_)
```

### 最优分箱数确定

分箱数量是一个重要的超参数。箱数太少会丢失信息，太多则无法起到平滑作用。基于决策树的分箱方法可以自动确定最优分箱边界：

**原理**：训练一个浅层决策树，树节点的划分阈值自然形成分箱边界。这种方法的优势在于：
- 分箱结果与目标变量相关（有监督）
- 自动适应数据分布
- 信息损失最小化

```python
from sklearn.tree import DecisionTreeClassifier

def optimal_binning(X, y, max_bins=10):
    """
    基于决策树的最优分箱
    
    原理：训练一个浅层决策树，树节点划分阈值即为分箱边界
    优点：分箱结果与目标变量相关，信息损失少
    
    参数:
        X: 特征值（一维数组）
        y: 目标变量
        max_bins: 最大分箱数
    
    返回:
        分箱边界阈值列表
    """
    dt = DecisionTreeClassifier(max_leaf_nodes=max_bins)
    dt.fit(X.reshape(-1, 1), y)
    # tree_.threshold存储所有节点的阈值，-2表示叶节点
    thresholds = dt.tree_.threshold[dt.tree_.threshold > -2]
    return sorted(thresholds)
```

---

## 四、核方法

核方法（Kernel Methods）是一种强大的非线性特征变换技术，通过核技巧隐式地将数据映射到高维特征空间，无需显式计算高维映射。

### 核技巧原理

直接计算高维映射 $\phi(x)$ 可能维度极高，甚至无穷维。核技巧通过核函数直接计算高维空间中的内积：

$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

这样，我们可以在高维空间中进行计算，而无需显式地构造高维特征。

### 常用核函数

不同的核函数对应不同的特征空间映射：

| 核函数 | 公式 | 适用场景 |
|--------|------|----------|
| 线性核 | $K = x \cdot x'$ | 线性可分数据，计算效率最高 |
| 多项式核 | $K = (x \cdot x' + c)^d$ | 特征间存在多项式关系 |
| RBF核（高斯核） | $K = \exp(-\gamma\|x-x'\|^2)$ | 复杂非线性关系，最常用 |
| Sigmoid核 | $K = \tanh(\gamma x \cdot x' + r)$ | 模拟神经网络激活函数 |

### Kernel PCA

Kernel PCA是PCA的非线性扩展，通过核技巧在高维空间进行主成分分析，适合处理非线性结构的数据降维。

```python
from sklearn.decomposition import KernelPCA

# RBF核的Kernel PCA
# kernel='rbf': 径向基函数核，能处理复杂非线性结构
# n_components=2: 降维到2维
# gamma=10: RBF核参数，越大映射越复杂
kpca_rbf = KernelPCA(kernel='rbf', n_components=2, gamma=10)

# 多项式核的Kernel PCA
# kernel='poly': 多项式核，适合多项式关系的数据
# degree=3: 多项式次数
kpca_poly = KernelPCA(kernel='poly', n_components=2, degree=3)

X_rbf = kpca_rbf.fit_transform(X)   # RBF核变换结果
X_poly = kpca_poly.fit_transform(X)  # 多项式核变换结果
```

---

## 五、特殊数据变换

除了通用的数值变换外，不同类型的数据有其特有的变换方法。

### 1. 时间特征

时间数据包含丰富的周期性和趋势信息，需要通过特征工程提取出来。时间特征的难点在于处理周期性：例如月份1和12在数值上差11，但实际上只差1个月。

```python
import pandas as pd

def extract_time_features(df, time_col):
    """
    从时间列提取多种特征
    
    参数:
        df: DataFrame
        time_col: 时间列的列名
    
    返回:
        添加了时间特征的DataFrame
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # ===== 基础时间特征 =====
    df['year'] = df[time_col].dt.year        # 年份
    df['month'] = df[time_col].dt.month      # 月份(1-12)
    df['day'] = df[time_col].dt.day          # 日(1-31)
    df['hour'] = df[time_col].dt.hour        # 小时(0-23)
    df['dayofweek'] = df[time_col].dt.dayofweek  # 星期几(0=周一,6=周日)
    
    # ===== 衍生特征 =====
    # 是否周末：周六周日为1，工作日为0
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    # 是否月初：1号返回1
    df['is_month_start'] = df[time_col].dt.is_month_start.astype(int)
    # 是否月末：最后一天返回1
    df['is_month_end'] = df[time_col].dt.is_month_end.astype(int)
    
    # ===== 周期性编码 =====
    # 问题：月份1和12在数值上差11，但实际上只差1个月
    # 解决：用正弦/余弦编码，保持周期性
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)  # 月份正弦编码
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)  # 月份余弦编码
    
    return df
```

### 2. 文本特征

文本数据需要转换为数值向量才能被模型处理。常用的方法包括词袋模型和TF-IDF，它们将文本表示为词频向量。

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 词袋模型（Bag of Words）
# 将文本转换为词频向量，忽略词序
# max_features=1000: 只保留出现频率最高的1000个词
# ngram_range=(1, 2): 同时考虑单个词和相邻两个词的组合
count_vec = CountVectorizer(max_features=1000, ngram_range=(1, 2))
X_bow = count_vec.fit_transform(texts)

# TF-IDF：词频-逆文档频率
# 不仅考虑词频，还考虑词的区分度（稀有词权重高）
# 常见词（如"的"、"是"）会被降权
tfidf_vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(texts)
```

### 3. 频域特征

对于时序信号，时域特征（均值、方差等）可能无法完全描述信号特性。通过快速傅里叶变换（FFT）可以将信号转换到频域，提取频率相关的特征。这些特征在语音识别、振动分析、金融时序等领域有重要应用。

```python
def extract_frequency_features(signal, sampling_rate=1000):
    """
    从时序信号中提取频域特征
    
    参数:
        signal: 一维时序信号
        sampling_rate: 采样频率（Hz）
    
    返回:
        频域特征字典
    """
    # FFT变换：将时域信号转换为频域
    fft = np.fft.fft(signal)
    # 频率轴：计算每个FFT点对应的频率
    freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
    # 功率谱：信号在每个频率上的能量
    power = np.abs(fft) ** 2
    
    # 提取关键频域特征
    features = {
        # 主频：功率最大的频率，代表信号的主要振动模式
        'dominant_freq': freqs[np.argmax(power)],
        # 频谱质心：频率的加权平均，反映频谱的"重心"
        'spectral_centroid': np.sum(freqs * power) / np.sum(power),
        # 频谱能量：总功率，反映信号的总体强度
        'spectral_energy': np.sum(power),
        # 频谱熵：频率分布的复杂度，值越大越复杂
        'spectral_entropy': -np.sum((power/np.sum(power)) * np.log(power/np.sum(power) + 1e-10))
    }
    return features
```

---

## 六、变换方法选择指南

选择合适的特征变换方法需要综合考虑数据特性、模型类型和业务理解。

### 按数据分布选择

根据数据的统计特性选择变换方法：

| 数据特征 | 推荐变换 | 原因 |
|----------|----------|------|
| 右偏分布 | 对数变换、Box-Cox | 压缩大值，使分布更对称 |
| 左偏分布 | 幂变换（λ>1） | 放大大值，使分布更对称 |
| 多峰分布 | 分位数变换 | 强制平滑为单峰分布 |
| 有异常值 | 分位数变换、稳健变换 | 降低异常值影响 |

### 按模型需求选择

不同模型对输入数据的假设不同：

| 模型类型 | 推荐变换 | 原因 |
|----------|----------|------|
| 线性模型（线性回归、逻辑回归） | 对数、幂变换使其近似正态 | 线性模型假设误差正态分布 |
| 树模型（决策树、随机森林） | 通常不需要 | 树模型对特征尺度不敏感 |
| 神经网络 | 标准化、归一化 | 加速梯度下降收敛 |
| SVM | 标准化 + 核方法 | 距离计算需要统一尺度 |

### 代码示例：完整的变换管道

在实际项目中，通常需要组合多种变换方法。使用Pipeline可以确保变换的一致性和可复现性：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, StandardScaler

# 构建变换管道：依次应用多个变换
transform_pipeline = Pipeline([
    # 第1步：幂变换，使数据更接近正态分布
    ('power', PowerTransformer(method='yeo-johnson')),
    # 第2步：多项式特征扩展，捕捉非线性关系
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    # 第3步：标准化，使特征具有相同的尺度
    ('scaler', StandardScaler())
])

X_transformed = transform_pipeline.fit_transform(X)
```

---

## 小结

| 变换类型 | 核心目的 | 关键方法 |
|----------|----------|----------|
| 数值变换 | 改善分布 | 对数、Box-Cox、Yeo-Johnson |
| 多项式扩展 | 捕捉非线性 | PolynomialFeatures |
| 离散化 | 降低噪声 | 等宽/等频/K-means分箱 |
| 核方法 | 高维映射 | Kernel PCA、核技巧 |

选择合适的特征变换需要结合数据特性、模型需求和业务理解进行综合考量。
