# 数据预处理

数据预处理是机器学习流程中的关键环节，涉及数据清洗、缺失值处理、异常值检测、标准化和编码等技术，旨在提高数据质量、增强模型性能和稳定性。

## 核心概念

### 数据质量度量

在开始预处理之前，需要评估数据的质量状况。以下是三个关键的质量指标：

| 指标 | 公式 | 说明 |
|------|------|------|
| 完整性 | $Completeness = \frac{N_{valid}}{N_{total}} \times 100\%$ | 有效数据占比，反映缺失值严重程度 |
| 准确性 | 与真实值的偏差程度 | 数据正确性，衡量测量误差 |
| 一致性 | 违反约束规则的程度 | 数据逻辑性，如年龄不能为负 |

### 缺失机制分类

理解缺失值的产生机制对于选择正确的处理方法至关重要：

| 类型 | 英文 | 特点 | 处理建议 |
|------|------|------|----------|
| 完全随机缺失 | MCAR (Missing Completely At Random) | 缺失与所有变量无关，纯属偶然 | 可直接删除，不会引入偏差 |
| 随机缺失 | MAR (Missing At Random) | 缺失仅与观测到的其他变量有关 | 使用插补法，如KNN、MICE |
| 非随机缺失 | MNAR (Missing Not At Random) | 缺失与未观测值本身有关 | 需要模型法或领域知识处理 |

**判断技巧**：如果高收入人群更倾向于不填写收入，这属于MNAR；如果收入缺失与年龄有关（年轻人更可能不填），这属于MAR。

---

## 一、缺失值处理

缺失值是数据收集过程中常见的问题，可能由设备故障、用户拒绝回答、数据传输错误等原因造成。处理不当会导致信息损失或引入偏差。

### 1. 删除法

删除法是最简单直接的处理方式，但可能导致信息损失。

**适用条件**：
- 数据属于MCAR机制（缺失完全随机）
- 缺失比例较低（通常<5%）
- 样本量足够大，删除后仍有充足数据

```python
import pandas as pd
import numpy as np

# 删除含有缺失值的行（任何一列有缺失就删除该行）
df_dropped = df.dropna()

# 删除含有缺失值的列（任何一行有缺失就删除该列）
df_dropped_cols = df.dropna(axis=1)

# 仅删除全部缺失的行（只有当一行所有值都缺失时才删除）
df_dropped_all = df.dropna(how='all')

# 删除缺失超过阈值的行（保留至少70%非缺失值的行）
# thresh参数指定每行至少需要有多少个非缺失值
df_dropped_thresh = df.dropna(thresh=len(df.columns) * 0.7)
```

### 2. 单变量插补法

单变量插补法仅利用当前特征本身的信息来填充缺失值，不考虑特征间的相关性。这种方法计算简单、速度快，适合作为基准方法。

```python
from sklearn.impute import SimpleImputer

# 均值插补：用该特征的平均值填充缺失值
# 适用于数值特征，但当数据有极端值时效果不佳
imputer_mean = SimpleImputer(strategy='mean')

# 中位数插补：用该特征的中位数填充缺失值
# 对异常值更鲁棒，适合偏态分布的数据
imputer_median = SimpleImputer(strategy='median')

# 众数插补：用该特征出现频率最高的值填充缺失值
# 主要用于类别特征，也可用于离散数值特征
imputer_mode = SimpleImputer(strategy='most_frequent')

# 常数插补：用指定的常数值填充缺失值
# fill_value=0 表示用0填充，可根据业务需求设置其他值
imputer_const = SimpleImputer(strategy='constant', fill_value=0)

# 应用插补：fit_transform() 先拟合数据学习填充值，再进行变换
X_imputed = imputer_mean.fit_transform(X)
```

**各方法特点**：

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 均值 | 保持样本均值 | 降低方差、扭曲相关性 | 正态分布 |
| 中位数 | 对异常值鲁棒 | 可能引入偏差 | 偏态分布 |
| 众数 | 简单直接 | 信息损失 | 类别特征 |

### 3. 多变量插补法

多变量插补法利用特征间的相关性来预测缺失值，通常比单变量方法更准确，但计算成本更高。

**K近邻插补**：利用K个最近邻样本的加权平均值来插补缺失值。原理是"相似的样本应该有相似的值"，通过欧氏距离找到最相似的K个样本进行加权平均。

```python
from sklearn.impute import KNNImputer

# n_neighbors=5: 使用5个最近邻样本
# weights='distance': 距离越近的样本权重越大（距离加权）
imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = imputer.fit_transform(X)
```

**迭代插补（MICE）**：Multiple Imputation by Chained Equations，将每个有缺失的特征作为其他特征的函数来预测。通过多次迭代，逐步优化插补值，是目前最先进的插补方法之一。

```python
from sklearn.experimental import enable_iterative_imputer  # 启用实验性功能
from sklearn.impute import IterativeImputer

# max_iter=10: 最大迭代次数，每次迭代会改进插补值
# random_state=42: 设置随机种子确保结果可复现
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
```

### 4. 缺失值分析与可视化

在处理缺失值之前，应该先分析缺失的模式和比例。这有助于选择合适的处理策略，以及发现数据收集过程中的潜在问题。

```python
import matplotlib.pyplot as plt

def analyze_missing_values(df):
    """
    分析数据集中的缺失值情况
    
    参数:
        df: pandas DataFrame，待分析的数据集
    
    返回:
        missing_stats: 包含缺失统计信息的DataFrame
    """
    
    # 统计每个特征的缺失数量和缺失比例
    missing_stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),  # 每列缺失值个数
        'missing_ratio': df.isnull().sum() / len(df) * 100  # 缺失比例(%)
    }).sort_values('missing_ratio', ascending=False)  # 按缺失比例降序排列
    
    # 创建1行2列的子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：缺失比例柱状图
    missing_stats['missing_ratio'].plot(kind='bar', ax=axes[0])
    axes[0].set_title('缺失值比例')
    axes[0].set_ylabel('比例 (%)')
    
    # 右图：缺失模式热图
    # df.isnull().T: 转置后每行代表一个特征，白色表示缺失，黑色表示有值
    axes[1].imshow(df.isnull().T, cmap='binary', aspect='auto')
    axes[1].set_title('缺失模式热图')
    axes[1].set_xlabel('样本')
    axes[1].set_ylabel('特征')
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()
    
    return missing_stats
```

---

## 二、异常值检测

异常值是指与大部分数据显著不同的观测值，可能由测量错误、数据录入错误或真实的极端事件造成。正确识别和处理异常值对模型性能至关重要。

### 1. 统计方法

统计方法基于数据的分布特性来识别异常值，计算简单、可解释性强。

**3σ 原则**（正态分布假设）：假设数据服从正态分布，超出3倍标准差的数据认为是异常值。根据正态分布特性，99.7%的数据落在均值±3σ范围内。

```python
def detect_outliers_sigma(data, threshold=3):
    """
    基于3σ原则的异常值检测
    
    参数:
        data: 一维数组，待检测的数据
        threshold: 标准差倍数阈值，默认为3（覆盖99.7%的正常数据）
    
    返回:
        布尔数组，True表示该位置是异常值
    """
    mean = np.mean(data)  # 计算均值
    std = np.std(data)    # 计算标准差
    lower = mean - threshold * std  # 下界：均值-3σ
    upper = mean + threshold * std  # 上界：均值+3σ
    return (data < lower) | (data > upper)  # 返回超出范围的值
```

**IQR 方法**（箱线图）：基于四分位距（Interquartile Range），不依赖分布假设。这是箱线图中识别异常值的标准方法，对偏态分布更加鲁棒。

```python
def detect_outliers_iqr(data, multiplier=1.5):
    """
    基于IQR（四分位距）的异常值检测
    
    参数:
        data: 一维数组，待检测的数据
        multiplier: IQR倍数，默认1.5（标准箱线图设置）
    
    返回:
        布尔数组，True表示该位置是异常值
    """
    Q1 = np.percentile(data, 25)  # 第一四分位数（25%分位点）
    Q3 = np.percentile(data, 75)  # 第三四分位数（75%分位点）
    IQR = Q3 - Q1  # 四分位距
    lower = Q1 - multiplier * IQR  # 下界
    upper = Q3 + multiplier * IQR  # 上界
    return (data < lower) | (data > upper)
```

**方法对比**：

| 方法 | 假设 | 优点 | 缺点 |
|------|------|------|------|
| 3σ | 正态分布 | 简单直观 | 对非正态数据效果差 |
| IQR | 无分布假设 | 鲁棒性好 | 对极端异常敏感 |

### 2. 机器学习方法

机器学习方法不依赖分布假设，能够处理高维数据和复杂的异常模式。

**孤立森林**：通过随机划分数据来隔离异常点，异常点更容易被孤立。异常数据通常数量少且特征值独特，因此只需要较少的划分就能被隔离，路径长度更短。

```python
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(X, contamination=0.1):
    """
    孤立森林异常值检测
    
    参数:
        X: 特征矩阵
        contamination: 异常值比例的估计值，默认0.1表示约10%为异常
    
    返回:
        outliers: 布尔数组，True表示异常值
        scores: 异常得分（越负越异常）
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso.fit_predict(X) == -1  # -1表示异常，1表示正常
    scores = iso.decision_function(X)     # 异常得分
    return outliers, scores
```

**局部异常因子（LOF）**：比较样本的局部密度与邻居的局部密度，密度显著低于邻居的被判定为异常。LOF适合检测局部异常（在某个区域内异常，但在全局看可能不明显）。

```python
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(X, n_neighbors=20, contamination=0.1):
    """
    LOF（局部异常因子）异常值检测
    
    参数:
        X: 特征矩阵
        n_neighbors: 计算局部密度时考虑的邻居数量
        contamination: 异常值比例的估计值
    
    返回:
        outliers: 布尔数组，True表示异常值
        scores: LOF得分（越大越异常）
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(X) == -1  # -1表示异常
    scores = -lof.negative_outlier_factor_  # 转换为正值，越大越异常
    return outliers, scores
```

### 3. 异常值处理策略

检测到异常值后，需要根据业务场景选择合适的处理方式。不是所有异常值都是错误，有些可能是真实的有价值信息。

```python
# 1. 删除异常值：直接剔除被标记为异常的样本
# 注意：删除后数据量减少，可能影响模型训练
df_clean = df[~outliers]

# 2. 缩尾处理（Winsorization）：将极端值替换为指定百分位的值
# limits=[0.05, 0.05] 表示将最小的5%和最大的5%的数据进行缩尾
from scipy.stats.mstats import winsorize
data_winsorized = winsorize(data, limits=[0.05, 0.05])

# 3. 对数变换：压缩大值、放大小值，降低极端值的影响
# np.log1p(x) = log(1+x)，避免log(0)的问题
data_log = np.log1p(data)
```

---

## 三、标准化与归一化

标准化和归一化是将特征缩放到相同量级的过程。这对于基于距离的算法（如KNN、SVM）和梯度下降优化（如神经网络）尤为重要，可以加速收敛并提高模型性能。

### 方法对比

不同方法适用于不同场景，需要根据数据特性和模型需求选择：

| 方法 | 公式 | 结果范围 | 适用场景 | 异常值敏感度 |
|------|------|----------|----------|--------------|
| Z-score | $\frac{x-\mu}{\sigma}$ | 均值0，方差1 | 正态分布、PCA、SVM | 高 |
| Min-Max | $\frac{x-x_{min}}{x_{max}-x_{min}}$ | [0, 1] | 神经网络、图像 | 高 |
| Robust | $\frac{x-median}{IQR}$ | 无固定范围 | 有异常值 | 低 |

### 代码实现

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Z-score标准化：将数据转换为均值为0、标准差为1的分布
# 适用于：PCA、SVM、线性回归等假设数据正态分布的模型
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# Min-Max归一化：将数据线性缩放到[0,1]区间
# 适用于：神经网络输入、图像像素值等需要固定范围的数据
scaler_mm = MinMaxScaler(feature_range=(0, 1))
X_mm = scaler_mm.fit_transform(X)

# 稳健标准化：使用中位数和IQR，对异常值不敏感
# 适用于：数据中存在异常值的情况
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
```

### 可视化对比

通过可视化可以直观理解不同标准化方法对数据分布的影响：

```python
def compare_scaling_methods(X):
    """
    比较不同标准化方法的效果
    
    参数:
        X: 一维特征数组
    """
    
    methods = {
        'Original': X,  # 原始数据
        'Z-score': StandardScaler().fit_transform(X.reshape(-1, 1)).flatten(),
        'Min-Max': MinMaxScaler().fit_transform(X.reshape(-1, 1)).flatten(),
        'Robust': RobustScaler().fit_transform(X.reshape(-1, 1)).flatten()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(methods.items()):
        # 绘制直方图
        axes[idx].hist(data, bins=30, alpha=0.7, edgecolor='black')
        # 标题显示方法名称、均值和标准差
        axes[idx].set_title(f'{name}\nMean: {data.mean():.2f}, Std: {data.std():.2f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
```

---

## 四、类别特征编码

机器学习模型通常只能处理数值输入，因此需要将类别特征转换为数值。不同的编码方法适用于不同类型的类别特征和模型。

### 编码方法对比

选择编码方法时需要考虑：类别是否有顺序、类别数量多少、模型类型等因素：

| 方法 | 原理 | 维度变化 | 适用场景 |
|------|------|----------|----------|
| 标签编码 | 类别→整数 | 不变 | 有序类别 |
| 独热编码 | 类别→二进制向量 | 增加维度 | 无序类别、类别少 |
| 目标编码 | 类别→目标均值 | 不变 | 高基数类别 |
| 频率编码 | 类别→出现频率 | 不变 | 类别有重要性 |

### 代码实现

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# 标签编码：将类别转换为0,1,2,...的整数
# 适用于：有序类别（如小、中、大）或目标变量
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 独热编码：将每个类别扩展为二进制列（0或1）
# sparse_output=False 返回密集数组而非稀疏矩阵
# handle_unknown='ignore' 对未知类别返回全0向量
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_onehot = ohe.fit_transform(X_categorical)

# 序数编码：按照指定顺序将类别转换为整数
# categories参数指定类别的顺序（从小到大对应的数值）
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = oe.fit_transform(X_categorical)

# 目标编码：用该类别的目标变量均值来编码
# 适用于高基数类别特征（如城市名、用户ID等）
def target_encode(series, target, smoothing=1):
    """
    目标编码实现
    
    参数:
        series: 类别特征Series
        target: 目标变量Series
        smoothing: 平滑参数，值越大越倾向于全局均值
    
    返回:
        编码后的Series
    """
    # 计算每个类别的目标均值
    means = target.groupby(series).mean()
    # 统计每个类别的样本数
    counts = series.value_counts()
    # 全局目标均值
    global_mean = target.mean()
    
    # 平滑处理：样本数少的类别更倾向于全局均值
    # 避免小样本类别过拟合
    smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return series.map(smoothed)
```

---

## 五、预处理管道

在实际项目中，预处理通常包含多个步骤。使用Pipeline可以将这些步骤串联起来，确保训练集和测试集使用相同的预处理逻辑，避免数据泄露。

### 完整示例

以下示例展示了如何为数值特征和类别特征分别构建处理管道，并组合成一个统一的预处理器：

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessing_pipeline(num_features, cat_features):
    """
    创建预处理管道
    
    参数:
        num_features: 数值特征列名列表
        cat_features: 类别特征列名列表
    
    返回:
        ColumnTransformer对象
    """
    
    # 数值特征处理管道
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 中位数填充缺失值
        ('scaler', RobustScaler())  # 稳健标准化
    ])
    
    # 类别特征处理管道
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填充缺失值
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 独热编码
    ])
    
    # 使用ColumnTransformer对不同类型的列应用不同的变换
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),  # 数值特征的处理
            ('cat', cat_transformer, cat_features)   # 类别特征的处理
        ]
    )
    
    return preprocessor

# 使用示例
preprocessor = create_preprocessing_pipeline(
    num_features=['age', 'income'],  # 数值特征
    cat_features=['city', 'gender']  # 类别特征
)
X_processed = preprocessor.fit_transform(X_train)
```

---

## 六、最佳实践

### 防止数据泄露

数据泄露是预处理中最常见的错误，会导致模型在验证集上表现虚高，但在实际应用中效果很差。核心原则是：**所有预处理参数必须仅从训练集学习**。

```python
# ❌ 错误：在整个数据集上拟合标准化器
# 这会导致测试集的信息泄露到训练过程中，产生过于乐观的评估结果
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 包含测试集信息

# ✅ 正确：仅在训练集上拟合标准化器
# fit() 在训练集上学习参数（均值、标准差）
# transform() 用训练集学到的参数变换训练集和测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 仅训练集学习参数
X_test_scaled = scaler.transform(X_test)        # 应用相同变换到测试集
```

### 时间序列数据

时间序列数据有特殊的预处理要求，必须严格按时间顺序分割，避免"用未来预测过去"的问题。

```python
# 时间序列数据必须按时间顺序分割，不能随机打乱
# 否则会用未来数据预测过去，导致数据泄露
train_size = int(len(X) * 0.8)  # 前80%作为训练集
X_train, X_test = X[:train_size], X[train_size:]  # 按时间切分
y_train, y_test = y[:train_size], y[train_size:]

# 标准化参数仅从训练集计算，避免未来信息泄露
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 仅在训练集上学习参数
X_test_scaled = scaler.transform(X_test)        # 用训练集参数变换测试集
```

### 检查清单

- [ ] 分析缺失值模式和比例
- [ ] 选择合适的缺失值处理方法
- [ ] 检测并处理异常值
- [ ] 选择适当的标准化方法
- [ ] 对类别特征进行编码
- [ ] 构建可复用的预处理管道
- [ ] 确保无数据泄露

---

## 小结

| 环节 | 关键点 | 推荐方法 |
|------|--------|----------|
| 缺失值 | 分析缺失机制 | MAR用插补，MCAR可删除 |
| 异常值 | 结合业务理解 | IQR统计法、孤立森林 |
| 标准化 | 考虑异常值和分布 | 有异常值用Robust |
| 编码 | 考虑类别基数 | 低基数用独热，高基数用目标编码 |

数据预处理是特征工程的基础，合理的预处理策略能够显著提升模型性能和稳定性。
