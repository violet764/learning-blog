# 数据预处理

## 概述

数据预处理是机器学习流程中的关键环节，涉及数据清洗、缺失值处理、异常值检测、标准化和编码等技术，旨在提高数据质量、增强模型性能和稳定性。

## 数学基础与理论框架

### 数据质量度量

**完整性度量**：
$$Completeness = \frac{N_{valid}}{N_{total}} \times 100\%$$

**一致性度量**：基于数据约束和规则的违反程度评估

**准确性度量**：通过与真实值的比较评估数据准确性

### 概率分布理论

**正态分布假设**：许多预处理方法基于数据服从正态分布的假设

**中心极限定理**：大样本下，样本均值近似正态分布

## 缺失值处理

### 缺失机制分析

#### 1. 完全随机缺失（MCAR）
缺失与观测值和未观测值均无关：$P(M|Y) = P(M)$

#### 2. 随机缺失（MAR）
缺失仅与观测值有关：$P(M|Y) = P(M|Y_{obs})$

#### 3. 非随机缺失（MNAR）
缺失与未观测值有关：$P(M|Y) = P(M|Y_{miss})$

### 缺失值处理方法

#### 1. 删除法
**数学原理**：基于完整案例分析

**适用条件**：
- MCAR缺失机制
- 缺失比例较低（通常<5%）
- 样本量足够大

**Python实现**：
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
import matplotlib.pyplot as plt

def missing_value_analysis(df):
    """缺失值分析"""
    
    # 缺失值统计
    missing_stats = df.isnull().sum()
    missing_percentage = (missing_stats / len(df)) * 100
    
    # 可视化缺失模式
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    missing_stats.plot(kind='bar')
    plt.title('缺失值数量')
    plt.xticks(rotation=45)
    
    plt.subplot(122)
    missing_percentage.plot(kind='bar')
    plt.title('缺失值比例 (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 缺失模式分析
    missing_pattern = df.isnull().astype(int)
    plt.figure(figsize=(10, 8))
    plt.imshow(missing_pattern.T, cmap='binary', aspect='auto')
    plt.xlabel('样本索引')
    plt.ylabel('特征')
    plt.title('缺失值模式热图')
    plt.colorbar(label='缺失(1)/存在(0)')
    plt.show()
    
    return missing_stats, missing_percentage
```

#### 2. 单变量插补法

**均值/中位数/众数插补**：
```python
def univariate_imputation(df, strategy='mean'):
    """单变量插补"""
    
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), 
                             columns=df.columns, index=df.index)
    
    # 插补效果评估
    original_stats = df.describe()
    imputed_stats = df_imputed.describe()
    
    print(f"{strategy}插补前后统计对比:")
    comparison = pd.concat([original_stats, imputed_stats], 
                          keys=['Original', 'Imputed'], axis=1)
    print(comparison.round(3))
    
    return df_imputed
```

**数学性质**：
- 均值插补：保持样本均值不变
- 中位数插补：对异常值更鲁棒
- 众数插补：适用于类别特征

#### 3. 多变量插补法

**迭代插补（MICE算法）**：
```python
def multivariate_imputation(df, max_iter=10):
    """多变量迭代插补"""
    
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df),
                             columns=df.columns, index=df.index)
    
    # 收敛性分析
    print(f"迭代插补收敛信息:")
    print(f"迭代次数: {imputer.n_iter_}")
    
    return df_imputed
```

**K近邻插补**：
```python
def knn_imputation(df, n_neighbors=5):
    """K近邻插补"""
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df),
                             columns=df.columns, index=df.index)
    
    return df_imputed
```

#### 4. 高级插补方法

**基于模型的插补**：
```python
def model_based_imputation(df):
    """基于模型的插补"""
    
    from sklearn.ensemble import RandomForestRegressor
    
    df_imputed = df.copy()
    
    for col in df.columns[df.isnull().any()]:
        # 分离有缺失和无缺失的样本
        missing_mask = df[col].isnull()
        complete_data = df[~missing_mask]
        missing_data = df[missing_mask]
        
        if len(complete_data) > 0:
            # 训练预测模型
            X_train = complete_data.drop(columns=[col])
            y_train = complete_data[col]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 预测缺失值
            X_pred = missing_data.drop(columns=[col])
            predictions = model.predict(X_pred)
            
            # 填充缺失值
            df_imputed.loc[missing_mask, col] = predictions
    
    return df_imputed
```

### 缺失值处理评估

```python
def evaluate_imputation_methods(original_df, methods):
    """评估不同插补方法"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # 人工引入缺失值（模拟）
    df_with_missing = original_df.copy()
    missing_mask = np.random.random(original_df.shape) < 0.1  # 10%缺失
    df_with_missing[missing_mask] = np.nan
    
    evaluation_results = []
    
    for method_name, imputation_func in methods.items():
        df_imputed = imputation_func(df_with_missing)
        
        # 计算插补误差
        mse = mean_squared_error(original_df.values[missing_mask], 
                               df_imputed.values[missing_mask])
        mae = mean_absolute_error(original_df.values[missing_mask],
                                df_imputed.values[missing_mask])
        
        evaluation_results.append({
            'Method': method_name,
            'MSE': mse,
            'MAE': mae,
            'Correlation': np.corrcoef(original_df.values[missing_mask].flatten(),
                                     df_imputed.values[missing_mask].flatten())[0, 1]
        })
    
    return pd.DataFrame(evaluation_results)
```

## 异常值检测

### 统计方法

#### 1. 3σ原则（正态分布假设）
**数学原理**：对于正态分布，99.7%的数据落在μ±3σ范围内

**实现**：
```python
def three_sigma_rule(data, threshold=3):
    """3σ原则异常值检测"""
    
    mean = np.mean(data)
    std = np.std(data)
    
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers, lower_bound, upper_bound
```

#### 2. 箱线图方法（IQR）
**数学公式**：
- Q1 = 25th百分位数，Q3 = 75th百分位数
- IQR = Q3 - Q1
- 异常值边界：Q1 - 1.5×IQR 和 Q3 + 1.5×IQR

**实现**：
```python
def iqr_method(data, multiplier=1.5):
    """IQR方法异常值检测"""
    
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers, lower_bound, upper_bound
```

#### 3. 马氏距离
**数学公式**：
$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

**实现**：
```python
def mahalanobis_distance(X, threshold=3):
    """马氏距离异常值检测"""
    
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.pinv(cov)
    
    distances = []
    for x in X:
        diff = x - mean
        distance = np.sqrt(diff @ inv_cov @ diff.T)
        distances.append(distance)
    
    distances = np.array(distances)
    outlier_threshold = np.mean(distances) + threshold * np.std(distances)
    outliers = distances > outlier_threshold
    
    return outliers, distances
```

### 机器学习方法

#### 1. 孤立森林（Isolation Forest）
```python
def isolation_forest_outliers(X, contamination=0.1):
    """孤立森林异常值检测"""
    
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(X) == -1
    
    # 异常分数
    anomaly_scores = iso_forest.decision_function(X)
    
    return outliers, anomaly_scores
```

#### 2. 局部异常因子（LOF）
```python
def local_outlier_factor(X, n_neighbors=20, contamination=0.1):
    """局部异常因子检测"""
    
    from sklearn.neighbors import LocalOutlierFactor
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(X) == -1
    
    # 异常分数（负的LOF值）
    anomaly_scores = -lof.negative_outlier_factor_
    
    return outliers, anomaly_scores
```

### 异常值处理策略

#### 1. 删除法
适用于异常值数量较少且对分析影响较大的情况

#### 2. 缩尾法（Winsorization）
**数学原理**：将极端值替换为指定分位数的值

**实现**：
```python
def winsorize_data(data, limits=(0.05, 0.05)):
    """数据缩尾处理"""
    
    from scipy.stats.mstats import winsorize
    
    data_winsorized = winsorize(data, limits=limits)
    
    return data_winsorized
```

#### 3. 变换法
通过对数变换、Box-Cox变换等降低异常值影响

## 数据标准化与归一化

### 标准化方法

#### 1. Z-score标准化
**数学公式**：$z = \frac{x - \mu}{\sigma}$

**性质**：均值=0，标准差=1

**实现**：
```python
def z_score_normalization(X):
    """Z-score标准化"""
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
```

#### 2. Min-Max归一化
**数学公式**：$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

**性质**：值域[0, 1]

**实现**：
```python
def min_max_normalization(X, feature_range=(0, 1)):
    """Min-Max归一化"""
    
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
```

#### 3. 稳健标准化
**数学公式**：$x' = \frac{x - median}{IQR}$

**优势**：对异常值鲁棒

**实现**：
```python
def robust_scaling(X):
    """稳健标准化"""
    
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
```

### 标准化方法比较

```python
def compare_scaling_methods(X):
    """比较不同标准化方法"""
    
    methods = {
        'Z-score': StandardScaler(),
        'Min-Max': MinMaxScaler(),
        'Robust': RobustScaler()
    }
    
    comparison_results = []
    
    for name, scaler in methods.items():
        X_scaled = scaler.fit_transform(X)
        
        stats = {
            'Method': name,
            'Mean': np.mean(X_scaled, axis=0).mean(),
            'Std': np.std(X_scaled, axis=0).mean(),
            'Min': np.min(X_scaled),
            'Max': np.max(X_scaled),
            'Range': np.ptp(X_scaled)
        }
        
        comparison_results.append(stats)
    
    return pd.DataFrame(comparison_results)
```

## 分类变量编码

### 编码方法

#### 1. 独热编码（One-Hot Encoding）
**数学原理**：将类别特征转换为二进制向量

**实现**：
```python
def one_hot_encoding(categorical_data, drop_first=False):
    """独热编码"""
    
    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False)
    encoded_data = encoder.fit_transform(categorical_data.reshape(-1, 1))
    
    return encoded_data, encoder
```

#### 2. 标签编码（Label Encoding）
**数学原理**：将类别映射为整数

**实现**：
```python
def label_encoding(categorical_data):
    """标签编码"""
    
    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    encoded_data = encoder.fit_transform(categorical_data)
    
    return encoded_data, encoder
```

#### 3. 目标编码（Target Encoding）
**数学原理**：使用目标变量的统计量进行编码

**实现**：
```python
def target_encoding(categorical_data, target, smoothing=10):
    """目标编码"""
    
    # 计算每个类别的目标均值
    target_mean = target.groupby(categorical_data).mean()
    
    # 全局均值
    global_mean = target.mean()
    
    # 类别频率
    category_size = categorical_data.value_counts()
    
    # 平滑处理
    smoothing_factor = 1 / (1 + np.exp(-(category_size - smoothing) / smoothing))
    
    # 计算编码值
    encoded_values = smoothing_factor * target_mean + (1 - smoothing_factor) * global_mean
    
    # 应用编码
    encoded_data = categorical_data.map(encoded_values)
    
    return encoded_data, encoded_values
```

#### 4. 频率编码（Frequency Encoding）
```python
def frequency_encoding(categorical_data):
    """频率编码"""
    
    frequency_map = categorical_data.value_counts() / len(categorical_data)
    encoded_data = categorical_data.map(frequency_map)
    
    return encoded_data, frequency_map
```

### 编码方法评估

```python
def evaluate_encoding_methods(X_categorical, y, encoding_methods):
    """评估不同编码方法"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    results = []
    
    for method_name, encoding_func in encoding_methods.items():
        X_encoded, _ = encoding_func(X_categorical, y)
        
        # 确保数据格式正确
        if len(X_encoded.shape) == 1:
            X_encoded = X_encoded.reshape(-1, 1)
        
        # 模型性能评估
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
        
        results.append({
            'Method': method_name,
            'Mean_Accuracy': np.mean(scores),
            'Std_Accuracy': np.std(scores),
            'Feature_Dimension': X_encoded.shape[1]
        })
    
    return pd.DataFrame(results)
```

## 高级预处理技术

### 数据泄露预防

#### 时间序列数据预处理
```python
def time_series_preprocessing(time_series, train_ratio=0.8):
    """时间序列数据预处理"""
    
    # 时序分割（防止数据泄露）
    train_size = int(len(time_series) * train_ratio)
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    # 仅在训练集上计算统计量
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    
    # 应用相同的变换到测试集
    train_scaled = (train_data - train_mean) / train_std
    test_scaled = (test_data - train_mean) / train_std
    
    return train_scaled, test_scaled
```

### 自动化预处理管道

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """创建自动化预处理管道"""
    
    # 数值特征处理
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # 类别特征处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor
```

## 数学理论与深度分析

### 预处理方法的统计性质

#### 标准化方法的数学性质
- **线性变换不变性**：标准化不改变变量间的线性关系
- **分布形状保持**：标准化不改变数据的分布形状
- **尺度统一**：使不同尺度的特征具有可比性

#### 缺失值处理的统计影响
- **偏差引入**：不当的缺失值处理可能引入系统性偏差
- **方差变化**：插补方法可能影响估计的方差
- **相关性扭曲**：可能扭曲变量间的相关性结构

### 最优预处理策略

#### 基于信息理论的方法
**目标**：最大化预处理后数据的信息量

**数学框架**：
$$\max_{\phi} I(\phi(X); Y) - \lambda R(\phi)$$
其中R(φ)是正则化项，控制预处理复杂度

#### 基于模型性能的方法
**目标**：通过交叉验证选择最优预处理组合

**实现**：网格搜索或贝叶斯优化

## 实践案例与最佳实践

### 金融数据预处理

```python
def financial_data_preprocessing(df):
    """金融数据预处理管道"""
    
    # 1. 缺失值处理
    df_cleaned = df.dropna(subset=['price', 'volume'])  # 关键特征不能缺失
    
    # 2. 异常值处理（基于行业知识）
    price_outliers = three_sigma_rule(df_cleaned['price'], threshold=4)
    df_cleaned = df_cleaned[~price_outliers[0]]
    
    # 3. 特征工程
    df_cleaned['log_return'] = np.log(df_cleaned['price'] / df_cleaned['price'].shift(1))
    df_cleaned['volume_change'] = df_cleaned['volume'].pct_change()
    
    # 4. 标准化
    scaler = RobustScaler()
    numerical_features = ['log_return', 'volume_change']
    df_cleaned[numerical_features] = scaler.fit_transform(df_cleaned[numerical_features])
    
    return df_cleaned
```

### 文本数据预处理

```python
def text_data_preprocessing(text_corpus):
    """文本数据预处理"""
    
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 1. 文本清洗
    def clean_text(text):
        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned_corpus = [clean_text(text) for text in text_corpus]
    
    # 2. 特征提取
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(cleaned_corpus)
    
    return X_tfidf, vectorizer
```

### 最佳实践指南

1. **数据探索先行**：充分理解数据特性后再选择预处理方法
2. **防止数据泄露**：确保预处理不引入未来信息
3. **方法组合使用**：根据数据特性组合多种预处理技术
4. **领域知识融合**：结合业务理解选择适当的预处理策略
5. **自动化与监控**：建立可复用的预处理管道和监控机制

## 性能评估与比较

### 预处理方法评估框架

```python
def benchmark_preprocessing_methods(datasets, preprocessing_methods):
    """预处理方法基准测试"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    results = []
    
    for dataset_name, (X, y) in datasets.items():
        for method_name, preprocessor in preprocessing_methods.items():
            try:
                # 应用预处理
                if hasattr(preprocessor, 'fit_transform'):
                    X_processed = preprocessor.fit_transform(X, y)
                else:
                    X_processed = preprocessor.fit_transform(X)
                
                # 性能评估
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(model, X_processed, y, cv=5, scoring='accuracy')
                
                results.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Mean_Accuracy': np.mean(scores),
                    'Std_Accuracy': np.std(scores),
                    'Feature_Dimension': X_processed.shape[1]
                })
            except Exception as e:
                print(f"Error in {method_name} on {dataset_name}: {e}")
    
    return pd.DataFrame(results)
```

## 总结与展望

数据预处理是机器学习成功的关键因素，合理的预处理策略能够：
- 显著提升模型性能
- 增强模型稳定性
- 改善模型可解释性
- 加速模型训练过程

未来的发展方向包括：
- 自动化预处理流水线
- 自适应预处理方法
- 深度学习驱动的预处理
- 可解释性预处理技术

选择合适的预处理方法需要综合考虑数据特性、模型需求和计算资源，建立系统化的预处理流程是确保机器学习项目成功的重要保障。