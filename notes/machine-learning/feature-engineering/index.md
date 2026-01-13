# 特征工程

## 概述

数据预处理和特征构建技术，通过优化输入特征提升模型性能。特征工程是机器学习流程中至关重要的环节。

## 基础概念

### 特征类型
- **数值特征**：连续或离散的数值
- **类别特征**：有限个离散值
- **文本特征**：自然语言文本
- **时间特征**：时间序列数据
- **空间特征**：地理位置数据

### 特征工程目标
- 提高模型性能
- 减少计算复杂度
- 增强模型可解释性
- 处理数据质量问题

## Python实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import make_classification
from scipy import stats

def data_preprocessing_demo():
    """
    数据预处理演示
    """
    print("=== 数据预处理演示 ===")
    
    # 生成包含不同问题的数据
    np.random.seed(42)
    n_samples = 1000
    
    # 数值特征（包含异常值）
    feature1 = np.random.normal(50, 10, n_samples)
    feature1[0:10] = [200, 250, 180, 220, 190, 210, 230, 240, 260, 270]  # 添加异常值
    
    # 数值特征（不同尺度）
    feature2 = np.random.uniform(0, 1, n_samples)
    feature3 = np.random.exponential(2, n_samples)
    
    # 类别特征
    categories = ['A', 'B', 'C', 'D']
    feature4 = np.random.choice(categories, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4
    })
    
    # 目标变量
    y = (feature1 + feature2 * 100 + feature3 * 10 > 150).astype(int)
    
    # 数据探索
    print("原始数据统计:")
    print(df.describe())
    
    # 可视化原始数据分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df['feature1'], bins=30, alpha=0.7)
    axes[0, 0].set_title('Feature1 (含异常值)')
    
    axes[0, 1].hist(df['feature2'], bins=30, alpha=0.7)
    axes[0, 1].set_title('Feature2 (均匀分布)')
    
    axes[1, 0].hist(df['feature3'], bins=30, alpha=0.7)
    axes[1, 0].set_title('Feature3 (指数分布)')
    
    df['feature4'].value_counts().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Feature4 (类别分布)')
    
    plt.tight_layout()
    plt.show()
    
    return df, y

def feature_scaling_comparison(df):
    """
    特征缩放方法比较
    """
    print("\n=== 特征缩放方法比较 ===")
    
    # 选择数值特征
    numeric_features = ['feature1', 'feature2', 'feature3']
    X_numeric = df[numeric_features]
    
    # 不同的缩放方法
    scalers = {
        'StandardScaler': StandardScaler(),      # 标准化 (均值0，方差1)
        'MinMaxScaler': MinMaxScaler(),          # 最小最大缩放 (0-1范围)
        'RobustScaler': RobustScaler()           # 稳健缩放 (对异常值鲁棒)
    }
    
    # 应用缩放
    scaled_data = {}
    for name, scaler in scalers.items():
        scaled_data[name] = scaler.fit_transform(X_numeric)
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始数据
    for i, feature in enumerate(numeric_features):
        axes[0, 0].hist(X_numeric[feature], alpha=0.5, label=feature)
    axes[0, 0].set_title('原始数据')
    axes[0, 0].legend()
    
    # 缩放后数据
    for idx, (name, data) in enumerate(scaled_data.items()):
        row, col = (idx + 1) // 2, (idx + 1) % 2
        for i in range(data.shape[1]):
            axes[row, col].hist(data[:, i], alpha=0.5, label=numeric_features[i])
        axes[row, col].set_title(name)
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 统计比较
    print("\n缩放方法统计比较:")
    comparison_df = pd.DataFrame({
        'Method': ['Original'] + list(scalers.keys()),
        'Mean_feature1': [X_numeric['feature1'].mean()] + [data[:, 0].mean() for data in scaled_data.values()],
        'Std_feature1': [X_numeric['feature1'].std()] + [data[:, 0].std() for data in scaled_data.values()],
        'Min_feature1': [X_numeric['feature1'].min()] + [data[:, 0].min() for data in scaled_data.values()],
        'Max_feature1': [X_numeric['feature1'].max()] + [data[:, 0].max() for data in scaled_data.values()]
    })
    
    print(comparison_df.round(3))
    
    return scaled_data

def categorical_encoding_demo(df):
    """
    类别特征编码演示
    """
    print("\n=== 类别特征编码演示 ===")
    
    # 不同的编码方法
    categorical_feature = df['feature4']
    
    # 标签编码
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(categorical_feature)
    
    # 独热编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(categorical_feature.values.reshape(-1, 1))
    
    # 序数编码（自定义顺序）
    ordinal_encoder = OrdinalEncoder(categories=[['A', 'B', 'C', 'D']])
    ordinal_encoded = ordinal_encoder.fit_transform(categorical_feature.values.reshape(-1, 1))
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始类别分布
    categorical_feature.value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('原始类别分布')
    
    # 标签编码
    pd.Series(label_encoded).value_counts().sort_index().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('标签编码')
    
    # 独热编码（显示第一个特征的分布）
    pd.Series(onehot_encoded[:, 0]).value_counts().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('独热编码 - 特征A')
    
    # 序数编码
    pd.Series(ordinal_encoded.flatten()).value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('序数编码')
    
    plt.tight_layout()
    plt.show()
    
    # 编码信息
    print("\n编码方法比较:")
    print(f"原始类别数量: {len(categorical_feature.unique())}")
    print(f"标签编码维度: 1")
    print(f"独热编码维度: {onehot_encoded.shape[1]}")
    print(f"序数编码维度: 1")
    
    return {
        'label_encoded': label_encoded,
        'onehot_encoded': onehot_encoded,
        'ordinal_encoded': ordinal_encoded
    }

def feature_selection_methods(X, y):
    """
    特征选择方法演示
    """
    print("\n=== 特征选择方法演示 ===")
    
    # 生成更多特征的数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=42)
    
    # 特征选择方法
    
    # 1. 过滤法 - 基于统计检验
    selector_kbest = SelectKBest(score_func=f_classif, k=10)
    X_kbest = selector_kbest.fit_transform(X, y)
    kbest_scores = selector_kbest.scores_
    
    # 2. 包裹法 - 递归特征消除
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_rfe = RFE(estimator=estimator, n_features_to_select=10)
    X_rfe = selector_rfe.fit_transform(X, y)
    rfe_ranking = selector_rfe.ranking_
    
    # 3. 嵌入法 - 基于模型的特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    
    # 可视化特征选择结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # KBest分数
    axes[0].bar(range(len(kbest_scores)), kbest_scores)
    axes[0].set_title('SelectKBest 特征分数')
    axes[0].set_xlabel('特征索引')
    axes[0].set_ylabel('F-value')
    
    # RFE排名
    axes[1].bar(range(len(rfe_ranking)), rfe_ranking)
    axes[1].set_title('RFE 特征排名')
    axes[1].set_xlabel('特征索引')
    axes[1].set_ylabel('排名 (1=最好)')
    
    # 特征重要性
    axes[2].bar(range(len(feature_importances)), feature_importances)
    axes[2].set_title('随机森林特征重要性')
    axes[2].set_xlabel('特征索引')
    axes[2].set_ylabel('重要性')
    
    plt.tight_layout()
    plt.show()
    
    # 选择结果比较
    top_kbest = np.argsort(kbest_scores)[-10:][::-1]
    top_rfe = np.where(rfe_ranking == 1)[0]
    top_importance = np.argsort(feature_importances)[-10:][[::-1]]
    
    print("\n特征选择结果:")
    print(f"SelectKBest 选择的特征: {top_kbest}")
    print(f"RFE 选择的特征: {top_rfe}")
    print(f"特征重要性选择的特征: {top_importance}")
    print(f"共同选择的特征: {set(top_kbest) & set(top_rfe) & set(top_importance)}")
    
    return {
        'kbest_features': top_kbest,
        'rfe_features': top_rfe,
        'importance_features': top_importance
    }

def feature_engineering_advanced():
    """
    高级特征工程技术
    """
    print("\n=== 高级特征工程技术 ===")
    
    # 1. 多项式特征
    from sklearn.preprocessing import PolynomialFeatures
    
    X_simple = np.array([[1, 2], [3, 4], [5, 6]])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_simple)
    
    print("多项式特征生成:")
    print(f"原始特征: {X_simple}")
    print(f"多项式特征: {X_poly}")
    print(f"特征名称: {poly.get_feature_names_out(['x1', 'x2'])}")
    
    # 2. 分箱（离散化）
    from sklearn.preprocessing import KBinsDiscretizer
    
    X_continuous = np.random.normal(0, 1, 100).reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_binned = discretizer.fit_transform(X_continuous)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(X_continuous, bins=30, alpha=0.7)
    plt.title('原始连续特征')
    
    plt.subplot(122)
    plt.hist(X_binned, bins=5, alpha=0.7)
    plt.title('分箱后特征')
    plt.tight_layout()
    plt.show()
    
    # 3. 文本特征提取
    print("\n文本特征提取:")
    
    documents = [
        '机器学习是人工智能的重要分支',
        '深度学习是机器学习的一个子领域',
        '自然语言处理使用深度学习技术',
        '计算机视觉和自然语言处理都是AI的应用'
    ]
    
    # 词袋模型
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(documents)
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(documents)
    
    print("词袋模型特征:")
    print(count_vectorizer.get_feature_names_out())
    print("TF-IDF特征矩阵形状:", X_tfidf.shape)
    
    # 4. 时间特征工程
    print("\n时间特征工程:")
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    time_features = pd.DataFrame({'date': dates})
    
    # 提取时间特征
    time_features['year'] = time_features['date'].dt.year
    time_features['month'] = time_features['date'].dt.month
    time_features['day'] = time_features['date'].dt.day
    time_features['dayofweek'] = time_features['date'].dt.dayofweek
    time_features['is_weekend'] = time_features['dayofweek'].isin([5, 6]).astype(int)
    
    print("时间特征示例:")
    print(time_features.head())
    
    return {
        'polynomial_features': X_poly,
        'binned_features': X_binned,
        'text_features': X_tfidf,
        'time_features': time_features
    }

def feature_engineering_pipeline():
    """
    完整的特征工程流程
    """
    print("\n=== 完整特征工程流程 ===")
    
    # 生成综合数据集
    np.random.seed(42)
    n_samples = 1000
    
    # 数值特征
    numerical_data = np.column_stack([
        np.random.normal(50, 10, n_samples),      # 特征1
        np.random.uniform(0, 100, n_samples),     # 特征2
        np.random.exponential(2, n_samples)       # 特征3
    ])
    
    # 类别特征
    categorical_data = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    
    # 创建完整数据集
    df = pd.DataFrame(numerical_data, columns=['num1', 'num2', 'num3'])
    df['cat1'] = categorical_data
    
    # 目标变量
    y = ((df['num1'] > 55) & (df['num2'] > 50) & (df['cat1'].isin(['A', 'C']))).astype(int)
    
    # 特征工程管道
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    
    # 定义数值和类别特征的处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['num1', 'num2', 'num3']),
            ('cat', categorical_transformer, ['cat1'])
        ]
    )
    
    # 完整的建模管道
    from sklearn.linear_model import LogisticRegression
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # 评估管道性能
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(model_pipeline, df, y, cv=5, scoring='accuracy')
    
    print(f"特征工程管道性能: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # 对比无特征工程的基线模型
    baseline_model = LogisticRegression()
    
    # 简单编码类别特征
    df_baseline = df.copy()
    df_baseline['cat1'] = LabelEncoder().fit_transform(df_baseline['cat1'])
    
    baseline_scores = cross_val_score(baseline_model, df_baseline, y, cv=5, scoring='accuracy')
    
    print(f"基线模型性能: {baseline_scores.mean():.3f} ± {baseline_scores.std():.3f}")
    print(f"性能提升: {(scores.mean() - baseline_scores.mean()):.3f}")
    
    return model_pipeline, scores

# 示例使用
if __name__ == "__main__":
    # 运行特征工程演示
    print("=== 特征工程完整演示 ===")
    
    # 1. 数据预处理
    df, y = data_preprocessing_demo()
    
    # 2. 特征缩放
    scaled_data = feature_scaling_comparison(df)
    
    # 3. 类别编码
    encoded_data = categorical_encoding_demo(df)
    
    # 4. 特征选择
    X, y_selection = make_classification(n_samples=1000, n_features=20, 
                                        n_informative=10, random_state=42)
    selected_features = feature_selection_methods(X, y_selection)
    
    # 5. 高级特征工程
    advanced_features = feature_engineering_advanced()
    
    # 6. 完整管道
    pipeline, scores = feature_engineering_pipeline()
    
    print("\n=== 特征工程最佳实践总结 ===")
    print("1. 始终进行数据探索和理解")
    print("2. 根据特征类型选择合适的处理方法")
    print("3. 尝试多种特征选择方法")
    print("4. 考虑领域知识创建新特征")
    print("5. 使用交叉验证评估特征工程效果")
    print("6. 建立可复用的特征工程管道")
```

## 数学基础

### 特征缩放公式

**标准化**：$z = \frac{x - \mu}{\sigma}$

**最小最大缩放**：$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

**稳健缩放**：$x' = \frac{x - median}{IQR}$

### 特征选择指标

**相关系数**：$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$

**互信息**：$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$

**卡方统计量**：$\chi^2 = \sum \frac{(O - E)^2}{E}$

## 特征工程流程

### 1. 数据理解与探索
- 数据质量评估
- 特征分布分析
- 缺失值处理
- 异常值检测

### 2. 数据清洗
- 处理缺失值（删除、填充、插值）
- 处理异常值（删除、缩尾、转换）
- 数据类型转换

### 3. 特征变换
- 数值特征：缩放、归一化、对数变换
- 类别特征：编码、嵌入、目标编码
- 文本特征：词袋模型、TF-IDF、词嵌入
- 时间特征：周期性编码、时间窗口

### 4. 特征构建
- 多项式特征
- 交互特征
- 统计特征（均值、方差、分位数）
- 领域知识特征

### 5. 特征选择
- 过滤法：基于统计指标
- 包裹法：基于模型性能
- 嵌入法：基于模型重要性

### 6. 特征评估
- 模型性能评估
- 特征重要性分析
- 稳定性检验

## 应用场景

### 表格数据
- 金融风控：交易特征、用户行为特征
- 医疗诊断：生理指标、病史特征
- 电商推荐：用户画像、商品特征

### 文本数据
- 情感分析：词频、TF-IDF、词向量
- 文本分类：n-gram、主题模型
- 信息提取：命名实体识别、关系抽取

### 时间序列
- 股票预测：技术指标、市场情绪
- 设备监控：趋势特征、周期性特征
- 用户行为：序列模式、时间间隔

### 图像数据
- 特征提取：SIFT、HOG、CNN特征
- 数据增强：旋转、缩放、颜色变换
- 降维：PCA、自动编码器

## 最佳实践

### 数据质量优先
- 垃圾进，垃圾出
- 重视数据清洗
- 建立数据质量监控

### 迭代优化
- 从简单开始
- 逐步增加复杂度
- 持续评估效果

### 自动化管道
- 建立可复用的流程
- 版本控制特征
- 监控特征漂移

### 领域知识融合
- 理解业务背景
- 利用专家知识
- 创建有意义的特征

## 常见陷阱

### 数据泄露
- 使用未来信息
- 测试集信息污染训练集
- 时间序列数据的时序泄露

### 过拟合特征
- 创建过于复杂的特征
- 基于目标变量创建特征
- 忽略特征重要性评估

### 维度灾难
- 特征过多导致稀疏性
- 计算复杂度急剧增加
- 模型泛化能力下降

### 特征漂移
- 数据分布随时间变化
- 特征含义发生变化
- 需要定期更新特征工程

## 前沿技术

### 自动特征工程
- 基于遗传算法
- 基于强化学习
- 基于神经网络

### 可解释性特征
- SHAP值分析
- LIME局部解释
- 特征重要性可视化

### 联邦特征学习
- 隐私保护的特征工程
- 分布式特征提取
- 跨域特征迁移

特征工程是机器学习成功的关键因素，"数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限"。