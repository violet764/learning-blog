# 特征工程

<div align="center">
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Feature Engineering" width="200">
</div>

> "数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。"  
> —— 吴恩达

特征工程是机器学习流程中最关键的环节之一，通过数据预处理、特征变换和特征选择等技术，将原始数据转化为更适合模型学习的特征，从而显著提升模型性能。

## 📚 内容概览

本章节包含以下内容：

| 文件 | 主题 | 核心内容 |
|------|------|----------|
| [数据预处理](./preprocessing.md) | 数据清洗与准备 | 缺失值处理、异常值检测、标准化、编码 |
| [特征变换](./feature-transformation.md) | 特征转换与构建 | 数值变换、多项式特征、核方法、离散化 |
| [特征选择](./feature-selection.md) | 特征筛选与降维 | 过滤法、包裹法、嵌入法、稳定性选择 |

---

## 🔗 核心概念

### 特征类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **数值特征** | 连续或离散的数值 | 年龄、温度、收入 |
| **类别特征** | 有限个离散值 | 性别、城市、职业 |
| **文本特征** | 自然语言文本 | 评论、标题、描述 |
| **时间特征** | 时间序列数据 | 日期、时间戳、周期 |
| **空间特征** | 地理位置数据 | 经纬度、地址、区域 |

### 特征工程流程

```
原始数据 → 数据预处理 → 特征变换 → 特征选择 → 模型训练
    ↓           ↓           ↓           ↓
 数据清洗    缺失/异常    构建新特征   筛选最优子集
 类型转换    标准化编码   多项式/核    降低维度
```

---

## 📖 章节详解

### 1. 数据预处理

数据预处理是特征工程的基础，主要解决数据质量问题：

- **缺失值处理**：删除法、插补法（均值/中位数/众数、KNN、MICE）
- **异常值检测**：统计方法（3σ、IQR）、机器学习方法（孤立森林、LOF）
- **标准化归一化**：Z-score、Min-Max、Robust Scaling
- **类别编码**：独热编码、标签编码、目标编码

👉 [详细内容](./preprocessing.md)

### 2. 特征变换

特征变换通过数学函数改善数据分布、增强表达能力：

- **数值变换**：对数变换、Box-Cox、Yeo-Johnson
- **多项式特征**：特征交互、非线性扩展
- **核方法**：Kernel PCA、核技巧
- **离散化**：等宽分箱、等频分箱、聚类分箱

👉 [详细内容](./feature-transformation.md)

### 3. 特征选择

特征选择从原始特征中选择最相关的子集：

- **过滤法**：方差阈值、相关系数、卡方检验、互信息
- **包裹法**：前向选择、后向消除、递归特征消除（RFE）
- **嵌入法**：L1正则化（LASSO）、树模型重要性

👉 [详细内容](./feature-selection.md)

---

## 🛠️ 快速入门示例

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 构建完整的特征工程管道
def create_feature_engineering_pipeline(numerical_features, categorical_features):
    """创建特征工程管道"""
    
    # 数值特征处理
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 完整管道（包含特征选择）
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif, k=10))
    ])
    
    return pipeline

# 使用示例
# pipeline = create_feature_engineering_pipeline(['age', 'income'], ['city', 'gender'])
# X_transformed = pipeline.fit_transform(X, y)
```

---

## 📊 方法对比速查表

### 缺失值处理方法对比

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 删除法 | MCAR、缺失少 | 简单、无偏 | 损失信息 |
| 均值插补 | 数值特征 | 保持均值 | 降低方差 |
| KNN插补 | 有相关性特征 | 利用相似样本 | 计算量大 |
| MICE | 多变量缺失 | 保持相关性 | 复杂 |

### 标准化方法对比

| 方法 | 公式 | 适用场景 | 对异常值 |
|------|------|----------|----------|
| Z-score | $(x-\mu)/\sigma$ | 正态分布 | 敏感 |
| Min-Max | $(x-x_{min})/(x_{max}-x_{min})$ | 有界数据 | 敏感 |
| Robust | $(x-median)/IQR$ | 有异常值 | 鲁棒 |

### 特征选择方法对比

| 方法 | 计算速度 | 过拟合风险 | 适用场景 |
|------|----------|------------|----------|
| 过滤法 | 快 | 低 | 初步筛选、高维数据 |
| 包裹法 | 慢 | 高 | 精确选择、小数据集 |
| 嵌入法 | 中 | 中 | 常规场景、平衡选择 |

---

## ⚠️ 常见陷阱

### 数据泄露
- **问题**：使用测试集信息进行预处理
- **解决**：预处理参数仅在训练集上计算

### 过拟合
- **问题**：特征过多或过度工程化
- **解决**：使用交叉验证评估，正则化约束

### 维度灾难
- **问题**：特征数量远大于样本量
- **解决**：特征选择、降维技术

---

## 🔗 相关资源

- [Scikit-learn 预处理文档](https://scikit-learn.org/stable/modules/preprocessing.html)
- [特征工程实战（英文）](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle 特征工程教程](https://www.kaggle.com/learn/feature-engineering)

---

***持续更新中*** ✨
