# 无监督学习

无监督学习是机器学习的一个重要分支，其核心特点是**训练数据没有标签**。算法的目标是从无标签数据中发现隐藏的结构、模式或知识。

## 🎯 核心思想

给定数据集 $D = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$，其中 $\mathbf{x}_i \in \mathcal{X}$ 是输入特征，**没有对应的标签信息**。无监督学习的目标是：

- 📊 **发现数据的内在结构** - 聚类分析
- 📉 **简化数据表示** - 降维技术
- 🔍 **识别异常模式** - 异常检测
- 🔗 **发现关联关系** - 关联规则挖掘

## 📚 知识结构

### 核心模块

| 模块 | 内容 | 难度 | 文档 |
|------|------|------|------|
| 聚类分析 | K-means、层次聚类、DBSCAN、GMM | ⭐⭐ | [聚类分析](./clustering.md) |
| 降维技术 | PCA、LDA、t-SNE、流形学习 | ⭐⭐⭐ | [降维技术](./dimensionality-reduction.md) |
| 关联与异常 | Apriori、孤立森林、LOF | ⭐⭐ | [关联规则与异常检测](./association-anomaly.md) |
| 高级结构 | KD树、空间索引 | ⭐⭐⭐ | [高级数据结构](./advanced-structures.md) |

### 学习路径

```
聚类分析 ──────→ 降维技术 ──────→ 关联规则 ──────→ 异常检测
    │                │                │               │
    ▼                ▼                ▼               ▼
 K-means          PCA             Apriori        孤立森林
 层次聚类          LDA             FP-Growth       LOF
 DBSCAN          t-SNE                           One-Class SVM
 GMM             流形学习
```

## 🔄 与监督学习的对比

| 特性 | 监督学习 | 无监督学习 |
|------|----------|------------|
| 数据 | 有标签 | 无标签 |
| 目标 | 预测标签 | 发现结构 |
| 评估 | 准确率、F1等 | 轮廓系数、重构误差等 |
| 应用 | 分类、回归 | 聚类、降维、异常检测 |
| 难点 | 标签获取 | 结果解释、评估 |

## 🎯 算法选择指南

### 聚类算法选择

```
数据特点                    推荐算法
────────────────────────────────────────
球形簇、规模大              K-means
任意形状、含噪声            DBSCAN
需要层次结构                层次聚类
需要概率解释                GMM
```

### 降维方法选择

```
任务需求                    推荐方法
────────────────────────────────────────
特征压缩、去噪              PCA
分类预处理                  LDA
数据可视化                  t-SNE / UMAP
非线性结构                  流形学习
```

## 📐 数学基础

无监督学习涉及的核心数学概念：

### 线性代数
- 特征值分解与奇异值分解（SVD）
- 协方差矩阵与相关矩阵
- 正交投影与子空间

### 概率统计
- 概率密度估计
- 高斯混合模型
- 贝叶斯推断

### 优化理论
- EM算法
- 梯度下降
- KKT条件

## 💻 快速示例

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成示例数据
np.random.seed(42)
X = np.random.randn(300, 10)

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"聚类标签: {np.unique(labels)}")
```

## 📖 详细内容

- [聚类分析](./clustering.md) - 详解 K-means、层次聚类、DBSCAN、GMM 等算法
- [降维技术](./dimensionality-reduction.md) - 深入 PCA、LDA、t-SNE 等方法
- [关联规则与异常检测](./association-anomaly.md) - Apriori 算法与异常检测技术
- [高级数据结构](./advanced-structures.md) - KD树等空间索引结构

---

**前置知识**：[数学基础](../math-foundation/linear-algebra.md)  
**后续学习**：[模型评估](../model-evaluation/cross-validation.md)
