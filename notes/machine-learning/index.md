# 机器学习知识体系

本知识体系专注于**传统机器学习算法**的深度解析，强调**数学基础**和**算法原理**的严谨推导。按照系统性学习路径分为四大模块，从数学基础到实际应用，构建完整的机器学习知识框架。

## 📚 知识结构

### 第一阶段：数学基础

| 模块 | 文档 | 核心内容 |
|------|------|----------|
| 线性代数 | [linear-algebra.md](./math-foundation/linear-algebra.md) | 向量空间、特征值分解、SVD、协方差矩阵 |
| 概率统计 | [probability-statistics.md](./math-foundation/probability-statistics.md) | 概率分布、参数估计、假设检验 |
| 优化理论 | [optimization-theory.md](./math-foundation/optimization-theory.md) | 凸优化、梯度方法、KKT 条件 |

### 第二阶段：监督学习

**[监督学习模块](./supervised-learning/index.md)** 包含五个子模块：

| 子模块 | 核心算法 | 文档入口 |
|--------|----------|----------|
| 线性模型 | 线性回归、逻辑回归、正则化 | [linear-models](./supervised-learning/linear-models/index.md) |
| 支持向量机 | SVM 理论、核方法、SMO 实现 | [svm](./supervised-learning/svm/index.md) |
| 树模型与集成 | 决策树、随机森林、AdaBoost、GBDT、XGBoost、LightGBM | [tree-models](./supervised-learning/tree-models/index.md) |
| 贝叶斯方法 | 朴素贝叶斯、EM 算法、高斯过程 | [bayesian-methods](./supervised-learning/bayesian-methods/index.md) |
| 理论框架 | ERM 原则、泛化误差、偏差-方差 | [framework](./supervised-learning/framework/supervised-learning.md) |

### 第三阶段：无监督学习

**[无监督学习模块](./unsupervised-learning/index.md)**：

| 文档 | 核心内容 |
|------|----------|
| [clustering.md](./unsupervised-learning/clustering.md) | K-means、层次聚类、DBSCAN、GMM |
| [dimensionality-reduction.md](./unsupervised-learning/dimensionality-reduction.md) | PCA、LDA、流形学习、自编码器 |
| [association-anomaly.md](./unsupervised-learning/association-anomaly.md) | Apriori 算法、孤立森林、LOF |
| [advanced-structures.md](./unsupervised-learning/advanced-structures.md) | KD 树原理与应用 |

### 第四阶段：评估与工程

| 模块 | 文档 | 核心内容 |
|------|------|----------|
| 交叉验证 | [cross-validation.md](./model-evaluation/cross-validation.md) | K-Fold、留一法、分层抽样 |
| 统计检验 | [statistical-tests.md](./model-evaluation/statistical-tests.md) | t 检验、McNemar 检验、Wilcoxon 检验 |
| 学习理论 | [learning-theory.md](./model-evaluation/learning-theory.md) | PAC 学习、VC 维、泛化界 |
| 特征工程 | [index.md](./feature-engineering/index.md) | 特征选择、特征变换、数据预处理 |

## 🚀 快速导航

### 按学习阶段

- **初学者**：数学基础 → 线性模型 → 模型评估
- **实践者**：线性模型 → 树模型（XGBoost）→ 项目实践
- **研究者**：理论框架 → SVM → 学习理论

### 按算法类型

**监督学习**：
- [线性回归](./supervised-learning/linear-models/linear-regression.md) - 回归任务基础
- [逻辑回归](./supervised-learning/linear-models/logistic-regression.md) - 二分类经典
- [SVM](./supervised-learning/svm/svm-theory.md) - 最大间隔分类器
- [决策树](./supervised-learning/tree-models/decision-trees.md) - 可解释性强
- [XGBoost](./supervised-learning/tree-models/xgboost.md) - 表格数据利器
- [朴素贝叶斯](./supervised-learning/bayesian-methods/naive-bayes.md) - 文本分类首选

**无监督学习**：
- [聚类分析](./unsupervised-learning/clustering.md) - K-means、DBSCAN、GMM
- [降维技术](./unsupervised-learning/dimensionality-reduction.md) - PCA、LDA、流形学习
- [异常检测](./unsupervised-learning/association-anomaly.md) - 孤立森林、LOF

## 📖 学习路径

```
数学基础              监督学习              无监督学习           评估工程
    │                    │                      │                   │
    ▼                    ▼                      ▼                   ▼
线性代数 ──────→ 线性模型 ──────→ 聚类分析 ──────→ 交叉验证
概率统计 ──────→ SVM ────────────→ 降维技术 ──────→ 统计检验
优化理论 ──────→ 树模型 ──────────→ 异常检测 ──────→ 特征工程
                 贝叶斯方法 ────────→
```

## 🎯 核心特色

| 特色 | 说明 |
|------|------|
| 数学深度 | 完整推导、收敛性证明、误差分析 |
| 代码实现 | 核心算法手写实现、sklearn 应用示例 |
| 实践指导 | 算法选择、参数调优、常见问题 |
| 结构清晰 | 模块化组织、循序渐进、前后衔接 |

## 💡 算法选择速查

| 数据特点 | 推荐算法 |
|----------|----------|
| 小样本、线性关系 | 线性回归/逻辑回归 |
| 小样本、非线性 | SVM + 核函数 |
| 表格数据、追求精度 | XGBoost / LightGBM |
| 大规模数据 | LightGBM / 线性模型 |
| 需要可解释性 | 决策树 / 线性模型 |
| 需要不确定性估计 | 贝叶斯方法 |
| 无标签数据 | K-means / PCA |

## 🔧 相关资源

- [深度学习笔记](../deep-learning/index.md)
- [强化学习笔记](../reinforcement-learning/index.md)
- [PyTorch 教程](../deep-learning/pytorch/index.md)

---

**版本**：v2.2  
**更新日期**：2026年3月1日