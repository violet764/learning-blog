# 监督学习数学理论框架

本文档聚焦监督学习的**数学理论基础**，涵盖统计学习理论、模型选择理论、优化理论等核心内容。各算法的详细推导请参考对应的子模块文档。

## 1. 统计学习理论

### 1.1 问题形式化

**监督学习问题**：给定训练数据集 $D = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_n, y_n)\}$，其中 $\mathbf{x}_i \in \mathcal{X}$ 是输入，$y_i \in \mathcal{Y}$ 是输出。假设数据由未知分布 $P(\mathbf{x}, y)$ 独立同分布地生成，目标是学习映射 $f: \mathcal{X} \to \mathcal{Y}$。

**任务类型**：
- **回归**：$\mathcal{Y} \subseteq \mathbb{R}$
- **分类**：$\mathcal{Y} = \{1, 2, \ldots, K\}$

### 1.2 风险与经验风险

**风险函数（期望风险）**：
$$R(f) = \mathbb{E}_{(\mathbf{x},y) \sim P}[L(f(\mathbf{x}), y)]$$

**经验风险**：
$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n L(f(\mathbf{x}_i), y_i)$$

**ERM 原则**：$\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{R}(f)$

### 1.3 泛化误差分解

泛化误差可分解为：

$$R(\hat{f}) - R(f^*) = \underbrace{R(\hat{f}) - R(f_{\mathcal{F}})}_{\text{估计误差}} + \underbrace{R(f_{\mathcal{F}}) - R(f^*)}_{\text{近似误差}}$$

- **估计误差**：有限样本导致的误差，与假设空间复杂度相关
- **近似误差**：假设空间表达能力有限导致的误差

### 1.4 VC 维与泛化界

**VC 维**：假设空间 $\mathcal{F}$ 能打散的最大样本数，衡量假设空间复杂度。

**泛化误差界**：以概率至少 $1-\delta$，有

$$R(f) \leq \hat{R}(f) + \mathcal{O}\left(\sqrt{\frac{VC(\mathcal{F}) \log n + \log(1/\delta)}{n}}\right)$$

**核心洞察**：模型复杂度（VC 维）与样本量的权衡决定了泛化能力。

## 2. 模型选择理论

### 2.1 偏差-方差权衡

**误差分解**：
$$\mathbb{E}[(\hat{y} - y)^2] = \text{Bias}^2(\hat{y}) + \text{Var}(\hat{y}) + \sigma^2$$

| 误差来源 | 定义 | 与模型复杂度关系 |
|----------|------|------------------|
| 偏差 | $\mathbb{E}[\hat{y}] - y$ | 复杂度↑ → 偏差↓ |
| 方差 | $\mathbb{E}[(\hat{y} - \mathbb{E}[\hat{y}])^2]$ | 复杂度↑ → 方差↑ |
| 噪声 | $\sigma^2$ | 不可约减 |

**实践意义**：需要在偏差和方差之间寻找平衡点。

### 2.2 结构风险最小化

在经验风险基础上加入正则化项：

$$\min_{f \in \mathcal{F}} \hat{R}(f) + \lambda \Omega(f)$$

常见正则化：
- **L2 正则化**：$\Omega(f) = \|\mathbf{w}\|_2^2$（Ridge）
- **L1 正则化**：$\Omega(f) = \|\mathbf{w}\|_1$（Lasso）

### 2.3 模型选择准则

| 准则 | 公式 | 特点 |
|------|------|------|
| AIC | $-2\ln L + 2k$ | 偏向选择更复杂模型 |
| BIC | $-2\ln L + k\ln n$ | 样本量大时惩罚更强 |
| 交叉验证 | $CV_k$ 误差 | 直接估计泛化误差 |

## 3. 优化理论

### 3.1 凸优化基础

**凸集**：对任意 $\mathbf{x}, \mathbf{y} \in C$，有 $\lambda \mathbf{x} + (1-\lambda)\mathbf{y} \in C$，$\lambda \in [0,1]$

**凸函数**：$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$

**重要性质**：凸函数的局部最优即全局最优。

### 3.2 梯度下降

**批量梯度下降**：
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla \hat{R}(\mathbf{w}^{(t)})$$

**随机梯度下降（SGD）**：
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla L(f(\mathbf{x}_i), y_i)$$

**收敛性**：
- 凸函数：$O(1/\sqrt{T})$
- 强凸函数：$O(1/T)$
- 强凸+光滑：线性收敛

### 3.3 约束优化与 KKT 条件

**原始问题**：
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \; h_j(\mathbf{x}) = 0$$

**KKT 条件**（最优解的必要条件）：
1. $\nabla f(\mathbf{x}^*) + \sum_i \lambda_i \nabla g_i(\mathbf{x}^*) + \sum_j \nu_j \nabla h_j(\mathbf{x}^*) = 0$
2. $g_i(\mathbf{x}^*) \leq 0$
3. $h_j(\mathbf{x}^*) = 0$
4. $\lambda_i \geq 0$
5. $\lambda_i g_i(\mathbf{x}^*) = 0$（互补松弛）

## 4. 评估与检验

### 4.1 评估指标

**分类指标**：

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| 准确率 | $\frac{TP+TN}{TP+TN+FP+FN}$ | 类别均衡 |
| 精确率 | $\frac{TP}{TP+FP}$ | 关注误报 |
| 召回率 | $\frac{TP}{TP+FN}$ | 关注漏报 |
| F1 | $2 \cdot \frac{P \cdot R}{P + R}$ | 精确率与召回率平衡 |

**回归指标**：

| 指标 | 公式 | 特点 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 对异常值敏感 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 鲁棒性更好 |
| $R^2$ | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 可解释性强 |

### 4.2 统计检验

详见 [统计检验理论](../../model-evaluation/statistical-tests.md)。

### 4.3 交叉验证

详见 [交叉验证方法](../../model-evaluation/cross-validation.md)。

## 5. 核心挑战

| 挑战 | 描述 | 应对策略 |
|------|------|----------|
| 过拟合 | 模型在训练集表现好、测试集差 | 正则化、交叉验证、早停 |
| 欠拟合 | 模型无法捕捉数据规律 | 增加模型复杂度、特征工程 |
| 维度灾难 | 高维数据稀疏、距离失效 | 降维、特征选择 |
| 数据不平衡 | 类别样本数差异大 | 重采样、代价敏感学习 |

## 6. 相关资源

### 各算法详细文档

- **线性模型**：[linear-models](../linear-models/index.md)
- **支持向量机**：[svm](../svm/index.md)
- **树模型与集成**：[tree-models](../tree-models/index.md)
- **贝叶斯方法**：[bayesian-methods](../bayesian-methods/index.md)

### 延伸阅读

- [模型评估与选择](../../model-evaluation/cross-validation.md)
- [学习理论](../../model-evaluation/learning-theory.md)
- [特征工程](../../feature-engineering/feature-selection.md)

---

[返回监督学习](../index.md)
