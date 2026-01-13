# 逻辑回归算法

## 1. 算法概述

逻辑回归是一种用于解决二分类问题的监督学习算法，尽管名字中有"回归"，但实际上是一种分类算法。它通过逻辑函数将线性组合映射到概率空间。

### 数学模型

**基本形式：**
$$ P(y=1|\mathbf{x}) = \sigma(\mathbf{x}^T\boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}} $$

其中：
- $P(y=1|\mathbf{x})$：给定特征$\mathbf{x}$时目标变量为1的概率
- $\sigma(z)$：sigmoid函数，$\sigma(z) = \frac{1}{1+e^{-z}}$
- $\boldsymbol{\beta}$：模型参数向量

### 决策边界

分类决策基于概率阈值（通常为0.5）：
$$ \hat{y} = \begin{cases} 
1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\
0 & \text{if } P(y=1|\mathbf{x}) < 0.5 
\end{cases} $$

## 2. 损失函数与参数估计

### 2.1 交叉熵损失函数

逻辑回归使用**交叉熵损失函数（Cross-Entropy Loss）**，也称为对数损失函数：

对于单个样本 $(\mathbf{x}^{(i)}, y^{(i)})$ 的损失：
$$ L^{(i)}(\boldsymbol{\beta}) = -\left[ y^{(i)} \log(h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) \right] $$

对于所有样本的平均损失：
$$ J(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)})) \right] $$

其中 $h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) = \sigma(\mathbf{x}^{(i)T}\boldsymbol{\beta})$。

### 2.2 最大似然估计等价性

交叉熵损失函数与最大似然估计等价：

**似然函数：**
$$ L(\boldsymbol{\beta}) = \prod_{i=1}^n P(y_i|\mathbf{x}_i) = \prod_{i=1}^n [\sigma(\mathbf{x}_i^T\boldsymbol{\beta})]^{y_i}[1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta})]^{1-y_i} $$

**对数似然函数：**
$$ \ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i\ln\sigma(\mathbf{x}_i^T\boldsymbol{\beta}) + (1-y_i)\ln(1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta}))] $$

最大似然估计等价于最小化负对数似然，即交叉熵损失。

### 2.3 梯度下降法

**梯度计算详细推导：**

令 $a^{(i)} = h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) = \sigma(z^{(i)})$，其中 $z^{(i)} = \mathbf{x}^{(i)T}\boldsymbol{\beta}$

**Sigmoid函数导数：**
$$ \sigma'(z) = \sigma(z)(1-\sigma(z)) $$

**单个样本损失对 $z^{(i)}$ 的导数：**
$$ \frac{\partial L^{(i)}}{\partial z^{(i)}} = \frac{\partial L^{(i)}}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial z^{(i)}} $$

第一部分：
$$ \frac{\partial L^{(i)}}{\partial a^{(i)}} = -\left[ \frac{y^{(i)}}{a^{(i)}} - \frac{1 - y^{(i)}}{1 - a^{(i)}} \right] = \frac{1 - y^{(i)}}{1 - a^{(i)}} - \frac{y^{(i)}}{a^{(i)}} $$

第二部分：
$$ \frac{\partial a^{(i)}}{\partial z^{(i)}} = a^{(i)}(1 - a^{(i)}) $$

合并：
$$ \frac{\partial L^{(i)}}{\partial z^{(i)}} = \left[ \frac{1 - y^{(i)}}{1 - a^{(i)}} - \frac{y^{(i)}}{a^{(i)}} \right] \cdot a^{(i)}(1 - a^{(i)}) = a^{(i)} - y^{(i)} $$

**对参数 $\beta_j$ 的梯度：**
$$ \frac{\partial L^{(i)}}{\partial \beta_j} = \frac{\partial L^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \beta_j} = (a^{(i)} - y^{(i)}) \cdot x_j^{(i)} $$

**平均梯度：**
$$ \frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} $$

**向量形式：**
$$ \nabla J(\boldsymbol{\beta}) = \frac{1}{m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)}) \mathbf{x}^{(i)} = \frac{1}{m} \mathbf{X}^T(\mathbf{h}_{\boldsymbol{\beta}} - \mathbf{y}) $$

**参数更新：**
$$ \boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla J(\boldsymbol{\beta}^{(t)}) $$

## 3. 正则化方法

### 3.1 L2正则化（岭回归风格）

$$ \ell_{reg}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2 $$

### 3.2 L1正则化（Lasso风格）

$$ \ell_{reg}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \lambda\|\boldsymbol{\beta}\|_1 $$

## 4. 模型评估

### 4.1 分类评价指标

- **准确率（Accuracy）：** $\frac{TP+TN}{TP+TN+FP+FN}$
- **精确率（Precision）：** $\frac{TP}{TP+FP}$
- **召回率（Recall）：** $\frac{TP}{TP+FN}$
- **F1分数：** $2 \times \frac{Precision \times Recall}{Precision + Recall}$

### 4.2 ROC曲线和AUC

- ROC曲线：真阳性率 vs 假阳性率
- AUC：曲线下面积，衡量模型整体性能

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 生成二分类数据
X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0, 
                          n_informative=4, n_clusters_per_class=1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 逻辑回归模型
logistic = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
logistic.fit(X_train, y_train)

# 预测
y_pred = logistic.predict(X_test)
y_pred_proba = logistic.predict_proba(X_test)[:, 1]

# 模型评估
print("准确率:", accuracy_score(y_test, y_pred))
print("精确率:", precision_score(y_test, y_pred))
print("召回率:", recall_score(y_test, y_pred))
print("F1分数:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_proba))

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC曲线 (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 系数解释
print("\n模型系数:", logistic.coef_)
print("截距项:", logistic.intercept_)

# 计算特征重要性（使用系数绝对值）
feature_importance = np.abs(logistic.coef_[0])
print("特征重要性:", feature_importance)
```

## 6. 多分类扩展

### 6.1 One-vs-Rest (OvR)

对于K类分类问题，训练K个二分类器：
- 第k个分类器：将第k类作为正类，其他所有类作为负类

### 6.2 Softmax回归（多分类逻辑回归）

**概率模型：**
$$ P(y=k|\mathbf{x}) = \frac{e^{\mathbf{x}^T\boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T\boldsymbol{\beta}_j}} $$

## 7. 数学推导

### 7.1 Sigmoid函数的性质

**导数计算：**
$$ \sigma'(z) = \sigma(z)(1-\sigma(z)) $$

**证明：**
$$ \sigma(z) = \frac{1}{1+e^{-z}} $$
$$ \sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z)) $$

### 7.2 梯度推导

对数似然函数：
$$ \ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i\ln\sigma_i + (1-y_i)\ln(1-\sigma_i)] $$

其中$\sigma_i = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$

对$\beta_j$求导：
$$ \frac{\partial\ell}{\partial\beta_j} = \sum_{i=1}^n \left[\frac{y_i}{\sigma_i} - \frac{1-y_i}{1-\sigma_i}\right]\frac{\partial\sigma_i}{\partial\beta_j} $$

由于$\frac{\partial\sigma_i}{\partial\beta_j} = \sigma_i(1-\sigma_i)x_{ij}$，代入得：
$$ \frac{\partial\ell}{\partial\beta_j} = \sum_{i=1}^n (y_i - \sigma_i)x_{ij} $$

## 8. 应用场景

1. **信用评分**：预测客户是否会违约
2. **医疗诊断**：预测患者是否患有某种疾病
3. **营销响应**：预测客户是否会响应营销活动
4. **垃圾邮件检测**：判断邮件是否为垃圾邮件

## 9. 优缺点分析

### 优点：
- 输出具有概率解释
- 计算效率高
- 可解释性强（系数可解释为对数几率）
- 对线性可分数据表现良好

### 缺点：
- 假设特征与对数几率呈线性关系
- 对非线性关系建模能力有限
- 对多重共线性敏感
- 需要较大的样本量

## 10. 实践建议

1. **特征工程**：确保特征与目标变量有线性关系
2. **数据预处理**：标准化连续特征，处理类别特征
3. **正则化选择**：根据特征数量和数据量选择L1或L2正则化
4. **阈值调整**：根据业务需求调整分类阈值

---

[下一节：支持向量机](./support-vector-machines.md)