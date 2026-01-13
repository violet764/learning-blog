# 决策树与集成方法

# 决策树算法详解

## 1. 决策树基础

### 1.1 算法概述

决策树是一种基于树形结构的监督学习算法，通过一系列if-then规则对数据进行分类或回归。

### 1.2 树的基本结构

- **根节点**：包含所有训练数据的起始节点
- **内部节点**：根据特征进行数据划分的节点
- **叶节点**：最终的分类或回归结果
- **分支**：特征划分的条件

## 2. 信息熵与分裂准则

### 2.1 信息熵

信息熵衡量数据集的不确定性或混乱程度：
$$
H(D) = -\sum_{i=1}^{n} p_i \log_2 (p_i)
$$

其中：
- $D$：数据集
- $n$：类别的数量
- $p_i$：类别 $i$ 在数据集 $D$ 中的比例

**熵的性质**：
- 当数据集完全纯净时（只有一个类别），熵为0
- 当各类别均匀分布时，熵达到最大值
- 对于二分类问题，最大熵为1

### 2.2 信息增益  

信息增益衡量使用特征A进行分裂后，数据集不确定性的减少程度：
$$
Gain(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)
$$

其中：
- $A$：选择的特征
- $V$：特征 $A$ 的取值个数
- $D_v$：特征 $A$ 取值为 $v$ 的子集
- $|D_v|$：子集 $D_v$ 的样本数
- $|D|$：总样本数

**ID3算法**使用信息增益作为特征选择标准。

### 2.3 信息增益率

为了解决信息增益偏向取值多的特征的问题，**C4.5算法**使用信息增益率：
$$
GainRatio(D,A) = \frac{Gain(D,A)}{SplitInfo(D,A)}
$$

分裂信息（固有值）衡量特征取值的分布均匀程度：
$$
SplitInfo(D,A) = -\sum_{v=1}^V \frac{|D_v|}{|D|} \log_2\left( \frac{|D_v|}{|D|} \right)
$$

### 2.4 基尼系数

基尼系数衡量数据集的不纯度：
$$
Gini(D) = 1 - \sum_{i=1}^{n} p_i^2
$$

基尼增益衡量使用特征A分裂后的不纯度减少程度：
$$
\Delta Gini = Gini(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} Gini(D_v)
$$

**CART算法**使用基尼系数作为分裂标准。

## 2. 决策树构建算法

### 2.1 特征选择准则

#### 2.1.1 信息增益（ID3算法）

**信息熵：**
$$ H(D) = -\sum_{k=1}^K p_k \log_2 p_k $$

**条件熵：**
$$ H(D|A) = \sum_{i=1}^n \frac{|D_i|}{|D|} H(D_i) $$

**信息增益：**
$$ g(D, A) = H(D) - H(D|A) $$

#### 2.1.2 信息增益比（C4.5算法）

**特征A的固有值：**
$$ IV(A) = -\sum_{i=1}^n \frac{|D_i|}{|D|} \log_2 \frac{|D_i|}{|D|} $$

**信息增益比：**
$$ g_R(D, A) = \frac{g(D, A)}{IV(A)} $$

#### 2.1.3 基尼指数（CART算法）

**基尼指数：**
$$ \text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2 $$

**基尼指数增益：**
$$ \Delta \text{Gini}(D, A) = \text{Gini}(D) - \sum_{i=1}^n \frac{|D_i|}{|D|} \text{Gini}(D_i) $$

### 2.2 剪枝策略

#### 2.2.1 预剪枝

在构建过程中提前停止树的生长：
- 最大深度限制
- 最小样本分割数
- 最小信息增益阈值

#### 2.2.2 后剪枝

先构建完整的树，然后进行剪枝：
- 代价复杂度剪枝
- 错误率降低剪枝

## 3. 集成学习方法

### 3.1 Bagging（装袋法）

#### 3.1.1 基本思想

通过自助采样法构建多个基学习器，然后进行投票或平均。

#### 3.1.2 随机森林

**算法步骤：**
1. 从原始数据集中进行B次自助采样
2. 对每个自助样本构建决策树
3. 在节点分裂时，随机选择部分特征进行考虑
4. 所有树的结果进行投票或平均

**数学公式：**
对于分类问题：
$$ \hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_B(\mathbf{x})\} $$

对于回归问题：
$$ \hat{y} = \frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x}) $$

### 3.2 Boosting（提升法）

#### 3.2.1 AdaBoost

**算法步骤：**
1. 初始化样本权重$w_i = \frac{1}{n}$
2. 对于每轮$t=1,2,\dots,T$：
   - 用当前权重训练弱学习器$h_t$
   - 计算错误率$\epsilon_t = \sum_{i=1}^n w_i I(y_i \neq h_t(\mathbf{x}_i))$
   - 计算学习器权重$\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
   - 更新样本权重：$w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$
   - 归一化权重
3. 最终分类器：$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(\mathbf{x})\right)$

#### 3.2.2 梯度提升树（GBDT）

**算法思想：**
通过梯度下降的方式逐步改进模型，每一轮训练一个新的决策树来拟合前一轮模型的残差。

**数学推导：**
设损失函数为$L(y, F(\mathbf{x}))$，目标是找到函数$F$使得损失最小：
$$ F^*(\mathbf{x}) = \arg\min_F \mathbb{E}[L(y, F(\mathbf{x}))] $$

使用梯度下降：
$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) - \rho_m \nabla_F L(y, F_{m-1}(\mathbf{x})) $$

其中，第m棵树$h_m(\mathbf{x})$拟合负梯度：
$$ -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F(\mathbf{x})=F_{m-1}(\mathbf{x})} $$

#### 3.2.3 XGBoost

**目标函数：**
$$ \text{Obj}^{(t)} = \sum_{i=1}^n L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t) $$

其中正则化项：
$$ \Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2 $$

使用二阶泰勒展开近似：
$$ \text{Obj}^{(t)} \approx \sum_{i=1}^n [L(y_i, \hat{y}^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t) $$

其中$g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$, $h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)})$

## 4. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

# 分类任务示例
print("=== 分类任务 ===")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 单棵决策树
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

# 随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# 梯度提升树
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_gb = rf_clf.predict(X_test)

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

print("决策树准确率:", accuracy_score(y_test, y_pred_tree))
print("随机森林准确率:", accuracy_score(y_test, y_pred_rf))
print("梯度提升树准确率:", accuracy_score(y_test, y_pred_gb))
print("AdaBoost准确率:", accuracy_score(y_test, y_pred_ada))

# 特征重要性可视化
feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'tree': tree_clf.feature_importances_,
    'rf': rf_clf.feature_importances_,
    'gb': gb_clf.feature_importances_
})

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(feature_importance['feature'], feature_importance['tree'])
plt.title('决策树特征重要性')

plt.subplot(1, 3, 2)
plt.bar(feature_importance['feature'], feature_importance['rf'])
plt.title('随机森林特征重要性')

plt.subplot(1, 3, 3)
plt.bar(feature_importance['feature'], feature_importance['gb'])
plt.title('梯度提升树特征重要性')
plt.tight_layout()
plt.show()

# 回归任务示例
print("\n=== 回归任务 ===")
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# 回归树
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_train_reg, y_train_reg)
y_pred_tree_reg = tree_reg.predict(X_test_reg)

# 随机森林回归
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)

# 梯度提升回归
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_reg.fit(X_train_reg, y_train_reg)
y_pred_gb_reg = gb_reg.predict(X_test_reg)

print("回归树MSE:", mean_squared_error(y_test_reg, y_pred_tree_reg))
print("随机森林回归MSE:", mean_squared_error(y_test_reg, y_pred_rf_reg))
print("梯度提升回归MSE:", mean_squared_error(y_test_reg, y_pred_gb_reg))

# 决策树可视化
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, filled=True, feature_names=[f'feature_{i}' for i in range(X.shape[1])])
plt.title('决策树结构')
plt.show()
```

## 3. 详细数学计算示例

### 3.1 实际数据集分析

**示例数据集**
假设我们有一个简单的天气数据集用于预测是否适合打网球：

| 天气 | 温度 | 湿度 | 风速 | 是否打网球 |
|------|------|------|------|------------|
| 晴朗 | 高   | 高   | 弱   | 否         |
| 晴朗 | 高   | 高   | 强   | 否         |
| 多云 | 高   | 高   | 弱   | 是         |
| 下雨 | 中   | 高   | 弱   | 是         |
| 下雨 | 低   | 正常 | 弱   | 是         |
| 下雨 | 低   | 正常 | 强   | 否         |
| 多云 | 低   | 正常 | 强   | 是         |
| 晴朗 | 中   | 高   | 弱   | 否         |
| 晴朗 | 低   | 正常 | 弱   | 是         |
| 下雨 | 中   | 正常 | 弱   | 是         |
| 晴朗 | 中   | 正常 | 强   | 是         |
| 多云 | 中   | 高   | 强   | 是         |
| 多云 | 高   | 正常 | 弱   | 是         |
| 下雨 | 中   | 高   | 强   | 否         |

### 3.2 根节点信息熵计算

首先计算整个数据集的熵：
- 总样本数：14
- "是"类别样本数：9
- "否"类别样本数：5

$$
H(D) = -\left( \frac{9}{14} \log_2 \frac{9}{14} + \frac{5}{14} \log_2 \frac{5}{14} \right)
$$

计算各项：
- $\frac{9}{14} \approx 0.6429$, $\log_2 0.6429 \approx -0.6374$
- $\frac{5}{14} \approx 0.3571$, $\log_2 0.3571 \approx -1.4854$

$$
H(D) = -(0.6429 \times -0.6374 + 0.3571 \times -1.4854) \approx 0.940
$$

### 3.3 天气特征的信息增益计算

天气特征的取值分布：
- **晴朗**：5个样本（2个"是"，3个"否"）
- **多云**：4个样本（4个"是"，0个"否"）  
- **下雨**：5个样本（3个"是"，2个"否"）

计算各子集的熵：  
**晴朗子集的熵**：
$$
H(D_{晴朗}) = -\left( \frac{2}{5} \log_2 \frac{2}{5} + \frac{3}{5} \log_2 \frac{3}{5} \right) \approx 0.971
$$

**多云子集的熵**：
$$
H(D_{多云}) = -\left( \frac{4}{4} \log_2 \frac{4}{4} + \frac{0}{4} \log_2 \frac{0}{4} \right) = 0
$$

**下雨子集的熵**：
$$
H(D_{下雨}) = -\left( \frac{3}{5} \log_2 \frac{3}{5} + \frac{2}{5} \log_2 \frac{2}{5} \right) \approx 0.971
$$

计算加权平均熵：
$$
H_{天气}(D) = \frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.971 \approx 0.694
$$

计算信息增益：
$$
Gain(D, 天气) = H(D) - H_{天气}(D) = 0.940 - 0.694 = 0.246
$$

### 3.4 温度特征的信息增益计算

温度特征的取值分布：
- **高**：4个样本（2个"是"，2个"否"）
- **中**：6个样本（4个"是"，2个"否"）
- **低**：4个样本（3个"是"，1个"否"）

计算各子集的熵：  
**高温度子集的熵**：
$$
H(D_{高}) = -\left( \frac{2}{4} \log_2 \frac{2}{4} + \frac{2}{4} \log_2 \frac{2}{4} \right) = 1.0
$$

**中温度子集的熵**：
$$
H(D_{中}) = -\left( \frac{4}{6} \log_2 \frac{4}{6} + \frac{2}{6} \log_2 \frac{2}{6} \right) \approx 0.918
$$

**低温度子集的熵**：
$$
H(D_{低}) = -\left( \frac{3}{4} \log_2 \frac{3}{4} + \frac{1}{4} \log_2 \frac{1}{4} \right) \approx 0.811
$$

计算加权平均熵：
$$
H_{温度}(D) = \frac{4}{14} \times 1.0 + \frac{6}{14} \times 0.918 + \frac{4}{14} \times 0.811 \approx 0.911
$$

计算信息增益：
$$
Gain(D, 温度) = 0.940 - 0.911 = 0.029
$$

### 3.5 基尼系数计算示例

根节点的基尼系数：
$$
Gini(D) = 1 - \left( \left(\frac{9}{14}\right)^2 + \left(\frac{5}{14}\right)^2 \right) = 1 - (0.413 + 0.128) = 0.459
$$

天气特征的基尼增益：  
**晴朗子集的基尼系数**：
$$
Gini(D_{晴朗}) = 1 - \left( \left(\frac{2}{5}\right)^2 + \left(\frac{3}{5}\right)^2 \right) = 1 - (0.16 + 0.36) = 0.48
$$

**多云子集的基尼系数**：
$$
Gini(D_{多云}) = 1 - \left( \left(\frac{4}{4}\right)^2 + \left(\frac{0}{4}\right)^2 \right) = 0
$$

**下雨子集的基尼系数**：
$$
Gini(D_{下雨}) = 1 - \left( \left(\frac{3}{5}\right)^2 + \left(\frac{2}{5}\right)^2 \right) = 1 - (0.36 + 0.16) = 0.48
$$

计算基尼增益：
$$
\Delta Gini_{天气} = 0.459 - \left( \frac{5}{14} \times 0.48 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.48 \right) = 0.459 - 0.343 = 0.116
$$

### 3.6 特征选择结果比较

| 特征 | 信息增益 | 基尼增益 |
|------|----------|----------|
| 天气 | 0.246    | 0.116    |
| 温度 | 0.029    | 0.013    |

**结论**：天气特征的信息增益和基尼增益都最大，因此选择天气作为根节点的分裂特征。

### 5.2 梯度提升的数学原理

对于平方损失函数$L(y, F(\mathbf{x})) = \frac{1}{2}(y - F(\mathbf{x}))^2$：

**梯度计算：**
$$ g_i = \frac{\partial L}{\partial F(\mathbf{x}_i)} = -(y_i - F(\mathbf{x}_i)) $$

**Hessian矩阵：**
$$ h_i = \frac{\partial^2 L}{\partial F(\mathbf{x}_i)^2} = 1 $$

因此，第m棵树拟合的是残差$y_i - F_{m-1}(\mathbf{x}_i)$。

## 6. 应用场景

### 6.1 决策树应用
- **医疗诊断**：基于症状判断疾病
- **信用评分**：评估客户信用风险
- **客户细分**：根据特征进行客户分群

### 6.2 集成方法应用
- **推荐系统**：随机森林处理高维稀疏数据
- **图像识别**：梯度提升树用于特征提取
- **金融风控**：多种集成方法组合使用

## 7. 优缺点分析

### 7.1 决策树
**优点：**
- 可解释性强
- 不需要特征标准化
- 能够处理混合类型特征
- 对缺失值相对鲁棒

**缺点：**
- 容易过拟合
- 对数据波动敏感
- 忽略特征间的相关性

### 7.2 集成方法
**优点：**
- 显著提高预测精度
- 降低过拟合风险
- 能够处理复杂非线性关系

**缺点：**
- 计算成本较高
- 模型解释性变差
- 需要仔细的参数调优

## 8. 实践建议

1. **数据预处理**：处理缺失值，编码类别特征
2. **参数调优**：使用网格搜索优化超参数
3. **模型选择**：根据问题复杂度选择合适的方法
4. **特征工程**：利用特征重要性进行特征选择
5. **模型解释**：使用SHAP值等方法解释复杂模型

---

[下一节：k近邻与朴素贝叶斯](./knn-naive-bayes.md)