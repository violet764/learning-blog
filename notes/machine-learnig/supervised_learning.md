# 有监督学习算法

## KNN
K-最近邻（K-Nearest Neighbors，简称KNN）是一种基于实例的监督学习算法，可用于分类和回归任务。它的核心思想是：**相似的对象具有相似的特征**。

**基本思想**
给定一个测试样本，KNN算法在训练集中找到与该样本最相似的K个样本（即最近的邻居），然后根据这K个样本的标签来预测测试样本的标签。

KNN算法使用距离函数来度量样本之间的相似性。常用的<span style="background-color: #8c30b9ff; padding: 2px 4px; border-radius: 3px; color: #333;">距离度量</span>包括：

::: tip 常用距离
欧几里得距离（最常用）
$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

曼哈顿距离
$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

闵可夫斯基距离（通用形式）
$$
d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$
- 当p=1时，为曼哈顿距离
- 当p=2时，为欧几里得距离

:::

**算法的计算步骤：**
1. **计算距离**：计算测试样本与所有训练样本之间的距离
2. **排序**：将距离按升序排列，选择距离最小的K个样本
3. **投票/平均**：
   - **分类问题**：统计K个最近邻中各类别的出现频率，选择频率最高的类别
   - **回归问题**：计算K个最近邻的目标值的平均值

K值的选择对算法性能有重要影响：
- **K值太小**：模型复杂，容易过拟合，对噪声敏感
- **K值太大**：模型简单，可能欠拟合，决策边界平滑
- **经验法则**：K通常取奇数，避免平票情况；常用交叉验证选择最优K值


简单代码实现：
```python
# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./boston.csv')

# 1. 加载数据
# 分离特征（X）和目标变量（y）——目标列默认为'MEDV'，需根据实际列名调整
X = df.drop("MEDV", axis=1)  # 所有列除了房价列，作为特征
y = df["MEDV"]     

# 2. 数据预处理：划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 测试集占20%，随机种子保证结果可复现
)

# 3. 特征标准化（KNN对特征尺度敏感，必须标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合+转换
X_test_scaled = scaler.transform(X_test)        # 测试集仅转换（避免数据泄露）

# 4. 构建KNN回归模型
knn = KNeighborsRegressor(n_neighbors=5)  # 选择K=5（可调整）

# 5. 模型训练
knn.fit(X_train_scaled, y_train)

# 6. 模型预测
y_pred = knn.predict(X_test_scaled)

# 7. 模型评估
mse = mean_squared_error(y_test, y_pred)  # 均方误差
rmse = np.sqrt(mse)                       # 均方根误差
r2 = r2_score(y_test, y_pred)             # 决定系数（越接近1越好）

# 输出评估结果
print("KNN回归模型评估结果：")
print(f"均方误差(MSE)：{mse:.2f}")
print(f"均方根误差(RMSE)：{rmse:.2f}")
print(f"决定系数(R²)：{r2:.2f}")

```  
```plaintext
KNN回归模型评估结果：
均方误差(MSE)：20.61
均方根误差(RMSE)：4.54
决定系数(R²)：0.72
```

---

## 决策树

决策树是一种模仿人类决策过程的监督学习算法，可用于分类和回归任务。它通过一系列规则构建树状结构，实现对数据的分类或预测。

**决策树的工作原理:** 决策树通过递归地将数据集分割成更小的子集来构建树结构。具体步骤如下：  

1. 选择最佳特征：根据某种标准（如[信息增益、基尼指数](./decision_tree.md)等）选择最佳特征进行分割。
2. 分割数据集：根据选定的特征将数据集分成多个子集。
3. 递归构建子树：对每个子集重复上述过程，直到满足停止条件（如所有样本属于同一类别、达到最大深度等）。
4. 生成叶节点：当满足停止条件时，生成叶节点并赋予类别或值。  


**决策树的剪枝**：决策树容易过拟合，为了减少过拟合，决策树需要进行剪枝处理：

1. 预剪枝：在构建过程中提前停止树的生长。
2. 后剪枝：先构建完整的树，然后自底向上剪去不必要的子树。

**分类问题示例（鸢尾花数据集）**：

```python
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
feature_names = iris.feature_names
y = iris.target  # 目标：鸢尾花种类（0: setosa, 1: versicolor, 2: virginica）
target_names = iris.target_names

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 创建决策树分类器,决策树回归器用DecisionTreeRegressor类
dtree = DecisionTreeClassifier(
    criterion='gini',      # 分裂标准：基尼系数
    max_depth=3,          # 最大深度，防止过拟合
    min_samples_split=5,  # 内部节点最小样本数
    min_samples_leaf=2,   # 叶节点最小样本数
    random_state=42
)

# 4. 训练模型
dtree.fit(X_train, y_train)

# 5. 预测
y_pred = dtree.predict(X_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树分类准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))
```

---

## 线性回归  

线性回归是最基础且广泛应用的回归算法，用于建立连续目标变量与一个或多个特征之间的线性关系模型。

**基本概念**

**核心思想**：通过线性方程来描述特征与目标变量之间的关系：
$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n + \epsilon
$$

其中：
- $y$：目标变量（因变量）
- $x_1, x_2, \cdots, x_n$：特征变量（自变量）
- $\beta_0$：截距项（偏置）
- $\beta_1, \beta_2, \cdots, \beta_n$：系数（权重）
- $\epsilon$：误差项

**模型求解方法**

**1. 最小二乘法（OLS）**

最小二乘法通过最小化残差平方和来求解模型参数：
$$
\min_{\beta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \min_{\beta} \sum_{i=1}^{m} (y_i - \beta_0 - \beta_1x_{i1} - \cdots - \beta_nx_{in})^2
$$

**矩阵形式求解**：[具体求解过程](./OLS.md)
$$
\beta = (X^TX)^{-1}X^Ty
$$

**2. 梯度下降法** 

对于大规模数据集，使用迭代优化方法：
$$
\beta_j := \beta_j - \alpha \frac{\partial}{\partial\beta_j}J(\beta)
$$

其中 $J(\beta)$ 是损失函数，$\alpha$ 是学习率。

**线性回归的假设条件**

线性回归模型的有效性依赖于以下假设：

1. **线性关系**：特征与目标变量之间存在线性关系
2. **独立性**：观测值之间相互独立
3. **同方差性**：误差项的方差恒定
4. **正态分布**：误差项服从正态分布
5. **无多重共线性**：特征之间不存在高度相关性


**简单线性回归实现：**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 特征：0-10之间的随机数
y = 2.5 * X.squeeze() + 1.5 + np.random.randn(100) * 2  # 线性关系加噪声

# 创建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X, y)

# 预测
y_pred = lr.predict(X)

# 模型参数
print(f"截距 (bias): {lr.intercept_:.4f}")
print(f"系数 (weight): {lr.coef_[0]:.4f}")

# 模型评估
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='实际数据')
plt.plot(X, y_pred, color='red', linewidth=2, label='回归直线')
plt.xlabel('特征 X')
plt.ylabel('目标 y')
plt.title('简单线性回归')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```


---

## 逻辑回归  

逻辑回归虽然名字中有"回归"，但实际上是一种广泛使用的分类算法，特别适用于二分类问题。它通过逻辑函数将线性回归的输出映射到概率空间。


**核心思想**：使用逻辑函数（Sigmoid函数）将线性组合的结果映射到(0,1)区间，表示属于某个类别的概率。


**逻辑函数（Sigmoid函数）**：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

其中 $z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n$

**预测概率**：
$$
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \cdots + \beta_nx_n)}}
$$


[逻辑回归的数学推导](./logistic.md)

**多分类逻辑回归**

**1. One-vs-Rest (OvR)**  
为每个类别训练一个二分类器，选择概率最高的类别。

**2. Softmax回归**
$$
P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}
$$

**模型评估指标（分类问题）**

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# 常用分类评估指标
accuracy = accuracy_score(y_true, y_pred)          # 准确率
precision = precision_score(y_true, y_pred)       # 精确率
recall = recall_score(y_true, y_pred)             # 召回率
f1 = f1_score(y_true, y_pred)                      # F1分数
roc_auc = roc_auc_score(y_true, y_pred_proba)     # AUC值
```


**逻辑回归实现：**

```python
from sklearn.datasets import load_iris

# 加载鸢尾花数据集（三分类问题）
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("鸢尾花数据集信息:")
print(f"类别数量: {len(np.unique(y))}")
print(f"类别名称: {target_names}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 创建多分类逻辑回归模型
multi_log_reg = LogisticRegression(
    multi_class='multinomial',  # 多分类策略：softmax
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    random_state=42
)

# 训练模型
multi_log_reg.fit(X_train, y_train)

# 预测
y_pred = multi_log_reg.predict(X_test)
y_pred_proba = multi_log_reg.predict_proba(X_test)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n多分类逻辑回归准确率: {accuracy:.4f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('多分类逻辑回归混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient_class0': multi_log_reg.coef_[0],
    'coefficient_class1': multi_log_reg.coef_[1],
    'coefficient_class2': multi_log_reg.coef_[2]
})

print("\n特征系数（每个类别的权重）:")
print(feature_importance)
```


**逻辑回归的正则化**

逻辑回归容易过拟合，需要正则化：

**1. L1正则化（Lasso）**
- 产生稀疏解，可用于特征选择
- 惩罚项：$\lambda \sum |\beta_j|$

**2. L2正则化（Ridge）**
- 防止系数过大
- 惩罚项：$\lambda \sum \beta_j^2$

**3. 弹性网络（Elastic Net）**
- L1和L2正则化的组合

```python
# 不同正则化类型的比较
from sklearn.linear_model import LogisticRegression

# L1正则化
l1_log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
l1_log_reg.fit(X_train, y_train)

# L2正则化（默认）
l2_log_reg = LogisticRegression(penalty='l2', C=1.0)
l2_log_reg.fit(X_train, y_train)

print("L1正则化系数:", l1_log_reg.coef_)
print("L2正则化系数:", l2_log_reg.coef_)
```
