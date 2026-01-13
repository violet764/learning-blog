# 线性回归算法

## 1. 算法概述

线性回归是监督学习中最基础的回归算法，用于建立输入特征与连续目标变量之间的线性关系模型。

### 数学模型

**基本形式：**
$$ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p + \epsilon $$

其中：
- $y$：目标变量（连续值）
- $x_i$：第i个特征变量
- $\beta_0$：截距项
- $\beta_i$：第i个特征的系数
- $\epsilon$：误差项（服从正态分布）

### 矩阵表示
$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon} $$

其中：
- $\mathbf{y} \in \mathbb{R}^n$：目标向量
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$：设计矩阵（包含截距项）
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$：参数向量

## 2. 参数估计方法

### 2.1 最小二乘法（OLS）

**目标函数：**
$$ \min_{\boldsymbol{\beta}} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 = \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 $$

**解析解：**
$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} $$

### 2.2 梯度下降法

对于大规模数据，使用迭代优化方法：

**梯度更新：**
$$ \boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla J(\boldsymbol{\beta}^{(t)}) $$

**梯度计算：**
$$ \nabla J(\boldsymbol{\beta}) = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$

## 3. 正则化方法

### 3.1 岭回归（Ridge Regression）

**目标函数：**
$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_2^2 $$

**解析解：**
$$ \hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} $$

### 3.2 Lasso回归

**目标函数：**
$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 $$

## 4. 模型评估

### 4.1 评价指标

- **均方误差（MSE）：** $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **均方根误差（RMSE）：** $\sqrt{MSE}$
- **平均绝对误差（MAE）：** $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **决定系数（R²）：** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### 4.2 假设检验

- t检验：检验单个系数的显著性
- F检验：检验模型整体显著性

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 生成示例数据
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
true_coef = np.array([2.5, -1.2, 0.8])
y = X.dot(true_coef) + np.random.randn(n_samples) * 0.5

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 普通线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# 模型评估
print("线性回归 - MSE:", mean_squared_error(y_test, y_pred_lr), "R²:", r2_score(y_test, y_pred_lr))
print("岭回归 - MSE:", mean_squared_error(y_test, y_pred_ridge), "R²:", r2_score(y_test, y_pred_ridge))
print("Lasso回归 - MSE:", mean_squared_error(y_test, y_pred_lasso), "R²:", r2_score(y_test, y_pred_lasso))

# 系数比较
print("\n真实系数:", true_coef)
print("线性回归系数:", lr.coef_)
print("岭回归系数:", ridge.coef_)
print("Lasso回归系数:", lasso.coef_)
```

## 6. 数学推导

### 6.1 最小二乘法的推导

设损失函数：
$$ J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 $$

展开得：
$$ J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$
$$ = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} $$

对$\boldsymbol{\beta}$求导并令导数为零：
$$ \frac{\partial J}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0 $$

解得：
$$ \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} $$
$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} $$

### 6.2 岭回归的贝叶斯解释

从贝叶斯角度看，岭回归等价于给参数$\boldsymbol{\beta}$加上高斯先验：
$$ p(\boldsymbol{\beta}) = \mathcal{N}(0, \frac{1}{\lambda}\mathbf{I}) $$

后验分布为：
$$ p(\boldsymbol{\beta}|\mathbf{X},\mathbf{y}) \propto p(\mathbf{y}|\mathbf{X},\boldsymbol{\beta})p(\boldsymbol{\beta}) $$

最大后验估计（MAP）即为岭回归的解。

## 7. 应用场景

1. **房价预测**：基于房屋特征预测价格
2. **销量预测**：基于历史数据预测未来销量
3. **金融风险评估**：基于客户特征预测信用风险
4. **医疗诊断**：基于生理指标预测疾病风险

## 8. 优缺点分析

### 优点：
- 模型简单，解释性强
- 计算效率高，有解析解
- 理论基础完善

### 缺点：
- 对非线性关系建模能力有限
- 对异常值敏感
- 需要满足线性回归的基本假设

---

[下一节：逻辑回归算法](./logistic-regression.md)