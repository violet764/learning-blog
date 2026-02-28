# 线性回归

线性回归是监督学习中最基础、最重要的回归算法。它通过建立输入特征与连续目标变量之间的**线性关系**来进行预测，是理解更复杂模型的基石。

📌 **核心思想**：假设目标变量与特征之间存在线性关系，通过最小化预测误差来估计模型参数。

## 基本概念

### 数学模型

**单变量线性回归**：

$$y = \beta_0 + \beta_1 x + \epsilon$$

其中：
- $y$：目标变量（连续值）
- $x$：特征变量
- $\beta_0$：截距项（偏置）
- $\beta_1$：回归系数（斜率）
- $\epsilon$：误差项，假设 $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**多变量线性回归**：

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

**矩阵形式**：

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

其中：
- $\mathbf{y} \in \mathbb{R}^n$：目标向量
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$：设计矩阵，第一列为全1（对应截距）
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$：参数向量
- $\boldsymbol{\epsilon} \in \mathbb{R}^n$：误差向量

### 模型假设

线性回归的有效性依赖于以下假设：

| 假设 | 描述 | 违反后果 |
|------|------|----------|
| **线性性** | 因变量与自变量呈线性关系 | 模型拟合不足 |
| **独立性** | 误差项相互独立 | 标准误估计偏差 |
| **同方差性** | 误差方差恒定 | 参数检验失效 |
| **正态性** | 误差服从正态分布 | 置信区间不准确 |
| **无多重共线性** | 特征之间不完全相关 | 参数估计不稳定 |

## 参数估计

### 最小二乘法（OLS）

最小二乘法通过最小化残差平方和来估计参数：

**目标函数**：

$$J(\boldsymbol{\beta}) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$

**解析解推导**：

展开目标函数：
$$J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

对 $\boldsymbol{\beta}$ 求导并令其为零：
$$\frac{\partial J}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0$$

解得**正规方程**：
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

💡 **几何解释**：OLS 解是目标向量 $\mathbf{y}$ 在由 $\mathbf{X}$ 的列向量张成的子空间上的正交投影。

### 梯度下降法

当数据量大或矩阵不可逆时，使用迭代优化：

**梯度计算**：
$$\nabla J(\boldsymbol{\beta}) = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

**参数更新**：
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla J(\boldsymbol{\beta}^{(t)})$$

其中 $\eta$ 为学习率。

### 三种梯度下降变体

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **批量梯度下降** | 每次使用全部数据 | 小数据集，稳定收敛 |
| **随机梯度下降** | 每次使用一个样本 | 大数据集，快速迭代 |
| **小批量梯度下降** | 每次使用一小批数据 | 平衡效率与稳定性 |

## 模型评估

### 回归评价指标

**均方误差（MSE）**：
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**均方根误差（RMSE）**：
$$RMSE = \sqrt{MSE}$$

**平均绝对误差（MAE）**：
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**决定系数（R²）**：
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

📌 **R² 的解释**：模型解释的方差比例，取值范围 $[0, 1]$，越接近1表示拟合越好。

### 统计检验

**系数显著性检验（t检验）**：
$$t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}$$

**模型整体显著性检验（F检验）**：
$$F = \frac{(SS_{tot} - SS_{res})/p}{SS_{res}/(n-p-1)}$$

## 代码示例

### 示例1：基本线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 2
true_coef, true_intercept = 2.5, 1.0
y = true_coef * X.ravel() + true_intercept + np.random.randn(n_samples) * 0.5

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 输出结果
print(f"真实系数: {true_coef}, 估计系数: {model.coef_[0]:.4f}")
print(f"真实截距: {true_intercept}, 估计截距: {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"R²: {r2_score(y, y_pred):.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='数据点')
plt.plot(X, y_pred, 'r-', linewidth=2, label='回归线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 示例2：多元线性回归与特征分析

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 生成多元数据
np.random.seed(42)
n_samples = 200
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coef = np.array([3.0, -2.0, 1.5, 0.0, 0.5])  # 第4个特征无影响
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（可选，但对解释系数重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = model.predict(X_test_scaled)
print(f"测试集 MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"测试集 R²: {r2_score(y_test, y_pred):.4f}")

# 系数分析
print("\n特征系数分析:")
for i, coef in enumerate(model.coef_):
    print(f"  特征 {i+1}: {coef:.4f} (真实值: {true_coef[i]})")
```

### 示例3：从零实现线性回归

```python
import numpy as np

class LinearRegressionManual:
    """手写线性回归，支持OLS和梯度下降"""
    
    def __init__(self, method='ols', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.method == 'ols':
            # 使用正规方程
            X_b = np.c_[np.ones(n_samples), X]  # 添加偏置列
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
            
        else:  # 梯度下降
            for _ in range(self.n_iterations):
                # 预测
                y_pred = X @ self.weights + self.bias
                
                # 计算损失
                loss = np.mean((y_pred - y) ** 2)
                self.loss_history.append(loss)
                
                # 计算梯度
                dw = (2 / n_samples) * X.T @ (y_pred - y)
                db = (2 / n_samples) * np.sum(y_pred - y)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """计算 R² 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

# 使用示例
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 2.0 + np.random.randn(100) * 0.1

# OLS 方法
model_ols = LinearRegressionManual(method='ols')
model_ols.fit(X, y)
print(f"OLS - R²: {model_ols.score(X, y):.4f}")

# 梯度下降方法
model_gd = LinearRegressionManual(method='gd', learning_rate=0.1, n_iterations=1000)
model_gd.fit(X, y)
print(f"GD - R²: {model_gd.score(X, y):.4f}")
print(f"GD - 最终损失: {model_gd.loss_history[-1]:.6f}")
```

## 常见问题与注意事项

### ⚠️ 多重共线性问题

当特征之间高度相关时，$(\mathbf{X}^T\mathbf{X})$ 接近奇异矩阵，导致参数估计不稳定。

**解决方案**：
- 移除相关性高的特征
- 使用正则化方法（Ridge、Lasso）
- 主成分分析（PCA）降维

### ⚠️ 异常值敏感性

线性回归对异常值敏感，单个异常点可能显著影响回归线。

**解决方案**：
- 使用 RANSAC 等鲁棒方法
- 使用 MAE 代替 MSE 作为损失函数
- 预处理时检测并处理异常值

### ⚠️ 非线性关系

线性回归假设线性关系，无法捕捉非线性模式。

**解决方案**：
- 多项式回归
- 特征变换（如 log、sqrt）
- 使用非线性模型

### ✅ 实践建议

1. **数据预处理**：标准化连续特征，处理缺失值
2. **特征工程**：探索特征与目标的线性关系
3. **残差分析**：检查模型假设是否满足
4. **交叉验证**：评估模型泛化能力
5. **正则化**：当特征多或共线性时使用

## 参考资料

- [最小二乘法的几何解释](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [线性回归假设检验](https://en.wikipedia.org/wiki/Regression_validation)
