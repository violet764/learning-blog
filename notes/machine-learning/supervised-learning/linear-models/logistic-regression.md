# 逻辑回归

逻辑回归（Logistic Regression）是解决**二分类问题**的经典算法。尽管名字中带有"回归"，但它实际上是一种分类算法。通过 Sigmoid 函数将线性组合映射到 [0, 1] 区间，输出可以解释为概率。

📌 **核心思想**：建立特征与目标类别之间的线性关系，但通过 Sigmoid 函数将输出转化为概率值。

## 基本概念

### 数学模型

**Sigmoid 函数**：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Sigmoid 函数将任意实数映射到 $(0, 1)$ 区间，具有优美的 S 形曲线：

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, 'b-', linewidth=2, label='σ(z)')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid 函数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**逻辑回归模型**：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{x}^T\boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}}$$

$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = \frac{e^{-\mathbf{x}^T\boldsymbol{\beta}}}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}}$$

### 对数几率解释

**几率（Odds）**：事件发生与不发生的概率比

$$odds = \frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = e^{\mathbf{x}^T\boldsymbol{\beta}}$$

**对数几率（Log-odds / Logit）**：

$$\log(odds) = \mathbf{x}^T\boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

💡 **关键洞察**：逻辑回归假设特征与**对数几率**呈线性关系，系数 $\beta_j$ 表示特征 $x_j$ 每增加一个单位，对数几率的变化量。

### 决策边界

分类决策基于概率阈值（默认 0.5）：

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{if } P(y=1|\mathbf{x}) < 0.5 \end{cases}$$

决策边界 $\mathbf{x}^T\boldsymbol{\beta} = 0$ 是一个超平面（线性边界）。

## 参数估计

### 最大似然估计

逻辑回归通过**最大似然估计**求解参数。

**似然函数**：

$$L(\boldsymbol{\beta}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i) = \prod_{i=1}^{n} \sigma(\mathbf{x}_i^T\boldsymbol{\beta})^{y_i} \cdot [1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta})]^{1-y_i}$$

**对数似然函数**：

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \ln\sigma(\mathbf{x}_i^T\boldsymbol{\beta}) + (1-y_i)\ln(1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta})) \right]$$

### 交叉熵损失

最大化对数似然等价于最小化交叉熵损失：

$$J(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \ln\hat{y}_i + (1-y_i)\ln(1-\hat{y}_i) \right]$$

其中 $\hat{y}_i = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$。

### 梯度推导

令 $z_i = \mathbf{x}_i^T\boldsymbol{\beta}$，$\hat{y}_i = \sigma(z_i)$

**Sigmoid 导数**：
$$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

**损失对参数的梯度**：

$$\frac{\partial J}{\partial \beta_j} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_{ij}$$

**向量形式**：
$$\nabla J(\boldsymbol{\beta}) = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$$

📌 **注意**：逻辑回归没有解析解（因为损失函数非凸），必须使用迭代优化方法。

### 优化方法

| 方法 | 特点 |
|------|------|
| **梯度下降** | 简单直接，但收敛慢 |
| **牛顿法** | 利用二阶导数，收敛快但计算量大 |
| **BFGS/L-BFGS** | 拟牛顿法，平衡效率与稳定性 |
| **坐标下降** | 适用于 L1 正则化 |

## 多分类扩展

### One-vs-Rest (OvR)

对于 $K$ 类问题，训练 $K$ 个二分类器：
- 第 $k$ 个分类器：类别 $k$ vs 其他所有类别
- 预测时选择概率最高的类别

### Softmax 回归（多项逻辑回归）

**概率模型**：

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{x}^T\boldsymbol{\beta}_k}}{\sum_{j=1}^{K} e^{\mathbf{x}^T\boldsymbol{\beta}_j}}$$

**交叉熵损失**：

$$J(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbf{1}[y_i=k] \ln P(y_i=k|\mathbf{x}_i)$$

## 模型评估

### 分类指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **准确率** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 分类正确的比例 |
| **精确率** | $\frac{TP}{TP+FP}$ | 预测为正中真正为正的比例 |
| **召回率** | $\frac{TP}{TP+FN}$ | 真正为正中被预测为正的比例 |
| **F1分数** | $\frac{2 \cdot P \cdot R}{P + R}$ | 精确率与召回率的调和平均 |

### ROC 曲线与 AUC

**ROC 曲线**：以假阳性率（FPR）为横轴，真阳性率（TPR）为纵轴绘制的曲线

**AUC**：ROC 曲线下面积，衡量模型区分正负类的能力
- AUC = 0.5：随机猜测
- AUC = 1.0：完美分类器

## 代码示例

### 示例1：基本二分类

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# 生成二分类数据
X, y = make_classification(
    n_samples=1000, n_features=4, n_redundant=0, 
    n_informative=4, random_state=42
)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 正类概率

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 查看系数
print("\n模型系数:", model.coef_)
print("截距项:", model.intercept_)
```

### 示例2：ROC 曲线绘制

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 获取预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('ROC 曲线')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC: {roc_auc:.4f}")
```

### 示例3：从零实现逻辑回归

```python
import numpy as np

class LogisticRegressionManual:
    """手写逻辑回归，支持梯度下降优化"""
    
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid 函数，防止数值溢出"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_loss(self, y, y_pred):
        """计算交叉熵损失"""
        epsilon = 1e-15  # 防止 log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # 记录损失
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# 使用示例
np.random.seed(42)
X = np.random.randn(200, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = LogisticRegressionManual(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
print(f"训练准确率: {model.score(X, y):.4f}")
print(f"最终损失: {model.loss_history[-1]:.4f}")
```

### 示例4：多分类（Softmax）

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载鸢尾花数据集（3分类）
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 Softmax 回归（multinomial）
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## 常见问题与注意事项

### ⚠️ 线性可分问题

当数据完全线性可分时，参数估计可能不收敛（系数趋向无穷大）。

**解决方案**：
- 添加正则化
- 使用更小的学习率
- 检查数据是否存在完全分离

### ⚠️ 特征缩放

逻辑回归对特征尺度敏感，建议标准化特征。

### ⚠️ 类别不平衡

当正负样本比例失衡时，模型可能偏向多数类。

**解决方案**：
- 调整分类阈值
- 使用 `class_weight='balanced'`
- 过采样/欠采样技术

### ✅ 实践建议

1. **特征工程**：确保特征与对数几率有线性关系
2. **正则化**：默认使用 L2 正则化防止过拟合
3. **阈值选择**：根据业务需求调整分类阈值
4. **概率校准**：如果需要准确的概率，考虑使用 CalibratedClassifierCV

## 与线性回归的对比

| 方面 | 线性回归 | 逻辑回归 |
|------|----------|----------|
| **任务类型** | 回归 | 分类 |
| **输出** | 连续值 | 概率（0-1） |
| **激活函数** | 恒等函数 | Sigmoid |
| **损失函数** | 均方误差 | 交叉熵 |
| **优化方法** | 解析解/梯度下降 | 迭代优化 |
| **假设** | 线性关系 | 线性对数几率 |
