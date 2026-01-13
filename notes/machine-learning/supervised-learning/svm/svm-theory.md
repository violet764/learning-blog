# 支持向量机（SVM）

## 1. 算法概述

支持向量机是一种强大的监督学习算法，主要用于分类任务，也可用于回归。其核心思想是寻找一个最优超平面，使得不同类别之间的间隔最大化。

### 基本概念

- **支持向量**：距离超平面最近的样本点
- **间隔**：支持向量到超平面的距离
- **最优超平面**：最大化间隔的决策边界

## 2. 线性可分情况

### 2.1 数学建模

对于线性可分数据，寻找超平面：
$$ \mathbf{w}^T\mathbf{x} + b = 0 $$

使得对于所有样本：
$$ y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \dots, n $$

### 2.2 间隔最大化

**间隔计算：**
$$ \text{Margin} = \frac{2}{\|\mathbf{w}\|} $$

**优化问题：**
$$ \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 $$
$$ \text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \dots, n $$

## 3. 线性不可分情况

### 3.1 软间隔SVM

引入松弛变量$\xi_i \geq 0$：
$$ \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i $$
$$ \text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 $$

其中$C > 0$是惩罚参数。

### 3.2 对偶问题

**拉格朗日函数：**
$$ L(\mathbf{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i] - \sum_{i=1}^n \mu_i\xi_i $$

**对偶问题：**
$$ \max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j $$
$$ \text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0 $$

## 4. 核技巧与非线性SVM

### 4.1 核函数

通过特征映射$\phi(\mathbf{x})$将数据映射到高维空间：
$$ K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle $$

**常用核函数：**
- 线性核：$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$
- 多项式核：$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma\mathbf{x}_i^T\mathbf{x}_j + r)^d$
- RBF核：$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$
- Sigmoid核：$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma\mathbf{x}_i^T\mathbf{x}_j + r)$

### 4.2 非线性SVM对偶问题

$$ \max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) $$

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification, make_circles
from sklearn.preprocessing import StandardScaler

# 生成非线性可分数据
X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 线性SVM（在非线性数据上效果差）
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)

# 非线性SVM（使用RBF核）
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)

# 预测
y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)

# 模型评估
print("线性SVM分类报告:")
print(classification_report(y_test, y_pred_linear))

print("\nRBF核SVM分类报告:")
print(classification_report(y_test, y_pred_rbf))

# 超参数调优
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()

plot_decision_boundary(linear_svm, X_train, y_train, "线性SVM决策边界")
plot_decision_boundary(rbf_svm, X_train, y_train, "RBF核SVM决策边界")
```

## 6. 支持向量回归（SVR）

### 6.1 数学模型

SVR寻找一个函数$f(\mathbf{x}) = \mathbf{w}^T\phi(\mathbf{x}) + b$，使得：
$$ |y_i - f(\mathbf{x}_i)| \leq \epsilon + \xi_i $$

**优化问题：**
$$ \min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n (\xi_i + \xi_i^*) $$
$$ \text{s.t. } \begin{cases}
y_i - f(\mathbf{x}_i) \leq \epsilon + \xi_i \\
f(\mathbf{x}_i) - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases} $$

## 7. 数学推导

### 7.1 间隔最大化推导

对于线性可分情况，间隔为：
$$ \text{Margin} = \min_{i} \frac{|\mathbf{w}^T\mathbf{x}_i + b|}{\|\mathbf{w}\|} $$

由于$y_i \in \{-1, 1\}$，且对于支持向量有$y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$，因此间隔为$\frac{1}{\|\mathbf{w}\|}$。

最大化间隔等价于最小化$\|\mathbf{w}\|$，为方便优化，改为最小化$\frac{1}{2}\|\mathbf{w}\|^2$。

### 7.2 对偶问题推导

构建拉格朗日函数：
$$ L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] $$

对$\mathbf{w}$和$b$求偏导并令为零：
$$ \frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 \Rightarrow \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i $$
$$ \frac{\partial L}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 $$

代入拉格朗日函数得到对偶问题。

## 8. 应用场景

1. **文本分类**：新闻分类、情感分析
2. **图像识别**：手写数字识别、人脸检测
3. **生物信息学**：基因表达数据分析
4. **金融风控**：信用评分、欺诈检测

## 9. 优缺点分析

### 优点：
- 在高维空间中表现优秀
- 对非线性问题有很好的处理能力
- 泛化能力强（基于结构风险最小化）
- 对异常值相对鲁棒

### 缺点：
- 对大规模数据集训练较慢
- 对参数$C$和核函数选择敏感
- 难以解释（特别是使用复杂核函数时）
- 需要特征标准化

## 10. 实践建议

1. **数据预处理**：标准化特征，处理缺失值
2. **核函数选择**：从简单核开始（线性→RBF）
3. **参数调优**：使用网格搜索优化$C$和$\gamma$
4. **模型评估**：使用交叉验证评估性能
5. **计算优化**：对于大数据集考虑使用线性SVM或近似方法

---

[下一节：决策树与集成方法](./decision-trees-ensemble.md)