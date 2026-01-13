# 正则化理论

## 1. 正则化基本概念

### 1.1 过拟合问题

在机器学习中，当模型过于复杂时，可能会过度拟合训练数据中的噪声，导致在测试集上表现不佳。这种现象称为过拟合。

**数学描述：**
设训练误差为$E_{train}$，测试误差为$E_{test}$，过拟合表现为：
$$ E_{train} \ll E_{test} $$

### 1.2 正则化原理

正则化通过在损失函数中添加惩罚项来限制模型复杂度：
$$ J(\boldsymbol{\theta}) = L(\boldsymbol{\theta}) + \lambda R(\boldsymbol{\theta}) $$
其中：
- $L(\boldsymbol{\theta})$：原始损失函数
- $R(\boldsymbol{\theta})$：正则化项
- $\lambda$：正则化参数，控制惩罚强度

## 2. L2正则化（岭回归）

### 2.1 数学模型

对于线性回归问题，L2正则化的目标函数为：
$$ J(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2 $$

### 2.2 解析解

**矩阵形式：**
$$ J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2 $$

对$\boldsymbol{\beta}$求导并令导数为零：
$$ \frac{\partial J}{\partial \boldsymbol{\beta}} = -\frac{1}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + \lambda\boldsymbol{\beta} = 0 $$

解得：
$$ \hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + n\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} $$

### 2.3 几何解释

L2正则化等价于在参数空间中对参数向量施加球形约束：
$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 \quad \text{s.t.} \quad \|\boldsymbol{\beta}\|_2^2 \leq t $$

## 3. L1正则化（Lasso回归）

### 3.1 数学模型

L1正则化的目标函数：
$$ J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 $$

### 3.2 特征选择性质

L1正则化具有特征选择能力，可以将不重要的特征的系数压缩为0。

**几何解释：** L1正则化在参数空间中施加菱形约束，使得最优解更容易落在坐标轴上。

### 3.3 求解方法

由于L1正则化不可导，需要使用特殊优化算法：
- 坐标下降法
- 近端梯度法
- 最小角回归（LARS）

## 4. Elastic Net

### 4.1 混合正则化

Elastic Net结合了L1和L2正则化：
$$ J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2 $$

### 4.2 优势

- 继承了L1正则化的特征选择能力
- 解决了L1正则化在高维数据中的局限性
- 对于高度相关的特征，倾向于选择所有相关特征

## 5. 贝叶斯解释

### 5.1 L2正则化的贝叶斯解释

从贝叶斯角度看，L2正则化等价于给参数施加高斯先验：
$$ p(\boldsymbol{\beta}) = \mathcal{N}(\boldsymbol{\beta}|\mathbf{0}, \frac{1}{\lambda}\mathbf{I}) $$

后验分布：
$$ p(\boldsymbol{\beta}|\mathbf{X},\mathbf{y}) \propto p(\mathbf{y}|\mathbf{X},\boldsymbol{\beta})p(\boldsymbol{\beta}) $$

最大后验估计（MAP）即为岭回归的解。

### 5.2 L1正则化的贝叶斯解释

L1正则化等价于拉普拉斯先验：
$$ p(\beta_j) = \frac{\lambda}{2} e^{-\lambda|\beta_j|} $$

## 6. 正则化参数选择

### 6.1 交叉验证

使用k折交叉验证选择最优的$\lambda$值：
$$ \hat{\lambda} = \arg\min_{\lambda} \frac{1}{k} \sum_{i=1}^k E_{test}^{(i)}(\lambda) $$

### 6.2 信息准则

**AIC（Akaike信息准则）：**
$$ AIC = 2k - 2\ln(L) $$

**BIC（贝叶斯信息准则）：**
$$ BIC = k\ln(n) - 2\ln(L) $$

其中k是参数个数，L是似然函数值。

## 7. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 生成示例数据
print("=== 正则化方法比较 ===")
X, y, true_coef = make_regression(n_samples=100, n_features=20, n_informative=5, 
                                 noise=0.1, coef=True, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("真实系数中有", np.sum(true_coef != 0), "个非零系数")
print("真实系数中有", np.sum(true_coef == 0), "个零系数")

# 1. 不同正则化方法比较
print("\n=== 不同正则化方法比较 ===")

# 定义正则化参数范围
alphas = np.logspace(-4, 4, 100)

ridge_scores = []
lasso_scores = []
elasticnet_scores = []

for alpha in alphas:
    # 岭回归
    ridge = Ridge(alpha=alpha)
    ridge_scores.append(-cross_val_score(ridge, X_scaled, y, cv=5, 
                                       scoring='neg_mean_squared_error').mean())
    
    # Lasso回归
    lasso = Lasso(alpha=alpha)
    lasso_scores.append(-cross_val_score(lasso, X_scaled, y, cv=5, 
                                       scoring='neg_mean_squared_error').mean())
    
    # Elastic Net
    elasticnet = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elasticnet_scores.append(-cross_val_score(elasticnet, X_scaled, y, cv=5, 
                                            scoring='neg_mean_squared_error').mean())

# 绘制正则化路径
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogx(alphas, ridge_scores)
plt.xlabel('正则化参数 λ')
plt.ylabel('MSE')
plt.title('岭回归正则化路径')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.semilogx(alphas, lasso_scores)
plt.xlabel('正则化参数 λ')
plt.ylabel('MSE')
plt.title('Lasso回归正则化路径')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.semilogx(alphas, elasticnet_scores)
plt.xlabel('正则化参数 λ')
plt.ylabel('MSE')
plt.title('Elastic Net正则化路径')
plt.grid(True)

# 2. 系数路径分析
print("\n=== 系数路径分析 ===")

# 使用较大的alpha范围来观察系数压缩
alphas_coef = np.logspace(-2, 2, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas_coef:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

plt.subplot(2, 2, 4)
for i in range(ridge_coefs.shape[1]):
    plt.semilogx(alphas_coef, ridge_coefs[:, i], alpha=0.7)
plt.xlabel('正则化参数 λ')
plt.ylabel('系数值')
plt.title('岭回归系数路径')
plt.grid(True)

plt.tight_layout()
plt.show()

# 单独绘制Lasso系数路径
plt.figure(figsize=(10, 6))
for i in range(lasso_coefs.shape[1]):
    plt.semilogx(alphas_coef, lasso_coefs[:, i], alpha=0.7)
plt.xlabel('正则化参数 λ')
plt.ylabel('系数值')
plt.title('Lasso回归系数路径')
plt.grid(True)
plt.show()

# 3. 最优参数选择
print("\n=== 最优参数选择 ===")

# 网格搜索寻找最优参数
param_grid = {'alpha': np.logspace(-4, 2, 50)}

# 岭回归
ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_scaled, y)
print(f"岭回归最优参数: alpha={ridge_grid.best_params_['alpha']:.6f}")
print(f"岭回归最佳分数: {-ridge_grid.best_score_:.6f}")

# Lasso回归
lasso_grid = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_scaled, y)
print(f"Lasso回归最优参数: alpha={lasso_grid.best_params_['alpha']:.6f}")
print(f"Lasso回归最佳分数: {-lasso_grid.best_score_:.6f}")

# Elastic Net参数网格
param_grid_en = {
    'alpha': np.logspace(-4, 2, 20),
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
}

elasticnet_grid = GridSearchCV(ElasticNet(), param_grid_en, cv=5, 
                              scoring='neg_mean_squared_error')
elasticnet_grid.fit(X_scaled, y)
print(f"Elastic Net最优参数: alpha={elasticnet_grid.best_params_['alpha']:.6f}, "
      f"l1_ratio={elasticnet_grid.best_params_['l1_ratio']:.2f}")
print(f"Elastic Net最佳分数: {-elasticnet_grid.best_score_:.6f}")

# 4. 特征选择效果比较
print("\n=== 特征选择效果比较 ===")

# 使用最优参数训练模型
best_ridge = Ridge(alpha=ridge_grid.best_params_['alpha'])
best_ridge.fit(X_scaled, y)

best_lasso = Lasso(alpha=lasso_grid.best_params_['alpha'])
best_lasso.fit(X_scaled, y)

best_elasticnet = ElasticNet(alpha=elasticnet_grid.best_params_['alpha'],
                            l1_ratio=elasticnet_grid.best_params_['l1_ratio'])
best_elasticnet.fit(X_scaled, y)

# 统计非零系数数量
ridge_nonzero = np.sum(np.abs(best_ridge.coef_) > 1e-6)
lasso_nonzero = np.sum(np.abs(best_lasso.coef_) > 1e-6)
elasticnet_nonzero = np.sum(np.abs(best_elasticnet.coef_) > 1e-6)

print(f"岭回归非零系数数量: {ridge_nonzero}/{len(true_coef)}")
print(f"Lasso回归非零系数数量: {lasso_nonzero}/{len(true_coef)}")
print(f"Elastic Net非零系数数量: {elasticnet_nonzero}/{len(true_coef)}")
print(f"真实非零系数数量: {np.sum(true_coef != 0)}/{len(true_coef)}")

# 5. 系数估计准确性比较
print("\n=== 系数估计准确性比较 ===")

# 计算系数估计的均方误差
ridge_coef_error = np.mean((best_ridge.coef_ - true_coef)**2)
lasso_coef_error = np.mean((best_lasso.coef_ - true_coef)**2)
elasticnet_coef_error = np.mean((best_elasticnet.coef_ - true_coef)**2)

print(f"岭回归系数估计MSE: {ridge_coef_error:.6f}")
print(f"Lasso回归系数估计MSE: {lasso_coef_error:.6f}")
print(f"Elastic Net系数估计MSE: {elasticnet_coef_error:.6f}")

# 6. 可视化系数估计结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(range(len(true_coef)), true_coef, alpha=0.7, label='真实系数')
plt.scatter(range(len(true_coef)), best_ridge.coef_, alpha=0.7, label='岭回归估计')
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('岭回归系数估计')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(range(len(true_coef)), true_coef, alpha=0.7, label='真实系数')
plt.scatter(range(len(true_coef)), best_lasso.coef_, alpha=0.7, label='Lasso估计')
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('Lasso回归系数估计')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(range(len(true_coef)), true_coef, alpha=0.7, label='真实系数')
plt.scatter(range(len(true_coef)), best_elasticnet.coef_, alpha=0.7, label='Elastic Net估计')
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('Elastic Net系数估计')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 正则化在逻辑回归中的应用
print("\n=== 逻辑回归中的正则化 ===")
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成分类数据
X_clf, y_clf = make_classification(n_samples=100, n_features=20, n_informative=5,
                                  n_redundant=10, random_state=42)

# 不同正则化方法的逻辑回归
logistic_l2 = LogisticRegression(penalty='l2', C=1.0, random_state=42)
logistic_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
logistic_elastic = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, 
                                     solver='saga', random_state=42)

models = [logistic_l2, logistic_l1, logistic_elastic]
model_names = ['L2正则化', 'L1正则化', 'Elastic Net']

for name, model in zip(model_names, models):
    scores = cross_val_score(model, X_clf, y_clf, cv=5, scoring='accuracy')
    print(f"{name} 平均准确率: {scores.mean():.3f} ± {scores.std():.3f}")

# 8. 正则化参数的影响分析
print("\n=== 正则化参数对模型复杂度的影响 ===")

# 分析不同C值（正则化强度的倒数）对模型的影响
C_values = np.logspace(-3, 3, 50)
accuracy_scores = []
nonzero_coefs = []

for C in C_values:
    logistic = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
    scores = cross_val_score(logistic, X_clf, y_clf, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())
    
    # 拟合模型并统计非零系数
    logistic.fit(X_clf, y_clf)
    nonzero_coefs.append(np.sum(np.abs(logistic.coef_) > 1e-6))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogx(C_values, accuracy_scores)
plt.xlabel('C值 (1/λ)')
plt.ylabel('交叉验证准确率')
plt.title('正则化强度对准确率的影响')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(C_values, nonzero_coefs)
plt.xlabel('C值 (1/λ)')
plt.ylabel('非零系数数量')
plt.title('正则化强度对模型复杂度的影响')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 8. 正则化理论深入

### 8.1 泛化误差分析

根据统计学习理论，泛化误差可以分解为：
$$ \text{泛化误差} = \text{偏差}^2 + \text{方差} + \text{噪声} $$

正则化通过增加偏差来减少方差，从而优化泛化性能。

### 8.2 稳定性理论

**定义：** 如果一个学习算法在训练数据的小扰动下产生的假设变化不大，则称该算法是稳定的。

**定理：** 正则化算法通常具有更好的稳定性。

### 8.3 正则化路径的数学性质

**单调性：** 随着λ增大，系数向0收缩。
**分段线性性：** Lasso的系数路径是分段线性的。

## 9. 实践建议

### 9.1 方法选择

- **特征数量远大于样本数量**：优先考虑L1或Elastic Net
- **特征高度相关**：使用L2或Elastic Net
- **需要特征选择**：使用L1正则化
- **数值稳定性重要**：使用L2正则化

### 9.2 参数调优

- 使用交叉验证选择最优λ值
- 对于Elastic Net，需要同时优化λ和l1_ratio
- 考虑使用贝叶斯优化等高级调参方法

### 9.3 模型解释

- L1正则化产生的稀疏模型更容易解释
- 可以通过系数大小评估特征重要性
- 注意正则化对系数估计的压缩效应

---

[下一节：支持向量机理论](../svm/svm-theory.md)