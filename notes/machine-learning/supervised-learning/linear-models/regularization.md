# 正则化

正则化是防止模型过拟合的核心技术。通过在损失函数中添加惩罚项，限制模型复杂度，从而提升泛化能力。

📌 **核心思想**：在最小化训练误差的同时，控制模型复杂度，找到"最简"的有效模型。

## 基本概念

### 过拟合问题

当模型过于复杂时，可能会"记住"训练数据中的噪声，导致在测试集上表现不佳。

$$\text{泛化误差} = \text{偏差}^2 + \text{方差} + \text{噪声}$$

- **高偏差（欠拟合）**：模型太简单，无法捕捉数据规律
- **高方差（过拟合）**：模型太复杂，拟合了噪声

### 正则化框架

正则化后的目标函数：

$$J(\boldsymbol{\theta}) = \underbrace{L(\boldsymbol{\theta})}_{\text{损失项}} + \underbrace{\lambda R(\boldsymbol{\theta})}_{\text{正则项}}$$

其中：
- $L(\boldsymbol{\theta})$：原始损失函数（如 MSE、交叉熵）
- $R(\boldsymbol{\theta})$：正则化项，衡量模型复杂度
- $\lambda$：正则化强度，控制惩罚程度

## L2 正则化（Ridge）

### 数学模型

对于线性回归，L2 正则化的目标函数：

$$J(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$$

其中 $\|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^{p}\beta_j^2$ 是 L2 范数的平方。

### 解析解

对 $\boldsymbol{\beta}$ 求导并令其为零：

$$\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

📌 **关键特性**：添加 $\lambda\mathbf{I}$ 使矩阵总是可逆，解决了多重共线性问题！

### 几何解释

L2 正则化等价于在参数空间施加球形约束：

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 \quad \text{s.t.} \quad \|\boldsymbol{\beta}\|_2^2 \leq t$$

最优解是椭圆等高线与圆的切点，不会落在坐标轴上（系数不会精确为 0）。

## L1 正则化（Lasso）

### 数学模型

$$J(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$$

其中 $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^{p}|\beta_j|$ 是 L1 范数。

### 稀疏性解释

L1 正则化产生**稀疏解**——部分系数精确为零。

**几何理解**：L1 约束区域是菱形（超立方体），最优解更容易落在顶点上（坐标轴上），使某些系数为零。

💡 **特征选择**：Lasso 自动进行特征选择，保留重要特征，剔除无关特征。

### 求解方法

由于 L1 范数在零点不可导，需要特殊优化算法：

- **坐标下降法**：逐个坐标优化，适用于 Lasso
- **近端梯度法（Proximal Gradient）**：处理非光滑正则项
- **最小角回归（LARS）**：高效计算整个正则化路径

## Elastic Net

### 混合正则化

Elastic Net 结合 L1 和 L2 的优势：

$$J(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\left(\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2\right)$$

其中 $\alpha \in [0, 1]$ 控制 L1 和 L2 的混合比例：
- $\alpha = 1$：纯 Lasso
- $\alpha = 0$：纯 Ridge
- $0 < \alpha < 1$：Elastic Net

### 优势

| 特性 | Lasso | Ridge | Elastic Net |
|------|-------|-------|-------------|
| **特征选择** | ✅ | ❌ | ✅ |
| **处理共线性** | ❌（随机选一个） | ✅ | ✅（选择组） |
| **稳定性** | 较低 | 高 | 高 |

## 贝叶斯解释

### L2 正则化 ≈ 高斯先验

从贝叶斯角度，L2 正则化等价于参数的高斯先验：

$$p(\boldsymbol{\beta}) = \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I}) = \prod_{j=1}^{p} \frac{1}{\sqrt{2\pi}\tau}\exp\left(-\frac{\beta_j^2}{2\tau^2}\right)$$

最大后验估计（MAP）：
$$\hat{\boldsymbol{\beta}}_{MAP} = \arg\max_{\boldsymbol{\beta}} p(\boldsymbol{\beta}|\mathbf{X},\mathbf{y}) \propto p(\mathbf{y}|\mathbf{X},\boldsymbol{\beta})p(\boldsymbol{\beta})$$

对应正则化参数 $\lambda = \sigma^2/\tau^2$。

### L1 正则化 ≈ 拉普拉斯先验

L1 正则化对应拉普拉斯先验：

$$p(\beta_j) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)$$

拉普拉斯分布在零点有尖峰，更倾向于产生零值（稀疏解）。

## 正则化参数选择

### 交叉验证

使用 K 折交叉验证选择最优 $\lambda$：

$$\hat{\lambda} = \arg\min_{\lambda} \frac{1}{K}\sum_{k=1}^{K} E_{valid}^{(k)}(\lambda)$$

### 信息准则

**AIC（Akaike 信息准则）**：
$$AIC = 2k - 2\ln(L)$$

**BIC（贝叶斯信息准则）**：
$$BIC = k\ln(n) - 2\ln(L)$$

其中 $k$ 是有效参数个数，$L$ 是似然函数值。

## 代码示例

### 示例1：正则化方法对比

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# 生成数据：20个特征，仅5个有用
np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=100, n_features=20, n_informative=5,
    noise=10, coef=True, random_state=42
)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"真实非零系数数量: {np.sum(np.abs(true_coef) > 0.01)}")

# 普通线性回归（无正则化）
lr = LinearRegression()
lr.fit(X_scaled, y)
lr_nonzero = np.sum(np.abs(lr.coef_) > 0.01)
print(f"普通回归非零系数: {lr_nonzero}")

# 不同正则化方法
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_scaled, y)
    nonzero = np.sum(np.abs(model.coef_) > 0.01)
    print(f"{name} 非零系数: {nonzero}")
```

### 示例2：正则化路径

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# 生成数据
np.random.seed(42)
X, y, _ = make_regression(n_samples=100, n_features=10, n_informative=3, noise=5, random_state=42)
X = StandardScaler().fit_transform(X)

# 计算正则化路径
alphas_lasso, coefs_lasso, _ = lasso_path(X, y)
alphas_enet, coefs_enet, _ = enet_path(X, y, l1_ratio=0.7)

# 绘制系数路径
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lasso 路径
ax1 = axes[0]
for i in range(coefs_lasso.shape[0]):
    ax1.plot(np.log10(alphas_lasso), coefs_lasso[i, :], alpha=0.7)
ax1.set_xlabel('log10(λ)')
ax1.set_ylabel('系数值')
ax1.set_title('Lasso 正则化路径')
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()  # λ 从大到小

# Elastic Net 路径
ax2 = axes[1]
for i in range(coefs_enet.shape[0]):
    ax2.plot(np.log10(alphas_enet), coefs_enet[i, :], alpha=0.7)
ax2.set_xlabel('log10(λ)')
ax2.set_ylabel('系数值')
ax2.set_title('Elastic Net 正则化路径')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
```

### 示例3：交叉验证选择最优参数

```python
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成数据
np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=30, n_informative=5, noise=10, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge CV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Ridge 最优 alpha: {ridge_cv.alpha_:.4f}")
print(f"Ridge 测试 R²: {ridge_cv.score(X_test, y_test):.4f}")

# Lasso CV
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"\nLasso 最优 alpha: {lasso_cv.alpha_:.4f}")
print(f"Lasso 测试 R²: {lasso_cv.score(X_test, y_test):.4f}")
print(f"Lasso 选择的特征数: {np.sum(np.abs(lasso_cv.coef_) > 0.01)}")

# Elastic Net CV
enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], cv=5, max_iter=10000)
enet_cv.fit(X_train, y_train)
print(f"\nElastic Net 最优 alpha: {enet_cv.alpha_:.4f}")
print(f"Elastic Net 最优 l1_ratio: {enet_cv.l1_ratio_:.2f}")
print(f"Elastic Net 测试 R²: {enet_cv.score(X_test, y_test):.4f}")
```

### 示例4：逻辑回归中的正则化

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# 生成分类数据
X, y = make_classification(n_samples=200, n_features=50, n_informative=5, 
                          n_redundant=10, random_state=42)

# 不同正则化设置
configs = [
    ('L2, C=1.0', LogisticRegression(penalty='l2', C=1.0, max_iter=1000)),
    ('L2, C=0.1', LogisticRegression(penalty='l2', C=0.1, max_iter=1000)),
    ('L1, C=1.0', LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)),
    ('L1, C=0.1', LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=1000)),
]

print("逻辑回归正则化效果对比:")
print("-" * 50)

for name, model in configs:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    model.fit(X, y)
    n_features = np.sum(np.abs(model.coef_) > 0.001)
    print(f"{name}: 准确率={scores.mean():.3f}±{scores.std():.3f}, 使用特征数={n_features}")
```

## 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 特征数 >> 样本数 | Lasso / Elastic Net | 特征选择，防止过拟合 |
| 特征高度相关 | Ridge / Elastic Net | 系数稳定，不随机选择 |
| 需要特征选择 | Lasso | 产生稀疏解 |
| 数值稳定性优先 | Ridge | 始终有解 |
| 不确定 | Elastic Net | 兼顾两者优点 |

## 常见问题与注意事项

### ⚠️ 标准化重要性

正则化对特征尺度敏感！使用正则化前必须标准化特征。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 推荐：使用 Pipeline 确保标准化
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

### ⚠️ 截距项

通常**不对截距项**进行正则化。sklearn 默认不惩罚截距。

### ⚠️ 参数解释

正则化后的系数**有偏估计**，解释时需注意：
- 系数大小反映相对重要性，但被压缩
- Lasso 系数为零的特征不一定完全无用

### ✅ 实践建议

1. **先尝试 Ridge**：稳定、计算快
2. **特征选择用 Lasso**：自动选择特征
3. **不确定用 Elastic Net**：通过交叉验证选择最优混合比例
4. **必须标准化**：确保特征在同一尺度
5. **使用交叉验证**：sklearn 提供 `*CV` 类自动调参
