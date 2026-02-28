# 梯度提升

梯度提升（Gradient Boosting）是一种强大的 Boosting 集成方法，通过迭代地添加弱学习器来改进模型。与 AdaBoost 不同，梯度提升使用**梯度下降**的方法在**函数空间**中优化任意可微损失函数。

📌 **核心思想**：每一轮训练一个新的模型来拟合前一轮模型的残差（负梯度），逐步逼近最优函数。

## 函数空间优化

### 问题建模

给定训练数据 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，目标是找到函数 $F^*$ 使损失函数最小：

$$
F^*(\mathbf{x}) = \arg\min_F \mathbb{E}_{(\mathbf{x}, y)}[L(y, F(\mathbf{x}))]
$$

传统优化在参数空间进行，而梯度提升在**函数空间**进行优化。

### 梯度下降视角

假设当前模型为 $F_{m-1}(\mathbf{x})$，我们希望找到增量 $h_m(\mathbf{x})$ 使得：

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + h_m(\mathbf{x})
$$

损失函数的变化：

$$
L(y_i, F_m(\mathbf{x}_i)) = L(y_i, F_{m-1}(\mathbf{x}_i) + h_m(\mathbf{x}_i))
$$

使用梯度下降的方向：

$$
h_m(\mathbf{x}_i) \approx -\eta \cdot g_m(\mathbf{x}_i)
$$

其中 $g_m$ 是负梯度：

$$
g_m(\mathbf{x}_i) = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}
$$

## 算法流程

### 完整步骤

```
输入：训练数据 {x_i, y_i}，损失函数 L，迭代次数 M，学习率 η

1. 初始化模型：
   F_0(x) = argmin_γ Σ L(y_i, γ)

2. 对于 m = 1 到 M：
   a) 计算负梯度（伪残差）：
      r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]|_{F=F_{m-1}}
   
   b) 用弱学习器拟合伪残差：
      h_m = argmin_h Σ (r_im - h(x_i))²
   
   c) 线性搜索找最优步长：
      ρ_m = argmin_ρ Σ L(y_i, F_{m-1}(x_i) + ρ·h_m(x_i))
   
   d) 更新模型：
      F_m(x) = F_{m-1}(x) + η·ρ_m·h_m(x)

输出：最终模型 F_M(x)
```

### 常见损失函数及梯度

| 任务 | 损失函数 | 负梯度（伪残差） |
|------|---------|-----------------|
| 回归 | $\frac{1}{2}(y-F)^2$ | $y - F$ |
| 回归 | $\|y-F\|$ | $\text{sign}(y-F)$ |
| 回归 | Huber | 分段函数 |
| 分类 | 对数损失 | $y - \sigma(F)$ |

对于**平方损失**，伪残差恰好是真实残差 $y_i - F_{m-1}(\mathbf{x}_i)$。

## 梯度提升树（GBDT）

当弱学习器选择决策树时，称为梯度提升决策树（GBDT）。

### 决策树拟合

每轮训练一棵回归树，拟合当前的伪残差。树的预测值是叶子节点的平均值。

### 正则化

**收缩（Shrinkage）**：通过学习率控制每棵树的贡献

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
$$

- 较小的学习率需要更多的树
- 通常 $\eta \in [0.01, 0.3]$

**子采样（Subsampling）**：每次迭代随机选择部分样本训练

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}), \quad \text{使用 } S_m \subset D
$$

**特征采样**：每次分裂随机选择部分特征

## 代码示例

### 基础使用

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# 分类任务
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练 GBDT 分类器
gb_clf = GradientBoostingClassifier(
    n_estimators=100,        # 树的数量
    learning_rate=0.1,       # 学习率
    max_depth=3,             # 树的最大深度
    min_samples_split=5,     # 分裂所需最小样本数
    subsample=0.8,           # 子采样比例
    random_state=42
)
gb_clf.fit(X_train, y_train)

# 评估
y_pred = gb_clf.predict(X_test)
print(f"测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### 回归任务

```python
# 回归任务
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               n_informative=8, noise=0.1, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3)

# 训练 GBDT 回归器
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb_reg.fit(X_train_r, y_train_r)

# 评估
y_pred_r = gb_reg.predict(X_test_r)
print(f"均方误差: {mean_squared_error(y_test_r, y_pred_r):.4f}")
```

### 学习曲线分析

```python
# 使用 staged_predict 获取每轮迭代的预测
train_scores = []
test_scores = []

for y_pred_train, y_pred_test in zip(
    gb_clf.staged_predict(X_train), 
    gb_clf.staged_predict(X_test)
):
    train_scores.append(accuracy_score(y_train, y_pred_train))
    test_scores.append(accuracy_score(y_test, y_pred_test))

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, 'b-', label='训练准确率')
plt.plot(range(1, len(test_scores) + 1), test_scores, 'r-', label='测试准确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('GBDT 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 早停法

```python
# 使用早停防止过拟合
gb_early = GradientBoostingClassifier(
    n_estimators=500,          # 设置较大的迭代次数
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.1,   # 验证集比例
    n_iter_no_change=10,       # 连续10轮无改进则停止
    tol=1e-4,                  # 改进阈值
    random_state=42
)
gb_early.fit(X_train, y_train)

print(f"实际迭代次数: {gb_early.n_estimators_}")
print(f"测试准确率: {gb_early.score(X_test, y_test):.4f}")
```

### 特征重要性

```python
import pandas as pd

# 特征重要性
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': gb_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排名：")
print(importance_df.head(10))

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('特征重要性')
plt.title('GBDT 特征重要性排名')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 不同损失函数

```python
# 回归任务使用不同损失函数
losses = ['squared_error', 'absolute_error', 'huber']
results = []

for loss in losses:
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        loss=loss,
        random_state=42
    )
    gb.fit(X_train_r, y_train_r)
    y_pred = gb.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    results.append({'损失函数': loss, 'MSE': mse})

print("不同损失函数比较：")
print(pd.DataFrame(results).to_string(index=False))
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# 网格搜索
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 使用小数据集演示
grid_search.fit(X[:500], y[:500])

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
```

### 从零实现 GBDT

```python
class SimpleGBDT:
    """简化的 GBDT 回归实现"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None
    
    def fit(self, X, y):
        # 初始化：使用均值
        self.initial_pred = np.mean(y)
        F = np.full(len(y), self.initial_pred)
        
        for _ in range(self.n_estimators):
            # 计算残差（平方损失的负梯度）
            residuals = y - F
            
            # 拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 更新预测
            update = tree.predict(X)
            F += self.learning_rate * update
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

from sklearn.tree import DecisionTreeRegressor

# 测试
simple_gb = SimpleGBDT(n_estimators=100, learning_rate=0.1, max_depth=3)
simple_gb.fit(X_train_r, y_train_r)
y_pred_simple = simple_gb.predict(X_test_r)
print(f"\n自定义 GBDT 均方误差: {mean_squared_error(y_test_r, y_pred_simple):.4f}")
```

### 偏依赖图

```python
from sklearn.inspection import PartialDependenceDisplay

# 选择最重要的特征
important_features = importance_df['feature'][:2].tolist()
feature_indices = [int(f.split('_')[1]) for f in important_features]

# 绘制偏依赖图
fig, ax = plt.subplots(figsize=(12, 5))
PartialDependenceDisplay.from_estimator(
    gb_clf, X_train, feature_indices,
    ax=ax, grid_resolution=20
)
plt.suptitle('偏依赖图')
plt.tight_layout()
plt.show()
```

## GBDT 的优势

### 与 AdaBoost 对比

| 特性 | AdaBoost | GBDT |
|------|----------|------|
| 损失函数 | 指数损失 | 任意可微损失 |
| 优化方式 | 重加权样本 | 梯度下降 |
| 异常值 | 敏感 | 较鲁棒 |
| 适用任务 | 主要是分类 | 分类、回归、排序 |

### 与随机森林对比

| 特性 | 随机森林 | GBDT |
|------|---------|------|
| 训练方式 | 并行 | 顺序 |
| 树的构建 | 独立 | 依赖前一棵 |
| 主要目标 | 降低方差 | 降低偏差 |
| 过拟合风险 | 低 | 中等 |
| 参数调优 | 简单 | 较复杂 |

## 优缺点分析

### 优点

✅ **高预测精度**：在多种任务上表现优异  
✅ **灵活的损失函数**：可根据任务选择合适的损失  
✅ **处理混合特征**：无需特征预处理  
✅ **特征重要性**：提供特征评估  
✅ **可解释性**：偏依赖图等工具辅助理解

### 缺点

❌ **训练时间长**：顺序训练，难以并行  
❌ **参数调优复杂**：多个超参数需要调整  
❌ **容易过拟合**：需要正则化和早停  
❌ **内存占用大**：存储多棵树

## 最佳实践

### 参数设置建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `n_estimators` | 100-1000 | 配合早停 |
| `learning_rate` | 0.01-0.1 | 越小需要越多树 |
| `max_depth` | 3-6 | 浅树防过拟合 |
| `subsample` | 0.7-0.9 | 随机性防过拟合 |
| `min_samples_leaf` | 1-10 | 控制叶节点大小 |

### 调参策略

1. **固定学习率，调整树数量**：使用早停找到最优迭代次数
2. **调整树深度**：根据数据复杂度选择
3. **调整正则化**：子采样、特征采样
4. **微调学习率**：减小学习率，增加树数量

---

[上一节：AdaBoost](./adaboost.md) | [下一节：XGBoost](./xgboost.md)
