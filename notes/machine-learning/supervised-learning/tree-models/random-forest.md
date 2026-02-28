# 随机森林

随机森林（Random Forest）是一种基于 Bagging 思想的集成学习算法，通过构建多棵决策树并聚合它们的预测结果来提升模型的泛化能力。它是机器学习中最常用、最强大的算法之一。

📌 **核心思想**：通过"自助采样 + 随机特征选择"构建多棵相互独立的决策树，最后投票或平均得到最终预测。

## Bagging 原理

### 自助采样（Bootstrap）

从原始数据集 $D$ 中有放回地随机抽取 $n$ 个样本，形成新的训练集 $D'$。

**重要性质**：
- 每次采样约有 **63.2%** 的样本被选中
- 剩余约 **36.8%** 的样本称为"包外数据"（Out-of-Bag, OOB）

$$
\lim_{n \to \infty} \left(1 - \frac{1}{n}\right)^n = \frac{1}{e} \approx 0.368
$$

### Bagging 策略

Bagging（Bootstrap Aggregating）的基本流程：

1. 从原始数据集进行 $B$ 次自助采样
2. 对每个采样集训练一个基学习器
3. 对所有基学习器的预测进行聚合

**分类问题**：投票
$$
\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_B(\mathbf{x})\}
$$

**回归问题**：平均
$$
\hat{y} = \frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x})
$$

## 随机森林算法

### 算法流程

随机森林在 Bagging 的基础上增加了**特征随机性**：

```
输入：训练数据 D，树的数量 B，特征子集大小 m
输出：随机森林模型

For b = 1 to B:
    1. 从 D 中有放回采样得到 D_b
    2. 构建决策树 T_b:
       For 每个节点:
           a. 从全部 p 个特征中随机选择 m 个特征
           b. 在这 m 个特征中选择最优分裂
           c. 分裂节点
    3. 保存树 T_b

预测：
    分类：多数投票
    回归：平均值
```

### 特征子集大小

通常推荐的 $m$ 值：

| 任务类型 | 推荐值 | 说明 |
|---------|--------|------|
| 分类 | $m = \sqrt{p}$ | $p$ 为总特征数 |
| 回归 | $m = p/3$ | 或 $m = \log_2 p$ |

### 为什么随机森林有效？

**降低方差**：设单棵树的方差为 $\sigma^2$，树之间的相关系数为 $\rho$，则随机森林的方差为：

$$
\text{Var}_{RF} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

- **$\rho$ 越小越好**：特征随机性降低了树之间的相关性
- **$B$ 越大越好**：增加树数量可以降低第二项

## 特征重要性

随机森林可以自然地提供特征重要性评估。

### 基于基尼不纯度的重要性

计算每个特征在所有树中带来的不纯度下降的平均值：

$$
\text{Importance}_j = \frac{1}{B}\sum_{b=1}^B \sum_{t \in T_b} \Delta\text{Gini}(t, j)
$$

### 排列重要性（Permutation Importance）

通过打乱特征值观察性能下降程度：

1. 计算模型在原始数据上的性能
2. 对于每个特征，随机打乱该特征的值
3. 重新计算性能，记录下降幅度
4. 下降越多，特征越重要

```python
from sklearn.inspection import permutation_importance

# 计算排列重要性
result = permutation_importance(rf, X_test, y_test, n_repeats=10)
```

## 包外误差

利用 OOB 数据可以在不使用额外验证集的情况下评估模型：

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB 误差估计: {rf.oob_score_:.4f}")
```

## 代码示例

### 基础使用

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# 分类任务
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林
rf_clf = RandomForestClassifier(
    n_estimators=100,        # 树的数量
    max_depth=10,            # 最大深度
    min_samples_split=5,     # 分裂所需最小样本数
    min_samples_leaf=2,      # 叶节点最小样本数
    max_features='sqrt',     # 每次分裂考虑的特征数
    bootstrap=True,          # 是否使用自助采样
    oob_score=True,          # 是否计算OOB分数
    n_jobs=-1,               # 并行计算
    random_state=42
)
rf_clf.fit(X_train, y_train)

# 评估
print(f"训练准确率: {rf_clf.score(X_train, y_train):.4f}")
print(f"测试准确率: {rf_clf.score(X_test, y_test):.4f}")
print(f"OOB分数: {rf_clf.oob_score_:.4f}")
```

### 特征重要性分析

```python
import pandas as pd

# 获取特征重要性
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排名：")
print(importance_df.head(10))

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性排名')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 回归任务

```python
# 回归任务
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               n_informative=8, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)
print(f"均方误差: {mean_squared_error(y_test, y_pred):.4f}")
```

### 树数量对性能的影响

```python
# 分析树数量的影响
n_estimators_range = range(10, 201, 10)
train_scores = []
test_scores = []
oob_scores = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n_est, 
        oob_score=True, 
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='训练准确率')
plt.plot(n_estimators_range, test_scores, 'r-', label='测试准确率')
plt.plot(n_estimators_range, oob_scores, 'g--', label='OOB分数')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('树数量对性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
```

### 与单棵决策树对比

```python
from sklearn.tree import DecisionTreeClassifier

# 单棵决策树
tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train, y_train)

# 随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# 对比结果
print("模型性能对比：")
print("-" * 50)
print(f"单棵决策树 - 训练: {tree.score(X_train, y_train):.4f}, 测试: {tree.score(X_test, y_test):.4f}")
print(f"随机森林   - 训练: {rf.score(X_train, y_train):.4f}, 测试: {rf.score(X_test, y_test):.4f}")

# 方差分析：多次训练看稳定性
tree_scores = []
rf_scores = []

for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    t = DecisionTreeClassifier(max_depth=10, random_state=seed)
    t.fit(X_tr, y_tr)
    tree_scores.append(t.score(X_te, y_te))
    
    r = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    r.fit(X_tr, y_tr)
    rf_scores.append(r.score(X_te, y_te))

print(f"\n稳定性分析（10次实验）：")
print(f"决策树: 均值={np.mean(tree_scores):.4f}, 标准差={np.std(tree_scores):.4f}")
print(f"随机森林: 均值={np.mean(rf_scores):.4f}, 标准差={np.std(rf_scores):.4f}")
```

## 极端随机树（Extra Trees）

极端随机树（Extremely Randomized Trees）是随机森林的变体：

| 特点 | 随机森林 | 极端随机树 |
|------|---------|-----------|
| 分裂点选择 | 寻找最优分裂点 | 随机选择分裂点 |
| 训练速度 | 中等 | 更快 |
| 方差 | 中等 | 更低 |
| 适用场景 | 通用 | 高方差、噪声数据 |

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
print(f"极端随机树测试准确率: {et.score(X_test, y_test):.4f}")
```

## 优缺点分析

### 优点

✅ **高准确率**：通常优于单棵决策树  
✅ **抗过拟合**：通过平均降低方差  
✅ **并行化**：各树可独立训练  
✅ **无需调参**：默认参数通常表现良好  
✅ **特征重要性**：自然提供特征评估  
✅ **处理高维数据**：能有效处理大量特征

### 缺点

❌ **模型体积大**：需要存储多棵树  
❌ **预测速度**：比单棵树慢  
❌ **可解释性降低**：多树集成难以直观解释  
❌ **外推能力弱**：对训练数据范围外的预测不准

## 最佳实践

### 参数设置建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `n_estimators` | 100-500 | 越多越好，但收益递减 |
| `max_depth` | None 或 10-20 | 根据数据复杂度调整 |
| `min_samples_leaf` | 1-5 | 增大可防止过拟合 |
| `max_features` | 'sqrt' (分类) / 'sqrt' 或 1/3 (回归) | 默认值通常效果不错 |
| `bootstrap` | True | 推荐开启，可使用 OOB |

### 使用场景

**适合使用随机森林**：
- 需要高精度预测
- 数据包含复杂交互作用
- 需要特征重要性分析
- 对模型可解释性要求不高

**不适合使用随机森林**：
- 需要极快预测速度
- 内存受限的部署环境
- 数据量非常小
- 需要外推预测

---

[上一节：决策树](./decision-trees.md) | [下一节：AdaBoost](./adaboost.md)
