# XGBoost

XGBoost（eXtreme Gradient Boosting）是 GBDT 的高效实现，通过二阶泰勒展开、正则化、并行化等多项优化，成为机器学习竞赛和工业应用中最受欢迎的算法之一。

📌 **核心优势**：在传统 GBDT 基础上引入二阶导数信息、正则化项和高效的工程实现，显著提升训练速度和泛化能力。

## 核心创新

### 二阶泰勒展开

传统 GBDT 使用一阶梯度信息，XGBoost 使用二阶泰勒展开近似损失函数：

$$
L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)
$$

其中：
- $g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$ —— 一阶导数
- $h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)})$ —— 二阶导数

**优势**：
- 更精确的优化方向
- 支持自定义损失函数（只需提供一阶、二阶导数）
- 收敛更快

### 正则化目标函数

XGBoost 的目标函数包含正则化项：

$$
\text{Obj}^{(t)} = \sum_{i=1}^n L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)
$$

正则化项 $\Omega(f_t)$ 控制模型复杂度：

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

其中：
- $T$ —— 叶子节点数量
- $w_j$ —— 叶子节点 $j$ 的权重
- $\gamma$ —— 叶子节点数的惩罚系数
- $\lambda$ —— L2 正则化系数

**正则化的作用**：
- $\gamma$ 鼓励更少的叶子节点
- $\lambda$ 鼓励更小的叶子权重

### 最优分裂公式

将目标函数重写为关于叶子节点权重的形式：

$$
\tilde{\text{Obj}}^{(t)} = \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T
$$

其中 $I_j$ 是叶子节点 $j$ 上的样本集合。

最优叶子权重：

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

最优目标值：

$$
\text{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

**分裂增益**：

$$
\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} - \gamma
$$

其中 $G_L = \sum_{i \in I_L} g_i$，$H_L = \sum_{i \in I_L} h_i$。

## 关键特性

### 缺失值处理

XGBoost 自动学习缺失值的最优处理方向：

1. 在分裂时，将缺失值样本分别尝试放入左子树和右子树
2. 选择使增益最大的方向作为缺失值的"默认方向"
3. 预测时，缺失值按照训练时学到的方向处理

### 特征重要性类型

XGBoost 提供三种特征重要性：

| 类型 | 说明 |
|------|------|
| `weight` | 特征被选为分裂点的次数 |
| `gain` | 特征带来的平均增益 |
| `cover` | 特征影响的样本数 |

### 树构建策略

XGBoost 支持两种分裂策略：

**精确贪心算法**：
- 遍历所有特征的所有可能分裂点
- 找到全局最优分裂
- 计算开销大

**近似算法**：
- 将连续特征分桶
- 只在分位点处考虑分裂
- 适合大规模数据

## 代码示例

### 基础使用

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# 分类任务
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 DMatrix（XGBoost 高效数据结构）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',  # 二分类
    'max_depth': 6,
    'eta': 0.1,                      # 学习率
    'gamma': 0.1,                    # 最小分裂增益
    'lambda': 1.0,                   # L2 正则化
    'alpha': 0.0,                    # L1 正则化
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}

# 训练模型
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, verbose_eval=20)

# 预测
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)
print(f"\n测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### Sklearn 接口

```python
from xgboost import XGBClassifier, XGBRegressor

# 使用 Sklearn 接口
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    gamma=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"测试准确率: {xgb_clf.score(X_test, y_test):.4f}")
```

### 回归任务

```python
# 回归任务
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               n_informative=8, noise=0.1, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3)

# 回归模型
xgb_reg = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
xgb_reg.fit(X_train_r, y_train_r)

y_pred_r = xgb_reg.predict(X_test_r)
print(f"均方误差: {mean_squared_error(y_test_r, y_pred_r):.4f}")
```

### 早停法

```python
# 早停法
xgb_early = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)
xgb_early.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"最佳迭代次数: {xgb_early.best_iteration}")
print(f"测试准确率: {xgb_early.score(X_test, y_test):.4f}")
```

### 特征重要性

```python
# 不同类型的特征重要性
import pandas as pd

importance_types = ['weight', 'gain', 'cover']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, imp_type in enumerate(importance_types):
    importance = model.get_score(importance_type=imp_type)
    features = list(importance.keys())
    values = list(importance.values())
    
    # 排序
    sorted_idx = np.argsort(values)[::-1][:10]
    
    axes[idx].barh(range(10), [values[i] for i in sorted_idx])
    axes[idx].set_yticks(range(10))
    axes[idx].set_yticklabels([features[i] for i in sorted_idx])
    axes[idx].set_xlabel('重要性')
    axes[idx].set_title(f'特征重要性 ({imp_type})')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.show()
```

### 交叉验证

```python
# 使用 XGBoost 内置交叉验证
params_cv = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.1,
    'eval_metric': 'logloss'
}

cv_results = xgb.cv(
    params_cv, 
    dtrain, 
    num_boost_round=100,
    nfold=5,
    stratified=True,
    early_stopping_rounds=20,
    verbose_eval=20
)

print(f"\n最佳迭代次数: {len(cv_results)}")
print(f"最佳测试误差: {cv_results['test-logloss-mean'].min():.4f}")
```

### 处理缺失值

```python
# 创建含缺失值的数据
X_missing = X_train.copy()
np.random.seed(42)
missing_mask = np.random.rand(*X_missing.shape) < 0.1
X_missing[missing_mask] = np.nan

# XGBoost 自动处理缺失值
xgb_missing = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    missing=np.nan,  # 指定缺失值
    random_state=42
)
xgb_missing.fit(X_missing, y_train)

print(f"含缺失值训练准确率: {xgb_missing.score(X_test, y_test):.4f}")
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 网格搜索
grid_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 使用小数据集演示
grid_search.fit(X[:500], y[:500])

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

### 模型保存与加载

```python
# 保存模型
model.save_model('xgboost_model.json')
print("模型已保存")

# 加载模型
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')

# 验证
y_pred_loaded = loaded_model.predict(dtest)
y_pred_loaded = (y_pred_loaded > 0.5).astype(int)
print(f"加载模型准确率: {accuracy_score(y_test, y_pred_loaded):.4f}")

import os
os.remove('xgboost_model.json')  # 清理
```

### 多分类任务

```python
from sklearn.datasets import load_iris

# 多分类
iris = load_iris()
X_multi, y_multi = iris.data, iris.target
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# 多分类模型
xgb_multi = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='multi:softmax',  # 多分类
    num_class=3,                # 类别数
    random_state=42
)
xgb_multi.fit(X_train_m, y_train_m)

print(f"多分类准确率: {xgb_multi.score(X_test_m, y_test_m):.4f}")
```

## 关键参数

### 核心参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `n_estimators` | 树的数量 | 100-1000 |
| `max_depth` | 树的最大深度 | 3-10 |
| `learning_rate` (eta) | 学习率 | 0.01-0.3 |
| `min_child_weight` | 叶子节点最小权重和 | 1-10 |

### 正则化参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `gamma` | 最小分裂增益 | 0-5 |
| `reg_lambda` | L2 正则化 | 0-10 |
| `reg_alpha` | L1 正则化 | 0-10 |

### 采样参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `subsample` | 样本采样比例 | 0.5-1.0 |
| `colsample_bytree` | 特征采样比例 | 0.5-1.0 |

## 与其他算法比较

| 特性 | XGBoost | GBDT | 随机森林 |
|------|---------|------|---------|
| 二阶导数 | ✅ | ❌ | - |
| 正则化 | ✅ | ❌ | - |
| 缺失值处理 | 自动 | 需预处理 | 需预处理 |
| 并行化 | 特征级 | 无 | 树级 |
| 训练速度 | 快 | 中等 | 快 |

## 最佳实践

### 调参策略

1. **固定学习率**：先设置较大的学习率（0.1），调整树的数量
2. **调整树深度**：根据数据复杂度选择 `max_depth`
3. **调整正则化**：`gamma`、`reg_lambda`、`reg_alpha`
4. **调整采样**：`subsample`、`colsample_bytree`
5. **微调学习率**：减小学习率，增加树数量

### 常见问题

**Q1: 如何处理不平衡数据？**
```python
# 计算类别权重
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb_imb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    ...
)
```

**Q2: 如何加速训练？**
- 使用 `tree_method='hist'`（直方图算法）
- 设置 `n_jobs=-1` 并行计算
- 减少 `max_depth` 和 `n_estimators`

---

[上一节：梯度提升](./gradient-boosting.md) | [下一节：LightGBM](./lightgbm.md)
