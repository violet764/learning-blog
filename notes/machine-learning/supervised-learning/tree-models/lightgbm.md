# LightGBM

LightGBM（Light Gradient Boosting Machine）是微软开发的高效梯度提升框架，通过直方图算法、单边梯度采样、互斥特征捆绑等技术，大幅提升了训练速度和内存效率，特别适合大规模数据处理。

📌 **核心优势**：训练速度快、内存占用低、支持大规模数据、原生处理类别特征。

## 核心技术

### 直方图算法

传统 GBDT 需要遍历所有特征值寻找最优分裂点，LightGBM 将连续特征离散化为直方图：

```
特征值: [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.3, 2.8, 3.1]
                ↓ 离散化为 k 个桶
直方图: 桶1[0-1]: 梯度和=0.5, 样本数=4
        桶2[1-2]: 梯度和=0.3, 样本数=2
        桶3[2-3]: 梯度和=0.8, 样本数=3
        桶4[3-4]: 梯度和=0.2, 样本数=1
```

**优势**：
- 内存使用：从 $O(\text{#data})$ 降到 $O(\text{#bins})$
- 计算复杂度：从 $O(\text{#data} \times \text{#features})$ 降到 $O(\text{#bins} \times \text{#features})$
- 直方图差分加速：兄弟节点的直方图可通过父节点减去得到

### 叶子生长策略（Leaf-wise）

与传统层次生长（Level-wise）不同，LightGBM 采用叶子生长策略：

```
层次生长 (Level-wise):          叶子生长 (Leaf-wise):
        [根]                          [根]
       /    \                        /    \
     [A]    [B]                    [A]    [B]
    /  \    /  \                  /  \
  [C] [D] [E] [F]               [C] [D]
                                (只有增益最大的叶子继续分裂)
```

**对比**：

| 策略 | 特点 | 优点 | 缺点 |
|------|------|------|------|
| Level-wise | 同一层所有节点同时分裂 | 不易过拟合 | 效率低，很多分裂收益小 |
| Leaf-wise | 每次选增益最大的叶子分裂 | 效率高，精度好 | 可能过拟合 |

**控制过拟合**：通过 `max_depth` 限制树的最大深度。

### 单边梯度采样（GOSS）

观察发现，梯度大的样本对信息增益贡献更大。GOSS 的策略：

1. 保留所有梯度大的样本（前 $a\%$）
2. 从梯度小的样本中随机采样 $b\%$
3. 在计算增益时，对采样的低梯度样本乘以权重 $(1-a)/b$

**数学表示**：

$$
\tilde{V}_j(d) = \frac{1}{n} \left( \sum_{x_i \in A} g_i + \frac{1-a}{b} \sum_{x_i \in B} g_i \right)^2
$$

其中 $A$ 是高梯度样本集，$B$ 是随机采样的低梯度样本集。

**效果**：大幅减少计算量，同时保持估计的准确性。

### 互斥特征捆绑（EFB）

稀疏特征空间中，许多特征是互斥的（不同时取非零值）。EFB 将互斥特征捆绑：

```
原始特征:
样本1: 特征A=1, 特征B=0, 特征C=0
样本2: 特征A=0, 特征B=1, 特征C=0
样本3: 特征A=0, 特征B=0, 特征C=1
                ↓ 捆绑
捆绑特征:
样本1: 捆绑特征=1
样本2: 捆绑特征=2
样本3: 捆绑特征=3
```

**算法步骤**：
1. 构建特征冲突图（同时非零的特征有冲突）
2. 图着色算法找出可捆绑的特征组
3. 合并特征值

**效果**：显著减少特征维度，加速训练。

## 类别特征处理

LightGBM 原生支持类别特征，无需独热编码：

**最优分裂算法**：
1. 按目标值统计量对类别排序
2. 在排序后的类别上寻找最优分割点

**优势**：
- 避免独热编码导致的特征空间膨胀
- 更精确的分裂（考虑类别顺序）
- 处理高基数类别特征

```python
# 指定类别特征
lgb_train = lgb.Dataset(X, y, categorical_feature=[0, 2, 5])
```

## 代码示例

### 基础使用

```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# 分类任务
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建数据集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# 训练模型
model = lgb.train(
    params, 
    lgb_train, 
    num_boost_round=100,
    valid_sets=[lgb_test],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
)

# 预测
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
print(f"\n测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### Sklearn 接口

```python
from lightgbm import LGBMClassifier, LGBMRegressor

# 使用 Sklearn 接口
lgb_clf = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(10)]
)

print(f"测试准确率: {lgb_clf.score(X_test, y_test):.4f}")
```

### 回归任务

```python
# 回归任务
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               n_informative=8, noise=0.1, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3)

# 回归模型
lgb_reg = LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgb_reg.fit(X_train_r, y_train_r)

y_pred_r = lgb_reg.predict(X_test_r)
print(f"均方误差: {mean_squared_error(y_test_r, y_pred_r):.4f}")
```

### 不同 Boosting 类型

```python
# 比较不同 boosting 类型
boosting_types = ['gbdt', 'dart', 'goss']
results = []

for boost_type in boosting_types:
    params_temp = params.copy()
    params_temp['boosting'] = boost_type
    
    model_temp = lgb.train(
        params_temp, lgb_train, 
        num_boost_round=50,
        valid_sets=[lgb_test],
        callbacks=[lgb.log_evaluation(0)]  # 静默
    )
    
    y_pred_temp = (model_temp.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    
    results.append({
        'boosting_type': boost_type,
        '准确率': acc,
        '迭代次数': model_temp.current_iteration()
    })

print("不同 Boosting 类型比较：")
import pandas as pd
print(pd.DataFrame(results).to_string(index=False))
```

### 特征重要性

```python
# 特征重要性
importance = model.feature_importance(importance_type='split')
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("特征重要性排名：")
print(importance_df.head(10))

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('特征重要性')
plt.title('LightGBM 特征重要性排名')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 类别特征处理

```python
# 创建含类别特征的数据
np.random.seed(42)
n_samples = 1000

# 数值特征
num_feature = np.random.randn(n_samples)

# 类别特征（高基数）
categories = [f'cat_{i}' for i in range(20)]
cat_feature = np.random.choice(range(len(categories)), n_samples)

# 目标变量
y_cat = (num_feature + cat_feature * 0.1 > 0).astype(int)

X_cat = np.column_stack([num_feature, cat_feature])

# 指定类别特征
lgb_train_cat = lgb.Dataset(
    X_cat, y_cat, 
    categorical_feature=[1]  # 第二列是类别特征
)

params_cat = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'verbose': -1
}

model_cat = lgb.train(params_cat, lgb_train_cat, num_boost_round=50)
print("类别特征处理完成")
```

### 交叉验证

```python
# LightGBM 内置交叉验证
cv_results = lgb.cv(
    params,
    lgb_train,
    num_boost_round=100,
    nfold=5,
    stratified=True,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
)

print(f"\n最佳迭代次数: {len(cv_results['valid binary_logloss-mean'])}")
print(f"最佳测试误差: {min(cv_results['valid binary_logloss-mean']):.4f}")
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# 网格搜索
grid_search = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X[:500], y[:500])

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

### 模型保存与加载

```python
# 保存模型
model.save_model('lightgbm_model.txt')
print("模型已保存")

# 加载模型
loaded_model = lgb.Booster(model_file='lightgbm_model.txt')

# 验证
y_pred_loaded = (loaded_model.predict(X_test) > 0.5).astype(int)
print(f"加载模型准确率: {accuracy_score(y_test, y_pred_loaded):.4f}")

import os
os.remove('lightgbm_model.txt')  # 清理
```

## 关键参数

### 核心参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `num_leaves` | 叶子节点数 | 15-511（控制复杂度） |
| `max_depth` | 树的最大深度 | -1（无限制）或 6-10 |
| `learning_rate` | 学习率 | 0.01-0.3 |
| `n_estimators` | 树的数量 | 100-1000 |

### 直方图参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `max_bin` | 最大分桶数 | 255（默认） |
| `min_data_in_bin` | 每桶最小样本数 | 3（默认） |

### 正则化参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `lambda_l1` | L1 正则化 | 0-10 |
| `lambda_l2` | L2 正则化 | 0-10 |
| `min_gain_to_split` | 最小分裂增益 | 0-5 |
| `min_child_samples` | 叶节点最小样本数 | 5-100 |

### 采样参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `feature_fraction` | 特征采样比例 | 0.7-1.0 |
| `bagging_fraction` | 样本采样比例 | 0.7-1.0 |
| `bagging_freq` | 采样频率 | 5 |

## 与其他算法比较

| 特性 | LightGBM | XGBoost | CatBoost |
|------|----------|---------|----------|
| 训练速度 | ⚡ 最快 | 快 | 中等 |
| 内存占用 | 💾 最低 | 高 | 中等 |
| 直方图算法 | ✅ 原生 | 可选 | ✅ |
| 类别特征 | ✅ 原生支持 | 需编码 | ✅ 自动处理 |
| 缺失值 | 自动处理 | 自动处理 | 自动处理 |
| 大规模数据 | ✅ 最优 | 好 | 好 |

## 最佳实践

### 参数调优策略

1. **调整 `num_leaves`**：这是控制模型复杂度的核心参数
2. **调整 `min_child_samples`**：防止叶子节点样本过少
3. **调整正则化参数**：`lambda_l1`、`lambda_l2`
4. **调整采样参数**：`feature_fraction`、`bagging_fraction`

### 处理过拟合

```python
# 减少过拟合的参数设置
params = {
    'num_leaves': 31,           # 减少
    'min_child_samples': 20,    # 增加
    'max_depth': 6,             # 限制深度
    'lambda_l1': 0.1,           # L1 正则化
    'lambda_l2': 0.1,           # L2 正则化
    'feature_fraction': 0.8,    # 特征采样
    'bagging_fraction': 0.8,    # 样本采样
}
```

### 加速训练

```python
# 加速训练的参数设置
params = {
    'max_bin': 127,             # 减少分桶数
    'learning_rate': 0.1,       # 较大学习率
    'feature_fraction': 0.7,    # 减少特征
    'bagging_fraction': 0.7,    # 减少样本
    'device': 'gpu',            # 使用 GPU
    'num_threads': -1,          # 并行计算
}
```

### 处理大规模数据

```python
# 大规模数据处理
params = {
    'boosting': 'goss',         # 使用 GOSS
    'top_rate': 0.2,            # 保留 20% 高梯度样本
    'other_rate': 0.1,          # 采样 10% 低梯度样本
    'max_bin': 63,              # 减少分桶
}
```

## 常见问题

### Q1: LightGBM 与 XGBoost 如何选择？

| 场景 | 推荐 |
|------|------|
| 大规模数据 | LightGBM |
| 类别特征多 | LightGBM 或 CatBoost |
| 内存受限 | LightGBM |
| 需要最高精度 | 两者都尝试，比较结果 |

### Q2: `num_leaves` 如何设置？

一般建议：`num_leaves <= 2^(max_depth)`，例如 `max_depth=6` 时，`num_leaves` 最大为 64。

### Q3: 如何处理类别特征？

```python
# 方法1：指定类别特征列
lgb_train = lgb.Dataset(X, y, categorical_feature=[0, 2])

# 方法2：使用 pandas 的 category 类型
df['cat_column'] = df['cat_column'].astype('category')
```

---

[上一节：XGBoost](./xgboost.md)
