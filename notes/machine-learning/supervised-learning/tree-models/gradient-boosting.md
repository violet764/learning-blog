# 梯度提升算法

## 1. 算法概述

梯度提升是一种强大的集成学习方法，通过逐步添加弱学习器来改进模型性能。每个新模型都拟合前一轮模型的残差，从而逐步减少预测误差。

### 1.1 基本思想

梯度提升的核心思想是使用梯度下降的方法在函数空间中进行优化，通过迭代地添加弱学习器来改进模型。

### 1.2 算法特点

- **顺序训练**：模型按顺序构建，每个模型修正前一个模型的错误
- **任意损失函数**：可以处理回归、分类、排序等多种任务
- **强大的预测能力**：在实践中通常达到state-of-the-art性能
- **可解释性**：提供特征重要性度量

## 2. 数学原理

### 2.1 函数空间优化

梯度提升将机器学习问题视为函数空间中的优化问题：
$$ F^*(\mathbf{x}) = \arg\min_F \mathbb{E}[L(y, F(\mathbf{x}))] $$

其中L是损失函数。由于直接求解困难，使用梯度下降方法逐步逼近最优解。

### 2.2 算法推导

设当前模型为$F_{m-1}(\mathbf{x})$，我们希望找到增量$h_m(\mathbf{x})$使得：
$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \rho_m h_m(\mathbf{x}) $$

其中$h_m(\mathbf{x})$拟合负梯度：
$$ -g_m(\mathbf{x}_i) = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}} $$

### 2.3 具体算法步骤

**输入**：训练数据$\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，损失函数L，弱学习器h，迭代次数M

1. 初始化模型：$F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$
2. 对于m=1到M：
   a. 计算伪残差：$r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$
   b. 拟合弱学习器：$h_m = \arg\min_h \sum_{i=1}^n (r_{im} - h(\mathbf{x}_i))^2$
   c. 计算步长：$\rho_m = \arg\min_\rho \sum_{i=1}^n L(y_i, F_{m-1}(\mathbf{x}_i) + \rho h_m(\mathbf{x}_i))$
   d. 更新模型：$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \rho_m h_m(\mathbf{x})$
3. 输出最终模型：$F_M(\mathbf{x})$

## 3. 常用变体

### 3.1 Gradient Boosting Machine (GBM)

**特点**：
- 使用决策树作为弱学习器
- 支持自定义损失函数
- 包含正则化防止过拟合

### 3.2 XGBoost (Extreme Gradient Boosting)

**改进**：
- 二阶泰勒展开近似损失函数
- 正则化项控制模型复杂度
- 并行化和硬件优化
- 处理缺失值的智能策略

### 3.3 LightGBM

**优化**：
- 基于直方图的算法加速
- 带深度限制的叶子生长策略
- 类别特征的最优分割
- 更少的内存占用

### 3.4 CatBoost

**特色**：
- 自动处理类别特征
- 有序提升防止目标泄漏
- 对称树结构
- 强大的GPU支持

## 4. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.inspection import permutation_importance
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

# 1. 基本梯度提升分类示例
print("=== 梯度提升分类示例 ===")

# 生成分类数据
X_class, y_class = make_classification(n_samples=1000, n_features=20, 
                                      n_informative=15, n_redundant=5,
                                      random_state=42)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42)

# 训练梯度提升分类器
gbm_classifier = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbm_classifier.fit(X_train_class, y_train_class)

# 预测与评估
y_pred_class = gbm_classifier.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"测试集准确率: {accuracy:.4f}")

# 2. 梯度提升回归示例
print("\n=== 梯度提升回归示例 ===")

# 生成回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               n_informative=8, noise=0.1, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# 训练梯度提升回归器
gbm_regressor = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbm_regressor.fit(X_train_reg, y_train_reg)

# 预测与评估
y_pred_reg = gbm_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"测试集均方误差: {mse:.4f}")

# 3. 学习曲线分析
print("\n=== 学习曲线分析 ===")

# 记录每轮迭代的损失
train_errors = []
test_errors = []

# 逐步训练并记录误差
for n_est in range(1, 101, 5):
    gbm_temp = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1, 
                                        max_depth=3, random_state=42)
    gbm_temp.fit(X_train_reg, y_train_reg)
    
    train_error = mean_squared_error(y_train_reg, gbm_temp.predict(X_train_reg))
    test_error = mean_squared_error(y_test_reg, gbm_temp.predict(X_test_reg))
    
    train_errors.append(train_error)
    test_errors.append(test_error)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101, 5), train_errors, 'b-', label='训练误差')
plt.plot(range(1, 101, 5), test_errors, 'r-', label='测试误差')
plt.xlabel('弱学习器数量')
plt.ylabel('均方误差')
plt.title('梯度提升学习曲线')
plt.legend()
plt.grid(True)
plt.show()

# 4. 特征重要性分析
print("\n=== 特征重要性分析 ===")

feature_importance = gbm_classifier.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X_class.shape[1])]

# 创建重要性数据框
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("特征重要性排名:")
print(importance_df.head(10))

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('特征重要性')
plt.title('梯度提升特征重要性排名（前10）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 5. XGBoost示例
print("\n=== XGBoost示例 ===")

# 转换为DMatrix格式（XGBoost专用）
dtrain = xgb.DMatrix(X_train_class, label=y_train_class)
dtest = xgb.DMatrix(X_test_class, label=y_test_class)

# 设置参数
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# 训练模型
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# 预测
xgb_pred_proba = xgb_model.predict(dtest)
xgb_pred = (xgb_pred_proba > 0.5).astype(int)

xgb_accuracy = accuracy_score(y_test_class, xgb_pred)
print(f"XGBoost测试集准确率: {xgb_accuracy:.4f}")

# 6. LightGBM示例
print("\n=== LightGBM示例 ===")

# 创建数据集
lgb_train = lgb.Dataset(X_train_class, y_train_class)
lgb_test = lgb.Dataset(X_test_class, y_test_class, reference=lgb_train)

# 设置参数
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

# 训练模型
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100,
                      valid_sets=[lgb_test], callbacks=[lgb.log_evaluation(50)])

# 预测
lgb_pred_proba = lgb_model.predict(X_test_class)
lgb_pred = (lgb_pred_proba > 0.5).astype(int)

lgb_accuracy = accuracy_score(y_test_class, lgb_pred)
print(f"LightGBM测试集准确率: {lgb_accuracy:.4f}")

# 7. 不同梯度提升算法比较
print("\n=== 不同梯度提升算法比较 ===")

from sklearn.ensemble import HistGradientBoostingClassifier

# 定义不同算法
algorithms = {
    'GBM (sklearn)': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'HistGBM': HistGradientBoostingClassifier(max_iter=100, random_state=42)
}

comparison_results = []

for name, algo in algorithms.items():
    # 训练时间测量
    import time
    start_time = time.time()
    
    algo.fit(X_train_class, y_train_class)
    train_time = time.time() - start_time
    
    # 预测准确率
    y_pred = algo.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred)
    
    comparison_results.append({
        '算法': name,
        '准确率': accuracy,
        '训练时间(秒)': train_time
    })

# 添加XGBoost和LightGBM结果
comparison_results.extend([
    {'算法': 'XGBoost', '准确率': xgb_accuracy, '训练时间(秒)': 'N/A'},
    {'算法': 'LightGBM', '准确率': lgb_accuracy, '训练时间(秒)': 'N/A'}
])

comparison_df = pd.DataFrame(comparison_results)
print("\n梯度提升算法比较:")
print(comparison_df.to_string(index=False))

# 8. 超参数调优
print("\n=== 超参数调优 ===")

from sklearn.model_selection import GridSearchCV

# 使用较小的数据集进行调优演示
X_small = X_class[:500]
y_small = y_class[:500]

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# 网格搜索
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_small, y_small)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 9. 早停法（Early Stopping）
print("\n=== 早停法应用 ===")

# 使用验证集进行早停
gbm_early = GradientBoostingClassifier(
    n_estimators=1000,  # 设置较大的迭代次数
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.1,  # 验证集比例
    n_iter_no_change=5,      # 连续5轮无改进则停止
    random_state=42
)

gbm_early.fit(X_train_class, y_train_class)

print(f"实际使用的弱学习器数量: {gbm_early.n_estimators_}")
print(f"早停轮数: {1000 - gbm_early.n_estimators_}")

# 10. 偏依赖图（Partial Dependence Plot）
print("\n=== 偏依赖图分析 ===")

from sklearn.inspection import PartialDependenceDisplay

# 选择最重要的两个特征
important_features = importance_df['feature'][:2].tolist()
feature_indices = [int(f.split('_')[1]) for f in important_features]

fig, ax = plt.subplots(figsize=(12, 5))

PartialDependenceDisplay.from_estimator(
    gbm_classifier, X_train_class, feature_indices,
    ax=ax, grid_resolution=20
)

plt.suptitle('偏依赖图')
plt.tight_layout()
plt.show()

# 11. 模型解释性：SHAP值
print("\n=== SHAP值分析 ===")

try:
    import shap
    
    # 创建解释器
    explainer = shap.TreeExplainer(gbm_classifier)
    shap_values = explainer.shap_values(X_test_class)
    
    # 摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_class, feature_names=feature_names)
    plt.tight_layout()
    plt.show()
    
    print("SHAP分析完成")
    
except ImportError:
    print("SHAP库未安装，跳过SHAP分析")

# 12. 处理不平衡数据
print("\n=== 处理不平衡数据 ===")

# 创建不平衡数据集
from sklearn.utils import class_weight

X_imbalanced, y_imbalanced = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    weights=[0.9, 0.1], random_state=42  # 90% vs 10%
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.3, random_state=42
)

print(f"少数类比例: {np.mean(y_imbalanced == 1):.2%}")

# 使用类别权重
gbm_balanced = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)

gbm_balanced.fit(X_train_imb, y_train_imb)

# 评估
from sklearn.metrics import classification_report, confusion_matrix

y_pred_imb = gbm_balanced.predict(X_test_imb)
print("\n不平衡数据分类报告:")
print(classification_report(y_test_imb, y_pred_imb))

# 绘制混淆矩阵
cm = confusion_matrix(y_test_imb, y_pred_imb)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.xticks([0, 1])
plt.yticks([0, 1])

# 添加数值标签
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')

plt.tight_layout()
plt.show()
```

## 5. 高级特性与优化

### 5.1 正则化技术

**收缩（Shrinkage）**：通过学习率控制每棵树的贡献：
$$ F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu h_m(\mathbf{x}) $$
其中$\nu$是学习率，通常取0.01-0.1。

**子采样（Subsampling）**：每次迭代随机选择部分样本。

**特征采样**：每次分裂随机选择部分特征。

### 5.2 损失函数选择

**回归问题**：
- 平方损失：$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$
- 绝对损失：$L(y, \hat{y}) = |y - \hat{y}|$
- Huber损失：结合平方和绝对损失的优点

**分类问题**：
- 对数损失：$L(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
- 指数损失：$L(y, \hat{y}) = \exp(-y\hat{y})$

### 5.3 数值稳定性

**二阶导数**：XGBoost使用二阶泰勒展开提高数值稳定性。

**分位数损失**：对异常值更鲁棒的损失函数。

## 6. 实践建议

### 6.1 参数调优策略

1. **学习率与树数量**：固定学习率，增加树数量直到性能饱和
2. **树深度**：从较浅的树开始，逐步增加深度
3. **正则化参数**：根据过拟合程度调整子采样比例等

### 6.2 算法选择指南

**选择梯度提升当**：
- 需要最高的预测精度
- 数据量适中或较大
- 有足够的计算资源

**选择具体变体**：
- **XGBoost**：需要最好性能和最多功能
- **LightGBM**：处理大规模数据，需要快速训练
- **CatBoost**：包含大量类别特征

### 6.3 部署考虑

**内存使用**：梯度提升模型通常比随机森林占用更多内存。

**预测速度**：树数量较多时预测可能较慢。

**模型解释**：使用SHAP等工具提高模型可解释性。

## 7. 理论深入

### 7.1 函数空间视角

梯度提升可以看作在再生核希尔伯特空间（RKHS）中的梯度下降。

### 7.2 统计一致性

在适当条件下，梯度提升具有统计一致性，即当样本数趋于无穷时收敛到最优解。

### 7.3 泛化误差界

基于Rademacher复杂度的泛化误差界为梯度提升提供了理论保证。

---

[下一节：高斯过程](../bayesian-methods/gaussian-processes.md)