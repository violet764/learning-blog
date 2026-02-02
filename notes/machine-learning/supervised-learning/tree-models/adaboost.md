# AdaBoost (自适应增强算法)

## 1. 算法概述

AdaBoost（Adaptive Boosting）是一种经典的集成学习方法，通过组合多个弱分类器来构建一个强分类器。其核心思想是"三个臭皮匠，顶个诸葛亮"，通过不断调整样本权重来关注难以分类的样本。

### 1.1 基本思想

AdaBoost的核心在于自适应地调整训练样本的权重分布，使得后续的弱分类器能够专注于之前分类错误的样本。

### 1.2 算法特点

- **自适应权重调整**：每轮迭代后重新调整样本权重
- **顺序训练**：弱分类器按顺序训练，每个分类器修正前一个的错误
- **线性组合**：最终分类器是弱分类器的加权投票
- **理论保证**：具有严格的泛化误差上界

## 2. 数学原理

### 2.1 算法流程

设训练数据集为 $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_m, y_m)\}$，其中 $y_i \in \{-1, +1\}$。

**步骤1**：初始化样本权重
$$ w_i^{(1)} = \frac{1}{m}, \quad i = 1, 2, \dots, m $$

**步骤2**：对于 $t = 1, 2, \dots, T$（T为弱分类器数量）：
1. 使用当前权重分布 $w^{(t)}$ 训练弱分类器 $h_t$
2. 计算分类误差率：
   $$ \epsilon_t = \sum_{i=1}^m w_i^{(t)} I(h_t(x_i) \neq y_i) $$
3. 计算分类器权重：
   $$ \alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right) $$
4. 更新样本权重：
   $$ w_i^{(t+1)} = \frac{w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))}{Z_t} $$
   其中 $Z_t$ 是归一化因子：
   $$ Z_t = \sum_{i=1}^m w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i)) $$

**步骤3**：构建最终分类器
$$ H(x) = \text{sign} \left( \sum_{t=1}^T \alpha_t h_t(x) \right) $$

### 2.2 权重更新公式推导

权重更新公式来源于指数损失函数的最小化：
$$ L(y, f(x)) = \exp(-y f(x)) $$

通过梯度下降方法可以推导出上述权重更新规则。

### 2.3 理论保证

AdaBoost的泛化误差上界为：
$$ \mathbb{P}[H(x) \neq y] \leq \prod_{t=1}^T 2\sqrt{\epsilon_t(1-\epsilon_t)} $$

当每个弱分类器的误差率 $\epsilon_t < 0.5$ 时，整体误差会指数下降。

## 3. 算法变体

### 3.1 Real AdaBoost

使用概率估计而不是硬分类，输出的是实数而不是±1。

### 3.2 Gentle AdaBoost

使用牛顿法进行优化，对异常值更鲁棒。

### 3.3 LogitBoost

基于逻辑回归的损失函数，适用于概率估计。

### 3.4 SAMME 和 SAMME.R

多类别版本的AdaBoost，分别适用于离散和连续输出。

## 4. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd

print("=== AdaBoost分类示例 ===")

# 1. 基础AdaBoost分类
# 生成分类数据
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 使用决策树桩作为弱分类器
base_estimator = DecisionTreeClassifier(max_depth=1)

# 创建AdaBoost分类器
adaboost = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# 训练模型
adaboost.fit(X_train, y_train)

# 预测与评估
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 2. 弱分类器性能分析
print("\n=== 弱分类器性能分析 ===")

# 记录每个弱分类器的误差率
error_rates = []
classifier_weights = []

for t, (error, weight) in enumerate(zip(adaboost.estimator_errors_, adaboost.estimator_weights_)):
    error_rates.append(error)
    classifier_weights.append(weight)
    if t < 10:  # 显示前10个分类器
        print(f"弱分类器 {t+1}: 误差率={error:.4f}, 权重={weight:.4f}")

# 3. 可视化误差率和权重变化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(error_rates)+1), error_rates, 'b-o')
plt.xlabel('弱分类器序号')
plt.ylabel('误差率')
plt.title('弱分类器误差率变化')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(classifier_weights)+1), classifier_weights, 'r-o')
plt.xlabel('弱分类器序号')
plt.ylabel('分类器权重')
plt.title('分类器权重变化')
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. 复杂数据集演示 - 月亮形数据
print("\n=== 复杂数据集演示 ===")

# 生成月亮形数据
X_moons, y_moons = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42)

# 比较不同基础分类器的效果
estimators = {
    '决策树桩 (max_depth=1)': DecisionTreeClassifier(max_depth=1),
    '浅层决策树 (max_depth=2)': DecisionTreeClassifier(max_depth=2),
    '深层决策树 (max_depth=5)': DecisionTreeClassifier(max_depth=5)
}

results = []

for name, base_est in estimators.items():
    # 创建AdaBoost分类器
    ada_model = AdaBoostClassifier(
        estimator=base_est,
        n_estimators=50,
        random_state=42
    )
    
    # 训练和评估
    ada_model.fit(X_train_m, y_train_m)
    y_pred_m = ada_model.predict(X_test_m)
    accuracy_m = accuracy_score(y_test_m, y_pred_m)
    
    results.append({
        '基础分类器': name,
        '准确率': accuracy_m,
        '弱分类器数量': len(ada_model.estimators_)
    })
    
    # 可视化决策边界（仅对第一个模型）
    if name == '决策树桩 (max_depth=1)':
        fig, ax = plt.subplots(figsize=(8, 6))
        DecisionBoundaryDisplay.from_estimator(
            ada_model, X_moons, response_method="predict",
            alpha=0.5, ax=ax
        )
        ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, 
                  edgecolors='k', cmap=plt.cm.Paired)
        ax.set_title(f'AdaBoost决策边界 ({name})')
        plt.show()

# 显示比较结果
results_df = pd.DataFrame(results)
print("\n不同基础分类器的AdaBoost性能比较:")
print(results_df.to_string(index=False))

# 5. 学习曲线分析
print("\n=== 学习曲线分析 ===")

n_estimators_range = range(1, 201, 10)
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    
    train_score = ada_temp.score(X_train, y_train)
    test_score = ada_temp.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='训练准确率', linewidth=2)
plt.plot(n_estimators_range, test_scores, 'r-', label='测试准确率', linewidth=2)
plt.xlabel('弱分类器数量')
plt.ylabel('准确率')
plt.title('AdaBoost学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. 特征重要性分析
print("\n=== 特征重要性分析 ===")

feature_importance = adaboost.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

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
plt.title('AdaBoost特征重要性排名（前10）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. 处理不平衡数据
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

# 使用AdaBoost处理不平衡数据
ada_imbalanced = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

ada_imbalanced.fit(X_train_imb, y_train_imb)

# 评估不平衡数据分类效果
y_pred_imb = ada_imbalanced.predict(X_test_imb)
print("\n不平衡数据分类报告:")
print(classification_report(y_test_imb, y_pred_imb))

# 8. 与单个决策树比较
print("\n=== 与单个决策树比较 ===")

# 单个深层决策树
deep_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
deep_tree.fit(X_train, y_train)
deep_tree_score = deep_tree.score(X_test, y_test)

# AdaBoost with shallow trees
ada_shallow = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_shallow.fit(X_train, y_train)
ada_score = ada_shallow.score(X_test, y_test)

comparison = pd.DataFrame({
    '模型': ['深层决策树 (max_depth=10)', 'AdaBoost (决策树桩)'],
    '测试准确率': [deep_tree_score, ada_score],
    '模型复杂度': ['单个复杂模型', '100个简单模型组合']
})

print("模型性能比较:")
print(comparison.to_string(index=False))

# 9. 超参数调优
print("\n=== 超参数调优 ===")

from sklearn.model_selection import GridSearchCV

# 使用较小的数据集进行调优演示
X_small = X[:500]
y_small = y[:500]

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0, 1.5],
    'estimator__max_depth': [1, 2, 3]  # 基础分类器的深度
}

# 创建基础分类器
base_tree = DecisionTreeClassifier(random_state=42)

# 网格搜索
grid_search = GridSearchCV(
    AdaBoostClassifier(estimator=base_tree, random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_small, y_small)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 10. 实际应用案例：信用卡欺诈检测模拟
print("\n=== 实际应用案例：信用卡欺诈检测模拟 ===")

# 模拟信用卡交易数据（简化版）
n_samples = 2000
fraud_ratio = 0.02  # 2%的欺诈交易

# 生成模拟特征：交易金额、时间、地点等
np.random.seed(42)
transaction_amount = np.random.exponential(100, n_samples)
time_of_day = np.random.uniform(0, 24, n_samples)
location_risk = np.random.beta(2, 5, n_samples)  # 地点风险分数

X_fraud = np.column_stack([transaction_amount, time_of_day, location_risk])

# 生成标签：基于特征的简单规则
fraud_prob = (transaction_amount > 500) * 0.3 + (time_of_day < 6) * 0.2 + location_risk
fraud_prob = fraud_prob / fraud_prob.max()
y_fraud = (fraud_prob > np.percentile(fraud_prob, 100*(1-fraud_ratio))).astype(int)

print(f"欺诈交易比例: {y_fraud.mean():.2%}")

# 使用AdaBoost进行欺诈检测
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)

ada_fraud = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    random_state=42
)

ada_fraud.fit(X_train_f, y_train_f)

# 评估欺诈检测性能
from sklearn.metrics import precision_recall_curve, auc

y_pred_proba = ada_fraud.predict_proba(X_test_f)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_f, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"PR-AUC得分: {pr_auc:.4f}")

# 绘制PR曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('AdaBoost欺诈检测PR曲线')
plt.grid(True, alpha=0.3)
plt.show()
```

## 5. 算法优缺点分析

### 5.1 优点

1. **高准确率**：在实践中通常能达到很好的分类性能
2. **无需特征缩放**：对特征的尺度不敏感
3. **不易过拟合**：通过调整弱分类器复杂度控制过拟合
4. **提供特征重要性**：可以用于特征选择
5. **理论基础坚实**：有严格的误差上界保证

### 5.2 缺点

1. **对噪声敏感**：噪声样本会被赋予高权重
2. **训练时间较长**：需要顺序训练多个弱分类器
3. **弱分类器选择**：需要选择合适的弱分类器类型
4. **数据不平衡问题**：可能偏向多数类

## 6. 实践建议

### 6.1 参数调优策略

1. **弱分类器数量**：从较少的数量开始，逐渐增加直到性能饱和
2. **学习率**：较小的学习率需要更多的弱分类器，但可能获得更好的泛化性能
3. **弱分类器复杂度**：简单的弱分类器（如决策树桩）通常效果更好

### 6.2 适用场景

**适合使用AdaBoost当**：
- 需要高分类准确率
- 数据特征维度适中
- 有足够的训练时间
- 对模型解释性要求不高

**不适合使用AdaBoost当**：
- 数据包含大量噪声
- 需要快速预测
- 数据极度不平衡
- 需要概率输出

### 6.3 与其他算法的比较

| 特性 | AdaBoost | 随机森林 | 梯度提升 |
|------|----------|----------|----------|
| 训练方式 | 顺序 | 并行 | 顺序 |
| 过拟合倾向 | 中等 | 低 | 中等 |
| 处理噪声 | 较差 | 较好 | 中等 |
| 训练速度 | 中等 | 快 | 慢 |
| 可解释性 | 中等 | 高 | 中等 |

## 7. 理论深入

### 7.1 统计学习视角

AdaBoost可以看作是在函数空间中进行坐标下降优化，最小化指数损失函数。

### 7.2 间隔理论

AdaBoost最大化分类间隔（margin），这是其泛化能力的理论基础。

### 7.3 VC维分析

AdaBoost的VC维与其弱分类器的复杂度相关，但实践中的泛化性能通常优于理论界限。

## 8. 扩展应用

### 8.1 多类别分类

通过一对多（One-vs-Rest）或一对一（One-vs-One）策略扩展至多类别问题。

### 8.2 回归问题

AdaBoost.R2等变体可以处理回归任务。

### 8.3 异常检测

通过关注难以分类的样本，AdaBoost可以用于异常检测。

---

[下一节：随机森林](./random-forest.md)