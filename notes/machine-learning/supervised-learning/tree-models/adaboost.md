# AdaBoost

AdaBoost（Adaptive Boosting，自适应增强）是最经典的 Boosting 算法之一，通过自适应地调整样本权重，让后续的弱分类器专注于之前分类错误的样本，最终组合成一个强分类器。

📌 **核心思想**："三个臭皮匠，顶个诸葛亮" —— 通过组合多个弱学习器来构建强学习器。

## Boosting vs Bagging

| 特性 | Bagging | Boosting |
|------|---------|----------|
| 训练方式 | 并行 | 顺序 |
| 样本权重 | 均等 | 动态调整 |
| 学习器权重 | 均等 | 按性能加权 |
| 主要目标 | 降低方差 | 降低偏差 |
| 代表算法 | 随机森林 | AdaBoost, GBDT |

## 算法原理

### 基本流程

假设训练数据集为 $D = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_m, y_m)\}$，其中 $y_i \in \{-1, +1\}$。

**Step 1：初始化样本权重**

$$
w_i^{(1)} = \frac{1}{m}, \quad i = 1, 2, \dots, m
$$

**Step 2：迭代训练（$t = 1, 2, \dots, T$）**

1. 使用当前权重分布训练弱分类器 $h_t$
2. 计算加权误差率：
   $$
   \epsilon_t = \sum_{i=1}^m w_i^{(t)} \cdot \mathbb{I}(h_t(\mathbf{x}_i) \neq y_i)
   $$
3. 计算分类器权重：
   $$
   \alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)
   $$
4. 更新样本权重：
   $$
   w_i^{(t+1)} = \frac{w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(\mathbf{x}_i))}{Z_t}
   $$
   其中 $Z_t$ 是归一化因子：
   $$
   Z_t = \sum_{i=1}^m w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(\mathbf{x}_i))
   $$

**Step 3：构建最终分类器**

$$
H(\mathbf{x}) = \text{sign} \left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)
$$

### 权重更新解析

关键观察权重更新公式：

$$
w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(\mathbf{x}_i))
$$

| 情况 | $y_i h_t(\mathbf{x}_i)$ | 权重变化 |
|------|------------------------|----------|
| 分类正确 | $= +1$ | 权重降低（乘以 $e^{-\alpha_t}$）|
| 分类错误 | $= -1$ | 权重增加（乘以 $e^{+\alpha_t}$）|

**直观理解**：分类错误的样本在下一轮会受到更多关注。

### 分类器权重 $\alpha_t$

$\alpha_t$ 反映了弱分类器 $h_t$ 的重要性：

```
    α_t
      │
  2.0 ┤              ╭──────
      │            ╱
  1.0 ┤         ╱
      │       ╱
  0.5 ┤     ╱
      │   ╱
  0.0 ┼──●─────────────────→ ε_t
      0  0.1  0.3  0.5
```

- 当 $\epsilon_t < 0.5$ 时，$\alpha_t > 0$（分类器有效）
- 当 $\epsilon_t = 0.5$ 时，$\alpha_t = 0$（分类器无贡献）
- 当 $\epsilon_t > 0.5$ 时，$\alpha_t < 0$（反向预测）

## 理论保证

### 训练误差上界

AdaBoost 的训练误差有如下上界：

$$
\frac{1}{m} \sum_{i=1}^m \mathbb{I}(H(\mathbf{x}_i) \neq y_i) \leq \prod_{t=1}^T 2\sqrt{\epsilon_t(1-\epsilon_t)}
$$

**关键结论**：只要每个弱分类器的误差率 $\epsilon_t < 0.5$，训练误差会以指数速度下降。

### 间隔理论

AdaBoost 实际上是在最大化分类间隔：

$$
\text{margin}(\mathbf{x}, y) = \frac{y \sum_{t=1}^T \alpha_t h_t(\mathbf{x})}{\sum_{t=1}^T \alpha_t}
$$

间隔越大，分类置信度越高，泛化能力越强。

## 算法变体

### SAMME（多类别 AdaBoost）

将二分类扩展到多分类：

$$
\alpha_t = \frac{1}{2} \ln \left( \frac{1-\epsilon_t}{\epsilon_t} \right) + \ln(K-1)
$$

其中 $K$ 为类别数。

### SAMME.R（实数版本）

输出实数值而非类别标签，通常收敛更快。

### AdaBoost.R2（回归版本）

使用相对误差或平方误差处理回归问题。

## 代码示例

### 基础使用

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用决策树桩作为弱分类器
base_estimator = DecisionTreeClassifier(max_depth=1)

# 创建 AdaBoost 分类器
ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=42
)
ada.fit(X_train, y_train)

# 评估
y_pred = ada.predict(X_test)
print(f"测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### 分析弱分类器性能

```python
# 查看每个弱分类器的误差率和权重
print("弱分类器分析：")
print("-" * 50)
for t in range(min(10, len(ada.estimators_))):
    error = ada.estimator_errors_[t]
    weight = ada.estimator_weights_[t]
    print(f"弱分类器 {t+1}: 误差率={error:.4f}, 权重={weight:.4f}")

# 可视化误差率和权重变化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, len(ada.estimator_errors_) + 1), ada.estimator_errors_, 'b-o')
axes[0].set_xlabel('弱分类器序号')
axes[0].set_ylabel('误差率')
axes[0].set_title('弱分类器误差率变化')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(ada.estimator_weights_) + 1), ada.estimator_weights_, 'r-o')
axes[1].set_xlabel('弱分类器序号')
axes[1].set_ylabel('权重')
axes[1].set_title('分类器权重变化')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 不同基分类器对比

```python
from sklearn.linear_model import LogisticRegression

# 不同的基分类器
base_estimators = {
    '决策树桩 (depth=1)': DecisionTreeClassifier(max_depth=1),
    '浅层决策树 (depth=2)': DecisionTreeClassifier(max_depth=2),
    '深层决策树 (depth=5)': DecisionTreeClassifier(max_depth=5),
}

print("不同基分类器性能对比：")
print("-" * 60)

results = []
for name, base_est in base_estimators.items():
    ada_model = AdaBoostClassifier(
        estimator=base_est,
        n_estimators=50,
        random_state=42
    )
    ada_model.fit(X_train, y_train)
    train_acc = ada_model.score(X_train, y_train)
    test_acc = ada_model.score(X_test, y_test)
    results.append({
        '基分类器': name,
        '训练准确率': train_acc,
        '测试准确率': test_acc
    })
    print(f"{name:25s} | 训练: {train_acc:.4f} | 测试: {test_acc:.4f}")
```

### 学习曲线分析

```python
# 分析弱分类器数量对性能的影响
n_estimators_range = range(1, 201, 5)
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    train_scores.append(ada_temp.score(X_train, y_train))
    test_scores.append(ada_temp.score(X_test, y_test))

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='训练准确率', linewidth=2)
plt.plot(n_estimators_range, test_scores, 'r-', label='测试准确率', linewidth=2)
plt.xlabel('弱分类器数量')
plt.ylabel('准确率')
plt.title('AdaBoost 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 从零实现 AdaBoost

```python
class SimpleAdaBoost:
    """简化的 AdaBoost 二分类实现"""
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # 初始化权重
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # 训练弱分类器（决策树桩）
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=weights)
            
            # 预测
            y_pred = stump.predict(X)
            
            # 计算加权误差
            incorrect = (y_pred != y).astype(float)
            error = np.dot(weights, incorrect)
            
            # 避免除零
            if error >= 0.5:
                break
            
            # 计算分类器权重
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # 更新样本权重
            weights *= np.exp(-alpha * y * y_pred)
            weights /= weights.sum()  # 归一化
            
            self.estimators.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        # 加权投票
        predictions = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.estimators):
            predictions += alpha * stump.predict(X)
        return np.sign(predictions)

# 测试自定义实现
simple_ada = SimpleAdaBoost(n_estimators=50)
simple_ada.fit(X_train, y_train)
y_pred_custom = simple_ada.predict(X_test)
print(f"\n自定义 AdaBoost 准确率: {accuracy_score(y_test, y_pred_custom):.4f}")
```

### 处理不平衡数据

```python
from sklearn.metrics import classification_report

# 创建不平衡数据
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    weights=[0.9, 0.1], random_state=42  # 90% vs 10%
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42
)

print(f"少数类比例: {np.mean(y_imb == 1):.2%}")

# 使用 AdaBoost
ada_imb = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada_imb.fit(X_train_imb, y_train_imb)

# 评估
y_pred_imb = ada_imb.predict(X_test_imb)
print("\n不平衡数据分类报告：")
print(classification_report(y_test_imb, y_pred_imb))
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0, 1.5],
    'estimator__max_depth': [1, 2, 3]
}

# 网格搜索
base_tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    AdaBoostClassifier(estimator=base_tree, random_state=42),
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

## 优缺点分析

### 优点

✅ **高准确率**：通常能达到很好的分类性能  
✅ **自动特征选择**：通过权重调整关注重要特征  
✅ **不易过拟合**：实践中表现良好（有理论支持）  
✅ **简单灵活**：可与各种弱学习器结合  
✅ **理论保证**：有严格的误差上界

### 缺点

❌ **对噪声敏感**：噪声样本会被赋予高权重，影响性能  
❌ **训练时间长**：顺序训练，无法并行  
❌ **依赖弱分类器**：需要选择合适的弱学习器  
❌ **不平衡数据**：可能偏向多数类

## 与其他算法比较

| 特性 | AdaBoost | 随机森林 | GBDT |
|------|----------|----------|------|
| 训练方式 | 顺序 | 并行 | 顺序 |
| 主要优化 | 指数损失 | 降低方差 | 任意可微损失 |
| 过拟合风险 | 中等 | 低 | 中等 |
| 处理噪声 | 较差 | 较好 | 中等 |
| 参数调优 | 相对简单 | 简单 | 较复杂 |

## 最佳实践

### 参数设置建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `n_estimators` | 50-500 | 足够大，配合早停 |
| `learning_rate` | 0.1-1.0 | 较小值需要更多弱分类器 |
| `estimator` | 决策树桩 | 简单模型通常效果更好 |

### 适用场景

**适合使用 AdaBoost**：
- 二分类任务
- 数据噪声较少
- 需要快速原型验证
- 弱分类器容易获得

**不适合使用 AdaBoost**：
- 数据包含大量噪声
- 需要快速训练
- 数据极度不平衡
- 需要概率输出

---

[上一节：随机森林](./random-forest.md) | [下一节：梯度提升](./gradient-boosting.md)
