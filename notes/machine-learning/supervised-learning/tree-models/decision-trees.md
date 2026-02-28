# 决策树

决策树是一种基于树形结构进行决策的监督学习算法，通过一系列的 if-then 规则对数据进行分类或回归。它模拟了人类的决策过程，具有直观、可解释性强的特点，是许多集成学习方法（如随机森林、梯度提升树）的基础。

📌 **核心思想**：将特征空间划分为若干个互不重叠的区域，每个区域对应一个预测值。

## 基本概念

### 树的结构

决策树由三种类型的节点组成：

| 节点类型 | 说明 | 特点 |
|---------|------|------|
| **根节点** | 包含所有训练数据的起始节点 | 树的入口点 |
| **内部节点** | 根据特征进行数据划分的节点 | 包含分裂条件和分支 |
| **叶节点** | 最终的分类或回归结果 | 不再分裂，输出预测值 |

### 决策过程

```
                    [根节点: 天气?]
                    /      |      \
               晴朗/    多云|       \下雨
                 /         |         \
        [温度?]      [是:打球]    [风速?]
        /    \                      /    \
      高      低                  强      弱
      |        |                   |       |
   [否]      [是]              [否]      [是]
```

从根节点开始，根据特征的值沿着相应的分支向下移动，直到到达叶节点，叶节点的值即为预测结果。

## 分裂准则

决策树的核心问题是：**在每个节点应该选择哪个特征进行分裂？** 不同的分裂准则对应不同的决策树算法。

### 信息熵

信息熵衡量数据集的不确定性或混乱程度：

$$
H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

其中：
- $D$：当前节点的数据集
- $K$：类别的数量
- $p_k$：类别 $k$ 在数据集 $D$ 中的比例

**熵的性质**：
- 当数据集完全纯净时（只有一个类别），熵为 **0**
- 当各类别均匀分布时，熵达到 **最大值** $\log_2 K$
- 对于二分类问题，最大熵为 **1**

```
        熵值变化曲线（二分类）
        
熵值 1.0 ┤        ╱╲
          │      ╱  ╲
      0.5 ┤    ╱      ╲
          │  ╱          ╲
      0.0 ┼╱              ╲
          └─────────────────→ 正类比例
         0.0   0.5        1.0
```

### 信息增益（ID3 算法）

信息增益衡量使用特征 $A$ 进行分裂后，数据集不确定性的减少程度：

$$
\text{Gain}(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)
$$

其中：
- $V$：特征 $A$ 的取值个数
- $D_v$：特征 $A$ 取值为 $v$ 的样本子集
- $|D_v|/|D|$：子集样本数的权重

💡 **选择策略**：选择信息增益最大的特征作为分裂特征。

⚠️ **缺点**：信息增益偏向于取值较多的特征（如身份证号），因为这样的特征能将数据分得更细，使得分裂后的熵更低。

### 信息增益率（C4.5 算法）

为了解决信息增益偏向取值多的特征的问题，C4.5 使用信息增益率：

$$
\text{GainRatio}(D, A) = \frac{\text{Gain}(D, A)}{\text{SplitInfo}(D, A)}
$$

其中分裂信息衡量特征取值的分布均匀程度：

$$
\text{SplitInfo}(D, A) = -\sum_{v=1}^{V} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}
$$

**修正原理**：如果一个特征的取值很多，其 SplitInfo 值会很大，从而降低增益率。

### 基尼系数（CART 算法）

基尼系数衡量数据集的不纯度：

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

基尼增益（分裂后的不纯度减少）：

$$
\Delta \text{Gini} = \text{Gini}(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} \text{Gini}(D_v)
$$

**基尼系数的特点**：
- 计算更简单（不需要对数运算）
- 对于二分类问题：$\text{Gini}(D) = 2p(1-p)$
- 取值范围：$[0, 1 - 1/K]$

### 三种准则的比较

| 准则 | 算法 | 计算复杂度 | 特点 |
|------|------|-----------|------|
| 信息增益 | ID3 | 较高（对数运算） | 偏向多值特征 |
| 信息增益率 | C4.5 | 较高 | 修正了偏向问题 |
| 基尼系数 | CART | 较低 | 计算简单，最常用 |

## 决策树构建流程

### 完整算法步骤

```
输入：训练数据集 D，特征集 A
输出：决策树 T

1. 如果 D 中所有样本属于同一类别 C：
      返回叶节点，标记为类别 C

2. 如果 A 为空 或 D 中样本在 A 上取值相同：
      返回叶节点，标记为 D 中样本数最多的类别

3. 从 A 中选择最优分裂特征 a*

4. 对于 a* 的每个可能取值 a*_v：
      - 划分出子集 D_v
      - 如果 D_v 为空：
          添加叶节点，标记为 D 中样本数最多的类别
      - 否则：
          递归构建子树 Tree(D_v, A - {a*})

5. 返回决策树 T
```

### 停止条件

决策树停止生长的条件：
1. 当前节点所有样本属于同一类别
2. 特征集为空，或所有特征取值相同
3. 样本数量小于预设阈值
4. 树的深度达到预设限制
5. 分裂后的增益小于预设阈值

## 剪枝策略

决策树容易过拟合训练数据，剪枝是防止过拟合的重要手段。

### 预剪枝（Pre-pruning）

在构建过程中提前停止树的生长：

| 方法 | 说明 | 参数 |
|------|------|------|
| 最大深度限制 | 限制树的最大深度 | `max_depth` |
| 最小样本分割数 | 节点分裂所需的最小样本数 | `min_samples_split` |
| 最小叶节点样本数 | 叶节点所需的最小样本数 | `min_samples_leaf` |
| 最小信息增益 | 分裂所需的最小增益阈值 | `min_impurity_decrease` |

**优点**：训练速度快，计算资源消耗少  
**缺点**：可能欠拟合，错过后续有价值的分裂

### 后剪枝（Post-pruning）

先构建完整的树，然后自底向上进行剪枝：

**代价复杂度剪枝（CCP）**：
$$
R_\alpha(T) = R(T) + \alpha |T|
$$

其中：
- $R(T)$：树的预测误差
- $|T|$：叶节点数量
- $\alpha$：复杂度参数

**剪枝过程**：计算每个非叶节点的剪枝阈值 $\alpha$，逐步剪去 $\alpha$ 最小的子树。

**优点**：通常能获得更好的泛化性能  
**缺点**：需要构建完整树，计算开销大

## 代码示例

### 基础使用

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载示例数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 创建并训练决策树
clf = DecisionTreeClassifier(
    criterion='gini',        # 分裂准则：'gini' 或 'entropy'
    max_depth=3,             # 最大深度
    min_samples_split=5,     # 分裂所需最小样本数
    min_samples_leaf=2,      # 叶节点最小样本数
    random_state=42
)
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.title("决策树结构可视化")
plt.tight_layout()
plt.show()
```

### 信息增益计算示例

```python
import numpy as np

def entropy(y):
    """计算信息熵"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature_idx):
    """计算信息增益"""
    # 父节点熵
    parent_entropy = entropy(y)
    
    # 特征取值
    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    
    # 加权子节点熵
    child_entropy = 0
    for v, c in zip(values, counts):
        mask = X[:, feature_idx] == v
        child_entropy += (c / len(y)) * entropy(y[mask])
    
    return parent_entropy - child_entropy

# 示例：天气数据集
# 特征：[天气(0:晴朗,1:多云,2:下雨), 温度(0:高,1:中,2:低)]
# 标签：是否打网球(0:否, 1:是)
X = np.array([
    [0, 0], [0, 0], [1, 0], [2, 1], [2, 2],
    [2, 2], [1, 2], [0, 1], [0, 2], [2, 1],
    [0, 1], [1, 1], [1, 0], [2, 1]
])
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

print(f"数据集熵: {entropy(y):.4f}")
print(f"天气特征信息增益: {information_gain(X, y, 0):.4f}")
print(f"温度特征信息增益: {information_gain(X, y, 1):.4f}")
```

### 回归树示例

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 生成回归数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 训练回归树
reg = DecisionTreeRegressor(max_depth=4, random_state=42)
reg.fit(X, y)

# 预测
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = reg.predict(X_test)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="数据")
plt.plot(X_test, y_pred, color="cornflowerblue", label="预测", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("决策树回归")
plt.legend()
plt.show()
```

### 剪枝效果对比

```python
from sklearn.datasets import make_classification

# 生成复杂数据
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 不同剪枝策略的模型
models = {
    '无剪枝': DecisionTreeClassifier(random_state=42),
    '预剪枝(max_depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    '预剪枝(min_samples_leaf=10)': DecisionTreeClassifier(min_samples_leaf=10, random_state=42),
}

print("剪枝效果对比：")
print("-" * 50)
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name:30s} | 训练: {train_score:.4f} | 测试: {test_score:.4f}")
```

## 连续特征处理

对于连续特征，CART 算法采用二分法进行处理：

1. 将特征值排序
2. 计算所有相邻值的中间点作为候选分裂点
3. 选择使基尼增益最大的分裂点

```python
def find_best_split(X, y, feature_idx):
    """寻找连续特征的最佳分裂点"""
    values = np.sort(np.unique(X[:, feature_idx]))
    best_gain = -np.inf
    best_threshold = None
    
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # 计算基尼增益
        n = len(y)
        n_left, n_right = left_mask.sum(), right_mask.sum()
        
        if n_left == 0 or n_right == 0:
            continue
            
        def gini(y):
            _, counts = np.unique(y, return_counts=True)
            p = counts / len(y)
            return 1 - np.sum(p ** 2)
        
        gain = gini(y) - (n_left / n) * gini(y[left_mask]) - (n_right / n) * gini(y[right_mask])
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    return best_threshold, best_gain
```

## 优缺点分析

### 优点

✅ **可解释性强**：树结构直观，易于理解和解释  
✅ **无需特征预处理**：不需要归一化、标准化  
✅ **处理混合类型特征**：可同时处理数值和类别特征  
✅ **对缺失值鲁棒**：某些实现支持自动处理缺失值  
✅ **计算效率高**：预测时间复杂度为 $O(\log n)$

### 缺点

❌ **容易过拟合**：特别是深层树  
❌ **不稳定**：数据的微小变化可能导致完全不同的树  
❌ **局部最优**：贪心算法无法保证全局最优  
❌ **忽略特征相关性**：独立考虑每个特征的分裂

## 常见问题

### Q1: 为什么决策树不需要特征标准化？

决策树基于特征值的大小关系进行分裂，而不是距离计算。标准化只改变特征的数值范围，不改变其相对大小关系，因此不影响分裂点的选择。

### Q2: 如何处理类别特征？

- **sklearn**：需要先进行编码（标签编码或独热编码）
- **LightGBM/CatBoost**：原生支持类别特征，无需编码

### Q3: 决策树与随机森林的关系？

随机森林是决策树的集成，通过构建多棵决策树并取平均/投票来提升性能和降低过拟合风险。

---

[下一节：随机森林](./random-forest.md)
