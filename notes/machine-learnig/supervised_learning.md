# 有监督学习算法

## KNN
K-最近邻（K-Nearest Neighbors，简称KNN）是一种基于实例的监督学习算法，可用于分类和回归任务。它的核心思想是：**相似的对象具有相似的特征**。

**基本思想**
给定一个测试样本，KNN算法在训练集中找到与该样本最相似的K个样本（即最近的邻居），然后根据这K个样本的标签来预测测试样本的标签。

KNN算法使用距离函数来度量样本之间的相似性。常用的<span style="background-color: #8c30b9ff; padding: 2px 4px; border-radius: 3px; color: #333;">距离度量</span>包括：

::: tip 常用距离
欧几里得距离（最常用）
$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

曼哈顿距离
$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

闵可夫斯基距离（通用形式）
$$
d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$
- 当p=1时，为曼哈顿距离
- 当p=2时，为欧几里得距离

:::

**算法的计算步骤：**
1. **计算距离**：计算测试样本与所有训练样本之间的距离
2. **排序**：将距离按升序排列，选择距离最小的K个样本
3. **投票/平均**：
   - **分类问题**：统计K个最近邻中各类别的出现频率，选择频率最高的类别
   - **回归问题**：计算K个最近邻的目标值的平均值

K值的选择对算法性能有重要影响：
- **K值太小**：模型复杂，容易过拟合，对噪声敏感
- **K值太大**：模型简单，可能欠拟合，决策边界平滑
- **经验法则**：K通常取奇数，避免平票情况；常用交叉验证选择最优K值


简单代码实现：
```python
# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./boston.csv')

# 1. 加载数据
# 分离特征（X）和目标变量（y）——目标列默认为'MEDV'，需根据实际列名调整
X = df.drop("MEDV", axis=1)  # 所有列除了房价列，作为特征
y = df["MEDV"]     

# 2. 数据预处理：划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 测试集占20%，随机种子保证结果可复现
)

# 3. 特征标准化（KNN对特征尺度敏感，必须标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集拟合+转换
X_test_scaled = scaler.transform(X_test)        # 测试集仅转换（避免数据泄露）

# 4. 构建KNN回归模型
knn = KNeighborsRegressor(n_neighbors=5)  # 选择K=5（可调整）

# 5. 模型训练
knn.fit(X_train_scaled, y_train)

# 6. 模型预测
y_pred = knn.predict(X_test_scaled)

# 7. 模型评估
mse = mean_squared_error(y_test, y_pred)  # 均方误差
rmse = np.sqrt(mse)                       # 均方根误差
r2 = r2_score(y_test, y_pred)             # 决定系数（越接近1越好）

# 输出评估结果
print("KNN回归模型评估结果：")
print(f"均方误差(MSE)：{mse:.2f}")
print(f"均方根误差(RMSE)：{rmse:.2f}")
print(f"决定系数(R²)：{r2:.2f}")

```  
```plaintext
KNN回归模型评估结果：
均方误差(MSE)：20.61
均方根误差(RMSE)：4.54
决定系数(R²)：0.72
```

---

## 决策树

决策树是一种模仿人类决策过程的监督学习算法，可用于分类和回归任务。它通过一系列规则构建树状结构，实现对数据的分类或预测。

**核心思想**：通过递归地将数据集划分为更纯净的子集，构建一个树形结构来做出决策。


**特征选择标准**

**熵的定义**：信息熵衡量数据的不确定性：
$$
H(D) = -\sum_{i=1}^{n} p_i \log_2 (p_i)
$$
$p_i$ 是类别 $i$ 在数据集 $D$ 中的比例,$n$ 是类别的数量。

**信息增益**：信息增益 = 父节点的熵 - 子节点的加权平均熵,**ID3算法**使用信息增益作为特征选择标准。

$$
Gain(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)
$$

其中：

$D$：当前数据集，$A$：选择的特征，$V$：特征 $A$ 的取值个数，$D_v$：特征 $A$ 取值为 $v$ 的子集  
$|D_v|$：子集 $D_v$ 的样本数，$|D|$：总样本数

**信息增益率** = 信息增益 / 分裂信息，**C4.5算法**使用信息增益率防止偏向取值多的特征

$$GainRatio(D,A) = \frac{Gain(D,A)}{SplitInfo(D,A)}$$


**分裂信息（固有值）：**

$$SplitInfo(D,A) = -\sum_{v=1}^V \frac{|D_v|}{|D|} \log_2\left( \frac{|D_v|}{|D|} \right)$$


**基尼系数的定义**：基尼系数衡量数据的不纯度，**CART算法**使用基尼系数作为分裂标准
$$
Gini(D) = 1 - \sum_{i=1}^{n} p_i^2
$$

**基尼增益**：
$$
\Delta Gini = Gini(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} Gini(D_v)
$$


**停止条件**  ：决策树构建过程中需要设置停止条件防止过拟合：

- **最小样本数**：节点包含的样本数少于设定值
- **最大深度**：树达到预设的最大深度
- **纯度阈值**：节点的纯度达到设定阈值
- **最小信息增益**：分裂带来的信息增益小于设定值  


**决策树的剪枝**：为了减少过拟合，决策树需要进行剪枝处理：

1. 预剪枝：在构建过程中提前停止树的生长。
2. 后剪枝：先构建完整的树，然后自底向上剪去不必要的子树。
3. 代价复杂度剪枝：
$$
R_\alpha(T) = R(T) + \alpha|T|
$$
其中 $R(T)$ 是误差率，$|T|$ 是叶节点数，$\alpha$ 是复杂度参数。


**分类问题示例（鸢尾花数据集）**：

```python
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
feature_names = iris.feature_names
y = iris.target  # 目标：鸢尾花种类（0: setosa, 1: versicolor, 2: virginica）
target_names = iris.target_names

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 创建决策树分类器,决策树回归器用DecisionTreeRegressor类
dtree = DecisionTreeClassifier(
    criterion='gini',      # 分裂标准：基尼系数
    max_depth=3,          # 最大深度，防止过拟合
    min_samples_split=5,  # 内部节点最小样本数
    min_samples_leaf=2,   # 叶节点最小样本数
    random_state=42
)

# 4. 训练模型
dtree.fit(X_train, y_train)

# 5. 预测
y_pred = dtree.predict(X_test)

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树分类准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))
```
