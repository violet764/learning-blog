# k近邻与朴素贝叶斯

## 1. k近邻算法（KNN）

### 1.1 算法概述

k近邻是一种基于实例的学习算法，其核心思想是"物以类聚"。新样本的类别由其k个最近邻居的多数投票决定。

### 1.2 数学原理

#### 1.2.1 距离度量

**欧氏距离：**
$$ d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^p (x_{ik} - x_{jk})^2} $$

**曼哈顿距离：**
$$ d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k=1}^p |x_{ik} - x_{jk}| $$

**闵可夫斯基距离：**
$$ d(\mathbf{x}_i, \mathbf{x}_j) = \left(\sum_{k=1}^p |x_{ik} - x_{jk}|^q\right)^{1/q} $$

**余弦相似度：**
$$ \text{cosine}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|} $$

#### 1.2.2 分类规则

对于新样本$\mathbf{x}$，找到其k个最近邻居$N_k(\mathbf{x})$，然后：
$$ \hat{y} = \arg\max_{c} \sum_{\mathbf{x}_i \in N_k(\mathbf{x})} I(y_i = c) $$

#### 1.2.3 回归规则

对于回归问题：
$$ \hat{y} = \frac{1}{k} \sum_{\mathbf{x}_i \in N_k(\mathbf{x})} y_i $$

或者加权平均：
$$ \hat{y} = \frac{\sum_{\mathbf{x}_i \in N_k(\mathbf{x})} w_i y_i}{\sum_{\mathbf{x}_i \in N_k(\mathbf{x})} w_i} $$

其中权重$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$

### 1.3 算法优化

#### 1.3.1 KD树

用于加速最近邻搜索的数据结构：
- 构建时间复杂度：$O(n\log n)$
- 查询时间复杂度：$O(\log n)$（平均情况）

#### 1.3.2 Ball树

另一种高效的空间分割数据结构，特别适用于高维数据。

## 2. 朴素贝叶斯

### 2.1 算法概述

朴素贝叶斯基于贝叶斯定理，并假设特征之间条件独立。虽然这个假设在现实中往往不成立，但算法在实际应用中表现良好。

### 2.2 数学原理

#### 2.2.1 贝叶斯定理

$$ P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})} $$

#### 2.2.2 朴素贝叶斯分类器

基于条件独立性假设：
$$ P(\mathbf{x}|y) = \prod_{j=1}^p P(x_j|y) $$

因此：
$$ P(y|\mathbf{x}) \propto P(y) \prod_{j=1}^p P(x_j|y) $$

分类决策：
$$ \hat{y} = \arg\max_{y} P(y) \prod_{j=1}^p P(x_j|y) $$

### 2.3 不同数据类型的处理

#### 2.3.1 高斯朴素贝叶斯

对于连续特征，假设$P(x_j|y) \sim \mathcal{N}(\mu_{yj}, \sigma_{yj}^2)$：
$$ P(x_j|y) = \frac{1}{\sqrt{2\pi\sigma_{yj}^2}} \exp\left(-\frac{(x_j - \mu_{yj})^2}{2\sigma_{yj}^2}\right) $$

#### 2.3.2 多项式朴素贝叶斯

用于文本分类等计数数据：
$$ P(x_j|y) = \frac{N_{yj} + \alpha}{N_y + \alpha p} $$

其中$N_{yj}$是特征j在类别y中出现的次数，$N_y$是类别y的总词数。

#### 2.3.3 伯努利朴素贝叶斯

用于二值特征：
$$ P(x_j|y) = \begin{cases} 
P(x_j=1|y) & \text{if } x_j = 1 \\
1 - P(x_j=1|y) & \text{if } x_j = 0 
\end{cases} $$

## 3. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_iris, make_classification, make_blobs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# KNN分类示例
print("=== KNN分类示例 ===")
X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0, 
                          n_informative=4, n_clusters_per_class=1, random_state=42)

# 数据标准化（KNN对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 不同k值的KNN比较
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    accuracy_scores.append(scores.mean())

# 绘制k值与准确率的关系
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('k值')
plt.ylabel('交叉验证准确率')
plt.title('KNN中k值选择对性能的影响')
plt.grid(True)
plt.show()

# 选择最佳k值
best_k = k_values[np.argmax(accuracy_scores)]
print(f"最佳k值: {best_k}")

# 使用最佳k值训练模型
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_knn = best_knn.predict(X_test)

print("KNN准确率:", accuracy_score(y_test, y_pred_knn))

# KNN回归示例
print("\n=== KNN回归示例 ===")
X_reg, y_reg = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
y_reg = np.sin(X_reg[:, 0]) + np.cos(X_reg[:, 1]) + np.random.normal(0, 0.1, 300)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)
y_pred_knn_reg = knn_reg.predict(X_test_reg)

mse = np.mean((y_test_reg - y_pred_knn_reg)**2)
print("KNN回归MSE:", mse)

# 朴素贝叶斯分类示例
print("\n=== 朴素贝叶斯分类示例 ===")
# 使用鸢尾花数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# 高斯朴素贝叶斯（适用于连续特征）
gnb = GaussianNB()
gnb.fit(X_train_iris, y_train_iris)
y_pred_gnb = gnb.predict(X_test_iris)

print("高斯朴素贝叶斯准确率:", accuracy_score(y_test_iris, y_pred_gnb))
print("\n分类报告:")
print(classification_report(y_test_iris, y_pred_gnb, target_names=iris.target_names))

# 多项式朴素贝叶斯示例（文本数据）
print("\n=== 多项式朴素贝叶斯（模拟文本数据） ===")
# 模拟词频数据
X_text = np.random.poisson(lam=2, size=(1000, 100))  # 1000个文档，100个特征（词）
y_text = np.random.randint(0, 3, 1000)  # 3个类别

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state=42)

mnb = MultinomialNB()
mnb.fit(X_train_text, y_train_text)
y_pred_mnb = mnb.predict(X_test_text)

print("多项式朴素贝叶斯准确率:", accuracy_score(y_test_text, y_pred_mnb))

# 伯努利朴素贝叶斯示例（二值特征）
print("\n=== 伯努利朴素贝叶斯（二值特征） ===")
X_binary = (X_text > 0).astype(int)  # 转换为二值特征
y_binary = y_text

X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train_binary)
y_pred_bnb = bnb.predict(X_test_binary)

print("伯努利朴素贝叶斯准确率:", accuracy_score(y_test_binary, y_pred_bnb))

# 模型比较
models = {
    'KNN': best_knn,
    '高斯朴素贝叶斯': gnb,
    '多项式朴素贝叶斯': mnb,
    '伯努利朴素贝叶斯': bnb
}

print("\n=== 模型性能比较 ===")
for name, model in models.items():
    if name == 'KNN':
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    elif name == '高斯朴素贝叶斯':
        y_pred = model.predict(X_test_iris)
        acc = accuracy_score(y_test_iris, y_pred)
    else:
        if name == '多项式朴素贝叶斯':
            y_pred = model.predict(X_test_text)
            acc = accuracy_score(y_test_text, y_pred)
        else:
            y_pred = model.predict(X_test_binary)
            acc = accuracy_score(y_test_binary, y_pred)
    print(f"{name}: {acc:.4f}")

# 特征重要性分析（朴素贝叶斯）
print("\n=== 高斯朴素贝叶斯特征重要性 ===")
# 计算每个特征在每个类别下的方差倒数（方差越小，特征越重要）
feature_importance = 1 / np.array([gnb.theta_.var(axis=0), gnb.sigma_.var(axis=0)]).mean(axis=0)

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('特征索引')
plt.ylabel('重要性得分')
plt.title('朴素贝叶斯特征重要性（基于方差）')
plt.show()
```

## 4. 数学推导详解

### 4.1 KNN的偏差-方差权衡

**偏差：** 随着k增大，模型变得简单，偏差增大
**方差：** 随着k减小，模型变得复杂，方差增大

**泛化误差：**
$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise} $$

### 4.2 朴素贝叶斯的拉普拉斯平滑

为了避免零概率问题，使用拉普拉斯平滑：
$$ P(x_j|y) = \frac{N_{yj} + \alpha}{N_y + \alpha d} $$

其中$\alpha$是平滑参数，$d$是特征的可能取值数。

### 4.3 对数概率计算

为避免数值下溢，使用对数概率：
$$ \log P(y|\mathbf{x}) = \log P(y) + \sum_{j=1}^p \log P(x_j|y) - \log P(\mathbf{x}) $$

由于$P(\mathbf{x})$对所有类别相同，可以忽略：
$$ \hat{y} = \arg\max_{y} \left[\log P(y) + \sum_{j=1}^p \log P(x_j|y)\right] $$

## 5. 应用场景

### 5.1 KNN应用
- **推荐系统**：基于用户相似度的推荐
- **图像识别**：手写数字识别
- **医疗诊断**：基于病例相似度的诊断
- **地理信息系统**：空间数据分类

### 5.2 朴素贝叶斯应用
- **垃圾邮件过滤**：文本分类的经典应用
- **情感分析**：评论情感极性判断
- **文档分类**：新闻分类、主题识别
- **医疗诊断**：症状与疾病的概率关系

## 6. 优缺点分析

### 6.1 KNN优缺点
**优点：**
- 原理简单，易于理解
- 对异常值不敏感
- 无需训练阶段（惰性学习）
- 能够处理多分类问题

**缺点：**
- 计算复杂度高（需要存储所有训练数据）
- 对特征尺度敏感
- 高维数据效果差（维度灾难）
- 需要选择合适的k值和距离度量

### 6.2 朴素贝叶斯优缺点
**优点：**
- 训练和预测速度快
- 对缺失数据不敏感
- 能够处理多分类问题
- 适合高维数据

**缺点：**
- 特征独立性假设在现实中往往不成立
- 对输入数据的分布形式敏感
- 需要足够的训练数据来估计概率

## 7. 实践建议

### 7.1 KNN实践建议
1. **数据预处理**：必须进行特征标准化
2. **距离度量选择**：根据数据类型选择合适的距离函数
3. **k值选择**：使用交叉验证选择最优k值
4. **降维处理**：对高维数据考虑使用PCA等降维方法

### 7.2 朴素贝叶斯实践建议
1. **数据分布检查**：验证数据是否符合算法假设
2. **平滑参数调整**：根据数据特点调整拉普拉斯平滑参数
3. **特征选择**：移除相关性强的特征
4. **模型选择**：根据特征类型选择合适的变体

---

[下一节：无监督学习算法](../unsupervised-learning/index.md)