# 朴素贝叶斯分类器

朴素贝叶斯（Naive Bayes）是一类基于贝叶斯定理的概率分类算法，其核心假设是特征之间条件独立。尽管这个假设在现实中往往不成立，但朴素贝叶斯在文本分类、垃圾邮件过滤等任务中表现出色，是机器学习中最实用的分类算法之一。

## 基本概念

### 贝叶斯定理回顾

贝叶斯定理描述了在已知某些条件下，事件发生概率的更新方式：

$$ P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})} $$

其中：
- $P(y|\mathbf{x})$：**后验概率**，给定特征 $\mathbf{x}$ 时类别为 $y$ 的概率
- $P(\mathbf{x}|y)$：**似然概率**，类别为 $y$ 时观测到特征 $\mathbf{x}$ 的概率
- $P(y)$：**先验概率**，类别 $y$ 出现的概率
- $P(\mathbf{x})$：**证据因子**，特征 $\mathbf{x}$ 出现的概率

### 朴素贝叶斯的核心假设

📌 **条件独立性假设**：假设给定类别标签时，各特征之间相互独立。

$$ P(\mathbf{x}|y) = P(x_1, x_2, \ldots, x_p|y) = \prod_{j=1}^{p} P(x_j|y) $$

这个假设大大简化了计算，使得我们无需估计指数级的联合概率分布。

### 分类决策规则

基于贝叶斯定理和条件独立性假设，分类决策规则为：

$$ \hat{y} = \arg\max_{y} P(y) \prod_{j=1}^{p} P(x_j|y) $$

由于 $P(\mathbf{x})$ 对所有类别相同，可以忽略。实际计算中，为避免数值下溢，通常使用对数概率：

$$ \hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{j=1}^{p} \log P(x_j|y) \right] $$

## 三种常见变体

根据特征类型的不同，朴素贝叶斯有不同的变体：

### 高斯朴素贝叶斯

适用于**连续特征**，假设每个特征在给定类别下服从高斯分布：

$$ P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right) $$

参数估计：
- 均值：$\mu_{c,j} = \frac{1}{n_c}\sum_{i: y_i=c} x_{ij}$
- 方差：$\sigma_{c,j}^2 = \frac{1}{n_c}\sum_{i: y_i=c} (x_{ij} - \mu_{c,j})^2$

### 多项式朴素贝叶斯

适用于**离散计数特征**（如文本分类中的词频）：

$$ P(x_j|y=c) = \frac{N_{c,j} + \alpha}{N_c + \alpha \cdot p} $$

其中：
- $N_{c,j}$：类别 $c$ 中特征 $j$ 的总计数
- $N_c$：类别 $c$ 中所有特征的总计数
- $\alpha$：平滑参数（拉普拉斯平滑）
- $p$：特征维度

### 伯努利朴素贝叶斯

适用于**二值特征**（如文档中是否出现某词）：

$$ P(x_j|y=c) = \begin{cases} \theta_{c,j} & \text{if } x_j = 1 \\ 1 - \theta_{c,j} & \text{if } x_j = 0 \end{cases} $$

其中 $\theta_{c,j}$ 是类别 $c$ 中特征 $j$ 为 1 的概率。

## 参数估计与拉普拉斯平滑

### 最大似然估计

对于离散特征，使用最大似然估计：

$$ \hat{P}(x_j=v|y=c) = \frac{\text{count}(x_j=v, y=c)}{\text{count}(y=c)} $$

### 拉普拉斯平滑

⚠️ **零概率问题**：如果某个特征值在训练集中未出现，会导致概率为 0，进而使整个乘积为 0。

**解决方案**：拉普拉斯平滑（加一平滑）

$$ \hat{P}(x_j=v|y=c) = \frac{\text{count}(x_j=v, y=c) + \alpha}{\text{count}(y=c) + \alpha \cdot |V_j|} $$

其中 $\alpha > 0$ 是平滑参数，$|V_j|$ 是特征 $j$ 的可能取值数。

## 代码示例

### 高斯朴素贝叶斯

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 训练高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 预测
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

# 评估模型
print("=" * 50)
print("高斯朴素贝叶斯 - 鸢尾花分类")
print("=" * 50)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 查看学习到的参数
print("\n各类别各特征的均值 (theta_):")
for i, name in enumerate(target_names):
    print(f"  {name}: {gnb.theta_[i]}")

print("\n各类别各特征的方差 (sigma_):")
for i, name in enumerate(target_names):
    print(f"  {name}: {gnb.sigma_[i]}")

# 可视化决策边界（使用前两个特征）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 训练模型（仅使用前两个特征便于可视化）
gnb_2d = GaussianNB()
gnb_2d.fit(X_train[:, :2], y_train)

# 创建网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 预测网格点
Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
ax1 = axes[0]
ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.set_title('高斯朴素贝叶斯决策边界')
plt.colorbar(scatter, ax=ax1)

# 混淆矩阵
ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=target_names, yticklabels=target_names)
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')
ax2.set_title('混淆矩阵')

plt.tight_layout()
plt.show()
```

### 多项式朴素贝叶斯 - 文本分类

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

# 模拟新闻文本数据
documents = [
    "股市大涨 投资者 乐观 经济 增长",           # 财经
    "股票 下跌 市场 恐慌 抛售",                  # 财经
    "银行 利率 贷款 理财 投资",                  # 财经
    "足球 比赛 冠军 球队 胜利",                  # 体育
    "运动员 奥运 金牌 训练 比赛",                # 体育
    "篮球 NBA 球星 季后赛 总冠军",               # 体育
    "科技 创新 人工智能 发展 未来",              # 科技
    "手机 新品 发布 智能 功能",                  # 科技
    "互联网 公司 软件 编程 技术",                # 科技
]

labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # 0:财经, 1:体育, 2:科技
label_names = ['财经', '体育', '科技']

# 创建管道：词频统计 -> 多项式朴素贝叶斯
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # 将文本转换为词频向量
    ('classifier', MultinomialNB(alpha=1.0))  # alpha=1.0 拉普拉斯平滑
])

# 训练模型
pipeline.fit(documents, labels)

# 测试新文档
test_docs = [
    "股票 投资 理财",
    "足球 冠军 比赛",
    "人工智能 科技 创新"
]

print("=" * 50)
print("多项式朴素贝叶斯 - 文本分类")
print("=" * 50)

test_vectors = pipeline.named_steps['vectorizer'].transform(test_docs)
predictions = pipeline.predict(test_docs)
probabilities = pipeline.predict_proba(test_docs)

for doc, pred, prob in zip(test_docs, predictions, probabilities):
    print(f"\n文档: '{doc}'")
    print(f"预测类别: {label_names[pred]}")
    print(f"类别概率: {dict(zip(label_names, prob.round(4)))}")

# 可视化词频矩阵
vectorizer = pipeline.named_steps['vectorizer']
feature_names = vectorizer.get_feature_names_out()

print(f"\n词汇表大小: {len(feature_names)}")
print(f"词汇表: {feature_names[:10]}...")  # 显示前10个词
```

### 伯努利朴素贝叶斯 - 二值特征

```python
from sklearn.naive_bayes import BernoulliNB

# 模拟文档-词项矩阵（二值：是否包含某词）
# 每行代表一个文档，每列代表一个词是否出现
X_binary = np.array([
    # 股票, 投资, 足球, 比赛, 科技, 人工智能
    [1, 1, 0, 0, 0, 0],  # 财经文档
    [1, 0, 0, 0, 0, 0],  # 财经文档
    [0, 0, 1, 1, 0, 0],  # 体育文档
    [0, 0, 1, 0, 0, 0],  # 体育文档
    [0, 0, 0, 0, 1, 1],  # 科技文档
    [0, 0, 0, 0, 1, 0],  # 科技文档
])

y_binary = np.array([0, 0, 1, 1, 2, 2])  # 0:财经, 1:体育, 2:科技

# 训练伯努利朴素贝叶斯
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_binary, y_binary)

print("=" * 50)
print("伯努利朴素贝叶斯 - 二值特征分类")
print("=" * 50)

# 测试样本
X_test_binary = np.array([
    [1, 0, 0, 0, 0, 0],  # 只包含"股票" -> 财经
    [0, 0, 1, 1, 0, 0],  # 包含"足球"和"比赛" -> 体育
    [0, 0, 0, 0, 1, 1],  # 包含"科技"和"人工智能" -> 科技
])

predictions = bnb.predict(X_test_binary)
probabilities = bnb.predict_proba(X_test_binary)

feature_names = ['股票', '投资', '足球', '比赛', '科技', '人工智能']
for features, pred, prob in zip(X_test_binary, predictions, probabilities):
    active_features = [feature_names[i] for i, v in enumerate(features) if v == 1]
    print(f"\n特征: {active_features}")
    print(f"预测类别: {label_names[pred]}")
    print(f"类别概率: {dict(zip(label_names, prob.round(4)))}")

# 查看特征对各类别的贡献
print("\n特征 log 概率 (feature_log_prob_):")
print("(数值越大，表示该特征对该类别越重要)")
print(f"{'特征':<10}", end='')
for name in label_names:
    print(f"{name:<12}", end='')
print()
for i, fname in enumerate(feature_names):
    print(f"{fname:<10}", end='')
    for c in range(3):
        print(f"{bnb.feature_log_prob_[c, i]:<12.4f}", end='')
    print()
```

### 从零实现朴素贝叶斯

```python
class MyGaussianNB:
    """从零实现高斯朴素贝叶斯分类器"""
    
    def fit(self, X, y):
        """训练模型：计算各类别的先验概率和高斯参数"""
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # 计算先验概率
        self.priors_ = np.array([
            np.sum(y == c) / n_samples for c in self.classes_
        ])
        
        # 计算各类别各特征的均值和方差
        self.means_ = np.array([
            X[y == c].mean(axis=0) for c in self.classes_
        ])
        self.vars_ = np.array([
            X[y == c].var(axis=0) for c in self.classes_
        ])
        
        return self
    
    def _gaussian_pdf(self, X, mean, var):
        """计算高斯概率密度函数"""
        return np.exp(-0.5 * (X - mean)**2 / var) / np.sqrt(2 * np.pi * var)
    
    def predict_proba(self, X):
        """预测后验概率"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probs = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # log P(y=c)
            log_prior = np.log(self.priors_[i])
            
            # log P(x|y=c) = sum log P(x_j|y=c)
            log_likelihood = np.sum(
                np.log(self._gaussian_pdf(X, self.means_[i], self.vars_[i]) + 1e-9),
                axis=1
            )
            
            log_probs[:, i] = log_prior + log_likelihood
        
        # 转换为概率（使用 softmax）
        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """预测类别"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# 测试自己实现的分类器
print("=" * 50)
print("从零实现高斯朴素贝叶斯")
print("=" * 50)

my_gnb = MyGaussianNB()
my_gnb.fit(X_train, y_train)
my_pred = my_gnb.predict(X_test)

print(f"自实现准确率: {accuracy_score(y_test, my_pred):.4f}")
print(f"sklearn准确率: {accuracy_score(y_test, gnb.predict(X_test)):.4f}")
```

## 模型特点分析

### 优点

✅ **训练速度快**：只需计算各类别的统计量，无需迭代优化  
✅ **预测效率高**：直接计算概率，时间复杂度为 $O(p \cdot |C|)$  
✅ **对小样本友好**：即使训练数据较少也能取得不错效果  
✅ **可处理多分类**：天然支持多分类问题  
✅ **对缺失数据鲁棒**：训练时可以忽略缺失特征，预测时可以跳过缺失项  
✅ **可解释性强**：概率输出直观反映分类置信度

### 缺点

⚠️ **独立性假设过强**：特征间往往存在相关性，假设不成立时性能受限  
⚠️ **对输入数据形式敏感**：需要根据特征类型选择合适的变体  
⚠️ **零频率问题**：需要平滑处理避免概率为零

### 适用场景

| 场景 | 推荐变体 | 说明 |
|------|----------|------|
| 文本分类（词频） | 多项式 NB | 词频特征天然符合多项分布 |
| 垃圾邮件过滤 | 伯努利 NB / 多项式 NB | 基于词袋模型 |
| 情感分析 | 多项式 NB | 词频或 TF-IDF 特征 |
| 连续特征分类 | 高斯 NB | 假设特征服从高斯分布 |
| 实时预测系统 | 任意变体 | 训练和预测速度快 |

## 与其他算法的比较

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 生成分类数据
X_comp, y_comp = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, n_clusters_per_class=2, random_state=42
)

X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
    X_comp, y_comp, test_size=0.3, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_comp)
X_test_scaled = scaler.transform(X_test_comp)

# 定义多个分类器
classifiers = {
    '高斯朴素贝叶斯': GaussianNB(),
    'K近邻': KNeighborsClassifier(n_neighbors=5),
    '决策树': DecisionTreeClassifier(max_depth=5),
    '支持向量机': SVC(kernel='rbf'),
    '逻辑回归': LogisticRegression(max_iter=1000)
}

print("=" * 50)
print("分类器性能比较")
print("=" * 50)

results = []
for name, clf in classifiers.items():
    # 交叉验证
    cv_scores = cross_val_score(clf, X_train_scaled, y_train_comp, cv=5)
    
    # 测试集评估
    clf.fit(X_train_scaled, y_train_comp)
    test_score = clf.score(X_test_scaled, y_test_comp)
    
    results.append({
        '算法': name,
        '交叉验证均值': cv_scores.mean(),
        '交叉验证标准差': cv_scores.std(),
        '测试集准确率': test_score
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

## 实践建议

1. **特征选择**：移除高度相关的特征，符合独立性假设
2. **平滑参数调优**：通过交叉验证选择最佳的 $\alpha$ 值
3. **特征工程**：对连续特征可尝试离散化后使用多项式 NB
4. **概率校准**：朴素贝叶斯输出的概率可能不够准确，可使用 `CalibratedClassifierCV` 进行校准
5. **集成学习**：可与其他分类器组合使用，如投票或堆叠

---

[上一节：梯度提升](../tree-models/gradient-boosting.md) | [下一节：EM算法](./em-algorithm.md)
