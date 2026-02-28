# 特征选择

特征选择是从原始特征集合中选择最相关特征子集的过程，旨在提高模型性能、减少过拟合、增强模型可解释性。

## 核心概念

特征选择是机器学习中的关键步骤，其目标是从原始特征集中选择一个子集，使得该子集能够最大程度地表达目标变量的信息。

### 为什么需要特征选择？

特征过多会带来一系列问题，影响模型的性能和可解释性：

| 问题 | 影响 |
|------|------|
| 维度灾难 | 特征过多导致样本在特征空间中变得稀疏，模型难以学习 |
| 过拟合风险 | 噪声特征会干扰模型学习，导致模型在训练集上表现好但泛化能力差 |
| 计算成本 | 训练和预测时间随特征数增加而显著增长 |
| 可解释性 | 特征过多使得难以理解模型的决策逻辑 |

### 方法分类

根据特征选择过程与模型训练的关系，可以分为三类：

| 类型 | 原理 | 特点 |
|------|------|------|
| **过滤法（Filter）** | 基于统计特性评分，独立于模型 | 计算快速、不依赖特定模型，但可能忽略特征间的交互 |
| **包裹法（Wrapper）** | 基于模型性能选择特征子集 | 准确度高，但计算量大、容易过拟合 |
| **嵌入法（Embedded）** | 训练过程中自动进行特征选择 | 平衡了效率和准确性，但依赖于特定模型 |

---

## 一、过滤法（Filter Methods）

过滤法是最简单快速的特征选择方法。它基于特征的统计特性（如方差、相关性、互信息等）对特征进行评分和排序，然后选择得分最高的特征。这种方法独立于后续的学习算法，计算效率高。

### 1. 方差选择法

移除方差低于阈值的特征：方差过低意味着该特征变化很小，信息量有限

**核心思想**：如果一个特征在所有样本中的值几乎相同（方差接近0），那么它对区分不同样本几乎没有贡献。

```python
from sklearn.feature_selection import VarianceThreshold

# 移除方差低于0.01的特征
# 方差小的特征对模型的贡献通常较小
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# 查看被保留的特征索引
# get_support()返回布尔数组，True表示该特征被保留
selected_features = selector.get_support(indices=True)
```

**适用场景**：初步筛选，快速移除变化很小的特征，减少后续处理的特征数量

### 2. 相关系数法

相关系数衡量特征与目标变量之间的线性相关程度。皮尔逊相关系数的值在[-1, 1]之间，绝对值越大表示相关性越强。

$$r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$$

**注意**：相关系数只能检测线性关系，对于非线性关系效果有限。例如，特征X与目标Y呈二次函数关系时，相关系数可能接近0。

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# 分类问题：使用F检验（ANOVA）
# F检验衡量特征在不同类别间的方差差异
# k=10: 选择得分最高的10个特征
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 回归问题：使用F回归检验
# 衡量特征与目标变量之间的线性关系强度
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# 查看特征得分：得分越高，特征越重要
scores = selector.scores_
```

### 3. 卡方检验

卡方检验是用于检验类别变量之间独立性的统计方法。在特征选择中，它检验特征与目标变量是否独立。如果不独立，说明特征与目标相关。

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

其中O是观测频数，E是期望频数。卡方值越大，说明特征与目标越相关。

**适用条件**：特征必须是非负值（如词频、计数），常用于文本分类。

```python
from sklearn.feature_selection import chi2

# 卡方检验要求数据必须非负
# 常用于文本分类中的词频特征选择
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)  # X必须非负
```

### 4. 互信息法

互信息（Mutual Information）衡量两个变量之间的信息共享程度。与相关系数不同，互信息能够捕捉非线性关系。

$$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

**直观理解**：互信息表示知道特征X后，对目标Y的不确定性减少了多少。互信息为0表示X和Y独立，值越大表示关系越强。

**优点**：能够发现任意类型的关系（包括非线性）
**缺点**：计算复杂度较高，需要估计概率分布

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# 分类问题的互信息
# 能捕捉特征与目标之间的非线性关系
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### 过滤法对比

选择过滤方法时，需要根据数据类型和是否需要捕捉非线性关系来决定：

| 方法 | 能发现非线性 | 适用数据类型 | 计算复杂度 | 使用建议 |
|------|--------------|--------------|------------|----------|
| 方差选择 | - | 数值 | O(n) | 作为预处理步骤快速筛选 |
| 相关系数 | ❌ 仅线性 | 数值 | O(n) | 线性模型前适用 |
| 卡方检验 | ❌ | 非负数值/类别 | O(n) | 文本分类、类别特征 |
| 互信息 | ✅ | 任意 | O(n log n) | 需要捕捉非线性关系时 |

---

## 二、包裹法（Wrapper Methods）

包裹法将特征选择视为一个搜索问题，使用特定的机器学习算法来评估每个特征子集的性能。虽然计算成本高，但通常能获得更优的特征子集。

### 1. 递归特征消除（RFE）

RFE（Recursive Feature Elimination）是一种贪心搜索算法，通过反复训练模型并移除最不重要的特征来逐步减少特征数量。

**工作流程**：
1. 用所有特征训练模型
2. 根据模型参数（如系数绝对值或特征重要性）排名特征
3. 移除排名最低的一个或多个特征
4. 重复上述过程直到达到指定的特征数量

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFE原理：反复训练模型，每次移除最不重要的特征
# estimator: 基础模型，需要有coef_或feature_importances_属性
# n_features_to_select: 最终保留的特征数量
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# 查看特征排名：ranking_=1表示被选中，值越大排名越靠后
ranking = rfe.ranking_
# support_返回布尔数组，True表示被选中
support = rfe.support_
```

### 2. 带交叉验证的RFE（RFECV）

RFECV在RFE的基础上加入了交叉验证，可以自动确定最优的特征数量。它通过在不同特征数量下进行交叉验证，选择使模型性能最优的特征数。

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# RFECV自动确定最优特征数
# step=1: 每次迭代移除1个特征
# cv=5: 5折交叉验证
# scoring='accuracy': 使用准确率作为评估指标
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
X_selected = rfecv.fit_transform(X, y)

# 最优特征数量：交叉验证得分最高时对应的特征数
print(f"最优特征数: {rfecv.n_features_}")

# 可视化：特征数量与模型性能的关系
import matplotlib.pyplot as plt
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('特征数量')
plt.ylabel('交叉验证得分')
plt.show()
```

### 3. 前向选择与后向消除

前向选择和后向消除是两种基础的搜索策略：

**前向选择**：从空集开始，每次添加一个最能提升模型性能的特征
- 优点：适合特征数量很多时，计算量相对较小
- 缺点：一旦添加某个特征，就无法移除

**后向消除**：从全部特征开始，每次移除一个对模型性能影响最小的特征
- 优点：可以考虑特征间的组合效应
- 缺点：初始计算量大

```python
from sklearn.feature_selection import SequentialFeatureSelector

# 前向选择：从空集开始，逐步添加最能提升性能的特征
# direction='forward': 前向选择策略
sfs_forward = SequentialFeatureSelector(
    estimator, n_features_to_select=10, direction='forward', cv=5
)
X_selected = sfs_forward.fit_transform(X, y)

# 后向消除：从全部特征开始，逐步移除对性能影响最小的特征
# direction='backward': 后向消除策略
sfs_backward = SequentialFeatureSelector(
    estimator, n_features_to_select=10, direction='backward', cv=5
)
X_selected = sfs_backward.fit_transform(X, y)
```

### 包裹法对比

不同包裹法的计算复杂度和适用场景差异较大：

| 方法 | 搜索策略 | 计算复杂度 | 适用场景 |
|------|----------|------------|----------|
| RFE | 递归消除 | O(n × 训练次数) | 中等规模数据，已知目标特征数 |
| RFECV | 带CV的递归消除 | 更高（需要多次CV） | 需要自动确定最优特征数 |
| 前向选择 | 逐步添加 | O(n² × 训练次数) | 特征较少时，希望减少计算量 |
| 后向消除 | 逐步移除 | O(n² × 训练次数) | 特征较多时，考虑特征组合效应 |

---

## 三、嵌入法（Embedded Methods）

嵌入法将特征选择融入模型训练过程中，模型在训练的同时自动完成特征选择。这种方法兼顾了效率和准确性。

### 1. L1正则化（LASSO）

L1正则化通过在损失函数中添加权重绝对值惩罚项，促使不重要特征的权重变为零，从而实现自动特征选择。

$$\min_w \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \|w\|_1$$

**为什么L1能产生稀疏解**：
- L1惩罚的等高线是菱形，与损失函数的等高线容易在坐标轴上相切
- 这意味着某些特征的权重会精确地等于0
- 相比之下，L2惩罚倾向于使所有特征的权重都接近0但不等于0

```python
from sklearn.linear_model import LassoCV, LogisticRegressionCV

# 回归问题：LassoCV自动选择最优正则化系数
# penalty='l1': 使用L1正则化
# cv=5: 5折交叉验证选择最优alpha
lasso = LassoCV(cv=5)
lasso.fit(X, y)
# 系数接近0的特征可以认为不重要
selected = np.abs(lasso.coef_) > 0.001  # 非零系数对应的特征

# 分类问题
# solver='saga': 支持L1正则化的优化算法
lr = LogisticRegressionCV(penalty='l1', solver='saga', cv=5)
lr.fit(X, y)
selected = np.abs(lr.coef_) > 0.001
```

### 2. 树模型特征重要性

树模型（如决策树、随机森林、XGBoost）可以自然地给出特征重要性评分。重要性通常基于特征在分裂节点上的信息增益或基尼不纯度减少量累积计算。

```python
from sklearn.ensemble import RandomForestClassifier

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
# 原理：基于特征在分裂节点上的信息增益/基尼不纯度减少量
importance = rf.feature_importances_

# 可视化特征重要性
indices = np.argsort(importance)[::-1]  # 按重要性降序排列的索引
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), indices)
plt.title('特征重要性')
plt.show()
```

### 3. 基于模型的特征选择

SelectFromModel是一个通用的特征选择工具，可以与任何提供特征重要性或系数属性的模型配合使用。它根据阈值自动选择重要特征。

```python
from sklearn.feature_selection import SelectFromModel

# 基于随机森林的特征选择
# threshold='median': 选择重要性大于中位数的特征
# 也可以设置为具体数值，如threshold=0.01
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)
X_selected = selector.fit_transform(X, y)

# 基于L1正则化的特征选择
# L1正则化自动产生稀疏解，系数为0的特征被剔除
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='saga'),
    threshold='mean'
)
X_selected = selector.fit_transform(X, y)
```

---

## 四、高级方法

除了上述基本方法，还有一些更高级的特征选择技术，它们针对特定问题设计了更复杂的选择策略。

### 1. 最小冗余最大相关性（mRMR）

mRMR（Minimum Redundancy Maximum Relevance）是一种考虑特征间冗余性的选择方法。它不仅追求特征与目标的高相关性，还希望选中的特征之间尽可能不冗余。

$$\max_{S} \left[ \frac{1}{|S|} \sum_{f_i \in S} I(f_i; y) - \frac{1}{|S|^2} \sum_{f_i,f_j \in S} I(f_i; f_j) \right]$$

**核心思想**：
- 第一项：最大化特征与目标的平均互信息（相关性）
- 第二项：最小化特征之间的平均互信息（冗余性）
- 平衡两者，选择信息量丰富且互不重复的特征子集

```python
# 简化实现
def mrmr_selection(X, y, n_features):
    """
    mRMR特征选择简化实现
    
    参数:
        X: 特征矩阵
        y: 目标变量
        n_features: 要选择的特征数量
    
    返回:
        被选中特征的索引列表
    """
    from sklearn.metrics import mutual_info_regression
    
    n_samples, n_feats = X.shape
    
    # 计算每个特征与目标变量的互信息（相关性）
    mi_target = np.array([mutual_info_regression(X[:, i].reshape(-1, 1), y)[0] 
                          for i in range(n_feats)])
    
    selected = []      # 已选择的特征
    remaining = list(range(n_feats))  # 待选择的特征
    
    # 选择第一个特征：选择与目标相关性最大的
    selected.append(np.argmax(mi_target))
    remaining.remove(selected[0])
    
    # 贪心策略：逐步添加特征
    while len(selected) < n_features:
        best_score = -np.inf
        best_feat = None
        
        for feat in remaining:
            # 相关性：特征与目标的信息共享
            relevance = mi_target[feat]
            # 冗余性：新特征与已选特征的互信息
            redundancy = np.mean([mutual_info_regression(
                X[:, feat].reshape(-1, 1), X[:, s])[0] for s in selected])
            # mRMR得分 = 相关性 - 冗余性
            score = relevance - redundancy
            
            if score > best_score:
                best_score = score
                best_feat = feat
        
        selected.append(best_feat)
        remaining.remove(best_feat)
    
    return selected
```

### 2. 稳定性选择

稳定性选择（Stability Selection）通过自助采样（Bootstrap）来评估特征选择的稳定性。一个特征如果在多次采样的特征选择中被频繁选中，说明该特征是稳定重要的。

**核心思想**：
- 好的特征应该在不同的数据子集上都表现出重要性
- 不稳定的特征可能是噪声或与特定样本相关

```python
from sklearn.utils import resample

def stability_selection(X, y, base_selector, n_iterations=100, threshold=0.8):
    """
    稳定性选择
    
    参数:
        X: 特征矩阵
        y: 目标变量
        base_selector: 基础特征选择器
        n_iterations: 自助采样次数
        threshold: 选择阈值，被选中比例超过此值的特征被保留
    
    返回:
        stable_features: 布尔数组，True表示稳定特征
        stability_scores: 每个特征被选中的频率
    """
    n_features = X.shape[1]
    selection_counts = np.zeros(n_features)  # 记录每个特征被选中的次数
    
    for _ in range(n_iterations):
        # 自助采样：有放回地抽取样本
        X_sample, y_sample = resample(X, y, random_state=None)
        
        # 在采样数据上进行特征选择
        selector = base_selector
        selector.fit(X_sample, y_sample)
        selected = selector.get_support()  # 返回布尔数组
        
        selection_counts += selected
    
    # 计算选择频率：被选中的比例
    stability_scores = selection_counts / n_iterations
    
    # 选择稳定特征：频率超过阈值
    stable_features = stability_scores >= threshold
    
    return stable_features, stability_scores
```

### 3. Boruta算法

Boruta是一种基于随机森林的特征选择算法，其独特之处在于能够找出所有与目标相关的特征，而不仅仅是选择固定数量的特征。

**工作原理**：
1. 为每个原始特征创建"影子特征"（随机打乱原始特征的值）
2. 用所有特征（原始+影子）训练随机森林
3. 比较原始特征与影子特征的重要性
4. 重要性显著高于影子特征的原始特征被认为是重要的

```python
from boruta import BorutaPy

# Boruta算法原理：
# 1. 创建影子特征（原始特征的打乱副本）
# 2. 训练随机森林，比较原始特征与影子特征的重要性
# 3. 重要性显著高于影子特征的特征被选中

rf = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimators='auto': 自动确定迭代次数
# max_iter=100: 最大迭代次数
boruta = BorutaPy(rf, n_estimators='auto', max_iter=100, random_state=42)
boruta.fit(X, y)

# 结果解读
selected = boruta.support_        # 确认重要的特征（应保留）
tentative = boruta.support_weak_  # 可能重要的特征（可考虑保留）
```

---

## 五、方法对比与选择

### 综合对比

在实际应用中，需要根据数据规模、计算资源和模型需求选择合适的方法：

| 方法类型 | 速度 | 准确性 | 过拟合风险 | 适用场景 |
|----------|------|--------|------------|----------|
| 过滤法 | ⭐⭐⭐ 最快 | ⭐⭐ 一般 | 低 | 初步筛选、高维数据、快速原型 |
| 包裹法 | ⭐ 最慢 | ⭐⭐⭐ 最高 | 高 | 精确选择、小数据集、追求最优性能 |
| 嵌入法 | ⭐⭐ 中等 | ⭐⭐⭐ 较高 | 中 | 常规场景、平衡效率和准确性 |

### 选择指南

根据数据规模选择合适的策略：

```
数据规模?
├── 高维数据（特征数 > 1000）
│   └── 过滤法初步筛选 → 嵌入法精细选择
│       （先用方差/相关性快速降维，再用模型精细选择）
├── 中等规模（特征数 100-1000）
│   └── 嵌入法 或 包裹法
│       （可以直接使用嵌入法，或尝试RFE）
└── 低维数据（特征数 < 100）
    └── 包裹法（RFECV）获得最优子集
        （特征数少，可以承受包裹法的计算成本）
```

### 综合示例

在实际项目中，通常会组合多种方法，形成分层的特征选择策略：

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 综合特征选择管道：组合多种方法
feature_selection_pipeline = Pipeline([
    # 第1步：方差过滤（快速初步筛选）
    # 移除方差低于0.01的特征，这些特征变化很小
    ('variance', VarianceThreshold(threshold=0.01)),
    # 第2步：统计检验（进一步筛选）
    # 使用F检验选择与目标相关性最高的50个特征
    ('kbest', SelectKBest(score_func=f_classif, k=50)),
    # 第3步：基于模型的选择（精细筛选）
    # 使用随机森林选择重要性超过中位数的特征
    ('model_select', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold='median'
    ))
])

X_selected = feature_selection_pipeline.fit_transform(X, y)
```

---

## 六、最佳实践

特征选择是一个容易出错的环节，以下是一些重要的最佳实践。

### 1. 避免数据泄露

数据泄露是特征选择中最常见的错误，会导致模型在验证时表现虚高，但在实际应用中效果很差。

```python
# ❌ 错误：在全部数据上做特征选择
# 这会导致测试集信息泄露到特征选择过程中
# 模型评估结果会过于乐观，泛化能力差
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # 包含测试集

# ✅ 正确：仅在训练集上选择特征
# fit()只在训练集上学习哪些特征重要
# transform()用相同的规则变换训练集和测试集
selector = SelectKBest(k=10)
selector.fit(X_train, y_train)                # 仅在训练集上学习
X_train_selected = selector.transform(X_train)  # 变换训练集
X_test_selected = selector.transform(X_test)    # 变换测试集
```

### 2. 评估特征选择效果

特征选择后应该评估其对模型性能的影响。理想情况下，特征选择应该：
- 减少特征数量（提高效率、增强可解释性）
- 保持或提升模型性能

```python
from sklearn.model_selection import cross_val_score

def evaluate_feature_selection(X, y, selector, model):
    """
    评估特征选择效果
    
    参数:
        X: 原始特征矩阵
        y: 目标变量
        selector: 特征选择器
        model: 评估用的模型
    
    返回:
        scores_original: 使用原始特征的交叉验证得分
        scores_selected: 使用选择后特征的交叉验证得分
    """
    # 应用特征选择
    X_selected = selector.fit_transform(X, y)
    
    # 原始特征性能：使用所有特征训练模型
    scores_original = cross_val_score(model, X, y, cv=5)
    
    # 选择后性能：仅使用选中的特征训练模型
    scores_selected = cross_val_score(model, X_selected, y, cv=5)
    
    # 打印对比结果
    print(f"原始特征 ({X.shape[1]}个): {scores_original.mean():.3f} ± {scores_original.std():.3f}")
    print(f"选择后 ({X_selected.shape[1]}个): {scores_selected.mean():.3f} ± {scores_selected.std():.3f}")
    
    return scores_original, scores_selected
```

### 3. 检查清单

进行特征选择时，请确保完成以下检查：

- [ ] 了解数据特性和业务背景（哪些特征理论上应该重要）
- [ ] 先用过滤法快速筛选（移除明显的无关特征）
- [ ] 再用包裹法或嵌入法精细选择（考虑特征组合效应）
- [ ] 评估选择后模型的性能变化（是否真的提升或持平）
- [ ] 验证特征选择的稳定性（多次运行结果是否一致）
- [ ] 确保无数据泄露（仅在训练集上进行选择）

---

## 小结

| 方法 | 核心思想 | 代表算法 |
|------|----------|----------|
| 过滤法 | 统计特性评分 | 方差选择、卡方、互信息 |
| 包裹法 | 模型性能驱动 | RFE、RFECV、前向/后向选择 |
| 嵌入法 | 训练过程自动 | LASSO、树模型重要性 |

特征选择是一个迭代优化的过程，需要结合数据特性、模型需求和业务理解进行综合考量。实践中常采用"过滤法初筛 + 嵌入法/包裹法精筛"的组合策略。
