# 交叉验证与模型评估

## 1. 交叉验证技术：数学基础与理论

### 1.1 基本概念与统计原理

交叉验证是一种评估模型泛化能力的统计方法，通过将数据集划分为训练集和测试集多次来获得更稳健的性能估计。

**数学框架**：设数据集$D = \{(x_1, y_1), \dots, (x_n, y_n)\}$，学习算法$A$，损失函数$L$。

**真实风险**：$R(h) = \mathbb{E}_{(x,y) \sim P}[L(h(x), y)]$

**经验风险**：$R_{emp}(h) = \frac{1}{n} \sum_{i=1}^n L(h(x_i), y_i)$

**交叉验证目标**：估计$\mathbb{E}_{D \sim P^n}[R(h_D)]$，其中$h_D$是在数据集$D$上训练的模型。

### 1.2 常用交叉验证方法

#### 1.2.1 留出法（Hold-out）

将数据集随机划分为训练集和测试集：
- 训练集：通常70-80%
- 测试集：通常20-30%

**优点：** 计算简单快速
**缺点：** 评估结果对数据划分敏感

#### 1.2.2 k折交叉验证（k-Fold Cross Validation）

**数学定义**：将数据集$D$划分为k个大小近似相等的互斥子集$D_1, \dots, D_k$。

**算法过程**：
对于$i = 1, \dots, k$：
- 训练集：$D_{-i} = D \setminus D_i$
- 测试集：$D_i$
- 训练模型：$h_i = A(D_{-i})$
- 计算得分：$\text{Score}_i = \frac{1}{|D_i|} \sum_{(x,y) \in D_i} L(h_i(x), y)$

**交叉验证估计**：
$$ \widehat{R}_{CV}(k) = \frac{1}{k} \sum_{i=1}^k \text{Score}_i $$

**统计性质**：
- **偏差**：当$k > 1$时，k折交叉验证是有偏估计，但偏差小于留出法
- **方差**：k折交叉验证的方差通常小于留出法
- **最优k值**：实践中常用k=5或k=10，在偏差和方差间取得平衡

**数学证明**：
设$m = n/k$为每个折的大小，则：
$$ \mathbb{E}[\widehat{R}_{CV}(k)] = R(h) + O\left(\frac{1}{m}\right) $$

**方差分析**：
$$ \text{Var}(\widehat{R}_{CV}(k)) \approx \frac{1}{k} \text{Var}(\widehat{R}_{test}) + \frac{2}{k} \text{Cov}(\widehat{R}_i, \widehat{R}_j) $$

#### 1.2.3 留一法交叉验证（Leave-One-Out Cross Validation）

k折交叉验证的特例，其中k等于样本数n：
- 每次用n-1个样本训练，1个样本测试
- 重复n次

**优点：** 几乎无偏的估计
**缺点：** 计算成本高

#### 1.2.4 分层k折交叉验证（Stratified k-Fold）

保持每个折中类别比例与原始数据集一致，特别适用于不平衡数据集。

#### 1.2.5 时间序列交叉验证（Time Series Split）

对于时间序列数据，按时间顺序划分训练集和测试集，避免未来信息泄露。

## 2. 分类问题评估指标

### 2.1 混淆矩阵（Confusion Matrix）

```
               预测为正例    预测为反例
实际为正例      TP（真阳性）   FN（假阴性）
实际为反例      FP（假阳性）   TN（真阴性）
```

### 2.2 基本指标

**准确率（Accuracy）：**
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

**精确率（Precision）：**
$$ \text{Precision} = \frac{TP}{TP + FP} $$

**召回率（Recall）：**
$$ \text{Recall} = \frac{TP}{TP + FN} $$

**F1分数（F1-Score）：**
$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

### 2.3 ROC曲线与AUC：理论分析与统计性质

**ROC曲线（Receiver Operating Characteristic）**：以假阳性率（FPR）为横轴，真阳性率（TPR）为纵轴的曲线。

**数学定义**：
设分类器输出得分$s(x)$，阈值$t$，则：
$$ TPR(t) = P(s(x) > t | Y=1) $$
$$ FPR(t) = P(s(x) > t | Y=0) $$

**ROC曲线性质**：
- **完美分类器**：ROC曲线经过(0,1)点
- **随机分类器**：ROC曲线为对角线y=x
- **曲线下面积AUC**：衡量分类器整体性能

**AUC的统计解释**：
$$ AUC = P(s(X_1) > s(X_0) | Y_1=1, Y_0=0) $$
即随机选择一个正例样本的得分高于随机选择一个负例样本得分的概率。

**AUC的数学性质**：
- $AUC \in [0, 1]$
- $AUC = 0.5$：随机分类
- $AUC = 1$：完美分类
- $AUC < 0.5$：分类器性能差于随机猜测

**AUC与基尼系数关系**：
$$ Gini = 2 \times AUC - 1 $$

**AUC的方差估计**：
使用Mann-Whitney U统计量：
$$ AUC = \frac{U}{n_1 n_0} $$
其中$U = \sum_{i=1}^{n_1} \sum_{j=1}^{n_0} I(s(x_i^+) > s(x_j^-))$

### 2.4 多分类问题指标

**宏平均（Macro Average）：** 对每个类别的指标求平均
**微平均（Micro Average）：** 将所有类别的预测结果汇总后计算指标

## 3. 回归问题评估指标：统计理论与性质分析

### 3.1 基本指标的统计性质

**均方误差（MSE）**：
**数学定义**：$$ MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

**统计性质**：
- **偏差-方差分解**：$MSE = \text{Bias}^2 + \text{Variance} + \text{Noise}$
- **一致性**：当$n \to \infty$时，MSE收敛到真实风险
- **效率**：MSE是方差的无偏估计量

**数学推导**：
设真实函数$f(x)$，估计函数$\hat{f}(x)$，噪声$\epsilon \sim N(0, \sigma^2)$：
$$ \mathbb{E}[MSE] = \mathbb{E}[(f(x) - \hat{f}(x))^2] + \sigma^2 $$

**均方根误差（RMSE）**：
$$ RMSE = \sqrt{MSE} $$

**性质**：
- 与原始数据单位一致
- 对异常值敏感度低于MSE
- 平方根变换使得误差分布更接近正态分布

**平均绝对误差（MAE）**：
**数学定义**：$$ MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| $$

**统计性质**：
- **稳健性**：对异常值不敏感
- **期望值**：$\mathbb{E}[MAE] = \mathbb{E}[|\epsilon + (f(x)-\hat{f}(x))|]$
- **与中位数的关系**：MAE最小化对应中位数回归

**MSE与MAE的比较**：
- **MSE**：可微，便于优化，但对异常值敏感
- **MAE**：稳健，但不可微，优化困难
- **Huber损失**：结合MSE和MAE的优点

### 3.2 相对指标

**平均绝对百分比误差（MAPE）：**
$$ MAPE = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| $$

**决定系数（R²）：**
$$ R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$

## 4. 模型选择与超参数优化

### 4.1 网格搜索（Grid Search）

遍历所有可能的超参数组合，选择性能最好的组合。

### 4.2 随机搜索（Random Search）

从超参数空间中随机采样，相比网格搜索更高效。

### 4.3 贝叶斯优化（Bayesian Optimization）

使用贝叶斯方法构建目标函数的概率模型，智能选择下一个超参数组合。

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

# 生成示例数据
print("=== 分类问题评估 ===")
X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                       n_redundant=5, n_classes=2, random_state=42)

# 生成回归数据
print("\n=== 回归问题评估 ===")
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, n_informative=8, 
                               noise=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)
X_reg_scaled = scaler.fit_transform(X_reg)

# 1. 交叉验证比较
print("\n=== 交叉验证方法比较 ===")

# 定义分类器
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 不同交叉验证方法
cv_methods = {
    '5折交叉验证': 5,
    '10折交叉验证': 10,
    '分层5折交叉验证': StratifiedKFold(n_splits=5),
    '留一法': len(X_class_scaled)  # 计算成本高，仅用于演示
}

for name, cv in cv_methods.items():
    if name == '留一法':
        # 留一法计算成本高，这里使用5折代替演示
        scores = cross_val_score(classifier, X_class_scaled, y_class, cv=5)
    else:
        scores = cross_val_score(classifier, X_class_scaled, y_class, cv=cv)
    
    print(f"{name}: 平均准确率 = {scores.mean():.3f} ± {scores.std():.3f}")

# 2. 多指标交叉验证
print("\n=== 多指标交叉验证 ===")
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(classifier, X_class_scaled, y_class, cv=5, scoring=scoring)

for metric in scoring:
    mean_score = np.mean(scores[f'test_{metric}'])
    std_score = np.std(scores[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")

# 3. 混淆矩阵可视化
print("\n=== 混淆矩阵分析 ===")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_class_scaled, y_class, 
                                                    test_size=0.3, random_state=42)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 4. ROC曲线分析
print("\n=== ROC曲线分析 ===")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 5. 回归问题评估
print("\n=== 回归模型评估 ===")
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, 
                                                                    test_size=0.3, random_state=42)

regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)

# 计算回归指标
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"均方误差 (MSE): {mse:.3f}")
print(f"均方根误差 (RMSE): {rmse:.3f}")
print(f"平均绝对误差 (MAE): {mae:.3f}")
print(f"决定系数 (R²): {r2:.3f}")

# 可视化预测结果
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测 vs 真实值')

plt.subplot(1, 2, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')

plt.tight_layout()
plt.show()

# 6. 超参数优化
print("\n=== 超参数优化 ===")

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 随机搜索（更高效）
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 随机尝试20组参数
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("随机搜索最佳参数:", random_search.best_params_)
print("随机搜索最佳分数:", random_search.best_score_)

# 7. 学习曲线分析
print("\n=== 学习曲线分析 ===")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    classifier, X_class_scaled, y_class, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练得分")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证得分")
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend(loc="best")
plt.grid(True)
plt.show()

# 8. 模型比较
print("\n=== 多模型比较 ===")
models = {
    '逻辑回归': LogisticRegression(),
    '随机森林': RandomForestClassifier(),
    '支持向量机': SVC(probability=True)
}

results = []
for name, model in models.items():
    scores = cross_validate(model, X_class_scaled, y_class, cv=5, 
                           scoring=['accuracy', 'f1', 'roc_auc'])
    
    results.append({
        '模型': name,
        '准确率': f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
        'F1分数': f"{scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}",
        'AUC': f"{scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}"
    })

results_df = pd.DataFrame(results)
print("\n模型性能比较:")
print(results_df)
```

## 6. 实践建议

### 6.1 交叉验证选择
- **小数据集**：使用留一法或分层k折交叉验证
- **大数据集**：使用简单的留出法或5折交叉验证
- **时间序列数据**：必须使用时间序列交叉验证
- **不平衡数据**：使用分层k折交叉验证

### 6.2 评估指标选择
- **平衡数据集**：使用准确率
- **不平衡数据集**：使用F1分数、AUC
- **需要控制假阳性**：关注精确率
- **需要控制假阴性**：关注召回率

### 6.3 超参数优化
- **参数空间小**：使用网格搜索
- **参数空间大**：使用随机搜索或贝叶斯优化
- **计算资源有限**：使用较少的折数和迭代次数

### 6.4 模型诊断
- **学习曲线**：判断是否欠拟合或过拟合
- **验证曲线**：确定最佳超参数
- **残差分析**：检查模型假设是否满足

## 7. 数学原理

### 7.1 偏差-方差分解

泛化误差可以分解为：
$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

- **偏差**：模型预测值与真实值的差异
- **方差**：模型对训练数据变化的敏感度
- **不可约误差**：数据本身的噪声

### 7.2 k折交叉验证的方差

k折交叉验证的方差约为：
$$ \text{Var}(\hat{\mu}_{CV}) \approx \frac{1}{k} \text{Var}(\hat{\mu}_{test}) $$

其中$\hat{\mu}_{test}$是单次测试的估计值。

---

[下一节：特征工程与数据预处理](./feature-engineering.md)