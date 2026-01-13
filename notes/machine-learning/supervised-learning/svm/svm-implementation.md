# 支持向量机实现细节

## 1. SVM优化算法

### 1.1 序列最小优化（SMO）算法

SMO算法是解决SVM对偶问题的高效算法，通过分解大规模优化问题为一系列小规模子问题。

#### 1.1.1 算法思想

将大规模QP问题分解为两个变量的子问题：
$$ \min_{\alpha_i, \alpha_j} \frac{1}{2}K_{ii}\alpha_i^2 + \frac{1}{2}K_{jj}\alpha_j^2 + y_iy_jK_{ij}\alpha_i\alpha_j + \cdots $$

约束条件：
$$ y_i\alpha_i + y_j\alpha_j = -\sum_{k \neq i,j} y_k\alpha_k = \zeta $$
$$ 0 \leq \alpha_i, \alpha_j \leq C $$

#### 1.1.2 算法步骤

1. **启发式选择两个拉格朗日乘子**
2. **解析求解两个变量的优化问题**
3. **更新阈值b**
4. **重复直到收敛**

### 1.2 坐标下降法

对于线性SVM，可以使用坐标下降法进行优化：

**对偶坐标下降（DCD）：**
$$ \min_{\alpha} f(\alpha) = \frac{1}{2}\alpha^TQ\alpha - \mathbf{1}^T\alpha $$

**坐标更新：**
$$ \alpha_i^{new} = \min\left(\max\left(\alpha_i - \frac{\nabla_i f(\alpha)}{Q_{ii}}, 0\right), C\right) $$

## 2. 线性SVM的高效实现

### 2.1 PEGASOS算法

**原始目标函数：**
$$ \min_{\mathbf{w}} \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i\mathbf{w}^T\mathbf{x}_i) $$

**随机梯度下降步骤：**
1. 随机选择样本$(\mathbf{x}_t, y_t)$
2. 计算梯度：$\nabla_t = \lambda\mathbf{w} - \mathbb{I}[y_t\mathbf{w}^T\mathbf{x}_t < 1]y_t\mathbf{x}_t$
3. 更新权重：$\mathbf{w} \leftarrow \mathbf{w} - \eta_t\nabla_t$
4. 投影到球面：$\mathbf{w} \leftarrow \min\left(1, \frac{1}{\sqrt{\lambda}\|\mathbf{w}\|}\right)\mathbf{w}$

### 2.2 LibLinear实现

LibLinear是针对大规模线性分类问题的高效库，主要特点：
- 使用坐标下降法
- 支持L1和L2正则化
- 优化的数据结构
- 并行计算支持

## 3. 核SVM的实现优化

### 3.1 缓存策略

由于核矩阵计算昂贵，使用缓存存储常用核值：
- **LRU缓存**：最近最少使用策略
- **分块计算**：将核矩阵分块计算和存储
- **近似方法**：使用低秩近似减少计算量

### 3.2 收缩启发式

对于远离边界的样本，其拉格朗日乘子α不会改变，可以暂时从优化中移除：
- **主动集方法**：只优化可能改变的变量
- **收敛检测**：定期检查所有变量是否需要重新优化

## 4. 多类SVM实现

### 4.1 一对多（One-vs-Rest）

对于K类问题，训练K个二分类器：
$$ f_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + b_k $$
预测：$\hat{y} = \arg\max_{k} f_k(\mathbf{x})$

### 4.2 一对一（One-vs-One）

训练$\binom{K}{2}$个二分类器，通过投票决定最终类别。

### 4.3 直接多类SVM

**Crammer-Singer多类SVM：**
$$ \min_{\mathbf{w}_1,\dots,\mathbf{w}_K} \frac{1}{2}\sum_{k=1}^K \|\mathbf{w}_k\|^2 + C\sum_{i=1}^n \xi_i $$
约束：$\mathbf{w}_{y_i}^T\mathbf{x}_i - \mathbf{w}_k^T\mathbf{x}_i \geq 1 - \xi_i, \quad \forall k \neq y_i$

## 5. Python实现示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import time

# 1. 线性SVM与核SVM性能比较
print("=== 线性SVM与核SVM性能比较 ===")

# 生成不同复杂度的数据
np.random.seed(42)

# 线性可分数据
X_linear, y_linear = make_blobs(n_samples=1000, centers=2, 
                               cluster_std=0.8, random_state=42)

# 非线性数据
X_nonlinear, y_nonlinear = make_classification(n_samples=1000, n_features=2, 
                                              n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1, 
                                              class_sep=0.8, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_linear_scaled = scaler.fit_transform(X_linear)
X_nonlinear_scaled = scaler.fit_transform(X_nonlinear)

# 比较不同SVM变体
svm_variants = {
    '线性SVM': LinearSVC(random_state=42),
    '线性核SVM': SVC(kernel='linear', random_state=42),
    'RBF核SVM': SVC(kernel='rbf', random_state=42),
    '多项式核SVM': SVC(kernel='poly', degree=3, random_state=42)
}

# 测试在两类数据上的性能
datasets = [
    (X_linear_scaled, y_linear, '线性可分数据'),
    (X_nonlinear_scaled, y_nonlinear, '非线性数据')
]

results = []

for X, y, data_name in datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    for name, svm in svm_variants.items():
        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            '数据集': data_name,
            '模型': name,
            '准确率': accuracy,
            '训练时间': train_time
        })

# 显示结果
import pandas as pd
results_df = pd.DataFrame(results)
print("\n性能比较结果:")
print(results_df.to_string(index=False))

# 2. SMO算法原理演示
print("\n=== SMO算法原理演示 ===")

def simple_smo_demo(X, y, C=1.0, max_iter=1000):
    """简化的SMO算法演示"""
    n_samples, n_features = X.shape
    
    # 初始化参数
    alpha = np.zeros(n_samples)
    b = 0.0
    
    # 预计算核矩阵（这里使用线性核）
    K = X @ X.T
    
    for iteration in range(max_iter):
        num_changed = 0
        
        for i in range(n_samples):
            # 计算误差
            E_i = np.sum(alpha * y * K[:, i]) + b - y[i]
            
            # 检查KKT条件
            if (y[i] * E_i < -0.001 and alpha[i] < C) or (y[i] * E_i > 0.001 and alpha[i] > 0):
                # 随机选择第二个变量
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                
                # 计算第二个变量的误差
                E_j = np.sum(alpha * y * K[:, j]) + b - y[j]
                
                # 保存旧值
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                # 计算边界
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                # 计算eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                # 更新alpha_j
                alpha[j] = alpha[j] - y[j] * (E_i - E_j) / eta
                
                # 剪辑alpha_j
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                
                if abs(alpha[j] - alpha_j_old) < 0.00001:
                    continue
                
                # 更新alpha_i
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                
                # 更新b
                b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] \
                     - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] \
                     - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                num_changed += 1
        
        if num_changed == 0:
            break
    
    # 计算权重向量
    w = np.sum((alpha * y).reshape(-1, 1) * X, axis=0)
    
    return w, b, alpha, iteration

# 在小数据集上演示
X_small, y_small = make_blobs(n_samples=50, centers=2, random_state=42)
X_small_scaled = scaler.fit_transform(X_small)

w, b, alpha, iterations = simple_smo_demo(X_small_scaled, y_small)
print(f"SMO算法收敛所需迭代次数: {iterations}")
print(f"支持向量数量: {np.sum(alpha > 0.001)}")
print(f"权重向量范数: {np.linalg.norm(w):.4f}")

# 3. 大规模线性SVM优化
print("\n=== 大规模线性SVM优化 ===")

# 生成大规模数据
X_large, y_large = make_classification(n_samples=10000, n_features=100, 
                                      n_informative=50, random_state=42)
X_large_scaled = scaler.fit_transform(X_large)

# 比较不同优化算法的训练时间
algorithms = {
    'liblinear': LinearSVC(penalty='l2', loss='squared_hinge', 
                          dual=True, random_state=42),
    '坐标下降': LinearSVC(penalty='l1', loss='squared_hinge', 
                        dual=False, random_state=42),
    'SGD': LinearSVC(penalty='l2', loss='hinge', 
                    dual=False, random_state=42)
}

# 使用数据子集进行快速比较
X_subset = X_large_scaled[:1000]
y_subset = y_large[:1000]

X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
    X_subset, y_subset, test_size=0.3, random_state=42)

large_results = []

for name, svm in algorithms.items():
    start_time = time.time()
    svm.fit(X_train_large, y_train_large)
    train_time = time.time() - start_time
    
    y_pred = svm.predict(X_test_large)
    accuracy = accuracy_score(y_test_large, y_pred)
    
    large_results.append({
        '算法': name,
        '准确率': accuracy,
        '训练时间(秒)': train_time
    })

large_df = pd.DataFrame(large_results)
print("\n大规模数据优化算法比较:")
print(large_df.to_string(index=False))

# 4. 多类SVM实现比较
print("\n=== 多类SVM实现比较 ===")

# 生成多类数据
X_multi, y_multi = make_classification(n_samples=1000, n_features=20, 
                                      n_classes=3, n_informative=15, 
                                      random_state=42)
X_multi_scaled = scaler.fit_transform(X_multi)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi_scaled, y_multi, test_size=0.3, random_state=42)

# 不同多类策略
multi_class_strategies = {
    '一对多 (OvR)': SVC(kernel='linear', decision_function_shape='ovr', 
                       random_state=42),
    '一对一 (OvO)': SVC(kernel='linear', decision_function_shape='ovo', 
                       random_state=42),
    'Crammer-Singer': LinearSVC(multi_class='crammer_singer', 
                               random_state=42)
}

multi_results = []

for name, svm in multi_class_strategies.items():
    start_time = time.time()
    svm.fit(X_train_multi, y_train_multi)
    train_time = time.time() - start_time
    
    y_pred = svm.predict(X_test_multi)
    accuracy = accuracy_score(y_test_multi, y_pred)
    
    multi_results.append({
        '多类策略': name,
        '准确率': accuracy,
        '训练时间(秒)': train_time
    })

multi_df = pd.DataFrame(multi_results)
print("\n多类SVM策略比较:")
print(multi_df.to_string(index=False))

# 5. 超参数调优实践
print("\n=== 超参数调优实践 ===")

# 使用网格搜索优化SVM参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# 在小数据集上进行调优（避免计算时间过长）
X_tune, y_tune = make_classification(n_samples=500, n_features=10, 
                                    random_state=42)
X_tune_scaled = scaler.fit_transform(X_tune)

X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(
    X_tune_scaled, y_tune, test_size=0.3, random_state=42)

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', 
                          n_jobs=-1, verbose=1)

grid_search.fit(X_train_tune, y_train_tune)

print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 使用最佳参数训练最终模型
best_svm = grid_search.best_estimator_
y_pred_tune = best_svm.predict(X_test_tune)
final_accuracy = accuracy_score(y_test_tune, y_pred_tune)
print(f"测试集准确率: {final_accuracy:.4f}")

# 6. 支持向量分析
print("\n=== 支持向量分析 ===")

# 训练一个RBF核SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=42)
svm_rbf.fit(X_train_tune, y_train_tune)

# 分析支持向量
support_vectors = svm_rbf.support_vectors_
support_vector_indices = svm_rbf.support_
alphas = np.abs(svm_rbf.dual_coef_[0])

print(f"支持向量数量: {len(support_vectors)}")
print(f"支持向量比例: {len(support_vectors) / len(X_train_tune):.2%}")
print(f"最大拉格朗日乘子: {np.max(alphas):.4f}")
print(f"最小拉格朗日乘子: {np.min(alphas):.4f}")

# 可视化支持向量的分布
plt.figure(figsize=(10, 6))

# 使用PCA将高维数据降维到2D进行可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tune_scaled)
X_train_pca = pca.transform(X_train_tune)

# 绘制所有训练样本
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_tune, 
           cmap='viridis', alpha=0.3, label='训练样本')

# 突出显示支持向量
sv_indices_pca = pca.transform(support_vectors)
plt.scatter(sv_indices_pca[:, 0], sv_indices_pca[:, 1], 
           s=100, facecolors='none', edgecolors='red', 
           linewidths=2, label='支持向量')

plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('支持向量分布')
plt.legend()
plt.grid(True)
plt.show()

# 7. 模型复杂度与泛化能力分析
print("\n=== 模型复杂度与泛化能力分析 ===")

# 分析不同C值对模型复杂度和性能的影响
C_values = np.logspace(-3, 3, 20)
train_scores = []
test_scores = []
sv_counts = []

for C in C_values:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train_tune, y_train_tune)
    
    # 训练集和测试集准确率
    train_score = svm.score(X_train_tune, y_train_tune)
    test_score = svm.score(X_test_tune, y_test_tune)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    sv_counts.append(len(svm.support_vectors_))

# 绘制学习曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.semilogx(C_values, train_scores, 'b-', label='训练准确率')
ax1.semilogx(C_values, test_scores, 'r-', label='测试准确率')
ax1.set_xlabel('正则化参数 C')
ax1.set_ylabel('准确率')
ax1.set_title('准确率 vs 正则化参数')
ax1.legend()
ax1.grid(True)

ax2.semilogx(C_values, sv_counts, 'g-')
ax2.set_xlabel('正则化参数 C')
ax2.set_ylabel('支持向量数量')
ax2.set_title('模型复杂度 vs 正则化参数')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 6. 实现优化技巧

### 6.1 内存优化

- **稀疏矩阵表示**：对于稀疏数据，使用稀疏矩阵格式
- **核缓存策略**：实现高效的核值缓存机制
- **分批处理**：对于超大规模数据，使用分批训练

### 6.2 计算优化

- **并行计算**：利用多核CPU进行并行优化
- **GPU加速**：对于线性SVM，可以使用GPU进行矩阵运算
- **近似方法**：使用随机特征或Nyström方法近似核矩阵

### 6.3 数值稳定性

- **正则化处理**：避免矩阵奇异性问题
- **数值精度**：使用双精度浮点数提高数值稳定性
- **收敛检测**：实现鲁棒的收敛判断条件

## 7. 实际应用考虑

### 7.1 数据预处理

- **特征标准化**：SVM对特征尺度敏感，必须进行标准化
- **异常值处理**：异常值可能影响决策边界
- **类别平衡**：对于不平衡数据，考虑类别权重

### 7.2 模型选择

- **线性vs非线性**：根据数据特性选择合适的核函数
- **参数调优**：使用交叉验证选择最优参数
- **模型解释**：线性SVM更容易解释，核SVM需要特殊技术

### 7.3 部署考虑

- **预测速度**：线性SVM预测速度快，适合实时应用
- **模型大小**：支持向量数量影响模型存储大小
- **增量学习**：考虑在线学习或增量更新需求

---

[下一节：决策树算法](../tree-models/decision-trees.md)