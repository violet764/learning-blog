# 关联规则与异常检测

关联规则挖掘和异常检测是无监督学习中的两个重要任务。关联规则用于发现数据项之间的**有趣关系**，异常检测用于识别**偏离正常模式**的数据点。

## 第一部分：关联规则挖掘

### 核心概念

关联规则挖掘最经典的场景是**购物篮分析**：哪些商品经常被一起购买？

**形式化定义**：
- 项集 $I = \{i_1, i_2, \dots, i_m\}$：所有商品的集合
- 事务 $T \subseteq I$：一次购买的商品集合
- 数据库 $D = \{T_1, T_2, \dots, T_n\}$：所有事务的集合
- 关联规则 $X \to Y$：如果购买 $X$，则可能购买 $Y$

### 评价指标

**支持度（Support）**：项集出现的频率
$$\text{support}(X) = \frac{|\{T \in D : X \subseteq T\}|}{|D|}$$

**置信度（Confidence）**：规则的可靠程度
$$\text{confidence}(X \to Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}$$

**提升度（Lift）**：规则的有效性（相对于随机情况）
$$\text{lift}(X \to Y) = \frac{\text{confidence}(X \to Y)}{\text{support}(Y)}$$

- $\text{lift} > 1$：正相关，$X$ 促进 $Y$ 的购买
- $\text{lift} = 1$：独立，无关联
- $\text{lift} < 1$：负相关，$X$ 抑制 $Y$ 的购买

---

## Apriori 算法

Apriori 是最经典的关联规则挖掘算法，基于**先验性质**进行高效剪枝。

### Apriori 性质

**频繁项集的所有非空子集也必须是频繁的**。

逆否命题：如果一个项集不频繁，则其所有超集也不频繁。

这个性质大大减少了需要检查的项集数量。

### 算法步骤

**第一阶段：挖掘频繁项集**

```
1. 扫描数据库，计算所有 1-项集的支持度
2. 删除支持度 < min_support 的项集，得到 L₁
3. 用 Lₖ 生成候选 (k+1)-项集 Cₖ₊₁
4. 扫描数据库，计算 Cₖ₊₁ 的支持度
5. 删除不频繁的候选项集，得到 Lₖ₊₁
6. 重复 3-5 直到无法生成新的频繁项集
```

**第二阶段：生成关联规则**

```
对于每个频繁项集 l：
  对于每个非空子集 s：
    生成规则 s → (l - s)
    如果置信度 ≥ min_confidence，保留规则
```

### 候选项集生成

从 $L_k$ 生成 $C_{k+1}$（连接+剪枝）：

**连接步骤**：两个 $k$-项集连接生成 $(k+1)$-项集

**剪枝步骤**：利用 Apriori 性质删除包含非频繁子集的候选项集

### 代码实现

```python
import numpy as np
from itertools import combinations

class Apriori:
    """Apriori 算法实现"""
    
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
    
    def fit(self, transactions):
        """挖掘频繁项集和关联规则"""
        self.transactions = [set(t) for t in transactions]
        self.n_transactions = len(self.transactions)
        
        # 挖掘频繁项集
        self.frequent_itemsets = self._find_frequent_itemsets()
        
        # 生成关联规则
        self.rules = self._generate_rules()
        
        return self
    
    def _find_frequent_itemsets(self):
        """挖掘所有频繁项集"""
        frequent_itemsets = {}
        
        # 获取所有单项
        items = set()
        for t in self.transactions:
            items.update(t)
        
        # 计算 1-项集支持度
        L1 = {}
        for item in items:
            support = self._calculate_support({item})
            if support >= self.min_support:
                L1[frozenset([item])] = support
        
        frequent_itemsets[1] = L1
        k = 2
        
        # 迭代生成更高阶频繁项集
        while True:
            Ck = self._generate_candidates(frequent_itemsets[k-1], k)
            
            if not Ck:
                break
            
            # 计算支持度
            Lk = {}
            for candidate in Ck:
                support = self._calculate_support(candidate)
                if support >= self.min_support:
                    Lk[candidate] = support
            
            if not Lk:
                break
            
            frequent_itemsets[k] = Lk
            k += 1
        
        return frequent_itemsets
    
    def _generate_candidates(self, prev_L, k):
        """生成候选 k-项集"""
        candidates = set()
        items = set()
        
        for itemset in prev_L.keys():
            items.update(itemset)
        
        # 生成所有 k-项集
        for combo in combinations(items, k):
            candidate = frozenset(combo)
            
            # 剪枝：检查所有 (k-1)-子集是否频繁
            valid = True
            for subset in combinations(candidate, k-1):
                if frozenset(subset) not in prev_L:
                    valid = False
                    break
            
            if valid:
                candidates.add(candidate)
        
        return candidates
    
    def _calculate_support(self, itemset):
        """计算项集支持度"""
        count = sum(1 for t in self.transactions if itemset.issubset(t))
        return count / self.n_transactions
    
    def _generate_rules(self):
        """生成关联规则"""
        rules = []
        
        # 合并所有频繁项集
        all_frequent = {}
        for k, Lk in self.frequent_itemsets.items():
            all_frequent.update(Lk)
        
        # 从 2-项集开始生成规则
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, support in self.frequent_itemsets[k].items():
                # 生成所有非空真子集作为前件
                items = list(itemset)
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # 计算置信度
                        antecedent_support = all_frequent[antecedent]
                        confidence = support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # 计算提升度
                            consequent_support = all_frequent[consequent]
                            lift = confidence / consequent_support
                            
                            rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        return sorted(rules, key=lambda x: x['lift'], reverse=True)
    
    def print_rules(self, top_n=10):
        """打印关联规则"""
        print(f"\n{'='*60}")
        print(f"关联规则（Top {top_n}）")
        print(f"{'='*60}")
        
        for i, rule in enumerate(self.rules[:top_n], 1):
            print(f"\n规则 {i}: {rule['antecedent']} → {rule['consequent']}")
            print(f"  支持度: {rule['support']:.3f}")
            print(f"  置信度: {rule['confidence']:.3f}")
            print(f"  提升度: {rule['lift']:.3f}")


# 示例：购物篮分析
def shopping_basket_demo():
    """购物篮分析示例"""
    
    # 模拟购物数据
    transactions = [
        {'牛奶', '面包', '黄油'},
        {'啤酒', '面包'},
        {'牛奶', '啤酒', '黄油'},
        {'牛奶', '面包'},
        {'啤酒', '面包'},
        {'牛奶', '啤酒'},
        {'牛奶', '面包', '黄油'},
        {'啤酒', '黄油'},
        {'牛奶', '面包'},
        {'啤酒', '面包', '黄油'},
        {'牛奶', '面包', '鸡蛋'},
        {'面包', '鸡蛋', '黄油'},
        {'牛奶', '鸡蛋'},
        {'啤酒', '鸡蛋'},
        {'牛奶', '面包', '啤酒'}
    ]
    
    # 运行 Apriori
    apriori = Apriori(min_support=0.2, min_confidence=0.5)
    apriori.fit(transactions)
    
    # 打印频繁项集
    print("频繁项集:")
    for k, Lk in apriori.frequent_itemsets.items():
        print(f"\n{k}-项集:")
        for itemset, support in Lk.items():
            print(f"  {set(itemset)}: 支持度 = {support:.3f}")
    
    # 打印关联规则
    apriori.print_rules()
    
    return apriori

# 运行演示
apriori_result = shopping_basket_demo()
```

### FP-Growth 算法

FP-Growth 是 Apriori 的改进，**不需要生成候选项集**，效率更高。

**核心思想**：
1. 构建 FP 树（频繁模式树）压缩数据
2. 从 FP 树直接挖掘频繁模式

**优点**：
- 只需扫描数据库两次
- 不生成候选项集
- 对于稠密数据效率高

---

## 第二部分：异常检测

异常检测是识别**与大多数数据显著不同**的数据点的过程。

### 异常类型

| 类型 | 描述 | 示例 |
|------|------|------|
| 点异常 | 单个数据点异常 | 信用卡欺诈交易 |
| 上下文异常 | 在特定上下文中异常 | 夏天穿棉衣 |
| 集体异常 | 一组数据点集体异常 | ECG 信号中的异常波形 |

### 异常定义

- **全局异常**：与整个数据集相比显著不同
- **局部异常**：在局部邻域内显著不同

---

## 统计方法

### Z-Score 方法

假设数据服从正态分布，异常点是偏离均值超过 $k$ 个标准差的点。

$$z = \frac{x - \mu}{\sigma}$$

通常取 $k=3$（3σ原则），覆盖 99.7% 的正常数据。

### IQR 方法（箱线图）

使用四分位数识别异常：

$$\text{下界} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{上界} = Q_3 + 1.5 \times \text{IQR}$$

其中 $\text{IQR} = Q_3 - Q_1$

### 马氏距离

考虑特征相关性的距离度量：

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

### 代码实现

```python
from scipy import stats

def statistical_anomaly_detection(X, method='zscore', threshold=3):
    """统计方法异常检测"""
    
    if method == 'zscore':
        # Z-score 方法
        z_scores = np.abs(stats.zscore(X, axis=0))
        anomalies = np.any(z_scores > threshold, axis=1)
    
    elif method == 'iqr':
        # IQR 方法
        anomalies = np.zeros(X.shape[0], dtype=bool)
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            anomalies |= (X[:, i] < lower) | (X[:, i] > upper)
    
    elif method == 'mahalanobis':
        # 马氏距离
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        inv_cov = np.linalg.pinv(cov)
        
        distances = []
        for x in X:
            diff = x - mean
            d = np.sqrt(diff @ inv_cov @ diff)
            distances.append(d)
        
        distances = np.array(distances)
        threshold = np.percentile(distances, 97.5)  # 取 97.5% 分位数
        anomalies = distances > threshold
    
    return anomalies
```

---

## 孤立森林（Isolation Forest）

孤立森林基于一个简单而强大的思想：**异常点更容易被孤立**。

### 核心原理

- 正常点：密集区域，需要很多次划分才能孤立
- 异常点：稀疏区域，只需少量划分就能孤立

**路径长度**：从根节点到叶子节点的边数。异常点路径短。

### 异常分数

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

其中：
- $h(x)$：路径长度
- $c(n)$：二叉搜索树的平均路径长度（归一化因子）
- $E(h(x))$：多棵树的平均路径长度

**分数解释**：
- $s \approx 1$：异常
- $s \approx 0.5$：正常
- $s < 0.5$：非常正常

### 代码实现

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_demo():
    """孤立森林演示"""
    
    # 生成数据（含异常）
    np.random.seed(42)
    X_normal = np.random.randn(300, 2)
    X_anomaly = np.random.uniform(low=-6, high=6, size=(20, 2))
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0]*300 + [1]*20)
    
    # 孤立森林
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    
    # 转换标签：-1 为异常
    anomalies = y_pred == -1
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', alpha=0.5, label='正常')
    plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', marker='x', s=100, label='异常')
    plt.title('孤立森林异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 评估
    from sklearn.metrics import classification_report
    print(classification_report(y_true, anomalies, target_names=['正常', '异常']))
    
    return iso_forest

iso_model = isolation_forest_demo()
```

---

## 局部异常因子（LOF）

LOF 基于**局部密度**比较，能识别**局部异常**。

### 核心概念

**k-距离**：点到第 k 个最近邻的距离

**可达距离**：
$$\text{reach-dist}_k(p, o) = \max\{k\text{-distance}(o), d(p, o)\}$$

**局部可达密度**：
$$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$

**局部异常因子**：
$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$

**解释**：
- $\text{LOF} \approx 1$：与邻居密度相似，正常
- $\text{LOF} > 1$：比邻居密度低，异常

### 代码实现

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_demo():
    """LOF 演示"""
    
    # 生成不同密度的数据
    np.random.seed(42)
    
    # 密集簇
    X_dense = np.random.randn(100, 2) * 0.3
    
    # 稀疏簇
    X_sparse = np.random.randn(50, 2) * 0.5 + [3, 3]
    
    # 局部异常（在稀疏簇中）
    X_local_anomaly = np.random.randn(5, 2) * 0.1 + [3, 3]
    
    X = np.vstack([X_dense, X_sparse, X_local_anomaly])
    
    # LOF 检测
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred = lof.fit_predict(X)
    
    anomalies = y_pred == -1
    
    # 获取 LOF 分数
    lof_scores = -lof.negative_outlier_factor_
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', alpha=0.5, label='正常')
    plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', marker='x', s=100, label='异常')
    plt.title('LOF 异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # LOF 分数分布
    plt.subplot(1, 2, 2)
    plt.hist(lof_scores, bins=30, edgecolor='black')
    plt.axvline(x=1.5, color='red', linestyle='--', label='阈值')
    plt.xlabel('LOF 分数')
    plt.ylabel('频数')
    plt.title('LOF 分数分布')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return lof

lof_model = lof_demo()
```

---

## 一类 SVM（One-Class SVM）

一类 SVM 在特征空间中找到**包含大部分数据的最小超球面**。

### 原理

寻找超平面将数据与原点分开：

$$\min_{\mathbf{w}, \rho, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho$$

约束：$\mathbf{w}^T \phi(\mathbf{x}_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$

其中 $\nu \in (0, 1)$ 控制异常点比例的上界。

### 代码实现

```python
from sklearn.svm import OneClassSVM

def ocsvm_demo():
    """一类 SVM 演示"""
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(200, 2)
    X_test = np.vstack([
        np.random.randn(50, 2),
        np.random.uniform(low=-4, high=4, size=(10, 2))
    ])
    
    # 训练
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(X)
    
    # 预测
    y_pred = ocsvm.predict(X_test)
    anomalies = y_pred == -1
    
    # 可视化决策边界
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues_r')
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    
    plt.scatter(X_test[~anomalies, 0], X_test[~anomalies, 1], c='green', alpha=0.5, label='正常')
    plt.scatter(X_test[anomalies, 0], X_test[anomalies, 1], c='red', marker='x', s=100, label='异常')
    
    plt.title('一类 SVM 异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return ocsvm

ocsvm_model = ocsvm_demo()
```

---

## 方法比较与选择

### 综合对比

| 方法 | 时间复杂度 | 可解释性 | 高维适用 | 局部异常 | 新数据 |
|------|-----------|---------|---------|---------|--------|
| Z-Score | $O(n)$ | 高 | 中 | ✗ | ✓ |
| 孤立森林 | $O(n)$ | 中 | ✓ | ✓ | ✓ |
| LOF | $O(n^2)$ 或 $O(n \log n)$ | 中 | 中 | ✓ | ✗ |
| 一类 SVM | $O(n^2 \sim n^3)$ | 低 | ✓ | ✗ | ✓ |

### 选择指南

```
数据特点                    推荐方法
──────────────────────────────────────────────
单变量、正态分布            Z-Score
多变量、无分布假设          孤立森林
需要检测局部异常            LOF
已知正常数据、有监督思维    一类 SVM
高维数据                    孤立森林 / 一类 SVM
需要快速训练                孤立森林 / 统计方法
```

---

## 实战案例：信用卡欺诈检测

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def fraud_detection_demo():
    """信用卡欺诈检测演示"""
    
    # 模拟信用卡交易数据
    np.random.seed(42)
    n_normal = 10000
    n_fraud = 100
    
    # 正常交易
    X_normal = np.random.randn(n_normal, 10)
    
    # 欺诈交易（异常模式）
    X_fraud = np.random.randn(n_fraud, 10) * 3 + 5
    
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0]*n_normal + [1]*n_fraud)  # 0: 正常, 1: 欺诈
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 只用正常数据训练
    X_train_normal = X_train[y_train == 0]
    
    # 方法对比
    methods = {
        'Isolation Forest': IsolationForest(contamination=0.01, random_state=42),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True),
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.01)
    }
    
    print("=" * 50)
    print("信用卡欺诈检测结果对比")
    print("=" * 50)
    
    for name, model in methods.items():
        # 训练
        model.fit(X_train_normal)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred = (y_pred == -1).astype(int)  # 转换为 0/1
        
        # 评估
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred, target_names=['正常', '欺诈']))
    
    return methods

fraud_models = fraud_detection_demo()
```

---

## 小结

### 关联规则

- **Apriori**：基于先验性质，通过剪枝提高效率
- **FP-Growth**：构建 FP 树，无需候选项集
- **应用**：购物篮分析、推荐系统、网络日志分析

### 异常检测

- **统计方法**：简单快速，适合单变量或有分布假设
- **孤立森林**：高效、适合高维、能检测局部异常
- **LOF**：专门检测局部异常，但计算复杂度高
- **一类 SVM**：适合有明确正常数据边界的场景

---

**上一节**：[降维技术](./dimensionality-reduction.md)  
**下一节**：[高级数据结构](./advanced-structures.md)
