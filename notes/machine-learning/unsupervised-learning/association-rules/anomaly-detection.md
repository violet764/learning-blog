# 异常检测

## 概述

识别不符合预期模式的数据点，也称为离群点检测。在数据挖掘和机器学习中具有重要应用。

## 核心概念

### 异常定义
- **全局异常**：与整个数据集相比显著不同的点
- **局部异常**：在局部邻域内显著不同的点
- **上下文异常**：在特定上下文中异常的点

### 异常类型
- **点异常**：单个数据点的异常
- **集体异常**：一组相关数据点的异常
- **上下文异常**：在特定情境下的异常

## 主要方法

### 统计方法

#### 3σ原则（三标准差原则）
对于正态分布数据，99.7%的数据落在均值±3σ范围内。

$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.9973$

#### 箱线图检测
使用四分位数识别异常：
- 下四分位数Q1
- 上四分位数Q3
- 四分位距IQR = Q3 - Q1
- 异常点：< Q1 - 1.5×IQR 或 > Q3 + 1.5×IQR

#### 马氏距离
考虑数据相关性的距离度量：

$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$

### 机器学习方法

#### 孤立森林（Isolation Forest）
通过随机划分快速隔离异常点。

#### 局部异常因子（LOF）
基于局部密度比较的异常检测。

#### 一类SVM（One-Class SVM）
在特征空间中找到包含大部分数据的最小超球面。

## Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from scipy import stats

def statistical_anomaly_detection(X, method='zscore', threshold=3):
    """
    统计方法异常检测
    """
    if method == 'zscore':
        # Z-score方法
        z_scores = np.abs(stats.zscore(X))
        anomalies = np.any(z_scores > threshold, axis=1)
    
    elif method == 'iqr':
        # 箱线图方法
        anomalies = np.zeros(X.shape[0], dtype=bool)
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_anomalies = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
            anomalies = anomalies | column_anomalies
    
    elif method == 'mahalanobis':
        # 马氏距离方法
        cov = np.cov(X.T)
        inv_cov = np.linalg.pinv(cov)
        mean = np.mean(X, axis=0)
        
        anomalies = np.zeros(X.shape[0], dtype=bool)
        for i, x in enumerate(X):
            diff = x - mean
            mahalanobis_dist = np.sqrt(diff @ inv_cov @ diff.T)
            if mahalanobis_dist > threshold:
                anomalies[i] = True
    
    return anomalies

def isolation_forest_anomaly(X, contamination=0.1):
    """
    孤立森林异常检测
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)
    
    # 将预测转换为布尔数组（-1表示异常）
    anomalies = predictions == -1
    
    return anomalies, iso_forest

def lof_anomaly(X, n_neighbors=20, contamination=0.1):
    """
    局部异常因子检测
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(X)
    
    anomalies = predictions == -1
    
    return anomalies, lof

def one_class_svm_anomaly(X, nu=0.1, kernel='rbf', gamma='scale'):
    """
    一类SVM异常检测
    """
    oc_svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    predictions = oc_svm.fit_predict(X)
    
    anomalies = predictions == -1
    
    return anomalies, oc_svm

def anomaly_detection_comparison():
    """
    比较不同异常检测算法
    """
    # 生成包含异常的数据
    np.random.seed(42)
    
    # 正常数据
    X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, 
                            center_box=(0, 0), random_state=42)
    
    # 异常数据
    X_anomaly = np.random.uniform(low=-8, high=8, size=(20, 2))
    
    # 合并数据
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 300 + [1] * 20)  # 0:正常, 1:异常
    
    # 应用不同异常检测方法
    methods = [
        ('Z-score', statistical_anomaly_detection(X, 'zscore', 2.5)),
        ('IQR', statistical_anomaly_detection(X, 'iqr')),
        ('Isolation Forest', isolation_forest_anomaly(X, 0.1)[0]),
        ('LOF', lof_anomaly(X, 20, 0.1)[0]),
        ('One-Class SVM', one_class_svm_anomaly(X, 0.1)[0])
    ]
    
    # 计算评估指标
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    results = []
    for name, anomalies in methods:
        precision = precision_score(y, anomalies)
        recall = recall_score(y, anomalies)
        f1 = f1_score(y, anomalies)
        
        results.append({
            'Method': name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # 原始数据
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Normal', alpha=0.6)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Anomaly', alpha=0.8)
    axes[0].set_title('Original Data')
    axes[0].legend()
    
    # 各方法检测结果
    for i, (name, anomalies) in enumerate(methods, 1):
        axes[i].scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', label='Normal', alpha=0.6)
        axes[i].scatter(X[anomalies, 0], X[anomalies, 1], c='red', label='Detected Anomaly', alpha=0.8)
        axes[i].set_title(f'{name}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印评估结果
    print("\n异常检测算法比较:")
    print("-" * 50)
    for result in results:
        print(f"{result['Method']:15} Precision: {result['Precision']:.3f} "
              f"Recall: {result['Recall']:.3f} F1-Score: {result['F1-Score']:.3f}")
    
    return results

def time_series_anomaly_detection():
    """
    时间序列异常检测示例
    """
    # 生成时间序列数据
    t = np.linspace(0, 100, 1000)
    signal = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)
    
    # 添加异常点
    anomaly_indices = [200, 500, 800]
    signal[anomaly_indices] = [3, -2, 4]  # 明显偏离正常范围的异常值
    
    # 使用滑动窗口检测异常
    window_size = 50
    anomalies = np.zeros(len(signal), dtype=bool)
    
    for i in range(window_size, len(signal) - window_size):
        window = signal[i-window_size:i+window_size]
        
        # 计算Z-score
        z_score = np.abs((signal[i] - np.mean(window)) / np.std(window))
        
        if z_score > 3:  # 3σ阈值
            anomalies[i] = True
    
    # 可视化时间序列异常检测
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'b-', label='Time Series', alpha=0.7)
    plt.scatter(t[anomalies], signal[anomalies], c='red', s=50, 
               label='Detected Anomalies')
    plt.scatter(t[anomaly_indices], signal[anomaly_indices], c='orange', 
               s=100, marker='x', label='True Anomalies')
    plt.title('Time Series Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 比较不同异常检测算法
    print("异常检测算法比较演示:")
    results = anomaly_detection_comparison()
    
    # 时间序列异常检测
    print("\n时间序列异常检测演示:")
    time_series_anomaly_detection()
    
    # 实际应用示例：信用卡欺诈检测
    print("\n信用卡欺诈检测示例:")
    
    # 模拟信用卡交易数据
    np.random.seed(42)
    n_transactions = 1000
    
    # 正常交易特征（金额，频率等）
    normal_amounts = np.random.lognormal(mean=3, sigma=1, size=n_transactions-20)
    normal_freq = np.random.poisson(lam=5, size=n_transactions-20)
    
    # 欺诈交易特征
    fraud_amounts = np.random.lognormal(mean=6, sigma=2, size=20)  # 金额异常大
    fraud_freq = np.random.poisson(lam=20, size=20)  # 频率异常高
    
    # 合并数据
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    frequencies = np.concatenate([normal_freq, fraud_freq])
    
    X_fraud = np.column_stack([amounts, frequencies])
    y_fraud = np.array([0]*(n_transactions-20) + [1]*20)
    
    # 使用孤立森林检测欺诈
    fraud_anomalies, model = isolation_forest_anomaly(X_fraud, contamination=0.02)
    
    from sklearn.metrics import classification_report
    print("\n欺诈检测结果:")
    print(classification_report(y_fraud, fraud_anomalies, 
                               target_names=['Normal', 'Fraud']))
```

## 数学基础

### 统计异常检测

#### Z-score标准化
$z = \frac{x - \mu}{\sigma}$

其中μ是均值，σ是标准差。

#### 马氏距离公式
$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$

### 机器学习算法

#### 孤立森林原理
异常点更容易被随机划分隔离，路径长度较短。

异常分数：$s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}$

其中c(n)是平均路径长度。

#### 局部异常因子（LOF）

**可达距离**：$reach-dist_k(p, o) = max(k-distance(o), d(p, o))$

**局部可达密度**：$lrd_k(p) = 1 / \left( \frac{\sum_{o \in N_k(p)} reach-dist_k(p, o)}{|N_k(p)|} \right)$

**局部异常因子**：$LOF_k(p) = \frac{\sum_{o \in N_k(p)} \frac{lrd_k(o)}{lrd_k(p)}}{|N_k(p)|}$

## 评估指标

### 混淆矩阵
- **真阳性（TP）**：正确识别的异常
- **假阳性（FP）**：正常点被误判为异常
- **真阴性（TN）**：正确识别的正常点
- **假阴性（FN）**：异常点被漏判

### 常用指标
- **精确率**：$Precision = \frac{TP}{TP + FP}$
- **召回率**：$Recall = \frac{TP}{TP + FN}$
- **F1分数**：$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **AUC-ROC**：接收者操作特征曲线下面积

## 应用场景

### 欺诈检测
- 信用卡欺诈
- 保险欺诈
- 网络欺诈

### 系统监控
- 网络入侵检测
- 服务器性能监控
- 应用异常检测

### 质量控制
- 制造过程监控
- 产品缺陷检测
- 流程异常检测

### 医疗诊断
- 疾病异常检测
- 医疗影像分析
- 生理信号监测

## 挑战与解决方案

### 数据不平衡
- 异常样本稀少
- 使用过采样或欠采样技术
- 调整分类阈值

### 概念漂移
- 数据分布随时间变化
- 使用在线学习算法
- 定期更新模型

### 高维数据
- 维度灾难问题
- 使用降维技术
- 特征选择方法

## 最佳实践

### 数据预处理
- 处理缺失值
- 特征标准化
- 处理类别特征

### 模型选择
- 根据数据类型选择算法
- 考虑计算复杂度
- 评估模型可解释性

### 参数调优
- 使用交叉验证
- 考虑业务需求
- 平衡精确率和召回率

## 新兴技术

### 深度学习异常检测
- 自编码器
- 生成对抗网络
- 深度信念网络

### 时间序列异常检测
- LSTM网络
- 注意力机制
- 变分自编码器

### 图异常检测
- 图神经网络
- 社区检测
- 节点异常检测