# 最小二乘法求解线性回归
## 1. 线性回归基本模型
线性回归旨在构建自变量与因变量之间的线性关系模型，单变量线性回归的基本形式为：
$$y = wx + b + \varepsilon$$
其中：
- $x$：自变量（特征）
- $y$：因变量（标签）
- $w$：权重系数
- $b$：偏置项
- $\varepsilon$：随机误差项（满足均值为0、方差恒定的假设）

多变量线性回归可表示为矩阵形式（更便于推导）：
$$Y = XW + \varepsilon$$
其中：
- $Y \in \mathbb{R}^{n \times 1}$：n个样本的因变量向量
- $X \in \mathbb{R}^{n \times (d+1)}$：n个样本的特征矩阵（d为特征维度，第一列全为1，对应偏置项$b$）
- $W \in \mathbb{R}^{(d+1) \times 1}$：参数向量（$[b, w_1, w_2, ..., w_d]^T$）
- $\varepsilon \in \mathbb{R}^{n \times 1}$：误差向量

## 2. 最小二乘法核心思想
最小二乘法（Ordinary Least Squares, OLS）的目标是找到最优参数$W$，使得模型预测值与真实值的**残差平方和（RSS）** 最小：
$$\min_{W} \ RSS(W) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = (Y - XW)^T(Y - XW)$$
其中$\hat{y}_i$为第$i$个样本的预测值，$\hat{Y} = XW$为预测值向量。

## 3. 参数求解（矩阵形式推导）
### 3.1 损失函数展开
将残差平方和展开：
$$
\begin{align*}
RSS(W) &= (Y - XW)^T(Y - XW) \\
&= Y^TY - W^TX^TY - Y^TXW + W^TX^TXW \\
&= Y^TY - 2W^TX^TY + W^TX^TXW
\end{align*}
$$
（注：$W^TX^TY$是标量，其转置$Y^TXW$与自身相等，故合并为$-2W^TX^TY$）

### 3.2 求导并令导数为0
对$W$求偏导（矩阵求导）：
$$\frac{\partial RSS(W)}{\partial W} = -2X^TY + 2X^TXW$$
令偏导数为0（极值条件），求解最优参数$W^*$：
$$-2X^TY + 2X^TXW = 0$$
化简得：
$$X^TXW = X^TY$$

### 3.3 最终参数解
若$X^TX$可逆（满秩，即特征无多重共线性），则：
$$W^* = (X^TX)^{-1}X^TY$$
该式即为最小二乘法求解线性回归参数的核心公式。

<div style="height: 2px; background: linear-gradient(to right, #FF85C0, #2196F3, #4CAF50); border-radius: 1px; margin: 20px 0;"></div>  

[返回](./supervised_learning.md#线性回归)