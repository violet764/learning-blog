# 逻辑回归的损失函数与梯度下降求解

## 一、模型定义

逻辑回归的目标是预测二分类结果（$y \in \{0,1\}$），其模型公式定义如下：
$$
h_w(x) = p(y = 1|x) = \sigma(w^T x + b)
$$

### 公式各部分说明：
- $x$：输入特征向量（$x \in \mathbb{R}^n$）
- $w$：权重向量（$w \in \mathbb{R}^n$，用于衡量不同特征对预测结果的影响程度）
- $b$：偏置项（$b \in \mathbb{R}$，模型的截距，用于调整预测结果的基准）
- $\sigma(z) = \frac{1}{1+e^{-z}}$：Sigmoid函数，作用是将线性组合 $w^T x + b$ 的输出映射到 $[0,1]$ 区间

### 核心作用：
Sigmoid函数将模型的线性输出值（$w^T x + b$）转换为0到1之间的数值，因此该结果可以直接作为样本属于类别1的概率。

## 二、损失函数

逻辑回归的损失函数是**交叉熵损失函数（Cross-Entropy Loss）**，也称为对数损失函数（Log Loss），其形式如下：
$$
J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)})) \right]
$$

其中：
- $m$：训练样本的数量
- $y^{(i)}$：第 $i$ 个样本的真实标签（0或1）
- $h_w(x^{(i)})$：第 $i$ 个样本的预测概率

## 三、梯度推导过程

### 1. Sigmoid函数的导数
首先，我们需要计算Sigmoid函数的导数：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
$$
\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

证明过程：
$$
\sigma'(z) = \frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right)
= \frac{e^{-z}}{(1 + e^{-z})^2}
= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}
= \sigma(z) \cdot (1 - \sigma(z))
$$

### 2. 损失函数对单个样本的梯度
对于单个样本 $(x^{(i)}, y^{(i)})$，损失函数为：
$$
L^{(i)}(w,b) = -\left[ y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)})) \right]
$$

令 $a^{(i)} = h_w(x^{(i)}) = \sigma(z^{(i)})$，其中 $z^{(i)} = w^T x^{(i)} + b$

计算对 $z^{(i)}$ 的导数：
$$
\frac{\partial L^{(i)}}{\partial z^{(i)}} = \frac{\partial L^{(i)}}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial z^{(i)}}
$$

第一部分：
$$
\frac{\partial L^{(i)}}{\partial a^{(i)}} = -\left[ \frac{y^{(i)}}{a^{(i)}} - \frac{1 - y^{(i)}}{1 - a^{(i)}} \right]
= \frac{1 - y^{(i)}}{1 - a^{(i)}} - \frac{y^{(i)}}{a^{(i)}}
$$

第二部分（利用Sigmoid函数的导数）：
$$
\frac{\partial a^{(i)}}{\partial z^{(i)}} = a^{(i)}(1 - a^{(i)})
$$

合并：
$$
\frac{\partial L^{(i)}}{\partial z^{(i)}} = \left[ \frac{1 - y^{(i)}}{1 - a^{(i)}} - \frac{y^{(i)}}{a^{(i)}} \right] \cdot a^{(i)}(1 - a^{(i)})
= (1 - y^{(i)})a^{(i)} - y^{(i)}(1 - a^{(i)})
= a^{(i)} - y^{(i)}
$$

### 3. 对权重 $w$ 的梯度
$$
\frac{\partial L^{(i)}}{\partial w_j} = \frac{\partial L^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}
= (a^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

因此，对所有样本的平均梯度为：
$$
\frac{\partial J(w,b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

向量形式：
$$
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) x^{(i)}
$$

### 4. 对偏置 $b$ 的梯度
$$
\frac{\partial L^{(i)}}{\partial b} = \frac{\partial L^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial b}
= (a^{(i)} - y^{(i)}) \cdot 1
$$

因此：
$$
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})
$$

<div style="height: 2px; background: linear-gradient(to right, #FF85C0, #2196F3, #4CAF50); border-radius: 1px; margin: 20px 0;"></div>  

[返回](./supervised_learning.md#逻辑回归)
