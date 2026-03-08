# 深度学习基础面试题

本章节整理了深度学习基础相关的面试题目，涵盖神经网络基础、优化器、激活函数、正则化、梯度问题等核心主题。

---

## 一、神经网络基础

### Q1: 请解释前向传播和反向传播的过程

**基础回答：**

前向传播是输入数据经过各层神经网络计算得到输出的过程，反向传播是根据损失函数计算梯度并更新参数的过程。

**深入回答：**

**前向传播**：
```
输入 x → 线性变换 z = Wx + b → 激活函数 a = σ(z) → 输出
逐层传递，最终得到预测值 ŷ，计算损失 L(ŷ, y)
```

**反向传播**：
$$
\begin{aligned}
1. &\ \text{计算输出层梯度：} \frac{\partial L}{\partial a} \\
2. &\ \text{逐层反向传递：利用链式法则 } \frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W} \\
3. &\ \text{累积梯度，更新参数}
\end{aligned}
$$

**追问：反向传播为什么高效？**

反向传播利用**动态规划**思想，每个节点的梯度只计算一次，然后复用于前序节点的梯度计算。复杂度从 O(n²) 降低到 O(n)，其中 n 是网络节点数。

**追问：什么情况下梯度无法传播？**

- 使用不可导的激活函数（如 step function）
- 激活函数导数恒为 0（如 sigmoid 在饱和区）
- 网络结构中存在断开路径（如某些特殊架构）

---

### Q2: 什么是梯度消失和梯度爆炸？如何解决？

**基础回答：**

梯度消失是指反向传播时梯度逐层衰减，导致浅层参数无法有效更新。梯度爆炸是指梯度逐层放大，导致参数更新不稳定。

**深入回答：**

**原因分析**：

以 n 层网络为例，梯度传递公式：
$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_n} \cdot \prod_{i=1}^{n} \left(\frac{\partial a_i}{\partial z_i} \cdot W_i\right)
$$

当 $\left|\frac{\partial a}{\partial z} \cdot W\right| < 1$ 时，连乘导致指数级衰减（消失）
当 $\left|\frac{\partial a}{\partial z} \cdot W\right| > 1$ 时，连乘导致指数级增长（爆炸）

**追问：具体分析 Sigmoid 为什么会导致梯度消失？**

$$
\begin{aligned}
\text{sigmoid}(x) &= \frac{1}{1+e^{-x}} \\
\text{sigmoid}'(x) &= \text{sigmoid}(x) \cdot (1 - \text{sigmoid}(x))
\end{aligned}
$$

当 $x$ 很大或很小时：
- $\text{sigmoid}(x) \to 1$ 或 $0$
- $\text{sigmoid}'(x) \to 0$（最大值仅为 0.25）

因此深层网络中，梯度经过多层 sigmoid 后会接近 0。

**解决方案总结**：

| 问题 | 解决方案 | 原理 |
|------|----------|------|
| 梯度消失 | ReLU 激活函数 | 正区间导数恒为 1 |
| 梯度消失 | 残差连接 | 提供梯度直通路径 |
| 梯度消失 | BatchNorm | 规范化激活值分布 |
| 梯度爆炸 | 梯度裁剪 | 限制梯度最大值 |
| 梯度爆炸 | 合理初始化 | Xavier/He 初始化 |
| 两者皆有 | LSTM/GRU | 门控机制控制信息流 |

**RNN 结构示意**（梯度在时间步传递）：

![RNN 结构](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91RNN.png)

---

### Q3: 为什么神经网络需要激活函数？

**基础回答：**

激活函数引入非线性，使神经网络能够学习复杂的非线性映射。没有激活函数，多层网络等价于单层线性变换。

**深入回答：**

**数学证明**：

无激活函数的多层网络：
$$
\begin{aligned}
h_1 &= W_1 x + b_1 \\
h_2 &= W_2 h_1 + b_2 = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'
\end{aligned}
$$

可以合并为一个线性变换，失去多层网络的意义。

**追问：激活函数需要满足哪些性质？**

1. **非线性**：这是最核心的要求
2. **可微性**：反向传播需要计算梯度
3. **单调性**（可选）：保证损失函数凸性
4. **输出范围有界**（可选）：防止激活值过大
5. **计算高效**：频繁调用需要快速计算

**追问：为什么 Transformer 中使用 GELU 而不是 ReLU？**


$$\text{GELU}(x) = x \cdot \Phi(x) \text{(Φ 是标准正态分布的 CDF)} $$  
$$\text{ReLU}(x) = \max(0, x)$$


- GELU 在 0 附近更平滑，梯度变化更连续
- GELU 有非单调性，允许负值有非零输出
- 实验表明 GELU 在 Transformer 架构上效果更好

---

## 二、优化器

### Q4: SGD 和 Adam 的区别是什么？各有什么优缺点？

**基础回答：**

SGD 是最基础的优化器，每次使用一个或一批样本更新参数。Adam 结合了动量和自适应学习率。

**深入回答：**

**SGD 更新规则**：
$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)
$$

**Adam 更新规则**：
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\nabla L(\theta_t) & \text{ 一阶矩估计（动量）} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(\theta_t))^2 & \text{ 二阶矩估计} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} & \text{ 偏差修正} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{aligned}
$$

**追问：Adam 为什么需要偏差修正？**

初始时 $m_0 = 0$，导致早期估计偏向 0。修正后：
- 早期：学习率被放大，加速收敛
- 后期：修正因子趋近 1，恢复正常

**追问：为什么大模型训练更常用 AdamW 而不是 Adam？**

| 优化器 | 权重衰减实现 | 问题 |
|--------|--------------|------|
| Adam | L2 正则化加到损失函数 | 与自适应学习率冲突 |
| AdamW | 直接从参数减去衰减项 | 解耦，效果更好 |

**公式对比**：

$$ \text{Adam: } \text{loss} = \text{loss} + \lambda \cdot \|\theta\|^2 $$

$$ \text{AdamW: } \theta_{t+1} = \theta_t - \eta \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda\theta_t\right) $$

**追问：什么情况下 SGD 优于 Adam？**

- **泛化性能**：SGD 往往有更好的泛化能力
- **收敛后期**：Adam 可能在最优解附近震荡
- **资源受限**：SGD 内存占用更小
- **CV 任务**：图像分类等任务常用 SGD + momentum

---

### Q5: 什么是学习率衰减？常用的衰减策略有哪些？

**基础回答：**

学习率衰减是在训练过程中逐渐降低学习率的策略，帮助模型更好地收敛。

**深入回答：**

**为什么需要衰减？**

- 训练初期：大学习率加速收敛
- 训练后期：小学习率精细调优，避免震荡

**常用策略**：

| 策略 | 公式 | 特点 |
|------|------|------|
| 阶梯衰减 | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | 每隔 s 步乘以 γ |
| 指数衰减 | $\eta_t = \eta_0 \cdot \gamma^t$ | 连续指数衰减 |
| 余弦衰减 | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ | 平滑衰减 |
| 线性预热+衰减 | 先线性增加再衰减 | 大模型训练常用 |

**追问：为什么大模型训练需要 warmup？**

**预热实现**：

$$ \eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{\text{warmup\_steps}} & \text{if } t < \text{warmup\_steps} \\ \text{decay\_strategy}(t) & \text{otherwise} \end{cases} $$

原因：
1. 早期参数随机，梯度不稳定，大学习率会导致参数更新过大
2. Warmup 让模型先适应数据分布，再使用大学习率
3. 大模型参数多，更容易受到早期不稳定梯度的影响

---

### Q6: 什么是梯度裁剪？为什么需要它？

**基础回答：**

梯度裁剪是限制梯度大小的技术，防止梯度爆炸导致训练不稳定。

**深入回答：**

**常用方法**：

**代码实现**：

```python
# 按值裁剪
grad = clip(grad, -threshold, threshold)

# 按范数裁剪（更常用）
if grad_norm > max_norm:
    grad = grad * max_norm / grad_norm
```

**追问：为什么按范数裁剪更常用？**

按范数裁剪保留了梯度方向，只调整大小：
- 保持优化方向不变
- 自动适应不同参数的梯度尺度
- 不需要手动设置每个参数的阈值

**追问：在 Transformer 训练中，梯度裁剪通常设置多少？**

常用 `max_norm = 1.0`，但具体值需要根据任务调整：
- 太小：收敛慢，可能欠拟合
- 太大：无法有效防止梯度爆炸

---

## 三、激活函数

### Q7: 常用激活函数有哪些？各有什么特点？

**基础回答：**

常用激活函数包括 ReLU、Sigmoid、Tanh、GELU、Swish 等。

**深入回答：**

| 激活函数 | 公式 | 优点 | 缺点 |
|----------|------|------|------|
| **ReLU** | $\max(0, x)$ | 计算快，解决梯度消失 | 死亡 ReLU 问题 |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | 输出 $(0,1)$，适合概率 | 梯度消失，非零中心 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 零中心 | 梯度消失 |
| **GELU** | $x \cdot \Phi(x)$ | 平滑，Transformer 常用 | 计算稍复杂 |
| **Swish** | $x \cdot \text{sigmoid}(\beta x)$ | 平滑，自门控 | 需调参 |
| **SwiGLU** | $\text{Swish}(xW) \odot (xV)$ | GLM/LLaMA 使用 | 参数量增加 |

**SwiGLU 结构图**（LLaMA 等模型使用）：

![SwiGLU 结构图](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91SwiGLU.png)

**追问：什么是"死亡 ReLU"问题？如何解决？**

当输入恒为负时，ReLU 输出恒为 0，梯度为 0，参数不再更新。

解决方法：
- **LeakyReLU**：$f(x) = \max(\alpha x, x)$，α 通常为 0.01
- **PReLU**：α 作为可学习参数
- **ELU**：负区间使用指数函数，输出均值接近 0

**追问：为什么 Tanh 比 Sigmoid 更好？**

1. **零中心**：Tanh 输出范围 (-1, 1)，均值接近 0
2. **梯度更大**：Tanh 导数最大为 1，Sigmoid 最大为 0.25
3. **收敛更快**：零中心化使梯度下降方向更稳定

---

### Q8: 为什么 Transformer 中使用 GELU？

**基础回答：**

GELU 在 Transformer 架构上表现更好，且梯度更平滑。

**深入回答：**

**GELU 特点**：

$$
\text{GELU}(x) = x \cdot \Phi(x) \quad \text{Φ 是标准正态分布 CDF}
$$

**近似计算（实际使用）**：
$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
$$

**追问：GELU 相比 ReLU 的优势？**

1. **平滑性**：在 0 附近平滑过渡，梯度变化连续
2. **非单调性**：负值区域有非零输出，避免完全"死亡"
3. **概率解释**：可以理解为 x 被保留的概率
4. **实践效果**：在 BERT、GPT 等模型上效果更好

---

## 四、正则化

### Q9: 什么是过拟合？如何防止过拟合？

**基础回答：**

过拟合是指模型在训练集上表现很好，但在测试集上表现差，即泛化能力不足。

**深入回答：**

**防止过拟合的方法**：

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **Dropout** | 随机丢弃神经元 | 全连接层 |
| **L1/L2 正则化** | 惩励参数大小 | 通用 |
| **BatchNorm** | 规范化激活值分布 | 深层网络 |
| **LayerNorm** | 按层归一化 | Transformer |
| **数据增强** | 扩充训练数据 | CV 任务 |
| **早停** | 验证集损失上升时停止 | 通用 |
| **模型压缩** | 减少参数量 | 大模型 |

**追问：Dropout 为什么能防止过拟合？**

1. **集成效应**：相当于训练多个子网络的集成
2. **减少协同**：神经元不能过度依赖其他神经元
3. **增加鲁棒性**：迫使每个神经元都能独立工作

**追问：为什么 Transformer 使用 LayerNorm 而不是 BatchNorm？**

![BatchNorm 与 LayerNorm 对比](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91BatchNorm%E4%B8%8ELayerNorm.png)

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 批次维度 | 特征维度 |
| 对批次大小敏感 | 是 | 否 |
| 适合变长序列 | 否 | 是 |
| 推理时行为 | 需要统计量 | 不需要 |

Transformer 处理变长序列，且批次大小可能较小，LayerNorm 更合适。

---

### Q10: L1 和 L2 正则化有什么区别？

**基础回答：**

L1 正则化是参数绝对值之和，L2 正则化是参数平方和。L1 会产生稀疏解，L2 使参数更小更均匀。

**深入回答：**

**正则化公式**：

$$ L_1 = L_{\text{loss}} + \lambda \sum|w_i| \quad \text{(L1 正则化)} $$

$$ L_2 = L_{\text{loss}} + \lambda \sum w_i^2 \quad \text{(L2 正则化)} $$

**追问：从梯度角度分析为什么 L1 产生稀疏解？**

$
\begin{aligned}
\text{L1 梯度:} \quad \frac{\partial L}{\partial w} &= \frac{\partial L_{\text{loss}}}{\partial w} + \lambda \cdot \text{sign}(w) \\
\text{L2 梯度:} \quad \frac{\partial L}{\partial w} &= \frac{\partial L_{\text{loss}}}{\partial w} + 2\lambda w
\end{aligned}
$

- L1 梯度在 w=0 处不可导，会出现"尖点"
- L1 的梯度大小恒为 λ，与 w 大小无关
- 当 w 接近 0 时，L2 梯度也接近 0，但 L1 仍有 λ 的推动力

**追问：实际中如何选择 L1 或 L2？**

| 场景 | 推荐 | 原因 |
|------|------|------|
| 特征选择 | L1 | 自动选择重要特征 |
| 防止过拟合 | L2 | 更稳定，不产生稀疏 |
| 不知道选哪个 | Elastic Net | L1 + L2 结合 |

---

### Q11: BatchNorm 的工作原理是什么？

**基础回答：**

BatchNorm 对每个批次的数据进行归一化，使均值为 0、方差为 1，然后通过可学习的参数进行缩放和平移。

**深入回答：**

**BatchNorm 计算公式**：

$$ \mu_B = \frac{1}{m} \sum x_i \quad \text{(批次均值)} $$

$$ \sigma_B^2 = \frac{1}{m} \sum(x_i - \mu_B)^2 \quad \text{(批次方差)} $$

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}} \quad \text{(归一化)} $$

$$ y_i = \gamma \hat{x}_i + \beta \quad \text{(缩放和平移)} $$

**归一化方法对比**：

![RMSNorm 与其他归一化方法](https://raw.githubusercontent.com/changyeyu/LLM-RL-Visualized/master/images_chinese/png_small/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91RMSNorm.png)

**追问：为什么需要 γ 和 β？**

如果只有归一化，网络的表达能力会受到限制：
- $\gamma$ 恢复数据的缩放能力
- $\beta$ 恢复数据的平移能力
- 网络可以学习"撤销"BN，如果这对任务有利

**追问：BatchNorm 在小批次时为什么效果不好？**

- 小批次的统计量（均值、方差）不稳定
- 不能很好地代表整体分布
- 解决方法：使用 GroupNorm 或 LayerNorm 替代

---

## 五、参数初始化

### Q12: 为什么参数初始化很重要？

**基础回答：**

好的初始化可以：
1. 避免梯度消失/爆炸
2. 加速收敛
3. 影响最终模型性能

**深入回答：**

**不好的初始化会导致**：
- 全 0 初始化：所有神经元输出相同，对称性无法打破
- 过大初始化：激活值饱和，梯度消失
- 过小初始化：激活值过小，信号衰减

**追问：Xavier 初始化的原理是什么？**

目标：使各层输出的方差保持一致

假设输入和权重独立且均值为 0：
$$
\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

为了 $\text{Var}(y) = \text{Var}(x)$，需要：
$$
\text{Var}(w) = \frac{1}{n_{\text{in}}}
$$

考虑前向和反向传播，取折中：
$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

**追问：为什么 ReLU 使用 He 初始化而不是 Xavier？**

ReLU 会将一半的输入置为 0，方差减半。因此需要将初始化方差加倍：
$$
\text{He 初始化：Var}(w) = \frac{2}{n_{\text{in}}}
$$

---

## 六、损失函数

### Q13: 交叉熵损失和均方误差有什么区别？

**基础回答：**

交叉熵损失用于分类任务，均方误差用于回归任务。交叉熵损失配合 softmax 更适合分类。

**深入回答：**

**损失函数公式**：

$$ \text{MSE} = \frac{1}{n} \sum(y_i - \hat{y}_i)^2 $$

$$ \text{CE} = -\sum y_i \log(\hat{y}_i) $$

**追问：为什么分类任务用交叉熵而不是 MSE？**

1. **学习速度**：
   - MSE + Sigmoid：$\text{梯度} = (\hat{y}-y) \cdot \hat{y}(1-\hat{y}) \cdot x$
   - CE + Sigmoid：$\text{梯度} = (\hat{y}-y) \cdot x$
   
   当 $\hat{y}$ 接近 0 或 1 时，MSE 梯度趋近 0，学习变慢

2. **优化曲面**：
   - MSE 对于分类问题是非凸的
   - CE 对于分类问题是凸的（配合 softmax）

**追问：什么是 Label Smoothing？**

**Label Smoothing 公式**：

$$ y = [0, 0, 1, 0] \quad \text{(one-hot 原始标签)} $$

$$ y_{\text{smooth}} = (1-\varepsilon)y + \frac{\varepsilon}{K} = \left[\frac{\varepsilon}{4}, \frac{\varepsilon}{4}, 1-\varepsilon+\frac{\varepsilon}{4}, \frac{\varepsilon}{4}\right] $$

好处：
- 防止模型过于自信
- 提高泛化能力
- 对标注错误有一定鲁棒性

---

### Q14: 什么是 Focal Loss？为什么提出它？

**基础回答：**

Focal Loss 是为了解决类别不平衡问题而提出的损失函数，通过降低易分类样本的权重来关注难分类样本。

**深入回答：**

**Focal Loss 公式**：

$$ \text{CE} = -\log(p_t) \quad \text{(标准交叉熵)} $$

$$ \text{FL} = -\alpha_t(1-p_t)^\gamma \log(p_t) \quad \text{(Focal Loss)} $$

其中：
- $\alpha_t$：类别权重，解决类别不平衡
- $\gamma$：聚焦参数，降低易分类样本权重
- $p_t$：正确类别的预测概率

**追问：γ 如何影响模型行为？**

| γ 值 | 易分类样本权重 | 难分类样本权重 |
|------|----------------|----------------|
| 0 | 无衰减（等同 CE） | 无变化 |
| 2 | 显著衰减 | 相对放大 |

当 $p_t = 0.9$ 时：
- CE 权重：1.0
- FL($\gamma=2$) 权重：$(1-0.9)^2 = 0.01$

易分类样本的损失被大幅降低，模型更关注难分类样本。

---

## 七、综合问题

### Q15: 深度学习模型训练的完整流程是什么？

**参考回答：**

```
1. 数据准备
   ├── 数据收集与清洗
   ├── 数据预处理（归一化、tokenization等）
   ├── 数据增强（可选）
   └── 划分训练/验证/测试集

2. 模型设计
   ├── 选择模型架构
   ├── 定义损失函数
   └── 选择优化器

3. 训练
   ├── 参数初始化
   ├── 前向传播计算损失
   ├── 反向传播计算梯度
   ├── 参数更新
   └── 周期性验证

4. 调优
   ├── 超参数搜索
   ├── 正则化调整
   └── 学习率调度

5. 评估与部署
   ├── 测试集评估
   ├── 模型压缩（可选）
   └── 部署上线
```

**追问：如何判断模型是否过拟合？**

```
监控指标：
├── 训练损失持续下降，验证损失上升 → 过拟合
├── 训练准确率远高于验证准确率 → 过拟合
├── 两者都高且接近 → 欠拟合
└── 两者都低且接近 → 正常
```

---

### Q16: 如何调试深度学习模型？

**参考回答：**

**调试清单**：

```
1. 数据检查
   ├── 数据分布是否正常
   ├── 标签是否正确
   └── 数据增强是否合理

2. 初始检查
   ├── 模型能否过拟合小数据集（应该能做到 100%）
   ├── 损失是否下降
   └── 梯度是否正常（检查梯度范数）

3. 训练过程检查
   ├── 学习率是否合适
   ├── 是否有 NaN/Inf
   ├── 验证集指标是否正常
   └── 梯度是否消失/爆炸

4. 超参数检查
   ├── batch size
   ├── 学习率
   ├── 正则化强度
   └── 网络深度/宽度
```

**追问：如果损失不下降怎么办？**

1. 检查学习率：太大导致震荡，太小导致收敛慢
2. 检查数据：标签是否正确，预处理是否合理
3. 检查模型：是否有梯度流动，激活值是否正常
4. 尝试更简单的模型：先确保能跑通
5. 增大学习率：可能是初始化问题

---

## 📝 总结

### 核心知识点

| 主题 | 核心要点 |
|------|----------|
| **前向/反向传播** | 链式法则，动态规划优化 |
| **梯度问题** | 残差连接、ReLU、BN、梯度裁剪 |
| **激活函数** | 非线性、导数特性、死亡问题 |
| **优化器** | 动量、自适应学习率、AdamW |
| **正则化** | Dropout、L1/L2、LayerNorm |
| **初始化** | Xavier、He 初始化原理 |
| **损失函数** | 交叉熵、Focal Loss、Label Smoothing |

### 常见追问方向

1. **原理层面**：为什么这样设计？数学推导？
2. **对比层面**：A 和 B 有什么区别？什么场景用哪个？
3. **实践层面**：实际项目中怎么用？遇到过什么问题？
4. **代码层面**：能手写实现吗？有哪些细节？

---

*[下一章：Transformer 架构面试题 →](./transformer.md)*
