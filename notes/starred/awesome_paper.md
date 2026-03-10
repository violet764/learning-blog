# 📄 优质论文收集

这里收集了各领域经典与前沿的学术论文，按领域分类整理。

## 目录

- [深度学习基础](#深度学习基础)
- [大语言模型](#大语言模型)
- [计算机视觉](#计算机视觉)
- [强化学习](#强化学习)
- [推荐系统](#推荐系统)
- [分布式系统](#分布式系统)

---

## 深度学习基础

### 🔥 Attention Is All You Need (2017)

- **作者**: Vaswani et al. (Google)
- **链接**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **标签**: 🔥 经典 ⭐ 推荐

**一句话总结**: 提出了 Transformer 架构，彻底改变了 NLP 乃至整个 AI 领域的范式。

**核心贡献**:
- 提出自注意力机制（Self-Attention）
- 完全抛弃 RNN/CNN，实现并行化
- 成为 GPT、BERT 等模型的基础架构

**为什么值得读**: 这是现代深度学习最重要的论文之一，理解 Transformer 是理解当前 AI 的基础。

---

### 🔥 Deep Residual Learning for Image Recognition (2015)

- **作者**: He et al. (Microsoft Research)
- **链接**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- **标签**: 🔥 经典 ⭐ 推荐

**一句话总结**: 提出残差连接，解决了深层网络训练困难的问题。

**核心贡献**:
- 提出残差学习框架
- 解决梯度消失/爆炸问题
- 使得训练数百层网络成为可能

**为什么值得读**: ResNet 是现代深度学习的基石，残差连接被广泛应用于各种架构。

---

### 🔥 Adam: A Method for Stochastic Optimization (2014)

- **作者**: Kingma & Ba
- **链接**: [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- **标签**: 🔥 经典 🛠️ 实践

**一句话总结**: 提出了目前最广泛使用的优化器 Adam。

**核心贡献**:
- 结合 Momentum 和 RMSprop 的优点
- 自适应学习率
- 几乎不需要调参

**为什么值得读**: 理解优化器对调试和改进模型至关重要。

---

### 💡 Batch Normalization: Accelerating Deep Network Training (2015)

- **作者**: Ioffe & Szegedy
- **链接**: [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- **标签**: 💡 启发 🛠️ 实践

**一句话总结**: 通过归一化加速训练并提高模型稳定性。

**核心贡献**:
- 解决内部协变量偏移问题
- 允许使用更大学习率
- 减少对初始化的依赖

---

### 💡 Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)

- **作者**: Srivastava et al.
- **链接**: [JMLR](https://jmlr.org/papers/v15/srivastava14a.html)
- **标签**: 💡 启发 🛠️ 实践

**一句话总结**: 简单有效的正则化方法，防止过拟合。

**核心贡献**:
- 训练时随机丢弃神经元
- 相当于集成多个子网络
- 几乎不增加计算成本

---

## 大语言模型

### 🔥 Language Models are Few-Shot Learners (GPT-3, 2020)

- **作者**: Brown et al. (OpenAI)
- **链接**: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **标签**: 🔥 经典 ⭐ 推荐

**一句话总结**: 证明了大规模语言模型的涌现能力，开启了大模型时代。

**核心贡献**:
- 175B 参数的 GPT-3 模型
- 展示 Few-shot/Zero-shot 能力
- 证明了规模定律（Scaling Laws）

**为什么值得读**: 理解大模型能力的来源，以及为什么规模很重要。

---

### 🔥 BERT: Pre-training of Deep Bidirectional Transformers (2018)

- **作者**: Devlin et al. (Google)
- **链接**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **标签**: 🔥 经典 ⭐ 推荐

**一句话总结**: 开创了双向预训练语言模型的范式。

**核心贡献**:
- 掩码语言模型（MLM）
- 下一句预测（NSP）
- 刷新多项 NLP 任务 SOTA

---

### 💡 LLaMA: Open and Efficient Foundation Language Models (2023)

- **作者**: Touvron et al. (Meta)
- **链接**: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- **标签**: 💡 启发 ⭐ 推荐

**一句话总结**: 开源社区最重要的基础模型，证明小模型也能有强大能力。

**核心贡献**:
- 仅用公开数据训练
- 证明数据质量比数量更重要
- 推动开源大模型生态发展

---

### 💡 Training language models to follow instructions with human feedback (InstructGPT, 2022)

- **作者**: Ouyang et al. (OpenAI)
- **链接**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **标签**: 💡 启发 ⭐ 推荐

**一句话总结**: RLHF 技术的开创性工作，让模型学会遵循指令。

**核心贡献**:
- 提出 RLHF 训练范式
- 人类反馈对齐模型行为
- 大幅提升模型有用性和安全性

---

### ⭐ Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)

- **作者**: Wei et al. (Google)
- **链接**: [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **标签**: ⭐ 推荐 💡 启发

**一句话总结**: 发现让模型"一步步思考"可以大幅提升推理能力。

**核心贡献**:
- 提出思维链提示（CoT）
- 显著提升数学和逻辑推理能力
- 开启提示工程研究方向

---

## 计算机视觉

### 🔥 ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, 2012)

- **作者**: Krizhevsky et al.
- **链接**: [NIPS 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- **标签**: 🔥 经典 📖 入门

**一句话总结**: 深度学习革命的起点，AlexNet 在 ImageNet 上取得突破性成绩。

**核心贡献**:
- 首次在大规模数据集上成功应用深度 CNN
- 引入 ReLU 激活函数和 Dropout
- 开启深度学习时代

**为什么值得读**: 了解深度学习如何开始统治计算机视觉领域。

---

### 🔥 U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)

- **作者**: Ronneberger et al.
- **链接**: [MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- **标签**: 🔥 经典 🛠️ 实践

**一句话总结**: 图像分割领域的经典架构，被广泛应用于各种分割任务。

**核心贡献**:
- 编码器-解码器对称结构
- 跳跃连接保留细节信息
- 小数据集也能训练好

---

### 💡 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, 2020)

- **作者**: Dosovitskiy et al. (Google)
- **链接**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **标签**: 💡 启发 ⭐ 推荐

**一句话总结**: 将 Transformer 应用于图像，开创视觉 Transformer 时代。

**核心贡献**:
- 图像分块嵌入
- 纯 Transformer 架构处理图像
- 大规模预训练效果超越 CNN

---

### ⭐ CLIP: Learning Transferable Visual Models From Natural Language Supervision (2021)

- **作者**: Radford et al. (OpenAI)
- **链接**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **标签**: ⭐ 推荐 💡 启发

**一句话总结**: 图文对齐的里程碑工作，开启多模态时代。

**核心贡献**:
- 对比学习对齐图文表示
- 零样本迁移能力强
- 支持 Open-vocabulary 检测/分割

---

## 强化学习

### 🔥 Playing Atari with Deep Reinforcement Learning (DQN, 2013)

- **作者**: Mnih et al. (DeepMind)
- **链接**: [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- **标签**: 🔥 经典 📖 入门

**一句话总结**: 深度强化学习的开山之作，首次用神经网络玩 Atari 游戏。

**核心贡献**:
- 提出经验回放机制
- 目标网络稳定训练
- 开启深度强化学习时代

---

### 🔥 Mastering the game of Go with deep neural networks (AlphaGo, 2016)

- **作者**: Silver et al. (DeepMind)
- **链接**: [Nature](https://www.nature.com/articles/nature16961)
- **标签**: 🔥 经典 ⭐ 推荐

**一句话总结**: AI 历史里程碑，AlphaGo 击败人类围棋世界冠军。

**核心贡献**:
- 结合监督学习和强化学习
- 蒙特卡洛树搜索 + 深度神经网络
- 证明 AI 可以在复杂策略游戏中超越人类

---

### 💡 Proximal Policy Optimization Algorithms (PPO, 2017)

- **作者**: Schulman et al. (OpenAI)
- **链接**: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **标签**: 💡 启发 🛠️ 实践 ⭐ 推荐

**一句话总结**: 最流行的策略梯度算法，简单高效且稳定。

**核心贡献**:
- 截断目标函数防止过大更新
- 比TRPO更简单易实现
- 成为 RLHF 的核心算法

---

### ⭐ Training language models to follow instructions with human feedback (2022)

见大语言模型部分，RLHF 的重要应用。

---

## 推荐系统

### 🔥 Neural Collaborative Filtering (NCF, 2017)

- **作者**: He et al.
- **链接**: [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)
- **标签**: 🔥 经典 🛠️ 实践

**一句话总结**: 将神经网络引入协同过滤，开启神经推荐系统时代。

**核心贡献**:
- 用神经网络学习用户-物品交互
- 替代传统矩阵分解
- 支持非线性交互建模

---

### 💡 Deep & Cross Network for Ad Click Predictions (DCN, 2017)

- **作者**: Wang et al.
- **链接**: [arXiv:1708.05123](https://arxiv.org/abs/1708.05123)
- **标签**: 💡 启发 🛠️ 实践

**一句话总结**: 工业实践中广泛使用的 CTR 预估模型。

**核心贡献**:
- 显式学习特征交叉
- 保持低计算复杂度
- 易于部署和优化

---

## 分布式系统

### 🔥 MapReduce: Simplified Data Processing on Large Clusters (2004)

- **作者**: Dean & Ghemawat (Google)
- **链接**: [OSDI 2004](https://dl.acm.org/doi/10.5555/1251254.1251264)
- **标签**: 🔥 经典 📖 入门

**一句话总结**: 分布式计算范式的奠基之作。

**核心贡献**:
- 提出MapReduce编程模型
- 自动并行化和容错
- 影响了Hadoop、Spark等系统

---

### 🔥 Bigtable: A Distributed Storage System for Structured Data (2006)

- **作者**: Chang et al. (Google)
- **链接**: [OSDI 2006](https://dl.acm.org/doi/10.5555/1298455.1298475)
- **标签**: 🔥 经典 🛠️ 实践

**一句话总结**: 大规模结构化数据存储的经典设计。

**核心贡献**:
- 列族存储模型
- LSM-tree存储结构
- 影响了HBase、Cassandra等系统

---

### 💡 The Google File System (2003)

- **作者**: Ghemawat et al. (Google)
- **链接**: [SOSP 2003](https://dl.acm.org/doi/10.1145/945445.945450)
- **标签**: 💡 启发 📖 入门

**一句话总结**: 分布式文件系统的经典设计。

**核心贡献**:
- Master-Slave架构
- 大文件优化设计
- 容错和自动恢复机制

---

### ⭐ Raft: In Search of an Understandable Consensus Algorithm (2014)

- **作者**: Ongaro & Ousterhout
- **链接**: [USENIX ATC 2014](https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro)
- **标签**: ⭐ 推荐 📖 入门 💡 启发

**一句话总结**: 比 Paxos 更易理解的一致性算法，工业界广泛使用。

**核心贡献**:
- 分解为Leader选举、日志复制、安全性
- 易于理解和实现
- 被etcd、Consul等采用

---

## 阅读建议

### 入门路线

1. 先读经典论文的博客解读（如 Transformer、ResNet）
2. 结合代码实现一起看（HuggingFace、PyTorch官方实现）
3. 写笔记总结核心思想

### 进阶路线

1. 关注领域最新进展（arXiv、Twitter、论文博客）
2. 复现关键论文
3. 尝试改进或应用到新领域

### 论文资源

- [arXiv](https://arxiv.org/) - 预印本论文库
- [Papers With Code](https://paperswithcode.com/) - 论文+代码
- [Connected Papers](https://www.connectedpapers.com/) - 论文关系图谱
- [Semantic Scholar](https://www.semanticscholar.org/) - 学术搜索引擎
