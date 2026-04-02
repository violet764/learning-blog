# 世界模型经典论文解读

本章汇总世界模型领域的里程碑论文，帮助读者按时间线理解技术演进。

---

## 论文时间线

```
2018                    2020                    2022                    2023                    2024
  │                       │                       │                       │                       │
  ▼                       ▼                       ▼                       ▼                       ▼
World Models         Dreamer v1               IRIS                 Dreamer v3              UniSim/Genie
  │                       │                       │                       │                       │
  │    VAE+MDN-RNN        │    RSSM              │    Transformer        │    通用算法            │    通用世界模型
  │    想象训练           │    端到端             │    离散token          │    固定超参            │    真实数据训练
  │                       │    想象规划           │                       │                       │
  └───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
```

---

## World Models (2018)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | World Models |
| 作者 | David Ha, Jürgen Schmidhuber |
| 发表 | NeurIPS 2018 Workshop |
| 链接 | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) |
| 代码 | [worldmodels.github.io](https://worldmodels.github.io/) |

### 核心思想

论文提出"在梦中学习"的范式：

1. **VAE**：将图像压缩为紧凑的潜在向量
2. **MDN-RNN**：预测下一个潜在状态
3. **控制器**：在潜在空间中训练策略

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    World Models 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   图像 o_t ──→ [VAE编码器] ──→ z_t                          │
│                                    │                        │
│   z_t, a_t ──→ [MDN-RNN] ──→ 预测 z_{t+1}                   │
│                                    │                        │
│   z_t ──→ [控制器 C] ──→ 动作 a_t                            │
│                                                             │
│   训练流程:                                                  │
│   1. 随机收集数据 → 训练 VAE                                 │
│   2. 用 VAE 编码数据 → 训练 MDN-RNN                          │
│   3. 在潜在空间训练控制器 (CMA-ES)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键创新

| 创新 | 描述 |
|------|------|
| 分离训练 | VAE、RNN、控制器分阶段训练 |
| 潜在空间规划 | 在低维空间进行规划 |
| 混合密度网络 | 建模多模态状态分布 |

### 局限性

- 分阶段训练，非端到端
- 使用进化算法训练控制器，效率低
- 无法处理复杂环境

### 一句话总结

> 开创性地展示了在压缩潜在空间训练智能体的可能性。

---

## Dream to Control (Dreamer v1, 2020)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Dream to Control: Learning Behaviors by Latent Imagination |
| 作者 | Danijar Hafner, Timothy Lillicrop, Jimmy Ba |
| 发表 | ICLR 2020 |
| 链接 | [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) |

### 核心思想

实现端到端的想象规划：

1. **RSSM**：循环状态空间模型
2. **想象规划**：在潜在空间预测未来轨迹
3. **策略梯度**：通过世界模型反向传播

### 关键架构：RSSM

```
RSSM 的核心创新:
├── 确定性状态 h_t: 由 GRU 更新，保持长期记忆
├── 随机性状态 s_t: 概率分布，建模不确定性
└── 联合状态: (h_t, s_t) 共同表示环境状态
```

### 损失函数

$$
\mathcal{L} = \mathbb{E}_{q(s_{1:T}|o_{1:T})}[\sum_t (\beta_{KL} L_{KL}^t + \beta_{pred} L_{pred}^t + \beta_{recon} L_{recon}^t)]
$$

### 关键创新

| 创新 | 描述 |
|------|------|
| RSSM | 确定性+随机性状态 |
| 想象规划 | 在学到的模型中规划 |
| 端到端 | 策略梯度通过世界模型传播 |

### 一句话总结

> 首次实现端到端的想象规划，显著提升样本效率。

---

## Dreamer v2 (2021)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Mastering Atari with Discrete World Models |
| 作者 | Danijar Hafner et al. |
| 发表 | ICLR 2021 |
| 链接 | [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) |

### 核心贡献

使用**离散潜在状态**实现 Atari 突破：

```
离散潜在状态:
├── 将连续潜在空间离散化
├── 使用 Gumbel-Softmax 训练
├── 更好地建模多模态分布
└── 在 Atari 100k 上超越人类
```

### 性能

```
Atari 100k Benchmark:
├── Dreamer v2: 134% 人类平均
├── SimPLe: 79%
├── OTRainbow: 20%
└── DER: 14%
```

### 一句话总结

> 通过离散潜在状态，首次在 Atari 上实现超人性能。

---

## IRIS (2022)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Implicit Reward without Intermediate Supervision |
| 作者 | Michal Bortkiewicz et al. |
| 发表 | ICLR 2023 |
| 链接 | [arXiv:2210.14301](https://arxiv.org/abs/2210.14301) |

### 核心思想

使用 Transformer 作为世界模型：

```
IRIS 架构:
├── VAE: 将观测编码为离散 token
├── GPT: 自回归预测下一个 token
└── 想象规划: 在 token 空间规划
```

### 与 Dreamer 对比

| 方面 | Dreamer | IRIS |
|------|---------|------|
| 架构 | RSSM (RNN) | Transformer |
| 并行性 | 串行 | 并行 |
| 长期依赖 | 可能遗忘 | 更好 |
| 复杂度 | O(T) | O(T²) |

### 一句话总结

> 展示了 Transformer 作为世界模型的潜力。

---

## Dreamer v3 (2023)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Mastering Diverse Domains through World Models |
| 作者 | Danijar Hafner et al. |
| 发表 | arXiv 2023 |
| 链接 | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |

### 核心贡献

实现**通用世界模型**：

1. **固定超参数**：所有任务使用相同配置
2. **多样化任务**：从 Atari 到 Minecraft
3. **鲁棒性**：无需任务特定调优

### 关键技术改进

| 改进 | 描述 |
|------|------|
| Symlog 预测 | 处理不同尺度的奖励 |
| Free bits | 防止 KL 塌缩 |
| 更大容量 | 增加模型大小 |
| 分离学习率 | actor-critic 使用不同学习率 |

### 性能

| 域 | 性能 |
|---|------|
| Atari 100k | 164% 人类水平 |
| Minecraft | 首次实现钻石获取 |
| DMC | SOTA |
| BSUITE | 强泛化能力 |

### 一句话总结

> 实现了固定超参数的通用世界模型算法。

---

## UniSim (2024)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | UniSim: A Universal Simulator of Real-World Interactions |
| 作者 | Google DeepMind |
| 发表 | arXiv 2024 |
| 链接 | [arXiv:2310.01778](https://arxiv.org/abs/2310.01778) |

### 核心思想

从真实交互数据学习通用世界模型：

```
UniSim 特点:
├── 数据来源: 真实世界交互记录
├── 多模态: 文本、图像、动作
├── 通用性: 可模拟多种场景
└── 应用: 机器人、自动驾驶等
```

### 与传统世界模型对比

| 方面 | 传统世界模型 | UniSim |
|------|-------------|--------|
| 数据 | 特定任务收集 | 大规模真实数据 |
| 训练 | 从零开始 | 预训练+微调 |
| 泛化 | 任务特定 | 多场景泛化 |

### 一句话总结

> 展示了从真实数据学习通用世界模型的可能性。

---

## Genie (2024)

### 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | Genie: Generative Interactive Environments |
| 作者 | Google DeepMind |
| 发表 | arXiv 2024 |
| 链接 | [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) |

### 核心思想

从无标签视频学习可交互的生成环境：

```
Genie 能力:
├── 输入: 图像或文本描述
├── 输出: 可交互的2D游戏
├── 训练: 大量游戏视频(无标签)
└── 应用: 游戏生成、环境创建
```

### 架构

```
Genie 架构:
├── 视频分词器: 将视频帧编码为离散 token
├── 潜在动作模型: 推断帧间的隐动作
├── 动态模型: 预测下一帧 token
└── 解码器: 生成视频帧
```

### 一句话总结

> 实现了从无标签视频学习可交互世界模型。

---

## 论文阅读建议

### 阅读顺序

推荐按以下顺序阅读：

```
入门路径:
1. World Models (2018) - 理解基本思想
2. Dreamer v1 (2020) - 掌握 RSSM 和想象规划
3. Dreamer v2 (2021) - 理解离散潜在状态

进阶路径:
4. Dreamer v3 (2023) - 学习通用世界模型
5. IRIS (2022) - 了解 Transformer 世界模型
6. UniSim/Genie (2024) - 探索前沿方向
```

### 阅读技巧

| 技巧 | 描述 |
|------|------|
| 先读博客 | 论文之前先找博客/讲解视频 |
| 关注方法 | 理解核心方法而非实验细节 |
| 动手实现 | 尝试复现关键组件 |
| 对比阅读 | 比较不同论文的异同 |

---

## 小结

世界模型论文演进反映了以下趋势：

1. **架构演进**：VAE+RNN → RSSM → Transformer
2. **训练方式**：分离训练 → 端到端
3. **通用性**：任务特定 → 通用算法
4. **数据来源**：模拟器 → 真实数据

---

## 参考资料

### 论文链接

- [World Models](https://arxiv.org/abs/1803.10122)
- [Dreamer v1](https://arxiv.org/abs/1912.01603)
- [Dreamer v2](https://arxiv.org/abs/2010.02193)
- [Dreamer v3](https://arxiv.org/abs/2301.04104)
- [IRIS](https://arxiv.org/abs/2210.14301)
- [UniSim](https://arxiv.org/abs/2310.01778)
- [Genie](https://arxiv.org/abs/2402.15391)

### 代码仓库

- [Dreamer 官方代码](https://github.com/danijar/dreamer)
- [Dreamer v3 代码](https://github.com/danijar/dreamerv3)
- [World Models 代码](https://github.com/hardmaru/WorldModelsExperiments)

### 推荐博客

- [World Models 交互式论文](https://worldmodels.github.io/)
- [李宏毅 World Model 讲解](https://www.youtube.com/watch?v=YlrsMp9vJEI)
