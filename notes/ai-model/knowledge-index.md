# LLM 知识点索引

本文档帮助你快速定位每个知识点的学习位置，点击链接可直接跳转到具体章节。

---

## 第一章. LLM底层技术解析

### 1.1 Transformer知识必备

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 1.1.1 基础架构 | [Transformer架构](/notes/deep-learning/07-transformer#一-transformer-架构概述) | 编码器-解码器结构、自注意力机制 |
| 1.1.2 模型参数量计算 | [模型规模估算-参数量计算](/notes/ai-model/llm/model-scaling#参数量计算) | Transformer参数量公式推导 |
| 1.1.3 手推显存占用 | [模型规模估算-显存估算](/notes/ai-model/llm/model-scaling#显存估算) | 参数/梯度/优化器/激活显存计算 |
| 1.1.4 手推计算量FLOPs | [模型规模估算-FLOPs计算](/notes/ai-model/llm/model-scaling#flops计算) | Attention/FFN FLOPs推导 |
| 1.1.5 训练时间估算 | [模型规模估算-训练时间](/notes/ai-model/llm/model-scaling#训练时间估算) | 基于算力的训练时间预估 |

### 1.2 Attention机制及其变种

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 1.2.1 为什么需要Attention | [注意力机制概述](/notes/ai-model/llm/attention-mechanisms#一-注意力机制概述) | 序列建模的痛点 |
| 1.2.2 如何理解Self-Attention | [缩放点积注意力](/notes/ai-model/llm/attention-mechanisms#二-缩放点积注意力) | 缩放点积注意力、Q/K/V |
| 1.2.3 Self-Attention with trainable weights | [基本原理](/notes/ai-model/llm/attention-mechanisms#21-基本原理) | 权重矩阵的作用 |
| 1.2.4 什么是Causal Attention | [因果注意力](/notes/ai-model/llm/attention-mechanisms#25-因果注意力causal-attention) | 因果掩码、单向注意力 |
| 1.2.5 MHA/MQA/GQA原理及区别 | [多头注意力与变体](/notes/ai-model/llm/attention-mechanisms#三-多头注意力mha) | 多头/多查询/分组查询注意力对比 |

### 1.3 Tokenizer知识必备

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 1.3.1 如何理解LLM中的token | [分词概述](/notes/ai-model/llm/tokenization#分词概述) | word/subword/char分词粒度 |
| 1.3.2 Tokenizer算法 | [BPE算法详解](/notes/ai-model/llm/tokenization#bpe算法详解) | BPE/BBPE/Wordpiece/ULM算法 |
| 1.3.3 如何衡量Tokenizer的好坏 | [分词器评估](/notes/ai-model/llm/tokenization#分词器评估) | 词汇表大小权衡、压缩率评估 |
| 1.3.4 训练自己的LLM Tokenizer | [自定义分词器](/notes/ai-model/llm/tokenization#自定义分词器) | SentencePiece训练示例 |

### 1.4 位置编码

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 1.4.1 绝对位置编码 | [位置编码](/notes/ai-model/llm/embedding#四-位置编码) | 正弦余弦编码详解 |
| 1.4.2 相对位置编码 | [相对位置编码](/notes/ai-model/llm/embedding#71-相对位置编码relative-pe) | 相对位置关系证明 |
| 1.4.3 旋转位置编码(RoPE) | [RoPE位置编码](/notes/ai-model/llm/embedding#五-旋转位置编码rope) | RoPE数学推导与实现 |
| 1.4.4 ALiBi位置编码 | [ALiBi位置编码](/notes/ai-model/llm/embedding#六-alibi位置编码) | ALiBi线性偏置注意力 |

---

## 第二章. LLM视角下的强化学习

### 2.1 基于人类反馈的强化学习(RLHF)

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 2.1.1 整体流程 | [RLHF概述](/notes/reinforcement-learning/llm-rl#三阶段流程) | SFT→RM→PPO三阶段流程 |
| 2.1.2 NLP中的强化学习 | [RLHF核心动机](/notes/reinforcement-learning/llm-rl#核心动机) | 文本生成即序列决策 |
| 2.1.3 价值函数 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | 优势函数计算、GAE |
| 2.1.4 四个重要角色 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | actor/reference/reward/critic model |

### 2.2 PPO原理与代码解读

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 2.2.1 Actor loss计算 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | PPO目标函数推导 |
| 2.2.2 为什么引入Advantage | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | 降低方差、稳定训练 |
| 2.2.3 为什么引入GAE | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | 广义优势估计 |
| 2.2.4 PPO-epoch引入新约束 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | KL散度约束、clip约束 |
| 2.2.5 实际收益优化与预估收益优化 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | 价值函数训练 |
| 2.2.6 DeepSpeed中的RLHF代码解读 | [PPO策略优化](/notes/reinforcement-learning/llm-rl#ppo-策略优化) | RLHFTrainer实现 |

### 2.3 强化学习进阶概念

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 2.3.1 On-policy vs Off-policy | [On-policy与Off-policy](/notes/reinforcement-learning/rl-basics#on-policy-vs-off-policy) | 策略学习范式对比 |
| 2.3.2 Online vs Offline learning | [Online与Offline学习](/notes/reinforcement-learning/rl-basics#online-vs-offline-learning) | 在线/离线学习区别 |
| 2.3.3 Distribution shift与Over-optimization | [分布偏移问题](/notes/reinforcement-learning/llm-alignment#distribution-shift分布偏移) | 分布偏移问题 |
| 2.3.4 为什么Online learning更好 | [Online与Offline学习](/notes/reinforcement-learning/rl-basics#online-vs-offline-learning) | 在线学习优势分析 |
| 2.3.5 过程监督vs结果监督 | [过程监督vs结果监督](/notes/reinforcement-learning/llm-alignment#过程监督-vs-结果监督) | PRM vs ORM |
| 2.3.6 训练状态的切换 | [RLHF原理与实践](/notes/reinforcement-learning/llm-rl) | 训练流程管理 |

---

## 第三章. LLM之Pre-Training

### 3.1 超大规模数据工程

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 3.1.1 数据来源有哪些 | [预训练技术-数据处理](/notes/ai-model/llm/pretraining#2-大规模数据处理技术) | 网页、书籍、代码、对话数据 |
| 3.1.2 数据处理的流程 | [预训练技术-数据预处理](/notes/ai-model/llm/pretraining#21-数据预处理流程) | 清洗、去重、质量过滤 |
| 3.1.3 数据配比的思路 | [数据配比策略](/notes/ai-model/llm/pretraining#23-数据配比策略) | DoReMi动态配比 |
| 3.1.4 数据质量的评估 | [数据质量评估](/notes/ai-model/llm/pretraining#24-数据质量评估) | 质量评估指标与工具 |

### 3.2 Scaling Laws

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 3.2.1 Scaling Laws的前生今世 | [模型缩放定律](/notes/ai-model/llm/pretraining#4-模型缩放定律) | Kaplan定律、Chinchilla定律 |
| 3.2.2 什么是计算最优 | [计算最优](/notes/ai-model/llm/model-scaling#计算最优) | Chinchilla最优分配 |
| 3.2.3 不同预算下的计算最优模型 | [计算最优](/notes/ai-model/llm/model-scaling#计算最优) | 预算-模型-数据权衡 |
| 3.2.4 缩放预测 | [模型缩放定律](/notes/ai-model/llm/pretraining#42-缩放定律验证实验) | 损失预测曲线 |
| 3.2.5 从Loss预测到指标预测 | [模型缩放定律](/notes/ai-model/llm/pretraining#4-模型缩放定律) | Loss与下游任务关系 |

### 3.3 预训练技术拆解

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 3.3.1 Language Modeling: MLM到CLM | [预训练目标函数](/notes/ai-model/llm/pretraining#1-预训练目标函数设计) | 掩码语言模型、因果语言模型 |
| 3.3.2 为什么选择Decoder-Only架构 | [架构选择](/notes/ai-model/llm/pretraining#14-架构选择为什么-decoder-only-成为主流) | 架构选择理由 |
| 3.3.3 激活函数原理及区别 | [神经网络基础-激活函数](/notes/deep-learning/01-neural-network-basics#激活函数) | ReLU/GELU/SwiGLU |
| 3.3.4 Normalization全面解析 | [Normalization技术](/notes/ai-model/llm/pretraining#10-normalization-技术详解) | BN/LN/IN/GN对比 |
| 3.3.5 训练过程监控 | [训练监控和调试](/notes/ai-model/llm/pretraining#9-训练监控和调试) | 损失曲线、梯度监控 |
| 3.3.6 CheckPoint评估 | [Checkpoint管理](/notes/ai-model/llm/pretraining#11-checkpoint-管理) | 检查点保存与恢复 |

### 3.4 预训练优化技术

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 3.4.1 梯度积累 | [训练优化策略](/notes/ai-model/llm/pretraining#72-梯度累积与裁剪) | GradientAccumulator实现 |
| 3.4.2 学习率调度 | [学习率调度](/notes/ai-model/llm/pretraining#71-学习率调度) | Warmup、余弦退火 |
| 3.4.3 正则化方法 | [正则化与归一化](/notes/deep-learning/04-regularization) | Dropout、Weight Decay |
| 3.4.4 Epoch对性能的影响 | [训练优化策略](/notes/ai-model/llm/pretraining#7-训练优化策略) | 训练轮数选择 |
| 3.4.5 数据重复对性能的影响 | [数据重复影响](/notes/ai-model/llm/pretraining#12-数据重复对性能的影响) | 重复数据问题 |

### 3.5 增量预训练

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 3.5.1 为什么要增量预训练 | [增量预训练概述](/notes/ai-model/llm/continued-pretraining#基本概念) | 领域自适应需求 |
| 3.5.2 如何确定数据量 | [增量预训练](/notes/ai-model/llm/continued-pretraining#数据量确定) | 数据量估算方法 |
| 3.5.3 增量预训练流程 | [增量预训练](/notes/ai-model/llm/continued-pretraining#增量预训练流程) | 完整训练流水线 |

---

## 第四章. LLM之Post-Training

### 4.1 有监督微调(SFT)

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 4.1.1 SFT的基本概念 | [微调与对齐-指令微调](/notes/ai-model/llm/finetuning-alignment#指令微调instruction-tuning原理) | 指令微调原理 |
| 4.1.2 SFT的目标是什么 | [微调与对齐-指令微调目标](/notes/ai-model/llm/finetuning-alignment#指令微调目标) | 对齐用户意图 |
| 4.1.3 数据质量、数量及多样性 | [微调与对齐-数据准备](/notes/ai-model/llm/finetuning-alignment#数据准备) | 数据工程实践 |
| 4.1.4 参数高效微调 | [微调与对齐-LoRA](/notes/ai-model/llm/finetuning-alignment#lora低秩自适应) | Prompting/Adapter/LoRA |
| 4.1.5 模型融合 | [微调与对齐-模型融合](/notes/ai-model/llm/finetuning-alignment#模型融合) | 模型权重融合 |

### 4.2 偏好学习与对齐

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 4.2.1 偏好学习的定义 | [大模型对齐技术](/notes/reinforcement-learning/llm-alignment) | 人类偏好建模 |
| 4.2.2 SFT→RLHF流程原理 | [RLHF原理与实践](/notes/reinforcement-learning/llm-rl) | 三阶段训练原理 |
| 4.2.3 Reward Model训练 | [奖励模型](/notes/reinforcement-learning/llm-rl#奖励模型rm) | 奖励模型训练方法 |
| 4.2.4 Reward Hacking现象 | [Reward Hacking](/notes/reinforcement-learning/llm-alignment#reward-hacking奖励投机) | 奖励投机问题 |
| 4.2.5 经典偏好学习算法 | [DPO直接偏好优化](/notes/reinforcement-learning/dpo-preference-optimization) | DPO/RLOO/GRPO |

### 4.3 如何提升特定领域能力

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 4.3.1 长上下文理解 | [微调与对齐-领域增强](/notes/ai-model/llm/finetuning-alignment#领域增强) | 长文本处理技术 |
| 4.3.2 数学推理 | [微调与对齐-领域增强](/notes/ai-model/llm/finetuning-alignment#领域增强) | 数学能力提升 |
| 4.3.3 代码能力 | [微调与对齐-领域增强](/notes/ai-model/llm/finetuning-alignment#领域增强) | 代码生成能力 |
| 4.3.4 多语言理解 | [微调与对齐-领域增强](/notes/ai-model/llm/finetuning-alignment#领域增强) | 多语言支持 |
| 4.3.5 工具使用 | [ReAct与工具调用](/notes/ai-applications/agent/agent-react-tools) | 工具调用能力 |

### 4.4 合成数据的利用

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 4.4.1 合成数据的产品化应用 | [微调与对齐-合成数据](/notes/ai-model/llm/finetuning-alignment#合成数据) | 产品级应用场景 |
| 4.4.2 预训练阶段的合成数据应用 | [预训练技术](/notes/ai-model/llm/pretraining#2-大规模数据处理技术) | 预训练数据增强 |
| 4.4.3 合成数据的重点发展方向 | [微调与对齐-合成数据](/notes/ai-model/llm/finetuning-alignment#合成数据) | 发展趋势 |
| 4.4.4 合成数据与模型蒸馏 | [推理优化-知识蒸馏](/notes/ai-model/llm/inference-optimization#知识蒸馏) | 知识蒸馏技术 |

---

## 第五章. LLM之分布式训练

### 5.1 基本概念

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 5.1.1 节点编号 | [分布式基础-硬件拓扑](/notes/ai-model/distributed-training/basics#硬件拓扑) | Node概念 |
| 5.1.2 全局进程编号 | [分布式基础-进程组](/notes/ai-model/distributed-training/basics#进程组-process-group) | Global Rank |
| 5.1.3 局部进程编号 | [分布式基础-硬件拓扑](/notes/ai-model/distributed-training/basics#硬件拓扑) | Local Rank |
| 5.1.4 全局总进程数 | [分布式基础-进程组](/notes/ai-model/distributed-training/basics#进程组-process-group) | World Size |
| 5.1.5 主节点 | [分布式基础-硬件拓扑](/notes/ai-model/distributed-training/basics#硬件拓扑) | Master Node |

### 5.2 模型参数的存储格式

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 5.2.1 FP16格式 | [模型规模估算-精度类型](/notes/ai-model/llm/model-scaling#精度类型对比) | 半精度浮点 |
| 5.2.2 BF16格式 | [模型规模估算-精度类型](/notes/ai-model/llm/model-scaling#精度类型对比) | Brain Float16 |
| 5.2.3 FP32格式 | [模型规模估算-精度类型](/notes/ai-model/llm/model-scaling#精度类型对比) | 单精度浮点 |
| 5.2.4 显存问题 | [模型规模估算-显存估算](/notes/ai-model/llm/model-scaling#显存估算) | 精度与显存权衡 |

### 5.3 并行策略详解

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 5.3.1 数据并行 | [数据并行](/notes/ai-model/distributed-training/data-parallel) | DDP、FSDP |
| 5.3.2 张量并行 | [张量并行](/notes/ai-model/distributed-training/tensor-parallel) | Megatron-LM风格 |
| 5.3.3 流水线并行 | [流水线并行](/notes/ai-model/distributed-training/pipeline-parallel) | GPipe、PipeDream |
| 5.3.4 序列并行 | [张量并行-序列并行](/notes/ai-model/distributed-training/tensor-parallel#序列并行) | Sequence Parallelism |
| 5.3.5 多维度混合并行 | [分布式训练概述](/notes/ai-model/distributed-training/) | 3D/4D并行 |

### 5.4 DeepSpeed原理介绍

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 5.4.1 显存占用分析 | [DeepSpeed实践](/notes/ai-model/distributed-training/deepspeed) | 显存组成分析 |
| 5.4.2 混合精度训练 | [混合精度训练](/notes/ai-model/llm/pretraining#5-混合精度训练) | AMP原理与实现 |
| 5.4.3 Ring-AllReduce算法 | [分布式基础-集合通信](/notes/ai-model/distributed-training/basics#集合通信操作) | 通信原语 |
| 5.4.4 ZeRO优化与卸载 | [ZeRO优化](/notes/ai-model/distributed-training/zero-optimization) | ZeRO-1/2/3详解 |
| 5.4.5 通信量分析 | [DeepSpeed实践](/notes/ai-model/distributed-training/deepspeed) | 通信开销分析 |

---

## 第六章. LLM之加速训练及推理

### 6.1 FlashAttention V1

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 6.1.1 计算限制与存储约束 | [FlashAttention](/notes/ai-model/llm/attention-mechanisms#八-flashattention) | IO感知注意力 |
| 6.1.2 GPU上的数据存储与处理 | [FlashAttention](/notes/ai-model/llm/attention-mechanisms#八-flashattention) | HBM与SRAM |
| 6.1.3 前向计算流程 | [FlashAttention](/notes/ai-model/llm/attention-mechanisms#八-flashattention) | 分块计算 |
| 6.1.4 反向传播计算流程 | [FlashAttention](/notes/ai-model/llm/attention-mechanisms#八-flashattention) | 重计算策略 |
| 6.1.5 计算负荷和显存需求 | [FlashAttention](/notes/ai-model/llm/attention-mechanisms#八-flashattention) | 复杂度分析 |

### 6.2 FlashAttention V2

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 6.2.1 整体流程 | [推理优化-FlashAttention](/notes/ai-model/llm/inference-optimization#flashattention-v2) | V2架构 |
| 6.2.2 相比于V1的优化 | [推理优化-FlashAttention](/notes/ai-model/llm/inference-optimization#flashattention-v2) | 并行优化 |
| 6.2.3 Thread Blocks | [推理优化-FlashAttention](/notes/ai-model/llm/inference-optimization#flashattention-v2) | 线程块优化 |
| 6.2.4 Warp-level并行 | [推理优化-FlashAttention](/notes/ai-model/llm/inference-optimization#flashattention-v2) | Warp级并行 |

### 6.3 VLLM系列

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 6.3.1 整体架构 | [推理优化-vLLM](/notes/ai-model/llm/inference-optimization#vllm推理框架) | vLLM架构设计 |
| 6.3.2 模型加载与显存分配 | [推理优化-PagedAttention](/notes/ai-model/llm/inference-optimization#pagedattention核心原理) | PagedAttention |
| 6.3.3 调度策略 | [推理优化-vLLM](/notes/ai-model/llm/inference-optimization#vllm核心特性) | 连续批处理 |
| 6.3.4 Block管理器 | [推理优化-vLLM](/notes/ai-model/llm/inference-optimization#block管理器) | KV Cache分页管理 |

### 6.4 常见推理问题

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 6.4.1 推理时显存问题 | [推理优化-显存管理](/notes/ai-model/llm/inference-optimization#显存管理) | 显存管理与释放 |
| 6.4.2 INT8 vs FP16速度对比 | [推理优化-量化对比](/notes/ai-model/llm/inference-optimization#量化压缩技术) | 量化推理 |
| 6.4.3 推理参数设置 | [推理优化-参数设置](/notes/ai-model/llm/inference-optimization#推理参数) | temperature/TopK/TopP |

---

## 第七章. LLM之性能评估

### 7.1 大模型幻觉

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 7.1.1 什么是LLM的幻觉 | [模型幻觉-定义](/notes/ai-model/llm/hallucination#什么是幻觉) | 幻觉定义与分类 |
| 7.1.2 幻觉一定是有害的吗 | [模型幻觉-影响](/notes/ai-model/llm/hallucination#幻觉的负面影响) | 创造性vs事实性 |
| 7.1.3 为什么需要解决LLM的幻觉问题 | [模型幻觉-影响](/notes/ai-model/llm/hallucination#幻觉的负面影响) | 可信度需求 |

### 7.2 幻觉来源与缓解

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 7.2.1 幻觉分类 | [模型幻觉-分类](/notes/ai-model/llm/hallucination#幻觉的分类) | 事实性/忠实性幻觉 |
| 7.2.2 幻觉检测 | [模型幻觉-检测](/notes/ai-model/llm/hallucination#幻觉检测) | 检测方法与工具 |
| 7.2.3 幻觉来源 | [模型幻觉-来源](/notes/ai-model/llm/hallucination#幻觉来源分析) | 数据、模型、训练层面 |
| 7.2.4 幻觉缓解 | [模型幻觉-缓解](/notes/ai-model/llm/hallucination#幻觉缓解) | RAG、自我验证等 |

### 7.3 评测方案

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 7.3.1 LLM的评估方法 | [模型评测-方法论](/notes/ai-model/llm/evaluation#评测方法论) | 基准测试、LLM-as-Judge |
| 7.3.2 3H原则 | [模型评测-3H原则](/notes/ai-model/llm/evaluation#3h原则) | Helpfulness/Honesty/Harmlessness |
| 7.3.3 如何衡量LLM水平 | [模型评测-通用能力](/notes/ai-model/llm/evaluation#通用能力评测基准) | MMLU、HumanEval等基准 |
| 7.3.4 LLM的评估工具 | [模型评测-评估工具](/notes/ai-model/llm/evaluation#评估工具) | lm-eval、OpenCompass |

---

## 第八章. LLM之检索增强

### 8.1 RAG解决的问题

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 8.1.1 长尾知识 | [RAG基础-长尾知识](/notes/ai-applications/rag/rag-basics#问题一长尾知识缺失) | 低频知识覆盖 |
| 8.1.2 私有数据 | [RAG基础-私有数据](/notes/ai-applications/rag/rag-basics#问题二私有数据处理) | 企业数据整合 |
| 8.1.3 数据时效性 | [RAG基础-时效性](/notes/ai-applications/rag/rag-basics#问题三数据时效性) | 知识更新问题 |
| 8.1.4 来源验证和可解释性 | [RAG基础-可解释性](/notes/ai-applications/rag/rag-basics#问题四可解释性) | 引用溯源 |

### 8.2 RAG的关键模块

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 8.2.1 数据和索引模块 | [检索器模块](/notes/ai-applications/rag/retriever) | 向量数据库、文档索引 |
| 8.2.2 查询和检索模块 | [检索器模块](/notes/ai-applications/rag/retriever) | Embedding、检索策略 |
| 8.2.3 回复生成模块 | [生成器模块](/notes/ai-applications/rag/generator) | Prompt设计、上下文融合 |

### 8.3 RAG vs SFT

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 8.3.1 数据层面 | [RAG vs SFT](/notes/ai-applications/rag/rag-vs-sft) | 数据需求对比 |
| 8.3.2 外部知识库 | [RAG vs SFT](/notes/ai-applications/rag/rag-vs-sft) | 知识管理方式 |
| 8.3.3 模型定制 | [RAG vs SFT](/notes/ai-applications/rag/rag-vs-sft) | 定制化程度 |
| 8.3.4 缓解幻觉 | [RAG vs SFT](/notes/ai-applications/rag/rag-vs-sft) | 幻觉问题处理 |

### 8.4 Agents技术

| 知识点 | 链接 | 说明 |
|--------|------|------|
| 8.4.1 LLM Agents综述 | [Agent基础概念](/notes/ai-applications/agent/agent-basics) | Agent概念与架构 |
| 8.4.2 通用智能核心原则 | [Agent基础概念](/notes/ai-applications/agent/agent-basics) | 自主决策能力 |
| 8.4.3 目标导向的系统架构 | [多Agent编排](/notes/ai-applications/agent/agent-orchestration) | 多Agent协作 |
| 8.4.4 前瞻性分析 | [Agent导览](/notes/ai-applications/agent/) | 发展趋势 |

---

## 学习建议

1. **入门路径**：[Transformer架构](/notes/deep-learning/07-transformer) → [注意力机制](/notes/ai-model/llm/attention-mechanisms) → [分词技术](/notes/ai-model/llm/tokenization) → [嵌入层](/notes/ai-model/llm/embedding)

2. **进阶路径**：[预训练技术](/notes/ai-model/llm/pretraining) → [微调与对齐](/notes/ai-model/llm/finetuning-alignment) → [RLHF](/notes/reinforcement-learning/llm-rl) → [分布式训练](/notes/ai-model/distributed-training/)

3. **实践路径**：[推理优化](/notes/ai-model/llm/inference-optimization) → [RAG](/notes/ai-applications/rag/) → [Agent开发](/notes/ai-applications/agent/)