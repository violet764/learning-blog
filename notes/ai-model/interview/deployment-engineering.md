# 模型部署与工程化面试题

本章节整理了大模型部署与工程化相关的面试题目，涵盖推理优化、分布式训练、模型压缩、服务架构等核心技术。

---

## 一、推理优化基础

### Q1: 大模型推理面临哪些主要挑战？

**基础回答：**

大模型推理主要面临显存占用大、计算延迟高、吞吐量低、成本高昂等挑战。

**深入回答：**

**挑战分析**：

```
1. 显存挑战
   ├── 模型参数: 7B 模型 FP16 需要 14GB
   ├── KV Cache: 长序列占用大量显存
   ├── 激活值: 中间计算结果存储
   └── 框架开销: PyTorch 等框架额外占用

2. 计算挑战
   ├── 自回归生成: 逐 token 生成，无法并行
   ├── Attention 复杂度: O(n²) 随序列长度增长
   ├── 内存带宽: 访存成为瓶颈
   └── GPU 利用率: 小批量时利用率低

3. 吞吐量挑战
   ├── 单请求延迟高
   ├── 批处理效率低
   └── 资源调度困难

4. 成本挑战
   ├── 硬件成本: 高端 GPU 昂贵
   ├── 运营成本: 电费、机房
   └── 开发成本: 工程实现复杂
```

**追问：如何量化显存占用？**

```python
# 模型参数显存
params_memory = num_params * bytes_per_param
# LLaMA 7B FP16: 7B * 2 = 14GB

# KV Cache 显存
kv_memory = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * 2
# LLaMA 7B, seq_len=2048, batch=1: 约 2GB

# 激活值显存（推理时通常不需要保存）
activation_memory = batch_size * seq_len * hidden_dim * n_layers

# 总显存 ≈ params + kv_cache + overhead
```

---

### Q2: KV Cache 的原理和优化方法？

**基础回答：**

KV Cache 通过缓存历史 Key 和 Value，避免重复计算，是自回归模型推理的核心优化。

**深入回答：**

**基本原理**：

```python
# 无 KV Cache
for t in range(max_len):
    # 每次都重新计算所有位置的 K, V
    K, V = compute_kv(input_ids[:t+1])
    output = attention(Q[t], K, V)

# 有 KV Cache
cache = []
for t in range(max_len):
    # 只计算新位置的 K, V
    K_t, V_t = compute_kv(input_ids[t])
    cache.append((K_t, V_t))
    # 使用缓存
    K, V = torch.cat([c[0] for c in cache]), torch.cat([c[1] for c in cache])
    output = attention(Q[t], K, V)
```

**优化方法**：

| 方法 | 原理 | 效果 |
|------|------|------|
| **MQA/GQA** | 多头共享 KV | 减少 4-8 倍 |
| **KV Cache 量化** | INT8/INT4 存储 | 减少 2-4 倍 |
| **PagedAttention** | 按需分配 | 减少碎片 |
| **滑动窗口** | 只保留最近 N 个 | 固定大小 |
| **Streaming LLM** | 保留注意力汇聚点 | 无限长 |

**追问：PagedAttention 如何优化 KV Cache？**

```
传统 KV Cache:
├── 预分配最大长度内存
├── 实际使用长度不确定
└── 内存碎片严重

PagedAttention (vLLM):
├── 将 KV Cache 分成固定大小 block (如 16 tokens)
├── 按需分配 block
├── 类似操作系统的虚拟内存管理
├── 支持 block 共享（parallel sampling）

内存利用率: 从 ~20% 提升到 ~96%
```

---

### Q3: vLLM 的核心优化技术是什么？

**基础回答：**

vLLM 通过 PagedAttention、连续批处理、优化的 CUDA kernel 等技术，显著提升大模型推理吞吐量。

**深入回答：**

**核心技术**：

```
vLLM 三大核心:

1. PagedAttention
   ├── Block 级内存管理
   ├── 按需分配，消除预分配浪费
   ├── 支持内存共享
   └── 支持 preemption 和 swapping

2. Continuous Batching
   ├── 请求完成即释放资源
   ├── 新请求随时加入
   └── 最大化 GPU 利用率

3. 优化的 CUDA Kernel
   ├── Paged Attention Kernel
   ├── 融合算子
   └── 高效内存访问
```

**追问：Continuous Batching 如何工作？**

```python
# 传统静态批处理
batch = get_requests(batch_size)
for request in batch:
    # 所有请求完成后才开始下一批
    generate_until_done(request)

# Continuous Batching
while has_requests():
    # 新请求随时加入
    batch = get_running_requests() + get_new_requests()
    
    # 每个请求独立生成
    for request in batch:
        token = generate_one_token(request)
        if request.done:
            # 完成即移除，释放资源
            remove_from_batch(request)
```

**追问：vLLM 和 HuggingFace Transformers 性能对比？**

| 场景 | vLLM | HF Transformers |
|------|------|-----------------|
| **吞吐量** | 10-20x 更高 | 基准 |
| **显存效率** | ~96% | ~20-40% |
| **延迟** | 相当或更低 | 基准 |
| **功能完整** | 较新，快速迭代 | 完整 |

---

## 二、量化技术

### Q4: 模型量化的原理是什么？

**基础回答：**

量化将模型参数从高精度（FP32/FP16）转换为低精度（INT8/INT4），减少模型大小和计算量。

**深入回答：**

**量化公式**：

```python
# 量化 (FP → INT)
Q = round(R / scale) + zero_point

# 反量化 (INT → FP)
R = scale * (Q - zero_point)

# scale: 缩放因子
# zero_point: 零点偏移
```

**量化分类**：

| 类型 | 方法 | 精度损失 | 训练需求 |
|------|------|----------|----------|
| **PTQ (训练后量化)** | 直接量化 | 较大 | 无 |
| **QAT (量化感知训练)** | 训练时模拟量化 | 较小 | 需要 |
| **混合精度** | 部分层量化 | 中等 | 可选 |

**对称 vs 非对称量化**：

```python
# 对称量化 (zero_point = 0)
scale = max(|W|) / 127  # INT8
Q = round(W / scale)

# 非对称量化
scale = (max(W) - min(W)) / 255
zero_point = round(-min(W) / scale)
Q = round(W / scale) + zero_point
```

---

### Q5: GPTQ 和 AWQ 的区别是什么？

**基础回答：**

GPTQ 基于 OBQ（Optimal Brain Quantization）逐层量化，AWQ 基于激活感知保护重要权重。

**深入回答：**

**GPTQ 原理**：

```
GPTQ 核心:
1. 逐层量化
2. 使用校准数据计算 Hessian 矩阵
3. 按重要性顺序量化权重
4. 量化后更新未量化权重以补偿误差

公式:
argmin ||W_q x - W x||²
通过求解 Hessian 矩阵逆来更新权重
```

**AWQ 原理**：

```
AWQ 核心发现:
1. 权重重要性不同
2. 重要权重对激活值大的通道影响大
3. 保护重要权重（保持高精度或缩小量化范围）

方法:
1. 分析激活值确定重要权重
2. 对重要权重应用更小的量化范围
3. 通过缩放因子调整
```

**对比**：

| 方面 | GPTQ | AWQ |
|------|------|-----|
| **量化速度** | 较慢（需校准） | 较快 |
| **推理效果** | 略好 | 相当 |
| **显存需求** | 量化时较高 | 较低 |
| **实现复杂度** | 中等 | 较低 |

---

### Q6: INT4 量化如何保证精度？

**基础回答：**

INT4 量化精度损失较大，需要通过特殊技术如 GPTQ-AWQ、Double Quantization 等方法缓解。

**深入回答：**

**精度保证方法**：

```
1. 选择性量化
   ├── 保留某些层为高精度
   ├── Embedding 层通常保持 FP16
   └── Attention 输出层保持高精度

2. Double Quantization (QLoRA)
   ├── 量化量化参数本身
   ├── 进一步减少显存
   └── 通常使用 NF4 (NormalFloat4)

3. 激活感知 (AWQ)
   ├── 保护重要权重
   └── 基于激活值动态调整

4. 混合精度
   ├── 不同层使用不同精度
   └── 根据敏感度选择
```

**精度损失评估**：

| 量化精度 | 模型大小 | 典型效果损失 |
|----------|----------|--------------|
| FP16 | 100% | 0% |
| INT8 | 50% | <1% |
| INT4 | 25% | 1-3% |
| INT3 | 19% | 3-5% |

---

## 三、推理框架

### Q7: 主流 LLM 推理框架有哪些？各有什么特点？

**基础回答：**

主流推理框架包括 vLLM、TensorRT-LLM、llama.cpp、ONNX Runtime 等，各有适用场景。

**深入回答：**

**框架对比**：

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **vLLM** | PagedAttention、高吞吐 | 高并发服务 |
| **TensorRT-LLM** | NVIDIA 优化、极致性能 | NVIDIA GPU |
| **llama.cpp** | CPU 支持、跨平台 | 边缘设备、个人使用 |
| **ONNX Runtime** | 通用性好 | 多后端部署 |
| **MLC-LLM** | 多平台支持 | 移动端部署 |
| **TGI** | HuggingFace 生态 | 快速部署 |

**追问：如何选择推理框架？**

```
选择决策树:

高并发生产环境?
├── 是 → vLLM 或 TensorRT-LLM
│   ├── NVIDIA GPU 优先 → TensorRT-LLM
│   └── 多种 GPU → vLLM
│
└── 否 → 个人/边缘使用?
    ├── 是 → llama.cpp
    └── 否 → ONNX Runtime 或 TGI

其他考虑:
├── 是否需要流式输出
├── 是否需要多模态支持
├── 是否需要量化支持
└── 社区活跃度和文档完善度
```

---

### Q8: TensorRT-LLM 的核心优化技术？

**基础回答：**

TensorRT-LLM 是 NVIDIA 推出的高性能推理框架，通过算子融合、量化优化、内核优化等技术实现极致性能。

**深入回答：**

**核心优化**：

```
1. 算子融合
   ├── Attention 融合: QKV 计算 + Softmax + 输出
   ├── MLP 融合: 激活函数 + 矩阵乘
   └── LayerNorm 融合

2. 量化优化
   ├── INT8/INT4 权重量化
   ├── INT8 激活量化
   └── SmoothQuant 量化方案

3. 内核优化
   ├── FlashAttention 集成
   ├── Tensor Core 优化
   └── 内存访问优化

4. Inflight Batching
   ├── 类似 Continuous Batching
   └── NVIDIA 特定优化

5. Multi-block Attention
   ├── 长序列优化
   └── 并行计算优化
```

**追问：TensorRT-LLM vs vLLM？**

| 方面 | TensorRT-LLM | vLLM |
|------|--------------|------|
| **性能** | 极致（NVIDIA 优化） | 非常好 |
| **硬件支持** | 仅 NVIDIA | 多种 GPU |
| **易用性** | 需要编译 | 开箱即用 |
| **功能** | 较完整 | 快速迭代 |
| **社区** | NVIDIA 维护 | 开源社区 |

---

## 四、分布式推理

### Q9: 大模型分布式推理的主要策略有哪些？

**基础回答：**

分布式推理策略包括张量并行、流水线并行、序列并行等，根据模型大小和硬件配置选择。

**深入回答：**

**并行策略**：

```
1. 张量并行 (Tensor Parallelism, TP)
   ├── 层内切分
   ├── Attention 头切分到不同 GPU
   ├── FFN 列切分
   └── 通信点: Attention 输出、FFN 输出

2. 流水线并行 (Pipeline Parallelism, PP)
   ├── 层间切分
   ├── 不同层在不同 GPU
   └── 通信点: 层间激活值

3. 序列并行 (Sequence Parallelism, SP)
   ├── 序列维度切分
   ├── 用于超长序列
   └── Ring Attention 实现

4. 专家并行 (Expert Parallelism, EP)
   ├── MoE 模型专用
   ├── 不同专家在不同 GPU
   └── 动态路由通信
```

**并行选择决策**：

```
单卡能放下?
├── 是 → 单卡推理
└── 否 → 需要多卡
    ├── 单机多卡 → 张量并行
    └── 多机多卡 → TP + PP
        ├── TP 通信频繁，适合 NVLink
        └── PP 通信少，适合多机

序列超长?
└── 加入序列并行
```

**追问：TP 和 PP 的通信开销对比？**

```python
# 张量并行通信
# 每个 Transformer 层需要 2 次 All-Reduce
communication_per_layer = 2 * hidden_dim * 4  # FP32

# 流水线并行通信
# 每个设备边界需要 1 次 P2P 通信
communication_per_boundary = batch_size * seq_len * hidden_dim

# TP 通信频繁但数据量固定
# PP 通信少但数据量随 batch 和 seq 增大
# 实际选择需要根据具体场景权衡
```

---

### Q10: 如何实现高吞吐推理服务？

**基础回答：**

高吞吐推理需要优化批处理策略、请求调度、缓存策略等多个方面。

**深入回答：**

**优化策略**：

```
1. 批处理优化
   ├── Continuous Batching
   ├── 动态批量大小调整
   └── 请求分组（相似长度）

2. 调度优化
   ├── 优先级队列
   ├── 预测生成长度
   └── 资源分配策略

3. 缓存优化
   ├── Prefix Caching: 共享前缀缓存
   ├── 语义缓存: 相似查询复用
   └── 结果缓存: 完全相同查询

4. 资源优化
   ├── GPU 显存池化
   ├── 多模型共享 GPU
   └── 自动扩缩容
```

**追问：Prefix Caching 如何实现？**

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}  # prefix_hash -> kv_cache
    
    def get(self, prefix_tokens):
        """检查是否有匹配的 prefix 缓存"""
        prefix_hash = hash(tuple(prefix_tokens))
        return self.cache.get(prefix_hash)
    
    def set(self, prefix_tokens, kv_cache):
        """缓存 prefix 的 KV Cache"""
        prefix_hash = hash(tuple(prefix_tokens))
        self.cache[prefix_hash] = kv_cache

# 使用场景: System Prompt 相同时复用 KV Cache
# 避免重复计算
```

---

## 五、模型压缩

### Q11: 模型压缩的主要方法有哪些？

**基础回答：**

模型压缩方法包括量化、剪枝、知识蒸馏、低秩分解等，各有优缺点。

**深入回答：**

**压缩方法对比**：

| 方法 | 原理 | 压缩比 | 精度损失 | 难度 |
|------|------|--------|----------|------|
| **量化** | 降低精度 | 2-8x | 小 | 低 |
| **剪枝** | 移除不重要的参数 | 2-10x | 中 | 中 |
| **蒸馏** | 用大模型教小模型 | 取决于模型 | 小 | 高 |
| **低秩分解** | 矩阵分解 | 2-4x | 小 | 中 |

**追问：结构化剪枝 vs 非结构化剪枝？**

```python
# 非结构化剪枝
# 移除单个权重，产生稀疏矩阵
mask = (|W| > threshold)
W_pruned = W * mask
# 优点: 压缩比高
# 缺点: 需要稀疏计算支持

# 结构化剪枝
# 移除整行/列/通道
importance = compute_importance(W)
keep_indices = top_k(importance, k)
W_pruned = W[keep_indices, :]
# 优点: 标准硬件支持
# 缺点: 精度损失可能更大
```

**追问：知识蒸馏如何应用于 LLM？**

```
LLM 蒸馏方法:

1. 白盒蒸馏
   ├── 使用 Teacher 的 logits
   ├── KL 散度损失
   └── 需要访问 Teacher 内部

2. 黑盒蒸馏
   ├── 只使用 Teacher 输出
   ├── 构建合成数据
   └── 不需要访问内部

3. 渐进式蒸馏
   ├── 从小模型开始
   ├── 逐步增加规模
   └── 减少训练成本

示例: DistilGPT, Alpaca (Self-Instruct)
```

---

### Q12: 如何评估压缩后模型的质量？

**基础回答：**

评估压缩模型需要考虑基准测试、特定任务性能、延迟吞吐等多个维度。

**深入回答：**

**评估维度**：

```
1. 精度评估
   ├── 通用基准 (MMLU, C-Eval)
   ├── 任务特定指标
   ├── 与原始模型对比
   └── 退化分析

2. 效率评估
   ├── 推理延迟
   ├── 吞吐量
   ├── 显存占用
   └── 能耗

3. 稳定性评估
   ├── 输出一致性
   ├── 极端情况处理
   └── 鲁棒性测试

4. 实用性评估
   ├── 部署复杂度
   ├── 兼容性
   └── 维护成本
```

**评估代码示例**：

```python
def evaluate_compressed_model(original_model, compressed_model, test_data):
    results = {}
    
    # 1. 精度对比
    original_acc = evaluate(original_model, test_data)
    compressed_acc = evaluate(compressed_model, test_data)
    results['accuracy_drop'] = original_acc - compressed_acc
    
    # 2. 延迟对比
    original_latency = measure_latency(original_model)
    compressed_latency = measure_latency(compressed_model)
    results['speedup'] = original_latency / compressed_latency
    
    # 3. 显存对比
    original_memory = measure_memory(original_model)
    compressed_memory = measure_memory(compressed_model)
    results['memory_reduction'] = original_memory / compressed_memory
    
    return results
```

---

## 六、生产部署

### Q13: 如何设计高可用的 LLM 服务架构？

**基础回答：**

高可用 LLM 服务需要考虑负载均衡、容错机制、监控告警、弹性伸缩等方面。

**深入回答：**

**架构设计**：

```
┌─────────────────────────────────────────────────────────┐
│                    高可用 LLM 服务架构                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  接入层                                                  │
│  ├── API Gateway (认证、限流、路由)                       │
│  ├── 负载均衡器 (Round Robin / Least Connections)        │
│  └── 协议转换 (REST / gRPC / WebSocket)                  │
│                                                         │
│  服务层                                                  │
│  ├── 多实例部署 (多 AZ / 多 Region)                       │
│  ├── 健康检查 (心跳、指标监控)                            │
│  └── 故障转移 (自动摘除、自动恢复)                        │
│                                                         │
│  推理层                                                  │
│  ├── GPU 集群 (多副本、异构)                             │
│  ├── 模型服务 (vLLM / TensorRT-LLM)                      │
│  └── 批处理队列 (Continuous Batching)                    │
│                                                         │
│  存储层                                                  │
│  ├── 模型存储 (对象存储)                                 │
│  ├── 缓存层 (Redis / Prefix Cache)                       │
│  └── 日志/监控存储                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**追问：如何实现优雅降级？**

```
降级策略:

1. 模型降级
   ├── 主模型不可用时切换到小模型
   ├── 提示用户模型能力下降
   └── 关键任务优先分配资源

2. 功能降级
   ├── 关闭非核心功能
   ├── 减少生成长度
   └── 简化提示处理

3. 质量降级
   ├── 降低采样精度
   ├── 使用更快的解码策略
   └── 减少候选数量

4. 流量降级
   ├── 限流保护
   ├── 排队等待
   └── 拒绝非关键请求
```

---

### Q14: 如何进行推理服务的容量规划？

**基础回答：**

容量规划需要考虑 QPS、延迟要求、峰值流量、冗余度等因素。

**深入回答：**

**规划方法**：

```python
def capacity_planning(target_qps, latency_sla, model_config):
    # 1. 单实例性能基准
    single_instance_qps = benchmark_single_instance(model_config)
    single_instance_latency = benchmark_latency(model_config)
    
    # 2. 考虑延迟 SLA
    if single_instance_latency > latency_sla:
        # 需要优化或选择更快模型
        pass
    
    # 3. 计算所需实例数
    # 考虑峰值因子和冗余
    peak_factor = 1.5  # 峰值流量倍数
    redundancy = 1.2   # 冗余度
    instances_needed = (target_qps * peak_factor / single_instance_qps) * redundancy
    
    # 4. GPU 资源计算
    gpus_per_instance = model_config.gpus_needed
    total_gpus = instances_needed * gpus_per_instance
    
    return {
        'instances': math.ceil(instances_needed),
        'gpus': math.ceil(total_gpus),
        'estimated_latency': single_instance_latency
    }
```

**容量规划清单**：

| 项目 | 考虑因素 |
|------|----------|
| **流量预估** | 日均 QPS、峰值 QPS、增长趋势 |
| **延迟要求** | P50/P95/P99 延迟 SLA |
| **模型选择** | 模型大小 vs 性能权衡 |
| **硬件配置** | GPU 型号、数量、网络 |
| **冗余设计** | 多 AZ 部署、故障转移 |
| **弹性伸缩** | 自动扩缩容策略 |

---

### Q15: 推理服务的监控指标有哪些？

**基础回答：**

推理服务监控包括业务指标、系统指标、模型指标等多个维度。

**深入回答：**

**监控指标体系**：

```
1. 业务指标
   ├── QPS (每秒请求数)
   ├── 成功率
   ├── 响应时间 (P50/P95/P99)
   ├── 错误率 (按错误类型分类)
   └── 用户满意度

2. 系统指标
   ├── GPU 利用率
   ├── GPU 显存使用率
   ├── GPU 温度
   ├── CPU 利用率
   ├── 内存使用率
   ├── 网络带宽
   └── 磁盘 I/O

3. 模型指标
   ├── 生成长度分布
   ├── 输入长度分布
   ├── Token 吞吐量
   ├── KV Cache 命中率
   └── 批处理效率

4. 成本指标
   ├── GPU 小时成本
   ├── 单次请求成本
   ├── 单 token 成本
   └── 能耗
```

**告警策略**：

```yaml
# 告警规则示例
alerts:
  - name: high_latency
    condition: p99_latency > 2000ms
    severity: warning
    
  - name: gpu_memory_high
    condition: gpu_memory_usage > 90%
    severity: warning
    
  - name: error_rate_high
    condition: error_rate > 1%
    severity: critical
    
  - name: service_down
    condition: instance_health < 50%
    severity: critical
```

---

## 七、工程实践

### Q16: 如何优化推理服务的冷启动时间？

**基础回答：**

冷启动优化包括模型预加载、权重预热、请求预热等方法。

**深入回答：**

**优化策略**：

```
1. 模型加载优化
   ├── 预加载模型到内存
   ├── 使用 mmap 加载
   ├── 模型分片并行加载
   └── 压缩权重格式

2. 预热优化
   ├── 预热 CUDA Kernel
   ├── 预热内存分配
   ├── 发送预热请求
   └── 预编译计算图

3. 架构优化
   ├── 保持最小实例数
   ├── 预启动备用实例
   ├── 模型权重共享
   └── 镜像预热
```

**预热代码示例**：

```python
def warmup_model(model, warmup_prompts):
    """模型预热"""
    model.eval()
    with torch.no_grad():
        for prompt in warmup_prompts:
            # 发送几个预热请求
            _ = model.generate(prompt, max_length=50)
    
    # 清空 CUDA 缓存
    torch.cuda.empty_cache()
    
    print("Model warmup complete")
```

---

### Q17: 如何处理推理服务的长尾延迟？

**基础回答：**

长尾延迟优化需要分析原因，针对性解决 GC 停顿、批处理抖动、资源竞争等问题。

**深入回答：**

**长尾延迟原因**：

| 原因 | 解决方法 |
|------|----------|
| **GC 停顿** | 对象池化、减少临时对象 |
| **批处理抖动** | 连续批处理、请求对齐 |
| **资源竞争** | 资源隔离、优先级队列 |
| **网络延迟** | 连接复用、就近部署 |
| **冷请求** | 请求预热、缓存预热 |

**优化实践**：

```python
# 1. 请求超时和重试
def infer_with_timeout(request, timeout=30):
    try:
        result = model.generate(request, timeout=timeout)
        return result
    except TimeoutError:
        # 快速失败，避免堆积
        return fallback_response()

# 2. 请求优先级队列
class PriorityQueue:
    def __init__(self):
        self.high_priority = Queue()
        self.normal_priority = Queue()
    
    def get_next_request(self):
        if not self.high_priority.empty():
            return self.high_priority.get()
        return self.normal_priority.get()

# 3. 请求取消机制
async def generate_with_cancel(request, cancel_event):
    for token in model.stream_generate(request):
        if cancel_event.is_set():
            break
        yield token
```

---

## 📝 总结

### 核心知识点

| 主题 | 核心要点 |
|------|----------|
| **推理优化** | KV Cache、vLLM、Continuous Batching |
| **量化技术** | PTQ/QAT、GPTQ/AWQ、INT4/INT8 |
| **推理框架** | vLLM、TensorRT-LLM、llama.cpp |
| **分布式推理** | TP/PP/SP、并行选择策略 |
| **模型压缩** | 量化、剪枝、蒸馏 |
| **生产部署** | 高可用架构、容量规划、监控告警 |

### 面试高频追问

1. **原理层面**：KV Cache 原理、量化公式
2. **对比选择**：vLLM vs TensorRT-LLM、TP vs PP
3. **实践问题**：冷启动优化、长尾延迟
4. **系统设计**：高可用架构、容量规划

---

*[返回面试指南目录](./index.md)*
