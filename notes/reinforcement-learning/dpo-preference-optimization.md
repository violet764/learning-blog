# 直接偏好优化（DPO）

直接偏好优化（Direct Preference Optimization, DPO）是一种无需训练奖励模型的 LLM 对齐方法，由 Rafailov 等人于 2023 年提出。相比传统的 RLHF（需要训练奖励模型 + PPO 优化），DPO 更简单、更稳定、更易于实现。

## DPO 核心思想

### RLHF 的问题

传统 RLHF 流程存在以下问题：

1. **流程复杂**：需要训练奖励模型，然后使用 PPO 优化策略
2. **训练不稳定**：PPO 需要精细调参，KL 惩罚系数敏感
3. **计算成本高**：需要同时加载策略模型、参考模型、奖励模型、价值模型
4. **奖励模型过拟合**：RM 可能无法准确反映人类偏好

### DPO 的洞见

DPO 的关键洞见是：**可以直接从偏好数据中学习最优策略，无需显式训练奖励模型**。

给定偏好数据 $(x, y_w, y_l)$，其中 $y_w$ 是被偏好的回复，$y_l$ 是被拒绝的回复。

** Bradley-Terry 模型**假设人类偏好的概率为：

$$
p(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))
$$

**最优策略与奖励函数的关系**：

$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

其中 $Z(x)$ 是配分函数（难以计算），但在 DPO 损失中会被消去。

### DPO 损失函数

将最优策略代入 Bradley-Terry 模型，得到 DPO 损失：

$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**直观理解**：
- 增加 $\pi_\theta(y_w|x)$ 同时降低 $\pi_\theta(y_l|x)$
- $\beta$ 控制偏离参考模型的程度
- 无需显式计算奖励函数

## DPO 实现

### 基础实现

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class DPOTrainer:
    """DPO 训练器"""
    def __init__(self, model_name, beta=0.1, lr=1e-6):
        self.beta = beta
        
        # 策略模型（被优化）
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 参考模型（冻结）
        self.reference = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.reference.parameters():
            param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def get_log_probs(self, model, input_ids, attention_mask):
        """计算序列的 log 概率"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # 去掉最后一个 token
        labels = input_ids[:, 1:]  # 去掉第一个 token
        
        # 计算 log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集每个位置的 log 概率
        per_token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 创建 mask，只计算非 padding 位置
        mask = attention_mask[:, 1:].float()
        
        # 返回序列总 log 概率
        return (per_token_log_probs * mask).sum(dim=-1)
    
    def compute_dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                         ref_chosen_logps, ref_rejected_logps):
        """计算 DPO 损失"""
        # 计算 log 比率
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO 损失
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def train_step(self, batch):
        """单步训练"""
        # 编码
        chosen_inputs = self.tokenizer(
            batch['chosen'], return_tensors='pt', padding=True, truncation=True
        )
        rejected_inputs = self.tokenizer(
            batch['rejected'], return_tensors='pt', padding=True, truncation=True
        )
        
        # 计算策略模型的 log 概率
        policy_chosen_logps = self.get_log_probs(
            self.policy, chosen_inputs['input_ids'], chosen_inputs['attention_mask']
        )
        policy_rejected_logps = self.get_log_probs(
            self.policy, rejected_inputs['input_ids'], rejected_inputs['attention_mask']
        )
        
        # 计算参考模型的 log 概率（无需梯度）
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.reference, chosen_inputs['input_ids'], chosen_inputs['attention_mask']
            )
            ref_rejected_logps = self.get_log_probs(
                self.reference, rejected_inputs['input_ids'], rejected_inputs['attention_mask']
            )
        
        # 计算 DPO 损失
        loss = self.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()


def prepare_preference_data(prompts, chosen_responses, rejected_responses):
    """准备偏好数据"""
    data = []
    for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
        data.append({
            'prompt': prompt,
            'chosen': prompt + " " + chosen,
            'rejected': prompt + " " + rejected
        })
    return data


# 使用示例
def train_dpo():
    """DPO 训练示例"""
    trainer = DPOTrainer(
        model_name="gpt2",
        beta=0.1,
        lr=1e-6
    )
    
    # 假设已有偏好数据
    preference_data = [
        {
            'chosen': "解释什么是机器学习：机器学习是人工智能的一个分支...",
            'rejected': "解释什么是机器学习：机器学习就是..."
        },
        # ... 更多数据
    ]
    
    for epoch in range(3):
        for batch in preference_data:
            loss = trainer.train_step(batch)
            print(f"Loss: {loss:.4f}")
    
    trainer.policy.save_pretrained("./dpo_model")
```

### 完整训练流程

```python
from datasets import Dataset
from torch.utils.data import DataLoader

class DPOTrainingPipeline:
    """DPO 完整训练流程"""
    def __init__(self, model_name, output_dir, beta=0.1, lr=5e-7, batch_size=4):
        self.output_dir = output_dir
        
        self.trainer = DPOTrainer(
            model_name=model_name,
            beta=beta,
            lr=lr
        )
        
        self.batch_size = batch_size
    
    def load_dataset(self, dataset_path):
        """加载偏好数据集"""
        # 支持多种格式：JSON, JSONL, HuggingFace Dataset
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=dataset_path)
        return dataset['train']
    
    def format_batch(self, batch):
        """格式化批次数据"""
        return {
            'chosen': [item for item in batch['chosen']],
            'rejected': [item for item in batch['rejected']]
        }
    
    def train(self, dataset, epochs=1, save_steps=500):
        """训练"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.format_batch
        )
        
        global_step = 0
        total_loss = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = self.trainer.train_step(batch)
                total_loss += loss
                global_step += 1
                
                if global_step % 10 == 0:
                    avg_loss = total_loss / 10
                    print(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    total_loss = 0
                
                if global_step % save_steps == 0:
                    self.trainer.policy.save_pretrained(
                        f"{self.output_dir}/checkpoint-{global_step}"
                    )
        
        # 保存最终模型
        self.trainer.policy.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")
```

## DPO 变体与扩展

### IPO（Identity Preference Optimization）

IPO 使用二次惩罚代替 sigmoid：

$$
\mathcal{L}_{IPO} = \mathbb{E}\left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]
$$

**优点**：对偏好数据噪声更鲁棒

### KTO（Kahneman-Tversky Optimization）

KTO 基于前景理论，不需要成对偏好数据：

$$
\mathcal{L}_{KTO} = \mathbb{E}_{x,y \sim \mathcal{D}} \left[ w(y) \cdot (1 - \sigma(\beta \cdot (z(x,y) - z_{ref}))) \right]
$$

**优点**：
- 不需要成对比较数据
- 可以利用更多现有数据

### ORPO（Odds Ratio Preference Optimization）

ORPO 将偏好学习整合到 SFT 中：

$$
\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}
$$

其中 $\mathcal{L}_{OR}$ 是基于赔率比的偏好损失。

### CPO（Contrastive Preference Optimization）

CPO 使用对比学习方法：

$$
\mathcal{L}_{CPO} = -\mathbb{E}\left[ \log \frac{\exp(s(x, y_w))}{\exp(s(x, y_w)) + \exp(s(x, y_l))} \right]
$$

## DPO vs RLHF 对比

| 方面 | RLHF (PPO) | DPO |
|------|------------|-----|
| 流程复杂度 | 高（SFT → RM → PPO） | 低（直接优化） |
| 计算资源 | 需要4个模型 | 需要2个模型 |
| 训练稳定性 | 敏感，需精细调参 | 稳定 |
| 数据要求 | 偏好数据 | 偏好数据 |
| 效果 | 成熟，广泛验证 | 竞争性强 |
| 适用场景 | 大规模生产 | 快速实验、资源受限 |

### 何时选择 DPO

- **数据量较小**（< 100K 偏好对）
- **计算资源有限**
- **需要快速迭代实验**
- **团队缺乏 RLHF 调参经验**

### 何时选择 RLHF

- **大规模生产系统**
- **数据量很大**（> 1M 偏好对）
- **需要精细控制 KL 散度**
- **已有成熟的 RLHF 流程**

## 实践建议

### 超参数调优

```python
# 推荐的超参数范围
dpo_config = {
    'beta': 0.1,          # KL 惩罚系数，通常 0.05 - 0.5
    'learning_rate': 5e-7, # 通常比 SFT 小 10x
    'batch_size': 64,      # 越大越稳定
    'epochs': 1,           # 通常 1-3 个 epoch
    'max_length': 512,     # 序列长度
    'warmup_ratio': 0.1,   # 预热步数
}
```

### 数据质量

```python
# 数据质量检查
def check_preference_quality(data):
    """检查偏好数据质量"""
    issues = []
    
    # 检查回复长度差异
    for item in data:
        chosen_len = len(item['chosen'].split())
        rejected_len = len(item['rejected'].split())
        
        if abs(chosen_len - rejected_len) > 100:
            issues.append(f"长度差异过大: chosen={chosen_len}, rejected={rejected_len}")
    
    # 检查是否有重复
    chosen_set = set(item['chosen'] for item in data)
    if len(chosen_set) < len(data) * 0.95:
        issues.append("存在大量重复数据")
    
    return issues
```

### 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 损失不下降 | 学习率过大或过小 | 调整学习率，使用 warmup |
| 生成质量下降 | beta 过大 | 降低 beta 或减少训练步数 |
| 过拟合偏好数据 | 数据量不足 | 增加数据，使用正则化 |
| 参考模型偏离 | SFT 不充分 | 加强 SFT 阶段训练 |

## 知识点关联

```
偏好数据 (Preference Data)
         ↓
    ┌────┴────┐
    ↓         ↓
  RLHF      DPO
    ↓         ↓
奖励模型   直接优化策略
    ↓         ↓
  PPO     更简单稳定
    ↓         ↓
    └────┬────┘
         ↓
     对齐模型
```

## 核心考点

### 必须掌握

| 概念 | 说明 |
|------|------|
| DPO 损失函数 | $\mathcal{L} = -\mathbb{E}[\log \sigma(\beta(\log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}))]$ |
| beta 参数 | 控制 KL 惩罚强度，通常 0.05-0.5 |
| 参考模型 | 冻结的 SFT 模型，作为优化锚点 |

### 面试高频问题

1. **DPO 相比 RLHF 的优势和劣势？**
   - 优势：简单、稳定、计算成本低
   - 劣势：无法使用强化学习探索，可能限制性能上限

2. **DPO 的 beta 参数如何选择？**
   - 过大：策略难以偏离参考模型
   - 过小：可能过拟合偏好数据

3. **DPO 是否可以完全替代 RLHF？**
   - 在多数场景下可以，但对于极高要求的场景，RLHF 可能仍有优势

## 参考资料

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [IPO: Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2310.12036)
- [KTO: Model Alignment as Prospect Theory Optimization](https://arxiv.org/abs/2402.01306)

---

**上一章**：[大模型强化学习（RLHF）](./llm-rl.md)  
**下一章**：[大模型对齐技术](./llm-alignment.md)
