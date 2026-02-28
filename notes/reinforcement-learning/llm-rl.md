# 大模型强化学习

本章深入探讨大语言模型（LLM）中的强化学习技术，特别是从人类反馈中强化学习（RLHF）。这是当前 AI 领域最前沿的技术之一，广泛应用于 ChatGPT、Claude 等大模型的训练和对齐。

## RLHF 概述

### 核心动机

传统语言模型训练存在以下问题：
- **目标错位**：下一个 token 预测 ≠ 有用的回复
- **难以定义奖励**：什么是"好"的回答？
- **安全性问题**：模型可能生成有害内容

RLHF 通过**人类偏好数据**学习奖励函数，引导模型生成更符合人类期望的内容。

### 三阶段流程

```
基础模型 → SFT（监督微调） → RM（奖励模型） → PPO（策略优化） → 对齐模型
```

| 阶段 | 目标 | 数据 |
|------|------|------|
| 监督微调（SFT） | 学习指令遵循 | 高质量指令-回复对 |
| 奖励模型（RM） | 学习人类偏好 | 成对比较数据 |
| 策略优化（PPO） | 最大化奖励 + KL 约束 | 提示词 |

## 监督微调（SFT）

### 目标

使基础模型具备基本的指令遵循能力：

$$
\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log P(y|x; \theta) \right]
$$

### 实现

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def sft_training(model_name, dataset, output_dir):
    """监督微调训练"""
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def format_instruction(sample):
        """格式化指令数据"""
        if sample.get('input'):
            return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
        else:
            return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
    
    def tokenize(sample):
        text = format_instruction(sample)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    
    return model
```

## 奖励模型（RM）

### 目标

从人类偏好数据中学习一个奖励函数 $r_\phi(x, y)$，能够评估回复质量。

### 偏好数据格式

```python
preference_data = [
    {
        "prompt": "解释什么是机器学习",
        "chosen": "机器学习是人工智能的一个分支...",  # 被偏好的回复
        "rejected": "机器学习就是让机器学习..."      # 被拒绝的回复
    },
    # ... 更多数据
]
```

### Bradley-Terry 模型

假设人类选择 $y_w$（winner）优于 $y_l$（loser）的概率：

$$
P(y_w > y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
$$

**损失函数**：

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

### 实现

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """奖励模型"""
    def __init__(self, model_name, hidden_dim=768):
        super(RewardModel, self).__init__()
        
        self.base_model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        """计算奖励值"""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state[:, -1, :]
        
        reward = self.reward_head(last_hidden)
        return reward


def train_reward_model(model, dataloader, optimizer, epochs=1):
    """训练奖励模型"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            # 编码 chosen 和 rejected
            chosen_ids = batch['chosen_input_ids']
            chosen_mask = batch['chosen_attention_mask']
            rejected_ids = batch['rejected_input_ids']
            rejected_mask = batch['rejected_attention_mask']
            
            # 计算奖励
            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)
            
            # Bradley-Terry 损失
            loss = -nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    
    return model
```

## PPO 策略优化

### 目标函数

PPO 在大模型中的目标函数包含三项：

$$
\mathcal{L}(\theta) = \mathbb{E}\left[ \mathcal{L}_{CLIP} - c_1 \mathcal{L}_{VF} + c_2 \mathcal{L}_{ENT} \right]
$$

**KL 惩罚版本**：

$$
\mathcal{L}(\theta) = \mathbb{E}\left[ r(\theta) \hat{A} - c_{KL} \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]
$$

其中：
- $r(\theta) = \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$：策略比率
- $\hat{A}$：优势估计
- $D_{KL}$：KL 散度惩罚，防止模型偏离参考模型太远

### 核心组件

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

class RLHFTrainer:
    """RLHF 训练器"""
    def __init__(self, model_name, reward_model, kl_coef=0.1, lr=1e-6):
        
        # 策略模型（被优化）
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 参考模型（冻结，用于 KL 惩罚）
        self.reference = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.reference.parameters():
            param.requires_grad = False
        
        # 奖励模型（冻结）
        self.reward_model = reward_model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.kl_coef = kl_coef
        self.clip_epsilon = 0.2
    
    def generate(self, prompts, max_length=100):
        """生成回复"""
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True,
            truncation=True
        )
        
        outputs = self.policy.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        responses = self.tokenizer.batch_decode(
            outputs.sequences[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return responses, outputs
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        """计算序列的 log 概率"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # 去掉最后一个 token
        labels = input_ids[:, 1:]  # 去掉第一个 token
        
        log_probs = -nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        )
        
        return log_probs.reshape(labels.shape).sum(dim=-1)
    
    def compute_kl_divergence(self, prompts, responses):
        """计算 KL 散度"""
        texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.reference, inputs.input_ids, inputs.attention_mask
            )
        
        policy_log_probs = self.compute_log_probs(
            self.policy, inputs.input_ids, inputs.attention_mask
        )
        
        kl = policy_log_probs - ref_log_probs
        return kl
    
    def compute_rewards(self, prompts, responses):
        """计算奖励"""
        texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            rewards = self.reward_model(inputs.input_ids, inputs.attention_mask)
        
        return rewards.squeeze()
    
    def train_step(self, prompts):
        """单步训练"""
        # 生成回复
        responses, gen_outputs = self.generate(prompts)
        
        # 计算奖励
        rewards = self.compute_rewards(prompts, responses)
        
        # 计算 KL 惩罚
        kl_penalty = self.compute_kl_divergence(prompts, responses)
        
        # 总奖励 = 奖励模型分数 - KL 惩罚
        total_rewards = rewards - self.kl_coef * kl_penalty
        
        # 计算优势（简化版）
        advantages = total_rewards - total_rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # 计算 PPO 损失
        texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        policy_log_probs = self.compute_log_probs(
            self.policy, inputs.input_ids, inputs.attention_mask
        )
        
        # 使用旧策略的 log 概率（简化：用参考模型近似）
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(
                self.reference, inputs.input_ids, inputs.attention_mask
            )
        
        ratio = torch.exp(policy_log_probs - old_log_probs)
        
        # PPO 裁剪
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 优化
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'reward': rewards.mean().item(),
            'kl': kl_penalty.mean().item()
        }
```

## 完整 RLHF 流程

```python
def run_rlhf_pipeline():
    """完整 RLHF 流程"""
    
    # ===== 阶段 1：监督微调 =====
    print("Stage 1: Supervised Fine-Tuning")
    
    base_model = "gpt2"  # 或其他基础模型
    sft_model = sft_training(
        model_name=base_model,
        dataset=sft_dataset,
        output_dir="./sft_model"
    )
    
    # ===== 阶段 2：训练奖励模型 =====
    print("Stage 2: Reward Model Training")
    
    reward_model = RewardModel(sft_model.config.name_or_path)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
    
    reward_model = train_reward_model(
        model=reward_model,
        dataloader=preference_dataloader,
        optimizer=optimizer,
        epochs=1
    )
    
    # ===== 阶段 3：PPO 优化 =====
    print("Stage 3: PPO Optimization")
    
    trainer = RLHFTrainer(
        model_name="./sft_model",
        reward_model=reward_model,
        kl_coef=0.1,
        lr=1e-6
    )
    
    for epoch in range(10):
        for batch_prompts in prompt_dataloader:
            metrics = trainer.train_step(batch_prompts)
            
            if random.random() < 0.1:  # 打印部分训练信息
                print(f"Reward: {metrics['reward']:.3f}, KL: {metrics['kl']:.3f}")
    
    # 保存最终模型
    trainer.policy.save_pretrained("./rlhf_final_model")
    print("RLHF Training Complete!")
```

## 安全对齐

### 安全奖励

```python
class SafetyRewardModel:
    """安全奖励模型"""
    def __init__(self, safety_classifier):
        self.classifier = safety_classifier
    
    def compute_safety_reward(self, text):
        """计算安全奖励"""
        safety_score = self.classifier.predict(text)
        return safety_score
    
    def combined_reward(self, text, quality_reward, safety_weight=0.3):
        """组合质量和安全奖励"""
        safety_reward = self.compute_safety_reward(text)
        return (1 - safety_weight) * quality_reward + safety_weight * safety_reward
```

### Constitution AI

基于原则的对齐方法：

```python
PRINCIPLES = [
    "回复应该有帮助且无害",
    "回复应该诚实，不误导用户",
    "回复应该尊重用户隐私",
    # ... 更多原则
]

def principle_based_reward(response, principles):
    """基于原则的奖励"""
    rewards = []
    for principle in principles:
        # 使用 LLM 评估回复是否符合原则
        score = evaluate_against_principle(response, principle)
        rewards.append(score)
    return min(rewards)  # 取最低分确保所有原则都满足
```

## 知识点关联

```
基础语言模型
      ↓
监督微调（SFT）→ 指令遵循能力
      ↓
奖励模型训练（RM）→ 人类偏好学习
      ↓
PPO 优化 → 策略提升
      ↓
安全对齐 → 符合人类价值观
      ↓
最终 RLHF 模型
```

## 核心考点

### 必须掌握

| 技术 | 作用 |
|------|------|
| SFT | 赋予模型指令遵循能力 |
| Bradley-Terry 模型 | 偏好学习的理论基础 |
| KL 惩罚 | 防止策略偏离太远 |
| 安全对齐 | 确保输出符合人类价值观 |

### 关键公式

| 公式 | 说明 |
|------|------|
| $\mathcal{L}_{RM} = -\mathbb{E}[\log \sigma(r_w - r_l)]$ | 奖励模型损失 |
| $r = r_{model} - \beta \cdot D_{KL}$ | 带 KL 惩罚的奖励 |

## 学习建议

### 实践要点

1. **数据质量**：SFT 和 RM 数据质量至关重要
2. **KL 系数**：通常 0.02~0.1，过大会限制学习
3. **奖励模型过拟合**：监控 RM 在验证集的表现

### 后续方向

- **DPO**：直接偏好优化，无需奖励模型
- **Constitutional AI**：基于原则的对齐
- **RLAIF**：AI 反馈强化学习

---

**上一章**：[多智能体强化学习](./multi-agent-rl.md)  
**下一章**：[强化学习实战环境](./rl-environments.md)
