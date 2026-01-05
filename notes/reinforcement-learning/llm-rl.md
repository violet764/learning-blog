# 大模型强化学习

## 章节概述
本章将深入探讨大语言模型（LLM）中的强化学习技术，特别是从人类反馈中强化学习（RLHF）。这是当前AI领域最前沿的技术之一，广泛应用于ChatGPT、Claude等大模型的训练和优化。我们将详细讲解RLHF的完整流程、关键技术实现和实际应用案例。

## 核心知识点

### 1. RLHF基本框架

#### 1.1 RLHF三阶段流程
RLHF包含三个核心阶段：

1. **监督微调（SFT）**：使用高质量对话数据微调基础模型
2. **奖励模型训练（RM）**：学习人类偏好，构建奖励函数
3. **策略优化（PPO）**：使用奖励模型指导策略优化

#### 1.2 RLHF整体架构
```python
class RLHFPipeline:
    """RLHF完整流程"""
    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # 三个阶段模型
        self.sft_model = None
        self.reward_model = None
        self.rl_model = None
        
    def supervised_finetuning(self, sft_data):
        """阶段1：监督微调"""
        # 使用高质量对话数据微调
        pass
        
    def train_reward_model(self, preference_data):
        """阶段2：训练奖励模型"""
        # 学习人类偏好
        pass
        
    def policy_optimization(self, ppo_config):
        """阶段3：策略优化"""
        # 使用PPO优化策略
        pass
```

### 2. 奖励模型训练

#### 2.1 人类偏好数据
奖励模型训练需要成对的偏好数据：
- **提示（Prompt）**：输入文本
- **回应A vs 回应B**：两个模型生成的回应
- **人类评分**：哪个回应更好

#### 2.2 奖励模型架构
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """奖励模型：评估文本质量"""
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
        super(RewardModel, self).__init__()
        
        # 基础语言模型
        self.lm = AutoModel.from_pretrained(model_name)
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 值头（用于基线）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # 获取最后一层隐藏状态
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # 使用[CLS] token或平均池化
        if hasattr(self.lm.config, 'pooler_type') and self.lm.config.pooler_type == 'cls':
            pooled_output = last_hidden_state[:, 0]  # [CLS] token
        else:
            # 平均池化
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
        
        # 计算奖励和值
        reward = self.reward_head(pooled_output)
        value = self.value_head(pooled_output)
        
        return reward, value
```

#### 2.3 偏好损失函数
使用Bradley-Terry模型：
```python
class PreferenceLoss(nn.Module):
    """偏好损失函数"""
    def __init__(self):
        super(PreferenceLoss, self).__init__()
    
    def forward(self, rewards_chosen, rewards_rejected):
        """
        rewards_chosen: 被选中的回应奖励
        rewards_rejected: 被拒绝的回应奖励
        """
        # 计算对数几率
        logits = rewards_chosen - rewards_rejected
        
        # 使用sigmoid交叉熵损失
        # 目标：chosen > rejected，所以标签为1
        labels = torch.ones_like(logits)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        return loss

# 奖励模型训练循环
def train_reward_model_epoch(model, dataloader, optimizer):
    """训练奖励模型的一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 解包批次数据
        chosen_input_ids = batch['chosen_input_ids']
        chosen_attention_mask = batch['chosen_attention_mask']
        rejected_input_ids = batch['rejected_input_ids']
        rejected_attention_mask = batch['rejected_attention_mask']
        
        # 前向传播
        rewards_chosen, _ = model(chosen_input_ids, chosen_attention_mask)
        rewards_rejected, _ = model(rejected_input_ids, rejected_attention_mask)
        
        # 计算损失
        loss_fn = PreferenceLoss()
        loss = loss_fn(rewards_chosen, rewards_rejected)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 3. PPO在大模型中的应用

#### 3.1 大模型PPO的特殊挑战
- **计算成本**：大模型前向传播昂贵
- **序列生成**：需要处理可变长度序列
- **KL散度惩罚**：防止策略偏离SFT模型太远

#### 3.2 大模型PPO实现
```python
class LLMPPO:
    """大语言模型的PPO实现"""
    def __init__(self, policy_model, reward_model, ref_model, 
                 ppo_config, generation_config):
        
        self.policy_model = policy_model  # 被优化的策略
        self.reward_model = reward_model  # 奖励模型
        self.ref_model = ref_model        # 参考模型（SFT阶段模型）
        
        self.ppo_config = ppo_config
        self.generation_config = generation_config
        
        # 优化器
        self.optimizer = optim.AdamW(
            policy_model.parameters(), 
            lr=ppo_config['learning_rate']
        )
        
        # 经验缓冲区
        self.buffer = []
    
    def generate_responses(self, prompts):
        """生成回应"""
        responses = []
        log_probs = []
        
        for prompt in prompts:
            # 使用策略模型生成
            input_ids = self.policy_model.tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    input_ids,
                    max_length=self.generation_config['max_length'],
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 提取生成文本和log概率
            response_ids = outputs.sequences[0, len(input_ids[0]):]
            response = self.policy_model.tokenizer.decode(response_ids)
            
            # 计算log概率（简化实现）
            scores = outputs.scores
            log_prob = self._compute_log_prob(response_ids, scores)
            
            responses.append(response)
            log_probs.append(log_prob)
        
        return responses, log_probs
    
    def compute_rewards(self, prompts, responses):
        """计算奖励"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # 组合提示和回应
            text = prompt + " " + response
            
            # 编码
            inputs = self.reward_model.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            # 计算奖励
            with torch.no_grad():
                reward, _ = self.reward_model(**inputs)
            
            rewards.append(reward.item())
        
        return rewards
    
    def compute_kl_penalty(self, prompts, responses, log_probs):
        """计算KL散度惩罚"""
        kl_penalties = []
        
        for prompt, response, log_prob in zip(prompts, responses, log_probs):
            text = prompt + " " + response
            
            # 参考模型的log概率
            inputs = self.ref_model.tokenizer(text, return_tensors='pt')
            
            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs, labels=inputs['input_ids'])
                ref_log_prob = -ref_outputs.loss * inputs['input_ids'].size(1)  # 近似
            
            # KL散度 = 当前策略log_prob - 参考策略log_prob
            kl_divergence = log_prob - ref_log_prob
            kl_penalty = -self.ppo_config['kl_coef'] * kl_divergence
            
            kl_penalties.append(kl_penalty.item())
        
        return kl_penalties
    
    def ppo_update(self):
        """PPO更新步骤"""
        if len(self.buffer) < self.ppo_config['batch_size']:
            return
        
        # 采样批次
        batch = random.sample(self.buffer, self.ppo_config['batch_size'])
        
        prompts = [item['prompt'] for item in batch]
        responses = [item['response'] for item in batch]
        old_log_probs = torch.tensor([item['log_prob'] for item in batch])
        rewards = torch.tensor([item['reward'] for item in batch])
        kl_penalties = torch.tensor([item['kl_penalty'] for item in batch])
        
        # 总奖励 = 环境奖励 + KL惩罚
        total_rewards = rewards + kl_penalties
        
        # 标准化优势
        advantages = total_rewards - total_rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # 重新计算当前策略的log概率
        current_log_probs = []
        for prompt, response in zip(prompts, responses):
            # 简化实现，实际需要更精确的计算
            text = prompt + " " + response
            inputs = self.policy_model.tokenizer(text, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.policy_model(**inputs, labels=inputs['input_ids'])
                log_prob = -outputs.loss * inputs['input_ids'].size(1)
            
            current_log_probs.append(log_prob.item())
        
        current_log_probs = torch.tensor(current_log_probs)
        
        # 策略比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_config['clip_epsilon'], 
                           1 + self.ppo_config['clip_epsilon']) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 优化
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return policy_loss.item()
```

### 4. 指令调优与对齐

#### 4.1 指令调优技术
```python
class InstructionTuning:
    """指令调优"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def format_instruction(self, instruction, input_text=None):
        """格式化指令"""
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    def train_on_instructions(self, instruction_dataset, epochs=3):
        """在指令数据上训练"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in instruction_dataset:
                instructions = batch['instruction']
                inputs = batch.get('input', [None] * len(instructions))
                responses = batch['response']
                
                # 格式化文本
                formatted_texts = []
                labels = []
                
                for instruction, input_text, response in zip(instructions, inputs, responses):
                    prompt = self.format_instruction(instruction, input_text)
                    full_text = prompt + response
                    
                    formatted_texts.append(full_text)
                    
                    # 创建标签（只对回应部分计算损失）
                    prompt_len = len(self.tokenizer.encode(prompt))
                    response_len = len(self.tokenizer.encode(response))
                    
                    # 标签：prompt部分为-100，response部分为实际token
                    label = [-100] * prompt_len + self.tokenizer.encode(response)
                    labels.append(label)
                
                # 编码和训练
                inputs = self.tokenizer(formatted_texts, padding=True, truncation=True, 
                                       return_tensors='pt', max_length=512)
                labels_tensor = torch.tensor(labels)
                
                outputs = self.model(**inputs, labels=labels_tensor)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(instruction_dataset):.4f}")
```

#### 4.2 安全对齐技术
```python
class SafetyAlignment:
    """安全对齐"""
    def __init__(self, model, safety_classifier):
        self.model = model
        self.safety_classifier = safety_classifier
    
    def safety_reward(self, text):
        """安全奖励"""
        # 使用安全分类器评估文本安全性
        safety_score = self.safety_classifier.predict(text)
        return safety_score
    
    def combined_reward(self, text, quality_reward, safety_weight=0.3):
        """组合奖励：质量 + 安全性"""
        safety_reward = self.safety_reward(text)
        
        # 加权组合
        total_reward = (1 - safety_weight) * quality_reward + safety_weight * safety_reward
        return total_reward
```

### 5. 实战案例：聊天机器人RLHF

#### 5.1 完整RLHF流程实现
```python
def run_rlhf_pipeline():
    """完整RLHF流程"""
    
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 2. 监督微调（SFT）
    print("阶段1: 监督微调")
    sft_trainer = InstructionTuning(base_model, tokenizer)
    sft_trainer.train_on_instructions(sft_dataset)
    
    # 保存SFT模型
    sft_model = base_model
    sft_model.save_pretrained("./sft_model")
    
    # 3. 训练奖励模型
    print("阶段2: 训练奖励模型")
    reward_model = RewardModel()
    
    for epoch in range(reward_training_epochs):
        loss = train_reward_model_epoch(reward_model, preference_dataloader, reward_optimizer)
        print(f"Reward Model Epoch {epoch}, Loss: {loss:.4f}")
    
    # 4. PPO优化
    print("阶段3: PPO优化")
    
    # 加载参考模型（SFT模型）
    ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")
    
    # 创建PPO训练器
    ppo_config = {
        'learning_rate': 1e-6,
        'batch_size': 4,
        'kl_coef': 0.1,
        'clip_epsilon': 0.2
    }
    
    generation_config = {
        'max_length': 100,
        'temperature': 0.7
    }
    
    ppo_trainer = LLMPPO(base_model, reward_model, ref_model, ppo_config, generation_config)
    
    # PPO训练循环
    for ppo_epoch in range(ppo_epochs):
        # 生成回应
        prompts = sample_prompts(ppo_batch_size)
        responses, log_probs = ppo_trainer.generate_responses(prompts)
        
        # 计算奖励
        rewards = ppo_trainer.compute_rewards(prompts, responses)
        
        # 计算KL惩罚
        kl_penalties = ppo_trainer.compute_kl_penalty(prompts, responses, log_probs)
        
        # 存储经验
        for i in range(len(prompts)):
            ppo_trainer.buffer.append({
                'prompt': prompts[i],
                'response': responses[i],
                'log_prob': log_probs[i],
                'reward': rewards[i],
                'kl_penalty': kl_penalties[i]
            })
        
        # 更新策略
        if len(ppo_trainer.buffer) >= ppo_config['batch_size']:
            loss = ppo_trainer.ppo_update()
            print(f"PPO Epoch {ppo_epoch}, Loss: {loss:.4f}")
    
    # 保存最终模型
    base_model.save_pretrained("./rlhf_final_model")
    print("RLHF训练完成!")
```

#### 5.2 评估与测试
```python
def evaluate_rlhf_model(model, tokenizer, test_prompts):
    """评估RLHF模型"""
    results = []
    
    for prompt in test_prompts:
        # 生成回应
        inputs = tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除提示部分，只保留回应
        response = response[len(prompt):].strip()
        
        results.append({
            'prompt': prompt,
            'response': response
        })
    
    return results

# 测试示例
test_prompts = [
    "请用简单的语言解释量子力学",
    "写一个关于友谊的短故事",
    "如何学习编程？给出具体建议"
]

final_model = AutoModelForCausalLM.from_pretrained("./rlhf_final_model")
results = evaluate_rlhf_model(final_model, tokenizer, test_prompts)

for result in results:
    print(f"提示: {result['prompt']}")
    print(f"回应: {result['response']}")
    print("-" * 50)
```

## 知识点间关联逻辑

### RLHF技术栈
```
基础语言模型 (GPT, LLaMA等)
    ↓
监督微调 (SFT) → 指令遵循能力
    ↓
奖励模型训练 (RM) → 人类偏好学习
    ↓
PPO优化 → 策略提升 + 安全对齐
    ↓
最终RLHF模型
```

## 章节核心考点汇总

### 必须掌握的技术
1. **RLHF三阶段**：SFT → RM → PPO
2. **奖励模型训练**：偏好数据、Bradley-Terry模型
3. **大模型PPO**：KL散度惩罚、序列生成处理

### 重要概念
- **人类偏好学习**：从比较数据中学习质量评估
- **策略约束**：防止过度优化导致的模型退化
- **安全对齐**：确保模型输出符合人类价值观

## 学习建议/后续延伸方向

### 学习建议
1. **理解完整流程**：掌握RLHF的三个阶段及其作用
2. **动手实现**：尝试简化版的RLHF流程
3. **关注最新进展**：RLHF技术仍在快速发展

### 后续学习方向
1. ** Constitutional AI**：基于原则的对齐方法
2. **模型评估**：如何客观评估大模型质量
3. **多模态RLHF**：图像、视频等模态的扩展

---

**下一章预告**：在掌握了大模型强化学习后，下一章将介绍强化学习实战环境，包括OpenAI Gym、自定义环境开发等实用技术。