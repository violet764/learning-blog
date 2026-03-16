# 增量预训练技术详解

## 章节概述

增量预训练（Continued Pre-training），又称领域自适应预训练（Domain-Adaptive Pre-training, DAPT），是指在已有预训练模型基础上，使用特定领域数据继续进行预训练的技术。本章深入解析增量预训练的核心原理、流程设计和实践策略，帮助读者掌握如何将通用大模型适配到特定专业领域。

## 基本概念

### 什么是增量预训练

增量预训练是指在已经完成大规模通用预训练的模型基础上，继续使用特定领域数据进行预训练的过程。与从零开始的全量预训练不同，增量预训练利用了模型已有的通用知识，仅需相对较少的数据和计算资源就能获得领域适应能力。

**核心思想：**
$$
\theta_{\text{domain}} = \theta_{\text{general}} + \Delta\theta
$$

其中 $\theta_{\text{general}}$ 是通用预训练模型的参数，$\Delta\theta$ 是通过领域数据学习到的参数增量。

**数学形式化：**
给定通用预训练模型 $P(x; \theta_{\text{general}})$ 和领域数据分布 $\mathcal{D}_{\text{domain}}$，增量预训练的目标是：

$$
\mathcal{L}_{\text{CPT}} = -\mathbb{E}_{x \sim \mathcal{D}_{\text{domain}}} \left[ \sum_{t=1}^{n} \log P(x_t | x_{<t}; \theta) \right]
$$

### 为什么需要增量预训练

📌 **通用模型的局限性：**

| 问题类型 | 具体表现 | 示例 |
|---------|---------|------|
| 领域知识缺失 | 专业术语、概念理解不准确 | 医疗诊断术语、法律条文 |
| 特殊表达习惯 | 行业特有的表达方式和逻辑 | 合同语言、学术写作 |
| 最新信息缺失 | 预训练截止日期后的新知识 | 新法规、新技术 |
| 任务特定能力 | 特定任务的最佳实践 | 代码生成、数据分析 |

💡 **增量预训练的优势：**

1. **知识注入**：将专业领域知识注入通用模型
2. **语言适应**：学习领域特定的语言表达和逻辑
3. **成本效益**：相比全量预训练，数据和计算成本大幅降低
4. **快速迭代**：可随领域数据更新持续优化模型

### 增量预训练 vs 全量预训练

```python
class PretrainingComparison:
    """预训练方式对比分析"""
    
    @staticmethod
    def compare_pretraining_approaches():
        """对比增量预训练与全量预训练"""
        
        comparison = {
            '全量预训练': {
                '数据规模': 'TB级通用数据（数万亿tokens）',
                '训练时间': '数周到数月',
                '计算资源': '数千GPU/TPU',
                '适用场景': '构建通用基础模型',
                '知识范围': '广泛但浅层',
                '成本': '极高（百万美元级）'
            },
            '增量预训练': {
                '数据规模': 'GB到TB级领域数据（数十亿tokens）',
                '训练时间': '数小时到数天',
                '计算资源': '数十到数百GPU',
                '适用场景': '领域适配、知识更新',
                '知识范围': '专精且深入',
                '成本': '中等（万美元到数十万美元）'
            }
        }
        
        return comparison
    
    @staticmethod
    def estimate_training_cost(model_size, data_size, method='continued'):
        """估算训练成本"""
        
        # 基础计算量（每token的FLOPs）
        flops_per_token = 6 * model_size  # 简化估算
        
        if method == 'full':
            # 全量预训练需要更多epoch
            total_tokens = data_size * 1  # 通常1个epoch
        else:
            # 增量预训练可能需要多个epoch
            total_tokens = data_size * 2  # 约2个epoch
        
        total_flops = flops_per_token * total_tokens
        
        # 假设A100 GPU的利用率为40%
        gpu_flops_per_second = 312e12 * 0.4
        gpu_hours = total_flops / gpu_flops_per_second / 3600
        
        return {
            'total_flops': total_flops,
            'gpu_hours': gpu_hours,
            'estimated_cost_usd': gpu_hours * 2.0  # 约$2/GPU小时
        }

# 示例：7B模型增量预训练成本估算
comparison = PretrainingComparison()
cost = comparison.estimate_training_cost(
    model_size=7e9,  # 7B参数
    data_size=10e9,  # 100亿tokens
    method='continued'
)
print(f"增量预训练估算: {cost['gpu_hours']:.0f} GPU小时, 约${cost['estimated_cost_usd']:.0f}")
# 增量预训练估算: 1870 GPU小时, 约$3739
```


## 领域自适应预训练

### 领域数据选择

数据质量是增量预训练成功的关键因素。需要从多个维度评估和筛选领域数据。

**数据来源分类：**

| 数据来源 | 特点 | 质量评估重点 |
|---------|------|------------|
| 学术文献 | 权威性高、专业性强 | 引用次数、期刊等级 |
| 行业文档 | 实践性强、时效性好 | 来源可靠性、更新频率 |
| 专业论坛 | 讨论深入、覆盖面广 | 用户信誉、回答质量 |
| 政策法规 | 权威性强、准确度高 | 官方来源、时效性 |
| 内部文档 | 定制化程度高 | 数据清洗、脱敏处理 |

```python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class DataQualityMetrics:
    """数据质量评估指标"""
    text_quality: float      # 文本质量分数
    domain_relevance: float  # 领域相关度
    information_density: float  # 信息密度
    duplication_score: float  # 重复度（越低越好）
    
    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        return (
            0.3 * self.text_quality +
            0.3 * self.domain_relevance +
            0.25 * self.information_density +
            0.15 * (1 - self.duplication_score)
        )


class DomainDataSelector:
    """领域数据选择器"""
    
    def __init__(self, domain_keywords: List[str], min_quality_score: float = 0.6):
        self.domain_keywords = set(domain_keywords)
        self.min_quality_score = min_quality_score
        self.seen_texts = set()  # 用于去重
    
    def evaluate_text_quality(self, text: str) -> float:
        """评估文本质量"""
        score = 1.0
        
        # 长度检查
        if len(text) < 50:
            score *= 0.5
        elif len(text) > 10000:
            score *= 0.9  # 过长的文本可能包含噪声
        
        # 语言规范性检查
        # 检查乱码比例
        garbage_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff,.!?;:\-()]', text)) / len(text)
        score *= max(0, 1 - garbage_ratio * 5)
        
        # 检查句子完整性
        sentences = re.split(r'[.!?。！？]', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
        sentence_ratio = complete_sentences / max(len(sentences), 1)
        score *= sentence_ratio
        
        return min(1.0, max(0.0, score))
    
    def evaluate_domain_relevance(self, text: str) -> float:
        """评估领域相关度"""
        text_lower = text.lower()
        
        # 统计领域关键词出现次数
        keyword_count = sum(1 for kw in self.domain_keywords if kw in text_lower)
        
        # 归一化：考虑文本长度
        text_length = len(text)
        density = keyword_count / (text_length / 1000)  # 每千字关键词数
        
        # 使用sigmoid函数归一化
        relevance = 1 / (1 + np.exp(-2 * (density - 2)))
        
        return relevance
    
    def evaluate_information_density(self, text: str) -> float:
        """评估信息密度"""
        # 简化的信息密度评估
        # 1. 词汇多样性
        words = list(text)
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # 2. 实词比例（简化：非停用词比例）
        stop_chars = set('的是不在有了和与或但如被把这那')
        content_ratio = 1 - sum(1 for c in text if c in stop_chars) / len(text)
        
        # 3. 标点密度（反映句子结构）
        punctuation_count = len(re.findall(r'[,.!?;:，。！？；：]', text))
        punct_density = min(1.0, punctuation_count / (len(text) / 50))
        
        return 0.4 * diversity + 0.4 * content_ratio + 0.2 * punct_density
    
    def check_duplication(self, text: str) -> float:
        """检查重复度"""
        # 使用文本哈希进行快速去重
        text_hash = hash(text[:100])  # 使用前100字符作为指纹
        
        if text_hash in self.seen_texts:
            return 1.0  # 完全重复
        
        # 检查n-gram重复
        ngrams = set()
        duplicate_count = 0
        n = 5
        
        for i in range(len(text) - n):
            ngram = text[i:i+n]
            if ngram in ngrams:
                duplicate_count += 1
            ngrams.add(ngram)
        
        duplication_ratio = duplicate_count / max(len(text) - n, 1)
        
        self.seen_texts.add(text_hash)
        return duplication_ratio
    
    def select_data(self, texts: List[str]) -> Tuple[List[str], List[DataQualityMetrics]]:
        """筛选高质量领域数据"""
        selected_texts = []
        metrics_list = []
        
        for text in texts:
            metrics = DataQualityMetrics(
                text_quality=self.evaluate_text_quality(text),
                domain_relevance=self.evaluate_domain_relevance(text),
                information_density=self.evaluate_information_density(text),
                duplication_score=self.check_duplication(text)
            )
            
            if metrics.overall_score >= self.min_quality_score:
                selected_texts.append(text)
                metrics_list.append(metrics)
        
        return selected_texts, metrics_list


# 医疗领域数据选择示例
medical_keywords = [
    '诊断', '治疗', '症状', '患者', '临床', '病理', '药物',
    '手术', '医学', '疾病', '预后', '检查', '影像', '实验室'
]

selector = DomainDataSelector(medical_keywords, min_quality_score=0.6)
```

### 数据配比策略

增量预训练中，如何混合通用数据和领域数据是关键决策。

**核心原则：**
- 纯领域数据可能导致灾难性遗忘
- 适当混入通用数据保持模型通用能力
- 领域数据占比通常在 50%-90% 之间

```python
class DataMixingStrategy:
    """数据配比策略"""
    
    def __init__(self, domain_ratio: float = 0.7, 
                 domain_data: List[str] = None,
                 general_data: List[str] = None):
        """
        Args:
            domain_ratio: 领域数据占比
            domain_data: 领域数据列表
            general_data: 通用数据列表
        """
        self.domain_ratio = domain_ratio
        self.domain_data = domain_data or []
        self.general_data = general_data or []
    
    def compute_optimal_ratio(self, 
                               domain_size: int,
                               general_size: int,
                               target_tokens: int) -> float:
        """
        计算最优领域数据占比
        
        基于经验法则：
        - 领域数据量充足时，可以更高比例使用领域数据
        - 领域数据量不足时，需要更多通用数据防止过拟合
        """
        # 计算领域数据能提供的epoch数
        domain_epochs = target_tokens / domain_size if domain_size > 0 else 0
        
        # 根据epoch数调整比例
        # epoch数越多，说明领域数据充足，可以使用更高比例
        if domain_epochs >= 3:
            optimal_ratio = 0.85
        elif domain_epochs >= 2:
            optimal_ratio = 0.75
        elif domain_epochs >= 1:
            optimal_ratio = 0.65
        else:
            optimal_ratio = 0.5
        
        return optimal_ratio
    
    def create_mixed_dataset(self, 
                             total_samples: int,
                             curriculum_learning: bool = True) -> List[Dict]:
        """
        创建混合数据集
        
        Args:
            total_samples: 总样本数
            curriculum_learning: 是否使用课程学习策略
        """
        domain_samples = int(total_samples * self.domain_ratio)
        general_samples = total_samples - domain_samples
        
        # 随机采样
        import random
        
        selected_domain = random.sample(
            self.domain_data, 
            min(domain_samples, len(self.domain_data))
        )
        selected_general = random.sample(
            self.general_data, 
            min(general_samples, len(self.general_data))
        )
        
        # 构建数据集
        dataset = []
        
        if curriculum_learning:
            # 课程学习：先通用后领域，逐步增加领域数据难度
            # 阶段1：通用数据为主（热身）
            warmup_ratio = 0.1
            warmup_samples = int(total_samples * warmup_ratio)
            
            for i in range(warmup_samples):
                if i < len(selected_general):
                    dataset.append({
                        'text': selected_general[i],
                        'type': 'general',
                        'phase': 'warmup'
                    })
            
            # 阶段2：混合训练
            remaining_domain = selected_domain
            remaining_general = selected_general[warmup_samples:]
            
            mixed_samples = total_samples - warmup_samples
            domain_in_mixed = int(mixed_samples * self.domain_ratio)
            
            for i in range(mixed_samples):
                if i < domain_in_mixed and remaining_domain:
                    dataset.append({
                        'text': remaining_domain.pop(0) if remaining_domain else '',
                        'type': 'domain',
                        'phase': 'mixed'
                    })
                elif remaining_general:
                    dataset.append({
                        'text': remaining_general.pop(0) if remaining_general else '',
                        'type': 'general',
                        'phase': 'mixed'
                    })
        else:
            # 随机混合
            for text in selected_domain:
                dataset.append({'text': text, 'type': 'domain'})
            for text in selected_general:
                dataset.append({'text': text, 'type': 'general'})
            
            random.shuffle(dataset)
        
        return dataset


class AdaptiveDataScheduler:
    """自适应数据调度器"""
    
    def __init__(self, 
                 initial_domain_ratio: float = 0.5,
                 final_domain_ratio: float = 0.85,
                 total_steps: int = 10000):
        """
        Args:
            initial_domain_ratio: 初始领域数据比例
            final_domain_ratio: 最终领域数据比例
            total_steps: 总训练步数
        """
        self.initial_ratio = initial_domain_ratio
        self.final_ratio = final_domain_ratio
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_current_ratio(self) -> float:
        """获取当前领域数据比例（线性增加）"""
        progress = min(self.current_step / self.total_steps, 1.0)
        ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
        return ratio
    
    def step(self):
        """更新步数"""
        self.current_step += 1
```

### 学习率设置

增量预训练的学习率策略与全量预训练有显著不同。

**关键原则：**
1. **学习率应低于全量预训练**：避免破坏已学习的知识
2. **使用更长的warmup**：让模型平稳适应新数据
3. **考虑分层学习率**：底层保持较低学习率，顶层可适当提高

```python
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

class ContinuedPretrainingScheduler:
    """增量预训练学习率调度器"""
    
    def __init__(self,
                 optimizer,
                 max_lr: float = 5e-5,        # 峰值学习率（低于全量预训练）
                 min_lr: float = 1e-6,        # 最小学习率
                 warmup_steps: int = 500,     # warmup步数
                 total_steps: int = 10000,    # 总训练步数
                 warmup_ratio: float = 0.05): # warmup比例
        
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        if warmup_ratio > 0:
            self.warmup_steps = int(total_steps * warmup_ratio)
        
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self) -> LambdaLR:
        """创建学习率调度器"""
        
        def lr_lambda(current_step: int) -> float:
            # Warmup阶段
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # 衰减阶段：余弦衰减到min_lr
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # 确保不低于min_lr
            lr_ratio = (self.max_lr - self.min_lr) * cosine_decay / self.max_lr
            min_ratio = self.min_lr / self.max_lr
            
            return max(min_ratio, lr_ratio)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self):
        """更新学习率"""
        self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class LayerWiseLearningRate:
    """分层学习率策略"""
    
    def __init__(self, model, base_lr: float = 5e-5, layer_decay: float = 0.75):
        """
        Args:
            model: 预训练模型
            base_lr: 基础学习率（应用于顶层）
            layer_decay: 层衰减系数
                - 底层: base_lr * layer_decay^(num_layers)
                - 顶层: base_lr
        """
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.layer_indices = self._get_layer_indices()
    
    def _get_layer_indices(self) -> Dict[str, int]:
        """获取每层的索引"""
        layer_indices = {}
        
        for name, _ in self.model.named_parameters():
            # 解析层号（假设模型结构类似 Transformer）
            if 'layers.' in name:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                layer_indices[name] = layer_num
            elif 'embed' in name:
                layer_indices[name] = -1  # 嵌入层
            else:
                layer_indices[name] = 0   # 默认
        
        return layer_indices
    
    def get_parameter_groups(self) -> List[Dict]:
        """获取分层学习率的参数组"""
        num_layers = max(self.layer_indices.values()) + 1
        parameter_groups = {}
        
        for name, param in self.model.named_parameters():
            layer_idx = self.layer_indices.get(name, 0)
            
            # 计算该层的学习率
            if layer_idx == -1:
                # 嵌入层使用最低学习率
                lr = self.base_lr * (self.layer_decay ** (num_layers + 1))
            else:
                # 从顶层到底层逐渐降低
                lr = self.base_lr * (self.layer_decay ** (num_layers - layer_idx))
            
            if lr not in parameter_groups:
                parameter_groups[lr] = {'params': [], 'lr': lr}
            
            parameter_groups[lr]['params'].append(param)
        
        return list(parameter_groups.values())


def create_optimizer_with_layer_lr(model, base_lr=5e-5, weight_decay=0.01):
    """创建带分层学习率的优化器"""
    layer_wise_lr = LayerWiseLearningRate(model, base_lr)
    param_groups = layer_wise_lr.get_parameter_groups()
    
    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )
    
    return optimizer
```

### 训练步数确定

确定合适的训练步数是增量预训练的关键决策。

```python
class TrainingStepsCalculator:
    """训练步数计算器"""
    
    def __init__(self, 
                 model_size: int,           # 模型参数量
                 data_size: int,            # 数据token数
                 batch_size: int = 256,     # 批次大小
                 seq_length: int = 2048):   # 序列长度
        
        self.model_size = model_size
        self.data_size = data_size
        self.batch_size = batch_size
        self.seq_length = seq_length
    
    def compute_tokens_per_step(self) -> int:
        """计算每步处理的token数"""
        return self.batch_size * self.seq_length
    
    def compute_steps_per_epoch(self) -> int:
        """计算每个epoch的步数"""
        tokens_per_step = self.compute_tokens_per_step()
        return self.data_size // tokens_per_step
    
    def estimate_optimal_steps(self) -> Dict[str, int]:
        """
        估算最优训练步数
        
        经验法则：
        - 增量预训练通常需要1-3个epoch
        - 数据量小时需要更多epoch但不建议超过5个
        - 数据量大时1-2个epoch即可
        """
        steps_per_epoch = self.compute_steps_per_epoch()
        tokens_per_step = self.compute_tokens_per_step()
        
        # 计算Chinchilla最优token数（简化版）
        chinchilla_tokens = 20 * self.model_size  # 简化估计
        
        # 计算实际可用的epoch数
        if self.data_size < chinchilla_tokens * 0.1:
            # 数据量严重不足，需要更多epoch
            recommended_epochs = min(5, chinchilla_tokens // self.data_size)
        elif self.data_size < chinchilla_tokens * 0.5:
            # 数据量适中
            recommended_epochs = 3
        else:
            # 数据量充足
            recommended_epochs = 1
        
        optimal_steps = steps_per_epoch * recommended_epochs
        
        return {
            'steps_per_epoch': steps_per_epoch,
            'recommended_epochs': recommended_epochs,
            'optimal_steps': optimal_steps,
            'total_tokens_trained': optimal_steps * tokens_per_step,
            'data_reuse_factor': optimal_steps * tokens_per_step / self.data_size
        }
    
    def compute_convergence_estimate(self, 
                                      validation_loss_history: List[float]) -> Dict:
        """
        基于验证损失估计收敛
        
        当验证损失连续N步没有显著下降时，可以考虑停止训练
        """
        if len(validation_loss_history) < 10:
            return {'status': 'insufficient_data'}
        
        # 计算最近10步的平均改进
        recent_losses = validation_loss_history[-10:]
        improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        # 判断是否收敛
        if improvement < 0.001:  # 改进小于0.1%
            return {
                'status': 'converged',
                'recent_improvement': improvement,
                'recommendation': '可以考虑停止训练'
            }
        elif improvement < 0.01:
            return {
                'status': 'nearly_converged',
                'recent_improvement': improvement,
                'recommendation': '接近收敛，可以开始评估'
            }
        else:
            return {
                'status': 'training',
                'recent_improvement': improvement,
                'recommendation': '继续训练'
            }


# 示例使用
calculator = TrainingStepsCalculator(
    model_size=7_000_000_000,    # 7B模型
    data_size=10_000_000_000,    # 100亿tokens
    batch_size=256,
    seq_length=2048
)

steps_info = calculator.estimate_optimal_steps()
```

## 增量预训练流程

### 数据准备

完整的数据准备流程包括数据收集、清洗、去重、质量过滤等环节。

```python
import json
import hashlib
from typing import Iterator, Optional
from pathlib import Path

class DataPreparationPipeline:
    """数据准备流水线"""
    
    def __init__(self, 
                 output_dir: str,
                 tokenizer_name: str = "gpt2",
                 max_length: int = 2048):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_length = max_length
        self.min_length = 50  # 最小序列长度
        
        # 去重相关
        self.hash_set = set()
        self.ngram_set = set()
    
    def clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除不可见字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 移除URL（可选，取决于任务需求）
        # text = re.sub(r'https?://\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        return text.strip()
    
    def compute_text_hash(self, text: str) -> str:
        """计算文本哈希"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def compute_ngram_signature(self, text: str, n: int = 13) -> set:
        """计算n-gram签名"""
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def is_duplicate(self, text: str) -> bool:
        """检查是否重复"""
        # 精确去重
        text_hash = self.compute_text_hash(text)
        if text_hash in self.hash_set:
            return True
        
        # n-gram去重（模糊去重）
        ngrams = self.compute_ngram_signature(text)
        overlap = len(ngrams & self.ngram_set)
        if overlap / max(len(ngrams), 1) > 0.8:  # 80%以上重叠视为重复
            return True
        
        # 更新集合
        self.hash_set.add(text_hash)
        self.ngram_set.update(ngrams)
        
        return False
    
    def process_document(self, 
                         text: str, 
                         source: str = "unknown") -> Optional[Dict]:
        """处理单个文档"""
        # 清洗
        text = self.clean_text(text)
        
        # 长度过滤
        if len(text) < self.min_length:
            return None
        
        # 去重
        if self.is_duplicate(text):
            return None
        
        return {
            'text': text,
            'source': source,
            'length': len(text),
            'hash': self.compute_text_hash(text)
        }
    
    def process_file(self, 
                     input_path: str, 
                     source: str = "unknown") -> Iterator[Dict]:
        """处理单个文件"""
        input_path = Path(input_path)
        
        if input_path.suffix == '.jsonl':
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        if processed := self.process_document(text, source):
                            yield processed
                    except json.JSONDecodeError:
                        continue
        
        elif input_path.suffix == '.txt':
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if processed := self.process_document(text, source):
                        yield processed
        
        elif input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get('text', '') if isinstance(item, dict) else str(item)
                        if processed := self.process_document(text, source):
                            yield processed
    
    def prepare_dataset(self, 
                        input_paths: List[str],
                        output_name: str = "train",
                        shard_size: int = 100000) -> Dict:
        """准备完整数据集"""
        output_file = self.output_dir / f"{output_name}.jsonl"
        stats = {
            'total_docs': 0,
            'processed_docs': 0,
            'filtered_docs': 0,
            'duplicate_docs': 0,
            'total_tokens': 0
        }
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            shard_count = 0
            
            for input_path in input_paths:
                source = Path(input_path).stem
                
                for doc in self.process_file(input_path, source):
                    stats['total_docs'] += 1
                    
                    out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    stats['processed_docs'] += 1
                    stats['total_tokens'] += doc['length']
                    
                    # 分片
                    if stats['processed_docs'] % shard_size == 0:
                        shard_count += 1
            
        return stats
```

### 模型初始化

增量预训练的模型初始化需要特别注意权重加载和学习率预热。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ContinuedPretrainingInitializer:
    """增量预训练模型初始化器"""
    
    def __init__(self, 
                 base_model_path: str,
                 device: str = "cuda"):
        
        self.base_model_path = base_model_path
        self.device = device
    
    def load_base_model(self, 
                        use_flash_attention: bool = True) -> AutoModelForCausalLM:
        """加载基础模型"""
        
        # 加载模型配置
        model_kwargs = {
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'trust_remote_code': True
        }
        
        # Flash Attention 2 加速
        if use_flash_attention:
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
            except ImportError:
                print("Flash Attention 2 不可用，使用默认注意力机制")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            **model_kwargs
        )
        
        return model
    
    def prepare_model_for_training(self, model) -> AutoModelForCausalLM:
        """准备模型进行训练"""
        
        # 启用梯度检查点（节省显存）
        model.gradient_checkpointing_enable()
        
        # 冻结嵌入层（可选，减少参数）
        # model.get_input_embeddings().weight.requires_grad = False
        
        # 确保模型在训练模式
        model.train()
        
        return model
    
    def compute_initial_loss(self, model, tokenizer, sample_text: str) -> float:
        """计算初始损失（用于验证模型加载正确）"""
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(sample_text, return_tensors='pt').to(self.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            initial_loss = outputs.loss.item()
        
        model.train()
        return initial_loss
    
    def verify_model_loaded_correctly(self, model, tokenizer) -> bool:
        """验证模型是否正确加载"""
        
        # 测试生成
        test_prompt = "这是一个测试"
        inputs = tokenizer(test_prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"测试生成: {generated}")
        return len(generated) > len(test_prompt)


class WeightPerturbationHandler:
    """权重扰动处理器（可选，用于防止过拟合）"""
    
    @staticmethod
    def add_noise_to_embeddings(model, noise_scale: float = 0.01):
        """向嵌入层添加小噪声（打破初始对称性）"""
        with torch.no_grad():
            embed_weight = model.get_input_embeddings().weight
            noise = torch.randn_like(embed_weight) * noise_scale
            embed_weight.add_(noise)
        
        return model
    
    @staticmethod
    def reinitialize_top_layers(model, num_layers: int = 1):
        """重新初始化顶层（用于领域适应）"""
        # 获取模型的层数
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            print("无法识别模型结构，跳过重新初始化")
            return model
        
        # 重新初始化最后几层
        for i in range(-num_layers, 0):
            layer = layers[i]
            
            # 重新初始化注意力层
            for module in layer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dim() >= 2:
                        torch.nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        return model
```

### 训练配置

完整的训练配置包括分布式训练设置、混合精度训练、梯度累积等。

```python
from dataclasses import dataclass, field
from typing import Optional, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class ContinuedPretrainingConfig:
    """增量预训练配置"""
    
    # 模型配置
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")
    use_flash_attention: bool = field(default=True)
    
    # 数据配置
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    max_length: int = field(default=2048)
    preprocessing_num_workers: int = field(default=4)
    
    # 训练配置
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=8)
    
    # 学习率配置
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=500)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    min_lr: float = field(default=1e-6)
    
    # 优化配置
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    
    # 日志和保存
    output_dir: str = field(default="./output")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    
    # 其他
    seed: int = field(default=42)
    dataloader_num_workers: int = field(default=4)


class ContinuedPretrainingTrainer:
    """增量预训练训练器"""
    
    def __init__(self, config: ContinuedPretrainingConfig):
        self.config = config
        
        # 初始化分布式训练
        self._setup_distributed()
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
        # 准备数据
        self._prepare_data()
        
        # 设置优化器和调度器
        self._setup_optimizer_and_scheduler()
    
    def _setup_distributed(self):
        """设置分布式训练"""
        if dist.is_available() and dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1
    
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        initializer = ContinuedPretrainingInitializer(
            self.config.model_name_or_path
        )
        
        self.model = initializer.load_base_model(
            use_flash_attention=self.config.use_flash_attention
        )
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _prepare_data(self):
        """准备数据集"""
        from torch.utils.data import DataLoader
        from datasets import load_dataset
        
        # 加载数据集
        data_files = {}
        if self.config.train_file:
            data_files['train'] = self.config.train_file
        if self.config.validation_file:
            data_files['validation'] = self.config.validation_file
        
        self.dataset = load_dataset('json', data_files=data_files)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config.max_length,
                return_special_tokens_mask=True
            )
        
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=['text']
        )
        
        # 创建DataLoader
        self.train_dataloader = DataLoader(
            self.tokenized_dataset['train'],
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
    
    def _setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        # 分层学习率
        param_groups = LayerWiseLearningRate(
            self.model, 
            self.config.learning_rate
        ).get_parameter_groups()
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 学习率调度器
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.lr_scheduler = ContinuedPretrainingScheduler(
            self.optimizer,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    
    def train(self):
        """执行训练"""
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config.num_train_epochs):
            epoch_loss = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # 移动数据到设备
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        1.0
                    )
                    
                    # 更新参数
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 日志
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / global_step
                        current_lr = self.lr_scheduler.get_lr()
                        print(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    
                    # 保存检查点
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)
            
            print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(self.train_dataloader):.4f}")
        
        return global_step
    
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        output_dir = f"{self.config.output_dir}/checkpoint-{step}"
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Checkpoint saved to {output_dir}")
```

### 评估与验证

增量预训练需要特别关注领域知识学习和通用能力保持之间的平衡。

```python
import numpy as np
from typing import Dict, List, Tuple

class ContinuedPretrainingEvaluator:
    """增量预训练评估器"""
    
    def __init__(self, 
                 base_model,
                 continued_model,
                 tokenizer,
                 domain_test_data: List[str],
                 general_test_data: List[str]):
        
        self.base_model = base_model
        self.continued_model = continued_model
        self.tokenizer = tokenizer
        self.domain_test_data = domain_test_data
        self.general_test_data = general_test_data
    
    def compute_perplexity(self, model, texts: List[str]) -> float:
        """计算困惑度"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def compute_knowledge_retention(self) -> Dict[str, float]:
        """
        计算知识保留率
        
        对比基础模型和增量模型在通用数据上的表现，
        确保模型没有过度遗忘通用知识
        """
        base_ppl = self.compute_perplexity(self.base_model, self.general_test_data)
        continued_ppl = self.compute_perplexity(self.continued_model, self.general_test_data)
        
        retention_rate = 1 - (continued_ppl - base_ppl) / base_ppl
        
        return {
            'base_model_general_ppl': base_ppl,
            'continued_model_general_ppl': continued_ppl,
            'knowledge_retention_rate': retention_rate,
            'degradation': continued_ppl - base_ppl
        }
    
    def compute_domain_improvement(self) -> Dict[str, float]:
        """
        计算领域改进率
        
        评估增量预训练在领域数据上的效果
        """
        base_ppl = self.compute_perplexity(self.base_model, self.domain_test_data)
        continued_ppl = self.compute_perplexity(self.continued_model, self.domain_test_data)
        
        improvement_rate = (base_ppl - continued_ppl) / base_ppl
        
        return {
            'base_model_domain_ppl': base_ppl,
            'continued_model_domain_ppl': continued_ppl,
            'improvement_rate': improvement_rate,
            'absolute_improvement': base_ppl - continued_ppl
        }
    
    def evaluate_forgetting_rate(self, 
                                  task_samples: Dict[str, List[str]]) -> Dict:
        """
        评估灾难性遗忘
        
        通过特定任务样本测试模型是否丢失了原有能力
        """
        results = {}
        
        for task_name, samples in task_samples.items():
            base_ppl = self.compute_perplexity(self.base_model, samples)
            continued_ppl = self.compute_perplexity(self.continued_model, samples)
            
            forgetting_rate = (continued_ppl - base_ppl) / base_ppl
            
            results[task_name] = {
                'base_ppl': base_ppl,
                'continued_ppl': continued_ppl,
                'forgetting_rate': forgetting_rate,
                'status': '严重遗忘' if forgetting_rate > 0.1 else 
                          '轻微遗忘' if forgetting_rate > 0.05 else '正常'
            }
        
        return results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        retention = self.compute_knowledge_retention()
        improvement = self.compute_domain_improvement()
        
        report = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║              增量预训练评估报告                              ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ 领域适应效果:                                                ║
        ║   - 基础模型领域困惑度: {retention['base_model_general_ppl']:.2f}                        ║
        ║   - 增量模型领域困惑度: {improvement['continued_model_domain_ppl']:.2f}                        ║
        ║   - 改进率: {improvement['improvement_rate']*100:.1f}%                                       ║
        ║                                                              ║
        ║ 知识保留情况:                                                ║
        ║   - 基础模型通用困惑度: {retention['base_model_general_ppl']:.2f}                        ║
        ║   - 增量模型通用困惑度: {retention['continued_model_general_ppl']:.2f}                        ║
        ║   - 知识保留率: {retention['knowledge_retention_rate']*100:.1f}%                                     ║
        ║                                                              ║
        ║ 综合评价: {'✅ 成功' if improvement['improvement_rate'] > 0.05 and retention['knowledge_retention_rate'] > 0.9 else '⚠️ 需要调优'}                                          ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
```

## 实践案例

### 医疗领域增量预训练

```python
class MedicalDomainPretraining:
    """医疗领域增量预训练实践"""
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.domain_keywords = [
            '诊断', '治疗', '症状', '患者', '临床', '病理', '药物',
            '手术', '医学', '疾病', '预后', '检查', '影像', '实验室',
            '内科', '外科', '儿科', '妇产科', '肿瘤', '心血管'
        ]
    
    def prepare_medical_data(self, data_sources: Dict[str, str]) -> List[str]:
        """
        准备医疗数据
        
        数据来源示例:
        - 医学文献: PubMed, 中文医学期刊
        - 临床指南: 各专科临床指南
        - 医疗问答: 在线医疗平台问答
        - 医学教材: 医学院校教材
        """
        pipeline = DataPreparationPipeline(output_dir="./medical_data")
        selector = DomainDataSelector(self.domain_keywords)
        
        all_texts = []
        
        for source_name, source_path in data_sources.items():
            # 处理数据文件
            for doc in pipeline.process_file(source_path, source_name):
                # 额外的医疗数据过滤
                if self._is_valid_medical_text(doc['text']):
                    all_texts.append(doc['text'])
        
        return all_texts
    
    def _is_valid_medical_text(self, text: str) -> bool:
        """验证是否为有效的医疗文本"""
        # 检查是否包含足够的医疗关键词
        keyword_count = sum(1 for kw in self.domain_keywords if kw in text)
        
        # 医疗文本通常较长且有专业术语
        return len(text) > 200 and keyword_count >= 2
    
    def configure_training(self) -> ContinuedPretrainingConfig:
        """配置医疗领域训练参数"""
        config = ContinuedPretrainingConfig(
            model_name_or_path=self.base_model_path,
            
            # 医疗数据通常需要更多训练
            num_train_epochs=2,
            
            # 医疗文本通常较长
            max_length=4096,
            
            # 医疗领域学习率可以稍低，保持稳定性
            learning_rate=3e-5,
            warmup_ratio=0.08,
            
            # 医疗数据可能包含敏感信息，注意处理
            output_dir="./medical_pretraining_output"
        )
        
        return config
    
    def evaluate_medical_knowledge(self, model, tokenizer) -> Dict:
        """评估医疗知识掌握程度"""
        
        # 医疗知识测试问题
        test_questions = [
            "糖尿病的主要症状有哪些？",
            "高血压患者的日常注意事项是什么？",
            "什么是心肌梗死？",
            "抗生素的正确使用方法是什么？"
        ]
        
        results = []
        
        for question in test_questions:
            inputs = tokenizer(question, return_tensors='pt').to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'question': question,
                'answer': answer
            })
        
        return results


def run_medical_pretraining():
    """运行医疗领域增量预训练"""
    
    # 初始化
    medical_pt = MedicalDomainPretraining(
        base_model_path="meta-llama/Llama-2-7b-hf"
    )
    
    # 数据源配置
    data_sources = {
        'medical_journals': './data/medical_journals.jsonl',
        'clinical_guidelines': './data/clinical_guidelines.jsonl',
        'medical_qa': './data/medical_qa.jsonl'
    }
    
    # 准备数据
    print("准备医疗数据...")
    medical_data = medical_pt.prepare_medical_data(data_sources)
    
    # 配置训练
    config = medical_pt.configure_training()
    
    # 执行训练
    print("开始医疗领域增量预训练...")
    trainer = ContinuedPretrainingTrainer(config)
    trainer.train()
    
    print("医疗领域增量预训练完成！")
```

### 法律领域增量预训练

```python
class LegalDomainPretraining:
    """法律领域增量预训练实践"""
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.legal_categories = [
            '刑法', '民法', '商法', '行政法', '宪法',
            '合同法', '公司法', '劳动法', '知识产权法'
        ]
    
    def prepare_legal_data(self) -> Dict[str, List[str]]:
        """准备法律数据"""
        legal_data = {
            'statutes': [],      # 法条
            'cases': [],         # 判例
            'contracts': [],     # 合同模板
            'legal_opinions': [] # 法律意见书
        }
        
        # 处理不同类型的法律文档
        # 法条处理需要保持完整性
        # 判例需要包含案件事实和判决结果
        # 合同模板需要保持格式规范
        
        return legal_data
    
    def configure_training(self) -> ContinuedPretrainingConfig:
        """配置法律领域训练参数"""
        config = ContinuedPretrainingConfig(
            model_name_or_path=self.base_model_path,
            
            # 法律文本结构性强，可以适当增加长度
            max_length=8192,
            
            # 法律条文要求精确，学习率可以更低
            learning_rate=2e-5,
            
            # 法律领域数据通常质量较高，可以适当减少epoch
            num_train_epochs=1,
            
            output_dir="./legal_pretraining_output"
        )
        
        return config
    
    def create_legal_prompt_template(self) -> str:
        """创建法律领域提示模板"""
        template = """
        作为一名专业的法律顾问，请根据以下法律条文和案例，回答用户的问题。
        
        相关法条：
        {statutes}
        
        相关案例：
        {cases}
        
        用户问题：{question}
        
        法律分析：
        """
        return template
```

### 代码领域增量预训练

```python
class CodeDomainPretraining:
    """代码领域增量预训练实践"""
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.programming_languages = [
            'python', 'javascript', 'java', 'cpp', 'go', 'rust'
        ]
    
    def prepare_code_data(self, 
                          code_repositories: List[str],
                          documentation: List[str]) -> List[Dict]:
        """准备代码数据"""
        code_samples = []
        
        for repo_path in code_repositories:
            # 遍历代码仓库
            for file_path in self._find_code_files(repo_path):
                code_content = self._read_code_file(file_path)
                
                if self._is_valid_code(code_content):
                    # 提取代码和注释
                    code_info = self._extract_code_info(
                        code_content, 
                        file_path
                    )
                    code_samples.append(code_info)
        
        return code_samples
    
    def _find_code_files(self, repo_path: str) -> List[str]:
        """查找代码文件"""
        code_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.go', '.rs',
            '.ts', '.jsx', '.tsx', '.h', '.hpp'
        }
        
        code_files = []
        # 实际实现中需要遍历目录
        
        return code_files
    
    def _extract_code_info(self, code: str, file_path: str) -> Dict:
        """提取代码信息"""
        import os
        
        # 获取文件扩展名
        ext = os.path.splitext(file_path)[1]
        language = self._ext_to_language(ext)
        
        # 提取函数和类定义
        functions = self._extract_functions(code, language)
        classes = self._extract_classes(code, language)
        
        return {
            'code': code,
            'language': language,
            'file_path': file_path,
            'functions': functions,
            'classes': classes
        }
    
    def _ext_to_language(self, ext: str) -> str:
        """扩展名转语言名"""
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust'
        }
        return mapping.get(ext.lower(), 'unknown')
    
    def configure_training(self) -> ContinuedPretrainingConfig:
        """配置代码领域训练参数"""
        config = ContinuedPretrainingConfig(
            model_name_or_path=self.base_model_path,
            
            # 代码需要更长的上下文
            max_length=4096,
            
            # 代码语法严格，学习率可以适中
            learning_rate=5e-5,
            
            num_train_epochs=1,
            
            output_dir="./code_pretraining_output"
        )
        
        return config
```

## 注意事项与最佳实践

### 灾难性遗忘问题

灾难性遗忘（Catastrophic Forgetting）是增量预训练面临的主要挑战，指模型在学习新知识时丢失了已学习的旧知识。

**问题分析：**

```
灾难性遗忘的表现：
┌─────────────────────────────────────────────────────────────┐
│  原有能力           │  增量预训练后状态         │  影响程度  │
├─────────────────────────────────────────────────────────────┤
│  通用语言理解       │  可能下降                 │  ⚠️ 中等   │
│  常识推理           │  可能下降                 │  ⚠️ 中等   │
│  指令遵循           │  通常保持                 │  ✅ 轻微   │
│  多语言能力         │  可能显著下降             │  ⚠️ 严重   │
│  通用世界知识       │  部分遗忘                 │  ⚠️ 中等   │
└─────────────────────────────────────────────────────────────┘
```

**缓解策略：**

```python
class CatastrophicForgettingMitigator:
    """灾难性遗忘缓解器"""
    
    def __init__(self, model, general_data_ratio: float = 0.2):
        self.model = model
        self.general_data_ratio = general_data_ratio
        
        # 存储原始模型的关键参数
        self.original_params = {}
        self._store_original_params()
    
    def _store_original_params(self):
        """存储原始参数用于约束"""
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
    
    def elastic_weight_consolidation_loss(self, 
                                           current_params,
                                           fisher_diagonal: Dict[str, torch.Tensor],
                                           lambda_ewc: float = 1000) -> torch.Tensor:
        """
        弹性权重巩固（EWC）损失
        
        通过Fisher信息矩阵对重要参数施加约束，防止过度修改
        
        数学形式：
        L_EWC = λ/2 * Σ_i F_i * (θ_i - θ*_i)^2
        """
        loss = 0
        
        for name, param in current_params.items():
            if name in self.original_params:
                # Fisher信息矩阵对角元素表示参数重要性
                fisher = fisher_diagonal.get(name, torch.ones_like(param))
                original = self.original_params[name]
                
                loss += (fisher * (param - original) ** 2).sum()
        
        return lambda_ewc / 2 * loss
    
    def compute_fisher_diagonal(self, 
                                dataloader, 
                                num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵的对角近似
        
        用于评估每个参数对模型输出的重要性
        """
        fisher_diagonal = {}
        
        # 初始化
        for name, param in self.model.named_parameters():
            fisher_diagonal[name] = torch.zeros_like(param)
        
        self.model.eval()
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            self.model.zero_grad()
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 累积梯度平方
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_diagonal[name] += param.grad.data ** 2
            
            sample_count += 1
        
        # 归一化
        for name in fisher_diagonal:
            fisher_diagonal[name] /= sample_count
        
        return fisher_diagonal


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, 
                 general_data: List[str],
                 buffer_size: int = 10000,
                 sample_ratio: float = 0.2):
        """
        Args:
            general_data: 通用数据样本
            buffer_size: 缓冲区大小
            sample_ratio: 每个batch中通用数据的比例
        """
        self.buffer = general_data[:buffer_size]
        self.sample_ratio = sample_ratio
    
    def sample(self, batch_size: int) -> List[str]:
        """从缓冲区采样"""
        import random
        sample_size = int(batch_size * self.sample_ratio)
        return random.sample(self.buffer, min(sample_size, len(self.buffer)))
    
    def create_mixed_batch(self,
                           domain_batch: List[str],
                           batch_size: int) -> List[str]:
        """创建混合批次"""
        general_samples = self.sample(batch_size)
        return domain_batch + general_samples
```

### 数据质量控制

```python
class DataQualityController:
    """数据质量控制"""
    
    def __init__(self, quality_thresholds: Dict[str, float] = None):
        self.thresholds = quality_thresholds or {
            'min_length': 100,
            'max_length': 50000,
            'min_quality_score': 0.5,
            'max_duplicate_ratio': 0.3,
            'min_domain_relevance': 0.4
        }
    
    def validate_batch(self, batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """验证批次数据质量"""
        valid_samples = []
        invalid_samples = []
        
        for sample in batch:
            validation_result = self._validate_sample(sample)
            
            if validation_result['is_valid']:
                valid_samples.append(sample)
            else:
                sample['validation_issues'] = validation_result['issues']
                invalid_samples.append(sample)
        
        return valid_samples, invalid_samples
    
    def _validate_sample(self, sample: Dict) -> Dict:
        """验证单个样本"""
        issues = []
        
        text = sample.get('text', '')
        
        # 长度检查
        if len(text) < self.thresholds['min_length']:
            issues.append(f"文本过短: {len(text)} < {self.thresholds['min_length']}")
        
        if len(text) > self.thresholds['max_length']:
            issues.append(f"文本过长: {len(text)} > {self.thresholds['max_length']}")
        
        # 质量分数检查
        quality_score = sample.get('quality_score', 1.0)
        if quality_score < self.thresholds['min_quality_score']:
            issues.append(f"质量分数过低: {quality_score}")
        
        # 领域相关性检查
        domain_relevance = sample.get('domain_relevance', 1.0)
        if domain_relevance < self.thresholds['min_domain_relevance']:
            issues.append(f"领域相关性过低: {domain_relevance}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    def generate_quality_report(self, 
                                 dataset_stats: Dict) -> str:
        """生成数据质量报告"""
        report = f"""
        ╔══════════════════════════════════════════════════╗
        ║              数据质量报告                        ║
        ╠══════════════════════════════════════════════════╣
        ║ 总样本数: {dataset_stats['total_samples']:,}                              ║
        ║ 有效样本数: {dataset_stats['valid_samples']:,}                            ║
        ║ 有效率: {dataset_stats['valid_rate']*100:.1f}%                                  ║
        ║                                                  ║
        ║ 平均长度: {dataset_stats['avg_length']:.0f}                                 ║
        ║ 平均质量分数: {dataset_stats['avg_quality']:.2f}                             ║
        ║ 平均领域相关性: {dataset_stats['avg_relevance']:.2f}                          ║
        ╚══════════════════════════════════════════════════╝
        """
        return report
```

### 训练稳定性

```python
class TrainingStabilityManager:
    """训练稳定性管理"""
    
    def __init__(self, 
                 model,
                 loss_spike_threshold: float = 3.0,
                 gradient_norm_threshold: float = 10.0):
        
        self.model = model
        self.loss_spike_threshold = loss_spike_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        
        self.loss_history = []
        self.gradient_norm_history = []
    
    def check_loss_spike(self, current_loss: float) -> bool:
        """检测损失突增"""
        if len(self.loss_history) < 10:
            self.loss_history.append(current_loss)
            return False
        
        # 计算最近的平均损失
        recent_avg = np.mean(self.loss_history[-10:])
        
        # 检测突增
        is_spike = current_loss > recent_avg * self.loss_spike_threshold
        
        if is_spike:
            print(f"⚠️ 损失突增检测: 当前 {current_loss:.4f} > 平均 {recent_avg:.4f} * {self.loss_spike_threshold}")
        
        self.loss_history.append(current_loss)
        
        return is_spike
    
    def check_gradient_explosion(self) -> bool:
        """检测梯度爆炸"""
        total_norm = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        is_explosion = total_norm > self.gradient_norm_threshold
        self.gradient_norm_history.append(total_norm)
        
        if is_explosion:
            print(f"⚠️ 梯度爆炸检测: 梯度范数 {total_norm:.4f} > {self.gradient_norm_threshold}")
        
        return is_explosion
    
    def recover_from_instability(self, optimizer, original_params: Dict = None):
        """从不稳定状态恢复"""
        print("正在从不稳定状态恢复...")
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 如果有原始参数，恢复部分参数
        if original_params:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in original_params:
                        param.data = original_params[name].clone()
        
        print("恢复完成")
    
    def get_stability_metrics(self) -> Dict:
        """获取稳定性指标"""
        return {
            'loss_variance': np.var(self.loss_history[-100:]) if len(self.loss_history) >= 100 else 0,
            'gradient_norm_trend': np.mean(self.gradient_norm_history[-100:]) if len(self.gradient_norm_history) >= 100 else 0,
            'loss_trend': 'decreasing' if len(self.loss_history) >= 50 and 
                         np.mean(self.loss_history[-25:]) < np.mean(self.loss_history[-50:-25]) else 'stable_or_increasing'
        }
```

## 知识点关联

### 技术演进关系

```
全量预训练（从零开始，计算成本极高）
    ↓ 知识需求
领域适配需求（特定行业、特定场景）
    ↓ 技术方案
增量预训练（利用已有知识，高效适配）
    ↓ 挑战
灾难性遗忘问题
    ↓ 解决方案
EWC、经验回放、混合训练策略
```

### 与其他技术的关系

| 相关技术 | 关系描述 |
|---------|---------|
| 全量预训练 | 增量预训练的基础，提供通用知识 |
| 指令微调 | 增量预训练后可接指令微调提升任务能力 |
| PEFT | 可结合使用，仅训练部分参数 |
| RLHF | 增量预训练后的对齐技术 |

## 章节核心考点

### 关键技术原理
- 增量预训练的数学形式化
- 领域数据选择和质量控制策略
- 学习率调度的特殊考虑
- 灾难性遗忘的成因和缓解方法

### 实践技能要求
- 设计完整的数据准备流水线
- 配置增量预训练参数
- 实现训练稳定性监控
- 评估领域适配效果

### 数学基础考点
- EWC损失函数推导
- Fisher信息矩阵的计算
- 学习率衰减曲线设计
- 困惑度计算原理

## 学习建议

### 深入学习建议
1. **实践先行**：在小规模数据上尝试完整流程
2. **对比实验**：对比不同数据配比和学习率设置的效果
3. **监控分析**：深入分析训练过程中的各项指标
4. **案例研究**：研究领域模型的成功案例（如Med-PaLM）

### 延伸方向
- **多领域适配**：同时适配多个领域
- **持续学习**：增量预训练的增量式扩展
- **参数高效方法**：结合LoRA等PEFT技术
- **评估体系**：构建领域知识的评估基准

---

*通过本章学习，您将掌握如何高效地将通用大模型适配到特定领域，在保持通用能力的同时注入专业知识。*
