# 视觉语言模型

视觉语言模型（Vision-Language Models, VLM）是一类能够同时理解和处理视觉信息（图像、视频）与语言信息（文本）的深度学习模型。它们在图像描述、视觉问答、图像-文本检索等任务中表现出色，是实现通用人工智能的重要方向。

## 章节概述

本章将系统介绍视觉语言模型的核心概念和关键技术：

| 主题 | 核心内容 | 应用场景 |
|------|----------|----------|
| CLIP | 对比学习、图像-文本对齐 | 图像检索、零样本分类 |
| BLIP/BLIP-2 | Q-Former、引导式预训练 | 图文理解、生成任务 |
| LLaVA | 视觉编码器+LLM融合 | 多模态对话 |
| 指令微调 | 多模态指令数据构建 | 任务泛化 |
| VQA/Captioning/Grounding | 下游任务 | 问答、描述、定位 |

## 视觉语言模型概述

### 什么是视觉语言模型

视觉语言模型旨在学习**视觉模态**与**语言模态**之间的语义对应关系，使模型能够：

1. **理解图像内容**：识别图像中的物体、场景、动作等
2. **生成语言描述**：用自然语言描述图像内容
3. **跨模态推理**：基于图像回答问题或执行指令
4. **跨模态检索**：根据文本找图像或根据图像找文本

### 发展历程

```
2015: Show and Tell (NIC) → 首个端到端图像描述模型
2019: ViLBERT, LXMERT → 双流架构预训练
2021: CLIP, ALIGN → 大规模对比学习
2022: BLIP, Flamingo → 统一理解与生成
2023: LLaVA, BLIP-2 → 与LLM融合
2024: GPT-4V, Gemini → 通用多模态大模型
```

### 核心架构类型

```
┌─────────────────────────────────────────────────────────────┐
│                    视觉语言模型架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 双流架构 (Two-Stream)                                    │
│     ┌──────┐    ┌──────┐    ┌──────────┐                   │
│     │Image │───▶│Vision│    │          │                   │
│     │Encoder│   │Encoder│───▶│Cross-Attn│───▶ Output       │
│     └──────┘    └──────┘    │          │                   │
│     ┌──────┐    ┌──────┐    │          │                   │
│     │Text  │───▶│Text  │───▶│          │                   │
│     │Input │    │Encoder│   └──────────┘                   │
│     └──────┘    └──────┘                                    │
│                                                             │
│  2. 单流架构 (Single-Stream)                                 │
│     ┌──────┐    ┌──────┐    ┌──────────────────┐           │
│     │Image │───▶│Vision│───▶│                  │           │
│     │      │    │Encoder│   │   Joint Encoder  │───▶Output │
│     └──────┘    └──────┘    │   (BERT-style)   │           │
│     ┌──────┐    ┌──────┐    │                  │           │
│     │Text  │───▶│Embed │───▶│                  │           │
│     └──────┘    └──────┘    └──────────────────┘           │
│                                                             │
│  3. LLM融合架构 (LLM-based)                                  │
│     ┌──────┐    ┌────────┐    ┌────────┐    ┌─────┐       │
│     │Image │───▶│Vision  │───▶│        │    │     │       │
│     │      │    │Encoder │   │Adapter │───▶│ LLM │───▶Out │
│     └──────┘    └────────┘    │        │    │     │       │
│     ┌──────┐                  │        │    │     │       │
│     │Text  │─────────────────▶│        │───▶│     │       │
│     └──────┘                  └────────┘    └─────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## CLIP：对比语言-图像预训练

### 核心思想

CLIP（Contrastive Language-Image Pre-training）由OpenAI提出，通过**对比学习**在大规模图像-文本对上预训练，实现零样本迁移能力。

**核心洞察**：
- 不预测固定类别，而是学习图像与文本的**语义对齐**
- 预训练阶段学习通用视觉概念
- 通过自然语言实现灵活的零样本分类

### 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                       CLIP 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     图像批次 N                   文本批次 N                   │
│     ┌─────┐                     ┌─────┐                    │
│     │ I₁  │                     │ T₁  │                    │
│     │ I₂  │                     │ T₂  │                    │
│     │ ... │                     │ ... │                    │
│     │ Iₙ  │                     │ Tₙ  │                    │
│     └──┬──┘                     └──┬──┘                    │
│        │                           │                        │
│        ▼                           ▼                        │
│   ┌─────────┐                ┌─────────┐                   │
│   │ Image   │                │  Text   │                   │
│   │ Encoder │                │ Encoder │                   │
│   │ (ViT)   │                │(Trans.) │                   │
│   └────┬────┘                └────┬────┘                   │
│        │                           │                        │
│        ▼                           ▼                        │
│   ┌─────────┐                ┌─────────┐                   │
│   │ Wᵢ ∈ ℝᵈ │                │ Wₜ ∈ ℝᵈ │                   │
│   │ 图像特征 │                │ 文本特征 │                   │
│   └────┬────┘                └────┬────┘                   │
│        │                           │                        │
│        │      L2 归一化            │                        │
│        │         ↓                 │                        │
│        │    fᵢ = Wᵢ/‖Wᵢ‖          │                        │
│        │    fₜ = Wₜ/‖Wₜ‖          │                        │
│        │                           │                        │
│        └───────────┬───────────────┘                        │
│                    │                                        │
│                    ▼                                        │
│              ┌───────────┐                                  │
│              │ 相似度矩阵 │                                  │
│              │ S = fᵢfₜᵀ │                                  │
│              │   × τ     │                                  │
│              └─────┬─────┘                                  │
│                    │                                        │
│                    ▼                                        │
│         对称对比学习损失 (N×N)                               │
│         对角线为正样本对，其余为负样本                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 对比学习损失函数

CLIP使用**对称对比损失**，对于批次中的N个图像-文本对：

**图像到文本的损失**：

$$
\mathcal{L}_{i \to t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)}
$$

**文本到图像的损失**：

$$
\mathcal{L}_{t \to i} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(t_i, v_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(t_i, v_j) / \tau)}
$$

**总损失**：

$$
\mathcal{L}_{CLIP} = \frac{\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}}{2}
$$

其中：
- $v_i$：图像编码器输出的特征向量
- $t_i$：文本编码器输出的特征向量
- $\tau$：可学习的温度参数
- $\text{sim}(a, b) = \cos(a, b) = \frac{a^T b}{\|a\| \|b\|}$

### 零样本分类

CLIP的核心优势是**零样本分类**，无需微调即可在新数据集上分类：

```
分类步骤：
1. 为每个类别构造提示文本："a photo of {class_name}"
2. 用文本编码器编码所有类别描述
3. 用图像编码器编码待分类图像
4. 计算图像特征与所有文本特征的相似度
5. 选择相似度最高的类别作为预测结果
```

**数学表示**：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \text{sim}(f_I(I), f_T(\text{"a photo of " } c))
$$

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPLoss(nn.Module):
    """CLIP对比学习损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / temperature))
        )
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """
        Args:
            image_features: [batch_size, dim] 图像特征
            text_features: [batch_size, dim] 文本特征
        Returns:
            loss: 对比学习损失
        """
        # L2归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵 [batch, batch]
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # 构造标签：对角线为正样本
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 对称交叉熵损失
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2


class CLIPZeroShotClassifier:
    """CLIP零样本分类器"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def classify(self, image, class_names: list, prompt: str = "a photo of {}"):
        """
        零样本分类
        
        Args:
            image: PIL图像
            class_names: 类别名称列表
            prompt: 提示模板
        
        Returns:
            预测类别和概率分布
        """
        # 构造文本提示
        texts = [prompt.format(c) for c in class_names]
        
        # 预处理
        inputs = self.processor(
            text=texts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 获取相似度分数
            logits = outputs.logits_per_image  # [1, num_classes]
            probs = logits.softmax(dim=-1)
        
        # 返回预测结果
        pred_idx = probs.argmax().item()
        return class_names[pred_idx], probs[0].cpu().numpy()


# 使用示例
if __name__ == "__main__":
    from PIL import Image
    
    # 创建零样本分类器
    classifier = CLIPZeroShotClassifier()
    
    # 定义类别
    classes = ["cat", "dog", "bird", "car", "tree"]
    
    # 加载图像（示例）
    # image = Image.open("test.jpg")
    # pred_class, probs = classifier.classify(image, classes)
    # print(f"预测类别: {pred_class}")
    # print(f"概率分布: {probs}")
```

---

## BLIP/BLIP-2：引导式语言-图像预训练

### BLIP核心贡献

BLIP（Bootstrapping Language-Image Pre-training）解决了现有VLM的两个问题：

1. **数据噪声**：网络爬取的图像-文本对质量参差不齐
2. **任务差距**：理解任务与生成任务需要不同的架构

**核心创新**：

- **CapFilt**：使用模型过滤噪声数据并生成高质量描述
- **统一架构**：单模型同时支持理解和生成任务

### BLIP架构

```
┌─────────────────────────────────────────────────────────────┐
│                       BLIP 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Vision Encoder (ViT)                    │   │
│  │    Image Patches → Patch Embed → Transformer        │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Image-grounded Text Encoder               │   │
│  │                                                      │   │
│  │   [CLS] + Image Features + Text → BERT Encoder      │   │
│  │                    ↓                                 │   │
│  │        用于图文检索、VQA理解任务                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Image-grounded Text Decoder               │   │
│  │                                                      │   │
│  │   [DEC] + Image Features + Text → GPT Decoder       │   │
│  │                    ↓                                 │   │
│  │        用于图像描述、生成任务                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### BLIP-2：Q-Former

BLIP-2引入了**Q-Former**（Querying Transformer），实现了视觉编码器与LLM的高效桥接：

```
┌─────────────────────────────────────────────────────────────┐
│                     BLIP-2 Q-Former                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                                         │
│   │   Vision     │                                         │
│   │   Encoder    │                                         │
│   │   (Frozen)   │                                         │
│   └──────┬───────┘                                         │
│          │                                                  │
│          │ Visual Features                                  │
│          ▼                                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Q-Former                             │     │
│   │  ┌─────────────────────────────────────────────┐ │     │
│   │  │  Learnable Queries: Q₁, Q₂, ..., Qₙ         │ │     │
│   │  │       [n_queries, dim]                       │ │     │
│   │  └────────────────────┬────────────────────────┘ │     │
│   │                       │                           │     │
│   │                       ▼                           │     │
│   │  ┌─────────────────────────────────────────────┐ │     │
│   │  │   Cross-Attention Layer                      │ │     │
│   │  │   Q: Learnable Queries                       │ │     │
│   │  │   K, V: Visual Features                      │ │     │
│   │  └────────────────────┬────────────────────────┘ │     │
│   │                       │                           │     │
│   │                       ▼                           │     │
│   │  ┌─────────────────────────────────────────────┐ │     │
│   │  │   Self-Attention + FFN                      │ │     │
│   │  │   (with text tokens for understanding)      │ │     │
│   │  └─────────────────────────────────────────────┘ │     │
│   └──────────────────────┬───────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│   ┌──────────────────────────────────────────────────┐     │
│   │                 LLM (Frozen)                      │     │
│   │   Query Representations → LLM → Text Generation  │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Q-Former数学表示

Q-Former包含一组**可学习的查询向量**：

$$
Q = [q_1, q_2, ..., q_N] \in \mathbb{R}^{N \times d}
$$

通过交叉注意力从视觉特征中提取信息：

$$
\text{CrossAttn}(Q, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $K, V$ 来自视觉编码器的输出特征。

**两阶段预训练**：

1. **阶段一**：冻结视觉编码器，训练Q-Former学习视觉-语言表示
2. **阶段二**：冻结LLM，训练Q-Former适配LLM的输入空间

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QFormer(nn.Module):
    """简化版Q-Former实现"""
    
    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # 视觉特征投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vision_features: [batch, num_patches, vision_dim]
            attention_mask: 可选的注意力掩码
        
        Returns:
            query_features: [batch, num_queries, hidden_dim]
        """
        batch_size = vision_features.size(0)
        
        # 扩展查询向量到批次 [batch, num_queries, hidden_dim]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 投影视觉特征
        vision_proj = self.vision_proj(vision_features)
        
        # 拼接查询和视觉特征
        # 这里简化处理，实际BLIP-2使用交叉注意力
        combined = torch.cat([queries, vision_proj], dim=1)
        
        # 通过Transformer
        output = self.transformer(combined)
        
        # 提取查询特征
        query_features = output[:, :self.queries.size(0), :]
        
        return self.layer_norm(query_features)


class BLIP2ForImageTextRetrieval(nn.Module):
    """BLIP-2图文检索模型"""
    
    def __init__(
        self,
        vision_model: nn.Module,
        qformer: QFormer,
        text_proj: nn.Module,
        temperature: float = 0.07
    ):
        super().__init__()
        self.vision_model = vision_model
        self.qformer = qformer
        self.text_proj = text_proj
        
        # 用于检索的投影层
        self.image_proj = nn.Linear(768, 256)
        self.text_proj_out = nn.Linear(768, 256)
        
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / temperature))
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        图文检索的对比学习
        """
        # 提取视觉特征
        vision_output = self.vision_model(pixel_values)
        vision_features = vision_output.last_hidden_state
        
        # 通过Q-Former
        query_features = self.qformer(vision_features)
        
        # 池化得到图像表示
        image_embeds = self.image_proj(query_features.mean(dim=1))
        
        # 文本编码（简化）
        # text_embeds = self.text_proj(input_ids)
        text_embeds = self.text_proj_out(
            torch.randn(input_ids.size(0), 768, device=input_ids.device)
        )
        
        # 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_embeds.t()
        
        return logits


# 使用示例
if __name__ == "__main__":
    # 创建Q-Former
    qformer = QFormer(
        vision_dim=768,
        hidden_dim=768,
        num_queries=32,
        num_heads=12,
        num_layers=6
    )
    
    # 模拟视觉特征
    vision_features = torch.randn(2, 196, 768)  # batch=2, 14x14 patches
    
    # 前向传播
    query_features = qformer(vision_features)
    print(f"Query features shape: {query_features.shape}")  # [2, 32, 768]
```

---

## LLaVA：大型语言和视觉助手

### 架构设计

LLaVA（Large Language and Vision Assistant）采用简单而有效的方式将视觉编码器与LLM连接：

```
┌─────────────────────────────────────────────────────────────┐
│                       LLaVA 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────────────────────────────────────────┐     │
│   │              输入图像                              │     │
│   └─────────────────────┬────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │         Vision Encoder (CLIP ViT)                 │     │
│   │         输出: [H*W, hidden_dim]                   │     │
│   └─────────────────────┬────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │         Vision-Language Adapter                   │     │
│   │         可训练的线性投影层                         │     │
│   │         W ∈ ℝ^(vision_dim × llm_dim)             │     │
│   └─────────────────────┬────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │         Visual Tokens (图像token序列)              │     │
│   │         与文本token拼接                           │     │
│   └─────────────────────┬────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │              LLM (Vicuna/Llama)                   │     │
│   │         自回归生成文本响应                         │     │
│   └─────────────────────┬────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│   ┌──────────────────────────────────────────────────┐     │
│   │              文本输出                              │     │
│   │   "这张图片显示了一只橙色的猫正在..."              │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

**1. 视觉编码器**：使用预训练的CLIP ViT，提取图像特征

$$
V = f_{\text{CLIP}}(I) \in \mathbb{R}^{N \times d_v}
$$

其中 $N$ 是patch数量，$d_v$ 是视觉特征维度。

**2. 视觉-语言适配器**：简单的线性投影

$$
H_v = V \cdot W \in \mathbb{R}^{N \times d_l}
$$

其中 $W \in \mathbb{R}^{d_v \times d_l}$ 是可训练参数，$d_l$ 是LLM的嵌入维度。

**3. 输入格式**：将图像token与文本token拼接

```
输入序列: <image> [图像token序列] </image> 用户指令 文本
```

### 训练策略

**阶段一：预训练（特征对齐）**

- 冻结视觉编码器和LLM
- 只训练适配器层
- 目标：让LLM理解视觉特征

**阶段二：视觉指令微调**

- 冻结视觉编码器
- 训练适配器和LLM（LoRA）
- 目标：提升多模态对话能力

### PyTorch实现

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

class LlavaVisionTower(nn.Module):
    """LLaVA视觉编码器"""
    
    def __init__(self, vision_model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_tower.requires_grad_(False)  # 冻结
    
    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [batch, 3, H, W] 归一化后的图像
        
        Returns:
            image_features: [batch, num_patches, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.vision_tower(images)
        return outputs.last_hidden_state


class LlavaProjector(nn.Module):
    """视觉-语言投影层"""
    
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        # 两层MLP投影
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, vision_features: torch.Tensor):
        """
        Args:
            vision_features: [batch, num_patches, vision_dim]
        
        Returns:
            projected_features: [batch, num_patches, llm_dim]
        """
        return self.projector(vision_features)


class LlavaModel(nn.Module):
    """简化的LLaVA模型"""
    
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        llm_model_name: str = "lmsys/vicuna-7b-v1.5",
        vision_dim: int = 1024,
        llm_dim: int = 4096
    ):
        super().__init__()
        
        # 视觉编码器（冻结）
        self.vision_tower = LlavaVisionTower(vision_model_name)
        
        # 投影层（可训练）
        self.projector = LlavaProjector(vision_dim, llm_dim)
        
        # LLM（可选冻结部分参数）
        self.llm = LlamaForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 特殊token
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        Args:
            images: [batch, 3, H, W] 输入图像
            input_ids: [batch, seq_len] 输入token ids
            attention_mask: [batch, seq_len] 注意力掩码
            labels: [batch, seq_len] 标签（用于训练）
        
        Returns:
            outputs: LLM输出
        """
        # 1. 提取视觉特征
        vision_features = self.vision_tower(images)
        
        # 2. 投影到LLM空间
        image_tokens = self.projector(vision_features)
        
        # 3. 获取文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 4. 替换图像占位符（简化示例）
        # 实际实现需要更复杂的token处理
        # 这里假设已经处理好embedding拼接
        
        # 5. LLM前向传播
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        images: torch.Tensor,
        prompt: str,
        tokenizer: LlamaTokenizer,
        max_new_tokens: int = 512
    ):
        """生成回答"""
        # 编码文本
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(images.device)
        
        # 提取并投影视觉特征
        vision_features = self.vision_tower(images)
        image_embeds = self.projector(vision_features)
        
        # 获取文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 拼接（简化：实际需要处理位置）
        inputs_embeds = torch.cat([
            image_embeds.expand(text_embeds.size(0), -1, -1),
            text_embeds
        ], dim=1)
        
        # 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = LlavaModel(
        vision_model_name="openai/clip-vit-large-patch14",
        llm_model_name="lmsys/vicuna-7b-v1.5"
    )
    
    # 模拟输入
    images = torch.randn(1, 3, 336, 336)  # CLIP输入尺寸
    prompt = "<image>\nDescribe this image in detail."
    
    # 生成回答
    # tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    # response = model.generate(images, prompt, tokenizer)
    # print(response)
```

---

## 多模态指令微调

### 指令数据构建

多模态指令微调的核心是构建高质量的**视觉-语言指令数据**：

```
┌─────────────────────────────────────────────────────────────┐
│                   指令数据格式                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入:                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ <image>                                              │   │
│  │ [视觉token序列]                                       │   │
│  │ </image>                                              │   │
│  │ USER: 请描述这张图片中的主要物体。                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  输出:                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ASSISTANT: 这张图片显示了一个阳光明媚的公园场景。      │   │
│  │ 图中可以看到一片绿色的草坪，几棵大树提供阴凉，         │   │
│  │ 还有几个孩子在草地上玩耍...                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 数据来源

**1. 描述数据转换**

将图像描述数据转换为对话格式：

```python
# 原始数据
{"image": "xxx.jpg", "caption": "一只猫坐在沙发上"}

# 转换后
{
    "image": "xxx.jpg",
    "conversations": [
        {"role": "user", "content": "描述这张图片。"},
        {"role": "assistant", "content": "一只猫坐在沙发上"}
    ]
}
```

**2. VQA数据转换**

```python
# 原始VQA数据
{"image": "xxx.jpg", "question": "图中有几只猫?", "answer": "两只"}

# 转换后
{
    "image": "xxx.jpg",
    "conversations": [
        {"role": "user", "content": "图中有几只猫?"},
        {"role": "assistant", "content": "图中有两只猫。"}
    ]
}
```

**3. 自指令生成**

使用GPT-4生成更丰富的指令数据：

```python
# 基于边界框信息生成详细描述
bounding_boxes = [
    {"label": "cat", "bbox": [100, 100, 200, 200]},
    {"label": "sofa", "bbox": [50, 150, 300, 250]}
]

prompt = f"""
Given the objects in the image: {bounding_boxes}
Generate a detailed description of the image.
"""
```

### 训练损失

**自回归语言模型损失**：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, V; \theta)
$$

其中 $V$ 是视觉特征，$x_t$ 是第 $t$ 个token。

**LoRA微调**：

对于大模型，使用LoRA进行参数高效微调：

$$
W' = W + \Delta W = W + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

### PyTorch实现

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict
from transformers import Trainer

@dataclass
class MultiModalConversation:
    """多模态对话数据结构"""
    image_path: str
    conversations: List[Dict[str, str]]


class MultiModalDataCollator:
    """多模态数据整理器"""
    
    def __init__(self, tokenizer, image_processor, max_length=2048):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __call__(self, batch: List[MultiModalConversation]):
        """
        整理一个批次的数据
        
        Args:
            batch: 多模态对话列表
        
        Returns:
            整理后的批次数据
        """
        images = []
        input_ids = []
        labels = []
        
        for item in batch:
            # 处理图像
            image = self.load_image(item.image_path)
            images.append(self.image_processor(image, return_tensors="pt"))
            
            # 构建对话文本
            text = self.format_conversation(item.conversations)
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids.append(tokenized.input_ids)
            
            # 构建标签（掩盖用户输入）
            label = self.create_labels(item.conversations, tokenized)
            labels.append(label)
        
        return {
            "pixel_values": torch.cat([img["pixel_values"] for img in images]),
            "input_ids": torch.cat(input_ids),
            "labels": torch.cat(labels)
        }
    
    def format_conversation(self, conversations):
        """格式化对话"""
        text = ""
        for conv in conversations:
            if conv["role"] == "user":
                text += f"USER: {conv['content']} "
            else:
                text += f"ASSISTANT: {conv['content']}</s>"
        return text
    
    def create_labels(self, conversations, tokenized):
        """创建标签，掩盖用户输入部分"""
        labels = tokenized.input_ids.clone()
        
        # 找到用户输入的位置并掩盖
        # 简化实现：实际需要更精确的定位
        user_start = tokenized.input_ids[0].tolist().index(
            self.tokenizer.encode("USER:")[0]
        )
        
        # 掩盖用户输入部分
        labels[:, :user_start] = -100
        
        return labels
    
    def load_image(self, path):
        """加载图像（简化）"""
        from PIL import Image
        return Image.open(path).convert("RGB")


class MultiModalTrainer(Trainer):
    """多模态模型训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失
        
        Args:
            model: 多模态模型
            inputs: 包含pixel_values, input_ids, labels的字典
        
        Returns:
            loss: 训练损失
        """
        outputs = model(
            images=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs["labels"]
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer, CLIPImageProcessor
    
    # 创建数据处理组件
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    collator = MultiModalDataCollator(tokenizer, image_processor)
    
    # 示例数据
    data = [
        MultiModalConversation(
            image_path="image1.jpg",
            conversations=[
                {"role": "user", "content": "描述这张图片。"},
                {"role": "assistant", "content": "这是一张展示日落的美丽照片。"}
            ]
        )
    ]
    
    # 处理数据
    batch = collator(data)
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}")
```

---

## 视觉问答（VQA）

### 任务定义

视觉问答要求模型根据图像内容回答自然语言问题：

$$
\text{Answer} = f(I, Q; \theta)
$$

其中 $I$ 是输入图像，$Q$ 是问题。

### 任务类型

```
┌─────────────────────────────────────────────────────────────┐
│                      VQA任务类型                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 开放式问答 (Open-ended)                                  │
│     Q: 图中有几个人？                                        │
│     A: 三个人                                               │
│                                                             │
│  2. 多选题 (Multiple Choice)                                 │
│     Q: 图中的天气如何？                                      │
│     A: A. 晴天  B. 雨天  C. 阴天  D. 雪天                    │
│                                                             │
│  3. 是非题 (Yes/No)                                          │
│     Q: 图中有人在骑车吗？                                    │
│     A: 是                                                   │
│                                                             │
│  4. 计数问题 (Counting)                                      │
│     Q: 图中有几只狗？                                        │
│     A: 4                                                    │
│                                                             │
│  5. 推理问题 (Reasoning)                                     │
│     Q: 为什么这个人打着伞？                                  │
│     A: 因为正在下雨                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 评估指标

**1. 准确率（Accuracy）**

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{pred}_i \in \text{answers}_i]
$$

**2. VQA Score**

考虑到人工标注的一致性：

$$
\text{VQA Score} = \min\left(\frac{\text{count}}{3}, 1\right)
$$

其中 `count` 是预测答案在10个人工标注中出现的次数。

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoModelForCausalLM

class VQAModel(nn.Module):
    """视觉问答模型"""
    
    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-base-patch32",
        llm: str = "gpt2",
        num_answers: int = 3129  # VQA v2标准答案数
    ):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = CLIPModel.from_pretrained(vision_encoder).vision_model
        
        # 文本编码器（用于问题）
        self.text_encoder = CLIPModel.from_pretrained(vision_encoder).text_model
        
        # 多模态融合层
        self.fusion = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_answers)
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Args:
            pixel_values: [batch, 3, H, W] 图像
            input_ids: [batch, seq_len] 问题token
            attention_mask: [batch, seq_len] 注意力掩码
        
        Returns:
            logits: [batch, num_answers] 答案logits
        """
        # 编码图像
        vision_output = self.vision_encoder(pixel_values)
        image_features = vision_output.last_hidden_state  # [batch, num_patches, dim]
        
        # 编码问题
        text_output = self.text_encoder(input_ids, attention_mask=attention_mask)
        question_features = text_output.last_hidden_state  # [batch, seq_len, dim]
        
        # 多模态融合：问题作为Query，图像作为Key和Value
        fused, _ = self.fusion(
            query=question_features,
            key=image_features,
            value=image_features
        )
        
        # 池化
        pooled = fused.mean(dim=1)  # [batch, dim]
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits


class GenerativeVQA(nn.Module):
    """生成式VQA模型"""
    
    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-large-patch14",
        llm: str = "lmsys/vicuna-7b-v1.5"
    ):
        super().__init__()
        
        # 视觉编码器（冻结）
        self.vision_encoder = CLIPModel.from_pretrained(vision_encoder).vision_model
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm)
        
        # 视觉-语言适配器
        vision_dim = 1024  # CLIP ViT-L
        llm_dim = self.llm.config.hidden_size
        
        self.adapter = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        生成式VQA训练
        """
        # 提取视觉特征
        with torch.no_grad():
            vision_output = self.vision_encoder(pixel_values)
            image_features = vision_output.last_hidden_state
        
        # 投影到LLM空间
        image_embeds = self.adapter(image_features)
        
        # 获取文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 构建输入嵌入：[图像tokens] + [问题tokens]
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 调整attention_mask
        image_mask = torch.ones(
            image_embeds.size(0), image_embeds.size(1),
            device=attention_mask.device
        )
        extended_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # 调整labels
        if labels is not None:
            image_labels = torch.full(
                (labels.size(0), image_embeds.size(1)),
                -100,  # 忽略图像部分的损失
                device=labels.device
            )
            labels = torch.cat([image_labels, labels], dim=1)
        
        # LLM前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        question: str,
        tokenizer,
        max_length: int = 50
    ):
        """生成答案"""
        # 编码问题
        inputs = tokenizer(question, return_tensors="pt").to(pixel_values.device)
        
        # 提取视觉特征
        with torch.no_grad():
            vision_output = self.vision_encoder(pixel_values)
            image_features = vision_output.last_hidden_state
        
        # 投影
        image_embeds = self.adapter(image_features)
        
        # 获取文本嵌入
        text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        
        # 拼接
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_length,
            do_sample=False
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 使用示例
if __name__ == "__main__":
    # 分类式VQA
    model = VQAModel(num_answers=3129)
    
    # 模拟输入
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 49408, (2, 20))  # CLIP词表大小
    
    # 前向传播
    logits = model(images, questions)
    print(f"Logits shape: {logits.shape}")  # [2, 3129]
    
    # 预测
    predictions = logits.argmax(dim=-1)
    print(f"Predictions: {predictions}")
```

---

## 图像描述（Image Captioning）

### 任务定义

图像描述任务要求模型生成描述图像内容的自然语言句子：

$$
\text{Caption} = \arg\max_{y} P(y | I; \theta) = \prod_{t=1}^{T} P(y_t | y_{<t}, I; \theta)
$$

### 评估指标

**1. BLEU**

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

其中 $p_n$ 是n-gram精确度，BP是短句惩罚。

**2. CIDEr**

专门为图像描述设计的指标，基于TF-IDF加权：

$$
\text{CIDEr} = \frac{1}{m} \sum_{j=1}^{m} \frac{\mathbf{g}^n(c) \cdot \mathbf{g}^n(S_j)}{\|\mathbf{g}^n(c)\| \|\mathbf{g}^n(S_j)\|}
$$

**3. METEOR**

考虑同义词匹配和词干匹配。

**4. ROUGE-L**

基于最长公共子序列。

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor

class ImageCaptioningModel(nn.Module):
    """基于Transformer的图像描述模型"""
    
    def __init__(
        self,
        vision_encoder: str = "google/vit-base-patch16-224",
        text_decoder: str = "gpt2",
        max_length: int = 50
    ):
        super().__init__()
        
        self.encoder_decoder = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_encoder, text_decoder
        )
        self.max_length = max_length
        
        # 配置解码器
        self.encoder_decoder.config.decoder_start_token_id = 50256  # GPT2 EOS
        self.encoder_decoder.config.eos_token_id = 50256
        self.encoder_decoder.config.pad_token_id = 50256
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        Args:
            pixel_values: [batch, 3, H, W] 图像
            labels: [batch, seq_len] 目标描述
        
        Returns:
            outputs: 模型输出
        """
        outputs = self.encoder_decoder(
            pixel_values=pixel_values,
            labels=labels
        )
        return outputs
    
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        tokenizer: GPT2Tokenizer
    ):
        """生成图像描述"""
        outputs = self.encoder_decoder.generate(
            pixel_values=pixel_values,
            max_length=self.max_length,
            num_beams=4,
            early_stopping=True
        )
        
        captions = []
        for output in outputs:
            caption = tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        
        return captions


class AttentiveCaptionModel(nn.Module):
    """带注意力机制的图像描述模型"""
    
    def __init__(
        self,
        encoder_dim: int = 2048,
        decoder_dim: int = 512,
        embed_dim: int = 300,
        vocab_size: int = 10000,
        attention_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # 视觉编码器（假设预训练的CNN）
        self.encoder_dim = encoder_dim
        
        # 注意力模块
        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        # 解码器
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def init_hidden_state(self, encoder_out):
        """初始化LSTM隐藏状态"""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Args:
            encoder_out: [batch, num_pixels, encoder_dim] 图像特征
            encoded_captions: [batch, max_seq_len] 编码的描述
            caption_lengths: 描述长度列表
        
        Returns:
            predictions: [batch, max_seq_len, vocab_size]
            alphas: [batch, max_seq_len, num_pixels] 注意力权重
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.fc.out_features
        
        # 排序按长度降序
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # 嵌入
        embeddings = self.embedding(encoded_captions)
        
        # 初始化
        h, c = self.init_hidden_state(encoder_out)
        
        # 解码
        max_len = max(caption_lengths)
        predictions = torch.zeros(batch_size, max_len, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)
        
        for t in range(max_len - 1):
            # 注意力
            context, alpha = self.attention(encoder_out, h)
            
            # 门控
            gate = torch.sigmoid(self.f_beta(h))
            context = gate * context
            
            # LSTM步骤
            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], context], dim=1),
                (h, c)
            )
            
            # 预测
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        
        return predictions, alphas, sort_ind


class Attention(nn.Module):
    """注意力模块"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: [batch, num_pixels, encoder_dim]
            decoder_hidden: [batch, decoder_dim]
        
        Returns:
            context: [batch, encoder_dim] 加权上下文向量
            alpha: [batch, num_pixels] 注意力权重
        """
        att1 = self.encoder_att(encoder_out)  # [batch, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # [batch, 1, attention_dim]
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # [batch, num_pixels]
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch, encoder_dim]
        
        return context, alpha


# 使用示例
if __name__ == "__main__":
    from transformers import ViTImageProcessor, GPT2Tokenizer
    
    # 简单版本
    model = ImageCaptioningModel()
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 模拟输入
    from PIL import Image
    # image = Image.open("test.jpg")
    # pixel_values = processor(images=image, return_tensors="pt").pixel_values
    # captions = model.generate_caption(pixel_values, tokenizer)
    # print(f"Generated caption: {captions[0]}")
```

---

## 视觉定位（Visual Grounding）

### 任务定义

视觉定位要求模型根据自然语言描述定位图像中的对应区域：

$$
\text{BBox} = f(I, \text{Query}; \theta)
$$

输出通常是一个边界框 $(x, y, w, h)$ 或分割掩码。

### 任务类型

```
┌─────────────────────────────────────────────────────────────┐
│                    视觉定位任务类型                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 指代表达定位 (Referring Expression Comprehension)        │
│     输入: 图像 + "穿红衣服的女孩"                             │
│     输出: 边界框 [x, y, w, h]                               │
│                                                             │
│  2. 指代表达分割 (Referring Expression Segmentation)         │
│     输入: 图像 + "前景中的猫"                                 │
│     输出: 分割掩码                                           │
│                                                             │
│  3. 短语定位 (Phrase Grounding)                              │
│     输入: 图像 + 描述文本                                     │
│     输出: 文本中每个名词短语的边界框                          │
│                                                             │
│  4. 视觉关系检测 (Visual Relation Detection)                 │
│     输入: 图像 + "猫坐在沙发上"                               │
│     输出: 主语(猫) + 关系(坐在...上) + 宾语(沙发)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 评估指标

**IoU（Intersection over Union）**：

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|}
$$

**Precision@k**：IoU超过阈值（通常0.5）的预测比例。

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class VisualGroundingModel(nn.Module):
    """视觉定位模型"""
    
    def __init__(
        self,
        vision_model: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        num_queries: int = 100
    ):
        super().__init__()
        
        # 视觉编码器
        clip = CLIPModel.from_pretrained(vision_model)
        self.vision_encoder = clip.vision_model
        self.text_encoder = clip.text_model
        
        vision_dim = clip.config.vision_config.hidden_size
        text_dim = clip.config.text_config.hidden_size
        
        # 特征融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 视觉和文本投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 边界框预测头
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 4)  # [x, y, w, h]
        )
        
        # 目标性分数
        self.confidence_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Args:
            pixel_values: [batch, 3, H, W]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            bbox: [batch, 4] 预测的边界框
            confidence: [batch, 1] 置信度
        """
        # 编码视觉特征
        vision_output = self.vision_encoder(pixel_values)
        image_features = vision_output.last_hidden_state  # [batch, num_patches, dim]
        
        # 编码文本特征
        text_output = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state  # [batch, seq_len, dim]
        
        # 投影
        image_proj = self.vision_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        # 交叉注意力：文本作为Query，图像作为Key和Value
        fused, attention_weights = self.cross_attention(
            query=text_proj,
            key=image_proj,
            value=image_proj
        )
        
        # 全局池化
        pooled = fused.mean(dim=1)  # [batch, hidden_dim]
        
        # 预测边界框
        bbox = self.bbox_head(pooled)
        bbox = torch.sigmoid(bbox)  # 归一化到[0, 1]
        
        # 置信度
        confidence = torch.sigmoid(self.confidence_head(pooled))
        
        return bbox, confidence, attention_weights


class GroundingDINOStyle(nn.Module):
    """类Grounding DINO架构"""
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_decoder_layers: int = 6,
        num_queries: int = 100
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # 可学习的查询
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 编码器投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 预测头
        self.class_head = nn.Linear(hidden_dim, 1)  # 二分类：目标/背景
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ):
        """
        Args:
            vision_features: [batch, num_patches, vision_dim]
            text_features: [batch, seq_len, text_dim]
        
        Returns:
            pred_boxes: [batch, num_queries, 4]
            pred_logits: [batch, num_queries, 1]
        """
        batch_size = vision_features.size(0)
        
        # 投影
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # 拼接文本和视觉作为记忆
        memory = torch.cat([text_proj, vision_proj], dim=1)
        
        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 解码
        decoder_output = self.decoder(query_embed, memory)
        
        # 预测
        pred_logits = self.class_head(decoder_output)
        pred_boxes = torch.sigmoid(self.bbox_head(decoder_output))
        
        return pred_boxes, pred_logits


def compute_iou(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """
    计算两组边界框的IoU
    
    Args:
        bbox1: [N, 4] (x1, y1, x2, y2)
        bbox2: [M, 4] (x1, y1, x2, y2)
    
    Returns:
        iou: [N, M] IoU矩阵
    """
    # 扩展维度以便广播
    bbox1 = bbox1.unsqueeze(1)  # [N, 1, 4]
    bbox2 = bbox2.unsqueeze(0)  # [1, M, 4]
    
    # 计算交集
    inter_x1 = torch.max(bbox1[..., 0], bbox2[..., 0])
    inter_y1 = torch.max(bbox1[..., 1], bbox2[..., 1])
    inter_x2 = torch.min(bbox1[..., 2], bbox2[..., 2])
    inter_y2 = torch.min(bbox1[..., 3], bbox2[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou


class GroundingLoss(nn.Module):
    """视觉定位损失函数"""
    
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_boxes: torch.Tensor
    ):
        """
        Args:
            pred_boxes: [batch, num_queries, 4]
            pred_logits: [batch, num_queries, 1]
            gt_boxes: [batch, 4]
        
        Returns:
            total_loss: 总损失
        """
        batch_size = pred_boxes.size(0)
        num_queries = pred_boxes.size(1)
        
        total_loss = 0
        
        for i in range(batch_size):
            # 计算所有查询与GT的IoU
            ious = compute_iou(pred_boxes[i], gt_boxes[i:i+1])  # [num_queries, 1]
            
            # 找到最佳匹配
            best_iou, best_idx = ious.max(dim=0)
            
            # 分类损失
            target_cls = torch.zeros(num_queries, 1, device=pred_logits.device)
            if best_iou > self.iou_threshold:
                target_cls[best_idx] = 1
            
            cls_loss = self.cls_loss(pred_logits[i], target_cls)
            
            # 边界框损失（只对正样本计算）
            if best_iou > self.iou_threshold:
                bbox_loss = self.bbox_loss(pred_boxes[i, best_idx], gt_boxes[i])
            else:
                bbox_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            total_loss += cls_loss + bbox_loss
        
        return total_loss / batch_size


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = VisualGroundingModel()
    
    # 模拟输入
    images = torch.randn(2, 3, 224, 224)
    queries = torch.randint(0, 49408, (2, 20))
    
    # 前向传播
    bbox, conf, attn = model(images, queries)
    print(f"Predicted bbox: {bbox.shape}")  # [2, 4]
    print(f"Confidence: {conf.shape}")  # [2, 1]
    
    # 计算IoU
    pred = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    gt = torch.tensor([[0.15, 0.25, 0.55, 0.65]])
    iou = compute_iou(pred, gt)
    print(f"IoU: {iou.item():.4f}")
```

---

## 知识点关联

```
┌─────────────────────────────────────────────────────────────┐
│                    视觉语言模型知识图谱                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      预训练基础                              │
│                          │                                  │
│              ┌───────────┼───────────┐                     │
│              │           │           │                      │
│              ▼           ▼           ▼                      │
│           ┌─────┐    ┌─────┐    ┌─────┐                   │
│           │CLIP │    │BLIP │    │其他 │                    │
│           │对比学│    │引导式│    │...  │                    │
│           │习   │    │预训练│    │     │                    │
│           └──┬──┘    └──┬──┘    └──┬──┘                   │
│              │           │           │                      │
│              └───────────┼───────────┘                     │
│                          │                                  │
│                          ▼                                  │
│                   视觉编码器                                │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────┴───────────┐                     │
│              │                       │                      │
│              ▼                       ▼                      │
│         适配器层                  Q-Former                  │
│         (LLaVA)                  (BLIP-2)                   │
│              │                       │                      │
│              └───────────┬───────────┘                     │
│                          │                                  │
│                          ▼                                  │
│                    LLM融合                                 │
│                          │                                  │
│              ┌───────────┼───────────┐                     │
│              │           │           │                      │
│              ▼           ▼           ▼                      │
│         ┌───────┐   ┌───────┐   ┌───────┐                 │
│         │  VQA  │   │Caption│   │Ground │                 │
│         │ 视觉问答│   │图像描述│   │视觉定位│                │
│         └───────┘   └───────┘   └───────┘                 │
│              │           │           │                      │
│              └───────────┼───────────┘                     │
│                          │                                  │
│                          ▼                                  │
│                   指令微调                                 │
│                          │                                  │
│                          ▼                                  │
│                   多模态Agent                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键技术依赖关系

| 技术 | 依赖 | 应用 |
|------|------|------|
| LLaVA | CLIP视觉编码器 + LLM | 多模态对话 |
| BLIP-2 | 冻结视觉编码器 + Q-Former + 冻结LLM | 理解+生成 |
| VQA | 视觉编码 + 多模态融合 + 分类/生成 | 视觉问答 |
| Captioning | 视觉编码 + 自回归解码 | 图像描述 |
| Grounding | 视觉编码 + 文本编码 + 检测头 | 区域定位 |

---

## 核心考点

### 📌 概念理解

1. **CLIP的对比学习原理**
   - 为什么对比学习能实现零样本分类？
   - 温度参数 $\tau$ 的作用是什么？

2. **Q-Former的设计动机**
   - 为什么需要Q-Former？
   - 可学习查询向量的作用是什么？

3. **多模态融合策略**
   - 早期融合 vs 晚期融合
   - 交叉注意力 vs 拼接融合

### 📌 技术实现

4. **损失函数设计**
   - 对比损失的计算方式
   - 多模态任务的联合训练

5. **训练策略**
   - 冻结策略：哪些参数冻结，为什么？
   - 指令微调数据的构建方法

6. **评估指标**
   - VQA Score的计算
   - IoU的定义和应用

### 📌 实践应用

7. **模型选择**
   - 不同任务适合的模型架构
   - 计算资源与模型规模的权衡

8. **数据构建**
   - 如何构建高质量的指令数据？
   - 数据增强策略

---

## 学习建议

### 🎯 推荐学习路径

```
阶段一：基础理解（1-2周）
├── 学习CLIP原理和代码实现
├── 理解对比学习的数学基础
└── 实践零样本分类任务

阶段二：架构深入（2-3周）
├── 学习BLIP/BLIP-2架构
├── 理解Q-Former的设计
└── 实践图文检索任务

阶段三：LLM融合（2-3周）
├── 学习LLaVA架构
├── 理解视觉-语言适配
└── 实践多模态对话

阶段四：下游任务（2-3周）
├── VQA、Captioning、Grounding
├── 指令微调实践
└── 自定义任务开发
```

### 📚 推荐资源

**论文**：
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- BLIP: Bootstrapping Language-Image Pre-training
- LLaVA: Visual Instruction Tuning
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

**代码库**：
- HuggingFace Transformers
- LAVIS (Salesforce)
- LLaVA Official

**数据集**：
- COCO Captions
- VQA v2
- RefCOCO/RefCOCO+/RefCOCOg
- LLaVA-Instruct-150K

### ⚠️ 常见问题

1. **视觉特征维度不匹配**
   - 解决：使用投影层或适配器

2. **训练不稳定**
   - 解决：调整学习率、使用梯度裁剪

3. **推理速度慢**
   - 解决：使用量化、知识蒸馏

4. **长序列处理**
   - 解决：图像特征压缩、稀疏注意力

---

## 参考资料

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
2. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML 2022.
3. Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
4. Liu, H., et al. "Visual Instruction Tuning." NeurIPS 2023.
