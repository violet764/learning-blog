# 重排序（Reranking）详解

重排序（Reranking）是 RAG 系统中提升检索质量的关键技术。它通过在初始检索后对候选文档进行更精细的相关性评分，显著提升最终检索结果的质量。本文将详细介绍重排序的原理、方法和实践应用。

## 基本概念

### 什么是重排序？

重排序是一种**两阶段检索策略**：

1. **第一阶段（粗检索）**：使用快速但相对粗糙的方法（如 BM25、向量检索）从大规模文档库中召回候选文档
2. **第二阶段（精排序）**：使用更精确但计算开销更大的模型对候选文档重新打分排序

```
┌───────────────────────────────────────────────────────────┐
│                      两阶段检索流程                        │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   查询 Query                                              │
│      │                                                    │
│      ▼                                                    │
│   ┌─────────────────┐                                     │
│   │  第一阶段：粗检索 │  ← 快速召回（BM25/向量检索）       │
│   │   召回 Top-K    │     K 通常为 50-1000               │
│   └────────┬────────┘                                     │
│            │                                              │
│            ▼                                              │
│   ┌─────────────────┐                                     │
│   │ 第二阶段：重排序 │  ← 精确评分（Cross-Encoder）        │
│   │   输出 Top-N    │     N 通常为 5-20                  │
│   └────────┬────────┘                                     │
│            │                                              │
│            ▼                                              │
│        最终结果                                          │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 为什么需要重排序？

| 问题 | 描述 | 重排序的解决方案 |
|------|------|------------------|
| 召回精度有限 | 粗检索方法无法精确判断语义相关性 | 使用更强的模型精细评分 |
| 向量压缩损失 | 向量检索压缩了语义信息 | Cross-Encoder 保留完整语义 |
| 计算效率权衡 | 精确模型太慢，无法处理全量数据 | 只对候选文档精确评分 |
| 领域适应需求 | 通用模型在特定领域效果不佳 | 可针对领域微调重排序模型 |

---

## 重排序模型分类

### Cross-Encoder vs Bi-Encoder

这是重排序中最核心的概念区分：

```
┌─────────────────────────────────────────────────────────────┐
│                    Bi-Encoder (双塔编码器)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Query ──→ [Encoder] ──→ 向量 Q                          │
│                                              ↘              │
│                                               点积/余弦     │
│                                              ↙              │
│    Document ──→ [Encoder] ──→ 向量 D                       │
│                                                             │
│    特点：Query 和 Document 独立编码，可预计算文档向量        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Cross-Encoder (交叉编码器)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    [SEP] Query [SEP] Document [SEP]                         │
│              │                                              │
│              ▼                                              │
│         [Encoder]                                           │
│              │                                              │
│              ▼                                              │
│          相关性分数                                         │
│                                                             │
│    特点：Query 和 Document 一起输入，建模深层交互            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 详细对比

| 特性 | Bi-Encoder | Cross-Encoder |
|------|------------|---------------|
| 编码方式 | 独立编码 | 联合编码 |
| 计算速度 | 快（向量预计算） | 慢（每对都要编码） |
| 语义交互 | 无深层交互 | 全注意力交互 |
| 适用阶段 | 粗检索（召回） | 精排序（重排序） |
| 典型模型 | Sentence-BERT, DPR | MonoBERT, BGE-Reranker |

### 重排序模型架构类型

#### 1. 基于 Cross-Encoder 的重排序

最主流的重排序方法，将 Query 和 Document 拼接后输入模型：

```python
# Cross-Encoder 重排序示例
from sentence_transformers import CrossEncoder

# 加载预训练的重排序模型
model = CrossEncoder('BAAI/bge-reranker-base')

def rerank(query: str, documents: list, top_k: int = 5):
    """
    使用 Cross-Encoder 进行重排序
    
    Args:
        query: 用户查询
        documents: 候选文档列表
        top_k: 返回前 k 个结果
    
    Returns:
        重排序后的文档列表
    """
    # 构造 query-document 对
    pairs = [(query, doc) for doc in documents]
    
    # 批量计算相关性分数
    scores = model.predict(pairs)
    
    # 按分数排序
    ranked_results = sorted(
        zip(documents, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return ranked_results[:top_k]

# 使用示例
query = "什么是深度学习？"
candidates = [
    "深度学习是机器学习的一个分支，使用神经网络进行特征学习。",
    "学习深度的概念在心理学中指的是认知的深度。",
    "机器学习包括监督学习、无监督学习和强化学习。",
    "深度神经网络在图像识别领域取得了突破性进展。"
]

results = rerank(query, candidates, top_k=3)
for doc, score in results:
    print(f"分数: {score:.4f}")
    print(f"文档: {doc}\n")
```

#### 2. ColBERT 风格的 Late Interaction

ColBERT（Contextualized Late Interaction over BERT）提出了一种折中方案：

```
┌─────────────────────────────────────────────────────────────┐
│                    ColBERT Late Interaction                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Query: [q1, q2, q3]                                       │
│            │                                                │
│            ▼                                                │
│   Query Vectors: [v_q1, v_q2, v_q3]                        │
│                          │                                  │
│                          ▼ MaxSim 操作                      │
│   Doc Vectors: [v_d1, v_d2, v_d3, v_d4, ...]               │
│                          │                                  │
│                          ▼                                  │
│   Score = Σ max(sim(q_i, d_j))                             │
│                                                             │
│   特点：保留 Token 级别的向量，计算细粒度相似度              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# ColBERT 风格重排序的简化实现
import torch
import torch.nn.functional as F

def colbert_score(query_embeddings, doc_embeddings):
    """
    计算 ColBERT 风格的相关性分数
    
    Args:
        query_embeddings: [num_query_tokens, hidden_dim]
        doc_embeddings: [num_doc_tokens, hidden_dim]
    
    Returns:
        相关性分数
    """
    # 计算所有 query-doc token 对的相似度
    similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T)
    
    # 对每个 query token，取最大相似度（MaxSim）
    max_sim_per_query, _ = similarity_matrix.max(dim=1)
    
    # 求和作为最终分数
    score = max_sim_per_query.sum()
    
    return score
```

---

## 主流重排序模型

### 开源模型一览

| 模型 | 开发者 | 特点 | 参数量 |
|------|--------|------|--------|
| BGE-Reranker | BAAI | 中文效果好，多尺寸可选 | 100M-560M |
| MonoBERT | Google | 经典 Cross-Encoder | 110M |
| MonoT5 | Google | 使用 T5 生成式模型 | 220M-770M |
| RankT5 | Google | T5 架构，端到端训练 | 220M-770M |
| Cohere Rerank | Cohere | 商业 API，多语言支持 | - |
| Jina Reranker | Jina AI | 轻量级，速度快 | 35M-660M |

### BGE-Reranker 使用示例

BGE-Reranker 是目前中文场景最推荐的开源重排序模型：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class BGERReranker:
    """BGE 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化 BGE 重排序器
        
        Args:
            model_name: 模型名称，可选:
                - BAAI/bge-reranker-base (109M 参数)
                - BAAI/bge-reranker-large (330M 参数)
                - BAAI/bge-reranker-v2-m3 (560M 参数，多语言)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def compute_scores(self, query: str, documents: list) -> list:
        """
        计算 query 与多个文档的相关性分数
        
        Args:
            query: 用户查询
            documents: 文档列表
        
        Returns:
            分数列表（与文档顺序对应）
        """
        # 构造输入对
        pairs = [[query, doc] for doc in documents]
        
        # Tokenization
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        
        return scores.cpu().numpy().tolist()
    
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """
        重排序
        
        Returns:
            [(文档索引, 分数, 文档内容), ...]
        """
        scores = self.compute_scores(query, documents)
        
        # 排序
        ranked = sorted(
            enumerate(scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [(idx, score, documents[idx]) for idx, score in ranked]


# 使用示例
if __name__ == "__main__":
    reranker = BGERReranker("BAAI/bge-reranker-base")
    
    query = "如何提高大模型的推理能力？"
    candidates = [
        "通过思维链(CoT)提示可以显著提高大模型的推理能力，让模型逐步思考。",
        "大模型的参数量越大，推理能力越强，但计算成本也越高。",
        "思维链是一种提示技术，通过引导模型进行中间推理步骤来提升效果。",
        "强化学习可以用于优化大模型的行为，使其更好地对齐人类意图。",
        "知识蒸馏是一种模型压缩技术，可以将大模型的知识迁移到小模型中。"
    ]
    
    results = reranker.rerank(query, candidates, top_k=3)
    
    print(f"查询: {query}\n")
    print("重排序结果:")
    for rank, (idx, score, doc) in enumerate(results, 1):
        print(f"\n第{rank}名 (分数: {score:.4f}):")
        print(f"  {doc[:100]}...")
```

---

## 重排序在 RAG 中的应用

### 完整的 RAG 检索流程

```
┌───────────────────────────────────────────────────────────────┐
│                      RAG 检索增强生成流程                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  用户查询: "什么是 Transformer 的注意力机制？"                 │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────┐         │
│  │              第一阶段：多路召回                   │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │         │
│  │  │ BM25    │  │ 向量检索 │  │ 关键词   │      │         │
│  │  │ 稀疏检索 │  │ 稠密检索 │  │ 匹配     │      │         │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘      │         │
│  │       │             │             │            │         │
│  │       └─────────────┼─────────────┘            │         │
│  │                     ▼                          │         │
│  │              合并去重 (约 50-100 篇)            │         │
│  └─────────────────────────────────────────────────┘         │
│                        │                                      │
│                        ▼                                      │
│  ┌─────────────────────────────────────────────────┐         │
│  │              第二阶段：重排序                     │         │
│  │                                                  │         │
│  │    Query + Candidate Docs → Reranker → Top-5    │         │
│  │                                                  │         │
│  └─────────────────────────────────────────────────┘         │
│                        │                                      │
│                        ▼                                      │
│  ┌─────────────────────────────────────────────────┐         │
│  │              第三阶段：生成回答                   │         │
│  │                                                  │         │
│  │    LLM(Query + Top-5 Docs) → 最终答案           │         │
│  │                                                  │         │
│  └─────────────────────────────────────────────────┘         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### LlamaIndex 中的重排序集成

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

# 方式 1: 使用 Cohere API
def build_rag_with_cohere_rerank():
    """使用 Cohere 重排序的 RAG"""
    
    # 加载文档
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 构建索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 创建重排序器
    cohere_rerank = CohereRerank(
        api_key="your-cohere-api-key",
        top_n=5  # 重排序后保留的文档数
    )
    
    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=20,  # 第一阶段召回数量
        node_postprocessors=[cohere_rerank]
    )
    
    return query_engine


# 方式 2: 使用本地模型
def build_rag_with_local_rerank():
    """使用本地模型重排序的 RAG"""
    
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # 使用 HuggingFace 模型
    rerank = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=20,
        node_postprocessors=[rerank]
    )
    
    return query_engine
```

### LangChain 中的重排序集成

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_rag_with_reranking():
    """带重排序的 LangChain RAG"""
    
    # 创建向量存储
    vectorstore = FAISS.from_texts(
        texts=your_texts,
        embedding=OpenAIEmbeddings()
    )
    
    # 基础检索器
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}  # 召回 20 篇
    )
    
    # 重排序模型
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(
        model=model,
        top_n=5  # 重排序后保留 5 篇
    )
    
    # 组合检索器
    retrieval_chain = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return retrieval_chain
```

---

## 重排序策略优化

### 1. 批量处理优化

重排序计算量大，需要合理使用批处理：

```python
class BatchReranker:
    """支持批量处理的高效重排序器"""
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def rerank_batch(
        self, 
        queries: list, 
        documents_list: list,
        top_k: int = 5
    ) -> list:
        """
        批量重排序多个查询
        
        Args:
            queries: 查询列表
            documents_list: 每个查询对应的候选文档列表
            top_k: 每个查询返回的文档数
        
        Returns:
            每个查询的重排序结果列表
        """
        all_results = []
        
        for query, documents in zip(queries, documents_list):
            scores = []
            
            # 分批处理文档
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_scores = self._compute_batch_scores(query, batch_docs)
                scores.extend(batch_scores)
            
            # 排序
            ranked = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            all_results.append([
                (idx, score, documents[idx]) 
                for idx, score in ranked
            ])
        
        return all_results
    
    def _compute_batch_scores(self, query: str, documents: list) -> list:
        """计算一批文档的分数"""
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        
        return scores.cpu().numpy().tolist()
```

### 2. 分数融合策略

当使用多路召回时，需要融合不同来源的分数：

```python
import numpy as np

def reciprocal_rank_fusion(
    results_list: list, 
    k: int = 60
) -> list:
    """
    Reciprocal Rank Fusion (RRF) 分数融合
    
    Args:
        results_list: 多个检索结果列表，每个元素是 [(doc_id, score), ...]
        k: RRF 参数
    
    Returns:
        融合后的排序结果
    """
    fused_scores = {}
    
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    
    # 排序
    ranked = sorted(
        fused_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return ranked


def weighted_score_fusion(
    results_list: list, 
    weights: list
) -> list:
    """
    加权分数融合
    
    Args:
        results_list: 多个检索结果列表
        weights: 各检索器的权重
    
    Returns:
        融合后的排序结果
    """
    doc_scores = {}
    
    for results, weight in zip(results_list, weights):
        # 分数归一化
        scores = [score for _, score in results]
        if scores:
            min_s, max_s = min(scores), max(scores)
            normalized = [(doc_id, (score - min_s) / (max_s - min_s + 1e-6)) 
                         for doc_id, score in results]
        else:
            normalized = results
        
        for doc_id, score in normalized:
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += weight * score
    
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked
```

### 3. 动态 Top-K 选择

根据查询复杂度动态调整召回和重排序数量：

```python
def adaptive_retrieval(
    query: str,
    retriever,
    reranker,
    complexity_estimator=None
) -> list:
    """
    自适应检索：根据查询复杂度调整策略
    """
    # 评估查询复杂度
    if complexity_estimator:
        complexity = complexity_estimator(query)
    else:
        # 简单启发式：根据查询长度判断
        complexity = len(query.split()) / 10  # 归一化
    
    # 根据复杂度调整参数
    if complexity < 0.3:  # 简单查询
        top_k_recall = 10
        top_k_rerank = 3
    elif complexity < 0.7:  # 中等查询
        top_k_recall = 30
        top_k_rerank = 5
    else:  # 复杂查询
        top_k_recall = 50
        top_k_rerank = 10
    
    # 第一阶段召回
    candidates = retriever.retrieve(query, top_k=top_k_recall)
    
    # 第二阶段重排序
    results = reranker.rerank(query, candidates, top_k=top_k_rerank)
    
    return results
```

---

## 性能评估

### 常用评估指标

| 指标 | 全称 | 含义 |
|------|------|------|
| MRR | Mean Reciprocal Rank | 第一个相关文档排名的倒数均值 |
| NDCG | Normalized Discounted Cumulative Gain | 考虑位置权重的排序质量 |
| MAP | Mean Average Precision | 平均精度均值 |
| Hit Rate | Hit Rate@K | 前 K 结果中包含相关文档的比例 |

```python
def evaluate_reranker(
    queries: list,
    ground_truths: list,
    retriever,
    reranker,
    k_values: list = [1, 5, 10]
) -> dict:
    """
    评估重排序效果
    
    Args:
        queries: 查询列表
        ground_truths: 每个查询的相关文档 ID 列表
        retriever: 基础检索器
        reranker: 重排序器
        k_values: 评估的 K 值
    
    Returns:
        评估指标字典
    """
    metrics = {f"Hit@{k}": [] for k in k_values}
    metrics["MRR"] = []
    
    for query, relevant_docs in zip(queries, ground_truths):
        # 召回
        candidates = retriever.retrieve(query, top_k=50)
        
        # 重排序
        ranked = reranker.rerank(query, candidates, top_k=max(k_values))
        ranked_doc_ids = [doc_id for doc_id, _, _ in ranked]
        
        # 计算指标
        # Hit Rate
        for k in k_values:
            hit = int(any(doc_id in relevant_docs for doc_id in ranked_doc_ids[:k]))
            metrics[f"Hit@{k}"].append(hit)
        
        # MRR
        mrr = 0
        for rank, doc_id in enumerate(ranked_doc_ids, 1):
            if doc_id in relevant_docs:
                mrr = 1 / rank
                break
        metrics["MRR"].append(mrr)
    
    # 计算均值
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 实践建议

### 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 中文通用 | BGE-Reranker-base | 平衡效果和速度 |
| 中文高精度 | BGE-Reranker-large | 更好的效果，稍慢 |
| 多语言 | BGE-Reranker-v2-m3 | 支持 100+ 语言 |
| 英文通用 | MonoBERT / MonoT5 | 成熟稳定 |
| 资源受限 | Jina-Reranker-tiny | 极轻量，速度快 |
| 商业生产 | Cohere Rerank | API 调用，无需维护 |

### 调参建议

1. **召回数量**：通常设置为重排序输出的 5-10 倍
2. **重排序输出**：通常 5-10 篇文档足够
3. **批大小**：根据 GPU 内存调整，一般 16-32
4. **序列长度**：建议 512，长文档可适当增加

### 常见问题与解决

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 重排序后效果变差 | 召回文档质量差 | 提高召回数量或质量 |
| 延迟过高 | 批处理效率低 | 增大 batch_size，使用 GPU |
| 内存溢出 | 序列过长 | 减小 batch_size 或截断长度 |
| 领域效果差 | 领域适配不足 | 使用领域数据微调 |

---

## 总结

### 关键要点

1. **两阶段检索**：粗检索 + 重排序是平衡效率与效果的最佳实践
2. **Cross-Encoder**：重排序的核心技术，通过联合编码捕获深层语义交互
3. **多路召回 + 重排序**：结合稀疏检索和稠密检索的优势
4. **参数调优**：根据场景调整召回数量、重排序数量和模型选择

### 最佳实践

- ✅ 召回数量至少是最终输出的 5 倍
- ✅ 使用 GPU 加速重排序
- ✅ 实施批处理提高吞吐量
- ✅ 根据评估结果迭代优化
- ✅ 考虑领域适配微调

---

## 参考资料

- [Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)
- [BGE Reranker Models](https://huggingface.co/BAAI/bge-reranker-base)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
