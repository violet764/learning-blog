# LlamaIndex 检索策略

检索是 RAG 系统的核心环节，检索质量直接决定了生成答案的质量。本文将介绍 LlamaIndex 提供的各种检索策略及其优化方法。

---

## 📌 检索基础

### 检索流程概述

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG 检索流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   用户查询                                                              │
│     │                                                                   │
│     ↓                                                                   │
│   ┌─────────────┐                                                      │
│   │ 查询预处理   │  重写、扩展、意图识别                                 │
│   └─────────────┘                                                      │
│     │                                                                   │
│     ↓                                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│   │ 向量检索    │ 或  │ 关键词检索   │ 或  │  混合检索    │            │
│   └─────────────┘     └─────────────┘     └─────────────┘            │
│     │                       │                   │                     │
│     └───────────────────────┼───────────────────┘                     │
│                             ↓                                          │
│                       ┌─────────────┐                                  │
│                       │   重排序     │  Reranker 提升精度               │
│                       └─────────────┘                                  │
│                             ↓                                          │
│                       ┌─────────────┐                                  │
│                       │ Top-K 节点   │                                  │
│                       └─────────────┘                                  │
│                             ↓                                          │
│                       ┌─────────────┐                                  │
│                       │ LLM 生成    │                                  │
│                       └─────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 基本检索器

```python
from llama_index.core import VectorStoreIndex

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 获取检索器
retriever = index.as_retriever(
    similarity_top_k=5  # 返回 top 5 相关节点
)

# 执行检索
nodes = retriever.retrieve("什么是深度学习？")

for node in nodes:
    print(f"分数: {node.score:.4f}")
    print(f"内容: {node.text[:100]}...")
```

---

## 🔍 检索器类型

### VectorIndexRetriever

基于向量相似度的检索器，最常用：

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 创建向量检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    # 可选：向量数据库特定参数
    vector_store_query_mode="default"  # default, sparse, hybrid
)

# 执行检索
nodes = retriever.retrieve("查询内容")
```

### KeywordTableRetriever

基于关键词的检索器：

```python
from llama_index.core.retrievers import KeywordTableRetriever
from llama_index.core import KeywordTableIndex

# 需要关键词索引
index = KeywordTableIndex.from_documents(documents)

# 创建关键词检索器
retriever = KeywordTableRetriever(
    index=index,
    similarity_top_k=5
)
```

### SummaryIndexRetriever

遍历所有节点的检索器：

```python
from llama_index.core.retrievers import SummaryIndexRetriever
from llama_index.core import SummaryIndex

index = SummaryIndex.from_documents(documents)

# 遍历所有节点
retriever = SummaryIndexRetriever(index=index)
```

### KGTableRetriever

知识图谱检索器：

```python
from llama_index.core.retrievers import KGTableRetriever
from llama_index.core import KnowledgeGraphIndex

index = KnowledgeGraphIndex.from_documents(documents)

# 知识图谱检索
retriever = KGTableRetriever(
    index=index,
    similarity_top_k=5
)
```

---

## 🔄 混合检索

混合检索结合多种检索方式，提升召回质量。

### 向量 + 关键词混合

```python
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableRetriever
from llama_index.core.retrievers import QueryFusionRetriever

# 创建向量检索器
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=5
)

# 创建关键词检索器
keyword_retriever = KeywordTableRetriever(
    index=keyword_index,
    similarity_top_k=5
)

# 混合检索器
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    similarity_top_k=5,
    mode="reciprocal_rerank"  # 融合模式
)

# 执行混合检索
nodes = fusion_retriever.retrieve("查询内容")
```

### 融合模式

| 模式 | 说明 |
|------|------|
| `reciprocal_rerank` | 倒数排名融合（推荐） |
| `simple` | 简单合并去重 |
| `distillation` | 知识蒸馏融合 |

```python
# 倒数排名融合公式
# score = sum(1 / (k + rank_i)) for each retriever
# k 通常设为 60

fusion_retriever = QueryFusionRetriever(
    retrievers=[retriever1, retriever2],
    similarity_top_k=10,
    mode="reciprocal_rerank",
    num_queries=1  # 查询扩展数量
)
```

### QueryFusionRetriever 使用示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai import OpenAI

# 加载文档并创建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建基础检索器
base_retriever = index.as_retriever(similarity_top_k=5)

# 创建融合检索器（自动生成多个查询变体）
fusion_retriever = QueryFusionRetriever(
    retrievers=[base_retriever],
    similarity_top_k=5,
    num_queries=3,  # 生成 3 个查询变体
    llm=OpenAI(model="gpt-4o-mini"),
    mode="reciprocal_rerank"
)

# 执行检索
nodes = fusion_retriever.retrieve("机器学习模型如何优化？")
```

---

## 📈 重排序（Reranking）

重排序是对检索结果进行二次精排，显著提升检索精度。

### 为什么需要重排序？

```
┌────────────────────────────────────────────────────────────────────┐
│                      检索 vs 重排序                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  向量检索 (粗排)              重排序 (精排)                         │
│  ┌────────────────┐          ┌────────────────┐                   │
│  │ 快速、高召回    │    →     │ 精确、高精度    │                   │
│  │ Top-100 候选   │          │ Top-5 精选     │                   │
│  │ 向量相似度      │          │ 语义相关性      │                   │
│  └────────────────┘          └────────────────┘                   │
│                                                                    │
│  问题: 向量空间中的"近"        解决: 用更强模型重新评估              │
│        不等于语义相关               查询-文档相关性                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 使用 Cohere Reranker

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import VectorStoreIndex

# 创建索引和查询引擎
index = VectorStoreIndex.from_documents(documents)

# 添加 Cohere 重排序
cohere_rerank = CohereRerank(
    api_key="your-cohere-api-key",
    top_n=5  # 重排序后保留的节点数
)

query_engine = index.as_query_engine(
    similarity_top_k=10,  # 先检索 10 个
    node_postprocessors=[cohere_rerank]  # 重排序保留 5 个
)

response = query_engine.query("查询内容")
```

### 使用 BGE Reranker（本地）

```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# 本地 BGE 重排序模型
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=5
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)
```

### 使用 FlashRank Reranker

```python
from llama_index.postprocessor.flashrank_rerank import FlashRankReranker

# 轻量级重排序
reranker = FlashRankReranker(
    model="ms-marco-MiniLM-L-12-v2",
    top_n=5
)

query_engine = index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[reranker]
)
```

### 使用 Jina Reranker

```python
from llama_index.postprocessor.jinaai_rerank import JinaRerank

reranker = JinaRerank(
    api_key="your-jina-api-key",
    top_n=5
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker]
)
```

### Reranker 对比

| Reranker | 特点 | 延迟 | 精度 |
|----------|------|------|------|
| Cohere | 云端服务，多语言支持 | 低 | 高 |
| BGE | 本地部署，中文友好 | 中 | 高 |
| FlashRank | 轻量级，速度快 | 低 | 中 |
| Jina | 云端服务，多语言 | 低 | 高 |
| Cross-Encoder | 本地，可自定义 | 高 | 最高 |

---

## 🔧 高级检索技术

### 自动检索（Auto-Retrieval）

自动从文档元数据中提取过滤条件：

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# 定义元数据字段
vector_store_info = VectorStoreInfo(
    content_info="技术文档",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="string",
            description="文档类别：技术、产品、运营"
        ),
        MetadataInfo(
            name="date",
            type="string",
            description="文档发布日期"
        )
    ]
)

# 创建自动检索器
from llama_index.core.retrievers import VectorIndexAutoRetriever

auto_retriever = VectorIndexAutoRetriever(
    index=index,
    vector_store_info=vector_store_info,
    similarity_top_k=5
)

# 自动推断过滤条件
# 查询 "2024年的技术文档" 会自动添加 metadata 过滤
nodes = auto_retriever.retrieve("2024年发布的技术文档")
```

### 递归检索（Recursive Retrieval）

通过层级关系进行递归检索：

```python
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.node_parser import HierarchicalNodeParser

# 层级解析：创建父子节点关系
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents(documents)

# 获取叶子节点
leaf_nodes = node_parser.get_leaf_nodes(nodes)

# 创建递归检索器
vector_index = VectorStoreIndex(leaf_nodes)
retriever = vector_index.as_retriever(similarity_top_k=5)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": retriever},
    node_dict={n.node_id: n for n in nodes}  # 所有节点的映射
)

# 检索时会自动返回父节点获取更多上下文
```

### 元数据过滤

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# 创建过滤条件
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="技术"),
        MetadataFilter(key="year", value=2024, operator=">=")
    ],
    condition="and"  # and 或 or
)

# 带过滤的检索
retriever = index.as_retriever(
    similarity_top_k=5,
    filters=filters
)

nodes = retriever.retrieve("查询内容")
```

### 查询重写

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# HyDE (假设性文档嵌入) 查询重写
hyde = HyDEQueryTransform(include_original=True)

query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(
    query_engine=query_engine,
    query_transform=hyde
)

# HyDE 会先生成一个假设性回答
# 然后用这个假设回答来检索
response = query_engine.query("查询内容")
```

### 多查询检索

```python
from llama_index.core.retrievers import AutoMultiQueryRetriever

# 自动生成多个查询变体
multi_query_retriever = AutoMultiQueryRetriever(
    retriever=index.as_retriever(similarity_top_k=5),
    llm=Settings.llm
)

# 内部会生成多个语义相近的查询
nodes = multi_query_retriever.retrieve("如何优化机器学习模型？")
```

---

## 📊 检索评估

### 使用 Ragas 评估

```python
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

# 准备评估数据
data = {
    "question": ["问题1", "问题2"],
    "contexts": [["检索到的上下文1"], ["检索到的上下文2"]],
    "ground_truth": ["真实答案1", "真实答案2"]
}

dataset = Dataset.from_dict(data)

# 评估
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall]
)

print(results)
```

### 自定义评估函数

```python
def evaluate_retrieval(query, relevant_doc_ids, retrieved_nodes):
    """简单的检索评估"""
    retrieved_ids = [node.node_id for node in retrieved_nodes]
    
    # 计算指标
    hits = len(set(retrieved_ids) & set(relevant_doc_ids))
    precision = hits / len(retrieved_ids) if retrieved_ids else 0
    recall = hits / len(relevant_doc_ids) if relevant_doc_ids else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "hits": hits
    }
```

---

## 💡 最佳实践

### 检索参数调优

```python
# 1. 调整 top_k
# 小数据集: top_k=3-5
# 大数据集: top_k=10-20（配合重排序）

# 2. 使用混合检索
# 向量检索 + 关键词检索 + 重排序

# 3. 根据数据特点选择
# 长文档: 增大 chunk_size
# 多主题: 使用元数据过滤
# 专业术语: 考虑关键词检索
```

### 完整检索 Pipeline

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# 1. 加载数据
documents = SimpleDirectoryReader("./data").load_data()

# 2. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 创建检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=15  # 粗检索
)

# 4. 创建重排序器
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=5  # 精检索
)

# 5. 组装查询引擎
query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[reranker]
)

# 6. 执行查询
response = query_engine.query("你的问题")
```

---

## 📚 参考资料

- [LlamaIndex 检索器文档](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)
- [重排序模块](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
- [混合检索示例](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/)

---

*下一章：[RAG 应用构建](./llamaindex-rag.md) →*
