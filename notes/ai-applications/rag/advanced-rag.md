# 高级 RAG 技术

随着 RAG 技术的发展，基础的检索-生成模式已经不能满足复杂场景的需求。本章将介绍一系列高级 RAG 技术，包括混合检索、重排序、多跳检索、GraphRAG 等，帮助你构建更强大的 RAG 系统。

---

## 🎯 高级 RAG 技术概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     高级 RAG 技术全景图                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    检索前优化 (Pre-Retrieval)                │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│   │  │ 查询改写  │  │ 查询扩展  │  │ 假设文档  │                  │  │
│   │  │ Query    │  │ Query    │  │ HyDE     │                  │  │
│   │  │ Rewrite  │  │ Expand   │  │          │                  │  │
│   │  └──────────┘  └──────────┘  └──────────┘                  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    检索优化 (Retrieval)                      │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│   │  │ 混合检索  │  │ 多向量检索│  │ 路由检索  │                  │  │
│   │  │ Hybrid   │  │ Multi    │  │ Routing  │                  │  │
│   │  │ Search   │  │ Vector   │  │          │                  │  │
│   │  └──────────┘  └──────────┘  └──────────┘                  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    检索后优化 (Post-Retrieval)               │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│   │  │ 重排序    │  │ 上下文压缩│  │ 上下文选择│                  │  │
│   │  │ Rerank   │  │ Compress │  │ Selection│                  │  │
│   │  └──────────┘  └──────────┘  └──────────┘                  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    高级架构 (Advanced)                       │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│   │  │ 多跳检索  │  │ GraphRAG │  │ 自适应RAG │                  │  │
│   │  │ Multi-   │  │ 知识图谱  │  │ Adaptive │                  │  │
│   │  │ Hop      │  │          │  │          │                  │  │
│   │  └──────────┘  └──────────┘  └──────────┘                  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 混合检索（Hybrid Retrieval）

### 为什么需要混合检索

单一的检索方法都有局限：

| 方法 | 优势 | 劣势 |
|------|------|------|
| **向量检索** | 语义理解强 | 精确匹配弱 |
| **关键词检索 (BM25)** | 精确匹配强 | 语义理解弱 |
| **混合检索** | 兼顾语义和精确 | 实现复杂度高 |

### 混合检索原理

```
查询："Python 异常处理最佳实践"

                    ┌─────────────────────┐
                    │      用户查询        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │   向量检索       │              │   关键词检索     │
    │  (语义相似)      │              │   (BM25)        │
    └────────┬────────┘              └────────┬────────┘
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ Doc1: 异常处理   │              │ Doc1: Python异常 │ 0.85
    │ Doc2: 错误处理   │              │ Doc3: 异常最佳   │ 0.72
    │ Doc3: 代码规范   │              │ Doc5: Python处理 │ 0.65
    └────────┬────────┘              └────────┬────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   分数融合        │
                    │ RRF / 加权平均   │
                    └────────┬────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │   最终排序结果   │
                    │ 1. Doc1 (综合)  │
                    │ 2. Doc3 (综合)  │
                    │ 3. Doc2 (向量)  │
                    └─────────────────┘
```

### 实战：LangChain 混合检索

```python
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. 准备文档
documents = [
    Document(page_content="Python 异常处理使用 try-except 语句捕获错误。", metadata={"id": 1}),
    Document(page_content="Java 的 try-catch 用于异常处理。", metadata={"id": 2}),
    Document(page_content="Python 异常处理的最佳实践包括使用具体的异常类型。", metadata={"id": 3}),
    Document(page_content="错误处理是编程中的重要技能。", metadata={"id": 4}),
    Document(page_content="Python 中的 raise 语句用于主动抛出异常。", metadata={"id": 5}),
]

# 2. 创建向量检索器
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Chroma.from_documents(documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 4. 创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 向量和关键词各占 50%
)

# 5. 检索
query = "Python 异常处理最佳实践"
results = ensemble_retriever.get_relevant_documents(query)

print(f"查询: {query}")
print("\n混合检索结果:")
for i, doc in enumerate(results[:5], 1):
    print(f"{i}. {doc.page_content}")
```

### RRF（Reciprocal Rank Fusion）融合算法

```python
from typing import List, Dict
from langchain.schema import Document

def reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    RRF 融合算法
    Args:
        results_list: 多个检索器的结果列表
        k: RRF 参数，通常取 60
    Returns:
        融合后的排序结果
    """
    # 为每个文档计算 RRF 分数
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, 1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            
            # RRF 分数累加
            rrf_score = 1 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
    
    # 按分数排序
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]

# 使用示例
vector_results = vector_retriever.get_relevant_documents(query)
bm25_results = bm25_retriever.get_relevant_documents(query)

final_results = reciprocal_rank_fusion([vector_results, bm25_results])
```

---

## 📊 重排序（Reranking）

### 为什么需要重排序

```
检索阶段的限制：
├── 向量相似度计算是近似的
├── BM25 只考虑词频，不考虑语义
├── 初始检索可能返回无关文档
└── 需要更精细的相关性判断

重排序的优势：
├── 使用更强大的模型精确计算相关性
├── 考虑查询-文档的深度交互
├── 提高最终结果的质量
└── 可解释性更强
```

### Cross-Encoder vs Bi-Encoder

```
Bi-Encoder（双塔编码器）- 用于检索

查询 ──→ [Encoder] ──→ 向量A ──┐
                              ├──→ 余弦相似度
文档 ──→ [Encoder] ──→ 向量B ──┘

特点：向量可预计算，检索速度快


Cross-Encoder（交叉编码器）- 用于重排序

[查询 + 文档] ──→ [Encoder] ──→ 相关性分数

特点：精确度高，但计算量大，用于小规模重排序
```

### 实战：使用 Cohere Rerank

```python
# pip install cohere

import cohere

co = cohere.Client("your-api-key")

def rerank_documents(query: str, documents: List[str], top_n: int = 5) -> List[dict]:
    """
    使用 Cohere Rerank API 重排序文档
    """
    results = co.rerank(
        model="rerank-multilingual-v2.0",  # 多语言模型
        query=query,
        documents=documents,
        top_n=top_n
    )
    
    reranked = []
    for result in results:
        reranked.append({
            "document": documents[result.index],
            "relevance_score": result.relevance_score,
            "index": result.index
        })
    
    return reranked

# 使用示例
query = "Python 如何处理异常？"
docs = [
    "Python 使用 try-except 语句处理异常。",
    "Java 和 Python 都是流行的编程语言。",
    "异常处理是程序健壮性的关键。",
    "Python 的 raise 语句可以主动抛出异常。"
]

reranked = rerank_documents(query, docs, top_n=3)
print(f"查询: {query}\n")
for item in reranked:
    print(f"分数: {item['relevance_score']:.4f}")
    print(f"文档: {item['document']}\n")
```

### 实战：使用本地 Rerank 模型

```python
# pip install sentence-transformers

from sentence_transformers import CrossEncoder
from typing import List, Tuple

class LocalReranker:
    """本地重排序模型"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        对文档进行重排序
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果
        Returns:
            排序后的 (文档, 分数) 列表
        """
        # 构造 query-document 对
        pairs = [(query, doc) for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]

# 使用示例
reranker = LocalReranker()

query = "深度学习的应用场景有哪些？"
docs = [
    "深度学习在图像识别领域取得了突破性进展。",
    "Python 是一门流行的编程语言。",
    "深度学习可以应用于自然语言处理任务。",
    "机器学习是人工智能的一个分支。"
]

reranked = reranker.rerank(query, docs, top_k=3)
print(f"查询: {query}\n")
for doc, score in reranked:
    print(f"分数: {score:.4f} - {doc}")
```

### LangChain 中的重排序

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.cross_encoders import HuggingFaceCrossEncoder

# 创建基础检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 创建重排序器
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
compressor = CrossEncoderReranker(
    model=model,
    top_n=5  # 重排序后保留 5 个文档
)

# 创建带重排序的检索器
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 检索
results = reranking_retriever.get_relevant_documents("什么是机器学习？")
```

---

## 🔗 多跳检索（Multi-Hop Retrieval）

### 什么是多跳检索

某些复杂问题需要多个步骤才能回答，单次检索无法获取所有必要信息。

```
单跳问题：
"Python 的作者是谁？" → 检索 → "Guido van Rossum"

多跳问题：
"Python 作者创建的另一个编程语言是什么？"
→ 第一次检索: "Python 作者是 Guido van Rossum"
→ 第二次检索: "Guido van Rossum 创建的其他语言"
→ 最终答案: "ABC 语言"
```

### 多跳检索架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     多跳检索流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   问题: "Python 作者创建的另一个编程语言是什么？"                     │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────────┐                      │
│   │ 第一跳: 检索 "Python 作者"               │                      │
│   │ 结果: "Python 由 Guido van Rossum 创建"  │                      │
│   └───────────────────┬─────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │ 提取关键信息: Guido van Rossum           │                      │
│   └───────────────────┬─────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │ 第二跳: 检索 "Guido van Rossum 其他语言" │                      │
│   │ 结果: "他还创建了 ABC 语言"              │                      │
│   └───────────────────┬─────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │ 综合回答: "ABC 语言"                     │                      │
│   └─────────────────────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 实战：实现多跳检索

```python
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

class MultiHopRetriever:
    """多跳检索器"""
    
    def __init__(self, vectorstore, llm, max_hops: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_hops = max_hops
    
    def retrieve(self, question: str) -> Dict:
        """
        执行多跳检索
        Returns:
            包含所有检索文档和最终答案的字典
        """
        all_docs = []
        current_query = question
        hop_history = []
        
        for hop in range(self.max_hops):
            # 检索
            docs = self.vectorstore.similarity_search(current_query, k=3)
            all_docs.extend(docs)
            
            # 记录这一跳
            hop_history.append({
                "hop": hop + 1,
                "query": current_query,
                "docs": [doc.page_content[:100] for doc in docs]
            })
            
            # 判断是否需要继续检索
            if hop < self.max_hops - 1:
                # 生成下一步查询
                next_query = self._generate_next_query(question, docs)
                if next_query is None:
                    break
                current_query = next_query
        
        # 生成最终答案
        answer = self._generate_answer(question, all_docs)
        
        return {
            "answer": answer,
            "documents": all_docs,
            "hop_history": hop_history
        }
    
    def _generate_next_query(self, original_question: str, docs: List) -> str:
        """生成下一跳的查询"""
        template = """分析问题和已检索的文档，判断是否需要继续检索。

原始问题: {question}

已检索文档:
{docs}

请回答:
1. 是否需要继续检索？(是/否)
2. 如果需要，下一个查询应该是什么？

格式:
需要继续: 是/否
下一个查询: ..."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "docs"]
        )
        
        docs_text = "\n".join([doc.page_content for doc in docs])
        response = self.llm.invoke(
            prompt.format(question=original_question, docs=docs_text)
        )
        
        # 解析响应
        if "需要继续: 否" in response.content:
            return None
        
        lines = response.content.split("\n")
        for line in lines:
            if "下一个查询:" in line:
                return line.split("下一个查询:")[1].strip()
        
        return None
    
    def _generate_answer(self, question: str, docs: List) -> str:
        """生成最终答案"""
        context = "\n\n".join([doc.page_content for doc in docs])
        
        template = """基于以下文档回答问题。

文档:
{context}

问题: {question}

回答:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return self.llm.invoke(prompt.format(context=context, question=question)).content


# 使用示例
llm = ChatOpenAI(model="gpt-4")
multi_hop = MultiHopRetriever(vectorstore, llm, max_hops=3)

result = multi_hop.retrieve("Python 作者创建的另一个编程语言是什么？")
print("答案:", result["answer"])
print("\n检索历史:")
for hop in result["hop_history"]:
    print(f"  第{hop['hop']}跳: {hop['query']}")
```

---

## 🕸️ GraphRAG

### 什么是 GraphRAG

GraphRAG 将知识图谱与 RAG 结合，利用实体关系增强检索能力。

```
传统 RAG vs GraphRAG

传统 RAG:
用户问题 → 向量检索 → 文档片段 → 生成答案
         (可能遗漏关联信息)

GraphRAG:
用户问题 → 实体识别 → 图谱查询 → 关联实体 → 生成答案
         (利用实体关系扩展上下文)
```

### GraphRAG 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GraphRAG 架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    知识图谱构建                               │  │
│   │  文档 ─→ 实体抽取 ─→ 关系抽取 ─→ 知识图谱                     │  │
│   │         (NER)       (RE)                                    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    查询处理                                  │  │
│   │  问题 ─→ 实体识别 ─→ 图谱遍历 ─→ 相关子图                    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    上下文融合                                 │  │
│   │  向量检索结果 + 图谱关联信息 → 增强上下文                      │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    答案生成                                   │  │
│   │  增强上下文 + 问题 → LLM → 答案                              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 实战：简单 GraphRAG 实现

```python
from typing import List, Dict, Set
import networkx as nx
from langchain.chat_models import ChatOpenAI

class SimpleGraphRAG:
    """简化的 GraphRAG 实现"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.graph = nx.Graph()
        self.entity_to_docs = {}  # 实体到文档的映射
    
    def build_graph(self, documents: List[Dict]):
        """
        从文档构建知识图谱
        Args:
            documents: 包含 content 和 entities 的文档列表
        """
        for doc_id, doc in enumerate(documents):
            content = doc["content"]
            entities = doc.get("entities", [])
            
            # 添加实体节点
            for entity in entities:
                if entity not in self.entity_to_docs:
                    self.entity_to_docs[entity] = []
                self.entity_to_docs[entity].append(doc_id)
                
                # 添加节点
                self.graph.add_node(entity, type="entity")
            
            # 添加实体间的边（共现关系）
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if self.graph.has_edge(entity1, entity2):
                        self.graph[entity1][entity2]["weight"] += 1
                    else:
                        self.graph.add_edge(entity1, entity2, weight=1)
    
    def extract_entities(self, text: str) -> List[str]:
        """使用 LLM 提取实体"""
        prompt = f"""从以下文本中提取关键实体（人名、地名、组织、概念等）。

文本: {text}

请以 JSON 列表格式输出实体: ["实体1", "实体2", ...]"""
        
        response = self.llm.invoke(prompt)
        # 简化处理，实际应使用 JSON 解析
        entities = []
        for line in response.content.split('"'):
            if len(line) > 1 and len(line) < 20:
                entities.append(line.strip())
        return entities
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        执行 GraphRAG 查询
        """
        # 1. 从问题中提取实体
        query_entities = self.extract_entities(question)
        
        # 2. 向量检索
        vector_docs = self.vectorstore.similarity_search(question, k=k*2)
        
        # 3. 图谱扩展：找到相关实体
        related_entities = set()
        for entity in query_entities:
            if entity in self.graph:
                # 获取相邻节点
                neighbors = list(self.graph.neighbors(entity))
                related_entities.update(neighbors[:5])
        
        # 4. 根据相关实体获取额外文档
        graph_doc_ids = set()
        for entity in related_entities:
            if entity in self.entity_to_docs:
                graph_doc_ids.update(self.entity_to_docs[entity])
        
        # 5. 融合上下文
        all_docs = vector_docs + [
            self.documents[doc_id] 
            for doc_id in graph_doc_ids 
            if doc_id < len(self.documents)
        ]
        
        # 6. 生成答案
        context = "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else doc["content"] 
            for doc in all_docs[:10]
        ])
        
        answer_prompt = f"""基于以下上下文回答问题。

上下文:
{context}

问题: {question}

回答:"""
        
        answer = self.llm.invoke(answer_prompt).content
        
        return {
            "answer": answer,
            "query_entities": query_entities,
            "related_entities": list(related_entities),
            "source_docs": all_docs[:k]
        }


# 使用示例
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

# 准备数据
documents = [
    {
        "content": "Python 是由 Guido van Rossum 创建的编程语言。",
        "entities": ["Python", "Guido van Rossum", "编程语言"]
    },
    {
        "content": "Guido van Rossum 还创建了 ABC 语言。",
        "entities": ["Guido van Rossum", "ABC语言", "编程语言"]
    },
    {
        "content": "Python 广泛应用于机器学习和数据科学。",
        "entities": ["Python", "机器学习", "数据科学"]
    }
]

# 构建向量存储
texts = [doc["content"] for doc in documents]
vectorstore = Chroma.from_texts(texts, embeddings)

# 构建 GraphRAG
graph_rag = SimpleGraphRAG(vectorstore, llm)
graph_rag.documents = documents
graph_rag.build_graph(documents)

# 查询
result = graph_rag.query("Python 作者还创建了什么？")
print("答案:", result["answer"])
print("相关实体:", result["related_entities"])
```

---

## 🔄 查询改写与扩展

### 查询改写（Query Rewrite）

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.rewrite_prompt = PromptTemplate(
            template="""你是一个搜索专家。请将用户的问题改写得更容易检索到相关信息。

原始问题: {question}

改写要求:
1. 保留原始问题的核心意图
2. 使用更具体、更明确的关键词
3. 可以拆分成多个子问题

改写后的问题:""",
            input_variables=["question"]
        )
    
    def rewrite(self, question: str) -> List[str]:
        """改写查询"""
        response = self.llm.invoke(
            self.rewrite_prompt.format(question=question)
        )
        
        # 解析改写结果
        queries = [question]  # 保留原始问题
        for line in response.content.split("\n"):
            line = line.strip()
            if line and not line.startswith("改写"):
                queries.append(line)
        
        return queries[:3]  # 返回最多3个查询

# 使用
rewriter = QueryRewriter(ChatOpenAI(model="gpt-4"))
queries = rewriter.rewrite("那个做 Python 的人是谁？")
# 输出: ["那个做 Python 的人是谁？", "Python 编程语言创始人", "Guido van Rossum Python"]
```

### HyDE（Hypothetical Document Embeddings）

```python
class HyDERetriever:
    """
    HyDE: 假设文档嵌入
    思路：让 LLM 先生成一个假设的回答文档，用该文档进行检索
    """
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        
        self.hypothetical_prompt = PromptTemplate(
            template="""请为以下问题生成一个假设性的回答文档。

问题: {question}

假设文档:""",
            input_variables=["question"]
        )
    
    def retrieve(self, question: str, k: int = 5):
        """使用 HyDE 检索"""
        # 1. 生成假设文档
        hypothetical_doc = self.llm.invoke(
            self.hypothetical_prompt.format(question=question)
        ).content
        
        print(f"假设文档: {hypothetical_doc[:200]}...")
        
        # 2. 用假设文档检索
        docs = self.vectorstore.similarity_search(
            hypothetical_doc,  # 用假设文档而非原始问题检索
            k=k
        )
        
        return docs

# 使用示例
hyde = HyDERetriever(ChatOpenAI(model="gpt-4"), vectorstore)
docs = hyde.retrieve("如何优化 RAG 系统的检索效果？")
```

---

## 📈 高级 RAG 最佳实践

### 系统优化清单

```
检索优化:
├── [ ] 使用混合检索（向量 + 关键词）
├── [ ] 添加重排序模块
├── [ ] 优化文档切分粒度
├── [ ] 增加元数据过滤
└── [ ] 定期更新向量索引

生成优化:
├── [ ] 使用 Few-shot Prompt
├── [ ] 控制输出格式
├── [ ] 添加引用标注
├── [ ] 处理冲突信息
└── [ ] 设置合理的温度参数

评估优化:
├── [ ] 建立测试数据集
├── [ ] 定期评估检索质量
├── [ ] 监控幻觉率
├── [ ] 收集用户反馈
└── [ ] A/B 测试不同配置
```

### 常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| 检索结果不相关 | 增加重排序、调整检索参数 |
| 答案幻觉 | 加强 Prompt 约束、降低温度 |
| 答案不完整 | 增加检索文档数、使用多跳检索 |
| 响应速度慢 | 缓存、并行检索、减少文档数 |
| 长尾知识缺失 | 扩充知识库、优化文档切分 |

---

## 🔗 相关内容

- [检索器模块](./retriever.md) - 向量数据库、Embedding 模型详解
- [生成器模块](./generator.md) - Prompt 设计、上下文融合
- [RAG vs SFT 对比](./rag-vs-sft.md) - 技术选型指南
