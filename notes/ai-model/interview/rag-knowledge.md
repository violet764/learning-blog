# RAG 与知识增强面试题

本章节整理了检索增强生成（RAG）与知识增强相关的面试题目，涵盖 RAG 原理、向量数据库、检索优化、知识图谱等核心技术。

---

## 一、RAG 基础原理

### Q1: 什么是 RAG？为什么需要 RAG？

**基础回答：**

RAG（Retrieval-Augmented Generation）是一种结合检索系统和生成模型的技术，通过检索外部知识增强大模型的生成能力。

**深入回答：**

**RAG 解决的问题**：

```
大模型的局限性:

1. 知识时效性
   ├── 训练数据有截止日期
   ├── 无法获取最新信息
   └── 知识逐渐过时

2. 领域专业性
   ├── 通用模型缺乏领域知识
   ├── 企业私有数据无法学习
   └── 专业问题回答不准确

3. 事实准确性
   ├── 幻觉问题
   ├── 编造不存在的信息
   └── 无法溯源验证

4. 可解释性
   ├── 黑盒生成
   ├── 无法提供来源
   └── 可信度难评估
```

**RAG 优势**：

| 优势 | 说明 |
|------|------|
| **知识更新** | 无需重新训练即可更新知识 |
| **事实准确** | 基于检索的真实文档生成 |
| **可溯源** | 提供引用来源 |
| **领域适配** | 轻松接入领域知识 |
| **成本较低** | 避免大规模微调 |

**追问：RAG vs 微调 如何选择？**

```
选择决策:

适合 RAG 的场景:
├── 需要实时/最新信息
├── 需要引用来源
├── 知识频繁更新
├── 私有数据场景
└── 领域问答系统

适合微调的场景:
├── 需要改变模型行为模式
├── 学习特定格式/风格
├── 知识相对稳定
├── 需要更深的领域理解
└── 任务相对固定

混合方案:
├── 先微调学习领域语言风格
└── 再用 RAG 注入具体知识
```

---

### Q2: RAG 的核心流程是什么？

**基础回答：**

RAG 核心流程包括索引阶段（文档处理和向量化）和查询阶段（检索和生成）。

**深入回答：**

**完整流程**：

```
RAG 系统架构:

┌─────────────────────────────────────────────────────────┐
│                      索引阶段                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  原始文档                                                │
│     ↓                                                   │
│  文档解析 (PDF/HTML/DOCX 解析)                           │
│     ↓                                                   │
│  文本分块 (Chunking)                                     │
│     ↓                                                   │
│  向量化 (Embedding)                                      │
│     ↓                                                   │
│  向量数据库存储                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      查询阶段                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户查询                                                │
│     ↓                                                   │
│  Query 理解与改写                                        │
│     ↓                                                   │
│  向量检索 + 关键词检索                                    │
│     ↓                                                   │
│  重排序 (Reranking)                                      │
│     ↓                                                   │
│  上下文构建                                              │
│     ↓                                                   │
│  LLM 生成回答                                            │
│     ↓                                                   │
│  后处理与引用标注                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例**：

```python
class RAGSystem:
    def __init__(self, embedding_model, vector_db, llm):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm
    
    def index_documents(self, documents):
        """索引文档"""
        chunks = []
        for doc in documents:
            # 分块
            doc_chunks = self.chunk_text(doc)
            chunks.extend(doc_chunks)
        
        # 向量化
        embeddings = self.embedding_model.encode(chunks)
        
        # 存储
        self.vector_db.add(chunks, embeddings)
    
    def query(self, question, top_k=5):
        """查询"""
        # 1. 向量化查询
        query_embedding = self.embedding_model.encode(question)
        
        # 2. 检索
        results = self.vector_db.search(query_embedding, top_k)
        
        # 3. 构建上下文
        context = "\n".join([r['text'] for r in results])
        
        # 4. 生成回答
        prompt = f"""
基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文：
{context}

问题：{question}

回答：
"""
        answer = self.llm.generate(prompt)
        
        return {
            'answer': answer,
            'sources': results
        }
```

---

## 二、向量数据库

### Q3: 向量数据库的工作原理是什么？

**基础回答：**

向量数据库存储和检索向量嵌入，通过相似度计算（如余弦相似度）找到最相似的向量。

**深入回答：**

**核心概念**：

```
向量数据库工作原理:

1. 向量表示
   ├── 将文本转换为高维向量
   ├── 语义相似的文本向量距离近
   └── 典型维度: 384, 768, 1536

2. 相似度计算
   ├── 余弦相似度: cos(a, b) = (a·b) / (|a||b|)
   ├── 欧氏距离: ||a - b||
   ├── 内积: a · b
   └── 最常用: 余弦相似度

3. 索引结构
   ├── 暴力搜索: 准确但慢
   ├── IVF (倒排索引): 分桶加速
   ├── HNSW (分层导航小世界): 图结构加速
   └── PQ (乘积量化): 压缩存储
```

**索引类型对比**：

| 索引类型 | 查询速度 | 准确度 | 内存占用 | 适用场景 |
|----------|----------|--------|----------|----------|
| **Flat** | 慢 | 100% | 高 | 小规模精确搜索 |
| **IVF** | 中 | 高 | 中 | 大规模搜索 |
| **HNSW** | 快 | 高 | 高 | 高性能要求 |
| **PQ** | 快 | 中 | 低 | 内存受限场景 |

**追问：HNSW 如何工作？**

```
HNSW (Hierarchical Navigable Small World):

结构:
├── 多层图结构
├── 上层稀疏，下层稠密
└── 类似跳表的思想

搜索过程:
1. 从最上层开始
2. 在每层找到最近邻
3. 进入下一层继续搜索
4. 直到最底层得到结果

优点:
├── 查询速度快
├── 支持增量更新
└── 高召回率
```

---

### Q4: 主流向量数据库有哪些？如何选择？

**基础回答：**

主流向量数据库包括 Milvus、Pinecone、Weaviate、Qdrant、Chroma 等，各有特点。

**深入回答：**

**对比分析**：

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Milvus** | 开源 | 分布式、高性能、功能丰富 | 大规模生产环境 |
| **Pinecone** | 托管 | 全托管、易用 | 快速部署、无运维需求 |
| **Weaviate** | 开源 | 语义搜索、GraphQL | 语义丰富的应用 |
| **Qdrant** | 开源 | Rust 实现、高性能 | 性能敏感场景 |
| **Chroma** | 开源 | 轻量、Python 原生 | 原型开发、小规模应用 |
| **FAISS** | 库 | Meta 开源、纯向量搜索 | 本地向量搜索 |

**选择决策树**：

```
向量数据库选择:

是否有运维能力?
├── 否 → Pinecone (托管服务)
│
└── 是 → 数据规模?
    ├── 小 (<100万) → Chroma / FAISS
    │
    └── 大 (>100万) → 性能要求?
        ├── 极高 → Milvus / Qdrant
        └── 中等 → Weaviate
```

---

## 三、检索优化

### Q5: 文档分块有哪些策略？

**基础回答：**

文档分块策略包括固定长度分块、语义分块、递归分块等，影响检索质量和生成效果。

**深入回答：**

**分块策略**：

```python
# 1. 固定长度分块
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    """固定长度分块"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# 2. 句子级分块
def sentence_chunk(text):
    """按句子分块"""
    import nltk
    sentences = nltk.sent_tokenize(text)
    return sentences

# 3. 递归分块
def recursive_chunk(text, chunk_size=500, separators=["\n\n", "\n", ". ", " "]):
    """递归分块，保持语义完整性"""
    if len(text) <= chunk_size:
        return [text]
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                if len(current) + len(part) <= chunk_size:
                    current += part + sep
                else:
                    if current:
                        chunks.append(current)
                    current = part + sep
            if current:
                chunks.append(current)
            return chunks
    
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 4. 语义分块
def semantic_chunk(text, embedding_model, threshold=0.8):
    """基于语义相似度分块"""
    sentences = split_sentences(text)
    embeddings = [embedding_model.encode(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

**分块参数选择**：

| 参数 | 推荐 | 说明 |
|------|------|------|
| **chunk_size** | 300-500 | 太小丢失上下文，太大检索不精确 |
| **overlap** | 50-100 | 保持上下文连贯性 |
| **分隔符** | 段落优先 | 保持语义完整性 |

---

### Q6: 如何提高检索质量？

**基础回答：**

提高检索质量可以从查询理解、混合检索、重排序、查询扩展等方面优化。

**深入回答：**

**优化策略**：

```
检索优化方法:

1. 查询理解与改写
   ├── 查询扩展: 添加同义词、相关词
   ├── 查询分解: 将复杂查询分解
   ├── HyDE: 生成假设文档再检索
   └── Query Rewrite: 改写为更易检索的形式

2. 混合检索
   ├── 向量检索 + 关键词检索
   ├── 多向量检索 (不同 embedding)
   └── 结果融合 (RRF)

3. 重排序
   ├── Cross-encoder 重排序
   ├── LLM 重排序
   └── 多因子综合排序

4. 检索后处理
   ├── 去重
   ├── 多样性保证
   └── 上下文窗口扩展
```

**HyDE 示例**：

```python
def hyde_retrieval(query, llm, embedding_model, vector_db, top_k=5):
    """Hypothetical Document Embeddings"""
    
    # 1. 生成假设文档
    prompt = f"请生成一段可能回答以下问题的文档：\n问题：{query}"
    hypothetical_doc = llm.generate(prompt)
    
    # 2. 对假设文档向量化
    hyde_embedding = embedding_model.encode(hypothetical_doc)
    
    # 3. 用假设文档的向量检索
    results = vector_db.search(hyde_embedding, top_k)
    
    return results
```

**混合检索示例**：

```python
def hybrid_retrieval(query, embedding_model, vector_db, keyword_index, alpha=0.5, top_k=10):
    """混合检索：向量 + 关键词"""
    
    # 1. 向量检索
    query_embedding = embedding_model.encode(query)
    vector_results = vector_db.search(query_embedding, top_k * 2)
    
    # 2. 关键词检索
    keyword_results = keyword_index.search(query, top_k * 2)
    
    # 3. RRF (Reciprocal Rank Fusion) 融合
    def rrf_score(rank, k=60):
        return 1 / (k + rank)
    
    scores = {}
    for rank, result in enumerate(vector_results):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + alpha * rrf_score(rank)
    
    for rank, result in enumerate(keyword_results):
        doc_id = result['id']
        scores[doc_id] = scores.get(doc_id, 0) + (1-alpha) * rrf_score(rank)
    
    # 4. 排序返回
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
```

---

### Q7: 什么是 Reranking？为什么需要 Reranking？

**基础回答：**

Reranking 是对初步检索结果进行重新排序的过程，使用更精确的模型计算文档与查询的相关性。

**深入回答：**

**为什么需要 Reranking**：

```
向量检索的局限:
├── Bi-encoder 只能捕获粗粒度相似度
├── 向量化过程损失部分语义信息
├── 难以处理细粒度匹配
└── 无法利用查询和文档的交互信息

Reranking 优势:
├── Cross-encoder 能捕获细粒度交互
├── 更精确的相关性判断
├── 可以加入更多特征
└── 提升最终检索质量
```

**Reranking 实现**：

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        """重排序"""
        # 构建查询-文档对
        pairs = [(query, doc['text']) for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top-k
        return [doc for doc, score in scored_docs[:top_k]]

# LLM Reranking
def llm_rerank(query, documents, llm):
    """使用 LLM 进行重排序"""
    results = []
    
    for doc in documents:
        prompt = f"""
请评估以下文档与查询的相关性，给出1-10分的评分。

查询：{query}
文档：{doc['text']}

评分（只需给出数字）：
"""
        score = float(llm.generate(prompt).strip())
        results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results]
```

---

## 四、高级 RAG 技术

### Q8: 什么是 GraphRAG？有什么优势？

**基础回答：**

GraphRAG 将知识图谱与 RAG 结合，利用图结构捕获实体关系，提供更丰富的上下文信息。

**深入回答：**

**GraphRAG 架构**：

```
GraphRAG 流程:

1. 知识图谱构建
   ├── 实体抽取
   ├── 关系抽取
   ├── 图谱存储
   └── 图谱索引

2. 图增强检索
   ├── 实体链接
   ├── 图遍历
   ├── 子图提取
   └── 多跳推理

3. 增强生成
   ├── 图结构上下文
   ├── 实体关系描述
   └── 推理路径
```

**代码示例**：

```python
class GraphRAG:
    def __init__(self, kg, embedding_model, llm):
        self.kg = kg  # 知识图谱
        self.embedding_model = embedding_model
        self.llm = llm
    
    def query(self, question):
        """GraphRAG 查询"""
        
        # 1. 识别问题中的实体
        entities = self.extract_entities(question)
        
        # 2. 在知识图谱中查找相关子图
        subgraph = self.kg.get_subgraph(entities, hops=2)
        
        # 3. 向量检索补充文档
        vector_results = self.vector_search(question)
        
        # 4. 构建增强上下文
        context = self.build_context(subgraph, vector_results)
        
        # 5. 生成回答
        answer = self.llm.generate(self.build_prompt(question, context))
        
        return answer
    
    def build_context(self, subgraph, vector_results):
        """构建图增强上下文"""
        context = "知识图谱信息：\n"
        
        # 添加三元组信息
        for triple in subgraph.triples:
            context += f"- {triple.subject} {triple.predicate} {triple.object}\n"
        
        context += "\n相关文档：\n"
        for doc in vector_results:
            context += f"- {doc['text']}\n"
        
        return context
```

**优势对比**：

| 方面 | 普通 RAG | GraphRAG |
|------|----------|----------|
| **关系捕获** | 弱 | 强 |
| **多跳推理** | 困难 | 支持 |
| **结构化知识** | 不支持 | 原生支持 |
| **复杂度** | 低 | 高 |
| **适用场景** | 事实问答 | 关系推理 |

---

### Q9: 如何实现多轮对话 RAG？

**基础回答：**

多轮对话 RAG 需要考虑对话历史、上下文理解、查询改写等问题，保持对话连贯性。

**深入回答：**

**实现方案**：

```python
class ConversationalRAG:
    def __init__(self, embedding_model, vector_db, llm):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm
        self.history = []
    
    def query(self, question):
        """多轮对话查询"""
        
        # 1. 查询改写（融入历史）
        rewritten_query = self.rewrite_query(question)
        
        # 2. 检索
        context = self.retrieve(rewritten_query)
        
        # 3. 生成回答
        answer = self.generate(question, context)
        
        # 4. 更新历史
        self.history.append({"question": question, "answer": answer})
        
        return answer
    
    def rewrite_query(self, question):
        """查询改写"""
        if not self.history:
            return question
        
        # 构建改写提示
        history_text = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']}"
            for h in self.history[-3:]  # 最近3轮
        ])
        
        prompt = f"""
对话历史：
{history_text}

当前问题：{question}

请将当前问题改写为一个独立、完整的问题（包含必要的上下文）：
"""
        return self.llm.generate(prompt)
    
    def generate(self, question, context):
        """生成回答"""
        # 包含历史的生成
        history_context = "\n".join([
            f"问：{h['question']}\n答：{h['answer']}"
            for h in self.history[-2:]
        ])
        
        prompt = f"""
之前的对话：
{history_context}

检索到的相关信息：
{context}

当前问题：{question}

请基于以上信息回答问题：
"""
        return self.llm.generate(prompt)
```

---

### Q10: 如何评估 RAG 系统效果？

**基础回答：**

RAG 评估包括检索评估（召回率、精确率）和生成评估（相关性、准确性、流畅性）。

**深入回答：**

**评估维度**：

```
RAG 评估框架:

1. 检索质量评估
   ├── 召回率 (Recall): 相关文档被检索到的比例
   ├── 精确率 (Precision): 检索结果中相关文档的比例
   ├── MRR (Mean Reciprocal Rank): 第一个相关结果的排名
   └── NDCG: 考虑排序的评估指标

2. 生成质量评估
   ├── 准确性: 回答是否正确
   ├── 相关性: 回答是否与问题相关
   ├── 完整性: 回答是否完整
   ├── 流畅性: 语言是否通顺
   └── 引用准确性: 引用是否正确

3. 端到端评估
   ├── RAGAS: 专门针对 RAG 的评估框架
   ├── ARES: 自动化 RAG 评估
   └── 人工评估
```

**RAGAS 评估示例**：

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

def evaluate_rag(test_data, rag_system):
    """使用 RAGAS 评估 RAG 系统"""
    
    results = []
    for sample in test_data:
        # 获取 RAG 结果
        response = rag_system.query(sample['question'])
        
        results.append({
            'question': sample['question'],
            'answer': response['answer'],
            'contexts': [r['text'] for r in response['sources']],
            'ground_truth': sample['answer']
        })
    
    # 计算 RAGAS 指标
    scores = evaluate(
        results,
        metrics=[faithfulness, answer_relevancy, context_relevancy]
    )
    
    return scores
```

---

## 五、实战问题

### Q11: RAG 系统常见问题及解决方案？

**基础回答：**

RAG 系统常见问题包括检索不相关、上下文过长、回答不准确等，需要针对性解决。

**深入回答：**

**问题与解决方案**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **检索不相关** | Embedding 质量差、分块不合理 | 换更好的 Embedding、优化分块 |
| **上下文过长** | 检索文档太多、文档太长 | 限制文档数量、压缩上下文 |
| **回答不准确** | 检索质量差、Prompt 不当 | 提升检索质量、优化 Prompt |
| **回答幻觉** | 上下文信息不足 | 明确要求"不知道时说明" |
| **响应慢** | 检索慢、生成慢 | 优化索引、缓存、并行 |
| **成本高** | LLM 调用频繁 | 缓存结果、优化检索数量 |

**代码示例**：

```python
def robust_rag_query(question, rag_system, max_retries=3):
    """鲁棒的 RAG 查询"""
    
    for attempt in range(max_retries):
        try:
            # 1. 检索
            results = rag_system.retrieve(question)
            
            if not results:
                return "抱歉，没有找到相关信息。"
            
            # 2. 检查相关性
            if results[0]['score'] < 0.5:
                # 相关性太低，尝试查询改写
                rewritten = rag_system.rewrite_query(question)
                results = rag_system.retrieve(rewritten)
            
            # 3. 生成
            answer = rag_system.generate(question, results)
            
            # 4. 验证回答
            if "我不知道" in answer or len(answer) < 10:
                # 回答质量不佳，重试
                continue
            
            return {
                'answer': answer,
                'sources': results,
                'confidence': results[0]['score']
            }
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    return "抱歉，处理您的问题时出现错误，请稍后重试。"
```

---

### Q12: 如何构建企业级 RAG 系统？

**基础回答：**

企业级 RAG 系统需要考虑数据管道、多租户、权限控制、监控运维等方面。

**深入回答：**

**系统架构**：

```
企业级 RAG 系统架构:

┌─────────────────────────────────────────────────────────┐
│                      接入层                              │
│  ├── API Gateway                                        │
│  ├── 认证鉴权                                            │
│  └── 流量控制                                            │
├─────────────────────────────────────────────────────────┤
│                      应用层                              │
│  ├── 多租户管理                                          │
│  ├── 对话管理                                            │
│  └── 权限控制                                            │
├─────────────────────────────────────────────────────────┤
│                      RAG 核心                            │
│  ├── Query 理解                                          │
│  ├── 检索引擎                                            │
│  ├── 重排序                                              │
│  └── 生成引擎                                            │
├─────────────────────────────────────────────────────────┤
│                      数据层                              │
│  ├── 向量数据库                                          │
│  ├── 知识图谱                                            │
│  ├── 文档存储                                            │
│  └── 缓存层                                              │
├─────────────────────────────────────────────────────────┤
│                      运维层                              │
│  ├── 监控告警                                            │
│  ├── 日志收集                                            │
│  ├── 数据管道                                            │
│  └── 自动化运维                                          │
└─────────────────────────────────────────────────────────┘
```

**关键设计**：

```python
class EnterpriseRAG:
    def __init__(self, config):
        self.config = config
        self.tenant_manager = TenantManager()
        self.permission_manager = PermissionManager()
        self.cache = CacheManager()
    
    def query(self, tenant_id, user_id, question):
        """企业级查询"""
        
        # 1. 租户隔离检查
        tenant = self.tenant_manager.get_tenant(tenant_id)
        
        # 2. 权限检查
        if not self.permission_manager.check(user_id, 'query'):
            raise PermissionError("无查询权限")
        
        # 3. 缓存检查
        cache_key = f"{tenant_id}:{question}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 4. 检索（带权限过滤）
        results = self.retrieve_with_permission(
            question, 
            tenant_id,
            user_id
        )
        
        # 5. 生成
        answer = self.generate(question, results)
        
        # 6. 缓存结果
        self.cache.set(cache_key, answer, ttl=3600)
        
        # 7. 记录日志
        self.log_query(tenant_id, user_id, question, answer)
        
        return answer
    
    def retrieve_with_permission(self, question, tenant_id, user_id):
        """带权限过滤的检索"""
        # 获取用户可访问的文档范围
        accessible_docs = self.permission_manager.get_accessible_docs(
            tenant_id, user_id
        )
        
        # 在权限范围内检索
        results = self.vector_db.search(
            question,
            filter={'doc_id': {'$in': accessible_docs}}
        )
        
        return results
```

---

## 📝 总结

### 核心知识点

| 主题 | 核心要点 |
|------|----------|
| **RAG 原理** | 索引阶段、查询阶段、与微调对比 |
| **向量数据库** | 相似度计算、索引类型、选型决策 |
| **检索优化** | 分块策略、混合检索、Reranking |
| **高级技术** | GraphRAG、多轮对话、评估方法 |
| **工程实践** | 问题排查、企业级架构 |

### 面试高频追问

1. **原理层面**：RAG 流程是什么？向量检索原理？
2. **优化层面**：如何提高检索质量？分块策略有哪些？
3. **对比层面**：RAG vs 微调？不同向量数据库对比？
4. **实践层面**：常见问题如何解决？如何设计企业级系统？

---

*[返回面试指南目录](./index.md)*
