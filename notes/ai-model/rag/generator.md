# 生成器模块（Generator）

生成器是 RAG 系统的第二个核心组件，负责将检索到的文档与用户查询结合，通过大语言模型生成准确、有依据的回答。本章将详细介绍 Prompt 设计、上下文融合策略以及生成器优化技巧。

---

## 🎯 生成器的作用

### 生成器的定位

```
┌─────────────────────────────────────────────────────────────────────┐
│                       RAG 生成器工作流程                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   用户查询 + 检索到的文档                                            │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────┐                      │
│   │           Prompt 组装                    │                      │
│   │  ┌─────────────────────────────────┐   │                      │
│   │  │ 系统指令                         │   │                      │
│   │  │ + 检索上下文                     │   │                      │
│   │  │ + 用户问题                       │   │                      │
│   │  │ + 输出格式要求                   │   │                      │
│   │  └─────────────────────────────────┘   │                      │
│   └───────────────────┬─────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │           LLM 推理                       │                      │
│   │        (GPT-4 / Claude / 开源模型)       │                      │
│   └───────────────────┬─────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │           输出处理                       │                      │
│   │  格式化 + 引用标注 + 后处理              │                      │
│   └─────────────────────────────────────────┘                      │
│                       │                                             │
│                       ▼                                             │
│                   最终回答                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 生成器的核心任务

| 任务 | 描述 | 关键点 |
|------|------|--------|
| **上下文理解** | 理解检索文档的内容 | 文档相关性判断 |
| **信息整合** | 从多文档中提取关键信息 | 去重、融合、排序 |
| **答案生成** | 生成准确、流畅的回答 | 避免幻觉、保持一致性 |
| **来源标注** | 标注信息的来源文档 | 可追溯性 |

---

## 📝 Prompt 设计

### Prompt 的基本结构

```python
# RAG Prompt 的基本模板

RAG_PROMPT_TEMPLATE = """
你是一个专业的问答助手。请根据提供的上下文文档回答用户问题。

## 上下文文档
{context}

## 用户问题
{question}

## 回答要求
1. 仅使用上下文文档中的信息回答问题
2. 如果文档中没有相关信息，请诚实说明
3. 回答要简洁准确，不要编造信息
4. 标注信息来源，格式为 [来源X]

## 回答
"""
```

### 不同类型的 Prompt 模板

#### 1. 基础问答模板

```python
from langchain.prompts import PromptTemplate

BASIC_QA_PROMPT = PromptTemplate(
    template="""使用以下上下文回答问题。如果无法从上下文中找到答案，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：""",
    input_variables=["context", "question"]
)
```

#### 2. 带引用的问答模板

```python
CITATION_PROMPT = PromptTemplate(
    template="""你是一个专业的问答助手。请基于提供的文档回答问题，并为每个关键信息标注来源。

文档列表：
{context}

问题：{question}

回答要求：
1. 综合使用多个文档中的信息
2. 每个事实后标注来源编号，如 [文档1]
3. 如果不同文档有冲突信息，请注明
4. 如果文档中没有相关信息，明确说明

回答：""",
    input_variables=["context", "question"]
)
```

#### 3. 对话式问答模板

```python
CONVERSATIONAL_PROMPT = PromptTemplate(
    template="""你是一个友好的对话助手。请基于提供的知识库回答用户问题，同时保持对话的连贯性。

知识库内容：
{context}

对话历史：
{chat_history}

当前问题：{question}

回答要求：
1. 参考对话历史理解问题的完整意图
2. 使用知识库中的信息回答
3. 保持回答的自然和对话感
4. 如果需要更多信息，可以追问

回答：""",
    input_variables=["context", "chat_history", "question"]
)
```

#### 4. 结构化输出模板

```python
STRUCTURED_PROMPT = PromptTemplate(
    template="""请基于以下文档回答问题，并按指定格式输出。

文档：
{context}

问题：{question}

请按以下 JSON 格式输出：
{{
    "answer": "你的回答",
    "confidence": "高/中/低",
    "sources": ["来源1", "来源2"],
    "related_questions": ["相关问题1", "相关问题2"]
}}

输出：""",
    input_variables=["context", "question"]
)
```

### Prompt 设计原则

```
Prompt 设计黄金法则：

1. 明确角色设定
   └── "你是一个专业的XX助手..."

2. 清晰的任务说明
   └── "请基于文档回答问题，不要编造信息"

3. 具体的约束条件
   └── "回答不超过200字"，"必须标注来源"

4. 示例引导（Few-shot）
   └── 提供正确回答的示例

5. 处理边界情况
   └── "如果文档中没有相关信息，请说不知道"

6. 输出格式规范
   └── 指定 JSON、Markdown 等格式
```

### Few-shot Prompt 示例

```python
from langchain.prompts import FewShotPromptTemplate

# 示例数据
examples = [
    {
        "context": "Python 是由 Guido van Rossum 创建的编程语言，发布于 1991 年。",
        "question": "Python 是什么时候发布的？",
        "answer": "Python 发布于 1991 年 [文档1]。"
    },
    {
        "context": "机器学习是人工智能的一个子领域。深度学习是机器学习的一个分支。",
        "question": "深度学习和人工智能有什么关系？",
        "answer": "深度学习是机器学习的一个分支 [文档1]，而机器学习是人工智能的一个子领域 [文档1]，因此深度学习属于人工智能领域。"
    },
    {
        "context": "该公司的产品价格在 100-500 元之间。",
        "question": "公司去年营收是多少？",
        "answer": "文档中没有提供关于公司去年营收的信息。"
    }
]

# 创建示例模板
example_prompt = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="文档：{context}\n问题：{question}\n回答：{answer}"
)

# 创建 Few-shot Prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请根据文档回答问题，遵循以下示例格式：\n",
    suffix="\n文档：{context}\n问题：{question}\n回答：",
    input_variables=["context", "question"],
    example_separator="\n\n"
)

# 使用
formatted_prompt = few_shot_prompt.format(
    context="RAG 是一种将检索和生成结合的技术。",
    question="什么是 RAG？"
)
print(formatted_prompt)
```

---

## 🔄 上下文融合策略

### 上下文组织方式

当检索到多个文档时，如何组织这些上下文？LangChain 提供了四种主要策略：

```
上下文融合策略：

1. Stuff（填充）
   └── 将所有文档直接拼接到 Prompt 中

2. Map-Reduce（映射归约）
   └── 先对每个文档单独处理，再合并结果

3. Refine（精炼）
   └── 迭代优化，逐步完善答案

4. Map-Rerank（映射重排）
   └── 对每个文档评分，选择最佳答案
```

### 1. Stuff 策略

最简单直接的方式，将所有检索文档放入一个 Prompt：

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Stuff 策略（默认）
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 指定策略
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

result = qa_chain({"query": "什么是机器学习？"})
print(result["result"])
```

**适用场景**：
- 文档数量少（通常 < 4 个）
- 文档内容较短
- 需要综合多个文档回答

**限制**：
- 受 LLM 上下文窗口限制
- 文档过多可能导致信息丢失

### 2. Map-Reduce 策略

先分别处理每个文档，再合并结果：

```python
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Map 阶段：对每个文档生成摘要
map_template = """对以下文档内容进行摘要，提取与问题相关的信息：

文档：{docs}
问题：{question}

相关摘要："""

map_prompt = PromptTemplate(template=map_template, input_variables=["docs", "question"])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce 阶段：合并所有摘要
reduce_template = """基于以下多个文档摘要，回答问题：

摘要：
{doc_summaries}

问题：{question}

综合回答："""

reduce_prompt = PromptTemplate(
    template=reduce_template, 
    input_variables=["doc_summaries", "question"]
)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# 组合成 Map-Reduce 链
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_document_chain=reduce_chain,
    document_variable_name="docs"
)

# 使用 RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True
)
```

**适用场景**：
- 文档数量多
- 需要并行处理
- 文档之间相对独立

**优缺点**：
- ✅ 可处理大量文档
- ✅ 可并行处理，速度快
- ❌ 可能丢失文档间的关联信息
- ❌ API 调用次数多，成本高

### 3. Refine 策略

迭代式优化答案：

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Refine 策略的工作流程：
# 文档1 → 初始答案
# 文档2 + 初始答案 → 优化答案
# 文档3 + 优化答案 → 进一步优化的答案
# ...
```

**工作流程**：
```
初始问题
    │
    ▼
┌─────────────┐
│   文档 1    │ ──→ 初始答案
└─────────────┘
    │
    ▼
┌─────────────┐
│   文档 2    │ ──→ 基于初始答案优化
└─────────────┘
    │
    ▼
┌─────────────┐
│   文档 3    │ ──→ 继续优化
└─────────────┘
    │
    ▼
  最终答案
```

**适用场景**：
- 需要完整、详尽的答案
- 文档之间有补充关系
- 答案质量优先于速度

### 4. Map-Rerank 策略

对每个文档单独回答并打分：

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_rerank",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 每个文档生成一个答案并给出置信度分数
# 选择分数最高的答案作为最终答案
```

**适用场景**：
- 只需要一个最相关的文档
- 文档之间关联性弱
- 需要知道答案的置信度

### 策略选择指南

```
文档数量？
├── 少量 (< 5个)
│   └── Stuff 策略
│
├── 中等 (5-10个)
│   ├── 需要综合回答？ ──→ Refine 策略
│   └── 文档独立？     ──→ Map-Reduce 策略
│
└── 大量 (> 10个)
    ├── 需要最相关答案？ ──→ Map-Rerank 策略
    └── 需要综合回答？   ──→ Map-Reduce + Reduce 文档数
```

---

## 🛠️ 生成器实现实战

### 完整的 RAG 生成器实现

```python
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGGenerator:
    """完整的 RAG 生成器实现"""
    
    def __init__(
        self, 
        model_name: str = "gpt-4",
        temperature: float = 0,
        max_tokens: int = 1000
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.prompt_template = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """创建 Prompt 模板"""
        template = """你是一个专业的问答助手。请基于提供的文档回答用户问题。

文档内容：
{context}

用户问题：{question}

回答要求：
1. 仅使用文档中的信息回答，不要编造
2. 如果文档中没有相关信息，请诚实说明
3. 回答要准确、简洁
4. 使用 [文档X] 格式标注信息来源

回答："""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def format_context(self, documents: List[Document]) -> str:
        """格式化检索到的文档"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content
            formatted_docs.append(f"[文档{i}] 来源：{source}\n{content}")
        return "\n\n".join(formatted_docs)
    
    def generate(
        self, 
        question: str, 
        documents: List[Document]
    ) -> Dict:
        """生成回答"""
        context = self.format_context(documents)
        
        response = self.chain.run(
            context=context,
            question=question
        )
        
        return {
            "answer": response,
            "source_documents": documents
        }
    
    def generate_with_history(
        self, 
        question: str, 
        documents: List[Document],
        chat_history: List[Dict]
    ) -> Dict:
        """带对话历史的生成"""
        history_text = self._format_history(chat_history)
        
        template = """你是一个专业的问答助手。请基于提供的文档和对话历史回答用户问题。

文档内容：
{context}

对话历史：
{chat_history}

当前问题：{question}

回答要求：
1. 参考对话历史理解问题的完整意图
2. 使用文档中的信息回答
3. 保持回答的自然和连贯

回答："""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        context = self.format_context(documents)
        
        response = chain.run(
            context=context,
            chat_history=history_text,
            question=question
        )
        
        return {
            "answer": response,
            "source_documents": documents
        }
    
    def _format_history(self, chat_history: List[Dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in chat_history:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}：{msg['content']}")
        return "\n".join(formatted)


# 使用示例
generator = RAGGenerator(model_name="gpt-4")

# 假设已有检索到的文档
retrieved_docs = [
    Document(
        page_content="RAG 是检索增强生成技术，由 Facebook AI 提出。",
        metadata={"source": "AI论文综述"}
    ),
    Document(
        page_content="RAG 结合了信息检索和文本生成的优势。",
        metadata={"source": "技术博客"}
    )
]

result = generator.generate(
    question="什么是 RAG？",
    documents=retrieved_docs
)

print("回答：", result["answer"])
print("\n来源文档：")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata['source']}")
```

### LlamaIndex 生成器实现

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.response_synthesizers import get_response_synthesizer

# 1. 准备数据
documents = SimpleDirectoryReader("./docs").load_data()

# 2. 配置 LLM
llm = OpenAI(model="gpt-4", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)

# 3. 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

# 4. 自定义响应合成器
from llama_index.response_synthesizers import ResponseMode

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,  # 或 REFINE, TREE_SUMMARIZE
    llm=llm,
    verbose=True
)

# 5. 创建查询引擎
query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer,
    similarity_top_k=3
)

# 6. 查询
response = query_engine.query("什么是 RAG 技术？")
print(response.response)

# 查看来源
for node in response.source_nodes:
    print(f"来源: {node.node.metadata}")
    print(f"内容: {node.node.text[:100]}...")
```

---

## ⚡ 生成器优化技巧

### 1. 控制输出长度

```python
# 方法一：在 Prompt 中明确长度要求
prompt = PromptTemplate(
    template="""基于以下文档回答问题，回答不超过 {max_words} 字。

文档：{context}
问题：{question}

回答：""",
    input_variables=["context", "question", "max_words"]
)

# 方法二：使用 LLM 参数
llm = ChatOpenAI(
    model="gpt-4",
    max_tokens=200  # 限制输出 token 数
)
```

### 2. 提高事实准确性

```python
# 使用更严格的 Prompt
STRICT_PROMPT = PromptTemplate(
    template="""你是一个严谨的问答助手。

文档内容：
{context}

问题：{question}

重要规则：
1. 只使用文档中明确提到的信息
2. 不要进行推测或推断
3. 如果文档信息不足以回答，必须说"根据提供的文档，无法回答这个问题"
4. 标注每个事实的来源

回答：""",
    input_variables=["context", "question"]
)

# 降低温度以减少随机性
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### 3. 处理冲突信息

```python
CONFLICT_HANDLING_PROMPT = PromptTemplate(
    template="""分析以下文档，回答问题。如果文档间存在冲突信息，请指出。

文档：
{context}

问题：{question}

请按以下格式回答：
1. 主要回答：基于大多数文档的一致信息
2. 信息冲突：如果存在冲突，列出不同观点及其来源
3. 推荐答案：基于可信度或最新信息给出最终答案

回答：""",
    input_variables=["context", "question"]
)
```

### 4. 结构化输出

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 定义输出结构
response_schemas = [
    ResponseSchema(name="answer", description="问题的回答"),
    ResponseSchema(name="confidence", description="置信度：高/中/低"),
    ResponseSchema(name="sources", description="信息来源列表"),
    ResponseSchema(name="limitations", description="回答的局限性")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 创建带格式说明的 Prompt
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""基于文档回答问题。

文档：{context}

问题：{question}

{format_instructions}""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions}
)

# 解析输出
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(context=context, question=question)
parsed_output = output_parser.parse(response)

print(f"回答: {parsed_output['answer']}")
print(f"置信度: {parsed_output['confidence']}")
print(f"来源: {parsed_output['sources']}")
```

---

## 📊 生成质量评估

### 自动评估指标

```python
from typing import List
import numpy as np

def evaluate_answer_quality(
    generated_answer: str,
    reference_answer: str,
    context: str
) -> dict:
    """评估生成答案的质量"""
    
    # 1. 忠实度（Faithfulness）：答案是否基于上下文
    # 使用 NLI 模型或 LLM 评估
    faithfulness = evaluate_faithfulness(generated_answer, context)
    
    # 2. 相关性（Relevance）：答案是否回答了问题
    # 需要原始问题
    relevance = evaluate_relevance(generated_answer, reference_answer)
    
    # 3. 完整性（Completeness）：是否覆盖了关键信息
    completeness = evaluate_completeness(generated_answer, reference_answer)
    
    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "completeness": completeness
    }

def evaluate_faithfulness(answer: str, context: str) -> float:
    """评估答案对上下文的忠实度"""
    # 简化版：检查答案中的关键陈述是否在上下文中
    # 生产环境应使用 NLI 模型
    
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 将答案分解为句子
    answer_sentences = answer.split('。')
    context_embedding = model.encode(context, convert_to_tensor=True)
    
    scores = []
    for sentence in answer_sentences:
        if sentence.strip():
            sent_embedding = model.encode(sentence, convert_to_tensor=True)
            score = util.cos_sim(sent_embedding, context_embedding).item()
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0
```

### 使用 RAGAS 评估框架

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # 忠实度
    answer_relevancy,       # 答案相关性
    context_relevancy,      # 上下文相关性
    context_recall          # 上下文召回率
)

# 准备评估数据
# 需要包含：question, answer, contexts, ground_truth
eval_data = {
    "question": ["什么是 RAG？", "如何选择向量数据库？"],
    "answer": [
        "RAG 是检索增强生成技术...",
        "选择向量数据库需要考虑..."
    ],
    "contexts": [
        ["RAG 是一种将检索和生成结合的技术..."],
        ["向量数据库的选择取决于规模..."]
    ],
    "ground_truth": [
        "RAG 是 Retrieval-Augmented Generation 的缩写...",
        "向量数据库选择应考虑性能、规模、成本..."
    ]
}

# 评估
results = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_relevancy]
)

print("评估结果：")
for metric, score in results.items():
    print(f"  {metric}: {score:.4f}")
```

---

## 🔗 相关内容

- [检索器模块](./retriever.md) - 学习如何高效检索文档
- [高级 RAG 技术](./advanced-rag.md) - 重排序、多跳检索等进阶内容
- [RAG vs SFT 对比](./rag-vs-sft.md) - 深入了解两种方案的适用场景
