# LlamaIndex 高级用法与优化

本章将介绍 LlamaIndex 的高级功能和生产级优化策略，帮助你构建高性能、可扩展的 RAG 应用。

---

## 📌 Agent 与工具调用

LlamaIndex 提供了强大的 Agent 框架，支持 LLM 自主决策和工具调用。

### Agent 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LlamaIndex Agent 架构                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                      ┌─────────────┐                                   │
│                      │  用户输入   │                                    │
│                      └─────────────┘                                   │
│                            │                                           │
│                            ↓                                           │
│                      ┌─────────────┐                                   │
│                      │    Agent    │                                   │
│                      │  (LLM决策)  │                                   │
│                      └─────────────┘                                   │
│                            │                                           │
│           ┌────────────────┼────────────────┐                         │
│           ↓                ↓                ↓                         │
│     ┌───────────┐    ┌───────────┐    ┌───────────┐                  │
│     │  Tool 1   │    │  Tool 2   │    │  Tool N   │                  │
│     │ RAG检索   │    │ 网络搜索  │    │ 数据库查询 │                  │
│     └───────────┘    └───────────┘    └───────────┘                  │
│           │                │                │                         │
│           └────────────────┼────────────────┘                         │
│                            ↓                                           │
│                      ┌─────────────┐                                   │
│                      │  整合响应   │                                    │
│                      └─────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### ReAct Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# ========== 定义自定义工具 ==========

def multiply(a: int, b: int) -> int:
    """两个数相乘"""
    return a * b

def add(a: int, b: int) -> int:
    """两个数相加"""
    return a + b

# 创建函数工具
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# ========== 创建 RAG 查询工具 ==========

rag_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="knowledge_base",
        description="知识库检索工具，用于查询技术文档和产品信息"
    )
)

# ========== 创建 Agent ==========

agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool, rag_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True  # 显示推理过程
)

# ========== 执行查询 ==========

response = agent.chat("文档中提到的参数值是多少？然后把它乘以 2")
print(response)
```

### Agent 类型对比

| Agent 类型 | 特点 | 适用场景 |
|------------|------|----------|
| **ReActAgent** | 推理+行动循环，透明可控 | 通用场景 |
| **OpenAIAgent** | 原生函数调用，速度快 | OpenAI 模型 |
| **FunctionCallingAgent** | 支持更多模型 | 非 OpenAI 模型 |

```python
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.anthropic import Anthropic

# 使用 Anthropic 模型
agent = FunctionCallingAgent.from_tools(
    tools=[multiply_tool, add_tool],
    llm=Anthropic(model="claude-3-sonnet"),
    verbose=True
)
```

### 工具类型

```python
from llama_index.core.tools import (
    FunctionTool,           # 函数工具
    QueryEngineTool,        # 查询引擎工具
    ToolMetadata           # 工具元数据
)

# 1. 函数工具 - 包装任意 Python 函数
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    # 实现搜索逻辑
    return "搜索结果..."

search_tool = FunctionTool.from_defaults(
    fn=search_web,
    name="web_search",
    description="搜索网络获取实时信息"
)

# 2. 查询引擎工具 - 包装 RAG 查询
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

rag_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="doc_search",
        description="搜索内部文档知识库"
    )
)

# 3. 异步函数工具
async def async_search(query: str) -> str:
    """异步搜索"""
    import asyncio
    await asyncio.sleep(1)
    return f"异步搜索结果: {query}"

async_tool = FunctionTool.from_defaults(fn=async_search)
```

---

## 🔄 流式输出与异步

### 流式查询

```python
from llama_index.core import VectorStoreIndex

# 创建流式查询引擎
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

# 执行流式查询
streaming_response = query_engine.query("总结文档的主要内容")

# 方式1：迭代器输出
for text in streaming_response.response_gen:
    print(text, end="", flush=True)

# 方式2：获取完整响应
full_response = streaming_response.response
```

### 流式对话

```python
# 创建流式对话引擎
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    streaming=True
)

# 流式对话
response = chat_engine.stream_chat("文档的核心观点是什么？")

for text in response.response_gen:
    print(text, end="", flush=True)
```

### 异步操作

```python
import asyncio
from llama_index.core import VectorStoreIndex

async def async_rag_workflow():
    """异步 RAG 工作流"""
    
    # 异步构建索引
    index = await VectorStoreIndex(documents).build_index_from_nodes()
    
    # 异步查询
    query_engine = index.as_query_engine()
    response = await query_engine.aquery("查询内容")
    
    return response

# 运行异步任务
response = asyncio.run(async_rag_workflow())

# 异步流式查询
async def async_stream_query():
    query_engine = index.as_query_engine(streaming=True)
    response = await query_engine.aquery("查询内容")
    
    async for text in response.async_response_gen():
        print(text, end="", flush=True)

asyncio.run(async_stream_query())
```

---

## ⚡ 性能优化

### 1. 嵌入缓存

```python
from llama_index.core import Settings
from llama_index.core.embeddings.cache import BaseEmbeddingCache
from llama_index.embeddings.openai import OpenAIEmbedding

# 使用内存缓存
from llama_index.embeddings.cache import InMemoryEmbeddingCache

cache = InMemoryEmbeddingCache()
embed_model = OpenAIEmbedding()
embed_model.cache = cache
Settings.embed_model = embed_model

# 使用 Redis 缓存（生产推荐）
from llama_index.storage.cache.redis import RedisCache

redis_cache = RedisCache(redis_url="redis://localhost:6379")
embed_model.cache = redis_cache
```

### 2. 批量处理

```python
from llama_index.core import VectorStoreIndex, Document

# 批量插入优化
def batch_insert_documents(index, documents, batch_size=100):
    """批量插入文档以提高效率"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        for doc in batch:
            index.insert(doc)
        print(f"已处理 {min(i + batch_size, len(documents))}/{len(documents)}")

# 批量嵌入
async def batch_embed(texts: list[str], embed_model, batch_size=100):
    """批量计算嵌入"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await embed_model.aget_text_embedding_batch(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### 3. 向量数据库优化

```python
# Chroma 优化配置
import chromadb
from chromadb.config import Settings as ChromaSettings

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# 创建优化的集合
collection = chroma_client.get_or_create_collection(
    name="optimized_collection",
    metadata={
        "hnsw:space": "cosine",      # 相似度度量
        "hnsw:M": 16,                 # HNSW 参数
        "hnsw:ef_construction": 200   # 构建参数
    }
)

# Milvus 优化配置
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="optimized",
    dim=1536,
    index_params={
        "index_type": "IVF_FLAT",    # 索引类型
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    }
)
```

### 4. 查询优化

```python
# 减少不必要的检索
query_engine = index.as_query_engine(
    similarity_top_k=5,    # 适度减少 top_k
    streaming=True         # 启用流式减少等待感
)

# 使用更快的模型
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o-mini")  # 比 gpt-4 更快更便宜

# 使用本地模型加速
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama3", request_timeout=60.0)
```

### 5. 索引持久化

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引避免重复构建
def save_index(index, persist_dir="./storage"):
    index.storage_context.persist(persist_dir=persist_dir)

# 加载已保存的索引
def load_index(persist_dir="./storage"):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

# 使用示例
import os

persist_dir = "./storage"
if os.path.exists(persist_dir):
    index = load_index(persist_dir)
else:
    index = VectorStoreIndex.from_documents(documents)
    save_index(index, persist_dir)
```

---

## 🏭 生产部署

### FastAPI 部署

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import uvicorn

app = FastAPI(title="RAG API")

# 初始化索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

class QueryRequest(BaseModel):
    question: str
    session_id: str = None

@app.post("/query")
async def query(request: QueryRequest):
    """非流式查询"""
    response = chat_engine.chat(request.question)
    return {"response": response.response}

@app.post("/stream")
async def stream_query(request: QueryRequest):
    """流式查询"""
    response = chat_engine.stream_chat(request.question)
    
    async def generate():
        for text in response.response_gen:
            yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
    depends_on:
      - redis
      - chroma

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  chroma:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - ./chroma_data:/chroma/data
```

### 水平扩展

```python
# 使用 Redis 作为共享缓存和会话存储
from redis import Redis
import json

redis_client = Redis(host='redis', port=6379)

def get_or_create_index(session_id: str):
    """从 Redis 获取或创建索引"""
    cache_key = f"index:{session_id}"
    
    # 尝试从缓存获取
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 创建新索引
    index = create_index()
    
    # 存储到 Redis
    redis_client.setex(cache_key, 3600, json.dumps(index))
    
    return index

def store_chat_history(session_id: str, messages: list):
    """存储对话历史"""
    key = f"chat:{session_id}"
    redis_client.setex(key, 86400, json.dumps(messages))  # 24小时过期

def get_chat_history(session_id: str) -> list:
    """获取对话历史"""
    key = f"chat:{session_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else []
```

---

## 🔧 自定义组件

### 自定义 LLM

```python
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from typing import Any

class MyCustomLLM(CustomLLM):
    """自定义 LLM 实现"""
    
    def __init__(self, model_name: str = "custom-model"):
        self.model_name = model_name
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            model_name=self.model_name
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 实现你的模型调用逻辑
        response_text = self._call_model(prompt)
        return CompletionResponse(text=response_text)
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        # 实现流式输出
        for token in self._stream_call_model(prompt):
            yield CompletionResponse(text=token, delta=token)
    
    def _call_model(self, prompt: str) -> str:
        # 调用你的模型 API
        return "模型响应"
    
    def _stream_call_model(self, prompt: str):
        # 流式调用
        yield "流式响应"

# 使用自定义 LLM
from llama_index.core import Settings

Settings.llm = MyCustomLLM()
```

### 自定义嵌入模型

```python
from llama_index.core.embeddings import BaseEmbedding
from typing import List

class MyEmbedding(BaseEmbedding):
    """自定义嵌入模型"""
    
    def __init__(self, model_name: str = "custom-embedding"):
        super().__init__()
        self.model_name = model_name
    
    def _get_query_embedding(self, query: str) -> List[float]:
        # 实现查询嵌入
        return self._embed(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        # 实现文本嵌入
        return self._embed(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 批量嵌入
        return [self._embed(text) for text in texts]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        # 异步查询嵌入
        return self._embed(query)
    
    def _embed(self, text: str) -> List[float]:
        # 调用你的嵌入模型
        return [0.1] * 768  # 示例：返回 768 维向量

# 使用自定义嵌入模型
Settings.embed_model = MyEmbedding()
```

### 自定义检索器

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List

class MyRetriever(BaseRetriever):
    """自定义检索器"""
    
    def __init__(self, index, top_k: int = 5):
        self.index = index
        self.top_k = top_k
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 实现你的检索逻辑
        query = query_bundle.query_str
        
        # 例如：结合多种检索方式
        vector_nodes = self._vector_search(query)
        keyword_nodes = self._keyword_search(query)
        
        # 合并和重排序
        merged = self._merge_results(vector_nodes, keyword_nodes)
        
        return merged[:self.top_k]
    
    def _vector_search(self, query: str) -> List[NodeWithScore]:
        # 向量检索
        pass
    
    def _keyword_search(self, query: str) -> List[NodeWithScore]:
        # 关键词检索
        pass
    
    def _merge_results(self, *results) -> List[NodeWithScore]:
        # 合并结果
        pass
```

---

## 📊 监控与调试

### 日志配置

```python
import logging
from llama_index.core import set_global_handler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# LlamaIndex 调试模式
set_global_handler("simple")  # 简单输出
# 或
set_global_handler("debug")   # 详细调试信息
```

### 回调监控

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# 创建调试处理器
debug_handler = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([debug_handler])

# 设置全局回调
from llama_index.core import Settings
Settings.callback_manager = callback_manager

# 执行查询后查看事件
response = query_engine.query("查询内容")

# 打印事件时间线
for event in debug_handler.get_event_pairs():
    print(f"{event[0].type_}: {event[1].time - event[0].time:.3f}s")
```

### Langfuse 集成

```python
from langfuse.llama_index import LlamaIndexCallbackHandler

# 配置 Langfuse 监控
langfuse_handler = LlamaIndexCallbackHandler(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"
)

Settings.callback_manager = CallbackManager([langfuse_handler])

# 执行查询会自动记录到 Langfuse
response = query_engine.query("查询内容")
```

---

## 💡 常见问题

### Q1: 如何处理大量文档？

```python
# 使用增量索引
def incremental_index(new_docs, persist_dir):
    if os.path.exists(persist_dir):
        index = load_index(persist_dir)
    else:
        index = VectorStoreIndex([])
    
    for doc in new_docs:
        index.insert(doc)
    
    index.storage_context.persist(persist_dir)
```

### Q2: 如何控制成本？

```python
# 1. 使用更便宜的模型
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. 减少检索数量
query_engine = index.as_query_engine(similarity_top_k=3)

# 3. 使用缓存
Settings.embed_model.cache = InMemoryEmbeddingCache()
```

### Q3: 如何处理长文档？

```python
# 使用层级检索
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import RecursiveRetriever

# 层级分块
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

# 递归检索，自动获取父节点上下文
```

---

## 📚 参考资料

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [Agent 指南](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- [自定义组件](https://docs.llamaindex.ai/en/stable/module_guides/models/)
- [生产部署](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

---

*恭喜你完成 LlamaIndex 学习！🎉 返回 [LlamaIndex 学习指南](./index.md)*
