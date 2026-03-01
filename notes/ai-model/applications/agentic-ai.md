# AI智能体 (AI Agent)

AI智能体是能够自主感知环境、做出决策并执行动作以实现目标的智能系统。随着大语言模型的发展，基于LLM的智能体成为实现通用人工智能（AGI）的重要路径。本笔记将系统介绍AI Agent的核心概念、架构设计、关键技术及主流框架。

## 章节概述

```
AI智能体
├── Agent概述
│   ├── 定义与特征
│   └── 核心组件（感知、大脑、行动）
├── Agent架构设计
│   ├── ReAct架构
│   ├── Plan-and-Execute架构
│   └── Multi-Agent架构
├── RAG技术原理
│   ├── 检索机制
│   ├── 增强生成
│   └── 向量数据库
├── 工具调用与函数执行
│   ├── Function Calling
│   └── 工具选择策略
├── 记忆系统设计
│   ├── 短期记忆
│   └── 长期记忆
├── Agent框架
│   ├── LangChain
│   ├── AutoGPT
│   └── CrewAI
└── 评估与安全
    ├── 评估指标
    └── 安全机制
```

---

## 一、AI Agent概述

### 1.1 定义与特征

**AI Agent** 是一个能够自主感知环境、进行推理决策、执行动作以达成目标的智能实体。与传统的被动响应系统不同，Agent具有**主动性**和**自主性**。

#### 核心特征

| 特征 | 描述 | 示例 |
|------|------|------|
| **自主性** | 无需人工干预即可运行 | 自动完成复杂任务链 |
| **反应性** | 感知环境变化并响应 | 根据用户反馈调整策略 |
| **主动性** | 主动采取行动达成目标 | 主动搜索信息补充知识 |
| **社交性** | 与人类或其他Agent交互 | 多Agent协作完成任务 |

#### Agent vs 传统LLM

```
传统LLM应用:
用户输入 → LLM推理 → 输出文本

AI Agent:
用户输入 → [感知 → 推理 → 行动] 循环 → 达成目标
                    ↑____反馈____↓
```

### 1.2 核心组件

AI Agent通常包含四个核心组件：

#### 📌 感知模块 (Perception)

负责接收和理解外部输入，包括：
- 文本输入（用户指令）
- 多模态输入（图像、音频）
- 环境状态（系统状态、执行结果）

#### 🧠 大脑模块 (Brain)

核心推理引擎，负责：
- 意图理解与任务规划
- 推理决策
- 知识检索与整合

#### 🛠️ 行动模块 (Action)

执行具体操作，包括：
- 工具调用（API、函数执行）
- 文本生成
- 外部系统交互

#### 💾 记忆模块 (Memory)

存储和检索信息：
- 短期记忆（对话上下文）
- 长期记忆（知识库、向量存储）

---

## 二、Agent架构设计

### 2.1 ReAct架构

**ReAct (Reasoning + Acting)** 是最经典的Agent架构，将推理与行动交织进行。

#### 架构原理

```
Thought → Action → Observation → Thought → ...
```

**核心思想**：在每一步中，Agent先进行推理（Thought），然后选择执行动作（Action），观察执行结果（Observation），再基于结果进行下一步推理。

#### 数学建模

Agent的决策过程可建模为马尔可夫决策过程（MDP）：

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle
$$

其中：
- $\mathcal{S}$：状态空间（环境状态、对话历史）
- $\mathcal{A}$：动作空间（可用工具集合）
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$：状态转移函数
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$：奖励函数
- $\gamma$：折扣因子

Agent的目标是学习最优策略 $\pi^*$：

$$
\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t)\right]
$$

#### 代码实现

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """动作类型枚举"""
    SEARCH = "search"
    CALCULATE = "calculate"
    TOOL_CALL = "tool_call"
    FINISH = "finish"

@dataclass
class AgentStep:
    """Agent执行步骤"""
    thought: str          # 推理过程
    action: str           # 执行动作
    action_input: Dict    # 动作参数
    observation: str      # 观察结果

class ReActAgent:
    """ReAct架构Agent实现"""
    
    def __init__(self, llm, tools: Dict[str, callable], max_iterations: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.history: List[AgentStep] = []
    
    def think(self, query: str) -> AgentStep:
        """推理阶段：生成Thought和Action"""
        prompt = self._build_prompt(query)
        response = self.llm.generate(prompt)
        
        # 解析LLM输出
        thought, action, action_input = self._parse_response(response)
        return AgentStep(thought, action, action_input, "")
    
    def act(self, step: AgentStep) -> str:
        """行动阶段：执行工具调用"""
        if step.action == "finish":
            return step.action_input.get("answer", "")
        
        tool = self.tools.get(step.action)
        if tool:
            return str(tool(**step.action_input))
        return f"Error: Unknown tool {step.action}"
    
    def run(self, query: str) -> str:
        """完整执行流程"""
        for _ in range(self.max_iterations):
            # 推理
            step = self.think(query)
            
            # 行动
            observation = self.act(step)
            step.observation = observation
            self.history.append(step)
            
            # 检查是否完成
            if step.action == "finish":
                return observation
            
            # 更新query为当前状态
            query = self._update_context(query, step)
        
        return "Max iterations reached without completion."
    
    def _build_prompt(self, query: str) -> str:
        """构建ReAct提示词"""
        history_str = "\n".join([
            f"Thought: {s.thought}\nAction: {s.action}\nObservation: {s.observation}"
            for s in self.history
        ])
        
        return f"""Answer the following question using the ReAct framework.

Available tools: {list(self.tools.keys())}

Previous steps:
{history_str}

Question: {query}

Follow this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: {{"param": "value"}}
"""

# 使用示例
def calculator(expression: str) -> float:
    """简单计算器工具"""
    return eval(expression)

def search(query: str) -> str:
    """模拟搜索工具"""
    return f"Search results for: {query}"

tools = {
    "calculator": calculator,
    "search": search
}

# agent = ReActAgent(llm=your_llm, tools=tools)
# result = agent.run("What is 25 * 4 + 10?")
```

### 2.2 Plan-and-Execute架构

**Plan-and-Execute** 架构将任务分解为规划与执行两个独立阶段，适合复杂任务。

#### 架构流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   输入任务   │ ──→ │   规划器     │ ──→ │   执行器     │
└─────────────┘     │ (Planner)   │     │ (Executor)  │
                    └─────────────┘     └─────────────┘
                           │                   │
                           ↓                   ↓
                    ┌─────────────┐     ┌─────────────┐
                    │  任务列表    │ ──→ │  执行结果    │
                    │ [Task1, ...]│     │  [R1, ...]  │
                    └─────────────┘     └─────────────┘
```

#### 数学建模

规划问题可形式化为：

$$
\text{Plan} = \arg\max_{p \in \mathcal{P}} P(p | \text{Task}, \mathcal{K})
$$

其中：
- $\mathcal{P}$：所有可能的计划空间
- $\mathcal{K}$：领域知识
- $P(p | \text{Task}, \mathcal{K})$：给定任务和知识下计划的条件概率

#### 代码实现

```python
from typing import List, Tuple
import json

class PlanAndExecuteAgent:
    """Plan-and-Execute架构Agent"""
    
    def __init__(self, planner_llm, executor_llm, tools: Dict[str, callable]):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
    
    def plan(self, task: str) -> List[Dict]:
        """规划阶段：将任务分解为子任务"""
        prompt = f"""Decompose the following task into subtasks.

Task: {task}

Output format (JSON array):
[
    {{"step": 1, "description": "...", "tool": "...", "params": {{}}}},
    ...
]
"""
        response = self.planner.generate(prompt)
        return json.loads(response)
    
    def execute_step(self, step: Dict) -> str:
        """执行单个子任务"""
        tool_name = step.get("tool")
        params = step.get("params", {})
        
        if tool_name in self.tools:
            return self.tools[tool_name](**params)
        return self.executor.generate(step["description"])
    
    def run(self, task: str) -> Tuple[str, List[Dict]]:
        """完整执行流程"""
        # 1. 规划
        plan = self.plan(task)
        results = []
        
        # 2. 执行
        for step in plan:
            result = self.execute_step(step)
            step["result"] = result
            results.append(step)
        
        # 3. 汇总
        summary = self._summarize(task, results)
        return summary, results
    
    def _summarize(self, task: str, results: List[Dict]) -> str:
        """汇总执行结果"""
        results_str = "\n".join([
            f"Step {r['step']}: {r['description']} -> {r.get('result', 'N/A')}"
            for r in results
        ])
        
        prompt = f"""Summarize the results for the original task.

Original Task: {task}

Execution Results:
{results_str}

Provide a comprehensive answer:
"""
        return self.executor.generate(prompt)

# 使用示例
# agent = PlanAndExecuteAgent(planner, executor, tools)
# result, steps = agent.run("Research and compare the top 3 AI frameworks")
```

### 2.3 Multi-Agent架构

**多智能体系统** 由多个协作Agent组成，每个Agent负责特定领域或任务。

#### 架构模式

```
                ┌──────────────┐
                │   协调器      │
                │ (Coordinator)│
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Agent A │   │ Agent B │   │ Agent C │
   │ (专家1)  │   │ (专家2)  │   │ (专家3)  │
   └─────────┘   └─────────┘   └─────────┘
        │              │              │
        └──────────────┴──────────────┘
                       ↓
                ┌──────────────┐
                │   共享记忆    │
                └──────────────┘
```

#### 协作模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **层级协作** | 上层Agent分发任务，下层执行 | 复杂工作流 |
| **对等协作** | Agent间平等交互协商 | 创意生成、辩论 |
| **竞争协作** | 多Agent竞争选出最优解 | 决策优化 |
| **接力协作** | Agent按顺序传递任务 | 流水线任务 |

#### 博弈论建模

多Agent协作可建模为合作博弈：

$$
v(S) = \text{联盟 } S \text{ 能获得的价值}
$$

核心概念：
- **夏普利值 (Shapley Value)**：衡量每个Agent的贡献

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[v(S \cup \{i\}) - v(S)]
$$

#### 代码实现

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Message:
    """Agent间通信消息"""
    sender: str
    receiver: str
    content: str
    metadata: Dict = None

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, role: str, llm):
        self.name = name
        self.role = role
        self.llm = llm
        self.memory: List[Message] = []
    
    @abstractmethod
    def process(self, message: Message) -> Message:
        """处理消息"""
        pass
    
    def receive(self, message: Message):
        """接收消息"""
        self.memory.append(message)

class CoordinatorAgent(BaseAgent):
    """协调器Agent"""
    
    def __init__(self, name: str, llm, agents: Dict[str, BaseAgent]):
        super().__init__(name, "coordinator", llm)
        self.agents = agents
    
    def process(self, message: Message) -> Message:
        """分发任务到合适的Agent"""
        # 分析任务，决定由哪个Agent处理
        task_type = self._classify_task(message.content)
        
        if task_type in self.agents:
            agent = self.agents[task_type]
            response = agent.process(message)
            return response
        
        return Message(
            sender=self.name,
            receiver=message.sender,
            content="Unable to route task to appropriate agent."
        )
    
    def _classify_task(self, task: str) -> str:
        """任务分类"""
        prompt = f"""Classify the task type:
Task: {task}
Types: {list(self.agents.keys())}
Output only the type name:"""
        return self.llm.generate(prompt).strip()

class ExpertAgent(BaseAgent):
    """专家Agent"""
    
    def __init__(self, name: str, role: str, llm, expertise: str):
        super().__init__(name, role, llm)
        self.expertise = expertise
    
    def process(self, message: Message) -> Message:
        """处理专业任务"""
        prompt = f"""You are a {self.expertise} expert.
Task: {message.content}
Provide a professional response:"""
        
        response = self.llm.generate(prompt)
        return Message(
            sender=self.name,
            receiver=message.sender,
            content=response
        )

class MultiAgentSystem:
    """多智能体系统"""
    
    def __init__(self, coordinator: CoordinatorAgent):
        self.coordinator = coordinator
        self.history: List[Message] = []
    
    def run(self, user_input: str) -> str:
        """执行用户请求"""
        message = Message(
            sender="user",
            receiver="coordinator",
            content=user_input
        )
        
        response = self.coordinator.process(message)
        self.history.append(message)
        self.history.append(response)
        
        return response.content

# 使用示例
# researcher = ExpertAgent("researcher", "expert", llm, "research")
# coder = ExpertAgent("coder", "expert", llm, "programming")
# coordinator = CoordinatorAgent("coordinator", llm, 
#                                {"research": researcher, "code": coder})
# system = MultiAgentSystem(coordinator)
# result = system.run("Research Python best practices")
```

---

## 三、RAG技术原理

### 3.1 检索增强生成概述

**RAG (Retrieval-Augmented Generation)** 通过检索外部知识库来增强LLM的生成能力，解决幻觉问题和知识时效性问题。

#### RAG流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Query   │ → │ Retriever│ → │  Reranker│ → │   LLM    │
│ (用户查询)│    │ (检索器)  │    │ (重排序)  │    │ (生成)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                      ↑
                ┌──────────┐
                │ Vector DB│
                │ (向量库)  │
                └──────────┘
```

### 3.2 向量检索原理

#### 文本嵌入

将文本映射到高维向量空间：

$$
\mathbf{e} = f_{\text{embed}}(\text{text}) \in \mathbb{R}^d
$$

常用嵌入模型：
- **Word2Vec**: $\mathbf{e}_w = \mathbf{W} \cdot \mathbf{1}_w$
- **BERT**: $\mathbf{e} = \text{BERT}_{\text{[CLS]}}(\text{text})$
- **Sentence-BERT**: $\mathbf{e} = \text{mean}(\text{BERT}(\text{text}))$

#### 相似度计算

**余弦相似度**：

$$
\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{i=1}^{d} q_i \cdot d_i}{\sqrt{\sum_{i=1}^{d} q_i^2} \cdot \sqrt{\sum_{i=1}^{d} d_i^2}}
$$

**欧氏距离**：

$$
\text{dist}(\mathbf{q}, \mathbf{d}) = \|\mathbf{q} - \mathbf{d}\|_2 = \sqrt{\sum_{i=1}^{d}(q_i - d_i)^2}
$$

**点积相似度**：

$$
\text{sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q}^T \mathbf{d} = \sum_{i=1}^{d} q_i \cdot d_i
$$

### 3.3 检索算法

#### 稠密检索 (Dense Retrieval)

使用双塔模型进行检索：

$$
P(d|q) = \frac{\exp(\text{sim}(f_q(q), f_d(d)))}{\sum_{d' \in \mathcal{D}} \exp(\text{sim}(f_q(q), f_d(d')))}
$$

训练目标（对比学习）：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+))}{\exp(\text{sim}(q, d^+)) + \sum_{d^- \in \mathcal{D}^-} \exp(\text{sim}(q, d^-))}
$$

#### 混合检索

结合关键词检索和语义检索：

$$
\text{score}_{\text{hybrid}} = \alpha \cdot \text{score}_{\text{BM25}} + (1-\alpha) \cdot \text{score}_{\text{dense}}
$$

**BM25算法**：

$$
\text{BM25}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

其中：
- $f(q_i, d)$：词$q_i$在文档$d$中的词频
- $k_1, b$：超参数
- $\text{avgdl}$：平均文档长度

### 3.4 代码实现

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    """文档类"""
    id: str
    content: str
    embedding: np.ndarray = None
    metadata: dict = None

class SimpleVectorStore:
    """简单向量存储"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, docs: List[Document]):
        """添加文档"""
        self.documents.extend(docs)
        if self.embeddings is None:
            self.embeddings = np.array([d.embedding for d in docs])
        else:
            new_embeddings = np.array([d.embedding for d in docs])
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """相似度检索"""
        # 计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(docs_norm, query_norm)
        
        # 获取top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [(self.documents[i], similarities[i]) for i in top_k_indices]

class RAGSystem:
    """RAG系统实现"""
    
    def __init__(self, embedder, llm, vector_store: SimpleVectorStore):
        self.embedder = embedder
        self.llm = llm
        self.vector_store = vector_store
    
    def index_documents(self, documents: List[str]):
        """索引文档"""
        docs = []
        for i, content in enumerate(documents):
            embedding = self.embedder.embed(content)
            docs.append(Document(
                id=f"doc_{i}",
                content=content,
                embedding=embedding
            ))
        self.vector_store.add_documents(docs)
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.similarity_search(query_embedding, k)
        return [doc for doc, score in results]
    
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """生成回答"""
        context = "\n\n".join([doc.content for doc in context_docs])
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.llm.generate(prompt)
    
    def query(self, question: str, k: int = 3) -> str:
        """完整RAG流程"""
        # 1. 检索
        relevant_docs = self.retrieve(question, k)
        
        # 2. 生成
        answer = self.generate(question, relevant_docs)
        
        return answer

# 高级RAG：带重排序
class RerankedRAG(RAGSystem):
    """带重排序的RAG"""
    
    def __init__(self, embedder, llm, vector_store, reranker):
        super().__init__(embedder, llm, vector_store)
        self.reranker = reranker
    
    def retrieve_with_rerank(self, query: str, k: int = 10, final_k: int = 3) -> List[Document]:
        """检索并重排序"""
        # 初始检索更多文档
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.similarity_search(query_embedding, k)
        
        # 重排序
        docs = [doc for doc, _ in results]
        rerank_scores = self.reranker.rerank(query, docs)
        
        # 按重排序分数排序
        reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in reranked[:final_k]]
    
    def query(self, question: str) -> str:
        """使用重排序的查询"""
        relevant_docs = self.retrieve_with_rerank(question)
        return self.generate(question, relevant_docs)
```

---

## 四、工具调用与函数执行

### 4.1 Function Calling原理

**Function Calling** 允许LLM调用外部工具/API，扩展其能力边界。

#### 工作流程

```
1. 定义工具 → 2. LLM决策 → 3. 执行工具 → 4. 返回结果 → 5. 生成响应
```

#### 工具定义格式

```json
{
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位"
            }
        },
        "required": ["city"]
    }
}
```

### 4.2 工具选择策略

#### 选择概率建模

给定工具集合 $\mathcal{T} = \{t_1, t_2, ..., t_n\}$，LLM选择工具的概率：

$$
P(t_i | q, \mathcal{T}) = \frac{\exp(f(q, t_i))}{\sum_{j=1}^{n} \exp(f(q, t_j))}
$$

其中 $f(q, t)$ 是查询与工具的匹配函数。

#### 参数生成

对于工具 $t$ 的参数 $\theta_t$，生成过程：

$$
P(\theta_t | q, t) = \prod_{k \in \text{params}(t)} P(\theta_k | q, t, \text{schema}_k)
$$

### 4.3 代码实现

```python
import json
from typing import List, Dict, Callable, Any
from dataclasses import dataclass
import inspect

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict
    function: Callable

class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, func: Callable = None, name: str = None, description: str = None):
        """注册工具（支持装饰器模式）"""
        def decorator(f: Callable):
            tool_name = name or f.__name__
            tool_desc = description or f.__doc__ or ""
            
            # 自动推断参数schema
            params_schema = self._infer_parameters(f)
            
            tool = Tool(
                name=tool_name,
                description=tool_desc,
                parameters=params_schema,
                function=f
            )
            self.tools[tool_name] = tool
            return f
        
        if func:
            return decorator(func)
        return decorator
    
    def _infer_parameters(self, func: Callable) -> Dict:
        """从函数签名推断参数schema"""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            param_type = "string"  # 默认类型
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                param_type = type_map.get(param.annotation, "string")
            
            properties[name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def get_tool_schemas(self) -> List[Dict]:
        """获取所有工具的schema"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def execute(self, tool_name: str, arguments: Dict) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        return tool.function(**arguments)

class ToolCallingAgent:
    """支持工具调用的Agent"""
    
    def __init__(self, llm, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry
    
    def run(self, query: str) -> str:
        """执行查询"""
        messages = [{"role": "user", "content": query}]
        tool_schemas = self.registry.get_tool_schemas()
        
        while True:
            # LLM决策
            response = self.llm.chat(
                messages=messages,
                tools=tool_schemas
            )
            
            # 检查是否需要调用工具
            if not response.tool_calls:
                return response.content
            
            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                result = self.registry.execute(tool_name, arguments)
                
                # 添加工具结果到消息历史
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

# 使用示例
registry = ToolRegistry()

@registry.register(description="计算数学表达式")
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

@registry.register(description="搜索网络信息")
def search_web(query: str, num_results: int = 5) -> List[str]:
    """搜索网络"""
    return [f"Result {i} for '{query}'" for i in range(num_results)]

@registry.register(description="获取当前时间")
def get_current_time(timezone: str = "UTC") -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# agent = ToolCallingAgent(llm, registry)
# result = agent.run("What is 123 * 456?")
```

---

## 五、记忆系统设计

### 5.1 记忆系统概述

Agent的记忆系统模拟人类记忆，分为**短期记忆**和**长期记忆**。

```
┌─────────────────────────────────────────────────┐
│                  Agent记忆系统                    │
├─────────────────────────────────────────────────┤
│  感知输入  ─→  短期记忆  ←→  长期记忆  ─→  输出   │
│               (工作记忆)      (知识存储)         │
│               容量有限        容量无限           │
│               快速访问        持久化             │
└─────────────────────────────────────────────────┘
```

### 5.2 短期记忆

#### 滑动窗口

最简单的短期记忆实现，保留最近 $k$ 轮对话：

$$
\text{Memory}_t = \{(u_{t-k+1}, r_{t-k+1}), ..., (u_t, r_t)\}
$$

#### 注意力机制记忆

使用注意力机制对历史信息加权：

$$
\mathbf{m}_t = \sum_{i=1}^{t} \alpha_{t,i} \cdot \mathbf{h}_i
$$

其中注意力权重：

$$
\alpha_{t,i} = \frac{\exp(\mathbf{q}_t^T \mathbf{h}_i)}{\sum_{j=1}^{t} \exp(\mathbf{q}_t^T \mathbf{h}_j)}
$$

#### Token预算控制

控制记忆的token数量：

$$
\text{Memory} = \text{SelectTopK}(\{m_i\}, \text{score}(m_i), \text{budget})
$$

### 5.3 长期记忆

#### 向量存储记忆

将记忆存储为向量，支持语义检索：

$$
\text{Recall}(q) = \text{TopK}_{\mathbf{m} \in \mathcal{M}} \text{sim}(\mathbf{q}, \mathbf{m})
$$

#### 记忆重要性评分

评估记忆的重要性：

$$
\text{Importance}(m) = \alpha \cdot \text{Recency}(m) + \beta \cdot \text{Frequency}(m) + \gamma \cdot \text{Relevance}(m)
$$

#### 记忆巩固

将短期记忆转换为长期记忆的过程：

$$
P(\text{consolidate} | m) = \sigma(\mathbf{w}^T \phi(m) + b)
$$

### 5.4 代码实现

```python
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    importance: float = 0.0
    access_count: int = 0
    metadata: Dict = field(default_factory=dict)

class ShortTermMemory:
    """短期记忆（滑动窗口 + 重要性排序）"""
    
    def __init__(self, max_tokens: int = 4096, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.memories: List[MemoryItem] = []
    
    def add(self, item: MemoryItem):
        """添加记忆"""
        self.memories.append(item)
        self._prune()
    
    def _prune(self):
        """修剪超出预算的记忆"""
        while self._estimate_tokens() > self.max_tokens and len(self.memories) > 1:
            # 移除最不重要的记忆
            min_idx = min(range(len(self.memories)), 
                         key=lambda i: self._score(self.memories[i]))
            self.memories.pop(min_idx)
    
    def _estimate_tokens(self) -> int:
        """估算token数量"""
        if self.tokenizer:
            text = " ".join(m.content for m in self.memories)
            return len(self.tokenizer.encode(text))
        return sum(len(m.content.split()) for m in self.memories)
    
    def _score(self, item: MemoryItem) -> float:
        """记忆保留分数"""
        recency = (datetime.now() - item.timestamp).total_seconds()
        return item.importance * 0.5 + item.access_count * 0.3 - recency * 0.2
    
    def get_context(self) -> str:
        """获取上下文"""
        return "\n".join([m.content for m in self.memories])

class LongTermMemory:
    """长期记忆（向量存储 + 重要性筛选）"""
    
    def __init__(self, embedder, similarity_threshold: float = 0.7):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.memories: List[MemoryItem] = []
        self.embeddings: np.ndarray = None
    
    def store(self, item: MemoryItem):
        """存储记忆"""
        # 检查是否与现有记忆重复
        if self._is_duplicate(item):
            return False
        
        self.memories.append(item)
        self._update_embeddings(item.embedding)
        return True
    
    def _is_duplicate(self, item: MemoryItem) -> bool:
        """检查重复"""
        if self.embeddings is None:
            return False
        
        similarity = np.max(np.dot(self.embeddings, item.embedding))
        return similarity > self.similarity_threshold
    
    def _update_embeddings(self, new_embedding: np.ndarray):
        """更新嵌入矩阵"""
        if self.embeddings is None:
            self.embeddings = new_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
    
    def recall(self, query: str, k: int = 5) -> List[MemoryItem]:
        """召回相关记忆"""
        if self.embeddings is None:
            return []
        
        query_embedding = self.embedder.embed(query)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 结合相似度和重要性
        scores = similarities * 0.7 + np.array([m.importance for m in self.memories]) * 0.3
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # 更新访问计数
        for idx in top_k_indices:
            self.memories[idx].access_count += 1
        
        return [self.memories[i] for i in top_k_indices]
    
    def forget(self, threshold: float = 0.1):
        """遗忘不重要的记忆"""
        self.memories = [m for m in self.memories if m.importance > threshold]
        self.embeddings = np.array([m.embedding for m in self.memories])

class AgentMemory:
    """Agent完整记忆系统"""
    
    def __init__(self, embedder, stm_config: Dict = None, ltm_config: Dict = None):
        self.embedder = embedder
        self.stm = ShortTermMemory(**(stm_config or {}))
        self.ltm = LongTermMemory(embedder, **(ltm_config or {}))
    
    def remember(self, content: str, importance: float = 0.5, metadata: Dict = None):
        """记忆信息"""
        embedding = self.embedder.embed(content)
        
        item = MemoryItem(
            id=f"mem_{datetime.now().timestamp()}",
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # 存入短期记忆
        self.stm.add(item)
        
        # 重要信息同时存入长期记忆
        if importance > 0.7:
            self.ltm.store(item)
    
    def recall(self, query: str, use_stm: bool = True, use_ltm: bool = True) -> List[str]:
        """回忆相关信息"""
        results = []
        
        if use_stm:
            results.extend([m.content for m in self.stm.memories])
        
        if use_ltm:
            ltm_memories = self.ltm.recall(query)
            results.extend([m.content for m in ltm_memories])
        
        return results
    
    def consolidate(self):
        """记忆巩固：将短期记忆转移到长期记忆"""
        for item in self.stm.memories:
            if item.importance > 0.5 and item.access_count > 2:
                self.ltm.store(item)
```

---

## 六、Agent框架

### 6.1 LangChain

**LangChain** 是最流行的LLM应用开发框架，提供丰富的组件和工具链。

#### 核心组件

| 组件 | 功能 | 说明 |
|------|------|------|
| **Model I/O** | 模型接口 | LLM、Chat Model、Embeddings |
| **Chains** | 链式调用 | 组合多个组件形成工作流 |
| **Agents** | 智能体 | 自主决策和工具调用 |
| **Memory** | 记忆系统 | 对话历史和状态管理 |
| **Retrieval** | 检索系统 | RAG相关组件 |

#### 代码示例

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 初始化LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 定义工具
def get_word_length(word: str) -> int:
    """返回单词长度"""
    return len(word)

tools = [
    Tool(
        name="get_word_length",
        func=get_word_length,
        description="Returns the length of a word"
    )
]

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 执行
# result = agent_executor.invoke({"input": "How many letters are in 'hello'?"})
```

### 6.2 AutoGPT

**AutoGPT** 是早期自主Agent的代表，能够自主设定目标、分解任务、循环执行。

#### 核心特点

- **自主目标设定**：根据用户意图自动分解目标
- **自我反思**：评估执行结果并调整策略
- **长期记忆**：支持持久化存储
- **文件操作**：直接操作文件系统

#### 简化实现

```python
from typing import List, Dict
import json

class AutoGPTAgent:
    """简化版AutoGPT"""
    
    def __init__(self, llm, tools: Dict, memory, max_cycles: int = 100):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_cycles = max_cycles
        self.history: List[Dict] = []
    
    def run(self, goal: str) -> str:
        """自主执行目标"""
        self.memory.remember(f"GOAL: {goal}", importance=1.0)
        
        for cycle in range(self.max_cycles):
            # 1. 思考下一步
            thought = self._think(goal)
            
            # 2. 选择动作
            action = self._decide_action(thought)
            
            # 3. 执行
            result = self._execute(action)
            
            # 4. 记录历史
            self.history.append({
                "cycle": cycle,
                "thought": thought,
                "action": action,
                "result": result
            })
            
            # 5. 记忆
            self.memory.remember(
                f"Action: {action['name']}, Result: {result}",
                importance=0.5
            )
            
            # 6. 检查是否完成
            if action["name"] == "task_complete":
                return result
        
        return "Max cycles reached"
    
    def _think(self, goal: str) -> str:
        """思考阶段"""
        context = self._build_context()
        
        prompt = f"""You are an autonomous AI agent.

GOAL: {goal}

CONTEXT:
{context}

HISTORY:
{json.dumps(self.history[-5:], indent=2)}

What should you think about next? Respond with your reasoning."""
        
        return self.llm.generate(prompt)
    
    def _decide_action(self, thought: str) -> Dict:
        """决策阶段"""
        prompt = f"""Based on your thought: {thought}

Available actions:
{list(self.tools.keys())}
- task_complete (finish the goal)

Choose an action and provide parameters in JSON format:
{{"name": "action_name", "params": {{}}}}"""
        
        response = self.llm.generate(prompt)
        return json.loads(response)
    
    def _execute(self, action: Dict) -> str:
        """执行动作"""
        if action["name"] == "task_complete":
            return action.get("params", {}).get("summary", "Task completed")
        
        tool = self.tools.get(action["name"])
        if tool:
            return tool(**action.get("params", {}))
        return "Unknown action"
    
    def _build_context(self) -> str:
        """构建上下文"""
        memories = self.memory.recall("", use_ltm=True)
        return "\n".join(memories[-10:])
```

### 6.3 CrewAI

**CrewAI** 是专注于多Agent协作的框架，支持角色扮演和团队协作。

#### 核心概念

| 概念 | 描述 |
|------|------|
| **Agent** | 具有角色、目标、工具的智能体 |
| **Task** | 需要完成的具体任务 |
| **Crew** | Agent团队，定义协作方式 |
| **Process** | 任务执行流程（顺序/层级） |

#### 代码示例

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Task:
    """任务定义"""
    description: str
    agent: str  # 分配的Agent
    expected_output: str
    context: List['Task'] = None

class CrewAIAgent:
    """CrewAI风格Agent"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm, tools: List = None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.tools = tools or []
    
    def execute_task(self, task: Task, context: str = "") -> str:
        """执行任务"""
        prompt = f"""You are {self.role}.

Backstory: {self.backstory}

Goal: {self.goal}

Task: {task.description}

Expected Output: {task.expected_output}

Context: {context}

Provide your output:"""
        
        return self.llm.generate(prompt)

class Crew:
    """Agent团队"""
    
    def __init__(self, agents: List[CrewAIAgent], tasks: List[Task], process: str = "sequential"):
        self.agents = {a.role: a for a in agents}
        self.tasks = tasks
        self.process = process
        self.outputs: Dict[str, str] = {}
    
    def run(self) -> str:
        """执行团队任务"""
        if self.process == "sequential":
            return self._sequential_execution()
        elif self.process == "hierarchical":
            return self._hierarchical_execution()
    
    def _sequential_execution(self) -> str:
        """顺序执行"""
        context = ""
        
        for task in self.tasks:
            agent = self.agents.get(task.agent)
            if agent:
                # 收集依赖任务的输出
                if task.context:
                    dep_context = "\n".join([
                        self.outputs.get(t.description, "")
                        for t in task.context
                    ])
                    context = f"{context}\n{dep_context}"
                
                # 执行任务
                output = agent.execute_task(task, context)
                self.outputs[task.description] = output
                context = output
        
        return context
    
    def _hierarchical_execution(self) -> str:
        """层级执行"""
        # Manager分配任务
        manager = self.agents.get("manager") or list(self.agents.values())[0]
        
        results = []
        for task in self.tasks:
            agent = self.agents.get(task.agent)
            if agent:
                output = agent.execute_task(task)
                results.append(f"{task.agent}: {output}")
        
        # Manager汇总
        summary_task = Task(
            description="Summarize all team outputs",
            agent="manager",
            expected_output="Final summary"
        )
        
        return manager.execute_task(summary_task, "\n".join(results))

# 使用示例
# researcher = CrewAIAgent(
#     role="researcher",
#     goal="Research and gather information",
#     backstory="Expert researcher with 10 years experience",
#     llm=llm
# )
#
# writer = CrewAIAgent(
#     role="writer",
#     goal="Write engaging content",
#     backstory="Professional content writer",
#     llm=llm
# )
#
# crew = Crew(
#     agents=[researcher, writer],
#     tasks=[
#         Task("Research AI trends", "researcher", "Key AI trends summary"),
#         Task("Write blog post", "writer", "Blog post about AI trends")
#     ],
#     process="sequential"
# )
#
# result = crew.run()
```

---

## 七、评估与安全

### 7.1 Agent评估指标

#### 任务完成评估

**成功率 (Success Rate)**：

$$
\text{SR} = \frac{\text{Number of successful tasks}}{\text{Total tasks}}
$$

**完成效率**：

$$
\text{Efficiency} = \frac{1}{\text{Average steps to completion}}
$$

#### 轨迹评估

评估Agent执行路径的最优性：

$$
\text{Trajectory Score} = \frac{\text{Optimal Steps}}{\text{Actual Steps}}
$$

#### 成本评估

**Token成本**：

$$
\text{Cost} = \sum_{i=1}^{n} (\text{Input Tokens}_i \times p_{\text{in}} + \text{Output Tokens}_i \times p_{\text{out}})
$$

### 7.2 评估基准

| 基准 | 描述 | 特点 |
|------|------|------|
| **AgentBench** | 多任务评估基准 | 编程、数据库、Web等 |
| **WebShop** | 电商任务评估 | 真实网页交互 |
| **ToolBench** | 工具使用评估 | 16000+真实API |
| **ALFWorld** | 家庭任务评估 | 文本+视觉环境 |

### 7.3 安全机制

#### 提示注入防御

检测和防御恶意提示：

$$
P(\text{injection}) = \sigma(f_{\text{detector}}(\text{input}))
$$

防御策略：
1. **输入过滤**：检测并移除恶意模式
2. **权限隔离**：限制Agent可执行的操作
3. **输出验证**：检查输出是否符合预期

#### 工具调用安全

**工具白名单**：只允许调用预定义的工具

**参数验证**：

$$
\text{Valid}(\theta) = \mathbb{1}[\theta \in \text{Schema}]
$$

**沙箱执行**：在隔离环境中执行工具

### 7.4 代码实现

```python
from typing import List, Dict, Callable
from dataclasses import dataclass
import re

@dataclass
class SafetyCheck:
    """安全检查结果"""
    is_safe: bool
    reason: str
    risk_level: str  # "low", "medium", "high"

class AgentSafeguard:
    """Agent安全防护"""
    
    def __init__(self):
        # 危险模式列表
        self.dangerous_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s*[:=]\s*['\"]",
            r"<\|.*?\|>",
            r"jailbreak",
            r"override\s+safety"
        ]
        
        # 工具调用限制
        self.tool_restrictions = {
            "file_write": {"max_size": 1024 * 1024},  # 1MB
            "execute_code": {"timeout": 30, "sandbox": True}
        }
    
    def check_input(self, user_input: str) -> SafetyCheck:
        """检查输入安全性"""
        # 检测危险模式
        for pattern in self.dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return SafetyCheck(
                    is_safe=False,
                    reason=f"Detected potentially malicious pattern",
                    risk_level="high"
                )
        
        # 检测过长输入
        if len(user_input) > 10000:
            return SafetyCheck(
                is_safe=False,
                reason="Input exceeds maximum length",
                risk_level="medium"
            )
        
        return SafetyCheck(is_safe=True, reason="", risk_level="low")
    
    def check_tool_call(self, tool_name: str, arguments: Dict) -> SafetyCheck:
        """检查工具调用安全性"""
        if tool_name not in self.tool_restrictions:
            return SafetyCheck(is_safe=True, reason="", risk_level="low")
        
        restrictions = self.tool_restrictions[tool_name]
        
        # 检查文件大小限制
        if "max_size" in restrictions:
            content = arguments.get("content", "")
            if len(content) > restrictions["max_size"]:
                return SafetyCheck(
                    is_safe=False,
                    reason="Content exceeds maximum size",
                    risk_level="medium"
                )
        
        return SafetyCheck(is_safe=True, reason="", risk_level="low")
    
    def sanitize_output(self, output: str) -> str:
        """清理输出"""
        # 移除敏感信息
        sensitive_patterns = [
            (r'\b\d{16}\b', '[CREDIT_CARD]'),  # 信用卡号
            (r'\b\d{6,}\b', '[ID_NUMBER]'),     # 长数字
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]')  # 邮箱
        ]
        
        for pattern, replacement in sensitive_patterns:
            output = re.sub(pattern, replacement, output)
        
        return output

class SafeAgent:
    """带安全防护的Agent"""
    
    def __init__(self, agent, safeguard: AgentSafeguard):
        self.agent = agent
        self.safeguard = safeguard
    
    def run(self, query: str) -> str:
        """安全执行"""
        # 1. 输入检查
        input_check = self.safeguard.check_input(query)
        if not input_check.is_safe:
            return f"Request blocked: {input_check.reason}"
        
        # 2. 执行（带工具调用检查）
        original_execute = self.agent.registry.execute
        
        def safe_execute(tool_name, arguments):
            check = self.safeguard.check_tool_call(tool_name, arguments)
            if not check.is_safe:
                raise PermissionError(f"Tool call blocked: {check.reason}")
            return original_execute(tool_name, arguments)
        
        self.agent.registry.execute = safe_execute
        result = self.agent.run(query)
        
        # 3. 输出清理
        return self.safeguard.sanitize_output(result)

# 使用示例
# safeguard = AgentSafeguard()
# safe_agent = SafeAgent(base_agent, safeguard)
# result = safe_agent.run(user_query)
```

---

## 知识点关联

```
AI Agent知识图谱

                    ┌─────────────┐
                    │  AI Agent   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │ 架构设计 │      │ 核心技术 │      │  框架   │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                 │                 │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │• ReAct  │      │• RAG    │      │• LangCh │
    │• Plan-Ex│      │• Tools  │      │• AutoGP │
    │• Multi-A│      │• Memory │      │• CrewAI │
    └─────────┘      └─────────┘      └─────────┘
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │                 │
           ┌──────┴──────┐   ┌──────┴──────┐
           │   评估指标   │   │  安全机制   │
           └─────────────┘   └─────────────┘
```

**关联知识**：
- **LLM基础**：Agent依赖LLM作为核心推理引擎
- **Prompt Engineering**：架构设计依赖有效的提示策略
- **向量数据库**：RAG和长期记忆的核心组件
- **强化学习**：Agent决策可建模为RL问题
- **分布式系统**：Multi-Agent系统的协作机制

---

## 核心考点

### 📌 考点1：Agent架构选择

**问题**：何时使用ReAct vs Plan-and-Execute架构？

**答案**：
- **ReAct**：适合需要即时反馈、动态调整的任务（如对话、搜索）
- **Plan-and-Execute**：适合复杂、可分解的任务（如项目管理、研究报告）

### 📌 考点2：RAG优化策略

**问题**：如何提升RAG检索质量？

**答案**：
1. **检索策略**：混合检索（关键词+语义）、多路召回
2. **重排序**：Cross-Encoder重排提升精度
3. **查询优化**：查询扩展、查询改写
4. **分块策略**：合理chunk size，保持语义完整性

### 📌 考点3：记忆系统设计

**问题**：短期记忆和长期记忆如何协同工作？

**答案**：
- 短期记忆处理当前对话上下文，快速访问
- 长期记忆存储重要信息，支持跨会话检索
- 通过**记忆巩固**机制将短期记忆转移到长期记忆
- 使用重要性评分决定存储位置和保留时间

### 📌 考点4：安全性考量

**问题**：Agent面临的主要安全风险及防护措施？

**答案**：
| 风险 | 防护措施 |
|------|----------|
| 提示注入 | 输入过滤、指令隔离 |
| 工具滥用 | 权限控制、沙箱执行 |
| 数据泄露 | 输出脱敏、访问控制 |
| 资源耗尽 | 调用限制、超时控制 |

---

## 学习建议

### 📚 推荐学习路径

```
入门阶段 (1-2周)
├── 理解Agent基本概念
├── 学习LangChain基础组件
└── 实现简单ReAct Agent

进阶阶段 (2-4周)
├── 深入RAG技术
├── 掌握记忆系统设计
└── 实现多工具Agent

高级阶段 (4-8周)
├── Multi-Agent系统设计
├── Agent评估与优化
└── 安全性增强

实践阶段 (持续)
├── 构建完整Agent应用
├── 参与开源项目
└── 跟踪前沿研究
```

### 🎯 实践项目建议

1. **入门项目**：基于LangChain构建一个简单的问答Agent
2. **进阶项目**：实现一个带RAG的知识库助手
3. **高级项目**：构建多Agent协作的自动化工作流系统

### 📖 推荐资源

- **论文**：ReAct (2022), Toolformer (2023), AutoGPT
- **框架文档**：LangChain Docs, CrewAI Docs
- **基准测试**：AgentBench, WebShop, ToolBench

---

## 参考资料

1. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
2. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
3. Park, J.S., et al. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv 2023.
4. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024.
