# 智能体（Agent）开发

Agent（智能体）是 LangChain 中最强大的概念之一。与普通的链式调用不同，Agent 能够根据输入自主决策使用哪些工具、执行哪些操作，实现真正的智能行为。

---

## 🤖 什么是 Agent

### Agent vs Chain

```
Chain（链）:
输入 → 步骤1 → 步骤2 → 步骤3 → 输出
（预定义的执行路径）

Agent（智能体）:
输入 → [思考] → [选择工具] → [执行] → [观察结果] → [继续/结束] → 输出
（自主决策的执行路径）
```

### Agent 核心概念

| 概念 | 说明 |
|------|------|
| **Agent** | 大脑，负责推理和决策 |
| **Tools** | 手脚，执行具体操作的能力 |
| **AgentExecutor** | 运行环境，管理 Agent 的执行循环 |

### Agent 工作流程

```
┌─────────────────────────────────────────────────────┐
│                    Agent 执行循环                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│   用户输入                                          │
│      ↓                                             │
│   ┌──────────┐                                     │
│   │  思考    │ ← 分析问题，决定下一步               │
│   └────┬─────┘                                     │
│        ↓                                           │
│   ┌──────────┐                                     │
│   │ 选择工具 │ ← 根据需求选择合适的工具             │
│   └────┬─────┘                                     │
│        ↓                                           │
│   ┌──────────┐                                     │
│   │ 执行工具 │ ← 调用工具获取结果                   │
│   └────┬─────┘                                     │
│        ↓                                           │
│   ┌──────────┐                                     │
│   │ 观察结果 │ ← 分析执行结果                       │
│   └────┬─────┘                                     │
│        │                                           │
│        ├─→ 需要继续？ ─→ 返回思考                   │
│        │                                           │
│        └─→ 任务完成？ ─→ 返回最终答案               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ 创建 Agent

### 基础 Agent 创建

使用 `create_tool_calling_agent` 创建支持工具调用的 Agent：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

# 1. 定义工具
@tool
def get_word_length(word: str) -> int:
    """返回单词的长度"""
    return len(word)

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘"""
    return a * b

tools = [get_word_length, multiply]

# 2. 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程
])

# 3. 创建 Agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(model, tools, prompt)

# 4. 创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 执行
result = agent_executor.invoke({"input": "单词'hello'的长度乘以3是多少？"})
print(result["output"])
```

### 使用 LangGraph（推荐）

LangGraph 是 LangChain 新推荐的 Agent 构建方式，提供更灵活的控制：

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 定义工具
@tool
def search(query: str) -> str:
    """搜索网络信息"""
    # 实际应用中接入搜索API
    return f"搜索结果：{query}"

@tool  
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 创建 Agent
model = ChatOpenAI(model="gpt-4o-mini")
tools = [search, calculator]

agent = create_react_agent(model, tools)

# 执行
result = agent.invoke({
    "messages": [("user", "搜索Python的最新版本，然后告诉我2024年发布了几个版本")]
})
```

---

## 🎭 Agent 类型

### 1. ReAct Agent

ReAct（Reasoning + Acting）是最经典的 Agent 模式：

```python
from langchain.agents import create_react_agent

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有帮助的助手。

你可以使用以下工具：
{tool_names}

使用工具时遵循以下格式：
Thought: 思考下一步做什么
Action: 工具名称
Action Input: 工具输入
Observation: 工具输出
... (重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_react_agent(model, tools, prompt)
```

### 2. Tool Calling Agent

利用模型原生工具调用能力的 Agent（推荐）：

```python
from langchain.agents import create_tool_calling_agent

# 现代模型（GPT-4o, Claude等）原生支持工具调用
# 不需要解析文本，直接结构化调用
agent = create_tool_calling_agent(model, tools, prompt)
```

### 3. Structured Chat Agent

处理复杂结构化输入的 Agent：

```python
from langchain.agents import create_structured_chat_agent

# 适用于需要复杂参数的工具
agent = create_structured_chat_agent(model, tools, prompt)
```

### Agent 类型对比

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **Tool Calling** | 使用模型原生能力 | 现代模型，推荐首选 |
| **ReAct** | 经典推理模式 | 教学理解、简单任务 |
| **Structured Chat** | 结构化输入 | 复杂参数的工具 |

---

## ⚙️ AgentExecutor 配置

### 常用配置参数

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    # 执行控制
    max_iterations=10,          # 最大迭代次数
    max_execution_time=60,      # 最大执行时间（秒）
    early_stopping_method="generate",  # 早停策略
    
    # 输出控制
    verbose=True,               # 打印详细日志
    handle_parsing_errors=True, # 处理解析错误
    
    # 返回控制
    return_intermediate_steps=True,  # 返回中间步骤
)
```

### 错误处理

```python
from langchain_core.exceptions import OutputParserException

try:
    result = agent_executor.invoke({"input": "复杂任务"})
except OutputParserException as e:
    print(f"解析错误：{e}")
except Exception as e:
    print(f"执行错误：{e}")
```

---

## 🧠 ReAct 模式详解

### ReAct 工作原理

ReAct 将"推理"（Reasoning）和"行动"（Acting）交织进行：

```
用户问题："北京今天天气如何？需要带伞吗？"

[Thought 1] 我需要先查询北京今天的天气情况
[Action 1] search_weather
[Input 1] {"city": "北京"}
[Observation 1] 北京今天多云转小雨，气温18-25℃，降水概率60%

[Thought 2] 现在我知道了天气情况，有60%的降水概率
[Action 2] 无需工具，可以直接回答
[Final Answer] 北京今天多云转小雨，气温18-25℃，降水概率60%。
              建议携带雨伞，因为有较高概率下雨。
```

### ReAct Prompt 模板

```python
REACT_PROMPT = """回答以下问题，尽可能使用工具帮助解答。

你可以使用的工具：
{tools}

工具名称：{tool_names}

使用以下格式：

Question: 用户的问题
Thought: 你应该思考做什么
Action: 要使用的工具，必须是 [{tool_names}] 之一
Action Input: 工具的输入
Observation: 工具的输出
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}"""
```

---

## 🎯 实践示例：多功能助手

下面是一个完整的多功能 Agent 示例：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from typing import Optional
import json

# ========== 定义工具 ==========

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"
    
    Returns:
        天气信息字符串
    """
    # 模拟天气API
    weather_data = {
        "北京": "晴，温度15-25℃，空气质量良好",
        "上海": "多云，温度18-26℃，有轻微雾霾",
        "广州": "小雨，温度22-28℃，湿度较高",
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如"2+3*4"
    
    Returns:
        计算结果
    """
    try:
        # 安全计算
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"计算结果：{result}"
        return "无效表达式"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库获取信息
    
    Args:
        query: 搜索查询
    
    Returns:
        搜索结果
    """
    # 模拟知识库搜索
    knowledge = {
        "Python": "Python是一种高级编程语言，以简洁易读著称。",
        "机器学习": "机器学习是AI的一个分支，让计算机从数据中学习。",
        "深度学习": "深度学习使用神经网络进行特征学习和模式识别。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于'{query}'的信息"

@tool
def create_todo_list(tasks: str) -> str:
    """
    创建待办事项列表
    
    Args:
        tasks: 任务列表，用逗号分隔
    
    Returns:
        格式化的待办事项
    """
    task_list = [t.strip() for t in tasks.split(",")]
    result = "📋 待办事项：\n"
    for i, task in enumerate(task_list, 1):
        result += f"  {i}. ☐ {task}\n"
    return result

# ========== 创建 Agent ==========

# 初始化模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 定义工具列表
tools = [get_weather, calculate, search_knowledge, create_todo_list]

# 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个多功能助手，可以帮助用户：
    - 查询天气
    - 进行数学计算
    - 搜索知识
    - 创建待办事项
    
    根据用户需求选择合适的工具。如果不需要工具，直接回答即可。"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 创建 Agent
agent = create_tool_calling_agent(model, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# ========== 使用示例 ==========

if __name__ == "__main__":
    # 测试不同类型的任务
    queries = [
        "北京和上海今天天气怎么样？",
        "帮我算一下 123 * 456 + 789",
        "什么是机器学习？",
        "帮我创建一个待办列表：写代码、开会、写文档"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"用户：{query}")
        print("-"*50)
        result = agent_executor.invoke({"input": query})
        print(f"助手：{result['output']}")
```

---

## 📊 Agent 调试技巧

### 启用详细日志

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # 打印详细执行过程
)
```

### 查看中间步骤

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True
)

result = agent_executor.invoke({"input": "..."})

# 查看所有中间步骤
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"工具: {action.tool}")
    print(f"输入: {action.tool_input}")
    print(f"输出: {observation}")
    print("-" * 30)
```

### 使用 LangSmith 追踪

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 所有执行都会在 LangSmith 中记录
```

---

## 📋 Agent 开发检查清单

### ✅ 设计阶段

- [ ] 明确 Agent 需要解决的问题
- [ ] 确定需要哪些工具
- [ ] 设计工具的输入输出格式
- [ ] 选择合适的 Agent 类型

### ✅ 实现阶段

- [ ] 编写工具函数并添加文档字符串
- [ ] 创建合适的提示词模板
- [ ] 配置 AgentExecutor 参数
- [ ] 添加错误处理

### ✅ 测试阶段

- [ ] 测试单个工具功能
- [ ] 测试 Agent 决策正确性
- [ ] 测试边界情况和错误处理
- [ ] 评估性能和成本

---

## 🔗 相关章节

- [工具调用](./langchain-tools.md) - 详细了解工具的定义和使用
- [链与 LCEL](./langchain-chains.md) - 理解链式调用基础
- [记忆系统](./langchain-memory.md) - 为 Agent 添加记忆能力
