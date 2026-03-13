# 记忆系统

记忆系统让 LLM 应用能够"记住"之前的对话内容，实现连贯的多轮对话。LangChain 提供了多种记忆组件，从简单的消息存储到智能的对话摘要。

---

## 🧠 为什么需要记忆

### 无记忆 vs 有记忆

```
无记忆对话：
用户：我叫张三
AI：你好，张三！
用户：我叫什么名字？
AI：抱歉，我不知道你的名字。（❌ 无法记住）

有记忆对话：
用户：我叫张三
AI：你好，张三！
用户：我叫什么名字？
AI：你叫张三。（✅ 记住了之前的信息）
```

### 记忆的核心挑战

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| **Token 限制** | 模型有上下文长度限制 | 摘要、滑动窗口 |
| **相关性** | 不是所有历史都重要 | 选择性记忆 |
| **持久化** | 重启后记忆丢失 | 数据库存储 |
| **多用户** | 不同用户不同记忆 | 会话管理 |

---

## 📦 记忆类型概览

```
记忆类型
├── 基于消息
│   ├── ConversationBufferMemory     # 完整存储
│   ├── ConversationBufferWindowMemory # 滑动窗口
│   └── ConversationTokenBufferMemory  # Token限制
│
├── 基于摘要
│   ├── ConversationSummaryMemory      # 摘要历史
│   └── ConversationSummaryBufferMemory # 摘要+最新
│
└── 基于知识
    ├── VectorStoreMemory              # 向量检索
    └── EntityMemory                   # 实体记忆
```

---

## 🔧 使用 LangGraph 管理记忆（推荐）

在新版本的 LangChain 中，推荐使用 LangGraph 来管理对话记忆。

### 基础对话记忆

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 创建记忆存储
memory = MemorySaver()

# 创建带记忆的 Agent
model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(model, tools=[], checkpointer=memory)

# 配置会话ID
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一轮对话
result1 = agent.invoke(
    {"messages": [("user", "我叫张三")]},
    config
)
print(result1["messages"][-1].content)

# 第二轮对话（能记住之前的内容）
result2 = agent.invoke(
    {"messages": [("user", "我叫什么名字？")]},
    config
)
print(result2["messages"][-1].content)  # "你叫张三"
```

### 手动管理消息历史

```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessageGraph

# 使用消息图管理历史
class ChatWithHistory:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.history = []
    
    def chat(self, message: str) -> str:
        # 添加用户消息
        self.history.append(HumanMessage(content=message))
        
        # 调用模型（传入历史）
        response = self.model.invoke(self.history)
        
        # 添加AI回复到历史
        self.history.append(response)
        
        return response.content
    
    def get_history(self):
        return self.history
    
    def clear(self):
        self.history = []

# 使用
chat = ChatWithHistory()
print(chat.chat("我叫李四"))        # 你好，李四！
print(chat.chat("我叫什么名字？"))  # 你叫李四
```

---

## 💾 使用 RunnableWithMessageHistory

LangChain 提供了 `RunnableWithMessageHistory` 来包装任何 Runnable，为其添加记忆能力。

### 基础用法

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict

# 创建基础链
model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，用中文回答问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

# 创建消息历史存储
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 包装链，添加记忆
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用
config = {"configurable": {"session_id": "user-1"}}

# 第一轮
response1 = chain_with_history.invoke(
    {"input": "我叫王五"},
    config
)
print(response1)

# 第二轮
response2 = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config
)
print(response2)  # "你叫王五"
```

### 不同会话隔离

```python
# 用户1的会话
config1 = {"configurable": {"session_id": "user-1"}}
chain_with_history.invoke({"input": "我喜欢蓝色"}, config1)

# 用户2的会话（隔离）
config2 = {"configurable": {"session_id": "user-2"}}
chain_with_history.invoke({"input": "我喜欢红色"}, config2)

# 用户1再次对话（记得自己的偏好）
chain_with_history.invoke({"input": "我喜欢什么颜色？"}, config1)  # "你喜欢蓝色"
```

---

## 🗄️ 持久化存储

### 文件存储

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

def get_session_history(session_id: str):
    return FileChatMessageHistory(f"./chat_history/{session_id}.json")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### Redis 存储

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

REDIS_URL = "redis://localhost:6379"

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id,
        url=REDIS_URL
    )

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### SQLite 存储

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id,
        connection="sqlite:///chat_history.db"
    )
```

---

## ✂️ 记忆管理策略

### 策略一：限制消息数量

```python
from langchain_core.messages import trim_messages

def trim_history(messages, max_tokens=1000):
    """修剪消息历史，保持在token限制内"""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",  # 保留最新的消息
        token_counter=len,  # 简单计数，实际应用应使用tiktoken
        include_system=True,
    )

class SmartChat:
    def __init__(self, max_history=10):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.history = []
        self.max_history = max_history
    
    def chat(self, message: str) -> str:
        self.history.append(HumanMessage(content=message))
        
        # 限制历史长度
        if len(self.history) > self.max_history * 2:  # 人类+AI消息
            # 保留系统消息（如果有）和最新的消息
            self.history = self.history[-self.max_history * 2:]
        
        response = self.model.invoke(self.history)
        self.history.append(response)
        
        return response.content
```

### 策略二：滑动窗口

```python
class SlidingWindowChat:
    def __init__(self, window_size=5):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.history = []
        self.window_size = window_size  # 保留最近N轮对话
    
    def chat(self, message: str) -> str:
        self.history.append(HumanMessage(content=message))
        
        response = self.model.invoke(
            self.history[-self.window_size * 2:]  # 只传最近N轮
        )
        
        self.history.append(response)
        return response.content
```

### 策略三：摘要记忆

当对话很长时，使用摘要来压缩历史：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class SummaryMemoryChat:
    def __init__(self, summary_threshold=10):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.history = []
        self.summary = ""
        self.summary_threshold = summary_threshold
    
    def summarize(self, messages):
        """生成对话摘要"""
        prompt = ChatPromptTemplate.from_template(
            "请总结以下对话的关键信息：\n{conversation}"
        )
        chain = prompt | self.model
        conversation = "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages
        ])
        return chain.invoke({"conversation": conversation}).content
    
    def chat(self, message: str) -> str:
        self.history.append(HumanMessage(content=message))
        
        # 如果历史太长，生成摘要
        if len(self.history) > self.summary_threshold * 2:
            self.summary = self.summarize(self.history[:-self.summary_threshold])
            self.history = self.history[-self.summary_threshold:]
        
        # 构建包含摘要的上下文
        context = []
        if self.summary:
            context.append(SystemMessage(
                content=f"之前的对话摘要：{self.summary}"
            ))
        context.extend(self.history)
        
        response = self.model.invoke(context)
        self.history.append(response)
        
        return response.content
```

---

## 🎯 完整示例：带记忆的对话助手

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Optional
import os

class ConversationalAssistant:
    """带记忆的对话助手"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "你是一个有帮助的AI助手。",
        max_history: int = 20
    ):
        self.model = ChatOpenAI(model=model_name, temperature=0.7)
        self.system_prompt = system_prompt
        self.max_history = max_history
        
        # 会话存储
        self._store: Dict[str, ChatMessageHistory] = {}
        
        # 创建带记忆的链
        self._setup_chain()
    
    def _setup_chain(self):
        """设置对话链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | self.model | StrOutputParser()
        
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]
    
    def chat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """发送消息并获取回复"""
        config = {"configurable": {"session_id": session_id}}
        
        response = self.chain_with_history.invoke(
            {"input": message},
            config
        )
        
        # 可选：修剪历史
        self._trim_history(session_id)
        
        return response
    
    def _trim_history(self, session_id: str):
        """修剪过长的历史"""
        history = self._get_session_history(session_id)
        messages = history.messages
        
        if len(messages) > self.max_history * 2:
            # 保留最新的消息
            history.clear()
            for msg in messages[-self.max_history * 2:]:
                history.add_message(msg)
    
    def get_history(self, session_id: str = "default") -> list:
        """获取会话历史"""
        return self._get_session_history(session_id).messages
    
    def clear_history(self, session_id: str = "default"):
        """清除会话历史"""
        if session_id in self._store:
            self._store[session_id].clear()
    
    def list_sessions(self) -> list:
        """列出所有会话"""
        return list(self._store.keys())


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建助手
    assistant = ConversationalAssistant(
        model_name="gpt-4o-mini",
        system_prompt="你是一个友好的AI助手，能够记住之前的对话内容。请用中文回答问题。",
        max_history=10
    )
    
    # 模拟多轮对话
    print("=" * 50)
    print("多轮对话示例")
    print("=" * 50)
    
    # 第一轮
    user_msg = "你好，我叫张三"
    print(f"用户：{user_msg}")
    response = assistant.chat(user_msg, session_id="user-1")
    print(f"AI：{response}\n")
    
    # 第二轮
    user_msg = "我喜欢编程，特别是Python"
    print(f"用户：{user_msg}")
    response = assistant.chat(user_msg, session_id="user-1")
    print(f"AI：{response}\n")
    
    # 第三轮（测试记忆）
    user_msg = "我叫什么名字？我喜欢什么？"
    print(f"用户：{user_msg}")
    response = assistant.chat(user_msg, session_id="user-1")
    print(f"AI：{response}\n")
    
    # 查看历史
    print("=" * 50)
    print("对话历史：")
    for msg in assistant.get_history("user-1"):
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content[:50]}...")
    
    print("\n" + "=" * 50)
    print("多用户隔离示例")
    print("=" * 50)
    
    # 用户2的对话（隔离）
    assistant.chat("我叫李四，我喜欢画画", session_id="user-2")
    response = assistant.chat("我叫什么名字？", session_id="user-2")
    print(f"用户2的AI回复：{response}")  # 应该回答李四
    
    # 用户1的历史不受影响
    response = assistant.chat("我叫什么名字？", session_id="user-1")
    print(f"用户1的AI回复：{response}")  # 应该回答张三
```

---

## 📊 记忆类型选择指南

| 记忆类型 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **BufferMemory** | 短对话、演示 | 简单、完整 | 长对话消耗大量Token |
| **WindowMemory** | 中等长度对话 | 可控的Token消耗 | 可能丢失早期重要信息 |
| **SummaryMemory** | 长对话 | 节省Token、保留关键信息 | 需要额外的摘要调用 |
| **VectorMemory** | 需要检索历史 | 智能检索相关信息 | 实现复杂、需要向量库 |

---

## 📋 记忆系统检查清单

### ✅ 设计阶段

- [ ] 确定是否需要持久化存储
- [ ] 评估对话长度和Token限制
- [ ] 决定是否需要多用户隔离
- [ ] 选择合适的记忆策略

### ✅ 实现阶段

- [ ] 设置会话ID管理机制
- [ ] 实现历史修剪或摘要
- [ ] 添加清除历史的功能
- [ ] 处理异常情况

### ✅ 测试阶段

- [ ] 测试多轮对话记忆正确性
- [ ] 测试多用户隔离
- [ ] 测试长对话的处理
- [ ] 测试持久化和恢复

---

## 🔗 相关章节

- [智能体开发](./langchain-agents.md) - 为 Agent 添加记忆能力
- [链与 LCEL](./langchain-chains.md) - 在链中使用记忆
- [工具调用](./langchain-tools.md) - 持久化存储工具
