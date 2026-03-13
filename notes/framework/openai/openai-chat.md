# Chat Completions API

Chat Completions API 是 OpenAI 最核心的 API，用于生成对话回复。本章详细介绍消息结构、参数配置、多轮对话和最佳实践。

---

## 📝 消息结构

### 基本消息格式

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "请介绍一下 Python"}
]
```

### 角色说明

| 角色 | 说明 | 使用场景 |
|------|------|----------|
| `system` | 系统指令 | 设定助手行为、角色、限制 |
| `user` | 用户消息 | 用户输入的问题或指令 |
| `assistant` | 助手回复 | 历史对话中的模型回复 |
| `tool` | 工具消息 | Function Calling 的返回结果 |

### 系统提示词

系统提示词用于设定模型的行为：

```python
from openai import OpenAI

client = OpenAI()

# 基础角色设定
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个专业的 Python 开发者，擅长解释代码。"},
        {"role": "user", "content": "什么是装饰器？"}
    ]
)

# 复杂系统提示词
system_prompt = """
你是一个专业的技术文档撰写助手。请遵循以下规则：

1. 使用 Markdown 格式
2. 代码示例添加注释
3. 解释要简洁清晰
4. 必要时提供示例输出

回答时要结构化，使用标题和列表。
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "解释 Python 的列表推导式"}
    ]
)
```

---

## ⚙️ 核心参数

### 参数概览

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",           # 必需：模型名称
    messages=[...],                # 必需：消息列表
    
    # 输出控制
    temperature=0.7,               # 随机性 (0-2)
    top_p=0.9,                     # 核采样 (0-1)
    max_tokens=1024,               # 最大输出长度
    stop=["END"],                  # 停止词
    
    # 功能控制
    frequency_penalty=0.0,         # 频率惩罚 (-2 到 2)
    presence_penalty=0.0,          # 存在惩罚 (-2 到 2)
    logit_bias={},                 # Token 偏置
    seed=42,                       # 随机种子
    
    # 输出格式
    response_format={"type": "json_object"},  # JSON 模式
    
    # 其他
    n=1,                           # 生成数量
    stream=False,                  # 是否流式
    user="user-123"                # 用户标识
)
```

### temperature（温度）

控制输出的随机性：

```python
from openai import OpenAI

client = OpenAI()

# 低温度：更确定、更一致的输出
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "什么是 AI？"}],
    temperature=0.0  # 几乎每次相同
)

# 中等温度：平衡创造性和一致性
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一首诗"}],
    temperature=0.7  # 适度的创造性
)

# 高温度：更多样、更有创造性
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一个创意故事"}],
    temperature=1.5  # 高创造性，可能不太一致
)
```

**温度选择建议：**

| 任务类型 | 推荐值 | 说明 |
|----------|--------|------|
| 代码生成 | 0.0 - 0.2 | 确定性输出 |
| 数据分析 | 0.0 - 0.3 | 准确性优先 |
| 文档撰写 | 0.3 - 0.5 | 一致性为主 |
| 一般对话 | 0.5 - 0.7 | 平衡 |
| 创意写作 | 0.7 - 1.0 | 更有创意 |
| 头脑风暴 | 1.0 - 1.5 | 高度发散 |

### max_tokens

限制输出长度：

```python
# 短回复
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用一句话解释什么是机器学习"}],
    max_tokens=50
)

# 长回复
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一篇关于人工智能的文章"}],
    max_tokens=2000
)
```

### stop 序列

设置停止词：

```python
# 使用单个停止词
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "列出三个编程语言："}],
    stop=["\n\n"]  # 遇到两个换行停止
)

# 使用多个停止词
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "翻译成英文：你好"}],
    stop=["。", ".", "\n"]  # 遇到任一停止
)
```

### frequency_penalty & presence_penalty

控制内容重复：

```python
# frequency_penalty: 惩罚频繁出现的词
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "写一段描述春天的文字"}],
    frequency_penalty=0.8  # 减少重复词汇
)

# presence_penalty: 惩罚已出现过的词
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "列出 10 个不同的创意主题"}],
    presence_penalty=1.0  # 鼓励谈论新话题
)
```

**惩罚参数对比：**

| 参数 | 作用 | 适用场景 |
|------|------|----------|
| `frequency_penalty` | 根据出现频率惩罚 | 减少重复词汇 |
| `presence_penalty` | 只要出现就惩罚 | 讨论更多样话题 |

### seed（随机种子）

实现可重复输出：

```python
# 相同 seed 产生相似输出
for i in range(3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "生成一个随机数"}],
        seed=42,
        temperature=0.0
    )
    print(f"第 {i+1} 次: {response.choices[0].message.content}")

# 检查是否是确定性输出
print(f"系统指纹: {response.system_fingerprint}")
```

---

## 🔄 多轮对话

### 基本多轮对话

```python
from openai import OpenAI

client = OpenAI()

class ChatSession:
    """简单的多轮对话管理"""
    
    def __init__(self, system_prompt="你是一个有帮助的助手。"):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
    
    def chat(self, user_input):
        """发送消息并获取回复"""
        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 调用 API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        # 获取助手回复
        assistant_message = response.choices[0].message.content
        
        # 添加到历史
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear(self):
        """清空对话历史"""
        self.messages = [self.messages[0]]  # 保留系统提示

# 使用
session = ChatSession("你是一个 Python 专家。")

print(session.chat("什么是列表推导式？"))
print(session.chat("给我一个例子"))  # 会记住上文
print(session.chat("和 map 函数有什么区别？"))
```

### 带上下文限制的对话

```python
from openai import OpenAI
import tiktoken

client = OpenAI()

class SmartChatSession:
    """带上下文管理的对话"""
    
    def __init__(self, model="gpt-4o-mini", max_tokens=4000):
        self.model = model
        self.max_tokens = max_tokens
        self.messages = []
    
    def count_tokens(self, messages):
        """计算消息的 token 数"""
        encoding = tiktoken.encoding_for_model(self.model)
        total = 0
        for msg in messages:
            total += len(encoding.encode(msg["content"]))
            total += 4  # 消息格式开销
        return total
    
    def trim_messages(self):
        """裁剪旧消息以适应上下文限制"""
        while self.count_tokens(self.messages) > self.max_tokens:
            if len(self.messages) > 1:
                self.messages.pop(0)  # 删除最旧的消息
            else:
                break
    
    def chat(self, user_input):
        """发送消息"""
        self.messages.append({"role": "user", "content": user_input})
        
        # 裁剪历史
        self.trim_messages()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

# 使用
session = SmartChatSession(max_tokens=2000)
```

---

## 📤 响应格式

### JSON 模式

强制输出 JSON 格式：

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system", 
            "content": "你是一个数据提取助手，始终输出有效的 JSON。"
        },
        {
            "role": "user", 
            "content": "提取以下文本中的人物信息：张三，28岁，工程师，住在北京"
        }
    ],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "张三", "age": 28, "job": "工程师", "city": "北京"}
```

### 结构化输出（Structured Outputs）

使用 Pydantic 模型定义输出格式：

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# 定义输出结构
class Person(BaseModel):
    name: str
    age: int
    occupation: str
    city: str

class PersonList(BaseModel):
    persons: list[Person]

# 使用结构化输出
response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "提取信息：李四，35岁，医生，上海；王五，42岁，教师，广州"}
    ],
    response_format=PersonList
)

# 直接获取解析后的对象
result = response.choices[0].message.parsed
print(result.persons)
# [Person(name='李四', age=35, occupation='医生', city='上海'), 
#  Person(name='王五', age=42, occupation='教师', city='广州')]
```

---

## 🎯 实践示例

### 示例1：对话机器人

```python
from openai import OpenAI
from typing import Optional
import json

client = OpenAI()

class ChatBot:
    """功能完整的对话机器人"""
    
    def __init__(
        self,
        system_prompt: str = "你是一个有帮助的助手。",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_history: int = 20
    ):
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # 限制历史长度
            if len(self.messages) > self.max_history * 2 + 1:
                # 保留系统消息和最近的消息
                system_msg = self.messages[0]
                recent = self.messages[-(self.max_history * 2):]
                self.messages = [system_msg] + recent
            
            return assistant_message
            
        except Exception as e:
            # 出错时移除最后添加的用户消息
            self.messages.pop()
            raise e
    
    def get_history(self) -> list:
        """获取对话历史"""
        return self.messages.copy()
    
    def save_history(self, filepath: str):
        """保存对话历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
    
    def load_history(self, filepath: str):
        """加载对话历史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.messages = json.load(f)
    
    def clear(self):
        """清空对话历史"""
        system_msg = self.messages[0]
        self.messages = [system_msg]

# 使用示例
bot = ChatBot(
    system_prompt="你是一个友好的 Python 学习助手。",
    temperature=0.5
)

while True:
    user_input = input("你: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    try:
        response = bot.chat(user_input)
        print(f"助手: {response}\n")
    except Exception as e:
        print(f"错误: {e}")
```

### 示例2：角色扮演系统

```python
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()

@dataclass
class Character:
    """角色定义"""
    name: str
    personality: str
    background: str
    speaking_style: str

def create_character_prompt(character: Character) -> str:
    """创建角色提示词"""
    return f"""
你现在要扮演 {character.name}。

【角色背景】
{character.background}

【性格特点】
{character.personality}

【说话风格】
{character.speaking_style}

请完全沉浸在这个角色中，用第一人称回答。保持角色一致性，不要跳出角色。
"""

class RoleplayChat:
    """角色扮演对话系统"""
    
    def __init__(self, character: Character):
        self.character = character
        self.messages = [
            {"role": "system", "content": create_character_prompt(character)}
        ]
    
    def chat(self, user_input: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages + [{"role": "user", "content": user_input}]
        )
        
        reply = response.choices[0].message.content
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": reply})
        
        return reply

# 使用
sherlock = Character(
    name="夏洛克·福尔摩斯",
    personality="极度聪明、观察敏锐、有点傲慢、缺乏社交技巧",
    background="19世纪末伦敦的咨询侦探，住在贝克街221B",
    speaking_style="说话简洁有力，经常推理演绎，喜欢用'显而易见'这类词"
)

game = RoleplayChat(sherlock)
print(game.chat("你好，福尔摩斯先生"))
```

---

## 📋 参数速查表

### 常用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必需 | 模型名称 |
| `messages` | list | 必需 | 消息列表 |
| `temperature` | float | 1.0 | 随机性 (0-2) |
| `top_p` | float | 1.0 | 核采样 |
| `max_tokens` | int | inf | 最大输出 token |
| `stop` | list | None | 停止序列 |
| `n` | int | 1 | 生成数量 |
| `stream` | bool | False | 流式输出 |

### 输出控制参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `frequency_penalty` | -2 到 2 | 频率惩罚 |
| `presence_penalty` | -2 到 2 | 存在惩罚 |
| `logit_bias` | dict | Token 偏置 |
| `seed` | int | 随机种子 |

---

## 🔗 相关章节

- [基础 API 调用](./openai-basics.md) - 环境配置
- [流式响应处理](./openai-streaming.md) - 流式输出
- [Function Calling](./openai-functions.md) - 工具调用
