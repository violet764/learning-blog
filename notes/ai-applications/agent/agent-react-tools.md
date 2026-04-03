# ReAct 模式与工具调用

## 什么是 ReAct？

ReAct（Reasoning + Acting）是一种将**推理**和**行动**交替进行的 Agent 范式，由 Yao 等人在 2022 年提出。

### 核心思想

传统的 Chain-of-Thought（CoT）提示虽然能让模型进行推理，但无法获取外部信息，容易产生幻觉。ReAct 通过将推理与行动结合，让模型能够：

1. **思考（Thought）**：进行推理，决定下一步做什么
2. **行动（Action）**：调用工具与环境交互
3. **观察（Observation）**：获取工具执行结果
4. **循环**：重复上述过程直到完成任务

### ReAct vs CoT

```
┌─────────────────────────────────────────────────────────────┐
│ Chain-of-Thought (CoT)                                      │
│                                                             │
│ 问题 → 思考 → 思考 → 思考 → 答案                              │
│        (纯推理，无法获取新信息)                                │
│                                                             │
│ 问题：Apple Remote 最初设计用于控制什么程序？                  │
│ 思考：Apple Remote 是苹果的遥控器...                          │
│ 思考：它最初用于 Front Row 媒体中心...                         │
│ 思考：Front Row 是一个媒体播放程序...                          │
│ 答案：Front Row（但可能基于过时或错误的知识）                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ReAct                                                       │
│                                                             │
│ 问题 → 思考 → 行动 → 观察 → 思考 → 行动 → ... → 答案          │
│        (推理与行动交替，可获取真实信息)                         │
│                                                             │
│ 问题：Apple Remote 最初设计用于控制什么程序？                  │
│ 思考1：我需要搜索 Apple Remote 的相关信息                      │
│ 行动1：Search[Apple Remote]                                  │
│ 观察1：Apple Remote 是苹果设计的遥控器，用于控制...            │
│ 思考2：找到了，它最初设计用于 Front Row，但还有其他设备能控制... │
│ 行动2：Search[Front Row compatible devices]                  │
│ 观察2：除了 Apple Remote，iPhone、iPad 也可以控制...          │
│ 思考3：我已经有了完整答案                                     │
│ 答案：Apple Remote 最初设计用于控制 Front Row...              │
└─────────────────────────────────────────────────────────────┘
```

## ReAct 工作流程

### 完整执行示例

```
问题：科罗拉多造山带东部延伸区域的 elevation range 是多少？

Thought 1: 我需要搜索科罗拉多造山带，找到东部延伸的区域，然后查找该区域的海拔范围。
Action 1: Search[科罗拉多造山带]
Observation 1: 科罗拉多造山带是科罗拉多及周边地区的一次造山运动事件。

Thought 2: 没有提到东部区域，我需要查找东部区域的信息。
Action 2: Lookup[东部区域]
Observation 2: 东部区域延伸至高平原，称为中央平原造山带。

Thought 3: 东部区域延伸至高平原，我需要搜索高平原的海拔范围。
Action 3: Search[高平原]
Observation 3: 高平原指两个不同的陆地区域...

Thought 4: 我需要搜索美国高平原。
Action 4: Search[美国高平原]
Observation 4: 高平原是大平原的子区域。从东到西，海拔从约 1,800 英尺上升到 7,000 英尺。

Thought 5: 高平原海拔范围是 1,800 到 7,000 英尺，这就是答案。
Action 5: Finish[1,800 到 7,000 英尺]
```

### ReAct 提示词模板

```python
REACT_PROMPT = """
你是一个可以使用搜索工具的智能助手。

对于每一步，你需要：
1. Thought: 思考当前状态，决定下一步
2. Action: 选择一个行动（Search[query] 或 Finish[answer]）

可用工具：
- Search[query]: 搜索信息
- Lookup[keyword]: 在当前页面查找关键词
- Finish[answer]: 给出最终答案

示例：
Question: 苹果公司成立于哪一年？
Thought 1: 我需要搜索苹果公司的成立信息
Action 1: Search[苹果公司成立时间]
Observation 1: 苹果公司由乔布斯等人于1976年4月1日创立
Thought 2: 我找到了答案
Action 2: Finish[1976年4月1日]

现在开始：
Question: {question}
"""
```

## Function Calling（工具调用）

现代 LLM（如 GPT-4、Claude）原生支持 **Function Calling**，这是一种结构化的工具调用方式，比 ReAct 的文本解析更可靠。

### 工具定义格式

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认摄氏度"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### Function Calling 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 用户请求                                                 │
│     "北京今天天气怎么样？"                                    │
│                          ▼                                  │
│  2. LLM 决策                                                 │
│     {                                                       │
│       "name": "get_weather",                                │
│       "arguments": {"city": "北京", "unit": "celsius"}      │
│     }                                                       │
│                          ▼                                  │
│  3. 执行工具                                                 │
│     get_weather("北京", "celsius")                          │
│     → {"temp": 25, "condition": "晴", "humidity": 45}      │
│                          ▼                                  │
│  4. 返回结果给 LLM                                           │
│     工具结果: {"temp": 25, "condition": "晴"}               │
│                          ▼                                  │
│  5. LLM 生成最终回答                                         │
│     "北京今天天气晴朗，气温 25°C，湿度 45%..."               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 代码实现

```python
from openai import OpenAI

client = OpenAI()

def run_agent(user_message, tools, tool_implementations):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # 检查是否有工具调用
        if message.tool_calls:
            # 将助手的消息加入历史
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # 执行工具
                if function_name in tool_implementations:
                    result = tool_implementations[function_name](**arguments)
                else:
                    result = f"未知工具: {function_name}"
                
                # 将工具结果加入历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            # 没有工具调用，返回最终答案
            return message.content


# 工具实现
def get_weather(city, unit="celsius"):
    # 实际实现会调用天气 API
    return {"temp": 25, "condition": "晴", "humidity": 45}

def search_web(query):
    # 实际实现会调用搜索 API
    return f"搜索结果: {query}..."

tool_implementations = {
    "get_weather": get_weather,
    "search_web": search_web
}

# 运行
result = run_agent(
    "北京今天天气怎么样？",
    tools,
    tool_implementations
)
print(result)
```

## ReAct Agent vs Tool-Calling Agent

| 特性 | ReAct Agent | Tool-Calling Agent |
|------|-------------|-------------------|
| 输出格式 | 自由文本 + 解析 | 结构化 JSON |
| 可靠性 | 依赖文本解析，可能出错 | 原生支持，更可靠 |
| 灵活性 | 可输出任意推理过程 | 受限于工具定义 |
| 调试性 | 易于查看推理过程 | 需要额外日志 |
| 兼容性 | 任何 LLM | 需模型支持 Function Calling |

### ReAct 的优势

1. **推理透明**：每一步思考都清晰可见
2. **灵活性强**：不局限于预定义的工具格式
3. **模型无关**：任何能生成文本的 LLM 都可以用

### Function Calling 的优势

1. **可靠性高**：结构化输出，解析不会出错
2. **并行调用**：支持一次调用多个工具
3. **类型安全**：参数有明确的类型定义

## 思维链（Chain-of-Thought）

思维链是一种让模型"逐步思考"的提示技术，与 ReAct 和工具调用配合使用效果更佳。

### 基本原理

```
普通提示：
Q: 小明有 23 个苹果，给了小红 5 个，又买了 8 个，还剩多少？
A: 26 个

思维链提示：
Q: 小明有 23 个苹果，给了小红 5 个，又买了 8 个，还剩多少？
A: 让我一步步思考：
   1. 小明最初有 23 个苹果
   2. 给了小红 5 个，所以剩下 23 - 5 = 18 个
   3. 又买了 8 个，所以现在有 18 + 8 = 26 个
   答案是 26 个
```

### 在 Agent 中使用 CoT

```python
SYSTEM_PROMPT = """
你是一个智能助手，可以使用工具完成任务。

在回答问题时，请遵循以下步骤：
1. 理解问题，识别需要什么信息
2. 制定计划，决定使用什么工具
3. 执行工具调用，获取信息
4. 分析结果，得出结论

每次思考时，请明确写出你的推理过程。
"""
```

## 多工具调用

现代 LLM 支持在一次响应中调用多个工具：

```python
# 用户问题
"比较北京和上海今天的天气"

# LLM 可能一次性调用两个工具
response.tool_calls = [
    {
        "id": "call_1",
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "北京"}'
        }
    },
    {
        "id": "call_2", 
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "上海"}'
        }
    }
]
```

### 并行执行

```python
import asyncio

async def execute_tools_parallel(tool_calls, tool_implementations):
    """并行执行多个工具调用"""
    tasks = []
    for tc in tool_calls:
        func = tool_implementations[tc.function.name]
        args = json.loads(tc.function.arguments)
        tasks.append(asyncio.create_task(func(**args)))
    
    results = await asyncio.gather(*tasks)
    return results
```

## 工具设计的最佳实践

### 1. 清晰的工具描述

```python
# 好的描述
{
    "name": "search_files",
    "description": "在指定目录中搜索匹配模式的文件。返回文件路径列表。",
    "parameters": {
        "properties": {
            "directory": {
                "description": "要搜索的目录路径，必须是绝对路径"
            },
            "pattern": {
                "description": "文件名匹配模式，支持通配符 * 和 ?"
            }
        }
    }
}

# 不好的描述
{
    "name": "search",
    "description": "搜索文件",
    "parameters": {...}
}
```

### 2. 合理的参数设计

```python
# 好的设计：参数有默认值，降低使用难度
{
    "name": "list_files",
    "parameters": {
        "directory": {"type": "string"},
        "recursive": {
            "type": "boolean", 
            "default": false,
            "description": "是否递归搜索子目录"
        },
        "max_results": {
            "type": "integer",
            "default": 100,
            "description": "最大返回结果数"
        }
    }
}
```

### 3. 有意义的错误信息

```python
def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在。请检查路径是否正确。"
    except PermissionError:
        return f"错误：没有权限读取文件 '{path}'。"
    except Exception as e:
        return f"读取文件时出错：{str(e)}"
```

### 4. 工具数量控制

```
工具数量 vs 决策效果

决策质量
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  3-5 个工具：决策清晰
    ▓▓▓▓▓▓▓▓░░░░░░░░░░░  5-10 个工具：开始困惑
    ▓▓▓░░░░░░░░░░░░░░░░  10+ 个工具：决策困难
```

**建议：**
- 核心工具控制在 5-7 个
- 使用工具分类（文件操作、网络请求、代码执行等）
- 考虑动态工具加载

## 完整的 Agent 实现

```python
import json
from openai import OpenAI

class Agent:
    def __init__(self, model="gpt-4o", system_prompt=None):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.tools = []
        self.tool_implementations = {}
        self.messages = []
    
    def default_system_prompt(self):
        return """你是一个智能助手，可以使用工具完成任务。
请根据用户的问题选择合适的工具，并在得到结果后给出清晰的回答。"""
    
    def register_tool(self, definition, implementation):
        """注册一个工具"""
        self.tools.append(definition)
        self.tool_implementations[definition["function"]["name"]] = implementation
    
    def run(self, user_message, max_iterations=10):
        """运行 Agent"""
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            self.messages.append(message)
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result = self._execute_tool(tool_call)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                return message.content
        
        return "达到最大迭代次数，任务可能未完成"
    
    def _execute_tool(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if name in self.tool_implementations:
            try:
                return self.tool_implementations[name](**args)
            except Exception as e:
                return f"工具执行错误：{str(e)}"
        return f"未知工具：{name}"


# 使用示例
agent = Agent()

# 注册工具
agent.register_tool(
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    lambda expression: eval(expression)  # 注意：实际使用需要安全处理
)

# 运行
result = agent.run("计算 (23 + 17) * 2 的结果")
print(result)
```

## 小结

- **ReAct** 是一种将推理和行动交替进行的 Agent 范式
- **Function Calling** 是现代 LLM 原生支持的结构化工具调用方式
- **思维链** 让模型逐步推理，提高复杂任务的准确性
- 工具设计需要注意：清晰的描述、合理的参数、有意义的错误信息、控制工具数量

ReAct 模式的核心价值在于：**让 LLM 能够通过工具与真实世界交互，获取实时信息，执行实际操作**。这是从"聊天机器人"到"智能代理"的关键一步。
