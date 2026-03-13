# Function Calling

Function Calling（函数调用）让模型能够调用外部工具和 API，是实现 AI Agent 的核心技术。本章介绍函数定义、参数解析和工具调用的完整流程。

---

## 🎯 什么是 Function Calling

### 基本概念

Function Calling 允许模型生成结构化的函数调用参数，而不是自由文本。这样可以让模型与外部系统交互，获取实时数据或执行操作。

```
用户请求: "北京今天天气怎么样？"
     ↓
模型识别需要调用工具
     ↓
生成函数调用: get_weather(city="北京")
     ↓
程序执行函数获取数据
     ↓
将结果返回给模型
     ↓
模型生成自然语言回复
```

### 应用场景

| 场景 | 示例 |
|------|------|
| **信息查询** | 查询天气、股价、新闻 |
| **数据库操作** | 查询、更新数据 |
| **API 调用** | 调用第三方服务 |
| **计算任务** | 数学运算、数据分析 |
| **代码执行** | 运行代码、执行脚本 |
| **外部操作** | 发送邮件、创建文件 |

---

## 🔧 定义函数

### 函数结构

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",           # 函数名称
            "description": "获取指定城市的天气信息",  # 功能描述
            "parameters": {                  # 参数定义
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]  # 必需参数
            }
        }
    }
]
```

### 参数类型

支持的 JSON Schema 类型：

```python
# 字符串
"name": {
    "type": "string",
    "description": "用户名"
}

# 数字
"age": {
    "type": "integer",
    "description": "年龄"
}

# 布尔
"is_active": {
    "type": "boolean",
    "description": "是否激活"
}

# 枚举
"status": {
    "type": "string",
    "enum": ["pending", "active", "completed"],
    "description": "状态"
}

# 数组
"tags": {
    "type": "array",
    "items": {"type": "string"},
    "description": "标签列表"
}

# 嵌套对象
"address": {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "street": {"type": "string"}
    }
}
```

### 完整函数定义示例

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "在商品数据库中搜索产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food", "books"],
                        "description": "商品类别"
                    },
                    "price_range": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number", "description": "最低价格"},
                            "max": {"type": "number", "description": "最高价格"}
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "下单购买商品",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品ID"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "购买数量"
                    },
                    "shipping_address": {
                        "type": "string",
                        "description": "收货地址"
                    }
                },
                "required": ["product_id", "quantity", "shipping_address"]
            }
        }
    }
]
```

---

## 🚀 基本使用

### 单次函数调用

```python
from openai import OpenAI
import json

client = OpenAI()

# 定义工具
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
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 发送请求
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"  # 让模型决定是否调用工具
)

# 检查是否需要调用工具
message = response.choices[0].message

if message.tool_calls:
    print("模型请求调用工具:")
    for tool_call in message.tool_calls:
        print(f"函数名: {tool_call.function.name}")
        print(f"参数: {tool_call.function.arguments}")
        
        # 解析参数
        args = json.loads(tool_call.function.arguments)
        print(f"解析后的参数: {args}")
```

### 完整的工具调用流程

```python
from openai import OpenAI
import json

client = OpenAI()

# 模拟天气 API
def get_weather(city: str) -> dict:
    """模拟获取天气信息"""
    # 实际应用中这里会调用真实的天气 API
    weather_data = {
        "北京": {"temperature": 25, "weather": "晴天", "humidity": 45},
        "上海": {"temperature": 28, "weather": "多云", "humidity": 60},
        "广州": {"temperature": 32, "weather": "雷阵雨", "humidity": 80}
    }
    return weather_data.get(city, {"temperature": 20, "weather": "未知", "humidity": 50})

# 可用的工具函数
available_functions = {
    "get_weather": get_weather
}

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def run_conversation(user_message: str) -> str:
    """运行对话，处理工具调用"""
    
    # 第一步：发送用户消息
    messages = [{"role": "user", "content": user_message}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    # 第二步：检查是否需要调用工具
    if message.tool_calls:
        # 添加助手消息（包含工具调用）
        messages.append(message)
        
        # 第三步：执行工具调用
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 执行函数
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            # 添加工具响应
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })
        
        # 第四步：获取最终回复
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    # 没有工具调用，直接返回
    return message.content

# 使用
result = run_conversation("北京今天天气怎么样？适合外出吗？")
print(result)
```

---

## ⚙️ tool_choice 参数

### 选项说明

```python
# auto: 让模型决定是否调用工具（默认）
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# none: 不调用任何工具
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="none"
)

# required: 必须调用至少一个工具
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="required"
)

# 指定调用的工具
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

### 使用场景

| 值 | 场景 |
|---|------|
| `auto` | 一般对话，让模型自主判断 |
| `none` | 纯文本生成，不需要工具 |
| `required` | 必须使用工具的场景 |
| 指定函数 | 强制调用特定工具 |

---

## 🔄 多工具调用

### 并行调用

模型可以一次请求调用多个工具：

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "获取城市空气质量",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city):
    return {"city": city, "temperature": 25, "weather": "晴"}

def get_air_quality(city):
    return {"city": city, "aqi": 50, "level": "优"}

available_functions = {
    "get_weather": get_weather,
    "get_air_quality": get_air_quality
}

def run_multi_tool_conversation(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    if message.tool_calls:
        messages.append(message)
        
        # 并行执行所有工具调用
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            result = available_functions[function_name](**function_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(result, ensure_ascii=False)
            })
        
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    return message.content

# 使用：模型可能会同时调用两个工具
result = run_multi_tool_conversation("北京今天的天气和空气质量怎么样？")
print(result)
```

---

## 🎯 实践示例

### 示例1：智能助手

```python
from openai import OpenAI
import json
from datetime import datetime

client = OpenAI()

class SmartAssistant:
    """智能助手，支持多种工具"""
    
    def __init__(self):
        self.tools = self._define_tools()
        self.functions = {
            "get_current_time": self._get_current_time,
            "calculate": self._calculate,
            "search_web": self._search_web,
            "send_email": self._send_email
        }
        self.conversation_history = []
    
    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "时区，如：Asia/Shanghai"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如：2+2, 100*5"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "搜索网络信息",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "发送邮件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string",
                                "description": "收件人邮箱"
                            },
                            "subject": {
                                "type": "string",
                                "description": "邮件主题"
                            },
                            "body": {
                                "type": "string",
                                "description": "邮件内容"
                            }
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            }
        ]
    
    def _get_current_time(self, timezone="Asia/Shanghai"):
        now = datetime.now()
        return {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}
    
    def _calculate(self, expression):
        try:
            result = eval(expression)  # 实际应用中应使用更安全的方式
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _search_web(self, query):
        # 模拟搜索结果
        return {"query": query, "results": [f"关于 {query} 的搜索结果..."]}
    
    def _send_email(self, to, subject, body):
        # 模拟发送邮件
        return {"status": "sent", "to": to, "subject": subject}
    
    def chat(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            tools=self.tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            self.conversation_history.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                # 执行函数
                result = self.functions[func_name](**func_args)
                
                # 添加工具响应
                self.conversation_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            # 获取最终回复
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history
            )
            
            assistant_message = final_response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        
        self.conversation_history.append({"role": "assistant", "content": message.content})
        return message.content

# 使用
assistant = SmartAssistant()

print(assistant.chat("现在几点了？"))
print(assistant.chat("帮我计算 123 * 456"))
print(assistant.chat("刚才计算的结果加上 100 是多少？"))
```

### 示例2：结构化数据提取

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

client = OpenAI()

class Person(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    company: Optional[str] = None

def extract_person_info(text: str) -> Person:
    """从文本中提取人物信息"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_person",
                "description": "从文本中提取人物信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "姓名"},
                        "age": {"type": "integer", "description": "年龄"},
                        "occupation": {"type": "string", "description": "职业"},
                        "company": {"type": "string", "description": "公司"}
                    },
                    "required": ["name"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "从用户输入中提取人物信息，调用 extract_person 函数。"
            },
            {"role": "user", "content": text}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_person"}}
    )
    
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    return Person(**args)

# 使用
texts = [
    "张三今年28岁，是一名软件工程师，在阿里巴巴工作",
    "李四，35岁，医生",
    "王五是我的朋友"
]

for text in texts:
    person = extract_person_info(text)
    print(f"原文: {text}")
    print(f"提取: {person}\n")
```

---

## 📋 最佳实践

### ✅ 函数定义建议

```python
# 1. 提供清晰的描述
{
    "name": "search_products",
    "description": "在商品数据库中搜索产品。支持按关键词、类别和价格范围筛选。",  # 详细描述
    ...
}

# 2. 参数描述要具体
"properties": {
    "query": {
        "type": "string",
        "description": "搜索关键词，支持商品名称、品牌等"  # 说明用途
    }
}

# 3. 使用枚举限制选项
"category": {
    "type": "string",
    "enum": ["electronics", "clothing", "food"],  # 限制可选值
    "description": "商品类别"
}

# 4. 设置默认值（在描述中说明）
"limit": {
    "type": "integer",
    "description": "返回结果数量，默认10",
    "default": 10  # 如果支持
}
```

### ❌ 常见问题

```python
# 1. 描述不清晰
# ❌
"description": "搜索"  # 太模糊
# ✅
"description": "在商品数据库中按关键词搜索产品"

# 2. 缺少必需参数
# ❌ 不指定 required
# ✅ 明确必需参数
"required": ["query"]

# 3. 参数类型错误
# ❌ 期望数字但用字符串
# ✅ 使用正确的类型
"age": {"type": "integer"}
```

---

## 🔗 相关章节

- [Chat Completions](./openai-chat.md) - 对话 API
- [Assistants API](./openai-assistants.md) - 助手 API
- [LangChain](../langchain/index.md) - Agent 开发框架
