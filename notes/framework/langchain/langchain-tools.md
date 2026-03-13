# 工具调用与集成

工具（Tools）是 LangChain Agent 的核心能力扩展。通过工具，Agent 可以与外部世界交互，执行搜索、计算、API调用等操作。本章将介绍如何定义、使用和集成各种工具。

---

## 🔧 工具基础概念

### 什么是工具

工具是 Agent 可以调用的函数，包含三个关键要素：

```
┌─────────────────────────────────────────┐
│                Tool 结构                 │
├─────────────────────────────────────────┤
│                                         │
│  📛 名称 (name)                         │
│     └─ 工具的唯一标识                   │
│                                         │
│  📝 描述 (description)                  │
│     └─ 告诉模型何时使用此工具           │
│                                         │
│  ⚙️ 函数 (function)                     │
│     └─ 实际执行的代码                   │
│                                         │
│  📥 参数模式 (args_schema)              │
│     └─ 定义输入参数的结构和类型         │
│                                         │
└─────────────────────────────────────────┘
```

### 工具的作用

| 场景 | 工具示例 |
|------|----------|
| 信息获取 | 搜索引擎、数据库查询、API调用 |
| 计算 | 数学运算、数据分析、代码执行 |
| 文件操作 | 读写文件、文档处理 |
| 外部服务 | 发送邮件、调用云服务 |

---

## 🛠️ 定义工具

### 方式一：@tool 装饰器（推荐）

最简单的方式，通过装饰器自动推断参数类型：

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """
    将两个数字相乘
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之积
    """
    return a * b

# 工具会自动从类型注解和文档字符串中推断参数结构
print(multiply.name)         # "multiply"
print(multiply.description)  # "将两个数字相乘"
```

### 方式二：StructuredTool 类

需要更多控制时使用：

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 定义参数结构
class CalculatorInput(BaseModel):
    a: float = Field(description="第一个数字")
    b: float = Field(description="第二个数字")
    operation: str = Field(description="运算类型：add/subtract/multiply/divide")

# 定义函数
def calculator(a: float, b: float, operation: str) -> float:
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float('inf')
    }
    return operations.get(operation, 0)

# 创建工具
calculator_tool = StructuredTool(
    name="calculator",
    description="执行基本数学运算",
    func=calculator,
    args_schema=CalculatorInput
)
```

### 方式三：从函数创建

```python
from langchain_core.tools import Tool

def get_word_length(word: str) -> int:
    return len(word)

# 简单创建（适合单参数函数）
tool = Tool(
    name="word_length",
    description="返回单词的长度",
    func=get_word_length
)
```

---

## 📝 工具定义最佳实践

### 编写好的描述

描述是模型决定是否使用工具的关键：

```python
@tool
def search_web(query: str) -> str:
    """
    在网络上搜索信息
    
    当需要查询以下内容时使用此工具：
    - 最新新闻和事件
    - 实时数据（天气、股价等）
    - 具体的产品或服务信息
    - 你不确定的事实性问题
    
    不适用于：
    - 常识性问题
    - 数学计算
    - 翻译任务
    
    Args:
        query: 搜索查询，应该简洁具体
    """
    # 实际实现...
    pass
```

### 使用 Pydantic 进行参数验证

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Literal
from datetime import date

class WeatherInput(BaseModel):
    """天气查询参数"""
    city: str = Field(description="城市名称")
    date: date = Field(description="查询日期，格式：YYYY-MM-DD")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )
    
    @validator('city')
    def city_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('城市名不能为空')
        return v.strip()

@tool(args_schema=WeatherInput)
def get_weather(city: str, date: date, unit: str = "celsius") -> str:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        date: 查询日期
        unit: 温度单位
    """
    # 实际实现会调用天气API
    return f"{city}在{date}的天气：晴，温度25{unit[0].upper()}"
```

---

## 📦 内置工具

LangChain 提供了许多内置工具，可以直接使用。

### 常用内置工具

```python
from langchain_community.tools import (
    # 搜索工具
    DuckDuckGoSearchRun,
    GoogleSearchAPIWrapper,
    BingSearchAPIWrapper,
    
    # 维基百科
    WikipediaQueryRun,
    
    # Shell 命令
    ShellTool,
    
    # Python REPL
    PythonREPLTool,
    
    # 数学计算
    WolframAlphaQueryRun,
)

# DuckDuckGo 搜索（无需API Key）
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
result = search.invoke("Python最新版本")

# 维基百科
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wiki.invoke("人工智能")

# Python REPL（执行代码）
from langchain_experimental.tools import PythonREPLTool
python_repl = PythonREPLTool()
result = python_repl.invoke("print([x**2 for x in range(10)])")
```

### 安装额外依赖

```bash
# 搜索相关
pip install duckduckgo-search
pip install wikipedia

# Google搜索
pip install google-api-python-client

# Wolfram Alpha
pip install wolframalpha
```

---

## 🌐 工具集成示例

### 集成搜索 API

```python
from langchain_core.tools import tool
import requests

@tool
def search_news(query: str, limit: int = 5) -> str:
    """
    搜索最新新闻
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量
    """
    # 使用 NewsAPI（需要API Key）
    api_key = "your-api-key"
    url = f"https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "pageSize": limit,
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["status"] != "ok":
        return "搜索失败"
    
    results = []
    for article in data["articles"][:limit]:
        results.append(f"- {article['title']}\n  {article['url']}")
    
    return "\n".join(results)
```

### 集成数据库查询

```python
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
import pandas as pd

# 数据库连接
engine = create_engine("sqlite:///company.db")

@tool
def query_database(sql: str) -> str:
    """
    执行SQL查询并返回结果
    
    Args:
        sql: SQL查询语句
    """
    try:
        # 安全检查（仅允许SELECT）
        if not sql.strip().upper().startswith("SELECT"):
            return "只允许执行SELECT查询"
        
        df = pd.read_sql(text(sql), engine)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"查询错误：{str(e)}"
```

### 集成外部 API

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import requests

class WeatherInput(BaseModel):
    city: str = Field(description="城市名称（英文）")
    country: Optional[str] = Field(default=None, description="国家代码")

@tool(args_schema=WeatherInput)
def get_current_weather(city: str, country: str = None) -> str:
    """
    获取当前天气
    
    使用 OpenWeatherMap API 获取实时天气数据
    """
    api_key = "your-openweather-api-key"
    
    # 构建查询
    location = f"{city},{country}" if country else city
    url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",
        "lang": "zh_cn"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200:
        return f"获取天气失败：{data.get('message', '未知错误')}"
    
    # 解析天气数据
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    
    return f"{city}当前天气：{weather}，温度{temp}℃（体感{feels_like}℃），湿度{humidity}%"
```

---

## 🔄 工具链与组合

### 工具链

将多个工具串联执行：

```python
from langchain_core.tools import tool

@tool
def get_user_id(username: str) -> int:
    """根据用户名获取用户ID"""
    # 模拟数据库查询
    users = {"alice": 1, "bob": 2, "charlie": 3}
    return users.get(username, -1)

@tool
def get_user_orders(user_id: int) -> str:
    """根据用户ID获取订单列表"""
    # 模拟订单查询
    orders = {
        1: ["订单A001", "订单A002"],
        2: ["订单B001"],
        3: ["订单C001", "订单C002", "订单C003"]
    }
    return str(orders.get(user_id, []))

# Agent 会自动链接这两个工具
# 用户问："alice的订单有哪些？"
# Agent 会先调用 get_user_id("alice") 得到 1
# 然后调用 get_user_orders(1) 得到订单列表
```

### 条件工具选择

```python
from langchain_core.tools import tool
from typing import Literal

@tool
def process_text(text: str, operation: Literal["uppercase", "lowercase", "reverse"]) -> str:
    """
    处理文本
    
    Args:
        text: 要处理的文本
        operation: 操作类型
            - uppercase: 转大写
            - lowercase: 转小写  
            - reverse: 反转文本
    """
    operations = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "reverse": text[::-1]
    }
    return operations.get(operation, text)
```

---

## 🎯 完整示例：多功能工具集

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
import json
from datetime import datetime

# ========== 定义工具 ==========

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量")

@tool(args_schema=SearchInput)
def mock_search(query: str, num_results: int = 5) -> str:
    """
    模拟搜索引擎搜索
    
    用于搜索各类信息，包括：
    - 技术文档和教程
    - 产品信息
    - 新闻资讯
    """
    # 模拟搜索结果
    mock_data = {
        "Python": [
            {"title": "Python官方文档", "url": "https://docs.python.org"},
            {"title": "Python教程 - 菜鸟教程", "url": "https://runoob.com/python"},
        ],
        "AI": [
            {"title": "OpenAI API文档", "url": "https://platform.openai.com"},
            {"title": "LangChain文档", "url": "https://python.langchain.com"},
        ],
    }
    
    for key, results in mock_data.items():
        if key.lower() in query.lower():
            return json.dumps(results[:num_results], ensure_ascii=False)
    
    return f"未找到与'{query}'相关的结果"

@tool
def current_time() -> str:
    """
    获取当前时间
    
    当用户询问当前时间、日期或需要时间信息时使用
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式，如 "2+3*4" 或 "sqrt(16)"
    """
    import math
    
    # 支持的数学函数
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # 安全检查
        allowed = set("0123456789+-*/.() sqrtincotagl10ep")
        if all(c in allowed for c in expression.lower()):
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"计算结果：{result}"
        return "表达式包含不允许的字符"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def json_formatter(json_string: str) -> str:
    """
    格式化JSON字符串
    
    Args:
        json_string: 需要格式化的JSON字符串
    """
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON解析错误：{str(e)}"

@tool
def todo_list(tasks: str, action: str = "add") -> str:
    """
    管理待办事项
    
    Args:
        tasks: 任务内容（多个任务用逗号分隔）
        action: 操作类型 - "add"添加, "clear"清空
    """
    # 使用全局变量存储（实际应用应使用数据库）
    if not hasattr(todo_list, "storage"):
        todo_list.storage = []
    
    if action == "clear":
        todo_list.storage = []
        return "待办事项已清空"
    
    new_tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    todo_list.storage.extend(new_tasks)
    
    result = "📋 当前待办事项：\n"
    for i, task in enumerate(todo_list.storage, 1):
        result += f"  {i}. ☐ {task}\n"
    
    return result

# ========== 创建 Agent ==========

def create_assistant():
    """创建多功能助手"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [mock_search, current_time, calculate, json_formatter, todo_list]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个多功能助手，可以使用以下工具帮助用户：
        
        1. mock_search - 搜索网络信息
        2. current_time - 获取当前时间
        3. calculate - 执行数学计算
        4. json_formatter - 格式化JSON
        5. todo_list - 管理待办事项
        
        根据用户需求选择合适的工具，如果不需要工具直接回答即可。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(model, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )

# ========== 使用示例 ==========

if __name__ == "__main__":
    assistant = create_assistant()
    
    # 测试不同功能
    queries = [
        "现在几点了？",
        "帮我算一下 123 * 456",
        "搜索Python教程",
        "帮我添加待办：写代码、开会、写报告",
        '格式化这个JSON：{"name":"test","value":123}',
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"用户：{query}")
        result = assistant.invoke({"input": query})
        print(f"助手：{result['output']}")
```

---

## 📋 工具开发检查清单

### ✅ 定义工具时

- [ ] 工具名称简洁且具有描述性
- [ ] 文档字符串清晰说明用途和使用场景
- [ ] 参数有明确的类型注解和描述
- [ ] 考虑参数验证和错误处理

### ✅ 集成外部服务时

- [ ] API Key 通过环境变量管理
- [ ] 实现适当的错误处理
- [ ] 考虑请求超时和重试
- [ ] 注意敏感信息保护

### ✅ 测试工具时

- [ ] 单独测试工具功能
- [ ] 测试边界情况和错误输入
- [ ] 验证 Agent 能正确选择工具
- [ ] 检查工具执行效率

---

## 🔗 相关章节

- [智能体开发](./langchain-agents.md) - 如何让 Agent 使用工具
- [链与 LCEL](./langchain-chains.md) - 工具也可以作为链的一部分
