# CLI Agent 实战教程

本教程将带你从零构建一个命令行 AI Agent，具备读写文件、执行命令、搜索代码等能力。

## 什么是 CLI Agent？

CLI Agent 是运行在终端中的 AI 代理，与 ChatGPT 等聊天机器人有本质区别：

| 特性 | 聊天机器人 | CLI Agent |
|------|-----------|-----------|
| 运行环境 | 网页/App | 终端 |
| 文件操作 | 无法访问 | 可读写本地文件 |
| 命令执行 | 无法执行 | 可运行 shell 命令 |
| 代码运行 | 无法运行 | 可执行代码并获取结果 |
| 项目上下文 | 无 | 理解整个代码库 |

**典型应用场景：**
- 代码调试和修复
- 自动化测试
- 文档生成
- 代码重构
- 项目探索

## 项目架构

```
cli-agent/
├── main.py           # 入口文件
├── agent.py          # Agent 核心逻辑
├── tools/
│   ├── __init__.py
│   ├── file_ops.py   # 文件操作工具
│   ├── shell.py      # Shell 命令工具
│   └── search.py     # 搜索工具
├── config.py         # 配置文件
└── requirements.txt
```

## Step 1: 项目初始化

```bash
mkdir cli-agent
cd cli-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install openai pydantic rich
```

创建 `requirements.txt`:

```
openai>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
```

## Step 2: 配置与基础设置

创建 `config.py`:

```python
from pydantic import BaseModel
from typing import List
import os

class AgentConfig(BaseModel):
    """Agent 配置"""
    name: str = "CLI-Agent"
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_steps: int = 10
    allowed_commands: List[str] = ["ls", "cat", "grep", "find", "python", "pytest"]
    max_file_size: int = 100000  # 最大文件大小（字节）
    workspace: str = os.getcwd()

# 系统提示词
SYSTEM_PROMPT = """你是一个强大的命令行 AI 助手，可以帮助用户处理各种编程任务。

你有以下工具可用：
- read_file: 读取文件内容
- write_file: 写入文件
- list_directory: 列出目录内容
- run_command: 执行 shell 命令
- search_files: 搜索文件

工作原则：
1. 在修改文件前，先读取并理解现有内容
2. 执行命令时注意安全，避免危险操作
3. 遇到错误时，分析原因并提供解决方案
4. 保持响应简洁，专注于解决问题
"""
```

## Step 3: 实现工具集

创建 `tools/__init__.py`:

```python
from .file_ops import read_file, write_file, list_directory
from .shell import run_command
from .search import search_files

__all__ = [
    "read_file", "write_file", "list_directory",
    "run_command", "search_files"
]
```

创建 `tools/file_ops.py`:

```python
import os
from typing import List, Optional

def read_file(path: str) -> str:
    """
    读取文件内容。
    
    Args:
        path: 文件路径（绝对路径或相对路径）
    
    Returns:
        文件内容，或错误信息
    """
    try:
        # 安全检查：防止路径遍历攻击
        if ".." in path:
            return "错误：不允许使用 '..' 进行路径遍历"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 限制返回内容大小
        max_size = 50000
        if len(content) > max_size:
            return content[:max_size] + f"\n\n... [文件过大，已截断，总长度: {len(content)}]"
        
        return content
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限读取文件 '{path}'"
    except Exception as e:
        return f"读取文件时出错：{str(e)}"


def write_file(path: str, content: str) -> str:
    """
    写入文件。
    
    Args:
        path: 文件路径
        content: 要写入的内容
    
    Returns:
        操作结果信息
    """
    try:
        # 创建目录（如果不存在）
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"成功写入文件 '{path}'，共 {len(content)} 个字符"
    except PermissionError:
        return f"错误：没有权限写入文件 '{path}'"
    except Exception as e:
        return f"写入文件时出错：{str(e)}"


def list_directory(path: str = ".") -> str:
    """
    列出目录内容。
    
    Args:
        path: 目录路径，默认为当前目录
    
    Returns:
        目录内容列表
    """
    try:
        items = os.listdir(path)
        
        result = []
        for item in sorted(items):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"📄 {item} ({size} bytes)")
        
        return "\n".join(result) if result else "目录为空"
    except FileNotFoundError:
        return f"错误：目录 '{path}' 不存在"
    except PermissionError:
        return f"错误：没有权限访问目录 '{path}'"
    except Exception as e:
        return f"列出目录时出错：{str(e)}"
```

创建 `tools/shell.py`:

```python
import subprocess
from typing import List, Optional
from config import AgentConfig

# 危险命令黑名单
DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf ~", "rm -rf *",
    "mkfs", "dd if=", "> /dev/sd",
    "chmod -R 777", "chown -R",
]

def run_command(command: str, timeout: int = 30) -> str:
    """
    执行 shell 命令。
    
    Args:
        command: 要执行的命令
        timeout: 超时时间（秒）
    
    Returns:
        命令输出或错误信息
    """
    # 安全检查
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in command:
            return f"错误：命令包含危险操作 '{dangerous}'，已阻止执行"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = []
        if result.stdout:
            output.append(f"标准输出:\n{result.stdout}")
        if result.stderr:
            output.append(f"标准错误:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"退出码: {result.returncode}")
        
        return "\n".join(output) if output else "命令执行成功（无输出）"
    
    except subprocess.TimeoutExpired:
        return f"错误：命令执行超时（{timeout}秒）"
    except Exception as e:
        return f"执行命令时出错：{str(e)}"
```

创建 `tools/search.py`:

```python
import os
import fnmatch
from typing import List, Optional

def search_files(pattern: str, directory: str = ".", max_results: int = 20) -> str:
    """
    搜索匹配模式的文件。
    
    Args:
        pattern: 文件名模式（支持通配符 * 和 ?）
        directory: 搜索目录
        max_results: 最大结果数
    
    Returns:
        匹配的文件列表
    """
    try:
        matches = []
        
        for root, dirs, files in os.walk(directory):
            # 跳过隐藏目录和常见排除目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git']]
            
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    rel_path = os.path.relpath(os.path.join(root, filename), directory)
                    matches.append(rel_path)
                    
                    if len(matches) >= max_results:
                        break
            
            if len(matches) >= max_results:
                break
        
        if matches:
            return f"找到 {len(matches)} 个匹配文件:\n" + "\n".join(matches)
        else:
            return f"未找到匹配 '{pattern}' 的文件"
    
    except Exception as e:
        return f"搜索文件时出错：{str(e)}"


def search_in_files(keyword: str, directory: str = ".", file_pattern: str = "*.py") -> str:
    """
    在文件中搜索关键词。
    
    Args:
        keyword: 搜索关键词
        directory: 搜索目录
        file_pattern: 文件类型过滤
    
    Returns:
        包含关键词的文件和行号
    """
    try:
        results = []
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git']]
            
            for filename in files:
                if fnmatch.fnmatch(filename, file_pattern):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if keyword.lower() in line.lower():
                                    rel_path = os.path.relpath(filepath, directory)
                                    results.append(f"{rel_path}:{line_num}: {line.strip()}")
                    except:
                        continue
        
        if results:
            return "\n".join(results[:50])  # 限制结果数量
        else:
            return f"未找到包含 '{keyword}' 的内容"
    
    except Exception as e:
        return f"搜索时出错：{str(e)}"
```

## Step 4: Agent 核心实现

创建 `agent.py`:

```python
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from config import AgentConfig, SYSTEM_PROMPT
from tools import read_file, write_file, list_directory, run_command, search_files

console = Console()

class CLIAgent:
    """命令行 AI Agent"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.client = OpenAI()
        self.messages: List[Dict[str, Any]] = []
        self.tools = self._define_tools()
        self.tool_implementations = {
            "read_file": lambda args: read_file(args["path"]),
            "write_file": lambda args: write_file(args["path"], args["content"]),
            "list_directory": lambda args: list_directory(args.get("path", ".")),
            "run_command": lambda args: run_command(args["command"]),
            "search_files": lambda args: search_files(args["pattern"], args.get("directory", ".")),
        }
    
    def _define_tools(self) -> List[Dict]:
        """定义工具列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "写入文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径"},
                            "content": {"type": "string", "description": "文件内容"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "列出目录内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "目录路径，默认当前目录"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "执行 shell 命令",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "要执行的命令"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "搜索匹配模式的文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "文件名模式"},
                            "directory": {"type": "string", "description": "搜索目录"}
                        },
                        "required": ["pattern"]
                    }
                }
            }
        ]
    
    def run(self, user_message: str) -> str:
        """运行 Agent 处理用户消息"""
        # 初始化消息历史
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        for step in range(self.config.max_steps):
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=self.config.temperature
            )
            
            message = response.choices[0].message
            self.messages.append(message)
            
            # 处理工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result = self._execute_tool(tool_call)
                    
                    # 显示工具调用信息
                    console.print(Panel(
                        f"[yellow]工具调用:[/] {tool_call.function.name}\n"
                        f"[yellow]参数:[/] {tool_call.function.arguments}\n"
                        f"[yellow]结果:[/] {result[:200]}...",
                        title="🔧 Tool Call",
                        border_style="blue"
                    ))
                    
                    # 将结果加入消息历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                # 没有工具调用，返回最终答案
                return message.content or "无法生成回答"
        
        return "达到最大步数限制"
    
    def _execute_tool(self, tool_call) -> str:
        """执行工具调用"""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "错误：无法解析工具参数"
        
        if name in self.tool_implementations:
            try:
                return self.tool_implementations[name](args)
            except Exception as e:
                return f"工具执行错误：{str(e)}"
        
        return f"未知工具：{name}"
    
    def chat_loop(self):
        """交互式聊天循环"""
        console.print(Panel(
            "[bold green]CLI Agent 已启动！[/]\n"
            "输入你的问题，输入 'exit' 或 'quit' 退出。",
            title="🤖 CLI Agent",
            border_style="green"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold blue]你:[/] ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]再见！[/]")
                    break
                
                if not user_input:
                    continue
                
                # 处理用户输入
                with console.status("[bold green]思考中...[/]"):
                    response = self.run(user_input)
                
                # 显示回答
                console.print(Panel(
                    Markdown(response),
                    title="🤖 Agent",
                    border_style="green"
                ))
            
            except KeyboardInterrupt:
                console.print("\n[yellow]已中断[/]")
                break
            except Exception as e:
                console.print(f"[red]错误：{str(e)}[/]")
```

## Step 5: 主程序入口

创建 `main.py`:

```python
#!/usr/bin/env python3
"""CLI Agent 入口文件"""

import sys
from agent import CLIAgent
from config import AgentConfig

def main():
    # 解析命令行参数
    config = AgentConfig()
    
    # 如果提供了参数，作为单次查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        agent = CLIAgent(config)
        response = agent.run(query)
        print(response)
    else:
        # 否则进入交互模式
        agent = CLIAgent(config)
        agent.chat_loop()

if __name__ == "__main__":
    main()
```

## Step 6: 运行测试

```bash
# 设置 API Key
export OPENAI_API_KEY="your-api-key"

# 交互模式
python main.py

# 单次查询
python main.py "列出当前目录的文件"
python main.py "读取 README.md 文件"
python main.py "帮我找一下所有的 Python 文件"
```

## 进阶功能

### 1. 添加 Python 代码执行

```python
def run_python(code: str) -> str:
    """在沙箱中执行 Python 代码"""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout or result.stderr
    finally:
        os.unlink(temp_file)
```

### 2. 添加上下文压缩

```python
def compress_history(messages: List[Dict], max_tokens: int = 4000) -> List[Dict]:
    """压缩对话历史"""
    # 保留系统消息
    system_messages = [m for m in messages if m["role"] == "system"]
    
    # 压缩其他消息
    other_messages = [m for m in messages if m["role"] != "system"]
    
    # 简单策略：保留最近的消息
    # 更好的策略：使用 LLM 生成摘要
    compressed = other_messages[-10:]  # 保留最近 10 条
    
    return system_messages + compressed
```

### 3. 添加确认机制

```python
def ask_confirmation(action: str) -> bool:
    """对危险操作请求用户确认"""
    response = input(f"即将执行: {action}\n确认执行？(y/n): ")
    return response.lower() == 'y'

# 在工具执行前检查
if name == "run_command" and args["command"].startswith("rm"):
    if not ask_confirmation(args["command"]):
        return "用户取消了操作"
```

### 4. 使用 MCP（Model Context Protocol）

MCP 是一个标准化协议，用于连接 AI 模型与外部工具：

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# 创建 MCP 服务器连接
desktop_commander = MCPServerStdio(
    command="npx",
    args=["-y", "@wonderwhy-er/desktop-commander"],
    tool_prefix="desktop"
)

# 创建 Agent
agent = Agent(
    model="gpt-4o",
    mcp_servers=[desktop_commander]
)
```

## 安全最佳实践

### 1. 命令白名单

```python
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find",
    "python", "pytest", "git status", "git log"
}

def is_command_allowed(command: str) -> bool:
    base_cmd = command.split()[0]
    return base_cmd in ALLOWED_COMMANDS
```

### 2. 路径限制

```python
def is_path_allowed(path: str, allowed_dirs: List[str]) -> bool:
    """检查路径是否在允许的目录内"""
    abs_path = os.path.abspath(path)
    return any(
        abs_path.startswith(os.path.abspath(d)) 
        for d in allowed_dirs
    )
```

### 3. 操作日志

```python
import logging

logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_tool_call(tool_name: str, args: dict, result: str):
    logging.info(f"Tool: {tool_name}, Args: {args}, Result: {result[:100]}")
```

## 小结

构建一个 CLI Agent 的关键步骤：

1. **定义工具集**：明确 Agent 能做什么
2. **实现安全措施**：防止危险操作
3. **设计交互循环**：让 Agent 能持续工作
4. **优化上下文管理**：避免 token 限制
5. **添加用户确认**：对敏感操作请求批准

CLI Agent 的核心是"**LLM + 工具 + 循环**"，但真正的价值在于：
- **上下文工程**：如何在正确的时间提供正确的信息
- **工具设计**：如何让工具既强大又安全
- **用户体验**：如何让交互流畅自然

下一步可以探索：多 Agent 协作、更复杂的工具链、与其他系统集成等。
