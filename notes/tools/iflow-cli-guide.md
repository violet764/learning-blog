# iFlow CLI 完全使用指南

> iFlow CLI 是一款免费的终端 AI 助手，支持多种免费大模型、SubAgent 代理系统、MCP 扩展。本文介绍如何高效使用 iFlow CLI，以及如何让它更好地为你服务。

## 什么是 iFlow CLI

iFlow CLI 是一款直接在终端中运行的 AI 助手，具有以下特点：

- **完全免费** - 支持 Kimi K2、Qwen3 Coder、DeepSeek v3 等免费模型
- **SubAgent 系统** - 将 CLI 从通用助手转变为专家团队
- **MCP 开放市场** - 一键安装有用的 MCP 工具和智能体扩展
- **多模态支持** - 可以直接粘贴图片（Ctrl+V）
- **自然语言交互** - 用日常对话驱动 AI

---

## 安装与配置

### 安装

**Windows 用户：**

```powershell
# 使用 npm 安装
npm install -g @iflow-ai/iflow-cli

# 启动
iflow
```

**国内用户推荐使用 nvm：**

```powershell
# 下载 nvm
# https://cloud.iflow.cn/iflow-cli/nvm-setup.exe

# 配置镜像
nvm node_mirror https://npmmirror.com/mirrors/node/
nvm npm_mirror https://npmmirror.com/mirrors/npm/

# 安装 Node.js 22
nvm install 22
nvm use 22

# 安装 iFlow CLI
npm install -g @iflow-ai/iflow-cli
```

### 身份验证

首次运行 `iflow` 时，选择：
1. **推荐**：iFlow 原生认证（网页登录）
2. **备选**：使用 API Key

获取 API Key：
1. 注册 [iFlow 账户](https://iflow.cn)
2. 进入个人设置 → 点击"重置"生成 API Key

### 配置文件

配置文件位于 `~/.iflow/settings.json`：

```json
{
  "theme": "Default",
  "selectedAuthType": "iflow",
  "apiKey": "your-api-key",
  "baseUrl": "https://apis.iflow.cn/v1",
  "modelName": "Qwen3-Coder",
  "searchApiKey": "your-api-key"
}
```

**支持的模型：**
- `Qwen3-Coder` - 代码专家
- `Kimi-K2` - 通用大模型
- `DeepSeek-v3` - 深度推理
- `GLM4.5` - 智谱清言

---

## 核心功能

### 四种运行模式

| 模式 | 权限 | 适用场景 |
|------|------|----------|
| **yolo** | 最大权限，可执行任何操作 | 信任环境下的快速开发 |
| **接受编辑** | 仅文件修改权限 | 日常编码 |
| **计划模式** | 先规划后执行 | 复杂功能开发 |
| **默认模式** | 无自动权限 | 谨慎操作 |

### 斜杠命令

在对话中输入 `/` 可触发命令：

| 命令 | 功能 |
|------|------|
| `/init` | 项目初始化，生成 IFLOW.md |
| `/memory` | 管理项目记忆 |
| `/tools` | 查看可用工具 |
| `/clear` | 清空对话历史 |
| `/compress` | 压缩对话，节省 token |
| `/chat` | 对话状态管理（保存/恢复） |
| `/stats` | 查看会话统计 |
| `/mcp` | MCP 服务器管理 |
| `/bug` | 提交问题反馈 |
| `/quit` | 退出 CLI |

### At 命令（文件引用）

使用 `@` 引用文件或目录：

```
@src/main.py 这个文件有什么问题？
@docs/ 总结这个目录下的所有文档
@config.json 解释一下这个配置文件
```

### Shell 命令

使用 `!` 执行系统命令：

```
!ls -la          # 查看文件列表
!git status      # 查看 Git 状态
!npm install     # 安装依赖
```

输入 `!` 切换 Shell 模式，再次输入退出。

---

## SubAgent 代理系统

代理是 iFlow CLI 最强大的特性——将任务委托给专门的子代理处理。

### 内置代理

| 代理 | 职责 | 使用时机 |
|------|------|----------|
| **planner** | 功能规划 | 开始新功能前 |
| **architect** | 系统设计 | 技术选型、架构决策 |
| **tdd-guide** | 测试驱动开发 | 编写新功能、修复 Bug |
| **code-reviewer** | 代码审查 | 每次代码修改后 |
| **security-reviewer** | 安全审查 | 上线前、处理敏感数据 |
| **build-error-resolver** | 构建错误 | 编译失败时 |
| **e2e-runner** | 端到端测试 | 关键用户流程 |
| **refactor-cleaner** | 死代码清理 | 代码优化 |
| **python-reviewer** | Python 审查 | Python 代码后 |
| **database-reviewer** | 数据库审查 | SQL/ORM 相关 |

### 使用代理

**方式一：直接请求**

```
请使用 planner 代理帮我规划用户认证功能
请使用 code-reviewer 审查我刚修改的代码
```

**方式二：自动委托**

iFlow CLI 会根据任务类型自动选择合适的代理：
- 复杂功能请求 → planner
- 代码修改 → code-reviewer
- Bug 修复 → tdd-guide
- 架构决策 → architect

### 并行执行

多个独立任务可以同时执行：

```
# 好的方式（并行）
同时检查 src/auth.ts 和 src/api.ts 中的安全问题

# 不好的方式（串行）
先检查 src/auth.ts，然后再检查 src/api.ts
```

---

## MCP 服务器扩展

MCP（Model Context Protocol）允许连接外部服务和工具。

### 查看可用 MCP

```
/mcp
```

### 常用 MCP

| MCP | 功能 |
|-----|------|
| **context7** | 实时文档查询 |
| **sequential-thinking** | 链式推理 |
| **playwright** | 浏览器自动化 |
| **pdf-reader** | PDF 文档阅读 |
| **chrome-devtools** | Chrome 开发工具 |
| **desktop-commander** | 桌面控制 |
| **code-runner** | 代码执行 |

### 配置 MCP

在 `settings.json` 中添加：

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@iflow-mcp/context7-mcp@1.0.0"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@iflow-mcp/server-sequential-thinking@0.6.2"]
    }
  }
}
```

### 从开放市场安装

访问 [心流开放平台](https://platform.iflow.cn/) 一键安装 MCP。

---

## 项目配置：IFLOW.md

每个项目根目录应创建 `IFLOW.md` 文件，这是项目的"记忆文件"。

### 生成方式

```
/init
```

这会扫描代码库并生成包含以下内容的 IFLOW.md：
- 项目结构
- 代码规范
- 使用约束
- 技术栈说明

### 手动编写模板

```markdown
# 项目名称

## 技术栈
- 前端：Vue 3 + TypeScript + Vite
- 后端：Python + FastAPI
- 数据库：PostgreSQL

## 项目结构
src/
├── components/   # 组件
├── views/        # 页面
├── api/          # API 接口
└── utils/        # 工具函数

## 开发规范
- 使用 pnpm 作为包管理器
- 所有组件使用 TypeScript
- API 必须添加错误处理

## 常用命令
- pnpm dev    # 开发服务器
- pnpm build  # 构建生产版本
- pnpm test   # 运行测试
```

---

## 自定义代理

在 `~/.iflow/agents/` 目录下创建代理定义文件。

### 代理文件格式

```markdown
---
name: my-custom-agent
description: 自定义代理描述
tools: ["Read", "Grep", "Glob"]
model: opus
---

你是一个专门的代理，负责...

## 你的职责
- 任务1
- 任务2

## 输出格式
...
```

### 示例：创建代码审查代理

```markdown
---
name: code-reviewer
description: 代码质量审查专家
tools: ["Read", "Grep", "Glob"]
---

你是代码审查专家。

## 审查维度
1. 代码质量
2. 安全问题
3. 性能问题
4. 可维护性

## 输出格式
### 文件：xxx
#### 问题
| 级别 | 行号 | 描述 | 建议 |
|------|------|------|------|
| 严重 | 42 | 硬编码密钥 | 使用环境变量 |
```

---

## 自定义命令

在 `~/.iflow/commands/` 目录下创建命令定义文件。

### 命令文件格式

```markdown
---
name: review
description: 代码审查命令
---

当代码需要审查时，你将：

1. 检查代码质量
2. 检查安全问题
3. 输出审查报告

## 输出格式
...
```

---

## Hooks 自动化

Hooks 在特定事件发生时自动执行脚本。

### 配置 Hooks

在 `~/.iflow/hooks/hooks.json` 中配置：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "tool == \"Bash\" && tool_input.command matches \"git push\"",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"[Hook] 推送前请确认更改已审查...\""
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"\\.(ts|tsx)$\"",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"[Hook] 文件已编辑，考虑运行类型检查\""
          }
        ]
      }
    ]
  }
}
```

### Hook 类型

| 类型 | 触发时机 |
|------|----------|
| PreToolUse | 工具执行前 |
| PostToolUse | 工具执行后 |
| SessionStart | 会话开始时 |
| SessionEnd | 会话结束时 |
| Stop | 响应结束后 |

---

## 最佳实践

### 高效提问

**具体明确：**

```
# ❌ 模糊
改进这个代码

# ✅ 具体
为 src/auth/login.ts 中的 handleLogin 函数添加输入验证，
使用 zod schema 验证 email 和 password 字段
```

**提供上下文：**

```
# ❌ 缺乏上下文
修复这个 bug

# ✅ 提供上下文
用户在提交注册表单时，如果 email 已存在，
页面显示 500 错误而不是友好提示。
错误日志显示 UniqueConstraintError。
请修复 src/api/register.ts 中的错误处理
```

### 工作流程

```
1. /init          # 初始化项目
2. 描述需求        # 清晰说明要做什么
3. 使用代理        # 复杂任务委托给专门代理
4. 代码审查        # code-reviewer 检查代码
5. 测试验证        # 确保功能正常
6. 提交代码        # git commit
```

### 安全检查清单

每次提交前检查：
- 无硬编码密钥
- 所有用户输入已验证
- SQL 使用参数化查询
- XSS 防护
- 错误消息不泄露敏感信息

---

## 功能对比

| 功能 | iFlow CLI | Claude Code |
|------|-----------|-------------|
| Todo 规划 | ✅ | ✅ |
| SubAgent | ✅ | ✅ |
| 自定义命令 | ✅ | ✅ |
| Plan 模式 | ✅ | ✅ |
| 内置开放市场 | ✅ | ❌ |
| 多模态 | ✅ | ⚠️ 国内受限 |
| 搜索 | ✅ | ❌ |
| **免费** | ✅ | ❌ |
| Hook | ✅ | ✅ |

---

## 常见问题

### 如何切换模型？

编辑 `~/.iflow/settings.json` 中的 `modelName` 字段。

### 如何查看 token 使用量？

```
/stats
```

### 如何保存/恢复对话？

```
/chat save v1      # 保存当前对话
/chat list         # 查看已保存对话
/chat resume v1    # 恢复对话
```

### 如何压缩对话历史？

```
/compress
```

建议在逻辑断点处手动压缩，避免丢失重要上下文。

---

## 相关链接

| 资源 | 链接 |
|------|------|
| 心流开放平台 | https://platform.iflow.cn/ |
| MCP 市场 | https://platform.iflow.cn/mcp |
| 智能体扩展 | https://platform.iflow.cn/agents |
| GitHub 仓库 | https://github.com/iflow-ai/iflow-cli |
