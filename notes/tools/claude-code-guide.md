# Claude Code 完全使用指南

> 基于 [everything-claude-code](https://github.com/affaan-m/everything-claude-code) 项目总结，涵盖 Claude Code 的核心概念、最佳实践与高级技巧。

## 什么是 Claude Code

Claude Code 是 Anthropic 官方推出的 **CLI（命令行）AI 编程助手**，直接在终端中运行，能够理解整个代码库并通过自然语言完成软件工程任务：编写代码、修复 Bug、重构、编写测试、生成文档等。

与 IDE 插件不同，Claude Code 以 **Agent（智能体）** 模式运行——它不仅能读写文件，还能执行命令、搜索代码、管理 Git，甚至协调多个子代理完成复杂任务。

---

## 核心架构

### 模型选择策略

Claude Code 支持三种模型，合理选择可节省 60% 以上成本：

| 模型 | 适用场景 | 成本 |
|------|----------|------|
| **Opus** | 架构设计、复杂调试、深度推理 | 最高 |
| **Sonnet** | 日常编码、代码审查、测试编写（**80%+ 任务推荐**） | 中等 |
| **Haiku** | 简单查询、格式化、快速搜索 | 最低 |

推荐配置（在 `settings.json` 中）：

```json
{
  "model": "sonnet",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "50",
    "CLAUDE_CODE_SUBAGENT_MODEL": "haiku"
  }
}
```

> `MAX_THINKING_TOKENS` 默认值为 31999，大多数任务 10000 已足够，减少不必要的 token 消耗。

### 上下文窗口管理

Claude Code 拥有 200k token 的上下文窗口，但实际可用量取决于配置：

- 每个 **MCP 工具描述** 都会占用 token
- 启用过多 MCP 可能将可用上下文压缩至 ~70k
- 建议：启用 MCP 不超过 10 个，活跃工具不超过 80 个

**手动压缩优于自动压缩：**
- 不要在实现过程中途压缩（会丢失变量名和文件路径）
- 在逻辑断点处手动执行 `/compact`：研究完成、里程碑达成、调试结束、放弃失败方案时

---

## 项目配置：CLAUDE.md

每个项目根目录应创建 `CLAUDE.md` 文件，这是 Claude Code 的 **项目级指令文件**，定义了项目特定的规则和上下文。

### 基本模板

```markdown
# 项目名称

## 概述
技术栈：Next.js 14 + TypeScript + Supabase + Tailwind CSS
目的：用户管理后台系统

## 关键规则
- 使用 pnpm 作为包管理器
- 所有组件使用函数式组件 + Hooks
- 数据库查询必须使用参数化查询
- 所有 API 路由必须添加认证检查

## 项目结构
src/
├── app/          # Next.js App Router 页面
├── components/   # 可复用 UI 组件
├── lib/          # 工具函数和配置
├── hooks/        # 自定义 React Hooks
└── types/        # TypeScript 类型定义

## 开发命令
- `pnpm dev` — 启动开发服务器
- `pnpm test` — 运行测试
- `pnpm lint` — 代码检查
- `pnpm build` — 构建生产版本

## Git 规范
- feat: 新功能
- fix: Bug 修复
- docs: 文档更新
- refactor: 重构
```

### 配置层级

Claude Code 支持多层配置，优先级从高到低：

| 层级 | 位置 | 作用 |
|------|------|------|
| 项目级 | `项目根目录/CLAUDE.md` | 项目特定规则 |
| 用户级 | `~/.claude/CLAUDE.md` | 个人偏好和通用规则 |
| 企业级 | 管理员配置 | 组织统一标准 |

---

## 核心工具系统

### 内置工具

Claude Code 内置以下工具能力：

| 工具 | 功能 | 说明 |
|------|------|------|
| **Read** | 读取文件 | 支持代码、图片、PDF、Jupyter Notebook |
| **Write** | 创建文件 | 写入新文件内容 |
| **Edit** | 编辑文件 | 精确字符串替换，保留上下文 |
| **Bash** | 执行命令 | 运行终端命令（git、npm、docker 等） |
| **Glob** | 文件搜索 | 按模式匹配文件名 |
| **Grep** | 内容搜索 | 基于 ripgrep 的正则搜索 |
| **WebFetch** | 网页抓取 | 获取和分析网页内容 |
| **WebSearch** | 网络搜索 | 搜索最新信息 |
| **Task** | 子代理 | 启动专门的子代理处理复杂任务 |

### 常用斜杠命令

在 Claude Code 中输入 `/` 可触发内置命令：

| 命令 | 功能 |
|------|------|
| `/help` | 查看帮助 |
| `/compact` | 压缩上下文，释放 token |
| `/clear` | 清空对话历史 |
| `/cost` | 查看当前会话 token 消耗 |
| `/model` | 切换模型 |
| `/permissions` | 管理工具权限 |

---

## 代理系统（Agents）

代理是 Claude Code 最强大的特性之一——将特定任务委托给 **专门的子代理** 处理，每个代理有独立的工具权限和专业领域。

### 内置代理类型

| 代理 | 职责 | 推荐模型 | 使用时机 |
|------|------|----------|----------|
| **planner** | 功能规划、需求分析 | Opus | 开始新功能前 |
| **architect** | 系统设计、架构决策 | Opus | 技术选型、大型重构 |
| **tdd-guide** | 测试驱动开发 | Sonnet | 编写新功能、修复 Bug |
| **code-reviewer** | 代码审查 | Sonnet | 每次代码修改后（**必须使用**） |
| **security-reviewer** | 安全审查 | Opus | 上线前、处理敏感数据时 |
| **build-error-resolver** | 构建错误修复 | Sonnet | 编译失败时 |
| **e2e-runner** | 端到端测试 | Sonnet | 测试关键用户流程 |
| **refactor-cleaner** | 死代码清理 | Sonnet | 重构优化时 |
| **database-reviewer** | 数据库审查 | Sonnet | SQL/ORM 查询审查 |

### 代理使用示例

```
# 让 planner 代理帮你规划功能
"请使用 planner 代理帮我规划用户认证功能的实现方案"

# 代码审查
"请使用 code-reviewer 代理审查我刚修改的代码"

# 安全检查
"请使用 security-reviewer 代理检查 API 端点的安全性"
```

### 代理协作工作流

`/orchestrate` 命令支持预定义的多代理协作流程：

| 工作流 | 代理链 | 适用场景 |
|--------|--------|----------|
| **feature** | planner → tdd-guide → code-reviewer → security-reviewer | 新功能开发 |
| **bugfix** | planner → tdd-guide → code-reviewer | Bug 修复 |
| **refactor** | architect → code-reviewer → tdd-guide | 代码重构 |
| **security** | security-reviewer → code-reviewer → architect | 安全审计 |

---

## Hooks 事件钩子

Hooks 是 Claude Code 的事件驱动自动化系统，在特定事件发生时自动执行脚本。

### Hook 类型

| 类型 | 触发时机 | 用途 |
|------|----------|------|
| **PreToolUse** | 工具执行前 | 验证、警告、拦截（exit 2 阻止执行） |
| **PostToolUse** | 工具执行后 | 分析输出、提供反馈 |
| **UserPromptSubmit** | 用户发送消息时 | 拦截和处理用户输入 |
| **Stop** | Claude 响应结束后 | 会话结束分析 |
| **SessionStart** | 会话开始时 | 加载上下文、检测环境 |
| **SessionEnd** | 会话结束时 | 持久化状态、提取模式 |
| **PreCompact** | 上下文压缩前 | 保存重要状态 |

### Hook 配置示例

在 `.claude/settings.json` 中配置：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "node .claude/hooks/block-dev-server.js",
        "description": "阻止在非 tmux 环境中运行开发服务器"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\"",
        "description": "编辑后自动格式化"
      }
    ],
    "SessionStart": [
      {
        "command": "node .claude/hooks/load-context.js",
        "description": "加载上次会话上下文"
      }
    ]
  }
}
```

### 实用 Hook 示例

**阻止意外提交敏感文件：**

```javascript
// .claude/hooks/check-secrets.js
const input = JSON.parse(process.argv[1] || '{}');
const command = input.tool_input?.command || '';

if (command.includes('git add') && command.includes('.env')) {
  console.error('⚠️ 检测到尝试提交 .env 文件，已阻止');
  process.exit(2); // exit 2 = 阻止工具执行
}
```

**TypeScript 类型检查：**

```javascript
// .claude/hooks/ts-check.js
const input = JSON.parse(process.argv[1] || '{}');
const filePath = input.tool_input?.file_path || '';

if (filePath.endsWith('.ts') || filePath.endsWith('.tsx')) {
  const { execSync } = require('child_process');
  try {
    execSync('npx tsc --noEmit', { stdio: 'pipe' });
  } catch (e) {
    console.log('TypeScript 类型错误:\n' + e.stdout?.toString());
  }
}
```

---

## Skills 技能系统

Skills 是可复用的知识模块，为特定领域提供最佳实践和模式指导。

### 按领域分类的核心技能

**前端开发：**
- `frontend-patterns` — React、Next.js 模式和最佳实践
- `coding-standards` — TypeScript/JavaScript 编码规范

**后端开发：**
- `backend-patterns` — API 设计、数据库优化、服务端模式
- `api-design` — RESTful API 设计（路由命名、状态码、分页、错误响应）
- `postgres-patterns` — PostgreSQL 查询优化和索引策略

**测试：**
- `tdd-workflow` — 测试驱动开发完整流程
- `e2e-testing` — Playwright 端到端测试模式

**安全：**
- `security-review` — OWASP Top 10 安全检查清单

**DevOps：**
- `deployment-patterns` — CI/CD、Docker、健康检查、回滚策略
- `docker-patterns` — Docker Compose、网络、容器安全
- `database-migrations` — 数据库迁移最佳实践

**特定语言/框架：**
- `python-patterns`、`python-testing` — Python 最佳实践
- `golang-patterns`、`golang-testing` — Go 最佳实践
- `django-patterns`、`django-security` — Django 全栈
- `springboot-patterns`、`springboot-security` — Spring Boot 全栈
- `cpp-coding-standards`、`cpp-testing` — C++ 和 GoogleTest

---

## 测试驱动开发（TDD）

everything-claude-code 将 TDD 作为 **强制性要求**，最低测试覆盖率 80%。

### TDD 工作流（红-绿-重构）

```
1. RED    — 先写测试（必然失败）
2. RUN    — 运行测试，确认失败原因正确
3. GREEN  — 写最少的代码让测试通过
4. RUN    — 运行测试，确认通过
5. REFACTOR — 优化代码，保持测试绿色
6. VERIFY — 确认覆盖率 ≥ 80%
```

### 必须测试的边界情况

```
- null / undefined / 空值
- 空数组 / 空对象
- 无效类型输入
- 边界值（0, -1, MAX_INT）
- 错误路径和异常
- 竞态条件
- 大数据量
- 特殊字符（Unicode、SQL 注入字符串）
```

### 测试类型要求

| 类型 | 覆盖范围 | 运行频率 |
|------|----------|----------|
| **单元测试** | 单个函数/方法 | 每次修改 |
| **集成测试** | 模块间交互 | 每次提交 |
| **端到端测试** | 完整用户流程 | PR 合并前 |

### 使用 /tdd 命令

```
# 启动 TDD 工作流
/tdd "实现用户登录功能"

# Claude 会：
# 1. 先创建测试文件
# 2. 编写失败的测试用例
# 3. 实现最小通过代码
# 4. 重构并验证覆盖率
```

---

## 安全最佳实践

### 每次提交前的安全检查清单

```
□ 无硬编码的密钥/密码/Token
□ 所有用户输入已验证和清理
□ SQL 查询使用参数化（防注入）
□ HTML 输出已转义（防 XSS）
□ CSRF 保护已启用
□ 认证和授权检查完整
□ API 端点有速率限制
□ 错误消息不泄露敏感信息
□ 依赖项无已知漏洞
```

### 常见安全模式

**参数化查询（防 SQL 注入）：**

```typescript
// ❌ 危险
const query = `SELECT * FROM users WHERE id = ${userId}`;

// ✅ 安全
const query = 'SELECT * FROM users WHERE id = $1';
const result = await db.query(query, [userId]);
```

**输入验证：**

```typescript
// ✅ 在系统边界验证所有输入
import { z } from 'zod';

const UserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
  age: z.number().int().min(0).max(150),
});

function createUser(input: unknown) {
  const validated = UserSchema.parse(input);
  return db.users.create(validated);
}
```

### 安全扫描工具

```bash
# 使用 AgentShield 扫描 Claude Code 配置
npx ecc-agentshield scan           # 快速扫描
npx ecc-agentshield scan --fix     # 自动修复安全问题
npx ecc-agentshield scan --opus    # 深度分析（3 个 Opus 代理）
```

---

## 编码规范

### 不可变性原则

```typescript
// ❌ 直接修改原对象
function modify(obj, field, value) {
  obj[field] = value;
  return obj;
}

// ✅ 返回新对象
function update(obj, field, value) {
  return { ...obj, [field]: value };
}
```

### 文件组织

```
- 多个小文件 > 少量大文件
- 典型文件：200-400 行，上限 800 行
- 按功能/领域组织，而非按类型
- 高内聚、低耦合
```

### 避免过度工程

```
- 不要添加未被请求的功能
- 不要为假设的未来需求设计
- 不要为单次使用的操作创建抽象
- 三行相似的代码优于一个过早的抽象
- 只在系统边界（用户输入、外部 API）做验证
- 只在逻辑不自明时添加注释
```

---

## MCP 服务器（外部集成）

MCP（Model Context Protocol）允许 Claude Code 连接外部服务和工具。

### 常用 MCP 配置

| MCP | 功能 | 推荐场景 |
|-----|------|----------|
| **github** | PR/Issue 管理 | 所有 GitHub 项目 |
| **memory** | 跨会话持久化记忆 | 长期项目 |
| **sequential-thinking** | 链式推理 | 复杂问题分析 |
| **supabase** | 数据库操作 | Supabase 项目 |
| **vercel** | 部署管理 | Vercel 部署 |
| **firecrawl** | 网页抓取 | 需要爬取网页时 |
| **context7** | 实时文档查询 | 查询最新 API 文档 |

### 配置方式

在 `.claude/settings.json` 中：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### MCP 性能注意事项

- 每个 MCP 的工具描述都会消耗上下文 token
- 启用过多 MCP 会显著减少可用上下文
- 建议：每个项目只启用需要的 MCP
- 使用 `disabledMcpServers` 按项目禁用不需要的 MCP

---

## 会话管理与记忆持久化

### 跨会话保持上下文

Claude Code 的每次会话默认是独立的。通过以下策略实现记忆持久化：

**方法一：SessionEnd Hook 自动保存**

```javascript
// .claude/hooks/save-session.js
const fs = require('fs');
const summary = {
  timestamp: new Date().toISOString(),
  workingOn: '用户认证功能',
  completedSteps: ['数据库模型', 'API 路由'],
  nextSteps: ['前端表单', '测试'],
  failedApproaches: ['JWT 存储在 localStorage（不安全）']
};
fs.writeFileSync('.claude/session-state.json', JSON.stringify(summary, null, 2));
```

**方法二：CLI 参数注入上下文**

```bash
# 创建上下文别名
alias claude-dev='claude --system-prompt "$(cat ~/.claude/contexts/dev.md)"'
alias claude-review='claude --system-prompt "$(cat ~/.claude/contexts/review.md)"'

# 使用时自动加载对应上下文
claude-dev
```

**方法三：CLAUDE.md 持续更新**

在项目 `CLAUDE.md` 中维护"当前状态"部分：

```markdown
## 当前开发状态
- ✅ 用户注册 API
- ✅ 数据库模型
- 🔄 用户登录（进行中）
- ⬜ 密码重置
- ⬜ OAuth 集成
```

### 手动压缩最佳时机

```
✅ 研究阶段完成，准备开始编码
✅ 一个里程碑完成，开始下一个
✅ 调试完成，准备实施修复
✅ 放弃一个失败的方案，准备尝试新方案

❌ 不要在编码实现过程中压缩
❌ 不要在多文件修改进行到一半时压缩
```

---

## 高级技巧

### 1. 并行工具调用

Claude Code 可以同时执行多个独立的工具调用，提高效率：

```
# 好的请求方式（Claude 会并行执行）
"请同时检查 src/auth.ts 和 src/api.ts 中的安全问题"

# 不好的方式（串行执行）
"先检查 src/auth.ts，然后再检查 src/api.ts"
```

### 2. 持续学习系统（Instinct）

everything-claude-code 提供了基于 **直觉（Instinct）** 的学习系统：

```
1. Hook 在开发过程中捕获提示和工具调用
2. 后台代理（Haiku）检测重复模式
3. 创建原子化的 "instinct"，带有置信度分数（0.3-0.9）
4. 用户修正 → 提高置信度
5. /evolve 将相关 instinct 聚类为 skill/command/agent
6. 支持导出/导入，团队共享
```

**常用命令：**

| 命令 | 功能 |
|------|------|
| `/instinct-status` | 查看所有已学习的直觉及其置信度 |
| `/instinct-export` | 导出直觉，分享给团队 |
| `/instinct-import` | 从团队或其他来源导入直觉 |
| `/evolve` | 将相关直觉聚类为技能/命令/代理 |
| `/skill-create` | 分析 git 历史，生成技能文件 |

### 3. 多模型协作

| 命令 | 功能 |
|------|------|
| `/multi-plan` | 多模型协作规划 |
| `/multi-execute` | 多模型协作执行 |
| `/multi-backend` | 后端多服务工作流 |
| `/multi-frontend` | 前端多服务工作流 |

### 4. 验证循环（Verification Loop）

在提交前执行完整的验证：

```
/verify

# 验证步骤：
# 1. 类型检查（tsc --noEmit）
# 2. 代码检查（eslint）
# 3. 单元测试 + 覆盖率
# 4. 集成测试
# 5. 安全扫描
# 6. 构建验证
```

### 5. 高效的提示技巧

**具体且明确：**

```
# ❌ 模糊
"改进这个代码"

# ✅ 具体
"为 src/auth/login.ts 中的 handleLogin 函数添加输入验证，
使用 zod schema 验证 email 和 password 字段"
```

**提供上下文：**

```
# ❌ 缺乏上下文
"修复这个 bug"

# ✅ 提供上下文
"用户在提交注册表单时，如果 email 已存在，
页面显示 500 错误而不是友好提示。
错误日志显示 UniqueConstraintError。
请修复 src/api/register.ts 中的错误处理"
```

**分步骤请求复杂任务：**

```
# 第一步：了解现状
"请先分析 src/api/ 目录下的认证相关代码，
告诉我当前的认证架构"

# 第二步：规划方案（基于第一步的分析）
"基于你的分析，请使用 /plan 命令规划
添加 OAuth2 Google 登录的实现方案"

# 第三步：确认后执行
"方案 A 看起来不错，请开始实现"
```

---

## 快速参考

### 日常开发工作流

```
1. 打开终端，进入项目目录
2. 运行 claude 启动会话
3. 描述你要做的事情
4. 使用 /plan 规划复杂功能
5. 使用 /tdd 编写测试和代码
6. 使用 code-reviewer 代理审查代码
7. 使用 /verify 验证所有检查通过
8. 提交代码
```

### Token 优化速查表

| 策略 | 节省效果 |
|------|----------|
| Sonnet 替代 Opus | ~60% 成本 |
| MAX_THINKING_TOKENS=10000 | ~30% 思考 token |
| 手动 /compact | 避免上下文浪费 |
| 禁用不用的 MCP | 恢复可用上下文 |
| Haiku 处理子任务 | 子代理成本最小化 |

### 文件组织速查表

```
✅ 200-400 行/文件（上限 800）
✅ 按功能/领域组织
✅ 高内聚、低耦合
✅ 多个小文件优于大文件
❌ 不要按类型组织（所有 utils 放一起）
❌ 不要创建 God Object
```

---

## 安装 Everything Claude Code

### 方式一：作为插件安装（推荐）

```bash
# 在 Claude Code 中执行
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

### 方式二：手动安装

```bash
git clone https://github.com/affaan-m/everything-claude-code.git

# 使用安装脚本（按需选择语言）
./install.sh typescript
./install.sh python golang

# 或手动复制
mkdir -p ~/.claude/rules ~/.claude/agents ~/.claude/commands ~/.claude/skills

# 复制规则（必需）
cp -r everything-claude-code/rules/common/* ~/.claude/rules/

# 复制代理
cp everything-claude-code/agents/*.md ~/.claude/agents/

# 复制命令
cp everything-claude-code/commands/*.md ~/.claude/commands/

# 复制技能
cp -r everything-claude-code/skills/* ~/.claude/skills/
```

---

## 实用资源

| 资源 | 链接 |
|------|------|
| Everything Claude Code 仓库 | https://github.com/affaan-m/everything-claude-code |
| Claude Code 官方文档 | https://docs.anthropic.com/en/docs/claude-code |
| Claude Code GitHub Issues | https://github.com/anthropics/claude-code/issues |
| Anthropic API 文档 | https://docs.anthropic.com |
| AgentShield 安全扫描 | `npx ecc-agentshield scan` |
| Skill Creator | https://ecc.tools |
