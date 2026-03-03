# OpenClaw：自托管 AI Agent 框架详解

## 什么是 OpenClaw？

OpenClaw 是一个开源的**自托管 AI Agent 框架**，由奥地利开发者 Peter Steinberger 创建。它允许你在自己的设备上运行个人 AI 助手，通过你已经在使用的消息平台（WhatsApp、Telegram、Discord、Slack、Signal、iMessage 等）与 AI 交互。

与 ChatGPT 等传统聊天机器人不同，OpenClaw 的 Agent 是**持久化运行**的——它们可以按计划唤醒、本地存储记忆、自主执行多步骤任务。Andrej Karpathy 称其为"我所见过的最令人难以置信的科幻风格的东西"。

OpenClaw 在 2026 年初迅速走红，短短几周内在 GitHub 上获得了超过 18 万颗星，成为增长最快的开源项目之一。

## 核心理念

OpenClaw 的核心洞见是：**把 AI 助手当作基础设施问题，而不是提示工程问题**。

传统的做法是通过精心设计的提示词让 LLM "记住"上下文或安全地行动，而 OpenClaw 在模型周围构建了一个结构化的执行环境：

- **会话管理**：每个对话都有独立的会话状态
- **记忆系统**：支持语义搜索的历史上下文
- **工具沙箱**：安全的工具执行环境
- **消息路由**：统一的多平台消息分发

LLM 提供智能，OpenClaw 提供操作系统。

## 整体架构

OpenClaw 采用 **Hub-and-Spoke（枢纽-辐射）架构**，以 Gateway 为中心控制平面，连接用户输入渠道和 AI Agent 运行时。

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户交互层（Channel Adapters）                 │
│  WhatsApp | Telegram | Discord | Slack | Signal | iMessage | Web │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Gateway（控制平面）                          │
│                   ws://127.0.0.1:18789                          │
│         会话管理 | 访问控制 | 消息路由 | 状态协调                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────┐
│  Agent Runtime │    │   Canvas Server   │    │  CLI / Apps   │
│  (Pi Agent)    │    │   (A2UI Host)     │    │  (控制界面)    │
└───────────────┘    └───────────────────┘    └───────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      工具层（Tools）                              │
│   Bash | Browser | Files | Canvas | Cron | Sessions | Nodes     │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. Gateway（网关控制平面）

Gateway 是整个系统的核心，位于 `src/gateway/server.ts`，运行在 Node.js 22+ 上。

**主要职责：**

- **消息路由**：所有消息平台通过 Gateway 连接和分发
- **访问控制**：白名单验证、配对系统、权限检查
- **会话协调**：管理会话状态、在线状态、健康监控
- **定时任务**：Cron 作业调度

**设计原则：**

- 每个 Host 只有一个 Gateway（防止 WhatsApp 会话冲突）
- 所有 WebSocket 帧都经过 JSON Schema 验证
- 事件驱动而非轮询
- 幂等性保证（side-effect 操作需要 idempotency key）

### 2. Channel Adapters（渠道适配器）

每个消息平台都有专门的适配器，实现统一的接口：

| 适配器 | 技术栈 | 认证方式 |
|--------|--------|----------|
| WhatsApp | Baileys | QR 码配对 |
| Telegram | grammY | Bot Token |
| Discord | discord.js | Bot Token |
| Slack | Bolt | OAuth |
| Signal | signal-cli | 设备配对 |
| iMessage | BlueBubbles | macOS 集成 |

**适配器职责：**

1. **认证**：处理各平台的身份验证
2. **入站解析**：提取文本、媒体附件、回复上下文
3. **访问控制**：白名单检查、DM 配对策略
4. **出站格式化**：Markdown 转换、消息分块、媒体上传

### 3. Agent Runtime（代理运行时）

Agent Runtime 是 AI 交互的核心执行环境，位于 `src/agents/piembeddedrunner.ts`。

**执行流程：**

```
消息到达 → 会话解析 → 上下文组装 → 模型调用 → 工具执行 → 状态持久化
```

#### 会话类型

| 会话类型 | 标识符 | 权限级别 |
|----------|--------|----------|
| 主会话 | `agent:<id>:main` | 完全权限 |
| 私信会话 | `agent:<id>:<channel>:dm:<id>` | 沙箱隔离 |
| 群组会话 | `agent:<id>:<channel>:group:<id>` | 受限权限 |

#### 系统提示架构

OpenClaw 通过组合多个源文件构建系统提示：

```
~/.openclaw/workspace/
├── AGENTS.md      # 核心指令（必需）
├── SOUL.md        # 个性与语气（可选）
├── TOOLS.md       # 工具使用约定（可选）
├── MEMORY.md      # 长期记忆（仅主会话）
└── skills/
    └── <skill>/
        └── SKILL.md  # 技能定义
```

### 4. Memory System（记忆系统）

OpenClaw 维护可搜索的对话记忆，存储在 `~/.openclaw/memory/<agentId>.sqlite`。

**工作原理：**

1. **自动索引**：消息到达时自动索引
2. **混合搜索**：向量相似度 + BM25 关键词匹配
3. **语义检索**：查询相关历史上下文注入当前对话

**嵌入模型选择优先级：**

1. 本地嵌入模型（`local.modelPath`）
2. OpenAI Embeddings（有 API Key）
3. Gemini Embeddings（有 API Key）
4. 禁用记忆搜索

### 5. Canvas 与 A2UI

Canvas 是 Agent 驱动的可视化工作区，运行在独立端口（默认 18793）。

**A2UI（Agent-to-UI）框架：**

Agent 生成带有特殊属性的 HTML，创建交互式界面：

```html
<div a2ui-component="task-list">
  <button a2ui-action="complete" a2ui-param-id="123">
    标记完成
  </button>
</div>
```

用户点击按钮 → 客户端发送事件 → Canvas 转发给 Agent → Agent 处理并更新界面。

### 6. 插件系统

OpenClaw 通过插件扩展功能，位于 `extensions/` 目录：

- **渠道插件**：添加新的消息平台
- **记忆插件**：替代存储后端（向量数据库、知识图谱）
- **工具插件**：自定义能力
- **模型插件**：自定义 LLM 提供商

插件发现机制：扫描 `package.json` 中的 `openclaw.extensions` 字段。

## 消息处理流程

以 WhatsApp 消息为例，完整流程如下：

### Phase 1: 接收

Baileys 库接收 WebSocket 事件 → WhatsApp 适配器解析消息

### Phase 2: 访问控制

```
发送者是否在白名单？ 
  ├─ 否 → 消息被拒绝
  └─ 是 → 继续
首次私信？
  ├─ 是 → 触发配对流程
  └─ 否 → 路由到会话
```

### Phase 3: 会话解析

- 直接来自用户 → `main` 会话（完全权限）
- 私信渠道 → `dm:<channel>:<id>`（沙箱隔离）
- 群组消息 → `group:<channel>:<id>`（受限权限）

### Phase 4: 上下文组装

1. 加载会话历史
2. 读取 AGENTS.md、SOUL.md、TOOLS.md
3. 注入相关技能
4. 语义搜索历史上下文

### Phase 5: 模型调用

流式请求到配置的模型提供商（Anthropic、OpenAI、Gemini、本地模型）

### Phase 6: 工具执行

```
模型请求工具调用
  ├─ bash 命令 → 可能执行在 Docker 沙箱
  ├─ 浏览器操作 → Chromium CDP 自动化
  └─ 工具结果 → 流式返回模型
```

### Phase 7: 响应交付

响应块流式返回 → WhatsApp 适配器格式化 → 通过 Baileys 发送 → 持久化会话状态

## 安全架构

### 访问控制

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"],
      "groups": {
        "*": { "requireMention": true }
      }
    }
  }
}
```

- **白名单**：指定允许交互的用户
- **配对策略**：`pairing`（需批准）、`open`（接受所有）、`disabled`（拒绝）
- **群组规则**：提及要求、群组白名单

### 沙箱隔离

```json
{
  "agents": {
    "defaults": {
      "sandbox": {
        "mode": "non-main"
      }
    }
  }
}
```

- **main 会话**：直接在主机执行（受信任）
- **非 main 会话**：在 Docker 容器中执行（隔离）

### 凭证管理

- 敏感数据存储在 `~/.openclaw/credentials/`
- 文件权限 0600（仅所有者读写）
- 自动排除在版本控制之外

## 安装与配置

### 快速开始

```bash
# 全局安装
npm install -g openclaw@latest

# 运行引导向导
openclaw onboard --install-daemon

# 启动 Gateway
openclaw gateway --port 18789
```

### 最小配置

`~/.openclaw/openclaw.json`:

```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-6"
  }
}
```

### 多 Agent 路由

```json
{
  "agents": {
    "mapping": {
      "group:discord:123456": {
        "workspace": "~/.openclaw/workspaces/discord-bot",
        "model": "anthropic/claude-sonnet-4-5",
        "systemPromptOverrides": {
          "SOUL.md": "你是一个友好的 Discord 管理员..."
        }
      },
      "dm:telegram:*": {
        "workspace": "~/.openclaw/workspaces/support-agent",
        "model": "openai/gpt-4o",
        "sandbox": { "mode": "always" }
      }
    }
  }
}
```

## 安全风险与最佳实践

### 已知风险

1. **权限过大**：Agent 以用户身份运行，继承所有权限
2. **恶意插件**：ClawHub 上发现 400+ 恶意技能
3. **凭证暴露**：本地存储的凭证可能被窃取
4. **Moltbook 泄露**：数据库暴露导致用户数据泄露

### 最佳实践

1. **最小权限原则**：只授予必要的工具权限
2. **沙箱隔离**：非信任会话使用 Docker 沙箱
3. **凭证管理**：使用外部 secrets manager
4. **审查插件**：安装前仔细检查第三方技能
5. **定期审计**：运行 `openclaw doctor` 检查配置

## 生态系统

### 官方应用

| 应用 | 平台 | 功能 |
|------|------|------|
| macOS App | macOS | 菜单栏控制、Voice Wake、远程管理 |
| iOS Node | iOS | Canvas、语音唤醒、相机、屏幕录制 |
| Android Node | Android | 完整设备能力、通知、位置、短信 |

### 技能平台

ClawHub 是技能注册中心，Agent 可以自动搜索和拉取新技能。

### 会话工具（Agent 间通信）

- `sessions_list`：发现活跃会话
- `sessions_send`：向其他会话发送消息
- `sessions_history`：获取其他会话的对话历史
- `sessions_spawn`：编程创建新会话

## 总结

OpenClaw 代表了 AI Agent 发展的重要方向：**持久化、多渠道、自托管**。

**优势：**

- 真正的自主 Agent，而非被动响应的聊天机器人
- 支持几乎所有主流消息平台
- 开源自托管，数据隐私可控
- 高度可扩展的插件系统

**挑战：**

- 安全风险需要谨慎管理
- 配置复杂度较高
- 对模型质量依赖大

OpenClaw 不是"奇点"，也不是 AGI，它是构建在 LLM 之上的复杂自动化软件。它指向的未来是：**持久的个人 Agent 可以跨越你的数字生活行动**。但这个未来也需要我们认真对待安全问题——正如 Steinberger 本人所承认的，没有"绝对安全"的设置。
