# Claude Code 架构深度解析

Claude Code 是 Anthropic 开发的 AI 编程助手 CLI 工具，其架构设计堪称 AI Agent 开发的最佳实践教材。本文档深入分析其核心架构设计与实现原理。

> **为什么学习 Claude Code 架构？**
> 
> 如果你正在开发 AI Agent、Chatbot 或任何与 LLM 交互的应用，Claude Code 的设计模式值得深入学习：
> - 如何实现流式响应，让用户感知延迟更低
> - 如何设计权限系统，保证 AI 操作安全可控
> - 如何管理有限的上下文窗口，避免 token 超限
> - 如何支持多 Agent 协作，实现复杂任务分解

## 项目概览

Claude Code 源码是一个庞大的 TypeScript 项目，其规模和复杂度令人印象深刻。从这些数据可以看出，一个成熟的 AI 编程工具需要多少工程量。

| 指标 | 数值 | 说明 |
|------|------|------|
| 文件总数 | 1,884 个 (.ts/.tsx) | TypeScript 源码文件数量 |
| 代码体积 | 33 MB | 不含 node_modules |
| 核心入口文件 | main.tsx (4,683 行) | 启动初始化逻辑 |
| 查询引擎 | QueryEngine.ts (1,295 行) + query.ts (1,729 行) | 对话循环核心 |
| 内置工具 | 43 个独立模块 | Bash、Read、Edit 等 |
| 斜杠命令 | 100+ 个 | 如 /commit、/review |
| React Hooks | 85 个 | 终端 UI 状态管理 |
| 服务模块 | 35+ 个 | API、MCP、权限等 |
| 技术栈 | TypeScript + React + Ink + Zod + Bun | Ink 是终端 React 框架 |

其核心架构可以用一句话概括：**一个带权限系统的流式工具执行循环**。

这句话包含了三个关键点：
1. **流式**：API 边返回边处理，不等完整响应
2. **工具执行循环**：调用 API → 执行工具 → 再调用 API，循环往复
3. **权限系统**：每次工具执行前都要检查是否允许

这个设计涵盖了现代 AI Agent 的所有核心挑战：如何高效地与 LLM 交互、如何安全地执行工具、如何管理有限的上下文窗口、如何支持多 Agent 协作、如何实现可扩展性。

### 核心目录结构

```
claude-code-cli/
├── entrypoints/        # 启动入口
├── query/              # 核心 Agent 循环引擎
├── tools/              # 45+ 个工具实现
├── commands/           # 100+ 个命令
├── services/           # 业务逻辑（API、MCP、分析等）
├── utils/swarm/        # 多 Agent 协作系统
├── skills/             # 技能系统
├── bridge/             # 远程会话桥接
└── bootstrap/          # 启动初始化
```

### 整体架构图

```mermaid
graph TB
    subgraph UI层
        REPL[REPL 交互界面]
        Dialog[对话框组件]
        Progress[进度条]
    end
    
    subgraph 核心引擎层
        QueryEngine[QueryEngine<br/>会话管理]
        AgentLoop[Agent Loop<br/>循环控制]
        ToolExecutor[Tool Executor<br/>工具执行]
    end
    
    subgraph 工具层
        BuiltIn[内置工具<br/>Bash/Read/Edit...]
        MCP[MCP 工具<br/>外部服务]
        Skill[Skill 工具<br/>用户扩展]
    end
    
    subgraph 服务层
        APIClient[API Client]
        Permission[Permission Service]
        Context[Context Service]
        MCPService[MCP Service]
    end
    
    subgraph 状态层
        GlobalState[Global State]
        AppState[App State]
        Session[Session 持久化]
    end
    
    REPL --> QueryEngine
    Dialog --> QueryEngine
    QueryEngine --> AgentLoop
    AgentLoop --> ToolExecutor
    ToolExecutor --> BuiltIn
    ToolExecutor --> MCP
    ToolExecutor --> Skill
    AgentLoop --> APIClient
    ToolExecutor --> Permission
    QueryEngine --> Context
    MCPService --> MCP
    QueryEngine --> GlobalState
    QueryEngine --> AppState
    QueryEngine --> Session
```

### 技术栈选型

| 组件 | 选型 | 原因 |
|------|------|------|
| 运行时 | Node.js / Bun | Bun 更快，Node.js 兼容性更好 |
| 语言 | TypeScript | 强类型保证，减少运行时错误 |
| 终端 UI | React + Ink | 组件化思维，复用 React 生态 |
| API | Anthropic Messages | 支持流式返回，实现「流式优先」 |

## 文档目录

- [核心机制](./core.md) - 整体架构、启动流程、Agent 循环、上下文管理
- [安全与扩展](./security.md) - 工具系统、权限系统、Skill 与 Hook
- [多 Agent 协作](./swarm.md) - Swarm 系统、上下文隔离、Team 管理

## 核心设计原则

从 Claude Code 的架构中可以提炼出几条通用的 AI Agent 设计原则，这些原则不仅适用于 Claude Code，也对任何 AI Agent 开发都有借鉴意义。

### 一、流式优先

传统模式下，需要等待 API 完整响应后才能开始处理。流式模式下，API 开始返回后就可以边接收边处理，用户体感延迟大幅下降，而且能更早发现问题。

```mermaid
sequenceDiagram
    participant User as 用户
    participant Agent as Agent
    participant API as Claude API
    participant Tool as 工具
    
    Note over User,Tool: 传统模式（串行）
    User->>Agent: 发送请求
    Agent->>API: 调用 API
    API-->>Agent: 等待完整响应...
    API-->>Agent: 完整响应返回
    Agent->>Tool: 执行工具
    Tool-->>Agent: 返回结果
    Agent-->>User: 显示结果
    
    Note over User,Tool: 流式模式（并行）
    User->>Agent: 发送请求
    Agent->>API: 调用 API
    API-->>Agent: 开始流式返回
    loop 边接收边处理
        API-->>Agent: 返回片段
        Agent->>Tool: 发现工具调用，立即执行
        Tool-->>Agent: 返回结果
        Agent-->>User: 实时显示
    end
```

这个原则的核心思想是：**不要让工具执行成为瓶颈**。当 API 返回包含工具调用时，立刻执行，不等整条消息返回完。这意味着工具执行和 API 响应可以并行进行，充分利用了流式 API 的特性。

### 二、权限是管道，不是开关

单一的 allow/deny 太粗糙。实际产品需要多层级、可配置、支持自动分类的权限决策链。

```mermaid
flowchart LR
    A[工具调用请求] --> B{规则匹配}
    B -->|命中 allow| C[允许执行]
    B -->|命中 deny| D[拒绝执行]
    B -->|未命中| E{历史决策}
    
    E -->|有缓存| C
    E -->|有缓存| D
    E -->|无缓存| F{Tran 分类器}
    
    F -->|安全操作| C
    F -->|危险操作| G{Claude API}
    F -->|不确定| G
    
    G -->|allow| C
    G -->|deny| D
    
    style B fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style G fill:#fce4ec
```

设计思路是「由快到慢，由简单到复杂」递进：
- 第一层是最快的规则匹配，亚毫秒级
- 第二层是历史决策缓存，避免重复询问
- 第三层是本地分类器，能捕捉复杂场景
- 第四层是独立的 LLM 调用，处理边界情况

### 三、工具要声明并发安全性

Agent 经常会一口气调好几个工具，系统需要决定哪些可以并行，哪些必须串行。这个信息不应该由框架去猜，而应该由工具自己声明。

> **为什么需要并发控制？**
> 
> 假设 Agent 同时发起三个请求：读取文件 A、读取文件 B、编辑文件 C。
> - 两个读取可以并行，互不影响
> - 但编辑必须等读取完成，否则可能读到旧内容
> - 如果有两个编辑操作，必须串行，否则会产生竞态条件

```typescript
// 工具定义示例
const toolDefinition = {
  name: "Read",
  
  // 并发安全：只读操作可以并行
  isConcurrencySafe: () => true,
  
  // 输入验证
  inputSchema: z.object({
    file_path: z.string(),
    offset: z.number().optional(),
    limit: z.number().optional()
  }),
  
  // 执行逻辑
  call: async (input, context) => {
    return await fs.readFile(input.file_path, 'utf-8');
  }
};

const editToolDefinition = {
  name: "Edit",
  
  // 需要串行：修改操作必须排队
  isConcurrencySafe: () => false,
  
  inputSchema: z.object({
    file_path: z.string(),
    old_string: z.string(),
    new_string: z.string()
  }),
  
  call: async (input, context) => {
    // 编辑文件逻辑
  }
};
```

**执行队列策略**：

```mermaid
flowchart TB
    subgraph 并发安全工具
        R1[Read file1]
        R2[Read file2] 
        G1[Glob *.ts]
        S1[Search pattern]
    end
    
    subgraph 串行工具
        W1[Write file]
        E1[Edit file]
        B1[Bash command]
    end
    
    R1 --> R1r[结果1]
    R2 --> R2r[结果2]
    G1 --> G1r[结果3]
    S1 --> S1r[结果4]
    
    R1r & R2r & G1r & S1r --> W1
    W1 --> W1r[完成]
    W1r --> E1
    E1 --> E1r[完成]
    E1r --> B1
    
    style R1 fill:#c8e6c9
    style R2 fill:#c8e6c9
    style G1 fill:#c8e6c9
    style S1 fill:#c8e6c9
    style W1 fill:#ffcdd2
    style E1 fill:#ffcdd2
    style B1 fill:#ffcdd2
```

### 四、上下文管理是核心挑战

Agent 的 context window 就像一个固定大小的背包，你需要决定装什么、不装什么、什么时候清理。

> **为什么上下文管理这么重要？**
> 
> - Claude 的上下文窗口虽然很大（200K tokens），但不是无限的
> - 每次对话都会消耗 token，长对话很快就会填满
> - 工具结果（如读取大文件）可能占用大量空间
> - 超过限制会导致 API 报错，对话中断
> - 好的管理策略能让 Agent 记住重要信息，忘记无关内容

```mermaid
flowchart TB
    subgraph 上下文来源
        CLAUDE[CLAUDE.md<br/>项目配置]
        Memory[Memory 文件<br/>用户画像/反馈]
        Git[Git 状态<br/>当前变更]
        MCP[MCP 指令<br/>服务器配置]
    end
    
    subgraph 管理策略
        Budget[Token 预算分配]
        Snip[历史裁剪]
        Compress[消息压缩]
        Lazy[延迟加载]
    end
    
    subgraph 输出
        Prompt[最终 Prompt]
    end
    
    CLAUDE --> Budget
    Memory --> Lazy
    Git --> Budget
    MCP --> Lazy
    
    Budget --> Snip
    Lazy --> Snip
    Snip --> Compress
    Compress --> Prompt
    
    style Budget fill:#e3f2fd
    style Snip fill:#fff8e1
    style Compress fill:#fce4ec
    style Lazy fill:#e8f5e9
```

**Token 预算分配示例**：

```
总 Context Window: 200K tokens
├── System Prompt: ~10K tokens
│   ├── CLAUDE.md: ~3K
│   ├── Memory 索引: ~1K
│   ├── Git 状态: ~1K
│   └── 工具列表(名字): ~5K
├── 对话历史: ~150K tokens
│   ├── 用户消息: ~20K
│   ├── 助手消息: ~30K
│   └── 工具结果: ~100K
├── 预留空间: ~40K tokens
│   └── 下轮响应空间
```

### 五、渐进式披露省 Token

工具 schema、skill 内容、MCP 指令，都不该一开始就全塞进 prompt。按需加载，用多少拿多少。

> **什么是渐进式披露？**
> 
> 假设有 100 个工具，每个工具的 schema 平均 500 tokens：
> - 全部发送：50,000 tokens，占用大量上下文
> - 只发名字：100 tokens，节省 99.8%
> - 需要时再加载完整定义：按需付费
> 
> 这就是渐进式披露的核心思想——先给目录，再给详情。

```mermaid
sequenceDiagram
    participant Model as 模型
    participant Agent as Agent
    participant TS as ToolSearch
    participant Tool as 工具定义
    
    Note over Model,Tool: 初始状态：只发工具名
    Agent->>Model: 可用工具: Read, Edit, Bash, Glob...
    
    Model->>Agent: 我要用 Read 工具
    Agent->>TS: 搜索 Read 工具定义
    TS->>Tool: 获取完整 schema
    Tool-->>TS: 返回 inputSchema
    TS-->>Agent: 返回完整定义
    Agent->>Model: Read 工具定义: {file_path: string, ...}
    
    Model->>Agent: 调用 Read({file_path: "/src/index.ts"})
```

### 六、多 Agent 协作不一定要多进程

传统做法是为每个 Agent 启动独立的进程，但这带来大量开销。Claude Code 使用 AsyncLocalStorage 实现进程内上下文隔离。

> **为什么多进程开销大？**
> 
> - 每个进程需要独立的内存空间（Node.js 基础开销 ~30MB）
> - 进程间通信需要序列化/反序列化（JSON 转换）
> - 启动新进程需要时间（~100-500ms）
> - 进程管理复杂（崩溃重启、资源清理）
> 
> AsyncLocalStorage 让多个 Agent 共享进程，用"逻辑隔离"替代"物理隔离"。

```mermaid
graph TB
    subgraph 传统方案
        P1[进程1: Agent A<br/>内存: 100MB]
        P2[进程2: Agent B<br/>内存: 100MB]
        P3[进程3: Agent C<br/>内存: 100MB]
        IPC[进程间通信<br/>序列化开销]
        P1 <-.-> IPC
        P2 <-.-> IPC
        P3 <-.-> IPC
    end
    
    subgraph Claude Code 方案
        Process[单进程<br/>共享内存: 150MB]
        CTX1[上下文A<br/>session, 权限, 历史]
        CTX2[上下文B<br/>session, 权限, 历史]
        CTX3[上下文C<br/>session, 权限, 历史]
        ALS[AsyncLocalStorage]
        
        Process --> ALS
        ALS --> CTX1
        ALS --> CTX2
        ALS --> CTX3
    end
    
    style P1 fill:#ffcdd2
    style P2 fill:#ffcdd2
    style P3 fill:#ffcdd2
    style Process fill:#c8e6c9
    style ALS fill:#bbdefb
```

**AsyncLocalStorage 工作原理**：

```typescript
import { AsyncLocalStorage } from 'async_hooks';

// 创建上下文存储
const agentContext = new AsyncLocalStorage<AgentContext>();

// 运行 Agent A
agentContext.run({ sessionId: 'A', permissions: ['read'] }, () => {
  // 这里所有异步调用都能访问上下文 A
  console.log(agentContext.getStore()?.sessionId); // 'A'
  
  // 即使是异步操作
  setTimeout(() => {
    console.log(agentContext.getStore()?.sessionId); // 仍然是 'A'
  }, 100);
});

// 运行 Agent B（完全隔离）
agentContext.run({ sessionId: 'B', permissions: ['read', 'write'] }, () => {
  console.log(agentContext.getStore()?.sessionId); // 'B'
});
```

### 七、状态管理：极简主义

整个应用的状态管理只有 **34 行代码**，不依赖任何第三方库：

```typescript
// state/store.ts — 完整源码
type Listener = () => void
type OnChange<T> = (args: { newState: T; oldState: T }) => void

export type Store<T> = {
  getState: () => T
  setState: (updater: (prev: T) => T) => void
  subscribe: (listener: Listener) => () => void
}

export function createStore<T>(
  initialState: T,
  onChange?: OnChange<T>,
): Store<T> {
  let state = initialState
  const listeners = new Set<Listener>()
  
  return {
    getState: () => state,
    setState: (updater: (prev: T) => T) => {
      const prev = state
      const next = updater(prev)
      if (Object.is(next, prev)) return  // 核心：引用相等检查
      state = next
      onChange?.({ newState: next, oldState: prev })
      for (const listener of listeners) listener()
    },
    subscribe: (listener: Listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}
```

配合 React 18 的 `useSyncExternalStore` 实现最小化重渲染：

```typescript
export function useAppState<T>(selector: (state: AppState) => T): T {
  const store = useAppStore();
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getState()),
  );
}
```

**为什么不用 Zustand 或 Redux？** 因为 Claude Code 的状态更新模式非常简单——没有中间件、没有 devtools、没有异步 action。一个 `Object.is()` + `Set<Listener>` 就够了。34 行代码 vs Zustand 的 1000+ 行，性能更好，依赖更少。

### 八、Feature Gate：编译时优化

Claude Code 使用两层特性开关，实现编译时消除和运行时控制。

```mermaid
flowchart TB
    subgraph 编译时
        Bun[Bun Bundle]
        DCE[Dead Code Elimination]
        External[外部构建]
        Internal[内部构建]
    end
    
    subgraph 运行时
        Config[配置文件]
        Env[环境变量]
        Statsig[Statsig 特性开关]
    end
    
    Bun --> DCE
    DCE --> External
    DCE --> Internal
    
    Config --> Check{feature 检查}
    Env --> Check
    Statsig --> Check
    
    Check -->|true| Enable[启用功能]
    Check -->|false| Disable[禁用功能]
    
    style DCE fill:#c8e6c9
    style Check fill:#fff3e0
```

**编译时消除示例**：

```typescript
import { feature } from 'bun:bundle';

// 编译时完全消除 — 外部构建中这段代码不存在
if (feature('COORDINATOR_MODE')) {
  const module = require('./coordinatorMode.js')
}
```

**已发现的 Feature Gate 列表**：

| Flag | 说明 |
|------|------|
| `COORDINATOR_MODE` | 多智能体编排 |
| `VOICE_MODE` | 语音输入 |
| `HISTORY_SNIP` | 历史截断压缩 |
| `TRANSCRIPT_CLASSIFIER` | ML 权限分类器 |
| `REACTIVE_COMPACT` | 主动上下文压缩 |
| `CONTEXT_COLLAPSE` | 语义上下文折叠 |
| `KAIROS` | Assistant 模式 |
| `BRIDGE_MODE` | 远程桥接 |
| `PROACTIVE` | 主动提示 |
| `AGENT_TRIGGERS` | Cron 触发器 |
| `UDS_INBOX` | Unix Socket 收件箱 |
| `DAEMON` | 守护进程模式 |
| `WEB_BROWSER_TOOL` | 浏览器工具 |
| `TERMINAL_PANEL` | 终端面板 |
| `EXPERIMENTAL_SKILL_SEARCH` | 技能搜索 |

## 彩蛋：Buddy 宠物系统

Claude Code 内置了一个可爱的宠物系统（Buddy），为严肃的编程工作增添一丝趣味。

### 物种列表

```typescript
// buddy/types.ts
export const SPECIES = [
  'duck', 'goose', 'blob', 'cat', 'dragon',
  'octopus', 'owl', 'penguin', 'turtle', 'snail',
  'ghost', 'axolotl', 'capybara', 'cactus', 'robot',
  'rabbit', 'mushroom', 'chonk',
] as const  // 18 种物种

export const RARITIES = ['common', 'uncommon', 'rare', 'epic', 'legendary'] as const
```

### 宠物属性

```typescript
export type CompanionBones = {
  rarity: Rarity
  species: Species
  eye: Eye
  hat: Hat
  shiny: boolean          // 闪光版
  stats: Record<StatName, number>  // Debugging, Patience, Chaos, Wisdom, Snark
}

export type CompanionSoul = {
  name: string           // AI 生成的名字
  personality: string    // AI 生成的性格
}
```

### 生成机制

用 `hash(userId)` 确定性生成，保证同一用户在所有会话中看到同一只宠物。首次 "孵化" 时由 Claude 生成名字和性格。

```mermaid
flowchart TB
    User[用户 ID] --> Hash[hash 函数]
    Hash --> Seed[随机种子]
    
    Seed --> Species[物种选择]
    Seed --> Rarity[稀有度]
    Seed --> Stats[属性值]
    Seed --> Shiny{闪光?}
    
    Species --> Bones[CompanionBones]
    Rarity --> Bones
    Stats --> Bones
    Shiny --> Bones
    
    Bones --> First{首次孵化?}
    
    First -->|是| Claude[Claude 生成]
    Claude --> Name[名字]
    Claude --> Personality[性格]
    
    First -->|否| Load[加载已有]
    
    Name --> Soul[CompanionSoul]
    Personality --> Soul
    Load --> Soul
    
    Bones --> Display[显示宠物]
    Soul --> Display
    
    style Hash fill:#e3f2fd
    style Claude fill:#c8e6c9
    style Shiny fill:#fff3e0
```

**技术实现**：

- `CompanionSprite.tsx` (45KB) 实现了帧动画
- 支持多种表情和动作
- 宠物会随着用户操作做出反应