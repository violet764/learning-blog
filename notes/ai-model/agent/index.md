# AI Agent 学习指南

本系列文章带你从零开始学习 AI Agent 的核心概念、设计模式和实现方法。

## 系列目录

### 1. [Agent 基础概念](./agent-basics.md)

了解什么是 AI Agent，它的核心组件，以及与传统聊天机器人的区别。

**主要内容：**
- Agent 的定义与分类
- 核心组件：LLM、记忆系统、工具集、编排控制器
- Agent 工作原理：感知-思考-行动循环
- 上下文工程的关键挑战

### 2. [ReAct 模式与工具调用](./agent-react-tools.md)

深入学习 ReAct（Reasoning + Acting）模式和 Function Calling 机制。

**主要内容：**
- ReAct 核心思想与工作流程
- Function Calling 的结构化工具调用
- 思维链（Chain-of-Thought）技术
- 工具设计的最佳实践

### 3. [CLI Agent 实战教程](./agent-cli-tutorial.md)

从零构建一个功能完整的命令行 AI Agent。

**主要内容：**
- 项目架构设计
- 文件操作、Shell 命令、搜索工具实现
- 安全最佳实践
- 进阶功能扩展

### 4. [多 Agent 编排模式](./agent-orchestration.md)

学习如何设计多 Agent 协作系统。

**主要内容：**
- 五种核心编排模式
- 顺序、并发、群聊、交接、磁性编排
- 模式选择指南
- 实现注意事项

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                      学习路径建议                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  入门：Agent 基础概念                                        │
│    ↓                                                        │
│  理解：ReAct 模式与工具调用                                   │
│    ↓                                                        │
│  实践：CLI Agent 实战教程                                    │
│    ↓                                                        │
│  进阶：多 Agent 编排模式                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心要点回顾

### Agent 本质

```
Agent = LLM + 工具 + 循环
```

- **LLM**：提供推理和决策能力
- **工具**：提供与外部世界交互的能力
- **循环**：让 Agent 能够迭代执行直到完成任务

### 关键挑战

1. **上下文工程**：如何在正确的时间提供正确的信息
2. **工具设计**：如何设计有用且安全的工具
3. **安全控制**：如何限制 Agent 的行为边界
4. **评估监控**：如何衡量和改进 Agent 性能

### 设计原则

- **简单优先**：使用满足需求的最低复杂度方案
- **安全第一**：对危险操作进行限制和确认
- **可观测性**：记录所有操作，便于调试和审计
- **渐进增强**：从简单开始，根据需要添加复杂度

## 推荐资源

### 论文
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### 开源项目
- [LangChain](https://github.com/langchain-ai/langchain) - Agent 开发框架
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - 自主 Agent 实现
- [OpenClaw](./../tools/openclaw-intro.md) - 自托管 AI Agent 框架

### 文档
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [OpenAI: Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Microsoft: AI Agent Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
