# 多 Agent 编排模式

当单个 Agent 无法处理复杂任务时，我们需要多个 Agent 协作。本节介绍常见的多 Agent 编排模式。

## 为什么需要多 Agent？

随着任务复杂度增加，单个 Agent 面临挑战：

| 挑战 | 描述 |
|------|------|
| 工具过多 | 太多工具导致决策困难 |
| 提示词过长 | 难以提供清晰的指导 |
| 安全边界 | 不同任务需要不同权限 |
| 专业分工 | 需要不同领域的专业知识 |

**多 Agent 的优势：**

- **专业化**：每个 Agent 专注于特定领域
- **可扩展**：可以独立添加或修改 Agent
- **可维护**：测试和调试更容易
- **可优化**：每个 Agent 可以使用不同的模型和策略

## 复杂度层级

在选择多 Agent 架构前，评估是否真的需要：

```
┌─────────────────────────────────────────────────────────────┐
│ 复杂度层级                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Level 1: 直接模型调用                                        │
│ 单个 LLM 调用，精心设计的提示词                               │
│ 适用：分类、摘要、翻译等单步任务                               │
│                                                             │
│ Level 2: 单 Agent + 工具                                     │
│ 一个 Agent 推理、选择工具、执行多步操作                        │
│ 适用：单领域内的动态工具使用                                   │
│                                                             │
│ Level 3: 多 Agent 编排                                       │
│ 多个专业 Agent 协作                                          │
│ 适用：跨领域、跨安全边界、需要并行处理的任务                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**原则：使用满足需求的最低复杂度方案。**

## 五种核心编排模式

### 1. 顺序编排（Sequential Orchestration）

将 Agent 串联成线性流水线，每个 Agent 处理前一个的输出。

```
输入 → Agent A → Agent B → Agent C → 输出
```

**特点：**
- 预定义的执行顺序
- 每个阶段专注于特定任务
- 类似管道（Pipeline）模式

**适用场景：**
- 多阶段处理，有清晰的依赖关系
- 数据转换流水线
- 渐进式优化（起草 → 审核 → 润色）

**示例：合同生成流水线**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 模板选择     │ → │ 条款定制     │ → │ 合规检查     │ → │ 风险评估     │
│ Agent       │    │ Agent       │    │ Agent       │    │ Agent       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ↓                  ↓                  ↓                  ↓
  选择合同模板      修改标准条款        检查法律法规        评估风险等级
```

**实现代码：**

```python
class SequentialOrchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def run(self, input_data: str) -> str:
        result = input_data
        for agent in self.agents:
            result = agent.run(result)
        return result

# 使用示例
agents = [
    TemplateSelectionAgent(),
    ClauseCustomizationAgent(),
    ComplianceCheckAgent(),
    RiskAssessmentAgent()
]

orchestrator = SequentialOrchestrator(agents)
contract = orchestrator.run("生成一份销售合同...")
```

**避免使用场景：**
- 阶段可以并行执行
- 只有少数阶段，单 Agent 可以处理
- 需要动态路由或回溯

### 2. 并发编排（Concurrent Orchestration）

多个 Agent 同时处理同一任务，从不同角度提供分析。

```
         ┌─────────────┐
         │   Agent A   │
         └──────┬──────┘
                │
┌─────────────┐ │ ┌─────────────┐
│   Agent B   │─┼─│   Agent C   │
└─────────────┘ │ └─────────────┘
                │
         ┌──────▼──────┐
         │   汇总结果   │
         └─────────────┘
```

**特点：**
- 独立并行执行
- 多视角分析
- 可聚合或独立输出

**适用场景：**
- 需要多角度分析
- 时间敏感场景（并行减少延迟）
- 投票或共识决策

**示例：股票分析系统**

```
                    ┌─────────────────┐
                    │   股票代码输入   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ 基本面分析     │  │ 技术面分析     │  │ 情绪分析      │
│ Agent         │  │ Agent         │  │ Agent        │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ 综合投资建议 │
                    └─────────────┘
```

**实现代码：**

```python
import asyncio
from typing import List, Dict

class ConcurrentOrchestrator:
    def __init__(self, agents: List[Agent], aggregation_strategy: str = "merge"):
        self.agents = agents
        self.aggregation_strategy = aggregation_strategy
    
    async def run_async(self, input_data: str) -> Dict:
        # 并行执行所有 Agent
        tasks = [agent.run_async(input_data) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        # 聚合结果
        if self.aggregation_strategy == "merge":
            return self._merge_results(results)
        elif self.aggregation_strategy == "vote":
            return self._vote_results(results)
        else:
            return {"individual_results": results}
    
    def _merge_results(self, results: List) -> Dict:
        """合并所有结果"""
        merged = {}
        for i, result in enumerate(results):
            merged[f"agent_{i}"] = result
        return merged
    
    def _vote_results(self, results: List) -> Dict:
        """投票选择结果"""
        # 实现投票逻辑
        pass

# 使用示例
agents = [
    FundamentalAnalysisAgent(),
    TechnicalAnalysisAgent(),
    SentimentAnalysisAgent()
]

orchestrator = ConcurrentOrchestrator(agents, aggregation_strategy="merge")
result = await orchestrator.run_async("AAPL")
```

**避免使用场景：**
- Agent 之间需要依赖或顺序
- 资源受限，无法并行
- 结果冲突难以解决

### 3. 群聊编排（Group Chat Orchestration）

多个 Agent 在共享对话线程中协作，通过讨论解决问题。

```
┌─────────────────────────────────────────────────────────────┐
│                     共享对话线程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Agent A: 我认为我们应该先分析用户需求...                     │
│  Agent B: 同意，我注意到用户提到了性能问题...                 │
│  Agent C: 从安全角度来看，我们需要考虑...                     │
│  Agent A: 综合大家的意见，我建议...                          │
│  ...                                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │   Chat Manager    │
                    │   (协调发言顺序)   │
                    └───────────────────┘
```

**特点：**
- 共享对话上下文
- 通过讨论达成共识
- 支持人类参与

**适用场景：**
- 需要头脑风暴和讨论
- 迭代优化（Maker-Checker 循环）
- 多学科协作

**Maker-Checker 循环**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────┐         ┌──────────┐                        │
│   │  Maker   │ ──────▶ │ Checker  │                        │
│   │ (创建者)  │         │ (检查者)  │                        │
│   └────┬─────┘         └────┬─────┘                        │
│        │                    │                               │
│        │              ┌─────┴─────┐                         │
│        │              │           │                         │
│        │         通过 ▼     不通过 │                         │
│        │         ┌───────┐       │                          │
│        │         │ 最终   │       │                          │
│        │         │ 结果   │       │                          │
│        │         └───────┘       │                          │
│        │                         │                          │
│        └─────────────────────────┘                          │
│                   (反馈修改建议)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**实现代码：**

```python
class GroupChatOrchestrator:
    def __init__(self, agents: List[Agent], max_rounds: int = 10):
        self.agents = agents
        self.max_rounds = max_rounds
        self.conversation = []
    
    def run(self, task: str) -> str:
        self.conversation = [{"role": "user", "content": task}]
        
        for round_num in range(self.max_rounds):
            # 选择下一个发言的 Agent
            speaker = self._select_speaker(round_num)
            
            # Agent 发言
            response = speaker.run(self._format_context())
            self.conversation.append({
                "role": "agent",
                "agent_name": speaker.name,
                "content": response
            })
            
            # 检查是否达成共识或完成任务
            if self._is_task_complete():
                return self._extract_final_answer()
        
        return "讨论未能在限定轮次内达成结论"
    
    def _select_speaker(self, round_num: int) -> Agent:
        """选择下一个发言者"""
        # 可以实现轮询、投票、或基于内容的动态选择
        return self.agents[round_num % len(self.agents)]
    
    def _is_task_complete(self) -> bool:
        """检查任务是否完成"""
        # 检查最后几条消息是否有完成标志
        last_messages = self.conversation[-3:]
        for msg in last_messages:
            if "任务完成" in msg.get("content", ""):
                return True
        return False

# Maker-Checker 示例
class MakerCheckerOrchestrator:
    def __init__(self, maker: Agent, checker: Agent, max_iterations: int = 5):
        self.maker = maker
        self.checker = checker
        self.max_iterations = max_iterations
    
    def run(self, task: str) -> str:
        work = self.maker.run(task)
        
        for _ in range(self.max_iterations):
            review = self.checker.run(f"请审查以下内容:\n\n{work}")
            
            if "通过" in review or "approved" in review.lower():
                return work
            
            # 反馈给 Maker 进行修改
            work = self.maker.run(f"根据以下反馈修改内容:\n{review}\n\n原内容:\n{work}")
        
        return work
```

**避免使用场景：**
- 简单任务，讨论是过度设计
- 实时性要求高
- 难以确定任务完成条件

### 4. 交接编排（Handoff Orchestration）

Agent 根据任务需求动态转移控制权给更合适的 Agent。

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Triage     │ ──▶ │  Technical  │ ──▶ │  Billing    │
│  Agent      │     │  Agent      │     │  Agent      │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │                   │                   │
      ▼                   ▼                   ▼
  分类路由            技术支持             账单问题
```

**特点：**
- 动态路由
- 单一 Agent 活跃
- 专家系统模式

**适用场景：**
- 最佳处理者未知
- 需求在处理过程中明确
- 多领域问题

**示例：客服系统**

```
用户: "我的订单 #12345 还没到货，而且我多扣了钱"

         ┌─────────────────────┐
         │   Triage Agent      │
         │   (分流 Agent)       │
         └──────────┬──────────┘
                    │
                    ▼ 分析问题类型
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌────────┐    ┌────────┐
│ 物流    │    │ 支付    │    │ 退款    │
│ Agent  │    │ Agent  │    │ Agent  │
└────────┘    └────────┘    └────────┘
    │               │               │
    └───────────────┼───────────────┘
                    │
                    ▼
            问题解决或升级人工
```

**实现代码：**

```python
from typing import Optional, Tuple
from enum import Enum

class AgentType(Enum):
    TRIAGE = "triage"
    TECHNICAL = "technical"
    BILLING = "billing"
    HUMAN = "human"

class HandoffOrchestrator:
    def __init__(self, agents: Dict[AgentType, Agent]):
        self.agents = agents
        self.current_agent = agents[AgentType.TRIAGE]
    
    def run(self, user_input: str) -> Tuple[str, Optional[AgentType]]:
        """运行并可能交接给其他 Agent"""
        response = self.current_agent.run(user_input)
        
        # 检查是否需要交接
        handoff_target = self._extract_handoff(response)
        
        if handoff_target:
            # 交接给目标 Agent
            self.current_agent = self.agents[handoff_target]
            # 转交上下文
            context = self._prepare_handoff_context(user_input, response)
            return self.current_agent.run(context), handoff_target
        
        return response, None
    
    def _extract_handoff(self, response: str) -> Optional[AgentType]:
        """从响应中提取交接目标"""
        if "转接技术支持" in response:
            return AgentType.TECHNICAL
        elif "转接账单部门" in response:
            return AgentType.BILLING
        elif "转接人工客服" in response:
            return AgentType.HUMAN
        return None
    
    def _prepare_handoff_context(self, original_input: str, previous_response: str) -> str:
        """准备交接上下文"""
        return f"用户原始问题: {original_input}\n\n前一位 Agent 的分析: {previous_response}"
```

**避免使用场景：**
- 可以预先确定处理者
- 任务路由是确定性的
- 可能导致无限交接循环

### 5. 磁性编排（Magentic Orchestration）

用于开放式复杂问题，由 Manager Agent 动态构建和调整任务计划。

```
┌─────────────────────────────────────────────────────────────┐
│                      Manager Agent                          │
│                   (动态规划与协调)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  任务账本 (Task Ledger):                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Goal: 恢复服务可用性                                   │   │
│  │ Subgoals:                                            │   │
│  │   [✓] 诊断问题                                        │   │
│  │   [→] 制定修复方案  ← 当前                             │   │
│  │   [ ] 执行修复                                        │   │
│  │   [ ] 验证结果                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ 诊断 Agent    │  │ 基础设施       │  │ 通信 Agent    │
│              │  │ Agent        │  │              │
└───────────────┘  └───────────────┘  └───────────────┘
```

**特点：**
- 动态规划
- 可回溯和调整
- 完整审计追踪

**适用场景：**
- 开放式问题，无预定路径
- 需要生成可审计的计划
- Agent 有工具可影响外部系统

**示例：SRE 事故响应**

```python
class MagenticOrchestrator:
    def __init__(self, manager: Agent, workers: Dict[str, Agent]):
        self.manager = manager
        self.workers = workers
        self.task_ledger = []
    
    def run(self, incident: str) -> str:
        # 初始化任务账本
        self.task_ledger = [
            {"goal": "恢复服务", "status": "pending", "subtasks": []}
        ]
        
        max_iterations = 20
        for _ in range(max_iterations):
            # Manager 分析当前状态并更新计划
            plan_update = self.manager.run(self._get_context())
            self._update_ledger(plan_update)
            
            # 执行下一个任务
            next_task = self._get_next_task()
            if not next_task:
                break
            
            # 分配给合适的 Worker
            worker = self._select_worker(next_task)
            result = worker.run(next_task["description"])
            
            # 更新任务状态
            next_task["status"] = "completed"
            next_task["result"] = result
        
        return self._generate_final_report()
    
    def _get_context(self) -> str:
        """构建当前上下文"""
        return f"""
        事故描述: {incident}
        当前任务账本: {json.dumps(self.task_ledger, indent=2)}
        请分析并决定下一步行动。
        """
    
    def _update_ledger(self, plan_update: str):
        """根据 Manager 输出更新任务账本"""
        # 解析并更新任务列表
        pass
    
    def _get_next_task(self) -> Optional[Dict]:
        """获取下一个待执行任务"""
        for task in self.task_ledger:
            if task["status"] == "pending":
                return task
        return None
    
    def _select_worker(self, task: Dict) -> Agent:
        """根据任务类型选择 Worker"""
        task_type = task.get("type", "general")
        return self.workers.get(task_type, self.workers["general"])
```

**避免使用场景：**
- 有明确的解决路径
- 时间敏感
- 可能频繁停滞

## 模式对比

| 模式 | 协调方式 | 路由方式 | 最佳场景 | 注意事项 |
|------|----------|----------|----------|----------|
| 顺序 | 线性流水线 | 预定义顺序 | 多阶段渐进处理 | 早期失败会传播 |
| 并发 | 并行独立 | 确定性或动态 | 多视角分析 | 结果冲突需解决 |
| 群聊 | 对话讨论 | Chat Manager 控制 | 共识建立、迭代优化 | 对话循环风险 |
| 交接 | 动态委托 | Agent 自主决定 | 专家动态选择 | 无限交接循环 |
| 磁性 | 计划-构建-执行 | Manager 动态分配 | 开放式复杂问题 | 收敛慢、易停滞 |

## 实现注意事项

### 1. 上下文与状态管理

```python
class ContextManager:
    def __init__(self, max_context_size: int = 4000):
        self.max_context_size = max_context_size
    
    def prepare_context(
        self, 
        agent: Agent, 
        full_history: List, 
        strategy: str = "summary"
    ) -> List:
        """为 Agent 准备上下文"""
        
        if strategy == "full":
            # 完整历史（可能超限）
            return full_history
        
        elif strategy == "summary":
            # 压缩历史为摘要
            return self._summarize_history(full_history)
        
        elif strategy == "selective":
            # 选择性提取相关内容
            return self._extract_relevant(full_history, agent.role)
        
        else:
            # 最小上下文
            return full_history[-5:]
```

### 2. 确定性路由 vs 动态路由

```python
# 确定性路由
def deterministic_route(input_type: str) -> Agent:
    routes = {
        "technical": technical_agent,
        "billing": billing_agent,
        "general": general_agent
    }
    return routes.get(input_type, general_agent)

# 动态路由（由 Agent 决定）
def dynamic_route(user_input: str) -> Tuple[Agent, str]:
    # 路由 Agent 分析并决定
    route_decision = router_agent.run(user_input)
    target = parse_route_decision(route_decision)
    return agents[target], route_decision
```

### 3. 错误处理与回退

```python
class RobustOrchestrator:
    def run_with_fallback(self, task: str) -> str:
        try:
            return self.primary_agent.run(task)
        except Exception as e:
            logging.error(f"Primary agent failed: {e}")
            return self.fallback_agent.run(task)
    
    def run_with_retry(self, task: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return self.agent.run(task)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
```

### 4. 监控与可观测性

```python
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AgentTrace:
    agent_name: str
    input: str
    output: str
    timestamp: datetime
    duration_ms: float
    tokens_used: int

class ObservableOrchestrator:
    def __init__(self):
        self.traces: List[AgentTrace] = []
    
    def trace_agent_call(self, agent: Agent, input_data: str) -> str:
        start = time.time()
        
        try:
            output = agent.run(input_data)
            success = True
        except Exception as e:
            output = str(e)
            success = False
        
        trace = AgentTrace(
            agent_name=agent.name,
            input=input_data[:500],  # 截断
            output=output[:500],
            timestamp=datetime.now(),
            duration_ms=(time.time() - start) * 1000,
            tokens_used=agent.last_token_count
        )
        self.traces.append(trace)
        
        if not success:
            raise Exception(output)
        
        return output
```

## 小结

多 Agent 编排是处理复杂任务的关键技术：

- **顺序编排**：适合有明确阶段依赖的流程
- **并发编排**：适合需要多角度分析的任务
- **群聊编排**：适合需要讨论和共识的场景
- **交接编排**：适合专家动态选择的场景
- **磁性编排**：适合开放式复杂问题

选择模式时，遵循"简单优先"原则：
1. 先评估是否真的需要多 Agent
2. 选择能满足需求的最低复杂度模式
3. 实现完整的监控和错误处理
4. 根据实际运行情况持续优化
