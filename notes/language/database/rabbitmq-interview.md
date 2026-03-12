# RabbitMQ 面试题

本文整理了 RabbitMQ 高频面试题，涵盖基础概念、消息可靠性、高级特性、高可用架构、实战经验和场景设计六个方面，帮助您系统掌握 RabbitMQ 核心知识。

## 基础篇

### 1. RabbitMQ 是什么？适用于什么场景？

**答案：**

RabbitMQ 是一个开源的消息代理软件，实现了 AMQP（高级消息队列协议）。它以可靠性、灵活的路由机制和丰富的客户端库支持而闻名。

**核心特性：**

| 特性 | 说明 |
|------|------|
| **可靠性** | 支持消息持久化、确认机制、事务 |
| **灵活路由** | 四种 Exchange 类型，支持复杂路由规则 |
| **多协议** | AMQP、STOMP、MQTT 等 |
| **多语言客户端** | Java、Python、Go、PHP、.NET 等 |
| **管理界面** | Web UI 监控管理 |

**适用场景：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 典型应用场景                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 异步解耦                                            │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ 订单服务 │────>│ MQ 队列  │────>│ 支付服务 │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│                                                         │
│  2. 流量削峰                                            │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ 秒杀请求 │────>│ MQ 缓冲  │────>│ 后端服务 │          │
│  │ (1万QPS) │     │ (削峰)   │     │ (1千QPS) │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│                                                         │
│  3. 延迟任务                                            │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ 订单创建 │────>│ 延迟队列 │────>│ 超时取消 │          │
│  │   T=0    │     │  T=30min │     │  T=30min │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│                                                         │
│  4. 发布订阅                                            │
│  ┌─────────┐     ┌─────────┐                           │
│  │ 广播服务 │────>│ Fanout  │────> 多个消费者           │
│  └─────────┘     │ Exchange │                           │
│                  └─────────┘                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**追问：RabbitMQ 不适合什么场景？**

**追问答案：**

- **海量日志收集**：Kafka 更适合，吞吐量更高
- **消息回溯需求**：RabbitMQ 消息消费后删除，不支持回溯
- **超高吞吐场景**：Kafka 单机可达百万级，RabbitMQ 万级
- **流式处理**：Kafka Streams / Flink 更合适

---

### 2. RabbitMQ 和 Kafka 有什么区别？如何选型？

**答案：**

两者设计理念根本不同：RabbitMQ 是**智能路由、简单消费**，Kafka 是**简单路由、智能消费**。

```
┌─────────────────────────────────────────────────────────┐
│              设计理念对比                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RabbitMQ：消息代理 (Message Broker)                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Producer ──> Exchange ──> Queue ──> Consumer    │   │
│  │              (智能路由)    (存储)                │   │
│  │  • 路由逻辑在 Broker 端                          │   │
│  │  • Push 模式，主动推送                           │   │
│  │  • 消息消费后删除                                │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Kafka：分布式日志系统 (Distributed Log)               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Producer ──> Topic ──> Partition ──> Consumer   │   │
│  │              (简单Topic) (分区存储)              │   │
│  │  • 路由逻辑简单                                  │   │
│  │  • Pull 模式，主动拉取                           │   │
│  │  • 消息持久化，支持回溯                          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**详细对比表：**

| 对比维度 | RabbitMQ | Kafka |
|---------|----------|-------|
| **定位** | 通用消息代理 | 分布式日志系统 |
| **吞吐量** | 1-5万/秒 | 10-100万/秒 |
| **延迟** | 微秒级 | 毫秒级 |
| **消息路由** | 复杂（4种Exchange） | 简单（Topic+Partition） |
| **消费模式** | Push | Pull |
| **消息持久化** | 可选 | 默认持久化 |
| **消息回溯** | ❌ 不支持 | ✅ 支持 |
| **事务支持** | ✅ 原生支持 | ⚠️ 支持（性能影响大） |
| **协议** | AMQP、STOMP、MQTT | 自有协议 |

**选型决策树：**

```
开始选型
    │
    ├── 需要复杂消息路由？
    │       ├── 是 ───────────────> RabbitMQ ✅
    │       └── 否
    │               ├── 吞吐量 > 10万/秒？──> Kafka ✅
    │               └── 否
    │                       ├── 需要消息回溯？──> Kafka ✅
    │                       └── 否
    │                               ├── 需要微秒级延迟？──> RabbitMQ ✅
    │                               └── 根据团队熟悉度选择
```

**追问：能否在同一个项目中同时使用 RabbitMQ 和 Kafka？**

**追问答案：**

可以，各取所长：

```
┌─────────────────────────────────────────────────────────┐
│              混合使用场景示例                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  业务消息 ──> RabbitMQ ──> 业务服务                     │
│  (低延迟、复杂路由)                                      │
│                                                         │
│  日志/事件 ──> Kafka ──> 大数据平台                     │
│  (高吞吐、可回溯)                                        │
│                                                         │
│  注意事项：                                              │
│  • 运维成本增加                                         │
│  • 需要统一监控告警                                     │
│  • 明确边界，避免混用                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3. AMQP 协议是什么？有什么优势？

**答案：**

AMQP（Advanced Message Queuing Protocol）是一个开放标准的消息队列协议，定义了消息格式和通信规则。


**协议层次结构：**

```
┌─────────────────────────────────────────────────────────┐
│              AMQP 协议层次                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              应用层 (Application)                │   │
│  │  业务逻辑：订单处理、消息通知等                   │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              AMQP 协议层                         │   │
│  │  • Connection: 连接管理                          │   │
│  │  • Channel: 通道管理                             │   │
│  │  • Exchange: 交换机操作                          │   │
│  │  • Queue: 队列操作                               │   │
│  │  • Basic: 消息操作                               │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              传输层 (TCP/SSL)                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**AMQP 核心优势：**

| 优势 | 说明 |
|------|------|
| **标准化** | 统一协议，不同厂商实现可互通 |
| **可靠性** | 内置消息确认、持久化、事务机制 |
| **灵活性** | 支持多种消息模式（点对点、发布订阅、RPC） |
| **语言无关** | 任何语言都可以实现客户端 |

**追问：AMQP 0-9-1 和 AMQP 1.0 有什么区别？**

**追问答案：**

- **AMQP 0-9-1**：RabbitMQ 原生支持，功能更丰富
- **AMQP 1.0**：OASIS 标准，但功能简化，RabbitMQ 通过插件支持

```python
# AMQP 连接示例
import pika

# 连接参数
parameters = pika.ConnectionParameters(
    host='localhost',
    port=5672,
    virtual_host='/',
    credentials=pika.PlainCredentials('guest', 'guest')
)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()
```

---

### 4. RabbitMQ 的核心概念有哪些？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 核心概念模型                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐                                           │
│  │ Producer │                                           │
│  │ 生产者    │                                           │
│  └────┬─────┘                                           │
│       │ 1. 发布消息                                     │
│       ▼                                                 │
│  ┌──────────┐     ┌──────────┐                         │
│  │ Exchange │────>│ Binding  │                         │
│  │  交换机   │     │   绑定   │                         │
│  │ (路由中心)│     │(路由规则)│                         │
│  └──────────┘     └────┬─────┘                         │
│                        │                                │
│       ┌────────────────┴────────────────┐              │
│       ▼                                 ▼              │
│  ┌──────────┐                     ┌──────────┐        │
│  │  Queue   │                     │  Queue   │        │
│  │  队列    │                     │  队列    │        │
│  │(消息存储) │                     │(消息存储) │        │
│  └────┬─────┘                     └────┬─────┘        │
│       │                                 │              │
│       ▼                                 ▼              │
│  ┌──────────┐                     ┌──────────┐        │
│  │ Consumer │                     │ Consumer │        │
│  │  消费者   │                     │  消费者   │        │
│  └──────────┘                     └──────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**核心概念详解：**

| 概念 | 说明 |
|------|------|
| **Producer** | 消息生产者，负责发送消息到 Exchange |
| **Consumer** | 消息消费者，从 Queue 获取并处理消息 |
| **Exchange** | 交换机，接收消息并根据路由规则分发 |
| **Queue** | 队列，存储消息，等待消费者消费 |
| **Binding** | 绑定，定义 Exchange 与 Queue 的关系 |
| **Routing Key** | 路由键，消息路由的标识 |
| **Virtual Host** | 虚拟主机，逻辑隔离，相当于租户 |
| **Channel** | 通道，复用 TCP 连接，轻量级 |

**追问：为什么需要 Virtual Host？**

**追问答案：**

Virtual Host 用于逻辑隔离，类似于数据库中的"数据库"概念：

```python
# 不同 Virtual Host 完全隔离
vhost_prod = pika.ConnectionParameters(host='localhost', virtual_host='/production')
vhost_dev = pika.ConnectionParameters(host='localhost', virtual_host='/development')

# 同一个 RabbitMQ 实例可以有多个 vhost
# 不同 vhost 的 Exchange、Queue、用户权限完全隔离
```

---

### 5. Exchange 有哪几种类型？

**答案：**

RabbitMQ 提供四种 Exchange 类型：

```
┌─────────────────────────────────────────────────────────┐
│              四种 Exchange 类型对比                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Direct Exchange（直连）                             │
│     • 精确匹配 routing_key                              │
│     • 点对点消息                                        │
│     • 性能最高                                          │
│                                                         │
│  2. Fanout Exchange（扇出）                             │
│     • 忽略 routing_key                                  │
│     • 广播到所有绑定队列                                │
│     • 发布订阅模式                                      │
│                                                         │
│  3. Topic Exchange（主题）                              │
│     • 通配符匹配：* 匹配一个单词，# 匹配多个             │
│     • 灵活路由                                          │
│     • 事件驱动场景                                      │
│                                                         │
│  4. Headers Exchange（头）                              │
│     • 基于消息头属性匹配                                │
│     • 支持多条件组合                                    │
│     • 使用较少                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
import pika

channel = connection.channel()

# 1. Direct Exchange
channel.exchange_declare(exchange='direct.logs', exchange_type='direct')
channel.queue_bind(queue='error_logs', exchange='direct.logs', routing_key='error')

# 2. Fanout Exchange
channel.exchange_declare(exchange='fanout.news', exchange_type='fanout')
channel.queue_bind(queue='news.email', exchange='fanout.news')
channel.queue_bind(queue='news.sms', exchange='fanout.news')

# 3. Topic Exchange
channel.exchange_declare(exchange='topic.events', exchange_type='topic')
channel.queue_bind(queue='all_orders', exchange='topic.events', routing_key='order.#')
channel.queue_bind(queue='order_paid', exchange='topic.events', routing_key='order.paid')

# 4. Headers Exchange
channel.exchange_declare(exchange='headers.match', exchange_type='headers')
channel.queue_bind(
    queue='vip_orders',
    exchange='headers.match',
    arguments={'x-match': 'all', 'vip': 'true'}
)
```

**追问：什么场景用什么 Exchange？**

**追问答案：**

| 场景 | 推荐 Exchange | 原因 |
|------|--------------|------|
| 点对点消息、RPC | Direct | 精确路由，性能最优 |
| 广播通知、日志 | Fanout | 简单广播，一对多 |
| 事件驱动、多条件路由 | Topic | 通配符灵活匹配 |
| 复杂多属性路由 | Headers | 基于多条件组合 |

---

### 6. Channel 是什么？为什么需要 Channel？

**答案：**

Channel 是复用 TCP 连接的轻量级通道。每个 Channel 代表一个独立的会话。

```
┌─────────────────────────────────────────────────────────┐
│              Connection vs Channel                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  不使用 Channel（每个操作一个连接）：                    │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ 生产者1  │────>│ RabbitMQ │     │         │          │
│  └─────────┘     │         │     │         │          │
│  ┌─────────┐     │  连接1   │     │         │          │
│  │ 生产者2  │────>│         │     │         │          │
│  └─────────┘     │  连接2   │     │         │          │
│  ┌─────────┐     │         │     │         │          │
│  │ 消费者  │────>│  连接3   │     │         │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│                                                         │
│  问题：TCP 连接开销大（三次握手、资源占用）             │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  使用 Channel（复用一个 TCP 连接）：                    │
│  ┌─────────┐                                           │
│  │ 生产者1  │──┐                                        │
│  └─────────┘  │     ┌─────────┐                        │
│  ┌─────────┐  │     │         │                        │
│  │ 生产者2  │──┼────>│ RabbitMQ │                        │
│  └─────────┘  │     │  1个TCP │                        │
│  ┌─────────┐  │     │  3个Channel│                      │
│  │ 消费者  │──┘     └─────────┘                        │
│  └─────────┘                                           │
│                                                         │
│  优势：减少 TCP 连接数，降低资源消耗                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Channel 使用原则：**

```python
# ❌ 错误：每个操作创建新连接
def send_message(msg):
    conn = pika.BlockingConnection(...)  # 开销大
    ch = conn.channel()
    ch.basic_publish(...)
    conn.close()

# ✅ 正确：复用连接，使用多个 Channel
connection = pika.BlockingConnection(...)

# 一个连接可以有多个 Channel
channel_producer = connection.channel()
channel_consumer = connection.channel()

# Channel 之间相互独立，互不影响
```

**追问：Channel 数量有限制吗？**

**追问答案：**

有，但通常不是问题：

- 每个连接默认最多 2047 个 Channel
- 可通过 `channel_max` 配置调整
- 实际使用建议每个线程使用独立 Channel，但复用连接

---

### 7. 什么是消息持久化？如何保证？

**答案：**

消息持久化是将消息存储到磁盘，防止 RabbitMQ 重启后消息丢失。

**持久化三要素：**

```
┌─────────────────────────────────────────────────────────┐
│              消息持久化三要素                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Exchange 持久化                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  channel.exchange_declare(                      │   │
│  │      exchange='my.exchange',                    │   │
│  │      durable=True  ← 关键参数                   │   │
│  │  )                                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. Queue 持久化                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  channel.queue_declare(                         │   │
│  │      queue='my.queue',                          │   │
│  │      durable=True  ← 关键参数                   │   │
│  │  )                                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. Message 持久化                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  channel.basic_publish(                         │   │
│  │      ...,                                       │   │
│  │      properties=pika.BasicProperties(           │   │
│  │          delivery_mode=2  ← 持久化消息          │   │
│  │      )                                          │   │
│  │  )                                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ⚠️ 三者缺一不可！任何一项非持久化，消息都可能丢失      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**完整持久化代码：**

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 1. 声明持久化 Exchange
channel.exchange_declare(
    exchange='durable.exchange',
    exchange_type='direct',
    durable=True  # 持久化
)

# 2. 声明持久化 Queue
channel.queue_declare(
    queue='durable.queue',
    durable=True  # 持久化
)

# 3. 绑定
channel.queue_bind(
    queue='durable.queue',
    exchange='durable.exchange',
    routing_key='test'
)

# 4. 发送持久化消息
channel.basic_publish(
    exchange='durable.exchange',
    routing_key='test',
    body='Persistent message',
    properties=pika.BasicProperties(
        delivery_mode=2,  # 2 = 持久化
    )
)
```

**追问：持久化对性能有什么影响？**

**追问答案：**

- **写入延迟增加**：需要写入磁盘
- **吞吐量下降**：持久化后约 1-2万/秒
- **建议**：
  - 非关键消息可以非持久化
  - 使用 SSD 提升磁盘性能
  - 合理设置刷盘策略

---

### 8. 什么是 vhost？有什么作用？

**答案：**

Virtual Host（vhost）是 RabbitMQ 的逻辑隔离单元，类似于数据库中的"数据库"概念。

```
┌─────────────────────────────────────────────────────────┐
│              Virtual Host 隔离示意                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              RabbitMQ Server                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐      │   │
│  │  │   vhost: /      │  │  vhost: /prod   │      │   │
│  │  │  (默认)         │  │  (生产环境)      │      │   │
│  │  │                 │  │                 │      │   │
│  │  │  ┌───┐ ┌───┐   │  │  ┌───┐ ┌───┐   │      │   │
│  │  │  │ Q1│ │ Q2│   │  │  │ Q1│ │ Q2│   │      │   │
│  │  │  └───┘ └───┘   │  │  └───┘ └───┘   │      │   │
│  │  │                 │  │                 │      │   │
│  │  │  user: guest    │  │  user: prod_user│      │   │
│  │  └─────────────────┘  └─────────────────┘      │   │
│  │                                                  │   │
│  │  ┌─────────────────┐                            │   │
│  │  │  vhost: /dev    │                            │   │
│  │  │  (开发环境)      │                            │   │
│  │  │  user: dev_user │                            │   │
│  │  └─────────────────┘                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  不同 vhost 完全隔离：Exchange、Queue、用户权限         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**vhost 使用场景：**

| 场景 | 说明 |
|------|------|
| **多租户隔离** | 不同租户使用不同 vhost |
| **环境隔离** | dev/test/prod 环境隔离 |
| **权限控制** | 不同用户访问不同 vhost |

**操作命令：**

```bash
# 创建 vhost
rabbitmqctl add_vhost /production

# 删除 vhost
rabbitmqctl delete_vhost /production

# 列出所有 vhost
rabbitmqctl list_vhosts

# 设置用户权限
rabbitmqctl set_permissions -p /production user ".*" ".*" ".*"
```

**追问：vhost 和集群是什么关系？**

**追问答案：**

- vhost 是**逻辑隔离**，在单个节点或集群内
- 集群是**物理分布**，多个节点组成
- 一个集群可以有多个 vhost
- vhost 在集群所有节点间同步

---

## 消息可靠性篇

### 1. 如何保证消息不丢失？

**答案：**

消息丢失可能发生在三个阶段，需要针对性解决：

```
┌─────────────────────────────────────────────────────────┐
│              消息丢失的三个阶段                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  阶段1：生产者 → Broker                                 │
│  ┌─────────┐     ┌─────────┐                           │
│  │ 生产者   │──?──>│ RabbitMQ │  网络故障、Broker崩溃   │
│  └─────────┘     └─────────┘                           │
│                                                         │
│  阶段2：Broker 存储                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 RabbitMQ                        │   │
│  │  ┌─────────┐                                    │   │
│  │  │  Queue   │──?──  Broker重启、机器宕机        │   │
│  │  │ [消息]   │                                    │   │
│  │  └─────────┘                                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  阶段3：Broker → 消费者                                 │
│  ┌─────────┐     ┌─────────┐                           │
│  │ RabbitMQ │──?──>│ 消费者   │  消费者处理失败、崩溃   │
│  └─────────┘     └─────────┘                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**完整解决方案：**

```python
import pika

# ═════════════════════════════════════════════════════════
# 阶段1：生产者确认机制
# ═════════════════════════════════════════════════════════

def setup_publisher_confirms():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # 开启发布确认模式
    channel.confirm_delivery()
    
    try:
        # 发送消息（同步等待确认）
        channel.basic_publish(
            exchange='',
            routing_key='test.queue',
            body='Important message',
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print("消息确认成功")
    except pika.exceptions.UnroutableError:
        print("消息路由失败")
    except pika.exceptions.ChannelClosed:
        print("通道被关闭，消息可能未投递")
    
    return channel

# ═════════════════════════════════════════════════════════
# 阶段2：持久化
# ═════════════════════════════════════════════════════════

def setup_durable_queue():
    channel = connection.channel()
    
    # 持久化 Exchange
    channel.exchange_declare(
        exchange='durable.exchange',
        exchange_type='direct',
        durable=True
    )
    
    # 持久化 Queue
    channel.queue_declare(
        queue='durable.queue',
        durable=True
    )

# ═════════════════════════════════════════════════════════
# 阶段3：消费者手动 ACK
# ═════════════════════════════════════════════════════════

def consumer_with_manual_ack():
    def callback(ch, method, properties, body):
        try:
            # 业务处理
            process_message(body)
            # 成功后确认
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            # 失败后拒绝（可设置是否重新入队）
            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=False  # 不重新入队，进入死信队列
            )
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='durable.queue',
        on_message_callback=callback,
        auto_ack=False  # 关键：手动确认
    )
```

**追问：消息什么时候算"成功投递"？**

**追问答案：**

消息成功投递的定义取决于业务需求：

| 级别 | 定义 | 可靠性 | 性能 |
|------|------|--------|------|
| **最低** | 生产者 send 完成 | ❌ | ⭐⭐⭐⭐⭐ |
| **低** | Broker 收到（Publisher Confirm） | ⚠️ | ⭐⭐⭐⭐ |
| **中** | 消息持久化到磁盘 | ✅ | ⭐⭐⭐ |
| **高** | 消费者 ACK | ✅✅ | ⭐⭐ |
| **最高** | 消费者处理完成 + 业务 ACK | ✅✅✅ | ⭐ |


---

### 2. 生产者确认机制是什么？

**答案：**

Publisher Confirm 是 RabbitMQ 提供的机制，确保生产者知道消息是否成功到达 Broker。

```
┌─────────────────────────────────────────────────────────┐
│              Publisher Confirm 工作流程                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐                ┌─────────────┐        │
│  │   Producer  │                │  RabbitMQ   │        │
│  └──────┬──────┘                └──────┬──────┘        │
│         │                              │                │
│         │  1. channel.confirm_select() │                │
│         │─────────────────────────────>│                │
│         │                              │                │
│         │  2. basic_publish(msg)       │                │
│         │─────────────────────────────>│                │
│         │                              │                │
│         │  3. 消息写入队列             │                │
│         │                              │                │
│         │  4. ack/nack                 │                │
│         │<─────────────────────────────│                │
│         │                              │                │
│         │  ack = 消息成功入队          │                │
│         │  nack = 消息入队失败         │                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**三种确认模式：**

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# ═════════════════════════════════════════════════════════
# 模式1：同步确认（每条消息单独确认）
# ═════════════════════════════════════════════════════════
channel.confirm_delivery()

try:
    channel.basic_publish(exchange='', routing_key='test', body='msg1')
    print("消息确认成功")
except pika.exceptions.UnroutableError:
    print("消息路由失败")

# ═════════════════════════════════════════════════════════
# 模式2：异步确认（批量确认，性能更高）
# ═════════════════════════════════════════════════════════

def on_confirm(method_frame):
    """确认回调函数"""
    if method_frame.method.NAME == 'Basic.Ack':
        print(f"消息 {method_frame.method.delivery_tag} 确认成功")
    else:
        print(f"消息 {method_frame.method.delivery_tag} 确认失败")

channel.confirm_delivery()
# 注意：pika 库的异步确认需要使用 SelectConnection 而非 BlockingConnection

# ═════════════════════════════════════════════════════════
# 模式3：事务（性能较差，不推荐）
# ═════════════════════════════════════════════════════════
channel.tx_select()  # 开启事务
try:
    channel.basic_publish(exchange='', routing_key='test', body='msg')
    channel.tx_commit()  # 提交
except Exception:
    channel.tx_rollback()  # 回滚
```

**Confirm vs Transaction：**

| 对比 | Transaction | Confirm |
|------|-------------|---------|
| **性能** | 慢（每次提交都同步等待） | 快（可异步批量确认） |
| **可靠性** | 高 | 高 |
| **推荐** | ❌ 不推荐 | ✅ 推荐 |

**追问：Confirm 模式下消息丢失的情况？**

**追问答案：**

Confirm 只保证消息到达 Broker，以下情况仍可能丢失：

1. **消息到达 Broker 但未持久化** → Broker 崩溃丢失
2. **消息路由失败**（Exchange 存在但没有匹配的队列）
3. **Broker 在持久化过程中崩溃**

解决方案：Confirm + 持久化 + 死信队列组合使用。

---

### 3. 消费者 ACK 机制是什么？

**答案：**

Consumer ACK 是消费者告知 Broker 消息处理结果的机制。

```
┌─────────────────────────────────────────────────────────┐
│              Consumer ACK 工作流程                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐                ┌─────────────┐        │
│  │  RabbitMQ   │                │  Consumer   │        │
│  │    Queue    │                │             │        │
│  └──────┬──────┘                └──────┬──────┘        │
│         │                              │                │
│         │  1. 投递消息                 │                │
│         │─────────────────────────────>│                │
│         │                              │                │
│         │  2. 消息状态变为 unacked     │                │
│         │                              │                │
│         │  3. 消费者处理消息           │                │
│         │                              │                │
│         │  4. basic_ack / basic_nack   │                │
│         │<─────────────────────────────│                │
│         │                              │                │
│         │  ack  → 消息从队列删除       │                │
│         │  nack → 消息重新入队或死信   │                │
│         │  超时 → 消息重新变为 ready   │                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**ACK 使用示例：**

```python
def consumer_callback(ch, method, properties, body):
    delivery_tag = method.delivery_tag
    
    try:
        message = json.loads(body)
        
        # 业务处理
        result = process_order(message)
        
        if result.success:
            # 处理成功，确认消息
            ch.basic_ack(delivery_tag=delivery_tag)
            logger.info(f"消息处理成功: {delivery_tag}")
            
        elif result.retryable:
            # 可重试错误，重新入队
            ch.basic_nack(delivery_tag=delivery_tag, requeue=True)
            logger.warning(f"消息重新入队: {delivery_tag}")
            
        else:
            # 不可恢复错误，拒绝消息（进入死信队列）
            ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
            logger.error(f"消息进入死信队列: {delivery_tag}")
            
    except Exception as e:
        logger.exception(f"处理异常: {e}")
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)

# 设置手动确认
channel.basic_qos(prefetch_count=1)
channel.basic_consume(
    queue='orders',
    on_message_callback=consumer_callback,
    auto_ack=False  # 关键：手动确认模式
)
```

**追问：auto_ack=True 有什么风险？**

**追问答案：**

- 消息投递后立即从队列删除
- 消费者处理失败或崩溃，消息永久丢失
- **仅适用于**：非关键消息、日志收集等可丢失场景

---

### 4. ACK / NACK / Reject 的区别？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              三种确认方式对比                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    ACK                           │   │
│  │  含义：消息处理成功，从队列删除                   │   │
│  │  用法：basic_ack(delivery_tag, multiple=False)  │   │
│  │  multiple=True: 批量确认 <= delivery_tag 的消息  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   NACK                           │   │
│  │  含义：消息处理失败                              │   │
│  │  用法：basic_nack(delivery_tag, requeue, multiple)│   │
│  │  requeue=True: 重新入队                         │   │
│  │  requeue=False: 进入死信队列                    │   │
│  │  支持批量操作                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                  Reject                          │   │
│  │  含义：拒绝单条消息                              │   │
│  │  用法：basic_reject(delivery_tag, requeue)      │   │
│  │  只能处理单条消息，不支持批量                    │   │
│  │  requeue 参数同 NACK                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码对比：**

```python
# ACK - 确认成功
channel.basic_ack(delivery_tag=1)
channel.basic_ack(delivery_tag=5, multiple=True)  # 批量确认 1-5

# NACK - 否定确认
channel.basic_nack(delivery_tag=1, requeue=True)   # 重新入队
channel.basic_nack(delivery_tag=1, requeue=False)  # 进入死信队列
channel.basic_nack(delivery_tag=5, requeue=False, multiple=True)  # 批量拒绝

# Reject - 拒绝单条
channel.basic_reject(delivery_tag=1, requeue=True)
```

**追问：什么时候用 requeue=True，什么时候用 False？**

**追问答案：**

| requeue | 场景 | 说明 |
|---------|------|------|
| **True** | 临时性错误（网络抖动、服务暂时不可用） | 消息重新入队，等待重试 |
| **False** | 永久性错误（数据格式错误、业务校验失败） | 进入死信队列，避免无限重试 |

---

### 5. 什么是持久化？如何完整保证？

**答案：**

详见基础篇第7题。总结：**Exchange 持久化 + Queue 持久化 + Message 持久化**，三者缺一不可。

---

### 6. 什么是 Publisher Confirm？如何使用？

**答案：**

详见消息可靠性篇第2题。

---

### 7. 什么是消费者 Prefetch？为什么要设置？

**答案：**

Prefetch 限制了 Broker 向消费者推送的未确认消息数量。

```
┌─────────────────────────────────────────────────────────┐
│              Prefetch 作用示意                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  不设置 Prefetch（不公平分发）：                        │
│  ┌─────────┐                                           │
│  │  Queue  │ [M1, M2, M3, M4, M5, M6]                  │
│  └────┬────┘                                           │
│       │                                                 │
│       ├────────────> Consumer A: [M1, M2, M3] (处理快)  │
│       └────────────> Consumer B: [M4, M5, M6] (处理慢)  │
│                                                         │
│  问题：Consumer A 处理完空闲，Consumer B 积压           │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  设置 Prefetch=1（公平分发）：                          │
│  ┌─────────┐                                           │
│  │  Queue  │ [M1, M2, M3, M4, M5, M6]                  │
│  └────┬────┘                                           │
│       │                                                 │
│       ├────> Consumer A: [M1] → ACK → [M3] → ACK → ...  │
│       └────> Consumer B: [M2] (处理中...)               │
│                                                         │
│  好处：处理快的消费者获得更多消息                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 设置 prefetch_count = 1
# 每个消费者最多同时持有 1 条未确认消息
channel.basic_qos(prefetch_count=1)

# prefetch_count 选择建议：
# • CPU 密集型：prefetch = CPU核心数
# • IO 密集型：prefetch = 10-50
# • 简单任务：prefetch = 1-5
```

**追问：Prefetch 设置过大或过小有什么问题？**

**追问答案：**

| Prefetch | 问题 |
|----------|------|
| **过小（=1）** | 网络开销大，吞吐量下降 |
| **过大** | 消息分配不均，可能积压在某个消费者 |
| **不设置** | 消息全部分配给先连接的消费者 |

---

### 8. 如何实现消息幂等性？

**答案：**

消息可能重复投递，消费者需要保证幂等性（多次处理结果一致）。

```
┌─────────────────────────────────────────────────────────┐
│              消息重复投递场景                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  场景1：消费者 ACK 丢失                                 │
│  ┌─────────┐     ┌─────────┐                           │
│  │  MQ     │────>│ Consumer│ 处理成功                  │
│  │         │     │ 发送 ACK│                           │
│  │         │<─X──│ (丢失)  │ ACK 网络丢失              │
│  │         │     └─────────┘                           │
│  │ 重新投递│                                           │
│  └─────────┘                                           │
│                                                         │
│  场景2：消费者处理超时                                  │
│  ┌─────────┐     ┌─────────┐                           │
│  │  MQ     │────>│ Consumer│ 处理时间长                │
│  │         │     │ 超时未ACK│                          │
│  │ 重新投递│     └─────────┘                           │
│  └─────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**幂等性实现方案：**

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def consume_with_idempotency(ch, method, properties, body):
    """幂等消费示例"""
    message = json.loads(body)
    message_id = message['message_id']  # 生产者生成的唯一ID
    
    # 方案1：Redis SETNX
    key = f"msg:processed:{message_id}"
    if redis_client.setnx(key, "1"):
        redis_client.expire(key, 86400)  # 24小时过期
        
        try:
            # 处理业务
            process_order(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            redis_client.delete(key)  # 失败时删除标记
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    else:
        # 消息已处理，直接确认
        print(f"消息 {message_id} 已处理，跳过")
        ch.basic_ack(delivery_tag=method.delivery_tag)

# 方案2：数据库唯一索引
def process_order(message):
    order_id = message['order_id']
    
    # INSERT 时利用唯一索引防止重复
    try:
        db.execute(
            "INSERT INTO order_processing (order_id, status) VALUES (?, 'processed')",
            (order_id,)
        )
    except UniqueConstraintError:
        # 已处理过，直接返回
        return
```

**追问：Redis 方案有什么风险？**

**追问答案：**

- Redis 挂掉时无法判断是否处理过
- 解决方案：
  1. 使用 Redis 持久化（AOF）
  2. Redis 集群高可用
  3. 降级到数据库唯一索引

---

### 9. 如何保证消息的顺序性？

**答案：**

RabbitMQ 单个 Queue 内消息有序，但多消费者并行消费会打乱顺序。

```
┌─────────────────────────────────────────────────────────┐
│              消息顺序性问题                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  问题：多消费者并行消费打乱顺序                         │
│                                                         │
│  Queue: [M1, M2, M3]                                    │
│           │                                             │
│           ├──> Consumer A: M1 (处理中...)               │
│           ├──> Consumer B: M2 (处理快，先完成)          │
│           └──> Consumer C: M3 (处理中...)               │
│                                                         │
│  实际完成顺序：M2 → M1 → M3 ❌                          │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  解决方案：单消费者 + 单线程                            │
│                                                         │
│  Queue: [M1, M2, M3]                                    │
│           │                                             │
│           └──> 单个 Consumer: M1 → M2 → M3 ✅           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**解决方案：**

```python
# 方案1：单消费者（性能受限）
channel.basic_qos(prefetch_count=1)
# 只启动一个消费者实例

# 方案2：分队列（推荐）
# 将需要保序的消息路由到同一队列
# 例如：订单状态变更，按订单ID路由到固定队列

def get_queue_name(order_id, num_queues=10):
    """根据订单ID计算队列编号"""
    queue_index = hash(order_id) % num_queues
    return f"order.queue.{queue_index}"

# 发送时指定路由
queue_name = get_queue_name(order_id='ORDER_001')
channel.basic_publish(
    exchange='',
    routing_key=queue_name,
    body=json.dumps(order_message)
)

# 每个队列只有一个消费者
```

**追问：性能和顺序性如何权衡？**

**追问答案：**

| 方案 | 顺序性 | 性能 | 适用场景 |
|------|--------|------|----------|
| 单消费者单线程 | ✅ 完全保序 | ⭐ | 顺序要求严格 |
| 分队列 | ✅ 组内保序 | ⭐⭐⭐ | 订单状态等 |
| 多消费者 | ❌ 不保序 | ⭐⭐⭐⭐⭐ | 无顺序要求 |

---

### 10. 消息积压怎么处理？

**答案：**

详见实战篇第1题。

---

### 11. 消息重复消费怎么解决？

**答案：**

详见消息可靠性篇第8题（幂等性）。

---

### 12. RabbitMQ 如何做消息追踪？

**答案：**

RabbitMQ 提供 Firehose 和 rabbitmq_tracing 插件进行消息追踪。

**方案1：Firehose（实时追踪）**

```bash
# 开启 Firehose
rabbitmqctl trace_on

# Firehose 会将消息发送到 amq.rabbitmq.trace 交换机
# 可以创建队列绑定该交换机进行追踪
```

**方案2：Tracing 插件（持久化追踪）**

```bash
# 启用插件
rabbitmq-plugins enable rabbitmq_tracing

# 在管理界面创建 trace，消息会记录到文件
# /var/log/rabbitmq/tracing.log
```

**方案3：自定义追踪**

```python
# 每条消息生成唯一 ID
message_id = str(uuid.uuid4())

# 发送时记录
channel.basic_publish(
    exchange='',
    routing_key='orders',
    body=json.dumps({'message_id': message_id, 'data': ...}),
    properties=pika.BasicProperties(
        message_id=message_id,
        headers={'trace_id': trace_id}
    )
)

# 消费时记录
def callback(ch, method, properties, body):
    message_id = properties.message_id
    log_message_event(message_id, 'consumed')
```


---

## 高级特性篇

### 1. 什么是死信队列（DLX）？

**答案：**

死信队列（Dead Letter Exchange）用于存储无法被正常消费的消息。

```
┌─────────────────────────────────────────────────────────┐
│              死信队列工作原理                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 正常队列                         │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │ 消息变成死信的条件：                     │   │   │
│  │  │ 1. 消息被拒绝 (nack/reject, requeue=false)│  │   │
│  │  │ 2. 消息 TTL 过期                         │   │   │
│  │  │ 3. 队列达到最大长度                      │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│                         │ 死信                                          │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Dead Letter Exchange                │   │
│  │                   (DLX)                          │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │               死信队列                           │   │
│  │  用于：错误排查、消息重试、告警                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 声明死信交换机和队列
channel.exchange_declare(exchange='dlx.exchange', exchange_type='direct')
channel.queue_declare(queue='dlx.queue', durable=True)
channel.queue_bind(queue='dlx.queue', exchange='dlx.exchange', routing_key='dlx')

# 声明正常队列，绑定死信交换机
args = {
    'x-dead-letter-exchange': 'dlx.exchange',      # 死信交换机
    'x-dead-letter-routing-key': 'dlx',            # 死信路由键
    'x-message-ttl': 30000,                        # 消息 TTL 30秒
    'x-max-length': 10000                          # 队列最大长度
}
channel.queue_declare(queue='normal.queue', durable=True, arguments=args)
```

**追问：死信消息有什么特殊信息？**

**追问答案：**

死信消息会携带额外的 headers：

```python
# 死信消息 headers
{
    'x-death': [
        {
            'count': 1,                    # 死信次数
            'reason': 'rejected',          # 死信原因：rejected/expired/maxlen
            'queue': 'normal.queue',       # 原队列
            'exchange': '',                # 原交换机
            'routing-keys': ['normal.queue'],  # 原路由键
            'time': timestamp              # 死信时间
        }
    ]
}
```

---

### 2. 死信产生的条件有哪些？

**答案：**

| 条件 | 说明 | 场景 |
|------|------|------|
| **消息被拒绝** | nack/reject + requeue=false | 消息格式错误、业务校验失败 |
| **消息 TTL 过期** | 消息存活时间超过设定值 | 延迟任务超时 |
| **队列溢出** | 队列长度超过 x-max-length | 流量过大 |

**代码示例：**

```python
# 场景1：消息被拒绝
def callback(ch, method, properties, body):
    try:
        process_message(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except FatalError:
        # 拒绝消息且不重新入队 → 进入死信队列
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

# 场景2：消息 TTL 过期
channel.queue_declare(
    queue='ttl.queue',
    arguments={
        'x-message-ttl': 60000,  # 消息存活60秒
        'x-dead-letter-exchange': 'dlx.exchange'
    }
)

# 场景3：队列溢出
channel.queue_declare(
    queue='maxlen.queue',
    arguments={
        'x-max-length': 1000,  # 最多1000条消息
        'x-overflow': 'reject-publish-dlx',  # 溢出时发送到DLX
        'x-dead-letter-exchange': 'dlx.exchange'
    }
)
```

---

### 3. 什么是延迟队列？如何实现？

**答案：**

延迟队列用于消息在指定时间后被消费，常用于订单超时取消、定时任务等场景。

```
┌─────────────────────────────────────────────────────────┐
│              延迟队列工作原理                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  方案：TTL + 死信队列                                   │
│                                                         │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ 生产者   │────>│ TTL 队列     │────>│ DLX         │  │
│  │         │     │ (等待过期)   │     │ (死信交换机) │  │
│  └─────────┘     │ TTL=30min   │     └──────┬──────┘  │
│                  └─────────────┘            │          │
│                           │                 │          │
│                           ▼                 ▼          │
│                    消息过期后         ┌─────────┐      │
│                    变成"死信"         │ 消费队列 │      │
│                                       │ 处理订单 │      │
│                                       └─────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**实现方式一：队列级 TTL**

```python
# 声明死信交换机和消费队列
channel.exchange_declare(exchange='dlx', exchange_type='direct')
channel.queue_declare(queue='process.queue', durable=True)
channel.queue_bind(queue='process.queue', exchange='dlx', routing_key='process')

# 声明延迟队列（固定 TTL）
channel.queue_declare(
    queue='delay.30min',
    durable=True,
    arguments={
        'x-message-ttl': 1800000,  # 30分钟 = 30 * 60 * 1000
        'x-dead-letter-exchange': 'dlx',
        'x-dead-letter-routing-key': 'process'
    }
)

# 发送消息到延迟队列
channel.basic_publish(
    exchange='',
    routing_key='delay.30min',
    body='Order created, wait for payment'
)
# 消息将在30分钟后进入 process.queue
```

**实现方式二：消息级 TTL**

```python
# 消息级别 TTL，每条消息可以有不同的延迟时间
channel.basic_publish(
    exchange='',
    routing_key='delay.queue',  # 这个队列 TTL=0 或很大
    body='Order message',
    properties=pika.BasicProperties(
        expiration='60000',  # 60秒后过期
    )
)
```

**实现方式三：RabbitMQ 延迟插件（推荐）**

```bash
# 安装插件
rabbitmq-plugins enable rabbitmq_delayed_message_exchange
```

```python
# 使用延迟交换机
channel.exchange_declare(
    exchange='delayed.exchange',
    exchange_type='x-delayed-message',
    arguments={'x-delayed-type': 'direct'}
)

# 发送延迟消息
channel.basic_publish(
    exchange='delayed.exchange',
    routing_key='order.cancel',
    body='Order will be cancelled',
    properties=pika.BasicProperties(
        headers={'x-delay': 30000}  # 延迟30秒
    )
)
```

**追问：两种延迟队列实现方式对比？**

**追问答案：**

| 对比 | TTL+DLX | 延迟插件 |
|------|---------|----------|
| **原理** | 消息过期变成死信 | 延迟交换机直接延迟 |
| **消息顺序** | 先进先出，短 TTL 可能被阻塞 | 无此问题 |
| **灵活性** | 消息级 TTL 可变 | 直接设置延迟时间 |
| **依赖** | 无 | 需要安装插件 |
| **推荐** | 简单场景 | 复杂延迟场景 |

---

### 4. TTL 是什么？如何设置？

**答案：**

TTL（Time To Live）是消息存活时间，过期消息会被删除或进入死信队列。

```
┌─────────────────────────────────────────────────────────┐
│              TTL 设置方式                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 队列级 TTL                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  x-message-ttl = 60000                          │   │
│  │  该队列所有消息统一 60 秒过期                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. 消息级 TTL                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  properties.expiration = '30000'                │   │
│  │  单条消息单独设置 TTL                           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. 队列过期时间                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  x-expires = 3600000                            │   │
│  │  队列无消费者使用 1 小时后自动删除               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 队列级 TTL
channel.queue_declare(
    queue='ttl.queue',
    arguments={
        'x-message-ttl': 60000,  # 60秒
    }
)

# 消息级 TTL
channel.basic_publish(
    exchange='',
    routing_key='queue',
    body='message',
    properties=pika.BasicProperties(
        expiration='30000',  # 30秒（字符串）
    )
)

# 两者同时设置，取较小值
```

---

### 5. 什么是优先级队列？

**答案：**

优先级队列允许高优先级消息优先被消费。

```python
# 声明优先级队列
channel.queue_declare(
    queue='priority.queue',
    arguments={
        'x-max-priority': 10  # 支持 0-10 共 11 个优先级
    }
)

# 发送高优先级消息
channel.basic_publish(
    exchange='',
    routing_key='priority.queue',
    body='Urgent message',
    properties=pika.BasicProperties(
        priority=10  # 最高优先级
    )
)

# 发送低优先级消息
channel.basic_publish(
    exchange='',
    routing_key='priority.queue',
    body='Normal message',
    properties=pika.BasicProperties(
        priority=1  # 低优先级
    )
)
```

**注意事项：**

- 优先级越高，数值越大
- 消费者空闲时才生效
- 性能有一定开销，谨慎使用

---

### 6. 什么是消息确认超时？

**答案：**

消费者长时间未 ACK，消息会重新变为 ready 状态。

```
┌─────────────────────────────────────────────────────────┐
│              消息确认超时机制                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  消息投递后状态变化：                                   │
│                                                         │
│  ready → unacked → (超时) → ready                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  timeout = heartbeat * 2                        │   │
│  │  默认：heartbeat=60s, timeout≈120s             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  消费者崩溃时：                                         │
│  • 连接断开，所有 unacked 消息立即变为 ready           │
│                                                         │
│  消费者假死时：                                         │
│  • 超时后消息重新变为 ready                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 设置合理的 heartbeat
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='localhost',
        heartbeat=30,  # 心跳间隔 30 秒
        blocked_connection_timeout=300
    )
)

# 消费者处理时间不应超过超时时间
def callback(ch, method, properties, body):
    try:
        # 设置处理超时
        with timeout(60):  # 60秒超时
            process_message(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except TimeoutError:
        # 处理超时，NACK 并重新入队
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
```

---

### 7. 什么是 RPC 模式？如何实现？

**答案：**

RPC（远程过程调用）模式使用 RabbitMQ 实现同步调用。

```
┌─────────────────────────────────────────────────────────┐
│              RPC 模式工作流程                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐                ┌──────────┐              │
│  │  Client  │                │  Server  │              │
│  │  (调用方) │                │ (服务方) │              │
│  └────┬─────┘                └────┬─────┘              │
│       │                           │                     │
│       │  1. 请求 (reply_to, corr_id)                   │
│       │──────────────────────────>│                     │
│       │                           │                     │
│       │  2. 处理请求             │                     │
│       │                           │                     │
│       │  3. 响应 (routing_key=reply_to)                │
│       │<──────────────────────────│                     │
│       │                           │                     │
│       │  4. 根据 corr_id 匹配响应  │                    │
│       │                           │                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
import pika
import uuid
import json

class RPCClient:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.channel = self.connection.channel()
        
        # 声明临时回复队列
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        
        # 监听回复队列
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
        
        self.response = None
        self.corr_id = None
    
    def on_response(self, ch, method, props, body):
        """收到响应"""
        if self.corr_id == props.correlation_id:
            self.response = json.loads(body)
    
    def call(self, method_name, params):
        """发起 RPC 调用"""
        self.response = None
        self.corr_id = str(uuid.uuid4())
        
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps({'method': method_name, 'params': params})
        )
        
        # 等待响应
        while self.response is None:
            self.connection.process_data_events()
        
        return self.response

# 使用
client = RPCClient()
result = client.call('add', {'a': 1, 'b': 2})
print(result)  # {'result': 3}
```

---

### 8. 如何实现消息重试机制？

**答案：**

```python
MAX_RETRIES = 3

def callback(ch, method, properties, body):
    message = json.loads(body)
    
    # 获取重试次数
    retry_count = properties.headers.get('x-retry-count', 0) if properties.headers else 0
    
    try:
        process_message(message)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except RetryableError as e:
        if retry_count < MAX_RETRIES:
            # 重试：发送到延迟队列
            channel.basic_publish(
                exchange='',
                routing_key='retry.queue',
                body=body,
                properties=pika.BasicProperties(
                    headers={'x-retry-count': retry_count + 1}
                )
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"消息重试 {retry_count + 1}/{MAX_RETRIES}")
        else:
            # 超过最大重试次数，进入死信队列
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print("超过最大重试次数，进入死信队列")
    
    except FatalError:
        # 致命错误，直接进入死信队列
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
```


---

## 高可用篇

### 1. RabbitMQ 如何保证高可用？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 高可用架构                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Level 1: 单节点持久化                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • Exchange、Queue、Message 持久化              │   │
│  │  • 防止重启丢失消息                             │   │
│  │  • 无法防止机器宕机                             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Level 2: 普通集群                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 多节点共享元数据                             │   │
│  │  • 消息只存在创建队列的节点                     │   │
│  │  • 节点宕机，队列消失                           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Level 3: 镜像队列集群                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 队列消息同步到多个节点                       │   │
│  │  • 主节点宕机，从节点自动接管                   │   │
│  │  • 同步复制，性能有损耗                         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Level 4: Quorum Queue（仲裁队列）                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • RabbitMQ 3.8+ 新方案                         │   │
│  │  • 基于 Raft 协议                               │   │
│  │  • 数据一致性和可用性更好                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**追问：生产环境推荐哪种方案？**

**追问答案：**

- **RabbitMQ 3.8+**：推荐 Quorum Queue
- **旧版本**：镜像队列集群
- **关键配置**：至少 3 节点，奇数节点

---

### 2. 镜像队列是什么？

**答案：**

镜像队列（Mirrored Queue）将队列复制到多个节点，实现高可用。

```
┌─────────────────────────────────────────────────────────┐
│              镜像队列工作原理                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   Cluster                        │   │
│  │                                                  │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │   │
│  │  │  Node A   │  │  Node B   │  │  Node C   │   │   │
│  │  │  (Master) │  │ (Mirror)  │  │ (Mirror)  │   │   │
│  │  │           │  │           │  │           │   │   │
│  │  │  ┌─────┐  │  │  ┌─────┐  │  │  ┌─────┐  │   │   │
│  │  │  │Queue│  │  │  │Queue│  │  │  │Queue│  │   │   │
│  │  │  │[M1] │  │  │  │[M1] │  │  │  │[M1] │  │   │   │
│  │  │  │[M2] │  │  │  │[M2] │  │  │  │[M2] │  │   │   │
│  │  │  └─────┘  │  │  └─────┘  │  │  └─────┘  │   │   │
│  │  │           │  │           │  │           │   │   │
│  │  │  写入 ────┼──> 同步 ────┼──> 同步     │   │   │
│  │  └───────────┘  └───────────┘  └───────────┘   │   │
│  │                                                  │   │
│  │  Node A 宕机 → Node B 或 C 自动成为 Master      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**配置策略：**

```bash
# 通过策略配置镜像
rabbitmqctl set_policy ha-all ".*" '{"ha-mode":"all","ha-sync-mode":"automatic"}'

# ha-mode 选项：
# • all: 镜像到所有节点
# • exactly: 镜像到指定数量节点
# • nodes: 镜像到指定节点

# 示例：镜像到 2 个节点
rabbitmqctl set_policy ha-two "^ha\." '{"ha-mode":"exactly","ha-params":2}'
```

**镜像队列缺点：**

- 同步复制，性能损耗
- 网络分区时可能脑裂
- 已被 Quorum Queue 取代

---

### 3. Quorum Queue 是什么？

**答案：**

Quorum Queue 是 RabbitMQ 3.8+ 引入的新队列类型，基于 Raft 协议实现。

```
┌─────────────────────────────────────────────────────────┐
│              Quorum Queue vs 镜像队列                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    Quorum Queue              镜像队列    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  一致性协议       Raft               主从复制    │   │
│  │  数据存储         磁盘 (WAL)         内存        │   │
│  │  脑裂处理         自动恢复           需手动干预  │   │
│  │  性能             较高               较低        │   │
│  │  推荐使用         ✅                 ❌ (已废弃) │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**创建 Quorum Queue：**

```python
channel.queue_declare(
    queue='quorum.queue',
    arguments={'x-queue-type': 'quorum'}
)
```

**追问：Quorum Queue 有什么限制？**

**追问答案：**

- 最少需要 3 个节点（Raft 需要）
- 不支持非持久化
- 不支持消息优先级
- 不支持 exclusivity（独占队列）

---

### 4. 普通集群和镜像集群有什么区别？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              普通集群 vs 镜像集群                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  普通集群：                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 元数据（队列信息）在所有节点同步              │   │
│  │  • 消息只存储在创建队列的节点                    │   │
│  │  • 节点宕机，该节点队列消失                      │   │
│  │  • 适合：扩展吞吐量，非关键业务                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  镜像集群：                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 队列数据同步到多个节点                        │   │
│  │  • 主节点宕机，从节点接管                        │   │
│  │  • 适合：关键业务，高可用要求                    │   │
│  │  • 注意：已被 Quorum Queue 取代                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**追问：如何选择集群类型？**

**追问答案：**

| 需求 | 推荐 |
|------|------|
| 高吞吐量、允许少量丢失 | 普通集群 |
| 高可用、数据安全 | Quorum Queue |
| 旧版本、必须高可用 | 镜像队列 |

---

### 5. 什么是网络分区？如何处理？

**答案：**

网络分区（Network Partition）是集群节点间网络中断导致的脑裂问题。

```
┌─────────────────────────────────────────────────────────┐
│              网络分区示意                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  正常状态：                                              │
│  ┌───────────┐         ┌───────────┐                   │
│  │  Node A   │<───────>│  Node B   │                   │
│  │  Master   │         │  Mirror   │                   │
│  └───────────┘         └───────────┘                   │
│                                                         │
│  网络分区后：                                            │
│  ┌───────────┐         ┌───────────┐                   │
│  │  Node A   │    X    │  Node B   │                   │
│  │  Master   │ (断开)  │  Master   │ ← 两个 Master!   │
│  └───────────┘         └───────────┘                   │
│                                                         │
│  问题：                                                  │
│  • 两个分区各自成为 Master                              │
│  • 客户端写入不同分区，数据不一致                       │
│  • 恢复后需要人工干预                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**分区处理策略：**

```bash
# 配置分区处理策略
# 在 rabbitmq.conf 中设置
cluster_partition_handling = pause_minority  # 推荐
# 或
cluster_partition_handling = autoheal
```

| 策略 | 说明 |
|------|------|
| **ignore** | 忽略分区（不推荐） |
| **pause_minority** | 少数派节点暂停（推荐） |
| **autoheal** | 自动恢复，选择多数派 |

**追问：如何监控网络分区？**

**追问答案：**

```bash
# 查看分区状态
rabbitmqctl cluster_status

# 管理界面可以看到分区警告
# 建议配置告警：监听 partition 状态
```

---

### 6. 如何做 RabbitMQ 监控？

**答案：**

**关键监控指标：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 关键监控指标                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 基础指标                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • connections: 连接数                          │   │
│  │  • channels: 通道数                             │   │
│  │  • queues: 队列数                               │   │
│  │  • consumers: 消费者数                          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. 消息指标                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • messages_ready: 待消费消息数                 │   │
│  │  • messages_unacked: 未确认消息数               │   │
│  │  • message_stats.publish: 发布速率              │   │
│  │  • message_stats.deliver: 投递速率              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. 性能指标                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • disk_free: 磁盘剩余空间                      │   │
│  │  • mem_used: 内存使用                           │   │
│  │  • fd_used: 文件描述符使用                      │   │
│  │  • sockets_used: socket 使用                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  4. 集群指标                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • partition: 网络分区状态                      │   │
│  │  • node_health: 节点健康状态                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**监控方案：**

```yaml
# Prometheus + Grafana 监控
# rabbitmq_prometheus 插件
rabbitmq-plugins enable rabbitmq_prometheus

# 指标端点
http://localhost:15692/metrics
```

**告警规则示例：**

```yaml
# 告警配置
groups:
  - name: rabbitmq
    rules:
      - alert: RabbitMQQueueDepth
        expr: rabbitmq_queue_messages_ready > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "队列消息积压"

      - alert: RabbitMQMemoryHigh
        expr: rabbitmq_mem_used / rabbitmq_mem_limit > 0.8
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ 内存使用过高"
```

---

## 实战篇

### 1. 消息积压怎么处理？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              消息积压处理方案                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: 排查原因                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 消费者故障？                                  │   │
│  │  • 消费速度慢？                                  │   │
│  │  • 突发流量？                                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Step 2: 临时方案                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. 增加消费者实例                              │   │
│  │  2. 临时转发到新队列分流                        │   │
│  │  3. 扩容消费者资源                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Step 3: 长期方案                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. 优化消费逻辑（批量、异步）                   │   │
│  │  2. 设置队列最大长度告警                        │   │
│  │  3. 监控消费延迟                                │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 方案1：批量消费提升吞吐
def batch_consumer():
    messages = []
    
    def callback(ch, method, properties, body):
        messages.append((method.delivery_tag, body))
        
        if len(messages) >= 100:  # 批量处理
            process_batch(messages)
            # 批量确认
            for tag, _ in messages:
                ch.basic_ack(delivery_tag=tag)
            messages.clear()
    
    channel.basic_qos(prefetch_count=100)
    channel.basic_consume(queue='backlog.queue', on_message_callback=callback)

# 方案2：转发到多个队列分流
def redistribute_messages():
    # 将积压队列的消息转发到多个新队列
    for i in range(10):
        channel.queue_declare(queue=f'shard.{i}')
    
    while True:
        method, properties, body = channel.basic_get(queue='backlog.queue')
        if method is None:
            break
        
        shard = hash(body) % 10
        channel.basic_publish(
            exchange='',
            routing_key=f'shard.{shard}',
            body=body
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)
```

**追问：生产环境遇到过消息积压吗？怎么解决的？**

**追问答案：**

根据实际情况回答，可参考以下思路：

1. **场景描述**：大促活动导致订单消息积压
2. **原因**：消费者依赖的外部服务响应慢
3. **解决**：
   - 临时增加消费者实例
   - 异步化外部调用
   - 设置降级策略
4. **预防**：监控告警、容量规划

---

### 2. 消息重复消费怎么解决？

**答案：**

详见消息可靠性篇第8题（幂等性）。

---

### 3. 消息顺序性怎么保证？

**答案：**

详见消息可靠性篇第9题。

---

### 4. 如何实现消息的延迟重试？

**答案：**

```python
# 延迟重试方案：TTL + DLX
def setup_retry_queue():
    # 死信交换机
    channel.exchange_declare(exchange='dlx', exchange_type='direct')
    channel.queue_declare(queue='retry.queue', durable=True)
    channel.queue_bind(queue='retry.queue', exchange='dlx', routing_key='retry')
    
    # 延迟队列（重试等待）
    for delay in [1000, 5000, 30000, 60000]:  # 1s, 5s, 30s, 1min
        queue_name = f'delay.{delay}'
        channel.queue_declare(
            queue=queue_name,
            arguments={
                'x-message-ttl': delay,
                'x-dead-letter-exchange': 'dlx',
                'x-dead-letter-routing-key': 'retry'
            }
        )

def consumer_with_retry(ch, method, properties, body):
    retry_count = properties.headers.get('x-retry-count', 0) if properties.headers else 0
    
    try:
        process_message(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        if retry_count < 4:
            # 根据重试次数选择延迟队列
            delays = [1000, 5000, 30000, 60000]
            delay_queue = f'delay.{delays[retry_count]}'
            
            ch.basic_publish(
                exchange='',
                routing_key=delay_queue,
                body=body,
                properties=pika.BasicProperties(
                    headers={'x-retry-count': retry_count + 1}
                )
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
```

---

### 5. RabbitMQ 如何做性能优化？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 性能优化策略                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 生产者优化                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 使用异步 Confirm 替代同步                     │   │
│  │  • 批量发送消息                                  │   │
│  │  • 合理设置持久化（非关键消息可非持久化）         │   │
│  │  • 连接池复用                                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. Broker 优化                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 增加内存限制                                  │   │
│  │  • 使用 SSD 存储                                 │   │
│  │  • 调整文件描述符限制                            │   │
│  │  • 集群扩展                                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. 消费者优化                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 合理设置 prefetch_count                      │   │
│  │  • 批量确认                                      │   │
│  │  • 异步处理                                      │   │
│  │  • 消费者并发                                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**配置优化：**

```bash
# rabbitmq.conf
# 内存限制
vm_memory_high_watermark.relative = 0.6

# 磁盘限制
disk_free_limit.absolute = 10GB

# 文件描述符
# 需要系统层面设置 ulimit -n 65535

# 消息索引优化
queue_index_embed_msgs_below = 4096  # 小消息内联存储
```

---

### 6. 如何排查 RabbitMQ 性能问题？

**答案：**

**排查步骤：**

```bash
# 1. 查看队列状态
rabbitmqctl list_queues name messages_ready messages_unacked consumers

# 2. 查看连接和通道
rabbitmqctl list_connections
rabbitmqctl list_channels

# 3. 查看节点状态
rabbitmqctl status

# 4. 查看网络流量
rabbitmqctl node_health_check

# 5. 开启 Firehose 追踪
rabbitmqctl trace_on
```

**常见问题及解决：**

| 问题 | 表现 | 解决 |
|------|------|------|
| 消费者处理慢 | messages_unacked 多 | 优化消费逻辑、增加消费者 |
| 磁盘 IO 高 | 写入慢 | 使用 SSD、减少持久化 |
| 内存不足 | 被阻塞 | 增加内存、流量控制 |
| 连接数过多 | 资源耗尽 | 连接池、监控告警 |


---

## 场景篇

### 1. 订单超时取消如何实现？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              订单超时取消架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐                                        │
│  │  订单服务    │                                        │
│  │ 创建订单     │                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         │ 发送消息到延迟队列                             │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │ 延迟队列    │  TTL = 30分钟                          │
│  │ delay.queue│                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         │ 消息过期后变成死信                             │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │ 死信交换机  │                                        │
│  │    DLX     │                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │ 取消队列    │────>│ 订单服务    │                   │
│  │ cancel.q   │     │ 检查并取消   │                   │
│  └─────────────┘     └─────────────┘                   │
│                                                         │
│  处理逻辑：                                              │
│  1. 订单创建时发送延迟消息                              │
│  2. 30分钟后消息到达取消队列                            │
│  3. 检查订单状态，未支付则取消                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码实现：**

```python
import pika
import json
import time

# ═════════════════════════════════════════════════════════
# 初始化延迟队列架构
# ═════════════════════════════════════════════════════════

def setup_delay_queue():
    channel = connection.channel()
    
    # 死信交换机（用于超时消息）
    channel.exchange_declare(exchange='order.dlx', exchange_type='direct')
    
    # 取消订单队列
    channel.queue_declare(queue='order.cancel', durable=True)
    channel.queue_bind(queue='order.cancel', exchange='order.dlx', routing_key='cancel')
    
    # 延迟队列（30分钟）
    channel.queue_declare(
        queue='order.delay.30min',
        durable=True,
        arguments={
            'x-message-ttl': 1800000,  # 30分钟
            'x-dead-letter-exchange': 'order.dlx',
            'x-dead-letter-routing-key': 'cancel'
        }
    )

# ═════════════════════════════════════════════════════════
# 创建订单时发送延迟消息
# ═════════════════════════════════════════════════════════

def create_order(order_id, user_id, amount):
    # 1. 创建订单记录
    db.execute(
        "INSERT INTO orders (order_id, user_id, amount, status) VALUES (?, ?, ?, 'pending')",
        (order_id, user_id, amount)
    )
    
    # 2. 发送延迟消息
    channel.basic_publish(
        exchange='',
        routing_key='order.delay.30min',
        body=json.dumps({
            'order_id': order_id,
            'created_at': time.time()
        }),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 持久化
        )
    )
    
    return order_id

# ═════════════════════════════════════════════════════════
# 消费超时消息，取消订单
# ═════════════════════════════════════════════════════════

def cancel_order_consumer(ch, method, properties, body):
    message = json.loads(body)
    order_id = message['order_id']
    
    # 查询订单状态
    order = db.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
    
    if order['status'] == 'pending':
        # 未支付，取消订单
        db.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))
        
        # 恢复库存
        restore_inventory(order_id)
        
        logger.info(f"订单 {order_id} 超时取消")
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

**追问：如果支付成功刚好在超时取消时发生怎么办？**

**追问答案：**

使用分布式锁或数据库乐观锁：

```python
def cancel_order_consumer(ch, method, properties, body):
    order_id = json.loads(body)['order_id']
    
    # 使用乐观锁更新，只有 pending 状态才能取消
    result = db.execute(
        "UPDATE orders SET status = 'cancelled' WHERE order_id = ? AND status = 'pending'",
        (order_id,)
    )
    
    if result.rowcount > 0:
        restore_inventory(order_id)
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

---

### 2. 异步解耦怎么设计？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              订单系统异步解耦架构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  同步模式（问题）：                                     │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ 订单服务 │────>│ 支付服务 │────>│ 物流服务 │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│       │                                               │
│       └────> 通知服务                                  │
│                                                         │
│  问题：                                                 │
│  • 任何服务故障导致整个链路失败                         │
│  • 响应时间长（所有服务延迟之和）                       │
│  • 扩展困难                                            │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  异步模式（解决方案）：                                 │
│                                                         │
│  ┌─────────┐                                           │
│  │ 订单服务 │                                           │
│  │ (订单创建)│                                          │
│  └────┬────┘                                           │
│       │                                                 │
│       │  订单创建事件                                   │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────┐       │
│  │                Topic Exchange                │       │
│  │               order.events                   │       │
│  └───────────────────┬─────────────────────────┘       │
│                      │                                  │
│       ┌──────────────┼──────────────┐                  │
│       │              │              │                  │
│       ▼              ▼              ▼                  │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│  │ 支付队列 │   │ 物流队列 │   │ 通知队列 │             │
│  │ order.* │   │order.paid│   │ order.* │             │
│  └────┬────┘   └────┬────┘   └────┬────┘             │
│       │              │              │                  │
│       ▼              ▼              ▼                  │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│  │ 支付服务 │   │ 物流服务 │   │ 通知服务 │             │
│  └─────────┘   └─────────┘   └─────────┘             │
│                                                         │
│  好处：                                                 │
│  • 解耦：各服务独立部署                                │
│  • 异步：响应快                                        │
│  • 可靠：消息持久化                                    │
│  • 扩展：新增服务只需订阅                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# 订单服务：发布订单事件
def publish_order_event(event_type, order_data):
    channel.basic_publish(
        exchange='order.events',
        routing_key=f'order.{event_type}',
        body=json.dumps(order_data),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type='application/json'
        )
    )

# 支付服务：订阅订单创建事件
channel.queue_declare(queue='payment.order_created', durable=True)
channel.queue_bind(
    queue='payment.order_created',
    exchange='order.events',
    routing_key='order.created'
)

# 通知服务：订阅所有订单事件
channel.queue_declare(queue='notification.orders', durable=True)
channel.queue_bind(
    queue='notification.orders',
    exchange='order.events',
    routing_key='order.#'
)
```

---

### 3. 流量削峰怎么实现？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              秒杀系统流量削峰架构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户请求 (10万 QPS)                                    │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────┐       │
│  │                网关层                        │       │
│  │  • 限流（令牌桶/漏桶）                       │       │
│  │  • 黑名单过滤                                │       │
│  └─────────────────────┬───────────────────────┘       │
│                        │                                │
│                        ▼                                │
│  ┌─────────────────────────────────────────────┐       │
│  │             RabbitMQ 队列                    │       │
│  │  ┌─────────────────────────────────────┐   │       │
│  │  │  seckill.queue                       │   │       │
│  │  │  [M1][M2][M3]...[M100000]           │   │       │
│  │  └─────────────────────────────────────┘   │       │
│  │  缓冲请求，平滑流量                          │       │
│  └─────────────────────┬───────────────────────┘       │
│                        │                                │
│                        │ 按消费速度推送                  │
│                        ▼                                │
│  ┌─────────────────────────────────────────────┐       │
│  │              后端服务 (1000 QPS)             │       │
│  │  • 库存校验                                  │       │
│  │  • 订单创建                                  │       │
│  │  • 数据库写入                                │       │
│  └─────────────────────────────────────────────┘       │
│                                                         │
│  关键配置：                                              │
│  • 队列最大长度限制                                     │
│  • 消费者 prefetch 设置                                 │
│  • 消息持久化                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# ═════════════════════════════════════════════════════════
# 秒杀入口：请求入队
# ═════════════════════════════════════════════════════════

def seckill_entry(user_id, product_id):
    # 快速校验
    if not check_user_qualification(user_id):
        return {'success': False, 'message': '无参与资格'}
    
    # 构造消息
    message = {
        'user_id': user_id,
        'product_id': product_id,
        'timestamp': time.time()
    }
    
    # 异步入队
    try:
        channel.basic_publish(
            exchange='',
            routing_key='seckill.queue',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化
            )
        )
        return {'success': True, 'message': '排队中，请稍后查询结果'}
    except Exception:
        return {'success': False, 'message': '系统繁忙'}

# ═════════════════════════════════════════════════════════
# 后端消费：处理秒杀请求
# ═════════════════════════════════════════════════════════

def seckill_consumer(ch, method, properties, body):
    message = json.loads(body)
    user_id = message['user_id']
    product_id = message['product_id']
    
    try:
        # 库存扣减（使用 Redis 原子操作）
        stock = redis.decr(f'stock:{product_id}')
        
        if stock >= 0:
            # 创建订单
            order_id = create_order(user_id, product_id)
            
            # 异步通知用户
            notify_user(user_id, order_id)
            
            logger.info(f"秒杀成功: user={user_id}, order={order_id}")
        else:
            # 库存不足
            redis.incr(f'stock:{product_id}')  # 恢复
            logger.info(f"秒杀失败: user={user_id}, 库存不足")
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        logger.exception(f"秒杀处理异常: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# 设置 prefetch，控制消费速度
channel.basic_qos(prefetch_count=10)
```

**追问：如何防止超卖？**

**追问答案：**

```python
# 使用 Redis 原子操作 + Lua 脚本
lua_script = """
if redis.call('get', KEYS[1]) > 0 then
    return redis.call('decr', KEYS[1])
else
    return -1
end
"""

stock = redis.eval(lua_script, 1, f'stock:{product_id}')
if stock >= 0:
    # 有库存
    create_order(user_id, product_id)
```

---

### 4. 分布式事务怎么处理？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              分布式事务处理方案                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  场景：下单 + 扣库存 + 扣余额                           │
│                                                         │
│  方案：本地消息表 + 最终一致性                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   订单服务                       │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │  1. 本地事务：                           │   │   │
│  │  │     - 创建订单                          │   │   │
│  │  │     - 写入消息表（待发送）               │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  │                    │                            │   │
│  │                    ▼                            │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │  2. 定时任务：                           │   │   │
│  │  │     - 扫描消息表                         │   │   │
│  │  │     - 发送消息到 MQ                      │   │   │
│  │  │     - 更新消息状态                       │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              RabbitMQ                           │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   库存服务                       │   │
│  │  3. 消费消息：                                   │   │
│  │     - 扣减库存（幂等）                           │   │
│  │     - 发送 ACK                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
# ═════════════════════════════════════════════════════════
# 订单服务：本地消息表
# ═════════════════════════════════════════════════════════

def create_order_with_message(order_data):
    """创建订单并写入消息表"""
    message_id = str(uuid.uuid4())
    
    # 本地事务
    with db.transaction():
        # 1. 创建订单
        db.execute(
            "INSERT INTO orders (order_id, ...) VALUES (...)",
            (...)
        )
        
        # 2. 写入消息表
        db.execute(
            """INSERT INTO message_outbox 
               (message_id, exchange, routing_key, body, status) 
               VALUES (?, ?, ?, ?, 'pending')""",
            (message_id, 'order.exchange', 'order.created', json.dumps(order_data))
        )

def send_pending_messages():
    """定时任务：发送待发送消息"""
    messages = db.execute(
        "SELECT * FROM message_outbox WHERE status = 'pending' LIMIT 100"
    )
    
    for msg in messages:
        try:
            channel.basic_publish(
                exchange=msg['exchange'],
                routing_key=msg['routing_key'],
                body=msg['body'],
                properties=pika.BasicProperties(
                    message_id=msg['message_id'],
                    delivery_mode=2
                )
            )
            
            # 更新状态
            db.execute(
                "UPDATE message_outbox SET status = 'sent' WHERE message_id = ?",
                (msg['message_id'],)
            )
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ═════════════════════════════════════════════════════════
# 库存服务：幂等消费
# ═════════════════════════════════════════════════════════

def consume_order_created(ch, method, properties, body):
    message = json.loads(body)
    message_id = properties.message_id
    order_id = message['order_id']
    
    # 幂等检查
    if is_processed(message_id):
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
    
    # 扣减库存
    deduct_stock(message['product_id'], message['quantity'])
    
    # 标记已处理
    mark_processed(message_id)
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

---

### 5. 项目的 MQ 架构是怎样的？

**答案：**

根据实际项目回答，可参考以下架构：

```
┌─────────────────────────────────────────────────────────┐
│              典型电商 MQ 架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                  RabbitMQ 集群                   │   │
│  │           (3节点, Quorum Queue)                 │   │
│  │                                                  │   │
│  │  Exchanges:                                      │   │
│  │  • order.events (Topic)                         │   │
│  │  • payment.events (Topic)                       │   │
│  │  • notification.fanout (Fanout)                 │   │
│  │                                                  │   │
│  │  Queues:                                         │   │
│  │  • order.created → 支付、库存、通知              │   │
│  │  • order.paid → 物流、积分                       │   │
│  │  • order.delay.30min → 超时取消                 │   │
│  │  • dlx.queue → 死信处理                          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  监控：                                                 │
│  • Prometheus + Grafana                                │
│  • 告警：队列积压、内存、磁盘                          │
│                                                         │
│  可靠性：                                               │
│  • 持久化：Exchange + Queue + Message                  │
│  • 确认：Publisher Confirm + Consumer ACK              │
│  • 高可用：Quorum Queue                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 6. 如何设计一个高可用的消息系统？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              高可用消息系统设计                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 架构层面                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • RabbitMQ 集群：至少 3 节点                   │   │
│  │  • Quorum Queue：数据一致性                     │   │
│  │  • 跨机房部署：异地多活                         │   │
│  │  • 负载均衡：HAProxy/Nginx                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. 可靠性层面                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 消息持久化                                    │   │
│  │  • Publisher Confirm                            │   │
│  │  • Consumer 手动 ACK                            │   │
│  │  • 死信队列                                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. 监控层面                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • Prometheus + Grafana                         │   │
│  │  • 关键指标：队列深度、延迟、错误率             │   │
│  │  • 告警规则：积压、内存、分区                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  4. 业务层面                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • 幂等性设计                                    │   │
│  │  • 重试机制                                      │   │
│  │  • 降级策略                                      │   │
│  │  • 熔断限流                                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 7. 如何处理消息消费失败？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              消息消费失败处理策略                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                                                  │   │
│  │  消费失败                                        │   │
│  │      │                                           │   │
│  │      ▼                                           │   │
│  │  ┌─────────────────┐                            │   │
│  │  │ 判断错误类型     │                            │   │
│  │  └────────┬────────┘                            │   │
│  │           │                                      │   │
│  │     ┌─────┴─────┐                               │   │
│  │     │           │                               │   │
│  │     ▼           ▼                               │   │
│  │  临时错误    永久错误                            │   │
│  │     │           │                               │   │
│  │     ▼           ▼                               │   │
│  │  重试机制    死信队列                            │   │
│  │     │           │                               │   │
│  │     ▼           ▼                               │   │
│  │  成功/失败   人工处理                            │   │
│  │                                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**代码示例：**

```python
MAX_RETRIES = 3

def smart_consumer(ch, method, properties, body):
    message = json.loads(body)
    retry_count = properties.headers.get('x-retry-count', 0) if properties.headers else 0
    
    try:
        # 业务处理
        process_message(message)
        
        # 成功确认
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except TemporaryError as e:
        # 临时性错误：重试
        if retry_count < MAX_RETRIES:
            # 发送到延迟重试队列
            send_to_retry_queue(body, retry_count + 1)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.warning(f"重试 {retry_count + 1}/{MAX_RETRIES}: {e}")
        else:
            # 超过重试次数，进入死信队列
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            logger.error("超过最大重试次数，进入死信队列")
    
    except PermanentError as e:
        # 永久性错误：直接进入死信队列
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        logger.error(f"永久性错误，进入死信队列: {e}")
    
    except Exception as e:
        # 未知错误：记录日志，进入死信队列
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        logger.exception(f"未知错误: {e}")
```

---

### 8. RabbitMQ 的最佳实践有哪些？

**答案：**

```
┌─────────────────────────────────────────────────────────┐
│              RabbitMQ 最佳实践                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 生产者                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ✅ 使用异步 Confirm 替代事务                    │   │
│  │  ✅ 合理设置消息 TTL                             │   │
│  │  ✅ 关键消息持久化                               │   │
│  │  ✅ 连接复用，多 Channel                         │   │
│  │  ✅ 设置合理的 heartbeat                         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  2. Broker                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ✅ 使用 Quorum Queue 替代镜像队列               │   │
│  │  ✅ 合理设置内存和磁盘告警                       │   │
│  │  ✅ 监控关键指标                                 │   │
│  │  ✅ 定期备份配置                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  3. 消费者                                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ✅ 手动 ACK，避免 auto_ack                     │   │
│  │  ✅ 合理设置 prefetch_count                     │   │
│  │  ✅ 实现幂等性                                   │   │
│  │  ✅ 异常处理完善                                 │   │
│  │  ✅ 监控消费延迟                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  4. 运维                                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ✅ 至少 3 节点集群                              │   │
│  │  ✅ 监控告警完善                                 │   │
│  │  ✅ 制定故障恢复预案                             │   │
│  │  ✅ 定期演练                                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 参考资料

- [RabbitMQ 官方文档](https://www.rabbitmq.com/documentation.html)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
- [AMQP 0-9-1 协议规范](https://www.rabbitmq.com/resources/specs/amqp0-9-1.pdf)
- [RabbitMQ Quorum Queues](https://www.rabbitmq.com/quorum-queues.html)
