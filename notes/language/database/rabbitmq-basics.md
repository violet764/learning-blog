# RabbitMQ 基础与实践

RabbitMQ 是一个开源的消息代理软件，实现了高级消息队列协议（AMQP）。它以其可靠性、灵活的路由机制和丰富的客户端库支持而闻名，广泛应用于分布式系统、微服务架构和异步任务处理场景。

## 概述

RabbitMQ 最初由 Rabbit Technologies 开发，现由 VMware（Pivotal）维护。它支持多种消息协议，其中 AMQP 0-9-1 是核心协议。RabbitMQ 的设计理念是**"智能路由、简单消费"**，消息在到达队列前经过复杂的路由逻辑，消费者只需从队列中获取消息。

### 核心优势

| 特性 | 说明 |
|------|------|
| 🔄 **灵活路由** | 通过交换机和路由键实现复杂的消息分发 |
| 🛡️ **消息可靠** | 支持消息确认、持久化和事务 |
| 🌐 **多协议支持** | AMQP、STOMP、MQTT、HTTP |
| 🔌 **丰富客户端** | 支持 Java、Python、Go、.NET、JavaScript 等 |
| 📊 **可视化管理** | 内置 Web 管理界面 |
| 🔧 **插件生态** | 延迟消息、优先级队列、集群联邦等 |

## RabbitMQ 3.x 新特性

RabbitMQ 3.x 版本引入了许多重要改进，2024-2025 年推荐使用 **3.12+** 或 **4.x** 版本。

### 3.12+ 版本重要更新

```yaml
# 版本特性对比
3.12:
  - Quorum Queue 成为默认队列类型
  - 改进的内存管理算法
  - Khepri 元数据存储（实验性）
  - MQTTv5 支持
  
3.13:
  - Super Streams（超级流）
  - 改进的 Streams 性能
  - OAuth 2.0 增强
  
4.0:
  - Khepri 元数据存储默认启用
  - 经典队列 v2
  - 移除传统镜像队列支持
```

### Quorum Queue vs Classic Queue

```python
# Python: 声明 Quorum Queue（推荐）
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Quorum Queue - 高可用、持久化、Raft 共识
channel.queue_declare(
    queue='orders.quorum',
    durable=True,
    arguments={
        'x-queue-type': 'quorum'  # 3.12+ 推荐的队列类型
    }
)

# Classic Queue - 传统队列，适合低延迟场景
channel.queue_declare(
    queue='notifications.classic',
    durable=True,
    arguments={
        'x-queue-type': 'classic'
    }
)
```

::: tip 📌 选择建议
- **Quorum Queue**: 生产环境首选，数据安全优先
- **Classic Queue**: 开发测试、低延迟要求的非关键数据
- **Stream**: 大数据量、消费者需要重新消费历史消息
:::

## AMQP 协议基础

AMQP（Advanced Message Queuing Protocol）是应用层协议，定义了消息的格式和传输规则。

### 协议层次

```
┌─────────────────────────────────────┐
│           应用层 (Application)        │
├─────────────────────────────────────┤
│         AMQP 协议层                   │
│  ┌─────────────────────────────┐    │
│  │  命令层 (Class/Method)       │    │
│  ├─────────────────────────────┤    │
│  │  内容层 (Header/Body)        │    │
│  └─────────────────────────────┘    │
├─────────────────────────────────────┤
│         传输层 (TCP/SSL)             │
└─────────────────────────────────────┘
```

### 核心概念模型

AMQP 模型包含以下核心组件：

```
                    ┌──────────────┐
                    │   Producer   │
                    └──────┬───────┘
                           │ 消息 + Routing Key
                           ▼
                    ┌──────────────┐
                    │   Exchange   │ ◄── 交换机：路由消息
                    └──────┬───────┘
                           │ Binding
                           ▼
                    ┌──────────────┐
                    │    Queue     │ ◄── 队列：存储消息
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Consumer   │
                    └──────────────┘
```

## 核心概念详解

### Exchange（交换机）

交换机接收生产者发送的消息，并根据路由规则将消息推送到队列。

```python
# 声明交换机
channel.exchange_declare(
    exchange='orders.direct',      # 交换机名称
    exchange_type='direct',        # 类型
    durable=True,                  # 持久化
    auto_delete=False              # 不自动删除
)
```

### Queue（队列）

队列是存储消息的缓冲区，消费者从队列中获取消息。

```python
# 声明队列
channel.queue_declare(
    queue='order.created',
    durable=True,                  # 持久化
    exclusive=False,               # 非独占
    auto_delete=False,             # 不自动删除
    arguments={
        'x-message-ttl': 86400000,        # 消息 TTL：24小时
        'x-max-length': 10000,            # 最大消息数
        'x-overflow': 'reject-publish-dlx' # 溢出策略
    }
)
```

### Binding（绑定）

绑定定义了交换机和队列之间的关系，将路由规则关联起来。

```python
# 将队列绑定到交换机
channel.queue_bind(
    queue='order.created',
    exchange='orders.direct',
    routing_key='order.created'    # 路由键
)
```

### Routing Key（路由键）

路由键是消息的路由标识，交换机根据路由键决定消息的目标队列。

```python
# 发布消息时指定路由键
channel.basic_publish(
    exchange='orders.direct',
    routing_key='order.created',   # 路由键
    body=json.dumps({'order_id': '12345', 'amount': 99.99}),
    properties=pika.BasicProperties(
        delivery_mode=2,           # 持久化消息
        content_type='application/json'
    )
)
```

## 交换机类型

RabbitMQ 提供四种交换机类型，每种类型实现不同的路由策略。

### 1. Direct Exchange（直连交换机）

精确匹配路由键，适合点对点消息传递。

```python
# Direct Exchange 示例
channel.exchange_declare(exchange='logs.direct', exchange_type='direct')

# 创建队列
channel.queue_declare(queue='logs.error')
channel.queue_declare(queue='logs.info')

# 绑定队列（精确匹配）
channel.queue_bind(queue='logs.error', exchange='logs.direct', routing_key='error')
channel.queue_bind(queue='logs.info', exchange='logs.direct', routing_key='info')

# 发送消息
channel.basic_publish(exchange='logs.direct', routing_key='error', body='Error message')
# 只有 logs.error 队列会收到消息
```

```
          routing_key="error"
Producer ───────────────────────► [Exchange: logs.direct]
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
            [Queue: logs.error]                    [Queue: logs.info]
            routing_key="error" ✓                  routing_key="error" ✗
```

### 2. Fanout Exchange（扇出交换机）

广播消息到所有绑定的队列，忽略路由键。

```python
# Fanout Exchange 示例（发布/订阅模式）
channel.exchange_declare(exchange='notifications.fanout', exchange_type='fanout')

# 多个消费者队列
channel.queue_declare(queue='notification.email')
channel.queue_declare(queue='notification.sms')
channel.queue_declare(queue='notification.push')

# 绑定队列（无需路由键）
channel.queue_bind(queue='notification.email', exchange='notifications.fanout')
channel.queue_bind(queue='notification.sms', exchange='notifications.fanout')
channel.queue_bind(queue='notification.push', exchange='notifications.fanout')

# 发送广播消息
channel.basic_publish(
    exchange='notifications.fanout',
    routing_key='',  # 路由键被忽略
    body='System maintenance at 2:00 AM'
)
# 三个队列都会收到消息
```

### 3. Topic Exchange（主题交换机）

支持通配符匹配，适合复杂路由场景。

```python
# Topic Exchange 示例
channel.exchange_declare(exchange='events.topic', exchange_type='topic')

# 绑定模式
# * 匹配一个单词
# # 匹配零个或多个单词

channel.queue_bind(queue='all.orders', exchange='events.topic', routing_key='order.#')
channel.queue_bind(queue='order.paid', exchange='events.topic', routing_key='order.paid')
channel.queue_bind(queue='all.events', exchange='events.topic', routing_key='#')

# 发送消息
channel.basic_publish(exchange='events.topic', routing_key='order.created', body='...')
# all.orders 和 all.events 收到

channel.basic_publish(exchange='events.topic', routing_key='order.paid', body='...')
# all.orders、order.paid、all.events 都收到
```

```
路由键格式: <domain>.<event>.<sub_event>

order.created          → order.# ✓, # ✓
order.paid.success     → order.# ✓, # ✓  
user.registered        → # ✓
payment.refunded       → payment.# ✓, # ✓
```

### 4. Headers Exchange（头交换机）

基于消息头属性匹配，适用于多条件路由。

```python
# Headers Exchange 示例
channel.exchange_declare(exchange='router.headers', exchange_type='headers')

# 绑定队列时指定匹配条件
channel.queue_bind(
    queue='premium.orders',
    exchange='router.headers',
    routing_key='',  # 路由键被忽略
    arguments={
        'x-match': 'all',      # 'all' 或 'any'
        'vip': 'true',
        'region': 'cn'
    }
)

# 发送消息时携带 headers
channel.basic_publish(
    exchange='router.headers',
    routing_key='',
    body='Premium order content',
    properties=pika.BasicProperties(
        headers={
            'vip': 'true',
            'region': 'cn'
        }
    )
)
# 消息会被路由到 premium.orders 队列
```

### 交换机类型对比

| 类型 | 路由方式 | 使用场景 | 性能 |
|------|---------|----------|------|
| **Direct** | 精确匹配路由键 | 点对点、RPC | ⭐⭐⭐⭐⭐ |
| **Fanout** | 广播到所有队列 | 发布/订阅、广播通知 | ⭐⭐⭐⭐ |
| **Topic** | 通配符匹配 | 事件驱动、日志收集 | ⭐⭐⭐ |
| **Headers** | 头属性匹配 | 多条件路由 | ⭐⭐ |

## 消息确认机制

消息确认是 RabbitMQ 保证**"至少一次投递"**（At-Least-Once Delivery）的关键机制。

### ACK（确认）

消费者成功处理消息后发送确认。

```python
# 消费者确认模式
def callback(ch, method, properties, body):
    try:
        # 处理消息
        process_message(body)
        # 手动确认（推荐）
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Message acknowledged: {method.delivery_tag}")
    except Exception as e:
        # 处理失败，重新入队
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

# 开启手动确认模式
channel.basic_consume(
    queue='orders',
    on_message_callback=callback,
    auto_ack=False  # 关键：关闭自动确认
)
```

### NACK（否定确认）

消费者处理失败时，可以选择重新入队或拒绝。

```python
# NACK 示例
def callback(ch, method, properties, body):
    try:
        process_message(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except RetryableError as e:
        # 可重试错误，重新入队
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    except FatalError as e:
        # 不可恢复错误，发送到死信队列
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
```

### Reject（拒绝）

拒绝单条消息，与 NACK 类似但只能处理单条。

```python
# Reject 示例
def callback(ch, method, properties, body):
    if is_invalid(body):
        # 拒绝消息，不重新入队（进入死信队列）
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    else:
        process_and_ack(ch, method, body)
```

### 确认机制对比

```python
# 三种确认方式对比
# ┌─────────┬──────────────┬─────────────────┬────────────────┐
# │   方式   │    作用范围   │     requeue     │      场景       │
# ├─────────┼──────────────┼─────────────────┼────────────────┤
# │   ACK   │ 单条/批量     │  N/A            │ 处理成功        │
# │   NACK  │ 单条/批量     │  可选择         │ 处理失败        │
# │ Reject  │ 仅单条        │  可选择         │ 明确拒绝        │
# └─────────┴──────────────┴─────────────────┴────────────────┘

# 批量确认（提高性能）
def callback(ch, method, properties, body):
    process_message(body)
    # 消息序号连续时，可以批量确认
    # 确认当前消息及之前所有未确认消息
    ch.basic_ack(delivery_tag=method.delivery_tag, multiple=True)
```

## 消费者 Prefetch 配置

Prefetch 控制消费者未确认消息的最大数量，是实现**公平分发**的关键。

### 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefetch 机制示意                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  prefetch_count = 2                                         │
│                                                             │
│  Consumer A: [msg1(未确认)] [msg2(未确认)] ← 达到上限，暂停   │
│  Consumer B: [msg3(未确认)] [msg4(未确认)] ← 达到上限，暂停   │
│  Consumer C: [msg5(未确认)] ← 继续接收                       │
│                                                             │
│  当 Consumer A 确认 msg1 后，可以接收新消息                   │
└─────────────────────────────────────────────────────────────┘
```

### 配置示例

```python
# Prefetch 配置
# 设置每个消费者最多同时处理的消息数
channel.basic_qos(prefetch_count=1)  # 公平分发

# 消费者
channel.basic_consume(queue='tasks', on_message_callback=callback, auto_ack=False)

# 完整消费者示例
def start_consumer():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='localhost',
            heartbeat=30,  # 心跳间隔
            blocked_connection_timeout=300
        )
    )
    channel = connection.channel()
    
    # 关键：设置 prefetch
    # prefetch_count=1 表示公平分发，每次只处理一条消息
    # prefetch_count=10 表示可以同时处理10条消息，提高吞吐量
    channel.basic_qos(prefetch_count=1)
    
    def callback(ch, method, properties, body):
        try:
            process_task(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"处理失败: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    channel.basic_consume(queue='tasks', on_message_callback=callback, auto_ack=False)
    channel.start_consuming()
```

### Prefetch 最佳实践

```python
# Prefetch 值设置建议
# ┌─────────────────┬────────────┬───────────────────────────────┐
# │     场景         │ 推荐值      │ 说明                           │
# ├─────────────────┼────────────┼───────────────────────────────┤
# │ 任务处理时间差异大 │ 1          │ 公平分发，避免快消费者空闲       │
# │ 任务处理时间稳定  │ 5-10       │ 提高吞吐量                      │
# │ 高吞吐量场景     │ 10-50      │ 批量处理优化                    │
# │ 消息处理很重     │ 1          │ 防止消费者过载                  │
# └─────────────────┴────────────┴───────────────────────────────┘

# 全局 Prefetch vs 通道 Prefetch
channel.basic_qos(
    prefetch_count=1,
    global_qos=False  # False: 每个消费者独立计数
                       # True: 整个通道共享限制（RabbitMQ 不完全支持）
)
```

## 死信队列（DLX）设计

死信队列（Dead Letter Exchange）用于存储无法被正常消费的消息。

### 死信产生条件

```yaml
死信触发条件:
  1. 消息被拒绝 (basic.reject/basic_nack) 且 requeue=false
  2. 消息 TTL 过期
  3. 队列达到最大长度
```

### 死信队列配置

```python
# 死信队列完整配置
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 1. 声明死信交换机
channel.exchange_declare(exchange='dlx.exchange', exchange_type='direct', durable=True)

# 2. 声明死信队列
channel.queue_declare(queue='dlx.queue', durable=True)
channel.queue_bind(queue='dlx.queue', exchange='dlx.exchange', routing_key='dlx')

# 3. 声明业务队列，配置死信交换机
channel.queue_declare(
    queue='business.queue',
    durable=True,
    arguments={
        'x-dead-letter-exchange': 'dlx.exchange',      # 死信交换机
        'x-dead-letter-routing-key': 'dlx',            # 死信路由键
        'x-message-ttl': 60000,                        # 消息 TTL: 60秒
        'x-max-length': 10000                          # 最大长度
    }
)

# 4. 声明业务交换机并绑定
channel.exchange_declare(exchange='business.exchange', exchange_type='direct', durable=True)
channel.queue_bind(queue='business.queue', exchange='business.exchange', routing_key='business')
```

### 死信队列消费处理

```python
# 死信队列消费者 - 记录失败消息
def dlx_callback(ch, method, properties, body):
    # 获取原始消息信息
    original_exchange = properties.headers.get('x-death', [{}])[0].get('exchange')
    original_queue = properties.headers.get('x-death', [{}])[0].get('queue')
    death_reason = properties.headers.get('x-death', [{}])[0].get('reason')
    death_time = properties.headers.get('x-death', [{}])[0].get('time')
    death_count = properties.headers.get('x-death', [{}])[0].get('count', 1)
    
    # 记录到数据库或发送告警
    log_failed_message(
        original_queue=original_queue,
        reason=death_reason,
        message_body=body.decode(),
        death_count=death_count
    )
    
    # 确认死信消息
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    # 可选：超过重试次数后，发送到人工处理队列
    if death_count > 3:
        send_to_manual_queue(body)
```

### 死信队列架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        死信队列架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Producer                                                       │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────┐                                            │
│  │ Business Exchange│                                           │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────┐                            │
│  │     Business Queue              │                            │
│  │  ┌───────────────────────────┐  │                            │
│  │  │ x-dead-letter-exchange    │──┐                           │
│  │  │ x-message-ttl: 60000      │  │                           │
│  │  │ x-max-length: 10000       │  │                           │
│  │  └───────────────────────────┘  │                           │
│  └─────────────────────────────────┘                           │
│           │ 消费失败/TTL过期/队列满                               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   DLX Exchange  │◄──────────────────────┘                   │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │   DLX Queue     │────►│  告警/重试/人工  │                    │
│  └─────────────────┘     └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 延迟队列实现

RabbitMQ 本身不支持延迟队列，但可以通过插件或 TTL + DLX 组合实现。

### 方案一：rabbitmq_delayed_message_exchange 插件

```bash
# 安装延迟消息插件
rabbitmq-plugins enable rabbitmq_delayed_message_exchange

# 重启 RabbitMQ
systemctl restart rabbitmq-server
```

```python
# 使用延迟消息插件
channel.exchange_declare(
    exchange='delayed.exchange',
    exchange_type='x-delayed-message',  # 延迟交换机类型
    durable=True,
    arguments={
        'x-delayed-type': 'direct'      # 底层交换机类型
    }
)

# 声明队列
channel.queue_declare(queue='delayed.queue', durable=True)
channel.queue_bind(queue='delayed.queue', exchange='delayed.exchange', routing_key='delayed')

# 发送延迟消息（延迟 30 秒）
channel.basic_publish(
    exchange='delayed.exchange',
    routing_key='delayed',
    body='Delayed message content',
    properties=pika.BasicProperties(
        headers={'x-delay': 30000}  # 延迟时间：毫秒
    )
)
```

### 方案二：TTL + DLX（无需插件）

```python
# TTL + DLX 实现延迟队列
# 原理：消息在 TTL 队列过期后，转发到目标队列

# 1. 声明目标队列（实际处理消息）
channel.queue_declare(queue='process.queue', durable=True)

# 2. 声明目标交换机
channel.exchange_declare(exchange='process.exchange', exchange_type='direct', durable=True)
channel.queue_bind(queue='process.queue', exchange='process.exchange', routing_key='process')

# 3. 声明延迟队列（消息在这里等待过期）
channel.queue_declare(
    queue='delay.30s.queue',
    durable=True,
    arguments={
        'x-message-ttl': 30000,                    # 30秒过期
        'x-dead-letter-exchange': 'process.exchange',  # 过期后转发
        'x-dead-letter-routing-key': 'process'          # 路由键
    }
)

# 4. 发送消息到延迟队列
channel.basic_publish(
    exchange='',
    routing_key='delay.30s.queue',
    body='Message will be processed after 30 seconds'
)
```

### 多级延迟队列

```python
# 多级延迟队列 - 不同延迟时间
delay_levels = [
    ('delay.5s.queue', 5000),
    ('delay.30s.queue', 30000),
    ('delay.1m.queue', 60000),
    ('delay.5m.queue', 300000),
    ('delay.1h.queue', 3600000),
]

# 创建延迟队列
for queue_name, ttl in delay_levels:
    channel.queue_declare(
        queue=queue_name,
        durable=True,
        arguments={
            'x-message-ttl': ttl,
            'x-dead-letter-exchange': 'process.exchange',
            'x-dead-letter-routing-key': 'process'
        }
    )

# 根据需要选择延迟级别发送消息
def send_delayed_message(body, delay_seconds):
    """发送延迟消息"""
    # 找到合适的延迟队列
    if delay_seconds <= 5:
        routing_key = 'delay.5s.queue'
    elif delay_seconds <= 30:
        routing_key = 'delay.30s.queue'
    elif delay_seconds <= 60:
        routing_key = 'delay.1m.queue'
    elif delay_seconds <= 300:
        routing_key = 'delay.5m.queue'
    else:
        routing_key = 'delay.1h.queue'
    
    channel.basic_publish(exchange='', routing_key=routing_key, body=body)
```

## 镜像队列与 Quorum Queue

### 经典镜像队列（Classic Mirrored Queue）

::: warning ⚠️ 注意
RabbitMQ 4.0 已移除经典镜像队列支持，建议使用 Quorum Queue。
:::

```python
# 镜像队列策略（3.x 版本）
# 通过管理界面或命令配置
"""
rabbitmqctl set_policy ha-orders "^orders\." \
  '{"ha-mode":"exactly","ha-params":2,"ha-sync-mode":"automatic"}'
"""

# ha-mode 选项:
# - all: 镜像到所有节点
# - exactly: 镜像到指定数量节点
# - nodes: 镜像到指定节点
```

### Quorum Queue（仲裁队列）

Quorum Queue 是 RabbitMQ 3.8+ 引入的现代队列类型，使用 Raft 协议保证数据一致性。

```python
# Quorum Queue 声明
channel.queue_declare(
    queue='critical.orders',
    durable=True,
    arguments={
        'x-queue-type': 'quorum',
        'x-quorum-initial-group-size': 3,  # 初始仲裁组大小
        'x-delivery-limit': 10,            # 最大投递次数
        'x-dead-letter-strategy': 'at-least-once'  # 死信策略
    }
)
```

### Quorum Queue 特性

```yaml
Quorum Queue 优势:
  数据安全:
    - 基于 Raft 共识协议
    - 多节点数据复制
    - 节点故障自动恢复
    
  性能特点:
    - 写入需多数节点确认
    - 吞吐量略低于经典队列
    - 延迟更稳定
    
  适用场景:
    - 数据不能丢失的关键业务
    - 订单、支付、消息通知
    - 需要Exactly-Once语义
```

### Quorum Queue vs Classic Queue

```python
# 性能对比配置示例
import pika
import time

def benchmark_queues():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # Classic Queue - 低延迟
    channel.queue_declare(queue='bench.classic', durable=True, 
                          arguments={'x-queue-type': 'classic'})
    
    # Quorum Queue - 高可靠
    channel.queue_declare(queue='bench.quorum', durable=True,
                          arguments={'x-queue-type': 'quorum'})
    
    # 测试发送性能
    test_messages = 10000
    
    # Classic Queue 发送测试
    start = time.time()
    for i in range(test_messages):
        channel.basic_publish(exchange='', routing_key='bench.classic', body='test')
    classic_publish_time = time.time() - start
    
    # Quorum Queue 发送测试
    start = time.time()
    for i in range(test_messages):
        channel.basic_publish(exchange='', routing_key='bench.quorum', body='test')
    quorum_publish_time = time.time() - start
    
    print(f"Classic Queue: {classic_publish_time:.2f}s")
    print(f"Quorum Queue: {quorum_publish_time:.2f}s")
```

## 消息持久化策略

消息持久化确保 RabbitMQ 重启后消息不丢失。

### 三层持久化

```yaml
消息持久化需要三层保障:
  1. Exchange 持久化 - durable=True
  2. Queue 持久化 - durable=True  
  3. Message 持久化 - delivery_mode=2
```

### 持久化配置

```python
# 完整持久化配置
def publish_persistent_message():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='localhost',
            port=5672,
            credentials=pika.PlainCredentials('admin', 'password')
        )
    )
    channel = connection.channel()
    
    # 1. 声明持久化交换机
    channel.exchange_declare(
        exchange='orders.persistent',
        exchange_type='direct',
        durable=True  # 关键：交换机持久化
    )
    
    # 2. 声明持久化队列
    channel.queue_declare(
        queue='orders.persistent',
        durable=True,  # 关键：队列持久化
        arguments={
            'x-queue-type': 'quorum'  # Quorum Queue 自动持久化
        }
    )
    
    channel.queue_bind(
        queue='orders.persistent',
        exchange='orders.persistent',
        routing_key='order'
    )
    
    # 3. 发送持久化消息
    channel.basic_publish(
        exchange='orders.persistent',
        routing_key='order',
        body=json.dumps({'order_id': '12345'}),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 关键：消息持久化 (PERSISTENT_DELIVERY_MODE)
            content_type='application/json',
            content_encoding='utf-8',
            headers={'version': '1.0'},
            timestamp=int(time.time())
        )
    )
    
    print("持久化消息发送成功")
```

### 持久化性能权衡

```python
# 持久化配置对比
# ┌───────────────────┬────────────┬────────────┬─────────────┐
# │       配置         │   安全性    │   性能      │   适用场景   │
# ├───────────────────┼────────────┼────────────┼─────────────┤
# │ 非持久化           │ 低         │ 最高        │ 缓存、日志   │
# │ 仅队列持久化       │ 中         │ 高          │ 可重放数据   │
# │ 全持久化           │ 高         │ 中          │ 关键业务     │
# │ Quorum Queue      │ 最高       │ 中低        │ 金融、交易   │
# └───────────────────┴────────────┴────────────┴─────────────┘

# 消息大小建议
# ⚠️ 单条消息建议 < 1MB
# 大消息应存储在对象存储，RabbitMQ 只传递引用

def send_large_file_reference(file_url, metadata):
    """大文件处理：只传递引用"""
    channel.basic_publish(
        exchange='files.exchange',
        routing_key='file.process',
        body=json.dumps({
            'file_url': file_url,  # 文件存储地址
            'size': metadata['size'],
            'checksum': metadata['checksum']
        }),
        properties=pika.BasicProperties(
            delivery_mode=2,
            headers={'file-type': 'reference'}
        )
    )
```

## 现代最佳实践

### 1. 长连接消费者模式

```python
# 推荐：长连接消费者
import pika
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliableConsumer:
    """可靠的消费者，支持自动重连"""
    
    def __init__(self, amqp_url, queue_name):
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.should_reconnect = True
        
    def connect(self):
        """建立连接"""
        while self.should_reconnect:
            try:
                self.connection = pika.BlockingConnection(
                    pika.URLParameters(self.amqp_url)
                )
                self.channel = self.connection.channel()
                logger.info("连接成功")
                return True
            except Exception as e:
                logger.error(f"连接失败: {e}，5秒后重试")
                time.sleep(5)
        return False
    
    def on_message(self, ch, method, properties, body):
        """消息处理回调"""
        try:
            self.process_message(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"处理失败: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def process_message(self, body):
        """业务处理逻辑"""
        # 实际业务代码
        pass
    
    def run(self):
        """启动消费者"""
        while self.should_reconnect:
            try:
                if self.connect():
                    self.channel.basic_qos(prefetch_count=1)
                    self.channel.basic_consume(
                        queue=self.queue_name,
                        on_message_callback=self.on_message,
                        auto_ack=False
                    )
                    logger.info("开始消费...")
                    self.channel.start_consuming()
            except pika.exceptions.AMQPConnectionError:
                logger.warning("连接断开，尝试重连...")
            except KeyboardInterrupt:
                logger.info("用户中断")
                self.should_reconnect = False
            finally:
                if self.connection and not self.connection.is_closed:
                    self.connection.close()

# 使用示例
if __name__ == '__main__':
    consumer = ReliableConsumer(
        amqp_url='amqp://admin:password@localhost:5672/%2F',
        queue_name='orders'
    )
    consumer.run()
```

### 2. 错误处理与重试机制

```python
# 智能重试机制
import json
from datetime import datetime

class RetryHandler:
    """消息重试处理器"""
    
    MAX_RETRIES = 3
    RETRY_DELAYS = [1000, 5000, 30000]  # 1s, 5s, 30s
    
    def __init__(self, channel):
        self.channel = channel
    
    def handle_with_retry(self, ch, method, properties, body):
        """带重试的消息处理"""
        # 获取重试次数
        retry_count = self.get_retry_count(properties)
        
        try:
            # 业务处理
            self.process_business_logic(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except RetryableError as e:
            if retry_count < self.MAX_RETRIES:
                # 延迟重试
                self.schedule_retry(body, retry_count, properties)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # 超过重试次数，发送到死信队列
                logger.error(f"超过最大重试次数: {retry_count}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                
        except FatalError as e:
            logger.error(f"致命错误: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def get_retry_count(self, properties):
        """获取当前重试次数"""
        if properties.headers and 'x-retry-count' in properties.headers:
            return properties.headers['x-retry-count']
        return 0
    
    def schedule_retry(self, body, retry_count, old_properties):
        """调度重试"""
        # 获取延迟时间
        delay = self.RETRY_DELAYS[retry_count] if retry_count < len(self.RETRY_DELAYS) else self.RETRY_DELAYS[-1]
        
        # 发送到延迟队列
        self.channel.basic_publish(
            exchange='retry.exchange',
            routing_key='retry',
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                headers={
                    'x-retry-count': retry_count + 1,
                    'x-delay': delay,
                    'x-first-failure-time': old_properties.headers.get('x-first-failure-time', datetime.now().isoformat())
                }
            )
        )
    
    def process_business_logic(self, body):
        """实际业务逻辑"""
        data = json.loads(body)
        # 业务处理代码...
        pass
```

### 3. 连接池管理

```python
# 连接池实现（使用 pika_pool）
import pika
from pika_pool import Pool

# 配置连接池
pool = Pool(
    create=lambda: pika.BlockingConnection(
        pika.URLParameters('amqp://admin:password@localhost:5672/%2F')
    ),
    max_size=10,        # 最大连接数
    max_overflow=5,     # 最大溢出连接
    timeout=10,         # 获取连接超时
    recycle=3600,       # 连接回收时间
    stale=60,           # 连接空闲过期时间
)

# 使用连接池发布消息
def publish_with_pool(exchange, routing_key, body):
    with pool.acquire() as connection:
        channel = connection.channel()
        channel.confirm_delivery()  # 开启发布确认
        
        try:
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            return True
        except pika.exceptions.UnroutableError:
            logger.error("消息无法路由")
            return False
```

### 4. 生产部署检查清单

```yaml
# 生产部署检查清单

## 基础配置
- [ ] 集群部署（至少 3 节点）
- [ ] 内存阈值设置（vm_memory_high_watermark: 0.6）
- [ ] 磁盘空间阈值（disk_free_limit: 10GB）
- [ ] 文件描述符限制（ulimit -n 65536）

## 安全配置
- [ ] 启用 TLS 加密
- [ ] 配置用户权限（最小权限原则）
- [ ] 禁用 guest 用户远程访问
- [ ] 启用 Management 插件认证

## 高可用配置
- [ ] 使用 Quorum Queue 替代 Classic Queue
- [ ] 配置镜像策略（3.x 版本）
- [ ] 设置队列 TTL 和最大长度
- [ ] 配置死信队列

## 性能优化
- [ ] 开启发布确认（Publisher Confirms）
- [ ] 合理设置 Prefetch Count
- [ ] 消息批量处理
- [ ] 连接池配置

## 监控告警
- [ ] 队列积压告警
- [ ] 消费者连接数监控
- [ ] 节点资源使用监控
- [ ] 网络分区检测

## 运维
- [ ] 定期备份消息
- [ ] 日志收集与分析
- [ ] 升级测试流程
- [ ] 灾难恢复预案
```

### 5. 监控指标

```python
# 监控关键指标
import requests

def get_rabbitmq_metrics():
    """获取 RabbitMQ 监控指标"""
    api_url = 'http://localhost:15672/api'
    auth = ('admin', 'password')
    
    # 队列状态
    queues = requests.get(f'{api_url}/queues', auth=auth).json()
    
    metrics = {
        'total_messages': sum(q['messages'] for q in queues),
        'unacked_messages': sum(q['messages_unacknowledged'] for q in queues),
        'queue_count': len(queues),
        'consumer_count': sum(q.get('consumers', 0) for q in queues),
    }
    
    # 关键告警指标
    alerts = []
    for queue in queues:
        # 队列积压告警
        if queue['messages'] > 10000:
            alerts.append(f"队列 {queue['name']} 积压 {queue['messages']} 条消息")
        
        # 无消费者告警
        if queue.get('consumers', 0) == 0 and queue['messages'] > 0:
            alerts.append(f"队列 {queue['name']} 无消费者")
    
    return metrics, alerts
```

## 常见问题与解决方案

### 1. 消息积压处理

```python
# 消息积压临时处理方案
def handle_message_backlog(channel, queue_name):
    """处理消息积压"""
    # 1. 检查积压情况
    queue_info = channel.queue_declare(queue=queue_name, passive=True)
    message_count = queue_info.method.message_count
    
    print(f"当前积压消息数: {message_count}")
    
    if message_count > 10000:
        # 2. 临时增加消费者
        print("建议增加消费者实例")
        
        # 3. 批量消费处理
        def batch_callback(ch, method, properties, body):
            # 批量处理逻辑
            messages = json.loads(body) if isinstance(body, str) else [body]
            for msg in messages:
                process_message(msg)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        
        # 提高预取数量
        channel.basic_qos(prefetch_count=100)
```

### 2. 消息重复消费

```python
# 幂等性处理
import hashlib

def idempotent_consume(ch, method, properties, body):
    """幂等消费处理"""
    message_id = properties.message_id or hashlib.md5(body).hexdigest()
    
    # 检查是否已处理
    if is_message_processed(message_id):
        logger.info(f"消息 {message_id} 已处理，跳过")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
    
    try:
        # 处理消息
        process_message(body)
        
        # 标记已处理
        mark_message_processed(message_id)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
```

### 3. 连接泄漏检测

```python
# 连接健康检查
def check_connection_health():
    """检查连接健康状态"""
    api_url = 'http://localhost:15672/api'
    auth = ('admin', 'password')
    
    connections = requests.get(f'{api_url}/connections', auth=auth).json()
    
    for conn in connections:
        # 检查连接空闲时间
        if conn.get('idle_since'):
            idle_time = time.time() - parse_time(conn['idle_since'])
            if idle_time > 3600:  # 空闲超过1小时
                logger.warning(f"连接 {conn['name']} 空闲 {idle_time}s")
        
        # 检查通道数
        if conn.get('channels', 0) > 100:
            logger.warning(f"连接 {conn['name']} 通道数过多: {conn['channels']}")
```

## 参考资料

- [RabbitMQ 官方文档](https://www.rabbitmq.com/docs/)
- [AMQP 0-9-1 协议规范](https://www.rabbitmq.com/resources/specs/amqp0-9-1.pdf)
- [RabbitMQ Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)
- [Pika Python 客户端](https://pika.readthedocs.io/)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
