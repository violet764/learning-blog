# Kafka 面试题

本文整理了 Kafka 消息队列的高频面试题，涵盖基础概念、高性能原理、消息可靠性、消费者机制、架构设计和实战场景六大模块。

---

## 基础篇

### 1. Kafka 是什么？用于什么场景？

**答案：**

Kafka 是由 LinkedIn 开发、后贡献给 Apache 的**分布式流处理平台**，核心定位是**高吞吐量的分布式发布订阅消息系统**。

**核心特性：**
- 🚀 **高吞吐量**：单机每秒处理百万级消息
- 📦 **持久化存储**：消息持久化到磁盘，支持回溯消费
- 🔄 **分布式架构**：天然支持水平扩展
- ⏱️ **低延迟**：毫秒级消息传递

**主要应用场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                     Kafka 应用场景                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│   消息队列       │    日志收集      │    流处理               │
│  系统解耦        │   ELK/LGTM      │   Flink/Spark           │
│  削峰填谷        │   日志聚合       │   实时计算               │
├─────────────────┼─────────────────┼─────────────────────────┤
│   用户活动追踪    │    事件溯源      │    大数据管道            │
│   点击流分析     │   CQRS 架构      │   数据湖入湖             │
│   用户行为采集    │   事件驱动架构    │   ETL 数据同步           │
└─────────────────┴─────────────────┴─────────────────────────┘
```

**代码示例 - 生产者发送消息：**

```java
// 创建生产者配置
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("acks", "all"); // 确保消息可靠性

// 创建生产者实例
Producer<String, String> producer = new KafkaProducer<>(props);

// 发送消息
producer.send(new ProducerRecord<>("my-topic", "key1", "Hello Kafka"), 
    (metadata, exception) -> {
        if (exception == null) {
            System.out.println("发送成功: partition=" + metadata.partition() + 
                             ", offset=" + metadata.offset());
        } else {
            System.err.println("发送失败: " + exception.getMessage());
        }
    });

producer.close();
```

**代码示例 - 消费者订阅消息：**

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "my-group");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("auto.offset.reset", "earliest");
props.put("enable.auto.commit", "false"); // 手动提交 offset

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("my-topic"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        System.out.printf("partition=%d, offset=%d, key=%s, value=%s%n",
            record.partition(), record.offset(), record.key(), record.value());
    }
    consumer.commitSync(); // 手动同步提交 offset
}
```

**追问：Kafka 和传统消息队列（如 RabbitMQ）的本质区别是什么？**

**追问答案：**

核心区别在于**设计理念**：

| 维度 | Kafka | RabbitMQ |
|------|-------|----------|
| 消息保留 | 持久化保留，支持回放 | 消费后删除 |
| 吞吐量 | 百万级/秒 | 万级/秒 |
| 消费模式 | 拉取模式（Pull） | 推送模式（Push） |
| 顺序保证 | 分区内严格有序 | 单队列有序 |
| 适用场景 | 大数据、日志、流处理 | 传统业务消息 |

---

### 2. Kafka 和 RabbitMQ 的区别？怎么选型？

**答案：**

**架构对比图：**

```
Kafka 架构：                          RabbitMQ 架构：
┌─────────────────────────┐          ┌─────────────────────────┐
│      Producer           │          │      Producer           │
└───────────┬─────────────┘          └───────────┬─────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│    Broker Cluster       │          │      Exchange           │
│  ┌─────┐ ┌─────┐        │          │   (路由分发)             │
│  │P0   │ │P1   │ Topic  │          └───────────┬─────────────┘
│  │rep  │ │rep  │        │                      │
│  └─────┘ └─────┘        │                      ▼
└───────────┬─────────────┘          ┌─────────────────────────┐
            │                        │       Queue             │
            ▼                        └───────────┬─────────────┘
┌─────────────────────────┐                      │
│    Consumer Group       │                      ▼
│  ┌─────┐ ┌─────┐        │          ┌─────────────────────────┐
│  │C1   │ │C2   │        │          │      Consumer           │
│  └─────┘ └─────┘        │          └─────────────────────────┘
└─────────────────────────┘
```

**详细对比：**

| 对比维度 | Kafka | RabbitMQ |
|---------|-------|----------|
| **吞吐量** | 百万级/秒 | 万级/秒（约5万） |
| **延迟** | 毫秒级（约5-10ms） | 微秒级（约0.1ms） |
| **消息保留** | 基于时间/大小策略保留 | 消费后立即删除 |
| **消费模式** | Pull 模式 | Push 模式 |
| **协议支持** | 自定义协议 | AMQP、MQTT、STOMP |
| **事务支持** | 有限的事务支持 | 完整的事务支持 |
| **消息优先级** | 不支持 | 支持 |
| **延迟消息** | 不原生支持 | 原生支持 |
| **消息轨迹** | 需要额外实现 | 原生支持 |
| **管理界面** | 需要第三方工具 | 内置 Web 管理界面 |

**选型决策树：**

```
                    开始选型
                       │
          ┌────────────┴────────────┐
          │                         │
    高吞吐量需求？              复杂路由需求？
    (>10万/秒)                 (Topic Exchange等)
          │                         │
      是  │                     是  │
          ▼                         ▼
      ┌───────┐                ┌─────────┐
      │Kafka  │                │RabbitMQ │
      └───────┘                └─────────┘
          │                         │
          │                         │
    消息回溯需求？              延迟消息需求？
          │                         │
      是  │                     是  │
          ▼                         ▼
      ┌───────┐                ┌─────────┐
      │Kafka  │                │RabbitMQ │
      └───────┘                └─────────┘
```

**追问：如果业务需要延迟消息怎么办？Kafka 能实现吗？**

**追问答案：**

Kafka 原生不支持延迟消息，但可以通过以下方案实现：

```java
// 方案1：应用层实现延迟（推荐简单场景）
public void sendDelayedMessage(String topic, String message, long delayMs) {
    // 将延迟时间放入消息头
    ProducerRecord<String, String> record = new ProducerRecord<>(topic, message);
    record.headers().add("delay_until", 
        String.valueOf(System.currentTimeMillis() + delayMs).getBytes());
    producer.send(record);
}

// 消费时检查是否到达延迟时间
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        Header delayHeader = record.headers().lastHeader("delay_until");
        if (delayHeader != null) {
            long delayUntil = Long.parseLong(new String(delayHeader.value()));
            if (System.currentTimeMillis() < delayUntil) {
                // 暂停消费，等待延迟时间到达
                Thread.sleep(100);
                continue;
            }
        }
        // 处理消息
        processMessage(record);
    }
}

// 方案2：使用多级延迟队列（推荐生产环境）
// 创建多个延迟级别的 Topic：delay_1s, delay_5s, delay_30s, delay_1min...
// 消息到期后转发到目标 Topic
```

---

### 3. Kafka 的核心概念有哪些？

**答案：**

Kafka 核心概念层次结构：

```
┌────────────────────────────────────────────────────────────────┐
│                        Kafka 核心概念                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                      Broker (节点)                        │ │
│  │  Kafka 集群中的单个服务节点，负责消息存储和转发             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                      Topic (主题)                         │ │
│  │  消息的逻辑分类，类似数据库中的表                          │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  Partition 0  │  Partition 1  │  Partition 2     │  │ │
│  │  │  ┌─────────┐  │  ┌─────────┐  │  ┌─────────┐     │  │ │
│  │  │  │Offset 0 │  │  │Offset 0 │  │  │Offset 0 │     │  │ │
│  │  │  │Offset 1 │  │  │Offset 1 │  │  │Offset 1 │     │  │ │
│  │  │  │Offset 2 │  │  │Offset 2 │  │  │Offset 2 │     │  │ │
│  │  │  │  ...    │  │  │  ...    │  │  │  ...    │     │  │ │
│  │  │  └─────────┘  │  └─────────┘  │  └─────────┘     │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌────────────────────┐    ┌────────────────────┐             │
│  │   Producer         │    │   Consumer Group   │             │
│  │   (生产者)          │    │   (消费者组)        │             │
│  │   发送消息到 Topic  │    │   ┌─────┐ ┌─────┐ │             │
│  │                    │    │   │ C1  │ │ C2  │ │             │
│  │   ┌─────────────┐  │    │   └─────┘ └─────┘ │             │
│  │   │ 消息序列化   │  │    │   组内消费者分摊分区│             │
│  │   │ 分区选择    │  │    └────────────────────┘             │
│  │   └─────────────┘  │                                       │
│  └────────────────────┘                                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    Message (消息)                         │ │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────────┐ │ │
│  │  │  Key    │  Value  │ Headers │ Timestamp│ Partition   │ │ │
│  │  │  可选   │  消息体  │ 键值对  │ 时间戳   │ 分区号      │ │ │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

**核心概念详解：**

| 概念 | 说明 | 类比 |
|------|------|------|
| **Broker** | Kafka 集群节点 | 数据库服务器实例 |
| **Topic** | 消息分类的逻辑概念 | 数据库表 |
| **Partition** | Topic 的物理分片 | 表的分表 |
| **Offset** | 消息在分区中的位置 | 自增主键 |
| **Replica** | 分区的副本 | 主从复制 |
| **Consumer Group** | 消费者逻辑分组 | 消费组 |

**追问：Record 和 Message 是同一个概念吗？**

**追问答案：**

是的，在不同语境下称呼不同：
- **Record**：代码层面，如 `ProducerRecord`、`ConsumerRecord`
- **Message**：概念层面，描述消息队列中的消息
- **Event**：事件驱动架构中的称呼，强调消息的业务含义

---

### 4. Topic 和 Partition 的关系？

**答案：**

**关系图解：**

```
Topic: orders
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Partition 0 (Leader: Broker1, Follower: Broker2, Broker3)    │
│   ┌───────┬───────┬───────┬───────┬───────┬───────┐            │
│   │  msg0 │  msg1 │  msg2 │  msg3 │  msg4 │  msg5 │ ...        │
│   └───────┴───────┴───────┴───────┴───────┴───────┘            │
│   Offset:  0       1       2       3       4       5            │
│                                                                 │
│   Partition 1 (Leader: Broker2, Follower: Broker1, Broker3)    │
│   ┌───────┬───────┬───────┬───────┬───────┐                    │
│   │  msg0 │  msg1 │  msg2 │  msg3 │  msg4 │ ...                │
│   └───────┴───────┴───────┴───────┴───────┘                    │
│   Offset:  0       1       2       3       4                    │
│                                                                 │
│   Partition 2 (Leader: Broker3, Follower: Broker1, Broker2)    │
│   ┌───────┬───────┬───────┬───────┐                            │
│   │  msg0 │  msg1 │  msg2 │  msg3 │ ...                        │
│   └───────┴───────┴───────┴───────┘                            │
│   Offset:  0       1       2       3                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心要点：**

1. **逻辑与物理分离**
   - Topic 是逻辑概念，对用户可见
   - Partition 是物理概念，实际存储单元

2. **分区内有序**
   - 每个分区内消息严格按 offset 有序
   - 不同分区之间消息无顺序保证

3. **并行处理能力**
   - 分区数决定了最大并行度
   - 一个分区只能被消费者组内一个消费者消费

**分区策略代码示例：**

```java
// 默认分区器逻辑
public class DefaultPartitioner implements Partitioner {
    public int partition(String topic, Object key, byte[] keyBytes, 
                         Object value, byte[] valueBytes, Cluster cluster) {
        
        List<PartitionInfo> partitions = cluster.partitionsForTopic(topic);
        int numPartitions = partitions.size();
        
        if (keyBytes == null) {
            // 无 key 时，使用轮询或粘性分区
            return stickyPartitionCache.nextPartition(topic, cluster, prevPartition);
        }
        
        // 有 key 时，使用 hash 取模
        return Utils.toPositive(Utils.murmur2(keyBytes)) % numPartitions;
    }
}

// 自定义分区器：按业务字段分区
public class OrderPartitioner implements Partitioner {
    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                         Object value, byte[] valueBytes, Cluster cluster) {
        if (value instanceof Order) {
            Order order = (Order) value;
            // 按订单类型分区：0-普通订单, 1-秒杀订单, 2-预售订单
            return order.getType().getCode() % cluster.partitionCountForTopic(topic);
        }
        return 0;
    }
}

// 配置自定义分区器
props.put(ProducerConfig.PARTITIONER_CLASS_CONFIG, OrderPartitioner.class.getName());
```

**追问：分区数怎么确定？越多越好吗？**

**追问答案：**

**分区数确定公式：**

```
分区数 ≈ max(目标吞吐量 / 单分区吞吐量, 消费者数量)

例如：
- 目标吞吐量：100万/秒
- 单分区吞吐量：10万/秒
- 消费者数量：10个

分区数 = max(100万/10万, 10) = 10 个
```

**分区数过多的风险：**

```
❌ 过多分区的问题：

1. 客户端内存压力
   ┌────────────────────────────────────┐
   │ 每个分区需要独立的内存缓冲区        │
   │ Producer: buffer.memory / 分区数   │
   │ Consumer: fetch.max.bytes × 分区数 │
   └────────────────────────────────────┘

2. Broker 端压力
   - 文件句柄数 × 分区数 × 副本数
   - Controller 故障恢复时间增长
   - Leader 选举时间增长

3. 可用性降低
   - 分区越多，单个分区故障概率越高
   - 分区迁移耗时增加
```

**推荐配置：**

```yaml
# 一般建议
单 Broker 分区数: < 2000
整个集群分区数: < 10000
单个 Topic 分区数: 根据实际吞吐量计算，一般 3-100
```

---

### 5. Kafka 的消息模型是什么？

**答案：**

Kafka 采用**发布-订阅模型（Pub/Sub）**，同时支持**点对点模式**。

**两种模型对比：**

```
发布-订阅模型（Pub/Sub）：
┌──────────┐
│ Producer │
└────┬─────┘
     │
     ▼
┌──────────┐      ┌───────────────┐
│  Topic   │──────▶ Consumer Grp A│
└──────────┘      │   ┌───┐       │
     │            │   │C1 │       │
     │            │   └───┘       │
     │            └───────────────┘
     │            
     │            ┌───────────────┐
     └────────────▶ Consumer Grp B│
                  │   ┌───┐ ┌───┐ │
                  │   │C2 │ │C3 │ │
                  │   └───┘ └───┘ │
                  └───────────────┘

特点：同一条消息会被所有订阅的消费者组消费


点对点模型（Queue）：
┌──────────┐
│ Producer │
└────┬─────┘
     │
     ▼
┌──────────┐      ┌───────────────┐
│  Topic   │──────▶ Consumer Grp  │
└──────────┘      │   ┌───┐ ┌───┐ │
                  │   │C1 │ │C2 │ │
                  │   └───┘ └───┘ │
                  └───────────────┘

特点：一条消息只会被组内一个消费者消费
```

**Kafka 消费者组机制：**

```java
// 消费者组实现点对点模式
// 同一个消费者组内的消费者分摊分区

Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-processors"); // 相同 group.id 形成一个消费组
// ...

// 消费组内消费者与分区的分配关系：
// 
// Topic (3个分区)        Consumer Group "order-processors"
// ┌─────────────┐        ┌─────────────────┐
// │ Partition 0 │───────▶│   Consumer A    │
// ├─────────────┤        ├─────────────────┤
// │ Partition 1 │───────▶│   Consumer B    │
// ├─────────────┤        ├─────────────────┤
// │ Partition 2 │───────▶│   Consumer C    │
// └─────────────┘        └─────────────────┘
```

**追问：消费者组内消费者数量超过分区数会怎样？**

**追问答案：**

```
场景：3 个分区，5 个消费者

┌─────────────┐        ┌─────────────────┐
│ Partition 0 │───────▶│   Consumer 1    │ ✓ 消费
├─────────────┤        ├─────────────────┤
│ Partition 1 │───────▶│   Consumer 2    │ ✓ 消费
├─────────────┤        ├─────────────────┤
│ Partition 2 │───────▶│   Consumer 3    │ ✓ 消费
└─────────────┘        ├─────────────────┤
                       │   Consumer 4    │ ✗ 空闲
                       ├─────────────────┤
                       │   Consumer 5    │ ✗ 空闲
                       └─────────────────┘

结论：
- 消费者数 > 分区数时，部分消费者会空闲
- 建议消费者数 ≤ 分区数
- 可以动态增加分区来提高并行度
```

---

### 6. Kafka 消息的 Key 有什么作用？

**答案：**

**Key 的三大作用：**

```
1. 分区路由
   ┌─────────────────────────────────────────────┐
   │  相同 Key 的消息会被发送到同一个分区          │
   │  保证相同 Key 的消息严格有序                  │
   │                                             │
   │  Key="order_001" ────▶ Partition 0         │
   │  Key="order_001" ────▶ Partition 0         │
   │  Key="order_001" ────▶ Partition 0         │
   │                                             │
   │  Key="order_002" ────▶ Partition 2         │
   │  Key="order_002" ────▶ Partition 2         │
   └─────────────────────────────────────────────┘

2. 日志压缩（Log Compaction）
   ┌─────────────────────────────────────────────┐
   │  相同 Key 只保留最新的 Value                 │
   │  适合存储最新状态（如用户信息变更）           │
   │                                             │
   │  Key="user_1", Value="name=A"               │
   │  Key="user_1", Value="name=B"  ← 保留       │
   │  Key="user_1", Value="name=C"  ← 保留       │
   └─────────────────────────────────────────────┘

3. 消息追踪
   ┌─────────────────────────────────────────────┐
   │  通过 Key 追踪特定业务的消息流向              │
   │  如订单 ID、用户 ID 等                       │
   └─────────────────────────────────────────────┘
```

**代码示例：**

```java
// 场景1：保证同一订单的消息有序
public void sendOrderEvents(Order order) {
    // 使用订单 ID 作为 Key，确保同一订单的所有事件进入同一分区
    String key = order.getOrderId();
    
    // 订单创建事件
    producer.send(new ProducerRecord<>("orders", key, 
        new OrderEvent("CREATED", order)));
    
    // 订单支付事件
    producer.send(new ProducerRecord<>("orders", key, 
        new OrderEvent("PAID", order)));
    
    // 订单发货事件
    producer.send(new ProducerRecord<>("orders", key, 
        new OrderEvent("SHIPPED", order)));
}

// 场景2：日志压缩 - 存储用户最新状态
// Topic 配置：cleanup.policy=compact
public void updateUserProfile(User user) {
    // Key = 用户ID，Value = 最新用户信息
    // Kafka 会保留每个用户的最新记录
    producer.send(new ProducerRecord<>("user-profiles", 
        user.getId(), user.toJson()));
}
```

**追问：Key 为 null 时消息怎么分配分区？**

**追问答案：**

```java
// Key 为 null 时的分区策略演进

// 旧版本（2.4 之前）：轮询策略
// 优点：负载均匀
// 缺点：频繁建立连接，延迟高

// 新版本（2.4+）：粘性分区器（Sticky Partitioner）
// 优点：批量发送到同一分区，提高吞吐量
// 工作原理：

┌────────────────────────────────────────────────────┐
│ Sticky Partitioner 工作流程                         │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. 选择一个分区作为"粘性分区"                       │
│     ┌─────────────┐                               │
│     │ Partition 1 │ ← 当前粘性分区                 │
│     └─────────────┘                               │
│                                                    │
│  2. 消息批量发送到粘性分区                          │
│     batch.size = 16KB 或 linger.ms = 5ms          │
│     ┌─────────────────────────────┐               │
│     │ msg1, msg2, ..., msgN       │ ──▶ Part 1   │
│     └─────────────────────────────┘               │
│                                                    │
│  3. 批次满或超时后，切换到新的粘性分区               │
│     ┌─────────────┐                               │
│     │ Partition 2 │ ← 新的粘性分区                 │
│     └─────────────┘                               │
│                                                    │
└────────────────────────────────────────────────────┘

// 配置粘性分区器
props.put("partitioner.class", 
    "org.apache.kafka.clients.producer.UniformStickyPartitioner");
```

---

### 7. Kafka 消息格式是怎样的？

**答案：**

**消息格式演进：**

```
消息格式版本：
┌─────────────┬─────────────┬──────────────────────────────────┐
│   版本      │   协议版本   │           主要特性                │
├─────────────┼─────────────┼──────────────────────────────────┤
│  Message V0 │  0.10 之前   │ 基础消息格式                       │
│  Message V1 │  0.10-0.11  │ 增加时间戳                         │
│  Message V2 │  0.11+      │ 增加Headers、支持事务              │
└─────────────┴─────────────┴──────────────────────────────────┘
```

**V2 消息格式详解：**

```
Record（单条消息）：
┌──────────────────────────────────────────────────────────────┐
│ length (varint)        │ 消息总长度                          │
│ attributes (int8)      │ 属性（压缩类型、时间戳类型）          │
│ timestampDelta (varint)│ 相对于批次的时间戳增量               │
│ offsetDelta (varint)   │ 相对于批次的偏移量增量               │
│ keyLength (varint)     │ Key 长度                           │
│ key (bytes)            │ Key 内容                           │
│ valueLength (varint)   │ Value 长度                         │
│ value (bytes)          │ Value 内容                         │
│ headers (varint)       │ Headers 数组                       │
│   ├── headerKey       │ Header Key                         │
│   └── headerValue     │ Header Value                       │
└──────────────────────────────────────────────────────────────┘

Record Batch（消息批次）：
┌──────────────────────────────────────────────────────────────┐
│ baseOffset (int64)     │ 批次起始偏移量                       │
│ batchLength (int32)    │ 批次总长度                          │
│ partitionLeaderEpoch  │ 分区 Leader 年代                    │
│ magic (int8)           │ 消息格式版本 (2)                    │
│ crc (int32)            │ CRC32 校验                          │
│ attributes (int16)     │ 属性（压缩、事务、控制消息）          │
│ lastOffsetDelta       │ 最后一条消息的偏移量增量              │
│ baseTimestamp (int64) │ 批次起始时间戳                       │
│ maxTimestamp (int64)  │ 批次最大时间戳                       │
│ producerId (int64)    │ 生产者 ID（事务/幂等）               │
│ producerEpoch (int16) │ 生产者年代                          │
│ baseSequence (int32)  │ 起始序列号                          │
│ records (varint array)│ 消息记录数组                        │
└──────────────────────────────────────────────────────────────┘
```

**Java 代码查看消息内容：**

```java
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        System.out.println("━━━━━━━━━━ 消息详情 ━━━━━━━━━━");
        System.out.println("Topic: " + record.topic());
        System.out.println("Partition: " + record.partition());
        System.out.println("Offset: " + record.offset());
        System.out.println("Timestamp: " + new Date(record.timestamp()));
        System.out.println("TimestampType: " + record.timestampType());
        System.out.println("Key: " + record.key());
        System.out.println("Value: " + record.value());
        
        // Headers
        System.out.println("Headers:");
        for (Header header : record.headers()) {
            System.out.println("  " + header.key() + ": " + 
                new String(header.value()));
        }
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}
```

**追问：消息压缩是在哪里进行的？支持哪些压缩算法？**

**追问答案：**

```
压缩位置：Producer 端压缩，Broker 端透传，Consumer 端解压

┌──────────┐     压缩      ┌──────────┐    透传     ┌──────────┐    解压     ┌──────────┐
│ Producer │ ───────────▶ │  Broker  │ ──────────▶ │  Broker  │ ──────────▶ │ Consumer │
└──────────┘              └──────────┘             └──────────┘             └──────────┘

支持的压缩算法对比：
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   算法       │  压缩比      │  CPU 消耗   │   推荐场景   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   none      │   1:1       │   最低      │   网络带宽充足│
│   gzip      │   1:3~5     │   高        │   带宽受限   │
│   snappy    │   1:2~3     │   中        │   平衡场景   │
│   lz4       │   1:2~3     │   低        │   高吞吐场景 │
│   zstd      │   1:3~4     │   中        │   Kafka推荐 │
└─────────────┴─────────────┴─────────────┴─────────────┘

// 配置压缩
props.put("compression.type", "zstd"); // 推荐 zstd 或 lz4
props.put("batch.size", 32768);        // 批次大小影响压缩效果
```

---

### 8. Kafka 的存储机制是怎样的？

**答案：**

**存储架构图：**

```
Kafka 存储目录结构：

/tmp/kafka-logs/
├── topic-partition/
│   ├── orders-0/
│   │   ├── 00000000000000000000.log    # 日志文件（消息数据）
│   │   ├── 00000000000000000000.index  # 偏移量索引
│   │   ├── 00000000000000000000.timeindex # 时间戳索引
│   │   └── leader-epoch-checkpoint     # Leader 年代检查点
│   │
│   ├── orders-1/
│   │   └── ...
│   │
│   └── orders-2/
│       └── ...
│
├── __consumer_offsets-0/    # 消费者组 offset 存储
├── __consumer_offsets-1/
└── ...
```

**日志段（Log Segment）机制：**

```
Log Segment 结构：

┌─────────────────────────────────────────────────────────────────┐
│                      Partition Log                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Active Segment (可写入)                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Segment 2 (当前活跃)                                      │   │
│  │ 文件名: 00000000000000000025.log (baseOffset = 25)       │   │
│  │ ┌─────┬─────┬─────┬─────┐                               │   │
│  │ │msg25│msg26│msg27│ ... │  ← 追加写入                   │   │
│  │ └─────┴─────┴─────┴─────┘                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Closed Segments (只读)                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Segment 1                                                │   │
│  │ 文件名: 00000000000000000000.log (baseOffset = 0)        │   │
│  │ ┌─────┬─────┬─────┬─────┬─────┐                         │   │
│  │ │msg0 │msg1 │msg2 │...  │msg24│                         │   │
│  │ └─────┴─────┴─────┴─────┴─────┘                         │   │
│  │                                                          │   │
│  │ 配套索引文件:                                             │   │
│  │ 00000000000000000000.index     (offset → position)      │   │
│  │ 00000000000000000000.timeindex (timestamp → offset)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Segment 滚动条件：
1. log.segment.bytes = 1GB（默认）- 达到大小限制
2. log.roll.ms/hours = 168h（默认）- 达到时间限制
```

**索引机制：**

```
稀疏索引设计：

.index 文件结构（Offset 索引）：
┌─────────────┬───────────────┐
│  Offset     │   Position    │
├─────────────┼───────────────┤
│     0       │      0        │
│    28       │    1024       │  ← 每隔约 4KB 建一个索引项
│    56       │    2048       │
│    84       │    3072       │
│   ...       │    ...        │
└─────────────┴───────────────┘

.timeindex 文件结构（时间戳索引）：
┌─────────────┬───────────────┐
│ Timestamp   │    Offset     │
├─────────────┼───────────────┤
│ 1640000000  │      0        │
│ 1640000005  │     30        │
│ 1640000010  │     58        │
│   ...       │    ...        │
└─────────────┴───────────────┘

查找流程：
1. 二分查找 .index 文件，定位到近似位置
2. 从 .log 文件中的位置开始顺序扫描
3. 找到目标 offset 的消息
```

**追问：为什么 Kafka 用稀疏索引而不是稠密索引？**

**追问答案：**

```
稀疏索引 vs 稠密索引：

稠密索引（每条消息一个索引项）：
┌────────────────────────────────────────────────────────┐
│ 优点：查找精确，无需扫描                                │
│ 缺点：索引文件巨大，占用大量内存                        │
│       如果消息 100 字节，索引项 16 字节                 │
│       索引大小 ≈ 消息大小的 16%                         │
└────────────────────────────────────────────────────────┘

稀疏索引（每隔 N 字节一个索引项）：
┌────────────────────────────────────────────────────────┐
│ 优点：索引文件小，可完全加载到内存                      │
│       索引大小 ≈ 消息大小的 0.1%                        │
│ 缺点：需要小范围顺序扫描                                │
└────────────────────────────────────────────────────────┘

Kafka 选择稀疏索引的原因：
1. 消息通常是顺序消费的，不需要精确随机访问
2. 内存效率高，索引可完全放入 Page Cache
3. 磁盘空间占用小
4. 稀疏索引 + 顺序扫描的组合在大多数场景下足够高效
```

---

### 9. Kafka 支持哪些消息语义？

**答案：**

**三种消息语义：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    三种消息传递语义                              │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│  At Most Once   │  消息最多传递一次，可能丢失                    │
│  最多一次        │  ┌─────┐         ┌─────────┐                 │
│                 │  │ P   │──msg──▶ │  C      │                 │
│                 │  └─────┘   ✗     └─────────┘                 │
│                 │  网络故障导致消息丢失                           │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  At Least Once  │  消息至少传递一次，可能重复                    │
│  最少一次        │  ┌─────┐      ┌─────────┐                    │
│                 │  │ P   │─msg─▶│  C      │ ✓                  │
│                 │  └─────┘      └─────────┘                    │
│                 │       └─msg─▶│  C      │ ✓ 重复消费          │
│                 │               └─────────┘                    │
│                 │  重试导致消息重复                              │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│ Exactly Once    │  消息精确传递一次，不丢失不重复                │
│ 精确一次        │  ┌─────┐      ┌─────────┐                    │
│                 │  │ P   │─msg─▶│  C      │ ✓                  │
│                 │  └─────┘      └─────────┘                    │
│                 │  事务 + 幂等性保证                            │
│                 │                                               │
└─────────────────┴───────────────────────────────────────────────┘
```

**各语义的实现方式：**

```java
// At Most Once - 最简单，但可能丢消息
props.put("enable.auto.commit", "true");     // 自动提交 offset
props.put("auto.commit.interval.ms", "1000"); // 每秒提交一次
props.put("acks", "0");                       // 不等待确认
// 风险：消息发送后崩溃，或 offset 提交后处理失败

// At Least Once - 常用配置
props.put("enable.auto.commit", "false");    // 手动提交
props.put("acks", "all");                     // 等待所有副本确认

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        processMessage(record);  // 处理消息
    }
    consumer.commitSync();  // 处理完再提交 offset
}
// 风险：处理成功但提交失败，下次重新消费导致重复

// Exactly Once - 最可靠，最复杂
// 生产者端
props.put("enable.idempotence", "true");  // 开启幂等性
props.put("acks", "all");

// 或使用事务
props.put("transactional.id", "my-transactional-id");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

try {
    producer.beginTransaction();
    producer.send(new ProducerRecord<>("topic", "key", "value"));
    // 可以同时发送到多个 Topic
    producer.send(new ProducerRecord<>("topic2", "key", "value"));
    producer.commitTransaction();
} catch (Exception e) {
    producer.abortTransaction();
}

// 消费者端（配合事务）
props.put("isolation.level", "read_committed"); // 只读已提交的事务消息
```

**追问：Exactly Once 在跨系统场景下怎么实现？**

**追问答案：**

```
跨系统 Exactly Once 方案：

方案1：两阶段提交（2PC）
┌─────────────────────────────────────────────────────────────┐
│  优点：强一致性                                              │
│  缺点：性能差，实现复杂，需要协调者                           │
└─────────────────────────────────────────────────────────────┘

方案2：本地消息表（推荐）
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ 业务操作      │      │ 本地消息表    │                    │
│  │ UPDATE ...   │─────▶│ INSERT msg   │  同一本地事务       │
│  └──────────────┘      └──────────────┘                    │
│                              │                              │
│                              ▼                              │
│                        ┌──────────────┐                    │
│                        │ 定时任务发送  │                    │
│                        │ 到 Kafka     │                    │
│                        └──────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

方案3：Kafka Connect + 事务
┌─────────────────────────────────────────────────────────────┐
│  Kafka Connect 支持与外部系统的事务集成                       │
│  如 JDBC Sink Connector 可实现精确一次写入数据库              │
└─────────────────────────────────────────────────────────────┘

方案4：幂等消费（最实用）
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  消费者端实现幂等性：                                        │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ 1. 业务主键去重（Redis/数据库唯一索引）                 │ │
│  │ 2. 使用消息 ID 作为幂等键                              │ │
│  │ 3. 状态机设计（订单状态流转）                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘

// 代码示例：幂等消费者
public void consume(ConsumerRecord<String, String> record) {
    String messageId = record.key(); // 或从 header 获取
    
    // 1. 检查是否已处理（Redis）
    if (redis.setnx("processed:" + messageId, "1", 24 * 3600)) {
        return; // 已处理，跳过
    }
    
    // 2. 处理消息
    processMessage(record.value());
    
    // 3. 幂等性保证：即使步骤2重复执行，结果也一样
}
```

---

### 10. Kafka 的 Topic 命名有什么规范？

**答案：**

**命名规范：**

```
命名规则：
┌─────────────────────────────────────────────────────────────┐
│ 1. 长度限制：1-249 字符                                      │
│ 2. 字符限制：字母、数字、点(.)、下划线(_)、减号(-)            │
│ 3. 不能以 __ 开头（保留给内部 Topic）                        │
│ 4. 不能以 . 开头或结尾                                       │
│ 5. 不能包含连续的 ..                                         │
└─────────────────────────────────────────────────────────────┘

推荐命名格式：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  <业务域>.<应用名>.<数据类型>.<环境>                        │
│                                                             │
│  示例：                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ order.service.events.prod        # 订单服务事件       │   │
│  │ user.profile.updates.test        # 用户资料更新       │   │
│  │ payment.notification.alert.stag  # 支付通知告警       │   │
│  │ logistics.tracking.stream.prod   # 物流追踪流         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

内部 Topic（以 __ 开头）：
┌─────────────────────────────────────────────────────────────┐
│ __consumer_offsets    # 消费者组 offset 存储               │
│ __transaction_state   # 事务状态存储                        │
│ __schema_registry     # Schema Registry                    │
│ __cluster_metadata    # KRaft 模式元数据                    │
└─────────────────────────────────────────────────────────────┘
```

**追问：Topic 命名和性能有关系吗？**

**追问答案：**

有一定关系，主要体现在：

```
1. ZooKeeper/KRaft 元数据存储
   Topic 名称会存储在元数据中，过长名称会占用更多内存

2. 监控和日志
   名称过长会影响监控展示和日志可读性

3. 分区分配算法
   分区分配时会使用 Topic 名称进行计算，不影响性能

建议：
- 使用简短但有意义的名称
- 保持命名一致性
- 使用前缀区分环境
```


---

## 高性能篇

### 1. Kafka 为什么高性能？

**答案：**

Kafka 高性能的四大核心设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                  Kafka 高性能四大支柱                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐                         │
│  │   顺序写入     │  │   零拷贝       │                         │
│  │ Sequential I/O│  │ Zero Copy     │                         │
│  │               │  │               │                         │
│  │ 磁盘顺序写     │  │ sendfile      │                         │
│  │ 比随机写快     │  │ 减少用户态     │                         │
│  │ 6000 倍       │  │ 内核态切换     │                         │
│  └───────────────┘  └───────────────┘                         │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐                         │
│  │   页缓存       │  │  批量处理      │                         │
│  │ Page Cache    │  │ Batching      │                         │
│  │               │  │               │                         │
│  │ OS 级缓存      │  │ 批量发送       │                         │
│  │ 写缓冲+读缓存  │  │ 批量压缩       │                         │
│  │ 减少磁盘 I/O  │  │ 批量确认       │                         │
│  └───────────────┘  └───────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

性能数据对比：
┌──────────────────┬────────────────┬────────────────┐
│      操作         │    速度        │     说明       │
├──────────────────┼────────────────┼────────────────┤
│ 内存访问          │ ~100 GB/s      │ 最快           │
│ 顺序磁盘读        │ ~200 MB/s      │ SSD 可达更高   │
│ 随机磁盘读        │ ~100 KB/s      │ 慢 2000 倍     │
│ 千兆网络传输      │ ~100 MB/s      │ 网络瓶颈       │
└──────────────────┴────────────────┴────────────────┘
```

**性能测试代码：**

```java
// 生产者性能测试
public class ProducerPerformance {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", StringSerializer.class.getName());
        props.put("value.serializer", StringSerializer.class.getName());
        
        // 高性能配置
        props.put("batch.size", 65536);           // 64KB 批次
        props.put("linger.ms", 10);               // 等待 10ms 凑批
        props.put("buffer.memory", 67108864);     // 64MB 缓冲区
        props.put("compression.type", "lz4");     // 压缩
        props.put("acks", "1");                    // 平衡可靠性和性能
        props.put("max.in.flight.requests.per.connection", 5);
        
        Producer<String, String> producer = new KafkaProducer<>(props);
        
        String message = generateMessage(1024); // 1KB 消息
        int messageCount = 1_000_000;
        
        long start = System.currentTimeMillis();
        
        for (int i = 0; i < messageCount; i++) {
            producer.send(new ProducerRecord<>("perf-test", 
                String.valueOf(i), message));
        }
        
        producer.flush();
        long elapsed = System.currentTimeMillis() - start;
        
        System.out.printf("发送 %d 条消息，耗时 %d ms%n", messageCount, elapsed);
        System.out.printf("吞吐量: %.2f 条/秒%n", messageCount * 1000.0 / elapsed);
        System.out.printf("吞吐量: %.2f MB/秒%n", 
            messageCount * 1.0 / elapsed * 1000 / 1024 / 1024);
        
        producer.close();
    }
}
```

**追问：为什么顺序写比随机写快这么多？**

**追问答案：**

```
磁盘结构原理：

┌─────────────────────────────────────────────────────────────┐
│                      机械硬盘结构                            │
│                                                             │
│    ┌─────────────────────────────────────┐                 │
│    │         磁盘盘片                      │                 │
│    │    ┌───────────────────────┐       │                 │
│    │    │ ──────────────────────│       │                 │
│    │    │ ──────────────────────│       │                 │
│    │    │ ──────────────────────│       │                 │
│    │    │          ┌───┐        │       │                 │
│    │    │          │头 │←磁头   │       │                 │
│    │    │          └───┘        │       │                 │
│    │    └───────────────────────┘       │                 │
│    └─────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

顺序写：
┌─────────────────────────────────────────────────────────────┐
│ 数据写入位置：1 → 2 → 3 → 4 → 5 → 6 → ...                  │
│ 磁头移动：几乎不动，数据连续写入                              │
│ 时间消耗：仅数据传输时间                                      │
│ 吞吐量：≈ 200 MB/s (HDD) / 500+ MB/s (SSD)                  │
└─────────────────────────────────────────────────────────────┘

随机写：
┌─────────────────────────────────────────────────────────────┐
│ 数据写入位置：100 → 5 → 9999 → 42 → 8888 → ...              │
│ 磁头移动：频繁寻道，每次移动约 10ms                           │
│ 时间消耗：寻道时间 + 旋转延迟 + 数据传输                      │
│ 吞吐量：≈ 100 KB/s ~ 1 MB/s                                 │
└─────────────────────────────────────────────────────────────┘

Kafka 的设计：
┌─────────────────────────────────────────────────────────────┐
│ 1. 每个 Partition 是一个独立的日志文件                        │
│ 2. 消息只能追加到文件末尾（Append-Only）                      │
│ 3. 不允许修改已写入的消息                                     │
│ 4. 删除是批量删除整个日志段                                   │
│ → 天然符合顺序写的模式                                        │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 什么是顺序写？为什么快？

**答案：**

**顺序写原理图解：**

```
传统数据库随机写 vs Kafka 顺序写：

传统数据库（B+树结构）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  写入请求                                                   │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────┐                                                │
│  │ 随机位置 │  需要找到对应的数据页                           │
│  │ 更新    │                                                │
│  └─────────┘                                                │
│     │                                                       │
│     ├──▶ Page 5 修改 ──▶ 磁盘寻道                           │
│     ├──▶ Page 100 修改 ──▶ 磁盘寻道                         │
│     ├──▶ Page 23 修改 ──▶ 磁盘寻道                          │
│     └──▶ Page 88 修改 ──▶ 磁盘寻道                          │
│                                                             │
│  性能瓶颈：频繁的磁盘寻道                                    │
└─────────────────────────────────────────────────────────────┘

Kafka 顺序写：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  写入请求                                                   │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────────────────────────────────┐           │
│  │  Log File (Append Only)                     │           │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐     │           │
│  │  │msg1 │msg2 │msg3 │msg4 │msg5 │ NEW │←───│ 追加写入   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┘     │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  无需寻道，直接追加到文件末尾                                │
│  性能：接近内存速度                                          │
└─────────────────────────────────────────────────────────────┘
```

**代码模拟对比：**

```java
// 顺序写性能测试
public class SequentialWriteTest {
    public static void main(String[] args) throws IOException {
        // 顺序写
        try (FileOutputStream fos = new FileOutputStream("sequential.bin")) {
            byte[] data = new byte[1024 * 1024]; // 1MB 数据
            long start = System.currentTimeMillis();
            
            for (int i = 0; i < 1000; i++) { // 写入 1GB
                fos.write(data);
            }
            
            long elapsed = System.currentTimeMillis() - start;
            System.out.printf("顺序写 1GB 耗时: %d ms, 速度: %.2f MB/s%n",
                elapsed, 1000.0 * 1000 / elapsed);
        }
    }
}

// 随机写性能测试
public class RandomWriteTest {
    public static void main(String[] args) throws IOException {
        // 随机写
        try (RandomAccessFile raf = new RandomAccessFile("random.bin", "rw")) {
            byte[] data = new byte[1024]; // 1KB 数据
            Random random = new Random();
            long start = System.currentTimeMillis();
            
            for (int i = 0; i < 1000000; i++) { // 写入 1GB (1M 次)
                long position = (long) (random.nextDouble() * 1024 * 1024 * 1024);
                raf.seek(position);
                raf.write(data);
            }
            
            long elapsed = System.currentTimeMillis() - start;
            System.out.printf("随机写 1GB 耗时: %d ms, 速度: %.2f MB/s%n",
                elapsed, 1000.0 * 1000 / elapsed);
        }
    }
}

// 典型输出：
// 顺序写 1GB 耗时: 5000 ms, 速度: 200.00 MB/s
// 随机写 1GB 耗时: 600000 ms, 速度: 1.67 MB/s
```

**追问：SSD 也需要顺序写吗？**

**追问答案：**

```
SSD 的特性：

┌─────────────────────────────────────────────────────────────┐
│ SSD (固态硬盘) 结构                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │                    Block (块)                      │     │
│  │  ┌─────────┬─────────┬─────────┬─────────┐       │     │
│  │  │  Page   │  Page   │  Page   │  Page   │       │     │
│  │  │  4KB    │  4KB    │  4KB    │  4KB    │       │     │
│  │  └─────────┴─────────┴─────────┴─────────┘       │     │
│  │                 Block = 256KB ~ 4MB              │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  特点：                                                     │
│  1. 读：随机读和顺序读速度相近                               │
│  2. 写：写入新 Page 快，覆盖写需要擦除整个 Block             │
│  3. 擦除：Block 级别擦除，不能只擦除单个 Page                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

SSD 的写入放大问题：

随机写场景：
┌─────────────────────────────────────────────────────────────┐
│ 1. 写入 Page A（在其他 Block 中）                            │
│ 2. SSD 必须擦除整个 Block（包含有用的数据）                   │
│ 3. 将有用的数据和 Page A 一起写入新 Block                     │
│ 4. 实际写入量 > 请求写入量（写入放大）                        │
│                                                             │
│ 写入放大系数：随机写 ≈ 3-10x，顺序写 ≈ 1x                    │
└─────────────────────────────────────────────────────────────┘

结论：即使是 SSD，顺序写仍然更优
- 减少写入放大
- 延长 SSD 寿命
- 提高吞吐量
```

---

### 3. 什么是零拷贝？

**答案：**

**传统数据传输 vs 零拷贝：**

```
传统数据传输（4 次拷贝，4 次上下文切换）：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  磁盘文件 ──────▶ 网络传输                                   │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   磁盘    │    │  内核态   │    │  用户态   │              │
│  │          │    │          │    │          │              │
│  │   文件    │───▶│Page Cache│───▶│ 用户缓冲  │  拷贝 1      │
│  │          │    │          │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       DMA              │              │                     │
│       拷贝             │              │                     │
│       (0)             │              │                     │
│                       ▼              │                     │
│                  ┌──────────┐        │                     │
│                  │ Socket   │◀───────┘  拷贝 2              │
│                  │ Buffer   │                              │
│                  └──────────┘                              │
│                       │                                     │
│                       ▼                                     │
│                  ┌──────────┐                              │
│                  │ 网卡 DMA  │  拷贝 3                      │
│                  └──────────┘                              │
│                                                             │
│  总计：4 次数据拷贝 + 4 次上下文切换                         │
└─────────────────────────────────────────────────────────────┘

零拷贝（sendfile，2 次拷贝，2 次上下文切换）：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   磁盘    │    │  内核态   │    │   网卡    │              │
│  │          │    │          │    │          │              │
│  │   文件    │───▶│Page Cache│───▶│ Socket   │              │
│  │          │    │          │    │ Buffer   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       DMA              │              DMA                   │
│       拷贝             │              拷贝                   │
│       (0)             │              (0)                    │
│                       │                                     │
│  ✗ 跳过用户态拷贝    │                                     │
│                       │                                     │
│  总计：2 次数据拷贝 + 2 次上下文切换                         │
└─────────────────────────────────────────────────────────────┘
```

**Kafka 零拷贝实现：**

```java
// Kafka 使用 Java NIO 的 transferTo 实现零拷贝
// 底层调用 Linux sendfile 系统调用

// FileChannel.transferTo 方法
public class ZeroCopyExample {
    public void transferFile(FileChannel source, SocketChannel target) 
            throws IOException {
        long position = 0;
        long count = source.size();
        
        // 零拷贝传输，数据直接从文件系统缓存到网卡
        long transferred = source.transferTo(position, count, target);
        
        // 避免了数据在用户态和内核态之间来回拷贝
    }
}

// Kafka 源码中的使用（简化版）
public class FileRecords {
    public long writeTo(GatheringByteChannel channel, long position, int length) 
            throws IOException {
        // 使用 FileChannel.transferTo 实现零拷贝
        return fileRecords.transferTo(channel, position, length);
    }
}
```

**性能对比：**

```
┌─────────────────────────────────────────────────────────────┐
│                   零拷贝性能提升                             │
├───────────────────┬─────────────────┬───────────────────────┤
│       方式         │   CPU 消耗      │    吞吐量提升         │
├───────────────────┼─────────────────┼───────────────────────┤
│ 传统方式           │   高            │    基准               │
│ 零拷贝 sendfile    │   低 50-70%     │    提升 2-3 倍        │
└───────────────────┴─────────────────┴───────────────────────┘
```

**追问：Windows 系统支持零拷贝吗？**

**追问答案：**

```
不同操作系统的零拷贝支持：

Linux：
┌─────────────────────────────────────────────────────────────┐
│ ✓ sendfile() 系统调用 - 完整零拷贝支持                       │
│ ✓ splice() - 管道零拷贝                                     │
│ ✓ Kafka 性能最佳                                            │
└─────────────────────────────────────────────────────────────┘

Windows：
┌─────────────────────────────────────────────────────────────┐
│ ✓ TransmitFile() API - 类似 sendfile                        │
│ 但 Java NIO 在 Windows 上的实现略有不同                      │
│ 性能略低于 Linux                                             │
└─────────────────────────────────────────────────────────────┘

macOS：
┌─────────────────────────────────────────────────────────────┐
│ ✓ sendfile() 系统调用                                        │
│ 实现方式与 Linux 略有差异                                    │
└─────────────────────────────────────────────────────────────┘

建议：
- 生产环境推荐使用 Linux 部署 Kafka
- Windows/macOS 主要用于开发测试
```

---

### 4. 页缓存是什么？

**答案：**

**页缓存（Page Cache）原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    操作系统内存层次                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    CPU 缓存                          │   │
│  │  L1 Cache  │  L2 Cache  │  L3 Cache                 │   │
│  │  最快       │  快        │  较快                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    内存 (RAM)                        │   │
│  │  ┌───────────────────────┬───────────────────────┐  │   │
│  │  │      应用程序内存      │     Page Cache       │  │   │
│  │  │      (用户进程)        │   (OS 自动管理)       │  │   │
│  │  │                       │                       │  │   │
│  │  │  - JVM Heap          │  - 文件系统缓存        │  │   │
│  │  │  - 堆外内存           │  - 磁盘读取缓存        │  │   │
│  │  │                       │  - 写入缓冲区         │  │   │
│  │  └───────────────────────┴───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     磁盘                             │   │
│  │  持久化存储，速度最慢                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Kafka 利用页缓存的方式：**

```
写入流程：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Producer                                                   │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Page Cache (写入缓冲)                   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Message 1  │  Message 2  │  Message 3  │   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  写入即返回（异步刷盘）                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         │ 异步刷盘（后台线程）               │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     磁盘                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  写入延迟 ≈ 内存写入速度                                     │
└─────────────────────────────────────────────────────────────┘

读取流程（热数据）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Consumer                                                   │
│     │                                                       │
│     ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Page Cache (读取缓存)                   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Message 1  │  Message 2  │  Message 3  │   │   │   │  ← 命中缓存
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  直接从内存返回，无需访问磁盘                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  读取延迟 ≈ 内存读取速度                                     │
└─────────────────────────────────────────────────────────────┘
```

**Kafka 配置参数：**

```properties
# 刷盘策略（通常使用默认值，让 OS 管理）
# 同步刷盘（不推荐，严重影响性能）
log.flush.interval.messages=1      # 每条消息刷盘
log.flush.interval.ms=0            # 立即刷盘

# 推荐配置（让 OS 管理 Page Cache）
# 不设置或设置较大值
log.flush.interval.messages=10000  # 每 10000 条消息刷盘
log.flush.interval.ms=1000         # 每秒刷盘一次

# 日志保留策略
log.retention.hours=168            # 保留 7 天
log.retention.bytes=1073741824     # 每个分区最大 1GB
log.segment.bytes=1073741824       # 每个日志段 1GB
```

**追问：Kafka 为什么不建议手动刷盘？**

**追问答案：**

```
手动刷盘的问题：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 性能损失                                                │
│     每次写入都要等待磁盘 I/O 完成                            │
│     吞吐量降低 10-100 倍                                     │
│                                                             │
│  2. 与 Page Cache 冲突                                      │
│     OS 的 Page Cache 已经做了优化                            │
│     手动刷盘打乱 OS 的优化策略                               │
│                                                             │
│  3. 不必要的刷盘                                            │
│     不是所有消息都需要立即持久化                             │
│     副本机制已经提供了冗余                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Kafka 的可靠性保证：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  数据不丢失的保证：                                          │
│                                                             │
│  1. 多副本机制                                              │
│     ┌───────────────────────────────────────────────┐      │
│     │  Leader (Broker 1)                            │      │
│     │  ┌───────────────────────────────────────┐   │      │
│     │  │  Message 1  │  Message 2  │  Message 3 │   │      │
│     │  └───────────────────────────────────────┘   │      │
│     └───────────────────────────────────────────────┘      │
│              │                    │                         │
│              ▼                    ▼                         │
│     ┌───────────────┐    ┌───────────────┐                 │
│     │ Follower (B2) │    │ Follower (B3) │                 │
│     └───────────────┘    └───────────────┘                 │
│                                                             │
│  2. ACKS 机制                                               │
│     acks=all: 所有副本确认后才返回成功                       │
│                                                             │
│  3. 即使单个 Broker 崩溃，数据仍然安全                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

推荐策略：
- 生产环境依赖副本机制保证可靠性
- 让 OS 管理 Page Cache
- 不手动配置刷盘参数
```

---

### 5. 批处理和压缩有什么作用？

**答案：**

**批处理原理：**

```
单条发送 vs 批量发送：

单条发送（低效）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Producer                         Broker                    │
│  ┌─────────┐                     ┌─────────┐               │
│  │ msg1    │───网络请求 1────────▶│         │               │
│  └─────────┘                     │         │               │
│  ┌─────────┐                     │         │               │
│  │ msg2    │───网络请求 2────────▶│         │               │
│  └─────────┘                     │         │               │
│  ┌─────────┐                     │         │               │
│  │ msg3    │───网络请求 3────────▶│         │               │
│  └─────────┘                     └─────────┘               │
│                                                             │
│  每条消息都需要一次网络请求，开销大                           │
└─────────────────────────────────────────────────────────────┘

批量发送（高效）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Producer                         Broker                    │
│  ┌───────────────────────┐       ┌─────────┐               │
│  │ msg1 │ msg2 │ msg3    │───────▶│         │               │
│  └───────────────────────┘       │         │               │
│        一次网络请求               └─────────┘               │
│                                                             │
│  多条消息合并为一次网络请求，大幅减少开销                     │
└─────────────────────────────────────────────────────────────┘
```

**批处理配置：**

```java
// 批处理相关配置
Properties props = new Properties();

// 1. batch.size - 批次大小（字节）
// 默认 16KB，建议 32KB-64KB
props.put("batch.size", 65536);

// 2. linger.ms - 等待时间
// 默认 0，建议 5-20ms
// 即使批次未满，也会在等待时间后发送
props.put("linger.ms", 10);

// 3. buffer.memory - 生产者缓冲区
// 默认 32MB
props.put("buffer.memory", 67108864);

// 4. max.block.ms - 缓冲区满时阻塞时间
props.put("max.block.ms", 60000);
```

**压缩效果对比：**

```
┌─────────────────────────────────────────────────────────────┐
│                    压缩算法对比                              │
├───────────────┬───────────┬───────────┬─────────────────────┤
│     算法       │  压缩比    │  CPU 消耗 │  推荐场景           │
├───────────────┼───────────┼───────────┼─────────────────────┤
│  none         │   1:1     │   最低    │  带宽充足           │
│  gzip         │   1:3~5   │   高      │  带宽受限，存储优先  │
│  snappy       │   1:2~3   │   中      │  平衡场景           │
│  lz4          │   1:2~3   │   低      │  高吞吐场景         │
│  zstd (推荐)  │   1:3~4   │   中      │  Kafka 2.1+ 推荐    │
└───────────────┴───────────┴───────────┴─────────────────────┘
```

**压缩配置：**

```java
// 压缩配置
props.put("compression.type", "zstd");  // Kafka 2.1+ 推荐

// 压缩在批次级别进行
// 批次越大，压缩效果越好

// 性能测试
public void testCompression() {
    String message = generateJsonMessage(1024); // 1KB JSON 消息
    
    // 无压缩
    int rawSize = message.getBytes().length;  // 1024 bytes
    
    // zstd 压缩（压缩比约 3:1）
    int compressedSize = compress(message).length; // ~341 bytes
    
    // 网络传输节省 66%
}
```

**追问：压缩是在生产者还是 Broker 端进行的？**

**追问答案：**

```
压缩位置：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│  │ Producer │         │  Broker  │         │ Consumer │    │
│  │          │         │          │         │          │    │
│  │  压缩    │────────▶│ 透传     │────────▶│  解压    │    │
│  │  (CPU)   │  压缩后  │  不解压  │  压缩后  │  (CPU)   │    │
│  │          │  数据    │          │  数据    │          │    │
│  └──────────┘         └──────────┘         └──────────┘    │
│                                                             │
│  好处：                                                     │
│  1. Broker 不需要解压/压缩，节省 CPU                        │
│  2. 网络传输量减少                                          │
│  3. 磁盘存储量减少                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Broker 端压缩配置：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  # Broker 可以覆盖生产者的压缩设置（不推荐）                 │
│  compression.type=producer  # 使用生产者指定的压缩（推荐）   │
│  compression.type=zstd      # 强制使用 zstd 压缩            │
│                                                             │
│  推荐：让生产者决定压缩方式                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

端到端压缩过程：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Producer 收集一批消息                                   │
│  2. 对整个批次进行压缩                                      │
│  3. 发送压缩后的批次到 Broker                               │
│  4. Broker 存储压缩后的数据                                 │
│  5. Consumer 读取压缩数据                                   │
│  6. Consumer 解压整个批次                                   │
│  7. Consumer 处理单条消息                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 6. 生产者高性能如何配置？

**答案：**

**生产者高性能配置详解：**

```java
Properties props = new Properties();

// ============ 核心性能配置 ============

// 1. 批次大小 - 影响吞吐量
props.put("batch.size", 65536);          // 64KB，根据消息大小调整

// 2. 等待时间 - 影响延迟
props.put("linger.ms", 10);              // 等待 10ms 凑批

// 3. 缓冲区大小 - 影响突发流量处理能力
props.put("buffer.memory", 67108864);    // 64MB

// 4. 压缩类型 - 影响网络和存储
props.put("compression.type", "lz4");    // 或 zstd

// 5. ACKS 配置 - 影响可靠性和性能的平衡
props.put("acks", "1");                   // Leader 确认即可

// 6. 并发请求数 - 影响吞吐量
props.put("max.in.flight.requests.per.connection", 5);

// 7. 请求超时
props.put("request.timeout.ms", 30000);
props.put("delivery.timeout.ms", 120000);

// 8. 重试配置
props.put("retries", 3);
props.put("retry.backoff.ms", 100);

// 9. 幂等性（保证顺序时需要限制并发请求）
props.put("enable.idempotence", "true");  // 自动设置 max.in.flight=5
```

**性能调优建议：**

```
┌─────────────────────────────────────────────────────────────┐
│                    场景化配置建议                            │
├─────────────────┬───────────────────────────────────────────┤
│                 │                                           │
│  高吞吐场景     │  batch.size=128KB                         │
│  (日志收集)     │  linger.ms=20                             │
│                 │  compression.type=zstd                    │
│                 │  acks=1                                   │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│  低延迟场景     │  batch.size=16KB                          │
│  (在线交易)     │  linger.ms=0                              │
│                 │  compression.type=none                    │
│                 │  acks=1                                   │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│  高可靠场景     │  batch.size=32KB                          │
│  (金融交易)     │  linger.ms=5                              │
│                 │  compression.type=zstd                    │
│                 │  acks=all                                 │
│                 │  enable.idempotence=true                  │
│                 │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

**异步发送回调：**

```java
// 高性能异步发送
public class HighThroughputProducer {
    private final KafkaProducer<String, String> producer;
    private final AtomicLong sentCount = new AtomicLong(0);
    private final AtomicLong errorCount = new AtomicLong(0);
    
    public void sendAsync(String topic, String key, String value) {
        ProducerRecord<String, String> record = 
            new ProducerRecord<>(topic, key, value);
        
        // 异步回调，不阻塞发送线程
        producer.send(record, (metadata, exception) -> {
            if (exception != null) {
                errorCount.incrementAndGet();
                // 异常处理：记录日志、发送告警等
                handleError(exception, record);
            } else {
                sentCount.incrementAndGet();
            }
        });
    }
    
    // 批量发送优化
    public void sendBatch(String topic, List<Message> messages) {
        for (Message msg : messages) {
            producer.send(new ProducerRecord<>(topic, msg.getKey(), msg.getValue()));
        }
        // 不需要每次都 flush，让 Kafka 自动批量发送
    }
    
    // 优雅关闭
    public void close() {
        producer.flush();  // 确保所有消息发送完成
        producer.close(Duration.ofSeconds(30));
    }
}
```

**追问：buffer.memory 满了会怎样？**

**追问答案：**

```
缓冲区满的处理流程：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Producer 发送消息                                          │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RecordAccumulator                       │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              buffer.memory                   │   │   │
│  │  │  ┌────────────────────────────────────────┐ │   │   │
│  │  │  │ Batch1 │ Batch2 │ Batch3 │ ... │ Full │ │   │   │
│  │  │  └────────────────────────────────────────┘ │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│                   缓冲区已满？                               │
│                         │                                   │
│           ┌─────────────┴─────────────┐                    │
│           │                           │                    │
│           ▼                           ▼                    │
│         否                          是                      │
│           │                           │                    │
│           ▼                           ▼                    │
│     加入缓冲区              max.block.ms 内阻塞等待         │
│                               (默认 60s)                   │
│                                     │                       │
│                          ┌──────────┴──────────┐          │
│                          │                     │          │
│                          ▼                     ▼          │
│                     缓冲区释放            超时抛出          │
│                     (消息发送完成)      BufferExhaustedException │
│                                                             │
└─────────────────────────────────────────────────────────────┘

解决方案：
1. 增大 buffer.memory
2. 减小 batch.size 或 linger.ms，加快发送
3. 增加 max.block.ms（不推荐）
4. 监控 buffer-available-bytes 指标
```

---

### 7. 消费者高性能如何配置？

**答案：**

**消费者高性能配置：**

```java
Properties props = new Properties();

// ============ 核心性能配置 ============

// 1. 单次拉取最小字节数 - 影响吞吐量
props.put("fetch.min.bytes", 1024);      // 默认 1B，建议 1KB-1MB

// 2. 单次拉取最大字节数 - 影响吞吐量
props.put("fetch.max.bytes", 52428800);  // 默认 50MB

// 3. 单个分区最大拉取字节数
props.put("max.partition.fetch.bytes", 1048576); // 默认 1MB

// 4. 拉取等待时间 - 影响延迟
props.put("fetch.max.wait.ms", 500);     // 默认 500ms

// 5. 心跳间隔 - 影响消费组稳定性
props.put("heartbeat.interval.ms", 3000); // 默认 3s

// 6. 会话超时 - 影响故障检测
props.put("session.timeout.ms", 30000);  // 默认 10s

// 7. 最大轮询间隔 - 影响处理超时检测
props.put("max.poll.interval.ms", 300000); // 默认 5 分钟

// 8. 单次轮询最大记录数 - 影响批量处理
props.put("max.poll.records", 500);      // 默认 500

// 9. 连接缓冲区
props.put("receive.buffer.bytes", 65536);
props.put("send.buffer.bytes", 131072);
```

**高性能消费者实现：**

```java
public class HighThroughputConsumer {
    private final KafkaConsumer<String, String> consumer;
    private final ExecutorService executor;
    
    public void consume(String topic) {
        consumer.subscribe(Collections.singletonList(topic));
        
        while (true) {
            // 批量拉取消息
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(1000));
            
            if (records.isEmpty()) continue;
            
            // 方案1：批量处理（单线程）
            processBatch(records);
            
            // 方案2：多线程处理
            // List<Future<?>> futures = new ArrayList<>();
            // for (ConsumerRecord<String, String> record : records) {
            //     futures.add(executor.submit(() -> process(record)));
            // }
            // waitForCompletion(futures);
            
            // 提交 offset
            consumer.commitSync();
        }
    }
    
    private void processBatch(ConsumerRecords<String, String> records) {
        // 批量处理优化
        List<String> values = new ArrayList<>();
        for (ConsumerRecord<String, String> record : records) {
            values.add(record.value());
        }
        
        // 批量写入数据库/缓存
        batchInsert(values);
    }
}
```

**多线程消费模式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    多线程消费模式对比                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式1：单消费者 + 多线程处理（推荐）                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Consumer Thread                                     │   │
│  │  ┌──────────┐                                       │   │
│  │  │ Consumer │──poll──▶ records                      │   │
│  │  └──────────┘            │                          │   │
│  │                          ▼                          │   │
│  │                    ┌─────────┐                      │   │
│  │                    │ 分发器   │                      │   │
│  │                    └─────────┘                      │   │
│  │                     │  │  │                         │   │
│  │            ┌────────┼──┼──┼────────┐               │   │
│  │            ▼        ▼  ▼  ▼        ▼               │   │
│  │        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │   │
│  │        │Worker│ │Worker│ │Worker│ │Worker│        │   │
│  │        │  1   │ │  2   │ │  3   │ │  4   │        │   │
│  │        └──────┘ └──────┘ └──────┘ └──────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│  优点：简单，offset 管理简单                                 │
│  缺点：处理失败会影响整批消息                                │
│                                                             │
│  模式2：多消费者实例                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Consumer Group                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │Consumer 1│ │Consumer 2│ │Consumer 3│            │   │
│  │  │ Partition│ │ Partition│ │ Partition│            │   │
│  │  │    0     │ │    1     │ │    2     │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  │       │            │            │                   │   │
│  │       ▼            ▼            ▼                   │   │
│  │   ┌──────┐    ┌──────┐    ┌──────┐                │   │
│  │   │Thread│    │Thread│    │Thread│                │   │
│  │   └──────┘    └──────┘    └──────┘                │   │
│  └─────────────────────────────────────────────────────┘   │
│  优点：真正的并行消费                                        │
│  缺点：消费者数不能超过分区数                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：fetch.min.bytes 和 fetch.max.wait.ms 怎么权衡？**

**追问答案：**

```
两个参数的配合：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  fetch.min.bytes=1MB, fetch.max.wait.ms=500ms              │
│                                                             │
│  情况1：数据充足                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 时间轴: 0ms ──────────────▶ 100ms                   │   │
│  │        │                             │              │   │
│  │        ▼                             ▼              │   │
│  │     开始拉取           累积 1MB，立即返回            │   │
│  │                                                     │   │
│  │  效果：低延迟，高吞吐                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  情况2：数据不足                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 时间轴: 0ms ─────────────────────────▶ 500ms        │   │
│  │        │                                       │    │   │
│  │        ▼                                       ▼    │   │
│  │     开始拉取                       等待超时，返回   │   │
│  │                                    当前数据(100KB)  │   │
│  │                                                     │   │
│  │  效果：最多等待 500ms                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

场景化配置：

┌─────────────────┬───────────────────────────────────────────┐
│      场景        │               配置建议                    │
├─────────────────┼───────────────────────────────────────────┤
│  高吞吐场景      │  fetch.min.bytes=1MB                     │
│  (日志处理)      │  fetch.max.wait.ms=1000                  │
│                 │  等待凑够批次再返回                        │
├─────────────────┼───────────────────────────────────────────┤
│  低延迟场景      │  fetch.min.bytes=1                       │
│  (在线服务)      │  fetch.max.wait.ms=100                   │
│                 │  有数据立即返回                            │
├─────────────────┼───────────────────────────────────────────┤
│  平衡场景        │  fetch.min.bytes=100KB                   │
│                 │  fetch.max.wait.ms=500                    │
│                 │  平衡延迟和吞吐                            │
└─────────────────┴───────────────────────────────────────────┘
```

---

### 8. 如何进行 Kafka 性能测试？

**答案：**

**使用 Kafka 自带工具：**

```bash
# 生产者性能测试
kafka-producer-perf-test.sh \
  --topic perf-test \
  --num-records 1000000 \
  --record-size 1024 \
  --throughput -1 \
  --producer-props \
    bootstrap.servers=localhost:9092 \
    compression.type=lz4 \
    batch.size=65536

# 输出示例：
# 1000000 records sent, 125000.0 records/sec (122.07 MB/sec)
# 8.00 ms avg latency, 250.00 ms max latency
# 0 records sent in 0-1 ms, 500000 in 1-10 ms, ...

# 消费者性能测试
kafka-consumer-perf-test.sh \
  --topic perf-test \
  --messages 1000000 \
  --bootstrap-server localhost:9092 \
  --group test-group

# 输出示例：
# start.time, end.time, data.consumed.in.MB, MB.sec, 
# data.consumed.in.nMsg, nMsg.sec
# 2024-01-01 10:00:00, 2024-01-01 10:00:10, 976.56, 97.66, 
# 1000000, 100000
```

**自定义性能测试：**

```java
public class KafkaPerformanceTest {
    
    public void testProducer(ProducerConfig config) {
        KafkaProducer<String, String> producer = new KafkaProducer<>(config.props);
        
        long startTime = System.currentTimeMillis();
        long totalBytes = 0;
        int messageCount = config.messageCount;
        int messageSize = config.messageSize;
        String payload = generatePayload(messageSize);
        
        for (int i = 0; i < messageCount; i++) {
            producer.send(new ProducerRecord<>(config.topic, 
                String.valueOf(i), payload), (metadata, exception) -> {
                if (exception == null) {
                    totalBytes += messageSize;
                }
            });
        }
        
        producer.flush();
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.println("━━━━━━ 性能测试结果 ━━━━━━");
        System.out.printf("消息数量: %d%n", messageCount);
        System.out.printf("消息大小: %d bytes%n", messageSize);
        System.out.printf("总耗时: %d ms%n", elapsed);
        System.out.printf("吞吐量: %.2f 条/秒%n", 
            messageCount * 1000.0 / elapsed);
        System.out.printf("吞吐量: %.2f MB/秒%n", 
            totalBytes / elapsed / 1000.0);
        System.out.printf("平均延迟: %.2f ms%n", 
            elapsed * 1.0 / messageCount);
        
        producer.close();
    }
    
    public void testConsumer(ConsumerConfig config) {
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(config.props);
        consumer.subscribe(Collections.singletonList(config.topic));
        
        long startTime = System.currentTimeMillis();
        long totalBytes = 0;
        int messageCount = 0;
        int targetCount = config.messageCount;
        
        while (messageCount < targetCount) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(1000));
            
            for (ConsumerRecord<String, String> record : records) {
                messageCount++;
                totalBytes += record.value().length();
            }
        }
        
        long elapsed = System.currentTimeMillis() - startTime;
        
        System.out.println("━━━━━━ 消费者性能 ━━━━━━");
        System.out.printf("消费消息: %d%n", messageCount);
        System.out.printf("消费耗时: %d ms%n", elapsed);
        System.out.printf("消费速度: %.2f 条/秒%n", 
            messageCount * 1000.0 / elapsed);
        
        consumer.close();
    }
}
```

**性能监控指标：**

```
┌─────────────────────────────────────────────────────────────┐
│                    关键性能指标                              │
├─────────────────────┬───────────────────────────────────────┤
│       指标           │               说明                    │
├─────────────────────┼───────────────────────────────────────┤
│  out-byte-rate      │  发送字节速率                         │
│  out-record-rate    │  发送消息速率                         │
│  io-wait-time-ns    │  I/O 等待时间                         │
│  batch-size-avg     │  平均批次大小                         │
│  record-send-rate   │  消息发送速率                         │
│  request-rate       │  请求发送速率                         │
│  response-rate      │  响应接收速率                         │
├─────────────────────┼───────────────────────────────────────┤
│  Consumer 指标       │                                       │
├─────────────────────┼───────────────────────────────────────┤
│  bytes-consumed-rate│  消费字节速率                         │
│  records-consumed-rate│ 消费消息速率                        │
│  fetch-rate         │  拉取请求速率                         │
│  fetch-latency-avg  │  拉取延迟                             │
└─────────────────────┴───────────────────────────────────────┘
```

**追问：如何定位 Kafka 性能瓶颈？**

**追问答案：**

```
性能瓶颈排查流程：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Producer 端                                             │
│     ├── 检查 batch-size-avg（批次是否足够大）               │
│     ├── 检查 record-send-rate（发送速率）                   │
│     ├── 检查 io-wait-time（I/O 等待时间）                   │
│     └── 检查 buffer-available-bytes（缓冲区剩余）           │
│                                                             │
│  2. Network                                                  │
│     ├── 检查网络带宽利用率                                   │
│     ├── 检查 TCP 重传率                                     │
│     └── 检查网络延迟                                        │
│                                                             │
│  3. Broker 端                                               │
│     ├── 检查 disk I/O（磁盘读写速率）                       │
│     ├── 检查 CPU 使用率                                     │
│     ├── 检查 JVM GC 情况                                    │
│     ├── 检查 Page Cache 命中率                              │
│     └── 检查 Under Replicated Partitions                    │
│                                                             │
│  4. Consumer 端                                             │
│     ├── 检查 fetch-latency-avg（拉取延迟）                  │
│     ├── 检查 records-consumed-rate（消费速率）              │
│     ├── 检查 lag（消费延迟）                                │
│     └── 检查处理线程是否阻塞                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

常见瓶颈及解决方案：

┌─────────────────┬───────────────────────────────────────────┐
│     瓶颈        │               解决方案                    │
├─────────────────┼───────────────────────────────────────────┤
│  磁盘 I/O       │  增加分区数，使用 SSD                     │
│  网络带宽       │  启用压缩，增加带宽                       │
│  CPU（压缩）    │  使用更快的压缩算法（lz4）                │
│  内存           │  增加 JVM 堆内存和 Page Cache             │
│  Consumer Lag   │  增加消费者数量或分区数                   │
└─────────────────┴───────────────────────────────────────────┘
```


---

## 消息可靠性篇

### 1. 怎么保证消息不丢失？

**答案：**

**消息丢失的三个环节：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消息传递链路                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐│
│  │ Producer │───▶│  Broker  │───▶│  Broker  │───▶│Consumer││
│  │ (生产者) │    │ (Leader) │    │(Follower)│    │(消费者)││
│  └──────────┘    └──────────┘    └──────────┘    └───────┘│
│       │               │               │              │     │
│       ▼               ▼               ▼              ▼     │
│   可能丢失         可能丢失        可能丢失       可能丢失   │
│   网络故障         宕机未同步      宕机未同步     处理失败   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**完整可靠性保证方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                 各环节可靠性保证                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Producer 端：                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. acks=all (等待所有 ISR 副本确认)                  │   │
│  │ 2. retries=3+ (自动重试)                            │   │
│  │ 3. enable.idempotence=true (幂等性)                 │   │
│  │ 4. 回调确认发送成功                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Broker 端：                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. replication.factor >= 3 (多副本)                 │   │
│  │ 2. min.insync.replicas >= 2 (最小同步副本数)        │   │
│  │ 3. unclean.leader.election.enable=false (禁止脏选举)│   │
│  │ 4. 多 Broker 分布在不同机架/机房                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Consumer 端：                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. enable.auto.commit=false (手动提交 offset)       │   │
│  │ 2. 先处理消息，再提交 offset                         │   │
│  │ 3. 实现幂等性处理                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**完整配置代码：**

```java
// Producer 端可靠性配置
Properties producerProps = new Properties();
producerProps.put("bootstrap.servers", "broker1:9092,broker2:9092,broker3:9092");
producerProps.put("acks", "all");                    // 最可靠
producerProps.put("retries", Integer.MAX_VALUE);     // 无限重试
producerProps.put("max.in.flight.requests.per.connection", 5); // 幂等性限制
producerProps.put("enable.idempotence", true);       // 开启幂等性
producerProps.put("delivery.timeout.ms", 120000);    // 发送超时

// 发送消息并确认
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        // 发送失败，记录日志并重试
        log.error("发送失败: {}", exception.getMessage());
        // 可能需要重试或存储到死信队列
    }
});

// Consumer 端可靠性配置
Properties consumerProps = new Properties();
consumerProps.put("bootstrap.servers", "broker1:9092,broker2:9092,broker3:9092");
consumerProps.put("group.id", "my-group");
consumerProps.put("enable.auto.commit", false);      // 手动提交
consumerProps.put("isolation.level", "read_committed"); // 只读已提交

// 消费逻辑
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        try {
            processMessage(record);      // 先处理
            consumer.commitSync();       // 处理成功后提交
        } catch (Exception e) {
            // 处理失败，不提交 offset，下次重新消费
            log.error("处理失败，等待重试", e);
        }
    }
}
```

**追问：acks=all 和 min.insync.replicas 是什么关系？**

**追问答案：**

```
两者配合确保数据可靠性：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  场景：Topic 配置 replication.factor=3, min.insync.replicas=2│
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Topic Partition                   │   │
│  │                                                     │   │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐        │   │
│  │   │ Leader  │    │Follower │    │Follower │        │   │
│  │   │ Broker1 │    │ Broker2 │    │ Broker3 │        │   │
│  │   │   ISR   │    │   ISR   │    │  OSR    │        │   │
│  │   └─────────┘    └─────────┘    └─────────┘        │   │
│  │        │              │                             │   │
│  │        └──────┬───────┘                             │   │
│  │               ▼                                     │   │
│  │         Leader + 1 Follower = 2 个副本              │   │
│  │         满足 min.insync.replicas=2                  │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  acks=all 的含义：                                          │
│  - 等待所有 ISR 中的副本确认                                │
│  - 不是所有副本（ISR + OSR）                               │
│                                                             │
│  min.insync.replicas 的含义：                               │
│  - ISR 中最少需要多少个副本                                 │
│  - 如果 ISR 数量 < min.insync.replicas，生产者会收到异常    │
│                                                             │
│  配合效果：                                                 │
│  - min.insync.replicas=2 保证至少有 2 个副本同步           │
│  - acks=all 确保这 2 个副本都确认后才返回成功               │
│  - 即使 1 个 Broker 宕机，数据仍然安全                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

配置建议：
- 生产环境：replication.factor=3, min.insync.replicas=2, acks=all
- 可容忍 1 个节点故障，数据不丢失
```

---

### 2. 生产者端怎么保证不丢消息？

**答案：**

**生产者端可靠性配置详解：**

```java
// 生产者端完整可靠性配置
Properties props = new Properties();

// ============ 核心可靠性参数 ============

// 1. acks - 确认机制
// 0: 不等待确认（最快，可能丢失）
// 1: 等待 Leader 确认（默认，Leader 宕机可能丢失）
// all/-1: 等待所有 ISR 副本确认（最可靠）
props.put("acks", "all");

// 2. retries - 重试次数
props.put("retries", Integer.MAX_VALUE);

// 3. 重试间隔
props.put("retry.backoff.ms", 100);

// 4. 发送超时（包括重试时间）
props.put("delivery.timeout.ms", 120000);

// 5. 幂等性 - 防止重试导致的重复
props.put("enable.idempotence", true);

// 6. 事务支持 - 跨分区原子写入
// props.put("transactional.id", "my-txn-id");

// 7. 请求超时
props.put("request.timeout.ms", 30000);
```

**可靠发送实现：**

```java
public class ReliableProducer {
    private final KafkaProducer<String, String> producer;
    
    // 同步发送（最可靠）
    public void sendSync(String topic, String key, String value) {
        ProducerRecord<String, String> record = 
            new ProducerRecord<>(topic, key, value);
        
        try {
            RecordMetadata metadata = producer.send(record).get();
            log.info("发送成功: partition={}, offset={}", 
                metadata.partition(), metadata.offset());
        } catch (InterruptedException | ExecutionException e) {
            log.error("发送失败", e);
            // 处理失败：重试、存储死信队列、告警等
            handleSendFailure(record, e);
        }
    }
    
    // 异步发送（带回调）
    public void sendAsync(String topic, String key, String value) {
        ProducerRecord<String, String> record = 
            new ProducerRecord<>(topic, key, value);
        
        producer.send(record, new Callback() {
            @Override
            public void onCompletion(RecordMetadata metadata, Exception exception) {
                if (exception != null) {
                    log.error("发送失败", exception);
                    // 关键：回调中处理失败
                    handleSendFailure(record, exception);
                } else {
                    log.debug("发送成功: partition={}, offset={}", 
                        metadata.partition(), metadata.offset());
                }
            }
        });
    }
    
    // 带重试的发送
    public void sendWithRetry(String topic, String key, String value, int maxRetries) {
        ProducerRecord<String, String> record = 
            new ProducerRecord<>(topic, key, value);
        
        int retryCount = 0;
        while (retryCount < maxRetries) {
            try {
                producer.send(record).get();
                return; // 成功则返回
            } catch (Exception e) {
                retryCount++;
                log.warn("发送失败，第 {} 次重试", retryCount, e);
                
                if (retryCount >= maxRetries) {
                    // 超过最大重试次数，存储到死信队列或数据库
                    saveToDeadLetterQueue(record, e);
                }
            }
        }
    }
    
    // 事务发送（跨分区原子性）
    public void sendTransactional(List<ProducerRecord<String, String>> records) {
        producer.beginTransaction();
        try {
            for (ProducerRecord<String, String> record : records) {
                producer.send(record);
            }
            producer.commitTransaction();
        } catch (Exception e) {
            producer.abortTransaction();
            log.error("事务发送失败", e);
        }
    }
}
```

**追问：max.in.flight.requests.per.connection 和可靠性有什么关系？**

**追问答案：**

```
参数作用：

max.in.flight.requests.per.connection 控制单个连接上最多可以有多个未确认的请求。

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  场景1：max.in.flight=5, enable.idempotence=false           │
│                                                             │
│  发送顺序: msg1 → msg2 → msg3 → msg4 → msg5                │
│           ───────────────────────────────▶                  │
│                                                             │
│  msg1 失败重试，msg2 成功                                    │
│                                                             │
│  结果: msg2, msg3, msg4, msg5, msg1 (msg1 乱序)             │
│                                                             │
│  问题：如果 msg1~msg5 是同一订单的事件，顺序被打乱！          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景2：enable.idempotence=true                             │
│                                                             │
│  Kafka 自动设置 max.in.flight=5                            │
│  并为每条消息分配序列号                                      │
│                                                             │
│  Broker 端会检测并拒绝重复的序列号                           │
│  保证消息顺序和去重                                          │
│                                                             │
│  结果：即使重试也能保证顺序                                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景3：需要严格顺序，且不使用幂等性                         │
│                                                             │
│  设置 max.in.flight.requests.per.connection=1              │
│  每次只能有一个未确认请求                                    │
│                                                             │
│  缺点：吞吐量大幅下降                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

推荐配置：
- 启用幂等性：enable.idempotence=true（自动管理 max.in.flight）
- 需要严格顺序：使用幂等性 + 单分区
```

---

### 3. 消费者端怎么保证不丢消息？

**答案：**

**消费者端可靠性要点：**

```
┌─────────────────────────────────────────────────────────────┐
│                消费者端消息丢失场景                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  错误做法：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 自动提交 offset                                    │   │
│  │ 2. 提交后处理消息                                     │   │
│  │ 3. 处理失败，但 offset 已提交                         │   │
│  │ 4. 消息丢失，下次不会重新消费                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  正确做法：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 手动提交 offset                                    │   │
│  │ 2. 先处理消息                                         │   │
│  │ 3. 处理成功后再提交 offset                            │   │
│  │ 4. 处理失败不提交，下次重新消费                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**可靠消费实现：**

```java
public class ReliableConsumer {
    private final KafkaConsumer<String, String> consumer;
    
    // ============ 手动提交 offset ============
    
    // 方式1：同步提交（最可靠）
    public void consumeWithSyncCommit() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                try {
                    // 先处理消息
                    processMessage(record);
                    
                    // 处理成功，同步提交 offset
                    consumer.commitSync(Collections.singletonMap(
                        new TopicPartition(record.topic(), record.partition()),
                        new OffsetAndMetadata(record.offset() + 1)
                    ));
                } catch (Exception e) {
                    // 处理失败，不提交 offset
                    log.error("处理消息失败: {}", record.value(), e);
                    // 可以选择重试或发送到死信队列
                }
            }
        }
    }
    
    // 方式2：批量处理 + 批量提交
    public void consumeWithBatchCommit() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            if (records.isEmpty()) continue;
            
            boolean allSuccess = true;
            for (ConsumerRecord<String, String> record : records) {
                try {
                    processMessage(record);
                } catch (Exception e) {
                    allSuccess = false;
                    log.error("处理消息失败", e);
                    break;
                }
            }
            
            if (allSuccess) {
                // 全部成功，提交最后一条消息的 offset
                consumer.commitSync();
            }
            // 如果失败，不提交 offset，下次重新消费
        }
    }
    
    // 方式3：异步提交 + 重试补偿
    public void consumeWithAsyncCommit() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                processMessage(record);
            }
            
            // 异步提交（不阻塞消费）
            consumer.commitAsync((offsets, exception) -> {
                if (exception != null) {
                    log.warn("异步提交失败，将在下次提交时重试", exception);
                    // 异步提交失败可以忽略，下次会重新提交
                }
            });
        }
    }
    
    // 方式4：处理失败发送到死信队列
    public void consumeWithDLQ() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                try {
                    processMessage(record);
                } catch (Exception e) {
                    log.error("处理失败，发送到死信队列", e);
                    // 发送到死信队列
                    sendToDeadLetterQueue(record, e);
                }
            }
            
            consumer.commitSync();
        }
    }
    
    private void sendToDeadLetterQueue(ConsumerRecord<String, String> record, 
                                        Exception error) {
        // 构造死信消息
        DeadLetterMessage dlqMessage = new DeadLetterMessage();
        dlqMessage.setOriginalTopic(record.topic());
        dlqMessage.setOriginalPartition(record.partition());
        dlqMessage.setOriginalOffset(record.offset());
        dlqMessage.setKey(record.key());
        dlqMessage.setValue(record.value());
        dlqMessage.setError(error.getMessage());
        dlqMessage.setTimestamp(System.currentTimeMillis());
        
        // 发送到死信 Topic
        ProducerRecord<String, String> dlqRecord = new ProducerRecord<>(
            record.topic() + ".DLQ",
            record.key(),
            serialize(dlqMessage)
        );
        deadLetterProducer.send(dlqRecord);
    }
}
```

**配置参数：**

```java
Properties props = new Properties();

// 关键配置
props.put("enable.auto.commit", "false");  // 禁用自动提交
props.put("auto.offset.reset", "earliest"); // 从最早开始消费

// 处理超时相关
props.put("max.poll.interval.ms", "300000");  // 两次 poll 最大间隔
props.put("max.poll.records", "100");         // 单次 poll 最大消息数

// 会话超时
props.put("session.timeout.ms", "30000");     // 会话超时
props.put("heartbeat.interval.ms", "10000");  // 心跳间隔
```

**追问：commitSync 和 commitAsync 怎么选择？**

**追问答案：**

```
对比分析：

┌─────────────────┬───────────────────────────────────────────┐
│    特性          │                 说明                      │
├─────────────────┼───────────────────────────────────────────┤
│ commitSync      │ 阻塞等待提交完成                          │
│                 │ 失败会自动重试                            │
│                 │ 确保提交成功                              │
│                 │ 影响消费吞吐量                            │
├─────────────────┼───────────────────────────────────────────┤
│ commitAsync     │ 非阻塞，立即返回                          │
│                 │ 失败不重试                                │
│                 │ 可能丢失提交                              │
│                 │ 不影响消费吞吐量                          │
└─────────────────┴───────────────────────────────────────────┘

推荐方案：组合使用

public void consumeWithMixedCommit() {
    try {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                processMessage(record);
            }
            
            // 正常消费时异步提交
            consumer.commitAsync();
        }
    } finally {
        try {
            // 关闭前同步提交，确保最后一次提交成功
            consumer.commitSync();
        } finally {
            consumer.close();
        }
    }
}
```

---

### 4. Broker 端怎么保证不丢消息？

**答案：**

**Broker 端可靠性机制：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Broker 可靠性保证                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 多副本机制                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │   Producer                                           │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │   ┌───────────┐                                     │   │
│  │   │  Leader   │ Broker 1                            │   │
│  │   │  ISR      │ ◀─── 写入成功                       │   │
│  │   └───────────┘                                     │   │
│  │        │                                             │   │
│  │        │ 同步复制                                    │   │
│  │        ▼                                             │   │
│  │   ┌───────────┐    ┌───────────┐                    │   │
│  │   │ Follower  │    │ Follower  │                    │   │
│  │   │  ISR      │    │   OSR     │                    │   │
│  │   │ Broker 2  │    │ Broker 3  │                    │   │
│  │   └───────────┘    └───────────┘                    │   │
│  │                                                     │   │
│  │  ISR (In-Sync Replicas): 同步副本集合               │   │
│  │  OSR (Out-of-Sync Replicas): 不同步副本             │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 禁止脏选举                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ unclean.leader.election.enable=false                │   │
│  │                                                     │   │
│  │ Leader 宕机时，只从 ISR 中选举新 Leader              │   │
│  │ 即使 ISR 为空也不选举 OSR                           │   │
│  │ 宁可不可用也不丢数据                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 最小同步副本                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ min.insync.replicas=2                               │   │
│  │                                                     │   │
│  │ ISR 中最少副本数，不足则拒绝写入                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Broker 配置：**

```properties
# ============ 副本配置 ============
# 默认副本数
default.replication.factor=3

# 最小同步副本数
min.insync.replicas=2

# 禁止不干净的 Leader 选举
unclean.leader.election.enable=false

# ============ 日志配置 ============
# 日志刷新策略（通常让 OS 管理）
# log.flush.interval.messages=10000
# log.flush.interval.ms=1000

# 日志保留
log.retention.hours=168
log.retention.bytes=-1
log.segment.bytes=1073741824

# ============ 副本同步配置 ============
# 副本拉取间隔
replica.lag.time.max.ms=30000

# 副本 Socket 缓冲区
replica.socket.receive.buffer.bytes=65536
replica.socket.timeout.ms=30000
```

**追问：ISR 是什么？如何维护？**

**追问答案：**

```
ISR (In-Sync Replicas) 机制：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ISR 定义：                                                 │
│  - 与 Leader 保持同步的副本集合                              │
│  - 由 Leader 维护                                           │
│  - 动态变化                                                  │
│                                                             │
│  同步判断标准：                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ replica.lag.time.max.ms=30000 (默认)                │   │
│  │                                                     │   │
│  │ Follower 在 30 秒内没有发送拉取请求                  │   │
│  │ 或者拉取后没有追上 Leader 的 LEO                     │   │
│  │ → 从 ISR 中移除                                      │   │
│  │                                                     │   │
│  │ Follower 追上 Leader 的 LEO                         │   │
│  │ → 加入 ISR                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ISR 变化示例：                                              │
│                                                             │
│  时间轴：                                                    │
│  T0: ISR=[0,1,2]  (Leader=0, Follower=1,2)                 │
│  │                                                          │
│  T1: Broker 2 网络故障                                       │
│  │   → Follower 2 超过 30s 未同步                          │
│  │   → ISR=[0,1]                                           │
│  │                                                          │
│  T2: Broker 2 恢复                                          │
│  │   → Follower 2 追赶 Leader                              │
│  │   → ISR=[0,1,2]                                         │
│  │                                                          │
│  T3: Leader 0 宕机                                          │
│  │   → 从 ISR=[1,2] 中选举新 Leader (假设选举 1)           │
│  │   → ISR=[1,2]                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

关键概念：

LEO (Log End Offset): 日志末端偏移量
- 下一条等待写入的消息的 offset
- 每个副本都有自己的 LEO

HW (High Watermark): 高水位
- ISR 中所有副本都已同步的消息 offset
- 消费者只能消费到 HW 之前的消息
- 确保数据一致性

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Leader: LEO=10, HW=8                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ msg0 │ msg1 │ ... │ msg7 │ msg8 │ msg9 │            │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ▲       ▲                      │
│                              │       └── LEO=10 (下一条)    │
│                              └── HW=8 (消费者可见)          │
│                                                             │
│  Follower 1: LEO=9                                          │
│  Follower 2: LEO=8                                          │
│                                                             │
│  HW = min(所有 ISR 副本的 LEO) = min(10, 9, 8) = 8         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. 什么是 ACKS？各有什么区别？

**答案：**

**ACKS 三种取值详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                    acks 取值对比                             │
├─────────────────┬───────────────────────────────────────────┤
│                 │                                           │
│  acks=0         │  不等待任何确认                            │
│                 │                                           │
│  ┌──────────┐   │  Producer                  Broker         │
│  │          │   │    │                         │           │
│  │ Producer │───┼────┼─────发送消息───────────▶│           │
│  │          │   │    │                         │           │
│  │   发送   │   │    │◀───立即返回成功─────────│           │
│  │   完成   │   │    │   (不等待 Broker 响应)  │           │
│  └──────────┘   │                                           │
│                 │  性能：最高                                │
│                 │  可靠性：最低（可能丢失）                   │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│  acks=1         │  等待 Leader 确认（默认）                   │
│                 │                                           │
│  ┌──────────┐   │  Producer                  Broker         │
│  │          │   │    │                       Leader         │
│  │ Producer │───┼────┼─────发送消息─────────▶┌───────┐     │
│  │          │   │    │                       │ 写入  │     │
│  │   等待   │   │    │                       │ 成功  │     │
│  │          │   │    │◀───确认成功───────────└───────┘     │
│  └──────────┘   │                                           │
│                 │  性能：中等                                │
│                 │  可靠性：中（Leader 宕机可能丢失）          │
│                 │                                           │
├─────────────────┼───────────────────────────────────────────┤
│                 │                                           │
│  acks=all/-1    │  等待所有 ISR 副本确认                      │
│                 │                                           │
│  ┌──────────┐   │  Producer                  Broker         │
│  │          │   │    │                       Leader         │
│  │ Producer │───┼────┼─────发送消息─────────▶┌───────┐     │
│  │          │   │    │                       │ 写入  │     │
│  │   等待   │   │    │                       └───┬───┘     │
│  │          │   │    │                           │ 同步    │
│  │          │   │    │        ┌──────────────────┤         │
│  │          │   │    │        ▼                  ▼         │
│  │          │   │    │   ┌─────────┐      ┌─────────┐     │
│  │          │   │    │   │Follower │      │Follower │     │
│  │          │   │    │   │  ISR    │      │  ISR    │     │
│  │          │   │    │   └─────────┘      └─────────┘     │
│  │          │   │    │                           │         │
│  │          │   │    │◀───────全部确认成功───────┘         │
│  └──────────┘   │                                           │
│                 │  性能：较低                                │
│                 │  可靠性：最高（配合 min.insync.replicas）   │
│                 │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

**性能对比：**

```java
// 性能测试代码
public class AckPerformanceTest {
    public void testAcks() {
        int[] acksValues = {0, 1, -1};
        int messageCount = 100000;
        
        for (int acks : acksValues) {
            Properties props = new Properties();
            props.put("acks", String.valueOf(acks));
            props.put("batch.size", 16384);
            props.put("linger.ms", 0);
            // ... 其他配置
            
            KafkaProducer<String, String> producer = new KafkaProducer<>(props);
            
            long start = System.currentTimeMillis();
            for (int i = 0; i < messageCount; i++) {
                producer.send(new ProducerRecord<>("test", "msg" + i)).get();
            }
            long elapsed = System.currentTimeMillis() - start;
            
            System.out.printf("acks=%d: %d ms, %.2f msg/s%n", 
                acks, elapsed, messageCount * 1000.0 / elapsed);
            
            producer.close();
        }
    }
}

// 典型输出：
// acks=0:  5000 ms, 20000.00 msg/s
// acks=1:  8000 ms, 12500.00 msg/s
// acks=-1: 15000 ms, 6666.67 msg/s
```

**追问：acks=all 时 ISR 只有一个副本怎么办？**

**追问答案：**

```
问题分析：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  场景：replication.factor=3, min.insync.replicas=2          │
│                                                             │
│  正常情况：ISR=[Leader, Follower1, Follower2]               │
│  acks=all 时需要等待 3 个副本确认                            │
│                                                             │
│  异常情况：                                                  │
│  Follower1 和 Follower2 都故障                              │
│  ISR=[Leader]                                               │
│                                                             │
│  问题：acks=all 只需要等待 1 个副本（Leader）                │
│  与 min.insync.replicas=2 的期望不符                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

解决方案：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  min.insync.replicas 参数的作用：                            │
│                                                             │
│  当 ISR 数量 < min.insync.replicas 时：                     │
│  - 生产者会收到 NotEnoughReplicas 异常                      │
│  - 写入操作被拒绝                                           │
│                                                             │
│  配置示例：                                                  │
│  replication.factor=3                                       │
│  min.insync.replicas=2                                      │
│  acks=all                                                   │
│                                                             │
│  保证：至少有 2 个副本同步成功才能返回成功                   │
│        如果 ISR 只有 1 个，则拒绝写入                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

// 生产者处理异常
try {
    producer.send(record).get();
} catch (ExecutionException e) {
    if (e.getCause() instanceof NotEnoughReplicasException) {
        // ISR 数量不足，需要告警或降级处理
        log.error("ISR 副本数量不足，写入被拒绝");
    }
}
```

---

### 6. 什么是消息重复？怎么解决？

**答案：**

**消息重复产生的原因：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消息重复场景                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景1：生产者重试                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Producer              Broker                        │   │
│  │    │                    │                           │   │
│  │    │───发送 msg1 ──────▶│ 写入成功                   │   │
│  │    │                    │                           │   │
│  │    │◀───响应丢失 ───────│ 网络故障                   │   │
│  │    │                    │                           │   │
│  │    │───重试 msg1 ──────▶│ 再次写入（重复）           │   │
│  │    │                    │                           │   │
│  │    │◀───响应成功 ───────│                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  场景2：消费者重复消费                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Consumer              Broker                        │   │
│  │    │                    │                           │   │
│  │    │◀───拉取消息 ───────│ msg1, msg2                │   │
│  │    │                    │                           │   │
│  │    │───处理消息 ────────│                           │   │
│  │    │   (处理成功)       │                           │   │
│  │    │                    │                           │   │
│  │    │───提交 offset ────▶│ 提交失败（网络故障）       │   │
│  │    │                    │                           │   │
│  │    │◀───重平衡 ─────────│ Consumer 重启             │   │
│  │    │                    │                           │   │
│  │    │◀───重新拉取 ───────│ msg1, msg2（重复消费）    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消息去重方案                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案1：生产者幂等性（单分区幂等）                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ enable.idempotence=true                              │   │
│  │                                                     │   │
│  │ Broker 为每个生产者维护序列号                        │   │
│  │ 相同序列号的消息只保留一份                           │   │
│  │                                                     │   │
│  │ 限制：只能保证单分区幂等                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案2：生产者事务（跨分区幂等）                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ transactional.id="unique-txn-id"                     │   │
│  │                                                     │   │
│  │ 支持跨多个分区的原子写入                             │   │
│  │ 可以实现精确一次语义                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案3：消费者端幂等（推荐）                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 消费者端实现去重逻辑：                               │   │
│  │ 1. 数据库唯一索引                                    │   │
│  │ 2. Redis 去重                                        │   │
│  │ 3. 业务状态机                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**消费者端幂等实现：**

```java
public class IdempotentConsumer {
    private final KafkaConsumer<String, String> consumer;
    private final RedisTemplate<String, String> redis;
    
    // 方式1：Redis 去重
    public void consumeWithRedisDedup() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                String messageId = extractMessageId(record);
                
                // Redis SETNX 原子操作
                Boolean isNew = redis.opsForValue()
                    .setIfAbsent("msg:" + messageId, "1", Duration.ofHours(24));
                
                if (isNew) {
                    // 新消息，处理
                    processMessage(record);
                } else {
                    // 重复消息，跳过
                    log.warn("重复消息，跳过: {}", messageId);
                }
            }
            
            consumer.commitSync();
        }
    }
    
    // 方式2：数据库唯一索引
    public void consumeWithDbDedup() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                try {
                    // 插入消息处理记录（有唯一索引）
                    insertMessageLog(record);
                    // 处理消息
                    processMessage(record);
                } catch (DuplicateKeyException e) {
                    // 唯一索引冲突，说明已处理过
                    log.warn("重复消息，跳过: {}", record.key());
                }
            }
            
            consumer.commitSync();
        }
    }
    
    // 方式3：业务状态机
    public void consumeWithStateMachine() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                OrderEvent event = parseOrderEvent(record);
                Order order = orderRepository.findById(event.getOrderId());
                
                // 状态机检查：只有特定状态才能转换
                if (order.canTransitionTo(event.getNewStatus())) {
                    order.transitionTo(event.getNewStatus());
                    orderRepository.save(order);
                } else {
                    log.warn("重复或非法状态转换，跳过: {}", event);
                }
            }
            
            consumer.commitSync();
        }
    }
    
    private String extractMessageId(ConsumerRecord<String, String> record) {
        // 从消息头或消息体中提取唯一 ID
        Header idHeader = record.headers().lastHeader("messageId");
        if (idHeader != null) {
            return new String(idHeader.value());
        }
        // 使用 topic + partition + offset 作为 ID
        return String.format("%s-%d-%d", 
            record.topic(), record.partition(), record.offset());
    }
}
```

**追问：Exactly-Once 怎么保证？**

**追问答案：**

```
Exactly-Once 完整方案：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  端到端 Exactly-Once = 幂等生产者 + 事务 + 幂等消费者         │
│                                                             │
│  1. 生产者端：                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ // 配置                                              │   │
│  │ props.put("enable.idempotence", true);              │   │
│  │ props.put("transactional.id", "unique-txn-id");     │   │
│  │                                                     │   │
│  │ // 事务发送                                          │   │
│  │ producer.initTransactions();                        │   │
│  │ producer.beginTransaction();                        │   │
│  │ producer.send(record1);                             │   │
│  │ producer.send(record2);                             │   │
│  │ producer.commitTransaction();                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 消费者端：                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ // 配置                                              │   │
│  │ props.put("isolation.level", "read_committed");     │   │
│  │                                                     │   │
│  │ // 只读取已提交的事务消息                            │   │
│  │ // 配合幂等消费逻辑                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Kafka Streams（推荐）：                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ // 配置                                              │   │
│  │ props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG,│   │
│  │           StreamsConfig.EXACTLY_ONCE_V2);           │   │
│  │                                                     │   │
│  │ // Kafka Streams 自动处理事务和 offset              │   │
│  │ // 实现端到端精确一次                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Kafka Streams 示例：

Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "my-app");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, 
          StreamsConfig.EXACTLY_ONCE_V2);

StreamsBuilder builder = new StreamsBuilder();
builder.stream("input-topic")
       .mapValues(value -> process(value))
       .to("output-topic");

KafkaStreams streams = new KafkaStreams(builder.build(), props);
streams.start();
```

---

### 7. 幂等性是怎么实现的？

**答案：**

**幂等性实现原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    生产者幂等性原理                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心概念：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PID (Producer ID): 生产者唯一标识                    │   │
│  │ Sequence Number: 每条消息的序列号                     │   │
│  │ Epoch: 生产者年代（用于检测僵尸实例）                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  工作流程：                                                  │
│                                                             │
│  1. 生产者初始化                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Producer              Broker (Coordinator)          │   │
│  │    │                         │                      │   │
│  │    │──InitProducerId───────▶│                       │   │
│  │    │                         │ 分配 PID + Epoch     │   │
│  │    │◀──PID=100,Epoch=0──────│                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 发送消息（带序列号）                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Producer              Broker (Leader)               │   │
│  │    │                         │                      │   │
│  │    │──Send(PID=100,Seq=0)──▶│ 保存 (PID, Seq)      │   │
│  │    │                         │                      │   │
│  │    │◀──Ack──────────────────│                       │   │
│  │    │                         │                      │   │
│  │    │──Send(PID=100,Seq=1)──▶│ 保存 (PID, Seq)      │   │
│  │    │   (网络故障，超时)      │                      │   │
│  │    │                         │                      │   │
│  │    │──Retry(PID=100,Seq=1)─▶│ 检测到重复 Seq       │   │
│  │    │                         │ 丢弃重复消息         │   │
│  │    │◀──Ack──────────────────│                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Broker 端状态维护：                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 每个分区维护：                                       │   │
│  │ Key: <PID, Partition>                               │   │
│  │ Value: <Epoch, LastSequence>                        │   │
│  │                                                     │   │
│  │ 去重逻辑：                                           │   │
│  │ if (seq == lastSeq + 1) 正常接收                    │   │
│  │ if (seq == lastSeq)     重复，丢弃                  │   │
│  │ if (seq < lastSeq)      过期，拒绝                  │   │
│  │ if (seq > lastSeq + 1)  有缺口，拒绝                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**幂等性配置：**

```java
// 开启幂等性（Kafka 0.11+）
Properties props = new Properties();
props.put("enable.idempotence", true);  // 一行配置即可

// 幂等性开启后，以下配置会自动设置：
// max.in.flight.requests.per.connection = 5 (如果未设置)
// retries = Integer.MAX_VALUE (如果未设置)
// acks = all (如果未设置)

// 注意：幂等性只保证单分区内的去重
// 如果需要跨分区原子性，需要使用事务
```

**限制说明：**

```
┌─────────────────────────────────────────────────────────────┐
│                    幂等性限制                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 单分区幂等                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 同一消息发送到多个分区，无法保证去重                  │   │
│  │ 需要使用事务来保证跨分区原子性                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 单会话幂等                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PID 在 Broker 端维护                                 │   │
│  │ 生产者重启后获得新的 PID                              │   │
│  │ 无法跨会话去重                                       │   │
│  │                                                     │   │
│  │ 解决：使用事务 + transactional.id                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 序列号缺口                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 如果消息 0 和 2 发送成功，但消息 1 失败               │   │
│  │ Broker 会拒绝消息 2（有序列号缺口）                   │   │
│  │                                                     │   │
│  │ 解决：max.in.flight.requests 限制并发请求数          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：事务是如何实现的？**

**追问答案：**

```
Kafka 事务实现原理：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  核心组件：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Transaction Coordinator (事务协调器)              │   │
│  │    - 每个 Producer 有一个专门的协调器                │   │
│  │    - 负责管理事务状态                                │   │
│  │                                                     │   │
│  │ 2. __transaction_state (内部 Topic)                 │   │
│  │    - 存储事务状态                                    │   │
│  │    - 类似 __consumer_offsets                        │   │
│  │                                                     │   │
│  │ 3. transactional.id                                 │   │
│  │    - 生产者的唯一标识                                │   │
│  │    - 用于跨会话恢复事务状态                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  事务流程：                                                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 初始化事务                                        │   │
│  │    producer.initTransactions();                      │   │
│  │    → 向协调器注册，获取 PID 和 Epoch                 │   │
│  │                                                     │   │
│  │ 2. 开始事务                                          │   │
│  │    producer.beginTransaction();                      │   │
│  │    → 协调器记录事务开始                              │   │
│  │                                                     │   │
│  │ 3. 发送消息                                          │   │
│  │    producer.send(record);                            │   │
│  │    → 消息带有事务标记（未提交）                       │   │
│  │                                                     │   │
│  │ 4. 提交事务                                          │   │
│  │    producer.commitTransaction();                     │   │
│  │    → 两阶段提交：                                    │   │
│  │      a. 协调器写入 Prepared 状态                     │   │
│  │      b. 向所有分区写入 Commit 标记                   │   │
│  │      c. 协调器写入 Complete 状态                     │   │
│  │                                                     │   │
│  │ 5. 回滚事务                                          │   │
│  │    producer.abortTransaction();                      │   │
│  │    → 向所有分区写入 Abort 标记                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  消费者过滤：                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ isolation.level=read_committed                       │   │
│  │                                                     │   │
│  │ 消费者只返回已提交的事务消息                         │   │
│  │ 过滤掉已回滚的事务消息                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

// 事务代码示例
public class TransactionalProducer {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("transactional.id", "my-transactional-id");
        props.put("key.serializer", StringSerializer.class.getName());
        props.put("value.serializer", StringSerializer.class.getName());
        
        KafkaProducer<String, String> producer = new KafkaProducer<>(props);
        producer.initTransactions();
        
        try {
            producer.beginTransaction();
            
            // 发送多条消息到多个分区/Topic
            producer.send(new ProducerRecord<>("topic1", "key1", "value1"));
            producer.send(new ProducerRecord<>("topic2", "key2", "value2"));
            
            // 原子性提交
            producer.commitTransaction();
        } catch (Exception e) {
            producer.abortTransaction();
        } finally {
            producer.close();
        }
    }
}
```


---

## 消费者篇

### 1. 消费者组是什么？

**答案：**

**消费者组概念：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消费者组机制                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  消费者组（Consumer Group）特性：                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 组内消费者分摊分区（负载均衡）                     │   │
│  │ 2. 组间消费者独立消费（广播模式）                     │   │
│  │ 3. 组内消费者共享 Group ID                           │   │
│  │ 4. 每个分区只能被组内一个消费者消费                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  图解：一个 Topic 被多个消费者组消费                         │
│                                                             │
│  Topic (3 个分区)                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐                │   │
│  │ │Partition│ │Partition│ │Partition│                │   │
│  │ │    0    │ │    1    │ │    2    │                │   │
│  │ └─────────┘ └─────────┘ └─────────┘                │   │
│  └─────────────────────────────────────────────────────┘   │
│        │           │           │                            │
│        ▼           ▼           ▼                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Consumer Group A                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │Consumer 1│ │Consumer 2│ │Consumer 3│            │   │
│  │  │  Part 0  │ │  Part 1  │ │  Part 2  │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  │                                                     │   │
│  │  每个消费者消费一个分区                              │   │
│  └─────────────────────────────────────────────────────┘   │
│        │           │           │                            │
│        ▼           ▼           ▼                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Consumer Group B                        │   │
│  │  ┌──────────────────────────────────────┐          │   │
│  │  │           Consumer 1                  │          │   │
│  │  │      Part 0 + Part 1 + Part 2         │          │   │
│  │  └──────────────────────────────────────┘          │   │
│  │                                                     │   │
│  │  一个消费者消费所有分区（消费者数 < 分区数）         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Group A 和 Group B 独立消费，同一消息被两个组各消费一次     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**消费者组配置：**

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-processor-group");  // 关键配置
props.put("key.deserializer", StringDeserializer.class.getName());
props.put("value.deserializer", StringDeserializer.class.getName());

// 其他重要配置
props.put("auto.offset.reset", "earliest");       // 新组从哪里开始消费
props.put("enable.auto.commit", false);           // 手动提交 offset
props.put("max.poll.records", 500);               // 单次拉取最大记录数
props.put("max.poll.interval.ms", 300000);        // 两次 poll 最大间隔

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("orders"));
```

**追问：消费者组数量有什么限制？**

**追问答案：**

```
消费者组数量限制：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 单组内消费者数量                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 消费者数 > 分区数：部分消费者空闲                     │   │
│  │ 消费者数 = 分区数：每个消费者一个分区                 │   │
│  │ 消费者数 < 分区数：部分消费者消费多个分区             │   │
│  │                                                     │   │
│  │ 建议：消费者数 ≤ 分区数                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 消费者组总数                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Broker 配置：group.max.size                          │   │
│  │ 默认：Integer.MAX_VALUE                              │   │
│  │                                                     │   │
│  │ 实际限制因素：                                       │   │
│  │ - Broker 内存（每个组需要维护状态）                  │   │
│  │ - __consumer_offsets 分区数（默认 50）              │   │
│  │ - 性能考虑                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 分区数和消费者数的最佳实践                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 分区数 = 目标吞吐量 / 单消费者吞吐量                  │   │
│  │                                                     │   │
│  │ 例如：                                              │   │
│  │ 目标吞吐量：100 万/秒                               │   │
│  │ 单消费者吞吐量：5 万/秒                              │   │
│  │ 分区数 = 100 / 5 = 20                              │   │
│  │ 消费者数 = 20（每个消费者一个分区）                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 什么是 Rebalance？

**答案：**

**Rebalance 机制详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Rebalance 触发条件                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Rebalance：重新分配分区给消费者组内的消费者                  │
│                                                             │
│  触发条件：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 消费者加入/离开消费者组                           │   │
│  │ 2. 消费者超时（session.timeout.ms）                  │   │
│  │ 3. 消费者处理超时（max.poll.interval.ms）            │   │
│  │ 4. Topic 分区数变化                                  │   │
│  │ 5. 订阅的 Topic 数量变化                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Rebalance 过程：                                            │
│                                                             │
│  时间轴 ────────────────────────────────────────────────▶   │
│                                                             │
│  状态：STABLE ──▶ PREPARE_REBALANCE ──▶ COMPLETING_REBALANCE │
│                 ▲                    │                      │
│                 │                    ▼                      │
│                 └───────── STABLE ◀─┘                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 所有消费者停止消费                                │   │
│  │ 2. 所有消费者发送 JoinGroup 请求                     │   │
│  │ 3. Leader 消费者计算分区分配方案                     │   │
│  │ 4. 分配方案发送给所有消费者                          │   │
│  │ 5. 消费者按新方案开始消费                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Rebalance 图解：**

```
Rebalance 前后分区分配变化：

Rebalance 前：
┌─────────────────────────────────────────────────────────────┐
│  Topic (4 个分区)                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Part 0 │ │  Part 1 │ │  Part 2 │ │  Part 3 │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       │           │           │           │                 │
│       ▼           ▼           ▼           ▼                 │
│  ┌──────────┐ ┌──────────┐                                │
│  │Consumer 1│ │Consumer 2│                                │
│  │Part 0,1  │ │Part 2,3  │                                │
│  └──────────┘ └──────────┘                                │
└─────────────────────────────────────────────────────────────┘

Consumer 3 加入（触发 Rebalance）：

Rebalance 后：
┌─────────────────────────────────────────────────────────────┐
│  Topic (4 个分区)                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Part 0 │ │  Part 1 │ │  Part 2 │ │  Part 3 │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       │           │           │           │                 │
│       ▼           ▼           ▼           ▼                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │Consumer 1│ │Consumer 2│ │Consumer 3│                   │
│  │  Part 0  │ │  Part 1  │ │Part 2,3  │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

**追问：Rebalance 有什么问题？**

**追问答案：**

```
Rebalance 的问题：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 消费暂停（Stop-the-World）                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Rebalance 期间，所有消费者停止消费                    │   │
│  │ 导致消费延迟飙升                                     │   │
│  │                                                     │   │
│  │ 时间轴：                                            │   │
│  │ ───────────────────────────────────────────────▶    │   │
│  │   消费中    │ Rebalance │   消费恢复                │   │
│  │             │  (暂停)   │                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 重复消费                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Rebalance 前，消息可能已拉取但未处理完成              │   │
│  │ Rebalance 后，新消费者重新拉取这些消息                │   │
│  │ 导致重复消费                                         │   │
│  │                                                     │   │
│  │ 解决：                                               │   │
│  │ - 手动提交 offset                                   │   │
│  │ - 消费者端幂等处理                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 连锁反应                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 一个消费者的变化触发整个组的 Rebalance               │   │
│  │ 大规模消费者组影响更大                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. 频繁 Rebalance                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 配置不当导致消费者频繁被判定为离线                    │   │
│  │ 如 session.timeout.ms 设置过短                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. 怎么避免不必要的 Rebalance？

**答案：**

**避免 Rebalance 的策略：**

```java
// 关键配置优化
Properties props = new Properties();

// ============ 心跳和会话配置 ============

// 1. 心跳间隔 - 必须小于 session.timeout.ms
props.put("heartbeat.interval.ms", 3000);  // 建议: session.timeout.ms 的 1/3

// 2. 会话超时 - 消费者失联判定时间
props.put("session.timeout.ms", 10000);    // 建议: 10-30 秒

// 3. 最大轮询间隔 - 处理超时判定
props.put("max.poll.interval.ms", 300000); // 建议: 根据处理时间设置

// 4. 单次最大拉取记录数 - 避免处理时间过长
props.put("max.poll.records", 500);

// ============ 静态成员配置（Kafka 2.3+）============

// 5. 静态成员 ID - 避免短暂离线触发 Rebalance
props.put("group.instance.id", "consumer-1-static");

// 6. 会话超时延长（配合静态成员）
props.put("session.timeout.ms", 300000);   // 可设置更长
```

**静态成员机制：**

```
┌─────────────────────────────────────────────────────────────┐
│                    静态成员机制                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  配置 group.instance.id 后：                                │
│                                                             │
│  1. 消费者短暂离线不会触发 Rebalance                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 消费者重启/网络抖动：                                 │   │
│  │ - 保留分区分配                                       │   │
│  │ - 等待消费者重新加入                                 │   │
│  │ - 只在 session.timeout.ms 超时后才踢出               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 适用场景                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 消费者重启部署                                     │   │
│  │ - 网络短暂故障                                       │   │
│  │ - 计划内维护                                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 注意事项                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 每个消费者必须有唯一的 group.instance.id          │   │
│  │ - 消费者数量不能超过分区数                           │   │
│  │ - 离线期间分区不会被消费                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
public class StableConsumer {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("group.id", "stable-consumer-group");
        
        // 关键：静态成员配置
        props.put("group.instance.id", "consumer-" + getLocalIp());
        props.put("session.timeout.ms", 300000);    // 5 分钟
        props.put("heartbeat.interval.ms", 10000);  // 10 秒
        
        props.put("max.poll.interval.ms", 600000);  // 10 分钟
        props.put("max.poll.records", 100);
        props.put("enable.auto.commit", false);
        
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("my-topic"));
        
        try {
            while (true) {
                ConsumerRecords<String, String> records = 
                    consumer.poll(Duration.ofMillis(1000));
                
                // 处理消息（确保在 max.poll.interval.ms 内完成）
                for (ConsumerRecord<String, String> record : records) {
                    processMessage(record);
                }
                
                // 手动提交
                consumer.commitSync();
            }
        } finally {
            consumer.close();
        }
    }
    
    // 处理时间监控
    private static void processMessage(ConsumerRecord<String, String> record) {
        long start = System.currentTimeMillis();
        
        // 业务处理
        doProcess(record);
        
        long elapsed = System.currentTimeMillis() - start;
        // 监控处理时间，预警
        if (elapsed > 5000) {
            log.warn("消息处理时间过长: {} ms", elapsed);
        }
    }
}
```

**追问：如何检测和排查 Rebalance 问题？**

**追问答案：**

```
Rebalance 监控指标：

┌─────────────────────────────────────────────────────────────┐
│                    关键监控指标                              │
├─────────────────────┬───────────────────────────────────────┤
│       指标           │               说明                    │
├─────────────────────┼───────────────────────────────────────┤
│ join-rate           │ 每秒 JoinGroup 请求数                 │
│ sync-rate           │ 每秒 SyncGroup 请求数                 │
│ heartbeat-rate      │ 心跳发送频率                          │
│ heartbeat-response-max│ 心跳响应最大延迟                    │
│ commit-rate         │ Offset 提交频率                       │
│ assigned-partitions │ 分配的分区数                          │
└─────────────────────┴───────────────────────────────────────┘

排查步骤：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 检查日志                                                │
│     grep "Revoking previously assigned partitions" kafka.log│
│     grep "Member .* leaving" kafka.log                      │
│                                                             │
│  2. 检查消费者状态                                          │
│     kafka-consumer-groups.sh --describe --group <group-id> │
│                                                             │
│  3. 检查处理时间                                            │
│     - 是否超过 max.poll.interval.ms                         │
│     - 是否有慢消息处理                                      │
│                                                             │
│  4. 检查网络                                                │
│     - 心跳是否正常发送                                      │
│     - 是否有网络抖动                                        │
│                                                             │
│  5. 检查 GC                                                 │
│     - 是否有长时间 GC 暂停                                  │
│     - 是否需要调整 JVM 参数                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 4. 三种消费语义是什么？

**答案：**

**三种消费语义详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                    三种消费语义                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. At Most Once（最多一次）                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 配置：                                              │   │
│  │ enable.auto.commit = true                           │   │
│  │ auto.commit.interval.ms = 1000                      │   │
│  │                                                     │   │
│  │ 流程：                                              │   │
│  │ 1. 拉取消息                                         │   │
│  │ 2. 自动提交 offset（可能处理前）                     │   │
│  │ 3. 处理消息（可能失败）                              │   │
│  │                                                     │   │
│  │ 风险：消息丢失（offset 已提交但处理失败）            │   │
│  │ 适用：允许丢失数据的场景（如日志）                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. At Least Once（至少一次）                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 配置：                                              │   │
│  │ enable.auto.commit = false                          │   │
│  │                                                     │   │
│  │ 流程：                                              │   │
│  │ 1. 拉取消息                                         │   │
│  │ 2. 处理消息                                         │   │
│  │ 3. 成功后手动提交 offset                            │   │
│  │                                                     │   │
│  │ 风险：消息重复（处理成功但提交失败）                 │   │
│  │ 适用：大多数业务场景，配合幂等消费                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Exactly Once（精确一次）                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 配置：                                              │   │
│  │ enable.auto.commit = false                          │   │
│  │ isolation.level = read_committed                    │   │
│  │                                                     │   │
│  │ 流程：                                              │   │
│  │ 1. 使用事务消费者                                   │   │
│  │ 2. 处理消息 + 写入结果（原子操作）                   │   │
│  │ 3. 事务提交 offset                                  │   │
│  │                                                     │   │
│  │ 保证：消息不丢失、不重复                            │   │
│  │ 适用：高可靠性场景（金融、支付）                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
// At Most Once
public class AtMostOnceConsumer {
    public void consume() {
        Properties props = new Properties();
        props.put("enable.auto.commit", "true");           // 自动提交
        props.put("auto.commit.interval.ms", "1000");
        
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("topic"));
        
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            for (ConsumerRecord<String, String> record : records) {
                // offset 可能已经提交，处理失败会丢消息
                processMessage(record);
            }
        }
    }
}

// At Least Once
public class AtLeastOnceConsumer {
    public void consume() {
        Properties props = new Properties();
        props.put("enable.auto.commit", "false");          // 手动提交
        
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("topic"));
        
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            for (ConsumerRecord<String, String> record : records) {
                try {
                    processMessage(record);
                    // 处理成功后提交
                    consumer.commitSync(Collections.singletonMap(
                        new TopicPartition(record.topic(), record.partition()),
                        new OffsetAndMetadata(record.offset() + 1)
                    ));
                } catch (Exception e) {
                    // 处理失败不提交，下次重新消费
                    log.error("处理失败", e);
                }
            }
        }
    }
}

// Exactly Once（事务消费）
public class ExactlyOnceConsumer {
    public void consume() {
        Properties props = new Properties();
        props.put("enable.auto.commit", "false");
        props.put("isolation.level", "read_committed");
        
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("topic"));
        
        // 配合事务生产者
        KafkaProducer<String, String> producer = createTransactionalProducer();
        producer.initTransactions();
        
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            if (records.isEmpty()) continue;
            
            // 开启事务
            producer.beginTransaction();
            
            try {
                for (ConsumerRecord<String, String> record : records) {
                    // 处理消息并发送到输出 Topic
                    String result = processMessage(record);
                    producer.send(new ProducerRecord<>("output-topic", record.key(), result));
                }
                
                // 提交事务（包括 offset）
                Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>();
                for (TopicPartition partition : records.partitions()) {
                    List<ConsumerRecord<String, String>> partitionRecords = records.records(partition);
                    long lastOffset = partitionRecords.get(partitionRecords.size() - 1).offset();
                    offsets.put(partition, new OffsetAndMetadata(lastOffset + 1));
                }
                producer.sendOffsetsToTransaction(offsets, "my-group");
                producer.commitTransaction();
                
            } catch (Exception e) {
                producer.abortTransaction();
                log.error("事务回滚", e);
            }
        }
    }
}
```

---

### 5. Offset 怎么管理？

**答案：**

**Offset 管理方式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Offset 管理方式                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Kafka 内部管理（默认）                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 存储位置：__consumer_offsets Topic                   │   │
│  │ Key: <group.id, topic, partition>                   │   │
│  │ Value: offset + metadata                            │   │
│  │                                                     │   │
│  │ 优点：无需额外存储，自动管理                         │   │
│  │ 缺点：不支持事务语义                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 外部存储管理                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 存储位置：数据库、Redis 等                           │   │
│  │                                                     │   │
│  │ 优点：支持事务语义，与业务数据一致                   │   │
│  │ 缺点：实现复杂，需要额外存储                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**外部存储管理 Offset：**

```java
public class ExternalOffsetConsumer {
    private final KafkaConsumer<String, String> consumer;
    private final OffsetStore offsetStore;  // 自定义 offset 存储
    
    public void consume(String topic) {
        // 禁用自动提交
        Properties props = new Properties();
        props.put("enable.auto.commit", "false");
        // ...
        
        consumer = new KafkaConsumer<>(props);
        
        // 手动分配分区
        List<PartitionInfo> partitions = consumer.partitionsFor(topic);
        List<TopicPartition> topicPartitions = partitions.stream()
            .map(p -> new TopicPartition(topic, p.partition()))
            .collect(Collectors.toList());
        consumer.assign(topicPartitions);
        
        // 从外部存储读取 offset
        for (TopicPartition tp : topicPartitions) {
            Long offset = offsetStore.getOffset(tp);
            if (offset != null) {
                consumer.seek(tp, offset);
            } else {
                consumer.seekToBeginning(Collections.singletonList(tp));
            }
        }
        
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                try {
                    // 原子性：处理消息 + 更新 offset
                    processWithOffset(record);
                } catch (Exception e) {
                    log.error("处理失败", e);
                }
            }
        }
    }
    
    // 原子性处理：数据库事务 + offset 更新
    @Transactional
    private void processWithOffset(ConsumerRecord<String, String> record) {
        // 1. 处理业务逻辑
        processBusiness(record);
        
        // 2. 更新 offset 到数据库
        offsetStore.saveOffset(
            new TopicPartition(record.topic(), record.partition()),
            record.offset() + 1
        );
    }
}

// Offset 存储接口
public interface OffsetStore {
    Long getOffset(TopicPartition partition);
    void saveOffset(TopicPartition partition, long offset);
}

// 数据库实现
public class DatabaseOffsetStore implements OffsetStore {
    private final JdbcTemplate jdbcTemplate;
    
    @Override
    public Long getOffset(TopicPartition partition) {
        String sql = "SELECT offset FROM consumer_offsets " +
                     "WHERE group_id = ? AND topic = ? AND partition_id = ?";
        return jdbcTemplate.queryForObject(sql, Long.class, 
            groupId, partition.topic(), partition.partition());
    }
    
    @Override
    public void saveOffset(TopicPartition partition, long offset) {
        String sql = "INSERT INTO consumer_offsets " +
                     "(group_id, topic, partition_id, offset) VALUES (?, ?, ?, ?) " +
                     "ON DUPLICATE KEY UPDATE offset = ?";
        jdbcTemplate.update(sql, 
            groupId, partition.topic(), partition.partition(), offset, offset);
    }
}
```

**追问：__consumer_offsets 的分区数如何影响性能？**

**追问答案：**

```
__consumer_offsets 分区说明：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  默认配置：offsets.topic.num.partitions = 50                │
│                                                             │
│  分区作用：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 每个 Consumer Group 的 offset 存储在一个分区       │   │
│  │ - 分区通过 hash(group.id) 确定                       │   │
│  │ - 分区数决定了并发写入能力                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  性能影响：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 消费者组数量多时：                                   │   │
│  │ - 分区数不足会导致写入竞争                           │   │
│  │ - 建议：消费者组数 / 10 = 最小分区数                │   │
│  │                                                     │   │
│  │ 例如：500 个消费者组                                │   │
│  │ 建议分区数：50（默认值足够）                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  注意事项：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 分区数只能增加，不能减少                           │   │
│  │ - 增加分区需要滚动重启 Broker                        │   │
│  │ - 建议提前规划好分区数                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 6. 消费者拉取模式是怎样的？

**答案：**

**Pull 模式 vs Push 模式：**

```
┌─────────────────────────────────────────────────────────────┐
│                Kafka Pull 模式优势                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Pull 模式（Kafka 采用）：                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Consumer              Broker                        │   │
│  │    │                    │                           │   │
│  │    │──poll()───────────▶│                           │   │
│  │    │                    │ 返回消息                  │   │
│  │    │◀──messages─────────│ (或空)                    │   │
│  │    │                    │                           │   │
│  │    │    处理消息        │                           │   │
│  │    │                    │                           │   │
│  │    │──poll()───────────▶│                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  优势：                                                     │
│  1. 消费者控制消费速率，不会被压垮                          │
│  2. 消费者可以批量拉取，提高吞吐量                          │
│  3. 天然支持回溯消费                                        │
│                                                             │
│  Push 模式（RabbitMQ 等采用）：                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Consumer              Broker                        │   │
│  │    │                    │                           │   │
│  │    │◀──message1─────────│ 主动推送                  │   │
│  │    │                    │                           │   │
│  │    │◀──message2─────────│                           │   │
│  │    │                    │                           │   │
│  │    │◀──message3─────────│ 消费者可能被压垮          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题：                                                     │
│  1. 消费速率不可控，可能压垮消费者                          │
│  2. 需要复杂的流控机制                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**poll() 方法详解：**

```java
// poll 方法行为
public ConsumerRecords<K, V> poll(Duration timeout) {
    // 1. 检查是否有待处理的协调请求
    // 2. 更新心跳（如果需要）
    // 3. 从服务器拉取数据
    // 4. 返回拉取的记录
    
    // timeout 参数：
    // - 如果缓冲区有数据，立即返回
    // - 如果缓冲区无数据，等待最多 timeout 时间
    // - timeout=0 表示非阻塞，立即返回
}

// 合理配置 poll 行为
Properties props = new Properties();

// 拉取相关配置
props.put("fetch.min.bytes", 1);           // 最小拉取字节数
props.put("fetch.max.bytes", 52428800);     // 最大拉取字节数 (50MB)
props.put("fetch.max.wait.ms", 500);        // 最大等待时间
props.put("max.partition.fetch.bytes", 1048576); // 单分区最大拉取
props.put("max.poll.records", 500);         // 单次最大记录数

// 使用示例
while (true) {
    // 阻塞最多 1 秒
    ConsumerRecords<String, String> records = 
        consumer.poll(Duration.ofMillis(1000));
    
    if (records.isEmpty()) {
        // 无数据，可以做一些其他工作
        continue;
    }
    
    // 处理记录
    for (ConsumerRecord<String, String> record : records) {
        processMessage(record);
    }
    
    // 提交 offset
    consumer.commitSync();
}
```

---

### 7. 消费者分区分配策略有哪些？

**答案：**

**分区分配策略对比：**

```
┌─────────────────────────────────────────────────────────────┐
│                    分区分配策略                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Range（默认）                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 按 Topic 分别分配                                    │   │
│  │                                                     │   │
│  │ Topic A (7 分区) + 3 消费者：                        │   │
│  │ C1: [0,1,2]  C2: [3,4]  C3: [5,6]                   │   │
│  │                                                     │   │
│  │ Topic B (7 分区) + 3 消费者：                        │   │
│  │ C1: [0,1,2]  C2: [3,4]  C3: [5,6]                   │   │
│  │                                                     │   │
│  │ 问题：多个 Topic 时，C1 压力更大                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. RoundRobin                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 所有 Topic 的分区一起轮询分配                        │   │
│  │                                                     │   │
│  │ Topic A (3 分区) + Topic B (3 分区) + 2 消费者：    │   │
│  │ C1: [A0, A2, B1]                                    │   │
│  │ C2: [A1, B0, B2]                                    │   │
│  │                                                     │   │
│  │ 优点：负载更均衡                                     │   │
│  │ 要求：所有消费者订阅相同的 Topic                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Sticky（Kafka 0.11+）                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 尽量保持之前的分配关系                               │   │
│  │                                                     │   │
│  │ Rebalance 时：                                       │   │
│  │ - 保留现有分配                                       │   │
│  │ - 只重新分配需要变化的分区                           │   │
│  │                                                     │   │
│  │ 优点：减少 Rebalance 影响                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. CooperativeSticky（Kafka 2.4+）                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 渐进式 Rebalance                                    │   │
│  │ - 不需要停止所有消费                                 │   │
│  │ - 分批迁移分区                                       │   │
│  │ - 消费不中断                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**配置方式：**

```java
Properties props = new Properties();

// 设置分区分配策略
props.put("partition.assignment.strategy", 
    "org.apache.kafka.clients.consumer.StickyAssignor");

// 多个策略（按优先级）
props.put("partition.assignment.strategy", 
    "org.apache.kafka.clients.consumer.StickyAssignor," +
    "org.apache.kafka.clients.consumer.RoundRobinAssignor");

// Cooperative Sticky（推荐）
props.put("partition.assignment.strategy",
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
```

---

### 8. 如何实现多线程消费？

**答案：**

**多线程消费模式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    多线程消费模式                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式1：单消费者 + 工作线程池                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌──────────┐                                       │   │
│  │  │Consumer  │──poll──▶ records                      │   │
│  │  │ Thread   │             │                         │   │
│  │  └──────────┘             ▼                         │   │
│  │                     ┌─────────┐                      │   │
│  │                     │ 分发器   │                      │   │
│  │                     └─────────┘                      │   │
│  │                      │  │  │                         │   │
│  │              ┌───────┼──┼──┼───────┐                │   │
│  │              ▼       ▼  ▼  ▼       ▼                │   │
│  │          ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │   │
│  │          │Worker│ │Worker│ │Worker│ │Worker│        │   │
│  │          │  1   │ │  2   │ │  3   │ │  4   │        │   │
│  │          └──────┘ └──────┘ └──────┘ └──────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│  优点：实现简单，offset 管理集中                            │
│  缺点：处理失败会影响整批消息                               │
│                                                             │
│  模式2：多消费者实例                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Consumer Group                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │Consumer 1│ │Consumer 2│ │Consumer 3│            │   │
│  │  │ Partition│ │ Partition│ │ Partition│            │   │
│  │  │    0     │ │    1     │ │    2     │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  │       │            │            │                   │   │
│  │       ▼            ▼            ▼                   │   │
│  │   ┌──────┐    ┌──────┐    ┌──────┐                │   │
│  │   │Thread│    │Thread│    │Thread│                │   │
│  │   └──────┘    └──────┘    └──────┘                │   │
│  └─────────────────────────────────────────────────────┘   │
│  优点：真正的并行消费                                       │
│  缺点：消费者数受限于分区数                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
// 模式1：单消费者 + 线程池
public class WorkerThreadConsumer {
    private final KafkaConsumer<String, String> consumer;
    private final ExecutorService executor;
    
    public void start() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            if (records.isEmpty()) continue;
            
            // 提交任务到线程池
            List<Future<?>> futures = new ArrayList<>();
            for (ConsumerRecord<String, String> record : records) {
                Future<?> future = executor.submit(() -> {
                    try {
                        processMessage(record);
                    } catch (Exception e) {
                        log.error("处理失败", e);
                    }
                });
                futures.add(future);
            }
            
            // 等待所有任务完成
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    log.error("任务执行异常", e);
                }
            }
            
            // 提交 offset
            consumer.commitSync();
        }
    }
}

// 模式2：多消费者实例
public class MultiConsumerInstance {
    public static void main(String[] args) {
        int consumerCount = 3;
        ExecutorService executor = Executors.newFixedThreadPool(consumerCount);
        
        for (int i = 0; i < consumerCount; i++) {
            executor.submit(() -> {
                Properties props = createConsumerConfig();
                props.put("group.id", "my-group");
                
                KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
                consumer.subscribe(Arrays.asList("my-topic"));
                
                try {
                    while (true) {
                        ConsumerRecords<String, String> records = 
                            consumer.poll(Duration.ofMillis(100));
                        
                        for (ConsumerRecord<String, String> record : records) {
                            processMessage(record);
                        }
                        
                        consumer.commitSync();
                    }
                } finally {
                    consumer.close();
                }
            });
        }
    }
}
```


---

## 架构篇

### 1. Kafka 的架构是怎样的？

**答案：**

**Kafka 整体架构图：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Kafka 集群架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ZooKeeper / KRaft                             │   │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐                │   │
│  │    │  Node 1  │    │  Node 2  │    │  Node 3  │                │   │
│  │    │(Leader)  │    │(Follower)│    │(Follower)│                │   │
│  │    └──────────┘    └──────────┘    └──────────┘                │   │
│  │                                                                  │   │
│  │    存储：Controller 选举、Broker 注册、Topic 元数据              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ 元数据同步                               │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Broker Cluster                                │   │
│  │                                                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │   Broker 1   │  │   Broker 2   │  │   Broker 3   │          │   │
│  │  │   (Controller)│  │              │  │              │          │   │
│  │  │              │  │              │  │              │          │   │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │          │   │
│  │  │  │Topic A │  │  │  │Topic A │  │  │  │Topic B │  │          │   │
│  │  │  │Part 0  │  │  │  │Part 1  │  │  │  │Part 0  │  │          │   │
│  │  │  │Leader  │  │  │  │Leader  │  │  │  │Leader  │  │          │   │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │          │   │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │          │   │
│  │  │  │Topic A │  │  │  │Topic B │  │  │  │Topic A │  │          │   │
│  │  │  │Part 2  │  │  │  │Part 0  │  │  │  │Part 1  │  │          │   │
│  │  │  │Follower│  │  │  │Follower│  │  │  │Follower│  │          │   │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                   │
│         │                    │                    │                   │
│         ▼                    ▼                    ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│  │  Producer   │     │  Producer   │     │  Consumer   │             │
│  │  Group 1    │     │  Group 2    │     │  Group 1    │             │
│  └─────────────┘     └─────────────┘     └─────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心组件说明：**

| 组件 | 作用 | 说明 |
|------|------|------|
| **Broker** | Kafka 节点 | 负责消息存储和转发 |
| **Controller** | 集群控制器 | 管理 Partition Leader 选举、Broker 上下线 |
| **ZooKeeper/KRaft** | 元数据存储 | 存储 Broker 注册、Topic 配置、Controller 选举 |
| **Producer** | 消息生产者 | 发送消息到 Broker |
| **Consumer** | 消息消费者 | 从 Broker 拉取消息 |

**追问：Controller 是什么？有什么作用？**

**追问答案：**

```
Controller 详解：

┌─────────────────────────────────────────────────────────────┐
│                    Controller 作用                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Controller 是 Kafka 集群中的特殊 Broker：                   │
│                                                             │
│  职责：                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Broker 上下线管理                                 │   │
│  │    - 检测 Broker 故障                                │   │
│  │    - 触发 Leader 选举                                │   │
│  │                                                     │   │
│  │ 2. Partition Leader 选举                            │   │
│  │    - ISR 中选举新 Leader                             │   │
│  │    - 更新元数据                                      │   │
│  │                                                     │   │
│  │ 3. Topic 管理                                        │   │
│  │    - 创建/删除 Topic                                │   │
│  │    - 分区分配                                        │   │
│  │                                                     │   │
│  │ 4. 副本重新分配                                      │   │
│  │    - 分区迁移                                        │   │
│  │    - 副本扩缩容                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Controller 选举：                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Broker 启动时尝试在 ZK 创建 /controller 节点      │   │
│  │ 2. 第一个成功的 Broker 成为 Controller              │   │
│  │ 3. 其他 Broker 监听 Controller 变化                  │   │
│  │ 4. Controller 故障时，其他 Broker 竞争成为新 Controller│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Controller 故障恢复：                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 新 Controller 从 ZK 读取元数据                    │   │
│  │ 2. 初始化所有 Partition 的状态                       │   │
│  │ 3. 恢复进行中的操作                                  │   │
│  │ 4. 恢复时间取决于分区数量                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 副本机制是什么？

**答案：**

**副本机制详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                    副本架构                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Topic: orders (3 分区, 3 副本)                             │
│                                                             │
│  Partition 0:                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Broker 1           Broker 2           Broker 3     │   │
│  │  ┌──────────┐      ┌──────────┐      ┌──────────┐  │   │
│  │  │ Leader   │─────▶│ Follower │─────▶│ Follower │  │   │
│  │  │  ISR     │ 复制  │  ISR     │ 复制  │  ISR     │  │   │
│  │  │          │      │          │      │          │  │   │
│  │  │ 写入+读取│      │ 只有读取 │      │ 只有读取 │  │   │
│  │  │ (活跃)   │      │ (热备)   │      │ (热备)   │  │   │
│  │  └──────────┘      └──────────┘      └──────────┘  │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  副本角色：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Leader:                                             │   │
│  │ - 处理所有读写请求                                   │   │
│  │ - 维护 ISR 列表                                      │   │
│  │ - 推进 HW (High Watermark)                          │   │
│  │                                                     │   │
│  │ Follower:                                           │   │
│  │ - 从 Leader 拉取数据                                 │   │
│  │ - 不处理客户端请求                                   │   │
│  │ - 故障时可被选为 Leader                              │   │
│  │                                                     │   │
│  │ ISR (In-Sync Replicas):                             │   │
│  │ - 与 Leader 保持同步的副本集合                       │   │
│  │ - 只有 ISR 中的副本才能被选为 Leader                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**副本同步机制：**

```
┌─────────────────────────────────────────────────────────────┐
│                    副本同步流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Leader (Broker 1)          Follower (Broker 2)            │
│  ┌───────────────┐          ┌───────────────┐              │
│  │ LEO = 10      │          │ LEO = 8       │              │
│  │ HW = 8        │          │ HW = 8        │              │
│  │               │          │               │              │
│  │ ┌───────────┐ │          │ ┌───────────┐ │              │
│  │ │0 1 2 3 4 5│ │          │ │0 1 2 3 4 5│ │              │
│  │ │6 7 8 9   │ │          │ │6 7       │ │              │
│  │ └───────────┘ │          │ └───────────┘ │              │
│  └───────┬───────┘          └───────┬───────┘              │
│          │                          │                       │
│          │◀───── FetchRequest ──────│                       │
│          │      (fetchOffset=8)     │                       │
│          │                          │                       │
│          │───── FetchResponse ─────▶│                       │
│          │  (messages 8,9, HW=8)    │                       │
│          │                          │                       │
│          │     Follower 更新        │                       │
│          │     LEO = 10             │                       │
│          │     HW = min(LEO, LeaderHW) = 8                 │
│          │                          │                       │
│          │◀───── FetchRequest ──────│                       │
│          │      (fetchOffset=10)    │                       │
│          │                          │                       │
│          │───── FetchResponse ─────▶│                       │
│          │      (no data, HW=10)    │                       │
│          │                          │                       │
│          │     Follower 更新        │                       │
│          │     HW = 10              │                       │
│          │                          │                       │
└─────────────────────────────────────────────────────────────┘

关键概念：
- LEO (Log End Offset): 日志末端偏移量，下一条消息的位置
- HW (High Watermark): 高水位，所有 ISR 同步的位置
- 消费者只能消费到 HW 之前的消息
```

**追问：如何选择合适的副本数？**

**追问答案：**

```
副本数选择指南：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  副本数 vs 可用性 vs 成本：                                  │
│                                                             │
│  ┌─────────────┬─────────────────┬───────────────────────┐ │
│  │  副本数      │   可容忍故障数   │        成本           │ │
│  ├─────────────┼─────────────────┼───────────────────────┤ │
│  │     1       │       0         │     1x 存储空间       │ │
│  │     2       │       1         │     2x 存储空间       │ │
│  │     3       │       2         │     3x 存储空间       │ │
│  │     5       │       4         │     5x 存储空间       │ │
│  └─────────────┴─────────────────┴───────────────────────┘ │
│                                                             │
│  推荐配置：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 开发/测试环境：replication.factor = 1                │   │
│  │ 生产环境：      replication.factor = 3              │   │
│  │ 高可用场景：    replication.factor = 5              │   │
│  │                                                     │   │
│  │ 配合 min.insync.replicas = 2                        │   │
│  │ 可容忍 1 个节点故障而不丢失数据                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  注意事项：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 副本数不能超过 Broker 数量                        │   │
│  │ 2. 副本分布应尽量均匀（跨机架/机房）                  │   │
│  │ 3. 副本数增加会影响写入性能                          │   │
│  │ 4. 可以动态增加副本数（但很耗时）                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. Leader 和 Follower 的作用？

**答案：**

```
┌─────────────────────────────────────────────────────────────┐
│                Leader 和 Follower 职责                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Leader 副本职责：                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 处理所有生产者的写入请求                          │   │
│  │ 2. 处理所有消费者的读取请求                          │   │
│  │ 3. 维护和更新 ISR 列表                               │   │
│  │ 4. 推进 HW (High Watermark)                         │   │
│  │ 5. 响应 Follower 的 Fetch 请求                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Follower 副本职责：                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 从 Leader 拉取消息进行同步                        │   │
│  │ 2. 不处理客户端请求（Kafka 不支持 Follower 读）      │   │
│  │ 3. 故障时可被选为 Leader                             │   │
│  │ 4. 发送心跳证明存活                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Leader 选举流程：                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  1. Leader 故障检测                                  │   │
│  │     Controller 通过 ZK 监听 Broker 状态              │   │
│  │     或通过心跳超时检测                               │   │
│  │                                                     │   │
│  │  2. 触发 Leader 选举                                 │   │
│  │     Controller 从 ISR 中选择新 Leader               │   │
│  │     如果 ISR 为空：                                  │   │
│  │     - unclean.leader.election.enable=true           │   │
│  │       从 OSR 中选择（可能丢数据）                    │   │
│  │     - unclean.leader.election.enable=false          │   │
│  │       分区不可用                                     │   │
│  │                                                     │   │
│  │  3. 更新元数据                                       │   │
│  │     通知所有 Broker 新的 Leader 信息                 │   │
│  │                                                     │   │
│  │  4. 客户端感知                                       │   │
│  │     客户端通过元数据更新获取新 Leader                │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Leader 选举时间线：**

```
时间线：

T0: Leader 宕机
│
├── T1: session.timeout.ms 超时（默认 10s）
│        Controller 检测到 Leader 故障
│
├── T2: Controller 触发 Leader 选举
│        从 ISR 中选择新 Leader
│
├── T3: 更新 ZK 元数据
│        通知所有 Broker
│
├── T4: 客户端 metadata.refresh.interval.ms
│        获取新的 Leader 信息
│
└── T5: 恢复服务

总恢复时间 ≈ session.timeout.ms + 选举时间 + 元数据同步时间
           ≈ 10s + 1s + 1s = 约 12 秒（默认配置）
```

---

### 4. KRaft 模式是什么？

**答案：**

**KRaft 模式详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                ZooKeeper vs KRaft 模式                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ZooKeeper 模式（传统）：                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    ZooKeeper                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  ZK 1    │ │  ZK 2    │ │  ZK 3    │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │        │            │            │                 │   │
│  │        └────────────┼────────────┘                 │   │
│  │                     │                               │   │
│  │                     ▼                               │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Kafka Cluster                  │   │   │
│  │  │  Broker1 │ Broker2 │ Broker3 │ ...         │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│  问题：                                                     │
│  - 需要维护两套系统                                         │
│  - 元数据分散存储                                           │
│  - 扩缩容复杂                                               │
│                                                             │
│  KRaft 模式（Kafka 3.0+）：                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Kafka Cluster (KRaft)                  │   │
│  │                                                     │   │
│  │  Controller 节点：                                   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │Controller│ │Controller│ │Controller│           │   │
│  │  │  (Leader)│ │(Follower)│ │(Follower)│           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │        │            │            │                 │   │
│  │        └────────────┼────────────┘                 │   │
│  │                     │                               │   │
│  │  Broker 节点：                                       │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ Broker1 │ Broker2 │ Broker3 │ ...          │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  元数据存储在内部 Topic: __cluster_metadata         │   │
│  └─────────────────────────────────────────────────────┘   │
│  优势：                                                     │
│  - 架构简化，无需 ZK                                        │
│  - 元数据集中管理                                           │
│  - 扩缩容更简单                                             │
│  - 更好的性能                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：为什么要去掉 ZooKeeper？**

**追问答案：**

```
ZooKeeper 的问题：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 架构复杂                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 需要部署和维护两套系统（ZK + Kafka）               │   │
│  │ - 运维成本高                                         │   │
│  │ - 需要两套监控和告警                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 元数据不一致                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Topic 元数据存在 ZK                                │   │
│  │ - 分区状态存在 Controller 内存                       │   │
│  │ - 两边同步可能不一致                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 扩容复杂                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 分区数受限于 ZK 性能                               │   │
│  │ - ZK 本身也有扩展限制                                │   │
│  │ - 大规模集群性能下降                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. Controller 故障恢复慢                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 新 Controller 需要从 ZK 加载所有元数据             │   │
│  │ - 分区多时恢复时间长                                 │   │
│  │ - 可能影响服务可用性                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  KRaft 的改进：                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 使用 Kafka 自己的 Raft 协议实现一致性             │   │
│  │ 2. 元数据存储在 __cluster_metadata Topic            │   │
│  │ 3. Controller 故障恢复更快                           │   │
│  │ 4. 支持 100 万+ 分区                                 │   │
│  │ 5. 架构简化，运维成本降低                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. Kafka 如何实现高可用？

**答案：**

**高可用架构设计：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka 高可用设计                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 多副本机制                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ replication.factor >= 3                              │   │
│  │ min.insync.replicas >= 2                             │   │
│  │                                                     │   │
│  │ 副本分布在不同机架/机房                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. Controller 高可用                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 任何 Broker 都可以成为 Controller                    │   │
│  │ Controller 故障时自动选举新的 Controller             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Broker 故障转移                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Leader 故障时，从 ISR 选举新 Leader                  │   │
│  │ 客户端自动重连新 Leader                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. 机架感知                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ broker.rack 配置机架信息                             │   │
│  │ 副本自动分布到不同机架                               │   │
│  │ 机架故障时保证数据不丢失                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  5. 跨机房部署                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │    机房 A                    机房 B                  │   │
│  │  ┌──────────┐            ┌──────────┐              │   │
│  │  │ Broker 1 │            │ Broker 4 │              │   │
│  │  │ Broker 2 │◀──────────▶│ Broker 5 │              │   │
│  │  │ Broker 3 │  复制      │ Broker 6 │              │   │
│  │  └──────────┘            └──────────┘              │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**高可用配置示例：**

```properties
# ============ 副本配置 ============
default.replication.factor=3
min.insync.replicas=2

# ============ 故障检测配置 ============
# Broker 心跳
broker.heartbeat.interval.ms=1000
broker.session.timeout.ms=10000

# 副本同步超时
replica.lag.time.max.ms=30000

# ============ 选举配置 ============
# 禁止脏选举
unclean.leader.election.enable=false

# ============ 机架感知 ============
# 每个 Broker 配置
broker.rack=rack-1  # 或 rack-2, rack-3 等

# ============ 生产者配置 ============
acks=all
enable.idempotence=true
retries=2147483647
```

---

### 6. Kafka 如何进行容量规划？

**答案：**

**容量规划公式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    容量规划关键指标                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 磁盘容量                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 总磁盘容量 = 每日消息量 × 消息大小 × 保留天数        │   │
│  │             × 副本数 × 压缩比 × 冗余系数            │   │
│  │                                                     │   │
│  │ 例如：                                              │   │
│  │ - 每日消息量：1 亿条                                │   │
│  │ - 平均消息大小：1 KB                                │   │
│  │ - 保留天数：7 天                                    │   │
│  │ - 副本数：3                                         │   │
│  │ - 压缩比：0.3 (压缩后 30%)                          │   │
│  │ - 冗余系数：1.2                                     │   │
│  │                                                     │   │
│  │ 总容量 = 1亿 × 1KB × 7 × 3 × 0.3 × 1.2            │   │
│  │        = 756 GB                                     │   │
│  │                                                     │   │
│  │ 每个 Broker = 756 / 3 = 252 GB                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 分区数                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 分区数 = max(目标吞吐量 / 单分区吞吐量,              │   │
│  │             目标并发消费者数)                        │   │
│  │                                                     │   │
│  │ 例如：                                              │   │
│  │ - 目标吞吐量：100 万条/秒                           │   │
│  │ - 单分区吞吐量：5 万条/秒                           │   │
│  │                                                     │   │
│  │ 分区数 = 100万 / 5万 = 20 个                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Broker 数量                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Broker 数量 = max(总磁盘容量 / 单 Broker 磁盘,       │   │
│  │                  目标吞吐量 / 单 Broker 吞吐量)      │   │
│  │                                                     │   │
│  │ 考虑因素：                                          │   │
│  │ - 副本分布（至少 >= replication.factor）            │   │
│  │ - 机架分布                                          │   │
│  │ - 网络带宽                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. 内存和网络                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ JVM Heap: 4-8 GB (不要太大)                         │   │
│  │ Page Cache: 越大越好，建议 50% 系统内存              │   │
│  │ 网络带宽: 消息量 × 消息大小 × 副本数                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


---

## 实战篇

### 1. 消息积压怎么处理？

**答案：**

**消息积压诊断与处理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消息积压诊断                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 查看消费者组 Lag                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ kafka-consumer-groups.sh --describe --group my-group│   │
│  │                                                     │   │
│  │ 输出：                                              │   │
│  │ TOPIC  PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG│   │
│  │ orders 0          1000            100000         99000│   │
│  │ orders 1          500             50000          49500│   │
│  │ orders 2          2000            200000         198000│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 分析积压原因                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 消费者处理速度慢                                  │   │
│  │ - 消费者数量不足                                    │   │
│  │ - 下游系统瓶颈                                      │   │
│  │ - 消费者故障/频繁 Rebalance                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                    积压处理方案                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案1：增加消费者数量（需同时增加分区）                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │ 当前：3 分区 + 3 消费者                             │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐               │   │
│  │ │Consumer │ │Consumer │ │Consumer │               │   │
│  │ │   1     │ │   2     │ │   3     │               │   │
│  │ │ Part 0  │ │ Part 1  │ │ Part 2  │               │   │
│  │ └─────────┘ └─────────┘ └─────────┘               │   │
│  │                                                     │   │
│  │ 调整：增加分区到 6 个                                │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ │Consumer │ │Consumer │ │Consumer │ │Consumer │ │Consumer │ │Consumer │ │
│  │ │   1     │ │   2     │ │   3     │ │   4     │ │   5     │ │   6     │ │
│  │ │ Part 0  │ │ Part 1  │ │ Part 2  │ │ Part 3  │ │ Part 4  │ │ Part 5  │ │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
│  │                                                     │   │
│  │ # 增加分区                                          │   │
│  │ kafka-topics.sh --alter --topic orders             │   │
│  │   --partitions 6 --bootstrap-server localhost:9092 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案2：临时消费者组 + 转发                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │ 积压 Topic        新 Topic（扩大分区）             │   │
│  │ ┌─────────┐      ┌───────────────────┐             │   │
│  │ │ orders  │      │ orders-temp       │             │   │
│  │ │ 3 分区  │──────▶│ 12 分区           │             │   │
│  │ │         │ 转发  │                   │             │   │
│  │ └─────────┘      └───────────────────┘             │   │
│  │      ▲                   │                          │   │
│  │      │                   ▼                          │   │
│  │  原 Consumer      临时 Consumer Group               │   │
│  │  (暂停消费)        (12 个消费者并行)                │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案3：优化消费者处理速度                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 批量处理代替单条处理                              │   │
│  │ - 异步处理非核心逻辑                                │   │
│  │ - 增加本地缓存减少数据库访问                        │   │
│  │ - 优化慢查询                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码示例：**

```java
// 临时消费者组方案
public class TempConsumerGroup {
    
    // 1. 转发消费者：将积压 Topic 转发到大分区 Topic
    public void forwardToTempTopic() {
        KafkaConsumer<String, String> sourceConsumer = createConsumer("original-group");
        KafkaProducer<String, String> producer = createProducer();
        
        sourceConsumer.subscribe(Arrays.asList("orders"));
        
        while (true) {
            ConsumerRecords<String, String> records = 
                sourceConsumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                // 转发到临时 Topic
                producer.send(new ProducerRecord<>("orders-temp", 
                    record.key(), record.value()));
            }
            
            sourceConsumer.commitSync();
        }
    }
    
    // 2. 扩容消费者组：消费临时 Topic
    public void consumeFromTempTopic() {
        Properties props = createConsumerConfig();
        props.put("group.id", "temp-consumer-group-" + getInstanceId());
        
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Arrays.asList("orders-temp"));
        
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            // 批量处理
            processBatch(records);
            
            consumer.commitSync();
        }
    }
}
```

**追问：如何预防消息积压？**

**追问答案：**

```
预防措施：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 监控告警                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 设置 Lag 阈值告警（如 Lag > 10000）                │   │
│  │ - 监控消费速率变化                                   │   │
│  │ - 监控消费者状态                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 容量规划                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 预估峰值流量，预留 50% 冗余                        │   │
│  │ - 分区数 >= 峰值消费者数                            │   │
│  │ - 定期压测验证容量                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 限流和降级                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 生产者限流                                         │   │
│  │ - 消费者降级（跳过非核心处理）                       │   │
│  │ - 下游服务熔断                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. 架构优化                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 核心业务和非核心业务分离                           │   │
│  │ - 高峰期弹性扩容                                     │   │
│  │ - 批量处理优化                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 消息顺序性怎么保证？

**答案：**

**Kafka 消息顺序保证：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka 顺序性保证                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 分区内有序（天然保证）                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │ Partition 0:                                        │   │
│  │ ┌─────┬─────┬─────┬─────┬─────┐                    │   │
│  │ │msg1 │msg2 │msg3 │msg4 │msg5 │  严格按顺序       │   │
│  │ └─────┴─────┴─────┴─────┴─────┘                    │   │
│  │                                                     │   │
│  │ 同一分区内的消息严格按写入顺序存储和消费            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. 分区间无序                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │ Partition 0:  msg1 → msg2 → msg3                   │   │
│  │ Partition 1:  msgA → msgB → msgC                   │   │
│  │                                                     │   │
│  │ msg2 可能在 msgA 之前或之后消费                     │   │
│  │ 不同分区之间没有顺序保证                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 全局有序（需要特殊设计）                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 方案：Topic 只有一个分区                            │   │
│  │ 缺点：牺牲并行度和吞吐量                            │   │
│  │ 通常不推荐                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**业务顺序保证方案：**

```java
// 方案1：使用 Key 保证同一业务实体的消息进入同一分区

// 订单消息：使用订单 ID 作为 Key
public class OrderProducer {
    public void sendOrderEvents(Order order) {
        // 同一订单的所有事件使用相同的 Key
        String key = order.getOrderId();
        
        // 发送顺序：创建 → 支付 → 发货 → 完成
        producer.send(new ProducerRecord<>("orders", key, 
            new OrderEvent("CREATED", order)));
        producer.send(new ProducerRecord<>("orders", key, 
            new OrderEvent("PAID", order)));
        producer.send(new ProducerRecord<>("orders", key, 
            new OrderEvent("SHIPPED", order)));
        producer.send(new ProducerRecord<>("orders", key, 
            new OrderEvent("COMPLETED", order)));
        
        // 这些消息会进入同一分区，保证顺序
    }
}

// 方案2：生产者配置保证重试时不乱序
Properties props = new Properties();

// 开启幂等性（保证单分区内有序）
props.put("enable.idempotence", true);

// 如果不开启幂等性，需要限制并发请求数
props.put("max.in.flight.requests.per.connection", 1);
// 注意：这会严重影响性能

// 方案3：消费者端按业务 ID 缓冲排序
public class OrderAwareConsumer {
    private final Map<String, List<OrderEvent>> eventBuffer = new ConcurrentHashMap<>();
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                OrderEvent event = parseEvent(record);
                String orderId = event.getOrderId();
                
                // 缓冲事件
                eventBuffer.computeIfAbsent(orderId, k -> new ArrayList<>())
                    .add(event);
                
                // 检查是否可以处理
                tryProcessEvent(orderId);
            }
        }
    }
    
    private void tryProcessEvent(String orderId) {
        List<OrderEvent> events = eventBuffer.get(orderId);
        
        // 按事件顺序处理
        events.sort(Comparator.comparing(OrderEvent::getSequence));
        
        for (OrderEvent event : events) {
            if (canProcess(event)) {
                processEvent(event);
                events.remove(event);
            }
        }
    }
}
```

**追问：如果消费者处理失败重试，顺序会不会乱？**

**追问答案：**

```
消费者重试场景：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  场景：msg1, msg2, msg3 顺序到达                            │
│                                                             │
│  情况1：msg2 处理失败，不提交 offset                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 消费 msg1，成功，提交 offset = 1                  │   │
│  │ 2. 消费 msg2，失败，不提交 offset                    │   │
│  │ 3. 消费 msg3，成功（但不提交，因为 msg2 未处理）     │   │
│  │ 4. 重新拉取：从 offset = 1 开始                      │   │
│  │ 5. 再次消费 msg2, msg3                              │   │
│  │                                                     │   │
│  │ 问题：msg1 被重复消费                               │   │
│  │ 顺序：msg1 → msg2 → msg3（顺序正确）                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  解决方案：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 幂等消费（处理重复消息）                          │   │
│  │ 2. 本地事务 + 状态检查                              │   │
│  │ 3. 死信队列（多次失败后跳过）                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  代码示例：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ // 幂等消费 + 状态检查                              │   │
│  │ public void processOrderEvent(OrderEvent event) {   │   │
│  │     Order order = orderRepo.findById(event.getOrderId());│
│  │                                                     │   │
│  │     // 状态机检查：只有特定状态才能转换              │   │
│  │     if (order.canTransition(event.getType())) {     │   │
│  │         order.transition(event.getType());          │   │
│  │         orderRepo.save(order);                      │   │
│  │     } else {                                        │   │
│  │         // 状态已转换，可能是重复消息，跳过          │   │
│  │         log.warn("跳过重复或非法事件: {}", event);   │   │
│  │     }                                               │   │
│  │ }                                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. 怎么做消息追踪？

**答案：**

**消息追踪方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                    消息追踪方案                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案1：消息头携带追踪信息                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │   │
│  │  │ Producer │───▶│  Kafka   │───▶│ Consumer │      │   │
│  │  └──────────┘    └──────────┘    └──────────┘      │   │
│  │       │                               │             │   │
│  │       │  添加 Headers                  │  读取 Headers│   │
│  │       ▼                               ▼             │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ Headers:                                     │   │   │
│  │  │ - trace-id: abc123                          │   │   │
│  │  │ - span-id: span1                            │   │   │
│  │  │ - parent-span-id: span0                     │   │   │
│  │  │ - timestamp-produce: 1640000000000          │   │   │
│  │  │ - timestamp-consume: 1640000001000          │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案2：日志记录                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - 生产者发送时记录日志                              │   │
│  │ - 消费者消费时记录日志                              │   │
│  │ - 日志收集到 ELK/LGTM 等系统                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案3：专用追踪系统                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - OpenTelemetry                                     │   │
│  │ - Jaeger/Zipkin                                     │   │
│  │ - SkyWalking                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
// 生产者端添加追踪信息
public class TracingProducer {
    private final KafkaProducer<String, String> producer;
    private final Tracer tracer;  // OpenTelemetry Tracer
    
    public void sendWithTracing(String topic, String key, String value) {
        // 创建 Span
        Span span = tracer.spanBuilder("kafka.produce")
            .setAttribute("topic", topic)
            .startSpan();
        
        try {
            ProducerRecord<String, String> record = 
                new ProducerRecord<>(topic, key, value);
            
            // 注入追踪信息到消息头
            SpanContext spanContext = span.getSpanContext();
            record.headers().add("trace-id", 
                spanContext.getTraceId().getBytes());
            record.headers().add("span-id", 
                spanContext.getSpanId().getBytes());
            record.headers().add("produce-time", 
                String.valueOf(System.currentTimeMillis()).getBytes());
            
            producer.send(record, (metadata, exception) -> {
                if (exception != null) {
                    span.recordException(exception);
                    span.setStatus(StatusCode.ERROR);
                } else {
                    span.setAttribute("partition", metadata.partition());
                    span.setAttribute("offset", metadata.offset());
                }
                span.end();
            });
        } catch (Exception e) {
            span.recordException(e);
            span.setStatus(StatusCode.ERROR);
            span.end();
            throw e;
        }
    }
}

// 消费者端提取追踪信息
public class TracingConsumer {
    private final KafkaConsumer<String, String> consumer;
    private final Tracer tracer;
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                // 提取追踪信息
                String traceId = getHeader(record, "trace-id");
                String parentSpanId = getHeader(record, "span-id");
                long produceTime = Long.parseLong(getHeader(record, "produce-time"));
                
                // 创建消费者 Span
                Span span = tracer.spanBuilder("kafka.consume")
                    .setParent(Context.current()
                        .with(TraceContextPropagator.getInstance()
                            .extract(Context.current(), 
                                new HeadersExtractor(record.headers()))))
                    .setAttribute("topic", record.topic())
                    .setAttribute("partition", record.partition())
                    .setAttribute("offset", record.offset())
                    .setAttribute("latency", 
                        System.currentTimeMillis() - produceTime)
                    .startSpan();
                
                try {
                    processMessage(record);
                } finally {
                    span.end();
                }
            }
            
            consumer.commitSync();
        }
    }
    
    private String getHeader(ConsumerRecord<String, String> record, String key) {
        Header header = record.headers().lastHeader(key);
        return header != null ? new String(header.value()) : null;
    }
}
```

---

### 4. Kafka 怎么监控？

**答案：**

**关键监控指标：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka 监控指标                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Broker 级别                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 指标                      │ 说明                     │   │
│  ├──────────────────────────┼──────────────────────────┤   │
│  │ BytesInPerSec            │ 每秒写入字节数           │   │
│  │ BytesOutPerSec           │ 每秒读取字节数           │   │
│  │ MessagesInPerSec         │ 每秒写入消息数           │   │
│  │ TotalTimeMs              │ 请求处理时间             │   │
│  │ UnderReplicatedPartitions │ 未完全同步的分区数       │   │
│  │ OfflinePartitionsCount   │ 离线分区数               │   │
│  │ ActiveControllerCount    │ 活跃 Controller 数       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. Topic 级别                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ MessagesInPerSec         │ 每秒消息数               │   │
│  │ BytesInPerSec            │ 每秒字节数               │   │
│  │ BytesOutPerSec           │ 每秒读取字节数           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. 生产者级别                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ record-send-rate         │ 消息发送速率             │   │
│  │ record-error-rate        │ 消息发送错误率           │   │
│  │ request-latency-avg      │ 请求平均延迟             │   │
│  │ batch-size-avg           │ 平均批次大小             │   │
│  │ buffer-available-bytes   │ 缓冲区可用字节数         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. 消费者级别                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ records-consumed-rate    │ 消费速率                 │   │
│  │ records-lag-max          │ 最大 Lag                 │   │
│  │ fetch-rate               │ 拉取请求速率             │   │
│  │ fetch-latency-avg        │ 拉取延迟                 │   │
│  │ commit-rate              │ 提交速率                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**监控工具：**

```
┌─────────────────────────────────────────────────────────────┐
│                    监控工具选择                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Kafka 自带工具                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ # 查看消费者组状态                                   │   │
│  │ kafka-consumer-groups.sh --describe --group my-group│   │
│  │                                                     │   │
│  │ # 查看 Topic 详情                                   │   │
│  │ kafka-topics.sh --describe --topic my-topic        │   │
│  │                                                     │   │
│  │ # 查看 Broker JMX 指标                              │   │
│  │ JConsole / VisualVM                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  2. Prometheus + Grafana                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ JMX Exporter → Prometheus → Grafana                 │   │
│  │                                                     │   │
│  │ 配置 JMX Exporter:                                  │   │
│  │ KAFKA_OPTS="-javaagent:jmx_prometheus_javaagent.jar=7071:config.yml"│
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  3. Kafka Manager / CMAK                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Web UI 管理 Kafka 集群                              │   │
│  │ 查看 Topic、Consumer Group、Broker 状态             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  4. Burrow（Lag 监控专用）                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 专门监控消费者 Lag                                  │   │
│  │ 提供 HTTP API 和告警功能                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**告警规则示例：**

```yaml
# Prometheus 告警规则
groups:
  - name: kafka-alerts
    rules:
      # Broker 告警
      - alert: KafkaBrokerDown
        expr: kafka_server_broker_state != 3
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Kafka Broker {{ $labels.instance }} is down"
      
      # Under Replicated Partitions
      - alert: UnderReplicatedPartitions
        expr: kafka_server_underreplicated > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Under replicated partitions detected"
      
      # 消费者 Lag 告警
      - alert: ConsumerLagHigh
        expr: kafka_consumer_lag > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Consumer group {{ $labels.group }} lag is high"
      
      # 生产者错误率
      - alert: ProducerErrorRateHigh
        expr: rate(kafka_producer_error_rate[5m]) > 0.01
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Producer error rate is high"
```

---

### 5. 生产环境遇到过什么问题？

**答案：**

**常见生产问题及解决方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                    生产环境常见问题                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题1：频繁 Rebalance                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：消费暂停，日志大量 Rebalance 记录              │   │
│  │ 原因：session.timeout.ms 过短，处理超时              │   │
│  │ 解决：                                               │   │
│  │ - 增加 session.timeout.ms                           │   │
│  │ - 使用静态成员 (group.instance.id)                  │   │
│  │ - 优化消费逻辑，减少处理时间                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题2：磁盘空间不足                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：Broker 无法写入，生产者报错                    │   │
│  │ 原因：日志保留策略配置不当，或突发流量               │   │
│  │ 解决：                                               │   │
│  │ - 调整 log.retention.hours/bytes                    │   │
│  │ - 扩容磁盘                                          │   │
│  │ - 清理不用的 Topic                                  │   │
│  │ - 启用压缩                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题3：消息积压                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：Lag 持续增长，消费延迟高                       │   │
│  │ 原因：消费速度跟不上生产速度                         │   │
│  │ 解决：                                               │   │
│  │ - 增加分区数和消费者数                              │   │
│  │ - 临时扩容消费者组                                  │   │
│  │ - 优化消费逻辑                                      │   │
│  │ - 批量处理                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题4：Leader 选举超时                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：Broker 宕机后，分区长时间无 Leader             │   │
│  │ 原因：ISR 为空，或 Controller 负载高                 │   │
│  │ 解决：                                               │   │
│  │ - 检查 ISR 状态                                     │   │
│  │ - 增加 min.insync.replicas                          │   │
│  │ - 分散 Controller 负载                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题5：Producer 发送超时                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：生产者报 TimeoutException                     │   │
│  │ 原因：网络问题、Broker 负载高、批次配置不当          │   │
│  │ 解决：                                               │   │
│  │ - 检查网络连通性                                    │   │
│  │ - 增加超时时间                                      │   │
│  │ - 调整 batch.size 和 linger.ms                      │   │
│  │ - 检查 Broker 磁盘和网络                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  问题6：消费者 OOM                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 现象：消费者进程崩溃                                 │   │
│  │ 原因：max.poll.records 过大，消息处理占用内存        │   │
│  │ 解决：                                               │   │
│  │ - 减少 max.poll.records                            │   │
│  │ - 减少 fetch.max.bytes                             │   │
│  │ - 增加 JVM 堆内存                                   │   │
│  │ - 优化消息处理逻辑                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**排查工具和命令：**

```bash
# 1. 查看集群概览
kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# 2. 查看 Consumer Group 详情
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group my-group

# 3. 查看 Topic 分区详情
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe --topic my-topic

# 4. 查看日志目录大小
du -sh /var/kafka-logs/*

# 5. 查看 Broker 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type brokers --entity-name 1 --describe

# 6. 查看消费者组是否活跃
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --list

# 7. 重置消费者组 offset（慎用）
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group --reset-offsets --to-earliest --topic my-topic --execute

# 8. 查看 Broker 日志
tail -f /var/log/kafka/server.log | grep -i "error\|warn"
```

---

## 参考资料

- [Apache Kafka 官方文档](https://kafka.apache.org/documentation/)
- [Kafka 权威指南](https://book.douban.comsubject/...)
- [Kafka 源码解析](https://github.com/apache/kafka)
- [Confluent Kafka 最佳实践](https://docs.confluent.io/platform/current/kafka/operations.html)
