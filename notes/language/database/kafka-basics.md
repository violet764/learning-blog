# Apache Kafka 学习笔记

Apache Kafka 是一个分布式流处理平台，最初由 LinkedIn 开发，后成为 Apache 顶级项目。Kafka 以高吞吐、低延迟、高可用著称，广泛应用于消息队列、事件溯源、流处理、日志聚合等场景。Kafka 3.x 带来了革命性的 KRaft 模式，彻底告别了 ZooKeeper 依赖。

## Kafka 3.x 新特性

### 🚀 KRaft 模式（取代 ZooKeeper）

KRaft（Kafka Raft）是 Kafka 3.x 最重要的变化，使用 Kafka 自己的 Raft 协议替代 ZooKeeper 进行元数据管理。

| 对比项 | ZooKeeper 模式 | KRaft 模式 |
|--------|---------------|------------|
| 架构复杂度 | 需要 ZooKeeper 集群 | 无外部依赖 |
| 元数据存储 | ZooKeeper | 内部 Topic `__cluster_metadata` |
| Controller | 单一 Active Controller | 多个 Controller（Raft 选举） |
| 扩展性 | 受 ZooKeeper 限制 | 更好的扩展性 |
| 运维成本 | 高（两套集群） | 低（单一系统） |
| 推荐状态 | 已弃用 | **生产推荐** |

**KRaft 架构图**：

```
┌─────────────────────────────────────────────────────────┐
│                    KRaft Cluster                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Controller Quorum                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│  │  │Controller│  │Controller│  │Controller│      │   │
│  │  │  Leader  │  │ Follower │  │ Follower │      │   │
│  │  └──────────┘  └──────────┘  └──────────┘      │   │
│  │       │              │              │           │   │
│  │       └──────────────┼──────────────┘           │   │
│  │                      │                          │   │
│  │              __cluster_metadata                 │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌───────────────────────┼───────────────────────────┐ │
│  │                       ▼                           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │ │
│  │  │  Broker  │  │  Broker  │  │  Broker  │        │ │
│  │  │   Node   │  │   Node   │  │   Node   │        │ │
│  │  └──────────┘  └──────────┘  └──────────┘        │ │
│  │              Data Topics                         │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**KRaft 模式配置**：

```bash
# server.properties（KRaft 模式）

# 节点角色：controller、broker 或 combined
process.roles=broker,controller

# 节点 ID
node.id=1

# Controller Quorum 配置
controller.quorum.voters=1@host1:9093,2@host2:9093,3@host3:9093

# 监听器配置
listeners=PLAINTEXT://:9092,CONTROLLER://:9093
inter.broker.listener.name=PLAINTEXT
advertised.listeners=PLAINTEXT://host1:9092

# Controller 监听器
controller.listener.names=CONTROLLER

# 日志目录
log.dirs=/var/kafka/data
```

### 📝 其他重要改进

| 特性 | 说明 |
|------|------|
| **分层存储** | 支持冷热数据分离，降低存储成本 |
| **事务改进** | 更高效的事务协调器 |
| **性能优化** | 更快的 Leader 选举和恢复 |
| **Java 11+** | 支持更新的 Java 版本 |
| **Metrics 改进** | 更丰富的监控指标 |

## 核心概念

### 📌 Broker

Broker 是 Kafka 集群中的服务节点，负责接收、存储和发送消息。

```
┌─────────────────────────────────────────────────────────┐
│                    Kafka Cluster                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Broker 1 │    │ Broker 2 │    │ Broker 3 │          │
│  │ Leader   │    │ Follower │    │ Follower │          │
│  │ Topic-A  │    │ Topic-A  │    │ Topic-A  │          │
│  │ P0, P2   │    │ P1       │    │ P0, P1, P2│         │
│  └──────────┘    └──────────┘    └──────────┘          │
│        │               │               │                │
│        └───────────────┴───────────────┘                │
│                    Replication                          │
└─────────────────────────────────────────────────────────┘
```

**Broker 关键配置**：

```bash
# server.properties

# 基础配置
broker.id=1                          # Broker 唯一标识
log.dirs=/var/kafka/data             # 数据目录（多个用逗号分隔）

# 网络配置
num.network.threads=3                # 网络线程数
num.io.threads=8                     # IO 线程数
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# 日志配置
num.partitions=3                     # 默认分区数
num.recovery.threads.per.data.dir=1  # 恢复线程数
log.retention.hours=168              # 日志保留时间（7天）
log.segment.bytes=1073741824         # 日志段大小（1GB）
log.retention.check.interval.ms=300000

# 复制配置
default.replication.factor=3         # 默认副本数
min.insync.replicas=2                # 最小同步副本数
```

### 📌 Topic 与 Partition

**Topic** 是消息的逻辑分类，**Partition** 是 Topic 的物理分片，实现并行处理。

```
Topic: orders
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Partition 0          Partition 1          Partition 2  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
│  │ Offset: 0   │     │ Offset: 0   │     │ Offset: 0   ││
│  │ Offset: 1   │     │ Offset: 1   │     │ Offset: 1   ││
│  │ Offset: 2   │     │ Offset: 2   │     │ Offset: 2   ││
│  │ Offset: 3   │     │ Offset: 3   │     │ Offset: 3   ││
│  │    ...      │     │    ...      │     │    ...      ││
│  └─────────────┘     └─────────────┘     └─────────────┘│
│                                                         │
│  Broker 1            Broker 2            Broker 3       │
│  (Leader)            (Leader)            (Leader)       │
└─────────────────────────────────────────────────────────┘
```

**Partition 分区原理**：

- 每个 Partition 是一个有序的、不可变的消息序列
- 消息在 Partition 内按 Offset 顺序存储
- 不同 Partition 之间消息无顺序保证
- Partition 数量决定了最大并行度

**Topic 操作示例**：

```bash
# 创建 Topic
kafka-topics.sh --create \
  --topic orders \
  --partitions 6 \
  --replication-factor 3 \
  --bootstrap-server localhost:9092

# 查看 Topic 详情
kafka-topics.sh --describe \
  --topic orders \
  --bootstrap-server localhost:9092

# 输出示例：
# Topic: orders     PartitionCount: 6    ReplicationFactor: 3
# Topic: orders     Partition: 0    Leader: 1    Replicas: 1,2,3    Isr: 1,2,3
# Topic: orders     Partition: 1    Leader: 2    Replicas: 2,3,1    Isr: 2,3,1

# 修改 Topic 配置
kafka-configs.sh --alter \
  --topic orders \
  --add-config retention.ms=86400000 \
  --bootstrap-server localhost:9092

# 增加 Partition（只能增加，不能减少）
kafka-topics.sh --alter \
  --topic orders \
  --partitions 12 \
  --bootstrap-server localhost:9092
```

**Java/Python 客户端创建 Topic**：

```java
// Java 客户端
import org.apache.kafka.clients.admin.*;

Properties props = new Properties();
props.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");

try (AdminClient admin = AdminClient.create(props)) {
    NewTopic topic = new NewTopic("orders", 6, (short) 3);
    admin.createTopics(Collections.singletonList(topic)).all().get();
}
```

```python
# Python 客户端
from kafka import KafkaAdminClient
from kafka.admin import NewTopic

admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:9092"
)

topic = NewTopic(
    name="orders",
    num_partitions=6,
    replication_factor=3
)
admin_client.create_topics([topic])
```

### 📌 Consumer Group

Consumer Group 是 Kafka 实现伸缩性消费的核心机制。

```
Topic: orders (6 Partitions)
┌─────────────────────────────────────────────────────────┐
│  P0    P1    P2    P3    P4    P5                       │
└──┬─────┬─────┬─────┬─────┬─────┬──┘                    │
   │     │     │     │     │     │                        │
   ▼     ▼     ▼     ▼     ▼     ▼                        │
┌──────────────────────────────────────┐                 │
│         Consumer Group A             │                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐│                 │
│  │Consumer │  │Consumer │  │Consumer ││                 │
│  │   1     │  │   2     │  │   3     ││                 │
│  │ P0,P1   │  │ P2,P3   │  │ P4,P5   ││                 │
│  └─────────┘  └─────────┘  └─────────┘│                 │
└──────────────────────────────────────┘                 │
                                                          │
┌──────────────────────────────────────┐                 │
│         Consumer Group B             │                 │
│  ┌─────────┐  ┌─────────┐            │                 │
│  │Consumer │  │Consumer │            │                 │
│  │   1     │  │   2     │            │                 │
│  │ P0,P1,P2│  │ P3,P4,P5│            │                 │
│  └─────────┘  └─────────┘            │                 │
└──────────────────────────────────────┘                 │
```

**Consumer Group 特性**：

- 同一组内的 Consumer 共同消费 Topic
- 每个 Partition 只能被同组内一个 Consumer 消费
- 不同组独立消费，互不影响
- Consumer 数量超过 Partition 数量时，部分 Consumer 闲置

**消费组配置示例**：

```java
// Java 消费者配置
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-processor-group");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

// 消费语义配置
props.put("enable.auto.commit", "false");  // 手动提交，推荐
props.put("auto.offset.reset", "earliest"); // 从最早开始消费
props.put("max.poll.records", "500");       // 单次最大拉取记录数
props.put("max.poll.interval.ms", "300000"); // 两次 poll 的最大间隔

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("orders"));
```

```python
# Python 消费者配置
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'orders',
    bootstrap_servers=['localhost:9092'],
    group_id='order-processor-group',
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    max_poll_records=500,
    value_deserializer=lambda x: x.decode('utf-8')
)

for message in consumer:
    print(f"Partition: {message.partition}, Offset: {message.offset}")
    print(f"Key: {message.key}, Value: {message.value}")
    
    # 手动提交
    consumer.commit()
```

### 📌 Offset 与提交策略

Offset 是消息在 Partition 中的唯一标识，消费者通过 Offset 追踪消费进度。

```
Partition Log:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
        ↑                               ↑
    Committed Offset               Current Offset
    (已提交位置)                    (当前消费位置)
```

**Offset 提交策略**：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 自动提交 | 定期自动提交 | 对消息丢失不敏感 |
| 同步手动提交 | 处理后同步提交 | 可靠性优先 |
| 异步手动提交 | 处理后异步提交 | 性能优先 |
| 指定 Offset 提交 | 精确控制提交位置 | Exactly-Once 语义 |

```java
// Java Offset 提交示例

// 1. 自动提交（不推荐用于关键业务）
props.put("enable.auto.commit", "true");
props.put("auto.commit.interval.ms", "5000");

// 2. 同步手动提交（推荐）
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);
    }
    try {
        consumer.commitSync();  // 阻塞直到提交成功
    } catch (CommitFailedException e) {
        log.error("Commit failed", e);
    }
}

// 3. 异步手动提交
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);
    }
    consumer.commitAsync((offsets, exception) -> {
        if (exception != null) {
            log.error("Commit failed for offsets {}", offsets, exception);
        }
    });
}

// 4. 指定 Offset 提交（精细控制）
Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>();
for (ConsumerRecord<String, String> record : records) {
    process(record);
    offsets.put(
        new TopicPartition(record.topic(), record.partition()),
        new OffsetAndMetadata(record.offset() + 1)
    );
}
consumer.commitSync(offsets);
```

## 生产者配置与性能优化

### 🔧 核心配置参数

```java
// Java 生产者配置
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// 可靠性配置
props.put("acks", "all");                    // 确认机制
props.put("enable.idempotence", "true");     // 幂等性
props.put("retries", "3");                    // 重试次数
props.put("max.in.flight.requests.per.connection", "5"); // 最大飞行请求数

// 批处理配置
props.put("batch.size", "16384");            // 批次大小（16KB）
props.put("linger.ms", "5");                  // 等待时间
props.put("buffer.memory", "33554432");      // 缓冲区大小（32MB）

// 压缩配置
props.put("compression.type", "lz4");        // 压缩算法

// 性能配置
props.put("max.block.ms", "60000");          // 最大阻塞时间
props.put("request.timeout.ms", "30000");    // 请求超时
props.put("delivery.timeout.ms", "120000");  // 交付超时

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```

### 📊 ACKS 确认机制

acks 参数决定了生产者认为消息发送成功的条件：

```
acks=0:  生产者不等待确认
acks=1:  等待 Leader 确认
acks=all (或 -1): 等待所有 ISR 副本确认
```

```
Producer          Broker (Topic with replication.factor=3)
   │                    │
   │  ┌────────────┐    │  ┌──────────────┐
   │  │   Leader   │    │  │   Broker 1   │
   │  │  Broker 1  │    │  │  (Leader)    │
   │  └────────────┘    │  └──────┬───────┘
   │                    │         │
   │  Send Message      │         │ Replicate
   │ ──────────────────>│         │
   │                    │         ▼
   │                    │  ┌──────────────┐
   │                    │  │   Broker 2   │
   │                    │  │  (Follower)  │
   │                    │  └──────────────┘
   │                    │         │
   │                    │         │ Replicate
   │                    │         ▼
   │                    │  ┌──────────────┐
   │                    │  │   Broker 3   │
   │                    │  │  (Follower)  │
   │                    │  └──────────────┘
```

| acks 值 | 可靠性 | 吞吐量 | 数据丢失风险 |
|---------|--------|--------|-------------|
| 0 | 最低 | 最高 | 高（可能丢失） |
| 1 | 中等 | 中等 | 中（Leader 故障可能丢失） |
| all | 最高 | 最低 | 低（需要 min.insync.replicas 配合） |

```java
// 不同场景的 acks 配置

// 场景1：日志收集，允许丢失
props.put("acks", "0");
props.put("retries", "0");

// 场景2：普通业务，平衡可靠性和性能
props.put("acks", "1");
props.put("retries", "3");

// 场景3：金融交易，不允许丢失
props.put("acks", "all");
props.put("enable.idempotence", "true");
props.put("min.insync.replicas", "2");  // Broker 端配置
```

### 🚀 批处理与压缩

**批处理原理**：

```
Producer Buffer:
┌─────────────────────────────────────────────┐
│  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ Batch 1 │  │ Batch 2 │  │ Batch 3 │     │
│  │ 16KB    │  │ 16KB    │  │ 16KB    │     │
│  └─────────┘  └─────────┘  └─────────┘     │
│       ↓            ↓            ↓           │
│   Partition 0  Partition 1  Partition 2    │
└─────────────────────────────────────────────┘

Timeline:
Time ──────────────────────────────────────>
     │         │         │         │
     Send      linger.ms Send      Send
     (batch    wait      (batch    (batch
      full)    more       full)     full)
```

**压缩算法对比**：

| 算法 | 压缩率 | CPU 消耗 | 推荐场景 |
|------|--------|----------|----------|
| none | 1.0x | 最低 | 网络带宽充足 |
| gzip | 2.5-3.0x | 高 | 带宽受限，CPU 充足 |
| snappy | 2.0-2.5x | 低 | 平衡性能和压缩 |
| lz4 | 2.0-2.5x | 最低 | 高吞吐场景（**推荐**） |
| zstd | 2.5-3.5x | 中等 | Kafka 2.1+（推荐） |

```java
// 批处理优化配置
Properties props = new Properties();

// 批处理大小：当达到此大小时立即发送
props.put("batch.size", "32768");  // 32KB

// 等待时间：等待更多消息凑够一个批次
props.put("linger.ms", "10");  // 10ms

// 压缩类型
props.put("compression.type", "zstd");  // 推荐 zstd 或 lz4

// 缓冲区大小
props.put("buffer.memory", "67108864");  // 64MB
```

### ⚡ 高吞吐量生产者示例

```java
import org.apache.kafka.clients.producer.*;

public class HighThroughputProducer {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        
        // 高吞吐配置
        props.put("acks", "1");                    // 平衡可靠性和性能
        props.put("batch.size", "65536");          // 64KB 批次
        props.put("linger.ms", "20");              // 等待20ms
        props.put("compression.type", "lz4");      // LZ4 压缩
        props.put("buffer.memory", "134217728");   // 128MB 缓冲
        props.put("max.in.flight.requests.per.connection", "5");
        
        KafkaProducer<String, String> producer = new KafkaProducer<>(props);
        
        // 异步发送回调
        Callback callback = (metadata, exception) -> {
            if (exception != null) {
                System.err.println("Send failed: " + exception.getMessage());
            }
        };
        
        // 批量发送
        for (int i = 0; i < 100000; i++) {
            ProducerRecord<String, String> record = 
                new ProducerRecord<>("orders", "key" + i, "value" + i);
            producer.send(record, callback);
        }
        
        // 确保所有消息发送完成
        producer.flush();
        producer.close();
    }
}
```

```python
# Python 高吞吐生产者
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    
    # 序列化
    key_serializer=lambda k: k.encode('utf-8'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    
    # 性能配置
    acks=1,
    batch_size=65536,
    linger_ms=20,
    compression_type='lz4',
    buffer_memory=134217728,
    
    # 重试配置
    retries=3,
    max_in_flight_requests_per_connection=5
)

# 异步发送
def on_send_success(record_metadata):
    print(f"Sent to partition {record_metadata.partition} at offset {record_metadata.offset}")

def on_send_error(excp):
    print(f"Send failed: {excp}")

for i in range(100000):
    producer.send(
        'orders',
        key=f'key{i}',
        value={'id': i, 'data': f'value{i}'}
    ).add_callback(on_send_success).add_errback(on_send_error)

# 确保发送完成
producer.flush()
producer.close()
```

## 消费者配置与消费语义

### 🔄 三种消费语义

```
┌─────────────────────────────────────────────────────────┐
│                   消费语义对比                           │
├─────────────────┬───────────────────┬───────────────────┤
│  At-Most-Once   │   At-Least-Once   │   Exactly-Once    │
│  最多一次       │   至少一次        │   精确一次        │
├─────────────────┼───────────────────┼───────────────────┤
│  可能丢失消息   │  可能重复消费     │  不丢不重         │
│  性能最高       │  性能中等         │  性能最低         │
│  实现简单       │  实现简单         │  实现复杂         │
└─────────────────┴───────────────────┴───────────────────┘
```

### At-Most-Once（最多一次）

消息可能丢失，但不会重复消费。

```java
// At-Most-Once 实现
Properties props = new Properties();
props.put("enable.auto.commit", "true");       // 自动提交
props.put("auto.commit.interval.ms", "1000"); // 每1秒提交

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("orders"));

while (true) {
    // 自动提交可能发生在处理消息之前
    // 如果处理失败，消息已经提交，导致丢失
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        // 处理可能失败，但 Offset 已经自动提交
        process(record);
    }
}
```

### At-Least-Once（至少一次）

消息不会丢失，但可能重复消费（最常用）。

```java
// At-Least-Once 实现
Properties props = new Properties();
props.put("enable.auto.commit", "false");  // 手动提交

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("orders"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        try {
            // 先处理消息
            process(record);
            // 处理成功后再提交
            // 如果提交失败，下次重新消费时会重复处理
            consumer.commitSync();
        } catch (Exception e) {
            // 处理失败，不提交，下次重新消费
            log.error("Process failed", e);
        }
    }
}
```

**幂等性处理（解决重复消费）**：

```java
// 业务端幂等性设计
public void process(ConsumerRecord<String, String> record) {
    String messageId = record.key();  // 使用唯一消息ID
    
    // 检查是否已处理
    if (processedCache.putIfAbsent(messageId, true) != null) {
        log.info("Message {} already processed, skipping", messageId);
        return;
    }
    
    try {
        // 业务处理
        doBusiness(record.value());
    } catch (Exception e) {
        // 处理失败，移除标记，允许重试
        processedCache.invalidate(messageId);
        throw e;
    }
}

// 使用 Redis 实现分布式幂等性
public void processWithRedis(ConsumerRecord<String, String> record) {
    String messageId = record.key();
    String key = "processed:" + messageId;
    
    // SETNX + EXPIRE 原子操作
    Boolean success = redis.setIfAbsent(key, "1", Duration.ofDays(1));
    if (!success) {
        log.info("Message {} already processed", messageId);
        return;
    }
    
    // 业务处理
    doBusiness(record.value());
}
```

### Exactly-Once（精确一次）

最严格的语义，需要事务支持。

```java
// Exactly-Once 实现（Kafka 事务）
Properties props = new Properties();
props.put("enable.auto.commit", "false");
props.put("isolation.level", "read_committed");  // 只读取已提交的消息
props.put("enable.idempotence", "true");         // 生产者幂等

// 生产者配置
Properties producerProps = new Properties();
producerProps.put("transactional.id", "order-processor-1");  // 事务ID

KafkaProducer<String, String> producer = new KafkaProducer<>(producerProps);
producer.initTransactions();

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("orders"));

try {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        
        if (records.isEmpty()) continue;
        
        // 开启事务
        producer.beginTransaction();
        
        try {
            for (ConsumerRecord<String, String> record : records) {
                // 处理消息并发送到输出 Topic
                String result = process(record.value());
                producer.send(new ProducerRecord<>("orders-processed", record.key(), result));
            }
            
            // 提交事务（原子性：消费 Offset + 输出消息）
            producer.sendOffsetsToTransaction(
                getOffsets(records),
                consumer.groupMetadata()
            );
            producer.commitTransaction();
            
        } catch (Exception e) {
            // 事务回滚
            producer.abortTransaction();
        }
    }
} finally {
    producer.close();
    consumer.close();
}

// 辅助方法：获取 Offset 映射
private Map<TopicPartition, OffsetAndMetadata> getOffsets(ConsumerRecords<String, String> records) {
    Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>();
    for (ConsumerRecord<String, String> record : records) {
        offsets.put(
            new TopicPartition(record.topic(), record.partition()),
            new OffsetAndMetadata(record.offset() + 1)
        );
    }
    return offsets;
}
```

### ⚙️ 消费者性能调优

```java
Properties props = new Properties();

// 拉取配置
props.put("fetch.min.bytes", "1");              // 最小拉取字节数
props.put("fetch.max.bytes", "52428800");       // 最大拉取字节数（50MB）
props.put("fetch.max.wait.ms", "500");          // 最大等待时间
props.put("max.partition.fetch.bytes", "1048576"); // 单分区最大拉取（1MB）

// 处理配置
props.put("max.poll.records", "500");           // 单次最大记录数
props.put("max.poll.interval.ms", "300000");    // 两次 poll 最大间隔（5分钟）

// 会话配置
props.put("session.timeout.ms", "10000");       // 会话超时（10秒）
props.put("heartbeat.interval.ms", "3000");     // 心跳间隔（3秒）

// 连接配置
props.put("connections.max.idle.ms", "540000"); // 连接最大空闲时间
```

## 分区策略与重平衡

### 📊 分区策略

Kafka 通过分区策略决定消息发送到哪个 Partition。

```
┌─────────────────────────────────────────────────────────┐
│                    分区策略选择                          │
├─────────────────────────────────────────────────────────┤
│  Key 存在:  Hash(Key) % PartitionCount                  │
│  Key 不存在: Sticky Partition 或 Round Robin            │
└─────────────────────────────────────────────────────────┘

Producer                    Kafka Cluster
   │                        ┌──────────────────┐
   │  Key="order-123"       │ Partition 0      │
   │  ─────────────────────>│ Hash=5           │
   │                        │                  │
   │  Key="order-456"       │ Partition 1      │
   │  ─────────────────────>│ Hash=2           │
   │                        │                  │
   │  Key=null              │ Partition 2      │
   │  ─────────────────────>│ Sticky/Random    │
   │                        └──────────────────┘
```

**内置分区器**：

| 分区器 | 说明 |
|--------|------|
| DefaultPartitioner | Key 存在用 Hash，不存在用 Sticky |
| UniformStickyPartitioner | 始终使用 Sticky（无 Key 场景） |
| RoundRobinPartitioner | 轮询分配（已弃用） |

**自定义分区器**：

```java
// 自定义分区器：按业务规则分区
public class OrderPartitioner implements Partitioner {
    
    @Override
    public int partition(String topic, Object key, byte[] keyBytes, 
                         Object value, byte[] valueBytes, Cluster cluster) {
        
        String orderType = extractOrderType(value.toString());
        
        // 按订单类型分区
        switch (orderType) {
            case "VIP":
                return 0;  // VIP 订单专门分区
            case "NORMAL":
                return 1 + (key.hashCode() & Integer.MAX_VALUE) % (cluster.partitionCountForTopic(topic) - 1);
            default:
                return cluster.partitionCountForTopic(topic) - 1;
        }
    }
    
    @Override
    public void configure(Map<String, ?> configs) {}
    
    @Override
    public void close() {}
}

// 使用自定义分区器
Properties props = new Properties();
props.put("partitioner.class", "com.example.OrderPartitioner");
```

### 🔄 消费者组重平衡（Rebalance）

当 Consumer Group 成员变化或 Partition 数量变化时，触发重平衡。

```
Rebalance 场景:
1. 新 Consumer 加入组
2. Consumer 离开组（主动或崩溃）
3. Consumer 会话超时
4. Partition 数量变化
5. Topic 订阅变化
```

```
Rebalance 过程:

Consumer Group "order-processors"
┌────────────────────────────────────────────────────────┐
│  Before Rebalance                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Consumer 1│  │Consumer 2│  │Consumer 3│            │
│  │ P0, P1   │  │ P2, P3   │  │ P4, P5   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└────────────────────────────────────────────────────────┘
                         │
                     Consumer 2 离开
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  After Rebalance                                       │
│  ┌──────────┐              ┌──────────┐               │
│  │Consumer 1│              │Consumer 3│               │
│  │ P0, P1   │              │ P4, P5   │               │
│  │ P2       │              │ P3       │               │
│  └──────────┘              └──────────┘               │
└────────────────────────────────────────────────────────┘
```

**重平衡协议演进**：

| 协议 | 特点 | 问题 |
|------|------|------|
| Eager（旧） | 全部停止消费，重新分配 | Stop-the-World |
| Cooperative（新） | 渐进式重分配 | 更平滑，但需要多轮 |

**消费者配置优化**：

```java
Properties props = new Properties();

// 会话和心跳
props.put("session.timeout.ms", "45000");       // 会话超时（默认45秒）
props.put("heartbeat.interval.ms", "15000");    // 心跳间隔（默认3秒）
props.put("max.poll.interval.ms", "300000");    // 处理间隔（默认5分钟）

// 静态成员（减少重平衡）
props.put("group.instance.id", "consumer-1");   // 静态成员ID

// 分区分配策略
props.put("partition.assignment.strategy", 
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");

// 重平衡监听器
consumer.subscribe(Collections.singletonList("orders"), new ConsumerRebalanceListener() {
    @Override
    public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
        // 分区被撤销前，提交当前处理进度
        consumer.commitSync();
        log.info("Partitions revoked: {}", partitions);
    }
    
    @Override
    public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
        // 分区分配后，可能需要初始化状态
        log.info("Partitions assigned: {}", partitions);
    }
});
```

**避免不必要的重平衡**：

```java
// 问题1：处理时间过长
// max.poll.interval.ms 应大于单批次最大处理时间
props.put("max.poll.interval.ms", "600000");  // 10分钟

// 问题2：网络抖动
// 适当增加 session.timeout.ms
props.put("session.timeout.ms", "60000");  // 60秒

// 问题3：消费者频繁上下线
// 使用静态成员
props.put("group.instance.id", "static-consumer-1");
// 配合 session.timeout.ms 给予足够恢复时间

// 问题4：GC 导致心跳中断
// 优化 JVM GC 配置，避免长时间 STW
```

## 日志压缩与保留策略

### 🗂️ 日志段结构

Kafka 的日志由多个日志段（Log Segment）组成：

```
Log Segment Structure:
┌─────────────────────────────────────────────────────────┐
│  Segment 1 (Active)                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ .log        │  │ .index      │  │ .timeindex  │     │
│  │ 消息数据     │  │ Offset索引  │  │ 时间戳索引  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                         │
│  Segment 2 (Closed)                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ .log        │  │ .index      │  │ .timeindex  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 保留策略

Kafka 支持两种日志保留策略：

```
Retention Policy:
1. Delete:  超过时间/大小限制后删除旧消息
2. Compact: 保留每个 Key 的最新值
```

**Delete 策略配置**：

```bash
# server.properties 或 Topic 级别配置

# 时间维度保留
log.retention.hours=168           # 保留7天
log.retention.minutes=10080       # 或用分钟
log.retention.ms=604800000        # 或用毫秒（优先级最高）

# 大小维度保留
log.retention.bytes=1073741824    # 单分区最大1GB
retention.bytes=-1                # -1表示无限制

# 日志段大小
log.segment.bytes=1073741824      # 单段1GB

# 清理检查间隔
log.retention.check.interval.ms=300000  # 5分钟检查一次
```

**Compact 策略原理**：

```
Compaction Process:
Before Compaction:
┌─────────────────────────────────────────────┐
│ K1:V1 │ K2:V1 │ K1:V2 │ K3:V1 │ K2:V2 │ K1:V3 │
└─────────────────────────────────────────────┘

After Compaction:
┌─────────────────────────────────────────────┐
│         │ K3:V1 │ K2:V2 │ K1:V3 │           │
└─────────────────────────────────────────────┘
         Tail (可清理)     Head (保留最新)

规则：每个 Key 只保留最新值
```

```bash
# Compact 策略配置
log.cleanup.policy=compact              # 压缩策略
log.cleaner.enable=true                 # 启用清理器
log.cleaner.min.compaction.lag.ms=0     # 最小压缩延迟
log.cleaner.max.compaction.lag.ms=9223372036854775807  # 最大压缩延迟
log.cleaner.min.cleanable.ratio=0.5     # 可清理比例阈值
log.cleaner.delete.retention.ms=86400000 # 删除标记保留时间

# Topic 级别配置
kafka-configs.sh --alter --topic user-state \
  --add-config cleanup.policy=compact \
  --bootstrap-server localhost:9092
```

**适用场景对比**：

| 场景 | 推荐策略 | 说明 |
|------|----------|------|
| 日志收集 | Delete | 时效性数据 |
| 用户状态 | Compact | 只需最新状态 |
| 事件溯源 | Delete | 需要完整历史 |
| 缓存同步 | Compact | Key-Value 场景 |
| 交易记录 | Delete | 不可丢失任何记录 |

### 混合策略

```bash
# 同时使用 Delete 和 Compact
log.cleanup.policy=delete,compact

# 先压缩，再按时间/大小删除
log.retention.hours=168        # 7天后删除
log.cleaner.enable=true        # 同时压缩
```

## 监控指标与运维

### 📈 关键监控指标

**生产者指标**：

```
kafka.producer:type=producer-metrics
├── record-send-rate          # 发送速率（记录/秒）
├── record-error-rate         # 错误率
├── request-latency-avg       # 平均请求延迟
├── batch-size-avg            # 平均批次大小
├── compression-rate          # 压缩率
├── record-queue-time-avg     # 记录排队时间
└── io-wait-time-ns-avg       # IO 等待时间
```

**消费者指标**：

```
kafka.consumer:type=consumer-metrics
├── fetch-rate                # 拉取速率
├── fetch-latency-avg         # 拉取延迟
├── records-consumed-rate     # 消费速率
├── commit-rate               # 提交速率
└── commit-latency-avg        # 提交延迟

kafka.consumer:type=consumer-coordinator-metrics
├── assigned-partitions       # 分配的分区数
├── commit-rate               # 提交频率
└── rebalance-latency-avg     # 重平衡延迟
```

**Broker 指标**：

```
kafka.server:type=BrokerTopicMetrics
├── MessagesInPerSec          # 消息入速率
├── BytesInPerSec             # 字节入速率
├── BytesOutPerSec            # 字节出速率
├── TotalFetchRequestsPerSec  # Fetch 请求速率
└── TotalProduceRequestsPerSec # Produce 请求速率

kafka.server:type=ReplicaManager
├── UnderReplicatedPartitions # 副本不足的分区数 ⚠️
├── OfflinePartitionsCount    # 离线分区数 ⚠️
├── ActiveControllerCount     # 活跃 Controller 数
└── IsrShrinksPerSec          # ISR 收缩速率 ⚠️
```

### 📊 Consumer Lag 监控

Consumer Lag 是最重要的消费健康指标：

```
Lag = Log End Offset - Consumer Offset

┌─────────────────────────────────────────────────────────┐
│ Partition 0                                             │
│ ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐      │
│ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │      │
│ └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘      │
│                           ↑         ↑                   │
│                    Consumer     Log End                  │
│                    Offset       Offset                   │
│                      5           12                      │
│                                                         │
│                    Lag = 12 - 5 = 7                      │
└─────────────────────────────────────────────────────────┘
```

**监控 Lag 的方法**：

```bash
# 命令行查看
kafka-consumer-groups.sh --describe \
  --group order-processor-group \
  --bootstrap-server localhost:9092

# 输出示例：
# GROUP              TOPIC           PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# order-processor    orders          0          1000            1050            50
# order-processor    orders          1          2000            2000            0
# order-processor    orders          2          1500            1520            20
```

**Python 监控脚本**：

```python
from kafka import KafkaAdminClient
from kafka import KafkaConsumer

def check_lag(bootstrap_servers, group_id):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False
    )
    
    # 获取消费组 Offset
    partitions = consumer.partitions_for_topic('orders')
    topic_partitions = [TopicPartition('orders', p) for p in partitions]
    
    end_offsets = consumer.end_offsets(topic_partitions)
    committed = consumer.committed(topic_partitions)
    
    total_lag = 0
    for tp in topic_partitions:
        end_offset = end_offsets[tp]
        committed_offset = committed[tp] if committed[tp] else 0
        lag = end_offset - committed_offset
        total_lag += lag
        print(f"Partition {tp.partition}: Lag = {lag}")
    
    print(f"Total Lag: {total_lag}")
    
    consumer.close()
    admin.close()

# 告警阈值
LAG_THRESHOLD = 10000

if total_lag > LAG_THRESHOLD:
    send_alert(f"Consumer lag too high: {total_lag}")
```

### 🔍 常见运维命令

```bash
# 查看集群概览
kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# 查看 Topic 列表
kafka-topics.sh --list --bootstrap-server localhost:9092

# 查看 Topic 详情
kafka-topics.sh --describe --topic orders --bootstrap-server localhost:9092

# 查看 Consumer Group 列表
kafka-consumer-groups.sh --list --bootstrap-server localhost:9092

# 查看 Consumer Group 详情
kafka-consumer-groups.sh --describe --group my-group --bootstrap-server localhost:9092

# 重置 Consumer Group Offset
kafka-consumer-groups.sh --reset-offsets \
  --group my-group \
  --topic orders \
  --to-earliest \
  --execute \
  --bootstrap-server localhost:9092

# 查看 Broker 配置
kafka-configs.sh --describe \
  --broker 1 \
  --bootstrap-server localhost:9092

# 查看 LogDirs 信息
kafka-log-dirs.sh --describe \
  --broker-list 1,2,3 \
  --bootstrap-server localhost:9092

# 检查集群配置一致性
kafka-cluster.sh cluster-id --bootstrap-server localhost:9092
```

## 安全配置

### 🔐 安全机制概述

Kafka 支持多层安全机制：

```
┌─────────────────────────────────────────────────────────┐
│                    Kafka Security                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              Authentication (认证)               │   │
│  │  • SASL/PLAIN (用户名密码)                       │   │
│  │  • SASL/SCRAM (加盐挑战响应)                     │   │
│  │  • SASL/GSSAPI (Kerberos)                       │   │
│  │  • SASL/OAUTHBEARER (OAuth 2.0)                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Encryption (加密)                   │   │
│  │  • TLS/SSL (传输层加密)                         │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Authorization (授权)                │   │
│  │  • ACL (访问控制列表)                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### SASL/SCRAM 配置（推荐）

```bash
# 1. 创建用户
kafka-configs.sh --zookeeper localhost:2181 \
  --alter --add-config 'SCRAM-SHA-256=[password=secret],SCRAM-SHA-512=[password=secret]' \
  --entity-type users --entity-name admin

# 2. Broker 配置 (server.properties)
listeners=SASL_SSL://:9092
advertised.listeners=SASL_SSL://host:9092

security.inter.broker.protocol=SASL_SSL
sasl.mechanism.inter.broker.protocol=SCRAM-SHA-256
sasl.enabled.mechanisms=SCRAM-SHA-256,SCRAM-SHA-512

# SSL 配置
ssl.keystore.location=/path/to/keystore.jks
ssl.keystore.password=keystore-password
ssl.truststore.location=/path/to/truststore.jks
ssl.truststore.password=truststore-password

# 3. JAAS 配置文件 (kafka_server_jaas.conf)
KafkaServer {
    org.apache.kafka.common.security.scram.ScramLoginModule required
    username="admin"
    password="admin-secret";
};

# 4. 启动时指定 JAAS 文件
export KAFKA_OPTS="-Djava.security.auth.login.config=/path/to/kafka_server_jaas.conf"
bin/kafka-server-start.sh config/server.properties
```

### 客户端安全配置

```java
// Java 客户端安全配置
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");

// 安全协议
props.put("security.protocol", "SASL_SSL");
props.put("sasl.mechanism", "SCRAM-SHA-256");
props.put("sasl.jaas.config",
    "org.apache.kafka.common.security.scram.ScramLoginModule required " +
    "username=\"admin\" password=\"admin-secret\";");

// SSL 配置
props.put("ssl.truststore.location", "/path/to/truststore.jks");
props.put("ssl.truststore.password", "truststore-password");

// 如果使用客户端证书
props.put("ssl.keystore.location", "/path/to/keystore.jks");
props.put("ssl.keystore.password", "keystore-password");
```

```python
# Python 客户端安全配置
from kafka import KafkaProducer, KafkaConsumer

# SASL/PLAIN
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    security_protocol='SASL_PLAINTEXT',
    sasl_mechanism='PLAIN',
    sasl_plain_username='admin',
    sasl_plain_password='admin-secret'
)

# SASL/SCRAM
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    security_protocol='SASL_SSL',
    sasl_mechanism='SCRAM-SHA-256',
    sasl_plain_username='admin',
    sasl_plain_password='admin-secret',
    ssl_cafile='/path/to/ca.pem'
)
```

### ACL 授权配置

```bash
# 启用 ACL
authorizer.class.name=kafka.security.authorizer.AclAuthorizer
allow.everyone.if.no.acl.found=false

# 创建 ACL 规则
# 允许用户 alice 读取 orders Topic
kafka-acls.sh --authorizer-properties zookeeper.connect=localhost:2181 \
  --add --allow-principal User:alice \
  --operation Read --topic orders

# 允许用户 bob 写入 orders Topic
kafka-acls.sh --authorizer-properties zookeeper.connect=localhost:2181 \
  --add --allow-principal User:bob \
  --operation Write --topic orders

# 允许消费组访问
kafka-acls.sh --authorizer-properties zookeeper.connect=localhost:2181 \
  --add --allow-principal User:alice \
  --operation Read --group order-consumers

# 列出所有 ACL
kafka-acls.sh --authorizer-properties zookeeper.connect=localhost:2181 --list

# 删除 ACL
kafka-acls.sh --authorizer-properties zookeeper.connect=localhost:2181 \
  --remove --allow-principal User:alice --operation Read --topic orders
```

## 事件驱动架构设计

### 🏗️ 事件溯源模式

```
Traditional CRUD:
┌─────────┐     ┌─────────┐
│  App    │────>│ Database│  只保存当前状态
└─────────┘     └─────────┘

Event Sourcing:
┌─────────┐     ┌─────────┐
│  App    │────>│  Kafka  │  保存所有状态变更事件
└─────────┘     │  Event  │
                │  Store  │
                └─────────┘
                      │
                      ▼
                ┌─────────┐
                │  View   │  从事件重建状态
                │  Store  │
                └─────────┘
```

**事件溯源实现**：

```java
// 订单事件定义
public abstract class OrderEvent {
    private String orderId;
    private long timestamp;
    // ...
}

public class OrderCreated extends OrderEvent {
    private String customerId;
    private List<OrderItem> items;
    private BigDecimal totalAmount;
}

public class OrderPaid extends OrderEvent {
    private String paymentId;
    private BigDecimal paidAmount;
}

public class OrderShipped extends OrderEvent {
    private String trackingNumber;
}

public class OrderCancelled extends OrderEvent {
    private String reason;
}

// 事件生产者
public class OrderService {
    private final KafkaProducer<String, OrderEvent> producer;
    
    public void createOrder(String orderId, CreateOrderRequest request) {
        OrderCreated event = new OrderCreated(
            orderId, request.getCustomerId(), request.getItems()
        );
        producer.send(new ProducerRecord<>("order-events", orderId, event));
    }
    
    public void payOrder(String orderId, String paymentId, BigDecimal amount) {
        OrderPaid event = new OrderPaid(orderId, paymentId, amount);
        producer.send(new ProducerRecord<>("order-events", orderId, event));
    }
}

// 事件消费者（重建状态）
public class OrderProjection {
    private final Map<String, OrderState> stateStore = new ConcurrentHashMap<>();
    
    public void processEvent(OrderEvent event) {
        OrderState state = stateStore.computeIfAbsent(
            event.getOrderId(), k -> new OrderState()
        );
        
        if (event instanceof OrderCreated) {
            OrderCreated e = (OrderCreated) event;
            state.setStatus("CREATED");
            state.setItems(e.getItems());
            state.setTotalAmount(e.getTotalAmount());
        } else if (event instanceof OrderPaid) {
            state.setStatus("PAID");
            state.setPaidAmount(((OrderPaid) event).getPaidAmount());
        } else if (event instanceof OrderShipped) {
            state.setStatus("SHIPPED");
        } else if (event instanceof OrderCancelled) {
            state.setStatus("CANCELLED");
        }
    }
}
```

### 📨 CQRS 模式

```
Command Query Responsibility Segregation:

┌─────────────────────────────────────────────────────────┐
│                      CQRS Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │ Command │────>│  Kafka  │────>│ Command │           │
│  │   API   │     │  Topic  │     │ Handler │           │
│  └─────────┘     └─────────┘     └────┬────┘           │
│                                        │                │
│                                        ▼                │
│                                  ┌─────────┐           │
│                                  │  Event  │           │
│                                  │  Store  │           │
│                                  └────┬────┘           │
│                                       │                │
│                                       ▼                │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │  Query  │<────│   Read  │<────│ Project │           │
│  │   API   │     │  Model  │     │   ion   │           │
│  └─────────┘     └─────────┘     └─────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 🔗 Saga 编排模式

```java
// 订单处理 Saga
public class OrderSaga {
    
    private final KafkaProducer<String, SagaEvent> producer;
    
    public void handle(OrderCreated event) {
        // Step 1: 预留库存
        ReserveInventory command = new ReserveInventory(
            event.getOrderId(), event.getItems()
        );
        producer.send(new ProducerRecord<>("inventory-commands", 
            event.getOrderId(), command));
    }
    
    public void handle(InventoryReserved event) {
        // Step 2: 处理支付
        ProcessPayment command = new ProcessPayment(
            event.getOrderId(), event.getTotalAmount()
        );
        producer.send(new ProducerRecord<>("payment-commands",
            event.getOrderId(), command));
    }
    
    public void handle(PaymentProcessed event) {
        // Step 3: 安排配送
        ArrangeShipping command = new ArrangeShipping(
            event.getOrderId(), event.getShippingAddress()
        );
        producer.send(new ProducerRecord<>("shipping-commands",
            event.getOrderId(), command));
    }
    
    // 补偿事务
    public void handle(PaymentFailed event) {
        // 回滚库存
        ReleaseInventory command = new ReleaseInventory(event.getOrderId());
        producer.send(new ProducerRecord<>("inventory-commands",
            event.getOrderId(), command));
        
        // 取消订单
        CancelOrder command = new CancelOrder(event.getOrderId(), "Payment failed");
        producer.send(new ProducerRecord<>("order-commands",
            event.getOrderId(), command));
    }
}
```

## 成本优化策略

### 💰 存储优化

```bash
# 1. 合理设置保留时间
# 根据业务需求设置，避免过长保留
log.retention.hours=72  # 3天而非默认7天

# 2. 使用日志压缩
# 对于 Key-Value 类数据
cleanup.policy=compact

# 3. 分层存储（Kafka 3.x）
# 冷数据自动迁移到廉价存储
remote.log.storage.system.enable=true
remote.log.storage.manager.class.path=...
```

### 💰 网络优化

```java
// 批处理和压缩
props.put("batch.size", "65536");      // 增大批次
props.put("linger.ms", "10");          // 适当等待
props.put("compression.type", "zstd"); // 高压缩率

// 消费者优化
props.put("fetch.min.bytes", "1048576");  // 1MB 最小拉取
props.put("fetch.max.wait.ms", "500");    // 500ms 最大等待
```

### 💰 集群规划

```
┌─────────────────────────────────────────────────────────┐
│                集群规模估算                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  吞吐量需求: 100MB/s                                     │
│  单 Broker 能力: 50MB/s                                  │
│  需要Broker数: 100 / 50 * 2 (冗余) = 4                  │
│                                                         │
│  存储需求: 10TB                                          │
│  单磁盘容量: 2TB                                         │
│  需要磁盘: 10 / 2 * 3 (副本) = 15                        │
│                                                         │
│  内存需求: Page Cache = 吞吐量 * 30秒                    │
│          = 100MB * 30 = 3GB per Broker                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 常见问题与最佳实践

### ⚠️ 消息丢失问题

```java
// 避免消息丢失的最佳实践

// 生产者端
Properties props = new Properties();
props.put("acks", "all");                    // 等待所有副本确认
props.put("enable.idempotence", "true");     // 启用幂等性
props.put("retries", "Integer.MAX_VALUE");   // 无限重试
props.put("max.in.flight.requests.per.connection", "5"); // 保持顺序

// Broker 端
// min.insync.replicas=2  // 至少2个副本同步
// unclean.leader.election.enable=false  // 禁止非同步副本成为Leader

// 消费者端
props.put("enable.auto.commit", "false");   // 手动提交
// 处理成功后再提交
consumer.commitSync();
```

### ⚠️ 消息重复问题

```java
// 消费端幂等性设计

// 方案1：数据库唯一约束
@Transactional
public void processOrder(OrderEvent event) {
    // 利用数据库唯一约束
    orderRepository.saveWithUniqueConstraint(event);
}

// 方案2：Redis 分布式锁
public void processWithDedup(String messageId, ConsumerRecord record) {
    String key = "processed:" + messageId;
    Boolean success = redis.setIfAbsent(key, "1", Duration.ofDays(1));
    if (success) {
        process(record);
    }
}

// 方案3：本地缓存（单机）
private final Cache<String, Boolean> processedCache = 
    Caffeine.newBuilder()
        .expireAfterWrite(Duration.ofHours(24))
        .maximumSize(100000)
        .build();
```

### ⚠️ 消息积压问题

```java
// 诊断和处理消息积压

// 1. 监控 Lag
if (lag > LAG_THRESHOLD) {
    // 告警
    alertService.sendAlert("Consumer lag too high: " + lag);
}

// 2. 临时扩容
// 增加 Consumer 实例（注意 Partition 数量限制）

// 3. 消费提速
// 增加并行度
props.put("max.poll.records", "1000");  // 增加单次拉取
// 使用多线程处理
ExecutorService executor = Executors.newFixedThreadPool(10);

// 4. 跳过积压（紧急情况）
kafka-consumer-groups.sh --reset-offsets \
  --group my-group --topic orders \
  --to-latest --execute

// 5. 转存后异步处理
// 将积压消息转发到新 Topic，用新消费者处理
```

## 参考资料

- [Apache Kafka 官方文档](https://kafka.apache.org/documentation/)
- [Kafka KRaft 模式指南](https://developer.confluent.io/learn/kraft/)
- [Kafka 权威指南](https://book.douban.com/subject/27179853/)
- [Kafka Definitive Guide](https://www.confluent.io/resources/kafka-the-definitive-guide/)
