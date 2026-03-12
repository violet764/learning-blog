# Kafka 进阶面试题

本文是 Kafka 面试题的进阶篇，聚焦于更深入的技术细节和实战场景。适合有一定 Kafka 基础，想要深入理解底层原理和解决复杂问题的开发者。

---

## 一、高性能原理篇

### 1. Kafka 顺序写为什么比随机写快？

**答案：**

Kafka 的高性能很大程度上依赖于**顺序写**，这是其核心设计理念之一。

#### 磁盘 IO 的性能差异

| 操作类型 | 机械硬盘 | SSD | 说明 |
|---------|---------|-----|------|
| 顺序写 | 100-200 MB/s | 500-3000 MB/s | 磁头移动少，预读有效 |
| 随机写 | 0.5-2 MB/s | 50-200 MB/s | 大量寻道时间，性能下降 100 倍 |

#### 为什么顺序写快？

```
机械硬盘结构：
┌─────────────────────────────────────┐
│  磁盘盘面                           │
│    ┌─────────────────────────┐      │
│    │    ┌───────────────┐    │      │
│    │    │   磁头臂       │    │      │
│    │    │      ↓        │    │      │
│    │    │   ┌───┐       │    │      │
│    │    │   │磁头│       │    │      │
│    │    └───┴───┘       │    │      │
│    │        磁道         │    │      │
│    └─────────────────────────┘      │
└─────────────────────────────────────┘

顺序写：磁头基本不动，持续写入
随机写：磁头频繁移动，寻道时间占比高
```

#### Kafka 的实现方式

```java
// Kafka 日志追加写入的核心逻辑
public class LogSegment {
    private final FileRecords log;
    
    // 追加写入 - 顺序写
    public LogAppendInfo append(long offset, byte[] records) {
        // 直接追加到文件末尾
        long position = log.sizeInBytes();  // 当前位置
        log.append(records);  // 顺序追加
        return new LogAppendInfo(offset, position);
    }
}
```

#### 对比传统消息队列

```yaml
# Kafka: Append-Only Log
Producer → [Topic-Partition-1] → 文件末尾追加
Producer → [Topic-Partition-2] → 文件末尾追加

# RabbitMQ: 随机写
Producer → Queue → 消息随机插入到索引结构
Consumer → 随机读取 → 消息状态更新
```

**追问：那读取时不是要随机读吗？**

**答案：**

Kafka 通过以下方式优化读取：

```java
// 1. 稀疏索引 - 不索引每条消息
public class OffsetIndex {
    // 每隔 4KB 建立一个索引项
    private static final int INDEX_INTERVAL_BYTES = 4096;
    
    public OffsetPosition lookup(long targetOffset) {
        // 二分查找索引
        int slot = binarySearch(entries, targetOffset);
        // 返回近似位置，然后顺序扫描
        return entries[slot];
    }
}
```

```yaml
# 2. 页缓存 + 预读
# 操作系统会自动预读相邻数据
# Consumer 通常是顺序消费，预读命中率极高

# 3. Consumer 顺序消费
# 大多数场景下 Consumer 按顺序消费
# 从上次消费位置继续，本身就是顺序读
```

---

### 2. Kafka 零拷贝是怎么实现的？sendfile 和 mmap 的区别？

**答案：**

零拷贝（Zero Copy）是 Kafka 高吞吐的关键技术，它减少了数据在内核态和用户态之间的拷贝次数。

#### 传统 IO 的数据拷贝过程

```
传统读取文件并发送到网络：

磁盘 → 内核缓冲区 → 用户缓冲区 → 内核Socket缓冲区 → 网卡
  │        │           │            │            │
  └──DMA───┴──CPU拷贝──┴──CPU拷贝───┴──DMA拷贝───┘

总共：4 次拷贝（2次DMA + 2次CPU），4 次上下文切换
```

#### sendfile 零拷贝

```
sendfile 零拷贝：

磁盘 → 内核缓冲区 → 网卡
  │        │        │
  └──DMA───┴──DMA───┘

总共：2 次拷贝（2次DMA），2 次上下文切换
CPU 不参与数据拷贝！
```

#### Kafka 中的实现

```java
// FileRecords.java - Kafka 使用 FileChannel.transferTo
public class FileRecords extends AbstractRecords {
    
    public long writeTo(GatheringByteChannel channel, long position, int length) 
            throws IOException {
        // 使用 transferTo 实现零拷贝
        return fileRecords.transferTo(channel, position, length);
    }
}

// 底层调用 Linux sendfile 系统调用
// int sendfile(int out_fd, int in_fd, off_t *offset, size_t count);
```

#### mmap 的实现和区别

```java
// mmap - 内存映射文件
public class MappedByteBuffer {
    // 将文件映射到内存
    // 用户空间直接访问内核缓冲区
}

// Kafka 的索引文件使用 mmap
public class OffsetIndex {
    private final MappedByteBuffer mmap;
    
    public OffsetPosition lookup(long offset) {
        // 直接从 mmap 读取，无需系统调用
        return mmap.getInt(offset);  // 零拷贝
    }
}
```

#### sendfile vs mmap 对比

| 特性 | sendfile | mmap |
|-----|----------|------|
| 适用场景 | 文件 → 网络 | 文件 → 内存 |
| 拷贝次数 | 2次 DMA | 2次 DMA + 1次 CPU |
| CPU 参与 | 无 | 有（映射时） |
| 适用数据 | 大文件传输 | 随机访问、小文件 |
| Kafka 使用 | 日志传输 | 索引文件 |

```yaml
# Kafka 配置优化
log.dirs=/data/kafka-logs  # 使用 SSD 或高速磁盘

# 零拷贝依赖操作系统支持
# Linux: sendfile 系统调用
# Windows: TransmitFile API
```

**追问：Kafka 为什么比 RabbitMQ 快？**

**答案：**

```yaml
性能对比（单机）：
  Kafka: 100万+ TPS
  RabbitMQ: 5-10万 TPS

差异原因：

1. 存储模型：
  Kafka: 顺序写 + 零拷贝
  RabbitMQ: 随机写 + 传统 IO

2. 消息消费：
  Kafka: Pull 模式，批量拉取
  RabbitMQ: Push 模式，单条推送

3. 协议：
  Kafka: 自定义二进制协议
  RabbitMQ: AMQP 协议（更复杂）

4. 功能定位：
  Kafka: 追求吞吐，适合日志、大数据
  RabbitMQ: 追求可靠性，适合业务消息
```

---

### 3. Kafka 页缓存的作用？怎么配置？

**答案：**

页缓存（Page Cache）是操作系统提供的文件缓存机制，Kafka 充分利用它来实现高性能。

#### 页缓存的工作原理

```
┌─────────────────────────────────────────────────┐
│                   物理内存                       │
│  ┌───────────────────────────────────────────┐  │
│  │              页缓存 (Page Cache)           │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         │  │
│  │  │Page1│ │Page2│ │Page3│ │Page4│ ...     │  │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘         │  │
│  └─────┼───────┼───────┼───────┼─────────────┘  │
│        │       │       │       │                │
└────────┼───────┼───────┼───────┼────────────────┘
         │       │       │       │
    ┌────┴───┐ ┌─┴────┐ ┌┴─────┐ ┌┴─────┐
    │数据块1 │ │数据块2│ │数据块3│ │数据块4│
    └────────┘ └───────┘ └──────┘ └──────┘
         磁盘上的文件数据块
```

#### Kafka 与页缓存的交互

```java
// Producer 写入流程
Producer.send()
    ↓
Kafka Broker 接收请求
    ↓
写入页缓存（不直接刷盘！）
    ↓
立即返回 ACK  ← 非常快！
    ↓
（后台异步刷盘）

// Consumer 读取流程
Consumer.poll()
    ↓
Broker 检查页缓存
    ↓
命中？→ 直接返回（内存速度）
    ↓
未命中？→ 从磁盘加载到页缓存 → 返回
```

#### 关键配置参数

```yaml
# broker 配置

# 日志刷盘策略（通常依赖操作系统）
log.flush.interval.messages=10000    # 每 N 条消息刷盘
log.flush.interval.ms=1000           # 每 N 毫秒刷盘

# 建议：让操作系统管理刷盘
# Kafka 默认不主动刷盘，依赖页缓存

# 操作系统层面优化
# /etc/sysctl.conf
vm.dirty_ratio=80                    # 脏页占比达到 80% 才刷盘
vm.dirty_background_ratio=5          # 后台开始刷盘的阈值
vm.swappiness=1                      # 尽量不使用 swap
```

#### 生产环境配置建议

```yaml
# 1. 内存分配
# Broker 堆内存：4-8GB 即可
# 剩余内存：留给页缓存！

# JVM 启动参数
KAFKA_HEAP_OPTS="-Xms6g -Xmx6g"

# 2. 日志目录配置
# 多块磁盘提高并行度
log.dirs=/disk1/kafka,/disk2/kafka,/disk3/kafka

# 3. 禁用 swap
sudo swapoff -a
# 或设置 vm.swappiness=1
```

#### 页缓存命中率监控

```bash
# 查看页缓存使用情况
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           31Gi       8.0Gi       2.0Gi       1.0Gi        21Gi        22Gi

# 查看 Kafka 进程的页缓存
$ vmtouch -v /data/kafka-logs/topic-0/*.log

# 使用 cachestat 监控缓存命中率
$ cachestat 1
    HITS   MISSES  DIRTIES  RATIO   BUFFERS_MB  CACHED_MB
    1234      567      123   68.5%         456       7890
```

**追问：页缓存数据丢失怎么办？**

**答案：**

```yaml
数据丢失风险分析：

1. 写入页缓存但未刷盘 → 机器宕机 → 数据丢失

2. Kafka 的解决方案：
   - 多副本机制：同步到多个 Broker
   - ISR 机制：确保至少同步到 ISR 列表
   - acks=all：等待所有 ISR 副本确认

3. 生产环境最佳实践：
   - acks=all
   - min.insync.replicas=2
   - replication.factor=3
   
   即使一个 Broker 宕机，数据仍然安全！

4. 与 RabbitMQ 的对比：
   - RabbitMQ: 每条消息都刷盘
   - Kafka: 依赖副本 + 页缓存
   - 性能差距的核心原因
```


### 4. Kafka 批处理的最佳大小怎么确定？

**答案：**

批处理是 Kafka 提升吞吐的重要手段，但批次大小的选择需要权衡吞吐和延迟。

#### 批处理的工作流程

```
Producer 端批处理：

┌─────────────────────────────────────────────────┐
│                   Producer                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │msg1 │ │msg2 │ │msg3 │ │msg4 │ ...          │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘              │
│     └───────┴───────┴───────┘                  │
│             ↓                                   │
│     ┌─────────────────┐                        │
│     │   RecordAccumulator    │                 │
│     │   按分区组织批次        │                 │
│     │  ┌───┐ ┌───┐ ┌───┐    │                 │
│     │  │Batch1│ │Batch2│ │Batch3│  │          │
│     │  │P0   │ │P1   │ │P2   │  │            │
│     │  └───┘ └───┘ └───┘    │                 │
│     └─────────────────┘                        │
│             ↓                                   │
│        Sender 线程发送                         │
└─────────────────────────────────────────────────┘
```

#### 关键配置参数

```yaml
# Producer 批处理配置
batch.size=16384          # 批次最大字节数 (默认 16KB)
linger.ms=0               # 等待时间 (默认 0ms)
buffer.memory=33554432    # 缓冲区总大小 (默认 32MB)

# 批次大小对性能的影响
# batch.size 太小：
#   - 网络请求频繁
#   - 吞吐量低
# batch.size 太大：
#   - 内存占用高
#   - 延迟增加
```

#### 如何确定最佳批次大小

```java
// 性能测试代码示例
public class BatchSizeBenchmark {
    
    public static void main(String[] args) {
        int[] batchSizes = {16384, 32768, 65536, 131072, 262144};
        long[] lingerMs = {0, 5, 10, 20, 50};
        
        for (int batchSize : batchSizes) {
            for (long linger : lingerMs) {
                // 配置 Producer
                Properties props = new Properties();
                props.put("batch.size", batchSize);
                props.put("linger.ms", linger);
                props.put("compression.type", "lz4");
                
                // 测试吞吐量
                double throughput = measureThroughput(props);
                double latency = measureLatency(props);
                
                System.out.printf("batch=%d, linger=%d → throughput=%.2f TPS, latency=%.2f ms%n",
                    batchSize, linger, throughput, latency);
            }
        }
    }
}
```

#### 不同场景的最佳配置

```yaml
# 场景1：高吞吐（日志收集）
batch.size=1048576        # 1MB
linger.ms=50              # 等待 50ms 凑批次
compression.type=lz4      # 压缩
buffer.memory=67108864    # 64MB 缓冲区

# 场景2：低延迟（实时交易）
batch.size=16384          # 16KB（默认值）
linger.ms=0               # 不等待
compression.type=none     # 不压缩（减少CPU）

# 场景3：均衡（一般业务）
batch.size=32768          # 32KB
linger.ms=5               # 等待 5ms
compression.type=lz4
```

#### 监控批次效率

```bash
# 使用 Kafka Producer Metrics 监控
# 关键指标：
# - batch-size-avg: 平均批次大小
# - batch-size-max: 最大批次大小
# - compression-rate: 压缩率
# - record-send-rate: 发送速率

# JMX 指标
kafka.producer:type=producer-metrics,client-id=*
  - record-send-rate
  - batch-size-avg
  - compression-rate-avg
```

---

### 5. Kafka 压缩算法怎么选？各有什么优缺点？

**答案：**

Kafka 支持多种压缩算法，选择合适的压缩算法可以显著提升性能。

#### 支持的压缩算法

| 算法 | 压缩率 | 压缩速度 | 解压速度 | CPU 消耗 | 适用场景 |
|-----|--------|---------|---------|---------|---------|
| none | 0% | 最快 | 最快 | 最低 | 网络带宽充足 |
| gzip | 60-70% | 慢 | 快 | 高 | 带宽受限、数据重复率高 |
| snappy | 40-50% | 快 | 快 | 低 | 实时系统、低延迟 |
| lz4 | 50-60% | 最快 | 最快 | 最低 | 高吞吐场景 |
| zstd | 60-70% | 中等 | 快 | 中 | 新版本、综合最优 |

#### 压缩配置

```yaml
# Producer 端压缩
compression.type=lz4

# 各算法特点详解
```

```java
// 压缩性能测试
public class CompressionBenchmark {
    
    public static void main(String[] args) throws Exception {
        String data = generateTestData(1024 * 1024); // 1MB 数据
        
        // GZIP
        long start = System.currentTimeMillis();
        byte[] gzip = compressGzip(data);
        System.out.printf("GZIP: %d ms, ratio=%.2f%%%n", 
            System.currentTimeMillis() - start,
            100.0 * (1 - gzip.length / (double)data.length()));
        
        // LZ4
        start = System.currentTimeMillis();
        byte[] lz4 = compressLz4(data);
        System.out.printf("LZ4: %d ms, ratio=%.2f%%%n",
            System.currentTimeMillis() - start,
            100.0 * (1 - lz4.length / (double)data.length()));
        
        // ZSTD
        start = System.currentTimeMillis();
        byte[] zstd = compressZstd(data);
        System.out.printf("ZSTD: %d ms, ratio=%.2f%%%n",
            System.currentTimeMillis() - start,
            100.0 * (1 - zstd.length / (double)data.length()));
    }
}
```

#### 端到端压缩流程

```
┌─────────────────────────────────────────────────────────────┐
│                    压缩流程                                  │
│                                                              │
│  Producer            Broker              Consumer            │
│  ┌────────┐         ┌────────┐          ┌────────┐          │
│  │ 原始   │         │        │          │        │          │
│  │ 消息   │  ─────→ │ 压缩   │ ──────→  │ 解压   │          │
│  │ 批次   │         │ 批次   │          │ 消息   │          │
│  └────────┘         └────────┘          └────────┘          │
│      ↓                  ↓                   ↓               │
│   压缩               直接存储             解压消费           │
│  （一次）            （不解压）           （一次）           │
│                                                              │
│  优势：Broker 不解压，减少 CPU 开销                          │
└─────────────────────────────────────────────────────────────┘
```

#### 生产环境建议

```yaml
# 推荐：LZ4 或 ZSTD

# LZ4 - 高吞吐场景首选
compression.type=lz4
# 优点：压缩解压都很快，CPU 消耗低
# 缺点：压缩率一般

# ZSTD - Kafka 2.1.0+ 支持
compression.type=zstd
# 优点：压缩率接近 gzip，速度接近 lz4
# 缺点：需要较新版本

# 不推荐 GZIP
# 虽然压缩率高，但 CPU 消耗大
# 适合带宽极度受限、CPU 资源充足的场景
```

**追问：压缩在 Producer 还是 Broker 做？**

**答案：**

```yaml
压缩位置选择：

1. Producer 压缩（推荐）：
   - 生产者压缩，Broker 直接存储压缩数据
   - 网络传输量小
   - Broker CPU 开销低
   - Consumer 解压一次即可

2. Broker 压缩：
   # broker 端配置
   compression.type=producer  # 默认，使用 producer 的压缩
   compression.type=lz4       # broker 强制压缩
   
   - 不推荐 broker 强制压缩
   - 会增加 broker CPU 负担
   - 可能导致重复压缩

3. 最佳实践：
   # Producer
   compression.type=lz4
   
   # Broker
   compression.type=producer  # 保持 producer 压缩不变
```

---

### 6. Kafka 的 IO 模型是什么？为什么高效？

**答案：**

Kafka 采用 Reactor 模式 + Java NIO，实现了高并发的网络 IO 处理。

#### Reactor 模式架构

```
Kafka Broker 网络层架构：

                    ┌─────────────────────────────────┐
                    │         Acceptor 线程           │
                    │     (监听新连接，轮询模式)       │
                    └─────────────┬───────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ↓                   ↓                   ↓
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   Processor 1   │ │   Processor 2   │ │   Processor N   │
    │   (Selector)    │ │   (Selector)    │ │   (Selector)    │
    │                 │ │                 │ │                 │
    │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
    │ │ Channel 1   │ │ │ │ Channel 1   │ │ │ │ Channel 1   │ │
    │ │ Channel 2   │ │ │ │ Channel 2   │ │ │ │ Channel 2   │ │
    │ │ ...         │ │ │ │ ...         │ │ │ │ ...         │ │
    │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ↓
                    ┌─────────────────────────────────┐
                    │       Request Handler Pool      │
                    │    (工作线程池，处理业务逻辑)    │
                    │  ┌─────┐ ┌─────┐ ┌─────┐       │
                    │  │ T1  │ │ T2  │ │ T3  │ ...   │
                    │  └─────┘ └─────┘ └─────┘       │
                    └─────────────────────────────────┘
```

#### 核心组件代码解析

```java
// SocketServer.scala - Acceptor
class Acceptor(val endPoint: EndPoint) extends AbstractServerThread {
    
    private val nioSelector = NSelector.open()
    private val serverChannel = openServerSocket(endPoint.port)
    
    override def run(): Unit = {
        serverChannel.register(nioSelector, SelectionKey.OP_ACCEPT)
        while (isRunning) {
            // 轮询等待新连接
            val ready = nioSelector.select(500)
            if (ready > 0) {
                val keys = nioSelector.selectedKeys().iterator()
                while (keys.hasNext) {
                    val key = keys.next()
                    if (key.isAcceptable) {
                        // 接受新连接，分配给 Processor
                        val channel = serverChannel.accept()
                        assignNewConnection(channel)
                    }
                }
            }
        }
    }
}
```

```java
// Processor 处理 IO
class Processor(val id: Int) extends AbstractServerThread {
    
    private val selector = KSelector.open()
    private val newConnections = new ConcurrentLinkedQueue[SocketChannel]()
    
    override def run(): Unit = {
        while (isRunning) {
            // 处理新连接
            configureNewConnections()
            
            // 轮询 IO 事件
            selector.poll(300)
            
            // 处理读事件
            val readChannels = selector.completedReceives()
            for (receive <- readChannels) {
                // 提交到请求队列
                requestQueue.put(receive)
            }
            
            // 处理写事件
            val writeChannels = selector.completedSends()
            // ...
        }
    }
}
```

#### 配置优化

```yaml
# broker 端网络配置

# Processor 线程数（建议 = CPU 核数）
num.network.threads=3

# IO 线程数（建议 = 2 * CPU 核数）
num.io.threads=8

# 网络缓冲区配置
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# 请求队列大小
queued.max.requests=500
```

#### 与传统 BIO 对比

```java
// 传统 BIO - 每个连接一个线程
ServerSocket server = new ServerSocket(9092);
while (true) {
    Socket client = server.accept();  // 阻塞
    new Thread(() -> {
        // 每个连接一个线程
        // 1万连接 = 1万线程！
        handleClient(client);
    }).start();
}

// Kafka NIO - Reactor 模式
// 1个 Acceptor + N 个 Processor
// 可以处理数十万连接
```

---

### 7. Kafka 分区数怎么确定？越多越好吗？

**答案：**

分区数是影响 Kafka 性能的关键参数，但并非越多越好。

#### 分区数的影响因素

```
分区数与性能关系：

                    吞吐量
                      ↑
                      │           ┌─────────── 饱和
                      │          /│
                      │         / │
                      │        /  │
                      │       /   │ ───── 过多分区，性能下降
                      │      /    │
                      │     /     │
                      │    /      │
                      │   /       │
                      │  /        │
                      │ /         │
                      └───────────┼─────────────→ 分区数
                      0          最优分区数

原因：
1. 分区过少：并行度不够，吞吐受限
2. 分区过多：元数据开销、内存占用、Rebalance 时间增加
```

#### 分区数计算方法

```yaml
# 方法1：基于吞吐量计算
目标吞吐量 = T (条/秒)
单分区吞吐量 = P (条/秒)
分区数 = ceil(T / P)

# 示例：
# 目标：100万 TPS
# 单分区实测：5万 TPS
# 分区数 = ceil(1000000 / 50000) = 20

# 方法2：基于 Consumer 并行度
# 分区数 >= Consumer 实例数
# 否则会有 Consumer 空闲
```

#### 分区过多的代价

```java
// 1. Broker 端内存开销
// 每个分区都需要：
// - Log 对象
// - 多个索引文件（offset、time、txn）
// - 副本管理器中的 Partition 对象

// 经验值：每个分区约占用 1-2MB 堆内存
// 1000 个分区 ≈ 1-2GB 堆内存

// 2. Controller 元数据管理
// Controller 需要管理所有分区的状态
// 分区越多，Controller 压力越大

// 3. Rebalance 时间增加
// 分区分配的计算复杂度与分区数正相关
```

#### 生产环境建议

```yaml
# 经验值
单 Broker 分区数：< 4000
单集群总分区数：< 20000

# 推荐做法
1. 从小规模开始，根据监控逐步扩展
2. 分区数 = Consumer 数量 × (1~2)
3. 考虑未来扩展，预留一定余量

# 配置示例
# Topic 创建时指定
bin/kafka-topics.sh --create \
  --topic my-topic \
  --partitions 12 \
  --replication-factor 3 \
  --bootstrap-server localhost:9092
```

#### 监控分区健康度

```bash
# 查看分区分布
$ bin/kafka-topics.sh --describe --topic my-topic \
  --bootstrap-server localhost:9092

# 监控指标
# - Under Replicated Partitions
# - Offline Partitions Count  
# - Partition Count per Broker
```

---

### 8. Kafka 为什么不支持读写分离？

**答案：**

Kafka 采用主写主读模式，不支持从副本提供读服务，这是基于性能和一致性的综合考量。

#### 为什么不支持读写分离

```
传统主从读写分离：

                    写请求
                      ↓
              ┌───────────────┐
              │   Master      │
              │   (Leader)    │
              └───────┬───────┘
                      │ 同步复制
              ┌───────┴───────┐
              ↓               ↓
        ┌───────────┐  ┌───────────┐
        │  Slave 1  │  │  Slave 2  │
        │  (Read)   │  │  (Read)   │
        └───────────┘  └───────────┘
              ↑               ↑
         读请求          读请求

问题：
1. 数据一致性问题（主从延迟）
2. 同步机制复杂
3. 消息场景不适合读写分离
```

#### Kafka 的主写主读模式

```yaml
Kafka 架构：

Leader：处理所有读写请求
Follower：只做副本同步，不处理客户端请求

优点：
1. 强一致性：
   - Consumer 只从 Leader 读取
   - 不会读到旧数据

2. 简单高效：
   - 无需处理读写分离的一致性问题
   - 消息生产消费天然有序

3. 适合消息场景：
   - 消息通常生产后立即消费
   - 生产消费在同一分区，都在 Leader
   - 页缓存命中率高
```

#### 消息场景 vs 数据库场景

```yaml
数据库场景（适合读写分离）：
- 读多写少
- 读请求可容忍短暂不一致
- 可以利用多个从库分担读压力

消息队列场景（不适合读写分离）：
- 生产和消费速率相近
- 消费通常是顺序的，页缓存命中率高
- Leader 已经能高效处理所有请求
- 读写分离反而增加复杂度
```

#### Follower 可用场景

```java
// Follower 虽然不处理客户端请求，但有以下用途：

// 1. 副本同步 - 保证数据可靠性
// 2. 成为 Leader 候选 - 故障时接管
// 3. 提供 Fetch 请求 - 其他副本同步数据

// 特殊情况：Consumer 可以从 Follower 读取（需要配置）
// 用于跨机房场景
Properties props = new Properties();
props.put("client.rack", "rack1");  // 指定机架
// Consumer 会优先从同机架的副本读取（如果它是 Leader）
// 注意：这仍然是读 Leader，只是选择就近的 Leader
```


---

## 二、消息可靠性深度篇

### 1. ISR 的动态调整机制是怎样的？

**答案：**

ISR（In-Sync Replicas）是 Kafka 保证数据可靠性的核心机制，它会动态调整以保证可用性和可靠性的平衡。

#### ISR 的工作原理

```
ISR 动态调整流程：

┌─────────────────────────────────────────────────────────────┐
│                        Controller                            │
│                      (ISR 管理)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Leader   │   │ Follower  │   │ Follower  │
    │  (ISR)    │←──│   (ISR)   │←──│  (OSR)    │
    │  Partition│   │  同步中   │   │  落后太多  │
    └───────────┘   └───────────┘   └───────────┘
          │               ↑               │
          │    ISR 扩张   │   ISR 收缩    │
          └───────────────┴───────────────┘

ISR 列表动态变化：
- 同步进度达标 → OSR 进入 ISR
- 同步进度落后 → ISR 退出到 OSR
```

#### 关键参数配置

```yaml
# ISR 调整的核心参数

# 副本落后阈值（时间）
replica.lag.time.max.ms=30000  # 默认 30 秒
# 如果 Follower 超过此时间未同步，会被踢出 ISR

# 最小 ISR 大小
min.insync.replicas=2  # 默认 1
# ISR 中最少需要多少副本才能接受写入

# ISR 扩张检查周期
replica.lag.time.max.ms=30000
```

#### ISR 变化的触发场景

```java
// 1. Follower 落后被踢出 ISR
// 场景：Follower 宕机、网络延迟、磁盘 IO 慢

// Leader 端检测逻辑
class ReplicaManager {
    def maybeExpandOrShrinkIsr(partition: Partition): Unit = {
        val isr = partition.isr
        val outOfSyncReplicas = isr.filter { replica =>
            // 检查是否超过同步时间阈值
            val lastCaughtUpTime = replica.lastCaughtUpTimeMs
            val currentTime = time.milliseconds()
            currentTime - lastCaughtUpTime > replicaLagTimeMaxMs
        }
        
        if (outOfSyncReplicas.nonEmpty) {
            // 从 ISR 中移除
            partition.shrinkIsr(outOfSyncReplicas)
        }
    }
}

// 2. OSR 追上后加入 ISR
// 场景：Follower 恢复、追赶进度完成

def maybeExpandIsr(): Unit = {
    val leaderEndOffset = localLog.logEndOffset
    for (replica <- replicaMap.values) {
        // 检查是否追上 Leader
        if (!isr.contains(replica) && 
            replica.logEndOffset >= leaderEndOffset) {
            // 加入 ISR
            partition.expandIsr(replica)
        }
    }
}
```

#### ISR 变化的监控

```bash
# 关键监控指标

# 1. Under Replicated Partitions
# ISR 中副本数 < 配置副本数的分区
$ bin/kafka-topics.sh --describe --under-replicated-partitions \
  --bootstrap-server localhost:9092

# 2. ISR 变化监控
# JMX 指标
kafka.server:type=ReplicaManager,name=IsrShrinksPerSec
kafka.server:type=ReplicaManager,name=IsrExpandsPerSec
kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions

# 3. 查看 ISR 详情
$ bin/kafka-topics.sh --describe --topic my-topic \
  --bootstrap-server localhost:9092
Topic: my-topic    Partition: 0    Leader: 1    Replicas: 1,2,3    Isr: 1,2
```

**追问：min.insync.replicas=2 但 ISR 只剩1个会怎样？**

**答案：**

```yaml
场景分析：
min.insync.replicas=2
replication.factor=3
当前 ISR = [1]  （只剩 Leader）

情况1：acks=all
- 写入请求失败！
- 错误信息：NotEnoughReplicas
- 因为 ISR 大小(1) < min.insync.replicas(2)

情况2：acks=1
- 写入成功，但违反可靠性保证
- 不推荐这种配置组合

最佳实践：
1. min.insync.replicas ≤ replication.factor
2. 设置 min.insync.replicas=2，至少3副本
3. 监控 Under Replicated Partitions

# 配置示例
min.insync.replicas=2
replication.factor=3
# 即使一个副本宕机，ISR=2，仍可正常工作
```

---

### 2. unclean.leader.election 允许的话有什么问题？

**答案：**

`unclean.leader.election.enable` 是一个关键配置，影响数据丢失风险和可用性的权衡。

#### 配置含义

```yaml
# 默认值
unclean.leader.election.enable=false

# true：允许非 ISR 副本成为 Leader
# false：只允许 ISR 中的副本成为 Leader
```

#### 场景分析

```
场景：ISR 中所有副本都不可用

┌─────────────────────────────────────────────────────────────┐
│  Topic: my-topic, Partition: 0                              │
│  Replicas: [1, 2, 3]                                        │
│  ISR: [1, 2]                                                │
│                                                             │
│  状态：Leader=1, Follower=2, Follower=3(OSR)               │
│        Leader 和 Follower(2) 都宕机                         │
│        只剩 Follower(3)，但它不在 ISR 中                    │
└─────────────────────────────────────────────────────────────┘

如果 unclean.leader.election.enable=false：
  → 分区不可用，等待 ISR 副本恢复
  → 数据不丢失，但服务中断

如果 unclean.leader.election.enable=true：
  → Follower(3) 成为新 Leader
  → 服务恢复，但可能丢失数据
  → Follower(3) 的数据落后于 ISR 副本
```

#### 数据丢失风险

```java
// 数据丢失示例
// 假设：
// Leader(offset=1000), Follower2(offset=990), Follower3(offset=500, OSR)
// Leader 和 Follower2 宕机

// 如果 Follower3 成为 Leader：
// - offset 500-989 的消息对 Consumer 不可见
// - 这些消息可能已经提交给客户端（acks=all）

// 数据不一致：
// Producer 认为消息已提交
// Consumer 可能收不到这些消息
```

#### 生产环境建议

```yaml
# 推荐配置
unclean.leader.election.enable=false

# 理由：
# 1. 消息队列通常更看重数据可靠性
# 2. 短暂不可用比数据丢失更容易接受
# 3. 合理配置副本数和 min.insync.replicas 可避免此情况

# 避免此问题的最佳实践：
min.insync.replicas=2
replication.factor=3
acks=all

# 这样即使一个副本宕机，ISR 仍有 2 个副本
# 除非 ISR 中所有副本同时宕机，否则不会触发 unclean 选举
```

#### 监控和告警

```yaml
# 关键监控指标

# 1. Offline Partitions
# 有 Leader 不可用的分区数
kafka.controller:type=KafkaController,name=OfflinePartitionsCount

# 2. Preferred Replica 不在 ISR
# 理想 Leader 不在 ISR 中
kafka.controller:type=KafkaController,name=PreferredReplicaImbalanceCount

# 告警规则示例
alerts:
  - name: OfflinePartitions
    expr: kafka_controller_offline_partitions_count > 0
    severity: critical
    message: "有分区不可用，请检查 Broker 状态"
```

---

### 3. Kafka 幂等性是怎么实现的？PID 和序号的作用？

**答案：**

Kafka 的幂等性保证单个 Producer 实例内消息不重复，通过 PID（Producer ID）和序号（Sequence Number）实现。

#### 幂等性的实现原理

```
┌─────────────────────────────────────────────────────────────┐
│                    幂等性保证流程                            │
│                                                             │
│  Producer                      Broker                       │
│    │                             │                          │
│    │  1. InitProducerId         │                          │
│    │  ──────────────────────→   │                          │
│    │  ←────── PID=100 ─────────│                          │
│    │                             │                          │
│    │  2. 发送消息 (PID=100, Seq=0)                        │
│    │  ──────────────────────→   │                          │
│    │  ←───── ACK ─────────────  │                          │
│    │                             │                          │
│    │  3. 重试发送 (PID=100, Seq=0)  ← 网络超时            │
│    │  ──────────────────────→   │                          │
│    │  ←── ACK (已存在，不写入) ─│  ← Broker 检测到重复     │
│    │                             │                          │
└─────────────────────────────────────────────────────────────┘
```

#### PID 和 Sequence Number

```java
// Producer 端
public class ProducerState {
    private final long producerId;        // PID: 唯一标识
    private final short epoch;            // Epoch: 防止僵尸实例
    private int sequenceNumber;           // 序号：每条消息递增
}

// 消息发送时的序列号分配
public class RecordAccumulator {
    public RecordAppendResult append(TopicPartition tp, ...) {
        // 每个分区维护独立的序号
        int sequence = sequenceNumbers.getOrDefault(tp, 0);
        batch.setProducerState(pid, epoch, sequence);
        sequenceNumbers.put(tp, sequence + batch.recordCount);
    }
}
```

```java
// Broker 端去重逻辑
public class ProducerStateManager {
    // 每个 (PID, 分区) 维护一个状态
    private Map<Long, ProducerStateEntry> producers;
    
    public boolean isDuplicate(long pid, int sequence, int partition) {
        ProducerStateEntry state = producers.get(pid);
        if (state == null) {
            return false;  // 新 Producer
        }
        
        // 检查序号是否已存在
        if (sequence <= state.lastSequenceNumber) {
            return true;  // 重复消息
        }
        
        // 检查序号是否连续
        if (sequence != state.lastSequenceNumber + 1) {
            throw new OutOfOrderSequenceException();
        }
        
        return false;
    }
}
```

#### 配置启用

```yaml
# 启用幂等性
enable.idempotence=true

# 幂等性启用后，以下配置自动设置：
acks=all
retries=Integer.MAX_VALUE
max.in.flight.requests.per.connection=5  # 可保证有序（<=5）

# 如果手动设置会报错
```

#### 幂等性的限制

```yaml
幂等性保证范围：

✅ 保证：
- 单个 Producer 实例
- 单个 Topic-Partition
- 消息不重复、不丢失、有序

❌ 不保证：
- 跨 Producer 实例（每个 Producer 有独立 PID）
- 跨 Topic-Partition
- 跨会话（Producer 重启后 PID 会变化）

示例：
Producer-A 发送消息到 Partition-0 → 幂等保证
Producer-A 发送消息到 Partition-1 → 各自幂等，不保证跨分区
Producer-B 发送消息 → 与 Producer-A 无关
```

**追问：max.in.flight.requests.per.connection 为什么限制为5？**

**答案：**

```yaml
# 序号连续性要求

场景：max.in.flight.requests.per.connection=5

请求发送顺序：batch1(seq=0), batch2(seq=1), batch3(seq=2)...
如果 batch1 失败，batch2、batch3 成功：
- Broker 期望下一个序号是 0
- 但收到的是 1、2
- 会拒绝这些请求

Kafka 的解决方案（启用幂等性后）：
- Broker 会缓存最近的 5 个批次
- 允许乱序到达（在一定范围内）
- 但最终写入时保证有序

为什么是5？
- 太小：并发度低，吞吐下降
- 太大：Broker 内存占用高
- 5 是经验值，平衡了性能和可靠性
```

---

### 4. Kafka 事务消息是怎么保证 Exactly-Once 的？

**答案：**

Kafka 事务机制保证跨分区的原子性写入，实现 Exactly-Once 语义。

#### 事务消息的应用场景

```yaml
场景1：consume-process-produce
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Topic A │ →  │  处理    │ →  │  Topic B │
│ (消费)   │    │  逻辑    │    │ (生产)   │
└──────────┘    └──────────┘    └──────────┘
要求：消费 A 和生产 B 要么都成功，要么都失败

场景2：跨分区原子写入
┌──────────┐
│ Producer │ → Topic A, Partition 0
│          │ → Topic A, Partition 1
│          │ → Topic B, Partition 0
└──────────┘
要求：所有分区的写入要么都成功，要么都回滚
```

#### 事务实现架构

```
┌─────────────────────────────────────────────────────────────┐
│                      事务架构                                │
│                                                             │
│  Producer                   Transaction Coordinator         │
│    │                              │                        │
│    │  1. InitProducerId          │                        │
│    │  ─────────────────────→     │                        │
│    │  ←─── PID, Epoch ─────────  │                        │
│    │                              │                        │
│    │  2. BeginTxn               │                        │
│    │  ─────────────────────→     │                        │
│    │                              │                        │
│    │  3. 发送消息到多个分区       │                        │
│    │  ─────────────────────→     Broker (多个分区)        │
│    │                              │                        │
│    │  4. SendOffsets (如果是消费)│                        │
│    │  ─────────────────────→     │                        │
│    │                              │                        │
│    │  5. CommitTxn / AbortTxn    │                        │
│    │  ─────────────────────→     │                        │
│    │                              │                        │
│    │  ←─── Transaction Result ───│                        │
│    │                              │                        │
└─────────────────────────────────────────────────────────────┘
```

#### 事务相关配置

```yaml
# Producer 配置
enable.idempotence=true
transactional.id=my-transactional-id  # 必须唯一

# Consumer 配置（消费事务消息）
isolation.level=read_committed  # 只读取已提交的消息
# 默认值：read_uncommitted（读取所有消息）
```

#### 代码示例

```java
// 事务性生产者
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("transactional.id", "my-txn-id");
props.put("enable.idempotence", "true");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);

// 初始化事务
producer.initTransactions();

try {
    // 开始事务
    producer.beginTransaction();
    
    // 发送消息到多个分区
    producer.send(new ProducerRecord<>("topic-a", "key1", "value1"));
    producer.send(new ProducerRecord<>("topic-b", "key2", "value2"));
    
    // 如果是 consume-process-produce 模式
    // 提交消费 offset
    producer.sendOffsetsToTransaction(
        Collections.singletonMap(
            new TopicPartition("input-topic", 0),
            new OffsetAndMetadata(offset)
        ),
        "consumer-group-id"
    );
    
    // 提交事务
    producer.commitTransaction();
    
} catch (Exception e) {
    // 回滚事务
    producer.abortTransaction();
}
```

#### Consumer 端处理

```java
// 消费事务消息
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "my-group");
props.put("isolation.level", "read_committed");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("topic-a", "topic-b"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        // 只会收到已提交事务的消息
        // 控制消息（COMMIT/ABORT）对用户不可见
        System.out.println(record.value());
    }
}
```

**追问：跨分区事务怎么保证原子性？**

**答案：**

```yaml
# 两阶段提交协议

阶段1：Prepare
- Coordinator 向所有参与分区发送 Prepare 请求
- 分区写入事务日志，标记为 "PREPARE"
- 返回 ACK 给 Coordinator

阶段2：Commit/Abort
- Coordinator 收到所有 Prepare ACK
- 向所有分区发送 Commit（或 Abort）请求
- 分区更新事务状态为 "COMMITTED" 或 "ABORTED"
- 返回 ACK

# 恢复机制
如果 Coordinator 故障：
- 新 Coordinator 从 __transaction_state 恢复事务状态
- 检查各分区的 Prepare 状态
- 决定继续 Commit 还是 Abort

# 消费者过滤
Consumer 设置 isolation.level=read_committed：
- 只消费 COMMITTED 状态的消息
- 忽略 ABORTED 事务的消息
- 等待进行中的事务（有超时配置）
```


### 5. Kafka 事务协调器的工作流程？

**答案：**

事务协调器（Transaction Coordinator）是 Kafka 实现事务的核心组件，负责管理事务的生命周期。

#### 事务协调器的角色

```
┌─────────────────────────────────────────────────────────────┐
│                    事务协调器架构                            │
│                                                             │
│         ┌────────────────────────────────────────┐         │
│         │      Transaction Coordinator           │         │
│         │      (Broker 中的特定实例)              │         │
│         │                                        │         │
│         │  ┌──────────────────────────────────┐ │         │
│         │  │    __transaction_state Topic     │ │         │
│         │  │    (事务状态持久化)               │ │         │
│         │  └──────────────────────────────────┘ │         │
│         └────────────────────────────────────────┘         │
│                          │                                 │
│         ┌────────────────┼────────────────┐               │
│         ↓                ↓                ↓               │
│    ┌─────────┐     ┌─────────┐      ┌─────────┐          │
│    │Partition│     │Partition│      │Partition│          │
│    │   A     │     │   B     │      │   C     │          │
│    │(数据分区)│     │(数据分区)│      │(数据分区)│          │
│    └─────────┘     └─────────┘      └─────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Transaction Coordinator 的职责：
1. 分配 PID 和 Epoch
2. 管理事务状态转换
3. 协调两阶段提交
4. 恢复中断的事务
```

#### 事务状态机

```
事务状态转换图：

                    InitProducerId
                         ↓
                ┌────────────────┐
                │     EMPTY      │
                └───────┬────────┘
                        │ BeginTxn
                        ↓
                ┌────────────────┐
                │     ONGOING    │
                └───────┬────────┘
                        │
            ┌───────────┴───────────┐
            │ EndTxn(Commit)        │ EndTxn(Abort)
            ↓                       ↓
    ┌────────────────┐      ┌────────────────┐
    │ PREPARE_COMMIT │      │ PREPARE_ABORT  │
    └───────┬────────┘      └───────┬────────┘
            │                       │
            │ Write Commit Mark     │ Write Abort Mark
            ↓                       ↓
    ┌────────────────┐      ┌────────────────┐
    │ COMPLETE_COMMIT│      │ COMPLETE_ABORT │
    └────────────────┘      └────────────────┘
            │                       │
            └───────────┬───────────┘
                        │ Transaction Complete
                        ↓
                ┌────────────────┐
                │     EMPTY      │
                └────────────────┘
```

#### 事务协调器的工作流程

```java
// 1. 查找事务协调器
// 基于 transactional.id 的哈希值确定 coordinator 所在的 partition
// 该 partition 的 leader 就是 transaction coordinator

// Producer 端
public class TransactionManager {
    public void findCoordinator() {
        // 向集群查询 coordinator
        // FindCoordinatorRequest(key=transactional.id)
    }
}

// 2. InitProducerId - 初始化事务
public class TransactionCoordinator {
    public void handleInitProducerId(String transactionalId) {
        // 检查是否已有该 transactional.id 的 producer
        ProducerState state = loadProducerState(transactionalId);
        
        if (state != null && state.epoch > 0) {
            // 已存在，返回现有 PID，递增 epoch
            // 防止 zombie fencing（僵尸生产者）
            return new InitProducerIdResult(state.pid, state.epoch + 1);
        } else {
            // 新 producer，分配新 PID
            long newPid = generateNewPid();
            return new InitProducerIdResult(newPid, 0);
        }
    }
}

// 3. BeginTxn - 开始事务
public synchronized void beginTransaction() {
    ensureTransactional();
    state.transitionTo(State.IN_TRANSACTION);
}

// 4. EndTxn - 结束事务（两阶段提交）
public void handleEndTxn(long producerId, short epoch, boolean isCommit) {
    // 阶段 1：Prepare
    if (isCommit) {
        state.transitionTo(State.PREPARE_COMMIT);
    } else {
        state.transitionTo(State.PREPARE_ABORT);
    }
    
    // 写入 __transaction_state（内部 topic）
    writeTransactionState(producerId, epoch, state);
    
    // 阶段 2：向所有参与分区发送 marker
    for (TopicPartition partition : partitions) {
        writeTxnMarker(partition, isCommit);
    }
    
    // 完成
    if (isCommit) {
        state.transitionTo(State.COMPLETE_COMMIT);
    } else {
        state.transitionTo(State.COMPLETE_ABORT);
    }
}
```

#### 关键配置

```yaml
# Broker 端配置

# 事务最大超时时间
transaction.max.timeout.ms=900000  # 15分钟，默认值

# 事务日志分区数
# __transaction_state topic 的分区数
transaction.state.log.num.partitions=50

# 事务日志副本数
transaction.state.log.replication.factor=3

# 事务日志最小 ISR
transaction.state.log.min.isr=2
```

**追问：Zombie Fencing 是什么？**

**答案：**

```yaml
Zombie Fencing（僵尸隔离）：

问题场景：
- Producer A 正在写入事务
- Producer A 宕机（或假死）
- Producer A'（新实例，相同 transactional.id）启动
- Producer A 恢复，尝试继续写入

危害：
- 两个 Producer 同时写入，数据不一致

解决方案：Epoch 机制
1. 每个 transactional.id 关联一个 epoch
2. 新 Producer 启动时，epoch + 1
3. Broker 收到旧 epoch 的请求，直接拒绝

# 实现细节
Producer A:  epoch=1  → 正常写入
Producer A 宕机
Producer A': epoch=2  → 新实例，正常写入
Producer A 恢复: epoch=1 的请求 → Broker 拒绝
              Producer FencedException
```

---

### 6. Kafka 如何保证消息不丢失？

**答案：**

Kafka 消息丢失可能发生在多个环节，需要从 Producer、Broker、Consumer 三端综合考虑。

#### 消息丢失的风险点

```
消息生命周期：

Producer          Broker           Consumer
    │               │                  │
    │  1. 发送失败   │                  │
    │  ──────×      │                  │
    │               │                  │
    │  2. 写入未同步  │                  │
    │  ──────→      │  Leader 写入     │
    │               │  未同步到 Follower│
    │               │  Leader 宕机     │
    │               │                  │
    │               │  3. 日志未刷盘    │
    │               │  写入页缓存      │
    │               │  宕机丢失        │
    │               │                  │
    │               │                  │  4. 消费未提交
    │               │  ─────────────→  │  消费完成，未提交
    │               │                  │  重平衡后重复消费
    │               │                  │  （不算丢失，但重复）
```

#### Producer 端保证

```java
// 配置
Properties props = new Properties();

// 1. 确认机制：所有 ISR 确认
props.put("acks", "all");

// 2. 重试机制：无限重试
props.put("retries", Integer.MAX_VALUE);

// 3. 幂等性：防止重复
props.put("enable.idempotence", true);

// 4. 回调确认
producer.send(record, new Callback() {
    @Override
    public void onCompletion(RecordMetadata metadata, Exception exception) {
        if (exception != null) {
            // 处理发送失败
            log.error("发送失败", exception);
            // 可以选择重试或记录到本地
        }
    }
});

// 5. 阻塞发送（确保成功）
RecordMetadata metadata = producer.send(record).get();
```

#### Broker 端保证

```yaml
# 1. 副本配置
replication.factor=3              # 3 副本
min.insync.replicas=2             # 最小 ISR=2
unclean.leader.election.enable=false  # 禁止非 ISR 选举

# 2. 日志刷盘（可选，通常依赖页缓存）
# 高可靠场景可以配置
log.flush.interval.messages=10000
log.flush.interval.ms=1000

# 3. 副本同步
replica.lag.time.max.ms=30000     # ISR 检测间隔
```

#### Consumer 端保证

```java
// 手动提交 offset
Properties props = new Properties();
props.put("enable.auto.commit", "false");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        try {
            // 1. 处理消息
            processMessage(record);
            
            // 2. 处理成功后提交
            consumer.commitSync(Collections.singletonMap(
                new TopicPartition(record.topic(), record.partition()),
                new OffsetAndMetadata(record.offset() + 1)
            ));
            
        } catch (Exception e) {
            // 3. 处理失败，不提交 offset
            // 下次重新消费
            log.error("处理失败", e);
        }
    }
}
```

#### 可靠性配置总结

```yaml
# 生产环境推荐配置

# Producer
acks=all
retries=2147483647
enable.idempotence=true
max.in.flight.requests.per.connection=5
delivery.timeout.ms=120000

# Broker
replication.factor=3
min.insync.replicas=2
unclean.leader.election.enable=false

# Consumer
enable.auto.commit=false
isolation.level=read_committed  # 如果使用事务
```

---

### 7. Kafka 如何保证消息顺序？

**答案：**

Kafka 只保证单个分区内的消息顺序，跨分区不保证顺序。

#### 分区内有序的实现

```
┌─────────────────────────────────────────────────────────────┐
│                  分区内有序                                  │
│                                                             │
│  Topic: my-topic, Partition: 0                              │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                         │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  ← offset 顺序递增      │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                         │
│    ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓                           │
│  Consumer 按顺序读取：0 → 1 → 2 → 3 → ...                   │
│                                                             │
│  保证：                                                      │
│  1. Producer 发送顺序 = Broker 存储顺序                      │
│  2. Broker 存储顺序 = Consumer 读取顺序                      │
└─────────────────────────────────────────────────────────────┘
```

#### 失序场景和解决

```yaml
# 场景1：重试导致失序
Producer 发送：msg1 → msg2 → msg3
msg1 失败，msg2 成功，msg3 成功
msg1 重试成功后：msg2 → msg3 → msg1（失序！）

解决：启用幂等性
enable.idempotence=true
max.in.flight.requests.per.connection=5（或更小）

# 场景2：多线程消费
Consumer A 处理 msg1（慢）
Consumer B 处理 msg2（快）
实际完成顺序：msg2 → msg1（失序！）

解决：单线程消费或顺序锁
```

#### 代码实现顺序消费

```java
// 方法1：单线程顺序消费
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    // 单线程处理，保证顺序
    for (ConsumerRecord<String, String> record : records) {
        processMessage(record);  // 顺序处理
    }
    
    consumer.commitSync();
}
```

```java
// 方法2：多线程 + 分区队列
// 每个 partition 一个队列，保证分区内有序
public class OrderedConsumer {
    
    private final Map<TopicPartition, BlockingQueue<ConsumerRecord>> queues = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newFixedThreadPool(partitionCount);
    
    public void start() {
        // 每个分区一个处理线程
        for (TopicPartition tp : assignment) {
            executor.submit(() -> {
                while (true) {
                    ConsumerRecord record = queues.get(tp).take();
                    processMessage(record);  // 该分区内顺序处理
                }
            });
        }
    }
    
    public void consume() {
        while (true) {
            ConsumerRecords records = consumer.poll(Duration.ofMillis(100));
            
            // 按 partition 分发到不同队列
            for (TopicPartition tp : records.partitions()) {
                queues.get(tp).addAll(records.records(tp));
            }
        }
    }
}
```

#### 跨分区有序方案

```java
// 如果需要全局有序，只能用单分区
// 但会牺牲吞吐量

// 创建单分区 Topic
bin/kafka-topics.sh --create \
  --topic ordered-topic \
  --partitions 1 \
  --replication-factor 3 \
  --bootstrap-server localhost:9092

// 或者使用消息中的时间戳/序号，在消费端排序
// 适用于允许一定延迟的场景
```

---

## 三、Rebalance 深度篇

### 1. Rebalance 的触发条件有哪些？

**答案：**

Rebalance 是 Consumer Group 成员变化时的重新分区分配过程，了解触发条件有助于减少不必要的 Rebalance。

#### Rebalance 触发条件

```
┌─────────────────────────────────────────────────────────────┐
│                    Rebalance 触发条件                        │
│                                                             │
│  1. Consumer 加入/离开                                       │
│     ┌─────────┐                                             │
│     │Consumer│ → 新 Consumer 加入 Group                     │
│     └─────────┘                                             │
│     ┌─────────┐                                             │
│     │Consumer│ → Consumer 主动离开（close/unsubscribe）      │
│     └─────────┘                                             │
│                                                             │
│  2. Consumer 宕机/失联                                       │
│     ┌─────────┐                                             │
│     │Consumer│ → session.timeout.ms 超时                    │
│     └─────────┘   被 Coordinator 认为宕机                    │
│                                                             │
│  3. Topic 分区数变化                                         │
│     ┌─────────┐                                             │
│     │Partition│ → 分区数增加/减少                            │
│     │  +N     │   触发重新分配                               │
│     └─────────┘                                             │
│                                                             │
│  4. Topic 订阅变化                                           │
│     ┌─────────┐                                             │
│     │Consumer│ → 订阅新的 Topic                              │
│     │subscrib│ → 取消订阅 Topic                             │
│     └─────────┘                                             │
│                                                             │
│  5. Group 协调器变化                                         │
│     ┌─────────┐                                             │
│     │Coordinator│ → Broker 宕机，Coordinator 迁移            │
│     └─────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 最常见的 Rebalance 原因：Consumer 失联

```yaml
# Consumer 失联配置

session.timeout.ms=10000    # 默认 10 秒
# Coordinator 超过此时间没收到心跳，认为 Consumer 失联

heartbeat.interval.ms=3000  # 默认 3 秒
# Consumer 发送心跳的间隔

max.poll.interval.ms=300000 # 默认 5 分钟
# 两次 poll 的最大间隔

# 失联判定流程
1. Consumer 每隔 heartbeat.interval.ms 发送心跳
2. 如果超过 session.timeout.ms 没收到心跳
3. Coordinator 标记 Consumer 为 dead
4. 触发 Rebalance
```

#### 避免不必要的 Rebalance

```java
// 常见问题：处理时间过长
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        // 如果处理时间超过 max.poll.interval.ms
        // 会被认为宕机，触发 Rebalance
        processMessage(record);  // ⚠️ 可能很慢
    }
}

// 解决方案1：减少每次 poll 的消息数
max.poll.records=100

// 解决方案2：增加 max.poll.interval.ms
max.poll.interval.ms=600000  // 10 分钟

// 解决方案3：异步处理 + 手动管理 offset
```

#### Rebalance 监控

```bash
# JMX 监控指标
kafka.consumer:type=consumer-coordinator-metrics,client-id=*
  - rebalance-rate-per-hour
  - rebalance-latency-avg
  - rebalance-latency-max

# 日志监控
# 查找 Rebalance 相关日志
grep "Revoking previously assigned partitions" /var/log/kafka/consumer.log
grep "Joined group" /var/log/kafka/consumer.log
```


### 2. Eager 和 Cooperative 协议的区别？

**答案：**

Kafka Consumer 支持两种 Rebalance 协议：Eager（急切）和 Cooperative（协作），它们在分区分配方式上有本质区别。

#### Eager 协议（默认，旧版）

```
Eager Rebalance 流程：

阶段1：所有 Consumer 放弃分区
Consumer A: [P0, P1] → []  放弃所有分区
Consumer B: [P2, P3] → []  放弃所有分区
Consumer C: []           新加入

阶段2：重新分配
Consumer A: [P0]
Consumer B: [P2]
Consumer C: [P1, P3]

问题：停止消费 → 重新分配 → 恢复消费
     消费暂停时间较长！
```

```yaml
# Eager 协议特点
优点：
  - 实现简单
  - 状态清理彻底，无冲突

缺点：
  - "Stop-the-world"：所有 Consumer 停止消费
  - 消费暂停时间长
  - 频繁 Rebalance 影响大
```

#### Cooperative 协议（增量 Rebalance）

```
Cooperative Rebalance 流程：

初始状态：
Consumer A: [P0, P1]
Consumer B: [P2, P3]

Consumer C 加入：
阶段1：只调整需要变化的分区
Consumer A: [P0, P1] → [P0]  只放弃 P1
Consumer B: [P2, P3] → [P2]  只放弃 P3
Consumer C: [P1, P3]         接收放弃的分区

特点：
  - Consumer A 继续消费 P0
  - Consumer B 继续消费 P2
  - 只有部分分区需要迁移
  - 消费暂停时间短
```

```yaml
# Cooperative 协议特点
优点：
  - 渐进式：不需要全部停止
  - 消费暂停时间短
  - 适合大规模 Consumer Group

缺点：
  - 可能需要多轮 Rebalance 才能稳定
  - 实现复杂
```

#### 配置使用

```java
// 使用 Cooperative 协议
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "my-group");

// 设置分区分配策略（Kafka 2.4+）
// RangeAssignor + CooperativeStickyAssignor
props.put("partition.assignment.strategy", 
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");

// 或者使用混合策略
props.put("partition.assignment.strategy",
    "org.apache.kafka.clients.consumer.RangeAssignor," +
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
```

#### 协议对比

| 特性 | Eager | Cooperative |
|-----|-------|-------------|
| 消费暂停 | 全部暂停 | 部分暂停 |
| Rebalance 轮数 | 1 轮 | 可能多轮 |
| 分区迁移 | 全部重新分配 | 增量迁移 |
| 适用场景 | 小规模 Consumer | 大规模 Consumer |
| Kafka 版本 | 所有版本 | 2.4+ |

---

### 3. 怎么避免 Rebalance 导致的消费暂停？

**答案：**

Rebalance 会导致消费暂停，影响系统可用性，以下是减少 Rebalance 影响的策略。

#### 问题分析

```
Rebalance 导致的消费暂停：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ Consumer 1  ──────┐                                         │
│ Consumer 2  ──────┼──── 正常消费 ────┼──── Rebalance ────→  │
│ Consumer 3  ──────┘                 │                      │
│                                     ↓                      │
│                          ┌───────────────────────┐        │
│                          │    Stop the World     │        │
│                          │    所有 Consumer 暂停  │        │
│                          │    等待重新分配        │        │
│                          │    (通常几秒到几十秒)  │        │
│                          └───────────────────────┘        │
│                                     │                      │
│                                     ↓                      │
│                          ─────── 恢复消费 ───────→         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 策略1：合理配置心跳和超时

```yaml
# 避免因处理慢触发 Rebalance

# 心跳配置
heartbeat.interval.ms=3000     # 心跳间隔
session.timeout.ms=30000       # 会话超时（建议 >= heartbeat * 10）

# Poll 间隔
max.poll.interval.ms=600000    # 两次 poll 最大间隔（根据处理时间调整）
max.poll.records=100           # 每次 poll 的消息数（减少单次处理时间）

# 原则：
# session.timeout.ms > 处理一批消息的时间
# max.poll.interval.ms > 处理所有拉取消息的时间
```

#### 策略2：使用 Cooperative 协议

```java
Properties props = new Properties();
// 使用增量 Rebalance
props.put("partition.assignment.strategy", 
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
```

#### 策略3：静态成员（Static Membership）

```java
// Kafka 2.3+ 特性
Properties props = new Properties();
props.put("group.instance.id", "consumer-instance-1");  // 静态成员 ID

// 配置
session.timeout.ms=300000       # 静态成员可以设置更长的超时

// 效果：
// Consumer 短暂离线（重启、升级）后重新加入
// 不会触发 Rebalance，直接恢复之前的分区分配
```

#### 策略4：优雅关闭

```java
// 主动离开前，优雅关闭
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    // 1. 停止拉取新消息
    // 2. 处理完当前消息
    // 3. 提交 offset
    consumer.commitSync();
    // 4. 关闭消费者
    consumer.close(Duration.ofSeconds(30));
}));
```

#### 策略5：消费者健康检查

```java
public class HealthyConsumer {
    
    private volatile boolean healthy = true;
    
    public void start() {
        // 健康检查线程
        Thread healthCheck = new Thread(() -> {
            while (healthy) {
                try {
                    // 检查外部依赖（数据库、网络等）
                    checkDependencies();
                    Thread.sleep(5000);
                } catch (Exception e) {
                    healthy = false;
                    // 主动离开，避免 Rebalance 等待超时
                    consumer.close();
                }
            }
        });
        healthCheck.start();
        
        // 消费循环
        while (healthy) {
            ConsumerRecords records = consumer.poll(Duration.ofMillis(100));
            process(records);
        }
    }
}
```

---

### 4. 静态成员（Static Member）是什么？

**答案：**

静态成员（Static Membership）是 Kafka 2.3 引入的特性，允许 Consumer 在短暂离线后重新加入时保持原有分区分配，避免触发 Rebalance。

#### 工作原理

```
传统 Consumer Group：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ Consumer A (id: dynamic) ──┐                                │
│ Consumer B (id: dynamic) ──┼── Group ──→ 任意 Consumer 离开 │
│ Consumer C (id: dynamic) ──┘     │     触发 Rebalance      │
│                                  ↓                          │
│                          所有 Consumer 重新分配              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

静态成员 Consumer Group：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ Consumer A (group.instance.id=static-A) ──┐                 │
│ Consumer B (group.instance.id=static-B) ──┼── Group         │
│ Consumer C (group.instance.id=static-C) ──┘                 │
│                                                             │
│ Consumer B 短暂离线（重启/升级）                             │
│     ↓                                                       │
│ Coordinator 等待 Consumer B 重新加入                        │
│     ↓                                                       │
│ Consumer B 重新加入，恢复原有分区分配                        │
│     ↓                                                       │
│ 不触发 Rebalance！Consumer A/C 继续消费                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 配置使用

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "my-group");

// 设置静态成员 ID
props.put("group.instance.id", "consumer-static-1");

// 可以设置更长的 session timeout
props.put("session.timeout.ms", "300000");  // 5 分钟

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
```

#### 适用场景

```yaml
场景1：应用升级/重启
  - 传统方式：重启每个 Consumer 都触发 Rebalance
  - 静态成员：重启后直接恢复，不触发 Rebalance

场景2：临时故障恢复
  - Consumer 短暂不可用（GC、网络抖动）
  - 恢复后直接继续消费

场景3：维护窗口
  - 计划内的停机维护
  - 恢复后无需 Rebalance
```

#### 注意事项

```yaml
# 1. group.instance.id 必须唯一
# 同一个 Group 内，不同 Consumer 不能使用相同的 static id

# 2. session.timeout.ms 可以设置更长
# 但也不能太长，否则真正宕机的 Consumer 会阻塞分区

# 3. 最大静态成员数
max.group.instance.id=100000  # Broker 端配置

# 4. 不适用于频繁上下线的场景
# 如果 Consumer 频繁离开，静态成员反而会占用分区
```

---

### 5. Rebalance 时消息怎么处理？

**答案：**

Rebalance 过程中，Consumer 会停止消费，需要正确处理未完成的消息和 offset。

#### Rebalance 的消息处理流程

```
Rebalance 详细流程：

┌─────────────────────────────────────────────────────────────┐
│  1. Consumer 收到 Rebalance 通知                            │
│     - onPartitionsRevoked() 回调被调用                      │
│     - 停止处理新消息                                        │
│                                                             │
│  2. 处理进行中的消息                                         │
│     - 完成当前处理中的消息                                   │
│     - 或者回滚，等待下次重新处理                             │
│                                                             │
│  3. 提交 offset                                             │
│     - 同步提交当前 offset                                   │
│     - 确保消息不会丢失                                       │
│                                                             │
│  4. 释放资源                                                │
│     - 关闭文件句柄、数据库连接等                            │
│     - 清理状态                                              │
│                                                             │
│  5. 等待重新分配                                             │
│     - Consumer 等待 Coordinator 分配新分区                  │
│                                                             │
│  6. 接收新分区                                               │
│     - onPartitionsAssigned() 回调被调用                     │
│     - 初始化新分区的状态                                    │
│                                                             │
│  7. 恢复消费                                                │
│     - 从上次提交的 offset 开始消费                          │
└─────────────────────────────────────────────────────────────┘
```

#### 代码实现

```java
public class RebalanceAwareConsumer implements ConsumerRebalanceListener {
    
    private final KafkaConsumer<String, String> consumer;
    private final Map<TopicPartition, OffsetAndMetadata> currentOffsets = new ConcurrentHashMap<>();
    
    public RebalanceAwareConsumer(KafkaConsumer<String, String> consumer) {
        this.consumer = consumer;
    }
    
    @Override
    public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
        System.out.println("Partitions revoked: " + partitions);
        
        // 1. 提交当前处理的 offset
        consumer.commitSync(currentOffsets);
        
        // 2. 清理资源
        for (TopicPartition partition : partitions) {
            cleanupPartitionResources(partition);
            currentOffsets.remove(partition);
        }
    }
    
    @Override
    public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
        System.out.println("Partitions assigned: " + partitions);
        
        // 初始化新分区
        for (TopicPartition partition : partitions) {
            // 从上次提交的 offset 开始
            long lastOffset = getStoredOffset(partition);
            if (lastOffset >= 0) {
                consumer.seek(partition, lastOffset);
            }
        }
    }
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                processMessage(record);
                
                // 记录当前 offset
                currentOffsets.put(
                    new TopicPartition(record.topic(), record.partition()),
                    new OffsetAndMetadata(record.offset() + 1)
                );
            }
            
            // 异步提交
            consumer.commitAsync(currentOffsets, null);
        }
    }
}
```

#### 消息重复/丢失风险

```yaml
风险1：Rebalance 前未提交 offset
- 消息已处理但未提交
- Rebalance 后从头消费 → 重复消费
- 解决：onPartitionsRevoked 中同步提交

风险2：处理过程中 Rebalance
- 消息处理一半，Rebalance 发生
- 可能导致数据不一致
- 解决：使用事务或幂等处理

风险3：异步提交未完成
- commitAsync 还没完成就 Rebalance
- 可能丢失 offset 提交
- 解决：Rebalance 前使用 commitSync
```

#### 最佳实践

```java
// 完整的 Rebalance 处理模式
public class SafeConsumer {
    
    private final AtomicBoolean rebalanceInProgress = new AtomicBoolean(false);
    
    public void consume() {
        while (true) {
            if (rebalanceInProgress.get()) {
                // Rebalance 中，等待
                Thread.sleep(100);
                continue;
            }
            
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                // 幂等处理
                processIdempotent(record);
            }
        }
    }
    
    @Override
    public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
        rebalanceInProgress.set(true);
        
        try {
            // 等待进行中的处理完成
            waitForPendingProcessing();
            
            // 同步提交
            consumer.commitSync();
            
        } finally {
            rebalanceInProgress.set(false);
        }
    }
}
```

---

### 6. Rebalance 监控怎么做？

**答案：**

Rebalance 监控对于保障系统稳定性至关重要，可以从多个维度进行监控告警。

#### 关键监控指标

```yaml
# Consumer 端 JMX 指标

# 1. Rebalance 相关
kafka.consumer:type=consumer-coordinator-metrics,client-id=*
  - rebalance-rate-per-hour       # 每小时 Rebalance 次数
  - rebalance-total               # 总 Rebalance 次数
  - rebalance-latency-avg         # Rebalance 平均延迟
  - rebalance-latency-max         # Rebalance 最大延迟

# 2. 心跳相关
kafka.consumer:type=consumer-coordinator-metrics,client-id=*
  - heartbeat-rate                # 心跳速率
  - heartbeat-response-time-max   # 心跳响应时间

# 3. 分区分配
kafka.consumer:type=consumer-coordinator-metrics,client-id=*
  - assigned-partitions           # 分配的分区数
  - partition-assigned-latency-avg # 分区分配延迟

# 4. 消费延迟
kafka.consumer:type=consumer-fetch-manager-metrics,client-id=*
  - records-lag-max               # 最大消息积压
  - records-consumed-rate         # 消费速率
```

#### 监控配置示例

```java
// 使用 Micrometer 监控
public class ConsumerMetrics {
    
    private final MeterRegistry registry;
    private final Counter rebalanceCounter;
    private final Timer rebalanceTimer;
    
    public ConsumerMetrics(MeterRegistry registry) {
        this.registry = registry;
        this.rebalanceCounter = Counter.builder("kafka.consumer.rebalance")
            .description("Kafka consumer rebalance count")
            .register(registry);
        this.rebalanceTimer = Timer.builder("kafka.consumer.rebalance.latency")
            .description("Kafka consumer rebalance latency")
            .register(registry);
    }
    
    public void recordRebalance(long durationMs) {
        rebalanceCounter.increment();
        rebalanceTimer.record(durationMs, TimeUnit.MILLISECONDS);
    }
}
```

#### Prometheus 告警规则

```yaml
# prometheus/alerts.yml

groups:
  - name: kafka-consumer
    rules:
      # Rebalance 频率告警
      - alert: KafkaConsumerRebalanceTooFrequent
        expr: |
          rate(kafka_consumer_coordinator_rebalance_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kafka Consumer Rebalance 过于频繁"
          description: "Consumer {{ $labels.client_id }} 在过去5分钟内 Rebalance 频率过高"

      # Rebalance 延迟告警
      - alert: KafkaConsumerRebalanceSlow
        expr: |
          kafka_consumer_coordinator_rebalance_latency_avg > 30000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kafka Consumer Rebalance 耗时过长"
          description: "Consumer {{ $labels.client_id }} Rebalance 平均延迟超过 30 秒"

      # 消费延迟告警
      - alert: KafkaConsumerLagHigh
        expr: |
          kafka_consumer_fetch_manager_records_lag_max > 100000
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Kafka Consumer 消费延迟过高"
          description: "Consumer {{ $labels.client_id }} 消费延迟超过 10 万条"
```

#### 日志分析

```bash
# 分析 Rebalance 日志

# 1. 统计 Rebalance 次数
grep "Joined group" /var/log/kafka/consumer.log | wc -l

# 2. 查找 Rebalance 原因
grep "Rebalance" /var/log/kafka/consumer.log | tail -100

# 3. 分析 Consumer 离开原因
grep "LeaveGroup\|Member leaving" /var/log/kafka/server.log

# 4. 查看 Coordinator 处理 Rebalance 的日志
grep "Preparing for rebalance" /var/log/kafka/server.log
```

#### 监控大屏展示

```
┌─────────────────────────────────────────────────────────────┐
│                   Kafka Consumer 监控大屏                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Consumer Group 状态          Rebalance 统计                │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │ Total Groups: 50     │    │ Last Hour: 12        │      │
│  │ Stable: 45           │    │ Last Day: 89         │      │
│  │ Rebalancing: 3       │    │ Avg Latency: 2.3s    │      │
│  │ Empty: 2             │    │ Max Latency: 15s     │      │
│  └──────────────────────┘    └──────────────────────┘      │
│                                                             │
│  消费延迟 Top 10              告警列表                       │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │ order-consumer: 50K  │    │ ⚠️ group-A 频繁      │      │
│  │ log-consumer: 30K    │    │ ⚠️ group-B 延迟高    │      │
│  │ event-consumer: 20K  │    │ 🔴 group-C 消费停止  │      │
│  └──────────────────────┘    └──────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


---

## 四、存储与日志篇

### 1. 日志段（Log Segment）是怎么组织的？

**答案：**

Kafka 的消息存储以日志段（Log Segment）为单位组织，每个 Segment 包含日志文件和索引文件。

#### 日志段结构

```
Topic-Partition 目录结构：

/data/kafka-logs/my-topic-0/
├── 00000000000000000000.log      # 日志文件（消息数据）
├── 00000000000000000000.index    # offset 索引
├── 00000000000000000000.timeindex # 时间索引
├── 00000000000000000000.txnindex # 事务索引（如果有事务消息）
│
├── 00000000000000567890.log      # 下一个 Segment
├── 00000000000000567890.index
├── 00000000000000567890.timeindex
│
└── leader-epoch-checkpoint       # Leader Epoch 检查点

文件命名规则：
- 以该 Segment 的起始 offset 命名
- 例如：00000000000000567890 表示起始 offset = 567890
```

#### 日志文件格式

```
.log 文件结构（每条消息）：

┌─────────────────────────────────────────────────────────────┐
│  Offset (8 bytes)                                           │
│  Size (4 bytes)                                             │
│  CRC (4 bytes)                                              │
│  Magic (1 byte) - 版本号                                    │
│  Attributes (1 byte) - 压缩类型等                           │
│  Timestamp (8 bytes)                                        │
│  Key Length (4 bytes)                                       │
│  Key (variable)                                             │
│  Value Length (4 bytes)                                     │
│  Value (variable)                                           │
│  Headers (variable)                                         │
└─────────────────────────────────────────────────────────────┘

# 批量消息（Message V2）
┌─────────────────────────────────────────────────────────────┐
│  Record Batch                                               │
│  ├── Base Offset (8 bytes)                                  │
│  ├── Batch Length (4 bytes)                                 │
│  ├── Partition Leader Epoch (4 bytes)                       │
│  ├── Magic (1 byte) - 2                                     │
│  ├── CRC (4 bytes)                                          │
│  ├── Attributes (2 bytes)                                   │
│  ├── Last Offset Delta (4 bytes)                            │
│  ├── Base Timestamp (8 bytes)                               │
│  ├── Max Timestamp (8 bytes)                                │
│  ├── Producer ID (8 bytes)                                  │
│  ├── Producer Epoch (2 bytes)                               │
│  ├── Base Sequence (4 bytes)                                │
│  ├── Records Count (4 bytes)                                │
│  └── Records [...]                                          │
└─────────────────────────────────────────────────────────────┘
```

#### Segment 滚动策略

```yaml
# Broker 配置

# 1. 按 size 滚动
log.segment.bytes=1073741824  # 1GB（默认）

# 2. 按时间滚动
log.roll.hours=168            # 7 天（默认）
# 或
log.roll.ms=604800000

# 3. 滚动检查周期
log.roll.jitter.hours=0       # 滚动时间抖动

# 实际滚动条件
Segment 创建后，满足以下任一条件即滚动：
- 文件大小达到 log.segment.bytes
- 时间达到 log.roll.hours
```

#### 日志段管理

```java
// LogSegment 核心方法
public class LogSegment {
    
    private final FileRecords log;           // 日志文件
    private final OffsetIndex offsetIndex;   // offset 索引
    private final TimeIndex timeIndex;       // 时间索引
    
    // 追加消息
    public LogAppendInfo append(long largestOffset, 
                                long largestTimestamp,
                                byte[] records) throws IOException {
        
        // 1. 写入日志文件
        long physicalPosition = log.sizeInBytes();
        log.append(records);
        
        // 2. 更新索引
        // 稀疏索引：每隔一定字节数建立索引项
        if (physicalPosition - lastIndexPosition >= indexIntervalBytes) {
            offsetIndex.append(largestOffset, physicalPosition);
            timeIndex.append(largestTimestamp, physicalPosition);
        }
        
        return new LogAppendInfo(largestOffset, largestTimestamp);
    }
    
    // 读取消息
    public FetchDataInfo read(long startOffset, int maxSize) {
        // 1. 通过索引找到物理位置
        long physicalPosition = offsetIndex.lookup(startOffset);
        
        // 2. 从日志文件读取
        return log.read(physicalPosition, maxSize);
    }
}
```

---

### 2. 索引文件有哪些？稀疏索引怎么工作？

**答案：**

Kafka 使用多种索引文件加速消息查找，采用稀疏索引设计以减少内存占用。

#### 索引类型

```
索引文件类型：

1. Offset Index (.index)
   - offset → 物理位置
   - 用于根据 offset 快速定位消息

2. Time Index (.timeindex)
   - 时间戳 → 物理位置
   - 用于根据时间查找消息

3. Transaction Index (.txnindex)
   - 事务消息专用
   - 记录事务状态

稀疏索引示意图：

.log 文件（消息）：
┌─────────────────────────────────────────────────────────────┐
│ msg0 │ msg1 │ msg2 │ msg3 │ msg4 │ msg5 │ msg6 │ msg7 │ ...│
│ 0    │ 1    │ 2    │ 3    │ 4    │ 5    │ 6    │ 7    │    │
└─────────────────────────────────────────────────────────────┘
   │                             │
   ↓                             ↓
.index 文件（稀疏索引）：
┌───────────────┐
│ offset:0      │
│ position:0    │
├───────────────┤
│ offset:3      │    ← 每 4KB 建一个索引项
│ position:xxxx │
├───────────────┤
│ offset:7      │
│ position:xxxx │
└───────────────┘
```

#### Offset Index 工作原理

```java
// OffsetIndex 实现
public class OffsetIndex {
    
    // 索引项大小：8 bytes offset + 4 bytes position = 12 bytes
    private static final int ENTRY_SIZE = 12;
    
    // MappedByteBuffer 实现零拷贝
    private final MappedByteBuffer mmap;
    
    // 查找方法：二分查找
    public OffsetPosition lookup(long targetOffset) {
        // 1. 二分查找找到 <= targetOffset 的最大索引项
        int slot = binarySearch(targetOffset);
        
        if (slot < 0) {
            return null;
        }
        
        // 2. 返回该索引项的 offset 和物理位置
        return new OffsetPosition(
            getOffset(slot),
            getPosition(slot)
        );
    }
    
    // 添加索引项
    public void append(long offset, int position) {
        // 检查是否满足索引间隔
        if (entries > 0 && offset - lastOffset < indexInterval) {
            return;  // 不添加，太密集
        }
        
        // 写入 mmap
        mmap.putInt((int) offset);
        mmap.putInt(position);
        entries++;
    }
}
```

#### 查找流程

```
根据 offset 查找消息：

步骤1：二分查找 .index 文件
       找到 <= targetOffset 的最大索引项
       
步骤2：从索引项的物理位置开始
       在 .log 文件中顺序扫描
       
步骤3：找到目标 offset 的消息

示例：查找 offset = 5 的消息

.index 文件：
  offset:3 → position:1024
  offset:7 → position:2048
  
查找流程：
  1. 二分查找找到 offset:3（<=5 的最大值）
  2. 从 position:1024 开始读取 .log
  3. 顺序扫描找到 offset:5 的消息
  4. 返回消息内容

时间复杂度：
  - 索引查找：O(log N)
  - 顺序扫描：O(M)，M = 索引间隔内的消息数
  - 总体：接近 O(log N)
```

#### 索引配置

```yaml
# 索引相关配置

# 索引间隔（字节数）
log.index.interval.bytes=4096  # 默认 4KB
# 每 4KB 消息数据建一个索引项

# 索引文件大小
log.index.size.max.bytes=10485760  # 默认 10MB

# 索引预热
log.index.granularity=4096

# 时间索引间隔
log.roll.hours=168
```

---

### 3. 日志清理（Log Compaction）原理？

**答案：**

Log Compaction 是 Kafka 的一种日志清理策略，保留每个 key 的最新值，适合实现变更日志。

#### Log Compaction vs Delete

```
日志清理策略对比：

1. Delete（默认）
   - 基于时间或大小删除旧日志
   - 适用于普通消息队列

   时间线：
   ┌──────────────────────────────────────────────────────────┐
   │ old │ old │ old │ recent │ recent │ recent │ new        │
   └──────────────────────────────────────────────────────────┘
                                      ↑
                               保留时间阈值
                               左边的全部删除

2. Compaction
   - 保留每个 key 的最新值
   - 适用于状态存储、变更日志

   清理前：
   ┌──────────────────────────────────────────────────────────┐
   │ K1:V1 │ K2:V1 │ K1:V2 │ K3:V1 │ K2:V2 │ K1:V3 │ K4:V1  │
   └──────────────────────────────────────────────────────────┘
   
   清理后：
   ┌──────────────────────────────────────────────────────────┐
   │ K3:V1 │ K2:V2 │ K1:V3 │ K4:V1 │                         │
   └──────────────────────────────────────────────────────────┘
           每个 key 只保留最新值
```

#### Compaction 工作原理

```
Compaction 流程：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  原始日志段：                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Segment 1: [K1:V1, K2:V1, K1:V2]                    │   │
│  │ Segment 2: [K3:V1, K2:V2, K1:V3]                    │   │
│  │ Segment 3: [K4:V1, K5:V1]                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  Compaction 过程：                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 标记要清理的 Segment                              │   │
│  │ 2. 构建 key → 最新 offset 的 HashMap                │   │
│  │ 3. 复制保留的消息到新文件                            │   │
│  │ 4. 原子替换旧文件                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  清理后：                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Segment 1: [K2:V1]      (其他已被压缩)              │   │
│  │ Segment 2: [K3:V1, K2:V2, K1:V3]                    │   │
│  │ Segment 3: [K4:V1, K5:V1]                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 配置启用

```yaml
# Topic 级别配置
cleanup.policy=compact  # 或 compact,delete（同时启用）

# Broker 默认配置
log.cleanup.policy=delete  # 默认删除策略

# Compaction 相关参数
log.cleaner.enable=true
log.cleaner.threads=1
log.cleaner.io.max.bytes.per.second=Double.MaxValue
log.cleaner.dedupe.buffer.size=134217728  # 128MB 去重缓冲区
log.cleaner.io.buffer.size=524288         # 512KB IO 缓冲区

# 清理脏数据比例阈值
log.cleaner.min.cleanable.ratio=0.5
# 脏数据比例超过 50% 才会触发清理

# 最小压缩时间间隔
log.cleaner.min.compaction.lag.ms=0
```

#### 应用场景

```java
// 场景：使用 Kafka 实现 KV 存储

// Producer：写入更新
producer.send(new ProducerRecord<>("state-topic", "user-1", 
    "{\"name\":\"Alice\",\"age\":25}"));
    
producer.send(new ProducerRecord<>("state-topic", "user-1", 
    "{\"name\":\"Alice\",\"age\":26}"));  // 更新

// Consumer：重建状态
Properties props = new Properties();
props.put("auto.offset.reset", "earliest");
// 从头读取，只保留每个 key 的最新值

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("state-topic"));

Map<String, String> state = new HashMap<>();
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        state.put(record.key(), record.value());  // 覆盖旧值
    }
    // 最终 state 中保存的是每个 key 的最新值
}
```

---

### 4. 日志删除策略？

**答案：**

Kafka 提供多种日志删除策略，基于时间和大小自动清理过期数据。

#### 删除策略类型

```yaml
# 日志删除策略配置
cleanup.policy=delete  # 或 compact, 或两者

# 1. 基于时间删除
log.retention.hours=168    # 默认 7 天
log.retention.minutes=...
log.retention.ms=...

# 2. 基于大小删除
log.retention.bytes=1073741824  # 默认 -1（无限制）

# 3. 基于日志段删除
log.segment.delete.delay.ms=60000  # 删除延迟
```

#### 删除流程

```
日志删除流程：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  日志管理器定时检查：                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  log.retention.check.interval.ms = 5min             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  检查每个日志段：                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Segment 1: 创建时间 8 天前 → 超过 7 天 → 删除       │   │
│  │ Segment 2: 创建时间 5 天前 → 保留                   │   │
│  │ Segment 3: 创建时间 1 天前 → 保留                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│  删除符合条件的日志段：                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 关闭文件句柄                                      │   │
│  │ 2. 删除 .log、.index、.timeindex 文件               │   │
│  │ 3. 更新元数据                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 基于大小的删除

```yaml
# 当日志总大小超过阈值，删除最旧的 Segment

log.retention.bytes=10737418240  # 10GB

# 工作流程：
# 1. 检查 Topic-Partition 的总大小
# 2. 如果超过 log.retention.bytes
# 3. 从最旧的 Segment 开始删除
# 4. 直到总大小 < log.retention.bytes

# 注意：这是每个 Partition 的限制
# 总磁盘占用 = log.retention.bytes × 分区数 × 副本数
```

#### 配置优先级

```yaml
# 配置优先级（从高到低）
1. Topic 级别配置
   bin/kafka-configs.sh --alter --topic my-topic \
     --add-config retention.ms=86400000

2. Broker 默认配置
   log.retention.hours=168

# 时间单位优先级
retention.ms > retention.minutes > retention.hours
```

---

### 5. 长时间运行后磁盘空间问题？

**答案：**

Kafka 长时间运行后可能遇到磁盘空间问题，需要从预防和监控两方面处理。

#### 常见问题场景

```yaml
场景1：分区数过多，总量超预期
  - 100 个 Topic × 10 分区 × 3 副本 = 3000 个 Partition
  - 每个 Partition 10GB 保留 = 30TB
  - 磁盘规划不足

场景2：retention 配置不当
  - 默认 7 天，某些 Topic 数据量暴增
  - 7 天数据超过磁盘容量

场景3：Compaction 失效
  - compact 策略的 Topic，清理线程卡住
  - 数据只增不减

场景4：生产者速度 > 消费者速度
  - 消息积压，磁盘持续增长
```

#### 预防措施

```yaml
# 1. 合理设置保留策略
# 按数据重要性分级

# 高频日志（可丢失）
retention.hours=1
retention.bytes=1073741824  # 1GB

# 普通业务（重要）
retention.hours=72
retention.bytes=10737418240  # 10GB

# 核心数据（最重要）
retention.hours=168
retention.bytes=-1  # 无大小限制，但监控告警

# 2. 磁盘容量规划
# 容量 = 保留数据量 × 分区数 × 副本数 × 冗余系数(1.2)

# 3. 监控告警
# 磁盘使用率 > 70% 预警
# 磁盘使用率 > 85% 告警
# 磁盘使用率 > 95% 紧急
```

#### 磁盘空间监控

```bash
# 1. 查看磁盘使用
df -h /data/kafka-logs

# 2. 查看 Topic 大小
bin/kafka-log-dirs.sh --describe --bootstrap-server localhost:9092 \
  --topic-list my-topic

# 3. 查看各 Partition 大小
du -sh /data/kafka-logs/*

# 4. JMX 监控
kafka.log:type=Log,name=Size,topic=*,partition=*
kafka.log:type=Log,name=NumLogSegments,topic=*,partition=*
```

#### 紧急处理

```bash
# 场景1：临时调整保留时间
bin/kafka-configs.sh --alter --topic big-topic \
  --add-config retention.ms=3600000 \
  --bootstrap-server localhost:9092
# 保留时间改为 1 小时，快速释放空间

# 场景2：手动删除旧日志段
# 找到最旧的 Segment 文件
ls -lt /data/kafka-logs/my-topic-0/ | tail -n 10
# 手动删除（不推荐，可能导致数据不一致）

# 场景3：迁移数据到新磁盘
# 1. 停止 Broker
# 2. 复制数据到新磁盘
# 3. 修改 log.dirs 配置
# 4. 启动 Broker

# 场景4：删除非必要 Topic
bin/kafka-topics.sh --delete --topic unused-topic \
  --bootstrap-server localhost:9092
```

#### 自动化运维脚本

```bash
#!/bin/bash
# disk_monitor.sh - 磁盘空间监控脚本

THRESHOLD_WARNING=70
THRESHOLD_CRITICAL=85

# 获取磁盘使用率
USAGE=$(df /data/kafka-logs | tail -1 | awk '{print $5}' | tr -d '%')

if [ $USAGE -gt $THRESHOLD_CRITICAL ]; then
    echo "CRITICAL: Disk usage is ${USAGE}%"
    # 发送告警
    # 自动调整最大 Topic 的保留时间
    /opt/kafka/bin/kafka-configs.sh --alter --topic largest-topic \
        --add-config retention.ms=3600000 \
        --bootstrap-server localhost:9092
elif [ $USAGE -gt $THRESHOLD_WARNING ]; then
    echo "WARNING: Disk usage is ${USAGE}%"
fi
```


---

## 五、大厂实战篇

### 1. 消息积压怎么处理？

**答案：**

消息积压是 Kafka 生产环境最常见的问题之一，需要从预防、监控、处理三个层面解决。

#### 积压原因分析

```
消息积压常见原因：

┌─────────────────────────────────────────────────────────────┐
│                      生产端                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 生产者速率突增（促销活动、流量高峰）               │   │
│  │ 2. 批量导入数据                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│                      消费端                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 消费者处理能力不足                                │   │
│  │ 2. 下游服务响应慢（数据库、外部API）                  │   │
│  │ 3. 消费者频繁 Rebalance                              │   │
│  │ 4. 消费者宕机                                        │   │
│  │ 5. 消费逻辑异常导致重试                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 监控与告警

```yaml
# 关键监控指标

# 1. 消费延迟（Lag）
kafka.consumer:type=consumer-fetch-manager-metrics,client-id=*
  - records-lag-max
  - records-lag-avg

# 2. 消费速率
kafka.consumer:type=consumer-fetch-manager-metrics,client-id=*
  - records-consumed-rate
  - bytes-consumed-rate

# 3. 生产速率
kafka.producer:type=producer-metrics,client-id=*
  - record-send-rate

# 告警规则
alerts:
  - name: ConsumerLagHigh
    expr: kafka_consumer_lag > 100000
    severity: warning
    
  - name: ConsumerLagCritical
    expr: kafka_consumer_lag > 1000000
    severity: critical
```

#### 处理方案

```java
// 方案1：临时扩容 Consumer

// 原有：3 个 Consumer，12 个 Partition
// 积压时：扩展到 12 个 Consumer

// 注意：
// 1. Consumer 数量不能超过 Partition 数量
// 2. 需要同步增加下游服务处理能力

// 方案2：临时转发积压消息
// 创建一个高吞吐的临时 Topic
// 将积压消息转发到临时 Topic
// 用多个 Consumer 并行处理

public class LagHandler {
    
    private final KafkaConsumer<String, String> mainConsumer;
    private final KafkaProducer<String, String> producer;
    
    public void handleLag() {
        // 检测积压
        Map<TopicPartition, OffsetAndMetadata> committed = 
            mainConsumer.committed(Collections.singleton(new TopicPartition("topic", 0)));
        
        long lag = getLag();
        
        if (lag > THRESHOLD) {
            // 启动临时处理线程
            startTemporaryConsumers(10);  // 10 个临时 Consumer
        }
    }
    
    private void startTemporaryConsumers(int count) {
        ExecutorService executor = Executors.newFixedThreadPool(count);
        for (int i = 0; i < count; i++) {
            executor.submit(new TemporaryConsumer("topic"));
        }
    }
}
```

```java
// 方案3：批量快速跳过积压（数据可丢弃场景）

public void skipLag() {
    // 获取最新 offset
    Map<TopicPartition, Long> endOffsets = consumer.endOffsets(assignment);
    
    // 直接跳到最新
    for (Map.Entry<TopicPartition, Long> entry : endOffsets.entrySet()) {
        consumer.seek(entry.getKey(), entry.getValue());
    }
    
    // 或者跳到指定时间点
    long timestamp = System.currentTimeMillis() - 3600000;  // 1小时前
    Map<TopicPartition, OffsetAndTimestamp> offsets = 
        consumer.offsetsForTimes(Collections.singletonMap(tp, timestamp));
    
    for (Map.Entry<TopicPartition, OffsetAndTimestamp> entry : offsets.entrySet()) {
        if (entry.getValue() != null) {
            consumer.seek(entry.getKey(), entry.getValue().offset());
        }
    }
}
```

#### 预防措施

```java
// 1. 生产者限流
public class RateLimitedProducer {
    
    private final RateLimiter rateLimiter = RateLimiter.create(10000);  // 10000 TPS
    
    public void send(String topic, String key, String value) {
        rateLimiter.acquire();  // 限流
        producer.send(new ProducerRecord<>(topic, key, value));
    }
}

// 2. 消费者健康监控
public class HealthyConsumer {
    
    private final long maxProcessTime = 1000;  // 最大处理时间
    
    public void consume() {
        while (true) {
            ConsumerRecords records = consumer.poll(Duration.ofMillis(100));
            
            long startTime = System.currentTimeMillis();
            process(records);
            long processTime = System.currentTimeMillis() - startTime;
            
            // 处理时间过长告警
            if (processTime > maxProcessTime * records.count()) {
                alertSlowProcessing(processTime, records.count());
            }
        }
    }
}

// 3. 背压机制
public class BackPressureConsumer {
    
    private final Semaphore semaphore = new Semaphore(1000);  // 最多处理 1000 条
    
    public void consume() {
        while (true) {
            ConsumerRecords records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord record : records) {
                semaphore.acquire();  // 获取信号量
                executor.submit(() -> {
                    try {
                        process(record);
                    } finally {
                        semaphore.release();
                    }
                });
            }
        }
    }
}
```

---

### 2. 怎么实现延迟消息？

**答案：**

Kafka 原生不支持延迟消息，但可以通过多种方式实现。

#### 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Topic 分层 | 实现简单 | Topic 数量多 | 延迟级别固定 |
| 时间轮 | 高效、精确 | 实现复杂 | 高性能需求 |
| 外部调度 | 灵活 | 依赖外部系统 | 已有调度系统 |
| 升级 Kafka | 原生支持 | 需要升级 | 新版本 2.7+ |

#### 方案1：Topic 分层（推荐）

```
实现原理：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Producer                                                   │
│     │                                                       │
│     ↓ 发送延迟消息                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Scheduler Service                                    │   │
│  │                                                      │   │
│  │  延迟级别：                                          │   │
│  │  - delay_1s  (1秒后)                                │   │
│  │  - delay_5s  (5秒后)                                │   │
│  │  - delay_30s (30秒后)                               │   │
│  │  - delay_1m  (1分钟后)                              │   │
│  │  - delay_5m  (5分钟后)                              │   │
│  │  - delay_1h  (1小时后)                              │   │
│  └─────────────────────────────────────────────────────┘   │
│     │                                                       │
│     ↓ 等待延迟时间到达                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Topic: delay_5s                                      │   │
│  │                                                      │   │
│  │ Consumer: DelayScheduler                            │   │
│  │   - 等待 5 秒                                        │   │
│  │   - 转发到目标 Topic                                │   │
│  └─────────────────────────────────────────────────────┘   │
│     │                                                       │
│     ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Topic: target-topic                                  │   │
│  │ Consumer: BusinessConsumer                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```java
// 延迟消息发送
public class DelayMessageProducer {
    
    private final KafkaProducer<String, String> producer;
    private final Map<Integer, String> delayTopics = Map.of(
        1, "delay_1s",
        5, "delay_5s",
        30, "delay_30s",
        60, "delay_1m",
        300, "delay_5m",
        3600, "delay_1h"
    );
    
    public void sendDelayMessage(String targetTopic, String key, String value, int delaySeconds) {
        // 找到合适的延迟 Topic
        String delayTopic = findNearestDelayTopic(delaySeconds);
        
        // 构造延迟消息
        JSONObject message = new JSONObject();
        message.put("targetTopic", targetTopic);
        message.put("key", key);
        message.put("value", value);
        message.put("executeTime", System.currentTimeMillis() + delaySeconds * 1000);
        
        producer.send(new ProducerRecord<>(delayTopic, key, message.toJSONString()));
    }
    
    private String findNearestDelayTopic(int delaySeconds) {
        // 找到 >= delaySeconds 的最小延迟级别
        return delayTopics.entrySet().stream()
            .filter(e -> e.getKey() >= delaySeconds)
            .min(Map.Entry.comparingByKey())
            .map(Map.Entry::getValue)
            .orElse("delay_1h");
    }
}
```

```java
// 延迟消息消费者（调度器）
public class DelayScheduler {
    
    public void start() {
        for (String delayTopic : delayTopics.values()) {
            new Thread(() -> consumeDelayTopic(delayTopic)).start();
        }
    }
    
    private void consumeDelayTopic(String delayTopic) {
        KafkaConsumer<String, String> consumer = createConsumer(delayTopic);
        
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                JSONObject message = JSONObject.parseObject(record.value());
                long executeTime = message.getLong("executeTime");
                long waitTime = executeTime - System.currentTimeMillis();
                
                if (waitTime > 0) {
                    // 还没到执行时间，等待
                    Thread.sleep(waitTime);
                }
                
                // 转发到目标 Topic
                producer.send(new ProducerRecord<>(
                    message.getString("targetTopic"),
                    message.getString("key"),
                    message.getString("value")
                ));
            }
            
            consumer.commitSync();
        }
    }
}
```

#### 方案2：时间轮算法

```java
// 时间轮实现延迟消息
public class TimingWheel {
    
    private final int tickMs;           // 每格时间（毫秒）
    private final int wheelSize;        // 格数
    private final long interval;        // 总时间范围
    private final List<TimerTaskEntry>[] buckets;
    private long currentTimeMs;
    
    public TimingWheel(int tickMs, int wheelSize) {
        this.tickMs = tickMs;
        this.wheelSize = wheelSize;
        this.interval = tickMs * wheelSize;
        this.buckets = new List[wheelSize];
        for (int i = 0; i < wheelSize; i++) {
            buckets[i] = new CopyOnWriteArrayList<>();
        }
        this.currentTimeMs = System.currentTimeMillis();
    }
    
    // 添加延迟任务
    public boolean add(TimerTaskEntry entry) {
        long delayMs = entry.delayMs;
        
        if (delayMs < tickMs) {
            // 已过期，立即执行
            return false;
        }
        
        if (delayMs < interval) {
            // 在当前时间轮范围内
            int bucketIndex = (int) ((currentTimeMs + delayMs) / tickMs % wheelSize);
            buckets[bucketIndex].add(entry);
            return true;
        }
        
        // 超出当前时间轮范围，需要溢出处理
        return false;
    }
    
    // 时间推进
    public void advanceClock(long timeMs) {
        if (timeMs >= currentTimeMs + tickMs) {
            currentTimeMs = timeMs - (timeMs % tickMs);
            
            // 处理当前格的任务
            int bucketIndex = (int) (currentTimeMs / tickMs % wheelSize);
            for (TimerTaskEntry entry : buckets[bucketIndex]) {
                entry.execute();
            }
            buckets[bucketIndex].clear();
        }
    }
}

class TimerTaskEntry {
    String topic;
    String key;
    String value;
    long delayMs;
    
    void execute() {
        // 发送消息到目标 Topic
    }
}
```

---

### 3. 怎么保证消息顺序消费？

**答案：**

Kafka 只保证分区内的顺序，跨分区需要特殊处理。

#### 方案1：单分区（简单场景）

```java
// 创建单分区 Topic
// 优点：实现简单
// 缺点：吞吐受限

// 发送时指定 key，相同 key 的消息会进入同一分区
producer.send(new ProducerRecord<>("ordered-topic", orderId, message));
```

#### 方案2：按 key 分区

```java
// 相同 key 的消息进入同一分区，保证有序
public class OrderedProducer {
    
    public void sendOrdered(String topic, String orderId, String message) {
        // 使用 orderId 作为 key
        // Kafka 会将相同 key 的消息发送到同一分区
        producer.send(new ProducerRecord<>(topic, orderId, message));
    }
}

// 消费者单线程处理
public class OrderedConsumer {
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            // 单线程处理，保证顺序
            for (ConsumerRecord<String, String> record : records) {
                process(record);  // 顺序处理
            }
            
            consumer.commitSync();
        }
    }
}
```

#### 方案3：分区队列并行

```java
// 每个分区一个处理队列，保证分区内有序
public class PartitionOrderedConsumer {
    
    private final Map<TopicPartition, BlockingQueue<ConsumerRecord<String, String>>> queues;
    private final ExecutorService executor;
    
    public void start() {
        // 为每个分区创建独立队列和处理线程
        for (TopicPartition tp : assignment) {
            queues.put(tp, new LinkedBlockingQueue<>());
            executor.submit(() -> processPartition(tp));
        }
        
        // 消费线程
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (TopicPartition tp : records.partitions()) {
                queues.get(tp).addAll(records.records(tp));
            }
        }
    }
    
    private void processPartition(TopicPartition tp) {
        BlockingQueue<ConsumerRecord<String, String>> queue = queues.get(tp);
        
        while (true) {
            ConsumerRecord<String, String> record = queue.take();
            process(record);  // 分区内顺序处理
        }
    }
}
```

#### 方案4：全局有序（牺牲性能）

```java
// 使用时间窗口 + 排序
public class GlobalOrderedConsumer {
    
    private final PriorityBlockingQueue<OrderedMessage> buffer = 
        new PriorityBlockingQueue<>(1000, Comparator.comparingLong(m -> m.timestamp));
    
    private final long windowSize = 5000;  // 5秒窗口
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            // 放入缓冲区
            for (ConsumerRecord<String, String> record : records) {
                buffer.offer(new OrderedMessage(record));
            }
            
            // 取出窗口内的消息
            long now = System.currentTimeMillis();
            while (!buffer.isEmpty()) {
                OrderedMessage msg = buffer.peek();
                if (now - msg.timestamp >= windowSize) {
                    buffer.poll();
                    process(msg.record);
                } else {
                    break;
                }
            }
        }
    }
}
```

---

### 4. Kafka 怎么做消息追踪？

**答案：**

消息追踪对于问题排查和业务监控非常重要，Kafka 本身提供基础能力，也可以扩展实现。

#### 内置追踪能力

```java
// 1. Producer 端记录
public class TraceableProducer {
    
    private final KafkaProducer<String, String> producer;
    
    public void send(String topic, String key, String value) {
        // 生成 Trace ID
        String traceId = UUID.randomUUID().toString();
        
        // 添加到消息 Header
        Headers headers = new RecordHeaders();
        headers.add("traceId", traceId.getBytes());
        headers.add("timestamp", String.valueOf(System.currentTimeMillis()).getBytes());
        
        ProducerRecord<String, String> record = new ProducerRecord<>(
            topic, null, key, value, headers
        );
        
        // 发送并记录
        producer.send(record, (metadata, exception) -> {
            if (exception == null) {
                log.info("Message sent - traceId: {}, topic: {}, partition: {}, offset: {}",
                    traceId, metadata.topic(), metadata.partition(), metadata.offset());
            }
        });
    }
}

// 2. Consumer 端提取
public class TraceableConsumer {
    
    public void consume() {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, String> record : records) {
                // 提取 Trace ID
                Header traceIdHeader = record.headers().lastHeader("traceId");
                String traceId = traceIdHeader != null ? 
                    new String(traceIdHeader.value()) : "unknown";
                
                // 记录消费日志
                log.info("Message consumed - traceId: {}, topic: {}, partition: {}, offset: {}",
                    traceId, record.topic(), record.partition(), record.offset());
                
                process(record);
            }
        }
    }
}
```

#### 使用 Kafka Interceptor

```java
// Producer Interceptor
public class TracingProducerInterceptor implements ProducerInterceptor<String, String> {
    
    @Override
    public ProducerRecord<String, String> onSend(ProducerRecord<String, String> record) {
        // 添加追踪信息
        Headers headers = record.headers();
        headers.add("traceId", generateTraceId());
        headers.add("producerTime", String.valueOf(System.currentTimeMillis()).getBytes());
        
        return record;
    }
    
    @Override
    public void onAcknowledgement(RecordMetadata metadata, Exception exception) {
        // 记录发送结果
        TraceLogger.log(metadata, exception);
    }
}

// Consumer Interceptor
public class TracingConsumerInterceptor implements ConsumerInterceptor<String, String> {
    
    @Override
    public ConsumerRecords<String, String> onConsume(ConsumerRecords<String, String> records) {
        for (ConsumerRecord<String, String> record : records) {
            // 记录消费信息
            TraceLogger.logConsume(record);
        }
        return records;
    }
}

// 配置
props.put("interceptor.classes", 
    "com.example.TracingProducerInterceptor");
```

#### 外部追踪系统集成

```java
// 集成 OpenTelemetry
public class OpenTelemetryTracing {
    
    private final Tracer tracer;
    
    public void sendWithTracing(String topic, String key, String value) {
        // 创建 Span
        Span span = tracer.spanBuilder("kafka.produce")
            .setAttribute("topic", topic)
            .setAttribute("key", key)
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            // 注入 Trace Context 到消息头
            TextMapSetter<Headers> setter = (headers, key1, value1) -> 
                headers.add(key1, value1.getBytes());
            
            OpenTelemetry.getPropagators().getTextMapPropagator()
                .inject(Context.current(), new RecordHeaders(), setter);
            
            producer.send(new ProducerRecord<>(topic, key, value));
            
        } finally {
            span.end();
        }
    }
    
    public void consumeWithTracing(ConsumerRecord<String, String> record) {
        // 从消息头提取 Trace Context
        TextMapGetter<Headers> getter = (headers, key) -> {
            Header header = headers.lastHeader(key);
            return header != null ? new String(header.value()) : null;
        };
        
        Context context = OpenTelemetry.getPropagators().getTextMapPropagator()
            .extract(Context.current(), record.headers(), getter);
        
        // 创建消费 Span
        Span span = tracer.spanBuilder("kafka.consume")
            .setParent(context)
            .setAttribute("topic", record.topic())
            .setAttribute("partition", record.partition())
            .setAttribute("offset", record.offset())
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            process(record);
        } finally {
            span.end();
        }
    }
}
```

---

### 5. Kafka 怎么做灰度发布？

**答案：**

Kafka 灰度发布需要考虑生产者和消费者两个层面的平滑切换。

#### 灰度发布架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka 灰度发布架构                        │
│                                                             │
│  生产者灰度：                                                │
│  ┌───────────┐     ┌───────────┐                           │
│  │ Producer  │     │ Producer  │                           │
│  │ v1 (90%)  │     │ v2 (10%)  │  ← 灰度版本               │
│  └─────┬─────┘     └─────┬─────┘                           │
│        │                 │                                  │
│        └────────┬────────┘                                  │
│                 ↓                                           │
│         ┌───────────────┐                                   │
│         │    Topic      │                                   │
│         │  (双版本兼容)  │                                   │
│         └───────┬───────┘                                   │
│                 │                                           │
│  消费者灰度：    │                                           │
│        ┌────────┴────────┐                                  │
│        ↓                 ↓                                  │
│  ┌───────────┐     ┌───────────┐                           │
│  │ Consumer  │     │ Consumer  │                           │
│  │ v1        │     │ v2 (灰度) │                           │
│  └───────────┘     └───────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 生产者灰度策略

```java
// 1. 按比例灰度
public class GrayProducer {
    
    private final double grayRatio = 0.1;  // 10% 流量走新版本
    private final KafkaProducer<String, String> producerV1;
    private final KafkaProducer<String, String> producerV2;
    
    public void send(String topic, String key, String value) {
        // 根据比例选择版本
        if (shouldUseV2(key)) {
            producerV2.send(new ProducerRecord<>(topic, key, value));
        } else {
            producerV1.send(new ProducerRecord<>(topic, key, value));
        }
    }
    
    private boolean shouldUseV2(String key) {
        // 基于 key hash 实现稳定灰度
        return Math.abs(key.hashCode()) % 100 < grayRatio * 100;
    }
}

// 2. 按特征灰度
public class FeatureGrayProducer {
    
    public void send(String topic, String key, String value, User user) {
        // 指定用户走新版本
        if (isGrayUser(user)) {
            sendWithV2(topic, key, value);
        } else {
            sendWithV1(topic, key, value);
        }
    }
    
    private boolean isGrayUser(User user) {
        return grayUserIds.contains(user.getId()) || 
               user.isInternal();  // 内部用户
    }
}
```

#### 消费者灰度策略

```java
// 1. 独立 Consumer Group
public class GrayConsumer {
    
    // v1 Consumer Group
    private final KafkaConsumer<String, String> consumerV1;
    
    // v2 Consumer Group（灰度）
    private final KafkaConsumer<String, String> consumerV2;
    
    public void start() {
        // v1 消费者
        propsV1.put("group.id", "consumer-group-v1");
        consumerV1 = new KafkaConsumer<>(propsV1);
        
        // v2 消费者（灰度，独立消费）
        propsV2.put("group.id", "consumer-group-v2");
        consumerV2 = new KafkaConsumer<>(propsV2);
        
        // 各自独立消费，互不影响
        new Thread(() -> consumeV1()).start();
        new Thread(() -> consumeV2()).start();
    }
}

// 2. 同组灰度（需要特殊处理）
public class SameGroupGrayConsumer {
    
    public void consume() {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        
        for (ConsumerRecord<String, String> record : records) {
            // 根据消息特征判断处理逻辑
            if (isGrayMessage(record)) {
                processV2(record);  // 新逻辑
            } else {
                processV1(record);  // 旧逻辑
            }
        }
    }
}
```

#### Topic 灰度迁移

```java
// Topic 级别灰度
public class TopicGrayMigration {
    
    private final KafkaProducer<String, String> producer;
    private final String oldTopic = "topic-v1";
    private final String newTopic = "topic-v2";
    
    // 双写阶段
    public void sendDualWrite(String key, String value) {
        // 同时写入新旧 Topic
        producer.send(new ProducerRecord<>(oldTopic, key, value));
        producer.send(new ProducerRecord<>(newTopic, key, value));
    }
    
    // 切换阶段
    public void sendToNew(String key, String value) {
        // 只写入新 Topic
        producer.send(new ProducerRecord<>(newTopic, key, value));
    }
}
```

---

### 6. 跨机房复制怎么做？

**答案：**

Kafka 跨机房复制用于实现异地多活、灾备等场景，需要考虑网络延迟和数据一致性。

#### 跨机房复制方案

```
┌─────────────────────────────────────────────────────────────┐
│                    跨机房复制架构                            │
│                                                             │
│      机房 A (主)                    机房 B (备)              │
│  ┌───────────────┐              ┌───────────────┐          │
│  │   Kafka       │              │   Kafka       │          │
│  │   Cluster A   │──────────────│   Cluster B   │          │
│  └───────────────┘              └───────────────┘          │
│         │                              │                    │
│         │                              │                    │
│  ┌───────────────┐              ┌───────────────┐          │
│  │  MirrorMaker  │              │  MirrorMaker  │          │
│  │   或 uReplicator│─────────────│   或 uReplicator│        │
│  └───────────────┘              └───────────────┘          │
│                                                             │
│  复制方向：                                                  │
│  - Active-Passive：单向复制，A → B                          │
│  - Active-Active：双向复制，A ↔ B（需避免循环）             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### MirrorMaker 2.0 配置

```yaml
# mm2.properties
clusters = A, B

A.bootstrap.servers = broker-a1:9092,broker-a2:9092
B.bootstrap.servers = broker-b1:9092,broker-b2:9092

# A → B 复制
A->B.enabled = true
A->B.topics = .*
A->B.groups = .*

# B → A 复制（双向）
B->A.enabled = true
B->A.topics = .*
B->A.groups = .*

# Topic 命名规则
# 复制后的 Topic 格式：source.topic-name
# 例如：A.topic1 → B 中变成 A.topic1

# 心跳配置
replication.policy.class = org.apache.kafka.connect.mirror.IdentityReplicationPolicy
```

```bash
# 启动 MirrorMaker 2.0
bin/connect-mirror-maker.sh mm2.properties
```

#### uReplicator（Uber 方案）

```yaml
# uReplicator 适合大规模复制
# 相比 MirrorMaker：
# 1. 更高的吞吐量
# 2. 更低的延迟
# 3. 支持动态 Topic 发现

# Controller 配置
controller.config:
  src.kafka.bootstrap.servers: "broker-a1:9092,broker-a2:9092"
  dest.kafka.bootstrap.servers: "broker-b1:9092,broker-b2:9092"
  topic.whitelist: ".*"
  topic.blacklist: ".*__consumer_offsets.*"
```

#### 消费者 Failover

```java
// 消费者切换到备机房
public class FailoverConsumer {
    
    private KafkaConsumer<String, String> consumer;
    private String currentCluster = "A";
    
    public void failover() {
        // 1. 保存当前消费位置
        Map<TopicPartition, OffsetAndMetadata> offsets = 
            consumer.committed(consumer.assignment());
        
        // 2. 关闭当前消费者
        consumer.close();
        
        // 3. 连接备机房
        Properties props = new Properties();
        props.put("bootstrap.servers", getBackupBrokers());
        props.put("group.id", "failover-group");
        consumer = new KafkaConsumer<>(props);
        
        // 4. 从保存的位置开始消费
        // 注意：需要转换 Topic 名称（MirrorMaker 复制后会有前缀）
        String newTopic = "A." + oldTopic;
        consumer.subscribe(Collections.singletonList(newTopic));
        
        // 5. Seek 到之前的位置
        for (Map.Entry<TopicPartition, OffsetAndMetadata> entry : offsets.entrySet()) {
            TopicPartition newTp = new TopicPartition(
                "A." + entry.getKey().topic(), 
                entry.getKey().partition()
            );
            consumer.seek(newTp, entry.getValue().offset());
        }
    }
}
```

---

### 7. 线上 Kafka 问题排查经验？

**答案：**

线上 Kafka 问题排查需要系统化的方法论，以下是常见问题的排查思路。

#### 排查工具箱

```bash
# 1. 集群状态检查
# Broker 是否在线
bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# Topic 详情
bin/kafka-topics.sh --describe --topic my-topic \
  --bootstrap-server localhost:9092

# Consumer Group 状态
bin/kafka-consumer-groups.sh --describe --group my-group \
  --bootstrap-server localhost:9092

# 2. 消息查看
# 从头查看消息
bin/kafka-console-consumer.sh --topic my-topic \
  --from-beginning --bootstrap-server localhost:9092

# 查看特定 offset 的消息
bin/kafka-console-consumer.sh --topic my-topic \
  --partition 0 --offset 1000 \
  --bootstrap-server localhost:9092 --max-messages 10

# 3. 性能测试
# 生产者性能
bin/kafka-producer-perf-test.sh --topic test \
  --num-records 100000 --record-size 1000 \
  --throughput -1 --bootstrap-server localhost:9092

# 消费者性能
bin/kafka-consumer-perf-test.sh --topic test \
  --messages 100000 --bootstrap-server localhost:9092

# 4. 日志分析
# Broker 日志
tail -f /var/log/kafka/server.log

# Controller 日志
grep "Controller" /var/log/kafka/server.log
```

#### 常见问题排查

```yaml
问题1：生产者发送超时
现象：producer.send() 超时或失败

排查步骤：
1. 检查网络连通性：telnet broker-host 9092
2. 检查 Broker 状态：jps -l，查看 Kafka 进程
3. 检查 Topic 是否存在：kafka-topics.sh --list
4. 检查磁盘空间：df -h
5. 检查 Broker 日志：是否有错误

常见原因：
- Broker 宕机
- 网络分区
- Topic 不存在
- 磁盘满

问题2：消费者 Lag 持续增长
现象：消费延迟越来越大

排查步骤：
1. 查看消费者状态：kafka-consumer-groups.sh --describe
2. 检查消费者日志：是否有异常
3. 检查下游服务：响应时间、错误率
4. 检查 Broker 负载：CPU、IO、网络

常见原因：
- 消费者处理慢
- 下游服务瓶颈
- Rebalance 频繁
- 消费者数量不足

问题3：Broker 频繁切换 Leader
现象：Controller 日志显示频繁选举

排查步骤：
1. 检查 Broker 健康状态
2. 检查网络延迟：ping、traceroute
3. 检查 Zookeeper 连接状态
4. 检查 Broker GC 日志

常见原因：
- Broker GC 频繁
- 网络抖动
- Zookeeper 不稳定
- 磁盘 IO 高
```

#### 排查脚本示例

```bash
#!/bin/bash
# kafka_health_check.sh - Kafka 健康检查脚本

BOOTSTRAP_SERVER="localhost:9092"

echo "=== Kafka Health Check ==="
echo ""

# 1. Broker 状态
echo "1. Broker Status:"
bin/kafka-broker-api-versions.sh --bootstrap-server $BOOTSTRAP_SERVER 2>&1 | head -5

# 2. Topic 列表
echo ""
echo "2. Topics:"
bin/kafka-topics.sh --list --bootstrap-server $BOOTSTRAP_SERVER

# 3. Under Replicated Partitions
echo ""
echo "3. Under Replicated Partitions:"
bin/kafka-topics.sh --describe --under-replicated-partitions \
  --bootstrap-server $BOOTSTRAP_SERVER

# 4. Offline Partitions
echo ""
echo "4. Offline Partitions:"
bin/kafka-topics.sh --describe --unavailable-partitions \
  --bootstrap-server $BOOTSTRAP_SERVER 2>&1

# 5. Consumer Group Lag
echo ""
echo "5. Consumer Groups with High Lag:"
for group in $(bin/kafka-consumer-groups.sh --list --bootstrap-server $BOOTSTRAP_SERVER); do
    lag=$(bin/kafka-consumer-groups.sh --describe --group $group \
        --bootstrap-server $BOOTSTRAP_SERVER 2>/dev/null | \
        awk 'NR>1 {sum+=$5} END {print sum}')
    if [ ! -z "$lag" ] && [ "$lag" -gt 10000 ]; then
        echo "  $group: LAG = $lag"
    fi
done

# 6. Disk Usage
echo ""
echo "6. Disk Usage:"
df -h /data/kafka-logs 2>/dev/null | tail -1

echo ""
echo "=== Health Check Complete ==="
```

#### JMX 监控关键指标

```yaml
# Broker 关键 JMX 指标

# 1. 请求处理
kafka.network:type=RequestMetrics,name=RequestsPerSec
kafka.network:type=RequestChannel,name=RequestQueueSize

# 2. 日志
kafka.log:type=Log,name=Size
kafka.log:type=Log,name=NumLogSegments
kafka.log:type=LogManager,name=OfflineLogDirectoryCount

# 3. 副本
kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions
kafka.server:type=ReplicaManager,name=OfflineReplicaCount
kafka.server:type=ReplicaManager,name=IsrShrinksPerSec

# 4. Controller
kafka.controller:type=KafkaController,name=OfflinePartitionsCount
kafka.controller:type=KafkaController,name=ActiveControllerCount
```

---

## 参考资料

- [Kafka 官方文档](https://kafka.apache.org/documentation/)
- [Kafka 内部原理](https://developer.confluent.io/courses/apache-kafka/internals/)
- [Kafka 运维实践](https://docs.confluent.io/platform/current/kafka/operations.html)
- [Kafka 性能优化](https://www.confluent.io/blog/apache-kafka-purgatory-hierarchical-timing-wheels/)
