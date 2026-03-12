# Redis 面试题

> 本文档整理 Redis 相关的高频面试题。

---

## 一、基础概念

### 1. Redis 是什么？有什么特点？

**答案：**

Redis 是一个开源的内存数据结构存储系统，可用作：

- 数据库
- 缓存
- 消息队列
- 分布式锁

**特点：**

| 特性 | 说明 |
|------|------|
| 性能 | 内存存储，读写速度极快 |
| 数据类型 | String、List、Set、Hash、ZSet、Stream、Geo 等 |
| 持久化 | RDB + AOF |
| 高可用 | 主从复制、哨兵、集群 |
| 功能丰富 | 发布订阅、Lua 脚本、事务、管道 |

---

### 2. Redis 为什么快？

**答案：**

```
1. 内存存储
   ├── 数据在内存中，读写速度极快
   └── 内存访问 ~100ns，磁盘访问 ~10ms

2. 单线程模型
   ├── 避免线程切换开销
   ├── 避免锁竞争
   └── IO 多路复用

3. IO 多路复用
   ├── epoll（Linux）
   ├── kqueue（macOS）
   └── 单线程处理大量连接

4. 高效数据结构
   ├── SDS（简单动态字符串）
   ├── 哈希表
   ├── 跳表
   ├── 压缩列表（listpack）
   └── 整数集合
```

**追问：Redis 6.0 为什么引入多线程？**

> 单线程处理网络 IO 成为瓶颈。Redis 6.0 引入多线程处理网络读写（解码、编码），但命令执行仍是单线程。

---

## 二、数据类型

### 3. Redis 有哪些数据类型？

**答案：**

| 类型 | 底层实现 | 适用场景 |
|------|----------|----------|
| String | SDS、int | 缓存、计数器、分布式锁 |
| List | quicklist | 消息队列、时间线 |
| Set | intset、hashtable | 标签、共同关注 |
| Hash | listpack、hashtable | 用户信息、购物车 |
| ZSet | listpack、skiplist+hashtable | 排行榜、延迟队列 |
| Stream | listpack | 消息队列 |
| Geo | ZSet | 地理位置 |
| HyperLogLog | 稀疏/密集编码 | 基数统计 |
| Bitmap | String | 签到、布隆过滤器 |

---

### 4. String 类型的应用场景？

**答案：**

```bash
# 1. 缓存
SET user:1 '{"name":"张三","age":25}'
GET user:1

# 2. 计数器
INCR page:views
INCRBY article:1:likes 1

# 3. 分布式锁
SET lock:order:123 "uuid" NX PX 30000

# 4. 限流
INCR rate:limit:user:1
EXPIRE rate:limit:user:1 60

# 5. 分布式 ID
INCR global:id
```

---

### 5. ZSet 实现排行榜？

**答案：**

```bash
# 添加成员和分数
ZADD leaderboard 100 user:1
ZADD leaderboard 200 user:2
ZADD leaderboard 150 user:3

# 获取排行榜（降序）
ZREVRANGE leaderboard 0 9 WITHSCORES

# 获取用户排名
ZREVRANK leaderboard user:1

# 获取用户分数
ZSCORE leaderboard user:1

# 增加分数
ZINCRBY leaderboard 10 user:1
```

---

## 三、持久化

### 6. RDB 和 AOF 的区别？

**答案：**

| 特性 | RDB | AOF |
|------|-----|-----|
| 方式 | 快照 | 写命令追加 |
| 文件大小 | 小 | 大 |
| 恢复速度 | 快 | 慢 |
| 数据安全 | 可能丢失 | 更安全 |
| 系统开销 | 低（fork 时高） | 高（持续写入） |

```
RDB 触发条件：
├── SAVE（阻塞）
├── BGSAVE（后台 fork）
└── 配置自动触发（save 900 1）

AOF 触发条件：
├── appendonly yes
└── appendfsync everysec
```

**追问：如何选择？**

> - 数据安全性要求高：AOF + RDB
> - 允许少量丢失：RDB
> - 生产环境推荐：同时开启（AOF 优先加载）

---

### 7. AOF 重写是什么？

**答案：**

AOF 文件会越来越大，重写可以压缩文件大小。

```
重写前：
SET key1 value1
SET key1 value2
SET key1 value3
SET key2 value1
DEL key2

重写后：
SET key1 value3

原理：
1. fork 子进程
2. 子进程根据当前内存数据生成新的 AOF
3. 主进程继续接收命令，写入重写缓冲区
4. 子进程完成后，主进程将缓冲区追加到新文件
5. 原子替换旧 AOF 文件
```

---

## 四、高可用

### 8. 主从复制原理？

**答案：**

```
1. 建立连接
   slave → master: SYNC/PSYNC

2. 全量复制（首次）
   ├── master 执行 BGSAVE 生成 RDB
   ├── master 将 RDB 发送给 slave
   ├── slave 加载 RDB
   └── master 发送期间的写命令

3. 增量复制（断线重连）
   ├── master 维持复制积压缓冲区
   ├── slave 发送偏移量
   └── master 发送缺失的数据

4. 命令传播
   master 收到写命令 → 发送给所有 slave
```

---

### 9. 哨兵机制？

**答案：**

```
哨兵（Sentinel）功能：
├── 监控：检测 master、slave 是否正常
├── 通知：通知应用方故障
├── 自动故障转移：选举新的 master
└── 配置提供：提供 master 地址

故障转移流程：
1. 哨兵检测到 master 下线（主观下线）
2. 多数哨兵确认（客观下线）
3. 选举领头哨兵
4. 领头哨兵选举新 master
   ├── 选择数据最新的 slave
   ├── 排除不健康的 slave
   └── 优先级配置
5. 通知其他 slave 复制新 master
6. 通知客户端新 master 地址
```

---

### 10. Redis Cluster 原理？

**答案：**

```
数据分片：
├── 16384 个槽位
├── 每个节点负责一部分槽位
└── key 的槽位 = CRC16(key) % 16384

集群结构：
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Master1 │   │ Master2 │   │ Master3 │
│ 0-5460  │   │ 5461-10922│ │ 10923-16383│
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
┌────┴────┐   ┌────┴────┐   ┌────┴────┐
│ Slave1  │   │ Slave2  │   │ Slave3  │
└─────────┘   └─────────┘   └─────────┘

客户端访问：
1. 客户端连接任意节点
2. 节点计算 key 的槽位
3. 如果槽位不在当前节点，返回 MOVED 重定向
4. 客户端缓存槽位映射，下次直接访问
```

---

## 五、缓存问题

### 11. 缓存穿透怎么解决？

**答案：**

```
问题：查询不存在的数据，缓存没有，数据库也没有
结果：请求直接打到数据库

解决方案：

1. 缓存空值
   GET user:999 → null → SET user:999 "" EX 60
   缺点：内存占用

2. 布隆过滤器
   ├── 预热时将所有 key 加入布隆过滤器
   ├── 查询前先问布隆过滤器
   └── 不存在则直接返回
   优点：内存占用小
   缺点：有误判率

3. 参数校验
   在应用层拦截非法请求
```

---

### 12. 缓存击穿怎么解决？

**答案：**

```
问题：热点 key 过期，大量请求瞬间打到数据库

解决方案：

1. 热点数据永不过期
   ├── 不设置过期时间
   └── 后台异步更新

2. 互斥锁
   SETNX lock:key 1
   ├── 获取锁的线程查数据库
   ├── 其他线程等待
   └── 查完后释放锁

3. 逻辑过期
   ├── value 中包含过期时间字段
   ├── 过期后不立即删除
   └── 后台线程异步更新
```

---

### 13. 缓存雪崩怎么解决？

**答案：**

```
问题：大量 key 同时过期，或 Redis 宕机
结果：所有请求打到数据库

解决方案：

1. 过期时间随机化
   SET key value EX 3600 + random(600)

2. 多级缓存
   Redis → 本地缓存 → 数据库

3. 熔断降级
   ├── Hystrix / Sentinel
   └── 数据库压力大时返回降级数据

4. 高可用
   ├── 主从复制
   ├── 哨兵
   └── 集群
```

---

### 14. 缓存和数据库一致性？

**答案：**

```
方案一：先删缓存，再更新数据库
问题：并发时可能产生脏数据
解决：延迟双删

方案二：先更新数据库，再删缓存（推荐）
问题：删缓存失败
解决：消息队列重试

方案三：写数据库，发消息删缓存
├── 更新数据库
├── 发送消息到 MQ
└── 消费者删除缓存，失败重试

方案四：订阅 binlog
├── Canal 监听 MySQL binlog
└── 异步删除 Redis 缓存
```

---

## 六、其他问题

### 15. Redis 如何实现分布式锁？

**答案：**

```bash
# 加锁
SET lock:key "uuid" NX PX 30000

# 释放锁（Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

**追问：Redisson 有什么优化？**

> 1. **看门狗机制**：自动续期，防止业务未执行完锁过期
> 2. **可重入锁**：基于 Hash 实现
> 3. **等待队列**：获取失败时加入队列等待
> 4. **RedLock**：多节点加锁，防止单点故障

---

### 16. Redis 内存淘汰策略？

**答案：**

| 策略 | 说明 |
|------|------|
| noeviction | 不淘汰，内存满时报错（默认） |
| allkeys-lru | 所有 key 中 LRU 淘汰 |
| volatile-lru | 设置了过期时间的 key 中 LRU 淘汰 |
| allkeys-lfu | 所有 key 中 LFU 淘汰 |
| volatile-lfu | 设置了过期时间的 key 中 LFU 淘汰 |
| allkeys-random | 随机淘汰 |
| volatile-random | 设置过期时间的 key 中随机淘汰 |
| volatile-ttl | 淘汰即将过期的 key |

---

### 17. Big Key 问题？

**答案：**

```
问题：
├── 单个 key 的 value 过大
├── 集合类型元素过多
└── 影响：内存不均衡、阻塞、网络拥塞

发现方法：
├── redis-cli --bigkeys
├── MEMORY USAGE key
└── RDB 分析工具

解决方法：
├── 拆分：大集合拆成小集合
├── 压缩：使用压缩算法
├── 本地缓存：大 value 缓存在本地
└── 避免存储大对象
```

---

## 小结

本文档涵盖了 Redis 面试的高频考点：

- Redis 特点与快的原因
- 数据类型与应用场景
- 持久化机制
- 高可用方案
- 缓存三大问题
- 分布式锁实现
- 内存淘汰策略
