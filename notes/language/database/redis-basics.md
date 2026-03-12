# Redis 学习笔记

Redis（Remote Dictionary Server）是一个开源的、基于内存的高性能键值数据库。在深入 Redis 之前，我们需要先理解一个核心问题：**为什么我们需要 Redis？为什么不能只用 MySQL？**

## 为什么需要 Redis？

### 🤔 背景：传统数据库的瓶颈

假设我们有一个电商网站，用户访问商品详情页时：

```
用户请求 → 应用服务器 → MySQL 查询 → 返回数据
                        ↓
                   磁盘 I/O（慢！）
```

**问题分析：**

| 操作类型 | 耗时 | 原因 |
|---------|------|------|
| MySQL 查询（无索引） | 10-100ms | 需要扫描大量数据行 |
| MySQL 查询（有索引） | 1-10ms | 仍需要磁盘随机 I/O |
| Redis 查询 | 0.1-1ms | 纯内存操作，无磁盘 I/O |

```python
# 模拟：MySQL 查询延迟
import time

def get_product_from_mysql(product_id):
    """从 MySQL 查询商品信息"""
    start = time.time()
    # 模拟磁盘 I/O 延迟
    time.sleep(0.01)  # 10ms
    product = db.query(f"SELECT * FROM products WHERE id = {product_id}")
    print(f"MySQL 查询耗时: {(time.time() - start) * 1000:.2f}ms")
    return product

def get_product_from_redis(product_id):
    """从 Redis 查询商品信息"""
    start = time.time()
    product = redis.get(f"product:{product_id}")
    print(f"Redis 查询耗时: {(time.time() - start) * 1000:.2f}ms")
    return product

# 输出：
# MySQL 查询耗时: 10.23ms
# Redis 查询耗时: 0.15ms
```

### 📊 为什么内存比磁盘快？

```
CPU ←→ L1 Cache (1ns)
    ←→ L2 Cache (4ns)
    ←→ L3 Cache (12ns)
    ←→ 内存 (100ns)
    ←→ SSD (100,000ns = 0.1ms)
    ←→ HDD (10,000,000ns = 10ms)
```

**内存访问速度是 SSD 的 1000 倍，是 HDD 的 100000 倍！**

### 🎯 什么场景需要缓存？

| 场景 | 特点 | 是否适合 Redis |
|------|------|---------------|
| 热点数据访问 | 高频读取，低频更新 | ✅ 非常适合 |
| 会话存储 | 需要快速访问，可容忍丢失 | ✅ 非常适合 |
| 排行榜 | 需要实时排序 | ✅ 利用 Sorted Set |
| 计数器 | 高频增减操作 | ✅ 利用原子操作 |
| 分布式锁 | 需要跨进程同步 | ✅ 利用 SET NX |
| 复杂事务 | 需要 ACID 保证 | ❌ 用 MySQL |
| 大量数据存储 | 成本敏感 | ❌ 用 MySQL/磁盘 |
| 持久化要求高 | 不能丢数据 | ❌ 用 MySQL |

### 🔄 Redis 与 MySQL 的定位

```
┌─────────────────────────────────────────────────────┐
│                    应用架构                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Redis（缓存层）          MySQL（持久层）           │
│   ┌─────────────┐         ┌─────────────┐          │
│   │ 热点数据     │         │ 全量数据     │          │
│   │ 会话信息     │         │ 事务数据     │          │
│   │ 临时数据     │         │ 历史记录     │          │
│   │ 计数/排行    │         │ 关系数据     │          │
│   └─────────────┘         └─────────────┘          │
│        ↑                        ↑                   │
│    亚毫秒级                   毫秒级                 │
│    可丢失                     不丢失                 │
│    成本高                     成本低                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**结论：Redis 和 MySQL 不是替代关系，而是互补关系。Redis 负责快，MySQL 负责稳。**

---

## Redis 7.x 新特性

### 🚀 核心改进概览

Redis 7.x 是 Redis 历史上最重要的版本之一，带来了以下关键改进：

| 特性 | 解决的问题 | 说明 |
|------|-----------|------|
| **Function** | Lua 脚本管理复杂 | 新函数系统，支持持久化和复制 |
| **多部分 AOF** | AOF 重写阻塞 | 拆分为基础文件 + 增量文件 |
| **ACL v2** | 权限管理不够细粒度 | 支持更细粒度的权限管理 |
| **Sharded Pub/Sub** | 集群环境 Pub/Sub 性能差 | 分片发布订阅 |
| **Client Eviction** | 内存压力大时崩溃 | 客户端驱逐策略 |

### 📝 新增命令示例

```bash
# 7.x 新增的列表操作命令
# LMOVE：原子性地将元素从一个列表移动到另一个列表
LMOVE source destination LEFT RIGHT   # 从左边弹出，右边插入
BLMOVE source destination LEFT RIGHT 0  # 阻塞版本

# Set 操作增强
SINTERCARD 2 set1 set2         # 返回交集元素数量（不返回具体元素）
SMISMEMBER set1 a b c          # 批量检查多个元素是否存在

# Sorted Set 增强
ZMPOP 1 zset MIN COUNT 5       # 批量弹出分数最低的5个元素
BZMPOP 0 1 zset MAX COUNT 10   # 阻塞版本，弹出分数最高的10个
ZINTERCARD 2 zset1 zset2       # 返回交集元素数量
```

---

## 核心数据结构与底层实现

理解 Redis 数据结构的底层实现，是写出高性能代码的关键。**为什么 Redis 要这样设计？每个设计选择背后都有深刻的原因。**

### 📌 String（字符串）

#### 为什么需要 SDS 而不是 C 字符串？

**问题：C 字符串的缺陷**

```c
// C 字符串
char* str = "hello";

// 问题1：获取长度需要 O(N) 遍历
int len = strlen(str);  // 需要遍历到 '\0'

// 问题2：容易缓冲区溢出
char buf[10] = "hello";
strcat(buf, " world");  // 溢出！buf 只有 10 字节

// 问题3：每次修改都需要内存重分配
strcat(str, " world");  // 需要重新分配内存
```

**解决：SDS（Simple Dynamic String）**

```c
// Redis SDS 结构
struct sdshdr {
    int len;      // 已使用长度：O(1) 获取
    int free;     // 剩余可用空间：防止溢出
    char buf[];   // 字节数组
};
```

**为什么 SDS 更好？**

| 特性 | C 字符串 | SDS | 为什么重要？ |
|------|---------|-----|-------------|
| 获取长度 | O(N) | O(1) | 高频操作，性能关键 |
| 缓冲区溢出 | 可能溢出 | 自动扩展 | 安全性 |
| 内存重分配 | 每次都需要 | 预分配/惰性释放 | 性能优化 |
| 二进制安全 | 遇到 `\0` 截止 | 任意数据 | 支持图片等二进制数据 |

```python
# Redis String 操作
import redis
r = redis.Redis()

# 基本操作
r.set('user:1001', '张三')           # 存储
r.get('user:1001')                   # 获取

# 为什么推荐带过期时间的设置？
# 因为很多数据是临时的，不设置过期会导致内存泄漏
r.setex('session:abc', 3600, 'data') # 1小时后自动删除

# 分布式锁：NX = 不存在才设置，EX = 过期时间
# 为什么需要过期？防止死锁（进程崩溃后锁无法释放）
acquired = r.set('lock:order:123', 'uuid', nx=True, ex=30)
if acquired:
    # 获得锁，执行业务
    pass
```

### 📌 Hash（哈希）

#### 为什么用 Hash 存储对象而不是 JSON String？

**问题：JSON String 存储对象的缺陷**

```python
# ❌ 不推荐：用 String 存储 JSON
user_data = {'name': '张三', 'age': 25, 'email': 'zhang@example.com'}
r.set('user:1001', json.dumps(user_data))

# 问题1：只更新一个字段也需要读取整个对象、修改、写回
data = json.loads(r.get('user:1001'))
data['age'] = 26  # 只改年龄
r.set('user:1001', json.dumps(data))  # 全量写入

# 问题2：无法对单个字段做原子操作（如年龄+1）
```

**解决：用 Hash 存储**

```python
# ✅ 推荐：用 Hash 存储对象
r.hset('user:1001', mapping={
    'name': '张三',
    'age': 25,
    'email': 'zhang@example.com'
})

# 优势1：可以只更新单个字段
r.hset('user:1001', 'age', 26)

# 优势2：可以对单个字段做原子操作
r.hincrby('user:1001', 'age', 1)  # 年龄+1，原子操作

# 优势3：只获取需要的字段，减少网络传输
r.hget('user:1001', 'name')  # 只获取 name
```

#### Hash 底层实现：为什么有两种编码？

```
Hash 编码选择逻辑：
                    
数据量少？ ──── Yes ───→ ListPack（紧凑）
    │                    - 连续内存
    │                    - 内存效率高
    No                   - 查找 O(N)
    ↓
HashTable（高效）
    - 哈希表
    - 查找 O(1)
    - 内存开销大
```

**为什么这样设计？**

| 场景 | 编码 | 原因 |
|------|------|------|
| 字段少（<512）、值小（<64字节） | ListPack | 内存紧凑，省空间 |
| 字段多或值大 | HashTable | 查找效率高 |

```bash
# 配置阈值
hash-max-listpack-entries 512   # 字段数超过 512 转换为 HashTable
hash-max-listpack-value 64      # 值超过 64 字节转换
```

### 📌 List（列表）

#### 为什么 List 底层用 QuickList？

**历史演变：**

```
Redis 3.2 之前：
    ZipList（压缩列表）- 内存紧凑，但连锁更新问题
    LinkedList（双向链表）- 内存碎片严重

Redis 3.2+：
    QuickList = LinkedList + ZipList
              = 多个 ZipList 组成的双向链表
```

**为什么 QuickList 是最佳选择？**

```
┌─────────────────────────────────────────────────────┐
│                    QuickList 结构                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ ZipList  │ ←→ │ ZipList  │ ←→ │ ZipList  │      │
│  │ [a,b,c]  │    │ [d,e,f]  │    │ [g,h,i]  │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       ↑                ↑                ↑          │
│   内存紧凑        内存紧凑          内存紧凑         │
│                                                     │
│  优点：                                              │
│  1. 兼顾链表的插入效率（两端插入 O(1)）               │
│  2. 兼顾 ZipList 的内存效率（连续内存）              │
│  3. 支持中间节点压缩（进一步节省内存）                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```python
# List 常见应用场景
import redis
r = redis.Redis()

# 场景1：消息队列（FIFO）
r.lpush('task:queue', 'task1')  # 左边进
r.lpush('task:queue', 'task2')
task = r.rpop('task:queue')     # 右边出

# 场景2：最新消息列表（只保留最新 N 条）
r.lpush('chat:1001:messages', '消息内容')
r.ltrim('chat:1001:messages', 0, 99)  # 只保留最新 100 条

# 场景3：阻塞队列（消费者等待新消息）
result = r.brpop('task:queue', timeout=5)  # 最多等5秒
```

### 📌 Set（集合）

#### Set 底层实现：为什么有两种编码？

```
Set 编码选择逻辑：
                    
全是整数？ ──── Yes ───→ IntSet（整数集合）
    │                    - 有序数组
    │                    - 二分查找 O(logN)
    No                   - 内存极紧凑
    ↓
HashTable（哈希表）
    - 查找 O(1)
    - 内存开销大
```

```python
# Set 常见应用场景
import redis
r = redis.Redis()

# 场景1：标签系统
r.sadd('user:1001:tags', '技术', '编程', 'AI')
r.sadd('user:1002:tags', '技术', '设计')

# 找共同标签（交集）
common = r.sinter('user:1001:tags', 'user:1002:tags')  # {'技术'}

# 场景2：社交关系
r.sadd('user:1001:following', 1002, 1003)  # 关注的人
r.sadd('user:1002:followers', 1001)        # 粉丝

# 场景3：抽奖系统
r.sadd('lottery:users', 'user1', 'user2', 'user3')
winner = r.spop('lottery:users')  # 随机弹出一个人
```

### 📌 Sorted Set（有序集合）

#### 为什么用跳表而不是红黑树？

这是一个经典的面试题。Sorted Set 底层使用 **跳表（Skip List）+ 哈希表**，为什么不选择红黑树？

**对比分析：**

| 特性 | 跳表 | 红黑树 | 为什么 Redis 选择跳表？ |
|------|------|--------|------------------------|
| 查找单元素 | O(logN) | O(logN) | 相同 |
| 范围查询 | O(logN + M) | O(logN + M) 但复杂 | **跳表更简单高效** |
| 实现复杂度 | 简单 | 复杂 | **跳表更容易调试维护** |
| 内存占用 | 指针多 | 指针少 | 红黑树略优 |
| 并发友好 | 好 | 差 | 跳表更适合并发 |

**关键原因：范围查询效率**

```python
# 范围查询：获取分数在 100-200 之间的用户
# 跳表：找到起点后顺序遍历即可
# 红黑树：需要复杂的中序遍历

# Sorted Set 范围查询
r.zadd('leaderboard', {'player1': 100, 'player2': 150, 'player3': 200})
r.zrangebyscore('leaderboard', 100, 200)  # 范围查询，高效！
```

**跳表原理图：**

```
查找元素 50 的过程：

Level 3:        10 ------------------------> 100
                  ↓                          ↓
Level 2:        10 ---------> 50 ---------> 100
                  ↓            ↓             ↓
Level 1:        10 --> 30 --> 50 --> 70 --> 100

查找路径：10 → 10 → 50（找到！）
时间复杂度：O(logN)
```

```python
# Sorted Set 常见应用场景
import redis
r = redis.Redis()

# 场景1：游戏排行榜
r.zadd('game:leaderboard', {'player1': 1000, 'player2': 1500, 'player3': 1200})

# 获取 Top 10
top10 = r.zrevrange('game:leaderboard', 0, 9, withscores=True)

# 获取玩家排名（从高到低）
rank = r.zrevrank('game:leaderboard', 'player1')  # 返回索引

# 场景2：延时任务队列（分数是执行时间戳）
import time
r.zadd('delayed:tasks', {
    'task1': time.time() + 60,    # 60秒后执行
    'task2': time.time() + 120,   # 120秒后执行
})

# 获取已到期的任务
now = time.time()
ready_tasks = r.zrangebyscore('delayed:tasks', 0, now)
```

### 📌 数据结构对比总结

| 数据结构 | 底层实现 | 时间复杂度 | 适用场景 |
|---------|---------|-----------|---------|
| String | SDS | O(1) | 缓存、计数器、分布式锁 |
| Hash | ListPack / HashTable | O(1) | 对象存储、购物车 |
| List | QuickList | 两端 O(1) | 消息队列、最新列表 |
| Set | IntSet / HashTable | O(1) | 标签、社交关系、抽奖 |
| Sorted Set | SkipList + HashTable | O(logN) | 排行榜、延时队列 |

---

## 过期策略

### 🤔 为什么需要过期策略？

**问题背景：**

```python
# 假设我们存储了 100 万个带过期时间的 key
for i in range(1000000):
    r.setex(f'session:{i}', 3600, f'data{i}')  # 1小时后过期

# 问题：1小时后，这 100 万个 key 需要被删除
# 如何高效地删除过期的 key？
```

**两种朴素方案的问题：**

| 方案 | 描述 | 问题 |
|------|------|------|
| 定时器删除 | 每个 key 设置一个定时器，到期删除 | 创建 100 万个定时器，CPU 消耗巨大 |
| 惰性删除 | 只有访问时才检查是否过期 | 过期 key 永不被访问则永不删除，内存泄漏 |

### 📋 Redis 的解决方案：定期删除 + 惰性删除

```
┌─────────────────────────────────────────────────────┐
│              Redis 过期策略                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│   惰性删除（Lazy Expiration）                        │
│   ┌─────────────────────────────────────────┐       │
│   │ 访问 key 时检查是否过期                   │       │
│   │ → 过期则删除，返回 null                   │       │
│   │ → 未过期则返回值                          │       │
│   │                                          │       │
│   │ 优点：CPU 友好，只处理被访问的 key         │       │
│   │ 缺点：过期但不被访问的 key 会占用内存      │       │
│   └─────────────────────────────────────────┘       │
│                        +                            │
│   定期删除（Periodic Expiration）                    │
│   ┌─────────────────────────────────────────┐       │
│   │ 每隔一段时间随机抽查部分 key              │       │
│   │ → 发现过期的就删除                        │       │
│   │                                          │       │
│   │ 抽样策略：                                │       │
│   │ 1. 每秒执行 10 次（可配置）               │       │
│   │ 2. 每次随机检查 20 个 key                 │       │
│   │ 3. 如果发现超过 25% 过期，继续检查        │       │
│   └─────────────────────────────────────────┘       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**为什么是这种组合？**

```
CPU 消耗 ◄──────────────────────────────────► 内存效率

定时器删除     定期删除+惰性删除     惰性删除
（太高）          （平衡）          （内存泄漏）
```

```python
# 配置参数
# redis.conf

# 每秒执行过期检查的频率（默认 10 次）
hz 10

# 增强过期清理（会消耗更多 CPU）
activerehashing yes
```

---

## 内存淘汰策略

### 🤔 为什么需要内存淘汰？

**问题背景：**

```
Redis 是基于内存的数据库，内存有限。
当内存用完时，怎么办？

选项1：报错，拒绝写入 ❌ → 服务不可用
选项2：删除一些数据，腾出空间 ✅ → 内存淘汰策略
```

### 📋 8 种内存淘汰策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **noeviction** | 不淘汰，内存满时返回错误 | 数据不能丢失，必须有运维监控 |
| **allkeys-lru** | 所有 key 中淘汰最近最少使用的 | 缓存场景，热点数据保留 |
| **allkeys-lfu** | 所有 key 中淘汰最少使用的 | 缓存场景，访问频率更重要 |
| **allkeys-random** | 所有 key 中随机淘汰 | 无访问模式，简单随机 |
| **volatile-lru** | 设置了过期时间的 key 中淘汰 LRU | 有些数据不能淘汰 |
| **volatile-lfu** | 设置了过期时间的 key 中淘汰 LFU | 有些数据不能淘汰 |
| **volatile-random** | 设置了过期时间的 key 中随机淘汰 | 简单随机 |
| **volatile-ttl** | 淘汰即将过期的 key（TTL 小的） | 业务有明确的过期需求 |

### 🎯 如何选择淘汰策略？

```
┌─────────────────────────────────────────────────────┐
│              淘汰策略选择决策树                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  数据可以全部淘汰吗？                                │
│      │                                              │
│      ├─ Yes → 有明显的热点数据吗？                   │
│      │           │                                  │
│      │           ├─ Yes → allkeys-lru               │
│      │           │                                   │
│      │           └─ No → allkeys-random             │
│      │                                              │
│      └─ No → 有明确过期时间吗？                      │
│                │                                    │
│                ├─ Yes → volatile-lru / volatile-ttl │
│                │                                    │
│                └─ No → noeviction（需监控）          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```bash
# redis.conf 配置
maxmemory 4gb                      # 最大内存限制
maxmemory-policy allkeys-lru       # 淘汰策略

# LRU 检查样本数（越大越精确，但消耗 CPU）
maxmemory-samples 5
```

```python
# 监控内存使用
import redis
r = redis.Redis()

info = r.info('memory')
print(f"已用内存: {info['used_memory_human']}")
print(f"最大内存: {info['maxmemory_human']}")
print(f"内存碎片率: {info['mem_fragmentation_ratio']}")

# 查看淘汰统计
info = r.info('stats')
print(f"淘汰次数: {info['evicted_keys']}")
```

---

## 持久化策略

### 🤔 为什么需要持久化？

**问题：Redis 是基于内存的，重启后数据怎么办？**

```
场景1：服务器重启
    Redis 进程退出 → 内存数据全部丢失 → 服务恢复后缓存全部失效

场景2：服务器宕机
    突然断电 → 内存数据全部丢失 → 数据能否恢复？

持久化的目的：将内存数据保存到磁盘，重启后可以恢复。
```

### 💾 RDB（Redis Database）

**原理：快照**

```
RDB 工作流程：

1. Redis 调用 fork() 创建子进程
2. 子进程将内存数据写入临时文件
3. 写入完成后，替换旧的 RDB 文件

为什么用子进程？
- 父进程可以继续处理请求（不阻塞）
- 子进程共享父进程的内存页（写时复制）
```

**配置：**

```bash
# redis.conf

# 自动触发条件（满足任一即触发）
save 900 1      # 900秒内有1次修改
save 300 10     # 300秒内有10次修改
save 60 10000   # 60秒内有10000次修改

# 文件配置
dbfilename dump.rdb
dir /var/lib/redis

# 压缩（节省空间，消耗 CPU）
rdbcompression yes
rdbchecksum yes
```

**优缺点分析：**

| 优点 | 缺点 |
|------|------|
| 文件紧凑，适合备份 | 可能丢失快照后的数据 |
| 恢复速度快（直接加载） | Fork 大数据集时可能阻塞 |
| 对性能影响小（子进程） | 不适合实时持久化 |

**为什么 RDB 可能丢失数据？**

```
时间线：
t0: 执行 BGSAVE，开始创建快照
t1: 写入 key1, key2, key3
t2: 快照完成，保存到磁盘
t3: 写入 key4, key5
t4: 服务器宕机！

恢复后：key1, key2, key3 存在
        key4, key5 丢失！（不在快照中）
```

### 💾 AOF（Append Only File）

**原理：日志追加**

```
AOF 工作流程：

1. 每个写命令追加到 AOF 文件
2. AOF 文件越来越大
3. 需要定期重写（压缩）

例如：
原始命令：
SET key1 value1
SET key1 value2
SET key1 value3

重写后：
SET key1 value3  # 只保留最终结果
```

**配置：**

```bash
# redis.conf
appendonly yes
appendfilename "appendonly.aof"

# 同步策略（关键！）
appendfsync always     # 每次写入都同步（最安全，最慢）
appendfsync everysec   # 每秒同步一次（推荐，最多丢1秒）
appendfsync no         # 由操作系统决定（最快，可能丢30秒）

# AOF 重写触发条件
auto-aof-rewrite-percentage 100  # 文件大小翻倍时重写
auto-aof-rewrite-min-size 64mb   # 最小重写大小
```

**三种同步策略对比：**

| 策略 | 数据安全性 | 性能 | 说明 |
|------|-----------|------|------|
| always | 最高 | 最慢 | 每次写入都 fsync |
| everysec | 高 | 中等 | 最多丢失 1 秒数据 |
| no | 低 | 最快 | 依赖 OS，可能丢失 30 秒 |

**优缺点分析：**

| 优点 | 缺点 |
|------|------|
| 数据更安全（最多丢1秒） | 文件较大 |
| 可读性好，便于分析 | 恢复速度较慢 |
| 支持重写压缩 | 写入性能略低于 RDB |

### 💾 混合持久化（Redis 7.x 优化）

**为什么需要混合持久化？**

```
问题：
- RDB 恢复快，但可能丢数据
- AOF 数据安全，但恢复慢

解决：
- 结合两者的优点
- 先用 RDB 快速恢复基础数据
- 再用 AOF 增量补充最新数据
```

**Redis 7.x 多部分 AOF：**

```
appendonlydir/
├── appendonly.aof.manifest       # 清单文件
├── appendonly.aof.1.base.rdb     # 基础文件（RDB 格式）
└── appendonly.aof.1.incr.aof     # 增量文件（AOF 格式）
```

**配置：**

```bash
# redis.conf
appendonly yes
appenddirname "appendonlydir"
appendfilename "appendonly.aof"
aof-use-rdb-preamble yes   # 开启混合持久化
```

**恢复流程：**

```
1. 加载 appendonly.aof.1.base.rdb（RDB 格式，快速）
2. 重放 appendonly.aof.1.incr.aof（AOF 格式，增量）
3. 恢复完成！
```

### 📊 持久化策略选择

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 纯缓存（可丢失） | 不持久化 | 性能最高 |
| 允许分钟级丢失 | RDB | 恢复快 |
| 允许秒级丢失 | AOF (everysec) | 数据安全 |
| 要求高可用 | 混合持久化 | 兼顾恢复速度和数据安全 |

---

## 高可用架构

### 🔄 主从复制

#### 为什么需要主从复制？

```
问题：单点 Redis 的瓶颈

1. 读请求量太大，单机 QPS 不够（Redis 约 10 万 QPS）
2. 单机内存有限，无法存储更多数据
3. 单机故障，服务不可用

解决：主从复制
- 主节点：处理写请求
- 从节点：处理读请求（读写分离）
```

**主从复制原理：**

```
┌─────────────────────────────────────────────────────┐
│                  主从复制流程                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Master                                            │
│   ┌─────────┐                                       │
│   │ 写请求   │                                       │
│   │ 数据变更 │                                       │
│   └────┬────┘                                       │
│        │                                            │
│        │ 同步数据                                    │
│        ▼                                            │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│   │ Slave 1 │    │ Slave 2 │    │ Slave 3 │        │
│   │ 读请求   │    │ 读请求   │    │ 读请求   │        │
│   └─────────┘    └─────────┘    └─────────┘        │
│                                                     │
│   读写分离：写走 Master，读走 Slave                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**配置：**

```bash
# 从节点配置（redis.conf）
replicaof 192.168.1.100 6379   # 指定主节点
replica-read-only yes          # 从节点只读

# 无盘复制（直接网络传输，不写磁盘）
repl-diskless-sync yes
```

**Python 读写分离示例：**

```python
from redis import Redis

# 写操作 - 连接主节点
master = Redis(host='192.168.1.100', port=6379)
master.set('key', 'value')  # 写

# 读操作 - 连接从节点
slave = Redis(host='192.168.1.101', port=6379)
value = slave.get('key')    # 读
```

#### 主从复制的问题

```
问题：Master 故障怎么办？

Master 宕机 → 无法写入 → 需要手动切换到 Slave
                    ↓
              人为干预，效率低
                    ↓
              需要自动故障转移！
```

### 🛡️ 哨兵模式（Sentinel）

#### 为什么需要哨兵？

```
主从复制的问题：
1. Master 故障需要人工介入
2. 客户端需要知道新的 Master 地址
3. 故障转移过程不可控

哨兵的作用：
1. 监控：持续检查 Master 和 Slave 状态
2. 通知：发现问题通知管理员
3. 自动故障转移：自动将 Slave 提升为 Master
```

**哨兵架构：**

```
┌─────────────────────────────────────────────────────┐
│                   哨兵架构                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │ Sentinel │  │ Sentinel │  │ Sentinel │         │
│   │    1     │  │    2     │  │    3     │         │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│        │              │              │              │
│        └──────────────┼──────────────┘              │
│                       │                             │
│                       ▼                             │
│                  ┌─────────┐                        │
│                  │ Master  │                        │
│                  └────┬────┘                        │
│                       │                             │
│              ┌────────┼────────┐                    │
│              ▼        ▼        ▼                    │
│          ┌───────┐ ┌───────┐ ┌───────┐             │
│          │Slave 1│ │Slave 2│ │Slave 3│             │
│          └───────┘ └───────┘ └───────┘             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**哨兵配置：**

```bash
# sentinel.conf

# 监控主节点，2 表示需要 2 个哨兵同意才能进行故障转移
sentinel monitor mymaster 192.168.1.100 6379 2

# 主观下线判定时间（多久无响应认为下线）
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时时间
sentinel failover-timeout mymaster 180000

# 同时可以对新的 Master 同步的 Slave 数量
sentinel parallel-syncs mymaster 1
```

**故障转移流程：**

```
1. Sentinel 检测到 Master 下线（主观下线）
2. 多个 Sentinel 确认（客观下线）
3. Sentinel 选举 Leader（负责故障转移）
4. 选举新的 Master（选择数据最新的 Slave）
5. 其他 Slave 复制新 Master
6. 客户端连接新 Master
7. 原 Master 恢复后变成 Slave
```

**Python 使用哨兵：**

```python
from redis.sentinel import Sentinel

# 连接哨兵
sentinel = Sentinel([
    ('192.168.1.100', 26379),
    ('192.168.1.101', 26379),
    ('192.168.1.102', 26379)
], socket_timeout=0.1)

# 获取 Master 连接（写操作）
master = sentinel.master_for('mymaster', socket_timeout=0.1)
master.set('key', 'value')

# 获取 Slave 连接（读操作）
slave = sentinel.slave_for('mymaster', socket_timeout=0.1)
value = slave.get('key')

# 自动故障转移后，客户端会自动获取新 Master 地址
```

### 🔀 Redis Cluster 集群

#### 为什么需要 Cluster？

```
哨兵模式的问题：
1. 主从复制是全量复制，每个节点都有全量数据
2. 单机内存有限，无法水平扩展存储容量
3. 写操作仍然只能由 Master 处理

Cluster 的优势：
1. 数据分片：数据分布在多个节点，支持海量数据
2. 水平扩展：添加节点即可扩展容量
3. 高可用：每个分片都有主从，自动故障转移
```

**Cluster 架构：**

```
┌─────────────────────────────────────────────────────┐
│                Redis Cluster 架构                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│   槽位分配：16384 个槽位分布在各 Master              │
│                                                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│   │ Master1 │    │ Master2 │    │ Master3 │        │
│   │ 0-5461  │    │5462-10922│   │10923-16383│       │
│   └────┬────┘    └────┬────┘    └────┬────┘        │
│        │              │              │              │
│        ▼              ▼              ▼              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│   │ Replica │    │ Replica │    │ Replica │        │
│   └─────────┘    └─────────┘    └─────────┘        │
│                                                     │
│   特点：                                             │
│   - 数据分散在多个节点，无中心                       │
│   - 每个节点负责一部分槽位                          │
│   - 客户端可以连接任意节点                          │
│   - 自动重定向到正确的节点                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**槽位与数据分布：**

```
CRC16(key) % 16384 = 槽位号

例如：
CRC16('user:1001') % 16384 = 8192
→ key 'user:1001' 存储在负责槽位 8192 的节点（Master2）
```

**配置：**

```bash
# redis.conf（每个节点）
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
cluster-announce-ip 192.168.1.100
cluster-announce-port 6379
cluster-announce-bus-port 16379
```

**创建集群：**

```bash
# 创建 3 主 3 从的集群
redis-cli --cluster create \
  192.168.1.101:6379 192.168.1.102:6379 192.168.1.103:6379 \
  192.168.1.104:6379 192.168.1.105:6379 192.168.1.106:6379 \
  --cluster-replicas 1
```

**Python 连接集群：**

```python
from redis.cluster import RedisCluster

# 连接集群（只需指定一个节点）
rc = RedisCluster(
    host='192.168.1.101',
    port=6379,
    max_connections=100,
    decode_responses=True
)

# 正常使用，集群自动路由
rc.set('user:1001', '张三')
rc.get('user:1001')

# Hash Tags：让相关 key 在同一节点
# 使用 {} 包裹相同部分
rc.set('user:{1001}:profile', '...')
rc.set('user:{1001}:orders', '...')
# 这两个 key 会在同一节点，支持事务
```

### 📊 高可用架构对比

| 特性 | 主从复制 | 哨兵模式 | Cluster |
|------|---------|---------|---------|
| 数据分片 | ❌ | ❌ | ✅ |
| 自动故障转移 | ❌ | ✅ | ✅ |
| 水平扩展 | ❌ | ❌ | ✅ |
| 部署复杂度 | 低 | 中 | 高 |
| 适用数据量 | 小 | 中 | 大 |
| 客户端支持 | 简单 | 需要哨兵支持 | 需要集群支持 |

---

## 缓存设计模式

### 📖 Cache-Aside Pattern（旁路缓存）

**最常用的缓存模式，应用负责维护缓存。**

```
┌─────────────────────────────────────────────────────┐
│              Cache-Aside 读流程                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Client ──→ 查缓存 ──→ 命中？── Yes ──→ 返回       │
│                 │                                   │
│                 No                                  │
│                 │                                   │
│                 ▼                                   │
│              查数据库                                │
│                 │                                   │
│                 ▼                                   │
│              写缓存                                  │
│                 │                                   │
│                 ▼                                   │
│               返回                                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```python
def get_user(user_id):
    """Cache-Aside 读操作"""
    cache_key = f'user:{user_id}'
    
    # 1. 先查缓存
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 2. 缓存未命中，查数据库
    user = db.query(User).get(user_id)
    if user:
        # 3. 写入缓存（设置过期时间）
        r.setex(cache_key, 3600, json.dumps(user.to_dict()))
    
    return user

def update_user(user_id, **data):
    """Cache-Aside 写操作"""
    # 1. 更新数据库
    db.query(User).filter_by(id=user_id).update(data)
    db.commit()
    
    # 2. 删除缓存（为什么删除而不是更新？）
    r.delete(f'user:{user_id}')
```

**为什么删除缓存而不是更新缓存？**

```
场景：并发更新

方案1：更新缓存
Thread A: 更新 DB = 1 → 更新缓存 = 1
Thread B: 更新 DB = 2 → 更新缓存 = 2

如果执行顺序是：
A 更新 DB = 1
B 更新 DB = 2
B 更新缓存 = 2
A 更新缓存 = 1  ← 缓存是 1，但 DB 是 2！数据不一致！

方案2：删除缓存
无论谁先删，下次读取时都会从 DB 加载最新值
```

### 📖 Read-Through / Write-Through

**缓存层负责数据读写，应用只与缓存交互。**

```python
class ReadThroughCache:
    """Read-Through：缓存层负责加载数据"""
    
    def __init__(self, redis_client, db_loader, ttl=3600):
        self.r = redis_client
        self.loader = db_loader
        self.ttl = ttl
    
    def get(self, key):
        # 缓存层负责加载数据
        cached = self.r.get(key)
        if cached is not None:
            return json.loads(cached)
        
        # 从数据库加载并写入缓存
        data = self.loader(key)
        if data:
            self.r.setex(key, self.ttl, json.dumps(data))
        return data


class WriteThroughCache:
    """Write-Through：写入时同步更新缓存和数据库"""
    
    def __init__(self, redis_client, db_writer, ttl=3600):
        self.r = redis_client
        self.writer = db_writer
        self.ttl = ttl
    
    def set(self, key, value):
        # 先写数据库
        self.writer(key, value)
        # 再更新缓存
        self.r.setex(key, self.ttl, json.dumps(value))
```

### 📖 Write-Behind（异步写入）

**写入时只更新缓存，异步写入数据库。**

```python
import threading
import queue

class WriteBehindCache:
    """Write-Behind：异步写入数据库"""
    
    def __init__(self, redis_client, db_writer, ttl=3600):
        self.r = redis_client
        self.writer = db_writer
        self.ttl = ttl
        self.write_queue = queue.Queue()
        
        # 启动异步写入线程
        self._start_writer_thread()
    
    def set(self, key, value):
        # 立即更新缓存（快速响应）
        self.r.setex(key, self.ttl, json.dumps(value))
        # 异步写入数据库
        self.write_queue.put(('set', key, value))
    
    def _start_writer_thread(self):
        def worker():
            while True:
                op, key, value = self.write_queue.get()
                try:
                    if op == 'set':
                        self.writer(key, value)
                except Exception as e:
                    print(f"Write error: {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
```

**对比：**

| 模式 | 读流程 | 写流程 | 一致性 | 复杂度 |
|------|-------|--------|--------|--------|
| Cache-Aside | 先缓存后 DB | 先 DB 后删缓存 | 最终一致 | 低 |
| Read-Through | 缓存层加载 | 应用写 DB | 最终一致 | 中 |
| Write-Through | 缓存层加载 | 缓存层写 DB | 强一致 | 中 |
| Write-Behind | 缓存层加载 | 异步写 DB | 弱一致 | 高 |

---

## 缓存三大问题

### 🚨 缓存穿透（Cache Penetration）

#### 什么是缓存穿透？

```
定义：查询一个根本不存在的数据，缓存和数据库都没有。

请求流程：
Client ──→ Redis（没有）──→ MySQL（也没有）──→ 返回空

问题：
如果有恶意攻击，大量请求不存在的 key：
1. 缓存永远不会有数据（因为没有数据可缓存）
2. 所有请求都打到数据库
3. 数据库压力过大，可能崩溃
```

#### 解决方案

**方案1：缓存空值**

```python
def get_user(user_id):
    cache_key = f'user:{user_id}'
    
    cached = r.get(cache_key)
    if cached is not None:
        # 可能是真实数据，也可能是空值标记
        if cached == 'NULL':
            return None
        return json.loads(cached)
    
    # 查数据库
    user = db.query(User).get(user_id)
    
    if user:
        r.setex(cache_key, 3600, json.dumps(user.to_dict()))
    else:
        # 缓存空值，防止穿透
        # 注意：过期时间要短一些
        r.setex(cache_key, 60, 'NULL')
    
    return user
```

**方案2：布隆过滤器**

```
什么是布隆过滤器？
- 一种概率型数据结构
- 可以判断元素"可能存在"或"一定不存在"
- 空间效率极高，适合海量数据

原理：
1. 使用多个哈希函数
2. 将元素映射到位数组的多个位置
3. 查询时检查这些位置是否都为 1

为什么适合解决缓存穿透？
- 可以快速判断 key 是否可能存在
- 如果布隆过滤器说不存在，那就一定不存在
- 直接返回，不需要查缓存和数据库
```

```python
# 使用 RedisBloom 模块（或 Python 实现）
from pybloom_live import ScalableBloomFilter

# 初始化布隆过滤器
bf = ScalableBloomFilter(initial_capacity=1000000, error_rate=0.001)

# 启动时加载所有存在的 key
for user_id in db.query(User.id).all():
    bf.add(user_id)

def get_user_with_bloom(user_id):
    # 先用布隆过滤器判断
    if user_id not in bf:
        # 一定不存在，直接返回
        return None
    
    # 可能存在，走正常流程
    return get_user(user_id)
```

### 🚨 缓存击穿（Cache Breakdown）

#### 什么是缓存击穿？

```
定义：某个热点 key 过期瞬间，大量并发请求同时访问。

场景：
- 热门商品信息
- 热门文章
- 秒杀商品

问题：
key 过期瞬间：
Thread 1: 查缓存没有 → 查数据库 → 写缓存
Thread 2: 查缓存没有 → 查数据库 → 写缓存
Thread 3: 查缓存没有 → 查数据库 → 写缓存
...
所有线程同时查数据库，数据库瞬间压力巨大！
```

#### 解决方案

**方案1：互斥锁**

```python
import time

def get_user_with_lock(user_id):
    cache_key = f'user:{user_id}'
    lock_key = f'lock:{cache_key}'
    
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 尝试获取锁
    lock_acquired = r.set(lock_key, '1', nx=True, ex=10)
    
    if lock_acquired:
        try:
            # 获得锁，查数据库
            user = db.query(User).get(user_id)
            if user:
                r.setex(cache_key, 3600, json.dumps(user.to_dict()))
            return user
        finally:
            r.delete(lock_key)
    else:
        # 未获得锁，等待后重试
        time.sleep(0.05)
        return get_user_with_lock(user_id)
```

**方案2：逻辑过期（永不过期）**

```python
def get_user_logical_expire(user_id):
    cache_key = f'user:{user_id}'
    
    cached = r.get(cache_key)
    if cached:
        data = json.loads(cached)
        # 检查逻辑过期时间
        if data['expire_time'] > time.time():
            return data['value']
        else:
            # 已过期，异步刷新
            threading.Thread(
                target=refresh_cache,
                args=(user_id,)
            ).start()
            # 先返回旧数据
            return data['value']
    
    # 缓存不存在，查数据库
    return refresh_cache(user_id)

def refresh_cache(user_id):
    user = db.query(User).get(user_id)
    if user:
        data = {
            'value': user.to_dict(),
            'expire_time': time.time() + 3600
        }
        r.set(f'user:{user_id}', json.dumps(data))
```

### 🚨 缓存雪崩（Cache Avalanche）

#### 什么是缓存雪崩？

```
定义：大量缓存 key 在同一时间集中过期，或 Redis 宕机。

场景1：批量设置相同过期时间
- 缓存预热时，所有数据设置 1 小时过期
- 1 小时后，所有 key 同时过期
- 所有请求打到数据库

场景2：Redis 宕机
- 所有缓存失效
- 所有请求打到数据库
```

#### 解决方案

**方案1：过期时间加随机值**

```python
import random

def set_cache_with_random_ttl(key, value, base_ttl=3600):
    # 基础过期时间 + 随机值（0-20%）
    random_offset = random.randint(0, int(base_ttl * 0.2))
    ttl = base_ttl + random_offset
    r.setex(key, ttl, json.dumps(value))

# 批量预热时
for product in products:
    set_cache_with_random_ttl(
        f'product:{product.id}',
        product.to_dict(),
        base_ttl=3600
    )
```

**方案2：多级缓存**

```python
class MultiLevelCache:
    """多级缓存：本地缓存 + Redis"""
    
    def __init__(self, redis_client, db_loader, local_ttl=60, redis_ttl=3600):
        self.r = redis_client
        self.loader = db_loader
        self.local_ttl = local_ttl
        self.redis_ttl = redis_ttl
        self.local_cache = {}  # 简单实现，生产环境用 LRU Cache
        self.local_expire = {}
    
    def get(self, key):
        now = time.time()
        
        # L1: 本地缓存
        if key in self.local_cache:
            if now < self.local_expire.get(key, 0):
                return self.local_cache[key]
        
        # L2: Redis
        cached = self.r.get(key)
        if cached:
            value = json.loads(cached)
            self._set_local(key, value)
            return value
        
        # L3: 数据库
        value = self.loader(key)
        if value:
            self._set_redis(key, value)
            self._set_local(key, value)
        return value
    
    def _set_local(self, key, value):
        self.local_cache[key] = value
        self.local_expire[key] = time.time() + self.local_ttl
    
    def _set_redis(self, key, value):
        self.r.setex(key, self.redis_ttl, json.dumps(value))
```

**方案3：熔断降级**

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def get_product(product_id):
    """熔断保护：连续失败后降级"""
    cache_key = f'product:{product_id}'
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    product = db.query(Product).get(product_id)
    if product:
        r.setex(cache_key, 3600, json.dumps(product.to_dict()))
    return product

# 降级处理
def get_product_fallback(product_id):
    # 返回默认值或静态数据
    return {'id': product_id, 'name': '商品加载中...', 'price': 0}
```

### 📊 缓存三大问题对比

| 问题 | 定义 | 根本原因 | 解决方案 |
|------|------|---------|---------|
| 缓存穿透 | 查询不存在的数据 | 恶意请求 | 缓存空值、布隆过滤器 |
| 缓存击穿 | 热点 key 过期 | 并发访问 | 互斥锁、逻辑过期 |
| 缓存雪崩 | 大量 key 同时过期 | 过期时间集中 | 随机过期、多级缓存、熔断 |

---

## 分布式锁

### 🤔 为什么需要分布式锁？

```
场景：秒杀系统扣库存

单机环境：
Thread 1: 读取库存 = 10 → 扣减 → 库存 = 9
Thread 2: 读取库存 = 10 → 扣减 → 库存 = 9  ← 并发问题！
                    ↑
              应该是 8

分布式环境：
Server 1: 读取库存 = 10 → 扣减 → 库存 = 9
Server 2: 读取库存 = 10 → 扣减 → 库存 = 9  ← 跨进程并发问题！

解决：分布式锁
- 跨进程、跨机器的锁
- 保证同一时刻只有一个客户端可以执行
```

### 🔒 基于 Redis 实现分布式锁

**基础版本：**

```python
import uuid
import time

class DistributedLock:
    """分布式锁基础实现"""
    
    def __init__(self, redis_client, lock_name, ttl=30):
        self.r = redis_client
        self.lock_name = f'lock:{lock_name}'
        self.ttl = ttl
        self.identifier = str(uuid.uuid4())  # 唯一标识
    
    def acquire(self, timeout=10):
        """获取锁"""
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            # SET key value NX EX ttl
            # NX: 不存在才设置
            # EX: 设置过期时间
            if self.r.set(self.lock_name, self.identifier, nx=True, ex=self.ttl):
                return True
            time.sleep(0.001)
        
        return False
    
    def release(self):
        """释放锁"""
        # 使用 Lua 脚本保证原子性
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        self.r.eval(script, 1, self.lock_name, self.identifier)


# 使用示例
lock = DistributedLock(r, 'seckill:product:1001', ttl=30)

if lock.acquire(timeout=5):
    try:
        # 执行业务逻辑
        stock = int(r.get('stock:product:1001'))
        if stock > 0:
            r.decr('stock:product:1001')
            print("秒杀成功！")
        else:
            print("库存不足！")
    finally:
        lock.release()
else:
    print("获取锁失败，请稍后重试")
```

**为什么释放锁要用 Lua 脚本？**

```
问题：如果用普通命令释放锁

Thread 1:
1. GET lock:key → 返回 "uuid1"（是自己的锁）
2. 此时锁刚好过期，被 Thread 2 获取
3. DEL lock:key → 删除了 Thread 2 的锁！

Lua 脚本保证原子性：
"判断锁是否属于自己的" 和 "删除锁" 是一个原子操作
```

### 🔄 看门狗机制（自动续期）

**问题：锁过期但业务未完成**

```
场景：
1. 获取锁，TTL = 30 秒
2. 业务执行需要 40 秒
3. 30 秒后锁过期，其他线程获取锁
4. 40 秒时释放锁 → 释放了别人的锁！

解决：看门狗机制
- 后台线程定期续期
- 业务完成才停止续期
```

```python
import threading

class DistributedLockWithWatchdog:
    """带看门狗的分布式锁"""
    
    def __init__(self, redis_client, lock_name, ttl=30):
        self.r = redis_client
        self.lock_name = f'lock:{lock_name}'
        self.ttl = ttl
        self.identifier = str(uuid.uuid4())
        self._stop_event = threading.Event()
        self._watchdog_thread = None
    
    def acquire(self, timeout=10):
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            if self.r.set(self.lock_name, self.identifier, nx=True, ex=self.ttl):
                # 获取锁成功，启动看门狗
                self._start_watchdog()
                return True
            time.sleep(0.001)
        
        return False
    
    def _start_watchdog(self):
        """启动看门狗线程"""
        def watchdog():
            while not self._stop_event.is_set():
                # 每 TTL/3 时间续期一次
                time.sleep(self.ttl / 3)
                if not self._stop_event.is_set():
                    self._renew()
        
        self._watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        self._watchdog_thread.start()
    
    def _renew(self):
        """续期"""
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        self.r.eval(script, 1, self.lock_name, self.identifier, self.ttl)
    
    def release(self):
        """释放锁"""
        # 停止看门狗
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1)
        
        # 释放锁
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        self.r.eval(script, 1, self.lock_name, self.identifier)
```

### 📋 Redisson 分布式锁（推荐）

生产环境推荐使用成熟的库，如 Java 的 Redisson，Python 的 redis-py-lock。

```python
# 使用 redis-py-lock
from redis_lock import Lock

with Lock(r, 'seckill:product:1001', expire=30, auto_renewal=True):
    # 自动续期，自动释放
    stock = int(r.get('stock:product:1001'))
    if stock > 0:
        r.decr('stock:product:1001')
```

---

## 内存优化技巧

### 🗜️ 选择合适的数据结构

```python
# ❌ 错误：用 String 存储大量小对象
for i in range(100000):
    r.set(f'item:{i}', json.dumps({'id': i, 'value': f'v{i}'}))

# 内存占用：约 100000 * (key长度 + value长度 + 元数据) ≈ 20MB

# ✅ 正确：用 Hash 存储
fields = {str(i): json.dumps({'id': i, 'value': f'v{i}'}) 
          for i in range(100000)}
r.hset('items', mapping=fields)

# 内存占用：约 5MB（节省 75%）
```

### 🗜️ Key 命名优化

```python
# ❌ 过长的 Key
r.set('this_is_a_very_long_key_name_that_wastes_memory:1001', 'value')

# ✅ 简短的 Key
r.set('usr:1001', 'value')

# 每个字符占用 1 字节，100 字节的 key 比比 10 字节的多占用 90 字节
# 如果有 100 万个 key，就是 90MB 的额外开销！
```

### 🗜️ 使用 Pipeline 批量操作

```python
# ❌ 每次操作都发送网络请求
for i in range(1000):
    r.set(f'key:{i}', f'value:{i}')
# 1000 次网络往返

# ✅ 使用 Pipeline
pipe = r.pipeline()
for i in range(1000):
    pipe.set(f'key:{i}', f'value:{i}')
pipe.execute()
# 1 次网络往返
```

### 🗜️ 内存配置优化

```bash
# redis.conf

# 数据结构优化阈值
hash-max-listpack-entries 512   # Hash 超过 512 字段转为 HashTable
hash-max-listpack-value 64      # Hash 值超过 64 字节转为 HashTable
list-max-listpack-size -2       # List 节点大小
set-max-intset-entries 512      # Set 超过 512 元素转为 HashTable
zset-max-listpack-entries 128   # ZSet 超过 128 元素转为 SkipList

# 内存淘汰策略
maxmemory 4gb
maxmemory-policy allkeys-lru
```

---

## 反模式与最佳实践

### ⚠️ 避免 Big Key

```python
# ❌ 错误：存储大对象
big_data = json.dumps(large_object)  # 10MB
r.set('big:key', big_data)

# 问题：
# 1. 网络传输慢
# 2. 阻塞其他操作
# 3. 主从同步慢

# ✅ 正确：拆分为多个小 Key
for i, chunk in enumerate(chunks(large_object, 1000)):
    r.hset('big:key', f'chunk:{i}', json.dumps(chunk))

# 检查 Big Key
# redis-cli --bigkeys
# 或
# MEMORY USAGE key
```

### ⚠️ 避免 O(N) 操作

```python
# ❌ 危险：KEYS 命令
keys = r.keys('*')  # 阻塞！遍历所有 key

# ✅ 正确：使用 SCAN
cursor = 0
while True:
    cursor, keys = r.scan(cursor, match='user:*', count=100)
    # 处理 keys
    if cursor == 0:
        break
```

### ⚠️ 合理设置 TTL

```python
# ❌ 没有过期时间
r.set('session:abc', session_data)

# ✅ 设置合理的过期时间
r.setex('session:abc', 3600, session_data)

# ✅ 预热时设置随机过期时间
import random
ttl = 3600 + random.randint(0, 600)
r.setex(cache_key, ttl, value)
```

### ⚠️ 热 Key 优化

```python
# 问题：单个 key 被高频访问
def get_hot_data():
    return r.get('hot:key')  # 单点压力

# 解决方案1：本地缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_hot_data_cached():
    return r.get('hot:key')

# 解决方案2：热 Key 分片
def get_hot_key_sharded(key, shards=10):
    shard = random.randint(0, shards - 1)
    return r.get(f'{key}:{shard}')

def set_hot_key_sharded(key, value, shards=10):
    pipe = r.pipeline()
    for i in range(shards):
        pipe.set(f'{key}:{i}', value)
    pipe.execute()
```

---

## 参考资料

- [Redis 官方文档](https://redis.io/docs/)
- [Redis 7.x Release Notes](https://redis.io/docs/latest/operate/oss_and_stack/management/upgrading/)
- [Redis 设计与实现](http://redisbook.com/)
- [redis-py 文档](https://redis-py.readthedocs.io/)