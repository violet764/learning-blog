# Redis 面试题

Redis 是目前最流行的键值存储数据库，在缓存、分布式锁、消息队列等场景广泛应用。本文整理了 Redis 面试高频问题，从基础到高级，帮助系统掌握 Redis 知识体系。

---

## 基础篇

### 1. Redis 是什么？有什么特点？

**答案：**

Redis（Remote Dictionary Server）是一个开源的、基于内存的键值对存储数据库，支持多种数据结构。

**核心特点：**

| 特性 | 说明 |
|------|------|
| 基于内存 | 数据存储在内存中，读写速度极快 |
| 数据结构丰富 | 支持 String、Hash、List、Set、ZSet、Stream 等 |
| 持久化 | 支持 RDB 和 AOF 两种持久化方式 |
| 高可用 | 支持主从复制、哨兵模式、集群模式 |
| 单线程模型 | 核心命令执行是单线程的 |
| 支持事务 | 提供简单的事务机制 |

**典型应用场景：**

```bash
# 缓存
SET user:1001 '{"name":"张三","age":25}'
EXPIRE user:1001 3600

# 分布式锁
SET lock:order 1 NX EX 30

# 计数器
INCR page:view:home

# 排行榜
ZADD leaderboard 100 user1
ZADD leaderboard 200 user2
ZREVRANGE leaderboard 0 9 WITHSCORES

# 消息队列
LPUSH queue:task '{"type":"email","to":"user@example.com"}'
RPOP queue:task
```

**可能的追问：Redis 为什么选择单线程？**

> Redis 的核心操作是 CPU 密集型，多线程反而会增加上下文切换开销。同时，单线程避免了并发竞争问题，代码更简洁、更易维护。Redis 6.0 引入多线程主要用于网络 I/O，核心命令执行仍是单线程。

---

### 2. Redis 为什么这么快？

**答案：**

Redis 的高性能主要来自以下几个方面：

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis 高性能原因                          │
├─────────────────────────────────────────────────────────────┤
│  1. 基于内存存储                                             │
│     └─ 内存访问速度: ~100ns vs 磁盘: ~10ms                   │
│                                                              │
│  2. 单线程模型                                               │
│     └─ 无上下文切换、无锁竞争                                │
│                                                              │
│  3. IO 多路复用                                             │
│     └─ epoll 实现高并发连接处理                              │
│                                                              │
│  4. 高效数据结构                                             │
│     └─ SDS、跳表、压缩列表等优化                             │
└─────────────────────────────────────────────────────────────┘
```

**性能对比：**

```python
# 不同存储介质访问时间对比
内存访问:    ~100 纳秒
SSD 访问:    ~100 微秒（比内存慢 1000 倍）
HDD 访问:    ~10 毫秒（比内存慢 100000 倍）
```

**IO 多路复用示意：**

```c
// Redis 使用 epoll 实现事件循环
while (!stop) {
    // 同时监听多个连接，有事件就处理
    int nready = epoll_wait(epfd, events, MAX_EVENTS, timeout);
    
    for (int i = 0; i < nready; i++) {
        // 处理就绪的事件
        processEvent(events[i]);
    }
}
```

**可能的追问：单线程会不会成为瓶颈？**

> 对于大多数场景不会。Redis 的瓶颈通常是内存大小或网络带宽，而非 CPU。如果确实需要更高性能，可以：
> 1. 使用 Redis Cluster 水平扩展
> 2. 开启 Redis 6.0 的多线程网络 I/O
> 3. 使用 Pipeline 批量执行命令

---

### 3. Redis 支持哪些数据类型？各有什么应用场景？

**答案：**

Redis 支持 5 种基本数据类型和几种高级数据类型：

**基础数据类型：**

```
┌──────────────┬───────────────────────┬─────────────────────────┐
│   数据类型   │       底层实现        │       应用场景          │
├──────────────┼───────────────────────┼─────────────────────────┤
│ String       │ SDS (简单动态字符串)  │ 缓存、计数器、分布式锁  │
│ Hash         │ 压缩列表/哈希表       │ 对象存储、购物车        │
│ List         │ 压缩列表/双向链表     │ 消息队列、文章列表      │
│ Set          │ 整数集合/哈希表       │ 标签、共同关注、抽奖    │
│ ZSet         │ 压缩列表/跳表+哈希表  │ 排行榜、延迟队列        │
└──────────────┴───────────────────────┴─────────────────────────┘
```

**代码示例：**

```bash
# ========== String ==========
SET token:user1 "abc123xyz"
GET token:user1
INCR article:read:count    # 计数器
SET lock:order 1 NX EX 10  # 分布式锁

# ========== Hash ==========
HSET user:1001 name "张三" age 25 city "北京"
HGET user:1001 name
HGETALL user:1001
HINCRBY user:1001 age 1    # 年龄 +1

# ========== List ==========
LPUSH news:list "新闻1"     # 头部插入
RPUSH news:list "新闻3"     # 尾部插入
LRANGE news:list 0 -1       # 获取全部
LPOP news:list              # 头部弹出

# ========== Set ==========
SADD tag:article:1 "Redis" "数据库" "缓存"
SADD tag:article:2 "MySQL" "数据库"
SINTER tag:article:1 tag:article:2  # 交集：共同标签
SPOP lucky:draw 1                    # 抽奖

# ========== ZSet ==========
ZADD leaderboard 100 user1 200 user2 150 user3
ZREVRANGE leaderboard 0 2 WITHSCORES  # Top 3
ZRANK leaderboard user1                # 查询排名
```

**高级数据类型：**

```bash
# Bitmap - 位图，用于签到、在线状态
SETBIT user:signin:20240101 1001 1  # 用户1001签到
GETBIT user:signin:20240101 1001    # 查询是否签到
BITCOUNT user:signin:20240101       # 统计签到人数

# HyperLogLog - 基数统计，用于UV统计
PFADD uv:page:home user1 user2 user3
PFCOUNT uv:page:home  # 返回去重后的数量（有误差）

# Geo - 地理位置计算
GEOADD locations 116.40 39.90 "北京"
GEODIST locations "北京" "上海" km

# Stream - 消息队列（Redis 5.0+）
XADD mystream * field1 value1
XREAD COUNT 2 STREAMS mystream 0
```

**可能的追问：Hash 和 String 存对象有什么区别？**

> 1. **Hash 适合频繁更新字段**：可以单独修改某个字段，不需要读取整个对象
> 2. **String 适合整体读写**：对象作为一个整体存取，适合数据完整性要求高的场景
> 3. **内存效率**：字段少时 Hash 更省内存，字段多时 String 更紧凑

---

### 4. 为什么用 Redis 做缓存而不是 MySQL？

**答案：**

对比 MySQL 和 Redis 作为缓存的差异：

```
┌─────────────────┬────────────────────┬────────────────────────┐
│     维度        │      MySQL         │        Redis           │
├─────────────────┼────────────────────┼────────────────────────┤
│ 存储介质        │ 磁盘为主           │ 内存为主               │
│ 读写速度        │ 毫秒级             │ 微秒级（快 100+ 倍）   │
│ QPS             │ ~1000-5000         │ ~100000+               │
│ 数据结构        │ 关系型表           │ 丰富键值类型           │
│ 事务            │ ACID 事务          │ 简单事务               │
│ 持久化          │ 强持久化           │ 可选持久化             │
│ 查询复杂度      │ 支持复杂 SQL       │ 简单键值操作           │
└─────────────────┴────────────────────┴────────────────────────┘
```

**典型架构：**

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  客户端  │────▶│  应用   │────▶│  Redis  │────▶│  MySQL  │
└─────────┘     └─────────┘     │ (缓存)  │     │(持久化) │
                                └─────────┘     └─────────┘
                                     │
                                Cache Miss
                                     │
                                     ▼
                                从 MySQL 读取
                                写入 Redis
```

**缓存使用模式：**

```java
// 读缓存模式
public User getUser(Long id) {
    // 1. 先查缓存
    User user = redis.get("user:" + id);
    if (user != null) {
        return user;
    }
    
    // 2. 缓存未命中，查数据库
    user = mysql.query("SELECT * FROM user WHERE id = ?", id);
    
    // 3. 写入缓存
    if (user != null) {
        redis.setex("user:" + id, 3600, user);
    }
    return user;
}
```

**可能的追问：Redis 可以完全替代 MySQL 吗？**

> 不能。Redis 的局限性：
> 1. **数据容量有限**：受内存限制，不适合存储海量数据
> 2. **查询能力弱**：不支持复杂 SQL，无法做多表关联
> 3. **数据可靠性**：虽然支持持久化，但相比 MySQL 仍较弱
> 4. **事务支持弱**：不支持 ACID 事务，无法保证复杂业务一致性
>
> 正确做法是 **Redis 做缓存 + MySQL 做持久化**，各取所长。

---

### 5. Redis 是单线程的，为什么还能高性能？

**答案：**

Redis 单线程高性能的原因：

**1. 纯内存操作**

```
CPU 处理速度: ~3 GHz（每秒 30 亿次操作）
内存访问延迟: ~100 纳秒
Redis 单次操作: ~1-5 微秒

理论上单线程 QPS = 1秒 / 1微秒 = 100,000+
```

**2. 避免多线程开销**

```
多线程开销：
┌─────────────────────────────────────────┐
│ 1. 线程创建/销毁开销                     │
│ 2. 上下文切换（保存/恢复寄存器、栈指针） │
│ 3. 锁竞争和死锁检测                      │
│ 4. 缓存失效（CPU 缓存行失效）            │
└─────────────────────────────────────────┘

单线程优势：无需锁、无切换、简单可靠
```

**3. IO 多路复用**

```c
// Redis 事件循环核心逻辑
void aeMain(aeEventLoop *eventLoop) {
    while (!eventLoop->stop) {
        // epoll 监听多个 socket，有事件就处理
        numevents = aeApiPoll(eventLoop, tvp);
        for (j = 0; j < numevents; j++) {
            // 处理就绪的事件
            fe->rfileProc(eventLoop, fd, fe->clientData, mask);
        }
    }
}
```

**4. 高效数据结构**

```
SDS      → O(1) 获取长度，避免 C 字符串 O(n)
哈希表   → O(1) 平均查找
跳表     → O(log n) 查找，比红黑树实现简单
压缩列表 → 小数据量内存紧凑存储
```

**可能的追问：Redis 6.0 为什么引入多线程？**

> Redis 6.0 引入多线程处理 **网络 I/O**（读写 socket 数据），而非命令执行：
>
> ```
> Redis 6.0 多线程架构：
> ┌─────────────────────────────────────────────────────┐
> │  主线程（命令执行）                                  │
> │     ↓                                               │
> │  多线程网络 I/O（读取请求、发送响应）                │
> │     - 可配置 1-8 个 I/O 线程                         │
> │     - 只处理网络数据，不执行命令                     │
> └─────────────────────────────────────────────────────┘
> ```
>
> **开启方式：**
> ```bash
> # redis.conf
> io-threads 4            # I/O 线程数
> io-threads-do-reads yes # 开启读多线程
> ```
>
> 性能提升：在网络成为瓶颈时，可提升 50%-100% 吞吐量。

---

### 6. Redis 和 Memcached 有什么区别？

**答案：**

| 对比项 | Redis | Memcached |
|--------|-------|-----------|
| 数据类型 | 5 种基本类型 + 高级类型 | 仅支持 String |
| 持久化 | 支持 RDB/AOF | 不支持 |
| 集群 | 原生支持 Cluster | 需要客户端分片 |
| 线程模型 | 单线程（6.0 网络 I/O 多线程） | 多线程 |
| 内存管理 | 自定义内存分配器 | Slab Allocator |
| 事务 | 支持简单事务 | 不支持 |
| 过期策略 | 惰性删除 + 定期删除 | 惰性删除 |

**选择建议：**

```yaml
# 选择 Redis 的场景
- 需要复杂数据结构（如排行榜、消息队列）
- 需要持久化
- 需要高可用（主从、集群）
- 需要分布式锁、事务等高级功能

# 选择 Memcached 的场景
- 简单键值缓存
- 多核服务器，QPS 要求极高
- 已有成熟的 Memcached 运维体系
```

**可能的追问：多线程的 Memcached 为什么不一定比 Redis 快？**

> 1. **内存操作不是瓶颈**：Redis 单次操作仅 1-5 微秒，瓶颈在网络
> 2. **锁竞争开销**：Memcached 多线程需要加锁，反而增加开销
> 3. **数据结构效率**：Redis 的跳表、压缩列表等数据结构优化好
> 4. **实际测试**：Redis 单线程性能已达 10 万+ QPS，足够大多数场景

---

### 7. Redis 如何实现事务？

**答案：**

Redis 事务通过 MULTI、EXEC、DISCARD、WATCH 命令实现：

```bash
# 基本事务
MULTI                # 开启事务
SET account:a 100
SET account:b 50
INCRBY account:a -20
INCRBY account:b 20
EXEC                 # 执行事务

# WATCH 实现乐观锁
WATCH account:a      # 监视 key
val = GET account:a  # 获取值
MULTI
SET account:a (val - 100)
EXEC                 # 如果 account:a 被其他客户端修改，事务失败
```

**事务特性：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis 事务特性                            │
├─────────────────────────────────────────────────────────────┤
│ 1. 批量执行：MULTI 到 EXEC 之间的命令原子执行               │
│ 2. 隔离性：事务执行期间不会被其他命令打断                    │
│ 3. 不支持回滚：命令执行失败不会回滚已执行的命令              │
│ 4. 无 ACID：不满足传统数据库的 ACID 特性                    │
└─────────────────────────────────────────────────────────────┘
```

**事务错误处理：**

```bash
# 情况1：语法错误（入队时检测）
MULTI
SET key value
INCR key key key    # 语法错误
EXEC                # 返回错误，所有命令都不执行

# 情况2：运行时错误（执行时检测）
MULTI
SET key "hello"
INCR key            # 对字符串 INCR，运行时错误
EXEC                # SET 执行成功，INCR 失败，不回滚
```

**可能的追问：Redis 事务为什么不能回滚？**

> Redis 作者的观点：
> 1. **Redis 命令错误通常来自编程 bug**，不应该在线上出现
> 2. **回滚机制复杂**：需要维护操作日志，增加复杂度
> 3. **保持简洁**：Redis 追求简单高效，不支持回滚保持代码简洁
> 4. **性能考虑**：回滚需要额外开销，影响性能
>
> 如果需要原子性操作，建议使用 Lua 脚本。

---

### 8. Redis 的 Lua 脚本有什么用？

**答案：**

Lua 脚本可以保证多条命令原子执行，弥补事务的不足：

```bash
# 扣减库存 + 记录日志（原子执行）
EVAL "
    local stock = tonumber(redis.call('GET', KEYS[1]))
    if stock > 0 then
        redis.call('DECR', KEYS[1])
        redis.call('LPUSH', KEYS[2], ARGV[1])
        return 1
    end
    return 0
" 2 stock:product:1 log:product:1 '{"action":"buy","time":"2024-01-01"}'
```

**Lua 脚本优势：**

```javascript
// 复杂的分布式锁释放（原子性）
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end

// 对应 Java 代码
String script = "if redis.call('get', KEYS[1]) == ARGV[1] then " +
                "return redis.call('del', KEYS[1]) " +
                "else return 0 end";
redis.eval(script, 1, "lock:order", uuid);
```

**使用注意：**

```bash
# 使用 SCRIPT LOAD 预加载脚本（避免重复传输）
SCRIPT LOAD "return redis.call('GET', KEYS[1])"
# 返回 SHA1: 4e6d8fc8bb01276962cce5371fa795a236599bb4

# 使用 EVALSHA 执行
EVALSHA 4e6d8fc8bb01276962cce5371fa795a236599bb4 1 key
```

**可能的追问：Lua 脚本有什么注意事项？**

> 1. **避免死循环**：Lua 脚本阻塞 Redis，长时间执行会影响其他请求
> 2. **超时限制**：默认 5 秒，可通过 lua-time-limit 配置
> 3. **脚本缓存**：大脚本用 SHA 引用，减少网络传输
> 4. **不要做复杂计算**：保持脚本简单高效

---

## 数据结构与底层实现篇

### 1. SDS 是什么？为什么比 C 字符串好？

**答案：**

SDS（Simple Dynamic String，简单动态字符串）是 Redis 自定义的字符串实现。

**SDS 数据结构：**

```c
struct sdshdr {
    int len;     // 已使用长度
    int free;    // 剩余可用长度
    char buf[];  // 存储实际数据（兼容 C 字符串）
};

// 示例：存储 "Redis"
// ┌─────┬─────┬─────────────────────┐
// │ len │free │ R │ e │ d │ i │ s │\0 │
// │  5  │  3  │   │   │   │   │   │   │
// └─────┴─────┴─────────────────────┘
```

**SDS vs C 字符串对比：**

| 特性 | C 字符串 | SDS |
|------|----------|-----|
| 获取长度 | O(n) 遍历 | O(1) 直接读取 |
| 缓冲区溢出 | 容易溢出 | 自动扩展，安全 |
| 内存重分配 | 每次修改都要 | 预分配 + 惰性释放 |
| 二进制安全 | 遇 \0 截止 | 支持任意数据 |
| 兼容性 | - | 兼容 C 字符串函数 |

**空间预分配策略：**

```c
// 扩展时预分配空间
if (new_len < 1MB) {
    free = new_len;      // 翻倍分配
} else {
    free = 1MB;          // 固定 +1MB
}

// 示例：追加字符串
sds sdscat(sds s, const char *t) {
    size_t len = strlen(t);
    s = sdsMakeRoomFor(s, len);  // 自动扩展
    memcpy(s+len, t, len+1);
    s->len += len;
    return s;
}
```

**可能的追问：什么是惰性空间释放？**

> 当缩短字符串时，SDS 不会立即释放多余内存，而是记录到 free 字段：
>
> ```c
> // 缩短 "Redis World" 为 "Redis"
> // 缩短前
> // ┌─────┬─────┬────────────────────────────┐
> // │ 11  │  0  │ R e d i s   W o r l d  \0  │
> // └─────┴─────┴────────────────────────────┘
> 
> // 缩短后（不释放内存）
> // ┌─────┬─────┬────────────────────────────┐
> // │  5  │  6  │ R e d i s   W o r l d  \0  │
> // └─────┴─────┴────────────────────────────┘
> ```
>
> 下次追加时直接使用 free 空间，避免内存重分配。如果需要释放，可调用 `sdsRemoveFreeSpace`。

---

### 2. 为什么 Redis 用跳表而不用红黑树实现 ZSet？

**答案：**

跳表（Skip List）是一种概率数据结构，通过多层索引实现快速查找。

**跳表结构示意：**

```
Level 4:        ┌───────→ 50 ────────────────→ NULL
                │
Level 3:   HEAD → 10 ────────→ 30 ────────→ 50 → NULL
                           │
Level 2:   HEAD → 10 → 20 → 30 → 40 → 50 → NULL
                    │         │
Level 1:   HEAD → 10 → 20 → 30 → 40 → 50 → NULL
                ↓    ↓    ↓    ↓    ↓
           [节点]  [节点] [节点] [节点] [节点]
```

**查找过程（查找 40）：**

```
Level 4: HEAD → ... → 50 (40 < 50, 下跳)
Level 3: HEAD → 10 → 30 → 50 (40 < 50, 下跳)
Level 2: ... 30 → 40 (找到!)
```

**为什么选跳表而不是红黑树？**

| 对比项 | 跳表 | 红黑树 |
|--------|------|--------|
| 实现复杂度 | 简单（~100 行代码） | 复杂（~300+ 行代码） |
| 查找效率 | O(log n) | O(log n) |
| 插入/删除 | O(log n)，简单 | O(log n)，需旋转平衡 |
| 范围查询 | 天然支持（遍历底层链表） | 需要中序遍历 |
| 内存占用 | 每节点 ~1.33 个指针 | 每节点固定 3 个指针 |
| 并发友好 | 容易实现无锁版本 | 锁竞争复杂 |

**范围查询优势：**

```c
// 跳表范围查询：找到起点后直接遍历链表
ZRangeResult* zslRange(zskiplist *zsl, double min, double max) {
    // 1. 找到第一个 >= min 的节点
    x = zslFirstInRange(zsl, min, max);
    
    // 2. 遍历链表直到 > max
    while (x && x->score <= max) {
        result.add(x);
        x = x->level[0].forward;  // 直接访问下一节点
    }
}

// 红黑树范围查询：需要中序遍历，递归或栈实现
```

**可能的追问：跳表的随机层数如何确定？**

> Redis 使用概率性层数生成算法：
>
> ```c
> int zslRandomLevel(void) {
>     int level = 1;
>     // 每层晋升概率 25%
>     while ((random() & 0xFFFF) < (0.25 * 0xFFFF)) {
>         level++;
>     }
>     return level < ZSKIPLIST_MAXLEVEL ? level : ZSKIPLIST_MAXLEVEL;
> }
> 
> // 理论平均层数：1/(1-0.25) = 1.33
> // 空间复杂度：O(n)（比红黑树稍高）
> ```

---

### 3. QuickList 是什么？为什么 Redis 用它？

**答案：**

QuickList（快速列表）是 Redis 3.2 引入的 List 底层实现，是 ziplist 和 linkedlist 的结合体。

**演进历史：**

```
┌─────────────────────────────────────────────────────────────┐
│  Redis List 实现演变                                         │
├─────────────────────────────────────────────────────────────┤
│  早期: linkedlist (双向链表)                                 │
│        ↓ 内存碎片多，指针开销大                              │
│                                                              │
│  优化: ziplist (压缩列表)                                    │
│        ↓ 连续内存，但插入/删除 O(n)，连锁更新问题            │
│                                                              │
│  最终: quicklist = linkedlist + ziplist                     │
│        分段存储，兼顾内存效率和操作效率                      │
└─────────────────────────────────────────────────────────────┘
```

**QuickList 结构：**

```
quicklist
    │
    ├── quicklistNode 1 ──→ quicklistNode 2 ──→ quicklistNode 3
    │        │                    │                    │
    │        ↓                    ↓                    ↓
    │   ┌─────────┐         ┌─────────┐         ┌─────────┐
    │   │ ziplist │         │ ziplist │         │ ziplist │
    │   │ [a,b,c] │         │ [d,e,f] │         │ [g,h,i] │
    │   └─────────┘         └─────────┘         └─────────┘
    │
    └── 每个 ziplist 节点存储多个元素
```

**配置参数：**

```bash
# redis.conf
list-max-ziplist-size -2    # 单个 ziplist 大小限制（负数表示字节）
list-compress-depth 1       # 压缩深度，0 表示不压缩
```

**内存优化效果：**

```python
# 存储 1000 个元素

# 纯链表: 每节点 ~32 字节（2 指针 + 数据）
1000 * 32 = 32,000 字节

# QuickList: 每个压缩列表存 512 个元素
2 个节点 * 32 + 2 * 512 * 8 ≈ 8,256 字节

# 内存节省 ~75%
```

**可能的追问：什么是压缩深度？**

> 为了节省内存，QuickList 支持压缩中间节点（LZF 算法）：
>
> ```
> 压缩深度 = 1:
> [head] ↔ [压缩] ↔ [压缩] ↔ [压缩] ↔ [tail]
>            ↑                    ↑
>         中间节点压缩，首尾节点不压缩（频繁访问）
> 
> 压缩深度 = 0：所有节点都不压缩
> ```

---

### 4. Redis 字典是如何实现的？

**答案：**

Redis 字典使用哈希表实现，采用链地址法解决冲突。

**字典数据结构：**

```c
// 哈希表节点
typedef struct dictEntry {
    void *key;
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
    struct dictEntry *next;  // 链表指向下一节点
} dictEntry;

// 哈希表
typedef struct dictht {
    dictEntry **table;  // 哈希表数组
    unsigned long size; // 哈希表大小
    unsigned long sizemask; // 哈希表大小掩码
    unsigned long used; // 已有节点数量
} dictht;

// 字典
typedef struct dict {
    dictType *type;     // 类型特定函数
    void *privdata;     // 私有数据
    dictht ht[2];       // 两个哈希表（用于 rehash）
    int rehashidx;      // rehash 进度（-1 表示未进行）
} dict;
```

**哈希冲突解决：**

```
哈希表数组:
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ ...
└────┴────┴────┴────┘
  │
  ▼
┌─────────┐
│ key1    │ → ┌─────────┐
│ value1  │   │ key3    │ → NULL (链地址法)
└─────────┘   │ value3  │
              └─────────┘
```

**扩容条件：**

```c
// 扩容判断
if (d->ht[0].used >= d->ht[0].size &&
    (dict_can_resize || d->ht[0].used / d->ht[0].size > dict_force_resize_ratio)) {
    return dictExpand(d, d->ht[0].used * 2);
}

// 扩容规则：
// 1. 服务器未执行 BGSAVE/BGREWRITEAOF：负载因子 >= 1 触发扩容
// 2. 服务器正在执行 BGSAVE/BGREWRITEAOF：负载因子 >= 5 触发扩容
```

**可能的追问：为什么有两个哈希表？**

> 两个哈希表用于实现 **渐进式 Rehash**：
>
> - `ht[0]`：当前使用的哈希表
> - `ht[1]`：Rehash 目标哈希表
>
> Rehash 过程中，两个表同时使用，新数据写入 `ht[1]`，旧数据逐步迁移。

---

### 5. Redis 为什么用渐进式 Rehash？

**答案：**

渐进式 Rehash 将一次性大量迁移分散到多次操作中，避免阻塞服务。

**Rehash 过程：**

```
┌─────────────────────────────────────────────────────────────┐
│                     渐进式 Rehash 流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 初始化                                                   │
│     ht[1] 分配空间 = ht[0].used * 2                          │
│     rehashidx = 0                                           │
│                                                              │
│  2. 渐进迁移（每次操作迁移部分）                              │
│     for (i = 0; i < 每次迁移数量; i++) {                     │
│         迁移 ht[0][rehashidx] 的所有节点到 ht[1]             │
│         rehashidx++                                         │
│     }                                                        │
│                                                              │
│  3. 完成迁移                                                 │
│     ht[0] = ht[1]                                           │
│     释放 ht[1]                                               │
│     rehashidx = -1                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**迁移时机：**

```c
// 每个 CRUD 操作都会触发迁移一部分
void dictRehashMilliseconds(dict *d, int ms) {
    long long start = timeInMilliseconds();
    int rehashes = 0;
    
    while (dictRehash(d, 100)) {  // 每次迁移 100 个槽位
        rehashes += 100;
        if (timeInMilliseconds() - start > ms) break;
    }
}
```

**查询策略：**

```c
// Rehash 期间的查询：先查 ht[0]，再查 ht[1]
dictEntry *dictFind(dict *d, const void *key) {
    // 查 ht[0]
    h = dictHashKey(d, key) & d->ht[0].sizemask;
    he = d->ht[0].table[h];
    while (he) {
        if (dictCompareKeys(d, key, he->key)) return he;
        he = he->next;
    }
    
    // 如果在 rehash，再查 ht[1]
    if (dictIsRehashing(d)) {
        h = dictHashKey(d, key) & d->ht[1].sizemask;
        he = d->ht[1].table[h];
        while (he) {
            if (dictCompareKeys(d, key, he->key)) return he;
            he = he->next;
        }
    }
    return NULL;
}
```

**可能的追问：为什么不是一次性 Rehash？**

> 一次性 Rehash 的问题：
>
> ```python
> # 假设字典有 1000 万个元素
> # 一次性迁移需要：
> # - 遍历所有元素
> # - 重新计算哈希
> # - 移动到新表
> 
> # 估算耗时
> 1000万 * 100纳秒 ≈ 1秒
> 
> # Redis 单线程，这 1 秒内无法响应其他请求！
> ```
>
> 渐进式 Rehash 将这 1 秒的工作分散到多次操作中，每次仅迁移少量数据，确保服务响应。

---

### 6. intset 是什么？有什么特点？

**答案：**

intset（整数集合）是 Set 类型在元素都是整数且数量较少时的底层实现。

**数据结构：**

```c
typedef struct intset {
    uint32_t encoding;  // 编码方式：int16/int32/int64
    uint32_t length;    // 元素数量
    int8_t contents[];  // 有序整数数组
} intset;

// 示例：存储 {1, 3, 5}
// ┌───────────┬───────────┬─────────────────┐
// │ encoding  │  length   │ 1 │ 3 │ 5 │ ...  │
// │  INT16    │     3     │   │   │   │      │
// └───────────┴───────────┴─────────────────┘
```

**编码升级：**

```c
// 初始 int16 编码，插入 int32 范围的值
intset *intsetAdd(intset *is, int64_t value, uint8_t *success) {
    uint8_t valenc = _intsetValueEncoding(value);
    
    // 需要升级编码
    if (valenc > intrev32ifbe(is->encoding)) {
        // 1. 计算新大小
        uint32_t newlen = intrev32ifbe(is->length) + 1;
        size_t newsize = newlen * sizeof(int64_t);
        
        // 2. 扩展内存
        is = zrealloc(is, sizeof(intset) + newsize);
        
        // 3. 迁移旧元素（从后往前，避免覆盖）
        for (i = intrev32ifbe(is->length); i >= 0; i--) {
            _intsetSet(is, i, _intsetGetEncoded(is, i, oldenc));
        }
        
        // 4. 更新编码
        is->encoding = INT64;
    }
    
    // 5. 插入新元素（保持有序）
    // ...
}
```

**升级示例：**

```
原集合 {1, 2, 3} (int16 编码):
┌─────────┬───────┬───────────────────┐
│ INT16   │   3   │  1  │  2  │  3   │
└─────────┴───────┴───────────────────┘
         每个 2 字节

插入 65535 (需要 int32):
┌─────────┬───────┬─────────────────────────────────────────┐
│ INT32   │   4   │    1    │    2    │    3    │  65535   │
└─────────┴───────┴─────────────────────────────────────────┘
         每个 4 字节
```

**特点：**

| 特点 | 说明 |
|------|------|
| 有序存储 | 元素按从小到大排列 |
| 无重复 | 自动去重 |
| 编码升级 | 根据最大值自动选择 int16/int32/int64 |
| 内存紧凑 | 连续内存，无指针开销 |

**可能的追问：为什么只升级不降级？**

> Redis 设计考虑：
> 1. **使用场景**：Set 删除元素后，通常不会再插入更大的数
> 2. **复杂度**：降级需要遍历所有元素判断是否可以降级
> 3. **性能**：避免频繁的编码切换开销

---

### 7. ziplist 有什么优缺点？

**答案：**

ziplist（压缩列表）是一块连续内存，用于存储小数据量场景。

**内存布局：**

```
┌──────────┬───────────┬─────────┬─────────────┬─────┬─────────┬──────────┐
│ zlbytes  │ zltail    │ zllen   │ entry1      │ ... │ entryN  │ zlend    │
│ 4 bytes  │ 4 bytes   │ 2 bytes │ 变长        │     │ 变长    │ 1 byte   │
└──────────┴───────────┴─────────┴─────────────┴─────┴─────────┴──────────┘
     │          │           │
     │          │           └─ 元素数量
     │          └─ 尾节点偏移量（快速定位尾部）
     └─ ziplist 总字节数
```

**Entry 结构：**

```c
// 每个 entry 包含
// 1. prevlen：前一个 entry 的长度（用于反向遍历）
// 2. encoding：当前 entry 的编码类型
// 3. data：实际数据

// prevlen 编码：
// < 254: 1 字节存储
// >= 254: 5 字节存储（第 1 字节为 254，后 4 字节存实际长度）
```

**优点：**

```yaml
内存效率:
  - 连续内存，无指针开销
  - 小数据紧凑存储
  
示例:
  # 存储 3 个整数 {1, 2, 3}
  # 普通链表: 3 * 32 = 96 字节
  # ziplist: 约 15 字节
```

**缺点 - 连锁更新：**

```
场景：插入导致 prevlen 字段从 1 字节变为 5 字节

原状态: [entry1(253)] → [entry2(253)] → [entry3(253)]
                  │           │
                  └─ prevlen  └─ prevlen 都是 1 字节

插入一个 254 字节的 entry:
[entry1(253)] → [new(254)] → [entry2] → [entry3]
                          ↑
                      需要 5 字节存 prevlen

连锁反应:
- entry2 的 prevlen 需要从 1 → 5 字节
- entry2 大小变化，entry3 的 prevlen 也需要更新
- 最坏情况 O(n²)
```

**可能的追问：如何避免连锁更新？**

> Redis 7.0 引入 listpack 替代 ziplist：
>
> ```c
> // listpack 的每个 entry 存储：
> // - encoding: 编码类型
> // - data: 数据
> // - backlen: 当前 entry 长度（存自己，不存前一个）
> 
> // 反向遍历时，读取 backlen 即可计算前一个 entry 位置
> // 不再依赖前一个 entry 的长度，避免连锁更新
> ```

---

### 8. Redis 的对象系统是怎样的？

**答案：**

Redis 使用对象系统封装底层数据结构，支持类型检查、内存回收、对象共享等功能。

**对象结构：**

```c
typedef struct redisObject {
    unsigned type:4;        // 类型（4 bit）
    unsigned encoding:4;    // 编码方式（4 bit）
    unsigned lru:LRU_BITS;  // LRU 时间（24 bit）
    int refcount;           // 引用计数
    void *ptr;              // 指向底层数据结构
} robj;

// type 取值
#define OBJ_STRING 0
#define OBJ_LIST 1
#define OBJ_SET 2
#define OBJ_ZSET 3
#define OBJ_HASH 4

// encoding 取值（同一类型可能有多种编码）
#define OBJ_ENCODING_RAW 0        // raw SDS
#define OBJ_ENCODING_INT 1        // 整数
#define OBJ_ENCODING_HT 2         // 哈希表
#define OBJ_ENCODING_ZIPLIST 5    // 压缩列表
#define OBJ_ENCODING_INTSET 6     // 整数集合
#define OBJ_ENCODING_SKIPLIST 7   // 跳表
#define OBJ_ENCODING_EMBSTR 8     // embstr SDS
#define OBJ_ENCODING_QUICKLIST 9  // 快速列表
```

**类型与编码对应：**

```
┌──────────┬─────────────────────────────────────────────────┐
│  类型    │              编码方式                           │
├──────────┼─────────────────────────────────────────────────┤
│ String   │ int（整数）、embstr（短字符串）、raw（长字符串） │
│ List     │ quicklist（快速列表）                           │
│ Set      │ intset（整数集合）、ht（哈希表）                │
│ ZSet     │ ziplist（小数据）、skiplist+ht（大数据）        │
│ Hash     │ ziplist（小数据）、ht（哈希表）                 │
└──────────┴─────────────────────────────────────────────────┘
```

**编码转换条件：**

```c
// Hash 类型转换条件
// ziplist → hashtable
if (hash_max_ziplist_entries < num_fields ||    // 字段数 > 512
    hash_max_ziplist_value < max_value_len) {   // 任意值长度 > 64
    convert_to_hashtable();
}

// ZSet 类型转换条件
// ziplist → skiplist+hashtable
if (zset_max_ziplist_entries < num_members ||   // 成员数 > 128
    zset_max_ziplist_value < max_value_len) {   // 最大值长度 > 64
    convert_to_skiplist();
}
```

**可能的追问：embstr 和 raw 的区别？**

> String 类型的两种编码：
>
> ```c
> // embstr：短字符串（<= 44 字节）
> // redisObject 和 SDS 分配在一块连续内存
> ┌─────────────────────────────────────────┐
> │ redisObject (16) │ SDS header (3) │ data │
> └─────────────────────────────────────────┘
> 
> // raw：长字符串（> 44 字节）
> // redisObject 和 SDS 分别分配内存
> ┌───────────────┐     ┌───────────────────┐
> │ redisObject   │ ──→ │ SDS header │ data │
> └───────────────┘     └───────────────────┘
> ```
>
> embstr 优势：
> - 一次内存分配（减少分配次数）
> - 缓存友好（连续内存）

---

### 9. Redis 如何实现 LRU 淘汰？

**答案：**

Redis 使用近似 LRU 算法，通过采样实现高效淘汰。

**LRU 时钟：**

```c
// redisObject 中的 lru 字段（24 bit）
// 存储的是 LRU 时间戳（秒级，低 24 位）

// 获取对象空闲时间
unsigned long estimateObjectIdleTime(robj *o) {
    unsigned long lruclock = LRU_CLOCK();
    if (lruclock >= o->lru) {
        return (lruclock - o->lru) * 1000;  // 毫秒
    } else {
        return (lruclock + (LRU_CLOCK_MAX - o->lru)) * 1000;
    }
}
```

**近似 LRU 算法：**

```c
// 淘汰过程
void evictionPoolPopulate(dict *sampledict, evictionPool *pool) {
    // 1. 随机采样 N 个 key（默认 5 个）
    count = dictGetSomeKeys(sampledict, samples, server.maxmemory_samples);
    
    // 2. 计算每个 key 的空闲时间
    for (j = 0; j < count; j++) {
        idle = estimateObjectIdleTime(o);
        // 3. 加入淘汰池（按空闲时间排序）
        poolPush(pool, key, idle);
    }
    
    // 4. 淘汰池满后，删除空闲时间最长的 key
}
```

**淘汰池工作原理：**

```
淘汰池（16 个槽位）：
┌─────────────────────────────────────────────────────────────┐
│ 最短空闲时间 ←──────────────────────────────────→ 最长空闲时间 │
├─────────┬─────────┬─────────┬─────────┬─────────────────────┤
│ key1    │ key2    │ key3    │ key4    │ ...     │ key16     │
│ idle:1s │ idle:5s │ idle:10s│ idle:20s│         │ idle:300s │
└─────────┴─────────┴─────────┴─────────┴─────────────────────┘
                                ↑
                            新 key 只在空闲时间更长时才能入池
```

**可能的追问：为什么不使用精确 LRU？**

> 1. **精确 LRU 需要链表**：每次访问都要移动节点到头部，O(n) 复杂度
> 2. **内存开销大**：需要额外的指针维护链表
> 3. **近似 LRU 足够好**：采样 5 个即可达到 80% 精度，采样 10 个达到 95% 精度
>
> ```
> 精确 LRU vs 近似 LRU 效果对比：
> ┌────────────────────────────────────┐
> │ ████████████ 精确 LRU             │
> │ ██████████  近似 LRU (10 samples) │
> │ ████████    近似 LRU (5 samples)  │
> └────────────────────────────────────┘
> ```

---

### 10. LFU 淘汰策略是如何实现的？

**答案：**

LFU（Least Frequently Used）记录访问频率，优先淘汰访问频率最低的数据。

**实现方式：**

```c
// lru 字段（24 bit）在 LFU 模式下的含义
// ├── ldt (16 bit): 上次访问时间（分钟级）
// └── counter (8 bit): 对数计数器

// 计数器增长算法（对数增长，防止热点数据计数值溢出）
uint8_t LFULogIncr(uint8_t counter) {
    if (counter == 255) return 255;
    
    double r = (double)rand() / RAND_MAX;
    double baseval = counter - LFU_INIT_VAL;  // 初始值 5
    
    double p = 1.0 / (baseval * server.lfu_log_factor + 1);
    if (r < p) counter++;
    
    return counter;
}

// 随着访问次数增加，增长概率降低
// counter=0: p=1.0（必定增长）
// counter=10: p=0.09（约 11 次访问增长 1 次）
// counter=100: p=0.0099
```

**计数器衰减：**

```c
// 根据距离上次访问的时间，衰减计数器
unsigned long LFUDecrAndReturn(robj *o) {
    unsigned long ldt = o->lru >> 8;
    unsigned long counter = o->lru & 255;
    
    // 计算衰减量 = 分钟差 * 衰减因子
    unsigned long num_periods = server.lfu_decay_time 
                              ? (timeInMinutes() - ldt) / server.lfu_decay_time 
                              : 0;
    
    if (num_periods) {
        counter = (num_periods > counter) ? 0 : counter - num_periods;
    }
    
    return counter;
}
```

**配置参数：**

```bash
# redis.conf
lfu-log-factor 10     # 计数器增长因子，越大增长越慢
lfu-decay-time 1      # 衰减时间（分钟），越大衰减越慢
```

**可能的追问：LFU vs LRU 如何选择？**

> ```yaml
> 选择 LRU 场景:
>   - 数据访问模式相对均匀
>   - 有明显的热点数据
>   - 旧数据逐渐不再访问
> 
> 选择 LFU 场景:
>   - 需要长期热点识别
>   - 防止历史热点占用缓存
>   - 如：新闻推荐、电商商品
> ```

---

## 持久化篇

### 1. RDB 和 AOF 有什么区别？

**答案：**

RDB（Redis Database）和 AOF（Append Only File）是 Redis 的两种持久化方式。

**对比：**

| 特性 | RDB | AOF |
|------|-----|-----|
| 存储内容 | 内存快照（二进制） | 写命令（文本） |
| 文件大小 | 小（压缩后） | 大（需重写） |
| 恢复速度 | 快 | 慢 |
| 数据安全性 | 可能丢失几分钟数据 | 最多丢失 1 秒 |
| 系统资源 | CPU 消耗大（fork+压缩） | 磁盘 IO 消耗大 |

**RDB 工作原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                       RDB 快照过程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  主进程                                                      │
│    │                                                         │
│    ├─── fork() ───→ 子进程                                   │
│    │                  │                                      │
│    │                  ├── 遍历内存                           │
│    ↓                  ├── 序列化数据                         │
│  继续处理请求         ├── 写入临时文件                       │
│                      ├── 替换旧 RDB 文件                     │
│                      └── 退出                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**AOF 工作原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                       AOF 写入过程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  写命令 → AOF 缓冲区 → fsync 策略 → AOF 文件                │
│                                                              │
│  fsync 策略:                                                │
│  - always: 每个命令都 fsync（最安全，最慢）                  │
│  - everysec: 每秒 fsync（推荐，平衡安全和性能）              │
│  - no: 由操作系统决定（最快，可能丢数据）                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**可能的追问：RDB 的 fork 为什么会阻塞？**

> fork 使用 **写时复制（Copy-On-Write）**：
>
> ```
> 初始状态：
> ┌─────────────────────┐
> │ 主进程页表 → 物理内存 │
> └─────────────────────┘
> 
> fork 后：
> ┌─────────────────────┐
> │ 主进程页表 ──┐       │
> │              ├──→ 物理内存（共享）
> │ 子进程页表 ──┘       │
> └─────────────────────┘
> 
> 主进程写入时：
> ┌─────────────────────┐
> │ 主进程页表 → 新物理页 │ (复制+修改)
> │ 子进程页表 → 原物理页 │ (快照数据)
> └─────────────────────┘
> ```
>
> 如果主进程写入量大，会产生大量内存复制，导致：
> 1. 内存占用翻倍（最坏情况）
> 2. 大量缺页中断
> 3. fork 本身也可能较慢（需要复制页表）

---

### 2. AOF 重写是什么？有什么作用？

**答案：**

AOF 重写压缩文件体积，去除冗余命令。

**重写原理：**

```
# 原始 AOF 文件（大量冗余）
SET counter 0
INCR counter
INCR counter
INCR counter
...
INCR counter      # 执行了 100 次 INCR

# 重写后（只保留最终状态）
SET counter 100
```

**重写过程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    AOF 重写流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 主进程 fork 子进程                                       │
│                                                              │
│  2. 子进程：                                                 │
│     - 遍历内存，生成新 AOF 文件（只记录最终状态）            │
│                                                              │
│  3. 主进程：                                                 │
│     - 继续处理请求                                           │
│     - 新写命令 → AOF 重写缓冲区                              │
│                                                              │
│  4. 子进程完成后：                                           │
│     - 主进程将重写缓冲区追加到新 AOF                         │
│     - 原子替换旧 AOF 文件                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**触发条件：**

```bash
# redis.conf
auto-aof-rewrite-percentage 100   # 文件大小比上次重写后增长 100%
auto-aof-rewrite-min-size 64mb    # 文件至少 64MB 才触发
```

**可能的追问：AOF 重写会阻塞吗？**

> 1. **fork 阶段**：会短暂阻塞（毫秒级）
> 2. **重写阶段**：子进程执行，不阻塞主进程
> 3. **信号处理阶段**：追加重写缓冲区，可能短暂阻塞
>
> 注意：如果重写期间写入量大，重写缓冲区可能堆积大量数据，最后追加阶段可能耗时较长。

---

### 3. 为什么需要混合持久化？

**答案：**

混合持久化结合 RDB 和 AOF 的优点，Redis 4.0 引入。

**三种持久化方案对比：**

```
┌─────────────────────────────────────────────────────────────┐
│  方案 1: 纯 RDB                                             │
│  优点: 恢复快，文件小                                        │
│  缺点: 可能丢失几分钟数据                                    │
├─────────────────────────────────────────────────────────────┤
│  方案 2: 纯 AOF                                             │
│  优点: 数据更安全（最多丢 1 秒）                             │
│  缺点: 恢复慢，文件大                                        │
├─────────────────────────────────────────────────────────────┤
│  方案 3: 混合持久化 (RDB + AOF)                             │
│  优点: RDB 恢复快 + AOF 增量数据安全                         │
│  缺点: 需要 Redis 4.0+                                       │
└─────────────────────────────────────────────────────────────┘
```

**混合持久化文件结构：**

```
┌─────────────────────────────────────────────────────────────┐
│                     混合 AOF 文件                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 RDB 格式数据                         │    │
│  │            （重写时的内存快照）                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 AOF 格式数据                         │    │
│  │       （重写期间的新增命令）                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**配置方式：**

```bash
# redis.conf
aof-use-rdb-preamble yes   # 开启混合持久化
```

**恢复流程：**

```python
def load_mixed_aof():
    # 1. 先加载 RDB 部分（快速恢复大部分数据）
    load_rdb_part()
    
    # 2. 再重放 AOF 增量部分
    replay_aof_incremental()
```

**可能的追问：生产环境如何选择持久化方案？**

> ```yaml
> 推荐方案:
>   - 开启 RDB 定时快照（每 5 分钟，且至少 100 个 key 变化）
>   - 开启 AOF（everysec）
>   - 开启混合持久化
>   
> 配置示例:
>   save 900 1      # 900秒内至少1个key变化
>   save 300 10     # 300秒内至少10个key变化
>   appendonly yes
>   appendfsync everysec
>   aof-use-rdb-preamble yes
> ```

---

### 4. 如何保证 RDB 快照期间的数据一致性？

**答案：**

RDB 使用 fork + 写时复制保证一致性。

**核心机制：**

```
┌─────────────────────────────────────────────────────────────┐
│                写时复制（Copy-On-Write）                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  快照开始时刻:                                               │
│  ┌───────────────────────────────────┐                      │
│  │ Key A = 1                         │                      │
│  │ Key B = 2                         │                      │
│  │ Key C = 3                         │                      │
│  └───────────────────────────────────┘                      │
│         ↓                                                    │
│  fork 后，父子进程共享物理内存                               │
│                                                              │
│  主进程修改 Key A = 100:                                     │
│  ┌───────────────────────────────────┐                      │
│  │ Key A = 100  ← 新页（主进程写）   │                      │
│  │ Key B = 2    ← 原页（共享）       │                      │
│  │ Key C = 3    ← 原页（共享）       │                      │
│  └───────────────────────────────────┘                      │
│         ↓                                                    │
│  子进程仍然看到快照时刻的数据:                               │
│  ┌───────────────────────────────────┐                      │
│  │ Key A = 1    ← 原页（子进程读）   │                      │
│  │ Key B = 2                         │                      │
│  │ Key C = 3                         │                      │
│  └───────────────────────────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**内存消耗：**

```
假设:
- Redis 内存: 8GB
- 快照期间写入量: 1GB

最坏情况内存消耗: 8GB + 1GB = 9GB

建议:
- 服务器内存应大于 Redis 内存 2 倍
- 避免快照期间大量写入
```

**可能的追问：如何减少 RDB 快照的影响？**

> 1. **错峰执行**：选择低峰期执行 BGSAVE
> 2. **减少写入**：快照期间减少数据变更
> 3. **内存预留**：预留足够的空闲内存
> 4. **监控告警**：监控 fork 耗时和内存使用

---

### 5. AOF 文件损坏如何恢复？

**答案：**

Redis 提供工具修复 AOF 文件。

**检查和修复：**

```bash
# 检查 AOF 文件完整性
redis-check-aof appendonly.aof

# 发现错误输出
# 0x         8c3a4: Expected prefix '*', got: 'S'

# 修复 AOF 文件
redis-check-aof --fix appendonly.aof

# 交互式确认
# This will shrink the AOF from 1024 bytes to 512 bytes
# Continue? [y/N]: y
```

**修复原理：**

```
AOF 文件格式:
*3          # 命令有 3 部分
$3          # 第一部分长度 3
SET         # 第一部分内容
$3          # 第二部分长度 3
key         # 第二部分内容
$5          # 第三部分长度 5
value       # 第三部分内容

如果中间损坏，修复工具会截断到损坏点之前
```

**预防措施：**

```bash
# 1. 定期备份 AOF
cp appendonly.aof appendonly.aof.backup

# 2. 开启混合持久化（RDB 部分更容易恢复）
aof-use-rdb-preamble yes

# 3. 监控 AOF 重写状态
redis-cli INFO persistence | grep aof
```

**可能的追问：RDB 文件损坏怎么办？**

> RDB 是二进制格式，损坏后难以修复：
>
> ```bash
> # 检查 RDB 文件
> redis-check-rdb dump.rdb
> 
> # 输出
> # [offset 0] Checking RDB file dump.rdb
> # [offset 26] AUX field redis-ver = '7.0.0'
> # [offset 133] CRC error
> ```
>
> RDB 损坏通常需要：
> 1. 使用备份恢复
> 2. 如果有 AOF，从 AOF 恢复

---

### 6. 生产环境持久化如何配置？

**答案：**

根据业务场景选择配置方案。

**方案一：高可靠（推荐）**

```bash
# redis.conf

# RDB 配置
save 900 1      # 15分钟内至少1个key变化
save 300 10     # 5分钟内至少10个key变化
save 60 10000   # 1分钟内至少10000个key变化

# AOF 配置
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-use-rdb-preamble yes

# 其他优化
stop-writes-on-bgsave-error yes
rdbcompression yes
```

**方案二：极致性能（可接受数据丢失）**

```bash
# 只开 RDB，较长间隔
save 900 1
appendonly no
```

**方案三：完全持久化**

```bash
# AOF everysec + RDB 兜底
appendonly yes
appendfsync everysec
save 900 1
aof-use-rdb-preamble yes
```

**监控指标：**

```bash
# 查看持久化状态
redis-cli INFO persistence

# 关键指标
rdb_last_bgsave_status:ok        # RDB 状态
rdb_last_bgsave_time_sec:0       # RDB 耗时
aof_enabled:1                    # AOF 开启
aof_last_bgrewrite_status:ok     # AOF 重写状态
aof_current_size:1024            # AOF 当前大小
```

**可能的追问：主从架构持久化如何配置？**

> - **主节点**：可以关闭持久化，专注写入性能
> - **从节点**：开启持久化，负责数据备份
>
> 但这样有风险：主节点重启后数据丢失，会同步空数据给从节点。建议主节点也开启 AOF（everysec）。

---

## 缓存问题篇

### 1. 什么是缓存穿透？如何解决？

**答案：**

缓存穿透是指查询不存在的数据，缓存和数据库都没有，导致每次请求都穿透到数据库。

**问题示意：**

```
┌─────────────────────────────────────────────────────────────┐
│                      缓存穿透流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  恶意请求: GET /user?id=-1                                   │
│       ↓                                                      │
│  ┌─────────┐                                                 │
│  │  Redis  │ → 不存在（key:-1 从未存过）                     │
│  └─────────┘                                                 │
│       ↓                                                      │
│  ┌─────────┐                                                 │
│  │  MySQL  │ → 不存在                                        │
│  └─────────┘                                                 │
│       ↓                                                      │
│  返回 null（不缓存）                                         │
│       ↓                                                      │
│  下次同样请求 → 再次查 MySQL                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

**1. 缓存空对象**

```java
public User getUser(Long id) {
    String key = "user:" + id;
    
    // 1. 查缓存
    String value = redis.get(key);
    if (value != null) {
        return "NULL".equals(value) ? null : JSON.parse(value);
    }
    
    // 2. 查数据库
    User user = userDao.findById(id);
    
    // 3. 缓存结果（包括空值）
    if (user != null) {
        redis.setex(key, 3600, JSON.stringify(user));
    } else {
        // 缓存空对象，设置较短过期时间
        redis.setex(key, 60, "NULL");
    }
    
    return user;
}
```

**2. 布隆过滤器**

```java
// 启动时预热所有有效 ID 到布隆过滤器
public void initBloomFilter() {
    BloomFilter<Long> filter = BloomFilter.create(Funnels.longFunnel(), 1000000, 0.01);
    List<Long> allIds = userDao.findAllIds();
    for (Long id : allIds) {
        filter.put(id);
    }
    // 存入 Redis（RedisBloom 模块）
}

public User getUserWithBloom(Long id) {
    // 1. 先判断是否可能存在
    if (!bloomFilter.mightContain(id)) {
        return null;  // 一定不存在，直接返回
    }
    
    // 2. 查缓存、查数据库
    return getUser(id);
}
```

**方案对比：**

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 缓存空对象 | 实现简单 | 占用内存、可能有数据不一致 | 空值较少 |
| 布隆过滤器 | 内存占用小 | 有误判率、实现复杂 | 数据量大、空值多 |

**可能的追问：布隆过滤器为什么会有误判？**

> 布隆过滤器使用多个哈希函数映射到位数组：
>
> ```
> 元素 x 通过 3 个哈希函数映射到位数组
> 
> h1(x) → bit[5] = 1
> h2(x) → bit[17] = 1
> h3(x) → bit[31] = 1
> 
> 查询 y 时，如果 bit[5], bit[17], bit[31] 都为 1，
> 可能是 y 确实存在，也可能是其他元素恰好设置了这些位
> 
> 误判率 ≈ (1 - e^(-kn/m))^k
> 
> m: 位数组大小
> n: 元素数量
> k: 哈希函数数量
> ```

---

### 2. 什么是缓存击穿？如何解决？

**答案：**

缓存击穿是指热点 key 过期瞬间，大量请求同时穿透到数据库。

**问题示意：**

```
┌─────────────────────────────────────────────────────────────┐
│                      缓存击穿场景                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  热点 key: "hot:product:1" (QPS 10000)                      │
│                                                              │
│  T1 时刻: key 过期                                           │
│       ↓                                                      │
│  10000 个请求同时到达                                        │
│       ↓                                                      │
│  ┌─────────┐                                                 │
│  │  Redis  │ → 全部 miss                                     │
│  └─────────┘                                                 │
│       ↓                                                      │
│  10000 个请求同时查 MySQL                                    │
│       ↓                                                      │
│  MySQL 瞬间被打挂                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

**1. 互斥锁（推荐）**

```java
public User getHotData(Long id) {
    String key = "hot:" + id;
    String lockKey = "lock:" + id;
    
    // 1. 查缓存
    String value = redis.get(key);
    if (value != null) {
        return JSON.parse(value);
    }
    
    // 2. 尝试获取锁
    try {
        boolean locked = redis.setnx(lockKey, "1", 10);  // 10秒超时
        if (locked) {
            // 获取锁成功，查数据库
            User user = userDao.findById(id);
            redis.setex(key, 3600, JSON.stringify(user));
            return user;
        } else {
            // 获取锁失败，等待并重试
            Thread.sleep(50);
            return getHotData(id);  // 递归重试
        }
    } finally {
        redis.del(lockKey);
    }
}
```

**2. 永不过期 + 逻辑过期**

```java
// 数据结构
class CacheData {
    Object data;
    long expireTime;  // 逻辑过期时间
}

public Object getWithLogicalExpire(String key) {
    CacheData cached = redis.get(key);
    
    // 未过期，直接返回
    if (cached != null && cached.expireTime > System.currentTimeMillis()) {
        return cached.data;
    }
    
    // 已过期，异步刷新
    if (cached != null) {
        executor.submit(() -> {
            // 重建缓存
            Object data = loadData();
            CacheData newData = new CacheData(data, System.currentTimeMillis() + 3600000);
            redis.set(key, newData);
        });
        return cached.data;  // 返回旧数据
    }
    
    // 缓存不存在，同步加载
    return loadAndCache(key);
}
```

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 互斥锁 | 强一致，实现简单 | 有等待延迟 |
| 永不过期 | 无穿透风险 | 可能返回脏数据 |
| 逻辑过期 | 性能好 | 实现复杂 |

**可能的追问：如何识别热点 key？**

> ```java
> // 方案1: 客户端统计
> Map<String, AtomicLong> counter = new ConcurrentHashMap<>();
> 
> // 方案2: Redis MONITOR 命令分析
> redis-cli MONITOR | head -n 100000 | grep -oE '"GET "[^"]+"' | sort | uniq -c
> 
> // 方案3: 使用 Redis 的 CLIENT LIST + INFO stats
> // 方案4: 专门的 hotspot 检测系统
> ```

---

### 3. 什么是缓存雪崩？如何解决？

**答案：**

缓存雪崩是指大量缓存同时失效或 Redis 宕机，所有请求直接打到数据库。

**问题场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                      缓存雪崩场景                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  场景1: 大量 key 同时过期                                   │
│  ────────────────────────                                   │
│  某活动 0 点开始，设置了大量 key 过期时间为 24 小时          │
│  第二天 0 点，大量 key 同时过期                             │
│                                                              │
│  场景2: Redis 宕机                                          │
│  ────────────────────────                                   │
│  Redis 节点故障，所有请求直接访问数据库                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

**1. 过期时间随机化**

```java
// 避免同时过期
int baseExpire = 3600;  // 基础过期时间 1 小时
int randomExpire = ThreadLocalRandom.current().nextInt(0, 600);  // 随机 0-10 分钟
redis.setex(key, baseExpire + randomExpire, value);
```

**2. 多级缓存**

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│   应用    │ ──→ │ 本地缓存  │ ──→ │  Redis    │ ──→ │  MySQL    │
│  缓存     │     │ (Caffeine)│     │  集群     │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘

多级缓存配置:
- L1: 本地缓存（Caffeine），过期时间短
- L2: Redis 缓存，过期时间长
```

**3. 服务降级和熔断**

```java
// 使用 Hystrix/Sentinel 熔断
@HystrixCommand(fallbackMethod = "getFallback")
public User getUser(Long id) {
    return cacheService.get(id);
}

public User getFallback(Long id) {
    // 返回默认值或降级数据
    return User.defaultUser();
}
```

**4. Redis 高可用**

```yaml
架构方案:
  - 主从复制: 数据冗余
  - 哨兵模式: 自动故障转移
  - Redis Cluster: 分片 + 高可用
```

**可能的追问：如何快速恢复雪崩后的系统？**

> 1. **快速恢复 Redis**：重启 Redis，从 RDB/AOF 恢复
> 2. **限流降级**：启动限流，保护数据库
> 3. **预热缓存**：逐步预热热点数据
> 4. **持久层优化**：临时关闭非核心业务查询

---

### 4. 如何保证缓存和数据库的一致性？

**答案：**

缓存和数据库一致性是分布式系统的经典问题，需要权衡一致性要求和性能。

**常见方案对比：**

```
┌─────────────────────────────────────────────────────────────┐
│                    一致性方案对比                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  方案1: 先更新数据库，再更新缓存                             │
│  ────────────────────────────────                           │
│  问题: 并发时可能产生脏数据                                  │
│        线程A更新DB→线程B更新DB→线程B更新缓存→线程A更新缓存  │
│        结果: 缓存是旧值                                      │
│                                                              │
│  方案2: 先删除缓存，再更新数据库                             │
│  ────────────────────────────────                           │
│  问题: 读写并发时，可能缓存脏数据                            │
│        线程A删缓存→线程B读DB(旧值)→线程B写缓存→线程A更新DB  │
│        结果: 缓存是旧值                                      │
│                                                              │
│  方案3: 先更新数据库，再删除缓存（推荐）                     │
│  ────────────────────────────────                           │
│  问题: 删除缓存失败，导致不一致                              │
│        解决: 重试机制或消息队列保证最终一致                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**推荐方案实现：**

```java
// 基础版本
public void updateUser(User user) {
    // 1. 更新数据库
    userDao.update(user);
    
    // 2. 删除缓存
    redis.del("user:" + user.getId());
}

// 增强版: 消息队列保证最终一致
public void updateUser(User user) {
    // 1. 更新数据库
    userDao.update(user);
    
    // 2. 发送删除消息
    mq.send("cache-delete", "user:" + user.getId());
}

// 消费者
@Consumer(topic = "cache-delete")
public void onMessage(String key) {
    redis.del(key);
    // 删除失败可以重试
}
```

**使用 Canal 订阅 Binlog：**

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  应用   │ ──→ │  MySQL  │ ──→ │  Canal  │ ──→ │  Redis  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                    │
                    └── Binlog 同步到 Canal Server
                             │
                             └── Canal Client 消费并更新缓存
```

**可能的追问：为什么是删除缓存而不是更新缓存？**

> 1. **避免并发问题**：两个线程同时更新，可能后更新的覆盖先更新的
> 2. **延迟双删问题**：删除更简单，下次查询时自动重建
> 3. **节省资源**：如果数据频繁更新但很少读取，更新缓存是浪费
> 4. **复杂计算**：有些缓存值计算成本高，删除后按需计算

---

### 5. 什么是延迟双删？

**答案：**

延迟双删解决"先删缓存再更新数据库"场景下的一致性问题。

**问题场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                 先删缓存的问题                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  线程A: 删除缓存 key                                        │
│     ↓                                                        │
│  线程B: 读缓存 miss                                         │
│     ↓                                                        │
│  线程B: 读数据库（旧值）                                     │
│     ↓                                                        │
│  线程B: 写缓存（旧值）                                       │
│     ↓                                                        │
│  线程A: 更新数据库（新值）                                   │
│     ↓                                                        │
│  结果: 缓存是旧值，数据库是新值                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**延迟双删方案：**

```java
public void updateWithDoubleDelete(Long id, User user) {
    // 1. 第一次删除缓存
    redis.del("user:" + id);
    
    // 2. 更新数据库
    userDao.update(user);
    
    // 3. 延迟一段时间后，再次删除缓存
    executor.schedule(() -> {
        redis.del("user:" + id);
    }, 500, TimeUnit.MILLISECONDS);  // 延迟 500ms
}

// 时序：
// T0: 删除缓存
// T1: 更新数据库
// T500ms: 再次删除缓存（此时线程B已将旧值写入缓存）
```

**延迟时间如何确定？**

```java
// 延迟时间 > 读操作耗时 + 几十毫秒缓冲
// 一般设置 500ms - 1000ms

// 更可靠的方案：监听 Binlog 删除
```

**可能的追问：延迟双删有什么缺点？**

> 1. **延迟时间难以确定**：太短可能无效，太长影响一致性
> 2. **增加复杂度**：需要异步任务调度
> 3. **第二次删除仍可能失败**
> 4. **不能保证强一致**：只是降低不一致的概率
>
> 更推荐：先更新数据库 + 删除缓存 + 消息队列重试

---

### 6. 布隆过滤器原理是什么？

**答案：**

布隆过滤器是一种空间效率很高的概率型数据结构，用于判断元素是否在集合中。

**数据结构：**

```
┌─────────────────────────────────────────────────────────────┐
│                    布隆过滤器结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  位数组（m 个 bit，初始全为 0）                              │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐         │
│  │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ ...     │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘         │
│    0   1   2   3   4   5   6   7   8   9  10  11           │
│                                                              │
│  添加元素 "Redis":                                           │
│  h1("Redis") = 2   → bit[2] = 1                             │
│  h2("Redis") = 5   → bit[5] = 1                             │
│  h3("Redis") = 9   → bit[9] = 1                             │
│                                                              │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐         │
│  │ 0 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ 0 │ 0 │ 1 │ 0 │ 0 │ ...     │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘         │
│          ↑           ↑           ↑                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**查询过程：**

```python
def might_contain(element):
    for i in range(k):  # k 个哈希函数
        position = hash_functions[i](element) % m
        if bit_array[position] == 0:
            return False  # 一定不存在
    return True  # 可能存在（可能误判）
```

**特性：**

```
┌─────────────────────────────────────────────────────────────┐
│  特性          │  说明                                      │
├────────────────┼────────────────────────────────────────────┤
│  空间效率      │  一个元素约 10-20 bit                      │
│  时间效率      │  O(k)，k 是哈希函数数量                    │
│  不存在误判    │  返回 False 则一定不存在                   │
│  存在误判      │  返回 True 可能实际不存在                  │
│  不可删除      │  删除可能影响其他元素                      │
└─────────────────────────────────────────────────────────────┘
```

**Redis 中使用布隆过滤器：**

```bash
# RedisBloom 模块
BF.ADD myfilter item1
BF.EXISTS myfilter item1    # 返回 1（可能存在）
BF.EXISTS myfilter item2    # 返回 0（一定不存在）

# Java 客户端使用 Guava
BloomFilter<String> filter = BloomFilter.create(
    Funnels.stringFunnel(Charset.defaultCharset()),
    1000000,  // 预期元素数量
    0.01      // 误判率 1%
);
```

**可能的追问：如何降低误判率？**

> ```python
> # 误判率公式
> p = (1 - e^(-kn/m))^k
> 
> # 最优哈希函数数量
> k = (m/n) * ln(2)
> 
> # 位数组大小
> m = -n * ln(p) / (ln(2))^2
> 
> # 示例：100万元素，1% 误判率
> # m ≈ 9.6M bit ≈ 1.2MB
> # k ≈ 7 个哈希函数
> 
> # 降低误判率方法：
> # 1. 增加位数组大小
> # 2. 增加哈希函数数量（但会增加计算开销）
> # 3. 使用布隆过滤器变体（如布隆计数器）
> ```

---

### 7. 如何实现缓存预热？

**答案：**

缓存预热是在系统启动或低峰期，提前加载热点数据到缓存。

**实现方案：**

**1. 启动时预热**

```java
@PostConstruct
public void warmup() {
    // 加载热点数据
    List<Product> hotProducts = productService.getHotProducts(1000);
    for (Product p : hotProducts) {
        redis.setex("product:" + p.getId(), 3600, JSON.stringify(p));
    }
    
    // 预热布隆过滤器
    List<Long> allIds = productService.getAllIds();
    for (Long id : allIds) {
        bloomFilter.put(id);
    }
}
```

**2. 定时任务预热**

```java
@Scheduled(cron = "0 0 4 * * ?")  // 每天凌晨 4 点
public void scheduledWarmup() {
    // 分析昨日热点
    List<String> hotKeys = analyzeHotKeys();
    
    // 批量预热
    for (String key : hotKeys) {
        if (!redis.exists(key)) {
            Object data = loadFromDB(key);
            redis.setex(key, 3600, data);
        }
    }
}
```

**3. 灰度发布预热**

```java
public void warmupForNewVersion() {
    // 新版本上线前，预热缓存
    // 1. 识别热点接口
    // 2. 批量调用接口填充缓存
    // 3. 监控缓存命中率
    
    List<String> apis = Arrays.asList(
        "/api/product/list",
        "/api/product/detail",
        "/api/user/info"
    );
    
    for (String api : apis) {
        httpGet("http://localhost:8080" + api);
    }
}
```

**批量预热脚本：**

```bash
#!/bin/bash
# Redis 批量预热脚本

# 从 MySQL 导出热点数据
mysql -e "SELECT id FROM products WHERE is_hot=1" | while read id; do
    # 查询并写入 Redis
    data=$(mysql -N -e "SELECT JSON_OBJECT('id', id, 'name', name, 'price', price) FROM products WHERE id=$id")
    redis-cli SET "product:$id" "$data"
done

echo "Cache warmup completed!"
```

**可能的追问：预热时要注意什么？**

> 1. **避免阻塞**：分批预热，控制并发
> 2. **错峰执行**：选择低峰期，减少对业务影响
> 3. **监控进度**：记录预热完成度
> 4. **异常处理**：预热失败不影响系统启动

---

### 8. 如何防止缓存被恶意刷？

**答案：**

恶意刷缓存可能导致 Redis 内存耗尽、带宽耗尽。

**防护措施：**

**1. 接口限流**

```java
// 使用 Guava RateLimiter
RateLimiter limiter = RateLimiter.create(100);  // 每秒 100 个请求

@GetMapping("/product/{id}")
public Product getProduct(@PathVariable Long id) {
    if (!limiter.tryAcquire()) {
        throw new RateLimitException("请求过于频繁");
    }
    return productService.getProduct(id);
}

// 或使用 Redis + Lua 实现分布式限流
```

**2. 请求签名验证**

```java
// 客户端：timestamp + nonce + sign
// sign = MD5(appKey + timestamp + nonce + appSecret)

public boolean verifySign(Request request) {
    long timestamp = request.getTimestamp();
    String nonce = request.getNonce();
    String sign = request.getSign();
    
    // 1. 检查时间戳（5 分钟内有效）
    if (System.currentTimeMillis() - timestamp > 300000) {
        return false;
    }
    
    // 2. 检查 nonce 是否重复（防重放）
    if (redis.exists("nonce:" + nonce)) {
        return false;
    }
    redis.setex("nonce:" + nonce, 300, "1");
    
    // 3. 验证签名
    String expectedSign = MD5(appKey + timestamp + nonce + appSecret);
    return expectedSign.equals(sign);
}
```

**3. IP 黑白名单**

```java
public boolean checkIp(String ip) {
    // 白名单优先
    if (isInWhitelist(ip)) {
        return true;
    }
    
    // 检查黑名单
    if (redis.sismember("blacklist:ip", ip)) {
        return false;
    }
    
    // 检查访问频率
    String key = "access:" + ip;
    long count = redis.incr(key);
    if (count == 1) {
        redis.expire(key, 60);  // 1 分钟窗口
    }
    
    // 超过阈值加入黑名单
    if (count > 1000) {
        redis.sadd("blacklist:ip", ip);
        return false;
    }
    
    return true;
}
```

**4. 关键参数校验**

```java
public Product getProduct(Long id) {
    // 基础校验
    if (id == null || id <= 0) {
        return null;
    }
    
    // 布隆过滤器校验（防止恶意查询不存在的 ID）
    if (!bloomFilter.mightContain(id)) {
        return null;
    }
    
    return productService.getProduct(id);
}
```

**可能的追问：如何检测异常流量？**

> ```java
> // 监控指标
> // 1. 单 IP 请求频率
> // 2. 缓存命中率异常下降
> // 3. 请求的 key 分布异常
> 
> // 使用 Redis 统计
> long uniqueKeys = redis.pfcount("request:keys:today");
> long totalRequests = redis.get("request:total:today");
> 
> // 如果 uniqueKeys/totalRequests 比例异常，可能是恶意刷
> ```

---

## 高可用篇

### 1. Redis 主从复制原理是什么？

**答案：**

主从复制实现数据冗余和读写分离，提高可用性。

**复制流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    主从复制流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Master                        Slave                        │
│    │                             │                          │
│    │  1. SYNC/PSYNC 命令        │                          │
│    │ ←───────────────────────────│                          │
│    │                             │                          │
│    │  2. 执行 BGSAVE             │                          │
│    │     生成 RDB 快照           │                          │
│    │                             │                          │
│    │  3. 发送 RDB 文件           │                          │
│    │ ────────────────────────────→                          │
│    │                             │                          │
│    │  4. 发送缓冲区增量命令       │                          │
│    │ ────────────────────────────→                          │
│    │                             │                          │
│    │  5. 后续写命令持续同步       │                          │
│    │ ────────────────────────────→                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**PSYNC（部分重同步）：**

```c
// 复制偏移量
struct replication {
    long long master_repl_offset;    // 主节点偏移量
    char replid[41];                 // 主节点运行 ID
};

// 从节点断线重连
// 1. 发送 PSYNC <replid> <offset>
// 2. 主节点判断：
//    - replid 匹配且 offset 在复制积压缓冲区内 → 部分重同步
//    - 否则 → 全量同步

// 复制积压缓冲区（FIFO 队列）
// 存储最近执行的写命令
backlog_size = 1MB;  // 默认大小
```

**配置示例：**

```bash
# 从节点配置
replicaof 192.168.1.100 6379

# 只读模式（推荐开启）
replica-read-only yes

# 复制积压缓冲区大小
repl-backlog-size 1mb
```

**可能的追问：主从复制延迟如何监控？**

> ```bash
> # 查看复制信息
> redis-cli INFO replication
> 
> # 关键指标
> master_repl_offset: 10000     # 主节点偏移量
> slave0: offset=9500,lag=1     # 从节点偏移量和延迟秒数
> 
> # 监控延迟
> redis-cli --latency -h slave_ip
> ```

---

### 2. Redis 哨兵模式是如何工作的？

**答案：**

哨兵（Sentinel）监控主从节点，自动故障转移。

**哨兵架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                     哨兵架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│           ┌───────┐    ┌───────┐    ┌───────┐              │
│           │Sentinel│   │Sentinel│   │Sentinel│              │
│           │   1   │   │   2   │   │   3   │              │
│           └───┬───┘    └───┬───┘    └───┬───┘              │
│               │            │            │                   │
│               └────────────┼────────────┘                   │
│                            │                                │
│              ┌─────────────┼─────────────┐                  │
│              ↓             ↓             ↓                  │
│         ┌────────┐   ┌────────┐   ┌────────┐               │
│         │ Master │   │ Slave1 │   │ Slave2 │               │
│         └────────┘   └────────┘   └────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**故障转移流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    故障转移流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 主观下线（SDOWN）                                        │
│     单个哨兵检测到主节点无响应                               │
│                                                              │
│  2. 客观下线（ODOWN）                                        │
│     足够多哨兵（quorum）确认主节点下线                       │
│                                                              │
│  3. 选举领头哨兵                                             │
│     使用 Raft 算法选举                                       │
│                                                              │
│  4. 选举新主节点                                             │
│     - 排除下线节点                                           │
│     - 选择复制偏移量最大的从节点                             │
│     - 优先选择优先级高的节点                                 │
│                                                              │
│  5. 故障转移                                                 │
│     - 新主节点执行 SLAVEOF NO ONE                           │
│     - 其他从节点复制新主节点                                 │
│     - 更新配置                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**配置示例：**

```bash
# sentinel.conf
port 26379
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 60000
```

**可能的追问：哨兵如何发现从节点和其他哨兵？**

> ```bash
> # 哨兵通过以下方式发现：
> 
> # 1. 发现从节点
> #    主节点返回从节点列表
> SENTINEL REPLICAS mymaster
> 
> # 2. 发现其他哨兵
> #    通过发布/订阅频道 _sentinel_:hello
> #    每个哨兵定期发布自己的信息
> 
> # 3. 客户端订阅频道获取主节点变化
> SENTINEL GET-MASTER-ADDR-BY-NAME mymaster
> ```

---

### 3. Redis Cluster 是什么？如何工作？

**答案：**

Redis Cluster 是 Redis 的分布式解决方案，支持自动分片和高可用。

**集群架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                   Redis Cluster 架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  槽位范围: 0-16383 (共 16384 个槽)                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │          Node1          │          Node2               ││
│  │   槽位: 0-5461          │   槽位: 5462-10922           ││
│  │   ┌─────┐    ┌─────┐   │   ┌─────┐    ┌─────┐        ││
│  │   │Master│───│Slave│   │   │Master│───│Slave│        ││
│  │   └─────┘    └─────┘   │   └─────┘    └─────┘        ││
│  ├─────────────────────────┼─────────────────────────────┤│
│  │          Node3          │          Node4              ││
│  │   槽位: 10923-16383     │   槽位: 分配中              ││
│  │   ┌─────┐    ┌─────┐   │   ┌─────┐    ┌─────┐        ││
│  │   │Master│───│Slave│   │   │Master│───│Slave│        ││
│  │   └─────┘    └─────┘   │   └─────┘    └─────┘        ││
│  └─────────────────────────┴─────────────────────────────┘│
│                                                              │
│  Gossip 协议: 节点间通信，交换状态信息                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**槽位分配：**

```bash
# 计算键所属槽位
CRC16(key) % 16384

# 示例
CRC16("user:1001") % 16384 = 12540  # 属于 Node3
```

**MOVED 重定向：**

```bash
# 客户端请求错误的节点
GET user:1001
# 返回 MOVED 12540 192.168.1.103:6379

# 客户端重新请求正确的节点
GET user:1001
# 返回 "张三"
```

**集群配置：**

```bash
# redis.conf
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
cluster-require-full-coverage yes  # 槽位不全时是否拒绝服务
```

**创建集群：**

```bash
# 创建 3 主 3 从集群
redis-cli --cluster create \
  192.168.1.101:6379 \
  192.168.1.102:6379 \
  192.168.1.103:6379 \
  192.168.1.104:6379 \
  192.168.1.105:6379 \
  192.168.1.106:6379 \
  --cluster-replicas 1
```

**可能的追问：集群模式下如何处理批量操作？**

> ```bash
> # 问题：MSET/MGET 要求所有 key 在同一槽位
> MSET user:1 a user:2 b  # 可能报错
> 
> # 解决：使用 hash tag
> MSET user:{1}:name a user:{1}:age 20  # {1} 保证同一槽位
> 
> # 原理：只计算 {} 内的部分
> CRC16("{1}") % 16384 = 固定值
> ```

---

### 4. 一致性哈希是什么？

**答案：**

一致性哈希是分布式系统的数据分片策略，解决节点增减时的数据迁移问题。

**普通哈希 vs 一致性哈希：**

```
┌─────────────────────────────────────────────────────────────┐
│  普通哈希: hash(key) % n                                    │
│  问题: 节点数变化时，大量数据需要迁移                        │
│  3节点 → 4节点: 约 75% 数据迁移                              │
├─────────────────────────────────────────────────────────────┤
│  一致性哈希:                                                │
│  1. 将哈希值空间组织成环（0 ~ 2^32-1）                      │
│  2. 节点和数据都映射到环上                                  │
│  3. 数据顺时针找到最近的节点                                │
│  优点: 节点增减只影响相邻节点的数据                          │
│  3节点 → 4节点: 约 25% 数据迁移                              │
└─────────────────────────────────────────────────────────────┘
```

**哈希环示意：**

```
                    0
                    │
         Node A ────┼──── Node D
            │       │       │
            │       │       │
     key1   │       │       │   key3
        │   │       │       │   │
        ▼   │       │       │   ▼
       ╭────────────────────────────╮
       │     key2      key4        │
       │                            │
       │       哈希环              │
       │                            │
       ╰────────────────────────────╯
                    │
         Node B ────┴──── Node C

key1 → 顺时针找到 Node A
key2 → 顺时针找到 Node B
key3 → 顺时针找到 Node C
key4 → 顺时针找到 Node B
```

**虚拟节点解决数据倾斜：**

```
问题: 节点少时，数据分布不均匀

解决: 每个物理节点映射多个虚拟节点

Node A → Node A#1, Node A#2, Node A#3, ...
Node B → Node B#1, Node B#2, Node B#3, ...

虚拟节点均匀分布在环上，使数据分布更均匀
```

**可能的追问：Redis Cluster 为什么不用一致性哈希？**

> Redis Cluster 使用 **哈希槽** 而非一致性哈希：
>
> | 对比 | 一致性哈希 | 哈希槽 |
> |------|-----------|--------|
> | 数据迁移 | 需要计算每个 key 的新位置 | 只需迁移对应槽位 |
> | 管理性 | 虚拟节点复杂 | 槽位可手动分配 |
> | 查找效率 | O(log n) 二分查找 | O(1) 直接定位 |
> 
> 哈希槽更适合 Redis 的场景，因为：
> 1. 槽位数量固定，易于管理
> 2. 支持手动调整槽位分配
> 3. 客户端可以缓存槽位映射

---

### 5. Redis 主从切换会导致数据丢失吗？

**答案：**

主从切换可能丢失数据，需要理解原因并配置优化。

**数据丢失场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                    数据丢失原因                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 异步复制延迟                                             │
│     ───────────────                                         │
│     主节点写入 → 返回成功 → 复制到从节点前主节点宕机         │
│     结果: 从节点未收到数据                                   │
│                                                              │
│  2. 脑裂（Split Brain）                                      │
│     ─────────────────                                       │
│     网络分区导致主节点被隔离                                 │
│     哨兵选举新主节点                                         │
│     原主节点恢复后成为从节点，数据丢失                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**脑裂示意：**

```
          网络分区
             │
    ┌────────┴────────┐
    │                 │
┌───┴───┐         ┌───┴───┐
│ 旧主  │         │ 新主  │
│ (隔离)│         │(选举) │
└───────┘         └───────┘
    │
    └── 期间接收的写入，恢复后会丢失
```

**配置优化：**

```bash
# redis.conf

# 最少从节点数
# 主节点至少有 N 个从节点且延迟小于 M 秒才能写入
min-replicas-to-write 1
min-replicas-max-lag 10

# 效果：
# - 减少异步复制丢失风险
# - 防止脑裂（隔离的主节点无法写入）
```

**数据丢失量化：**

```bash
# 查看复制偏移量差异
redis-cli INFO replication

# master_repl_offset: 10000
# slave0: offset=9900

# 最大丢失 = master_repl_offset - min(slave_offset)
# 上例: 10000 - 9900 = 100 条命令
```

**可能的追问：如何保证强一致性？**

> Redis 默认是最终一致性。要实现强一致性：
>
> 1. **WAIT 命令**：等待复制完成
> ```bash
> SET key value
> WAIT 1 5000  # 等待至少 1 个从节点确认，超时 5 秒
> ```
>
> 2. **同步复制**（性能影响大）
> ```bash
> # 每个写命令都等待复制
> ```
>
> 3. **使用其他方案**：如 Raft 协议的 etcd/Consul

---

### 6. 如何监控 Redis 集群状态？

**答案：**

全面的监控是保障 Redis 稳定的关键。

**关键监控指标：**

```
┌─────────────────────────────────────────────────────────────┐
│                    监控指标分类                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  性能指标                                                    │
│  ├── instantaneous_ops_per_sec: 当前 QPS                    │
│  ├── latency: 延迟                                          │
│  └── hit_rate: 缓存命中率                                   │
│                                                              │
│  内存指标                                                    │
│  ├── used_memory: 已用内存                                  │
│  ├── used_memory_peak: 内存峰值                             │
│  ├── mem_fragmentation_ratio: 内存碎片率                    │
│  └── evicted_keys: 被淘汰的 key 数量                        │
│                                                              │
│  持久化指标                                                  │
│  ├── rdb_last_bgsave_status: RDB 状态                       │
│  ├── aof_last_bgrewrite_status: AOF 重写状态                │
│  └── aof_delayed_fsync: AOF 延迟 fsync 次数                 │
│                                                              │
│  集群指标                                                    │
│  ├── cluster_state: 集群状态 (ok/fail)                      │
│  ├── cluster_slots_assigned: 分配的槽位数                   │
│  └── cluster_known_nodes: 已知节点数                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**常用监控命令：**

```bash
# 基本信息
redis-cli INFO server
redis-cli INFO memory
redis-cli INFO stats
redis-cli INFO replication

# 实时监控命令
redis-cli MONITOR  # 注意：影响性能

# 延迟监控
redis-cli --latency
redis-cli --latency-history

# 慢查询日志
redis-cli SLOWLOG GET 10

# 客户端连接
redis-cli CLIENT LIST
```

**Prometheus + Grafana 监控：**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

# 使用 redis_exporter
docker run -d --name redis-exporter \
  -e REDIS_ADDR=redis://redis:6379 \
  -p 9121:9121 \
  oliver006/redis_exporter
```

**告警规则示例：**

```yaml
# alert.rules
groups:
  - name: redis
    rules:
      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"

      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage > 90%"

      - alert: RedisHighLatency
        expr: redis_latency_seconds > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis latency > 100ms"
```

**可能的追问：如何排查 Redis 性能问题？**

> 排查思路：
> 1. **检查慢查询日志**：`SLOWLOG GET`
> 2. **检查大 Key**：`redis-cli --bigkeys`
> 3. **检查内存碎片**：INFO memory
> 4. **检查延迟**：`--latency`
> 5. **检查客户端连接**：CLIENT LIST
> 6. **检查持久化状态**：INFO persistence
> 7. **使用 Redis SLOWLOG 分析命令**

---

### 7. Redis 大 Key 如何发现和处理？

**答案：**

大 Key 是指占用大量内存的 key，可能导致阻塞和性能问题。

**大 Key 定义：**

```yaml
String 类型: value 超过 10KB
Hash 类型: 元素超过 5000 个
List 类型: 元素超过 10000 个
Set 类型: 元素超过 10000 个
ZSet 类型: 元素超过 10000 个
```

**发现大 Key：**

```bash
# 方法1: redis-cli --bigkeys
redis-cli --bigkeys

# 方法2: 使用 debug object
redis-cli DEBUG OBJECT key

# 方法3: 使用 memory usage（Redis 4.0+）
redis-cli MEMORY USAGE key

# 方法4: 使用 RDB 分析工具
redis-rdb-tools --command json dump.rdb
```

**处理大 Key：**

```bash
# 1. 删除大 Key（不要直接 DEL）
# String 大 Key: UNLINK
UNLINK big_string_key

# Hash 大 Key: 使用 hscan 渐进删除
hscan big_hash 0
hdel big_hash field1 field2 ...
# 或使用 Lua 脚本

# List 大 Key: 使用 ltrim
LTRIM big_list 0 1000  # 只保留前 1000 个
# 或 lpop/rpop 逐步删除

# 2. 拆分大 Key
# 原: big_hash: {field1, field2, ..., field10000}
# 拆: hash1:{field1-1000}, hash2:{field1001-2000}, ...
```

**Lua 脚本渐进删除：**

```lua
-- 渐进删除 Hash 大 Key
local key = KEYS[1]
local cursor = ARGV[1]
local batch = 100

local reply = redis.call('HSCAN', key, cursor, 'COUNT', batch)
cursor = reply[1]
local fields = reply[2]

if #fields > 0 then
    redis.call('HDEL', key, unpack(fields))
end

if cursor == '0' then
    return 'DONE'
else
    return cursor
end
```

**可能的追问：大 Key 有什么危害？**

> 1. **内存不均衡**：集群模式下，大 Key 导致某节点内存占用高
> 2. **阻塞操作**：删除大 Key 会阻塞主线程
> 3. **网络拥塞**：读取大 Key 占用带宽
> 4. **过期删除慢**：大 Key 过期删除耗时长
> 5. **迁移困难**：大 Key 迁移耗时，可能导致迁移失败

---

### 8. Redis 如何做容量规划？

**答案：**

合理的容量规划保障 Redis 稳定运行。

**规划公式：**

```
总内存需求 = 数据内存 + 预留内存 + 内存碎片

详细计算:
├── 数据内存
│   ├── key 内存: 每个 key 约 40-80 字节（包含 Redis 对象开销）
│   └── value 内存: 根据数据类型计算
│
├── 预留内存
│   ├── 复制缓冲区: 约 100MB
│   ├── 客户端缓冲区: 正常客户端各约 1MB
│   └── AOF 缓冲区: 约 100MB
│
└── 内存碎片
    └── 碎片率通常 1.0-1.5，预留 50%
```

**内存估算示例：**

```python
# 假设存储 100 万个用户信息（Hash 结构）

# 每个 key: user:{id}
key_memory = 1000000 * 40  # 40MB

# 每个 value: Hash 包含 10 个字段，每字段平均 50 字节
# Hash 内存 = 元素数 * (字段名 + 字段值 + 指针开销)
value_memory = 1000000 * 10 * (20 + 50 + 16)  # 860MB

# 总数据内存
data_memory = key_memory + value_memory  # 900MB

# 考虑碎片和预留
total_memory = data_memory * 1.5 + 500MB  # 约 1.85GB

# 建议: 4GB 服务器
```

**QPS 规划：**

```
单 Redis 实例 QPS 能力: ~10 万（简单操作）

实际 QPS 规划:
├── 复杂命令: 降低到 ~5 万
├── 大 Value: 降低到 ~3 万
└── 预留缓冲: 建议 70% 负载

集群 QPS = 单节点 QPS × 主节点数
```

**集群规模规划：**

```yaml
数据量: 100GB
单节点内存: 32GB
复制因子: 2（主从复制）

计算:
├── 可用内存/节点: 32GB * 0.6 = 19.2GB（预留 40%）
├── 主节点数: 100GB / 19.2GB ≈ 6
├── 从节点数: 6（每个主节点 1 个从）
└── 总节点数: 12
```

**可能的追问：如何评估内存碎片率？**

> ```bash
> # 查看内存碎片率
> redis-cli INFO memory
> 
> # mem_fragmentation_ratio = used_memory_rss / used_memory
> # > 1.5: 碎片较多，考虑重启整理
> # < 1.0: 内存紧张，可能使用 swap
> 
> # Redis 4.0+ 主动整理碎片
> activedefrag yes
> ```

---

## 分布式锁篇

### 1. Redis 如何实现分布式锁？

**答案：**

分布式锁用于分布式系统中协调多进程/多机器的资源访问。

**基本实现：**

```bash
# 加锁
SET lock:order:123 "uuid-xxxx" NX PX 30000

# NX: 不存在才设置
# PX: 设置过期时间（毫秒）
# value 使用 UUID 标识锁的持有者

# 解锁（Lua 脚本保证原子性）
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
```

**完整实现示例：**

```java
public class RedisLock {
    
    private Jedis jedis;
    private String lockKey;
    private String lockValue;
    private int expireTime = 30; // 秒
    
    public RedisLock(Jedis jedis, String lockKey) {
        this.jedis = jedis;
        this.lockKey = lockKey;
        this.lockValue = UUID.randomUUID().toString();
    }
    
    // 加锁
    public boolean tryLock() {
        String result = jedis.set(lockKey, lockValue, "NX", "EX", expireTime);
        return "OK".equals(result);
    }
    
    // 加锁（带超时）
    public boolean tryLock(long timeout, TimeUnit unit) throws InterruptedException {
        long end = System.currentTimeMillis() + unit.toMillis(timeout);
        while (System.currentTimeMillis() < end) {
            if (tryLock()) {
                return true;
            }
            Thread.sleep(100);  // 重试间隔
        }
        return false;
    }
    
    // 解锁
    public boolean unlock() {
        String script = 
            "if redis.call('GET', KEYS[1]) == ARGV[1] then " +
            "return redis.call('DEL', KEYS[1]) " +
            "else return 0 end";
        Object result = jedis.eval(script, 
            Collections.singletonList(lockKey), 
            Collections.singletonList(lockValue));
        return Long.valueOf(1).equals(result);
    }
}
```

**可能的追问：为什么 value 要用 UUID？**

> 使用 UUID 标识锁持有者，防止误删其他客户端的锁：
>
> ```
> 客户端A 获取锁，value = "uuid-a"
> 客户端A 执行业务时间过长，锁过期自动释放
> 客户端B 获取锁，value = "uuid-b"
> 客户端A 执行完毕，尝试释放锁
> 
> 如果不用 UUID 判断，A 会错误释放 B 的锁！
> 
> 使用 UUID 后：
> A 释放时发现 value != "uuid-a"，不会释放 B 的锁
> ```

---

### 2. setnx 实现分布式锁有什么问题？

**答案：**

早期使用 SETNX + EXPIRE 组合实现锁，存在原子性问题。

**问题实现：**

```java
// 错误实现（非原子）
public boolean tryLock() {
    // 1. SETNX 加锁
    long result = jedis.setnx(lockKey, lockValue);
    if (result == 1) {
        // 2. 设置过期时间
        jedis.expire(lockKey, 30);
        return true;
    }
    return false;
}
```

**问题分析：**

```
┌─────────────────────────────────────────────────────────────┐
│              SETNX + EXPIRE 的问题                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  问题1: 原子性问题                                           │
│  ─────────────────                                          │
│  SETNX 成功后，EXPIRE 前进程崩溃                             │
│  → 锁永不过期，死锁                                          │
│                                                              │
│  问题2: 无法设置 value                                       │
│  ───────────────────                                        │
│  SETNX 不支持 value 参数                                     │
│  → 无法区分锁持有者                                          │
│  → 可能误删其他客户端的锁                                     │
│                                                              │
│  问题3: 无法等待重试                                         │
│  ─────────────────                                          │
│  需要客户端自行实现重试逻辑                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**正确实现：**

```java
// 正确实现（Redis 2.6.12+）
// SET key value NX PX milliseconds
public boolean tryLock() {
    String result = jedis.set(lockKey, lockValue, "NX", "EX", 30);
    return "OK".equals(result);
}
```

**可能的追问：锁过期时间如何设置？**

> 设置过期时间需要权衡：
>
> ```yaml
> 太短的风险:
>   - 业务未执行完，锁过期
>   - 其他客户端获取锁，导致并发问题
> 
> 太长的风险:
>   - 客户端崩溃后，锁需要等待较长时间才能释放
>   - 资源长时间被占用
> 
> 建议做法:
>   1. 根据业务执行时间预估，设置合理过期时间
>   2. 使用看门狗机制自动续期
>   3. 记录锁持有时间，监控异常
> ```

---

### 3. Redisson 是如何实现分布式锁的？

**答案：**

Redisson 是 Redis 的 Java 客户端，提供了完善的分布式锁实现。

**基本使用：**

```java
// 引入依赖
// implementation 'org.redisson:redisson:3.23.0'

Config config = new Config();
config.useSingleServer()
    .setAddress("redis://127.0.0.1:6379");

RedissonClient redisson = Redisson.create(config);

// 获取锁
RLock lock = redisson.getLock("myLock");

try {
    // 加锁（自动续期）
    lock.lock();
    
    // 或指定等待时间和过期时间
    // lock.tryLock(10, 30, TimeUnit.SECONDS);
    
    // 执行业务
    doBusiness();
    
} finally {
    // 解锁
    lock.unlock();
}
```

**Redisson 锁的特点：**

```
┌─────────────────────────────────────────────────────────────┐
│                  Redisson 锁特性                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 可重入锁                                                 │
│     同一线程可多次获取同一把锁                               │
│                                                              │
│  2. 看门狗自动续期                                           │
│     默认过期时间 30s                                         │
│     每隔 10s 自动续期                                        │
│                                                              │
│  3. 公平锁                                                   │
│     按请求顺序获取锁                                         │
│                                                              │
│  4. 读写锁                                                   │
│     支持共享读锁和排他写锁                                   │
│                                                              │
│  5. 联锁                                                     │
│     同时锁定多个资源                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**锁的数据结构：**

```bash
# Redisson 锁在 Redis 中的存储
# Hash 结构

HGETALL myLock
# 返回:
# "uuid:threadId" -> "重入次数"

# 示例
HGETALL myLock
# "d8a5e3c0-1234-4567-89ab:1" -> "2"
# 表示某个线程重入 2 次
```

**加锁 Lua 脚本简化版：**

```lua
-- 加锁脚本
if (redis.call('exists', KEYS[1]) == 0) then
    -- 锁不存在，创建锁
    redis.call('hset', KEYS[1], ARGV[2], 1);
    redis.call('pexpire', KEYS[1], ARGV[1]);
    return nil;
end;

if (redis.call('hexists', KEYS[1], ARGV[2]) == 1) then
    -- 锁存在且是当前线程，重入次数 +1
    redis.call('hincrby', KEYS[1], ARGV[2], 1);
    redis.call('pexpire', KEYS[1], ARGV[1]);
    return nil;
end;

-- 锁被其他线程持有，返回剩余过期时间
return redis.call('pttl', KEYS[1]);
```

**可能的追问：Redisson 如何实现可重入？**

> Redisson 使用 Hash 结构存储锁：
>
> ```
> key: myLock
> value: Hash {
>     "uuid:thread-id-1": "2",  # 线程1 重入 2 次
>     "uuid:thread-id-2": "1"   # 线程2 持有 1 次（等待）
> }
> ```
>
> 加锁时：
> 1. 检查锁是否存在或是否是当前线程持有
> 2. 如果是当前线程，重入次数 +1
> 3. 否则等待
>
> 解锁时：
> 1. 重入次数 -1
> 2. 如果重入次数为 0，删除锁

---

### 4. 什么是看门狗机制？

**答案：**

看门狗（Watchdog）机制自动为锁续期，防止业务未执行完锁就过期。

**工作原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    看门狗工作机制                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  T0: 加锁成功，过期时间 30s                                  │
│      │                                                       │
│  T10s: 看门狗检测，续期到 30s                                │
│      │                                                       │
│  T20s: 看门狗检测，续期到 30s                                │
│      │                                                       │
│  T25s: 业务执行完毕，解锁                                    │
│      │                                                       │
│  看门狗停止                                                  │
│                                                              │
│  如果客户端崩溃:                                             │
│  ──────────────────                                         │
│  看门狗停止，锁在最后续期后 30s 自动过期                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**源码分析：**

```java
// Redisson 看门狗核心逻辑
private void scheduleExpirationRenewal(long threadId) {
    // 每 10 秒（internalLockLeaseTime / 3）执行一次续期
    Timeout task = commandExecutor.getConnectionManager()
        .newTimeout(new TimerTask() {
            @Override
            public void run(Timeout timeout) throws Exception {
                // 续期 Lua 脚本
                RFuture<Boolean> future = renewExpirationAsync(threadId);
                future.onComplete((res, e) -> {
                    if (res) {
                        // 续期成功，继续调度
                        scheduleExpirationRenewal(threadId);
                    }
                });
            }
        }, internalLockLeaseTime / 3, TimeUnit.MILLISECONDS);
}
```

**配置参数：**

```java
Config config = new Config();
config.setLockWatchdogTimeout(30000);  // 看门狗超时时间，默认 30s
// 续期间隔 = watchdogTimeout / 3 = 10s
```

**可能的追问：什么情况下看门狗不生效？**

> ```java
> // 情况1: 指定过期时间
> lock.lock(10, TimeUnit.SECONDS);  // 不会启动看门狗
> 
> // 情况2: 使用 tryLock 并指定租约时间
> lock.tryLock(0, 10, TimeUnit.SECONDS);  // 不会启动看门狗
> 
> // 情况3: 不带参数的 lock() 或 tryLock()
> lock.lock();  // 会启动看门狗
> lock.tryLock();  // 会启动看门狗
> ```

---

### 5. 什么是 Redlock 算法？

**答案：**

Redlock（Redis Distributed Lock）是 Redis 作者提出的分布式锁算法，解决单点问题。

**问题背景：**

```
单节点 Redis 分布式锁的问题:

场景1: 主从异步复制
┌─────────────────────────────────────────────────────────────┐
│  1. 客户端A 在 Master 获取锁                                 │
│  2. 锁数据还未同步到 Slave                                   │
│  3. Master 宕机，Slave 升级为 Master                         │
│  4. 客户端B 在新 Master 获取锁                               │
│  5. A 和 B 同时持有锁！                                      │
└─────────────────────────────────────────────────────────────┘

解决方案: Redlock - 在多个独立 Redis 节点获取锁
```

**Redlock 算法步骤：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Redlock 算法                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  假设 5 个独立 Redis 节点                                    │
│                                                              │
│  1. 获取当前时间戳 T1                                        │
│                                                              │
│  2. 依次向 5 个节点请求加锁                                  │
│     SET lock_key random_value NX PX ttl                     │
│                                                              │
│  3. 计算获取锁消耗的时间: T2 - T1                            │
│                                                              │
│  4. 判断是否成功:                                            │
│     - 在大多数节点(>=3)获取成功                              │
│     - 消耗时间 < 锁过期时间                                  │
│                                                              │
│  5. 成功: 有效时间 = 过期时间 - 消耗时间                     │
│     失败: 向所有节点释放锁                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Redisson 实现：**

```java
Config config1 = new Config();
config1.useSingleServer().setAddress("redis://node1:6379");

Config config2 = new Config();
config2.useSingleServer().setAddress("redis://node2:6379");

Config config3 = new Config();
config3.useSingleServer().setAddress("redis://node3:6379");

RedissonClient client1 = Redisson.create(config1);
RedissonClient client2 = Redisson.create(config2);
RedissonClient client3 = Redisson.create(config3);

// 创建 Redlock
RLock lock1 = client1.getLock("myLock");
RLock lock2 = client2.getLock("myLock");
RLock lock3 = client3.getLock("myLock");

RedissonRedLock redLock = new RedissonRedLock(lock1, lock2, lock3);

try {
    redLock.lock();
    // 业务逻辑
} finally {
    redLock.unlock();
}
```

**可能的追问：Redlock 有什么争议？**

> 分布式系统专家 Martin Kleppmann 对 Redlock 的质疑：
>
> 1. **时钟依赖**：算法依赖系统时钟，时钟跳跃可能导致问题
> 2. **网络延迟**：长时间 GC 或网络延迟可能导致锁失效
> 3. **过于复杂**：相比 fencing token 等方案更复杂
>
> 作者 Antirez 的回应：
> - 时钟同步是基础设施问题
> - 合理设置超时时间可以规避
> - 对于大多数场景足够安全

---

### 6. 分布式锁有什么常见坑？

**答案：**

分布式锁实现和使用中有很多陷阱。

**常见问题及解决方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                  分布式锁常见坑                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  坑1: 锁过期但业务未执行完                                   │
│  ─────────────────────────                                  │
│  原因: 业务执行时间超过锁过期时间                            │
│  解决: 使用看门狗机制或合理设置过期时间                      │
│                                                              │
│  坑2: 误删其他客户端的锁                                     │
│  ───────────────────────                                    │
│  原因: 解锁时不校验锁持有者                                  │
│  解决: 使用 UUID 标识，Lua 脚本校验后删除                    │
│                                                              │
│  坑3: 锁不可重入                                             │
│  ─────────────────                                          │
│  原因: 简单实现不支持同一线程多次加锁                        │
│  解决: 使用 Redisson 可重入锁                                │
│                                                              │
│  坑4: 主从切换导致锁丢失                                     │
│  ───────────────────────                                    │
│  原因: 主节点获取锁后宕机，从节点未同步                      │
│  解决: 使用 Redlock 或等待锁过期                             │
│                                                              │
│  坑5: 锁竞争导致性能问题                                     │
│  ───────────────────────                                    │
│  原因: 大量客户端竞争同一把锁                                │
│  解决: 减小锁粒度，使用分段锁                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**正确使用示例：**

```java
public void processOrder(String orderId) {
    String lockKey = "lock:order:" + orderId;
    RLock lock = redisson.getLock(lockKey);
    
    try {
        // 设置等待时间和租约时间
        if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
            try {
                // 业务逻辑
                doProcess(orderId);
            } finally {
                // 确保释放锁
                if (lock.isHeldByCurrentThread()) {
                    lock.unlock();
                }
            }
        } else {
            throw new RuntimeException("获取锁失败");
        }
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException("获取锁被中断", e);
    }
}
```

**可能的追问：如何选择分布式锁方案？**

> ```yaml
> Redis 分布式锁:
>   适用: 性能要求高、可接受极端情况下不一致
>   方案: Redisson（推荐）
> 
> Zookeeper 分布式锁:
>   适用: 一致性要求高
>   特点: CP 系统，更可靠但性能较低
> 
> etcd 分布式锁:
>   适用: 云原生环境
>   特点: 强一致性，支持租约
> 
> 数据库锁:
>   适用: 低并发场景
>   特点: 简单但性能差
> ```

---

## 过期与淘汰篇

### 1. Redis 过期策略是什么？

**答案：**

Redis 采用惰性删除 + 定期删除两种策略。

**惰性删除：**

```
┌─────────────────────────────────────────────────────────────┐
│                    惰性删除                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  触发时机: 访问 key 时检查是否过期                           │
│                                                              │
│  流程:                                                      │
│  1. 客户端请求 GET key                                      │
│  2. 检查 key 是否过期                                       │
│  3. 如果过期，删除 key，返回 null                           │
│  4. 如果未过期，返回 value                                  │
│                                                              │
│  优点: CPU 友好，只在访问时检查                              │
│  缺点: 过期 key 可能长期占用内存                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**定期删除：**

```
┌─────────────────────────────────────────────────────────────┐
│                    定期删除                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  触发时机: 周期性执行（每秒 10 次左右）                      │
│                                                              │
│  流程:                                                      │
│  1. 随机抽取 20 个设置了过期时间的 key                      │
│  2. 删除其中已过期的 key                                    │
│  3. 如果过期 key 比例 > 25%，重复步骤 1-2                   │
│  4. 每次执行时间不超过 25ms                                 │
│                                                              │
│  优点: 平衡内存和 CPU                                       │
│  缺点: 无法保证过期 key 及时删除                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**源码逻辑：**

```c
// 定期删除核心逻辑
void activeExpireCycle(int type) {
    // 每次执行的默认时间限制
    timelimit = 1000000 * ACTIVE_EXPIRE_CYCLE_SLOW_TIME_PERC / 100;
    
    do {
        // 随机抽取 20 个 key
        long max = 20;
        long expired = 0;
        
        for (int j = 0; j < max; j++) {
            if (expireIfNeeded(key)) {
                expired++;
            }
        }
        
        // 如果过期比例 > 25%，继续
        if (expired <= max * 25 / 100) break;
        
    } while (elapsed < timelimit);
}
```

**可能的追问：为什么不用定时删除？**

> 定时删除（为每个 key 创建定时器）的问题：
>
> 1. **内存开销大**：每个过期 key 都需要一个定时器
> 2. **CPU 消耗大**：大量定时器触发时 CPU 繁忙
> 3. **实现复杂**：需要维护定时器队列
>
> Redis 选择惰性删除 + 定期删除的折中方案。

---

### 2. Redis 内存淘汰策略有哪些？

**答案：**

当内存达到上限时，Redis 需要淘汰部分数据。

**淘汰策略：**

```
┌─────────────────────────────────────────────────────────────┐
│                  内存淘汰策略                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  不淘汰策略:                                                 │
│  ├── noeviction: 不淘汰，内存满时返回错误（默认）           │
│                                                              │
│  全局淘汰:                                                   │
│  ├── allkeys-lru: 所有 key 中淘汰最久未使用的               │
│  ├── allkeys-lfu: 所有 key 中淘汰访问频率最低的             │
│  ├── allkeys-random: 所有 key 中随机淘汰                    │
│                                                              │
│  过期 key 淘汰:                                              │
│  ├── volatile-lru: 设置过期的 key 中淘汰最久未使用的        │
│  ├── volatile-lfu: 设置过期的 key 中淘汰访问频率最低的      │
│  ├── volatile-random: 设置过期的 key 中随机淘汰             │
│  ├── volatile-ttl: 淘汰即将过期的 key                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**策略选择建议：**

```yaml
缓存场景:
  策略: allkeys-lru 或 allkeys-lfu
  原因: 淘汰冷数据，保留热点数据

有部分数据需要持久保留:
  策略: volatile-lru
  原因: 只淘汰有设置过期时间的 key

数据访问均匀:
  策略: allkeys-random 或 volatile-random
  原因: 随机淘汰，开销小

需要保留最新数据:
  策略: volatile-ttl
  原因: 优先淘汰即将过期的 key
```

**配置：**

```bash
# redis.conf
maxmemory 4gb                    # 最大内存限制
maxmemory-policy allkeys-lru     # 淘汰策略
maxmemory-samples 5              # LRU/LFU 采样数量
```

**可能的追问：LRU 和 LFU 如何选择？**

> ```yaml
> LRU 适合:
>   - 数据访问有明显时间局部性
>   - 热点数据会持续被访问
>   - 旧数据逐渐不再访问
> 
> LFU 适合:
>   - 需要识别长期热点
>   - 防止历史热点占用缓存
>   - 如推荐系统、热门商品
> 
> 建议: 不确定时先用 LRU，监控效果后调整
> ```

---

### 3. LRU 和 LFU 的区别是什么？

**答案：**

LRU 和 LFU 是两种不同的缓存淘汰策略。

**LRU（Least Recently Used）：**

```
淘汰最久未访问的数据

示例:
时间线: T1 → T2 → T3 → T4
访问顺序: A → B → C → A

缓存状态变化:
T1: [A]
T2: [B, A]      # B 新加入，A 变旧
T3: [C, B, A]   # C 新加入
T4: [A, C, B]   # A 被访问，移到最前

如果需要淘汰: 删除 B（最久未访问）
```

**LFU（Least Frequently Used）：**

```
淘汰访问频率最低的数据

示例:
访问历史: A(5次) B(3次) C(1次) D(1次)

缓存状态:
[A:5] → [B:3] → [C:1] → [D:1]

如果需要淘汰: 删除 C 或 D（频率最低）
```

**对比：**

| 对比项 | LRU | LFU |
|--------|-----|-----|
| 关注点 | 最近访问时间 | 访问频率 |
| 实现复杂度 | 较简单 | 较复杂 |
| 新数据友好度 | 高（可能淘汰老数据） | 低（频率低） |
| 历史热点 | 可能被淘汰 | 保留 |
| 突发流量 | 适应快 | 适应慢 |

**Redis 中的实现：**

```c
// LRU: 使用 24 bit 存储访问时间戳
typedef struct redisObject {
    unsigned lru:LRU_BITS;  // 访问时间
    // ...
} robj;

// LFU: 使用 16 bit 存储时间 + 8 bit 存储计数器
// 高 16 位: 上次访问时间（分钟级）
// 低 8 位: 对数计数器
```

**可能的追问：LFU 计数器为什么用对数增长？**

> 防止热点数据计数值溢出，同时保持区分度：
>
> ```python
> # 线性增长
> # counter: 1, 2, 3, ..., 255
> # 问题: 热点数据很快达到 255，无法区分
> 
> # 对数增长
> # 访问 100 次: counter ≈ 10
> # 访问 10000 次: counter ≈ 20
> # 访问 1000000 次: counter ≈ 30
> 
> # 公式: counter++ 概率 = 1 / (counter * factor + 1)
> # factor 越大，增长越慢
> ```

---

### 4. 如何选择淘汰策略？

**答案：**

根据业务场景选择合适的淘汰策略。

**决策流程：**

```
                    开始
                      │
                      ▼
            ┌─────────────────┐
            │ 是否所有数据可淘汰│
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │ 是                   │ 否
          ▼                      ▼
   ┌──────────────┐      ┌──────────────┐
   │ 访问模式是什么│      │ 只淘汰过期 key│
   └──────┬───────┘      └──────────────┘
          │
   ┌──────┼──────┐
   │      │      │
   ▼      ▼      ▼
 时间    频率    均匀
 局部性  热点   随机
   │      │      │
   ▼      ▼      ▼
allkeys- allkeys- allkeys-
  lru     lfu    random
```

**场景示例：**

```yaml
场景1: 用户信息缓存
特点: 热点用户频繁访问，冷用户逐渐不再访问
推荐: allkeys-lru

场景2: 商品信息缓存
特点: 热门商品长期热门，冷门商品访问少
推荐: allkeys-lfu

场景3: 验证码缓存
特点: 短期有效，设置过期时间
推荐: volatile-ttl

场景4: 配置信息缓存
特点: 部分配置不能淘汰
推荐: volatile-lru（配置不设过期时间）

场景5: 会话缓存
特点: 访问相对均匀
推荐: allkeys-random
```

**配置监控：**

```bash
# 查看淘汰统计
redis-cli INFO stats | grep evicted

# evicted_keys: 1000  # 被淘汰的 key 数量

# 监控命令
redis-cli --stat

# 调整采样数量（提高精确度）
maxmemory-samples 10  # 默认 5，增大可提高 LRU/LFU 精确度
```

**可能的追问：淘汰策略对性能有什么影响？**

> ```yaml
> 性能开销排序（从低到高）:
>   1. noeviction: 无开销
>   2. random: 随机选择，开销小
>   3. ttl: 按过期时间排序，开销中等
>   4. lru: 需要采样计算，开销较大
>   5. lfu: 需要采样 + 计数器衰减，开销最大
> 
> 建议:
>   - 增加 maxmemory-samples 可提高精确度，但增加开销
>   - 监控 evicted_keys 数量，评估淘汰策略效果
>   - 如果淘汰过于频繁，考虑增加内存或优化数据
> ```

---

### 5. 如何监控 Redis 内存使用？

**答案：**

有效监控 Redis 内存使用，预防问题。

**关键指标：**

```bash
# 查看内存信息
redis-cli INFO memory

# 关键输出
used_memory:1000000           # 已使用内存（字节）
used_memory_human:1.00M       # 已使用内存（可读）
used_memory_rss:2000000       # 操作系统分配的内存
used_memory_peak:1500000      # 内存使用峰值
used_memory_lua:37888         # Lua 引擎内存
mem_fragmentation_ratio:2.0   # 内存碎片率
mem_allocator:jemalloc-5.1.0  # 内存分配器
```

**内存碎片分析：**

```
┌─────────────────────────────────────────────────────────────┐
│                  内存碎片率解读                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  mem_fragmentation_ratio = used_memory_rss / used_memory    │
│                                                              │
│  < 1.0:                                                     │
│    含义: RSS < 逻辑内存                                      │
│    原因: 使用了 swap                                         │
│    解决: 增加物理内存，检查 swap 使用                        │
│                                                              │
│  1.0 - 1.5: 正常                                            │
│                                                              │
│  > 1.5:                                                     │
│    含义: 内存碎片较多                                        │
│    原因: 频繁增删数据，jemalloc 无法重用内存                 │
│    解决: 重启 Redis 或开启 activedefrag                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**内存碎片整理：**

```bash
# Redis 4.0+ 开启主动碎片整理
activedefrag yes

# 配置参数
active-defrag-ignore-bytes 100mb      # 超过 100MB 碎片才整理
active-defrag-threshold-lower 10      # 碎片率 > 10% 开始整理
active-defrag-threshold-upper 100     # 碎片率 > 100% 全力整理
active-defrag-cycle-min 1             # 整理 CPU 最小占用 1%
active-defrag-cycle-max 25            # 整理 CPU 最大占用 25%
```

**内存分析脚本：**

```bash
#!/bin/bash
# 监控 Redis 内存

MEMORY_INFO=$(redis-cli INFO memory)

USED=$(echo "$MEMORY_INFO" | grep used_memory: | cut -d: -f2 | tr -d '\r')
RSS=$(echo "$MEMORY_INFO" | grep used_memory_rss: | cut -d: -f2 | tr -d '\r')
FRAG_RATIO=$(echo "scale=2; $RSS / $USED" | bc)

echo "Used Memory: $USED bytes"
echo "RSS Memory: $RSS bytes"
echo "Fragmentation Ratio: $FRAG_RATIO"

if [ $(echo "$FRAG_RATIO > 1.5" | bc) -eq 1 ]; then
    echo "Warning: High memory fragmentation!"
fi
```

**可能的追问：如何减少内存碎片？**

> ```yaml
> 方法:
>   1. 使用合适的数据结构
>      - 小数据用 ziplist/intset
>      - 避免 key 过长
>   
>   2. 避免频繁创建删除
>      - 批量操作替代单个操作
>      - 使用过期策略而非手动删除
>   
>   3. 开启碎片整理
>      activedefrag yes
>   
>   4. 定期重启
>      - 低峰期重启 Redis
>      - 主从切换实现无感重启
> ```

---

### 6. 如何处理 Redis 内存满的情况？

**答案：**

内存满时需要及时处理，防止服务不可用。

**紧急处理：**

```bash
# 1. 检查内存使用
redis-cli INFO memory

# 2. 查看大 key
redis-cli --bigkeys

# 3. 手动清理（谨慎）
# 删除无用的 key
redis-cli --scan --pattern "temp:*" | xargs redis-cli DEL

# 4. 临时增加内存限制
redis-cli CONFIG SET maxmemory 8gb
```

**排查步骤：**

```
┌─────────────────────────────────────────────────────────────┐
│                  内存满排查流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 检查大 Key                                               │
│     redis-cli --bigkeys                                     │
│     redis-cli MEMORY USAGE key                              │
│                                                              │
│  2. 分析 key 分布                                            │
│     redis-cli --scan --pattern "*" | head -1000             │
│     统计各前缀的 key 数量                                    │
│                                                              │
│  3. 检查过期策略                                             │
│     是否有过期时间设置                                       │
│     淘汰策略是否生效                                         │
│                                                              │
│  4. 检查客户端缓冲区                                         │
│     redis-cli CLIENT LIST                                   │
│     查看是否有大输出缓冲区                                   │
│                                                              │
│  5. 检查内存碎片                                             │
│     INFO memory                                             │
│     mem_fragmentation_ratio                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**预防措施：**

```bash
# 1. 设置合理的内存限制和淘汰策略
maxmemory 4gb
maxmemory-policy allkeys-lru

# 2. 设置内存告警阈值
# 监控 used_memory / maxmemory > 80%

# 3. 定期清理
# 设置合理的过期时间
EXPIRE key 3600

# 4. 集群扩展
# 数据量大时使用 Redis Cluster 分片
```

**可能的追问：noeviction 策略下内存满会怎样？**

> ```bash
> # 默认策略 noeviction
> # 内存满时写入操作返回错误
> 
> SET key value
> # (error) OOM command not allowed when used memory > 'maxmemory'.
> 
> # 只读操作正常
> GET key
> # 正常返回
> 
> # 影响的命令: SET, HSET, LPUSH, SADD, ZADD 等
> # 不影响的命令: GET, HGET, LLEN, SISMEMBER 等
> ```
