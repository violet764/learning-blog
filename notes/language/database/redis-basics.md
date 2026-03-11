# Redis 学习笔记

Redis（Remote Dictionary Server）是一个开源的、基于内存的高性能键值数据库，支持多种数据结构，广泛应用于缓存、会话存储、消息队列、实时分析等场景。Redis 7.x 带来了许多新特性和性能优化，本文将全面介绍 Redis 的核心概念和现代最佳实践。

## Redis 7.x 新特性

### 🚀 核心改进

Redis 7.x 是 Redis 历史上最重要的版本之一，带来了大量新特性：

| 特性 | 说明 |
|------|------|
| **Function** | 替代 Lua 脚本的新函数系统，支持持久化和复制 |
| **多部分 AOF** | AOF 拆分为基础文件 + 增量文件，提高重写效率 |
| **ACL v2** | 增强的访问控制，支持更细粒度的权限管理 |
| **Sharded Pub/Sub** | 分片发布订阅，集群环境下更高效 |
| **Client Eviction** | 客户端驱逐策略，内存压力大时断开连接 |

### 📝 新增命令

```bash
# 7.x 新增的列表操作命令
LPUSHX  # 仅当列表存在时才插入
RPUSHX
LMOVE   # 原子性地将元素从一个列表移动到另一个列表
BLMOVE  # 阻塞版本

# Set 操作增强
SINTERCARD  # 返回交集元素数量，不返回具体元素
SISMEMBER   # 支持检查多个元素

# Sorted Set 增强
ZMPOP       # 批量弹出分数最高/最低的元素
BZMPOP      # 阻塞版本
ZINTERCARD  # 返回交集元素数量
```

### 🔧 配置优化

```bash
# redis.conf 新增配置项

# 多部分 AOF 配置
appendonly yes
appendfilename "appendonly.aof"
appenddirname "appendonlydir"

# 客户端驱逐策略
maxmemory-clients max-clients 100000  # 最大客户端连接内存
client-eviction-mode allkeys-lru      # 驱逐策略

# ACL 配置文件
aclfile /etc/redis/users.acl
```

## 核心数据结构

### 📌 String（字符串）

String 是 Redis 最基本的数据类型，可以存储字符串、整数或二进制数据（如图片）。

**底层实现**：SDS（Simple Dynamic String）

```bash
# 基本操作
SET user:1001 "张三"
GET user:1001                    # "张三"

# 设置过期时间（推荐）
SET session:abc123 "data" EX 3600  # 1小时后过期

# 不存在时才设置（分布式锁常用）
SET lock:order:123 "uuid" NX EX 30

# 批量操作（提升性能）
MSET key1 "value1" key2 "value2"
MGET key1 key2

# 数值操作
INCR page:views                  # 自增
INCRBY stock:apple 100           # 增加100
DECRBY stock:apple 10            # 减少10
```

**Python 客户端示例**：

```python
import redis
from redis.connection import ConnectionPool

# 连接池配置
pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=100,
    socket_timeout=5,
    socket_connect_timeout=5,
    decode_responses=True  # 自动解码为字符串
)
r = redis.Redis(connection_pool=pool)

# 基本操作
r.set('user:1001', '张三', ex=3600)  # 设置并添加过期时间
r.set('lock:order', 'uuid', nx=True, ex=30)  # 分布式锁

# 批量操作
r.mset({'key1': 'value1', 'key2': 'value2'})
values = r.mget(['key1', 'key2'])

# 数值操作
r.incr('page:views')
r.incrby('stock:apple', 100)
```

### 📌 Hash（哈希）

Hash 适合存储对象，比将对象 JSON 序列化存储更节省内存。

**底层实现**：ListPack（小数据量）/ HashTable（大数据量）

```bash
# 基本操作
HSET user:1001 name "张三" age 25 email "zhangsan@example.com"
HGET user:1001 name               # "张三"
HGETALL user:1001                 # 获取所有字段

# 批量设置
HMSET product:2001 name "iPhone" price 6999 stock 100

# 数值操作
HINCRBY user:1001 age 1           # 年龄+1

# 条件更新（7.x新增）
HSETNX user:1001 created_at "2024-01-01"  # 仅当字段不存在时设置

# 删除字段
HDEL user:1001 email
```

**应用场景**：

```python
# 存储购物车
def add_to_cart(user_id, product_id, quantity):
    key = f'cart:{user_id}'
    r.hset(key, product_id, quantity)
    r.expire(key, 86400 * 7)  # 7天过期

def get_cart(user_id):
    return r.hgetall(f'cart:{user_id}')

# 存储用户信息
def update_user(user_id, **kwargs):
    key = f'user:{user_id}'
    r.hset(key, mapping=kwargs)
    return r.hgetall(key)
```

### 📌 List（列表）

List 是双端链表，支持从两端插入和弹出，适合实现队列和栈。

**底层实现**：ListPack（小数据量）/ QuickList（大数据量）

```bash
# 队列操作（FIFO）
LPUSH queue:tasks "task1"         # 左边插入
RPOP queue:tasks                  # 右边弹出

# 栈操作（LIFO）
LPUSH stack:data "data1"
LPOP stack:data

# 阻塞操作（消息队列常用）
BLPOP queue:tasks 5               # 阻塞5秒等待数据

# 获取列表内容
LRANGE queue:tasks 0 -1           # 获取所有元素
LLEN queue:tasks                  # 列表长度

# 7.x 新增：原子移动
LMOVE queue:tasks queue:processed LEFT RIGHT  # 从左边弹出，右边插入
```

**应用场景**：

```python
# 简单消息队列
def push_task(task_data):
    r.lpush('task:queue', task_data)

def pop_task(timeout=5):
    result = r.brpop('task:queue', timeout)
    return result[1] if result else None

# 最新消息列表
def add_message(chat_id, message):
    key = f'chat:{chat_id}:messages'
    r.lpush(key, message)
    r.ltrim(key, 0, 99)  # 只保留最新100条

def get_messages(chat_id, count=50):
    return r.lrange(f'chat:{chat_id}:messages', 0, count - 1)
```

### 📌 Set（集合）

Set 存储唯一的无序元素，适合存储标签、好友关系等。

**底层实现**：IntSet（纯整数）/ HashTable

```bash
# 基本操作
SADD user:1001:tags "技术" "编程" "AI"
SMEMBERS user:1001:tags           # 获取所有成员
SISMEMBER user:1001:tags "技术"   # 检查是否存在

# 集合运算
SADD user:1002:tags "技术" "设计"
SINTER user:1001:tags user:1002:tags    # 交集：共同标签
SUNION user:1001:tags user:1002:tags    # 并集
SDIFF user:1001:tags user:1002:tags     # 差集

# 7.x 新增：返回交集数量
SINTERCARD 2 user:1001:tags user:1002:tags

# 随机元素
SRANDMEMBER user:1001:tags 2      # 随机取2个
SPOP user:1001:tags               # 随机弹出
```

**应用场景**：

```python
# 社交关系：关注/粉丝
def follow(user_id, target_id):
    r.sadd(f'user:{user_id}:following', target_id)
    r.sadd(f'user:{target_id}:followers', user_id)

def get_mutual_followers(user_id, other_id):
    return r.sinter(f'user:{user_id}:followers', f'user:{other_id}:followers')

# 标签系统
def add_tags(article_id, tags):
    for tag in tags:
        r.sadd(f'tag:{tag}:articles', article_id)
        r.sadd(f'article:{article_id}:tags', tag)

def get_articles_by_tags(tags):
    return r.sinter(*[f'tag:{tag}:articles' for tag in tags])
```

### 📌 Sorted Set（有序集合）

Sorted Set 是带分数的有序集合，适合排行榜、带权重的队列。

**底层实现**：ListPack（小数据量）/ SkipList + HashTable（大数据量）

```bash
# 基本操作
ZADD leaderboard 100 "player1" 200 "player2" 150 "player3"
ZRANGE leaderboard 0 -1 WITHSCORES    # 按分数升序
ZREVRANGE leaderboard 0 9 WITHSCORES  # 按分数降序（排行榜）

# 获取排名
ZRANK leaderboard "player1"           # 升序排名
ZREVRANK leaderboard "player1"        # 降序排名

# 分数操作
ZINCRBY leaderboard 50 "player1"      # 分数+50

# 范围查询
ZRANGEBYSCORE leaderboard 100 200     # 分数在100-200之间

# 7.x 新增：批量弹出
ZMPOP 1 leaderboard MIN COUNT 5       # 弹出分数最低的5个
```

**应用场景**：

```python
# 游戏排行榜
def update_score(player_id, score):
    r.zadd('game:leaderboard', {player_id: score})

def get_top_players(count=10):
    return r.zrevrange('game:leaderboard', 0, count - 1, withscores=True)

def get_player_rank(player_id):
    return r.zrevrank('game:leaderboard', player_id) + 1

# 延时队列
def add_delayed_task(task_id, execute_at_timestamp):
    r.zadd('delayed:tasks', {task_id: execute_at_timestamp})

def get_ready_tasks():
    now = time.time()
    return r.zrangebyscore('delayed:tasks', 0, now)
```

### 📌 Stream（数据流）

Stream 是 Redis 5.0 引入的日志型数据结构，适合消息队列和事件溯源。

```bash
# 添加消息
XADD mystream * name "张三" action "login"
XADD mystream * name "李四" action "purchase" amount 100

# 读取消息
XRANGE mystream - +                  # 读取所有
XRANGE mystream - + COUNT 10         # 读取10条
XREAD COUNT 10 BLOCK 5000 STREAMS mystream $  # 阻塞读取新消息

# 消费者组
XGROUP CREATE mystream group1 $ MKSTREAM
XREADGROUP GROUP group1 consumer1 COUNT 1 STREAMS mystream >
XACK mystream group1 1526569495631-0  # 确认消息

# 消息持久化
XTRIM mystream MAXLEN 1000           # 保留最新1000条
XLEN mystream                        # 消息数量
```

**应用场景**：

```python
# 事件溯源
def append_event(stream_name, event_type, data):
    return r.xadd(stream_name, {'type': event_type, 'data': data})

def read_events(stream_name, start_id='0', count=10):
    return r.xrange(stream_name, start_id, '+', count=count)

# 消费者组处理
def process_messages(stream_name, group_name, consumer_name):
    while True:
        messages = r.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: '>'},
            count=10,
            block=5000
        )
        for stream, msgs in messages:
            for msg_id, data in msgs:
                # 处理消息
                process(data)
                # 确认消息
                r.xack(stream_name, group_name, msg_id)
```

### 📌 Bitmap（位图）

Bitmap 不是独立的数据类型，而是 String 上的位操作，适合存储布尔型数据。

```bash
# 基本操作
SETBIT user:signin:2024:01 0 1      # 第0位设为1（第1天签到）
GETBIT user:signin:2024:01 0        # 获取第0位的值

# 统计
BITCOUNT user:signin:2024:01        # 统计1的个数（签到天数）

# 位运算
BITOP AND result user1:signin user2:signin  # 与运算
BITOP OR result user1:signin user2:signin   # 或运算
```

**应用场景**：

```python
# 用户签到
def sign_in(user_id, date=None):
    date = date or datetime.now()
    key = f'signin:{user_id}:{date.year}:{date.month}'
    day = date.day - 1  # 位索引从0开始
    r.setbit(key, day, 1)

def get_sign_in_count(user_id, year, month):
    key = f'signin:{user_id}:{year}:{month}'
    return r.bitcount(key)

def is_signed_in(user_id, day, year, month):
    key = f'signin:{user_id}:{year}:{month}'
    return bool(r.getbit(key, day - 1))

# 在线用户统计
def mark_online(user_id):
    today = datetime.now().strftime('%Y-%m-%d')
    r.setbit(f'online:{today}', user_id, 1)

def get_online_count():
    today = datetime.now().strftime('%Y-%m-%d')
    return r.bitcount(f'online:{today}')
```

### 📌 HyperLogLog

HyperLogLog 是用于基数估计的概率数据结构，误差约 0.81%，占用仅 12KB。

```bash
# 基本操作
PFADD page:views:today user1 user2 user3 user1  # 添加（自动去重）
PFCOUNT page:views:today                        # 估计唯一值数量

# 合并
PFADD page:views:yesterday user1 user4
PFMERGE page:views:week page:views:today page:views:yesterday
PFCOUNT page:views:week
```

**应用场景**：

```python
# UV 统计
def record_page_view(page_id, user_id):
    r.pfadd(f'page:uv:{page_id}:{date.today()}', user_id)

def get_unique_visitors(page_id, date):
    return r.pfcount(f'page:uv:{page_id}:{date}')
```

## 数据结构底层实现原理

### 🔬 SDS（Simple Dynamic String）

```
struct sdshdr {
    int len;      // 已使用长度
    int free;     // 剩余可用空间
    char buf[];   // 字节数组
};
```

**优势**：
- O(1) 获取长度（C 字符串需要遍历）
- 防止缓冲区溢出
- 减少内存重分配次数（空间预分配 + 惰性释放）
- 二进制安全（可以存储任意数据）

### 🔬 QuickList

QuickList 是 List 的底层实现，是 ZipList 的双向链表：

```
QuickList = LinkedList + ZipList
         = 多个节点组成的双向链表
         = 每个节点是一个 ZipList
```

**优势**：
- 兼顾链表的插入效率和 ZipList 的内存效率
- 支持中间节点的压缩（进一步节省内存）

### 🔬 SkipList（跳跃表）

SkipList 是 Sorted Set 的底层实现之一：

```
Level 4:        10 -----------------------------> 100
Level 3:        10 -------------> 50 ---------> 100
Level 2:        10 -----> 30 ---> 50 ---------> 100
Level 1:        10 -> 20 -> 30 -> 50 -> 70 -> 100
```

**特点**：
- 平均 O(logN) 的查找、插入、删除
- 实现简单，比平衡树更容易调试
- 范围查询效率高

### 🔬 整数集合（IntSet）

当 Set 只包含整数时，使用 IntSet 存储：

```c
typedef struct intset {
    uint32_t encoding;  // 编码方式：int16/int32/int64
    uint32_t length;    // 元素数量
    int8_t contents[];  // 整数数组
} intset;
```

**特点**：
- 有序数组，二分查找
- 自动升级编码（int16 → int32 → int64）
- 内存紧凑

## 持久化策略

### 💾 RDB（Redis Database）

RDB 是 Redis 的快照持久化，将内存数据保存到磁盘的二进制文件。

**配置**：

```bash
# redis.conf
save 900 1      # 900秒内有1次修改就保存
save 300 10     # 300秒内有10次修改就保存
save 60 10000   # 60秒内有10000次修改就保存

# RDB 文件名
dbfilename dump.rdb

# 存储目录
dir /var/lib/redis

# 压缩
rdbcompression yes
rdbchecksum yes
```

**手动触发**：

```bash
# 同步保存（阻塞）
SAVE

# 异步保存（推荐）
BGSAVE
```

**优缺点**：

| 优点 | 缺点 |
|------|------|
| 文件紧凑，适合备份 | 可能丢失最后一次快照后的数据 |
| 恢复速度快 | Fork 大数据集时可能阻塞 |
| 对性能影响小 | 不适合实时持久化 |

### 💾 AOF（Append Only File）

AOF 记录所有写命令，是追加式日志。

**配置**：

```bash
# redis.conf
appendonly yes
appendfilename "appendonly.aof"

# 同步策略
appendfsync always     # 每次写入都同步（最安全，最慢）
appendfsync everysec   # 每秒同步一次（推荐）
appendfsync no         # 由操作系统决定

# AOF 重写
auto-aof-rewrite-percentage 100  # 文件大小翻倍时重写
auto-aof-rewrite-min-size 64mb   # 最小重写大小
```

**AOF 重写**：

```bash
# 手动触发重写
BGREWRITEAOF
```

**优缺点**：

| 优点 | 缺点 |
|------|------|
| 数据更安全，最多丢失1秒 | 文件较大 |
| 可读性好，便于分析 | 写入性能略低于 RDB |
| 支持重写压缩 | 恢复速度较慢 |

### 💾 混合持久化（Redis 7.x 优化）

Redis 7.x 的多部分 AOF 将文件拆分为：

```
appendonlydir/
├── appendonly.aof.manifest  # 清单文件
├── appendonly.aof.1.base.rdb     # 基础文件（RDB格式）
└── appendonly.aof.1.incr.aof     # 增量文件（AOF格式）
```

**配置**：

```bash
# redis.conf
appendonly yes
appenddirname "appendonlydir"
appendfilename "appendonly.aof"

# 开启混合持久化
aof-use-rdb-preamble yes
```

**优势**：
- 恢复时先加载 RDB 基础文件，再重放增量 AOF
- 结合了 RDB 的恢复速度和 AOF 的数据安全

## 主从复制与哨兵模式

### 🔄 主从复制

Redis 支持异步复制，一个主节点可以有多个从节点。

**配置**：

```bash
# 从节点配置（redis.conf）
replicaof 192.168.1.100 6379

# 只读模式（从节点默认开启）
replica-read-only yes

# 复制配置
repl-diskless-sync yes        # 无盘复制（网络传输）
repl-timeout 60               # 复制超时
```

**命令行操作**：

```bash
# 在从节点执行
REPLICAOF 192.168.1.100 6379  # 设置主节点
REPLICAOF NO ONE              # 取消复制，成为主节点

# 查看复制状态
INFO replication
```

**Python 客户端读写分离**：

```python
from redis import Redis
from redis.sentinel import Sentinel

# 使用 Sentinel 实现读写分离
sentinel = Sentinel([
    ('sentinel1', 26379),
    ('sentinel2', 26379),
    ('sentinel3', 26379)
], socket_timeout=0.1)

# 获取主节点连接（写操作）
master = sentinel.master_for('mymaster', socket_timeout=0.1)
master.set('key', 'value')

# 获取从节点连接（读操作）
slave = sentinel.slave_for('mymaster', socket_timeout=0.1)
slave.get('key')
```

### 🛡️ 哨兵模式（Sentinel）

Sentinel 提供监控、通知和自动故障转移功能。

**配置（sentinel.conf）**：

```bash
# 监控主节点
sentinel monitor mymaster 192.168.1.100 6379 2

# 主观下线判定
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时
sentinel failover-timeout mymaster 180000

# 同时可以有多少个从节点对新主节点同步
sentinel parallel-syncs mymaster 1
```

**哨兵常用命令**：

```bash
# 查看主节点状态
SENTINEL master mymaster

# 查看从节点列表
SENTINEL replicas mymaster

# 查看哨兵列表
SENTINEL sentinels mymaster

# 获取当前主节点地址
SENTINEL get-master-addr-by-name mymaster

# 手动故障转移
SENTINEL failover mymaster
```

## Redis Cluster 集群部署

Redis Cluster 是 Redis 的分布式解决方案，支持自动分片和故障转移。

### 架构设计

```
                    Cluster Bus (16384 slots)
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Master1 │    │ Master2 │    │ Master3 │    │ Master4 │
│ 0-4095  │    │ 4096-   │    │ 8192-   │    │12288-   │
│         │    │ 8191    │    │ 12287   │    │16383    │
└────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Replica │    │ Replica │    │ Replica │    │ Replica │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 集群配置

```bash
# redis.conf（每个节点）
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
cluster-announce-ip 192.168.1.100
cluster-announce-port 6379
cluster-announce-bus-port 16379

# 最小集群配置（至少3主3从）
```

### 创建集群

```bash
# Redis 7.x 使用 redis-cli 创建集群
redis-cli --cluster create \
  192.168.1.101:6379 192.168.1.102:6379 192.168.1.103:6379 \
  192.168.1.104:6379 192.168.1.105:6379 192.168.1.106:6379 \
  --cluster-replicas 1

# 检查集群状态
redis-cli --cluster check 192.168.1.101:6379

# 添加节点
redis-cli --cluster add-node new_host:6379 existing_host:6379

# 重新分片
redis-cli --cluster reshard host:6379
```

### Python 客户端连接集群

```python
from redis.cluster import RedisCluster

# 连接集群
rc = RedisCluster(
    host='192.168.1.101',
    port=6379,
    max_connections=100,
    socket_timeout=5,
    decode_responses=True
)

# 正常使用，集群自动路由
rc.set('user:1001', '张三')
rc.get('user:1001')

# 批量操作（使用 pipeline）
with rc.pipeline() as pipe:
    pipe.set('key1', 'value1')
    pipe.set('key2', 'value2')
    pipe.execute()
```

### Hash Tags

当需要将相关的 Key 分配到同一节点时，使用 Hash Tags：

```bash
# {} 内的内容决定槽位分配
user:{1001}:profile  # 都在同一个槽位
user:{1001}:orders
user:{1001}:cart

# 事务操作
MULTI
SET user:{1001}:name "张三"
SET user:{1001}:age 25
EXEC
```

## 缓存设计模式

### 📖 Cache-Aside Pattern

最常用的缓存模式，应用负责维护缓存。

```python
def get_user(user_id):
    """读操作：先查缓存，缓存不存在则查数据库"""
    cache_key = f'user:{user_id}'
    
    # 1. 先查缓存
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 2. 缓存不存在，查数据库
    user = db.query(User).get(user_id)
    if user:
        # 3. 写入缓存
        r.setex(cache_key, 3600, json.dumps(user.to_dict()))
    
    return user

def update_user(user_id, **data):
    """写操作：先更新数据库，再删除缓存"""
    # 1. 更新数据库
    db.query(User).filter_by(id=user_id).update(data)
    db.commit()
    
    # 2. 删除缓存（推荐删除而非更新）
    r.delete(f'user:{user_id}')
```

### 📖 Read-Through Pattern

缓存层负责读取数据，应用只与缓存交互。

```python
class ReadThroughCache:
    """Read-Through 缓存模式"""
    
    def __init__(self, redis_client, db_loader, ttl=3600):
        self.r = redis_client
        self.loader = db_loader  # 数据加载函数
        self.ttl = ttl
    
    def get(self, key):
        # 缓存层负责加载数据
        cached = self.r.get(key)
        if cached is not None:
            return json.loads(cached)
        
        # 从数据库加载
        data = self.loader(key)
        if data:
            self.r.setex(key, self.ttl, json.dumps(data))
        
        return data

# 使用
def load_user(user_id):
    return db.query(User).get(user_id)

cache = ReadThroughCache(r, load_user)
user = cache.get('user:1001')
```

### 📖 Write-Through Pattern

写入时同步更新缓存和数据库。

```python
class WriteThroughCache:
    """Write-Through 缓存模式"""
    
    def __init__(self, redis_client, db_writer, ttl=3600):
        self.r = redis_client
        self.writer = db_writer  # 数据写入函数
        self.ttl = ttl
    
    def set(self, key, value):
        # 先写入数据库
        self.writer(key, value)
        
        # 再更新缓存
        self.r.setex(key, self.ttl, json.dumps(value))
    
    def delete(self, key):
        # 删除数据库记录
        self.writer(key, None)
        
        # 删除缓存
        self.r.delete(key)
```

### 📖 Write-Behind Pattern

写入时只更新缓存，异步写入数据库。

```python
import threading
import queue

class WriteBehindCache:
    """Write-Behind 缓存模式"""
    
    def __init__(self, redis_client, db_writer, ttl=3600):
        self.r = redis_client
        self.writer = db_writer
        self.ttl = ttl
        self.write_queue = queue.Queue()
        
        # 启动异步写入线程
        self._start_writer_thread()
    
    def set(self, key, value):
        # 立即更新缓存
        self.r.setex(key, self.ttl, json.dumps(value))
        
        # 加入异步写入队列
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

## 内存优化技巧

### 🗜️ 数据结构选择

```python
# ❌ 错误：存储大量小对象
for i in range(100000):
    r.set(f'item:{i}', json.dumps({'id': i, 'value': f'v{i}'}))

# ✅ 正确：使用 Hash 存储
fields = {f'{i}': json.dumps({'id': i, 'value': f'v{i}'}) 
          for i in range(100000)}
r.hset('items', mapping=fields)

# ✅ 正确：使用压缩列表优化的小 Hash
# 配置：hash-max-listpack-entries 512
# 配置：hash-max-listpack-value 64
```

### 🗜️ Key 命名约定

```bash
# 推荐的 Key 命名规范
user:1001:profile      # 业务:ID:属性
order:2024:01:pending  # 业务:时间:状态
cache:product:2001     # 用途:业务:ID

# 控制长度，避免过长 Key
# ❌ this_is_a_very_long_key_name_that_wastes_memory:1001
# ✅ usr:1001
```

### 🗜️ 内存配置

```bash
# redis.conf
maxmemory 4gb                    # 最大内存限制
maxmemory-policy allkeys-lru     # 驱逐策略

# 数据结构优化配置
hash-max-listpack-entries 512    # Hash 转换阈值
hash-max-listpack-value 64
list-max-listpack-size -2        # List 压缩
set-max-intset-entries 512       # Set 转换阈值
zset-max-listpack-entries 128    # Sorted Set 转换阈值
zset-max-listpack-value 64
```

### 🗜️ 内存分析命令

```bash
# 查看内存使用
MEMORY USAGE key

# 查看内存统计
MEMORY STATS

# 分析大 Key
MEMORY DOCTOR

# 查看内存分配详情
INFO memory
```

## 客户端连接池配置

### Python (redis-py)

```python
from redis import Redis
from redis.connection import ConnectionPool

# 连接池配置
pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    password='your_password',
    
    # 连接池大小
    max_connections=100,
    
    # 超时配置
    socket_timeout=5,              # 操作超时
    socket_connect_timeout=5,      # 连接超时
    
    # 连接健康检查
    health_check_interval=30,      # 每30秒检查一次
    
    # 重试配置
    retry_on_timeout=True,
    
    # 编码
    decode_responses=True,
    
    # 连接池行为
    connection_class=Connection,   # 默认连接类
)

# 创建客户端
r = Redis(connection_pool=pool)

# 使用上下文管理器
def get_redis():
    return Redis(connection_pool=pool)

# 异步客户端 (redis-py 4.2+)
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.connection import ConnectionPool as AsyncPool

async_pool = AsyncPool(
    host='localhost',
    port=6379,
    max_connections=100,
    decode_responses=True
)

async def get_async_redis():
    return AsyncRedis(connection_pool=async_pool)
```

### Java (Jedis / Lettuce)

```java
// Jedis 连接池
JedisPoolConfig config = new JedisPoolConfig();
config.setMaxTotal(100);           // 最大连接数
config.setMaxIdle(50);             // 最大空闲连接
config.setMinIdle(10);             // 最小空闲连接
config.setMaxWaitMillis(3000);     // 获取连接最大等待时间
config.setTestWhileIdle(true);     // 空闲检查
config.setTimeBetweenEvictionRunsMillis(30000);

JedisPool pool = new JedisPool(config, "localhost", 6379, 5000, "password");

try (Jedis jedis = pool.getResource()) {
    jedis.set("key", "value");
}

// Lettuce (Spring Boot 推荐)
RedisStandaloneConfiguration config = new RedisStandaloneConfiguration();
config.setHostName("localhost");
config.setPort(6379);
config.setPassword("password");

LettuceConnectionFactory factory = new LettuceConnectionFactory(config);
factory.setShareNativeConnection(true);  // 共享连接
factory.setPoolConfig(new GenericObjectPoolConfig<>());
```

### Go (go-redis)

```go
import "github.com/redis/go-redis/v9"

rdb := redis.NewClient(&redis.Options{
    Addr:         "localhost:6379",
    Password:     "",
    DB:           0,
    
    // 连接池配置
    PoolSize:     100,              // 连接池大小
    MinIdleConns: 10,               // 最小空闲连接
    MaxIdleConns: 50,               // 最大空闲连接
    
    // 超时配置
    DialTimeout:  5 * time.Second,
    ReadTimeout:  3 * time.Second,
    WriteTimeout: 3 * time.Second,
    PoolTimeout:  4 * time.Second,
    
    // 健康检查
    ConnMaxIdleTime: 5 * time.Minute,
    ConnMaxLifetime: 30 * time.Minute,
})
```

## 常见反模式与最佳实践

### ⚠️ 避免 Big Key

```python
# ❌ 错误：存储大对象
big_data = json.dumps(large_object)  # 假设有 10MB
r.set('big:key', big_data)

# ✅ 正确：拆分为多个小 Key
for i, chunk in enumerate(chunks(large_object, 1000)):
    r.hset('big:key', f'chunk:{i}', json.dumps(chunk))

# 检查 Big Key
# redis-cli --bigkeys
# 或使用 MEMORY USAGE 命令
```

### ⚠️ 设置合理的 TTL

```python
# ❌ 错误：没有过期时间
r.set('session:abc', session_data)

# ✅ 正确：设置合理的过期时间
r.setex('session:abc', 3600, session_data)  # 1小时过期

# 缓存预热时设置随机过期时间，避免缓存雪崩
import random
ttl = 3600 + random.randint(0, 600)  # 1小时 + 随机0-10分钟
```

### ⚠️ 避免阻塞命令

```python
# ❌ 错误：在生产环境使用阻塞命令
result = r.keys('*')  # 阻塞，O(N)

# ✅ 正确：使用 SCAN 命令
cursor = 0
while True:
    cursor, keys = r.scan(cursor, match='user:*', count=100)
    # 处理 keys
    if cursor == 0:
        break

# ❌ 错误：大范围删除
r.delete(*r.keys('temp:*'))

# ✅ 正确：分批删除
def delete_pattern(pattern, batch_size=100):
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=batch_size)
        if keys:
            r.delete(*keys)
        if cursor == 0:
            break
```

### ⚠️ 热 Key 处理

```python
# ❌ 问题：单个 Key 被高频访问
def get_hot_data():
    return r.get('hot:key')

# ✅ 解决方案1：本地缓存 + Redis
from functools import lru_cache
import time

class HotKeyCache:
    def __init__(self, redis_client, local_ttl=1):
        self.r = redis_client
        self.local_ttl = local_ttl
        self.local_cache = {}
        self.local_expire = {}
    
    def get(self, key):
        now = time.time()
        
        # 先查本地缓存
        if key in self.local_cache:
            if now < self.local_expire.get(key, 0):
                return self.local_cache[key]
        
        # 查 Redis
        value = self.r.get(key)
        if value:
            self.local_cache[key] = value
            self.local_expire[key] = now + self.local_ttl
        
        return value

# ✅ 解决方案2：热 Key 分片
def get_hot_key_sharded(key, shards=10):
    shard = random.randint(0, shards - 1)
    return r.get(f'{key}:{shard}')

def set_hot_key_sharded(key, value, shards=10):
    pipe = r.pipeline()
    for i in range(shards):
        pipe.set(f'{key}:{i}', value)
    pipe.execute()
```

## 多级缓存架构

### 架构设计

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Local Cache │ ← L1 缓存（进程内，毫秒级）
│   (内存)    │
└──────┬──────┘
       │ Miss
       ▼
┌─────────────┐
│   Redis     │ ← L2 缓存（分布式，亚毫秒级）
│   Cluster   │
└──────┬──────┘
       │ Miss
       ▼
┌─────────────┐
│  Database   │ ← 数据源
└─────────────┘
```

### 实现示例

```python
import time
from threading import Lock
from functools import lru_cache

class MultiLevelCache:
    """多级缓存实现"""
    
    def __init__(self, redis_client, db_loader, local_ttl=1, redis_ttl=3600):
        self.r = redis_client
        self.loader = db_loader
        self.local_ttl = local_ttl
        self.redis_ttl = redis_ttl
        
        # 本地缓存
        self._local_cache = {}
        self._local_expire = {}
        self._lock = Lock()
    
    def get(self, key):
        # L1: 本地缓存
        with self._lock:
            if key in self._local_cache:
                if time.time() < self._local_expire.get(key, 0):
                    return self._local_cache[key]
                del self._local_cache[key]
                del self._local_expire[key]
        
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
        with self._lock:
            self._local_cache[key] = value
            self._local_expire[key] = time.time() + self.local_ttl
    
    def _set_redis(self, key, value):
        self.r.setex(key, self.redis_ttl, json.dumps(value))
    
    def invalidate(self, key):
        """失效缓存"""
        with self._lock:
            self._local_cache.pop(key, None)
            self._local_expire.pop(key, None)
        self.r.delete(key)
```

## 参考资料

- [Redis 官方文档](https://redis.io/docs/)
- [Redis 7.x Release Notes](https://redis.io/docs/latest/operate/oss_and_stack/management/upgrading/)
- [Redis 设计与实现](http://redisbook.com/)
- [redis-py 文档](https://redis-py.readthedocs.io/)
