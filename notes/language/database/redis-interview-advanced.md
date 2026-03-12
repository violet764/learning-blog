# Redis 进阶面试题

本文是 Redis 面试题的进阶篇，聚焦于更深入的技术细节和实战场景，涵盖数据结构原理、持久化机制、集群架构、分布式锁实现以及大厂实战经验。

---

## 一、数据结构深度篇

### 1. SDS（Simple Dynamic String）的结构是什么？为什么要存已用长度？

**答案：**

SDS 是 Redis 自己实现的字符串结构，而不是直接使用 C 语言的原生字符串。

**SDS 结构定义：**

```c
struct __attribute__ ((__packed__)) sdshdr8 {
    uint8_t len;        // 已使用长度（1字节）
    uint8_t alloc;      // 总分配长度（不含头和结束符）
    unsigned char flags; // 类型标识
    char buf[];         // 实际存储字符串的柔性数组
};
```

Redis 根据字符串长度提供多种 SDS 类型：
- `sdshdr5`：长度 < 32
- `sdshdr8`：长度 < 256
- `sdshdr16`：长度 < 65536
- `sdshdr32`：长度 < 4GB
- `sdshdr64`：长度 >= 4GB

**存储已用长度的好处：**

| 特性 | C 字符串 | SDS |
|------|----------|-----|
| 获取长度 | O(n) 遍历 | O(1) 直接读取 |
| 防止缓冲区溢出 | ❌ 无检查 | ✅ 空间检查 |
| 二进制安全 | ❌ 依赖 '\0' | ✅ len 决定长度 |
| 内存重分配 | 每次修改都可能 | 预分配+惰性释放 |

**代码示例：**

```c
// C 字符串获取长度：O(n)
size_t strlen_c(const char *s) {
    size_t len = 0;
    while (*s++) len++;
    return len;
}

// SDS 获取长度：O(1)
static inline size_t sdslen(const sds s) {
    unsigned char flags = s[-1];
    switch(flags & SDS_TYPE_MASK) {
        case SDS_TYPE_8:
            return SDS_HDR(8,s)->len;
        // ...
    }
}
```

**追问：SDS 的空间预分配策略是什么？**

**追问答案：**

当 SDS 需要扩展时，Redis 会预分配额外空间：

```c
// 扩展策略
new_len = old_len + add_len;

if (new_len < SDS_MAX_PREALLOC) {  // SDS_MAX_PREALLOC = 1024*1024 (1MB)
    new_alloc = new_len * 2;        // 小于 1MB：翻倍
} else {
    new_alloc = new_len + SDS_MAX_PREALLOC;  // 大于 1MB：只加 1MB
}
```

惰性释放：缩短字符串时不立即释放内存，而是保留空间供后续使用。

---

### 2. embstr 和 raw 编码的区别？什么时候转换？

**答案：**

Redis 的字符串对象有三种编码方式：`int`、`embstr`、`raw`。

**编码对比：**

```
┌─────────────────────────────────────────────────────────────┐
│                      embstr 编码                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ redisObject | sdshdr | 字符串内容                      │  │
│  │   (16字节)  | (3字节) | (实际内容)                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                    ↑ 连续内存，一次分配                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       raw 编码                             │
│  ┌──────────────┐        ┌──────────────────────────────┐  │
│  │ redisObject  │  ───→  │ sdshdr | 字符串内容            │  │
│  │   (16字节)   │  指针  │       | (实际内容)             │  │
│  └──────────────┘        └──────────────────────────────┘  │
│          ↑                           ↑                      │
│      对象内存                      SDS 内存                  │
│      (两次分配，内存不连续)                                   │
└─────────────────────────────────────────────────────────────┘
```

**转换条件：**

```c
// 编码选择逻辑
if (value 是整数 && 在 long 范围内) {
    encoding = OBJ_ENCODING_INT;
} else if (字符串长度 <= 44) {
    encoding = OBJ_ENCODING_EMBSTR;  // 44 字节是经验值
} else {
    encoding = OBJ_ENCODING_RAW;
}
```

**为什么是 44 字节？**

```
redisObject = 16 字节
sdshdr8 = 3 字节（len + alloc + flags）
字符串结尾 '\0' = 1 字节
剩余可用 = 64 - 16 - 3 - 1 = 44 字节
```

embstr 设计为一次内存分配，正好放入 Redis 的 jemalloc 的 64 字节内存块。

**追问：embstr 为什么不能修改？**

**追问答案：**

embstr 是只读的，任何修改操作都会导致转换为 raw 编码：

```bash
SET key "hellohellohellohellohellohellohellohellohellohello"  # 50字节，raw
SET key2 "hello"                                              # 5字节，embstr
APPEND key2 "world"                                           # 触发转换为 raw
```

原因：
1. embstr 内存连续，没有预留扩展空间
2. 修改需要重新分配整个内存块
3. 直接转 raw 更高效

---

### 3. QuickList 为什么用 listpack 替代 ziplist？

**答案：**

QuickList 是 List 的底层实现，是 linkedlist 和 ziplist 的结合体。在 Redis 7.0 中，ziplist 被替换为 listpack。

**ziplist 的问题：**

```
┌────────────────────────────────────────────────────────────┐
│                    ziplist 结构                            │
│  ┌──────┬──────┬────────┬────────┬─────┬────────┬────────┐│
│  │zlbytes│zltail│entry1  │entry2  │ ... │entryN  │zlend   ││
│  │ 4字节 │ 4字节 │        │        │     │        │ 1字节  ││
│  └──────┴──────┴────────┴────────┴─────┴────────┴────────┘│
│                              ↑                             │
│              每个 entry 的 prevlen 字段记录前一节点长度      │
└────────────────────────────────────────────────────────────┘

问题：级联更新（Cascade Update）
- 插入/删除节点可能导致 prevlen 字段大小变化（1字节 ↔ 5字节）
- 一个节点变化，后续所有节点可能都需要更新
- 最坏情况：O(n²) 的时间复杂度
```

**listpack 的改进：**

```
┌────────────────────────────────────────────────────────────┐
│                    listpack 结构                           │
│  ┌──────┬────────┬────────┬─────┬────────┬────────┐       │
│  │total │entry1  │entry2  │ ... │entryN  │end     │       │
│  │bytes │        │        │     │        │ 0xFF   │       │
│  └──────┴────────┴────────┴─────┴────────┴────────┘       │
│                          ↑                                 │
│        每个 entry 记录自己的长度，而非前一节点长度           │
└────────────────────────────────────────────────────────────┘

优势：
- 每个节点独立，修改不影响其他节点
- 解决了级联更新问题
- 内存使用更紧凑
```

**追问：QuickList 为什么不直接用 linkedlist？**

**追问答案：**

linkedlist 的问题：
1. 每个节点都有前后指针（16字节开销）
2. 内存碎片严重
3. 缓存不友好

QuickList 折中方案：
```
┌──────────────────────────────────────────────────────────────┐
│                       QuickList                              │
│  head ──→ [QuickListNode] ──→ [QuickListNode] ──→ ...       │
│                │                      │                      │
│                ↓                      ↓                      │
│           [listpack]            [listpack]                  │
│           (压缩节点)            (压缩节点)                   │
│                                                              │
│ 配置参数：                                                    │
│ - list-max-listpack-size: 每个节点的大小限制                  │
│ - list-compress-depth: 压缩深度（中间节点压缩）               │
└──────────────────────────────────────────────────────────────┘
```

---

### 4. SkipList 的查询时间复杂度？为什么用 1/4 概率？

**答案：**

SkipList 是 ZSET 的底层实现之一，通过多层索引实现快速查找。

**结构示意：**

```
Level 4:        ┌───────→[50]──────────────────────────────→NULL
                |
Level 3:        └───────→[50]────────────→[70]─────────────→NULL
                |                      |
Level 2:   [20]─┴───────→[50]─→[60]────┴────[70]──────────→NULL
           |           |         |           |
Level 1:   [20]─→[30]─→[50]─→[60]─→[70]─→[80]─→[90]────────→NULL
           |     |     |     |     |     |     |
Level 0:   [20]─→[30]─→[40]─→[50]─→[60]─→[70]─→[80]─→[90]──→NULL
```

**时间复杂度分析：**

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 查找 | O(log n) | 类似二分查找 |
| 插入 | O(log n) | 先查找，再插入各层 |
| 删除 | O(log n) | 先查找，再删除各层 |
| 范围查询 | O(log n + k) | k 为返回元素个数 |

**为什么用 1/4 概率上跳？**

```c
// Redis 源码中的概率设置
#define ZSKIPLIST_P 0.25  // 1/4 概率

int zslRandomLevel(void) {
    int level = 1;
    while ((random() & 0xFFFF) < (ZSKIPLIST_P * 0xFFFF)) {
        level += 1;
    }
    return level;
}
```

**数学证明：**

```
期望层数 = 1/(1-p) = 1/(1-0.25) = 4/3 ≈ 1.33 层

总指针数期望 = n × (1/(1-p)) = n × 4/3

空间复杂度 = O(n)

对比 1/2 概率：
- 1/2 概率：期望层数 = 2，总指针数 = 2n
- 1/4 概率：期望层数 = 1.33，总指针数 = 1.33n
```

**追问：为什么不用红黑树？**

**追问答案：**

SkipList 相比红黑树的优势：
1. **实现简单**：代码量少，易于理解和维护
2. **范围查询高效**：找到起点后直接遍历底层链表
3. **内存友好**：不需要旋转操作，内存分配更简单
4. **并发优势**：更容易实现无锁版本

```c
// SkipList 范围查询示例
unsigned long zslGetRank(zskiplist *zsl, double score, sds ele) {
    zskiplistNode *x = zsl->header;
    unsigned long rank = 0;
    
    for (int i = zsl->level-1; i >= 0; i--) {
        while (x->level[i].forward &&
               (x->level[i].forward->score < score ||
                (x->level[i].forward->score == score &&
                 sdscmp(x->level[i].forward->ele, ele) <= 0))) {
            rank += x->level[i].span;
            x = x->level[i].forward;
        }
    }
    return rank;
}
```

---

### 5. intset 什么时候升级？能降级吗？

**答案：**

intset 是 Set 的底层编码之一，用于存储整数值。

**intset 结构：**

```c
typedef struct intset {
    uint32_t encoding;  // 编码类型：int16、int32、int64
    uint32_t length;    // 元素个数
    int8_t contents[];  // 存储元素的柔性数组
} intset;
```

**升级触发条件：**

```bash
# 初始状态：int16 编码
SADD set 1 2 3      # encoding = INTSET_ENC_INT16

# 插入 int16 范围外的值，触发升级
SADD set 65535      # 超出 int16 范围，升级为 int32
SADD set 2147483647 # 超出 int32 范围，升级为 int64
```

**升级过程：**

```
步骤1: 插入新元素 65535 到 [1, 2, 3]
       发现超出 int16 范围，需要升级到 int32

步骤2: 重新计算内存大小
       旧大小 = 3 × 2字节 = 6字节
       新大小 = 4 × 4字节 = 16字节

步骤3: 从后往前移动元素（避免覆盖）
       [1, 2, 3] → [1, 2, 3, _]
       移动 3 到位置 3: [1, 2, _, 3]
       移动 2 到位置 2: [1, _, 2, 3]
       移动 1 到位置 1: [_, 1, 2, 3]

步骤4: 插入新元素
       [65535, 1, 2, 3] → 排序后 [1, 2, 3, 65535]
```

**追问：为什么不能降级？**

**追问答案：**

Redis **不支持降级**，原因：

1. **复杂度高**：需要判断所有元素是否都满足更小编码
2. **收益低**：降级只节省少量内存，但需要全量遍历
3. **历史原因**：设计时认为升级是低频操作

```c
// 源码中只有升级逻辑，没有降级逻辑
static intset *intsetUpgradeAndAdd(intset *is, int64_t value) {
    uint8_t curenc = intrev32ifbe(is->encoding);
    uint8_t newenc = _intsetValueEncoding(value);
    int length = intrev32ifbe(is->length);
    int prepend = value < 0 ? 1 : 0;

    is->encoding = intrev32ifbe(newenc);
    // 只有扩展，没有缩小的逻辑
    // ...
}
```

---

### 6. Redis 7.0 的 listpack 有什么改进？

**答案：**

listpack 是 Redis 7.0 引入的新数据结构，用于替代 ziplist。

**主要改进：**

| 特性 | ziplist | listpack |
|------|---------|----------|
| 级联更新 | ❌ 存在 | ✅ 已解决 |
| 内存布局 | prevlen 在前 | 元素长度在后 |
| 元素编码 | 复杂 | 统一格式 |
| Stream 使用 | listpacks | 纯 listpack |

**元素编码格式：**

```
┌─────────────────────────────────────────────────────────────┐
│                    listpack 元素格式                        │
│  ┌──────────────────────┬──────────────────────┐           │
│  │      element         │    element-len       │           │
│  │  (string/integer)    │    (变长编码)         │           │
│  └──────────────────────┴──────────────────────┘           │
│                            ↑                                │
│                   记录自己的长度，而非前一元素                │
└─────────────────────────────────────────────────────────────┘

element-len 编码规则：
- 长度 < 128：1字节存储
- 长度 < 16383：2字节存储
- 更大：5字节存储
```

**应用场景：**

```bash
# Hash 底层编码
HSET user name "alice" age 25
# Redis 7.0 之前：小数据用 ziplist，大数据用 hashtable
# Redis 7.0：小数据用 listpack，大数据用 hashtable

# List 底层的 QuickList 节点
# Redis 7.0 之前：ziplist 作为节点
# Redis 7.0：listpack 作为节点
```

**追问：每种数据类型的编码转换条件？**

**追问答案：**

```c
// String 编码转换
int → embstr → raw
|     |        |
整数   ≤44字节  >44字节

// List 编码转换
listpack → QuickList
|
元素少且小

// Hash 编码转换
listpack → hashtable
|
元素数 > 512 或单值 > 64 字节
（可通过 hash-max-listpack-entries/field-value 配置）

// Set 编码转换
intset → hashtable
|
元素数 > 128 或非整数
（可通过 set-max-intset-entries 配置）

// ZSet 编码转换
listpack → skiplist + dict
|
元素数 > 128 或单值 > 64 字节
（可通过 zset-max-listpack-entries/field-value 配置）
```

---

### 7. Redis 的 Hash 表是怎么实现的？rehash 怎么做？

**答案：**

Redis 使用字典（dict）作为核心数据结构，底层是哈希表。

**字典结构：**

```c
typedef struct dict {
    dictType *type;      // 类型特定函数
    void *privdata;      // 私有数据
    dictht ht[2];        // 两个哈希表（用于 rehash）
    long rehashidx;      // rehash 进度索引（-1 表示未进行）
    int16_t pauserehash; // rehash 暂停标记
} dict;

typedef struct dictht {
    dictEntry **table;   // 哈希表数组
    unsigned long size;  // 哈希表大小
    unsigned long sizemask; // 哈希表大小掩码
    unsigned long used;  // 已有节点数量
} dictht;
```

**渐进式 rehash 过程：**

```
阶段1: 初始状态
┌─────────────────────────────────┐
│  ht[0]: size=4, used=3          │
│  [k1] → [k2] → [k3] → NULL      │
└─────────────────────────────────┘
rehashidx = -1

阶段2: 开始 rehash（扩容）
┌─────────────────────────────────┐
│  ht[0]: size=4                  │
│  [k1] → [k2] → [k3] → NULL      │
├─────────────────────────────────┤
│  ht[1]: size=8, used=0          │
│  [空] → [空] → ... → NULL       │
└─────────────────────────────────┘
rehashidx = 0

阶段3: 渐进迁移（每次操作迁移一部分）
┌─────────────────────────────────┐
│  ht[0]: bucket[0..1] 已迁移     │
│  NULL → [k3] → ...              │
├─────────────────────────────────┤
│  ht[1]: bucket[0..1] 已填充     │
│  [k1] → [k2] → [空] → ...       │
└─────────────────────────────────┘
rehashidx = 2

阶段4: rehash 完成
┌─────────────────────────────────┐
│  ht[0]: 释放，置空              │
├─────────────────────────────────┤
│  ht[1]: size=8, used=3          │
│  [k1] → [k2] → [k3] → ...       │
└─────────────────────────────────┘
rehashidx = -1，ht[1] 变为 ht[0]
```

**rehash 触发条件：**

```c
// 扩容条件
if (used / size >= 5) {           // 负载因子 >= 5
    // 强制扩容
} else if (used / size >= 1 && !正在rehash && 没有执行BGSAVE/BGREWRITEAOF) {
    // 正常扩容
}

// 缩容条件
if (used / size < 0.1) {          // 负载因子 < 0.1
    // 开始缩容
}
```

**追问：为什么用两个哈希表？**

**追问答案：**

渐进式 rehash 的关键：
1. **避免阻塞**：一次性迁移百万级数据会阻塞很久
2. **分散开销**：将 rehash 分摊到每次操作中
3. **平滑过渡**：读取时同时查两个表

```c
// 渐进式 rehash 的实现
static void _dictRehashStep(dict *d) {
    // 每次迁移一个桶
    if (d->iterators == 0) {
        dictRehash(d, 1);
    }
}

// 迁移 n 个桶
int dictRehash(dict *d, int n) {
    int emptyvisits = n * 10;
    while (n-- && d->ht[0].used != 0) {
        dictEntry *de, *nextde;
        // 迁移逻辑...
    }
    // 完成检查...
}
```

---

### 8. Redis 的过期策略是什么？

**答案：**

Redis 采用**惰性删除 + 定期删除**的组合策略。

**惰性删除：**

```c
// 每次访问 key 时检查过期
robj *lookupKeyRead(redisDb *db, robj *key) {
    robj *val;
    expireIfNeeded(db, key);  // 先检查过期
    val = dictFetchValue(db->dict, key);
    return val;
}

int expireIfNeeded(redisDb *db, robj *key) {
    if (!keyIsExpired(db, key)) return 0;
    
    // 从数据库删除
    dictDelete(db->dict, key->ptr);
    dictDelete(db->expires, key->ptr);
    return 1;
}
```

**定期删除：**

```c
// 定期删除逻辑
void activeExpireCycle(int type) {
    // 默认每次处理 16 个数据库
    // 每个数据库随机检查 20 个 key
    // 如果过期比例 > 25%，继续检查
    
    do {
        for (j = 0; j < dbs_per_call; j++) {
            // 随机选择 key
            if (num_expired > ACTIVE_EXPIRE_CYCLE_LOOKUPS_PER_LOOP / 4) {
                // 过期比例高，继续
            } else {
                break;
            }
        }
    } while (iteration < 16 && elapsed < 25);  // 最多 25ms
}
```

**策略对比：**

| 策略 | 优点 | 缺点 |
|------|------|------|
| 定时删除 | 内存友好 | CPU 压力大，可能阻塞 |
| 惰性删除 | CPU 友好 | 内存泄漏风险 |
| 定期删除 | 折中方案 | 参数需要调优 |
| Redis 组合 | 平衡 CPU 和内存 | 仍有少量过期 key 残留 |

**追问：过期 key 怎么保证不被 RDB 持久化？**

**追问答案：**

```c
// RDB 保存时的检查
int rdbSaveRio(rio *rdb, int *error, int flags, rdbSaveInfo *rsi) {
    // 遍历所有 key
    while((de = dictNext(di)) != NULL) {
        sds keystr = dictGetKey(de);
        robj key, *o = dictGetVal(de);
        long long expire;
        
        // 获取过期时间
        expire = getExpire(db, &key);
        
        // 如果过期了，跳过不保存
        if (expire != -1 && expire < now) continue;
        
        // 保存 key-value
        rdbSaveKeyValuePair(rdb, &key, o, expire);
    }
}
```

---

### 9. Redis 的内存淘汰策略有哪些？

**答案：**

当内存达到 `maxmemory` 限制时，Redis 提供多种淘汰策略。

**策略列表：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| noeviction | 不淘汰，写入报错 | 数据不可丢失 |
| allkeys-lru | 所有 key，LRU 淘汰 | 缓存场景 |
| volatile-lru | 仅过期 key，LRU 淘汰 | 混合使用场景 |
| allkeys-lfu | 所有 key，LFU 淘汰 | 热点数据明显 |
| volatile-lfu | 仅过期 key，LFU 淘汰 | 混合使用场景 |
| allkeys-random | 所有 key，随机淘汰 | 无访问热点 |
| volatile-random | 仅过期 key，随机淘汰 | 混合使用场景 |
| volatile-ttl | 淘汰 TTL 最短的 | 有明确时效性 |

**LRU 实现原理：**

```c
// Redis 使用采样近似 LRU
// 每个 key 记录最后访问时间（24位）

typedef struct redisObject {
    unsigned type:4;
    unsigned encoding:4;
    unsigned lru:LRU_BITS;  // 24位 LRU 时间戳
    int refcount;
    void *ptr;
} robj;

// 淘汰时随机采样
#define MAXMEMORY_FLAG_LRU (1<<0)
#define EVPOOL_SIZE 16  // 采样池大小

void evictionPoolPopulate(int dbid, dict *sampledict, 
                          struct evictionPoolEntry *pool) {
    // 随机采样 N 个 key
    // 放入淘汰池，按空闲时间排序
    // 淘汰空闲时间最长的
}
```

**LFU 实现原理：**

```c
// LFU 使用计数器 + 衰减机制
// lru 字段重新解释：
// - 高 16 位：最后衰减时间
// - 低 8 位：访问计数

uint8_t LFULogIncr(uint8_t counter) {
    if (counter == 255) return 255;
    double r = (double)rand()/RAND_MAX;
    double baseval = counter - LFU_INIT_VAL;
    if (baseval < 0) baseval = 0;
    double p = 1.0/(baseval*server.lfu_log_factor+1);
    if (r < p) counter++;
    return counter;
}

// 衰减：根据时间间隔减少计数
unsigned long LFUDecrAndReturn(robj *o) {
    unsigned long ldt = LRU_DEC(o->lru);
    unsigned long counter = LFU_GET(o->lru);
    unsigned long num_periods = server.lfu_decay_time;
    
    if (num_periods != 0) {
        unsigned long decay = (time(NULL) - ldt) / num_periods;
        if (decay > counter) counter = 0;
        else counter -= decay;
    }
    return counter;
}
```

**追问：如何选择合适的淘汰策略？**

**追问答案：**

```bash
# 纯缓存场景：allkeys-lru
maxmemory-policy allkeys-lru

# 有过期时间的缓存：volatile-lru
maxmemory-policy volatile-lru

# 热点数据明显：allkeys-lfu
maxmemory-policy allkeys-lfu

# 数据不能丢失：noeviction
maxmemory-policy noeviction
```

---

### 10. Redis 的 bigkey 怎么发现和处理？

**答案：**

bigkey 是指占用内存过大或元素过多的 key。

**bigkey 定义：**

| 类型 | bigkey 标准 |
|------|------------|
| String | value > 10KB |
| Hash | 元素数 > 5000 |
| List | 元素数 > 5000 |
| Set | 元素数 > 5000 |
| ZSet | 元素数 > 5000 |

**发现方法：**

```bash
# 方法1: redis-cli --bigkeys
redis-cli --bigkeys -i 0.1  # 每 100 条命令休眠 0.1 秒

# 方法2: MEMORY USAGE 命令
MEMORY USAGE mykey  # 返回字节数

# 方法3: 使用 SCAN + 预估
redis-cli --scan | xargs redis-cli MEMORY USAGE
```

**处理方案：**

```bash
# 错误做法：直接 DEL
DEL bigkey  # 可能阻塞几秒甚至几十秒

# 正确做法：分批删除
# String 类型：直接 UNLINK（异步删除）
UNLINK bigstring

# Hash 类型：分批 HDEL
HSCAN bighash 0 COUNT 100
# 获取一批字段后执行
HDEL bighash field1 field2 ... field100

# List 类型：使用 LTRIM 逐步裁剪
LTRIM biglist 0 -101  # 保留最后 100 个
LTRIM biglist 0 -101
# ... 循环直到为空

# Set 类型：分批 SREM
SSCAN bigset 0 COUNT 100
SREM bigset member1 member2 ...

# ZSet 类型：分批 ZREM
ZSCAN bigzset 0 COUNT 100
ZREM bigzset member1 member2 ...
```

**Python 删除脚本示例：**

```python
import redis

def delete_big_hash(r, key, batch_size=100):
    """分批删除大 Hash"""
    cursor = 0
    while True:
        cursor, fields = r.hscan(key, cursor, count=batch_size)
        if fields:
            r.hdel(key, *fields.keys())
        if cursor == 0:
            break
    r.delete(key)

def delete_big_zset(r, key, batch_size=100):
    """分批删除大 ZSet"""
    while r.zcard(key) > 0:
        # 获取并删除最小的 batch_size 个元素
        members = r.zrange(key, 0, batch_size - 1)
        if members:
            r.zrem(key, *members)
    r.delete(key)

client = redis.Redis(host='localhost', port=6379)
delete_big_hash(client, 'big_hash_key')
```

**追问：bigkey 有什么危害？**

**追问答案：**

1. **内存不均衡**：集群中某节点内存占用高
2. **阻塞风险**：删除、序列化操作耗时长
3. **网络拥塞**：大 value 传输占用带宽
4. **过期删除阻塞**：过期 bigkey 删除时卡顿

---


## 二、持久化深度篇

### 1. RDB 的 fork() 为什么会阻塞？

**答案：**

RDB 持久化通过 `fork()` 创建子进程来执行，但 fork 操作可能会阻塞主进程。

**阻塞原因分析：**

```
┌─────────────────────────────────────────────────────────────┐
│                    fork() 阻塞原因                          │
├─────────────────────────────────────────────────────────────┤
│  1. 内存复制开销                                            │
│     - fork() 需要复制父进程的页表                            │
│     - 页表大小 = 进程内存 / 页大小 × 条目大小                 │
│     - 例如：10GB 内存，页表约 20MB                           │
│     - 复制 20MB 需要几百毫秒                                 │
├─────────────────────────────────────────────────────────────┤
│  2. COW 缺页中断                                            │
│     - fork() 后主进程首次写内存触发缺页                      │
│     - 大量写入时产生大量缺页中断                             │
├─────────────────────────────────────────────────────────────┤
│  3. 透明大页（THP）问题                                      │
│     - 开启 THP 时，页大小为 2MB                              │
│     - 复制页表更快，但 COW 复制内存更慢                       │
└─────────────────────────────────────────────────────────────┘
```

**影响因素：**

| 因素 | 影响 | 说明 |
|------|------|------|
| 内存大小 | 正相关 | 内存越大，页表越大 |
| 写入频率 | 正相关 | 写入越多，COW 越多 |
| THP 状态 | 复杂 | 开启可能更慢 |
| 系统负载 | 正相关 | 系统繁忙时 fork 更慢 |

**优化方案：**

```bash
# 1. 关闭透明大页
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 2. 控制 Redis 内存大小
# 单实例不超过 10GB

# 3. 监控 fork 耗时
INFO stats | grep latest_fork_usec

# 4. 合理配置 RDB 触发条件
# 避免频繁触发
save 900 1     # 900秒内至少1次修改
save 300 10    # 300秒内至少10次修改
save 60 10000  # 60秒内至少10000次修改
```

**追问：如何判断 fork 是否阻塞？**

**追问答案：**

```bash
# 查看 latest_fork_usec
redis-cli INFO stats | grep latest_fork_usec
# latest_fork_usec:1234  # 上次 fork 耗时 1.234ms

# 如果 > 1秒，需要优化
# 如果 > 10秒，严重影响生产

# 日志中的警告
# Background saving started by pid 1234
# * Fork CoW for RDB: current 1024 MB, peak 1024 MB, average 1024 MB
```

---

### 2. 写时复制（COW）是怎么工作的？

**答案：**

COW（Copy-On-Write）是 fork() 实现内存共享的核心机制。

**工作原理：**

```
阶段1: fork() 刚完成
┌─────────────────────────────────────────────────────────────┐
│   父进程                    子进程                          │
│  ┌──────────┐              ┌──────────┐                    │
│  │ 页表副本  │              │ 页表副本  │                    │
│  └────┬─────┘              └────┬─────┘                    │
│       │                         │                          │
│       └──────────┬──────────────┘                          │
│                  ↓                                          │
│         ┌────────────────┐                                  │
│         │  共享物理内存   │  ← 所有页都标记为只读            │
│         │  [Page A][B][C]│                                  │
│         └────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘

阶段2: 父进程修改 Page A
┌─────────────────────────────────────────────────────────────┐
│   父进程                    子进程                          │
│  ┌──────────┐              ┌──────────┐                    │
│  │ 页表副本  │              │ 页表副本  │                    │
│  └────┬─────┘              └────┬─────┘                    │
│       │                         │                          │
│       ↓                         ↓                          │
│  ┌─────────┐            ┌────────────────┐                 │
│  │Page A'  │            │  共享物理内存   │                 │
│  │(新副本) │            │  [Page A][B][C]│                 │
│  └─────────┘            └────────────────┘                 │
│       ↑                    ↑                                │
│   父进程独有              子进程指向原页                     │
└─────────────────────────────────────────────────────────────┘
```

**代码层面理解：**

```c
// fork() 后的内存状态
// 父进程和子进程共享物理内存
// 页表项标记为只读

// 当父进程写入时
void *ptr = shared_memory;
ptr[0] = 'x';  // 触发缺页中断

// 缺页中断处理流程
void page_fault_handler(address) {
    if (is_cow_page(address)) {
        // 1. 分配新的物理页
        new_page = alloc_page();
        
        // 2. 复制原页内容
        copy_page(new_page, original_page);
        
        // 3. 更新页表
        update_page_table(address, new_page, READ_WRITE);
        
        // 4. 重新执行写入指令
    }
}
```

**RDB 期间的内存增长：**

```
假设：
- Redis 内存：10GB
- RDB 耗时：5分钟
- 写入 QPS：50000
- 每次写入：100字节

理论新增内存 = 50000 × 100 × 300秒 / (1024³) ≈ 1.4GB

实际情况：由于 COW，大约复制 10-20% 的内存
```

**追问：如何减少 RDB 期间的内存占用？**

**追问答案：**

```bash
# 1. 减少 RDB 期间的写入
# 业务层面控制写入量

# 2. 使用更大内存的机器
# 确保内存足够容纳 COW 开销

# 3. 合理配置触发条件
# 在低峰期触发 RDB

# 4. 使用 AOF 替代
# AOF 不需要 fork 全量内存
```

---

### 3. AOF 重写时还能写入吗？怎么保证数据不丢？

**答案：**

AOF 重写期间可以继续写入，通过**双重写机制**保证数据不丢失。

**AOF 重写流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    AOF 重写时序图                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  主进程                      子进程                         │
│    │                           │                           │
│    │──── fork() ──────────────→│                           │
│    │                           │                           │
│    │  正常处理写请求            │  生成新 AOF 文件          │
│    │       │                    │       │                   │
│    │       ↓                    │       ↓                   │
│    │  ┌─────────────────┐       │  [新 AOF 文件]            │
│    │  │1. 写入旧 AOF    │       │                           │
│    │  │2. 写入 AOF 缓冲区│       │                           │
│    │  └─────────────────┘       │                           │
│    │                           │                           │
│    │←──── 完成信号 ────────────│                           │
│    │                           │                           │
│    │  将 AOF 缓冲区写入新文件   │                           │
│    │       │                    │                           │
│    │       ↓                    │                           │
│    │  原子性替换旧 AOF          │                           │
│    │  rename(new_aof, aof)     │                           │
│    │                           │                           │
└─────────────────────────────────────────────────────────────┘
```

**关键数据结构：**

```c
struct redisServer {
    sds aof_buf;          // AOF 写缓冲区（正常写入）
    list *aof_rewrite_buf_blocks;  // AOF 重写缓冲区（重写期间累积）
    int aof_fd;           // AOF 文件描述符
    int aof_rewrite_scheduled;  // 是否有重写等待
    pid_t aof_child_pid;  // 子进程 PID
};
```

**追问：如果重写过程中崩溃会怎样？**

**追问答案：**

```
场景1: 子进程崩溃
- 主进程收到信号
- 重写失败，旧 AOF 文件完整无损
- 数据零丢失

场景2: 主进程崩溃
- 重写未完成，新 AOF 文件不完整
- 重启时加载旧 AOF 文件
- AOF 重写缓冲区的数据丢失（未写入磁盘）
- 丢失 = 重写期间未刷盘的数据

场景3: 机器断电
- 两个 AOF 文件都可能不完整
- Redis 启动时检测并选择完整文件
- 使用 redis-check-aof 修复
```

```bash
# 修复 AOF 文件
redis-check-aof --fix appendonly.aof

# 检查 AOF 文件状态
redis-check-aof appendonly.aof
```

---

### 4. AOF 的三种同步策略各有什么优缺点？

**答案：**

AOF 提供三种刷盘策略，由 `appendfsync` 配置控制。

**三种策略对比：**

| 策略 | 配置值 | 刷盘时机 | 数据安全 | 性能 | 推荐场景 |
|------|--------|----------|----------|------|----------|
| Always | always | 每次写入 | 最高 | 最差 | 数据极重要 |
| Everysec | everysec | 每秒一次 | 较高 | 较好 | **默认推荐** |
| No | no | 由操作系统决定 | 最低 | 最好 | 可容忍丢失 |

**实现原理：**

```c
// always: 每次写入都刷盘
void flushAppendOnlyFile(int force) {
    if (server.aof_fsync == AOF_FSYNC_ALWAYS) {
        aof_fsync(server.aof_fd);  // 阻塞刷盘
    }
}

// everysec: 后台线程每秒刷盘
void aof_background_fsync(int fd) {
    // 创建后台任务
    bioCreateBackgroundJob(BIO_AOF_FSYNC, fd);
}

// no: 交给操作系统
// 不主动调用 fsync
```

**everysec 的数据丢失窗口：**

```
┌─────────────────────────────────────────────────────────────┐
│                    时间轴示意                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T0          T1          T2          T3                     │
│  │           │           │           │                      │
│  ↓           ↓           ↓           ↓                      │
│  [刷盘]      [写入内存]   [断电!]     [本该刷盘]             │
│                                                             │
│  T1-T2 之间的写入丢失，最多丢失约 1 秒数据                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：always 策略为什么性能差？**

**追问答案：**

```c
// 每次 fsync 都是系统调用
// 需要等待磁盘写入完成

// 性能对比（假设）
// always:  QPS ≈ 几千（每次写入都等磁盘）
// everysec: QPS ≈ 几万（写入内存，后台刷盘）
// no:      QPS ≈ 几十万（完全不等待）

// SSD 和 HDD 的差异
// SSD fsync: ~0.1ms
// HDD fsync: ~10ms（相差 100 倍）
```

---

### 5. 混合持久化的具体过程？

**答案：**

混合持久化是 Redis 4.0 引入的特性，结合 RDB 和 AOF 的优点。

**配置方式：**

```bash
# 开启 AOF
appendonly yes

# 开启混合持久化
aof-use-rdb-preamble yes
```

**持久化过程：**

```
┌─────────────────────────────────────────────────────────────┐
│                   混合持久化文件结构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RDB 格式数据                            │   │
│  │  ┌─────┬─────┬──────────────────────────┬─────┐    │   │
│  │  │REDIS│版本│    数据库内容（压缩）       │校验和│    │   │
│  │  └─────┴─────┴──────────────────────────┴─────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AOF 格式增量数据                        │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │ SET key1 value1                              │  │   │
│  │  │ SET key2 value2                              │  │   │
│  │  │ ... (重写期间的新增操作)                       │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**加载顺序：**

```c
int loadAppendOnlyFile(char *filename) {
    // 1. 检查是否为混合格式
    if (checkRdbPreamble(filename)) {
        // 2. 先加载 RDB 部分
        rdbLoadRio();
    }
    
    // 3. 再加载 AOF 增量部分
    while(1) {
        if (readCommand() == EOF) break;
        executeCommand();
    }
}
```

**对比优势：**

| 特性 | RDB | AOF | 混合持久化 |
|------|-----|-----|-----------|
| 恢复速度 | 快 | 慢 | 较快 |
| 数据安全 | 可能丢失 | 最多丢 1 秒 | 最多丢 1 秒 |
| 文件大小 | 小 | 大 | 较小 |
| 可读性 | 不可读 | 可读 | 部分可读 |

**追问：大 Redis 实例怎么做持久化优化？**

**追问答案：**

```bash
# 1. 使用混合持久化
appendonly yes
aof-use-rdb-preamble yes

# 2. 调整 AOF 重写阈值
auto-aof-rewrite-percentage 100   # 文件翻倍时触发
auto-aof-rewrite-min-size 64mb    # 最小文件大小

# 3. 控制 AOF 刷盘策略
appendfsync everysec

# 4. 分片降低单实例内存
# 每个 Redis 实例控制在 10GB 以内

# 5. 使用 SSD 硬盘
# 提高磁盘 IO 性能

# 6. 监控持久化状态
INFO persistence
# aof_rewrite_in_progress: 0
# aof_last_bgrewrite_status: ok
# aof_current_size: 1024
```

---

### 6. RDB 和 AOF 各自的优缺点？

**答案：**

| 维度 | RDB | AOF |
|------|-----|-----|
| **文件大小** | 小（压缩二进制） | 大（文本命令） |
| **恢复速度** | 快 | 慢 |
| **数据安全** | 低（可能丢失分钟级数据） | 高（最多丢 1 秒） |
| **系统开销** | fork 时有开销 | 持续写入开销 |
| **适用场景** | 备份、灾难恢复 | 高可用、实时性要求高 |

**选择建议：**

```
┌─────────────────────────────────────────────────────────────┐
│                    持久化选择决策树                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  数据可以丢失几分钟？                                        │
│       │                                                     │
│       ├── 是 ──→ 只用 RDB                                   │
│       │         （简单、高效）                               │
│       │                                                     │
│       └── 否 ──→ 数据量 > 10GB？                            │
│                    │                                        │
│                    ├── 是 ──→ 混合持久化                     │
│                    │         （兼顾速度和安全）              │
│                    │                                        │
│                    └── 否 ──→ AOF                           │
│                              （数据更安全）                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 7. 如何选择持久化策略？

**答案：**

根据业务场景选择：

```bash
# 场景1: 纯缓存，可接受数据丢失
save ""              # 关闭 RDB
appendonly no        # 关闭 AOF

# 场景2: 数据重要，允许丢失几分钟
save 900 1
save 300 10
appendonly no

# 场景3: 数据很重要，允许丢失 1 秒
appendonly yes
appendfsync everysec

# 场景4: 数据极重要，不能丢失
appendonly yes
appendfsync always

# 场景5: 大数据量，推荐配置
appendonly yes
appendfsync everysec
aof-use-rdb-preamble yes
save 900 1
```

---

### 8. AOF 文件损坏怎么修复？

**答案：**

```bash
# 检查 AOF 文件
redis-check-aof appendonly.aof

# 修复 AOF 文件
redis-check-aof --fix appendonly.aof

# 修复时会提示
# AOF analyzed: this file looks truncated.
# Would you like to use the preceding AOF file as is? (y/n)
# 输入 y 确认

# 如果有 RDB 备份，可以结合使用
# 1. 备份当前 AOF
# 2. 尝试修复
# 3. 如果修复失败，使用 RDB 恢复
```

---


## 三、集群深度篇

### 1. Redis Cluster 的槽位是怎么分配的？

**答案：**

Redis Cluster 采用**哈希槽（Hash Slot）**分区，共有 16384 个槽位。

**槽位分配原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    槽位分配示意图                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Key 计算槽位公式：                                          │
│  slot = CRC16(key) % 16384                                  │
│                                                             │
│  16384 个槽位分配到 3 个主节点：                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    16384 槽位                        │  │
│  │  [0-5460]  [5461-10922]  [10923-16383]              │  │
│  │     ↓           ↓              ↓                     │  │
│  │  Node A      Node B         Node C                   │  │
│  │  (主节点)    (主节点)       (主节点)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  为什么是 16384？                                           │
│  - CRC16 输出 16 位，最大 65535                             │
│  - 16384 = 2^14，计算方便                                   │
│  - 槽位太少：数据分布不均匀                                  │
│  - 槽位太多：节点间消息交换开销大                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**槽位相关命令：**

```bash
# 查看集群槽位分配
CLUSTER NODES

# 查看某个 key 的槽位
CLUSTER KEYSLOT mykey
# 返回：14687

# 查看槽位属于哪个节点
CLUSTER GETKEYSINSLOT 14687 10
# 返回该槽位中的 key

# 手动迁移槽位
CLUSTER SETSLOT 1000 IMPORTING <source_node_id>
CLUSTER SETSLOT 1000 MIGRATING <target_node_id>
```

**追问：槽位数量可以修改吗？**

**追问答案：**

槽位数量 **16384 是固定的**，不能修改。

原因：
1. 编码在协议和代码中
2. 修改会导致不兼容
3. 16384 对大多数场景足够

---

### 2. 客户端怎么知道 key 在哪个节点？

**答案：**

客户端通过**槽位映射表**定位 key 所在节点。

**定位流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Key 定位流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  客户端                                                     │
│    │                                                        │
│    │ 1. 连接集群任意节点                                    │
│    │ 2. 获取槽位映射表 CLUSTER SLOTS                        │
│    ↓                                                        │
│  ┌───────────────────────────────────────────────────┐     │
│  │              本地槽位映射缓存                       │     │
│  │  ┌─────────┬─────────┬─────────┐                 │     │
│  │  │0-5460   │5461-10922│10923-16383│              │     │
│  │  │Node A   │Node B   │Node C    │               │     │
│  │  └─────────┴─────────┴─────────┘                 │     │
│  └───────────────────────────────────────────────────┘     │
│    │                                                        │
│    │ 3. 计算槽位：CRC16(key) % 16384                        │
│    │ 4. 查映射表，定位节点                                  │
│    │ 5. 直接向目标节点发送命令                              │
│    ↓                                                        │
│  目标节点                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**CLUSTER SLOTS 返回示例：**

```bash
CLUSTER SLOTS

# 返回格式：
# 1) 1) start slot
#    2) end slot
#    3) master ip
#    4) master port
#    5) slave ip (可选)
#    6) slave port (可选)

1) 1) 0
   2) 5460
   3) "127.0.0.1"
   4) 7000
   5) "127.0.0.1"
   6) 7003

2) 1) 5461
   2) 10922
   3) "127.0.0.1"
   4) 7001
```

**映射表更新机制：**

```c
// 客户端收到 MOVED 响应时更新映射
if (reply.type == MOVED) {
    // 更新本地缓存
    updateSlotCache(reply.slot, reply.node);
    // 重定向到新节点
    redirectTo(reply.node);
}
```

---

### 3. MOVED 和 ASK 重定向的区别？

**答案：**

MOVED 和 ASK 是两种不同的重定向机制。

**对比分析：**

| 特性 | MOVED | ASK |
|------|-------|-----|
| 触发场景 | 槽位已永久迁移 | 槽位正在迁移中 |
| 客户端处理 | 更新本地槽位映射 | 不更新映射，仅本次重定向 |
| 持续性 | 永久有效 | 临时一次性 |
| 后续请求 | 直接访问新节点 | 继续尝试原节点 |

**MOVED 示例：**

```bash
# 槽位已迁移完成
SET mykey value
# 返回：MOVED 14687 127.0.0.1:7002

# 客户端更新映射：
# slot 14687 → Node 7002
# 后续直接访问 7002
```

**ASK 示例：**

```bash
# 槽位正在迁移
SET mykey value
# 返回：ASK 14687 127.0.0.1:7002

# 客户端处理：
# 1. 发送 ASKING 命令到新节点
# 2. 执行原命令
# 3. 不更新本地映射（因为迁移可能未完成）

ASKING
SET mykey value
```

**迁移过程详解：**

```
┌─────────────────────────────────────────────────────────────┐
│                    槽位迁移过程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段1: 开始迁移                                            │
│  Node A (源)                    Node B (目标)               │
│  slot 1000: IMPORTING          slot 1000: MIGRATING        │
│                                                             │
│  阶段2: 迁移数据                                            │
│  while (有数据未迁移):                                      │
│      MIGRATE key ... (原子迁移单个 key)                     │
│                                                             │
│  阶段3: 迁移期间访问                                        │
│  客户端 → Node A                                            │
│    │                                                        │
│    ├── key 已迁移 → MOVED 到 Node B                         │
│    └── key 未迁移 → ASKING 到 Node B 处理写入               │
│                                                             │
│  阶段4: 迁移完成                                            │
│  CLUSTER SETSLOT 1000 NODE B  (更新槽位归属)                │
│  后续请求 → MOVED 到 Node B                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 4. 集群中怎么执行 Lua 脪本？

**答案：**

Redis Cluster 对 Lua 脚本有限制：**所有 key 必须在同一个槽位**。

**限制原因：**

```lua
-- 错误示例：key 在不同槽位
local v1 = redis.call('GET', KEYS[1])  -- key1 → slot 100
local v2 = redis.call('GET', KEYS[2])  -- key2 → slot 200
-- 报错：CROSSSLOT Keys in request don't hash to the same slot
```

**解决方案：**

```lua
-- 方案1: 使用 Hash Tag
-- 确保相同 hash tag 的 key 落到同一槽位
local v1 = redis.call('GET', 'user:{1001}:name')
local v2 = redis.call('GET', 'user:{1001}:age')
-- 只有 {} 内的内容参与计算槽位
-- {1001} → 相同槽位

-- 方案2: 传入所有相关 key
EVAL script 2 user:{1001}:name user:{1001}:age
```

**Hash Tag 原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Hash Tag 计算                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  key: "user:{1001}:name"                                    │
│        └────┬────┘                                          │
│           hash tag                                          │
│                                                             │
│  计算：CRC16("{1001}") % 16384 = slot                       │
│  而非：CRC16("user:{1001}:name") % 16384                    │
│                                                             │
│  以下 key 都在同一个槽位：                                   │
│  user:{1001}:name  → slot X                                 │
│  user:{1001}:age   → slot X                                 │
│  user:{1001}:city  → slot X                                 │
│  order:{1001}:list → slot X                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：如果必须跨槽位怎么办？**

**追问答案：**

```python
# 使用客户端分步执行
import redis

cluster = redis.RedisCluster(host='localhost', port=7000)

# 分步获取
def get_user_info(user_id):
    name = cluster.get(f'user:{user_id}:name')
    age = cluster.get(f'user:{user_id}:age')
    return {'name': name, 'age': age}

# 或使用 Hash Tag 统一槽位
def get_user_info_v2(user_id):
    # 使用 Lua 脚本
    script = """
        local name = redis.call('GET', KEYS[1])
        local age = redis.call('GET', KEYS[2])
        return {name, age}
    """
    return cluster.eval(script, 2, 
                       f'user:{{{user_id}}}:name',
                       f'user:{{{user_id}}}:age')
```

---

### 5. 集群怎么处理故障转移？

**答案：**

Redis Cluster 采用**自动故障转移**机制，无需哨兵。

**故障检测：**

```
┌─────────────────────────────────────────────────────────────┐
│                    故障检测机制                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  节点间通过 Gossip 协议交换状态                              │
│                                                             │
│  PFAIL（疑似下线）：                                        │
│  - 单个节点认为另一节点下线                                  │
│  - 标记为 PFAIL                                             │
│                                                             │
│  FAIL（确定下线）：                                         │
│  - 大多数主节点认为某节点 PFAIL                              │
│  - 第一个发现的节点广播 FAIL 消息                           │
│  - 触发故障转移                                             │
│                                                             │
│  时间线：                                                   │
│  T0: Node B 无响应                                          │
│  T1 (500ms): Node A 标记 Node B 为 PFAIL                    │
│  T2: Node A 通知其他节点                                    │
│  T3: 多数节点同意 → 标记 FAIL                               │
│  T4: 从节点发起选举                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**故障转移流程：**

```
阶段1: 从节点发现主节点下线
       ↓
阶段2: 从节点发起选举
       ↓
阶段3: 请求其他主节点投票
       ↓
阶段4: 获得多数票后成为新主节点
       ↓
阶段5: 广播新配置信息
```

**选举机制：**

```c
// 从节点发起选举
void clusterHandleSlaveFailover(void) {
    // 1. 检查条件：主节点下线足够久
    if (mstime() - my_master->voted_time < cluster_failover_auth_time)
        return;
    
    // 2. 设置选举纪元
    server.cluster->failover_auth_epoch++;
    
    // 3. 广播投票请求
    clusterRequestFailoverAuth();
}

// 主节点投票
void clusterSendFailoverAuthIfNeeded(clusterNode *node) {
    // 检查条件：
    // 1. 该从节点的主节点确实下线
    // 2. 选举纪元更新
    // 3. 未投过票或投给同一个节点
    
    // 投票
    clusterSendFailoverAuth(node);
}
```

**追问：集群的节点间怎么通信？**

**追问答案：**

Redis Cluster 使用 **Gossip 协议**进行节点间通信：

```
┌─────────────────────────────────────────────────────────────┐
│                    Gossip 协议                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  消息类型：                                                 │
│  - MEET: 加入集群                                           │
│  - PING: 心跳检测                                           │
│  - PONG: 响应消息                                           │
│  - FAIL: 节点下线通知                                       │
│  - PUBLISH: 发布订阅消息                                    │
│                                                             │
│  通信频率：                                                 │
│  - 每秒随机选择 5 个节点发送 PING                           │
│  - 最久未通信的节点优先                                     │
│                                                             │
│  消息内容：                                                 │
│  - 发送者已知的集群状态                                     │
│  - 其他节点的主从关系                                       │
│  - 槽位分配信息                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 6. 一致性哈希虚拟节点的作用？

**答案：**

一致性哈希是分布式系统常用的分区策略，Redis Cluster 使用槽位而非传统一致性哈希。

**传统一致性哈希：**

```
┌─────────────────────────────────────────────────────────────┐
│                  一致性哈希环                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Node A                                   │
│                       ●                                     │
│                     ╱   ╲                                   │
│                   ╱       ╲                                 │
│    Key 1 ●    ╱           ╲    ● Key 2                     │
│              ╱               ╲                             │
│            ╱                   ╲                           │
│   Node C ●                       ● Node B                  │
│            ╲                   ╱                           │
│              ╲               ╱                             │
│                ╲           ╱                               │
│     Key 3 ●     ╲       ╱     ● Key 4                      │
│                    ╲   ╱                                    │
│                      ●                                      │
│                  Node D                                     │
│                                                             │
│  问题：节点少时数据分布不均匀                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**虚拟节点的作用：**

```
┌─────────────────────────────────────────────────────────────┐
│                    虚拟节点                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  每个物理节点映射多个虚拟节点：                              │
│                                                             │
│  Node A → A#1, A#2, A#3, ..., A#100                         │
│  Node B → B#1, B#2, B#3, ..., B#100                         │
│  Node C → C#1, C#2, C#3, ..., C#100                         │
│                                                             │
│  好处：                                                     │
│  1. 数据分布更均匀                                          │
│  2. 节点故障时影响更小                                      │
│  3. 扩缩容时数据迁移更平滑                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Redis Cluster 的方案对比：**

| 特性 | 一致性哈希 | Redis Cluster |
|------|-----------|---------------|
| 分区单位 | 虚拟节点 | 哈希槽 |
| 数量 | 可配置（如 1000/节点） | 固定 16384 |
| 迁移粒度 | key 级别 | 槽位级别 |
| 定位效率 | O(log N) | O(1) 查表 |

---

### 7. 集群的节点间怎么通信？

**答案：**

已在问题 5 的追问中详细解答。核心是 Gossip 协议。

---

### 8. 集群怎么处理批量操作？

**答案：**

```bash
# 问题：不同 key 可能在不同节点
MSET key1 value1 key2 value2
# 报错：CROSSSLOT Keys in request don't hash to the same slot

# 解决方案1: Hash Tag
MSET user:{1001}:name alice user:{1001}:age 25

# 解决方案2: Pipeline 分组
# 客户端按节点分组后分别执行

# 解决方案3: 串行执行
# 简单但性能差
```

**Java 客户端示例：**

```java
JedisCluster cluster = new JedisCluster(nodes);

// 使用 Hash Tag
cluster.mset("user:{1001}:name", "alice", 
             "user:{1001}:age", "25");

// 或使用 Pipeline 分组执行
Map<JedisPool, List<String>> nodeKeyMap = groupKeysByNode(keys);
for (Map.Entry<JedisPool, List<String>> entry : nodeKeyMap.entrySet()) {
    Jedis jedis = entry.getKey().getResource();
    Pipeline pipeline = jedis.pipelined();
    for (String key : entry.getValue()) {
        pipeline.get(key);
    }
    pipeline.sync();
}
```

---

### 9. 集群的客户端连接池怎么配置？

**答案：**

```java
// JedisCluster 连接池配置
GenericObjectPoolConfig<Jedis> poolConfig = new GenericObjectPoolConfig<>();
poolConfig.setMaxTotal(100);        // 最大连接数
poolConfig.setMaxIdle(50);          // 最大空闲连接
poolConfig.setMinIdle(10);          // 最小空闲连接
poolConfig.setMaxWaitMillis(3000);  // 获取连接超时
poolConfig.setTestWhileIdle(true);  // 空闲时测试连接
poolConfig.setTimeBetweenEvictionRunsMillis(30000);

Set<HostAndPort> nodes = new HashSet<>();
nodes.add(new HostAndPort("127.0.0.1", 7000));
nodes.add(new HostAndPort("127.0.0.1", 7001));
nodes.add(new HostAndPort("127.0.0.1", 7002));

JedisCluster cluster = new JedisCluster(nodes, 
    2000,  // 连接超时
    2000,  // 读写超时
    5,     // 重试次数
    "password",
    poolConfig);
```

---

### 10. 集群扩缩容怎么做？

**答案：**

```bash
# 扩容：添加新节点
# 1. 启动新节点
redis-server redis-7003.conf

# 2. 加入集群
redis-cli --cluster add-node 127.0.0.1:7003 127.0.0.1:7000

# 3. 重新分配槽位
redis-cli --cluster reshard 127.0.0.1:7000
# 按提示输入迁移槽位数量和目标节点

# 4. 添加从节点
redis-cli --cluster add-node 127.0.0.1:7004 127.0.0.1:7000 \
    --cluster-slave --cluster-master-id <master_id>

# 缩容：移除节点
# 1. 迁移槽位
redis-cli --cluster reshard 127.0.0.1:7000
# 将目标节点的槽位迁移到其他节点

# 2. 移除节点
redis-cli --cluster del-node 127.0.0.1:7000 <node_id>
```

---


## 四、分布式锁深度篇

### 1. Redisson 的加锁流程？

**答案：**

Redisson 是 Redis 分布式锁的主流实现，提供了完善的锁机制。

**加锁 Lua 脚本：**

```lua
-- Redisson 加锁核心脚本
if (redis.call('exists', KEYS[1]) == 0) then
    -- 锁不存在，创建锁
    redis.call('hset', KEYS[1], ARGV[2], 1);
    redis.call('pexpire', KEYS[1], ARGV[1]);
    return nil;
end;
if (redis.call('hexists', KEYS[1], ARGV[2]) == 1) then
    -- 当前线程持有锁，重入计数+1
    redis.call('hincrby', KEYS[1], ARGV[2], 1);
    redis.call('pexpire', KEYS[1], ARGV[1]);
    return nil;
end;
-- 锁被其他线程持有，返回剩余 TTL
return redis.call('pttl', KEYS[1]);
```

**加锁流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Redisson 加锁流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  客户端                     Redis                          │
│    │                          │                            │
│    │── EVALSHA(script) ──────→│                            │
│    │    KEYS[1] = "myLock"    │                            │
│    │    ARGV[1] = 30000 (TTL) │                            │
│    │    ARGV[2] = "threadId"  │                            │
│    │                          │                            │
│    │←───── nil ───────────────│  加锁成功                  │
│    │                          │                            │
│    │── 启动看门狗线程 ────────→│                            │
│    │    (每 10 秒续期)        │                            │
│    │                          │                            │
│    │←───── TTL ───────────────│  加锁失败，返回剩余时间    │
│    │                          │                            │
│    │── 订阅锁释放消息 ────────→│                            │
│    │    (阻塞等待)            │                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Java 代码示例：**

```java
RedissonClient redisson = Redisson.create(config);
RLock lock = redisson.getLock("myLock");

try {
    // 尝试获取锁
    boolean acquired = lock.tryLock(10, 30, TimeUnit.SECONDS);
    if (acquired) {
        // 执行业务逻辑
        doBusiness();
    }
} finally {
    // 释放锁
    if (lock.isHeldByCurrentThread()) {
        lock.unlock();
    }
}
```

**追问：为什么用 Lua 脚本？**

**追问答案：**

保证原子性：
1. **检查 + 加锁** 必须原子
2. **重入判断 + 计数增加** 必须原子
3. 避免并发问题

---

### 2. 看门狗机制怎么实现的？

**答案：**

看门狗（Watchdog）是 Redisson 的自动续期机制，防止业务未执行完锁就过期。

**实现原理：**

```java
// Redisson 看门狗实现
private void scheduleExpirationRenewal(long threadId) {
    // 创建定时任务
    Timeout task = commandExecutor.getConnectionManager()
        .newTimeout(new TimerTask() {
            @Override
            public void run(Timeout timeout) throws Exception {
                // 续期逻辑
                RFuture<Boolean> future = renewExpirationAsync(threadId);
                future.onComplete((res, e) -> {
                    if (res) {
                        // 续期成功，重新调度
                        scheduleExpirationRenewal(threadId);
                    }
                });
            }
        }, internalLockLeaseTime / 3, TimeUnit.MILLISECONDS);
        // 默认 30 秒 / 3 = 每 10 秒续期一次
}
```

**续期 Lua 脚本：**

```lua
-- 看门狗续期脚本
if (redis.call('hexists', KEYS[1], ARGV[2]) == 1) then
    redis.call('pexpire', KEYS[1], ARGV[1]);
    return 1;
end;
return 0;
```

**工作流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    看门狗时间线                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T0         T10        T20        T30        T40            │
│  │          │          │          │          │              │
│  ↓          ↓          ↓          ↓          ↓              │
│  加锁       续期        续期        续期       释放锁        │
│  TTL=30s    TTL=30s    TTL=30s    TTL=30s                  │
│                                                             │
│  业务执行时间 > 30s 时：                                     │
│  - 看门狗每 10 秒自动续期                                   │
│  - 业务完成后释放锁，停止续期                               │
│                                                             │
│  业务执行时间 < 30s 时：                                     │
│  - 业务完成，主动释放锁                                     │
│  - 看门狗停止                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：什么时候关闭看门狗？**

**追问答案：**

```java
// 释放锁时关闭看门狗
protected RFuture<Boolean> unlockInnerAsync(long threadId) {
    // Lua 脚本删除锁
    // 如果删除成功，取消续期任务
    return commandExecutor.evalWriteAsync(getName(), 
        LongCodec.INSTANCE, RedisCommands.EVAL_BOOLEAN,
        "if (redis.call('hexists', KEYS[1], ARGV[3]) == 0) then " +
        "    return nil; " +
        "end; " +
        "local counter = redis.call('hincrby', KEYS[1], ARGV[3], -1); " +
        "if (counter > 0) then " +
        "    return 0; " +
        "else " +
        "    redis.call('del', KEYS[1]); " +
        "    return 1; " +
        "end;",
        Arrays.asList(getName()), 
        internalLockLeaseTime, getLockName(threadId));
}
```

---

### 3. Redisson 为什么用 Hash 结构存锁？

**答案：**

使用 Hash 结构是为了支持**可重入锁**。

**Hash 结构存储：**

```
┌─────────────────────────────────────────────────────────────┐
│                    锁的 Hash 结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Key: "myLock"                                              │
│  Type: Hash                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Field                  │ Value                      │   │
│  ├────────────────────────┼────────────────────────────┤   │
│  │ "threadId-1"           │ 2  (重入次数)              │   │
│  │ "threadId-2"           │ 1  (另一线程等待)          │   │
│  └────────────────────────┴────────────────────────────┘   │
│                                                             │
│  优势：                                                     │
│  1. 支持重入计数                                            │
│  2. 支持同一线程多次获取                                    │
│  3. 解锁时计数减 1，到 0 才删除                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**对比简单 String 实现：**

```bash
# 简单实现（不支持重入）
SET lock:mylock threadId NX EX 30

# 问题：
# 1. 无法实现重入
# 2. 无法区分不同线程
# 3. 无法记录获取次数
```

---

### 4. Redlock 算法的争议点？

**答案：**

Redlock 是 Redis 作者提出的分布式锁算法，使用多个独立 Redis 实例。

**算法流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Redlock 算法                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  假设有 5 个独立的 Redis 实例                               │
│                                                             │
│  1. 获取当前时间戳 T1                                       │
│                                                             │
│  2. 依次向 5 个实例请求加锁                                 │
│     SET lock resource_name nonce NX PX TTL                  │
│                                                             │
│  3. 计算获取锁耗时                                          │
│     elapsed = T2 - T1                                       │
│                                                             │
│  4. 判断加锁成功                                            │
│     - 获取了大多数锁（>= 3 个）                             │
│     - 锁有效时间 > elapsed                                  │
│                                                             │
│  5. 加锁失败则释放所有实例的锁                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**争议点：**

| 争议点 | 作者观点 | 反对观点 |
|--------|----------|----------|
| 时钟依赖 | 使用本地时钟 | 网络延迟影响判断 |
| 安全性 | 分布式更安全 | GC 暂停可能导致问题 |
| 复杂性 | 必要的复杂 | ZooKeeper 更可靠 |
| 边界情况 | 罕见可忽略 | 生产中可能遇到 |

**Martin Kleppmann 的反对意见：**

```
问题场景：
1. 客户端 A 获取了 3 个实例的锁
2. 客户端 A 发生长时间 GC 暂停
3. 锁过期，客户端 B 获取了 3 个实例的锁
4. 客户端 A 恢复，认为自己还持有锁
5. 两个客户端同时持有"锁"
```

**追问：生产中如何选择？**

**追问答案：**

```java
// 方案1: 单 Redis + Redisson（大多数场景够用）
RLock lock = redisson.getLock("myLock");

// 方案2: Redlock（对安全性要求极高）
RLock lock1 = redisson1.getLock("myLock");
RLock lock2 = redisson2.getLock("myLock");
RLock lock3 = redisson3.getLock("myLock");

RedissonRedLock redLock = new RedissonRedLock(lock1, lock2, lock3);
redLock.lock();

// 方案3: ZooKeeper（强一致性需求）
InterProcessMutex lock = new InterProcessMutex(client, "/locks/myLock");
```

---

### 5. 联锁（MultiLock）怎么用？

**答案：**

联锁用于同时锁定多个资源，所有锁都获取成功才算成功。

**使用场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                    联锁场景                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景：转账操作                                             │
│  - 需要同时锁定转出账户和转入账户                           │
│  - 避免部分锁定导致死锁                                     │
│                                                             │
│  账户 A ──→ 账户 B                                          │
│     ↓           ↓                                           │
│  lock_A     lock_B                                          │
│     └─────┬─────┘                                           │
│           ↓                                                 │
│        MultiLock                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
// 创建多个锁
RLock lockA = redisson.getLock("account:A");
RLock lockB = redisson.getLock("account:B");

// 创建联锁
RedissonMultiLock multiLock = new RedissonMultiLock(lockA, lockB);

try {
    // 尝试获取所有锁
    boolean acquired = multiLock.tryLock(10, 30, TimeUnit.SECONDS);
    if (acquired) {
        // 执行转账逻辑
        transfer(accountA, accountB, amount);
    }
} finally {
    multiLock.unlock();
}
```

**加锁顺序：**

```java
// Redisson 会按顺序获取锁
// 内部实现确保所有锁获取成功或全部回滚

// 加锁顺序问题：
// 如果 A→B 和 B→A 并发，可能死锁
// 解决：统一加锁顺序（按 key 排序）
List<String> keys = Arrays.asList("account:A", "account:B");
Collections.sort(keys);  // 确保顺序一致
```

---

### 6. 读写锁怎么实现？

**答案：**

Redisson 提供了分布式读写锁（ReadWriteLock）。

**实现原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    读写锁状态                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Key: "myLock"                                              │
│                                                             │
│  读锁模式：                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Field              │ Value                          │   │
│  ├────────────────────┼────────────────────────────────┤   │
│  │ mode               │ "read"                         │   │
│  │ thread-1           │ 1 (读锁计数)                   │   │
│  │ thread-2           │ 1                              │   │
│  │ readers            │ 2 (总读锁数)                   │   │
│  └────────────────────┴────────────────────────────────┘   │
│                                                             │
│  写锁模式：                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Field              │ Value                          │   │
│  ├────────────────────┼────────────────────────────────┤   │
│  │ mode               │ "write"                        │   │
│  │ thread-1           │ 1 (写锁计数)                   │   │
│  └────────────────────┴────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码示例：**

```java
RReadWriteLock rwLock = redisson.getReadWriteLock("myLock");

// 读锁（共享锁）
RLock readLock = rwLock.readLock();
readLock.lock();
try {
    // 读操作，允许多个线程同时读
    String value = readFromCache();
} finally {
    readLock.unlock();
}

// 写锁（排他锁）
RLock writeLock = rwLock.writeLock();
writeLock.lock();
try {
    // 写操作，独占访问
    writeToCache(value);
} finally {
    writeLock.unlock();
}
```

**锁互斥规则：**

| 当前状态 | 请求读锁 | 请求写锁 |
|---------|---------|---------|
| 无锁 | ✅ 允许 | ✅ 允许 |
| 读锁 | ✅ 允许 | ❌ 阻塞 |
| 写锁 | ❌ 阻塞 | ❌ 阻塞 |

---

### 7. 分布式锁的续期问题？

**答案：**

锁续期是分布式锁的关键问题，处理不当会导致业务中断或锁失效。

**问题场景：**

```
┌─────────────────────────────────────────────────────────────┐
│                    锁过期问题                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景1: 业务执行时间 > 锁过期时间                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ T0          T30          T35                          │  │
│  │ │           │            │                            │  │
│  │ ↓           ↓            ↓                            │  │
│  │ 加锁TTL=30s  锁过期      业务完成                      │  │
│  │             (其他线程可能获取锁)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  场景2: 网络延迟导致续期失败                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 客户端 ──────→ Redis                                   │  │
│  │   续期请求       ↑                                     │  │
│  │                  │ 网络超时                            │  │
│  │   续期失败 ←─────┘                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**解决方案：**

```java
// 方案1: 使用 Redisson 看门狗（自动续期）
RLock lock = redisson.getLock("myLock");
lock.lock();  // 默认开启看门狗
// 每 10 秒自动续期到 30 秒

// 方案2: 手动设置足够长的过期时间
lock.lock(5, TimeUnit.MINUTES);  // 不开启看门狗

// 方案3: 手动续期
RLock lock = redisson.getLock("myLock");
if (lock.tryLock()) {
    try {
        // 长时间业务
        doLongTask();
    } finally {
        lock.unlock();
    }
}

// 方案4: 结合业务超时
try {
    if (lock.tryLock(5, 300, TimeUnit.SECONDS)) {
        Future<?> future = executor.submit(() -> doTask());
        try {
            future.get(60, TimeUnit.SECONDS);  // 业务超时控制
        } catch (TimeoutException e) {
            future.cancel(true);
        }
    }
} finally {
    lock.unlock();
}
```

**追问：如何避免锁误删？**

**追问答案：**

```lua
-- 释放锁时检查是否是自己的锁
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end

-- 使用 Lua 脚本保证原子性
-- 避免检查和删除之间的时间窗口
```

---

### 8. 分布式锁的公平性怎么保证？

**答案：**

Redisson 提供了公平锁实现，按请求顺序获取锁。

**公平锁实现：**

```java
// 创建公平锁
RLock fairLock = redisson.getFairLock("myFairLock");

// 使用方式与普通锁相同
fairLock.lock();
try {
    doBusiness();
} finally {
    fairLock.unlock();
}
```

**实现原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    公平锁队列                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Redis 数据结构：                                           │
│  1. 锁持有者：String                                        │
│     "myLock:lock" → "thread-1"                              │
│                                                             │
│  2. 等待队列：List                                          │
│     "myLock:queue" → [thread-2, thread-3, thread-4]         │
│                                                             │
│  3. 等待超时：Sorted Set（按超时时间排序）                   │
│     "myLock:timeout" → {thread-2: 1000, thread-3: 2000}     │
│                                                             │
│  流程：                                                     │
│  1. 尝试获取锁                                              │
│  2. 失败则加入队列尾部                                      │
│  3. 订阅锁释放消息                                          │
│  4. 收到消息后队列头部尝试获取                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**公平锁 vs 非公平锁：**

| 特性 | 公平锁 | 非公平锁 |
|------|--------|----------|
| 获取顺序 | 先到先得 | 随机竞争 |
| 吞吐量 | 较低 | 较高 |
| 饥饿问题 | 不会出现 | 可能出现 |
| 实现复杂度 | 高 | 低 |

---


## 五、大厂实战篇

### 1. 热 Key 怎么发现？怎么解决？

**答案：**

热 Key 指访问频率极高的 key，可能导致单个节点压力过大。

**发现问题：**

```bash
# 方法1: MONITOR 命令（生产慎用）
redis-cli MONITOR | grep -o "GET\|SET\|HGET" | sort | uniq -c | sort -nr

# 方法2: 使用 Redis 4.0+ 的 MEMORY DOCTOR
MEMORY DOCTOR

# 方法3: 使用 hotkey 分析工具
redis-cli --hotkeys

# 方法4: 客户端统计
# 在应用层统计 key 访问频率
```

**解决方案：**

```
┌─────────────────────────────────────────────────────────────┐
│                    热 Key 解决方案                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方案1: 本地缓存                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    应用层                           │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           本地缓存（如 Guava Cache）         │   │   │
│  │  │           热数据缓存到 JVM 内存              │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                         ↓ 未命中                    │   │
│  │                    Redis 集群                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案2: 热 Key 备份（多 Key）                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  hotkey_1, hotkey_2, hotkey_3, hotkey_4             │   │
│  │       ↓        ↓        ↓        ↓                  │   │
│  │    Node A   Node B   Node C   Node D                │   │
│  │                                                     │   │
│  │  读写时随机选择一个备份 key                          │   │
│  │  写入时同步更新所有备份                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  方案3: 限流                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  对热 Key 的访问进行限流                             │   │
│  │  保护 Redis 不被压垮                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码示例：**

```java
// 方案1: 本地缓存
public class HotKeyCache {
    private Cache<String, String> localCache = Caffeine.newBuilder()
        .maximumSize(10000)
        .expireAfterWrite(5, TimeUnit.SECONDS)
        .build();
    
    public String get(String key) {
        // 先查本地缓存
        String value = localCache.getIfPresent(key);
        if (value != null) {
            return value;
        }
        // 查 Redis
        value = redis.get(key);
        if (value != null) {
            localCache.put(key, value);
        }
        return value;
    }
}

// 方案2: 热 Key 备份
public String getHotKey(String key) {
    // 随机选择备份
    int suffix = ThreadLocalRandom.current().nextInt(4);
    String backupKey = key + "_" + suffix;
    return redis.get(backupKey);
}

public void setHotKey(String key, String value) {
    // 写入所有备份
    for (int i = 0; i < 4; i++) {
        redis.set(key + "_" + i, value);
    }
}
```

---

### 2. 大 Key 怎么删除？为什么不能直接 DEL？

**答案：**

大 Key 直接删除会导致 Redis 阻塞，影响其他请求。

**为什么不能直接 DEL？**

```
┌─────────────────────────────────────────────────────────────┐
│                    DEL 阻塞原理                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DEL 执行过程：                                             │
│  1. 释放内存（main 函数中同步执行）                         │
│  2. 对于 Hash/Set/ZSet，需要遍历所有元素释放                │
│  3. 期间阻塞所有其他命令                                    │
│                                                             │
│  阻塞时间估算：                                             │
│  - 100 万元素的 Hash：约 100ms                              │
│  - 1000 万元素的 Set：约 1s                                 │
│  - 大 String（100MB）：约 10-100ms                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**删除方案：**

```bash
# 方案1: 使用 UNLINK（Redis 4.0+）
# 异步删除，不阻塞主线程
UNLINK bigkey

# 方案2: Hash 类型分批删除
while true; do
    fields=$(redis-cli HSCAN bighash 0 COUNT 100 | head -n -1)
    if [ -z "$fields" ]; then
        redis-cli DEL bighash
        break
    fi
    redis-cli HDEL bighash $fields
done

# 方案3: Set 类型分批删除
while true; do
    members=$(redis-cli SSCAN bigset 0 COUNT 100)
    # ...
done

# 方案4: 使用 lazy-free 配置
# 在 redis.conf 中配置
lazyfree-lazy-server-del yes
lazyfree-lazy-user-del yes
```

**Python 删除脚本：**

```python
import redis

def delete_large_hash(redis_client, key, batch_size=100):
    """分批删除大 Hash"""
    cursor = '0'
    while cursor != 0:
        cursor, fields = redis_client.hscan(key, cursor=cursor, count=batch_size)
        if fields:
            redis_client.hdel(key, *fields.keys())
    redis_client.delete(key)

def delete_large_zset(redis_client, key, batch_size=100):
    """分批删除大 ZSet"""
    while redis_client.zcard(key) > 0:
        # 删除分数最小的 batch_size 个元素
        members = redis_client.zrange(key, 0, batch_size - 1)
        if members:
            redis_client.zrem(key, *members)

def delete_large_set(redis_client, key, batch_size=100):
    """分批删除大 Set"""
    cursor = '0'
    while cursor != 0:
        cursor, members = redis_client.sscan(key, cursor=cursor, count=batch_size)
        if members:
            redis_client.srem(key, *members)
```

---

### 3. Redis 怎么做限流？

**答案：**

Redis 可以实现多种限流算法。

**方案1: 简单计数器**

```java
public boolean rateLimit(String key, int limit, int window) {
    long count = redis.incr(key);
    if (count == 1) {
        redis.expire(key, window, TimeUnit.SECONDS);
    }
    return count <= limit;
}
```

**方案2: 滑动窗口**

```java
public boolean slidingWindow(String key, int limit, int window) {
    long now = System.currentTimeMillis();
    String member = String.valueOf(now);
    
    // 移除窗口外的记录
    redis.zremrangeByScore(key, 0, now - window * 1000);
    
    // 添加当前请求
    redis.zadd(key, now, member);
    
    // 统计窗口内请求数
    long count = redis.zcard(key);
    
    // 设置过期时间
    redis.expire(key, window, TimeUnit.SECONDS);
    
    return count <= limit;
}
```

**方案3: 令牌桶**

```java
public boolean tokenBucket(String key, int capacity, int rate) {
    // Lua 脚本实现
    String script = """
        local tokens = redis.call('get', KEYS[1])
        local last_time = redis.call('get', KEYS[2])
        local now = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local rate = tonumber(ARGV[3])
        
        if tokens == false then
            tokens = capacity
            last_time = now
        end
        
        local elapsed = now - tonumber(last_time)
        local new_tokens = math.min(capacity, tonumber(tokens) + elapsed * rate)
        
        if new_tokens >= 1 then
            redis.call('set', KEYS[1], new_tokens - 1)
            redis.call('set', KEYS[2], now)
            return 1
        else
            return 0
        end
        """;
    
    return redis.eval(script, Arrays.asList(key + ":tokens", key + ":time"),
        System.currentTimeMillis(), capacity, rate);
}
```

**方案4: 漏桶**

```java
public boolean leakyBucket(String key, int capacity, int rate) {
    String script = """
        local water = tonumber(redis.call('get', KEYS[1]) or 0)
        local last_time = tonumber(redis.call('get', KEYS[2]) or 0)
        local now = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local rate = tonumber(ARGV[3])
        
        -- 计算漏出的水量
        local elapsed = now - last_time
        local leaked = elapsed * rate
        water = math.max(0, water - leaked)
        
        if water + 1 <= capacity then
            redis.call('set', KEYS[1], water + 1)
            redis.call('set', KEYS[2], now)
            return 1
        else
            return 0
        end
        """;
    
    return redis.eval(script, Arrays.asList(key + ":water", key + ":time"),
        System.currentTimeMillis(), capacity, rate);
}
```

---

### 4. Redis 怎么实现延迟队列？

**答案：**

使用 ZSet 的 Score 存储执行时间，实现延迟队列。

**实现原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                    延迟队列原理                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ZSet 结构：                                                │
│  Key: "delay_queue"                                         │
│  Member: 任务 ID                                            │
│  Score: 执行时间戳                                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Member        │ Score (执行时间)                     │   │
│  ├───────────────┼────────────────────────────────────┤   │
│  │ task_001      │ 1700000001 (T1)                    │   │
│  │ task_002      │ 1700000005 (T2)                    │   │
│  │ task_003      │ 1700000010 (T3)                    │   │
│  └───────────────┴────────────────────────────────────┘   │
│                                                             │
│  消费流程：                                                 │
│  1. ZRANGEBYSCORE 查询 score <= 当前时间的任务              │
│  2. ZREM 删除已消费的任务（保证原子性）                      │
│  3. 执行任务                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**代码实现：**

```java
public class DelayQueue {
    private RedisClient redis;
    private String queueKey = "delay_queue";
    
    // 生产者：添加延迟任务
    public void produce(String taskId, long delaySeconds) {
        long executeTime = System.currentTimeMillis() + delaySeconds * 1000;
        redis.zadd(queueKey, executeTime, taskId);
    }
    
    // 消费者：获取到期任务
    public String consume() {
        String script = """
            local tasks = redis.call('ZRANGEBYSCORE', KEYS[1], 0, ARGV[1], 'LIMIT', 0, 1)
            if #tasks > 0 then
                redis.call('ZREM', KEYS[1], tasks[1])
                return tasks[1]
            end
            return nil
            """;
        
        return redis.eval(script, Collections.singletonList(queueKey),
            System.currentTimeMillis());
    }
    
    // 消费者线程
    public void startConsumer() {
        new Thread(() -> {
            while (true) {
                String task = consume();
                if (task != null) {
                    processTask(task);
                } else {
                    Thread.sleep(1000);  // 无任务时休眠
                }
            }
        }).start();
    }
}
```

**使用 Redisson 延迟队列：**

```java
RedissonClient redisson = Redisson.create(config);

// 创建延迟队列
RBlockingQueue<String> queue = redisson.getBlockingQueue("delay_queue");
RDelayedQueue<String> delayedQueue = redisson.getDelayedQueue(queue);

// 添加延迟任务
delayedQueue.offer("task_001", 10, TimeUnit.SECONDS);

// 消费任务
while (true) {
    String task = queue.take();  // 阻塞获取
    processTask(task);
}
```

---

### 5. Redis 怎么实现排行榜？

**答案：**

使用 ZSet 的天然排序特性实现排行榜。

**基本操作：**

```java
// 添加/更新分数
redis.zadd("leaderboard", 100, "player1");
redis.zadd("leaderboard", 200, "player2");
redis.zadd("leaderboard", 150, "player3");

// 获取排行榜（降序）
redis.zrevrange("leaderboard", 0, 9, "WITHSCORES");

// 获取玩家排名（从 0 开始）
redis.zrevrank("leaderboard", "player1");

// 获取玩家分数
redis.zscore("leaderboard", "player1");

// 增加分数
redis.zincrby("leaderboard", 50, "player1");

// 获取分数区间内的玩家
redis.zrevrangebyscore("leaderboard", 200, 100, "WITHSCORES");
```

**完整排行榜实现：**

```java
public class Leaderboard {
    private RedisClient redis;
    private String key;
    
    public Leaderboard(String key) {
        this.key = key;
    }
    
    // 更新玩家分数
    public void updateScore(String playerId, double score) {
        redis.zadd(key, score, playerId);
    }
    
    // 增加分数
    public void addScore(String playerId, double delta) {
        redis.zincrby(key, delta, playerId);
    }
    
    // 获取排行榜 Top N
    public List<RankInfo> getTopN(int n) {
        Set<Tuple> tuples = redis.zrevrangeWithScores(key, 0, n - 1);
        List<RankInfo> result = new ArrayList<>();
        int rank = 1;
        for (Tuple tuple : tuples) {
            result.add(new RankInfo(
                rank++,
                tuple.getElement(),
                tuple.getScore()
            ));
        }
        return result;
    }
    
    // 获取玩家排名
    public RankInfo getPlayerRank(String playerId) {
        Long rank = redis.zrevrank(key, playerId);
        Double score = redis.zscore(key, playerId);
        if (rank == null || score == null) {
            return null;
        }
        return new RankInfo(rank + 1, playerId, score);
    }
    
    // 获取玩家周围排名（前后各 N 名）
    public List<RankInfo> getAround(String playerId, int n) {
        Long rank = redis.zrevrank(key, playerId);
        if (rank == null) return Collections.emptyList();
        
        long start = Math.max(0, rank - n);
        long end = rank + n;
        
        Set<Tuple> tuples = redis.zrevrangeWithScores(key, start, end);
        // ...
    }
}
```

---

### 6. Redis 怎么实现分布式 Session？

**答案：**

使用 Redis 存储用户 Session，实现分布式环境下的会话共享。

**实现方案：**

```java
@Configuration
@EnableRedisHttpSession
public class SessionConfig {
    @Bean
    public LettuceConnectionFactory connectionFactory() {
        return new LettuceConnectionFactory("localhost", 6379);
    }
}

// Spring Session 自动配置后
// Session 存储在 Redis 中

@RestController
public class UserController {
    @GetMapping("/login")
    public String login(HttpSession session, @RequestParam String userId) {
        session.setAttribute("userId", userId);
        return "登录成功";
    }
    
    @GetMapping("/user")
    public String getUser(HttpSession session) {
        String userId = (String) session.getAttribute("userId");
        return "当前用户: " + userId;
    }
}
```

**Redis 存储结构：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Session 存储结构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Key 格式：spring:session:sessions:{sessionId}              │
│  Type: Hash                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Field                    │ Value                    │   │
│  ├──────────────────────────┼──────────────────────────┤   │
│  │ creationTime             │ 1700000000000            │   │
│  │ lastAccessedTime         │ 1700000001000            │   │
│  │ maxInactiveInterval      │ 1800                     │   │
│  │ sessionAttr:userId       │ user123                  │   │
│  │ sessionAttr:userName     │ alice                    │   │
│  └──────────────────────────┴──────────────────────────┘   │
│                                                             │
│  过期索引：spring:session:sessions:expires:{sessionId}      │
│  Type: String                                               │
│  TTL: session 过期时间                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**手动实现：**

```java
public class RedisSession {
    private RedisClient redis;
    private int expireSeconds = 1800;  // 30 分钟
    
    // 创建 Session
    public String createSession(String userId) {
        String sessionId = UUID.randomUUID().toString();
        Map<String, String> session = new HashMap<>();
        session.put("userId", userId);
        session.put("createTime", String.valueOf(System.currentTimeMillis()));
        redis.hset("session:" + sessionId, session);
        redis.expire("session:" + sessionId, expireSeconds, TimeUnit.SECONDS);
        return sessionId;
    }
    
    // 获取 Session
    public Map<String, String> getSession(String sessionId) {
        String key = "session:" + sessionId;
        Map<String, String> session = redis.hgetAll(key);
        if (session.isEmpty()) {
            return null;
        }
        // 刷新过期时间
        redis.expire(key, expireSeconds, TimeUnit.SECONDS);
        return session;
    }
    
    // 删除 Session
    public void deleteSession(String sessionId) {
        redis.del("session:" + sessionId);
    }
}
```

---

### 7. 项目中遇到过的 Redis 问题？

**答案：**

**问题1: 缓存穿透**

```
场景：恶意请求大量不存在的 key
问题：请求直达数据库
解决：
- 布隆过滤器过滤不存在的 key
- 缓存空值，设置较短过期时间
```

```java
// 布隆过滤器方案
public Object getWithBloom(String key) {
    // 先检查布隆过滤器
    if (!bloomFilter.mightContain(key)) {
        return null;  // 一定不存在
    }
    // 查缓存
    Object value = redis.get(key);
    if (value != null) {
        return value;
    }
    // 查数据库
    value = db.query(key);
    if (value != null) {
        redis.set(key, value);
    }
    return value;
}

// 空值缓存方案
public Object getWithNullCache(String key) {
    Object value = redis.get(key);
    if ("NULL".equals(value)) {
        return null;
    }
    if (value != null) {
        return value;
    }
    value = db.query(key);
    if (value != null) {
        redis.set(key, value);
    } else {
        redis.setex(key, "NULL", 60);  // 缓存空值 60 秒
    }
    return value;
}
```

**问题2: 缓存雪崩**

```
场景：大量缓存同时过期
问题：瞬间大量请求到数据库
解决：
- 过期时间加随机值
- 多级缓存
- 熔断降级
```

```java
// 随机过期时间
int expire = 3600 + ThreadLocalRandom.current().nextInt(600);
redis.setex(key, value, expire);

// 多级缓存
public Object getWithMultiCache(String key) {
    // L1: 本地缓存
    Object value = localCache.get(key);
    if (value != null) return value;
    
    // L2: Redis 缓存
    value = redis.get(key);
    if (value != null) {
        localCache.put(key, value);
        return value;
    }
    
    // L3: 数据库
    value = db.query(key);
    if (value != null) {
        redis.set(key, value);
        localCache.put(key, value);
    }
    return value;
}
```

**问题3: 缓存与数据库不一致**

```
场景：更新数据库后缓存未更新
问题：读到旧数据
解决：
- 延迟双删
- 订阅 Binlog 更新缓存
```

```java
// 延迟双删
public void update(String key, Object value) {
    // 1. 删除缓存
    redis.del(key);
    // 2. 更新数据库
    db.update(key, value);
    // 3. 延迟删除缓存
    executor.schedule(() -> redis.del(key), 500, TimeUnit.MILLISECONDS);
}
```

---

### 8. Redis 的性能优化有哪些？

**答案：**

```bash
# 1. 网络优化
# 使用 Pipeline 批量执行命令
redis-cli --pipe < commands.txt

# 2. 内存优化
# 选择合适的数据结构
# 使用 Hash 替代多个 String

# 3. 持久化优化
# 根据场景选择合适的持久化策略
appendfsync everysec

# 4. 连接优化
# 使用连接池
# 控制连接数

# 5. 命令优化
# 避免使用 KEYS、FLUSHALL 等阻塞命令
# 使用 SCAN 替代 KEYS

# 6. 架构优化
# 主从读写分离
# 集群分片
```

**性能监控命令：**

```bash
# 查看慢查询日志
SLOWLOG GET 10

# 查看内存使用
INFO memory

# 查看客户端连接
CLIENT LIST

# 查看命令统计
INFO commandstats

# 实时监控
INFO stats
```

---

## 参考资料

1. [Redis 官方文档](https://redis.io/documentation)
2. [Redisson 官方文档](https://github.com/redisson/redisson/wiki)
3. 《Redis 设计与实现》- 黄健宏
4. 《Redis 开发与运维》- 付磊 张益军
