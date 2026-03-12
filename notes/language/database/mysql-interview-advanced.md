# MySQL 进阶面试题

本文档是 MySQL 面试题的进阶篇，聚焦于 MVCC、锁机制、索引原理、性能优化和架构设计等深入话题。

## MVCC 深度篇

### 1. Read View 的四个核心字段是什么？各有什么用？

**答案：**

Read View 是 MVCC 中实现事务可见性判断的核心数据结构，包含四个关键字段：

| 字段名 | 含义 | 作用 |
|--------|------|------|
| `m_ids` | 生成 Read View 时活跃的事务 ID 列表 | 记录所有未提交的事务 ID |
| `min_trx_id` | `m_ids` 中最小的事务 ID | 快速判断版本是否在活跃事务范围内 |
| `max_trx_id` | 生成 Read View 时系统应分配的下一个事务 ID | 判断版本是否是未来事务产生的 |
| `creator_trx_id` | 创建该 Read View 的事务 ID | 判断是否是自己修改的版本 |

```sql
-- Read View 结构伪代码
struct ReadView {
    List<trx_id> m_ids;        // 活跃事务列表
    trx_id min_trx_id;         // 最小活跃事务ID
    trx_id max_trx_id;         // 下一个待分配的事务ID
    trx_id creator_trx_id;     // 创建者事务ID
};
```

**工作原理图解：**

```
事务 ID 时间线：
┌────────────────────────────────────────────────────────────┐
│  100   101   102   103   104   105   106   107   108      │
│   │     │     │     │     │     │     │     │     │       │
│   │     │     │     │     │     │     │     │     │       │
│  已提交  │    活跃   │    已提交  │    未来事务             │
│         │           │           │                          │
│      m_ids = [102, 104]                                    │
│      min_trx_id = 102                                      │
│      max_trx_id = 108                                      │
└────────────────────────────────────────────────────────────┘
```

**追问：Read View 是什么时候创建的？**

**追问答案：**
- RC 隔离级别：每次 SELECT 语句执行时都创建新的 Read View
- RR 隔离级别：事务中第一次 SELECT 语句执行时创建，后续复用

---

### 2. 如何判断一个版本对当前事务可见？

**答案：**

通过 Read View 和数据版本的 `trx_id`（事务ID）进行可见性判断，规则如下：

```python
def is_visible(version_trx_id, read_view):
    # 1. 如果是自己修改的，当然可见
    if version_trx_id == read_view.creator_trx_id:
        return True
    
    # 2. 如果版本在 min_trx_id 之前，说明生成该版本的事务已提交，可见
    if version_trx_id < read_view.min_trx_id:
        return True
    
    # 3. 如果版本 >= max_trx_id，说明是未来事务产生的，不可见
    if version_trx_id >= read_view.max_trx_id:
        return False
    
    # 4. 如果版本在活跃事务列表中，说明未提交，不可见
    if version_trx_id in read_view.m_ids:
        return False
    
    # 5. 不在活跃列表且在范围内，说明已提交，可见
    return True
```

**完整判断流程图：**

```
                    ┌─────────────────────┐
                    │ trx_id == creator?  │
                    └──────────┬──────────┘
                               │
              ┌────────YES─────┴─────NO────────┐
              ▼                               ▼
         ┌────────┐                 ┌─────────────────────┐
         │ 可见 ✓ │                 │ trx_id < min_trx_id?│
         └────────┘                 └──────────┬──────────┘
                                               │
                          ┌────────YES─────────┴──────NO─────────┐
                          ▼                                     ▼
                     ┌────────┐                    ┌─────────────────────┐
                     │ 可见 ✓ │                    │ trx_id >= max_trx_id?│
                     └────────┘                    └──────────┬──────────┘
                                                             │
                                    ┌────────YES─────────────┴──────NO────────┐
                                    ▼                                         ▼
                               ┌──────────┐                        ┌───────────────────┐
                               │ 不可见 ✗ │                        │ trx_id in m_ids?  │
                               └──────────┘                        └─────────┬─────────┘
                                                                            │
                                              ┌────────YES──────────────────┴──────NO────────┐
                                              ▼                                             ▼
                                         ┌──────────┐                                  ┌────────┐
                                         │ 不可见 ✗ │                                  │ 可见 ✓ │
                                         └──────────┘                                  └────────┘
```

**代码示例：**

```sql
-- 假设当前事务 ID = 105，Read View 如下：
-- m_ids = [102, 104], min_trx_id = 102, max_trx_id = 108

-- 某行数据的版本链：
-- 版本1: trx_id = 100, name = 'Alice'
-- 版本2: trx_id = 102, name = 'Bob'   (未提交)
-- 版本3: trx_id = 103, name = 'Carol' (已提交)

-- 判断结果：
-- 版本1: 100 < 102 (min) → 可见
-- 版本2: 102 in m_ids → 不可见，继续找上一版本
-- 版本3: 103 < 108, 103 not in m_ids → 可见
```

**追问：如果所有版本都不可见怎么办？**

**追问答案：**
如果遍历完整个版本链都找不到可见版本，说明该行对当前事务不可见，返回空结果（相当于行不存在）。

---

### 3. RR 和 RC 的 Read View 生成时机有什么不同？

**答案：**

这是 RR（可重复读）和 RC（读已提交）隔离级别实现差异的核心所在：

| 隔离级别 | Read View 生成时机 | 效果 |
|---------|-------------------|------|
| RC | 每次 SELECT 执行时生成 | 能看到已提交的新数据 |
| RR | 事务第一次 SELECT 时生成，后续复用 | 整个事务期间看到一致的数据快照 |

**RC 隔离级别示例：**

```sql
-- 事务A (RC隔离级别)
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 创建 Read View 1
-- 结果: name = 'Alice'

-- 此时事务B提交了修改
-- UPDATE users SET name = 'Bob' WHERE id = 1; COMMIT;

SELECT * FROM users WHERE id = 1;  -- 创建新的 Read View 2
-- 结果: name = 'Bob' (能看到事务B的提交)
COMMIT;
```

**RR 隔离级别示例：**

```sql
-- 事务A (RR隔离级别)
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 创建 Read View，后续复用
-- 结果: name = 'Alice'

-- 此时事务B提交了修改
-- UPDATE users SET name = 'Bob' WHERE id = 1; COMMIT;

SELECT * FROM users WHERE id = 1;  -- 复用第一次的 Read View
-- 结果: name = 'Alice' (仍然看到同样的数据)
COMMIT;
```

**原理对比图：**

```
RC 隔离级别：
时间线 ──────────────────────────────────────────────►
        │                    │                    │
      SELECT1             SELECT2              SELECT3
        │                    │                    │
        ▼                    ▼                    ▼
    ReadView1            ReadView2            ReadView3
    (新的快照)           (新的快照)           (新的快照)

RR 隔离级别：
时间线 ──────────────────────────────────────────────►
        │                    │                    │
      SELECT1             SELECT2              SELECT3
        │                    │                    │
        ▼                    │                    │
    ReadView1 ──────────复用──────────────复用────────►
    (第一次创建)              │                    │
```

**追问：为什么 RR 能解决不可重复读问题？**

**追问答案：**
因为 RR 在整个事务期间使用同一个 Read View，多次读取同样的数据时，可见性判断条件不变，所以看到的数据是一致的。

---

### 4. MVCC 能解决幻读吗？为什么？

**答案：**

这是一个经典面试题，答案需要分情况讨论：

**MVCC 快照读场景：可以部分解决幻读**

```sql
-- 事务A (RR隔离级别)
BEGIN;
SELECT * FROM users WHERE age > 20;  -- 快照读，创建 Read View
-- 结果: 3条记录

-- 事务B 插入新数据并提交
INSERT INTO users (name, age) VALUES ('New', 25); COMMIT;

SELECT * FROM users WHERE age > 20;  -- 快照读，复用 Read View
-- 结果: 仍然是3条记录（没有看到新插入的记录）
-- ✅ 幻读被避免了
COMMIT;
```

**当前读场景：MVCC 无法解决幻读**

```sql
-- 事务A (RR隔离级别)
BEGIN;
SELECT * FROM users WHERE age > 20;  -- 快照读
-- 结果: 3条记录

-- 事务B 插入新数据并提交
INSERT INTO users (name, age) VALUES ('New', 25); COMMIT;

SELECT * FROM users WHERE age > 20 FOR UPDATE;  -- 当前读！
-- 结果: 4条记录（看到新插入的记录）
-- ❌ 幻读发生了
COMMIT;
```

**完整的幻读解决方案：Next-Key Lock**

```
┌─────────────────────────────────────────────────────────────┐
│                    幻读解决方案                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   快照读 ──────────► MVCC ──────► 解决                      │
│                                                             │
│   当前读 ──────────► Next-Key Lock ─► 解决                  │
│                    (记录锁 + 间隙锁)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：为什么当前读会看到新数据？**

**追问答案：**
当前读（如 `SELECT ... FOR UPDATE`、`SELECT ... LOCK IN SHARE MODE`、`UPDATE`、`DELETE`、`INSERT`）会读取最新已提交的数据，不走 MVCC 快照，因此可能看到其他事务新插入的记录。

---

### 5. 当前读和快照读的区别？

**答案：**

| 特性 | 快照读 | 当前读 |
|-----|-------|-------|
| 读取方式 | 读取 MVCC 快照版本 | 读取最新已提交版本 |
| 是否加锁 | 不加锁 | 加锁 |
| SQL 语句 | 普通 SELECT | SELECT FOR UPDATE 等 |
| 一致性 | 可能读取历史数据 | 读取最新数据 |

**快照读示例：**

```sql
-- 快照读：普通 SELECT 语句
SELECT * FROM users WHERE id = 1;
SELECT * FROM users WHERE age > 20;
```

**当前读示例：**

```sql
-- 当前读：加锁的 SELECT
SELECT * FROM users WHERE id = 1 FOR UPDATE;      -- 排他锁
SELECT * FROM users WHERE id = 1 LOCK IN SHARE MODE;  -- 共享锁 (MySQL 5.7)
SELECT * FROM users WHERE id = 1 FOR SHARE;       -- 共享锁 (MySQL 8.0)

-- 当前读：DML 操作
UPDATE users SET name = 'New' WHERE id = 1;
DELETE FROM users WHERE id = 1;
INSERT INTO users (name, age) VALUES ('Test', 20);
```

**工作原理对比：**

```
快照读流程：
┌──────────┐    ┌────────────┐    ┌──────────────┐
│  SELECT  │───►│   MVCC     │───►│ Undo Log     │
│  语句    │    │  Read View │    │ 版本链遍历    │
└──────────┘    └────────────┘    └──────────────┘

当前读流程：
┌──────────┐    ┌────────────┐    ┌──────────────┐
│ SELECT   │───►│  加锁      │───►│ 读取最新版本  │
│ FOR UDPATE│    │ (行锁/间隙锁)│    │ (已提交数据)  │
└──────────┘    └────────────┘    └──────────────┘
```

**追问：为什么 UPDATE 是当前读？**

**追问答案：**
UPDATE 需要修改数据，必须确保修改的是最新版本，否则会造成数据不一致。如果基于快照读的旧数据修改，会覆盖其他事务已提交的修改。

---

### 6. 为什么 RC 每次都生成新的 Read View？

**答案：**

这是由 RC 隔离级别的语义决定的：**读取已提交的数据**。

**语义分析：**

```sql
-- RC 隔离级别的语义：可以看到其他事务已提交的修改
-- 事务A
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 此时事务B未提交
-- 结果: name = 'Alice'

-- 事务B: UPDATE users SET name = 'Bob' WHERE id = 1; COMMIT;

SELECT * FROM users WHERE id = 1;  -- 此时事务B已提交
-- 必须能看到: name = 'Bob'（已提交的数据）
COMMIT;
```

**每次生成新 Read View 的原因：**

```
┌─────────────────────────────────────────────────────────────┐
│  RC 隔离级别的设计目标：                                      │
│                                                             │
│  1. 读已提交（Read Committed）= 可以看到已提交的最新数据       │
│  2. 但不能看到未提交的数据（脏读防护）                         │
│                                                             │
│  实现方式：每次 SELECT 都检查当前有哪些事务是活跃的             │
│           （即生成新的 Read View）                            │
└─────────────────────────────────────────────────────────────┘
```

**如果复用 Read View 会怎样：**

```sql
-- 假设复用 Read View
BEGIN;
-- 第一次 SELECT，m_ids = [102, 103]
SELECT * FROM users WHERE id = 1;  -- 看不到事务 102, 103 的修改

-- 事务 102, 103 都提交了

SELECT * FROM users WHERE id = 1;  
-- 如果复用 Read View，m_ids 仍然是 [102, 103]
-- 仍然看不到已提交的修改 → 违反 RC 语义！
COMMIT;
```

**追问：RR 复用 Read View 为什么不会有问题？**

**追问答案：**
RR（可重复读）的语义是：事务期间多次读取同一数据，结果必须一致。复用 Read View 正好实现了这个语义，让整个事务看到同一个数据快照。

---

### 7. Undo Log 版本链是如何组织的？

**答案：**

每条记录的隐藏字段 `roll_pointer` 指向 Undo Log 中的旧版本，形成版本链：

**记录的隐藏字段：**

```
┌─────────────────────────────────────────────────────────────┐
│                    InnoDB 行记录格式                         │
├─────────────────────────────────────────────────────────────┤
│  db_trx_id  │  db_roll_ptr  │  db_row_id  │  用户数据列...  │
│  (6字节)    │   (7字节)     │   (6字节)   │                │
└─────────────────────────────────────────────────────────────┘
│             │               │
│             │               └── 自增主键（无显式主键时使用）
│             └── 指向 Undo Log 中上一版本的指针
└── 最后修改该行的事务 ID
```

**版本链示例：**

```sql
-- 初始数据
INSERT INTO users (id, name) VALUES (1, 'Alice');  -- trx_id = 100

-- 第一次更新
UPDATE users SET name = 'Bob' WHERE id = 1;        -- trx_id = 101

-- 第二次更新
UPDATE users SET name = 'Carol' WHERE id = 1;      -- trx_id = 102

-- 第三次更新
UPDATE users SET name = 'David' WHERE id = 1;      -- trx_id = 103
```

**形成的版本链：**

```
当前数据 (trx_id = 103, name = 'David')
    │
    │ roll_pointer
    ▼
Undo Log 版本1 (trx_id = 102, name = 'Carol')
    │
    │ roll_pointer
    ▼
Undo Log 版本2 (trx_id = 101, name = 'Bob')
    │
    │ roll_pointer
    ▼
Undo Log 版本3 (trx_id = 100, name = 'Alice')
    │
    ▼
  NULL (初始版本)
```

**追问：Undo Log 什么时候会被清理？**

**追问答案：**
当没有任何活跃事务需要某个 Undo Log 版本时（即该版本对所有活跃事务都不可见或已有更新版本），Purge 线程会清理这些 Undo Log。

---

### 8. MVCC 如何实现非阻塞读？

**答案：**

MVCC 的核心优势是通过版本链实现读写互不阻塞：

**传统锁机制的问题：**

```
┌─────────────────────────────────────────────────────────────┐
│  传统锁机制：读操作需要获取共享锁，写操作需要获取排他锁        │
│                                                             │
│  读锁与写锁互斥 → 读操作会阻塞写操作                          │
│                                                             │
│  时间线：                                                    │
│  事务A: ───[读锁]────────[读锁]────────[读锁]──────────►     │
│  事务B:      ──────[等待写锁...]──────[获取写锁]─────►        │
│                    ↑                                         │
│              被阻塞等待                                       │
└─────────────────────────────────────────────────────────────┘
```

**MVCC 的非阻塞读：**

```
┌─────────────────────────────────────────────────────────────┐
│  MVCC 机制：读操作读取快照版本，不需要加锁                    │
│                                                             │
│  时间线：                                                    │
│  事务A: ───[快照读]─────[快照读]─────[快照读]──────────►      │
│  事务B:      ──────[写操作]─────[写操作]──────────────►       │
│                    ↑                                         │
│              无阻塞，各自操作                                 │
└─────────────────────────────────────────────────────────────┘
```

**代码示例：**

```sql
-- 事务A：读操作（快照读，不加锁）
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 走 MVCC，不阻塞事务B
-- 事务B 的更新不会影响事务A 的读取

-- 事务B：写操作
BEGIN;
UPDATE users SET name = 'New' WHERE id = 1;  -- 只加行锁
COMMIT;

-- 事务A 继续读取
SELECT * FROM users WHERE id = 1;  -- 仍然读取快照版本，不阻塞
COMMIT;
```

**追问：MVCC 有什么缺点？**

**追问答案：**
1. **空间开销**：Undo Log 占用额外存储空间
2. **维护成本**：需要维护版本链，频繁更新会产生大量 Undo Log
3. **清理压力**：Purge 线程需要异步清理过期版本
4. **不适用于当前读**：当前读仍需要加锁

---

## 锁机制深度篇

### 1. 行锁什么时候加？什么时候升级为表锁？

**答案：**

**行锁加锁时机：**

```sql
-- 1. SELECT ... FOR UPDATE（当前读）
SELECT * FROM users WHERE id = 1 FOR UPDATE;  -- 加排他行锁

-- 2. SELECT ... FOR SHARE（当前读）
SELECT * FROM users WHERE id = 1 FOR SHARE;   -- 加共享行锁

-- 3. UPDATE 语句
UPDATE users SET name = 'New' WHERE id = 1;   -- 加排他行锁

-- 4. DELETE 语句
DELETE FROM users WHERE id = 1;               -- 加排他行锁
```

**行锁升级为表锁的情况：**

```sql
-- 1. 索引失效，全表扫描
-- 假设 name 列没有索引
UPDATE users SET age = 20 WHERE name = 'Alice';  -- 表锁！

-- 2. 使用了不满足索引条件的索引
-- 联合索引 (a, b)，只用 b 作为条件
UPDATE users SET c = 1 WHERE b = 10;  -- 可能退化为表锁

-- 3. 锁定大量行时（MySQL 会判断是否升级）
-- 锁定超过一定比例的行时，可能升级为表锁以减少锁开销
```

**验证示例：**

```sql
-- 查看当前锁情况
SELECT 
    OBJECT_SCHEMA,
    OBJECT_NAME,
    LOCK_TYPE,
    LOCK_MODE,
    LOCK_STATUS
FROM performance_schema.data_locks
WHERE OBJECT_SCHEMA = 'test';

-- 情况1：使用主键，行锁
-- UPDATE users SET name = 'X' WHERE id = 1;
-- LOCK_TYPE: RECORD, LOCK_MODE: X, REC_NOT_GAP

-- 情况2：无索引条件，表锁
-- UPDATE users SET name = 'X' WHERE name = 'Alice';  -- name 无索引
-- LOCK_TYPE: TABLE, LOCK_MODE: IX
```

**追问：如何避免行锁升级为表锁？**

**追问答案：**
1. 确保 WHERE 条件使用索引
2. 检查 EXPLAIN 确认索引生效
3. 避免在大表上进行全表更新的操作
4. 合理设计索引覆盖查询条件

---

### 2. 间隙锁在什么隔离级别下才有？

**答案：**

间隙锁（Gap Lock）只在 **REPEATABLE READ** 隔离级别下存在，READ COMMITTED 隔离级别下没有间隙锁。

**为什么只在 RR 下存在？**

```
┌─────────────────────────────────────────────────────────────┐
│  间隙锁的目的：防止幻读                                       │
│                                                             │
│  RR 隔离级别需要解决幻读 → 需要间隙锁                          │
│  RC 隔离级别允许幻读 → 不需要间隙锁                            │
└─────────────────────────────────────────────────────────────┘
```

**间隙锁示例：**

```sql
-- 设置 RR 隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 事务A
BEGIN;
-- 假设 users 表有 id = 1, 5, 10 三条记录
SELECT * FROM users WHERE id > 3 AND id < 8 FOR UPDATE;
-- 加锁情况：
-- 1. id = 5 的记录锁
-- 2. (1, 5) 间隙锁
-- 3. (5, 10) 间隙锁

-- 事务B - 尝试插入
INSERT INTO users (id, name) VALUES (4, 'test');  -- 被阻塞！
INSERT INTO users (id, name) VALUES (7, 'test');  -- 被阻塞！
INSERT INTO users (id, name) VALUES (2, 'test');  -- 被阻塞！
INSERT INTO users (id, name) VALUES (20, 'test'); -- 成功！不在间隙内

COMMIT;
```

**RC 隔离级别对比：**

```sql
-- 设置 RC 隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 事务A
BEGIN;
SELECT * FROM users WHERE id > 3 AND id < 8 FOR UPDATE;
-- 只加 id = 5 的记录锁，没有间隙锁

-- 事务B - 尝试插入
INSERT INTO users (id, name) VALUES (4, 'test');  -- 成功！

COMMIT;
```

**隔离级别与锁对照表：**

| 锁类型 | READ UNCOMMITTED | READ COMMITTED | REPEATABLE READ | SERIALIZABLE |
|-------|-------------------|----------------|-----------------|--------------|
| 记录锁 | ✓ | ✓ | ✓ | ✓ |
| 间隙锁 | ✗ | ✗ | ✓ | ✓ |
| Next-Key Lock | ✗ | ✗ | ✓ | ✓ |

**追问：间隙锁之间会冲突吗？**

**追问答案：**
间隙锁之间不会冲突！间隙锁的唯一目的是阻止其他事务向间隙中插入记录，间隙锁可以共存。

```sql
-- 事务A
SELECT * FROM users WHERE id > 3 AND id < 8 FOR UPDATE;  -- 间隙锁 (1,5), (5,10)

-- 事务B
SELECT * FROM users WHERE id > 3 AND id < 8 FOR UPDATE;  -- 同样的间隙锁，不冲突！
-- 成功获取锁
```

---

### 3. Next-Key Lock 是怎么工作的？

**答案：**

Next-Key Lock = 记录锁（Record Lock）+ 间隙锁（Gap Lock），锁定记录本身以及记录前面的间隙。

**工作原理：**

```
假设表中有记录：id = 10, 20, 30, 40

索引结构：
┌───────┬───────┬───────┬───────┬───────┐
│ (-∞,10]│ (10,20]│ (20,30]│ (30,40]│ (40,+∞)│
└───────┴───────┴───────┴───────┴───────┘
    ▲        ▲        ▲        ▲        ▲
    │        │        │        │        │
  Next-Key  Next-Key  Next-Key  Next-Key  间隙锁
  Lock 1    Lock 2    Lock 3    Lock 4    (临键锁)
```

**具体示例：**

```sql
-- 假设 users 表有 id = 10, 20, 30, 40 四条记录

-- 事务A
BEGIN;
SELECT * FROM users WHERE id = 20 FOR UPDATE;
-- 加锁：Next-Key Lock (10, 20]
-- 即：间隙锁 (10, 20) + 记录锁 id=20

-- 事务B 尝试操作
INSERT INTO users VALUES (15, 'test');  -- 阻塞！在间隙 (10,20) 内
UPDATE users SET name = 'X' WHERE id = 20;  -- 阻塞！记录锁冲突
INSERT INTO users VALUES (5, 'test');   -- 成功！
INSERT INTO users VALUES (25, 'test');  -- 成功！

COMMIT;
```

**范围查询示例：**

```sql
-- 事务A
BEGIN;
SELECT * FROM users WHERE id >= 20 AND id < 30 FOR UPDATE;
-- 加锁范围：
-- 1. Next-Key Lock (10, 20]   -- id = 20
-- 2. Gap Lock (20, 30)        -- id 在 20 到 30 之间

-- 事务B
INSERT INTO users VALUES (15, 'test');  -- 阻塞！(10, 20) 间隙
INSERT INTO users VALUES (25, 'test');  -- 阻塞！(20, 30) 间隙
INSERT INTO users VALUES (30, 'test');  -- 成功！(30 不在锁范围内)

COMMIT;
```

**追问：唯一索引和非唯一索引的 Next-Key Lock 有什么区别？**

**追问答案：**
- **唯一索引**：等值查询时，退化为记录锁（因为值唯一，不可能插入相同值）
- **非唯一索引**：等值查询时，仍然是 Next-Key Lock

```sql
-- 假设 id 是主键（唯一索引）
SELECT * FROM users WHERE id = 20 FOR UPDATE;
-- 只加记录锁 id = 20，没有间隙锁

-- 假设 age 是普通索引，有值 20, 20, 30
SELECT * FROM users WHERE age = 20 FOR UPDATE;
-- 加 Next-Key Lock (-∞, 20] 和 (20, 20]，还要加主键的记录锁
```

---

### 4. 插入意向锁是什么？

**答案：**

插入意向锁是一种特殊的间隙锁，专门用于 INSERT 操作。

**特点：**

| 特性 | 说明 |
|-----|------|
| 锁类型 | 间隙锁的一种 |
| 触发时机 | INSERT 操作时 |
| 锁定范围 | 待插入记录所在的间隙 |
| 冲突规则 | 与间隙锁冲突，彼此之间不冲突 |

**工作原理：**

```
假设有记录 id = 10, 20, 30

间隙划分：(−∞, 10), (10, 20), (20, 30), (30, +∞)

事务A 持有 Gap Lock (10, 20)
└─► 阻止其他事务向该间隙插入记录

事务B 尝试 INSERT id = 15
└─► 需要获取 (10, 20) 的插入意向锁
└─► 被事务A 的 Gap Lock 阻塞
```

**示例演示：**

```sql
-- 事务A
BEGIN;
SELECT * FROM users WHERE id > 10 AND id < 20 FOR UPDATE;
-- 获取间隙锁 (10, 20)

-- 事务B
BEGIN;
INSERT INTO users (id, name) VALUES (15, 'test');
-- 尝试获取插入意向锁，被阻塞！
-- 等待事务A 释放间隙锁

-- 事务A
COMMIT;  -- 释放间隙锁

-- 事务B 此时可以获取插入意向锁，插入成功
COMMIT;
```

**多个 INSERT 不冲突：**

```sql
-- 事务A
INSERT INTO users (id, name) VALUES (15, 'test1');
-- 获取 (10, 20) 的插入意向锁

-- 事务B
INSERT INTO users (id, name) VALUES (16, 'test2');
-- 同样获取 (10, 20) 的插入意向锁
-- 插入意向锁之间不冲突，可以并行插入！

COMMIT;
```

**追问：插入意向锁的设计目的是什么？**

**追问答案：**
提高并发插入性能。同一间隙内的多个插入操作可以并行进行，只要没有其他事务持有该间隙的间隙锁。这避免了插入操作之间不必要的互斥。

---

### 5. 意向锁的作用是什么？

**答案：**

意向锁是表级锁，用于表示事务对表中某些行的锁定意图，提高表锁和行锁之间的兼容性判断效率。

**意向锁类型：**

| 锁类型 | 含义 | 说明 |
|-------|------|------|
| IS（意向共享锁） | 想要对某些行加共享锁 | SELECT ... FOR SHARE |
| IX（意向排他锁） | 想要对某些行加排他锁 | SELECT ... FOR UPDATE, UPDATE, DELETE, INSERT |

**为什么需要意向锁？**

```
┌─────────────────────────────────────────────────────────────┐
│  没有意向锁时的问题：                                         │
│                                                             │
│  事务A: 对某些行加了行锁                                      │
│  事务B: 想要加表锁                                            │
│                                                             │
│  如何判断是否冲突？                                           │
│  → 需要扫描整张表检查是否有行锁！效率极低                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  有意向锁时：                                                 │
│                                                             │
│  事务A: 加行锁前，先加意向锁（IS 或 IX）                       │
│  事务B: 想要加表锁，只需检查表级意向锁                          │
│                                                             │
│  → 只需检查表级锁，O(1) 判断！                                │
└─────────────────────────────────────────────────────────────┘
```

**锁兼容性矩阵：**

```
         表锁请求
         S      X
      ┌─────┬─────┐
    IS│  ✓  │  ✗  │
已有    ├─────┼─────┤
    IX│  ✗  │  ✗  │
      └─────┴─────┘

✓ = 兼容   ✗ = 冲突

说明：
- IS 和 S 兼容：想加共享行锁，和表共享锁不冲突
- IX 和 S 冲突：想加排他行锁，和表共享锁冲突
- IS/X 和 X 都冲突：表排他锁和任何意向锁都冲突
```

**示例：**

```sql
-- 事务A
BEGIN;
SELECT * FROM users WHERE id = 1 FOR UPDATE;
-- 先加 IX（意向排他锁）在表上
-- 再加 X（排他锁）在 id=1 的行上

-- 事务B
LOCK TABLES users READ;
-- 想要加 S（共享表锁）
-- 检测到 IX 存在，冲突，等待！

-- 事务A
COMMIT;  -- 释放 IX

-- 事务B 现在可以获取 S 表锁
```

**追问：意向锁什么时候释放？**

**追问答案：**
意向锁在事务结束时释放（COMMIT 或 ROLLBACK），不会在语句执行完就释放。

---

### 6. 什么情况下会死锁？死锁检测怎么做？

**答案：**

**死锁产生的四个必要条件：**

1. **互斥条件**：资源只能被一个事务占用
2. **请求与保持**：持有资源的同时请求新资源
3. **不可剥夺**：不能强制释放其他事务的资源
4. **循环等待**：事务间形成循环等待关系

**经典死锁场景：**

```sql
-- 场景：交叉更新
-- 初始数据：id=1 balance=100, id=2 balance=200

-- 事务A                          -- 事务B
BEGIN;                            BEGIN;
UPDATE accounts                   UPDATE accounts
SET balance = balance - 50        SET balance = balance - 30
WHERE id = 1;  -- 锁住 id=1       WHERE id = 2;  -- 锁住 id=2
                                  
UPDATE accounts                   UPDATE accounts
SET balance = balance + 50        SET balance = balance + 30
WHERE id = 2;  -- 等待 id=2       WHERE id = 1;  -- 等待 id=1
                                  
-- 死锁！事务A等B，事务B等A
```

**死锁检测方法：**

```sql
-- 1. 查看死锁日志
SHOW ENGINE INNODB STATUS;

-- 2. 查看当前锁等待
SELECT 
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.innodb_lock_waits w
INNER JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
INNER JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;

-- 3. MySQL 8.0 使用 performance_schema
SELECT * FROM performance_schema.data_locks;
SELECT * FROM performance_schema.data_lock_waits;
```

**死锁检测配置：**

```sql
-- 查看死锁检测是否开启
SHOW VARIABLES LIKE 'innodb_deadlock_detect';

-- 开启死锁检测（默认开启）
SET GLOBAL innodb_deadlock_detect = ON;

-- 设置锁等待超时（秒）
SET innodb_lock_wait_timeout = 50;
```

**追问：如何分析和解决死锁？**

**追问答案：**

**分析方法：**
1. 开启死锁日志：`SET GLOBAL innodb_print_all_deadlocks = ON;`
2. 分析死锁日志，找出冲突的 SQL
3. 绘制锁等待图，确定循环依赖

**解决方案：**

```sql
-- 1. 统一加锁顺序
-- 所有事务按相同顺序访问资源
-- 比如都按 id 升序处理

-- 2. 大事务拆小事务
-- 减少锁持有时间

-- 3. 降低隔离级别
-- RC 比 RR 锁范围小

-- 4. 添加合适的索引
-- 减少锁范围

-- 5. 使用乐观锁
-- 不依赖数据库锁机制
UPDATE products 
SET stock = stock - 1, version = version + 1
WHERE id = 1 AND version = 10;
```

---

### 7. 如何分析线上死锁问题？

**答案：**

**完整的死锁分析流程：**

**步骤1：收集死锁信息**

```sql
-- 开启死锁日志记录
SET GLOBAL innodb_print_all_deadlocks = ON;

-- 查看最近的死锁信息
SHOW ENGINE INNODB STATUS\G
-- 在 LATEST DETECTED DEADLOCK 部分查看详情
```

**步骤2：解读死锁日志**

```
LATEST DETECTED DEADLOCK
------------------------
2024-01-15 10:30:00 0x7f8a4c0b4700
*** (1) TRANSACTION:
TRANSACTION 12345, ACTIVE 2 sec starting index read
mysql tables in use 1, locked 1
LOCK WAIT 2 lock struct(s), heap size 1136, 1 row lock(s)
MySQL thread id 10, OS thread handle 140234567890432, query id 100 localhost root updating
UPDATE users SET name = 'Bob' WHERE id = 2   -- 事务1执行的SQL
*** (1) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 4 n bits 72 index PRIMARY of table `test`.`users`
Record lock, heap no 3 PHYSICAL RECORD: n_columns 4; compact format; info bits 0
 -- 等待 id=2 的锁

*** (2) TRANSACTION:
TRANSACTION 12346, ACTIVE 1 sec starting index read
mysql tables in use 1, locked 1
3 lock struct(s), heap size 1136, 2 row lock(s)
MySQL thread id 11, OS thread handle 140234567890433, query id 101 localhost root updating
UPDATE users SET name = 'Alice' WHERE id = 1  -- 事务2执行的SQL
*** (2) HOLDS THE LOCK(S):
RECORD LOCKS space id 58 page no 4 n bits 72 index PRIMARY of table `test`.`users`
Record lock, heap no 3 PHYSICAL RECORD: n_columns 4; compact format; info bits 0
 -- 持有 id=2 的锁
*** (2) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 4 n bits 72 index PRIMARY of table `test`.`users`
Record lock, heap no 2 PHYSICAL RECORD: n_columns 4; compact format; info bits 0
 -- 等待 id=1 的锁
```

**步骤3：绘制锁等待图**

```
事务1 (TRX 12345)              事务2 (TRX 12346)
      │                              │
      │ 持有 id=1 的锁               │ 持有 id=2 的锁
      │                              │
      ▼                              ▼
  等待 id=2 的锁 <─────────────→  等待 id=1 的锁
  
  形成循环等待 → 死锁！
```

**步骤4：优化建议**

```sql
-- 1. 统一访问顺序
-- 所有事务按 id 升序处理
-- 应用层代码改造：
public void transfer(int fromId, int toId, int amount) {
    int first = Math.min(fromId, toId);
    int second = Math.max(fromId, toId);
    
    // 按顺序锁定
    lockRow(first);
    lockRow(second);
    // ... 业务逻辑
}

-- 2. 使用 SELECT FOR UPDATE 预锁定
BEGIN;
SELECT * FROM users WHERE id IN (1, 2) ORDER BY id FOR UPDATE;
-- 按顺序锁定，避免交叉
UPDATE users SET name = 'Bob' WHERE id = 2;
UPDATE users SET name = 'Alice' WHERE id = 1;
COMMIT;

-- 3. 设置合理的锁等待超时
SET innodb_lock_wait_timeout = 10;  -- 10秒超时
```

---

### 8. 行锁的三种算法分别是什么？

**答案：**

InnoDB 行锁有三种算法：

| 锁类型 | 锁定范围 | 说明 |
|-------|---------|------|
| Record Lock | 单条记录 | 锁定索引记录本身 |
| Gap Lock | 间隙 | 锁定记录之间的间隙，不含记录 |
| Next-Key Lock | 记录 + 前间隙 | Record Lock + Gap Lock |

**详细说明：**

**1. Record Lock（记录锁）**

```sql
-- 唯一索引等值查询，精确匹配
SELECT * FROM users WHERE id = 1 FOR UPDATE;
-- 只锁定 id=1 这条记录

-- 图示：
-- ... [-∞] -- [10] -- [20] -- [30] -- [+∞] ...
--                    ↑
--                   锁定
```

**2. Gap Lock（间隙锁）**

```sql
-- 范围查询或非唯一索引等值查询
-- 假设有记录 id = 10, 20, 30
SELECT * FROM users WHERE id > 15 AND id < 25 FOR UPDATE;
-- 锁定间隙 (10, 20) 和 (20, 30)

-- 图示：
-- ... [-∞] -- [10] === [20] === [30] -- [+∞] ...
--                   ↑       ↑
--                间隙锁   间隙锁
```

**3. Next-Key Lock（临键锁）**

```sql
-- 默认情况下的加锁算法
-- 假设有记录 id = 10, 20, 30
SELECT * FROM users WHERE id <= 20 FOR UPDATE;
-- 锁定 (-∞, 10], (10, 20]

-- 图示：
-- ... [-∞] === [10] === [20] -- [30] -- [+∞] ...
--         ↑         ↑
--    Next-Key   Next-Key
--    (-∞,10]    (10,20]
```

**退化规则：**

```sql
-- 1. 唯一索引 + 等值查询 + 记录存在 → 退化为 Record Lock
SELECT * FROM users WHERE id = 20 FOR UPDATE;  -- id 是主键
-- 只锁定 id=20 这条记录

-- 2. 唯一索引 + 等值查询 + 记录不存在 → 退化为 Gap Lock
SELECT * FROM users WHERE id = 25 FOR UPDATE;  -- id=25 不存在
-- 锁定间隙 (20, 30)

-- 3. 非唯一索引 + 等值查询 → Next-Key Lock + Gap Lock
SELECT * FROM users WHERE age = 20 FOR UPDATE;  -- age 是普通索引
-- 锁定 age 索引的 (-∞, 20], (20, 20] 等
```

**追问：什么时候使用哪种锁？**

**追问答案：**
MySQL 自动选择，遵循以下规则：
- RR 隔离级别 + 非唯一索引 → Next-Key Lock
- RR 隔离级别 + 唯一索引 + 等值匹配 → Record Lock
- RC 隔离级别 → 只有 Record Lock
- INSERT 操作 → 插入意向锁

---

## 索引深度篇

### 1. 为什么 B+树比 B树更适合数据库？

**答案：**

B+树相比 B树有三个关键优势，使其成为数据库索引的最佳选择：

**结构对比：**

```
B树结构：
                    [30]
                   /    \
            [10,20]    [40,50,60]
            /  |  \     /  |   |  \
          数据 数据 数据 数据 数据 数据 数据
          ↑
        每个节点都存储数据

B+树结构：
                    [30]
                   /    \
            [10,20]    [40,50,60]
            /  |  \     /  |   |  \
          叶子节点     叶子节点     叶子节点
          ↓   ↓   ↓   ↓   ↓    ↓    ↓
         数据 数据 数据 数据 数据 数据 数据
          ↑___________________________↑
                  链表连接所有叶子节点
```

**三大优势：**

| 优势 | 说明 | 影响 |
|-----|------|------|
| **单节点存储更多键值** | 非叶子节点不存数据，只存键值 | 树更矮，磁盘 I/O 更少 |
| **查询性能稳定** | 所有数据都在叶子节点 | 查询路径长度一致 |
| **范围查询高效** | 叶子节点形成有序链表 | 范围扫描无需回溯 |

**详细分析：**

**1. 磁盘 I/O 效率**

```sql
-- 假设：
-- 磁盘页大小：16KB
-- 主键大小：8字节
-- 数据大小：1KB

-- B树每个节点能存储的记录数：
-- (16KB) / (8B + 1KB) ≈ 15 条记录

-- B+树非叶子节点能存储的键值数：
-- 16KB / 8B = 2048 个键值
-- 每个指针 8 字节：(16KB) / (8B + 8B) ≈ 1024 个指针

-- 结论：B+树比 B树矮很多，I/O 次数更少
```

**2. 范围查询性能**

```sql
-- 查询范围：id BETWEEN 10 AND 100

-- B树：需要中序遍历，多次回溯父节点
-- 时间复杂度：O(n * log_m(N))

-- B+树：找到起点后，沿链表顺序扫描
-- 时间复杂度：O(log_m(N) + n)

-- 示例
SELECT * FROM users WHERE id BETWEEN 10 AND 100;
-- B+树：找到 id=10 的叶子节点，沿链表扫描到 id=100
```

**3. 查询稳定性**

```sql
-- B树：不同数据在不同层级
SELECT * FROM users WHERE id = 1;   -- 可能在第2层找到
SELECT * FROM users WHERE id = 100; -- 可能在第4层找到
-- 性能不一致

-- B+树：所有数据都在叶子节点
SELECT * FROM users WHERE id = 1;   -- 一定到叶子节点
SELECT * FROM users WHERE id = 100; -- 一定到叶子节点
-- 性能一致
```

**追问：为什么 MySQL 选择 B+树而不是 Hash 索引？**

**追问答案：**
- Hash 索引不支持范围查询
- Hash 索引不支持排序
- Hash 索引不支持模糊查询
- Hash 索引可能有哈希冲突

---

### 2. 聚簇索引和非聚簇索引的根本区别？

**答案：**

**核心区别：数据存储位置不同**

```
聚簇索引：
┌─────────────────────────────────────┐
│          B+树结构                    │
│                                     │
│    [30]                             │
│   /    \                            │
│ [10,20]  [40,50]                    │
│                                     │
│ 叶子节点存储：                       │
│ ┌─────────────────────────────────┐ │
│ │ id=10 | name=Alice | age=20 ... │ │
│ └─────────────────────────────────┘ │
│        完整的行数据！                │
└─────────────────────────────────────┘

非聚簇索引（辅助索引）：
┌─────────────────────────────────────┐
│          B+树结构                    │
│                                     │
│    [30]                             │
│   /    \                            │
│ [10,20]  [40,50]                    │
│                                     │
│ 叶子节点存储：                       │
│ ┌───────────────────┐               │
│ │ age=20 | id=10   │               │
│ └───────────────────┘               │
│   索引值 + 主键值                    │
└─────────────────────────────────────┘
```

**对比表格：**

| 特性 | 聚簇索引 | 非聚簇索引 |
|-----|---------|-----------|
| 叶子节点存储 | 完整行数据 | 索引列值 + 主键值 |
| 数量 | 每张表只有一个 | 可以有多个 |
| 主键查询 | 直接找到数据 | 需要回表 |
| 数据顺序 | 物理有序 | 逻辑有序 |
| 插入性能 | 可能页分裂 | 影响较小 |

**查询示例：**

```sql
-- 聚簇索引查询（主键）
SELECT * FROM users WHERE id = 1;
-- 直接在聚簇索引中找到数据，一次 I/O

-- 非聚簇索引查询（辅助索引）
SELECT * FROM users WHERE age = 20;
-- 1. 在 age 索引中找到 age=20 的记录，获取主键 id
-- 2. 回表到聚簇索引查找完整数据
-- 需要两次 I/O
```

**实际存储结构：**

```sql
-- 创建表
CREATE TABLE users (
    id INT PRIMARY KEY,       -- 聚簇索引
    name VARCHAR(50),
    age INT,
    INDEX idx_age (age)       -- 非聚簇索引
) ENGINE=InnoDB;

-- 聚簇索引叶子节点：
-- | id | name | age | 其他列... |
-- | 1  | Alice| 20  | ...      |
-- | 2  | Bob  | 25  | ...      |

-- 非聚簇索引 idx_age 叶子节点：
-- | age | id |
-- | 20  | 1  |
-- | 25  | 2  |
```

**追问：为什么建议使用自增主键？**

**追问答案：**
1. **插入性能好**：新记录追加到末尾，不会产生页分裂
2. **空间利用率高**：页填充率高，减少碎片
3. **范围查询高效**：物理连续，顺序 I/O

```sql
-- 自增主键：顺序插入
INSERT INTO users (id, name) VALUES (1, 'A'), (2, 'B'), (3, 'C');
-- 数据顺序存储，页填充率高

-- UUID 主键：随机插入
INSERT INTO users (id, name) VALUES ('uuid-xxx', 'A');
-- 数据位置随机，频繁页分裂，碎片化严重
```

---

### 3. 为什么辅助索引要回表？

**答案：**

回表是因为辅助索引的叶子节点只存储索引列值和主键值，不存储完整的行数据。

**回表过程图解：**

```
查询：SELECT * FROM users WHERE age = 20;

步骤1：在辅助索引中查找
┌─────────────────────────────────────┐
│        age 索引 B+树                 │
│                                     │
│           [30]                      │
│          /    \                     │
│      [20,25]  [40,50]               │
│        ↓                            │
│   叶子节点：                         │
│   | age | id |                      │
│   | 20  | 1  |  ← 找到主键 id=1     │
│   | 25  | 3  |                      │
└─────────────────────────────────────┘

步骤2：回表到聚簇索引
┌─────────────────────────────────────┐
│        主键索引 B+树                 │
│                                     │
│           [30]                      │
│          /    \                     │
│      [10,20]  [40,50]               │
│        ↓                            │
│   叶子节点：                         │
│   | id | name | age | ... |         │
│   | 1  | Alice| 20  | ... | ← 完整数据│
└─────────────────────────────────────┘
```

**为什么要这样设计？**

```
┌─────────────────────────────────────────────────────────────┐
│  如果辅助索引也存储完整数据：                                  │
│                                                             │
│  问题1：空间浪费                                             │
│  - 每个索引都存完整数据，表有 10 个索引就存 10 份数据          │
│  - 存储成本急剧增加                                          │
│                                                             │
│  问题2：维护成本高                                           │
│  - 更新数据时需要修改所有索引                                 │
│  - 性能严重下降                                              │
│                                                             │
│  当前设计优势：                                               │
│  - 辅助索引只存索引列 + 主键，空间高效                        │
│  - 更新数据只需修改聚簇索引                                   │
│  - 回表代价可控                                              │
└─────────────────────────────────────────────────────────────┘
```

**避免回表的方法：覆盖索引**

```sql
-- 普通查询：需要回表
SELECT * FROM users WHERE age = 20;
-- 辅助索引 → 回表获取所有列

-- 覆盖索引：不需要回表
SELECT id, age FROM users WHERE age = 20;
-- 辅助索引中已有 id 和 age，直接返回

-- 建立联合索引优化
CREATE INDEX idx_age_name ON users(age, name);

SELECT id, age, name FROM users WHERE age = 20;
-- 联合索引包含所有需要的列，覆盖索引，无需回表
```

**Explain 验证：**

```sql
EXPLAIN SELECT id, age FROM users WHERE age = 20;
-- Extra: Using index  ← 覆盖索引

EXPLAIN SELECT * FROM users WHERE age = 20;
-- Extra: NULL  ← 需要回表
```

**追问：回表一定是坏事吗？**

**追问答案：**
不一定。如果回表次数很少（如只查少量记录），代价可以接受。但如果回表次数很多（如查大量数据），应该考虑覆盖索引优化。

---

### 4. 联合索引的存储结构？

**答案：**

联合索引遵循**最左前缀原则**，按索引列顺序构建 B+树。

**存储结构：**

```sql
-- 创建联合索引
CREATE INDEX idx_name_age ON users(name, age);

-- 存储结构示意：
-- 索引值按 (name, age) 排序
┌─────────────────────────────────────┐
│     idx_name_age 索引 B+树           │
│                                     │
│         [('Bob', 20)]               │
│         /          \                │
│   [('Alice',18)] [('Carol',22)]     │
│                                     │
│   叶子节点（有序）：                  │
│   | name  | age | id |              │
│   | Alice | 18  | 1  |              │
│   | Alice | 20  | 5  |              │
│   | Bob   | 20  | 2  |              │
│   | Carol | 22  | 3  |              │
└─────────────────────────────────────┘

-- 排序规则：先按 name 排序，name 相同再按 age 排序
```

**最左前缀原则：**

```sql
-- 联合索引 idx_name_age(name, age)

-- ✅ 能走索引
SELECT * FROM users WHERE name = 'Alice';
SELECT * FROM users WHERE name = 'Alice' AND age = 18;
SELECT * FROM users WHERE name = 'Alice' AND age > 15;

-- ❌ 不能走索引
SELECT * FROM users WHERE age = 18;  -- 缺少最左列
SELECT * FROM users WHERE age = 18 AND name = 'Alice';  -- 顺序不对，但优化器可能优化

-- ✅ 部分走索引（只用到 name）
SELECT * FROM users WHERE name = 'Alice' AND age LIKE '1%';
```

**索引下推优化：**

```sql
-- MySQL 5.6 之前：索引下推前
SELECT * FROM users WHERE name LIKE 'A%' AND age = 18;
-- 1. 索引找到所有 name LIKE 'A%' 的记录
-- 2. 回表获取完整数据
-- 3. 在 Server 层过滤 age = 18

-- MySQL 5.6+：索引下推后
SELECT * FROM users WHERE name LIKE 'A%' AND age = 18;
-- 1. 索引直接过滤 name LIKE 'A%' AND age = 18
-- 2. 只回表满足条件的记录
-- 减少回表次数！

-- Explain 查看
EXPLAIN SELECT * FROM users WHERE name LIKE 'A%' AND age = 18;
-- Extra: Using index condition  ← 使用了索引下推
```

**追问：联合索引的设计原则是什么？**

**追问答案：**

```sql
-- 1. 区分度高的列放前面
-- name 区分度高，age 区分度低
CREATE INDEX idx_name_age ON users(name, age);  -- 好
-- CREATE INDEX idx_age_name ON users(age, name);  -- 差

-- 2. 覆盖常用查询
-- 经常查询 name 和 age
CREATE INDEX idx_name_age ON users(name, age);

-- 3. 考虑排序
-- 经常按 name, age 排序
CREATE INDEX idx_name_age ON users(name, age);
-- 索引天然有序，避免 filesort

-- 4. 避免冗余索引
-- 已有 (name, age)，不需要单独建 (name)
```

---

### 5. 索引条件下推（ICP）的原理？

**答案：**

索引条件下推（Index Condition Pushdown）是 MySQL 5.6 的优化特性，将 WHERE 条件的过滤下推到存储引擎层。

**传统方式（无 ICP）：**

```
┌──────────────┐              ┌──────────────┐
│   Server 层  │              │  存储引擎层   │
├──────────────┤              ├──────────────┤
│              │◄──返回记录──│  读取索引    │
│  过滤条件    │              │  回表取数据  │
│  age = 18    │              │              │
│              │──请求记录──►│              │
└──────────────┘              └──────────────┘

问题：大量无用的回表操作
```

**ICP 方式：**

```
┌──────────────┐              ┌──────────────┐
│   Server 层  │              │  存储引擎层   │
├──────────────┤              ├──────────────┤
│              │◄──返回匹配──│  读取索引    │
│              │   的记录    │  ↓          │
│              │              │  ICP 过滤   │
│              │              │  age = 18   │
│              │              │  ↓          │
│              │              │  回表取数据  │
└──────────────┘              └──────────────┘

优化：先在索引层过滤，减少回表
```

**示例对比：**

```sql
-- 表结构
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    city VARCHAR(50),
    INDEX idx_name (name)
);

-- 查询
SELECT * FROM users WHERE name LIKE 'A%' AND age = 18;

-- 无 ICP：
-- 1. 索引找到所有 name LIKE 'A%' 的记录（假设 1000 条）
-- 2. 回表 1000 次
-- 3. Server 层过滤 age = 18（假设只剩 10 条）
-- 回表次数：1000 次

-- 有 ICP：
-- 1. 索引找到 name LIKE 'A%' 的记录
-- 2. 在索引层就用 age = 18 过滤（假设只剩 10 条）
-- 3. 只回表 10 次
-- 回表次数：10 次
```

**Explain 查看 ICP：**

```sql
EXPLAIN SELECT * FROM users WHERE name LIKE 'A%' AND age = 18\G

-- 输出：
Extra: Using index condition  ← 使用了 ICP

-- 关闭 ICP 对比
SET optimizer_switch='index_condition_pushdown=off';
EXPLAIN SELECT * FROM users WHERE name LIKE 'A%' AND age = 18\G
-- 输出：
Extra: Using where  ← 没有 ICP
```

**追问：ICP 适用于什么场景？**

**追问答案：**
1. WHERE 条件有索引列和非索引列
2. 索引列的条件可以被下推
3. 范围查询或 LIKE 查询结合其他条件

---

### 6. 什么是索引失效？常见场景有哪些？

**答案：**

索引失效是指查询没有使用预期的索引，导致全表扫描。

**常见失效场景：**

**1. 对索引列使用函数或运算**

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE YEAR(create_time) = 2024;
SELECT * FROM users WHERE id + 1 = 10;
SELECT * FROM users WHERE SUBSTRING(name, 1, 3) = 'Ali';

-- ✅ 索引生效
SELECT * FROM users WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01';
SELECT * FROM users WHERE id = 9;
SELECT * FROM users WHERE name LIKE 'Ali%';
```

**2. 隐式类型转换**

```sql
-- name 是 VARCHAR 类型
-- ❌ 索引失效（字符串与数字比较）
SELECT * FROM users WHERE name = 123;  -- 隐式转换为数字

-- ✅ 索引生效
SELECT * FROM users WHERE name = '123';
```

**3. LIKE 以通配符开头**

```sql
-- ❌ 索引失效
SELECT * FROM users WHERE name LIKE '%Alice';
SELECT * FROM users WHERE name LIKE '%Ali%';

-- ✅ 索引生效
SELECT * FROM users WHERE name LIKE 'Alice%';
SELECT * FROM users WHERE name LIKE 'Ali%';
```

**4. OR 条件导致失效**

```sql
-- ❌ 索引失效（name 有索引，age 无索引）
SELECT * FROM users WHERE name = 'Alice' OR age = 18;

-- ✅ 两个条件都有索引
CREATE INDEX idx_age ON users(age);
SELECT * FROM users WHERE name = 'Alice' OR age = 18;  -- 索引生效
```

**5. 联合索引不满足最左前缀**

```sql
-- 联合索引 idx_name_age(name, age)
-- ❌ 索引失效
SELECT * FROM users WHERE age = 18;

-- ✅ 索引生效
SELECT * FROM users WHERE name = 'Alice' AND age = 18;
```

**6. NOT IN、NOT LIKE、<>**

```sql
-- ❌ 可能索引失效
SELECT * FROM users WHERE age NOT IN (18, 20);
SELECT * FROM users WHERE name NOT LIKE 'A%';
SELECT * FROM users WHERE age <> 18;
```

**7. IS NULL / IS NOT NULL**

```sql
-- ❌ 可能索引失效（取决于数据分布）
SELECT * FROM users WHERE name IS NULL;
SELECT * FROM users WHERE name IS NOT NULL;
```

**如何检测索引失效：**

```sql
-- 使用 EXPLAIN 分析
EXPLAIN SELECT * FROM users WHERE YEAR(create_time) = 2024\G

-- 关键指标：
-- type: ALL → 全表扫描（索引失效）
-- type: ref/range → 使用索引
-- key: NULL → 没使用索引
-- key: idx_name → 使用了 idx_name 索引
```

**追问：如何优化这些索引失效场景？**

**追问答案：**
1. 避免对索引列使用函数，改写查询条件
2. 确保类型匹配，避免隐式转换
3. 使用覆盖索引优化 LIKE '%xxx%'
4. 为 OR 条件的每个列建立索引
5. 合理设计联合索引顺序

---

### 7. 什么是索引下推和索引覆盖的区别？

**答案：**

两者都是索引优化技术，但解决的问题不同：

**对比：**

| 特性 | 覆盖索引 | 索引下推（ICP） |
|-----|---------|----------------|
| 解决问题 | 避免回表 | 减少回表次数 |
| 触发条件 | 查询列都在索引中 | WHERE 条件部分可下推 |
| 优化层次 | 完全不回表 | 过滤后回表 |
| Extra 显示 | Using index | Using index condition |

**覆盖索引：**

```sql
-- 联合索引
CREATE INDEX idx_name_age ON users(name, age);

-- 覆盖索引：查询列都在索引中
SELECT name, age FROM users WHERE name = 'Alice';
-- Extra: Using index
-- 完全不需要回表！
```

**索引下推：**

```sql
-- 单列索引
CREATE INDEX idx_name ON users(name);

-- 查询需要 name（索引列）和 age（非索引列）
SELECT * FROM users WHERE name LIKE 'A%' AND age = 18;
-- Extra: Using index condition
-- 在索引层过滤 age = 18，减少回表次数
```

**组合使用：**

```sql
-- 联合索引
CREATE INDEX idx_name_age ON users(name, age);

-- 同时满足覆盖索引和 ICP
SELECT name, age FROM users WHERE name LIKE 'A%' AND age = 18;
-- Extra: Using where; Using index
-- 覆盖索引生效，无需回表
```

**图解对比：**

```
覆盖索引流程：
┌─────────────────────────────────────────┐
│  SELECT name, age FROM users            │
│  WHERE name = 'Alice'                   │
│                                         │
│  ┌─────────────┐                        │
│  │ idx_name_age│                        │
│  │  B+树       │                        │
│  │             │                        │
│  │ 叶子节点：   │                        │
│  │ name,age,id │                        │
│  └──────┬──────┘                        │
│         │                               │
│         ▼                               │
│    直接返回结果                          │
│    （不回表）                            │
└─────────────────────────────────────────┘

索引下推流程：
┌─────────────────────────────────────────┐
│  SELECT * FROM users                    │
│  WHERE name LIKE 'A%' AND age = 18      │
│                                         │
│  ┌─────────────┐      ┌─────────────┐   │
│  │  idx_name   │      │  聚簇索引   │   │
│  │  B+树       │      │             │   │
│  │             │ ICP  │             │   │
│  │ 过滤:       │ 过滤 │ 完整数据    │   │
│  │ age=18      ├─────►│             │   │
│  └─────────────┘      └─────────────┘   │
│         │                    │          │
│         │    减少回表次数     │          │
└─────────────────────────────────────────┘
```

---

### 8. MySQL 8.0 的索引新特性有哪些？

**答案：**

MySQL 8.0 引入了多项重要的索引新特性：

**1. 降序索引（Descending Index）**

```sql
-- MySQL 5.7：降序索引只是语法支持，实际仍是升序
-- MySQL 8.0：真正支持降序存储

-- 创建降序索引
CREATE INDEX idx_name_desc ON users(name DESC);

-- 联合索引混合排序
CREATE INDEX idx_name_age ON users(name ASC, age DESC);

-- 优化场景：ORDER BY name ASC, age DESC
SELECT * FROM users ORDER BY name, age DESC;
-- MySQL 8.0 可以直接使用索引顺序，避免 filesort
```

**2. 隐藏索引（Invisible Index）**

```sql
-- 创建隐藏索引
CREATE INDEX idx_name ON users(name) INVISIBLE;

-- 隐藏索引对优化器不可见，但实际存在
SELECT * FROM users WHERE name = 'Alice';
-- 不会使用 idx_name

-- 用途：安全删除索引
-- 1. 先设为隐藏，观察是否有问题
ALTER TABLE users ALTER INDEX idx_name INVISIBLE;
-- 2. 确认无问题后再删除
DROP INDEX idx_name ON users;

-- 强制使用隐藏索引（测试用）
SET SESSION optimizer_switch='use_invisible_indexes=on';
```

**3. 函数索引（Functional Index）**

```sql
-- MySQL 5.7：对函数无法使用索引
SELECT * FROM users WHERE YEAR(create_time) = 2024;  -- 索引失效

-- MySQL 8.0：创建函数索引
CREATE INDEX idx_year ON users((YEAR(create_time)));

-- 或者使用计算列
ALTER TABLE users ADD COLUMN create_year INT 
    GENERATED ALWAYS AS (YEAR(create_time)) STORED;
CREATE INDEX idx_create_year ON users(create_year);

-- 查询现在可以使用索引
SELECT * FROM users WHERE YEAR(create_time) = 2024;  -- 索引生效
```

**4. 通用表表达式优化索引使用**

```sql
-- CTE 中可以使用索引
WITH user_stats AS (
    SELECT user_id, COUNT(*) as order_count
    FROM orders
    GROUP BY user_id
)
SELECT u.name, s.order_count
FROM users u
JOIN user_stats s ON u.id = s.user_id;
```

**Explain 分析函数索引：**

```sql
EXPLAIN SELECT * FROM users WHERE YEAR(create_time) = 2024\G

-- MySQL 8.0 输出：
key: idx_year
Extra: Using index condition
```

**追问：降序索引有什么实际用途？**

**追问答案：**
主要用于优化多列排序查询，当排序方向与索引方向一致时，可以避免额外的排序操作。

```sql
-- 场景：按创建时间降序、更新时间升序查询
SELECT * FROM posts ORDER BY create_time DESC, update_time ASC LIMIT 10;

-- 传统索引：需要 filesort
-- MySQL 8.0 降序索引
CREATE INDEX idx_time ON posts(create_time DESC, update_time ASC);
-- 直接利用索引顺序，无需排序
```

---

## 性能优化实战篇

### 1. EXPLAIN 的 rows 和 filtered 字段怎么分析？

**答案：**

EXPLAIN 是分析 SQL 执行计划的核心工具，rows 和 filtered 是重要指标：

**字段含义：**

| 字段 | 含义 | 说明 |
|-----|------|------|
| rows | 预估扫描行数 | MySQL 估计需要扫描的行数 |
| filtered | 过滤百分比 | WHERE 条件过滤后剩余的百分比 |

**完整 EXPLAIN 示例：**

```sql
EXPLAIN SELECT * FROM users WHERE name = 'Alice' AND age = 18\G

-- 输出：
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: users
   partitions: NULL
         type: ref
possible_keys: idx_name_age
          key: idx_name_age
      key_len: 158
          ref: const,const
         rows: 10        -- 预估扫描 10 行
     filtered: 100.00    -- 100% 过滤（无额外过滤）
        Extra: Using index condition
```

**rows 分析：**

```sql
-- rows 越小越好，但要注意：
-- 1. rows 是估计值，可能不准确
-- 2. 小表全表扫描的 rows 也小

-- 查看实际扫描行数（MySQL 8.0）
EXPLAIN ANALYZE SELECT * FROM users WHERE name = 'Alice';

-- 输出包含实际执行信息：
-- -> Index lookup on users using idx_name (name='Alice')  (cost=1.25 rows=1) (actual time=0.02..0.03 rows=1 loops=1)
```

**filtered 分析：**

```sql
-- filtered = 100%：所有扫描的行都满足条件，效率高
-- filtered 较低：扫描了很多无用的行，需要优化

-- 示例：filtered 较低的情况
EXPLAIN SELECT * FROM users WHERE name = 'Alice' AND city = 'Beijing'\G

-- 假设只有 idx_name 索引
-- rows: 1000  (name = 'Alice' 有 1000 条)
-- filtered: 10.00  (只有 10% 满足 city = 'Beijing')
-- 实际返回：100 条，但扫描了 1000 条

-- 优化：建立联合索引
CREATE INDEX idx_name_city ON users(name, city);

-- 优化后
-- rows: 100
-- filtered: 100.00
```

**计算公式：**

```
实际需要处理的行数 ≈ rows × (filtered / 100)

示例：
rows = 1000, filtered = 10%
实际处理行数 ≈ 1000 × 0.1 = 100 行
```

**分析技巧：**

```sql
-- 1. 关注 rows × filtered 的乘积
-- 乘积越大，说明扫描的无用行越多

-- 2. 多表 JOIN 时，注意驱动表的选择
EXPLAIN SELECT * FROM users u JOIN orders o ON u.id = o.user_id WHERE u.name = 'Alice'\G

-- 如果 users 表 rows 小，应作为驱动表
-- 查看 Extra 是否有 "Using join buffer"（说明使用了连接缓冲）

-- 3. 使用 EXPLAIN FORMAT=JSON 获取详细信息
EXPLAIN FORMAT=JSON SELECT * FROM users WHERE name = 'Alice'\G
```

**追问：rows 估计不准确怎么办？**

**追问答案：**

```sql
-- 1. 更新统计信息
ANALYZE TABLE users;

-- 2. 查看 MySQL 8.0 的 EXPLAIN ANALYZE（实际执行）
EXPLAIN ANALYZE SELECT * FROM users WHERE name = 'Alice';

-- 3. 使用 optimizer_trace 查看优化器决策过程
SET optimizer_trace='enabled=on';
SELECT * FROM users WHERE name = 'Alice';
SELECT * FROM information_schema.OPTIMIZER_TRACE\G
SET optimizer_trace='enabled=off';
```

---

### 2. Extra 字段出现的各种值代表什么？

**答案：**

Extra 字段包含了执行计划的额外信息，是优化 SQL 的重要参考：

**常见值及含义：**

| Extra 值 | 含义 | 优化建议 |
|---------|------|---------|
| Using index | 覆盖索引，无需回表 | ✅ 好 |
| Using where | Server 层过滤 | 检查是否可用索引优化 |
| Using index condition | 索引下推 | ✅ 好 |
| Using temporary | 使用临时表 | ⚠️ 可能需要优化 |
| Using filesort | 文件排序 | ⚠️ 考虑添加索引 |
| Using join buffer | 使用连接缓冲 | ⚠️ 考虑添加索引 |
| Using intersect | 索引合并（交集） | 可优化为联合索引 |
| Using union | 索引合并（并集） | 可优化为联合索引 |
| Using sort_union | 索引合并（排序并集） | 可优化为联合索引 |

**详细示例：**

**1. Using index（覆盖索引）**

```sql
-- 好情况：查询列都在索引中
EXPLAIN SELECT name, age FROM users WHERE name = 'Alice';
-- Extra: Using index

-- 说明：直接从索引获取数据，无需回表
```

**2. Using filesort（文件排序）**

```sql
-- 需要排序但无法使用索引
EXPLAIN SELECT * FROM users ORDER BY create_time;
-- Extra: Using filesort

-- 优化：为排序列添加索引
CREATE INDEX idx_create_time ON users(create_time);

EXPLAIN SELECT * FROM users ORDER BY create_time;
-- Extra: NULL（使用了索引顺序）
```

**3. Using temporary（临时表）**

```sql
-- GROUP BY 使用临时表
EXPLAIN SELECT name, COUNT(*) FROM users GROUP BY name;
-- Extra: Using temporary; Using filesort

-- 优化：添加索引
CREATE INDEX idx_name ON users(name);

EXPLAIN SELECT name, COUNT(*) FROM users GROUP BY name;
-- Extra: Using index
```

**4. Using join buffer（连接缓冲）**

```sql
-- 被驱动表没有合适索引
EXPLAIN SELECT * FROM users u JOIN orders o ON u.name = o.customer_name;
-- Extra: Using join buffer (Block Nested Loop)

-- 优化：为连接条件添加索引
CREATE INDEX idx_customer ON orders(customer_name);
```

**5. Using intersect（索引合并）**

```sql
-- 多个索引条件取交集
-- 假设有 idx_name 和 idx_age 两个独立索引
EXPLAIN SELECT * FROM users WHERE name = 'Alice' AND age = 18;
-- Extra: Using intersect(idx_name, idx_age); Using where

-- 优化：改为联合索引
CREATE INDEX idx_name_age ON users(name, age);
```

**组合情况分析：**

```sql
-- 常见组合
EXPLAIN SELECT name, COUNT(*) FROM users WHERE age > 18 GROUP BY name;

-- 可能输出：
-- Extra: Using index condition; Using temporary; Using filesort
-- 分析：
-- 1. Using index condition: age 条件下推
-- 2. Using temporary: GROUP BY 需要临时表
-- 3. Using filesort: GROUP BY 需要排序

-- 优化：创建合适的联合索引
CREATE INDEX idx_age_name ON users(age, name);
```

**追问：Using filesort 一定是坏事吗？**

**追问答案：**
不一定。如果结果集很小，filesort 在内存中完成，性能影响不大。只有当排序数据量大时才需要优化。

```sql
-- 查看 sort_buffer_size（排序缓冲区大小）
SHOW VARIABLES LIKE 'sort_buffer_size';

-- 默认 256KB，足够处理小结果集
-- 大数据量排序时，会使用磁盘临时文件，性能下降
```

---

### 3. count(*)、count(1)、count(字段) 的区别？

**答案：**

三者性能和语义有细微差别：

**语义区别：**

| 函数 | 语义 | NULL 处理 |
|-----|------|----------|
| `COUNT(*)` | 统计行数 | 不忽略 NULL |
| `COUNT(1)` | 统计行数（每行 1 个） | 不忽略 NULL |
| `COUNT(字段)` | 统计字段非 NULL 值数 | 忽略 NULL |

**性能对比：**

```sql
-- MySQL 优化器会优化 COUNT(*) 和 COUNT(1)
-- 在 InnoDB 中，两者性能几乎相同

-- COUNT(*)：MySQL 推荐
SELECT COUNT(*) FROM users;
-- InnoDB 会选择最小的辅助索引来统计

-- COUNT(1)：与 COUNT(*) 相同
SELECT COUNT(1) FROM users;
-- 优化器会将其优化为 COUNT(*)

-- COUNT(字段)：可能更慢
SELECT COUNT(name) FROM users;
-- 需要读取字段值，判断是否为 NULL
-- 如果字段没有索引，需要全表扫描
```

**实际测试：**

```sql
-- 创建测试表
CREATE TABLE test_count (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    INDEX idx_age (age)
);

INSERT INTO test_count VALUES 
(1, 'Alice', 20),
(2, NULL, 25),    -- name 为 NULL
(3, 'Carol', NULL), -- age 为 NULL
(4, 'David', 30);

-- COUNT(*)
SELECT COUNT(*) FROM test_count;      -- 结果: 4

-- COUNT(1)  
SELECT COUNT(1) FROM test_count;      -- 结果: 4

-- COUNT(name)：忽略 NULL
SELECT COUNT(name) FROM test_count;   -- 结果: 3

-- COUNT(age)：忽略 NULL
SELECT COUNT(age) FROM test_count;    -- 结果: 3
```

**执行计划分析：**

```sql
-- COUNT(*) 使用最小索引
EXPLAIN SELECT COUNT(*) FROM test_count;
-- key: idx_age（选择最小的辅助索引）

-- COUNT(字段)
EXPLAIN SELECT COUNT(name) FROM test_count;
-- key: NULL（全表扫描，因为 name 没有索引）

-- 添加索引后
CREATE INDEX idx_name ON test_count(name);
EXPLAIN SELECT COUNT(name) FROM test_count;
-- key: idx_name（使用索引）
```

**性能优化建议：**

```sql
-- 1. 优先使用 COUNT(*)
SELECT COUNT(*) FROM users;

-- 2. 如果只需要大致数量，可以使用
SHOW TABLE STATUS LIKE 'users';
-- Rows 字段是估计值

-- 3. 大表计数优化：维护计数表
CREATE TABLE table_counts (
    table_name VARCHAR(50) PRIMARY KEY,
    row_count BIGINT
);

-- 通过触发器或应用层维护计数
```

**追问：为什么 InnoDB 不像 MyISAM 那样存储精确的行数？**

**追问答案：**
因为 InnoDB 支持 MVCC，同一时刻不同事务可能看到不同数量的行。

```sql
-- 事务A
BEGIN;
SELECT COUNT(*) FROM users;  -- 看到某些行

-- 事务B
DELETE FROM users WHERE id = 1; COMMIT;

-- 事务A 再次查询
SELECT COUNT(*) FROM users;  -- 仍然看到被删除的行（MVCC）

-- 所以 InnoDB 无法存储一个"精确"的全局行数
```

---

### 4. 大表 DDL 怎么做才不影响业务？

**答案：**

大表 DDL（如添加索引、修改列）可能长时间锁表，影响业务。有多种解决方案：

**传统 DDL 的问题：**

```sql
-- 传统方式：直接执行 DDL
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 问题：
-- 1. 全表拷贝，数据量大时耗时长
-- 2. 锁表，阻塞读写操作
-- 3. 可能造成主从延迟
```

**方案一：Online DDL（MySQL 5.6+）**

```sql
-- Online DDL：支持在线修改，不阻塞读写
ALTER TABLE users 
ADD COLUMN phone VARCHAR(20), 
ALGORITHM=INPLACE, LOCK=NONE;

-- ALGORITHM 选项：
-- COPY: 创建临时表拷贝数据（会锁表）
-- INPLACE: 原地修改（推荐）

-- LOCK 选项：
-- EXCLUSIVE: 排他锁（阻塞读写）
-- SHARED: 共享锁（允许读，阻塞写）
-- NONE: 无锁（允许读写）
```

**支持的 Online DDL 操作：**

```sql
-- ✅ 支持 INPLACE（即时完成或快速完成）
ALTER TABLE users ADD INDEX idx_name (name);
ALTER TABLE users DROP INDEX idx_name;
ALTER TABLE users ADD COLUMN phone VARCHAR(20);  -- 添加到最后
ALTER TABLE users DROP COLUMN phone;

-- ⚠️ 需要 COPY（较慢）
ALTER TABLE users ADD COLUMN phone VARCHAR(20) FIRST;  -- 添加到开头
ALTER TABLE users MODIFY COLUMN phone INT;  -- 修改类型
ALTER TABLE users CHANGE COLUMN phone mobile VARCHAR(20);  -- 重命名
```

**方案二：pt-online-schema-change（Percona Toolkit）**

```bash
# 安装 Percona Toolkit
# 使用 pt-online-schema-change 执行 DDL

pt-online-schema-change \
  --alter "ADD COLUMN phone VARCHAR(20)" \
  --execute D=database,t=users \
  --max-load=Threads_running=50 \
  --critical-load=Threads_running=100

# 工作原理：
# 1. 创建新表（修改后的结构）
# 2. 创建触发器同步增量数据
# 3. 分批拷贝旧表数据
# 4. 交换表名
# 5. 删除旧表
```

**方案三：gh-ost（GitHub 开源）**

```bash
# gh-ost: 无触发器的在线 DDL 工具

gh-ost \
  --max-load=Threads_running=25 \
  --critical-load=Threads_running=1000 \
  --chunk-size=1000 \
  --throttle-control-replicas="..." \
  --database=database \
  --table=users \
  --alter="ADD COLUMN phone VARCHAR(20)" \
  --allow-on-master \
  --execute

# 优势：
# 1. 无触发器，性能影响小
# 2. 可暂停、可回滚
# 3. 支持主从切换
```

**方案四：分时段执行**

```sql
-- 在业务低峰期执行
-- 1. 先查看表大小
SELECT 
    table_name,
    ROUND(data_length/1024/1024, 2) AS 'Data(MB)',
    ROUND(index_length/1024/1024, 2) AS 'Index(MB)',
    table_rows
FROM information_schema.tables 
WHERE table_schema = 'your_database';

-- 2. 评估执行时间（测试环境）
-- 3. 在低峰期执行
```

**对比表：**

| 方案 | 阻塞读写 | 性能影响 | 复杂度 |
|-----|---------|---------|-------|
| Online DDL | 最小 | 低 | 低 |
| pt-osc | 无 | 中（触发器开销） | 中 |
| gh-ost | 无 | 低 | 高 |

**追问：Online DDL 的 ALGORITHM=INPLACE 真的不锁表吗？**

**追问答案：**
不完全准确。INPLACE 分两个阶段：
1. **准备阶段**：短暂锁表（通常毫秒级）
2. **执行阶段**：不锁表，允许读写
3. **完成阶段**：短暂锁表

```sql
-- 对于添加索引，几乎不阻塞
ALTER TABLE users ADD INDEX idx_name (name), ALGORITHM=INPLACE;

-- 对于某些操作（如添加列），执行阶段不阻塞，但首尾有短暂锁
-- 通常可以接受
```

---

### 5. 深分页问题怎么优化？

**答案：**

深分页（如 `LIMIT 1000000, 10`）性能很差，因为需要扫描前 100 万条记录。

**问题分析：**

```sql
-- 深分页查询
SELECT * FROM users ORDER BY id LIMIT 1000000, 10;

-- 执行过程：
-- 1. 扫描前 1000010 条记录
-- 2. 丢弃前 1000000 条
-- 3. 返回后 10 条

-- 性能问题：大量无用的扫描和丢弃
```

**优化方案一：子查询优化**

```sql
-- 原始查询
SELECT * FROM users ORDER BY id LIMIT 1000000, 10;

-- 优化：使用子查询先获取 ID
SELECT * FROM users 
WHERE id >= (
    SELECT id FROM users ORDER BY id LIMIT 1000000, 1
) 
ORDER BY id LIMIT 10;

-- 原理：
-- 1. 子查询只扫描索引，不需要回表
-- 2. 主查询从指定位置开始，只扫描 10 条
```

**优化方案二：JOIN 优化**

```sql
SELECT u.* FROM users u
INNER JOIN (
    SELECT id FROM users ORDER BY id LIMIT 1000000, 10
) t ON u.id = t.id;

-- 原理：子查询使用覆盖索引，只扫描索引
-- JOIN 操作直接定位到目标行
```

**优化方案三：游标分页（推荐）**

```sql
-- 传统分页
SELECT * FROM users ORDER BY id LIMIT 0, 10;
SELECT * FROM users ORDER BY id LIMIT 10, 10;
SELECT * FROM users ORDER BY id LIMIT 20, 10;

-- 游标分页：记录上一页最后一条记录的 ID
-- 第一页
SELECT * FROM users ORDER BY id LIMIT 10;

-- 假设第一页最后一条 id = 10
-- 第二页
SELECT * FROM users WHERE id > 10 ORDER BY id LIMIT 10;

-- 第三页（假设第二页最后 id = 20）
SELECT * FROM users WHERE id > 20 ORDER BY id LIMIT 10;

-- 优势：无论翻到第几页，都只扫描需要的记录
```

**优化方案四：业务限制**

```sql
-- 限制最大页数，不让用户翻到太深的页
-- 例如：只允许查看前 100 页

-- 或者：搜索引擎式，只显示前 N 条结果
SELECT * FROM users WHERE name LIKE '%keyword%' LIMIT 100;
-- 不告诉用户总数，只显示"有更多结果"
```

**性能对比：**

```sql
-- 测试表：100 万条记录

-- 原始查询
SELECT * FROM users ORDER BY id LIMIT 900000, 10;
-- 执行时间：约 2 秒

-- 子查询优化
SELECT * FROM users 
WHERE id >= (SELECT id FROM users ORDER BY id LIMIT 900000, 1)
ORDER BY id LIMIT 10;
-- 执行时间：约 0.3 秒

-- 游标分页
SELECT * FROM users WHERE id > 900000 ORDER BY id LIMIT 10;
-- 执行时间：约 0.01 秒
```

**追问：游标分页有什么缺点？**

**追问答案：**
1. **不能跳页**：只能顺序翻页
2. **不能倒序翻页**：需要额外实现
3. **数据变化影响**：如果有新数据插入，可能看到重复数据

```sql
-- 解决重复数据问题：使用唯一且有序的字段
-- 或者记录时间戳作为辅助判断
SELECT * FROM users 
WHERE id > ? AND create_time >= ? 
ORDER BY id LIMIT 10;
```

---

### 6. 线上 SQL 慢怎么排查？

**答案：**

完整的 SQL 慢查询排查流程：

**步骤1：开启慢查询日志**

```sql
-- 查看慢查询配置
SHOW VARIABLES LIKE '%slow_query%';
SHOW VARIABLES LIKE 'long_query_time';

-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1;  -- 超过 1 秒记录
SET GLOBAL log_queries_not_using_indexes = ON;  -- 记录没走索引的查询

-- 慢查询日志位置
SHOW VARIABLES LIKE 'slow_query_log_file';
```

**步骤2：分析慢查询日志**

```bash
# 使用 mysqldumpslow 分析
mysqldumpslow -s t -t 10 /var/log/mysql/slow.log

# 参数说明：
# -s t: 按查询时间排序
# -t 10: 显示前 10 条
# -s c: 按查询次数排序
# -s l: 按锁定时间排序
# -s r: 按返回记录数排序

# 使用 pt-query-digest（更强大）
pt-query-digest /var/log/mysql/slow.log > slow_report.txt
```

**步骤3：定位具体 SQL**

```sql
-- 查看 information_schema 中的慢查询
SELECT * FROM sys.statements_with_runtimes_in_95th_percentile\G

-- 查看当前正在执行的查询
SELECT * FROM information_schema.processlist 
WHERE command != 'Sleep' 
ORDER BY time DESC\G

-- 查看锁等待情况
SELECT 
    r.trx_id waiting_trx_id,
    r.trx_mysql_thread_id waiting_thread,
    r.trx_query waiting_query,
    b.trx_id blocking_trx_id,
    b.trx_mysql_thread_id blocking_thread,
    b.trx_query blocking_query
FROM information_schema.innodb_lock_waits w
JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;
```

**步骤4：分析执行计划**

```sql
-- 获取 SQL 执行计划
EXPLAIN SELECT * FROM users WHERE name = 'Alice'\G

-- 关键指标：
-- type: ALL → 全表扫描，需要优化
-- rows: 扫描行数，越小越好
-- Extra: Using filesort, Using temporary → 可能需要优化

-- MySQL 8.0：使用 EXPLAIN ANALYZE
EXPLAIN ANALYZE SELECT * FROM users WHERE name = 'Alice'\G
```

**步骤5：优化 SQL**

```sql
-- 常见优化场景：

-- 1. 索引缺失
-- 慢 SQL：
SELECT * FROM users WHERE name = 'Alice';
-- 优化：
CREATE INDEX idx_name ON users(name);

-- 2. 索引失效
-- 慢 SQL：
SELECT * FROM users WHERE YEAR(create_time) = 2024;
-- 优化：
SELECT * FROM users WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01';

-- 3. 深分页
-- 慢 SQL：
SELECT * FROM users LIMIT 1000000, 10;
-- 优化：
SELECT * FROM users WHERE id > 1000000 ORDER BY id LIMIT 10;

-- 4. 大量 JOIN
-- 慢 SQL：多表 JOIN 没有索引
-- 优化：为连接字段添加索引

-- 5. SELECT *
-- 慢 SQL：
SELECT * FROM users WHERE name = 'Alice';
-- 优化：只查需要的列
SELECT id, name FROM users WHERE name = 'Alice';
```

**排查流程图：**

```
┌─────────────────────────────────────────────────────────────┐
│                    SQL 慢查询排查流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 开启慢查询日志                                           │
│     ↓                                                       │
│  2. 分析日志定位慢 SQL                                       │
│     ↓                                                       │
│  3. EXPLAIN 分析执行计划                                     │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │ 问题类型：                            │                  │
│  │ • 全表扫描 → 添加索引                 │                  │
│  │ • 索引失效 → 修改 SQL                 │                  │
│  │ • 深分页 → 游标分页                   │                  │
│  │ • 大结果集 → 减少返回字段             │                  │
│  │ • 复杂 JOIN → 分解查询                │                  │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  4. 优化并验证                                               │
│     ↓                                                       │
│  5. 上线监控                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**追问：如何预防慢 SQL 上线？**

**追问答案：**

```sql
-- 1. 开发环境强制 EXPLAIN
-- CI/CD 流程中自动检查

-- 2. 设置 SQL 审核规则
-- 使用工具如 SOAR、SQL Advisor

-- 3. 限制大查询
SET GLOBAL max_execution_time = 30000;  -- 30 秒超时

-- 4. 监控告警
-- 配置 Prometheus + Grafana 监控慢查询数量
```

---

### 7. 如何优化大表的 COUNT 查询？

**答案：**

大表 COUNT 查询性能差是常见问题，有多种优化方案：

**问题分析：**

```sql
-- 精确计数：需要全表扫描
SELECT COUNT(*) FROM users;  -- 百万级数据可能需要几秒
```

**方案一：使用估计值**

```sql
-- 使用 information_schema 的估计值
SELECT table_rows 
FROM information_schema.tables 
WHERE table_schema = 'database' AND table_name = 'users';

-- 使用 SHOW TABLE STATUS
SHOW TABLE STATUS LIKE 'users';
-- Rows 字段是估计值，瞬间返回
```

**方案二：维护计数表**

```sql
-- 创建计数表
CREATE TABLE table_counts (
    table_name VARCHAR(50) PRIMARY KEY,
    row_count BIGINT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 初始化
INSERT INTO table_counts (table_name, row_count) 
SELECT 'users', COUNT(*) FROM users;

-- 方式1：触发器维护
CREATE TRIGGER after_user_insert 
AFTER INSERT ON users FOR EACH ROW
BEGIN
    UPDATE table_counts SET row_count = row_count + 1 WHERE table_name = 'users';
END;

CREATE TRIGGER after_user_delete 
AFTER DELETE ON users FOR EACH ROW
BEGIN
    UPDATE table_counts SET row_count = row_count - 1 WHERE table_name = 'users';
END;

-- 方式2：应用层维护
-- INSERT 时 +1，DELETE 时 -1
BEGIN;
INSERT INTO users (name) VALUES ('test');
UPDATE table_counts SET row_count = row_count + 1 WHERE table_name = 'users';
COMMIT;

-- 查询：瞬间返回
SELECT row_count FROM table_counts WHERE table_name = 'users';
```

**方案三：Redis 缓存计数**

```python
# Python 示例
import redis
import mysql.connector

r = redis.Redis(host='localhost', port=6379)

def get_user_count():
    # 先查缓存
    count = r.get('user_count')
    if count:
        return int(count)
    
    # 缓存没有，查数据库并缓存
    conn = mysql.connector.connect(...)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    # 缓存结果
    r.set('user_count', count, ex=3600)  # 1小时过期
    return count

def increment_user_count():
    r.incr('user_count')

def decrement_user_count():
    r.decr('user_count')
```

**方案四：分区计数**

```sql
-- 如果表有分区（如按日期分区），可以并行计数
SELECT COUNT(*) FROM users PARTITION (p202401);
SELECT COUNT(*) FROM users PARTITION (p202402);
-- 应用层汇总

-- 或者分批计数
SELECT COUNT(*) FROM users WHERE id BETWEEN 1 AND 1000000;
SELECT COUNT(*) FROM users WHERE id BETWEEN 1000001 AND 2000000;
-- 并行执行
```

**条件计数优化：**

```sql
-- 慢查询：需要全表扫描
SELECT COUNT(*) FROM users WHERE status = 'active';

-- 优化1：添加索引
CREATE INDEX idx_status ON users(status);
-- 仍然需要扫描索引，但比全表扫描快

-- 优化2：使用覆盖索引
CREATE INDEX idx_status ON users(status);
EXPLAIN SELECT COUNT(*) FROM users WHERE status = 'active';
-- Extra: Using index

-- 优化3：汇总表（如果按条件分组统计频繁）
CREATE TABLE user_stats (
    status VARCHAR(20) PRIMARY KEY,
    count INT
);

-- 定期更新或使用触发器
INSERT INTO user_stats (status, count)
SELECT status, COUNT(*) FROM users GROUP BY status
ON DUPLICATE KEY UPDATE count = VALUES(count);
```

**追问：COUNT 查询会不会锁表？**

**追问答案：**
不会。InnoDB 的 COUNT 使用 MVCC，不会加锁，不影响其他事务的读写。

```sql
-- 事务A
BEGIN;
SELECT COUNT(*) FROM users;  -- 不加锁，使用 MVCC

-- 事务B 可以同时进行写操作
INSERT INTO users (name) VALUES ('test');

-- 事务A 的 COUNT 看不到事务B 的修改（MVCC）
```

---

### 8. 如何分析和优化 ORDER BY 查询？

**答案：**

ORDER BY 优化是 SQL 调优的重要部分：

**问题场景：**

```sql
-- 大表排序查询
SELECT * FROM users ORDER BY create_time DESC LIMIT 10;
-- 如果 create_time 没有索引，需要 filesort
```

**分析执行计划：**

```sql
EXPLAIN SELECT * FROM users ORDER BY create_time DESC LIMIT 10\G

-- 关注 Extra 字段：
-- 1. "Using filesort" → 需要额外排序
-- 2. NULL → 使用了索引顺序，无需额外排序
```

**优化方案一：添加索引**

```sql
-- 为排序字段添加索引
CREATE INDEX idx_create_time ON users(create_time);

-- 再次查看执行计划
EXPLAIN SELECT * FROM users ORDER BY create_time DESC LIMIT 10\G
-- Extra: NULL（或没有 filesort）
-- 使用索引的有序性，避免排序
```

**优化方案二：联合索引优化**

```sql
-- 复杂查询：WHERE + ORDER BY
SELECT * FROM users WHERE status = 'active' ORDER BY create_time DESC LIMIT 10;

-- 单独的索引可能不够
CREATE INDEX idx_status ON users(status);  -- 只能优化 WHERE
CREATE INDEX idx_create_time ON users(create_time);  -- 只能优化 ORDER BY

-- 联合索引：同时优化 WHERE 和 ORDER BY
CREATE INDEX idx_status_time ON users(status, create_time);

-- 执行计划
EXPLAIN SELECT * FROM users WHERE status = 'active' ORDER BY create_time DESC LIMIT 10\G
-- key: idx_status_time
-- Extra: Using index condition
```

**优化方案三：覆盖索引**

```sql
-- 只查询索引列
SELECT id, create_time FROM users ORDER BY create_time DESC LIMIT 10;

-- 如果有 idx_create_time 索引
-- Extra: Using index（覆盖索引，无需回表）
```

**优化方案四：减少排序数据量**

```sql
-- 原始查询：排序大量数据
SELECT * FROM users WHERE status = 'active' ORDER BY create_time;

-- 优化1：添加 LIMIT
SELECT * FROM users WHERE status = 'active' ORDER BY create_time LIMIT 100;

-- 优化2：子查询先筛选 ID
SELECT u.* FROM users u
INNER JOIN (
    SELECT id FROM users WHERE status = 'active' ORDER BY create_time LIMIT 100
) t ON u.id = t.id;
```

**Filesort 优化参数：**

```sql
-- 查看排序缓冲区大小
SHOW VARIABLES LIKE 'sort_buffer_size';

-- 增大排序缓冲区（临时优化）
SET SESSION sort_buffer_size = 4*1024*1024;  -- 4MB

-- 查看排序算法
SHOW VARIABLES LIKE 'max_length_for_sort_data';

-- 如果排序数据量超过 max_length_for_sort_data
-- MySQL 会使用双路排序（两次 I/O）
-- 否则使用单路排序（一次 I/O，但内存占用大）
```

**排序算法对比：**

```
单路排序（Sort Buffer 足够）：
┌─────────────────────────────────────┐
│  1. 读取所有需要的列到 Sort Buffer   │
│  2. 在内存中排序                     │
│  3. 返回结果                         │
└─────────────────────────────────────┘
优点：I/O 少
缺点：内存占用大

双路排序（Sort Buffer 不足）：
┌─────────────────────────────────────┐
│  1. 读取排序字段 + 主键到 Sort Buffer│
│  2. 排序后得到有序的主键列表         │
│  3. 根据主键回表获取完整数据         │
└─────────────────────────────────────┘
优点：内存占用小
缺点：I/O 多
```

**追问：什么时候 ORDER BY 无法使用索引？**

**追问答案：**

```sql
-- 1. 多个字段排序方向不一致
SELECT * FROM users ORDER BY create_time DESC, update_time ASC;
-- 需要降序索引（MySQL 8.0）

-- 2. 排序字段在不同的索引
SELECT * FROM users ORDER BY name, create_time;
-- 如果 name 和 create_time 是独立的索引

-- 3. 排序字段使用了函数
SELECT * FROM users ORDER BY YEAR(create_time);

-- 4. JOIN 查询，排序字段不在驱动表
-- 5. 排序字段类型不一致
```

---

## 架构设计篇

### 1. 分库分表后主键 ID 怎么生成？

**答案：**

分库分表后，传统的自增主键会产生 ID 冲突问题，需要使用分布式 ID 生成方案：

**方案一：UUID**

```sql
-- 优点：简单，无需额外组件
-- 缺点：无序、长、索引性能差

SELECT UUID();  -- '550e8400-e29b-41d4-a716-446655440000'

-- 问题：
-- 1. 36 字符，存储空间大
-- 2. 无序，导致索引页分裂
-- 3. 不适合做聚簇索引主键
```

**方案二：雪花算法（Snowflake）**

```
Snowflake ID 结构（64 位）：
┌─────────────────────────────────────────────────────────────┐
│  1位  │     41位时间戳      │ 10位机器ID  │   12位序列号    │
│ 符号位 │ (毫秒级时间戳)       │ (5数据中心+5机器)│  (同一毫秒内)  │
└─────────────────────────────────────────────────────────────┘

特点：
- 18 位长整型数字
- 趋势递增（有利于索引）
- 包含时间信息（可反解析）
- 分布式唯一
```

```java
// Java 实现示例
public class SnowflakeIdGenerator {
    private final long twepoch = 1288834974657L;
    private final long workerIdBits = 5L;
    private final long datacenterIdBits = 5L;
    private final long maxWorkerId = -1L ^ (-1L << workerIdBits);
    private final long maxDatacenterId = -1L ^ (-1L << datacenterIdBits);
    private final long sequenceBits = 12L;
    
    private final long workerIdShift = sequenceBits;
    private final long datacenterIdShift = sequenceBits + workerIdBits;
    private final long timestampLeftShift = sequenceBits + workerIdBits + datacenterIdBits;
    private final long sequenceMask = -1L ^ (-1L << sequenceBits);
    
    private long workerId;
    private long datacenterId;
    private long sequence = 0L;
    private long lastTimestamp = -1L;
    
    public synchronized long nextId() {
        long timestamp = System.currentTimeMillis();
        
        if (timestamp < lastTimestamp) {
            throw new RuntimeException("时钟回拨");
        }
        
        if (lastTimestamp == timestamp) {
            sequence = (sequence + 1) & sequenceMask;
            if (sequence == 0) {
                timestamp = tilNextMillis(lastTimestamp);
            }
        } else {
            sequence = 0L;
        }
        
        lastTimestamp = timestamp;
        
        return ((timestamp - twepoch) << timestampLeftShift)
                | (datacenterId << datacenterIdShift)
                | (workerId << workerIdShift)
                | sequence;
    }
}
```

**方案三：数据库号段模式**

```sql
-- 创建 ID 生成表
CREATE TABLE id_generator (
    biz_type VARCHAR(64) PRIMARY KEY,
    max_id BIGINT NOT NULL,
    step INT NOT NULL,
    version INT NOT NULL DEFAULT 0
);

-- 初始化
INSERT INTO id_generator (biz_type, max_id, step, version) 
VALUES ('order', 0, 1000, 0);

-- 获取 ID 号段（乐观锁）
UPDATE id_generator 
SET max_id = max_id + step, version = version + 1
WHERE biz_type = 'order' AND version = ?;

-- 应用端缓存号段，用完再取
-- 例如：获取 1-1000，用完再获取 1001-2000
```

**方案四：Redis 自增**

```python
# Redis 生成 ID
import redis

r = redis.Redis(host='localhost', port=6379)

def generate_id(key='order_id'):
    # 使用 INCR 原子递增
    return r.incr(key)

# 或使用时间戳 + 序列号
def generate_order_id():
    timestamp = int(time.time() * 1000)
    seq = r.incr(f'order_seq:{timestamp}')
    return f'{timestamp}{seq:04d}'
```

**方案对比：**

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| UUID | 简单、无依赖 | 无序、长 | 小规模应用 |
| Snowflake | 有序、高性能 | 时钟回拨问题 | 大多数场景 |
| 号段模式 | 高可用、可读 | 依赖数据库 | 中等规模 |
| Redis | 简单、原子性 | 依赖 Redis | 已有 Redis 场景 |

**追问：Snowflake 时钟回拨怎么解决？**

**追问答案：**

```java
// 时钟回拨解决方案
public synchronized long nextId() {
    long timestamp = System.currentTimeMillis();
    
    if (timestamp < lastTimestamp) {
        long offset = lastTimestamp - timestamp;
        
        // 小范围回拨，等待
        if (offset <= 5) {
            try {
                wait(offset << 1);
                timestamp = System.currentTimeMillis();
                if (timestamp < lastTimestamp) {
                    throw new RuntimeException("时钟回拨");
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        } else {
            // 大范围回拨，拒绝服务或使用备用方案
            throw new RuntimeException("时钟回拨超过阈值");
        }
    }
    
    // ... 正常生成逻辑
}
```

---

### 2. 分库分表后的分页查询怎么处理？

**答案：**

分库分表后，跨库分页是一个复杂问题，需要特殊处理：

**问题场景：**

```sql
-- 假设订单表分到 4 个库
-- 每个库查询 LIMIT 0, 10，合并后结果不准确

-- 错误方式：
-- 库1: SELECT * FROM orders LIMIT 0, 10
-- 库2: SELECT * FROM orders LIMIT 0, 10
-- 库3: SELECT * FROM orders LIMIT 0, 10
-- 库4: SELECT * FROM orders LIMIT 0, 10
-- 合并后排序取前 10 → 错误！
```

**方案一：全局排序合并**

```sql
-- 查询第 N 页，每页 M 条

-- 1. 每个库查询前 N*M 条数据
SELECT * FROM orders ORDER BY create_time LIMIT (N * M);

-- 2. 应用层合并所有结果
-- 3. 全局排序
-- 4. 取第 (N-1)*M 到 N*M 条

-- 示例：第 2 页，每页 10 条
-- 库1: SELECT * FROM orders ORDER BY create_time LIMIT 20
-- 库2: SELECT * FROM orders ORDER BY create_time LIMIT 20
-- 库3: SELECT * FROM orders ORDER BY create_time LIMIT 20
-- 库4: SELECT * FROM orders ORDER BY create_time LIMIT 20
-- 合并 80 条 → 排序 → 取第 11-20 条
```

**方案二：游标分页（推荐）**

```sql
-- 记录上一页最后的值，避免全局排序

-- 第一页
SELECT * FROM orders ORDER BY create_time, id LIMIT 10;
-- 记录最后一条的 (create_time, id)

-- 第二页：只查大于最后一条的记录
-- 库1: SELECT * FROM orders 
--       WHERE create_time > ? OR (create_time = ? AND id > ?)
--       ORDER BY create_time, id LIMIT 10
-- 库2: 同上
-- ...

-- 合并少量数据，排序后取前 10 条
```

**方案三：二次查询法**

```sql
-- 第一步：查询每个分片的第一页
-- 库1: SELECT * FROM orders ORDER BY create_time LIMIT 0, 10
-- 得到最大值 t1

-- 第二步：用最大值查询每个分片
-- 库1: SELECT * FROM orders WHERE create_time <= t1 ORDER BY create_time
-- 库2: SELECT * FROM orders WHERE create_time <= t1 ORDER BY create_time
-- ...

-- 第三步：合并所有结果，取前 10 条
```

**方案四：汇总表**

```sql
-- 创建汇总表
CREATE TABLE order_summary (
    id BIGINT PRIMARY KEY,
    order_no VARCHAR(50),
    user_id BIGINT,
    create_time DATETIME,
    db_index INT  -- 记录数据在哪个分片
);

-- 同步数据到汇总表（可以使用 binlog 同步）

-- 分页查询走汇总表
SELECT * FROM order_summary ORDER BY create_time LIMIT 10, 10;

-- 获取结果后，根据 db_index 到对应分片获取完整数据
```

**方案五：禁止深分页**

```sql
-- 限制最大页数，如只允许查看前 100 页
-- 搜索引擎模式：只显示前 N 条结果

-- 或者提供筛选条件，缩小数据范围
SELECT * FROM orders 
WHERE create_time BETWEEN '2024-01-01' AND '2024-01-31'
ORDER BY create_time LIMIT 0, 10;
```

**对比：**

| 方案 | 优点 | 缺点 |
|-----|------|------|
| 全局排序 | 简单 | 性能差，页越深越慢 |
| 游标分页 | 性能好 | 不能跳页 |
| 二次查询 | 相对准确 | 实现复杂 |
| 汇总表 | 性能好 | 额外存储、同步延迟 |
| 禁止深分页 | 简单 | 用户体验受限 |

**追问：Elasticsearch 能解决分库分表分页问题吗？**

**追问答案：**
可以。将数据同步到 ES，利用 ES 的分布式排序能力：

```java
// ES 分页查询
SearchRequest request = new SearchRequest("orders");
SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();
sourceBuilder.query(QueryBuilders.matchAllQuery());
sourceBuilder.sort("create_time", SortOrder.DESC);
sourceBuilder.from(100);
sourceBuilder.size(10);

// ES 内部会协调各分片，合并结果
// 但 deep paging 仍有性能问题
```

---

### 3. 主从延迟怎么解决？

**答案：**

主从延迟是 MySQL 主从架构的常见问题，需要多方面处理：

**延迟产生原因：**

```
┌─────────────────────────────────────────────────────────────┐
│                      主从延迟原因                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 主库并发写入，从库单线程回放（MySQL 5.6 之前）            │
│                                                             │
│  2. 大事务执行时间过长                                       │
│     DELETE FROM logs WHERE create_time < '2020-01-01'       │
│     → 删除百万条记录，binlog 很大                            │
│                                                             │
│  3. 从库硬件性能差于主库                                     │
│                                                             │
│  4. 网络延迟                                                 │
│                                                             │
│  5. 从库上有复杂查询，占用资源                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**监控延迟：**

```sql
-- 方式1：SHOW SLAVE STATUS
SHOW SLAVE STATUS\G

-- 关键指标：
-- Seconds_Behind_Master: 0  ← 延迟秒数
-- Relay_Master_Log_File 和 Read_Master_Log_Pos
-- Exec_Master_Log_Pos

-- 方式2：使用心跳表
-- 主库定时更新心跳表
UPDATE heartbeat SET ts = NOW() WHERE id = 1;

-- 从库读取心跳表，计算差值
SELECT TIMESTAMPDIFF(SECOND, ts, NOW()) FROM heartbeat;

-- 方式3：performance_schema（MySQL 5.7+）
SELECT * FROM performance_schema.replication_connection_status;
SELECT * FROM performance_schema.replication_applier_status_by_worker;
```

**解决方案：**

**1. 开启并行复制（MySQL 5.7+）**

```sql
-- 从库配置
-- 基于 COMMIT_ORDER 的并行复制（MySQL 5.7）
slave_parallel_type = LOGICAL_CLOCK
slave_parallel_workers = 8

-- 基于 WRITESET 的并行复制（MySQL 8.0）
binlog_transaction_dependency_tracking = WRITESET
transaction_write_set_extraction = XXHASH64
slave_parallel_type = LOGICAL_CLOCK
slave_parallel_workers = 8
```

**2. 拆分大事务**

```sql
-- 原始：大事务
DELETE FROM logs WHERE create_time < '2020-01-01';

-- 优化：分批删除
DELETE FROM logs WHERE create_time < '2020-01-01' LIMIT 1000;
-- 循环执行直到删除完成
```

**3. 读写分离策略**

```sql
-- 方式1：关键业务强制读主库
-- 下单后立即查询订单详情
SELECT /*+ MASTER */ * FROM orders WHERE id = 123;

-- 方式2：延迟检测
-- 读取时检查延迟，超过阈值读主库
if (slave_delay > 1s) {
    read_from_master();
} else {
    read_from_slave();
}

-- 方式3：异步确认
-- 写入后等待从库同步
// 写入主库
insert into orders ...
// 等待从库同步
wait_for_slave_sync(order_id);
// 读取从库
select from orders where id = order_id
```

**4. 引入缓存**

```java
// 写入流程
@Transactional
public void createOrder(Order order) {
    // 1. 写入主库
    orderMapper.insert(order);
    // 2. 写入缓存
    redis.set("order:" + order.getId(), order, 60);  // 缓存60秒
}

// 读取流程
public Order getOrder(Long orderId) {
    // 1. 先查缓存
    Order order = redis.get("order:" + orderId);
    if (order != null) {
        return order;
    }
    // 2. 查从库
    order = orderMapper.selectById(orderId);
    return order;
}
```

**5. 使用半同步复制**

```sql
-- 主库配置
plugin-load = "rpl_semi_sync_master=semisync_master.so"
rpl_semi_sync_master_enabled = 1
rpl_semi_sync_master_timeout = 1000  # 超时后降级为异步

-- 从库配置
plugin-load = "rpl_semi_sync_slave=semisync_slave.so"
rpl_semi_sync_slave_enabled = 1

-- 效果：主库写入后，至少等待一个从库确认收到 binlog
-- 延迟可控制在毫秒级
```

**追问：什么场景必须读主库？**

**追问答案：**

```sql
-- 1. 写后立即读（一致性要求高）
INSERT INTO orders ...;
SELECT * FROM orders WHERE id = LAST_INSERT_ID();  -- 必须读主库

-- 2. 金融类业务（余额、交易）
UPDATE accounts SET balance = balance - 100 WHERE user_id = 1;
SELECT balance FROM accounts WHERE user_id = 1;  -- 必须读主库

-- 3. 用户修改个人信息后立即查看
UPDATE users SET phone = '138...' WHERE id = 1;
SELECT * FROM users WHERE id = 1;  -- 读主库确保看到最新数据
```

---

### 4. 读写分离中间件的工作原理？

**答案：**

读写分离中间件负责将 SQL 路由到主库或从库，实现读写分离：

**架构图：**

```
┌─────────────────────────────────────────────────────────────┐
│                        应用层                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   读写分离中间件                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  SQL 解析   │→│  路由决策   │→│  连接池管理  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  主库   │   │ 从库1   │   │ 从库2   │
   │ (写)    │   │ (读)    │   │ (读)    │
   └─────────┘   └─────────┘   └─────────┘
```

**路由规则：**

```sql
-- 写操作 → 主库
INSERT INTO users ...
UPDATE users ...
DELETE FROM users ...

-- 读操作 → 从库
SELECT * FROM users ...

-- 特殊情况：
-- 1. 事务中的 SELECT → 主库
BEGIN;
SELECT * FROM users WHERE id = 1;  -- 走主库
UPDATE users SET name = '...' WHERE id = 1;
COMMIT;

-- 2. LAST_INSERT_ID() → 主库
SELECT LAST_INSERT_ID();  -- 走主库

-- 3. 用户指定走主库
SELECT /*+ MASTER */ * FROM users ...
```

**常用中间件：**

| 中间件 | 类型 | 特点 |
|-------|------|------|
| MySQL Router | 官方 | 轻量，功能简单 |
| ProxySQL | 代理 | 功能强大，支持查询缓存 |
| MyCat | 代理 | 国产，支持分库分表 |
| ShardingSphere | 代理/客户端 | Apache 项目，功能全面 |
| MaxScale | 代理 | MariaDB 出品 |

**ProxySQL 配置示例：**

```sql
-- 添加服务器
INSERT INTO mysql_servers (hostgroup_id, hostname, port) 
VALUES (10, '192.168.1.10', 3306);  -- 主库

INSERT INTO mysql_servers (hostgroup_id, hostname, port) 
VALUES (20, '192.168.1.11', 3306);  -- 从库1

INSERT INTO mysql_servers (hostgroup_id, hostname, port) 
VALUES (20, '192.168.1.12', 3306);  -- 从库2

-- 配置路由规则
INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup)
VALUES (1, '^SELECT', 20);  -- SELECT 走从库

INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup)
VALUES (2, '.*', 10);  -- 其他走主库

-- 加载配置
LOAD MYSQL SERVERS TO RUNTIME;
LOAD MYSQL QUERY RULES TO RUNTIME;
```

**ShardingSphere-JDBC 示例：**

```java
// 配置读写分离
@Configuration
public class ShardingJdbcConfig {
    
    @Bean
    public DataSource dataSource() {
        MasterSlaveRuleConfiguration masterSlaveRuleConfig = 
            new MasterSlaveRuleConfiguration(
                "ds_master_slave",
                "ds_master",
                Arrays.asList("ds_slave_0", "ds_slave_1")
            );
        
        return MasterSlaveDataSourceFactory.createDataSource(
            createDataSourceMap(),
            masterSlaveRuleConfig,
            new Properties()
        );
    }
    
    // 强制走主库
    public Order getOrderForceMaster(Long id) {
        HintManager.getInstance().setMasterRouteOnly();
        return orderMapper.selectById(id);
    }
}
```

**追问：中间件如何处理主从延迟？**

**追问答案：**

```sql
-- 1. 延迟检测
-- 中间件定期检测从库延迟，动态调整路由

-- 2. 主从一致性读
-- 同一会话中，写入后一定时间内的读走主库
-- ProxySQL 配置：
INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup, apply)
VALUES (100, '^SELECT', 10, 0)  -- 默认走主库（hostgroup 10）
ON DUPLICATE KEY UPDATE ...;

-- 3. 用户 Hint 强制路由
SELECT /*+ MASTER */ * FROM users WHERE id = 1;

-- 4. 延迟阈值控制
-- 超过阈值，停止向该从库发送查询
```

---

### 5. 项目中怎么做高可用设计？

**答案：**

MySQL 高可用设计需要考虑故障检测、自动切换、数据一致性等多方面：

**高可用架构演进：**

```
单机模式：
┌─────────┐
│ MySQL   │  ← 单点故障
└─────────┘

主从复制：
┌─────────┐     复制     ┌─────────┐
│ Master  │ ──────────→ │ Slave   │
└─────────┘             └─────────┘
     ↑
   手动切换

MHA/Orchestrator：
┌─────────┐     复制     ┌─────────┐
│ Master  │ ──────────→ │ Slave1  │
└─────────┘             └─────────┘
     │                        │
     │   管理节点             │
     │  ┌─────────┐          │
     └──│   MHA   │──────────┘
        └─────────┘
             ↑
         自动故障转移

MGR/InnoDB Cluster：
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node1   │ ←→ │ Node2   │ ←→ │ Node3   │
│ (Primary)│    │ (Secondary)│   │ (Secondary)│
└─────────┘     └─────────┘     └─────────┘
     ↑               ↑               ↑
     └───────────────┴───────────────┘
              组复制，自动选主
```

**方案一：MHA（Master High Availability）**

```
MHA 架构：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────────┐                                         │
│   │ MHA Manager  │  ← 监控、故障检测、故障转移               │
│   └──────┬───────┘                                         │
│          │                                                 │
│   ┌──────┴──────────────────────────┐                      │
│   │                                 │                      │
│   ▼                                 ▼                      │
│ ┌─────────┐    复制    ┌─────────┐ ┌─────────┐            │
│ │ Master  │ ─────────→ │ Slave1  │ │ Slave2  │            │
│ │(写入点) │            │(候选主) │ │(候选主) │            │
│ └─────────┘            └─────────┘ └─────────┘            │
│                                                             │
│ 故障转移流程：                                               │
│ 1. 检测 Master 故障                                         │
│ 2. 从 Slave 中选择数据最新的作为新 Master                    │
│ 3. 其他 Slave 指向新 Master                                 │
│ 4. VIP 漂移到新 Master                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**MHA 配置示例：**

```ini
# /etc/masterha_default.cnf
[server default]
user=mha
password=mhapass
repl_user=repl
repl_password=replpass
manager_workdir=/var/log/masterha
manager_log=/var/log/masterha/manager.log
remote_workdir=/var/log/masterha
ssh_user=root
ping_interval=3

[server1]
hostname=192.168.1.10
candidate_master=1

[server2]
hostname=192.168.1.11
candidate_master=1

[server3]
hostname=192.168.1.12
candidate_master=0
```

**方案二：MySQL Group Replication（MGR）**

```sql
-- 单主模式配置
[mysqld]
server_id=1
gtid_mode=ON
enforce_gtid_consistency=ON
binlog_checksum=NONE
log_bin=binlog
log_slave_updates=ON
binlog_format=ROW
master_info_repository=TABLE
relay_log_info_repository=TABLE
transaction_write_set_extraction=XXHASH64
loose-group_replication_group_name="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
loose-group_replication_start_on_boot=ON
loose-group_replication_local_address="192.168.1.10:33061"
loose-group_replication_group_seeds="192.168.1.10:33061,192.168.1.11:33061,192.168.1.12:33061"
loose-group_replication_bootstrap_group=OFF
loose-group_replication_single_primary_mode=ON  -- 单主模式
```

**方案三：MySQL InnoDB Cluster（MySQL 8.0）**

```javascript
// MySQL Shell 创建集群
// 连接到主节点
\connect root@192.168.1.10:3306

// 创建集群
var cluster = dba.createCluster('myCluster');

// 添加节点
cluster.addInstance('root@192.168.1.11:3306');
cluster.addInstance('root@192.168.1.12:3306');

// 查看集群状态
cluster.status();

// 输出：
{
    "clusterName": "myCluster",
    "defaultReplicaSet": {
        "name": "default",
        "primary": "192.168.1.10:3306",
        "status": "OK",
        "topology": {
            "192.168.1.10:3306": {
                "address": "192.168.1.10:3306",
                "mode": "R/W",
                "readReplicas": 0,
                "role": "HA",
                "status": "ONLINE"
            },
            "192.168.1.11:3306": {
                "address": "192.168.1.11:3306",
                "mode": "R/O",
                "readReplicas": 0,
                "role": "HA",
                "status": "ONLINE"
            }
        }
    }
}
```

**高可用设计要点：**

| 要点 | 说明 |
|-----|------|
| 故障检测 | 心跳检测、延迟检测 |
| 自动切换 | VIP 漂移或 DNS 切换 |
| 数据一致性 | 半同步复制、MGR |
| 健康检查 | 定期检查复制状态、延迟 |
| 降级策略 | 主库故障时的备用方案 |
| 监控告警 | 实时监控、及时告警 |

**追问：如何设计异地多活架构？**

**追问答案：**

```
异地多活架构：

                    ┌─────────────┐
                    │  全局 DNS   │
                    │  / LB      │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │  北京机房  │    │  上海机房  │    │  广州机房  │
   │  Primary  │←──→│ Secondary │←──→│ Secondary │
   └───────────┘    └───────────┘    └───────────┘

设计要点：
1. 数据同步：使用 MySQL 异步复制或同步工具
2. 流量路由：就近接入，故障时切换
3. 冲突解决：全局唯一 ID、时间戳比较
4. 降级策略：单机房故障时其他机房接管
```

---

### 6. 如何设计 MySQL 备份恢复方案？

**答案：**

完善的备份恢复方案是数据安全的最后防线：

**备份策略分类：**

| 备份类型 | 说明 | 恢复速度 | 存储空间 |
|---------|------|---------|---------|
| 全量备份 | 完整数据拷贝 | 快 | 大 |
| 增量备份 | 仅备份变化数据 | 中 | 小 |
| 日志备份 | binlog 备份 | 慢（需重放） | 最小 |

**方案一：mysqldump 逻辑备份**

```bash
# 全量备份
mysqldump -u root -p --single-transaction --routines --triggers \
  --all-databases > full_backup_$(date +%Y%m%d).sql

# 参数说明：
# --single-transaction: InnoDB 一致性快照，不锁表
# --routines: 备份存储过程和函数
# --triggers: 备份触发器
# --master-data=2: 记录 binlog 位置

# 恢复
mysql -u root -p < full_backup_20240115.sql
```

**方案二：Percona XtraBackup 物理备份**

```bash
# 安装
yum install percona-xtrabackup

# 全量备份
xtrabackup --backup --target-dir=/backup/full --user=root --password=pass

# 增量备份
xtrabackup --backup --target-dir=/backup/inc1 \
  --incremental-basedir=/backup/full --user=root --password=pass

# 准备备份（应用 redo log）
xtrabackup --prepare --target-dir=/backup/full

# 恢复
xtrabackup --copy-back --target-dir=/backup/full

# 修改权限
chown -R mysql:mysql /var/lib/mysql
```

**方案三：binlog 备份 + 全量备份**

```bash
# 定期全量备份（每天）
0 2 * * * /usr/bin/mysqldump --single-transaction --master-data=2 \
  --all-databases > /backup/full_$(date +\%Y\%m\%d).sql

# 实时备份 binlog
# 配置 my.cnf
[mysqld]
log_bin = /var/lib/mysql/mysql-bin
binlog_format = ROW
expire_logs_days = 7

# 或者使用 mysqlbinlog 工具
mysqlbinlog --read-from-remote-server --raw --stop-never \
  --host=localhost --port=3306 --user=backup --password=pass \
  mysql-bin.000001
```

**恢复到指定时间点：**

```bash
# 1. 恢复全量备份
mysql -u root -p < full_backup_20240115.sql

# 2. 查看全量备份的 binlog 位置
head -50 full_backup_20240115.sql | grep "CHANGE MASTER TO"

# 3. 重放 binlog 到指定时间点
mysqlbinlog --start-position=154 --stop-datetime="2024-01-15 14:00:00" \
  mysql-bin.000003 | mysql -u root -p
```

**自动化备份脚本：**

```bash
#!/bin/bash
# mysql_backup.sh

BACKUP_DIR="/data/backup"
DATE=$(date +%Y%m%d)
MYSQL_USER="backup"
MYSQL_PASS="backup_password"

# 创建备份目录
mkdir -p $BACKUP_DIR/$DATE

# 全量备份
mysqldump -u$MYSQL_USER -p$MYSQL_PASS --single-transaction \
  --master-data=2 --all-databases | gzip > $BACKUP_DIR/$DATE/full.sql.gz

# 备份 binlog
cp /var/lib/mysql/mysql-bin.* $BACKUP_DIR/$DATE/

# 删除 7 天前的备份
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;

# 验证备份
if [ -f "$BACKUP_DIR/$DATE/full.sql.gz" ]; then
    echo "Backup completed successfully"
    # 发送通知
    curl -X POST "https://api.notification.com/send" \
      -d "message=MySQL backup completed"
else
    echo "Backup failed"
    # 发送告警
    curl -X POST "https://api.notification.com/alert" \
      -d "message=MySQL backup failed"
fi
```

**备份验证：**

```bash
# 定期验证备份可恢复性
# 1. 创建测试实例
# 2. 恢复备份
# 3. 验证数据完整性
# 4. 记录恢复时间

# 恢复测试脚本
#!/bin/bash
# 恢复到测试环境
mysql -h test-db -u root -p < /backup/full_20240115.sql

# 数据校验
checksum_prod=$(mysql -h prod-db -e "CHECKSUM TABLE users" | tail -1)
checksum_test=$(mysql -h test-db -e "CHECKSUM TABLE users" | tail -1)

if [ "$checksum_prod" == "$checksum_test" ]; then
    echo "Backup verified successfully"
else
    echo "Backup verification failed"
fi
```

**追问：如何设计容灾演练方案？**

**追问答案：**

```
容灾演练流程：

1. 准备阶段
   - 确定演练目标和范围
   - 准备演练环境
   - 通知相关人员

2. 执行阶段
   - 模拟故障场景（主库宕机、网络中断等）
   - 执行故障转移
   - 验证业务恢复

3. 验证阶段
   - 数据一致性验证
   - 功能验证
   - 性能验证

4. 恢复阶段
   - 恢复原始架构
   - 验证业务正常

5. 总结阶段
   - 记录演练过程
   - 分析问题和改进点
   - 更新运维文档

演练频率建议：
- 每季度：完整容灾演练
- 每月：备份恢复演练
- 每周：监控告警测试
```

