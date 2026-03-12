# MySQL 面试题

MySQL 是最流行的关系型数据库之一，其面试题涵盖了索引、事务、锁机制、性能优化和架构设计等多个核心领域。本文将从基础到高级，系统梳理 MySQL 高频面试题。

## 索引篇

索引是 MySQL 性能优化的核心，深入理解索引原理对于数据库优化至关重要。

### 1. 为什么 MySQL 使用 B+树而不是 B树作为索引结构？

**答案：**

B+树相比 B树更适合数据库索引，主要原因如下：

1. **叶子节点存储所有数据**：B+树的所有数据都存储在叶子节点，非叶子节点只存储索引键值，使得每个节点能存储更多索引项，降低树的高度，减少磁盘 I/O 次数。

2. **范围查询效率高**：B+树叶子节点通过双向链表连接，范围查询只需遍历链表，而 B树需要中序遍历，效率较低。

3. **查询性能稳定**：B+树所有查询都要走到叶子节点，查询时间相对稳定；B树可能在任意节点找到数据，性能波动较大。

```
B树结构示意（数据分布在各层）：
        [30]
       /    \
   [10,20]  [40,50]
    数据也在中间节点

B+树结构示意（数据只在叶子节点）：
        [30]
       /    \
  [10,20] → [40,50] → NULL
   叶子节点用链表连接，数据只在叶子节点
```

**追问：为什么不用 Hash 索引或二叉树？**

**追问答案：**
- **Hash 索引**：只支持等值查询，不支持范围查询和排序，无法利用索引进行 ORDER BY 操作。
- **二叉树/红黑树**：树的高度较高，对于大规模数据，磁盘 I/O 次数多，性能不如 B+树。

---

### 2. 什么是聚簇索引和非聚簇索引？

**答案：**

| 特性 | 聚簇索引 | 非聚簇索引 |
|------|----------|------------|
| 数据存储 | 索引和数据存储在一起 | 索引和数据分开存储 |
| 叶子节点 | 存储完整的行数据 | 存储主键值 |
| 数量 | 一张表只能有一个 | 可以有多个 |
| 查询效率 | 主键查询效率高 | 需要回表查询 |

**聚簇索引结构：**

```
聚簇索引（主键索引）：
        [id: 30]
           /    \
    [id: 10,20]  [id: 40,50]
        ↓            ↓
   叶子节点存储完整行数据
   [id=10, name='张三', age=25]
```

**非聚簇索引结构：**

```
非聚簇索引（辅助索引）：
        [name: '李']
           /    \
    ['张','王']  ['李','赵']
        ↓            ↓
   叶子节点存储主键值
   [name='张三', id=10]  → 回表查询完整数据
```

**追问：为什么非聚簇索引叶子节点存主键值而不是地址？**

**追问答案：**
当发生行移动或数据页分裂时，如果存储地址，需要更新所有相关索引的指针。存储主键值虽然需要回表，但避免了维护指针的开销，且主键值不会改变。

---

### 3. 什么是覆盖索引？为什么能提升性能？

**答案：**

覆盖索引是指查询的所有字段都在索引中，不需要回表查询就能获取所有数据。


**示例：**

```sql
-- 创建联合索引
CREATE INDEX idx_name_age ON user(name, age);

-- 覆盖索引查询（不需要回表）
SELECT name, age FROM user WHERE name = '张三';

-- 非覆盖索引查询（需要回表）
SELECT name, age, email FROM user WHERE name = '张三';
```

**性能提升原因：**
1. 避免回表操作，减少磁盘 I/O
2. 索引数据比行数据小，更多的数据能缓存在内存中
3. 对于 InnoDB，辅助索引可直接返回结果

**追问：如何判断是否使用了覆盖索引？**

**追问答案：**
使用 `EXPLAIN` 查看执行计划，如果 `Extra` 字段显示 `Using index`，说明使用了覆盖索引。

```sql
EXPLAIN SELECT name, age FROM user WHERE name = '张三';
-- Extra: Using index
```

---

### 4. 什么情况下索引会失效？

**答案：**

索引失效的常见场景：

**1. 使用 `LIKE` 以通配符开头**

```sql
-- 索引失效
SELECT * FROM user WHERE name LIKE '%张';

-- 索引有效
SELECT * FROM user WHERE name LIKE '张%';
```

**2. 对索引列使用函数或运算**

```sql
-- 索引失效
SELECT * FROM user WHERE YEAR(create_time) = 2023;
SELECT * FROM user WHERE age + 1 = 25;

-- 索引有效（改写后）
SELECT * FROM user WHERE create_time >= '2023-01-01' AND create_time < '2024-01-01';
SELECT * FROM user WHERE age = 24;
```

**3. 使用 `OR` 连接非索引列**

```sql
-- 假设 name 有索引，email 没有索引
SELECT * FROM user WHERE name = '张三' OR email = 'test@example.com';
-- 索引失效，需要全表扫描
```

**4. 数据类型隐式转换**

```sql
-- name 是 varchar 类型
SELECT * FROM user WHERE name = 123;  -- 索引失效
SELECT * FROM user WHERE name = '123';  -- 索引有效
```

**5. 违反最左前缀原则**

```sql
-- 联合索引 (name, age, email)
SELECT * FROM user WHERE age = 25;  -- 索引失效
SELECT * FROM user WHERE name = '张三' AND age = 25;  -- 索引有效
```

**6. `NOT IN`、`NOT LIKE`、`<>` 操作**

```sql
SELECT * FROM user WHERE age NOT IN (25, 30);  -- 索引可能失效
SELECT * FROM user WHERE age <> 25;  -- 索引可能失效
```

**追问：如何避免索引失效？**

**追问答案：**
1. 避免在索引列上使用函数或运算
2. 使用覆盖索引避免回表
3. 遵循最左前缀原则
4. 避免隐式类型转换
5. 使用 `EXPLAIN` 分析执行计划

---

### 5. 最左前缀原则是什么？

**答案：**

最左前缀原则是指联合索引在使用时，必须从索引的最左列开始且不跳过中间列。

**示例：**

```sql
-- 创建联合索引
CREATE INDEX idx_name_age_email ON user(name, age, email);

-- 索引使用情况分析
SELECT * FROM user WHERE name = '张三';  -- ✅ 使用索引（最左列）
SELECT * FROM user WHERE name = '张三' AND age = 25;  -- ✅ 使用索引
SELECT * FROM user WHERE name = '张三' AND age = 25 AND email = 'a@b.com';  -- ✅ 使用索引

SELECT * FROM user WHERE age = 25;  -- ❌ 索引失效（跳过最左列）
SELECT * FROM user WHERE name = '张三' AND email = 'a@b.com';  -- ⚠️ 部分使用（只用 name）
SELECT * FROM user WHERE age = 25 AND email = 'a@b.com';  -- ❌ 索引失效
```

**B+树索引结构示意：**

```
联合索引 (name, age, email) 的 B+树排序规则：
先按 name 排序，name 相同时按 age 排序，age 相同时按 email 排序

叶子节点：
[('张三', 20, 'a@b.com')]
[('张三', 25, 'b@b.com')]
[('李四', 20, 'c@b.com')]
[('李四', 30, 'd@b.com')]
[('王五', 25, 'e@b.com')]

跳过 name 直接查 age 无法利用索引的有序性
```


**追问：如果查询条件顺序与索引列顺序不一致，索引会失效吗？**

**追问答案：**
不会。MySQL 优化器会自动调整查询条件的顺序，使其符合最左前缀原则。

```sql
-- 这两条语句效果相同
SELECT * FROM user WHERE age = 25 AND name = '张三';
SELECT * FROM user WHERE name = '张三' AND age = 25;
-- 优化器会自动调整为先匹配 name
```

---

### 6. 什么是索引下推（Index Condition Pushdown，ICP）？

**答案：**

索引下推是 MySQL 5.6 引入的优化技术，将索引条件的过滤下推到存储引擎层，减少回表次数。

**没有 ICP 的情况：**

```
查询：SELECT * FROM user WHERE name LIKE '张%' AND age = 25;
索引：(name, age)

1. 存储引擎：根据 name LIKE '张%' 扫描索引，找到所有 name 以"张"开头的记录
2. 服务层：对每条记录进行回表，获取完整行数据
3. 服务层：再判断 age = 25 条件
```

**使用 ICP 的情况：**

```
1. 存储引擎：扫描索引时直接判断 name LIKE '张%' AND age = 25
2. 存储引擎：只返回满足条件的记录的主键
3. 服务层：对满足条件的记录回表
```

**示例对比：**

```sql
-- 创建联合索引
CREATE INDEX idx_name_age ON user(name, age);

-- 使用 ICP（Extra: Using index condition）
EXPLAIN SELECT * FROM user WHERE name LIKE '张%' AND age = 25;
```

**追问：ICP 适用于什么场景？**

**追问答案：**
适用于联合索引中部分列可以使用索引，部分列不能使用索引但可以作为过滤条件的场景。比如 `LIKE` 范围查询后面还有其他条件时。

---

### 7. 联合索引的设计原则是什么？

**答案：**

联合索引设计需要考虑以下原则：

**1. 最左前缀原则**
- 把最常用的查询条件放在最左边
- 把选择性高（区分度高）的列放在左边

**2. 覆盖索引优化**
- 如果查询只需要索引列，可以避免回表

**3. 索引列顺序**

```sql
-- 假设查询场景
SELECT * FROM user WHERE name = '张三' AND age = 25;
SELECT * FROM user WHERE name = '张三';

-- 分析列的选择性
SELECT 
    COUNT(DISTINCT name) / COUNT(*) as name_selectivity,
    COUNT(DISTINCT age) / COUNT(*) as age_selectivity
FROM user;

-- 假设 name 选择性更高，应该把 name 放在前面
CREATE INDEX idx_name_age ON user(name, age);
```

**4. 避免冗余索引**

```sql
-- 冗余索引（idx_name 是 idx_name_age 的前缀）
CREATE INDEX idx_name ON user(name);
CREATE INDEX idx_name_age ON user(name, age);

-- 只需要保留一个
CREATE INDEX idx_name_age ON user(name, age);
```

**追问：如何判断列的选择性？**

**追问答案：**
选择性 = 不同值的数量 / 总行数。选择性越高，索引效果越好。

```sql
SELECT 
    COUNT(DISTINCT column_name) / COUNT(*) as selectivity
FROM table_name;
-- selectivity 越接近 1 越好
```

---

### 8. EXPLAIN 各字段的含义是什么？

**答案：**

`EXPLAIN` 是分析 SQL 执行计划的核心工具：

| 字段 | 含义 | 重要值 |
|------|------|--------|
| id | 查询标识符 | 相同则顺序执行，不同则大的先执行 |
| select_type | 查询类型 | SIMPLE、PRIMARY、SUBQUERY、DERIVED |
| table | 访问的表 | - |
| partitions | 匹配的分区 | - |
| **type** | 访问类型 | system > const > eq_ref > ref > range > index > ALL |
| possible_keys | 可能使用的索引 | - |
| key | 实际使用的索引 | - |
| key_len | 使用的索引长度 | 越短越好 |
| ref | 索引比较的列 | - |
| rows | 预估扫描行数 | 越少越好 |
| filtered | 条件过滤的百分比 | 越高越好 |
| **Extra** | 额外信息 | Using index、Using where、Using filesort |

**type 字段详解：**

```sql
-- system：表只有一行（系统表）
-- const：主键或唯一索引查询，最多一条
EXPLAIN SELECT * FROM user WHERE id = 1;

-- eq_ref：关联查询时使用主键或唯一索引
EXPLAIN SELECT * FROM user u JOIN order o ON u.id = o.user_id;

-- ref：非唯一索引查询
EXPLAIN SELECT * FROM user WHERE name = '张三';

-- range：索引范围扫描
EXPLAIN SELECT * FROM user WHERE age BETWEEN 20 AND 30;

-- index：全索引扫描
EXPLAIN SELECT id FROM user;

-- ALL：全表扫描（最差，需要优化）
EXPLAIN SELECT * FROM user WHERE age + 1 = 25;
```

**Extra 字段详解：**

```
Using index：使用覆盖索引
Using where：服务层过滤数据
Using index condition：使用索引下推
Using filesort：需要额外排序（可能需要优化）
Using temporary：使用临时表（可能需要优化）
```

**追问：key_len 如何计算？**

**追问答案：**

`key_len` 表示使用的索引长度，计算规则：
- 字符串：`长度 × 字符集最大长度 + 是否允许NULL(1) + 变长字段(2)`
- int：4字节 + 是否允许NULL(1)
- bigint：8字节 + 是否允许NULL(1)

```sql
-- name VARCHAR(50) NOT NULL, utf8mb4 编码
-- key_len = 50 × 4 + 0 + 2 = 202

-- age INT NULL
-- key_len = 4 + 1 = 5
```

---

### 9. 什么是回表？如何减少回表？

**答案：**

回表是指通过辅助索引查找数据时，需要先在索引树找到主键值，再回到主键索引树查找完整数据的过程。

**回表过程示意：**

```
辅助索引 B+树                 主键索引 B+树（聚簇索引）
     [name]                       [id]
      /  \                        /  \
  [张]   [李]                  [10]   [30]
    ↓       ↓                     ↓       ↓
[id=10]  [id=30]              完整行数据   完整行数据
    ↓
    └─────→ 回表查询 ─────→ 在聚簇索引中查找完整数据
```

**减少回表的方法：**

```sql
-- 1. 使用覆盖索引
CREATE INDEX idx_name_age ON user(name, age);
SELECT name, age FROM user WHERE name = '张三';  -- 不回表

-- 2. 使用主键查询
SELECT * FROM user WHERE id = 1;  -- 直接在聚簇索引中查找

-- 3. 延迟关联
-- 优化前：需要回表 10000 次
SELECT * FROM user WHERE name LIKE '张%' LIMIT 10000;

-- 优化后：先查主键，再关联，减少回表
SELECT u.* FROM user u
INNER JOIN (SELECT id FROM user WHERE name LIKE '张%' LIMIT 10000) t
ON u.id = t.id;
```

**追问：为什么辅助索引存主键而不是地址？**

**追问答案：**
1. 当发生行移动（如页分裂）时，主键值不变，无需更新辅助索引
2. 如果存地址，每次行移动都需要更新所有辅助索引，开销巨大
3. InnoDB 的设计权衡：增加一次回表的代价换取索引维护的简化

---

### 10. 什么是自适应哈希索引（AHI）？

**答案：**

自适应哈希索引是 InnoDB 自动为热点数据建立的哈希索引，无需人工干预。

**工作原理：**
- InnoDB 监控索引页的访问频率
- 当某些索引页被频繁访问时，自动为其建立哈希索引
- 哈希索引建立在 B+树索引之上

**特点：**
- 只能用于等值查询（WHERE col = value）
- 自动创建，无法手动干预
- 适合查询模式稳定、频繁访问的场景

**查看 AHI 状态：**

```sql
SHOW ENGINE INNODB STATUS\G
-- HASH INDEXES 部分

-- 关闭 AHI（某些高并发写入场景可能需要关闭）
SET GLOBAL innodb_adaptive_hash_index = OFF;
```

**追问：什么情况下应该关闭 AHI？**

**追问答案：**
1. 高并发写入场景，AHI 可能成为瓶颈（需要频繁更新哈希表）
2. 内存紧张时，AHI 占用内存
3. 查询模式不稳定，AHI 命中率低

---

### 11. 什么是索引页分裂？如何避免？

**答案：**

索引页分裂是指当索引页已满时，需要将页面一分为二来插入新数据。

**页分裂过程：**

```
插入前（页面已满）：
[1, 3, 5, 7, 9]

插入 4 后发生页分裂：
          [5]
         /   \
    [1, 3, 4]  [5, 7, 9]

页分裂的影响：
1. 移动数据，消耗 CPU 和 I/O
2. 产生页内碎片
3. 可能导致二级索引存储的指针失效（如果存地址的话）
```

**避免页分裂的方法：**

```sql
-- 1. 使用自增主键（顺序插入）
CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50)
);

-- 2. 设置合适的填充因子
-- innodb_fill_factor = 70 表示页面留 30% 空间用于后续插入

-- 3. 批量插入时排序
INSERT INTO user (id, name) VALUES 
(1, 'a'), (2, 'b'), (3, 'c');  -- 顺序插入
```

---

## 事务篇

事务是数据库保证数据一致性的核心机制，深入理解事务对于高并发系统设计至关重要。

### 1. ACID 特性是什么？

**答案：**

ACID 是事务的四个基本特性：

| 特性 | 含义 | 说明 |
|------|------|------|
| **A**tomicity（原子性） | 事务是不可分割的工作单位 | 要么全部成功，要么全部失败回滚 |
| **C**onsistency（一致性） | 事务前后数据保持一致状态 | 数据库从一个一致状态变为另一个一致状态 |
| **I**solation（隔离性） | 多个事务互不干扰 | 各事务独立执行，不互相影响 |
| **D**urability（持久性） | 事务提交后永久保存 | 即使系统崩溃也能恢复 |

**MySQL 如何保证 ACID：**

```
原子性：undo log（回滚日志）
一致性：应用层约束 + 数据库约束
隔离性：锁机制 + MVCC
持久性：redo log（重做日志）
```


**追问：事务的一致性是怎么保证的？**

**追问答案：**
一致性由多个层面共同保证：
1. 数据库层面：约束（主键、外键、唯一键、检查约束）、触发器
2. 应用层面：业务逻辑的正确性
3. 原子性和隔离性的保证也是一致性的基础

---

### 2. 四种隔离级别是什么？各解决什么问题？

**答案：**

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 实现方式 |
|----------|------|------------|------|----------|
| 读未提交（READ UNCOMMITTED） | ❌ | ❌ | ❌ | 无 |
| 读已提交（READ COMMITTED） | ✅ | ❌ | ❌ | MVCC |
| 可重复读（REPEATABLE READ） | ✅ | ✅ | ✅* | MVCC + 间隙锁 |
| 串行化（SERIALIZABLE） | ✅ | ✅ | ✅ | 锁 |

*注：MySQL InnoDB 在 RR 级别通过 MVCC + Next-Key Lock 也解决了幻读问题。

**各问题解释：**

```sql
-- 脏读：读到其他事务未提交的数据
-- 事务A                          事务B
UPDATE user SET age=30 WHERE id=1;
                                  SELECT age FROM user WHERE id=1;  -- 读到30
ROLLBACK;                         -- 数据恢复为原来的值
                                  -- 事务B读到了脏数据

-- 不可重复读：同一事务两次读取结果不同（针对UPDATE）
-- 事务A                          事务B
SELECT age FROM user WHERE id=1;  -- age=20
                                  UPDATE user SET age=30 WHERE id=1;
                                  COMMIT;
SELECT age FROM user WHERE id=1;  -- age=30，两次读取不一致

-- 幻读：同一事务两次读取记录数不同（针对INSERT/DELETE）
-- 事务A                          事务B
SELECT * FROM user WHERE age>20;  -- 返回2条
                                  INSERT INTO user(age) VALUES(25);
                                  COMMIT;
SELECT * FROM user WHERE age>20;  -- 返回3条，多了幻影行
```

**设置隔离级别：**

```sql
-- 查看当前隔离级别
SELECT @@transaction_isolation;

-- 设置隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET GLOBAL TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

**追问：MySQL 默认隔离级别是什么？为什么选择它？**

**追问答案：**
MySQL 默认是 REPEATABLE READ（可重复读）。选择原因：
1. 解决了脏读、不可重复读、幻读问题
2. 相比串行化，并发性更好
3. 相比读已提交，数据一致性更强
4. 通过 MVCC 实现非阻塞读，性能较好

---

### 3. 什么是 MVCC？怎么实现的？

**答案：**

MVCC（Multi-Version Concurrency Control，多版本并发控制）是 InnoDB 实现高并发的核心技术，通过保存数据的历史版本实现非阻塞读。

**MVCC 核心组件：**

**1. 隐藏列**
每行数据包含三个隐藏字段：
- `DB_TRX_ID`（6字节）：最后修改该行的事务ID
- `DB_ROLL_PTR`（7字节）：指向 undo log 的指针
- `DB_ROW_ID`（6字节）：隐藏主键（如果没有显式主键）

**2. undo log（回滚日志）**
存储数据的历史版本，形成版本链：

```
当前数据：name='李四', DB_TRX_ID=100, DB_ROLL_PTR →
undo log: name='张三', DB_TRX_ID=99, DB_ROLL_PTR →
undo log: name='王五', DB_TRX_ID=98, DB_ROLL_PTR → NULL

版本链从新到旧串联
```

**3. Read View（读视图）**
判断数据版本可见性的快照。

**MVCC 工作流程：**

```
事务A（id=100）读取数据：
1. 获取当前数据的 DB_TRX_ID
2. 根据 Read View 判断该版本是否可见
3. 如果不可见，通过 DB_ROLL_PTR 找到上一个版本
4. 重复判断，直到找到可见版本或版本链结束
```

**追问：MVCC 解决了什么问题？**

**追问答案：**
1. 解决了读写冲突，读操作不阻塞写操作
2. 实现了快照读，避免了加锁读的性能问题
3. 为 RC 和 RR 隔离级别提供了实现基础

---

### 4. Read View 是什么？怎么判断可见性？

**答案：**

Read View 是事务进行快照读时生成的一个"快照"，用于判断数据版本的可见性。

**Read View 包含的字段：**

```
m_ids：生成 Read View 时活跃的事务ID列表
min_trx_id：m_ids 中最小的事务ID
max_trx_id：生成 Read View 时系统应分配的下一个事务ID
creator_trx_id：创建该 Read View 的事务ID
```

**可见性判断规则：**

```
被访问版本的 DB_TRX_ID = creator_trx_id
    → 自己修改的，可见 ✓

DB_TRX_ID < min_trx_id
    → 事务已提交，可见 ✓

DB_TRX_ID >= max_trx_id
    → 事务在 Read View 生成后才开启，不可见 ✗

min_trx_id <= DB_TRX_ID < max_trx_id
    → 需要判断 DB_TRX_ID 是否在 m_ids 中
    → 在 m_ids 中：事务未提交，不可见 ✗
    → 不在 m_ids 中：事务已提交，可见 ✓
```

**图解可见性判断：**

```
Read View: m_ids=[100, 102], min_trx_id=100, max_trx_id=103

事务ID分配：
... ← 99(已提交) ← 100(活跃) ← 101(已提交) ← 102(活跃) ← 103(待分配) ...

判断：
DB_TRX_ID = 99 < min_trx_id → 已提交，可见
DB_TRX_ID = 100 在 m_ids 中 → 活跃，不可见
DB_TRX_ID = 101 不在 m_ids 中且在范围内 → 已提交，可见
DB_TRX_ID = 102 在 m_ids 中 → 活跃，不可见
DB_TRX_ID = 103 >= max_trx_id → 后来开启，不可见
```

**追问：RC 和 RR 隔离级别的 Read View 有什么区别？**

**追问答案：**
- **RC（读已提交）**：每次 SELECT 都生成新的 Read View
- **RR（可重复读）**：只在事务第一次 SELECT 时生成 Read View，后续复用

```sql
-- RC 级别：每次读取都生成新的 Read View
-- 所以能读到其他事务提交的新数据

-- RR 级别：整个事务期间使用同一个 Read View
-- 所以读取结果始终一致，实现可重复读
```


---

### 5. 什么是幻读？MySQL 如何解决幻读？

**答案：**

幻读是指同一事务中，两次相同条件的查询返回的记录数不同（"幻影"行出现）。

**幻读场景：**

```sql
-- 事务A                              事务B
SELECT * FROM user WHERE age > 20;
-- 返回 id=1, id=2 两条记录
                                      INSERT INTO user(id, age) VALUES(3, 25);
                                      COMMIT;
SELECT * FROM user WHERE age > 20;
-- 返回 id=1, id=2, id=3 三条记录
-- 出现了幻读！
```

**MySQL 解决幻读的方法：**

**1. 快照读（普通 SELECT）**
MVCC 通过 Read View 解决幻读。

```sql
-- RR 隔离级别下，第一次 SELECT 生成 Read View
-- 后续 SELECT 复用同一个 Read View
-- 即使其他事务插入了新数据，也看不到
SELECT * FROM user WHERE age > 20;
```

**2. 当前读（SELECT ... FOR UPDATE）**
通过 Next-Key Lock 解决幻读。

```sql
-- 对查询范围的记录加锁，并锁住间隙
SELECT * FROM user WHERE age > 20 FOR UPDATE;
-- 加锁范围：(负无穷, age=20] 和 (age=20, 正无穷)
-- 阻止其他事务在该范围内插入
```

**Next-Key Lock 示意图：**

```
表数据：id=1(age=10), id=2(age=20), id=3(age=30)

查询：SELECT * FROM user WHERE age=20 FOR UPDATE

加锁情况：
- Record Lock：锁定 id=2 这一行
- Gap Lock：锁定 (age=10, age=20) 和 (age=20, age=30) 两个间隙
- Next-Key Lock = Record Lock + Gap Lock

效果：其他事务无法在 age=10 到 age=30 之间插入数据
```

**追问：为什么 RR 级别的普通 SELECT 不会加锁？**

**追问答案：**
为了提高并发性能。普通 SELECT 使用 MVCC 快照读，不加锁也能保证一致性。只有在当前读（FOR UPDATE、FOR SHARE）或 UPDATE、DELETE 操作时才加锁。

---

### 6. redo log 和 undo log 的作用是什么？

**答案：**

| 日志类型 | 作用 | 写入时机 | 存储位置 |
|----------|------|----------|----------|
| **redo log** | 保证持久性，崩溃恢复 | 事务提交时 | 磁盘（循环写） |
| **undo log** | 保证原子性，回滚、MVCC | 数据修改时 | 共享表空间 |

**redo log（重做日志）：**

```
作用：崩溃恢复，保证持久性

工作流程：
1. 事务提交时，先写 redo log buffer
2. 根据 innodb_flush_log_at_trx_commit 决定刷盘时机
3. 崩溃后，通过 redo log 重做已提交的事务

写入模式：
[checkpoint] → [已写入] → [write pos] → [空闲区域] → [checkpoint]
循环写入，覆盖已 checkpoint 的区域
```

**配置参数：**

```sql
-- 0：每秒刷盘，可能丢失1秒数据
-- 1：每次提交刷盘（默认，最安全）
-- 2：每次提交写入os cache，每秒刷盘
SET GLOBAL innodb_flush_log_at_trx_commit = 1;
```

**undo log（回滚日志）：**

```
作用：
1. 事务回滚时恢复数据（原子性）
2. MVCC 中构建历史版本（快照读）

工作流程：
1. 修改数据前，先将旧值写入 undo log
2. 回滚时，从 undo log 恢复数据
3. MVCC 中，通过 undo log 构建版本链
```

**undo log 类型：**

```sql
-- insert undo log：insert 产生，事务提交后立即删除
-- update undo log：update/delete 产生，保留供 MVCC 使用
```

**追问：为什么 redo log 比 undo log 更重要？**

**追问答案：**
1. redo log 保证持久性，数据不丢失是最基本的要求
2. redo log 是顺序写，性能高；数据页是随机写
3. undo log 在事务提交后可以清理，而 redo log 需要持久保存直到 checkpoint

---

### 7. 事务隔离级别是怎么实现的？

**答案：**

不同隔离级别通过不同机制实现：

**读未提交（RU）：**
- 直接读取最新数据，不加锁，不使用 MVCC

**读已提交（RC）：**
- MVCC 实现
- 每次 SELECT 生成新的 Read View
- 能看到其他事务已提交的修改

**可重复读（RR）：**
- MVCC + Next-Key Lock 实现
- 只在第一次 SELECT 生成 Read View
- 配合间隙锁防止幻读

**串行化（SERIALIZABLE）：**
- 读操作加共享锁，写操作加排他锁
- 完全串行执行，并发度最低

**实现机制对比：**

```
                    读操作                写操作
RU                  无锁读取              加行锁
RC                  MVCC（每次新ReadView） 行锁
RR                  MVCC（复用ReadView）   行锁 + 间隙锁
SERIALIZABLE        共享锁                排他锁
```

**代码示例：**

```sql
-- 查看隔离级别
SELECT @@transaction_isolation;

-- RC 级别演示
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT * FROM user WHERE id = 1;  -- 生成 Read View 1
-- 其他事务修改并提交
SELECT * FROM user WHERE id = 1;  -- 生成 Read View 2，可能读到新数据
COMMIT;

-- RR 级别演示
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT * FROM user WHERE id = 1;  -- 生成 Read View
-- 其他事务修改并提交
SELECT * FROM user WHERE id = 1;  -- 复用 Read View，数据不变
COMMIT;
```

---

### 8. 什么是脏页？什么时候会刷脏页？

**答案：**

脏页（Dirty Page）是指内存中被修改但还未刷入磁盘的数据页。

**Buffer Pool 工作机制：**

```
查询流程：
1. 查询数据 → 检查 Buffer Pool
2. 缓存命中 → 直接返回
3. 缓存未命中 → 从磁盘读取到 Buffer Pool

修改流程：
1. 修改数据 → 更新 Buffer Pool 中的页（成为脏页）
2. 写入 redo log（保证持久性）
3. 后台线程异步刷脏页
```

**刷脏页的时机：**

```
1. Buffer Pool 空间不足
   - 需要淘汰页时，如果是脏页需要先刷盘
   - 通过 LRU 算法淘汰最近最少使用的页

2. 后台定时刷盘
   - innodb_page_cleaners 控制刷盘线程数
   - 定期检查并刷新脏页

3. redo log 写满
   - 当 redo log 循环写满时，需要暂停写入
   - 强制刷脏页，推进 checkpoint

4. 正常关闭 MySQL
   - 所有脏页都需要刷入磁盘

5. checkpoint
   - 推进 checkpoint 位置时刷盘
```

**相关参数：**

```sql
-- 脏页比例阈值，超过则加速刷盘
SET GLOBAL innodb_max_dirty_pages_pct = 75;

-- 脏页比例低水位，低于此值不主动刷盘
SET GLOBAL innodb_max_dirty_pages_pct_lwm = 10;

-- 刷盘时是否阻塞用户请求
SET GLOBAL innodb_flush_neighbors = 0;  -- SSD 设为 0
```


---

### 9. 什么是 WAL 机制？有什么优势？

**答案：**

WAL（Write-Ahead Logging，预写日志）是先写日志、再写磁盘的策略。

**WAL 工作流程：**

```
传统方式（先写数据）：
修改数据 → 写入磁盘（随机I/O，慢）→ 返回成功

WAL 方式（先写日志）：
修改数据 → 写入 redo log（顺序I/O，快）→ 返回成功
         → 后台异步刷脏页
```

**WAL 优势：**

```
1. 性能优化
   - redo log 顺序写，速度远快于数据页随机写
   - 组提交：多个事务合并一次刷盘

2. 崩溃恢复
   - 即使数据页未刷盘，也能通过 redo log 恢复

3. 减少磁盘 I/O
   - 批量刷脏页，合并多次修改
```

**追问：为什么顺序写比随机写快？**

**追问答案：**
顺序写只需移动磁头一次，然后连续写入；随机写需要频繁移动磁头到不同位置。对于 SSD，顺序写也能更好地利用预读和并行写入。

---

### 10. 长事务有什么问题？如何避免？

**答案：**

长事务存在的问题：

```
1. undo log 无法清理
   - MVCC 需要保留历史版本
   - 长事务不结束，undo log 不能删除
   - 占用大量空间

2. 锁竞争
   - 长事务持有锁时间长
   - 阻塞其他事务

3. 主从延迟
   - 大事务执行时间长
   - 从库回放慢

4. 连接池耗尽
   - 长时间占用连接
   - 高并发时连接不够用

5. 回滚代价大
   - 长事务回滚时间长
   - undo 操作可能失败
```

**避免长事务：**

```sql
-- 1. 设置事务超时时间
SET SESSION innodb_lock_wait_timeout = 50;
SET SESSION MAX_EXECUTION_TIME = 10000;  -- 10秒

-- 2. 监控长事务
SELECT * FROM information_schema.innodb_trx
WHERE TIME_TO_SEC(TIMEDIFF(NOW(), trx_started)) > 60;

-- 3. 拆分大事务
-- 错误：一个事务做太多事
BEGIN;
-- 大量操作
COMMIT;

-- 正确：拆分成多个小事务
BEGIN;
-- 小批量操作
COMMIT;
BEGIN;
-- 小批量操作
COMMIT;

-- 4. 只读事务使用快照读
START TRANSACTION READ ONLY;
```

---

## 锁机制篇

锁是数据库实现并发控制的核心机制，理解锁对排查并发问题至关重要。

### 1. MySQL 有哪些锁类型？

**答案：**

MySQL 锁可以从不同维度分类：

**按锁粒度分类：**

| 锁类型 | 粒度 | 开销 | 并发度 | 应用场景 |
|--------|------|------|--------|----------|
| 全局锁 | 整个数据库 | 低 | 最低 | 全库逻辑备份 |
| 表级锁 | 整个表 | 较低 | 较低 | MyISAM、DDL |
| 行级锁 | 单行记录 | 高 | 最高 | InnoDB |

**按锁类型分类：**

```
共享锁（S锁）：读锁，多个事务可同时持有
排他锁（X锁）：写锁，独占，阻塞其他锁

意向共享锁（IS）：事务想获取某行的S锁，先在表级别加IS
意向排他锁（IX）：事务想获取某行的X锁，先在表级别加IX
```

**InnoDB 行锁类型：**

```
Record Lock：锁单个记录
Gap Lock：锁记录之间的间隙（不包含记录本身）
Next-Key Lock：Record Lock + Gap Lock，锁记录及其前面的间隙
Insert Intention Lock：插入意向锁，用于 insert 操作
```

**锁兼容矩阵：**

```
        S      X      IS     IX
S       ✓      ✗      ✓      ✗
X       ✗      ✗      ✗      ✗
IS      ✓      ✗      ✓      ✓
IX      ✗      ✗      ✓      ✓
```

**SQL 语句加锁情况：**

```sql
-- 共享锁
SELECT * FROM user WHERE id = 1 LOCK IN SHARE MODE;
SELECT * FROM user WHERE id = 1 FOR SHARE;  -- MySQL 8.0+

-- 排他锁
SELECT * FROM user WHERE id = 1 FOR UPDATE;
UPDATE user SET name = 'test' WHERE id = 1;
DELETE FROM user WHERE id = 1;
INSERT INTO user VALUES (1, 'test');
```

**追问：意向锁的作用是什么？**

**追问答案：**
意向锁是为了让表级锁和行级锁能够协调工作。当事务想加表锁时，不需要检查每行是否有行锁，只需检查是否有意向锁即可，提高效率。

---

### 2. 什么是行锁、间隙锁、Next-Key Lock？

**答案：**

**Record Lock（记录锁）：**
锁定单条索引记录。

```sql
-- 对 id=5 的记录加 Record Lock
SELECT * FROM user WHERE id = 5 FOR UPDATE;
-- 锁住 id=5 这一行
```

**Gap Lock（间隙锁）：**
锁定两个记录之间的间隙，防止幻读。

```sql
-- 假设有记录 id=1, 5, 10
SELECT * FROM user WHERE id > 3 AND id < 8 FOR UPDATE;
-- Gap Lock 锁住 (1, 5) 和 (5, 10) 两个间隙
-- 阻止其他事务在间隙中插入
```

**Next-Key Lock：**
Record Lock + Gap Lock，锁定记录及其前面的间隙。

```
假设有记录 id=1, 5, 10

执行：SELECT * FROM user WHERE id = 5 FOR UPDATE

Next-Key Lock 加锁范围：
- (1, 5]：锁定记录 5 及其前面的间隙

效果：
- 阻止其他事务插入 id=2,3,4
- 阻止其他事务修改 id=5
```

**图解 Next-Key Lock：**

```
索引记录：    1     5     10     15     20
              │     │      │      │      │
间隙：      (-∞,1) (1,5) (5,10) (10,15) (15,20) (20,+∞)

Next-Key Lock (5, 10]：
- 锁定间隙 (5, 10)
- 锁定记录 10
- 等效于 Gap Lock(5,10) + Record Lock(10)
```

**追问：什么情况下只会加 Record Lock？**

**追问答案：**
当查询使用唯一索引且是等值查询时，只会加 Record Lock，不会加 Gap Lock。

```sql
-- id 是主键（唯一索引）
SELECT * FROM user WHERE id = 5 FOR UPDATE;
-- 只锁 id=5 这一行，不加间隙锁

-- 如果是范围查询，仍会加间隙锁
SELECT * FROM user WHERE id >= 5 FOR UPDATE;
-- 加 Record Lock(5) + Gap Lock(5, 正无穷)
```

---

### 3. 什么情况下会死锁？怎么解决？

**答案：**

死锁是指两个或多个事务相互等待对方释放锁，形成循环等待。

**死锁场景：**

```sql
-- 事务A                           事务B
BEGIN;                            BEGIN;
UPDATE user SET age=1 WHERE id=1;
                                  UPDATE user SET age=2 WHERE id=2;
UPDATE user SET age=1 WHERE id=2;  -- 等待B释放id=2的锁
                                  UPDATE user SET age=2 WHERE id=1;  
                                  -- 等待A释放id=1的锁
                                  -- 死锁形成！
```


**死锁检测与处理：**

```sql
-- 查看死锁信息
SHOW ENGINE INNODB STATUS;

-- 查看锁等待情况
SELECT * FROM information_schema.innodb_lock_waits;

-- 开启死锁检测（默认开启）
SET GLOBAL innodb_deadlock_detect = ON;
```

**解决死锁的方法：**

```
1. 死锁检测
   - InnoDB 自动检测死锁
   - 回滚代价最小的事务

2. 设置超时时间
   - innodb_lock_wait_timeout（默认50秒）
   - 超时自动回滚

3. 应用层预防
   - 按固定顺序访问表和行
   - 大事务拆小事务
   - 尽量使用索引访问数据
```

**预防死锁最佳实践：**

```sql
-- 1. 统一访问顺序
-- 所有事务都按 id 升序访问
SELECT * FROM user WHERE id IN (1, 2, 3) FOR UPDATE;

-- 2. 减少锁持有时间
BEGIN;
-- 先准备好数据，再加锁
UPDATE user SET age = 20 WHERE id = 1;
COMMIT;  -- 快速提交

-- 3. 避免长事务
-- 4. 合理设计索引，避免全表扫描
```

**追问：如何查看最近的死锁信息？**

**追问答案：**

```sql
-- 方法1：查看 InnoDB 状态
SHOW ENGINE INNODB STATUS\G
-- 找到 LATEST DETECTED DEADLOCK 部分

-- 方法2：开启死锁日志
SET GLOBAL innodb_print_all_deadlocks = ON;
-- 死锁信息会写入错误日志
```

---

### 4. 乐观锁和悲观锁的区别？

**答案：**

| 特性 | 悲观锁 | 乐观锁 |
|------|--------|--------|
| 思想 | 假设会冲突，先加锁 | 假设不会冲突，提交时检查 |
| 实现 | SELECT ... FOR UPDATE | 版本号/时间戳 |
| 适用场景 | 冲突多的场景 | 冲突少的场景 |
| 性能 | 锁开销大，阻塞等待 | 无锁开销，可能重试 |

**悲观锁实现：**

```sql
-- 先加锁再操作
BEGIN;
SELECT balance FROM account WHERE id = 1 FOR UPDATE;  -- 加排他锁
UPDATE account SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

**乐观锁实现：**

```sql
-- 使用版本号
-- 表结构：id, balance, version

-- 1. 读取数据和版本号
SELECT id, balance, version FROM account WHERE id = 1;
-- balance=1000, version=1

-- 2. 更新时检查版本号
UPDATE account 
SET balance = balance - 100, version = version + 1
WHERE id = 1 AND version = 1;

-- 3. 检查影响行数
-- 如果 affected_rows = 0，说明版本已变化，需要重试
```

**CAS（Compare And Swap）实现：**

```sql
-- 不使用版本号，直接比较值
UPDATE account 
SET balance = 900  -- 新值
WHERE id = 1 AND balance = 1000;  -- 期望的旧值

-- 如果 balance 已经不是 1000，更新失败
```

**追问：什么场景适合乐观锁？什么场景适合悲观锁？**

**追问答案：**
- **乐观锁**：读多写少、冲突少的场景（如商品浏览、博客阅读）
- **悲观锁**：写多读少、冲突多的场景（如秒杀抢购、账户转账）

---

### 5. MySQL 加锁规则是什么？

**答案：**

InnoDB 加锁规则总结（RR 隔离级别）：

**基本原则：**

```
1. 加锁的基本单位是 Next-Key Lock
2. 查找过程中访问到的对象才会加锁
3. 索引上的等值查询有特殊规则
```

**等值查询规则：**

```sql
-- 假设表有记录 id=1, 5, 10, 15

-- 规则1：唯一索引等值查询，匹配到记录
SELECT * FROM user WHERE id = 5 FOR UPDATE;
-- 只加 Record Lock(5)，因为唯一索引保证了没有其他5

-- 规则2：唯一索引等值查询，未匹配到记录
SELECT * FROM user WHERE id = 3 FOR UPDATE;
-- 加 Gap Lock(1, 5)，阻止插入3

-- 规则3：普通索引等值查询
-- 假设 age 是普通索引，有 age=20, 25, 25, 30
SELECT * FROM user WHERE age = 25 FOR UPDATE;
-- Record Lock(两条age=25) + Gap Lock(20, 25) + Gap Lock(25, 30)

-- 规则4：唯一索引范围查询
SELECT * FROM user WHERE id >= 5 FOR UPDATE;
-- Next-Key Lock(5, 10] + Gap Lock(10, +∞)
```

**范围查询规则：**

```sql
-- 规则5：范围查询，无论是否唯一索引，都加 Next-Key Lock
SELECT * FROM user WHERE id > 5 FOR UPDATE;
-- Next-Key Lock(5, 10] + Next-Key Lock(10, +∞)

-- 规则6：<= 范围查询特殊处理
SELECT * FROM user WHERE id <= 5 FOR UPDATE;
-- Next-Key Lock(-∞, 1] + Next-Key Lock(1, 5] + Gap Lock(5, 10)
```

**无索引查询：**

```sql
SELECT * FROM user WHERE name = '张三' FOR UPDATE;
-- name 无索引，全表扫描
-- 所有记录加 Next-Key Lock
-- 等效于锁全表
```

**追问：如何分析和排查锁问题？**

**追问答案：**

```sql
-- 1. 查看当前锁等待
SELECT * FROM information_schema.innodb_lock_waits;

-- 2. 查看当前事务
SELECT * FROM information_schema.innodb_trx;

-- 3. 查看 InnoDB 锁信息
SELECT * FROM performance_schema.data_locks;  -- MySQL 8.0+

-- 4. 查看死锁日志
SHOW ENGINE INNODB STATUS\G
```

---

### 6. 什么是意向锁？有什么作用？

**答案：**

意向锁是表级锁，用于表示事务有意向在表中的某些行上加锁。

**意向锁类型：**

```
IS（意向共享锁）：事务想在某些行上加 S 锁
IX（意向排他锁）：事务想在某些行上加 X 锁
```

**意向锁的作用：**

```
场景：事务A 在某行加了 X 锁，事务B 想加表锁

没有意向锁：
- 事务B 需要扫描全表检查每行是否有锁
- 性能极差

有意向锁：
- 事务A 加行锁前，先加表级 IX 锁
- 事务B 加表锁时，检查是否有意向锁冲突
- IX 与 X 表锁冲突，直接阻塞
```

**锁兼容矩阵：**

```
            表S    表X    IS     IX
表S          ✓      ✗      ✓      ✗
表X          ✗      ✗      ✗      ✗
IS           ✓      ✗      ✓      ✓
IX           ✗      ✗      ✓      ✓
```

**示例：**

```sql
-- 事务A
BEGIN;
SELECT * FROM user WHERE id = 1 FOR UPDATE;
-- InnoDB 自动在表上加 IX 锁，在行上加 X 锁

-- 事务B
LOCK TABLES user READ;
-- 需要加表 S 锁
-- 与 IX 冲突，被阻塞
```

---

### 7. 什么是自增锁？

**答案：**

自增锁是插入数据时为自增列分配ID的特殊表级锁。

**自增锁模式（innodb_autoinc_lock_mode）：**

```
0：传统模式
   - 所有 INSERT 都加表级自增锁
   - 语句结束释放，并发度最低
   - 保证自增值连续

1：连续模式（默认）
   - 简单 INSERT 用轻量级锁
   - 批量 INSERT 用表级自增锁
   - 平衡并发和连续性

2：交叉模式
   - 所有 INSERT 用轻量级锁
   - 并发度最高
   - 自增值可能不连续
```

**示例：**

```sql
-- 查看自增锁模式
SHOW VARIABLES LIKE 'innodb_autoinc_lock_mode';

-- 传统模式（0）
INSERT INTO user (name) VALUES ('a'), ('b'), ('c');
-- 获得连续 ID：1, 2, 3

-- 交叉模式（2）并发插入
-- 事务A                        事务B
INSERT ... VALUES ('a'),('b')   INSERT ... VALUES ('c')
-- 可能分配：A: 1,3            B: 2
-- ID 不连续
```

**追问：为什么自增主键不推荐使用很大的值？**

**追问答案：**
1. 自增值用完后，INSERT 会失败
2. 使用 uuid 或很大的自增值会增大索引空间
3. 大的自增值可能导致索引页分裂

---

### 8. 什么是元数据锁（MDL）？

**答案：**

元数据锁（Metadata Lock）是 MySQL 5.5 引入的，用于保护表结构的一致性。

**MDL 类型：**

```
MDL_READ：读锁，SELECT 时自动加
MDL_WRITE：写锁，ALTER TABLE 时加
MDL_SHARED_READ：共享读锁
MDL_SHARED_WRITE：共享写锁
MDL_EXCLUSIVE：排他锁
```

**MDL 阻塞场景：**

```sql
-- 事务A
BEGIN;
SELECT * FROM user;  -- 加 MDL_READ
-- 长时间未提交

-- 事务B
ALTER TABLE user ADD COLUMN age INT;  -- 需要 MDL_WRITE
-- 被 A 阻塞！

-- 事务C
SELECT * FROM user;  -- 需要 MDL_READ
-- 被 B 阻塞！（即使 B 还没获得锁）
-- 形成队列：A → B(等待) → C(等待)
```

**解决 MDL 阻塞：**

```sql
-- 1. 设置等待超时
SET SESSION lock_wait_timeout = 5;  -- 5秒超时

-- 2. ALTER 前检查是否有长事务
SELECT * FROM information_schema.innodb_trx
WHERE TIME_TO_SEC(TIMEDIFF(NOW(), trx_started)) > 60;

-- 3. Online DDL（MySQL 5.6+）
ALTER TABLE user ADD COLUMN age INT, ALGORITHM=INPLACE, LOCK=NONE;
```


---

## 性能优化篇

性能优化是 MySQL 面试的核心考察点，需要掌握从分析到优化的完整方法论。

### 1. 慢查询怎么排查？

**答案：**

慢查询排查分为发现、定位、分析、优化四个步骤。

**1. 开启慢查询日志：**

```sql
-- 查看慢查询配置
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';

-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
SET GLOBAL long_query_time = 1;  -- 超过1秒记录
SET GLOBAL log_queries_not_using_indexes = ON;  -- 记录未用索引的查询
```

**2. 分析慢查询日志：**

```bash
# 使用 mysqldumpslow 分析
mysqldumpslow -s t -t 10 /var/log/mysql/slow.log
# -s t: 按查询时间排序
# -t 10: 显示前10条

# 输出示例
Count: 100  Time=2.50s  Lock=0.01s  Rows=1000
SELECT * FROM user WHERE name LIKE '%张%'
```

**3. 使用 EXPLAIN 分析：**

```sql
EXPLAIN SELECT * FROM user WHERE name LIKE '%张%'\G

-- 重点关注：
-- type: ALL 表示全表扫描
-- key: NULL 表示未用索引
-- rows: 预估扫描行数
-- Extra: Using filesort, Using temporary 表示需要优化
```

**4. 使用 SHOW PROFILE：**

```sql
-- 开启 profiling
SET profiling = ON;

-- 执行查询
SELECT * FROM user WHERE name LIKE '张%';

-- 查看执行详情
SHOW PROFILE;

-- 查看具体耗时
SHOW PROFILE CPU, BLOCK IO FOR QUERY 1;
```

**追问：除了慢查询日志，还有什么方法发现性能问题？**

**追问答案：**

```sql
-- 1. 查看当前运行的查询
SHOW PROCESSLIST;
SELECT * FROM information_schema.processlist 
WHERE TIME > 5;  -- 执行超过5秒的

-- 2. 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS\G

-- 3. 使用 Performance Schema
SELECT * FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC LIMIT 10;

-- 4. 监控指标
SHOW GLOBAL STATUS LIKE 'Innodb_row_lock%';
SHOW GLOBAL STATUS LIKE 'Slow_queries';
```

---

### 2. EXPLAIN 各字段含义是什么？

**答案：**

`EXPLAIN` 返回的字段详解：

**id - 查询标识符：**

```sql
-- 相同 id：顺序执行
EXPLAIN SELECT * FROM user u JOIN order o ON u.id = o.user_id;
-- id=1, id=1

-- 不同 id：大的先执行（子查询）
EXPLAIN SELECT * FROM user WHERE id IN (SELECT user_id FROM order);
-- id=2 先执行子查询，id=1 再执行主查询
```

**select_type - 查询类型：**

| 类型 | 说明 |
|------|------|
| SIMPLE | 简单查询，不含子查询或 UNION |
| PRIMARY | 最外层查询 |
| SUBQUERY | 子查询中的第一个 SELECT |
| DERIVED | 派生表（FROM 子句中的子查询） |
| UNION | UNION 中的第二个及之后的 SELECT |

**type - 访问类型（从好到差）：**

```
system > const > eq_ref > ref > fulltext > ref_or_null 
> index_merge > unique_subquery > index_subquery 
> range > index > ALL
```

```sql
-- system/const：主键或唯一索引等值查询
EXPLAIN SELECT * FROM user WHERE id = 1;

-- eq_ref：JOIN 时使用主键或唯一索引
EXPLAIN SELECT * FROM user u JOIN order o ON u.id = o.user_id;

-- ref：非唯一索引等值查询
EXPLAIN SELECT * FROM user WHERE name = '张三';

-- range：索引范围扫描
EXPLAIN SELECT * FROM user WHERE age BETWEEN 20 AND 30;

-- index：全索引扫描
EXPLAIN SELECT id FROM user;

-- ALL：全表扫描（最差）
EXPLAIN SELECT * FROM user WHERE age + 1 = 25;
```

**key 和 key_len：**

```sql
-- key：实际使用的索引
-- key_len：使用的索引长度（越短越好）

EXPLAIN SELECT * FROM user WHERE name = '张三' AND age = 25;
-- key: idx_name_age
-- key_len: 202 + 5 = 207（name varchar(50) utf8mb4 + age int）
```

**Extra - 额外信息：**

```sql
-- Using index：覆盖索引
EXPLAIN SELECT name, age FROM user WHERE name = '张三';

-- Using where：服务层过滤
EXPLAIN SELECT * FROM user WHERE age > 20;

-- Using index condition：索引下推
EXPLAIN SELECT * FROM user WHERE name LIKE '张%' AND age = 25;

-- Using filesort：文件排序（可能需要优化）
EXPLAIN SELECT * FROM user ORDER BY create_time;

-- Using temporary：使用临时表（可能需要优化）
EXPLAIN SELECT DISTINCT age FROM user;
```

**追问：什么情况下 type=ALL 需要优化？**

**追问答案：**
- 小表全表扫描可能比索引更快
- 需要结合 rows 和数据量判断
- 如果确实需要优化，考虑添加合适索引或改写 SQL

---

### 3. 如何优化大表查询？

**答案：**

大表查询优化需要从多个维度入手：

**1. 索引优化：**

```sql
-- 避免索引失效
-- 错误
SELECT * FROM user WHERE YEAR(create_time) = 2023;
-- 正确
SELECT * FROM user WHERE create_time >= '2023-01-01' AND create_time < '2024-01-01';

-- 使用覆盖索引
CREATE INDEX idx_name_age ON user(name, age);
SELECT name, age FROM user WHERE name = '张三';
```

**2. 查询优化：**

```sql
-- 只查需要的列
SELECT id, name FROM user WHERE age > 20;
-- 而不是
SELECT * FROM user WHERE age > 20;

-- 分页优化
-- 错误：深分页性能差
SELECT * FROM user LIMIT 1000000, 10;

-- 正确：使用子查询
SELECT * FROM user u 
INNER JOIN (SELECT id FROM user LIMIT 1000000, 10) t 
ON u.id = t.id;

-- 或使用 WHERE 条件
SELECT * FROM user WHERE id > 1000000 LIMIT 10;
```

**3. 分区表：**

```sql
-- 按范围分区
CREATE TABLE orders (
    id INT,
    order_date DATE,
    amount DECIMAL(10,2)
) PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 查询时只扫描相关分区
SELECT * FROM orders WHERE order_date >= '2023-01-01';
```

**4. 分库分表：**

```sql
-- 垂直分表：大字段拆分
-- 主表
CREATE TABLE user (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT
);

-- 扩展表
CREATE TABLE user_extra (
    user_id INT PRIMARY KEY,
    intro TEXT,
    avatar LONGBLOB
);

-- 水平分表：按规则拆分
-- user_0, user_1, ..., user_99
-- 按 user_id % 100 分片
```

**追问：深分页为什么慢？如何优化？**

**追问答案：**

```sql
-- 深分页问题
SELECT * FROM user LIMIT 1000000, 10;
-- 需要扫描前 1000010 行，丢弃前 1000000 行

-- 优化方法1：延迟关联
SELECT u.* FROM user u
INNER JOIN (SELECT id FROM user ORDER BY id LIMIT 1000000, 10) t
ON u.id = t.id;
-- 子查询只扫描索引，快

-- 优化方法2：记住上次位置
SELECT * FROM user WHERE id > 1000000 ORDER BY id LIMIT 10;
-- 直接定位，无需扫描前面的行
```

---

### 4. 如何优化 COUNT 查询？

**答案：**

COUNT 查询优化策略：

**COUNT 各种用法的区别：**

```sql
-- COUNT(*)：统计总行数（推荐）
SELECT COUNT(*) FROM user;

-- COUNT(1)：与 COUNT(*) 等价
SELECT COUNT(1) FROM user;

-- COUNT(列名)：统计非 NULL 值数量
SELECT COUNT(email) FROM user;  -- 不统计 email 为 NULL 的

-- COUNT(DISTINCT 列名)：统计不同值的数量
SELECT COUNT(DISTINCT age) FROM user;
```

**优化策略：**

```sql
-- 1. 使用覆盖索引
CREATE INDEX idx_status ON user(status);
SELECT COUNT(*) FROM user WHERE status = 1;

-- 2. 使用近似值
EXPLAIN SELECT COUNT(*) FROM user;
-- rows 字段给出估算值

-- 3. 维护计数表
CREATE TABLE user_count (total INT);
UPDATE user_count SET total = total + 1;  -- 插入时更新

-- 4. 利用缓存
-- Redis 缓存总数，定期同步

-- 5. 条件计数优化
-- 错误
SELECT COUNT(*) FROM user WHERE status = 1;
SELECT COUNT(*) FROM user WHERE status = 2;

-- 正确：一次查询
SELECT 
    SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) as status_1,
    SUM(CASE WHEN status = 2 THEN 1 ELSE 0 END) as status_2
FROM user;
```

**追问：COUNT(*) 和 COUNT(列名) 哪个快？**

**追问答案：**
- `COUNT(*)` 通常更快，InnoDB 会选择最小的辅助索引扫描
- `COUNT(列名)` 需要判断是否为 NULL，可能更慢
- 如果列有 NOT NULL 约束，性能相当

---

### 5. 如何优化 ORDER BY？

**答案：**

ORDER BY 优化策略：

**1. 使用索引排序：**

```sql
-- 创建索引
CREATE INDEX idx_age_name ON user(age, name);

-- 利用索引排序
SELECT * FROM user ORDER BY age, name;  -- Using index

-- 反向排序
SELECT * FROM user ORDER BY age DESC, name DESC;  -- Using index

-- 混合排序无法使用索引
SELECT * FROM user ORDER BY age ASC, name DESC;  -- Using filesort
```

**2. 避免文件排序（filesort）：**

```sql
-- filesort 触发条件：
-- 1. 排序列没有索引
-- 2. 索引列顺序与 ORDER BY 不一致
-- 3. 混合 ASC/DESC

-- 查看 sort_buffer_size
SHOW VARIABLES LIKE 'sort_buffer_size';

-- 增大排序缓冲区
SET SESSION sort_buffer_size = 256000;
```

**3. 优化分页排序：**

```sql
-- 错误：深分页 + 排序
SELECT * FROM user ORDER BY create_time LIMIT 1000000, 10;
-- 需要对所有数据排序

-- 正确：延迟关联
SELECT u.* FROM user u
INNER JOIN (
    SELECT id FROM user ORDER BY create_time LIMIT 1000000, 10
) t ON u.id = t.id;

-- 正确：记录位置
SELECT * FROM user 
WHERE create_time < '上次的时间' 
ORDER BY create_time DESC 
LIMIT 10;
```

**追问：什么是 filesort？**

**追问答案：**
filesort 是 MySQL 在内存中（或磁盘上）对数据进行排序的过程，与文件系统无关。

- 数据量小于 `sort_buffer_size`：内存排序
- 数据量大：分块排序后合并，临时文件在磁盘

---

### 6. 如何优化 JOIN 查询？

**答案：**

JOIN 查询优化策略：

**1. 确保 JOIN 列有索引：**

```sql
-- 为关联列创建索引
CREATE INDEX idx_user_id ON orders(user_id);

-- JOIN 时使用索引
SELECT u.name, o.amount
FROM user u
JOIN orders o ON u.id = o.user_id;
```

**2. 小表驱动大表：**

```sql
-- JOIN 执行过程：
-- 遍历驱动表的每一行
-- 根据关联条件在被驱动表中查找

-- 原则：小表驱动大表（减少循环次数）
-- 如果 user 表小，orders 表大
SELECT u.name, o.amount
FROM user u
JOIN orders o ON u.id = o.user_id;
-- MySQL 优化器会自动选择小表作为驱动表

-- 强制指定驱动表（STRAIGHT_JOIN）
SELECT /*+ STRAIGHT_JOIN */ u.name, o.amount
FROM user u
JOIN orders o ON u.id = o.user_id;
```

**3. 使用 Index Nested Loop Join：**

```
三种 JOIN 算法：

1. Simple Nested Loop Join
   - 驱动表每行都扫描被驱动表
   - 性能最差

2. Block Nested Loop Join (BNL)
   - 将驱动表数据读入 join_buffer
   - 批量匹配，减少扫描次数
   - 没有 JOIN 列索引时使用

3. Index Nested Loop Join (NLJ)
   - 被驱动表的 JOIN 列有索引
   - 直接索引查找，性能最好
```

**4. 优化 JOIN Buffer：**

```sql
-- 查看 join_buffer_size
SHOW VARIABLES LIKE 'join_buffer_size';

-- 增大 JOIN 缓冲区
SET SESSION join_buffer_size = 256000;

-- BNL 算法消耗 join_buffer
-- 确保索引以使用 NLJ 算法
```


---

### 7. 如何优化大表 DDL？

**答案：**

大表 DDL 操作会锁表，影响线上业务。优化方法：

**1. Online DDL（MySQL 5.6+）：**

```sql
-- Online DDL 语法
ALTER TABLE user 
ADD COLUMN age INT,
ALGORITHM=INPLACE,  -- 在线执行
LOCK=NONE;          -- 不锁表

-- ALGORITHM 选项：
-- COPY：创建临时表，复制数据（阻塞写）
-- INPLACE：原地修改（不阻塞读写）

-- LOCK 选项：
-- EXCLUSIVE：排他锁
-- SHARED：共享锁
-- NONE：无锁
```

**2. pt-online-schema-change 工具：**

```bash
# Percona Toolkit 工具
pt-online-schema-change \
  --alter "ADD COLUMN age INT" \
  --execute \
  D=database,t=user

# 工作原理：
# 1. 创建新表
# 2. 创建触发器同步增量数据
# 3. 分批复制旧表数据到新表
# 4. 重命名表
```

**3. gh-ost 工具：**

```bash
# GitHub 开源的在线 DDL 工具
gh-ost \
  --max-load=Threads_running=25 \
  --critical-load=Threads_running=1000 \
  --chunk-size=1000 \
  --throttle-control-replicas="..." \
  --database=database \
  --table=user \
  --alter="ADD COLUMN age INT" \
  --execute

# 特点：无触发器，基于 binlog 同步
```

**追问：哪些 DDL 支持 Online？**

**追问答案：**

| 操作 | 是否支持 Online | 说明 |
|------|-----------------|------|
| 添加索引 | ✅ | INPLACE |
| 删除索引 | ✅ | INPLACE |
| 添加列 | ✅ | INPLACE（末尾添加） |
| 删除列 | ✅ | INPLACE |
| 修改列类型 | ❌ | 需要 COPY |
| 添加主键 | ✅ | INPLACE |
| 删除主键 | ❌ | 需要 COPY |

---

### 8. 分库分表的原则是什么？

**答案：**

分库分表是解决单表数据量过大的最终手段。

**什么时候需要分库分表：**

```
单表数据量 > 500万（建议值）
单库数据量 > 100GB（建议值）
单机 QPS > 10000（建议值）
```

**分库 vs 分表：**

```
分表：解决单表数据量过大
- 单库内拆分多表
- 索引效率提升
- 但仍在同一实例，并发能力有限

分库：解决单机性能瓶颈
- 拆分到多个实例
- 分散 I/O 和 CPU 压力
- 提高并发能力
```

**垂直拆分：**

```
垂直分表：按字段拆分
user 表：
- 常用字段：id, name, age → user_base
- 不常用字段：intro, avatar → user_extra

垂直分库：按业务拆分
- 用户库：user 相关表
- 订单库：order 相关表
- 商品库：product 相关表
```

**水平拆分：**

```
水平分表：按行拆分
user_0: user_id % 4 = 0
user_1: user_id % 4 = 1
user_2: user_id % 4 = 2
user_3: user_id % 4 = 3

水平分库：按规则分散到不同实例
db_0: user_id % 4 = 0, 1
db_1: user_id % 4 = 2, 3
```

**分片键选择原则：**

```
1. 选择查询频繁的字段
2. 数据分布均匀
3. 范围查询友好

常用分片键：
- 用户ID：适合用户相关业务
- 时间：适合日志、订单等时序数据
- 地区：适合地域性强的业务
```

**分片算法：**

```sql
-- 1. Hash 取模
user_id % 分片数

-- 2. 范围分片
user_id < 1000000 → 分片1
user_id < 2000000 → 分片2

-- 3. 一致性哈希
-- 解决扩容时数据迁移问题

-- 4. 地理位置
根据地区字段分片
```

**分库分表带来的问题：**

```
1. 跨库 JOIN
   - 应用层聚合
   - 数据冗余
   - 全局表

2. 分布式事务
   - 最终一致性
   - TCC / Seata

3. 全局唯一 ID
   - Snowflake 雪花算法
   - 号段模式

4. 聚合查询
   - 每个分片查询后合并
   - 空间换时间
```

---

## 架构篇

理解 MySQL 架构有助于设计高可用、高性能的数据库系统。

### 1. MyISAM 和 InnoDB 的区别是什么？

**答案：**

| 特性 | MyISAM | InnoDB |
|------|--------|--------|
| 事务 | ❌ 不支持 | ✅ 支持 |
| 外键 | ❌ 不支持 | ✅ 支持 |
| 锁粒度 | 表锁 | 行锁 |
| MVCC | ❌ 不支持 | ✅ 支持 |
| 崩溃恢复 | ❌ | ✅ redo log |
| 全文索引 | ✅ 原生支持 | ✅ 5.6+ 支持 |
| 空间占用 | 较小 | 较大 |
| 存储限制 | 256TB | 64TB |

**存储结构：**

```
MyISAM：
- .frm：表结构定义
- .MYD：数据文件
- .MYI：索引文件
- 数据和索引分离存储

InnoDB：
- .frm：表结构定义（8.0前）
- .ibd：数据和索引（共享表空间或独立表空间）
- 数据和索引存储在一起（聚簇索引）
```

**计数对比：**

```sql
-- MyISAM：存储精确行数
SELECT COUNT(*) FROM user;  -- O(1)

-- InnoDB：需要扫描
SELECT COUNT(*) FROM user;  -- O(n)
```

**追问：什么时候用 MyISAM？**

**追问答案：**
现在几乎不需要。InnoDB 在大多数场景下都优于 MyISAM：
- MyISAM 只适合纯读场景
- MyISAM 表锁在高并发下性能差
- MyISAM 崩溃可能丢失数据

---

### 2. 主从复制原理是什么？

**答案：**

MySQL 主从复制基于 binlog 实现。

**复制流程：**

```
主库                     从库
  │                        │
  ├─1. 写入数据              │
  │    ↓                    │
  ├─2. 写入 binlog           │
  │    ↓                    │
  │                    3. Dump Thread 发送 binlog
  │    ←─────────────────────┤
  │                        ↓
  │                    4. IO Thread 接收
  │                        ↓
  │                    5. 写入 relay log
  │                        ↓
  │                    6. SQL Thread 回放
  │                        ↓
  │                    7. 数据同步完成
```

**复制模式：**

```sql
-- 异步复制（默认）
-- 主库写入成功立即返回，不等待从库

-- 半同步复制
-- 至少一个从库确认收到 binlog 后返回
rpl_semi_sync_master_enabled = 1

-- 全同步复制
-- 所有从库确认后才返回（性能差，很少用）

-- GTID 复制
-- 全局事务ID，方便管理复制
gtid_mode = ON
```

**复制格式：**

```sql
-- STATEMENT：记录 SQL 语句
-- 问题：某些函数可能导致不一致

-- ROW：记录行数据变化（推荐）
-- 优点：数据一致性最好

-- MIXED：混合模式
-- 一般用 STATEMENT，特殊情况下用 ROW
```

**配置主从复制：**

```sql
-- 主库配置
[mysqld]
server-id = 1
log_bin = mysql-bin
binlog_format = ROW

-- 从库配置
[mysqld]
server-id = 2
relay_log = mysql-relay

-- 从库执行
CHANGE MASTER TO
  MASTER_HOST = '主库IP',
  MASTER_USER = 'repl',
  MASTER_PASSWORD = 'password',
  MASTER_LOG_FILE = 'mysql-bin.000001',
  MASTER_LOG_POS = 0;

START SLAVE;
```

---

### 3. 如何解决主从延迟？

**答案：**

主从延迟是指从库回放 binlog 落后于主库。

**延迟原因：**

```
1. 从库单线程回放（5.6 之前）
2. 大事务执行时间长
3. 从库硬件性能差
4. 网络延迟
5. 从库执行大量查询
```

**解决方案：**

**1. 开启并行复制（MySQL 5.7+）：**

```sql
-- 从库配置
[mysqld]
slave_parallel_type = LOGICAL_CLOCK
slave_parallel_workers = 8  -- 并行线程数
```

**2. 优化大事务：**

```sql
-- 拆分大事务
-- 错误：一个大事务
BEGIN;
-- 大量操作
COMMIT;

-- 正确：多个小事务
BEGIN;
-- 小批量操作
COMMIT;
```

**3. 使用半同步复制：**

```sql
-- 至少一个从库同步后才返回
-- 降低数据丢失风险
rpl_semi_sync_master_enabled = 1
rpl_semi_sync_master_wait_no_slave = 1
```

**4. 读写分离策略：**

```sql
-- 方法1：关键业务读主库
-- 方法2：延迟检测，从库延迟大时读主库
-- 方法3：使用 ProxySQL 自动路由

-- 方法4：应用层判断
if (需要实时数据) {
    读主库
} else {
    读从库
}
```

**追问：如何监控主从延迟？**

**追问答案：**

```sql
-- 在从库执行
SHOW SLAVE STATUS\G

-- 关注字段
Seconds_Behind_Master: 0  -- 延迟秒数（0 表示无延迟）

-- 使用 pt-heartbeat 监控
pt-heartbeat -D test --update --replace --check
```

---

### 4. MySQL 高可用架构有哪些？

**答案：**

**1. 主从复制 + VIP 切换：**

```
        VIP (虚拟IP)
           │
    ┌──────┴──────┐
    │             │
  Master       Slave
    │             │
  宕机时，VIP 漂移到 Slave
```

**2. MHA（Master High Availability）：**

```
特点：
- 自动故障转移
- 30秒内完成切换
- 尽量保证数据不丢失

架构：
  Master
     │
  ┌──┴──┐
Slave1 Slave2
     │
  MHA Manager（监控 + 切换）
```

**3. MGR（MySQL Group Replication）：**

```
特点：
- MySQL 官方方案
- 基于 Paxos 协议
- 支持多主模式

架构：
  ┌─────────────────────┐
  │   Replication Group │
  │  ┌───┐ ┌───┐ ┌───┐  │
  │  │M1 │ │M2 │ │M3 │  │
  │  └───┘ └───┘ └───┘  │
  └─────────────────────┘
      Paxos 一致性协议
```

**4. Orchestrator：**

```
特点：
- GitHub 开源
- 可视化拓扑管理
- 自动故障检测和恢复
```

**对比：**

| 方案 | 自动切换 | 数据零丢失 | 复杂度 |
|------|----------|------------|--------|
| 主从+VIP | ❌ | ❌ | 低 |
| MHA | ✅ | ⚠️ | 中 |
| MGR | ✅ | ✅ | 高 |
| Orchestrator | ✅ | ⚠️ | 中 |

---

### 5. 什么是 MySQL 半同步复制？

**答案：**

半同步复制是介于异步复制和全同步复制之间的方案。

**三种复制模式对比：**

```
异步复制：
Master 写入 → 立即返回
             ↓
         异步发送 binlog

半同步复制：
Master 写入 → 等待至少一个 Slave 确认 → 返回
             ↓
         收到 ack 后返回

全同步复制：
Master 写入 → 等待所有 Slave 确认 → 返回
             ↓
         全部 ack 后返回
```

**配置半同步复制：**

```sql
-- 安装插件
INSTALL PLUGIN rpl_semi_sync_master SONAME 'semisync_master.so';
INSTALL PLUGIN rpl_semi_sync_slave SONAME 'semisync_slave.so';

-- 主库配置
SET GLOBAL rpl_semi_sync_master_enabled = 1;
SET GLOBAL rpl_semi_sync_master_timeout = 1000;  -- 超时降级

-- 从库配置
SET GLOBAL rpl_semi_sync_slave_enabled = 1;
```

**增强半同步（MySQL 5.7+）：**

```
-- rpl_semi_sync_master_wait_point
AFTER_COMMIT（默认）：事务提交后等待 ack
AFTER_SYNC：收到 ack 后再提交（推荐，数据更安全）

-- AFTER_SYNC 流程：
1. Master 写入 binlog
2. 发送给 Slave
3. Slave 写入 relay log，返回 ack
4. Master 收到 ack，提交事务
5. 返回客户端成功

-- 优点：即使 Master 崩溃，也不会丢失已确认的事务
```

---

### 6. 如何实现 MySQL 读写分离？

**答案：**

读写分离是分散数据库压力的有效手段。

**实现方式：**

**1. 应用层实现：**

```java
// 伪代码
public Connection getConnection(boolean isRead) {
    if (isRead) {
        return slaveDataSource.getConnection();
    } else {
        return masterDataSource.getConnection();
    }
}

// 读操作
Connection conn = getConnection(true);  // 从库
// 写操作
Connection conn = getConnection(false);  // 主库
```

**2. 中间件实现：**

```
常用中间件：
- MyCat：国产开源，功能全面
- ProxySQL：高性能 MySQL 代理
- MySQL Router：官方推荐
- ShardingSphere：Apache 项目

架构：
    应用
      │
  中间件（路由）
      │
  ┌───┴───┐
Master  Slave
  │       │
  写      读
```

**3. ProxySQL 配置示例：**

```sql
-- 定义服务器组
INSERT INTO mysql_servers (hostgroup_id, hostname, port) 
VALUES (10, 'master', 3306);

INSERT INTO mysql_servers (hostgroup_id, hostname, port) 
VALUES (20, 'slave', 3306);

-- 定义路由规则
INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup)
VALUES (1, '^SELECT', 20);  -- SELECT 走从库

INSERT INTO mysql_query_rules (rule_id, match_pattern, destination_hostgroup)
VALUES (2, '.*', 10);       -- 其他走主库
```

**注意事项：**

```
1. 主从延迟问题
   - 实时性要求高的读走主库
   - 写后读场景需要走主库

2. 事务处理
   - 事务中的读需要走主库

3. 负载均衡
   - 多个从库需要负载均衡
   - 健康检查剔除故障节点
```

---

### 7. MySQL 如何保证数据不丢失？

**答案：**

MySQL 通过多种机制保证数据不丢失：

**1. redo log（重做日志）：**

```
作用：保证持久性

写入机制：
1. 事务提交时，先写 redo log
2. 根据配置决定刷盘时机
3. 崩溃后通过 redo log 恢复

配置：
innodb_flush_log_at_trx_commit = 1  -- 最安全
```

**2. binlog（归档日志）：**

```
作用：主从复制、数据恢复

写入机制：
1. 事务提交时写入 binlog cache
2. 根据 sync_binlog 决定刷盘

配置：
sync_binlog = 1  -- 每次提交刷盘
```

**3. 双一原则：**

```sql
-- 推荐配置，保证数据安全
innodb_flush_log_at_trx_commit = 1
sync_binlog = 1
```

**4. 组提交（Group Commit）：**

```
优化：多个事务一起刷盘，减少 I/O

流程：
1. 多个事务进入队列
2. Leader 线程批量刷盘
3. 所有事务一起提交
```

**5. 数据页刷盘：**

```
时机：
1. Buffer Pool 空间不足
2. 后台定时刷盘
3. 正常关闭
4. redo log 写满

配置：
innodb_flush_method = O_DIRECT  -- 绕过 OS cache
```

**追问：如果机器断电，数据会丢失吗？**

**追问答案：**
- 如果配置了 `双一原则`，数据不会丢失
- redo log 已刷盘的事务可以恢复
- 未完成的事务通过 undo log 回滚
- 如果 innodb_flush_log_at_trx_commit=0，可能丢失 1 秒数据

---

### 8. 如何设计 MySQL 参数优化？

**答案：**

关键参数优化建议：

**内存相关：**

```sql
-- InnoDB Buffer Pool（最重要）
-- 建议设为物理内存的 60%-80%
innodb_buffer_pool_size = 8G

-- Buffer Pool 实例数
-- 减少 latch 竞争
innodb_buffer_pool_instances = 8

-- 日志缓冲区
innodb_log_buffer_size = 16M

-- 连接缓冲区
sort_buffer_size = 256K
join_buffer_size = 256K
read_buffer_size = 128K
```

**I/O 相关：**

```sql
-- redo log 刷盘策略
innodb_flush_log_at_trx_commit = 1

-- binlog 刷盘策略
sync_binlog = 1

-- 刷盘方式
innodb_flush_method = O_DIRECT

-- 后台刷盘线程
innodb_page_cleaners = 4

-- 脏页比例
innodb_max_dirty_pages_pct = 75
```

**连接相关：**

```sql
-- 最大连接数
max_connections = 1000

-- 等待超时
wait_timeout = 28800
interactive_timeout = 28800

-- 连接数限制
max_user_connections = 500
```

**慢查询相关：**

```sql
-- 慢查询日志
slow_query_log = ON
long_query_time = 1
log_queries_not_using_indexes = ON
```

**查看当前配置：**

```sql
SHOW VARIABLES LIKE 'innodb_buffer_pool_size';
SHOW VARIABLES LIKE 'innodb_flush%';

-- 动态修改（运行时生效）
SET GLOBAL innodb_buffer_pool_size = 8589934592;
```

---

## 总结

本文涵盖了 MySQL 面试的核心知识点，包括：

- **索引篇**：B+树原理、索引类型、索引优化
- **事务篇**：ACID、隔离级别、MVCC、日志机制
- **锁机制篇**：锁类型、死锁、加锁规则
- **性能优化篇**：慢查询、EXPLAIN、大表优化
- **架构篇**：存储引擎、主从复制、高可用

掌握这些内容，足以应对大多数 MySQL 面试场景。建议结合实际项目经验，形成自己的理解。
