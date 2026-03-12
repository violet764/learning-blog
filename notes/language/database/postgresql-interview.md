# PostgreSQL 面试题

PostgreSQL 是一款功能强大的开源关系型数据库，以其丰富的特性、标准的 SQL 支持和优秀的扩展性著称。本文将从基础到高级，系统梳理 PostgreSQL 高频面试题。

## 基础篇

### 1. PostgreSQL 和 MySQL 的区别？怎么选型？

**答案：**

PostgreSQL 和 MySQL 是最流行的两款开源数据库，对比如下：

| 特性 | PostgreSQL | MySQL |
|------|------------|-------|
| 定位 | 功能全面的对象关系型数据库 | 轻量级关系型数据库 |
| SQL 标准 | 更接近 SQL 标准 | 有较多扩展和差异 |
| 事务支持 | 完整的 ACID，支持 DDL 事务 | InnoDB 支持事务，DDL 不支持 |
| 索引类型 | B-tree、GIN、GiST、BRIN、Hash 等 | B-tree、Hash、Full-text、R-tree |
| JSON 支持 | JSON/JSONB 原生支持，功能强大 | JSON 支持，但功能较弱 |
| 并发控制 | MVCC 无 undo log，VACUUM 机制 | MVCC 有 undo log |
| 扩展性 | 高度可扩展，支持自定义类型/函数 | 扩展性相对有限 |
| 复杂查询 | 支持窗口函数、CTE、递归查询 | MySQL 8.0 后支持 |
| 复制 | 主从复制、逻辑复制、流复制 | 主从复制、组复制 |
| 适用场景 | 复杂业务、地理信息、JSON 文档 | Web 应用、简单业务、读密集 |

**选型建议：**

```
选择 PostgreSQL 的场景：
✅ 复杂查询多（窗口函数、CTE、递归）
✅ 需要存储 JSON 文档（JSONB 性能优秀）
✅ 地理信息系统（PostGIS 扩展）
✅ 需要高度的 SQL 标准兼容性
✅ 需要自定义类型、函数、索引
✅ DDL 事务支持（表结构变更可回滚）

选择 MySQL 的场景：
✅ 简单的 Web 应用，读写比例高
✅ 团队更熟悉 MySQL 生态
✅ 运维工具和文档更丰富
✅ 需要与 MySQL 生态工具集成
✅ 对简单查询性能要求高
```

**追问：为什么说 PostgreSQL 更接近 SQL 标准？**

**追问答案：**
PostgreSQL 严格遵循 SQL 标准，如：
- 字符串连接使用 `||` 而非 CONCAT
- 布尔类型使用 `TRUE/FALSE` 而非 1/0
- 支持标准的 `INTERVAL` 时间间隔类型
- 完整支持窗口函数、CTE 等标准特性
- 类型转换使用 `::` 或 `CAST`

---

### 2. PostgreSQL 有什么优势？

**答案：**

PostgreSQL 的核心优势：

**1. 丰富的数据类型**

```sql
-- 基本类型：int, varchar, timestamp 等
-- 高级类型：
-- 数组类型
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    tags TEXT[]
);
INSERT INTO orders (tags) VALUES (ARRAY['urgent', 'paid']);

-- JSON/JSONB 类型
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    profile JSONB
);

-- 范围类型
CREATE TABLE reservations (
    id SERIAL PRIMARY KEY,
    during TSRANGE
);
INSERT INTO reservations (during) VALUES ('[2024-01-01, 2024-01-05]');

-- 几何类型：point, line, polygon
-- 网络地址类型：inet, cidr, macaddr
-- UUID 类型
-- 自定义复合类型
```

**2. 强大的索引系统**

```sql
-- B-tree：默认索引，支持等值和范围查询
CREATE INDEX idx_name ON users(name);

-- GIN：支持数组、JSONB、全文搜索
CREATE INDEX idx_tags ON orders USING GIN(tags);
CREATE INDEX idx_profile ON users USING GIN(profile);

-- GiST：支持地理空间、范围类型
CREATE INDEX idx_location ON stores USING GIST(location);

-- BRIN：超大表的块范围索引
CREATE INDEX idx_created ON logs USING BRIN(created_at);
```

**3. 扩展性**

```sql
-- 安装扩展
CREATE EXTENSION postgis;      -- 地理信息
CREATE EXTENSION pg_trgm;      -- 模糊搜索
CREATE EXTENSION hstore;       -- 键值存储
CREATE EXTENSION pgvector;     -- 向量搜索
CREATE EXTENSION pg_stat_statements;  -- 性能监控

-- 自定义函数
CREATE OR REPLACE FUNCTION increment(i INTEGER) 
RETURNS INTEGER AS $$
BEGIN
    RETURN i + 1;
END;
$$ LANGUAGE plpgsql;
```

**4. 完整的事务支持**

```sql
-- DDL 也支持事务
BEGIN;
CREATE TABLE test_table (id INT);
INSERT INTO test_table VALUES (1);
ROLLBACK;  -- 表也回滚了，MySQL 中不行

-- Savepoint
BEGIN;
INSERT INTO users (name) VALUES ('张三');
SAVEPOINT sp1;
INSERT INTO users (name) VALUES ('李四');
ROLLBACK TO sp1;  -- 只回滚到 savepoint
COMMIT;  -- 只有张三被插入
```

**追问：PostgreSQL 的缺点是什么？**

**追问答案：**
1. 进程模型（非线程），连接开销大，需要连接池
2. VACUUM 机制可能影响性能，需要监控表膨胀
3. 学习曲线相对陡峭
4. 运维工具生态不如 MySQL 丰富
5. 并发连接数受限，需要 pgBouncer 等连接池

---

### 3. PostgreSQL 支持哪些数据类型？

**答案：**

PostgreSQL 支持丰富的数据类型：

**1. 数值类型**

| 类型 | 大小 | 范围 |
|------|------|------|
| SMALLINT | 2 字节 | -32768 到 +32767 |
| INTEGER | 4 字节 | -21亿 到 +21亿 |
| BIGINT | 8 字节 | -922亿亿 到 +922亿亿 |
| DECIMAL/NUMERIC | 可变 | 任意精度 |
| REAL | 4 字节 | 6 位精度浮点 |
| DOUBLE PRECISION | 8 字节 | 15 位精度浮点 |
| SERIAL | 4 字节 | 自增整数 |

```sql
-- 自增主键
CREATE TABLE users (
    id SERIAL PRIMARY KEY,        -- 自增整数
    name VARCHAR(100)
);

-- 或使用 IDENTITY（PostgreSQL 10+）
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);
```

**2. 字符类型**

| 类型 | 说明 |
|------|------|
| CHARACTER(n) / CHAR(n) | 定长，不足补空格 |
| CHARACTER VARYING(n) / VARCHAR(n) | 变长，有长度限制 |
| TEXT | 变长，无长度限制 |

```sql
-- TEXT 是 PostgreSQL 推荐使用的类型
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT  -- 无长度限制
);
```

**3. 时间类型**

```sql
-- 日期和时间
DATE                    -- 只存日期
TIME                    -- 只存时间
TIMESTAMP               -- 日期+时间
TIMESTAMPTZ             -- 带时区的时间戳
INTERVAL                -- 时间间隔

-- 示例
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_date DATE,
    event_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at_tz TIMESTAMPTZ DEFAULT NOW(),
    duration INTERVAL
);

INSERT INTO events (event_date, event_time, duration) 
VALUES ('2024-01-15', '14:30:00', '2 hours 30 minutes');

-- 时间计算
SELECT created_at + INTERVAL '1 day' AS tomorrow FROM events;
SELECT created_at_tz AT TIME ZONE 'Asia/Shanghai' FROM events;
```

**4. 布尔类型**

```sql
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    is_completed BOOLEAN DEFAULT FALSE
);

-- 三种写法
INSERT INTO tasks (is_completed) VALUES (TRUE);
INSERT INTO tasks (is_completed) VALUES ('t');      -- true
INSERT INTO tasks (is_completed) VALUES ('yes');    -- true
INSERT INTO tasks (is_completed) VALUES (FALSE);
INSERT INTO tasks (is_completed) VALUES ('f');      -- false
INSERT INTO tasks (is_completed) VALUES (NULL);     -- 未知
```

**5. 数组类型**

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    tags TEXT[],
    prices NUMERIC[]
);

-- 插入数据
INSERT INTO products (name, tags, prices) 
VALUES ('iPhone', ARRAY['phone', 'apple'], '{999, 1099, 1199}');

-- 查询数组
SELECT * FROM products WHERE 'phone' = ANY(tags);
SELECT * FROM products WHERE tags @> ARRAY['phone'];  -- 包含
SELECT * FROM products WHERE tags && ARRAY['phone', 'android'];  -- 交集
```

**追问：NUMERIC 和 FLOAT 有什么区别？什么时候用哪个？**

**追问答案：**
- **NUMERIC/DECIMAL**：精确计算，适合金额、科学计算
- **FLOAT/DOUBLE**：近似计算，适合科学计算、统计

```sql
-- 金额必须用 NUMERIC
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    balance NUMERIC(12, 2)  -- 精确到分
);

-- 科学计算可以用 FLOAT
CREATE TABLE measurements (
    id SERIAL PRIMARY KEY,
    temperature FLOAT,
    distance DOUBLE PRECISION
);
```

---

### 4. JSON 和 JSONB 的区别？

**答案：**

| 特性 | JSON | JSONB |
|------|------|-------|
| 存储格式 | 文本格式存储 | 二进制格式存储 |
| 插入速度 | 快（直接存） | 慢（需要解析转换） |
| 查询速度 | 慢（每次解析） | 快（已解析） |
| 索引支持 | 不支持 | 支持 GIN 索引 |
| 空格保留 | 保留 | 不保留 |
| 键顺序 | 保留 | 不保留（按字典序） |
| 重复键 | 保留最后一个 | 只保留最后一个 |

**使用示例：**

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    data JSON,
    data_b JSONB
);

-- JSON 和 JSONB 插入
INSERT INTO documents (data, data_b) VALUES 
('{"name": "张三", "age": 25}', '{"name": "张三", "age": 25}');

-- 查询操作符
SELECT data->'name' FROM documents;        -- 返回 JSON 值："张三"
SELECT data->>'name' FROM documents;       -- 返回文本：张三
SELECT data_b->'age' FROM documents;       -- 25

-- 嵌套查询
INSERT INTO documents (data_b) VALUES 
('{"user": {"profile": {"name": "李四"}}}');
SELECT data_b#>>'{user,profile,name}';     -- 李四

-- 数组访问
INSERT INTO documents (data_b) VALUES 
('{"tags": ["apple", "banana", "cherry"]}');
SELECT data_b->'tags'->0;                  -- "apple"
SELECT data_b->'tags'->>0;                 -- apple
```

**JSONB 高级操作：**

```sql
-- 包含查询 @>
SELECT * FROM documents WHERE data_b @> '{"name": "张三"}';

-- 检查键是否存在 ?
SELECT * FROM documents WHERE data_b ? 'name';

-- 检查多个键是否存在 ?|
SELECT * FROM documents WHERE data_b ?| ARRAY['name', 'age'];

-- 检查所有键是否存在 ?&
SELECT * FROM documents WHERE data_b ?& ARRAY['name', 'age'];

-- 更新操作
UPDATE documents 
SET data_b = data_b || '{"city": "北京"}'::jsonb;  -- 合并

UPDATE documents 
SET data_b = data_b - 'age';  -- 删除键
```

**追问：为什么推荐用 JSONB？**

**追问答案：**
推荐使用 JSONB 的原因：

1. **查询性能好**：数据已解析为二进制，无需每次解析
2. **支持索引**：可以创建 GIN 索引，大幅提升查询速度
3. **操作符丰富**：支持 `@>`、`?`、`?|` 等高效操作符

```sql
-- 创建 GIN 索引
CREATE INDEX idx_data_b ON documents USING GIN(data_b);

-- 使用索引的查询
EXPLAIN SELECT * FROM documents WHERE data_b @> '{"name": "张三"}';
-- Bitmap Index Scan on idx_data_b
```

**什么时候用 JSON？**
- 只需要存储，很少查询
- 需要保留原始格式（空格、键顺序）

---

### 5. PostgreSQL 的表空间是什么？

**答案：**

表空间是 PostgreSQL 中数据存储的物理位置，可以控制数据文件存放位置。

```sql
-- 查看默认表空间
SELECT spcname FROM pg_tablespace;

-- 创建表空间
CREATE TABLESPACE fast_storage LOCATION '/ssd/pgdata';
CREATE TABLESPACE slow_storage LOCATION '/hdd/pgdata';

-- 在指定表空间创建表
CREATE TABLE hot_data (
    id SERIAL PRIMARY KEY,
    data TEXT
) TABLESPACE fast_storage;

CREATE TABLE archive_data (
    id SERIAL PRIMARY KEY,
    data TEXT
) TABLESPACE slow_storage;

-- 移动表到其他表空间
ALTER TABLE hot_data SET TABLESPACE slow_storage;

-- 移动索引
ALTER INDEX idx_hot_data SET TABLESPACE fast_storage;

-- 设置默认表空间
SET default_tablespace = 'fast_storage';
```

**使用场景：**

```
1. 存储分层
   - 热数据放在 SSD（fast_storage）
   - 冷数据放在 HDD（slow_storage）

2. 磁盘空间管理
   - 当一个磁盘满了，可以创建新表空间在另一个磁盘

3. 性能优化
   - 把索引和数据放在不同磁盘，减少 I/O 竞争
   - 把 WAL 和数据放在不同磁盘
```

**追问：默认表空间 pg_default 和 pg_global 有什么区别？**

**追问答案：**
- `pg_default`：存储用户创建的数据库和表，位于 `$PGDATA/base`
- `pg_global`：存储共享系统表（如 pg_database、pg_tablespace），位于 `$PGDATA/global`

---

### 6. 如何查看 PostgreSQL 的版本和配置？

**答案：**

```sql
-- 查看版本
SELECT version();
-- PostgreSQL 15.4 on x86_64-pc-linux-gnu, compiled by gcc...

-- 查看配置
SHOW ALL;  -- 所有配置
SHOW shared_buffers;  -- 单个配置

-- 查看配置来源
SELECT name, setting, source FROM pg_settings 
WHERE name IN ('shared_buffers', 'work_mem', 'max_connections');

-- 通过配置文件查看
SHOW config_file;
-- /etc/postgresql/15/main/postgresql.conf

-- 查看数据目录
SHOW data_directory;

-- 修改配置
ALTER SYSTEM SET shared_buffers = '256MB';  -- 需要重启
ALTER SYSTEM SET work_mem = '64MB';         -- 需要 reload

-- 应用配置
SELECT pg_reload_conf();

-- 查看当前会话配置
SHOW work_mem;
SET work_mem = '128MB';  -- 只对当前会话有效
```

**重要配置参数：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| shared_buffers | 共享内存缓冲区 | 系统内存的 25% |
| work_mem | 单个查询的内存 | 64MB-256MB |
| maintenance_work_mem | 维护操作的内存 | 256MB-1GB |
| max_connections | 最大连接数 | 100-200 |
| effective_cache_size | 预估可用缓存 | 系统内存的 75% |
| checkpoint_completion_target | checkpoint 分散写入 | 0.9 |

**追问：ALTER SYSTEM 和直接修改配置文件有什么区别？**

**追问答案：**
- `ALTER SYSTEM`：修改存储在 `postgresql.auto.conf` 中的配置，不修改主配置文件
- 直接修改：修改 `postgresql.conf` 文件
- `postgresql.auto.conf` 优先级更高，会覆盖 `postgresql.conf` 的设置

---

### 7. 什么是序列（SEQUENCE）？

**答案：**

序列是 PostgreSQL 中用于生成唯一数字的数据库对象，常用于自增主键。

```sql
-- 创建序列
CREATE SEQUENCE user_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

-- 使用序列
SELECT nextval('user_id_seq');  -- 获取下一个值
SELECT currval('user_id_seq');  -- 获取当前会话最后获取的值
SELECT last_value FROM user_id_seq;  -- 查看序列当前值

-- 重置序列
ALTER SEQUENCE user_id_seq RESTART WITH 1;

-- 表关联序列（SERIAL 的本质）
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- 等价于下面
    name VARCHAR(100)
);

-- SERIAL 等价于
CREATE SEQUENCE users_id_seq;
CREATE TABLE users (
    id INT NOT NULL DEFAULT nextval('users_id_seq') PRIMARY KEY,
    name VARCHAR(100)
);
ALTER SEQUENCE users_id_seq OWNED BY users.id;
```

**IDENTITY 列（PostgreSQL 10+ 推荐）：**

```sql
-- 推荐：使用 IDENTITY 替代 SERIAL
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);

-- GENERATED ALWAYS: 不允许手动指定值
-- GENERATED BY DEFAULT: 允许手动指定值

-- 自定义序列参数
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY (
        START WITH 1000
        INCREMENT BY 2
    ) PRIMARY KEY,
    name VARCHAR(100)
);
```

**追问：SERIAL 和 IDENTITY 的区别？**

**追问答案：**

| 特性 | SERIAL | IDENTITY |
|------|--------|----------|
| 标准 | PostgreSQL 特有 | SQL 标准语法 |
| 值管理 | 可以手动插入任何值 | ALWAYS 模式禁止手动插入 |
| 序列管理 | 自动创建独立序列 | 序列与列绑定 |
| 权限 | 需要单独授权序列 | 自动管理权限 |
| 推荐 | 兼容旧版本 | PostgreSQL 10+ 推荐 |

---

### 8. 什么是 CTE（公共表表达式）？

**答案：**

CTE（Common Table Expression）是定义临时结果集的方式，可以提高复杂查询的可读性。

```sql
-- 基本 CTE
WITH active_users AS (
    SELECT id, name FROM users WHERE status = 'active'
)
SELECT * FROM active_users WHERE name LIKE '张%';

-- 多个 CTE
WITH 
monthly_sales AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount) AS total
    FROM orders
    GROUP BY month
),
avg_sales AS (
    SELECT AVG(total) AS avg_total FROM monthly_sales
)
SELECT month, total, 
       total - avg_total AS diff_from_avg
FROM monthly_sales, avg_sales;

-- CTE 与窗口函数结合
WITH ranked_products AS (
    SELECT id, name, category, price,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn
    FROM products
)
SELECT * FROM ranked_products WHERE rn <= 3;  -- 每个类别前三名
```

**递归 CTE：**

```sql
-- 组织架构树形查询
WITH RECURSIVE org_tree AS (
    -- 基础查询：顶级员工
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- 递归查询：下属员工
    SELECT e.id, e.name, e.manager_id, t.level + 1
    FROM employees e
    JOIN org_tree t ON e.manager_id = t.id
)
SELECT * FROM org_tree ORDER BY level;

-- 路径查询
WITH RECURSIVE paths AS (
    SELECT id, ARRAY[id] AS path
    FROM nodes
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT n.id, p.path || n.id
    FROM nodes n
    JOIN paths p ON n.parent_id = p.id
)
SELECT * FROM paths;

-- 生成数字序列
WITH RECURSIVE nums AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 10
)
SELECT * FROM nums;
```

**追问：CTE 可以提高性能吗？**

**追问答案：**
PostgreSQL 12 之前，CTE 是优化屏障（optimization fence），会物化结果。PostgreSQL 12+ 默认会内联非递归 CTE。

```sql
-- PostgreSQL 12+ 默认行为
WITH cte AS (
    SELECT * FROM large_table WHERE status = 'active'
)
SELECT * FROM cte WHERE id = 1;
-- 优化器会将条件 id=1 推入 CTE

-- 强制物化
WITH cte AS MATERIALIZED (
    SELECT * FROM large_table
)
SELECT * FROM cte;

-- 强制内联
WITH cte AS NOT MATERIALIZED (
    SELECT * FROM large_table
)
SELECT * FROM cte;
```

---

## MVCC 与事务篇

### 1. PostgreSQL 的 MVCC 是怎么实现的？

**答案：**

PostgreSQL 使用 MVCC（多版本并发控制）来实现事务隔离，核心机制是在每行数据中保存版本信息。

**隐藏的系统列：**

每行数据有四个隐藏的系统列：

| 列名 | 说明 |
|------|------|
| xmin | 插入该行的事务 ID |
| xmax | 删除/更新该行的事务 ID |
| cmin | 命令序号（同一事务中的命令顺序） |
| cmax | 删除命令序号 |
| ctid | 行的物理位置 (页号, 行号) |

```sql
-- 查看隐藏列
SELECT xmin, xmax, ctid, * FROM users;

-- 示例输出
--  xmin  | xmax  |  ctid  | id | name
-- -------+-------+--------+----+------
--  1001  | 0     | (0,1)  | 1  | 张三
--  1002  | 1005  | (0,2)  | 2  | 李四   -- 被 txid 1005 删除
```

**版本链示意图：**

```
INSERT 操作：
事务 1001: INSERT INTO users (id, name) VALUES (1, '张三');
结果: id=1, name='张三', xmin=1001, xmax=0

UPDATE 操作：
事务 1002: UPDATE users SET name='李四' WHERE id=1;

PostgreSQL 的 UPDATE = INSERT 新版本 + DELETE 旧版本

旧版本: id=1, name='张三', xmin=1001, xmax=1002
新版本: id=1, name='李四', xmin=1002, xmax=0, ctid=(0,2)

DELETE 操作：
事务 1003: DELETE FROM users WHERE id=1;
旧版本: id=1, name='李四', xmin=1002, xmax=1003  -- 标记删除
```

**MVCC 可见性规则：**

```
一行数据对当前事务可见的条件：

1. xmin 已提交 且 xmin < 当前事务ID
2. xmax 为空(0) 或 xmax 未提交 或 xmax >= 当前事务ID

简单理解：
- 我能看到别人已提交的插入
- 我看不到别人未提交的修改
- 我能看到自己未提交的修改
```

**追问：MVCC 的优点是什么？**

**追问答案：**
1. **读不阻塞写**：读取历史版本，不阻塞写入
2. **写不阻塞读**：写入创建新版本，不阻塞读取
3. **无需 undo log**：历史版本就在表中
4. **快照一致性**：同一事务看到一致的快照

---

### 2. PostgreSQL 和 MySQL 的 MVCC 有什么区别？

**答案：**

| 特性 | PostgreSQL | MySQL (InnoDB) |
|------|------------|----------------|
| 版本存储 | 新旧版本都在表中 | 新版本在表中，旧版本在 undo log |
| 历史版本位置 | 表数据文件 | undo log 文件 |
| 回滚机制 | 直接读取旧版本 | 从 undo log 恢复 |
| 清理机制 | VACUUM 清理死元组 | purge 线程清理 undo log |
| 读操作 | 可能访问多个版本 | 访问最新版本 + undo log |
| 存储开销 | 需要定期 VACUUM | undo log 可自动清理 |
| 长事务影响 | 表膨胀（无法 VACUUM） | undo log 膨胀 |

**结构对比图：**

```
PostgreSQL MVCC：
┌─────────────────────────────────────────┐
│              表数据文件                   │
│  ┌─────────┐   ┌─────────┐              │
│  │ 版本 1   │ → │ 版本 2   │ → 新版本...  │
│  │ xmin=100│   │ xmin=101│              │
│  │ xmax=101│   │ xmax=0  │              │
│  └─────────┘   └─────────┘              │
└─────────────────────────────────────────┘
历史版本和数据在同一个文件

MySQL MVCC：
┌────────────────┐    ┌─────────────────┐
│   表数据文件    │    │   Undo Log      │
│  ┌───────────┐ │    │ ┌─────────────┐ │
│  │ 最新版本   │ │ ←──│ │ 历史版本 1   │ │
│  │ DB_TRX_ID │ │    │ │ 历史版本 2   │ │
│  │ DB_ROLL_PTR│─┼────┼→│ 历史版本 3   │ │
│  └───────────┘ │    │ └─────────────┘ │
└────────────────┘    └─────────────────┘
最新版本在表中，历史版本在 undo log
```

**追问：两种方式各有什么优缺点？**

**追问答案：**

**PostgreSQL 方式优点：**
- 历史版本就在表中，回滚快
- 不需要单独的 undo log 存储
- VACUUM 可以批量清理

**PostgreSQL 方式缺点：**
- 表会膨胀，需要定期 VACUUM
- 长事务会阻止 VACUUM
- 空间回收不如 MySQL 及时

**MySQL 方式优点：**
- 表数据相对紧凑
- undo log 可以自动清理
- 长事务影响相对可控

**MySQL 方式缺点：**
- 回滚需要读取 undo log
- undo log 空间管理复杂
- 极端情况下 undo log 膨胀

---

### 3. 为什么 PostgreSQL 不需要 undo log？

**答案：**

PostgreSQL 不使用 undo log 的原因与其 MVCC 实现方式有关：

**核心原因：历史版本直接存储在表中**

```
MySQL 的做法：
UPDATE table SET name='李四' WHERE id=1;

1. 将旧值写入 undo log
2. 修改表中数据为新值
3. 回滚时从 undo log 恢复

PostgreSQL 的做法：
UPDATE table SET name='李四' WHERE id=1;

1. 旧行：设置 xmax = 当前事务ID（标记删除）
2. 插入新行：包含新值，xmin = 当前事务ID
3. 回滚时：设置新行的 xmax，清除旧行的 xmax
```

**数据组织方式：**

```sql
-- PostgreSQL 中 UPDATE 后的数据
SELECT xmin, xmax, ctid, * FROM users WHERE id = 1;

-- 结果（假设 UPDATE 在事务 1002 中执行）：
-- xmin | xmax  | ctid  | id | name
-- -----+-------+-------+----+------
-- 1001 | 1002  | (0,1) | 1  | 张三    ← 旧版本（逻辑删除）
-- 1002 | 0     | (0,2) | 1  | 李四    ← 新版本

-- 读取时根据 xmin/xmax 判断可见性
-- 不需要单独的 undo log
```

**优点：**

```
1. 回滚快速
   - 只需要修改 xmax 标记
   - 不需要从 undo log 恢复

2. 无需管理 undo log
   - 没有 undo log 空间问题
   - 没有 undo log 崩溃恢复

3. 简化恢复逻辑
   - 崩溃恢复后数据直接可用
   - 未提交事务的行通过 xmax 判断
```

**代价：**

```
1. 需要定期 VACUUM 清理死元组
2. 表可能膨胀
3. 更新操作实际上是插入 + 删除
```

**追问：PostgreSQL 的回滚是如何工作的？**

**追问答案：**
PostgreSQL 回滚非常简单：

```sql
-- 事务回滚
BEGIN;
INSERT INTO users (name) VALUES ('王五');  -- xmin=1003, xmax=0
ROLLBACK;

-- 回滚时：
-- 1. 将新插入行的 xmax 设置为当前事务ID
-- 2. 或者简单标记为无效
-- 结果：xmin=1003, xmax=1003（自己删除自己，不可见）

-- UPDATE 回滚
BEGIN;
UPDATE users SET name='赵六' WHERE id=1;
-- 旧行：xmax=1004
-- 新行：xmin=1004
ROLLBACK;

-- 回滚时：
-- 1. 旧行的 xmax 清除（恢复可见）
-- 2. 新行的 xmax 设置为 1004（标记删除）
```

---

### 4. VACUUM 是什么？为什么需要？

**答案：**

VACUUM 是 PostgreSQL 清理"死元组"（dead tuples）的机制，释放空间供重用。

**为什么需要 VACUUM：**

```
PostgreSQL 的 UPDATE/DELETE 不立即回收空间：

1. DELETE 操作
   DELETE FROM users WHERE id = 1;
   -- 只是在行上设置 xmax，数据仍在表中
   
2. UPDATE 操作
   UPDATE users SET name = '李四' WHERE id = 1;
   -- 旧行设置 xmax，新行插入
   -- 旧行成为"死元组"

死元组积累会导致：
- 表膨胀，占用过多磁盘空间
- 扫描性能下降
- 索引也会膨胀
```

**VACUUM 的工作：**

```sql
-- 普通 VACUUM：标记空间可重用，不释放磁盘
VACUUM users;

-- VACUUM FULL：重写表，释放磁盘空间（锁表）
VACUUM FULL users;

-- VACUUM ANALYZE：同时更新统计信息
VACUUM ANALYZE users;
```

**VACUUM 工作原理：**

```
VACUUM 前：
┌───────────────────────────────────────┐
│ 活跃行 │ 死元组 │ 活跃行 │ 死元组 │ 活跃行 │
└───────────────────────────────────────┘

VACUUM 后（空间标记可重用）：
┌───────────────────────────────────────┐
│ 活跃行 │ [可重用] │ 活跃行 │ [可重用] │ 活跃行 │
└───────────────────────────────────────┘

VACUUM FULL 后（重写表）：
┌─────────────────────┐
│ 活跃行 │ 活跃行 │ 活跃行 │
└─────────────────────┘
```

**自动 VACUUM：**

```sql
-- 查看自动 VACUUM 配置
SHOW autovacuum;
SHOW autovacuum_vacuum_threshold;
SHOW autovacuum_vacuum_scale_factor;

-- 默认触发条件：
-- dead_tuples > autovacuum_vacuum_threshold + 
--               autovacuum_vacuum_scale_factor * n_live_tuples

-- 即：死元组数 > 50 + 0.2 * 活跃元组数

-- 手动触发 VACUUM
VACUUM VERBOSE users;  -- VERBOSE 显示详细信息
```

**追问：VACUUM 和 VACUUM FULL 有什么区别？**

**追问答案：**

| 特性 | VACUUM | VACUUM FULL |
|------|--------|-------------|
| 锁类型 | SHARE UPDATE EXCLUSIVE | ACCESS EXCLUSIVE（锁表） |
| 空间处理 | 标记可重用 | 释放给操作系统 |
| 表重写 | 否 | 是 |
| 执行时间 | 快 | 慢 |
| 阻塞操作 | 不阻塞读写 | 阻塞所有操作 |
| 索引处理 | 简单清理 | 重建所有索引 |
| 使用场景 | 日常维护 | 严重表膨胀时 |

```sql
-- 日常维护用 VACUUM
VACUUM ANALYZE users;

-- 表严重膨胀时用 VACUUM FULL（需要维护窗口）
VACUUM FULL VERBOSE users;
```

---

### 5. 什么是表膨胀？怎么解决？

**答案：**

表膨胀是指表中存在大量死元组，导致表的实际大小远大于有效数据量。

**膨胀原因：**

```
1. 大量 UPDATE/DELETE 操作
   - 每次 UPDATE 产生新的死元组
   - DELETE 不立即释放空间

2. 长事务阻止 VACUUM
   - 长事务持有快照，VACUUM 无法清理
   - 死元组持续积累

3. autovacuum 配置不当
   - 触发阈值过高
   - 清理速度跟不上产生速度

4. 高并发更新热点
   - 同一行频繁更新
   - autovacuum 来不及清理
```

**检测表膨胀：**

```sql
-- 方法1：使用 pgstattuple 扩展
CREATE EXTENSION pgstattuple;
SELECT * FROM pgstattuple('users');

-- 输出：
-- dead_tuple_count: 100000
-- dead_tuple_percent: 45.5%

-- 方法2：估算膨胀率
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_ratio
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- 方法3：比较实际大小与估算大小
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::text)) AS actual_size,
    pg_size_pretty(pg_relation_size(tablename::text)) AS table_size
FROM pg_tables WHERE schemaname = 'public';
```

**解决方案：**

```sql
-- 1. 轻度膨胀：手动 VACUUM
VACUUM VERBOSE ANALYZE users;

-- 2. 中度膨胀：VACUUM FULL（需要锁表）
VACUUM FULL VERBOSE users;

-- 3. 重度膨胀：使用 pg_repack（在线重组，不锁表）
-- pg_repack 是扩展，需要单独安装
-- pg_repack -t users -d mydb

-- 4. 调整 autovacuum 参数
ALTER TABLE hot_table SET (
    autovacuum_vacuum_scale_factor = 0.05,  -- 降低触发阈值
    autovacuum_vacuum_cost_delay = 2        -- 加快清理速度
);
```

**预防措施：**

```sql
-- 1. 监控长事务
SELECT pid, 
       usename,
       state,
       now() - xact_start AS duration,
       query
FROM pg_stat_activity
WHERE state IN ('idle in transaction', 'active')
  AND now() - xact_start > INTERVAL '10 minutes';

-- 2. 设置事务超时
SET idle_in_transaction_session_timeout = '10min';

-- 3. 优化 autovacuum 配置
-- 在 postgresql.conf 中
autovacuum_vacuum_scale_factor = 0.1
autovacuum_vacuum_cost_limit = 1000
autovacuum_max_workers = 3
```

**追问：pg_repack 是什么？有什么优势？**

**追问答案：**
pg_repack 是 PostgreSQL 扩展，可以在不锁表的情况下重组表：

```
pg_repack 工作原理：
1. 创建新表，复制数据
2. 通过触发器同步增量变化
3. 最后短暂锁表，完成切换

优势：
- 最小化锁表时间
- 在线重建表
- 回收空间给操作系统
- 重建索引

使用：
pg_repack -t tablename -d databasename
```

---

### 6. xmin/xmax 是什么？

**答案：**

xmin 和 xmax 是 PostgreSQL 每行数据的隐藏系统列，用于 MVCC 版本控制。

**xmin（插入事务ID）：**

```
含义：创建该行的事务 ID
- INSERT 时设置为当前事务 ID
- 表示这行数据是哪个事务创建的
```

**xmax（删除事务ID）：**

```
含义：删除/更新该行的事务 ID
- DELETE 时设置为当前事务 ID
- UPDATE 时设置为当前事务 ID（标记旧行删除）
- 0 表示行未被删除
```

**详细示例：**

```sql
-- 查看隐藏列
SELECT xmin, xmax, ctid, * FROM users;

-- 场景1：INSERT
BEGIN;
-- txid = 1001
INSERT INTO users (id, name) VALUES (1, '张三');
-- 新行：xmin=1001, xmax=0, id=1, name='张三'
COMMIT;

-- 场景2：UPDATE
BEGIN;
-- txid = 1002
UPDATE users SET name = '李四' WHERE id = 1;
-- 旧行：xmin=1001, xmax=1002, id=1, name='张三'（标记删除）
-- 新行：xmin=1002, xmax=0, id=1, name='李四'
COMMIT;

-- 场景3：DELETE
BEGIN;
-- txid = 1003
DELETE FROM users WHERE id = 1;
-- 行：xmin=1002, xmax=1003（标记删除）
ROLLBACK;
-- 回滚后：xmax 恢复为 0
```

**ctid（行位置）：**

```sql
-- ctid = (页号, 行号)
SELECT ctid, * FROM users;
-- ctid = (0, 1) 表示第0页第1行
-- ctid = (0, 2) 表示第0页第2行

-- UPDATE 后 ctid 会变化
UPDATE users SET name = '王五' WHERE id = 1;
-- 旧行 ctid = (0, 1)
-- 新行 ctid = (0, 2)
```

**追问：如何通过 xmin/xmax 判断数据可见性？**

**追问答案：**

```sql
-- 简化的可见性判断逻辑
-- 事务 txid 查看某行是否可见：

-- 规则1：行被自己插入，可见
xmin = txid 且 xmin 已提交 → 可见

-- 规则2：行被已提交事务插入，且未被删除
xmin 已提交 且 xmin < txid 且 xmax = 0 → 可见

-- 规则3：行被删除但删除事务未提交
xmin 已提交 且 xmax 未提交 → 可见

-- 规则4：行被删除且删除事务已提交
xmin 已提交 且 xmax 已提交 且 xmax < txid → 不可见

-- 实际实现更复杂，需要考虑：
-- 1. 事务快照
-- 2. 已提交事务列表
-- 3. 当前事务内的命令序号
```

---

### 7. 什么是事务快照？

**答案：**

事务快照是 PostgreSQL 用来判断数据可见性的关键数据结构，记录了某一时刻所有活跃事务的状态。

**快照内容：**

```
快照包含：
- xmin: 最小的活跃事务 ID
- xmax: 下一个要分配的事务 ID
- xip_list: 当前活跃事务 ID 列表

含义：
- xmin 之前的事务都已提交
- xmax 及之后的事务都未开始
- xip_list 中的事务正在运行但未提交
```

**快照类型：**

```sql
-- 查看当前快照
SELECT pg_current_snapshot();

-- 输出示例：
-- 100:105:100,102,104
-- 含义：xmin=100, xmax=105, 活跃事务=[100,102,104]

-- 不同隔离级别的快照行为：
-- Read Committed：每次查询获取新快照
-- Repeatable Read/Serializable：事务开始时获取快照，整个事务复用
```

**快照与可见性判断：**

```
事务ID: ... 98 99 100 101 102 103 104 105 ...
              ↑               ↑
            xmin           xmax
              
活跃事务: 100, 102, 104

判断某行 xmin=101, xmax=0 的可见性：
1. xmin(101) >= snapshot.xmin(100) → 在范围内
2. xmin(101) < snapshot.xmax(105) → 在范围内
3. xmin(101) 不在 xip_list 中 → 已提交
4. xmax = 0 → 未被删除
结论：可见

判断某行 xmin=102, xmax=0 的可见性：
1. xmin(102) 在 xip_list 中 → 活跃未提交
结论：不可见
```

**快照使用示例：**

```sql
-- Read Committed：每次查询新快照
BEGIN;  -- txid = 100
SELECT * FROM users;  -- 快照1
-- 其他事务提交了新数据
SELECT * FROM users;  -- 快照2，可能看到新数据
COMMIT;

-- Repeatable Read：整个事务一个快照
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;  -- txid = 100
SELECT * FROM users;  -- 快照1
-- 其他事务提交了新数据
SELECT * FROM users;  -- 仍用快照1，看不到新数据
COMMIT;

-- 导出/导入快照
BEGIN;
SELECT pg_export_snapshot();  -- 返回快照ID '00000A-1'
-- 其他会话可以使用这个快照
SET TRANSACTION SNAPSHOT '00000A-1';
COMMIT;
```

**追问：xmin 在快照中是什么含义？**

**追问答案：**
快照中的 `xmin` 表示：在此快照时刻，所有 ID 小于 xmin 的事务都已提交完成。

```
快照: xmin=100, xmax=105, xip=[100,102,104]

含义：
- 事务 ID < 100 的事务：全部已提交
- 事务 ID >= 105 的事务：在快照时刻还未开始
- 事务 ID 在 100-104 之间：
  - 100, 102, 104 在活跃列表中：未提交
  - 101, 103 不在活跃列表中：已提交

这个设计避免了检查每个事务的提交状态，
只需比较 ID 和查找活跃列表即可判断可见性。
```

---

### 8. MVCC 可见性判断规则是什么？

**答案：**

MVCC 可见性判断是 PostgreSQL 事务隔离的核心，决定了一个事务能看到哪些版本的数据。

**基本规则：**

```
一行数据对事务 T 可见的条件：

1. 插入可见性（xmin）
   - xmin 已提交
   - xmin 不在活跃事务列表中
   - xmin < T 的快照 xmax

2. 删除可见性（xmax）
   - xmax = 0（未被删除）
   - 或 xmax 未提交
   - 或 xmax 在活跃事务列表中
   - 或 xmax >= T 的快照 xmax
```

**详细判断流程：**

```sql
-- 伪代码表示可见性判断
FUNCTION is_visible(tuple, snapshot):
    -- 检查插入事务 xmin
    IF tuple.xmin NOT committed THEN
        -- 插入事务未提交
        IF tuple.xmin == current_transaction THEN
            -- 自己插入的，可见（但需要检查 xmax）
            RETURN check_delete(tuple, snapshot)
        ELSE
            -- 别人未提交的，不可见
            RETURN FALSE
        END IF
    END IF
    
    -- xmin 已提交
    IF tuple.xmin >= snapshot.xmax THEN
        -- 在快照之后提交，不可见
        RETURN FALSE
    END IF
    
    IF tuple.xmin IN snapshot.xip_list THEN
        -- 在快照时活跃，不可见
        RETURN FALSE
    END IF
    
    -- xmin 可见，检查 xmax
    RETURN check_delete(tuple, snapshot)

FUNCTION check_delete(tuple, snapshot):
    IF tuple.xmax == 0 THEN
        -- 未被删除，可见
        RETURN TRUE
    END IF
    
    IF tuple.xmax NOT committed THEN
        -- 删除事务未提交
        IF tuple.xmax == current_transaction THEN
            -- 自己删除的，不可见
            RETURN FALSE
        ELSE
            -- 别人未提交删除，仍可见
            RETURN TRUE
        END IF
    END IF
    
    -- xmax 已提交
    IF tuple.xmax >= snapshot.xmax THEN
        -- 在快照之后提交删除，仍可见
        RETURN TRUE
    END IF
    
    IF tuple.xmax IN snapshot.xip_list THEN
        -- 在快照时活跃，仍可见
        RETURN TRUE
    END IF
    
    -- 已被删除，不可见
    RETURN FALSE
```

**示例分析：**

```sql
-- 假设快照: xmin=100, xmax=105, xip=[100,102,104]
-- 当前事务 ID = 103

-- 行1: xmin=99, xmax=0
-- 99 < 100 (xmin)，已提交
-- xmax=0，未被删除
-- 结果：可见 ✓

-- 行2: xmin=100, xmax=0
-- 100 在 xip 中，活跃未提交
-- 结果：不可见 ✗

-- 行3: xmin=101, xmax=0
-- 101 已提交（不在 xip 中，且 < xmax）
-- 结果：可见 ✓

-- 行4: xmin=98, xmax=103
-- 98 已提交
-- xmax=103 = 当前事务（自己删除的）
-- 结果：不可见 ✗

-- 行5: xmin=98, xmax=102
-- 98 已提交
-- 102 在 xip 中，删除事务未提交
-- 结果：可见 ✓
```

**追问：同一事务内的多个命令如何保证可见性？**

**追问答案：**
PostgreSQL 使用 cmin/cmax 来标记同一事务内的命令顺序：

```sql
BEGIN;
-- 命令1（cmin=0）
INSERT INTO users (id, name) VALUES (1, '张三');
-- 命令2（cmin=1）
SELECT * FROM users;  -- 能看到张三（cmin=0 < 当前 cmin=1）
-- 命令3（cmin=2）
UPDATE users SET name = '李四' WHERE id = 1;
-- 命令4（cmin=3）
SELECT * FROM users;  -- 能看到李四

-- cmin/cmax 规则：
-- - 只能看到 cmin < 当前命令序号的插入
-- - 只能看到 cmax >= 当前命令序号的删除
```

---

### 9. 什么是事务隔离级别？PostgreSQL 如何实现？

**答案：**

PostgreSQL 支持四种标准隔离级别，但内部实现略有不同。

**隔离级别对比：**

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | PostgreSQL 实现 |
|----------|------|------------|------|-----------------|
| READ UNCOMMITTED | ❌ | ❌ | ❌ | 等同于 READ COMMITTED |
| READ COMMITTED | ✅ | ❌ | ❌ | 每次查询新快照 |
| REPEATABLE READ | ✅ | ✅ | ❌* | 整个事务一个快照 |
| SERIALIZABLE | ✅ | ✅ | ✅ | SSI（可串行化快照隔离）|

*PostgreSQL 的 RR 级别实际上也能防止幻读。

**实现机制：**

```sql
-- READ COMMITTED
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- 每条 SQL 语句开始时获取新快照
SELECT * FROM users WHERE id = 1;  -- 快照1
-- 其他事务修改了 id=1 并提交
SELECT * FROM users WHERE id = 1;  -- 快照2，能看到修改
COMMIT;

-- REPEATABLE READ
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- 事务第一条 SQL 时获取快照，整个事务复用
SELECT * FROM users WHERE id = 1;  -- 快照1
-- 其他事务修改了 id=1 并提交
SELECT * FROM users WHERE id = 1;  -- 仍用快照1，看不到修改
COMMIT;

-- SERIALIZABLE
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 使用 SSI（Serializable Snapshot Isolation）
-- 检测读写冲突，必要时回滚
SELECT * FROM users WHERE id = 1;
UPDATE users SET name = 'test' WHERE id = 1;
-- 如果检测到串行化冲突，会报错并回滚
COMMIT;
```

**PostgreSQL 特殊行为：**

```sql
-- READ UNCOMMITTED 实际上是 READ COMMITTED
SHOW default_transaction_isolation;
-- PostgreSQL 不支持真正的脏读

-- REPEATABLE READ 防止幻读
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM users WHERE age > 20;  -- 返回2条
-- 其他事务插入了 age=25 并提交
SELECT * FROM users WHERE age > 20;  -- 仍返回2条（无幻读）
COMMIT;
```

**追问：PostgreSQL 的 SERIALIZABLE 如何实现？**

**追问答案：**
PostgreSQL 使用 SSI（Serializable Snapshot Isolation）实现可串行化：

```
SSI 原理：
1. 跟踪事务的读写依赖
2. 检测可能导致串行化异常的依赖环
3. 发现异常时，回滚其中一个事务

常见的串行化异常：
- 写偏序（Write Skew）
- 谓词锁

示例：
事务A: SELECT * FROM users WHERE status='active'
事务B: SELECT * FROM users WHERE status='active'
事务A: UPDATE users SET status='inactive' WHERE id=1
事务B: UPDATE users SET status='inactive' WHERE id=2
-- 两个事务都基于相同的查询结果，可能导致数据不一致

PostgreSQL 会检测到这种情况并回滚其中一个事务。
```

---

### 10. 什么是 HOT 更新？

**答案：**

HOT（Heap-Only Tuple）是 PostgreSQL 的优化机制，用于提高 UPDATE 性能。

**问题背景：**

```
普通 UPDATE 的开销：
1. 在表中插入新版本
2. 更新所有索引指向新版本
3. 旧索引项成为垃圾

如果有多个索引，每次 UPDATE 都要更新所有索引，开销很大！
```

**HOT 更新条件：**

```
当 UPDATE 满足以下条件时，可以使用 HOT：
1. 新元组和旧元组在同一个数据页
2. 没有更新索引列（索引列的值没变）
3. 不是 TOAST 字段更新

HOT 更新效果：
- 不更新索引
- 索引仍指向旧行，通过行指针链找到新行
- 大幅减少 I/O 和索引维护开销
```

**HOT 结构示意图：**

```
普通更新：
索引 → 新行（id=1, name='李四'）
索引 → 旧行（id=1, name='张三'）← 死元组

HOT 更新：
索引 → 行指针 → 旧行 → 新行
                    ↓
               HOT 链（同一页内）

索引仍然指向原来的行指针，
通过 HOT 链找到最新版本。
```

**代码示例：**

```sql
-- 创建测试表
CREATE TABLE hot_test (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    status VARCHAR(20)
);

CREATE INDEX idx_status ON hot_test(status);

-- HOT 更新（status 没变）
UPDATE hot_test SET name = '李四' WHERE id = 1;
-- 只更新 name，索引 idx_status 不需要更新

-- 非 HOT 更新（status 变了）
UPDATE hot_test SET status = 'inactive' WHERE id = 1;
-- status 变了，idx_status 需要更新

-- 查看 HOT 更新统计
SELECT n_tup_upd, n_tup_hot_upd 
FROM pg_stat_user_tables 
WHERE relname = 'hot_test';

-- n_tup_hot_upd / n_tup_upd 比例越高越好
```

**优化建议：**

```sql
-- 1. 填充因子设置，留出空间用于 HOT
ALTER TABLE hot_test SET (fillfactor = 80);
-- 80% 填充，留 20% 空间用于 HOT 更新

-- 2. 避免频繁更新索引列
-- 把频繁更新的列单独放表，不建索引

-- 3. 监控 HOT 比例
SELECT 
    schemaname,
    relname,
    n_tup_upd,
    n_tup_hot_upd,
    ROUND(n_tup_hot_upd * 100.0 / NULLIF(n_tup_upd, 0), 2) AS hot_ratio
FROM pg_stat_user_tables
WHERE n_tup_upd > 0
ORDER BY hot_ratio;
```

**追问：如何提高 HOT 更新比例？**

**追问答案：**

```sql
-- 1. 设置合适的 fillfactor
ALTER TABLE frequent_update_table SET (fillfactor = 70);
-- 留出更多空间用于 HOT

-- 2. 使用 CLUSTER 整理表
CLUSTER frequent_update_table USING pk_index;
-- 使相关数据物理上相邻

-- 3. 设计上避免频繁更新索引列
-- 把状态等频繁变更的字段放单独表

-- 4. 监控并及时 VACUUM
-- 死元组会阻碍 HOT 链的形成
```

---

### 11. PostgreSQL 中的 WAL 是什么？

**答案：**

WAL（Write-Ahead Logging，预写日志）是 PostgreSQL 保证数据持久性的核心机制。

**WAL 原理：**

```
WAL 核心思想：先写日志，再写数据

1. 数据修改时，先写入 WAL 日志
2. WAL 写入成功后，事务才能提交
3. 数据页可以延迟写入磁盘
4. 崩溃后通过 WAL 重放恢复数据

优势：
- 顺序写 WAL 比随机写数据页快很多
- 减少磁盘 I/O
- 保证 ACID 的持久性
```

**WAL 结构：**

```
WAL 文件位于: $PGDATA/pg_wal/

文件命名: 000000010000000000000001
          |时间线|   |日志序号  |

每个文件默认 16MB
循环写入，checkpoint 后可以复用
```

**WAL 配置：**

```sql
-- 查看 WAL 配置
SHOW wal_level;           -- minimal/replica/logical
SHOW max_wal_size;        -- WAL 最大大小
SHOW min_wal_size;        -- WAL 最小大小
SHOW wal_keep_size;       -- 保留的 WAL 大小（用于复制）

-- 设置 WAL 级别
-- minimal: 最少日志，不支持复制
-- replica: 支持主从复制
-- logical: 支持逻辑复制
ALTER SYSTEM SET wal_level = replica;

-- 同步提交配置
SHOW synchronous_commit;
-- on: 等待 WAL 刷盘后再提交（默认，安全）
-- off: 不等待 WAL 刷盘（快但可能丢数据）
-- remote_write: 等待备库写入 OS 缓存
-- remote_apply: 等待备库应用完成
```

**WAL 与恢复：**

```sql
-- 时间点恢复（PITR）
-- 1. 基础备份
pg_basebackup -D /backup/base -Fp -Xs -P

-- 2. 持续归档 WAL
archive_mode = on
archive_command = 'cp %p /archive/%f'

-- 3. 恢复到指定时间点
-- 在 recovery.conf 中设置
restore_command = 'cp /archive/%f %p'
recovery_target_time = '2024-01-15 10:00:00'
```

**追问：WAL 和 Checkpoint 的关系是什么？**

**追问答案：**

```
Checkpoint 的作用：
1. 将内存中的脏页刷入磁盘
2. 更新检查点位置
3. 之前的 WAL 可以回收

关系：
              Checkpoint 位置
                    ↓
WAL: [已刷盘数据] [可以重放] [正在写入]
                   ↑
              可能需要恢复的数据

Checkpoint 之后的数据已经写入磁盘，
恢复时只需要从 Checkpoint 开始重放 WAL。

配置：
checkpoint_timeout = 10min      -- 自动 checkpoint 间隔
max_wal_size = 1GB              -- WAL 最大大小触发 checkpoint
checkpoint_completion_target = 0.9  -- 分散写入，避免 I/O 峰值
```

---

### 12. 什么是 Freeze 操作？

**答案：**

Freeze 是 PostgreSQL 处理事务 ID 回卷（wraparound）的机制。

**事务 ID 回卷问题：**

```
PostgreSQL 事务 ID 是 32 位无符号整数：
- 范围: 0 ~ 4,294,967,295 (约 42 亿)
- 循环使用: ... → 42亿 → 0 → 1 → ...

问题：
- 如果不处理，事务 ID 回卷后会导致可见性判断错误
- 比如: xmin=100 在 xmin=42亿 看起来比 xmin=50 "更新"

解决方案：Freeze
- 将旧事务 ID 标记为 "frozen"
- Frozen 事务对所有事务可见
- 避免回卷问题
```

**Freeze 机制：**

```sql
-- 事务 ID 分区
-- 32 位事务 ID 分为三个区域：

|<-- 过去 -->|<-- 未来 -->|<-- 回卷区 -->|
0          20亿         42亿

-- 当事务 ID 接近 20 亿时，需要 freeze 旧数据

-- 查看 freeze 配置
SHOW vacuum_freeze_min_age;      -- 默认 5000 万
SHOW vacuum_freeze_table_age;    -- 默认 1.5 亿
SHOW autovacuum_freeze_max_age;  -- 默认 2 亿

-- 当表的年龄（pg_class.relfrozenxid）达到 vacuum_freeze_table_age 时，
-- 会强制进行 freeze
```

**Freeze 过程：**

```sql
-- 查看表的年龄
SELECT 
    relname,
    age(relfrozenxid) as xid_age,
    pg_size_pretty(pg_total_relation_size(oid)) as size
FROM pg_class
WHERE relkind = 'r'
ORDER BY xid_age DESC;

-- 手动触发 freeze
VACUUM FREEZE tablename;

-- Freeze 的效果：
-- 将 xmin 替换为 FrozenTransactionId (通常是 2)
-- 表示这些行对所有事务都可见
```

**监控 Freeze：**

```sql
-- 查看数据库年龄
SELECT 
    datname,
    age(datfrozenxid) as xid_age
FROM pg_database
ORDER BY xid_age DESC;

-- 年龄接近 20 亿时需要关注
-- PostgreSQL 会在 autovacuum_freeze_max_age 时强制 freeze

-- 查看是否正在进行 freeze
SELECT * FROM pg_stat_progress_vacuum;
```

**追问：为什么 Freeze 会影响性能？**

**追问答案：**

```
Freeze 对性能的影响：

1. 全表扫描
   - Freeze 需要扫描整个表
   - 大表可能需要很长时间

2. 写放大
   - Freeze 会更新每行的 xmin
   - 即使行没有变化也要写入

3. 阻塞问题
   - 如果 freeze 来不及做，数据库会进入保护模式
   - 只允许 freeze 操作，拒绝正常写入

优化建议：
1. 调整 freeze 参数，提前进行 freeze
2. 监控表年龄，在低峰期手动 freeze
3. 对大表进行分区，减少单次 freeze 时间

ALTER SYSTEM SET autovacuum_freeze_max_age = 150000000;
```

---

## 索引篇

### 1. PostgreSQL 有哪些索引类型？

**答案：**

PostgreSQL 提供了丰富的索引类型，适用于不同场景：

| 索引类型 | 适用场景 | 示例 |
|----------|----------|------|
| B-tree | 等值、范围、排序查询 | 默认索引类型 |
| Hash | 仅等值查询 | 少用，有缺陷 |
| GiST | 地理空间、范围、几何类型 | PostGIS |
| GIN | 数组、JSONB、全文搜索 | 包含查询 |
| BRIN | 超大表、有序数据 | 时序数据 |
| SP-GiST | 非平衡数据结构 | 四叉树、基数树 |

```sql
-- B-tree（默认）
CREATE INDEX idx_name ON users(name);

-- Hash
CREATE INDEX idx_name_hash ON users USING HASH(name);

-- GiST
CREATE INDEX idx_location ON stores USING GIST(location);

-- GIN
CREATE INDEX idx_tags ON articles USING GIN(tags);
CREATE INDEX idx_content ON articles USING GIN(to_tsvector('english', content));

-- BRIN
CREATE INDEX idx_created ON logs USING BRIN(created_at);

-- SP-GiST
CREATE INDEX idx_quad ON points USING SPGIST(point);
```

**各索引类型详解：**

```sql
-- 1. B-tree：最常用的索引
-- 支持: =, <, >, <=, >=, BETWEEN, IN, LIKE 'xxx%'
CREATE INDEX idx_age ON users(age);
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE name LIKE '张%';

-- 2. GIN：Generalized Inverted Index（倒排索引）
-- 适用: 数组、JSONB、全文搜索
CREATE INDEX idx_tags ON posts USING GIN(tags);
SELECT * FROM posts WHERE tags @> ARRAY['postgresql'];
SELECT * FROM posts WHERE tags && ARRAY['postgresql', 'mysql'];

CREATE INDEX idx_data ON docs USING GIN(data jsonb_path_ops);
SELECT * FROM docs WHERE data @> '{"name": "张三"}';

-- 3. GiST：Generalized Search Tree
-- 适用: 地理空间、范围类型、几何操作
CREATE EXTENSION postgis;
CREATE INDEX idx_geom ON locations USING GIST(geom);
SELECT * FROM locations WHERE ST_DWithin(geom, point, 1000);

-- 4. BRIN：Block Range Index
-- 适用: 超大表、自然排序的数据
CREATE INDEX idx_time ON logs USING BRIN(created_at);
-- 存储：每个数据块的 min/max 值
-- 优点：索引极小，适合时序数据
```

**追问：如何选择索引类型？**

**追问答案：**

```
选择决策树：

1. 是否是等值查询？
   是 → 考虑 Hash（但 B-tree 通常更好）
   否 → 继续

2. 是否是范围/排序查询？
   是 → B-tree

3. 数据类型是什么？
   数组/JSONB/全文 → GIN
   地理空间/范围 → GiST
   超大表且有序 → BRIN

4. 查询模式是什么？
   包含查询(@>, &&) → GIN
   最近邻搜索 → GiST 或 SP-GiST
```

---

### 2. B-tree、GIN、GiST、BRIN 各适合什么场景？

**答案：**

**B-tree 索引：**

```sql
-- 特点
-- 1. 默认索引类型
-- 2. 支持: =, <, >, <=, >=, BETWEEN, IN, LIKE 'xxx%'
-- 3. 自动用于主键和唯一约束

-- 适用场景
CREATE INDEX idx_price ON products(price);
SELECT * FROM products WHERE price BETWEEN 100 AND 500;
SELECT * FROM products ORDER BY price DESC;

CREATE INDEX idx_name ON users(name);
SELECT * FROM users WHERE name = '张三';
SELECT * FROM users WHERE name LIKE '张%';  -- 可以用索引

-- 复合 B-tree 索引
CREATE INDEX idx_cat_price ON products(category, price);
SELECT * FROM products WHERE category = 'electronics' AND price > 100;
-- 最左前缀原则

-- 注意：LIKE '%张' 不能用 B-tree
SELECT * FROM users WHERE name LIKE '%三';  -- 全表扫描
```

**GIN 索引：**

```sql
-- 特点
-- 1. 倒排索引结构
-- 2. 支持包含查询(@>, &&, ?)
-- 3. 索引项可能很多，更新较慢

-- 场景1：数组查询
CREATE INDEX idx_tags ON posts USING GIN(tags);
SELECT * FROM posts WHERE tags @> ARRAY['postgresql', 'database'];
SELECT * FROM posts WHERE tags && ARRAY['java', 'python'];

-- 场景2：JSONB 查询
CREATE INDEX idx_data ON docs USING GIN(data);
CREATE INDEX idx_data_path ON docs USING GIN(data jsonb_path_ops);  -- 更快

SELECT * FROM docs WHERE data @> '{"name": "张三"}';
SELECT * FROM docs WHERE data->'tags' ? 'postgresql';

-- 场景3：全文搜索
CREATE INDEX idx_fts ON articles USING GIN(to_tsvector('english', content));
SELECT * FROM articles WHERE to_tsvector('english', content) @@ to_tsquery('postgresql & index');

-- 使用 pg_trgm 模糊搜索
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_name_trgm ON users USING GIN(name gin_trgm_ops);
SELECT * FROM users WHERE name LIKE '%张%';  -- 可以用索引
```

**GiST 索引：**

```sql
-- 特点
-- 1. 平衡树结构
-- 2. 支持自定义操作符
-- 3. 适合空间数据和范围类型

-- 场景1：地理空间
CREATE EXTENSION postgis;
CREATE INDEX idx_geom ON locations USING GIST(geom);
SELECT * FROM locations WHERE ST_DWithin(geom, ST_MakePoint(116.4, 39.9), 0.01);
SELECT * FROM locations ORDER BY geom <-> ST_MakePoint(116.4, 39.9) LIMIT 10;

-- 场景2：范围类型
CREATE INDEX idx_during ON reservations USING GIST(during);
SELECT * FROM reservations WHERE during && TSRANGE('[2024-01-01, 2024-01-05]');

-- 场景3：排除约束
CREATE TABLE meetings (
    room_id INT,
    during TSRANGE,
    EXCLUDE USING GIST (room_id WITH =, during WITH &&)
);
-- 约束：同一会议室的时间段不能重叠
```

**BRIN 索引：**

```sql
-- 特点
-- 1. Block Range Index，存储每个数据块的摘要
-- 2. 索引极小（KB 级别 vs B-tree 的 GB 级别）
-- 3. 适合自然排序的超大表

-- 适用场景
-- 时序数据、日志数据、按时间插入的数据
CREATE TABLE logs (
    id SERIAL,
    created_at TIMESTAMP,
    message TEXT
);

CREATE INDEX idx_created_brin ON logs USING BRIN(created_at);
-- 索引存储：每128个块的 min/max created_at

SELECT * FROM logs WHERE created_at BETWEEN '2024-01-01' AND '2024-01-02';
-- 先通过 BRIN 定位可能的数据块，再扫描这些块

-- BRIN 参数
CREATE INDEX idx_created_brin ON logs USING BRIN(created_at) WITH (pages_per_range = 128);
-- pages_per_range: 每多少个块记录一次摘要

-- 对比
-- B-tree: 1亿行数据，索引约 2GB
-- BRIN: 1亿行数据，索引约 100KB
```

**对比总结：**

| 索引类型 | 索引大小 | 写入性能 | 查询性能 | 适用场景 |
|----------|----------|----------|----------|----------|
| B-tree | 中 | 中 | 高 | 通用场景 |
| GIN | 大 | 慢 | 高 | 数组、JSONB、全文 |
| GiST | 中 | 中 | 中 | 地理空间、范围 |
| BRIN | 极小 | 快 | 中（需扫描） | 超大表、有序数据 |

---

### 3. 什么是部分索引（Partial Index）？

**答案：**

部分索引是只对表中满足特定条件的行创建的索引，可以减少索引大小和维护开销。

```sql
-- 语法：WHERE 子句指定索引条件
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- 只对 status='active' 的行创建索引
-- 索引大小更小，维护开销更低

-- 查询时自动使用
SELECT * FROM users WHERE email = 'test@example.com' AND status = 'active';
-- 可以使用 idx_active_users

SELECT * FROM users WHERE email = 'test@example.com';
-- 不一定使用索引（优化器可能判断全表扫描更好）
```

**使用场景：**

```sql
-- 场景1：只索引有效数据
CREATE INDEX idx_valid_orders ON orders(user_id) WHERE status != 'cancelled';
-- 大量已取消订单不需要索引

-- 场景2：索引热点数据
CREATE INDEX idx_recent_logs ON logs(level, created_at) 
WHERE created_at > CURRENT_DATE - INTERVAL '30 days';
-- 只索引最近30天的日志

-- 场景3：索引非空值
CREATE INDEX idx_phone ON users(phone) WHERE phone IS NOT NULL;
-- 只索引有电话号码的用户

-- 场景4：分区式索引
CREATE INDEX idx_orders_2024 ON orders(created_at) 
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
```

**性能对比：**

```sql
-- 假设 users 表有 100万行，其中 10万是 active

-- 全表索引
CREATE INDEX idx_email_full ON users(email);
-- 索引大小：约 30MB

-- 部分索引
CREATE INDEX idx_email_partial ON users(email) WHERE status = 'active';
-- 索引大小：约 3MB

-- 性能测试
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com' AND status = 'active';
-- 使用部分索引，扫描行数更少
```

**追问：部分索引有什么限制？**

**追问答案：**

```sql
-- 限制1：查询条件必须匹配索引条件
SELECT * FROM users WHERE email = 'test@example.com';
-- 不一定能用部分索引（缺少 status = 'active' 条件）

-- 限制2：条件必须是不可变的
-- 错误：
CREATE INDEX idx_recent ON logs(created_at) WHERE created_at > now();
-- now() 是可变函数，不能用于部分索引

-- 正确：
CREATE INDEX idx_recent ON logs(created_at) WHERE created_at > '2024-01-01';

-- 限制3：复杂条件可能不支持
-- 部分索引条件必须是简单的布尔表达式
-- 不能包含子查询、聚合函数等
```

---

### 4. 什么是表达式索引？

**答案：**

表达式索引是对列经过函数或表达式计算后的结果创建索引。

**问题背景：**

```sql
-- 假设经常查询小写的邮箱
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';

-- 普通索引无法使用
CREATE INDEX idx_email ON users(email);
EXPLAIN SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
-- Seq Scan（全表扫描）

-- 原因：索引存储的是原始值，不是 LOWER(email) 的结果
```

**解决方案：表达式索引**

```sql
-- 创建表达式索引
CREATE INDEX idx_email_lower ON users(LOWER(email));

-- 查询可以使用索引
EXPLAIN SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
-- Index Scan using idx_email_lower
```

**常见使用场景：**

```sql
-- 场景1：大小写不敏感查询
CREATE INDEX idx_name_lower ON users(LOWER(name));
SELECT * FROM users WHERE LOWER(name) = 'zhang san';

-- 场景2：日期函数
CREATE INDEX idx_created_date ON orders(DATE(created_at));
SELECT * FROM orders WHERE DATE(created_at) = '2024-01-15';

-- 场景3：JSONB 路径
CREATE INDEX idx_data_name ON users((data->>'name'));
SELECT * FROM users WHERE data->>'name' = '张三';

-- 场景4：计算字段
CREATE INDEX idx_full_name ON users(CONCAT(first_name, ' ', last_name));
SELECT * FROM users WHERE CONCAT(first_name, ' ', last_name) = 'Zhang San';

-- 场景5：数组长度
CREATE INDEX idx_tags_count ON posts(ARRAY_LENGTH(tags, 1));
SELECT * FROM posts WHERE ARRAY_LENGTH(tags, 1) > 5;

-- 场景6：复合表达式索引
CREATE INDEX idx_user_date ON logs((user_id::text || '-' || DATE(created_at)::text));
```

**注意事项：**

```sql
-- 1. 写入性能影响
-- 每次插入/更新都要计算表达式
-- 复杂表达式会降低写入性能

-- 2. 索引大小
-- 表达式结果可能比原列更大

-- 3. 必须使用相同的表达式
-- 索引：CREATE INDEX idx_lower ON users(LOWER(name));
-- 有效：WHERE LOWER(name) = 'test'
-- 无效：WHERE name = 'test'（不匹配表达式）

-- 4. 可以组合普通列和表达式
CREATE INDEX idx_status_lower ON users(status, LOWER(email));
```

**追问：表达式索引和生成列有什么区别？**

**追问答案：**

```sql
-- 生成列（Generated Column）
ALTER TABLE users ADD COLUMN email_lower VARCHAR(100) 
    GENERATED ALWAYS AS (LOWER(email)) STORED;

CREATE INDEX idx_email_lower_gen ON users(email_lower);

-- 对比
-- 表达式索引：
-- - 不占用额外存储（索引存储计算结果）
-- - 查询时必须使用相同表达式
-- - 更新时自动计算

-- 生成列：
-- - 占用额外存储（列存储计算结果）
-- - 可以直接查询列
-- - 可以用于多个索引
-- - 可以设置约束

-- 选择建议：
-- - 如果表达式简单、查询固定 → 表达式索引
-- - 如果需要多次使用计算值 → 生成列
```

---

### 5. 如何分析和优化索引？

**答案：**

**查看索引使用情况：**

```sql
-- 查看表的索引
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'users';

-- 查看索引大小
SELECT 
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname = 'users';

-- 查看索引使用统计
SELECT 
    indexrelname AS index_name,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE relname = 'users'
ORDER BY idx_scan DESC;

-- 查找未使用的索引
SELECT 
    schemaname || '.' || relname AS table,
    indexrelname AS index,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    idx_scan AS scans
FROM pg_stat_user_indexes
JOIN pg_indexes ON indexrelname = indexname
WHERE idx_scan = 0
  AND indexname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;
```

**分析索引效率：**

```sql
-- 使用 EXPLAIN ANALYZE
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- 输出解读：
-- Index Scan using idx_email on users  (cost=... rows=1 width=...) (actual time=... rows=1 loops=1)
--   Index Cond: (email = 'test@example.com'::text)

-- 关键指标：
-- - rows：预估行数
-- - actual rows：实际行数（差异大说明统计信息不准）
-- - actual time：实际执行时间

-- 查看索引的 bloat（膨胀）
SELECT 
    current_database(),
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
JOIN pg_indexes ON indexrelname = indexname
WHERE pg_relation_size(indexrelid) > 10 * 8192  -- 大于 80KB
ORDER BY pg_relation_size(indexrelid) DESC;
```

**索引优化建议：**

```sql
-- 1. 删除冗余索引
-- 如果有 (a, b) 索引，(a) 索引就是冗余的
CREATE INDEX idx_a_b ON users(a, b);
-- DROP INDEX idx_a;  -- 冗余

-- 2. 使用部分索引减少大小
CREATE INDEX idx_active_email ON users(email) WHERE status = 'active';

-- 3. 使用 BRIN 替代 B-tree（超大有序表）
CREATE INDEX idx_created_brin ON logs USING BRIN(created_at);

-- 4. 使用 INCLUDE 包含额外列
CREATE INDEX idx_email_include ON users(email) INCLUDE (name, status);
-- 可以避免回表

-- 5. 并发创建索引（不锁表）
CREATE INDEX CONCURRENTLY idx_new ON users(created_at);
-- 注意：耗时更长，但不会阻塞写入

-- 6. 重建索引减少碎片
REINDEX INDEX idx_email;
-- 或并发重建
REINDEX INDEX CONCURRENTLY idx_email;
```

**追问：什么是索引膨胀？如何处理？**

**追问答案：**

```sql
-- 索引膨胀原因：
-- 1. 大量 UPDATE/DELETE 导致索引页碎片化
-- 2. VACUUM 清理表数据但索引未完全清理
-- 3. 页分裂导致空洞

-- 检测索引膨胀
SELECT 
    current_database(),
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    100 * pg_relation_size(indexrelid) / NULLIF(pg_relation_size(relid), 0) AS index_ratio
FROM pg_stat_user_indexes
JOIN pg_class ON pg_class.oid = indexrelid
ORDER BY pg_relation_size(indexrelid) DESC;

-- 处理方法
-- 1. REINDEX
REINDEX INDEX idx_email;

-- 2. REINDEX CONCURRENTLY（PostgreSQL 12+，不锁表）
REINDEX INDEX CONCURRENTLY idx_email;

-- 3. pg_repack 扩展（在线重组）
pg_repack -t tablename -d databasename
```

---

### 6. 什么是唯一索引和排除约束？

**答案：**

**唯一索引：**

```sql
-- 创建唯一索引
CREATE UNIQUE INDEX idx_email_unique ON users(email);

-- 主键自动创建唯一索引
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- 自动创建唯一索引
    email VARCHAR(100) UNIQUE  -- 也自动创建唯一索引
);

-- 复合唯一索引
CREATE UNIQUE INDEX idx_user_product ON orders(user_id, product_id);
-- 同一用户不能重复购买同一产品

-- 部分唯一索引（允许部分重复）
CREATE UNIQUE INDEX idx_email_active ON users(email) WHERE status = 'active';
-- 只有 active 用户的 email 需要唯一

-- NULL 值处理
-- PostgreSQL 中 NULL != NULL，所以：
INSERT INTO users (email) VALUES (NULL);
INSERT INTO users (email) VALUES (NULL);  -- 允许多个 NULL

-- 如果要限制 NULL 也唯一，使用：
CREATE UNIQUE INDEX idx_email_not_null ON users(email) WHERE email IS NOT NULL;
```

**排除约束（EXCLUDE）：**

```sql
-- 排除约束比唯一约束更灵活
-- 可以使用不同操作符检查冲突

-- 基本语法
CREATE TABLE bookings (
    room_id INT,
    during TSRANGE,
    EXCLUDE USING GIST (
        room_id WITH =,
        during WITH &&
    )
);

-- 解释：
-- room_id WITH =：room_id 相等时
-- during WITH &&：时间范围重叠时
-- 组合：同一房间的时间不能重叠

-- 测试
INSERT INTO bookings VALUES (1, '[2024-01-01 10:00, 2024-01-01 12:00]');
INSERT INTO bookings VALUES (1, '[2024-01-01 11:00, 2024-01-01 13:00]');
-- 错误：冲突！时间范围重叠

INSERT INTO bookings VALUES (2, '[2024-01-01 11:00, 2024-01-01 13:00]');
-- 成功：不同房间，不冲突

-- 更复杂的排除约束
CREATE TABLE shifts (
    employee_id INT,
    shift_date DATE,
    shift_type VARCHAR(10),
    EXCLUDE USING GIST (
        employee_id WITH =,
        shift_date WITH =
    )
);
-- 同一员工同一天只能有一个班次

-- 使用 btree_gist 扩展支持更多操作符
CREATE EXTENSION btree_gist;

CREATE TABLE documents (
    id SERIAL,
    owner_id INT,
    valid_range TSRANGE,
    EXCLUDE USING GIST (
        owner_id WITH =,
        valid_range WITH &&
    )
);
```

**唯一约束 vs 排除约束：**

```sql
-- 唯一约束
-- 只能使用等于比较
-- 适用于单列或多列组合必须唯一

-- 排除约束
-- 可以使用各种操作符（=, <>, && 等）
-- 适用于更复杂的约束逻辑
-- 需要对应的索引支持（如 GiST）

-- 示例：排班表，同一时段同一位置只能有一人
CREATE TABLE schedule (
    location_id INT,
    employee_id INT,
    time_slot TSRANGE,
    
    -- 排除约束：同一位置同一时间只能有一人
    EXCLUDE USING GIST (
        location_id WITH =,
        time_slot WITH &&
    )
);
```

**追问：排除约束如何影响写入性能？**

**追问答案：**

```sql
-- 排除约束需要索引支持
-- 每次插入/更新都要检查约束
-- 使用 GiST 索引时，检查成本较高

-- 性能影响：
-- 1. 插入时需要检查是否存在冲突
-- 2. GiST 索引的检查比 B-tree 慢
-- 3. 高并发写入时可能有锁竞争

-- 优化建议：
-- 1. 保持索引小而高效
-- 2. 在低峰期批量导入数据
-- 3. 考虑应用层检查 + 数据库约束双重保证

-- 查看排除约束
SELECT 
    conname,
    pg_get_constraintdef(oid)
FROM pg_constraint
WHERE contype = 'x';  -- x 表示排除约束
```

---

### 7. 什么是索引的条件过滤？

**答案：**

索引的条件过滤指 PostgreSQL 在索引扫描时应用过滤条件，减少回表次数。

**Index Cond vs Filter：**

```sql
-- 创建测试数据
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT,
    status VARCHAR(20),
    amount NUMERIC,
    created_at TIMESTAMP
);

CREATE INDEX idx_user_status ON orders(user_id, status);

-- 查询分析
EXPLAIN ANALYZE 
SELECT * FROM orders WHERE user_id = 100 AND status = 'completed';

-- 输出：
-- Index Scan using idx_user_status on orders
--   Index Cond: (user_id = 100) AND ((status)::text = 'completed'::text)

-- Index Cond：索引条件，在索引扫描时应用
-- Filter：过滤条件，在获取数据后应用
```

**过滤条件示例：**

```sql
-- 索引：(user_id, status)
-- 查询条件：(user_id, status, amount)

EXPLAIN ANALYZE 
SELECT * FROM orders WHERE user_id = 100 AND status = 'completed' AND amount > 1000;

-- 输出：
-- Index Scan using idx_user_status on orders
--   Index Cond: (user_id = 100) AND ((status)::text = 'completed'::text)
--   Filter: (amount > '1000'::numeric)

-- Index Cond：索引能覆盖的条件
-- Filter：索引无法覆盖，需要回表后判断的条件
```

**Rows Removed by Filter：**

```sql
EXPLAIN ANALYZE 
SELECT * FROM orders WHERE user_id = 100 AND amount > 1000;

-- 输出可能显示：
-- Rows Removed by Filter: 500

-- 含义：索引扫描找到 500+ 行，但 amount > 1000 过滤掉了 500 行
-- 说明索引不够高效，可能需要改进
```

**优化建议：**

```sql
-- 问题：Rows Removed by Filter 很多
-- 解决方案：扩展索引覆盖更多列

-- 原索引
CREATE INDEX idx_user ON orders(user_id);

-- 优化后的索引
CREATE INDEX idx_user_amount ON orders(user_id, amount);

-- 或者使用 INCLUDE
CREATE INDEX idx_user_include ON orders(user_id) INCLUDE (amount, status);

-- 或者部分索引
CREATE INDEX idx_user_large ON orders(user_id) WHERE amount > 1000;
```

**Index Only Scan：**

```sql
-- 如果索引包含所有查询列，可以避免回表
CREATE INDEX idx_covering ON orders(user_id, status) INCLUDE (amount);

EXPLAIN ANALYZE 
SELECT user_id, status, amount FROM orders WHERE user_id = 100;

-- Index Only Scan：只需要访问索引，不需要回表
-- 如果看到 "Heap Fetches: 0"，说明完全不需要访问表

-- 注意：需要表的 visibility map 可见
-- 如果表有大量更新，可能仍需要回表验证可见性
VACUUM orders;  -- 更新 visibility map
```

---

### 8. 如何使用 EXPLAIN 分析查询？

**答案：**

EXPLAIN 是 PostgreSQL 查询分析的核心工具。

**基本用法：**

```sql
-- 基本执行计划
EXPLAIN SELECT * FROM users WHERE id = 1;

-- 带实际执行统计
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1;

-- 更详细的统计
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) 
SELECT * FROM users WHERE id = 1;

-- JSON 格式（便于解析）
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM users WHERE id = 1;
```

**输出解读：**

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

/*
Index Scan using idx_email on users  (cost=0.42..8.44 rows=1 width=100) (actual time=0.025..0.026 rows=1 loops=1)
  Index Cond: (email = 'test@example.com'::text)
  Buffers: shared hit=4
Planning Time: 0.100 ms
Execution Time: 0.050 ms
*/

-- 解读：
-- 1. Index Scan：使用索引扫描
-- 2. cost=0.42..8.44：
--    - 0.42：启动成本（找到第一条记录的成本）
--    - 8.44：总成本
-- 3. rows=1：预估返回行数
-- 4. width=100：每行平均宽度（字节）
-- 5. actual time=0.025..0.026：实际执行时间（毫秒）
-- 6. rows=1：实际返回行数
-- 7. loops=1：执行次数
-- 8. Buffers: shared hit=4：从缓存读取 4 个页
```

**常见扫描类型：**

```sql
-- 1. Seq Scan：顺序扫描（全表扫描）
EXPLAIN SELECT * FROM users;
-- Seq Scan on users  (cost=0.00..1000.00 rows=10000 width=100)

-- 2. Index Scan：索引扫描 + 回表
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
-- Index Scan using idx_email on users

-- 3. Index Only Scan：仅索引扫描（不回表）
EXPLAIN SELECT email FROM users WHERE email = 'test@example.com';
-- Index Only Scan using idx_email on users

-- 4. Bitmap Index Scan + Bitmap Heap Scan
EXPLAIN SELECT * FROM users WHERE email LIKE 'test%';
-- Bitmap Index Scan on idx_email
--   Recheck Cond: (email ~~ 'test%'::text)
-- Bitmap Heap Scan on users

-- 5. Parallel Seq Scan：并行顺序扫描
SET max_parallel_workers_per_gather = 4;
EXPLAIN SELECT COUNT(*) FROM large_table;
-- Gather
--   Workers Planned: 4
--   -> Parallel Seq Scan on large_table
```

**识别问题：**

```sql
-- 问题1：预估行数与实际差异大
EXPLAIN ANALYZE SELECT * FROM users WHERE status = 'active';
-- rows=1000 (预估) vs rows=50000 (实际)
-- 解决：ANALYZE users; 或增加统计目标

-- 问题2：大量 Rows Removed by Filter
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 100 AND amount > 1000;
-- Rows Removed by Filter: 10000
-- 解决：优化索引覆盖更多条件

-- 问题3：全表扫描
EXPLAIN ANALYZE SELECT * FROM users WHERE name LIKE '%张%';
-- Seq Scan on users
-- 解决：使用 pg_trgm 索引或考虑其他查询方式

-- 问题4：排序溢出到磁盘
EXPLAIN ANALYZE SELECT * FROM large_table ORDER BY created_at;
-- Sort Method: external merge  Disk: 1000kB
-- 解决：增加 work_mem 或创建索引
```

**追问：如何优化慢查询？**

**追问答案：**

```sql
-- 优化步骤：

-- 1. 收集统计信息
ANALYZE tablename;

-- 2. 检查执行计划
EXPLAIN ANALYZE slow_query;

-- 3. 常见优化方法：

-- a) 添加索引
CREATE INDEX idx_column ON table(column);

-- b) 使用覆盖索引
CREATE INDEX idx_covering ON table(a, b) INCLUDE (c, d);

-- c) 部分索引
CREATE INDEX idx_partial ON table(column) WHERE condition;

-- d) 调整参数
SET work_mem = '256MB';
SET random_page_cost = 1.1;  -- SSD 环境

-- e) 重写查询
-- 避免函数索引列
-- WHERE DATE(created_at) = '2024-01-01'
-- 改为
-- WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02'

-- f) 分区表
-- 大表按时间或范围分区

-- g) 并行查询
SET max_parallel_workers_per_gather = 4;
```

---

## 性能优化篇

### 1. 如何分析慢查询？

**答案：**

**开启慢查询日志：**

```sql
-- 在 postgresql.conf 中配置
log_min_duration_statement = 1000  -- 记录超过 1 秒的查询

-- 或动态设置
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();

-- 查看配置
SHOW log_min_duration_statement;
```

**使用 pg_stat_statements：**

```sql
-- 创建扩展
CREATE EXTENSION pg_stat_statements;

-- 配置 postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all

-- 重启后使用
SELECT 
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as avg_time_ms,
    round((100 * total_exec_time / sum(total_exec_time) over())::numeric, 2) as percent,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

**分析查询：**

```sql
-- 1. 使用 EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM orders WHERE user_id = 100;

-- 2. 查看详细信息
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, SUMMARY) 
SELECT * FROM orders WHERE user_id = 100;

-- 3. 查看查询的 I/O 统计
SELECT * FROM pg_stat_io;

-- 4. 查看表的统计信息
SELECT 
    relname,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname = 'orders';
```

**常见慢查询原因及解决：**

```sql
-- 1. 缺少索引
EXPLAIN SELECT * FROM orders WHERE user_id = 100;
-- Seq Scan → 添加索引
CREATE INDEX idx_user_id ON orders(user_id);

-- 2. 索引未使用
EXPLAIN SELECT * FROM orders WHERE DATE(created_at) = '2024-01-01';
-- 函数导致索引失效
-- 改写查询
SELECT * FROM orders 
WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02';

-- 3. 统计信息过时
ANALYZE orders;

-- 4. 内存不足
-- 检查 work_mem
SHOW work_mem;
SET work_mem = '256MB';

-- 5. 锁等待
SELECT * FROM pg_stat_activity WHERE wait_event_type = 'Lock';
```

---

### 2. VACUUM 和 ANALYZE 的区别？

**答案：**

**VACUUM：**

```sql
-- 功能：清理死元组，标记空间可重用
-- 适用：DELETE、UPDATE 后的空间回收

-- 基本用法
VACUUM tablename;           -- 普通清理
VACUUM VERBOSE tablename;   -- 显示详细信息
VACUUM FULL tablename;      -- 重写表，释放磁盘（锁表）
VACUUM FREEZE tablename;    -- 冻结事务 ID

-- 自动 VACUUM
-- autovacuum 后台自动执行
-- 触发条件：
-- dead_tuples > vacuum_threshold + scale_factor * live_tuples
```

**ANALYZE：**

```sql
-- 功能：更新表的统计信息
-- 适用：优化器生成更好的执行计划

-- 基本用法
ANALYZE tablename;          -- 分析整个表
ANALYZE tablename(column);  -- 只分析特定列

-- 手动触发
ANALYZE VERBOSE tablename;

-- 查看统计信息
SELECT 
    attname,
    n_distinct,
    most_common_vals,
    histogram_bounds
FROM pg_stats
WHERE tablename = 'users';
```

**区别对比：**

| 特性 | VACUUM | ANALYZE |
|------|--------|---------|
| 主要功能 | 清理死元组 | 更新统计信息 |
| 空间回收 | 是（标记可重用） | 否 |
| 影响查询计划 | 间接影响 | 直接影响 |
| 锁类型 | SHARE UPDATE EXCLUSIVE | SHARE UPDATE EXCLUSIVE |
| 频率 | 取决于更新频率 | 取决于数据变化量 |
| 自动执行 | autovacuum | autoanalyze |

**组合使用：**

```sql
-- 通常组合使用
VACUUM ANALYZE tablename;

-- 先清理再更新统计
-- 统计信息基于活跃数据，排除死元组
```

**监控：**

```sql
-- VACUUM 监控
SELECT 
    relname,
    n_dead_tup,
    n_live_tup,
    last_vacuum,
    last_autovacuum,
    vacuum_count
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- ANALYZE 监控
SELECT 
    relname,
    last_analyze,
    last_autoanalyze,
    analyze_count
FROM pg_stat_user_tables
ORDER BY last_analyze;
```

---

### 3. pg_stat_statements 怎么用？

**答案：**

pg_stat_statements 是 PostgreSQL 的查询性能监控扩展。

**安装配置：**

```sql
-- 1. 在 postgresql.conf 中配置
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.max = 10000  -- 最多跟踪的语句数

-- 2. 重启 PostgreSQL

-- 3. 创建扩展
CREATE EXTENSION pg_stat_statements;

-- 4. 验证
SELECT * FROM pg_stat_statements LIMIT 1;
```

**常用查询：**

```sql
-- 1. 最慢的查询
SELECT 
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as avg_time_ms,
    round(min_exec_time::numeric, 2) as min_time_ms,
    round(max_exec_time::numeric, 2) as max_time_ms,
    rows,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- 2. 平均执行时间最长的查询
SELECT 
    calls,
    round(mean_exec_time::numeric, 2) as avg_time_ms,
    query
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- 3. 执行次数最多的查询
SELECT 
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    query
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 10;

-- 4. I/O 开销最大的查询
SELECT 
    calls,
    shared_blks_hit,
    shared_blks_read,
    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) as hit_ratio,
    query
FROM pg_stat_statements
ORDER BY shared_blks_read DESC
LIMIT 10;

-- 5. 返回行数最多的查询
SELECT 
    calls,
    rows,
    round(rows::numeric / calls, 2) as avg_rows,
    query
FROM pg_stat_statements
ORDER BY rows DESC
LIMIT 10;
```

**重置统计：**

```sql
-- 重置所有统计
SELECT pg_stat_statements_reset();

-- 重置特定查询（PostgreSQL 14+）
SELECT pg_stat_statements_reset(0, 0, queryid);
```

**监控脚本示例：**

```sql
-- 创建慢查询监控视图
CREATE VIEW slow_queries AS
SELECT 
    dbid,
    userid,
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as avg_time_ms,
    rows,
    query
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- 平均超过 100ms
ORDER BY total_exec_time DESC;

-- 查询
SELECT * FROM slow_queries LIMIT 20;
```

---

### 4. 连接池配置要点？

**答案：**

PostgreSQL 是进程模型，每个连接一个进程，连接开销大，需要连接池。

**问题背景：**

```sql
-- 查看最大连接数
SHOW max_connections;  -- 默认 100

-- 每个连接消耗资源：
-- - 独立进程（约 10MB 内存）
-- - work_mem × 每个操作
-- - 维护连接状态的开销

-- 查看当前连接
SELECT count(*) FROM pg_stat_activity;
```

**PgBouncer 配置：**

```ini
; pgbouncer.ini

[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
; 监听地址
listen_addr = 0.0.0.0
listen_port = 6432

; 连接池模式
; session: 会话级池化（最安全）
; transaction: 事务级池化（推荐，性能好）
; statement: 语句级池化（有限制）
pool_mode = transaction

; 池大小
max_client_conn = 1000        ; 最大客户端连接
default_pool_size = 20        ; 每个数据库的池大小
min_pool_size = 5             ; 最小池大小
reserve_pool_size = 5         ; 预留池大小

; 超时设置
server_connect_timeout = 3
server_idle_timeout = 600
server_lifetime = 3600
server_check_delay = 30

; 客户端超时
client_idle_timeout = 0
client_login_timeout = 60

; 日志
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1

; 管理用户
admin_users = postgres
```

**连接池模式对比：**

| 模式 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| session | 会话级 | 完全兼容 | 并发低 |
| transaction | 事务级 | 并发高 | 有些功能受限 |
| statement | 语句级 | 并发最高 | 不支持事务 |

```sql
-- transaction 模式限制：
-- 1. 不支持 SET、PREPARE、DISCARD 等会话语句
-- 2. 不支持 WITH HOLD CURSOR
-- 3. 不支持临时表
-- 4. 不支持 LISTEN/NOTIFY
-- 5. 不支持预备事务

-- 解决方案：
-- 使用 SET LOCAL 代替 SET
BEGIN;
SET LOCAL work_mem = '256MB';
-- 执行查询
COMMIT;  -- 设置自动回滚
```

**监控连接池：**

```sql
-- 连接到 pgbouncer 管理数据库
psql -p 6432 pgbouncer -U postgres

-- 查看统计
pgbouncer=# SHOW STATS;

-- 查看客户端
pgbouncer=# SHOW CLIENTS;

-- 查看服务端连接
pgbouncer=# SHOW POOLS;

-- 查看 databases 配置
pgbouncer=# SHOW DATABASES;
```

**PostgreSQL 端配置：**

```sql
-- 调整最大连接数（配合连接池）
ALTER SYSTEM SET max_connections = 200;

-- 确保有足够的连接给连接池使用
-- 如果 default_pool_size=20，有 10 个数据库
-- 需要 200 个连接

-- 调整内存参数
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET work_mem = '64MB';  -- 每个连接的 work_mem
```

---

### 5. 并行查询怎么配置？

**答案：**

PostgreSQL 支持并行查询来加速大表扫描和聚合操作。

**配置参数：**

```sql
-- 查看并行相关配置
SHOW max_parallel_workers;              -- 最大并行工作进程（默认8）
SHOW max_parallel_workers_per_gather;   -- 每个 Gather 的最大并行（默认2）
SHOW min_parallel_table_scan_size;      -- 触发并行的最小表大小
SHOW min_parallel_index_scan_size;      -- 触发并行的最小索引大小
SHOW parallel_tuple_cost;               -- 并行元组成本
SHOW parallel_setup_cost;               -- 启动并行的成本

-- 调整配置
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
SELECT pg_reload_conf();
```

**并行查询示例：**

```sql
-- 创建大表测试
CREATE TABLE large_table AS 
SELECT generate_series(1, 10000000) AS id, random() AS value;
ANALYZE large_table;

-- 查看执行计划
EXPLAIN ANALYZE SELECT COUNT(*) FROM large_table;

/*
Finalize Aggregate
  ->  Gather
        Workers Planned: 4
        Workers Launched: 4
        ->  Partial Aggregate
              ->  Parallel Seq Scan on large_table
*/

-- 强制并行（测试用）
SET parallel_tuple_cost = 0;
SET parallel_setup_cost = 0;
SET max_parallel_workers_per_gather = 4;

-- 禁用并行
SET max_parallel_workers_per_gather = 0;
```

**并行适用场景：**

```sql
-- 1. 全表扫描
SELECT COUNT(*) FROM large_table;
SELECT AVG(value) FROM large_table WHERE id > 1000;

-- 2. 索引扫描（PostgreSQL 10+）
CREATE INDEX idx_value ON large_table(value);
SELECT * FROM large_table WHERE value > 0.5;

-- 3. 聚合操作
SELECT id, COUNT(*) FROM large_table GROUP BY id;

-- 4. 连接查询
SELECT * FROM large_table t1 JOIN large_table t2 ON t1.id = t2.id;

-- 5. 排序（PostgreSQL 11+）
SELECT * FROM large_table ORDER BY value DESC LIMIT 1000;
```

**不适用并行的场景：**

```sql
-- 1. 小表查询（低于 min_parallel_table_scan_size）
SELECT * FROM small_table;

-- 2. 返回少量行的查询
SELECT * FROM large_table WHERE id = 1;

-- 3. 有锁定的查询
SELECT * FROM large_table FOR UPDATE;

-- 4. 可串行化隔离级别
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT * FROM large_table;  -- 不并行

-- 5. 非并行安全的函数
CREATE FUNCTION non_parallel_safe() RETURNS int AS $$ ... $$ LANGUAGE plpgsql;
SELECT non_parallel_safe() FROM large_table;  -- 不并行

-- 标记函数为并行安全
ALTER FUNCTION my_function() PARALLEL SAFE;
```

**监控并行查询：**

```sql
-- 查看并行工作进程使用情况
SELECT * FROM pg_stat_activity 
WHERE query LIKE '%Parallel%';

-- 查看执行计划中的并行信息
EXPLAIN (ANALYZE, VERBOSE) SELECT COUNT(*) FROM large_table;

-- 检查并行相关等待事件
SELECT wait_event, count(*) 
FROM pg_stat_activity 
WHERE wait_event LIKE '%parallel%'
GROUP BY wait_event;
```

---

### 6. 如何优化 PostgreSQL 内存使用？

**答案：**

**关键内存参数：**

```sql
-- 1. shared_buffers：共享缓冲区
-- 建议值：系统内存的 25%
SHOW shared_buffers;  -- 默认 128MB
ALTER SYSTEM SET shared_buffers = '4GB';

-- 2. work_mem：每个操作的内存
-- 建议值：64MB-256MB（注意连接数）
SHOW work_mem;  -- 默认 4MB
ALTER SYSTEM SET work_mem = '64MB';

-- 3. maintenance_work_mem：维护操作内存
-- 建议值：256MB-1GB
SHOW maintenance_work_mem;
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- 4. effective_cache_size：预估可用缓存
-- 建议值：系统内存的 75%
SHOW effective_cache_size;
ALTER SYSTEM SET effective_cache_size = '12GB';

-- 5. huge_pages：大页内存
-- 建议值：try 或 on（Linux）
SHOW huge_pages;
```

**内存使用计算：**

```
内存使用估算：

1. shared_buffers：固定占用
   - 设置值本身

2. work_mem：按连接数和操作数
   - work_mem × max_connections × operations_per_query
   - 例如：64MB × 200 × 3 = 38.4GB（可能超！）

3. maintenance_work_mem：维护操作时
   - VACUUM、CREATE INDEX 等操作临时使用

总内存需求：
shared_buffers + (work_mem × max_connections) + 其他

示例（32GB 内存服务器）：
shared_buffers = 8GB
work_mem = 64MB
max_connections = 200
连接池后实际连接 = 50

内存使用：
8GB + 64MB × 50 × 2 ≈ 14.4GB
剩余给操作系统和其他进程
```

**优化建议：**

```sql
-- 1. 根据实际调整 shared_buffers
-- 太大可能导致缓存竞争
-- 太小会导致更多磁盘 I/O

-- 2. 谨慎设置 work_mem
-- 在会话级别按需设置
SET LOCAL work_mem = '256MB';

-- 3. 使用连接池控制连接数
-- 减少连接数可以增加 work_mem

-- 4. 监控内存使用
SELECT 
    name,
    setting,
    unit,
    context
FROM pg_settings
WHERE name IN ('shared_buffers', 'work_mem', 'maintenance_work_mem', 'effective_cache_size');

-- 5. 查看缓存命中率
SELECT 
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    round(100.0 * sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)), 2) as ratio
FROM pg_statio_user_tables;

-- 缓存命中率应 > 99%
```

**NUMA 架构注意事项：**

```sql
-- NUMA 系统上，shared_buffers 不应超过单个 NUMA 节点
-- 查看 NUMA 配置（Linux）
-- numactl --hardware

-- 如果有多个 NUMA 节点，考虑：
-- 1. 减小 shared_buffers
-- 2. 使用 numactl 绑定 PostgreSQL 到特定节点
-- 3. 配置 huge_pages 减少内存管理开销
```

---

### 7. 如何监控 PostgreSQL 性能？

**答案：**

**系统视图监控：**

```sql
-- 1. 活动会话
SELECT 
    pid,
    usename,
    state,
    wait_event_type,
    wait_event,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- 2. 锁等待
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid;

-- 3. 表统计
SELECT 
    relname,
    n_live_tup,
    n_dead_tup,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_ratio,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- 4. 索引使用
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 5. I/O 统计
SELECT 
    relname,
    heap_blks_read,
    heap_blks_hit,
    round(100.0 * heap_blks_hit / NULLIF(heap_blks_hit + heap_blks_read, 0), 2) AS hit_ratio
FROM pg_statio_user_tables
ORDER BY heap_blks_read DESC;
```

**扩展监控：**

```sql
-- pg_stat_statements
CREATE EXTENSION pg_stat_statements;

-- pg_stat_kcache（需要安装）
-- 监控 CPU 和 I/O

-- pg_wait_sampling（需要安装）
-- 采样等待事件

-- 查看当前等待事件
SELECT 
    wait_event_type,
    wait_event,
    count(*)
FROM pg_stat_activity
WHERE wait_event IS NOT NULL
GROUP BY wait_event_type, wait_event
ORDER BY count DESC;
```

**监控脚本示例：**

```sql
-- 创建监控视图
CREATE VIEW monitoring.dashboard AS
SELECT 
    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active_connections,
    (SELECT count(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL) AS waiting_queries,
    (SELECT sum(n_dead_tup) FROM pg_stat_user_tables) AS total_dead_tuples,
    (SELECT round(100.0 * sum(heap_blks_hit) / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0), 2) 
     FROM pg_statio_user_tables) AS cache_hit_ratio,
    (SELECT pg_database_size(current_database())) AS database_size;

-- 查询监控面板
SELECT * FROM monitoring.dashboard;
```

---

## 高级特性篇

### 1. 什么是 TOAST？

**答案：**

TOAST（The Oversized-Attribute Storage Technique）是 PostgreSQL 处理大字段的技术。

**问题背景：**

```
PostgreSQL 数据页大小：8KB（默认）
单行数据不能超过一个数据页

问题：
- TEXT 类型可能很大
- JSONB 可能很大
- BYTEA 二进制数据可能很大

解决方案：TOAST
- 超过阈值的大字段单独存储
- 自动压缩和分块
- 对用户透明
```

**TOAST 策略：**

```sql
-- 查看列的 TOAST 策略
SELECT 
    a.attname,
    a.attstorage
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
WHERE c.relname = 'users' AND a.attnum > 0;

-- attstorage 值：
-- p: PLAIN - 不压缩，不存储到 TOAST 表
-- e: EXTERNAL - 不压缩，但可以存储到 TOAST 表
-- m: MAIN - 尝试压缩，不存 TOAST（默认）
-- x: EXTENDED - 尝试压缩，存 TOAST（变长类型默认）
```

**设置 TOAST 策略：**

```sql
-- 创建表时设置
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT STORAGE EXTERNAL,  -- 不压缩，允许随机访问
    data JSONB STORAGE EXTENDED     -- 压缩，存 TOAST（默认）
);

-- 修改现有列
ALTER TABLE documents ALTER COLUMN content SET STORAGE EXTERNAL;

-- 存储策略选择：
-- PLAIN：定长类型，不适合大字段
-- EXTERNAL：需要随机访问大字段（如 substring）
-- MAIN：尽量不存 TOAST，但可能压缩
-- EXTENDED：默认策略，先压缩再 TOAST
```

**TOAST 表结构：**

```sql
-- 每个有 TOAST 列的表都有对应的 TOAST 表
-- 命名：pg_toast.pg_toast_xxx

-- 查看 TOAST 表
SELECT relname FROM pg_class 
WHERE relname LIKE 'pg_toast%' AND relkind = 't';

-- TOAST 表结构
-- chunk_id: TOAST 值的 ID
-- chunk_seq: 分块序号
-- chunk_data: 分块数据（每块约 2KB）

-- TOAST 大小
SELECT 
    c.relname,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    pg_size_pretty(pg_relation_size(c.oid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(c.oid) - pg_relation_size(c.oid)) AS toast_index_size
FROM pg_class c
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE n.nspname = 'public' AND c.relkind = 'r'
ORDER BY pg_total_relation_size(c.oid) DESC;
```

**TOAST 压缩：**

```sql
-- 自动压缩
-- 当数据超过约 2KB 时，尝试压缩
-- 如果压缩后仍超过阈值，存入 TOAST 表

-- 查看压缩效果
SELECT 
    pg_column_size(content) AS compressed_size,
    length(content) AS original_size,
    round(100.0 * pg_column_size(content) / NULLIF(length(content), 0), 2) AS compression_ratio
FROM documents;
```

**追问：TOAST 对查询有什么影响？**

**追问答案：**

```sql
-- 影响1：查询大字段需要额外 I/O
-- TOAST 数据可能分散在多个页

-- 影响2：SELECT * 性能下降
-- 避免 SELECT *，只查需要的列

-- 影响3：压缩/解压 CPU 开销
-- 如果不需要压缩，设置 STORAGE EXTERNAL

-- 优化建议：
-- 1. 避免 SELECT *，只查需要的列
-- 2. 大字段单独表存储
-- 3. 根据访问模式选择存储策略
-- 4. 对大表考虑分区

-- 示例：大内容分离
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    created_at TIMESTAMP
);

CREATE TABLE article_contents (
    article_id INT PRIMARY KEY REFERENCES articles(id),
    content TEXT
);

-- 查询文章列表不需要 content
SELECT id, title, created_at FROM articles ORDER BY created_at DESC LIMIT 20;
```

---

### 2. 窗口函数怎么用？

**答案：**

窗口函数（Window Function）可以在不减少行数的情况下进行聚合计算。

**基本语法：**

```sql
-- 语法结构
function_name(expression) OVER (
    [PARTITION BY partition_expression]
    [ORDER BY sort_expression]
    [frame_clause]
)
```

**常用窗口函数：**

```sql
-- 创建测试数据
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    employee_id INT,
    sale_date DATE,
    amount NUMERIC
);

INSERT INTO sales (employee_id, sale_date, amount) VALUES
(1, '2024-01-01', 1000),
(1, '2024-01-02', 1500),
(1, '2024-01-03', 2000),
(2, '2024-01-01', 1200),
(2, '2024-01-02', 1800),
(2, '2024-01-03', 1600);

-- 1. ROW_NUMBER: 行号
SELECT 
    employee_id,
    sale_date,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num
FROM sales;

-- 2. RANK 和 DENSE_RANK: 排名
SELECT 
    employee_id,
    amount,
    RANK() OVER (ORDER BY amount DESC) AS rank,        -- 有空缺
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank  -- 无空缺
FROM sales;

-- 3. 聚合函数作为窗口函数
SELECT 
    employee_id,
    sale_date,
    amount,
    SUM(amount) OVER (PARTITION BY employee_id) AS total_by_employee,
    SUM(amount) OVER () AS grand_total,
    AVG(amount) OVER (PARTITION BY employee_id) AS avg_by_employee
FROM sales;

-- 4. 累计计算
SELECT 
    sale_date,
    amount,
    SUM(amount) OVER (ORDER BY sale_date) AS running_total,
    AVG(amount) OVER (
        ORDER BY sale_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3
FROM sales
WHERE employee_id = 1;

-- 5. LAG 和 LEAD: 访问前后行
SELECT 
    sale_date,
    amount,
    LAG(amount, 1) OVER (ORDER BY sale_date) AS prev_amount,
    LEAD(amount, 1) OVER (ORDER BY sale_date) AS next_amount,
    amount - LAG(amount, 1) OVER (ORDER BY sale_date) AS diff
FROM sales
WHERE employee_id = 1;

-- 6. FIRST_VALUE 和 LAST_VALUE
SELECT 
    employee_id,
    sale_date,
    amount,
    FIRST_VALUE(amount) OVER (PARTITION BY employee_id ORDER BY sale_date) AS first_sale,
    LAST_VALUE(amount) OVER (
        PARTITION BY employee_id 
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sale
FROM sales;

-- 7. NTILE: 分桶
SELECT 
    employee_id,
    amount,
    NTILE(4) OVER (ORDER BY amount DESC) AS quartile  -- 分成4组
FROM sales;
```

**Frame 子句：**

```sql
-- Frame 定义窗口范围
-- ROWS 或 RANGE

-- 示例
SELECT 
    sale_date,
    amount,
    -- 从开始到当前行
    SUM(amount) OVER (ORDER BY sale_date ROWS UNBOUNDED PRECEDING) AS cumulative,
    
    -- 前2行到当前行
    SUM(amount) OVER (ORDER BY sale_date ROWS 2 PRECEDING) AS last_3,
    
    -- 前1行到后1行
    SUM(amount) OVER (ORDER BY sale_date ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS window_3,
    
    -- 整个分区
    SUM(amount) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total
FROM sales;
```

**实际应用示例：**

```sql
-- 1. 每个员工销售额占部门比例
SELECT 
    employee_id,
    amount,
    amount / SUM(amount) OVER (PARTITION BY employee_id) AS percent_of_total
FROM sales;

-- 2. 同比/环比增长
SELECT 
    sale_date,
    amount,
    LAG(amount, 1) OVER (ORDER BY sale_date) AS prev_period,
    ROUND(100.0 * (amount - LAG(amount, 1) OVER (ORDER BY sale_date)) 
          / NULLIF(LAG(amount, 1) OVER (ORDER BY sale_date), 0), 2) AS growth_rate
FROM sales;

-- 3. 每组取前N条
WITH ranked AS (
    SELECT 
        employee_id,
        sale_date,
        amount,
        ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY amount DESC) AS rn
    FROM sales
)
SELECT * FROM ranked WHERE rn <= 2;  -- 每个员工销售额最高的2天
```

---

### 3. LATERAL JOIN 是什么？

**答案：**

LATERAL JOIN 允许子查询引用外部查询的列，实现类似循环的效果。

**基本语法：**

```sql
SELECT *
FROM table1 t1,
LATERAL (SELECT * FROM table2 WHERE table2.col = t1.col) t2;
```

**使用场景：**

```sql
-- 场景1：每个分组取 Top N

-- 创建测试数据
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT,
    amount NUMERIC,
    created_at TIMESTAMP
);

-- 传统方法（使用窗口函数）
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
    FROM orders
)
SELECT * FROM ranked WHERE rn <= 3;

-- 使用 LATERAL JOIN
SELECT o.*
FROM (SELECT DISTINCT user_id FROM orders) u
CROSS JOIN LATERAL (
    SELECT * FROM orders 
    WHERE user_id = u.user_id 
    ORDER BY amount DESC 
    LIMIT 3
) o;

-- 场景2：关联计算
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 获取每个用户及其最近3条订单
SELECT 
    u.id,
    u.name,
    o.id AS order_id,
    o.amount,
    o.created_at
FROM users u
CROSS JOIN LATERAL (
    SELECT id, amount, created_at
    FROM orders
    WHERE user_id = u.id
    ORDER BY created_at DESC
    LIMIT 3
) o;

-- 场景3：复杂的关联子查询
-- 查找每个用户高于其平均金额的订单
SELECT 
    u.id,
    u.name,
    o.amount,
    avg_stats.avg_amount
FROM users u
CROSS JOIN LATERAL (
    SELECT AVG(amount) AS avg_amount FROM orders WHERE user_id = u.id
) avg_stats
JOIN orders o ON o.user_id = u.id AND o.amount > avg_stats.avg_amount;
```

**LATERAL vs 普通子查询：**

```sql
-- 普通子查询：不能引用外部列
-- 错误
SELECT *
FROM users u
JOIN (SELECT * FROM orders WHERE user_id = u.id) o;  -- u.id 不可见

-- LATERAL：可以引用外部列
SELECT *
FROM users u
CROSS JOIN LATERAL (SELECT * FROM orders WHERE user_id = u.id) o;

-- LATERAL 执行过程：
-- 1. 从 users 取一行
-- 2. 用该行的 u.id 执行子查询
-- 3. 子查询结果与该行组合
-- 4. 重复步骤1-3，直到遍历完 users
```

**LATERAL 与函数调用：**

```sql
-- 返回集合的函数
CREATE OR REPLACE FUNCTION get_recent_orders(p_user_id INT)
RETURNS SETOF orders AS $$
    SELECT * FROM orders 
    WHERE user_id = p_user_id 
    ORDER BY created_at DESC 
    LIMIT 5;
$$ LANGUAGE sql;

-- 使用 LATERAL 调用
SELECT u.id, u.name, o.*
FROM users u
CROSS JOIN LATERAL get_recent_orders(u.id) o;

-- 等价于
SELECT u.id, u.name, o.*
FROM users u
CROSS JOIN LATERAL (
    SELECT * FROM get_recent_orders(u.id)
) o;
```

**追问：LATERAL JOIN 和普通 JOIN 有什么区别？**

**追问答案：**

```sql
-- 普通 JOIN
-- 子查询先执行，独立于外部查询
SELECT u.*, o.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
-- orders 表完整扫描（或使用索引）

-- LATERAL JOIN
-- 子查询为每一行外部数据重新执行
SELECT u.*, o.*
FROM users u
LEFT JOIN LATERAL (
    SELECT * FROM orders WHERE user_id = u.id LIMIT 3
) o ON true;
-- 对每个用户，单独执行一次子查询

-- 性能对比：
-- 1. 如果子查询结果很小（如 LIMIT 3），LATERAL 更高效
-- 2. 如果子查询需要大量数据，普通 JOIN 可能更好
-- 3. LATERAL 类似于嵌套循环连接

-- 何时使用 LATERAL：
-- 1. 需要引用外部查询的列
-- 2. 需要为每行执行不同的子查询
-- 3. 需要限制子查询结果数量（如每组 Top N）
```

---

### 4. pgvector 向量搜索怎么用？

**答案：**

pgvector 是 PostgreSQL 的向量搜索扩展，支持向量存储和相似度搜索。

**安装和配置：**

```sql
-- 安装扩展
CREATE EXTENSION vector;

-- 查看版本
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

**基本使用：**

```sql
-- 创建带向量列的表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)  -- OpenAI embedding 维度
);

-- 插入向量数据
INSERT INTO documents (content, embedding) 
VALUES ('Hello world', '[0.1, 0.2, 0.3, ...]'::vector);

-- 余弦相似度搜索
SELECT id, content, 
       1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- 距离运算符：
-- <-> : L2 距离
-- <#> : 内积（负值）
-- <=> : 余弦距离
```

**创建向量索引：**

```sql
-- IVFFlat 索引
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 参数说明：
-- lists: 聚类中心数量，建议 rows/1000

-- HNSW 索引（更高精度）
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- HNSW 参数：
-- m: 每个节点的连接数，默认 16
-- ef_construction: 构建时的搜索范围，默认 64
```

**完整示例（结合 OpenAI Embedding）：**

```sql
-- 创建文档表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(1536)
);

-- 创建 HNSW 索引
CREATE INDEX idx_embedding ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 搜索函数
CREATE OR REPLACE FUNCTION search_documents(query_embedding vector, limit_count INT DEFAULT 10)
RETURNS TABLE(id INT, title TEXT, content TEXT, similarity FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    ORDER BY d.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 使用搜索
SELECT * FROM search_documents('[0.1, 0.2, ...]'::vector, 5);
```

**混合搜索（向量 + 全文）：**

```sql
-- 结合全文搜索
CREATE INDEX idx_content_fts ON documents 
USING GIN(to_tsvector('english', content));

-- 混合查询
WITH vector_search AS (
    SELECT id, 1 - (embedding <=> '[...]'::vector) AS score
    FROM documents
    ORDER BY embedding <=> '[...]'::vector
    LIMIT 100
),
text_search AS (
    SELECT id, ts_rank(to_tsvector('english', content), to_tsquery('hello')) AS score
    FROM documents
    WHERE to_tsvector('english', content) @@ to_tsquery('hello')
    LIMIT 100
)
SELECT 
    COALESCE(v.id, t.id) AS id,
    COALESCE(v.score, 0) * 0.7 + COALESCE(t.score, 0) * 0.3 AS combined_score
FROM vector_search v
FULL OUTER JOIN text_search t ON v.id = t.id
ORDER BY combined_score DESC
LIMIT 10;
```

**性能优化：**

```sql
-- 1. 预热索引
SELECT pg_prewarm('idx_embedding');

-- 2. 设置查询时的 ef（HNSW）
SET hnsw.ef_search = 100;  -- 默认 40，增大可提高召回率

-- 3. 批量插入
-- 先插入数据，再创建索引
INSERT INTO documents (content, embedding) 
SELECT content, generate_embedding(content)
FROM source_table;

CREATE INDEX idx_embedding ON documents 
USING hnsw (embedding vector_cosine_ops);

-- 4. 监控
SELECT 
    schemaname,
    relname,
    indexrelname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE indexrelname = 'idx_embedding';
```

---

### 5. 什么是表继承？怎么用？

**答案：**

表继承是 PostgreSQL 的面向对象特性，允许表继承另一个表的结构和数据。

**基本语法：**

```sql
-- 父表
CREATE TABLE measurements (
    id SERIAL,
    city_id INT,
    log_date DATE,
    peaktemp INT,
    unitsales INT
);

-- 子表继承父表
CREATE TABLE measurements_2024_01 (
    CHECK (log_date >= '2024-01-01' AND log_date < '2024-02-01')
) INHERITS (measurements);

CREATE TABLE measurements_2024_02 (
    CHECK (log_date >= '2024-02-01' AND log_date < '2024-03-01')
) INHERITS (measurements);

-- 子表自动拥有父表的所有列
-- 可以添加额外列
ALTER TABLE measurements_2024_01 ADD COLUMN extra_data TEXT;
```

**数据操作：**

```sql
-- 插入到子表
INSERT INTO measurements_2024_01 (city_id, log_date, peaktemp, unitsales)
VALUES (1, '2024-01-15', 25, 100);

-- 查询父表会包含所有子表数据
SELECT * FROM measurements WHERE log_date = '2024-01-15';

-- 只查询父表（不包含子表）
SELECT * FROM ONLY measurements;

-- 更新和删除
UPDATE measurements SET peaktemp = 26 WHERE id = 1;
-- 自动路由到正确的子表
```

**约束排除：**

```sql
-- 启用约束排除
SET constraint_exclusion = on;

-- 查询时自动排除不符合条件的子表
EXPLAIN SELECT * FROM measurements WHERE log_date = '2024-01-15';
-- 只扫描 measurements_2024_01

-- 禁用约束排除
SET constraint_exclusion = off;
-- 会扫描所有子表
```

**表继承 vs 分区表：**

```sql
-- PostgreSQL 10+ 推荐使用原生分区表

-- 分区表语法
CREATE TABLE measurements (
    id SERIAL,
    city_id INT,
    log_date DATE,
    peaktemp INT,
    unitsales INT
) PARTITION BY RANGE (log_date);

CREATE TABLE measurements_2024_01 
    PARTITION OF measurements
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 分区表优势：
-- 1. 自动路由数据
-- 2. 更好的查询优化
-- 3. 无需约束排除配置
-- 4. 支持更多分区类型

-- 表继承适用于：
-- 1. 需要子表有额外列
-- 2. 需要灵活的继承关系
-- 3. 兼容旧版本
```

**查询继承关系：**

```sql
-- 查看表继承关系
SELECT 
    c.relname AS child,
    p.relname AS parent
FROM pg_inherits i
JOIN pg_class c ON i.inhrelid = c.oid
JOIN pg_class p ON i.inhparent = p.oid;

-- 查看表的所有子表
SELECT c.relname 
FROM pg_inherits i
JOIN pg_class c ON i.inhrelid = c.oid
WHERE i.inhparent = 'measurements'::regclass;
```

**追问：表继承和原生分区表如何选择？**

**追问答案：**

| 特性 | 表继承 | 原生分区表 |
|------|--------|------------|
| PostgreSQL 版本 | 所有版本 | 10+ |
| 自动数据路由 | 需要触发器 | 自动 |
| 查询优化 | 需要约束排除 | 自动 |
| 子表额外列 | 支持 | 不支持 |
| 分区类型 | 自定义 | RANGE/LIST/HASH |
| 多级继承 | 支持 | 支持 |
| 维护复杂度 | 较高 | 较低 |

**建议：**
- PostgreSQL 10+ 优先使用原生分区表
- 需要子表有不同结构时使用表继承
- 需要更复杂的继承关系时使用表继承

---

### 6. 什么是逻辑复制？

**答案：**

逻辑复制是基于复制标识（主键或唯一键）的数据同步方式，比物理复制更灵活。

**逻辑复制 vs 物理复制：**

| 特性 | 逻辑复制 | 物理复制 |
|------|----------|----------|
| 复制级别 | 行级（逻辑） | 块级（物理） |
| 跨版本 | 支持 | 不支持 |
| 跨平台 | 支持 | 不支持 |
| 选择性复制 | 支持（表级） | 全库复制 |
| 目标可读写 | 是 | 只读 |
| 延迟 | 较高 | 较低 |

**配置逻辑复制：**

```sql
-- 发布端（主库）
-- 1. 配置 postgresql.conf
wal_level = logical
max_replication_slots = 10
max_wal_senders = 10

-- 2. 创建发布
CREATE PUBLICATION my_publication FOR TABLE users, orders;

-- 或发布所有表
CREATE PUBLICATION all_tables FOR ALL TABLES;

-- 查看发布
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;
```

```sql
-- 订阅端（从库）
-- 1. 创建订阅
CREATE SUBSCRIPTION my_subscription
    CONNECTION 'host=192.168.1.1 dbname=mydb user=replicator password=xxx'
    PUBLICATION my_publication;

-- 2. 查看订阅状态
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;

-- 3. 管理订阅
ALTER SUBSCRIPTION my_subscription DISABLE;  -- 禁用
ALTER SUBSCRIPTION my_subscription ENABLE;   -- 启用
ALTER SUBSCRIPTION my_subscription REFRESH PUBLICATION;  -- 刷新
DROP SUBSCRIPTION my_subscription;  -- 删除
```

**选择性复制：**

```sql
-- 只复制特定表
CREATE PUBLICATION user_pub FOR TABLE users;

-- 复制 INSERT 和 UPDATE，不复制 DELETE
CREATE PUBLICATION insert_only_pub 
FOR TABLE users 
WITH (publish = 'insert, update');

-- 添加/移除表
ALTER PUBLICATION my_publication ADD TABLE products;
ALTER PUBLICATION my_publication DROP TABLE orders;
```

**复制槽管理：**

```sql
-- 查看复制槽
SELECT * FROM pg_replication_slots;

-- 手动创建复制槽
SELECT pg_create_logical_replication_slot('my_slot', 'pgoutput');

-- 删除复制槽
SELECT pg_drop_replication_slot('my_slot');

-- 注意：如果订阅者断开，复制槽会保留 WAL，可能导致磁盘满
-- 需要监控和清理
```

**冲突处理：**

```sql
-- 逻辑复制允许订阅端写入
-- 可能产生冲突

-- 查看复制冲突
SELECT * FROM pg_stat_subscription_conflicts;

-- 常见冲突：
-- 1. 主键冲突：订阅端已存在相同主键
-- 2. 外键冲突：引用的数据不存在
-- 3. 唯一约束冲突

-- 解决方案：
-- 1. 确保订阅端初始数据一致
-- 2. 使用 conflict_resolution 参数
-- 3. 手动处理后重启订阅
```

**监控逻辑复制：**

```sql
-- 发布端监控
SELECT 
    slot_name,
    plugin,
    slot_type,
    active,
    restart_lsn,
    confirmed_flush_lsn
FROM pg_replication_slots;

-- 订阅端监控
SELECT 
    subname,
    relid::regclass,
    received_lsn,
    latest_end_lsn,
    latest_end_time
FROM pg_stat_subscription;
```

---

### 7. 如何实现数据审计？

**答案：**

PostgreSQL 提供多种数据审计方案。

**方案1：触发器审计：**

```sql
-- 创建审计表
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    operation TEXT,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT DEFAULT current_user,
    changed_at TIMESTAMP DEFAULT now()
);

-- 创建通用审计触发器函数
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_data)
        VALUES (TG_TABLE_NAME, 'INSERT', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, new_data)
        VALUES (TG_TABLE_NAME, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_data)
        VALUES (TG_TABLE_NAME, 'DELETE', to_jsonb(OLD));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 在表上创建触发器
CREATE TRIGGER users_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

-- 查询审计日志
SELECT * FROM audit_log WHERE table_name = 'users' ORDER BY changed_at DESC;
```

**方案2：pgAudit 扩展：**

```sql
-- 安装 pgAudit
-- 在 postgresql.conf 中配置
shared_preload_libraries = 'pgaudit'
pgaudit.log = 'write, ddl'  -- 记录写入和 DDL 操作
pgaudit.log_client = on
pgaudit.log_level = 'log'

-- 重启 PostgreSQL

-- 使用
-- 所有操作都会记录到 PostgreSQL 日志
-- 格式：AUDIT: session, xxx, ...
```

**方案3：逻辑解码：**

```sql
-- 使用逻辑复制捕获变更
-- 创建复制槽
SELECT pg_create_logical_replication_slot('audit_slot', 'pgoutput');

-- 消费变更
SELECT * FROM pg_logical_slot_get_changes('audit_slot', NULL, NULL);

-- 输出格式包含：
-- - 操作类型（INSERT/UPDATE/DELETE）
-- - 表名
-- - 变更前后的数据
```

**方案4：系统审计：**

```sql
-- 配置 postgresql.conf
log_statement = 'ddl'              -- 记录 DDL
log_connections = on               -- 记录连接
log_disconnections = on            -- 记录断开
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

-- pgaudit 增强
pgaudit.log = 'read, write, ddl, role'
pgaudit.log_relation = on
```

**审计表分区：**

```sql
-- 对审计表按时间分区
CREATE TABLE audit_log (
    id SERIAL,
    table_name TEXT,
    operation TEXT,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT,
    changed_at TIMESTAMP
) PARTITION BY RANGE (changed_at);

-- 创建月度分区
CREATE TABLE audit_log_2024_01 
    PARTITION OF audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE audit_log_2024_02 
    PARTITION OF audit_log
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 定期清理旧分区
DROP TABLE audit_log_2023_01;
```

---

### 8. 什么是行级安全（RLS）？

**答案：**

行级安全（Row Level Security）允许在数据库层面控制用户对行的访问权限。

**基本使用：**

```sql
-- 创建测试表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    owner_id INT
);

-- 启用行级安全
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- 创建策略
CREATE POLICY documents_owner_policy ON documents
    USING (owner_id = current_setting('app.current_user_id')::INT);

-- 策略说明：
-- USING: 用于 SELECT, UPDATE, DELETE 的可见性条件
-- WITH CHECK: 用于 INSERT, UPDATE 的数据验证条件

-- 创建用户并授权
CREATE ROLE app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO app_user;

-- 测试
SET ROLE app_user;
SET app.current_user_id = '1';

SELECT * FROM documents;  -- 只能看到 owner_id=1 的行
```

**策略类型：**

```sql
-- 1. SELECT 策略
CREATE POLICY select_policy ON documents
    FOR SELECT
    USING (owner_id = current_user_id());

-- 2. INSERT 策略
CREATE POLICY insert_policy ON documents
    FOR INSERT
    WITH CHECK (owner_id = current_user_id());

-- 3. UPDATE 策略
CREATE POLICY update_policy ON documents
    FOR UPDATE
    USING (owner_id = current_user_id())      -- 可见的才能更新
    WITH CHECK (owner_id = current_user_id()); -- 更新后的数据也必须满足条件

-- 4. DELETE 策略
CREATE POLICY delete_policy ON documents
    FOR DELETE
    USING (owner_id = current_user_id());

-- 5. ALL 策略（默认）
CREATE POLICY all_policy ON documents
    USING (owner_id = current_user_id())
    WITH CHECK (owner_id = current_user_id());
```

**多策略组合：**

```sql
-- 表可以有多个策略
-- 默认：任意一个策略满足即可（OR）

CREATE POLICY owner_policy ON documents
    USING (owner_id = current_user_id());

CREATE POLICY admin_policy ON documents
    USING (current_user_is_admin());

-- 用户看到：owner_id = 自己 OR 是管理员

-- 修改为 AND 关系
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
```

**绕过 RLS：**

```sql
-- 表所有者默认绕过 RLS
-- 禁止所有者绕过
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- 超级用户默认绕过 RLS
-- 只读用户绕过
ALTER ROLE readonly_user BYPASSRLS;

-- 查看表是否启用 RLS
SELECT 
    relname,
    relrowsecurity,
    relforcerowsecurity
FROM pg_class
WHERE relname = 'documents';
```

**实际应用示例：**

```sql
-- 多租户系统
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    name TEXT
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    tenant_id INT REFERENCES tenants(id),
    amount NUMERIC,
    created_at TIMESTAMP DEFAULT now()
);

-- 为每个租户创建角色
CREATE ROLE tenant_1;
CREATE ROLE tenant_2;

-- 启用 RLS
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- 创建策略
CREATE POLICY tenant_policy ON orders
    USING (tenant_id = current_setting('app.tenant_id')::INT);

-- 授权
GRANT SELECT, INSERT, UPDATE, DELETE ON orders TO tenant_1;
GRANT SELECT, INSERT, UPDATE, DELETE ON orders TO tenant_2;

-- 使用
SET ROLE tenant_1;
SET app.tenant_id = '1';
INSERT INTO orders (tenant_id, amount) VALUES (1, 100);  -- 成功
INSERT INTO orders (tenant_id, amount) VALUES (2, 100);  -- 失败
SELECT * FROM orders;  -- 只看到 tenant_id=1 的数据
```

**追问：RLS 对性能有什么影响？**

**追问答案：**

```sql
-- RLS 会增加查询条件，可能影响性能

-- 1. 确保 RLS 条件使用的列有索引
CREATE INDEX idx_owner ON documents(owner_id);

-- 2. 查看 RLS 策略
SELECT 
    schemaname,
    tablename,
    policyname,
    cmd,
    qual,        -- USING 条件
    with_check   -- WITH CHECK 条件
FROM pg_policies
WHERE tablename = 'documents';

-- 3. 使用 EXPLAIN 分析
EXPLAIN SELECT * FROM documents;
-- 会显示 RLS 过滤条件

-- 4. 性能影响
-- - 每次查询都需要评估策略条件
-- - 复杂策略可能影响查询计划
-- - 建议策略条件简单，并使用索引
```

---

## 常见问题与注意事项

### 1. 连接问题

```sql
-- 查看最大连接数
SHOW max_connections;

-- 查看当前连接
SELECT count(*) FROM pg_stat_activity;

-- 查看连接详情
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    query
FROM pg_stat_activity;

-- 终止连接
SELECT pg_terminate_backend(pid);

-- 取消查询
SELECT pg_cancel_backend(pid);
```

### 2. 锁问题

```sql
-- 查看锁等待
SELECT 
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database = blocked_locks.database
    AND blocking_locks.relation = blocked_locks.relation
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;

-- 查看锁类型
SELECT locktype, mode, granted FROM pg_locks WHERE pid = xxx;
```

### 3. 空间问题

```sql
-- 查看数据库大小
SELECT pg_size_pretty(pg_database_size(current_database()));

-- 查看表大小
SELECT 
    relname,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- 查看表膨胀
SELECT * FROM pgstattuple('tablename');
```

### 4. 性能调优清单

```sql
-- 1. 更新统计信息
ANALYZE;

-- 2. 检查缺失索引
SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;

-- 3. 检查缓存命中率
SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) 
FROM pg_statio_user_tables;

-- 4. 检查慢查询
SELECT * FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;

-- 5. 检查表膨胀
SELECT relname, n_dead_tup FROM pg_stat_user_tables ORDER BY n_dead_tup DESC;

-- 6. 手动 VACUUM
VACUUM ANALYZE tablename;
```

---

## 参考资料

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/current/index.html)
- [PostgreSQL 中文文档](http://www.postgres.cn/docs/15/index.html)
- [pgvector 扩展](https://github.com/pgvector/pgvector)
- [PgBouncer 连接池](https://www.pgbouncer.org/)
