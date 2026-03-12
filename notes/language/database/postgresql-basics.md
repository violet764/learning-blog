# PostgreSQL 学习笔记

PostgreSQL 是一款功能强大的开源关系型数据库管理系统，以其**扩展性、标准兼容性和数据完整性**著称。本文将从"为什么需要"的角度系统介绍 PostgreSQL 的核心概念和最佳实践。

<div align="center">
  <img src="https://www.postgresql.org/media/img/about/press/elephant.png" alt="pgsql-logo" width="120">
</div>

## 为什么选择 PostgreSQL？

### 🤔 背景：数据库选型的困惑

在项目初期，我们经常面临数据库选型的问题。MySQL 和 PostgreSQL 是最流行的两个开源关系型数据库，它们各有特点。理解它们的差异，才能做出正确的选择。

### PostgreSQL vs MySQL 深度对比

| 特性 | PostgreSQL | MySQL | 为什么重要？ |
|------|------------|-------|-------------|
| **架构理念** | 对象关系型，强调数据完整性 | 纯关系型，强调简单高效 | PostgreSQL 更适合复杂数据建模 |
| **JSON 支持** | JSONB 二进制存储，支持索引 | JSON 文本存储，索引有限 | JSONB 查询性能高 10-100 倍 |
| **全文搜索** | 内置高性能全文搜索 | 需要额外配置或使用 ElasticSearch | 减少架构复杂度 |
| **复杂查询** | 支持窗口函数、递归 CTE、LATERAL | 部分支持 | 分析型场景更强 |
| **并发控制** | MVCC 无 Undo Log，VACUUM 回收 | MVCC 使用 Undo Log 自动回收 | 各有优缺点（见后文） |
| **扩展性** | 支持自定义类型、索引、函数 | 扩展能力有限 | 可打造专属数据库 |
| **标准兼容** | 高度兼容 SQL 标准 | 部分兼容 | 移植性更好 |
| **社区生态** | 活跃，企业级特性多 | 最流行，生态最大 | 根据团队熟悉度选择 |

### 📊 选择决策树

```
你的需求是什么？
│
├─ 需要 JSONB 查询/分析？
│   └─ PostgreSQL（JSONB 索引性能优势明显）
│
├─ 需要复杂分析查询（窗口函数、递归）？
│   └─ PostgreSQL（MySQL 功能有限）
│
├─ 需要全文搜索？
│   └─ PostgreSQL（内置，无需额外组件）
│
├─ 需要地理空间计算？
│   └─ PostgreSQL + PostGIS（行业标准）
│
├─ 需要 AI 向量搜索？
│   └─ PostgreSQL + pgvector（原生支持）
│
├─ 团队只熟悉 MySQL，简单 CRUD？
│   └─ MySQL（学习成本低）
│
└─ 需要极高写入性能，简单查询？
    └─ MySQL（简单场景更高效）
```

### 💡 核心差异解析

#### 1. 为什么 PostgreSQL 更适合复杂场景？

```sql
-- PostgreSQL：一条 SQL 完成复杂分析
WITH RECURSIVE category_tree AS (
    -- 递归查询分类层级
    SELECT id, name, parent_id, 1 AS level
    FROM categories WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, ct.level + 1
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
),
sales_with_rank AS (
    -- 窗口函数计算排名
    SELECT 
        product_id,
        category_id,
        amount,
        RANK() OVER (PARTITION BY category_id ORDER BY amount DESC) AS rank
    FROM sales
)
SELECT 
    ct.name AS category,
    p.name AS product,
    swr.amount,
    swr.rank
FROM sales_with_rank swr
JOIN category_tree ct ON swr.category_id = ct.id
JOIN products p ON swr.product_id = p.id
WHERE swr.rank <= 3;  -- 每个分类销量前3

-- MySQL 8.0+ 也支持，但 PostgreSQL 优化更好
```

#### 2. 为什么 PostgreSQL 的 JSONB 更强？

```sql
-- 场景：电商产品属性查询

-- PostgreSQL JSONB：支持索引，查询飞快
CREATE INDEX idx_products_attr ON products USING GIN (attributes jsonb_path_ops);

-- 包含查询（使用索引）
SELECT * FROM products 
WHERE attributes @> '{"brand": "Apple", "specs": {"ram": 16}}';
-- 执行时间：0.5ms（百万级数据）

-- MySQL JSON：文本解析，性能差
SELECT * FROM products 
WHERE JSON_CONTAINS(attributes, '{"brand": "Apple"}');
-- 执行时间：500ms（需要全表扫描解析 JSON）

-- 为什么？PostgreSQL 的 JSONB 是二进制格式，预先解析好结构
-- MySQL 的 JSON 是文本格式，每次查询都要解析
```

---

## MVCC 实现：PostgreSQL vs MySQL

### 🤔 背景：为什么需要 MVCC？

在传统的锁机制下，读操作和写操作会互相阻塞：
- 读操作需要加共享锁，写操作需要加排他锁
- 如果有人正在写入，其他人无法读取（读阻塞）
- 如果有人正在读取，写入必须等待（写阻塞）

**MVCC（多版本并发控制）** 解决了这个问题：每个事务看到的是数据的一个"快照"，读写互不阻塞。

### PostgreSQL MVCC 实现原理

```
PostgreSQL MVCC 核心机制：

┌─────────────────────────────────────────────────────────────┐
│                    数据文件（实际存储）                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 行版本1: id=1, name='Alice', xmin=100, xmax=空      │    │
│  │         ↑ 插入事务ID=100，未被删除/更新               │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 行版本2: id=1, name='Bob', xmin=101, xmax=102       │    │
│  │         ↑ 事务101更新  ↑ 事务102删除                  │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 行版本3: id=1, name='Charlie', xmin=102, xmax=空    │    │
│  │         ↑ 事务102更新后的新版本                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

没有 Undo Log！所有版本都存在数据文件中！
```

### 📊 PostgreSQL vs MySQL MVCC 对比

| 特性 | PostgreSQL | MySQL (InnoDB) |
|------|------------|----------------|
| **版本存储位置** | 数据文件中（多版本共存） | Undo Log 中（回滚段） |
| **版本回收机制** | VACUUM 手动/自动清理 | 自动清理 Undo Log |
| **读操作** | 直接读数据文件 | 读数据文件 + 回滚 Undo Log |
| **表膨胀问题** | 有（死元组占用空间） | 无（版本在 Undo Log） |
| **事务回滚** | 极快（只标记 xmax） | 需要应用 Undo Log |
| **历史数据查询** | 需要扩展（pg_dirtyread） | 可通过 Undo Log 恢复 |

### 为什么 PostgreSQL 不用 Undo Log？

```
设计权衡：

PostgreSQL 方式（数据文件存储多版本）：
┌────────────────────────────────────────────┐
│ ✅ 优点：                                   │
│    • 读操作简单直接，不需要回滚 Undo Log     │
│    • 事务回滚极快（只标记 xmax）             │
│    • 不需要维护额外的 Undo 空间              │
│ ❌ 缺点：                                   │
│    • 数据文件膨胀（死元组堆积）              │
│    • 需要 VACUUM 定期清理                   │
│    • 清理期间可能影响性能                    │
└────────────────────────────────────────────┘

MySQL 方式（Undo Log 存储旧版本）：
┌────────────────────────────────────────────┐
│ ✅ 优点：                                   │
│    • 数据文件大小稳定                       │
│    • 自动清理，无需维护                      │
│    • 历史版本查询方便                       │
│ ❌ 缺点：                                   │
│    • 读旧版本需要回滚，稍慢                  │
│    • 事务回滚需要应用 Undo Log              │
│    • 需要管理 Undo 空间                     │
└────────────────────────────────────────────┘
```

### MVCC 实战演示

```sql
-- 查看行的版本信息
SELECT 
    id,
    name,
    xmin,           -- 插入该行的事务 ID
    xmax,           -- 删除/更新该行的事务 ID
    ctid,           -- 物理位置 (页号, 行号)
    cmin,           -- 命令序号（插入）
    cmax            -- 命令序号（删除）
FROM products;

-- 示例输出：
--  id | name   | xmin | xmax |  ctid  
-- ----+--------+------+------+--------
--   1 | Apple  |  100 |    0 | (0,1)  -- 原始版本
--   1 | Banana |  101 |  102 | (0,2)  -- 更新后的版本（又被删除）
--   1 | Cherry |  102 |    0 | (0,3)  -- 最新版本

-- xmin=100 表示事务100插入了这行
-- xmax=102 表示事务102更新/删除了这行
-- xmax=0 表示这行当前是有效的（未被删除）
```

```sql
-- MVCC 可见性规则演示

-- 会话1：开始事务，更新数据
BEGIN;
SELECT txid_current();  -- 假设返回 1000
UPDATE products SET name = 'Updated' WHERE id = 1;
-- 此时：旧行 xmax=1000，新行 xmin=1000

-- 会话2：并行查询（看不到未提交的更新）
BEGIN;
SELECT txid_current();  -- 假设返回 1001
SELECT * FROM products WHERE id = 1;  
-- 仍然看到旧值！因为事务1000未提交

-- 会话1：提交
COMMIT;

-- 会话2：再次查询（仍然看不到更新！）
SELECT * FROM products WHERE id = 1;
-- REPEATABLE READ 隔离级别下，看到的是事务开始时的快照

-- 会话2：结束事务后重新查询
COMMIT;
BEGIN;
SELECT * FROM products WHERE id = 1;  -- 现在看到新值
COMMIT;
```

---

## VACUUM：为什么 PostgreSQL 需要？

### 🤔 背景：什么是表膨胀？

由于 PostgreSQL 的 MVCC 将所有版本都存在数据文件中，当数据频繁更新/删除时：

```
更新操作前：
┌────────────────────┐
│ 行A (有效)         │
│ 行B (有效)         │
│ 行C (有效)         │
└────────────────────┘

更新操作后（更新了行B）：
┌────────────────────┐
│ 行A (有效)         │
│ 行B (无效/死元组)  │ ← 旧版本变成"死元组"
│ 行C (有效)         │
│ 行B' (有效)        │ ← 新版本插入到末尾
└────────────────────┘

多次更新后：
┌────────────────────┐
│ 行A (有效)         │
│ 行B (死元组)       │
│ 行C (有效)         │
│ 行B' (死元组)      │
│ 行B'' (死元组)     │
│ 行B''' (有效)      │ ← 有效数据只有50%！
└────────────────────┘
表膨胀！空间浪费，查询变慢！
```

### 为什么 MySQL 没有这个问题？

```
MySQL (InnoDB) 的设计：

更新操作前：
┌────────────────────┐     ┌────────────────┐
│ 行A (有效)         │     │                │
│ 行B (有效)         │     │   Undo Log     │
│ 行C (有效)         │     │   (空)         │
└────────────────────┘     └────────────────┘

更新操作后（更新行B）：
┌────────────────────┐     ┌────────────────┐
│ 行A (有效)         │     │ 行B旧值        │ ← 旧值存入 Undo Log
│ 行B (新值)         │     │                │
│ 行C (有效)         │     │                │
└────────────────────┘     └────────────────┘

MySQL 直接在原位更新（某些情况），旧值存在 Undo Log
Undo Log 自动清理，不存在表膨胀问题
```

### VACUUM 的作用

```sql
-- VACUUM 的核心功能：
-- 1. 标记死元组的空间为可重用
-- 2. 更新表的统计信息（配合 ANALYZE）
-- 3. 防止事务 ID 回卷（VACUUM FREEZE）

-- 查看表的死元组数量
SELECT 
    relname AS 表名,
    n_live_tup AS 活跃元组,
    n_dead_tup AS 死元组,
    ROUND(n_dead_tup::numeric / 
          NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS 死元组比例
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- 示例输出：
--    表名    | 活跃元组 | 死元组 | 死元组比例
-- -----------+----------+--------+-----------
--  orders    |   100000 |  50000 |     33.33  ← 需要VACUUM！
--  products  |     5000 |    100 |      2.00
```

```sql
-- VACUUM 操作类型

-- 普通 VACUUM：标记空间可重用，不阻塞
VACUUM products;

-- VACUUM ANALYZE：清理 + 更新统计信息
VACUUM ANALYZE products;

-- VACUUM FULL：重建整个表，释放空间给操作系统
-- ⚠️ 警告：会锁表！生产环境慎用！
VACUUM FULL products;

-- 自动清理配置（postgresql.conf）
autovacuum = on                              -- 启用自动清理
autovacuum_vacuum_threshold = 50             -- 最少死元组数
autovacuum_vacuum_scale_factor = 0.1         -- 死元组达到10%时触发
autovacuum_analyze_scale_factor = 0.05       -- 变化达到5%时分析

-- 表级配置（覆盖全局设置）
ALTER TABLE hot_table SET (
    autovacuum_vacuum_scale_factor = 0.02,   -- 更激进的清理
    autovacuum_vacuum_threshold = 100
);
```

### VACUUM 最佳实践

```sql
-- 1. 监控自动清理状态
SELECT 
    relname AS 表名,
    last_vacuum AS 上次手动清理,
    last_autovacuum AS 上次自动清理,
    vacuum_count AS 手动清理次数,
    autovacuum_count AS 自动清理次数
FROM pg_stat_user_tables;

-- 2. 大批量更新分批进行
-- ❌ 不好的做法：一次性更新百万行
UPDATE large_table SET status = 'inactive' WHERE date < '2020-01-01';

-- ✅ 好的做法：分批更新
DO $$
DECLARE
    rows_affected INTEGER := 1;
BEGIN
    WHILE rows_affected > 0 LOOP
        UPDATE large_table 
        SET status = 'inactive' 
        WHERE date < '2020-01-01' 
          AND status != 'inactive'
          AND ctid IN (
              SELECT ctid FROM large_table 
              WHERE date < '2020-01-01' AND status != 'inactive'
              LIMIT 10000
          );
        
        GET DIAGNOSTICS rows_affected = ROW_COUNT;
        COMMIT;  -- 每批提交，让 autovacuum 有机会工作
    END LOOP;
END $$;

-- 3. 对频繁更新的表使用 pg_repack 扩展
-- 可以在不锁表的情况下重建表
```

---

## TOAST：大字段存储机制

### 🤔 背景：为什么需要 TOAST？

PostgreSQL 的数据页大小固定为 8KB，如果一个表的某行数据超过 8KB 怎么办？

```
问题场景：
┌────────────────────────────────────────┐
│ 产品描述字段存储了大段文本（50KB）        │
│ 无法放入单个 8KB 的数据页！              │
└────────────────────────────────────────┘

解决方案：TOAST (The Oversized-Attribute Storage Technique)
┌────────────────────────────────────────┐
│ 主表：存储指针                           │
│ ┌────────────────────────────────────┐ │
│ │ id=1, desc=TOAST指针 →             │ │
│ └────────────────────────────────────┘ │
│                                        │
│ TOAST表：存储实际大字段数据              │
│ ┌────────────────────────────────────┐ │
│ │ chunk_id=1, chunk_data='前8KB...'  │ │
│ │ chunk_id=1, chunk_data='中8KB...'  │ │
│ │ chunk_id=1, chunk_data='后8KB...'  │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
```

### TOAST 存储策略

```sql
-- PostgreSQL 四种 TOAST 策略

-- PLAIN：不压缩，不外部存储
-- 适用：小字段，不能超过 8KB
ALTER TABLE products ALTER COLUMN code SET STORAGE PLAIN;

-- EXTENDED（默认）：先压缩，再外部存储
-- 适用：大文本、JSONB
ALTER TABLE products ALTER COLUMN description SET STORAGE EXTENDED;

-- EXTERNAL：不压缩，直接外部存储
-- 适用：需要部分访问的大字段（如大文本搜索）
ALTER TABLE products ALTER COLUMN content SET STORAGE EXTERNAL;

-- MAIN：压缩，尽量不外部存储
-- 适用：希望压缩但避免外部存储开销
ALTER TABLE products ALTER COLUMN summary SET STORAGE MAIN;
```

### TOAST 实战示例

```sql
-- 创建包含大字段的表
CREATE TABLE articles (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,              -- 可能很大，自动使用 TOAST
    metadata JSONB             -- 也可能很大
);

-- 查看表的 TOAST 表
SELECT 
    relname AS 原表,
    (SELECT relname FROM pg_class WHERE oid = t.reltoastrelid) AS TOAST表
FROM pg_class t
WHERE relname = 'articles';

-- 输出示例：
--   原表    |   TOAST表
-- ----------+-----------------
--  articles | pg_toast_12345

-- TOAST 表结构（系统自动创建）
-- chunk_id   - TOAST 指针
-- chunk_seq  - 块序号
-- chunk_data - 实际数据块（每块约 2KB）

-- 访问大字段时自动处理，对用户透明
SELECT title, length(content) AS content_length
FROM articles
WHERE id = 1;
```

### TOAST 性能优化

```sql
-- 问题：访问 TOAST 字段有额外开销

-- ❌ 不好的做法：查询所有字段，包括不需要的大字段
SELECT * FROM articles;  -- 即使只需要 title，也要加载 TOAST

-- ✅ 好的做法：只查询需要的字段
SELECT id, title FROM articles;

-- ✅ 大字段单独查询
SELECT content FROM articles WHERE id = 1;

-- 使用 EXTERNAL 策略支持部分访问
-- 场景：只需要大文本的前 100 个字符
ALTER TABLE articles ALTER COLUMN content SET STORAGE EXTERNAL;
SELECT id, left(content, 100) AS preview FROM articles;
-- EXTERNAL 策略下，不需要加载整个 TOAST 值
```

---

## 丰富数据类型

PostgreSQL 提供了远超传统数据库的丰富数据类型，这是其核心优势之一。

### 为什么需要丰富数据类型？

```
传统数据库的问题：
┌──────────────────────────────────────────────────────────┐
│ 场景1：电商产品属性                                        │
│   不同产品有不同属性（手机：CPU、内存；衣服：尺码、颜色）    │
│   传统方案：多个关联表，查询复杂                            │
│   PostgreSQL 方案：JSONB 一列搞定                          │
├──────────────────────────────────────────────────────────┤
│ 场景2：文章标签                                            │
│   一篇文章有多个标签                                        │
│   传统方案：文章表 + 标签表 + 关联表                        │
│   PostgreSQL 方案：数组类型，一列存储                       │
├──────────────────────────────────────────────────────────┤
│ 场景3：酒店预订                                            │
│   需要存储入住日期范围，检查重叠                            │
│   传统方案：开始日期 + 结束日期，复杂查询                   │
│   PostgreSQL 方案：范围类型，原生支持重叠检测               │
└──────────────────────────────────────────────────────────┘
```

### JSONB 类型

#### 为什么需要 JSONB？

```sql
-- 传统方案：为每种产品建表
CREATE TABLE phones (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    cpu VARCHAR(50),
    ram INT,
    storage INT
);

CREATE TABLE clothes (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    size VARCHAR(10),
    color VARCHAR(20)
);

-- 问题：产品类型多了怎么办？每次都要建表、改代码？

-- PostgreSQL JSONB 方案：一张表搞定
CREATE TABLE products (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),           -- 产品类别
    attributes JSONB NOT NULL       -- 灵活的属性存储
);

-- 插入不同类型的产品
INSERT INTO products (name, category, attributes) VALUES
('iPhone 15', 'phone', '{"cpu": "A17", "ram": 8, "storage": 256}'),
('T-Shirt', 'clothes', '{"size": "L", "color": "blue", "material": "cotton"}'),
('MacBook', 'laptop', '{"cpu": "M3", "ram": 16, "storage": 512, "screen": 14}');
```

#### JSONB vs JSON

| 特性 | JSON | JSONB |
|------|------|-------|
| **存储格式** | 文本（原样存储） | 二进制（预先解析） |
| **插入速度** | 快（无需解析） | 慢（需要解析转换） |
| **查询速度** | 慢（每次解析） | 快（结构已解析） |
| **索引支持** | 有限 | 完整支持 GIN 索引 |
| **空格保留** | 保留 | 去除 |
| **键顺序** | 保留 | 排序 |

**结论：查询场景用 JSONB，仅存储不查询用 JSON**

#### JSONB 查询操作

```sql
-- 创建示例数据
CREATE TABLE products (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    attributes JSONB NOT NULL
);

INSERT INTO products (name, attributes) VALUES
('MacBook Pro', '{
    "brand": "Apple",
    "specs": {
        "cpu": "M3 Pro",
        "ram": 18,
        "storage": 512
    },
    "tags": ["laptop", "professional", "apple"],
    "price": 1999.00,
    "in_stock": true
}');

-- 1. 提取值
-- -> 返回 JSON 类型，->> 返回文本类型
SELECT 
    name,
    attributes->>'brand' AS brand,                    -- 文本
    attributes->'specs'->>'cpu' AS cpu,              -- 嵌套提取
    attributes->'specs'->'ram' AS ram_json           -- 保持 JSON 格式
FROM products;

-- 2. 条件查询
SELECT name FROM products 
WHERE attributes->>'brand' = 'Apple';                -- 文本比较

SELECT name FROM products 
WHERE (attributes->'specs'->'ram')::int > 16;        -- 类型转换

-- 3. 包含查询（推荐！使用索引）
SELECT name FROM products 
WHERE attributes @> '{"brand": "Apple"}';            -- 包含某个键值

SELECT name FROM products 
WHERE attributes @> '{"specs": {"ram": 18}}';        -- 嵌套包含

-- 4. 键存在检查
SELECT name FROM products 
WHERE attributes ? 'price';                          -- 存在某个键

SELECT name FROM products 
WHERE attributes ?& ARRAY['price', 'brand'];         -- 同时存在多个键

SELECT name FROM products 
WHERE attributes ?| ARRAY['price', 'brand'];         -- 存在任意一个键

-- 5. 数组查询
SELECT name FROM products 
WHERE attributes->'tags' ? 'professional';           -- 数组包含元素

-- 6. JSONB 路径查询（PostgreSQL 12+）
SELECT name FROM products 
WHERE attributes @? '$.specs.ram ? (@ > 16)';        -- JSONPath 语法
```

#### JSONB 索引优化

```sql
-- GIN 索引（通用）
CREATE INDEX idx_products_attr_gin ON products USING GIN (attributes);

-- GIN jsonb_path_ops 索引（更小更快，只支持 @> 操作符）
CREATE INDEX idx_products_attr_path ON products USING GIN (attributes jsonb_path_ops);

-- 表达式索引（针对特定字段）
CREATE INDEX idx_products_brand ON products ((attributes->>'brand'));
CREATE INDEX idx_products_price ON products (((attributes->>'price')::numeric));

-- 索引选择建议：
-- @> 包含查询为主 → jsonb_path_ops（更小更快）
-- 其他操作符 ? ?| ?& → 普通 GIN
-- 特定字段查询 → 表达式索引
```

#### JSONB 修改操作

```sql
-- 添加/更新字段（|| 合并）
UPDATE products 
SET attributes = attributes || '{"warranty": "1 year"}'::jsonb
WHERE id = 1;

-- 更新嵌套字段（jsonb_set）
UPDATE products
SET attributes = jsonb_set(
    attributes, 
    '{specs,ram}',     -- 路径
    '36'::jsonb        -- 新值
)
WHERE id = 1;

-- 删除字段（- 操作符）
UPDATE products
SET attributes = attributes - 'warranty'
WHERE id = 1;

-- 删除嵌套字段（#- 操作符）
UPDATE products
SET attributes = attributes #- '{specs,storage}'
WHERE id = 1;
```

### 数组类型

#### 为什么需要数组类型？

```sql
-- 场景：文章标签系统

-- 传统方案：需要三张表
-- articles(id, title, content)
-- tags(id, name)
-- article_tags(article_id, tag_id)

-- 查询某文章的所有标签：
SELECT t.name FROM tags t
JOIN article_tags at ON t.id = at.tag_id
WHERE at.article_id = 1;

-- PostgreSQL 数组方案：一张表搞定
CREATE TABLE articles (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    tags TEXT[] DEFAULT '{}'        -- 数组类型
);

-- 插入数据
INSERT INTO articles (title, tags) VALUES
('PostgreSQL 教程', ARRAY['database', 'sql', 'tutorial']),
('JSONB 详解', '{json, postgresql, advanced}');

-- 查询：简单直接！
SELECT title, tags FROM articles WHERE 'sql' = ANY(tags);
SELECT title, tags FROM articles WHERE tags @> ARRAY['database'];
```

#### 数组操作详解

```sql
-- 创建示例表
CREATE TABLE posts (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    ratings INTEGER[] DEFAULT '{}'
);

INSERT INTO posts (title, tags, ratings) VALUES
('PostgreSQL 教程', '{database, sql, tutorial}', '{5, 4, 5, 4}'),
('JSONB 详解', '{json, postgresql}', '{5, 5, 4}');

-- 1. 访问数组元素（索引从1开始！）
SELECT title, tags[1] AS first_tag, ratings[1:2] AS first_two
FROM posts;

-- 2. 数组长度
SELECT title, array_length(tags, 1) AS tag_count, cardinality(ratings) AS rating_count
FROM posts;

-- 3. 包含查询
SELECT title FROM posts WHERE 'sql' = ANY(tags);         -- ANY 语法
SELECT title FROM posts WHERE tags @> ARRAY['database'];  -- 包含操作符
SELECT title FROM posts WHERE tags && ARRAY['json', 'redis']; -- 重叠（任意一个匹配）

-- 4. 展开数组（unnest）
SELECT title, unnest(tags) AS tag FROM posts;

-- 5. 聚合为数组（array_agg）
SELECT array_agg(title) AS all_titles FROM posts;

-- 6. 数组函数
SELECT 
    title,
    array_length(tags, 1) AS 长度,
    array_to_string(tags, ', ') AS 拼接字符串,
    array_remove(tags, 'sql') AS 移除元素,
    array_append(tags, 'new') AS 追加元素,
    array_cat(tags, ARRAY['extra']) AS 连接数组
FROM posts;

-- 7. 数组索引（GIN）
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
```

### 范围类型

#### 为什么需要范围类型？

```sql
-- 场景：酒店预订系统，检查日期重叠

-- 传统方案：两个日期字段
CREATE TABLE reservations (
    id INT PRIMARY KEY,
    room_id INT,
    check_in DATE,
    check_out DATE
);

-- 检查重叠：复杂的条件判断
SELECT * FROM reservations r1, reservations r2
WHERE r1.id < r2.id
  AND r1.room_id = r2.room_id
  AND r1.check_in < r2.check_out
  AND r1.check_out > r2.check_in;  -- 很容易写错！

-- PostgreSQL 范围类型：原生支持
CREATE TABLE reservations (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    room_id INT NOT NULL,
    stay_dates DATERANGE NOT NULL
);

-- 插入数据（使用范围字面量）
INSERT INTO reservations (room_id, stay_dates) VALUES
(101, '[2024-03-01, 2024-03-05)'),  -- 闭开区间，3月1日入住，3月5日退房
(102, '[2024-03-03, 2024-03-07)');

-- 重叠检查：一个操作符搞定！
SELECT r1.room_id, r1.stay_dates, r2.stay_dates
FROM reservations r1, reservations r2
WHERE r1.id < r2.id AND r1.stay_dates && r2.stay_dates;

-- 某日期是否在范围内
SELECT * FROM reservations WHERE stay_dates @> DATE '2024-03-03';

-- 创建 GIST 索引加速范围查询
CREATE INDEX idx_reservations_dates ON reservations USING GIST (stay_dates);
```

#### 范围类型详解

```sql
-- 内置范围类型
-- int4range  - 整数范围
-- int8range  - 大整数范围
-- numrange   - 数值范围
-- tsrange    - 时间戳范围（无时区）
-- tstzrange  - 时间戳范围（有时区）
-- daterange  - 日期范围

-- 范围操作符
-- @>  包含
-- <@  被包含
-- &&  重叠
-- <<  严格左侧
-- >>  严格右侧
-- &<  不延伸到右侧
-- &>  不延伸到左侧
-- -|- 相邻

-- 实战示例
INSERT INTO reservations (room_id, stay_dates, price_range, meeting_time) VALUES
(101, '[2024-03-01, 2024-03-05)', '[100, 200)', '[2024-03-01 14:00, 2024-03-01 16:00)');

-- 范围函数
SELECT 
    room_id,
    lower(stay_dates) AS 开始日期,
    upper(stay_dates) AS 结束日期,
    upper(stay_dates) - lower(stay_dates) AS 天数
FROM reservations;

-- 排除约束：防止重叠预订
CREATE TABLE meetings (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    room_id INT NOT NULL,
    during TSRANGE,
    EXCLUDE USING GIST (
        room_id WITH =,    -- 同一房间
        during WITH &&     -- 时间重叠
    )
);

-- 尝试插入重叠会议会被拒绝！
INSERT INTO meetings (room_id, during) VALUES
(1, '[2024-03-01 10:00, 2024-03-01 11:00)'),
(1, '[2024-03-01 10:30, 2024-03-01 12:00)');  -- 错误！冲突
```

### 几何类型与 PostGIS

```sql
-- PostgreSQL 内置几何类型
-- point   - 点
-- line    - 线
-- lseg    - 线段
-- box     - 矩形
-- path    - 路径
-- polygon - 多边形
-- circle  - 圆

-- 简单几何计算
SELECT 
    point '(0,0)' <-> point '(3,4)' AS 距离;  -- 结果: 5

-- 复杂地理计算：使用 PostGIS 扩展
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE stores (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location GEOMETRY(Point, 4326)  -- WGS84 坐标系
);

-- 插入位置数据
INSERT INTO stores (name, location) VALUES
('北京店', ST_SetSRID(ST_MakePoint(116.4074, 39.9042), 4326)),
('上海店', ST_SetSRID(ST_MakePoint(121.4737, 31.2304), 4326));

-- 查找距离某点 100km 内的门店
SELECT name,
    ST_Distance(location::geography, 
                ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography) / 1000 AS distance_km
FROM stores
WHERE ST_DWithin(location::geography, 
                 ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography, 
                 100000)  -- 100km
ORDER BY distance_km;

-- 创建空间索引
CREATE INDEX idx_stores_location ON stores USING GIST (location);
```

---

## 高级 SQL 特性

PostgreSQL 支持丰富的高级 SQL 特性，让复杂查询变得简单。

### 窗口函数

#### 为什么需要窗口函数？

```sql
-- 场景：计算每个销售的累计金额和排名

-- 传统方案：需要子查询或多次查询
SELECT s1.*,
    (SELECT SUM(amount) FROM sales s2 WHERE s2.sale_date <= s1.sale_date) AS running_total,
    (SELECT COUNT(*) + 1 FROM sales s2 WHERE s2.amount > s1.amount) AS rank
FROM sales s1;
-- 性能差：O(n²) 复杂度

-- PostgreSQL 窗口函数：一行搞定
SELECT 
    id, product_name, amount, sale_date,
    SUM(amount) OVER (ORDER BY sale_date) AS running_total,
    RANK() OVER (ORDER BY amount DESC) AS rank
FROM sales;
-- 性能好：O(n log n) 复杂度
```

#### 窗口函数详解

```sql
-- 示例数据
CREATE TABLE sales (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_name VARCHAR(100),
    region VARCHAR(50),
    amount DECIMAL(10, 2),
    sale_date DATE DEFAULT CURRENT_DATE
);

INSERT INTO sales (product_name, region, amount, sale_date) VALUES
('Laptop', 'North', 1200.00, '2024-01-01'),
('Phone', 'North', 800.00, '2024-01-01'),
('Laptop', 'South', 1200.00, '2024-01-02'),
('Tablet', 'North', 500.00, '2024-01-02'),
('Phone', 'South', 850.00, '2024-01-03'),
('Laptop', 'East', 1100.00, '2024-01-03');

-- 窗口函数语法：
-- 函数名() OVER (
--     PARTITION BY 分组列      -- 可选
--     ORDER BY 排序列          -- 可选
--     ROWS/RANGE 窗口框架      -- 可选
-- )

-- 1. 聚合窗口函数
SELECT 
    product_name,
    region,
    amount,
    SUM(amount) OVER () AS 总销售额,
    SUM(amount) OVER (PARTITION BY region) AS 区域销售额,
    SUM(amount) OVER (ORDER BY sale_date) AS 累计销售额,
    AVG(amount) OVER (PARTITION BY region) AS 区域平均
FROM sales;

-- 2. 排名函数
SELECT 
    product_name,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS 行号,      -- 连续编号
    RANK() OVER (ORDER BY amount DESC) AS 排名,           -- 相同值排名相同，跳过
    DENSE_RANK() OVER (ORDER BY amount DESC) AS 紧凑排名, -- 相同值排名相同，不跳过
    NTILE(4) OVER (ORDER BY amount DESC) AS 四分位        -- 分成4组
FROM sales;

-- 3. 移动窗口
SELECT 
    product_name,
    amount,
    sale_date,
    AVG(amount) OVER (
        ORDER BY sale_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- 当前行及前2行
    ) AS 移动平均_3天,
    SUM(amount) OVER (
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- 从开始到当前
    ) AS 累计和
FROM sales;

-- 4. 偏移函数
SELECT 
    product_name,
    amount,
    sale_date,
    LAG(amount, 1) OVER (ORDER BY sale_date) AS 上一次金额,
    LEAD(amount, 1) OVER (ORDER BY sale_date) AS 下一次金额,
    LAG(amount, 1, 0) OVER (ORDER BY sale_date) AS 上一次金额_默认0,
    FIRST_VALUE(amount) OVER (PARTITION BY region ORDER BY sale_date) AS 区域首次金额,
    LAST_VALUE(amount) OVER (
        PARTITION BY region 
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS 区域最后金额
FROM sales;

-- 5. 实战：同比增长计算
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', sale_date) AS month,
        SUM(amount) AS monthly_total
    FROM sales
    GROUP BY DATE_TRUNC('month', sale_date)
)
SELECT 
    month,
    monthly_total,
    LAG(monthly_total, 1) OVER (ORDER BY month) AS 上月销售额,
    ROUND(
        (monthly_total - LAG(monthly_total, 1) OVER (ORDER BY month)) / 
        LAG(monthly_total, 1) OVER (ORDER BY month) * 100, 
    2) AS 环比增长率
FROM monthly_sales;
```

### CTE（公共表表达式）

#### 为什么需要 CTE？

```sql
-- 场景：多步骤复杂查询

-- 传统方案：嵌套子查询，难以阅读
SELECT * FROM (
    SELECT * FROM (
        SELECT region, AVG(amount) AS avg_amount
        FROM sales
        GROUP BY region
    ) r
    WHERE avg_amount > 1000
) t
WHERE ...

-- PostgreSQL CTE：清晰易读
WITH regional_avg AS (
    -- 第一步：计算区域平均
    SELECT region, AVG(amount) AS avg_amount
    FROM sales
    GROUP BY region
),
high_value_regions AS (
    -- 第二步：筛选高价值区域
    SELECT * FROM regional_avg WHERE avg_amount > 1000
)
-- 第三步：最终查询
SELECT s.*, h.avg_amount
FROM sales s
JOIN high_value_regions h ON s.region = h.region;
```

#### CTE 高级用法

```sql
-- 1. CTE 数据修改（WITH ... RETURNING）
WITH deleted AS (
    DELETE FROM sales WHERE amount < 100
    RETURNING *
)
INSERT INTO sales_archive SELECT * FROM deleted;

-- 2. 递归 CTE（处理层次结构）
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    manager_id INT REFERENCES employees(id)
);

INSERT INTO employees VALUES
(1, 'CEO', NULL),
(2, 'CTO', 1),
(3, 'CFO', 1),
(4, 'Dev Manager', 2),
(5, 'Senior Dev', 4);

WITH RECURSIVE org_tree AS (
    -- 锚点：顶层
    SELECT id, name, manager_id, 1 AS level, ARRAY[id] AS path
    FROM employees WHERE manager_id IS NULL
    
    UNION ALL
    
    -- 递归：下属
    SELECT e.id, e.name, e.manager_id, t.level + 1, t.path || e.id
    FROM employees e
    JOIN org_tree t ON e.manager_id = t.id
)
SELECT 
    REPEAT('  ', level - 1) || name AS 组织架构,
    level,
    path
FROM org_tree
ORDER BY path;

-- 结果：
--   组织架构     | level |  path
-- ---------------+-------+---------
--   CEO          |     1 | {1}
--     CTO        |     2 | {1,2}
--       Dev Mgr  |     3 | {1,2,4}
--         Sr Dev |     4 | {1,2,4,5}
--     CFO        |     2 | {1,3}

-- 3. MATERIALIZED vs NOT MATERIALIZED
WITH MATERIALIZED expensive AS (
    -- 复杂查询只执行一次，结果物化
    SELECT * FROM large_table WHERE complex_condition
)
SELECT * FROM expensive WHERE ...;

WITH NOT MATERIALIZED simple AS (
    -- 简单查询可能被内联优化
    SELECT * FROM small_table
)
SELECT * FROM simple WHERE ...;
```

### LATERAL JOIN

#### 为什么需要 LATERAL JOIN？

```sql
-- 场景：获取每个客户最近3个订单

-- 传统方案：使用窗口函数和子查询
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
    FROM orders
) t WHERE rn <= 3;

-- 问题：如果需要关联其他表的详细信息，会很复杂

-- PostgreSQL LATERAL：简洁优雅
SELECT 
    c.id AS customer_id,
    c.name,
    recent.order_id,
    recent.order_date,
    recent.total
FROM customers c
CROSS JOIN LATERAL (
    SELECT id AS order_id, order_date, total
    FROM orders o
    WHERE o.customer_id = c.id  -- 可以引用外部表的列！
    ORDER BY order_date DESC
    LIMIT 3
) recent;

-- LATERAL 的关键特性：
-- 子查询可以引用外部查询的列（类似相关子查询）
-- 但每行执行一次，结果可以返回多行
```

#### LATERAL 实战示例

```sql
-- 1. 获取每个产品的最贵订单项
SELECT 
    p.name AS product_name,
    top.order_id,
    top.quantity,
    top.price
FROM products p
CROSS JOIN LATERAL (
    SELECT order_id, quantity, price
    FROM order_items oi
    WHERE oi.product_id = p.id
    ORDER BY price DESC
    LIMIT 1
) top;

-- 2. LATERAL 与 JSONB 结合
SELECT 
    p.name,
    attr.key,
    attr.value
FROM products p
CROSS JOIN LATERAL jsonb_each_text(p.attributes) AS attr(key, value)
WHERE attr.key IN ('brand', 'price');

-- 3. 计算每组的 Top N
-- 场景：每个地区销量前3的产品
SELECT 
    region,
    product_name,
    amount,
    rank
FROM regions r
CROSS JOIN LATERAL (
    SELECT 
        product_name,
        amount,
        RANK() OVER (ORDER BY amount DESC) AS rank
    FROM sales s
    WHERE s.region = r.name
    LIMIT 3
) top_products;
```

---

## 索引类型详解

PostgreSQL 提供多种索引类型，每种都有其特定的适用场景。

### 为什么需要多种索引类型？

```
不同查询模式需要不同的索引结构：

┌─────────────────────────────────────────────────────────────────┐
│ 查询类型              │ 最适合的索引   │ 原因                    │
├─────────────────────────────────────────────────────────────────┤
│ 等值查询 (a=1)        │ B-tree        │ 平衡树，O(log n) 查找    │
│ 范围查询 (a>10)       │ B-tree        │ 有序存储，范围扫描       │
│ 排序查询 (ORDER BY a) │ B-tree        │ 直接按索引顺序返回       │
│ JSON 包含查询         │ GIN           │ 倒排索引，多值元素       │
│ 数组包含查询          │ GIN           │ 倒排索引，元素查找       │
│ 全文搜索              │ GIN/GiST      │ 词频索引                │
│ 范围重叠查询          │ GiST          │ R-tree，空间索引        │
│ 地理位置查询          │ GiST          │ R-tree，空间索引        │
│ 超大时序表            │ BRIN          │ 块范围摘要，极小         │
└─────────────────────────────────────────────────────────────────┘
```

### B-tree 索引（默认）

```sql
-- B-tree 是最通用的索引，适用于：等值、范围、排序、前缀 LIKE

-- 创建 B-tree 索引
CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_products_price ON products((attributes->>'price')::numeric);

-- 复合索引（注意列顺序！）
-- 最左前缀原则：查询条件必须从左边开始匹配
CREATE INDEX idx_products_category_price ON products(category, price);

-- 有效使用索引的查询：
WHERE category = 'electronics'                    -- 使用索引
WHERE category = 'electronics' AND price > 100    -- 使用索引
WHERE price > 100                                 -- 不使用索引！缺少最左列

-- 条件索引（部分索引）
CREATE INDEX idx_products_active ON products(name) 
WHERE status = 'active';  -- 只索引活跃产品

-- 表达式索引
CREATE INDEX idx_products_name_lower ON products(LOWER(name));
-- 查询时使用：WHERE LOWER(name) = 'iphone'

-- 查看索引定义
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'products';
```

### GIN 索引

```sql
-- GIN (Generalized Inverted Index) 倒排索引
-- 适用：多值元素查询（数组、JSONB、全文搜索）

-- 1. JSONB GIN 索引
CREATE INDEX idx_products_attr_gin ON products USING GIN (attributes);
-- 支持操作符：@> ? ?| ?&

-- jsonb_path_ops 更小更快，但只支持 @>
CREATE INDEX idx_products_attr_path ON products USING GIN (attributes jsonb_path_ops);

-- 2. 数组 GIN 索引
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
-- 查询：WHERE tags @> ARRAY['database']

-- 3. 全文搜索 GIN 索引
-- 创建全文搜索向量列
ALTER TABLE posts ADD COLUMN content_tsv TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED;

CREATE INDEX idx_posts_fts ON posts USING GIN (content_tsv);

-- 全文搜索查询
SELECT title FROM posts 
WHERE content_tsv @@ to_tsquery('postgresql & tutorial');

-- GIN 索引特点：
-- ✅ 查询快（多值元素查找）
-- ✅ 支持多种操作符
-- ❌ 建立索引慢（需要构建倒排列表）
-- ❌ 索引较大
-- ❌ 写入性能影响大
```

### GiST 索引

```sql
-- GiST (Generalized Search Tree) 通用搜索树
-- 适用：范围类型、几何类型、全文搜索

-- 1. 范围类型 GiST 索引
CREATE INDEX idx_reservations_dates ON reservations USING GIST (stay_dates);
-- 支持操作符：&& @> <@ << >> -|-

-- 2. 几何类型 GiST 索引
CREATE INDEX idx_locations_position ON locations USING GIST (position);

-- 3. PostGIS 空间索引
CREATE INDEX idx_stores_geom ON stores USING GIST (geom);

-- 4. 排除约束（使用 GiST）
CREATE TABLE meetings (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    room_id INT NOT NULL,
    during TSRANGE,
    EXCLUDE USING GIST (
        room_id WITH =,   -- 同一房间
        during WITH &&    -- 时间重叠时排除
    )
);

-- GiST 索引特点：
-- ✅ 支持复杂查询（范围、空间）
-- ✅ 支持排除约束
-- ❌ 查询可能需要回表验证
-- ❌ 建立索引较慢
```

### BRIN 索引

```sql
-- BRIN (Block Range Index) 块范围索引
-- 适用：超大表、有序数据（如时序数据）

-- 创建 BRIN 索引
CREATE INDEX idx_logs_created_at ON logs USING BRIN (created_at);

-- 多列 BRIN
CREATE INDEX idx_logs_time_region ON logs USING BRIN (created_at, region_id)
WITH (pages_per_range = 128);

-- BRIN 原理：
-- ┌────────────────────────────────────────────────┐
-- │ 数据文件（按时间顺序存储）                        │
-- │ [页1-128] 时间范围: 2024-01-01 ~ 2024-01-02    │
-- │ [页129-256] 时间范围: 2024-01-03 ~ 2024-01-04  │
-- │ ...                                            │
-- └────────────────────────────────────────────────┘
-- 
-- BRIN 索引（极小，只存储摘要）：
-- [块1] min=01-01, max=01-02
-- [块2] min=01-03, max=01-04
-- 
-- 查询时：快速跳过不相关的块

-- 索引大小对比
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes WHERE tablename = 'logs';

-- 示例结果：
-- B-tree: 1 GB
-- BRIN:   128 KB   ← 小得多！

-- BRIN 索引特点：
-- ✅ 极小的索引大小
-- ✅ 几乎不影响写入性能
-- ✅ 适合超大表
-- ❌ 只适合有序数据
-- ❌ 查询需要扫描更多块
```

### 索引选择指南

```sql
-- 索引选择决策树：

-- 1. 查询类型是什么？
--    等值/范围/排序 → B-tree
--    JSONB 包含 → GIN (jsonb_path_ops)
--    数组包含 → GIN
--    全文搜索 → GIN 或 GiST
--    范围重叠 → GiST
--    地理位置查询 → GiST
--    超大表+有序数据 → BRIN

-- 2. 查看索引使用情况
SELECT 
    indexrelname AS 索引名,
    relname AS 表名,
    idx_scan AS 使用次数,
    idx_tup_read AS 读取元组数,
    pg_size_pretty(pg_relation_size(indexrelid)) AS 索引大小
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 3. 查找未使用的索引
SELECT 
    indexrelname AS 索引名,
    pg_size_pretty(pg_relation_size(indexrelid)) AS 索引大小
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;

-- 4. 使用 EXPLAIN ANALYZE 验证索引使用
EXPLAIN ANALYZE 
SELECT * FROM products WHERE attributes @> '{"brand": "Apple"}';
```

---

## 事务与隔离级别

### 事务基础

```sql
-- PostgreSQL 事务控制
BEGIN;                    -- 开始事务
BEGIN TRANSACTION;        -- 同上
START TRANSACTION;        -- 同上

COMMIT;                   -- 提交事务
ROLLBACK;                 -- 回滚事务

-- 保存点
BEGIN;
INSERT INTO orders ...;
SAVEPOINT order_saved;
UPDATE inventory ...;
-- 如果更新失败，可以回滚到保存点
ROLLBACK TO order_saved;
COMMIT;

-- 事务参数设置
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN TRANSACTION READ ONLY;  -- 只读事务
```

### 隔离级别详解

```sql
-- PostgreSQL 支持四种隔离级别（SQL 标准）

-- 1. READ UNCOMMITTED ≈ READ COMMITTED（PostgreSQL 特殊处理）
-- PostgreSQL 不支持真正的脏读，自动升级为 READ COMMITTED

-- 2. READ COMMITTED（默认）
-- 每条语句看到语句开始时已提交的数据
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT * FROM products WHERE id = 1;  -- 看到版本A
-- 另一事务提交了更新
SELECT * FROM products WHERE id = 1;  -- 现在看到版本B
COMMIT;

-- 3. REPEATABLE READ
-- 整个事务看到事务开始时的快照
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM products WHERE id = 1;  -- 看到版本A
-- 另一事务提交了更新
SELECT * FROM products WHERE id = 1;  -- 仍然看到版本A
COMMIT;

-- 4. SERIALIZABLE
-- 最严格的隔离，检测序列化冲突
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 如果与其他事务产生序列化冲突，提交时会报错
COMMIT;  -- 可能报错：could not serialize access

-- 隔离级别对比表
┌──────────────────────┬────────────────┬─────────────────┬─────────────────┐
│ 隔离级别             │ 脏读           │ 不可重复读      │ 幻读            │
├──────────────────────┼────────────────┼─────────────────┼─────────────────┤
│ READ UNCOMMITTED     │ 不可能（PG）   │ 可能            │ 可能            │
│ READ COMMITTED       │ 不可能         │ 可能            │ 可能            │
│ REPEATABLE READ      │ 不可能         │ 不可能          │ 不可能（PG）    │
│ SERIALIZABLE         │ 不可能         │ 不可能          │ 不可能          │
└──────────────────────┴────────────────┴─────────────────┴─────────────────┘
-- 注：PostgreSQL 的 REPEATABLE READ 也防止幻读（超出 SQL 标准）
```

### 锁机制

```sql
-- 查看当前锁
SELECT 
    locktype,
    relation::regclass AS 表,
    mode AS 锁模式,
    granted AS 是否获取,
    pid AS 进程ID
FROM pg_locks
WHERE relation IS NOT NULL;

-- 行级锁
BEGIN;
SELECT * FROM products WHERE id = 1 FOR UPDATE;      -- 排他行锁
SELECT * FROM products WHERE id = 2 FOR SHARE;        -- 共享行锁
SELECT * FROM products WHERE id = 3 FOR UPDATE NOWAIT;  -- 不等待，立即失败
SELECT * FROM products WHERE id = 4 FOR UPDATE SKIP LOCKED;  -- 跳过已锁定行
COMMIT;

-- 表级锁
BEGIN;
LOCK TABLE products IN SHARE MODE;         -- 共享锁，阻止写入
LOCK TABLE products IN EXCLUSIVE MODE;     -- 排他锁，阻止读写
LOCK TABLE products IN ACCESS EXCLUSIVE MODE;  -- 最高级排他锁
COMMIT;

-- 咨询锁（应用级锁，用于分布式协调）
SELECT pg_advisory_lock(12345);      -- 获取锁
SELECT pg_try_advisory_lock(12345);  -- 尝试获取（失败返回 false）
SELECT pg_advisory_unlock(12345);    -- 释放锁

-- 查看阻塞情况
SELECT 
    blocked.pid AS 被阻塞进程,
    blocking.pid AS 阻塞源进程,
    blocked.query AS 被阻塞查询
FROM pg_stat_activity blocked
JOIN pg_locks blocked_lock ON blocked.pid = blocked_lock.pid
JOIN pg_locks blocking_lock ON blocked_lock.locktype = blocking_lock.locktype
    AND blocked_lock.relation = blocking_lock.relation
    AND blocked_lock.pid != blocking_lock.pid
JOIN pg_stat_activity blocking ON blocking_lock.pid = blocking.pid
WHERE NOT blocked_lock.granted;

-- 终止阻塞会话
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...;
```

---

## 性能调优

### 配置优化

```ini
# postgresql.conf 核心参数

# 内存配置
shared_buffers = 4GB              # 共享缓冲区，推荐系统内存的 25%
work_mem = 64MB                   # 每个查询操作的内存
maintenance_work_mem = 1GB        # 维护操作内存（VACUUM、CREATE INDEX）
effective_cache_size = 12GB       # 规划器估计的系统缓存（系统内存的 50-75%）

# 并行查询（PostgreSQL 16+）
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# WAL 配置
wal_buffers = 64MB
checkpoint_completion_target = 0.9
max_wal_size = 2GB
min_wal_size = 1GB

# SSD 优化
random_page_cost = 1.1            # SSD 环境（HDD 用 4.0）
effective_io_concurrency = 200    # SSD 环境

# 自动清理
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05

# 日志（用于调试）
log_min_duration_statement = 1000  # 记录超过 1 秒的查询
log_lock_waits = on
log_temp_files = 0
```

### 查询计划分析

```sql
-- 基本查询计划
EXPLAIN SELECT * FROM products WHERE name = 'iPhone';

-- 详细执行分析
EXPLAIN ANALYZE SELECT * FROM products WHERE name = 'iPhone';

-- 更详细的输出（包含缓冲区信息）
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) 
SELECT * FROM products WHERE name = 'iPhone';

-- JSON 格式（便于程序解析）
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM products WHERE name = 'iPhone';

-- 关键指标解读
-- Seq Scan：顺序扫描（全表扫描），大表上可能有问题
-- Index Scan：索引扫描
-- Bitmap Index Scan：位图索引扫描
-- Hash Join：哈希连接
-- Nested Loop：嵌套循环连接
-- Merge Join：合并连接

-- 成本估算 vs 实际
-- rows=1000 (实际 rows=50000) → 统计信息不准，需要 ANALYZE
```

### 统计信息优化

```sql
-- 更新统计信息
ANALYZE products;

-- 增加统计精度（默认 100，最大 10000）
ALTER TABLE products ALTER COLUMN name SET STATISTICS 500;
ANALYZE products;

-- 多列统计信息（检测列之间的相关性）
CREATE STATISTICS s1 (dependencies, ndistinct) ON region, product_name FROM sales;
ANALYZE sales;

-- 表达式统计信息
CREATE STATISTICS s2 ON (attributes->>'brand') FROM products;
ANALYZE products;

-- 查看统计信息
SELECT 
    attname AS 列名,
    n_distinct AS 不同值数量,
    most_common_vals AS 最常见值,
    most_common_freqs AS 最常见值频率
FROM pg_stats
WHERE tablename = 'products';
```

### 并行查询优化

```sql
-- 查看并行配置
SHOW max_parallel_workers_per_gather;
SHOW max_parallel_workers;
SHOW parallel_tuple_cost;
SHOW parallel_setup_cost;

-- 强制并行（调试用）
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET max_parallel_workers_per_gather = 4;

-- 查看并行执行计划
EXPLAIN ANALYZE 
SELECT COUNT(*) FROM large_table WHERE status = 'active';

-- 典型输出：
-- Finalize Aggregate
--   ->  Gather
--         Workers Planned: 2
--         Workers Launched: 2
--           ->  Partial Aggregate
--                 ->  Parallel Seq Scan on large_table

-- 防止并行的因素
-- 1. 数据量太小
-- 2. 查询太简单
-- 3. 使用了 volatile 函数
-- 4. 隔离级别为 SERIALIZABLE
-- 5. 表上设置了 parallel_safe = false
```

---

## pg_stat_statements 监控

### 安装配置

```sql
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- postgresql.conf 配置
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.max = 10000

-- 需要重启 PostgreSQL
```

### 监控查询

```sql
-- 最耗时的 SQL（定位性能瓶颈）
SELECT 
    calls AS 执行次数,
    ROUND(total_exec_time::numeric / 1000, 2) AS 总时间_秒,
    ROUND(mean_exec_time::numeric, 2) AS 平均时间_毫秒,
    ROUND((100 * total_exec_time / SUM(total_exec_time) OVER())::numeric, 2) AS 占比,
    rows AS 返回行数,
    LEFT(query, 100) AS 查询预览
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- 执行次数最多的 SQL（优化热点）
SELECT 
    calls AS 执行次数,
    ROUND(mean_exec_time::numeric, 2) AS 平均时间_毫秒,
    rows AS 返回行数,
    query
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 10;

-- 缓存命中率低的 SQL（需要优化索引或增加内存）
SELECT 
    query,
    calls,
    shared_blks_hit AS 缓存命中,
    shared_blks_read AS 磁盘读取,
    ROUND(
        shared_blks_hit::numeric / 
        NULLIF(shared_blks_hit + shared_blks_read, 0) * 100, 
    2) AS 缓存命中率
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;

-- 重置统计
SELECT pg_stat_statements_reset();
```

---

## 连接池：PgBouncer

### 为什么需要连接池？

```
没有连接池的问题：
┌──────────────────────────────────────────────────────────┐
│ 每个客户端连接 = 一个 PostgreSQL 进程（约 10MB 内存）       │
│                                                          │
│ 1000 个客户端 = 10GB 内存！                               │
│                                                          │
│ 连接建立开销：认证、初始化会话、分配资源                    │
│ 频繁创建/销毁连接 → 性能下降                               │
└──────────────────────────────────────────────────────────┘

使用连接池：
┌──────────────────────────────────────────────────────────┐
│ 客户端连接 → PgBouncer → PostgreSQL                      │
│                                                          │
│ 1000 客户端连接 → 100 数据库连接                          │
│ 连接复用，资源占用大幅降低                                 │
└──────────────────────────────────────────────────────────┘
```

### PgBouncer 配置

```ini
; pgbouncer.ini

[databases]
myapp = host=127.0.0.1 port=5432 dbname=myapp

[pgbouncer]
; 监听配置
listen_addr = 0.0.0.0
listen_port = 6432

; 认证
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

; 连接池模式
; session - 会话级（兼容性最好）
; transaction - 事务级（最高效，推荐）
; statement - 语句级（有限制）
pool_mode = transaction

; 连接池大小
default_pool_size = 25      ; 每个数据库的连接数
max_client_conn = 100       ; 最大客户端连接数
min_pool_size = 5           ; 最小保持连接数

; 超时设置
server_connect_timeout = 15
server_idle_timeout = 600
server_lifetime = 3600
client_idle_timeout = 0
client_login_timeout = 60

; 日志
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_users = stats
```

### 连接池模式对比

| 模式 | 说明 | 适用场景 | 限制 |
|------|------|----------|------|
| **Session** | 会话级复用 | 使用会话变量、临时表 | 效率最低 |
| **Transaction** | 事务级复用 | 大多数 OLTP 应用 | 不支持 SET、PREPARE、WITH HOLD |
| **Statement** | 语句级复用 | 简单查询 | 不支持事务、预处理语句 |

### PgBouncer 管理

```sql
-- 连接到管理接口
psql -p 6432 pgbouncer -U admin

-- 查看客户端连接
SHOW CLIENTS;

-- 查看服务端连接
SHOW SERVERS;

-- 查看连接池状态
SHOW POOLS;

-- 输出示例：
--  database | cl_active | cl_waiting | sv_active | sv_idle | sv_used
-- ----------+-----------+------------+-----------+---------+---------
--  myapp    |        10 |          2 |         5 |       8 |       5

-- 查看统计
SHOW STATS;

-- 重载配置
RELOAD;

-- 暂停/恢复
PAUSE myapp;
RESUME myapp;
```

---

## pgvector：向量搜索扩展

### 为什么需要向量搜索？

```
AI 应用场景：
┌──────────────────────────────────────────────────────────┐
│ • 语义搜索：根据含义而非关键词搜索                         │
│ • 相似推荐：找到相似的物品/内容                           │
│ • 图像搜索：根据图像特征找相似图片                         │
│ • RAG（检索增强生成）：为大模型提供相关上下文              │
└──────────────────────────────────────────────────────────┘

传统数据库无法高效处理向量相似度计算
PostgreSQL + pgvector = 原生向量支持，无需额外系统
```

### pgvector 使用

```sql
-- 安装扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建带向量列的表
CREATE TABLE documents (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536)  -- OpenAI text-embedding-ada-002 维度
);

-- 创建向量索引
-- IVFFlat 索引：适合中等数据量
CREATE INDEX idx_documents_embedding_ivf ON documents 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- HNSW 索引：更快更精确，适合大数据量
CREATE INDEX idx_documents_embedding_hnsw ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- 插入向量数据（通常由应用生成）
INSERT INTO documents (content, embedding) VALUES
('PostgreSQL 是一个强大的开源数据库', '[0.1, 0.2, 0.3, ...]'::vector),
('向量搜索是 AI 的核心技术', '[0.15, 0.25, 0.35, ...]'::vector);

-- 相似度搜索
SELECT 
    content,
    1 - (embedding <=> '[0.12, 0.22, ...]'::vector) AS 相似度
FROM documents
ORDER BY embedding <=> '[0.12, 0.22, ...]'::vector
LIMIT 5;

-- 距离度量：
-- <=> 余弦距离（常用）
-- <-> L2 距离
-- <#> 内积（负值）
```

### 向量搜索实战

```python
# Python 示例：生成嵌入并存储
import psycopg2
from openai import OpenAI

# 初始化
client = OpenAI()
conn = psycopg2.connect("dbname=myapp user=postgres")

# 生成嵌入
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# 插入文档
def insert_document(content):
    embedding = get_embedding(content)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding)
        )
    conn.commit()

# 相似度搜索
def search_similar(query, limit=5):
    query_embedding = get_embedding(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit)
        )
        return cur.fetchall()

# 使用
insert_document("PostgreSQL 支持 JSONB 类型和向量搜索")
results = search_similar("数据库的 JSON 功能")
for content, similarity in results:
    print(f"{similarity:.3f}: {content}")
```

---

## PostgreSQL 16/17 新特性

### PostgreSQL 16（2023）

```sql
-- 1. 并行查询增强
-- 支持 FULL JOIN 和 RIGHT JOIN 的并行执行
SET max_parallel_workers_per_gather = 4;
EXPLAIN ANALYZE SELECT * FROM large_table t1 FULL JOIN large_table t2 ON t1.id = t2.id;

-- 2. COPY 性能提升（约 30%）
COPY large_table FROM '/data/export.csv' WITH (FORMAT csv, FREEZE);

-- 3. 逻辑复制改进
-- 支持并行应用和大批量数据传输

-- 4. JSON 函数增强
SELECT json_array_elements('[1, 2, 3]');
SELECT json_exists('{"a": 1}', '$.a');

-- 5. 统计信息改进
-- 更精确的多列统计
```

### PostgreSQL 17（2024）

```sql
-- 1. JSON_TABLE（SQL/JSON 标准）
SELECT *
FROM json_table(
    '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',
    '$.users[*]' COLUMNS (
        id INT PATH '$.id',
        name TEXT PATH '$.name'
    )
) AS t;
-- 结果：
--  id | name  
-- ----+-------
--   1 | Alice
--   2 | Bob

-- 2. MERGE 语句增强
MERGE INTO products AS target
USING updates AS source
ON target.id = source.id
WHEN MATCHED THEN
    UPDATE SET price = source.price, stock = target.stock + source.stock
WHEN NOT MATCHED THEN
    INSERT (id, name, price, stock)
    VALUES (source.id, source.name, source.price, source.stock);

-- 3. 新增 pg_stat_io 视图
SELECT * FROM pg_stat_io;

-- 4. 改进的 VACUUM
-- 减少系统目录膨胀

-- 5. 更快的 B-tree 索引构建
```

---

## 常见问题与最佳实践

### 常见陷阱

```sql
-- 1. SERIAL vs IDENTITY
-- ❌ 旧方式
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));

-- ✅ 新方式（PostgreSQL 10+）
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);
-- IDENTITY 的优势：标准 SQL、更好的权限控制、支持 RESTART

-- 2. COUNT(*) 性能
-- ❌ 全表扫描，大表很慢
SELECT COUNT(*) FROM large_table;

-- ✅ 使用估计值
SELECT reltuples::bigint FROM pg_class WHERE relname = 'large_table';

-- 3. 索引未使用的常见原因
-- 类型不匹配
WHERE user_id = '123'  -- user_id 是 INT，应该用 WHERE user_id = 123

-- 函数包装
WHERE LOWER(name) = 'alice'  -- 需要函数索引
CREATE INDEX idx_name_lower ON users (LOWER(name));

-- LIKE 以通配符开头
WHERE name LIKE '%son'  -- 索引不使用

-- 4. 大表更新
-- ❌ 一次性更新
UPDATE huge_table SET status = 'inactive' WHERE date < '2020-01-01';

-- ✅ 分批更新
DO $$
DECLARE
    rows INTEGER;
BEGIN
    LOOP
        UPDATE huge_table 
        SET status = 'inactive' 
        WHERE date < '2020-01-01' AND status != 'inactive'
          AND ctid IN (
              SELECT ctid FROM huge_table 
              WHERE date < '2020-01-01' AND status != 'inactive'
              LIMIT 10000
          );
        GET DIAGNOSTICS rows = ROW_COUNT;
        EXIT WHEN rows = 0;
        COMMIT;
    END LOOP;
END $$;
```

### 安全最佳实践

```sql
-- 1. 参数化查询（防止 SQL 注入）
-- ❌ 不安全
-- cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

-- ✅ 安全
-- cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))

-- 2. 行级安全策略
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY orders_policy ON orders
    USING (customer_id = current_user_id());

-- 3. 数据加密
CREATE EXTENSION IF NOT EXISTS pgcrypto;

SELECT encrypt('敏感数据', '密钥', 'aes');
SELECT decrypt(encrypted_data, '密钥', 'aes');

-- 4. 密码哈希
SELECT crypt('password123', gen_salt('bf'));
-- 验证
SELECT crypt('password123', stored_hash) = stored_hash;

-- 5. 连接加密
-- pg_hba.conf
hostssl all all 0.0.0.0/0 scram-sha-256
```

---

## 参考资料

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/current/index.html)
- [PostgreSQL 17 Release Notes](https://www.postgresql.org/about/news/postgresql-17-released-2936/)
- [pgvector 扩展](https://github.com/pgvector/pgvector)
- [PgBouncer 文档](https://www.pgbouncer.org/)
- [PostgreSQL Wiki](https://wiki.postgresql.org/)
- [PostGIS 文档](https://postgis.net/documentation/)