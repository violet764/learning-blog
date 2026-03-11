# MySQL 学习笔记

MySQL 是世界上最流行的开源关系型数据库管理系统。本文档基于 MySQL 8.0+ 和 8.4 LTS 版本，深入讲解核心概念的设计原理、实现机制和最佳实践。

📌 **版本建议**：MySQL 8.0 将于 2026 年 4 月 EOL（End of Life），建议新项目直接使用 MySQL 8.4 LTS 或 Innovation 版本。

## 为什么选择 MySQL？

### 背景：数据库选型的考量因素

在选择数据库时，我们需要考虑以下核心问题：

| 考量维度 | 核心问题 | MySQL 的答案 |
|---------|---------|-------------|
| **成本** | 是否开源免费？LTS 版本支持多久？ | GPL 协议，社区版完全免费；8.4 LTS 支持至 2032 年 |
| **性能** | 能否支撑高并发读写？ | 单机可达数万 QPS，配合主从/分库分表可达百万级 |
| **可靠性** | 数据会丢失吗？事务支持如何？ | InnoDB 提供完整 ACID 事务，崩溃恢复能力强 |
| **生态** | 工具链是否完善？人才储备如何？ | 运维工具丰富，DBA 人才充足 |
| **学习曲线** | 上手难度如何？ | SQL 标准语法，文档完善，社区活跃 |

### MySQL 架构概览

理解架构有助于排查问题和性能优化：

```
┌─────────────────────────────────────────────────────────────┐
│                    连接层 (Connection Layer)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 连接池管理   │  │ 认证授权    │  │ 线程管理    │         │
│  │ 连接复用    │  │ 密码验证    │  │ 一连接一线程 │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    SQL 层 (SQL Layer)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ 解析器   │→│ 预处理器 │→│ 优化器   │→│ 执行器   │       │
│  │ 词法分析 │  │ 语义检查 │  │ 选择索引 │  │ 调用引擎 │       │
│  │ 语法分析 │  │ 权限检查 │  │ 生成计划 │  │ 返回结果 │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│                        ↑↓                                   │
│              ┌─────────────────┐                           │
│              │ 查询缓存(8.0删除)│                           │
│              └─────────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│                 存储引擎层 (Storage Engine)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   InnoDB    │  │   MyISAM    │  │   Memory    │  ...    │
│  │  事务型引擎  │  │ 非事务引擎  │  │  内存引擎   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                     文件系统层                               │
│   数据文件(.ibd)  日志文件(redo/undo/binlog)  配置文件       │
└─────────────────────────────────────────────────────────────┘
```

**为什么要采用分层架构？**

- **解耦合**：每层专注自己的职责，SQL 层与存储引擎分离，支持多种引擎
- **可扩展**：优化器可以独立演进，不影响存储引擎
- **可维护**：问题定位更清晰，连接问题、SQL 问题、存储问题分开排查

---

## MySQL 8.0+ 新特性

### 背景：为什么需要这些新特性？

MySQL 8.0 之前存在以下痛点：

| 痛点 | 旧版本问题 | 8.0 解决方案 |
|-----|-----------|-------------|
| 复杂分析查询困难 | 需要子查询或应用层处理 | 窗口函数 |
| 递归查询不支持 | 需要存储过程或多条 SQL | 递归 CTE |
| JSON 支持不完善 | 修改 JSON 困难 | JSON 函数增强、多值索引 |
| 索引删除风险 | 删除后无法回滚 | 隐藏索引 |

### 窗口函数（Window Functions）

**问题背景**：传统聚合函数会将多行聚合成一行，但很多时候我们需要在保留原始行的同时进行聚合计算。

例如：计算每个员工在其部门内的薪资排名，传统方式需要复杂的子查询：

```sql
-- 传统方式：复杂且性能差
SELECT 
    e1.employee_id,
    e1.name,
    e1.department_id,
    e1.salary,
    (SELECT COUNT(*) + 1 FROM employees e2 
     WHERE e2.department_id = e1.department_id 
     AND e2.salary > e1.salary) AS rank_in_dept
FROM employees e1;

-- 窗口函数：简洁且高效
SELECT 
    employee_id,
    name,
    department_id,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank_in_dept
FROM employees;
```

**核心原理**：窗口函数在查询结果集上定义一个"窗口"，对窗口内的行进行计算，但不会减少结果行数。

```
原始数据                        窗口函数处理后
┌─────┬──────┬───────┬───────┐           ┌─────┬──────┬───────┬───────┬──────────┐
│ id  │ name │ dept  │salary │           │ id  │ name │ dept  │salary │ rank     │
├─────┼──────┼───────┼───────┤           ├─────┼──────┼───────┼───────┼──────────┤
│ 1   │ 张三 │ D001  │ 10000 │           │ 1   │ 张三 │ D001  │ 10000 │    2     │
│ 2   │ 李四 │ D001  │ 15000 │    →      │ 2   │ 李四 │ D001  │ 15000 │    1     │
│ 3   │ 王五 │ D001  │ 8000  │           │ 3   │ 王五 │ D001  │ 8000  │    3     │
│ 4   │ 赵六 │ D002  │ 12000 │           │ 4   │ 赵六 │ D002  │ 12000 │    1     │
└─────┴──────┴───────┴───────┘           └─────┴──────┴───────┴───────┴──────────┘
                                        行数不变，增加了排名信息
```

**常用窗口函数分类**：

| 类别 | 函数 | 用途 |
|-----|------|-----|
| 排名函数 | ROW_NUMBER() | 连续排名（1,2,3,4） |
| | RANK() | 并列跳过（1,1,3,4） |
| | DENSE_RANK() | 并列不跳过（1,1,2,3） |
| | NTILE(n) | 分成 n 组 |
| 聚合函数 | SUM()/AVG()/COUNT() | 累计/移动计算 |
| | MAX()/MIN() | 窗口内极值 |
| 偏移函数 | LAG(col, n) | 前 n 行的值 |
| | LEAD(col, n) | 后 n 行的值 |
| | FIRST_VALUE()/LAST_VALUE() | 首尾值 |

```sql
-- ===== 实践示例 =====

-- 示例1：计算每个部门的薪资排名
SELECT 
    employee_id,
    name,
    department_id,
    salary,
    -- ROW_NUMBER: 连续排名，即使薪资相同也按顺序排名
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS row_num,
    -- RANK: 并列排名，会跳过名次（如两个第1名后是第3名）
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank_val,
    -- DENSE_RANK: 并列排名，不跳过名次（如两个第1名后是第2名）
    DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dense_rank_val
FROM employees;

-- 示例2：移动平均（7日滑动窗口）
-- 用于平滑数据波动，观察趋势
SELECT 
    date,
    sales,
    -- 计算当日及前6天共7天的平均值
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7days,
    -- 累计销售额（从第一天到当天）
    SUM(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_sum
FROM daily_sales;

-- 示例3：环比增长计算
SELECT 
    date,
    sales,
    -- 获取前一天的销售数据
    LAG(sales, 1) OVER (ORDER BY date) AS prev_day_sales,
    -- 计算环比增长
    sales - LAG(sales, 1) OVER (ORDER BY date) AS daily_change,
    -- 计算环比增长率
    ROUND((sales - LAG(sales, 1) OVER (ORDER BY date)) / 
          LAG(sales, 1) OVER (ORDER BY date) * 100, 2) AS growth_rate
FROM daily_sales;

-- 示例4：客户分层（四分位数）
SELECT 
    customer_id,
    total_spent,
    -- 将客户按消费额分成4组，1为最高消费组
    NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile,
    CASE NTILE(4) OVER (ORDER BY total_spent DESC)
        WHEN 1 THEN '高价值客户'
        WHEN 2 THEN '中高价值客户'
        WHEN 3 THEN '中等价值客户'
        ELSE '普通客户'
    END AS customer_tier
FROM customer_spending;
```

### 公用表表达式（CTE）

**问题背景**：复杂查询往往需要多层嵌套子查询，导致：
1. **可读性差**：嵌套层次深，难以理解
2. **维护困难**：修改逻辑需要找到对应的子查询
3. **重复代码**：相同的子查询可能在多处使用

```sql
-- 传统嵌套子查询：难以阅读和维护
SELECT *
FROM (
    SELECT user_id, SUM(amount) AS total
    FROM orders
    WHERE created_at > '2024-01-01'
    GROUP BY user_id
) AS user_totals
WHERE total > (
    SELECT AVG(total) FROM (
        SELECT user_id, SUM(amount) AS total
        FROM orders
        WHERE created_at > '2024-01-01'
        GROUP BY user_id
    ) AS t
);

-- 使用 CTE：逻辑清晰，代码复用
WITH user_totals AS (
    -- 第一步：计算每个用户的消费总额
    SELECT user_id, SUM(amount) AS total
    FROM orders
    WHERE created_at > '2024-01-01'
    GROUP BY user_id
),
avg_total AS (
    -- 第二步：计算平均消费额
    SELECT AVG(total) AS avg_value FROM user_totals
)
-- 第三步：筛选高消费用户
SELECT u.*
FROM user_totals u
CROSS JOIN avg_total a
WHERE u.total > a.avg_value;
```

**递归 CTE**：处理层级数据（组织架构、菜单树、家族关系等）

```sql
-- 实践：组织架构层级查询
-- 表结构：employees(id, name, manager_id)
-- manager_id 为 NULL 表示顶级管理者

WITH RECURSIVE org_hierarchy AS (
    -- 锚点成员：选择顶级管理者（递归起点）
    SELECT 
        id,
        name,
        manager_id,
        1 AS level,  -- 层级深度，从1开始
        CAST(name AS CHAR(1000)) AS path  -- 路径字符串，用于显示层级关系
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- 递归成员：连接下级员工
    -- 每次递归处理一层，level 递增
    SELECT 
        e.id,
        e.name,
        e.manager_id,
        h.level + 1,  -- 层级加1
        CONCAT(h.path, ' -> ', e.name)  -- 构建路径
    FROM employees e
    INNER JOIN org_hierarchy h ON e.manager_id = h.id  -- 连接条件：员工的管理者
    WHERE h.level < 10  -- 防止无限递归（安全限制）
)
SELECT 
    level,
    CONCAT(REPEAT('    ', level - 1), name) AS org_tree,  -- 缩进显示层级
    path
FROM org_hierarchy
ORDER BY path;

-- 输出示例：
-- level | org_tree        | path
-- ------|-----------------|------------------
--   1   | CEO             | CEO
--   2   |     CTO         | CEO -> CTO
--   3   |         Tech Lead| CEO -> CTO -> Tech Lead
--   2   |     CFO         | CEO -> CFO
```

### JSON 增强

**问题背景**：现代应用中，数据结构越来越灵活：
- 用户画像可能有几十个属性，但每个用户只有部分属性
- 商品规格因类目不同而差异巨大
- 传统关系型设计需要大量字段或关联表

**解决方案**：JSON 类型提供灵活的半结构化数据存储。

```sql
-- 创建带 JSON 列的表
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    attributes JSON,  -- 商品属性：颜色、尺寸、重量等
    tags JSON,        -- 标签数组
    INDEX idx_tags ((CAST(tags AS CHAR(100) ARRAY)))  -- MySQL 8.0.17+ 多值索引
);

-- 插入 JSON 数据
INSERT INTO products VALUES (
    1,
    'iPhone 15',
    '{"color": "黑色", "storage": 256, "price": 5999, "specs": {"weight": 171}}',
    '["手机", "苹果", "5G", "旗舰"]'
);

-- ===== JSON 查询操作 =====

-- 方式1：JSON_EXTRACT 函数
SELECT JSON_EXTRACT(attributes, '$.color');  -- 返回 "黑色"（带引号）

-- 方式2：-> 运算符（返回 JSON 格式）
SELECT attributes->'$.color';  -- 返回 "黑色"

-- 方式3：->> 运算符（返回文本，推荐）
SELECT attributes->>'$.color';  -- 返回 黑色（不带引号）

-- ===== JSON 修改操作 =====

-- JSON_SET：设置或更新值
UPDATE products 
SET attributes = JSON_SET(attributes, '$.weight', 180)
WHERE id = 1;

-- JSON_REMOVE：删除键
UPDATE products 
SET attributes = JSON_REMOVE(attributes, '$.storage')
WHERE id = 1;

-- ===== JSON 数组操作 =====

-- 检查数组是否包含某元素（需要多值索引）
SELECT * FROM products WHERE '苹果' MEMBER OF(tags);

-- JSON_TABLE：将 JSON 数组展开为行（MySQL 8.0+）
SELECT 
    p.id,
    p.name,
    jt.tag
FROM products p,
JSON_TABLE(
    p.tags,
    '$[*]' COLUMNS (tag VARCHAR(50) PATH '$')
) AS jt;

-- 输出：
-- id | name     | tag
-- ---|----------|------
-- 1  | iPhone 15| 手机
-- 1  | iPhone 15| 苹果
-- 1  | iPhone 15| 5G
-- 1  | iPhone 15| 旗舰
```

### 隐藏索引（Invisible Index）

**问题背景**：删除索引有风险：
- 可能导致某些查询性能骤降
- 删除后重建索引成本很高（大表可能需要数小时）
- 无法验证删除影响后再决定

**解决方案**：隐藏索引不会被优化器使用，但仍然维护更新。

```sql
-- 创建隐藏索引
CREATE INDEX idx_test ON users(email) INVISIBLE;

-- 修改现有索引为隐藏
ALTER TABLE users ALTER INDEX idx_email INVISIBLE;

-- 场景：灰度删除索引
-- 步骤1：将索引设为隐藏
ALTER TABLE users ALTER INDEX idx_email INVISIBLE;

-- 步骤2：观察一段时间，检查慢查询日志
-- 如果没有性能问题，继续步骤3

-- 步骤3：确认删除
DROP INDEX idx_email ON users;

-- 如果发现问题，立即恢复
ALTER TABLE users ALTER INDEX idx_email VISIBLE;
```

---

## 数据类型选择

### 背景：为什么数据类型选择很重要？

| 问题 | 影响 |
|-----|------|
| 选择过大 | 浪费存储空间，降低缓存效率，影响查询性能 |
| 选择过小 | 数据溢出，精度丢失，业务异常 |
| 类型不当 | 隐式转换导致索引失效，查询变慢 |

### 数值类型选择原则

```
                    数值范围对比
    0        127      32767    8.4M      2.1B        很大
    |----------|---------|--------|----------|----------|
    TINYINT   SMALLINT  MEDIUMINT   INT      BIGINT
       1B        2B        3B        4B        8B
```

**选择原则**：
1. **够用即可**：预估数据范围，选择最小能满足的类型
2. **考虑无符号**：纯非负数使用 UNSIGNED 可扩大正数范围
3. **金额用 DECIMAL**：避免浮点精度问题

```sql
CREATE TABLE example_numbers (
    -- 状态码：TINYINT 足够（0-255）
    status TINYINT UNSIGNED NOT NULL DEFAULT 0,
    
    -- 年龄：TINYINT UNSIGNED（0-255）
    age TINYINT UNSIGNED,
    
    -- 数量/计数：INT 足够（约 21 亿）
    quantity INT UNSIGNED NOT NULL DEFAULT 0,
    
    -- 主键：考虑数据量，选择 INT 或 BIGINT
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- 金额：DECIMAL 保证精度
    -- DECIMAL(19,4) 表示：总共19位，小数4位，即最大 999,999,999,999,999.9999
    price DECIMAL(19,4) NOT NULL,
    
    -- 百分比：DECIMAL(5,2)，范围 0.00 ~ 999.99%
    discount_rate DECIMAL(5,2) DEFAULT 0.00
);

-- ⚠️ 金额计算错误示例
-- FLOAT/DOUBLE 存在精度问题
SELECT 0.1 + 0.2;  -- 结果可能是 0.30000000000000004

-- ✅ 使用 DECIMAL 保证精度
SELECT CAST(0.1 AS DECIMAL(10,2)) + CAST(0.2 AS DECIMAL(10,2));  -- 结果 0.30
```

### 字符串类型选择原则

```
存储方式对比：

CHAR(n)：定长存储，不足补空格
┌────────────────────────────────────────┐
│ 'abc' → 'abc                         ' │  存储时补空格
│ 读取时自动去除尾部空格                  │
└────────────────────────────────────────┘

VARCHAR(n)：变长存储，实际长度+1-2字节前缀
┌────────────────────────────────────────┐
│ 'abc' → [长度3] + 'abc'                │  只存储实际内容
│ 最大长度 65535 字节                     │
└────────────────────────────────────────┘
```

**选择原则**：
1. **固定长度用 CHAR**：手机号、身份证号、MD5 值等
2. **变长用 VARCHAR**：姓名、邮箱、地址等
3. **长文本用 TEXT**：文章内容、评论等
4. **统一使用 utf8mb4**：支持完整 Unicode（包括 Emoji）

```sql
CREATE TABLE example_strings (
    id INT PRIMARY KEY,
    
    -- 固定长度：手机号（11位）、身份证号（18位）
    phone CHAR(11) NOT NULL,
    id_card CHAR(18) NOT NULL,
    
    -- 变长字符串
    name VARCHAR(50) NOT NULL,        -- 姓名，预留足够长度
    email VARCHAR(100),               -- 邮箱地址
    
    -- 长文本
    summary TEXT,                     -- 摘要
    content MEDIUMTEXT,               -- 文章内容（最大 16MB）
    
    -- 枚举类型：固定选项，节省空间
    -- ENUM 内部存储为整数（1, 2, 3...），占用 1-2 字节
    status ENUM('draft', 'published', 'archived') DEFAULT 'draft'
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ⚠️ utf8 vs utf8mb4
-- MySQL 的 "utf8" 实际是 utf8mb3，只支持 3 字节字符
-- utf8mb4 支持完整 Unicode，包括 Emoji
-- 推荐：统一使用 utf8mb4
```

### 时间类型选择原则

```
时间类型对比：

DATETIME                    TIMESTAMP
    │                          │
    ├─ 范围：1000-9999年        ├─ 范围：1970-2038年（注意！）
    ├─ 存储：8 字节             ├─ 存储：4 字节
    ├─ 时区：无自动转换         ├─ 时区：自动转换为当前时区
    └─ 业务时间推荐使用          └─ Unix 时间戳场景
```

**选择原则**：
1. **业务时间用 DATETIME**：创建时间、更新时间等
2. **TIMESTAMP 注意 2038 问题**：适合短期数据
3. **仅日期用 DATE**：生日、节假日等
4. **统一时区处理**：存储 UTC，展示时转换

```sql
CREATE TABLE example_time (
    id INT PRIMARY KEY,
    
    -- 业务时间：使用 DATETIME
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- 仅日期：生日、截止日期等
    birth_date DATE,
    deadline DATE,
    
    -- 时间段：开始时间、时长
    start_time TIME,
    duration TIME  -- 可表示时长，如 '12:30:00' 表示12小时30分
);

-- 时区处理最佳实践
-- 1. 数据库使用 UTC 时区
-- 2. 应用层存储 UTC 时间
-- 3. 展示时转换为用户时区

SELECT 
    created_at AS utc_time,
    CONVERT_TZ(created_at, '+00:00', '+08:00') AS beijing_time
FROM orders;
```

---

## 索引设计与优化

### 为什么需要索引？

**问题背景**：没有索引时，查找数据需要全表扫描。

```
假设表有 1000 万行数据：

无索引查找：
┌─────────────────────────────────────────────────────┐
│  逐行扫描，比较每一行 → O(n) 复杂度                  │
│  平均需要扫描 500 万行                               │
│  假设每行 1KB，需要读取约 5GB 数据                   │
│  磁盘 I/O 是最大瓶颈，可能需要数秒甚至数分钟          │
└─────────────────────────────────────────────────────┘

有索引查找：
┌─────────────────────────────────────────────────────┐
│  B+树索引查找 → O(log n) 复杂度                      │
│  log₂(10000000) ≈ 24                               │
│  只需要访问约 24 个节点                              │
│  每个节点约 16KB，总共约 384KB                       │
│  毫秒级响应                                         │
└─────────────────────────────────────────────────────┘
```

### 为什么 MySQL 使用 B+树而不是 B树？

这是面试高频问题，核心原因如下：

```
B树结构：
                    ┌─────────────────┐
                    │  10 │ 数据10     │  ← 内部节点也存储数据
                    └────────┬────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                          ▼
┌─────────────────┐                      ┌─────────────────┐
│  5 │ 数据5      │                      │  20 │ 数据20    │
│  8 │ 数据8      │                      │  25 │ 数据25    │
└─────────────────┘                      └─────────────────┘

B+树结构：
                    ┌─────────────────┐
                    │  10 │ 20 │ 30   │  ← 内部节点只存索引键
                    └────────┬────────┘           不存储数据，可存更多键
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  5 │ 数据5      │  │  15 │ 数据15    │  │  25 │ 数据25    │
│  8 │ 数据8      │  │  20 │ 数据20    │  │  30 │ 数据30    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              ↑
                    叶子节点通过指针连接
                    便于范围查询
```

| 对比项 | B树 | B+树 | 为什么 B+树更适合数据库？ |
|-------|-----|------|-------------------------|
| 内部节点 | 存储键和数据 | 只存储键 | B+树内部节点更小，一页可存更多键，树更矮，I/O 更少 |
| 叶子节点 | 存储部分数据 | 存储全部数据 | B+树查询稳定，都要到叶子节点 |
| 范围查询 | 需要中序遍历 | 顺序遍历叶子链表 | B+树范围查询更高效 |
| 单点查询 | 可能更快（内部节点命中） | 必须到叶子 | 数据库查询范围和等值都有，B+树综合更优 |

### 聚簇索引 vs 非聚簇索引

**核心区别**：数据和索引是否存储在一起。

```
聚簇索引（主键索引）：
┌─────────────────────────────────────────────────────┐
│              B+树叶子节点直接存储完整行数据          │
│                                                     │
│  一张表只能有一个聚簇索引（InnoDB 用主键）           │
│  主键查询不需要回表，效率最高                        │
└─────────────────────────────────────────────────────┘

非聚簇索引（二级索引）：
┌─────────────────────────────────────────────────────┐
│              B+树叶子节点存储：索引列值 + 主键值     │
│                                                     │
│  查找流程：                                         │
│  1. 在二级索引树找到对应叶子节点                     │
│  2. 获取主键值                                      │
│  3. 到聚簇索引树查找完整行数据（回表）               │
└─────────────────────────────────────────────────────┘

示例：
┌────────────────────────────────────────────────────┐
│ 表结构：users(id, name, email, age)                │
│ 主键：id                                           │
│ 二级索引：idx_email(email)                         │
│                                                    │
│ 查询：SELECT * FROM users WHERE email = 'a@b.com'  │
│                                                    │
│ 执行过程：                                          │
│ 1. 在 idx_email 树找到 email = 'a@b.com' 的叶子节点│
│ 2. 获取对应的主键值，如 id = 100                   │
│ 3. 在聚簇索引树找到 id = 100 的叶子节点            │
│ 4. 返回完整行数据                                  │
│                                                    │
│ 这就是"回表"过程，比主键查询多一次 I/O             │
└────────────────────────────────────────────────────┘
```

### 覆盖索引

**问题背景**：回表会增加 I/O 操作，如何避免？

**解决方案**：覆盖索引让查询所需的所有列都在索引中，无需回表。

```sql
-- 表结构
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    status VARCHAR(20),
    total_amount DECIMAL(10,2),
    created_at DATETIME,
    INDEX idx_user_status (user_id, status, total_amount)
);

-- ✅ 使用覆盖索引，无需回表
-- 查询的 user_id, status, total_amount 都在索引中
SELECT user_id, status, total_amount
FROM orders
WHERE user_id = 100;

-- 执行计划中 Extra 列显示 "Using index"

-- ❌ 需要回表
-- 查询 created_at 不在索引中
SELECT user_id, status, created_at
FROM orders
WHERE user_id = 100;

-- 执行计划中 Extra 列不显示 "Using index"
```

**如何判断是否使用覆盖索引**：

```sql
EXPLAIN SELECT user_id, status, total_amount FROM orders WHERE user_id = 100;

-- 查看 Extra 列
-- "Using index" → 使用覆盖索引
-- 无 "Using index" → 需要回表
```

### 最左前缀原则

**问题背景**：复合索引 (a, b, c)，哪些查询能用上索引？

**核心原理**：索引按照定义的列顺序构建，类似字符串的字典序排列。

```
复合索引 (status, created_at, user_id) 的存储顺序：

┌─────────────────────────────────────────────────────┐
│ 先按 status 排序，status 相同的按 created_at 排序   │
│ status 和 created_at 都相同的按 user_id 排序        │
│                                                     │
│ 类比：电话簿按"姓-名-电话"排序                      │
│ 查找"姓"可以快速定位                                │
│ 查找"姓+名"也可以快速定位                           │
│ 但只查找"名"无法利用排序优势                        │
└─────────────────────────────────────────────────────┘

索引结构示意：
status   created_at   user_id
────────────────────────────────
active   2024-01-01   100
active   2024-01-01   101      ← status 相同，按 created_at 排
active   2024-01-02   103
inactive 2024-01-01   200      ← status 变化，重新排序
pending  2024-01-01   300
```

**查询能否使用索引的判断**：

```sql
-- 复合索引：idx_status_created_user (status, created_at, user_id)

-- ✅ 可以使用索引（从最左开始，连续）
WHERE status = 'active'
WHERE status = 'active' AND created_at > '2024-01-01'
WHERE status = 'active' AND created_at > '2024-01-01' AND user_id = 100

-- ❌ 无法使用索引（跳过了 status）
WHERE created_at > '2024-01-01'
WHERE user_id = 100

-- ⚠️ 只能部分使用索引（只用到 status）
-- 跳过了 created_at，user_id 条件无法利用索引
WHERE status = 'active' AND user_id = 100

-- ⚠️ 范围查询会阻断后续索引使用
-- 用到 status 和 created_at，user_id 无法利用索引
WHERE status = 'active' AND created_at > '2024-01-01' AND user_id = 100
```

### 索引失效的常见情况

```sql
-- 1. 对索引列使用函数
-- ❌ 索引失效
WHERE YEAR(created_at) = 2024

-- ✅ 优化：改为范围查询
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'

-- 2. 隐式类型转换
-- ❌ 索引失效（phone 是 VARCHAR 类型，传入数字）
WHERE phone = 13800138000

-- ✅ 优化：使用字符串
WHERE phone = '13800138000'

-- 3. 前置模糊查询
-- ❌ 索引失效
WHERE name LIKE '%张%'

-- ✅ 后置模糊可以使用索引
WHERE name LIKE '张%'

-- 4. 对索引列进行计算
-- ❌ 索引失效
WHERE age + 1 = 25

-- ✅ 优化
WHERE age = 24

-- 5. 使用 NOT、!=、<>
-- ❌ 可能索引失效
WHERE status != 'active'

-- ✅ 优化：改写为 IN
WHERE status IN ('inactive', 'banned', 'pending')

-- 6. OR 连接非索引列
-- ❌ 如果 condition_a 列无索引，整个条件无法使用索引
WHERE indexed_col = 'value' OR condition_a = 'x'

-- ✅ 优化：使用 UNION
SELECT * FROM t WHERE indexed_col = 'value'
UNION
SELECT * FROM t WHERE condition_a = 'x'
```

### 索引下推（ICP）

**问题背景**：复合索引 (name, age)，查询 `WHERE name LIKE '张%' AND age > 20`

```
无索引下推（MySQL 5.6 之前）：
┌─────────────────────────────────────────────────────┐
│ 存储引擎层：                                         │
│   - 根据索引找到所有 name LIKE '张%' 的记录          │
│   - 返回这些记录的完整数据给 Server 层               │
│                                                     │
│ Server 层：                                         │
│   - 对返回的记录过滤 age > 20                       │
│                                                     │
│ 问题：如果 name LIKE '张%' 有 10000 条，              │
│       但 age > 20 只有 100 条，                      │
│       仍然要传输 10000 条数据                        │
└─────────────────────────────────────────────────────┘

有索引下推（MySQL 5.6+）：
┌─────────────────────────────────────────────────────┐
│ 存储引擎层：                                         │
│   - 在索引遍历时就应用 age > 20 条件                 │
│   - 只返回满足两个条件的记录                         │
│                                                     │
│ Server 层：                                         │
│   - 直接返回结果，无需再过滤                         │
│                                                     │
│ 优势：大大减少回表次数和数据传输量                    │
└─────────────────────────────────────────────────────┘
```

```sql
-- 查看是否使用索引下推
EXPLAIN SELECT * FROM users WHERE name LIKE '张%' AND age > 20;

-- Extra 列显示 "Using index condition" 表示使用了 ICP
```

### 索引设计最佳实践

```sql
-- ===== 1. 选择性原则 =====
-- 选择性 = DISTINCT(列) / COUNT(*)，越接近 1 越好
-- 高选择性列适合建索引

SELECT 
    COUNT(DISTINCT status) / COUNT(*) AS status_selectivity,  -- 低（如 0.01）
    COUNT(DISTINCT email) / COUNT(*) AS email_selectivity     -- 高（接近 1）
FROM users;

-- ===== 2. 复合索引列顺序原则 =====
-- 列顺序建议：等值查询列 → 范围查询列 → 排序列

-- 场景：经常按 status 过滤，按 created_at 排序
-- ✅ 推荐
CREATE INDEX idx_status_created ON orders(status, created_at);

-- 查询可以利用索引进行过滤和排序
SELECT * FROM orders WHERE status = 'paid' ORDER BY created_at DESC LIMIT 10;

-- ===== 3. 前缀索引 =====
-- 长字符串列，可以用前缀索引节省空间

-- 查看前缀的选择性，确定合适的长度
SELECT 
    COUNT(DISTINCT LEFT(email, 5)) / COUNT(*) AS prefix_5,
    COUNT(DISTINCT LEFT(email, 10)) / COUNT(*) AS prefix_10,
    COUNT(DISTINCT email) / COUNT(*) AS full_selectivity
FROM users;

-- 创建前缀索引
CREATE INDEX idx_email_prefix ON users(email(10));

-- ⚠️ 前缀索引无法用于覆盖索引和 ORDER BY

-- ===== 4. 避免冗余索引 =====
-- 如果已有索引 (a, b)，再建 (a) 就是冗余的

-- 查看表的索引
SHOW INDEX FROM users;

-- ===== 5. 定期维护索引 =====
-- 更新索引统计信息
ANALYZE TABLE users;

-- 重建表（优化碎片）
OPTIMIZE TABLE users;
```

---

## 事务与隔离级别

### 为什么需要事务？

**问题背景**：银行转账场景

```
转账场景：A 账户转 100 元给 B 账户

步骤1：A 账户余额减 100
步骤2：B 账户余额加 100

如果没有事务保护：
┌─────────────────────────────────────────────────────┐
│ 情况1：步骤1 成功，步骤2 失败                        │
│        → A 少了 100，B 没收到，100 元凭空消失        │
│                                                     │
│ 情况2：步骤1 成功后系统崩溃                          │
│        → 重启后数据不一致                           │
│                                                     │
│ 情况3：多个转账同时进行                              │
│        → 可能读到中间状态                           │
└─────────────────────────────────────────────────────┘

有事务保护：
┌─────────────────────────────────────────────────────┐
│ BEGIN;                                              │
│ UPDATE accounts SET balance = balance - 100         │
│   WHERE id = 'A';                                   │
│ UPDATE accounts SET balance = balance + 100         │
│   WHERE id = 'B';                                   │
│ COMMIT;                                             │
│                                                     │
│ 要么全部成功，要么全部失败，保证数据一致性            │
└─────────────────────────────────────────────────────┘
```

### ACID 特性详解

| 特性 | 含义 | 问题 | 实现机制 |
|-----|------|-----|---------|
| **A**tomicity（原子性） | 事务是不可分割的单位，要么全部成功，要么全部失败 | 操作执行一半失败，数据处于中间状态 | undo log（回滚日志） |
| **C**onsistency（一致性） | 事务使数据库从一个一致状态变到另一个一致状态 | 数据违反业务规则（如余额为负） | 约束、触发器、应用层校验 |
| **I**solation（隔离性） | 多个事务并发执行时互不干扰 | 读取到其他事务的中间状态 | 锁 + MVCC |
| **D**urability（持久性） | 事务完成后数据永久保存 | 系统崩溃导致已提交数据丢失 | redo log（重做日志） |

### 隔离级别详解

**问题背景**：并发事务可能产生的问题

```
问题1：脏读（Dirty Read）
┌─────────────────────────────────────────────────────┐
│ 事务A                          事务B                │
│ BEGIN;                                              │
│                                BEGIN;               │
│ UPDATE users SET age=20        ─────────→ age=20    │
│   WHERE id=1;                  (读取到未提交数据)   │
│ ROLLBACK;  ← 回滚了！          ─────────→ age=20?   │
│                                (但B已经用了这个值)  │
│                                                     │
│ 问题：读取了未提交的数据，可能是不一致的              │
└─────────────────────────────────────────────────────┘

问题2：不可重复读（Non-repeatable Read）
┌─────────────────────────────────────────────────────┐
│ 事务A                          事务B                │
│ BEGIN;                                              │
│ SELECT age FROM users          ─────────→ age=18    │
│                                BEGIN;               │
│                                UPDATE users         │
│                                  SET age=20         │
│                                COMMIT;              │
│ SELECT age FROM users          ─────────→ age=20    │
│                                                     │
│ 问题：同一事务内两次读取结果不同                      │
└─────────────────────────────────────────────────────┘

问题3：幻读（Phantom Read）
┌─────────────────────────────────────────────────────┐
│ 事务A                          事务B                │
│ BEGIN;                                              │
│ SELECT * FROM users            ─────────→ 10行      │
│   WHERE age > 18;                                   │
│                                BEGIN;               │
│                                INSERT INTO users    │
│                                  VALUES(..., age=20);│
│                                COMMIT;              │
│ SELECT * FROM users            ─────────→ 11行      │
│   WHERE age > 18;             (多了一行"幻影")      │
│                                                     │
│ 问题：同一条件查询，结果集行数变化                    │
└─────────────────────────────────────────────────────┘
```

**四种隔离级别**：

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 说明 |
|---------|------|-----------|------|------|
| READ UNCOMMITTED | ✓ | ✓ | ✓ | 基本不使用 |
| READ COMMITTED (RC) | ✗ | ✓ | ✓ | 每次读取都生成新快照 |
| REPEATABLE READ (RR) | ✗ | ✗ | ✓* | MySQL 默认，MVCC 解决大部分幻读 |
| SERIALIZABLE | ✗ | ✗ | ✗ | 完全串行，性能最差 |

*MySQL InnoDB 通过 MVCC + Next-Key Lock 在 RR 级别基本解决了幻读问题。

```sql
-- 查看当前隔离级别
SELECT @@transaction_isolation;

-- 设置隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET GLOBAL TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

### MVCC 实现原理

**核心思想**：为每行数据维护多个版本，读操作读取快照，写操作创建新版本，实现读写不冲突。

**为什么 MVCC 能实现读不加锁？**

```
传统方式：读操作加共享锁，写操作加排他锁
┌─────────────────────────────────────────────────────┐
│ 事务A（读）加共享锁                                  │
│ 事务B（写）想加排他锁 → 被阻塞                       │
│                                                     │
│ 问题：读写冲突，并发性能差                           │
└─────────────────────────────────────────────────────┘

MVCC 方式：读操作读取历史快照
┌─────────────────────────────────────────────────────┐
│ 事务A（读）读取数据的某个历史版本（快照）            │
│ 事务B（写）修改数据，创建新版本                      │
│                                                     │
│ 优势：读写不冲突，大幅提升并发性能                    │
└─────────────────────────────────────────────────────┘
```

**InnoDB MVCC 实现细节**：

```
每行数据的隐藏列：
┌─────────────────────────────────────────────────────┐
│ DB_TRX_ID    │ 最后修改该行的事务 ID                  │
│ DB_ROLL_PTR  │ 回滚指针，指向 undo log 中的旧版本    │
│ DB_ROW_ID    │ 行 ID（无主键时使用）                  │
└─────────────────────────────────────────────────────┘

Undo Log 链：
┌─────────────────────────────────────────────────────┐
│ 当前数据行                                           │
│   DB_TRX_ID = 103, DB_ROLL_PTR → undo log           │
│                                                     │
│ Undo Log（事务103修改前的版本）                      │
│   DB_TRX_ID = 102, DB_ROLL_PTR → undo log           │
│                                                     │
│ Undo Log（事务102修改前的版本）                      │
│   DB_TRX_ID = 100, DB_ROLL_PTR → NULL               │
│                                                     │
│ 通过回滚指针串联，形成版本链                          │
└─────────────────────────────────────────────────────┘
```

**Read View 可见性判断**：

```
Read View 创建时记录的信息：
┌─────────────────────────────────────────────────────┐
│ m_ids          │ 创建时活跃的事务 ID 列表            │
│ min_trx_id     │ m_ids 中最小的事务 ID              │
│ max_trx_id     │ 下一个将分配的事务 ID（非最大）     │
│ creator_trx_id │ 创建该 Read View 的事务 ID         │
└─────────────────────────────────────────────────────┘

判断某版本是否可见：
┌─────────────────────────────────────────────────────┐
│ if DB_TRX_ID == creator_trx_id:                     │
│     可见（自己修改的）                               │
│                                                     │
│ elif DB_TRX_ID < min_trx_id:                        │
│     可见（事务在 Read View 创建前已提交）            │
│                                                     │
│ elif DB_TRX_ID >= max_trx_id:                       │
│     不可见（事务在 Read View 创建后开启）            │
│                                                     │
│ elif DB_TRX_ID in m_ids:                            │
│     不可见（事务未提交）                             │
│                                                     │
│ else:                                               │
│     可见（事务已提交）                               │
└─────────────────────────────────────────────────────┘
```

**RC 和 RR 的 MVCC 区别**：

```
READ COMMITTED：
┌─────────────────────────────────────────────────────┐
│ 每次 SELECT 都创建新的 Read View                     │
│ 因此能看到其他事务已提交的修改                        │
│ → 可能出现不可重复读                                 │
└─────────────────────────────────────────────────────┘

REPEATABLE READ：
┌─────────────────────────────────────────────────────┐
│ 第一次 SELECT 时创建 Read View，后续复用             │
│ 因此整个事务期间看到的是同一快照                      │
│ → 实现可重复读                                       │
└─────────────────────────────────────────────────────┘
```

### Redo Log 和 Undo Log

**问题背景**：如何保证持久性和原子性？

```
问题1：持久性
┌─────────────────────────────────────────────────────┐
│ 事务提交后，数据写入磁盘                             │
│ 如果写入过程中系统崩溃，数据丢失怎么办？              │
│                                                     │
│ 解决：redo log                                      │
│ - 先写日志，后写数据（WAL：Write-Ahead Logging）    │
│ - 提交前确保 redo log 已持久化                       │
│ - 崩溃后可通过 redo log 恢复数据                     │
└─────────────────────────────────────────────────────┘

问题2：原子性
┌─────────────────────────────────────────────────────┐
│ 事务执行过程中需要回滚                               │
│ 如何恢复到修改前的状态？                             │
│                                                     │
│ 解决：undo log                                      │
│ - 修改数据前，先记录旧值到 undo log                  │
│ - 回滚时根据 undo log 恢复                          │
│ - 也是 MVCC 版本链的基础                            │
└─────────────────────────────────────────────────────┘
```

**WAL（Write-Ahead Logging）原理**：

```
传统方式：直接写数据文件
┌─────────────────────────────────────────────────────┐
│ 数据文件通常是随机 I/O                               │
│ 性能较差                                            │
└─────────────────────────────────────────────────────┘

WAL 方式：先写日志
┌─────────────────────────────────────────────────────┐
│ redo log 是顺序写入                                 │
│ 顺序 I/O 比随机 I/O 快很多                          │
│                                                     │
│ 写入流程：                                          │
│ 1. 修改数据 → 先记录到 redo log buffer              │
│ 2. 提交时 → 将 redo log 刷盘                        │
│ 3. 后台线程 → 异步将数据写入数据文件                 │
│                                                     │
│ 崩溃恢复：                                          │
│ - 重启时检查 redo log                               │
│ - 重放已提交但未写入数据文件的操作                   │
└─────────────────────────────────────────────────────┘
```

```sql
-- 查看日志配置
SHOW VARIABLES LIKE 'innodb_log%';

-- 关键参数
/*
innodb_log_file_size     - 每个 redo log 文件大小
innodb_log_files_in_group - redo log 文件数量
innodb_log_buffer_size    - redo log buffer 大小
innodb_flush_log_at_trx_commit - 日志刷盘策略
  = 0：每秒刷盘一次（可能丢失1秒数据）
  = 1：每次提交都刷盘（最安全，默认）
  = 2：每次提交写入OS缓存，每秒刷盘
*/
```

---

## 锁机制

### 为什么需要锁？

**问题背景**：并发访问数据时，如何保证数据一致性？

```
场景：两个事务同时修改同一行
┌─────────────────────────────────────────────────────┐
│ 事务A                          事务B                │
│ BEGIN;                                              │
│                                BEGIN;               │
│ UPDATE users SET age=age+1    ─────────→ age 变?    │
│   WHERE id=1;                                       │
│                                UPDATE users         │
│                                  SET age=age+2      │
│                                  WHERE id=1;        │
│ COMMIT;                                             │
│                                COMMIT;              │
│                                                     │
│ 问题：两个事务的修改可能相互覆盖                      │
└─────────────────────────────────────────────────────┘

解决方案：锁机制
┌─────────────────────────────────────────────────────┐
│ 事务A 先获取行的排他锁                               │
│ 事务B 等待 A 释放锁                                  │
│ 事务A 提交后，B 才能执行                             │
│                                                     │
│ 通过锁实现事务的隔离性                               │
└─────────────────────────────────────────────────────┘
```

### 锁的类型

**按锁粒度分类**：

```
全局锁：
┌─────────────────────────────────────────────────────┐
│ 锁定整个数据库实例                                   │
│ 用途：全库逻辑备份                                   │
│ 命令：FLUSH TABLES WITH READ LOCK;                  │
└─────────────────────────────────────────────────────┘

表级锁：
┌─────────────────────────────────────────────────────┐
│ 锁定整张表                                          │
│ 类型：表共享锁(S)、表排他锁(X)                       │
│ 意向锁：IS(意向共享)、IX(意向排他)                   │
│ 作用：表明事务意图，加速锁冲突检测                    │
└─────────────────────────────────────────────────────┘

行级锁：
┌─────────────────────────────────────────────────────┐
│ 只锁定需要的行，并发度最高                           │
│ 类型：                                               │
│ - Record Lock：记录锁，锁单行记录                    │
│ - Gap Lock：间隙锁，锁两个记录之间的间隙             │
│ - Next-Key Lock：记录锁 + 间隙锁                    │
└─────────────────────────────────────────────────────┘
```

**按锁模式分类**：

```
共享锁(S Lock)：
┌─────────────────────────────────────────────────────┐
│ 读锁，多个事务可同时持有                             │
│ 获取方式：SELECT ... LOCK IN SHARE MODE             │
│ 阻塞：其他事务的写操作                               │
└─────────────────────────────────────────────────────┘

排他锁(X Lock)：
┌─────────────────────────────────────────────────────┐
│ 写锁，独占锁                                        │
│ 获取方式：SELECT ... FOR UPDATE                     │
│        INSERT / UPDATE / DELETE 自动加锁            │
│ 阻塞：其他事务的读写操作                             │
└─────────────────────────────────────────────────────┘

兼容性矩阵：
        S锁    X锁
S锁     ✓      ✗
X锁     ✗      ✗
```

### Record Lock、Gap Lock、Next-Key Lock

**问题背景**：RR 隔离级别下如何防止幻读？

```
场景：防止幻读
┌─────────────────────────────────────────────────────┐
│ 表 users 有 id: 1, 5, 10                            │
│                                                     │
│ 事务A：SELECT * FROM users WHERE id=7 FOR UPDATE   │
│ 事务B：INSERT INTO users VALUES(7, ...)            │
│                                                     │
│ 如果只锁 id=7 这一行：                               │
│ - 表中原本没有 id=7 的记录                          │
│ - 无法锁定不存在的记录                              │
│ - 事务B 可以插入，产生幻读                          │
│                                                     │
│ 解决：Next-Key Lock                                 │
│ - 锁定 (5, 10] 这个范围                             │
│ - 事务B 无法在这个范围插入                          │
└─────────────────────────────────────────────────────┘
```

**三种行锁详解**：

```
假设表有记录 id: 1, 5, 10, 15

Record Lock（记录锁）：
┌─────────────────────────────────────────────────────┐
│ 只锁定索引记录本身                                   │
│                                                     │
│ SELECT * FROM users WHERE id = 5 FOR UPDATE        │
│ → 锁定 id = 5 这一行                                │
│                                                     │
│  1    5    10   15                                 │
│       ▲                                             │
│       锁                                            │
└─────────────────────────────────────────────────────┘

Gap Lock（间隙锁）：
┌─────────────────────────────────────────────────────┐
│ 锁定两个记录之间的间隙，不包含记录本身                │
│                                                     │
│ SELECT * FROM users WHERE id = 7 FOR UPDATE        │
│ → 锁定 (5, 10) 这个间隙                             │
│                                                     │
│  1    5    10   15                                 │
│       ├────┤                                        │
│        间隙锁                                       │
│                                                     │
│ 阻止其他事务在间隙中插入新记录                       │
└─────────────────────────────────────────────────────┘

Next-Key Lock：
┌─────────────────────────────────────────────────────┐
│ Record Lock + Gap Lock                             │
│ 锁定记录本身和前面的间隙                             │
│                                                     │
│ SELECT * FROM users WHERE id <= 10 FOR UPDATE      │
│ → 锁定 (-∞, 1], (1, 5], (5, 10]                    │
│                                                     │
│  1    5    10   15                                 │
│ ├────┼────┤                                         │
│ (-∞,1](1,5](5,10]                                  │
│                                                     │
│ 完全阻止幻读                                        │
└─────────────────────────────────────────────────────┘
```

**不同场景下的锁使用**：

```sql
-- 查看当前锁情况
SELECT * FROM performance_schema.data_locks;

-- 场景1：主键等值查询，记录存在 → Record Lock
SELECT * FROM users WHERE id = 5 FOR UPDATE;
-- 锁：id = 5 的记录锁

-- 场景2：主键等值查询，记录不存在 → Gap Lock
SELECT * FROM users WHERE id = 7 FOR UPDATE;
-- 锁：(5, 10) 间隙锁

-- 场景3：主键范围查询 → Next-Key Lock
SELECT * FROM users WHERE id <= 10 FOR UPDATE;
-- 锁：(-∞, 1], (1, 5], (5, 10]

-- 场景4：唯一索引等值查询，记录存在 → Record Lock
SELECT * FROM users WHERE email = 'a@b.com' FOR UPDATE;
-- 锁：email 索引记录锁 + 主键记录锁

-- 场景5：非唯一索引查询 → Next-Key Lock
SELECT * FROM users WHERE age = 20 FOR UPDATE;
-- 锁：age 索引的 Next-Key Lock + 主键记录锁
```

### 死锁

**问题背景**：两个事务相互等待对方持有的锁。

```
死锁场景：
┌─────────────────────────────────────────────────────┐
│ 事务A                          事务B                │
│ BEGIN;                                              │
│                                BEGIN;               │
│ UPDATE users SET ...           ─────────→ 获取id=1 │
│   WHERE id=1;                  的锁                 │
│                                UPDATE users ...     │
│                                  WHERE id=2;        │
│ UPDATE users SET ...           ─────────→ 等待id=2 │
│   WHERE id=2;                  的锁                 │
│ (等待B释放id=2的锁)                                 │
│                                UPDATE users ...     │
│                                  WHERE id=1;        │
│                                (等待A释放id=1的锁)  │
│                                                     │
│ 形成循环等待 → 死锁                                 │
└─────────────────────────────────────────────────────┘
```

**死锁检测和处理**：

```sql
-- 查看死锁检测配置
SHOW VARIABLES LIKE 'innodb_deadlock_detect';

-- 查看最近的死锁信息
SHOW ENGINE INNODB STATUS;
-- 在输出中查找 "LATEST DETECTED DEADLOCK" 部分

-- 设置锁等待超时
SHOW VARIABLES LIKE 'innodb_lock_wait_timeout';
SET innodb_lock_wait_timeout = 50;  -- 默认50秒
```

**死锁预防和解决**：

```sql
-- 1. 按固定顺序访问资源
-- ❌ 可能死锁
-- 事务A: UPDATE table1; UPDATE table2;
-- 事务B: UPDATE table2; UPDATE table1;

-- ✅ 按相同顺序
-- 事务A: UPDATE table1; UPDATE table2;
-- 事务B: UPDATE table1; UPDATE table2;

-- 2. 大事务拆分
-- 大事务持有锁的时间长，更容易产生死锁

-- 3. 合理使用索引
-- 无索引可能导致表锁，增加死锁概率

-- 4. 降低隔离级别（权衡）
-- RC 级别不使用 Gap Lock，死锁概率降低

-- 5. 添加重试逻辑（应用层）
-- 检测到死锁错误后，重试事务
```

---

## 性能优化

### 为什么需要执行计划分析？

**问题背景**：SQL 写法不同，性能可能差几个数量级。

```
示例：同样是查询，性能差异巨大

查询1：SELECT * FROM users WHERE email = 'a@b.com';
- 如果 email 有索引：命中索引，毫秒级响应
- 如果 email 无索引：全表扫描，可能需要几秒

查询2：SELECT * FROM users WHERE email LIKE '%@b.com';
- 即使有索引也无法使用（前置模糊）
- 必须全表扫描

通过 EXPLAIN 可以提前了解 MySQL 的执行计划，
发现潜在问题，优化后再上线。
```

### EXPLAIN 详解

```sql
-- 基本用法
EXPLAIN SELECT * FROM users WHERE email = 'a@b.com';

-- MySQL 8.0+ 更详细的分析
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'a@b.com';
```

**各字段含义详解**：

| 字段 | 含义 | 关键值 |
|-----|------|-------|
| id | 查询标识符 | 相同 id 表示同一查询，越大越先执行 |
| select_type | 查询类型 | SIMPLE, PRIMARY, SUBQUERY, DERIVED |
| table | 访问的表 | |
| **type** | **访问类型（最重要）** | 见下表 |
| possible_keys | 可能使用的索引 | |
| key | 实际使用的索引 | NULL 表示未用索引 |
| key_len | 使用的索引长度 | 越短越好 |
| ref | 索引比较的列或常量 | |
| rows | 预估检查的行数 | 越少越好 |
| filtered | 条件过滤后的百分比 | 越高越好 |
| Extra | 额外信息 | 见下表 |

**type 字段（从好到差）**：

```
system   > const > eq_ref > ref > range > index > ALL

system   : 单行表（系统表）
const    : 主键/唯一索引常量查询，最多一行
eq_ref   : 主键/唯一索引关联，每次关联一行
ref      : 非唯一索引查询
range    : 索引范围扫描（>, <, BETWEEN, IN）
index    : 索引全扫描（遍历索引树）
ALL      : 全表扫描（需要优化！）
```

**Extra 字段关键值**：

| 值 | 含义 | 建议 |
|---|------|-----|
| Using index | 使用覆盖索引 | 好，无需回表 |
| Using where | Server 层过滤 | 检查是否可下推 |
| Using index condition | 使用索引下推 | 好 |
| Using temporary | 使用临时表 | 检查是否可优化 |
| Using filesort | 文件排序 | 考虑添加索引 |
| Using join buffer | 使用连接缓冲 | 检查连接条件和索引 |

```sql
-- 优化案例1：全表扫描 → 使用索引
EXPLAIN SELECT * FROM users WHERE YEAR(created_at) = 2024;
-- type: ALL, Extra: Using where

-- 优化：避免函数操作
EXPLAIN SELECT * FROM users 
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
-- type: range, key: idx_created_at

-- 优化案例2：文件排序 → 使用索引排序
EXPLAIN SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;
-- Extra: Using filesort

-- 优化：添加索引
CREATE INDEX idx_created_desc ON orders(created_at DESC);
-- Extra: 空（使用索引排序）

-- 优化案例3：回表 → 覆盖索引
EXPLAIN SELECT user_id, status FROM orders WHERE user_id = 100;
-- 创建复合索引
CREATE INDEX idx_user_status ON orders(user_id, status);
-- Extra: Using index
```

### 慢查询优化

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;  -- 阈值2秒
SET GLOBAL log_queries_not_using_indexes = 'ON';

-- 查看配置
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';

-- 慢查询日志位置
SHOW VARIABLES LIKE 'slow_query_log_file';
```

**常见慢查询模式及优化**：

```sql
-- 1. 大偏移量分页
-- ❌ 问题：偏移量大时，需要扫描大量数据
SELECT * FROM orders ORDER BY id LIMIT 100000, 10;

-- ✅ 优化1：使用游标分页
SELECT * FROM orders WHERE id > 100000 ORDER BY id LIMIT 10;

-- ✅ 优化2：延迟关联
SELECT o.* FROM orders o
INNER JOIN (SELECT id FROM orders ORDER BY id LIMIT 100000, 10) t
ON o.id = t.id;

-- 2. 避免 SELECT *
-- ❌ 问题：查询不需要的列，浪费资源
SELECT * FROM users WHERE id = 1;

-- ✅ 优化：只查需要的列
SELECT id, name, email FROM users WHERE id = 1;

-- 3. OR 条件优化
-- ❌ 问题：OR 可能导致索引失效
SELECT * FROM users WHERE email = 'a@b.com' OR phone = '13800138000';

-- ✅ 优化：使用 UNION
SELECT * FROM users WHERE email = 'a@b.com'
UNION
SELECT * FROM users WHERE phone = '13800138000';

-- 4. 子查询优化
-- ❌ 问题：相关子查询效率低
SELECT * FROM orders o
WHERE user_id IN (SELECT id FROM users WHERE status = 'active');

-- ✅ 优化：使用 JOIN
SELECT o.* FROM orders o
INNER JOIN users u ON o.user_id = u.id
WHERE u.status = 'active';

-- 5. COUNT 优化
-- ❌ 问题：COUNT(*) 扫描全表
SELECT COUNT(*) FROM users;

-- ✅ 优化：使用估算值（精度要求不高时）
SHOW TABLE STATUS LIKE 'users';
-- 或使用信息_schema
SELECT TABLE_ROWS FROM information_schema.tables 
WHERE TABLE_SCHEMA = 'mydb' AND TABLE_NAME = 'users';
```

---

## 连接池配置

### 为什么需要连接池？

```
无连接池：
┌─────────────────────────────────────────────────────┐
│ 每次请求：                                          │
│ 1. 建立TCP连接（三次握手）                          │
│ 2. MySQL 认证                                       │
│ 3. 执行查询                                         │
│ 4. 关闭连接（四次挥手）                             │
│                                                     │
│ 问题：                                              │
│ - 建立连接耗时（通常 10-50ms）                      │
│ - 频繁创建/销毁连接消耗资源                         │
│ - 高并发时可能耗尽连接数                            │
└─────────────────────────────────────────────────────┘

有连接池：
┌─────────────────────────────────────────────────────┐
│ 应用启动时创建一批连接放入池中                       │
│                                                     │
│ 每次请求：                                          │
│ 1. 从池中获取空闲连接                               │
│ 2. 执行查询                                         │
│ 3. 归还连接到池中                                   │
│                                                     │
│ 优势：                                              │
│ - 复用连接，避免频繁创建                            │
│ - 连接数可控                                        │
│ - 响应更快                                          │
└─────────────────────────────────────────────────────┘
```

### 连接池大小计算

```
经验公式：
连接数 = (核心数 * 2) + 有效磁盘数

示例：
- 8核CPU，1块SSD
- 连接数 = 8 * 2 + 1 = 17
- 建议设置 maximumPoolSize = 20

注意：
- 连接不是越多越好
- 过多连接会增加上下文切换开销
- MySQL 有 max_connections 限制（默认 151）
```

### HikariCP 配置示例

```java
// Java HikariCP 配置
HikariConfig config = new HikariConfig();

// 基本配置
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb?useSSL=true&serverTimezone=UTC");
config.setUsername("app_user");
config.setPassword("strong_password");

// 连接池大小
config.setMaximumPoolSize(20);           // 最大连接数
config.setMinimumIdle(5);                // 最小空闲连接

// 超时配置
config.setConnectionTimeout(30000);      // 获取连接超时（毫秒）
config.setIdleTimeout(600000);           // 空闲连接超时（10分钟）
config.setMaxLifetime(1800000);          // 连接最大生命周期（30分钟）

// 连接验证
config.setConnectionTestQuery("SELECT 1");  // 验证连接的SQL
config.setValidationTimeout(3000);          // 验证超时

// 性能优化参数
config.addDataSourceProperty("cachePrepStmts", "true");        // 启用预处理语句缓存
config.addDataSourceProperty("prepStmtCacheSize", "250");      // 缓存大小
config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048"); // 缓存SQL最大长度
config.addDataSourceProperty("useServerPrepStmts", "true");    // 使用服务端预处理
config.addDataSourceProperty("rewriteBatchedStatements", "true"); // 批量重写优化

HikariDataSource dataSource = new HikariDataSource(config);
```

### Python 连接池示例

```python
# Python DBUtils 连接池
from dbutils.pooled_db import PooledDB
import pymysql

# 创建连接池
pool = PooledDB(
    creator=pymysql,          # 使用的数据库驱动
    maxconnections=20,        # 最大连接数
    mincached=5,              # 初始空闲连接数
    maxcached=10,             # 最大空闲连接数
    maxshared=3,              # 最大共享连接数
    blocking=True,            # 连接池耗尽时阻塞等待
    host='localhost',
    port=3306,
    user='app_user',
    password='strong_password',
    database='mydb',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 使用连接
def get_user(user_id):
    conn = pool.connection()  # 从池中获取连接
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
    finally:
        conn.close()  # 归还连接到池中，不是真正关闭
```

---

## 常见问题与解决方案

### 连接数耗尽

```sql
-- 查看当前连接数
SHOW STATUS LIKE 'Threads_connected';
SHOW PROCESSLIST;

-- 查看最大连接数配置
SHOW VARIABLES LIKE 'max_connections';

-- 查找长时间运行的查询
SELECT * FROM information_schema.processlist 
WHERE time > 60 
ORDER BY time DESC;

-- 终止问题连接
-- KILL <process_id>;
```

### 主从延迟

```sql
-- 查看从库状态
SHOW REPLICA STATUS\G

-- 关键指标
-- Seconds_Behind_Source: 延迟秒数
-- Replica_IO_Running: IO 线程状态
-- Replica_SQL_Running: SQL 线程状态

-- 解决方案：开启多线程复制
STOP REPLICA;
SET GLOBAL replica_parallel_workers = 4;
SET GLOBAL replica_parallel_type = 'LOGICAL_CLOCK';
START REPLICA;
```

### 磁盘空间不足

```sql
-- 查看数据库大小
SELECT 
    table_schema AS '数据库',
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS '大小(MB)'
FROM information_schema.tables
GROUP BY table_schema
ORDER BY SUM(data_length + index_length) DESC;

-- 查看大表
SELECT 
    table_name AS '表名',
    table_rows AS '行数',
    ROUND(data_length / 1024 / 1024, 2) AS '数据大小(MB)',
    ROUND(index_length / 1024 / 1024, 2) AS '索引大小(MB)'
FROM information_schema.tables
WHERE table_schema = 'mydb'
ORDER BY data_length DESC
LIMIT 10;

-- 清理 binlog
PURGE BINARY LOGS BEFORE DATE_SUB(NOW(), INTERVAL 7 DAY);
```

---

## 参考资料

- [MySQL 官方文档](https://dev.mysql.com/doc/)
- [MySQL 8.0 Reference Manual](https://dev.mysql.com/doc/refman/8.0/en/)
- [MySQL 8.4 Reference Manual](https://dev.mysql.com/doc/refman/8.4/en/)
- [High Performance MySQL, 4th Edition](https://www.oreilly.com/library/view/high-performance-mysql/9781492059674/)
- [Percona Blog](https://www.percona.com/blog/)