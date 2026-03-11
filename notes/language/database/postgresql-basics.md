# PostgreSQL 学习笔记

PostgreSQL 是一款功能强大的开源关系型数据库管理系统，以其扩展性、标准兼容性和数据完整性著称。本文将系统介绍 PostgreSQL 的现代用法和最佳实践。

<div align="center">
  <img src="https://www.postgresql.org/media/img/about/press/elephant.png" alt="pgsql-logo" width="120">
</div>

## PostgreSQL 16/17 新特性

### PostgreSQL 16 主要改进

PostgreSQL 16 于 2023 年 9 月发布，带来了显著的性能提升和新功能：

| 特性 | 说明 |
|------|------|
| **并行查询增强** | 支持 `FULL` 和 `RIGHT` 连接的并行执行 |
| **批量导入优化** | `COPY` 命令性能提升约 30% |
| **逻辑复制改进** | 支持双因素认证和并行应用 |
| **JSON 改进** | 新增 `json_array_elements` 等函数 |
| **统计信息增强** | 更精确的查询计划估算 |

### PostgreSQL 17 新特性（2024）

PostgreSQL 17 进一步增强了数据库能力：

```sql
-- 新增的 JSON_TABLE 功能（SQL/JSON 标准）
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
```

```sql
-- MERGE 语句增强
MERGE INTO products AS target
USING updates AS source
ON target.id = source.id
WHEN MATCHED THEN
    UPDATE SET price = source.price, stock = target.stock + source.stock
WHEN NOT MATCHED THEN
    INSERT (id, name, price, stock)
    VALUES (source.id, source.name, source.price, source.stock);
```

### 现代版本对比

```sql
-- 检查 PostgreSQL 版本
SELECT version();

-- 查看已安装扩展
SELECT * FROM pg_available_extensions ORDER BY name;

-- 检查扩展版本
SELECT extname, extversion FROM pg_extension;
```

---

## 安装与配置最佳实践

### 多平台安装

#### Windows 安装

```powershell
# 使用 Chocolatey 安装
choco install postgresql -y

# 或使用 Scoop
scoop install postgresql

# 使用 EnterpriseDB 安装器（推荐）
# 下载：https://www.postgresql.org/download/windows/
```

#### Linux 安装（Ubuntu/Debian）

```bash
# 添加 PostgreSQL 官方仓库
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# 安装 PostgreSQL 17
sudo apt update
sudo apt install postgresql-17 postgresql-contrib-17

# 启动服务
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

#### Docker 部署（云原生推荐）

```bash
# 拉取最新镜像
docker pull postgres:17

# 启动容器（生产环境配置）
docker run -d \
    --name postgres17 \
    --restart unless-stopped \
    -e POSTGRES_USER=appuser \
    -e POSTGRES_PASSWORD=secure_password \
    -e POSTGRES_DB=myapp \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v postgres_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    postgres:17 \
    -c shared_buffers=256MB \
    -c max_connections=200 \
    -c work_mem=4MB \
    -c effective_cache_size=1GB
```

### 配置文件优化

PostgreSQL 的主要配置文件是 `postgresql.conf`，以下是最关键的参数：

```ini
# postgresql.conf - 内存配置

# 共享缓冲区（推荐设置为系统内存的 25%）
shared_buffers = 256MB          # 开发环境
# shared_buffers = 4GB          # 生产环境（16GB 内存）

# 工作内存（每个查询操作可用内存）
work_mem = 4MB                  # 默认值，复杂查询可增加
# work_mem = 64MB               # 分析型工作负载

# 维护工作内存（VACUUM、CREATE INDEX 等）
maintenance_work_mem = 256MB

# 有效缓存大小（规划器估计）
effective_cache_size = 1GB      # 约系统内存的 50-75%

# WAL 配置
wal_buffers = 64MB
checkpoint_completion_target = 0.9

# 查询规划
random_page_cost = 1.1          # SSD 环境（传统 HDD 为 4.0）
effective_io_concurrency = 200  # SSD 环境
```

```ini
# postgresql.conf - 连接配置

max_connections = 200           # 最大连接数
superuser_reserved_connections = 3

# 连接超时
authentication_timeout = 60s
idle_in_transaction_session_timeout = 600000  # 10分钟
```

```ini
# postgresql.conf - 并行查询（PostgreSQL 16+）

max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# 并行顺序扫描
enable_parallel_append = on
enable_parallel_hash = on
```

### pg_hba.conf 访问控制

```
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# 本地连接
local   all             all                                     peer

# IPv4 本地连接
host    all             all             127.0.0.1/32            scram-sha-256

# IPv6 本地连接  
host    all             all             ::1/128                 scram-sha-256

# 允许应用服务器连接
host    myapp           appuser         192.168.1.0/24          scram-sha-256

# 只读副本连接
host    replication     replicator      192.168.1.10/32         scram-sha-256
```

---

## 丰富数据类型

PostgreSQL 提供了丰富的内置数据类型，是其强大功能的重要体现。

### 基本数据类型

```sql
-- 创建表展示基本类型
CREATE TABLE data_types_demo (
    -- 自增主键（现代方式：IDENTITY 替代 SERIAL）
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    
    -- 字符类型
    name VARCHAR(100) NOT NULL,          -- 可变长度
    description TEXT,                     -- 无限长度
    code CHAR(10),                        -- 固定长度
    
    -- 数值类型
    small_num SMALLINT,                   -- 2字节整数
    normal_num INTEGER,                   -- 4字节整数
    big_num BIGINT,                       -- 8字节整数
    price DECIMAL(10, 2),                 -- 精确小数
    scientific REAL,                      -- 单精度浮点
    precise DOUBLE PRECISION,             -- 双精度浮点
    
    -- 布尔类型
    is_active BOOLEAN DEFAULT true,
    
    -- 日期时间类型
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    birth_date DATE,
    meeting_time TIME,
    duration INTERVAL
);
```

### JSONB 类型 🌟

JSONB 是 PostgreSQL 最强大的特性之一，支持索引和高效查询：

```sql
-- 创建带 JSONB 列的表
CREATE TABLE products (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    attributes JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- 插入 JSON 数据
INSERT INTO products (name, attributes, metadata) VALUES
(
    'MacBook Pro',
    '{
        "brand": "Apple",
        "specs": {
            "cpu": "M3 Pro",
            "ram": 18,
            "storage": 512
        },
        "tags": ["laptop", "professional", "apple"],
        "price": 1999.00,
        "in_stock": true
    }',
    '{"source": "official", "updated": "2024-01-15"}'
),
(
    'ThinkPad X1',
    '{
        "brand": "Lenovo",
        "specs": {
            "cpu": "Intel i7",
            "ram": 32,
            "storage": 1024
        },
        "tags": ["laptop", "business"],
        "price": 1499.00,
        "in_stock": true
    }',
    '{"source": "reseller"}'
);
```

```sql
-- JSONB 查询操作符

-- 提取值（-> 返回 JSON，->> 返回文本）
SELECT 
    name,
    attributes->>'brand' AS brand,
    attributes->'specs'->>'cpu' AS cpu,
    attributes->'specs'->'ram' AS ram
FROM products;

-- 条件查询
SELECT name, attributes->>'brand' AS brand
FROM products
WHERE attributes->>'brand' = 'Apple';

-- 包含查询 (@>)
SELECT name
FROM products
WHERE attributes @> '{"brand": "Apple"}';

-- 检查键是否存在
SELECT name
FROM products
WHERE attributes ? 'tags';

-- 检查是否包含数组元素
SELECT name
FROM products
WHERE attributes->'tags' ? 'professional';

-- 数组元素查询（任意匹配）
SELECT name
FROM products
WHERE attributes->'tags' ?| ARRAY['professional', 'business'];
```

```sql
-- JSONB 索引创建

-- GIN 索引（支持 @>、?、?|、?& 操作符）
CREATE INDEX idx_products_attributes_gin ON products USING GIN (attributes);

-- GIN 索引带 jsonb_path_ops（更小更快，只支持 @>）
CREATE INDEX idx_products_attributes_path ON products USING GIN (attributes jsonb_path_ops);

-- 表达式索引（针对特定字段）
CREATE INDEX idx_products_brand ON products ((attributes->>'brand'));

-- 使用索引的查询示例
EXPLAIN ANALYZE
SELECT * FROM products WHERE attributes @> '{"brand": "Apple"}';
```

```sql
-- JSONB 修改操作

-- 更新整个 JSON
UPDATE products 
SET attributes = attributes || '{"warranty": "1 year"}'::jsonb
WHERE id = 1;

-- 更新嵌套字段
UPDATE products
SET attributes = jsonb_set(
    attributes,
    '{specs,ram}',
    '36'::jsonb
)
WHERE id = 1;

-- 删除字段
UPDATE products
SET attributes = attributes - 'warranty'
WHERE id = 1;

-- 删除嵌套字段
UPDATE products
SET attributes = attributes #- '{specs,storage}'
WHERE id = 1;
```

### 数组类型

```sql
-- 创建带数组列的表
CREATE TABLE posts (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    ratings INTEGER[] DEFAULT '{}',
    comments TEXT[][] DEFAULT '{}'
);

-- 插入数据
INSERT INTO posts (title, tags, ratings, comments) VALUES
('PostgreSQL 教程', ARRAY['database', 'sql', 'tutorial'], ARRAY[5, 4, 5, 4], ARRAY[['user1', 'great'], ['user2', 'helpful']]),
('JSONB 详解', '{json, postgresql, advanced}', '{5, 5, 4}', '{{"alice", "excellent"}, {"bob", "detailed"}}');

-- 数组查询
SELECT title, tags FROM posts;

-- 访问数组元素（索引从1开始）
SELECT title, tags[1] AS first_tag, ratings[1:2] AS first_two_ratings
FROM posts;

-- 数组长度
SELECT title, array_length(tags, 1) AS tag_count
FROM posts;

-- ANY 查询（数组中是否包含某元素）
SELECT title FROM posts WHERE 'sql' = ANY(tags);

-- @> 包含操作符
SELECT title FROM posts WHERE tags @> ARRAY['database'];

-- && 重叠操作符
SELECT title FROM posts WHERE tags && ARRAY['json', 'nosql'];

-- unnest 展开数组
SELECT title, unnest(tags) AS tag
FROM posts;

-- array_agg 聚合为数组
SELECT array_agg(name) AS all_names FROM products;

-- 使用数组函数
SELECT 
    title,
    array_length(tags, 1) AS tag_count,
    array_to_string(tags, ', ') AS tags_string,
    cardinality(ratings) AS rating_count  -- PostgreSQL 9.4+
FROM posts;
```

### 范围类型

```sql
-- 内置范围类型
-- int4range, int8range, numrange, tsrange, tstzrange, daterange

CREATE TABLE reservations (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    room_id INTEGER NOT NULL,
    stay_dates DATERANGE NOT NULL,
    price_range NUMRANGE,
    meeting_time TSRANGE
);

-- 插入数据
INSERT INTO reservations (room_id, stay_dates, price_range, meeting_time) VALUES
(101, '[2024-03-01, 2024-03-05)', '[100, 200)', '[2024-03-01 14:00, 2024-03-01 16:00)'),
(102, '[2024-03-03, 2024-03-07)', '[150, 300)', '[2024-03-03 09:00, 2024-03-03 11:00)');

-- 范围查询
SELECT * FROM reservations WHERE stay_dates @> DATE '2024-03-03';

-- 范围重叠检查
SELECT r1.room_id, r2.room_id
FROM reservations r1, reservations r2
WHERE r1.id < r2.id AND r1.stay_dates && r2.stay_dates;

-- 范围函数
SELECT 
    room_id,
    lower(stay_dates) AS check_in,
    upper(stay_dates) AS check_out,
    stay_dates - DATE '2024-03-01' AS days_in_range
FROM reservations;

-- 创建 GIST 索引加速范围查询
CREATE INDEX idx_reservations_dates ON reservations USING GIST (stay_dates);
```

### 几何类型

```sql
-- 几何类型：point, line, lseg, box, path, polygon, circle

CREATE TABLE locations (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position POINT NOT NULL,
    area POLYGON,
    boundary PATH,
    coverage CIRCLE
);

-- 插入数据
INSERT INTO locations (name, position, area, coverage) VALUES
('总部', '(40.7128, -74.0060)', '((0,0),(0,10),(10,10),(10,0))', '<(5,5),3>'),
('分部', '(34.0522, -118.2437)', '((5,5),(5,15),(15,15),(15,5))', '<(10,10),5>');

-- 几何计算
SELECT 
    name,
    position,
    area @> point '(2,3)' AS point_in_area,
    distance(position, point '(40.0, -74.0)') AS distance
FROM locations;

-- 使用 PostGIS 扩展进行更强大的地理计算
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE geo_locations (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    geom GEOMETRY(Point, 4326)  -- SRID 4326 = WGS84
);

-- 插入空间数据
INSERT INTO geo_locations (name, geom) VALUES
('New York', ST_SetSRID(ST_MakePoint(-74.0060, 40.7128), 4326)),
('Los Angeles', ST_SetSRID(ST_MakePoint(-118.2437, 34.0522), 4326));

-- 空间查询
SELECT 
    name,
    ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(-74.0, 40.7), 4326)::geography) / 1000 AS distance_km
FROM geo_locations
ORDER BY distance_km;
```

---

## 高级 SQL 特性

### 窗口函数

窗口函数允许在不减少行数的情况下进行聚合计算：

```sql
-- 示例数据
CREATE TABLE sales (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_name VARCHAR(100),
    sale_date DATE DEFAULT CURRENT_DATE,
    amount DECIMAL(10, 2),
    region VARCHAR(50)
);

INSERT INTO sales (product_name, amount, region, sale_date) VALUES
('Laptop', 1200.00, 'North', '2024-01-01'),
('Phone', 800.00, 'North', '2024-01-01'),
('Laptop', 1200.00, 'South', '2024-01-02'),
('Tablet', 500.00, 'North', '2024-01-02'),
('Phone', 850.00, 'South', '2024-01-03'),
('Laptop', 1100.00, 'East', '2024-01-03'),
('Tablet', 550.00, 'South', '2024-01-04'),
('Phone', 799.00, 'East', '2024-01-04');
```

```sql
-- 基本窗口函数
SELECT 
    product_name,
    region,
    amount,
    -- 累计总和
    SUM(amount) OVER (ORDER BY sale_date) AS running_total,
    -- 移动平均（最近3行）
    AVG(amount) OVER (
        ORDER BY sale_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg,
    -- 排名
    RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS region_rank,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS overall_rank,
    ROW_NUMBER() OVER (PARTITION BY region ORDER BY sale_date) AS row_num
FROM sales
ORDER BY sale_date, region;
```

```sql
-- 高级窗口函数应用

-- 计算每个区域销售额占比
SELECT 
    region,
    amount,
    SUM(amount) OVER () AS total_sales,
    ROUND(amount / SUM(amount) OVER () * 100, 2) AS pct_of_total,
    ROUND(amount / SUM(amount) OVER (PARTITION BY region) * 100, 2) AS pct_of_region
FROM sales;

-- 年度同比增长计算
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
    LAG(monthly_total, 1) OVER (ORDER BY month) AS prev_month,
    monthly_total - LAG(monthly_total, 1) OVER (ORDER BY month) AS diff,
    ROUND(
        (monthly_total - LAG(monthly_total, 1) OVER (ORDER BY month)) / 
        LAG(monthly_total, 1) OVER (ORDER BY month) * 100, 
    2) AS growth_pct
FROM monthly_sales;

-- FIRST_VALUE 和 LAST_VALUE
SELECT DISTINCT
    region,
    FIRST_VALUE(amount) OVER (
        PARTITION BY region 
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_sale,
    LAST_VALUE(amount) OVER (
        PARTITION BY region 
        ORDER BY sale_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sale
FROM sales;
```

### CTE（公共表表达式）

```sql
-- 基本 CTE
WITH high_value_sales AS (
    SELECT * FROM sales WHERE amount > 1000
),
regional_stats AS (
    SELECT region, AVG(amount) AS avg_amount
    FROM sales
    GROUP BY region
)
SELECT 
    h.product_name,
    h.region,
    h.amount,
    r.avg_amount,
    h.amount - r.avg_amount AS above_avg
FROM high_value_sales h
JOIN regional_stats r ON h.region = r.region;

-- CTE 与数据修改
WITH deleted_sales AS (
    DELETE FROM sales WHERE amount < 100
    RETURNING *
)
INSERT INTO sales_archive SELECT * FROM deleted_sales;

-- CTE 优化：MATERIALIZED vs NOT MATERIALIZED
WITH MATERIALIZED expensive_cte AS (
    -- 复杂查询只执行一次
    SELECT region, SUM(amount) AS total
    FROM sales
    GROUP BY region
)
SELECT * FROM expensive_cte WHERE total > 1000;

WITH NOT MATERIALIZED simple_cte AS (
    -- 简单查询可能被内联优化
    SELECT * FROM sales WHERE region = 'North'
)
SELECT * FROM simple_cte;
```

### 递归查询

递归 CTE 用于处理层次结构数据：

```sql
-- 组织架构示例
CREATE TABLE employees (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    manager_id INTEGER REFERENCES employees(id),
    department VARCHAR(50),
    salary DECIMAL(10, 2)
);

INSERT INTO employees (name, manager_id, department, salary) VALUES
('CEO', NULL, 'Executive', 200000),
('CTO', 1, 'Technology', 150000),
('CFO', 1, 'Finance', 150000),
('Dev Manager', 2, 'Technology', 120000),
('Finance Manager', 3, 'Finance', 120000),
('Senior Dev', 4, 'Technology', 100000),
('Junior Dev', 4, 'Technology', 70000),
('Accountant', 5, 'Finance', 80000);
```

```sql
-- 递归查询：获取员工层级
WITH RECURSIVE org_hierarchy AS (
    -- 基础查询（锚点）
    SELECT 
        id,
        name,
        manager_id,
        department,
        salary,
        1 AS level,
        ARRAY[id] AS path,
        name::TEXT AS path_names
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- 递归部分
    SELECT 
        e.id,
        e.name,
        e.manager_id,
        e.department,
        e.salary,
        h.level + 1,
        h.path || e.id,
        h.path_names || ' > ' || e.name
    FROM employees e
    INNER JOIN org_hierarchy h ON e.manager_id = h.id
)
SELECT 
    REPEAT('    ', level - 1) || name AS org_chart,
    department,
    level,
    path_names
FROM org_hierarchy
ORDER BY path;

-- 结果示例：
--     org_chart        | department  | level | path_names
-- ---------------------+-------------+-------+---------------------------
--  CEO                 | Executive   |     1 | CEO
--      CTO             | Technology  |     2 | CEO > CTO
--          Dev Manager | Technology  |     3 | CEO > CTO > Dev Manager
--              Senior Dev | Technology|   4 | CEO > CTO > Dev Manager > Senior Dev
--              Junior Dev | Technology|   4 | CEO > CTO > Dev Manager > Junior Dev
--      CFO             | Finance     |     2 | CEO > CFO
--          Finance Manager | Finance |   3 | CEO > CFO > Finance Manager
--              Accountant | Finance   |   4 | CEO > CFO > Finance Manager > Accountant
```

```sql
-- 递归查询：计算部门预算
WITH RECURSIVE dept_budget AS (
    -- 基础：叶子节点（没有下属）
    SELECT 
        id,
        name,
        manager_id,
        department,
        salary AS total_salary,
        salary AS direct_salary,
        0 AS team_salary,
        1 AS team_size
    FROM employees e
    WHERE NOT EXISTS (SELECT 1 FROM employees WHERE manager_id = e.id)
    
    UNION ALL
    
    -- 递归：向上汇总
    SELECT 
        e.id,
        e.name,
        e.manager_id,
        e.department,
        e.salary + db.team_salary AS total_salary,
        e.salary AS direct_salary,
        db.team_salary,
        db.team_size
    FROM employees e
    INNER JOIN dept_budget db ON db.manager_id = e.id
)
SELECT 
    name,
    department,
    direct_salary,
    team_salary,
    total_salary
FROM dept_budget
ORDER BY total_salary DESC;
```

### LATERAL JOIN

LATERAL 允许子查询引用外部查询的列：

```sql
-- 示例数据
CREATE TABLE orders (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE DEFAULT CURRENT_DATE,
    total DECIMAL(10, 2)
);

CREATE TABLE order_items (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_name VARCHAR(100),
    quantity INTEGER,
    price DECIMAL(10, 2)
);

INSERT INTO orders (customer_id, total) VALUES
(1, 500.00), (1, 300.00), (2, 450.00), (2, 200.00), (2, 150.00);

INSERT INTO order_items (order_id, product_name, quantity, price) VALUES
(1, 'Laptop', 1, 500.00),
(2, 'Phone', 2, 150.00),
(3, 'Tablet', 3, 150.00),
(4, 'Headphones', 1, 200.00),
(5, 'Mouse', 2, 75.00);
```

```sql
-- 获取每个客户最近的订单（LATERAL 方式）
SELECT 
    c.id AS customer_id,
    c.name,
    recent.order_id,
    recent.order_date,
    recent.total
FROM customers c
CROSS JOIN LATERAL (
    SELECT 
        o.id AS order_id,
        o.order_date,
        o.total
    FROM orders o
    WHERE o.customer_id = c.id
    ORDER BY o.order_date DESC
    LIMIT 3
) recent;

-- 获取每个订单最贵的商品
SELECT 
    o.id AS order_id,
    o.total,
    top_item.product_name,
    top_item.item_total
FROM orders o
CROSS JOIN LATERAL (
    SELECT 
        product_name,
        quantity * price AS item_total
    FROM order_items
    WHERE order_id = o.id
    ORDER BY quantity * price DESC
    LIMIT 1
) top_item;
```

```sql
-- LATERAL 与 JSONB 结合
SELECT 
    p.name AS product_name,
    attr.key AS attribute_name,
    attr.value AS attribute_value
FROM products p
CROSS JOIN LATERAL jsonb_each_text(p.attributes) AS attr(key, value)
WHERE attr.key IN ('brand', 'price');

-- 结果：
--  product_name | attribute_name | attribute_value
-- --------------+----------------+-----------------
--  MacBook Pro  | brand          | Apple
--  MacBook Pro  | price          | 1999.00
--  ThinkPad X1  | brand          | Lenovo
--  ThinkPad X1  | price          | 1499.00
```

---

## 索引类型

PostgreSQL 提供多种索引类型以适应不同的查询场景：

### B-tree 索引（默认）

```sql
-- B-tree 适用于等值、范围、排序查询
CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_products_price ON products((attributes->>'price'));

-- 复合索引
CREATE INDEX idx_products_brand_price ON products((attributes->>'brand'), (attributes->>'price'));

-- 条件索引
CREATE INDEX idx_products_active ON products(name) WHERE (attributes->>'in_stock')::boolean = true;

-- 查看索引使用情况
EXPLAIN ANALYZE SELECT * FROM products WHERE name = 'MacBook Pro';
```

### GIN 索引

GIN（Generalized Inverted Index）适用于多值元素：

```sql
-- JSONB GIN 索引
CREATE INDEX idx_products_attr_gin ON products USING GIN (attributes);

-- JSONB 路径索引（更小更快）
CREATE INDEX idx_products_attr_path ON products USING GIN (attributes jsonb_path_ops);

-- 数组 GIN 索引
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);

-- 全文搜索 GIN 索引
CREATE INDEX idx_posts_content_fts ON posts USING GIN (to_tsvector('english', content));

-- 使用 tsvector 列优化全文搜索
ALTER TABLE posts ADD COLUMN content_tsv TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED;

CREATE INDEX idx_posts_content_tsv ON posts USING GIN (content_tsv);

-- 查询示例
SELECT title FROM posts WHERE content_tsv @@ to_tsquery('postgresql & tutorial');
```

### GiST 索引

GiST（Generalized Search Tree）适用于几何和范围类型：

```sql
-- 范围类型 GiST 索引
CREATE INDEX idx_reservations_dates ON reservations USING GIST (stay_dates);

-- 几何类型 GiST 索引
CREATE INDEX idx_locations_position ON locations USING GIST (position);

-- PostGIS 几何索引
CREATE INDEX idx_geo_locations_geom ON geo_locations USING GIST (geom);

-- 排除约束（使用 GiST）
CREATE TABLE meetings (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    room_id INTEGER NOT NULL,
    during TSRANGE,
    EXCLUDE USING GIST (
        room_id WITH =,
        during WITH &&
    )
);

-- 插入重叠时间会失败
INSERT INTO meetings (room_id, during) VALUES
(1, '[2024-03-01 10:00, 2024-03-01 11:00)'),
(1, '[2024-03-01 10:30, 2024-03-01 12:00)');  -- 错误！时间冲突
```

### BRIN 索引

BRIN（Block Range Index）适用于大型有序表：

```sql
-- BRIN 索引非常小，适合时序数据
CREATE INDEX idx_logs_created_at ON logs USING BRIN (created_at);

-- 多列 BRIN 索引
CREATE INDEX idx_logs_time_region ON logs USING BRIN (created_at, region_id) WITH (pages_per_range = 128);

-- 查看索引大小对比
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes 
WHERE tablename = 'logs';

-- 结果示例：
-- B-tree:  100 MB
-- BRIN:    128 KB  (小得多，但查询需要扫描更多块)
```

### Hash 索引

```sql
-- Hash 索引只支持等值查询，不推荐常规使用
CREATE INDEX idx_users_email_hash ON users USING HASH (email);

-- PostgreSQL 10+ 支持 WAL，崩溃安全
-- 但 B-tree 通常更好，除非有特殊需求
```

### 索引使用分析

```sql
-- 查看表的所有索引
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'products';

-- 查看索引使用统计
SELECT 
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 查找未使用的索引
SELECT 
    indexrelname AS index_name,
    relname AS table_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;

-- 索引建议（需要 pg_stat_statements）
SELECT 
    query,
    calls,
    total_time,
    rows
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat%'
ORDER BY total_time DESC
LIMIT 10;
```

---

## 事务与 MVCC 实现

### MVCC 原理

PostgreSQL 使用多版本并发控制（MVCC）实现事务隔离：

```sql
-- 查看当前事务 ID
SELECT txid_current();

-- 查看系统快照
SELECT * FROM pg_current_snapshot();

-- 查看行版本信息
SELECT 
    xmin,  -- 插入该行的事务 ID
    xmax,  -- 删除/更新该行的事务 ID
    ctid,  -- 行的物理位置 (页号, 行号)
    cmin,  -- 命令序号（插入）
    cmax   -- 命令序号（删除）
FROM products LIMIT 5;
```

```sql
-- MVCC 行可见性演示

-- 会话 1
BEGIN;
SELECT txid_current();  -- 假设返回 100
UPDATE products SET name = 'Updated Name' WHERE id = 1;
-- 此时旧版本 xmax = 100，新版本 xmin = 100

-- 会话 2（并行执行）
BEGIN;
SELECT txid_current();  -- 假设返回 101
SELECT * FROM products WHERE id = 1;  -- 仍然看到旧版本
-- 因为事务 100 未提交

-- 会话 1
COMMIT;  -- 提交

-- 会话 2
SELECT * FROM products WHERE id = 1;  -- 仍然看到旧版本（快照隔离）
COMMIT;

-- 会话 2 新事务
BEGIN;
SELECT * FROM products WHERE id = 1;  -- 现在看到新版本
COMMIT;
```

### 事务隔离级别

```sql
-- PostgreSQL 支持四种隔离级别
-- Read Uncommitted ≈ Read Committed（PG 特殊处理）

-- 查看当前隔离级别
SHOW default_transaction_isolation;

-- 设置事务隔离级别
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- 设置默认隔离级别
ALTER DATABASE mydb SET default_transaction_isolation = 'read committed';
```

```sql
-- 隔离级别演示

-- READ COMMITTED（默认）：每条语句看到最新提交的数据
-- 会话 1
BEGIN;
SELECT * FROM products WHERE id = 1;  -- 看到版本 A

-- 会话 2
UPDATE products SET name = 'Version B' WHERE id = 1;
COMMIT;

-- 会话 1
SELECT * FROM products WHERE id = 1;  -- 现在看到版本 B
COMMIT;

-- REPEATABLE READ：整个事务看到一致快照
-- 会话 1
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM products WHERE id = 1;  -- 看到版本 A

-- 会话 2
UPDATE products SET name = 'Version C' WHERE id = 1;
COMMIT;

-- 会话 1
SELECT * FROM products WHERE id = 1;  -- 仍然看到版本 A
COMMIT;

-- SERIALIZABLE：完全隔离，检测序列化冲突
-- 会话 1
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT SUM(amount) FROM sales WHERE region = 'North';  -- 1000

-- 会话 2
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
INSERT INTO sales (amount, region) VALUES (100, 'North');
COMMIT;

-- 会话 1
INSERT INTO sales (amount, region) VALUES (200, 'North');
COMMIT;  -- 可能报错：序列化冲突
```

### 锁机制

```sql
-- 查看当前锁
SELECT 
    locktype,
    relation::regclass,
    mode,
    granted,
    pid
FROM pg_locks
WHERE relation IS NOT NULL;

-- 行级锁
BEGIN;
SELECT * FROM products WHERE id = 1 FOR UPDATE;  -- 排他行锁
SELECT * FROM products WHERE id = 2 FOR SHARE;    -- 共享行锁
SELECT * FROM products WHERE id = 3 FOR UPDATE NOWAIT;  -- 不等待，立即失败
SELECT * FROM products WHERE id = 4 FOR UPDATE SKIP LOCKED;  -- 跳过已锁定行
COMMIT;

-- 表级锁
BEGIN;
LOCK TABLE products IN SHARE MODE;        -- 共享锁，阻止写入
LOCK TABLE products IN EXCLUSIVE MODE;     -- 排他锁，阻止读写
LOCK TABLE products IN ACCESS EXCLUSIVE MODE;  -- 最高级排他锁
COMMIT;

-- 咨询锁（应用级锁）
-- 会话 1
SELECT pg_advisory_lock(12345);  -- 获取锁

-- 会话 2
SELECT pg_try_advisory_lock(12345);  -- 尝试获取，失败返回 false

-- 会话 1
SELECT pg_advisory_unlock(12345);  -- 释放锁
```

### 死锁检测与处理

```sql
-- 设置死锁超时
SET deadlock_timeout = '1s';

-- 查看等待事件
SELECT 
    pid,
    wait_event_type,
    wait_event,
    state,
    query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL;

-- 终止阻塞会话
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'active' 
    AND query NOT LIKE '%pg_stat_activity%'
    AND pid != pg_backend_pid();

-- 取消正在执行的查询
SELECT pg_cancel_backend(pid);
```

---

## 性能调优

### VACUUM 与垃圾回收

```sql
-- MVCC 产生死元组需要清理
-- 查看表的死元组数量
SELECT 
    relname AS table_name,
    n_live_tup AS live_tuples,
    n_dead_tup AS dead_tuples,
    ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_ratio
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- 手动 VACUUM（不阻塞）
VACUUM products;

-- VACUUM FULL（重建表，需要排他锁）
-- 警告：会锁表！
VACUUM FULL products;

-- ANALYZE 更新统计信息
ANALYZE products;

-- 同时执行
VACUUM ANALYZE products;
```

```sql
-- 自动清理配置
-- postgresql.conf
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min

-- 触发阈值
autovacuum_vacuum_threshold = 50        -- 最少死元组数
autovacuum_vacuum_scale_factor = 0.1    -- 死元组比例阈值
autovacuum_analyze_threshold = 50
autovacuum_analyze_scale_factor = 0.05

-- 表级配置（覆盖全局）
ALTER TABLE products SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_vacuum_threshold = 100
);

-- 查看自动清理状态
SELECT 
    relname,
    last_vacuum,
    last_autovacuum,
    vacuum_count,
    autovacuum_count
FROM pg_stat_user_tables
WHERE autovacuum_count > 0;
```

### 并行查询

```sql
-- PostgreSQL 16+ 并行查询增强

-- 查看并行配置
SHOW max_parallel_workers_per_gather;
SHOW max_parallel_workers;
SHOW parallel_tuple_cost;
SHOW parallel_setup_cost;

-- 强制并行查询
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET max_parallel_workers_per_gather = 4;

-- 查看并行执行计划
EXPLAIN ANALYZE 
SELECT COUNT(*) FROM large_table WHERE status = 'active';

-- 示例输出：
--                                                      QUERY PLAN
-- ---------------------------------------------------------------------------------------------------------------------
--  Finalize Aggregate  (cost=... rows=1) (actual time=... rows=1)
--    ->  Gather  (cost=... rows=3)
--          Workers Planned: 2
--          Workers Launched: 2
--            ->  Partial Aggregate  (cost=... rows=1)
--                  ->  Parallel Seq Scan on large_table  (cost=... rows=...)
--                        Filter: (status = 'active'::text)
```

### 查询计划分析

```sql
-- 基本查询计划
EXPLAIN SELECT * FROM products WHERE name = 'MacBook Pro';

-- 详细查询计划（实际执行）
EXPLAIN ANALYZE SELECT * FROM products WHERE name = 'MacBook Pro';

-- 更详细的输出
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) 
SELECT * FROM products WHERE name = 'MacBook Pro';

-- JSON 格式输出
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM products WHERE name = 'MacBook Pro';

-- 查看估计成本与实际行数差异
EXPLAIN ANALYZE 
SELECT * FROM products 
WHERE (attributes->>'price')::numeric > 1000;

-- 如果估计行数与实际差异大，需要 ANALYZE
ANALYZE products;
```

### 统计信息优化

```sql
-- 增加统计精度（默认 100）
ALTER TABLE products ALTER COLUMN name SET STATISTICS 500;
ANALYZE products;

-- 创建扩展统计信息（多列相关性）
CREATE STATISTICS s1 (dependencies, ndistinct) ON region, product_name FROM sales;
ANALYZE sales;

-- 查看统计信息
SELECT 
    attname,
    n_distinct,
    most_common_vals,
    most_common_freqs
FROM pg_stats
WHERE tablename = 'products';

-- 表达式统计信息
CREATE STATISTICS s2 ON (attributes->>'brand') FROM products;
ANALYZE products;
```

---

## pg_stat_statements 监控

### 安装与配置

```sql
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- postgresql.conf 配置
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.max = 10000

-- 重启后生效
```

### 监控查询

```sql
-- 最耗时的 SQL
SELECT 
    calls,
    ROUND(total_exec_time::numeric, 2) AS total_time_ms,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    ROUND((100 * total_exec_time / SUM(total_exec_time) OVER())::numeric, 2) AS pct,
    rows,
    query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- 执行次数最多的 SQL
SELECT 
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_time_ms,
    rows,
    query
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 10;

-- 平均 I/O 最高的 SQL
SELECT 
    calls,
    ROUND(shared_blks_hit::numeric / NULLIF(calls, 0), 2) AS avg_blks_hit,
    ROUND(shared_blks_read::numeric / NULLIF(calls, 0), 2) AS avg_blks_read,
    query
FROM pg_stat_statements
ORDER BY shared_blks_read DESC
LIMIT 10;

-- 重置统计
SELECT pg_stat_statements_reset();
```

### 性能问题定位

```sql
-- 查找慢查询
SELECT 
    dbid,
    queryid,
    calls,
    ROUND(total_exec_time::numeric / 1000, 2) AS total_sec,
    ROUND(mean_exec_time::numeric, 2) AS avg_ms,
    ROUND(min_exec_time::numeric, 2) AS min_ms,
    ROUND(max_exec_time::numeric, 2) AS max_ms,
    rows,
    LEFT(query, 100) AS query_preview
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- 平均超过 1 秒
ORDER BY mean_exec_time DESC;

-- 查找缓存命中率低的查询
SELECT 
    query,
    calls,
    shared_blks_hit,
    shared_blks_read,
    ROUND(
        shared_blks_hit::numeric / 
        NULLIF(shared_blks_hit + shared_blks_read, 0) * 100, 
    2) AS cache_hit_pct
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;
```

---

## 连接池配置（PgBouncer）

### PgBouncer 安装与配置

```ini
; pgbouncer.ini

[databases]
; 生产数据库配置
myapp = host=127.0.0.1 port=5432 dbname=myapp

[pgbouncer]
; 监听配置
listen_addr = 0.0.0.0
listen_port = 6432

; 认证
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

; 连接池模式
; session - 会话级（推荐，兼容性最好）
; transaction - 事务级（最高效，有兼容性要求）
; statement - 语句级（有限制）
pool_mode = transaction

; 连接池大小
default_pool_size = 25
max_client_conn = 100
min_pool_size = 5

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
log_stats = 1
stats_period = 60

; 管理接口
admin_users = admin
stats_users = stats
```

```
; userlist.txt
"admin" "SCRAM-SHA-256$..."
"appuser" "SCRAM-SHA-256$..."
```

### 连接池模式对比

| 模式 | 说明 | 适用场景 | 限制 |
|------|------|----------|------|
| **Session** | 会话级连接复用 | 需要会话变量、临时表 | 效率最低 |
| **Transaction** | 事务级连接复用 | 大多数应用 | 不支持 SET、PREPARE |
| **Statement** | 语句级连接复用 | 简单查询 | 不支持事务 |

### 监控与管理

```sql
-- 连接到 PgBouncer 管理数据库
psql -p 6432 pgbouncer -U admin

-- 查看客户端连接
SHOW CLIENTS;

-- 查看服务端连接
SHOW SERVERS;

-- 查看数据库配置
SHOW DATABASES;

-- 查看连接池状态
SHOW POOLS;

-- 输出示例：
--  database | cl_active | cl_waiting | sv_active | sv_idle | sv_used | sv_tested | sv_login | maxwait
-- ----------+-----------+------------+-----------+---------+---------+-----------+----------+---------
--  myapp    |        10 |          2 |         5 |       8 |       5 |         0 |        0 |       0

-- 查看统计信息
SHOW STATS;

-- 重新加载配置
RELOAD;

-- 暂停连接池
PAUSE myapp;

-- 恢复连接池
RESUME myapp;

-- 关闭所有连接
SHUTDOWN;
```

### 应用连接配置

```python
# Python (psycopg2)
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=6432,      # PgBouncer 端口
    database="myapp",
    user="appuser",
    password="secret"
)

# 推荐使用连接池
from psycopg2 import pool

connection_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    port=6432,
    database="myapp",
    user="appuser",
    password="secret"
)
```

```javascript
// Node.js (pg)
const { Pool } = require('pg');

const pool = new Pool({
  host: 'localhost',
  port: 6432,        // PgBouncer 端口
  database: 'myapp',
  user: 'appuser',
  password: 'secret',
  max: 10,           // 客户端连接数
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});
```

---

## 向量搜索扩展 pgvector

pgvector 是 PostgreSQL 的向量相似度搜索扩展，支持 AI/ML 应用：

### 安装与使用

```sql
-- 安装扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建带向量列的表
CREATE TABLE documents (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536)  -- OpenAI embeddings 维度
);

-- 创建向量索引
CREATE INDEX idx_documents_embedding ON documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 或使用 HNSW 索引（更快，更精确）
CREATE INDEX idx_documents_embedding_hnsw ON documents 
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

```sql
-- 插入向量数据
INSERT INTO documents (content, embedding) VALUES
('PostgreSQL 是一个强大的开源数据库', '[0.1, 0.2, ...]'::vector),
('向量搜索是 AI 的核心技术', '[0.3, 0.1, ...]'::vector);

-- 相似度搜索
SELECT 
    content,
    1 - (embedding <=> '[0.15, 0.25, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.15, 0.25, ...]'::vector
LIMIT 5;

-- 余弦相似度 (<=>)
-- L2 距离 (<->)
-- 内积 (<#>)
```

---

## 常见问题与注意事项

### 常见陷阱

1. **SERIAL vs IDENTITY**
```sql
-- ❌ 旧方式（不推荐）
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- ✅ 新方式（PostgreSQL 10+）
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(100)
);

-- IDENTITY 的优势：
-- - 标准 SQL 语法
-- - 更好的权限控制
-- - 支持 RESTART
ALTER TABLE users ALTER COLUMN id RESTART WITH 1000;
```

2. **COUNT(*) 性能**
```sql
-- ❌ 全表扫描，大表很慢
SELECT COUNT(*) FROM large_table;

-- ✅ 使用估计值
SELECT reltuples::bigint FROM pg_class WHERE relname = 'large_table';

-- ✅ 使用索引
SELECT COUNT(*) FROM large_table WHERE id > 0;  -- 如果 id 有索引
```

3. **大表更新**
```sql
-- ❌ 一次性更新大量数据
UPDATE huge_table SET status = 'inactive' WHERE date < '2020-01-01';

-- ✅ 分批更新
DO $$
DECLARE
    updated INTEGER;
BEGIN
    LOOP
        UPDATE huge_table 
        SET status = 'inactive' 
        WHERE date < '2020-01-01' 
            AND status != 'inactive'
            AND ctid IN (
                SELECT ctid FROM huge_table 
                WHERE date < '2020-01-01' AND status != 'inactive'
                LIMIT 10000
            );
        
        GET DIAGNOSTICS updated = ROW_COUNT;
        EXIT WHEN updated = 0;
        COMMIT;
    END LOOP;
END $$;
```

4. **索引未使用**
```sql
-- 常见原因：
-- 1. 类型不匹配
WHERE user_id = '123'  -- user_id 是 INTEGER，索引不使用

-- 2. 函数包装
WHERE LOWER(name) = 'alice'  -- 需要函数索引
CREATE INDEX idx_users_name_lower ON users (LOWER(name));

-- 3. LIKE 以通配符开头
WHERE name LIKE '%son'  -- 索引不使用
WHERE name LIKE 'son%'  -- 索引使用

-- 4. OR 条件
WHERE user_id = 1 OR status = 'active'  -- 可能不使用
-- 改写为 UNION ALL
SELECT * FROM users WHERE user_id = 1
UNION ALL
SELECT * FROM users WHERE status = 'active' AND user_id != 1;
```

### 安全最佳实践

```sql
-- 使用参数化查询防止 SQL 注入
-- ❌ 不安全
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

-- ✅ 安全
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))

-- 行级安全策略
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY orders_policy ON orders
    USING (customer_id = current_user_id());

-- 加密函数
CREATE EXTENSION IF NOT EXISTS pgcrypto;

SELECT encrypt('sensitive data', 'secret_key', 'aes');
SELECT decrypt(encrypted_data, 'secret_key', 'aes');
```

---

## 参考资料

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/current/index.html)
- [PostgreSQL 17 Release Notes](https://www.postgresql.org/about/news/postgresql-17-released-2936/)
- [pgvector 扩展](https://github.com/pgvector/pgvector)
- [PgBouncer 文档](https://www.pgbouncer.org/)
- [PostgreSQL Wiki](https://wiki.postgresql.org/)
