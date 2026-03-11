# MyBatis 持久层框架

> MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射，通过 XML 或注解将 SQL 语句与程序代码分离，实现了灵活的 SQL 控制。

## 持久层

### 分层架构

```
典型的三层架构：
│
├── 表现层（Presentation Layer）
│   └── Controller：接收请求，返回响应
│
├── 业务层（Business Layer）
│   └── Service：业务逻辑处理
│
└── 持久层（Persistence Layer）
    └── DAO/Repository：数据访问，与数据库交互

持久层的职责：
- 将数据保存到数据库（增、改）
- 从数据库查询数据（查）
- 从数据库删除数据（删）
```

### 为什么需要 MyBatis？

```java
// ========== 原生 JDBC：繁琐且容易出错 ==========
public User findById(Long id) {
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    
    try {
        // 1. 获取连接
        conn = DriverManager.getConnection(url, username, password);
        
        // 2. 编写 SQL（字符串拼接，容易出错）
        String sql = "SELECT id, name, age FROM user WHERE id = ?";
        
        // 3. 创建语句对象
        stmt = conn.prepareStatement(sql);
        stmt.setLong(1, id);
        
        // 4. 执行查询
        rs = stmt.executeQuery();
        
        // 5. 手动映射结果集
        if (rs.next()) {
            User user = new User();
            user.setId(rs.getLong("id"));
            user.setName(rs.getString("name"));
            user.setAge(rs.getInt("age"));
            return user;
        }
        return null;
    } catch (SQLException e) {
        throw new RuntimeException(e);
    } finally {
        // 6. 手动关闭资源（繁琐）
        if (rs != null) try { rs.close(); } catch (SQLException e) {}
        if (stmt != null) try { stmt.close(); } catch (SQLException e) {}
        if (conn != null) try { conn.close(); } catch (SQLException e) {}
    }
}

// ========== MyBatis：简洁且灵活 ==========
@Mapper
public interface UserMapper {
    @Select("SELECT * FROM user WHERE id = #{id}")
    User findById(Long id);
}

// 使用
User user = userMapper.findById(1L);  // 一行代码搞定！

// MyBatis 的优势：
// 1. SQL 与代码分离，易于维护
// 2. 自动映射结果集，减少样板代码
// 3. 支持动态 SQL，灵活拼接
// 4. 连接池管理，性能优化
```

### MyBatis vs 其他 ORM 框架

| 特性 | MyBatis | Hibernate/JPA |
|------|---------|---------------|
| SQL 控制 | 完全控制，手写 SQL | 自动生成，HQL |
| 学习曲线 | 较平缓 | 较陡峭 |
| 灵活性 | 高，适合复杂 SQL | 中，适合标准 CRUD |
| 性能调优 | SQL 级别调优 | 框架级别配置 |
| 适用场景 | 复杂查询、高性能要求 | 标准 CRUD、快速开发 |

---

## 快速开始

### 添加依赖

```xml
<!-- pom.xml -->

<!-- MyBatis Spring Boot Starter -->
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>3.0.3</version>
</dependency>

<!-- MySQL 驱动 -->
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-j</artifactId>
    <scope>runtime</scope>
</dependency>
```

### 配置文件

```yaml
# application.yml

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai&allowPublicKeyRetrieval=true
    username: root
    password: root
    driver-class-name: com.mysql.cj.jdbc.Driver
    # HikariCP 连接池配置
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5

mybatis:
  # Mapper XML 文件位置
  mapper-locations: classpath:mapper/*.xml
  
  # 实体类别名包路径
  type-aliases-package: com.example.entity
  
  # 配置
  configuration:
    # 开启驼峰命名转换：user_name → userName
    map-underscore-to-camel-case: true
    
    # 打印 SQL 日志（开发环境）
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
    
    # 开启二级缓存
    cache-enabled: true
```

### 实体类

```java
package com.example.entity;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 用户实体类
 * 对应数据库表：user
 */
@Data
public class User {
    /** 主键 ID */
    private Long id;
    
    /** 用户名 */
    private String username;
    
    /** 密码 */
    private String password;
    
    /** 邮箱 */
    private String email;
    
    /** 年龄 */
    private Integer age;
    
    /** 创建时间（对应数据库字段 create_time）*/
    private LocalDateTime createTime;
    
    /** 更新时间（对应数据库字段 update_time）*/
    private LocalDateTime updateTime;
}
```

### Mapper 接口

```java
package com.example.mapper;

import com.example.entity.User;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;

/**
 * 用户数据访问接口
 */
@Mapper  // 标记为 MyBatis Mapper，Spring 会自动创建实现类
public interface UserMapper {
    
    /** 查询所有用户 */
    List<User> findAll();
    
    /** 根据 ID 查询用户 */
    User findById(Long id);
    
    /** 根据用户名查询 */
    User findByUsername(String username);
    
    /** 插入用户 */
    int insert(User user);
    
    /** 更新用户 */
    int update(User user);
    
    /** 删除用户 */
    int deleteById(Long id);
}
```

### Mapper XML

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">

<!-- namespace 必须与 Mapper 接口全限定名一致 -->
<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- ========== 结果映射 ========== -->
    <!-- 将数据库列名映射到 Java 对象属性 -->
    <resultMap id="BaseResultMap" type="com.example.entity.User">
        <!-- id 标签用于主键，property 是属性名，column 是列名 -->
        <id column="id" property="id"/>
        <!-- result 标签用于普通字段 -->
        <result column="username" property="username"/>
        <result column="password" property="password"/>
        <result column="email" property="email"/>
        <result column="age" property="age"/>
        <!-- 如果开启了驼峰转换，这两行可以省略 -->
        <result column="create_time" property="createTime"/>
        <result column="update_time" property="updateTime"/>
    </resultMap>
    
    <!-- ========== 通用字段片段 ========== -->
    <!-- 可复用的 SQL 片段 -->
    <sql id="Base_Column_List">
        id, username, password, email, age, create_time, update_time
    </sql>
    
    <!-- ========== 查询所有 ========== -->
    <select id="findAll" resultMap="BaseResultMap">
        SELECT <include refid="Base_Column_List"/> FROM user
    </select>
    
    <!-- ========== 根据 ID 查询 ========== -->
    <!-- id 必须与 Mapper 接口方法名一致 -->
    <select id="findById" resultMap="BaseResultMap">
        SELECT <include refid="Base_Column_List"/> 
        FROM user 
        WHERE id = #{id}
        <!-- #{id} 是参数占位符，会自动转换为 PreparedStatement 的 ? -->
    </select>
    
    <!-- ========== 根据用户名查询 ========== -->
    <select id="findByUsername" resultMap="BaseResultMap">
        SELECT <include refid="Base_Column_List"/> 
        FROM user 
        WHERE username = #{username}
    </select>
    
    <!-- ========== 插入 ========== -->
    <!-- useGeneratedKeys：使用数据库自动生成的主键 -->
    <!-- keyProperty：将生成的主键值赋给对象的哪个属性 -->
    <insert id="insert" parameterType="com.example.entity.User" 
            useGeneratedKeys="true" keyProperty="id">
        INSERT INTO user (username, password, email, age, create_time)
        VALUES (#{username}, #{password}, #{email}, #{age}, NOW())
    </insert>
    
    <!-- ========== 更新 ========== -->
    <update id="update" parameterType="com.example.entity.User">
        UPDATE user
        SET username = #{username},
            email = #{email},
            age = #{age},
            update_time = NOW()
        WHERE id = #{id}
    </update>
    
    <!-- ========== 删除 ========== -->
    <delete id="deleteById">
        DELETE FROM user WHERE id = #{id}
    </delete>
</mapper>
```

### 使用 Mapper

```java
@Service
@RequiredArgsConstructor
public class UserService {
    
    private final UserMapper userMapper;
    
    public List<User> findAll() {
        return userMapper.findAll();
    }
    
    public User findById(Long id) {
        return userMapper.findById(id);
    }
    
    public User create(User user) {
        userMapper.insert(user);
        // insert 后，user.getId() 已经是自动生成的主键值
        return user;
    }
    
    public void update(User user) {
        userMapper.update(user);
    }
    
    public void delete(Long id) {
        userMapper.deleteById(id);
    }
}
```

---

## 注解方式

MyBatis 也支持使用注解配置 SQL，适合简单的 SQL。

```java
@Mapper
public interface UserMapper {
    
    // ========== 查询 ==========
    
    // 查询所有
    @Select("SELECT * FROM user")
    List<User> findAll();
    
    // 根据 ID 查询
    @Select("SELECT * FROM user WHERE id = #{id}")
    User findById(Long id);
    
    // 根据条件查询
    @Select("SELECT * FROM user WHERE username = #{username} AND age > #{minAge}")
    List<User> findByCondition(@Param("username") String username, 
                                @Param("minAge") Integer minAge);
    
    // ========== 插入 ==========
    
    @Insert("INSERT INTO user(username, password, email, age, create_time) " +
            "VALUES(#{username}, #{password}, #{email}, #{age}, NOW())")
    @Options(useGeneratedKeys = true, keyProperty = "id")  // 返回自增主键
    int insert(User user);
    
    // ========== 更新 ==========
    
    @Update("UPDATE user SET username=#{username}, email=#{email}, " +
            "age=#{age}, update_time=NOW() WHERE id=#{id}")
    int update(User user);
    
    // ========== 删除 ==========
    
    @Delete("DELETE FROM user WHERE id = #{id}")
    int deleteById(Long id);
    
    // ========== 结果映射 ==========
    
    // 当字段名和属性名不一致时，使用 @Results
    @Results({
        @Result(column = "id", property = "id"),
        @Result(column = "create_time", property = "createTime"),
        @Result(column = "update_time", property = "updateTime")
    })
    @Select("SELECT * FROM user WHERE username = #{username}")
    User findByUsername(String username);
}
```

### XML vs 注解选择

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 简单 SQL（单表 CRUD） | 注解 | 简洁直观 |
| 复杂 SQL（多表关联） | XML | 可读性好 |
| 动态 SQL | XML | 更灵活 |
| SQL 可维护性要求高 | XML | 集中管理 |

---

## 动态 SQL

动态 SQL 是 MyBatis 的核心特性，可以根据条件动态拼接 SQL。

### if 标签

```xml
<!-- if 标签：条件判断 -->
<select id="search" resultMap="BaseResultMap">
    SELECT * FROM user
    <where>
        <!-- test 中写 OGNL 表达式 -->
        <!-- 如果 username 不为 null 且不为空字符串，添加条件 -->
        <if test="username != null and username != ''">
            AND username LIKE CONCAT('%', #{username}, '%')
        </if>
        <if test="email != null and email != ''">
            AND email = #{email}
        </if>
        <if test="minAge != null">
            AND age >= #{minAge}
        </if>
        <if test="maxAge != null">
            AND age &lt;= #{maxAge}
        </if>
    </where>
</select>

<!-- 注意：
     1. <where> 标签会自动处理 AND/OR 前缀
     2. 在 XML 中，< 和 > 需要转义：&lt; 和 &gt;
     3. 或者使用 CDATA：<![CDATA[ age <= #{maxAge} ]]>
-->
```

### choose-when-otherwise 标签

```xml
<!-- choose-when-otherwise：类似 Java 的 switch-case -->
<!-- 只会匹配第一个满足条件的分支 -->
<select id="searchByCondition" resultMap="BaseResultMap">
    SELECT * FROM user
    <where>
        <choose>
            <when test="username != null">
                AND username = #{username}
            </when>
            <when test="email != null">
                AND email = #{email}
            </when>
            <otherwise>
                <!-- 所有条件都不满足时执行 -->
                AND age >= 18
            </otherwise>
        </choose>
    </where>
</select>
```

### where 和 set 标签

```xml
<!-- where 标签：自动处理 WHERE 关键字和 AND/OR 前缀 -->
<!-- 如果内部有内容，自动添加 WHERE -->
<!-- 自动去除开头的 AND 或 OR -->
<select id="search" resultMap="BaseResultMap">
    SELECT * FROM user
    <where>
        <if test="username != null">
            AND username = #{username}
        </if>
        <if test="email != null">
            AND email = #{email}
        </if>
    </where>
    <!-- 如果两个条件都为空，生成的 SQL：SELECT * FROM user -->
    <!-- 如果 username 不为空，生成的 SQL：SELECT * FROM user WHERE username = ? -->
</select>

<!-- set 标签：自动处理 SET 关键字和逗号 -->
<!-- 自动去除多余的逗号 -->
<update id="updateSelective">
    UPDATE user
    <set>
        <if test="username != null">
            username = #{username},
        </if>
        <if test="email != null">
            email = #{email},
        </if>
        <if test="age != null">
            age = #{age},
        </if>
        update_time = NOW()
    </set>
    WHERE id = #{id}
</update>
```

### foreach 标签

```xml
<!-- foreach：遍历集合或数组 -->

<!-- 批量查询 -->
<!-- collection：集合参数名，List 默认 list，数组默认 array -->
<!-- item：当前元素的变量名 -->
<!-- open/close：循环开始/结束的字符串 -->
<!-- separator：元素之间的分隔符 -->
<select id="findByIds" resultMap="BaseResultMap">
    SELECT * FROM user
    WHERE id IN
    <foreach collection="ids" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</select>
<!-- 生成：SELECT * FROM user WHERE id IN (1, 2, 3) -->

<!-- 批量插入 -->
<insert id="batchInsert">
    INSERT INTO user(username, password, email, create_time)
    VALUES
    <foreach collection="users" item="user" separator=",">
        (#{user.username}, #{user.password}, #{user.email}, NOW())
    </foreach>
</insert>
<!-- 生成：INSERT INTO user(...) VALUES (...), (...), (...) -->

<!-- 批量更新（MySQL） -->
<update id="batchUpdate">
    <foreach collection="users" item="user" separator=";">
        UPDATE user SET
            username = #{user.username},
            email = #{user.email}
        WHERE id = #{user.id}
    </foreach>
</update>
```

### foreach 属性说明

| 属性 | 说明 | 示例 |
|------|------|------|
| `collection` | 集合参数名 | List 默认 list，数组默认 array，或用 @Param 指定 |
| `item` | 当前元素的变量名 | `item="user"` 后用 `#{user.name}` |
| `index` | 当前元素的索引 | `index="i"` 后用 `#{i}` |
| `open` | 循环开始的字符串 | `open="("` |
| `close` | 循环结束的字符串 | `close=")"` |
| `separator` | 元素之间的分隔符 | `separator=","` |

### trim 标签

```xml
<!-- trim：自定义前缀后缀处理 -->
<!-- prefix：添加的前缀 -->
<!-- prefixOverrides：去除的前缀 -->
<!-- suffix：添加的后缀 -->
<!-- suffixOverrides：去除的后缀 -->

<select id="search" resultMap="BaseResultMap">
    SELECT * FROM user
    <trim prefix="WHERE" prefixOverrides="AND|OR">
        <if test="username != null">
            AND username = #{username}
        </if>
        <if test="email != null">
            AND email = #{email}
        </if>
    </trim>
</select>

<update id="updateSelective">
    UPDATE user
    <trim prefix="SET" suffixOverrides=",">
        <if test="username != null">
            username = #{username},
        </if>
        <if test="email != null">
            email = #{email},
        </if>
    </trim>
    WHERE id = #{id}
</update>
```

### sql 和 include 标签

```xml
<!-- sql：定义可重用的 SQL 片段 -->
<sql id="Base_Column_List">
    id, username, password, email, age, create_time, update_time
</sql>

<sql id="Search_Condition">
    <where>
        <if test="username != null and username != ''">
            AND username LIKE CONCAT('%', #{username}, '%')
        </if>
        <if test="email != null and email != ''">
            AND email = #{email}
        </if>
    </where>
</sql>

<!-- include：引用 SQL 片段 -->
<select id="search" resultMap="BaseResultMap">
    SELECT <include refid="Base_Column_List"/> FROM user
    <include refid="Search_Condition"/>
</select>
```

---

## 缓存机制

### 一级缓存（SqlSession 级别）

一级缓存默认开启，同一 SqlSession 中相同的查询会命中缓存。

```java
// 一级缓存示例
SqlSession session = sqlSessionFactory.openSession();
UserMapper mapper = session.getMapper(UserMapper.class);

// 第一次查询：查询数据库
User user1 = mapper.findById(1L);

// 第二次查询：命中缓存，不查数据库
User user2 = mapper.findById(1L);

System.out.println(user1 == user2);  // true，同一个对象

// 清除缓存
session.clearCache();

// 第三次查询：重新查询数据库
User user3 = mapper.findById(1L);

session.close();

// 一级缓存失效的情况：
// 1. 调用 session.clearCache()
// 2. 执行了增删改操作
// 3. session 关闭
```

### 二级缓存（Mapper 级别）

二级缓存跨 SqlSession 共享，需要手动开启。

```xml
<!-- mybatis-config.xml 或 application.yml -->
<settings>
    <setting name="cacheEnabled" value="true"/>
</settings>

<!-- Mapper XML 中开启二级缓存 -->
<mapper namespace="com.example.mapper.UserMapper">
    <!-- 简单开启 -->
    <cache/>
    
    <!-- 详细配置 -->
    <cache 
        eviction="LRU"           <!-- 缓存淘汰策略 -->
        flushInterval="60000"    <!-- 刷新间隔（毫秒） -->
        size="1024"              <!-- 缓存对象数量 -->
        readOnly="true"/>        <!-- 是否只读 -->
</mapper>
```

### 缓存淘汰策略

| 策略 | 说明 |
|------|------|
| `LRU` | 最近最少使用（默认） |
| `FIFO` | 先进先出 |
| `SOFT` | 软引用，内存不足时回收 |
| `WEAK` | 弱引用，GC 时回收 |

### 使用注解配置缓存

```java
@CacheNamespace  // 开启二级缓存
@Mapper
public interface UserMapper {
    @Select("SELECT * FROM user WHERE id = #{id}")
    User findById(Long id);
    
    @Options(useCache = true)  // 使用缓存
    @Select("SELECT * FROM user")
    List<User> findAll();
    
    @Options(flushCache = Options.FlushCachePolicy.TRUE)  // 清除缓存
    @Insert("INSERT INTO user...")
    int insert(User user);
}
```

---

## 关联查询

### 一对一关联

```java
// 用户实体
@Data
public class User {
    private Long id;
    private String username;
    private Profile profile;  // 用户详情（一对一）
}

// 用户详情实体
@Data
public class Profile {
    private Long id;
    private Long userId;
    private String nickname;
    private String avatar;
}
```

```xml
<!-- 嵌套结果映射（推荐）：一次 SQL 查询 -->
<resultMap id="UserWithProfileMap" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="username" property="username"/>
    <!-- 一对一关联：association -->
    <association property="profile" javaType="com.example.entity.Profile">
        <id column="profile_id" property="id"/>
        <result column="user_id" property="userId"/>
        <result column="nickname" property="nickname"/>
        <result column="avatar" property="avatar"/>
    </association>
</resultMap>

<select id="findUserWithProfile" resultMap="UserWithProfileMap">
    SELECT u.*, p.id as profile_id, p.user_id, p.nickname, p.avatar
    FROM user u
    LEFT JOIN profile p ON u.id = p.user_id
    WHERE u.id = #{id}
</select>

<!-- 嵌套查询：分开查询（会产生 N+1 问题，不推荐） -->
<resultMap id="UserWithProfileMap2" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="username" property="username"/>
    <!-- column：传给嵌套查询的参数 -->
    <!-- select：嵌套查询的语句 ID -->
    <association property="profile" 
                 column="id" 
                 select="com.example.mapper.ProfileMapper.findByUserId"/>
</resultMap>
```

### 一对多关联

```java
// 用户实体
@Data
public class User {
    private Long id;
    private String username;
    private List<Order> orders;  // 用户订单（一对多）
}

// 订单实体
@Data
public class Order {
    private Long id;
    private Long userId;
    private String orderNo;
    private BigDecimal amount;
}
```

```xml
<resultMap id="UserWithOrdersMap" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="username" property="username"/>
    <!-- 一对多关联：collection -->
    <!-- ofType：集合元素的类型 -->
    <collection property="orders" ofType="com.example.entity.Order">
        <id column="order_id" property="id"/>
        <result column="user_id" property="userId"/>
        <result column="order_no" property="orderNo"/>
        <result column="amount" property="amount"/>
    </collection>
</resultMap>

<select id="findUserWithOrders" resultMap="UserWithOrdersMap">
    SELECT u.*, o.id as order_id, o.user_id, o.order_no, o.amount
    FROM user u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.id = #{id}
</select>
```

---

## 插件机制

### 分页插件（PageHelper）

```xml
<!-- 添加依赖 -->
<dependency>
    <groupId>com.github.pagehelper</groupId>
    <artifactId>pagehelper-spring-boot-starter</artifactId>
    <version>2.0.0</version>
</dependency>
```

```java
@Service
public class UserService {
    
    @Autowired
    private UserMapper userMapper;
    
    public PageInfo<User> list(int pageNum, int pageSize) {
        // 设置分页参数（紧跟查询语句）
        // 只对接下来的一条查询语句生效
        PageHelper.startPage(pageNum, pageSize);
        
        // 执行查询
        List<User> users = userMapper.findAll();
        
        // 返回分页信息
        return new PageInfo<>(users);
    }
}

// Controller
@GetMapping
public Result<PageInfo<User>> list(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "10") int size) {
    return Result.success(userService.list(page, size));
}

// PageInfo 包含的信息：
// {
//   "list": [...],         // 当前页数据
//   "total": 100,          // 总记录数
//   "pages": 10,           // 总页数
//   "pageNum": 1,          // 当前页
//   "pageSize": 10,        // 每页大小
//   "hasNextPage": true,   // 是否有下一页
//   "hasPreviousPage": false  // 是否有上一页
// }
```

---

## 最佳实践

### 命名规范

```java
// Mapper 接口命名
UserMapper.java      // 实体 + Mapper

// XML 文件命名
UserMapper.xml       // 与接口同名

// 方法命名规范
findById             // 单条查询（by + 条件）
findAll / list       // 列表查询
findByNameAndAge     // 多条件查询
insert               // 插入
insertBatch          // 批量插入
update               // 更新
updateSelective      // 选择性更新（只更新非空字段）
delete               // 删除
deleteById           // 按 ID 删除
count                // 计数
exists               // 判断存在
```

### 参数传递

```java
// 单个参数：直接使用
User findById(Long id);
// XML: #{id}

// 多个参数：使用 @Param 注解
List<User> search(@Param("username") String username, 
                  @Param("age") Integer age);
// XML: #{username}, #{age}

// 对象参数：直接使用属性名
int insert(User user);
// XML: #{username}, #{password}, #{email}

// Map 参数：使用 key
List<User> search(Map<String, Object> params);
// XML: #{username}

// 参数传递方式：
// #{param}：预编译，安全，防止 SQL 注入（推荐）
// ${param}：字符串替换，不安全，可能 SQL 注入（仅用于动态表名、列名）
```

### 批量操作优化

```xml
<!-- 批量插入方式一：foreach（适合小批量） -->
<insert id="batchInsert">
    INSERT INTO user(username, password, email)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.username}, #{user.password}, #{user.email})
    </foreach>
</insert>

<!-- 批量插入方式二：多 VALUES（MySQL 推荐） -->
<insert id="batchInsert" useGeneratedKeys="true" keyProperty="id">
    INSERT INTO user(username, password, email) VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.username}, #{user.password}, #{user.email})
    </foreach>
</insert>
```

```java
// Java 批量插入优化
@Transactional
public void batchInsert(List<User> users) {
    // 分批插入，避免 SQL 过长
    int batchSize = 1000;
    for (int i = 0; i < users.size(); i += batchSize) {
        List<User> batch = users.subList(i, Math.min(i + batchSize, users.size()));
        userMapper.batchInsert(batch);
    }
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **Mapper 接口** | 定义数据访问方法 |
| **Mapper XML** | 配置 SQL 语句和映射规则 |
| **动态 SQL** | 根据条件动态拼接 SQL |
| **结果映射** | 将数据库列映射到对象属性 |
| **缓存** | 一级缓存（SqlSession 级别）、二级缓存（Mapper 级别） |

### MyBatis 核心要点

1. **SQL 与代码分离**：XML 或注解配置 SQL
2. **灵活的结果映射**：支持复杂关联查询
3. **强大的动态 SQL**：if、choose、foreach 等标签
4. **插件机制**：分页、性能监控等扩展
5. **缓存支持**：提升查询性能

### 常见问题

1. **字段名与属性名不一致**：使用 resultMap 或开启驼峰转换
2. **SQL 注入**：使用 `#{}` 而不是 `${}`
3. **N+1 问题**：使用嵌套结果映射而不是嵌套查询
4. **批量操作性能**：使用 foreach 批量插入，分批处理大数据量