# MyBatis 持久层框架

> MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射，通过 XML 或注解将 SQL 语句与程序代码分离，实现了灵活的 SQL 控制。

## 框架概述

MyBatis 前身是 iBATIS，是一个轻量级的 ORM（对象关系映射）框架。与 Hibernate 等全自动 ORM 框架不同，MyBatis 采用半自动映射策略，开发者可以精确控制 SQL 语句，适合对 SQL 有较高要求的项目。

### 🎯 核心特性

| 特性 | 说明 |
|------|------|
| **SQL 与代码分离** | 通过 XML 或注解配置 SQL |
| **灵活的映射** | 支持复杂的结果集映射 |
| **动态 SQL** | 条件拼接 SQL，避免字符串拼接 |
| **插件机制** | 支持自定义插件扩展功能 |
| **缓存支持** | 一级缓存、二级缓存 |

### MyBatis vs 其他框架

| 特性 | MyBatis | Hibernate | JPA |
|------|---------|-----------|-----|
| SQL 控制 | 完全控制 | 自动生成 | 自动生成 |
| 学习曲线 | 中等 | 较陡 | 较陡 |
| 灵活性 | 高 | 中 | 中 |
| 适用场景 | 复杂 SQL | 标准 CRUD | 标准 CRUD |

---

## 快速开始

### 添加依赖

```xml
<!-- MyBatis -->
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>3.0.3</version>
</dependency>

<!-- MySQL 驱动 -->
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-j</artifactId>
</dependency>
```

### 配置文件

```yaml
# application.yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai
    username: root
    password: root
    driver-class-name: com.mysql.cj.jdbc.Driver

mybatis:
  mapper-locations: classpath:mapper/*.xml  # Mapper XML 文件位置
  type-aliases-package: com.example.entity  # 实体类包路径
  configuration:
    map-underscore-to-camel-case: true      # 下划线转驼峰
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl  # SQL 日志
```

### 实体类

```java
@Data
public class User {
    private Long id;
    private String username;
    private String password;
    private String email;
    private Integer age;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
```

### Mapper 接口

```java
@Mapper
public interface UserMapper {
    
    // 查询所有用户
    List<User> findAll();
    
    // 根据 ID 查询
    User findById(Long id);
    
    // 插入用户
    int insert(User user);
    
    // 更新用户
    int update(User user);
    
    // 删除用户
    int deleteById(Long id);
}
```

### Mapper XML

```xml
<!-- resources/mapper/UserMapper.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">

<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- 结果映射 -->
    <resultMap id="BaseResultMap" type="com.example.entity.User">
        <id column="id" property="id"/>
        <result column="username" property="username"/>
        <result column="password" property="password"/>
        <result column="email" property="email"/>
        <result column="age" property="age"/>
        <result column="create_time" property="createTime"/>
        <result column="update_time" property="updateTime"/>
    </resultMap>
    
    <!-- 通用字段 -->
    <sql id="Base_Column_List">
        id, username, password, email, age, create_time, update_time
    </sql>
    
    <!-- 查询所有 -->
    <select id="findAll" resultMap="BaseResultMap">
        SELECT <include refid="Base_Column_List"/> FROM user
    </select>
    
    <!-- 根据 ID 查询 -->
    <select id="findById" resultMap="BaseResultMap">
        SELECT <include refid="Base_Column_List"/> 
        FROM user 
        WHERE id = #{id}
    </select>
    
    <!-- 插入 -->
    <insert id="insert" parameterType="com.example.entity.User" 
            useGeneratedKeys="true" keyProperty="id">
        INSERT INTO user (username, password, email, age, create_time)
        VALUES (#{username}, #{password}, #{email}, #{age}, NOW())
    </insert>
    
    <!-- 更新 -->
    <update id="update" parameterType="com.example.entity.User">
        UPDATE user
        SET username = #{username},
            email = #{email},
            age = #{age},
            update_time = NOW()
        WHERE id = #{id}
    </update>
    
    <!-- 删除 -->
    <delete id="deleteById">
        DELETE FROM user WHERE id = #{id}
    </delete>
</mapper>
```

---

## 注解方式

MyBatis 也支持使用注解配置 SQL，适合简单的 SQL 语句。

```java
@Mapper
public interface UserMapper {
    
    // 查询所有
    @Select("SELECT * FROM user")
    List<User> findAll();
    
    // 根据 ID 查询
    @Select("SELECT * FROM user WHERE id = #{id}")
    User findById(Long id);
    
    // 插入
    @Insert("INSERT INTO user(username, password, email, age, create_time) " +
            "VALUES(#{username}, #{password}, #{email}, #{age}, NOW())")
    @Options(useGeneratedKeys = true, keyProperty = "id")  // 返回自增主键
    int insert(User user);
    
    // 更新
    @Update("UPDATE user SET username=#{username}, email=#{email}, " +
            "age=#{age}, update_time=NOW() WHERE id=#{id}")
    int update(User user);
    
    // 删除
    @Delete("DELETE FROM user WHERE id = #{id}")
    int deleteById(Long id);
    
    // 结果映射
    @Results({
        @Result(column = "create_time", property = "createTime"),
        @Result(column = "update_time", property = "updateTime")
    })
    @Select("SELECT * FROM user WHERE username = #{username}")
    User findByUsername(String username);
}
```

### 📌 XML vs 注解选择

| 场景 | 推荐方式 |
|------|----------|
| 简单 SQL | 注解 |
| 复杂 SQL | XML |
| 动态 SQL | XML |
| SQL 可维护性要求高 | XML |

---

## 动态 SQL

动态 SQL 是 MyBatis 的核心特性，可以根据条件动态拼接 SQL 语句。

### if 标签

```xml
<select id="search" resultMap="BaseResultMap">
    SELECT * FROM user
    <where>
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
```

### choose-when-otherwise 标签

```xml
<!-- 类似 Java 的 switch-case -->
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
                AND age >= 18
            </otherwise>
        </choose>
    </where>
</select>
```

### where 和 set 标签

```xml
<!-- where 标签自动处理 AND/OR 前缀 -->
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
</select>

<!-- set 标签自动处理逗号 -->
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
<!-- 批量查询 -->
<select id="findByIds" resultMap="BaseResultMap">
    SELECT * FROM user
    WHERE id IN
    <foreach collection="ids" item="id" open="(" separator="," close=")">
        #{id}
    </foreach>
</select>

<!-- 批量插入 -->
<insert id="batchInsert">
    INSERT INTO user(username, password, email, create_time)
    VALUES
    <foreach collection="users" item="user" separator=",">
        (#{user.username}, #{user.password}, #{user.email}, NOW())
    </foreach>
</insert>

<!-- 批量更新 -->
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

| 属性 | 说明 |
|------|------|
| `collection` | 集合参数名，List 默认 list，数组默认 array |
| `item` | 循环中的当前元素变量名 |
| `index` | 当前元素索引 |
| `open` | 循环开始的字符串 |
| `close` | 循环结束的字符串 |
| `separator` | 元素之间的分隔符 |

### trim 标签

```xml
<!-- 自定义前缀后缀处理 -->
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
<!-- 定义可重用的 SQL 片段 -->
<sql id="Base_Column_List">
    id, username, password, email, age, create_time, update_time
</sql>

<sql id="Search_Condition">
    <where>
        <if test="username != null">
            AND username LIKE CONCAT('%', #{username}, '%')
        </if>
        <if test="email != null">
            AND email = #{email}
        </if>
    </where>
</sql>

<!-- 引用 SQL 片段 -->
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

User user1 = mapper.findById(1L);  // 查询数据库
User user2 = mapper.findById(1L);  // 命中缓存

System.out.println(user1 == user2);  // true，同一对象

session.close();
```

### 二级缓存（Mapper 级别）

二级缓存跨 SqlSession 共享，需要手动开启。

```xml
<!-- mybatis-config.xml -->
<settings>
    <setting name="cacheEnabled" value="true"/>
</settings>

<!-- Mapper XML -->
<mapper namespace="com.example.mapper.UserMapper">
    <!-- 开启二级缓存 -->
    <cache/>
    
    <!-- 或配置缓存属性 -->
    <cache 
        eviction="LRU"           // 缓存淘汰策略
        flushInterval="60000"    // 刷新间隔（毫秒）
        size="1024"              // 缓存对象数量
        readOnly="true"/>        // 是否只读
</mapper>
```

### 缓存淘汰策略

| 策略 | 说明 |
|------|------|
| `LRU` | 最近最少使用（默认） |
| `FIFO` | 先进先出 |
| `SOFT` | 软引用 |
| `WEAK` | 弱引用 |

### 使用注解配置缓存

```java
@CacheNamespace  // 开启二级缓存
@Mapper
public interface UserMapper {
    @Select("SELECT * FROM user WHERE id = #{id}")
    User findById(Long id);
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
<!-- 嵌套结果映射 -->
<resultMap id="UserWithProfileMap" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="username" property="username"/>
    <!-- 一对一关联 -->
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

<!-- 嵌套查询（会产生 N+1 问题） -->
<resultMap id="UserWithProfileMap2" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="username" property="username"/>
    <association property="profile" 
                 column="id" 
                 select="com.example.mapper.ProfileMapper.findByUserId"/>
</resultMap>

<select id="findById" resultMap="UserWithProfileMap2">
    SELECT id, username FROM user WHERE id = #{id}
</select>
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
    <!-- 一对多关联 -->
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
<dependency>
    <groupId>com.github.pagehelper</groupId>
    <artifactId>pagehelper-spring-boot-starter</artifactId>
    <version>2.0.0</version>
</dependency>
```

```java
@Service
public class UserService {
    
    public PageInfo<User> list(int pageNum, int pageSize) {
        // 设置分页参数
        PageHelper.startPage(pageNum, pageSize);
        
        // 执行查询
        List<User> users = userMapper.findAll();
        
        // 返回分页信息
        return new PageInfo<>(users);
    }
}
```

### 自定义插件

```java
@Intercepts({
    @Signature(type = StatementHandler.class, method = "prepare", 
               args = {Connection.class, Integer.class})
})
@Component
public class SqlLogPlugin implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        long start = System.currentTimeMillis();
        
        Object result = invocation.proceed();
        
        long elapsed = System.currentTimeMillis() - start;
        StatementHandler handler = (StatementHandler) invocation.getTarget();
        String sql = handler.getBoundSql().getSql();
        
        log.info("SQL: {}, 耗时: {}ms", sql, elapsed);
        
        return result;
    }
}
```

---

## 最佳实践

### 1. 命名规范

```java
// Mapper 接口命名
UserMapper.java      // 实体 + Mapper

// XML 文件命名
UserMapper.xml       // 与接口同名

// 方法命名
findById             // 单条查询
findAll / list       // 列表查询
insert               // 插入
update               // 更新
delete               // 删除
count                // 计数
```

### 2. 参数传递

```java
// 单个参数：直接使用
User findById(Long id);    // #{id}

// 多个参数：使用 @Param 注解
List<User> search(@Param("username") String username, 
                  @Param("age") Integer age);  // #{username}, #{age}

// 对象参数：直接使用属性名
int insert(User user);     // #{username}, #{password}

// Map 参数：使用 key
List<User> search(Map<String, Object> params);  // #{username}
```

### 3. 批量操作优化

```xml
<!-- 批量插入：使用 foreach -->
<insert id="batchInsert">
    INSERT INTO user(username, password, email)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.username}, #{user.password}, #{user.email})
    </foreach>
</insert>
```

```java
// 批量插入优化：使用 BatchExecutor
sqlSession.insert("com.example.mapper.UserMapper.batchInsert", userList);
sqlSession.commit();
```

---

## 参考资料

- [MyBatis 官方文档](https://mybatis.org/mybatis-3/)
- [MyBatis-Spring-Boot](https://mybatis.org/spring-boot-starter/mybatis-spring-boot-autoconfigure/)
- [PageHelper 分页插件](https://pagehelper.github.io/)
