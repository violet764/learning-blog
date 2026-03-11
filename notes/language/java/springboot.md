# Spring Boot 框架

> Spring Boot 是 Spring 生态的快速开发框架，通过**约定优于配置**的理念，大幅简化了 Spring 应用的初始搭建和开发过程。

## Spring Boot 

### 传统 Spring 项目的痛点

```xml
<!-- 传统 Spring 项目需要大量配置 -->

<!-- 1. web.xml 配置 -->
<web-app>
    <servlet>
        <servlet-name>dispatcher</servlet-name>
        <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
        <init-param>
            <param-name>contextConfigLocation</param-name>
            <param-value>/WEB-INF/dispatcher-servlet.xml</param-value>
        </init-param>
    </servlet>
</web-app>

<!-- 2. applicationContext.xml 配置 -->
<beans>
    <!-- 数据源配置 -->
    <bean id="dataSource" class="org.apache.commons.dbcp.BasicDataSource">
        <property name="url" value="jdbc:mysql://localhost:3306/mydb"/>
        <property name="username" value="root"/>
        <property name="password" value="root"/>
    </bean>
    
    <!-- 事务管理器 -->
    <bean id="transactionManager" class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
        <property name="dataSource" ref="dataSource"/>
    </bean>
    
    <!-- 视图解析器 -->
    <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="prefix" value="/WEB-INF/views/"/>
        <property name="suffix" value=".jsp"/>
    </bean>
    
    <!-- ... 还有更多配置 -->
</beans>

<!-- 3. 还需要单独安装和配置 Tomcat -->
<!-- 4. 还需要手动管理依赖版本 -->
```

### Spring Boot 的解决方案

```java
// 一个文件就能启动一个 Web 应用！
@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

// 配置文件（可选）
// application.yml
server:
  port: 8080
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: root

// Spring Boot 的优势：
// 1. 无需 XML 配置
// 2. 内置 Tomcat，无需单独安装
// 3. 自动配置，开箱即用
// 4. 统一管理依赖版本
// 5. 提供生产级监控
```

### 约定优于配置

**约定优于配置**是 Spring Boot 的核心理念：

| 方面 | 传统 Spring | Spring Boot |
|------|-------------|-------------|
| 配置文件 | 大量 XML | 少量 YAML/Properties |
| 依赖管理 | 手动指定版本 | 起步依赖自动管理 |
| 服务器 | 安装 Tomcat | 内嵌 Tomcat |
| 日志框架 | 手动配置 | 默认 Logback |
| 数据源 | 手动配置 | 自动配置 HikariCP |

---

## 框架概述

Spring Boot 由 Pivotal 团队开发，旨在简化 Spring 应用的开发。

### 核心特性

| 特性 | 说明 | 好处 |
|------|------|------|
| **自动配置** | 根据依赖自动配置 Spring 应用 | 减少配置工作 |
| **起步依赖** | 一个依赖包含所需全部组件 | 简化依赖管理 |
| **内嵌服务器** | 内置 Tomcat、Jetty、Undertow | 无需部署 WAR |
| **生产就绪** | 提供健康检查、指标监控等功能 | 快速上线 |
| **零配置** | 无需 XML 配置文件 | 开发更简洁 |

---

## 快速开始

### 创建项目

**方式一：Spring Initializr（网站）**
- 访问 https://start.spring.io/
- 选择 Java 版本、依赖
- 下载项目压缩包

**方式二：IDE 创建**
- IntelliJ IDEA: File → New → Project → Spring Initializr

**方式三：命令行**
```bash
curl https://start.spring.io/starter.zip -d type=maven-project -d language=java -d bootVersion=3.2.0 -d baseDir=myproject -d groupId=com.example -d artifactId=demo -d name=demo -d packageName=com.example.demo -d javaVersion=17 -o demo.zip
```

### 项目结构

```
my-springboot-app/
├── src/
│   ├── main/
│   │   ├── java/com/example/demo/
│   │   │   ├── DemoApplication.java    # 启动类（核心）
│   │   │   ├── controller/             # 控制器
│   │   │   ├── service/                # 业务层
│   │   │   ├── repository/             # 数据访问层
│   │   │   └── entity/                 # 实体类
│   │   └── resources/
│   │       ├── application.yml         # 配置文件
│   │       ├── static/                 # 静态资源（CSS/JS/图片）
│   │       └── templates/              # 模板文件（Thymeleaf）
│   └── test/                           # 测试代码
├── pom.xml                             # Maven 配置
└── mvnw                                # Maven Wrapper（无需安装 Maven）
```

### 主启动类

```java
// 主启动类：Spring Boot 应用的入口
@SpringBootApplication  // 核心注解（下面详解）
public class DemoApplication {
    public static void main(String[] args) {
        // 启动 Spring Boot 应用
        SpringApplication.run(DemoApplication.class, args);
    }
}

// @SpringBootApplication 是一个组合注解，等价于：
@SpringBootConfiguration    // 标记为配置类（等同于 @Configuration）
@EnableAutoConfiguration    // 启用自动配置（核心）
@ComponentScan(             // 组件扫描
    basePackages = "com.example"  // 扫描当前包及子包
)
public @interface SpringBootApplication { }
```

### 最小依赖（pom.xml）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project>
    <!-- 父项目：管理依赖版本 -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <java.version>17</java.version>
    </properties>
    
    <dependencies>
        <!-- Web 起步依赖：包含 Spring MVC + Tomcat -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
    </dependencies>
</project>

<!-- ⚠️ 起步依赖的好处：
     1. 不需要指定版本（由 spring-boot-starter-parent 管理）
     2. 一个依赖包含所有需要的组件
     3. 版本兼容性有保证
-->
```

---

## 自动配置原理

### @EnableAutoConfiguration 的工作流程

```
1. @EnableAutoConfiguration 触发自动配置
         ↓
2. 通过 @Import 导入 AutoConfigurationImportSelector
         ↓
3. 读取 META-INF/spring.factories 或
   META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
         ↓
4. 加载所有自动配置类（如 DataSourceAutoConfiguration）
         ↓
5. 根据条件注解判断是否生效
         ↓
6. 生效的配置类创建相应的 Bean
```

### 条件注解详解

```java
// Spring Boot 使用条件注解决定是否应用某个自动配置

@Configuration
@ConditionalOnClass(DataSource.class)  // 类路径存在 DataSource 类时生效
@ConditionalOnMissingBean(DataSource.class)  // 容器中不存在 DataSource Bean 时生效
@ConditionalOnProperty("spring.datasource.url")  // 配置文件存在该属性时生效
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean  // 如果用户没有自定义，才创建
    public DataSource dataSource() {
        // 自动创建数据源
    }
}
```

### 常用条件注解

| 注解 | 说明 | 使用场景 |
|------|------|----------|
| `@ConditionalOnClass` | 类路径存在指定类时生效 | 有某个依赖时才配置 |
| `@ConditionalOnMissingClass` | 类路径不存在指定类时生效 | 没有某个依赖时才配置 |
| `@ConditionalOnBean` | 容器中存在指定 Bean 时生效 | 有某个 Bean 时才配置 |
| `@ConditionalOnMissingBean` | 容器中不存在指定 Bean 时生效 | 用户未自定义时才自动配置 |
| `@ConditionalOnProperty` | 配置属性满足条件时生效 | 根据配置决定是否启用 |
| `@ConditionalOnWebApplication` | Web 应用时生效 | 只在 Web 环境生效 |

### 查看自动配置报告

```yaml
# application.yml
debug: true  # 开启调试模式，打印自动配置报告

# 启动后会打印：
# Positive matches（生效的配置）
# Negative matches（未生效的配置及原因）
```

---

## 起步依赖

### 常用起步依赖

| 依赖 | 包含内容 | 使用场景 |
|------|----------|----------|
| `spring-boot-starter-web` | Spring MVC + Tomcat + JSON | Web 开发 |
| `spring-boot-starter-data-jpa` | Spring Data JPA + Hibernate | ORM 数据访问 |
| `spring-boot-starter-data-redis` | Spring Data Redis | Redis 操作 |
| `spring-boot-starter-security` | Spring Security | 安全认证 |
| `spring-boot-starter-test` | JUnit + Mockito | 测试 |
| `spring-boot-starter-actuator` | 监控端点 | 生产监控 |
| `spring-boot-starter-aop` | Spring AOP | 切面编程 |
| `spring-boot-starter-validation` | 参数校验 | 参数验证 |

### 依赖示例

```xml
<dependencies>
    <!-- Web 开发：包含 Spring MVC、Tomcat、Jackson -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- JPA 数据访问：包含 Hibernate、Spring Data JPA -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    
    <!-- MySQL 驱动 -->
    <dependency>
        <groupId>com.mysql</groupId>
        <artifactId>mysql-connector-j</artifactId>
        <scope>runtime</scope>
    </dependency>
    
    <!-- Lombok：简化代码 -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
    
    <!-- 测试 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

---

## 配置管理

### 配置文件格式

```yaml
# application.yml（推荐，更易读）

# ========== 服务器配置 ==========
server:
  port: 8080                    # 服务端口
  servlet:
    context-path: /api          # 上下文路径

# ========== Spring 配置 ==========
spring:
  # 数据源配置
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai
    username: root
    password: root
    driver-class-name: com.mysql.cj.jdbc.Driver
    # HikariCP 连接池配置
    hikari:
      maximum-pool-size: 20      # 最大连接数
      minimum-idle: 5            # 最小空闲连接
  
  # JPA 配置
  jpa:
    hibernate:
      ddl-auto: update           # 自动更新表结构
    show-sql: true               # 显示 SQL
    properties:
      hibernate:
        format_sql: true         # 格式化 SQL
  
  # Redis 配置
  data:
    redis:
      host: localhost
      port: 6379
      database: 0
  
  # 文件上传
  servlet:
    multipart:
      max-file-size: 10MB       # 单文件最大大小
      max-request-size: 100MB   # 总请求最大大小

# ========== 日志配置 ==========
logging:
  level:
    root: info                   # 全局日志级别
    com.example: debug           # 特定包日志级别
  file:
    name: logs/app.log           # 日志文件路径
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"

# ========== 自定义配置 ==========
app:
  name: My Application
  version: 1.0.0
  features:
    - feature1
    - feature2
```

### 读取配置

```java
// ========== 方式一：@Value 注解 ==========
@Service
public class MyService {
    @Value("${app.name}")              // 注入配置值
    private String appName;
    
    @Value("${server.port:8080}")      // 带默认值
    private int port;
    
    @Value("${app.features}")          // 列表
    private List<String> features;
}

// ========== 方式二：@ConfigurationProperties（推荐）==========
@Component
@ConfigurationProperties(prefix = "app")  // 绑定 app.* 配置
@Data  // Lombok 自动生成 getter/setter
public class AppProperties {
    private String name;
    private String version;
    private List<String> features;
    private Map<String, String> config;
}

// 使用
@Service
@RequiredArgsConstructor
public class MyService {
    private final AppProperties appProperties;
    
    public void printInfo() {
        System.out.println(appProperties.getName());      // My Application
        System.out.println(appProperties.getFeatures());  // [feature1, feature2]
    }
}

// ⚠️ @ConfigurationProperties 的优势：
// 1. 类型安全（编译时检查）
// 2. 支持嵌套对象、列表、Map
// 3. 支持 YAML 和 Properties 格式
// 4. IDE 有配置提示（需要添加依赖）
```

### 多环境配置

```yaml
# application.yml（主配置）
spring:
  profiles:
    active: dev  # 激活 dev 环境

---
# application-dev.yml（开发环境）
server:
  port: 8080

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/dev_db

logging:
  level:
    com.example: debug

---
# application-prod.yml（生产环境）
server:
  port: 80

spring:
  datasource:
    url: jdbc:mysql://prod-server:3306/prod_db

logging:
  level:
    com.example: info
```

**激活方式：**

```bash
# 方式一：配置文件
spring.profiles.active=prod

# 方式二：命令行参数
java -jar app.jar --spring.profiles.active=prod

# 方式三：环境变量
export SPRING_PROFILES_ACTIVE=prod

# 方式四：IDE 配置
# Run → Edit Configurations → Active profiles
```

---

## Actuator 监控

Spring Boot Actuator 提供了生产级别的监控和管理功能。

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

### 配置端点

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,env,beans  # 暴露的端点
        # include: "*"  # 暴露所有端点（不推荐生产使用）
  endpoint:
    health:
      show-details: always  # 显示健康详情
```

### 常用端点

| 端点 | 说明 | 使用场景 |
|------|------|----------|
| `/actuator/health` | 应用健康状态 | 健康检查 |
| `/actuator/info` | 应用信息 | 版本信息 |
| `/actuator/metrics` | 性能指标 | 性能监控 |
| `/actuator/env` | 环境变量 | 配置查看 |
| `/actuator/beans` | 所有 Bean | 调试 |
| `/actuator/mappings` | URL 映射 | 接口调试 |
| `/actuator/threaddump` | 线程转储 | 线程问题排查 |

### 自定义健康检查

```java
@Component
public class DatabaseHealthIndicator implements HealthIndicator {
    
    @Autowired
    private DataSource dataSource;
    
    @Override
    public Health health() {
        try (Connection conn = dataSource.getConnection()) {
            // 执行简单查询验证数据库连接
            boolean valid = conn.isValid(1000);
            
            if (valid) {
                return Health.up()
                    .withDetail("database", "MySQL")
                    .withDetail("validationQuery", "isValid")
                    .build();
            } else {
                return Health.down()
                    .withDetail("error", "Database connection invalid")
                    .build();
            }
        } catch (SQLException e) {
            return Health.down()
                .withDetail("error", e.getMessage())
                .build();
        }
    }
}

// 访问 /actuator/health 会返回：
// {
//   "status": "UP",
//   "components": {
//     "database": { "status": "UP", "details": { "database": "MySQL" } },
//     "diskSpace": { "status": "UP" }
//   }
// }
```

---

## 常用功能示例

### 统一响应封装

```java
// 统一响应类
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Result<T> {
    private int code;        // 状态码
    private String message;  // 消息
    private T data;          // 数据
    
    // 成功响应
    public static <T> Result<T> success(T data) {
        return new Result<>(200, "success", data);
    }
    
    public static <T> Result<T> success() {
        return success(null);
    }
    
    // 失败响应
    public static <T> Result<T> error(String message) {
        return new Result<>(500, message, null);
    }
    
    public static <T> Result<T> error(int code, String message) {
        return new Result<>(code, message, null);
    }
}

// 使用
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @GetMapping("/{id}")
    public Result<User> getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        if (user == null) {
            return Result.error("用户不存在");
        }
        return Result.success(user);
    }
    
    @PostMapping
    public Result<User> create(@RequestBody UserDTO dto) {
        User user = userService.create(dto);
        return Result.success(user);
    }
}
```

### 全局异常处理

```java
// 全局异常处理器
@RestControllerAdvice  // 对所有 @RestController 生效
public class GlobalExceptionHandler {
    
    // 处理业务异常
    @ExceptionHandler(BusinessException.class)
    public Result<Void> handleBusinessException(BusinessException e) {
        return Result.error(e.getCode(), e.getMessage());
    }
    
    // 处理资源不存在
    @ExceptionHandler(EntityNotFoundException.class)
    public Result<Void> handleNotFound(EntityNotFoundException e) {
        return Result.error(404, "资源不存在");
    }
    
    // 处理参数校验异常
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<Void> handleValidation(MethodArgumentNotValidException e) {
        String message = e.getBindingResult()
            .getFieldErrors()
            .stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));
        return Result.error(400, "参数校验失败: " + message);
    }
    
    // 处理所有异常
    @ExceptionHandler(Exception.class)
    public Result<Void> handleException(Exception e) {
        log.error("系统异常", e);
        return Result.error("系统繁忙，请稍后重试");
    }
}
```

### 参数校验

```java
// 添加依赖
// spring-boot-starter-validation

// DTO 类
@Data
public class UserDTO {
    @NotBlank(message = "用户名不能为空")
    @Size(min = 3, max = 20, message = "用户名长度3-20位")
    private String username;
    
    @NotBlank(message = "密码不能为空")
    @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}$", 
             message = "密码必须包含大小写字母和数字，至少8位")
    private String password;
    
    @Email(message = "邮箱格式不正确")
    private String email;
    
    @Min(value = 18, message = "年龄不能小于18岁")
    @Max(value = 100, message = "年龄不能超过100岁")
    private Integer age;
}

// Controller
@RestController
@RequestMapping("/api/users")
@Validated  // 启用方法级参数校验
public class UserController {
    
    @PostMapping
    public Result<Void> create(@RequestBody @Valid UserDTO dto) {
        // 如果校验失败，会抛出 MethodArgumentNotValidException
        userService.create(dto);
        return Result.success();
    }
    
    @GetMapping("/check")
    public Result<Void> checkUsername(
            @RequestParam @NotBlank(message = "用户名不能为空") String username) {
        return Result.success();
    }
}
```

---

## 常见问题与注意事项

### 排除自动配置

```java
// 方式一：启动类注解排除
@SpringBootApplication(exclude = {
    DataSourceAutoConfiguration.class,
    HibernateJpaAutoConfiguration.class
})
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}

// 方式二：配置文件排除
spring:
  autoconfigure:
    exclude:
      - org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
```

### 修改默认端口

```yaml
server:
  port: 9090  # 默认 8080
```

### 数据库连接池配置

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20      # 最大连接数
      minimum-idle: 5            # 最小空闲连接
      idle-timeout: 300000       # 空闲超时（毫秒）
      connection-timeout: 30000  # 连接超时（毫秒）
      max-lifetime: 1800000      # 连接最大生命周期
      pool-name: MyHikariPool    # 连接池名称
```

---

## 小结

| 特性 | 说明 |
|------|------|
| **自动配置** | 根据依赖自动配置 Spring |
| **起步依赖** | 一个依赖包含所有需要的组件 |
| **内嵌服务器** | 内置 Tomcat，无需部署 |
| **配置管理** | YAML 格式，类型安全的配置绑定 |
| **多环境配置** | profile 机制切换环境 |
| **Actuator** | 生产级监控和管理 |

### Spring Boot vs 传统 Spring

| 方面 | 传统 Spring | Spring Boot |
|------|-------------|-------------|
| 配置方式 | 大量 XML | 注解 + YAML |
| 依赖管理 | 手动版本 | 起步依赖 |
| 应用服务器 | 安装 Tomcat | 内嵌 Tomcat |
| 启动方式 | 部署 WAR | java -jar |
| 开发效率 | 较低 | 较高 |