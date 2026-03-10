# Spring Boot 框架

> Spring Boot 是 Spring 生态的快速开发框架，通过**约定优于配置**的理念，大幅简化了 Spring 应用的初始搭建和开发过程。

## 框架概述

Spring Boot 由 Pivotal 团队开发，旨在简化 Spring 应用的开发。它提供了自动配置、起步依赖、内嵌服务器等特性，让开发者可以快速构建生产级别的 Spring 应用。

### 🎯 核心特性

| 特性 | 说明 |
|------|------|
| **自动配置** | 根据依赖自动配置 Spring 应用 |
| **起步依赖** | 简化依赖管理，一个依赖包含所需全部组件 |
| **内嵌服务器** | 内置 Tomcat、Jetty、Undertow |
| **生产就绪** | 提供健康检查、指标监控等功能 |
| **零配置** | 无需 XML 配置文件 |

### Spring Boot vs 传统 Spring

```
传统 Spring:
1. 配置 web.xml
2. 配置 applicationContext.xml
3. 配置 dispatcherServlet.xml
4. 配置数据库连接
5. 配置事务管理
6. 手动部署到 Tomcat

Spring Boot:
1. 添加起步依赖
2. 编写主启动类
3. 启动！
```

---

## 快速开始

### 创建项目

**方式一：Spring Initializr**
- 访问 https://start.spring.io/
- 选择依赖、版本
- 下载项目

**方式二：IDE 创建**
- IntelliJ IDEA: File → New → Project → Spring Initializr

### 项目结构

```
my-springboot-app/
├── src/
│   ├── main/
│   │   ├── java/com/example/demo/
│   │   │   ├── DemoApplication.java    # 启动类
│   │   │   ├── controller/
│   │   │   ├── service/
│   │   │   └── repository/
│   │   └── resources/
│   │       ├── application.yml         # 配置文件
│   │       ├── static/                 # 静态资源
│   │       └── templates/              # 模板文件
│   └── test/
├── pom.xml                             # Maven 配置
└── mvnw                                # Maven Wrapper
```

### 主启动类

```java
@SpringBootApplication  // 核心注解
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
```

### 最小依赖（pom.xml）

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.0</version>
</parent>

<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

---

## 自动配置原理

### @SpringBootApplication 注解

```java
@SpringBootApplication 是一个组合注解：
@SpringBootConfiguration    // 标记为配置类
@EnableAutoConfiguration    // 启用自动配置
@ComponentScan              // 组件扫描
public @interface SpringBootApplication { }
```

### 自动配置流程

```
1. @EnableAutoConfiguration 触发自动配置
         ↓
2. 通过 @Import 导入 AutoConfigurationImportSelector
         ↓
3. 读取 META-INF/spring.factories 或 META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
         ↓
4. 加载所有自动配置类（如 DataSourceAutoConfiguration）
         ↓
5. 根据条件注解判断是否生效
         ↓
6. 生效的配置类创建相应的 Bean
```

### 条件注解

```java
@Configuration
@ConditionalOnClass(DataSource.class)           // 类路径存在 DataSource 类
@ConditionalOnMissingBean(DataSource.class)      // 容器中不存在 DataSource Bean
@ConditionalOnProperty("spring.datasource.url")  // 配置文件存在该属性
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public DataSource dataSource() {
        // 创建数据源
    }
}
```

### 常用条件注解

| 注解 | 说明 |
|------|------|
| `@ConditionalOnClass` | 类路径存在指定类时生效 |
| `@ConditionalOnMissingClass` | 类路径不存在指定类时生效 |
| `@ConditionalOnBean` | 容器中存在指定 Bean 时生效 |
| `@ConditionalOnMissingBean` | 容器中不存在指定 Bean 时生效 |
| `@ConditionalOnProperty` | 配置属性满足条件时生效 |
| `@ConditionalOnWebApplication` | Web 应用时生效 |

---

## 起步依赖

### 常用起步依赖

| 依赖 | 说明 |
|------|------|
| `spring-boot-starter-web` | Web 开发，包含 Spring MVC、Tomcat |
| `spring-boot-starter-data-jpa` | JPA 数据访问 |
| `spring-boot-starter-data-redis` | Redis 支持 |
| `spring-boot-starter-security` | Spring Security |
| `spring-boot-starter-test` | 测试支持 |
| `spring-boot-starter-actuator` | 生产监控 |
| `spring-boot-starter-aop` | AOP 支持 |
| `spring-boot-starter-validation` | 参数校验 |

### 依赖示例

```xml
<dependencies>
    <!-- Web 开发 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- 数据访问 -->
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
    
    <!-- Lombok -->
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

### 配置文件

Spring Boot 支持 `application.properties` 和 `application.yml` 两种格式，推荐使用 YAML。

```yaml
# application.yml

# 服务端口
server:
  port: 8080
  servlet:
    context-path: /api

# 数据源配置
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai
    username: root
    password: root
    driver-class-name: com.mysql.cj.jdbc.Driver
  
  # JPA 配置
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: true
    properties:
      hibernate:
        format_sql: true
  
  # Redis 配置
  data:
    redis:
      host: localhost
      port: 6379
      password: 
      database: 0
  
  # 文件上传
  servlet:
    multipart:
      max-file-size: 10MB
      max-request-size: 100MB

# 日志配置
logging:
  level:
    root: info
    com.example: debug
  file:
    name: logs/app.log

# 自定义配置
app:
  name: My Application
  version: 1.0.0
```

### 读取配置

**方式一：@Value 注解**

```java
@Service
public class MyService {
    @Value("${app.name}")
    private String appName;
    
    @Value("${server.port:8080}")  // 默认值
    private int port;
}
```

**方式二：@ConfigurationProperties（推荐）**

```java
@Component
@ConfigurationProperties(prefix = "app")
@Data
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
        System.out.println(appProperties.getName());
    }
}
```

### 多环境配置

```yaml
# application.yml（主配置）
spring:
  profiles:
    active: dev  # 激活 dev 环境

---
# application-dev.yml
server:
  port: 8080

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/dev_db

---
# application-prod.yml
server:
  port: 80

spring:
  datasource:
    url: jdbc:mysql://prod-server:3306/prod_db
```

**激活方式：**
```bash
# 命令行参数
java -jar app.jar --spring.profiles.active=prod

# 环境变量
export SPRING_PROFILES_ACTIVE=prod
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
  endpoint:
    health:
      show-details: always  # 显示健康详情
```

### 常用端点

| 端点 | 说明 |
|------|------|
| `/actuator/health` | 应用健康状态 |
| `/actuator/info` | 应用信息 |
| `/actuator/metrics` | 性能指标 |
| `/actuator/env` | 环境变量 |
| `/actuator/beans` | 所有 Bean |
| `/actuator/mappings` | URL 映射 |
| `/actuator/threaddump` | 线程转储 |

### 自定义健康检查

```java
@Component
public class CustomHealthIndicator implements HealthIndicator {
    
    @Override
    public Health health() {
        // 自定义健康检查逻辑
        boolean isHealthy = checkHealth();
        
        if (isHealthy) {
            return Health.up()
                .withDetail("message", "服务正常")
                .build();
        } else {
            return Health.down()
                .withDetail("message", "服务异常")
                .build();
        }
    }
    
    private boolean checkHealth() {
        // 检查逻辑
        return true;
    }
}
```

---

## 常用功能示例

### 统一响应封装

```java
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Result<T> {
    private int code;
    private String message;
    private T data;
    
    public static <T> Result<T> success(T data) {
        return new Result<>(200, "success", data);
    }
    
    public static <T> Result<T> error(String message) {
        return new Result<>(500, message, null);
    }
}

// 使用
@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @GetMapping("/{id}")
    public Result<User> getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        return Result.success(user);
    }
}
```

### 全局异常处理

```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    @ExceptionHandler(Exception.class)
    public Result<Void> handleException(Exception e) {
        return Result.error(e.getMessage());
    }
    
    @ExceptionHandler(RuntimeException.class)
    public Result<Void> handleRuntimeException(RuntimeException e) {
        return Result.error("系统异常: " + e.getMessage());
    }
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<Void> handleValidationException(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getFieldError().getDefaultMessage();
        return Result.error("参数校验失败: " + message);
    }
}
```

### 参数校验

```java
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

@RestController
@RequestMapping("/api/users")
@Validated
public class UserController {
    
    @PostMapping
    public Result<Void> createUser(@RequestBody @Valid UserDTO userDTO) {
        userService.create(userDTO);
        return Result.success(null);
    }
    
    @GetMapping("/check")
    public Result<Void> checkUsername(
            @RequestParam @NotBlank(message = "用户名不能为空") String username) {
        return Result.success(null);
    }
}
```

---

## 常见问题与注意事项

### ⚠️ 端口冲突

```yaml
# 修改默认端口
server:
  port: 9090
```

### ⚠️ 数据库连接池

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20      # 最大连接数
      minimum-idle: 5            # 最小空闲连接
      idle-timeout: 300000       # 空闲超时
      connection-timeout: 30000  # 连接超时
```

### ⚠️ 排除自动配置

```java
@SpringBootApplication(exclude = {
    DataSourceAutoConfiguration.class,
    HibernateJpaAutoConfiguration.class
})
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
```

### ⚠️ 启动 Banner 自定义

在 `resources/banner.txt` 中自定义启动 banner：

```
${AnsiColor.BRIGHT_BLUE}
  ____  _             _   _              
 / ___|| |_ __ _ _ __| |_(_)_ __   __ _  
 \___ \| __/ _` | '__| __| | '_ \ / _` | 
  ___) | || (_| | |  | |_| | | | | (_| | 
 |____/ \__\__,_|_|   \__|_|_| |_|\__, | 
                                   |___/ 
${AnsiColor.DEFAULT}
Spring Boot Version: ${spring-boot.version}
```

---

## 参考资料

- [Spring Boot 官方文档](https://docs.spring.io/spring-boot/)
- [Spring Boot 自动配置原理](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.developing-auto-configuration)
- [Spring Boot Actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html)
