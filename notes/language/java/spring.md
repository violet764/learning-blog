# Spring 核心框架

> Spring 是 Java 企业级开发中最流行的框架，核心思想是**控制反转（IoC）**和**面向切面编程（AOP）**，通过依赖注入实现松耦合的应用架构。

## 框架概述

Spring 框架由 Rod Johnson 于 2003 年创建，是一个轻量级的控制反转和面向切面编程的容器框架。它提供了全面的基础设施支持，使得开发者可以专注于业务逻辑的实现。

### 🎯 核心特性

| 特性 | 说明 |
|------|------|
| **IoC 容器** | 控制反转，管理对象的生命周期和依赖关系 |
| **AOP** | 面向切面编程，实现横切关注点的模块化 |
| **事务管理** | 声明式事务，简化事务处理 |
| **MVC 框架** | Web 层解决方案，与 Spring 无缝集成 |
| **数据访问** | 简化 JDBC、ORM 框架的集成 |

### 📦 Spring 模块体系

```
Spring 框架
├── Core Container（核心容器）
│   ├── spring-core      # 基础组件
│   ├── spring-beans     # Bean 工厂
│   ├── spring-context   # 上下文
│   └── spring-expression # SpEL 表达式
├── AOP & Aspects        # 切面编程
├── Data Access          # 数据访问
├── Web                  # Web 层
└── Test                 # 测试支持
```

---

## IoC 容器与依赖注入

### 什么是 IoC？

**控制反转（Inversion of Control）** 是一种设计思想，将对象的创建和管理权从应用程序代码转移到外部容器（Spring 容器）。

### 传统方式 vs IoC 方式

```java
// ❌ 传统方式：对象自行创建依赖
public class UserService {
    private UserDao userDao = new UserDaoImpl(); // 紧耦合
}

// ✅ IoC 方式：依赖由容器注入
public class UserService {
    private UserDao userDao;
    
    // 构造器注入
    public UserService(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

### IoC 容器的核心接口

| 接口 | 说明 |
|------|------|
| `BeanFactory` | 基础容器，延迟加载 |
| `ApplicationContext` | 高级容器，立即加载，功能更丰富 |

### ApplicationContext 常用实现

```java
// XML 配置方式
ApplicationContext ctx = new ClassPathXmlApplicationContext("beans.xml");

// 注解配置方式
ApplicationContext ctx = new AnnotationConfigApplicationContext(AppConfig.class);

// Web 应用
WebApplicationContext ctx = WebApplicationContextUtils.getWebApplicationContext(servletContext);
```

---

## Bean 的配置方式

### 1. XML 配置方式

```xml
<!-- beans.xml -->
<beans xmlns="http://www.springframework.org/schema/beans">
    <!-- 使用构造器创建 Bean -->
    <bean id="userDao" class="com.example.dao.impl.UserDaoImpl"/>
    
    <!-- 使用构造器注入 -->
    <bean id="userService" class="com.example.service.UserService">
        <constructor-arg ref="userDao"/>
    </bean>
    
    <!-- 使用 setter 注入 -->
    <bean id="userService" class="com.example.service.UserService">
        <property name="userDao" ref="userDao"/>
        <property name="name" value="Spring"/>
    </bean>
</beans>
```

### 2. 注解配置方式

```java
// 配置类
@Configuration
@ComponentScan("com.example")
public class AppConfig {
    
    @Bean
    public UserDao userDao() {
        return new UserDaoImpl();
    }
    
    @Bean
    public UserService userService(UserDao userDao) {
        return new UserService(userDao);
    }
}
```

### 3. 组件扫描与自动装配

```java
// 组件注解
@Repository  // 持久层
@Service     // 业务层
@Controller  // 控制层
@Component   // 通用组件

// 使用示例
@Service
public class UserService {
    
    @Autowired  // 按类型自动装配
    private UserDao userDao;
    
    @Autowired
    @Qualifier("userDaoImpl")  // 按名称装配
    private UserDao userDao;
    
    @Resource(name = "userDao")  // JSR-250 注解
    private UserDao userDao;
}
```

---

## 依赖注入方式

### 1. 构造器注入（推荐）

```java
@Service
public class UserService {
    private final UserDao userDao;
    
    // 单构造器可省略 @Autowired
    public UserService(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

### 2. Setter 注入

```java
@Service
public class UserService {
    private UserDao userDao;
    
    @Autowired
    public void setUserDao(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

### 3. 字段注入（不推荐）

```java
@Service
public class UserService {
    @Autowired
    private UserDao userDao;  // 不利于测试和不可变设计
}
```

### 📌 注入方式对比

| 方式 | 优点 | 缺点 | 推荐度 |
|------|------|------|:------:|
| 构造器注入 | 保证依赖不为空、支持不可变对象 | 构造器参数多时显得臃肿 | ⭐⭐⭐ |
| Setter 注入 | 灵活、可选依赖 | 依赖可能为空 | ⭐⭐ |
| 字段注入 | 简洁 | 不利于测试、无法不可变 | ⭐ |

---

## Bean 的作用域

```java
@Component
@Scope("singleton")  // 默认值
public class UserService { }

@Component
@Scope("prototype")  // 每次获取创建新实例
public class PrototypeBean { }

// Web 作用域
@Scope("request")    // 每个 HTTP 请求一个实例
@Scope("session")    // 每个 HTTP 会话一个实例
@Scope("application") // 整个 Web 应用一个实例
```

### 作用域对比

| 作用域 | 说明 | 使用场景 |
|--------|------|----------|
| `singleton` | 单例，整个容器只有一个实例 | 无状态 Bean |
| `prototype` | 原型，每次获取创建新实例 | 有状态 Bean |
| `request` | 每个 HTTP 请求一个实例 | Web 应用 |
| `session` | 每个会话一个实例 | 用户会话信息 |
| `application` | 整个应用一个实例 | 全局配置 |

---

## Bean 生命周期

Spring Bean 的生命周期包括**实例化、属性赋值、初始化、销毁**四个阶段，并提供了多个扩展点。

```
实例化 → 属性赋值 → 初始化前处理 → 初始化 → 初始化后处理 → 使用 → 销毁前处理 → 销毁
```

### 生命周期回调

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    private String name;
    
    // 属性注入
    @Autowired
    public void setName(String name) {
        System.out.println("2. 属性注入");
        this.name = name;
    }
    
    // 初始化回调方式一：实现接口
    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("4. 初始化 - InitializingBean 接口");
    }
    
    // 初始化回调方式二：注解
    @PostConstruct
    public void init() {
        System.out.println("3. 初始化 - @PostConstruct");
    }
    
    // 初始化回调方式三：@Bean 注解属性
    // @Bean(initMethod = "customInit")
    public void customInit() {
        System.out.println("5. 自定义初始化方法");
    }
    
    // 销毁回调
    @PreDestroy
    public void destroy() {
        System.out.println("销毁 - @PreDestroy");
    }
}
```

### BeanPostProcessor 扩展点

```java
@Component
public class CustomBeanPostProcessor implements BeanPostProcessor {
    
    @Override
    public Object postProcessBeforeInitialization(Object bean, String beanName) {
        System.out.println("初始化前处理: " + beanName);
        return bean;
    }
    
    @Override
    public Object postProcessAfterInitialization(Object bean, String beanName) {
        System.out.println("初始化后处理: " + beanName);
        return bean;
    }
}
```

---

## AOP 面向切面编程

### 什么是 AOP？

**面向切面编程（Aspect-Oriented Programming）** 是一种编程范式，用于将横切关注点（如日志、事务、安全）从业务逻辑中分离出来。

### 核心概念

| 概念 | 说明 |
|------|------|
| **切面（Aspect）** | 横切关注点的模块化封装 |
| **连接点（JoinPoint）** | 程序执行的特定点（方法调用、异常抛出等） |
| **切点（Pointcut）** | 匹配连接点的表达式 |
| **通知（Advice）** | 在切点执行的动作 |
| **目标对象（Target）** | 被通知的对象 |
| **代理（Proxy）** | AOP 框架创建的对象 |

### 通知类型

```java
@Aspect
@Component
public class LoggingAspect {
    
    // 切点表达式：匹配 com.example.service 包下所有类的所有方法
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayer() {}
    
    // 前置通知
    @Before("serviceLayer()")
    public void beforeMethod(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        System.out.println("执行方法前: " + methodName);
    }
    
    // 后置通知（无论是否异常都执行）
    @After("serviceLayer()")
    public void afterMethod(JoinPoint joinPoint) {
        System.out.println("方法执行结束");
    }
    
    // 返回通知
    @AfterReturning(pointcut = "serviceLayer()", returning = "result")
    public void afterReturning(JoinPoint joinPoint, Object result) {
        System.out.println("方法返回值: " + result);
    }
    
    // 异常通知
    @AfterThrowing(pointcut = "serviceLayer()", throwing = "ex")
    public void afterThrowing(JoinPoint joinPoint, Exception ex) {
        System.out.println("方法抛出异常: " + ex.getMessage());
    }
    
    // 环绕通知
    @Around("serviceLayer()")
    public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        long start = System.currentTimeMillis();
        
        try {
            Object result = joinPoint.proceed();  // 执行目标方法
            return result;
        } finally {
            long end = System.currentTimeMillis();
            System.out.println("方法执行耗时: " + (end - start) + "ms");
        }
    }
}
```

### 切点表达式语法

```java
// execution(修饰符 返回值 包.类.方法(参数))

// 匹配所有 public 方法
execution(public * *(..))

// 匹配特定类的所有方法
execution(* com.example.service.UserService.*(..))

// 匹配特定包下所有类的所有方法
execution(* com.example.service.*.*(..))

// 匹配特定包及其子包下所有方法
execution(* com.example..*.*(..))

// 匹配带特定注解的方法
@annotation(com.example.annotation.Log)

// 匹配带特定注解的类
@within(com.example.annotation.Service)
```

### AOP 实战：日志切面

```java
@Aspect
@Component
@Slf4j
public class LogAspect {
    
    @Around("@annotation(logAnnotation)")
    public Object logExecution(ProceedingJoinPoint joinPoint, LogAnnotation logAnnotation) throws Throwable {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        log.info("{}.{} 开始执行, 参数: {}", className, methodName, Arrays.toString(args));
        
        long startTime = System.currentTimeMillis();
        try {
            Object result = joinPoint.proceed();
            long elapsed = System.currentTimeMillis() - startTime;
            log.info("{}.{} 执行成功, 耗时: {}ms, 返回: {}", className, methodName, elapsed, result);
            return result;
        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - startTime;
            log.error("{}.{} 执行失败, 耗时: {}ms, 异常: {}", className, methodName, elapsed, e.getMessage());
            throw e;
        }
    }
}

// 自定义注解
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LogAnnotation {
    String value() default "";
}

// 使用注解
@Service
public class UserService {
    @LogAnnotation("用户查询")
    public User findById(Long id) {
        return userDao.findById(id);
    }
}
```

---

## Spring 配置最佳实践

### 配置类示例

```java
@Configuration
@ComponentScan(basePackages = "com.example",
    excludeFilters = @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = Controller.class))
@PropertySource("classpath:application.properties")
@EnableTransactionManagement
public class AppConfig {
    
    @Value("${database.url}")
    private String dbUrl;
    
    @Bean
    public DataSource dataSource() {
        DriverManagerDataSource ds = new DriverManagerDataSource();
        ds.setUrl(dbUrl);
        ds.setUsername("root");
        ds.setPassword("password");
        return ds;
    }
    
    @Bean
    public JdbcTemplate jdbcTemplate(DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }
    
    @Bean
    public PlatformTransactionManager transactionManager(DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
```

---

## 常见问题与注意事项

### ⚠️ 循环依赖

```java
// 问题：A 依赖 B，B 依赖 A
@Service
public class ServiceA {
    @Autowired
    private ServiceB serviceB;
}

@Service
public class ServiceB {
    @Autowired
    private ServiceA serviceA;
}

// 解决方案 1：使用 @Lazy 延迟加载
@Service
public class ServiceA {
    @Autowired
    @Lazy
    private ServiceB serviceB;
}

// 解决方案 2：重构设计，消除循环依赖
```

### ⚠️ Bean 命名冲突

```java
// 默认 Bean 名称：类名首字母小写
@Service
public class UserServiceImpl {}  // beanName = "userServiceImpl"

// 指定 Bean 名称
@Service("userService")
public class UserServiceImpl {}  // beanName = "userService"
```

### ⚠️ 代理限制

```java
@Service
public class OrderService {
    
    // ❌ 同类内部调用，AOP 不生效
    public void placeOrder(Order order) {
        this.validateOrder(order);  // 直接调用，不走代理
    }
    
    @Transactional
    public void validateOrder(Order order) {
        // 事务不生效
    }
}

// ✅ 解决方案：注入自身或使用 AopContext
@Service
public class OrderService {
    @Autowired
    private OrderService self;  // 注入自身代理
    
    public void placeOrder(Order order) {
        self.validateOrder(order);  // 通过代理调用
    }
}
```

---

## 参考资料

- [Spring Framework 官方文档](https://docs.spring.io/spring-framework/reference/)
- [Spring IoC 容器](https://docs.spring.io/spring-framework/reference/core/beans.html)
- [Spring AOP](https://docs.spring.io/spring-framework/reference/core/aop.html)
