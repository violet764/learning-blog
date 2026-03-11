# Spring 核心框架

> Spring 是 Java 企业级开发中最流行的框架，核心思想是**控制反转（IoC）**和**面向切面编程（AOP）**，通过依赖注入实现松耦合的应用架构。

##  Spring

### 传统开发的问题

```java
// 传统开发方式的问题：紧耦合

// 数据访问层
public class UserDaoImpl implements UserDao {
    public User findById(Long id) {
        // 数据库查询逻辑
    }
}

// 业务层：直接创建依赖对象
public class UserServiceImpl implements UserService {
    // 问题一：硬编码创建依赖
    private UserDao userDao = new UserDaoImpl();
    
    // 问题二：如果 UserDaoImpl 需要修改构造参数，这里也要改
    // 问题三：无法替换实现（比如测试时想用 Mock）
    // 问题四：对象的生命周期难以管理
    
    public User getUser(Long id) {
        return userDao.findById(id);
    }
}

// 控制层：同样紧耦合
public class UserController {
    private UserService userService = new UserServiceImpl();
}
```

### Spring 的解决方案

```java
// Spring 开发方式：依赖注入（控制反转）

// 只需要定义接口和实现，不需要关心对象创建
@Repository
public class UserDaoImpl implements UserDao {
    public User findById(Long id) {
        // 数据库查询逻辑
    }
}

@Service
public class UserServiceImpl implements UserService {
    // 由 Spring 容器注入依赖，而不是自己创建
    // 这就是"控制反转"：对象的创建权交给 Spring
    private final UserDao userDao;
    
    // 构造器注入（推荐）
    public UserServiceImpl(UserDao userDao) {
        this.userDao = userDao;
    }
    
    public User getUser(Long id) {
        return userDao.findById(id);
    }
}

// Spring 容器负责：
// 1. 创建所有对象（Bean）
// 2. 管理对象之间的依赖关系
// 3. 管理对象的生命周期
```

### 什么是 IoC？

**IoC（Inversion of Control，控制反转）** 是一种设计思想：

| 方面 | 传统方式 | Spring 方式 |
|------|----------|-------------|
| 对象创建 | 程序员手动 new | Spring 容器创建 |
| 依赖关系 | 对象内部创建 | 外部容器注入 |
| 生命周期 | 程序员管理 | Spring 容器管理 |
| 解耦程度 | 紧耦合 | 松耦合 |

**通俗理解**：
- 传统方式：你要自己买菜、做饭（自己管理依赖）
- Spring 方式：你只要点餐，饭店给你送来（交给 Spring 管理）

---

## 框架概述

Spring 框架由 Rod Johnson 于 2003 年创建，是一个轻量级的控制反转和面向切面编程的容器框架。

### 核心特性

| 特性 | 说明 | 解决的问题 |
|------|------|-----------|
| **IoC 容器** | 控制反转，管理对象的生命周期和依赖关系 | 解耦，对象不需要自己创建依赖 |
| **AOP** | 面向切面编程，实现横切关注点的模块化 | 分离业务逻辑和公共逻辑 |
| **事务管理** | 声明式事务，简化事务处理 | 不需要手动管理事务 |
| **MVC 框架** | Web 层解决方案 | 构建 Web 应用 |
| **数据访问** | 简化 JDBC、ORM 框架的集成 | 统一数据访问方式 |

### Spring 模块体系

```
Spring 框架
├── Core Container（核心容器）
│   ├── spring-core      # IoC 和依赖注入的基础组件
│   ├── spring-beans     # Bean 工厂
│   ├── spring-context   # 上下文（扩展了 BeanFactory）
│   └── spring-expression # SpEL 表达式语言
│
├── AOP & Aspects        # 切面编程支持
├── Data Access          # 数据访问（JDBC、ORM、事务）
├── Web                  # Web 层（MVC、WebSocket）
└── Test                 # 测试支持
```

---

## IoC 容器与依赖注入

### IoC 容器的核心接口

| 接口 | 说明 | 使用场景 |
|------|------|----------|
| `BeanFactory` | 基础容器，延迟加载 | 资源受限环境 |
| `ApplicationContext` | 高级容器，立即加载 | 企业级应用（推荐） |

```java
// BeanFactory vs ApplicationContext

// BeanFactory：延迟加载
// 第一次使用 Bean 时才创建
BeanFactory factory = new XmlBeanFactory(new ClassPathResource("beans.xml"));
UserService service = (UserService) factory.getBean("userService");  // 此时才创建

// ApplicationContext：立即加载
// 容器启动时就创建所有单例 Bean
ApplicationContext ctx = new ClassPathXmlApplicationContext("beans.xml");
// 启动时已创建所有 Bean

// ApplicationContext 常用实现
// 1. XML 配置方式
ApplicationContext ctx1 = new ClassPathXmlApplicationContext("beans.xml");

// 2. 注解配置方式（推荐）
ApplicationContext ctx2 = new AnnotationConfigApplicationContext(AppConfig.class);

// 3. Web 应用
WebApplicationContext ctx3 = WebApplicationContextUtils
    .getWebApplicationContext(servletContext);
```

---

## Bean 的配置方式

### 1. XML 配置方式（传统方式）

```xml
<!-- beans.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
       http://www.springframework.org/schema/beans/spring-beans.xsd">
    
    <!-- 定义 Bean：id 是唯一标识，class 是全限定类名 -->
    <bean id="userDao" class="com.example.dao.impl.UserDaoImpl"/>
    
    <!-- 构造器注入 -->
    <bean id="userService" class="com.example.service.UserServiceImpl">
        <!-- ref 引用其他 Bean -->
        <constructor-arg ref="userDao"/>
    </bean>
    
    <!-- Setter 注入 -->
    <bean id="userService2" class="com.example.service.UserServiceImpl">
        <!-- name 是属性名，ref/value 是注入的值 -->
        <property name="userDao" ref="userDao"/>
        <property name="name" value="Spring"/>
    </bean>
</beans>

// 使用
ApplicationContext ctx = new ClassPathXmlApplicationContext("beans.xml");
UserService service = ctx.getBean(UserService.class);
```

### 2. 注解配置方式（推荐）

```java
// 配置类（替代 XML）
@Configuration  // 标记这是一个配置类
@ComponentScan("com.example")  // 组件扫描，自动发现 @Service、@Repository 等
@PropertySource("classpath:application.properties")  // 加载配置文件
public class AppConfig {
    
    // @Bean 方法定义的 Bean，方法名就是 Bean 的名称
    @Bean
    public UserDao userDao() {
        return new UserDaoImpl();
    }
    
    @Bean
    public UserService userService(UserDao userDao) {
        // 参数 userDao 会自动注入
        return new UserServiceImpl(userDao);
    }
    
    @Bean
    public DataSource dataSource(
        @Value("${database.url}") String url,    // 注入配置值
        @Value("${database.username}") String username
    ) {
        DriverManagerDataSource ds = new DriverManagerDataSource();
        ds.setUrl(url);
        ds.setUsername(username);
        return ds;
    }
}

// 使用
ApplicationContext ctx = new AnnotationConfigApplicationContext(AppConfig.class);
UserService service = ctx.getBean(UserService.class);
```

### 3. 组件扫描与自动装配（最常用）

```java
// 组件注解：标记类为 Spring Bean
@Repository   // 持久层（DAO）
@Service      // 业务层（Service）
@Controller   // 控制层（Controller）
@Component    // 通用组件

// 这些注解效果相同，区别在于语义清晰

// ========== DAO 层 ==========
@Repository  // 1. 标记为 Bean，自动扫描注册
public class UserDaoImpl implements UserDao {
    // ...
}

// ========== Service 层 ==========
@Service
public class UserServiceImpl implements UserService {
    
    // 自动装配：Spring 自动注入匹配的 Bean
    
    // 方式一：字段注入（不推荐）
    // @Autowired
    // private UserDao userDao;
    
    // 方式二：构造器注入（推荐）
    // 单构造器可省略 @Autowired
    private final UserDao userDao;
    
    public UserServiceImpl(UserDao userDao) {
        this.userDao = userDao;
    }
    
    // 方式三：Setter 注入
    // @Autowired
    // public void setUserDao(UserDao userDao) {
    //     this.userDao = userDao;
    // }
}

// ========== 按名称注入 ==========
@Service
public class OrderServiceImpl implements OrderService {
    
    @Autowired
    @Qualifier("userDaoImpl")  // 当有多个实现时，按名称注入
    private UserDao userDao;
    
    // 或者使用 @Resource（JSR-250 标准注解）
    @Resource(name = "userDaoImpl")
    private UserDao userDao2;
}

// @Autowired vs @Resource 的区别：
// @Autowired：Spring 注解，默认按类型注入
// @Resource：Java 标准注解，默认按名称注入
```

---

## 依赖注入方式详解

### 1. 构造器注入（推荐）

```java
@Service
public class UserService {
    private final UserDao userDao;     // final 保证不可变
    private final String appName;
    
    // 构造器注入
    // 优点：
    // 1. 保证依赖不为空（编译时检查）
    // 2. 支持不可变对象（final 字段）
    // 3. 便于单元测试（可以手动传入 Mock 对象）
    // 4. 单构造器可省略 @Autowired
    public UserService(UserDao userDao, 
                       @Value("${app.name}") String appName) {
        this.userDao = userDao;
        this.appName = appName;
    }
}
```

### 2. Setter 注入

```java
@Service
public class UserService {
    private UserDao userDao;
    
    // Setter 注入
    // 优点：
    // 1. 灵活，可选依赖
    // 2. 可以在运行时修改依赖
    // 缺点：
    // 1. 依赖可能为空
    // 2. 不能使用 final
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
    // 字段注入
    // 缺点：
    // 1. 不利于单元测试（需要反射设置字段）
    // 2. 无法使用 final
    // 3. 隐藏了类的依赖关系
    // 4. 容易违反单一职责（依赖太多时不明显）
    @Autowired
    private UserDao userDao;
}
```

### 注入方式对比

| 方式 | 优点 | 缺点 | 推荐度 |
|------|------|------|:------:|
| 构造器注入 | 保证依赖不为空、支持不可变 | 参数多时显得臃肿 | ⭐⭐⭐ |
| Setter 注入 | 灵活、可选依赖 | 依赖可能为空 | ⭐⭐ |
| 字段注入 | 简洁 | 不利于测试、无法不可变 | ⭐ |

---

## Bean 的作用域

```java
@Component
@Scope("singleton")  // 默认值，单例
public class UserService { }

@Component
@Scope("prototype")  // 每次获取创建新实例
public class PrototypeBean { }

// Web 环境专用
@Scope("request")     // 每个 HTTP 请求一个实例
@Scope("session")     // 每个 HTTP 会话一个实例
@Scope("application") // 整个 Web 应用一个实例
```

### 作用域详解

| 作用域 | 说明 | 使用场景 | 线程安全 |
|--------|------|----------|:--------:|
| `singleton` | 单例，整个容器只有一个实例 | 无状态 Bean（推荐） | 需要注意 |
| `prototype` | 原型，每次获取创建新实例 | 有状态 Bean | 安全 |
| `request` | 每个 HTTP 请求一个实例 | Web 应用 | 安全 |
| `session` | 每个会话一个实例 | 用户会话信息 | 安全 |

```java
// ⚠️ 单例 Bean 注入原型 Bean 的问题
@Scope("singleton")
public class SingletonBean {
    @Autowired
    private PrototypeBean prototypeBean;  // 问题：始终是同一个实例！
    
    // 解决方案一：使用 ObjectProvider
    @Autowired
    private ObjectProvider<PrototypeBean> prototypeProvider;
    
    public void method() {
        PrototypeBean bean = prototypeProvider.getObject();  // 每次获取新实例
    }
    
    // 解决方案二：使用 @Scope(proxyMode = ScopedProxyMode.TARGET_CLASS)
}

@Scope(value = "prototype", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class PrototypeBean { }
```

---

## Bean 生命周期

Spring Bean 的生命周期包括**实例化、属性赋值、初始化、销毁**四个阶段：

```
1. 实例化（Instantiation）
   └── Spring 调用构造方法创建 Bean 实例
   
2. 属性赋值（Populate）
   └── Spring 注入依赖（@Autowired、@Value 等）
   
3. 初始化（Initialization）
   ├── 处理 Aware 接口（BeanNameAware、ApplicationContextAware 等）
   ├── BeanPostProcessor.postProcessBeforeInitialization()
   ├── @PostConstruct 方法
   ├── InitializingBean.afterPropertiesSet()
   ├── 自定义 init-method
   └── BeanPostProcessor.postProcessAfterInitialization()
   
4. 使用（In Use）
   └── Bean 正常提供服务
   
5. 销毁（Destruction）
   ├── @PreDestroy 方法
   ├── DisposableBean.destroy()
   └── 自定义 destroy-method
```

### 生命周期回调示例

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    private String name;
    
    // 1. 实例化：构造方法
    public MyBean() {
        System.out.println("1. 构造方法");
    }
    
    // 2. 属性赋值：Setter 方法
    @Autowired
    public void setName(@Value("${app.name}") String name) {
        System.out.println("2. 属性注入: " + name);
        this.name = name;
    }
    
    // 3. 初始化回调方式一：@PostConstruct（推荐）
    @PostConstruct
    public void init() {
        System.out.println("3. @PostConstruct 初始化");
    }
    
    // 3. 初始化回调方式二：实现 InitializingBean 接口
    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("4. InitializingBean.afterPropertiesSet()");
    }
    
    // 3. 初始化回调方式三：@Bean 的 initMethod
    // @Bean(initMethod = "customInit")
    public void customInit() {
        System.out.println("5. 自定义初始化方法");
    }
    
    // 5. 销毁回调方式一：@PreDestroy（推荐）
    @PreDestroy
    public void destroy2() {
        System.out.println("销毁: @PreDestroy");
    }
    
    // 5. 销毁回调方式二：实现 DisposableBean 接口
    @Override
    public void destroy() throws Exception {
        System.out.println("销毁: DisposableBean.destroy()");
    }
}
```

---

## AOP 面向切面编程

### 什么是 AOP？

**面向切面编程（Aspect-Oriented Programming）** 将横切关注点（如日志、事务、安全）从业务逻辑中分离出来。

**为什么需要 AOP？**

```java
// 没有 AOP：业务代码混杂了日志、事务等非业务逻辑
public void transfer(String from, String to, double amount) {
    // 日志记录
    log.info("开始转账：从 {} 转账 {} 到 {}", from, amount, to);
    
    // 事务开始
    TransactionStatus status = transactionManager.getTransaction();
    try {
        // 实际业务逻辑（只有这几行是核心）
        accountDao.debit(from, amount);
        accountDao.credit(to, amount);
        
        // 事务提交
        transactionManager.commit(status);
        log.info("转账成功");
    } catch (Exception e) {
        transactionManager.rollback(status);
        log.error("转账失败", e);
        throw e;
    }
}

// 使用 AOP：业务代码只关注核心逻辑
@Transactional  // 事务由 AOP 自动处理
public void transfer(String from, String to, double amount) {
    accountDao.debit(from, amount);
    accountDao.credit(to, amount);
}
// 日志由 AOP 切面统一处理
```

### AOP 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **切面（Aspect）** | 横切关注点的模块化封装 | 日志处理模块 |
| **连接点（JoinPoint）** | 程序执行的特定点 | 方法的调用 |
| **切点（Pointcut）** | 匹配连接点的表达式 | 哪些方法需要增强 |
| **通知（Advice）** | 在切点执行的动作 | 执行日志记录 |
| **目标对象（Target）** | 被通知的对象 | UserService |
| **代理（Proxy）** | AOP 框架创建的对象 | 增强后的 UserService |

### 通知类型

```java
@Aspect
@Component
public class LoggingAspect {
    
    // 切点表达式：定义哪些方法需要被增强
    // execution(修饰符 返回值 包.类.方法(参数))
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayer() {}  // 切点方法，空实现，供引用
    
    // ========== 前置通知 ==========
    // 在目标方法执行前执行
    @Before("serviceLayer()")
    public void beforeMethod(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        System.out.println("【前置通知】方法 " + methodName + " 开始执行，参数：" + Arrays.toString(args));
    }
    
    // ========== 后置通知 ==========
    // 在目标方法执行后执行（无论是否异常）
    @After("serviceLayer()")
    public void afterMethod(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        System.out.println("【后置通知】方法 " + methodName + " 执行结束");
    }
    
    // ========== 返回通知 ==========
    // 在目标方法成功返回后执行
    @AfterReturning(pointcut = "serviceLayer()", returning = "result")
    public void afterReturning(JoinPoint joinPoint, Object result) {
        String methodName = joinPoint.getSignature().getName();
        System.out.println("【返回通知】方法 " + methodName + " 返回值：" + result);
    }
    
    // ========== 异常通知 ==========
    // 在目标方法抛出异常后执行
    @AfterThrowing(pointcut = "serviceLayer()", throwing = "ex")
    public void afterThrowing(JoinPoint joinPoint, Exception ex) {
        String methodName = joinPoint.getSignature().getName();
        System.out.println("【异常通知】方法 " + methodName + " 抛出异常：" + ex.getMessage());
    }
    
    // ========== 环绕通知 ==========
    // 最强大的通知，可以控制方法执行的全部过程
    @Around("serviceLayer()")
    public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();
        
        System.out.println("【环绕通知-前】方法 " + methodName + " 开始执行");
        long start = System.currentTimeMillis();
        
        try {
            // 执行目标方法
            Object result = joinPoint.proceed();
            
            long end = System.currentTimeMillis();
            System.out.println("【环绕通知-后】方法 " + methodName + " 执行成功，耗时：" + (end - start) + "ms");
            
            return result;  // 返回目标方法的返回值
        } catch (Exception e) {
            System.out.println("【环绕通知-异常】方法 " + methodName + " 抛出异常：" + e.getMessage());
            throw e;  // 重新抛出异常
        }
    }
}
```

### 切点表达式详解

```java
// execution 是最常用的切点表达式
// execution(修饰符? 返回值 包.类.方法(参数) 异常?)

// ========== 匹配所有 public 方法 ==========
execution(public * *(..))

// ========== 匹配特定类的所有方法 ==========
execution(* com.example.service.UserService.*(..))

// ========== 匹配特定包下所有类的所有方法 ==========
execution(* com.example.service.*.*(..))

// ========== 匹配特定包及其子包下所有方法 ==========
execution(* com.example..*.*(..))

// ========== 匹配特定参数的方法 ==========
execution(* com.example.service.*.find*(Long, ..))  // 第一个参数是 Long

// ========== 其他切点表达式 ==========
// 匹配带特定注解的方法
@annotation(com.example.annotation.Log)

// 匹配带特定注解的类
@within(com.example.annotation.Service)

// 匹配特定包下的类
within(com.example.service.*)

// 匹配实现了特定接口的类
this(com.example.service.BaseService)

// 组合切点表达式（&&、||、!）
@Pointcut("execution(* com.example.service.*.*(..)) && @annotation(log)")
public void serviceMethodWithLog(Log log) {}
```

### AOP 实战：日志切面

```java
// 自定义日志注解
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LogAnnotation {
    String value() default "";      // 操作描述
    boolean printArgs() default true;  // 是否打印参数
    boolean printResult() default true; // 是否打印返回值
}

// 日志切面
@Aspect
@Component
@Slf4j  // Lombok 注解，自动生成 log 对象
public class LogAspect {
    
    @Around("@annotation(logAnnotation)")
    public Object logExecution(ProceedingJoinPoint joinPoint, LogAnnotation logAnnotation) throws Throwable {
        // 获取方法信息
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        
        // 打印参数
        if (logAnnotation.printArgs()) {
            Object[] args = joinPoint.getArgs();
            log.info("{}.{} 开始执行, 参数: {}", className, methodName, Arrays.toString(args));
        }
        
        long startTime = System.currentTimeMillis();
        try {
            // 执行目标方法
            Object result = joinPoint.proceed();
            
            long elapsed = System.currentTimeMillis() - startTime;
            log.info("{}.{} 执行成功, 耗时: {}ms", className, methodName, elapsed);
            
            // 打印返回值
            if (logAnnotation.printResult() && result != null) {
                log.debug("返回值: {}", result);
            }
            
            return result;
        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - startTime;
            log.error("{}.{} 执行失败, 耗时: {}ms, 异常: {}", 
                className, methodName, elapsed, e.getMessage());
            throw e;
        }
    }
}

// 使用注解
@Service
public class UserService {
    @LogAnnotation(value = "查询用户", printArgs = true)
    public User findById(Long id) {
        return userDao.findById(id);
    }
    
    @LogAnnotation(value = "创建用户", printResult = false)
    public User create(UserDTO dto) {
        return userDao.save(dto);
    }
}
```

---

## 常见问题与注意事项

### 循环依赖

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

// 解决方案一：使用 @Lazy 延迟加载
@Service
public class ServiceA {
    @Autowired
    @Lazy  // 延迟加载，打破循环
    private ServiceB serviceB;
}

// 解决方案二：重构设计，消除循环依赖（推荐）
// 将公共部分提取到第三个 Service 中
```

### Bean 命名

```java
// 默认 Bean 名称：类名首字母小写
@Service
public class UserServiceImpl {}  // beanName = "userServiceImpl"

// 指定 Bean 名称
@Service("userService")
public class UserServiceImpl {}  // beanName = "userService"

// 多个实现时指定名称
@Service("userServiceA")
public class UserServiceA implements UserService {}

@Service("userServiceB")
public class UserServiceB implements UserService {}

// 使用时指定
@Autowired
@Qualifier("userServiceA")
private UserService userService;
```

### 代理限制

```java
@Service
public class OrderService {
    
    // ⚠️ 问题：同类内部调用，AOP 不生效
    public void placeOrder(Order order) {
        this.validateOrder(order);  // 直接调用，不走代理！
        // @Transactional、@Async 等注解都不生效
    }
    
    @Transactional
    public void validateOrder(Order order) {
        // 事务不生效
    }
}

// 解决方案一：注入自身
@Service
public class OrderService {
    @Autowired
    private OrderService self;  // 注入自身代理
    
    public void placeOrder(Order order) {
        self.validateOrder(order);  // 通过代理调用
    }
}

// 解决方案二：使用 AopContext
@Service
public class OrderService {
    public void placeOrder(Order order) {
        OrderService proxy = (OrderService) AopContext.currentProxy();
        proxy.validateOrder(order);
    }
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **IoC** | 控制反转，对象的创建和管理交给 Spring |
| **DI** | 依赖注入，Spring 自动注入依赖对象 |
| **Bean** | Spring 容器管理的对象 |
| **ApplicationContext** | Spring 容器，管理 Bean 的生命周期 |
| **AOP** | 面向切面编程，分离横切关注点 |
| **切面** | 横切关注点的模块化封装 |
| **通知** | 切面在切点执行的动作 |

### 最佳实践

1. **优先使用注解配置**：比 XML 更简洁
2. **使用构造器注入**：保证依赖不为空
3. **避免循环依赖**：重构设计
4. **Bean 设计为无状态**：单例 Bean 避免实例变量存储状态
5. **理解 AOP 代理机制**：内部调用不走代理