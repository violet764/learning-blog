# Spring 框架面试题

> 本文档整理 Spring 框架相关的高频面试题。

---

## 一、Spring 核心

### 1. 什么是 IOC 和 DI？

**答案：**

**IOC（控制反转）：**
- 传统方式：对象自己创建依赖
- IOC：对象的创建权交给 Spring 容器

**DI（依赖注入）：**
- Spring 在创建对象时，自动注入其依赖

```java
// 传统方式：自己创建依赖
public class UserService {
    private UserDao userDao = new UserDaoImpl();  // 耦合
}

// Spring 方式：依赖注入
@Service
public class UserService {
    @Autowired
    private UserDao userDao;  // 解耦，由 Spring 注入
}
```

**追问：IOC 容器初始化过程？**

```
1. Resource 定位：找到配置文件/注解
2. BeanDefinition 加载：解析配置，生成 BeanDefinition
3. BeanDefinition 注册：存入 BeanDefinitionRegistry
4. Bean 实例化：调用构造方法
5. 属性填充：注入依赖
6. 初始化：执行 @PostConstruct、afterPropertiesSet 等
7. 注册到单例池：放入 singletonObjects
```

---

### 2. Bean 的生命周期？

**答案：**

```
1. 实例化（Instantiation）
   └── 调用构造方法创建对象

2. 属性赋值（Populate）
   └── 注入依赖（@Autowired、@Value）

3. 初始化（Initialization）
   ├── 处理 Aware 接口
   ├── BeanPostProcessor.postProcessBeforeInitialization()
   ├── @PostConstruct
   ├── InitializingBean.afterPropertiesSet()
   ├── init-method
   └── BeanPostProcessor.postProcessAfterInitialization()
       └── AOP 代理在此创建

4. 使用（Ready）
   └── Bean 可以正常使用

5. 销毁（Destruction）
   ├── @PreDestroy
   ├── DisposableBean.destroy()
   └── destroy-method
```

**代码示例：**

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    @PostConstruct
    public void init() {
        System.out.println("1. @PostConstruct");
    }
    
    @Override
    public void afterPropertiesSet() {
        System.out.println("2. afterPropertiesSet");
    }
    
    @PreDestroy
    public void preDestroy() {
        System.out.println("3. @PreDestroy");
    }
    
    @Override
    public void destroy() {
        System.out.println("4. destroy");
    }
}
```

---

### 3. Spring 用了哪些设计模式？

**答案：**

| 模式 | 应用场景 |
|------|----------|
| 工厂模式 | BeanFactory 创建 Bean |
| 单例模式 | Bean 默认单例 |
| 代理模式 | AOP 实现 |
| 模板方法模式 | JdbcTemplate |
| 观察者模式 | ApplicationEvent |
| 策略模式 | Resource 加载 |
| 适配器模式 | HandlerAdapter |
| 装饰器模式 | BeanWrapper |

**追问：Spring 如何实现单例？**

```java
// DefaultSingletonBeanRegistry
private final Map<String, Object> singletonObjects = new ConcurrentHashMap<>();

public Object getSingleton(String beanName) {
    Object singletonObject = this.singletonObjects.get(beanName);
    if (singletonObject == null) {
        singletonObject = createBean(beanName);
        this.singletonObjects.put(beanName, singletonObject);
    }
    return singletonObject;
}
```

---

### 4. Spring 如何解决循环依赖？

**答案：**

```
循环依赖：A 依赖 B，B 依赖 A

解决方式：三级缓存

┌─────────────────────────────────────────────────┐
│ 一级缓存：singletonObjects                       │
│ 完整的单例 Bean                                   │
├─────────────────────────────────────────────────┤
│ 二级缓存：earlySingletonObjects                  │
│ 早期曝光的 Bean（未完成属性注入）                  │
├─────────────────────────────────────────────────┤
│ 三级缓存：singletonFactories                     │
│ Bean 工厂，用于生成早期 Bean 的代理对象            │
└─────────────────────────────────────────────────┘

解决流程：
1. 创建 A，暴露到三级缓存
2. 注入 B，创建 B
3. B 注入 A，从三级缓存获取 A（早期引用）
4. B 完成创建
5. A 完成创建
```

**追问：哪些循环依赖无法解决？**

> 1. 构造器注入的循环依赖（无法实例化）
> 2. @Async 注解的 Bean（代理时机问题）
> 3. prototype 作用域的 Bean

---

## 二、AOP

### 5. AOP 的实现原理？

**答案：**

AOP 基于**动态代理**实现：

```
有接口：JDK 动态代理
无接口：CGLIB 代理
```

**JDK 动态代理：**

```java
public class JdkProxy implements InvocationHandler {
    private Object target;
    
    public Object bind(Object target) {
        this.target = target;
        return Proxy.newProxyInstance(
            target.getClass().getClassLoader(),
            target.getClass().getInterfaces(),
            this
        );
    }
    
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        // 前置通知
        Object result = method.invoke(target, args);
        // 后置通知
        return result;
    }
}
```

**CGLIB 代理：**

```java
public class CglibProxy implements MethodInterceptor {
    public Object getProxy(Class<?> clazz) {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(clazz);
        enhancer.setCallback(this);
        return enhancer.create();
    }
    
    @Override
    public Object intercept(Object obj, Method method, Object[] args, MethodProxy proxy) throws Throwable {
        // 前置通知
        Object result = proxy.invokeSuper(obj, args);
        // 后置通知
        return result;
    }
}
```

**追问：JDK 动态代理和 CGLIB 的区别？**

| 特性 | JDK 动态代理 | CGLIB |
|------|--------------|-------|
| 要求 | 必须实现接口 | 无要求 |
| 原理 | 反射 | 继承子类 |
| 性能 | 生成快，执行慢 | 生成慢，执行快 |
| 限制 | 只能代理接口方法 | 不能代理 final 方法 |

---

### 6. AOP 的通知类型？

**答案：**

```java
@Aspect
@Component
public class MyAspect {
    
    // 前置通知
    @Before("execution(* com.example.service.*.*(..))")
    public void before(JoinPoint joinPoint) {
        System.out.println("方法执行前");
    }
    
    // 后置通知（无论是否异常）
    @After("execution(* com.example.service.*.*(..))")
    public void after() {
        System.out.println("方法执行后");
    }
    
    // 返回通知（方法成功返回）
    @AfterReturning(pointcut = "execution(* com.example.service.*.*(..))", returning = "result")
    public void afterReturning(Object result) {
        System.out.println("方法返回值：" + result);
    }
    
    // 异常通知
    @AfterThrowing(pointcut = "execution(* com.example.service.*.*(..))", throwing = "ex")
    public void afterThrowing(Exception ex) {
        System.out.println("方法异常：" + ex.getMessage());
    }
    
    // 环绕通知
    @Around("execution(* com.example.service.*.*(..))")
    public Object around(ProceedingJoinPoint joinPoint) throws Throwable {
        System.out.println("环绕前");
        Object result = joinPoint.proceed();  // 执行目标方法
        System.out.println("环绕后");
        return result;
    }
}
```

---

## 三、Spring 事务

### 7. Spring 事务传播机制？

**答案：**

| 传播行为 | 说明 |
|----------|------|
| REQUIRED（默认） | 有事务就加入，没有就新建 |
| REQUIRES_NEW | 总是新建事务，挂起当前事务 |
| SUPPORTS | 有事务就加入，没有就以非事务运行 |
| NOT_SUPPORTED | 以非事务运行，挂起当前事务 |
| MANDATORY | 必须在事务中，否则抛异常 |
| NEVER | 不能在事务中，否则抛异常 |
| NESTED | 有事务就创建嵌套事务 |

```java
@Service
public class OrderService {
    
    @Transactional(propagation = Propagation.REQUIRED)
    public void createOrder() {
        // 有事务就加入，没有就新建
    }
    
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void logOperation() {
        // 总是新事务，不影响外层事务
    }
}
```

---

### 8. Spring 事务失效的场景？

**答案：**

```
1. 方法不是 public
   @Transactional 只对 public 方法有效

2. 同类中自调用
   @Service
   public class UserService {
       public void methodA() {
           this.methodB();  // 绕过代理，事务失效
       }
       
       @Transactional
       public void methodB() { }
   }

3. 异常被 catch 吃掉
   @Transactional
   public void method() {
       try {
           // 业务代码
       } catch (Exception e) {
           // 异常被捕获，事务不回滚
       }
   }

4. 抛出 checked 异常
   @Transactional  // 默认只回滚 RuntimeException
   public void method() throws IOException {
       throw new IOException();  // 不回滚
   }
   
   // 解决方案
   @Transactional(rollbackFor = Exception.class)

5. 数据库引擎不支持事务
   MyISAM 不支持事务

6. 类没被 Spring 管理
   没有 @Service 等注解
```

---

## 四、Spring Boot

### 9. Spring Boot 自动配置原理？

**答案：**

```
@SpringBootApplication = 
    @SpringBootConfiguration      // 配置类
    + @EnableAutoConfiguration    // 开启自动配置
    + @ComponentScan              // 组件扫描

自动配置流程：
1. @EnableAutoConfiguration 导入 AutoConfigurationImportSelector
2. 扫描 META-INF/spring.factories 文件
3. 加载所有 EnableAutoConfiguration 配置类
4. 根据 @Conditional 条件判断是否生效
5. 生效的配置类注册 Bean
```

**条件注解：**

| 注解 | 条件 |
|------|------|
| @ConditionalOnClass | 类路径存在某类 |
| @ConditionalOnMissingClass | 类路径不存在某类 |
| @ConditionalOnBean | 容器中存在某 Bean |
| @ConditionalOnMissingBean | 容器中不存在某 Bean |
| @ConditionalOnProperty | 配置属性满足条件 |
| @ConditionalOnWebApplication | 是 Web 应用 |

---

### 10. Spring Boot 启动流程？

**答案：**

```
SpringApplication.run()
        │
        ▼
1. 创建 SpringApplication 对象
   ├── 判断应用类型（Servlet/Reactive）
   ├── 加载初始化器（ApplicationContextInitializer）
   └── 加载监听器（ApplicationListener）
        │
        ▼
2. 执行 run() 方法
   ├── 准备环境（Environment）
   ├── 打印 Banner
   ├── 创建 ApplicationContext
   ├── 预处理 Context
   ├── 刷新 Context（核心）
   │   ├── 执行 BeanFactoryPostProcessor
   │   ├── 注册 BeanPostProcessor
   │   ├── 初始化单例 Bean
   │   └── 完成刷新
   └── 发布启动完成事件
```

---

## 五、其他问题

### 11. @Autowired 和 @Resource 的区别？

**答案：**

| 特性 | @Autowired | @Resource |
|------|------------|-----------|
| 来源 | Spring | JDK（JSR-250） |
| 注入方式 | 默认按类型 | 默认按名称 |
| 指定名称 | @Qualifier | name 属性 |
| 必须注入 | required 属性 | 无 |
| 通用性 | 仅 Spring | 标准 Java |

```java
@Autowired
@Qualifier("userService")
private UserService userService;

@Resource(name = "userService")
private UserService userService;
```

---

### 12. Bean 的作用域？

**答案：**

| 作用域 | 说明 |
|--------|------|
| singleton | 单例（默认） |
| prototype | 每次获取创建新实例 |
| request | 每个 HTTP 请求一个实例 |
| session | 每个 HTTP 会话一个实例 |
| application | ServletContext 生命周期 |
| websocket | WebSocket 会话一个实例 |

```java
@Component
@Scope("prototype")
public class PrototypeBean { }
```

---

### 13. @Component 和 @Bean 的区别？

**答案：**

| 特性 | @Component | @Bean |
|------|------------|-------|
| 使用位置 | 类上 | 方法上 |
| 控制权 | Spring 自动创建 | 手动创建 |
| 适用场景 | 自己的类 | 第三方类 |

```java
@Component
public class MyComponent { }

@Configuration
public class AppConfig {
    @Bean
    public DataSource dataSource() {
        return new HikariDataSource();  // 第三方类
    }
}
```

---

## 小结

本文档涵盖了 Spring 框架面试的高频考点：

- IOC 和 DI 原理
- Bean 生命周期
- AOP 实现原理
- 事务传播与失效
- Spring Boot 自动配置
- 常用注解区别
