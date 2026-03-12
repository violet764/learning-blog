# Java 常见面试题与答案

> 本文档整理了 Java 开发岗位常见面试题，包含标准答案和面试官可能追问的内容。
> 
> **更多详细的面试题分类**，请访问 [面试专栏](./interview/index.md)，包含：
> - [Java 基础面试题](./interview/java-basics.md)
> - [Java 集合面试题](./interview/java-collections.md)
> - [Java 并发面试题](./interview/java-concurrent.md)
> - [JVM 面试题](./interview/java-jvm.md)
> - [Spring 框架面试题](./interview/spring.md)
> - [MySQL 面试题](./interview/mysql.md)
> - [Redis 面试题](./interview/redis.md)

---

## 一、Java 基础

### 1. Java 有哪些基本数据类型？占用多少字节？

**答案：**

| 类型 | 关键字 | 字节 | 取值范围 | 默认值 |
|------|--------|:----:|----------|--------|
| 字节型 | byte | 1 | -128 ~ 127 | 0 |
| 短整型 | short | 2 | -32768 ~ 32767 | 0 |
| 整型 | int | 4 | -2³¹ ~ 2³¹-1 | 0 |
| 长整型 | long | 8 | -2⁶³ ~ 2⁶³-1 | 0L |
| 单精度浮点 | float | 4 | ±3.4E38 | 0.0f |
| 双精度浮点 | double | 8 | ±1.7E308 | 0.0d |
| 字符型 | char | 2 | 0 ~ 65535 | '\u0000' |
| 布尔型 | boolean | 1/4 | true/false | false |

**追问：为什么 byte 类型是 -128 到 127？**

> byte 用 8 位二进制表示，最高位是符号位。
> 正数范围：00000000 ~ 01111111（0 ~ 127）
> 负数用补码表示：10000000 ~ 11111111
> 其中 10000000 表示 -128（这是补码的特殊情况）
> 所以范围是 -128 ~ 127，共 256 个值

---

### 2. == 和 equals() 的区别？

**答案：**

- **==**：比较的是内存地址（引用是否指向同一对象）；对于基本类型比较的是值
- **equals()**：默认也是比较地址，但可以被重写来比较内容

```java
String s1 = new String("hello");
String s2 = new String("hello");
String s3 = "hello";
String s4 = "hello";

System.out.println(s1 == s2);      // false，不同对象
System.out.println(s1.equals(s2)); // true，内容相同
System.out.println(s3 == s4);      // true，字符串常量池中同一对象
System.out.println(s1 == s3);      // false，堆 vs 常量池
```

**追问：hashCode() 和 equals() 的关系？**

> 1. 如果两个对象 equals() 相等，hashCode() 必须相等
> 2. 如果 hashCode() 相等，equals() 不一定相等（哈希冲突）
> 3. 重写 equals() 必须同时重写 hashCode()
> 4. 原因：HashMap、HashSet 等集合先比较 hashCode，再比较 equals

**追问：为什么重写 equals 必须重写 hashCode？**

```java
// 假设只重写 equals，没重写 hashCode
Person p1 = new Person("张三", 25);
Person p2 = new Person("张三", 25);

p1.equals(p2);  // true（重写了 equals）

HashSet<Person> set = new HashSet<>();
set.add(p1);
set.add(p2);    // p1 和 p2 hashCode 不同，都会被添加进去！
// 这就破坏了 Set 不重复的特性
```

---

### 3. String、StringBuilder、StringBuffer 的区别？

**答案：**

| 特性 | String | StringBuilder | StringBuffer |
|------|--------|---------------|--------------|
| 可变性 | 不可变 | 可变 | 可变 |
| 线程安全 | 安全（不可变） | 不安全 | 安全（synchronized） |
| 性能 | 低（每次创建新对象） | 高 | 中等 |
| 适用场景 | 少量字符串操作 | 单线程大量操作 | 多线程大量操作 |

```java
// String 不可变性示例
String s = "hello";
s = s + " world";  // 创建了新的 String 对象，原对象等待 GC

// StringBuilder 示例
StringBuilder sb = new StringBuilder("hello");
sb.append(" world");  // 在原对象上修改，无新对象
```

**追问：String 为什么设计为不可变？**

> 1. **安全性**：String 常用于存储敏感信息（如数据库连接字符串），不可变防止被篡改
> 2. **哈希缓存**：hashCode 可以缓存，提高 HashMap 等集合性能
> 3. **字符串常量池**：多个引用可指向同一字符串，节省内存
> 4. **线程安全**：不可变对象天然线程安全

**追问：String s = new String("abc") 创建了几个对象？**

> 分情况：
> - 如果常量池中已有 "abc"：创建 1 个对象（堆中的 String 对象）
> - 如果常量池中没有 "abc"：创建 2 个对象（常量池 + 堆各一个）

---

### 4. 重载（Overload）和重写（Override）的区别？

**答案：**

| 特性 | 重载 | 重写 |
|------|------|------|
| 发生位置 | 同一个类中 | 子父类之间 |
| 方法签名 | 方法名相同，参数不同 | 方法名、参数都相同 |
| 返回类型 | 无关 | 必须相同或协变返回类型 |
| 访问修饰符 | 无关 | 不能更严格 |
| 异常 | 无关 | 不能抛出更广的异常 |

```java
// 重载示例：方法名相同，参数不同
public void print(int a) { }
public void print(String s) { }
public void print(int a, int b) { }

// 重写示例：子类重新定义父类方法
class Animal {
    public void speak() { System.out.println("动物叫声"); }
}
class Dog extends Animal {
    @Override
    public void speak() { System.out.println("汪汪汪"); }
}
```

**追问：重写时可以修改返回类型吗？**

> 可以，但必须是协变返回类型（返回类型可以是父类方法返回类型的子类）
> ```java
> class Parent { public Animal getAnimal() { return new Animal(); } }
> class Child extends Parent {
>     @Override
>     public Dog getAnimal() { return new Dog(); }  // Dog 是 Animal 子类
> }
> ```

---

### 5. 接口和抽象类的区别？

**答案：**

| 特性 | 接口 | 抽象类 |
|------|------|--------|
| 关键字 | interface | abstract class |
| 多继承 | 一个类可实现多个接口 | 只能单继承 |
| 成员变量 | 只能是 public static final | 可以有各种类型 |
| 构造方法 | 不能有 | 可以有 |
| 方法 | JDK8 前只能抽象方法 | 可以有抽象和具体方法 |
| 设计理念 | 定义行为规范 | 代码复用 + 模板设计 |

**追问：JDK8 接口新增了哪些特性？**

> 1. **default 方法**：可以有默认实现
> 2. **static 方法**：可以有静态方法
> ```java
> interface MyInterface {
>     // 抽象方法
>     void abstractMethod();
>     
>     // 默认方法（JDK8）
>     default void defaultMethod() {
>         System.out.println("默认实现");
>     }
>     
>     // 静态方法（JDK8）
>     static void staticMethod() {
>         System.out.println("静态方法");
>     }
> }
> ```

**追问：什么时候用接口，什么时候用抽象类？**

> - **用接口**：定义能力、行为规范（如 Comparable、Serializable）
> - **用抽象类**：有共同属性和方法，需要代码复用（如 AbstractList）

---

### 6. Java 异常体系是怎样的？

**答案：**

```
                    Throwable
                       │
          ┌────────────┴────────────┐
          │                         │
       Error                     Exception
          │                         │
   (程序无法处理)          ┌─────────┴─────────┐
                          │                   │
                  Checked异常            Unchecked异常
                  (编译时检查)          (RuntimeException)
                          │                   │
                   IOException         NullPointerException
                   SQLException        ArrayIndexOutOfBoundsException
                   ClassNotFoundException  ArithmeticException
```

- **Error**：JVM 无法处理的严重错误（如 OutOfMemoryError）
- **Checked Exception**：必须处理（try-catch 或 throws）
- **Unchecked Exception**：运行时异常，可选处理

**追问：finally 块一定会执行吗？**

> 几乎一定会执行，除了：
> 1. 在 try/catch 块中调用了 System.exit()
> 2. 线程死亡
> 3. CPU 关机

**追问：try-with-resources 是什么？**

> JDK7 引入的自动资源管理语法，实现 AutoCloseable 接口的资源会自动关闭
> ```java
> // 传统方式
> FileInputStream fis = null;
> try {
>     fis = new FileInputStream("file.txt");
>     // 使用资源
> } finally {
>     if (fis != null) fis.close();
> }
> 
> // try-with-resources（推荐）
> try (FileInputStream fis = new FileInputStream("file.txt")) {
>     // 使用资源，自动关闭
> }
> ```

---

## 二、集合框架

### 1. ArrayList 和 LinkedList 的区别？

**答案：**

| 特性 | ArrayList | LinkedList |
|------|-----------|------------|
| 底层结构 | 动态数组 | 双向链表 |
| 随机访问 | O(1) | O(n) |
| 头部插入/删除 | O(n) | O(1) |
| 尾部插入/删除 | O(1) 均摊 | O(1) |
| 中间插入/删除 | O(n) | O(n) |
| 内存占用 | 连续，较小 | 分散，较大（存指针） |
| 适用场景 | 查询多，增删少 | 增删多，查询少 |

**追问：ArrayList 扩容机制是怎样的？**

> 1. 默认初始容量 10
> 2. 添加元素时检查容量，不够则扩容
> 3. 扩容为原容量的 1.5 倍（oldCapacity + oldCapacity >> 1）
> 4. 调用 Arrays.copyOf 复制到新数组

**追问：为什么 ArrayList 增删慢还要用它？**

> 1. 实际开发中，遍历操作远多于增删
> 2. CPU 缓存友好（连续内存）
> 3. 扩容是均摊成本，尾部插入实际是 O(1)
> 4. LinkedList 作者自己都说不用 LinkedList

---

### 2. HashMap 的底层实现原理？

**答案：**

**JDK 1.7：数组 + 链表**
**JDK 1.8：数组 + 链表 + 红黑树**

```java
// 基本结构
Node<K,V>[] table;  // 哈希桶数组

static class Node<K,V> {
    final int hash;
    final K key;
    V value;
    Node<K,V> next;  // 链表下一个节点
}
```

**put 操作流程：**
1. 计算 key 的 hash 值，确定数组下标
2. 如果该位置为空，直接插入
3. 如果有元素，遍历链表/红黑树
4. 如果 key 相同，覆盖 value
5. 如果不同，插入新节点
6. 链表长度 ≥ 8 且数组长度 ≥ 64 时，链表转红黑树

**追问：为什么 JDK 1.8 要引入红黑树？**

> 链表过长时（hash 冲突严重），查询效率从 O(1) 退化为 O(n)
> 红黑树查询效率为 O(log n)，提升性能

**追问：HashMap 的扩容机制是怎样的？**

> 1. 默认初始容量 16，负载因子 0.75
> 2. 当 size > capacity × loadFactor 时扩容
> 3. 扩容为原来的 2 倍
> 4. 重新计算每个元素的位置（rehash）

**追问：为什么容量是 2 的幂次方？**

> 计算下标时用 `(n - 1) & hash`，n 为 2 的幂次方时，n-1 的二进制全是 1
> 这样 & 运算能充分利用 hash 值的每一位，减少碰撞

**追问：HashMap 线程安全吗？有什么替代方案？**

> HashMap 不是线程安全的
> 替代方案：
> 1. **ConcurrentHashMap**：推荐，性能好
> 2. **Collections.synchronizedMap()**：包装成线程安全
> 3. **Hashtable**：老旧，性能差，不推荐

---

### 3. ConcurrentHashMap 是如何保证线程安全的？

**答案：**

**JDK 1.7：分段锁（Segment）**
- 将数据分成多个段，每段一把锁
- 默认 16 个 Segment，并发度 16

**JDK 1.8：CAS + synchronized**
- 锁粒度更细，锁单个节点
- 使用 CAS 进行无锁插入
- synchronized 用于链表/红黑树操作

```java
// JDK 1.8 put 操作大致流程
final V putVal(K key, V value, boolean onlyIfAbsent) {
    // 1. 计算 hash
    int hash = spread(key.hashCode());
    
    // 2. 遍历 table
    for (Node<K,V>[] tab = table;;) {
        Node<K,V> f; int n, i, fh;
        
        // 3. 如果位置为空，CAS 插入
        if (tab == null || (n = tab.length) == 0)
            tab = initTable();
        else if ((f = tabAt(tab, i = (n - 1) & hash)) == null) {
            if (casTabAt(tab, i, null, new Node<K,V>(hash, key, value, null)))
                break;  // CAS 成功
        }
        // 4. 如果在扩容，帮助扩容
        else if ((fh = f.hash) == MOVED)
            tab = helpTransfer(tab, f);
        // 5. 否则加锁插入
        else {
            synchronized (f) {
                // 插入或更新
            }
        }
    }
    return null;
}
```

**追问：为什么 JDK 1.8 不用分段锁了？**

> 1. 分段锁内存占用大
> 2. 锁粒度不够细
> 3. synchronized 在 JDK 1.6 后性能优化很多
> 4. CAS 无锁操作性能更好

---

### 4. HashSet 如何保证元素唯一？

**答案：**

HashSet 底层就是 HashMap，元素作为 key，value 是一个固定对象

```java
public class HashSet<E> {
    private transient HashMap<E,Object> map;
    private static final Object PRESENT = new Object();
    
    public boolean add(E e) {
        return map.put(e, PRESENT) == null;
    }
}
```

添加元素时：
1. 计算 hashCode
2. 如果 hashCode 相同，再调用 equals
3. 都相同则认为是重复元素，不添加

**追问：LinkedHashSet 如何保证有序？**

> LinkedHashSet 继承 HashSet，内部使用 LinkedHashMap
> LinkedHashMap 在 HashMap 基础上维护了一个双向链表，保证插入顺序

---

## 三、多线程与并发

### 1. 线程和进程的区别？

**答案：**

| 特性 | 进程 | 线程 |
|------|------|------|
| 定义 | 程序执行的最小单位 | CPU 调度的最小单位 |
| 资源 | 独立内存空间 | 共享进程内存 |
| 通信 | IPC（管道、消息队列等） | 直接读写共享变量 |
| 开销 | 大（创建、切换） | 小 |
| 稳定性 | 一个进程崩溃不影响其他 | 一个线程崩溃可能影响整个进程 |

**追问：创建线程有几种方式？**

> 1. **继承 Thread 类**
> ```java
> class MyThread extends Thread {
>     public void run() { System.out.println("线程执行"); }
> }
> new MyThread().start();
> ```
> 
> 2. **实现 Runnable 接口**（推荐）
> ```java
> class MyTask implements Runnable {
>     public void run() { System.out.println("线程执行"); }
> }
> new Thread(new MyTask()).start();
> ```
> 
> 3. **实现 Callable 接口**（有返回值）
> ```java
> class MyTask implements Callable<String> {
>     public String call() { return "结果"; }
> }
> FutureTask<String> task = new FutureTask<>(new MyTask());
> new Thread(task).start();
> String result = task.get();  // 获取返回值
> ```
> 
> 4. **线程池**（实际开发推荐）

---

### 2. 线程的生命周期（状态）？

**答案：**

```
                    ┌──────────────────┐
                    │                  │
                    ▼                  │
NEW ──start()──► RUNNABLE ──synchronized──► BLOCKED
                    │                        │
                    │ 获取锁                 │ 获取锁
                    ▼                        │
               WAITING/TIMED_WAITING ◄───────┘
                    │
                    │ run() 结束
                    ▼
                 TERMINATED
```

| 状态 | 说明 |
|------|------|
| NEW | 新建，未调用 start() |
| RUNNABLE | 可运行（就绪 + 运行中） |
| BLOCKED | 阻塞，等待获取锁 |
| WAITING | 等待，需要其他线程唤醒 |
| TIMED_WAITING | 超时等待，到时自动唤醒 |
| TERMINATED | 终止 |

---

### 3. synchronized 和 Lock 的区别？

**答案：**

| 特性 | synchronized | Lock |
|------|--------------|------|
| 层面 | JVM 关键字 | Java API |
| 获取锁 | 自动获取释放 | 手动 lock/unlock |
| 锁类型 | 可重入、非公平 | 可重入、公平/非公平可选 |
| 响应中断 | 不支持 | 支持 |
| 超时获取 | 不支持 | 支持 tryLock |
| 条件变量 | 单一 | 多个 Condition |
| 性能 | JDK 1.6 后优化，差距不大 | 高并发下略优 |

```java
// synchronized 用法
public synchronized void method() { }
public void method() {
    synchronized (this) { }
}

// Lock 用法
Lock lock = new ReentrantLock();
lock.lock();
try {
    // 临界区
} finally {
    lock.unlock();  // 必须手动释放
}
```

**追问：synchronized 的锁升级过程？**

> JDK 1.6 后 synchronized 优化，锁会升级但不降级：
> 
> 1. **无锁** → **偏向锁**：第一个线程访问时，在对象头记录线程 ID
> 2. **偏向锁** → **轻量级锁**：有竞争时，撤销偏向锁，用 CAS 获取锁
> 3. **轻量级锁** → **重量级锁**：CAS 失败多次，升级为重量级锁（阻塞等待）

**追问：ReentrantLock 如何实现公平锁？**

> 公平锁：先来先得，按请求顺序获取锁
> 非公平锁：可以插队，性能更好
> 
> ```java
> Lock fairLock = new ReentrantLock(true);   // 公平锁
> Lock unfairLock = new ReentrantLock(false); // 非公平锁（默认）
> ```

---

### 4. volatile 关键字的作用？

**答案：**

volatile 有两个主要作用：

**1. 保证可见性**
- 一个线程修改后，其他线程立即看到
- 通过内存屏障实现

**2. 禁止指令重排序**
- 防止 JVM 优化导致的问题
- 典型应用：单例模式双重检查

```java
// 单例模式双重检查
public class Singleton {
    private volatile static Singleton instance;
    
    public static Singleton getInstance() {
        if (instance == null) {              // 第一次检查
            synchronized (Singleton.class) {
                if (instance == null) {      // 第二次检查
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**追问：为什么需要 volatile？**

> `instance = new Singleton()` 不是原子操作，分三步：
> 1. 分配内存
> 2. 初始化对象
> 3. 将引用指向内存
> 
> JVM 可能重排序为 1→3→2，导致其他线程拿到未初始化的对象
> volatile 禁止重排序，保证有序性

**追问：volatile 能保证原子性吗？**

> 不能！volatile 只保证可见性和有序性
> ```java
> volatile int count = 0;
> count++;  // 不是原子操作，线程不安全
> ```
> 需要原子性时，使用 AtomicInteger 或 synchronized

---

### 5. 线程池的核心参数有哪些？

**答案：**

```java
public ThreadPoolExecutor(
    int corePoolSize,      // 核心线程数
    int maximumPoolSize,   // 最大线程数
    long keepAliveTime,    // 空闲线程存活时间
    TimeUnit unit,         // 时间单位
    BlockingQueue<Runnable> workQueue,  // 工作队列
    ThreadFactory threadFactory,        // 线程工厂
    RejectedExecutionHandler handler    // 拒绝策略
)
```

**任务执行流程：**
1. 线程数 < corePoolSize：创建新线程执行
2. 线程数 = corePoolSize：任务放入队列
3. 队列满了，线程数 < maximumPoolSize：创建非核心线程
4. 线程数 = maximumPoolSize：执行拒绝策略

**追问：线程池有哪些拒绝策略？**

| 策略 | 说明 |
|------|------|
| AbortPolicy | 抛出 RejectedExecutionException（默认） |
| CallerRunsPolicy | 由调用线程执行任务 |
| DiscardPolicy | 直接丢弃，不抛异常 |
| DiscardOldestPolicy | 丢弃队列最老任务，尝试重新提交 |

**追问：如何合理配置线程池参数？**

> - **CPU 密集型**：corePoolSize = CPU 核心数 + 1
> - **IO 密集型**：corePoolSize = CPU 核心数 × 2
> - **混合型**：根据 IO 等待时间占比调整

---

### 6. ThreadLocal 是什么？有什么问题？

**答案：**

ThreadLocal 提供线程局部变量，每个线程有独立的变量副本，互不干扰。

```java
// 典型应用：数据库连接、用户上下文
public class UserContext {
    private static ThreadLocal<User> userHolder = new ThreadLocal<>();
    
    public static void setUser(User user) {
        userHolder.set(user);
    }
    
    public static User getUser() {
        return userHolder.get();
    }
    
    public static void clear() {
        userHolder.remove();  // 必须清理！
    }
}
```

**追问：ThreadLocal 内存泄漏问题？**

> ThreadLocalMap 的 Entry 继承自 WeakReference，key 是弱引用
> - key（ThreadLocal）被 GC 回收 → value 还在
> - 如果线程不结束（线程池），value 无法回收 → 内存泄漏
> 
> 解决方案：使用完必须调用 remove() 方法

---

## 四、JVM

### 1. JVM 内存模型是怎样的？

**答案：**

```
┌─────────────────────────────────────────────┐
│                  JVM 运行时数据区              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │   方法区     │  │       堆             │  │
│  │ (Method Area)│  │      (Heap)         │  │
│  │  类信息、常量  │  │   对象实例、数组      │  │
│  │  静态变量     │  │                     │  │
│  └─────────────┘  └─────────────────────┘  │
│        ↓ 线程共享              ↓ 线程共享     │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │   虚拟机栈    │  │      本地方法栈      │  │
│  │ (VM Stack)   │  │ (Native Method Stack)│ │
│  │   栈帧、局部变量│  │    Native 方法      │  │
│  └─────────────┘  └─────────────────────┘  │
│        ↓ 线程私有              ↓ 线程私有     │
├─────────────────────────────────────────────┤
│              程序计数器                       │
│           (Program Counter)                 │
│              ↓ 线程私有                       │
└─────────────────────────────────────────────┘
```

| 区域 | 作用 | 线程安全 | 异常 |
|------|------|:--------:|------|
| 堆 | 存储对象实例 | 不安全 | OutOfMemoryError |
| 方法区 | 类信息、常量、静态变量 | 不安全 | OutOfMemoryError |
| 虚拟机栈 | 方法调用、局部变量 | 安全 | StackOverflowError、OOM |
| 本地方法栈 | Native 方法调用 | 安全 | StackOverflowError、OOM |
| 程序计数器 | 记录执行位置 | 安全 | 无 |

**追问：堆的内存结构？**

> JDK 1.8 后堆分为：
> - **新生代**（1/3）：Eden + Survivor0 + Survivor1
> - **老年代**（2/3）：存放长期存活对象
> 
> 新生代 GC（Minor GC）频繁，老年代 GC（Major/Full GC）较慢

---

### 2. 垃圾回收算法有哪些？

**答案：**

**1. 标记-清除（Mark-Sweep）**
- 标记需要回收的对象，然后清除
- 缺点：内存碎片、效率不高

**2. 复制算法（Copying）**
- 将内存分为两块，每次用一块
- 存活对象复制到另一块，清空当前块
- 优点：没有碎片
- 缺点：内存利用率低
- 适用：新生代（Eden:Survivor = 8:1）

**3. 标记-整理（Mark-Compact）**
- 标记存活对象，向一端移动
- 清理边界外的内存
- 优点：没有碎片
- 适用：老年代

**追问：如何判断对象可以被回收？**

> 1. **引用计数法**：有引用+1，引用失效-1，为 0 则回收
>    - 缺点：循环引用无法回收
> 
> 2. **可达性分析**（JVM 使用）：从 GC Roots 开始搜索
>    - GC Roots：栈中引用、静态变量、常量、本地方法栈引用等
>    - 不可达对象可回收

**追问：常见的垃圾收集器？**

| 收集器 | 类型 | 适用场景 |
|--------|------|----------|
| Serial | 单线程 + 复制 | 客户端模式、小内存 |
| Parallel Scavenge | 多线程 + 复制 | 吞吐量优先 |
| CMS | 并发 + 标记清除 | 响应时间优先 |
| G1 | 分区 + 复制+标记整理 | 大内存、低延迟 |
| ZGC | 并发 + 读屏障 | 超低延迟（< 10ms） |

---

### 3. 类加载过程是怎样的？

**答案：**

```
类加载过程：
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│ 加载    │ → │ 验证    │ → │ 准备    │ → │ 解析    │ → │ 初始化  │
│Loading │   │Verify  │   │Prepare │   │Resolve │   │Initialize│
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
```

1. **加载**：读取 class 文件，创建 Class 对象
2. **验证**：校验字节码正确性
3. **准备**：为静态变量分配内存，赋默认值
4. **解析**：符号引用转为直接引用
5. **初始化**：执行 `<clinit>` 方法，静态变量赋真实值

**追问：双亲委派模型是什么？**

> 类加载器层级：
> ```
> Bootstrap ClassLoader（启动类加载器）
>         ↑
> Extension ClassLoader（扩展类加载器）
>         ↑
> Application ClassLoader（应用类加载器）
>         ↑
>    Custom ClassLoader（自定义类加载器）
> ```
> 
> 工作流程：子类加载器先委托父类加载，父类无法加载才自己加载
> 
> 好处：
> 1. 避免类重复加载
> 2. 保护核心类（如 java.lang.String）不被篡改

**追问：如何打破双亲委派？**

> 自定义类加载器，重写 loadClass() 方法

---

## 五、Spring 框架

### 1. 什么是 IOC 和 DI？

**答案：**

**IOC（控制反转）**：对象的创建权交给 Spring 容器，而不是自己 new

**DI（依赖注入）**：Spring 在创建对象时，自动注入其依赖对象

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

> 1. **资源定位**：找到配置文件/注解
> 2. **BeanDefinition 加载**：解析配置，生成 BeanDefinition
> 3. **注册 BeanDefinition**：存入 BeanDefinitionRegistry
> 4. **实例化 Bean**：调用构造方法
> 5. **属性填充**：注入依赖
> 6. **初始化**：执行 @PostConstruct、afterPropertiesSet 等
> 7. **注册到单例池**：放入 singletonObjects

---

### 2. Spring Bean 的生命周期？

**答案：**

```
1. 实例化（Instantiation）
   └── 调用构造方法创建对象

2. 属性赋值（Populate）
   └── 注入依赖（@Autowired、@Value）

3. 初始化（Initialization）
   ├── 处理 Aware 接口（BeanNameAware、ApplicationContextAware）
   ├── BeanPostProcessor.postProcessBeforeInitialization()
   ├── @PostConstruct
   ├── InitializingBean.afterPropertiesSet()
   ├── init-method
   └── BeanPostProcessor.postProcessAfterInitialization()
       └── 此处创建 AOP 代理

4. 使用（Ready）
   └── Bean 可以正常使用

5. 销毁（Destruction）
   ├── @PreDestroy
   ├── DisposableBean.destroy()
   └── destroy-method
```

**追问：BeanPostProcessor 有什么用？**

> 对所有 Bean 进行后置处理
> 典型应用：
> - AOP 代理创建
> - @Autowired 注入
> - @Value 解析

---

### 3. Spring AOP 的实现原理？

**答案：**

AOP（面向切面编程）将通用逻辑从业务代码中抽离出来。

**实现方式：**
1. **JDK 动态代理**：目标类实现接口时使用
2. **CGLIB 代理**：目标类未实现接口时使用

```java
// 切面示例
@Aspect
@Component
public class LoggingAspect {
    
    @Before("execution(* com.example.service.*.*(..))")
    public void logBefore(JoinPoint joinPoint) {
        System.out.println("执行方法: " + joinPoint.getSignature().getName());
    }
    
    @AfterReturning(pointcut = "execution(* com.example.service.*.*(..))", returning = "result")
    public void logAfterReturning(Object result) {
        System.out.println("返回值: " + result);
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

### 4. Spring 事务传播行为有哪些？

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

**追问：REQUIRED 和 REQUIRES_NEW 的区别？**

> - **REQUIRED**：加入当前事务，一个方法失败，整个事务回滚
> - **REQUIRES_NEW**：独立新事务，互不影响
> 
> ```java
> @Transactional
> public void methodA() {
>     methodB();  // 如果 B 抛异常，A 也会回滚
> }
> 
> @Transactional(propagation = Propagation.REQUIRES_NEW)
> public void methodB() {
>     // 独立事务，A 回滚不影响 B
> }
> ```

**追问：事务失效的场景有哪些？**

> 1. 方法不是 public
> 2. 同类中自调用（绕过代理）
> 3. 异常被 catch 吃掉
> 4. 抛出 checked 异常（默认只回滚 RuntimeException）
> 5. 数据库引擎不支持事务

---

### 5. Spring Boot 自动配置原理？

**答案：**

**核心注解：@SpringBootApplication**

```java
@SpringBootApplication = 
    @SpringBootConfiguration      // 配置类
    + @EnableAutoConfiguration    // 开启自动配置
    + @ComponentScan              // 组件扫描
```

**@EnableAutoConfiguration 原理：**
1. 通过 @Import 导入 AutoConfigurationImportSelector
2. 扫描 META-INF/spring.factories 文件
3. 加载所有 EnableAutoConfiguration 配置类
4. 根据条件注解（@ConditionalOnXxx）决定是否生效

**追问：@Conditional 条件注解有哪些？**

| 注解 | 生效条件 |
|------|----------|
| @ConditionalOnClass | 类路径存在某类 |
| @ConditionalOnMissingClass | 类路径不存在某类 |
| @ConditionalOnBean | 容器中存在某 Bean |
| @ConditionalOnMissingBean | 容器中不存在某 Bean |
| @ConditionalOnProperty | 配置属性满足条件 |
| @ConditionalOnWebApplication | 是 Web 应用 |

---

## 六、MySQL 数据库

### 1. MySQL 索引的数据结构？

**答案：**

MySQL 使用 **B+树** 作为索引结构。

**B+树特点：**
1. 非叶子节点只存索引，不存数据
2. 叶子节点存所有数据，用指针连接成链表
3. 范围查询高效
4. 单个节点存更多索引，树更矮，磁盘 IO 更少

**追问：为什么不用 B 树、Hash、二叉树？**

| 结构 | 问题 |
|------|------|
| B 树 | 非叶子存数据，节点存索引少，树更高 |
| Hash | 不支持范围查询 |
| 二叉树 | 树太高，磁盘 IO 多 |
| 红黑树 | 同上 |

**追问：聚簇索引和非聚簇索引的区别？**

| 类型 | 说明 | 存储位置 |
|------|------|----------|
| 聚簇索引 | 主键索引，叶子存完整数据 | 数据文件 |
| 非聚簇索引 | 二级索引，叶子存主键值 | 索引文件 |

**追问：什么是回表？**

> 通过二级索引查到主键值，再回主键索引查数据
> 
> 避免：使用覆盖索引（查询字段都在索引中）

---

### 2. 事务的隔离级别？

**答案：**

| 隔离级别 | 脏读 | 不可重复读 | 幻读 |
|----------|:----:|:----------:|:----:|
| 读未提交（READ UNCOMMITTED） | ✓ | ✓ | ✓ |
| 读已提交（READ COMMITTED） | ✗ | ✓ | ✓ |
| 可重复读（REPEATABLE READ） | ✗ | ✗ | ✓ |
| 串行化（SERIALIZABLE） | ✗ | ✗ | ✗ |

MySQL 默认：可重复读

**追问：MySQL 如何解决幻读？**

> 1. **MVCC**：快照读，看到事务开始时的数据版本
> 2. **Next-Key Lock**：当前读，锁定记录和间隙，防止插入

---

### 3. SQL 优化有哪些方法？

**答案：**

1. **索引优化**
   - 避免 `SELECT *`，只查需要的字段
   - WHERE、ORDER BY、GROUP BY 字段建索引
   - 避免索引失效（函数、类型转换、!=、OR、%开头）

2. **查询优化**
   - 避免 `SELECT *`
   - 小表驱动大表
   - 用 EXISTS 替代 IN

3. **表结构优化**
   - 选择合适的字段类型
   - 适当的反范式（减少 JOIN）

**追问：索引什么情况下会失效？**

> 1. 索引列参与计算或函数
> 2. 类型隐式转换
> 3. LIKE 以 % 开头
> 4. 使用 != 或 <>
> 5. 使用 OR（除非所有字段都有索引）
> 6. 索引列允许 NULL 且查询 IS NULL

---

## 七、Redis

### 1. Redis 为什么快？

**答案：**

1. **基于内存**：读写都在内存，无磁盘 IO
2. **单线程**：无上下文切换、无锁竞争
3. **IO 多路复用**：epoll 模型，高并发
4. **高效数据结构**：SDS、哈希表、跳表等

**追问：为什么 Redis 6.0 引入多线程？**

> 单线程处理网络 IO 成了瓶颈
> 多线程处理网络 IO（读写数据），命令执行仍是单线程

---

### 2. Redis 缓存穿透、击穿、雪崩？

**答案：**

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 缓存穿透 | 查询不存在的数据，绕过缓存查 DB | 布隆过滤器、缓存空值 |
| 缓存击穿 | 热点 key 过期，大量请求打 DB | 热点数据永不过期、互斥锁 |
| 缓存雪崩 | 大量 key 同时过期 | 随机过期时间、多级缓存 |

**追问：布隆过滤器原理？**

> 位图 + 多个哈希函数
> - 判断存在：可能存在（有误判）
> - 判断不存在：一定不存在
> 
> 适用：缓存穿透防护、垃圾邮件过滤

---

### 3. Redis 持久化方式？

**答案：**

| 方式 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| RDB | 快照，定时保存数据 | 文件小、恢复快 | 可能丢数据 |
| AOF | 日志，记录写命令 | 数据更完整 | 文件大、恢复慢 |

**追问：生产环境如何选择？**

> 混合使用：RDB + AOF
> - RDB 用于快速恢复
> - AOF 保证数据完整性

---

## 八、场景题

### 1. 如何实现分布式锁？

**答案：**

**Redis 实现：**
```java
// 加锁
SET key value NX PX 30000  // NX 不存在才设置，PX 过期时间

// 释放锁（Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

**追问：Redis 分布式锁有什么问题？**

> 1. **锁过期**：业务执行时间 > 锁过期时间
>    - 解决：看门狗自动续期
> 2. **主从切换**：主节点宕机，锁丢失
>    - 解决：Redlock 算法

---

### 2. 如何设计一个秒杀系统？

**答案：**

```
秒杀架构：
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  用户    │ → │ CDN/静态 │ → │ 网关限流 │ → │  消息队列 │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
                                               ↓
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  数据库  │ ← │ 订单服务 │ ← │ 库存扣减 │ ← │ 消费处理  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

**核心设计：**
1. **前端**：静态化、按钮防抖、验证码
2. **网关**：限流、黑名单
3. **服务端**：预热缓存、库存预热到 Redis
4. **扣减库存**：Redis 原子操作 + 消息队列异步
5. **数据库**：乐观锁、分库分表

**追问：如何防止超卖？**

> ```java
> // Redis 原子扣减
> if redis.call("decr", "stock") >= 0 then
>     return 1  // 成功
> else
>     return 0  // 失败
> end
> ```

---

## 小结

本文档涵盖了 Java 面试的核心知识点，包括：
- Java 基础：数据类型、面向对象、异常处理
- 集合框架：List、Set、Map、线程安全
- 多线程并发：线程状态、锁机制、线程池
- JVM：内存模型、垃圾回收、类加载
- Spring 框架：IOC、AOP、事务
- 数据库：索引、事务、SQL 优化
- Redis：缓存问题、持久化
- 场景题：分布式锁、秒杀系统

**面试建议：**
1. 先理解原理，再背诵答案
2. 结合项目经验回答
3. 追问时展示深度思考
4. 不知道的诚实回答，不要硬编
