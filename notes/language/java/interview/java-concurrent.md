# Java 多线程并发面试题

> 本文档整理 Java 多线程与并发编程相关的高频面试题。

---

## 一、线程基础

### 1. 线程和进程的区别？

**答案：**

| 特性 | 进程 | 线程 |
|------|------|------|
| 定义 | 程序执行的最小单位 | CPU 调度的最小单位 |
| 资源 | 独立内存空间 | 共享进程内存 |
| 通信 | IPC（管道、消息队列） | 直接读写共享变量 |
| 创建开销 | 大 | 小 |
| 切换开销 | 大 | 小 |
| 稳定性 | 一个崩溃不影响其他 | 一个崩溃可能影响整个进程 |

**追问：为什么线程切换比进程快？**

> 线程共享进程内存，切换时不需要切换内存空间，只需要保存/恢复寄存器和栈指针。进程切换需要切换页表、刷新 TLB 等，开销更大。

---

### 2. 创建线程有几种方式？

**答案：**

```java
// 方式一：继承 Thread 类
class MyThread extends Thread {
    public void run() { /* 任务代码 */ }
}
new MyThread().start();

// 方式二：实现 Runnable 接口（推荐）
class MyTask implements Runnable {
    public void run() { /* 任务代码 */ }
}
new Thread(new MyTask()).start();

// 方式三：实现 Callable 接口（有返回值）
class MyCallable implements Callable<String> {
    public String call() { return "result"; }
}
FutureTask<String> task = new FutureTask<>(new MyCallable());
new Thread(task).start();
String result = task.get();

// 方式四：线程池（实际开发推荐）
ExecutorService executor = Executors.newFixedThreadPool(3);
executor.submit(() -> { /* 任务代码 */ });
```

| 方式 | 返回值 | 抛异常 | 推荐程度 |
|------|:------:|:------:|:--------:|
| 继承 Thread | 无 | 不能 | ⭐⭐ |
| 实现 Runnable | 无 | 不能 | ⭐⭐⭐⭐⭐ |
| 实现 Callable | 有 | 能 | ⭐⭐⭐⭐ |
| 线程池 | 可选 | 能 | ⭐⭐⭐⭐⭐ |

**追问：start() 和 run() 的区别？**

> - **start()**：启动新线程，JVM 会调用 run() 方法
> - **run()**：普通方法调用，在当前线程执行
> 
> 直接调用 run() 不会创建新线程！

---

### 3. 线程有哪些状态？

**答案：**

```java
public enum State {
    NEW,          // 新建：创建但未启动
    RUNNABLE,     // 可运行：就绪 + 运行中
    BLOCKED,      // 阻塞：等待获取锁
    WAITING,      // 等待：无限等待
    TIMED_WAITING,// 超时等待：有时间的等待
    TERMINATED    // 终止：执行完毕
}
```

**状态转换：**

```
         start()
NEW ─────────────► RUNNABLE ◄─────────────────────┐
                        │                          │
            synchronized│获取锁                    │
             等待锁     │                          │
                        ▼                          │
                    BLOCKED ──────────────────────┤
                        │                          │
        wait()/join()   │      notify()/notifyAll() │
                        ▼                          │
                    WAITING ──────────────────────┤
                        │                          │
   sleep(ms)/wait(ms)  │      超时                 │
                        ▼                          │
                TIMED_WAITING ─────────────────────┤
                        │                          │
                  run() 结束                       │
                        ▼                          │
                   TERMINATED ◄────────────────────┘
```

---

### 4. sleep() 和 wait() 的区别？

**答案：**

| 特性 | sleep() | wait() |
|------|---------|--------|
| 所属类 | Thread | Object |
| 释放锁 | 不释放 | 释放 |
| 使用位置 | 任意位置 | 同步块内 |
| 唤醒方式 | 超时或中断 | notify() 或超时 |
| 用途 | 暂停执行 | 线程通信 |

```java
// sleep()：不释放锁
synchronized(lock) {
    Thread.sleep(1000);  // 睡眠期间仍持有锁
}

// wait()：释放锁
synchronized(lock) {
    lock.wait();  // 释放锁并等待
    // 被 notify 后重新获取锁
}
```

---

## 二、线程同步

### 5. synchronized 的实现原理？

**答案：**

synchronized 基于 Monitor 实现。

```
对象头结构：
┌─────────────────────────────────────────┐
│ Mark Word（存储锁状态、hash、GC年龄等）   │
├─────────────────────────────────────────┤
│ Class Metadata Address（类型指针）       │
├─────────────────────────────────────────┤
│ Array Length（数组长度，仅数组对象）      │
└─────────────────────────────────────────┘

Mark Word 锁状态：
├── 无锁：对象 hashcode、分代年龄
├── 偏向锁：线程 ID、Epoch、分代年龄
├── 轻量级锁：指向栈中 Lock Record 的指针
└── 重量级锁：指向 Monitor 的指针
```

**Monitor 工作原理：**

```java
// 同步代码块
synchronized(obj) {
    // 代码
}

// 字节码层面
monitorenter  // 获取 Monitor
// 代码
monitorexit   // 释放 Monitor
```

---

### 6. synchronized 锁升级过程？

**答案：**

```
无锁 → 偏向锁 → 轻量级锁 → 重量级锁

（锁升级不可逆）
```

**详细过程：**

```
1. 无锁状态
   对象刚创建，没有任何线程访问

2. 偏向锁（第一个线程访问）
   - 在对象头记录线程 ID
   - 同一线程后续访问无需加锁
   - 适用于只有一个线程访问的场景

3. 轻量级锁（有轻度竞争）
   - 撤销偏向锁
   - 使用 CAS 尝试获取锁
   - 在栈中创建 Lock Record
   - 适用于短时间的竞争

4. 重量级锁（竞争激烈）
   - CAS 失败多次后升级
   - 使用 Monitor 锁
   - 未获取锁的线程阻塞
   - 适用于长时间持有锁的场景
```

**追问：锁可以降级吗？**

> 不可以。锁只能升级，不能降级。这是为了避免频繁的锁状态切换。

---

### 7. synchronized 和 Lock 的区别？

**答案：**

| 特性 | synchronized | Lock |
|------|--------------|------|
| 层面 | JVM 关键字 | Java API |
| 获取/释放 | 自动 | 手动 lock/unlock |
| 响应中断 | 不支持 | 支持 lockInterruptibly() |
| 超时获取 | 不支持 | 支持 tryLock(timeout) |
| 公平性 | 非公平 | 公平/非公平可选 |
| 条件变量 | 单一 | 多个 Condition |
| 性能 | JDK 6 后优化，差距不大 | 高并发下略优 |

**追问：什么时候用 Lock？**

> 1. 需要尝试获取锁（tryLock）
> 2. 需要响应中断
> 3. 需要公平锁
> 4. 需要多个条件变量
> 5. 需要读写分离（ReentrantReadWriteLock）

---

### 8. ReentrantLock 的实现原理？

**答案：**

ReentrantLock 基于 AQS（AbstractQueuedSynchronizer）实现。

```
AQS 核心结构：
┌─────────────────────────────────────┐
│           state（同步状态）          │
│           0 = 未锁定                 │
│           1 = 已锁定                 │
│           >1 = 重入次数              │
├─────────────────────────────────────┤
│           CLH 队列                   │
│   ┌─────┐   ┌─────┐   ┌─────┐      │
│   │Node │ → │Node │ → │Node │      │
│   │线程1│   │线程2│   │线程3│      │
│   └─────┘   └─────┘   └─────┘      │
└─────────────────────────────────────┘
```

**加锁流程：**

```java
// 简化的加锁逻辑
final void lock() {
    // 1. CAS 尝试获取锁
    if (compareAndSetState(0, 1))
        setExclusiveOwnerThread(Thread.currentThread());
    else
        // 2. CAS 失败，加入队列
        acquire(1);
}

final boolean acquire(int arg) {
    // 3. 再次尝试获取锁
    if (!tryAcquire(arg) && 
        // 4. 加入队列并阻塞
        acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
        selfInterrupt();
}
```

---

## 三、volatile 关键字

### 9. volatile 的作用？

**答案：**

volatile 有两大作用：

**1. 保证可见性**
```java
// 没有 volatile，线程可能看不到其他线程的修改
private volatile boolean running = true;

void stop() {
    running = false;  // 所有线程立即看到
}
```

**2. 禁止指令重排序**
```java
// 双重检查锁定单例
private static volatile Singleton instance;

public static Singleton getInstance() {
    if (instance == null) {
        synchronized (Singleton.class) {
            if (instance == null) {
                instance = new Singleton();
                // 没有 volatile 可能发生：
                // 1. 分配内存
                // 2. 将引用指向内存 ← 可能先执行
                // 3. 初始化对象     ← 可能后执行
            }
        }
    }
    return instance;
}
```

**追问：volatile 能保证原子性吗？**

> 不能！volatile 只保证可见性和有序性。
> ```java
> volatile int count = 0;
> count++;  // 不是原子操作，线程不安全！
> // 需要用 AtomicInteger 或 synchronized
> ```

---

### 10. JMM（Java 内存模型）是什么？

**答案：**

JMM 定义了多线程之间共享变量的访问规则。

```
         工作内存          工作内存
        ┌────────┐        ┌────────┐
        │ 变量副本 │        │ 变量副本 │
        └────┬───┘        └────┬───┘
             │                 │
        read/write        read/write
             │                 │
        ┌────┴─────────────────┴────┐
        │         主内存             │
        │      （共享变量）           │
        └───────────────────────────┘

JMM 三大特性：
├── 原子性：操作不可分割
├── 可见性：一个线程修改，其他线程可见
└── 有序性：禁止指令重排序
```

**追问：happens-before 规则？**

> 1. 程序顺序规则：同一线程中，前面的操作 happens-before 后面的操作
> 2. 监视器锁规则：unlock happens-before 后续的 lock
> 3. volatile 规则：写 happens-before 后续的读
> 4. 线程启动规则：start() happens-before 该线程中的任何操作
> 5. 线程终止规则：线程中的操作 happens-before join() 返回

---

## 四、线程池

### 11. 线程池的核心参数？

**答案：**

```java
public ThreadPoolExecutor(
    int corePoolSize,      // 核心线程数
    int maximumPoolSize,   // 最大线程数
    long keepAliveTime,    // 非核心线程空闲存活时间
    TimeUnit unit,         // 时间单位
    BlockingQueue<Runnable> workQueue,  // 工作队列
    ThreadFactory threadFactory,        // 线程工厂
    RejectedExecutionHandler handler    // 拒绝策略
)
```

**任务执行流程：**

```
                    提交任务
                       │
                       ▼
            ┌─────────────────────┐
            │ 线程数 < 核心线程数？ │
            └─────────────────────┘
                  │           │
                 是           否
                  │           │
                  ▼           ▼
            创建核心线程  ┌─────────────────────┐
                          │   队列是否已满？     │
                          └─────────────────────┘
                                │           │
                               否           是
                                │           │
                                ▼           ▼
                            加入队列   ┌─────────────────────┐
                                       │ 线程数 < 最大线程数？ │
                                       └─────────────────────┘
                                             │           │
                                            是           否
                                             │           │
                                             ▼           ▼
                                       创建非核心线程   执行拒绝策略
```

---

### 12. 线程池的拒绝策略？

**答案：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| AbortPolicy | 抛出 RejectedExecutionException | 默认，要求严格 |
| CallerRunsPolicy | 由调用线程执行 | 不丢弃任务 |
| DiscardPolicy | 直接丢弃，不抛异常 | 允许丢失 |
| DiscardOldestPolicy | 丢弃队列最老任务，重新提交 | 允许丢失旧任务 |

```java
// 自定义拒绝策略
executor.setRejectedExecutionHandler((r, e) -> {
    // 记录日志、存储到数据库等
    log.warn("任务被拒绝: " + r.toString());
});
```

---

### 13. 如何设置线程池参数？

**答案：**

```java
// CPU 密集型任务
int cpuCount = Runtime.getRuntime().availableProcessors();
ThreadPoolExecutor cpuPool = new ThreadPoolExecutor(
    cpuCount + 1,          // 核心线程数 = CPU 核心数 + 1
    cpuCount + 1,          // 最大线程数 = CPU 核心数 + 1
    0L, TimeUnit.MILLISECONDS,
    new LinkedBlockingQueue<>(100)
);

// IO 密集型任务
ThreadPoolExecutor ioPool = new ThreadPoolExecutor(
    cpuCount * 2,          // 核心线程数 = CPU 核心数 * 2
    cpuCount * 2,
    60L, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(200)
);
```

**经验公式：**
- CPU 密集型：核心线程数 = CPU 核心数 + 1
- IO 密集型：核心线程数 = CPU 核心数 / (1 - 阻塞系数)
- 混合型：根据 IO 等待时间占比调整

---

### 14. 为什么不推荐使用 Executors？

**答案：**

```java
// 问题 1：FixedThreadPool 和 SingleThreadPool
// 队列是 LinkedBlockingQueue，容量是 Integer.MAX_VALUE
// 可能堆积大量任务，导致 OOM
Executors.newFixedThreadPool(10);
Executors.newSingleThreadExecutor();

// 问题 2：CachedThreadPool
// 最大线程数是 Integer.MAX_VALUE
// 可能创建大量线程，导致 OOM
Executors.newCachedThreadPool();

// 推荐：手动创建线程池
new ThreadPoolExecutor(
    corePoolSize,
    maximumPoolSize,
    keepAliveTime,
    TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(queueCapacity),  // 有界队列
    new ThreadPoolExecutor.CallerRunsPolicy()
);
```

---

## 五、并发工具类

### 15. ThreadLocal 的原理和问题？

**答案：**

**原理：**

```java
class Thread {
    ThreadLocal.ThreadLocalMap threadLocals;  // 每个线程有独立的 Map
}

class ThreadLocalMap {
    static class Entry extends WeakReference<ThreadLocal<?>> {
        Object value;  // 存储 ThreadLocal 对应的值
    }
    Entry[] table;
}
```

```
Thread 对象
    │
    └── ThreadLocalMap（线程私有）
            │
            ├── Entry(key=ThreadLocal1, value=值1)
            ├── Entry(key=ThreadLocal2, value=值2)
            └── ...
```

**内存泄漏问题：**

```
1. Entry 的 key（ThreadLocal）是弱引用，会被 GC 回收
2. key 被回收后，value 还在
3. 如果线程不结束（线程池），value 无法回收
4. 解决：使用完必须调用 remove()
```

**追问：ThreadLocal 的应用场景？**

> 1. 数据库连接管理（每个线程独立连接）
> 2. 用户上下文（每个请求独立）
> 3. SimpleDateFormat（线程不安全，用 ThreadLocal 包装）
> 4. 事务管理（每个线程独立事务）

---

### 16. CountDownLatch 和 CyclicBarrier 的区别？

**答案：**

| 特性 | CountDownLatch | CyclicBarrier |
|------|----------------|---------------|
| 作用 | 等待其他线程完成 | 线程互相等待 |
| 计数方向 | 递减到 0 | 递增到目标值 |
| 可重用 | 一次性 | 可循环使用 |
| 触发者 | 其他线程 countDown | 线程自身 await |

```java
// CountDownLatch：主线程等待子线程
CountDownLatch latch = new CountDownLatch(3);
for (int i = 0; i < 3; i++) {
    new Thread(() -> {
        // 执行任务
        latch.countDown();  // 计数减 1
    }).start();
}
latch.await();  // 等待计数归零

// CyclicBarrier：线程互相等待
CyclicBarrier barrier = new CyclicBarrier(3, () -> {
    System.out.println("所有线程到达屏障");
});
for (int i = 0; i < 3; i++) {
    new Thread(() -> {
        // 执行阶段1
        barrier.await();  // 等待其他线程
        // 执行阶段2
    }).start();
}
```

---

### 17. Semaphore 的作用？

**答案：**

Semaphore 用于控制同时访问资源的线程数量。

```java
// 限流：最多 3 个线程同时访问
Semaphore semaphore = new Semaphore(3);

for (int i = 0; i < 10; i++) {
    new Thread(() -> {
        try {
            semaphore.acquire();  // 获取许可
            // 访问资源
            Thread.sleep(1000);
        } finally {
            semaphore.release();  // 释放许可
        }
    }).start();
}
```

**应用场景：**
- 数据库连接池
- API 限流
- 停车场管理

---

## 六、死锁

### 18. 死锁的条件和避免？

**答案：**

**四个必要条件：**

```
1. 互斥条件：资源只能被一个线程占用
2. 请求与保持：持有资源同时请求其他资源
3. 不可剥夺：资源不能被强制抢占
4. 循环等待：存在循环等待资源的关系

只要破坏任意一个条件，就能避免死锁
```

**避免策略：**

```java
// 1. 固定加锁顺序
public void transfer(Account from, Account to, int amount) {
    // 按 hash 排序加锁，避免循环等待
    Account first = from.hashCode() < to.hashCode() ? from : to;
    Account second = from.hashCode() < to.hashCode() ? to : from;
    
    synchronized(first) {
        synchronized(second) {
            from.debit(amount);
            to.credit(amount);
        }
    }
}

// 2. 使用 tryLock 超时
if (lock1.tryLock(100, TimeUnit.MILLISECONDS)) {
    try {
        if (lock2.tryLock(100, TimeUnit.MILLISECONDS)) {
            try {
                // 执行操作
            } finally {
                lock2.unlock();
            }
        }
    } finally {
        lock1.unlock();
    }
}
```

---

## 小结

本文档涵盖了 Java 多线程并发面试的高频考点：

- 线程创建与状态
- synchronized 实现原理与锁升级
- volatile 与 JMM
- 线程池参数与配置
- ThreadLocal 原理与问题
- 并发工具类
- 死锁条件与避免
