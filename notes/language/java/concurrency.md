# Java 多线程与并发编程

> 多线程是 Java 的核心特性之一，允许程序同时执行多个任务。本章节将系统讲解多线程的概念、API 使用和并发编程技巧。

---

## 写给初学者：什么是多线程？

### 单线程 vs 多线程

```
单线程（传统程序）：
┌─────────────────────────────────────────────┐
│ 时间线                                       │
│ ├── 任务A（下载文件）── 需要等待 10 秒        │
│ ├── 任务B（处理数据）── 需要等待 5 秒         │
│ └── 任务C（显示结果）── 需要等待 2 秒         │
│                                             │
│ 总耗时：10 + 5 + 2 = 17 秒                   │
│ 问题：任务A 等待时，CPU 闲置，浪费资源        │
└─────────────────────────────────────────────┘

多线程（并行执行）：
┌─────────────────────────────────────────────┐
│ 线程1：任务A（下载文件）── 10 秒              │
│ 线程2：任务B（处理数据）── 5 秒               │
│ 线程3：任务C（显示结果）── 2 秒               │
│                                             │
│ 总耗时：max(10, 5, 2) = 10 秒                │
│ 优势：多个任务同时执行，充分利用 CPU          │
└─────────────────────────────────────────────┘

生活中的例子：
├── 厨房做饭：一个厨师（单线程）vs 多个厨师（多线程）
├── 银行柜台：一个窗口（单线程）vs 多个窗口（多线程）
└── 高速公路：单车道（单线程）vs 多车道（多线程）
```

### 进程与线程

```
进程（Process）：
├── 定义：正在运行的程序实例
├── 特点：独立内存空间、独立资源
├── 示例：打开的浏览器、IDE、音乐播放器
└── 类比：一个工厂

线程（Thread）：
├── 定义：进程内的执行单元
├── 特点：共享进程内存、轻量级
├── 示例：浏览器中的多个标签页
└── 类比：工厂里的工人

关系：
┌─────────────────────────────────────┐
│           进程（浏览器）              │
│  ┌─────────┬─────────┬─────────┐   │
│  │ 线程1   │ 线程2   │ 线程3   │   │
│  │ 渲染页面 │ 下载文件 │ 播放音频 │   │
│  └─────────┴─────────┴─────────┘   │
│         共享内存空间                 │
└─────────────────────────────────────┘
```

### 并发 vs 并行

```
并发（Concurrency）：
├── 定义：多个任务交替执行（单核 CPU）
├── 原理：时间片轮转，快速切换
├── 示例：一个人同时看书和吃饭（交替进行）
└── 目的：提高响应性，避免阻塞

并行（Parallelism）：
├── 定义：多个任务同时执行（多核 CPU）
├── 原理：每个核运行一个线程
├── 示例：多个人同时看书和吃饭
└── 目的：提高吞吐量，加速计算

图解：
单核并发：
时间 ─────────────────────────────►
线程A ──┐   ┌──┐   ┌──┐   ┌──
        └───┘  └───┘  └───┘
线程B     ┌──┐   ┌──┐   ┌──┐
        ──┘  └───┘  └───┘  └──

多核并行：
核1：线程A ─────────────────────►
核2：线程B ─────────────────────►
```

---

## 线程基础

### 创建线程的三种方式

#### 方式一：继承 Thread 类

```java
/**
 * 方式一：继承 Thread 类
 * 
 * 优点：代码简单直接
 * 缺点：Java 单继承，不能再继承其他类
 */
public class MyThread extends Thread {
    
    @Override
    public void run() {
        // 线程执行的代码
        for (int i = 0; i < 5; i++) {
            // Thread.currentThread().getName() 获取当前线程名
            System.out.println(Thread.currentThread().getName() + ": " + i);
        }
    }
    
    public static void main(String[] args) {
        // 创建线程对象
        MyThread thread = new MyThread();
        
        // 启动线程（不是直接调用 run()！）
        thread.start();  // 启动新线程，JVM 会调用 run()
        
        // 主线程继续执行
        System.out.println("主线程执行中...");
    }
}
```

**初学者常见错误：**

```java
// ❌ 错误：直接调用 run()，不会启动新线程
thread.run();  // 相当于普通方法调用，在当前线程执行

// ✅ 正确：调用 start()，启动新线程
thread.start();  // 创建新线程，在新线程中执行 run()
```

#### 方式二：实现 Runnable 接口（推荐）

```java
/**
 * 方式二：实现 Runnable 接口
 * 
 * 优点：
 * 1. 避免 Java 单继承限制
 * 2. 任务与线程分离，更灵活
 * 3. 便于使用线程池
 * 
 * 推荐使用这种方式！
 */
public class MyRunnable implements Runnable {
    
    @Override
    public void run() {
        // 线程执行的代码
        for (int i = 0; i < 5; i++) {
            System.out.println(Thread.currentThread().getName() + ": " + i);
        }
    }
    
    public static void main(String[] args) {
        // 创建任务对象
        MyRunnable task = new MyRunnable();
        
        // 创建线程，传入任务
        Thread thread = new Thread(task, "我的线程");  // 第二个参数是线程名
        
        // 启动线程
        thread.start();
    }
}

// 使用 Lambda 表达式简化（Java 8+）
public class LambdaThread {
    public static void main(String[] args) {
        // Lambda 表达式创建 Runnable
        Thread thread = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Lambda线程: " + i);
            }
        });
        
        thread.start();
    }
}
```

#### 方式三：实现 Callable 接口（有返回值）

```java
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;
import java.util.concurrent.ExecutionException;

/**
 * 方式三：实现 Callable 接口
 * 
 * 优点：
 * 1. 可以有返回值
 * 2. 可以抛出异常
 * 
 * 适用场景：需要获取线程执行结果时
 */
public class MyCallable implements Callable<Integer> {
    
    @Override
    public Integer call() throws Exception {
        // 计算任务
        int sum = 0;
        for (int i = 1; i <= 100; i++) {
            sum += i;
        }
        return sum;  // 返回计算结果
    }
    
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        // 创建 Callable 任务
        MyCallable task = new MyCallable();
        
        // 包装成 FutureTask（FutureTask 实现了 Runnable 接口）
        FutureTask<Integer> futureTask = new FutureTask<>(task);
        
        // 创建线程并启动
        Thread thread = new Thread(futureTask);
        thread.start();
        
        // 主线程可以做其他事情
        System.out.println("主线程继续执行...");
        
        // 获取结果（会阻塞直到线程执行完成）
        Integer result = futureTask.get();
        System.out.println("计算结果: " + result);  // 输出: 5050
    }
}
```

### 三种方式对比

| 方式 | 返回值 | 抛异常 | 继承限制 | 推荐程度 |
|------|:------:|:------:|:--------:|:--------:|
| 继承 Thread | 无 | 不能 | 单继承 | ⭐⭐ |
| 实现 Runnable | 无 | 不能 | 无限制 | ⭐⭐⭐⭐⭐ |
| 实现 Callable | 有 | 能 | 无限制 | ⭐⭐⭐⭐ |

```java
// 实际开发中，推荐使用线程池 + Callable/Runnable
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ThreadPoolDemo {
    public static void main(String[] args) throws Exception {
        // 创建线程池
        ExecutorService executor = Executors.newFixedThreadPool(3);
        
        // 提交任务
        Future<Integer> future = executor.submit(() -> {
            int sum = 0;
            for (int i = 1; i <= 100; i++) {
                sum += i;
            }
            return sum;
        });
        
        // 获取结果
        System.out.println("结果: " + future.get());
        
        // 关闭线程池
        executor.shutdown();
    }
}
```

---

## 线程状态与生命周期

### 线程的六种状态

```java
// Thread.State 枚举定义了 6 种线程状态
public enum State {
    NEW,          // 新建
    RUNNABLE,     // 可运行（就绪 + 运行中）
    BLOCKED,      // 阻塞（等待锁）
    WAITING,      // 等待（无限等待）
    TIMED_WAITING,// 超时等待
    TERMINATED    // 终止
}
```

### 状态转换图

```
                    ┌────────────────────────────────────────┐
                    │                                        │
                    ▼                                        │
┌──────────┐   start()   ┌──────────────┐                   │
│   NEW    │ ──────────► │   RUNNABLE   │ ◄──────────────┐  │
│  新建    │             │  可运行       │                │  │
└──────────┘             └──────────────┘                │  │
                              │    ▲                     │  │
                    synchronized │    │ 获取锁             │  │
                    等待锁       │    │                   │  │
                              ▼    │                     │  │
                        ┌──────────────┐                 │  │
                        │   BLOCKED    │                 │  │
                        │  阻塞        │ ────────────────┘  │
                        └──────────────┘                    │
                              │                             │
        wait()/join()        │      notify()/notifyAll()   │
        ┌────────────────────┘      ┌──────────────────────┘
        ▼                           │
┌──────────────┐                    │
│   WAITING    │ ───────────────────┘
│  等待        │                    ▲
└──────────────┘                    │
        ▲                           │
        │ sleep(ms)/wait(ms)        │ 超时/notify
        │                           │
┌──────────────┐                    │
│TIMED_WAITING │ ───────────────────┘
│ 超时等待     │
└──────────────┘
        │
        │ run() 执行完毕
        ▼
┌──────────────┐
│ TERMINATED   │
│  终止        │
└──────────────┘
```

### 状态详解与代码示例

```java
public class ThreadStateDemo {
    
    public static void main(String[] args) throws InterruptedException {
        
        // ==================== NEW 状态 ====================
        Thread thread = new Thread(() -> {
            System.out.println("线程执行中...");
        });
        System.out.println("创建后状态: " + thread.getState());  // NEW
        
        // ==================== RUNNABLE 状态 ====================
        thread.start();
        System.out.println("启动后状态: " + thread.getState());  // RUNNABLE
        
        // ==================== BLOCKED 状态 ====================
        final Object lock = new Object();
        
        Thread t1 = new Thread(() -> {
            synchronized (lock) {
                try {
                    Thread.sleep(5000);  // 持有锁 5 秒
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        Thread t2 = new Thread(() -> {
            synchronized (lock) {  // 等待 t1 释放锁
                System.out.println("t2 获取到锁");
            }
        });
        
        t1.start();
        Thread.sleep(100);  // 确保 t1 先获取锁
        t2.start();
        Thread.sleep(100);  // 让 t2 尝试获取锁
        System.out.println("t2 等待锁状态: " + t2.getState());  // BLOCKED
        
        // ==================== WAITING 状态 ====================
        Thread t3 = new Thread(() -> {
            synchronized (lock) {
                try {
                    lock.wait();  // 无限等待
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        t3.start();
        Thread.sleep(100);
        System.out.println("t3 wait 后状态: " + t3.getState());  // WAITING
        
        // ==================== TIMED_WAITING 状态 ====================
        Thread t4 = new Thread(() -> {
            try {
                Thread.sleep(3000);  // 睡眠 3 秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        t4.start();
        Thread.sleep(100);
        System.out.println("t4 sleep 状态: " + t4.getState());  // TIMED_WAITING
        
        // ==================== TERMINATED 状态 ====================
        Thread.sleep(4000);  // 等待线程结束
        System.out.println("thread 结束状态: " + thread.getState());  // TERMINATED
    }
}
```

---

## 线程常用方法

### sleep() - 线程睡眠

```java
/**
 * sleep(ms)：让当前线程暂停执行指定毫秒
 * 
 * 特点：
 * 1. 不会释放锁
 * 2. 时间到后自动唤醒
 * 3. 可能被 interrupt() 打断
 */
public class SleepDemo {
    public static void main(String[] args) {
        Thread thread = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                System.out.println("倒计时: " + i);
                try {
                    Thread.sleep(1000);  // 暂停 1 秒
                } catch (InterruptedException e) {
                    System.out.println("线程被中断");
                    return;  // 提前结束
                }
            }
            System.out.println("倒计时结束！");
        });
        
        thread.start();
    }
}

// 模拟网络请求超时
public class NetworkTimeoutDemo {
    public static void main(String[] args) {
        Thread requestThread = new Thread(() -> {
            System.out.println("发起网络请求...");
            try {
                Thread.sleep(5000);  // 模拟请求耗时 5 秒
                System.out.println("请求成功");
            } catch (InterruptedException e) {
                System.out.println("请求超时，已取消");
            }
        });
        
        requestThread.start();
        
        // 主线程等待 2 秒后取消请求
        try {
            Thread.sleep(2000);
            requestThread.interrupt();  // 中断请求线程
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

### join() - 等待线程结束

```java
/**
 * join()：等待调用该方法的线程执行完毕
 * 
 * 使用场景：主线程需要等待子线程完成后再继续执行
 */
public class JoinDemo {
    public static void main(String[] args) throws InterruptedException {
        
        System.out.println("主线程开始");
        
        Thread t1 = new Thread(() -> {
            for (int i = 1; i <= 3; i++) {
                System.out.println("子线程执行: " + i);
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        t1.start();
        
        // 主线程等待 t1 执行完毕
        t1.join();  // 阻塞主线程，直到 t1 结束
        
        // 也可以设置超时时间
        // t1.join(2000);  // 最多等待 2 秒
        
        System.out.println("主线程结束");
        
        /* 输出顺序：
         * 主线程开始
         * 子线程执行: 1
         * 子线程执行: 2
         * 子线程执行: 3
         * 主线程结束    ← 确保在子线程之后执行
         */
    }
}
```

### interrupt() - 中断线程

```java
/**
 * interrupt()：设置线程的中断标志
 * 
 * 注意：只是设置标志，不会真正停止线程
 * 线程需要在代码中检查中断标志并做出响应
 */
public class InterruptDemo {
    
    public static void main(String[] args) throws InterruptedException {
        
        // ==================== 响应中断示例 ====================
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                // 检查是否被中断
                if (Thread.currentThread().isInterrupted()) {
                    System.out.println("线程被中断，准备退出");
                    return;  // 响应中断，退出线程
                }
                System.out.println("执行: " + i);
            }
        });
        
        thread1.start();
        Thread.sleep(10);  // 让线程执行一会儿
        thread1.interrupt();  // 中断线程
        
        
        // ==================== sleep 时被中断 ====================
        Thread thread2 = new Thread(() -> {
            try {
                System.out.println("开始睡眠...");
                Thread.sleep(10000);  // 睡眠 10 秒
                System.out.println("睡眠结束");
            } catch (InterruptedException e) {
                // sleep/wait/join 会检测中断并抛出异常
                // 同时会清除中断标志
                System.out.println("睡眠被中断！");
                
                // 可以选择重新设置中断标志
                Thread.currentThread().interrupt();
            }
        });
        
        thread2.start();
        Thread.sleep(1000);
        thread2.interrupt();  // 中断睡眠中的线程
    }
}
```

### yield() - 线程让步

```java
/**
 * yield()：让出 CPU 时间片，让其他线程执行
 * 
 * 特点：
 * 1. 只是建议，不保证一定让出
 * 2. 让出后可能立即再次获得 CPU
 * 3. 实际开发中很少使用
 */
public class YieldDemo {
    public static void main(String[] args) {
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("线程A: " + i);
                Thread.yield();  // 让出 CPU
            }
        });
        
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("线程B: " + i);
                Thread.yield();  // 让出 CPU
            }
        });
        
        t1.start();
        t2.start();
    }
}
```

### setPriority() - 设置优先级

```java
/**
 * 线程优先级：1-10，默认 5
 * 
 * 注意：优先级只是建议，不保证一定按优先级执行
 */
public class PriorityDemo {
    public static void main(String[] args) {
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("高优先级线程: " + i);
            }
        });
        
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("低优先级线程: " + i);
            }
        });
        
        t1.setPriority(Thread.MAX_PRIORITY);  // 10
        t2.setPriority(Thread.MIN_PRIORITY);  // 1
        
        t1.start();
        t2.start();
    }
}
```

### 守护线程

```java
/**
 * 守护线程（Daemon Thread）：
 * - 为其他线程服务的线程
 * - 当所有用户线程结束，守护线程自动结束
 * - 典型应用：GC 线程
 */
public class DaemonDemo {
    public static void main(String[] args) {
        Thread daemon = new Thread(() -> {
            while (true) {
                System.out.println("守护线程运行中...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        // 设置为守护线程（必须在 start() 前设置）
        daemon.setDaemon(true);
        daemon.start();
        
        // 主线程（用户线程）运行 2 秒后结束
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        System.out.println("主线程结束，守护线程也会随之结束");
        // 守护线程会在所有用户线程结束后自动终止
    }
}
```

---

## 线程同步

### 为什么需要同步？

```java
/**
 * 线程安全问题示例
 * 
 * 多个线程同时访问共享资源，可能导致数据不一致
 */
public class UnsafeDemo {
    
    // 共享变量：银行账户余额
    private static int balance = 1000;
    
    public static void main(String[] args) throws InterruptedException {
        
        // 创建两个线程同时取钱
        Thread t1 = new Thread(() -> {
            withdraw(800);  // 取 800
        });
        
        Thread t2 = new Thread(() -> {
            withdraw(800);  // 取 800
        });
        
        t1.start();
        t2.start();
        
        t1.join();
        t2.join();
        
        System.out.println("最终余额: " + balance);
        // 问题：余额可能变成负数！
    }
    
    public static void withdraw(int amount) {
        // 这里有线程安全问题！
        if (balance >= amount) {          // 步骤1：检查余额
            try {
                Thread.sleep(100);         // 模拟网络延迟
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            balance -= amount;             // 步骤2：扣除余额
            System.out.println("取款成功，余额: " + balance);
        } else {
            System.out.println("余额不足");
        }
    }
}

/*
问题分析：
时间线：
t1: 检查余额(1000>=800) ✓
t2: 检查余额(1000>=800) ✓    ← 此时 t1 还没扣款
t1: 扣款，余额=200
t2: 扣款，余额=-600   ← 超额取款！

根本原因：检查和扣款不是原子操作
解决方案：加锁，确保检查和扣款一起执行
*/
```

### synchronized 关键字

#### 同步方法

```java
/**
 * synchronized 方法：整个方法加锁
 * 
 * 锁对象：
 * - 实例方法：锁 this 对象
 * - 静态方法：锁 Class 对象
 */
public class SynchronizedMethodDemo {
    
    private static int balance = 1000;
    
    // 同步实例方法（锁 this）
    public synchronized void withdraw(int amount) {
        if (balance >= amount) {
            balance -= amount;
            System.out.println("取款成功，余额: " + balance);
        } else {
            System.out.println("余额不足");
        }
    }
    
    // 同步静态方法（锁 Class 对象）
    public static synchronized void deposit(int amount) {
        balance += amount;
        System.out.println("存款成功，余额: " + balance);
    }
}
```

#### 同步代码块

```java
/**
 * synchronized 代码块：只锁需要同步的代码
 * 
 * 优点：粒度更细，性能更好
 */
public class SynchronizedBlockDemo {
    
    private static int balance = 1000;
    private static final Object lock = new Object();  // 锁对象
    
    public static void withdraw(int amount) {
        // 使用 synchronized 代码块
        synchronized (lock) {  // 可以是任意对象
            if (balance >= amount) {
                balance -= amount;
                System.out.println("取款成功，余额: " + balance);
            } else {
                System.out.println("余额不足");
            }
        }
    }
    
    // 等价于同步实例方法
    public void withdraw2(int amount) {
        synchronized (this) {  // 锁当前对象
            // ...
        }
    }
    
    // 等价于同步静态方法
    public static void withdraw3(int amount) {
        synchronized (SynchronizedBlockDemo.class) {  // 锁 Class 对象
            // ...
        }
    }
}
```

#### synchronized 的特性

```java
/**
 * synchronized 的三大特性：
 * 1. 原子性：同步代码块要么全执行，要么全不执行
 * 2. 可见性：解锁前把共享变量刷新到主内存
 * 3. 可重入性：同一个线程可以多次获取同一把锁
 */
public class SynchronizedFeaturesDemo {
    
    public static void main(String[] args) {
        
        // ==================== 可重入性示例 ====================
        new Thread(() -> {
            methodA();
        }).start();
    }
    
    public synchronized void methodA() {
        System.out.println("methodA 执行");
        methodB();  // 可以再次获取锁（同一个线程）
    }
    
    public synchronized void methodB() {
        System.out.println("methodB 执行");
        // 如果不可重入，这里会死锁
    }
}
```

### Lock 接口

```java
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Lock 接口：比 synchronized 更灵活的锁
 * 
 * 优点：
 * 1. 可以手动获取/释放锁
 * 2. 可以尝试获取锁（tryLock）
 * 3. 可以响应中断
 * 4. 支持公平/非公平锁
 */
public class LockDemo {
    
    private static int balance = 1000;
    // 创建可重入锁
    private static final Lock lock = new ReentrantLock();
    
    public static void withdraw(int amount) {
        lock.lock();  // 获取锁
        try {
            // 临界区代码
            if (balance >= amount) {
                balance -= amount;
                System.out.println("取款成功，余额: " + balance);
            } else {
                System.out.println("余额不足");
            }
        } finally {
            lock.unlock();  // 必须在 finally 中释放锁
        }
    }
    
    // tryLock 示例
    public static void tryLockDemo() {
        if (lock.tryLock()) {  // 尝试获取锁，不阻塞
            try {
                // 获取锁成功
                System.out.println("获取锁成功");
            } finally {
                lock.unlock();
            }
        } else {
            // 获取锁失败
            System.out.println("获取锁失败，做其他事情");
        }
    }
    
    // 带超时的 tryLock
    public static void tryLockWithTimeout() throws InterruptedException {
        if (lock.tryLock(3, java.util.concurrent.TimeUnit.SECONDS)) {
            try {
                System.out.println("在 3 秒内获取到锁");
            } finally {
                lock.unlock();
            }
        } else {
            System.out.println("3 秒内未获取到锁");
        }
    }
}
```

### synchronized vs Lock

| 特性 | synchronized | Lock |
|------|--------------|------|
| 获取/释放 | 自动 | 手动 |
| 响应中断 | 不支持 | 支持 |
| 超时获取 | 不支持 | 支持 |
| 公平性 | 非公平 | 公平/非公平可选 |
| 条件变量 | 单一 | 多个 Condition |
| 性能 | JDK 6 后优化，差距不大 | 高并发下略优 |
| 使用场景 | 简单同步 | 复杂同步需求 |

```java
// 实际开发建议：
// 1. 简单同步：用 synchronized，代码简洁
// 2. 复杂需求：用 Lock，功能更强

// 简单场景推荐 synchronized
public synchronized void simpleMethod() {
    // 简单的同步逻辑
}

// 复杂场景推荐 Lock
public void complexMethod() throws InterruptedException {
    if (lock.tryLock(5, TimeUnit.SECONDS)) {
        try {
            // 复杂的同步逻辑
        } finally {
            lock.unlock();
        }
    }
}
```

---

## 线程通信

### wait() 和 notify()

```java
/**
 * wait() / notify() / notifyAll()
 * 
 * 必须在 synchronized 块中使用
 * 必须用锁对象调用
 */
public class ProducerConsumerDemo {
    
    public static void main(String[] args) {
        Buffer buffer = new Buffer(5);
        
        // 生产者线程
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 10; i++) {
                try {
                    buffer.put("商品" + i);
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        // 消费者线程
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 10; i++) {
                try {
                    buffer.take();
                    Thread.sleep(200);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        producer.start();
        consumer.start();
    }
}

/**
 * 缓冲区（生产者-消费者模式）
 */
class Buffer {
    private final String[] items;
    private int count = 0;
    private int putIndex = 0;
    private int takeIndex = 0;
    
    public Buffer(int capacity) {
        items = new String[capacity];
    }
    
    public synchronized void put(String item) throws InterruptedException {
        // 缓冲区满，等待消费者消费
        while (count == items.length) {
            System.out.println("缓冲区满，生产者等待...");
            this.wait();  // 释放锁并等待
        }
        
        items[putIndex] = item;
        putIndex = (putIndex + 1) % items.length;
        count++;
        System.out.println("生产: " + item + ", 缓冲区大小: " + count);
        
        // 唤醒等待的消费者
        this.notifyAll();  // 唤醒所有等待的线程
    }
    
    public synchronized String take() throws InterruptedException {
        // 缓冲区空，等待生产者生产
        while (count == 0) {
            System.out.println("缓冲区空，消费者等待...");
            this.wait();  // 释放锁并等待
        }
        
        String item = items[takeIndex];
        takeIndex = (takeIndex + 1) % items.length;
        count--;
        System.out.println("消费: " + item + ", 缓冲区大小: " + count);
        
        // 唤醒等待的生产者
        this.notifyAll();
        
        return item;
    }
}
```

### wait() vs sleep()

| 特性 | wait() | sleep() |
|------|--------|---------|
| 所属类 | Object | Thread |
| 释放锁 | 是 | 否 |
| 使用位置 | 同步块内 | 任意位置 |
| 唤醒方式 | notify() 或超时 | 超时或中断 |
| 用途 | 线程通信 | 暂停执行 |

---

## 线程池

### 为什么使用线程池？

```
不使用线程池的问题：
├── 每次任务都创建新线程，开销大
├── 线程数量不可控，可能耗尽系统资源
├── 线程无法复用，效率低
└── 无法统一管理和监控

线程池的优势：
├── 线程复用，减少创建/销毁开销
├── 控制并发数量，防止系统过载
├── 提供队列缓存任务
└── 统一管理和监控
```

### ThreadPoolExecutor 核心参数

```java
import java.util.concurrent.*;

/**
 * ThreadPoolExecutor 构造方法
 */
public ThreadPoolExecutor(
    int corePoolSize,      // 核心线程数（常驻线程）
    int maximumPoolSize,   // 最大线程数
    long keepAliveTime,    // 非核心线程空闲存活时间
    TimeUnit unit,         // 时间单位
    BlockingQueue<Runnable> workQueue,  // 工作队列
    ThreadFactory threadFactory,        // 线程工厂
    RejectedExecutionHandler handler    // 拒绝策略
) { }

/**
 * 参数详解示例
 */
public class ThreadPoolDemo {
    public static void main(String[] args) {
        
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2,                      // 核心线程数：始终存活
            4,                      // 最大线程数：高峰期最多 4 个
            60,                     // 空闲时间
            TimeUnit.SECONDS,       // 时间单位
            new ArrayBlockingQueue<>(10),  // 任务队列，容量 10
            Executors.defaultThreadFactory(),  // 默认线程工厂
            new ThreadPoolExecutor.AbortPolicy()  // 拒绝策略
        );
        
        // 执行任务
        for (int i = 0; i < 15; i++) {
            final int taskId = i;
            executor.execute(() -> {
                System.out.println("任务" + taskId + " 执行中，线程: " 
                    + Thread.currentThread().getName());
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        }
        
        executor.shutdown();  // 关闭线程池
    }
}
```

### 任务执行流程

```
                    提交任务
                       │
                       ▼
            ┌──────────────────┐
            │ 当前线程数 < 核心数？│
            └──────────────────┘
                  │       │
                 是       否
                  │       │
                  ▼       ▼
            创建核心线程   ┌──────────────────┐
                          │ 队列是否已满？    │
                          └──────────────────┘
                               │       │
                              否       是
                               │       │
                               ▼       ▼
                          加入队列  ┌──────────────────┐
                                   │ 当前线程数 < 最大数？│
                                   └──────────────────┘
                                        │       │
                                       是       否
                                        │       │
                                        ▼       ▼
                                  创建非核心线程  执行拒绝策略
```

### 四种拒绝策略

```java
/**
 * 拒绝策略：当线程池无法接受新任务时的处理方式
 */

// 1. AbortPolicy（默认）：抛出异常
new ThreadPoolExecutor.AbortPolicy()

// 2. CallerRunsPolicy：由调用线程执行任务
new ThreadPoolExecutor.CallerRunsPolicy()

// 3. DiscardPolicy：直接丢弃，不抛异常
new ThreadPoolExecutor.DiscardPolicy()

// 4. DiscardOldestPolicy：丢弃队列最老的任务，重新提交
new ThreadPoolExecutor.DiscardOldestPolicy()
```

### Executors 工具类

```java
import java.util.concurrent.*;

/**
 * Executors 提供的快捷创建方法
 * 
 * 注意：阿里规范不推荐直接使用 Executors，建议手动创建
 */
public class ExecutorsDemo {
    public static void main(String[] args) {
        
        // ==================== 固定大小线程池 ====================
        ExecutorService fixedPool = Executors.newFixedThreadPool(5);
        // 核心线程数 = 最大线程数 = 5
        // 适用：任务量固定的场景
        
        // ==================== 缓存线程池 ====================
        ExecutorService cachedPool = Executors.newCachedThreadPool();
        // 核心线程数 = 0，最大线程数 = Integer.MAX_VALUE
        // 适用：任务量波动大的场景
        // 警告：可能创建大量线程，导致 OOM
        
        // ==================== 单线程线程池 ====================
        ExecutorService singlePool = Executors.newSingleThreadExecutor();
        // 只有一个线程，保证任务顺序执行
        // 适用：需要顺序执行任务的场景
        
        // ==================== 定时任务线程池 ====================
        ScheduledExecutorService scheduledPool = Executors.newScheduledThreadPool(3);
        
        // 延迟执行
        scheduledPool.schedule(() -> {
            System.out.println("延迟 3 秒执行");
        }, 3, TimeUnit.SECONDS);
        
        // 周期执行
        scheduledPool.scheduleAtFixedRate(() -> {
            System.out.println("每 2 秒执行一次");
        }, 0, 2, TimeUnit.SECONDS);
        
        // ==================== 关闭线程池 ====================
        fixedPool.shutdown();  // 平滑关闭，等待任务完成
        // fixedPool.shutdownNow();  // 立即关闭，中断正在执行的任务
    }
}
```

### 推荐的线程池配置

```java
/**
 * 实际开发中推荐手动创建线程池
 */
public class RecommendedThreadPool {
    
    public static void main(String[] args) {
        
        // CPU 密集型任务：核心线程数 = CPU 核心数 + 1
        int cpuCount = Runtime.getRuntime().availableProcessors();
        
        ThreadPoolExecutor cpuIntensive = new ThreadPoolExecutor(
            cpuCount + 1,
            cpuCount + 1,
            0L,
            TimeUnit.MILLISECONDS,
            new LinkedBlockingQueue<>(100),
            new ThreadFactory() {
                private int count = 1;
                @Override
                public Thread newThread(Runnable r) {
                    return new Thread(r, "cpu-thread-" + count++);
                }
            },
            new ThreadPoolExecutor.CallerRunsPolicy()
        );
        
        // IO 密集型任务：核心线程数 = CPU 核心数 * 2
        ThreadPoolExecutor ioIntensive = new ThreadPoolExecutor(
            cpuCount * 2,
            cpuCount * 2,
            60L,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(200),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );
    }
}
```

---

## 并发工具类

### CountDownLatch - 倒计时器

```java
import java.util.concurrent.CountDownLatch;

/**
 * CountDownLatch：让一个或多个线程等待其他线程完成
 * 
 * 使用场景：等待多个任务全部完成后再继续
 */
public class CountDownLatchDemo {
    public static void main(String[] args) throws InterruptedException {
        
        // 创建倒计时器，计数为 3
        CountDownLatch latch = new CountDownLatch(3);
        
        // 启动 3 个工作线程
        for (int i = 1; i <= 3; i++) {
            final int workerId = i;
            new Thread(() -> {
                System.out.println("工作线程" + workerId + " 开始工作...");
                try {
                    Thread.sleep(1000 * workerId);  // 模拟工作耗时
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("工作线程" + workerId + " 完成工作");
                latch.countDown();  // 计数减 1
            }).start();
        }
        
        System.out.println("主线程等待所有工作线程完成...");
        latch.await();  // 阻塞，直到计数为 0
        
        System.out.println("所有工作线程已完成，主线程继续执行");
        
        /* 输出：
         * 主线程等待所有工作线程完成...
         * 工作线程1 开始工作...
         * 工作线程2 开始工作...
         * 工作线程3 开始工作...
         * 工作线程1 完成工作
         * 工作线程2 完成工作
         * 工作线程3 完成工作
         * 所有工作线程已完成，主线程继续执行
         */
    }
}
```

### CyclicBarrier - 循环栅栏

```java
import java.util.concurrent.CyclicBarrier;

/**
 * CyclicBarrier：让一组线程互相等待，都到达屏障点后再一起继续
 * 
 * 使用场景：多线程分阶段执行，每个阶段都要等待所有线程完成
 */
public class CyclicBarrierDemo {
    public static void main(String[] args) {
        
        int threadCount = 3;
        
        // 创建循环栅栏，3 个线程都到达后触发
        CyclicBarrier barrier = new CyclicBarrier(threadCount, () -> {
            System.out.println("=== 所有线程到达屏障点，开始下一阶段 ===");
        });
        
        for (int i = 1; i <= threadCount; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    // 第一阶段
                    System.out.println("线程" + threadId + " 第一阶段");
                    Thread.sleep(1000 * threadId);
                    barrier.await();  // 等待其他线程
                    
                    // 第二阶段
                    System.out.println("线程" + threadId + " 第二阶段");
                    Thread.sleep(500 * threadId);
                    barrier.await();  // 等待其他线程
                    
                    // 第三阶段
                    System.out.println("线程" + threadId + " 第三阶段");
                    
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        }
    }
}
```

### Semaphore - 信号量

```java
import java.util.concurrent.Semaphore;

/**
 * Semaphore：控制同时访问资源的线程数量
 * 
 * 使用场景：限流、数据库连接池
 */
public class SemaphoreDemo {
    public static void main(String[] args) {
        
        // 创建信号量，最多 3 个线程同时访问
        Semaphore semaphore = new Semaphore(3);
        
        // 10 个线程竞争 3 个资源
        for (int i = 1; i <= 10; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    // 获取许可（获取不到会阻塞）
                    semaphore.acquire();
                    
                    System.out.println("线程" + threadId + " 获取资源，开始工作");
                    Thread.sleep(2000);  // 模拟工作
                    System.out.println("线程" + threadId + " 释放资源");
                    
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    // 释放许可
                    semaphore.release();
                }
            }).start();
        }
    }
}
```

### 并发容器

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * 并发容器：线程安全的集合类
 */
public class ConcurrentCollections {
    
    public static void main(String[] args) {
        
        // ==================== ConcurrentHashMap ====================
        // 高并发 HashMap，性能优于 Hashtable 和 synchronizedMap
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("count", 1);
        map.computeIfAbsent("newKey", k -> 0);  // 原子操作
        
        // ==================== CopyOnWriteArrayList ====================
        // 写时复制，适合读多写少的场景
        CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
        list.add("item1");
        // 迭代时不会抛出 ConcurrentModificationException
        
        // ==================== BlockingQueue ====================
        // 阻塞队列，生产者-消费者模式
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);
        
        // 生产者
        new Thread(() -> {
            try {
                queue.put("data");  // 队列满时阻塞
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
        
        // 消费者
        new Thread(() -> {
            try {
                String data = queue.take();  // 队列空时阻塞
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
        
        // ==================== 原子类 ====================
        // AtomicInteger：线程安全的整数
        AtomicInteger atomicInt = new AtomicInteger(0);
        atomicInt.incrementAndGet();  // 原子自增
        atomicInt.compareAndSet(1, 10);  // CAS 操作
        
        // AtomicReference：原子引用
        AtomicReference<String> ref = new AtomicReference<>("initial");
        ref.compareAndSet("initial", "updated");
    }
}
```

---

## volatile 关键字

```java
/**
 * volatile：轻量级的同步机制
 * 
 * 作用：
 * 1. 保证可见性：一个线程修改后，其他线程立即看到
 * 2. 禁止指令重排序
 * 
 * 注意：不保证原子性！
 */
public class VolatileDemo {
    
    // ==================== 可见性问题 ====================
    // 不加 volatile，线程可能看不到其他线程的修改
    private static boolean running = true;
    // private static volatile boolean running = true;  // 正确做法
    
    public static void main(String[] args) throws InterruptedException {
        
        Thread worker = new Thread(() -> {
            System.out.println("工作线程开始...");
            while (running) {
                // 空循环，可能因为缓存看不到 running 的修改
            }
            System.out.println("工作线程结束");
        });
        
        worker.start();
        
        Thread.sleep(1000);
        running = false;  // 主线程修改标志
        System.out.println("主线程设置 running = false");
        
        // 如果 running 不加 volatile，工作线程可能不会停止
    }
}

/**
 * volatile 典型应用：双重检查锁定单例
 */
public class Singleton {
    
    // volatile 防止指令重排序
    private static volatile Singleton instance;
    
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null) {              // 第一次检查
            synchronized (Singleton.class) {
                if (instance == null) {      // 第二次检查
                    instance = new Singleton();
                    // 没有 volatile，可能发生：
                    // 1. 分配内存
                    // 2. 将引用指向内存  ← 可能先执行
                    // 3. 初始化对象      ← 可能后执行
                    // 导致其他线程拿到未初始化的对象
                }
            }
        }
        return instance;
    }
}

/**
 * volatile 不保证原子性示例
 */
class VolatileNotAtomic {
    private static volatile int count = 0;
    
    public static void main(String[] args) throws InterruptedException {
        Thread[] threads = new Thread[10];
        
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    count++;  // 不是原子操作！
                    // 实际是：读取 -> 加1 -> 写入
                }
            });
            threads[i].start();
        }
        
        for (Thread t : threads) {
            t.join();
        }
        
        System.out.println("count = " + count);
        // 期望 10000，实际可能小于 10000
        // 解决方案：使用 AtomicInteger 或 synchronized
    }
}
```

---

## 死锁

### 死锁示例

```java
/**
 * 死锁：两个或多个线程互相等待对方释放锁，导致永久阻塞
 */
public class DeadlockDemo {
    
    private static final Object lockA = new Object();
    private static final Object lockB = new Object();
    
    public static void main(String[] args) {
        
        Thread t1 = new Thread(() -> {
            synchronized (lockA) {
                System.out.println("线程1 获取 lockA");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("线程1 尝试获取 lockB");
                synchronized (lockB) {  // 等待 lockB
                    System.out.println("线程1 获取 lockB");
                }
            }
        });
        
        Thread t2 = new Thread(() -> {
            synchronized (lockB) {
                System.out.println("线程2 获取 lockB");
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("线程2 尝试获取 lockA");
                synchronized (lockA) {  // 等待 lockA
                    System.out.println("线程2 获取 lockA");
                }
            }
        });
        
        t1.start();
        t2.start();
        
        // 线程1 持有 lockA，等待 lockB
        // 线程2 持有 lockB，等待 lockA
        // 互相等待，死锁！
    }
}
```

### 死锁的四个必要条件

```
1. 互斥条件：资源只能被一个线程占用
2. 请求与保持条件：持有资源的同时请求其他资源
3. 不可剥夺条件：资源不能被强制抢占
4. 循环等待条件：存在循环等待资源的关系

只要破坏任意一个条件，就能避免死锁
```

### 避免死锁

```java
/**
 * 避免死锁的方法
 */
public class AvoidDeadlock {
    
    private static final Object lockA = new Object();
    private static final Object lockB = new Object();
    
    public static void main(String[] args) {
        
        // ==================== 方法1：固定加锁顺序 ====================
        Thread t1 = new Thread(() -> {
            synchronized (lockA) {  // 先获取 lockA
                synchronized (lockB) {  // 再获取 lockB
                    System.out.println("线程1 完成");
                }
            }
        });
        
        Thread t2 = new Thread(() -> {
            synchronized (lockA) {  // 先获取 lockA（与 t1 相同顺序）
                synchronized (lockB) {  // 再获取 lockB
                    System.out.println("线程2 完成");
                }
            }
        });
        
        // ==================== 方法2：使用 tryLock 超时 ====================
        Lock lock1 = new ReentrantLock();
        Lock lock2 = new ReentrantLock();
        
        Thread t3 = new Thread(() -> {
            try {
                if (lock1.tryLock(100, TimeUnit.MILLISECONDS)) {
                    try {
                        if (lock2.tryLock(100, TimeUnit.MILLISECONDS)) {
                            try {
                                System.out.println("获取两把锁成功");
                            } finally {
                                lock2.unlock();
                            }
                        }
                    } finally {
                        lock1.unlock();
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }
}
```

---

## 最佳实践

### 线程命名

```java
// 好的命名便于调试和监控
Thread thread = new Thread(() -> {}, "worker-thread-1");

// 使用 ThreadFactory
ThreadFactory factory = r -> new Thread(r, "pool-thread-" + counter++);
```

### 资源关闭

```java
// 正确关闭线程池
executor.shutdown();  // 平滑关闭
try {
    if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
        executor.shutdownNow();  // 强制关闭
    }
} catch (InterruptedException e) {
    executor.shutdownNow();
}
```

### 异常处理

```java
// 设置线程的异常处理器
Thread thread = new Thread(() -> {
    throw new RuntimeException("线程异常");
});
thread.setUncaughtExceptionHandler((t, e) -> {
    System.err.println("线程 " + t.getName() + " 发生异常: " + e.getMessage());
});
thread.start();
```

### 避免使用 ThreadLocal 不清理

```java
// ThreadLocal 必须清理，防止内存泄漏
ThreadLocal<User> userThreadLocal = new ThreadLocal<>();

try {
    userThreadLocal.set(currentUser);
    // 业务逻辑
} finally {
    userThreadLocal.remove();  // 必须清理！
}
```

---

## 小结

本文档涵盖了 Java 多线程与并发编程的核心内容：

| 主题 | 要点 |
|------|------|
| 线程基础 | 创建方式、生命周期、常用方法 |
| 线程同步 | synchronized、Lock、线程安全 |
| 线程通信 | wait/notify、生产者消费者模式 |
| 线程池 | ThreadPoolExecutor、拒绝策略、配置建议 |
| 并发工具 | CountDownLatch、CyclicBarrier、Semaphore |
| 并发容器 | ConcurrentHashMap、BlockingQueue、原子类 |
| volatile | 可见性、禁止重排、单例模式 |
| 死锁 | 四大条件、避免策略 |

**学习建议：**
1. 理解原理，不只是记住 API
2. 多写代码验证，观察线程行为
3. 学会使用调试工具（jconsole、jvisualvm）
4. 阅读并发经典书籍（《Java 并发编程实战》）
