# 模块与包管理 + 并发编程

> 掌握 Rust 的代码组织方式（模块系统）和并发编程基础。

## 模块系统

### 模块基础

Rust 使用模块系统组织代码：

```rust
// src/lib.rs 或 src/main.rs

mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {
            println!("添加到候补名单");
        }

        fn seat_at_table() {
            println!("安排座位");
        }
    }

    mod serving {
        fn take_order() {}

        fn serve_order() {}

        fn take_payment() {}
    }
}

// 调用模块中的函数
fn main() {
    front_of_house::hosting::add_to_waitlist();
}
```

### 模块规则

1. **文件即模块**：模块可以放在单独文件中
2. **pub 关键字**：控制可见性
3. **super 关键字**：访问父模块

```rust
mod front_of_house {
    pub mod hosting;

    pub fn eat_at_restaurant() {
        // 绝对路径
        crate::front_of_house::hosting::add_to_waitlist();

        // 相对路径
        hosting::add_to_waitlist();

        // super：父模块
        super::help_function();
    }
}
```

### pub 可见性

```rust
mod my_module {
    // 公有结构体
    pub struct Config {
        pub url: String,      // 公有字段
        timeout: u64,          // 私有字段
    }

    // 可以在外部创建
    impl Config {
        pub fn new(url: String) -> Config {
            Config { url, timeout: 30 }
        }
    }

    // 公有枚举（所有变体默认公有）
    pub enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
    }
}
```

### use 关键字

```rust
mod front_of_house {
    pub mod hosting;
}

// 导入模块
use crate::front_of_house::hosting;

// 重命名
use std::collections::HashMap as HashMapCustom;

// 导出父级
use front_of_house::hosting::{self, add_to_waitlist};

fn main() {
    hosting::add_to_waitlist();
}
```

---

## Cargo 包管理

### 项目结构

```
my_project/
├── src/
│   ├── main.rs          // 二进制入口
│   └── lib.rs           // 库入口
├── tests/               // 集成测试
│   └── integration_test.rs
├── benches/             // 性能基准测试
├── examples/           // 示例代码
├── Cargo.toml
└── Cargo.lock          // 依赖锁定文件
```

### Cargo.toml 配置

```toml
[package]
name = "my_crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A short description"
license = "MIT"

[dependencies]
# 精确版本
foo = "1.2.3"

# 范围版本
bar = "1.0"      # >= 1.0.0, < 2.0.0
baz = "=1.2.3"   # 精确匹配

# 从 git 仓库
rand = { git = "https://github.com/rust-lang/rand" }

# 本地包
mylib = { path = "../mylib" }

# 特性
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
# 开发依赖

[features]
default = ["default-feature"]
extra-feature = []
```

### 工作空间（Workspace）

```toml
# Cargo.toml (根目录)
[workspace]
members = [
    "crate_a",
    "crate_b",
]
resolver = "2"

[workspace.dependencies]
serde = "1.0"
```

```toml
# crate_a/Cargo.toml
[package]
name = "crate_a"

[dependencies]
serde = { workspace = true }
```

---

## 并发编程

### 线程创建

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // 创建线程
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("子线程: {}", i);
            thread::sleep(Duration::from_millis(100));
        }
    });

    // 主线程
    for i in 1..=3 {
        println!("主线程: {}", i);
        thread::sleep(Duration::from_millis(100));
    }

    // 等待子线程结束
    handle.join().unwrap();
}
```

### 线程闭包捕获变量

```rust
fn main() {
    let v = vec![1, 2, 3];

    // 移动语义：转移所有权到线程
    let handle = thread::spawn(move || {
        println!("闭包中的 vec: {:?}", v);
    });

    // v 在这里已不可用
    // println!("{:?}", v);  // 错误

    handle.join().unwrap();
}
```

### 消息传递（Channel）

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    // 创建通道
    let (tx, rx) = mpsc::channel();

    // 发送端
    thread::spawn(move || {
        let msg = "Hello from thread";
        tx.send(msg).unwrap();
    });

    // 接收端
    let received = rx.recv().unwrap();
    println!("收到: {}", received);
}
```

#### 多发送者

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    let tx1 = tx.clone();
    thread::spawn(move || {
        tx1.send("From 1").unwrap();
    });

    let tx2 = tx.clone();
    thread::spawn(move || {
        tx2.send("From 2").unwrap();
    });

    // 接收多次
    for received in rx {
        println!("收到: {}", received);
    }
}
```

### 共享状态（Mutex）

```rust
use std::sync::Mutex;
use std::thread;

fn main() {
    let counter = Mutex::new(0);

    let mut handles = vec![];

    for _ in 0..10 {
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("结果: {}", *counter.lock().unwrap());
}
```

### RwLock（读写锁）

```rust
use std::sync::RwLock;

fn main() {
    let lock = RwLock::new(5);

    // 读：多个并发读
    {
        let r1 = lock.read().unwrap();
        let r2 = lock.read().unwrap();
        println!("r1 = {}, r2 = {}", *r1, *r2);
    }

    // 写：独占写入
    {
        let mut w = lock.write().unwrap();
        *w += 1;
    }
}
```

### Send 与 Sync Trait

```rust
// 自动实现
// - Send：可以在线程间安全转移所有权
// - Sync：可以在线程间安全共享引用

// 手动实现（通常不需要）
use std::marker::PhantomData;

// 只有 Send 类型可以跨线程传递
// 只有 Sync 类型可以跨线程共享
```

---

## 常用并发模式

### 生产者-消费者

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    // 生产者线程
    thread::spawn(move || {
        for i in 0..5 {
            tx.send(i).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });

    // 消费者线程
    let handle = thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            println!("收到: {}", msg);
        }
    });

    handle.join().unwrap();
}
```

### 线程池

```rust
// 使用 crossbeam 或 rayon 等库
// 简单示例使用 threadpool crate
use std::thread;
use std::sync::mpsc::channel;

fn parallel_process(items: Vec<i32>) -> Vec<i32> {
    let (tx, rx) = channel();
    let n_threads = 4;
    let chunk_size = items.len() / n_threads;

    let mut handles = vec![];

    for i in 0..n_threads {
        let tx = tx.clone();
        let chunk: Vec<_> = items
            .iter()
            .skip(i * chunk_size)
            .take(chunk_size)
            .cloned()
            .collect();

        let handle = thread::spawn(move || {
            let result: Vec<_> = chunk.iter().map(|x| x * 2).collect();
            tx.send(result).unwrap();
        });
        handles.push(handle);
    }

    drop(tx); // 关闭发送端

    let mut results = vec![];
    while let Ok(chunk) = rx.recv() {
        results.extend(chunk);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    results
}
```

### 并行迭代（rayon）

```rust
// 在 Cargo.toml 添加 rayon = "1.7"

use rayon::prelude::*;

fn main() {
    let v: Vec<i32> = (0..1000).collect();

    // 并行 map
    let result: Vec<i32> = v.par_iter()
        .map(|x| x * 2)
        .collect();

    // 并行 filter
    let result: Vec<_> = v.par_iter()
        .filter(|x| *x % 2 == 0)
        .collect();

    // 并行归约
    let sum: i32 = v.par_iter()
        .sum();

    println!("Sum: {}", sum);
}
```

---

## 小结

### 模块系统
1. **模块定义**：`mod` 关键字
2. **文件组织**：模块可放在单独文件
3. **可见性**：`pub` 控制导出
4. **路径**：`use` 导入，`super` 访问父模块

### 包管理
1. **Cargo**：包管理器和构建工具
2. **Cargo.toml**：项目配置
3. **Workspace**：多 crate 管理
4. **依赖管理**：版本规范、特性、git

### 并发编程
1. **线程**：`thread::spawn`
2. **通道**：`mpsc::channel` 消息传递
3. **Mutex**：互斥锁保护共享数据
4. **RwLock**：读写锁
5. **Send/Sync**：线程安全 trait

下一章节我们将学习实战项目、性能优化以及不安全 Rust。
