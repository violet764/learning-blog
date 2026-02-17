# 实战项目 + 性能优化

> 将理论知识应用于实践，学习性能优化技巧。

## 实战项目

### 项目一：命令行工具（CLI）

#### 文件搜索工具

```rust
// Cargo.toml
// [dependencies]
// clap = { version = "4.0", features = ["derive"] }
// walkdir = "2.4"

use clap::Parser;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// 文件搜索工具
#[derive(Parser, Debug)]
#[command(name = "fsearch")]
#[command(about = "在目录中搜索文件", long_about = None)]
struct Args {
    /// 搜索目录
    #[arg(short, long, default_value = ".")]
    path: String,

    /// 文件名模式
    #[arg(short, long, default_value = "")]
    name: String,

    /// 包含文本
    #[arg(short, long, default_value = "")]
    content: String,

    /// 显示详细信息
    #[arg(short, long, default_value = false)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    for entry in WalkDir::new(&args.path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // 按名称过滤
        if !args.name.is_empty() {
            if let Some(file_name) = path.file_name() {
                if !file_name.to_string_lossy().contains(&args.name) {
                    continue;
                }
            }
        }

        // 按内容过滤
        if !args.content.is_empty() {
            if let Ok(contents) = fs::read_to_string(path) {
                if !contents.contains(&args.content) {
                    continue;
                }
            } else {
                continue;
            }
        }

        if args.verbose {
            println!("{} - {}", path.display(), path.metadata().map(|m| m.len()).unwrap_or(0));
        } else {
            println!("{}", path.display());
        }
    }
}
```

#### 参数解析示例

```rust
use clap::{Parser, ArgEnum};

#[derive(Parser, Debug)]
#[command(name = "myapp")]
struct Cli {
    /// 输出格式
    #[arg(short, long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// 输入文件
    #[arg(short, long)]
    input: Option<String>,

    /// 详细输出
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// 数字列表
    #[arg(short, long, value_delimiter = ',')]
    numbers: Vec<i32>,
}

#[derive(ArgEnum, Debug, Clone)]
enum Format {
    Text,
    Json,
    Csv,
}

fn main() {
    let cli = Cli::parse();
    println!("{:?}", cli);
}
```

---

### 项目二：简单 Web 服务

#### 使用 Axum

```rust
// Cargo.toml
// [dependencies]
// axum = "0.6"
// tokio = { version = "1", features = ["full"] }
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"

use axum::{
    routing::get,
    Router,
};
use std::net::SocketAddr;
use serde::{Deserialize, Serialize};

// 数据模型
#[derive(Serialize, Deserialize)]
struct User {
    id: u32,
    name: String,
    email: String,
}

// 处理器函数
async fn hello() -> &'static str {
    "Hello, World!"
}

async fn get_user(u32: axum::extract::Path<u32>) -> String {
    format!("User ID: {}", u32)
}

async fn create_user(axum::extract::Json(payload): axum::extract::Json<CreateUserRequest>) -> String {
    format!("Created user: {}", payload.name)
}

#[derive(Deserialize)]
struct CreateUserRequest {
    name: String,
    email: String,
}

#[tokio::main]
async fn main() {
    // 构建路由
    let app = Router::new()
        .route("/", get(hello))
        .route("/users/:id", get(get_user))
        .route("/users", axum::routing::post(create_user));

    // 绑定地址
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("服务器运行在 http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

#### 使用 Actix-web

```rust
// Cargo.toml
// [dependencies]
// actix-web = "4"
// actix-rt = "2"

use actix_web::{web, App, HttpResponse, HttpServer, Responder};

async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello!")
}

async fn greet(name: web::Path<String>) -> impl Responder {
    HttpResponse::Ok().body(format!("Hello, {}!", name))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(hello))
            .route("/{name}", web::get().to(greet))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
```

---

### 项目三：系统编程工具

#### 文件监控工具

```rust
// Cargo.toml
// [dependencies]
// notify = "5"

use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(".");

    // 创建 watcher
    let (tx, rx) = std::sync::mpsc::channel();

    let mut watcher = RecommendedWatcher::new(
        move |res| {
            tx.send(res).unwrap();
        },
        Config::default().with_poll_interval(Duration::from_secs(2)),
    )?;

    // 监听目录
    watcher.watch(path, RecursiveMode::Recursive)?;

    println!("监听目录: {:?}", path);

    // 处理事件
    for res in rx {
        match res {
            Ok(event) => {
                println!("事件: {:?}", event.kind);
                for path in event.paths {
                    println!("  路径: {}", path.display());
                }
            }
            Err(e) => {
                eprintln!("监控错误: {:?}", e);
            }
        }
    }

    Ok(())
}
```

---

## 性能优化

### 性能分析工具

#### cargo bench 基准测试

```rust
// src/lib.rs
use std::hint::black_box;

pub fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(fibonacci(10), 55);
    }
}

#[cfg(bench)]
mod benches {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_fibonacci(b: &mut Bencher) {
        b.iter(|| fibonacci(black_box(20)));
    }
}
```

```bash
# 运行基准测试
cargo bench

# 查看生成的报告
# target/release/deploy/
```

#### 使用 Criterion

```rust
// Cargo.toml
// [dev-dependencies]
// criterion = "0.5"

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn bench_fibonacci(c: &mut Criterion) {
    c.bench_function("fibonacci 20", |b| {
        b.iter(|| fibonacci(black_box(20)));
    });
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

```bash
# 运行
cargo bench --bench fibonacci

# 查看 HTML 报告
# target/criterion/
```

### 内存管理

#### Box<T> - 堆分配

```rust
// 栈上分配
let x = 5;

// 堆上分配
let x = Box::new(5);

// 递归类型需要 Box
enum List {
    Cons(i32, Box<List>),
    Nil,
}

let list = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Nil))));
```

#### Rc<T> - 引用计数

```rust
use std::rc::Rc;

let data = Rc::new(vec![1, 2, 3]);

let clone1 = Rc::clone(&data);
let clone2 = Rc::clone(&data);

println!("引用计数: {}", Rc::strong_count(&data));  // 3
```

#### Arc<T> - 原子引用计数（多线程）

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);

let handles: Vec<_> = (0..3).map(|_| {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        println!("{:?}", data);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

#### RefCell<T> - 内部可变性

```rust
use std::cell::RefCell;

let x = RefCell::new(vec![1, 2, 3]);

// 不可变借用
let borrowed = x.borrow();
println!("{:?}", borrowed);

// 可变借用
let mut borrowed = x.borrow_mut();
borrowed.push(4);
```

#### 组合：Rc<RefCell<T>>

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
}

let leaf = Rc::new(RefCell::new(Node {
    value: 3,
    children: vec![],
}));

let branch = Rc::new(RefCell::new(Node {
    value: 1,
    children: vec![Rc::clone(&leaf)],
}));

leaf.borrow_mut().value = 5;
```

---

### 减少拷贝与堆分配

#### 避免不必要的克隆

```rust
// 低效：每次迭代都克隆
let result: Vec<String> = items.iter()
    .map(|s| s.clone())
    .collect();

// 高效：借用
let result: Vec<&str> = items.iter()
    .map(|s| s.as_str())
    .collect();

// 使用引用
fn process(items: &[String]) { ... }
```

#### 预分配容量

```rust
// 预先知道大小
let mut vec = Vec::with_capacity(1000);

for i in 0..1000 {
    vec.push(i);
}

// 避免频繁重新分配
```

#### 使用栈代替堆

```rust
// Vec 栈分配数组
let arr = [1, 2, 3, 4, 5];

// smallvec 库：小规模栈数组
// use smallvec::SmallVec;
// let arr: SmallVec<[i32; 16]> = SmallVec::new();
```

#### 字符串优化

```rust
// String vs &str
fn print_str(s: &str) { ... }  // 接受任意字符串引用

// 使用 Cow 避免不必要的分配
use std::borrow::Cow;

fn process(input: &str) -> Cow<str> {
    if input.contains('$') {
        Cow::Owned(input.replace("$", "\\$"))
    } else {
        Cow::Borrowed(input)
    }
}
```

---

### 编译优化

```toml
# Cargo.toml
[profile.release]
opt-level = 3        # 优化级别 (0-3)
lto = true          # 链接时优化
codegen-units = 1   # 减少并行单元以优化
panic = 'abort'     # 减少 panic 代码
strip = true        # 剥离符号信息
```

```bash
# 优化构建
cargo build --release

# 查看优化效果
# - 使用 perf / Linux
# - 使用 cargo-flamegraph
```

---

## 小结

### 实战项目
1. **CLI 工具**：使用 clap 解析参数，walkdir 遍历目录
2. **Web 服务**：使用 axum 或 actix-web 快速构建 API
3. **系统工具**：notify 实现文件监控

### 性能优化
1. **性能分析**：cargo bench、Criterion
2. **内存管理**：Box、Rc、Arc、RefCell
3. **减少拷贝**：预分配、栈分配、借用
4. **编译优化**：release profile 配置

---

> 💡 **实践建议**：学习 Rust 最好的方式是动手实践。建议从简单的 CLI 工具开始，逐步挑战更复杂的项目。性能优化应该在有明确需求后再进行，过度优化是万恶之源。

下一章节我们将学习不安全 Rust 以及 Rust 生态与进阶方向。
