# 不安全 Rust + 生态与进阶方向

> 探索 Rust 的底层能力，了解生态系统与未来学习方向。

## 不安全 Rust

Rust 的 unsafe 关键字允许绕过编译器的安全检查，用于：
- 调用 C 代码
- 底层系统编程
- 性能优化
- 实现其他语言特性

### 何时使用 Unsafe

```rust
// 1. 解引用裸指针
// 2. 调用 unsafe 函数
// 3. 访问或修改可变静态变量
// 4. 实现 unsafe trait
// 5. 标记函数为 unsafe
```

### 五大 Unsafe 能力

#### 1. 解引用裸指针

```rust
unsafe {
    let mut num = 5;

    // 创建裸指针
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;

    // 解引用
    println!("{}", *r1);
    *r2 = 10;
}

// 裸指针特点：
// - 可以有空指针
// - 可以悬挂
// - 不执行借用检查
// - 可变和不可变指针可以共存
```

#### 2. 调用 unsafe 函数

```rust
unsafe fn dangerous() {
    println!("这是一个 unsafe 函数");
}

// 调用时必须标记 unsafe
unsafe {
    dangerous();
}

// 可以创建 unsafe 函数包装
fn safe_wrapper() {
    unsafe {
        dangerous();
    }
}
```

#### 3. 访问或修改可变静态变量

```rust
static mut COUNTER: i32 = 0;

fn main() {
    unsafe {
        COUNTER += 1;
        println!("{}", COUNTER);
    }
}

// 推荐：使用线程安全的静态变量
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn main() {
    COUNTER.fetch_add(1, Ordering::SeqCst);
    println!("{}", COUNTER.load(Ordering::SeqCst));
}
```

#### 4. 实现 unsafe trait

```rust
unsafe trait MyTrait {
    fn method(&self);
}

unsafe impl MyTrait for i32 {
    fn method(&self) {
        println!("impl MyTrait for i32");
    }
}

// 某些 trait 需要 unsafe 实现
unsafe impl Send for MyStruct {}
unsafe impl Sync for MyStruct {}
```

#### 5. 标记函数为 unsafe

```rust
// unsafe 函数：调用者必须保证安全
unsafe fn slice_assume_init(slice: &[MaybeUninit<u8>]) -> &[u8] {
    // 安全使用需要外部保证
    unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) }
}

// 使用示例
use std::mem::MaybeUninit;

fn main() {
    let slice: &[MaybeUninit<u8>] = &[MaybeUninit::new(42)];
    let initialized = unsafe { slice_assume_init(slice) };
    println!("{:?}", initialized);
}
```

### 安全使用原则

```rust
// 1. 最小化 unsafe 代码块
// 2. 将 unsafe 封装在安全抽象中

// 不推荐
unsafe {
    let ptr = &mut 5 as *mut i32;
    *ptr = 10;
}

// 推荐：封装成安全函数
struct SafeWrapper {
    value: i32,
}

impl SafeWrapper {
    fn new(value: i32) -> Self {
        SafeWrapper { value }
    }

    // 内部使用 unsafe，但对外提供安全接口
    fn get(&self) -> i32 {
        self.value
    }
}
```

---

## 常见 Unsafe 使用场景

### 与 C 代码互操作

```rust
// 声明外部 C 函数
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("abs(-5) = {}", abs(-5));
    }
}
```

### 自定义 Box

```rust
use std::mem::ManuallyDrop;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(value: T) -> Self {
        MyBox(value)
    }

    fn into_raw(this: Self) -> *mut T {
        // 将 Box 转换为裸指针
        unsafe { std::mem::transmute(this) }
    }

    unsafe fn from_raw(ptr: *mut T) -> Self {
        // 从裸指针恢复 Box
        Manually(BoxDrop::new::from_raw(ptr))
    }
}
```

---

## Rust 生态

### Web 开发

| 框架 | 描述 |
|------|------|
| [Axum](https://github.com/tokio-rs/axum) | 现代 Web 框架，基于 Tower |
| [Actix-web](https://actix.rs/) | 高性能 Web 框架 |
| [Rocket](https://rocket.rs/) | 简单易用的 Web 框架 |
| [Warp](https://github.com/seanmonstar/warp) | 基于 Tower 的轻量框架 |

### 异步编程

| 库 | 描述 |
|----|------|
| [Tokio](https://tokio.rs/) | 异步运行时 |
| [async-std](https://async.rs/) | 异步标准库 |
| [futures](https://rust-lang-nursery.github.io/futures-rs/) | 异步抽象 |

### 数据库

| 库 | 描述 |
|----|------|
| [SQLx](https://github.com/launchbadge/sqlx) | 异步 SQL 驱动 |
| [Diesel](https://diesel.rs/) | ORM 框架 |
| [Rusqlite](https://github.com/rusqlite/rusqlite) | SQLite 驱动 |
| [Redis-rs](https://github.com/mitsuhiko/redis-rs) | Redis 客户端 |

### 命令行

| 库 | 描述 |
|----|------|
| [Clap](https://github.com/clap-rs/clap) | 参数解析 |
| [StructOpt](https://github.com/TeXitoi/structopt) | 结构化参数解析 |
| [Indicatif](https://github.com/mitsuhiko/indicatif) | 进度条 |
| [Dialoguer](https://github.com/mitsuhiko/dialoguer) | 交互式 CLI |

### 网络

| 库 | 描述 |
|----|------|
| [reqwest](https://github.com/seanmonstar/reqwest) | HTTP 客户端 |
| [hyper](https://github.com/hyperium/hyper) | HTTP 库 |
| [tonic](https://github.com/hyperium/tonic) | gRPC 框架 |
| [WebSocket](https://github.com/websockets-rs/websockets) | WebSocket |

---

## 进阶学习方向

### 1. 异步编程

```rust
// Cargo.toml
// tokio = { version = "1", features = ["full"] }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 异步函数
    let result = fetch_data().await?;
    println!("{}", result);
    Ok(())
}

async fn fetch_data() -> Result<String, Box<dyn std::error::Error>> {
    let response = reqwest::get("https://httpbin.org/get").await?;
    let body = response.text().await?;
    Ok(body)
}
```

### 2. 宏编程

#### 声明宏

```rust
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}

let v = vec![1, 2, 3];
```

#### 过程宏

```rust
// Cargo.toml
// [lib]
// proc-macro = true

use quote::quote;
use syn;

#[proc_macro]
pub fn make_answer(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::LitStr);
    let answer = format!("Hello, {}!", input.value());

    quote! {
        fn answer() -> String {
            #answer
        }
    }.into()
}
```

### 3. 编译期编程

#### 编译期常量计算

```rust
// 使用 const 函数
const fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

const FIB_10: u32 = fibonacci(10);
```

#### 类型级编程

```rust
// 使用 trait 提供编译时计算
trait Double {
    const DOUBLE: Self;
}

impl Double for i32 {
    const DOUBLE: i32 = i32::MAX * 2;
}
```

---

## 官方学习资源

### 书籍

1. **The Rust Programming Language** (官方书)
   - https://doc.rust-lang.org/book/
   - 免费在线阅读

2. **Programming Rust** (O'Reilly)
   - 深入理解 Rust

3. **Rust for Rustaceans**
   - 进阶读物

### 在线学习

1. **Rust by Example**
   - https://doc.rust-lang.org/rust-by-example/

2. **Rustlings**
   - https://github.com/rust-lang/rustlings/
   - 小练习项目

3. **Exercism Rust Track**
   - https://exercism.org/tracks/rust

### 实践平台

1. **Rust Playground**
   - https://play.rust-lang.org/
   - 在线运行 Rust 代码

2. **Crates.io**
   - https://crates.io/
   - Rust 包仓库

---

## 学习路径建议

```
入门 (1-2周)
├── 安装环境，理解基础语法
├── 掌握所有权系统（最关键）
├── 完成简单练习
│
进阶 (2-4周)
├── Trait 与泛型
├── 错误处理
├── 集合类型与迭代器
├── 基础项目实战
│
深入 (4-8周)
├── 模块系统
├── 并发编程
├── 异步编程 (Tokio)
└── Web 开发
│
专家 (持续)
├── unsafe Rust
├── 宏编程
├── 编译器贡献
└── 领域深耕 (区块链/系统/嵌入式)
```

---

## 小结

### 不安全 Rust
1. **使用场景**：底层系统编程、C 互操作、性能优化
2. **五大能力**：解引用裸指针、调用 unsafe 函数、访问静态变量、实现 unsafe trait、标记 unsafe 函数
3. **安全原则**：最小化 unsafe，封装在安全抽象中

### 生态与进阶
1. **Web 开发**：Axum、Actix-web、Rocket
2. **异步编程**：Tokio、async-std
3. **进阶方向**：异步编程、宏编程、编译期编程
4. **学习资源**：官方文档、Rustlings、Exercism

---

## 总结

恭喜你完成了 Rust 学习笔记的全部内容！

### 核心概念回顾

1. **所有权系统**：Rust 的核心创新，编译期内存安全
2. **借用检查**：确保引用的有效性
3. **Trait**：Rust 的多态基础，比接口更灵活
4. **所有权 + 生命周期**：消除悬垂引用

### 继续学习

- 多写代码，熟练掌握所有权概念
- 阅读优秀开源项目源码
- 参与 Rust 社区
- 尝试贡献开源项目

---

> 💡 **提示**：Rust 的学习曲线较陡，但一旦掌握，你会对系统编程有全新的理解。坚持练习，多写代码！

祝你学习愉快！
