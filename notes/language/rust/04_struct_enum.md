# 结构体与枚举 + 错误处理

> 学习 Rust 中自定义数据类型的定义方式，以及如何安全地处理错误。

## 结构体（Struct）

结构体允许将多个相关的数据组合成一个类型。

### 定义结构体

```rust
// 命名结构体
struct User {
    username: String,
    email: String,
    active: bool,
    sign_in_count: u64,
}

// 创建实例
let user1 = User {
    email: String::from("user@example.com"),
    username: String::from("alice"),
    active: true,
    sign_in_count: 1,
};

// 访问字段
println!("{}", user1.email);

// 修改可变实例
let mut user2 = User {
    email: String::from("test@example.com"),
    username: String::from("bob"),
    active: false,
    sign_in_count: 0,
};
user2.active = true;
```

### 元组结构体

类似元组，但有命名字段：

```rust
struct Color(i32, i32, i32);
struct Point(f64, f64);

let black = Color(0, 0, 0);
let origin = Point(0.0, 0.0);

println!("R = {}", black.0);

// 解构
let Color(r, g, b) = black;
```

### 单元结构体

没有字段的结构体，常用于标记：

```rust
struct AlwaysEqual;

let _placeholder = AlwaysEqual;
```

### 结构体方法（impl 块）

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

// 为 Rectangle 实现方法
impl Rectangle {
    // self 参数表示方法属于实例
    fn area(&self) -> u32 {
        self.width * self.height
    }

    // 可变方法
    fn scale(&mut self, factor: u32) {
        self.width *= factor;
        self.height *= factor;
    }

    // 静态方法（关联函数）
    fn new(width: u32, height: u32) -> Rectangle {
        Rectangle { width, height }
    }

    // 可变借用作为返回值
    fn set_width(&mut self, width: u32) {
        self.width = width;
    }
}

fn main() {
    let rect = Rectangle::new(30, 50);
    println!("面积: {}", rect.area());

    let mut rect2 = Rectangle { width: 10, height: 20 };
    rect2.scale(2);
    println!("缩放后面积: {}", rect2.area());
}
```

### 结构体内存布局

```rust
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

println!("Point3D 大小: {} bytes", std::mem::size_of::<Point3D>());
// 24 bytes (3 * 8)

// 内存对齐优化：按字段大小排序
struct Optimized {
    a: u8,   // 1 byte -> 填充 7 bytes
    b: u64,  // 8 bytes
    c: u32,  // 4 bytes -> 填充 4 bytes
}
// 16 bytes vs 可能更大的非优化版本
```

---

## 枚举（Enum）

枚举表示一组相关的值：

### 基本枚举

```rust
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

let direction = Direction::Up;

match direction {
    Direction::Up => println!("向上"),
    Direction::Down => println!("向下"),
    Direction::Left => println!("向左"),
    Direction::Right => println!("向右"),
}
```

### 带数据的枚举

```rust
enum Message {
    Quit,                      // 无数据
    Move { x: i32, y: i32 },  // 匿名结构体
    Write(String),             // 单值
    ChangeColor(u8, u8, u8), // 元组
}

let m1 = Message::Quit;
let m2 = Message::Move { x: 10, y: 20 };
let m3 = Message::Write(String::from("hello"));
let m4 = Message::ChangeColor(255, 0, 0);
```

### 枚举方法

```rust
impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("退出"),
            Message::Move { x, y } => println!("移动到 ({}, {})", x, y),
            Message::Write(text) => println!("写入: {}", text),
            Message::ChangeColor(r, g, b) => println!("颜色: {}, {}, {}", r, g, b),
        }
    }
}

Message::Write(String::from("test")).call();
```

---

## Option 枚举

Rust 的空值安全机制：

```rust
enum Option<T> {
    Some(T),      // 有值
    None,        // 无值
}

// 使用示例
fn find_user(id: u32) -> Option<String> {
    if id == 1 {
        Some(String::from("Alice"))
    } else {
        None
    }
}

fn main() {
    let user = find_user(1);

    // 方式一：match
    match user {
        Some(name) => println!("找到用户: {}", name),
        None => println!("用户不存在"),
    }

    // 方式二：if let
    if let Some(name) = find_user(1) {
        println!("找到: {}", name);
    }

    // 方式三：unwrap/expect
    let name = find_user(1).unwrap();      // panic if None
    let name = find_user(1).unwrap_or(String::from("默认"));

    // 方式四：map
    let name_len = find_user(1).map(|s| s.len());
}
```

### Option 常用方法

```rust
let numbers = vec![1, 2, 3];

// first/last 返回 Option
numbers.first();   // Some(&1)
numbers.last();    // Some(&3)
numbers.get(10);  // None

// unwrap 系列
let x = Some(5).unwrap();     // 5
let x = None.unwrap_or(0);    // 0（默认值）
let x = None.unwrap_or_else(|| compute_default());

// map 转换
let x = Some(5).map(|v| v * 2);  // Some(10)
let x = None.map(|v| v * 2);     // None

// and_then 链式调用
fn get_user(id: u32) -> Option<String> { ... }
fn get_email(name: &str) -> Option<String> { ... }

let email = get_user(1)
    .and_then(|name| get_email(&name));
```

---

## 模式匹配（match）

### match 基本用法

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u32 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

// 匹配所有情况（exhaustive）
```

### 匹配 Option

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

let five = Some(5);
let six = plus_one(five);   // Some(6)
let none = plus_one(None);  // None
```

### match 守卫

```rust
let num = Some(5);

match num {
    Some(x) if x % 2 == 0 => println!("偶数: {}", x),
    Some(x) => println!("奇数: {}", x),
    None => println!("无值"),
}
```

### if let 简写

```rust
// 简单情况使用 if let 更简洁
let some_value = Some(3);

if let Some(x) = some_value {
    println!("值为: {}", x);
}

// 等价于
match some_value {
    Some(x) => println!("值为: {}", x),
    _ => {},
}
```

---

## 错误处理

### Result 枚举

Rust 的可恢复错误处理：

```rust
enum Result<T, E> {
    Ok(T),   // 成功，携带值
    Err(E),  // 错误，携带错误信息
}

// 文件操作示例
use std::fs::File;

fn open_file(path: &str) -> Result<File, std::io::Error> {
    let f = File::open(path)?;

    // 或显式处理
    // let f = match File::open(path) {
    //     Ok(file) => file,
    //     Err(e) => return Err(e),
    // };

    Ok(f)
}
```

### 处理 Result

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() {
    // 方式一：match
    match read_file("test.txt") {
        Ok(contents) => println!("文件内容: {}", contents),
        Err(e) => eprintln!("读取失败: {}", e),
    }

    // 方式二：unwrap / expect
    let contents = read_file("test.txt").unwrap();
    let contents = read_file("test.txt").expect("读取文件失败");

    // 方式三：if let
    if let Ok(contents) = read_file("test.txt") {
        println!("{}", contents);
    }
}
```

### ? 操作符

错误传播的简写：

```rust
use std::fs::File;
use std::io;
use std::io::Read;

fn read_file_contents(path: &str) -> Result<String, io::Error> {
    // ? 等价于：
    // let mut file = match File::open(path) {
    //     Ok(f) => f,
    //     Err(e) => return Err(e.into()),
    // };

    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// 链式使用 ?
fn get_home_dir() -> Result<String, std::env::VarError> {
    let home = std::env::var("HOME")?;
    let config_path = format!("{}/.config", home);
    Ok(config_path)
}
```

### 自定义错误类型

```rust
use std::fmt;

#[derive(Debug)]
enum MyError {
    IoError(std::io::Error),
    ParseError(std::num::ParseIntError),
    Custom(String),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyError::IoError(e) => write!(f, "IO错误: {}", e),
            MyError::ParseError(e) => write!(f, "解析错误: {}", e),
            MyError::Custom(msg) => write!(f, "自定义错误: {}", msg),
        }
    }
}

impl std::error::Error for MyError {}

fn parse_number(s: &str) -> Result<i32, MyError> {
    let num = s.parse::<i32>().map_err(MyError::ParseError)?;
    Ok(num)
}
```

### panic! 不可恢复错误

```rust
// 触发 panic
panic!("程序崩溃");

// 数组越界会 panic
let v = vec![1, 2, 3];
let _ = v[10];  // panic!

// unwrap 系列
let x = Some(5).unwrap();      // OK
let x = None.unwrap();          // panic!

let x = Some(5).expect("msg");  // OK
let x = None.expect("msg");    // panic!

// 何时使用 panic：
// - 原型开发时
// - 测试代码中
// - 确实不可能发生的情况
```

### 错误处理建议

```rust
// 1. 库代码：返回 Result
// 2. 应用代码：使用 ? 或 unwrap

// 在 main 中使用 ?
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string("test.txt")?;
    println!("{}", contents);
    Ok(())
}

// 使用 match 的完整示例
use std::fs::File;
use std::io::{self, Read};

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() {
    match read_file("config.txt") {
        Ok(data) => {
            println!("配置: {}", data);
        }
        Err(e) => {
            eprintln!("读取配置失败: {}", e);
            std::process::exit(1);
        }
    }
}
```

---

## 小结

本章我们学习了：

### 结构体
1. **命名结构体**：组合多个相关字段
2. **元组结构体**：带索引的命名类型
3. **单元结构体**：标记类型
4. **impl 块**：方法与关联函数

### 枚举与模式匹配
1. **基本枚举**：一组相关值
2. **带数据的枚举**：每个变体可携带不同数据
3. **Option 枚举**：安全的空值处理
4. **match**：穷尽匹配
5. **if let**：简化单分支匹配

### 错误处理
1. **Result<T, E>**：可恢复错误
2. **? 操作符**：错误传播
3. **panic!**：不可恢复错误
4. **自定义错误**：实现 Error trait

下一章节我们将学习泛型、Trait 和集合类型。
