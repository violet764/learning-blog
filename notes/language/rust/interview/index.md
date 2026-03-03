# Rust 核心面试题

> 高频面试问题，每题包含题目、答案、重点追问

---

## 一、所有权系统

### Q1：所有权三大规则是什么？

**答案：**

1. **每个值有且只有一个所有者**
2. **同一时刻只能有一个所有者**
3. **所有者离开作用域，值被释放**

```rust
let s1 = String::from("hello");
let s2 = s1;  // 所有权转移，s1 失效
// println!("{}", s1);  // 编译错误！

{
    let s3 = String::from("world");
}  // s3 在这里被 drop
```

**追问：Move 和 Copy 的区别？**
> Move：所有权转移，原变量失效（String、Vec 等）
> Copy：复制值，原变量仍有效（i32、f64 等基本类型）
> ```rust
> let x = 5;
> let y = x;  // Copy，x 仍有效
> let s1 = String::from("hello");
> let s2 = s1;  // Move，s1 失效
> ```

---

### Q2：借用规则是什么？

**答案：**

**两大规则：**
1. 要么多个不可变引用，要么一个可变引用
2. 引用必须始终有效

```rust
let mut s = String::from("hello");

let r1 = &s;
let r2 = &s;     // OK：多个不可变引用
// let r3 = &mut s;  // 错误！已有不可变引用

let r4 = &mut s; // OK：可变引用
// let r5 = &s;  // 错误！已有可变引用
```

**追问：为什么不能同时存在？**
> 不可变引用承诺值不变，可变引用会修改值，矛盾。这是为了防止数据竞争。

---

### Q3：什么是 NLL（非词法生命周期）？

**答案：**

NLL 使引用的生命周期到最后一次使用位置，而非作用域结束。

```rust
let mut s = String::from("hello");
let r = &s;
println!("{}", r);  // r 最后使用
s.push_str(" world");  // OK！r 已失效（NLL）
```

---

### Q4：生命周期标注的作用是什么？

**答案：**

生命周期标注帮助编译器验证引用的有效性，确保引用不会悬垂。

```rust
// 返回两个字符串中较长的
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 结构体包含引用
struct Excerpt<'a> {
    content: &'a str,
}
```

**省略规则：**
1. 每个引用参数获得一个生命周期
2. 只有一个输入时，赋给所有输出
3. 有 &self 时，self 的生命周期赋给输出

**追问：'static 是什么意思？**
> 'static 表示整个程序运行期间有效。字符串字面量就是 'static 生命周期：
> ```rust
> let s: &'static str = "hello world";
> ```

---

### Q5：哪些类型实现了 Copy？

**答案：**

自动实现 Copy 的类型：
- 所有整数、浮点类型（i32, f64 等）
- 布尔类型 bool
- 字符类型 char
- 元组（所有元素都是 Copy）
- 数组（元素是 Copy）

不实现 Copy 的类型：
- String、Vec、Box、HashMap 等
- 包含非 Copy 字段的自定义类型

```rust
#[derive(Copy, Clone)]
struct Point { x: i32, y: i32 }  // OK

struct Bad { name: String }  // 不能实现 Copy
```

---

## 二、类型系统

### Q6：Trait 是什么？与接口的区别？

**答案：**

Trait 定义一组方法签名，描述类型"能做什么"。

```rust
trait Summary {
    fn summarize(&self) -> String;
    
    // 默认实现
    fn author(&self) -> String {
        String::from("Unknown")
    }
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, self.content)
    }
}
```

**与接口的区别：**
- 可以在任何地方实现（不限于定义处）
- 支持默认实现
- 支持 Trait Bound（静态分发）
- 支持 Trait Object（动态分发）

---

### Q7：Trait Bound 如何使用？

**答案：**

```rust
// 泛型约束
fn notify<T: Summary>(item: &T) {
    println!("{}", item.summarize());
}

// 多重约束
fn process<T: Summary + Clone>(item: &T) {}

// where 子句（更清晰）
fn complex<T, U>(t: &T, u: &U)
where
    T: Summary + Clone,
    U: std::fmt::Display,
{}

// impl Trait 语法
fn print(item: &impl Summary) {}
```

---

### Q8：静态分发和动态分发的区别？

**答案：**

| 特性 | 静态分发 | 动态分发 |
|------|----------|----------|
| 实现 | 泛型 + Trait Bound | dyn Trait |
| 决定时机 | 编译时 | 运行时 |
| 性能 | 更好（单态化） | 略差（vtable） |
| 代码大小 | 更大 | 更小 |
| 灵活性 | 较低 | 更高 |

```rust
// 静态分发
fn static_dispatch<T: Summary>(item: &T) {}

// 动态分发
fn dynamic_dispatch(item: &dyn Summary) {}

// Trait Object 集合
let items: Vec<Box<dyn Summary>> = vec![...];
```

**追问：什么时候用动态分发？**
> 需要存储不同类型的集合、编译大小敏感、类型运行时确定时。

---

### Q9：什么是孤儿规则？

**答案：**

孤儿规则：只有当 trait 或类型至少有一个是当前 crate 定义的，才能为类型实现 trait。

```rust
// 错误：不能为外部类型实现外部 trait
// impl Display for Vec<i32> {}

// 解决：Newtype 模式
struct MyVec(Vec<i32>);
impl Display for MyVec { ... }
```

作用：防止不同 crate 的实现冲突。

---

### Q10：关联类型是什么？

**答案：**

关联类型让 Trait 更灵活，避免每次使用都指定泛型参数。

```rust
trait Container {
    type Item;  // 关联类型
    fn get(&self) -> Option<&Self::Item>;
    fn add(&mut self, item: Self::Item);
}

impl Container for MyBox {
    type Item = i32;  // 指定具体类型
    fn get(&self) -> Option<&i32> { ... }
}
```

对比泛型：
```rust
// 泛型：可以为同一类型实现多次
trait Container<T> { ... }
impl Container<i32> for MyBox { ... }
impl Container<String> for MyBox { ... }

// 关联类型：只能实现一次
trait Container { type Item; }
impl Container for MyBox { type Item = i32; }
// 不能再为 MyBox 实现其他 Item 类型
```

---

## 三、智能指针

### Q11：智能指针如何选择？

**答案：**

| 类型 | 所有权 | 线程安全 | 用途 |
|------|--------|----------|------|
| Box | 独占 | 是 | 堆分配、递归类型 |
| Rc | 共享 | 否 | 单线程共享 |
| Arc | 共享 | 是 | 多线程共享 |
| RefCell | 独占 | 否 | 内部可变性 |
| Mutex | 独占 | 是 | 多线程可变 |
| Cell | 独占 | 否 | Copy 类型的内部可变性 |

```rust
// Box：堆分配
let b = Box::new(5);

// Rc：单线程共享
let a = Rc::new(5);
let b = Rc::clone(&a);

// Arc：多线程共享
let data = Arc::new(vec![1, 2, 3]);

// RefCell：内部可变性
let data = RefCell::new(5);
*data.borrow_mut() += 1;
```

---

### Q12：Rc 和 Arc 的区别？

**答案：**

| 特性 | Rc | Arc |
|------|----|----|
| 引用计数 | 普通 | 原子操作 |
| 性能 | 更快 | 略慢 |
| 线程安全 | 仅单线程 | 可跨线程 |
| Send trait | 未实现 | 实现 |

```rust
// Rc 不能跨线程
let rc = Rc::new(5);
// thread::spawn(move || { let _ = rc; });  // 编译错误！

// Arc 可以
let arc = Arc::new(5);
let arc_clone = Arc::clone(&arc);
thread::spawn(move || { let _ = arc_clone; });  // OK
```

---

### Q13：什么是内部可变性？

**答案：**

内部可变性允许通过不可变引用修改数据。

```rust
use std::cell::RefCell;

struct Cache {
    data: RefCell<Vec<i32>>,
}

impl Cache {
    fn add(&self, value: i32) {  // &self 是不可变的
        self.data.borrow_mut().push(value);  // 但可以修改
    }
}
```

**Cell vs RefCell：**
```rust
// Cell：用于 Copy 类型，无借用检查
let c = Cell::new(5);
c.set(10);

// RefCell：用于非 Copy 类型，运行时借用检查
let r = RefCell::new(String::from("hello"));
r.borrow_mut().push_str(" world");
```

---

### Q14：循环引用如何解决？

**答案：**

使用 weak_ptr 打破循环引用。

```rust
use std::rc::{Rc, Weak};

struct Node {
    value: i32,
    next: Option<Rc<Node>>,
    prev: Option<Weak<Node>>,  // 用 Weak 打破循环
}

fn main() {
    let a = Rc::new(Node { value: 1, next: None, prev: None });
    let b = Rc::new(Node { value: 2, next: None, prev: Some(Rc::downgrade(&a)) });
    
    // Weak 不增加引用计数
    // 当需要访问时：weak.upgrade() 返回 Option<Rc<Node>>
}
```

---

## 四、错误处理

### Q15：Option 和 Result 的区别？

**答案：**

| 类型 | 用途 |
|------|------|
| Option | 值可能不存在 |
| Result | 操作可能失败 |

```rust
// Option
fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 { None } else { Some(a / b) }
}

// Result
fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
```

---

### Q16：如何处理 Option？

**答案：**

```rust
let opt = Some(5);

// match
match opt {
    Some(v) => println!("{}", v),
    None => println!("None"),
}

// if let
if let Some(v) = opt {
    println!("{}", v);
}

// unwrap 系列
let v = opt.unwrap();           // None 会 panic
let v = opt.unwrap_or(0);       // None 返回默认值
let v = opt.expect("必须存在"); // None panic 带信息

// map / and_then
opt.map(|v| v * 2);
opt.and_then(|v| Some(v + 1));
```

---

### Q17：? 操作符的原理？

**答案：**

`?` 做了两件事：
1. 如果是 Ok/Some，提取值继续
2. 如果是 Err/None，提前返回

```rust
fn read_file(path: &str) -> Result<String, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    // 等价于
    // let content = match std::fs::read_to_string(path) {
    //     Ok(c) => c,
    //     Err(e) => return Err(e),
    // };
    Ok(content)
}

// 可以链式调用
fn process() -> Result<String, std::io::Error> {
    let content = File::open("data.txt")?
        .read_to_string(&mut String::new())?;
    Ok(content)
}
```

---

### Q18：什么时候用 panic，什么时候用 Result？

**答案：**

| panic | Result |
|-------|--------|
| 不可恢复错误 | 可恢复错误 |
| 程序逻辑错误 | 预期内的失败 |
| 示例：数组越界 | 示例：文件不存在 |
| 调用者无法处理 | 调用者可以处理 |

```rust
// panic：程序错误
fn get(items: &[i32], index: usize) -> i32 {
    items[index]  // 越界会 panic，这是合理的
}

// Result：预期失败
fn read_config(path: &str) -> Result<Config, Error> {
    // 文件不存在是预期内的，用 Result
}
```

---

## 五、并发编程

### Q19：Send 和 Sync 是什么？

**答案：**

- **Send**：类型可以安全地在线程间转移所有权
- **Sync**：类型可以安全地在线程间共享引用（&T 是 Send）

```rust
// Send：可以 move 到其他线程
let data = vec![1, 2, 3];
thread::spawn(move || {
    println!("{:?}", data);  // data 被移动
});

// Sync：可以跨线程共享引用
let data = Arc::new(vec![1, 2, 3]);  // Arc<T> 实现了 Sync
```

**自动实现：**
- 大多数类型自动实现 Send + Sync
- 不自动实现：裸指针、Rc、RefCell

---

### Q20：如何在多线程间共享数据？

**答案：**

**方式1：Channel 消息传递**
```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();
thread::spawn(move || {
    tx.send("hello").unwrap();
});
let received = rx.recv().unwrap();
```

**方式2：Arc\<Mutex\> 共享状态**
```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let counter_clone = Arc::clone(&counter);

thread::spawn(move || {
    let mut num = counter_clone.lock().unwrap();
    *num += 1;
});
```

**方式3：Arc\<RwLock\> 读多写少**
```rust
let data = Arc::new(RwLock::new(vec![1, 2, 3]));

// 多个读者
let r1 = data.read().unwrap();
let r2 = data.read().unwrap();  // OK

// 写者独占
let mut w = data.write().unwrap();
```

---

### Q21：Mutex 和 RwLock 如何选择？

**答案：**

| 特性 | Mutex | RwLock |
|------|-------|--------|
| 读操作 | 独占 | 共享 |
| 写操作 | 独占 | 独占 |
| 适用场景 | 读写都多 | 读多写少 |
| 开销 | 较小 | 较大 |

```rust
// 读写都频繁：Mutex
let data = Arc::new(Mutex::new(HashMap::new()));

// 读多写少：RwLock
let cache = Arc::new(RwLock::new(HashMap::new()));
```

---

### Q22：如何避免死锁？

**答案：**

1. **固定加锁顺序**
2. **使用 try_lock 避免无限等待**
3. **减少锁持有时间**
4. **考虑无锁数据结构**

```rust
// 避免嵌套锁
// 错误
let mut a = mutex1.lock().unwrap();
let mut b = mutex2.lock().unwrap();  // 可能死锁

// 正确：同时锁定
use std::sync::MutexGuard;
let (a, b) = lock_both(&mutex1, &mutex2);
```

---

## 六、常见问题

### Q23：如何解决"同时存在可变和不可变引用"错误？

**答案：**

```rust
// 错误
let mut v = vec![1, 2, 3];
let first = &v[0];
v.push(4);  // 错误！

// 解决1：先复制，再修改
let first = v[0];  // Copy，不是借用
v.push(4);

// 解决2：分离作用域
{
    let first = &v[0];
    println!("{}", first);
}
v.push(4);  // first 已失效

// 解决3：使用 RefCell（运行时检查）
let v = RefCell::new(vec![1, 2, 3]);
let first = v.borrow()[0];
v.borrow_mut().push(4);
```

---

### Q24：如何解决"返回局部变量引用"错误？

**答案：**

```rust
// 错误
fn bad() -> &String {
    &String::from("hello")  // 返回临时值引用
}

// 解决1：返回所有权
fn good() -> String {
    String::from("hello")
}

// 解决2：传入可变引用
fn append(s: &mut String) {
    s.push_str(" world");
}

// 解决3：返回静态引用
fn get_static() -> &'static str {
    "hello"
}
```

---

### Q25：如何解决"迭代时修改集合"错误？

**答案：**

```rust
// 错误
let mut map = HashMap::new();
for (k, v) in &map {
    map.insert(*k, v + 1);  // 借用冲突！
}

// 解决1：先收集 key
let keys: Vec<_> = map.keys().cloned().collect();
for k in keys {
    map.entry(k).and_modify(|v| *v += 1);
}

// 解决2：使用 drain 或 into_iter
let old: HashMap<_, _> = map.drain().collect();
for (k, v) in old {
    map.insert(k, v + 1);
}

// 解决3：函数式方法
let new_map: HashMap<_, _> = map.into_iter()
    .map(|(k, v)| (k, v + 1))
    .collect();
```

---

### Q26：closure 的 Fn、FnMut、FnOnce 有什么区别？

**答案：**

| Trait | 捕获方式 | 调用次数 |
|-------|----------|----------|
| FnOnce | 获取所有权 | 一次 |
| FnMut | 可变引用 | 多次 |
| Fn | 不可变引用 | 多次 |

```rust
// FnOnce：消费捕获的变量
let s = String::from("hello");
let f = || drop(s);  // s 被 move
f();  // 只能调用一次

// FnMut：修改捕获的变量
let mut x = 0;
let mut f = || { x += 1; };
f();
f();  // 可以多次调用

// Fn：只读访问
let x = 10;
let f = || x + 1;  // 只读
f();
f();  // 可以多次调用
```

---

## 七、高级特性

### Q27：什么是泛型约束（where 子句）？

**答案：**

```rust
// 基本约束
fn foo<T: Clone + Default>(value: T) {}

// where 子句（更清晰）
fn bar<T, U>(t: T, u: U)
where
    T: Clone + std::fmt::Debug,
    U: Default + Into<String>,
{}

// 复杂约束
fn complex<T>()
where
    T: Iterator,
    T::Item: Clone + std::fmt::Debug,
{}
```

---

### Q28：什么是特征对象（Trait Object）？

**答案：**

特征对象是在运行时确定具体类型的动态类型。

```rust
// Trait Object：dyn Trait
let items: Vec<Box<dyn Summary>> = vec![
    Box::new(Article { ... }),
    Box::new(Tweet { ... }),
];

// 使用
for item in items {
    println!("{}", item.summarize());
}

// 限制：不能使用返回 Self 的方法
// trait Bad {
//     fn clone(&self) -> Self;  // 不能用于 dyn Bad
// }
```

---

### Q29：什么是 Newtype 模式？

**答案：**

Newtype 模式：用结构体包装一个类型，获得新类型。

```rust
// 为外部类型实现外部 trait
struct MyVec(Vec<i32>);

impl Display for MyVec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

// 类型安全
struct Miles(u32);
struct Kilometers(u32);

fn add_miles(a: Miles, b: Miles) -> Miles {
    Miles(a.0 + b.0)
}
// add_miles(Miles(1), Kilometers(1));  // 编译错误！
```

---

### Q30：什么是 Deref 强制转换？

**答案：**

Deref 强制转换：当类型实现 Deref trait 时，可以自动转换为引用。

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn hello(name: &str) {
    println!("Hello, {}!", name);
}

let m = MyBox(String::from("Rust"));
hello(&m);  // 自动转换为 &String，再转换为 &str
```

---

## 八、高频代码题

### Q31：实现线程安全的计数器

**答案：**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Counter {
    count: AtomicUsize,
}

impl Counter {
    fn new() -> Self {
        Counter { count: AtomicUsize::new(0) }
    }
    
    fn increment(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
    
    fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

fn main() {
    let counter = Arc::new(Counter::new());
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            for _ in 0..1000 {
                counter.increment();
            }
        }));
    }
    
    for h in handles { h.join().unwrap(); }
    println!("Count: {}", counter.get());  // 10000
}
```

---

### Q32：实现简单的 Option 处理

**答案：**

```rust
// 安全除法
fn safe_divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 { None } else { Some(a / b) }
}

// 使用
match safe_divide(10, 2) {
    Some(result) => println!("结果: {}", result),
    None => println!("除零错误"),
}

// 链式处理
fn process(a: i32, b: i32) -> Option<i32> {
    safe_divide(a, b)
        .map(|x| x * 2)
        .filter(|&x| x > 0)
}

// unwrap_or 提供默认值
let result = safe_divide(10, 0).unwrap_or(0);
```

---

### Q33：实现带生命周期的结构体

**答案：**

```rust
struct Excerpt<'a> {
    content: &'a str,
}

impl<'a> Excerpt<'a> {
    fn new(content: &'a str) -> Self {
        Excerpt { content }
    }
    
    fn get(&self) -> &'a str {
        self.content
    }
    
    fn first_word(&self) -> Option<&'a str> {
        self.content.split_whitespace().next()
    }
}

fn main() {
    let text = String::from("Hello, world!");
    let excerpt = Excerpt::new(&text);
    println!("{}", excerpt.get());
}
```

---

### Q34：实现简单的 Result 处理

**答案：**

```rust
use std::fs::File;
use std::io::Read;

fn read_file(path: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

fn main() {
    match read_file("data.txt") {
        Ok(content) => println!("{}", content),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// 使用 ? 传播错误
fn process_file() -> Result<i32, std::io::Error> {
    let content = read_file("data.txt")?;
    Ok(content.lines().count() as i32)
}
```

---

### Q35：实现简单的自定义错误类型

**答案：**

```rust
use std::fmt;
use std::error::Error;

#[derive(Debug)]
struct MyError {
    message: String,
}

impl MyError {
    fn new(msg: &str) -> Self {
        MyError { message: msg.to_string() }
    }
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MyError: {}", self.message)
    }
}

impl Error for MyError {}

// 使用
fn divide(a: i32, b: i32) -> Result<i32, MyError> {
    if b == 0 {
        Err(MyError::new("division by zero"))
    } else {
        Ok(a / b)
    }
}
```

---

## 📝 面试技巧

### Rust 面试重点
1. **所有权系统**：核心概念，必须理解透彻
2. **借用检查**：常见问题来源
3. **生命周期**：高级话题，展示深度
4. **并发安全**：Send/Sync 是 Rust 亮点

### 常见追问方向
- "为什么这样设计？"
- "和 GC/手动管理有什么区别？"
- "如何解决这类编译错误？"
- "线程安全是怎么保证的？"

### 常见失误
- 借用规则混淆
- 生命周期标注错误
- 智能指针选择不当
- 忘记处理 Option/Result
