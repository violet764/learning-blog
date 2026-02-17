# 基础语法与类型系统

> 掌握 Rust 的变量声明、基本数据类型、函数定义和流程控制结构。

## 变量与可变性

### 变量声明

Rust 使用 `let` 声明变量，默认不可变：

```rust
fn main() {
    let x = 5;           // 不可变变量
    let mut y = 10;      // 可变变量（需加 mut）

    println!("x = {}", x);
    println!("y = {}", y);

    y = 20;              // 合法：y 是可变的
    // x = 6;            // 错误：x 是不可变的
}
```

### mut 关键字

`mut`（mutable）用于声明可变绑定：

```rust
let mut counter = 0;
counter = counter + 1;  // OK
```

### 变量遮蔽（Shadowing）

可以在同一作用域内重新声明同名变量，实现"遮蔽"：

```rust
fn main() {
    let x = 5;
    let x = x + 1;      // 遮蔽：创建新变量
    let x = x * 2;      // 再次遮蔽

    println!("x = {}", x);  // 输出: x = 12
}
```

**遮蔽 vs mut：**

| 特性 | 遮蔽（let） | mut |
|------|------------|-----|
| 类型可变化 | ✅ | ❌ |
| 可重新赋值 | ✅（创建新变量） | ✅（修改原值） |
| 编译期行为 | 编译期安全 | 运行时可修改 |

```rust
// 遮蔽可以改变类型
let s = "hello";
let s = s.len();  // 类型从 &str 变成 usize

// mut 不能改变类型
let mut s = "hello";
// s = s.len();  // 错误：类型不匹配
```

### 常量

常量使用 `const` 声明，必须标注类型：

```rust
const MAX_SIZE: u32 = 100;
const PI: f64 = 3.1415926;

// 编译时常量（可在数组大小中使用）
const ARRAY_SIZE: usize = 100;
let arr = [0; ARRAY_SIZE];
```

---

## 基本数据类型

Rust 是静态类型语言，编译器在编译时确定所有类型。

### 标量类型

#### 整型

| 长度 | 有符号 | 无符号 | 用途 |
|------|--------|--------|------|
| 8 位 | `i8` | `u8` | 小整数、字节 |
| 16 位 | `i16` | `u16` | |
| 32 位 | `i32` | `u32` | 默认整数类型 |
| 64 位 | `i64` | `u64` | 大整数、时间戳 |
| 128 位 | `i128` | `u128` | 特殊场景 |
| 平台相关 | `isize` | `usize` | 指针大小、索引 |

```rust
// 十进制
let decimal = 98_222;  // 98222

// 十六进制
let hex = 0xff;

// 八进制
let octal = 0o77;

// 二进制
let binary = 0b1111_0000;

// 字节（仅 u8）
let byte = b'A';
```

#### 浮点型

```rust
let float64 = 3.14;      // f64，默认类型
let float32: f32 = 3.14; // 显式指定

// 特殊值
let inf: f64 = f64::INFINITY;
let neg_inf = f64::NEG_INFINITY;
let nan = f64::NAN;
```

#### 布尔型

```rust
let is_rust_awesome = true;
let is_difficult = false;
```

#### 字符类型

Rust 的 `char` 是 Unicode 标量值，占 4 字节：

```rust
let c1 = 'z';
let c2 = 'ℤ';
let emoji = '😀';

println!("字符占用 {} 字节", std::mem::size_of::<char>());  // 4 字节
```

### 复合类型

#### 元组（Tuple）

固定长度、多种类型的集合：

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);

// 解构
let (x, y, z) = tup;
println!("y = {}", y);

// 索引访问
println!("x = {}", tup.0);
```

#### 数组（Array）

同类型固定长度集合：

```rust
let arr = [1, 2, 3, 4, 5];
let months: [&str; 12] = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
];

// 访问元素
let first = arr[0];
let last = arr[4];

// 越界访问会导致 panic
// let out_of_bounds = arr[10];  // panic!
```

### 类型转换

Rust 不支持隐式类型转换，需要显式使用 `as` 或 `From/Into`：

```rust
let a: i32 = 10;
let b: i64 = a as i64;  // as 转换

// 使用 trait
let c: i32 = i64::from(a);  // From
let d: i64 = a.into();       // Into
```

---

## 函数

### 函数定义

```rust
fn function_name(param1: Type1, param2: Type2) -> ReturnType {
    // 函数体
    // 最后一行作为返回值（无分号）
    result
}

// 无返回值 -> 等同于 ()
fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

### 返回值

```rust
// 方式一：最后表达式（无分号）
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// 方式二：使用 return
fn subtract(a: i32, b: i32) -> i32 {
    return a - b;
}

// 方式三：空返回 / 提前返回
fn early_return(x: i32) -> i32 {
    if x < 0 {
        return 0;  // 提前返回
    }
    x
}
```

### 参数传递

```rust
// 值传递（拷贝）
fn print_value(mut x: i32) {
    x = x + 1;
    println!("{}", x);
}

fn main() {
    let v = 5;
    print_value(v);
    println!("{}", v);  // 仍是 5
}

// 引用传递
fn print_ref(x: &mut i32) {
    *x = *x + 1;
    println!("{}", *x);
}

fn main() {
    let mut v = 5;
    print_ref(&mut v);
    println!("{}", v);  // 变成 6
}
```

### 函数命名：蛇形命名法

```rust
fn calculate_total_price() {}  // ✅ 正确
fn calculateTotalPrice() {}    // ❌ 错误（驼峰命名）
fn CalculateTotalPrice() {}     // ❌ 错误（帕斯卡命名）
```

---

## 流程控制

### 条件判断：if / else

```rust
let number = 6;

if number % 4 == 0 {
    println!("能被 4 整除");
} else if number % 2 == 0 {
    println!("是偶数但不能被 4 整除");
} else {
    println!("是奇数");
}

// if 作为表达式（必须有 else）
let result = if number > 10 { "大" } else { "小" };
```

### 循环：loop / while / for

#### loop：无限循环

```rust
loop {
    println!("无限循环");
    break;  // 退出循环
}

// 带返回值
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;  // 返回值
    }
};
```

#### while：条件循环

```rust
let mut number = 3;
while number != 0 {
    println!("{}", number);
    number -= 1;
}
println!("发射！");
```

#### for：迭代循环

```rust
// 遍历范围
for i in 0..5 {
    println!("{}", i);  // 0, 1, 2, 3, 4
}

// 遍历数组
let arr = [10, 20, 30];
for element in arr.iter() {
    println!("{}", element);
}

// 带索引
for (index, value) in arr.iter().enumerate() {
    println!("{}: {}", index, value);
}

// 遍历字符串
for c in "hello".chars() {
    println!("{}", c);
}
```

### 流程控制关键字

```rust
// break：退出循环
// continue：跳过本次迭代
for i in 0..10 {
    if i % 2 == 0 {
        continue;  // 跳过偶数
    }
    if i == 7 {
        break;     // 遇到 7 退出
    }
    println!("{}", i);
}

// 嵌套循环 + 标签
'outer: for i in 0..3 {
    'inner: for j in 0..3 {
        if i == 1 && j == 1 {
            break 'outer;  // 直接跳出外层循环
        }
    }
}
```

---

## 代码组织：语句与表达式

Rust 中区分**语句**（Statement）和**表达式**（Expression）：

- **语句**：执行操作，不返回值
- **表达式**：计算并返回值

```rust
// 语句示例
let x = 5;           // 声明语句
let y = {
    let z = 10;
    z + 5             // 表达式：返回 15（注意无分号）
};                   // 语句结束

// 函数也是表达式
let sq = square(3);

fn square(x: i32) -> i32 {
    x * x  // 表达式作为返回值
}
```

---

## 小结

本章我们学习了：

1. **变量与可变性**：`let` 声明、`mut` 可变、变量遮蔽、`const` 常量
2. **基本数据类型**：
   - 标量类型：整型（i8-u128, isize, usize）、浮点型（f32, f64）、布尔型、字符型
   - 复合类型：元组、数组
3. **函数**：定义、参数、返回值、引用传递
4. **流程控制**：if/else、loop/while/for、break/continue

下一章我们将深入 Rust 最核心的概念——所有权与借用系统。
