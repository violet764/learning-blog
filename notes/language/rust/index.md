# Rust 编程语言指南

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Rust_programming_language_black_logo.svg/120px-Rust_programming_language_black_logo.svg.png" alt="Rust Logo" width="120">
</div>

> Rust 是一门安全、并发、实用的系统级编程语言，以独特的 ownership 系统和 borrow checker 实现内存安全，无需垃圾回收器。广泛应用于系统编程、WebAssembly、区块链、命令行工具等领域。

## 📚 文档结构概览

本套学习材料整合为 8 个章节，从入门到进阶系统地掌握 Rust 编程。

| 章节 | 文件 | 内容重点 | 学习难度 | 预计学习时间 |
|:----:|------|----------|:--------:|:------------:|
| 第1章 | [初识Rust](./初识Rust.md) | 设计理念、环境安装、第一个程序 | ⭐⭐ | 2-3小时 |
| 第2章 | [基础语法](./基础语法.md) | 变量、类型、函数、流程控制 | ⭐⭐ | 4-5小时 |
| 第3章 | [所有权与借用](./所有权与借用.md) | 所有权、借用、切片、字符串 | ⭐⭐⭐⭐ | 6-8小时 |
| 第4章 | [结构体与枚举](./结构体与枚举.md) | 结构体、枚举、模式匹配、错误处理 | ⭐⭐⭐ | 5-6小时 |
| 第5章 | [泛型与特征](./泛型与特征.md) | 泛型、Trait、集合类型、迭代器 | ⭐⭐⭐⭐ | 6-8小时 |
| 第6章 | [模块与并发](./模块与并发.md) | 模块系统、包管理、并发编程 | ⭐⭐⭐⭐ | 6-8小时 |
| 第7章 | [实战与优化](./实战与优化.md) | 实战项目、性能优化 | ⭐⭐⭐⭐ | 6-8小时 |
| 第8章 | [不安全Rust](./不安全Rust.md) | 不安全 Rust、生态、进阶方向 | ⭐⭐⭐⭐⭐ | 5-6小时 |

---

## 📖 核心学习内容

### 第一部分：入门与基础

#### 第1章 - Rust 初识
- [设计理念与适用场景](./初识Rust.md#设计理念与适用场景) - 内存安全、零成本抽象、并发安全
- [环境安装](./初识Rust.md#环境安装) - rustup/cargo 安装与配置
- [第一个程序](./初识Rust.md#第一个程序) - Hello World 与 Cargo 基础命令

#### 第2章 - 基础语法与类型系统
- [变量与可变性](./基础语法.md#变量与可变性) - let、mut、变量遮蔽、const
- [基本数据类型](./基础语法.md#基本数据类型) - 整型、浮点型、布尔型、字符型、元组、数组
- [函数](./基础语法.md#函数) - 函数定义、参数传递、返回值
- [流程控制](./基础语法.md#流程控制) - if/else、loop/while/for

#### 第3章 - 所有权与借用
- [所有权系统](./所有权与借用.md#所有权系统) - 三大规则、移动/复制语义
- [借用规则与生命周期](./所有权与借用.md#借用规则与生命周期) - 不可变/可变借用、借用规则
- [切片](./所有权与借用.md#切片slice) - &\[T\]、&str 切片
- [字符串](./所有权与借用.md#字符串) - String vs &str、UTF-8 编码

---

### 第二部分：核心特性

#### 第4章 - 结构体与枚举
- [结构体](./结构体与枚举.md#结构体struct) - 命名/元组/单元结构体、impl 块、方法
- [枚举](./结构体与枚举.md#枚举enum) - 枚举定义、带数据的枚举
- [Option 枚举](./结构体与枚举.md#option-枚举) - 安全的空值处理
- [模式匹配](./结构体与枚举.md#模式匹配match) - match、if let
- [错误处理](./结构体与枚举.md#错误处理) - Result、? 操作符、panic!、自定义错误

#### 第5章 - 泛型与特征
- [泛型](./泛型与特征.md#泛型) - 泛型函数、结构体、枚举、方法
- [Trait](./泛型与特征.md#trait) - 定义、实现、Trait Bound、默认实现
- [常用内置 Trait](./泛型与特征.md#常用内置-trait) - Debug、Clone、Copy、Eq、Ord 等
- [集合类型](./泛型与特征.md#集合类型) - Vec、HashMap、HashSet
- [迭代器](./泛型与特征.md#迭代器) - 迭代器适配器、消费者

---

### 第三部分：进阶与实战

#### 第6章 - 模块与并发
- [模块系统](./模块与并发.md#模块系统) - mod、pub、路径、use
- [Cargo 包管理](./模块与并发.md#cargo-包管理) - Cargo.toml、工作空间
- [线程创建](./模块与并发.md#并发编程) - thread::spawn、闭包捕获
- [消息传递](./模块与并发.md#消息传递channel) - mpsc::channel
- [共享状态](./模块与并发.md#共享状态mutex) - Mutex、RwLock
- [Send/Sync](./模块与并发.md#send-与-sync-trait) - 线程安全 trait

#### 第7章 - 实战与优化
- [CLI 工具](./实战与优化.md#项目一命令行工具cli) - 文件搜索工具、clap 参数解析
- [Web 服务](./实战与优化.md#项目二简单-web-服务) - Axum、Actix-web
- [系统编程工具](./实战与优化.md#项目三系统编程工具) - 文件监控
- [性能分析工具](./实战与优化.md#性能分析工具) - cargo bench、Criterion
- [内存管理](./实战与优化.md#内存管理) - Box、Rc、Arc、RefCell
- [减少拷贝与堆分配](./实战与优化.md#减少拷贝与堆分配) - 预分配、栈分配

---

### 第四部分：深入与拓展

#### 第8章 - 不安全Rust
- [不安全 Rust](./不安全Rust.md#不安全-rust) - 五大能力、使用场景、安全原则
- [Rust 生态](./不安全Rust.md#rust-生态) - Web 开发、异步编程、数据库、CLI、网络库
- [进阶学习方向](./不安全Rust.md#进阶学习方向) - 异步编程、宏编程、编译期编程
- [官方学习资源](./不安全Rust.md#官方学习资源) - 官方文档、书籍、在线平台

---

## 🚀 学习路径建议

### 初学者路径（4-6周）
1. **入门** → 第1章 初识Rust + 第2章 基础语法
2. **核心** → 第3章 所有权与借用（最关键）
3. **进阶** → 第4章 结构体与枚举

### 进阶路径（4-8周）
1. **类型系统** → 第5章 泛型与特征
2. **工程化** → 第6章 模块与并发
3. **实践** → 第7章 实战与优化

### 专业路径（持续学习）
1. **性能优化** → 第7章 性能优化技巧
2. **底层能力** → 第8章 不安全Rust
3. **领域深耕** → Web/区块链/嵌入式/系统编程

---

## 📝 学习建议

> 💡 **新手提示**：Rust 的所有权系统是核心概念，建议多花时间理解"所有权"+"借用"+"生命周期"这套机制，这是 Rust 区别于其他语言的关键。

> 🔥 **进阶提示**：Trait 是 Rust 的灵魂，它不仅是接口，还支持默认实现、泛型约束、trait object 等高级用法，是写出优雅 Rust 代码的基础。

> ⚡ **专业提示**：unsafe Rust 打开了通往底层的大门，但一定要确保 unsafe 块内的操作是安全的。建议先熟悉 Rust 的安全编程风格，再逐步接触 unsafe。

---

## 🔗 相关资源

- [Rust 官方文档](https://doc.rust-lang.org/)
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust By Example](https://doc.rust-lang.org/rust-by-example/)
- [Rust Cookbook](https://rust-lang-nursery.github.io/rust-cookbook/)
- [Awesome Rust](https://github.com/rust-unofficial/awesome-rust)
- [Rustlings 小练习](https://github.com/rust-lang/rustlings/)
- [Exercism Rust Track](https://exercism.org/tracks/rust)

---

***持续学习中，学习笔记在不断完善和扩展中*** :grinning: