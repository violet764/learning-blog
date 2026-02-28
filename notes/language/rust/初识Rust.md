# Rust 初识

> Rust 是一门系统级编程语言，以内存安全为核心设计目标，无需垃圾回收器即可实现安全编程。

## 设计理念与适用场景

### 核心设计理念

Rust 的设计理念围绕三个关键目标展开：

**1. 内存安全（Memory Safety）**

Rust 通过所有权系统（Ownership）和借用检查器（Borrow Checker）在编译期确保内存安全，完全避免了：
- 空指针引用
- 悬垂指针
- 缓冲区溢出
- 数据竞争（Data Race）

这使得 Rust 能够在不牺牲性能的前提下，实现与 C/C++ 相当的执行效率，同时提供内存安全保证。

**2. 零成本抽象（Zero-Cost Abstractions）**

Rust 允许使用高级抽象而不带来运行时开销：
- 泛型在编译期单态化
- 迭代器使用循环展开优化
- 闭包通过内联优化

**3. 并发安全（Concurrency Safety）**

Rust 的所有权系统天然防止数据竞争：
- 借用规则确保同一时刻只有一个可变引用
- `Send` 和 `Sync` trait 标记线程安全类型

### 适用场景

| 领域 | 应用案例 | Rust 优势 |
|------|----------|-----------|
| 系统编程 | 操作系统、嵌入式、驱动程序 | 底层控制、内存安全 |
| Web 开发 | WebAssembly、API 服务 | 高性能、异步 Runtime |
| 命令行工具 | CLI 应用、DevOps 工具 | 跨平台、零依赖部署 |
| 区块链 | 智能合约、共识算法 | 确定性执行、内存安全 |
| 网络服务 | 高性能服务器、分布式系统 | 异步 IO、并发安全 |

---

## 环境安装

### 安装 rustup

Rust 的官方安装工具是 `rustup`，它管理 Rust 工具链（编译器、标准库、Cargo）。

**Windows 安装：**

```powershell
# 方式一：使用 winget（推荐）
winget install Rustlang.Rust.MSVC

# 方式二：直接下载 rustup-init.exe
# 访问 https://win.rustup.rs/ 下载并运行
```

**macOS / Linux 安装：**

```bash
# 运行安装脚本
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 或使用 Homebrew（macOS）
brew install rustup-init && rustup-init
```

**验证安装：**

```bash
# 查看版本
rustc --version
cargo --version

# 输出示例：
# rustc 1.75.0 (82e1608df 2023-12-21)
# cargo 1.75.0 (126a90f49 2023-12-21)
```

### 工具链管理

```bash
# 查看已安装的工具链
rustup show

# 安装特定版本
rustup install 1.70.0

# 切换默认工具链
rustup default 1.75.0

# 更新到最新稳定版
rustup update

# 添加目标平台（如嵌入式开发）
rustup target add thumbv7em-none-eabihf
```

### IDE 配置

**VS Code（推荐）：**

1. 安装扩展：`rust-analyzer`（代码补全、跳转）
2. 安装扩展：`CodeLLDB`（调试器）
3. 安装扩展：`rust-syntax`（语法高亮）

**JetBrains 系列：**

- CLion + Rust 插件
- IntelliJ IDEA + Rust 插件

---

## 第一个程序

### Hello World

**创建项目：**

```bash
cargo new hello_rust
cd hello_rust
```

**项目结构：**

```
hello_rust/
├── src/
│   └── main.rs      # 入口文件
├── Cargo.toml       # 项目配置
└── target/          # 编译输出（自动生成）
```

**编写代码：**

```rust
// src/main.rs
fn main() {
    println!("Hello, world!");
}
```

**运行程序：**

```bash
# 开发模式运行（快速编译）
cargo run

# 或直接编译
cargo build
./target/debug/hello_rust   # Linux/macOS
.\target\debug\hello_rust.exe  # Windows
```

### Cargo 基础命令

Cargo 是 Rust 的包管理器和构建工具。

```bash
# ========== 项目管理 ==========
cargo new <project_name>    # 创建新项目（binary）
cargo new --lib <name>      # 创建库项目

# ========== 构建与运行 ==========
cargo build                 # 编译项目（debug 模式）
cargo build --release      # 编译发布版本（优化）
cargo run                  # 运行程序
cargo run --release        # 运行发布版本

# ========== 代码检查 ==========
cargo check                # 快速类型检查（不生成二进制）
cargo clippy               # 代码风格与潜在问题检查
cargo fmt                 # 代码格式化

# ========== 测试 ==========
cargo test                 # 运行测试
cargo test --release       # 发布模式下测试

# ========== 依赖管理 ==========
cargo add <crate>          # 添加依赖
cargo update               # 更新依赖版本
cargo tree                 # 查看依赖树
cargo outdated            # 检查过时依赖

# ========== 文档 ==========
cargo doc                 # 生成文档
cargo doc --open          # 生成并打开文档
```

### Cargo.toml 详解

```toml
[package]
name = "my_project"       # 项目名称
version = "0.1.0"         # 版本号
edition = "2021"          # Rust 版本
authors = ["Your Name"]

[dependencies]
# 外部依赖
serde = "1.0"            # 带版本约束
serde = { version = "1.0", features = ["derive"] }  # 带特性

[dev-dependencies]
# 开发依赖（测试用）

[build-dependencies]
# 构建脚本依赖

[profile.release]
opt-level = 3            # 优化级别
lto = true               # 链接时优化
codegen-units = 1        # 减少并行单元以优化
```

---

## 必备知识：注释与格式化

### 注释

```rust
// 单行注释

/*
 * 多行
 * 注释
 */

/// 文档注释 - 会生成 API 文档
/// # Examples
/// ```
/// let x = 5;
/// ```

//! 内部文档注释 - 用于 crate/模块级文档
```

### 代码格式化

Rust 强制代码风格，使用 `rustfmt` 统一格式：

```bash
# 格式化项目
cargo fmt

# 检查格式是否符合标准
cargo fmt -- --check
```

**常用格式规则：**
- 4 空格缩进
- 逗号后保留空格
- 命名使用蛇形命名（snake_case）：函数、变量
- 命名使用帕斯卡命名（PascalCase）：类型、结构体、枚举
- 命名使用全大写（SCREAMING_SNAKE_CASE）：常量

---

## 小结

本章节我们了解了：

1. **Rust 的核心设计理念**：内存安全、零成本抽象、并发安全
2. **适用场景**：系统编程、Web 开发、CLI 工具、区块链等
3. **环境安装**：使用 rustup 管理工具链
4. **第一个程序**：使用 Cargo 创建和运行项目
5. **Cargo 基础命令**：build、run、test、add 等常用命令

下一章节我们将学习 Rust 的基础语法与类型系统。
