# Java 编程语言

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/zh/thumb/8/88/Java_logo.svg/120px-Java_logo.svg.png" alt="Java Logo" width="120">
</div>

> Java 是一门面向对象的编程语言，由 Sun Microsystems（现为 Oracle）于 1995 年发布。其核心设计理念是"一次编写，到处运行"（Write Once, Run Anywhere），这得益于 Java 虚拟机（JVM）的跨平台特性。

## 📖 为什么学习 Java

### 核心特点

| 特点 | 说明 |
|------|------|
| **跨平台性** | 通过 JVM 实现一次编写，到处运行 |
| **面向对象** | 完善的 OOP 体系：封装、继承、多态 |
| **强类型** | 编译时类型检查，减少运行时错误 |
| **自动内存管理** | 垃圾回收机制（GC）自动管理内存 |
| **丰富的生态** | 庞大的标准库和第三方框架 |
| **多线程支持** | 内置多线程编程能力 |

### 应用领域

- **企业级应用**：Spring 生态，微服务架构
- **Android 开发**：移动应用开发主力语言
- **大数据处理**：Hadoop、Spark、Flink 等框架
- **后端服务**：Web 服务、API 开发

---

## 📚 学习路径

```
基础语法 → 面向对象 → 集合框架 → 异常处理
    ↓
泛型 → Lambda & Stream → 并发编程 → IO/NIO
```

### 推荐学习顺序

1. **[基础语法](./basics.md)** - 变量、数据类型、运算符、控制流
2. **[面向对象](./oop.md)** - 类、继承、多态、接口、抽象类
3. **[集合框架](./collections.md)** - List、Set、Map、Queue
4. **[泛型](./generics.md)** - 泛型类、泛型方法、类型擦除
5. **[异常处理](./exception.md)** - 异常层次、try-catch、自定义异常
6. **[Lambda与Stream](./lambda-stream.md)** - 函数式编程、流式处理

---

## 🔧 开发环境搭建

### JDK 安装

1. 下载 JDK：[Oracle JDK](https://www.oracle.com/java/technologies/downloads/) 或 [OpenJDK](https://adoptium.net/)
2. 配置环境变量：
   - `JAVA_HOME`：JDK 安装目录
   - `PATH`：添加 `%JAVA_HOME%\bin`

### 验证安装

```bash
java -version
javac -version
```

### IDE 推荐

| IDE | 特点 |
|-----|------|
| **IntelliJ IDEA** | 功能强大，智能提示，推荐使用 |
| **Eclipse** | 免费开源，插件丰富 |
| **VS Code** | 轻量级，需安装 Java 扩展包 |

---

## 🚀 Hello World

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

### 编译与运行

```bash
javac HelloWorld.java    # 编译
java HelloWorld          # 运行
```

---

## 📂 文档目录

### 语言基础

| 文档 | 内容 | 难度 |
|------|------|:----:|
| [基础语法](./basics.md) | 变量、数据类型、运算符、控制流 | ⭐ |
| [面向对象](./oop.md) | 类、继承、多态、接口、抽象类 | ⭐⭐ |
| [集合框架](./collections.md) | List、Set、Map、Queue 详解 | ⭐⭐ |
| [泛型](./generics.md) | 泛型类、泛型方法、通配符 | ⭐⭐ |
| [异常处理](./exception.md) | 异常体系、try-catch、自定义异常 | ⭐ |
| [Lambda与Stream](./lambda-stream.md) | 函数式接口、流式处理、Optional | ⭐⭐ |

### 开发框架

| 文档 | 内容 | 难度 |
|------|------|:----:|
| [Spring 核心](./spring.md) | IoC 容器、依赖注入、AOP、Bean 生命周期 | ⭐⭐⭐ |
| [Spring Boot](./springboot.md) | 自动配置、起步依赖、Actuator 监控 | ⭐⭐ |
| [Spring MVC](./springmvc.md) | RESTful API、控制器、请求处理、拦截器 | ⭐⭐ |
| [MyBatis](./mybatis.md) | SQL 映射、动态 SQL、缓存机制 | ⭐⭐ |
| [Spring Security](./spring-security.md) | 认证授权、JWT、安全配置 | ⭐⭐⭐ |

---

## 🔗 学习资源

- [Java 官方文档](https://docs.oracle.com/javase/)
- [Java API 文档](https://docs.oracle.com/en/java/javase/)
- [Java Tutorial](https://docs.oracle.com/javase/tutorial/)
- [LeetCode Java 刷题](https://leetcode.cn/)

---

*持续更新中...* 🚀
