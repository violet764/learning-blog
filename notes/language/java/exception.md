# Java 异常处理

> 异常处理是 Java 程序健壮性的重要保障。通过合理的异常处理机制，可以使程序在遇到错误时优雅地恢复或终止，而不是直接崩溃。

## 异常

### 程序错误的类型

在编程中，错误可以分为三类：

```java
// 1. 编译时错误（语法错误）
// 编译器会直接报错，程序无法运行
int x =        // 缺少分号，语法错误
String s = 123; // 类型不匹配

// 2. 运行时错误（异常）
// 编译通过，运行时出错
int[] arr = new int[5];
arr[10] = 1;  // ArrayIndexOutOfBoundsException

String s = null;
s.length();   // NullPointerException

int result = 10 / 0;  // ArithmeticException

// 3. 逻辑错误
// 程序能运行，但结果不正确
int sum = 0;
for (int i = 1; i < 10; i++) {  // 应该是 i <= 10
    sum += i;
}
// sum = 45，但期望 55

// 异常处理就是为了处理"运行时错误"
```

### 异常处理的作用

```java
// 没有异常处理
public static void main(String[] args) {
    String input = "abc";
    int num = Integer.parseInt(input);  // 抛出 NumberFormatException
    System.out.println("这行不会执行");  // 程序直接崩溃
}

// 有异常处理
public static void main(String[] args) {
    String input = "abc";
    try {
        int num = Integer.parseInt(input);
        System.out.println("数字是：" + num);
    } catch (NumberFormatException e) {
        // 捕获异常，程序不会崩溃
        System.out.println("输入不是有效的数字");
        // 可以选择优雅地处理或提示用户
    }
    System.out.println("程序继续执行");  // 这行会执行
}
```

---

## 异常体系

### 异常层次结构

Java 中所有异常都继承自 `Throwable` 类：

```
Throwable（所有异常的父类）
│
├── Error（错误）
│   ├── OutOfMemoryError        → 内存溢出
│   ├── StackOverflowError      → 栈溢出（无限递归）
│   └── VirtualMachineError     → 虚拟机错误
│
└── Exception（异常）
    ├── IOException（IO 异常）
    │   ├── FileNotFoundException  → 文件未找到
    │   └── SocketException        → 网络异常
    ├── SQLException              → 数据库异常
    ├── ClassNotFoundException    → 类未找到
    │
    └── RuntimeException（运行时异常，非受检异常）
        ├── NullPointerException         → 空指针
        ├── ArrayIndexOutOfBoundsException → 数组越界
        ├── IllegalArgumentException    → 非法参数
        ├── NumberFormatException        → 数字格式错误
        ├── ClassCastException           → 类型转换错误
        └── ArithmeticException          → 算术错误（如除零）
```

### 异常分类详解

| 类型 | 说明 | 示例 | 编译时检查 | 处理要求 |
|------|------|------|:----------:|:--------:|
| **Error** | JVM 无法恢复的严重错误 | OutOfMemoryError | ✗ | 通常无法处理 |
| **受检异常** | 程序应该预见的异常 | IOException, SQLException | ✓ | 必须处理 |
| **非受检异常** | 编程错误导致的异常 | NullPointerException | ✗ | 可选处理 |

```java
// Error 示例：通常由 JVM 抛出，程序无法恢复
public void stackOverflow() {
    stackOverflow();  // 无限递归 → StackOverflowError
}

public void outOfMemory() {
    List<byte[]> list = new ArrayList<>();
    while (true) {
        list.add(new byte[1024 * 1024]);  // 不断分配内存 → OutOfMemoryError
    }
}

// 受检异常示例：必须处理，否则编译不通过
public void readFile(String path) {
    // FileReader 构造方法抛出 FileNotFoundException（受检异常）
    FileReader reader = new FileReader(path);  // 编译错误！未处理异常
}

// 正确做法：使用 try-catch 或 throws
public void readFile(String path) {
    try {
        FileReader reader = new FileReader(path);
    } catch (FileNotFoundException e) {
        e.printStackTrace();
    }
}

// 非受检异常示例：编译器不强制处理
public void process(String s) {
    // 可能抛出 NullPointerException，但编译器不强制处理
    System.out.println(s.length());
}

// 运行时调用：
process(null);  // 抛出 NullPointerException
```

### 受检异常 vs 非受检异常的选择

```java
// 什么时候用受检异常？
// 当调用者可以合理地恢复时，使用受检异常

// 示例：文件操作
// 调用者可以处理文件不存在的情况（比如提示用户选择其他文件）
public String readFile(String path) throws IOException {
    // ...
}

// 什么时候用非受检异常？
// 当是编程错误或调用者无法恢复时，使用非受检异常

// 示例：参数校验
// 调用者传入了 null，这是编程错误，应该修复代码而不是捕获异常
public void process(String s) {
    if (s == null) {
        throw new IllegalArgumentException("参数不能为 null");
    }
    // ...
}
```

---

## try-catch-finally

### 基本语法

```java
// try-catch-finally 的执行流程
try {
    // 1. 尝试执行的代码
    // 如果抛出异常，跳转到对应的 catch 块
} catch (异常类型 变量名) {
    // 2. 异常处理代码
    // 只有抛出匹配类型的异常才会执行这里
} finally {
    // 3. 无论是否异常都会执行
    // 通常用于资源清理
}

// 完整示例
public class ExceptionDemo {
    public static void main(String[] args) {
        try {
            // 可能抛出异常的代码
            int result = 10 / 0;  // 抛出 ArithmeticException
            System.out.println("这行不会执行");
        } catch (ArithmeticException e) {
            // 异常处理
            System.out.println("除零错误：" + e.getMessage());  // / by zero
            // 可以获取异常信息
            System.out.println("异常类型：" + e.getClass().getName());
            e.printStackTrace();  // 打印完整的堆栈跟踪
        } finally {
            // 无论是否异常都会执行
            System.out.println("清理资源");
        }
        System.out.println("程序继续执行");
    }
}

// 输出：
// 除零错误：/ by zero
// 异常类型：java.lang.ArithmeticException
// java.lang.ArithmeticException: / by zero
//     at ExceptionDemo.main(ExceptionDemo.java:4)
// 清理资源
// 程序继续执行
```

### 多异常捕获

```java
// 方式一：多个 catch 块
try {
    // 可能抛出多种异常的代码
    FileReader reader = new FileReader("file.txt");  // FileNotFoundException
    int data = reader.read();  // IOException
} catch (FileNotFoundException e) {
    System.out.println("文件未找到：" + e.getMessage());
} catch (IOException e) {
    System.out.println("IO 异常：" + e.getMessage());
}

// 方式二：Java 7+ 多异常合并捕获
try {
    FileReader reader = new FileReader("file.txt");
    int data = reader.read();
} catch (FileNotFoundException | IOException e) {
    // 注意：FileNotFoundException 是 IOException 的子类
    // 所以这里只会捕获 IOException（父类）
    System.out.println("异常：" + e.getMessage());
}

// 正确的多异常合并（没有继承关系的异常）
try {
    // ...
} catch (NullPointerException | ArrayIndexOutOfBoundsException e) {
    System.out.println("异常：" + e.getMessage());
}

// catch 块的顺序：子类异常必须写在父类异常前面
try {
    // ...
} catch (FileNotFoundException e) {  // 子类，先捕获
    System.out.println("文件未找到");
} catch (IOException e) {  // 父类，后捕获
    System.out.println("IO 异常");
}
// 如果顺序反了，编译错误！父类会先捕获子类异常
```

### finally 执行时机

```java
// finally 总是会执行（除非 System.exit()）

// 情况一：正常执行
public int method1() {
    try {
        return 1;
    } finally {
        System.out.println("finally 执行");  // 会执行
    }
}
// 输出：finally 执行
// 返回：1

// 情况二：异常执行
public int method2() {
    try {
        throw new RuntimeException();
    } finally {
        System.out.println("finally 执行");  // 会执行
    }
}
// 输出：finally 执行
// 抛出：RuntimeException

// 情况三：try 和 finally 都有 return
public int method3() {
    try {
        return 1;
    } finally {
        return 2;  // ⚠️ finally 的 return 会覆盖 try 的 return
    }
}
// 返回：2（finally 的返回值）

// ⚠️ 不要在 finally 中使用 return！
// 这会吞掉 try 块中的异常

// 情况四：System.exit() 不会执行 finally
public void method4() {
    try {
        System.exit(0);  // 终止 JVM
    } finally {
        System.out.println("不会执行");
    }
}
```

---

## try-with-resources

📌 Java 7 引入的语法，自动关闭实现了 `AutoCloseable` 的资源。

### 传统方式 vs try-with-resources

```java
// ========== 传统方式：繁琐且容易遗漏关闭 ==========
public void readFileTraditional(String path) {
    BufferedReader reader = null;
    try {
        reader = new BufferedReader(new FileReader(path));
        String line = reader.readLine();
        // 处理数据
    } catch (IOException e) {
        e.printStackTrace();
    } finally {
        // 关闭资源
        if (reader != null) {
            try {
                reader.close();  // close 也可能抛出异常
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}

// ========== try-with-resources：简洁且自动关闭 ==========
public void readFileModern(String path) {
    // 在 try() 中声明的资源会自动关闭
    // 要求：资源必须实现 AutoCloseable 接口
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
        String line = reader.readLine();
        // 处理数据
    } catch (IOException e) {
        e.printStackTrace();
    }
    // 无需 finally，reader 会自动关闭
}

// ========== 多个资源 ==========
public void copyFile(String src, String dest) {
    try (FileInputStream fis = new FileInputStream(src);
         FileOutputStream fos = new FileOutputStream(dest)) {
        byte[] buffer = new byte[1024];
        int len;
        while ((len = fis.read(buffer)) != -1) {
            fos.write(buffer, 0, len);
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
    // fis 和 fos 都会自动关闭
    // 关闭顺序：后声明的先关闭（fos 先关闭，fis 后关闭）
}

// ========== 自定义可关闭资源 ==========
public class MyResource implements AutoCloseable {
    private String name;
    
    public MyResource(String name) {
        this.name = name;
        System.out.println(name + " 创建");
    }
    
    public void doSomething() {
        System.out.println(name + " 使用中");
    }
    
    @Override
    public void close() {
        System.out.println(name + " 关闭");
    }
}

// 使用
try (MyResource r1 = new MyResource("资源1");
     MyResource r2 = new MyResource("资源2")) {
    r1.doSomething();
    r2.doSomething();
}
// 输出：
// 资源1 创建
// 资源2 创建
// 资源1 使用中
// 资源2 使用中
// 资源2 关闭  （后创建的先关闭）
// 资源1 关闭
```

---

## throw 与 throws

### throw：抛出异常

`throw` 用于在代码中主动抛出一个异常对象。

```java
// 基本语法
throw new 异常类型("异常信息");

// 示例：参数校验
public void setAge(int age) {
    if (age < 0) {
        // 主动抛出异常
        throw new IllegalArgumentException("年龄不能为负数");
    }
    if (age > 150) {
        throw new IllegalArgumentException("年龄不能超过150岁");
    }
    this.age = age;
}

// 调用
try {
    person.setAge(-5);
} catch (IllegalArgumentException e) {
    System.out.println(e.getMessage());  // 年龄不能为负数
}

// 示例：业务逻辑异常
public void withdraw(double amount) {
    if (amount > balance) {
        throw new RuntimeException("余额不足");
    }
    balance -= amount;
}

// throw 可以抛出任何 Throwable 的子类
public void method() {
    // throw new Error("严重错误");     // 可以抛出 Error
    // throw new Exception("受检异常"); // 可以抛出 Exception
    throw new RuntimeException("运行时异常");  // 可以抛出 RuntimeException
}
```

### throws：声明异常

`throws` 用于在方法签名中声明该方法可能抛出的异常。

```java
// 基本语法
public 返回类型 方法名(参数) throws 异常类型1, 异常类型2 {
    // 方法体
}

// 示例：声明受检异常
// FileReader 构造方法声明了 throws FileNotFoundException
// 我们的方法如果不处理，必须继续声明 throws
public String readFile(String path) throws IOException {
    FileReader reader = new FileReader(path);  // 可能抛出 FileNotFoundException
    // ...
    return content;
}

// 调用者必须处理或继续声明
public void caller1() {
    try {
        readFile("test.txt");  // 处理异常
    } catch (IOException e) {
        e.printStackTrace();
    }
}

public void caller2() throws IOException {
    readFile("test.txt");  // 继续声明，让更上层处理
}

// 多个异常
public void multiException() throws IOException, SQLException, ClassNotFoundException {
    // 可能抛出多种受检异常的代码
}

// Java 7+ 可以简化多个异常声明
public void multiException() throws IOException, SQLException {
    // ...
}
```

### throw 与 throws 的区别

| 关键字 | 作用 | 位置 | 说明 |
|--------|------|------|------|
| `throw` | 抛出一个异常对象 | 方法体内 | 实际抛出异常 |
| `throws` | 声明方法可能抛出的异常 | 方法签名上 | 告知调用者需要处理 |

```java
// 对比示例
// throws：声明异常
public void method1() throws IllegalArgumentException {
    // throw：抛出异常
    throw new IllegalArgumentException("参数错误");
}

// 另一个例子
public String readFile(String path) throws IOException {  // throws 声明
    if (path == null) {
        throw new IllegalArgumentException("路径不能为空");  // throw 抛出
    }
    FileReader reader = new FileReader(path);  // 可能抛出 FileNotFoundException
    // ...
    return content;
}
```

---

## 自定义异常

### 创建自定义异常

```java
// ========== 自定义受检异常 ==========
// 继承 Exception
public class BusinessException extends Exception {
    private int errorCode;  // 错误码
    
    // 构造方法
    public BusinessException(String message) {
        super(message);
    }
    
    public BusinessException(int errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public BusinessException(String message, Throwable cause) {
        super(message, cause);  // 保留原始异常
    }
    
    public int getErrorCode() {
        return errorCode;
    }
}

// ========== 自定义非受检异常 ==========
// 继承 RuntimeException
public class ValidationException extends RuntimeException {
    public ValidationException(String message) {
        super(message);
    }
    
    public ValidationException(String message, Throwable cause) {
        super(message, cause);
    }
}

// ========== 使用自定义异常 ==========
public class AccountService {
    private double balance;
    
    // 受检异常：调用者必须处理
    public void withdraw(double amount) throws BusinessException {
        if (amount < 0) {
            throw new BusinessException(1001, "取款金额不能为负数");
        }
        if (amount > balance) {
            throw new BusinessException(1002, "余额不足");
        }
        balance -= amount;
    }
    
    // 非受检异常：调用者可选处理
    public void deposit(double amount) {
        if (amount <= 0) {
            throw new ValidationException("存款金额必须大于0");
        }
        balance += amount;
    }
}

// 调用
AccountService service = new AccountService();
try {
    service.withdraw(1000);
} catch (BusinessException e) {
    System.out.println("错误码：" + e.getErrorCode());
    System.out.println("错误信息：" + e.getMessage());
}
```

---

## 异常链

当捕获一个异常后抛出另一个异常时，应该保留原始异常信息。

```java
// 异常链：保留原始异常
public void method() throws BusinessException {
    try {
        // 数据库操作
        Connection conn = DriverManager.getConnection("...");
    } catch (SQLException e) {
        // 捕获 SQLException，抛出业务异常
        // 将原始异常作为 cause 传入
        throw new BusinessException("数据库操作失败", e);
    }
}

// 调用
try {
    service.method();
} catch (BusinessException e) {
    System.out.println("业务异常：" + e.getMessage());
    
    // 获取原始异常
    Throwable cause = e.getCause();
    if (cause != null) {
        System.out.println("原始异常：" + cause.getMessage());
    }
    
    // 打印完整的异常链
    e.printStackTrace();
}

// 异常链的好处：
// 1. 保留完整的错误信息，便于调试
// 2. 对调用者隐藏底层实现细节
// 3. 可以追踪问题的根源
```

---

## 常见异常及处理

### NullPointerException（空指针异常）

```java
// 最常见的异常！
String str = null;
str.length();  // NullPointerException

// 预防措施一：判空
if (str != null) {
    str.length();
}

// 预防措施二：使用 Optional（Java 8+）
Optional.ofNullable(str)
    .ifPresent(s -> System.out.println(s.length()));

// 预防措施三：使用 Objects.requireNonNull
public void setName(String name) {
    this.name = Objects.requireNonNull(name, "name 不能为 null");
}

// 预防措施四：返回空集合而不是 null
public List<User> getUsers() {
    List<User> users = userDao.findAll();
    return users != null ? users : Collections.emptyList();
}
```

### ArrayIndexOutOfBoundsException（数组越界）

```java
int[] arr = new int[5];
arr[5] = 10;  // ArrayIndexOutOfBoundsException

// 预防措施：检查索引范围
if (index >= 0 && index < arr.length) {
    arr[index] = 10;
}

// 或使用 ArrayList（会自动检查）
List<Integer> list = new ArrayList<>();
// list.get(0);  // IndexOutOfBoundsException（更详细的错误信息）
```

### NumberFormatException（数字格式异常）

```java
Integer.parseInt("abc");  // NumberFormatException

// 预防措施：使用正则校验或 try-catch
String input = "123abc";

// 方式一：正则校验
if (input.matches("\\d+")) {
    int num = Integer.parseInt(input);
}

// 方式二：try-catch
try {
    int num = Integer.parseInt(input);
} catch (NumberFormatException e) {
    System.out.println("无效的数字格式");
}

// 方式三：使用 Apache Commons Lang
// NumberUtils.isCreatable(input)
```

### ClassCastException（类型转换异常）

```java
Object obj = "Hello";
Integer num = (Integer) obj;  // ClassCastException

// 预防措施：使用 instanceof 检查
if (obj instanceof Integer) {
    Integer num = (Integer) obj;
}

// Java 16+ 模式匹配
if (obj instanceof Integer num) {
    // 直接使用 num
}
```

---

## 异常处理最佳实践

### ✅ 推荐做法

```java
// 1. 捕获具体异常，不要捕获 Exception
try {
    // ...
} catch (FileNotFoundException e) {
    // 处理文件未找到
    log.error("文件未找到", e);
} catch (IOException e) {
    // 处理其他 IO 异常
    log.error("IO 异常", e);
}

// 2. 不要吞掉异常，要记录或重新抛出
try {
    // ...
} catch (Exception e) {
    log.error("操作失败", e);  // 记录日志
    throw new BusinessException("操作失败", e);  // 包装后抛出
}

// 3. 使用 try-with-resources
try (Connection conn = dataSource.getConnection()) {
    // ...
}  // 自动关闭

// 4. 尽早失败（Fail Fast）
public void setAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("年龄不能为负数");
    }
    this.age = age;
}

// 5. 使用有意义的异常消息
throw new IllegalArgumentException("用户ID不能为空");
// 而不是
throw new IllegalArgumentException("参数错误");

// 6. 对于可恢复的情况使用受检异常
public void readFile(String path) throws FileNotFoundException {
    // 调用者可以处理：提示用户选择其他文件
}

// 7. 对于编程错误使用非受检异常
public void process(String s) {
    if (s == null) {
        throw new IllegalArgumentException("参数不能为 null");
    }
}
```

### ❌ 避免的做法

```java
// 1. 不要捕获 Exception 太宽泛
try {
    // ...
} catch (Exception e) {  // 不推荐：会捕获所有异常，包括 RuntimeException
    e.printStackTrace();
}

// 2. 不要使用空的 catch 块
try {
    // ...
} catch (Exception e) {
    // 吞掉异常，不做任何处理，隐藏了问题
}

// 3. 不要用异常控制程序流程
try {
    Integer.parseInt(input);
} catch (NumberFormatException e) {
    return 0;  // 不推荐，应该先检查格式
}

// 推荐做法：
if (input.matches("\\d+")) {
    return Integer.parseInt(input);
} else {
    return 0;
}

// 4. 不要在 finally 中使用 return
public int method() {
    try {
        return 1;
    } finally {
        return 2;  // 会覆盖 try 的返回值
    }
}

// 5. 不要抛出过于宽泛的异常
public void method() throws Exception {  // 不推荐
}

// 推荐：
public void method() throws IOException, SQLException {  // 具体的异常
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **Error** | JVM 严重错误，无法恢复 |
| **受检异常** | 编译时必须处理，通常是外部因素 |
| **非受检异常** | 运行时异常，编程错误导致 |
| **try-catch** | 捕获并处理异常 |
| **finally** | 无论是否异常都会执行 |
| **try-with-resources** | 自动关闭资源（推荐） |
| **throw** | 抛出异常对象 |
| **throws** | 声明方法可能抛出的异常 |
| **异常链** | 保留原始异常信息 |

### 异常处理的核心原则

1. **只捕获你能处理的异常**
2. **不要吞掉异常**
3. **使用具体异常而不是 Exception**
4. **提供有意义的异常消息**
5. **使用 try-with-resources 管理资源**
6. **对可恢复错误使用受检异常，对编程错误使用非受检异常**