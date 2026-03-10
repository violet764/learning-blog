# Java 异常处理

> 异常处理是 Java 程序健壮性的重要保障。通过合理的异常处理机制，可以使程序在遇到错误时优雅地恢复或终止，而不是直接崩溃。

## 异常体系

### 异常层次结构

```
Throwable
├── Error（错误）
│   ├── OutOfMemoryError
│   ├── StackOverflowError
│   └── VirtualMachineError
│
└── Exception（异常）
    ├── IOException
    │   ├── FileNotFoundException
    │   └── SocketException
    ├── SQLException
    ├── RuntimeException（非受检异常）
    │   ├── NullPointerException
    │   ├── ArrayIndexOutOfBoundsException
    │   ├── IllegalArgumentException
    │   ├── NumberFormatException
    │   └── ClassCastException
    └── ...其他受检异常
```

### 异常分类

| 类型 | 说明 | 示例 | 编译时检查 |
|------|------|------|:----------:|
| **Error** | JVM 无法恢复的严重错误 | OutOfMemoryError | ✗ |
| **受检异常** | 必须处理的异常 | IOException, SQLException | ✓ |
| **非受检异常** | 运行时异常，可选处理 | NullPointerException | ✗ |

---

## try-catch-finally

### 基本语法

```java
try {
    // 可能抛出异常的代码
    int result = 10 / 0;
} catch (ArithmeticException e) {
    // 异常处理
    System.out.println("除零错误：" + e.getMessage());
} finally {
    // 无论是否异常都会执行（用于资源清理）
    System.out.println("清理资源");
}
```

### 多异常捕获

```java
try {
    // ...
} catch (IOException e) {
    System.out.println("IO异常");
} catch (SQLException e) {
    System.out.println("SQL异常");
}

// Java 7+ 多异常合并捕获
try {
    // ...
} catch (IOException | SQLException e) {
    System.out.println("异常：" + e.getMessage());
}
```

### finally 执行时机

```java
public int test() {
    try {
        return 1;
    } finally {
        // finally 在 return 之前执行
        System.out.println("finally");
    }
}
// 输出：finally
// 返回：1
```

---

## try-with-resources

📌 Java 7 引入的语法，自动关闭实现了 `AutoCloseable` 的资源。

```java
// 传统方式
BufferedReader reader = null;
try {
    reader = new BufferedReader(new FileReader("file.txt"));
    String line = reader.readLine();
} catch (IOException e) {
    e.printStackTrace();
} finally {
    if (reader != null) {
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// try-with-resources（推荐）
try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
    String line = reader.readLine();
} catch (IOException e) {
    e.printStackTrace();
}

// 多个资源
try (FileInputStream fis = new FileInputStream("in.txt");
     FileOutputStream fos = new FileOutputStream("out.txt")) {
    // ...
}
```

---

## throw 与 throws

### throw：抛出异常

```java
public void setAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("年龄不能为负数");
    }
    this.age = age;
}
```

### throws：声明异常

```java
// 方法签名中声明可能抛出的受检异常
public void readFile(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(path));
    // ...
}
```

### 区别

| 关键字 | 作用 | 位置 |
|--------|------|------|
| `throw` | 抛出一个异常对象 | 方法体内 |
| `throws` | 声明方法可能抛出的异常 | 方法签名上 |

---

## 自定义异常

```java
// 自定义受检异常
public class BusinessException extends Exception {
    private int errorCode;
    
    public BusinessException(String message) {
        super(message);
    }
    
    public BusinessException(int errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public int getErrorCode() {
        return errorCode;
    }
}

// 自定义非受检异常
public class ValidationException extends RuntimeException {
    public ValidationException(String message) {
        super(message);
    }
}

// 使用
public void withdraw(double amount) throws BusinessException {
    if (amount > balance) {
        throw new BusinessException(1001, "余额不足");
    }
    balance -= amount;
}
```

---

## 异常链

```java
public void method() throws BusinessException {
    try {
        // 一些可能抛出 SQLException 的操作
    } catch (SQLException e) {
        // 保留原始异常信息
        throw new BusinessException("数据库操作失败", e);
    }
}

// 获取原始异常
try {
    method();
} catch (BusinessException e) {
    Throwable cause = e.getCause();  // 获取原始异常
    e.printStackTrace();  // 打印完整的异常链
}
```

---

## 常见异常及处理

### NullPointerException

```java
String str = null;
// str.length();  // NullPointerException

// 预防措施
if (str != null) {
    str.length();
}

// 使用 Optional（Java 8+）
Optional.ofNullable(str).ifPresent(s -> s.length());
```

### ArrayIndexOutOfBoundsException

```java
int[] arr = new int[5];
// arr[5] = 10;  // ArrayIndexOutOfBoundsException

// 预防措施
if (index >= 0 && index < arr.length) {
    arr[index] = 10;
}
```

### NumberFormatException

```java
// Integer.parseInt("abc");  // NumberFormatException

// 预防措施
try {
    int num = Integer.parseInt(input);
} catch (NumberFormatException e) {
    System.out.println("无效的数字格式");
}
```

---

## 异常处理最佳实践

### ✅ 推荐

```java
// 1. 捕获具体异常
try {
    // ...
} catch (FileNotFoundException e) {
    // 处理文件未找到
} catch (IOException e) {
    // 处理其他IO异常
}

// 2. 不要吞掉异常
try {
    // ...
} catch (Exception e) {
    log.error("操作失败", e);  // 记录日志
    throw e;  // 重新抛出或包装后抛出
}

// 3. 使用 try-with-resources
try (Connection conn = dataSource.getConnection()) {
    // ...
}

// 4. 尽早失败
public void setAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("年龄不能为负数");
    }
    this.age = age;
}
```

### ❌ 避免

```java
// 1. 捕获 Exception 太宽泛
try {
    // ...
} catch (Exception e) {  // 不推荐
    e.printStackTrace();
}

// 2. 空的 catch 块
try {
    // ...
} catch (Exception e) {
    // 吞掉异常，不做任何处理
}

// 3. 用异常控制流程
try {
    Integer.parseInt(input);
} catch (NumberFormatException e) {
    return 0;  // 不推荐，应先检查格式
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **Error** | JVM 严重错误，无法恢复 |
| **受检异常** | 编译时必须处理 |
| **非受检异常** | 运行时异常，可选处理 |
| **try-catch** | 捕获并处理异常 |
| **finally** | 无论是否异常都会执行 |
| **try-with-resources** | 自动关闭资源 |
| **throw** | 抛出异常对象 |
| **throws** | 声明方法可能抛出的异常 |
