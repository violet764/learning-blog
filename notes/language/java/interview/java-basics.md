# Java 基础面试题

> 本文档整理 Java 基础相关的高频面试题，包含语法、面向对象、集合等内容。

---

## 一、数据类型

### 1. Java 有哪些基本数据类型？各占多少字节？

**答案：**

| 类型 | 关键字 | 字节 | 位数 | 取值范围 | 默认值 |
|------|--------|:----:|:----:|----------|--------|
| 字节型 | byte | 1 | 8 | -128 ~ 127 | 0 |
| 短整型 | short | 2 | 16 | -32768 ~ 32767 | 0 |
| 整型 | int | 4 | 32 | -2³¹ ~ 2³¹-1 | 0 |
| 长整型 | long | 8 | 64 | -2⁶³ ~ 2⁶³-1 | 0L |
| 单精度浮点 | float | 4 | 32 | ±3.4E38 | 0.0f |
| 双精度浮点 | double | 8 | 64 | ±1.7E308 | 0.0d |
| 字符型 | char | 2 | 16 | 0 ~ 65535 | '\u0000' |
| 布尔型 | boolean | - | 1 | true/false | false |

**追问：为什么 byte 范围是 -128 到 127？**

> byte 用 8 位二进制表示，最高位是符号位（0 正 1 负）。
> - 正数范围：00000000 ~ 01111111（0 ~ 127）
> - 负数用补码表示：10000000 ~ 11111111
> - 10000000 表示 -128（补码的特殊值，没有对应的原码）
> - 所以范围是 -128 ~ 127，共 256 个值

**追问：boolean 占多少字节？**

> JVM 规范没有明确规定。实际上：
> - 单个 boolean 变量：JVM 编译后用 int 表示，占 4 字节
> - boolean 数组：每个元素占 1 字节

---

### 2. 自动类型转换和强制类型转换？

**答案：**

**自动类型转换（隐式）：** 小类型 → 大类型

```java
byte → short → int → long → float → double
              ↗
        char

// 示例
int a = 100;
long b = a;      // 自动转换
double c = b;    // 自动转换
```

**强制类型转换（显式）：** 大类型 → 小类型

```java
double d = 3.14;
int i = (int) d;    // 强制转换，精度丢失
// i = 3

long l = 1000L;
int j = (int) l;    // 可能溢出

// 溢出示例
int big = 128;
byte small = (byte) big;  // small = -128
```

**追问：为什么 long 可以自动转 float？**

> float 虽然是 4 字节，但采用 IEEE 754 浮点数表示法，能表示更大的范围。
> - long 最大值：9,223,372,036,854,775,807
> - float 最大值：约 3.4 × 10³⁸
> 
> 但会损失精度，因为 float 只有 23 位有效数字。

---

### 3. int 和 Integer 的区别？

**答案：**

| 特性 | int | Integer |
|------|-----|---------|
| 类型 | 基本类型 | 包装类型 |
| 默认值 | 0 | null |
| 存储 | 栈 | 堆 |
| 泛型 | 不支持 | 支持 |
| 比较 | == 比值 | == 比地址 |

```java
int a = 100;
Integer b = 100;     // 自动装箱
Integer c = new Integer(100);

a == b;              // true，自动拆箱比较值
b == c;              // false，比较对象地址
b.equals(c);         // true，比较值

// 享元模式缓存
Integer d = 127;
Integer e = 127;
d == e;              // true，缓存范围内

Integer f = 128;
Integer g = 128;
f == g;              // false，超出缓存范围
```

**追问：Integer 缓存范围？**

> -128 ~ 127（可通过 -XX:AutoBoxCacheMax 调整）
> 
> ```java
> public static Integer valueOf(int i) {
>     if (i >= IntegerCache.low && i <= IntegerCache.high)
>         return IntegerCache.cache[i + (-IntegerCache.low)];
>     return new Integer(i);
> }
> ```

---

## 二、字符串

### 4. String 为什么是不可变的？

**答案：**

```java
public final class String implements java.io.Serializable, Comparable<String>, CharSequence {
    private final char[] value;  // JDK 8
    // private final byte[] value;  // JDK 9+
    
    // 没有 setter 方法，无法修改
}
```

**原因：**

1. **安全性**：String 常用于存储敏感信息（如数据库连接字符串、URL）
2. **哈希缓存**：hashCode 可以缓存，提高 HashMap 性能
3. **字符串常量池**：多个引用可指向同一字符串，节省内存
4. **线程安全**：不可变对象天然线程安全

**追问：String 真的不可变吗？**

> 通过反射可以修改：
> ```java
> String s = "Hello";
> Field valueField = String.class.getDeclaredField("value");
> valueField.setAccessible(true);
> char[] value = (char[]) valueField.get(s);
> value[0] = 'h';  // s 变成 "hello"
> ```
> 但这是 hack 手段，实际开发中不应该这样做。

---

### 5. String、StringBuilder、StringBuffer 区别？

**答案：**

| 特性 | String | StringBuilder | StringBuffer |
|------|--------|---------------|--------------|
| 可变性 | 不可变 | 可变 | 可变 |
| 线程安全 | 安全（不可变） | 不安全 | 安全（synchronized） |
| 性能 | 低 | 高 | 中 |
| 适用场景 | 少量操作 | 单线程大量操作 | 多线程大量操作 |

```java
// String：每次操作都创建新对象
String s = "Hello";
s = s + " World";  // 创建了新对象

// StringBuilder：在原对象上修改
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");  // 同一个对象
```

**追问：StringBuilder 为什么快？**

> String 每次 + 操作：
> 1. 创建 StringBuilder 对象
> 2. append 原字符串
> 3. append 新字符串
> 4. toString 创建新 String
> 
> 直接用 StringBuilder 省去了多次创建对象的开销。

---

### 6. String s = new String("abc") 创建了几个对象？

**答案：**

分情况讨论：

**情况一：常量池中没有 "abc"**
```
创建 2 个对象：
1. 常量池中的 "abc"
2. 堆中的 String 对象
```

**情况二：常量池中已有 "abc"**
```
创建 1 个对象：
1. 堆中的 String 对象
```

```java
String s1 = "abc";           // 常量池创建
String s2 = new String("abc"); // 堆创建

System.out.println(s1 == s2);        // false
System.out.println(s1 == s2.intern()); // true
```

**追问：intern() 方法的作用？**

> 返回字符串的规范表示（常量池中的引用）
> - 如果常量池已有该字符串，返回引用
> - 如果没有，将此字符串加入常量池并返回引用
> 
> ```java
> String s1 = new String("a") + new String("b");
> String s2 = s1.intern();  // 常量池加入 "ab"
> String s3 = "ab";
> System.out.println(s1 == s3);  // true（JDK 7+）
> ```

---

## 三、面向对象

### 7. 面向对象的三大特性？

**答案：**

**封装：**
```java
public class Person {
    private String name;  // 私有属性
    private int age;
    
    // 公共的 getter/setter
    public String getName() { return name; }
    public void setAge(int age) {
        if (age > 0) this.age = age;  // 数据校验
    }
}
```

**继承：**
```java
public class Animal {
    protected String name;
    public void eat() { System.out.println("吃东西"); }
}

public class Dog extends Animal {
    public void bark() { System.out.println("汪汪"); }
    // 继承了 Animal 的属性和方法
}
```

**多态：**
```java
Animal animal = new Dog();  // 父类引用指向子类对象
animal.eat();               // 调用子类重写的方法

// 编译看左边，运行看右边
```

**追问：多态的实现条件？**

> 1. 继承关系
> 2. 方法重写
> 3. 父类引用指向子类对象

---

### 8. 重载和重写的区别？

**答案：**

| 特性 | 重载（Overload） | 重写（Override） |
|------|------------------|------------------|
| 发生位置 | 同一个类中 | 子父类之间 |
| 方法签名 | 方法名相同，参数不同 | 方法名、参数都相同 |
| 返回类型 | 无关 | 相同或协变返回类型 |
| 访问权限 | 无关 | 不能更严格 |
| 异常 | 无关 | 不能抛出更广的异常 |

```java
// 重载：方法名相同，参数不同
class Calculator {
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
    public int add(int a, int b, int c) { return a + b + c; }
}

// 重写：子类重新定义父类方法
class Animal {
    public void speak() { System.out.println("动物叫声"); }
}
class Dog extends Animal {
    @Override
    public void speak() { System.out.println("汪汪汪"); }
}
```

**追问：重写时可以修改返回类型吗？**

> 可以，但必须是协变返回类型（子类返回类型是父类返回类型的子类）
> ```java
> class Parent { public Animal getAnimal() { return new Animal(); } }
> class Child extends Parent {
>     @Override
>     public Dog getAnimal() { return new Dog(); }  // Dog 是 Animal 子类
> }
> ```

---

### 9. 接口和抽象类的区别？

**答案：**

| 特性 | 接口 | 抽象类 |
|------|------|--------|
| 关键字 | interface | abstract class |
| 多继承 | 一个类可实现多个接口 | 只能单继承 |
| 成员变量 | 只能是 public static final | 可以有各种类型 |
| 构造方法 | 不能有 | 可以有 |
| 方法 | JDK8 前只能抽象方法 | 可以有抽象和具体方法 |
| 设计理念 | 定义行为规范 | 代码复用 + 模板设计 |

**追问：JDK 8 接口新增了什么？**

> 1. **default 方法**：可以有默认实现
> 2. **static 方法**：可以有静态方法
> ```java
> interface MyInterface {
>     void abstractMethod();           // 抽象方法
>     
>     default void defaultMethod() {   // 默认方法
>         System.out.println("默认实现");
>     }
>     
>     static void staticMethod() {     // 静态方法
>         System.out.println("静态方法");
>     }
> }
> ```

**追问：什么时候用接口，什么时候用抽象类？**

> - **接口**：定义能力、行为规范（如 Comparable、Serializable）
> - **抽象类**：有共同属性和方法，需要代码复用（如 AbstractList）

---

### 10. 深拷贝和浅拷贝的区别？

**答案：**

```java
class Person implements Cloneable {
    String name;
    Address address;  // 引用类型
    
    // 浅拷贝
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();  // 只复制基本类型和引用地址
    }
    
    // 深拷贝
    public Person deepClone() throws CloneNotSupportedException {
        Person cloned = (Person) super.clone();
        cloned.address = address.clone();  // 递归复制引用对象
        return cloned;
    }
}
```

| 类型 | 浅拷贝 | 深拷贝 |
|------|--------|--------|
| 基本类型 | 复制值 | 复制值 |
| 引用类型 | 复制引用地址 | 创建新对象 |
| 修改影响 | 会影响原对象 | 不影响原对象 |

**追问：如何实现深拷贝？**

> 1. **重写 clone() 方法**：递归克隆引用对象
> 2. **序列化/反序列化**：
> ```java
> public <T> T deepCopy(T obj) throws Exception {
>     ByteArrayOutputStream bos = new ByteArrayOutputStream();
>     ObjectOutputStream oos = new ObjectOutputStream(bos);
>     oos.writeObject(obj);
>     
>     ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
>     ObjectInputStream ois = new ObjectInputStream(bis);
>     return (T) ois.readObject();
> }
> ```

---

## 四、其他基础

### 11. == 和 equals() 的区别？

**答案：**

- **==**：比较内存地址（引用是否指向同一对象）；对于基本类型比较值
- **equals()**：默认比较地址，可以重写比较内容

```java
String s1 = new String("hello");
String s2 = new String("hello");

s1 == s2;          // false，不同对象
s1.equals(s2);     // true，内容相同

String s3 = "hello";
String s4 = "hello";
s3 == s4;          // true，常量池同一对象
```

**追问：hashCode() 和 equals() 的关系？**

> 1. equals() 相等 → hashCode() 必须相等
> 2. hashCode() 相等 → equals() 不一定相等（哈希冲突）
> 3. 重写 equals() 必须同时重写 hashCode()
> 
> 原因：HashMap、HashSet 先比较 hashCode，再比较 equals

---

### 12. Java 异常体系？

**答案：**

```
                    Throwable
                       │
          ┌────────────┴────────────┐
          │                         │
       Error                     Exception
          │                         │
   (程序无法处理)          ┌─────────┴─────────┐
                          │                   │
                  Checked异常            Unchecked异常
                                          │
                   IOException         RuntimeException
                   SQLException        │
                   ClassNotFoundException ├── NullPointerException
                                         ├── ArrayIndexOutOfBoundsException
                                         ├── ArithmeticException
                                         └── ClassCastException
```

**追问：finally 块一定会执行吗？**

> 几乎一定会，除了：
> 1. 在 try/catch 块中调用了 System.exit()
> 2. 线程死亡
> 3. CPU 关机

**追问：try-with-resources？**

> JDK 7 引入，自动关闭实现 AutoCloseable 的资源
> ```java
> try (FileInputStream fis = new FileInputStream("file.txt")) {
>     // 使用资源，自动关闭
> } catch (IOException e) {
>     e.printStackTrace();
> }
> ```

---

### 13. Java 有哪几种引用类型？

**答案：**

| 引用类型 | 回收时机 | 用途 |
|----------|----------|------|
| 强引用 | 永不回收（除非不可达） | 普通对象 |
| 软引用 | 内存不足时回收 | 缓存 |
| 弱引用 | 下次 GC 时回收 | ThreadLocal、WeakHashMap |
| 虚引用 | 随时可能回收，get() 返回 null | 跟踪对象回收 |

```java
// 强引用
Object strong = new Object();

// 软引用
SoftReference<Object> soft = new SoftReference<>(new Object());

// 弱引用
WeakReference<Object> weak = new WeakReference<>(new Object());

// 虚引用
ReferenceQueue<Object> queue = new ReferenceQueue<>();
PhantomReference<Object> phantom = new PhantomReference<>(new Object(), queue);
```

**追问：ThreadLocal 为什么可能内存泄漏？**

> ThreadLocalMap 的 Entry 继承自 WeakReference，key（ThreadLocal）是弱引用。
> - key 被 GC 回收后，value 还在
> - 如果线程不结束（线程池），value 无法回收
> 
> 解决方案：使用完必须调用 remove()

---

### 14. 反射是什么？有什么应用？

**答案：**

反射是在运行时获取类的信息并操作类的能力。

```java
// 获取 Class 对象
Class<?> clazz = Class.forName("com.example.User");
Class<?> clazz = User.class;
Class<?> clazz = user.getClass();

// 创建实例
Object obj = clazz.newInstance();
User user = clazz.getDeclaredConstructor().newInstance();

// 获取方法并调用
Method method = clazz.getMethod("setName", String.class);
method.invoke(obj, "张三");

// 获取字段并修改
Field field = clazz.getDeclaredField("name");
field.setAccessible(true);  // 突破 private 限制
field.set(obj, "李四");
```

**应用场景：**
- 框架设计（Spring IoC）
- 动态代理
- 注解处理
- 单元测试

**追问：反射的缺点？**

> 1. 性能较低（可通过 Method.setAccessible(true) 优化）
> 2. 破坏封装性
> 3. 安全风险

---

### 15. Java 泛型的类型擦除？

**答案：**

Java 泛型在编译时进行类型检查，运行时类型信息被擦除。

```java
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

// 编译后都是 ArrayList
System.out.println(strings.getClass() == integers.getClass()); // true

// 类型擦除后，泛型参数变成 Object 或边界类型
// List<String> → List
// List<T extends Number> → List<Number>
```

**追问：泛型擦除的问题？**

> 1. 不能用基本类型作为泛型参数（`List<int>` 不行）
> 2. 不能创建泛型数组
> 3. 不能实例化泛型对象
> ```java
> // 都不行
> new T();
> new T[10];
> T.class;
> ```

---

### 16. Java 8 新特性有哪些？

**答案：**

1. **Lambda 表达式**
   ```java
   (a, b) -> a + b
   ```

2. **函数式接口**
   ```java
   @FunctionalInterface
   interface Calculator { int calc(int a, int b); }
   ```

3. **方法引用**
   ```java
   list.forEach(System.out::println);
   ```

4. **Stream API**
   ```java
   list.stream()
       .filter(s -> s.length() > 3)
       .map(String::toUpperCase)
       .collect(Collectors.toList());
   ```

5. **Optional**
   ```java
   Optional.ofNullable(user)
       .map(User::getName)
       .orElse("default");
   ```

6. **默认方法**
   ```java
   interface MyInterface {
       default void defaultMethod() { }
   }
   ```

7. **新的日期 API**
   ```java
   LocalDate.now();
   LocalDateTime.now();
   ```

---

## 小结

本文档涵盖了 Java 基础面试的高频考点：

- 数据类型与包装类
- 字符串特性与比较
- 面向对象三大特性
- 重载与重写
- 接口与抽象类
- 深浅拷贝
- 异常体系
- 引用类型
- 反射与泛型
- Java 8 新特性
