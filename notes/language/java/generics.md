# Java 泛型

> 泛型（Generics）是 Java 5 引入的特性，允许在定义类、接口和方法时使用类型参数。泛型提供了编译时类型安全检查，消除了大部分类型转换，是 Java 类型系统的重要组成部分。

## 泛型

### 没有泛型的痛苦

```java
// Java 5 之前（没有泛型）
import java.util.ArrayList;
import java.util.List;

public class WithoutGenerics {
    public static void main(String[] args) {
        // 创建集合时没有指定类型
        List list = new ArrayList();
        
        // 可以添加任何类型的对象
        list.add("Hello");
        list.add(123);          // 编译时不报错！
        list.add(new Person("张三", 25));
        
        // 取出时需要强制类型转换
        String s = (String) list.get(0);  // OK
        
        // 运行时错误！ClassCastException
        // String s2 = (String) list.get(1);  // 123 不是字符串
        
        // 问题：
        // 1. 编译时无法发现类型错误
        // 2. 每次取出都要强制转换，繁琐
        // 3. 容易出现运行时 ClassCastException
    }
}
```

### 泛型的解决方案

```java
// 使用泛型
import java.util.ArrayList;
import java.util.List;

public class WithGenerics {
    public static void main(String[] args) {
        // 指定集合存储的类型
        List<String> list = new ArrayList<>();
        
        list.add("Hello");
        // list.add(123);       // 编译时错误！类型不匹配
        // list.add(new Person("张三", 25));  // 编译时错误！
        
        // 取出时不需要强制转换，编译器自动处理
        String s = list.get(0);  // 直接是 String 类型
        
        // 好处：
        // 1. 编译时就能发现类型错误，更安全
        // 2. 不需要强制类型转换，更简洁
        // 3. 代码更易读，类型意图明确
    }
}
```

### 泛型的本质

泛型的本质是**参数化类型**：

```java
// 普通：类型是固定的
String name = "张三";        // name 的类型固定是 String

// 泛型：类型是参数化的（可以传入不同的类型）
List<String> stringList;    // 传入 String 类型
List<Integer> intList;      // 传入 Integer 类型
List<Person> personList;    // 传入 Person 类型
```

---

## 泛型基础

### 泛型类

泛型类是在类定义时使用类型参数，让类的字段和方法可以使用这个类型参数。

```java
// 定义泛型类
// <T> 是类型参数，可以是任何合法的标识符
// 习惯上用 T（Type）、E（Element）、K（Key）、V（Value）、N（Number）
public class Box<T> {
    // T 可以用在字段声明中
    private T value;
    
    // T 可以用在方法参数中
    public void set(T value) {
        this.value = value;
    }
    
    // T 可以用在返回值中
    public T get() {
        return value;
    }
}

// 使用泛型类
public class Main {
    public static void main(String[] args) {
        // 创建 Box 对象，指定类型参数为 String
        // 编译器会检查：set 方法只能传 String，get 方法返回 String
        Box<String> stringBox = new Box<String>();
        stringBox.set("Hello");
        String s = stringBox.get();  // 不需要强制转换
        
        // 菱形语法（Java 7+）：右边的类型参数可以省略，编译器会自动推断
        Box<Integer> intBox = new Box<>();  // 等价于 new Box<Integer>()
        intBox.set(123);
        int i = intBox.get();
        
        // ⚠️ 泛型类型参数必须是引用类型，不能是基本类型
        // Box<int> box = new Box<>();  // 编译错误！
        // Box<int> box = new Box<Integer>();  // 使用包装类
    }
}
```

### 泛型接口

接口也可以定义类型参数。

```java
// 定义泛型接口
public interface Generator<T> {
    // 生成一个类型为 T 的对象
    T generate();
}

// ========== 实现方式一：指定具体类型 ==========
// 实现 Generator 接口，指定类型参数为 String
public class StringGenerator implements Generator<String> {
    @Override
    public String generate() {
        return "Hello";
    }
}

// 使用
Generator<String> gen = new StringGenerator();
String s = gen.generate();

// ========== 实现方式二：保留类型参数 ==========
// 实现类也定义类型参数，传递给接口
public class GenericGenerator<T> implements Generator<T> {
    private T defaultValue;
    
    public GenericGenerator(T defaultValue) {
        this.defaultValue = defaultValue;
    }
    
    @Override
    public T generate() {
        return defaultValue;
    }
}

// 使用
Generator<Integer> intGen = new GenericGenerator<>(100);
Integer value = intGen.generate();
```

### 泛型方法

泛型方法是在方法声明中定义类型参数，可以在普通类或泛型类中定义。

```java
public class Utils {
    // 泛型方法：<T> 放在返回值类型之前
    // 这个 T 只在这个方法内部有效
    public static <T> T getFirst(List<T> list) {
        if (list == null || list.isEmpty()) {
            return null;
        }
        return list.get(0);
    }
    
    // 多个类型参数
    public static <K, V> void print(K key, V value) {
        System.out.println(key + ": " + value);
    }
    
    // 有界类型参数：限制类型参数的范围
    // <T extends Number> 表示 T 必须是 Number 或其子类
    public static <T extends Number> double sum(List<T> list) {
        double total = 0;
        for (T num : list) {
            // 因为 T extends Number，所以可以调用 Number 的方法
            total += num.doubleValue();
        }
        return total;
    }
    
    // 泛型方法 vs 泛型类
    // 泛型类的类型参数在整个类中有效
    // 泛型方法的类型参数只在方法内有效
}

// 调用泛型方法
public class Main {
    public static void main(String[] args) {
        // 类型推断：编译器根据参数自动推断类型
        String first = Utils.getFirst(Arrays.asList("A", "B", "C"));
        // 等价于显式指定类型：Utils.<String>getFirst(...)
        
        Utils.print("name", "张三");       // K=String, V=String
        Utils.print("age", 25);           // K=String, V=Integer
        
        double result = Utils.sum(Arrays.asList(1, 2, 3));          // List<Integer>
        double result2 = Utils.sum(Arrays.asList(1.0, 2.0, 3.0));   // List<Double>
        // Utils.sum(Arrays.asList("A", "B"));  // 编译错误！String 不是 Number 的子类
    }
}
```

---

## 类型参数命名约定

Java 泛型使用单个大写字母作为类型参数，这是约定俗成的命名规则：

| 符号 | 全称 | 含义 | 使用场景 |
|------|------|------|----------|
| `T` | Type | 类型 | 泛型类、方法的默认命名 |
| `E` | Element | 元素 | 集合中的元素类型（如 `List<E>`） |
| `K` | Key | 键 | Map 中的键类型（如 `Map<K, V>`） |
| `V` | Value | 值 | Map 中的值类型 |
| `N` | Number | 数值 | 数值类型 |
| `R` | Result | 结果 | 返回结果类型 |
| `S`, `U`, `V` | - | 第二、三、四个类型 | 多类型参数时 |

```java
// 标准库中的例子
public interface List<E> { ... }           // E 表示元素
public interface Map<K, V> { ... }         // K 表示键，V 表示值

// 多类型参数
public class Pair<K, V> {
    private K key;
    private V value;
    // ...
}
```

---

## 类型通配符

通配符用于表示未知类型，是 Java 泛型的重要特性。

### 为什么需要通配符？

```java
// 问题：泛型类型之间没有继承关系
List<Object> objList = new ArrayList<>();
List<String> strList = new ArrayList<>();

// objList = strList;  // 编译错误！
// 虽然 String 是 Object 的子类，但 List<String> 不是 List<Object> 的子类

// 原因：如果允许这样赋值，会出现类型安全问题
// objList.add(123);  // 如果 objList 指向 strList，这就把整数加进了字符串列表
// String s = strList.get(0);  // 运行时错误！

// 解决方案：使用通配符
List<?> anyList = strList;  // ✓ 可以赋值
```

### 无界通配符 `?`

`?` 表示未知类型，可以匹配任何类型。

```java
// 可以接收任何类型的 List
public void printList(List<?> list) {
    // ? 表示未知类型，只能当作 Object 使用
    for (Object elem : list) {  // 元素是 Object 类型
        System.out.println(elem);
    }
}

printList(Arrays.asList(1, 2, 3));           // List<Integer>
printList(Arrays.asList("A", "B", "C"));     // List<String>

// ⚠️ 无界通配符的限制
List<?> list = new ArrayList<String>();
// list.add("hello");  // 编译错误！不能添加元素（除了 null）
// 原因：编译器不知道 ? 是什么类型，所以不能安全地添加元素
list.add(null);         // ✓ null 是所有引用类型的值

// 可以安全地读取（当作 Object）
Object obj = list.get(0);
```

### 上界通配符 `? extends T`

`? extends T` 表示类型必须是 `T` 或 `T` 的子类。

**适用场景**：从集合中**读取**数据（生产者）。

```java
// 可以读取 Number 或其子类的元素
// 参数可以是 List<Integer>、List<Double>、List<Long> 等
public double sum(List<? extends Number> list) {
    double total = 0;
    for (Number num : list) {  // 可以安全地读取为 Number
        total += num.doubleValue();
    }
    return total;
}

sum(Arrays.asList(1, 2, 3));           // List<Integer>
sum(Arrays.asList(1.0, 2.0, 3.0));     // List<Double>
sum(Arrays.asList(1L, 2L, 3L));        // List<Long>

// ⚠️ 上界通配符的限制：不能添加元素
List<? extends Number> list = new ArrayList<Integer>();
// list.add(1);        // 编译错误！
// list.add(1.0);      // 编译错误！
// 原因：编译器只知道 ? 是 Number 的某个子类，但不知道具体是哪个
// 可能是 Integer，也可能是 Double，所以不能安全地添加任何非 null 元素

// 为什么这样设计？
// 如果允许添加，以下代码就会出现类型安全问题：
// List<? extends Number> list = new ArrayList<Integer>();
// list.add(1.0);  // 如果允许，就把 Double 加进了 Integer 列表
```

### 下界通配符 `? super T`

`? super T` 表示类型必须是 `T` 或 `T` 的父类。

**适用场景**：向集合中**写入**数据（消费者）。

```java
// 可以添加 Integer 或其子类的元素
// 参数可以是 List<Integer>、List<Number>、List<Object>
public void addNumbers(List<? super Integer> list) {
    list.add(1);      // ✓ Integer 是 Integer
    list.add(2);      // ✓ 可以安全地添加 Integer
    // list.add(1.0); // 编译错误！Double 不是 Integer
}

List<Number> numbers = new ArrayList<>();
addNumbers(numbers);  // Integer 添加到 Number 列表

List<Object> objects = new ArrayList<>();
addNumbers(objects);  // Integer 添加到 Object 列表

// ⚠️ 下界通配符的限制：读取时只能当作 Object
List<? super Integer> list = new ArrayList<Number>();
list.add(1);
Object obj = list.get(0);  // 只能作为 Object 读取
// Integer i = list.get(0);  // 编译错误！不能保证是 Integer
```

### PECS 原则

📌 **Producer-Extends, Consumer-Super**（生产者用 extends，消费者用 super）

这是选择通配符的核心原则：

```java
// 生产者：从集合中读取数据
// 使用 ? extends T
public void process(List<? extends Number> producer) {
    Number n = producer.get(0);  // 读取（生产）Number
    // producer.add(1);  // 不能写入
}

// 消费者：向集合中写入数据
// 使用 ? super T
public void populate(List<? super Integer> consumer) {
    consumer.add(1);  // 写入（消费）Integer
    // Integer i = consumer.get(0);  // 不能精确读取
}

// 同时读写：使用精确类型
public void both(List<Integer> list) {
    list.add(1);            // 写
    Integer i = list.get(0);  // 读
}

// 实际例子：Collections.copy 方法
public static <T> void copy(List<? super T> dest, List<? extends T> src) {
    // src 是生产者，从中读取数据 → ? extends T
    // dest 是消费者，往其中写入数据 → ? super T
    for (int i = 0; i < src.size(); i++) {
        dest.set(i, src.get(i));
    }
}
```

---

## 类型擦除

📌 Java 泛型是**编译时**特性，运行时类型信息会被擦除。这是 Java 为了兼容旧版本而做出的设计妥协。

### 擦除过程

```java
// 编译前（源代码）
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

strings.add("Hello");
String s = strings.get(0);

// 编译后（字节码，类型擦除）
List strings = new ArrayList();  // List<String> → List
List integers = new ArrayList();  // List<Integer> → List

strings.add("Hello");
String s = (String) strings.get(0);  // 编译器插入强制转换

// 运行时，两个 List 的类型相同
System.out.println(strings.getClass() == integers.getClass());  // true
// 都是 java.util.ArrayList
```

### 类型擦除的影响

```java
// 1. 不能用基本类型作为类型参数
// List<int> list;  // 编译错误！
List<Integer> list;  // 使用包装类

// 2. 不能实例化类型参数
public <T> void method() {
    // T t = new T();  // 编译错误！
    
    // 解决方案：使用反射或 Supplier
    // T t = (T) Class.forName("com.example.MyClass").newInstance();
}

// 使用 Supplier（推荐）
public <T> T create(Supplier<T> supplier) {
    return supplier.get();
}
MyClass obj = create(MyClass::new);

// 3. 不能创建泛型数组
// T[] array = new T[10];  // 编译错误！

// 解决方案一：使用 Object 数组并强制转换
T[] array = (T[]) new Object[10];  // 有未检查警告

// 解决方案二：使用反射
@SuppressWarnings("unchecked")
T[] array = (T[]) Array.newInstance(clazz, 10);

// 解决方案三：使用 List<T>（推荐）
List<T> list = new ArrayList<>();

// 4. 不能重载具有相同擦除后签名的方法
public class Example {
    // public void method(List<String> list) { }
    // public void method(List<Integer> list) { }  // 编译错误！
    // 两个方法擦除后都是 method(List)
}
```

### 桥接方法

编译器会自动生成桥接方法来保证多态的正确性：

```java
// 泛型类
public class Box<T> {
    private T value;
    public void set(T value) { this.value = value; }
}

// 子类
public class StringBox extends Box<String> {
    @Override
    public void set(String value) { ... }
}

// 编译后，StringBox 实际上有两个 set 方法：
// 1. public void set(String value)  // 你写的方法
// 2. public void set(Object value)  // 编译器生成的桥接方法
//    {
//        set((String) value);  // 调用上面的方法
//    }

// 这样才能保证多态的正确性：
Box<String> box = new StringBox();
box.set("Hello");  // 编译时调用 set(Object)，运行时调用桥接方法
```

---

## 泛型与继承

```java
// 泛型类型之间没有继承关系
List<Object> list1 = new ArrayList<String>();  // ❌ 编译错误

// 但可以向上转型为原始类型
List list2 = new ArrayList<String>();  // ✓ 不推荐，有警告

// 类型参数可以继承
// 这意味着 StringBox 是 Box<String> 的子类
public class StringBox extends Box<String> { }

Box<String> box = new StringBox();  // ✓ 向上转型

// 多个类型参数
public class Pair<K, V> { }
public class StringIntPair extends Pair<String, Integer> { }
```

---

## 泛型的实际应用

### 自定义泛型容器

```java
// 一个简单的泛型栈实现
public class Stack<T> {
    private final List<T> elements = new ArrayList<>();
    
    public void push(T item) {
        elements.add(item);
    }
    
    public T pop() {
        if (elements.isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.remove(elements.size() - 1);
    }
    
    public T peek() {
        if (elements.isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.get(elements.size() - 1);
    }
    
    public boolean isEmpty() {
        return elements.isEmpty();
    }
    
    public int size() {
        return elements.size();
    }
}

// 使用
Stack<String> stack = new Stack<>();
stack.push("A");
stack.push("B");
System.out.println(stack.pop());  // "B"
```

### 泛型工厂方法

```java
public class Factory {
    // 泛型工厂方法
    public static <T> T createInstance(Class<T> clazz) {
        try {
            return clazz.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("创建实例失败", e);
        }
    }
}

// 使用
Person person = Factory.createInstance(Person.class);
String str = Factory.createInstance(String.class);
```

### 类型安全的异构容器

```java
import java.util.HashMap;
import java.util.Map;

// 使用 Class 作为键，实现类型安全的异构容器
public class Context {
    private final Map<Class<?>, Object> map = new HashMap<>();
    
    // 存储任意类型的值
    public <T> void put(Class<T> type, T value) {
        map.put(type, value);
    }
    
    // 类型安全地获取值
    public <T> T get(Class<T> type) {
        return type.cast(map.get(type));
    }
}

// 使用
Context context = new Context();
context.put(String.class, "Hello");
context.put(Integer.class, 123);
context.put(Person.class, new Person("张三", 25));

String s = context.get(String.class);     // 不需要强制转换
Integer i = context.get(Integer.class);
Person p = context.get(Person.class);
```

---

## 小结

| 概念 | 语法 | 说明 |
|------|------|------|
| **泛型类** | `class Box<T>` | 类级别的类型参数 |
| **泛型接口** | `interface Generator<T>` | 接口级别的类型参数 |
| **泛型方法** | `<T> T method()` | 方法级别的类型参数 |
| **上界通配符** | `? extends T` | 读取数据（生产者），不能写入 |
| **下界通配符** | `? super T` | 写入数据（消费者），读取为 Object |
| **无界通配符** | `?` | 未知类型，只能读取为 Object |
| **类型擦除** | - | 泛型信息在编译后被擦除 |

### 最佳实践

1. **优先使用泛型**：避免原始类型，确保类型安全
2. **遵循 PECS 原则**：生产者用 extends，消费者用 super
3. **使用有界类型参数**：限制类型范围，提供更多操作
4. **避免创建泛型数组**：使用 `List<T>` 替代 `T[]`
5. **消除未检查警告**：确保类型安全后使用 `@SuppressWarnings("unchecked")`

### 初学者常见错误

1. **泛型类型不能是基本类型**：使用包装类 `Integer`、`Double` 等
2. **不能创建泛型数组**：使用 `List<T>` 或反射
3. **泛型类型之间没有继承关系**：`List<String>` 不是 `List<Object>` 的子类
4. **静态成员不能使用类的类型参数**：静态成员在类加载时存在，此时还没有具体类型