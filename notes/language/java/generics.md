# Java 泛型

> 泛型（Generics）是 Java 5 引入的特性，允许在定义类、接口和方法时使用类型参数。泛型提供了编译时类型安全检查，消除了大部分类型转换，是 Java 类型系统的重要组成部分。

## 泛型基础

### 为什么需要泛型

```java
// 没有泛型（Java 5 之前）
List list = new ArrayList();
list.add("Hello");
list.add(123);          // 编译时不报错！
String s = (String) list.get(0);  // 需要强制类型转换
// String s2 = (String) list.get(1);  // 运行时 ClassCastException！

// 使用泛型
List<String> list = new ArrayList<>();
list.add("Hello");
// list.add(123);       // 编译时错误！类型安全
String s = list.get(0);  // 无需转换
```

### 泛型类

```java
// 定义泛型类
public class Box<T> {
    private T value;
    
    public void set(T value) {
        this.value = value;
    }
    
    public T get() {
        return value;
    }
}

// 使用泛型类
Box<String> stringBox = new Box<>();
stringBox.set("Hello");
String s = stringBox.get();

Box<Integer> intBox = new Box<>();
intBox.set(123);
int i = intBox.get();

// 菱形语法（Java 7+）
Box<Double> doubleBox = new Box<>();  // 右边不需要写 Double
```

### 泛型接口

```java
// 定义泛型接口
public interface Generator<T> {
    T generate();
}

// 实现方式1：指定具体类型
public class StringGenerator implements Generator<String> {
    @Override
    public String generate() {
        return "Hello";
    }
}

// 实现方式2：保留类型参数
public class GenericGenerator<T> implements Generator<T> {
    @Override
    public T generate() {
        return null;
    }
}
```

### 泛型方法

```java
public class Utils {
    // 泛型方法：类型参数在方法返回值之前
    public static <T> T getFirst(List<T> list) {
        if (list.isEmpty()) return null;
        return list.get(0);
    }
    
    // 多个类型参数
    public static <K, V> void print(K key, V value) {
        System.out.println(key + ": " + value);
    }
    
    // 有界类型参数
    public static <T extends Number> double sum(List<T> list) {
        double total = 0;
        for (T num : list) {
            total += num.doubleValue();
        }
        return total;
    }
}

// 调用泛型方法（类型推断）
String first = Utils.getFirst(Arrays.asList("A", "B", "C"));
Utils.print("name", "张三");
double sum = Utils.sum(Arrays.asList(1, 2, 3));
```

---

## 类型参数命名约定

| 符号 | 含义 | 使用场景 |
|------|------|----------|
| `T` | Type（类型） | 泛型类、方法的默认命名 |
| `E` | Element（元素） | 集合中的元素类型 |
| `K` | Key（键） | Map 中的键类型 |
| `V` | Value（值） | Map 中的值类型 |
| `N` | Number（数值） | 数值类型 |
| `R` | Result（结果） | 返回结果类型 |

---

## 类型通配符

### 无界通配符 `?`

表示未知类型，通常用于：
- 编写可以使用任何类型的方法
- 作为泛型类型的参数

```java
// 可以接收任何类型的 List
public void printList(List<?> list) {
    for (Object elem : list) {
        System.out.println(elem);
    }
}

printList(Arrays.asList(1, 2, 3));
printList(Arrays.asList("A", "B", "C"));

// 限制：不能添加元素（除了 null）
List<?> list = new ArrayList<String>();
// list.add("hello");  // ❌ 编译错误
list.add(null);         // ✅ 唯一允许添加的值
```

### 上界通配符 `? extends T`

表示类型必须是 `T` 或 `T` 的子类。用于**读取**数据。

```java
// 可以读取 Number 或其子类的元素
public double sum(List<? extends Number> list) {
    double total = 0;
    for (Number num : list) {  // 可以安全地读取为 Number
        total += num.doubleValue();
    }
    return total;
}

sum(Arrays.asList(1, 2, 3));           // Integer
sum(Arrays.asList(1.0, 2.0, 3.0));     // Double
sum(Arrays.asList(1L, 2L, 3L));        // Long

// 限制：不能添加元素
List<? extends Number> list = new ArrayList<Integer>();
// list.add(1);  // ❌ 编译错误
```

### 下界通配符 `? super T`

表示类型必须是 `T` 或 `T` 的父类。用于**写入**数据。

```java
// 可以添加 Integer 或其子类的元素
public void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
    list.add(3);
}

List<Number> numbers = new ArrayList<>();
addNumbers(numbers);  // Integer 添加到 Number 列表

List<Object> objects = new ArrayList<>();
addNumbers(objects);  // Integer 添加到 Object 列表

// 限制：读取时只能得到 Object
List<? super Integer> list = new ArrayList<Number>();
list.add(1);
Object obj = list.get(0);  // 只能作为 Object 读取
```

### PECS 原则

📌 **Producer-Extends, Consumer-Super**

- **生产者**：从集合读取数据 → 使用 `? extends T`
- **消费者**：向集合写入数据 → 使用 `? super T`

```java
// 生产者：从集合读取数据
public void process(List<? extends Number> producer) {
    Number n = producer.get(0);  // 读取 Number
}

// 消费者：向集合写入数据
public void populate(List<? super Integer> consumer) {
    consumer.add(1);  // 添加 Integer
}

// 同时读写：使用精确类型
public void both(List<Integer> list) {
    list.add(1);          // 写
    Integer i = list.get(0);  // 读
}
```

---

## 类型擦除

📌 Java 泛型是**编译时**特性，运行时类型信息会被擦除。这是 Java 为了兼容旧版本而做出的设计妥协。

### 擦除过程

```java
// 编译前
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

// 编译后（类型擦除）
List strings = new ArrayList();
List integers = new ArrayList();

// 运行时，两个 List 的类型相同
System.out.println(strings.getClass() == integers.getClass());  // true
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

// 编译后，StringBox 会生成桥接方法
public void set(Object value) {
    set((String) value);  // 桥接方法
}
```

### 类型擦除的限制

```java
// ❌ 不能用基本类型作为类型参数
List<int> list;  // 编译错误

// ✅ 使用包装类
List<Integer> list;

// ❌ 不能实例化类型参数
public <T> void method() {
    T t = new T();  // 编译错误
}

// ✅ 使用反射或 Supplier
public <T> T create(Supplier<T> supplier) {
    return supplier.get();
}
T obj = create(MyClass::new);

// ❌ 不能创建泛型数组
T[] array = new T[10];  // 编译错误

// ✅ 使用 Object 数组或反射
T[] array = (T[]) new Object[10];
```

---

## 泛型与继承

```java
// 泛型类型之间没有继承关系
List<Object> list1 = new ArrayList<String>();  // ❌ 编译错误

// 但可以向上转型
List<String> list2 = new ArrayList<>();
Collection<String> coll = list2;  // ✅ 正确
```

---

## 小结

| 概念 | 语法 | 说明 |
|------|------|------|
| **泛型类** | `class Box<T>` | 类级别的类型参数 |
| **泛型接口** | `interface Generator<T>` | 接口级别的类型参数 |
| **泛型方法** | `<T> T method()` | 方法级别的类型参数 |
| **上界通配符** | `? extends T` | 读取数据（生产者） |
| **下界通配符** | `? super T` | 写入数据（消费者） |
| **无界通配符** | `?` | 未知类型 |
| **类型擦除** | - | 泛型信息在编译后被擦除 |

### 最佳实践

1. **优先使用泛型**：避免原始类型，确保类型安全
2. **遵循 PECS 原则**：生产者用 extends，消费者用 super
3. **使用有界类型参数**：限制类型范围，提供更多操作
4. **避免创建泛型数组**：使用 `List<T>` 替代 `T[]`
