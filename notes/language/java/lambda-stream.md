# Java Lambda 与 Stream

> Lambda 表达式和 Stream API 是 Java 8 引入的重要特性，使 Java 支持函数式编程风格，让代码更加简洁、易读。

## Lambda 表达式

### 基本语法

```java
(参数列表) -> { 方法体 }
```

### 示例

```java
// 无参数
Runnable r1 = () -> System.out.println("Hello");
Runnable r2 = () -> {
    System.out.println("Hello");
    System.out.println("World");
};

// 单参数（可省略括号）
Consumer<String> c1 = (s) -> System.out.println(s);
Consumer<String> c2 = s -> System.out.println(s);

// 多参数
BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
BiFunction<Integer, Integer, Integer> add2 = (a, b) -> {
    return a + b;
};

// 对比：匿名内部类 vs Lambda
// 匿名内部类
Runnable old = new Runnable() {
    @Override
    public void run() {
        System.out.println("Hello");
    }
};

// Lambda
Runnable now = () -> System.out.println("Hello");
```

---

## 函数式接口

📌 **函数式接口**是只有一个抽象方法的接口，可以使用 `@FunctionalInterface` 注解标注。

### 内置函数式接口

| 接口 | 参数 | 返回值 | 说明 |
|------|:----:|:------:|------|
| `Runnable` | 无 | 无 | 无参无返回值 |
| `Supplier<T>` | 无 | T | 提供数据 |
| `Consumer<T>` | T | 无 | 消费数据 |
| `Function<T, R>` | T | R | 类型转换 |
| `Predicate<T>` | T | boolean | 条件判断 |
| `BiFunction<T, U, R>` | T, U | R | 两个参数的函数 |

### 常用示例

```java
// Supplier：提供数据
Supplier<String> supplier = () -> "Hello";
String s = supplier.get();

// Consumer：消费数据
Consumer<String> consumer = str -> System.out.println(str);
consumer.accept("Hello");

// Function：类型转换
Function<String, Integer> toLength = str -> str.length();
int len = toLength.apply("Hello");  // 5

// Predicate：条件判断
Predicate<Integer> isEven = n -> n % 2 == 0;
boolean result = isEven.test(4);  // true

// BiFunction：两个参数
BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
int sum = add.apply(1, 2);  // 3
```

### 方法引用

```java
// 静态方法引用：类名::静态方法
Function<String, Integer> parser = Integer::parseInt;

// 实例方法引用：对象::实例方法
String str = "Hello";
Supplier<Integer> lengthSupplier = str::length;

// 类方法引用：类名::实例方法
Function<String, Integer> toLength = String::length;

// 构造方法引用：类名::new
Supplier<List<String>> listSupplier = ArrayList::new;
Function<Integer, List<String>> listCreator = ArrayList::new;
```

---

## Stream API

📌 **Stream** 是 Java 8 引入的流式处理 API，支持对集合进行函数式操作。

### 创建 Stream

```java
// 从集合创建
List<String> list = Arrays.asList("a", "b", "c");
Stream<String> stream = list.stream();
Stream<String> parallelStream = list.parallelStream();  // 并行流

// 从数组创建
String[] arr = {"a", "b", "c"};
Stream<String> stream2 = Arrays.stream(arr);

// 使用 Stream.of
Stream<String> stream3 = Stream.of("a", "b", "c");

// 无限流
Stream<Double> randoms = Stream.generate(Math::random);
Stream<Integer> naturals = Stream.iterate(0, n -> n + 1);

// 有限流（Java 9+）
Stream<Integer> limited = Stream.iterate(0, n -> n < 10, n -> n + 1);
```

### 中间操作

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// filter：过滤
Stream<Integer> evens = numbers.stream().filter(n -> n % 2 == 0);

// map：映射转换
Stream<Integer> squares = numbers.stream().map(n -> n * n);
Stream<String> strings = numbers.stream().map(Object::toString);

// flatMap：扁平化映射
List<List<Integer>> nested = Arrays.asList(
    Arrays.asList(1, 2),
    Arrays.asList(3, 4)
);
Stream<Integer> flattened = nested.stream().flatMap(List::stream);

// distinct：去重
Stream<Integer> distinct = numbers.stream().distinct();

// sorted：排序
Stream<Integer> sorted = numbers.stream().sorted();
Stream<Integer> sortedDesc = numbers.stream().sorted((a, b) -> b - a);

// limit：限制数量
Stream<Integer> first5 = numbers.stream().limit(5);

// skip：跳过元素
Stream<Integer> skip2 = numbers.stream().skip(2);

// peek：查看元素（主要用于调试）
numbers.stream()
    .peek(n -> System.out.println("处理: " + n))
    .forEach(System.out::println);
```

### 终端操作

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

// forEach：遍历
numbers.stream().forEach(System.out::println);

// collect：收集结果
List<Integer> list = numbers.stream().collect(Collectors.toList());
Set<Integer> set = numbers.stream().collect(Collectors.toSet());
String joined = numbers.stream()
    .map(String::valueOf)
    .collect(Collectors.joining(", "));

// toArray：转为数组
Integer[] arr = numbers.stream().toArray(Integer[]::new);

// reduce：归约
int sum = numbers.stream().reduce(0, Integer::sum);
Optional<Integer> max = numbers.stream().reduce(Integer::max);

// count：计数
long count = numbers.stream().count();

// min/max：最小/最大值
Optional<Integer> min = numbers.stream().min(Integer::compare);
Optional<Integer> max = numbers.stream().max(Integer::compare);

// findFirst/findAny：查找
Optional<Integer> first = numbers.stream().findFirst();
Optional<Integer> any = numbers.stream().findAny();

// anyMatch/allMatch/noneMatch：匹配
boolean hasEven = numbers.stream().anyMatch(n -> n % 2 == 0);
boolean allPositive = numbers.stream().allMatch(n -> n > 0);
boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);
```

---

## Collectors 工具类

```java
List<Person> people = Arrays.asList(
    new Person("张三", 25),
    new Person("李四", 30),
    new Person("王五", 25)
);

// 转集合
List<String> names = people.stream()
    .map(Person::getName)
    .collect(Collectors.toList());

Set<Integer> ages = people.stream()
    .map(Person::getAge)
    .collect(Collectors.toSet());

// 转 Map
Map<String, Integer> nameToAge = people.stream()
    .collect(Collectors.toMap(Person::getName, Person::getAge));

// 分组
Map<Integer, List<Person>> byAge = people.stream()
    .collect(Collectors.groupingBy(Person::getAge));

// 分区
Map<Boolean, List<Person>> partition = people.stream()
    .collect(Collectors.partitioningBy(p -> p.getAge() >= 30));

// 统计
IntSummaryStatistics stats = people.stream()
    .collect(Collectors.summarizingInt(Person::getAge));
// stats.getCount(), stats.getSum(), stats.getAverage(), stats.getMax(), stats.getMin()

// 拼接字符串
String nameStr = people.stream()
    .map(Person::getName)
    .collect(Collectors.joining(", ", "[", "]"));
// [张三, 李四, 王五]
```

---

## Optional 类

📌 **Optional** 是一个容器类，用于避免空指针异常。

### 创建 Optional

```java
// 创建 Optional
Optional<String> opt1 = Optional.of("Hello");      // 非空值
Optional<String> opt2 = Optional.ofNullable(null); // 可空值
Optional<String> opt3 = Optional.empty();          // 空值
```

### 常用方法

```java
Optional<String> opt = Optional.of("Hello");

// 判断是否有值
boolean present = opt.isPresent();    // true
boolean empty = opt.isEmpty();        // false（Java 11+）

// 获取值
String value = opt.get();             // 获取值（可能抛异常）

// 安全获取值
String val1 = opt.orElse("默认值");    // 有值返回值，否则返回默认值
String val2 = opt.orElseGet(() -> "默认值");  // 懒加载默认值
String val3 = opt.orElseThrow(() -> new RuntimeException());  // 无值时抛异常

// 消费值
opt.ifPresent(v -> System.out.println(v));
opt.ifPresentOrElse(
    v -> System.out.println(v),
    () -> System.out.println("空值")
);

// 转换
Optional<Integer> length = opt.map(String::length);
Optional<String> upper = opt.filter(s -> s.length() > 3).map(String::toUpperCase);

// 扁平化
Optional<String> result = opt.flatMap(s -> Optional.of(s.toUpperCase()));
```

### 实际应用

```java
public String getUserName(Long userId) {
    return Optional.ofNullable(userService.findById(userId))
        .map(User::getName)
        .orElse("未知用户");
}

// 链式调用避免空指针
String city = Optional.ofNullable(user)
    .map(User::getAddress)
    .map(Address::getCity)
    .orElse("未知");
```

---

## 实战示例

### 数据处理

```java
List<Person> people = Arrays.asList(
    new Person("张三", 25, "北京"),
    new Person("李四", 30, "上海"),
    new Person("王五", 25, "北京"),
    new Person("赵六", 35, "广州")
);

// 找出北京的用户名
List<String> beijingNames = people.stream()
    .filter(p -> "北京".equals(p.getCity()))
    .map(Person::getName)
    .collect(Collectors.toList());

// 按城市分组
Map<String, List<Person>> byCity = people.stream()
    .collect(Collectors.groupingBy(Person::getCity));

// 计算平均年龄
double avgAge = people.stream()
    .mapToInt(Person::getAge)
    .average()
    .orElse(0);

// 找出年龄最大的
Optional<Person> oldest = people.stream()
    .max(Comparator.comparing(Person::getAge));
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **Lambda** | 简洁的匿名函数语法 |
| **函数式接口** | 只有一个抽象方法的接口 |
| **方法引用** | 用 `::` 引用现有方法 |
| **Stream** | 流式处理集合数据 |
| **中间操作** | 返回新 Stream，可链式调用 |
| **终端操作** | 产生结果，终止 Stream |
| **Optional** | 避免 NullPointerException |
| **Collectors** | 收集 Stream 结果的工具类 |
