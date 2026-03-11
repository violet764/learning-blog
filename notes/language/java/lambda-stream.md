# Java Lambda 与 Stream

> Lambda 表达式和 Stream API 是 Java 8 引入的重要特性，使 Java 支持函数式编程风格，让代码更加简洁、易读。

## 函数式编程

### 命令式 vs 函数式

```java
// 需求：找出列表中所有偶数并求平方和

// ========== 命令式编程（怎么做）==========
// 告诉计算机"怎么做"每一步
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
int sum = 0;
for (Integer num : numbers) {
    if (num % 2 == 0) {        // 1. 遍历
        int square = num * num; // 2. 判断偶数
        sum += square;          // 3. 求平方
    }                           // 4. 累加
}
System.out.println(sum);  // 220

// ========== 函数式编程（做什么）==========
// 告诉计算机"做什么"，不关心具体实现
int sum2 = numbers.stream()
    .filter(n -> n % 2 == 0)   // 过滤偶数
    .map(n -> n * n)           // 求平方
    .reduce(0, Integer::sum);  // 累加
System.out.println(sum2);  // 220

// 函数式编程的优点：
// 1. 代码更简洁，意图更明确
// 2. 避免可变状态，减少 bug
// 3. 更容易并行化
// 4. 更容易测试
```

### 函数式编程核心概念

```
函数式编程的特点：
├── 函数是一等公民
│   ├── 函数可以赋值给变量
│   ├── 函数可以作为参数传递
│   └── 函数可以作为返回值
├── 纯函数（无副作用）
│   ├── 相同输入总是产生相同输出
│   └── 不修改外部状态
└── 不可变性
    └── 数据不可变，修改数据会返回新对象
```

---

## Lambda 表达式

### 基本语法

```java
// Lambda 表达式语法
(参数列表) -> { 方法体 }

// 箭头 -> 将参数和方法体分开
// 参数列表：方法的输入参数
// 方法体：方法的实现逻辑
```

### Lambda 示例详解

```java
import java.util.function.*;

public class LambdaDemo {
    public static void main(String[] args) {
        // ========== 无参数 ==========
        // Runnable 接口：void run()
        
        // 匿名内部类写法
        Runnable r1 = new Runnable() {
            @Override
            public void run() {
                System.out.println("Hello");
            }
        };
        
        // Lambda 写法
        Runnable r2 = () -> System.out.println("Hello");
        
        // 多行代码需要大括号
        Runnable r3 = () -> {
            System.out.println("Hello");
            System.out.println("World");
        };
        
        // ========== 单参数 ==========
        // Consumer 接口：void accept(T t)
        
        // 匿名内部类
        Consumer<String> c1 = new Consumer<String>() {
            @Override
            public void accept(String s) {
                System.out.println(s);
            }
        };
        
        // Lambda：参数类型可以省略（编译器自动推断）
        Consumer<String> c2 = (s) -> System.out.println(s);
        
        // 单参数时括号可以省略
        Consumer<String> c3 = s -> System.out.println(s);
        
        // ========== 多参数 ==========
        // BiFunction 接口：R apply(T t, U u)
        
        // 匿名内部类
        BiFunction<Integer, Integer, Integer> add1 = new BiFunction<>() {
            @Override
            public Integer apply(Integer a, Integer b) {
                return a + b;
            }
        };
        
        // Lambda
        BiFunction<Integer, Integer, Integer> add2 = (a, b) -> a + b;
        
        // 多行代码需要大括号和 return
        BiFunction<Integer, Integer, Integer> add3 = (a, b) -> {
            int sum = a + b;
            return sum;  // 必须显式 return
        };
        
        // ========== 类型推断 ==========
        // 编译器可以根据上下文推断参数类型
        (a, b) -> a + b  // a 和 b 的类型由目标类型决定
        
        // 也可以显式指定类型
        (Integer a, Integer b) -> a + b
    }
}
```

---

## 函数式接口

📌 **函数式接口**是只有一个抽象方法的接口，可以使用 `@FunctionalInterface` 注解标注。

### 什么是函数式接口？

```java
// 函数式接口：只有一个抽象方法的接口
@FunctionalInterface  // 可选注解，编译器会检查
public interface MyFunction {
    int apply(int a, int b);
}

// 使用 Lambda 实现
MyFunction add = (a, b) -> a + b;
MyFunction multiply = (a, b) -> a * b;

int result1 = add.apply(2, 3);      // 5
int result2 = multiply.apply(2, 3); // 6

// ⚠️ 如果接口有多个抽象方法，就不是函数式接口
// @FunctionalInterface  // 编译错误！
// public interface BadInterface {
//     void method1();
//     void method2();  // 两个抽象方法
// }
```

### 内置函数式接口

Java 8 在 `java.util.function` 包中提供了常用的函数式接口：

| 接口 | 方法签名 | 参数 | 返回值 | 说明 |
|------|----------|:----:|:------:|------|
| `Runnable` | `void run()` | 无 | 无 | 无参无返回值 |
| `Supplier<T>` | `T get()` | 无 | T | 提供数据（生产者） |
| `Consumer<T>` | `void accept(T)` | T | 无 | 消费数据 |
| `Function<T, R>` | `R apply(T)` | T | R | 类型转换 |
| `Predicate<T>` | `boolean test(T)` | T | boolean | 条件判断 |
| `BiFunction<T, U, R>` | `R apply(T, U)` | T, U | R | 两个参数的函数 |
| `UnaryOperator<T>` | `T apply(T)` | T | T | 一元操作 |
| `BinaryOperator<T>` | `T apply(T, T)` | T, T | T | 二元操作 |

```java
import java.util.function.*;

public class FunctionalInterfacesDemo {
    public static void main(String[] args) {
        // ========== Supplier：提供数据（无参有返回值）==========
        // 就像一个工厂，每次调用 get() 都返回一个值
        Supplier<String> supplier = () -> "Hello";
        String s = supplier.get();  // "Hello"
        
        // 常用场景：创建对象、提供默认值
        Supplier<List<String>> listSupplier = ArrayList::new;
        List<String> list = listSupplier.get();  // 新的空列表
        
        // ========== Consumer：消费数据（有参无返回值）==========
        // 就像一个处理器，接收数据并处理
        Consumer<String> consumer = str -> System.out.println(str);
        consumer.accept("Hello");  // 打印 Hello
        
        // andThen：链式消费
        Consumer<String> print = s -> System.out.println("打印：" + s);
        Consumer<String> save = s -> System.out.println("保存：" + s);
        print.andThen(save).accept("数据");
        // 输出：
        // 打印：数据
        // 保存：数据
        
        // ========== Function：类型转换 ==========
        // 接收一个类型的值，返回另一个类型的值
        Function<String, Integer> toLength = str -> str.length();
        int len = toLength.apply("Hello");  // 5
        
        // andThen：先应用当前函数，再应用另一个函数
        Function<String, Integer> toLen = String::length;
        Function<Integer, String> multiply = i -> "长度是" + i;
        Function<String, String> combined = toLen.andThen(multiply);
        String result = combined.apply("Hello");  // "长度是5"
        
        // compose：先应用参数函数，再应用当前函数
        Function<Integer, Integer> doubleIt = x -> x * 2;
        Function<Integer, Integer> addOne = x -> x + 1;
        Function<Integer, Integer> combined2 = doubleIt.compose(addOne);
        // 先 addOne，再 doubleIt
        int r = combined2.apply(5);  // (5 + 1) * 2 = 12
        
        // ========== Predicate：条件判断 ==========
        // 接收一个值，返回 true/false
        Predicate<Integer> isEven = n -> n % 2 == 0;
        boolean even = isEven.test(4);  // true
        
        // 组合谓词
        Predicate<Integer> isPositive = n -> n > 0;
        Predicate<Integer> isEvenAndPositive = isEven.and(isPositive);
        boolean r2 = isEvenAndPositive.test(4);  // true
        
        Predicate<Integer> isEvenOrPositive = isEven.or(isPositive);
        boolean r3 = isEvenOrPositive.test(-3);  // false
        
        Predicate<Integer> isNotEven = isEven.negate();  // 取反
        boolean r4 = isNotEven.test(3);  // true
        
        // ========== BiFunction：两个参数的函数 ==========
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        int sum = add.apply(1, 2);  // 3
        
        // ========== 基本类型特化接口 ==========
        // 避免自动装箱的性能开销
        IntFunction<String> intToStr = i -> "数字：" + i;
        String s2 = intToStr.apply(100);  // "数字：100"
        
        IntPredicate isGreaterThan10 = i -> i > 10;
        boolean r5 = isGreaterThan10.test(15);  // true
        
        IntSupplier randomInt = () -> (int)(Math.random() * 100);
        int random = randomInt.getAsInt();  // 随机整数
    }
}
```

### 方法引用

方法引用是 Lambda 的简写形式，用 `::` 操作符表示。

```java
import java.util.function.*;
import java.util.*;

public class MethodReferenceDemo {
    public static void main(String[] args) {
        // ========== 静态方法引用：类名::静态方法 ==========
        // Lambda：str -> Integer.parseInt(str)
        Function<String, Integer> parser = Integer::parseInt;
        int num = parser.apply("123");  // 123
        
        // Lambda：str -> Math.sqrt(str.length())
        Function<String, Double> sqrtLength = str -> Math.sqrt(str.length());
        // 或者用方法引用组合
        Function<String, Double> sqrtLength2 = String::length;
        
        // ========== 实例方法引用：对象::实例方法 ==========
        String str = "Hello";
        // Lambda：() -> str.length()
        Supplier<Integer> lengthSupplier = str::length;
        int len = lengthSupplier.get();  // 5
        
        // Lambda：s -> System.out.println(s)
        Consumer<String> printer = System.out::println;
        printer.accept("Hello");  // 打印 Hello
        
        // ========== 类方法引用：类名::实例方法 ==========
        // 第一个参数作为方法的调用者
        // Lambda：s -> s.length()
        Function<String, Integer> toLength = String::length;
        int len2 = toLength.apply("Hello");  // 5
        
        // Lambda：(s1, s2) -> s1.compareTo(s2)
        BiFunction<String, String, Integer> comparator = String::compareTo;
        int cmp = comparator.apply("A", "B");  // -1
        
        // ========== 构造方法引用：类名::new ==========
        // 无参构造
        // Lambda：() -> new ArrayList<String>()
        Supplier<List<String>> listSupplier = ArrayList::new;
        List<String> list = listSupplier.get();
        
        // 有参构造
        // Lambda：capacity -> new ArrayList<String>(capacity)
        Function<Integer, List<String>> listCreator = ArrayList::new;
        List<String> list2 = listCreator.apply(10);
        
        // ========== 数组构造引用 ==========
        // Lambda：size -> new int[size]
        IntFunction<int[]> arrayCreator = int[]::new;
        int[] arr = arrayCreator.apply(5);  // new int[5]
        
        // ========== 方法引用对照表 ==========
        /*
        Lambda 表达式              方法引用
        ────────────────────────────────────────────
        s -> s.length()            String::length
        s -> s.toUpperCase()       String::toUpperCase
        s -> Integer.parseInt(s)   Integer::parseInt
        s -> System.out.println(s) System.out::println
        () -> new ArrayList<>()    ArrayList::new
        (a, b) -> a + b            Integer::sum
        (s1, s2) -> s1.equals(s2)  String::equals
        */
    }
}
```

---

## Stream API

📌 **Stream** 是 Java 8 引入的流式处理 API，支持对集合进行函数式操作。

### 什么是 Stream？

```java
// Stream 是什么？
// - 不是数据结构，不存储数据
// - 不是集合，不会修改原数据
// - 是数据管道，从源处理数据
// - 支持串行和并行处理

// Stream vs Collection
// Collection：存储数据，数据在内存中
// Stream：处理数据，数据在管道中流动

// Stream 的特点
// 1. 声明式：描述做什么，而不是怎么做
// 2. 可复合：操作可以链式调用
// 3. 可并行：可以利用多核处理器
// 4. 惰性求值：中间操作不会立即执行
// 5. 只能消费一次：一个 Stream 只能使用一次
```

### 创建 Stream

```java
import java.util.stream.*;
import java.util.*;
import java.nio.file.*;

public class CreateStreamDemo {
    public static void main(String[] args) throws IOException {
        // ========== 从集合创建 ==========
        List<String> list = Arrays.asList("a", "b", "c");
        
        // 串行流
        Stream<String> stream = list.stream();
        
        // 并行流（多线程处理）
        Stream<String> parallelStream = list.parallelStream();
        
        // ========== 从数组创建 ==========
        String[] arr = {"a", "b", "c"};
        Stream<String> stream2 = Arrays.stream(arr);
        
        // ========== 使用 Stream.of ==========
        Stream<String> stream3 = Stream.of("a", "b", "c");
        Stream<Integer> stream4 = Stream.of(1, 2, 3);
        
        // ========== 无限流 ==========
        // generate：无限生成
        Stream<Double> randoms = Stream.generate(Math::random);
        // randoms.forEach(System.out::println);  // 无限打印随机数
        
        // iterate：无限迭代
        // iterate(初始值, 下一个值的函数)
        Stream<Integer> naturals = Stream.iterate(0, n -> n + 1);
        // naturals.forEach(System.out::println);  // 0, 1, 2, 3, ...
        
        // 限制无限流
        Stream<Integer> first10 = Stream.iterate(0, n -> n + 1)
            .limit(10);  // 只取前10个
        
        // Java 9+ 有限迭代
        // iterate(初始值, 终止条件, 下一个值的函数)
        Stream<Integer> limited = Stream.iterate(0, n -> n < 10, n -> n + 1);
        
        // ========== 基本类型流 ==========
        // 避免装箱拆箱的性能开销
        IntStream intStream = IntStream.range(1, 10);      // [1, 10)
        IntStream intStream2 = IntStream.rangeClosed(1, 10); // [1, 10]
        
        LongStream longStream = LongStream.range(1, 100);
        
        DoubleStream doubleStream = DoubleStream.of(1.0, 2.0, 3.0);
        
        // ========== 其他方式 ==========
        // 从文件创建
        Stream<String> lines = Files.lines(Paths.get("file.txt"));
        
        // 从字符串创建
        IntStream chars = "Hello".chars();  // 字符的 ASCII 码流
        
        // 创建空流
        Stream<String> empty = Stream.empty();
    }
}
```

### Stream 操作分类

```
Stream 操作分为两类：

1. 中间操作（Intermediate Operations）
   - 返回新的 Stream
   - 惰性求值（不会立即执行）
   - 可以链式调用
   - 如：filter, map, sorted, distinct, limit, skip

2. 终端操作（Terminal Operations）
   - 返回结果或副作用
   - 触发整个流的执行
   - 一个流只能有一个终端操作
   - 如：forEach, collect, reduce, count, min, max
```

### 中间操作

```java
import java.util.stream.*;
import java.util.*;
import java.util.function.*;

public class IntermediateOperationsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // ========== filter：过滤 ==========
        // 保留满足条件的元素
        // filter(Predicate)
        
        // 过滤偶数
        Stream<Integer> evens = numbers.stream()
            .filter(n -> n % 2 == 0);
        // 此时还没有执行！惰性求值
        
        // 必须有终端操作才会执行
        evens.forEach(System.out::println);  // 2, 4, 6, 8, 10
        
        // 多个 filter 可以链式调用
        Stream<Integer> result = numbers.stream()
            .filter(n -> n % 2 == 0)    // 偶数
            .filter(n -> n > 5);        // 大于5
        // 结果：6, 8, 10
        
        // ========== map：映射转换 ==========
        // 将每个元素转换为新的值
        // map(Function)
        
        // 平方
        Stream<Integer> squares = numbers.stream()
            .map(n -> n * n);
        // 结果：1, 4, 9, 16, 25, 36, 49, 64, 81, 100
        
        // 转字符串
        Stream<String> strings = numbers.stream()
            .map(Object::toString);
        // 结果："1", "2", "3", ...
        
        // ========== flatMap：扁平化映射 ==========
        // 将每个元素转换为一个流，然后合并所有流
        // flatMap(Function<T, Stream<R>>)
        
        List<List<Integer>> nested = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4),
            Arrays.asList(5, 6)
        );
        
        // 使用 map 会得到 Stream<Stream<Integer>>
        Stream<Stream<Integer>> streamOfStreams = nested.stream()
            .map(List::stream);
        
        // 使用 flatMap 得到 Stream<Integer>
        Stream<Integer> flattened = nested.stream()
            .flatMap(List::stream);
        // 结果：1, 2, 3, 4, 5, 6
        
        // 实际应用：拆分单词
        List<String> sentences = Arrays.asList("Hello World", "Java Stream");
        Stream<String> words = sentences.stream()
            .flatMap(s -> Arrays.stream(s.split(" ")));
        // 结果：Hello, World, Java, Stream
        
        // ========== distinct：去重 ==========
        Stream<Integer> distinct = Stream.of(1, 2, 2, 3, 3, 3, 4)
            .distinct();
        // 结果：1, 2, 3, 4
        
        // ========== sorted：排序 ==========
        // 默认自然排序
        Stream<Integer> sorted = Stream.of(5, 2, 8, 1, 9)
            .sorted();
        // 结果：1, 2, 5, 8, 9
        
        // 自定义排序
        Stream<Integer> sortedDesc = Stream.of(5, 2, 8, 1, 9)
            .sorted((a, b) -> b - a);  // 降序
        // 结果：9, 8, 5, 2, 1
        
        // 使用 Comparator
        Stream<Integer> sortedDesc2 = Stream.of(5, 2, 8, 1, 9)
            .sorted(Comparator.reverseOrder());
        
        // 对象排序
        List<Person> people = Arrays.asList(
            new Person("张三", 25),
            new Person("李四", 30),
            new Person("王五", 20)
        );
        Stream<Person> sortedByAge = people.stream()
            .sorted(Comparator.comparing(Person::getAge));
        // 按年龄升序：王五(20), 张三(25), 李四(30)
        
        // ========== limit：限制数量 ==========
        Stream<Integer> first5 = numbers.stream()
            .limit(5);
        // 结果：1, 2, 3, 4, 5
        
        // ========== skip：跳过元素 ==========
        Stream<Integer> skip5 = numbers.stream()
            .skip(5);
        // 结果：6, 7, 8, 9, 10
        
        // 分页效果
        int page = 2, size = 3;
        Stream<Integer> page2 = numbers.stream()
            .skip((page - 1) * size)  // 跳过前 3 个
            .limit(size);              // 取 3 个
        // 结果：4, 5, 6
        
        // ========== peek：查看元素 ==========
        // 主要用于调试，不影响流的内容
        List<Integer> result2 = numbers.stream()
            .peek(n -> System.out.println("处理前：" + n))
            .filter(n -> n % 2 == 0)
            .peek(n -> System.out.println("处理后：" + n))
            .collect(Collectors.toList());
        
        // ========== takeWhile：获取元素直到条件不满足（Java 9+）==========
        Stream<Integer> takeWhile = Stream.of(1, 2, 3, 4, 3, 2, 1)
            .takeWhile(n -> n < 4);
        // 结果：1, 2, 3（遇到 4 停止，后面不处理）
        
        // ========== dropWhile：丢弃元素直到条件不满足（Java 9+）==========
        Stream<Integer> dropWhile = Stream.of(1, 2, 3, 4, 3, 2, 1)
            .dropWhile(n -> n < 4);
        // 结果：4, 3, 2, 1（丢弃直到遇到 4）
    }
}

class Person {
    String name;
    int age;
    
    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    String getName() { return name; }
    int getAge() { return age; }
}
```

### 终端操作

```java
import java.util.stream.*;
import java.util.*;
import java.util.function.*;

public class TerminalOperationsDemo {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        
        // ========== forEach：遍历 ==========
        numbers.stream().forEach(System.out::println);
        // 或直接用集合的 forEach
        numbers.forEach(System.out::println);
        
        // forEachOrdered：保证顺序（并行流时有用）
        numbers.parallelStream()
            .forEachOrdered(System.out::println);
        
        // ========== collect：收集结果 ==========
        // 收集为 List
        List<Integer> list = numbers.stream()
            .collect(Collectors.toList());
        
        // 收集为 Set
        Set<Integer> set = numbers.stream()
            .collect(Collectors.toSet());
        
        // 收集为指定集合
        LinkedList<Integer> linkedList = numbers.stream()
            .collect(Collectors.toCollection(LinkedList::new));
        
        // 收集为 Map
        Map<String, Integer> nameToAge = people.stream()
            .collect(Collectors.toMap(
                Person::getName,  // key 映射函数
                Person::getAge,   // value 映射函数
                (v1, v2) -> v1    // key 冲突时的合并函数
            ));
        
        // ========== toArray：转为数组 ==========
        Integer[] arr = numbers.stream()
            .toArray(Integer[]::new);
        
        // ========== reduce：归约 ==========
        // 将流中的元素组合成一个值
        
        // 求和
        // reduce(初始值, 累加函数)
        int sum = numbers.stream()
            .reduce(0, (a, b) -> a + b);
        // 或使用方法引用
        int sum2 = numbers.stream()
            .reduce(0, Integer::sum);
        
        // 无初始值（返回 Optional）
        Optional<Integer> sum3 = numbers.stream()
            .reduce(Integer::sum);
        
        // 求最大值
        Optional<Integer> max = numbers.stream()
            .reduce(Integer::max);
        
        // 求最小值
        Optional<Integer> min = numbers.stream()
            .reduce(Integer::min);
        
        // ========== count：计数 ==========
        long count = numbers.stream()
            .filter(n -> n > 2)
            .count();  // 3
        
        // ========== min/max：最小/最大值 ==========
        Optional<Integer> minVal = numbers.stream()
            .min(Integer::compare);
        System.out.println(minVal.get());  // 1
        
        Optional<Integer> maxVal = numbers.stream()
            .max(Integer::compare);
        System.out.println(maxVal.get());  // 5
        
        // ========== findFirst/findAny：查找 ==========
        // findFirst：返回第一个元素
        Optional<Integer> first = numbers.stream()
            .filter(n -> n > 2)
            .findFirst();  // 3（总是返回第一个匹配的）
        
        // findAny：返回任意一个元素（并行流时可能更快）
        Optional<Integer> any = numbers.stream()
            .filter(n -> n > 2)
            .findAny();  // 可能是 3, 4, 5 中的任意一个
        
        // ========== anyMatch/allMatch/noneMatch：匹配 ==========
        // anyMatch：是否有任意元素满足条件
        boolean hasEven = numbers.stream()
            .anyMatch(n -> n % 2 == 0);  // true
        
        // allMatch：是否所有元素都满足条件
        boolean allPositive = numbers.stream()
            .allMatch(n -> n > 0);  // true
        
        // noneMatch：是否没有元素满足条件
        boolean noneNegative = numbers.stream()
            .noneMatch(n -> n < 0);  // true
    }
}
```

---

## Collectors 工具类

```java
import java.util.stream.*;
import java.util.*;
import java.util.function.*;

public class CollectorsDemo {
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("张三", 25, "北京"),
            new Person("李四", 30, "上海"),
            new Person("王五", 25, "北京"),
            new Person("赵六", 35, "广州")
        );
        
        // ========== 转集合 ==========
        List<String> names = people.stream()
            .map(Person::getName)
            .collect(Collectors.toList());
        
        Set<Integer> ages = people.stream()
            .map(Person::getAge)
            .collect(Collectors.toSet());
        
        // ========== 转 Map ==========
        // 名字 -> 年龄
        Map<String, Integer> nameToAge = people.stream()
            .collect(Collectors.toMap(
                Person::getName,   // key
                Person::getAge,    // value
                (v1, v2) -> v1     // key 冲突时保留第一个
            ));
        
        // ========== 分组 ==========
        // 按年龄分组
        Map<Integer, List<Person>> byAge = people.stream()
            .collect(Collectors.groupingBy(Person::getAge));
        // {25=[张三, 王五], 30=[李四], 35=[赵六]}
        
        // 按城市分组
        Map<String, List<Person>> byCity = people.stream()
            .collect(Collectors.groupingBy(Person::getCity));
        
        // 多级分组
        Map<String, Map<Integer, List<Person>>> byCityAndAge = people.stream()
            .collect(Collectors.groupingBy(
                Person::getCity,
                Collectors.groupingBy(Person::getAge)
            ));
        
        // 分组后统计数量
        Map<Integer, Long> countByAge = people.stream()
            .collect(Collectors.groupingBy(
                Person::getAge,
                Collectors.counting()
            ));
        // {25=2, 30=1, 35=1}
        
        // 分组后求平均值
        Map<String, Double> avgAgeByCity = people.stream()
            .collect(Collectors.groupingBy(
                Person::getCity,
                Collectors.averagingInt(Person::getAge)
            ));
        
        // ========== 分区 ==========
        // 按条件分为两组（true 和 false）
        Map<Boolean, List<Person>> partition = people.stream()
            .collect(Collectors.partitioningBy(p -> p.getAge() >= 30));
        // {false=[张三, 王五], true=[李四, 赵六]}
        
        // ========== 统计 ==========
        IntSummaryStatistics stats = people.stream()
            .collect(Collectors.summarizingInt(Person::getAge));
        stats.getCount();    // 4
        stats.getSum();      // 115
        stats.getAverage();  // 28.75
        stats.getMin();      // 25
        stats.getMax();      // 35
        
        // ========== 拼接字符串 ==========
        String nameStr = people.stream()
            .map(Person::getName)
            .collect(Collectors.joining(", "));
        // "张三, 李四, 王五, 赵六"
        
        String nameStr2 = people.stream()
            .map(Person::getName)
            .collect(Collectors.joining(", ", "[", "]"));
        // "[张三, 李四, 王五, 赵六]"
        
        // ========== 归约 ==========
        Integer totalAge = people.stream()
            .collect(Collectors.reducing(
                0,
                Person::getAge,
                Integer::sum
            ));
    }
}
```

---

## Optional 类

📌 **Optional** 是一个容器类，用于避免空指针异常。

### 为什么需要 Optional？

```java
// 传统方式的空指针问题
public String getCity(User user) {
    if (user != null) {
        Address address = user.getAddress();
        if (address != null) {
            return address.getCity();
        }
    }
    return "未知";
}

// 使用 Optional
public String getCity(User user) {
    return Optional.ofNullable(user)
        .map(User::getAddress)
        .map(Address::getCity)
        .orElse("未知");
}
```

### Optional 详解

```java
import java.util.Optional;

public class OptionalDemo {
    public static void main(String[] args) {
        // ========== 创建 Optional ==========
        // Optional.of(value)：value 不能为 null
        Optional<String> opt1 = Optional.of("Hello");
        
        // Optional.ofNullable(value)：value 可以为 null
        Optional<String> opt2 = Optional.ofNullable("Hello");
        Optional<String> opt3 = Optional.ofNullable(null);
        
        // Optional.empty()：空 Optional
        Optional<String> opt4 = Optional.empty();
        
        // ========== 判断是否有值 ==========
        Optional<String> opt = Optional.of("Hello");
        
        boolean present = opt.isPresent();   // true
        boolean empty = opt.isEmpty();       // false（Java 11+）
        
        // ========== 获取值 ==========
        // get()：有值返回，无值抛 NoSuchElementException
        String value = opt.get();  // "Hello"
        
        // ⚠️ 不推荐直接用 get()，因为可能抛异常
        // 推荐使用以下安全的方法：
        
        // ========== 安全获取值 ==========
        // orElse：有值返回值，无值返回默认值
        String val1 = opt.orElse("默认值");       // "Hello"
        String val2 = opt3.orElse("默认值");       // "默认值"
        
        // orElseGet：有值返回值，无值调用 Supplier 获取默认值
        // 懒加载，只有需要时才调用
        String val3 = opt.orElseGet(() -> "默认值");
        
        // orElseThrow：无值时抛出指定异常
        String val4 = opt.orElseThrow(() -> 
            new IllegalArgumentException("值为空"));
        
        // ========== 消费值 ==========
        // ifPresent：有值时执行 Consumer
        opt.ifPresent(v -> System.out.println(v));
        opt.ifPresent(System.out::println);
        
        // ifPresentOrElse：有值执行 Consumer，无值执行 Runnable（Java 9+）
        opt.ifPresentOrElse(
            v -> System.out.println("值：" + v),
            () -> System.out.println("无值")
        );
        
        // ========== 转换 ==========
        // map：转换值
        Optional<Integer> length = opt.map(String::length);  // Optional[5]
        
        // filter：过滤值
        Optional<String> filtered = opt.filter(s -> s.length() > 3);  // Optional[Hello]
        
        // flatMap：扁平化转换（转换函数返回 Optional）
        Optional<String> upper = opt.flatMap(s -> Optional.of(s.toUpperCase()));
        
        // ========== 实际应用 ==========
        // 链式调用避免空指针
        String city = Optional.ofNullable(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .orElse("未知");
        
        // 方法返回 Optional
        public Optional<User> findById(Long id) {
            User user = userRepository.findById(id);
            return Optional.ofNullable(user);
        }
        
        // 调用
        Optional<User> user = userService.findById(1L);
        String name = user.map(User::getName).orElse("未知");
    }
}
```

---

## 实战示例

### 数据处理

```java
import java.util.*;
import java.util.stream.*;

public class StreamPractice {
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("张三", 25, "北京"),
            new Person("李四", 30, "上海"),
            new Person("王五", 25, "北京"),
            new Person("赵六", 35, "广州"),
            new Person("钱七", 28, "上海")
        );
        
        // ========== 找出北京的用户名 ==========
        List<String> beijingNames = people.stream()
            .filter(p -> "北京".equals(p.getCity()))
            .map(Person::getName)
            .collect(Collectors.toList());
        System.out.println(beijingNames);  // [张三, 王五]
        
        // ========== 按城市分组 ==========
        Map<String, List<Person>> byCity = people.stream()
            .collect(Collectors.groupingBy(Person::getCity));
        
        // ========== 计算平均年龄 ==========
        double avgAge = people.stream()
            .mapToInt(Person::getAge)
            .average()
            .orElse(0);
        System.out.println(avgAge);  // 28.6
        
        // ========== 找出年龄最大的人 ==========
        Optional<Person> oldest = people.stream()
            .max(Comparator.comparing(Person::getAge));
        oldest.ifPresent(p -> System.out.println(p.getName()));  // 赵六
        
        // ========== 检查是否有人在北京 ==========
        boolean hasBeijing = people.stream()
            .anyMatch(p -> "北京".equals(p.getCity()));
        System.out.println(hasBeijing);  // true
        
        // ========== 按年龄排序 ==========
        List<Person> sortedByAge = people.stream()
            .sorted(Comparator.comparing(Person::getAge))
            .collect(Collectors.toList());
        
        // ========== 年龄去重 ==========
        Set<Integer> uniqueAges = people.stream()
            .map(Person::getAge)
            .collect(Collectors.toSet());
        
        // ========== 每个城市的平均年龄 ==========
        Map<String, Double> avgAgeByCity = people.stream()
            .collect(Collectors.groupingBy(
                Person::getCity,
                Collectors.averagingInt(Person::getAge)
            ));
        
        // ========== 年龄大于 25 的人名，用逗号连接 ==========
        String names = people.stream()
            .filter(p -> p.getAge() > 25)
            .map(Person::getName)
            .collect(Collectors.joining(", "));
        System.out.println(names);  // 李四, 赵六, 钱七
    }
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| **Lambda** | 简洁的匿名函数语法 |
| **函数式接口** | 只有一个抽象方法的接口 |
| **方法引用** | 用 `::` 引用现有方法 |
| **Stream** | 流式处理集合数据 |
| **中间操作** | 返回新 Stream，惰性求值 |
| **终端操作** | 产生结果，触发执行 |
| **Optional** | 避免 NullPointerException |
| **Collectors** | 收集 Stream 结果的工具类 |

### 最佳实践

1. **优先使用 Lambda 和 Stream**：代码更简洁、可读性更好
2. **使用方法引用**：比 Lambda 更简洁
3. **注意 Stream 的惰性求值**：中间操作不会立即执行
4. **使用 Optional 处理空值**：避免 NullPointerException
5. **合理使用并行流**：数据量大时考虑 `parallelStream()`