# Java 基础语法

> Java 是一门强类型语言，每个变量必须声明类型。本章节涵盖变量、数据类型、运算符和控制流语句，是 Java 编程的基石。

## 变量与数据类型

### 变量声明

```java
// 声明并初始化
int age = 25;
String name = "张三";

// 先声明后赋值
double price;
price = 99.9;

// 多变量声明
int x = 1, y = 2, z = 3;
```

### 命名规则

- 只能包含字母、数字、下划线 `_` 和美元符号 `$`
- 不能以数字开头
- 不能使用 Java 关键字
- 区分大小写
- 推荐使用驼峰命名法（camelCase）

```java
// ✅ 合法命名
int studentAge;
String userName;
double _price;
float $value;

// ❌ 非法命名
int 123abc;      // 不能以数字开头
int class;       // 不能使用关键字
int my-var;      // 不能包含连字符
```

### 变量作用域

```java
public class VariableScope {
    // 类变量（静态变量）- 类加载时存在
    static int classVar = 100;
    
    // 实例变量 - 对象创建时存在
    String instanceVar = "实例变量";
    
    public void method() {
        // 局部变量 - 方法执行时存在
        int localVar = 10;
    }
}
```

---

## 基本数据类型

Java 有 8 种基本数据类型（Primitive Types）。

### 整数类型

| 类型 | 字节数 | 取值范围 | 默认值 |
|------|:------:|----------|:------:|
| `byte` | 1 | $-128 \sim 127$ | 0 |
| `short` | 2 | $-32,768 \sim 32,767$ | 0 |
| `int` | 4 | $-2^{31} \sim 2^{31}-1$ | 0 |
| `long` | 8 | $-2^{63} \sim 2^{63}-1$ | 0L |

```java
byte b = 100;
short s = 10000;
int i = 100000;
long l = 10000000000L;  // long 需要加 L 后缀

// 数字字面量增强（Java 7+）
int million = 1_000_000;         // 下划线分隔，更易读
int binary = 0b1010;             // 二进制
int octal = 012;                 // 八进制
int hex = 0xFF;                  // 十六进制
```

### 浮点类型

| 类型 | 字节数 | 精度 | 默认值 |
|------|:------:|------|:------:|
| `float` | 4 | 6-7 位有效数字 | 0.0f |
| `double` | 8 | 15-16 位有效数字 | 0.0d |

```java
float f = 3.14f;      // float 需要加 f 后缀
double d = 3.14;      // 默认是 double 类型

// 科学计数法
double scientific = 1.5e10;    // 1.5 × 10^10
double tiny = 1.5e-10;         // 1.5 × 10^-10
```

### 字符与布尔类型

```java
// 字符类型（2字节，Unicode编码）
char c1 = 'A';           // 直接字符
char c2 = 65;            // Unicode 编码值
char c3 = '\u0041';      // Unicode 转义
char chinese = '中';      // 中文字符

// 转义字符
char newline = '\n';     // 换行
char tab = '\t';         // 制表符
char backslash = '\\';   // 反斜杠

// 布尔类型
boolean isTrue = true;
boolean isFalse = false;
```

## 类型转换

### 自动类型转换（隐式）

从小类型到大类型自动转换：

$$\text{byte} \to \text{short} \to \text{int} \to \text{long} \to \text{float} \to \text{double}$$

```java
int i = 100;
long l = i;        // int → long 自动转换
double d = l;      // long → double 自动转换

// 表达式中的自动提升
byte b1 = 10;
byte b2 = 20;
// byte b3 = b1 + b2;  // ❌ 编译错误！结果为 int
int result = b1 + b2;   // ✅ 正确
```

### 强制类型转换（显式）

```java
double d = 3.99;
int i = (int) d;     // 强制转换，结果为 3（截断小数）

// 溢出风险
int big = 300;
byte small = (byte) big;  // 结果为 44（溢出）
```

### 包装类与字符串转换

```java
// 自动装箱与拆箱
Integer boxed = 100;           // int → Integer
int unboxed = boxed;           // Integer → int

// 字符串与数值转换
int num = Integer.parseInt("123");      // 字符串 → int
String str = String.valueOf(123);       // int → 字符串
double d = Double.parseDouble("3.14");  // 字符串 → double
```

---

## 运算符

### 算术运算符

```java
int a = 10, b = 3;

int sum = a + b;     // 13（加法）
int diff = a - b;    // 7（减法）
int prod = a * b;    // 30（乘法）
int quot = a / b;    // 3（整数除法，截断）
int rem = a % b;     // 1（取余）

// 自增自减
int i = 5;
int x = i++;    // x=5, i=6（先使用，后自增）
int y = ++i;    // y=7, i=7（先自增，后使用）
```

### 关系与逻辑运算符

```java
// 关系运算符
boolean eq = (a == b);   // 等于
boolean ne = (a != b);   // 不等于
boolean gt = (a > b);    // 大于
boolean lt = (a < b);    // 小于

// 逻辑运算符（短路求值）
boolean p = true, q = false;
boolean and = p && q;    // 与（两边都为真才为真）
boolean or = p || q;     // 或（一边为真即为真）
boolean not = !p;        // 非（取反）
```

### 位运算符

```java
int a = 5;    // 二进制：0101
int b = 3;    // 二进制：0011

int and = a & b;    // 1  （按位与：0001）
int or = a | b;     // 7  （按位或：0111）
int xor = a ^ b;    // 6  （按位异或：0110）
int not = ~a;       // -6 （按位取反）
int left = a << 1;  // 10 （左移：1010）
int right = a >> 1; // 2  （右移：0010）
```

### 三元运算符

```java
int age = 18;
String status = (age >= 18) ? "成年" : "未成年";

// 嵌套使用
int score = 85;
String grade = (score >= 90) ? "A" : 
               (score >= 80) ? "B" : 
               (score >= 60) ? "C" : "D";
```

---

## 控制流语句

### 条件语句

```java
int score = 85;

// if-else
if (score >= 90) {
    System.out.println("优秀");
} else if (score >= 60) {
    System.out.println("及格");
} else {
    System.out.println("不及格");
}

// switch 语句
int day = 3;
switch (day) {
    case 1: System.out.println("星期一"); break;
    case 2: System.out.println("星期二"); break;
    default: System.out.println("其他");
}

// switch 表达式（Java 14+）
String result = switch (day) {
    case 1, 2, 3, 4, 5 -> "工作日";
    case 6, 7 -> "周末";
    default -> "无效";
};
```

### 循环语句

```java
// for 循环
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// 增强 for 循环（for-each）
int[] numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
    System.out.println(num);
}

// while 循环
int i = 0;
while (i < 5) {
    System.out.println(i);
    i++;
}

// do-while 循环（至少执行一次）
int j = 0;
do {
    System.out.println(j);
    j++;
} while (j < 5);
```

### 跳转语句

```java
// break - 跳出循环
for (int i = 0; i < 10; i++) {
    if (i == 5) break;
    System.out.println(i);  // 输出 0-4
}

// continue - 跳过本次迭代
for (int i = 0; i < 5; i++) {
    if (i == 2) continue;
    System.out.println(i);  // 输出 0, 1, 3, 4
}

// 带标签的 break（跳出多层循环）
outer:
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        if (i == 1 && j == 1) break outer;
    }
}
```

---

## 数组

### 数组声明与初始化

```java
// 声明
int[] arr1;
int arr2[];    // 不推荐

// 静态初始化
int[] arr3 = {1, 2, 3, 4, 5};

// 动态初始化
int[] arr4 = new int[5];       // 默认值为 0
String[] arr5 = new String[3]; // 默认值为 null

// 访问与修改
arr4[0] = 10;
int first = arr4[0];
int len = arr4.length;
```

### 数组操作

```java
int[] arr = {5, 2, 8, 1, 9};

// 排序
Arrays.sort(arr);  // [1, 2, 5, 8, 9]

// 二分查找（要求数组有序）
int index = Arrays.binarySearch(arr, 5);

// 转字符串
String str = Arrays.toString(arr);  // "[1, 2, 5, 8, 9]"

// 复制
int[] copy = Arrays.copyOf(arr, 3);  // 复制前3个元素
```

### 多维数组

```java
// 二维数组
int[][] matrix = {
    {1, 2, 3},
    {4, 5, 6}
};

// 访问
int val = matrix[0][1];  // 2

// 动态初始化
int[][] arr = new int[2][3];
```

---

## 输入输出

### 控制台输出

```java
System.out.println("Hello");      // 输出并换行
System.out.print("Hello");        // 输出不换行
System.out.printf("姓名：%s，年龄：%d%n", "张三", 25);  // 格式化输出
```

### 控制台输入

```java
import java.util.Scanner;

Scanner scanner = new Scanner(System.in);
System.out.print("请输入姓名：");
String name = scanner.nextLine();

System.out.print("请输入年龄：");
int age = scanner.nextInt();

scanner.close();
```

---

## var 类型推断（Java 10+）

```java
var name = "张三";           // 推断为 String
var age = 25;               // 推断为 int
var list = new ArrayList<String>();  // 推断为 ArrayList<String>

// 注意：var 只能用于局部变量，必须有初始化值
// var x;  // ❌ 编译错误
// var y = null;  // ❌ 编译错误
```

---

## 字符串

### 字符串创建

```java
// 直接创建
String s1 = "Hello";

// 通过构造方法
String s2 = new String("Hello");

// 字符串常量池
String s3 = "Hello";
System.out.println(s1 == s3);   // true（指向同一对象）
System.out.println(s1 == s2);   // false（不同对象）
```

### 常用方法

```java
String s = "Hello, World!";

// 长度与判断
int len = s.length();              // 13
boolean empty = s.isEmpty();       // false

// 访问字符
char c = s.charAt(0);              // 'H'

// 查找
int idx = s.indexOf("World");      // 7
int lastIdx = s.lastIndexOf("o");  // 8

// 截取
String sub = s.substring(0, 5);    // "Hello"

// 分割
String[] parts = "a,b,c".split(",");  // ["a", "b", "c"]

// 大小写转换
String upper = s.toUpperCase();    // "HELLO, WORLD!"
String lower = s.toLowerCase();    // "hello, world!"

// 去除空格
String trimmed = "  hello  ".trim();  // "hello"

// 替换
String replaced = s.replace("World", "Java");

// 拼接
String joined = String.join("-", "a", "b", "c");  // "a-b-c"
```

### StringBuilder

📌 对于频繁的字符串拼接操作，使用 `StringBuilder` 更高效。

```java
StringBuilder sb = new StringBuilder();

sb.append("Hello");
sb.append(", ");
sb.append("World");
sb.insert(5, " Java");

String result = sb.toString();  // "Hello Java, World"
```

---

## 小结

| 概念 | 要点 |
|------|------|
| **基本类型** | 8 种：byte, short, int, long, float, double, char, boolean |
| **类型转换** | 自动（小→大）、强制（大→小，可能丢失精度） |
| **运算符** | 算术、关系、逻辑、位运算、三元 |
| **控制流** | if-else, switch, for, while, do-while, break, continue |
| **数组** | 固定长度，索引从 0 开始 |
| **字符串** | 不可变，使用 StringBuilder 进行高效拼接 |
