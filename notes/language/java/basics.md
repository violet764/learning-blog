# Java 基础语法

> Java 是一门强类型语言，每个变量必须声明类型。本章节涵盖变量、数据类型、运算符和控制流语句，是 Java 编程的基石。


在学习具体语法之前，让我们先理解几个核心概念：

### Java 程序是如何运行的？

```
源代码(.java文件) 
    ↓ 编译器(javac)
字节码(.class文件) 
    ↓ Java虚拟机(JVM)
机器码 → 在操作系统上运行
```

**关键理解**：
- **编译**：Java 源代码需要先编译成字节码，这与其他解释型语言（如 Python）不同
- **JVM（Java虚拟机）**：是 Java 跨平台的关键，不同操作系统有不同版本的 JVM，但它们都能运行相同的字节码
- **一次编写，到处运行**：你写的 Java 代码可以在 Windows、Linux、Mac 上运行，无需修改

### 为什么 Java 是"强类型"语言？

强类型意味着：
1. 每个变量必须先声明类型，才能使用
2. 编译器会在编译时检查类型是否匹配
3. 类型不匹配会直接报错，而不是在运行时才发现问题

```java
// 强类型的例子
int age = 25;           // 必须声明 age 是 int 类型
// age = "二十五";      // 编译错误！不能把字符串赋给整数类型

// 对比弱类型语言（如 JavaScript）
// let age = 25;
// age = "二十五";       // JavaScript 允许这样做，但可能导致后续计算出错
```

**好处**：在编译阶段就能发现很多类型相关的错误，避免在运行时崩溃。

---

## 变量与数据类型

### 变量声明

```java
// 声明并初始化（推荐方式）
// 语法：类型 变量名 = 初始值;
int age = 25;           // 声明一个整数变量 age，值为 25
String name = "张三";   // 声明一个字符串变量 name，值为 "张三"
                        // String 是类类型，首字母大写

// 先声明后赋值（不推荐，容易忘记初始化）
double price;           // 只是声明，变量还没有值
price = 99.9;           // 后续赋值

// 多变量声明（相同类型）
int x = 1, y = 2, z = 3;  // 一次声明多个同类型变量，用逗号分隔
```

### 命名规则

Java 的命名规则是编译器强制要求的，违反会导致编译错误。

- 只能包含字母、数字、下划线 `_` 和美元符号 `$`
- 不能以数字开头
- 不能使用 Java 关键字（如 `class`、`int`、`public` 等）
- 区分大小写（`age` 和 `Age` 是两个不同的变量）
- 推荐使用驼峰命名法（camelCase）：第一个单词小写，后续单词首字母大写

```java
// ✅ 合法命名
int studentAge;         // 驼峰命名法，推荐
String userName;        // 驼峰命名法
double _price;          // 下划线开头（通常用于特殊变量）
float $value;           // 美元符号开头（不推荐，但在库代码中常见）

// ❌ 非法命名（会导致编译错误）
int 123abc;             // 不能以数字开头
int class;              // 不能使用关键字
int my-var;             // 不能包含连字符（连字符会被误认为是减号）
```

### 变量作用域

作用域决定了变量在哪里可以被访问。理解作用域对于避免命名冲突和内存管理很重要。

```java
public class VariableScope {
    // 类变量（静态变量）- 使用 static 修饰
    // 特点：类加载时就存在，所有对象共享同一个值
    // 内存位置：方法区
    static int classVar = 100;
    
    // 实例变量（成员变量）- 不使用 static 修饰
    // 特点：创建对象时才存在，每个对象有独立的值
    // 内存位置：堆内存
    String instanceVar = "实例变量";
    
    public void method() {
        // 局部变量 - 在方法内部声明
        // 特点：方法执行时创建，方法结束时销毁
        // 内存位置：栈内存
        // ⚠️ 局部变量没有默认值，必须手动初始化后才能使用
        int localVar = 10;
        
        // 作用域示例
        if (true) {
            // 块级作用域：只在 if 块内有效
            int blockVar = 20;
            System.out.println(blockVar);  // ✅ 可以访问
        }
        // System.out.println(blockVar);   // ❌ 编译错误！超出作用域
    }
}
```

**三种变量的对比**：

| 类型 | 关键字 | 生命周期 | 内存位置 | 默认值 | 访问方式 |
|------|--------|----------|----------|--------|----------|
| 类变量 | static | 类加载到类卸载 | 方法区 | 有默认值 | 类名.变量名 |
| 实例变量 | 无 | 对象创建到垃圾回收 | 堆 | 有默认值 | 对象.变量名 |
| 局部变量 | 无 | 方法执行期间 | 栈 | 无默认值 | 直接使用 |

---

## 基本数据类型

Java 有 8 种基本数据类型（Primitive Types），它们不是对象，直接存储数据值。

### 为什么区分基本类型和引用类型？

**基本类型**：直接存储数据值，存储在栈内存中
**引用类型**：存储对象的内存地址，对象存储在堆内存中

```
基本类型：int a = 10;     → 栈内存直接存储 10
引用类型：String s = "hello";  → 栈存储地址，堆存储实际对象
```

### 整数类型

整数类型用于存储没有小数部分的数字。Java 提供了 4 种整数类型，区别在于占用内存大小和取值范围。

| 类型 | 字节数 | 位数 | 取值范围 | 默认值 | 使用场景 |
|------|:------:|:----:|----------|:------:|----------|
| `byte` | 1 | 8 | $-128 \sim 127$ | 0 | 节省内存、文件/网络传输 |
| `short` | 2 | 16 | $-32,768 \sim 32,767$ | 0 | 较小的整数 |
| `int` | 4 | 32 | $-2^{31} \sim 2^{31}-1$（约±21亿） | 0 | **最常用**，一般整数运算 |
| `long` | 8 | 64 | $-2^{63} \sim 2^{63}-1$ | 0L | 大整数，如时间戳、ID |

```java
// byte：1字节，范围 -128 到 127
// 适用场景：处理二进制数据、节省内存的大数组
byte b = 100;           // 在范围内，编译通过
// byte b2 = 128;       // 编译错误！超出范围（127）

// short：2字节，范围约 -3.2万 到 3.2万
short s = 10000;        // 适用于明确知道数值不会太大的情况

// int：4字节，最常用的整数类型
// 范围约 ±21亿，大多数情况下足够使用
int i = 100000;
int population = 1400000000;  // 中国人口约14亿，int 可以存储

// long：8字节，超大整数
// ⚠️ 必须加 L 或 l 后缀，否则会被当作 int 处理
long l = 10000000000L;  // 必须加 L 后缀，否则编译错误（超出 int 范围）
long timestamp = System.currentTimeMillis();  // 时间戳通常用 long

// 数字字面量增强（Java 7+）- 使用下划线分隔，提高可读性
int million = 1_000_000;         // 100万，下划线不影响值，只是更易读
int creditCard = 1234_5678_9012_3456L;  // 银行卡号格式

// 不同进制的表示
int binary = 0b1010;             // 二进制，值为 10（0b 前缀）
int octal = 012;                 // 八进制，值为 10（0 前缀，容易混淆，不推荐）
int hex = 0xFF;                  // 十六进制，值为 255（0x 前缀）
```

### 浮点类型

浮点类型用于存储带小数的数字。

| 类型 | 字节数 | 精度 | 默认值 | 说明 |
|------|:------:|------|:------:|------|
| `float` | 4 | 6-7 位有效数字 | 0.0f | 单精度，节省内存 |
| `double` | 8 | 15-16 位有效数字 | 0.0d | **双精度，默认类型** |

```java
// float：单精度浮点数，必须加 f 或 F 后缀
// 精度有限，约 6-7 位有效数字，适合对精度要求不高的场景
float f = 3.14f;        // ⚠️ 必须加 f 后缀，否则 3.14 默认是 double 类型
float price = 99.99f;

// double：双精度浮点数，Java 默认的浮点类型
// 精度约 15-16 位有效数字，推荐用于科学计算、财务计算
double d = 3.14;        // 不需要后缀，默认就是 double
double pi = 3.14159265358979;  // 可以保留更多精度

// ⚠️ 浮点数的精度问题（重要！）
System.out.println(0.1 + 0.2);  // 输出 0.30000000000000004，不是精确的 0.3
// 原因：浮点数在计算机中是近似存储的
// 解决方案：对于金额等需要精确计算的场景，使用 BigDecimal 类

// 科学计数法
double scientific = 1.5e10;     // 1.5 × 10^10 = 15000000000
double tiny = 1.5e-10;          // 1.5 × 10^-10 = 0.00000000015
```

### 字符与布尔类型

```java
// char：字符类型，2字节，使用 Unicode 编码
// 可以存储一个中文字符、英文字符、符号等
char c1 = 'A';           // 英文字符
char c2 = '中';          // 中文字符（Unicode 编码）
char c3 = 65;            // 用数字表示，65 对应字符 'A'
char c4 = '\u0041';      // Unicode 转义表示，\u0041 对应 'A'

// 常用转义字符
char newline = '\n';     // 换行符
char tab = '\t';         // 制表符（Tab 键）
char backslash = '\\';   // 反斜杠（需要转义）
char quote = '\'';       // 单引号（需要转义）
char doubleQuote = '\"'; // 双引号（需要转义）

// ⚠️ char 只能存储一个字符
// char c = "A";         // 编译错误！双引号是字符串，不是字符
// char c = 'AB';        // 编译错误！不能存储多个字符

// boolean：布尔类型，只有 true 和 false 两个值
// 用于条件判断，占用 1 位（不是 1 字节）
boolean isTrue = true;
boolean isFalse = false;
// boolean b = 1;        // 编译错误！不能用数字表示
// boolean b = 0;        // 编译错误！不能用数字表示
```

## 类型转换

### 自动类型转换（隐式）

从小类型到大类型可以自动转换，不会丢失精度。

$$\text{byte} \to \text{short} \to \text{int} \to \text{long} \to \text{float} \to \text{double}$$

```java
// 自动类型转换：小类型 → 大类型
int i = 100;
long l = i;             // int → long，自动转换，不会丢失数据
double d = l;           // long → double，自动转换

// 特殊情况：char 可以自动转换为 int（获取 Unicode 编码值）
char c = 'A';
int code = c;           // code = 65（'A' 的 Unicode 编码）

// ⚠️ 表达式中的自动类型提升
// 当参与运算的操作数类型不同时，会自动提升为较大的类型
byte b1 = 10;
byte b2 = 20;
// byte b3 = b1 + b2;    // ❌ 编译错误！
// 原因：b1 + b2 的结果是 int 类型（byte 在运算时会提升为 int）
int result = b1 + b2;   // ✅ 正确：用 int 接收结果
byte b3 = (byte)(b1 + b2);  // ✅ 强制转换回 byte
```

### 强制类型转换（显式）

从大类型到小类型需要强制转换，可能丢失精度或溢出。

```java
// 强制类型转换：大类型 → 小类型
// 语法：(目标类型) 值
double d = 3.99;
int i = (int) d;        // 强制转换，结果为 3（直接截断小数，不是四舍五入）

// ⚠️ 溢出风险：当值超出目标类型范围时，会发生溢出
int big = 300;          // 300 超出 byte 范围（-128 ~ 127）
byte small = (byte) big; // 结果为 44（溢出）
// 原理：300 的二进制是 100101100，只取最后 8 位是 00101100 = 44

// ⚠️ 浮点数强制转换为整数时，直接截断小数部分
double pi = 3.14159;
int intPi = (int) pi;   // 结果是 3，不是 3 或 4（没有四舍五入）

// 正确的四舍五入方法
int rounded = (int)(pi + 0.5);  // 3.14159 + 0.5 = 3.64159，截断后为 3
// 或使用 Math.round()
int rounded2 = Math.round((float)pi);  // 3
```

### 包装类与字符串转换

每种基本类型都有对应的包装类（引用类型）：

| 基本类型 | 包装类 | 说明 |
|----------|--------|------|
| byte | Byte | |
| short | Short | |
| int | Integer | **最常用** |
| long | Long | |
| float | Float | |
| double | Double | **常用** |
| char | Character | |
| boolean | Boolean | |

```java
// 自动装箱（Autoboxing）：基本类型 → 包装类
// Java 5+ 自动完成，无需手动转换
Integer boxed = 100;    // int → Integer，编译器自动装箱
// 等价于：Integer boxed = Integer.valueOf(100);

// 自动拆箱（Unboxing）：包装类 → 基本类型
int unboxed = boxed;    // Integer → int，编译器自动拆箱
// 等价于：int unboxed = boxed.intValue();

// ⚠️ 装箱拆箱的注意点
Integer a = 100;
Integer b = 100;
System.out.println(a == b);   // true（缓存，-128 到 127 之间）

Integer c = 200;
Integer d = 200;
System.out.println(c == d);   // false！超出缓存范围，是不同对象
System.out.println(c.equals(d));  // true（使用 equals 比较）

// 字符串与数值转换
// 字符串 → 数值
int num = Integer.parseInt("123");       // 字符串转 int
double d = Double.parseDouble("3.14");   // 字符串转 double
long l = Long.parseLong("123456789");    // 字符串转 long

// 数值 → 字符串
String str1 = String.valueOf(123);       // int 转字符串
String str2 = Integer.toString(123);     // int 转字符串（另一种方式）
String str3 = 123 + "";                  // 空字符串拼接，简单但不推荐
```

---

## 运算符

### 算术运算符

```java
int a = 10, b = 3;

int sum = a + b;     // 加法：10 + 3 = 13
int diff = a - b;    // 减法：10 - 3 = 7
int prod = a * b;    // 乘法：10 * 3 = 30
int quot = a / b;    // 整数除法：10 / 3 = 3（截断小数部分）
int rem = a % b;     // 取余（取模）：10 % 3 = 1（10除以3余1）

// ⚠️ 整数除法的陷阱
int x = 5 / 2;       // 结果是 2，不是 2.5！
double y = 5 / 2;    // 结果是 2.0，不是 2.5！（先做整数除法，再转 double）
double z = 5.0 / 2;  // 结果是 2.5（有一个操作数是浮点数，结果就是浮点数）
double w = (double)5 / 2;  // 结果是 2.5（强制转换）

// 自增自减运算符
int i = 5;
int x1 = i++;    // 先使用 i 的值（5），然后 i 自增为 6
                 // x1 = 5, i = 6
int y1 = ++i;    // 先让 i 自增（变为 7），然后使用 i 的值
                 // y1 = 7, i = 7

int j = 5;
int x2 = j--;    // 先使用 j 的值（5），然后 j 自减为 4
                 // x2 = 5, j = 4
int y2 = --j;    // 先让 j 自减（变为 3），然后使用 j 的值
                 // y2 = 3, j = 3

// ⚠️ 自增自减的常见陷阱
int n = 5;
int result = n++ + ++n;  // 结果是什么？
// 解析：n++（值为5，n变为6）+ ++n（n先变为7，值为7）
// result = 5 + 7 = 12
// 建议：避免在复杂表达式中使用，代码可读性更重要
```

### 关系与逻辑运算符

```java
// 关系运算符：比较两个值，返回 boolean 结果
int a = 10, b = 3;

boolean eq = (a == b);   // 等于：10 == 3 → false
boolean ne = (a != b);   // 不等于：10 != 3 → true
boolean gt = (a > b);    // 大于：10 > 3 → true
boolean lt = (a < b);    // 小于：10 < 3 → false
boolean ge = (a >= b);   // 大于等于：10 >= 3 → true
boolean le = (a <= b);   // 小于等于：10 <= 3 → false

// ⚠️ 不要混淆 == 和 =
// == 是比较运算符，= 是赋值运算符
// if (a = 5)  // 编译错误！不能在条件中使用赋值

// 逻辑运算符：用于组合多个条件
boolean p = true, q = false;

// && 短路与：两个条件都为 true 才为 true
// 如果第一个条件为 false，不会计算第二个条件（短路）
boolean and = p && q;    // false
// 示例：避免空指针
String s = null;
// if (s != null && s.length() > 0)  // 安全：如果 s 为 null，不会执行 s.length()

// || 短路或：至少一个条件为 true 就为 true
// 如果第一个条件为 true，不会计算第二个条件（短路）
boolean or = p || q;     // true

// ! 非：取反
boolean not = !p;        // false

// & 和 | 非短路版本（不推荐使用）
// & 和 | 会计算两个条件，即使第一个条件已经能确定结果
boolean and2 = p & q;    // false（但会计算两个条件）
```

### 位运算符

位运算符直接对整数的二进制位进行操作，常用于底层编程、算法优化。

```java
int a = 5;    // 二进制：0101
int b = 3;    // 二进制：0011

// & 按位与：两位都为 1 才为 1
int and = a & b;    // 0101 & 0011 = 0001 = 1

// | 按位或：至少一位为 1 就为 1
int or = a | b;     // 0101 | 0011 = 0111 = 7

// ^ 按位异或：两位不同为 1，相同为 0
int xor = a ^ b;    // 0101 ^ 0011 = 0110 = 6
// 异或的性质：a ^ a = 0, a ^ 0 = a

// ~ 按位取反：0 变 1，1 变 0
int not = ~a;       // ~0101 = 11111111111111111111111111111010 = -6（补码）

// << 左移：左移 n 位相当于乘以 2^n
int left = a << 1;  // 0101 << 1 = 1010 = 10（相当于 5 * 2 = 10）
int left2 = a << 2; // 0101 << 2 = 10100 = 20（相当于 5 * 4 = 20）

// >> 右移（带符号）：右移 n 位相当于除以 2^n
int right = a >> 1; // 0101 >> 1 = 0010 = 2（相当于 5 / 2 = 2）

// >>> 无符号右移：高位补 0
int unsigned = -1 >>> 1;  // 正数结果

// 位运算的常见应用
// 1. 判断奇偶
boolean isOdd = (n & 1) == 1;  // 比 n % 2 == 1 更快

// 2. 交换两个数（不使用临时变量）
int x = 3, y = 5;
x = x ^ y;  // x = 3 ^ 5
y = x ^ y;  // y = (3 ^ 5) ^ 5 = 3
x = x ^ y;  // x = (3 ^ 5) ^ 3 = 5

// 3. 取模（当除数是 2 的幂时）
int mod = n & (m - 1);  // 等价于 n % m，当 m 是 2 的幂时
```

### 三元运算符

三元运算符是 if-else 的简化形式，适合简单的条件判断。

```java
// 语法：条件 ? 条件为真的值 : 条件为假的值
int age = 18;
String status = (age >= 18) ? "成年" : "未成年";
// 等价于：
// if (age >= 18) {
//     status = "成年";
// } else {
//     status = "未成年";
// }

// 嵌套使用（不推荐，可读性差）
int score = 85;
String grade = (score >= 90) ? "A" : 
               (score >= 80) ? "B" : 
               (score >= 60) ? "C" : "D";
// 等价于：
// if (score >= 90) grade = "A";
// else if (score >= 80) grade = "B";
// else if (score >= 60) grade = "C";
// else grade = "D";

// 常见用法：避免空指针
String name = user != null ? user.getName() : "未知";
// 使用 Optional 更优雅（Java 8+）
String name2 = Optional.ofNullable(user).map(User::getName).orElse("未知");
```

---

## 控制流语句

### 条件语句

```java
int score = 85;

// if 语句：单条件
if (score >= 60) {
    System.out.println("及格");
}

// if-else 语句：二选一
if (score >= 60) {
    System.out.println("及格");
} else {
    System.out.println("不及格");
}

// if-else if-else 语句：多选一
if (score >= 90) {
    System.out.println("优秀");
} else if (score >= 80) {
    System.out.println("良好");
} else if (score >= 60) {
    System.out.println("及格");
} else {
    System.out.println("不及格");
}

// switch 语句：多分支选择
// ⚠️ 如果没有 break，会继续执行下一个 case（穿透）
int day = 3;
switch (day) {
    case 1:
        System.out.println("星期一");
        break;  // 必须有 break，否则会继续执行下一个 case
    case 2:
        System.out.println("星期二");
        break;
    case 3:
        System.out.println("星期三");
        break;
    default:
        System.out.println("其他");
        // default 后的 break 可以省略
}

// switch 表达式（Java 14+）：更简洁，自动 break
String result = switch (day) {
    case 1, 2, 3, 4, 5 -> "工作日";  // 使用 -> 箭头语法
    case 6, 7 -> "周末";
    default -> "无效";
};

// switch 表达式使用 yield 返回值（Java 14+）
String result2 = switch (day) {
    case 1, 2, 3, 4, 5 -> {
        System.out.println("处理工作日");
        yield "工作日";  // 使用 yield 返回值
    }
    case 6, 7 -> "周末";
    default -> "无效";
};
```

### 循环语句

```java
// for 循环：已知循环次数
// 语法：for (初始化; 条件; 更新) { 循环体 }
for (int i = 0; i < 5; i++) {
    System.out.println(i);  // 输出 0, 1, 2, 3, 4
}

// for 循环的执行顺序：
// 1. int i = 0;（初始化，只执行一次）
// 2. i < 5;（条件判断，为 true 继续执行）
// 3. System.out.println(i);（循环体）
// 4. i++;（更新）
// 5. 回到第 2 步

// 增强 for 循环（for-each）：遍历数组或集合
// 语法：for (元素类型 变量名 : 数组/集合) { 循环体 }
int[] numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
    System.out.println(num);
}
// ⚠️ 无法获取当前索引，无法修改数组元素

// while 循环：先判断条件，再执行循环体
int i = 0;
while (i < 5) {
    System.out.println(i);
    i++;
}

// do-while 循环：先执行一次循环体，再判断条件
// 至少执行一次
int j = 0;
do {
    System.out.println(j);
    j++;
} while (j < 5);

// 死循环的几种写法
while (true) {
    // 需要在内部使用 break 退出
}

for (;;) {
    // 等价于 while (true)
}
```

### 跳转语句

```java
// break：跳出整个循环
for (int i = 0; i < 10; i++) {
    if (i == 5) break;  // 当 i 等于 5 时，跳出循环
    System.out.println(i);  // 输出 0, 1, 2, 3, 4
}

// continue：跳过本次迭代，继续下一次
for (int i = 0; i < 5; i++) {
    if (i == 2) continue;  // 当 i 等于 2 时，跳过本次循环
    System.out.println(i);  // 输出 0, 1, 3, 4
}

// 带标签的 break：跳出多层循环
// 标签名可以是任意合法标识符
outer:  // 定义标签
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        if (i == 1 && j == 1) {
            break outer;  // 跳出外层循环
        }
        System.out.println("i=" + i + ", j=" + j);
    }
}
// 输出：i=0, j=0 / i=0, j=1 / i=0, j=2 / i=1, j=0
// 然后直接退出外层循环

// 带标签的 continue：跳到外层循环的下一次迭代
outer:
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        if (j == 1) {
            continue outer;  // 跳到外层循环的下一次迭代
        }
        System.out.println("i=" + i + ", j=" + j);
    }
}
// 输出：i=0, j=0 / i=1, j=0 / i=2, j=0
```

---

## 数组

数组是存储相同类型元素的固定大小容器。

### 数组声明与初始化

```java
// 声明数组的两种语法
int[] arr1;      // 推荐：类型[] 变量名
int arr2[];      // 不推荐：类型 变量名[]（C/C++ 风格）

// 静态初始化：在声明时指定所有元素
int[] arr3 = {1, 2, 3, 4, 5};  // 编译器自动计算长度
int[] arr4 = new int[]{1, 2, 3};  // 完整语法

// 动态初始化：只指定长度，元素为默认值
int[] arr5 = new int[5];       // 5个元素，默认值都是 0
String[] arr6 = new String[3]; // 3个元素，默认值都是 null
boolean[] arr7 = new boolean[2]; // 默认值都是 false

// 访问数组元素
// 索引从 0 开始，最大索引是 长度-1
arr5[0] = 10;           // 修改第一个元素
int first = arr5[0];    // 获取第一个元素
int len = arr5.length;  // 获取数组长度（注意：length 是属性，不是方法）

// ⚠️ 数组越界：访问不存在的索引会抛出异常
// arr5[5] = 10;  // 运行时错误！ArrayIndexOutOfBoundsException
```

### 数组操作

```java
import java.util.Arrays;  // 需要导入 Arrays 工具类

int[] arr = {5, 2, 8, 1, 9};

// 排序
Arrays.sort(arr);  // 原地排序，arr 变为 [1, 2, 5, 8, 9]

// 二分查找（要求数组有序）
int index = Arrays.binarySearch(arr, 5);  // 返回索引 2
int notFound = Arrays.binarySearch(arr, 3);  // 返回负数（未找到）

// 转字符串（用于打印）
String str = Arrays.toString(arr);  // "[1, 2, 5, 8, 9]"

// 复制
int[] copy = Arrays.copyOf(arr, 3);  // 复制前 3 个元素：[1, 2, 5]
int[] copy2 = Arrays.copyOf(arr, 10);  // 扩展长度，多出的位置填充默认值 0

// 填充
int[] filled = new int[5];
Arrays.fill(filled, 100);  // 所有元素都设为 100：[100, 100, 100, 100, 100]

// 比较
int[] arr1 = {1, 2, 3};
int[] arr2 = {1, 2, 3};
boolean same = Arrays.equals(arr1, arr2);  // true（比较内容，不是引用）
```

### 多维数组

```java
// 二维数组：数组的数组
int[][] matrix = {
    {1, 2, 3},
    {4, 5, 6}
};
// matrix 是一个包含 2 个元素的数组
// 每个元素又是一个包含 3 个元素的数组

// 访问元素
int val = matrix[0][1];  // 第 1 行第 2 列，值为 2
int rows = matrix.length;      // 行数：2
int cols = matrix[0].length;   // 列数：3

// 动态初始化
int[][] arr = new int[2][3];  // 2 行 3 列，所有元素为 0

// 不规则数组（每行长度不同）
int[][] irregular = new int[3][];  // 只指定行数
irregular[0] = new int[1];  // 第 1 行有 1 个元素
irregular[1] = new int[2];  // 第 2 行有 2 个元素
irregular[2] = new int[3];  // 第 3 行有 3 个元素

// 遍历二维数组
for (int i = 0; i < matrix.length; i++) {
    for (int j = 0; j < matrix[i].length; j++) {
        System.out.print(matrix[i][j] + " ");
    }
    System.out.println();
}

// 使用增强 for 循环
for (int[] row : matrix) {
    for (int elem : row) {
        System.out.print(elem + " ");
    }
    System.out.println();
}
```

---

## 输入输出

### 控制台输出

```java
// println：输出后换行
System.out.println("Hello");      // 输出 Hello 并换行

// print：输出后不换行
System.out.print("Hello");        // 输出 Hello，不换行
System.out.print("World");        // 输出 HelloWorld

// printf：格式化输出
// %s 字符串，%d 整数，%f 浮点数，%n 换行（跨平台）
System.out.printf("姓名：%s，年龄：%d，分数：%.2f%n", "张三", 25, 89.567);
// 输出：姓名：张三，年龄：25，分数：89.57

// 常用格式化符号
// %s - 字符串
// %d - 十进制整数
// %f - 浮点数（%.2f 保留两位小数）
// %x - 十六进制整数
// %o - 八进制整数
// %n - 换行（推荐使用，跨平台）
// %% - 输出百分号
```

### 控制台输入

```java
import java.util.Scanner;  // 需要导入 Scanner 类

// 创建 Scanner 对象，从标准输入读取
Scanner scanner = new Scanner(System.in);

// 读取字符串
System.out.print("请输入姓名：");
String name = scanner.nextLine();  // 读取一行（包含空格）
// String word = scanner.next();   // 读取一个单词（空格分隔）

// 读取整数
System.out.print("请输入年龄：");
int age = scanner.nextInt();

// 读取浮点数
System.out.print("请输入分数：");
double score = scanner.nextDouble();

// ⚠️ nextLine 的陷阱：在 nextInt() 之后使用 nextLine()
int num = scanner.nextInt();       // 输入 123 后按回车
String line = scanner.nextLine();  // 直接读取了换行符，line 为空
// 解决方案：再调用一次 nextLine() 消耗换行符
// int num = scanner.nextInt();
// scanner.nextLine();  // 消耗换行符
// String line = scanner.nextLine();

// 关闭 Scanner（释放资源）
scanner.close();
// ⚠️ 关闭后不能再使用 System.in
```

---

## var 类型推断（Java 10+）

```java
// var：让编译器自动推断变量类型
// ⚠️ var 只能用于局部变量，必须有初始化值
var name = "张三";           // 编译器推断为 String
var age = 25;               // 编译器推断为 int
var list = new ArrayList<String>();  // 编译器推断为 ArrayList<String>
var map = new HashMap<String, Integer>();  // 推断为 HashMap<String, Integer>

// 编译后，var 会被替换为具体类型，运行时没有区别

// ⚠️ var 的限制
// var x;  // ❌ 编译错误！必须初始化
// var y = null;  // ❌ 编译错误！无法推断类型
// var z = {1, 2, 3};  // ❌ 编译错误！数组初始化需要显式类型
// var[] arr = {1, 2, 3};  // ❌ 编译错误！var 不能用于数组声明

// ⚠️ 使用建议
// - 类型明显时可使用 var 提高可读性
// - 类型不明确时，显式声明类型更好
var stream = list.stream().filter(s -> s.length() > 3);  // 类型复杂，var 更简洁
```

---

## 字符串

字符串是 Java 中最常用的引用类型。

### 字符串创建

```java
// 方式一：直接使用双引号创建（推荐）
// 字符串存储在字符串常量池中
String s1 = "Hello";

// 方式二：使用 new 关键字创建
// 在堆中创建新对象
String s2 = new String("Hello");

// 字符串常量池的作用
String s3 = "Hello";
String s4 = "Hello";
System.out.println(s1 == s3);   // true（指向常量池中同一个对象）
System.out.println(s1 == s2);   // false（s2 是堆中的新对象）
System.out.println(s1.equals(s2));  // true（内容相同）

// ⚠️ 字符串比较：永远使用 equals()，不要使用 ==
// == 比较的是引用（内存地址）
// equals() 比较的是内容
```

### 字符串不可变性

```java
// Java 字符串是不可变的（immutable）
// 一旦创建，内容就不能修改
String s = "Hello";
s = s + " World";  // 看似修改了 s，实际上是创建了新对象

// 原来的 "Hello" 对象仍然存在于常量池中
// s 现在指向新创建的 "Hello World" 对象

// 不可变性的好处：
// 1. 线程安全：多个线程可以安全共享
// 2. 字符串常量池可以优化内存
// 3. hashCode 可以缓存，提高 HashMap 性能
// 4. 安全性：字符串作为参数时不会被修改
```

### 常用方法

```java
String s = "Hello, World!";

// 长度与判断
int len = s.length();              // 13
boolean empty = s.isEmpty();       // false
boolean blank = s.isBlank();       // false（Java 11+，检查是否只有空白字符）

// 访问字符
char c = s.charAt(0);              // 'H'
char last = s.charAt(s.length() - 1);  // '!'（最后一个字符）

// 查找
int idx = s.indexOf("World");      // 7（第一次出现的位置）
int lastIdx = s.lastIndexOf("o");  // 8（最后一次出现的位置）
boolean contains = s.contains("World");  // true

// 截取
String sub1 = s.substring(7);      // "World!"（从索引 7 到末尾）
String sub2 = s.substring(0, 5);   // "Hello"（从索引 0 到 5，不包括 5）

// 分割
String[] parts = "a,b,c".split(",");  // ["a", "b", "c"]
String[] words = "a  b  c".split("\\s+");  // 按空格分割（正则表达式）

// 大小写转换
String upper = s.toUpperCase();    // "HELLO, WORLD!"
String lower = s.toLowerCase();    // "hello, world!"

// 去除空格
String trimmed = "  hello  ".trim();    // "hello"（去除首尾空格）
String stripped = "  hello  ".strip();  // "hello"（Java 11+，支持 Unicode 空白）

// 替换
String replaced = s.replace("World", "Java");  // "Hello, Java!"
String replacedAll = "a1b2c3".replaceAll("\\d", "X");  // "aXbXcX"（正则）

// 拼接
String joined = String.join("-", "a", "b", "c");  // "a-b-c"
String joined2 = String.join(", ", Arrays.asList("A", "B", "C"));  // "A, B, C"

// 判断开头/结尾
boolean starts = s.startsWith("Hello");  // true
boolean ends = s.endsWith("!");          // true

// 判断相等
boolean eq = s.equals("Hello, World!");        // true
boolean eqIg = s.equalsIgnoreCase("hello, world!");  // true（忽略大小写）
```

### StringBuilder

当需要频繁拼接字符串时，使用 StringBuilder 更高效。

```java
// ⚠️ 使用 + 拼接字符串的问题
String result = "";
for (int i = 0; i < 1000; i++) {
    result += i;  // 每次拼接都创建新对象，效率低
}

// ✅ 使用 StringBuilder
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.append(i);  // 在原对象上修改，效率高
}
String result = sb.toString();

// StringBuilder 常用方法
StringBuilder sb2 = new StringBuilder("Hello");

sb2.append(" World");       // 追加到末尾：Hello World
sb2.insert(5, ",");         // 在索引 5 插入：Hello, World
sb2.delete(5, 6);           // 删除索引 5-6（不含 6）：Hello World
sb2.reverse();              // 反转：dlroW olleH
sb2.replace(0, 5, "Hi");    // 替换：Hi World

int capacity = sb2.capacity();  // 获取容量（默认 16 + 字符串长度）

// StringBuilder vs StringBuffer
// StringBuilder：非线程安全，效率高，推荐使用
// StringBuffer：线程安全，效率低，遗留类

// ⚠️ 单条语句的字符串拼接，编译器会自动优化为 StringBuilder
String s = "a" + "b" + "c";  // 编译器优化：String s = "abc";
String s2 = "a" + variable + "c";  // 编译器优化为 StringBuilder
```

---

## 小结

| 概念 | 要点 | 常见错误 |
|------|------|----------------|
| **基本类型** | 8 种：byte, short, int, long, float, double, char, boolean | 忘记 long 的 L 后缀、float 的 f 后缀 |
| **类型转换** | 自动（小→大）、强制（大→小，可能丢失精度） | 整数除法精度丢失、强制转换溢出 |
| **运算符** | 算术、关系、逻辑、位运算、三元 | 混淆 == 和 =、整数除法 |
| **控制流** | if-else, switch, for, while, do-while | switch 忘记 break |
| **数组** | 固定长度，索引从 0 开始 | 数组越界、length 忘记是属性 |
| **字符串** | 不可变，使用 equals 比较 | 使用 == 比较字符串、频繁拼接 |

### 学习建议

1. **动手实践**：每学一个概念，都要在 IDE 中敲代码验证
2. **理解原理**：不要死记语法，理解背后的原理（如为什么字符串不可变）
3. **注意陷阱**：本节标注的 ⚠️ 都是常见错误，要特别注意
4. **善用调试**：使用 IDE 的调试功能，观察变量的值的变化