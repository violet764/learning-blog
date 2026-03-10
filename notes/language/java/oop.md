# Java 面向对象编程

> 面向对象编程（OOP）是 Java 的核心编程范式。Java 通过**封装**、**继承**、**多态**三大特性，实现代码的模块化、复用性和扩展性。

## 类与对象

### 类的定义

类是对象的模板，定义了对象的属性（字段）和行为（方法）。

```java
public class Person {
    // 实例变量（字段）
    private String name;
    private int age;
    
    // 静态变量（类变量）
    private static int count = 0;
    
    // 常量
    public static final String SPECIES = "人类";
    
    // 构造方法
    public Person() {
        this("未知", 0);
    }
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        count++;
    }
    
    // 实例方法
    public void sayHello() {
        System.out.println("你好，我是" + name);
    }
    
    // getter 和 setter
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getAge() { return age; }
    public void setAge(int age) { 
        if (age > 0) this.age = age; 
    }
}
```

### 对象创建与使用

```java
// 创建对象
Person p1 = new Person();
Person p2 = new Person("张三", 25);

// 使用对象
p2.sayHello();              // 调用方法
p2.setAge(26);              // 修改属性
System.out.println(p2.getName());  // 访问属性
```

---

## 封装

📌 **封装**是指隐藏对象的内部实现细节，通过访问控制符控制访问权限。

### 访问控制符

| 修饰符 | 同一类 | 同一包 | 子类 | 其他 |
|--------|:------:|:------:|:----:|:----:|
| `public` | ✓ | ✓ | ✓ | ✓ |
| `protected` | ✓ | ✓ | ✓ | ✗ |
| 默认（无修饰符） | ✓ | ✓ | ✗ | ✗ |
| `private` | ✓ | ✗ | ✗ | ✗ |

```java
public class BankAccount {
    private double balance;  // 私有字段，外部无法直接访问
    
    public double getBalance() {  // 公开方法，提供访问接口
        return balance;
    }
    
    public void deposit(double amount) {
        if (amount > 0) balance += amount;
    }
    
    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
}
```

---

## 继承

📌 **继承**允许子类继承父类的属性和方法，实现代码复用。

### 基本继承

```java
// 父类
public class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public void eat() {
        System.out.println(name + "在吃东西");
    }
    
    public void makeSound() {
        System.out.println(name + "发出声音");
    }
}

// 子类
public class Dog extends Animal {
    private String breed;
    
    public Dog(String name, String breed) {
        super(name);  // 调用父类构造方法
        this.breed = breed;
    }
    
    @Override
    public void makeSound() {  // 重写父类方法
        System.out.println(name + "汪汪叫");
    }
    
    public void fetch() {  // 子类特有方法
        System.out.println(name + "在捡球");
    }
}
```

### super 关键字

```java
public class Child extends Parent {
    public Child(int value) {
        super(value);        // 调用父类构造方法
    }
    
    @Override
    public void method() {
        super.method();      // 调用父类方法
    }
}
```

### final 关键字

```java
// final 类不能被继承
public final class Constants { }

// final 方法不能被重写
public class Base {
    public final void cannotOverride() { }
}

// final 变量只能赋值一次
private final int id = 100;
```

---

## 多态

📌 **多态**是指同一操作作用于不同对象，产生不同的行为。多态的实现依赖于：
- 继承或接口实现
- 方法重写
- 父类引用指向子类对象

### 向上转型与向下转型

```java
// 向上转型（自动）
Animal animal = new Dog("小黄", "金毛");
animal.makeSound();  // 调用子类重写的方法 → "小黄汪汪叫"

// animal.fetch();  // ❌ 编译错误！Animal 没有 fetch 方法

// 向下转型（需要强制转换）
if (animal instanceof Dog) {
    Dog dog = (Dog) animal;
    dog.fetch();  // ✅ 现在可以调用
}

// Java 16+ 模式匹配
if (animal instanceof Dog dog) {
    dog.fetch();
}
```

### 多态示例

```java
Animal[] animals = {
    new Dog("小黄", "金毛"),
    new Cat("小白", "波斯"),
    new Bird("小鸟", "麻雀")
};

for (Animal animal : animals) {
    animal.makeSound();  // 多态调用，输出不同的声音
}
// 输出：
// 小黄汪汪叫
// 小白喵喵叫
// 小鸟叽叽喳喳
```

---

## 抽象类

📌 **抽象类**不能被实例化，可以包含抽象方法（无实现）和具体方法（有实现）。

```java
public abstract class Shape {
    protected String color;
    
    public Shape(String color) {
        this.color = color;
    }
    
    // 抽象方法：子类必须实现
    public abstract double getArea();
    
    // 具体方法：子类可以直接使用
    public String getColor() {
        return color;
    }
    
    public void printInfo() {
        System.out.println("颜色：" + color + "，面积：" + getArea());
    }
}

// 具体子类
public class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double getArea() {
        return Math.PI * radius * radius;
    }
}

public class Rectangle extends Shape {
    private double width, height;
    
    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }
    
    @Override
    public double getArea() {
        return width * height;
    }
}
```

---

## 接口

📌 **接口**是一种纯抽象类型，定义了一组方法签名。Java 8 之后，接口也可以包含默认方法和静态方法。

### 接口定义与实现

```java
// 定义接口
public interface Drawable {
    double DEFAULT_SIZE = 1.0;  // 常量（默认 public static final）
    
    void draw();  // 抽象方法（默认 public abstract）
    
    // 默认方法（Java 8+）
    default void printInfo() {
        System.out.println("可绘制对象");
    }
    
    // 静态方法（Java 8+）
    static void staticMethod() {
        System.out.println("静态方法");
    }
}

// 实现接口
public class Circle implements Drawable {
    @Override
    public void draw() {
        System.out.println("绘制圆形");
    }
}
```

### 多接口实现

```java
public interface Flyable {
    void fly();
}

public interface Swimmable {
    void swim();
}

// 实现多个接口
public class Duck implements Flyable, Swimmable {
    @Override
    public void fly() { 
        System.out.println("鸭子飞行"); 
    }
    
    @Override
    public void swim() { 
        System.out.println("鸭子游泳"); 
    }
}
```

### 接口继承

```java
public interface A {
    void methodA();
}

public interface B {
    void methodB();
}

// 接口多继承
public interface C extends A, B {
    void methodC();
}

// 实现类需要实现所有方法
public class Impl implements C {
    @Override
    public void methodA() { }
    
    @Override
    public void methodB() { }
    
    @Override
    public void methodC() { }
}
```

### 接口 vs 抽象类

| 特性 | 接口 | 抽象类 |
|------|:----:|:------:|
| 多继承 | ✓（可实现多个） | ✗（单继承） |
| 构造方法 | ✗ | ✓ |
| 成员变量 | 只能有常量 | 可有各种类型 |
| 方法实现 | 默认方法、静态方法 | 抽象方法、具体方法 |
| 设计目的 | 定义行为契约 | 代码复用 |

---

## 内部类

📌 **内部类**是定义在另一个类内部的类，可以访问外部类的私有成员。

### 成员内部类

```java
public class Outer {
    private int outerField = 10;
    
    public class Inner {
        public void accessOuter() {
            System.out.println(outerField);         // 访问外部类成员
            System.out.println(Outer.this.outerField);  // 显式引用
        }
    }
}

// 创建内部类实例
Outer outer = new Outer();
Outer.Inner inner = outer.new Inner();
```

### 静态内部类

```java
public class Outer {
    private static int staticField = 10;
    
    public static class StaticInner {
        public void method() {
            System.out.println(staticField);  // 只能访问静态成员
        }
    }
}

// 创建静态内部类实例
Outer.StaticInner inner = new Outer.StaticInner();
```

### 匿名内部类

```java
interface Greeting {
    void greet();
}

// 使用匿名内部类
Greeting greeting = new Greeting() {
    @Override
    public void greet() {
        System.out.println("你好！");
    }
};
greeting.greet();
```

---

## 枚举类

📌 **枚举**是一组常量的集合，使用 `enum` 关键字定义。

### 基本枚举

```java
public enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

Day today = Day.MONDAY;

// switch 中使用
switch (today) {
    case MONDAY -> System.out.println("星期一");
    case FRIDAY -> System.out.println("星期五");
    default -> System.out.println("其他");
}
```

### 带字段和方法的枚举

```java
public enum Planet {
    MERCURY(3.303e+23, 2.4397e6),
    EARTH(5.976e+24, 6.37814e6),
    MARS(6.421e+23, 3.3972e6);
    
    private final double mass;
    private final double radius;
    
    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
    }
    
    public double getMass() { return mass; }
    public double getRadius() { return radius; }
    
    // 计算表面重力
    public double surfaceGravity() {
        return 6.67e-11 * mass / (radius * radius);
    }
}
```

### 枚举常用方法

```java
Day[] days = Day.values();           // 获取所有枚举值
Day day = Day.valueOf("FRIDAY");     // 字符串转枚举
int ordinal = day.ordinal();         // 获取序号（从0开始）
String name = day.name();            // 获取名称
```

---

## Object 类

所有类的父类，包含以下核心方法：

```java
public class Person {
    private String name;
    private int age;
    
    // 构造方法
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // 字符串表示
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
    
    // 判断相等
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Person person = (Person) obj;
        return age == person.age && Objects.equals(name, person.name);
    }
    
    // 哈希码
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
}
```

### Object 类核心方法

| 方法 | 说明 |
|------|------|
| `toString()` | 返回对象的字符串表示 |
| `equals(Object)` | 判断两个对象是否相等 |
| `hashCode()` | 返回对象的哈希码 |
| `getClass()` | 返回对象的运行时类 |
| `clone()` | 创建并返回对象的副本 |
| `finalize()` | 垃圾回收时调用（已废弃） |

---

## 小结

| 概念 | 说明 |
|------|------|
| **类** | 对象的模板，包含字段和方法 |
| **封装** | 隐藏实现细节，控制访问权限 |
| **继承** | 子类继承父类的特性，使用 `extends` |
| **多态** | 同一操作不同对象产生不同行为 |
| **抽象类** | 不能实例化，可包含抽象方法 |
| **接口** | 纯抽象，支持多实现 |
| **内部类** | 定义在类内部的类 |
| **枚举** | 一组常量的集合 |
