# Java 面向对象编程

> 面向对象编程（OOP）是 Java 的核心编程范式。Java 通过**封装**、**继承**、**多态**三大特性，实现代码的模块化、复用性和扩展性。

## 面向对象

### 面向对象的核心概念

```
类（Class）        → 对象的模板/蓝图
    ↓ 实例化
对象（Object）     → 类的具体实例
    ↓ 包含
属性（Field）      → 对象的数据（是什么）
方法（Method）     → 对象的行为（能做什么）
```

**生活中的类比**：
- **类** = 汽车设计图纸（规定了汽车应该有什么属性、什么功能）
- **对象** = 根据图纸制造的一辆辆具体的汽车
- **属性** = 颜色、品牌、速度等
- **方法** = 启动、加速、刹车等

---

## 类与对象

### 类的定义

类是对象的模板，定义了对象的属性（字段）和行为（方法）。

```java
// 类的定义语法
// public class 类名 { }
public class Person {
    // ========== 属性（字段/成员变量）==========
    // private 表示私有，外部无法直接访问（封装）
    private String name;    // 姓名
    private int age;        // 年龄
    
    // static 表示静态变量（类变量），所有对象共享
    // 用于记录创建了多少个 Person 对象
    private static int count = 0;
    
    // public static final 表示公共静态常量
    // 常量：值不能改变的变量，命名全大写
    public static final String SPECIES = "人类";
    
    // ========== 构造方法 ==========
    // 构造方法：用于创建对象时初始化属性
    // 特点：方法名与类名相同，没有返回值类型
    
    // 无参构造方法（默认构造方法）
    public Person() {
        // this() 调用另一个构造方法
        // 必须放在构造方法的第一行
        this("未知", 0);
    }
    
    // 有参构造方法
    public Person(String name, int age) {
        // this 关键字：表示当前对象
        // 用于区分同名的局部变量和成员变量
        this.name = name;
        this.age = age;
        count++;  // 每创建一个对象，计数器加 1
    }
    
    // ========== 方法 ==========
    // 实例方法：必须通过对象调用
    public void sayHello() {
        // 方法内部可以直接访问对象的属性
        System.out.println("你好，我是" + name + "，今年" + age + "岁");
    }
    
    // 静态方法：通过类名调用，不能访问实例变量
    public static int getCount() {
        return count;
    }
    
    // ========== getter 和 setter ==========
    // getter：获取属性值
    public String getName() { 
        return name; 
    }
    
    // setter：设置属性值（可以在设置时添加验证逻辑）
    public void setName(String name) { 
        this.name = name; 
    }
    
    public int getAge() { 
        return age; 
    }
    
    public void setAge(int age) { 
        // 可以在 setter 中添加验证逻辑
        if (age > 0 && age < 150) {
            this.age = age; 
        } else {
            System.out.println("年龄设置无效");
        }
    }
}
```

### 对象创建与使用

```java
public class Main {
    public static void main(String[] args) {
        // 创建对象：使用 new 关键字
        // 类名 对象名 = new 构造方法();
        
        // 方式一：使用无参构造
        Person p1 = new Person();  // name="未知", age=0
        
        // 方式二：使用有参构造
        Person p2 = new Person("张三", 25);  // name="张三", age=25
        
        // 使用对象
        p2.sayHello();              // 调用方法：你好，我是张三，今年25岁
        
        p2.setAge(26);              // 修改属性
        System.out.println(p2.getName());  // 访问属性：张三
        
        // 访问静态成员
        // 静态成员属于类，不属于某个对象
        System.out.println(Person.SPECIES);  // 人类
        System.out.println(Person.getCount());  // 2（创建了 2 个对象）
        
        // ⚠️ 静态成员也可以通过对象访问，但不推荐
        // System.out.println(p1.SPECIES);  // 不推荐
    }
}
```

### 对象的内存模型

理解对象在内存中的存储，有助于理解 Java 的行为：

```
栈内存（Stack）              堆内存（Heap）
┌─────────────┐            ┌─────────────────────┐
│ p1 = 0x100  │ ──────→    │ Person 对象          │
│ p2 = 0x200  │ ──────┐    │ name = "未知"        │
└─────────────┘       │    │ age = 0             │
                      │    └─────────────────────┘
                      │
                      │    ┌─────────────────────┐
                      └──→ │ Person 对象          │
                           │ name = "张三"        │
                           │ age = 25            │
                           └─────────────────────┘
方法区（Method Area）
┌─────────────────────┐
│ Person 类信息        │
│ static count = 2    │
│ static SPECIES = "人类"│
└─────────────────────┘
```

**关键理解**：
- **栈内存**：存储局部变量（如对象引用 p1, p2）
- **堆内存**：存储对象实例（实际的属性值）
- **方法区**：存储类信息、静态变量、常量

---

## 封装

📌 **封装**是指隐藏对象的内部实现细节，通过访问控制符控制访问权限。

### 为什么需要封装？

```java
// ❌ 没有封装的问题
public class Student {
    public int age;  // 公开的，任何人都可以直接修改
}

Student s = new Student();
s.age = -100;  // 不合理的值也能设置！
s.age = 1000;  // 数据不安全

// ✅ 使用封装
public class Student {
    private int age;  // 私有的，外部无法直接访问
    
    public void setAge(int age) {
        if (age > 0 && age < 150) {  // 验证逻辑
            this.age = age;
        } else {
            throw new IllegalArgumentException("年龄必须在0-150之间");
        }
    }
}
```

### 访问控制符

Java 提供了 4 种访问控制符，控制类、方法、属性的可见性：

| 修饰符 | 同一类内 | 同一包内 | 不同包子类 | 其他位置 |
|--------|:--------:|:--------:|:----------:|:--------:|
| `public` | ✓ | ✓ | ✓ | ✓ |
| `protected` | ✓ | ✓ | ✓ | ✗ |
| 默认（无修饰符） | ✓ | ✓ | ✗ | ✗ |
| `private` | ✓ | ✗ | ✗ | ✗ |

```java
package com.example;

public class AccessDemo {
    public int publicVar = 1;       // 任何地方都能访问
    protected int protectedVar = 2; // 同包或子类能访问
    int defaultVar = 3;             // 只能在同包内访问（默认）
    private int privateVar = 4;     // 只能在本类内访问
    
    public void method() {
        // 同一类内，所有都能访问
        System.out.println(publicVar);
        System.out.println(protectedVar);
        System.out.println(defaultVar);
        System.out.println(privateVar);
    }
}

// 不同包的子类
package com.other;
import com.example.AccessDemo;

public class Child extends AccessDemo {
    public void test() {
        System.out.println(publicVar);      // ✓
        System.out.println(protectedVar);   // ✓（子类可以访问）
        // System.out.println(defaultVar);  // ✗ 编译错误
        // System.out.println(privateVar);  // ✗ 编译错误
    }
}
```

### 封装的最佳实践

```java
public class BankAccount {
    // 私有属性：外部无法直接访问
    private String accountNumber;  // 账号
    private double balance;        // 余额
    private String owner;          // 户主
    
    // 构造方法
    public BankAccount(String accountNumber, String owner) {
        this.accountNumber = accountNumber;
        this.owner = owner;
        this.balance = 0;
    }
    
    // 公开的 getter 方法：提供只读访问
    public double getBalance() {
        return balance;
    }
    
    public String getOwner() {
        return owner;
    }
    
    // 不提供 setBalance，余额只能通过存取款修改
    
    // 存款：控制修改逻辑
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("存款成功：" + amount + "，余额：" + balance);
        } else {
            System.out.println("存款金额必须大于0");
        }
    }
    
    // 取款：控制修改逻辑
    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("取款成功：" + amount + "，余额：" + balance);
            return true;
        }
        System.out.println("取款失败：余额不足或金额无效");
        return false;
    }
}

// 使用
BankAccount account = new BankAccount("123456", "张三");
account.deposit(1000);    // 存款成功：1000.0，余额：1000.0
account.withdraw(500);    // 取款成功：500.0，余额：500.0
// account.balance = -100;  // 编译错误！无法直接访问
```

---

## 继承

📌 **继承**允许子类继承父类的属性和方法，实现代码复用。

### 继承的本质

继承是一种"is-a"关系：
- Dog **is a** Animal → Dog 继承 Animal
- Car **is a** Vehicle → Car 继承 Vehicle

```java
// 父类（基类/超类）
public class Animal {
    protected String name;  // protected：子类可以访问
    protected int age;
    
    // 父类构造方法
    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // 父类方法
    public void eat() {
        System.out.println(name + "在吃东西");
    }
    
    public void sleep() {
        System.out.println(name + "在睡觉");
    }
    
    // 父类方法（子类可以重写）
    public void makeSound() {
        System.out.println(name + "发出声音");
    }
}

// 子类（派生类）：使用 extends 关键字继承
public class Dog extends Animal {
    // 子类特有的属性
    private String breed;  // 品种
    
    // 子类构造方法
    public Dog(String name, int age, String breed) {
        // super()：调用父类的构造方法
        // 必须放在构造方法的第一行
        super(name, age);
        this.breed = breed;
    }
    
    // 子类特有的方法
    public void fetch() {
        System.out.println(name + "在捡球");
    }
    
    // 方法重写（Override）：修改父类的行为
    @Override  // 注解：表示这是重写的方法，编译器会检查
    public void makeSound() {
        System.out.println(name + "(" + breed + ")汪汪叫");
    }
    
    // 子类可以调用父类方法
    @Override
    public void eat() {
        super.eat();  // 先调用父类的方法
        System.out.println(name + "吃得很开心");
    }
}

// 使用
Dog dog = new Dog("小黄", 3, "金毛");
dog.eat();        // 调用继承的方法：小黄在吃东西 → 小黄吃得很开心
dog.makeSound();  // 调用重写的方法：小黄(金毛)汪汪叫
dog.fetch();      // 调用子类特有方法：小黄在捡球
```

### 继承的注意事项

```java
// 1. Java 只支持单继承：一个类只能有一个直接父类
// public class Dog extends Animal, Creature { }  // ❌ 编译错误！

// 2. 但支持多层继承
// Animal → Mammal → Dog（Dog 间接继承了 Animal）

// 3. 所有类都直接或间接继承自 Object 类
// public class Animal { } 等价于 public class Animal extends Object { }

// 4. 构造方法不能继承
// 子类必须在自己的构造方法中调用父类的构造方法
public class Cat extends Animal {
    public Cat(String name, int age) {
        super(name, age);  // 必须显式调用父类构造方法
    }
}

// 5. private 成员不能直接访问
// 子类不能直接访问父类的 private 成员，但可以通过父类的 public 方法访问
```

### super 关键字

```java
public class Child extends Parent {
    private String name;
    
    public Child(int value, String name) {
        super(value);        // 调用父类构造方法（必须在第一行）
        this.name = name;
    }
    
    @Override
    public void method() {
        super.method();      // 调用父类的方法
        super.field;         // 访问父类的属性（如果子类有同名属性）
    }
    
    // ⚠️ super 与 this 的区别
    // this：当前对象的引用
    // super：父类对象的引用（在子类中使用）
}
```

### final 关键字

```java
// final：最终的、不可改变的

// 1. final 类：不能被继承
public final class Constants { 
    // 工具类，不想被继承
}

// 2. final 方法：不能被重写
public class Base {
    public final void cannotOverride() { 
        // 这个方法不想被子类修改
    }
}

// 3. final 变量：只能赋值一次（常量）
public class Example {
    private final int id = 100;         // 直接初始化
    private final String name;          // 声明
    
    public Example(String name) {
        this.name = name;               // 构造方法中初始化
        // this.name = "other";         // ❌ 编译错误！不能再次赋值
    }
    
    // final 方法的参数
    public void process(final int value) {
        // value = 10;  // ❌ 编译错误！不能修改 final 参数
    }
}

// 4. final 引用变量：引用不能变，但对象内容可以变
final int[] arr = {1, 2, 3};
arr[0] = 10;        // ✓ 可以修改数组元素
// arr = new int[5];  // ❌ 编译错误！不能重新赋值
```

---

## 多态

📌 **多态**是指同一操作作用于不同对象，产生不同的行为。

### 多态的前提

1. 继承或接口实现
2. 方法重写
3. 父类引用指向子类对象

### 向上转型与向下转型

```java
// 向上转型（自动）：子类对象 → 父类引用
// 这就是多态的核心
Animal animal = new Dog("小黄", 3, "金毛");
animal.makeSound();  // 调用子类重写的方法 → 小黄(金毛)汪汪叫

// animal.fetch();  // ❌ 编译错误！
// 原因：编译时看引用类型（Animal），Animal 没有 fetch 方法
// 运行时看实际对象类型（Dog），但编译器不知道

// 向下转型（强制）：父类引用 → 子类引用
if (animal instanceof Dog) {  // 先判断是否是 Dog 类型
    Dog dog = (Dog) animal;    // 强制转换
    dog.fetch();  // ✓ 现在可以调用子类特有方法
}

// Java 16+ 模式匹配（更简洁）
if (animal instanceof Dog dog) {
    dog.fetch();  // 直接使用转换后的变量
}

// ⚠️ 类型转换异常
Animal cat = new Cat("小白", 2, "波斯");
// Dog dog = (Dog) cat;  // 运行时错误！ClassCastException
// 解决方案：使用 instanceof 先判断
```

### 多态的应用场景

```java
// 场景一：统一处理不同类型的对象
public void makeAnimalsSound(Animal[] animals) {
    for (Animal animal : animals) {
        animal.makeSound();  // 多态调用，每个动物发出不同的声音
    }
}

// 使用
Animal[] animals = {
    new Dog("小黄", 3, "金毛"),
    new Cat("小白", 2, "波斯"),
    new Bird("小鸟", 1, "麻雀")
};
makeAnimalsSound(animals);
// 输出：
// 小黄(金毛)汪汪叫
// 小白(波斯)喵喵叫
// 小鸟(麻雀)叽叽喳喳

// 场景二：策略模式
public interface PaymentStrategy {
    void pay(double amount);
}

public class CreditCardPayment implements PaymentStrategy {
    public void pay(double amount) { System.out.println("信用卡支付：" + amount); }
}

public class AlipayPayment implements PaymentStrategy {
    public void pay(double amount) { System.out.println("支付宝支付：" + amount); }
}

// 使用多态，无需知道具体支付方式
public void processPayment(PaymentStrategy strategy, double amount) {
    strategy.pay(amount);
}
```

### 动态绑定原理

```java
// 编译时：检查引用类型是否有该方法
// 运行时：根据实际对象类型调用对应的方法（动态绑定/虚方法调用）

Animal animal = new Dog("小黄", 3, "金毛");
animal.makeSound();

// 编译时：编译器检查 Animal 类是否有 makeSound() 方法 → 有，编译通过
// 运行时：JVM 发现实际对象是 Dog，调用 Dog.makeSound() 方法
```

---

## 抽象类

📌 **抽象类**不能被实例化，可以包含抽象方法（无实现）和具体方法（有实现）。

### 为什么需要抽象类？

当我们想定义一个通用模板，但某些方法的具体实现需要子类来决定时：

```java
// 问题：Animal 的 makeSound() 方法该返回什么？
// 每种动物的叫声都不同，在父类中无法给出统一实现

// ❌ 不好的设计：父类给出默认实现
public class Animal {
    public void makeSound() {
        System.out.println("某种声音");  // 没意义的实现
    }
}

// ✅ 好的设计：使用抽象方法
public abstract class Animal {
    // 抽象方法：只有声明，没有实现（用 abstract 修饰）
    // 子类必须实现这个方法
    public abstract void makeSound();
}
```

### 抽象类详解

```java
// 抽象类：使用 abstract 关键字修饰
public abstract class Shape {
    protected String color;
    
    // 抽象类可以有构造方法（供子类调用）
    public Shape(String color) {
        this.color = color;
    }
    
    // 抽象方法：没有方法体，用 abstract 修饰
    // 子类必须实现所有抽象方法（除非子类也是抽象类）
    public abstract double getArea();    // 计算面积
    public abstract double getPerimeter();  // 计算周长
    
    // 具体方法：抽象类可以有具体实现
    public String getColor() {
        return color;
    }
    
    // 模板方法模式：定义算法骨架，细节由子类实现
    public void printInfo() {
        System.out.println("颜色：" + color + 
                          "，面积：" + getArea() + 
                          "，周长：" + getPerimeter());
    }
}

// 具体子类：必须实现所有抽象方法
public class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);  // 调用父类构造方法
        this.radius = radius;
    }
    
    @Override
    public double getArea() {
        return Math.PI * radius * radius;  // πr²
    }
    
    @Override
    public double getPerimeter() {
        return 2 * Math.PI * radius;  // 2πr
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
    
    @Override
    public double getPerimeter() {
        return 2 * (width + height);
    }
}

// 使用
// Shape shape = new Shape("红色");  // ❌ 编译错误！抽象类不能实例化
Shape circle = new Circle("红色", 5);  // ✓ 向上转型
System.out.println(circle.getArea());  // 多态调用
```

### 抽象类的规则

```java
// 1. 抽象类不能实例化
// Animal a = new Animal();  // ❌ 编译错误

// 2. 抽象类可以有构造方法（供子类调用）
public abstract class Animal {
    public Animal(String name) { }  // ✓ 可以有构造方法
}

// 3. 抽象类可以有非抽象方法
public abstract class Animal {
    public abstract void makeSound();
    public void eat() { }  // ✓ 可以有具体方法
}

// 4. 有抽象方法的类必须是抽象类
// public class Animal {
//     public abstract void makeSound();  // ❌ 编译错误
// }

// 5. 抽象类的子类必须实现所有抽象方法（除非子类也是抽象类）
public abstract class BigCat extends Animal {
    // 可以不实现 makeSound()
}

public class Lion extends BigCat {
    @Override
    public void makeSound() { }  // 最终子类必须实现
}
```

---

## 接口

📌 **接口**是一种纯抽象类型，定义了一组方法签名。Java 8 之后，接口也可以包含默认方法和静态方法。

### 接口 vs 抽象类

| 特性 | 接口 | 抽象类 |
|------|:----:|:------:|
| 多继承 | ✓（可实现多个） | ✗（单继承） |
| 构造方法 | ✗ | ✓ |
| 成员变量 | 只能有常量 | 可有各种类型 |
| 方法实现 | 默认方法、静态方法 | 抽象方法、具体方法 |
| 设计目的 | 定义行为契约 | 代码复用 |

### 接口定义与实现

```java
// 接口定义：使用 interface 关键字
public interface Drawable {
    // 常量：默认 public static final
    double DEFAULT_SIZE = 1.0;
    
    // 抽象方法：默认 public abstract
    void draw();  // 没有方法体
    
    // 默认方法（Java 8+）：有默认实现，子类可以选择重写
    default void printInfo() {
        System.out.println("这是一个可绘制对象");
    }
    
    // 静态方法（Java 8+）：属于接口本身
    static void staticMethod() {
        System.out.println("接口的静态方法");
    }
}

// 实现接口：使用 implements 关键字
public class Circle implements Drawable {
    private double radius;
    
    public Circle(double radius) {
        this.radius = radius;
    }
    
    // 必须实现接口的所有抽象方法
    @Override
    public void draw() {
        System.out.println("绘制半径为 " + radius + " 的圆形");
    }
    
    // 可以选择重写默认方法
    @Override
    public void printInfo() {
        System.out.println("圆形，半径：" + radius);
    }
}

// 使用
Drawable circle = new Circle(5);
circle.draw();      // 调用实现的方法
circle.printInfo(); // 调用重写的默认方法
Drawable.staticMethod();  // 调用接口的静态方法
```

### 多接口实现

Java 单继承多实现：一个类只能继承一个父类，但可以实现多个接口。

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

// 使用
Duck duck = new Duck();
duck.fly();
duck.swim();

// 向上转型为不同接口类型
Flyable f = new Duck();
f.fly();  // ✓
// f.swim();  // ✗ Flyable 没有 swim 方法
```

### 接口继承

接口可以继承多个接口（接口的多继承）：

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

// 实现类需要实现所有继承的抽象方法
public class Impl implements C {
    @Override
    public void methodA() { }
    
    @Override
    public void methodB() { }
    
    @Override
    public void methodC() { }
}
```

### 接口的实际应用

```java
// 策略模式：通过接口定义行为
public interface SortStrategy {
    void sort(int[] arr);
}

public class BubbleSort implements SortStrategy {
    @Override
    public void sort(int[] arr) { /* 冒泡排序 */ }
}

public class QuickSort implements SortStrategy {
    @Override
    public void sort(int[] arr) { /* 快速排序 */ }
}

// 回调模式：通过接口实现回调
public interface Callback {
    void onSuccess(String result);
    void onError(String error);
}

public void doSomething(Callback callback) {
    try {
        // 执行操作
        callback.onSuccess("成功");
    } catch (Exception e) {
        callback.onError("失败：" + e.getMessage());
    }
}

// 依赖注入：通过接口解耦
public interface UserRepository {
    User findById(Long id);
}

public class UserService {
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;  // 注入依赖
    }
}
```

---

## 内部类

📌 **内部类**是定义在另一个类内部的类，可以访问外部类的私有成员。

### 为什么需要内部类？

1. 逻辑上相关的类放在一起
2. 隐藏实现细节
3. 可以访问外部类的私有成员
4. 实现多重继承（每个内部类可以继承不同的类）

### 成员内部类

```java
public class Outer {
    private int outerField = 10;
    
    // 成员内部类：定义在类内部，方法外部
    public class Inner {
        private int innerField = 20;
        
        public void accessOuter() {
            // 内部类可以直接访问外部类的成员（包括私有）
            System.out.println(outerField);  // 10
            
            // 访问自己的成员
            System.out.println(innerField);  // 20
            
            // 显式引用外部类对象
            System.out.println(Outer.this.outerField);  // 10
        }
    }
    
    public void useInner() {
        Inner inner = new Inner();
        inner.accessOuter();
    }
}

// 创建内部类实例
Outer outer = new Outer();
// 外部类实例.new 内部类构造方法
Outer.Inner inner = outer.new Inner();
inner.accessOuter();
```

### 静态内部类

```java
public class Outer {
    private static int staticField = 10;
    private int instanceField = 20;
    
    // 静态内部类：使用 static 修饰
    // 只能访问外部类的静态成员
    public static class StaticInner {
        public void method() {
            System.out.println(staticField);  // ✓ 可以访问静态成员
            // System.out.println(instanceField);  // ✗ 不能访问实例成员
        }
    }
}

// 创建静态内部类实例
// 不需要外部类实例
Outer.StaticInner inner = new Outer.StaticInner();
inner.method();
```

### 匿名内部类

```java
// 匿名内部类：没有名字的内部类，用于一次性使用

interface Greeting {
    void greet();
}

// 方式一：匿名内部类
Greeting greeting = new Greeting() {
    @Override
    public void greet() {
        System.out.println("你好！");
    }
};
greeting.greet();

// 方式二：Lambda 表达式（Java 8+，更简洁）
Greeting greeting2 = () -> System.out.println("你好！");
greeting2.greet();

// 常见用途：事件监听、线程创建
button.addActionListener(new ActionListener() {
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("按钮被点击");
    }
});

// 使用 Lambda 简化
button.addActionListener(e -> System.out.println("按钮被点击"));

// 创建线程
Thread thread = new Thread(new Runnable() {
    @Override
    public void run() {
        System.out.println("线程运行中");
    }
});

// 使用 Lambda 简化
Thread thread2 = new Thread(() -> System.out.println("线程运行中"));
```

---

## 枚举类

📌 **枚举**是一组常量的集合，使用 `enum` 关键字定义。

### 为什么需要枚举？

```java
// ❌ 不好的做法：使用整数常量
public static final int MONDAY = 1;
public static final int TUESDAY = 2;
// ...
int day = 1;
day = 100;  // 编译通过，但值无效

// ✅ 使用枚举
public enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}
Day day = Day.MONDAY;
// day = 100;  // 编译错误！类型不匹配
```

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

// 枚举的常用方法
Day[] days = Day.values();           // 获取所有枚举值
Day day = Day.valueOf("FRIDAY");     // 字符串转枚举
int ordinal = day.ordinal();         // 获取序号（从 0 开始）
String name = day.name();            // 获取名称
int compare = Day.MONDAY.compareTo(Day.FRIDAY);  // 比较
```

### 带字段和方法的枚举

```java
public enum Planet {
    // 枚举值（必须放在最前面）
    MERCURY(3.303e+23, 2.4397e6),
    EARTH(5.976e+24, 6.37814e6),
    MARS(6.421e+23, 3.3972e6);
    
    // 字段（私有，通过构造方法初始化）
    private final double mass;    // 质量（千克）
    private final double radius;  // 半径（米）
    
    // 构造方法（默认私有，不能是 public）
    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
    }
    
    // 方法
    public double getMass() { return mass; }
    public double getRadius() { return radius; }
    
    // 计算表面重力
    public double surfaceGravity() {
        return 6.67e-11 * mass / (radius * radius);
    }
    
    // 计算表面重量
    public double surfaceWeight(double otherMass) {
        return otherMass * surfaceGravity();
    }
}

// 使用
Planet earth = Planet.EARTH;
System.out.println("地球质量：" + earth.getMass());
System.out.println("地球表面重力：" + earth.surfaceGravity());
```

### 枚举实现接口

```java
public interface Operation {
    double apply(double x, double y);
}

public enum Calculator implements Operation {
    ADD("+") {
        @Override
        public double apply(double x, double y) { return x + y; }
    },
    SUBTRACT("-") {
        @Override
        public double apply(double x, double y) { return x - y; }
    },
    MULTIPLY("*") {
        @Override
        public double apply(double x, double y) { return x * y; }
    },
    DIVIDE("/") {
        @Override
        public double apply(double x, double y) { return x / y; }
    };
    
    private final String symbol;
    
    Calculator(String symbol) {
        this.symbol = symbol;
    }
    
    public String getSymbol() { return symbol; }
}

// 使用
double result = Calculator.ADD.apply(5, 3);  // 8.0
```

---

## Object 类

所有类的父类，每个类都直接或间接继承自 `Object`。

### Object 类的核心方法

```java
public class Person {
    private String name;
    private int age;
    
    // 构造方法
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // 1. toString()：返回对象的字符串表示
    // 默认返回：类名@哈希码（如 Person@15db9742）
    // 建议重写，返回有意义的信息
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
    
    // 2. equals()：判断两个对象是否相等
    // 默认比较引用（内存地址），等价于 ==
    // 建议重写，比较内容
    @Override
    public boolean equals(Object obj) {
        // 快速判断：同一对象
        if (this == obj) return true;
        // 判断 null 和类型
        if (obj == null || getClass() != obj.getClass()) return false;
        // 类型转换
        Person person = (Person) obj;
        // 比较内容
        return age == person.age && Objects.equals(name, person.name);
    }
    
    // 3. hashCode()：返回对象的哈希码
    // 相等的对象必须有相同的哈希码
    // 用于 HashMap、HashSet 等数据结构
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
    
    // 4. getClass()：返回对象的运行时类
    // 不需要重写
    // Class<?> clazz = person.getClass();  // Person.class
}

// 使用
Person p1 = new Person("张三", 25);
Person p2 = new Person("张三", 25);

System.out.println(p1.toString());     // Person{name='张三', age=25}
System.out.println(p1.equals(p2));     // true（内容相同）
System.out.println(p1 == p2);          // false（不同对象）
System.out.println(p1.hashCode() == p2.hashCode());  // true
```

### equals() 和 hashCode() 的契约

```java
// equals() 和 hashCode() 必须同时重写！
// 契约：
// 1. 如果 equals() 返回 true，hashCode() 必须相同
// 2. 如果 hashCode() 不同，equals() 必须返回 false

// ⚠️ 只重写 equals() 不重写 hashCode() 的问题
Person p1 = new Person("张三", 25);
Person p2 = new Person("张三", 25);

HashSet<Person> set = new HashSet<>();
set.add(p1);
set.add(p2);  // 如果没重写 hashCode()，两个对象会被都加入 set
              // 因为它们的 hashCode 不同

// 使用 IDE 或 Lombok 自动生成 equals() 和 hashCode()
// Lombok: @Data 或 @EqualsAndHashCode
```

---

## 小结

| 概念 | 说明 | 关键字/语法 |
|------|------|-------------|
| **类** | 对象的模板 | `class` |
| **对象** | 类的实例 | `new` |
| **封装** | 隐藏实现细节 | `private`, `public`, `protected` |
| **继承** | 子类继承父类 | `extends` |
| **多态** | 同一操作不同行为 | 向上转型、重写 |
| **抽象类** | 不能实例化 | `abstract` |
| **接口** | 纯抽象类型 | `interface`, `implements` |
| **内部类** | 类内部的类 | 嵌套定义 |
| **枚举** | 常量集合 | `enum` |

### 面向对象设计原则（SOLID）

1. **单一职责原则**：一个类只做一件事
2. **开闭原则**：对扩展开放，对修改关闭
3. **里氏替换原则**：子类可以替换父类
4. **接口隔离原则**：接口要小而专一
5. **依赖倒置原则**：依赖抽象，不依赖具体