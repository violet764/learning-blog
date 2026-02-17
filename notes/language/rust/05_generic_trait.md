# 泛型与 Trait + 集合类型

> Rust 的泛型系统提供代码复用能力，Trait 定义行为接口，集合类型提供丰富的数据结构。

## 泛型

### 泛型函数

```rust
// 泛型函数：适用于多种类型
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    let result = largest(&numbers);
    println!("最大数: {}", result);

    let chars = vec!['y', 'm', 'a', 'q'];
    let result = largest(&chars);
    println!("最大字符: {}", result);
}
```

### 泛型结构体

```rust
struct Point<T> {
    x: T,
    y: T,
}

struct MixedPoint<T, U> {
    x: T,
    y: U,
}

fn main() {
    let p1 = Point { x: 5, y: 10 };
    let p2 = Point { x: 1.0, y: 4.0 };

    let p3 = MixedPoint { x: 5, y: 4.0 };
}
```

### 泛型枚举

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

enum Option<T> {
    Some(T),
    None,
}

// 自定义泛型枚举
enum BinaryTree<T> {
    Leaf(T),
    Node(BinaryTree<T>, T, BinaryTree<T>),
}
```

### 泛型方法

```rust
struct Point<T> {
    x: T,
    y: T,
}

// 为泛型结构体实现方法
impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

// 仅针对特定类型实现
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

fn main() {
    let p1 = Point::new(5, 10);
    let p2 = Point::new(1.0, 2.0);

    println!("距离: {}", p2.distance_from_origin());
}
```

---

## Trait

### 定义 Trait

Trait 定义共享行为：

```rust
trait Summary {
    fn summarize(&self) -> String;

    // 默认实现
    fn summarize_author(&self) -> String {
        String::from("(Unknown)")
    }
}
```

### 实现 Trait

```rust
struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}, by {}", self.title, self.author)
    }

    fn summarize_author(&self) -> String {
        format!("@{}", self.author)
    }
}

fn main() {
    let article = Article {
        title: String::from("Rust 教程"),
        author: String::from("张三"),
        content: String::from("..."),
    };

    println!("{}", article.summarize());
    println!("{}", article.summarize_author());
}
```

### Trait 作为参数

```rust
fn notify(item: &impl Summary) {
    println!("通知: {}", item.summarize());
}

// 等价于 Trait Bound 语法
fn notify<T: Summary>(item: &T) {
    println!("通知: {}", item.summarize());
}
```

### Trait Bound 语法

```rust
// 单个 Trait
fn foo(item: &impl Summary) {}

// 多个 Trait
fn foo(item: &(impl Summary + Display)) {}

// 语法糖
fn foo<T: Summary>(item: &T) {}

// where 子句（复杂约束）
fn foo<T, U>(t: &T, u: &U) -> String
where
    T: Summary + Display,
    U: Clone + Debug,
{
    format!("{} {}", t.summarize(), u.clone())
}
```

### 常用内置 Trait

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct MyStruct {
    value: i32,
}

// Debug - 调试输出
println!("{:?}", my_struct);

// Clone - 克隆
let clone = my_struct.clone();

// Copy - 按位复制（自动推导）
let copy = my_struct;  // 移动还是复制取决于类型

// PartialEq / Eq - 相等比较
let eq = my_struct == other;

// PartialOrd / Ord - 排序
let ord = my_struct < other;

// Default - 默认值
let default = MyStruct::default();

// Hash - 可哈希
use std::collections::HashSet;
let set: HashSet<MyStruct> = HashSet::new();
```

---

## 集合类型

### Vec<T> 动态数组

```rust
// 创建
let mut v: Vec<i32> = Vec::new();
v.push(1);
v.push(2);
v.push(3);

let v = vec![1, 2, 3];  // 宏创建

// 访问
let third = v[2];           // panic if out of bounds
let third = v.get(2);       // 返回 Option<&T>
let third = v.get(2).unwrap();

// 遍历
for i in &v {
    println!("{}", i);
}

for i in &mut v {
    *i *= 2;
}
```

### Vec 常用操作

```rust
let mut v = vec![1, 2, 3];

v.push(4);           // 添加元素
v.pop();             // 移除最后一个
v.insert(1, 10);     // 插入到索引 1
v.remove(1);         // 移除索引 1 的元素

let slice = &v[1..3];     // 切片

v.len();              // 长度
v.is_empty();         // 是否为空

v.contains(&1);       // 是否包含
v.first();           // 第一个元素
v.last();            // 最后一个元素

v.clear();            // 清空
v.resize(10, 0);     // 调整大小

// 扩展
v.extend([4, 5, 6]);
v.extend_from_slice(&[7, 8, 9]);

// 追加
let mut v1 = vec![1, 2];
let v2 = vec![3, 4];
v1.extend(v2);  // v1: [1, 2, 3, 4]
```

### HashMap<K, V>

```rust
use std::collections::HashMap;

// 创建
let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

// 从迭代器创建
let teams = vec![String::from("Blue"), String::from("Yellow")];
let initial_scores = vec![10, 50];
let scores: HashMap<_, _> = teams.into_iter().zip(initial_scores.into_iter()).collect();

// 访问
let team_name = String::from("Blue");
let score = scores.get(&team_name);  // Option<&V>
let score = scores.get(&team_name).copied().unwrap_or(0);

// 遍历
for (key, value) in &scores {
    println!("{}: {}", key, value);
}

// 修改
scores.insert(String::from("Blue"), 25);  // 覆盖
scores.entry(String::from("Blue")).or_insert(25);  // 仅当不存在时插入

// 更新值
let text = "hello world wonderful world";
let mut word_count = HashMap::new();
for word in text.split_whitespace() {
    let count = word_count.entry(word).or_insert(0);
    *count += 1;
}
```

### HashSet

```rust
use std::collections::HashSet;

let mut set = HashSet::new();
set.insert(1);
set.insert(2);
set.insert(3);

set.contains(&1);      // 是否包含
set.remove(&1);        // 移除

// 集合操作
let a = vec![1, 2, 3].into_iter().collect::<HashSet<_>>();
let b = vec![2, 3, 4].into_iter().collect::<HashSet<_>>();

let union: HashSet<_> = a.union(&b).collect();       // 并集
let intersection: HashSet<_> = a.intersection(&b).collect();  // 交集
let difference: HashSet<_> = a.difference(&b).collect();       // 差集
```

### 其他集合类型

```rust
// LinkedList - 双端链表
use std::collections::LinkedList;
let mut list = LinkedList::new();
list.push_back(1);
list.push_front(0);

// VecDeque - 双端队列
use std::collections::VecDeque;
let mut deque = VecDeque::new();
deque.push_front(1);
deque.push_back(2);

// BinaryHeap - 最大堆
use std::collections::BinaryHeap;
let mut heap = BinaryHeap::new();
heap.push(3);
heap.push(1);
heap.push(5);
println!("{}", heap.pop());  // 5
```

---

## 迭代器

### 迭代器基础

```rust
let v = vec![1, 2, 3];
let mut iter = v.iter();

// next() 返回 Option<&T>
println!("{:?}", iter.next());  // Some(&1)
println!("{:?}", iter.next());  // Some(&2)
println!("{:?}", iter.next());  // Some(&3)
println!("{:?}", iter.next());  // None

// 遍历
for val in v.iter() {
    println!("{}", val);
}

// 转换为所有权迭代器
for val in v.into_iter() {
    println!("{}", val);  // val 现在是 i32，不是 &i32
}

// 可变迭代器
for val in v.iter_mut() {
    *val *= 2;
}
```

### 迭代器适配器

```rust
let v = vec![1, 2, 3, 4, 5];

// map - 转换
let v2: Vec<_> = v.iter().map(|x| x * 2).collect();

// filter - 过滤
let v3: Vec<_> = v.iter().filter(|x| *x > 2).collect();

// take/skip
let v4: Vec<_> = v.iter().take(3).collect();  // [1, 2, 3]
let v5: Vec<_> = v.iter().skip(3).collect();  // [4, 5]

// enumerate - 带索引
let v6: Vec<_> = v.iter().enumerate().collect();

// chain - 连接
let a = vec![1, 2];
let b = vec![3, 4];
let c: Vec<_> = a.iter().chain(b.iter()).collect();

// zip - 配对
let a = vec![1, 2, 3];
let b = vec![4, 5, 6];
let c: Vec<_> = a.iter().zip(b.iter()).collect();  // [(1,4), (2,5), (3,6)]
```

### 消费者

```rust
let v = vec![1, 2, 3, 4, 5];

// collect - 收集为集合
let sum: i32 = v.iter().sum();
let product: i32 = v.iter().product();

// fold - 折叠
let sum = v.iter().fold(0, |acc, x| acc + x);

// find - 查找第一个
let found = v.iter().find(|&&x| x > 3);  // Some(&4)

// position - 查找位置
let pos = v.iter().position(|&x| x == 3);  // Some(2)

// any/all - 判断
let any_big = v.iter().any(|&x| x > 10);  // false
let all_big = v.iter().all(|&x| x > 0);   // true
```

---

## 小结

### 泛型
1. **泛型函数**：适用于多种类型的函数
2. **泛型结构体/枚举**：参数化类型
3. **泛型方法**：为泛型实现方法

### Trait
1. **定义行为**：类似接口
2. **默认实现**：提供默认行为
3. **Trait Bound**：约束泛型类型
4. **常用内置 Trait**：Debug、Clone、Copy、Eq、Ord 等

### 集合类型
1. **Vec<T>**：动态数组，最常用
2. **HashMap<K, V>**：键值对映射
3. **HashSet<T>**：无序集合
4. **迭代器**：惰性操作的函数式风格

### 迭代器
1. **迭代器适配器**：map、filter、take 等（惰性）
2. **消费者**：collect、sum、fold 等（消费迭代器）

下一章节我们将学习模块系统、包管理和并发编程。
