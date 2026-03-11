# Java 集合框架

> Java 集合框架（Collections Framework）提供了一套高性能、高质量的数据结构实现，是 Java 中最常用的 API 之一。理解集合框架对于编写高效的 Java 程序至关重要。

## 集合

### 数组的局限性

在学习集合之前，先理解数组的局限性：

```java
// 数组的问题：
// 1. 长度固定，无法动态扩容
int[] arr = new int[5];
// arr.add(6);  // 无法添加第6个元素

// 2. 只能存储同一类型
// 3. 删除元素很麻烦（需要移动后面的元素）
// 4. 查找元素效率低（需要遍历）

// 集合的优势：
// 1. 长度可变，自动扩容
// 2. 提供丰富的操作方法
// 3. 不同的数据结构适应不同场景
// 4. 高效的查找、插入、删除操作
```

---

## 集合框架概览

```
Iterable（可迭代接口）
    │
    └── Collection（单值集合根接口）
        ├── List（有序、可重复）
        │   ├── ArrayList   → 动态数组，随机访问快
        │   ├── LinkedList  → 双向链表，插入删除快
        │   └── Vector      → 线程安全的 ArrayList（已过时）
        │
        ├── Set（无序、不可重复）
        │   ├── HashSet        → 哈希表，最快
        │   ├── LinkedHashSet  → 保持插入顺序
        │   └── TreeSet        → 红黑树，有序
        │
        └── Queue（队列）
            ├── LinkedList     → 普通队列
            ├── PriorityQueue  → 优先队列（堆）
            └── ArrayDeque     → 双端队列

Map（键值对集合，不继承 Collection）
├── HashMap        → 哈希表，最快
├── LinkedHashMap  → 保持插入/访问顺序
├── TreeMap        → 红黑树，有序
└── Hashtable      → 线程安全（遗留类，不推荐）
```

### 如何选择集合？

| 需求 | 推荐实现 | 原因 |
|------|----------|------|
| 频繁随机访问（通过索引获取） | `ArrayList` | 底层是数组，$O(1)$ 访问 |
| 频繁在头部/中间插入删除 | `LinkedList` | 底层是链表，$O(1)$ 插入删除 |
| 元素去重，不关心顺序 | `HashSet` | 哈希表，$O(1)$ 操作 |
| 元素去重，保持插入顺序 | `LinkedHashSet` | 链表 + 哈希表 |
| 元素去重，需要排序 | `TreeSet` | 红黑树，自动排序 |
| 键值对存储，不关心顺序 | `HashMap` | 哈希表，$O(1)$ 操作 |
| 键值对存储，保持顺序 | `LinkedHashMap` | 链表 + 哈希表 |
| 键值对存储，需要排序 | `TreeMap` | 红黑树，自动排序 |
| 优先级队列（堆） | `PriorityQueue` | 堆结构 |
| 栈操作 | `ArrayDeque` | 比 Stack 更高效 |
| 线程安全 | `ConcurrentHashMap` | 并发安全 |

---

## List 列表

📌 **List** 是有序集合，允许重复元素，可以通过索引访问。

### ArrayList

**底层结构**：动态数组（Object[]）

**特点**：
- 随机访问快：$O(1)$（通过索引直接访问）
- 插入删除慢：$O(n)$（需要移动元素）
- 末尾插入删除快：$O(1)$（均摊）

```java
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

// 创建 ArrayList
List<String> list = new ArrayList<>();           // 空列表，默认容量 10
List<String> list2 = new ArrayList<>(20);        // 指定初始容量
List<String> list3 = new ArrayList<>(Arrays.asList("A", "B", "C"));  // 从集合创建

// ========== 添加元素 ==========
list.add("张三");                    // 添加到末尾
list.add(0, "李四");                 // 在索引 0 处插入（后面的元素后移）
list.addAll(Arrays.asList("王五", "赵六"));  // 添加多个元素

// ⚠️ 在中间插入元素效率低
for (int i = 0; i < 10000; i++) {
    list.add(0, "item" + i);  // 每次都要移动所有元素，效率低！
}

// ========== 访问元素 ==========
String first = list.get(0);          // 通过索引访问，O(1)
int index = list.indexOf("张三");    // 查找元素索引，O(n)
int lastIdx = list.lastIndexOf("张三");  // 从后往前找

// ========== 修改元素 ==========
list.set(0, "更新后的值");            // 替换指定位置的元素

// ========== 删除元素 ==========
list.remove(0);                      // 按索引删除
list.remove("张三");                  // 按值删除（只删除第一个匹配的）
list.removeAll(Arrays.asList("A", "B"));  // 删除多个元素
list.clear();                        // 清空列表

// ⚠️ 删除元素时的陷阱
List<Integer> nums = new ArrayList<>(Arrays.asList(1, 2, 3));
nums.remove(1);     // 删除索引 1 的元素（值为 2），结果 [1, 3]
// nums.remove(Integer.valueOf(1));  // 删除值为 1 的元素，结果 [2, 3]

// ========== 大小与判断 ==========
int size = list.size();
boolean empty = list.isEmpty();
boolean contains = list.contains("张三");  // 是否包含某元素

// ========== 遍历方式 ==========
// 方式一：普通 for 循环（适合需要索引的场景）
for (int i = 0; i < list.size(); i++) {
    System.out.println(i + ": " + list.get(i));
}

// 方式二：增强 for 循环（最常用，简洁）
for (String s : list) {
    System.out.println(s);
}

// 方式三：Lambda 表达式（Java 8+）
list.forEach(s -> System.out.println(s));
list.forEach(System.out::println);  // 方法引用，更简洁

// 方式四：迭代器（适合需要删除元素的场景）
Iterator<String> it = list.iterator();
while (it.hasNext()) {
    String s = it.next();
    if (s.equals("删除")) {
        it.remove();  // 安全删除当前元素
    }
}

// ⚠️ 遍历时删除元素的陷阱
// for (String s : list) {
//     if (s.equals("删除")) {
//         list.remove(s);  // 运行时错误！ConcurrentModificationException
//     }
// }
// 原因：增强 for 循环使用迭代器，迭代过程中修改集合会抛出异常
```

### LinkedList

**底层结构**：双向链表

**特点**：
- 随机访问慢：$O(n)$（需要遍历链表）
- 插入删除快：$O(1)$（只需修改指针）
- 可以作为队列、栈、列表使用

```java
import java.util.LinkedList;

LinkedList<String> list = new LinkedList<>();

// ========== List 方法 ==========
list.add("A");                       // 添加到末尾
list.add(0, "First");                // 在索引 0 处插入
list.addFirst("头部");               // 添加到头部（比 ArrayList 高效）
list.addLast("尾部");                // 添加到尾部

String first = list.getFirst();      // 获取头部元素
String last = list.getLast();        // 获取尾部元素

list.removeFirst();                  // 删除头部元素
list.removeLast();                   // 删除尾部元素

// ========== 队列操作（FIFO：先进先出）==========
list.offer("尾部");                  // 入队（添加到尾部）
String head = list.poll();           // 出队（删除并返回头部元素，队空返回 null）
String peek = list.peek();           // 查看队头元素（不删除，队空返回 null）

// ========== 栈操作（LIFO：后进先出）==========
list.push("栈顶");                   // 入栈（添加到头部）
String top = list.pop();             // 出栈（删除并返回头部元素）

// ⚠️ LinkedList vs ArrayList 选择
// - 频繁在头部/中间插入删除 → LinkedList
// - 频繁随机访问 → ArrayList
// - 大多数情况 ArrayList 更好（内存连续，CPU 缓存友好）
```

### List 转数组

```java
List<Integer> list = Arrays.asList(1, 2, 3);

// 方式一：toArray(T[] array)
Integer[] arr1 = list.toArray(new Integer[0]);  // 传入空数组，自动创建合适大小
Integer[] arr2 = list.toArray(new Integer[list.size()]);  // 传入精确大小

// 方式二：stream（Java 8+）
int[] arr3 = list.stream().mapToInt(Integer::intValue).toArray();

// ⚠️ 数组转 List
String[] strArr = {"A", "B", "C"};
List<String> strList = Arrays.asList(strArr);  // 固定大小的 List
// strList.add("D");  // 运行时错误！UnsupportedOperationException

// 如果需要可修改的 List
List<String> mutableList = new ArrayList<>(Arrays.asList(strArr));
mutableList.add("D");  // ✓ 可以添加
```

---

## Set 集合

📌 **Set** 是不可重复的集合，不保证顺序（除 LinkedHashSet 和 TreeSet）。

### HashSet

**底层结构**：哈希表（数组 + 链表/红黑树）

**特点**：
- 添加、删除、查找：$O(1)$
- 元素无序
- 允许 null 元素
- 线程不安全

**重要**：存入 HashSet 的对象必须正确实现 `hashCode()` 和 `equals()` 方法！

```java
import java.util.HashSet;
import java.util.Set;

Set<String> set = new HashSet<>();

// ========== 添加元素 ==========
set.add("A");
set.add("B");
set.add("A");  // 重复元素不会被添加（add 返回 false）
System.out.println(set.size());  // 2

// ========== 判断与查找 ==========
boolean contains = set.contains("A");  // true
boolean isEmpty = set.isEmpty();

// ========== 删除元素 ==========
set.remove("A");
set.clear();

// ========== 遍历 ==========
for (String s : set) {
    System.out.println(s);  // 顺序不确定
}

// ⚠️ HashSet 如何判断元素重复？
// 1. 先比较 hashCode()
// 2. 如果 hashCode 相同，再比较 equals()
// 3. 都相同才认为是重复元素

// 自定义类存入 HashSet 必须重写 hashCode 和 equals
public class Person {
    private String name;
    private int age;
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Person person = (Person) o;
        return age == person.age && Objects.equals(name, person.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
}
```

### LinkedHashSet

**底层结构**：哈希表 + 双向链表

**特点**：
- 继承 HashSet，功能相同
- 额外维护一个双向链表，保持插入顺序
- 性能略低于 HashSet

```java
import java.util.LinkedHashSet;

Set<String> set = new LinkedHashSet<>();
set.add("C");
set.add("A");
set.add("B");

// 遍历顺序：C, A, B（按插入顺序，而不是字母顺序）
for (String s : set) {
    System.out.println(s);
}
```

### TreeSet

**底层结构**：红黑树（自平衡二叉搜索树）

**特点**：
- 元素自动排序（自然排序或自定义排序）
- 添加、删除、查找：$O(\log n)$
- 元素必须实现 `Comparable` 接口，或提供 `Comparator`

```java
import java.util.TreeSet;
import java.util.Comparator;

// 自然排序（元素实现 Comparable）
TreeSet<Integer> nums = new TreeSet<>();
nums.add(5);
nums.add(1);
nums.add(3);
// 遍历顺序：1, 3, 5（自动排序）

// 自定义排序（传入 Comparator）
TreeSet<String> words = new TreeSet<>((a, b) -> b.compareTo(a));  // 降序
words.add("Apple");
words.add("Banana");
// 遍历顺序：Banana, Apple

// ========== TreeSet 特有方法 ==========
Integer first = nums.first();           // 最小值：1
Integer last = nums.last();             // 最大值：5
Integer lower = nums.lower(3);          // 小于 3 的最大值：1
Integer higher = nums.higher(3);        // 大于 3 的最小值：5
Integer floor = nums.floor(3);          // 小于等于 3 的最大值：3
Integer ceiling = nums.ceiling(3);      // 大于等于 3 的最小值：3

// 范围查询
TreeSet<Integer> subset = new TreeSet<>(nums.subSet(1, 5));  // [1, 5)
TreeSet<Integer> headSet = new TreeSet<>(nums.headSet(3));   // 小于 3 的元素
TreeSet<Integer> tailSet = new TreeSet<>(nums.tailSet(3));   // 大于等于 3 的元素

// ⚠️ TreeSet 元素必须可比较
// TreeSet<Person> persons = new TreeSet<>();
// persons.add(new Person("张三", 25));  // 运行时错误！Person 没有实现 Comparable

// 解决方案一：实现 Comparable
public class Person implements Comparable<Person> {
    @Override
    public int compareTo(Person o) {
        return this.name.compareTo(o.name);  // 按姓名排序
    }
}

// 解决方案二：传入 Comparator
TreeSet<Person> persons = new TreeSet<>(Comparator.comparing(Person::getName));
```

---

## Map 映射

📌 **Map** 存储键值对（Key-Value），键不能重复，每个键最多映射一个值。

### HashMap

**底层结构**：数组 + 链表/红黑树（Java 8+）

**特点**：
- 添加、删除、查找：$O(1)$
- 键无序
- 允许 null 键和 null 值
- 线程不安全

**重要**：作为键的对象必须正确实现 `hashCode()` 和 `equals()` 方法！

```java
import java.util.HashMap;
import java.util.Map;

Map<String, Integer> map = new HashMap<>();           // 默认容量 16，负载因子 0.75
Map<String, Integer> map2 = new HashMap<>(32);        // 指定初始容量
Map<String, Integer> map3 = new HashMap<>(32, 0.5f);  // 指定容量和负载因子

// ========== 添加元素 ==========
map.put("张三", 25);               // 添加键值对
map.put("李四", 30);
map.put("张三", 26);               // 键已存在，会覆盖旧值，返回旧值 25

map.putIfAbsent("张三", 28);       // 键不存在时才添加（不会覆盖）

// ========== 访问元素 ==========
int age = map.get("张三");                 // 26
int age2 = map.getOrDefault("王五", 0);    // 0（键不存在，返回默认值）

// ========== 修改元素 ==========
map.replace("张三", 27);           // 替换指定键的值
map.replace("张三", 26, 27);       // 只有旧值匹配时才替换

// compute：根据旧值计算新值
map.compute("张三", (k, v) -> v == null ? 0 : v + 1);  // 年龄 + 1

// merge：合并值
map.merge("张三", 1, Integer::sum);  // 如果键存在，用函数合并；不存在则添加

// ========== 删除元素 ==========
map.remove("张三");               // 按键删除
map.remove("李四", 30);           // 键值都匹配才删除
map.clear();                      // 清空

// ========== 判断 ==========
boolean hasKey = map.containsKey("张三");
boolean hasValue = map.containsValue(25);
int size = map.size();
boolean empty = map.isEmpty();

// ⚠️ HashMap 的容量和负载因子
// 初始容量：默认 16
// 负载因子：默认 0.75（当元素数量 > 容量 * 负载因子时，扩容为原来的 2 倍）
// 建议：如果知道元素数量，指定初始容量避免频繁扩容
Map<String, Integer> map4 = new HashMap<>((int)(expectedSize / 0.75) + 1);
```

### Map 遍历

```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
map.put("B", 2);
map.put("C", 3);

// ========== 遍历键 ==========
for (String key : map.keySet()) {
    System.out.println(key);
}

// ========== 遍历值 ==========
for (Integer value : map.values()) {
    System.out.println(value);
}

// ========== 遍历键值对（推荐）==========
// 方式一：entrySet()
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

// 方式二：forEach + Lambda（Java 8+）
map.forEach((k, v) -> System.out.println(k + ": " + v));

// ========== 获取所有键/值 ==========
Set<String> keys = map.keySet();            // 所有键的 Set
Collection<Integer> values = map.values();  // 所有值的 Collection
Set<Map.Entry<String, Integer>> entries = map.entrySet();  // 所有键值对
```

### LinkedHashMap

**底层结构**：哈希表 + 双向链表

**特点**：
- 继承 HashMap，功能相同
- 可以保持插入顺序或访问顺序
- 可用于实现 LRU 缓存

```java
import java.util.LinkedHashMap;

// 保持插入顺序（默认）
Map<String, Integer> map = new LinkedHashMap<>();
map.put("C", 3);
map.put("A", 1);
map.put("B", 2);
// 遍历顺序：C=3, A=1, B=2（按插入顺序）

// 保持访问顺序（LRU 缓存）
// 参数：初始容量、负载因子、accessOrder=true 表示按访问顺序排序
Map<String, Integer> lru = new LinkedHashMap<>(16, 0.75f, true);
lru.put("A", 1);
lru.put("B", 2);
lru.put("C", 3);
lru.get("A");  // 访问 A 后，A 移到最后
// 遍历顺序：B=2, C=3, A=1（按访问顺序，最近访问的在最后）

// 实现 LRU 缓存（重写 removeEldestEntry 方法）
class LRUCache<K, V> extends LinkedHashMap<K, V> {
    private final int maxSize;
    
    public LRUCache(int maxSize) {
        super(maxSize, 0.75f, true);
        this.maxSize = maxSize;
    }
    
    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > maxSize;  // 超过最大容量时删除最久未访问的元素
    }
}
```

### TreeMap

**底层结构**：红黑树

**特点**：
- 键自动排序
- 添加、删除、查找：$O(\log n)$
- 键必须实现 `Comparable` 或提供 `Comparator`

```java
import java.util.TreeMap;

TreeMap<String, Integer> map = new TreeMap<>();
map.put("C", 3);
map.put("A", 1);
map.put("B", 2);
// 键顺序：A=1, B=2, C=3（自动按键排序）

// ========== TreeMap 特有方法 ==========
String firstKey = map.firstKey();            // 最小键："A"
String lastKey = map.lastKey();              // 最大键："C"
String lowerKey = map.lowerKey("B");         // 小于 "B" 的最大键："A"
String higherKey = map.higherKey("B");       // 大于 "B" 的最小键："C"

// 范围查询
Map<String, Integer> subMap = map.subMap("A", "C");  // ["A", "C")：A=1, B=2
Map<String, Integer> headMap = map.headMap("B");     // 小于 "B" 的：A=1
Map<String, Integer> tailMap = map.tailMap("B");     // 大于等于 "B" 的：B=2, C=3
```

---

## Queue 队列

📌 **Queue** 是先进先出（FIFO）的数据结构。

### PriorityQueue

**底层结构**：堆（完全二叉树，用数组实现）

**特点**：
- 元素按优先级出队（默认最小堆，最小元素先出）
- 添加、删除：$O(\log n)$
- 查看队头：$O(1)$
- 不允许 null 元素

```java
import java.util.PriorityQueue;
import java.util.Queue;

// 默认最小堆（最小元素先出队）
Queue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(5);
minHeap.offer(1);
minHeap.offer(3);

// 出队顺序：1, 3, 5（从小到大）
while (!minHeap.isEmpty()) {
    System.out.println(minHeap.poll());
}

// 最大堆（最大元素先出队）
Queue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
// 或使用 Comparator.reverseOrder()
Queue<Integer> maxHeap2 = new PriorityQueue<>(Comparator.reverseOrder());

maxHeap.offer(5);
maxHeap.offer(1);
maxHeap.offer(3);
// 出队顺序：5, 3, 1（从大到小）

// ========== 常用方法 ==========
minHeap.offer(4);       // 入队（添加元素）
Integer top = minHeap.peek();   // 查看队头（不删除），队空返回 null
Integer val = minHeap.poll();   // 出队（删除并返回队头），队空返回 null

// ⚠️ PriorityQueue 的遍历顺序不是有序的！
// 遍历时只是按数组顺序，不是按优先级
PriorityQueue<Integer> pq = new PriorityQueue<>(Arrays.asList(3, 1, 2));
for (int n : pq) {
    System.out.println(n);  // 可能输出 1, 3, 2 或其他顺序
}
// 正确的有序遍历方式：逐个 poll
while (!pq.isEmpty()) {
    System.out.println(pq.poll());  // 1, 2, 3
}
```

### ArrayDeque

**底层结构**：可扩容的环形数组

**特点**：
- 双端队列，可作为栈或队列使用
- 添加、删除：$O(1)$
- 比 `Stack`（已过时）和 `LinkedList` 更高效
- 不允许 null 元素

```java
import java.util.ArrayDeque;
import java.util.Deque;

Deque<Integer> deque = new ArrayDeque<>();

// ========== 栈操作（LIFO：后进先出）==========
deque.push(1);          // 入栈（添加到头部）
deque.push(2);          // 栈：[2, 1]
deque.push(3);          // 栈：[3, 2, 1]

int top = deque.peek();     // 查看栈顶：3
int val = deque.pop();      // 出栈：3，栈变为 [2, 1]

// ========== 队列操作（FIFO：先进先出）==========
deque.offer(1);         // 入队（添加到尾部）
deque.offer(2);         // 队列：[1, 2]
deque.offer(3);         // 队列：[1, 2, 3]

int head = deque.peek();    // 查看队头：1
int val2 = deque.poll();     // 出队：1，队列变为 [2, 3]

// ========== 双端操作 ==========
deque.addFirst(0);      // 头部添加
deque.addLast(3);       // 尾部添加
deque.removeFirst();    // 头部删除
deque.removeLast();     // 尾部删除

// ⚠️ ArrayDeque vs LinkedList
// ArrayDeque：基于数组，内存连续，更适合栈和队列
// LinkedList：基于链表，更适合频繁在中间插入删除
```

---

## 集合工具类

### Arrays

```java
import java.util.Arrays;

int[] arr = {5, 2, 8, 1, 9};

// 排序
Arrays.sort(arr);                       // [1, 2, 5, 8, 9]

// 二分查找（数组必须有序）
int index = Arrays.binarySearch(arr, 5); // 返回索引 2

// 填充
Arrays.fill(arr, 0);                    // [0, 0, 0, 0, 0]

// 复制
int[] copy = Arrays.copyOf(arr, 3);     // 复制前 3 个元素
int[] range = Arrays.copyOfRange(arr, 1, 4);  // 复制索引 1-3 的元素

// 转字符串
String str = Arrays.toString(arr);      // "[1, 2, 5, 8, 9]"

// 比较
int[] arr2 = {1, 2, 5, 8, 9};
boolean same = Arrays.equals(arr, arr2);  // true

// 多维数组
int[][] matrix = {{1, 2}, {3, 4}};
System.out.println(Arrays.deepToString(matrix));  // "[[1, 2], [3, 4]]"
```

### Collections

```java
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

List<Integer> list = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5));

// 排序
Collections.sort(list);                      // 升序：[1, 1, 3, 4, 5]
Collections.sort(list, (a, b) -> b - a);    // 降序：[5, 4, 3, 1, 1]

// 反转
Collections.reverse(list);                   // [5, 4, 3, 1, 1]

// 随机打乱
Collections.shuffle(list);

// 查找
int max = Collections.max(list);             // 最大值
int min = Collections.min(list);             // 最小值
int freq = Collections.frequency(list, 1);  // 元素 1 出现的次数

// 替换
Collections.replaceAll(list, 1, 10);         // 把所有 1 替换为 10

// 旋转
Collections.rotate(list, 2);                 // 向右旋转 2 位

// 交换
Collections.swap(list, 0, 1);                // 交换索引 0 和 1 的元素

// 不可变集合
List<Integer> unmodifiable = Collections.unmodifiableList(list);
// unmodifiable.add(6);  // 运行时错误！UnsupportedOperationException

// 空集合（不可变，节省内存）
List<String> emptyList = Collections.emptyList();
Set<String> emptySet = Collections.emptySet();
Map<String, String> emptyMap = Collections.emptyMap();

// 单元素集合（不可变）
List<String> singletonList = Collections.singletonList("only");
Set<String> singletonSet = Collections.singleton("only");
```

---

## 时间复杂度总结

| 集合类型 | 操作 | ArrayList | LinkedList | HashSet | TreeSet |
|----------|------|:---------:|:----------:|:-------:|:-------:|
| **添加** | add(e) | $O(1)$* | $O(1)$ | $O(1)$ | $O(\log n)$ |
| **添加** | add(index, e) | $O(n)$ | $O(1)$** | - | - |
| **获取** | get(index) | $O(1)$ | $O(n)$ | - | - |
| **查找** | contains(e) | $O(n)$ | $O(n)$ | $O(1)$ | $O(\log n)$ |
| **删除** | remove(e) | $O(n)$ | $O(n)$ | $O(1)$ | $O(\log n)$ |

*均摊时间复杂度
**需要先找到插入位置

---

## 小结

| 类型 | 实现类 | 底层结构 | 适用场景 |
|------|--------|----------|----------|
| **List** | ArrayList | 动态数组 | 随机访问多，末尾插入删除 |
| **List** | LinkedList | 双向链表 | 头部/中间插入删除多 |
| **Set** | HashSet | 哈希表 | 去重，不关心顺序 |
| **Set** | TreeSet | 红黑树 | 去重，需要排序 |
| **Map** | HashMap | 哈希表 | 键值对，不关心顺序 |
| **Map** | TreeMap | 红黑树 | 键值对，需要按键排序 |
| **Queue** | PriorityQueue | 堆 | 优先级队列 |
| **Deque** | ArrayDeque | 环形数组 | 栈、双端队列 |

### 常见错误

1. **遍历时删除元素**：使用迭代器或 Java 8+ 的 `removeIf`
2. **HashSet/HashMap 的键没有重写 hashCode 和 equals**：导致重复元素
3. **TreeSet/TreeMap 的元素没有实现 Comparable**：运行时抛出异常
4. **Arrays.asList 返回的 List 不能添加元素**：需要包装为 `new ArrayList<>()`
5. **HashMap 容量设置不当**：频繁扩容影响性能