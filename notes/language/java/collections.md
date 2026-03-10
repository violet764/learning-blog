# Java 集合框架

> Java 集合框架（Collections Framework）提供了一套高性能、高质量的数据结构实现，是 Java 中最常用的 API 之一。理解集合框架对于编写高效的 Java 程序至关重要。

## 集合框架概览

```
Collection（单值集合）
├── List（有序、可重复）
│   ├── ArrayList   → 动态数组，随机访问快
│   ├── LinkedList  → 双向链表，插入删除快
│   └── Vector      → 线程安全的 ArrayList
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

Map（键值对集合）
├── HashMap        → 哈希表，最快
├── LinkedHashMap  → 保持插入顺序
├── TreeMap        → 红黑树，有序
└── Hashtable      → 线程安全（遗留类）
```

---

## List 列表

📌 **List** 是有序集合，允许重复元素，可以通过索引访问。

### ArrayList

基于动态数组实现，随机访问快（$O(1)$），插入删除慢（$O(n)$）。

```java
import java.util.ArrayList;
import java.util.List;

List<String> list = new ArrayList<>();

// 添加元素
list.add("张三");
list.add(0, "李四");      // 在索引 0 处插入
list.addAll(Arrays.asList("王五", "赵六"));

// 访问元素
String first = list.get(0);
int index = list.indexOf("张三");

// 修改元素
list.set(0, "更新");

// 删除元素
list.remove(0);              // 按索引删除
list.remove("张三");          // 按值删除
list.clear();                // 清空

// 大小与判断
int size = list.size();
boolean empty = list.isEmpty();
boolean contains = list.contains("张三");

// 遍历
for (String s : list) {
    System.out.println(s);
}
list.forEach(System.out::println);  // Lambda
```

### LinkedList

基于双向链表实现，插入删除快（$O(1)$），随机访问慢（$O(n)$）。

```java
LinkedList<String> list = new LinkedList<>();

// List 方法
list.add("A");
list.addFirst("First");    // 添加到头部
list.addLast("Last");      // 添加到尾部

// 队列操作
list.offer("尾部");         // 入队
list.poll();               // 出队
list.peek();               // 查看队头

// 栈操作
list.push("栈顶");          // 入栈
list.pop();                // 出栈
```

### List 转数组

```java
List<Integer> list = Arrays.asList(1, 2, 3);

// 方式1：toArray
Integer[] arr1 = list.toArray(new Integer[0]);

// 方式2：stream（Java 8+）
int[] arr2 = list.stream().mapToInt(Integer::intValue).toArray();
```

---

## Set 集合

📌 **Set** 是不可重复的集合，不保证顺序（除 LinkedHashSet 和 TreeSet）。

### HashSet

基于哈希表实现，元素无序，效率最高。要求元素正确实现 `hashCode()` 和 `equals()`。

```java
Set<String> set = new HashSet<>();

set.add("A");
set.add("B");
set.add("A");           // 重复元素不会被添加
System.out.println(set.size());  // 2

set.remove("A");
boolean contains = set.contains("B");

// 遍历
for (String s : set) {
    System.out.println(s);
}
```

### LinkedHashSet

保持元素插入顺序。

```java
Set<String> set = new LinkedHashSet<>();
set.add("C");
set.add("A");
set.add("B");
// 遍历顺序：C, A, B（按插入顺序）
```

### TreeSet

基于红黑树实现，元素有序（自然排序或自定义排序）。要求元素实现 `Comparable` 或提供 `Comparator`。

```java
TreeSet<Integer> nums = new TreeSet<>();
nums.add(5);
nums.add(1);
nums.add(3);
// 顺序：1, 3, 5

// 自定义排序（降序）
TreeSet<String> words = new TreeSet<>((a, b) -> b.compareTo(a));

// 特有方法
Integer first = nums.first();           // 最小值
Integer last = nums.last();             // 最大值
Integer lower = nums.lower(3);          // 小于 3 的最大值
Integer higher = nums.higher(3);        // 大于 3 的最小值

// 范围操作
TreeSet<Integer> subset = new TreeSet<>(nums.subSet(1, 5));  // [1, 5)
```

---

## Map 映射

📌 **Map** 存储键值对，键不能重复，每个键最多映射一个值。

### HashMap

基于哈希表实现，键无序，效率最高。

```java
import java.util.HashMap;
import java.util.Map;

Map<String, Integer> map = new HashMap<>();

// 添加
map.put("张三", 25);
map.put("李四", 30);
map.putIfAbsent("张三", 28);    // 键不存在时才添加

// 访问
int age = map.get("张三");              // 25
int age2 = map.getOrDefault("王五", 0);  // 0（默认值）

// 修改
map.put("张三", 26);                   // 覆盖旧值
map.replace("张三", 27);               // 替换

// 删除
map.remove("张三");
map.remove("李四", 30);    // 键值都匹配才删除

// 判断
boolean hasKey = map.containsKey("张三");
boolean hasValue = map.containsValue(25);
int size = map.size();
```

### Map 遍历

```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
map.put("B", 2);

// 遍历键
for (String key : map.keySet()) {
    System.out.println(key);
}

// 遍历值
for (Integer value : map.values()) {
    System.out.println(value);
}

// 遍历键值对（推荐）
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

// Lambda 遍历
map.forEach((k, v) -> System.out.println(k + ": " + v));
```

### LinkedHashMap

保持键的插入顺序或访问顺序（可用于 LRU 缓存）。

```java
// 插入顺序（默认）
Map<String, Integer> map = new LinkedHashMap<>();

// 访问顺序（LRU 缓存）
Map<String, Integer> lru = new LinkedHashMap<>(16, 0.75f, true);
lru.put("A", 1);
lru.put("B", 2);
lru.get("A");  // 访问后，A 移到最后
// 顺序变为：B, A
```

### TreeMap

键有序（自然排序或自定义排序）。

```java
TreeMap<String, Integer> map = new TreeMap<>();
map.put("C", 3);
map.put("A", 1);
map.put("B", 2);
// 键顺序：A=1, B=2, C=3

// 特有方法
String firstKey = map.firstKey();
String lastKey = map.lastKey();
String lowerKey = map.lowerKey("B");     // "A"
String higherKey = map.higherKey("B");   // "C"

// 范围操作
Map<String, Integer> subMap = map.subMap("A", "C");  // [A, C)
```

---

## Queue 队列

📌 **Queue** 是先进先出（FIFO）的数据结构。

### PriorityQueue

优先队列，元素按优先级出队（默认最小堆）。

```java
import java.util.PriorityQueue;
import java.util.Queue;

// 最小堆（默认）
Queue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(5);
minHeap.offer(1);
minHeap.offer(3);
// 出队顺序：1, 3, 5

// 最大堆
Queue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

// 常用方法
minHeap.offer(4);       // 入队
int top = minHeap.peek();    // 查看队头（不删除）
int val = minHeap.poll();    // 出队（删除并返回）
```

### ArrayDeque

双端队列，可作为栈或队列使用，效率高于 `Stack` 和 `LinkedList`。

```java
import java.util.ArrayDeque;
import java.util.Deque;

Deque<Integer> deque = new ArrayDeque<>();

// 栈操作
deque.push(1);          // 入栈（添加到头部）
deque.push(2);
int top = deque.peek();     // 查看栈顶
int val = deque.pop();      // 出栈

// 队列操作
deque.offerLast(1);     // 尾部入队
int head = deque.pollFirst();  // 头部出队

// 双端操作
deque.addFirst(0);      // 头部添加
deque.addLast(3);       // 尾部添加
deque.removeFirst();    // 头部删除
deque.removeLast();     // 尾部删除
```

---

## 集合工具类

### Arrays

```java
int[] arr = {5, 2, 8, 1, 9};

Arrays.sort(arr);                       // 排序
int index = Arrays.binarySearch(arr, 5); // 二分查找（需有序）
Arrays.fill(arr, 0);                    // 填充
int[] copy = Arrays.copyOf(arr, 3);     // 复制前3个元素
String str = Arrays.toString(arr);      // 转字符串

// 多维数组
int[][] matrix = {{1, 2}, {3, 4}};
System.out.println(Arrays.deepToString(matrix));  // [[1, 2], [3, 4]]
```

### Collections

```java
List<Integer> list = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5));

Collections.sort(list);                      // 升序排序
Collections.sort(list, (a, b) -> b - a);    // 降序排序
Collections.reverse(list);                   // 反转
Collections.shuffle(list);                   // 随机打乱
int max = Collections.max(list);             // 最大值
int min = Collections.min(list);             // 最小值
int freq = Collections.frequency(list, 1);  // 元素出现次数

// 不可变集合
List<Integer> unmodifiable = Collections.unmodifiableList(list);
```

---

## 选择指南

| 需求 | 推荐实现 |
|------|----------|
| 频繁随机访问 | `ArrayList` |
| 频繁插入删除 | `LinkedList` |
| 元素去重，不关心顺序 | `HashSet` |
| 元素去重，保持顺序 | `LinkedHashSet` |
| 元素去重，需要排序 | `TreeSet` |
| 键值对存储，不关心顺序 | `HashMap` |
| 键值对存储，保持顺序 | `LinkedHashMap` |
| 键值对存储，需要排序 | `TreeMap` |
| 优先级队列 | `PriorityQueue` |
| 栈操作 | `ArrayDeque` |
| 线程安全 | `ConcurrentHashMap`、`CopyOnWriteArrayList` |

---

## 小结

| 类型 | 实现类 | 特点 | 时间复杂度 |
|------|--------|------|:----------:|
| **List** | ArrayList | 动态数组，随机访问快 | get: $O(1)$, add/remove: $O(n)$ |
| **List** | LinkedList | 双向链表，插入删除快 | get: $O(n)$, add/remove: $O(1)$ |
| **Set** | HashSet | 哈希表，无序，最快 | $O(1)$ |
| **Set** | TreeSet | 红黑树，有序 | $O(\log n)$ |
| **Map** | HashMap | 哈希表，无序，最快 | $O(1)$ |
| **Map** | TreeMap | 红黑树，有序 | $O(\log n)$ |
| **Queue** | PriorityQueue | 优先队列（堆） | $O(\log n)$ |
| **Deque** | ArrayDeque | 双端队列，可作栈 | $O(1)$ |
