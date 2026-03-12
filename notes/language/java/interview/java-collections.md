# Java 集合面试题

> 本文档整理 Java 集合框架相关的高频面试题。

---

## 一、集合框架概述

### 1. Java 集合框架的整体结构？

**答案：**

```
                        Iterable
                            │
                        Collection
                       ↙    ↓    ↘
                    List    Set    Queue
                     │       │       │
            ┌───────┼───┐   │   ┌───┼───────┐
            │       │   │   │   │   │       │
        ArrayList LinkedList HashSet PriorityQueue ArrayDeque
        Vector    │       │
                  │   LinkedHashSet
              Stack    TreeSet

                        Map
                         │
            ┌────────────┼────────────┐
            │            │            │
        HashMap     LinkedHashMap  TreeMap
            │
        Hashtable
            │
       ConcurrentHashMap
```

**追问：Collection 和 Collections 的区别？**

> - **Collection**：集合接口，List、Set、Queue 的父接口
> - **Collections**：工具类，提供排序、查找、同步等方法
> ```java
> Collections.sort(list);
> Collections.binarySearch(list, key);
> Collections.synchronizedList(list);
> ```

---

### 2. List、Set、Queue、Map 的区别？

**答案：**

| 特性 | List | Set | Queue | Map |
|------|------|-----|-------|-----|
| 元素类型 | 单个元素 | 单个元素 | 单个元素 | 键值对 |
| 有序性 | 有序 | 无序 | 有序 | key 无序 |
| 可重复 | 可重复 | 不可重复 | 可重复 | key 不可重复 |
| 常用实现 | ArrayList | HashSet | LinkedList | HashMap |

```java
List<String> list = new ArrayList<>();  // 有序可重复
list.add("a"); list.add("a");           // [a, a]

Set<String> set = new HashSet<>();      // 无序不可重复
set.add("a"); set.add("a");             // [a]

Map<String, Integer> map = new HashMap<>();
map.put("a", 1); map.put("a", 2);       // {a=2}，覆盖
```

---

## 二、List 接口

### 3. ArrayList 的实现原理？

**答案：**

```java
public class ArrayList<E> {
    transient Object[] elementData;  // 底层数组
    private int size;                // 元素个数
    
    public boolean add(E e) {
        ensureCapacityInternal(size + 1);  // 确保容量
        elementData[size++] = e;           // 添加元素
        return true;
    }
}
```

**核心特点：**
- 底层是动态数组
- 默认初始容量 10
- 扩容为原来的 1.5 倍
- 支持 RandomAccess 接口，随机访问 O(1)

**追问：ArrayList 扩容机制？**

> ```java
> private void grow(int minCapacity) {
>     int oldCapacity = elementData.length;
>     int newCapacity = oldCapacity + (oldCapacity >> 1);  // 1.5 倍
>     if (newCapacity < minCapacity)
>         newCapacity = minCapacity;
>     elementData = Arrays.copyOf(elementData, newCapacity);
> }
> ```

---

### 4. ArrayList 和 LinkedList 的区别？

**答案：**

| 特性 | ArrayList | LinkedList |
|------|-----------|------------|
| 底层结构 | 动态数组 | 双向链表 |
| 随机访问 | O(1) | O(n) |
| 头部插入/删除 | O(n) | O(1) |
| 尾部插入/删除 | O(1) 均摊 | O(1) |
| 中间插入/删除 | O(n) | O(n) |
| 内存占用 | 连续内存 | 额外指针开销 |
| 缓存友好 | 是 | 否 |

**追问：为什么实际开发中 ArrayList 更常用？**

> 1. CPU 缓存友好（连续内存，预加载）
> 2. 随机访问更快
> 3. 内存开销更小
> 4. LinkedList 的作者自己都说不用 LinkedList

---

### 5. ArrayList 是线程安全的吗？如何实现线程安全？

**答案：**

ArrayList 不是线程安全的。

**解决方案：**

```java
// 方式一：Collections.synchronizedList
List<String> list1 = Collections.synchronizedList(new ArrayList<>());

// 方式二：CopyOnWriteArrayList（推荐）
List<String> list2 = new CopyOnWriteArrayList<>();

// 方式三：手动加锁
List<String> list3 = new ArrayList<>();
synchronized(list3) {
    list3.add("element");
}
```

**追问：CopyOnWriteArrayList 原理？**

> 写时复制：
> 1. 写操作时复制一份新数组
> 2. 在新数组上修改
> 3. 将引用指向新数组
> 
> 适合读多写少的场景，缺点是写操作时内存占用翻倍。

---

## 三、Set 接口

### 6. HashSet 的实现原理？

**答案：**

HashSet 底层就是 HashMap，元素作为 key，value 是固定对象。

```java
public class HashSet<E> {
    private transient HashMap<E,Object> map;
    private static final Object PRESENT = new Object();
    
    public boolean add(E e) {
        return map.put(e, PRESENT) == null;
    }
}
```

**添加元素流程：**
1. 计算 hashCode
2. 根据哈希值确定数组位置
3. 如果位置为空，直接插入
4. 如果位置有元素，调用 equals 比较
5. 如果相等则不添加，不相等则加入链表

**追问：HashSet 如何保证元素唯一？**

> 通过 hashCode() 和 equals() 两个方法：
> - 先比较 hashCode
> - hashCode 相同再比较 equals
> - 都相同则认为是重复元素

---

### 7. HashSet、LinkedHashSet、TreeSet 的区别？

**答案：**

| 特性 | HashSet | LinkedHashSet | TreeSet |
|------|---------|---------------|---------|
| 底层结构 | HashMap | LinkedHashMap | TreeMap |
| 有序性 | 无序 | 插入顺序 | 排序顺序 |
| 排序方式 | 无 | 无 | 自然排序/定制排序 |
| 性能 | 最高 | 较高 | 较低 |
| null 元素 | 允许一个 | 允许一个 | 不允许 |

```java
Set<String> hashSet = new HashSet<>();
hashSet.add("c"); hashSet.add("a"); hashSet.add("b");
// 输出顺序不确定

Set<String> linkedHashSet = new LinkedHashSet<>();
linkedHashSet.add("c"); linkedHashSet.add("a"); linkedHashSet.add("b");
// 输出：c, a, b（插入顺序）

Set<String> treeSet = new TreeSet<>();
treeSet.add("c"); treeSet.add("a"); treeSet.add("b");
// 输出：a, b, c（自然排序）
```

---

## 四、Map 接口

### 8. HashMap 的实现原理？

**答案：**

**JDK 1.7：数组 + 链表**
**JDK 1.8：数组 + 链表 + 红黑树**

```java
public class HashMap<K,V> {
    Node<K,V>[] table;  // 哈希桶数组
    
    static class Node<K,V> implements Map.Entry<K,V> {
        final int hash;
        final K key;
        V value;
        Node<K,V> next;  // 链表
    }
}
```

**put 操作流程：**

```
put(key, value)
      │
      ▼
计算 hash = key.hashCode() ^ (key.hashCode() >>> 16)
      │
      ▼
确定下标 = (n - 1) & hash
      │
      ▼
┌─────────────────────────────────────┐
│ 位置为空？                           │
│     是 → 直接插入                    │
│     否 → 冲突处理                    │
└─────────────────────────────────────┘
      │
      ▼
遍历链表/红黑树
      │
      ├── key 相同 → 覆盖 value
      │
      └── key 不同 → 插入新节点
                │
                ▼
          链表长度 >= 8 && 数组长度 >= 64
                │
                ▼
            链表转红黑树
```

**追问：为什么 JDK 1.8 要引入红黑树？**

> 链表过长时（哈希冲突严重），查询效率从 O(1) 退化为 O(n)。
> 红黑树查询效率为 O(log n)，提升性能。

---

### 9. HashMap 的扩容机制？

**答案：**

```java
// 默认参数
static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;  // 16
static final float DEFAULT_LOAD_FACTOR = 0.75f;

// 扩容条件
size > capacity * loadFactor

// 扩容操作
void resize() {
    int newCapacity = oldCapacity << 1;  // 2 倍
    Node<K,V>[] newTab = new Node[newCapacity];
    // 重新分配元素（rehash）
}
```

**追问：为什么容量是 2 的幂次方？**

> 1. 计算下标用 `(n - 1) & hash`
> 2. n 为 2 的幂次方时，n-1 的二进制全是 1
> 3. 这样 & 运算能充分利用 hash 值的每一位
> 4. 减少哈希碰撞，分布更均匀

**追问：为什么负载因子是 0.75？**

> - 太小：空间浪费
> - 太大：哈希冲突增多，性能下降
> - 0.75 是时间和空间的平衡点（泊松分布）

---

### 10. HashMap 是线程安全的吗？

**答案：**

HashMap 不是线程安全的。多线程环境下可能导致：

1. **数据丢失**：并发 put 导致覆盖
2. **死循环**：JDK 1.7 并发扩容时链表成环
3. **数据不一致**：可见性问题

**追问：如何实现线程安全的 Map？**

```java
// 方式一：ConcurrentHashMap（推荐）
Map<String, String> map1 = new ConcurrentHashMap<>();

// 方式二：Collections.synchronizedMap
Map<String, String> map2 = Collections.synchronizedMap(new HashMap<>());

// 方式三：Hashtable（不推荐，性能差）
Map<String, String> map3 = new Hashtable<>();
```

---

### 11. ConcurrentHashMap 的实现原理？

**答案：**

**JDK 1.7：分段锁（Segment）**

```
ConcurrentHashMap
    │
    ├── Segment[0] → HashEntry[] → 链表
    ├── Segment[1] → HashEntry[] → 链表
    ├── ...
    └── Segment[15] → HashEntry[] → 链表
    
每个 Segment 一把锁，默认 16 个分段，并发度 16
```

**JDK 1.8：CAS + synchronized**

```java
final V putVal(K key, V value, boolean onlyIfAbsent) {
    // 计算 hash
    int hash = spread(key.hashCode());
    
    for (Node<K,V>[] tab = table;;) {
        Node<K,V> f; int n, i, fh;
        
        // 情况1：位置为空，CAS 插入
        if (tab == null || (n = tab.length) == 0)
            tab = initTable();
        else if ((f = tabAt(tab, i = (n - 1) & hash)) == null) {
            if (casTabAt(tab, i, null, new Node<K,V>(hash, key, value, null)))
                break;
        }
        // 情况2：正在扩容，帮助扩容
        else if ((fh = f.hash) == MOVED)
            tab = helpTransfer(tab, f);
        // 情况3：加锁插入
        else {
            synchronized (f) {
                // 遍历链表/红黑树，插入或更新
            }
        }
    }
    return null;
}
```

**追问：JDK 1.8 为什么不用分段锁？**

> 1. 分段锁内存占用大
> 2. 锁粒度不够细
> 3. synchronized 在 JDK 1.6 后性能优化很多
> 4. CAS 无锁操作性能更好

---

### 12. HashMap 和 Hashtable 的区别？

**答案：**

| 特性 | HashMap | Hashtable |
|------|---------|-----------|
| 线程安全 | 不安全 | 安全（synchronized） |
| null 键/值 | 允许 | 不允许 |
| 继承关系 | AbstractMap | Dictionary |
| 迭代器 | fail-fast | Enumeration |
| 性能 | 高 | 低 |
| 推荐程度 | 推荐 | 不推荐（用 ConcurrentHashMap） |

---

## 五、Queue 接口

### 13. Queue 和 Deque 的区别？

**答案：**

**Queue（单端队列）：**
```
先进先出（FIFO）
        ┌─────────────────┐
入队 →  [A][B][C][D][E]  → 出队
        └─────────────────┘
```

**Deque（双端队列）：**
```
两端都可以入队/出队
入队 ↘                   ↙ 出队
      ┌─────────────────┐
      [A][B][C][D][E]
      └─────────────────┘
出队 ↗                   ↖ 入队
```

```java
Queue<String> queue = new LinkedList<>();
queue.offer("a");     // 入队
queue.poll();         // 出队
queue.peek();         // 查看队首

Deque<String> deque = new ArrayDeque<>();
deque.addFirst("a");  // 队首入队
deque.addLast("b");   // 队尾入队
deque.removeFirst();  // 队首出队
deque.removeLast();   // 队尾出队
```

---

### 14. ArrayDeque 和 LinkedList 的区别？

**答案：**

| 特性 | ArrayDeque | LinkedList |
|------|------------|------------|
| 底层结构 | 动态数组 | 双向链表 |
| 内存占用 | 较小 | 较大（存指针） |
| null 元素 | 不允许 | 允许 |
| 缓存友好 | 是 | 否 |
| 性能 | 更好 | 较差 |

**推荐使用 ArrayDeque 作为队列或栈。**

---

### 15. 什么是阻塞队列？

**答案：**

BlockingQueue：支持阻塞操作的队列。

```java
// 特点
- 队列为空时，take() 阻塞等待
- 队列已满时，put() 阻塞等待

// 常见实现
ArrayBlockingQueue    // 有界，数组实现
LinkedBlockingQueue   // 可选有界，链表实现
PriorityBlockingQueue // 无界，优先级排序
SynchronousQueue      // 无缓冲，直接传递
DelayQueue            // 延迟队列
```

**典型应用：生产者-消费者模式**

```java
BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);

// 生产者
new Thread(() -> {
    queue.put("data");  // 队列满时阻塞
}).start();

// 消费者
new Thread(() -> {
    String data = queue.take();  // 队列空时阻塞
}).start();
```

---

## 六、其他问题

### 16. fail-fast 和 fail-safe 是什么？

**答案：**

**fail-fast（快速失败）：**
```java
List<String> list = new ArrayList<>();
list.add("a");
Iterator<String> it = list.iterator();
list.add("b");  // 修改集合
it.next();      // 抛出 ConcurrentModificationException
```

- 在迭代过程中检测结构变化
- 维护 modCount，迭代时检查是否变化
- 常见：ArrayList、HashMap

**fail-safe（安全失败）：**
```java
List<String> list = new CopyOnWriteArrayList<>();
list.add("a");
Iterator<String> it = list.iterator();
list.add("b");  // 修改的是副本
it.next();      // 正常工作，但可能看到旧数据
```

- 迭代的是集合的副本
- 不检测结构变化
- 常见：CopyOnWriteArrayList、ConcurrentHashMap

---

### 17. 如何选择合适的集合？

**答案：**

```
需要键值对？
├── 是 → Map
│        ├── 需要排序？→ TreeMap
│        ├── 需要线程安全？→ ConcurrentHashMap
│        └── 其他 → HashMap
│
└── 否 → Collection
         ├── 需要唯一？→ Set
         │            ├── 需要排序？→ TreeSet
         │            ├── 需要插入顺序？→ LinkedHashSet
         │            └── 其他 → HashSet
         │
         └── 不需要唯一 → List
                          ├── 查询多？→ ArrayList
                          ├── 增删多？→ LinkedList（但实际很少用）
                          └── 需要线程安全？→ CopyOnWriteArrayList
```

---

## 小结

本文档涵盖了 Java 集合框架面试的高频考点：

- 集合框架整体结构
- List 接口实现原理
- Set 接口与唯一性
- HashMap 实现原理与扩容
- ConcurrentHashMap 线程安全
- 阻塞队列
- fail-fast 与 fail-safe
- 集合选择策略
