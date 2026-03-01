# 排序算法

排序算法是计算机科学中最基础、最重要的算法之一。它将一组数据按照特定顺序（升序或降序）重新排列。排序算法广泛应用于数据库索引、搜索优化、数据分析等场景。

## 基本概念

### 稳定性

**稳定性**是指排序算法在处理相等元素时，能否保持它们原有的相对顺序。

- **稳定排序**：若 `a == b`，且排序前 `a` 在 `b` 之前，排序后 `a` 仍在 `b` 之前
- **不稳定排序**：相等元素的相对顺序可能发生改变

📌 **稳定性的意义**：在多关键字排序中，稳定性保证后续排序不会破坏之前的排序结果。

### 内排序 vs 外排序

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **内排序** | 数据全部加载到内存中排序 | 数据量较小，可完全放入内存 |
| **外排序** | 数据分批次在内存和外部存储间交换 | 数据量大，无法一次性载入内存 |

### 比较排序 vs 非比较排序

| 类型 | 原理 | 时间复杂度下界 | 代表算法 |
|------|------|----------------|----------|
| **比较排序** | 通过比较元素间大小关系排序 | O(n log n) | 快速排序、归并排序、堆排序 |
| **非比较排序** | 利用元素本身特性（如数值大小）排序 | O(n) | 计数排序、桶排序、基数排序 |

---

## 基础排序算法

基础排序算法思想简单、易于实现，但时间复杂度较高（O(n²)），适合小规模数据或作为教学入门。

### 选择排序

**核心思想**：每次从未排序部分选择最小元素，放到已排序部分的末尾。

**动画演示**：
```
初始: [5, 3, 8, 4, 2]

第1轮: 在全部元素中找最小值 2
       [5, 3, 8, 4, 2] → [2, 3, 8, 4, 5]
        ↑           ↑
       交换位置

第2轮: 在剩余元素中找最小值 3（已在正确位置）
       [2, 3, 8, 4, 5] → [2, 3, 8, 4, 5]
              ↑  ↑
           最小为3

第3轮: 在剩余元素中找最小值 4
       [2, 3, 8, 4, 5] → [2, 3, 4, 8, 5]
                 ↑  ↑
              交换位置

第4轮: 在剩余元素中找最小值 5
       [2, 3, 4, 8, 5] → [2, 3, 4, 5, 8]
                    ↑  ↑
                 交换位置

结果: [2, 3, 4, 5, 8]
```

**复杂度分析**：
- 时间复杂度：O(n²)（所有情况相同）
- 空间复杂度：O(1)
- 稳定性：❌ 不稳定

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

void selectionSort(vector<int>& arr) {
    int n = arr.size();
    
    // 外层循环：控制已排序部分的边界
    // i 表示当前需要放置最小元素的位置
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;  // 假设当前位置就是最小值
        
        // 内层循环：在未排序部分寻找最小值
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;  // 更新最小值索引
            }
        }
        
        // 将找到的最小值放到正确位置
        if (minIdx != i) {
            swap(arr[i], arr[minIdx]);
        }
    }
}
```

```python [Python]
def selection_sort(arr):
    """
    选择排序
    时间复杂度: O(n²)
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    n = len(arr)
    
    # 外层循环：控制已排序部分的边界
    for i in range(n - 1):
        min_idx = i  # 假设当前位置就是最小值
        
        # 内层循环：在未排序部分寻找最小值
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j  # 更新最小值索引
        
        # 将找到的最小值放到正确位置
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr
```
:::

### 冒泡排序

**核心思想**：相邻元素两两比较，将较大元素逐步"冒泡"到数组末尾。

**动画演示**：
```
初始: [5, 3, 8, 4, 2]

第1轮冒泡: 比较相邻元素，大的往后移
  [5, 3, 8, 4, 2] → [3, 5, 8, 4, 2]  比较5,3，交换
  [3, 5, 8, 4, 2] → [3, 5, 8, 4, 2]  比较5,8，不换
  [3, 5, 8, 4, 2] → [3, 5, 4, 8, 2]  比较8,4，交换
  [3, 5, 4, 8, 2] → [3, 5, 4, 2, 8]  比较8,2，交换
  最大值8已"冒泡"到末尾 ✓

第2轮冒泡:
  [3, 5, 4, 2, 8] → [3, 4, 2, 5, 8]
  次大值5已归位 ✓

第3轮冒泡:
  [3, 4, 2, 5, 8] → [3, 2, 4, 5, 8]
  4已归位 ✓

第4轮冒泡:
  [3, 2, 4, 5, 8] → [2, 3, 4, 5, 8]
  全部排序完成 ✓

结果: [2, 3, 4, 5, 8]
```

**复杂度分析**：
- 时间复杂度：O(n²)（最坏/平均），O(n)（最好，已排序）
- 空间复杂度：O(1)
- 稳定性：✅ 稳定

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    
    // 外层循环：控制冒泡轮数
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;  // 优化：检测是否发生交换
        
        // 内层循环：执行冒泡操作
        // 每轮结束后，最大的 i+1 个元素已归位
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        
        // 如果本轮没有交换，说明已经有序
        if (!swapped) break;
    }
}
```

```python [Python]
def bubble_sort(arr):
    """
    冒泡排序
    时间复杂度: O(n²) 最坏/平均, O(n) 最好(已排序)
    空间复杂度: O(1)
    稳定性: 稳定
    """
    n = len(arr)
    
    # 外层循环：控制冒泡轮数
    for i in range(n - 1):
        swapped = False  # 优化：检测是否发生交换
        
        # 内层循环：执行冒泡操作
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # 如果本轮没有交换，说明已经有序
        if not swapped:
            break
    
    return arr
```
:::

### 插入排序

**核心思想**：将元素逐个插入到已排序部分的正确位置，类似整理扑克牌。

**动画演示**：
```
初始: [5, 3, 8, 4, 2]

第1轮: 将 arr[1]=3 插入到已排序部分 [5] 中
  [5, 3, 8, 4, 2]
      ↓
  3 < 5，需要插入到5前面
  [3, 5, 8, 4, 2]
   ↑ 已排序: [3, 5]

第2轮: 将 arr[2]=8 插入到已排序部分 [3, 5] 中
  [3, 5, 8, 4, 2]
         ↓
  8 > 5，已在正确位置
  [3, 5, 8, 4, 2]
      ↑ 已排序: [3, 5, 8]

第3轮: 将 arr[3]=4 插入
  [3, 5, 8, 4, 2]
            ↓
  4 < 8, 4 < 5, 4 > 3，插入到3和5之间
  [3, 4, 5, 8, 2]
         ↑ 已排序: [3, 4, 5, 8]

第4轮: 将 arr[4]=2 插入
  2 插入到最前面
  [2, 3, 4, 5, 8]

结果: [2, 3, 4, 5, 8]
```

**复杂度分析**：
- 时间复杂度：O(n²)（最坏/平均），O(n)（最好，已排序）
- 空间复杂度：O(1)
- 稳定性：✅ 稳定

💡 **特点**：对于几乎有序的数据，插入排序效率很高，接近 O(n)。

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

void insertionSort(vector<int>& arr) {
    int n = arr.size();
    
    // 从第二个元素开始，逐个插入
    for (int i = 1; i < n; i++) {
        int key = arr[i];  // 当前待插入元素
        int j = i - 1;     // 已排序部分的最后一个位置
        
        // 向后移动大于 key 的元素
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        
        // 插入到正确位置
        arr[j + 1] = key;
    }
}
```

```python [Python]
def insertion_sort(arr):
    """
    插入排序
    时间复杂度: O(n²) 最坏/平均, O(n) 最好(已排序)
    空间复杂度: O(1)
    稳定性: 稳定
    """
    n = len(arr)
    
    # 从第二个元素开始，逐个插入
    for i in range(1, n):
        key = arr[i]  # 当前待插入元素
        j = i - 1     # 已排序部分的最后一个位置
        
        # 向后移动大于 key 的元素
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # 插入到正确位置
        arr[j + 1] = key
    
    return arr
```
:::

---

## 高效排序算法

高效排序算法的时间复杂度可以达到 O(n log n)，是处理大规模数据的首选。

### 希尔排序

**核心思想**：改进的插入排序。先将数组按间隔分组进行插入排序，逐渐减小间隔直到为1。

**动画演示**：
```
初始: [5, 3, 8, 4, 2, 7, 1, 6]
n = 8, 初始间隔 gap = 4

第1轮 (gap=4): 分成4组，每组插入排序
  组1: arr[0]=5, arr[4]=2 → [2, 5]    位置: 0, 4
  组2: arr[1]=3, arr[5]=7 → [3, 7]    位置: 1, 5
  组3: arr[2]=8, arr[6]=1 → [1, 8]    位置: 2, 6
  组4: arr[3]=4, arr[7]=6 → [4, 6]    位置: 3, 7
  
  结果: [2, 3, 1, 4, 5, 7, 8, 6]
         ↑     ↑     ↑     ↑
        组1   组2   组3   组4

第2轮 (gap=2): 分成2组，每组插入排序
  组1: arr[0,2,4,6] = [2, 1, 5, 8] → [1, 2, 5, 8]
  组2: arr[1,3,5,7] = [3, 4, 7, 6] → [3, 4, 6, 7]
  
  结果: [1, 3, 2, 4, 5, 6, 8, 7]

第3轮 (gap=1): 普通插入排序
  [1, 3, 2, 4, 5, 6, 8, 7] → [1, 2, 3, 4, 5, 6, 7, 8]

结果: [1, 2, 3, 4, 5, 6, 7, 8]
```

**复杂度分析**：
- 时间复杂度：O(n log n) ~ O(n²)，取决于间隔序列
- 空间复杂度：O(1)
- 稳定性：❌ 不稳定

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

void shellSort(vector<int>& arr) {
    int n = arr.size();
    
    // 使用 Knuth 间隔序列: gap = gap / 3 + 1
    for (int gap = n / 2; gap > 0; gap /= 2) {
        // 对每个间隔进行插入排序
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j = i;
            
            // 在当前间隔下进行插入排序
            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            
            arr[j] = temp;
        }
    }
}
```

```python [Python]
def shell_sort(arr):
    """
    希尔排序
    时间复杂度: O(n log n) ~ O(n²) 取决于间隔序列
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    n = len(arr)
    
    # 使用 Shell 原始间隔序列
    gap = n // 2
    
    while gap > 0:
        # 对每个间隔进行插入排序
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            # 在当前间隔下进行插入排序
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            arr[j] = temp
        
        gap //= 2  # 缩小间隔
    
    return arr
```
:::

### 快速排序 ⭐

**核心思想**：分治思想。选择一个基准元素，将数组分为小于基准和大于基准两部分，递归排序。

**动画演示**：
```
初始: [5, 3, 8, 4, 2, 7, 1, 6]
选择基准 pivot = arr[0] = 5

分区过程 (左右指针法):
  [5, 3, 8, 4, 2, 7, 1, 6]
   ↑                    ↑
  pivot               right
  
  1. right找小于5的: 找到1
  2. left找大于5的: 找到8
  3. 交换8和1:
  [5, 3, 1, 4, 2, 7, 8, 6]
         ↑        ↑
       left     right
  
  4. 继续移动...
  5. left与right相遇，交换pivot与相遇位置:
  [2, 3, 1, 4, 5, 7, 8, 6]
            ↑
         pivot位置
  
分区结果: 左边都<5, 右边都>5
  左边: [2, 3, 1, 4]  递归
  右边: [7, 8, 6]     递归

递归展开:
  [2, 3, 1, 4] → pivot=2
    分区: [1, 2, 3, 4]
  
  [7, 8, 6] → pivot=7
    分区: [6, 7, 8]

最终合并: [1, 2, 3, 4, 5, 6, 7, 8]
```

**复杂度分析**：
- 时间复杂度：O(n log n)（平均），O(n²)（最坏，已排序且选首/尾为基准）
- 空间复杂度：O(log n)（递归栈）
- 稳定性：❌ 不稳定

⚠️ **优化策略**：
1. 随机选择基准，避免最坏情况
2. 三数取中法选择基准
3. 小数组使用插入排序

::: code-group
```cpp [C++]
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

// 分区函数（左右指针法）
int partition(vector<int>& arr, int left, int right) {
    // 随机选择基准，避免最坏情况
    int randIdx = left + rand() % (right - left + 1);
    swap(arr[left], arr[randIdx]);
    
    int pivot = arr[left];  // 基准值
    int i = left, j = right;
    
    while (i < j) {
        // 从右向左找第一个小于pivot的元素
        while (i < j && arr[j] >= pivot) j--;
        // 从左向右找第一个大于pivot的元素
        while (i < j && arr[i] <= pivot) i++;
        
        if (i < j) {
            swap(arr[i], arr[j]);
        }
    }
    
    // 将基准放到正确位置
    swap(arr[left], arr[j]);
    return j;
}

void quickSortHelper(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    
    int pivotIdx = partition(arr, left, right);
    
    quickSortHelper(arr, left, pivotIdx - 1);   // 排序左半部分
    quickSortHelper(arr, pivotIdx + 1, right);  // 排序右半部分
}

void quickSort(vector<int>& arr) {
    srand(time(nullptr));  // 初始化随机种子
    quickSortHelper(arr, 0, arr.size() - 1);
}
```

```python [Python]
import random

def quick_sort(arr):
    """
    快速排序
    时间复杂度: O(n log n) 平均, O(n²) 最坏
    空间复杂度: O(log n) 递归栈
    稳定性: 不稳定
    """
    def partition(left, right):
        # 随机选择基准
        rand_idx = random.randint(left, right)
        arr[left], arr[rand_idx] = arr[rand_idx], arr[left]
        
        pivot = arr[left]  # 基准值
        i, j = left, right
        
        while i < j:
            # 从右向左找第一个小于pivot的元素
            while i < j and arr[j] >= pivot:
                j -= 1
            # 从左向右找第一个大于pivot的元素
            while i < j and arr[i] <= pivot:
                i += 1
            
            if i < j:
                arr[i], arr[j] = arr[j], arr[i]
        
        # 将基准放到正确位置
        arr[left], arr[j] = arr[j], arr[left]
        return j
    
    def quick_sort_helper(left, right):
        if left >= right:
            return
        
        pivot_idx = partition(left, right)
        quick_sort_helper(left, pivot_idx - 1)
        quick_sort_helper(pivot_idx + 1, right)
    
    quick_sort_helper(0, len(arr) - 1)
    return arr
```
:::

### 归并排序 ⭐

**核心思想**：分治思想。将数组分成两半，分别排序后再合并。

**动画演示**：
```
初始: [5, 3, 8, 4, 2, 7, 1, 6]

分解阶段 (递归拆分):
          [5, 3, 8, 4, 2, 7, 1, 6]
                    ↓ 拆分
      [5, 3, 8, 4]      [2, 7, 1, 6]
          ↓                 ↓
    [5, 3] [8, 4]     [2, 7] [1, 6]
      ↓     ↓           ↓     ↓
   [5][3] [8][4]     [2][7] [1][6]

合并阶段 (两两合并):
   [5][3] [8][4]     [2][7] [1][6]
      ↓     ↓           ↓     ↓
    [3,5] [4,8]      [2,7] [1,6]
          ↓               ↓
    [3,4,5,8]       [1,2,6,7]
              ↘     ↙
         [1,2,3,4,5,6,7,8]

合并过程详解 (以 [3,5] 和 [4,8] 为例):
  左数组: [3, 5]    右数组: [4, 8]
  
  比较左数组第一个(3)和右数组第一个(4)
  3 < 4, 取3, 左指针后移
  结果: [3, _, _, _]
  
  比较5和4: 5 > 4, 取4, 右指针后移
  结果: [3, 4, _, _]
  
  比较5和8: 5 < 8, 取5, 左指针后移
  结果: [3, 4, 5, _]
  
  左数组用完, 复制剩余右数组
  结果: [3, 4, 5, 8]
```

**复杂度分析**：
- 时间复杂度：O(n log n)（所有情况）
- 空间复杂度：O(n)（需要临时数组）
- 稳定性：✅ 稳定

💡 **特点**：时间复杂度稳定，适合链表排序和外部排序。

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

// 合并两个有序数组
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);  // 临时数组
    int i = left, j = mid + 1, k = 0;
    
    // 比较并合并
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {  // <= 保证稳定性
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // 复制剩余元素
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    // 将合并结果复制回原数组
    for (int p = 0; p < k; p++) {
        arr[left + p] = temp[p];
    }
}

void mergeSortHelper(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;  // 避免溢出
    mergeSortHelper(arr, left, mid);      // 排序左半部分
    mergeSortHelper(arr, mid + 1, right); // 排序右半部分
    merge(arr, left, mid, right);         // 合并
}

void mergeSort(vector<int>& arr) {
    mergeSortHelper(arr, 0, arr.size() - 1);
}
```

```python [Python]
def merge_sort(arr):
    """
    归并排序
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    稳定性: 稳定
    """
    def merge(left_arr, right_arr):
        """合并两个有序数组"""
        result = []
        i = j = 0
        
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:  # <= 保证稳定性
                result.append(left_arr[i])
                i += 1
            else:
                result.append(right_arr[j])
                j += 1
        
        # 添加剩余元素
        result.extend(left_arr[i:])
        result.extend(right_arr[j:])
        
        return result
    
    if len(arr) <= 1:
        return arr
    
    # 分割
    mid = len(arr) // 2
    left_arr = merge_sort(arr[:mid])
    right_arr = merge_sort(arr[mid:])
    
    # 合并
    return merge(left_arr, right_arr)
```
:::

### 堆排序

**核心思想**：利用堆的性质进行排序。先建大顶堆，然后反复取出堆顶（最大值）放到数组末尾。

**动画演示**：
```
初始: [5, 3, 8, 4, 2, 7, 1, 6]

第一步: 建堆 (从最后一个非叶子节点向上调整)
          
  原始数组可视化 (完全二叉树):
        5
      /   \
     3     8
    / \   / \
   4   2 7   1
  /
 6
  
  建堆过程 (从下往上调整):
  1. 调整节点3 (索引1): 子节点[4,2]的最大值是4
     4 > 3, 交换 → [5, 4, 8, 3, 2, 7, 1, 6]
  
  2. 调整节点5 (索引0): 子节点[4,8]的最大值是8
     8 > 5, 交换 → [8, 4, 5, 3, 2, 7, 1, 6]
     继续: 5的子节点[7,1]的最大值是7
     7 > 5, 交换 → [8, 4, 7, 3, 2, 5, 1, 6]
  
  大顶堆:
        8
      /   \
     4     7
    / \   / \
   3   2 5   1
  /
 6

第二步: 排序 (重复: 取堆顶 → 放末尾 → 调整堆)

  1. 取出8, 放到末尾, 调整堆
     [8, 4, 7, 3, 2, 5, 1, 6] → [7, 4, 6, 3, 2, 5, 1, 8]
                                            ↑堆顶↗     ↑已排序
  
  2. 取出7, 调整堆
     [7, 4, 6, 3, 2, 5, 1, 8] → [6, 4, 5, 3, 2, 1, 7, 8]
                                         ↑       ↑ 已排序
  
  ... 重复直到完成 ...

最终结果: [1, 2, 3, 4, 5, 6, 7, 8]
```

**复杂度分析**：
- 时间复杂度：O(n log n)（所有情况）
- 空间复杂度：O(1)
- 稳定性：❌ 不稳定

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

// 调整堆（大顶堆）
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;       // 当前节点
    int left = 2 * i + 1;  // 左子节点
    int right = 2 * i + 2; // 右子节点
    
    // 找出当前节点、左子节点、右子节点中的最大值
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    
    // 如果最大值不是当前节点，交换并继续调整
    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);  // 递归调整
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();
    
    // 建堆：从最后一个非叶子节点开始向上调整
    // 最后一个非叶子节点索引 = n / 2 - 1
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    
    // 排序：反复取出堆顶元素
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);  // 将堆顶（最大值）放到末尾
        heapify(arr, i, 0);    // 调整剩余元素为堆
    }
}
```

```python [Python]
def heap_sort(arr):
    """
    堆排序
    时间复杂度: O(n log n)
    空间复杂度: O(1)
    稳定性: 不稳定
    """
    def heapify(n, i):
        """调整堆（大顶堆）"""
        largest = i       # 当前节点
        left = 2 * i + 1  # 左子节点
        right = 2 * i + 2 # 右子节点
        
        # 找出当前节点、左子节点、右子节点中的最大值
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        # 如果最大值不是当前节点，交换并继续调整
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n, largest)  # 递归调整
    
    n = len(arr)
    
    # 建堆：从最后一个非叶子节点开始向上调整
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    
    # 排序：反复取出堆顶元素
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 将堆顶放到末尾
        heapify(i, 0)  # 调整剩余元素
    
    return arr
```
:::

---

## 线性时间排序

线性时间排序算法不通过比较元素大小来排序，而是利用元素的数值特性，时间复杂度可达到 O(n)。

### 计数排序

**核心思想**：统计每个元素出现的次数，然后按顺序输出。

**动画演示**：
```
初始: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

第一步: 统计每个元素出现的次数
  值:    0  1  2  3  4  5  6  7  8  9
  计数:  0  2  1  2  1  3  1  0  0  1
  
  计数数组:
  [0, 2, 1, 2, 1, 3, 1, 0, 0, 1]

第二步: 计算累计计数（确定每个元素的最终位置）
  累计:  0  2  3  5  6  9  10 10 10 11
  
  含义: 值1的元素放在位置0-1
        值2的元素放在位置2
        值3的元素放在位置3-4
        ...

第三步: 根据累计计数放置元素（从后向前遍历保证稳定性）
  处理顺序: 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5
  
  1. 处理最后一个5: 累计[5]=9, 放在位置8
     [_, _, _, _, _, _, _, _, 5, _, _]
     累计[5]-- → 8
  
  2. 处理3: 累计[3]=5, 放在位置4
     [_, _, _, _, 3, _, _, _, 5, _, _]
  
  ... 继续处理 ...

最终结果: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

**复杂度分析**：
- 时间复杂度：O(n + k)，k 为数据范围
- 空间复杂度：O(k)
- 稳定性：✅ 稳定

⚠️ **适用场景**：数据范围不大且为非负整数。

::: code-group
```cpp [C++]
#include <vector>
#include <algorithm>
using namespace std;

void countingSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    // 找出数据范围
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;
    
    // 统计每个元素出现的次数
    vector<int> count(range, 0);
    for (int num : arr) {
        count[num - minVal]++;
    }
    
    // 计算累计计数
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }
    
    // 从后向前遍历，保证稳定性
    vector<int> output(arr.size());
    for (int i = arr.size() - 1; i >= 0; i--) {
        int idx = arr[i] - minVal;
        output[count[idx] - 1] = arr[i];
        count[idx]--;
    }
    
    arr = output;
}
```

```python [Python]
def counting_sort(arr):
    """
    计数排序
    时间复杂度: O(n + k), k为数据范围
    空间复杂度: O(k)
    稳定性: 稳定
    适用: 数据范围不大的非负整数
    """
    if not arr:
        return arr
    
    # 找出数据范围
    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val + 1
    
    # 统计每个元素出现的次数
    count = [0] * range_val
    for num in arr:
        count[num - min_val] += 1
    
    # 计算累计计数
    for i in range(1, range_val):
        count[i] += count[i - 1]
    
    # 从后向前遍历，保证稳定性
    output = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        idx = arr[i] - min_val
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
    
    return output
```
:::

### 桶排序

**核心思想**：将数据分到多个桶中，每个桶单独排序后合并。

**动画演示**：
```
初始: [0.42, 0.32, 0.73, 0.25, 0.67, 0.89, 0.15, 0.54]

数据范围: [0, 1)，创建5个桶，每个桶范围 0.2

第一步: 分配元素到桶中
  桶0 [0, 0.2):   [0.15]
  桶1 [0.2, 0.4): [0.32, 0.25]
  桶2 [0.4, 0.6): [0.42, 0.54]
  桶3 [0.6, 0.8): [0.73, 0.67]
  桶4 [0.8, 1.0): [0.89]

第二步: 对每个桶进行排序
  桶0: [0.15]
  桶1: [0.25, 0.32]
  桶2: [0.42, 0.54]
  桶3: [0.67, 0.73]
  桶4: [0.89]

第三步: 合并所有桶
  [0.15] + [0.25, 0.32] + [0.42, 0.54] + [0.67, 0.73] + [0.89]
  = [0.15, 0.25, 0.32, 0.42, 0.54, 0.67, 0.73, 0.89]
```

**复杂度分析**：
- 时间复杂度：O(n + k)（平均），O(n²)（最坏，所有元素在一个桶中）
- 空间复杂度：O(n + k)
- 稳定性：✅ 稳定（使用稳定排序处理桶时）

::: code-group
```cpp [C++]
#include <vector>
#include <algorithm>
using namespace std;

void bucketSort(vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;
    
    // 创建 n 个桶
    vector<vector<float>> buckets(n);
    
    // 将元素分配到桶中
    for (float num : arr) {
        // 假设数据范围 [0, 1)
        int bucketIdx = num * n;
        buckets[bucketIdx].push_back(num);
    }
    
    // 对每个桶进行排序
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
    }
    
    // 合并所有桶
    int idx = 0;
    for (const auto& bucket : buckets) {
        for (float num : bucket) {
            arr[idx++] = num;
        }
    }
}
```

```python [Python]
def bucket_sort(arr):
    """
    桶排序
    时间复杂度: O(n + k) 平均
    空间复杂度: O(n + k)
    稳定性: 稳定
    适用: 数据均匀分布的情况
    """
    if len(arr) <= 1:
        return arr
    
    n = len(arr)
    
    # 创建 n 个桶
    buckets = [[] for _ in range(n)]
    
    # 将元素分配到桶中（假设数据范围 [0, 1)）
    for num in arr:
        bucket_idx = int(num * n)
        buckets[bucket_idx].append(num)
    
    # 对每个桶进行排序
    for bucket in buckets:
        bucket.sort()
    
    # 合并所有桶
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result
```
:::

### 基数排序

**核心思想**：按位数从低到高逐位排序，通常使用计数排序作为子过程。

**动画演示**：
```
初始: [170, 45, 75, 90, 802, 24, 2, 66]

第一步: 按个位数排序
  170 → 桶0,  90 → 桶0
  45 → 桶5,   75 → 桶5
  802 → 桶2,  2 → 桶2
  24 → 桶4
  66 → 桶6
  
  排序后: [170, 90, 802, 2, 24, 45, 75, 66]

第二步: 按十位数排序
  170 → 桶7,  90 → 桶9
  802 → 桶0,  2 → 桶0
  24 → 桶2
  45 → 桶4,   75 → 桶7
  66 → 桶6
  
  排序后: [802, 2, 24, 45, 66, 170, 75, 90]

第三步: 按百位数排序
  802 → 桶8
  其他 → 桶0
  
  排序后: [2, 24, 45, 66, 75, 90, 170, 802]

最终结果: [2, 24, 45, 66, 75, 90, 170, 802]
```

**复杂度分析**：
- 时间复杂度：O(d × (n + k))，d 为位数，k 为基数（通常为10）
- 空间复杂度：O(n + k)
- 稳定性：✅ 稳定

::: code-group
```cpp [C++]
#include <vector>
using namespace std;

// 使用计数排序对特定位进行排序
void countingSortByDigit(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    vector<int> count(10, 0);  // 0-9 十个数字
    
    // 统计当前位的数字出现次数
    for (int i = 0; i < n; i++) {
        int digit = (arr[i] / exp) % 10;
        count[digit]++;
    }
    
    // 计算累计计数
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }
    
    // 从后向前遍历，保证稳定性
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }
    
    arr = output;
}

void radixSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    // 找出最大值，确定位数
    int maxVal = *max_element(arr.begin(), arr.end());
    
    // 从低位到高位逐位排序
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        countingSortByDigit(arr, exp);
    }
}
```

```python [Python]
def radix_sort(arr):
    """
    基数排序
    时间复杂度: O(d * (n + k)), d为位数
    空间复杂度: O(n + k)
    稳定性: 稳定
    适用: 整数排序
    """
    if not arr:
        return arr
    
    def counting_sort_by_digit(arr, exp):
        """对特定位进行计数排序"""
        n = len(arr)
        output = [0] * n
        count = [0] * 10  # 0-9 十个数字
        
        # 统计当前位的数字出现次数
        for num in arr:
            digit = (num // exp) % 10
            count[digit] += 1
        
        # 计算累计计数
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # 从后向前遍历，保证稳定性
        for i in range(n - 1, -1, -1):
            digit = (arr[i] // exp) % 10
            output[count[digit] - 1] = arr[i]
            count[digit] -= 1
        
        return output
    
    # 找出最大值，确定位数
    max_val = max(arr)
    
    # 从低位到高位逐位排序
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr
```
:::

---

## 排序算法比较

### 综合对比表

| 算法 | 最好时间 | 平均时间 | 最坏时间 | 空间复杂度 | 稳定性 | 适用场景 |
|------|----------|----------|----------|------------|--------|----------|
| 选择排序 | O(n²) | O(n²) | O(n²) | O(1) | ❌ 不稳定 | 小规模数据 |
| 冒泡排序 | O(n) | O(n²) | O(n²) | O(1) | ✅ 稳定 | 小规模、教学 |
| 插入排序 | O(n) | O(n²) | O(n²) | O(1) | ✅ 稳定 | 小规模、近乎有序 |
| 希尔排序 | O(n log n) | O(n^1.3) | O(n²) | O(1) | ❌ 不稳定 | 中等规模 |
| 快速排序 | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ 不稳定 | 通用、大规模 |
| 归并排序 | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ 稳定 | 稳定性要求高、外部排序 |
| 堆排序 | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ 不稳定 | 空间受限场景 |
| 计数排序 | O(n + k) | O(n + k) | O(n + k) | O(k) | ✅ 稳定 | 数据范围小的整数 |
| 桶排序 | O(n + k) | O(n + k) | O(n²) | O(n + k) | ✅ 稳定 | 数据均匀分布 |
| 基数排序 | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | ✅ 稳定 | 整数、固定位数 |

### 选择建议

📌 **根据数据规模选择**：
- **小规模（n < 50）**：插入排序
- **中等规模（50 < n < 1000）**：希尔排序
- **大规模（n > 1000）**：快速排序、归并排序、堆排序

📌 **根据数据特点选择**：
- **近乎有序**：插入排序（接近 O(n)）
- **数据范围小**：计数排序
- **数据均匀分布**：桶排序
- **整数排序**：基数排序
- **需要稳定性**：归并排序、插入排序、冒泡排序

📌 **根据内存限制选择**：
- **内存充足**：归并排序
- **内存受限**：堆排序、快速排序
- **外部存储**：归并排序（外部排序）

---

## 常见问题

### 为什么快速排序通常比归并排序快？

1. **常数因子更小**：快速排序的内部循环更简洁
2. **局部性更好**：快速排序的内存访问模式更利于缓存
3. **原地排序**：不需要额外空间

### 什么时候选择稳定排序？

- 多关键字排序（如先按年龄排序，再按分数排序）
- 需要保持原始顺序的场景
- 数据库查询中的多字段排序

### 如何处理大数据量的排序？

1. **外部排序**：使用归并排序的分段处理
2. **多线程排序**：并行快速排序、并行归并排序
3. **采样优化**：根据数据分布选择合适算法
