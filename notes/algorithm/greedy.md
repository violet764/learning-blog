# 贪心算法

贪心算法（Greedy Algorithm）是一种在每一步选择中都采取当前状态下最优的选择，从而希望导致全局最优的算法策略。它的核心思想是**局部最优 → 全局最优**。

📌 **核心价值**：贪心算法通常简单高效，是解决优化问题的重要方法之一。虽然不能保证所有问题都能得到最优解，但在满足特定条件时，贪心策略可以获得最优解。

## 核心概念

### 贪心选择性质

**贪心选择性质**是指问题的全局最优解可以通过一系列局部最优选择来达到。即：**每一步做出的贪心选择，最终可以导致问题的最优解**。

::: tip 关键特征
- 不依赖于未来的选择
- 不依赖于之前的选择结果
- 每一步都是独立的"最优"决策
:::

```
示例：找零钱问题
需要找零 37 元，硬币面值 [25, 10, 5, 1]

贪心策略：每次选择面值最大的硬币
第1步: 选择 25 → 剩余 12
第2步: 选择 10 → 剩余 2
第3步: 选择 1  → 剩余 1
第4步: 选择 1  → 剩余 0

结果: 25 + 10 + 1 + 1 = 37，共 4 枚硬币 ✓ 最优解
```

⚠️ **注意**：贪心策略并不总是有效。例如，如果硬币面值为 [4, 3, 1]，需要找零 6 元：
- 贪心：4 + 1 + 1 = 6，共 3 枚
- 最优：3 + 3 = 6，共 2 枚

### 最优子结构

**最优子结构**是指问题的最优解包含其子问题的最优解。

::: tip 关键特征
- 当一个问题的最优解包含子问题的最优解时，称该问题具有最优子结构
- 贪心算法和动态规划都要求问题具有最优子结构
:::

```
示例：活动选择问题

活动集合: A = {a1, a2, ..., an}
问题: 选择最多的互不冲突的活动

如果 S 是 A 的最优解，且 S 包含活动 ak
那么 S - {ak} 是子问题 A' 的最优解
其中 A' 是与 ak 不冲突的所有活动
```

### 与动态规划的区别

贪心算法和动态规划都要求问题具有**最优子结构**，但两者的决策方式不同：

| 特性 | 贪心算法 | 动态规划 |
|------|----------|----------|
| 决策方式 | 自顶向下，每步做最优选择 | 自底向上，考虑所有可能的子问题 |
| 子问题关系 | 子问题独立，不重叠 | 子问题重叠，需要存储结果 |
| 适用条件 | 贪心选择性质 + 最优子结构 | 最优子结构 + 重叠子问题 |
| 时间复杂度 | 通常较低 | 通常较高 |
| 正确性 | 需要证明 | 总是正确 |

```
示例：0-1背包问题 vs 分数背包问题

物品信息：
  物品A: 价值 60, 重量 10, 单位价值 6
  物品B: 价值 100, 重量 20, 单位价值 5
  物品C: 价值 120, 重量 30, 单位价值 4
  背包容量: 50

分数背包问题（可以取部分物品）：
  贪心策略：按单位价值降序选取
  选择: A全部(10) + B全部(20) + C部分(20)
  总价值: 60 + 100 + 80 = 240 ✓ 最优解

0-1背包问题（物品只能完整选取）：
  贪心策略：按单位价值降序选取
  选择: A全部(10) + B全部(20) + C无法放入
  总价值: 60 + 100 = 160 ✗ 非最优
  
  最优解: B全部(20) + C全部(30)
  总价值: 100 + 120 = 220 ✓ 最优解
  
结论: 0-1背包问题不满足贪心选择性质，需要用动态规划
```

### 证明方法

贪心算法的正确性需要严格证明，常用方法包括：

#### 交换论证（Exchange Argument）

**核心思想**：证明贪心解可以逐步转化为最优解，而不会变差。

::: info 证明步骤
1. 假设存在一个最优解 OPT
2. 证明可以通过交换操作，将 OPT 转化为贪心解
3. 证明交换操作不会使解变差
4. 结论：贪心解是最优解
:::

```
示例：活动选择问题的交换论证

问题: 选择最多的互不冲突的活动
贪心策略: 按结束时间排序，每次选择最早结束的活动

证明:
1. 设贪心解为 G = {g1, g2, ..., gk}
2. 设最优解为 OPT = {o1, o2, ..., om}，按结束时间排序
3. 证明 |G| = |OPT|：

   首先，g1 的结束时间 ≤ o1 的结束时间（贪心选择的性质）
   
   将 o1 替换为 g1：
   - g1 比 o1 结束更早，不会与后续活动冲突
   - 替换后仍是最优解
   
   归纳地，可以将所有 oi 替换为 gi
   因此 m = k，贪心解是最优解
```

#### 数学归纳法

**核心思想**：证明贪心选择在任意步骤后，剩余问题的最优解 + 当前选择 = 全局最优解。

::: info 证明步骤
1. **基础情况**：证明对于最小规模问题，贪心选择正确
2. **归纳假设**：假设对于规模为 n-1 的子问题，贪心算法正确
3. **归纳步骤**：证明对于规模为 n 的问题，贪心选择 + 子问题最优解 = 全局最优解
:::

---

## 经典贪心问题

### 活动选择问题

::: info 问题描述
有 n 个活动，每个活动有开始时间和结束时间。选择最多的互不冲突的活动，两个活动不冲突当且仅当它们的时间区间不重叠。
:::

**解题思路**：
- 按结束时间排序
- 每次选择最早结束且不与已选活动冲突的活动

**贪心正确性**：选择最早结束的活动，能给后续活动留下最多的时间。

::: code-group

```cpp [C++ 实现]
#include <vector>
#include <algorithm>
using namespace std;

// 活动结构体
struct Activity {
    int start;
    int end;
    int id;  // 活动编号（可选）
};

vector<int> activitySelection(vector<Activity>& activities) {
    int n = activities.size();
    if (n == 0) return {};
    
    // 按结束时间排序
    sort(activities.begin(), activities.end(), 
         [](const Activity& a, const Activity& b) {
             return a.end < b.end;
         });
    
    vector<int> selected;  // 存储选中的活动编号
    selected.push_back(activities[0].id);  // 选择第一个活动
    int lastEnd = activities[0].end;       // 最后一个选中活动的结束时间
    
    // 遍历剩余活动
    for (int i = 1; i < n; i++) {
        // 如果当前活动开始时间 >= 上一个选中活动的结束时间
        // 说明不冲突，可以选择
        if (activities[i].start >= lastEnd) {
            selected.push_back(activities[i].id);
            lastEnd = activities[i].end;
        }
    }
    
    return selected;
}

// 简化版本：只返回最大活动数
int maxActivities(vector<vector<int>>& intervals) {
    // intervals[i] = {start, end}
    int n = intervals.size();
    if (n == 0) return 0;
    
    // 按结束时间排序
    sort(intervals.begin(), intervals.end(), 
         [](const vector<int>& a, const vector<int>& b) {
             return a[1] < b[1];
         });
    
    int count = 1;              // 选中活动数
    int lastEnd = intervals[0][1];  // 最后选中活动的结束时间
    
    for (int i = 1; i < n; i++) {
        if (intervals[i][0] >= lastEnd) {
            count++;
            lastEnd = intervals[i][1];
        }
    }
    
    return count;
}
```

```python [Python 实现]
from typing import List, Tuple

def activity_selection(activities: List[Tuple[int, int, int]]) -> List[int]:
    """
    活动选择问题
    :param activities: 活动列表，每个元素为 (start, end, id)
    :return: 选中的活动编号列表
    """
    n = len(activities)
    if n == 0:
        return []
    
    # 按结束时间排序
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0][2]]  # 选择第一个活动
    last_end = activities[0][1]    # 最后一个选中活动的结束时间
    
    # 遍历剩余活动
    for i in range(1, n):
        start, end, act_id = activities[i]
        # 如果当前活动开始时间 >= 上一个选中活动的结束时间
        if start >= last_end:
            selected.append(act_id)
            last_end = end
    
    return selected


def max_activities(intervals: List[List[int]]) -> int:
    """
    活动选择问题 - 简化版
    :param intervals: 区间列表，每个元素为 [start, end]
    :return: 最大活动数
    """
    n = len(intervals)
    if n == 0:
        return 0
    
    # 按结束时间排序
    intervals.sort(key=lambda x: x[1])
    
    count = 1                     # 选中活动数
    last_end = intervals[0][1]    # 最后选中活动的结束时间
    
    for i in range(1, n):
        if intervals[i][0] >= last_end:
            count += 1
            last_end = intervals[i][1]
    
    return count


# 测试示例
if __name__ == "__main__":
    # 活动列表：(开始时间, 结束时间, 活动编号)
    activities = [
        (1, 4, 1),
        (3, 5, 2),
        (0, 6, 3),
        (5, 7, 4),
        (3, 9, 5),
        (5, 9, 6),
        (6, 10, 7),
        (8, 11, 8),
        (8, 12, 9),
        (2, 14, 10),
        (12, 16, 11)
    ]
    
    result = activity_selection(activities)
    print(f"选中的活动编号: {result}")  # 输出: [1, 4, 8, 11]
```

:::

**复杂度分析**：
- 时间复杂度：O(n log n)（排序）+ O(n)（遍历）= O(n log n)
- 空间复杂度：O(1) 或 O(n)（取决于是否存储选中活动）

### 跳跃游戏

::: info 问题描述（LeetCode 55）
给定一个非负整数数组 nums，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。
:::

**解题思路**：
- 维护一个变量 `maxReach`，表示当前能到达的最远位置
- 遍历数组，更新 `maxReach`
- 如果当前位置超过了 `maxReach`，说明无法到达

::: code-group

```cpp [C++ 实现]
// 跳跃游戏 I：判断是否能到达终点
bool canJump(vector<int>& nums) {
    int n = nums.size();
    int maxReach = 0;  // 当前能到达的最远位置
    
    for (int i = 0; i < n; i++) {
        // 如果当前位置超过了能到达的最远位置，返回 false
        if (i > maxReach) return false;
        
        // 更新能到达的最远位置
        maxReach = max(maxReach, i + nums[i]);
        
        // 如果已经能到达终点，提前返回
        if (maxReach >= n - 1) return true;
    }
    
    return true;
}

// 跳跃游戏 II：计算到达终点的最小跳跃次数
int jump(vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return 0;
    
    int jumps = 0;        // 跳跃次数
    int curEnd = 0;       // 当前跳跃能到达的边界
    int curFarthest = 0;  // 下一次跳跃能到达的最远位置
    
    for (int i = 0; i < n - 1; i++) {
        // 更新下一次跳跃能到达的最远位置
        curFarthest = max(curFarthest, i + nums[i]);
        
        // 到达当前跳跃的边界，必须跳跃
        if (i == curEnd) {
            jumps++;
            curEnd = curFarthest;
            
            // 如果已经能到达终点，提前返回
            if (curEnd >= n - 1) break;
        }
    }
    
    return jumps;
}
```

```python [Python 实现]
def can_jump(nums: list[int]) -> bool:
    """
    跳跃游戏 I：判断是否能到达终点
    :param nums: 每个位置的最大跳跃长度
    :return: 是否能到达终点
    """
    n = len(nums)
    max_reach = 0  # 当前能到达的最远位置
    
    for i in range(n):
        # 如果当前位置超过了能到达的最远位置，返回 False
        if i > max_reach:
            return False
        
        # 更新能到达的最远位置
        max_reach = max(max_reach, i + nums[i])
        
        # 如果已经能到达终点，提前返回
        if max_reach >= n - 1:
            return True
    
    return True


def jump(nums: list[int]) -> int:
    """
    跳跃游戏 II：计算到达终点的最小跳跃次数
    :param nums: 每个位置的最大跳跃长度
    :return: 最小跳跃次数
    """
    n = len(nums)
    if n <= 1:
        return 0
    
    jumps = 0          # 跳跃次数
    cur_end = 0        # 当前跳跃能到达的边界
    cur_farthest = 0   # 下一次跳跃能到达的最远位置
    
    for i in range(n - 1):
        # 更新下一次跳跃能到达的最远位置
        cur_farthest = max(cur_farthest, i + nums[i])
        
        # 到达当前跳跃的边界，必须跳跃
        if i == cur_end:
            jumps += 1
            cur_end = cur_farthest
            
            # 如果已经能到达终点，提前返回
            if cur_end >= n - 1:
                break
    
    return jumps


# 测试示例
if __name__ == "__main__":
    # 跳跃游戏 I
    nums1 = [2, 3, 1, 1, 4]
    print(f"能否到达终点: {can_jump(nums1)}")  # True
    
    nums2 = [3, 2, 1, 0, 4]
    print(f"能否到达终点: {can_jump(nums2)}")  # False
    
    # 跳跃游戏 II
    nums3 = [2, 3, 1, 1, 4]
    print(f"最小跳跃次数: {jump(nums3)}")  # 2
```

:::

### 分发糖果

::: info 问题描述（LeetCode 135）
有 n 个孩子站成一排，每个孩子有一个评分 rating。你需要给每个孩子分发糖果，满足：
1. 每个孩子至少有一个糖果
2. 相邻两个孩子中，评分更高的孩子必须获得更多糖果

返回最少需要准备的糖果数。
:::

**解题思路**：
- 进行两次遍历
- 第一次从左到右：处理右孩子评分高于左孩子的情况
- 第二次从右到左：处理左孩子评分高于右孩子的情况

::: code-group

```cpp [C++ 实现]
int candy(vector<int>& ratings) {
    int n = ratings.size();
    if (n == 0) return 0;
    
    vector<int> candies(n, 1);  // 每个孩子至少一个糖果
    
    // 从左到右遍历：如果右边评分更高，右边糖果 = 左边 + 1
    for (int i = 1; i < n; i++) {
        if (ratings[i] > ratings[i - 1]) {
            candies[i] = candies[i - 1] + 1;
        }
    }
    
    // 从右到左遍历：如果左边评分更高，左边糖果 = max(当前, 右边 + 1)
    for (int i = n - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            candies[i] = max(candies[i], candies[i + 1] + 1);
        }
    }
    
    // 计算总糖果数
    int total = 0;
    for (int c : candies) {
        total += c;
    }
    
    return total;
}
```

```python [Python 实现]
def candy(ratings: list[int]) -> int:
    """
    分发糖果
    :param ratings: 每个孩子的评分
    :return: 最少需要的糖果数
    """
    n = len(ratings)
    if n == 0:
        return 0
    
    candies = [1] * n  # 每个孩子至少一个糖果
    
    # 从左到右遍历：如果右边评分更高，右边糖果 = 左边 + 1
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    
    # 从右到左遍历：如果左边评分更高，左边糖果 = max(当前, 右边 + 1)
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    
    return sum(candies)


# 测试示例
if __name__ == "__main__":
    ratings1 = [1, 0, 2]
    print(f"最少糖果数: {candy(ratings1)}")  # 5 (2, 1, 2)
    
    ratings2 = [1, 2, 2]
    print(f"最少糖果数: {candy(ratings2)}")  # 4 (1, 2, 1)
    
    ratings3 = [1, 3, 2, 2, 1]
    print(f"最少糖果数: {candy(ratings3)}")  # 7 (1, 2, 1, 2, 1)
```

:::

---

## 贪心与排序

许多贪心问题需要先排序，然后再进行贪心选择。排序的依据通常是问题的关键。

### 排序后贪心的策略

::: tip 常见排序策略
1. **按结束时间排序**：活动选择、区间调度
2. **按开始时间排序**：合并区间、会议室问题
3. **按差值/比值排序**：背包问题变种、最优服务次序
4. **按绝对值/优先级排序**：任务调度、重构字符串
:::

### 按终点排序 vs 按起点排序

| 排序方式 | 适用场景 | 典型问题 |
|----------|----------|----------|
| **按终点排序** | 选择最多不重叠区间 | 活动选择、无重叠区间 |
| **按起点排序** | 合并/分割区间 | 合并区间、插入区间 |
| **双端排序对比** | 需要考虑两端的情况 | 用最少数箭引爆气球 |

```
示例对比：

区间集合: [[1, 3], [2, 4], [3, 5]]

按终点排序: [[1, 3], [2, 4], [3, 5]]
  → 贪心选择最早结束的 [1, 3]
  → 剩余可选: [3, 5]
  → 选中的区间: [[1, 3], [3, 5]]

按起点排序: [[1, 3], [2, 4], [3, 5]]
  → 依次处理，合并重叠区间
  → 合并后: [[1, 5]]
```

---

## 经典题目

### 无重叠区间

::: info 问题描述（LeetCode 435）
给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
:::

**解题思路**：
- 等价于"选择最多不重叠区间"
- 按终点排序，贪心选择不重叠的区间
- 需要移除的数量 = 总数 - 最多可选数量

::: code-group

```cpp [C++ 实现]
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    int n = intervals.size();
    if (n <= 1) return 0;
    
    // 按终点排序
    sort(intervals.begin(), intervals.end(), 
         [](const vector<int>& a, const vector<int>& b) {
             return a[1] < b[1];
         });
    
    int count = 1;  // 选中的区间数
    int lastEnd = intervals[0][1];
    
    for (int i = 1; i < n; i++) {
        // 如果当前区间起点 >= 上一个选中区间的终点，说明不重叠
        if (intervals[i][0] >= lastEnd) {
            count++;
            lastEnd = intervals[i][1];
        }
        // 否则跳过（相当于移除这个区间）
    }
    
    return n - count;  // 需要移除的数量
}
```

```python [Python 实现]
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """
    无重叠区间：求需要移除的最小区间数
    :param intervals: 区间列表
    :return: 需要移除的区间数
    """
    n = len(intervals)
    if n <= 1:
        return 0
    
    # 按终点排序
    intervals.sort(key=lambda x: x[1])
    
    count = 1  # 选中的区间数
    last_end = intervals[0][1]
    
    for i in range(1, n):
        # 如果当前区间起点 >= 上一个选中区间的终点，说明不重叠
        if intervals[i][0] >= last_end:
            count += 1
            last_end = intervals[i][1]
    
    return n - count  # 需要移除的数量


# 测试示例
if __name__ == "__main__":
    intervals1 = [[1, 2], [2, 3], [3, 4], [1, 3]]
    print(f"需要移除: {erase_overlap_intervals(intervals1)}")  # 1
    
    intervals2 = [[1, 2], [1, 2], [1, 2]]
    print(f"需要移除: {erase_overlap_intervals(intervals2)}")  # 2
```

:::

### 用最少数量的箭引爆气球

::: info 问题描述（LeetCode 452）
在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上气球直径的开始和结束坐标。一支弓箭可以沿着 x 轴从不同点垂直射出。如果一支箭在坐标 x 处射出，且 x 在气球直径范围内，该气球会被引爆。求引爆所有气球所需的最少弓箭数。
:::

**解题思路**：
- 按起点排序（或终点排序）
- 维护当前箭能射爆的区间范围
- 当新区间与当前范围不重叠时，需要新箭

::: code-group

```cpp [C++ 实现]
int findMinArrowShots(vector<vector<int>>& points) {
    int n = points.size();
    if (n == 0) return 0;
    
    // 按起点排序
    sort(points.begin(), points.end(), 
         [](const vector<int>& a, const vector<int>& b) {
             return a[0] < b[0];
         });
    
    int arrows = 1;  // 至少需要一支箭
    int end = points[0][1];  // 当前箭能射到的最远位置
    
    for (int i = 1; i < n; i++) {
        // 如果当前气球的起点 > 当前箭能射到的最远位置
        // 需要新箭
        if (points[i][0] > end) {
            arrows++;
            end = points[i][1];
        } else {
            // 更新当前箭能射到的最远位置（取交集的终点）
            end = min(end, points[i][1]);
        }
    }
    
    return arrows;
}
```

```python [Python 实现]
def find_min_arrow_shots(points: list[list[int]]) -> int:
    """
    用最少数量的箭引爆气球
    :param points: 气球区间列表
    :return: 最少需要的箭数
    """
    n = len(points)
    if n == 0:
        return 0
    
    # 按起点排序
    points.sort(key=lambda x: x[0])
    
    arrows = 1  # 至少需要一支箭
    end = points[0][1]  # 当前箭能射到的最远位置
    
    for i in range(1, n):
        # 如果当前气球的起点 > 当前箭能射到的最远位置
        # 需要新箭
        if points[i][0] > end:
            arrows += 1
            end = points[i][1]
        else:
            # 更新当前箭能射到的最远位置（取交集的终点）
            end = min(end, points[i][1])
    
    return arrows


# 测试示例
if __name__ == "__main__":
    points1 = [[10, 16], [2, 8], [1, 6], [7, 12]]
    print(f"最少箭数: {find_min_arrow_shots(points1)}")  # 2
    
    points2 = [[1, 2], [3, 4], [5, 6], [7, 8]]
    print(f"最少箭数: {find_min_arrow_shots(points2)}")  # 4
```

:::

### 加油站问题

::: info 问题描述（LeetCode 134）
在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。你有一辆油箱容量无限的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，初始油箱为空。如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
:::

**解题思路**：
- 如果总油量 < 总消耗，一定无解
- 如果总油量 >= 总消耗，一定有解
- 从起点出发，如果某点油量不够，则从该点之后重新开始

::: code-group

```cpp [C++ 实现]
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int n = gas.size();
    int totalGas = 0;   // 总油量
    int totalCost = 0;  // 总消耗
    int tank = 0;       // 当前油箱
    int start = 0;      // 起点
    
    for (int i = 0; i < n; i++) {
        totalGas += gas[i];
        totalCost += cost[i];
        tank += gas[i] - cost[i];
        
        // 如果从当前起点无法到达 i+1，则从 i+1 重新开始
        if (tank < 0) {
            start = i + 1;
            tank = 0;
        }
    }
    
    // 总油量 < 总消耗，无解
    if (totalGas < totalCost) {
        return -1;
    }
    
    return start % n;  // 处理环
}
```

```python [Python 实现]
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    加油站问题
    :param gas: 每个加油站的油量
    :param cost: 到下一个加油站的消耗
    :return: 出发点编号或 -1
    """
    n = len(gas)
    total_gas = 0    # 总油量
    total_cost = 0   # 总消耗
    tank = 0         # 当前油箱
    start = 0        # 起点
    
    for i in range(n):
        total_gas += gas[i]
        total_cost += cost[i]
        tank += gas[i] - cost[i]
        
        # 如果从当前起点无法到达 i+1，则从 i+1 重新开始
        if tank < 0:
            start = i + 1
            tank = 0
    
    # 总油量 < 总消耗，无解
    if total_gas < total_cost:
        return -1
    
    return start % n  # 处理环


# 测试示例
if __name__ == "__main__":
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    print(f"出发点: {can_complete_circuit(gas, cost)}")  # 3
    
    gas2 = [2, 3, 4]
    cost2 = [3, 4, 3]
    print(f"出发点: {can_complete_circuit(gas2, cost2)}")  # -1
```

:::

### 分发饼干

::: info 问题描述（LeetCode 455）
有孩子数组和饼干数组，每个孩子有一个胃口值，每块饼干有一个尺寸。只有当饼干的尺寸 >= 孩子的胃口值时，孩子才能满足。求最多能满足多少孩子。
:::

**解题思路**：
- 将孩子和饼干都排序
- 用最小的能满足孩子的饼干去满足每个孩子

::: code-group

```cpp [C++ 实现]
int findContentChildren(vector<int>& g, vector<int>& s) {
    // g: 孩子胃口数组
    // s: 饼干尺寸数组
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    
    int child = 0;  // 孩子指针
    int cookie = 0; // 饼干指针
    int satisfied = 0;  // 满足的孩子数
    
    while (child < g.size() && cookie < s.size()) {
        // 如果当前饼干能满足当前孩子
        if (s[cookie] >= g[child]) {
            satisfied++;
            child++;
        }
        // 无论是否满足，饼干都要移动（满足则用掉，不满足则太小换大的）
        cookie++;
    }
    
    return satisfied;
}
```

```python [Python 实现]
def find_content_children(g: list[int], s: list[int]) -> int:
    """
    分发饼干
    :param g: 孩子胃口数组
    :param s: 饼干尺寸数组
    :return: 最多能满足的孩子数
    """
    g.sort()
    s.sort()
    
    child = 0      # 孩子指针
    cookie = 0     # 饼干指针
    satisfied = 0  # 满足的孩子数
    
    while child < len(g) and cookie < len(s):
        # 如果当前饼干能满足当前孩子
        if s[cookie] >= g[child]:
            satisfied += 1
            child += 1
        # 无论是否满足，饼干都要移动
        cookie += 1
    
    return satisfied


# 测试示例
if __name__ == "__main__":
    g1 = [1, 2, 3]
    s1 = [1, 1]
    print(f"满足的孩子数: {find_content_children(g1, s1)}")  # 1
    
    g2 = [1, 2]
    s2 = [1, 2, 3]
    print(f"满足的孩子数: {find_content_children(g2, s2)}")  # 2
```

:::

---

## 贪心问题判定技巧

### 什么时候考虑贪心？

::: tip 贪心问题的特征
1. **问题可以分解为若干子问题**
2. **每个子问题的最优解能导致全局最优解**
3. **问题具有最优子结构**
4. **每一步的选择只依赖于当前状态，不依赖于未来**
:::

### 如何判断贪心是否正确？

#### 尝试反例

最直接的方法是尝试构造反例，如果找不到反例，则可能贪心正确。

```
示例：判断找零钱问题的贪心是否正确

硬币面值 [1, 5, 10, 25]（美元硬币）
目标金额：37

贪心：25 + 10 + 1 + 1 = 37，共 4 枚
最优：25 + 10 + 1 + 1 = 37，共 4 枚 ✓

尝试找反例：金额 30
贪心：25 + 5 = 30，共 2 枚
最优：25 + 5 = 30，共 2 枚 ✓

结论：对于美元硬币面值，贪心正确


硬币面值 [1, 3, 4]
目标金额：6

贪心：4 + 1 + 1 = 6，共 3 枚
最优：3 + 3 = 6，共 2 枚 ✗

结论：对于 [1, 3, 4] 面值，贪心不正确
```

#### 分析问题结构

| 问题类型 | 贪心是否适用 | 原因 |
|----------|--------------|------|
| 活动选择 | ✅ 适用 | 按结束时间排序后，贪心选择不冲突的活动 |
| 背包问题（分数） | ✅ 适用 | 可以取部分物品，按单位价值排序 |
| 背包问题（0-1） | ❌ 不适用 | 物品不能分割，选择可能相互影响 |
| 最短路径 | ❌ 不适用（一般图） | 当前最优可能导致后续无法到达 |
| 最小生成树 | ✅ 适用 | Kruskal 和 Prim 都是贪心算法 |

### 常见贪心模式

#### 模式一：排序 + 贪心

```
1. 按某种标准排序（时间、权重、优先级等）
2. 按顺序处理，每一步做最优选择
```

适用问题：活动选择、区间调度、分发饼干、任务调度

#### 模式二：维护最值

```
1. 维护一个变量（最大值、最小值、范围等）
2. 遍历过程中更新这个变量
3. 根据变量判断是否需要新的选择
```

适用问题：跳跃游戏、加油站、买卖股票

#### 模式三：双指针

```
1. 使用两个指针（左/右、前/后）
2. 根据条件移动指针
3. 记录满足条件的选择
```

适用问题：分发饼干、两数之和、盛最多水的容器

### 贪心 vs 动态规划的选择

| 判断依据 | 选择贪心 | 选择动态规划 |
|----------|----------|--------------|
| 子问题关系 | 独立 | 重叠 |
| 决策依赖 | 只依赖当前状态 | 依赖多个子问题的解 |
| 是否需要回溯 | 不需要 | 需要考虑所有可能 |
| 问题特征 | 明确的贪心策略 | 最优子结构 + 重叠子问题 |
| 正确性验证 | 需要证明 | 总是正确 |

::: warning 实践建议
1. **先尝试贪心**：贪心通常更简单，先尝试是否能用贪心解决
2. **寻找反例**：如果贪心不正确，尝试构造反例
3. **考虑动态规划**：如果贪心不正确，考虑使用动态规划
4. **证明正确性**：如果使用贪心，最好能证明其正确性
:::

---

## 练习题推荐

| 难度 | 题目 | 核心技巧 |
|------|------|----------|
| ⭐ | [LeetCode 455. 分发饼干](https://leetcode.cn/problems/assign-cookies/) | 排序 + 双指针 |
| ⭐ | [LeetCode 860. 柠檬水找零](https://leetcode.cn/problems/lemonade-change/) | 模拟 + 贪心 |
| ⭐⭐ | [LeetCode 55. 跳跃游戏](https://leetcode.cn/problems/jump-game/) | 维护最远位置 |
| ⭐⭐ | [LeetCode 45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/) | 贪心优化 |
| ⭐⭐ | [LeetCode 435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/) | 区间调度 |
| ⭐⭐ | [LeetCode 452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/) | 区间合并 |
| ⭐⭐ | [LeetCode 134. 加油站](https://leetcode.cn/problems/gas-station/) | 单遍扫描 |
| ⭐⭐⭐ | [LeetCode 135. 分发糖果](https://leetcode.cn/problems/candy/) | 双向遍历 |
| ⭐⭐⭐ | [LeetCode 406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/) | 插入排序 + 贪心 |
| ⭐⭐⭐ | [LeetCode 763. 划分字母区间](https://leetcode.cn/problems/partition-labels/) | 区间贪心 |
