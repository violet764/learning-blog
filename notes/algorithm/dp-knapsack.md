# 背包问题

背包问题是动态规划中最经典、最重要的专题之一。它研究的核心问题是：**给定一组物品和背包容量，如何选择物品使得总价值最大**。

📌 **核心价值**：背包问题不仅是算法竞赛和面试的高频考点，更是理解动态规划"状态定义"和"状态转移"的绝佳载体。

## 问题概述

### 问题分类

背包问题根据物品的选取限制，可以分为以下几类：

| 类型 | 物品数量 | 选取限制 | 典型变形 |
|------|----------|----------|----------|
| **01背包** | 每种只有 1 个 | 每种物品最多选 1 次 | 分割等和子集 |
| **完全背包** | 每种无限个 | 每种物品可选无限次 | 零钱兑换 |
| **多重背包** | 每种有限个 | 每种物品最多选 s[i] 次 | 庆功会 |
| **混合背包** | 以上混合 | 不同物品有不同限制 | 复杂场景 |
| **分组背包** | 物品分组 | 每组最多选 1 个 | 分组选择 |

### 解题核心思想

所有背包问题都遵循相同的解题框架：

```mermaid
graph TD
    A[背包问题] --> B{物品数量限制}
    B -->|每种1个| C[01背包]
    B -->|每种无限| D[完全背包]
    B -->|每种有限| E[多重背包]
    
    C --> F[状态: dp[i][j] = 前i个物品,容量j的最大价值]
    F --> G[转移: dp[i][j] = max不选, 选]
    G --> H[空间优化: 二维→一维]
```

**状态定义通式**：
- `dp[j]` = 背包容量为 `j` 时的最大价值（或方案数）
- `j` 表示当前背包已使用的容量

**状态转移核心**：
- 不选物品 `i`：`dp[j]` 保持不变
- 选物品 `i`：`dp[j] = dp[j - w[i]] + v[i]`（前提：`j >= w[i]`）

---

## 01背包

### 问题描述

::: info 经典01背包问题
有 `n` 个物品和容量为 `W` 的背包，第 `i` 个物品的重量为 `w[i]`，价值为 `v[i]`。每种物品最多选择一次，求背包能装的最大价值。
:::

### 二维DP解法

**状态定义**：
- `dp[i][j]` = 考虑前 `i` 个物品，背包容量为 `j` 时的最大价值

**状态转移方程**：
```
dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])
```
- `dp[i-1][j]`：不选第 `i` 个物品
- `dp[i-1][j-w[i]] + v[i]`：选第 `i` 个物品（需要剩余容量足够）

**初始化**：
- `dp[0][j] = 0`：没有物品时，价值为 0

::: code-group

```cpp [C++ 二维DP]
int knapsack_01_2d(int W, vector<int>& w, vector<int>& v) {
    int n = w.size();
    // dp[i][j]: 前i个物品, 容量j时的最大价值
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; i++) {          // 遍历每个物品
        for (int j = 0; j <= W; j++) {      // 遍历每个容量
            // 不选第i个物品
            dp[i][j] = dp[i - 1][j];
            
            // 选第i个物品（如果容量足够）
            if (j >= w[i - 1]) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j - w[i - 1]] + v[i - 1]);
            }
        }
    }
    
    return dp[n][W];
}
```

```python [Python 二维DP]
def knapsack_01_2d(W: int, w: list[int], v: list[int]) -> int:
    """01背包 - 二维DP解法"""
    n = len(w)
    # dp[i][j]: 前i个物品, 容量j时的最大价值
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):           # 遍历每个物品
        for j in range(W + 1):          # 遍历每个容量
            # 不选第i个物品
            dp[i][j] = dp[i - 1][j]
            
            # 选第i个物品（如果容量足够）
            if j >= w[i - 1]:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - w[i - 1]] + v[i - 1])
    
    return dp[n][W]
```

:::

**复杂度分析**：
- 时间复杂度：O(n × W)
- 空间复杂度：O(n × W)

### 空间优化（一维DP）

**优化原理**：注意到 `dp[i][j]` 只依赖于 `dp[i-1][...]`，即只依赖上一行，可以压缩为一维数组。

**关键点**：遍历容量时必须**从大到小**，避免同一物品被重复选取。

```
为什么必须逆序遍历？

正序遍历（错误）：
dp[3] = dp[3-2] + v  → 用了更新后的 dp[1]，相当于选了两次
dp[4] = dp[4-2] + v  → 用了更新后的 dp[2]，相当于选了两次

逆序遍历（正确）：
dp[4] = dp[4-2] + v  → 用的是旧的 dp[2]
dp[3] = dp[3-2] + v  → 用的是旧的 dp[1]
```

::: code-group

```cpp [C++ 一维DP]
int knapsack_01_1d(int W, vector<int>& w, vector<int>& v) {
    int n = w.size();
    // dp[j]: 容量j时的最大价值
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {          // 遍历每个物品
        for (int j = W; j >= w[i]; j--) {  // 逆序遍历容量
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    
    return dp[W];
}
```

```python [Python 一维DP]
def knapsack_01_1d(W: int, w: list[int], v: list[int]) -> int:
    """01背包 - 一维DP空间优化"""
    n = len(w)
    # dp[j]: 容量j时的最大价值
    dp = [0] * (W + 1)
    
    for i in range(n):                # 遍历每个物品
        for j in range(W, w[i] - 1, -1):  # 逆序遍历容量
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    
    return dp[W]
```

:::

**复杂度分析**：
- 时间复杂度：O(n × W)
- 空间复杂度：O(W) ← 从 O(n×W) 优化到 O(W)

### 典型题目：分割等和子集

::: info LeetCode 416. 分割等和子集
给定一个只包含正整数的非空数组，判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**示例**：`nums = [1, 5, 11, 5]` → 返回 `true`（可以分割为 [1, 5, 5] 和 [11]）
:::

**解题思路**：
- 问题转化为：能否从数组中选出一些元素，使得它们的和等于 `sum / 2`
- 这是一个**恰好装满**的01背包问题
- 物品重量 = 物品价值 = `nums[i]`，背包容量 = `sum / 2`

::: code-group

```cpp [C++ 解法]
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2 != 0) return false;  // 奇数无法平分
        
        int target = sum / 2;
        // dp[j]: 容量j时能否恰好装满
        vector<bool> dp(target + 1, false);
        dp[0] = true;  // 容量为0时视为装满
        
        for (int num : nums) {
            for (int j = target; j >= num; j--) {  // 逆序遍历
                dp[j] = dp[j] || dp[j - num];
            }
        }
        
        return dp[target];
    }
};
```

```python [Python 解法]
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:  # 奇数无法平分
            return False
        
        target = total // 2
        # dp[j]: 容量j时能否恰好装满
        dp = [False] * (target + 1)
        dp[0] = True  # 容量为0时视为装满
        
        for num in nums:
            for j in range(target, num - 1, -1):  # 逆序遍历
                dp[j] = dp[j] or dp[j - num]
        
        return dp[target]
```

:::

### 典型题目：目标和

::: info LeetCode 494. 目标和
给定一个非负整数数组 `nums` 和一个整数 `target`，向数组中每个整数前添加 `+` 或 `-`，使得运算结果等于 `target`，返回不同表达式的数目。

**示例**：`nums = [1, 1, 1, 1, 1], target = 3` → 返回 `5`
:::

**解题思路**：
- 设添加 `+` 的元素和为 `P`，添加 `-` 的元素和为 `N`
- 则 `P - N = target`，且 `P + N = sum`
- 解得 `P = (sum + target) / 2`
- 问题转化为：有多少种方式选出元素使其和为 `P`（方案数问题）

::: code-group

```cpp [C++ 解法]
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        // (sum + target) 必须是非负偶数
        if ((sum + target) % 2 != 0 || sum + target < 0) return 0;
        
        int bagSize = (sum + target) / 2;
        // dp[j]: 填满容量j的方案数
        vector<int> dp(bagSize + 1, 0);
        dp[0] = 1;  // 容量为0有1种方案（什么都不选）
        
        for (int num : nums) {
            for (int j = bagSize; j >= num; j--) {  // 逆序遍历
                dp[j] += dp[j - num];  // 累加方案数
            }
        }
        
        return dp[bagSize];
    }
};
```

```python [Python 解法]
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        # (sum + target) 必须是非负偶数
        if (total + target) % 2 != 0 or total + target < 0:
            return 0
        
        bag_size = (total + target) // 2
        # dp[j]: 填满容量j的方案数
        dp = [0] * (bag_size + 1)
        dp[0] = 1  # 容量为0有1种方案
        
        for num in nums:
            for j in range(bag_size, num - 1, -1):  # 逆序遍历
                dp[j] += dp[j - num]  # 累加方案数
        
        return dp[bag_size]
```

:::

---

## 完全背包

### 与01背包的区别

| 特性 | 01背包 | 完全背包 |
|------|--------|----------|
| 物品数量 | 每种 1 个 | 每种无限个 |
| 容量遍历顺序 | **逆序** | **正序** |
| 典型应用 | 分割子集 | 零钱兑换 |

**核心区别**：完全背包中同一物品可以被多次选取，因此遍历容量时要**正序**，这样 `dp[j - w[i]]` 用的就是当前物品已经更新过的值，实现了"多次选取"。

```
完全背包正序遍历的含义：

dp[2] = dp[2-2] + v  → 选1次物品
dp[4] = dp[4-2] + v  → dp[2]已更新，相当于在此基础上再选1次 = 选2次物品
dp[6] = dp[6-2] + v  → dp[4]已更新，相当于在此基础上再选1次 = 选3次物品
```

### 状态转移方程

```
dp[j] = max(dp[j], dp[j - w[i]] + v[i])
```

与01背包相同，但遍历顺序从逆序变为正序。

::: code-group

```cpp [C++ 完全背包]
int knapsack_complete(int W, vector<int>& w, vector<int>& v) {
    int n = w.size();
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {          // 遍历每个物品
        for (int j = w[i]; j <= W; j++) {  // 正序遍历容量
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    
    return dp[W];
}
```

```python [Python 完全背包]
def knapsack_complete(W: int, w: list[int], v: list[int]) -> int:
    """完全背包 - 每种物品可选无限次"""
    n = len(w)
    dp = [0] * (W + 1)
    
    for i in range(n):                    # 遍历每个物品
        for j in range(w[i], W + 1):      # 正序遍历容量
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    
    return dp[W]
```

:::

### 典型题目：零钱兑换

::: info LeetCode 322. 零钱兑换
给定不同面额的硬币 `coins` 和一个总金额 `amount`，计算凑成总金额所需的**最少硬币个数**。如果无法凑成，返回 -1。

**示例**：`coins = [1, 2, 5], amount = 11` → 返回 `3`（11 = 5 + 5 + 1）
:::

**解题思路**：
- 这是一个**完全背包 + 恰好装满 + 最小值**问题
- 物品重量 = 物品价值 = `coins[i]`
- 求**最小硬币数**，初始化为 `INF`，`dp[0] = 0`

::: code-group

```cpp [C++ 解法]
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // dp[j]: 凑成金额j的最少硬币数
        const int INF = amount + 1;  // 一个不可能达到的大值
        vector<int> dp(amount + 1, INF);
        dp[0] = 0;  // 金额为0需要0个硬币
        
        for (int coin : coins) {
            for (int j = coin; j <= amount; j++) {  // 正序遍历（完全背包）
                dp[j] = min(dp[j], dp[j - coin] + 1);
            }
        }
        
        return dp[amount] == INF ? -1 : dp[amount];
    }
};
```

```python [Python 解法]
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        # dp[j]: 凑成金额j的最少硬币数
        INF = amount + 1  # 一个不可能达到的大值
        dp = [INF] * (amount + 1)
        dp[0] = 0  # 金额为0需要0个硬币
        
        for coin in coins:
            for j in range(coin, amount + 1):  # 正序遍历（完全背包）
                dp[j] = min(dp[j], dp[j - coin] + 1)
        
        return -1 if dp[amount] == INF else dp[amount]
```

:::

### 典型题目：完全平方数

::: info LeetCode 279. 完全平方数
给定整数 `n`，返回和为 `n` 的完全平方数的**最少数量**。

**示例**：`n = 12` → 返回 `3`（12 = 4 + 4 + 4）
:::

**解题思路**：
- 完全平方数可以无限使用 → 完全背包
- 物品为 `1², 2², 3², ...`，直到 `i² <= n`

::: code-group

```cpp [C++ 解法]
class Solution {
public:
    int numSquares(int n) {
        // dp[j]: 组成j的最少完全平方数个数
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;
        
        for (int i = 1; i * i <= n; i++) {  // 遍历完全平方数
            int square = i * i;
            for (int j = square; j <= n; j++) {  // 正序遍历
                dp[j] = min(dp[j], dp[j - square] + 1);
            }
        }
        
        return dp[n];
    }
};
```

```python [Python 解法]
class Solution:
    def numSquares(self, n: int) -> int:
        # dp[j]: 组成j的最少完全平方数个数
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        i = 1
        while i * i <= n:  # 遍历完全平方数
            square = i * i
            for j in range(square, n + 1):  # 正序遍历
                dp[j] = min(dp[j], dp[j - square] + 1)
            i += 1
        
        return dp[n]
```

:::

---

## 多重背包

### 问题描述

::: info 多重背包问题
有 `n` 种物品，第 `i` 种物品有 `s[i]` 个，重量为 `w[i]`，价值为 `v[i]`。背包容量为 `W`，求最大价值。
:::

### 基础解法：展开为01背包

将 `s[i]` 个相同物品展开为 `s[i]` 个独立物品，转换为01背包问题。

::: code-group

```cpp [C++ 基础解法]
int knapsack_multiple_basic(int W, vector<int>& w, vector<int>& v, vector<int>& s) {
    int n = w.size();
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        // 将s[i]个物品展开，按01背包处理
        for (int k = 0; k < s[i]; k++) {
            for (int j = W; j >= w[i]; j--) {  // 逆序遍历（01背包）
                dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
            }
        }
    }
    
    return dp[W];
}
```

```python [Python 基础解法]
def knapsack_multiple_basic(W: int, w: list[int], v: list[int], s: list[int]) -> int:
    """多重背包基础解法：展开为01背包"""
    n = len(w)
    dp = [0] * (W + 1)
    
    for i in range(n):
        # 将s[i]个物品展开，按01背包处理
        for _ in range(s[i]):
            for j in range(W, w[i] - 1, -1):  # 逆序遍历（01背包）
                dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    
    return dp[W]
```

:::

**复杂度分析**：
- 时间复杂度：O(W × Σs[i])，当 `s[i]` 很大时效率低

### 二进制优化

**优化原理**：将 `s[i]` 个物品用二进制拆分，例如 `s[i] = 13` 可拆分为 `1, 2, 4, 6`（即 `1 + 2 + 4 + 6 = 13`），任何 `1~13` 的数量都可以由这四个数组合得到。

```
二进制拆分原理：

s[i] = 13
拆分为: 1, 2, 4, 6 (注意: 最后一个数 = 13 - 1 - 2 - 4 = 6)

组合表示:
1 = 1
2 = 2
3 = 1 + 2
4 = 4
5 = 1 + 4
...
13 = 1 + 2 + 4 + 6

这样 13 个物品只需拆成 4 组，而非 13 个独立物品
```

::: code-group

```cpp [C++ 二进制优化]
int knapsack_multiple_binary(int W, vector<int>& w, vector<int>& v, vector<int>& s) {
    int n = w.size();
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        // 二进制拆分
        for (int k = 1; k <= s[i]; k *= 2) {
            int count = min(k, s[i] - k + 1);  // 当前组的大小
            int weight = count * w[i];
            int value = count * v[i];
            
            for (int j = W; j >= weight; j--) {  // 01背包
                dp[j] = max(dp[j], dp[j - weight] + value);
            }
        }
    }
    
    return dp[W];
}
```

```python [Python 二进制优化]
def knapsack_multiple_binary(W: int, w: list[int], v: list[int], s: list[int]) -> int:
    """多重背包 - 二进制优化"""
    n = len(w)
    dp = [0] * (W + 1)
    
    for i in range(n):
        k = 1
        remaining = s[i]
        while remaining > 0:
            # 当前组的大小
            count = min(k, remaining)
            weight = count * w[i]
            value = count * v[i]
            
            for j in range(W, weight - 1, -1):  # 01背包
                dp[j] = max(dp[j], dp[j - weight] + value)
            
            remaining -= count
            k *= 2
    
    return dp[W]
```

:::

**复杂度分析**：
- 时间复杂度：O(W × Σlog(s[i]))，显著优化

### 单调队列优化

对于大规模数据，可使用单调队列优化到 O(W × n)，但实现较复杂，竞赛中二进制优化通常够用。

::: details 单调队列优化思路

**核心思想**：将容量 `j` 按照 `j % w[i]` 分组，每组内用单调队列维护滑动窗口最大值。

```cpp
int knapsack_multiple_monotonic(int W, vector<int>& w, vector<int>& v, vector<int>& s) {
    int n = w.size();
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        vector<int> prev = dp;  // 保存上一轮结果
        for (int r = 0; r < w[i]; r++) {  // 按余数分组
            deque<int> dq;
            for (int j = r, k = 0; j <= W; j += w[i], k++) {
                // 维护单调队列（窗口大小为 s[i] + 1）
                while (!dq.empty() && k - dq.front() > s[i]) {
                    dq.pop_front();
                }
                while (!dq.empty() && prev[j - (k - dq.back()) * w[i]] - (k - dq.back()) * v[i] 
                       <= prev[j] - k * v[i]) {
                    dq.pop_back();
                }
                dq.push_back(k);
                dp[j] = prev[j - (k - dq.front()) * w[i]] + (k - dq.front()) * v[i];
            }
        }
    }
    
    return dp[W];
}
```

:::

---

## 混合背包与分组背包

### 混合背包

::: info 问题描述
物品分为三类：
- 01背包类：每种只能选 1 次
- 完全背包类：每种可选无限次
- 多重背包类：每种可选有限次

求最大价值。
:::

**解法思路**：根据物品类型选择不同的遍历方式。

::: code-group

```cpp [C++ 混合背包]
int knapsack_mixed(int W, vector<int>& w, vector<int>& v, vector<int>& s) {
    // s[i] = 0 表示完全背包
    // s[i] = 1 表示01背包
    // s[i] > 1 表示多重背包
    int n = w.size();
    vector<int> dp(W + 1, 0);
    
    for (int i = 0; i < n; i++) {
        if (s[i] == 0) {  // 完全背包：正序遍历
            for (int j = w[i]; j <= W; j++) {
                dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
            }
        } else if (s[i] == 1) {  // 01背包：逆序遍历
            for (int j = W; j >= w[i]; j--) {
                dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
            }
        } else {  // 多重背包：二进制拆分 + 逆序遍历
            for (int k = 1; k <= s[i]; k *= 2) {
                int count = min(k, s[i] - k + 1);
                int weight = count * w[i];
                int value = count * v[i];
                for (int j = W; j >= weight; j--) {
                    dp[j] = max(dp[j], dp[j - weight] + value);
                }
            }
        }
    }
    
    return dp[W];
}
```

```python [Python 混合背包]
def knapsack_mixed(W: int, w: list[int], v: list[int], s: list[int]) -> int:
    """
    混合背包
    s[i] = 0: 完全背包
    s[i] = 1: 01背包
    s[i] > 1: 多重背包
    """
    n = len(w)
    dp = [0] * (W + 1)
    
    for i in range(n):
        if s[i] == 0:  # 完全背包：正序遍历
            for j in range(w[i], W + 1):
                dp[j] = max(dp[j], dp[j - w[i]] + v[i])
        elif s[i] == 1:  # 01背包：逆序遍历
            for j in range(W, w[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - w[i]] + v[i])
        else:  # 多重背包：二进制拆分
            k = 1
            remaining = s[i]
            while remaining > 0:
                count = min(k, remaining)
                weight = count * w[i]
                value = count * v[i]
                for j in range(W, weight - 1, -1):
                    dp[j] = max(dp[j], dp[j - weight] + value)
                remaining -= count
                k *= 2
    
    return dp[W]
```

:::

### 分组背包

::: info 问题描述
物品被分为若干组，每组最多选择 1 个物品，求最大价值。
:::

**解法思路**：外层遍历组，内层遍历容量，最后遍历组内物品。

::: code-group

```cpp [C++ 分组背包]
int knapsack_group(int W, vector<vector<pair<int, int>>>& groups) {
    // groups[i] = 第i组的物品列表，每个物品用 {weight, value} 表示
    vector<int> dp(W + 1, 0);
    
    for (auto& group : groups) {  // 遍历每个组
        for (int j = W; j >= 0; j--) {  // 逆序遍历容量
            for (auto& item : group) {  // 遍历组内物品
                int w = item.first, v = item.second;
                if (j >= w) {
                    dp[j] = max(dp[j], dp[j - w] + v);
                }
            }
        }
    }
    
    return dp[W];
}
```

```python [Python 分组背包]
def knapsack_group(W: int, groups: list[list[tuple[int, int]]]) -> int:
    """
    分组背包
    groups[i] = 第i组的物品列表，每个物品用 (weight, value) 表示
    """
    dp = [0] * (W + 1)
    
    for group in groups:  # 遍历每个组
        for j in range(W, -1, -1):  # 逆序遍历容量
            for w, v in group:  # 遍历组内物品
                if j >= w:
                    dp[j] = max(dp[j], dp[j - w] + v)
    
    return dp[W]
```

:::

---

## 背包问题技巧总结

### 恰好装满 vs 最大价值

| 问题类型 | 初始化方式 | 说明 |
|----------|------------|------|
| 最大价值 | `dp[0...W] = 0` | 容量为 0~W 都可以是起点 |
| 恰好装满 | `dp[0] = 0, dp[1...W] = -INF` | 只有容量为 0 是合法起点 |

```cpp
// 最大价值问题
vector<int> dp(W + 1, 0);  // 全部初始化为0

// 恰好装满问题
vector<int> dp(W + 1, INT_MIN);  // 初始化为负无穷
dp[0] = 0;  // 只有容量0是合法的
```

### 方案数问题

当求方案数时，状态转移变为**累加**而非取最大值。

```cpp
// 求方案数
dp[j] += dp[j - w[i]];  // 累加方案数

// 求最大价值
dp[j] = max(dp[j], dp[j - w[i]] + v[i]);  // 取最大值
```

**方案数初始化**：
- `dp[0] = 1`：容量为 0 有 1 种方案（什么都不选）
- `dp[1...W] = 0`：其他容量初始方案数为 0

### 路径回溯

如果需要输出具体选择了哪些物品，需要额外记录或使用二维DP回溯。

```cpp
// 方法1：二维DP回溯
void backtrack(int i, int j, vector<vector<int>>& dp, vector<int>& w) {
    if (i == 0) return;
    if (dp[i][j] != dp[i-1][j]) {  // 选择了第i个物品
        cout << "选择物品 " << i << endl;
        backtrack(i - 1, j - w[i-1], dp, w);
    } else {
        backtrack(i - 1, j, dp, w);
    }
}

// 方法2：记录选择的物品
vector<int> choice(n, 0);  // 记录每个物品选择次数
// 在DP过程中记录选择情况
```

### 代码模板速查

```cpp
// 01背包模板
for (int i = 0; i < n; i++) {
    for (int j = W; j >= w[i]; j--) {  // 逆序
        dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}

// 完全背包模板
for (int i = 0; i < n; i++) {
    for (int j = w[i]; j <= W; j++) {  // 正序
        dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}

// 多重背包模板（二进制优化）
for (int i = 0; i < n; i++) {
    for (int k = 1; k <= s[i]; k *= 2) {
        int cnt = min(k, s[i] - k + 1);
        for (int j = W; j >= cnt * w[i]; j--) {  // 逆序
            dp[j] = max(dp[j], dp[j - cnt * w[i]] + cnt * v[i]);
        }
    }
}
```

---

## 练习题推荐

| 难度 | 题目 | 类型 | 核心技巧 |
|------|------|------|----------|
| ⭐⭐ | [LeetCode 416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/) | 01背包 | 恰好装满 |
| ⭐⭐ | [LeetCode 494. 目标和](https://leetcode.cn/problems/target-sum/) | 01背包 | 方案数 |
| ⭐⭐ | [LeetCode 1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/) | 01背包 | 转化思路 |
| ⭐⭐ | [LeetCode 322. 零钱兑换](https://leetcode.cn/problems/coin-change/) | 完全背包 | 最小数量 |
| ⭐⭐ | [LeetCode 518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/) | 完全背包 | 方案数 |
| ⭐⭐ | [LeetCode 279. 完全平方数](https://leetcode.cn/problems/perfect-squares/) | 完全背包 | 最小数量 |
| ⭐⭐⭐ | [LeetCode 377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/) | 完全背包 | 排列数 |
| ⭐⭐⭐ | [LeetCode 474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/) | 01背包 | 二维容量 |
| ⭐⭐⭐ | [AcWing 5. 多重背包问题 II](https://www.acwing.com/problem/content/5/) | 多重背包 | 二进制优化 |
