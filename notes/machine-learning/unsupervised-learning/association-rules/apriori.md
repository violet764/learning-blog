# Apriori算法

## 概述

挖掘频繁项集的经典算法，用于关联规则学习。基于Apriori性质进行高效剪枝。

## 算法原理

### Apriori性质

**频繁项集的所有非空子集也必须是频繁的**。

形式化表述：如果项集I是频繁的，则对于任意I'⊆I，I'也是频繁的。

### 支持度和置信度

- **支持度**：项集在事务中出现的频率
  $support(X) = \frac{|\{t \in T \mid X \subseteq t\}|}{|T|}$

- **置信度**：规则X→Y的可信度
  $confidence(X \rightarrow Y) = \frac{support(X \cup Y)}{support(X)}$

## 算法步骤

### 1. 频繁项集挖掘
1. 扫描事务数据库，计算所有1-项集的支持度
2. 删除支持度低于最小支持度的项集，得到频繁1-项集L₁
3. 使用Lₖ生成候选(k+1)-项集Cₖ₊₁
4. 扫描数据库，计算Cₖ₊₁的支持度
5. 删除支持度低于阈值的项集，得到Lₖ₊₁
6. 重复步骤3-5直到不能再生成频繁项集

### 2. 关联规则生成
1. 对于每个频繁项集l，生成所有非空子集
2. 对于每个子集s，生成规则s→(l-s)
3. 计算规则的置信度
4. 保留置信度高于最小置信度的规则

## Python实现

```python
import numpy as np
import pandas as pd
from itertools import combinations

class Apriori:
    """
    Apriori算法实现
    """
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.rules = []
    
    def fit(self, transactions):
        """
        挖掘频繁项集和关联规则
        """
        self.transactions = transactions
        self.n_transactions = len(transactions)
        
        # 挖掘频繁项集
        self._find_frequent_itemsets()
        
        # 生成关联规则
        self._generate_rules()
        
        return self
    
    def _find_frequent_itemsets(self):
        """
        挖掘频繁项集
        """
        # 初始化频繁1-项集
        items = set()
        for transaction in self.transactions:
            items.update(transaction)
        
        # 计算1-项集支持度
        k = 1
        frequent_items = {}
        for item in items:
            support = self._calculate_support([item])
            if support >= self.min_support:
                frequent_items[frozenset([item])] = support
        
        self.frequent_itemsets.append(frequent_items)
        
        # 迭代生成更高阶的频繁项集
        k = 2
        while True:
            # 生成候选k-项集
            candidates = self._generate_candidates(self.frequent_itemsets[k-2], k)
            
            if not candidates:
                break
            
            # 计算支持度并筛选
            frequent_k_items = {}
            for candidate in candidates:
                support = self._calculate_support(candidate)
                if support >= self.min_support:
                    frequent_k_items[candidate] = support
            
            if not frequent_k_items:
                break
            
            self.frequent_itemsets.append(frequent_k_items)
            k += 1
    
    def _generate_candidates(self, prev_frequent, k):
        """
        生成候选k-项集
        """
        candidates = set()
        
        # 获取所有项
        items = set()
        for itemset in prev_frequent.keys():
            items.update(itemset)
        
        # 生成所有可能的k-项集
        for combination in combinations(items, k):
            candidate = frozenset(combination)
            
            # 检查Apriori性质：所有(k-1)-子集必须是频繁的
            valid = True
            for subset in combinations(candidate, k-1):
                if frozenset(subset) not in prev_frequent:
                    valid = False
                    break
            
            if valid:
                candidates.add(candidate)
        
        return list(candidates)
    
    def _calculate_support(self, itemset):
        """
        计算项集支持度
        """
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        
        return count / self.n_transactions
    
    def _generate_rules(self):
        """
        生成关联规则
        """
        self.rules = []
        
        # 从2-项集开始生成规则
        for i in range(1, len(self.frequent_itemsets)):
            for itemset, support in self.frequent_itemsets[i].items():
                # 生成所有可能的规则
                items = list(itemset)
                
                # 生成所有非空真子集
                for j in range(1, len(items)):
                    for antecedent in combinations(items, j):
                        antecedent_set = frozenset(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # 计算置信度
                        antecedent_support = self.frequent_itemsets[len(antecedent_set)-1][antecedent_set]
                        confidence = support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': antecedent_set,
                                'consequent': consequent_set,
                                'support': support,
                                'confidence': confidence,
                                'lift': confidence / self.frequent_itemsets[len(consequent_set)-1][consequent_set]
                            })
    
    def get_rules(self):
        """
        获取关联规则
        """
        return self.rules
    
    def print_results(self):
        """
        打印结果
        """
        print("=== 频繁项集 ===")
        for i, itemsets in enumerate(self.frequent_itemsets, 1):
            print(f"\n{i}-项集:")
            for itemset, support in itemsets.items():
                print(f"  {set(itemset)}: 支持度 = {support:.3f}")
        
        print("\n=== 关联规则 ===")
        for rule in sorted(self.rules, key=lambda x: x['confidence'], reverse=True):
            print(f"{set(rule['antecedent'])} => {set(rule['consequent'])}")
            print(f"  支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, 提升度: {rule['lift']:.3f}")

def apriori_demo():
    """
    Apriori算法演示
    """
    # 示例事务数据（购物篮数据）
    transactions = [
        {'牛奶', '面包', '黄油'},
        {'啤酒', '面包'},
        {'牛奶', '啤酒', '黄油'},
        {'牛奶', '面包'},
        {'啤酒', '面包'},
        {'牛奶', '啤酒'},
        {'牛奶', '面包', '黄油'},
        {'啤酒', '黄油'},
        {'牛奶', '面包'},
        {'啤酒', '面包', '黄油'}
    ]
    
    # 创建Apriori模型
    apriori = Apriori(min_support=0.3, min_confidence=0.6)
    
    # 拟合模型
    apriori.fit(transactions)
    
    # 打印结果
    apriori.print_results()
    
    return apriori

# 高级功能：频繁模式增长（FP-Growth）
def fp_growth_implementation(transactions, min_support):
    """
    FP-Growth算法实现（频繁模式增长）
    """
    from collections import defaultdict
    
    # 1. 构建FP树
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    # 过滤频繁项
    frequent_items = {item for item, count in item_counts.items() 
                     if count / len(transactions) >= min_support}
    
    # 按频率排序
    sorted_items = sorted(frequent_items, key=lambda x: item_counts[x], reverse=True)
    
    # 2. 构建FP树（简化实现）
    # 这里简化实现，实际FP树更复杂
    
    print("频繁项:", sorted_items)
    print("项计数:", {item: item_counts[item] for item in sorted_items})
    
    return frequent_items

if __name__ == "__main__":
    # Apriori演示
    print("Apriori算法演示:")
    apriori_demo()
    
    # FP-Growth演示
    print("\n\nFP-Growth算法演示:")
    transactions = [
        {'A', 'B', 'C'},
        {'A', 'C'},
        {'A', 'D'},
        {'B', 'E', 'F'}
    ]
    fp_growth_implementation(transactions, min_support=0.3)
```

## 数学分析

### Apriori性质证明

设I是频繁项集，支持度s(I) ≥ min_support。对于任意I'⊆I，由于I'出现在所有包含I的事务中，因此：

s(I') ≥ s(I) ≥ min_support

所以I'也是频繁的。

### 候选项集数量

对于包含n个不同项的数据集，可能的k-项集数量为：

$C_n^k = \frac{n!}{k!(n-k)!}$

Apriori算法通过剪枝显著减少了候选项集数量。

## 复杂度分析

- **时间复杂度**：O(2ⁿ) 最坏情况，但实际中由于剪枝效果显著降低
- **空间复杂度**：O(n²) 需要存储候选项集和支持度计数

## 优化策略

### 1. 事务压缩
- 删除不包含任何频繁项的事务
- 减少扫描的数据量

### 2. 分区技术
- 将数据库分成多个分区
- 分别挖掘每个分区的频繁项集
- 合并结果

### 3. 采样技术
- 使用数据样本进行挖掘
- 调整支持度阈值

## FP-Growth算法

### 算法优势
- 不需要生成候选项集
- 使用FP树压缩数据
- 效率高于Apriori

### 算法步骤
1. 构建频繁模式树（FP树）
2. 从FP树挖掘频繁模式
3. 递归构建条件FP树

## 应用场景

### 购物篮分析
- 发现商品购买关联关系
- 优化商品摆放位置
- 制定促销策略

### 推荐系统
- 基于关联规则生成推荐
- 发现用户行为模式

### 网络日志分析
- 发现用户访问模式
- 优化网站结构

### 医疗诊断
- 发现症状与疾病的关联
- 辅助医疗决策

## 注意事项

### 参数选择
- **最小支持度**：太高会丢失重要规则，太低会产生过多无意义规则
- **最小置信度**：控制规则的可信度

### 规则评估
- **提升度**：衡量规则的有效性
- **杠杆度**：衡量规则的重要性
- **确信度**：衡量规则的强度

### 局限性
- 只能发现正相关，不能发现负相关
- 对稀疏数据效果差
- 计算复杂度高