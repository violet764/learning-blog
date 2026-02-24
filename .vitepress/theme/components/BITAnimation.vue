<template>
  <div class="bit-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <select v-model="opType" class="sel">
        <option value="update">单点更新</option>
        <option value="query">前缀查询</option>
        <option value="range">区间查询</option>
      </select>
      <input v-model.number="idx" type="number" min="1" :max="n" placeholder="位置" class="inp num" />
      <input v-model.number="val" type="number" placeholder="值" class="inp num" v-if="opType === 'update'" />
      <input v-model.number="rIdx" type="number" min="1" :max="n" placeholder="右端" class="inp num" v-if="opType === 'range'" />
      <button @click="run" class="btn btn-blue" :disabled="playing">执行</button>
      <button @click="resetAll" class="btn btn-gray">重置</button>
    </div>

    <!-- 原始数组显示 -->
    <div class="arr-row">
      <span class="lbl">原数组 a[]:</span>
      <div class="arr-cells">
        <div class="arr-cell" v-for="(v, i) in arr" :key="i">
          <span class="idx">{{ i }}</span>
          <span class="val" :class="{ hl: hlArr.includes(i) }">{{ v }}</span>
        </div>
      </div>
    </div>

    <!-- 树状数组显示 -->
    <div class="arr-row">
      <span class="lbl">树数组 t[]:</span>
      <div class="arr-cells">
        <div class="arr-cell" v-for="(v, i) in tree" :key="i">
          <span class="idx">{{ i }}</span>
          <span class="val" :class="{ hl: hlTree.includes(i), cur: curIdx === i }">{{ v }}</span>
          <span class="lb" v-if="i > 0">lowbit={{ lowbit(i) }}</span>
        </div>
      </div>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 二进制可视化 -->
    <div class="bin-row" v-if="showBin">
      <span class="lbl">二进制:</span>
      <span class="bin-val">{{ binStr }}</span>
      <span class="bin-desc">{{ binDesc }}</span>
    </div>

    <!-- 播放控制 -->
    <div class="ctl-row" v-if="acts.length">
      <button @click="prev" class="btn btn-sm" :disabled="step <= 0 || playing">上一步</button>
      <button @click="toggle" class="btn btn-sm btn-blue">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="nextStep" class="btn btn-sm" :disabled="step >= acts.length || playing">下一步</button>
      <button @click="resetAnim" class="btn btn-sm btn-gray">重置</button>
    </div>

    <div class="prog" v-if="acts.length">{{ step }} / {{ acts.length }}</div>

    <!-- 图例 -->
    <div class="leg">
      <span><i class="d hl"></i>涉及</span>
      <span><i class="d cur"></i>当前</span>
      <span class="tip">lowbit(x) = x & (-x) = 2^k (k是x末尾0的个数)</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const n = 8// 数组大小
const arr = ref([0, 1, 3, 5, 7, 2, 4, 6, 8])  // 原始数组（下标从1开始）
const tree = ref([0, 0, 0, 0, 0, 0, 0, 0, 0])  // 树状数组

const opType = ref('update')
const idx = ref(1)
const rIdx = ref(1)
const val = ref(1)
const msg = ref('选择操作类型并输入参数')
const playing = ref(false)
const step = ref(0)

const hlArr = ref([])
const hlTree = ref([])
const curIdx = ref(-1)
const showBin = ref(false)
const binStr = ref('')
const binDesc = ref('')

const acts = ref([])
let timer = null

// lowbit 函数
function lowbit(x) {
  return x & (-x)
}

// 初始化树状数组
function buildTree() {
  tree.value = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  for (let i = 1; i <= n; i++) {
    let j = i
    while (j <= n) {
      tree.value[j] += arr.value[i]
      j += lowbit(j)
    }
  }
}

// 生成更新动画
function genUpdateAnim(i, v) {
  acts.value = []
  const delta = v - arr.value[i]
  arr.value[i] = v

  acts.value.push({
    t: 'start',
    m: `更新位置 ${i} 的值为 ${v}，增量 delta = ${delta}`
  })

  if (delta !== 0) {
    let j = i
    while (j <= n) {
      const lb = lowbit(j)
      acts.value.push({
        t: 'update',
        idx: j,
        delta: delta,
        lb: lb,
        m: `t[${j}] += ${delta}，lowbit(${j}) = ${lb}`
      })
      j += lb
    }
  }

  acts.value.push({
    t: 'done',
    m: `更新完成！`
  })
}

// 生成前缀查询动画
function genQueryAnim(i) {
  acts.value = []
  let sum = 0

  acts.value.push({
    t: 'start',
    m: `查询前缀和 sum[${i}]`
  })

  let j = i
  while (j > 0) {
    const lb = lowbit(j)
    acts.value.push({
      t: 'query',
      idx: j,
      sum: sum + tree.value[j],
      lb: lb,
      m: `sum += t[${j}] = ${tree.value[j]}，j -= lowbit(${j}) = ${lb}`
    })
    sum += tree.value[j]
    j -= lb
  }

  acts.value.push({
    t: 'done',
    m: `前缀和 sum[${i}] = ${sum}`
  })
}

// 生成区间查询动画
function genRangeAnim(l, r) {
  acts.value = []

  acts.value.push({
    t: 'start',
    m: `区间查询 [${l}, ${r}]，需要计算 sum[${r}] - sum[${l-1}]`
  })

  // 先计算 sum[r]
  let sumR = 0
  let j = r
  acts.value.push({
    t: 'info',
    m: `--- 计算 sum[${r}] ---`
  })
  while (j > 0) {
    const lb = lowbit(j)
    acts.value.push({
      t: 'queryR',
      idx: j,
      sum: sumR + tree.value[j],
      lb: lb,
      m: `sumR += t[${j}] = ${tree.value[j]}`
    })
    sumR += tree.value[j]
    j -= lb
  }

  // 再计算 sum[l-1]
  let sumL = 0
  j = l - 1
  if (j > 0) {
    acts.value.push({
      t: 'info',
      m: `--- 计算 sum[${l-1}] ---`
    })
    while (j > 0) {
      const lb = lowbit(j)
      acts.value.push({
        t: 'queryL',
        idx: j,
        sum: sumL + tree.value[j],
        lb: lb,
        m: `sumL += t[${j}] = ${tree.value[j]}`
      })
      sumL += tree.value[j]
      j -= lb
    }
  }

  acts.value.push({
    t: 'done',
    m: `区间和 [${l}, ${r}] = sumR - sumL = ${sumR} - ${sumL} = ${sumR - sumL}`
  })
}

// 执行动作
function exec(a) {
  hlArr.value = []
  hlTree.value = []
  curIdx.value = -1
  showBin.value = false

  switch (a.t) {
    case 'start':
    case 'info':
      break

    case 'update':
      curIdx.value = a.idx
      hlTree.value.push(a.idx)
      tree.value[a.idx] += a.delta
      showBin.value = true
      binStr.value = `${a.idx} = ${a.idx.toString(2).padStart(4, '0')}`
      binDesc.value = `lowbit = ${a.lb} (${a.lb.toString(2)})`
      break

    case 'query':
      curIdx.value = a.idx
      hlTree.value.push(a.idx)
      showBin.value = true
      binStr.value = `${a.idx} = ${a.idx.toString(2).padStart(4, '0')}`
      binDesc.value = `lowbit = ${a.lb}，下个位置: ${a.idx - a.lb}`
      break

    case 'queryR':
      curIdx.value = a.idx
      hlTree.value.push(a.idx)
      break

    case 'queryL':
      curIdx.value = a.idx
      hlTree.value.push(a.idx)
      break

    case 'done':
      break
  }

  msg.value = a.m
}

// 下一步
function nextStep() {
  if (step.value >= acts.value.length) return
  exec(acts.value[step.value])
  step.value++

  if (step.value >= acts.value.length) {
    playing.value = false
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }
}

// 上一步
function prev() {
  if (step.value <= 0) return
  playing.value = false
  if (timer) {
    clearTimeout(timer)
    timer = null
  }

  step.value = 0
  resetAnim()
  for (let i = 0; i < step.value - 1; i++) {
    exec(acts.value[i])
  }
}

// 切换播放
function toggle() {
  if (playing.value) {
    playing.value = false
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  } else {
    if (step.value >= acts.value.length) resetAnim()
    playing.value = true
    autoPlay()
  }
}

// 自动播放
function autoPlay() {
  if (!playing.value) return
  if (step.value < acts.value.length) {
    nextStep()
    timer = setTimeout(autoPlay, 600)
  } else {
    playing.value = false
  }
}

// 重置动画
function resetAnim() {
  playing.value = false
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  step.value = 0
  hlArr.value = []
  hlTree.value = []
  curIdx.value = -1
  showBin.value = false
  msg.value = '选择操作类型并输入参数'
}

// 重置全部
function resetAll() {
  resetAnim()
  arr.value = [0, 1, 3, 5, 7, 2, 4, 6, 8]
  buildTree()
  msg.value = '已重置'
}

// 执行操作
function run() {
  if (playing.value) return

  resetAnim()

  if (opType.value === 'update') {
    if (idx.value < 1 || idx.value > n || val.value === undefined) {
      msg.value = '请输入有效的位置和值'
      return
    }
    genUpdateAnim(idx.value, val.value)
  } else if (opType.value === 'query') {
    if (idx.value < 1 || idx.value > n) {
      msg.value = '请输入有效的位置'
      return
    }
    genQueryAnim(idx.value)
  } else if (opType.value === 'range') {
    if (idx.value < 1 || rIdx.value < 1 || idx.value > rIdx.value || rIdx.value > n) {
      msg.value = '请输入有效的区间'
      return
    }
    genRangeAnim(idx.value, rIdx.value)
  }
}

onMounted(() => {
  buildTree()
})
</script>

<style scoped>
.bit-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.ctrl-row {
  display: flex;
  gap: 8px;
  margin-bottom: 15px;
  flex-wrap: wrap;
  align-items: center;
}

.sel, .inp {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
}

.inp.num { width: 60px; }

.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-blue { background: #3b82f6; color: white; }
.btn-gray { background: #6b7280; color: white; }
.btn-sm { padding: 5px 10px; font-size: 12px; }

.arr-row {
  margin-bottom: 12px;
}

.lbl {
  font-weight: 600;
  margin-right: 10px;
  color: #374151;
}

.arr-cells {
  display: flex;
  gap: 4px;
  margin-top: 5px;
  overflow-x: auto;
  padding: 5px 0;
}

.arr-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 45px;
}

.idx {
  font-size: 11px;
  color: #9ca3af;
  margin-bottom: 2px;
}

.val {
  width: 40px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  font-weight: 600;
  transition: all 0.2s;
}

.val.hl {
  background: #fef3c7;
  border-color: #f59e0b;
}

.val.cur {
  background: #3b82f6;
  color: white;
  border-color: #2563eb;
  transform: scale(1.1);
}

.lb {
  font-size: 10px;
  color: #6b7280;
  margin-top: 2px;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 12px;
  font-size: 13px;
}

.bin-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px;
  background: #fefce8;
  border-radius: 6px;
  font-size: 13px;
}

.bin-val {
  font-family: monospace;
  background: #fef3c7;
  padding: 2px 6px;
  border-radius: 3px;
}

.bin-desc {
  color: #92400e;
}

.ctl-row {
  display: flex;
  gap: 8px;
  justify-content: center;
  margin-bottom: 8px;
}

.prog { text-align: center; color: #6b7280; font-size: 12px; margin-bottom: 10px; }

.leg {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  align-items: center;
  padding: 10px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 12px;
}
.leg span { display: flex; align-items: center; gap: 5px; }

.d {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  display: inline-block;
}
.d.hl { background: #fef3c7; border: 2px solid #f59e0b; }
.d.cur { background: #3b82f6; }

.tip { color: #6b7280; font-size: 11px; }
</style>
