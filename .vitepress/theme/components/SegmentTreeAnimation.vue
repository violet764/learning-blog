<template>
  <div class="seg-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <select v-model="opType" class="sel">
        <option value="build">构建</option>
        <option value="update">单点更新</option>
        <option value="query">区间查询</option>
      </select>
      <input v-model.number="idx" type="number" min="0" :max="n-1" placeholder="位置" class="inp num" v-if="opType === 'update'" />
      <input v-model.number="val" type="number" placeholder="新值" class="inp num" v-if="opType === 'update'" />
      <input v-model.number="lIdx" type="number" min="0" :max="n-1" placeholder="左" class="inp num" v-if="opType === 'query'" />
      <input v-model.number="rIdx" type="number" min="0" :max="n-1" placeholder="右" class="inp num" v-if="opType === 'query'" />
      <button @click="run" class="btn btn-blue" :disabled="playing">执行</button>
      <button @click="resetAll" class="btn btn-gray">重置</button>
    </div>

    <!-- 原始数组 -->
    <div class="arr-row">
      <span class="lbl">原数组:</span>
      <div class="arr-cells">
        <div class="arr-cell" v-for="(v, i) in arr" :key="i">
          <span class="idx">[{{ i }}]</span>
          <span class="val" :class="{ hl: hlArr.includes(i) }">{{ v }}</span>
        </div>
      </div>
    </div>

    <!-- 线段树可视化 -->
    <div class="tree-area">
      <svg :width="svgW" :height="svgH" class="svg-tree">
        <g v-for="(node, i) in nodes" :key="'n' + i">
          <!-- 连线 -->
          <line v-if="node.parent >= 0"
            :x1="nodes[node.parent].x"
            :y1="nodes[node.parent].y"
            :x2="node.x"
            :y2="node.y"
            :class="['ln', node.hl ? 'hl' : '']" />
          <!-- 节点 -->
          <g :transform="`translate(${node.x}, ${node.y})`">
            <rect :x="-28" :y="-18" width="56" height="36" rx="6"
              :class="['nd', node.st]" />
            <text y="-4" class="tx-range">{{ node.l }}-{{ node.r }}</text>
            <text y="10" class="tx-val">{{ node.sum }}</text>
          </g>
        </g>
      </svg>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 查询结果 -->
    <div class="result" v-if="queryResult !== null">
      查询结果: <strong>{{ queryResult }}</strong>
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
      <span><i class="d cur"></i>当前</span>
      <span><i class="d hl"></i>路径</span>
      <span><i class="d done"></i>完成</span>
      <span><i class="d res"></i>结果</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const n = 8
const arr = ref([1, 3, 5, 7, 2, 4, 6, 8])
const tree = ref([])

const opType = ref('build')
const idx = ref(0)
const val = ref(0)
const lIdx = ref(0)
const rIdx = ref(3)
const msg = ref('选择操作类型')
const playing = ref(false)
const step = ref(0)
const queryResult = ref(null)

const hlArr = ref([])
const nodes = ref([])

const svgW = ref(700)
const svgH = ref(320)

const acts = ref([])
let timer = null

// 构建节点布局
function buildLayout() {
  nodes.value = []
  if (tree.value.length === 0) return

  const levelHeight = 65
  const startY = 35

  function calcPos(nodeIdx, l, r, x, width, depth) {
    if (nodeIdx >= tree.value.length || tree.value[nodeIdx] === undefined) return

    const y = startY + depth * levelHeight
    nodes.value.push({
      idx: nodeIdx,
      l: l,
      r: r,
      sum: tree.value[nodeIdx],
      x: x,
      y: y,
      parent: -1,
      st: '',
      hl: false
    })

    if (l === r) return

    const mid = Math.floor((l + r) / 2)
    const leftIdx = 2 * nodeIdx + 1
    const rightIdx = 2 * nodeIdx + 2
    const halfW = width / 2

    if (leftIdx < tree.value.length) {
      nodes.value[nodes.value.length - 1].leftChild = nodes.value.length
      calcPos(leftIdx, l, mid, x - halfW / 2, halfW, depth + 1)
      nodes.value[nodes.value.length - 1].parent = nodes.value.findIndex(n => n.leftChild === nodes.value.length - 1 || n.rightChild === nodes.value.length - 1)
    }
    if (rightIdx < tree.value.length) {
      nodes.value[nodes.value.length - 1].rightChild = nodes.value.length
      calcPos(rightIdx, mid + 1, r, x + halfW / 2, halfW, depth + 1)
    }
  }

  // 简化布局：按层级计算
  const levels = []
  function collectLevel(nodeIdx, l, r, depth) {
    if (nodeIdx >= tree.value.length || tree.value[nodeIdx] === undefined) return
    if (!levels[depth]) levels[depth] = []
    levels[depth].push({ idx: nodeIdx, l, r })
    if (l < r) {
      const mid = Math.floor((l + r) / 2)
      collectLevel(2 * nodeIdx + 1, l, mid, depth + 1)
      collectLevel(2 * nodeIdx + 2, mid + 1, r, depth + 1)
    }
  }
  collectLevel(0, 0, n - 1, 0)

  // 分配位置
  levels.forEach((level, depth) => {
    const y = startY + depth * levelHeight
    const gap = svgW.value / (level.length + 1)
    level.forEach((item, i) => {
      const x = gap * (i + 1)
      const parentIdx = depth === 0 ? -1 : nodes.value.findIndex(p => {
        const expectedLeft = 2 * p.idx + 1
        const expectedRight = 2 * p.idx + 2
        return expectedLeft === item.idx || expectedRight === item.idx
      })
      nodes.value.push({
        idx: item.idx,
        l: item.l,
        r: item.r,
        sum: tree.value[item.idx],
        x: x,
        y: y,
        parent: parentIdx,
        st: '',
        hl: false
      })
    })
  })
}

// 构建线段树
function buildSegTree() {
  tree.value = new Array(4 * n).fill(0)
  function build(node, l, r) {
    if (l === r) {
      tree.value[node] = arr.value[l]
      return
    }
    const mid = Math.floor((l + r) / 2)
    build(2 * node + 1, l, mid)
    build(2 * node + 2, mid + 1, r)
    tree.value[node] = tree.value[2 * node + 1] + tree.value[2 * node + 2]
  }
  build(0, 0, n - 1)
  buildLayout()
}

// 生成构建动画
function genBuildAnim() {
  acts.value = []

  function buildAnim(node, l, r) {
    acts.value.push({
      t: 'visit',
      node: node,
      l: l,
      r: r,
      m: `访问节点[${l}, ${r}]`
    })

    if (l === r) {
      acts.value.push({
        t: 'leaf',
        node: node,
        l: l,
        r: r,
        val: arr.value[l],
        m: `叶子节点 [${l}, ${r}] = arr[${l}] = ${arr.value[l]}`
      })
      tree.value[node] = arr.value[l]
      return
    }

    const mid = Math.floor((l + r) / 2)
    buildAnim(2 * node + 1, l, mid)
    buildAnim(2 * node + 2, mid + 1, r)

    tree.value[node] = tree.value[2 * node + 1] + tree.value[2 * node + 2]
    acts.value.push({
      t: 'merge',
      node: node,
      l: l,
      r: r,
      leftVal: tree.value[2 * node + 1],
      rightVal: tree.value[2 * node + 2],
      m: `合并 [${l}, ${mid}] + [${mid+1}, ${r}] = ${tree.value[2 * node + 1]} + ${tree.value[2 * node + 2]} = ${tree.value[node]}`
    })
  }

  acts.value.push({ t: 'start', m: '开始构建线段树...' })
  buildAnim(0, 0, n - 1)
  acts.value.push({ t: 'done', m: '线段树构建完成!' })
}

// 生成更新动画
function genUpdateAnim(pos, newVal) {
  acts.value = []
  queryResult.value = null

  const oldVal = arr.value[pos]
  const delta = newVal - oldVal

  acts.value.push({
    t: 'start',
    m: `更新位置 ${pos}: ${oldVal} → ${newVal}, delta = ${delta}`
  })

  hlArr.value = [pos]
  arr.value[pos] = newVal

  function updateAnim(node, l, r) {
    acts.value.push({
      t: 'visit',
      node: node,
      l: l,
      r: r,
      m: `访问节点[${l}, ${r}]`
    })

    tree.value[node] += delta

    if (l === r) {
      acts.value.push({
        t: 'update',
        node: node,
        l: l,
        r: r,
        m: `更新叶子节点 [${l}, ${r}] = ${tree.value[node]}`
      })
      return
    }

    const mid = Math.floor((l + r) / 2)
    if (pos <= mid) {
      updateAnim(2 * node + 1, l, mid)
    } else {
      updateAnim(2 * node + 2, mid + 1, r)
    }
  }

  updateAnim(0, 0, n - 1)
  acts.value.push({ t: 'done', m: `更新完成!` })
}

// 生成查询动画
function genQueryAnim(ql, qr) {
  acts.value = []
  queryResult.value = null

  acts.value.push({
    t: 'start',
    m: `查询区间 [${ql}, ${qr}] 的和`
  })

  let total = 0

  function queryAnim(node, l, r) {
    acts.value.push({
      t: 'visit',
      node: node,
      l: l,
      r: r,
      m: `检查节点[${l}, ${r}]`
    })

    // 完全包含
    if (ql <= l && r <= qr) {
      total += tree.value[node]
      acts.value.push({
        t: 'cover',
        node: node,
        l: l,
        r: r,
        sum: tree.value[node],
        total: total,
        m: `[${l}, ${r}] 完全在查询区间内，累加 ${tree.value[node]}，当前总和 = ${total}`
      })
      return
    }

    // 无交集
    if (r < ql || l > qr) {
      acts.value.push({
        t: 'skip',
        node: node,
        l: l,
        r: r,
        m: `[${l}, ${r}] 与查询区间无交集，跳过`
      })
      return
    }

    // 部分重叠，递归
    const mid = Math.floor((l + r) / 2)
    acts.value.push({
      t: 'recurse',
      node: node,
      l: l,
      r: r,
      m: `[${l}, ${r}] 部分重叠，递归查询`
    })

    queryAnim(2 * node + 1, l, mid)
    queryAnim(2 * node + 2, mid + 1, r)
  }

  queryAnim(0, 0, n - 1)

  acts.value.push({
    t: 'result',
    total: total,
    m: `查询完成! 区间 [${ql}, ${qr}] 的和 = ${total}`
  })

  queryResult.value = total
}

// 执行动作
function exec(a) {
  // 重置节点状态
  nodes.value.forEach(n => {
    n.st = ''
    n.hl = false
  })
  hlArr.value = []

  switch (a.t) {
    case 'start':
    case 'recurse':
      break

    case 'visit':
      const visitNode = nodes.value.find(n => n.l === a.l && n.r === a.r)
      if (visitNode) {
        visitNode.st = 'cur'
        // 高亮路径
        let p = visitNode.parent
        while (p >= 0) {
          nodes.value[p].hl = true
          p = nodes.value[p].parent
        }
      }
      break

    case 'leaf':
    case 'update':
      const leafNode = nodes.value.find(n => n.l === a.l && n.r === a.r)
      if (leafNode) {
        leafNode.st = 'done'
        leafNode.sum = a.val
      }
      hlArr.value = [a.l]
      break

    case 'merge':
      const mergeNode = nodes.value.find(n => n.l === a.l && n.r === a.r)
      if (mergeNode) {
        mergeNode.st = 'done'
        mergeNode.sum = a.leftVal + a.rightVal
      }
      break

    case 'cover':
      const coverNode = nodes.value.find(n => n.l === a.l && n.r === a.r)
      if (coverNode) {
        coverNode.st = 'res'
      }
      break

    case 'skip':
      const skipNode = nodes.value.find(n => n.l === a.l && n.r === a.r)
      if (skipNode) {
        skipNode.st = 'skip'
      }
      break

    case 'result':
      queryResult.value = a.total
      break

    case 'done':
      nodes.value.forEach(n => n.st = '')
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

  step.value = Math.max(0, step.value - 2)
  const targetStep = step.value

  // 重新执行
  step.value = 0
  nodes.value.forEach(n => {
    n.st = ''
    n.hl = false
  })
  hlArr.value = []

  for (let i = 0; i < targetStep; i++) {
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
    timer = setTimeout(autoPlay, 500)
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
  nodes.value.forEach(n => {
    n.st = ''
    n.hl = false
  })
  hlArr.value = []
  queryResult.value = null
  msg.value = '选择操作类型'
}

// 重置全部
function resetAll() {
  resetAnim()
  arr.value = [1, 3, 5, 7, 2, 4, 6, 8]
  buildSegTree()
  msg.value = '已重置'
}

// 执行操作
function run() {
  if (playing.value) return

  resetAnim()
  buildSegTree()

  if (opType.value === 'build') {
    genBuildAnim()
  } else if (opType.value === 'update') {
    if (idx.value < 0 || idx.value >= n) {
      msg.value = '请输入有效位置'
      return
    }
    genUpdateAnim(idx.value, val.value)
  } else if (opType.value === 'query') {
    if (lIdx.value < 0 || rIdx.value >= n || lIdx.value > rIdx.value) {
      msg.value = '请输入有效区间'
      return
    }
    genQueryAnim(lIdx.value, rIdx.value)
  }
}

onMounted(() => {
  buildSegTree()
})
</script>

<style scoped>
.seg-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.ctrl-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  align-items: center;
}

.sel, .inp {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
}

.inp.num { width: 55px; }

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
  color: #374151;
}

.arr-cells {
  display: flex;
  gap: 4px;
  margin-top: 5px;
}

.arr-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.idx {
  font-size: 10px;
  color: #9ca3af;
}

.val {
  width: 36px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border: 2px solid #e5e7eb;
  border-radius: 5px;
  font-weight: 600;
  font-size: 13px;
}

.val.hl {
  background: #fef3c7;
  border-color: #f59e0b;
}

.tree-area {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: auto;
  margin-bottom: 12px;
}

.svg-tree {
  display: block;
}

.ln {
  stroke: #cbd5e1;
  stroke-width: 2;
  transition: stroke 0.2s;
}
.ln.hl { stroke: #3b82f6; stroke-width: 3; }

.nd {
  fill: white;
  stroke: #94a3b8;
  stroke-width: 2;
  transition: all 0.2s;
}
.nd.cur { fill: #dbeafe; stroke: #3b82f6; }
.nd.done { fill: #dcfce7; stroke: #22c55e; }
.nd.res { fill: #fef3c7; stroke: #f59e0b; }
.nd.skip { fill: #f3f4f6; stroke: #d1d5db; }

.tx-range {
  font-size: 10px;
  text-anchor: middle;
  fill: #6b7280;
}
.tx-val {
  font-size: 12px;
  font-weight: 600;
  text-anchor: middle;
  fill: #1f2937;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
}

.result {
  padding: 10px;
  background: #dcfce7;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
  color: #166534;
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
  justify-content: center;
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
.d.cur { background: #dbeafe; border: 2px solid #3b82f6; }
.d.hl { background: #dbeafe; border: 2px solid #3b82f6; }
.d.done { background: #dcfce7; border: 2px solid #22c55e; }
.d.res { background: #fef3c7; border: 2px solid #f59e0b; }
</style>
