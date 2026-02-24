<template>
  <div class="uf-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <select v-model="opType" class="sel">
        <option value="find">查找根节点</option>
        <option value="union">合并集合</option>
      </select>
      <input v-model.number="x" type="number" min="0" :max="n-1" placeholder="元素x" class="inp num" />
      <input v-model.number="y" type="number" min="0" :max="n-1" placeholder="元素y" class="inp num" v-if="opType === 'union'" />
      <button @click="run" class="btn btn-blue" :disabled="playing">执行</button>
      <button @click="resetAll" class="btn btn-gray">重置</button>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">💡 每个节点指向父节点，根节点指向自己。同一集合的元素有相同的根。</span>
    </div>

    <!-- 父节点数组 -->
    <div class="arr-row">
      <span class="lbl">parent[]:</span>
      <div class="arr-cells">
        <div class="arr-cell" v-for="(p, i) in parent" :key="i">
          <span class="idx">{{ i }}</span>
          <span class="val" :class="{ hl: hlNodes.includes(i), cur: curNode === i, root: parent[i] === i }">{{ p }}</span>
        </div>
      </div>
    </div>

    <!-- 树形可视化 -->
    <div class="tree-area">
      <svg :width="svgW" :height="svgH" class="svg-tree">
        <!-- 连线 -->
        <g v-for="(e, i) in edges" :key="'e' + i">
          <line :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
            :class="['ln', e.hl ? 'hl' : '']" />
          <polygon v-if="e.showArrow"
            :points="arrowPoints(e.x1, e.y1, e.x2, e.y2)"
            class="arrow" />
        </g>
        <!-- 节点 -->
        <g v-for="(node, i) in nodes" :key="'n' + i">
          <circle :cx="node.x" :cy="node.y" r="20"
            :class="['nd', node.st, parent[node.idx] === node.idx ? 'root' : '']" />
          <text :x="node.x" :y="node.y + 5" class="tx">{{ node.idx }}</text>
        </g>
      </svg>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 路径压缩提示 -->
    <div class="path-row" v-if="pathNodes.length > 0">
      <span class="lbl">查找路径:</span>
      <span class="path-items">
        <span v-for="(p, i) in pathNodes" :key="i" class="path-item">
          {{ p }}
          <span v-if="i < pathNodes.length - 1"> → </span>
        </span>
      </span>
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
      <span><i class="d root"></i>根节点</span>
      <span><i class="d cur"></i>当前</span>
      <span><i class="d hl"></i>路径</span>
      <span><i class="d done"></i>已压缩</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const n = 8
const parent = ref([0, 1, 2, 3, 4, 5, 6, 7])
const rank = ref([0, 0, 0, 0, 0, 0, 0, 0])

const opType = ref('find')
const x = ref(0)
const y = ref(1)
const msg = ref('选择操作类型')
const playing = ref(false)
const step = ref(0)

const hlNodes = ref([])
const curNode = ref(-1)
const pathNodes = ref([])

const nodes = ref([])
const edges = ref([])

const svgW = ref(600)
const svgH = ref(300)

const acts = ref([])
let timer = null

// 计算箭头点
function arrowPoints(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1)
  const size = 8
  const px = x2 - 20 * Math.cos(angle)
  const py = y2 - 20 * Math.sin(angle)
  return `${px},${py} ${px - size * Math.cos(angle - 0.5)},${py - size * Math.sin(angle - 0.5)} ${px - size * Math.cos(angle + 0.5)},${py - size * Math.sin(angle + 0.5)}`
}

// 构建可视化
function buildVisualization() {
  nodes.value = []
  edges.value = []

  // 找出每个集合的根和成员
  const sets = {}
  for (let i = 0; i < n; i++) {
    const r = findRoot(i)
    if (!sets[r]) sets[r] = []
    sets[r].push(i)
  }

  // 为每个集合分配位置
  const roots = Object.keys(sets).map(Number)
  const setWidth = svgW.value / (roots.length + 1)
  
  roots.forEach((root, setIdx) => {
    const members = sets[root]
    const baseX = setWidth * (setIdx + 1)
    
    // 按层级排列
    const levels = {}
    members.forEach(m => {
      const level = getLevel(m)
      if (!levels[level]) levels[level] = []
      levels[level].push(m)
    })

    const maxLevel = Math.max(...Object.keys(levels).map(Number))
    const levelHeight = 60

    Object.entries(levels).forEach(([level, members]) => {
      const y = 40 + Number(level) * levelHeight
      const gap = 50
      const startX = baseX - (members.length - 1) * gap / 2

      members.forEach((m, i) => {
        const nodeX = startX + i * gap
        nodes.value.push({
          idx: m,
          x: nodeX,
          y: y,
          st: ''
        })

        // 添加边（指向父节点）
        if (parent.value[m] !== m) {
          const parentNode = nodes.value.find(n => n.idx === parent.value[m])
          if (parentNode) {
            edges.value.push({
              from: m,
              to: parent.value[m],
              x1: nodeX,
              y1: y,
              x2: parentNode.x,
              y2: parentNode.y,
              hl: false,
              showArrow: true
            })
          }
        }
      })
    })
  })
}

// 查找根（不带压缩）
function findRoot(x) {
  while (parent.value[x] !== x) {
    x = parent.value[x]
  }
  return x
}

// 获取层级深度
function getLevel(x) {
  let level = 0
  while (parent.value[x] !== x) {
    x = parent.value[x]
    level++
  }
  return level
}

// 生成查找动画
function genFindAnim(target) {
  acts.value = []
  pathNodes.value = []

  acts.value.push({
    t: 'start',
    m: `查找元素 ${target} 的根节点`
  })

  let current = target
  const path = []

  while (parent.value[current] !== current) {
    path.push(current)
    acts.value.push({
      t: 'visit',
      node: current,
      path: [...path],
      m: `访问节点 ${current}，父节点是 ${parent.value[current]}`
    })
    current = parent.value[current]
  }

  path.push(current)
  acts.value.push({
    t: 'found',
    node: current,
    path: path,
    m: `找到根节点 ${current}！`
  })

  // 路径压缩
  if (path.length > 2) {
    acts.value.push({
      t: 'compress',
      path: path,
      root: current,
      m: `路径压缩：将路径上的节点直接指向根节点 ${current}`
    })

    // 执行压缩
    for (let i = 0; i < path.length - 1; i++) {
      if (parent.value[path[i]] !== current) {
        parent.value[path[i]] = current
      }
    }
  }

  acts.value.push({
    t: 'done',
    m: `查找完成，根节点是 ${current}`
  })
}

// 生成合并动画
function genUnionAnim(a, b) {
  acts.value = []
  pathNodes.value = []

  acts.value.push({
    t: 'start',
    m: `合并元素 ${a} 和 ${b} 所在的集合`
  })

  // 找a的根
  let rootA = a
  const pathA = [a]
  while (parent.value[rootA] !== rootA) {
    rootA = parent.value[rootA]
    pathA.push(rootA)
  }

  acts.value.push({
    t: 'findA',
    node: rootA,
    path: pathA,
    m: `元素 ${a} 的根节点是 ${rootA}`
  })

  // 找b的根
  let rootB = b
  const pathB = [b]
  while (parent.value[rootB] !== rootB) {
    rootB = parent.value[rootB]
    pathB.push(rootB)
  }

  acts.value.push({
    t: 'findB',
    node: rootB,
    path: pathB,
    m: `元素 ${b} 的根节点是 ${rootB}`
  })

  if (rootA === rootB) {
    acts.value.push({
      t: 'done',
      m: `${a} 和 ${b} 已经在同一个集合中，无需合并`
    })
    return
  }

  // 按秩合并
  let newRoot, childRoot
  if (rank.value[rootA] < rank.value[rootB]) {
    newRoot = rootB
    childRoot = rootA
  } else if (rank.value[rootA] > rank.value[rootB]) {
    newRoot = rootA
    childRoot = rootB
  } else {
    newRoot = rootA
    childRoot = rootB
    rank.value[rootA]++
  }

  acts.value.push({
    t: 'union',
    newRoot: newRoot,
    childRoot: childRoot,
    m: `按秩合并：将根 ${childRoot} 挂到根 ${newRoot} 下`
  })

  parent.value[childRoot] = newRoot

  acts.value.push({
    t: 'done',
    m: `合并完成！新的集合根节点是 ${newRoot}`
  })
}

// 执行动作
function exec(a) {
  // 重置状态
  nodes.value.forEach(n => n.st = '')
  edges.value.forEach(e => e.hl = false)
  hlNodes.value = []
  curNode.value = -1

  switch (a.t) {
    case 'start':
      pathNodes.value = []
      break

    case 'visit':
      curNode.value = a.node
      hlNodes.value = a.path
      pathNodes.value = a.path
      const visitNode = nodes.value.find(n => n.idx === a.node)
      if (visitNode) visitNode.st = 'cur'
      // 高亮路径边
      for (let i = 0; i < a.path.length - 1; i++) {
        const edge = edges.value.find(e => e.from === a.path[i] && e.to === a.path[i + 1])
        if (edge) edge.hl = true
      }
      break

    case 'found':
      curNode.value = a.node
      hlNodes.value = a.path
      pathNodes.value = a.path
      const foundNode = nodes.value.find(n => n.idx === a.node)
      if (foundNode) foundNode.st = 'done'
      break

    case 'compress':
      pathNodes.value = a.path
      hlNodes.value = a.path
      // 标记压缩后的节点
      for (let i = 0; i < a.path.length - 1; i++) {
        const node = nodes.value.find(n => n.idx === a.path[i])
        if (node) node.st = 'done'
      }
      // 更新可视化
      buildVisualization()
      break

    case 'findA':
      pathNodes.value = a.path
      hlNodes.value = a.path
      const nodeA = nodes.value.find(n => n.idx === a.node)
      if (nodeA) nodeA.st = 'done'
      break

    case 'findB':
      pathNodes.value = a.path
      hlNodes.value = a.path
      const nodeB = nodes.value.find(n => n.idx === a.node)
      if (nodeB) nodeB.st = 'done'
      break

    case 'union':
      buildVisualization()
      const newRootNode = nodes.value.find(n => n.idx === a.newRoot)
      if (newRootNode) newRootNode.st = 'done'
      const childRootNode = nodes.value.find(n => n.idx === a.childRoot)
      if (childRootNode) childRootNode.st = 'hl'
      break

    case 'done':
      pathNodes.value = []
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

  step.value = Math.max(0, step.value - 1)
  // 简化：重新从头执行到目标步
  const targetStep = step.value
  step.value = 0
  resetAnim()
  for (let i = 0; i < targetStep; i++) {
    exec(acts.value[i])
    step.value++
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
  hlNodes.value = []
  curNode.value = -1
  pathNodes.value = []
  nodes.value.forEach(n => n.st = '')
  edges.value.forEach(e => e.hl = false)
  msg.value = '选择操作类型'
}

// 重置全部
function resetAll() {
  resetAnim()
  parent.value = [0, 1, 2, 3, 4, 5, 6, 7]
  rank.value = [0, 0, 0, 0, 0, 0, 0, 0]
  buildVisualization()
  msg.value = '已重置，每个元素独立成一个集合'
}

// 执行操作
function run() {
  if (playing.value) return

  resetAnim()
  buildVisualization()

  if (opType.value === 'find') {
    if (x.value < 0 || x.value >= n) {
      msg.value = '请输入有效元素'
      return
    }
    genFindAnim(x.value)
  } else if (opType.value === 'union') {
    if (x.value < 0 || x.value >= n || y.value < 0 || y.value >= n) {
      msg.value = '请输入有效元素'
      return
    }
    genUnionAnim(x.value, y.value)
  }
}

onMounted(() => {
  // 预设一些合并操作
  parent.value = [0, 0, 2, 2, 2, 5, 5, 5]
  rank.value = [1, 0, 1, 0, 0, 1, 0, 0]
  buildVisualization()
})
</script>

<style scoped>
.uf-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.ctrl-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
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

.tip-row { margin-bottom: 10px; }
.tip { font-size: 12px; color: #6b7280; }

.arr-row { margin-bottom: 12px; }

.lbl { font-weight: 600; color: #374151; margin-right: 8px; }

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

.idx { font-size: 10px; color: #9ca3af; }

.val {
  width: 32px;
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

.val.hl { background: #dbeafe; border-color: #93c5fd; }
.val.cur { background: #3b82f6; color: white; border-color: #2563eb; }
.val.root { border-color: #22c55e; }

.tree-area {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: auto;
  margin-bottom: 12px;
}

.svg-tree { display: block; }

.ln {
  stroke: #cbd5e1;
  stroke-width: 2;
  transition: stroke 0.2s;
}
.ln.hl { stroke: #3b82f6; stroke-width: 3; }

.arrow { fill: #cbd5e1; }
.ln.hl + .arrow { fill: #3b82f6; }

.nd {
  fill: white;
  stroke: #94a3b8;
  stroke-width: 2;
  transition: all 0.2s;
}
.nd.root { stroke: #22c55e; stroke-width: 3; }
.nd.cur { fill: #3b82f6; stroke: #2563eb; }
.nd.hl { fill: #dbeafe; stroke: #3b82f6; }
.nd.done { fill: #dcfce7; stroke: #22c55e; }

.tx {
  font-size: 14px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
  fill: #1f2937;
  pointer-events: none;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
}

.path-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  padding: 8px;
  background: #fefce8;
  border-radius: 6px;
  font-size: 13px;
}

.path-items { display: flex; align-items: center; }
.path-item { color: #92400e; font-weight: 600; }

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
  border-radius: 50%;
  display: inline-block;
}
.d.root { background: white; border: 3px solid #22c55e; }
.d.cur { background: #3b82f6; }
.d.hl { background: #dbeafe; border: 2px solid #3b82f6; }
.d.done { background: #dcfce7; border: 2px solid #22c55e; }
</style>
