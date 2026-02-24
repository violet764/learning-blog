<template>
  <div class="floyd-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <button @click="startAnim" class="btn btn-blue" :disabled="playing">开始演示</button>
      <button @click="resetAll" class="btn btn-gray" :disabled="playing">重置</button>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">Floyd算法：通过枚举中间点k，尝试松弛所有点对(i,j)的距离</span>
    </div>

    <!-- 图可视化 -->
    <div class="graph-area">
      <svg :width="svgW" :height="svgH" class="svg-graph">
        <!-- 边 -->
        <g v-for="(edge, idx) in edges" :key="'e' + idx">
          <line :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
            :class="['edge', edge.hl ? 'hl' : '']" />
          <text :x="edge.mx" :y="edge.my - 6" class="edge-w">{{ edge.w }}</text>
        </g>
        <!-- 节点 -->
        <g v-for="(node, idx) in graphNodes" :key="'n' + idx">
          <circle :cx="node.x" :cy="node.y" r="20" :class="['node', node.st]" />
          <text :x="node.x" :y="node.y + 5" class="node-label">{{ node.label }}</text>
        </g>
      </svg>
    </div>

    <!-- 当前中间点 -->
    <div class="k-row" v-if="currentK >= 0">
      <span class="lbl">当前中间点 k:</span>
      <span class="k-val">{{ currentK + 1 }}</span>
    </div>

    <!-- 距离矩阵 -->
    <div class="matrix-area">
      <div class="matrix-label">距离矩阵 dist[i][j]:</div>
      <div class="matrix-table">
        <div class="matrix-row header">
          <div class="cell empty"></div>
          <div class="cell" v-for="j in nodeCount" :key="'h' + j">{{ j }}</div>
        </div>
        <div class="matrix-row" v-for="i in nodeCount" :key="'r' + i">
          <div class="cell header">{{ i }}</div>
          <div class="cell" v-for="j in nodeCount" :key="'c' + j" :class="getCellClass(i - 1, j - 1)">
            {{ formatDist(distMatrix[(i-1) * nodeCount + (j-1)]) }}
          </div>
        </div>
      </div>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 播放控制 -->
    <div class="ctl-row" v-if="acts.length > 0">
      <button @click="prev" class="btn btn-sm" :disabled="step <= 0 || playing">上一步</button>
      <button @click="toggle" class="btn btn-sm btn-blue">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="nextStep" class="btn btn-sm" :disabled="step >= acts.length || playing">下一步</button>
    </div>

    <div class="prog" v-if="acts.length > 0">{{ step }} / {{ acts.length }}</div>

    <!-- 图例 -->
    <div class="leg">
      <span><i class="d k"></i>中间点k</span>
      <span><i class="d i"></i>起点i</span>
      <span><i class="d j"></i>终点j</span>
      <span><i class="d upd"></i>更新</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const nodeCount = 4
const INF = 99999

// 初始邻接矩阵
const initMatrix = [
  [0, 4, INF, 2],
  [INF, 0, 3, INF],
  [INF, INF, 0, 1],
  [INF, INF, INF, 0]
]

// 使用一维数组存储距离矩阵，避免响应式问题
const distMatrix = ref(new Array(nodeCount * nodeCount).fill(INF))

const msg = ref('点击"开始演示"查看Floyd算法过程')
const playing = ref(false)
const step = ref(0)
const currentK = ref(-1)
const hlI = ref(-1)
const hlJ = ref(-1)
const updI = ref(-1)
const updJ = ref(-1)

const acts = ref([])
let timer = null

const svgW = ref(450)
const svgH = ref(180)

// 图节点
const graphNodes = ref([
  { label: '1', x: 70, y: 70, st: '' },
  { label: '2', x: 180, y: 35, st: '' },
  { label: '3', x: 290, y: 70, st: '' },
  { label: '4', x: 180, y: 130, st: '' }
])

// 边
const edges = ref([])

// 初始化距离矩阵
function initDist() {
  for (let i = 0; i < nodeCount; i++) {
    for (let j = 0; j < nodeCount; j++) {
      distMatrix.value[i * nodeCount + j] = initMatrix[i][j]
    }
  }
}

// 获取距离
function getDist(i, j) {
  return distMatrix.value[i * nodeCount + j]
}

// 设置距离
function setDist(i, j, val) {
  distMatrix.value[i * nodeCount + j] = val
}

// 格式化显示
function formatDist(d) {
  if (d >= INF) return '∞'
  return String(d)
}

// 获取单元格样式
function getCellClass(i, j) {
  const classes = []
  if (i === j) classes.push('diag')
  if (i === hlI.value && j === hlJ.value) classes.push('hl-cell')
  if (i === updI.value && j === updJ.value) classes.push('upd')
  if (i === currentK.value || j === currentK.value) classes.push('k-rel')
  return classes
}

// 构建边
function buildEdges() {
  edges.value = []
  const edgeData = [
    { from: 0, to: 1, w: 4 },
    { from: 0, to: 3, w: 2 },
    { from: 1, to: 2, w: 3 },
    { from: 2, to: 3, w: 1 }
  ]

  edgeData.forEach(e => {
    const n1 = graphNodes.value[e.from]
    const n2 = graphNodes.value[e.to]
    edges.value.push({
      x1: n1.x,
      y1: n1.y,
      x2: n2.x,
      y2: n2.y,
      mx: (n1.x + n2.x) / 2,
      my: (n1.y + n2.y) / 2,
      w: e.w,
      hl: false
    })
  })
}

// 重置节点状态
function resetNodeStates() {
  graphNodes.value.forEach(n => n.st = '')
  edges.value.forEach(e => e.hl = false)
  currentK.value = -1
  hlI.value = -1
  hlJ.value = -1
  updI.value = -1
  updJ.value = -1
}

// 生成动画
function genAnim() {
  acts.value = []

  acts.value.push({
    type: 'start',
    msg: '初始化距离矩阵，dist[i][j]表示节点i到j的最短距离'
  })

  // Floyd三重循环
  for (let k = 0; k < nodeCount; k++) {
    acts.value.push({
      type: 'setK',
      k: k,
      msg: `=== 枚举中间点 k = ${k + 1} ===`
    })

    for (let i = 0; i < nodeCount; i++) {
      if (i === k) continue
      
      const dik = getDist(i, k)
      if (dik >= INF) continue

      for (let j = 0; j < nodeCount; j++) {
        if (j === k) continue

        const dkj = getDist(k, j)
        if (dkj >= INF) continue

        const oldDist = getDist(i, j)
        const newDist = dik + dkj

        acts.value.push({
          type: 'check',
          i: i,
          j: j,
          k: k,
          oldDist: oldDist,
          newDist: newDist,
          dik: dik,
          dkj: dkj,
          msg: `检查: dist[${i+1}][${j+1}]=${formatDist(oldDist)} vs dist[${i+1}][${k+1}]+dist[${k+1}][${j+1}]=${dik}+${dkj}=${newDist}`
        })

        if (newDist < oldDist) {
          setDist(i, j, newDist)
          acts.value.push({
            type: 'update',
            i: i,
            j: j,
            k: k,
            newDist: newDist,
            msg: `更新! dist[${i+1}][${j+1}] = ${newDist}`
          })
        }
      }
    }
  }

  acts.value.push({
    type: 'done',
    msg: 'Floyd算法完成！dist[i][j]即为节点i到j的最短距离'
  })
}

// 执行动作
function execAction(act) {
  resetNodeStates()

  switch (act.type) {
    case 'start':
      break

    case 'setK':
      currentK.value = act.k
      graphNodes.value[act.k].st = 'k-node'
      break

    case 'check':
      currentK.value = act.k
      hlI.value = act.i
      hlJ.value = act.j
      graphNodes.value[act.k].st = 'k-node'
      graphNodes.value[act.i].st = 'i-node'
      graphNodes.value[act.j].st = 'j-node'
      break

    case 'update':
      currentK.value = act.k
      updI.value = act.i
      updJ.value = act.j
      hlI.value = act.i
      hlJ.value = act.j
      graphNodes.value[act.k].st = 'k-node'
      break

    case 'done':
      break
  }

  msg.value = act.msg
}

// 下一步
function nextStep() {
  if (step.value >= acts.value.length) return
  execAction(acts.value[step.value])
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

  const targetStep = step.value - 1
  step.value = 0
  initDist()
  resetNodeStates()

  for (let i = 0; i < targetStep; i++) {
    execAction(acts.value[i])
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
    playing.value = true
    autoPlay()
  }
}

// 自动播放
function autoPlay() {
  if (!playing.value) return
  if (step.value < acts.value.length) {
    nextStep()
    timer = setTimeout(autoPlay, 400)
  } else {
    playing.value = false
  }
}

// 开始
function startAnim() {
  if (playing.value) return
  resetAnim()
  initDist()
  genAnim()
}

// 重置动画
function resetAnim() {
  playing.value = false
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  step.value = 0
  acts.value = []
  resetNodeStates()
  initDist()
  msg.value = '点击"开始演示"查看Floyd算法过程'
}

// 重置全部
function resetAll() {
  resetAnim()
}

onMounted(() => {
  buildEdges()
  initDist()
})
</script>

<style scoped>
.floyd-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.ctrl-row {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-blue { background: #3b82f6; color: white; }
.btn-gray { background: #6b7280; color: white; }
.btn-sm { padding: 5px 12px; font-size: 12px; }

.tip-row { margin-bottom: 10px; }
.tip { font-size: 12px; color: #6b7280; }

.graph-area {
  background: #f8fafc;
  border-radius: 8px;
  margin-bottom: 12px;
}

.svg-graph { display: block; }

.edge {
  stroke: #94a3b8;
  stroke-width: 2;
}
.edge.hl { stroke: #3b82f6; stroke-width: 3; }

.edge-w {
  font-size: 12px;
  fill: #475569;
  text-anchor: middle;
}

.node {
  fill: white;
  stroke: #64748b;
  stroke-width: 2;
  transition: all 0.2s;
}
.node.k-node { fill: #fef3c7; stroke: #f59e0b; stroke-width: 3; }
.node.i-node { fill: #dbeafe; stroke: #3b82f6; }
.node.j-node { fill: #dcfce7; stroke: #22c55e; }

.node-label {
  font-size: 14px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
}

.k-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  padding: 6px 12px;
  background: #fef3c7;
  border-radius: 6px;
}

.lbl { font-weight: 600; color: #374151; }
.k-val { font-size: 16px; font-weight: bold; color: #d97706; }

.matrix-area { margin-bottom: 10px; }
.matrix-label { font-weight: 600; color: #374151; margin-bottom: 6px; }

.matrix-table {
  display: inline-block;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  overflow: hidden;
}

.matrix-row { display: flex; }

.cell {
  width: 38px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #e2e8f0;
  font-size: 13px;
}

.cell.empty { border: none; }
.cell.header { background: #f1f5f9; font-weight: 600; }
.cell.diag { background: #f8fafc; color: #94a3b8; }
.cell.hl-cell { background: #dbeafe; }
.cell.k-rel { background: #fef3c7; }
.cell.upd { background: #dcfce7; color: #166534; }

.st-bar {
  padding: 8px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 8px;
  font-size: 13px;
}

.ctl-row { display: flex; gap: 8px; justify-content: center; margin-bottom: 6px; }
.prog { text-align: center; color: #6b7280; font-size: 12px; margin-bottom: 8px; }

.leg {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  justify-content: center;
  padding: 8px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 12px;
}
.leg span { display: flex; align-items: center; gap: 5px; }

.d {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  display: inline-block;
}
.d.k { background: #fef3c7; border: 2px solid #f59e0b; }
.d.i { background: #dbeafe; border: 2px solid #3b82f6; }
.d.j { background: #dcfce7; border: 2px solid #22c55e; }
.d.upd { background: #dcfce7; border: 2px solid #166534; }
</style>
