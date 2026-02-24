<template>
  <div class="dijkstra-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <button @click="startAnim" class="btn btn-blue" :disabled="playing">开始演示</button>
      <button @click="resetAll" class="btn btn-gray" :disabled="playing">重置</button>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">Dijkstra算法：从起点A出发，逐步确定到各点的最短距离，目标点H</span>
    </div>

    <!-- 图可视化 -->
    <div class="graph-area">
      <svg :width="svgW" :height="svgH" class="svg-graph">
        <!-- 边 -->
        <g v-for="(edge, i) in edges" :key="'e' + i">
          <line :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
            :class="['edge', edge.st]" />
          <circle :cx="edge.mx" :cy="edge.my" r="12" class="weight-bg" />
          <text :x="edge.mx" :y="edge.my + 4" class="edge-w">{{ edge.w }}</text>
        </g>
        <!-- 节点 -->
        <g v-for="(node, i) in graphNodes" :key="'n' + i">
          <circle :cx="node.x" :cy="node.y" r="22" :class="['node-outer', node.st]" />
          <circle :cx="node.x" :cy="node.y" r="16" :class="['node-inner', node.st]" />
          <text :x="node.x" :y="node.y + 5" class="node-label">{{ node.id }}</text>
          <!-- 距离标签 -->
          <g v-if="node.dist < INF">
            <rect :x="node.x - 14" :y="node.y - 38" width="28" height="18" rx="4" class="dist-bg" />
            <text :x="node.x" :y="node.y - 25" class="dist-label">{{ node.dist }}</text>
          </g>
        </g>
      </svg>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 当前队列 -->
    <div class="queue-row" v-if="queue.length > 0">
      <span class="lbl">优先队列:</span>
      <span class="queue-items">
        <span v-for="(item, i) in queue" :key="i" class="queue-item">
          {{ item.id }}({{ item.dist }})
        </span>
      </span>
    </div>

    <!-- 播放控制 -->
    <div class="ctl-row" v-if="acts.length">
      <button @click="prev" class="btn btn-sm" :disabled="step <= 0 || playing">上一步</button>
      <button @click="toggle" class="btn btn-sm btn-blue">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="nextStep" class="btn btn-sm" :disabled="step >= acts.length || playing">下一步</button>
    </div>

    <div class="prog" v-if="acts.length">{{ step }} / {{ acts.length }}</div>

    <!-- 图例 -->
    <div class="leg">
      <span><i class="d start"></i>起点</span>
      <span><i class="d cur"></i>当前</span>
      <span><i class="d queue"></i>候选</span>
      <span><i class="d done"></i>已确定</span>
      <span><i class="d target"></i>目标</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const INF = 99999
const START = 'A'
const TARGET = 'H'

const msg = ref('点击"开始演示"查看Dijkstra算法过程')
const playing = ref(false)
const step = ref(0)
const queue = ref([])

const acts = ref([])
let timer = null

const svgW = ref(450)
const svgH = ref(320)

// 图节点
const graphNodes = ref([
  { id: 'A', x: 60, y: 60, dist: INF, st: '' },
  { id: 'B', x: 170, y: 40, dist: INF, st: '' },
  { id: 'C', x: 300, y: 50, dist: INF, st: '' },
  { id: 'D', x: 55, y: 170, dist: INF, st: '' },
  { id: 'E', x: 180, y: 160, dist: INF, st: '' },
  { id: 'F', x: 320, y: 155, dist: INF, st: '' },
  { id: 'G', x: 90, y: 270, dist: INF, st: '' },
  { id: 'H', x: 270, y: 270, dist: INF, st: '' }
])

// 边（邻接表）
const adjacency = {
  'A': [['B', 4], ['D', 2]],
  'B': [['A', 4], ['C', 3], ['E', 5]],
  'C': [['B', 3], ['F', 4]],
  'D': [['A', 2], ['E', 1], ['G', 3]],
  'E': [['B', 5], ['D', 1], ['F', 2], ['H', 4]],
  'F': [['C', 4], ['E', 2], ['H', 5]],
  'G': [['D', 3], ['H', 2]],
  'H': [['E', 4], ['F', 5], ['G', 2]]
}

// 可视化边
const edges = ref([])

// 构建可视化边
function buildEdges() {
  edges.value = []
  const added = new Set()

  graphNodes.value.forEach(node => {
    const neighbors = adjacency[node.id] || []
    neighbors.forEach(([nid, w]) => {
      const key = [node.id, nid].sort().join('-')
      if (!added.has(key)) {
        added.add(key)
        const neighbor = graphNodes.value.find(n => n.id === nid)
        if (neighbor) {
          edges.value.push({
            u: node.id,
            v: nid,
            w: w,
            x1: node.x,
            y1: node.y,
            x2: neighbor.x,
            y2: neighbor.y,
            mx: (node.x + neighbor.x) / 2,
            my: (node.y + neighbor.y) / 2,
            st: ''
          })
        }
      }
    })
  })
}

// 获取节点
function getNode(id) {
  return graphNodes.value.find(n => n.id === id)
}

// 获取边
function getEdge(u, v) {
  return edges.value.find(e => 
    (e.u === u && e.v === v) || (e.u === v && e.v === u)
  )
}

// 重置状态
function resetNodes() {
  graphNodes.value.forEach(n => {
    n.dist = INF
    n.st = ''
  })
  edges.value.forEach(e => e.st = '')
  queue.value = []
}

// 生成动画
function genAnim() {
  acts.value = []
  resetNodes()

  // 初始化起点
  acts.value.push({
    t: 'init',
    m: `初始化：起点${START}的距离为0，其他点为∞`
  })

  const visited = new Set()
  const distMap = {}
  graphNodes.value.forEach(n => distMap[n.id] = INF)
  distMap[START] = 0

  // 优先队列 [{id, dist}]
  const pq = [{ id: START, dist: 0 }]

  acts.value.push({
    t: 'enqueue',
    pq: [...pq],
    m: `将起点${START}加入优先队列`
  })

  while (pq.length > 0) {
    // 取出距离最小的节点
    pq.sort((a, b) => a.dist - b.dist)
    const current = pq.shift()
    const u = current.id

    if (visited.has(u)) continue
    visited.add(u)

    acts.value.push({
      t: 'visit',
      node: u,
      pq: [...pq],
      m: `取出距离最小的节点 ${u}（距离=${distMap[u]}），标记为已确定`
    })

    // 到达目标
    if (u === TARGET) {
      acts.value.push({
        t: 'found',
        m: `到达目标节点 ${TARGET}！最短距离 = ${distMap[TARGET]}`
      })
      break
    }

    // 松弛邻居
    const neighbors = adjacency[u] || []
    neighbors.forEach(([v, w]) => {
      if (!visited.has(v)) {
        const oldDist = distMap[v]
        const newDist = distMap[u] + w

        acts.value.push({
          t: 'relax',
          u: u,
          v: v,
          w: w,
          oldDist: oldDist,
          newDist: newDist,
          m: `检查边 ${u}→${v}：dist[${v}]=${oldDist >= INF ? '∞' : oldDist} vs dist[${u}]+${w}=${distMap[u]}+${w}=${newDist}`
        })

        if (newDist < oldDist) {
          distMap[v] = newDist
          pq.push({ id: v, dist: newDist })

          acts.value.push({
            t: 'update',
            node: v,
            dist: newDist,
            pq: [...pq],
            m: `更新！dist[${v}] = ${newDist}，加入优先队列`
          })
        }
      }
    })
  }

  acts.value.push({
    t: 'done',
    m: `Dijkstra算法完成！${START}到${TARGET}的最短距离 = ${distMap[TARGET]}`
  })
}

// 执行动作
function exec(a) {
  switch (a.t) {
    case 'init':
      resetNodes()
      getNode(START).st = 'start'
      break

    case 'enqueue':
      queue.value = a.pq.map(p => ({ id: p.id, dist: p.dist }))
      break

    case 'visit':
      graphNodes.value.forEach(n => {
        if (n.st === 'cur') n.st = 'done'
      })
      getNode(a.node).st = 'cur'
      queue.value = a.pq.map(p => ({ id: p.id, dist: p.dist }))
      break

    case 'relax':
      const edge = getEdge(a.u, a.v)
      if (edge) edge.st = 'checking'
      break

    case 'update':
      const e = getEdge(a.u, a.v)
      if (e) e.st = 'relaxed'
      getNode(a.node).dist = a.dist
      getNode(a.node).st = 'queue'
      queue.value = a.pq.map(p => ({ id: p.id, dist: p.dist }))
      break

    case 'found':
      getNode(TARGET).st = 'target'
      break

    case 'done':
      graphNodes.value.forEach(n => {
        if (n.st === 'cur') n.st = 'done'
      })
      edges.value.forEach(e => e.st = '')
      getNode(START).st = 'start'
      getNode(TARGET).st = 'target'
      queue.value = []
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

  const targetStep = step.value - 1
  step.value = 0
  resetNodes()

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

// 开始动画
function startAnim() {
  if (playing.value) return
  resetAnim()
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
  resetNodes()
  msg.value = '点击"开始演示"查看Dijkstra算法过程'
}

// 重置全部
function resetAll() {
  resetAnim()
}

onMounted(() => {
  buildEdges()
})
</script>

<style scoped>
.dijkstra-box {
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

.tip-row { margin-bottom: 12px; }
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
  transition: all 0.3s;
}
.edge.checking { stroke: #f59e0b; stroke-width: 3; }
.edge.relaxed { stroke: #8b5cf6; stroke-width: 3; }

.weight-bg {
  fill: #f1f5f9;
  stroke: #cbd5e1;
}
.edge-w {
  font-size: 11px;
  font-weight: 600;
  fill: #475569;
  text-anchor: middle;
}

.node-outer {
  fill: white;
  stroke: #64748b;
  stroke-width: 2.5;
  transition: all 0.3s;
}
.node-outer.start { stroke: #10b981; stroke-width: 3.5; }
.node-outer.cur { stroke: #f97316; stroke-width: 3.5; }
.node-outer.queue { stroke: #fbbf24; }
.node-outer.done { stroke: #6366f1; }
.node-outer.target { stroke: #ef4444; stroke-width: 3.5; }

.node-inner {
  fill: white;
  transition: fill 0.3s;
}
.node-inner.start { fill: #10b981; }
.node-inner.cur { fill: #f97316; }
.node-inner.queue { fill: #fef3c7; }
.node-inner.done { fill: #c7d2fe; }
.node-inner.target { fill: #fecaca; }

.node-label {
  font-size: 13px;
  font-weight: 700;
  fill: #1f2937;
  text-anchor: middle;
  dominant-baseline: middle;
}

.dist-bg { fill: #ef4444; }
.dist-label {
  font-size: 11px;
  font-weight: 600;
  fill: white;
  text-anchor: middle;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
}

.queue-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  padding: 8px 12px;
  background: #fef3c7;
  border-radius: 6px;
}

.lbl { font-weight: 600; color: #374151; }
.queue-items { display: flex; gap: 8px; flex-wrap: wrap; }
.queue-item {
  background: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
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
  border-radius: 50%;
  display: inline-block;
}
.d.start { background: #10b981; }
.d.cur { background: #f97316; }
.d.queue { background: #fef3c7; border: 2px solid #fbbf24; }
.d.done { background: #c7d2fe; border: 2px solid #6366f1; }
.d.target { background: #fecaca; border: 2px solid #ef4444; }
</style>
