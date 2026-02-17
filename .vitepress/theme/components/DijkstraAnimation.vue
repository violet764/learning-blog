<template>
  <div class="dijkstra-wrapper">
    <div class="graph-area">
      <svg class="graph-svg" viewBox="0 0 500 380">
        <!-- 边 -->
        <g v-for="(edge, index) in edges" :key="'edge-' + index">
          <line
            :x1="getNode(edge.u).x"
            :y1="getNode(edge.u).y"
            :x2="getNode(edge.v).x"
            :y2="getNode(edge.v).y"
            class="edge-line"
            :class="getEdgeClass(edge)"
          />
          <!-- 边权重圆形背景 -->
          <circle
            :cx="(getNode(edge.u).x + getNode(edge.v).x) / 2"
            :cy="(getNode(edge.u).y + getNode(edge.v).y) / 2"
            r="12"
            class="weight-bg"
          />
          <text
            :x="(getNode(edge.u).x + getNode(edge.v).x) / 2"
            :y="(getNode(edge.u).y + getNode(edge.v).y) / 2 + 4"
            class="edge-weight"
          >{{ edge.w }}</text>
        </g>

        <!-- 节点 -->
        <g v-for="node in nodes" :key="node.id">
          <!-- 节点外圈 -->
          <circle
            :cx="node.x"
            :cy="node.y"
            r="22"
            class="node-outer"
            :class="getNodeClass(node.id)"
          />
          <!-- 节点内圈 -->
          <circle
            :cx="node.x"
            :cy="node.y"
            r="16"
            class="node-inner"
            :class="getNodeClass(node.id)"
          />
          <!-- 节点标签 -->
          <text :x="node.x" :y="node.y + 5" class="node-label">{{ node.id }}</text>
          <!-- 距离标签 -->
          <g v-if="node.distance !== Infinity" :transform="`translate(${node.x - 12}, ${node.y - 35})`">
            <rect x="0" y="0" width="24" height="18" rx="4" class="distance-bg" />
            <text x="12" y="13" class="distance-label">{{ node.distance }}</text>
          </g>
        </g>
      </svg>
    </div>

    <div class="legend">
      <div class="legend-title">图例</div>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-circle start"></div>
          <span>起点</span>
        </div>
        <div class="legend-item">
          <div class="legend-circle current"></div>
          <span>当前</span>
        </div>
        <div class="legend-item">
          <div class="legend-circle queue"></div>
          <span>候选</span>
        </div>
        <div class="legend-item">
          <div class="legend-circle visited"></div>
          <span>已确定</span>
        </div>
        <div class="legend-item">
          <div class="legend-line path"></div>
          <span>最短路径</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const nodes = ref([
  { id: 'A', x: 70, y: 70, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'B', x: 200, y: 45, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'C', x: 360, y: 55, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'D', x: 65, y: 200, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'E', x: 210, y: 175, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'F', x: 370, y: 195, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'G', x: 110, y: 320, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
  { id: 'H', x: 320, y: 310, distance: Infinity, visited: false, inQueue: false, current: false, pathEdge: false },
])

const edges = ref([
  { u: 'A', v: 'B', w: 4 },
  { u: 'A', v: 'D', w: 2 },
  { u: 'B', v: 'C', w: 3 },
  { u: 'B', v: 'E', w: 5 },
  { u: 'C', v: 'F', w: 4 },
  { u: 'D', v: 'E', w: 1 },
  { u: 'D', v: 'G', w: 3 },
  { u: 'E', v: 'F', w: 2 },
  { u: 'E', v: 'H', w: 4 },
  { u: 'F', v: 'H', w: 5 },
  { u: 'G', v: 'H', w: 2 },
])

let isRunning = false

function getNode(id) {
  return nodes.value.find(n => n.id === id)
}

function getNodeClass(id) {
  const node = getNode(id)
  if (node.id === 'A') return 'start'
  if (node.current) return 'current'
  if (node.visited) return 'visited'
  if (node.inQueue) return 'queue'
  return ''
}

function getEdgeClass(edge) {
  const u = getNode(edge.u)
  const v = getNode(edge.v)
  if (u.pathEdge && v.pathEdge) return 'path'
  if (u.visited && v.visited) return 'relaxed'
  return ''
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getNeighbors(nodeId) {
  const neighbors = []
  for (const edge of edges.value) {
    if (edge.u === nodeId) {
      neighbors.push({ id: edge.v, weight: edge.w })
    } else if (edge.v === nodeId) {
      neighbors.push({ id: edge.u, weight: edge.w })
    }
  }
  return neighbors
}

async function dijkstra() {
  const startNode = 'A'
  const targetNode = 'H'

  for (const node of nodes.value) {
    node.distance = Infinity
    node.visited = false
    node.inQueue = false
    node.current = false
    node.pathEdge = false
  }

  getNode(startNode).distance = 0
  getNode(startNode).inQueue = true

  await sleep(400)

  while (true) {
    let minDist = Infinity
    let current = null

    for (const node of nodes.value) {
      if (!node.visited && node.distance < minDist) {
        minDist = node.distance
        current = node.id
      }
    }

    if (current === null || minDist === Infinity) break

    const currentNode = getNode(current)
    currentNode.current = true
    currentNode.inQueue = false

    await sleep(500)

    currentNode.visited = true
    currentNode.current = false

    if (current === targetNode) {
      await showPath()
      break
    }

    const neighbors = getNeighbors(current)
    for (const neighbor of neighbors) {
      const neighborNode = getNode(neighbor.id)
      if (!neighborNode.visited) {
        const newDist = currentNode.distance + neighbor.weight
        if (newDist < neighborNode.distance) {
          neighborNode.distance = newDist
          neighborNode.inQueue = true
        }
      }
    }

    await sleep(300)
  }
}

async function showPath() {
  const target = 'H'
  let current = target

  while (current !== 'A') {
    const node = getNode(current)
    const neighbors = getNeighbors(current)

    for (const neighbor of neighbors) {
      const neighborNode = getNode(neighbor.id)
      if (node.distance === neighborNode.distance + neighbor.weight) {
        node.pathEdge = true
        neighborNode.pathEdge = true
        current = neighbor.id
        break
      }
    }

    await sleep(120)
  }
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    await dijkstra()
    await sleep(3500)

    for (const node of nodes.value) {
      node.distance = Infinity
      node.visited = false
      node.inQueue = false
      node.current = false
      node.pathEdge = false
    }
  }
}

onMounted(() => {
  animate()
})

onUnmounted(() => {
  isRunning = false
})
</script>

<style scoped>
.dijkstra-wrapper {
  display: flex;
  gap: 30px;
  align-items: flex-start;
  justify-content: center;
  padding: 25px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
}

.graph-area {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.graph-svg {
  width: 420px;
  height: 340px;
}

.edge-line {
  stroke: #94a3b8;
  stroke-width: 2.5;
  transition: all 0.4s ease;
}

.edge-line.relaxed {
  stroke: #818cf8;
  stroke-width: 3;
}

.edge-line.path {
  stroke: #10b981;
  stroke-width: 4;
  stroke-dasharray: 8 2;
}

.weight-bg {
  fill: #f1f5f9;
  stroke: #cbd5e1;
  stroke-width: 1;
}

.edge-weight {
  font-size: 11px;
  font-weight: bold;
  fill: #475569;
  text-anchor: middle;
}

.node-outer {
  fill: white;
  stroke-width: 3;
  transition: all 0.3s ease;
}

.node-outer.start {
  stroke: #10b981;
}

.node-outer.current {
  stroke: #f97316;
  animation: nodePulse 0.6s infinite;
}

.node-outer.visited {
  stroke: #6366f1;
}

.node-outer.queue {
  stroke: #fbbf24;
}

.node-inner {
  fill: white;
  transition: all 0.3s ease;
}

.node-inner.start {
  fill: #10b981;
}

.node-inner.current {
  fill: #f97316;
}

.node-inner.visited {
  fill: #6366f1;
}

.node-inner.queue {
  fill: #fbbf24;
}

.node-label {
  font-size: 14px;
  font-weight: bold;
  fill: white;
  text-anchor: middle;
  pointer-events: none;
}

.distance-bg {
  fill: #ef4444;
}

.distance-label {
  font-size: 11px;
  font-weight: bold;
  fill: white;
  text-anchor: middle;
}

@keyframes nodePulse {
  0%, 100% { r: 22; }
  50% { r: 26; }
}

.legend {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.legend-title {
  font-size: 14px;
  font-weight: bold;
  color: #334155;
  margin-bottom: 14px;
  text-align: center;
}

.legend-items {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: #475569;
}

.legend-circle {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 3px solid;
}

.legend-circle.start {
  background: #10b981;
  border-color: #059669;
}

.legend-circle.current {
  background: #f97316;
  border-color: #ea580c;
}

.legend-circle.queue {
  background: #fbbf24;
  border-color: #d97706;
}

.legend-circle.visited {
  background: #6366f1;
  border-color: #4f46e5;
}

.legend-line {
  width: 24px;
  height: 3px;
  background: #10b981;
  border-radius: 2px;
}
</style>
