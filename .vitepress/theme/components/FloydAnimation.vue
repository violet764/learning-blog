<template>
  <div class="floyd-wrapper">
    <div class="matrix-card">
      <div class="matrix-title">
        <span class="title-text">Floyd-Warshall 距离矩阵</span>
        <span class="subtitle">计算所有节点对最短路径</span>
      </div>

      <table class="matrix-table">
        <thead>
          <tr>
            <th class="corner-cell"></th>
            <th v-for="node in nodes" :key="node" class="header-cell">{{ node }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in matrix" :key="i">
            <th class="row-header">{{ nodes[i] }}</th>
            <td
              v-for="(val, j) in row"
              :key="j"
              class="matrix-cell"
              :class="getCellClass(i, j)"
            >
              {{ val === INF ? '∞' : val }}
            </td>
          </tr>
        </tbody>
      </table>

      <div class="k-indicator" v-if="currentK !== null">
        <div class="k-label">当前中转节点</div>
        <div class="k-value">{{ currentK }}</div>
      </div>

      <div class="compare-info" v-if="currentI !== null && currentJ !== null">
        <span class="compare-text">
          d[{{ nodes[currentI] }}][{{ nodes[currentJ] }}] > d[{{ nodes[currentI] }}][{{ currentK }}] + d[{{ currentK }}][{{ nodes[currentJ] }}]
        </span>
      </div>

      <div class="complete-msg" v-if="isComplete">
        <span>✓ 所有节点对最短路径计算完成</span>
      </div>
    </div>

    <div class="legend-card">
      <div class="legend-title">图例</div>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-box normal"></div>
          <span>原始距离</span>
        </div>
        <div class="legend-item">
          <div class="legend-box checking"></div>
          <span>检查中</span>
        </div>
        <div class="legend-item">
          <div class="legend-box updated"></div>
          <span>已更新</span>
        </div>
        <div class="legend-item">
          <div class="legend-box k-node"></div>
          <span>中转节点</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const nodes = ['A', 'B', 'C', 'D', 'E']
const INF = 999

const adjMatrix = [
  [0, 4, INF, 2, INF],
  [4, 0, 3, INF, 5],
  [3, INF, 0, 4, INF],
  [2, INF, 4, 0, 1],
  [INF, 5, INF, 1, 0]
]

const matrix = ref([])
const currentK = ref(null)
const currentI = ref(null)
const currentJ = ref(null)
const isComplete = ref(false)

let isRunning = false

function initMatrix() {
  matrix.value = adjMatrix.map(row => [...row])
  currentK.value = null
  currentI.value = null
  currentJ.value = null
  isComplete.value = false
}

function getCellClass(i, j) {
  if (currentK.value !== null) {
    if (i === currentK.value || j === currentK.value) {
      return 'k-node'
    }
  }
  if (currentI.value !== null && currentJ.value !== null) {
    if (i === currentI.value && j === currentJ.value) {
      return 'checking'
    }
  }
  if (matrix.value[i][j] !== adjMatrix[i][j] && matrix.value[i][j] !== INF) {
    return 'updated'
  }
  return ''
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function floyd() {
  initMatrix()
  await sleep(600)

  const n = nodes.length

  for (let k = 0; k < n; k++) {
    currentK.value = nodes[k]
    await sleep(700)

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j || i === k || j === k) continue

        currentI.value = i
        currentJ.value = j
        await sleep(180)

        const direct = matrix.value[i][j]
        const viaK = matrix.value[i][k] + matrix.value[k][j]

        if (viaK < direct) {
          matrix.value[i][j] = viaK
          await sleep(280)
        }
      }
    }

    currentI.value = null
    currentJ.value = null
  }

  currentK.value = null
  isComplete.value = true
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    await floyd()
    await sleep(4500)
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
.floyd-wrapper {
  display: flex;
  gap: 25px;
  align-items: flex-start;
  justify-content: center;
  padding: 25px;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border-radius: 16px;
}

.matrix-card {
  background: white;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.matrix-title {
  text-align: center;
  margin-bottom: 18px;
}

.title-text {
  display: block;
  font-size: 18px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 4px;
}

.subtitle {
  font-size: 12px;
  color: #64748b;
}

.matrix-table {
  border-collapse: separate;
  border-spacing: 3px;
}

.corner-cell {
  width: 36px;
  height: 36px;
}

.header-cell, .row-header {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  font-weight: 700;
  font-size: 16px;
  border-radius: 10px;
  text-align: center;
}

.header-cell {
  border-radius: 10px 10px 0 0;
}

.row-header {
  border-radius: 10px 0 0 10px;
}

.matrix-cell {
  width: 44px;
  height: 44px;
  text-align: center;
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 14px;
  font-weight: 600;
  border-radius: 8px;
  transition: all 0.25s ease;
}

.matrix-cell.normal {
  background: #f1f5f9;
  color: #64748b;
}

.matrix-cell.checking {
  background: linear-gradient(135deg, #f97316, #fb923c);
  color: white;
  transform: scale(1.1);
  box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4);
}

.matrix-cell.updated {
  background: linear-gradient(135deg, #22d3ee, #06b6d4);
  color: white;
  box-shadow: 0 2px 10px rgba(6, 182, 212, 0.3);
}

.matrix-cell.k-node {
  background: linear-gradient(135deg, #fbbf24, #f59e0b);
  color: white;
}

.k-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 18px;
  padding: 12px 20px;
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  border-radius: 10px;
}

.k-label {
  font-size: 13px;
  color: #92400e;
  font-weight: 600;
}

.k-value {
  font-size: 20px;
  font-weight: 800;
  color: #b45309;
  background: white;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(180, 83, 9, 0.2);
}

.compare-info {
  margin-top: 14px;
  padding: 10px 16px;
  background: #f8fafc;
  border-radius: 8px;
  text-align: center;
}

.compare-text {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 12px;
  color: #475569;
}

.complete-msg {
  margin-top: 14px;
  padding: 12px 20px;
  background: linear-gradient(135deg, #10b981, #34d399);
  border-radius: 10px;
  text-align: center;
  color: white;
  font-weight: 600;
  font-size: 14px;
}

.legend-card {
  background: white;
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.legend-title {
  font-size: 14px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 14px;
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

.legend-box {
  width: 22px;
  height: 22px;
  border-radius: 6px;
}

.legend-box.normal {
  background: #f1f5f9;
}

.legend-box.checking {
  background: linear-gradient(135deg, #f97316, #fb923c);
}

.legend-box.updated {
  background: linear-gradient(135deg, #22d3ee, #06b6d4);
}

.legend-box.k-node {
  background: linear-gradient(135deg, #fbbf24, #f59e0b);
}
</style>
