<template>
  <div class="bfs-wrapper">
    <div class="bfs-container">
      <div class="grid" :style="gridStyle">
        <div
          v-for="(cell, index) in cells"
          :key="index"
          class="cell"
          :class="cell.className"
        ></div>
      </div>
      <div class="status">{{ status }}</div>
    </div>
    <div class="legend">
      <h4>图例</h4>
      <div class="legend-item">
        <div class="legend-cell start"></div>
        <span>起点</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell end"></div>
        <span>终点</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell obstacle"></div>
        <span>障碍物</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell queue"></div>
        <span>队列中</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell path"></div>
        <span>最短路径</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const ROWS = 8
const COLS = 12
const DELAY = 150

const startPos = { row: 2, col: 2 }
const endPos = { row: 5, col: 9 }

const obstacles = [
  { row: 1, col: 4 }, { row: 1, col: 5 }, { row: 1, col: 6 },
  { row: 2, col: 4 }, { row: 3, col: 4 }, { row: 4, col: 4 },
  { row: 4, col: 5 }, { row: 4, col: 6 }, { row: 4, col: 7 },
  { row: 5, col: 7 }, { row: 6, col: 7 }, { row: 7, col: 7 },
  { row: 3, col: 8 }, { row: 3, col: 9 }, { row: 2, col: 9 }
]

const cells = ref([])
const status = ref('初始化中...')
let isRunning = false

const gridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${COLS}, 30px)`,
  gridTemplateRows: `repeat(${ROWS}, 30px)`
}))

function initGrid() {
  const newCells = []
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const isObstacle = obstacles.some(o => o.row === row && o.col === col)
      let className = 'cell'
      if (row === startPos.row && col === startPos.col) {
        className += ' start'
      } else if (row === endPos.row && col === endPos.col) {
        className += ' end'
      } else if (isObstacle) {
        className += ' obstacle'
      }
      newCells.push({
        row,
        col,
        isObstacle,
        visited: false,
        parent: null,
        className
      })
    }
  }
  cells.value = newCells
}

function resetGridState() {
  cells.value = cells.value.map(cell => {
    let className = 'cell'
    if (cell.row === startPos.row && cell.col === startPos.col) {
      className += ' start'
    } else if (cell.row === endPos.row && cell.col === endPos.col) {
      className += ' end'
    } else if (cell.isObstacle) {
      className += ' obstacle'
    }
    return { ...cell, visited: false, parent: null, className }
  })
}

function getIndex(row, col) {
  return row * COLS + col
}

function setCellClass(row, col, className) {
  const index = getIndex(row, col)
  if (cells.value[index]) {
    cells.value[index].className += ' ' + className
  }
}

function removeCellClass(row, col, className) {
  const index = getIndex(row, col)
  if (cells.value[index]) {
    cells.value[index].className = cells.value[index].className.replace(className, '').trim()
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function bfs() {
  const directions = [
    { dr: -1, dc: 0 },
    { dr: 1, dc: 0 },
    { dr: 0, dc: -1 },
    { dr: 0, dc: 1 }
  ]

  const queue = [startPos]
  const startIdx = getIndex(startPos.row, startPos.col)
  cells.value[startIdx].visited = true

  status.value = '从起点开始搜索...'
  await sleep(DELAY)

  while (queue.length > 0) {
    const current = queue.shift()
    const { row, col } = current

    if (row === endPos.row && col === endPos.col) {
      await showPath()
      return
    }

    if (!(row === startPos.row && col === startPos.col)) {
      setCellClass(row, col, 'processing')
      await sleep(DELAY)
      removeCellClass(row, col, 'processing')
      setCellClass(row, col, 'visited')
    }

    for (const dir of directions) {
      const newRow = row + dir.dr
      const newCol = col + dir.dc

      if (newRow < 0 || newRow >= ROWS || newCol < 0 || newCol >= COLS) {
        continue
      }

      const idx = getIndex(newRow, newCol)
      const neighbor = cells.value[idx]

      if (neighbor.isObstacle || neighbor.visited) {
        continue
      }

      neighbor.visited = true
      neighbor.parent = { row, col }

      if (!(newRow === endPos.row && newCol === endPos.col)) {
        setCellClass(newRow, newCol, 'queue')
      }

      queue.push({ row: newRow, col: newCol })
    }
  }

  status.value = '未找到路径'
}

async function showPath() {
  const path = []
  let current = endPos

  while (current) {
    path.unshift(current)
    const idx = getIndex(current.row, current.col)
    current = cells.value[idx].parent
  }

  status.value = `找到最短路径！长度: ${path.length - 1} 步`
  await sleep(DELAY * 2)

  for (const pos of path) {
    removeCellClass(pos.row, pos.col, 'queue')
    removeCellClass(pos.row, pos.col, 'visited')
    setCellClass(pos.row, pos.col, 'path')
    await sleep(DELAY)
  }
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    resetGridState()
    await bfs()
    await sleep(2000)
  }
}

onMounted(() => {
  initGrid()
  animate()
})

onUnmounted(() => {
  isRunning = false
})
</script>

<style scoped>
.bfs-wrapper {
  display: flex;
  gap: 30px;
  align-items: flex-start;
  justify-content: center;
  padding: 20px;
}

.bfs-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.grid {
  display: grid;
  gap: 2px;
  background: #e5e7eb;
  padding: 3px;
  border-radius: 6px;
}

.cell {
  width: 30px;
  height: 30px;
  background: white;
  border-radius: 3px;
  transition: all 0.15s ease;
}

.cell.start {
  background: #10b981;
}

.cell.end {
  background: #ef4444;
}

.cell.obstacle {
  background: #374151;
}

.cell.visited {
  background: #60a5fa;
}

.cell.queue {
  background: #fbbf24;
}

.cell.processing {
  background: #f97316;
  transform: scale(1.1);
}

.cell.path {
  background: #10b981;
  animation: pulse 0.4s ease;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.15); }
}

.status {
  font-size: 16px;
  color: #374151;
  font-weight: 500;
}

.legend {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 15px;
  background: #f9fafb;
  border-radius: 8px;
  min-width: 140px;
}

.legend h4 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #374151;
  font-weight: 600;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: #4b5563;
}

.legend-cell {
  width: 20px;
  height: 20px;
  border-radius: 3px;
}

.legend-cell.start {
  background: #10b981;
}

.legend-cell.end {
  background: #ef4444;
}

.legend-cell.obstacle {
  background: #374151;
}

.legend-cell.queue {
  background: #fbbf24;
}

.legend-cell.visited {
  background: #60a5fa;
}

.legend-cell.path {
  background: #10b981;
}
</style>
