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
        <div class="legend-cell visited-forward"></div>
        <span>起点搜索</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell visited-backward"></div>
        <span>终点搜索</span>
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
const DELAY = 180

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
        visitedForward: false,
        visitedBackward: false,
        parentForward: null,
        parentBackward: null,
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
    return {
      ...cell,
      visitedForward: false,
      visitedBackward: false,
      parentForward: null,
      parentBackward: null,
      className
    }
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

async function bidirectionalBFS() {
  const directions = [
    { dr: -1, dc: 0 },
    { dr: 1, dc: 0 },
    { dr: 0, dc: -1 },
    { dr: 0, dc: 1 }
  ]

  // 双向队列
  const queueForward = [startPos]
  const queueBackward = [endPos]

  // 标记起终点
  const startIdx = getIndex(startPos.row, startPos.col)
  const endIdx = getIndex(endPos.row, endPos.col)
  cells.value[startIdx].visitedForward = true
  cells.value[endIdx].visitedBackward = true

  status.value = '双向搜索开始...'
  await sleep(DELAY)

  while (queueForward.length > 0 && queueBackward.length > 0) {
    // 正向搜索一层
    const levelSizeForward = queueForward.length
    let foundForward = false

    for (let i = 0; i < levelSizeForward; i++) {
      const current = queueForward.shift()
      const { row, col } = current

      // 检查是否与反向搜索相遇
      const idx = getIndex(row, col)
      if (cells.value[idx].visitedBackward) {
        await showPath(row, col, 'forward')
        return
      }

      // 动画效果
      if (!(row === startPos.row && col === startPos.col)) {
        setCellClass(row, col, 'processing')
        await sleep(DELAY / 2)
        removeCellClass(row, col, 'processing')
        setCellClass(row, col, 'visited-forward')
      }

      // 探索邻居
      for (const dir of directions) {
        const newRow = row + dir.dr
        const newCol = col + dir.dc

        if (newRow < 0 || newRow >= ROWS || newCol < 0 || newCol >= COLS) continue

        const nIdx = getIndex(newRow, newCol)
        const neighbor = cells.value[nIdx]

        if (neighbor.isObstacle || neighbor.visitedForward) continue

        neighbor.visitedForward = true
        neighbor.parentForward = { row, col }
        queueForward.push({ row: newRow, col: newCol })

        // 动画显示加入搜索
        if (!(newRow === endPos.row && newCol === endPos.col)) {
          setCellClass(newRow, newCol, 'visited-forward')
        }
      }
    }

    // 反向搜索一层
    const levelSizeBackward = queueBackward.length

    for (let i = 0; i < levelSizeBackward; i++) {
      const current = queueBackward.shift()
      const { row, col } = current

      // 检查是否与正向搜索相遇
      const idx = getIndex(row, col)
      if (cells.value[idx].visitedForward) {
        await showPath(row, col, 'backward')
        return
      }

      // 动画效果
      if (!(row === endPos.row && col === endPos.col)) {
        setCellClass(row, col, 'processing')
        await sleep(DELAY / 2)
        removeCellClass(row, col, 'processing')
        setCellClass(row, col, 'visited-backward')
      }

      // 探索邻居
      for (const dir of directions) {
        const newRow = row + dir.dr
        const newCol = col + dir.dc

        if (newRow < 0 || newRow >= ROWS || newCol < 0 || newCol >= COLS) continue

        const nIdx = getIndex(newRow, newCol)
        const neighbor = cells.value[nIdx]

        if (neighbor.isObstacle || neighbor.visitedBackward) continue

        neighbor.visitedBackward = true
        neighbor.parentBackward = { row, col }
        queueBackward.push({ row: newRow, col: newCol })

        // 动画显示加入搜索
        if (!(newRow === startPos.row && newCol === startPos.col)) {
          setCellClass(newRow, newCol, 'visited-backward')
        }
      }
    }

    await sleep(DELAY)
  }

  status.value = '未找到路径'
}

async function showPath(meetRow, meetCol, direction) {
  // 构建正向路径
  const pathForward = []
  let current = { row: meetRow, col: meetCol }
  while (current) {
    pathForward.unshift(current)
    const idx = getIndex(current.row, current.col)
    current = cells.value[idx].parentForward
  }

  // 构建反向路径
  const pathBackward = []
  current = { row: meetRow, col: meetCol }
  while (current) {
    const idx = getIndex(current.row, current.col)
    current = cells.value[idx].parentBackward
    if (current) pathBackward.push(current)
  }

  const fullPath = [...pathForward, ...pathBackward]

  status.value = `找到最短路径！长度: ${fullPath.length - 1} 步`
  await sleep(DELAY * 2)

  // 动画显示路径
  for (const pos of fullPath) {
    removeCellClass(pos.row, pos.col, 'visited-forward')
    removeCellClass(pos.row, pos.col, 'visited-backward')
    setCellClass(pos.row, pos.col, 'path')
    await sleep(DELAY)
  }
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    resetGridState()
    await bidirectionalBFS()
    await sleep(2500)
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

.cell.visited-forward {
  background: #3b82f6;
}

.cell.visited-backward {
  background: #8b5cf6;
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

.legend-cell.visited-forward {
  background: #3b82f6;
}

.legend-cell.visited-backward {
  background: #8b5cf6;
}

.legend-cell.path {
  background: #10b981;
}
</style>
