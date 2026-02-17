<template>
  <div class="dfs-wrapper">
    <div class="grid" :style="gridStyle">
      <div
        v-for="(cell, index) in cells"
        :key="index"
        class="cell"
        :class="cell.className"
      ></div>
    </div>
    <div class="legend">
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
        <div class="legend-cell current"></div>
        <span>当前访问</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell visited"></div>
        <span>已访问</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell path"></div>
        <span>路径</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const ROWS = 8
const COLS = 12

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
    return {
      ...cell,
      visited: false,
      parent: null,
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

// 正确的 DFS - 使用递归模拟
async function dfs(row, col) {
  // 越界检查
  if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return false

  const idx = getIndex(row, col)
  const cell = cells.value[idx]

  // 障碍物或已访问
  if (cell.isObstacle || cell.visited) return false

  // 标记为已访问
  cell.visited = true

  // 动画效果
  if (!(row === startPos.row && col === startPos.col)) {
    setCellClass(row, col, 'current')
    await sleep(100)
    removeCellClass(row, col, 'current')
    setCellClass(row, col, 'visited')
  }

  // 找到终点
  if (row === endPos.row && col === endPos.col) {
    return true
  }

  // 四个方向：上、下、左、右（深度优先）
  const directions = [
    { dr: -1, dc: 0 },  // 上
    { dr: 1, dc: 0 },   // 下
    { dr: 0, dc: -1 },  // 左
    { dr: 0, dc: 1 }    // 右
  ]

  for (const dir of directions) {
    const newRow = row + dir.dr
    const newCol = col + dir.dc

    // 记录父节点
    const newIdx = getIndex(newRow, newCol)
    if (newRow >= 0 && newRow < ROWS && newCol >= 0 && newCol < COLS) {
      if (!cells.value[newIdx].isObstacle && !cells.value[newIdx].visited) {
        cells.value[newIdx].parent = { row, col }
      }
    }

    // 递归搜索
    if (await dfs(newRow, newCol)) {
      return true
    }
  }

  return false
}

// 显示路径
async function showPath() {
  const path = []
  let current = endPos

  while (current) {
    path.unshift(current)
    const idx = getIndex(current.row, current.col)
    current = cells.value[idx].parent
  }

  // 显示路径
  for (const pos of path) {
    removeCellClass(pos.row, pos.col, 'visited')
    setCellClass(pos.row, pos.col, 'path')
    await sleep(80)
  }
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    resetGridState()
    await dfs(startPos.row, startPos.col)
    await showPath()
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
.dfs-wrapper {
  display: flex;
  gap: 30px;
  align-items: flex-start;
  justify-content: center;
  padding: 20px;
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

.cell.current {
  background: #f97316;
  transform: scale(1.1);
}

.cell.visited {
  background: #60a5fa;
}

.cell.path {
  background: #10b981;
}

.legend {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 15px;
  background: #f9fafb;
  border-radius: 8px;
  min-width: 100px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #4b5563;
}

.legend-cell {
  width: 18px;
  height: 18px;
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

.legend-cell.current {
  background: #f97316;
}

.legend-cell.visited {
  background: #60a5fa;
}

.legend-cell.path {
  background: #10b981;
}
</style>
