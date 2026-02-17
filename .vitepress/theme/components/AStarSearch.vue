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
        <div class="legend-cell open"></div>
        <span>开放列表</span>
      </div>
      <div class="legend-item">
        <div class="legend-cell closed"></div>
        <span>已访问</span>
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
const DELAY = 120

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
        g: Infinity,
        h: 0,
        f: Infinity,
        parent: null,
        inOpen: false,
        inClosed: false,
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
      g: Infinity,
      h: 0,
      f: Infinity,
      parent: null,
      inOpen: false,
      inClosed: false,
      className
    }
  })
}

function getIndex(row, col) {
  return row * COLS + col
}

function heuristic(row1, col1, row2, col2) {
  // 曼哈顿距离
  return Math.abs(row1 - row2) + Math.abs(col1 - col2)
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

// 优先队列（最小堆）
class MinHeap {
  constructor() {
    this.heap = []
  }

  push(node) {
    this.heap.push(node)
    this.bubbleUp(this.heap.length - 1)
  }

  pop() {
    if (this.heap.length === 0) return null
    const min = this.heap[0]
    const last = this.heap.pop()
    if (this.heap.length > 0) {
      this.heap[0] = last
      this.bubbleDown(0)
    }
    return min
  }

  bubbleUp(index) {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2)
      if (this.heap[index].f >= this.heap[parentIndex].f) break
      [this.heap[index], this.heap[parentIndex]] = [this.heap[parentIndex], this.heap[index]]
      index = parentIndex
    }
  }

  bubbleDown(index) {
    while (true) {
      const leftChild = 2 * index + 1
      const rightChild = 2 * index + 2
      let smallest = index

      if (leftChild < this.heap.length && this.heap[leftChild].f < this.heap[smallest].f) {
        smallest = leftChild
      }
      if (rightChild < this.heap.length && this.heap[rightChild].f < this.heap[smallest].f) {
        smallest = rightChild
      }

      if (smallest === index) break
      [this.heap[index], this.heap[smallest]] = [this.heap[smallest], this.heap[index]]
      index = smallest
    }
  }

  isEmpty() {
    return this.heap.length === 0
  }

  update(node) {
    const index = this.heap.findIndex(n => n.row === node.row && n.col === node.col)
    if (index !== -1) {
      this.heap[index] = node
      this.bubbleUp(index)
      this.bubbleDown(index)
    }
  }

  contains(row, col) {
    return this.heap.some(n => n.row === row && n.col === col)
  }
}

async function aStarSearch() {
  const directions = [
    { dr: -1, dc: 0 },
    { dr: 1, dc: 0 },
    { dr: 0, dc: -1 },
    { dr: 0, dc: 1 }
  ]

  const openList = new MinHeap()

  // 初始化起点
  const startIdx = getIndex(startPos.row, startPos.col)
  cells.value[startIdx].g = 0
  cells.value[startIdx].h = heuristic(startPos.row, startPos.col, endPos.row, endPos.col)
  cells.value[startIdx].f = cells.value[startIdx].h
  cells.value[startIdx].inOpen = true

  openList.push({ ...startPos, f: cells.value[startIdx].f })

  status.value = 'A* 搜索开始...'
  await sleep(DELAY)

  while (!openList.isEmpty()) {
    // 取出 f 值最小的节点
    const current = openList.pop()
    const { row, col } = current
    const idx = getIndex(row, col)

    // 如果是终点，构建路径
    if (row === endPos.row && col === endPos.col) {
      await showPath()
      return
    }

    // 从开放列表移到关闭列表
    cells.value[idx].inOpen = false
    cells.value[idx].inClosed = true

    // 动画效果
    if (!(row === startPos.row && col === startPos.col)) {
      setCellClass(row, col, 'closed')
    }

    await sleep(DELAY / 2)

    // 探索四个方向
    for (const dir of directions) {
      const newRow = row + dir.dr
      const newCol = col + dir.dc

      // 边界检查
      if (newRow < 0 || newRow >= ROWS || newCol < 0 || newCol >= COLS) continue

      const nIdx = getIndex(newRow, newCol)
      const neighbor = cells.value[nIdx]

      // 跳过障碍物和已关闭节点
      if (neighbor.isObstacle || neighbor.inClosed) continue

      // 计算新的 g 值
      const newG = cells.value[idx].g + 1

      // 如果找到更好的路径
      if (newG < neighbor.g) {
        neighbor.g = newG
        neighbor.h = heuristic(newRow, newCol, endPos.row, endPos.col)
        neighbor.f = neighbor.g + neighbor.h
        neighbor.parent = { row, col }

        if (!neighbor.inOpen) {
          neighbor.inOpen = true
          openList.push({ row: newRow, col: newCol, f: neighbor.f })

          // 动画显示加入开放列表
          if (!(newRow === endPos.row && newCol === endPos.col)) {
            setCellClass(newRow, newCol, 'open')
          }
        } else {
          openList.update({ row: newRow, col: newCol, f: neighbor.f })
        }
      }
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

  // 动画显示路径
  for (const pos of path) {
    removeCellClass(pos.row, pos.col, 'open')
    removeCellClass(pos.row, pos.col, 'closed')
    setCellClass(pos.row, pos.col, 'path')
    await sleep(DELAY)
  }
}

async function animate() {
  if (isRunning) return
  isRunning = true

  while (true) {
    resetGridState()
    await aStarSearch()
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

.cell.open {
  background: #fbbf24;
}

.cell.closed {
  background: #60a5fa;
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

.legend-cell.open {
  background: #fbbf24;
}

.legend-cell.closed {
  background: #60a5fa;
}

.legend-cell.path {
  background: #10b981;
}
</style>
