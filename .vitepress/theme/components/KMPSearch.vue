<template>
  <div class="kmp-container">
    <div class="info-section">
      <!-- 文本串 -->
      <div class="text-row">
        <span class="label">文本串:</span>
        <div class="text-chars">
          <span
            v-for="(char, index) in textChars"
            :key="index"
            class="char"
            :class="getTextCharClass(index)"
          >{{ char }}</span>
        </div>
      </div>

      <!-- 模式串（带偏移，对齐到文本） -->
      <div class="pattern-row">
        <span class="label">模式串:</span>
        <div class="pattern-container">
          <div
            class="pattern-wrapper"
            :style="{ marginLeft: `${patternOffset * 32}px` }"
          >
            <span
              v-for="(char, index) in patternChars"
              :key="index"
              class="char"
              :class="getPatternCharClass(index)"
            >{{ char }}</span>
          </div>
        </div>
      </div>

      <!-- next数组 -->
      <div class="next-row">
        <span class="label">next数组:</span>
        <div class="chars">
          <span
            v-for="(num, index) in nextArray"
            :key="index"
            class="char next"
            :class="getNextClass(index)"
          >{{ num }}</span>
        </div>
      </div>
    </div>

    <div class="status">{{ status }}</div>

    <!-- 控制按钮 -->
    <div class="controls">
      <button @click="prevStep" class="btn prev-btn" :disabled="currentStep <= 0 || isPlaying">
        ⏮ 上一步
      </button>
      <button @click="togglePlay" class="btn play-btn" :class="isPlaying ? 'pause-style' : 'play-style'">
        {{ isPlaying ? '⏸ 暂停' : '▶ 播放' }}
      </button>
      <button @click="nextStep" class="btn next-btn" :disabled="currentStep >= animationQueue.length || isPlaying">
        下一步 ⏭
      </button>
      <button @click="resetAnimation" class="btn reset-btn">
        ↺ 重置
      </button>
    </div>

    <div class="progress-info">
      <span>进度: {{ currentStep }} / {{ animationQueue.length }}</span>
    </div>

    <!-- 图例 -->
    <div class="legend">
      <h4>图例</h4>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-box current-text"></div>
          <span>当前文本</span>
        </div>
        <div class="legend-item">
          <div class="legend-box current-pattern"></div>
          <span>当前模式</span>
        </div>
        <div class="legend-item">
          <div class="legend-box match"></div>
          <span>匹配</span>
        </div>
        <div class="legend-item">
          <div class="legend-box mismatch"></div>
          <span>不匹配</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const text = 'ABABABCABABABCABABC'
const pattern = 'ABABCABAB'

const textChars = ref([])
const patternChars = ref([])
const nextArray = ref([])
const status = ref('点击"播放"或"下一步"开始')

// 动画控制
const animationQueue = ref([])
const currentStep = ref(0)
const isPlaying = ref(false)

// 模式串偏移量（相对于文本串的起始位置）
const patternOffset = ref(0)

// 状态存储
const textCharStates = ref({})
const patternCharStates = ref({})
const nextStates = ref({})
const history = ref([])

let playTimer = null

function init() {
  textChars.value = text.split('')
  patternChars.value = pattern.split('')
  nextArray.value = new Array(pattern.length).fill(0)
  textCharStates.value = {}
  patternCharStates.value = {}
  nextStates.value = {}
  history.value = []
  currentStep.value = 0
  patternOffset.value = 0
}

// 保存当前状态到历史
function saveState(action) {
  history.value = history.value.slice(0, currentStep.value)
  history.value.push({
    textCharStates: JSON.parse(JSON.stringify(textCharStates.value)),
    patternCharStates: JSON.parse(JSON.stringify(patternCharStates.value)),
    nextStates: JSON.parse(JSON.stringify(nextStates.value)),
    nextArray: [...nextArray.value],
    patternOffset: patternOffset.value,
    status: status.value,
    action: action
  })
  currentStep.value = history.value.length
}

// 生成动画队列
function generateAnimation() {
  animationQueue.value = []

  // ====== 计算 next 数组阶段 ======
  animationQueue.value.push({
    type: 'status',
    value: '计算 next 数组（预处理模式串）...',
    phase: 'next'
  })

  nextArray.value[0] = 0

  let j = 0
  for (let i = 1; i < pattern.length; i++) {
    // 高亮比较的字符
    animationQueue.value.push({
      type: 'nextHighlight',
      indices: [j, i],
      value: `比较 pattern[${j}]='${pattern[j]}' 和 pattern[${i}]='${pattern[i]}'`
    })

    while (j > 0 && pattern[i] !== pattern[j]) {
      animationQueue.value.push({
        type: 'nextBacktrack',
        index: j,
        value: `不匹配，回溯 j = next[${j}] = ${nextArray.value[j - 1]}`
      })
      j = nextArray.value[j - 1]
      if (j > 0) {
        animationQueue.value.push({
          type: 'nextHighlight',
          indices: [j, i],
          value: `继续比较 pattern[${j}]='${pattern[j]}' 和 pattern[${i}]='${pattern[i]}'`
        })
      }
    }

    if (pattern[i] === pattern[j]) {
      j++
    }
    nextArray.value[i] = j

    animationQueue.value.push({
      type: 'nextSet',
      index: i,
      value: `next[${i}] = ${j}`
    })
  }

  animationQueue.value.push({
    type: 'nextClear',
    value: 'next 数组计算完成，开始匹配...'
  })

  // ====== 匹配阶段 ======
  j = 0
  let textIndex = 0

  while (textIndex < text.length) {
    // 设置模式串偏移（模式串第0个字符对齐到文本第textIndex个字符）
    animationQueue.value.push({
      type: 'setOffset',
      offset: textIndex,
      value: `模式串从位置 ${textIndex} 开始匹配`
    })

    // 当前要比较的文本位置和模式位置
    const currentTextIdx = textIndex + j

    if (currentTextIdx >= text.length) break

    // 高亮当前比较的字符
    animationQueue.value.push({
      type: 'compare',
      textIndex: currentTextIdx,
      patternIndex: j,
      value: `比较 text[${currentTextIdx}]='${text[currentTextIdx]}' 与 pattern[${j}]='${pattern[j]}'`
    })

    // 匹配或不匹配
    if (text[currentTextIdx] === pattern[j]) {
      animationQueue.value.push({
        type: 'match',
        textIndex: currentTextIdx,
        patternIndex: j,
        value: `匹配成功！text[${currentTextIdx}]='${text[currentTextIdx]}' == pattern[${j}]='${pattern[j]}'`
      })
      j++

      // 检查是否找到完整匹配
      if (j === pattern.length) {
        animationQueue.value.push({
          type: 'found',
          startIndex: textIndex,
          value: `找到完整匹配！起始位置: ${textIndex}`
        })

        // 继续搜索下一个匹配
        j = nextArray.value[j - 1]
        textIndex++
      }
    } else {
      // 不匹配，回溯
      animationQueue.value.push({
        type: 'mismatch',
        textIndex: currentTextIdx,
        patternIndex: j,
        value: `不匹配！j = next[${j}] = ${nextArray.value[j]}, 模式串右移`
      })

      // j 回溯，文本位置前进
      const nextJ = nextArray.value[j]
      const shift = j === 0 ? 1 : j - nextJ  // 当 j=0 时至少前进1
      j = nextJ
      textIndex += shift

      if (textIndex >= text.length) break
    }
  }

  animationQueue.value.push({
    type: 'status',
    value: '搜索完成！'
  })
}

function executeAction(action) {
  switch (action.type) {
    case 'status':
      status.value = action.value
      break

    case 'setOffset':
      patternOffset.value = action.offset
      status.value = action.value
      break

    case 'nextHighlight':
      nextStates.value = {}
      action.indices.forEach((idx, i) => {
        nextStates.value[idx] = i === 0 ? 'comparing' : 'current'
      })
      status.value = action.value
      break

    case 'nextBacktrack':
      nextStates.value[action.index] = 'backtrack'
      status.value = action.value
      break

    case 'nextSet':
      nextStates.value[action.index] = 'done'
      status.value = action.value
      break

    case 'nextClear':
      nextStates.value = {}
      status.value = action.value
      break

    case 'compare':
      // textIndex 是相对于文本串的位置
      // 模式串偏移 + patternIndex 应该等于 textIndex
      // 所以当前模式字符的位置是 textIndex - patternOffset
      const patternCharPos = action.textIndex - patternOffset.value
      textCharStates.value = { [action.textIndex]: 'current-text' }
      patternCharStates.value = { [patternCharPos]: 'current-pattern' }
      status.value = action.value
      break

    case 'mismatch':
      const mismatchPatternPos = action.textIndex - patternOffset.value
      textCharStates.value = { [action.textIndex]: 'mismatch' }
      patternCharStates.value = { [mismatchPatternPos]: 'mismatch' }
      status.value = action.value
      break

    case 'match':
      const matchPatternPos = action.textIndex - patternOffset.value
      textCharStates.value = { [action.textIndex]: 'match' }
      patternCharStates.value = { [matchPatternPos]: 'match' }
      status.value = action.value
      break

    case 'found':
      // 高亮找到的匹配子串
      textCharStates.value = {}
      for (let i = 0; i < pattern.length; i++) {
        textCharStates.value[action.startIndex + i] = 'found'
      }
      status.value = action.value
      break
  }
}

function nextStep() {
  if (currentStep.value >= animationQueue.value.length) return

  saveState(animationQueue.value[currentStep.value])
  executeAction(animationQueue.value[currentStep.value])
  currentStep.value++

  if (currentStep.value >= animationQueue.value.length) {
    isPlaying.value = false
    if (playTimer) {
      clearTimeout(playTimer)
      playTimer = null
    }
  }
}

function prevStep() {
  if (currentStep.value <= 0) return

  isPlaying.value = false
  if (playTimer) {
    clearTimeout(playTimer)
    playTimer = null
  }

  currentStep.value--
  const prevState = history.value[currentStep.value - 1]

  if (prevState) {
    textCharStates.value = JSON.parse(JSON.stringify(prevState.textCharStates))
    patternCharStates.value = JSON.parse(JSON.stringify(prevState.patternCharStates))
    nextStates.value = JSON.parse(JSON.stringify(prevState.nextStates))
    nextArray.value = [...prevState.nextArray]
    patternOffset.value = prevState.patternOffset
    status.value = prevState.status
  }
}

function togglePlay() {
  if (isPlaying.value) {
    isPlaying.value = false
    if (playTimer) {
      clearTimeout(playTimer)
      playTimer = null
    }
  } else {
    if (currentStep.value >= animationQueue.value.length) {
      resetAnimation()
    }
    isPlaying.value = true
    autoPlay()
  }
}

function autoPlay() {
  if (!isPlaying.value) return

  if (currentStep.value < animationQueue.value.length) {
    nextStep()
    playTimer = setTimeout(autoPlay, 700)
  } else {
    isPlaying.value = false
    status.value = '播放完毕，点击"重置"重新播放'
  }
}

function resetAnimation() {
  isPlaying.value = false
  if (playTimer) {
    clearTimeout(playTimer)
    playTimer = null
  }

  init()
  generateAnimation()
  status.value = '点击"播放"或"下一步"开始'
}

function getTextCharClass(index) {
  return textCharStates.value[index] || ''
}

function getPatternCharClass(index) {
  return patternCharStates.value[index] || ''
}

function getNextClass(index) {
  return nextStates.value[index] || ''
}

onMounted(() => {
  init()
  generateAnimation()
})

onUnmounted(() => {
  if (playTimer) {
    clearTimeout(playTimer)
  }
})
</script>

<style scoped>
.kmp-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 20px;
  font-family: 'Consolas', 'Monaco', monospace;
  overflow-x: auto;
}

.info-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.text-row, .pattern-row, .next-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.label {
  font-weight: 600;
  color: #374151;
  min-width: 80px;
  flex-shrink: 0;
}

.text-chars {
  display: flex;
  gap: 2px;
}

.pattern-container {
  position: relative;
  min-height: 36px;
}

.pattern-wrapper {
  display: flex;
  gap: 2px;
  transition: margin-left 0.3s ease;
}

.chars {
  display: flex;
  gap: 2px;
}

.char {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  font-size: 14px;
  transition: all 0.2s;
  flex-shrink: 0;
}

.char.next {
  width: 26px;
  height: 26px;
  font-size: 12px;
}

/* 文本字符状态 */
.current-text {
  background: #3b82f6 !important;
  color: white;
  border-color: #2563eb;
  transform: scale(1.15);
}

.match {
  background: #10b981 !important;
  color: white;
  border-color: #059669;
}

.mismatch {
  background: #ef4444 !important;
  color: white;
  border-color: #dc2626;
}

.found {
  background: #059669 !important;
  color: white;
  border-color: #047857;
  animation: pulse 0.5s ease;
}

/* 模式字符状态 */
.current-pattern {
  background: #f97316 !important;
  color: white;
  border-color: #ea580c;
  transform: scale(1.15);
}

/* next数组状态 */
.comparing {
  background: #fbbf24 !important;
  transform: scale(1.1);
}

.backtrack {
  background: #f97316 !important;
}

.current {
  background: #3b82f6 !important;
  color: white;
}

.done {
  background: #10b981 !important;
  color: white;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.status {
  font-size: 14px;
  color: #374151;
  padding: 12px 24px;
  background: #f3f4f6;
  border-radius: 6px;
  min-height: 24px;
  min-width: 300px;
  text-align: center;
}

.controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.btn {
  padding: 10px 18px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
}

.btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.prev-btn {
  background: #6366f1;
  color: white;
}

.play-style {
  background: #10b981;
  color: white;
}

.pause-style {
  background: #f59e0b;
  color: white;
}

.next-btn {
  background: #3b82f6;
  color: white;
}

.reset-btn {
  background: #6b7280;
  color: white;
}

.progress-info {
  font-size: 13px;
  color: #6b7280;
}

.legend {
  padding: 15px;
  background: #f9fafb;
  border-radius: 8px;
}

.legend h4 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #374151;
}

.legend-items {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #4b5563;
}

.legend-box {
  width: 18px;
  height: 18px;
  border-radius: 3px;
  border: 1px solid #d1d5db;
}

.legend-box.current-text {
  background: #3b82f6;
}

.legend-box.current-pattern {
  background: #f97316;
}

.legend-box.match {
  background: #10b981;
}

.legend-box.mismatch {
  background: #ef4444;
}
</style>
