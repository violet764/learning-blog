<template>
  <div class="trie-box">
    <!-- 控制区 -->
    <div class="control-row">
      <select v-model="opType" class="sel">
        <option value="insert">插入</option>
        <option value="search">搜索</option>
        <option value="startsWith">前缀</option>
        <option value="delete">删除</option>
      </select>
      <input v-model="word" placeholder="单词" class="inp" @keyup.enter="run" :disabled="playing" />
      <button @click="run" class="btn btn-blue" :disabled="playing || !word">执行</button>
      <button @click="resetAll" class="btn btn-gray">重置树</button>
    </div>

    <!-- 提示 -->
    <div class="tip">💡 树区域可拖动查看</div>

    <!-- 树显示 - 可拖动 -->
    <div class="tree-area" ref="treeArea" @mousedown="startDrag" @mousemove="onDrag" @mouseup="endDrag" @mouseleave="endDrag" @wheel="onWheel">
      <svg :width="viewW" :height="viewH" class="svg-tree">
        <g :transform="`translate(${panX}, ${panY}) scale(${scale})`">
          <!-- 连线 -->
          <line v-for="(e, i) in edges" :key="'e' + i"
            :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
            :class="['ln', e.hl && 'hl']" />
          <!-- 节点 -->
          <g v-for="(n, i) in nodes" :key="'n' + i">
            <circle :cx="n.x" :cy="n.y" :r="n.root ? 18 : 14" :class="['nd', n.st]" />
            <text :x="n.x" :y="n.y + 5" class="tx">{{ n.root ? 'root' : n.ch }}</text>
            <text v-if="n.end && !n.root" :x="n.x + 12" :y="n.y - 8" class="mk">★</text>
          </g>
        </g>
      </svg>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 路径 -->
    <div class="path-row" v-if="path.length">
      <span>路径:</span>
      <span v-for="(c, i) in path" :key="i" :class="['pc', i < dep && 'on']">{{ c }}</span>
    </div>

    <!-- 播放控制 -->
    <div class="ctl-row" v-if="acts.length">
      <button @click="prev" class="btn btn-sm" :disabled="idx <= 0 || playing">上一步</button>
      <button @click="toggle" class="btn btn-sm btn-blue">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="nextStep" class="btn btn-sm" :disabled="idx >= acts.length || playing">下一步</button>
      <button @click="resetAnim" class="btn btn-sm btn-gray">重置动画</button>
    </div>

    <div class="prog" v-if="acts.length">{{ idx }} / {{ acts.length }}</div>

    <!-- 图例 -->
    <div class="leg">
      <span><i class="d nrm"></i>普通</span>
      <span><i class="d cur"></i>当前</span>
      <span><i class="d fnd"></i>找到</span>
      <span><i class="d fail"></i>失败</span>
      <span><i class="d end"></i>结尾</span>
      <span><i class="d del"></i>删除</span>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'

const opType = ref('insert')
const word = ref('')
const msg = ref('输入单词开始操作')
const playing = ref(false)
const idx = ref(0)

// 视图尺寸
const viewW = ref(600)
const viewH = ref(320)

// 拖动和缩放
const panX = ref(0)
const panY = ref(0)
const scale = ref(1)
const isDragging = ref(false)
const dragStartX = ref(0)
const dragStartY = ref(0)
const startPanX = ref(0)
const startPanY = ref(0)

const nodes = ref([])
const edges = ref([])
const acts = ref([])
const path = ref([])
const dep = ref(0)

const treeArea = ref(null)

// Trie 数据结构
class TrieNode {
  constructor() {
    this.children = {}
    this.is_end = false
  }
}

const tree = reactive({ root: new TrieNode() })

// 拖动处理
function startDrag(e) {
  if (e.target.closest('circle') || e.target.closest('text')) return
  isDragging.value = true
  dragStartX.value = e.clientX
  dragStartY.value = e.clientY
  startPanX.value = panX.value
  startPanY.value = panY.value
}

function onDrag(e) {
  if (!isDragging.value) return
  const dx = e.clientX - dragStartX.value
  const dy = e.clientY - dragStartY.value
  panX.value = startPanX.value + dx
  panY.value = startPanY.value + dy
}

function endDrag() {
  isDragging.value = false
}

// 缩放处理
function onWheel(e) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  scale.value = Math.max(0.3, Math.min(2, scale.value * delta))
}

// 计算树布局
function layout() {
  nodes.value = []
  edges.value = []

  // 根节点
  nodes.value.push({
    x: viewW.value / 2,
    y: 40,
    root: true,
    end: false,
    st: '',
    ch: ''
  })

  // 递归布局子节点
  function walk(node, x, y, width, level) {
    const kids = Object.entries(node.children || {})
    if (!kids.length) return

    const childWidth = Math.max(width / kids.length, 60)
    const startX = x - (kids.length - 1) * childWidth / 2
    const gap = 60

    kids.forEach(([ch, child], i) => {
      const cx = startX + i * childWidth
      const cy = y + gap

      nodes.value.push({
        x: cx,
        y: cy,
        ch: ch,
        end: child.is_end,
        st: '',
        dataNode: child
      })

      edges.value.push({
        x1: x,
        y1: y + 18,
        x2: cx,
        y2: cy - 14,
        hl: false
      })

      walk(child, cx, cy, childWidth * 0.85, level + 1)
    })
  }

  walk(tree.root, viewW.value / 2, 40, viewW.value * 0.8, 1)
}

function resetLayout() {
  nodes.value = [{
    x: viewW.value / 2,
    y: 40,
    root: true,
    end: false,
    st: '',
    ch: ''
  }]
  edges.value = []
}

// 执行操作
function run() {
  if (!word.value || playing.value) return
  const w = word.value.toLowerCase().replace(/[^a-z]/g, '')
  if (!w) {
    msg.value = '请输入有效字母'
    return
  }
  genActs(w, opType.value)
}

// 生成动画动作
function genActs(w, type) {
  acts.value = []
  path.value = w.split('')
  idx.value = 0
  dep.value = 0

  switch (type) {
    case 'insert': genInsert(w); break
    case 'search': genSearch(w); break
    case 'startsWith': genPrefix(w); break
    case 'delete': genDelete(w); break
  }
}

// 插入动画
function genInsert(w) {
  let node = tree.root
  w.split('').forEach((c, i) => {
    acts.value.push({ t: 'hl', d: i + 1, c, m: `查找字符 '${c}'...` })

    if (!node.children[c]) {
      acts.value.push({ t: 'cr', d: i + 1, c, m: `创建新节点 '${c}'` })
      node.children[c] = new TrieNode()
    }

    node = node.children[c]
    acts.value.push({ t: 'mv', d: i + 1, c, m: `移动到 '${c}'` })
  })

  acts.value.push({ t: 'end', d: w.length, m: `标记单词结尾` })
  node.is_end = true
  acts.value.push({ t: 'done', m: `单词 '${w}' 插入完成!` })
}

// 搜索动画
function genSearch(w) {
  let node = tree.root
  for (let i = 0; i < w.length; i++) {
    const c = w[i]
    acts.value.push({ t: 'hl', d: i + 1, c, m: `查找 '${c}'...` })

    if (!node.children[c]) {
      acts.value.push({ t: 'fail', d: i + 1, m: `字符 '${c}' 不存在，单词未找到!` })
      return
    }

    node = node.children[c]
    acts.value.push({ t: 'ok', d: i + 1, m: `找到 '${c}'` })
  }

  if (node.is_end) {
    acts.value.push({ t: 'succ', d: w.length, m: `单词 '${w}' 存在!` })
  } else {
    acts.value.push({ t: 'fail', m: `'${w}' 只是前缀，不是完整单词` })
  }
}

// 前缀匹配动画
function genPrefix(w) {
  let node = tree.root
  for (let i = 0; i < w.length; i++) {
    const c = w[i]
    acts.value.push({ t: 'hl', d: i + 1, c, m: `查找 '${c}'...` })

    if (!node.children[c]) {
      acts.value.push({ t: 'fail', d: i + 1, m: `前缀不存在!` })
      return
    }

    node = node.children[c]
    acts.value.push({ t: 'ok', d: i + 1, m: `找到 '${c}'` })
  }

  acts.value.push({ t: 'succ', d: w.length, m: `前缀 '${w}' 存在!` })
}

// 删除动画
function genDelete(w) {
  let node = tree.root
  const pathNodes = [{ node, char: null }]

  // 查找单词
  for (let i = 0; i < w.length; i++) {
    const c = w[i]
    acts.value.push({ t: 'hl', d: i + 1, c, m: `查找 '${c}'...` })

    if (!node.children[c]) {
      acts.value.push({ t: 'fail', d: i + 1, m: `单词 '${w}' 不存在!` })
      return
    }

    node = node.children[c]
    pathNodes.push({ node, char: c })
    acts.value.push({ t: 'ok', d: i + 1, m: `找到 '${c}'` })
  }

  if (!node.is_end) {
    acts.value.push({ t: 'fail', m: `'${w}' 不是完整单词，无法删除` })
    return
  }

  // 标记删除
  acts.value.push({ t: 'del', d: w.length, m: `标记删除 '${w}'` })
  node.is_end = false

  // 从后向前删除无用节点
  for (let i = pathNodes.length - 1; i > 0; i--) {
    const { node: pNode, char } = pathNodes[i - 1]
    const cNode = pathNodes[i].node

    if (Object.keys(cNode.children).length === 0 && !cNode.is_end) {
      acts.value.push({ t: 'rml', d: i, c: pathNodes[i].char, m: `删除无用节点 '${pathNodes[i].char}'` })
      delete pNode.children[pathNodes[i].char]
    } else {
      break
    }
  }

  acts.value.push({ t: 'done', m: `单词 '${w}' 删除完成!` })
}

// 执行动作
function exec(a) {
  dep.value = a.d || 0

  // 重置节点状态
  nodes.value.forEach(n => n.st = '')
  edges.value.forEach(e => e.hl = false)

  // 高亮路径边
  for (let i = 0; i < dep.value - 1 && i < edges.value.length; i++) {
    edges.value[i].hl = true
  }

  // 设置当前节点状态
  if (dep.value > 0 && nodes.value[dep.value]) {
    const n = nodes.value[dep.value]
    switch (a.t) {
      case 'hl': n.st = 'cur'; break
      case 'ok': n.st = 'fnd'; break
      case 'fail': n.st = 'fail'; break
      case 'end': n.end = true; n.st = 'end'; break
      case 'del': n.st = 'del'; n.end = false; break
      case 'succ': n.st = 'fnd'; break
      case 'rml': n.st = 'del'; break
    }
  }

  // 更新布局
  if (a.t === 'cr' || a.t === 'rml' || a.t === 'done') {
    layout()
  }

  msg.value = a.m
}

// 下一步
function nextStep() {
  if (idx.value >= acts.value.length) return
  exec(acts.value[idx.value])
  idx.value++

  if (idx.value >= acts.value.length) {
    playing.value = false
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }
}

// 上一步
function prev() {
  if (idx.value <= 0) return
  playing.value = false
  if (timer) {
    clearTimeout(timer)
    timer = null
  }

  idx.value--
  dep.value = 0
  nodes.value.forEach(n => n.st = '')
  edges.value.forEach(e => e.hl = false)
  layout()

  for (let i = 0; i < idx.value; i++) {
    exec(acts.value[i])
  }
}

let timer = null

// 切换播放
function toggle() {
  if (playing.value) {
    playing.value = false
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  } else {
    if (idx.value >= acts.value.length) resetAnim()
    playing.value = true
    autoPlay()
  }
}

// 自动播放
function autoPlay() {
  if (!playing.value) return
  if (idx.value < acts.value.length) {
    nextStep()
    timer = setTimeout(autoPlay, 500)
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
  idx.value = 0
  dep.value = 0
  acts.value = []
  path.value = []
  layout()
  msg.value = '输入单词开始操作'
}

// 重置全部
function resetAll() {
  resetAnim()
  tree.root = new TrieNode()
  resetLayout()
  panX.value = 0
  panY.value = 0
  scale.value = 1
  msg.value = '字典树已重置'
}

onMounted(() => {
  resetLayout()
})

onUnmounted(() => {
  if (timer) clearTimeout(timer)
})
</script>

<style scoped>
.trie-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.control-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.tip {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 8px;
}

.sel, .inp {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
}

.inp { width: 100px; }

.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-blue { background: #3b82f6; color: white; }
.btn-blue:hover:not(:disabled) { background: #2563eb; }
.btn-gray { background: #6b7280; color: white; }
.btn-sm { padding: 5px 10px; font-size: 12px; }

.tree-area {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 12px;
  cursor: grab;
  user-select: none;
}

.tree-area:active { cursor: grabbing; }

.svg-tree {
  display: block;
  background: linear-gradient(to bottom, #f8fafc 0%, #f1f5f9 100%);
}

.ln {
  stroke: #cbd5e1;
  stroke-width: 2;
  transition: stroke 0.2s, stroke-width 0.2s;
}
.ln.hl { stroke: #3b82f6; stroke-width: 3; }

.nd {
  fill: white;
  stroke: #94a3b8;
  stroke-width: 2;
  transition: fill 0.2s, stroke 0.2s;
}
.nd.cur { fill: #3b82f6; stroke: #1d4ed8; }
.nd.fnd { fill: #10b981; stroke: #059669; }
.nd.fail { fill: #ef4444; stroke: #dc2626; }
.nd.end { stroke: #f59e0b; stroke-width: 3; }
.nd.del { fill: #f97316; stroke: #ea580c; }

.tx {
  font-size: 12px;
  font-weight: 600;
  text-anchor: middle;
  dominant-baseline: middle;
  fill: #374151;
  pointer-events: none;
}

.mk {
  font-size: 14px;
  fill: #f59e0b;
  font-weight: bold;
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
  gap: 6px;
  margin-bottom: 10px;
  font-size: 13px;
}

.pc {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #e2e8f0;
  border-radius: 4px;
  font-weight: 600;
  font-size: 12px;
  transition: background 0.2s;
}
.pc.on { background: #3b82f6; color: white; }

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
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid #94a3b8;
  display: inline-block;
}
.d.nrm { background: white; }
.d.cur { background: #3b82f6; border-color: #1d4ed8; }
.d.fnd { background: #10b981; border-color: #059669; }
.d.fail { background: #ef4444; border-color: #dc2626; }
.d.end { background: white; border-color: #f59e0b; border-width: 3px; }
.d.del { background: #f97316; border-color: #ea580c; }
</style>
