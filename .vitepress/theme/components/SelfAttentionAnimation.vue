<template>
  <div class="attention-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <button @click="startAnim" class="btn btn-blue" :disabled="playing">开始演示</button>
      <button @click="resetAll" class="btn btn-gray" :disabled="playing">重置</button>
      <select v-model="queryIdx" class="sel" :disabled="playing">
        <option v-for="i in seqLen" :key="i" :value="i-1">查询位置: {{ tokens[i-1] }}</option>
      </select>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">自注意力机制：计算每个token对所有token的注意力权重，捕捉序列内部依赖关系</span>
    </div>

    <!-- 主可视化区域 -->
    <div class="vis-area">
      <!-- 输入序列 -->
      <div class="seq-row">
        <span class="lbl">输入序列:</span>
        <div class="tokens">
          <span v-for="(tok, i) in tokens" :key="i" 
                :class="['tok', {active: i === queryIdx, highlight: attentionWeights[i] > 0.15}]">
            {{ tok }}
            <span class="idx">{{ i }}</span>
          </span>
        </div>
      </div>

      <!-- QKV可视化 -->
      <div class="qkv-area">
        <div class="qkv-col">
          <div class="qkv-title">Query (Q)</div>
          <div :class="['qkv-vec', 'q']">
            <span v-for="(v, i) in currentQ" :key="i" class="cell" :style="{opacity: Math.abs(v)/4 + 0.3}">
              {{ v.toFixed(1) }}
            </span>
          </div>
        </div>
        <div class="qkv-col">
          <div class="qkv-title">Key (K)</div>
          <div v-for="(k, i) in allK" :key="i" :class="['qkv-vec', 'k', {active: i === queryIdx}]">
            <span v-for="(v, j) in k" :key="j" class="cell" :style="{opacity: Math.abs(v)/4 + 0.3}">
              {{ v.toFixed(1) }}
            </span>
          </div>
        </div>
        <div class="qkv-col">
          <div class="qkv-title">Value (V)</div>
          <div v-for="(v, i) in allV" :key="i" :class="['qkv-vec', 'v', {active: i === queryIdx}]">
            <span v-for="(val, j) in v" :key="j" class="cell" :style="{opacity: Math.abs(val)/4 + 0.3}">
              {{ val.toFixed(1) }}
            </span>
          </div>
        </div>
      </div>

      <!-- 注意力分数计算 -->
      <div class="score-area" v-if="step >= 1">
        <div class="score-title">注意力分数 = Q·Kᵀ / √dₖ</div>
        <div class="scores">
          <div v-for="(s, i) in rawScores" :key="i" :class="['score-item', {active: i === queryIdx}]">
            <span class="tok-lbl">{{ tokens[i] }}</span>
            <span class="score-val">{{ s.toFixed(2) }}</span>
          </div>
        </div>
      </div>

      <!-- Softmax可视化 -->
      <div class="softmax-area" v-if="step >= 2">
        <div class="softmax-title">Softmax归一化 → 注意力权重</div>
        <div class="weights-vis">
          <div v-for="(w, i) in attentionWeights" :key="i" class="weight-bar">
            <span class="tok-lbl">{{ tokens[i] }}</span>
            <div class="bar-container">
              <div class="bar" :style="{width: (w * 100) + '%'}"></div>
            </div>
            <span class="weight-val">{{ (w * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>

      <!-- 加权求和 -->
      <div class="output-area" v-if="step >= 3">
        <div class="output-title">输出 = Σ(注意力权重 × Value)</div>
        <div class="output-vec">
          <span v-for="(v, i) in output" :key="i" class="cell output-cell">
            {{ v.toFixed(2) }}
          </span>
        </div>
      </div>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 公式展示 -->
    <div class="formula-area">
      <div class="formula">
        <strong>核心公式:</strong> Attention(Q,K,V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) × V
      </div>
      <div class="formula-detail" v-if="currentCalc">
        {{ currentCalc }}
      </div>
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
      <span><i class="d q"></i>Query</span>
      <span><i class="d k"></i>Key</span>
      <span><i class="d v"></i>Value</span>
      <span><i class="d active"></i>当前位置</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

const tokens = ['我', '爱', '学习', 'AI']
const seqLen = tokens.length
const dK = 4  // 向量维度

const queryIdx = ref(0)
const playing = ref(false)
const step = ref(0)
const msg = ref('选择查询位置，点击"开始演示"查看自注意力计算过程')
const acts = ref([])
let timer = null

// 模拟Q、K、V向量（简化展示）
const allQ = ref([
  [1.2, 0.8, -0.5, 0.3],
  [0.5, 1.5, 0.2, -0.8],
  [-0.3, 0.9, 1.1, 0.6],
  [0.7, -0.4, 0.8, 1.2]
])

const allK = ref([
  [0.9, 1.1, -0.3, 0.5],
  [0.6, 1.3, 0.4, -0.6],
  [-0.2, 0.7, 1.0, 0.4],
  [0.8, -0.2, 0.9, 1.0]
])

const allV = ref([
  [0.5, 0.8, 0.2, 0.1],
  [0.9, 0.3, 0.7, -0.2],
  [0.1, 0.6, 0.9, 0.4],
  [0.7, 0.5, 0.3, 0.8]
])

// 当前查询向量
const currentQ = computed(() => allQ.value[queryIdx.value])

// 计算注意力分数 Q·K^T
const rawScores = computed(() => {
  const q = currentQ.value
  const scores = []
  for (let i = 0; i < seqLen; i++) {
    let score = 0
    for (let j = 0; j < dK; j++) {
      score += q[j] * allK.value[i][j]
    }
    scores.push(score / Math.sqrt(dK))  // 缩放
  }
  return scores
})

// Softmax计算注意力权重
const attentionWeights = computed(() => {
  const scores = rawScores.value
  const maxScore = Math.max(...scores)
  const expScores = scores.map(s => Math.exp(s - maxScore))
  const sumExp = expScores.reduce((a, b) => a + b, 0)
  return expScores.map(e => e / sumExp)
})

// 加权求和得到输出
const output = computed(() => {
  const weights = attentionWeights.value
  const result = new Array(dK).fill(0)
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < dK; j++) {
      result[j] += weights[i] * allV.value[i][j]
    }
  }
  return result
})

// 当前计算公式
const currentCalc = computed(() => {
  if (step.value === 1) {
    return `Q[${tokens[queryIdx.value]}] · K^T / √${dK} = [${rawScores.value.map(s => s.toFixed(2)).join(', ')}]`
  } else if (step.value === 2) {
    return `softmax([${rawScores.value.map(s => s.toFixed(2)).join(', ')}]) = [${attentionWeights.value.map(w => (w*100).toFixed(1) + '%').join(', ')}]`
  } else if (step.value >= 3) {
    return `加权求和: [${output.value.map(o => o.toFixed(2)).join(', ')}]`
  }
  return ''
})

// 生成动画步骤
function genAnim() {
  acts.value = []
  acts.value.push({
    t: 'show_qkv',
    m: `展示输入序列的 Q、K、V 向量，当前位置: "${tokens[queryIdx.value]}"`
  })
  acts.value.push({
    t: 'calc_scores',
    m: `计算注意力分数: Q·Kᵀ / √dₖ，缩放因子√${dK}防止梯度消失`
  })
  acts.value.push({
    t: 'softmax',
    m: `Softmax归一化，转换为概率分布（注意力权重）`
  })
  acts.value.push({
    t: 'weighted_sum',
    m: `加权求和: 每个Value向量乘以对应的注意力权重后相加`
  })
  acts.value.push({
    t: 'done',
    m: `自注意力计算完成！"${tokens[queryIdx.value]}"的输出融合了所有token的信息`
  })
}

// 执行步骤
function exec(a) {
  switch (a.t) {
    case 'show_qkv':
      msg.value = a.m
      break
    case 'calc_scores':
      msg.value = a.m
      break
    case 'softmax':
      msg.value = a.m
      break
    case 'weighted_sum':
      msg.value = a.m
      break
    case 'done':
      msg.value = a.m
      break
  }
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
  step.value = 0
  msg.value = '选择查询位置，点击"开始演示"查看自注意力计算过程'
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
    timer = setTimeout(autoPlay, 1500)
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
  msg.value = '选择查询位置，点击"开始演示"查看自注意力计算过程'
}

// 重置全部
function resetAll() {
  resetAnim()
}

// 监听查询位置变化
watch(queryIdx, () => {
  resetAnim()
})

onMounted(() => {
  // 初始化
})
</script>

<style scoped>
.attention-box {
  padding: 15px;
  font-family: system-ui, sans-serif;
  font-size: 14px;
}

.ctrl-row {
  display: flex;
  gap: 10px;
  align-items: center;
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

.sel {
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #d1d5db;
  font-size: 13px;
}

.tip-row { margin-bottom: 12px; }
.tip { font-size: 12px; color: #6b7280; }

.vis-area {
  background: #f8fafc;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 12px;
}

.seq-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

.lbl { font-weight: 600; color: #374151; min-width: 70px; }

.tokens {
  display: flex;
  gap: 8px;
}

.tok {
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  padding: 6px 12px;
  font-weight: 500;
  position: relative;
  transition: all 0.3s;
}

.tok.active {
  border-color: #3b82f6;
  background: #eff6ff;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.tok.highlight {
  background: #fef3c7;
  border-color: #f59e0b;
}

.tok .idx {
  position: absolute;
  top: -8px;
  right: -5px;
  background: #6b7280;
  color: white;
  font-size: 10px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.qkv-area {
  display: flex;
  gap: 20px;
  margin-bottom: 15px;
}

.qkv-col {
  flex: 1;
}

.qkv-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
  text-align: center;
  font-size: 13px;
}

.qkv-vec {
  display: flex;
  gap: 4px;
  margin-bottom: 6px;
  justify-content: center;
  transition: all 0.3s;
}

.qkv-vec.active {
  transform: scale(1.05);
}

.qkv-vec.q .cell { background: #dbeafe; border-color: #3b82f6; }
.qkv-vec.k .cell { background: #dcfce7; border-color: #22c55e; }
.qkv-vec.v .cell { background: #fef3c7; border-color: #f59e0b; }
.qkv-vec.k.active .cell { background: #86efac; }
.qkv-vec.v.active .cell { background: #fde047; }

.cell {
  width: 36px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}

.output-cell {
  background: #c7d2fe;
  border-color: #6366f1;
  width: 45px;
  height: 32px;
  font-size: 12px;
}

.score-area, .softmax-area, .output-area {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px dashed #d1d5db;
}

.score-title, .softmax-title, .output-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  font-size: 13px;
}

.scores {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.score-item {
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.score-item.active {
  border-color: #3b82f6;
  background: #eff6ff;
}

.tok-lbl { font-size: 12px; font-weight: 600; color: #6b7280; }
.score-val { font-size: 14px; font-weight: 700; color: #1f2937; }

.weights-vis {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.weight-bar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bar-container {
  flex: 1;
  height: 20px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.bar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  transition: width 0.5s;
}

.weight-val {
  min-width: 50px;
  text-align: right;
  font-weight: 600;
  color: #374151;
}

.output-vec {
  display: flex;
  gap: 6px;
  justify-content: center;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
}

.formula-area {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 10px;
}

.formula {
  font-size: 14px;
  color: #92400e;
  margin-bottom: 6px;
}

.formula-detail {
  font-size: 12px;
  color: #78716c;
  font-family: monospace;
  background: white;
  padding: 8px;
  border-radius: 4px;
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
  border-radius: 4px;
  display: inline-block;
}
.d.q { background: #dbeafe; border: 2px solid #3b82f6; }
.d.k { background: #dcfce7; border: 2px solid #22c55e; }
.d.v { background: #fef3c7; border: 2px solid #f59e0b; }
.d.active { background: #eff6ff; border: 2px solid #3b82f6; }
</style>
