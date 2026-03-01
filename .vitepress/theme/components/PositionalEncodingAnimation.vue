<template>
  <div class="pe-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <button @click="startAnim" class="btn btn-blue" :disabled="playing">开始演示</button>
      <button @click="resetAll" class="btn btn-gray" :disabled="playing">重置</button>
      <span class="mode-sel">
        <label>编码类型:</label>
        <select v-model="encodingType" :disabled="playing">
          <option value="sinusoidal">正弦余弦</option>
          <option value="learned">学习式</option>
          <option value="rope">RoPE旋转</option>
        </select>
      </span>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">{{ encodingTips[encodingType] }}</span>
    </div>

    <!-- 正弦余弦编码可视化 -->
    <div class="vis-area" v-if="encodingType === 'sinusoidal'">
      <!-- 位置编码矩阵 -->
      <div class="pe-matrix">
        <div class="matrix-title">位置编码矩阵 PE (max_len=16, d=8)</div>
        <div class="matrix-grid">
          <div class="matrix-header">
            <span class="corner"></span>
            <span v-for="d in 8" :key="d" class="dim-label">d{{ d-1 }}</span>
          </div>
          <div v-for="(row, pos) in sinCosMatrix" :key="pos" class="matrix-row">
            <span class="pos-label">{{ pos }}</span>
            <span v-for="(val, d) in row" :key="d" 
                  class="matrix-cell"
                  :style="{background: getHeatColor(val)}">
              {{ val.toFixed(2) }}
            </span>
          </div>
        </div>
      </div>

      <!-- 波形可视化 -->
      <div class="wave-area" v-if="step >= 1">
        <div class="wave-title">不同维度的正弦/余弦波形</div>
        <div class="waves-container">
          <div v-for="d in [0, 2, 4, 6]" :key="d" class="wave-item">
            <div class="wave-label">维度 {{ d }} (sin)</div>
            <svg class="wave-svg" viewBox="0 0 200 50">
              <path :d="getSinPath(d)" class="wave-path sin" />
            </svg>
          </div>
          <div v-for="d in [1, 3, 5, 7]" :key="d" class="wave-item">
            <div class="wave-label">维度 {{ d }} (cos)</div>
            <svg class="wave-svg" viewBox="0 0 200 50">
              <path :d="getCosPath(d)" class="wave-path cos" />
            </svg>
          </div>
        </div>
      </div>

      <!-- 相对位置性质 -->
      <div class="relative-area" v-if="step >= 2">
        <div class="relative-title">相对位置关系：PE(pos+k) 可由 PE(pos) 线性表示</div>
        <div class="relative-demo">
          <div class="rel-formula">
            PE<sub>pos+k, 2i</sub> = PE<sub>pos, 2i</sub> × cos(k×θ<sub>i</sub>) + PE<sub>pos, 2i+1</sub> × sin(k×θ<sub>i</sub>)
          </div>
          <div class="rel-explain">
            这使得模型能够学习相对位置关系，而非仅绝对位置
          </div>
        </div>
      </div>
    </div>

    <!-- RoPE可视化 -->
    <div class="vis-area" v-if="encodingType === 'rope'">
      <div class="rope-intro">
        <strong>旋转位置编码 (RoPE)</strong>：通过旋转向量来编码位置信息
      </div>
      
      <!-- 2D旋转示意 -->
      <div class="rope-2d" v-if="step >= 0">
        <div class="rope-title">2D向量旋转示意</div>
        <svg class="rope-svg" viewBox="0 0 300 200">
          <!-- 坐标轴 -->
          <line x1="20" y1="100" x2="280" y2="100" class="axis" />
          <line x1="150" y1="20" x2="150" y2="180" class="axis" />
          
          <!-- 原始向量 -->
          <line x1="150" y1="100" x2="220" y2="60" class="vec original" />
          <circle cx="220" cy="60" r="4" class="point original" />
          
          <!-- 旋转后的向量 -->
          <line x1="150" y1="100" x2="180" y2="40" class="vec rotated" />
          <circle cx="180" cy="40" r="4" class="point rotated" />
          
          <!-- 旋转角度标注 -->
          <path d="M 170 80 A 20 20 0 0 1 165 65" class="angle-arc" />
          <text x="155" y="75" class="angle-text">θ</text>
          
          <!-- 标签 -->
          <text x="225" y="55" class="label">原始向量</text>
          <text x="175" y="35" class="label">旋转后</text>
        </svg>
        <div class="rope-formula">
          f(x, m) = x × e<sup>imθ</sup> (复数乘法实现旋转)
        </div>
      </div>

      <!-- 多位置对比 -->
      <div class="rope-positions" v-if="step >= 1">
        <div class="pos-title">不同位置的旋转角度</div>
        <div class="pos-grid">
          <div v-for="pos in [0, 1, 2, 3]" :key="pos" class="pos-item">
            <div class="pos-label">pos={{ pos }}</div>
            <svg class="pos-svg" viewBox="0 0 80 80">
              <line x1="40" y1="40" x2="40" y2="15" class="vec-base" />
              <line x1="40" y1="40" :x2="40 - 25*Math.sin(pos * 0.5)" :y2="40 - 25*Math.cos(pos * 0.5)" class="vec-rot" />
              <circle cx="40" cy="40" r="3" class="center" />
            </svg>
            <div class="angle">θ={{ (pos * 0.5).toFixed(2) }}</div>
          </div>
        </div>
      </div>

      <!-- RoPE优势 -->
      <div class="rope-advantages" v-if="step >= 2">
        <div class="adv-title">RoPE 优势</div>
        <div class="adv-list">
          <div class="adv-item">✓ 相对位置编码自然实现</div>
          <div class="adv-item">✓ 长度外推能力强</div>
          <div class="adv-item">✓ 计算效率高（复数乘法）</div>
          <div class="adv-item">✓ 现代LLM主流选择（LLaMA、GPT-NeoX等）</div>
        </div>
      </div>
    </div>

    <!-- 学习式编码 -->
    <div class="vis-area" v-if="encodingType === 'learned'">
      <div class="learned-intro">
        <strong>学习式位置编码</strong>：将位置编码作为可训练参数
      </div>
      
      <div class="learned-params" v-if="step >= 0">
        <div class="params-title">可训练参数矩阵</div>
        <div class="params-grid">
          <div v-for="pos in 16" :key="pos" class="param-row">
            <span class="pos-label">p{{ pos-1 }}</span>
            <div class="param-vec">
              <span v-for="d in 8" :key="d" 
                    class="param-cell"
                    :style="{background: getHeatColor(learnedPE[pos-1][d-1])}">
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="learned-compare" v-if="step >= 1">
        <div class="compare-title">对比分析</div>
        <table class="compare-table">
          <tr>
            <th>特性</th>
            <th>正弦余弦</th>
            <th>学习式</th>
          </tr>
          <tr>
            <td>参数量</td>
            <td>0</td>
            <td>max_len × d</td>
          </tr>
          <tr>
            <td>长度泛化</td>
            <td>✓ 支持</td>
            <td>✗ 固定长度</td>
          </tr>
          <tr>
            <td>灵活性</td>
            <td>固定模式</td>
            <td>可学习</td>
          </tr>
          <tr>
            <td>代表模型</td>
            <td>原始Transformer</td>
            <td>BERT、GPT-2</td>
          </tr>
        </table>
      </div>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 播放控制 -->
    <div class="ctl-row" v-if="acts.length">
      <button @click="prev" class="btn btn-sm" :disabled="step <= 0 || playing">上一步</button>
      <button @click="toggle" class="btn btn-sm btn-blue">{{ playing ? '暂停' : '播放' }}</button>
      <button @click="nextStep" class="btn btn-sm" :disabled="step >= acts.length || playing">下一步</button>
    </div>

    <div class="prog" v-if="acts.length">{{ step }} / {{ acts.length }}</div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const encodingType = ref('sinusoidal')
const playing = ref(false)
const step = ref(0)
const msg = ref('选择编码类型，点击"开始演示"查看位置编码原理')
const acts = ref([])
let timer = null

const encodingTips = {
  sinusoidal: '正弦余弦位置编码：使用不同频率的正弦和余弦函数编码位置，无需训练',
  learned: '学习式位置编码：将位置编码作为可训练参数，通过反向传播学习',
  rope: 'RoPE旋转位置编码：通过旋转向量编码位置，现代LLM主流选择'
}

// 计算正弦余弦编码矩阵
const sinCosMatrix = computed(() => {
  const matrix = []
  const maxLen = 16
  const d = 8
  
  for (let pos = 0; pos < maxLen; pos++) {
    const row = []
    for (let i = 0; i < d; i++) {
      const theta = pos / Math.pow(10000, (2 * Math.floor(i/2)) / d)
      if (i % 2 === 0) {
        row.push(Math.sin(theta))
      } else {
        row.push(Math.cos(theta))
      }
    }
    matrix.push(row)
  }
  return matrix
})

// 模拟学习式位置编码
const learnedPE = computed(() => {
  const matrix = []
  for (let pos = 0; pos < 16; pos++) {
    const row = []
    for (let d = 0; d < 8; d++) {
      // 模拟学习得到的随机值
      row.push(Math.sin((pos + 1) * (d + 1) * 0.3) * 0.5 + (Math.random() - 0.5) * 0.2)
    }
    matrix.push(row)
  }
  return matrix
})

// 获取热力图颜色
function getHeatColor(val) {
  const normalized = (val + 1) / 2  // 从[-1,1]映射到[0,1]
  const r = Math.floor(255 * (1 - normalized))
  const b = Math.floor(255 * normalized)
  return `rgb(${r}, 100, ${b})`
}

// 生成正弦波路径
function getSinPath(d) {
  const points = []
  for (let pos = 0; pos < 16; pos++) {
    const theta = pos / Math.pow(10000, (2 * Math.floor(d/2)) / 8)
    const y = 25 - Math.sin(theta) * 20
    const x = pos * 12 + 10
    points.push(`${pos === 0 ? 'M' : 'L'} ${x} ${y}`)
  }
  return points.join(' ')
}

// 生成余弦波路径
function getCosPath(d) {
  const points = []
  for (let pos = 0; pos < 16; pos++) {
    const theta = pos / Math.pow(10000, (2 * Math.floor(d/2)) / 8)
    const y = 25 - Math.cos(theta) * 20
    const x = pos * 12 + 10
    points.push(`${pos === 0 ? 'M' : 'L'} ${x} ${y}`)
  }
  return points.join(' ')
}

// 生成动画步骤
function genAnim() {
  acts.value = []
  
  if (encodingType.value === 'sinusoidal') {
    acts.value.push({ t: 'matrix', m: '展示位置编码矩阵，每行代表一个位置的编码' })
    acts.value.push({ t: 'waves', m: '不同维度使用不同频率的正弦/余弦函数' })
    acts.value.push({ t: 'relative', m: '正弦余弦编码具有相对位置性质，模型可学习相对位置关系' })
  } else if (encodingType.value === 'rope') {
    acts.value.push({ t: 'rotate', m: 'RoPE通过旋转操作编码位置信息' })
    acts.value.push({ t: 'positions', m: '不同位置对应不同的旋转角度' })
    acts.value.push({ t: 'advantages', m: 'RoPE具有相对位置、长程外推、高效计算等优势' })
  } else if (encodingType.value === 'learned') {
    acts.value.push({ t: 'params', m: '学习式位置编码是可训练参数矩阵' })
    acts.value.push({ t: 'compare', m: '对比：学习式编码灵活但固定长度，正弦余弦可泛化' })
  }
  
  acts.value.push({ t: 'done', m: `${encodingType.value}位置编码演示完成！` })
}

// 执行步骤
function exec(a) {
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
  step.value = 0
  msg.value = '选择编码类型，点击"开始演示"查看位置编码原理'
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
  msg.value = '选择编码类型，点击"开始演示"查看位置编码原理'
}

// 重置全部
function resetAll() {
  resetAnim()
}

// 监听编码类型变化
watch(encodingType, () => {
  resetAnim()
})

onMounted(() => {
  // 初始化
})
</script>

<style scoped>
.pe-box {
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

.mode-sel {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.mode-sel select {
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #d1d5db;
}

.tip-row { margin-bottom: 12px; }
.tip { font-size: 12px; color: #6b7280; }

.vis-area {
  background: #f8fafc;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 12px;
}

/* 矩阵样式 */
.pe-matrix {
  margin-bottom: 15px;
}

.matrix-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.matrix-grid {
  display: inline-block;
  margin: 0 auto;
}

.matrix-header {
  display: flex;
  gap: 2px;
  margin-bottom: 4px;
}

.corner { width: 30px; }
.dim-label {
  width: 50px;
  text-align: center;
  font-size: 11px;
  font-weight: 600;
  color: #6b7280;
}

.matrix-row {
  display: flex;
  gap: 2px;
  margin-bottom: 2px;
}

.pos-label {
  width: 30px;
  text-align: center;
  font-size: 11px;
  font-weight: 600;
  color: #6b7280;
  background: #f1f5f9;
  border-radius: 2px;
}

.matrix-cell {
  width: 50px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 500;
  border-radius: 2px;
  color: white;
}

/* 波形样式 */
.wave-area {
  border-top: 1px dashed #d1d5db;
  padding-top: 15px;
  margin-top: 15px;
}

.wave-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.waves-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}

.wave-item {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 8px;
}

.wave-label {
  font-size: 11px;
  font-weight: 600;
  color: #6b7280;
  text-align: center;
  margin-bottom: 4px;
}

.wave-svg {
  width: 100%;
  height: 50px;
}

.wave-path {
  fill: none;
  stroke-width: 2;
}

.wave-path.sin { stroke: #3b82f6; }
.wave-path.cos { stroke: #22c55e; }

/* 相对位置 */
.relative-area {
  border-top: 1px dashed #d1d5db;
  padding-top: 15px;
  margin-top: 15px;
}

.relative-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.rel-formula {
  background: white;
  padding: 10px;
  border-radius: 6px;
  font-family: monospace;
  font-size: 12px;
  text-align: center;
  margin-bottom: 8px;
}

.rel-explain {
  font-size: 12px;
  color: #6b7280;
  text-align: center;
}

/* RoPE样式 */
.rope-intro, .learned-intro {
  margin-bottom: 15px;
  text-align: center;
}

.rope-2d {
  margin-bottom: 15px;
}

.rope-title, .pos-title, .adv-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.rope-svg {
  width: 100%;
  max-width: 300px;
  height: 200px;
  margin: 0 auto;
  display: block;
}

.axis { stroke: #e5e7eb; stroke-width: 1; }
.vec { stroke-width: 2.5; }
.vec.original { stroke: #3b82f6; }
.vec.rotated { stroke: #8b5cf6; }
.vec-base { stroke: #d1d5db; stroke-width: 1.5; }
.vec-rot { stroke: #3b82f6; stroke-width: 2; }
.point { stroke-width: 0; }
.point.original { fill: #3b82f6; }
.point.rotated { fill: #8b5cf6; }
.center { fill: #6b7280; }
.angle-arc { fill: none; stroke: #f59e0b; stroke-width: 1.5; }
.angle-text { font-size: 12px; fill: #f59e0b; }
.label { font-size: 10px; fill: #6b7280; }

.rope-formula {
  background: white;
  padding: 8px;
  border-radius: 6px;
  font-family: monospace;
  font-size: 12px;
  text-align: center;
  margin-top: 10px;
}

.rope-positions {
  border-top: 1px dashed #d1d5db;
  padding-top: 15px;
  margin-top: 15px;
}

.pos-grid {
  display: flex;
  gap: 15px;
  justify-content: center;
}

.pos-item {
  text-align: center;
}

.pos-item .pos-label {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 5px;
}

.pos-svg {
  width: 80px;
  height: 80px;
}

.pos-item .angle {
  font-size: 11px;
  color: #6b7280;
  margin-top: 4px;
}

.rope-advantages {
  border-top: 1px dashed #d1d5db;
  padding-top: 15px;
  margin-top: 15px;
}

.adv-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.adv-item {
  background: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  border: 1px solid #e5e7eb;
}

/* 学习式编码样式 */
.learned-params {
  margin-bottom: 15px;
}

.params-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.params-grid {
  display: flex;
  flex-direction: column;
  gap: 3px;
  align-items: center;
}

.param-row {
  display: flex;
  gap: 4px;
  align-items: center;
}

.param-row .pos-label {
  width: 25px;
  font-size: 10px;
  font-weight: 600;
  color: #6b7280;
}

.param-vec {
  display: flex;
  gap: 2px;
}

.param-cell {
  width: 20px;
  height: 18px;
  border-radius: 2px;
}

.learned-compare {
  border-top: 1px dashed #d1d5db;
  padding-top: 15px;
  margin-top: 15px;
}

.compare-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
  text-align: center;
}

.compare-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.compare-table th, .compare-table td {
  border: 1px solid #e5e7eb;
  padding: 8px;
  text-align: center;
}

.compare-table th {
  background: #f8fafc;
  font-weight: 600;
}

.st-bar {
  padding: 10px;
  background: #f1f5f9;
  border-radius: 6px;
  text-align: center;
  margin-bottom: 10px;
  font-size: 13px;
}

.ctl-row {
  display: flex;
  gap: 8px;
  justify-content: center;
  margin-bottom: 8px;
}

.prog { text-align: center; color: #6b7280; font-size: 12px; margin-bottom: 10px; }
</style>
