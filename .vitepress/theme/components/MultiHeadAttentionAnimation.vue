<template>
  <div class="mha-box">
    <!-- 控制区 -->
    <div class="ctrl-row">
      <button @click="startAnim" class="btn btn-blue" :disabled="playing">开始演示</button>
      <button @click="resetAll" class="btn btn-gray" :disabled="playing">重置</button>
      <span class="head-sel">
        <label>头数:</label>
        <select v-model="numHeads" :disabled="playing">
          <option :value="2">2头</option>
          <option :value="4">4头</option>
          <option :value="8">8头</option>
        </select>
      </span>
    </div>

    <!-- 提示 -->
    <div class="tip-row">
      <span class="tip">多头注意力：将输入分割到多个头并行计算，每个头学习不同的注意力模式</span>
    </div>

    <!-- 主可视化区域 -->
    <div class="vis-area">
      <!-- 输入 -->
      <div class="input-area">
        <div class="input-title">输入向量 X (d=512)</div>
        <div class="input-vec">
          <span v-for="i in 8" :key="i" class="cell input-cell"></span>
          <span class="ellipsis">...</span>
        </div>
      </div>

      <!-- 分头投影 -->
      <div class="projection-area" v-if="step >= 1">
        <div class="arrow">↓ 线性投影</div>
        <div class="heads-container">
          <div v-for="h in numHeads" :key="h" :class="['head', {active: currentHead === h-1}]">
            <div class="head-label">Head {{ h }}</div>
            <div class="head-qkv">
              <div class="qkv-mini">
                <span class="lbl">Q</span>
                <div class="mini-vec">
                  <span v-for="i in 4" :key="i" class="mini-cell q-cell"></span>
                </div>
              </div>
              <div class="qkv-mini">
                <span class="lbl">K</span>
                <div class="mini-vec">
                  <span v-for="i in 4" :key="i" class="mini-cell k-cell"></span>
                </div>
              </div>
              <div class="qkv-mini">
                <span class="lbl">V</span>
                <div class="mini-vec">
                  <span v-for="i in 4" :key="i" class="mini-cell v-cell"></span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 各头注意力计算 -->
      <div class="attention-area" v-if="step >= 2">
        <div class="arrow">↓ 各头独立计算注意力</div>
        <div class="attentions-grid">
          <div v-for="h in numHeads" :key="h" :class="['attn-head', {active: currentHead === h-1}]">
            <div class="attn-label">Head {{ h }} 注意力</div>
            <div class="attn-pattern">
              <div v-for="i in 4" :key="i" class="attn-row">
                <span v-for="j in 4" :key="j" 
                      class="attn-cell" 
                      :style="{background: getAttnColor(h-1, i-1, j-1)}">
                </span>
              </div>
            </div>
            <div class="attn-desc">{{ headDescriptions[h-1] || '学习不同依赖' }}</div>
          </div>
        </div>
      </div>

      <!-- 拼接 -->
      <div class="concat-area" v-if="step >= 3">
        <div class="arrow">↓ Concat 拼接</div>
        <div class="concat-vec">
          <div v-for="h in numHeads" :key="h" :class="['concat-head', {active: currentHead === h-1}]">
            <span class="head-tag">H{{ h }}</span>
          </div>
        </div>
      </div>

      <!-- 输出投影 -->
      <div class="output-area" v-if="step >= 4">
        <div class="arrow">↓ Wᵒ 线性投影</div>
        <div class="output-vec">
          <div class="output-title">输出向量 (d=512)</div>
          <div class="output-cells">
            <span v-for="i in 8" :key="i" class="cell output-cell"></span>
            <span class="ellipsis">...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 状态 -->
    <div class="st-bar">{{ msg }}</div>

    <!-- 公式 -->
    <div class="formula-area">
      <div class="formula">
        <strong>多头注意力公式:</strong>
      </div>
      <div class="formula-text">
        MultiHead(Q,K,V) = Concat(head₁, ..., headₕ)Wᵒ
      </div>
      <div class="formula-text">
        headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
      </div>
    </div>

    <!-- 参数对比 -->
    <div class="params-table" v-if="step >= 4">
      <table>
        <tbody>
          <tr>
            <th>配置</th>
            <th>单头</th>
            <th>多头 (h={{ numHeads }})</th>
          </tr>
          <tr>
            <td>每头维度 dₖ</td>
            <td>512</td>
            <td>{{ 512/numHeads }}</td>
          </tr>
          <tr>
            <td>注意力模式</td>
            <td>1种</td>
            <td>{{ numHeads }}种</td>
          </tr>
          <tr>
            <td>表达能力</td>
            <td>有限</td>
            <td>丰富</td>
          </tr>
        </tbody>
      </table>
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
      <span><i class="d out"></i>输出</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const numHeads = ref(4)
const playing = ref(false)
const step = ref(0)
const currentHead = ref(-1)
const msg = ref('点击"开始演示"查看多头注意力机制')
const acts = ref([])
let timer = null

// 每个头的描述
const headDescriptions = ref([
  '关注局部依赖',
  '捕捉长距离关系',
  '学习语法结构',
  '建模语义关联',
  '关注位置信息',
  '捕捉层次结构',
  '学习对齐模式',
  '建模上下文'
])

// 预定义的注意力模式（简化展示）
const attnPatterns = ref([
  // Head 0: 对角线强
  [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]],
  // Head 1: 前向依赖
  [[0.7, 0.3, 0.1, 0.1], [0.2, 0.6, 0.3, 0.1], [0.1, 0.2, 0.6, 0.3], [0.1, 0.1, 0.2, 0.7]],
  // Head 2: 后向依赖
  [[0.3, 0.6, 0.2, 0.1], [0.1, 0.3, 0.5, 0.2], [0.1, 0.2, 0.3, 0.5], [0.1, 0.1, 0.2, 0.7]],
  // Head 3: 全局均匀
  [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
  // 更多头...
  [[0.5, 0.3, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1], [0.2, 0.3, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4]],
  [[0.2, 0.5, 0.2, 0.1], [0.1, 0.3, 0.4, 0.2], [0.1, 0.2, 0.4, 0.3], [0.1, 0.1, 0.3, 0.5]],
  [[0.4, 0.2, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2], [0.2, 0.2, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4]],
  [[0.3, 0.3, 0.2, 0.2], [0.2, 0.3, 0.3, 0.2], [0.2, 0.2, 0.3, 0.3], [0.3, 0.2, 0.2, 0.3]]
])

// 获取注意力颜色
function getAttnColor(head, row, col) {
  const weight = attnPatterns.value[head]?.[row]?.[col] || 0.25
  const intensity = Math.floor(weight * 255)
  return `rgb(${intensity}, ${100 + intensity/2}, ${255 - intensity/2})`
}

// 生成动画步骤
function genAnim() {
  acts.value = []
  
  acts.value.push({
    t: 'show_input',
    m: '输入向量X经过线性投影，生成每个头的Q、K、V'
  })
  
  acts.value.push({
    t: 'split_heads',
    m: `将维度d=512分割为${numHeads.value}个头，每个头维度dₖ=${512/numHeads.value}`
  })
  
  // 为每个头添加注意力计算步骤
  for (let h = 0; h < numHeads.value; h++) {
    acts.value.push({
      t: 'compute_head',
      h: h,
      m: `Head ${h+1} 独立计算注意力: ${headDescriptions.value[h] || '学习不同的注意力模式'}`
    })
  }
  
  acts.value.push({
    t: 'concat',
    m: `将${numHeads.value}个头的输出拼接，维度恢复为d=512`
  })
  
  acts.value.push({
    t: 'output',
    m: `通过Wᵒ线性投影得到最终输出，多头注意力完成！`
  })
}

// 执行步骤
function exec(a) {
  switch (a.t) {
    case 'show_input':
      currentHead.value = -1
      break
    case 'split_heads':
      break
    case 'compute_head':
      currentHead.value = a.h
      break
    case 'concat':
      currentHead.value = -1
      break
    case 'output':
      currentHead.value = -1
      break
  }
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
  currentHead.value = -1
  msg.value = '点击"开始演示"查看多头注意力机制'
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
    timer = setTimeout(autoPlay, 1200)
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
  currentHead.value = -1
  acts.value = []
  msg.value = '点击"开始演示"查看多头注意力机制'
}

// 重置全部
function resetAll() {
  resetAnim()
}

// 监听头数变化
watch(numHeads, () => {
  resetAnim()
})

onMounted(() => {
  // 初始化
})
</script>

<style scoped>
.mha-box {
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

.head-sel {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.head-sel select {
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

.input-area, .projection-area, .attention-area, .concat-area, .output-area {
  margin-bottom: 15px;
}

.input-title, .output-title {
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
  text-align: center;
}

.input-vec, .output-cells {
  display: flex;
  gap: 4px;
  justify-content: center;
  align-items: center;
}

.cell {
  width: 30px;
  height: 24px;
  border-radius: 4px;
  border: 1px solid #e5e7eb;
}

.input-cell { background: #e0e7ff; border-color: #6366f1; }
.output-cell { background: #c7d2fe; border-color: #6366f1; }
.ellipsis { color: #6b7280; margin: 0 8px; }

.arrow {
  text-align: center;
  color: #6b7280;
  font-size: 12px;
  margin: 10px 0;
}

.heads-container {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.head {
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  transition: all 0.3s;
}

.head.active {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.head-label {
  font-weight: 600;
  color: #374151;
  font-size: 12px;
  text-align: center;
  margin-bottom: 8px;
}

.head-qkv {
  display: flex;
  gap: 6px;
}

.qkv-mini {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.qkv-mini .lbl {
  font-size: 10px;
  font-weight: 600;
  color: #6b7280;
}

.mini-vec {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.mini-cell {
  width: 20px;
  height: 12px;
  border-radius: 2px;
  border: 1px solid;
}

.q-cell { background: #dbeafe; border-color: #3b82f6; }
.k-cell { background: #dcfce7; border-color: #22c55e; }
.v-cell { background: #fef3c7; border-color: #f59e0b; }

.attentions-grid {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.attn-head {
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  transition: all 0.3s;
}

.attn-head.active {
  border-color: #8b5cf6;
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
}

.attn-label {
  font-weight: 600;
  color: #374151;
  font-size: 11px;
  text-align: center;
  margin-bottom: 6px;
}

.attn-pattern {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.attn-row {
  display: flex;
  gap: 2px;
}

.attn-cell {
  width: 18px;
  height: 18px;
  border-radius: 2px;
}

.attn-desc {
  font-size: 10px;
  color: #6b7280;
  text-align: center;
  margin-top: 6px;
}

.concat-vec {
  display: flex;
  gap: 6px;
  justify-content: center;
}

.concat-head {
  background: linear-gradient(135deg, #c7d2fe, #a5b4fc);
  border: 2px solid #6366f1;
  border-radius: 6px;
  padding: 8px 12px;
  transition: all 0.3s;
}

.concat-head.active {
  background: linear-gradient(135deg, #818cf8, #6366f1);
  transform: scale(1.1);
}

.head-tag {
  font-weight: 700;
  color: #1f2937;
  font-size: 12px;
}

.concat-head.active .head-tag {
  color: white;
}

.output-vec {
  text-align: center;
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

.formula-text {
  font-size: 13px;
  color: #78716c;
  font-family: monospace;
  background: white;
  padding: 6px 10px;
  border-radius: 4px;
  margin-top: 4px;
}

.params-table {
  margin-bottom: 10px;
}

.params-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.params-table th, .params-table td {
  border: 1px solid #e5e7eb;
  padding: 8px;
  text-align: center;
}

.params-table th {
  background: #f8fafc;
  font-weight: 600;
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
.d.out { background: #c7d2fe; border: 2px solid #6366f1; }
</style>
