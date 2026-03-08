<template>
  <Teleport to="body">
    <div class="cursor-effects" v-if="!isMobile">
      <!-- 光带尾迹 -->
      <svg class="trail-container" ref="trailContainer">
        <defs>
          <linearGradient id="trailGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color: rgba(139, 92, 246, 0)" />
            <stop offset="100%" style="stop-color: rgba(139, 92, 246, 0.5)" />
          </linearGradient>
        </defs>
        <path 
          v-if="trailPath" 
          :d="trailPath" 
          fill="none" 
          stroke="url(#trailGradient)" 
          stroke-width="1.5" 
          stroke-linecap="round"
          class="trail-path"
        />
      </svg>
      
      <!-- 点击涟漪 -->
      <TransitionGroup name="ripple" tag="div" class="ripples-container">
        <div 
          v-for="ripple in clickRipples" 
          :key="ripple.id"
          class="click-ripple"
          :style="ripple.style"
        ></div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'

const isMobile = ref(true)

// 光标位置
const position = reactive({ x: -100, y: -100 })

// 尾迹系统
const trailPoints = ref([])
const trailPath = ref('')
const maxTrailPoints = 15

// 点击涟漪
const clickRipples = ref([])
let rippleId = 0

// 检测移动端
const checkMobile = () => {
  isMobile.value = window.innerWidth <= 768 || 'ontouchstart' in window
}

// 鼠标移动
const handleMouseMove = (e) => {
  position.x = e.clientX
  position.y = e.clientY
  
  // 添加尾迹点
  trailPoints.value.push({ x: e.clientX, y: e.clientY, time: Date.now() })
  
  // 限制尾迹长度
  if (trailPoints.value.length > maxTrailPoints) {
    trailPoints.value.shift()
  }
  
  // 更新尾迹路径
  updateTrailPath()
}

// 更新尾迹路径
const updateTrailPath = () => {
  if (trailPoints.value.length < 2) {
    trailPath.value = ''
    return
  }
  
  const points = trailPoints.value
  let path = `M ${points[0].x} ${points[0].y}`
  
  for (let i = 1; i < points.length; i++) {
    const xc = (points[i].x + points[i - 1].x) / 2
    const yc = (points[i].y + points[i - 1].y) / 2
    path += ` Q ${points[i - 1].x} ${points[i - 1].y} ${xc} ${yc}`
  }
  
  trailPath.value = path
}

// 清理旧尾迹点
const cleanupTrail = () => {
  const now = Date.now()
  trailPoints.value = trailPoints.value.filter(p => now - p.time < 200)
  updateTrailPath()
}

// 鼠标点击 - 涟漪效果
const handleClick = (e) => {
  const id = rippleId++
  clickRipples.value.push({
    id,
    style: {
      left: `${e.clientX}px`,
      top: `${e.clientY}px`
    }
  })
  
  setTimeout(() => {
    clickRipples.value = clickRipples.value.filter(r => r.id !== id)
  }, 600)
}

let cleanupInterval = null

onMounted(() => {
  checkMobile()
  window.addEventListener('resize', checkMobile)
  
  if (!isMobile.value) {
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('click', handleClick)
    
    // 定期清理尾迹
    cleanupInterval = setInterval(cleanupTrail, 50)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', checkMobile)
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('click', handleClick)
  
  if (cleanupInterval) {
    clearInterval(cleanupInterval)
  }
})
</script>

<style scoped>
.cursor-effects {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 99999;
}

/* 光带尾迹 */
.trail-container {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 99998;
}

.trail-path {
  filter: drop-shadow(0 0 3px rgba(139, 92, 246, 0.4));
}

/* 点击涟漪 */
.ripples-container {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 99997;
}

.click-ripple {
  position: fixed;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  border: 1px solid rgba(139, 92, 246, 0.8);
  animation: ripple-expand 0.6s ease-out forwards;
}

@keyframes ripple-expand {
  0% {
    width: 10px;
    height: 10px;
    opacity: 1;
    border-width: 1px;
  }
  100% {
    width: 40px;
    height: 40px;
    opacity: 0;
    border-width: 0.5px;
  }
}

.ripple-enter-active,
.ripple-leave-active {
  transition: all 0.3s ease;
}

/* 减少动画偏好 */
@media (prefers-reduced-motion: reduce) {
  .trail-container {
    display: none;
  }
}
</style>