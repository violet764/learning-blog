<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="vp-image-zoom"
      role="dialog"
      aria-modal="true"
      @click="close"
      :style="overlayStyle"
    >
      <div class="vp-image-zoom__controls">
        <button 
          class="vp-image-zoom__control-btn" 
          type="button" 
          aria-label="放大" 
          @click.stop="zoomIn"
          :disabled="scale >= maxScale"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 5V19M5 12H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
        
        <button 
          class="vp-image-zoom__control-btn" 
          type="button" 
          aria-label="缩小" 
          @click.stop="zoomOut"
          :disabled="scale <= minScale"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M5 12H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
        
        <button 
          class="vp-image-zoom__control-btn" 
          type="button" 
          aria-label="重置" 
          @click.stop="resetZoom"
          :disabled="scale === 1"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3M12 3V8M12 3H7" 
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        
        <div class="vp-image-zoom__scale-info">
          {{ Math.round(scale * 100) }}%
        </div>
      </div>
      
      <button 
        class="vp-image-zoom__close" 
        type="button" 
        aria-label="关闭" 
        @click.stop="close"
      >
        ×
      </button>
      
      <div 
        class="vp-image-zoom__wrapper"
        :style="wrapperStyle"
        @wheel.prevent="onWheel"
        @mousedown="startDrag"
        @touchstart="startDrag"
        ref="wrapperRef"
      >
        <div 
          class="vp-image-zoom__container"
          ref="containerRef"
        >
          <img 
            class="vp-image-zoom__img" 
            :src="src" 
            :alt="alt" 
            @click.stop
            @load="onImageLoad"
            ref="imageRef"
          />
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch, computed, nextTick } from 'vue'

const open = ref(false)
const src = ref('')
const alt = ref('')
const scale = ref(1)
const position = ref({ x: 0, y: 0 })
const isDragging = ref(false)
const dragStart = ref({ x: 0, y: 0 })
const wrapperRef = ref(null)
const containerRef = ref(null)
const imageRef = ref(null)

// 缩放限制
const minScale = 0.5
const maxScale = 3
const scaleStep = 0.25

// 计算是否可以拖拽
const canDrag = computed(() => {
  if (!wrapperRef.value) return false
  const rect = wrapperRef.value.getBoundingClientRect()
  const scaledWidth = rect.width * scale.value
  const scaledHeight = rect.height * scale.value
  
  // 获取视口尺寸
  const viewportWidth = window.innerWidth - 48 // 减去 padding
  const viewportHeight = window.innerHeight - 48
  
  // 只有当缩放后的尺寸大于视口尺寸时才允许拖拽
  return scaledWidth > viewportWidth || scaledHeight > viewportHeight
})

// 计算样式
const overlayStyle = computed(() => ({
  opacity: open.value ? 1 : 0,
  pointerEvents: open.value ? 'auto' : 'none',
}))

const wrapperStyle = computed(() => ({
  transform: `translate(${position.value.x}px, ${position.value.y}px) scale(${scale.value})`,
  transformOrigin: 'center center',
  cursor: isDragging.value ? 'grabbing' : (canDrag.value ? 'grab' : 'default'),
  transition: isDragging.value ? 'none' : 'transform 0.2s ease',
}))

// 关闭功能
const close = () => {
  open.value = false
  resetZoom()
  setTimeout(() => {
    src.value = ''
    alt.value = ''
  }, 300)
}

// 键盘事件
const onKeydown = (event) => {
  if (event.key === 'Escape') close()
  if (event.key === '+' || event.key === '=') zoomIn()
  if (event.key === '-' || event.key === '_') zoomOut()
  if (event.key === '0') resetZoom()
}

// 图片加载完成
const onImageLoad = () => {
  if (imageRef.value) {
    // 确保图片在容器内正确显示
    nextTick(() => {
      adjustContainerSize()
    })
  }
}

// 调整容器大小
const adjustContainerSize = () => {
  if (!imageRef.value || !containerRef.value) return
  
  const img = imageRef.value
  const container = containerRef.value
  
  // 设置容器最小尺寸
  const minWidth = 200
  const minHeight = 150
  
  // 根据图片实际大小调整容器
  container.style.width = `${Math.max(minWidth, img.naturalWidth)}px`
  container.style.height = `${Math.max(minHeight, img.naturalHeight)}px`
  
  // 初始居中
  centerImage()
}

// 居中图片
const centerImage = () => {
  position.value = { x: 0, y: 0 }
}

// 缩放功能
const zoomIn = () => {
  if (scale.value < maxScale) {
    const newScale = Math.min(maxScale, scale.value + scaleStep)
    
    // 以当前位置为中心缩放
    const scaleRatio = newScale / scale.value
    position.value = {
      x: position.value.x * scaleRatio,
      y: position.value.y * scaleRatio
    }
    
    scale.value = newScale
  }
}

const zoomOut = () => {
  if (scale.value > minScale) {
    const newScale = Math.max(minScale, scale.value - scaleStep)
    
    // 以当前位置为中心缩放
    const scaleRatio = newScale / scale.value
    position.value = {
      x: position.value.x * scaleRatio,
      y: position.value.y * scaleRatio
    }
    
    scale.value = newScale
    
    // 如果缩小后图片完全在视口内，则居中
    if (!canDrag.value) {
      centerImage()
    }
  }
}

const resetZoom = () => {
  scale.value = 1
  centerImage()
}

// 鼠标滚轮缩放
const onWheel = (event) => {
  event.preventDefault()
  const delta = Math.sign(event.deltaY) > 0 ? -scaleStep : scaleStep
  const newScale = Math.max(minScale, Math.min(maxScale, scale.value + delta))
  
  if (newScale !== scale.value) {
    // 计算以鼠标为中心的缩放
    if (wrapperRef.value) {
      const rect = wrapperRef.value.getBoundingClientRect()
      const wrapperCenterX = rect.left + rect.width / 2
      const wrapperCenterY = rect.top + rect.height / 2
      
      // 计算鼠标相对于包装器中心的位置
      const mouseOffsetX = event.clientX - wrapperCenterX
      const mouseOffsetY = event.clientY - wrapperCenterY
      
      // 计算缩放后的位置偏移，实现以鼠标为中心缩放
      const scaleRatio = newScale / scale.value
      position.value = {
        x: (position.value.x + mouseOffsetX) * scaleRatio - mouseOffsetX,
        y: (position.value.y + mouseOffsetY) * scaleRatio - mouseOffsetY
      }
    }
    
    scale.value = newScale
    
    // 如果缩小后图片完全在视口内，则居中
    if (newScale <= 1 && !canDrag.value) {
      centerImage()
    }
  }
}

// 拖拽功能
const startDrag = (event) => {
  if (!canDrag.value) return
  
  isDragging.value = true
  const clientX = event.type.includes('touch') ? event.touches[0].clientX : event.clientX
  const clientY = event.type.includes('touch') ? event.touches[0].clientY : event.clientY
  dragStart.value = { 
    x: clientX, 
    y: clientY, 
    startPos: { ...position.value } 
  }
  
  // 添加拖拽事件监听
  document.addEventListener('mousemove', doDrag)
  document.addEventListener('touchmove', doDrag, { passive: false })
  document.addEventListener('mouseup', stopDrag)
  document.addEventListener('touchend', stopDrag)
}

const doDrag = (event) => {
  if (!isDragging.value || !canDrag.value) return
  
  event.preventDefault()
  const clientX = event.type.includes('touch') ? event.touches[0].clientX : event.clientX
  const clientY = event.type.includes('touch') ? event.touches[0].clientY : event.clientY
  
  const deltaX = clientX - dragStart.value.x
  const deltaY = clientY - dragStart.value.y
  
  // 计算边界限制
  if (wrapperRef.value) {
    const rect = wrapperRef.value.getBoundingClientRect()
    const scaledWidth = rect.width * scale.value
    const scaledHeight = rect.height * scale.value
    
    // 获取视口尺寸
    const viewportWidth = window.innerWidth - 48 // 减去 padding
    const viewportHeight = window.innerHeight - 48
    
    // 计算最大可拖动范围
    const maxOffsetX = Math.max(0, (scaledWidth - viewportWidth) / 2)
    const maxOffsetY = Math.max(0, (scaledHeight - viewportHeight) / 2)
    
    // 计算新的位置
    const newX = dragStart.value.startPos.x + deltaX
    const newY = dragStart.value.startPos.y + deltaY
    
    // 限制拖动范围
    position.value = {
      x: Math.max(-maxOffsetX, Math.min(maxOffsetX, newX)),
      y: Math.max(-maxOffsetY, Math.min(maxOffsetY, newY))
    }
  }
}

const stopDrag = () => {
  isDragging.value = false
  document.removeEventListener('mousemove', doDrag)
  document.removeEventListener('touchmove', doDrag)
  document.removeEventListener('mouseup', stopDrag)
  document.removeEventListener('touchend', stopDrag)
}

// 点击图片区域外关闭
const isDocImage = (el) => {
  if (!(el instanceof HTMLImageElement)) return false
  if (!el.closest('.vp-doc')) return false
  if (el.closest('a')) return false
  return true
}

const onClick = (event) => {
  if (typeof window === 'undefined') return
  const target = event.target
  if (!isDocImage(target)) return
  
  const img = target
  src.value = img.currentSrc || img.src || ''
  alt.value = img.alt || ''
  
  if (src.value) {
    open.value = true
  }
}

watch(open, async (val) => {
  if (typeof document === 'undefined') return
  
  document.documentElement.style.overflow = val ? 'hidden' : ''
  
  if (val) {
    await nextTick()
    centerImage()
  }
})

onMounted(() => {
  if (typeof window === 'undefined') return
  document.addEventListener('click', onClick, true)
  window.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  if (typeof window === 'undefined') return
  document.removeEventListener('click', onClick, true)
  window.removeEventListener('keydown', onKeydown)
  document.documentElement.style.overflow = ''
  stopDrag()
})
</script>

<style scoped>
.vp-image-zoom {
  position: fixed;
  inset: 0;
  z-index: 10001;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  opacity: 0;
  transition: opacity 0.3s ease;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  overflow: hidden;
  cursor: default;
}

.vp-image-zoom__wrapper {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  max-width: 1200px;
  max-height: 80vh;
  touch-action: none;
  will-change: transform;
}

.vp-image-zoom__container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  min-width: 200px;
  min-height: 150px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  overflow: hidden;
}

.vp-image-zoom__img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.vp-image-zoom__controls {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 12px;
  padding: 8px;
  z-index: 10002;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.vp-image-zoom__control-btn {
  width: 40px;
  height: 40px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.vp-image-zoom__control-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-1px);
}

.vp-image-zoom__control-btn:active:not(:disabled) {
  transform: translateY(0);
}

.vp-image-zoom__control-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.vp-image-zoom__scale-info {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 12px;
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  min-width: 60px;
}

.vp-image-zoom__close {
  position: fixed;
  top: 24px;
  right: 24px;
  width: 48px;
  height: 48px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 9999px;
  background: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  color: #fff;
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 10002;
}

.vp-image-zoom__close:hover {
  background: rgba(0, 0, 0, 0.6);
  transform: scale(1.05);
}

.vp-image-zoom__close:active {
  transform: scale(0.95);
}

/* 动画效果 */
@keyframes zoomIn {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.vp-image-zoom-enter-active .vp-image-zoom__container {
  animation: zoomIn 0.3s ease forwards;
}

/* 移动端优化 */
@media (max-width: 768px) {
  .vp-image-zoom {
    padding: 12px;
  }
  
  .vp-image-zoom__wrapper {
    max-height: 70vh;
  }
  
  .vp-image-zoom__container {
    padding: 12px;
    min-width: 150px;
    min-height: 100px;
  }
  
  .vp-image-zoom__controls {
    bottom: 16px;
    padding: 6px;
    gap: 6px;
  }
  
  .vp-image-zoom__control-btn {
    width: 36px;
    height: 36px;
    font-size: 14px;
  }
  
  .vp-image-zoom__scale-info {
    padding: 0 8px;
    font-size: 13px;
    min-width: 50px;
  }
  
  .vp-image-zoom__close {
    top: 16px;
    right: 16px;
    width: 44px;
    height: 44px;
    font-size: 24px;
  }
}

/* 小屏幕优化 */
@media (max-width: 480px) {
  .vp-image-zoom__container {
    padding: 8px;
    min-width: 120px;
    min-height: 80px;
    border-radius: 8px;
  }
  
  .vp-image-zoom__controls {
    flex-wrap: wrap;
    justify-content: center;
    max-width: 90%;
  }
}
</style>