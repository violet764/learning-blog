<template>
  <div 
    class="scroll-buttons"
    :class="{ 'show': isShowButton }"
  >
    <button
      class="scroll-button"
      type="button"
      @click="scrollToTop"
      aria-label="回到顶部"
      title="回到顶部"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M12 4L12 20M12 4L6 10M12 4L18 10"
          stroke="white"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </button>

    <button
      class="scroll-button"
      type="button"
      @click="scrollToBottom"
      aria-label="跳到底部"
      title="跳到底部"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M12 20L12 4M12 20L6 14M12 20L18 14"
          stroke="white"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </button>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const isShowButton = ref(false)

// 回到顶部逻辑
const scrollToTop = () => {
  if (typeof window === 'undefined') return
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  })
}

// 跳到底部逻辑
const scrollToBottom = () => {
  if (typeof window === 'undefined') return
  const scrollHeight = document.documentElement.scrollHeight
  const maxScrollTop = Math.max(0, scrollHeight - window.innerHeight)
  window.scrollTo({
    top: maxScrollTop,
    behavior: 'smooth'
  })
}

// 滚动监听：超过100px显示按钮
const handleScroll = () => {
  if (typeof window === 'undefined') return
  const scrollTop = window.scrollY || document.documentElement.scrollTop
  isShowButton.value = scrollTop > 100
}

// 挂载/卸载时添加/移除监听（避免内存泄漏）
onMounted(() => {
  if (typeof window !== 'undefined') {
    window.addEventListener('scroll', handleScroll)
  }
})

onUnmounted(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('scroll', handleScroll)
  }
})
</script>

<style scoped>
/* 核心：固定在页面右下角，默认隐藏 */
.scroll-buttons {
  position: fixed;
  right: 24px;          /* 右侧间距（可微调） */
  bottom: 24px;         /* 底部间距（可微调） */
  z-index: 1000;        /* 确保不被遮挡 */
  display: flex;
  flex-direction: column; /* 垂直排列按钮 */
  gap: 10px;            /* 按钮之间的间距 */
  /* 默认隐藏：下移+透明，无点击事件 */
  opacity: 0;
  transform: translateY(20px);
  pointer-events: none;
  /* 平滑过渡动画（避免突兀） */
  transition: opacity 0.4s ease, transform 0.4s ease;
}

/* 滚动后显示按钮 */
.scroll-buttons.show {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

/* 按钮样式：紫色底 + 白色箭头 */
.scroll-button {
  width: 44px;
  height: 44px;
  border: none;         /* 去掉默认边框 */
  border-radius: 9999px;/* 圆形按钮 */
  background: #722ed1;  /* 紫色背景（可替换成你想要的紫色值） */
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(114, 46, 209, 0.3); /* 紫色系阴影 */
  transition: all 0.2s ease; /* hover动画 */
}

/* hover效果：浅紫色 + 轻微上浮 */
.scroll-button:hover {
  background: #9254de;  /* 浅紫色hover */
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(114, 46, 209, 0.4);
}

/* 点击时复位 */
.scroll-button:active {
  transform: translateY(0);
}

/* 键盘聚焦样式（无障碍） */
.scroll-button:focus-visible {
  outline: 2px solid #722ed1;
  outline-offset: 2px;
}

/* 移动端适配：微调间距和尺寸 */
@media (max-width: 768px) {
  .scroll-buttons {
    right: 16px;
    bottom: 16px;
    gap: 8px;
  }
  .scroll-button {
    width: 40px;
    height: 40px;
  }
}
</style>