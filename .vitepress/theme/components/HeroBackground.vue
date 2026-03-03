<template>
  <div class="home-bg-animation" v-if="isHomePage">
    <!-- 星星粒子 -->
    <div class="stars-container">
      <div 
        v-for="star in stars" 
        :key="star.id"
        class="star"
        :style="star.style"
      ></div>
    </div>
    
    <!-- 流星 -->
    <div class="meteors-container">
      <div 
        v-for="meteor in meteors" 
        :key="meteor.id"
        class="meteor"
        :style="meteor.style"
      ></div>
    </div>
    
    <!-- 浮动光晕 -->
    <div class="orbs-container">
      <div class="orb orb-1"></div>
      <div class="orb orb-2"></div>
      <div class="orb orb-3"></div>
    </div>
    
    <!-- 网格背景 -->
    <div class="grid-bg"></div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useData } from 'vitepress'

const { frontmatter } = useData()

const stars = ref([])
const meteors = ref([])

// 判断是否为首页 - 使用计算属性
const isHomePage = computed(() => {
  return frontmatter.value?.layout === 'home'
})

// 生成星星
const generateStars = () => {
  const count = window.innerWidth < 768 ? 40 : 80
  const newStars = []
  
  for (let i = 0; i < count; i++) {
    newStars.push({
      id: i,
      style: {
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        width: `${Math.random() * 3 + 1}px`,
        height: `${Math.random() * 3 + 1}px`,
        animationDelay: `${Math.random() * 4}s`,
        animationDuration: `${Math.random() * 3 + 2}s`
      }
    })
  }
  
  stars.value = newStars
}

// 生成流星
const generateMeteors = () => {
  const count = 5
  const newMeteors = []
  
  for (let i = 0; i < count; i++) {
    newMeteors.push({
      id: i,
      style: {
        left: `${Math.random() * 80 + 10}%`,
        top: `${Math.random() * 30}%`,
        animationDelay: `${i * 3 + Math.random() * 2}s`,
        animationDuration: `${Math.random() * 1 + 1.5}s`
      }
    })
  }
  
  meteors.value = newMeteors
}

const handleResize = () => {
  generateStars()
}

onMounted(() => {
  generateStars()
  generateMeteors()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.home-bg-animation {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  z-index: 0;
}

/* 星星 */
.stars-container {
  position: absolute;
  inset: 0;
}

.star {
  position: absolute;
  background: rgba(139, 92, 246, 0.8);
  border-radius: 50%;
  animation: twinkle ease-in-out infinite;
  box-shadow: 0 0 4px rgba(139, 92, 246, 0.6);
}

@keyframes twinkle {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1.2); }
}

/* 流星 */
.meteors-container {
  position: absolute;
  inset: 0;
}

.meteor {
  position: absolute;
  width: 120px;
  height: 2px;
  background: linear-gradient(90deg, rgba(139, 92, 246, 0.9), rgba(236, 72, 153, 0.5), transparent);
  transform: rotate(-45deg);
  animation: meteor-fall linear infinite;
  opacity: 0;
}

.meteor::before {
  content: '';
  position: absolute;
  left: 0;
  top: -2px;
  width: 6px;
  height: 6px;
  background: #fff;
  border-radius: 50%;
  box-shadow: 0 0 10px #8b5cf6, 0 0 20px #8b5cf6, 0 0 30px #ec4899;
}

@keyframes meteor-fall {
  0% {
    opacity: 0;
    transform: rotate(-45deg) translateX(-100px);
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: rotate(-45deg) translateX(calc(100vh + 200px));
  }
}

/* 浮动光晕 */
.orbs-container {
  position: absolute;
  inset: 0;
}

.orb {
  position: absolute;
  border-radius: 50%;
  animation: float 20s ease-in-out infinite;
}

.orb-1 {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
  left: 5%;
  top: 10%;
  animation-delay: 0s;
}

.orb-2 {
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, rgba(236, 72, 153, 0.15) 0%, transparent 70%);
  right: 10%;
  bottom: 20%;
  animation-delay: -7s;
}

.orb-3 {
  width: 350px;
  height: 350px;
  background: radial-gradient(circle, rgba(6, 182, 212, 0.12) 0%, transparent 70%);
  left: 40%;
  top: 40%;
  animation-delay: -14s;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0); }
  25% { transform: translate(40px, -40px); }
  50% { transform: translate(-30px, 30px); }
  75% { transform: translate(30px, 40px); }
}

/* 网格背景 */
.grid-bg {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(rgba(139, 92, 246, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(139, 92, 246, 0.04) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: grid-move 40s linear infinite;
}

@keyframes grid-move {
  0% { background-position: 0 0; }
  100% { background-position: 60px 60px; }
}

/* 暗色模式增强 */
html.dark .star {
  background: rgba(167, 139, 250, 0.9);
  box-shadow: 0 0 6px rgba(167, 139, 250, 0.8);
}

html.dark .orb-1 {
  background: radial-gradient(circle, rgba(167, 139, 250, 0.25) 0%, transparent 70%);
}

html.dark .orb-2 {
  background: radial-gradient(circle, rgba(244, 114, 182, 0.2) 0%, transparent 70%);
}

html.dark .orb-3 {
  background: radial-gradient(circle, rgba(34, 211, 238, 0.18) 0%, transparent 70%);
}

html.dark .grid-bg {
  background-image: 
    linear-gradient(rgba(167, 139, 250, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(167, 139, 250, 0.06) 1px, transparent 1px);
}

/* 减少动画偏好 */
@media (prefers-reduced-motion: reduce) {
  .star,
  .meteor,
  .orb,
  .grid-bg {
    animation: none;
  }
  
  .star {
    opacity: 0.6;
  }
}
</style>
