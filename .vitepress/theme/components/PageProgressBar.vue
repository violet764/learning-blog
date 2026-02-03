<!-- components/PageProgressBar.vue -->
<template>
  <div v-if="showBar" class="progress-bar" :style="{ width: `${progress}%` }"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const progress = ref(0)
const showBar = ref(false)

const updateProgress = () => {
  const scrollY = window.scrollY
  const winHeight = document.documentElement.scrollHeight - window.innerHeight
  progress.value = (scrollY / winHeight) * 100
}

onMounted(() => {
  window.addEventListener('scroll', updateProgress)
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateProgress)
})
</script>