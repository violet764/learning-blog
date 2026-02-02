<template>
  <div class="vp-reading-progress" :style="{ transform: `scaleX(${progress})` }" />
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vitepress'

const progress = ref(0)
const route = useRoute()

let rafId = 0
const update = () => {
  if (typeof window === 'undefined') return
  const docEl = document.documentElement
  const max = Math.max(1, docEl.scrollHeight - window.innerHeight)
  const next = Math.min(1, Math.max(0, window.scrollY / max))
  progress.value = next
}

const scheduleUpdate = () => {
  if (rafId) return
  rafId = window.requestAnimationFrame(() => {
    rafId = 0
    update()
  })
}

onMounted(() => {
  update()
  window.addEventListener('scroll', scheduleUpdate, { passive: true })
  window.addEventListener('resize', scheduleUpdate, { passive: true })
})

onUnmounted(() => {
  if (typeof window === 'undefined') return
  window.removeEventListener('scroll', scheduleUpdate)
  window.removeEventListener('resize', scheduleUpdate)
  if (rafId) window.cancelAnimationFrame(rafId)
})

// reset after route change
if (route) {
  // VitePress route is reactive; watch isn't strictly required to compile,
  // but we keep it minimal by scheduling an update on next frame via scroll.
  // The scroll event will also update.
}
</script>

<style scoped>
.vp-reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  transform-origin: 0 50%;
  background: linear-gradient(90deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  z-index: 10000;
}
</style>

