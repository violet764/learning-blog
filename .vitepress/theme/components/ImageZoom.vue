<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="vp-image-zoom"
      role="dialog"
      aria-modal="true"
      @click="close"
    >
      <button class="vp-image-zoom__close" type="button" aria-label="关闭" @click.stop="close">
        ×
      </button>
      <img class="vp-image-zoom__img" :src="src" :alt="alt" @click.stop />
    </div>
  </Teleport>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'

const open = ref(false)
const src = ref('')
const alt = ref('')

const close = () => {
  open.value = false
  src.value = ''
  alt.value = ''
}

const onKeydown = (event) => {
  if (event.key === 'Escape') close()
}

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
  if (src.value) open.value = true
}

watch(open, (val) => {
  if (typeof document === 'undefined') return
  document.documentElement.style.overflow = val ? 'hidden' : ''
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
}

.vp-image-zoom__img {
  max-width: min(1200px, 100%);
  max-height: calc(100vh - 48px);
  border-radius: 12px;
  box-shadow: 0 16px 60px rgba(0, 0, 0, 0.4);
  background: var(--vp-c-bg);
}

.vp-image-zoom__close {
  position: fixed;
  top: 16px;
  right: 16px;
  width: 40px;
  height: 40px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 9999px;
  background: rgba(0, 0, 0, 0.35);
  color: #fff;
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
}

.vp-image-zoom__close:hover {
  background: rgba(0, 0, 0, 0.55);
}
</style>

