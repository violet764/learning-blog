<template>
  <div class="vp-doc-meta" v-if="wordCount > 0">
    <span class="vp-doc-meta__item">字数：{{ wordCount }}</span>
    <span class="vp-doc-meta__dot" aria-hidden="true">·</span>
    <span class="vp-doc-meta__item">预计阅读：{{ readingMinutes }} 分钟</span>
    <template v-if="lastUpdatedText">
      <span class="vp-doc-meta__dot" aria-hidden="true">·</span>
      <span class="vp-doc-meta__item">更新：{{ lastUpdatedText }}</span>
    </template>
  </div>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useData, useRoute } from 'vitepress'

const wordCount = ref(0)
const readingMinutes = ref(1)
const lastUpdatedText = ref('')

const route = useRoute()
const { page } = useData()

const countText = (text) => {
  const cjk = (text.match(/[\u4E00-\u9FFF]/g) || []).length
  const latinWords = (text
    .replace(/[\u4E00-\u9FFF]/g, ' ')
    .trim()
    .match(/[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?/g) || []).length
  return cjk + latinWords
}

const recompute = () => {
  if (typeof window === 'undefined') return
  const el = document.querySelector('.vp-doc')
  if (!el) return
  const text = (el.textContent || '').replace(/\s+/g, ' ').trim()
  const count = countText(text)
  wordCount.value = count

  // Rough reading speed: ~300 chars/words per minute.
  readingMinutes.value = Math.max(1, Math.ceil(count / 300))
}

const updateLastUpdated = () => {
  const val = page.value?.lastUpdated
  if (!val) {
    lastUpdatedText.value = ''
    return
  }
  const date = typeof val === 'number' ? new Date(val) : new Date(String(val))
  if (Number.isNaN(date.getTime())) {
    lastUpdatedText.value = String(val)
    return
  }
  lastUpdatedText.value = date.toLocaleString()
}

onMounted(async () => {
  await nextTick()
  recompute()
  updateLastUpdated()
})

const stop = watch(
  () => route.path,
  async () => {
    await nextTick()
    // wait one more tick for content render
    await nextTick()
    recompute()
    updateLastUpdated()
  }
)

onUnmounted(() => {
  stop()
})
</script>

<style scoped>
.vp-doc-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0 16px;
  padding: 10px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1;
}

.vp-doc-meta__item {
  white-space: nowrap;
}

.vp-doc-meta__dot {
  opacity: 0.6;
}
</style>

