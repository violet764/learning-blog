<template>
  <div class="article-meta-bar">
    <!-- 元信息列表 -->
    <div class="meta-list">
      <span class="meta-item">
        📅 发表于 {{ formatTime(publishTime) }}
      </span>
      <span class="meta-item">
        🔄 更新于 {{ formatTime(updateTime) }}
      </span>
      <span class="meta-item">
        👁️ {{ viewCount || '-' }}次访问
      </span>
      <span class="meta-item">
        📝 {{ wordCount }}字
      </span>
      <span class="meta-item">
        ⏱️ {{ readTime }}分钟
      </span>
    </div>
  </div>
</template>

<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'

// 获取文章 frontmatter 数据
const { frontmatter } = useData()

// 格式化时间（YYYY/MM/DD）
const formatTime = (timeStr) => {
  if (!timeStr) return '-'
  const date = new Date(timeStr)
  return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')}`
}

// 响应式获取元信息
const publishTime = computed(() => frontmatter.value.publishTime)
const updateTime = computed(() => frontmatter.value.updateTime)
const viewCount = computed(() => frontmatter.value.viewCount || 0)
const wordCount = computed(() => frontmatter.value.wordCount || 0)
const readTime = computed(() => frontmatter.value.readTime || 0)
</script>

<style scoped>
.article-meta-bar {
  margin: 1rem 0 2rem;
  padding: 0.8rem 1.2rem;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
}

.meta-list {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: center;
}

.meta-item {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

/* 暗色主题适配 */
@media (prefers-color-scheme: dark) {
  .article-meta-bar {
    background: var(--vp-c-bg-soft);
    border-color: var(--vp-c-divider-dark);
  }
  .meta-item {
    color: var(--vp-c-text-2-dark);
  }
}
</style>