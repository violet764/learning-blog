<template>
  <div class="tag-cloud-wrapper">
    <div class="tag-list">
      <span v-for="tag in tags" :key="tag" class="tag-item">
        <a :href="`/tags/${tag}`" class="tag-link">
          #{{ tag }}
        </a>
      </span>
    </div>
  </div>
</template>

<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'

// 获取文章标签
const { frontmatter } = useData()
const tags = computed(() => frontmatter.value.tags || [])
</script>

<style scoped>
.tag-cloud-wrapper {
  margin: 1.5rem 0 3rem;
  padding: 1rem;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
}

.tag-item {
  list-style: none;
}

.tag-link {
  padding: 0.4rem 0.8rem;
  border-radius: 20px;
  font-size: 0.9rem;
  color: var(--vp-c-brand);
  background: var(--vp-c-brand-soft);
  text-decoration: none;
  transition: all 0.2s ease;
}

.tag-link:hover {
  background: var(--vp-c-brand);
  color: var(--vp-c-white);
  text-decoration: none;
}

/* 暗色主题适配 */
@media (prefers-color-scheme: dark) {
  .tag-cloud-wrapper {
    background: var(--vp-c-bg-soft-dark);
    border-color: var(--vp-c-divider-dark);
  }
  .tag-link {
    color: var(--vp-c-brand-dark);
    background: var(--vp-c-brand-soft-dark);
  }
  .tag-link:hover {
    background: var(--vp-c-brand-dark);
    color: var(--vp-c-white);
  }
}
</style>