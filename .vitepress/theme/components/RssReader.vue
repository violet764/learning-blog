<script setup>
import { computed, onMounted, ref } from 'vue'
import dayjs from 'dayjs'
import feedList from '../../rss-feeds.json'

const DEFAULT_PROXY = 'https://api.allorigins.win/raw?url='
const STORAGE_KEY = 'vitepress-rss-reader-custom-feeds-v1'

const items = ref([])
const loading = ref(false)
const error = ref('')
const failedFeeds = ref([])
const lastUpdated = ref('')
const customFeeds = ref([])
const formError = ref('')
const formMessage = ref('')
const opmlText = ref('')
const filterMode = ref('all')
const rangeStart = ref('')
const rangeEnd = ref('')
const categoryFilter = ref('all') // 分类筛选：all, domestic, international

const newFeed = ref({
  title: '',
  url: '',
  homepage: '',
  proxy: '',
  category: 'domestic' // 默认国内
})

const randomId = (prefix) => `${prefix}-${Math.random().toString(36).slice(2, 10)}`
const normalizeUrl = (url) => (url || '').trim()

const feedHost = (url) => {
  try {
    return new URL(url).host
  } catch {
    return ''
  }
}

const normalizeFeed = (raw, index, prefix = 'feed', isCustom = false) => {
  const url = normalizeUrl(raw.url || raw.xmlUrl)
  const title = (raw.title || raw.text || '').trim() || feedHost(url) || `Feed ${index + 1}`
  const homepage = normalizeUrl(raw.homepage || raw.htmlUrl || '')
  const proxy = raw.proxy
  const category = raw.category || 'domestic' // 默认国内

  return {
    id: (raw.id || '').trim() || randomId(prefix),
    title,
    url,
    homepage,
    proxy,
    category,
    maxItems: Number(raw.maxItems) > 0 ? Number(raw.maxItems) : 30,
    custom: isCustom
  }
}

const builtinFeeds = feedList
  .map((feed, index) => normalizeFeed(feed, index, `builtin-${index}`))
  .filter(feed => Boolean(feed.url))

const feeds = computed(() => [...builtinFeeds, ...customFeeds.value])
const hasFeeds = computed(() => feeds.value.length > 0)

const rangeError = computed(() => {
  if (filterMode.value !== 'range') return ''
  if (!rangeStart.value || !rangeEnd.value) return ''

  const start = dayjs(rangeStart.value).startOf('day')
  const end = dayjs(rangeEnd.value).endOf('day')
  if (!start.isValid() || !end.isValid()) return ''
  if (start.valueOf() > end.valueOf()) return '结束日期不能早于开始日期'
  return ''
})

const filteredItems = computed(() => {
  let filtered = items.value

  // 首先按分类筛选
  if (categoryFilter.value !== 'all') {
    filtered = filtered.filter(item => {
      // 找到对应的订阅源来获取分类信息
      const feed = feeds.value.find(f => f.title === item.sourceTitle)
      return feed && feed.category === categoryFilter.value
    })
  }

  // 再按时间筛选
  if (filterMode.value === 'month') {
    const startTs = dayjs().subtract(30, 'day').startOf('day').valueOf()
    filtered = filtered.filter(item => item.dateTs && item.dateTs >= startTs)
  }
  else if (filterMode.value === 'range') {
    if (rangeError.value) return []

    const start = rangeStart.value ? dayjs(rangeStart.value).startOf('day') : null
    const end = rangeEnd.value ? dayjs(rangeEnd.value).endOf('day') : null
    const startTs = start?.isValid() ? start.valueOf() : null
    const endTs = end?.isValid() ? end.valueOf() : null

    filtered = filtered.filter((item) => {
      if (!item.dateTs) return false
      if (startTs !== null && item.dateTs < startTs) return false
      if (endTs !== null && item.dateTs > endTs) return false
      return true
    })
  }

  return filtered
})

const setStatus = (message = '', isError = false) => {
  if (isError) {
    formError.value = message
    formMessage.value = ''
    return
  }
  formError.value = ''
  formMessage.value = message
}

const toConfigFeed = (feed) => {
  const output = {
    id: feed.id,
    title: feed.title,
    url: feed.url
  }
  if (feed.homepage) output.homepage = feed.homepage
  if (feed.proxy !== undefined && feed.proxy !== null && `${feed.proxy}`.trim() !== '') {
    output.proxy = feed.proxy
  }
  if (feed.maxItems && feed.maxItems !== 30) {
    output.maxItems = feed.maxItems
  }
  return output
}

const triggerDownload = (filename, content) => {
  if (typeof window === 'undefined') return
  const blob = new Blob([content], { type: 'application/json;charset=utf-8' })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}

const exportFeedsAsJson = async () => {
  const json = `${JSON.stringify(feeds.value.map(toConfigFeed), null, 2)}\n`
  triggerDownload('rss-feeds.export.json', json)

  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(json)
      setStatus('已导出 JSON，并复制到剪贴板，可直接覆盖 .vitepress/rss-feeds.json')
      return
    } catch {
      // Ignore clipboard errors and keep download-only success.
    }
  }

  setStatus('已导出 JSON 文件：rss-feeds.export.json')
}

const saveCustomFeeds = () => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(customFeeds.value))
}

const loadCustomFeeds = () => {
  if (typeof window === 'undefined') return []
  const cached = window.localStorage.getItem(STORAGE_KEY)
  if (!cached) return []

  try {
    const parsed = JSON.parse(cached)
    if (!Array.isArray(parsed)) return []
    return parsed
      .map((feed, index) => normalizeFeed(feed, index, 'custom', true))
      .filter(feed => Boolean(feed.url))
  } catch {
    return []
  }
}

const fallbackId = () => Math.random().toString(36).slice(2)

const parseDateTs = (value) => {
  if (!value) return 0
  const parsed = dayjs(value)
  return parsed.isValid() ? parsed.valueOf() : 0
}

const formatDate = (timestamp) => {
  if (!timestamp) return ''
  return dayjs(timestamp).format('YYYY-MM-DD HH:mm')
}

const toPlainText = (html) => {
  if (!html) return ''
  const doc = new DOMParser().parseFromString(html, 'text/html')
  return doc.body.textContent?.replace(/\s+/g, ' ').trim() ?? ''
}

const parseFeed = (xmlText, feed) => {
  const doc = new DOMParser().parseFromString(xmlText, 'application/xml')
  if (doc.querySelector('parsererror')) {
    throw new Error('RSS 解析失败，请检查订阅源 URL 或代理')
  }

  const rssItems = Array.from(doc.querySelectorAll('item'))
  if (rssItems.length) {
    return rssItems.map(item => {
      const dateRaw =
        item.querySelector('pubDate')?.textContent ||
        item.querySelector('dc\\:date')?.textContent ||
        ''
      const dateTs = parseDateTs(dateRaw)
      return {
        title: item.querySelector('title')?.textContent?.trim() || '无标题',
        link: item.querySelector('link')?.textContent?.trim() || '#',
        date: formatDate(dateTs),
        dateTs,
        summary: toPlainText(
          item.querySelector('description')?.textContent ||
          item.querySelector('content\\:encoded')?.textContent ||
          ''
        ),
        guid: item.querySelector('guid')?.textContent?.trim() ||
          item.querySelector('link')?.textContent?.trim() ||
          fallbackId(),
        sourceTitle: feed.title,
        sourceHomepage: feed.homepage
      }
    })
  }

  const atomEntries = Array.from(doc.querySelectorAll('entry'))
  if (atomEntries.length) {
    return atomEntries.map(entry => {
      const preferredLink = entry.querySelector('link[rel="alternate"]') || entry.querySelector('link')
      const dateRaw =
        entry.querySelector('updated')?.textContent ||
        entry.querySelector('published')?.textContent ||
        ''
      const dateTs = parseDateTs(dateRaw)
      return {
        title: entry.querySelector('title')?.textContent?.trim() || '无标题',
        link: preferredLink?.getAttribute('href')?.trim() || '#',
        date: formatDate(dateTs),
        dateTs,
        summary: toPlainText(
          entry.querySelector('summary')?.textContent ||
          entry.querySelector('content')?.textContent ||
          ''
        ),
        guid: entry.querySelector('id')?.textContent?.trim() ||
          preferredLink?.getAttribute('href')?.trim() ||
          fallbackId(),
        sourceTitle: feed.title,
        sourceHomepage: feed.homepage
      }
    })
  }

  return []
}

const buildRequestUrl = (feed) => {
  if (!feed) return ''
  if (feed.proxy === false) return feed.url

  const proxyText = typeof feed.proxy === 'string' ? feed.proxy.trim() : ''
  const proxyPrefix = proxyText || DEFAULT_PROXY
  if (proxyPrefix.includes('{url}')) {
    return proxyPrefix.replace('{url}', encodeURIComponent(feed.url))
  }
  return `${proxyPrefix}${encodeURIComponent(feed.url)}`
}

const dedupeAndSort = (rawItems) => {
  const seen = new Set()
  const merged = []

  for (const item of rawItems) {
    const key = `${item.link}|${item.guid}|${item.title}`
    if (seen.has(key)) continue
    seen.add(key)
    merged.push(item)
  }

  merged.sort((a, b) => b.dateTs - a.dateTs)
  return merged
}

const fetchOneFeed = async (feed) => {
  const requestUrl = buildRequestUrl(feed)
  const resp = await fetch(requestUrl)
  if (!resp.ok) throw new Error(`获取失败 (${resp.status})`)
  const text = await resp.text()
  const parsedItems = parseFeed(text, feed).slice(0, feed.maxItems || 30)
  return parsedItems
}

const loadAllFeeds = async () => {
  if (!hasFeeds.value) {
    items.value = []
    failedFeeds.value = []
    return
  }

  loading.value = true
  error.value = ''
  failedFeeds.value = []

  try {
    const jobs = feeds.value.map(async (feed) => {
      try {
        const feedItems = await fetchOneFeed(feed)
        return { feed, ok: true, items: feedItems }
      } catch (err) {
        return {
          feed,
          ok: false,
          message: err?.message || '加载失败'
        }
      }
    })

    const results = await Promise.all(jobs)
    const mergedItems = []
    const failures = []

    for (const result of results) {
      if (result.ok) {
        mergedItems.push(...result.items)
      } else {
        failures.push({
          title: result.feed.title,
          message: result.message
        })
      }
    }

    items.value = dedupeAndSort(mergedItems)
    failedFeeds.value = failures
    lastUpdated.value = dayjs().format('YYYY-MM-DD HH:mm')

    if (!items.value.length && failures.length) {
      error.value = '全部订阅源都加载失败，请检查代理或订阅源可用性。'
    }
  } finally {
    loading.value = false
  }
}

const setFilter = (mode) => {
  filterMode.value = mode
}

const addFeed = async () => {
  const url = normalizeUrl(newFeed.value.url)
  if (!/^https?:\/\//i.test(url)) {
    setStatus('RSS 地址必须以 http:// 或 https:// 开头', true)
    return
  }

  const duplicate = feeds.value.some(feed => feed.url.toLowerCase() === url.toLowerCase())
  if (duplicate) {
    setStatus('该订阅源已存在，无需重复添加', true)
    return
  }

  const feed = normalizeFeed({
    id: randomId('custom'),
    title: newFeed.value.title,
    url,
    homepage: newFeed.value.homepage,
    proxy: newFeed.value.proxy.trim() ? newFeed.value.proxy.trim() : undefined,
    category: newFeed.value.category
  }, customFeeds.value.length, 'custom', true)

  customFeeds.value = [feed, ...customFeeds.value]
  saveCustomFeeds()
  setStatus(`已添加订阅：${feed.title}（${feed.category === 'domestic' ? '国内' : '国外'}）`)
  newFeed.value = { title: '', url: '', homepage: '', proxy: '', category: 'domestic' }
  await loadAllFeeds()
}

const removeFeed = async (feedId) => {
  customFeeds.value = customFeeds.value.filter(feed => feed.id !== feedId)
  saveCustomFeeds()
  setStatus('已删除该自定义订阅')
  await loadAllFeeds()
}

const clearCustomFeeds = async () => {
  if (typeof window !== 'undefined') {
    const confirmed = window.confirm('确认清空所有自定义订阅源吗？')
    if (!confirmed) return
  }

  customFeeds.value = []
  saveCustomFeeds()
  setStatus('已清空所有自定义订阅')
  await loadAllFeeds()
}

const parseOpml = (source) => {
  const doc = new DOMParser().parseFromString(source, 'application/xml')
  if (doc.querySelector('parsererror')) {
    throw new Error('OPML 格式不正确，请检查内容')
  }

  const outlines = Array.from(doc.querySelectorAll('outline[xmlUrl]'))
  return outlines.map((node, index) => normalizeFeed({
    id: randomId('custom'),
    title: node.getAttribute('title') || node.getAttribute('text'),
    url: node.getAttribute('xmlUrl'),
    homepage: node.getAttribute('htmlUrl')
  }, index, 'custom', true))
}

const importOpml = async () => {
  if (!opmlText.value.trim()) {
    setStatus('请先粘贴 OPML 内容', true)
    return
  }

  try {
    const imported = parseOpml(opmlText.value).filter(feed => Boolean(feed.url))
    if (!imported.length) {
      setStatus('没有发现可导入的订阅源（需包含 xmlUrl）', true)
      return
    }

    const existing = new Set(feeds.value.map(feed => feed.url.toLowerCase()))
    const deduped = imported.filter(feed => !existing.has(feed.url.toLowerCase()))

    if (!deduped.length) {
      setStatus('导入完成：全部是重复订阅，未新增')
      return
    }

    customFeeds.value = [...deduped, ...customFeeds.value]
    saveCustomFeeds()
    setStatus(`导入完成：新增 ${deduped.length} 个订阅`)
    await loadAllFeeds()
  } catch (err) {
    setStatus(err?.message || '导入失败，请稍后重试', true)
  }
}

const handleOpmlFile = async (event) => {
  const file = event.target.files?.[0]
  if (!file) return
  opmlText.value = await file.text()
  setStatus(`已读取文件：${file.name}，点击“导入 OPML”开始导入`)
}

onMounted(async () => {
  customFeeds.value = loadCustomFeeds()
  await loadAllFeeds()
})
</script>

<template>
  <div class="rss-reader">
    <div class="rss-reader__controls">
      <div class="rss-reader__overview">
        <strong>聚合订阅流</strong>
        <span>共 {{ feeds.length }} 个订阅源</span>
        <span v-if="failedFeeds.length > 0">本次失败 {{ failedFeeds.length }} 个</span>
      </div>
      <div class="rss-reader__actions">
        <button
          class="rss-reader__button"
          type="button"
          :disabled="loading || !hasFeeds"
          @click="loadAllFeeds"
        >
          {{ loading ? '刷新中...' : '刷新全部订阅' }}
        </button>
        <span
          v-if="lastUpdated"
          class="rss-reader__timestamp"
        >
          最近更新：{{ lastUpdated }}
        </span>
      </div>
    </div>

    <details class="rss-reader__manager">
      <summary>管理订阅源（支持手动添加 / OPML 导入）</summary>

      <div class="rss-reader__manager-grid">
        <div class="rss-reader__panel">
          <h3>手动添加</h3>
          <input
            v-model.trim="newFeed.title"
            type="text"
            placeholder="标题（可选）"
          >
          <input
            v-model.trim="newFeed.url"
            type="text"
            placeholder="RSS URL（必填）"
          >
          <input
            v-model.trim="newFeed.homepage"
            type="text"
            placeholder="主页 URL（可选）"
          >
          <input
            v-model.trim="newFeed.proxy"
            type="text"
            placeholder="代理前缀（可选，默认 allorigins）"
          >
          <div class="rss-reader__category-selector">
            <label>分类：</label>
            <select v-model="newFeed.category">
              <option value="domestic">国内</option>
              <option value="international">国外</option>
            </select>
          </div>
          <button
            class="rss-reader__button"
            type="button"
            @click="addFeed"
          >
            添加订阅
          </button>
        </div>

        <div class="rss-reader__panel">
          <h3>OPML 导入</h3>
          <textarea
            v-model="opmlText"
            rows="8"
            placeholder="粘贴 OPML 内容，包含 outline 的 xmlUrl 属性"
          />
          <input
            type="file"
            accept=".opml,.xml,text/xml"
            @change="handleOpmlFile"
          >
          <div class="rss-reader__row">
            <button
              class="rss-reader__button"
              type="button"
              @click="importOpml"
            >
              导入 OPML
            </button>
            <button
              class="rss-reader__button"
              type="button"
              :disabled="feeds.length === 0"
              @click="exportFeedsAsJson"
            >
              导出 JSON
            </button>
            <button
              class="rss-reader__button rss-reader__button--danger"
              type="button"
              :disabled="customFeeds.length === 0"
              @click="clearCustomFeeds"
            >
              清空自定义
            </button>
          </div>
        </div>
      </div>

      <div v-if="formMessage" class="rss-reader__status">{{ formMessage }}</div>
      <div v-if="formError" class="rss-reader__status rss-reader__status--error">{{ formError }}</div>

      <div
        v-if="failedFeeds.length > 0"
        class="rss-reader__failed"
      >
        <h3>本次加载失败</h3>
        <ul>
          <li
            v-for="failed in failedFeeds"
            :key="failed.title"
          >
            {{ failed.title }}：{{ failed.message }}
          </li>
        </ul>
      </div>

      <div
        v-if="customFeeds.length > 0"
        class="rss-reader__custom-list"
      >
        <h3>自定义订阅（{{ customFeeds.length }}）</h3>
        <ul>
          <li
            v-for="feed in customFeeds"
            :key="feed.id"
          >
            <span class="rss-reader__feed-info">
              <span class="rss-reader__feed-title">{{ feed.title }}</span>
              <span class="rss-reader__category-tag" :class="`rss-reader__category-tag--${feed.category}`">
                {{ feed.category === 'domestic' ? '国内' : '国外' }}
              </span>
            </span>
            <button
              type="button"
              class="rss-reader__remove"
              @click="removeFeed(feed.id)"
            >
              删除
            </button>
          </li>
        </ul>
      </div>
    </details>

    <div class="rss-reader__filters">
      <div class="rss-reader__filter-tabs">
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': categoryFilter === 'all' }"
          type="button"
          @click="categoryFilter = 'all'"
        >
          全部分类
        </button>
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': categoryFilter === 'domestic' }"
          type="button"
          @click="categoryFilter = 'domestic'"
        >
          国内
        </button>
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': categoryFilter === 'international' }"
          type="button"
          @click="categoryFilter = 'international'"
        >
          国外
        </button>
      </div>
      
      <div class="rss-reader__filter-tabs">
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': filterMode === 'all' }"
          type="button"
          @click="setFilter('all')"
        >
          全部时间
        </button>
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': filterMode === 'month' }"
          type="button"
          @click="setFilter('month')"
        >
          最近 30 天
        </button>
        <button
          class="rss-reader__chip"
          :class="{ 'rss-reader__chip--active': filterMode === 'range' }"
          type="button"
          @click="setFilter('range')"
        >
          自定义时间段
        </button>
      </div>

      <div
        v-if="filterMode === 'range'"
        class="rss-reader__range"
      >
        <label>
          开始日期
          <input
            v-model="rangeStart"
            type="date"
          >
        </label>
        <label>
          结束日期
          <input
            v-model="rangeEnd"
            type="date"
          >
        </label>
      </div>

      <div class="rss-reader__filter-meta">
        当前显示 {{ filteredItems.length }} / {{ items.length }} 篇文章
      </div>
      <div
        v-if="rangeError"
        class="rss-reader__status rss-reader__status--error"
      >
        {{ rangeError }}
      </div>
    </div>

    <p class="rss-reader__hint">
      GitHub Pages 为静态托管。网页端新增的订阅保存在当前浏览器 LocalStorage，不会自动同步到其他设备。
    </p>

    <div
      v-if="!hasFeeds"
      class="rss-reader__state"
    >
      暂无订阅源，请编辑 <code>.vitepress/rss-feeds.json</code> 或在上方管理器中新增。
    </div>

    <div
      v-else-if="error"
      class="rss-reader__state rss-reader__state--error"
    >
      {{ error }}
    </div>

    <div
      v-else-if="loading"
      class="rss-reader__state"
    >
      正在刷新全部订阅源，请稍候...
    </div>

    <div
      v-else-if="filteredItems.length === 0"
      class="rss-reader__state"
    >
      当前筛选条件下暂无可展示文章。
    </div>

    <div
      v-else
      class="rss-reader__list"
    >
      <article
        v-for="item in filteredItems"
        :key="`${item.sourceTitle}-${item.guid}`"
        class="rss-reader__item"
      >
        <a
          class="rss-reader__title"
          :href="item.link"
          target="_blank"
          rel="noopener noreferrer"
        >
          {{ item.title }}
        </a>
        <div class="rss-reader__meta">
          <span v-if="item.date">{{ item.date }}</span>
          <span>• {{ item.sourceTitle }}</span>
          <a
            v-if="item.sourceHomepage"
            class="rss-reader__origin"
            :href="item.sourceHomepage"
            target="_blank"
            rel="noopener noreferrer"
          >
            源站
          </a>
        </div>
        <p
          v-if="item.summary"
          class="rss-reader__summary"
        >
          {{ item.summary }}
        </p>
      </article>
    </div>
  </div>
</template>

<style scoped>
.rss-reader__category-selector {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
}

.rss-reader__category-selector label {
  font-size: 0.9em;
  color: #666;
}

.rss-reader__category-selector select {
  padding: 4px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.rss-reader__feed-info {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.rss-reader__feed-title {
  flex: 1;
}

.rss-reader__category-tag {
  font-size: 0.8em;
  padding: 2px 6px;
  border-radius: 12px;
  background: #e0e0e0;
  color: #333;
  white-space: nowrap;
}

.rss-reader__category-tag--domestic {
  background: #e3f2fd;
  color: #1976d2;
}

.rss-reader__category-tag--international {
  background: #f3e5f5;
  color: #7b1fa2;
}
</style>
