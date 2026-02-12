<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import dayjs from 'dayjs'
import feedList from '../../rss-feeds.json'

const STORAGE_KEY = 'vitepress-rss-reader-custom-feeds-v1'
const HEALTH_STORAGE_KEY = 'vitepress-rss-reader-health-v1'
const FEED_SELECTION_STORAGE_KEY = 'vitepress-rss-reader-selection-v1'

const REQUEST_TIMEOUT_MS = 12000
const MAX_RETRIES = 2
const CONCURRENCY_LIMIT = 6
const FAILURE_SKIP_THRESHOLD = 4
const FAILURE_COOLDOWN_MS = 30 * 60 * 1000
const SUMMARY_PREVIEW_CHARS = 220

const DEFAULT_PROXY_POOL = [
  'https://api.allorigins.win/raw?url={url}',
  'https://corsproxy.io/?{url}',
  'https://cors.isomorphic-git.org/{url}'
]

const items = ref([])
const loading = ref(false)
const stopping = ref(false)
const error = ref('')
const failedFeeds = ref([])
const lastUpdated = ref('')
const customFeeds = ref([])
const formError = ref('')
const formMessage = ref('')
const opmlText = ref('')
const healthMap = ref({})
const selectedFeedIds = ref([])
const filterMode = ref('all')
const rangeStart = ref('')
const rangeEnd = ref('')
const refreshAbortController = ref(null)

const newFeed = ref({
  title: '',
  url: '',
  homepage: '',
  proxy: '',
  proxyPool: ''
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

const parseProxyPoolInput = (value) => {
  const raw = String(value || '').trim()
  if (!raw) return undefined
  return raw
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
}

const normalizeFeed = (raw, index, prefix = 'feed', isCustom = false) => {
  const url = normalizeUrl(raw.url || raw.xmlUrl)
  const title = (raw.title || raw.text || '').trim() || feedHost(url) || `Feed ${index + 1}`
  const homepage = normalizeUrl(raw.homepage || raw.htmlUrl || '')
  const proxy = raw.proxy
  const proxyPool = Array.isArray(raw.proxyPool)
    ? raw.proxyPool.map(p => String(p).trim()).filter(Boolean)
    : undefined

  return {
    id: (raw.id || '').trim() || randomId(prefix),
    title,
    url,
    homepage,
    proxy,
    proxyPool,
    maxItems: Number(raw.maxItems) > 0 ? Number(raw.maxItems) : 30,
    custom: isCustom
  }
}

const builtinFeeds = feedList
  .map((feed, index) => normalizeFeed(feed, index, `builtin-${index}`))
  .filter(feed => Boolean(feed.url))

const feeds = computed(() => [...builtinFeeds, ...customFeeds.value])
const hasFeeds = computed(() => feeds.value.length > 0)
const selectedFeedSet = computed(() => new Set(selectedFeedIds.value))
const activeFeeds = computed(() => feeds.value.filter(feed => selectedFeedSet.value.has(feed.id)))
const hasActiveFeeds = computed(() => activeFeeds.value.length > 0)

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
  if (filterMode.value === 'all') {
    return items.value
  }

  if (filterMode.value === 'month') {
    const startTs = dayjs().subtract(30, 'day').startOf('day').valueOf()
    return items.value.filter(item => item.dateTs && item.dateTs >= startTs)
  }

  if (filterMode.value === 'range') {
    if (rangeError.value) return []
    const start = rangeStart.value ? dayjs(rangeStart.value).startOf('day') : null
    const end = rangeEnd.value ? dayjs(rangeEnd.value).endOf('day') : null
    const startTs = start?.isValid() ? start.valueOf() : null
    const endTs = end?.isValid() ? end.valueOf() : null

    return items.value.filter((item) => {
      if (!item.dateTs) return false
      if (startTs !== null && item.dateTs < startTs) return false
      if (endTs !== null && item.dateTs > endTs) return false
      return true
    })
  }

  return items.value
})

const unhealthyCount = computed(() =>
  feeds.value.filter(feed => {
    const record = healthMap.value[feed.id]
    return record && Number(record.consecutiveFailures || 0) > 0
  }).length
)

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
  if (Array.isArray(feed.proxyPool) && feed.proxyPool.length > 0) {
    output.proxyPool = feed.proxyPool
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
      // ignore clipboard errors
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

const saveFeedSelection = () => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(FEED_SELECTION_STORAGE_KEY, JSON.stringify(selectedFeedIds.value))
}

const loadFeedSelection = () => {
  if (typeof window === 'undefined') return []
  const cached = window.localStorage.getItem(FEED_SELECTION_STORAGE_KEY)
  if (!cached) return []

  try {
    const parsed = JSON.parse(cached)
    if (!Array.isArray(parsed)) return []
    return parsed.map(id => String(id))
  } catch {
    return []
  }
}

const reconcileFeedSelection = (selectAllWhenEmpty = false) => {
  const allIds = new Set(feeds.value.map(feed => feed.id))
  const next = selectedFeedIds.value.filter(id => allIds.has(id))
  if (selectAllWhenEmpty && next.length === 0 && feeds.value.length > 0) {
    selectedFeedIds.value = feeds.value.map(feed => feed.id)
    return
  }
  selectedFeedIds.value = next
}

const saveHealthMap = () => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(HEALTH_STORAGE_KEY, JSON.stringify(healthMap.value))
}

const loadHealthMap = () => {
  if (typeof window === 'undefined') return {}
  const cached = window.localStorage.getItem(HEALTH_STORAGE_KEY)
  if (!cached) return {}
  try {
    const parsed = JSON.parse(cached)
    if (!parsed || typeof parsed !== 'object') return {}
    return parsed
  } catch {
    return {}
  }
}

const resetHealthStatus = () => {
  healthMap.value = {}
  saveHealthMap()
  setStatus('已重置源健康状态，下次会重新尝试全部订阅源')
}

const isFeedSelected = (feedId) => selectedFeedSet.value.has(feedId)

const toggleFeedSelection = (feedId) => {
  const next = new Set(selectedFeedIds.value)
  if (next.has(feedId)) {
    next.delete(feedId)
  } else {
    next.add(feedId)
  }
  selectedFeedIds.value = [...next]
}

const selectAllFeeds = () => {
  selectedFeedIds.value = feeds.value.map(feed => feed.id)
}

const invertFeedSelection = () => {
  const current = new Set(selectedFeedIds.value)
  selectedFeedIds.value = feeds.value
    .map(feed => feed.id)
    .filter(id => !current.has(id))
}

const clearFeedSelection = () => {
  selectedFeedIds.value = []
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

const buildSummaryPreview = (value) => {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (text.length <= SUMMARY_PREVIEW_CHARS) return text
  return `${text.slice(0, SUMMARY_PREVIEW_CHARS)}...`
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
        summary: buildSummaryPreview(toPlainText(
          item.querySelector('description')?.textContent ||
          item.querySelector('content\\:encoded')?.textContent ||
          ''
        )),
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
        summary: buildSummaryPreview(toPlainText(
          entry.querySelector('summary')?.textContent ||
          entry.querySelector('content')?.textContent ||
          ''
        )),
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

const normalizeProxyPattern = (pattern, encodedUrl) => {
  const text = String(pattern || '').trim()
  if (!text) return ''
  if (text.includes('{url}')) return text.replace('{url}', encodedUrl)
  return `${text}${encodedUrl}`
}

const buildRequestCandidates = (feed) => {
  const encodedUrl = encodeURIComponent(feed.url)

  if (feed.proxy === false) {
    return [feed.url]
  }

  const candidates = []
  if (Array.isArray(feed.proxyPool) && feed.proxyPool.length > 0) {
    for (const pattern of feed.proxyPool) {
      const built = normalizeProxyPattern(pattern, encodedUrl)
      if (built) candidates.push(built)
    }
  }

  if (typeof feed.proxy === 'string' && feed.proxy.trim()) {
    const built = normalizeProxyPattern(feed.proxy, encodedUrl)
    if (built) candidates.unshift(built)
  }

  if (!candidates.length) {
    for (const pattern of DEFAULT_PROXY_POOL) {
      const built = normalizeProxyPattern(pattern, encodedUrl)
      if (built) candidates.push(built)
    }
  }

  return [...new Set(candidates)]
}

const getHealthRecord = (feedId) => {
  return healthMap.value[feedId] || {
    consecutiveFailures: 0,
    lastError: '',
    lastCheckedAt: 0,
    lastSuccessAt: 0
  }
}

const markFeedSuccess = (feedId) => {
  const next = { ...healthMap.value }
  next[feedId] = {
    consecutiveFailures: 0,
    lastError: '',
    lastCheckedAt: Date.now(),
    lastSuccessAt: Date.now()
  }
  healthMap.value = next
}

const markFeedFailure = (feedId, message) => {
  const current = getHealthRecord(feedId)
  const next = { ...healthMap.value }
  next[feedId] = {
    consecutiveFailures: Number(current.consecutiveFailures || 0) + 1,
    lastError: message,
    lastCheckedAt: Date.now(),
    lastSuccessAt: Number(current.lastSuccessAt || 0)
  }
  healthMap.value = next
}

const shouldSkipFeed = (feedId) => {
  const record = getHealthRecord(feedId)
  const failures = Number(record.consecutiveFailures || 0)
  if (failures < FAILURE_SKIP_THRESHOLD) return null

  const elapsed = Date.now() - Number(record.lastCheckedAt || 0)
  if (elapsed >= FAILURE_COOLDOWN_MS) return null

  const remainMinutes = Math.max(1, Math.ceil((FAILURE_COOLDOWN_MS - elapsed) / 60000))
  return `连续失败 ${failures} 次，暂时跳过（约 ${remainMinutes} 分钟后重试）`
}

const createCancelledError = () => {
  const err = new Error('已取消刷新')
  err.code = 'CANCELLED'
  return err
}

const isCancelledError = (err) => {
  return Boolean(err && (err.code === 'CANCELLED' || err.message === '已取消刷新'))
}

const cancelRefresh = () => {
  if (!loading.value || !refreshAbortController.value) return
  stopping.value = true
  refreshAbortController.value.abort()
  setStatus('正在停止刷新...')
}

const fetchWithTimeout = async (url, outerSignal) => {
  if (outerSignal?.aborted) {
    throw createCancelledError()
  }

  const controller = new AbortController()
  let cancelledByOuter = false
  const onOuterAbort = () => {
    cancelledByOuter = true
    controller.abort()
  }

  if (outerSignal) {
    outerSignal.addEventListener('abort', onOuterAbort, { once: true })
  }

  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const resp = await fetch(url, { signal: controller.signal })
    return resp
  } catch (err) {
    if (cancelledByOuter || outerSignal?.aborted) {
      throw createCancelledError()
    }
    if (err?.name === 'AbortError') {
      throw new Error(`请求超时 (${Math.floor(REQUEST_TIMEOUT_MS / 1000)}s)`)
    }
    throw err
  } finally {
    clearTimeout(timer)
    if (outerSignal) {
      outerSignal.removeEventListener('abort', onOuterAbort)
    }
  }
}

const fetchOneFeedWithFallback = async (feed, signal) => {
  if (signal?.aborted) {
    return { ok: false, skipped: true, cancelled: true, message: '已取消刷新', feed }
  }

  const skipReason = shouldSkipFeed(feed.id)
  if (skipReason) {
    return { ok: false, skipped: true, message: skipReason, feed }
  }

  const candidates = buildRequestCandidates(feed)
  let lastError = new Error('Failed to fetch')

  for (const candidate of candidates) {
    if (signal?.aborted) {
      return { ok: false, skipped: true, cancelled: true, message: '已取消刷新', feed }
    }

    for (let i = 0; i <= MAX_RETRIES; i += 1) {
      if (signal?.aborted) {
        return { ok: false, skipped: true, cancelled: true, message: '已取消刷新', feed }
      }

      try {
        const resp = await fetchWithTimeout(candidate, signal)
        if (!resp.ok) {
          throw new Error(`获取失败 (${resp.status})`)
        }
        const text = await resp.text()
        const parsedItems = parseFeed(text, feed).slice(0, feed.maxItems || 30)
        markFeedSuccess(feed.id)
        return { ok: true, feed, items: parsedItems }
      } catch (err) {
        if (isCancelledError(err)) {
          return { ok: false, skipped: true, cancelled: true, message: '已取消刷新', feed }
        }
        lastError = err instanceof Error ? err : new Error(String(err))
      }
    }
  }

  markFeedFailure(feed.id, lastError.message || 'Failed to fetch')
  return {
    ok: false,
    skipped: false,
    feed,
    message: lastError.message || 'Failed to fetch'
  }
}

const mapWithConcurrency = async (list, limit, worker, signal) => {
  const results = new Array(list.length)
  if (!list.length) return results

  let cursor = 0
  const workers = Array.from({ length: Math.min(limit, list.length) }, async () => {
    while (true) {
      if (signal?.aborted) return
      const index = cursor
      cursor += 1
      if (index >= list.length) return
      results[index] = await worker(list[index], index)
    }
  })

  await Promise.all(workers)
  return results
}

const loadAllFeeds = async () => {
  if (!hasFeeds.value) {
    items.value = []
    failedFeeds.value = []
    return
  }

  if (!hasActiveFeeds.value) {
    items.value = []
    failedFeeds.value = []
    error.value = '当前未选择任何站点，请先勾选站点后再刷新。'
    return
  }

  loading.value = true
  stopping.value = false
  error.value = ''
  failedFeeds.value = []

  const runController = new AbortController()
  refreshAbortController.value = runController

  try {
    const results = await mapWithConcurrency(
      activeFeeds.value,
      CONCURRENCY_LIMIT,
      async (feed) => fetchOneFeedWithFallback(feed, runController.signal),
      runController.signal
    )

    const mergedItems = []
    const failures = []
    const wasCancelled = runController.signal.aborted

    for (const result of results) {
      if (!result) continue
      if (result?.ok) {
        mergedItems.push(...result.items)
      } else if (result) {
        if (result.cancelled) continue
        failures.push({
          title: result.feed.title,
          message: result.message,
          skipped: Boolean(result.skipped)
        })
      }
    }

    items.value = dedupeAndSort(mergedItems)
    failedFeeds.value = failures
    lastUpdated.value = dayjs().format('YYYY-MM-DD HH:mm')
    saveHealthMap()

    if (wasCancelled) {
      error.value = ''
      setStatus('已停止刷新，保留当前已完成加载的内容')
    } else if (!items.value.length && failures.length) {
      error.value = '全部订阅源都加载失败，请检查代理、网络或订阅源可用性。'
    }
  } finally {
    if (refreshAbortController.value === runController) {
      refreshAbortController.value = null
    }
    stopping.value = false
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
    proxyPool: parseProxyPoolInput(newFeed.value.proxyPool)
  }, customFeeds.value.length, 'custom', true)

  customFeeds.value = [feed, ...customFeeds.value]
  selectedFeedIds.value = [feed.id, ...selectedFeedIds.value]
  saveCustomFeeds()
  setStatus(`已添加订阅：${feed.title}`)
  newFeed.value = { title: '', url: '', homepage: '', proxy: '', proxyPool: '' }
  await loadAllFeeds()
}

const removeFeed = async (feedId) => {
  customFeeds.value = customFeeds.value.filter(feed => feed.id !== feedId)
  selectedFeedIds.value = selectedFeedIds.value.filter(id => id !== feedId)
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
  selectedFeedIds.value = selectedFeedIds.value.filter(id => !id.startsWith('custom-'))
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
    selectedFeedIds.value = [
      ...deduped.map(feed => feed.id),
      ...selectedFeedIds.value
    ]
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
  selectedFeedIds.value = loadFeedSelection()
  reconcileFeedSelection(true)
  healthMap.value = loadHealthMap()
  await loadAllFeeds()
})

watch(feeds, () => {
  reconcileFeedSelection(false)
})

watch(selectedFeedIds, () => {
  saveFeedSelection()
}, { deep: true })
</script>

<template>
  <div class="rss-reader">
    <div class="rss-reader__controls">
      <div class="rss-reader__overview">
        <strong>聚合订阅流</strong>
        <span>总源数 {{ feeds.length }}</span>
        <span>已选 {{ activeFeeds.length }}</span>
        <span>异常源 {{ unhealthyCount }}</span>
        <span v-if="failedFeeds.length > 0">本次失败 {{ failedFeeds.length }} 个</span>
      </div>
      <div class="rss-reader__actions">
        <button
          class="rss-reader__button"
          type="button"
          :disabled="loading || !hasActiveFeeds"
          @click="loadAllFeeds"
        >
          {{ loading ? '刷新中...' : '刷新全部订阅' }}
        </button>
        <button
          v-if="loading"
          class="rss-reader__button rss-reader__button--muted"
          type="button"
          :disabled="stopping"
          @click="cancelRefresh"
        >
          {{ stopping ? '停止中...' : '停止刷新' }}
        </button>
        <span
          v-if="lastUpdated"
          class="rss-reader__timestamp"
        >
          最近更新：{{ lastUpdated }}
        </span>
      </div>
    </div>

    <details class="rss-reader__sources">
      <summary>站点选择与反选</summary>
      <div class="rss-reader__sources-actions">
        <button
          class="rss-reader__button rss-reader__button--small"
          type="button"
          @click="selectAllFeeds"
        >
          全选
        </button>
        <button
          class="rss-reader__button rss-reader__button--small"
          type="button"
          @click="invertFeedSelection"
        >
          反选
        </button>
        <button
          class="rss-reader__button rss-reader__button--small"
          type="button"
          @click="clearFeedSelection"
        >
          全不选
        </button>
        <span class="rss-reader__sources-meta">
          当前已选 {{ activeFeeds.length }} / {{ feeds.length }}
        </span>
      </div>

      <div class="rss-reader__sources-list">
        <label
          v-for="feed in feeds"
          :key="`feed-selector-${feed.id}`"
          class="rss-reader__source-item"
        >
          <input
            type="checkbox"
            :checked="isFeedSelected(feed.id)"
            @change="toggleFeedSelection(feed.id)"
          >
          <span>{{ feed.title }}</span>
        </label>
      </div>
    </details>

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
            placeholder="主代理（可选，支持 {url} 占位符）"
          >
          <textarea
            v-model.trim="newFeed.proxyPool"
            rows="4"
            placeholder="备用代理池（可选，每行一个，支持 {url} 占位符）"
          />
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
              class="rss-reader__button"
              type="button"
              @click="resetHealthStatus"
            >
              重置健康状态
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
            {{ failed.title }}：{{ failed.message }}<span v-if="failed.skipped">（熔断跳过）</span>
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
            <span>{{ feed.title }}</span>
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
          :class="{ 'rss-reader__chip--active': filterMode === 'all' }"
          type="button"
          @click="setFilter('all')"
        >
          全部
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
      稳定化策略：并发 {{ CONCURRENCY_LIMIT }}、超时 {{ Math.floor(REQUEST_TIMEOUT_MS / 1000) }}s、重试 {{ MAX_RETRIES }} 次、连续失败熔断。
    </p>

    <p class="rss-reader__hint">
      GitHub Pages 为静态托管。网页端新增订阅和源健康状态保存在当前浏览器 LocalStorage，不会自动同步到其他设备。
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
