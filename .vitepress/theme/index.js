// https://vitepress.dev/guide/custom-theme
import { h, defineAsyncComponent } from 'vue'

import DefaultTheme from 'vitepress/theme'

// 全局使用的组件（同步加载）
import PageProgressBar from './components/PageProgressBar.vue'
import BackToTop from './components/BackToTop.vue'
import ImageZoom from './components/ImageZoom.vue'
import HeroBackground from './components/HeroBackground.vue'
import CursorEffect from './components/CursorEffect.vue'

// 按需加载的组件（异步加载，减少初始包体积）
const ArticleMeta = defineAsyncComponent(() => import('./components/ArticleMeta.vue'))
const TagCloud = defineAsyncComponent(() => import('./components/TagCloud.vue'))
const RssReader = defineAsyncComponent(() => import('./components/RssReader.vue'))

// 算法动画组件（仅在特定页面使用，异步加载）
const BFSAnimation = defineAsyncComponent(() => import('./components/BFSAnimation.vue'))
const BidirectionalSearch = defineAsyncComponent(() => import('./components/BidirectionalSearch.vue'))
const AStarSearch = defineAsyncComponent(() => import('./components/AStarSearch.vue'))
const KMPSearch = defineAsyncComponent(() => import('./components/KMPSearch.vue'))
const DFSAnimation = defineAsyncComponent(() => import('./components/DFSAnimation.vue'))
const DijkstraAnimation = defineAsyncComponent(() => import('./components/DijkstraAnimation.vue'))
const TrieAnimation = defineAsyncComponent(() => import('./components/TrieAnimation.vue'))
const BITAnimation = defineAsyncComponent(() => import('./components/BITAnimation.vue'))
const SegmentTreeAnimation = defineAsyncComponent(() => import('./components/SegmentTreeAnimation.vue'))
const UnionFindAnimation = defineAsyncComponent(() => import('./components/UnionFindAnimation.vue'))
const FloydAnimation = defineAsyncComponent(() => import('./components/FloydAnimation.vue'))

// Transformer 动画组件（仅在特定页面使用，异步加载）
const SelfAttentionAnimation = defineAsyncComponent(() => import('./components/SelfAttentionAnimation.vue'))
const MultiHeadAttentionAnimation = defineAsyncComponent(() => import('./components/MultiHeadAttentionAnimation.vue'))
const PositionalEncodingAnimation = defineAsyncComponent(() => import('./components/PositionalEncodingAnimation.vue'))

import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => [
        h(PageProgressBar),
        h(HeroBackground)
      ],
      'layout-bottom': () => [
        h(BackToTop), 
        h(ImageZoom),
        h(CursorEffect)
      ]
    })
  },
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)

    // 全局注册基础组件
    ctx.app.component('PageProgressBar', PageProgressBar)

    // 异步注册其他组件
    ctx.app.component('ArticleMeta', ArticleMeta)
    ctx.app.component('TagCloud', TagCloud)
    ctx.app.component('RssReader', RssReader)
    
    // 算法动画组件
    ctx.app.component('BFSAnimation', BFSAnimation)
    ctx.app.component('BidirectionalSearch', BidirectionalSearch)
    ctx.app.component('AStarSearch', AStarSearch)
    ctx.app.component('KMPSearch', KMPSearch)
    ctx.app.component('DFSAnimation', DFSAnimation)
    ctx.app.component('DijkstraAnimation', DijkstraAnimation)
    ctx.app.component('TrieAnimation', TrieAnimation)
    ctx.app.component('BITAnimation', BITAnimation)
    ctx.app.component('SegmentTreeAnimation', SegmentTreeAnimation)
    ctx.app.component('UnionFindAnimation', UnionFindAnimation)
    ctx.app.component('FloydAnimation', FloydAnimation)
    
    // Transformer 动画组件
    ctx.app.component('SelfAttentionAnimation', SelfAttentionAnimation)
    ctx.app.component('MultiHeadAttentionAnimation', MultiHeadAttentionAnimation)
    ctx.app.component('PositionalEncodingAnimation', PositionalEncodingAnimation)
  }
}