// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'

import DefaultTheme from 'vitepress/theme'
import PageProgressBar from './components/PageProgressBar.vue'
import BackToTop from './components/BackToTop.vue'
import ReadingProgress from './components/ReadingProgress.vue'
import DocMeta from './components/DocMeta.vue'
import ArticleMeta from './components/ArticleMeta.vue' // 自定义元信息组件
import TagCloud from './components/TagCloud.vue'     // 自定义标签云组件
import ImageZoom from './components/ImageZoom.vue'
import RssReader from './components/RssReader.vue'
import HeroBackground from './components/HeroBackground.vue'
import CursorEffect from './components/CursorEffect.vue'
import BFSAnimation from './components/BFSAnimation.vue'
import BidirectionalSearch from './components/BidirectionalSearch.vue'
import AStarSearch from './components/AStarSearch.vue'
import KMPSearch from './components/KMPSearch.vue'
import DFSAnimation from './components/DFSAnimation.vue'
import DijkstraAnimation from './components/DijkstraAnimation.vue'
import TrieAnimation from './components/TrieAnimation.vue'
import BITAnimation from './components/BITAnimation.vue'
import SegmentTreeAnimation from './components/SegmentTreeAnimation.vue'
import UnionFindAnimation from './components/UnionFindAnimation.vue'
import FloydAnimation from './components/FloydAnimation.vue'
import SelfAttentionAnimation from './components/SelfAttentionAnimation.vue'
import MultiHeadAttentionAnimation from './components/MultiHeadAttentionAnimation.vue'
import PositionalEncodingAnimation from './components/PositionalEncodingAnimation.vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => [
        h(PageProgressBar),
        h(HeroBackground),
        // h(ReadingProgress)
      ],
      // 'doc-before': () => h(DocMeta),
      // 使用布局插槽添加全局组件
      'layout-bottom': () => [
        h(BackToTop), 
        h(ImageZoom),
        h(CursorEffect)
      ]
    })
  },
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)

    ctx.app.component('PageProgressBar', PageProgressBar)

    ctx.app.component('ArticleMeta', ArticleMeta)
    ctx.app.component('TagCloud', TagCloud)
    ctx.app.component('RssReader', RssReader)
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
    ctx.app.component('SelfAttentionAnimation', SelfAttentionAnimation)
    ctx.app.component('MultiHeadAttentionAnimation', MultiHeadAttentionAnimation)
    ctx.app.component('PositionalEncodingAnimation', PositionalEncodingAnimation)
    // 可以在所有页面中添加点击效果
    if (typeof window !== 'undefined') {
      window.addEventListener('DOMContentLoaded', () => {
        const cards = document.querySelectorAll('.card.clickable')
        cards.forEach(card => {
          card.addEventListener('click', function() {
            // 添加点击动画
            this.style.transform = 'scale(0.98)'
            setTimeout(() => {
              this.style.transform = ''
            }, 150)
          })
        })
      })
    }
  }
}



