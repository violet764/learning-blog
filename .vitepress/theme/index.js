// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'

import DefaultTheme from 'vitepress/theme'
import PageProgressBar from './components/PageProgressBar.vue'
import BackToTop from './components/BackToTop.vue'
import ReadingProgress from './components/ReadingProgress.vue'
import DocMeta from './components/DocMeta.vue'
import ImageZoom from './components/ImageZoom.vue'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => [
        h(PageProgressBar),
        // h(ReadingProgress)
      ],
      // 'doc-before': () => h(DocMeta),
      // 使用布局插槽添加返回顶部按钮
      'layout-bottom': () => [
        h(BackToTop), 
        h(ImageZoom)
      ]
    })
  },
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)

    ctx.app.component('PageProgressBar', PageProgressBar)

  }
}
