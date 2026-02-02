// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'
import BackToTop from './components/BackToTop.vue'
import Demo from 'vitepress-theme-demoblock/dist/client/components/Demo.vue'
import DemoBlock from 'vitepress-theme-demoblock/dist/client/components/DemoBlock.vue'
import 'vitepress-theme-demoblock/dist/theme/styles/index.css'
import './style.css'

/** @type {import('vitepress').Theme} */
export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // 使用布局插槽添加返回顶部按钮
      'layout-bottom': () => h(BackToTop)
    })
  },
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)

    // 注册tabs插件
    enhanceAppWithTabs(ctx.app)

    // vitepress-theme-demoblock
    ctx.app.component('demo', Demo)
    ctx.app.component('DemoBlock', DemoBlock)
  }
}
