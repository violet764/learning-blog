import { defineConfig } from 'vitepress'

// VitePress HTML 插件配置
export const htmlPlugin = {
  name: 'vitepress-html',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url && req.url.endsWith('.html')) {
        // 让 HTML 文件能够被 VitePress 处理
        next()
      } else {
        next()
      }
    })
  }
}