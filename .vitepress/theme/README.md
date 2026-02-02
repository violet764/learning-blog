# 自定义主题（.vitepress/theme）

本目录用于扩展 VitePress 默认主题（`vitepress/theme`），并为本站提供额外的交互与样式。

## 已启用的功能

### 1) 右下角滚动按钮

- 位置：`.vitepress/theme/components/BackToTop.vue`
- 注入位置：`.vitepress/theme/index.js` 的 `layout-bottom`
- 功能：回到顶部 / 跳到底部

### 2) 阅读进度条

- 位置：`.vitepress/theme/components/ReadingProgress.vue`
- 注入位置：`.vitepress/theme/index.js` 的 `layout-top`
- 功能：根据页面滚动位置显示顶部进度条

### 3) 文档信息条（字数 / 预计阅读 / 更新时间）

- 位置：`.vitepress/theme/components/DocMeta.vue`
- 注入位置：`.vitepress/theme/index.js` 的 `doc-before`
- 说明：字数与阅读时间基于渲染后的 `.vp-doc` 文本粗略统计；更新时间读取 `page.lastUpdated`

### 4) 图片点击放大（Image Zoom）

- 位置：`.vitepress/theme/components/ImageZoom.vue`
- 注入位置：`.vitepress/theme/index.js` 的 `layout-bottom`
- 交互：点击正文图片（不包含链接中的图片）打开遮罩层；点击空白或按 `Esc` 关闭

### 5) demoblock（演示块）样式与组件注册

- 主题侧注册：`.vitepress/theme/index.js`
  - 注册 `demo` 组件（用于 `:::demo` 容器渲染）
  - 引入 `vitepress-theme-demoblock` 的主题样式
- 示例页面：`notes/tools/demoblock-example.md`

## 样式说明

- 全站样式入口：`.vitepress/theme/style.css`
- 目前包含：
  - 导航栏轻微磨砂效果
  - 文档区排版优化（行高、链接、表格）
  - 图片 hover 提示可放大
  - demoblock 圆角统一

## 如何关闭某个功能

在 `.vitepress/theme/index.js` 中移除对应组件的导入与插槽注入即可，例如：

- 移除阅读进度：删除 `ReadingProgress` 的 import 和 `layout-top` 插槽
- 移除图片放大：删除 `ImageZoom` 的 import 和 `layout-bottom` 中的挂载

