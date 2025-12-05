import { defineConfig } from 'vitepress'
import { set_sidebar } from '../.vitepress/utils/auto_sidebar.mjs';
// 知行代码集 - Python/C++/AI模型/深度学习(PyTorch) 专属配置
export default defineConfig({
  base:"/",
  head:[["link",{rel:"icon",href:"/spaceship.svg"}]],
  // 站点核心信息（与首页标题/定位一致）
  title: "知行代码集",
  description: "深耕Python/C++，拆解AI模型与深度学习(PyTorch)底层逻辑 | 个人技术学习笔记",
  themeConfig: {
    // sidebar:false, //关闭侧边栏
    aside:'right', //侧边栏位置
    outlineTitle:'目录',
    outline: [2,6],
    // 顶部导航栏
    logo: '/logo.svg',
    nav: [
      { text: '首页', link: '/' },
      // 编程语言（下拉菜单，包含Python/C++）
      
      { 
        text: '编程语言', 
        items: [
          { text: 'Python', link: '/notes/language/python/python.md' },
          { text: 'C++', link: '/notes/language/c++.md' },
          { text: '跨语言实践', link: '/notes/language/cross-lang.md' }
        ] 
      },
      // AI模型（独立入口，对齐首页Feature）
      { text: 'AI模型', link: '/notes/ai-model/' },
      // 深度学习
      { 
        text: '深度学习', 
        items: [
          { text: 'PyTorch核心教程', link: '/notes/deep-learning/pytorch.md' },
          { text: '实战项目', link: '/notes/deep-learning/project.md' },
          { text: '性能优化', link: '/notes/deep-learning/optimize.md' }
        ] 
      },
      // 算法刷题
      { text: '算法与刷题', link: '/notes/algorithm/' },
      // 学习路线
      { text: '学习路线', link: '/notes/roadmap.md' },
      // 工具环境
      { text: '工具与环境', link: '/notes/tools/' },
      // 收藏夹
      { text: '收藏夹', link: '/notes/starred.md' }
    ],

    // 侧边栏
    sidebar: {
      // 编程语言分类侧边栏
      '/notes/language/': [
        {
          text: 'Python',
          collapsible: true,
          items: [
            { text: '基础语法', link: '/notes/language/python/basic.md' },
            { text: '进阶特性', link: '/notes/language/python/advanced.md' },
            { text: '实战案例', link: '/notes/language/python/case.md' }
          ]
        },
        {
          text: 'C++',
          collapsible: true,
          items: [
            { text: '核心语法', link: '/notes/language/c++/basic.md' },
            { text: 'STL容器', link: '/notes/language/c++/stl.md' },
            { text: '高性能编程', link: '/notes/language/c++/high-perf.md' }
          ]
        },
        { text: '跨语言编程', link: '/notes/language/cross-lang.md' }
      ],

      // AI模型分类侧边栏（对齐首页「AI模型与算法」Feature）
      '/notes/ai-model/': [
        {
          text: '基础模型',
          collapsible: true,
          items: [
            { text: 'LLM大语言模型', link: '/notes/ai-model/llm.md' },
            { text: 'CNN/RNN网络', link: '/notes/ai-model/cnn-rnn.md' },
            { text: 'Transformer原理', link: '/notes/ai-model/transformer.md' }
          ]
        },
        { text: '模型训练调优', link: '/notes/ai-model/train.md' },
        { text: '模型推理部署', link: '/notes/ai-model/deploy.md' }
      ],

      // 深度学习分类侧边栏
      '/notes/deep-learning/': [
        { text: '环境配置', link: '/notes/deep-learning/env.md' },
        {
          text: 'PyTorch核心',
          collapsible: true,
          items: [
            { text: '张量与自动求导', link: '/notes/deep-learning/pytorch/tensor.md' },
            { text: '网络构建与训练', link: '/notes/deep-learning/pytorch/model.md' },
            { text: '数据加载与预处理', link: '/notes/deep-learning/pytorch/dataloader.md' }
          ]
        },
        { text: '实战项目', link: '/notes/deep-learning/project.md' },
        { text: '性能优化', link: '/notes/deep-learning/optimize.md' }
      ],

      // 算法与刷题侧边栏（对齐首页Feature）
      '/notes/algorithm/': [
        { text: '基础算法', link: '/notes/algorithm/basic.md' },
        { text: 'LeetCode刷题(Python/C++)', link: '/notes/algorithm/leetcode.md' },
        { text: '刷题技巧对比', link: '/notes/algorithm/language-compare.md' }
      ],

      // 工具与环境侧边栏（对齐首页Feature）
      '/notes/tools/': [
        { text: '开发环境配置', link: '/notes/tools/env-config.md' },
        { text: '调试技巧', link: '/notes/tools/debug.md' },
        { text: 'PyTorch部署工具', link: '/notes/tools/ai-deploy.md' }
      ],

      // 学习路线/收藏夹侧边栏（简化）
      '/notes/roadmap.md': [{ text: '学习路线', link: '/notes/roadmap.md' }],
      '/notes/starred.md': [{ text: '收藏夹', link: '/notes/starred.md' }]
    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/violet764' }
    ],

    // 页脚（与首页tagline一致）
    footer: {
      message: '深耕代码实践，拆解AI底层逻辑',
      copyright: 'Copyright © 2025 知行代码集 | 个人学习笔记'
    },

    // 增强功能（适配笔记查找/时效性）
    search: {
      provider: 'local'
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  }
})
