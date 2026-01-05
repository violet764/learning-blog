import { defineConfig } from 'vitepress'
import { set_sidebar } from '../.vitepress/utils/auto_sidebar.mjs';
import mathjax3 from 'markdown-it-mathjax3'
// 知行代码集 - Python/C++/AI模型/深度学习(PyTorch) 专属配置
export default 
  defineConfig({
  base:"/learning-blog/",
  head:[["link",{rel:"icon",href:"/learning-blog/spaceship.svg"}]],
  // 站点核心信息（与首页标题/定位一致）
  title: "星间飞行",
  description: "深耕Python/C++,拆解AI模型与深度学习(PyTorch)底层逻辑 | 个人技术学习笔记",
  // ========== 性能优化配置 ==========
  vite: {
    build: {
      chunkSizeWarningLimit: 1600,
      rollupOptions: {
        output: {
          manualChunks: {
            'vue': ['vue'],
            'markdown': ['@vue/compiler-sfc', 'markdown-it']
          }
        }
      }
    },
    ssr: {
      noExternal: ['shiki']
    }
  },
  // ========== 新增：markdown 配置（math: true 放在这里） ==========
  markdown: {
    math: true, // 启用VitePress内置的数学公式支持
    config: (md) => {
      md.use(mathjax3)
    },
    // 启用代码块行号
    lineNumbers: true,
    container: {
      tipLabel: '提示',
      warningLabel: '注意',
      dangerLabel: '警告',
      infoLabel: '信息',
      detailsLabel: '详情'
    },
    // 支持表情符号
    breaks: true,
    // 链接自动转换
    linkify: true,
    // 外部链接自动添加 target="_blank"
    externalLinks: {
      target: '_blank',
      rel: 'noopener noreferrer'
    }
  },
  // ==============================================================
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
          { text: 'Python', link: '/notes/language/python/' },
          { text: 'C++', link: '/notes/language/c++/' },
          { text: 'Markdown', link: '/notes/language/markdown/' }
        ] 
      },
      {
        text:'机器学习',
        items: [
          { text: '机器学习', link: '/notes/machine-learnig/ml.md' },
          { text: '有监督学习', link: 'notes/machine-learnig/supervised_learning.md' },
          { text: '无监督学习', link: 'notes/machine-learnig/unsupervised_learning.md' },

        ]
      },


      // 深度学习
      { 
        text: '深度学习', 
        items: [
          { text: '深度学习', link: '/notes/deep-learning/basic.md' },
          { text: '卷积神经网络', link: '/notes/deep-learning/cnn.md' },
          { text: '循环神经网络', link: '/notes/deep-learning/rnn.md' },
        ]
        
      },

      // 强化学习
      { text: '强化学习', 
        items:[
        { text: '导览', link: '/notes/reinforcement-learning/index.md' },
        { text: '强化学习', link: '/notes/reinforcement-learning/reinforcement_learning.md' },
        { text: 'Q-learning', link: '/notes/reinforcement-learning/Q-learning.md' },
        { text: 'DQN', link: '/notes/reinforcement-learning/DQN.md' },
        { text: 'DQN-Atari', link: '/notes/reinforcement-learning/DQN-Atari.md' },
        ]
      },

      // 算法刷题
      { text: '算法',
        items:[
        { text: '数据结构', link: '/notes/algorithm/数据结构与算法.md' },
        { text: '算法基础', link: '/notes/algorithm/算法应用.md' },

        {items:[
        { text: '搜索', link: '/notes/algorithm/搜索.md' },
        { text: '动态规划', link: '/notes/algorithm/动态规划.md' },
        { text: '字符串', link: '/notes/algorithm/字符串.md' },

        ]}
        ]
      },
      
      // 工具环境
      { text: '工具与环境',
        items:[

          { text: 'Shell', link: '/notes/tools/tools_shell.md' },
          { text: 'Vim', link: '/notes/tools/tools_vim.md' },
          { text: 'Git', link: '/notes/tools/tools_Git.md' },
          { text: 'Conda', link: '/notes/tools/tools_conda.md' },
          { text: 'IDE', link: '/notes/tools/IDE_introduce.md' },

        ] },

      // 收藏夹
      { text: '收藏夹', link: '/notes/starred' },
    
    ],

    // 侧边栏
    sidebar: {
      '/': [
        { text: '首页', link: '/' },
        { text: '笔记', link: '/posts/blog_setup.md' },
      ],
      
      // 编程语言分类侧边栏
      '/notes/language/': [
        {
          text: 'Python',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/python/index.md' },
            { text: '变量', link: '/notes/language/python/基础_变量.md' },
            { text: '容器', link: '/notes/language/python/基础_容器.md' },
            { text: '面向对象', link: '/notes/language/python/基础_函数与类.md' },
            { text: '装饰器', link: '/notes/language/python/基础_装饰器.md' },
            { text: '文件操作', link: '/notes/language/python/基础_文件操作.md' },
            { text: '正则表达式', link: '/notes/language/python/正则.md' },
            { text: 'Pandas', link: '/notes/language/python/Pandas.md' },
            { text: '数据可视化', link: '/notes/language/python/matplotlib.md' },

          ]
        },
        {
          text: 'C++',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/c++/index.md' },
            { text: '基础语法', link: '/notes/language/c++/basic.md' },
            { text: '函数与内存', link: '/notes/language/c++/func.md' },
            { text: '面向对象', link: '/notes/language/c++/oop.md' },
            { text: '进阶语法', link: '/notes/language/c++/advanced.md' },
            { text: 'STL容器', link: '/notes/language/c++/stl_container.md' },
            { text: 'STL算法', link: '/notes/language/c++/algorithm.md' },

          ]
        },
        {
          text: 'Markdown',
          collapsible: true,
          items: 
          [ { text: '导览', link: '/notes/language/markdown/index.md' },
            { text: '基础语法', link: '/notes/language/markdown/markdown-basic.md' },
            { text: '拓展语法', link: '/notes/language/markdown/markdown-advanced.md' },
            { text: '数学公式语法', link: '/notes/language/markdown/markdown-math.md' }
          ]
        },
      ],
      // 机器学习分类侧边栏
      '/notes/machine-learnig': [
        { text: '机器学习', link: '/notes/machine-learnig/ml.md' },
        { text: '有监督学习', link: '/notes/machine-learnig/supervised_learning.md' },
        { text: '无监督学习', link: '/notes/machine-learnig/unsupervised_learning.md' },
      
      
        
      ],

      // 深度学习分类侧边栏
      '/notes/deep-learning/': [
        { text: '导览', link: '/notes/deep-learning/index.md' },
        { text: '深度学习', link: '/notes/deep-learning/basic.md' },
        { text: '神经网络', link: '/notes/deep-learning/nn.md' },
        { text: '优化器', link: '/notes/deep-learning/optim.md' },
        { text: '卷积神经网络', link: '/notes/deep-learning/cnn.md' },
        { text: '循环神经网络', link: '/notes/deep-learning/rnn.md' },
        { text: '实战', link: '/notes/deep-learning/practical.md' },
      ],

      // 算法与刷题侧边栏（对齐首页Feature）
      '/notes/algorithm/': [
        { text: '数据结构', link: '/notes/algorithm/数据结构与算法.md' },
        { text: '算法基础', link: '/notes/algorithm/算法应用.md' },
        { text: '搜索', link: '/notes/algorithm/搜索.md' },
        { text: '动态规划', link: '/notes/algorithm/动态规划.md' },
        { text: '字符串', link: '/notes/algorithm/字符串.md' },
      ],

      // 工具与环境侧边栏（对齐首页Feature）
      '/notes/tools/': [

        { text: 'Shell', link: '/notes/tools/tools_shell.md' },
        { text: 'Vim', link: '/notes/tools/tools_vim.md' },
        { text: 'Git', link: '/notes/tools/tools_Git.md' },
        { text: 'Conda', link: '/notes/tools/tools_conda.md' },
        { text: 'IDE', link: '/notes/tools/IDE_introduce.md' },

      ],


      '/notes/starred': [
        { text: '经典论文', link: '/notes/starred/awesome_paper.md' },
        { text: '优质博客', link: '/notes/starred/awesome_blog.md' },


      ],

    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/violet764/learning-blog' },
      { icon: {
        svg: '<svg t="1764941384467" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1774" width="200" height="200"><path d="M1019.54782609 345.3106087c-3.20556522-142.1133913-127.15408696-169.36069565-127.15408696-169.36069566s-96.70121739-0.53426087-222.25252174-1.60278261l91.3586087-88.15304347s14.42504348-18.16486957-10.15095652-38.46678261c-24.576-20.30191304-26.17878261-11.21947826-34.72695653-5.87686957-7.47965217 5.3426087-117.00313043 112.72904348-136.23652174 131.96243479-49.68626087 0-101.50956522-0.53426087-151.73008695-0.53426087h17.63060869S315.392 43.98747826 306.84382609 38.1106087s-9.61669565-14.42504348-34.72695652 5.87686956c-24.576 20.30191304-10.15095652 38.46678261-10.15095653 38.46678261l93.49565218 90.82434783c-101.50956522 0-189.12834783 0.53426087-229.73217392 2.13704347C-5.69878261 213.34817391 4.45217391 345.3106087 4.45217391 345.3106087s1.60278261 283.15826087 0 426.34017391c14.42504348 143.18191304 124.48278261 166.15513043 124.48278261 166.15513043s43.8093913 1.06852174 76.39930435 1.06852174c3.20556522 9.08243478 5.87686957 53.96034783 56.0973913 53.96034783 49.68626087 0 56.0973913-53.96034783 56.09739131-53.96034783s365.96869565-1.60278261 396.42156522-1.60278261c1.60278261 15.49356522 9.08243478 56.63165217 59.30295652 56.09739131 49.68626087-1.06852174 53.42608696-59.30295652 53.42608695-59.30295652s17.09634783-1.60278261 67.85113044 0c118.60591304-21.90469565 125.55130435-160.81252174 125.55130435-160.81252174s-2.13704348-285.82956522-0.53426087-427.94295652z m-102.04382609 453.05321739c0 22.43895652-17.6306087 40.60382609-39.53530435 40.60382608h-721.25217391c-21.90469565 0-39.53530435-18.16486957-39.53530435-40.60382608V320.20034783c0-22.43895652 17.6306087-40.60382609 39.53530435-40.60382609h721.25217391c21.90469565 0 39.53530435 18.16486957 39.53530435 40.60382609v478.16347826z" fill="#1296db" p-id="1775"></path><path d="M409.088 418.816l-203.264 38.912 17.408 76.288 201.216-38.912zM518.656 621.056c-49.664 106.496-94.208 26.112-94.208 26.112l-33.28 21.504s65.536 89.6 128 21.504c73.728 68.096 130.048-22.016 130.048-22.016l-30.208-19.456c0-0.512-52.736 75.776-100.352-27.648zM619.008 495.104l201.728 38.912 16.896-76.288-202.752-38.912z" fill="#1296db" p-id="1776"></path></svg>'
      }, link: 'https://space.bilibili.com/107668498' },
      { icon: {
        svg: '<svg t="1765004841104" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4604" width="200" height="200"><path d="M701.28 766.86a58.9 58.9 0 0 1 83.1 83.52l-102.24 102c-94.3 94.16-248 95.52-344 3.16-0.56-0.52-43.26-42.4-184.12-180.54-93.72-92-103.06-238.94-14.88-333.36l164.44-176C391.12 172 552.46 161.68 652.54 242.6l149.34 120.78a58.92 58.92 0 0 1-74 91.76l-149.34-120.78c-52.34-42.34-144-36.52-189.06 11.84l-164.44 176c-42.94 46-38.22 120 11.26 168.54l183.26 179.7c49.86 48 130.48 47.3 179.42-1.58z" fill="#FFA116" p-id="4605"></path><path d="M452.94 664.7a58.96 58.96 0 0 1 0-118h434a58.96 58.96 0 0 1 0 118z" fill="#B3B3B3" p-id="4606"></path><path d="M534.22 18.68a58.9 58.9 0 1 1 86 80.56L225.1 522.28c-42.92 46-38.22 120 11.24 168.54l182.46 178.9A58.92 58.92 0 0 1 336.46 954L154.06 775.08c-93.7-92-103.04-238.94-14.84-333.36z" p-id="4607"></path></svg>'
      }, link: 'https://leetcode.cn/problemset/'}
    ],

    // 页脚（与首页tagline一致）
    footer: {
      message: '星际航行，智慧笔记',
      copyright: 'Copyright © 2025 星间飞行 | 个人学习笔记'
    },

    // 增强功能（适配笔记查找/时效性）
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        // 搜索快捷键
        hotKeys: ['s', '/'],
        // 排除搜索的文件
        exclude: ['node_modules/**', 'dist/**', '.vitepress/**']
      }
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    // 文档页脚
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    }
    },
    ignoreDeadLinks: true,
})
