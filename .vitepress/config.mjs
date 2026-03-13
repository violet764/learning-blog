import { defineConfig } from 'vitepress'
import { set_sidebar } from '../.vitepress/utils/auto_sidebar.mjs';
import { withMermaid } from 'vitepress-plugin-mermaid'
import mathjax3 from 'markdown-it-mathjax3';


// 知行代码集 - Python/C++/AI模型/深度学习(PyTorch) 专属配置
export default withMermaid(
  defineConfig({
    markdown: {
      math: true, // 启用VitePress内置的数学公式支持
      extendMarkdown: (md) => {
      md.use(katex)
      },
      config: (md) => {
        md.use(mathjax3);
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
  base:"/learning-blog/",
  head:[["link",{rel:"icon",href:"/learning-blog/spaceship.svg"}]],
  // 站点核心信息（与首页标题/定位一致）
  title: "星间飞行",
  description: "深耕Python/C++,拆解AI模型与深度学习(PyTorch)底层逻辑 | 个人技术学习笔记",
  // 添加 Vite 配置来处理 Mermaid 依赖
  vite: {
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'dayjs',
        'debug',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'd3',
        'khroma',
        'dompurify'
      ]
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            // 将大型依赖分离成独立 chunk
            'vendor-mermaid': ['mermaid', 'd3'],
            'vendor-cytoscape': ['cytoscape', 'cytoscape-cose-bilkent'],
          }
        }
      },
      chunkSizeWarningLimit: 1000
    }
  },
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
          { text: 'Java', link: '/notes/language/java/' },
          { text: 'C++', link: '/notes/language/c++/' },
          { text: 'Rust', link: '/notes/language/rust/' },
          { text: 'Markdown', link: '/notes/language/markdown/' },
          { text: '数据库', link: '/notes/language/database/' }
        ] 
      },
      { text: '机器学习',
        items: [
          { text: '导览', link: '/notes/machine-learning/index.md' },
          { text: '数学基础', link: '/notes/machine-learning/math-foundation/linear-algebra.md' },
          { text: '特征工程', link: '/notes/machine-learning/feature-engineering/index.md' },
          { text: '有监督学习', link: '/notes/machine-learning/supervised-learning/framework/supervised-learning.md' },
          { text: '无监督学习', link: '/notes/machine-learning/unsupervised-learning/index.md' },
          { text: '模型评估', link: '/notes/machine-learning/model-evaluation/cross-validation.md' }
        ]
      },


      // 深度学习
      { text: '深度学习', 
        items: [
          { text: '导览', link: '/notes/deep-learning/index.md' },
          { text: '神经网络基础', link: '/notes/deep-learning/01-neural-network-basics.md' },
          { text: 'CNN', link: '/notes/deep-learning/05-cnn.md' },
          { text: 'Transformer', link: '/notes/deep-learning/07-transformer.md' },
          { text: '生成模型', link: '/notes/deep-learning/08-generative-models.md' },
          { text: 'PyTorch', link: '/notes/deep-learning/pytorch/index.md' },
        ]
      },
      // AI大模型
      { text: 'AI大模型', 
        items: [
          { text: '导览', link: '/notes/ai-model/index.md' },
          { text: 'LLM', link: '/notes/ai-model/llm/index.md' },
          { text: '视觉模型', link: '/notes/ai-model/cv/index.md' },
          { text: '多模态', link: '/notes/ai-model/multimodal/index.md' },
          { text: 'AI智能体', link: '/notes/ai-model/applications/agentic-ai.md' },
          { text: '常用库', link: '/notes/framework' },
        ]
      },

      // 强化学习
      { text: '强化学习', 
        items:[
        { text: '导览', link: '/notes/reinforcement-learning/index.md' },
        { text: '强化学习基础', link: '/notes/reinforcement-learning/rl-basics.md' },
        { text: 'DQN与函数逼近', link: '/notes/reinforcement-learning/function-approximation.md' },
        { text: 'PPO/SAC/TD3', link: '/notes/reinforcement-learning/modern-algorithms.md' },
        { text: 'RLHF与大模型', link: '/notes/reinforcement-learning/llm-rl.md' },
        { text: 'DPO偏好优化', link: '/notes/reinforcement-learning/dpo-preference-optimization.md' },
        ]
      },

      // 算法刷题
      { text: '算法',
        items:[
        { text: '导览', link: '/notes/algorithm/index.md' },
        { text: '数据结构', link: '/notes/algorithm/linear-structures.md' },
        { text: '搜索算法', link: '/notes/algorithm/search-dfs.md' },
        { text: '动态规划', link: '/notes/algorithm/dp-basics.md' },
        { text: '图论算法', link: '/notes/algorithm/graph-basics.md' },
        ]
      },
      
      // 工具环境
      { text: '工具与环境',
        items:[

          { text: 'Shell', link: '/notes/tools/tools_shell.md' },
          { text: 'Bash', link: '/notes/tools/bash.md' },
          { text: 'Vim', link: '/notes/tools/tools_vim.md' },
          { text: 'Git', link: '/notes/tools/tools_Git.md' },
          { text: 'Conda', link: '/notes/tools/tools_conda.md' },
          { text: 'RSS', link: '/rss-reader/' },
        ] },

      // 收藏夹
      { text: '收藏夹', link: '/notes/starred' },
      
      // RSS阅读器
      // { text: 'RSS阅读器', link: '/rss-reader/' },
    
    ],

    // 侧边栏
    sidebar: {
      '/': [
        { text: '首页', link: '/' },
        { text: '笔记', link: '/posts/blog_setup.md' },
      ],
      '/rss-reader/': [
        { text: '使用说明', link: '/rss-reader/' },
        { text: 'RSS阅读器', link: '/rss-reader/reader' }
      ],
      
      // 编程语言分类侧边栏
      '/notes/language/python/': [ //想要在一个内容中定义多个侧边栏，在[]内部填写多个块即可
        {
          text: 'Python',
          collapsible: true,
          items: [
          { text: '导览', link: '/notes/language/python/index.md' },
          { text: '变量', link: '/notes/language/python/基础_变量.md' },
          { text: '容器(列表，元组，集合）', link: '/notes/language/python/基础_容器1.md' },
          { text: '容器(字典，字符串）', link: '/notes/language/python/基础_容器2.md' },
          { text: '函数', link: '/notes/language/python/基础_函数.md' },
          { text: '面向对象', link: '/notes/language/python/基础_面向对象.md' },
          { text: '异常处理', link: '/notes/language/python/基础_异常处理.md' },
          { text: '常用标准库', link: '/notes/language/python/常用库.md' },
          { text: '文件操作', link: '/notes/language/python/基础_文件操作.md' },
          { text: '正则表达式', link: '/notes/language/python/正则.md' },
          { text: 'Numpy', link: '/notes/language/python/numpy.md' },
          { text: 'Pandas1', link: '/notes/language/python/pandas1.md' },
          { text: 'Pandas2', link: '/notes/language/python/pandas2.md' },
          { text: '数据可视化', link: '/notes/language/python/matplotlib.md' },

          ]
        }],
      
      '/notes/language/c++/': [
        {
          text: 'C++',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/c++/index.md' },
            { text: '基础语法', link: '/notes/language/c++/基础语法.md' },
            { text: '函数', link: '/notes/language/c++/函数.md' },
            { text: '面向对象', link: '/notes/language/c++/面向对象.md' },
            { text: '模板与泛型', link: '/notes/language/c++/模板.md' },
            { text: 'STL', link: '/notes/language/c++/STL.md' },
            { text: 'IO与异常', link: '/notes/language/c++/IO与异常.md' },
            { text: '多线程编程', link: '/notes/language/c++/多线程编程.md' },
            { text: '多文件编程', link: '/notes/language/c++/多文件.md' },
            { text: '现代特性', link: '/notes/language/c++/现代特性.md' },
          ]
        },
        {
          text: '面试指南',
          collapsible: true,
          items: [
            { text: '核心面试题', link: '/notes/language/c++/interview/index.md' },
          ]
        }],
      
      '/notes/language/java/': [
        {
          text: 'Java',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/java/index.md' },
            { text: '基础语法', link: '/notes/language/java/basics.md' },
            { text: '面向对象', link: '/notes/language/java/oop.md' },
            { text: '集合框架', link: '/notes/language/java/collections.md' },
            { text: '泛型', link: '/notes/language/java/generics.md' },
            { text: '异常处理', link: '/notes/language/java/exception.md' },
            { text: 'Lambda与Stream', link: '/notes/language/java/lambda-stream.md' },
          ]
        },
        {
          text: '多线程与并发',
          collapsible: true,
          items: [
            { text: '并发编程', link: '/notes/language/java/concurrency.md' },
          ]
        },
        {
          text: '开发框架',
          collapsible: true,
          items: [
            { text: 'Spring 核心', link: '/notes/language/java/spring.md' },
            { text: 'Spring Boot', link: '/notes/language/java/springboot.md' },
            { text: 'Spring MVC', link: '/notes/language/java/springmvc.md' },
            { text: 'MyBatis', link: '/notes/language/java/mybatis.md' },
            { text: 'Spring Security', link: '/notes/language/java/spring-security.md' },
          ]
        },
        {
          text: '面试专栏',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/java/interview/index.md' },
            { text: 'Java 基础', link: '/notes/language/java/interview/java-basics.md' },
            { text: 'Java 集合', link: '/notes/language/java/interview/java-collections.md' },
            { text: 'Java 并发', link: '/notes/language/java/interview/java-concurrent.md' },
            { text: 'JVM', link: '/notes/language/java/interview/java-jvm.md' },
            { text: 'Spring 框架', link: '/notes/language/java/interview/spring.md' },
            { text: 'MySQL', link: '/notes/language/java/interview/mysql.md' },
            { text: 'Redis', link: '/notes/language/java/interview/redis.md' },
          ]
        }],


        '/notes/language/markdown/': [
        {
          text: 'Markdown',
          collapsible: true,
          items: 
          [ { text: '导览', link: '/notes/language/markdown/index.md' },
            { text: '基础语法', link: '/notes/language/markdown/markdown-basic.md' },
            { text: '拓展语法', link: '/notes/language/markdown/markdown-advanced.md' },
            { text: '数学公式语法', link: '/notes/language/markdown/markdown-math.md' },
          
          ]
        },
      ],

        '/notes/language/rust/': [
        {
          text: 'Rust',
          collapsible: true,
          items: 
          [ { text: '导览', link: '/notes/language/rust/index.md' },
            { text: '初识Rust', link: '/notes/language/rust/初识Rust.md' },
            { text: '基础语法', link: '/notes/language/rust/基础语法.md' },
            { text: '所有权与借用', link: '/notes/language/rust/所有权与借用.md' },
            { text: '结构体与枚举', link: '/notes/language/rust/结构体与枚举.md' },
            { text: '泛型与特征', link: '/notes/language/rust/泛型与特征.md' },
            { text: '模块与并发', link: '/notes/language/rust/模块与并发.md' },
            { text: '实战与优化', link: '/notes/language/rust/实战与优化.md' },
            { text: '不安全Rust', link: '/notes/language/rust/不安全Rust.md' },
          ]
        },
        {
          text: '面试指南',
          collapsible: true,
          items: [
            { text: '核心面试题', link: '/notes/language/rust/interview/index.md' },
          ]
        },
      ],

      '/notes/language/database/': [
        {
          text: '关系型数据库',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/language/database/index.md' },
            { text: 'MySQL基础', link: '/notes/language/database/mysql-basics.md' },
            { text: 'MySQL面试题', link: '/notes/language/database/mysql-interview.md' },
            { text: 'MySQL进阶面试题', link: '/notes/language/database/mysql-interview-advanced.md' },
            { text: 'PostgreSQL基础', link: '/notes/language/database/postgresql-basics.md' },
            { text: 'PostgreSQL面试题', link: '/notes/language/database/postgresql-interview.md' },
          ]
        },
        {
          text: 'NoSQL与消息队列',
          collapsible: true,
          items: [
            { text: 'Redis基础', link: '/notes/language/database/redis-basics.md' },
            { text: 'Redis面试题', link: '/notes/language/database/redis-interview.md' },
            { text: 'Redis进阶面试题', link: '/notes/language/database/redis-interview-advanced.md' },
            { text: 'Kafka基础', link: '/notes/language/database/kafka-basics.md' },
            { text: 'Kafka面试题', link: '/notes/language/database/kafka-interview.md' },
            { text: 'Kafka进阶面试题', link: '/notes/language/database/kafka-interview-advanced.md' },
            { text: 'RabbitMQ基础', link: '/notes/language/database/rabbitmq-basics.md' },
            { text: 'RabbitMQ面试题', link: '/notes/language/database/rabbitmq-interview.md' },
          ]
        },
        {
          text: '计算机网络',
          collapsible: true,
          items: [
            { text: '计算机网络面试题', link: '/notes/language/database/network-interview.md' },
          ]
        },
        {
          text: '计算机组成原理',
          collapsible: true,
          items: [
            { text: '计算机组成原理面试题', link: '/notes/language/database/computer-organization-interview.md' },
          ]
        },
      ],
      // 机器学习分类侧边栏
      '/notes/machine-learning/': [
        {
          text: '数学基础',
          collapsible: true,
          items: [
            { text: '线性代数', link: '/notes/machine-learning/math-foundation/linear-algebra.md' },
            { text: '概率统计', link: '/notes/machine-learning/math-foundation/probability-statistics.md' },
            { text: '优化理论', link: '/notes/machine-learning/math-foundation/optimization-theory.md' }
          ]
        },
        { text: '特征工程',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/machine-learning/feature-engineering/index.md' },
            { text: '数据预处理', link: '/notes/machine-learning/feature-engineering/preprocessing.md' },
            { text: '特征变换', link: '/notes/machine-learning/feature-engineering/feature-transformation.md' },
            { text: '特征选择', link: '/notes/machine-learning/feature-engineering/feature-selection.md' }
          ]
        },
      
        {
          text: '模型评估',
          collapsible: true,
          items: [
            { text: '交叉验证', link: '/notes/machine-learning/model-evaluation/cross-validation.md' },
            { text: '学习理论', link: '/notes/machine-learning/model-evaluation/learning-theory.md' },
            { text: '统计检验', link: '/notes/machine-learning/model-evaluation/statistical-tests.md' },

          ]
        }
      ],

      '/notes/machine-learning/supervised-learning/':[
        { text: '导览', link: '/notes/machine-learning/supervised-learning/index.md' },
        { text: '理论框架', link: '/notes/machine-learning/supervised-learning/framework/supervised-learning.md' },
        {
          text: '线性模型',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/machine-learning/supervised-learning/linear-models/index.md' },
            { text: '线性回归', link: '/notes/machine-learning/supervised-learning/linear-models/linear-regression.md' },
            { text: '逻辑回归', link: '/notes/machine-learning/supervised-learning/linear-models/logistic-regression.md' },
            { text: '正则化', link: '/notes/machine-learning/supervised-learning/linear-models/regularization.md' },

          ]
        },
        {
          text: '树模型与集成',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/machine-learning/supervised-learning/tree-models/index.md' },
            { text: '决策树', link: '/notes/machine-learning/supervised-learning/tree-models/decision-trees.md' },
            { text: '随机森林', link: '/notes/machine-learning/supervised-learning/tree-models/random-forest.md' },
            { text: 'AdaBoost', link: '/notes/machine-learning/supervised-learning/tree-models/adaboost.md' },
            { text: 'GBDT', link: '/notes/machine-learning/supervised-learning/tree-models/gradient-boosting.md' },
            { text: 'XGBoost', link: '/notes/machine-learning/supervised-learning/tree-models/xgboost.md' },
            { text: 'LightGBM', link: '/notes/machine-learning/supervised-learning/tree-models/lightgbm.md' },

          ]
        },
        {
          text: '支持向量机',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/machine-learning/supervised-learning/svm/index.md' },
            { text: 'SVM理论', link: '/notes/machine-learning/supervised-learning/svm/svm-theory.md' },
            { text: '核方法', link: '/notes/machine-learning/supervised-learning/svm/kernel-methods.md' },
            { text: 'SVM实现', link: '/notes/machine-learning/supervised-learning/svm/svm-implementation.md' }
          ]
        },

        {
          text: '贝叶斯方法',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/machine-learning/supervised-learning/bayesian-methods/index.md' },
            { text: '朴素贝叶斯', link: '/notes/machine-learning/supervised-learning/bayesian-methods/naive-bayes.md' },
            { text: 'EM算法', link: '/notes/machine-learning/supervised-learning/bayesian-methods/em-algorithm.md' },
            { text: '高斯过程', link: '/notes/machine-learning/supervised-learning/bayesian-methods/gaussian-processes.md' }
          ]
        }
      ],

      '/notes/machine-learning/unsupervised-learning/':[
        { text: '导览', link: '/notes/machine-learning/unsupervised-learning/index.md' },
        { text: '聚类分析', link: '/notes/machine-learning/unsupervised-learning/clustering.md' },
        { text: '降维技术', link: '/notes/machine-learning/unsupervised-learning/dimensionality-reduction.md' },
        { text: '关联规则与异常检测', link: '/notes/machine-learning/unsupervised-learning/association-anomaly.md' },
        { text: '高级数据结构', link: '/notes/machine-learning/unsupervised-learning/advanced-structures.md' }
      ],
      // 深度学习分类侧边栏
      '/notes/deep-learning/': [
        { text: '导览', link: '/notes/deep-learning/index.md' },
        {
          text: '基础理论',
          collapsible: true,
          items: [
            { text: '神经网络基础', link: '/notes/deep-learning/01-neural-network-basics.md' },
            { text: '优化算法', link: '/notes/deep-learning/02-optimizers.md' },
            { text: '激活函数', link: '/notes/deep-learning/03-activations.md' },
            { text: '正则化与归一化', link: '/notes/deep-learning/04-regularization.md' },
          ]
        },
        {
          text: '网络架构',
          collapsible: true,
          items: [
            { text: '卷积神经网络', link: '/notes/deep-learning/05-cnn.md' },
            { text: '循环神经网络', link: '/notes/deep-learning/06-rnn-family.md' },
            { text: 'Transformer', link: '/notes/deep-learning/07-transformer.md' },
            { text: '生成模型', link: '/notes/deep-learning/08-generative-models.md' },
          ]
        },
        { text: 'PyTorch实战', link: '/notes/deep-learning/pytorch/index.md' },
      ],

      '/notes/deep-learning/pytorch/': [
        {
          text: 'PyTorch',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/deep-learning/pytorch/index.md' },
            { text: '基础功能', link: '/notes/deep-learning/pytorch/pytorch-basics.md' },
            { text: '张量操作', link: '/notes/deep-learning/pytorch/pytorch-tensors.md' },
            { text: '神经网络', link: '/notes/deep-learning/pytorch/pytorch-nn.md' },
            { text: '数据处理', link: '/notes/deep-learning/pytorch/pytorch-data.md' },
            { text: '模型训练', link: '/notes/deep-learning/pytorch/pytorch-training.md' },
            { text: '高级功能', link: '/notes/deep-learning/pytorch/pytorch-advanced.md' },
            { text: '实战项目', link: '/notes/deep-learning/pytorch/pytorch-projects.md' }
          ]
      }
      ],

      '/notes/practice/': [
        { text: '导览', link: '/notes/practice/index.md' },
        { text: 'micrograd', link: '/notes/practice/nn_zero2hero/micrograd.md' },
        { text: 'makemore', link: '/notes/practice/nn_zero2hero/makemore.md' },
        { text: 'nn_zero2hero', link: '/notes/paper/chapters_1_3_detailed.md' },

      ],


      '/notes/ai-model/': [
        { text: '导览', link: '/notes/ai-model/index.md' },
        {
          text: '大语言模型',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/llm/index.md' },
            { text: '分词技术', link: '/notes/ai-model/llm/tokenization.md' },
            { text: '嵌入层', link: '/notes/ai-model/llm/embedding.md' },
            { text: '注意力机制', link: '/notes/ai-model/llm/attention-mechanisms.md' },
            { text: '模型架构', link: '/notes/ai-model/llm/model-architecture.md' },
            { text: '预训练技术', link: '/notes/ai-model/llm/pretraining.md' },
            { text: 'Bert', link: '/notes/ai-model/llm/bert-models.md' },
            { text: '微调与对齐', link: '/notes/ai-model/llm/finetuning-alignment.md' },
            { text: '经典模型', link: '/notes/ai-model/llm/llm-models.md' },
            { text: '推理优化', link: '/notes/ai-model/llm/inference-optimization.md' },
          ]
        },
        {
          text: '视觉模型',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/cv/index.md' },
            { text: '视觉模型演进', link: '/notes/ai-model/cv/vision-models.md' },
            { text: '目标检测', link: '/notes/ai-model/cv/object-detection.md' },
            { text: '图像生成', link: '/notes/ai-model/cv/image-generation.md' },
          ]
        },
        {
          text: '多模态模型',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/multimodal/index.md' },
            { text: '视觉语言模型', link: '/notes/ai-model/multimodal/vision-language.md' },
            { text: '多模态生成', link: '/notes/ai-model/multimodal/multimodal-generation.md' },
          ]
        },
        {
          text: '应用与前沿',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/applications/index.md' },
            { text: 'AI智能体', link: '/notes/ai-model/applications/agentic-ai.md' },

            { text: '未来趋势', link: '/notes/ai-model/applications/future-trends.md' },
          ]
        },
        {
          text: 'Agent 学习指南',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/agent/index.md' },
            { text: 'Agent 基础概念', link: '/notes/ai-model/agent/agent-basics.md' },
            { text: 'ReAct与工具调用', link: '/notes/ai-model/agent/agent-react-tools.md' },
            { text: 'CLI Agent实战', link: '/notes/ai-model/agent/agent-cli-tutorial.md' },
            { text: '多Agent编排模式', link: '/notes/ai-model/agent/agent-orchestration.md' },
            { text: 'OpenClaw', link: '/notes/ai-model/applications/openclaw-intro.md' }
          ]
        },
        {
          text: '面试指南',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/ai-model/interview/index.md' },
            { text: '深度学习基础', link: '/notes/ai-model/interview/dl-basics.md' },
            { text: 'Transformer架构', link: '/notes/ai-model/interview/transformer.md' },
            { text: '大模型核心原理', link: '/notes/ai-model/interview/llm-advanced.md' },
            { text: '微调与优化', link: '/notes/ai-model/interview/finetuning-optimization.md' },
            { text: '多模态模型', link: '/notes/ai-model/interview/multimodal.md' },
            { text: 'Prompt工程', link: '/notes/ai-model/interview/prompt-engineering.md' },
            { text: 'RAG与知识增强', link: '/notes/ai-model/interview/rag-knowledge.md' },
            { text: '模型部署与工程化', link: '/notes/ai-model/interview/deployment-engineering.md' },
            { text: '安全与伦理', link: '/notes/ai-model/interview/safety-ethics.md' },
            { text: '实战场景', link: '/notes/ai-model/interview/practical-scenarios.md' },
          ]
        },
      ],


      // 强化学习侧边栏
      '/notes/reinforcement-learning/': [
        { text: '导览', link: '/notes/reinforcement-learning/index.md' },
        {
          text: '基础理论',
          collapsible: true,
          items: [
            { text: '强化学习基础', link: '/notes/reinforcement-learning/rl-basics.md' },
            { text: '表格型方法', link: '/notes/reinforcement-learning/tabular-methods.md' },
          ]
        },
        {
          text: '深度强化学习',
          collapsible: true,
          items: [
            { text: '函数逼近与DQN', link: '/notes/reinforcement-learning/function-approximation.md' },
            { text: '策略梯度方法', link: '/notes/reinforcement-learning/policy-gradient.md' },
            { text: '现代算法(PPO/SAC/TD3)', link: '/notes/reinforcement-learning/modern-algorithms.md' },
          ]
        },
        {
          text: '大模型RL',
          collapsible: true,
          items: [
            { text: 'RLHF原理与实践', link: '/notes/reinforcement-learning/llm-rl.md' },
            { text: 'DPO直接偏好优化', link: '/notes/reinforcement-learning/dpo-preference-optimization.md' },
            { text: '大模型对齐技术', link: '/notes/reinforcement-learning/llm-alignment.md' },
          ]
        },
        {
          text: '前沿应用',
          collapsible: true,
          items: [
            { text: '多智能体RL', link: '/notes/reinforcement-learning/multi-agent-rl.md' },
          ]
        },
        {
          text: '工程实践',
          collapsible: true,
          items: [
            { text: '实战环境', link: '/notes/reinforcement-learning/rl-environments.md' },
            { text: '工程实践', link: '/notes/reinforcement-learning/rl-engineering.md' },
          ]
        },
      ],

      // 算法与刷题侧边栏
      '/notes/algorithm/': [
        { text: '导览', link: '/notes/algorithm/index.md' },
        {
          text: '算法基础',
          collapsible: true,
          items: [
            { text: '复杂度与二分', link: '/notes/algorithm/basics.md' },
            { text: '排序算法', link: '/notes/algorithm/sorting.md' },
            { text: '贪心算法', link: '/notes/algorithm/greedy.md' },
          ]
        },
        {
          text: '数据结构',
          collapsible: true,
          items: [
            { text: '线性结构', link: '/notes/algorithm/linear-structures.md' },
            { text: '哈希表与位集', link: '/notes/algorithm/hashing.md' },
            { text: '树基础', link: '/notes/algorithm/tree-basics.md' },
            { text: '高级树结构', link: '/notes/algorithm/tree-advanced.md' },
            { text: '并查集', link: '/notes/algorithm/union-find.md' },
          ]
        },
        {
          text: '图论算法',
          collapsible: true,
          items: [
            { text: '图基础', link: '/notes/algorithm/graph-basics.md' },
            { text: '最短路径', link: '/notes/algorithm/graph-shortest-path.md' },
            { text: '最小生成树', link: '/notes/algorithm/graph-spanning-tree.md' },
          ]
        },
        {
          text: '搜索算法',
          collapsible: true,
          items: [
            { text: 'DFS', link: '/notes/algorithm/search-dfs.md' },
            { text: 'BFS', link: '/notes/algorithm/search-bfs.md' },
            { text: '高级搜索', link: '/notes/algorithm/search-advanced.md' },
          ]
        },
        {
          text: '动态规划',
          collapsible: true,
          items: [
            { text: 'DP基础', link: '/notes/algorithm/dp-basics.md' },
            { text: '背包问题', link: '/notes/algorithm/dp-knapsack.md' },
            { text: '高级DP', link: '/notes/algorithm/dp-advanced.md' },
          ]
        },
        {
          text: '字符串算法',
          collapsible: true,
          items: [
            { text: '字符串基础', link: '/notes/algorithm/string-basics.md' },
            { text: '高级字符串', link: '/notes/algorithm/string-advanced.md' },
          ]
        },
      ],

      // 工具与环境侧边栏（对齐首页Feature）
      '/notes/tools/': [
        { text: 'Shell', link: '/notes/tools/tools_shell.md' },
        { text: 'Bash', link: '/notes/tools/bash.md' },
        { text: 'Vim', link: '/notes/tools/tools_vim.md' },
        { text: 'Git', link: '/notes/tools/tools_Git.md' },
        { text: 'Conda', link: '/notes/tools/tools_conda.md' },
        { text: 'IDE', link: '/notes/tools/IDE_introduce.md' },
        { text: 'Claude', link: '/notes/tools/claude-code-guide.md' },
        { text: 'iflow', link: '/notes/tools/iflow-cli-guide.md' },

        
      ],


      '/notes/starred': [
        { text: '经典论文', link: '/notes/starred/awesome_paper.md' },
        { text: '优质博客', link: '/notes/starred/awesome_blog.md' },
      ],

      // 开发框架侧边栏
      '/notes/framework/': [
        { text: '导览', link: '/notes/framework/index.md' },
        {
          text: 'Gradio',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/gradio/index.md' },
            { text: '基础组件与布局', link: '/notes/framework/gradio/gradio-basics.md' },
            { text: 'Interface快速构建', link: '/notes/framework/gradio/gradio-interface.md' },
            { text: 'Blocks灵活布局', link: '/notes/framework/gradio/gradio-blocks.md' },
            { text: '高级特性与部署', link: '/notes/framework/gradio/gradio-advanced.md' },
          ]
        },
        {
          text: 'LangChain',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/langchain/index.md' },
            { text: '基础概念', link: '/notes/framework/langchain/langchain-basics.md' },
            { text: '链与LCEL', link: '/notes/framework/langchain/langchain-chains.md' },
            { text: '智能体开发', link: '/notes/framework/langchain/langchain-agents.md' },
            { text: '工具调用', link: '/notes/framework/langchain/langchain-tools.md' },
            { text: '记忆系统', link: '/notes/framework/langchain/langchain-memory.md' },
          ]
        },
        {
          text: 'LlamaIndex',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/llamaindex/index.md' },
            { text: '基础概念与快速开始', link: '/notes/framework/llamaindex/llamaindex-basics.md' },
            { text: '索引类型与选择', link: '/notes/framework/llamaindex/llamaindex-index.md' },
            { text: '检索策略', link: '/notes/framework/llamaindex/llamaindex-retrieval.md' },
            { text: 'RAG应用构建', link: '/notes/framework/llamaindex/llamaindex-rag.md' },
            { text: '高级用法与优化', link: '/notes/framework/llamaindex/llamaindex-advanced.md' },
          ]
        },
        {
          text: 'Transformers',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/transformers/index.md' },
            { text: '基础用法', link: '/notes/framework/transformers/transformers-basics.md' },
            { text: 'Pipeline快速使用', link: '/notes/framework/transformers/transformers-pipelines.md' },
            { text: '模型训练与微调', link: '/notes/framework/transformers/transformers-training.md' },
            { text: '推理与部署', link: '/notes/framework/transformers/transformers-inference.md' },
            { text: '自定义模型与组件', link: '/notes/framework/transformers/transformers-custom.md' },
          ]
        },
        {
          text: 'OpenAI SDK',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/openai/index.md' },
            { text: '基础API调用', link: '/notes/framework/openai/openai-basics.md' },
            { text: 'Chat Completions', link: '/notes/framework/openai/openai-chat.md' },
            { text: 'Embeddings API', link: '/notes/framework/openai/openai-embeddings.md' },
            { text: 'Function Calling', link: '/notes/framework/openai/openai-functions.md' },
            { text: 'Assistants API', link: '/notes/framework/openai/openai-assistants.md' },
            { text: '流式响应处理', link: '/notes/framework/openai/openai-streaming.md' },
          ]
        },
        {
          text: 'vLLM',
          collapsible: true,
          items: [
            { text: '导览', link: '/notes/framework/vllm/index.md' },
            { text: '基础概念与安装', link: '/notes/framework/vllm/vllm-basics.md' },
            { text: '离线推理', link: '/notes/framework/vllm/vllm-inference.md' },
            { text: 'API服务器部署', link: '/notes/framework/vllm/vllm-server.md' },
            { text: '性能优化技巧', link: '/notes/framework/vllm/vllm-optimization.md' },
          ]
        },
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
    // Sitemap配置
    sitemap: {
      hostname: 'https://violet764.github.io/learning-blog/',
      transformItems(items) {
        // 可以在这里自定义sitemap条目
        return items.map(item => ({
          ...item,
          changefreq: 'monthly',
          priority: 0.8
        }))
      }
    }
})
)
