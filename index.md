---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "星间飞行"
  text: "Python · C++ · AI "
  tagline: "星际航行，智慧笔记"
  image:
    src: /background.png
    alt: 背景图片
  actions: 
    - theme: brand
      text: 快速开始 →
      link: /notes/intro
    - theme: alt
      text: 收藏夹
      link: /notes/starred

features:
  - icon: 🐍
    title: 编程语言
    details: 涵盖Python语法/进阶、C++核心特性、STL实战、跨语言编程技巧，附可直接运行的代码案例
    link: /notes/language/
    linkText: 查看笔记

  - icon: 🤖
    title: AI模型与算法
    details: 拆解主流AI模型（LLM/CNN/RNN）原理、训练调优、推理部署，从理论到工程落地
    link: /notes/ai-model/
    linkText: 深入学习

  - icon: ⚡
    title: 深度学习实战
    details: PyTorch/TensorFlow框架使用、数据预处理、模型训练、性能优化、实战项目复盘
    link: /notes/deep-learning/
    linkText: 实战案例

  - icon: 📝
    title: 刷题与算法
    details: 基于Python/C++的经典算法题、LeetCode刷题思路、数据结构优化技巧
    link: /notes/algorithm/
    linkText: 刷题笔记

  - icon: 📚
    title: 学习路线图
    details: Python/C++/AI/深度学习的系统学习路径，从入门到进阶的阶段目标与资源推荐
    link: /notes/roadmap/
    linkText: 规划路径

  - icon: 🛠️
    title: 工具与环境
    details: 开发环境配置、调试技巧、依赖管理、AI模型部署工具（Docker/TensorRT）使用教程
    link: /notes/tools/
    linkText: 工具指南
---

