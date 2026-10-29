import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'LLM 架构实验室',
  description: '从零实现语言模型，逐层理解网络结构，用受控实验检验设计。',
  base: process.env.SITE_BASE || '/',
  cleanUrls: false,
  lastUpdated: true,
  markdown: { math: true, lineNumbers: true },
  head: [['meta', { name: 'theme-color', content: '#176b57' }]],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'LLM 架构实验室',
    nav: [
      { text: '学习路线', link: '/guide/' },
      { text: '开始阅读', link: '/chapters/00-preparation' },
      { text: '实验规范', link: '/experiments' },
      { text: '参考资料', link: '/resources' }
    ],
    sidebar: [
      { text: '阅读之前', items: [{ text: '如何使用本教程', link: '/guide/' }] },
      { text: '第一部分 · 建立可信的基础', items: [
        { text: '00 必要准备', link: '/chapters/00-preparation' },
        { text: '01 语言模型在学什么', link: '/chapters/01-language-modeling' },
        { text: '02 因果多头注意力', link: '/chapters/02-attention' },
        { text: '03 组装最小 GPT', link: '/chapters/03-minimal-gpt' },
        { text: '04 可信训练基线', link: '/chapters/04-training' },
        { text: '05 Tokenizer 与数据管线', link: '/chapters/05-tokenization' }
      ] },
      { text: '第二部分 · 理解现代架构', items: [
        { text: '06 现代 Transformer Block', link: '/chapters/06-modern-block' },
        { text: '07 MHA、MQA 与 GQA', link: '/chapters/07-gqa' },
        { text: '08 自回归推理与 KV Cache', link: '/chapters/08-kv-cache' },
        { text: '09 性能测量与高效实现', link: '/chapters/09-performance' }
      ] },
      { text: '第三部分 · 走向架构实验', items: [
        { text: '10 长上下文实验', link: '/chapters/10-long-context' },
        { text: '11 MoE 入门', link: '/chapters/11-moe' },
        { text: '12 真实模型源码分析', link: '/chapters/12-model-reading' },
        { text: '13 独立架构实验', link: '/chapters/13-research' }
      ] },
      { text: '随手查阅', items: [
        { text: '统一实验规范', link: '/experiments' },
        { text: '术语表', link: '/glossary' },
        { text: '参考资料', link: '/resources' }
      ] }
    ],
    outline: { level: [2, 3], label: '本页内容' },
    search: { provider: 'local', options: { locales: { root: { translations: {
      button: { buttonText: '搜索章节', buttonAriaLabel: '搜索教程' },
      modal: { displayDetails: '显示详细列表', resetButtonTitle: '清除搜索', backButtonTitle: '返回', noResultsText: '未找到相关内容', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } }
    } } } } },
    docFooter: { prev: '上一章', next: '下一章' },
    sidebarMenuLabel: '章节目录',
    darkModeSwitchLabel: '切换深色模式',
    lightModeSwitchTitle: '切换浅色模式',
    darkModeSwitchTitle: '切换深色模式',
    returnToTopLabel: '返回顶部',
    lastUpdated: { text: '最后更新' },
    footer: { message: '原创中文讲义 · MIT License', copyright: '© 2026 zhanghuiwan · LLM Architecture Lab' }
  }
})
