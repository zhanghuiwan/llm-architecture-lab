---
layout: home
hero:
  name: LLM 网络结构
  text: 从零实现，逐层理解。
  tagline: 从一个 Token 的预测开始，走过 Transformer、GQA 与 KV Cache，读懂真实模型，用可复现的实验检验每一次设计。
  actions:
    - theme: brand
      text: 从第一章开始 →
      link: /chapters/00-preparation
    - theme: alt
      text: 查看完整学习路线
      link: /guide/
features:
  - title: 先理解，再实现
    details: 用数学、张量形状和最小代码解释同一个计算过程，从单头注意力逐步组装自己的语言模型。
    link: /chapters/02-attention
    linkText: 阅读注意力章节
  - title: 先验证，再比较
    details: 检查因果性、标签对齐和缓存一致性，固定实验预算，区分能力、计算与内存效率。
    link: /experiments
    linkText: 查看实验规范
  - title: 从小模型走向真实架构
    details: 逐步理解现代 Block、MoE 与长上下文，再将配置、公式与 Qwen、DeepSeek 的实现对应起来。
    link: /chapters/12-model-reading
    linkText: 进入源码分析
---

## 一本可以边读边做的架构讲义

本教程面向已有 PyTorch 或模型训练经验、希望系统理解语言模型结构的开发者。学习按先修关系与验收条件推进，不安排固定周数。

**当前版本：首版讲义。** 已提供完整章节路线、基础推导、关键代码与实验设计。尚未提供完整训练工程、预训练权重或实测性能排行榜；文中的实验任务需要读者执行，不能视为已复现结果。

| 阅读路径 | 章节 | 你将完成什么 |
| --- | --- | --- |
| 建立可信的基础 | [00—05](/chapters/00-preparation) | 理解训练目标，组装最小 GPT，建立固定数据与 Tokenizer 的训练基线 |
| 理解现代架构 | [06—09](/chapters/06-modern-block) | 分析现代 Block，推导 GQA 与缓存预算，检验增量推理并测量性能 |
| 走向架构实验 | [10—13](/chapters/10-long-context) | 研究长上下文与 MoE，分析真实模型，形成独立的架构实验报告 |

## 如何判断自己学会了

每章围绕一个问题展开，给出必要概念、张量或公式、动手任务和验收标准。先完成正确性检查，再扩大模型和训练规模。没有硬件时，可以先完成数学推导与小张量验证；训练与性能实验按实际资源选择规模。

阅读 [学习指南](/guide/) 了解先修关系，遇到术语时查阅 [术语表](/glossary)，开始比较结构前阅读 [统一实验规范](/experiments)。
