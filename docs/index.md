---
title: LLM 网络结构：原理、实现与实验
outline: false
---

# LLM 网络结构：原理、实现与实验

这本书面向已有 Python 和模型训练经验、希望理解语言模型内部计算的读者。从下一个 Token 的预测开始，逐步实现一个小型 GPT，再研究现代 Transformer、推理缓存和架构实验。

每章将公式、张量形状与可运行代码放在一起。配套实验使用内置的小数据和 CPU，先验证计算是否正确，再讨论结构的代价与适用范围。学习按先修关系推进，不设置周计划。

[阅读指南](/guide/) · [安装与运行实验](/labs) · [GitHub](https://github.com/zhanghuiwan/llm-architecture-lab)

## 第一部分：基础模型

- [00 必要准备](/chapters/00-preparation)：张量、广播、数值稳定与梯度。
- [01 语言模型在学什么](/chapters/01-language-modeling)：预测目标、标签对齐与交叉熵。
- [02 因果多头注意力](/chapters/02-attention)：从矩阵计算到因果性检查。
- [03 组装最小 GPT](/chapters/03-minimal-gpt)：完整模型、训练与生成。
- [04 可信训练基线](/chapters/04-training)：梯度累积、随机性与断点恢复。
- [05 Tokenizer 与数据管线](/chapters/05-tokenization)：字节 BPE、文档边界与样本构造。

## 第二部分：现代结构与推理

- [06 现代 Transformer Block](/chapters/06-modern-block)：RMSNorm、RoPE、SwiGLU 与权重绑定。
- [07 MHA、MQA 与 GQA](/chapters/07-gqa)：头分组、参数与缓存预算。
- [08 自回归推理与 KV Cache](/chapters/08-kv-cache)：完整前向与增量解码的一致性。
- [09 性能测量与高效实现](/chapters/09-performance)：正确性、计时范围与测量误差。

## 第三部分：架构实验

- [10 长上下文实验](/chapters/10-long-context)：局部注意力、信息可达性与任务测试。
- [11 MoE 入门](/chapters/11-moe)：路由、专家分派与容量限制。
- [12 真实模型源码分析](/chapters/12-model-reading)：从配置追踪 Qwen 与 DeepSeek 的计算路径。
- [13 独立架构实验](/chapters/13-research)：固定预算、配对种子与实验报告。

## 阅读与参与

本书不要求大模型训练集群，但需要理解梯度、优化器和训练／验证划分。微型实验用于理解局部机制，不能据此判断大模型能力或 GPU 性能。

遇到术语可以查阅[术语表](/glossary)，开展对照前阅读[统一实验规范](/experiments)。勘误与补充请见[贡献说明](https://github.com/zhanghuiwan/llm-architecture-lab/blob/main/CONTRIBUTING.md)和[问题列表](https://github.com/zhanghuiwan/llm-architecture-lab/issues)。本书与实验以 MIT 许可开放，引用资料遵循各自许可。
