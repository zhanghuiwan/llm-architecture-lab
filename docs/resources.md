# 参考资料

优先阅读原始课程、论文与官方实现。以下资料用于支持章节学习，性能和模型能力应回到对应报告的实验条件理解。

## 基础实现

- [Karpathy：Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)：字符语言模型与注意力的入门实现。
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)：按提交演进的 GPT-2 风格复现；其 README 对应另一段后续复现视频，不与上一条混作同一配套项目。
- [Stanford CS336](https://cs336.stanford.edu/) 与 [基础作业](https://github.com/stanford-cs336/assignment1-basics)：Tokenizer、模型、训练与后续系统知识。
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/4)：模型类型和工具生态的补充说明。

## 真实模型

- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3) 与 [技术报告](https://arxiv.org/abs/2505.09388)。
- [Qwen3 的固定版本 Transformers 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)：用于从配置追踪计算路径。
- [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) 与 [技术报告](https://arxiv.org/abs/2412.19437)。
- [ZGCM-1 官方仓库](https://github.com/zgcagi/ZGCM-1)：混合注意力的扩展案例；简化 SWA 实验不等于完整复现该模型。

## 系统与教学参考

- [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：并行训练与系统实现的进阶阅读。
- [《动手学深度学习》](https://zh.d2l.ai/)：参考其概念、代码、练习结合的教学方式。本教程为独立编写的首版讲义，未复制该书章节。

## 引用与更新约定

真实模型分析需要固定源码版本和具体变体。更新资料后检查公式、配置与实现是否仍一致；第三方资源可能变更，不将链接可访问等同于已复现实验。
