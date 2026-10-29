# 术语表

| 术语 | 本教程中的含义 | 相关章节 |
| --- | --- | --- |
| Token | 模型处理的离散单位，不必等同于一个字或单词 | [01](./chapters/01-language-modeling.md)、[05](./chapters/05-tokenization.md) |
| Logits | 归一化前的预测分数 | [01](./chapters/01-language-modeling.md) |
| Causal Mask | 阻止位置读取未来信息的访问约束 | [02](./chapters/02-attention.md) |
| Pre-Norm | 在子层计算前归一化输入的残差结构 | [03](./chapters/03-minimal-gpt.md) |
| RoPE | 用位置相关旋转作用于 Q/K 的位置编码方法 | [06](./chapters/06-modern-block.md) |
| RMSNorm | 基于均方根进行缩放的归一化 | [06](./chapters/06-modern-block.md) |
| SwiGLU | 使用 SiLU 门控与逐元素乘法的 FFN 形式 | [06](./chapters/06-modern-block.md) |
| GQA | 多个 Query 头共享一组 KV 头的组织方式 | [07](./chapters/07-gqa.md) |
| KV Cache | 自回归推理中复用历史 Key/Value 或相应缓存状态 | [08](./chapters/08-kv-cache.md) |
| Prefill | 处理提示词并建立缓存的阶段 | [08](./chapters/08-kv-cache.md) |
| Decode | 利用已有上下文逐步生成 Token 的阶段 | [08](./chapters/08-kv-cache.md) |
| FlashAttention | 通过分块与数据访问优化实现注意力的算法族 | [09](./chapters/09-performance.md) |
| SWA | 仅访问局部滑动窗口的注意力模式 | [10](./chapters/10-long-context.md) |
| MoE | 由路由器将 Token 分配给部分专家的结构 | [11](./chapters/11-moe.md) |
| MLA | DeepSeek 提出的多头潜在注意力设计 | [12](./chapters/12-model-reading.md) |

术语用于明确本书讨论范围，不替代具体模型与实现的定义。遇到不同论文使用同名概念时，应核对其计算与约束。
