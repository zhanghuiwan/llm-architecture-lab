# 12 把真实模型的配置、公式与源码对应起来

前面的章节分别研究了现代 Block、头分组、推理缓存和专家路由。现在需要把这些局部知识重新放回一个实际项目：配置决定投影矩阵的形状，张量布局决定归一化和位置编码的轴，推理实现决定缓存中究竟保存什么。真实模型阅读的产物应该是一份能逐项追溯的计算说明，而不是把仓库中的类名重新列一遍。

本章选择两个固定来源：Transformers `v4.51.3` 的 Qwen3 Dense 实现，以及 DeepSeek-V3 官方仓库的固定提交。配套实验不下载模型，不需要 Transformers 或官方推理工程，只用 CPU 和 PyTorch 验证两个关键关系：一个缩小的 Qwen 风格 Block 与参考注意力是否有相同的前向和梯度；一个简化 MLA 的显式展开与权重吸收是否在代数上等价。

## 学习目标与先修关系

先完成第 06 章的 RMSNorm、RoPE、SwiGLU，第 07 章的 GQA 和第 08 章的缓存语义。第 11 章的 MoE 帮助你读懂 DeepSeek 的 FFN 部分，但本章实验不会重新实现整个专家系统。

读完后应能从配置推导 Q/K/V/O 投影的形状，定位每个归一化的作用轴，区分模型结构与内核实现，并明确列出自己验证的范围。你还应能写出 MLA 的潜变量缓存预算，解释为何不能简单套用 GQA 的缓存公式。这些目标不要求下载数百 GB 权重，也不要求复现原报告中的能力或吞吐数字。

## 1. 先固定身份，再开始阅读

“Qwen 架构”或“DeepSeek 架构”不是足够精确的对象。同一个系列可以有 Dense、MoE、不同头数、不同上下文方案和不同推理后端。建议在阅读记录开头列出代码版本、配置来源、权重身份、精度、内核和许可。

| 对象 | 本章固定来源 | 本章实际读取的内容 |
| --- | --- | --- |
| Qwen3 Dense 的 Transformers 实现 | tag `v4.51.3`，Git 引用 `b99fa8251c7327130243a0b76e36d0a922492e41` | `configuration_qwen3.py`、`modeling_qwen3.py`、仓库许可 |
| DeepSeek-V3 官方推理实现 | commit `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` | `inference/model.py`、`inference/configs/config_671B.json`、`LICENSE-CODE` |
| 本章教学模型 | 本仓库 `experiments/12-model-reading/main.py` | 随机初始化的小型 FP64 计算，无官方权重 |

Qwen 案例分析的是这个版本的实现及配置类，不把配置类的默认值声称为某个已发布检查点的真实配置。分析具体的 Qwen3-某参数规模检查点时，还必须取得该检查点固定 revision 的 `config.json`，覆盖类默认值，并记录权重版本。本章没有加载或验证任何官方权重；这条边界会影响你可以得出的结论。

固定链接可从[Qwen3 配置](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/configuration_qwen3.py)、[Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)、[DeepSeek-V3 实现](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py)和[671B 配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json)打开。阅读时尽量使用固定链接，避免网页默认的 `main` 随时间移动。

## 2. 沿配置进入 Qwen3 的张量路径

从 `Qwen3Config` 查找 `hidden_size`、`num_attention_heads`、`num_key_value_heads`、`head_dim` 和 `intermediate_size`。本版本配置类默认值分别是 4096、32、32、128 和 22016，`attention_bias=False`，`rms_norm_eps=1e-6`。这只是代码中可核对的默认设置，不是所有 Qwen3 模型的共同数值。

最容易被忽略的是 `head_dim`。前面的最小 GPT 常取 $d_h=D/H_q$；这里的实现优先读取独立的 `head_dim` 字段。设模型宽度为 $D$，Query 头数为 $H_q$，KV 头数为 $H_{kv}$，头维度为 $d_h$，则投影的输出宽度为：

$$
D_Q=H_qd_h,\qquad D_K=D_V=H_{kv}d_h.
$$

没有数学定律要求 $D_Q=D$。`o_proj` 会把 $H_qd_h$ 重新映射回 $D$，以便残差相加。若沿用旧代码，先把 Q reshape 为 `[B,T,Hq,D//Hq]`，就可能在真实配置上直接报错，或者错误地截断／重排数据。

本章的快速配置特意取 $D=20,H_q=4,H_{kv}=2,d_h=6$：Q 的投影宽度是 24，KV 的宽度是 12，`D//Hq=5` 与真正的头维度不相等。这样的配置更容易暴露把简化假设当成通用规则的代码。它是教学配置，不对应一个已发布的 Qwen 检查点。

| 路径 | 输入与输出 | 阅读时应核对的问题 |
| --- | --- | --- |
| `Qwen3Model` 的嵌入层 | Token ID 到 `[B,T,D]` | 词表与隐藏宽度，是否有额外位置表 |
| `Qwen3DecoderLayer.input_layernorm` | `[B,T,D]` 不变 | RMSNorm 的最后一维是 D |
| `Qwen3Attention.q_proj/k_proj/v_proj` | 到 `Hq*dh` 或 `Hkv*dh` | bias 与独立 `head_dim` |
| `q_norm/k_norm` | `[B,T,H,dh]` 不变 | 沿每个头的 dh，而不是整个投影宽度 |
| 旋转位置编码与注意力 | 多头表示到 `[B,T,Hq,dh]` | 配对布局、位置、GQA 映射与 mask |
| `o_proj` 与第一次残差 | `[B,T,Hq*dh]` 到 `[B,T,D]` | 输出宽度与残差宽度匹配 |
| `post_attention_layernorm`、`Qwen3MLP` | `[B,T,D]` 到 `[B,T,D]` | 两个上投影、SiLU 门和下投影 |
| 模型末尾 norm 与语言模型头 | 到词表 logits | 是否共享 embedding 权重，不能靠类名猜测 |

本版本 `tie_word_embeddings` 默认是 `False`。前面小模型中的绑定节省不能直接套到具体检查点；需要同时检查配置、模型初始化和加载后的参数身份。阅读到这一层才算把“模型由哪些模块组成”推进为“具体权重如何参与计算”。

## 3. QK 归一化的轴比名称更重要

`q_norm` 与 `k_norm` 都是长度为 `head_dim` 的可学习向量，作用在每个位置、每个头的最后一维：

$$
\widehat q_{b,t,h,j}=\gamma_j\frac{q_{b,t,h,j}}
{\sqrt{d_h^{-1}\sum_{r=1}^{d_h}q_{b,t,h,r}^2+\epsilon}}.
$$

这里没有减均值。权重 $\gamma$ 在位置和各个头之间共享，但每个头根据自己的均方值归一化。若先把全部头展平，再在 $H_qd_h$ 上计算均方值，就让一个头的尺度依赖其他头。输出形状仍然正确，因此 shape 检查不能发现这种错误。

本章负向对照保留全部维度，却将均方值错误地跨头计算；在快速 seed42 的实际运行中，最终 Block 输出的最大差异约为 `0.159313`。这个数字表示该错误在当前随机输入上确实可检测，不是模型能力损失，也不是对训练稳定性的量化结论。

官方 `Qwen3RMSNorm` 会把输入转成 FP32 计算均方值，再转回输入精度。本章为了检查代数关系使用 FP64，并在 FP64 中归一化。两者的数学目标对应，浮点舍入路径不同。本文的误差不能解释为“与官方半精度推理逐位一致”；如果要验证混合精度实现，应另建相同 dtype、相同累积规则的参考。

## 4. RoPE 的坐标约定也需要追踪

Qwen3 这个版本的 `rotate_half` 将向量分成前后两半，用前半部分和后半部分的对应坐标配对。另一些实现使用相邻偶奇坐标配对。两种布局可以通过一致的坐标置换对应，但不能只替换旋转函数而保留不匹配的权重布局。

本章使用 split-half 布局。若 $d_h=6$，配对是 `(0,3)`、`(1,4)`、`(2,5)`。位置 p 对应的频率由偶数指数生成，再复制到前后两半。对 Q 和 K 使用同样的布局和位置后，点积可以表达相对位置关系。注意“使用同样的位置”指每个 Token 的绝对位置一致，并不意味着所有位置都传入零。

源码中的位置处理还可能包含 `rope_scaling`、不同后端和缓存位置。源码存在某种扩展逻辑，不代表每个检查点都启用，也不代表改大配置中的最大长度就具有长上下文能力。本章仅使用基础频率和完整的因果序列；缓存与偏移的系统检查见第 08 章，外推与任务边界见第 10 章。

## 5. 用一个完整 Block 检查路径

本章 `TinyQwenBlock` 包括两次隐藏维 RMSNorm、QK 每头 RMSNorm、基础 RoPE、GQA、注意力输出投影和 SwiGLU，两次残差都参与计算。它不包含 embedding、最终词表头、Tokenizer 或生成调度，因此不会被命名为完整 Qwen3 模型。

我们写两条路径：主路径不展开 KV，用分组 `einsum` 表达 GQA；参考路径将 KV 按组 `repeat_interleave`，交给 PyTorch SDPA。两条路径复制相同参数、输入和标量探针，对输出求同一线性标量损失，再比较输入梯度与全部参数梯度。这样能发现仅靠前向随机样本容易遗漏的投影、广播或归一化错误。

快速配置的参数量也可以手算：

$$
P=2D+2d_h+2DH_qd_h+2DH_{kv}d_h+3DF.
$$

$F$ 是 SwiGLU 中间宽度。本式不含 bias、embedding 和词表头。各项依次是两次隐藏归一化、两次头归一化、Q/O 投影、K/V 投影和三个 FFN 矩阵。代入 $D=20,H_q=4,H_{kv}=2,d_h=6,F=32$，得到 `3412`，与代码遍历参数的结果相同。参数计数对应这个 Block，不能拿来与完整模型的总参数直接比较。

除了数值参考，实验还扰动最后一个位置，确认更早位置的输出不变。参考对照和因果性质是不同检查：如果两条路径错误地使用相同的双向 mask，它们可能彼此一致，但仍然违反语言模型的因果要求。

## 6. 进入 DeepSeek-V3：先读具体配置

在固定版本的 `config_671B.json` 中，可以读到以下字段：`dim=7168`、`n_layers=61`、`n_heads=128`、`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`。权重类型字段为 `fp8`，它不能单独决定运行时缓存或所有中间张量的 dtype。

这份配置还指定 256 个路由专家、1 个共享专家、每 Token 激活 8 个路由专家，`n_dense_layers=3`。源码 `Block` 根据层号在 `MLP` 与 `MoE` 之间选择；不能把 61 层都按相同的 MoE 计算。路由配置还包含分组限制、sigmoid 分数和 `route_scale=2.5`，第 11 章简单 Top-k 的行为不等于这个完整路由器。

| 源码位置 | 阅读任务 |
| --- | --- |
| `ModelArgs` 与 `config_671B.json` | 区分 dataclass 默认值与具体运行配置覆盖值 |
| `MLA.__init__` | 找到 Query 分支、KV 低秩分支、位置分支和输出投影 |
| `MLA.forward` 的 `naive` 路径 | 观察潜变量如何展开为每个头的 K/V |
| `MLA.forward` 的 `absorb` 路径 | 观察哪些权重被移到 Query 和输出侧 |
| `Gate`、`MoE`、`Block` | 区分路由权重、共享专家、实际执行和层级选择 |

`ModelArgs` 的某些默认专家数量与 671B 配置不同，这正说明只读类定义不够。阅读顺序应该是启动参数如何加载配置、配置如何传到模块、模块怎样定义权重、前向最终选择哪条分支。

## 7. MLA 的潜变量与位置分支

标准 GQA 仍为若干 KV 头缓存展开后的 K/V。MLA 的关键是用一个共享低维潜变量表示内容部分，并将位置部分单独保存。下面只研究单层、无 padding、没有张量并行的简化形式。

设每个位置的潜变量为 $c_s\in\mathbb R^r$，第 h 个头的内容 Key 和 Value 来自两个上投影：

$$
k^C_{h,s}=c_sU^K_h,\qquad v_{h,s}=c_sU^V_h,
$$

其中 $U^K_h\in\mathbb R^{r\times d_n}$，$U^V_h\in\mathbb R^{r\times d_v}$。官方实现的潜变量来自下投影后的 RMSNorm；本章也保留这一归一化，然后讨论固定 c 之后的线性关系。

Query 分为内容部分 $q^C_{h,t}$ 和带 RoPE 的位置部分 $q^R_{h,t}$；Key 的位置部分 $k^R_s$ 在头之间共享。分数是：

$$
s_{h,t,s}=\frac{q^C_{h,t}(k^C_{h,s})^\top+
q^R_{h,t}(k^R_s)^\top}{\sqrt{d_n+d_r}}.
$$

这里 $d_r$ 是位置分支宽度。不能把缩放分母误写成 $\sqrt r$：改变计算顺序后，原始分数定义中的尺度不会自动随潜变量维度改变。因果 mask、softmax 和每个位置的权重定义也要保持一致。

RoPE 项需要单独保留，是因为位置相关的旋转会改变算子的形式。对所有展开 Key 的完整维度先旋转，再假设有一个与位置无关的固定权重可以随意吸收，一般不成立。解耦后的内容上投影不含这种位置依赖，而小的旋转位置分支显式参与点积。

教学 MLA 为减少辅助代码，复用本章的 split-half 旋转；固定版本 DeepSeek 的 `apply_rotary_emb` 使用复数视图，对相邻坐标配对。两者描述相同类型的二维旋转，但坐标排列不同。要对接官方权重必须一致转换投影坐标，不能直接替换旋转函数。本章两条随机参数路径使用同一套布局，因此检验的是展开与吸收的代数等价，没有声称权重布局与官方逐项兼容。

## 8. 权重吸收：改变计算顺序，保留函数

把内容 Key 的定义代入点积，利用矩阵乘法的结合律：

$$
q^C_{h,t}(c_sU^K_h)^\top
=\left(q^C_{h,t}(U^K_h)^\top\right)c_s^\top.
$$

左边先对每个历史位置展开 Key，再计算点积；右边把 Query 映射到潜变量空间，用潜变量 c 直接计算分数。这样推理缓存不必保存每个头展开的内容 Key，但仍需保存每个位置归一化后的 c 和位置 Key。

设 softmax 后的注意力权重是 $a_{h,t,s}$，Value 聚合也能改写：

$$
\sum_s a_{h,t,s}(c_sU^V_h)
=\left(\sum_s a_{h,t,s}c_s\right)U^V_h.
$$

左边先展开所有历史 Value；右边先在潜变量空间聚合，再执行每个头的 Value 上投影。随后仍需合头和输出投影。教材里的“吸收”是这两条代数改写，不意味着所有权重都消失，也不意味着任何顺序都有更好的实际吞吐。

实验使用两份相同的叶子张量，分别构造显式展开与吸收路径。我们比较未加 mask 的分数、softmax 权重、最终输出，以及输入、下投影、Query 投影、Key/Value 上投影和输出矩阵的全部梯度。直接比较未屏蔽分数还能避免对 `-inf - -inf` 求误差产生 NaN。

## 9. 缓存预算要与实际存储对应

对批量 B、缓存长度 T、头数 H、每个元素 e 字节，显式展开的 Key 包含内容和位置，Value 单独保存，教学形式的缓存 payload 是：

$$
M_{expanded}=BT H(d_n+d_r+d_v)e.
$$

共享潜变量与共享位置 Key 的缓存是：

$$
M_{latent}=BT(r+d_r)e.
$$

快速实验采用 $B=2,T=5,H=4,r=5,d_n=6,d_r=4,d_v=6$，FP64 每个元素 8 字节。公式得到展开缓存 `5120` 字节、潜变量缓存 `720` 字节，比例约 `7.111`。代码实际创建这两种缓存张量，并以 `numel()*element_size()` 累计 payload，核对与公式一致。这里不是通过只算公式就声称做过内存测量。

如果仅将 671B 配置代入，并假设两种表示都使用每元素 2 字节，那么单 Token、单层、逻辑全头的展开形式为 80 KiB，潜变量形式为 1152 字节。这个条件计算不代表官方实际进程的内存占用。多层、已分配最大容量、分页碎片、临时张量、权重、通信缓冲、量化 scale 都会改变实际需求。

张量并行时还要区分逻辑全头与每个 rank 的本地头。潜变量可能在 rank 之间重复存储，而展开表示的头可能被切分；不能把单层无并行的比例直接当成整个集群的节省倍数。FP8 权重也不自动意味着缓存每个元素一字节。先找到 `register_buffer` 的形状，再追踪运行时 dtype 和分片策略，才有资格报告系统预算。

## 10. 运行本章实验

按[实验指南](/labs)安装依赖，在仓库根目录执行：

```bash
.venv/bin/python experiments/12-model-reading/main.py --quick
.venv/bin/python experiments/12-model-reading/main.py
.venv/bin/python experiments/12-model-reading/main.py --quick --seed 7
.venv/bin/python experiments/12-model-reading/main.py --quick --output experiment-results/model-reading.json
```

完整代码位于[本章独立实验](https://github.com/zhanghuiwan/llm-architecture-lab/blob/main/experiments/12-model-reading/main.py)。脚本只依赖标准库与 PyTorch，内置全部配置和随机小张量。入口使用绝对脚本路径时可从任意工作目录运行，不读取其他章节文件。

作者在本项目的 Python 3.12、PyTorch `2.8.0+cpu` 环境中实际执行快速、常规及 seed7，三次均通过 37 项检查。快速 seed42 的 Qwen Block 前向最大误差约 `4.44e-16`，参数梯度最大误差约 `1.78e-15`，未来位置扰动造成的前缀误差为零。MLA 输出误差约 `4.44e-16`，全部被检查梯度中的最大误差约 `1.24e-14`。

常规配置增加序列、头维度和部分模型宽度，仍通过 37 项检查；展开缓存与潜变量缓存的实际张量 payload 为 `12096` 与 `1232` 字节。数值会随版本和设备变化，验收依据是代码中的阈值和性质，不要求打印的最后一位完全相同。这些运行没有训练任务，没有测量 GPU，也没有验证官方推理吞吐。

## 11. 常见错误与定位顺序

如果 Q 的 reshape 失败，先比较 `hidden_size`、`num_attention_heads` 和实际 `head_dim`，再看投影输出；不要通过截断向量来迁就旧假设。如果维度合法但参考前向不一致，依次检查 QK norm 的轴、RoPE 配对、GQA 头映射和缩放因子。

如果前向一致而梯度不同，确认复制的是相同参数数值、损失探针相同、没有在一条路径中意外 `detach`，并检查广播参数的梯度是否需要累加。仅比较输出不足以验证可训练实现，尤其是包含共享潜变量和上投影的路径。

如果 MLA 分数不一致，先关闭 softmax，逐项比较内容项和位置项；若内容项一致而位置项不同，优先检查 RoPE 的坐标布局和位置索引。如果分数一致而输出不同，检查 Value 上投影的方向以及合头顺序。不要把所有差异都归因于浮点精度。

如果缓存预算看起来惊人地小，检查你算的是有效长度还是预分配容量、单层还是全模型、单 rank 还是全系统、payload 还是峰值内存。比较方法应写在数字旁边，而不是放在结果发布后才补充。

## 12. 练习与验收

1. **换一个合法的独立头维度。** 保持 D 和头数不变，将快速配置的 `head_dim` 改为另一个偶数，重新推导 Q/K/V/O 形状和参数量。提示：输出投影的输入宽度也会改变，残差宽度仍为 D。
2. **构造归一化轴反例。** 让某个头的幅度放大，观察正确 per-head norm 与跨头 norm 对其他头的影响。提示：先比较归一化后的 Q，再比较 Block 输出，明确传播经过哪些矩阵。
3. **手推 MLA 的梯度。** 选一个标量输出，对某个 $U^K_h$ 元素做有限差分，与两条代数路径的 autograd 比较。提示：避免把 softmax 的权重视为常数；参数会同时改变分数和最终聚合。
4. **故意使用错误缩放。** 将吸收路径分母改为 $\sqrt r$，检查分数和概率先后在哪一步偏离。提示：该改动不会改变 shape，但不是原函数的等价重排。
5. **完成一个真实检查点身份表。** 选择一个合法可访问的 Qwen3 Dense 检查点，固定配置 revision，记录头维度、KV 头、词表共享和 RoPE 设置。提示：配置阅读与权重推理要分开写；未加载权重时不能报告生成能力。
6. **扩展缓存预算表。** 列出多个潜变量 rank 和上下文长度的理论 payload，再说明你没有测量的系统部分。提示：先固定 dtype 和分片假设，缓存节省不能直接推出端到端加速。

本章验收是：交付一张可定位到固定源码的计算路径表，运行教学实验并解释每个对照，准确区分源码阅读、代数验证、训练和性能测试。对简化的边界能够清楚说明，比给模型贴一个未经验证的“已复现”标签更有价值。

## 来源与许可

本章的教学代码根据公式独立编写，没有直接复制官方模型工程。Transformers 固定版本采用 [Apache License 2.0](https://github.com/huggingface/transformers/blob/v4.51.3/LICENSE)；DeepSeek-V3 的代码许可见 [MIT LICENSE-CODE](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/LICENSE-CODE)。模型权重另有各自的许可与使用条件，代码许可不能替代权重许可。

进一步阅读 [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)和 [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)。本章没有复现官方低精度内核、FP8 权重量化、YaRN、Query 低秩的完整可配置分支、专家并行、训练配方或原报告评测。MLA 缓存的代数检查只是理解这些系统之前的一块基础。
