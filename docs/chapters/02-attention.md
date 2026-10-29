# 02 因果多头注意力

本章解决一个问题：一个位置应该从已有的哪些位置读取信息，以及读取多少？先写清计算，再组织为多个头。

## Q、K、V 的作用

设输入 $X\in\mathbb{R}^{B\times T\times D}$。线性投影得到 Query、Key、Value。对单个头：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad Y=AV.
$$

Query 与 Key 决定匹配分数，Value 提供被聚合的信息。softmax 沿 Key 的位置维度计算。缩放有助于控制随头维度增长的分数幅度，但不能代替对数值稳定性的检查。

## 手写一个参考实现

下面的片段展示完整序列、无 padding、标准 MHA 的计算。它不是带缓存或 GQA 的通用实现。

```python
import math
import torch

def causal_attention(q, k, v):
    # q, k, v: [B, H, T, d_h]
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    allowed = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
```

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| Q/K/V | `[B,H,T,d_h]` | 每个头的序列表示 |
| scores/weights | `[B,H,T,T]` | 每个 Query 位置对各 Key 位置的分数／权重 |
| 输出 | `[B,H,T,d_h]` | 聚合后的 Value |

合并各头后得到 `[B,T,D]`，再经过输出投影。每个头可以学习不同的表示子空间，但不应预先假定某个头一定负责语法或指代。

## 正确性检查

最重要的检查不是能否运行，而是没有读取未来。在 eval 模式、相同权重下，只改变位置 $j$ 及之后的 Token，位置 $i<j$ 的输出应保持不变或在数值误差范围内一致。

随后把同一组 Q/K/V 输入手写实现和 PyTorch SDPA，在相同 mask、关闭 dropout 的条件下比较输出。不同 API 对布尔 mask 的约定可能不同，必须查对应接口，不凭名称推断 True 表示允许还是屏蔽。

## 常见错误

- 沿 Query 维而非 Key 维计算 softmax。
- 对已经包含 mask 的实现重复施加错误的因果规则。
- 缓存推理时 Q 长度与 K 长度不同，却沿用本章的方形 mask。
- 只检查输出 shape，没有检查未来信息泄漏。

## 验收标准

能够手算一个很短的序列，解释 mask 的每个位置，通过未来 Token 扰动检查，并与参考实现建立数值对照。
