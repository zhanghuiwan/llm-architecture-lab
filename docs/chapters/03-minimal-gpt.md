# 03 组装最小 GPT

把注意力接到完整网络中，建立从 Token ID 到训练损失的路径。首版先使用易理解的绝对位置 Embedding、LayerNorm 和普通 FFN，后面再逐项替换。

## 完整计算路径

```text
Token IDs [B,T]
  → Token Embedding + Position Embedding [B,T,D]
  → 多个 Transformer Block [B,T,D]
  → 最终归一化 [B,T,D]
  → LM Head [B,T,V]
  → 与右移标签计算交叉熵
```

Pre-Norm Block 可以写成：

$$
H'=H+\operatorname{Attention}(\operatorname{Norm}_1(H)),
$$

$$
H_{out}=H'+\operatorname{FFN}(\operatorname{Norm}_2(H')).
$$

Attention 在位置之间聚合信息；FFN 在每个位置上进行非线性变换。残差路径让模块学习对当前表示的更新，不自动保证任意深度都稳定。

## 一个 Block 的骨架

```python
from torch import nn

class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x, blocked):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=blocked, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))
```

这里的 `blocked` 为 `[T,T]` 的上三角布尔 mask，未来位置为 True；这是 `nn.MultiheadAttention` 的约定。前章手写注意力用于理解，本章使用库模块帮助检查组装过程。请再用自己实现的 Attention 替换它，验证两条路径一致。

## 从小到大的实现顺序

1. 构造词表与 Token/Position Embedding。
2. 只放一个 Block，检查 shape、梯度和因果性。
3. 加入最终归一化、LM Head 和右移标签。
4. 在极小数据上过拟合，确认模型确实能够学习。
5. 编写完整重算上下文的生成循环，先不用缓存。

## 参数量估算

标准 MHA、FFN 中间维度 $4D$、忽略偏置与归一化时，每层约为 $12D^2$ 个参数。若输入 Embedding 与输出头共享权重：

$$
P\approx 12LD^2+VD+T_{max}D.
$$

最后一项来自这里使用的可学习绝对位置表。若输出头不共享权重，再加 $VD$；使用 RoPE 时则没有这张可学习位置表。估算要与实际参数统计核对。

## 验收标准

能解释完整数据流；因果性检查通过；梯度有限；极小数据可以过拟合；能够自回归生成，并明确这不等于已经获得聊天能力。
