# 06 现代 Transformer Block

本章逐项替换最小 GPT 的组件。先理解计算改变，再判断参数预算与训练表现；不要一次替换全部组件后，给某一个组件归功。

## RMSNorm

$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}}\odot\gamma.
$$

它不减去均值，使用均方根缩放。与 LayerNorm 比较时，说明归一化维度、epsilon、是否带偏置以及累加精度。减少算子不一定意味着端到端训练明显加速，仍需测量。

## RoPE

RoPE 在每个头的通道对上按位置施加旋转。对一对坐标：

$$
R_\theta\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}a\cos\theta-b\sin\theta\\a\sin\theta+b\cos\theta\end{bmatrix}.
$$

位置进入旋转角度，配对方式、频率、旋转维度和位置偏移都属于实现契约。不要把“公式可以计算更大位置”理解成“未经训练就能可靠使用任意长上下文”。

## SwiGLU 与参数预算

$$
\operatorname{SwiGLU}(x)=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d.
$$

中间维度为 $M$ 时，忽略偏置的参数量约为 $3DM$。普通两层 FFN 的参数量约为 $2DM$。因此，同一个中间维度并不是等参数比较；与 $4D$ 的普通 FFN 匹配时，SwiGLU 可用约 $8D/3$ 的中间维度，再按实现要求取整。

## 权重共享与 QK-Norm

Token Embedding 与 LM Head 共享权重可以减少约 $VD$ 参数，但模型设计可能有不同取舍。QK-Norm 则针对 Query/Key 表示进行归一化；例如 Qwen3 的实现包含独立的 Q/K RMSNorm，不能仅用四个常见组件概括全部结构。

## 动手任务

以固定基线分别替换 RMSNorm、RoPE、SwiGLU。每项记录参数变化、数值检查和验证 loss。先单因素实验，再研究组合效果；不同归一化与残差配置可能需要重新审视初始化和学习率。

## 验收标准

能够写出组件公式、对应 shape、参数预算与测试方法，并区分观察到的效果与尚未验证的解释。
