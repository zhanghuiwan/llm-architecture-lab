# 安装与运行实验

每章的 `main.py` 都是完整的实验入口，包含必要模型、内置小数据和检查，不需要先执行其他章节。代码采用少量重复以便独立阅读；这与生产系统中的共享模块组织不同。

## 准备环境

需要 Python 3.12。CPU 即可运行主线实验，无需下载训练集或模型权重。以下命令适用于 Linux/macOS；Windows 请将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

```bash
git clone https://github.com/zhanghuiwan/llm-architecture-lab.git
cd llm-architecture-lab
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

CPU 索引避免为本书的小实验安装 CUDA 依赖。若使用现有 GPU 环境，需自行核对 PyTorch、驱动和硬件兼容性，GPU 性能不属于本书 CPU 检查的验证范围。

## 运行一章

在仓库根目录执行：

```bash
.venv/bin/python experiments/02-attention/main.py --quick
.venv/bin/python experiments/02-attention/main.py --seed 7
.venv/bin/python experiments/02-attention/main.py --quick --output experiment-results/attention.json
```

`--quick` 使用较小规模完成核心检查；不加该参数运行本章的常规规模。`--seed` 固定该次运行使用的随机种子，不能消除跨平台和跨版本差异。`--output` 将同一份 JSON 报告写到指定路径。报告中的 `checks` 是实际执行的布尔检查，`metrics` 记录误差、损失、参数量或测量结果。任何检查失败都应返回非零退出码。

## 全部运行

```bash
.venv/bin/python experiments/run_all.py --quick --output experiment-results/all-quick.json
.venv/bin/python experiments/run_all.py --output experiment-results/all-standard.json
```

总入口为每章创建独立的临时工作目录，使用同一个 Python 解释器运行。它检查退出码、JSON 格式、种子、非空检查集合和有限数值，最后返回通过的章节数。结果文件包含运行环境和各章原始报告；不要只保留一个“全部成功”的截图。

## 实验目录

| 章 | 入口目录 | 核心内容 |
| --- | --- | --- |
| 00 | `experiments/00-preparation/` | 张量、广播与梯度差分 |
| 01 | `experiments/01-language-modeling/` | 标签对齐、交叉熵与微型语言模型 |
| 02 | `experiments/02-attention/` | 因果注意力、前向与梯度对照 |
| 03 | `experiments/03-minimal-gpt/` | 完整 tiny GPT 与小数据训练 |
| 04 | `experiments/04-training/` | 梯度累积与断点恢复 |
| 05 | `experiments/05-tokenization/` | 字节 BPE 与数据边界 |
| 06 | `experiments/06-modern-block/` | 归一化、旋转位置编码与门控 |
| 07 | `experiments/07-gqa/` | 头分组与参数／缓存预算 |
| 08 | `experiments/08-kv-cache/` | 单步和分块增量解码 |
| 09 | `experiments/09-performance/` | 数值对照与实际计时 |
| 10 | `experiments/10-long-context/` | 局部注意力与信息传播 |
| 11 | `experiments/11-moe/` | 路由、专家与容量 |
| 12 | `experiments/12-model-reading/` | 微型模型路径与 MLA 代数 |
| 13 | `experiments/13-research/` | 配对训练与预算控制 |

每章正文给出该实验的原理、参数与结果解读，完整源码可在[仓库 experiments 目录](https://github.com/zhanghuiwan/llm-architecture-lab/tree/main/experiments)阅读。

## 结果的边界

小模型的损失下降只能证明对应实现能优化这份数据，不能证明具备通用语言能力。合成任务不能替代真实长上下文评测。CPU 计时不能推断 CUDA 内核的速度，理论缓存字节数也不等于实际进程的峰值内存。扩展实验时，请按[统一实验规范](/experiments)记录主要预算与额外差异。
