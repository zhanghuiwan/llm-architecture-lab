# 贡献与本地开发

欢迎提交勘误、推导补充、独立实验和可复现的负面结果。请先在 Issue 中说明较大的选题调整，修正文句或计算错误可直接提交 Pull Request。

## 章节约定

每章说明学习目标与先修关系，明确公式中的符号和张量形状，解释实现取舍，提供实验步骤、结果解读、常见错误以及带提示的习题。以中文为主，保留代码标识符。

`experiments/章节名/main.py` 是该章完整的可运行入口，只依赖 Python 标准库和声明的依赖；不导入其他章节、不下载数据或权重，不依赖启动目录。内置语料应为自行编写或许可明确的教学数据。

脚本支持 `--quick`、`--seed`、`--output`，标准输出为一个包含 `chapter`、`seed`、非空布尔字典 `checks` 和 `metrics` 的 JSON 对象。检查失败须返回非零退出码。所有报告数字应有限；计时记录环境和测量范围，不承诺优化实现一定更快。

## CPU 实验环境

需要 Python 3.12，下面命令适用于 Linux/macOS；Windows 请使用虚拟环境的 `Scripts/python.exe`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/run_all.py --quick
```

新实验应通过快速、常规规模和至少一个不同随机种子的检查。计时、误差和损失必须来自实际运行。说明微型数据、精度、硬件和实现边界；未执行的 GPU 或分布式路径应明确标记。

## 网站开发

需要 Node.js 22 或更新版本与 npm。

```bash
npm ci
npm run dev
```

提交前构建与 GitHub Pages 相同的基础路径：

```bash
SITE_BASE=/llm-architecture-lab/ npm run build
SITE_BASE=/llm-architecture-lab/ npm run preview
```

文章位于 `docs/chapters/`，目录与导航在 `docs/.vitepress/config.mts`，站点样式在 `docs/.vitepress/theme/style.css`。正文采用普通书籍排版，避免营销式介绍与重复卡片。

## 审查与许可

提交说明应包含问题、变更和实际验证命令。修改公式时检查代码是否仍对应；修改实验时核对文章中的命令、指标与解释。引用论文和官方实现时注明具体型号、版本及许可，不大段复制第三方代码。

CI 会先执行全部章节快速实验，再构建站点；只有主分支的成功构建才部署 Pages。不要提交依赖目录、生成站点、私有数据、凭据或模型权重。贡献按本仓库 MIT 许可提供，第三方内容保留原许可。
