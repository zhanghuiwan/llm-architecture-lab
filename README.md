# LLM 架构实验室

一套中文 LLM 网络结构教程：从语言建模与最小 GPT 开始，逐步理解现代 Transformer、GQA、KV Cache、MoE 与真实模型源码，再开展可复现的架构实验。按先修关系与验收标准推进，不按周排期。

采用 VitePress 构建，提供章节导航、页内目录、本地搜索、数学公式、代码高亮和深浅主题，适配桌面与移动端。

## 当前内容

- 14 章首版讲义：基础实现 → 现代结构 → 架构实验。
- 学习指南、统一实验规范、术语表与参考资料。
- 当前不包含完整训练工程、预训练权重或已验证的模型性能结果。

## 本地阅读与维护

需要 Node.js 22+ 与 npm。

```bash
npm ci
npm run dev
```

章节位于 `docs/chapters/`，导航配置位于 `docs/.vitepress/config.mts`。

```bash
npm run build
npm run preview
```

## GitHub Pages

仓库创建并推送后，在 Settings → Pages 中选择 GitHub Actions 作为部署源。`.github/workflows/pages.yml` 会在推送到 `main` 时构建与部署，自动使用仓库名称作为站点路径。

本地验证仓库路径部署：

```bash
SITE_BASE=/llm-architecture-lab/ npm run build
SITE_BASE=/llm-architecture-lab/ npm run preview
```

部署状态请以实际工作流结果为准；配置文件存在不代表站点已经发布。

## 参与完善

欢迎提交内容勘误、推导、练习与经过验证的参考实现。模型案例请注明具体型号和源码版本；实验请提供完整条件，不提交没有执行依据的性能或能力结论。引用第三方内容时保留来源，并遵守其许可证。

## 许可证

[MIT](LICENSE)。第三方模型、代码和数据集遵循其各自许可。
