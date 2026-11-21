# dcu-vllm-wiki — vLLM/SGLang DCU 适配证据库

DCU / HCU / MACA 加速器的 vLLM 与 SGLang 适配项目证据库：架构、版本对比、HIP 移植、DTK/DAS 兼容、
Docker 镜像、API 映射、历史优化决策与实验结论。回答 DCU 适配项目问题前先查证据。

## 核心内容
- 回答 DCU 架构/版本/HIP 移植/兼容性问题前先查证据
- 历史优化决策与已接受/拒绝实验记录
- 跨 dcu-adapt 多个源码 worktree 检索

## 目录
```
dcu-vllm-wiki/
├── SKILL.md                        证据库导航
├── references/                    推理优化历史等
├── scripts/index_worktree_history.sh  worktree 历史索引
└── agents/openai.yaml            agent 辅助配置
```

## 快速用法
DCU 适配项目问题 / 想查历史优化决策时先查本 skill 证据。
