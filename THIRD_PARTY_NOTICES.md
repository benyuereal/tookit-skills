# Third-Party Notices

本仓库（dcu-toolkit-open）集成了下列第三方内容，按其原始许可分发；本仓库主许可为 MIT（见 LICENSE）。

| 来源 | 内容 | 许可 | 说明 |
|---|---|---|---|
| BBuf/AI-Infra-Auto-Driven-SKILLS | 多个通用工程/性能分析 skill（llm-torch-profiler-analysis、model-pr-optimization-history、llm-pipeline-analysis、generate-profile、humanize-gen-plan、humanize-rlcr、llm-serving-capacity-planner、sglang-humanize-review、torch-profiler-layer-track、cookbook-* 等） | MIT（见各 SKILL.md 头部来源标注） | 上游基于 CUDA/NVIDIA，DCU 上未经本仓库实测，按「参考、可尝试、DCU 待验证」对待 |
| benyuereal/llm-optimize | dcu-vllm-precision 的参考实现 | MIT（目录内 LICENSE） | 已保留目录内 LICENSE |
| tensormux/kernel-skills | dcu-kernel-dev 方法论（结构化 kernel playbook） | MIT | 仅提炼方法，未搬运性能数字 |
| Dominic789654/auto-kernel-research | dcu-kernel-dev 优化循环协议 | MIT | 仅协议方法 |
| jhinpan/ROCmKernelWiki | dcu-kernelwiki 设计参考 | Apache-2.0 | 仅设计借鉴 |
| shibingzhang/XDZS2025 | DCU 原生算子参考 | **无 license** | 未随包分发，仅文内引用（见 dcu-kernel-dev §13.4），分发前需自行确认许可 |
| sgl-project/sglang（上游） | `dcu-sglang-source/scripts/dspark_upstream/` DSpark 上游源码快照（post PR #30261，2026-08-06 捕获） | Apache-2.0（目录内 LICENSE/NOTICE） | 上游 main 布局快照，py_compile 校验通过 |

> 说明：原仓库曾内嵌的 `sglang-review-corpus.jsonl.gz`（GitHub 用户 PR 评论语料）已从本包移除；`model-pr-optimization-history` 各模型族数据为 BBuf/AI-Infra-Auto-Driven-SKILLS（MIT）上游的公开 GitHub PR 历史文档，经内容扫描（无内部标识）后按需恢复被引用部分（2026-10-22 重新评估）。
>
> 海光 RCCL fork（custom-rccl）与海光 SGLang fork（sglang_dcu_ref）的完整源码树**不随本包分发**（见 `docs/STANDARDS.md` 上游代码合规红线）；需要原文时请自行从对应 DTK 包/镜像提取。
