# AGENTS.md — 跨 Agent 通用入口（dcu-toolkit-open）

> 版本：1.6.0（与仓库 .claude-plugin/marketplace.json 保持一致）
> 本文件是跨所有 agent（Claude / Codex / OpenCode / Cursor / Gemini / dsh …）的通用入口，
> 让任意 agent 在本仓库/项目里都知道如何路由到 DCU 领域 skill。
> 安装器（install.sh）也会在项目根生成同款文件（版本固定，指向 Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md + 各域 skill）。

## 我是谁 / 这里有啥

本仓库是海光 DCU（Hygon DCU）测试调优技能集开源版（dcu-toolkit-open）。
覆盖基础环境、性能剖析、推理框架部署适配、算子开发、算子卡、故障排查、工程实践等域。
技能按 Claude Code / OpenCode 插件包结构组织：`Plugins/<domain>/skills/<skill>/SKILL.md`。

## 入口：先读 dcu-orchestrator

- 总编排入口与工作流契约：[`Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md`](Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md)（dcu-orchestrator，中文）
- 多任务编排（五条内置工作流：推理优化 env-check→baseline→quantize、模型部署 precheck→launch→verify、
  故障排查 collect→locate→fix、算子优化 profile→optimize→validate→distill、调研复现 research-collect→env-reproduce(ask)→baseline-run），读
  [`Plugins/dcu-core/skills/dcu-orchestrator/workflows/*.yaml`](Plugins/dcu-core/skills/dcu-orchestrator/workflows/)
  与 `tasks/*.yaml`（任务契约字段说明见 `Plugins/dcu-core/skills/dcu-orchestrator/README.md`）。
- 编排辅助脚本：`render_workflow.py`（渲染就绪/依赖/执行顺序）、`verify_evidence.py`（校验 evidence 契约）、
  `dcu_distill.py`（把一次 run 的产物转成知识库台账 + 机器可读证据 JSON + 晋升提案）。

## 沉淀 / 知识库（用户态，重要）

- 沉淀写**用户知识库** `~/.config/dcu-toolkit/kb/`：`op-opt/` 算子优化台账由 `op-optimize` 工作流收尾的
  `op-distill` 节点自动写入（`dcu_distill.py` 生成台账 + 证据 JSON）；`issue/` 故障记录、`solution/` 部署方案卡片
  **暂无自动节点**，链路收尾时按包内 `data/kb/README.md` 的约定人工补写（模板见 `data/kb/templates/`）。
- **不要往 skill 树里写沉淀**：`<skills根>/` 与 `<skills根>/_dcu-toolkit-shared/` 是厂商内容，重装/升级以包内版本为准。
  install.sh 的清单式增量同步（清单在 `~/.config/dcu-toolkit/manifests/`）只删「上次装的 + 本轮包内已移除 + 本地未改过」的文件，
  用户新增文件永不删、本地改过的厂商文件被覆盖前备份为 `<file>`.local.bak —— 但记录类内容的正确落点仍是 `kb/`。
- 纪律：每条事实带 `source` + `verified_on`（日期/设备/gfx/DTK）；未验证结论标 `unverified`；
  性能数字标 `[历史数据，随硬件/DTK/版本变化]`，且不进可复用事实条目。
- 晋升：候选事实由 `op-distill` 产出晋升提案，**经人工/厂商审核**后才并入 skill（不自动改写厂商内容）。

## 领域 skill 路由（按意图）

| 意图关键词 | 首选 skill | 域 |
|---|---|---|
| 环境检测 / DTK / 驱动 / 容器 / hy-smi / NUMA | dcu-env、dcu-dtk、dcu-driver、dcu-hy-smi、dcu-cpu-affinity、dcu-docker、dcu-topology | env |
| 性能剖析 / 算子优化 / 推理优化 / benchmark | dcu-perf-analysis、dcu-op-optimization、dcu-inference-optimization、dcu-llm-benchmark、dcu-perf-profiling | perf |
| vLLM 部署 / 精度 / 自定义算子 / trace | dcu-vllm-deploy、dcu-vllm-precision、dcu-vllm-custom-op、dcu-vllm-trace-analysis、dcu-vllm-runtime-patch | inference |
| 模型部署 / 可行性 / 配方目录 | dcu-model-deploy-verify、dcu-model-deploy-catalog、dcu-model-portability | inference |
| CUDA→HIP 移植 / kernel 开发 / Triton | dcu-cuda-hip-port、dcu-kernel-dev、dcu-triton-kernels、dcu-kernelwiki | operators |
| 故障排查 / 精度 / crash / hang | dcu-vmfault-coredump、debug-cuda-crash、debug-distributed-hang、dcu-vllm-precision、dcu-train-precision | troubleshooting |
| 工程实践 / CI / 代码规范 / cookbook | ci-workflow-guide、env-var-conventions、cookbook-*、model-pr-* | engineering |
| 算子卡 / 某算子有哪几种实现 / 哪个档位用哪个 / kernel 改动算不算通过 / 怎么持续推进一轮优化 / **先算出预测再写代码** / 手写 HIP 慢但数值对 | op-sparse-mla、op-moe-w8a8、op-gemm-fp8、op-contract、op-bench、op-profiling、op-gate、op-optimize、op-roofline、op-hip-compile | ops |
| 全量路由索引 | dcu-skill-router（dcu-core） | core |

> 完整路由表见仓库根 `SKILL.md`（包级总入口）与 `dcu-skill-router`。

## 安装 / 接入

- 直接调用 `bash install.sh` 即可把全部 skills 装入本机各 agent 的 skills 目录。
- `--detect`（默认）自动扫描本机 agent 并逐层写入；`--tool <name>` 只装指定 agent；`--list` 只列出检测结果。
- 幂等：重装不会重复拷贝内容一致的 skill；登记表在 `~/.config/dcu-toolkit/dcu-toolkit-installed.json`。

## 安全

- 包内零密码、零密钥、零内网地址；真实凭证在 `~/.config/dcu-toolkit/credentials.json`（用户侧自行填写，chmod 600）。
- 远程密码认证用 `sshpass -f <文件>` 或 `sshpass -e`，禁用 `sshpass -p`。
- 发布前跑 `python3 sanitize.py <repo>` 断言零内部残留。
