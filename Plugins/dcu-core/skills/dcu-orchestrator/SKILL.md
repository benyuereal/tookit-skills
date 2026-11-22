---
name: dcu-orchestrator
description: "DCU 多任务薄编排层（公开版）。依据 workflow YAML 与 task 契约，做多任务编排：读 workflow → 算就绪节点 → 装配知识包 → 执行 → 采 evidence → 跑 acceptance → 低风险自检/高风险独立 verifier 或向用户申请 → 摘要批量审批 → pass 解锁下游 → 重试/on_exhaust。触发场景：任务跨 ≥2 个域、步骤不确定、需要按证据验收与审批推进（而不是单步问答）。内置五条工作流：推理优化（env-check→baseline→quantize）、模型部署验证（precheck→launch→verify）、故障排查（collect→locate→fix）、算子优化（profile→roofline→constrain→optimize→validate→distill）、调研复现（research-collect→env-reproduce→baseline-run）；亦可作为其它多步流程的骨架复用。不适用：单步明确任务直接点名叶子 skill；只问某个 skill 的知识、不需要多步编排时不要经此。"
version: 1.1.3
author: DCU Toolkit Contributors
license: MIT
kind: composite
---

# dcu-orchestrator — 公开版薄编排层

> 定位：本 orchestrator 是 DCU 领域多任务工作流的**薄编排层**。它不重复持有领域知识，
> 只负责把「workflow YAML + task 契约」翻译成一组可执行、可验收、可回退的任务序列，并把
> 每一步的证据、指纹、acceptance 结果记录下来。领域方法在 `Plugins/*/skills` 里的各 skill 中。
> 覆盖率：env-check / baseline / quantize / op 优化等阶段；可作为其它多步流程的骨架复用。

## 一、编排器行为（核心循环）

对一份 workflow（见 `workflows/*.yaml`）：

1. **读 workflow**：解析 `name / max_parallel / nodes[]`。
2. **算就绪节点**：找出所有 `depends_on` 依赖已满足（无未完成依赖）且未完成的节点。
3. **装配知识包**：根据节点 `task` 的 `executor_skills[]`，把对应 skill 的 SKILL.md 与
   相关 references 装配成该节点执行器的「知识包」（约上下文大小上限内）。
4. **执行**：调用对应域 skill 的方法完成 `goal`。type 不同，执行约束不同（见下）。
5. **采 evidence**：按 `evidence` 契约收集产物（`path` + `fingerprint[]`）。
6. **跑 acceptance**：逐一校验 `acceptance[]`；全部满足才算 pass。
7. **风险分流**：
   - **低风险**（`proceed: auto` 或 type=normal 且不涉设备写）：编排器自检即可放行。
   - **高风险**（gate/涉及改设备配置、重富化、量化写回、重刷镜像）：由**独立 verifier**
     （另一上下文/另一 subagent 复跑验收）复核，或**向用户申请**（`proceed: ask` 时）。
8. **摘要批量审批**：把本轮所有节点的通过/拒绝、evidence 摘要、剩余阻塞整理成一条审批摘要，
   走批量放行（低风险）或逐项人工放行（高风险/ask）。
9. **pass 解锁下游**：节点 pass 后，解除依赖它的下游节点的阻塞。
10. **重试 / on_exhaust**：节点失败时按其 `retries` 重试；耗尽后按 `on_exhaust`
    （exit | continue | rollback）处理。

### 节点/task 的 type 语义

| type | 含义 | 执行约束 |
|---|---|---|
| `normal` | 常规任务 | 完成后跑 acceptance，pass 即解锁下游 |
| `probe` | 探测/采集类（只读、不改变状态） | 产出 evidence；不 gate 下游，失败可 continue |
| `gate` | 门槛/审批类（改变状态或写回设备） | 必须独立 verifier 或向用户申请后才放行 |

## 二、任务契约（task/*.yaml）

每个任务是一个 YAML 文件，顶层字段（README 有完整表格）：

- `task/type` : normal | probe | gate
- `goal` : 一句话目标
- `executor_skills[]` : 该任务执行调用的 skill 名列表（可空，空则直接用 approach）
- `approach[]` : 执行步骤/方法（顺序步骤）
- `evidence` : 产物契约，`path` + `fingerprint[]`
- `acceptance[]` : 通过判据（全部满足才 pass）
- `out_of_scope[]` : 明确不做的事（防越界）
- `proceed` : auto | ask（是否需用户审批放行）
- `on_exhaust` : exit | continue | rollback（重试耗尽后的处理）

### evidence 契约

`evidence` 至少含：

- `path`：产物路径（相对工作流运行目录或仓库根）
- `fingerprint[]`：环境指纹字段，用于标记产物的可复现上下文。公共字段：
  `device`（设备型号）、`dtk`（DTK 版本）、`image`（镜像 tag）、`model`（模型版本）。
  领域附加：
  - **serving**（LLM 服务/推理服务）：加 `ttft`、`tpot`、`throughput`
  - **kernel**（算子/内核）：加 `latency`、`bandwidth`

示例 `evidence`：

```yaml
evidence:
  path: "runs/infer-optimize/baseline/result.json"
  fingerprint:
    device: "BW1000"
    dtk: "26.04"
    image: "vllm:0.18.1"
    model: "qwen3-32b"
    # serving 场景追加
    ttft: "ms"
    tpot: "ms"
    throughput: "tokens/s"
```

## 三、工作流契约（workflows/*.yaml）

顶层：

- `name` : 工作流名
- `max_parallel` : 同时最多执行的节点数
- `nodes[]` : 每个节点含
  - `id` : 节点唯一 id
  - `task` : 引用的任务名（对应 tasks/<task>.yaml，可为内联配置）
  - `depends_on[]` : 依赖的节点 id 列表（空 = 无依赖，可立即就绪）
  - `retries` : 失败重试次数
  - `proceed` : auto | ask（覆盖 task 级 proceed；节点级优先）

示例见 `workflows/infer-optimize.yaml`。

## 四、编排辅助脚本

- `scripts/dcu_distill.py`：把一次 run 的产物（如 runs/op-optimize/validate/result.json）确定性转成知识库条目 + 晋升提案 + 机器可读证据 JSON（数字只来自产物，缺字段标 MISSING，不推断）。
- `scripts/render_workflow.py`：读 workflow YAML，按依赖拓扑算「就绪节点」与执行顺序，
  打印每个节点的依赖/状态，便于人工核对或交给执行器。检测到依赖环 / 缺任务文件 / 缺契约字段时
  输出问题清单并**非零退出**（可作 CI/脚本门禁）；计划干净则退出 0。仅标准库。
- `scripts/verify_evidence.py`：校验某次运行收集的 evidence 是否满足任务契约
  （必须字段、fingerprint 是否齐全），输出校验报告。仅标准库。
- `scripts/workflow_state.py`：工作流状态文件与断点续跑。`mark` 时按节点名自动找
  `tasks/<node>.yaml` 做字段级契约校验（done 的产物必须过契约）；`resume` 算出
  「哪些节点不重跑（产物 sha256 复核有效）、哪些要重做（产物失效 / 状态停在 running
  = 进程被 kill）、哪些已超重试上限需人工」，只读不改。重试上限读 `workflows/<wf>.yaml`
  的节点 `retries`，不另设常量。仅标准库。

## 五、内置工作流

五条内置工作流（workflows/*.yaml + tasks/*.yaml 一一对应），均为串行链路（多数是「probe 采集 → normal 分析/验证 → gate 写状态需审批」的三段式；op-optimize 为六节点全链路）：

| 工作流 | 链路 | 适用场景 |
|---|---|---|
| `infer-optimize` | env-check → baseline → quantize(ask) | 推理优化主线：环境指纹 → 端到端基准 → 量化/重精度写回。量化通过后可追加探索节点（接 op-optimize 工作流的任务契约） |
| `model-deploy` | deploy-precheck → deploy-launch(ask) → deploy-verify | 模型部署验证主线：前置检查（卡位/镜像/权重/配方）→ 启动服务（gate）→ 与参照实现精度一致性验证 |
| `fault-triage` | fault-collect → fault-locate → fault-fix(ask) | 故障排查主线：现场采集（VMFault/hang/crash/精度/通信）→ 根因定位（结论+置信度）→ 修复执行与复测（gate） |
| `op-optimize` | op-profile → op-roofline → op-constrain → op-optimize(ask) → op-validate → op-distill(ask) | 算子优化主线：profile 热点采集 → 理论天花板与两级 ceiling_bound 判定 → 允许动作集合与硬件约束 → kernel/编译改动（gate，可回退）→ opverify 三闸四档端到端 A/B 判定（只有 ACCEPTED 才放行）→ 沉淀到用户知识库（台账+候选事实+晋升提案，gate） |
| `research-baseline` | research-collect → env-reproduce(ask) → baseline-run | 调研复现主线：外部调研 → 真机复现（占卡起服务，gate）→ 固定口径端到端基线 |

`op-optimize` 工作流每个节点由对应 op-* skill 执行：

- `op-profile` ← op-profiling / op-bench / op-contract（probe 只读采集）
- `op-roofline` ← op-roofline / op-profiling（理论天花板与地板达成率）
- `op-constrain` ← op-contract / op-roofline（允许动作集合与硬件约束）
- `op-optimize` ← op-optimize + dcu-kernel-dev / dcu-triton-kernels（gate 实施改动）
- `op-validate` ← op-gate / op-bench / op-profiling（三闸四档端到端判定）
- `op-distill` ← op-gate / dcu-kernelwiki / dcu-perf-cases / dcu-op-optimization（gate 沉淀）

各任务的 executor_skills / evidence 指纹字段 / acceptance 判据 / out_of_scope 边界见 `tasks/*.yaml` 注释，新增工作流按同一契约扩展。

> 脱敏要求：示例中的设备号、模型名、镜像 tag 均为公开占位（BW1000 / qwen3-32b / vllm:0.18.1 等），
> 无内部实测数据、无内网地址、无内部工单号。

## 六、使用建议

- 多步任务先用 `render_workflow.py` 核对就绪与顺序，再让执行器消费。
- 每次运行产生 `runs/<workflow>/<node-id>/...` 目录，存放 evidence 与 acceptance 报告。
  建议在包外工作目录运行（勿以包根目录为 CWD）；`runs/` 不入库（已 gitignore）。
- 高风险（gate）节点务必走「独立 verifier 或向用户申请」，不要只管自检。
- 证据与指纹要齐全，否则无法跨会话/跨设备复现。

## 七、与其它 skill 的关系

- 本 orchestration 是通用骨架；具体领域方法在 `dcu-env` / `dcu-perf` / `dcu-inference`
  / `dcu-operators` 等域的 `dcu-*` skill 中。
- `dcu-skill-router` 提供全量路由表；本 orchestrator 只做编排，不替代路由。

## 验收

1. `render_workflow.py` 能渲染出就绪节点与依赖顺序，无环。
2. 每个 task 的 evidence 字段能被 `verify_evidence.py` 校验通过。
3. `proceed: ask` 的节点确实在改状态/占卡前停下来等审批。
4. 一次 run 的产物能被 `dcu_distill.py` 转成知识库台账。

## 不适用场景

单步明确任务（直接点名叶子 skill，不必经编排层）；
需要的是「怎么优化」的方法本身（本 skill 只做编排，不定义判定口径）；
想改 workflow 契约字段（那是 `task/*.yaml` 的 schema 变更，需单独评审）。
