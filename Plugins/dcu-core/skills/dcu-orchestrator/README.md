# DCU 公开版薄编排层（orchestrator）

> 本目录提供 DCU 领域多任务工作流的**薄编排层**（公开版）。它不持有领域知识，
> 只负责把「workflow YAML + task 契约」翻译成可执行、可验收、可回退的任务序列，
> 并把每一步的证据、指纹、acceptance 结果记录下来。领域方法在各 `Plugins/*/skills` 的 skill 中。

## 目录结构

| 路径 | 说明 |
|---|---|
| `SKILL.md` | dcu-orchestrator 技能，中文，描述编排器行为与两种契约 |
| `scripts/render_workflow.py` | 解析 workflow，算就绪节点/执行批次（依赖拓扑 + max_parallel）；仅标准库 |
| `scripts/verify_evidence.py` | 校验 evidence 是否满足任务契约（path + fingerprint）；仅标准库 |
| `scripts/dcu_distill.py` | 把 run 产物转成知识库台账 / 证据 JSON / 晋升提案（确定性，不推断数字）；仅标准库 |
| `scripts/kbq.py` | 用户知识库（`kb/`）的索引与检索：`index` 列条目 + 五段合规性，`search <kw>` 按段标注命中；只读不建索引文件；仅标准库 |
| `scripts/workflow_state.py` | 工作流状态文件：`init` / `mark` / `inspect` / **`resume`** / `verify` / `finish`。`resume` 算断点续跑计划（哪些不重跑、哪些要重做、下一个是谁），只读不改 |
| `workflows/*.yaml` | 内置 5 条工作流：fault-triage / infer-optimize / model-deploy / op-optimize / research-baseline（详见 SKILL.md 第五节） |
| `tasks/env-check.yaml` | 任务：环境指纹探测（probe） |
| `tasks/baseline.yaml` | 任务：基准测量（normal，serving 指纹） |
| `tasks/quantize.yaml` | 任务：量化/精度写回（gate，需审批） |
| `tasks/deploy-precheck.yaml` | 任务：部署前置检查（probe：卡位/镜像/权重/配方） |
| `tasks/deploy-launch.yaml` | 任务：启动推理服务（gate，需审批） |
| `tasks/deploy-verify.yaml` | 任务：部署一致性验证（normal，对比参照实现） |
| `tasks/fault-collect.yaml` | 任务：故障现场采集（probe：日志/hy-smi/coredump） |
| `tasks/fault-locate.yaml` | 任务：根因定位（normal，结论+置信度） |
| `tasks/fault-fix.yaml` | 任务：修复执行与复测（gate，需审批，可回退） |
| `tasks/op-profile.yaml` | 任务：profile 热点采集（probe） |
| `tasks/op-roofline.yaml` | 任务：瓶颈判定与理论天花板（normal，ceiling + 地板达成率） |
| `tasks/op-constrain.yaml` | 任务：合法改动集合与硬件约束（normal，允许动作 + 可否证候选） |
| `tasks/op-optimize.yaml` | 任务：算子/kernel 优化改动（gate，需审批） |
| `tasks/op-validate.yaml` | 任务：端到端 A/B 判定（normal，opverify 三闸四档） |
| `tasks/op-distill.yaml` | 任务：优化成果沉淀到用户知识库（gate，需审批：台账 + 候选事实 + 晋升提案） |
| `README.md` | 本文件：契约文档与用法 |

## 任务契约（tasks/*.yaml）

每个任务一个 YAML 文件，顶层含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task/type` | normal | probe | gate | 任务类型。normal=常规；probe=只读探测；gate=改变状态/写回，走独立 verifier 或用户审批 |
| `goal` | string | 一句话目标 |
| `executor_skills[]` | list | 执行该任务调用的 skill 名（装配知识包时读其 SKILL.md） |
| `approach[]` | list | 执行步骤/方法（顺序） |
| `evidence` | object | 产物契约：`path` + `fingerprint[]` |
| `acceptance[]` | list | 通过判据（全部满足才 pass） |
| `out_of_scope[]` | list | 明确不做的事（防越界） |
| `proceed` | auto | ask | 是否需用户审批放行 |
| `on_exhaust` | exit | continue | rollback | 重试耗尽后的处理 |

### evidence 契约

- `path`：产物路径（相对运行目录或仓库根）。
- `fingerprint[]`：环境指纹，公共字段 `device` / `dtk` / `image` / `model`；
  领域附加：
  - **serving**：加 `ttft` / `tpot` / `throughput`
  - **kernel**：加 `latency` / `bandwidth`

### 各任务定义

| 任务 | 工作流 | type | fingerprint 附加 | proceed | on_exhaust |
|---|---|---|---|---|---|
| env-check | infer-optimize | probe | device/dtk/image/model | auto | continue |
| baseline | infer-optimize | normal | + ttft/tpot/throughput | auto | exit |
| quantize | infer-optimize | gate | + ttft/tpot/throughput | ask | rollback |
| deploy-precheck | model-deploy | probe | + weights_path/cards_free | auto | continue |
| deploy-launch | model-deploy | gate | + serve_endpoint/startup_args/mem_used_gb | ask | rollback |
| deploy-verify | model-deploy | normal | + cases_total/mismatch_ratio | auto | exit |
| fault-collect | fault-triage | probe | + fault_type/key_error | auto | continue |
| fault-locate | fault-triage | normal | + root_cause/confidence | auto | continue |
| fault-fix | fault-triage | gate | + fix_action/retest | ask | rollback |
| op-profile | op-optimize | probe | + profiler/top_kernel/top_ratio | auto | continue |
| op-roofline | op-optimize | normal | + op/shapes/cost_bound/ceiling_us/floor_attainment | auto | exit |
| op-constrain | op-optimize | normal | + op/shapes/verdict/allowed_actions/candidates | auto | continue |
| op-optimize | op-optimize | gate | + change_path/diff_summary/correctness | ask | rollback |
| op-validate | op-optimize | normal | + measured_ms/baseline_ms/gain/floor_attainment/verdict/invariants/accuracy | auto | exit |
| op-distill | op-optimize | gate | + op/gfx/source_run/gain/status | ask | continue |
## 工作流契约（workflows/*.yaml）

顶层：

- `name`：工作流名
- `max_parallel`：同时最多执行的节点数
- `nodes[]`：每个节点含
  - `id`：节点唯一 id
  - `task`：引用的任务名（对应 `tasks/<task>.yaml`）
  - `depends_on[]`：依赖的节点 id 列表（空 = 无依赖，最先就绪）
  - `retries`：失败重试次数
  - `proceed`：auto | ask（节点级覆盖 task 级）

## 编排器行为

1. 读 workflow（name/max_parallel/nodes）。
2. 算就绪节点（依赖全部完成）。
3. 装配知识包（读取 executor_skills[] 对应的 skill）。
4. 执行（按 type 约束）。
5. 采 evidence（path + fingerprint）。
6. 跑 acceptance（全部满足才 pass）。
7. 风险分流：低风险（proceed=auto / normal）自检放行；高风险（gate/写回）走独立 verifier 或向用户申请。
8. 摘要批量审批后：pass 解锁下游。
9. 重试 / on_exhaust（exit | continue | rollback）。

## 用法

```bash
# 渲染工作流就绪节点与执行批次（先核对顺序）
python3 scripts/render_workflow.py workflows/infer-optimize.yaml

# 校验某次运行收集的 evidence 是否满足任务契约（以 baseline 为例）
python3 scripts/verify_evidence.py runs/infer-optimize/baseline/result.json --task tasks/baseline.yaml

# 知识库检索（五段口径：结论 / 前提 / 已证伪 / 复现命令 / 口径）
python3 scripts/kbq.py index                      # 全部条目 + 缺哪段
python3 scripts/kbq.py search 口径 频率档          # 检索，命中处标出所属段落
python3 scripts/kbq.py search barrier --json       # 机器可读（--json 前后都能写）

# 断点续跑：中断（kill / 关机 / 换会话）之后先跑这个，再决定做什么
python3 scripts/workflow_state.py resume --run-dir runs/op-optimize/2026-10-28-xxx
python3 scripts/workflow_state.py resume --run-dir ... --json   # 机器可读，无活可干时 rc=1
```

`resume` 只**读**状态、只**算**计划，不改任何文件。它把三种容易搞错的判断做成确定性的：

| 状态 | 判断 | 为什么 |
|---|---|---|
| `done` | 重算产物 sha256，一致 → 不重跑；变了/删了 → **重做** | done 不等于可以跳过，产物可能被改写 |
| `running` | 一律当**中断**，重做 | 进程被 kill 时状态永远停在 running，没人会翻回来 |
| `failed` | 未超 `workflows/<wf>.yaml` 的 `retries` → 重试；超了 → **拦住，需人工** | 同一节点反复失败还继续跑就是烧时间 |

重试上限的权威点在 `workflows/*.yaml` 的节点 `retries` 上，脚本不另设常量；拿不到工作流定义时输出 `limits_error` 并把上限当作「未知」，不静默假设一个数字。
```

## 脱敏约定

本编排层的示例设备号 / 模型名 / 镜像 tag 均为公开占位（BW1000 / qwen3-32b / vllm:0.18.1 等），
不含内部实测数据、内网地址、内部工单号。发布前跑 `python3 sanitize.py <repo>` 断言零残留。

---

（本 README 由发布脚本生成）
