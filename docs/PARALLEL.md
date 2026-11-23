# 并行推进登记表

> 业主指令：多个 agent 并行推进，但**写范围必须切干净**，否则互相覆盖。
> 每次开工前在这里登记；完工后更新状态。**写范围重叠 = 立刻停止并报告。**

版本：2026-11-23 第 1 批

---

## 当前批次

| agent | 写范围 | 开始 | 状态 |
|---|---|---|---|
| **主线程** | `docs/` · `scripts/atlas_query.py` · `Plugins/ops/op-atlas/validate.py` · `docs/PARALLEL.md` | 15:45 | 进行中 |
| agent-1 GEMM 卡重写 | `Plugins/ops/op-atlas/cards/gemm-bf16.yaml` · `gemm-fp8.yaml` · `gemm-int8.yaml` · `perf/gemm-*.json` | 15:45 | 待启动 |
| agent-2 融合算子 I1-I4 | `Plugins/ops/op-atlas/cards/fusion-*.yaml`（新建 4 张）· `perf/fusion-*.json` | 15:45 | 待启动 |
| agent-3 三个真缺 + FP8 链 | `cards/batched-gemm.yaml` · `dequant-fp8-w4.yaml` · `gemm-swiglu-epilogue.yaml` · `cards/quant-per-token.yaml` | 15:45 | 待启动 |
| agent-4 constants + 调优规则 | `Plugins/ops/skills/op-roofline/constants.yaml` · `docs/STANDARDS.md` | 15:45 | 进行中 |

## 第 2 批（16:20 起）—— 代码质量，与第 1 批**路径不重叠**

| agent | 写范围 | 开始 | 状态 |
|---|---|---|---|
| **主线程 P0-1** | `Plugins/ops/skills/op-rmsnorm-fused/tests/test_adapter.py` · `Plugins/ops/hooks/ops_gate.py` · `Plugins/ops/skills/op-optimize/loop.py` | 16:20 | 进行中 |
| agent-Q2 except 分类 | `scripts/rank_impls.py` · `scripts/opcheck.py` · `Plugins/ops/op-atlas/validate.py` | 16:20 | 待启动 |

**第 2 批不许碰**（第 1 批在改）：`cards/*.yaml` · `perf/*.json` · `constants.yaml` · `docs/STANDARDS.md`

---

## 主线程独占（其他 agent 不许碰）

- `Plugins/ops/op-atlas/validate.py` —— 所有校验规则的唯一实现
- `Plugins/ops/op-atlas/INDEX.md` / `gaps.md` —— 生成物，只能由 `sync_index.py` / `gen_gaps.py` 生成
- `docs/PARALLEL.md` —— 本文件
- 任何 `git commit` / `git push` —— **只有主线程做**（避免并发 commit 冲突）

---

## GPU 互斥

只有一个 DCU，`claude-opbench` 也在跑。跑 bench 必须：

```bash
flock /tmp/opbench-gpu.lock -c "<bench 命令>"
```

开跑前 `hy-smi --showpids` 必须为空（连续 3 次采样，只有 PID 才算占用）。

---

## 已知阻塞（传给所有 agent）

### B1 · PANORAMA 的 fp8 579.7 TF 在本机无法复现

主线程实测（2026-11-23 15:39）：

| 项 | PANORAMA | 本机实测 |
|---|---|---|
| bf16 cublas(rocBLAS) | 175.7 TF | **248.9 TF** |
| bf16 hipblaslt | 188.5 TF | **187.0 TF** ✓ 吻合 |
| fp8 `_scaled_mm` | 579.7 TF | **`could not find valid hipblaslt solution`** |

- 已确认 `torch.backends.cuda.preferred_blas_library` 可用，切到 `hipblaslt` 后 bf16 到 187 TF（与 PANORAMA 吻合）
- 但**切后端后 fp8 依然报同一个错**
- `hipblaslt-bench` 存在，但**权限是 `-rw-r--r--`（没有可执行位）**，跑不起来

⇒ **agent-1 处理 gemm-fp8 时：不许照抄 579.7 写进 achievable。**
   写成「PANORAMA 记录 579.7 TF（来源 hbird-gemm/docs/perf.md §3），
   **本机未复现**（`_scaled_mm` 报 could not find valid hipblaslt solution，
   hipblaslt-bench 无可执行位）」，并标 `confidence: unverified`。

---

## 完成记录

（每完成一个 agent 在这里追加：agent 名 / commit / review 结论）

---

# 第 3 批（2026-11-24，本轮冲刺：E40 / G5 放置字段 / pmc_probe）

计划：`/root/atlas-handoff/SPRINT-PLAN.md`

| agent | 写范围（只能改这些） | 任务 | 状态 |
|---|---|---|---|
| **主线程** | `op-atlas/validate.py` · `scripts/scorecard.py` · `docs/` · `tests/test_floor_block.py` · **全部 git** | T1 = E40 | 进行中 |
| agent-P | `scripts/pmc_probe.py`(新) · `tests/test_pmc_probe.py`(新) · `op-profiling/references/counters.md`(新) | T3 = PMC 三态表 | 进行中 |
| agent-R | `scripts/results_schema.py`(新) · `tests/test_results_schema.py`(新) | T2 = placement schema | 进行中 |
| agent-B | `op-atlas/perf/*.json`（4 个已有文件，只加 placement 键） | T2 = 数据回填 | 进行中 |

**第 3 批不许碰**：`cards/*.yaml`（本轮不做卡片内容）· `scripts/floor.py` ·
`op-optimize/loop.py` · `INDEX.md` · `gaps.md`（生成物）· 任何 `git` 命令。

**GPU**：三张 agent 都不许跑 bench；需要 GPU 的由主线程串行执行
（`flock /tmp/opbench-gpu.lock`）。
