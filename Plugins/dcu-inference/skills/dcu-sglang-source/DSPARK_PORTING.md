# DSpark 移植参考（dcu-sglang-source）

> 配套源码： [`scripts/dspark_upstream/`](./scripts/dspark_upstream/)（上游 sglang main 快照，31 文件）
> 算法验证： [`scripts/dspark_sandbox_verify.py`](./scripts/dspark_sandbox_verify.py)（纯逻辑跑通，见末尾）

## 一、DSpark 是什么

DSpark = **Confidence-Driven Speculative Decoding**（置信度驱动的投机解码）。
- 上游 LMSYS 博客（2026-07-06）：*SGLang now supports DSpark on both dense and sparse models (e.g. Qwen3 and DeepSeek-V4)*，集成于 **PR #30261**。
- 在 DeepSeek-V4 上：复用模型自带的 sparse-MLA 路径（flash_mla），每个支持的 backend 只需从 packed layout 重建 varlen 元数据，**无需新 kernel**；非 sparse 模型（如 Qwen3）则有独立的 block drafter。
- DeepSeek-V4-Flash 的 0731 refresh 带了一个 **bundled DSpark draft head** 检查点（HuggingFace `DeepSeek-V4-Flash-DSpark`）。

### 与 DeepSeek-V4 其它机制区分（易混）
| 机制 | 类别 | DSpark 关系 |
|---|---|---|
| CSA / HCA 混合压缩注意力 | 注意力/压缩 | 无关（海光 fork 的 `dsv4/` 已实现这部分） |
| sparse-MLA + indexer (Lightning TopK) | 注意力 | DSpark 在 V4 上**复用**其 sparse-MLA 路径 |
| **DSpark** | **投机解码调度** | **本参考主题**，海光 fork 缺失 |

## 二、海光 fork 现状（关键结论）

**海光 fork 0.5.12（+g9a3a3e5f9）不支持 DSpark。**
- 在 <device-A> 设备镜像 `dsv4-flash-k100ai-sglang0.5.12-20260728-jupyter` 内全量 `grep -rinE 'dspark'` → **0 命中**（仅命中无关的 SpargeAttn 3 处）。
- 海光 fork 已落地 DeepSeek-V4 的**注意力/压缩部分**（`deepseek_v4_backend.py` + `dsv4/` 子目录：compressor/indexer/flash_mla_triton/tilelang_kernel/sparse_prefill），**但投机解码只到 eagle/mtp/ngram/dflash，没有 DSpark**。

## 三、上游 DSpark 源码清单（已抓取落地 scripts/dspark_upstream/）

> 注意：PR #30261 原列 `dspark_components/kernels/` 7 个文件在 main 上**已重构消失**（合并进 `sglang/kernels/ops/speculative/dspark/`）。本快照按 main 当前布局抓取。

```
scripts/dspark_upstream/sglang/
├── srt/speculative/
│   ├── dspark_components/              # 12 文件（无 kernels/ 子目录）
│   │   ├── __init__.py
│   │   ├── dspark_config.py            # DSparkDraftConfig / DSparkRuntimeConfig
│   │   ├── dspark_draft.py             # drafter（sparse/dense 两路）
│   │   ├── dspark_draft_sampler.py     # draft 采样（后续新增）
│   │   ├── dspark_block_accept_estimator.py  # ★ 置信度上界估计（confidence relay 核心）
│   │   ├── dspark_planner.py           # ★ 调度/两阶段回退（1117 行，最大文件）
│   │   ├── dspark_verify.py            # verify（716 行）
│   │   ├── dspark_kv_inject.py
│   │   ├── dspark_observability.py
│   │   ├── dspark_sps.py               # SpsCostTable 成本表（置信度 relay 用）
│   │   ├── dspark_sts.py
│   │   └── dspark_worker_v2.py         # ★ v2 worker（含 is_cuda 适配点）
│   ├── ragged_verify.py                # ragged verify（DSpark verify 复用）
│   ├── spec_registry.py / spec_info.py / base_spec_worker.py  # 被 PR 修改（注册 + 接口）
│   └── (draft_worker_common.py 等依赖 —— 海光 fork 缺失，见 §四)
├── srt/models/
│   ├── dspark.py                       # DSpark 模型侧封装（612 行）
│   └── deepseek_v4_dspark.py          # DeepSeek-V4 + DSpark 绑定（883 行）
├── srt/arg_groups/speculative_hook.py  # 命令行参数钩子（注册 --speculative dspark）
└── kernels/ops/speculative/dspark/    # 4 个 Triton kernel（真正的内核三件套）
    ├── dspark_accept.py                # accept 判定（864 行）
    ├── dspark_draft_model.py           # draft 模型前向（447 行）
    ├── dspark_schedule.py              # 调度（260 行）
    └── dspark_verify_window.py         # verify window（875 行）
```

## 四、依赖树 + 海光 fork 缺失模块（移植阻断点）

在 <device-A> 海光 fork 0.5.12 实测，DSpark 依赖的以下模块**不存在**：

| 缺失模块 | 用途 | 阻断级别 |
|---|---|---|
| `srt/speculative/draft_worker_common.py` | DSpark worker 基类（v2 体系） | **硬阻断**：import 即 `ImportError` |
| `kernels/ops/speculative/dspark/` | 4 个 Triton kernel 落点 | **硬阻断**：verify/draft 运行时缺 kernel |
| `srt/speculative/ragged_verify.py` | ragged verify 复用 | **硬阻断**：dspark_verify 依赖 |
| `srt/kv_canary/runner/future_tensor.py` | `DelayedDeviceHostHandler` | 硬阻断：estimator 依赖 |
| `srt/managers/overlap_utils.py` | overlap 调度 | ✅ 海光 fork **已有** |
| v2 worker 体系（`BaseSpecWorkerV2` 等） | DSpark worker 基础 | 部分缺失：`base_spec_worker.py` 仅 `BaseSpecWorker(ABC)`，无 v2 |

**结论**：DSpark 不是「加 `is_dcu()` 桩就能跑」——它依赖**上游 0.5.12 之后同期引入的多个模块**（`draft_worker_common`、v2 worker 体系、`kv_canary`、`ragged_verify`、`kernels/ops/speculative/dspark`）。海光 fork 0.5.12 早于这些提交，**必须先把上游相关 commit rebase 进海光 fork**，否则连锁 `ImportError`。

## 五、DCU / 海光适配点（待 rebase 后做）

1. **`is_dcu()` 分发**（非 vLLM 的 `is_rocm`）：`dspark_worker_v2.py` 有 2 处 `is_cuda()` 判断需改 `is_cuda() or is_hip()`：
   - 第 218 行：`and is_cuda()`
   - 第 331 行：`if is_cuda() and capture_decode_cuda_graph:`
2. **Triton kernel ROCm 验证**：4 个 kernel（accept/draft_model/schedule/verify_window）是 Triton，需验证 ROCm 下 `tl` 算子兼容性 + `warp_size=64`（参照 `dcu-kernel-dev` TileLang/Triton 已知问题）。
3. **CUDA graph 路径**：`SGLANG_PREP_IN_CUDA_GRAPH` 在 ROCm 下需验证等价性（planner 强制要求该 flag 或 `SGLANG_RAGGED_VERIFY_MODE=static`）。

## 六、移植 checklist

- [ ] 海光 fork rebase 上游 main 至含 PR #30261 及后续 DSpark 相关 commit（含 `draft_worker_common` / v2 worker / `kv_canary` / `ragged_verify` / `kernels/ops/speculative/dspark`）
- [ ] 落地 `dspark_components/` 12 文件 + `kernels/ops/speculative/dspark/` 4 kernel + 模型侧 2 文件 + 注册 3 文件
- [ ] `is_dcu()` 适配（§五.1 两处）
- [ ] Triton kernel ROCm 兼容性验证（§五.2）
- [ ] CUDA graph / ragged verify 模式在 ROCm 验证（§五.3）
- [ ] 起服务 `--speculative dspark --model deepseek-ai/DeepSeek-V4-Flash` 在 DCU 真机跑通（需 BW150/BW100 对应 fork 就绪）
- [ ] 性能对比：DSpark vs MTP vs 无投机（参照上游 383.7 tok/s @ accept~5 的基线，DCU 上重测）

## 七、版本错配警告

本快照来自 **sglang main（post PR #30261）**，**不是** 0.5.12。海光 fork 0.5.12 直接套用会因 `base_spec_worker` 接口、v2 worker 体系、依赖模块缺失而 **import 失败**。**严禁**直接把 `scripts/dspark_upstream/` 拷进海光 fork 0.5.12 的 `speculative/` 然后指望能跑——必须先 rebase 上游对应版本。

## 八、算法级验证（dspark_sandbox_verify.py）

因海光 fork 缺依赖、且端到端需 rebase，**无法做 import/推理级验证**。改为**纯逻辑 sandbox**：抽离 DSpark 核心算法（`SpsCostTable` 插值成本表、`OnlineCeiling` 置信上界估计、两阶段 relay 决策），用 mock 数据跑断言：

```
$ python3 scripts/dspark_sandbox_verify.py
[1] SpsCostTable 插值逻辑  ✓ 边界 clamp / 单调 / 插值精确
[2] OnlineCeiling 置信上界估计  高accept场景天花板=0.850 / 低accept场景=0.280
[3] 两阶段回退 relay 决策  ✓ full_draft/partial_relay/fallback_no_spec 三态正确
全部断言通过
```

> 这是「算法逻辑本身正确」的证据，**不等同于移植可运行**。端到端可运行需 §六 checklist 全部完成（海光 fork rebase 后）。
