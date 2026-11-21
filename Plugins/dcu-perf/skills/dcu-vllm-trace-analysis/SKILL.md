---
name: dcu-vllm-trace-analysis
description: 解析 vLLM V1 torch.profiler trace（.pt.trace.json[.gz]），定位 prefill/decode 阶段，按类别（gemm/w8a16/MoE/attn/triton/elementwise...）汇总 kernel 统计并产出 xlsx（DCU/ROCm 平台）。与 llm-torch-profiler-analysis（离线报告流）互补：本 skill 偏"按类别占比"视角，后者偏"kernel/空泡/融合"报告。
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# vLLM Trace Analysis

> 来源：整合自个人 skill 仓库 `<personal_skill_repo>`（`vllm-trace-analysis`），落库为 `dcu-vllm-trace-analysis`（2026-08-13）。脚本已 `py_compile` 通过，仅依赖 `c++filt`(binutils) + `gzip`，可选 pandas/openpyxl（无则 `--text` 纯文本输出）。

Analyze a vLLM V1 `torch.profiler` trace on Hygon DCU / ROCm. Two steps:

## Step 1 — Locate profiling stages

vLLM V1 names every engine step stage like:

- `execute_context_1(76)_generation_0(0)` — **prefill** (ctx_len=76, gen_len=0)
- `execute_context_0(0)_generation_1(1)` — **decode** (ctx_len=0, gen_len=1)

Classification rule: `gen_len > 0` → decode; `ctx_len > 0` → prefill; else other.

Detection method (autonomous, no manual digging):
1. Decompress if `.gz`: `gzip -dc trace.pt.trace.json.gz` (macOS `zcat` misreads `.gz` as `.Z`).
2. In `traceEvents`, find every event whose `name` matches
   `execute_(?:context|generation)_\d+\(\d+\)_generation_\d+\(\d+\)`.
3. **CPU/GPU pairing**: the same stage name appears on the CPU executor thread AND on the GPU side. The device thread = the `(pid,tid)` that hosts the most `cat=='kernel'` events (typically the `tid=0` of the torch process). GPU stage span = the stage event on that thread; CPU span = the other copy.
4. **Repeated names**: vLLM reuses the same stage name across steps of identical shape (e.g. the decode stage `execute_context_0(0)_generation_1(1)` appeared 8× in the reference trace). Keep per-instance spans:
   - `dev_dur_ms` = **sum** of the per-instance device spans (not the min→max union — the union inflates the number and the bubble).
   - kernel attribution requires the kernel to fall inside an actual instance (not a gap between repeats), narrowest containing span wins.
   - stages with **no device marker** (e.g. the CPU-only warming stage `execute_context_0(0)_generation_0(0)`) fall back to the whole CPU window and are flagged `dev_marked=False`.

## Step 2 — Per-category kernel stats → xlsx

Classify every kernel name by **first-match-wins** precedence (order matters):

| precedence | category   | regex pattern (case-insensitive) |
|---|---|---|
| 1 | `triton_poi`  | `triton_poi` (torch.compile pointwise fusions) |
| 2 | `triton_red`  | `triton_red\|triton_per` (torch.compile reduction / persist-reduction fusions) |
| 3 | `moe_prep`    | `moe_align\|align_block\|moe_sum\|silu\|topk` (MoE align / sum / silu / topk-gating prep) |
| 4 | `moe_gemm`    | `MOE_\|MARLIN\|moe_gemm\|fused_moe\|moe_c::\|expert` (MoE gemm1/gemm2: asm/marlin, triton, ck) |
| 5 | `attn`        | `flash\|attention\|mla\|fla\|kkt\|fused_h_o\|gated_delta\|gdn\|causal_conv\|recurrent\|chunk_gated\|mamba\|_attn` (fa/mla/fla + GDN spec) |
| 6 | `w8a16_gemm`  | `gemm_w8a16\|w8a16\|deepgemm` (deepgemm w8a16 GEMM) |
| 7 | `gemm`        | `^Cij\|rocblas\|gemm\|GEMM` (rocblas/`Cijk_*` GEMMs) |
| 8 | `elementwise` | `rmsnorm\|layernorm\|mrope\|elementwise\|pointwise\|residual\|gelu` (incl. `at::native::*elementwise*` and fused rmsnorm+mrope) |
| 9 | `memory`      | `memcpy\|memset\|zero_kv\|fill\|_copy_\|direct_copy` |
| 10 | `communication` | `nccl\|allreduce\|all_gather\|reduce_scatter\|send\|recv` |
| 11 | `other`      | fallback |

Precedence traps observed on the qwen3.5 w8a16 trace:
- `MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT` must land in `moe_gemm`, so `moe_gemm` is checked **before** `w8a16_gemm`.
- `gemm_w8a16_int8_kernel` (deepgemm) must land in `w8a16_gemm`, so `w8a16_gemm` is checked **before** plain `gemm`.
- torch.compile `triton_red_fused__to_copy_add_mean_moe_forward_...` kernels contain fused-op names (`moe_forward`, `copy`) that would pollute `moe_gemm`/`memory` — so `triton_red` is checked **before** moe/gemm/memory.
- The `topkGatingSoftmax` (mangled `_ZN4vllm3moe17topkGatingSoftmax...`) belongs to `moe_prep` via `topk`.

Kernel names are shortened for display: strip `void `, demangle `_Z...` names via `c++filt` (fallback regex), cut template/argument signatures, drop `(anonymous namespace)` groups, keep the last `::` component — so `void MOE_W8A16_..._MARLIN_...<__half,...>` → `MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP`.

Per phase (all prefills / all decodes) and per single representative stage, the xlsx reports for each category:
- **count**, **total_ms**, **avg_ms** (per-kernel), **pct_of_kernel_sum** (share of the *actual* kernel time, sums to ~100% — not of the stage duration, since decode stages are mostly GPU bubble and would make every category look tiny)
- **n_launch** — how many of the category's kernels had an individual launch event (the rest are graph-replayed)
- **cpu_launch_ms** = sum of matching `hipModuleLaunchKernel`/`hipLaunchKernel`/`cudaLaunchKernel` call durations (CPU launch overhead per category; only counts directly-launched kernels)
- **top_kernels** — top distinct kernel names (shortened/demangled) with counts, e.g. `MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP×160`
- stage rows include **gpu_bubble_ms** = stage dur − Σ kernel dur (GPU idle gaps)

Correlation linking: kernel events carry `args.correlation` matching `cat=='cuda_runtime'` launch events (`hipModuleLaunchKernel` for eager, `hipLaunchKernel`/`hipExtModuleLaunchKernel` for the newer launcher, `hipGraphLaunch` for graph replay). Directly-launched kernels get a per-kernel launch; graph-replayed kernels (CUDA-graph / torch.compile) share one `hipGraphLaunch` — they surface as "no matching launch", which is expected, not a bug. On the reference trace only the GDN spec kernels (l2norm/gating/kkt/h_o) plus a few others were directly launched; everything else was graph-replayed.

## Usage

```bash
python scripts/analyze_trace.py TRACE.pt.trace.json.gz          # -> TRACE.xlsx
python scripts/analyze_trace.py TRACE.pt.trace.json.gz --text   # no pandas needed
python scripts/analyze_trace.py TRACE --categories cats.json    # custom [name, regex] rules
```

## xlsx sheet layout

- `meta` — trace path, event count, device thread, stages found, kernel attribution caveats
- `stages` — every stage: phase, dev_marked, n_dev_instances / n_cpu_instances, dev_dur_ms (sum of device instances), n_kernels, kernel_sum_ms, gpu_bubble_ms, cpu_dur_ms (per-instance **complete stage period** = GPU annotation duration; the engine thread is blocked on the GPU for the whole stage, so this equals dev_dur_ms for device-marked stages), cpu_marker_ms (the narrower CPU-side `user_annotation` sum, kept for reference)
- `prefill_cat` / `decode_cat` — aggregated per-category stats across all stages of the phase
- `prefill_first` / `decode_first` — per-category stats of the first (representative) stage of the phase
- `categories` — the rule table used

## Interpretation notes

- GPU-bubble (stage dur − kernel sum) > ~10% means kernels aren't back-to-back; look for host-side stalls, autotune, CPU-GPU syncs, or graph-capture gaps. The moe_align_block_size_global_mem_kernel anomaly (22.4% of a prefill stage) was the standout hotspot in the reference trace.
- A large `cpu_launch_ms` for a category with many small kernels (triton_poi / elementwise) indicates launch-bound; consider fusion or torch.compile.
- For decode stages, per-kernel avg vs prefill avg reveals kernels that scale badly with small token counts.
