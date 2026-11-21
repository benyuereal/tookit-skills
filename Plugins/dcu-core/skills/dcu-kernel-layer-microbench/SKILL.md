---
name: dcu-kernel-layer-microbench
description: "DCU 上 LLM 推理 kernel 层微基准方法论：profiler 定位不到 decode 时间去向时，用微基准量化 kernel 开销与优化收益（BW100/DTK/SGLang DCU build 实测）。触发场景：整模型推理链路里缺 kernel 级 profile、要估算某个 kernel 在 step 时间里的占比、怀疑 kernel 融合/替换能带来收益。不适用：算子卡（op-*）的契约、bench 口径与改动判定走 ops 域的 op-bench / op-gate；不涉及具体算子验收的单次计时也走 op-bench。"
version: 1.0.1
author: DCU Engineer
license: MIT
---

# DCU kernel-layer micro-benchmark methodology

When the profiler can't tell you where decode time goes on DCU, micro-benchmarks can. This skill records the working method and the traps found while doing kernel-layer optimization of SGLang serving on Hygon DCU (BW100, DTK, SGLang 0.5.12+das DCU build). Complements (does not replace) `dcu-op-optimization` and `dcu-perf-profiling`.

## When to use
- Asked to squeeze more performance out of an LLM serving stack on DCU beyond launch-args tuning.
- Asked whether some upstream SGLang/vLLM optimization "can be used here".
- A candidate kernel/backend reports implausible speedups and needs a correctness gate.

## Core method: micro-benchmark rooflining (replaces missing GPU-kernel profiles)
1. **Establish the copy-BW roof**: timed `tensor.copy_` gives HBM peak (≈1.27 TB/s measured on BW100 single card). All memory-bound claims are % of this.
2. **GEMM benches must defeat L2 residency**: rotate 6 weight copies (~1 GB total) across timed iterations. Warm-cache benches showed a skinny GEMM at "94% peak" that cold rotation exposed at 42% — the difference decides whether a kernel target is real.
3. **GDN/hybrid-state decode kernels**: bench the actual SGLang fla kernel (`fused_recurrent_gated_delta_rule_packed_decode`) with real per-TP-rank shapes (v_heads×head_v×head_k fp32, r+w traffic). Measured 88-90% roof at B≥32 → no headroom; this is the expected state of the linear-attn decode path.
4. **Triton on DCU**: non-dot broadcast GEMV hits roof (~1.4 TB/s, 2.7x rocBLAS) but only at M=1; M≥8 needs compute beyond peak. `tl.dot` tensor-core split-K is 0.29-0.41x of rocBLAS at real verify work points (M=64-256) — do not plan DCU kernels around tl.dot.
5. Convert any micro-benchmark win to end-to-end share (e.g. draft-GEMV share of TPOT ≈ 3%) before building on it.

## Traps (each cost real time)
- **lightop.gemm returns stale results after first call** (relerr 1.0-1.6 on fresh inputs; suspected shape-keyed caching). It produced a fake "1.68x" GEMM win. Correctness gate for ANY candidate kernel: compare against `F.linear` ref with **fresh random inputs on every timed call**, and time with fresh inputs too.
- **SGLang profiler trace on DCU build**: `with_stack` traces balloon to GBs; flush can OOM-kill the server (SIGQUIT). Even with `with_stack:false` the trace may contain only python_function/cpu_op categories — no GPU kernel events. Don't loop on retries; go to micro-benchmarks. hipprof `--hip-trace` works but under Engine-embedded runs may export only warmup kernels.
- **Upstream feature portability must be audited against the installed build**: grep `server_args.py` for the flag before promising tests. On this DCU 0.5.12 build: `enable_fused_qk_norm_rope`/`enable_aiter_allreduce_fusion`/`DFLASH` present; `DSPARK`, `mamba_radix_cache_strategy`, `enable_linear_replayssm_spec` absent. Upstream cookbook's `--chunked-prefill-size 2048` (recommended for hybrid GDN on NVIDIA) measured **-5%/-4.5%** on DCU at 4K workload — upstream tuning numbers don't transfer.
- **End-to-end re-measure overrides micro-bench**: micro "wins" inside an already-fused stack (e.g. LightOp family ablation ±2%) can vanish; only quickbench-style A/B end-to-end numbers count (matches user's rule: success = end-to-end throughput/TTFT/TPOT only).

## Ops recipes
- **Transfer a bench script into a container**: bind-mount path on host = container path (check `docker inspect --format '{{range .Mounts}}...'`). ALWAYS verify landing (`ls -la hostfile`) after scp — a silent scp failure once wasted an entire queued benchmark run whose log looked fine.
- **Launch long benchmarks over ssh**: `ssh host 'cd D && setsid nohup bash -c "cmd > log 2>&1; echo DONE > flag" >/dev/null 2>&1 </dev/null & exit 0'` — ssh may still return exit 124 (timeout); the job usually survived. Verify by log file + `pgrep` before re-launching (a re-launch created duplicate-queue contention once).
- **HF model download from China-restricted DCU host**: `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download <repo> --local-dir <dir>` (new CLI syntax: `hf download`, not `huggingface-cli download`). Without `HF_HUB_DISABLE_XET=1` xet CAS server returns 401 through the mirror.

## DCU-specific spec-decode facts (SGLang DCU build)
- Tree EAGLE (`eagle-topk>1`) crashes at startup with fa3 backend: unpacked-tuple in `flashattention_backend.forward_extend` via `hybrid_linear_attn_backend` — tree path and fa3 are mutually exclusive; the runnable tree config (triton+page1) is end-to-end -18~-30%.
- DFLASH algorithm is accepted (overlap scheduler auto-disabled), needs external draft ckpt (e.g. `incoai/Qwen3.8-27B-DFlash2`, ~3.85 GB).

## Overlap note
内容与 `dcu-op-optimization`、`dcu-perf-profiling`、`llm-torch-profiler-analysis` 互补：那两个 skill 给 profiling 方法与闭环流程，本 skill 专注微基准手段与陷阱。
