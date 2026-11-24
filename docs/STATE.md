# 状态看板

> 自动生成，勿手改。重新生成：`python3 -I scripts/state.py --write`
> 阶段定义来自 `Plugins/ops/op-atlas/validate.py` 的 `STAGES` / `STAGE_NEEDS` / `BLOCKED_BY`。

阶段分布（自动生成，勿手改）
S0 画像      0 张   前置产出：—
S1 天花板     0 张   前置产出：semantics, floor_inputs
S2 基线      0 张   前置产出：last_verdict.comparison
S3 归因      0 张   前置产出：last_verdict.bottleneck
S4 资源画像    0 张   前置产出：last_verdict.resources
S5 选路      0 张   前置产出：last_verdict.next_probe
S6 实验      0 张   前置产出：last_verdict.rounds
S7 收敛      1 张   前置产出：last_verdict.verdict
blocked    0 张

## 分阶段清单

### S0 画像（0 张）
前置产出：—
- （无）

### S1 天花板（0 张）
前置产出：semantics, floor_inputs
- （无）

### S2 基线（0 张）
前置产出：last_verdict.comparison
- （无）

### S3 归因（0 张）
前置产出：last_verdict.bottleneck
- （无）

### S4 资源画像（0 张）
前置产出：last_verdict.resources
- （无）

### S5 选路（0 张）
前置产出：last_verdict.next_probe
- （无）

### S6 实验（0 张）
前置产出：last_verdict.rounds
- （无）

### S7 收敛（1 张）
前置产出：last_verdict.verdict
- rmsnorm-fused

### blocked（0 张）
- （无）

### 未接入（52 张，未写 progress.stage）
- batched-gemm
- causal-conv1d
- chunked-scan
- dequant-fp8-w4
- embedding-lookup
- flash-attention-fwd
- fused-add-rmsnorm
- fusion-add-rmsnorm-quant
- fusion-rmsnorm-rope
- fusion-rope-kv-write
- fusion-silu-mul-quant
- gated-linear-attn
- gemm-bf16
- gemm-fp8
- gemm-int8
- gemm-swiglu-epilogue
- hadamard-transform
- hc-split-sinkhorn
- kv-cache-paged-write
- layernorm
- lm-head-with-sampling
- mha-prefill
- mhc-fused
- mla-absorb
- mla-full
- mla-kv-cache-layout
- mla-kv-cache
- mlp-gelu-geglu
- mlp-swiglu
- moe-align-permute
- moe-combine-reduce
- moe-dispatch-combine
- moe-grouped-gemm-bf16
- moe-grouped-gemm
- moe-pipeline-fusion
- moe-shared-expert
- moe-swiglu-fused
- moe-topk-gating
- moe-w4a16
- moe-w8a8
- mqa-logits
- mtp-draft
- paged-attention-decode
- qk-norm
- quant-per-token
- rope-mla
- rope
- sampling-topk-topp
- sparse-mla
- topk-block-index
- verify-accept
- weight-prepack

### 非法（0 张）
- （无）
