from __future__ import annotations

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    scaled_mm_kernel,
)
from vllm.triton_utils import triton, tl
from vllm.v1.spec_decode.eagle import EagleProposer

K_FULL = 5120
K_SELECTOR = 1024
VOCAB = 248320
CANDIDATE_K = 2048
_INSTALLED = False

_CFG = {
    "bm": 32,
    "bn": 32,
    "bk": 512,
    "warps": 8,
    "waves": 2,
    "kp": 1,
    "lat": "mmac5-ds6",
}

# R067: the selector is a very different shape from the full-K candidate
# GEMM: M=1, K=1024, N=248320. Dedicated K100AI/gfx928 search found that
# doubling N tile width plus kpack=2 lifts effective weight bandwidth from
# ~589 to ~741 GB/s while remaining bitwise exact against R064.
_SELECTOR_CFG = {
    "bm": 32,
    "bn": 64,
    "bk": 512,
    "warps": 8,
    "waves": 2,
    "kp": 2,
    "lat": "mmac5-ds6",
}


def _scaled_logits_quantized(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Exact INT32 W8A8 scaled-mm for already-quantized one-token input."""
    w_nk = weight_nk[:n]
    ws = weight_scale[:n]
    w_kn = w_nk.t()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=xq.device)
    c = _SELECTOR_CFG if k == K_SELECTOR else _CFG
    grid = (triton.cdiv(n, c["bn"]),)
    scaled_mm_kernel[grid](
        xq,
        w_kn,
        xs,
        ws,
        out,
        None,
        1,
        n,
        k,
        xq.stride(0),
        xq.stride(1),
        w_kn.stride(0),
        w_kn.stride(1),
        out.stride(0),
        out.stride(1),
        tl.int32,
        BLOCK_SIZE_M=c["bm"],
        BLOCK_SIZE_N=c["bn"],
        BLOCK_SIZE_K=c["bk"],
        BLOCK_SIZE_SCALE_A=c["bm"],
        BLOCK_SIZE_SCALE_B=c["bn"],
        num_warps=c["warps"],
        num_stages=2,
        waves_per_eu=c["waves"],
        matrix_instr_nonkdim=16,
        kpack=c.get("kp", 1),
        mmac_layout_force=1,
        sched_latency=c["lat"],
    )
    return out


def _quantize_one(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    assert int(x.shape[0]) == 1 and int(x.shape[1]) == K_FULL
    xq, xs, zp = ops.scaled_int8_quant(x, None, None, symmetric=True)
    assert zp is None
    return xq, xs


def _ensure_selector_prepack(lm_head) -> None:
    if hasattr(lm_head, "k100_r062_selector_weight"):
        return
    # Distribution-neutral static subset: 1024 dimensions evenly cover all
    # 5120 hidden dimensions.  R061 held-out real MTP hidden states showed
    # 100% true-full-head-top1 recall inside approximate Top16K (and Top2K).
    idx_cpu = torch.linspace(0, K_FULL - 1, K_SELECTOR, dtype=torch.float64).round().to(torch.long)
    assert int(idx_cpu.unique().numel()) == K_SELECTOR
    idx = idx_cpu.to(lm_head.weight.device).contiguous()
    with torch.no_grad():
        wsub = torch.index_select(lm_head.weight, 1, idx).contiguous()
    lm_head.register_buffer("k100_r062_selector_idx", idx, persistent=False)
    lm_head.register_buffer("k100_r062_selector_weight", wsub, persistent=False)
    print(
        f"[K100 R067 selector] prepacked uniform K={K_SELECTOR} weight shape={tuple(wsub.shape)}",
        flush=True,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    orig_propose = EagleProposer.propose
    orig_greedy = EagleProposer._greedy_sample

    def propose(self, *args, **kwargs):
        self._k100_candidate_step = 0
        return orig_propose(self, *args, **kwargs)

    def greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
            return orig_greedy(self, hidden_states)

        step = int(getattr(self, "_k100_candidate_step", 0))
        self._k100_candidate_step = step + 1

        if int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]) != 1:
            return orig_greedy(self, hidden_states)

        lm_head = getattr(self.model, "lm_head", None)
        if (
            lm_head is None
            or getattr(lm_head, "weight", None) is None
            or lm_head.weight.dtype is not torch.int8
            or not hasattr(lm_head, "k100_27b_weight_scale")
            or tuple(lm_head.weight.shape) != (VOCAB, K_FULL)
        ):
            return orig_greedy(self, hidden_states)

        _ensure_selector_prepack(lm_head)

        # R064: every draft position independently runs the cheap 1024-D
        # full-vocabulary selector, then chooses its token with an exact
        # full-5120-D W8A8 GEMM on only Top2K.  No candidate list is shared
        # across steps, avoiding R062's trajectory degradation from a low-set-
        # recall approximate Top16K.  On 748 real R053 draft hidden states the
        # true full-head token was inside Top2K 99.20% overall (100% at steps
        # 0/1, 98.40% at steps 2/3).
        xq, xs = _quantize_one(hidden_states)
        xsub = torch.index_select(
            xq, 1, lm_head.k100_r062_selector_idx
        ).contiguous()
        approx = _scaled_logits_quantized(
            xsub,
            xs,
            lm_head.k100_r062_selector_weight,
            lm_head.k100_27b_weight_scale,
            VOCAB,
            K_SELECTOR,
        )
        ids = torch.topk(
            approx,
            CANDIDATE_K,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices[0].contiguous()
        cand_w = torch.index_select(lm_head.weight, 0, ids).contiguous()
        cand_s = torch.index_select(
            lm_head.k100_27b_weight_scale, 0, ids
        ).contiguous()
        exact = _scaled_logits_quantized(
            xq, xs, cand_w, cand_s, CANDIDATE_K, K_FULL
        )
        local_idx = exact.argmax(dim=-1).to(torch.long)
        return ids.index_select(0, local_idx)

    EagleProposer.propose = propose
    EagleProposer._greedy_sample = greedy
    _INSTALLED = True
    print(
        "[K100 R067 selector] K1024 BN64/kpack2 tuned -> exact Top2K",
        flush=True,
    )
