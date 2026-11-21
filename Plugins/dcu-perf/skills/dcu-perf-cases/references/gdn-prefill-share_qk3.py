# GDN prefill chunk_o.py 的 share_qk3 kernel（QK 共享去冗余）
# 来源：dcu-perf-cases 案例 39（Qwen3.8-27B GDN prefill kernel 优化，2026-10-08 <device-B> 节点 4×DCU TP4 实测）
# 目标文件：vllm/model_executor/layers/fla/ops/chunk_o.py
# 做法：保留原 chunk_fwd_kernel_o 不动，新增 chunk_fwd_kernel_o_share_qk3，
#       在 chunk_fwd_o() 加 env gate + 形状 gate。文件顶部需 import os。
#
# 核心洞察：H/Hg = 3，每个 query head 被 3 个 value head 共享。
# 原版 grid (V/BV, NT, B*H) 每个 value head 各算一遍 b_A=Q@K，QK 冗余算 3 次、K 冗余加载 3 次。
# 本 kernel grid 改 (NT, B*Hg)，一个 program 算一次 b_A，static_range 循环 3 个 value head 复用，
# 后两个 head 不再加载 K。

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=2)
        for BK in [64, 128]
        for num_warps in [4, 8]
    ],
    key=["K", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o_share_qk3(
    q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bhg = tl.program_id(0), tl.program_id(1)
    i_b, i_hg = i_bhg // Hg, i_bhg % Hg

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos = i_b * T

    q_head = q + (bos * Hg + i_hg) * K
    k_head = k + (bos * Hg + i_hg) * K
    value_head0 = i_hg * 3

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_o = tl.zeros([BT, V], dtype=tl.float32)

    # Head zero computes its QH output while the shared raw QK matrix is live.
    h0 = h + (i_tg * H + value_head0).to(tl.int64) * V * K
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q_head, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k_head, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h0, (V, K), (K, 1), (0, i_k * BK), (V, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

    if USE_G:
        g0 = g + bos * H + value_head0
        p_g = tl.make_block_ptr(g0, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A_head = b_A * exp(b_g[:, None] - b_g[None, :])
    else:
        b_A_head = b_A
    b_A_head = tl.where(m_A, b_A_head, 0)
    v0 = v + (bos * H + value_head0) * V
    o0 = o + (bos * H + value_head0) * V
    p_v = tl.make_block_ptr(v0, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
    p_o = tl.make_block_ptr(o0, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_out = b_o * scale + tl.dot(b_A_head.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_out.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    # The remaining two value heads reuse b_A and never load K.
    for value_delta in tl.static_range(1, 3):
        value_head = value_head0 + value_delta
        h_head = h + (i_tg * H + value_head).to(tl.int64) * V * K
        b_o = tl.zeros([BT, V], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_head, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h_head, (V, K), (K, 1), (0, i_k * BK), (V, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q, tl.trans(b_h))

        if USE_G:
            g_head = g + bos * H + value_head
            p_g = tl.make_block_ptr(g_head, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o = b_o * exp(b_g)[:, None]
            b_A_head = b_A * exp(b_g[:, None] - b_g[None, :])
        else:
            b_A_head = b_A
        b_A_head = tl.where(m_A, b_A_head, 0)
        v_head = v + (bos * H + value_head) * V
        o_head = o + (bos * H + value_head) * V
        p_v = tl.make_block_ptr(v_head, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
        p_o = tl.make_block_ptr(o_head, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_out = b_o * scale + tl.dot(b_A_head.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_out.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ---- chunk_fwd_o() 里的 gate（接在 o = torch.empty_like(v) 之后）----
#
#     o = torch.empty_like(v)
#
#     use_share_qk3 = (
#         os.getenv("VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3", "1").lower()
#         in ("1", "true", "yes", "on")
#         and H == 48 and Hg == 16 and K == 128 and V == 128
#         and BT == 64 and q.dtype == torch.bfloat16
#     )
#     if use_share_qk3:
#         chunk_fwd_kernel_o_share_qk3[(NT, B * Hg)](
#             q, k, v, h, g, o, cu_seqlens, chunk_indices, scale,
#             T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
#         )
#         return o
#     # 原路径 fallback（grid = (V/BV, NT, B*H)）
#
# gate 纪律：
# - env VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3 默认开，设 0 回退原 kernel 做 A/B。
# - 形状 gate 精确锁 Qwen3.8-27B GDN 形状，其它模型/形状自动 fallback，不误伤。
# - static_range(1, 3) 硬编码 3 个 value head，依赖 H/Hg == 3——形状 gate 必须含
#   H==48 and Hg==16，换模型（如 H/Hg≠3）要改循环上界。
