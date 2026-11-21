---
name: dcu-gdn-prefill-kernel-opt
description: GDN prefill kernel 优化。优化 DCU 上 GDN prefill TTFT 时加载。
version: 1.0.0
author: DCU Engineer
license: MIT
---

# GDN 线性注意力 Prefill Kernel 优化（dcu-gdn-prefill-kernel-opt）

> 定位：Qwen3.5/3.8 系列 GDN hybrid 模型（vLLM `fla/ops` 路径）在 DCU 上跑 vLLM 时，**prefill TTFT 的 kernel 级优化方案**。
> 已验证收益（同卡 TP4、同插件状态 A/B）：**prefill 快 33-62%（32K 输入快 2.7 倍，随长度单调放大），decode 快 ~8%**。
> 与 `dcu-inference-optimization` §12.4 的关系：那边是 TTFT O(N)+O(N²) 建模方法论（解释收益为什么随长度放大），本 skill 是具体 kernel 改法。
> 与 `dcu-kernel-layer-microbench` 的关系：那边实测 GDN **decode** kernel 88-90% roof 无 headroom（所以 decode 只快 8%），本 skill 优化 **prefill chunk kernel**。

## §0 为什么优化 GDN kernel

Qwen3.8-27B（model_type=qwen3_5）是 GDN 混合架构：**64 层里 48 层 linear_attention（GDN），16 层 full_attention**——75% 的层走 GDN 路径，GDN chunked kernel 快慢直接决定整体性能。

GDN 层形状（优化 gate 精确匹配）：`H=48`（value heads）、`Hg=16`（query heads）、`K=128`、`V=128`、`BT=64`（chunk size）、bf16。

**核心洞察**：`H/Hg = 3`，每个 query head 被 3 个 value head 共享。原版 `chunk_fwd_kernel_o` 的 grid 是 `(V/BV, NT, B*H)`——每个 value head 各算一遍 `b_A = Q@K`，但 3 个 value head 共享同一份 Q/K，**QK 矩阵被冗余算 3 次、K 被冗余加载 3 次**。

## §1 优化 #1（prefill 大头）：chunk_o.py 新增 share_qk3 kernel

**文件**：`vllm/model_executor/layers/fla/ops/chunk_o.py`
**做法**：保留原 `chunk_fwd_kernel_o` 不动，新增 `chunk_fwd_kernel_o_share_qk3`（grid 改 `(NT, B*Hg)`，一个 program 算一次 `b_A`，`static_range` 循环 3 个 value head 复用，后两个 head 不再加载 K），在 `chunk_fwd_o()` 加 env gate + 形状 gate。

### 1.1 新增 kernel（完整代码）

```python
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
```

### 1.2 chunk_fwd_o() 里的 gate

```python
    o = torch.empty_like(v)

    use_share_qk3 = (
        os.getenv("VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3", "1").lower()
        in ("1", "true", "yes", "on")
        and H == 48 and Hg == 16 and K == 128 and V == 128
        and BT == 64 and q.dtype == torch.bfloat16
    )
    if use_share_qk3:
        chunk_fwd_kernel_o_share_qk3[(NT, B * Hg)](
            q, k, v, h, g, o, cu_seqlens, chunk_indices, scale,
            T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
        )
        return o
    # 原路径 fallback（grid = (V/BV, NT, B*H)）
```

文件顶部需 `import os`。

**gate 纪律**：
- env `VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3` 默认开，设 `0` 回退原 kernel 做 A/B。
- 形状 gate 精确锁 Qwen3.8-27B GDN 形状，其它模型/形状自动 fallback，**不误伤**。
- `static_range(1, 3)` 硬编码 3 个 value head，依赖 `H/Hg == 3`——形状 gate 必须含 `H==48 and Hg==16`，换模型（如 H/Hg≠3）要改循环上界。

## §2 优化 #2：solve_tril.py 避免整矩阵零化

**文件**：`vllm/model_executor/layers/fla/ops/solve_tril.py`

1. `Ai = torch.zeros_like(A, dtype=output_dtype)` → `torch.empty_like`（不预清零）。
2. **必须**在 kernel 里显式零化严格上三角的 6 个 16×16 块（`Ai_12/Ai_13/Ai_14/Ai_23/Ai_24/Ai_34`，坐标 (0,16)(0,32)(0,48)(16,32)(16,48)(32,48)），用 `tl.store`（block_ptr）或 `desc_o.store`（TMA，更快）。

> ⚠️ **正确性关键**：`empty_like` 后不补 0 这 6 块，上三角是垃圾值，三角求解结果错，数学/代码类任务掉分。改完必跑精度验证（§5）。

## §3 优化 #3（可选）：chunk_delta_h.py 删 aiter 分支

**文件**：`vllm/model_executor/layers/fla/ops/chunk_delta_h.py`

hcu 插件 patch 进来的版本有 aiter 自定义 kernel 分支（`VLLM_HCU_USE_CUSTOM_AITER_FLA` 命中走 `launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`）。删掉该分支 + 顶部两个 import，统一走 Triton kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。收益不如 #1，属统一/稳定性质；环境 aiter 路径没问题时可只做 #1+#2。

## §4 实测收益（2026-10-08，4×DCU TP4，Qwen3.8-27B，vLLM 0.18.1+hcu 插件）

| 输入 | TTFT 原版→优化 | TPOT 原版→优化 |
|------|--------------|--------------|
| 8K  | 2.56s → 1.71s（快 33%） | 46.8 → 43.0ms（快 8%） |
| 16K | 7.07s → 3.58s（快 49%） | 48.2 → 44.1ms（快 8%） |
| 32K | 21.82s → 8.20s（快 62%） | 50.6 → 46.7ms（快 8%） |

**特征**：prefill 收益随输入长度**单调放大**（kernel 级优化特征，与 `dcu-inference-optimization` §12.4 的 O(N) 线性层建模一致）；decode 稳定 ~8%（decode 是 memory-bound 状态递推，见 `dcu-kernel-layer-microbench`：GDN decode kernel 已 88-90% roof）。

## §5 验证方法（必须做，别只信性能）

1. **性能 A/B**：同卡、同插件状态、唯一变量是 kernel。`VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3=1` vs `=0` 各起服务，固定 token 长度 prompt 测 8K/16K/32K × 128 输出的 TTFT/TPOT。
2. **精度**：GSM8K 或数学 benchmark，优化 vs 原版 ±0.5pp 内；temperature=0 下逐 token 对比应一致。重点盯 §2 的 6 块零化——没补 0 数学/代码任务会掉分。

## §6 踩坑

1. **`VLLM_PLUGINS=`（空串）会静默禁用 hcu 插件** ★最隐蔽：未设置（None）→ 加载全部；空串 → 解析成 `['']` → `"hcu" in ['']` False → 插件被跳过，**无任何报错**，只少了 "Platform plugin hcu is activated" 一行。后果：attention backend/算子分发全变，decode 掉速 7-8%（曾误判为 fork 代码问题）。A/B 时两组 `VLLM_PLUGINS` 必须完全一致。排查：`grep "Platform plugin hcu is activated" server.log`。
2. **源码编译 fork 会覆盖 hcu 插件 patch 过的文件**：插件靠字符串 patch 往 vLLM 源码注入类（如 `CpCommonAttentionMetadata`），源码编译会把 patch 过的文件覆盖回原版 → 插件激活时 `ImportError`。修复：编译后手动 `from vllm_hcu.patches import apply_patches; apply_patches()`（幂等）或跑 `vllm-hcu-apply-patches`。验证：`grep -c CpCommonAttentionMetadata $VLLM/v1/attention/backend.py` ≥1。
3. **共享机选卡**：跑前 `hy-smi` 看显存；杀自己残留 `pgrep -f "vllm serve" | xargs -r kill -9`，注意 EngineCore/Worker_TP 子进程要单独杀（父进程 kill 后子进程可能残留占显存）；绝不杀别人的服务。
4. **csrc MMAC i8 stub**：fork 的 `csrc/rocm/attention.cu` 把 `v_mmac_i32_16x16x32_i8` stub 掉（dcc 25.10 不支持）——int8 矩阵乘，bf16 dense 用不到，不影响本方案；换 DTK 版本时注意是否要恢复。

## §7 交付清单

- [ ] `chunk_o.py`：新增 share_qk3 kernel + gate（§1）
- [ ] `solve_tril.py`：`empty_like` + 6 上三角块补 0（§2，**重点验精度**）
- [ ] `chunk_delta_h.py`：删 aiter 分支（§3，可选）
- [ ] A/B 性能：env 开关 1 vs 0，同卡同插件（§5.1）
- [ ] 精度：GSM8K ±0.5pp（§5.2）
- [ ] 确认 `VLLM_PLUGINS` 两组一致（§6.1）
- [ ] 若源码编译：重跑 `apply_patches()`（§6.2）

## 使用须知

- 本方案针对 **vLLM 0.18.1 + hcu 插件** 的 `fla/ops` 路径；SGLang 的 GDN kernel 在 `sglang/kernels/ops/attention/fla/`，改法思路相同（QK 共享）但代码不同，勿直接搬。
- 形状 gate 锁死 Qwen3.8-27B；其它 GDN 模型（H/Hg 不同）需按 §1.2 纪律调整 gate 和 `static_range` 上界。
- 性能数字是 2026-10-08 单点实测，随硬件/DTK/版本变化，只作量级参考。
- 完整方案文档（含 stock/fork 源码对照）：原始实测文档（未随包分发） 
