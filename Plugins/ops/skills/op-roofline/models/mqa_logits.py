"""mqa_logits（DSA indexer 打分）的机器模型。

公式：logits[i,j] = Σ_h max(0, Q[i,h,:]·K[j,:]) · W[i,h]
  Q: [S, H, D]   K: [SKV, D]   W: [S, H]   D = indexer 头维（通常 128）

**算术强度只能按「kernel 实际实现的复用」算，不能按理论上限算。**
本文件初版在这里错过一次，方向完全反掉（2026-10-27 实测更正，见 kb/op-opt/mqa-logits-model-20261027.md §〇）：
初版假设「K 被同一 q-block 的全部 BM 行复用」，而真实 decode 内核的 grid 是
「一个 KV 块 × 一个 batch」——BM 行之间的复用**根本没有被实现**，next_n 也没并进 N 维。

所以模型只有一个核心变量 `q_per_ktile`：
    **多少条 query 共享同一份被流式读过的 K**
    ⇒ K 被整份读 `p = ceil(S / q_per_ktile)` 遍
    ⇒ AI = 2·q_per_ktile·H / kv_bytes

    q_per_ktile=1     ⇒ AI = 2H/bytes   （H=32/fp8 ⇒ 64 FLOP/B ⇒ **访存受限**）
        = 现成 decode 内核：只把 heads 并进 GEMM 的 N 维，query 之间不复用。
    q_per_ktile=next_n ⇒ AI 提高 next_n 倍（把 next_n 并进 N 维）
    q_per_ktile=BM    ⇒ AI = 2·BM·H/bytes（真 GEMM 把 BM 行 query 和 heads 一起当 N 维）
        = prefill 的正常 tile 形状 ⇒ 8192 FLOP/B ⇒ **算力受限**

机器平衡点 = peak_tf/bw = 540e12/1724e9 ≈ 313 FLOP/B。

推论（可直接指导设计，不必试）：
  1. **decode 是访存受限**，杠杆是「把 next_n / 跨 query 的复用并进 GEMM 的 N 维」= 重构数据流，
     不是调 tile 参数、不是提高 MMA 利用率。生产 fold 路径
     （`_deepgemm_fp8_paged_mqa_logits_nextn_fold`）做的正是这件事。
  2. **prefill 是算力受限**（q_per_ktile 大）。两档杠杆完全不同，报告里必须写档位。
  3. logits 输出 [S, SKV] **比 K 本身还大**（fp32：S·SKV·4 vs SKV·D·1）。设计上必须融合 topk。

**警告**：`peak_tf=540` 是在 K 很大的 dense GEMM 上标定的，本算子 K=D=128（fp8 16x16x32 只有 4 步），
该形状的实际峰值**未标定**。达成率的分子可信、分母不可信，引用时必须带这句。
"""

CU = 69


def analyze(S, H, SKV, D, BM=128, BN=128, kv_bytes=2, q_bytes=2,
            peak_tf=540.0, bw_gbs=1724.0, fuse_topk=False,
            q_per_ktile=1, next_n=1, fold_nextn=False):
    """返回各吞吐地板与算术强度。时间单位 µs。

    `q_per_ktile` 是本模型唯一的复用参数，**必须按 kernel 实际实现填**：
      - 现成 decode 内核 → 1（只用 heads 做 N，query 不复用）
      - 生产 fold 路径   → 用 `next_n=N, fold_nextn=True` 表达，等价于 q_per_ktile=N
      - 标准 prefill tile → BM（默认 128）

    不传 `q_per_ktile` 而传 `fold_nextn=True` 时，自动取 next_n。
    """
    if S <= 0 or H <= 0 or SKV <= 0 or D <= 0:
        raise SystemExit("S/H/SKV/D 必须为正")
    if fold_nextn:
        if next_n <= 0:
            raise SystemExit("fold_nextn=True 时 next_n 必须 >= 1")
        if S % next_n:
            raise SystemExit("fold_nextn 要求 S(%d) 能被 next_n(%d) 整除" % (S, next_n))
        if q_per_ktile == 1:                    # 未显式指定时才由 fold 推导
            q_per_ktile = next_n
    if q_per_ktile <= 0:
        raise SystemExit("q_per_ktile 必须 >= 1")

    flops = 2.0 * S * H * SKV * D
    mma = flops / 8192.0                      # 按 16x16x16 计条数（形状效率另说）

    # K 被整份流式读几遍 —— 只由「多少 query 共享一份 K」决定
    k_passes = -(-S // q_per_ktile)
    k_read = SKV * D * kv_bytes * k_passes
    q_read = S * H * D * q_bytes
    out = (S * SKV * 4) if not fuse_topk else (S * 64 * 4)

    mem_bytes = k_read + q_read + out
    mem_us = mem_bytes / (bw_gbs * 1e9) * 1e6
    comp_us = flops / (peak_tf * 1e12) * 1e6
    ai = flops / max(k_read, 1)

    return {
        "flops_tf_equiv": flops / 1e9, "mma": mma,
        "q_per_ktile": q_per_ktile, "k_passes": k_passes,
        "next_n": next_n, "fold_nextn": fold_nextn,
        "k_mb": k_read / 2**20, "q_mb": q_read / 2**20, "out_mb": out / 2**20,
        "mem_mb": mem_bytes / 2**20, "mem_us": mem_us, "comp_us": comp_us,
        "ai": ai,
        "balance_ai": peak_tf * 1e12 / (bw_gbs * 1e9),   # 机器平衡点
        "bound": "compute" if comp_us > mem_us else "memory",
        "peak_tf": peak_tf, "D": D,
    }


def verdict(r, measured_us=None):
    out = []
    out.append("复用 q_per_ktile=%d（机器平衡点 %.0f FLOP/B）"
               % (r["q_per_ktile"], r["balance_ai"]))
    out.append("算术强度 %.0f FLOP/B ⇒ 结构性**%s受限**"
               % (r["ai"], "算力" if r["bound"] == "compute" else "访存"))
    out.append("  K 流式读 %d 遍 ⇒ 读 K %.1f MB + 写 logits %.1f MB + 读 Q %.1f MB "
               "= %.1f MB ⇒ 内存地板 %.1f µs"
               % (r["k_passes"], r["k_mb"], r["out_mb"], r["q_mb"],
                  r["mem_mb"], r["mem_us"]))
    out.append("  %.1f GFLOP / 峰值 %.0f TF ⇒ 算力地板 %.1f µs"
               % (r["flops_tf_equiv"], r["peak_tf"], r["comp_us"]))
    if r["q_per_ktile"] == 1:
        out.append("  ⚠ q_per_ktile=1 = 现成 decode 内核：只把 heads 并进 GEMM 的 N 维，"
                   "query 之间完全不复用 K。")
        out.append("    把 next_n（或跨 query）并进 N 维可令 K 读量降到 1/N —— "
                   "这是**重构数据流**，不是调 tile 参数。")
    if r["out_mb"] > r["k_mb"]:
        out.append("  ⚠ logits 输出(%.1f MB) **大于** K 读入(%.1f MB) ⇒ 不融合 topk 就是在搬运比输入更大的中间量"
                   % (r["out_mb"], r["k_mb"]))
    if r["D"] <= 256:
        out.append("  ⚠ 峰值 %.0f TF 是在 **K 很大** 的 dense GEMM 上标定的；本算子 K=D=%d 只有 %d 步，"
                   "能否达到该峰值**未验证**——这是本模型最大的不确定项。"
                   % (r["peak_tf"], r["D"], -(-r["D"] // 32)))
        out.append("     要先做一个「同形状、只跑 QK、无 ReLU/无输出」的微基准，把该形状的实际 MMA 峰值钉住，"
                   "否则所有达成率都是拿错的尺子量。")
    if measured_us:
        m = measured_us
        tf = r["flops_tf_equiv"] * 1e9 / (m * 1e-6) / 1e12
        out.append("")
        out.append("实测 %.1f µs ⇒ %.1f TF = 峰值的 %.0f%%；内存地板占实测 %.0f%%"
                   % (m, tf, tf / r["peak_tf"] * 100, r["mem_us"] / m * 100))
        # 判据只基于「实测 vs 地板」。结构与实测不一致时，那种 disagreement 本身才是信息。
        if r["mem_us"] / m < 0.5:
            out.append("⇒ 实测比内存地板高 %.1f×，DRAM 只占 %.0f%% ⇒ 瓶颈**不在 DRAM**。"
                       % (m / r["mem_us"], r["mem_us"] / m * 100))
            out.append("   与「结构性访存受限」不矛盾：多读的那 %d 份被 L2 吸收了"
                       "（kb §四 实测 TCC_HIT 89%%）。" % (r["k_passes"] - 1))
            out.append("   ⇒ 瓶颈在**片上**（L2→CU 搬运、MMA 发射、同步）。"
                       "杠杆仍是减少重复搬运（fold/提高 q_per_ktile），而不是提高 DRAM 带宽。")
        else:
            out.append("⇒ 绑定约束：**访存**（K 被读 %d 遍，内存地板已占实测 %.0f%%）。"
                       % (r["k_passes"], r["mem_us"] / m * 100))
            out.append("   先查 next_n 折叠与跨 query 复用，别去调 tile 参数。")
    return "\n".join(out)
