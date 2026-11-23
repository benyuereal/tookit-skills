"""稀疏 MLA 前向的机器模型：给定设计 → 各吞吐地板 + 绑定约束。

不预测绝对耗时（那需要把依赖链也建模），只回答**哪个上限先到顶**。
这个回答足以决定「下一刀往哪切」，而且不会被拟合噪声污染。

关键量全部来自可数的东西：
  MMAs      = CTAs × blocks × [(bm/16)(bi/16)(d/16) × 2]      # QK 与 PV 条数相同
  读取字节  = CTAs × blocks × bi × d × 2B                     # 每 CTA 各自 gather
  CTA 轮数  = ceil(CTAs / (CU × CTA每CU))

三个地板：
  内存地板 = 字节 / 实测流式峰值带宽
  算力地板 = MMAs / (31.6 条/cycle × 时钟)
  并发上限 = 每 CU 驻留 CTA 数 × CU 数 × 每 CTA 的 wave 数（决定能有多少个独立依赖链同时在飞）
"""

CU_FALLBACK = 64            # constants.yaml:compute.cu_count（gfx938 实测，2026-11-13 由 69 更正为 64）
CU_FALLBACK_SOURCE = "constants.yaml:compute.cu_count"


# derive.py 的动态分发契约。
MODEL_SPEC = {
    "fn": "analyze",
    "args": ["s", "h", "topk", "d", "bm", "thr", "bi", "split"],
    "peak": "bf16",         # 分母用 constants.compute.bf16_gemm_peak_tf
    "bound": "mixed",
}

def _cu_count(device=None):
    """从设备查询 CU 数；读不到时退回 constants.yaml 的实测值并打印来源。

    为什么不再写死 69：原值 69 来自 hy-smi 的逐卡读数（69/70/71），与
    torch / hbird-gemm README / dmesg SE 位图三方给出的 64 不符，会让所有
    按 CU 归一的量（per-CU 摊销、rounds、waves_per_cu、MFMA/CU）偏 ~8%。
    详见 constants.yaml:compute.cu_count 的更正记录。
    """
    try:
        import torch
        n = torch.cuda.get_device_properties(device).multi_processor_count
        if n:
            return int(n)
    except Exception:
        pass
    import sys
    print("caveat CU 数无法从设备读取，退回 %s = %d" % (CU_FALLBACK_SOURCE, CU_FALLBACK),
          file=sys.stderr)
    return CU_FALLBACK


MMA_PEAK_PER_CYCLE = 25.2   # constants.yaml:compute.mma_issue_peak
MMA_PEAK_SOURCE = "constants.yaml:compute.mma_issue_peak（由实测 279 TF @1350MHz 反解，2026-11-23）"
# 历史（同一天改了两次，都值得记）：
#   19.7 ← 由「2048^3 实测 217 TF = 88% 峰值」反推，小形状不能当峰值
#   31.6 ← 由单轮采样的偶发高值 310 TF 反解，多轮采样后实际是 279
#   25.2 ← 279 TF / (8192 × 1350 MHz)。与 fp8 标称 560 的 2:1 关系自洽（279/280=99.7%）
CU = _cu_count()
# 每 CU 的 VGPR 文件（32-bit 寄存器数）。本项目所有 kernel 都用
# `-mllvm=-support-768-vgprs=true` 编译，故为 768 模式 = 4 SIMD × 64 lane × 768 = 196608；
# 未开该 flag 时才是 4 × 64 × 256 = 65536。
# 交叉验证（2026-11-13）：hbird-gemm prefill kernel 172 VGPR × 512 线程实测 2 CTA/CU，
# 只有 196608 能复现（65536 会算出 0）；本仓库的 LDS 探针（28672→40960 强制 1 CTA）实测 +68%，
# 也与「基线是 2 CTA/CU」一致。改回 65536 前先读这条。
VF_LIMIT = 196608
VF_LIMIT_NO768 = 65536
LDS_PER_CU = 64 * 1024
WAVE = 64


def analyze(S, H, topk, d, bm, thr, bi, split, replicate_h=None,
            clock_ghz=1.15, bw_gbs=1634.0, nvalid=None, lds_bytes=None):
    nvalid = topk if nvalid is None else nvalid
    replicate_h = replicate_h if replicate_h is not None else (2 if H >= 64 else 4 if H >= 32 else 1)
    h_per_block = H // replicate_h

    blocks = -(-nvalid // bi)
    ctas = S * replicate_h * split          # split-K：把每 CTA 的块序列再切成 split 段

    # --- FLOP 总量：与切分方式无关 ---------------------------------------
    # QK^T 是 [S,H]×[d] × [d]×[topk]，PV 是 [topk]×[S] × [topk]×[d]，各一遍。
    # 所以 FLOP = 4 × S × H × topk × d，**不含 bm、不含 replicate_h、不含 split**。
    #
    # 旧写法 mmas = ctas*blocks*(bm/16)(bi/16)(d/16)*2 代数上等于
    #   FLOP = 4 × S × replicate_h × bm × d × topk
    # 只有 replicate_h × bm == H 时才等于真值。默认 replicate_h 是按 H 分桶拍的，
    # 于是 H=32/bm=32 时高估 4×、H=128 时低估 2× —— 算力地板跟着错同样倍数，
    # 足以翻转 cost_bound。我们的形状恰好 H=64/bm=32/rep=2 命中，属侥幸（D2）。
    flops = 4.0 * S * H * topk * d
    mmas = flops / 8192.0                                        # 每条 MFMA 8192 FLOP

    kv_bytes = S * replicate_h * blocks * bi * d * 2   # 读 KV 的总量；split 只切分不重读

    # split-K 的额外代价：每段结局要写出 partial（m, l, acc），再一次性 combine。
    # 每个 CTA 的累加器是 [bm, d] fp32，写一遍再读一遍 ⇒ ctas × bm × d × 4B × 2。
    # 只在 split>1 时计入，且**不摊进 mem_us/mma_us** —— 它是并行度换来的固定开销。
    combine_bytes = (ctas * bm * d * 4 * 2) if split > 1 else 0

    mem_us = kv_bytes / (bw_gbs * 1e9) * 1e6
    mma_us = mmas / (MMA_PEAK_PER_CYCLE * clock_ghz * 1e9) * 1e6
    combine_us = combine_bytes / (bw_gbs * 1e9) * 1e6

    # --- LDS 预算（R6）------------------------------------------------------
    # 从生成 HIP 的 buf_dyn_shmem 偏移反出的结构式：
    #   3 份 KV 列分片（每片列宽固定 d/4=128；第 4 片与别的 buffer 别名复用）
    #   + 1 份 S 打分缓冲 [bm, bi] fp32
    # 注意：列分片宽是 d//4（kernel 结构决定），与 num_split（序列维切分）无关。
    # 校准：bm32/bi32 = 28672 B、bm64/bi32 = 32768 B，与生成代码逐个偏移完全一致。
    lds_used = 3 * bi * (d // 4) * 2 + bm * bi * 4
    lds_ratio = lds_used / LDS_PER_CU

    # 片上预算：按线程上限 256 VGPR 估（下界；真实值看 tools/hipbudget.py 再查编译产物）
    # 整数下取整：驻留 CTA 数只能是整数，2.28 要报 2（2026-11-13 修正，之前显示 2.3 误导）
    cta_by_vgpr = int(VF_LIMIT // (256.0 * thr))
    cta_by_lds = int(LDS_PER_CU // lds_used)
    cta_per_cu = max(1, min(cta_by_vgpr, cta_by_lds))
    waves = ctas * (thr // WAVE)
    waves_per_cu = waves / CU
    rounds = -(-ctas // (CU * int(cta_per_cu)))

    # 每 CTA 的串行长度：blocks 个「QK → 归约 → softmax → PV」链，split 后除以 split
    serial_chain = -(-blocks // split)

    # 摊到「每 CU 每块」的口径：总时间 × CU / 总块数 = 该 CU 处理一块要多久。
    # 这个口径下内存地板与算力地板可以直接相加/取大，做重叠分析。
    block_slots_total = ctas * serial_chain
    per_block = {
        "mem_us": mem_us * CU / block_slots_total,
        "mma_us": mma_us * CU / block_slots_total,
        "slots_per_cu": block_slots_total / CU,
    }

    return {
        "blocks": blocks, "ctas": ctas, "mmas": mmas, "kv_mb": kv_bytes / 2**20,
        "mem_us": mem_us, "mma_us": mma_us, "split": split,
        "combine_us": combine_us, "combine_mb": combine_bytes / 2**20,
        "per_block": per_block,
        "cta_per_cu": cta_per_cu, "rounds": rounds,
        "lds_kb": lds_used / 1024.0, "lds_ratio": lds_ratio,
        "lds_ok": lds_ratio <= 0.85,
        "waves_per_cu": waves_per_cu, "serial_chain": serial_chain,
        "h_per_block": h_per_block,
    }


def decompose(r, measured_us):
    """把实测摊到「每 CU 每块」，看它离「串行地板」和「重叠地板」各多远。"""
    pb = r["per_block"]
    m = measured_us * CU / (r["ctas"] * r["blocks"])
    serial = pb["mem_us"] + pb["mma_us"]
    overlap = max(pb["mem_us"], pb["mma_us"])
    return {
        "measured": m, "mem": pb["mem_us"], "mma": pb["mma_us"],
        "serial": serial, "overlap": overlap,
        "unexplained": m - serial,
        "vs_overlap": m / overlap,
    }


def verdict(r, measured_us=None):
    """给出绑定约束。判据：某地板到实测的占比。"""
    if measured_us is None:
        return "未给实测，只能说内存地板 %.0f µs / 算力地板 %.0f µs" % (r["mem_us"], r["mma_us"])
    m = measured_us
    d = decompose(r, m)
    pb = r["per_block"]
    out = []

    if not r["lds_ok"]:
        out.append("blocked **LDS 预算 %.0f KB = %.0f%% of 64 KB（R6）⇒ 本次数值预测作废。**"
                   % (r["lds_kb"], r["lds_ratio"] * 100))
        if r["lds_ratio"] > 1.0:
            out.append("   直接超限，编译期就该否掉，不需要上卡。")
        else:
            out.append("   剩余 %.0f KB 不足以再放任何 buffer，编译器会把东西挤进 VGPR/scratch，"
                       "并发与向量化会同时崩——此时「减少块数省固定开销」的账算不成立。"
                       % (64 - r["lds_kb"]))
        out.append("")

    out.append("每 CU 每块：内存 %.2f µs + 算力 %.2f µs ⇒ 串行地板 %.2f µs / 重叠地板 %.2f µs"
               % (d["mem"], d["mma"], d["serial"], d["overlap"]))
    out.append("实测每 CU 每块 %.2f µs  ⇒ 比重叠地板慢 %.1f×，比串行地板慢 %.1f×"
               % (d["measured"], d["vs_overlap"], d["measured"] / d["serial"]))
    if d["unexplained"] > 0.15 * d["serial"]:
        out.append("其中 **%.2f µs 连「内存+算力完全串行」都解释不了**（占实测 %.0f%%）——"
                   "这段是每块都有的固定延迟（屏障/归约/依赖等待），跟块大小无关。"
                   % (d["unexplained"], d["unexplained"] / d["measured"] * 100))

    # 一个 CTA 都装不下第二个 ⇒ 没有任何可切换对象来填这段等待（R3）
    if r["cta_per_cu"] <= 1.0:
        out.append("每 CU 只驻 %.1f 个 CTA ⇒ 等待期间没有别的 CTA 可调度（R3）。"
                   % r["cta_per_cu"])

    out.append("")
    out.append("⇒ 可达上界：把内存与算力完全重叠 = %.0f µs（现 %.0f µs，%.1f× 空间）"
               % (d["overlap"] * pb["slots_per_cu"], m, m / (d["overlap"] * pb["slots_per_cu"])))
    out.append("⇒ 绑定约束：**依赖链/并发**，不是吞吐。要动的是"
               "「每块固定开销」与「同时在飞的独立依赖链数」，不是继续抠指令数。")
    return "\n".join(out)
