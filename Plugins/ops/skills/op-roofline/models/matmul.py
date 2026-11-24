"""matmul 的 tile 级模型：给定 (M,N,K) 与 tile 配置，算资源、地板、预测。

为什么要这个模型
----------------
`templates.py` 的 matmul 模板是**流量级**——只算 bytes/flops，不碰 tile。
所以 agent 知道「这个 GEMM 该达到 3941 µs」，但**不知道怎么调 tile 才能达到**。

本模型补 tile 那一层：BM/BN/BK/warps/stages → LDS / VGPR / CTA/CU / 预测 MFU。

预测 MFU 的公式从哪来（不许拍脑袋）
-----------------------------------
三个实测点，来自 hbird-gemm 的同机实测（/workspace/hbird-gemm/docs/DESIGN.md）
与 hbird Phase 9 的 ladder：

  ① 纯 mmac issue floor         1.6017 ms = 686.5 TF   （无内存/无等待/无 barrier）
  ② 我们最好的结构 256×256×64          1.6708 ms = 658.1 TF（上述 floor 的 96%）
  ③ hipBLASLt 同形状                   1.8931 ms = 581 TF  （82-85% of floor）

三个点给出的规律：
  - **ISA 上限是 686.5 TF**（16×16×32 fp8 mmac，无更宽的指令）
  - 达到 ISA 上限 96% 需要「C store 被隐藏 + A 读不阻塞 + barrier 摊薄」
  - 只有 76-77%（我们的结构）到 82-85%（hipBLASLt）是**常规 tile 能到的区间**

所以预测 MFU 不能简单按 tile 大小插值——**tile 只决定两件事**：
  1. 能不能装下（LDS/VGPR 上限）→ 不能就 illegal
  2. 驻留 CTA/CU 够不够（决定延迟隐藏）
至于「装得下且驻留够」之后能到多少，取决于实现质量，tile 已经解释不了。

本模型因此分两段：
  legal 判定  → 硬约束（LDS <= 64KB, VGPR <= 255, grid >= 64）
  预测 MFU    → 用**分段经验式**，每段标注依据，并给 confidence

**这不是物理定律。** 它是从 3 个实测点推的经验式，只在 bf16/fp8 大 GEMM 上验证过。
输出必带 `model_limit`，别拿它当「预测即事实」。
"""

MODEL_SPEC = {
    "fn": "analyze",
    "args": ["m", "n", "k", "bm", "bn", "bk", "warps", "stages", "split_k"],
    "peak": "bf16",
    "bound": "compute",
}

# ---- 机器常数（与 constants.yaml 保持一致；取不到就用这里的回退值并标注）
PEAK_TF_BF16 = 279.0            # constants.yaml compute.bf16_gemm_peak_tf（achievable）
PEAK_BW_GBS = 1730.0            # bw-ladder 2_rw_2to1（GEMM 是 2读1写）
LDS_PER_CU = 65536              # B
VGPR_PER_CU = 196608            # 768-VGPR 模式
CU_COUNT = 64
WAVE_SIZE = 64
MFMA_FLOPS = 8192               # 16x16x16 bf16

# ---- 经验式用的锚点（见文件头「三个实测点」）
UTIL_CEILING = 0.96             # 极好实现能达到 ISA floor 的 96%（hbird 实测）
UTIL_TYPICAL = 0.80             # 常规已调优实现（hipBLASLt 的 82-85% 的中位）
UTIL_FLOOR = 0.55               # 驻留不足时的下限（低于此说明 tile 选坏了）


def _resources(bm, bn, bk, warps, stages, dtype_bytes, b_via_vgpr=True,
               lds_rows=None):
    """bm 决定累加器（影响 VGPR）；lds_rows 决定 LDS 占用（A 子块高度）。

    **这两个必须分开**：hbird 的 256×256 tile 用 32 行子块搬 A，
    LDS 只占 32*64*2*2=8KB，但**累加器仍是 256×256/512=128 VGPR**。
    第一版把整块 bm 传进 LDS 公式，算出 64KB，把 LDS 腿压到 1 CTA/CU ——
    比真实值严了 4 倍。
    """
    rows = lds_rows if lds_rows else bm
    """资源用量。

    **LDS 公式的关键修正（2026-11-24，被 hbird 实测推翻过一次）**

    我第一版写 `(BM*BK + BK*BN) * dtype * stages`，对 256×256×64 双缓冲
    算出 131072 B —— 但 hbird 的同 tile 实测 LDS 只要 **32768 B per CTA**
    （64 KB/CU 装 2 个 CTA，16 waves/CU）。

    根因：**B 不走 LDS，直接进 VGPR**（hbird DESIGN.md Phase 1：
    "A staged through LDS, B straight to VGPR"）。
    走 LDS 的只有 A：`BM*BK*dtype*stages` = 256*64*2*2 = 65536 B…
    仍不等于 32768。再查：实际是 **2 × 16 KB（BK=64）**，即单 stage 只 16 KB —
    说明 A 也是按 32-row 子块分步搬的（Phase 2：32-row sub-quarter, 3 buffers）。

    ⇒ 真实 LDS 取决于**设计**，不是单纯 tile 尺寸。
    本模型只能给**上界**（A+B 都过 LDS 的最坏情况），并在输出里标明；
    若调用方知道该实现 B 不走 LDS，传 `b_via_vgpr=True`（默认）只算 A。

    这也是 `model_limit` 必须存在的原因：**同一个 tile 不同设计，LDS 差 4 倍。**
    """
    if b_via_vgpr:
        # 只 A 走 LDS。**这是上界**：hbird 实测 256×256×64 只用 32KB/CTA，
        # 而本式给 (256*64*2*2)=64KB —— 因为它按 32-row 子块分步搬，
        # 单 stage 只 16KB（Phase 2 "32-row sub-quarter, 3 buffers"）。
        # lds_sub_rows 让调用方按实际子块高度收窄；默认按整块算（保守上界）。
        lds = rows * bk * dtype_bytes * stages
    else:
        lds = (rows * bk + bk * bn) * dtype_bytes * stages   # A+B 都走 LDS（上界）
    threads = warps * WAVE_SIZE
    # 累加器：每个输出元素一个 fp32 累加器，分摊到 threads 个线程上
    acc = (bm * bn) // threads if threads else 0
    # 其余开销（A/B 片段、地址计算、循环变量）。保守取值，宁可高估。
    vgpr = acc + 64
    by_vgpr = VGPR_PER_CU // (vgpr * threads) if vgpr * threads else 0
    by_lds = LDS_PER_CU // lds if lds else 0
    ctas = min(x for x in (by_vgpr, by_lds) if x > 0) if (by_vgpr or by_lds) else 0
    waves_per_cta = max(1, threads // WAVE_SIZE)
    return {"lds": lds, "vgpr": vgpr, "threads": threads,
            "ctas_per_cu": ctas, "waves_per_cta": waves_per_cta,
            "cta_per_cu_vgpr": by_vgpr, "cta_per_cu_lds": by_lds}


def _legality(m, n, k, bm, bn, bk, split_k, res):
    """硬约束。任一条不过 → illegal，并给出具体数字。"""
    errs = []
    if res["lds"] > LDS_PER_CU:
        errs.append("LDS=%d B > %d（BM=%d BN=%d BK=%d stages=%d）"
                    % (res["lds"], LDS_PER_CU, bm, bn, bk, res["stages"] if "stages" in res else 0))
    if res["vgpr"] > 255:
        errs.append("VGPR≈%d > 255（累加器 %d + 开销 64）" % (res["vgpr"], res["vgpr"] - 64))
    if k % bk != 0:
        errs.append("K=%d 不能被 BK=%d 整除（会引入 padding）" % (k, bk))
    if bk % 16 != 0:
        errs.append("BK=%d 不是 MFMA k 步长 16 的倍数" % bk)
    grid = _grid(m, n, bm, bn)
    if grid < CU_COUNT:
        errs.append("grid=%d < CU 数 %d（填不满，延迟藏不住）" % (grid, CU_COUNT))
    return errs, grid


def _grid(m, n, bm, bn):
    return ((m + bm - 1) // bm) * ((n + bn - 1) // bn)


def _predict_util(res, grid, bm, bn):
    """预测 MFU。**分段经验式，每段有依据，不是拟合出来的黑箱。**

    逻辑链（三个实测点支撑）：
      1. 驻留 CTA/CU == 0          → 根本跑不了
      2. 驻留少（<=1）且 grid 刚够  → 延迟藏不住，落在 UTIL_FLOOR ~ 0.65
      3. 驻留 >=2 且 tile 够大     → 常规已调优实现区间，UTIL_TYPICAL 附近
      4. tile 极大（>=256×256）    → 才有机会摸到 UTIL_CEILING

    **局限**：tile 之外的因素（epilogue 是否隐藏、A 读是否阻塞、barrier 摊薄）
    本式完全没建模。hbird 实测同样 256×256 的 tile，光把 C store 隐藏
    就从 522 TF 涨到 552 TF（+5.7%）。所以本式只能给量级，不能给精确值。
    """
    if res["ctas_per_cu"] <= 0:
        return 0.0, "驻留 0 CTA/CU —— 跑不起来"
    c = res["ctas_per_cu"]
    area = bm * bn
    # **split_k 的作用是补并行度**，只在 grid 不足以填满 CU 时才有意义。
    # grid 已经够（>= CU×4，能形成多波）时，split_k 只增加归约开销，不改预测。
    # 实测依据：hbird 的 M=192 档 "low S is the worst thing to do, S=6 is the fill optimum"
    # —— 说明 split 是补 fill，不是越多越好。
    grid_deficit = max(0.0, 1.0 - grid / (CU_COUNT * 4))
    # 段 2：驻留 1 且 tile 不大 → 下限档
    if c == 1 and area <= 128 * 128:
        u = UTIL_FLOOR + 0.10 * (1.0 - grid_deficit)
        why = "驻留 1 CTA/CU 且 tile ≤128×128 ⇒ 延迟藏不住（档位：下限）"
        if grid_deficit > 0:
            why += "；grid 缺口 %.0f%%，split_k 有用" % (grid_deficit * 100)
        return u, why
    # 段 3：驻留 >=2 → 常规区间
    if c >= 2 and area < 256 * 256:
        u = (UTIL_TYPICAL - 0.05) + 0.05 * min(1.0, (c - 2) / 2.0)
        return u, "驻留 %d CTA/CU ⇒ 常规已调优区间 %.0f%%~%.0f%%" % (
            c, (UTIL_TYPICAL - 0.05) * 100, UTIL_TYPICAL * 100)
    # 段 4：大 tile 才谈得上逼近 ISA floor
    if area >= 256 * 256 and c >= 1:
        u = UTIL_TYPICAL + (UTIL_CEILING - UTIL_TYPICAL) * min(1.0, (c) / 2.0)
        return u, "tile ≥256×256 且驻留 %d ⇒ 有机会摸 ISA floor 的 %.0f%%" % (
            c, UTIL_CEILING * 100)
    return UTIL_TYPICAL, "未命中任何档位，取常规中位"


def analyze(m, n, k, bm=128, bn=128, bk=64, warps=8, stages=3, split_k=1,
            dtype_bytes=2, b_via_vgpr=True, lds_sub_rows=None):
    """lds_sub_rows：A 的子块高度。给 None 按整块 bm（保守上界）；
    给实际值（如 hbird 的 32）则算出更贴近真实的 LDS。"""
    """返回资源、地板、预测。**illegal 配置给 reject_reason，不硬算。**"""
    res = _resources(bm, bn, bk, warps, stages, dtype_bytes, b_via_vgpr,
                     lds_rows=lds_sub_rows)
    res["lds_is_upper_bound"] = lds_sub_rows is None
    res["lds_note"] = ("A 按整块 %d 行算（上界）；实际实现若分块搬，LDS 会更小。"
                       "hbird 实测 256×256×64 用 32KB/CTA，本式给 64KB。"
                       % bm) if lds_sub_rows is None else ("A 按 %d 行子块算" % lds_sub_rows)
    res["stages"] = stages
    errs, grid = _legality(m, n, k, bm, bn, bk, split_k, res)
    grid *= split_k          # split_k 把 K 维切分，并行块数随之增加

    # **split_k 不改变总 FLOPs**——它只是把 K 维切给多个 CTA 并行算，
    # 每个 CTA 算 k/split_k，总量不变。第一版乘了 split_k，
    # 导致 floor 随 split_k 线性增大（k1=3941 / k2=7882 / k4=15764），那是错的。
    flops = 2 * m * n * k
    bytes_ = (m * k + k * n + m * n) * dtype_bytes
    t_compute_us = flops / (PEAK_TF_BF16 * 1e12) * 1e6
    t_memory_us = bytes_ / (PEAK_BW_GBS * 1e9) * 1e6
    t_floor_us = max(t_compute_us, t_memory_us)
    ai = flops / bytes_ if bytes_ else 0

    out = {
        "m": m, "n": n, "k": k,
        "tile": {"bm": bm, "bn": bn, "bk": bk, "warps": warps,
                 "stages": stages, "split_k": split_k},
        "resources": res, "grid": grid,
        "flops": flops, "bytes": bytes_, "ai": round(ai, 1),
        "bound": "compute" if t_compute_us >= t_memory_us else "memory",
        "t_compute_us": round(t_compute_us, 1),
        "t_memory_us": round(t_memory_us, 1),
        "t_floor_us": round(t_floor_us, 1),
        "model_limit": ("经验式，从 3 个实测点（hbird ISA floor 686.5 TF / "
                        "我们结构 658.1 / hipBLASLt 581）反推。只建模 tile 装得下与"
                        "驻留够不够；epilogue 隐藏、A 读阻塞、barrier 摊薄全没建模。"
                        "只能给量级，不能当预测即事实。"),
    }
    if errs:
        out["illegal"] = True
        out["reject_reason"] = errs
        out["confidence"] = "none"
        return out

    util, why = _predict_util(res, grid, bm, bn)
    # **grid 缺口是独立于 tile 的约束**：块数填不满 CU 时，再多驻留也没用。
    # hbird 实测支持这条：M=192 档 "low S is the worst thing to do"，
    # 因为小 segment 的 grid 填不满。这里统一乘一个折扣，
    # 而不是在某个档位分支里局部处理（第一版就是那样，导致只有 c==1 时生效）。
    grid_need = CU_COUNT * 4          # 4 波才谈得上充分并行
    if grid < grid_need:
        deficit = 1.0 - grid / grid_need
        util = util * (1.0 - 0.5 * deficit)   # 最多打对折
        why += "；grid=%d < %d，并行度缺口 %.0f%% ⇒ util ×%.2f" % (
            grid, grid_need, deficit * 100, 1.0 - 0.5 * deficit)
    out["predicted_mfu"] = round(util, 3)
    out["predicted_us"] = round(t_floor_us / util, 1) if util else None
    out["predicted_reason"] = why
    # confidence：有大 tile 实测支撑的高一点，其余都是 low
    big = bm * bn >= 256 * 256
    out["confidence"] = "medium" if big else "low"
    out["illegal"] = False
    return out
