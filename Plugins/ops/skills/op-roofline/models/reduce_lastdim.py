"""reduce-lastdim 的 tile 级模型：行内归约实现决定同样流量能跑多快。

为什么要这个模型
----------------
`templates.py` 的 `reduce-lastdim` 是**流量级**——只算 bytes（2读2写 = 268.4 MB），
不碰「行内归约怎么做」。于是 agent 知道「这个 rmsnorm 该搬 268.4 MB」，
但**不知道换成哪种归约实现、换多少 vec_width 才能跑满**。

本模型补归约那一层：block_rows / warps / vec_width / reduce_impl
→ LDS / 归约深度 / 预测 util。

预测 util 的公式从哪来（不许拍脑袋）
-----------------------------------
三个实测点，同一形状（M=8192, H=4096, bf16, 2读2写 = 268435456 B），
同一台机器、同一轮 interleaved 测量（op-rmsnorm-fused/results/impl-ranking.json
+ ledger.md 2026-11-24 复核）：

  ① lightop（厂商 HIP）  197.0 µs  达成率 0.811
  ② tilelang             221.0 µs  达成率 0.723
  ③ triton               242.0 µs  达成率 0.660

达成率分母 = 268435456 B / 1680 GB/s = 159.78 µs（1:1 档 achievable BW）。

三个点给出的规律：
  - **流量完全相同**，差 23% 全在归约实现：
      HIP 能手写 `__shfl_down_sync` / `ds_bpermute`，归约全在寄存器（LDS=0）；
      tilelang 能部分控制，编译器仍会插 LDS；
      triton 无法控制归约方式，编译器决定，通常走 LDS 树归约。
  - **H 越大，归约深度越大**，实现之间的差距越明显；H 小则被固定开销摊薄。
  - vec_width 决定每次访存搬多少，2→4 能减少访存指令数、提升有效带宽。

所以预测 util 不能按 bytes 插值——**bytes 三者一样**。它按 reduce_impl 分档，
每档一个实测锚点，再用 H / vec_width / warps / block_rows 做小幅修正
（修正在锚点形状处严格为 1.0，保证回测等于实测）。

**这不是物理定律。** 它是从 3 个实测点 + 归约机制推的经验式，只在一个形状上验证过。
输出必带 `model_limit`，别拿它当「预测即事实」。
"""

MODEL_SPEC = {
    "fn": "analyze",
    "args": ["M", "H", "block_rows", "warps", "vec_width", "reduce_impl"],
    "peak": "bf16",
    "bound": "memory",
}

# ---- 机器常数（与 constants.yaml / bw-ladder 1:1 档一致；取不到就用回退值并标注）
PEAK_BW_GBS = 1680.0            # bw-ladder 1_rw_1to1（rmsnorm 是 1读1写 / 2读2写）
LDS_PER_CU = 65536              # B，与 matmul.py 一致
WAVE_SIZE = 64
MAX_VEC_WIDTH = 4               # dwordx4 已是单指令上限，再大不再加分

# ---- 三个实测锚点（同形状 M=8192 H=4096 bf16，见文件头）
# 键是规范化后的 reduce_impl；值 = (达成率, 归约是否走 LDS, 依据)
REDUCE_ANCHOR = {
    "shfl": (0.811, False,
             "lightop 厂商 HIP：__shfl_down_sync/ds_bpermute，归约全在寄存器"),
    "tilelang": (0.723, True,
                 "tilelang：部分可控，编译器仍插 LDS 做跨 lane 交换"),
    "lds": (0.660, True,
            "triton：归约方式由编译器决定，通常走 LDS 树归约"),
}

# reduce_impl 的别名 → 规范键
_REDUCE_ALIAS = {
    "shfl": "shfl", "shuffle": "shfl", "hip": "shfl", "lightop": "shfl",
    "tilelang": "tilelang", "tile-lang": "tilelang",
    "lds": "lds", "triton": "lds", "lds-tree": "lds",
}

# 锚点形状（修正因子在此处必须严格为 1.0，回测才等于实测）
ANCHOR_H = 4096
ANCHOR_WARPS = 4
ANCHOR_VEC = 2

# 修正因子的指数（都很小——锚点已经吸收了主要差异，修正只做外推）
H_EXP = 0.03
WARPS_EXP = 0.08
VEC_EXP = 0.10
BLOCK_ROWS_EXP = 0.03
UTIL_CEILING = 0.95             # 经验式给不出 1.0


def _canon_impl(reduce_impl):
    return _REDUCE_ALIAS.get(str(reduce_impl).strip().lower())


def _reduce_depth(H, warps):
    """归约深度 = 线程内串行步 + 跨 lane 步 + 跨 warp 步。

    ceil(log2(H / threads))  行内元素分到每个线程后，线程内还要串行累加几步
    ceil(log2(warps))        跨 warp（每 warp 64 lane）的树深
    '跨 lane 的 log2(64)=6' 被吸收进第一项的分母，不单独列。
    """
    threads = warps * WAVE_SIZE
    import math
    inner = max(0, int(math.ceil(math.log2(max(1, H / threads))))) if H > threads else 0
    across = max(0, int(math.ceil(math.log2(max(1, warps)))))
    return inner + across


def _resources(H, block_rows, warps, vec_width, reduce_impl, dtype_bytes):
    """资源用量。**归约方式的差别就体现在 LDS 上**：
      shfl  → LDS = 0（全在寄存器里换）
      lds   → 每行 H 个 fp32 中间量落到 LDS：block_rows × H × 4 B
               （fp32 是因为归约/平方和必须 32 位精度，不随 dtype_bytes 变）
    """
    canon = _canon_impl(reduce_impl)
    # 未知实现按最保守的 lds 估资源（_legality 会先报 illegal，这里只为不让
    # 资源计算抛异常；预测分支不会走到）
    anchor = REDUCE_ANCHOR[canon] if canon else REDUCE_ANCHOR["lds"]
    goes_lds = anchor[1]
    lds = block_rows * H * 4 if goes_lds else 0
    threads = warps * WAVE_SIZE
    # 每个线程要持有的行内元素数（vec 化后的向量个数），用于说明占用
    vecs_per_thread = H // (threads * max(1, vec_width))
    return {
        "lds": lds,
        "reduce_impl": canon,
        "uses_lds": goes_lds,
        "threads": threads,
        "vecs_per_thread": vecs_per_thread,
        "reduce_depth": _reduce_depth(H, warps),
        "lds_note": ("归约全在寄存器（shfl/ds_bpermute），不占 LDS" if not goes_lds
                     else "归约经 LDS：block_rows(=%d) × H(=%d) × 4 B" % (block_rows, H)),
    }


def _legality(M, H, block_rows, warps, vec_width, outputs, reduce_impl, res):
    """硬约束。任一条不过 → illegal，并给出具体数字。"""
    errs = []
    if M <= 0:
        errs.append("M=%r 必须 > 0" % (M,))
    if H <= 0:
        errs.append("H=%r 必须 > 0" % (H,))
    if block_rows <= 0:
        errs.append("block_rows=%r 必须 >= 1（0 行没法活跃）" % (block_rows,))
    if warps <= 0:
        errs.append("warps=%r 必须 >= 1" % (warps,))
    if vec_width <= 0:
        errs.append("vec_width=%r 必须 >= 1" % (vec_width,))
    if H > 0 and (H & (H - 1)) != 0:
        errs.append("H=%d 不是 2 的幂（行内归约树要求 2 的幂，否则要 padding）" % H)
    if H > 0 and vec_width > 0 and H % vec_width != 0:
        errs.append("H=%d 不能被 vec_width=%d 整除（向量化 load 会跨界）" % (H, vec_width))
    if outputs not in (2, 4):
        errs.append("outputs=%r 只支持 2（1读1写）或 4（2读2写，含 residual_out）"
                    % (outputs,))
    if _canon_impl(reduce_impl) is None:
        errs.append("reduce_impl=%r 未知（可选 %s）"
                    % (reduce_impl, sorted(REDUCE_ANCHOR)))
    elif res["lds"] > LDS_PER_CU:
        errs.append("LDS=%d B > %d（block_rows=%d H=%d）"
                    % (res["lds"], LDS_PER_CU, block_rows, H))
    return errs


def _predict_util(res, H, warps, vec_width, block_rows):
    """预测达成率。**按 reduce_impl 分档，每档一个实测锚点 + 小幅修正。**

    修正因子在锚点形状（H=4096, warps=4, vec=2, block_rows=1）严格为 1.0，
    所以回测 M=8192 H=4096 时输出就是 0.811 / 0.723 / 0.660。

    修正的三条机制（各自有依据，不是拟合）：
      1. H 越大，行内归约占 kernel 时间的比例越大 → 实现差距放大；
         小 H 被固定开销摊薄，有效带宽略低。
      2. warps 越多，跨 warp 树浅、并发高，但同步开销也大 → 小幅变化。
      3. vec_width 越大，访存指令越少、有效带宽越高，到 4（dwordx4）封顶。
      4. block_rows 只对走 LDS 的实现加惩罚（LDS 占用与行数成正比）；
         shfl 实现每行独立，block_rows 不改变归约成本。
    """
    base = REDUCE_ANCHOR[res["reduce_impl"]][0]

    # 1. H：锚点处为 1
    h_eff = (float(H) / ANCHOR_H) ** H_EXP
    h_eff = min(1.06, max(0.82, h_eff))

    # 2. warps
    w_eff = (float(warps) / ANCHOR_WARPS) ** WARPS_EXP
    w_eff = min(1.08, max(0.90, w_eff))

    # 3. vec_width：2 → 4 上升，超过 4 封顶
    v = min(vec_width, MAX_VEC_WIDTH)
    v_eff = (float(v) / ANCHOR_VEC) ** VEC_EXP
    v_eff = min(1.10, max(0.90, v_eff))

    # 4. block_rows：只惩罚走 LDS 的实现
    if res["uses_lds"]:
        br_eff = (1.0 / max(1, block_rows)) ** BLOCK_ROWS_EXP
        br_eff = max(0.90, br_eff)
    else:
        br_eff = 1.0

    u = base * h_eff * w_eff * v_eff * br_eff
    u = min(UTIL_CEILING, u)
    why = ("%s ⇒ 锚点达成率 %.3f；H 修正 ×%.3f、warps ×%.3f、vec ×%.3f、"
           "block_rows ×%.3f" % (REDUCE_ANCHOR[res["reduce_impl"]][2], base,
                                 h_eff, w_eff, v_eff, br_eff))
    return u, why


def analyze(M, H, block_rows=1, warps=4, vec_width=2, reduce_impl="shfl",
            dtype_bytes=2, outputs=2, peak_bw_gbs=PEAK_BW_GBS):
    """返回资源、地板、预测。**illegal 配置给 reject_reason，不硬算。**

    outputs: 2 = 1读1写（默认，rmsnorm/layernorm）；4 = 2读2写（含 residual_out）。
    """
    res = _resources(H, block_rows, warps, vec_width, reduce_impl, dtype_bytes)
    errs = _legality(M, H, block_rows, warps, vec_width, outputs, reduce_impl, res)

    # 流量：2 读 2 写时 outputs=4；否则 1 读 1 写
    bytes_ = outputs * M * H * dtype_bytes
    t_memory_us = bytes_ / (peak_bw_gbs * 1e9) * 1e6
    grid = (M + block_rows - 1) // block_rows if block_rows > 0 else 0

    out = {
        "M": M, "H": H,
        "tile": {"block_rows": block_rows, "warps": warps,
                 "vec_width": vec_width, "reduce_impl": res["reduce_impl"]},
        "resources": res,
        "grid": grid,
        "bytes": bytes_,
        "flops": 0,
        "bound": "memory",
        "outputs": outputs,
        "t_memory_us": round(t_memory_us, 1),
        "t_floor_us": round(t_memory_us, 1),
        "model_limit": ("经验式，从 3 个同形状实测点（lightop 0.811 / tilelang 0.723 / "
                        "triton 0.660，M=8192 H=4096 bf16）反推。只建模归约方式、"
                        "归约深度、Vec 宽度与 LDS 占用；cache 行为、启动开销、"
                        "num_stages、寄存器压力全没建模。只在一个形状上验证过，"
                        "只能给量级，不能当预测即事实。"),
    }

    if errs:
        out["illegal"] = True
        out["reject_reason"] = errs
        out["confidence"] = "none"
        return out

    util, why = _predict_util(res, H, warps, vec_width, block_rows)
    out["predicted_util"] = round(util, 3)
    out["predicted_us"] = round(t_memory_us / util, 1) if util else None
    out["predicted_reason"] = why
    out["illegal"] = False

    # confidence：锚点形状附近给 high，区间内 medium，外推 low
    if H == ANCHOR_H and vec_width in (2, 4) and warps in (2, 4, 8):
        out["confidence"] = "high"
    elif 1024 <= H <= 16384 and vec_width <= 4:
        out["confidence"] = "medium"
    else:
        out["confidence"] = "low"
    return out
