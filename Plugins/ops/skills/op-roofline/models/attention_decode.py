"""attention-decode 的 tile 级模型：给定 decode 形状，算流量、地板与**并行度预测**。

为什么要这个模型
----------------
`templates.py` 的 `attention-decode` 只算流量（bytes / 名义 flops）——
它告诉 agent「这个 decode 该搬多少字节」，但**解释不了达成率差一个数量级**：

    paged-attention-decode（flash_attn，D=128）实测 of_peak 只有 0.03
    mla-absorb            （flash_mla，D=576）实测 of_peak 到 0.35

两者的流量口径几乎一样（都是「1 query 读全 KV」）。差的是**并行度**：
decode 每步只有一个 query，能并行的粒度只有
`batch × heads × ceil(kv_len/block_size) × num_split` 个 KV 块。
块数填不满 CU、或每块太小扛不住固定开销时，DRAM 再空也跑不满。

这个模型因此做三件事：
  1. 流量与内存地板（与 templates 同口径，可交叉校验）
  2. grid = 并行块数，明确它离「饱和」还差多远
  3. 用**饱和经验式**给出 predicted_util

预测公式从哪来（不许拍脑袋）
----------------------------
两组同机实测（BW1100）反推：

  ① op-optimize/cases/006-mla-decode-absorb/results/mla-decode.json
     mla-absorb，H=32 D=576 block=64
       sweep_batch: B=1 → 0.073, B=8 → 0.259, B=32 → 0.353
       sweep_kv   : KV=2048 → 0.128, 8192 → 0.259, 16384 → 0.312, 32768 → 0.345
  ② op-optimize/cases/008-paged-decode-ceiling/results/paged-decode.json
     paged (flash_attn)，H=32 D=128 block=64，KV=8192
       sweep_batch: B=1 → 0.006, B=8 → 0.021, B=32 → 0.027, B=64 → 0.029

把 11 个点摊到 (grid, util) 平面上，规律是**干净的**：

    util = U_ceiling(head_dim) × grid_waves / (grid_waves + W0)

  · grid_waves W = grid / CU            —— 并行块数折算成「多少波」
  · W0 = 288                            —— 半饱和波数（W=288 时吃到 U_ceiling 的一半）
  · U_ceiling = min(0.85, 0.031·(head_dim/128)^1.7)

为什么是这两个量、而不是「CU×4 的缺口」分档：
  在 anchor 里连最小的 B=1 也有 grid=4096（=64 波），早已超过 CU×4=256。
  所以「填满 64 CU × 4 波」是**必要不充分**——真正卡住 35% 的是
  「每块 KV 太小、固定开销摊不开」，需要上千波才饱和。W0=288 就是这个尺度。
  本式仍保留 grid < CU（< 64）时的 `low_parallelism` 硬提示（见输出），
  因为那是另一种坏法：CU 都填不满。

为什么 head_dim 进了 U_ceiling：
  D=128 档（paged）饱和后只到 0.029，D=576 档（mla）到 0.35，差 ~12×。
  head_dim 决定单块 KV 的读粒度；但这里它同时**混入了两个 kernel 的实现质量差**
  （paged flash-attn 是已知坏档）。所以 head_dim 是指标不是机制——
  这是本模型最大的不确定项，写进 model_limit。

**这不是物理定律。** 它是从 11 个点反推的经验式。输出必带 `model_limit`
与 `confidence`，别拿它当「预测即事实」。
"""

import math

MODEL_SPEC = {
    "fn": "analyze",
    "args": ["kv_len", "batch", "heads", "head_dim",
             "block_size", "num_split", "warps"],
    "peak": "bf16",
    "bound": "memory",
}

# ---- 机器常数（与 constants.yaml / 其它模型同源；取不到用这里的回退值）
CU_COUNT = 64
PEAK_BW_GBS = 1730.0       # bw-ladder 2_rw_2to1（decode 读主导但含 out 写）
PEAK_TF_BF16 = 279.0       # 仅用于算出算力地板作对照；decode 永远是访存受限

# ---- 经验式锚点（见文件头「预测公式从哪来」）
W_HALF = 288.0             # 半饱和波数
U_REF = 0.031              # head_dim=128 时的饱和上限
D_REF = 128.0              # U_REF 对应的 head_dim
D_EXP = 1.7                # 单块读粒度对饱和上限的指数
U_CAP = 0.85               # 饱和上限的硬天花板（再高就不是 decode 能到的区间）

# 反推来源，写进输出，让调用方能复查
_ANCHORS = [
    {"src": "cases/006-mla-decode-absorb", "kernel": "mla-absorb",
     "heads": 32, "head_dim": 576, "block_size": 64, "kv_len": 8192,
     "points": {"batch=1": 0.073, "batch=8": 0.259, "batch=32": 0.353}},
    {"src": "cases/006-mla-decode-absorb", "kernel": "mla-absorb",
     "heads": 32, "head_dim": 576, "block_size": 64, "batch": 8,
     "points": {"kv=2048": 0.128, "kv=8192": 0.259,
                "kv=16384": 0.312, "kv=32768": 0.345}},
    {"src": "cases/008-paged-decode-ceiling", "kernel": "paged (flash_attn)",
     "heads": 32, "head_dim": 128, "block_size": 64, "kv_len": 8192,
     "points": {"batch=1": 0.006, "batch=8": 0.021,
                "batch=32": 0.027, "batch=64": 0.029}},
]


def _grid(kv_len, batch, heads, block_size, num_split):
    """并行块数：batch × heads × KV 块数 × split。"""
    n_blocks = (kv_len + block_size - 1) // block_size
    return batch * heads * n_blocks * num_split


def _legality(kv_len, batch, heads, head_dim, block_size, num_split):
    """硬约束。任一条不过 → illegal，并给出具体数字。"""
    errs = []
    if block_size <= 0:
        errs.append("block_size=%d 必须为正（分块数会除零）" % block_size)
    if kv_len < block_size:
        errs.append("kv_len=%d < block_size=%d（一个 KV 块都装不满，grid 无意义）"
                    % (kv_len, block_size))
    if head_dim % 8 != 0:
        errs.append("head_dim=%d 不是 8 的倍数（向量化读会退化为标量）" % head_dim)
    if batch <= 0 or heads <= 0:
        errs.append("batch=%d / heads=%d 必须为正" % (batch, heads))
    if num_split <= 0:
        errs.append("num_split=%d 必须 >= 1" % num_split)
    if kv_len <= 0:
        errs.append("kv_len=%d 必须为正" % kv_len)
    return errs


def _u_ceiling(head_dim):
    """单块读粒度决定的饱和上限。D=128 → 0.031；D=576 → ~0.40。"""
    return min(U_CAP, U_REF * (float(head_dim) / D_REF) ** D_EXP)


def _predict_util(grid, head_dim):
    """预测 util（of_peak）。**饱和经验式，依据见文件头。**

    分两段相乘：
      1. 饱和上限 U_ceiling(head_dim) —— 单块读粒度能扛到的最高达成率
      2. 并行填充 fill(W) = W/(W+W0)  —— 当前块数离饱和有多远
    两者独立：块再大，grid 不够也吃不满（B=1 的教训）；
            grid 再多，D=128 的坏 kernel 也上不去（paged 的教训）。
    """
    waves = grid / float(CU_COUNT)
    fill = waves / (waves + W_HALF) if waves > 0 else 0.0
    u = _u_ceiling(head_dim) * fill
    why = ("并行波数 W=grid/CU=%.1f ⇒ 填充率 W/(W+%.0f)=%.3f；"
           "单块读粒度上限 U_ceil(head_dim=%d)=%.3f"
           % (waves, W_HALF, fill, head_dim, _u_ceiling(head_dim)))
    return u, why, waves, fill


def analyze(kv_len, batch, heads, head_dim, block_size=64,
            num_split=1, warps=4, dtype_bytes=2):
    """返回流量、地板、并行度与预测。**illegal 配置给 reject_reason，不硬算。**

    `warps` 为接口对齐保留（tile 配置的一部分），它只影响占用率，
    本模型不建模占用率——不假装它对预测有贡献。
    """
    # 合法性必须先于 grid：block_size=0 会让 _grid 除零，而不是「报错」。
    errs = _legality(kv_len, batch, heads, head_dim, block_size, num_split)
    grid = _grid(kv_len, batch, heads, block_size, num_split) if block_size > 0 else None

    # ---- 流量（与 templates.py 的 attention-decode 同口径：2·B·S·h·d·bytes）
    bytes_ = 2 * batch * heads * kv_len * head_dim * dtype_bytes
    flops = 4 * batch * heads * kv_len * head_dim
    t_memory_us = bytes_ / (PEAK_BW_GBS * 1e9) * 1e6
    t_compute_us = flops / (PEAK_TF_BF16 * 1e12) * 1e6
    t_floor_us = max(t_memory_us, t_compute_us)
    ai = flops / bytes_ if bytes_ else 0.0

    out = {
        "kv_len": kv_len, "batch": batch, "heads": heads, "head_dim": head_dim,
        "design": {"block_size": block_size, "num_split": num_split,
                   "warps": warps, "dtype_bytes": dtype_bytes},
        "grid": grid,
        "bytes": bytes_, "flops": flops, "ai": round(ai, 2),
        "bound": "compute" if t_compute_us > t_memory_us else "memory",
        "t_memory_us": round(t_memory_us, 1),
        "t_compute_us": round(t_compute_us, 1),
        "t_floor_us": round(t_floor_us, 1),
        # 从哪几个点反推的——直接写进输出，调用方不必翻源码
        "calibration": {
            "formula": ("util = U_ceiling(head_dim) * W/(W+%d); "
                        "W = grid/%d; U_ceiling = min(%.2f, %.4f*(head_dim/%d)^%.1f)"
                        % (int(W_HALF), CU_COUNT, U_CAP, U_REF, int(D_REF), D_EXP)),
            "anchors": _ANCHORS,
        },
        "model_limit": (
            "经验式，从 11 个同机实测点反推（case 006 mla-absorb ×7 + "
            "case 008 paged ×4）。只建模「并行波数」与「单块读粒度」；"
            "kernel 调度策略、paged 间接寻址开销、L2 命中、block_table 读全没建模。"
            "head_dim 是代理变量（混入了 paged vs mla 的实现质量差），不是机制。"
            "warps 不参与预测。只能给量级，不能当预测即事实。"),
    }

    if errs:
        out["illegal"] = True
        out["reject_reason"] = errs
        out["confidence"] = "none"
        return out

    assert grid is not None  # errs 为空 ⇒ block_size > 0 ⇒ grid 已算

    util, why, waves, fill = _predict_util(grid, head_dim)
    out["predicted_util"] = round(util, 3)
    out["predicted_us"] = round(t_floor_us / util, 1) if util else None
    out["predicted_reason"] = why

    # ---- 并行度硬提示：CU 都填不满是另一种坏法，与饱和度分开报
    if grid < CU_COUNT:
        out["low_parallelism"] = True
        out["low_parallelism_note"] = (
            "grid=%d < CU 数 %d —— 有 CU 完全空闲；调大 batch/heads 或 num_split，"
            "否则带宽永远吃不满。" % (grid, CU_COUNT))
    else:
        out["low_parallelism"] = False
        if fill < 0.5:
            out["low_parallelism_note"] = (
                "grid=%d 已过 CU 数，但 W=%.1f < 半饱和波数 %.0f —— "
                "并行度仍是主要瓶颈，num_split 有效。" % (grid, waves, W_HALF))

    # confidence：落在 anchor 的 heads/head_dim/block 组合里给 medium，其余 low
    covered = (heads == 32 and head_dim in (128, 576) and block_size == 64)
    out["confidence"] = "medium" if covered else "low"
    out["illegal"] = False
    return out
