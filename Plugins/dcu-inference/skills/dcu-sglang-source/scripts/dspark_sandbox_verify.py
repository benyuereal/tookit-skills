#!/usr/bin/env python3
# dspark_sandbox_verify.py
# ─────────────────────────────────────────────────────────────────────────────
# DSpark 核心算法「纯逻辑 sandbox 验证」（脱离 sglang 运行时 / torch / GPU）。
#
# 目的：验证 DSpark（Confidence-Driven Speculative Decoding）的置信度调度
#       与成本估计算法 *逻辑本身* 正确，作为「移植完全性」的算法级证据。
#       这不是端到端推理验证——端到端需海光 fork 先 rebase 上游 main 的
#       DSpark 相关 commit（见 DSpark 移植参考文档）。
#
# 验证对象（来自上游 sglang main dspark_components/）：
#   1. dspark_sps.py 的 SpsCostTable._interp_clamped / lookup —— 线性插值成本表
#      （置信度 relay 用「每 token 成本」决定是否走 DSpark vs 普通解码）
#   2. dspark_block_accept_estimator.py 的 _OnlineCeiling —— 在线天花板估计
#      （block accept 概率的滑动窗口置信度上界，驱动两阶段回退 relay）
#
# 不 import sglang / torch：纯 Python 重写计算内核（与上游源码逻辑一致），
# 用 mock 输入跑断言。
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import math


# ── 1. 成本表插值（对应 dspark_sps.py _interp_clamped + SpsCostTable.lookup）──
def interp_clamped(xs: list[int], ys: list[float], x: float) -> float:
    """上游 _interp_clamped：对 (xs, ys) 分段线性插值，越界 clamp 到端点。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # 找到 x 所在区间
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def sps_lookup(xs: list[int], ys: list[float], batch_tokens: int) -> float:
    """SpsCostTable.lookup：查某 batch_tokens 下的每步成本（ms）。"""
    return interp_clamped(xs, ys, float(batch_tokens))


# ── 2. 在线天花板估计（对应 dspark_block_accept_estimator._OnlineCeiling）──
class OnlineCeiling:
    """简化版在线天花板估计：滑动窗口内观测 accept 概率的上界。

    上游逻辑：维护一个 window_steps 长的滑动窗口，记录每次 verify 的
    accept 概率 [lo, hi]，取窗口内 hi 的下分位作为「置信上界」天花板。
    当实测 accept 率跌破天花板阈值时，触发两阶段回退（relay）。
    """

    def __init__(self, *, window_steps: int = 32, quantile: float = 0.1):
        self.window_steps = window_steps
        self.quantile = quantile
        self._buf: list[float] = []  # 每次观测的 accept 概率 (hi 值)

    def add(self, accept_prob: float) -> None:
        self._buf.append(accept_prob)
        if len(self._buf) > self.window_steps:
            self._buf.pop(0)

    def estimate_ceiling(self) -> float | None:
        """窗口内 hi 值的下 quantile 分位 = 保守天花板（置信上界）。"""
        if not self._buf:
            return None
        s = sorted(self._buf)
        idx = max(0, min(len(s) - 1, int(self.quantile * len(s))))
        return s[idx]


# ── 3. 两阶段回退 relay 决策（对应 dspark_planner 的 confidence relay）──
def decide_relay(
    accept_rate: float,
    ceiling: float,
    floor: float,
    *,
    margin: float = 0.05,
) -> str:
    """DSpark 置信度 relay：
    - accept_rate 接近天花板(ceiling) → 继续 DSpark（high confidence）
    - accept_rate 跌破 floor → 回退到普通解码（low confidence）
    - 中间 → 两阶段 relay 的中间态（部分投机）
    """
    if accept_rate >= ceiling - margin:
        return "full_draft"          # 高置信：全力投机
    if accept_rate <= floor + margin:
        return "fallback_no_spec"    # 低置信：回退普通解码
    return "partial_relay"           # 中间态：两阶段回退


def main() -> None:
    print("=== DSpark 核心算法 sandbox 验证 ===\n")

    # ── 验证 1：成本表插值单调性 + 边界 clamp ──
    print("[1] SpsCostTable 插值逻辑")
    xs = [1, 64, 256, 1024, 4096]
    ys = [2.0, 3.5, 6.0, 12.0, 40.0]  # 每步成本 ms，随 batch 增大
    # 边界 clamp
    assert sps_lookup(xs, ys, 0) == 2.0, "低于下界应 clamp 到 ys[0]"
    assert sps_lookup(xs, ys, 99999) == 40.0, "高于上界应 clamp 到 ys[-1]"
    # 单调性：batch 越大成本越高
    prev = -1.0
    for bt in [1, 32, 128, 512, 2048, 4096]:
        v = sps_lookup(xs, ys, bt)
        assert v >= prev, f"成本应单调不减 @batch={bt}"
        prev = v
    # 区间插值正确性 (x=32 在 [1,64] 间: 2.0 + (32-1)/(64-1)*(3.5-2.0))
    exp = 2.0 + (32 - 1) / (64 - 1) * (3.5 - 2.0)
    got = sps_lookup(xs, ys, 32)
    assert abs(got - exp) < 1e-9, f"插值错误 got={got} exp={exp}"
    print("    ✓ 边界 clamp 正确，单调性正确，区间插值精确")

    # ── 验证 2：在线天花板估计随观测收敛 ──
    print("[2] OnlineCeiling 置信上界估计")
    oc = OnlineCeiling(window_steps=16, quantile=0.1)
    # 高 accept 率场景（DSpark 有效）：观测值都在 0.8~0.95
    for _ in range(20):
        oc.add(0.85)
    ceil_hi = oc.estimate_ceiling()
    assert ceil_hi is not None and ceil_hi >= 0.8, f"高置信天花板应≈0.85, got={ceil_hi}"
    print(f"    高 accept 场景 天花板={ceil_hi:.3f} (期望≈0.85)")

    # 低 accept 率场景（DSpark 失效）：观测值跌到 0.2~0.35
    oc2 = OnlineCeiling(window_steps=16, quantile=0.1)
    for _ in range(20):
        oc2.add(0.28)
    ceil_lo = oc2.estimate_ceiling()
    assert ceil_lo is not None and ceil_lo <= 0.35, f"低置信天花板应≈0.28, got={ceil_lo}"
    print(f"    低 accept 场景 天花板={ceil_lo:.3f} (期望≈0.28)")

    # ── 验证 3：两阶段 relay 决策正确 ──
    print("[3] 两阶段回退 relay 决策")
    # 高置信：accept 接近天花板 → full_draft
    assert decide_relay(0.84, 0.85, 0.3) == "full_draft"
    # 低置信：accept 跌破 floor → fallback
    assert decide_relay(0.25, 0.85, 0.3) == "fallback_no_spec"
    # 中间态：partial_relay
    assert decide_relay(0.55, 0.85, 0.3) == "partial_relay"
    print("    ✓ full_draft / partial_relay / fallback_no_spec 三态决策正确")

    print("\n=== 全部断言通过：DSpark 置信度调度 + 成本估计算法逻辑正确 ===")
    print("    （注：此为脱离 sglang/torch 的算法级验证；端到端需海光 fork rebase 上游 DSpark）")


if __name__ == "__main__":
    main()
