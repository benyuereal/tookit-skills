#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_mode —— 测量模式的单一真源。**它决定一个结论能不能被 claim。**

为什么这个文件存在
------------------
R0 跑出的第一条记录说 `fused-3pass-triton` 比 `lightop-fused` **快 17.4%**。
同一形状、同一对实现，`ledger.md`（event 隔离）记的是**慢 6.7%**。**符号相反。**

根因不是测量误差，是结构缺口：`bench/ab.py` 用的是 `back_to_back`
（无 per-iter sync），自己写着「偏乐观约 23%」，而 `loop.py` 写记录时
**这个字段一个字都没进 record**（实测 `grep -c measure_mode loop.py` = 0）。

更严重的是：那个「23%」本身也是错的。实测两个实现的乐观度分别是
1.506 / 1.166（`k₁/k₂ = 1.292`）——**偏差逐实现变化，不存在任何可移植的修正因子**。
所以只有一条路：同模式才能比，且 back_to_back 永远不可 claim。

与 judge() 的关系（关键）
------------------------
`judge()` 只能判**噪声**（MAD），判不了**偏差**（bias）。
实测证据：
    R0 背靠背   : Δ/(3×MAD) = 79 倍   ← 看似极度显著，符号是反的
    ledger 隔离 : Δ/(3×MAD) = 1.01 倍 ← 只勉强过线，符号是对的
符号错的口径「显著性」高 78 倍，因为 back_to_back 的 MAD 极小（它测理想流水），
而偏差是系统性的、被极小的 MAD 完全掩盖。

⇒ **measure_mode 必须是独立前置门，不能折进显著性检验。**

用法
----
    from measure_mode import MODES, claimable, same_mode
    python3 -I scripts/measure_mode.py --selftest
"""
import sys

# 三态。**只有 claimable=True 的模式能产出「加速」结论。**
MODES = {
    "event_isolated": {
        "claimable": True,
        "note": "event 隔离 + 预分配 + 多次中位",
        "confidence": "high",
    },
    "end_to_end": {
        "claimable": True,
        "note": "每次 empty_like，含分配；反映真实调用成本",
        "confidence": "high",
    },
    "back_to_back": {
        "claimable": False,
        "note": ("无 per-iter sync，kernel 间可重叠。"
                 "可用于**筛方向**，不得进入「已验证」。"
                 "偏差逐实现变化（实测 k₁/k₂=1.29），无全局修正因子"),
        "confidence": "none",
    },
    "unknown": {
        "claimable": False,
        "note": "测量模式未声明 —— 缺了这个字段，结论的符号都不可信",
        "confidence": "none",
    },
}


def claimable(mode):
    """这个模式能不能产出可 claim 的「加速」结论。"""
    return bool((MODES.get(str(mode)) or {}).get("claimable"))


def same_mode(base_mode, cand_mode):
    """base 与 cand 必须同模式，否则比值无意义。"""
    if not base_mode or not cand_mode:
        return False, "有一侧没声明测量模式（base=%r cand=%r）" % (base_mode, cand_mode)
    if str(base_mode) != str(cand_mode):
        return False, ("base(%s) 与 cand(%s) 测量模式不同 —— 跨模式比值"
                       "不仅会翻转符号，而且乐观度逐实现变化，不存在可移植修正因子"
                       % (base_mode, cand_mode))
    return True, "同模式：%s" % base_mode


def downgrade_decision(decision, mode):
    """把一个 ACCEPTED_* 结论按测量模式降级。

    返回 (decision, why_suffix)。不改变 REJECTED_* ——
    拒绝结论不需要「可信」这个前提（拒绝本来就是保守的）。
    """
    if not str(decision).startswith("ACCEPTED"):
        return decision, ""
    if claimable(mode):
        return decision, ""
    return ("INCONCLUSIVE",
            "；测量模式=%s 不可用于主张（可用于筛方向），"
            "ACCEPTED 降级为 INCONCLUSIVE" % mode)


def problems(record):
    """一条 record 的 measure_mode 问题列表。空列表 = 没问题。"""
    probs = []
    m = record.get("measure_mode")
    if not m:
        probs.append("缺 measure_mode —— 缺这个字段，结论的符号都不可信")
        return probs
    if str(m) not in MODES:
        probs.append("measure_mode=%r 非法（只允许 %s）" % (m, sorted(MODES)))
        return probs
    if not claimable(m) and str(record.get("decision", "")).startswith("ACCEPTED"):
        probs.append("decision=%s 但 measure_mode=%s 不可 claim —— 必须降级"
                     % (record.get("decision"), m))
    return probs


# ------------------------------------------------------------------ selftest

def selftest():
    assert claimable("event_isolated") is True
    assert claimable("end_to_end") is True
    assert claimable("back_to_back") is False
    print("[ok] 只有 event_isolated / end_to_end 可 claim")

    ok, msg = same_mode("event_isolated", "event_isolated")
    assert ok, msg
    ok, msg = same_mode("event_isolated", "back_to_back")
    assert not ok and "跨模式" in msg, msg
    print("[ok] 跨模式比较被拒：%s" % msg[:40])

    ok, msg = same_mode("event_isolated", None)
    assert not ok and "没声明" in msg
    print("[ok] 有一侧没声明 → 拒")

    d, why = downgrade_decision("ACCEPTED_BITEXACT", "back_to_back")
    assert d == "INCONCLUSIVE", d
    print("[ok] back_to_back 的 ACCEPTED 降级为 INCONCLUSIVE")

    d, why = downgrade_decision("ACCEPTED_BITEXACT", "event_isolated")
    assert d == "ACCEPTED_BITEXACT", d
    print("[ok] event_isolated 的 ACCEPTED 保持不变")

    d, _ = downgrade_decision("REJECTED_NO_GAIN", "back_to_back")
    assert d == "REJECTED_NO_GAIN", d
    print("[ok] REJECTED 不降级（拒绝本来就是保守的）")

    p = problems({})
    assert any("缺 measure_mode" in x for x in p), p
    print("[ok] 缺字段 → 报问题")

    p = problems({"measure_mode": "back_to_back", "decision": "ACCEPTED_BITEXACT"})
    assert any("必须降级" in x for x in p), p
    print("[ok] back_to_back + ACCEPTED → 报「必须降级」")

    p = problems({"measure_mode": "event_isolated", "decision": "ACCEPTED_BITEXACT"})
    assert p == [], p
    print("[ok] 合法组合无问题")

    p = problems({"measure_mode": "made_up"})
    assert any("非法" in x for x in p), p
    print("[ok] 未知 mode → 报非法")

    print("\nselftest PASS（%d 项）" % 10)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    for k, v in MODES.items():
        print("%-16s claimable=%-5s %s" % (k, v["claimable"], v["note"][:60]))
    sys.exit(0)
