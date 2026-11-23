#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor —— 「还能优化多少」必须出一个数，不许靠感觉。

判据闸门（这是本文件存在的唯一理由）
------------------------------------
    remaining = t_actual / t_floor - 1

    remaining < 0.10          → at_floor，别再折腾
    0.10 <= remaining < 0.30  → 做结构变换（fusion / split / layout）
    remaining >= 0.30         → 【先怀疑基线/语义/天花板错了】，不是先挖
    remaining < 0             → 已在地板以下：说明上限估低了或流量算多了

**最后两条是 2026-11-23 的教训。** 那天在 rmsnorm 上连错四次：
  - 拿 ref-heavy（只输出 1 份）当基线 → 基线根本不可比
  - 用 1724 当分母（无标定）→ 达成率虚低
  - 0.03 那种数，第一反应是「空间大」，实际是分母错

所以：**remaining >= 0.30 不给优化建议，只给「回去查分母」。**

用法
----
    python3 -I scripts/floor.py --card rmsnorm-fused --shape prefill-8k
    python3 -I scripts/floor.py --card gemm-fp8 --shape prefill-8k
    python3 -I scripts/floor.py --traffic 268.4 --pattern row_reduction --actual-us 242.2
    python3 -I scripts/floor.py --selftest
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"
CARDS = REPO / "Plugins/ops/op-atlas/cards"
SKILLS = REPO / "Plugins/ops/skills"

# 卡 → 访存模式。新增卡时在这里登记；未登记则要求显式 --pattern。
CARD_PATTERN = {
    # 按「实际读写比」分类，不是按算子族。
    # rmsnorm 融合残差 = 2读2写 = 1:1
    "rmsnorm-fused": "rw_1to1",
    "fused-add-rmsnorm": "rw_1to1",
    "layernorm": "rw_1to1",
    "qk-norm": "rw_1to1",
    "moe-combine-reduce": "rw_1to1",
    "rope": "r_only",              # 纯旋转写回，读主导
    "rope-mla": "r_only",
    "mlp-swiglu": "rw_2to1",       # 读 2 写 1
    "mlp-gelu-geglu": "rw_2to1",
    "gemm-bf16": "rw_1to1",
    "gemm-fp8": "rw_1to1",
    "gemm-int8": "rw_1to1",
    "paged-attention-decode": "scatter_gather",
    "mla-absorb": "scatter_gather",
    "topk-block-index": "scatter_gather",
}

# 模式 → bw-ladder.yaml 的层键
# 按读写比分档（bw-ladder 的 ② 层）。旧的 "streaming"/"row_reduction" 已废弃——
# 那个分类混淆了「访存模式」与「实现质量」，导致用 torch 的 1214 当上限。
PATTERN_TO_LAYER = {
    "r_only": "2_read_only",
    "w_only": "2_write_only",
    "rw_1to1": "2_rw_1to1",
    "rw_2to1": "2_rw_2to1",
    "cross_row": "3_cross_row",
    "scatter_gather": "3_scatter_gather",
}

# (阈值, gate 名, 建议)。按阈值从大到小匹配：remaining >= 阈值 就采用该档。
GATE = ((0.30, "headroom", "先怀疑基线/语义/天花板错了，不是先挖"),
        (0.10, "structural", "做结构变换（fusion / split / layout）"),
        (0.0, "at_floor", "at_floor，别再折腾"))


def load_ladder():
    import yaml
    if not LADDER.is_file():
        raise SystemExit("缺少 %s —— 先跑 python3 -I scripts/bw_calibrate.py" % LADDER)
    return yaml.safe_load(LADDER.read_text(encoding="utf-8")) or {}


def ladder_value(ladder, pattern):
    key = PATTERN_TO_LAYER.get(pattern)
    if not key:
        return None, "未知访存模式 %r（可选 %s）" % (pattern, sorted(PATTERN_TO_LAYER))
    node = (ladder.get("layers") or {}).get(key) or {}
    val = node.get("achievable")
    if not val:
        return None, "该模式未标定：%s" % (node.get("reason") or "无记录")
    return float(val), "confidence=%s" % node.get("confidence", "?")


def gate(remaining):
    """返回 (判据, 建议)。remaining 可为 None（算不出）。"""
    if remaining is None:
        return "unknown", "缺数据，无法判定"
    if remaining < 0:
        return "below_floor", "已在地板以下 —— 说明上限估低了或流量算多了，**先查这个**"
    for th, name, advice in GATE:
        if remaining >= th:
            return name, advice
    return "at_floor", "at_floor，别再折腾"


def compute(traffic_mb, pattern, actual_us, ladder):
    peak, note = ladder_value(ladder, pattern)
    if not peak:
        return {"pattern": pattern, "error": note}
    t_floor_us = traffic_mb / peak * 1000.0      # MB / (GB/s) = ms → *1000 = µs
    remaining = (actual_us / t_floor_us - 1.0) if actual_us else None
    g, advice = gate(remaining)
    # 归约类算子：纯 copy 地板只是上界。加提示，避免把「结构性代价」误读成「优化空间」。
    if pattern in ("rw_1to1", "rw_2to1") and g == "headroom":
        advice += "（注意：若该算子含行内归约，这个 remaining 只是上界——" \
                  "归约惩罚尚未标定，见 bw-ladder 的 3_reduction_penalty）"
    return {
        "pattern": pattern,
        "traffic_mb": round(traffic_mb, 1),
        "denominator_gbs": peak,
        "denominator_note": note,
        "t_floor_us": round(t_floor_us, 1),
        "t_actual_us": actual_us,
        "remaining": round(remaining, 4) if remaining is not None else None,
        "utilization": round(t_floor_us / actual_us, 4) if actual_us else None,
        "gate": g,
        "advice": advice,
    }


# ---------------------------------------------------------------- 卡输入

def card_traffic_and_time(card_id, shape_name):
    """从卡片的 last_verdict 与 case results 推流量与实测时间。

    流量优先用 last_verdict 里显式声明的 (traffic_passes, inputs, outputs)；
    缺了就用 pattern 的默认口径，并**在输出里标注这是推定值**。
    """
    import yaml
    card = CARDS / ("%s.yaml" % card_id)
    if not card.is_file():
        raise SystemExit("找不到卡 %s" % card)
    d = yaml.safe_load(card.read_text(encoding="utf-8")) or {}
    lv = d.get("last_verdict") or {}
    r = lv.get("ratio") or {}

    # 形状参数
    shapes_p = SKILLS / ("op-%s" % card_id) / "shapes.yaml"
    M = H = None
    if shapes_p.is_file():
        sh = yaml.safe_load(shapes_p.read_text(encoding="utf-8")) or {}
        params = (sh.get(shape_name) or {}).get("params") or {}
        M, H = params.get("M"), params.get("H")

    ins = lv.get("inputs", r.get("inputs"))
    outs = lv.get("outputs", r.get("outputs"))
    passes = lv.get("traffic_passes", r.get("traffic_passes"))
    assumed = False
    if ins is None or outs is None:
        ins, outs = 2, 1          # 最常见形态，标注为推定
        assumed = True
    if passes is None:
        passes = ins + outs
        assumed = True

    # 流量：优先 M*H，其次用 ratio.scope 里记录的实测带宽反推
    traffic_mb = None
    if M and H:
        traffic_mb = passes * M * H * 2 / 1e6
    actual_us = None
    note = ""
    if r.get("value") and r.get("kind") == "utilization":
        # util = achieved/peak ⇒ 没有绝对时间就无法反推；留给调用方给 --actual-us
        note = "卡片只有利用率，没有绝对耗时；请用 --actual-us 显式给"
    return {"card": card_id, "shape": shape_name, "M": M, "H": H,
            "inputs": ins, "outputs": outs, "traffic_passes": passes,
            "traffic_mb": traffic_mb, "actual_us": actual_us,
            "assumed": assumed, "note": note,
            "status": d.get("status"), "ratio_value": r.get("value")}


def selftest():
    fake = {"layers": {
        "2_rw_1to1": {"achievable": 1200, "confidence": "high"},
        "2_read_only": {"achievable": 2000, "confidence": "high"},
        "cross_row": {"achievable": None, "reason": "未标定"},
        "3_cross_row": {"achievable": None, "reason": "未标定"},
        "3_scatter_gather": {"achievable": None, "reason": "未标定"},
    }}
    # 刚好在地板：268.4MB / 1200 GB/s = 223.7 µs
    r = compute(268.4, "rw_1to1", 223.7, fake)
    assert abs(r["remaining"]) < 0.001, r
    assert r["gate"] == "at_floor", r
    print("[ok] 贴地板 → at_floor（remaining=%.3f）" % r["remaining"])

    # 有 15% 空间 → 结构变换
    r = compute(268.4, "rw_1to1", 223.7 * 1.15, fake)
    assert r["gate"] == "structural", r
    print("[ok] 15%% 空间 → 结构变换（remaining=%.3f）" % r["remaining"])

    # 40% 空间 → 先怀疑分母（这是今天最重要的那条）
    r = compute(268.4, "rw_1to1", 223.7 * 1.40, fake)
    assert r["gate"] == "headroom", r
    assert "先怀疑" in r["advice"], r
    print("[ok] 40%% 空间 → 先怀疑基线/天花板（remaining=%.3f）" % r["remaining"])

    # 实际比地板还快 → 说明上限估低了
    r = compute(268.4, "rw_1to1", 200.0, fake)
    assert r["gate"] == "below_floor", r
    print("[ok] 快过地板 → below_floor（remaining=%.3f）" % r["remaining"])

    # 未标定的模式 → 明确报错，不瞎给数
    r = compute(268.4, "scatter_gather", 100.0, fake)
    assert "error" in r, r
    print("[ok] 未标定模式 → 拒绝出数：%s" % r["error"])

    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="算本卡地板与剩余空间")
    ap.add_argument("--card")
    ap.add_argument("--shape", default="prefill-8k")
    ap.add_argument("--pattern", choices=sorted(PATTERN_TO_LAYER))
    ap.add_argument("--traffic", type=float, help="流量 MB")
    ap.add_argument("--actual-us", type=float, help="实测耗时 µs")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ladder = load_ladder()
    if a.traffic is not None and a.pattern:
        r = compute(a.traffic, a.pattern, a.actual_us, ladder)
        _print(r, a.json)
        return 0 if "error" not in r else 1

    if not a.card:
        ap.error("要么 --card --shape，要么 --traffic --pattern --actual-us")

    info = card_traffic_and_time(a.card, a.shape)
    pattern = a.pattern or CARD_PATTERN.get(a.card)
    if not pattern:
        print("卡 %s 未登记访存模式，请用 --pattern 显式给" % a.card, file=sys.stderr)
        return 2
    traffic = a.traffic if a.traffic is not None else info["traffic_mb"]
    if traffic is None:
        print("算不出流量（缺 M/H）。用 --traffic 显式给。", file=sys.stderr)
        return 2
    actual = a.actual_us
    if actual is None:
        print("缺实测耗时。用 --actual-us 给。", file=sys.stderr)
        print("  卡 %s / 形状 %s：M=%s H=%s inputs=%s outputs=%s passes=%s%s"
              % (a.card, a.shape, info["M"], info["H"], info["inputs"],
                 info["outputs"], info["traffic_passes"],
                 "（推定）" if info["assumed"] else ""), file=sys.stderr)
        return 2
    r = compute(traffic, pattern, actual, ladder)
    r["card"] = a.card
    r["shape"] = a.shape
    r["traffic_source"] = ("推定" if info["assumed"] else "卡片声明")
    _print(r, a.json)
    return 0


def _print(r, as_json):
    if "error" in r:
        print("ERROR: %s" % r["error"], file=sys.stderr)
        return
    if as_json:
        import json
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print("pattern    : %s" % r["pattern"])
    print("traffic    : %.1f MB" % r["traffic_mb"])
    print("denominator: %.0f GB/s（%s）" % (r["denominator_gbs"], r["denominator_note"]))
    print("t_floor    : %.1f us" % r["t_floor_us"])
    print("t_actual   : %s us" % (r["t_actual_us"] if r["t_actual_us"] else "?"))
    if r["remaining"] is None:
        print("remaining  : ?")
    else:
        print("remaining  : %+.1f%%  → %s" % (r["remaining"] * 100, r["gate"]))
        print("建议       : %s" % r["advice"])


if __name__ == "__main__":
    sys.exit(main())
