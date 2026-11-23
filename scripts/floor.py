#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor —— 「还能优化多少」必须出一个数，不许靠感觉。

判据闸门（这是本文件存在的唯一理由）
------------------------------------
    remaining = t_actual / t_floor_binding - 1

    remaining < 0.10          → at_floor，别再折腾
    0.10 <= remaining < 0.30  → 做结构变换（fusion / split / layout）
    remaining >= 0.30         → 【先怀疑基线/语义/天花板错了】，不是先挖
    remaining < 0             → 已在地板以下：说明上限估低了或流量算多了

**最后两条是 2026-11-23 的教训。** 那天在 rmsnorm 上连错四次：
  - 拿 ref-heavy（只输出 1 份）当基线 → 基线根本不可比
  - 用 1724 当分母（无标定）→ 达成率虚低
  - 0.03 那种数，第一反应是「空间大」，实际是分母错

所以：**remaining >= 0.30 不给优化建议，只给「回去查分母」。**

两个地板，取紧的那个（2026-11-23 A 任务）
------------------------------------------
    memory floor  = 字节 / bw-ladder 的 achievable（按读写比分档）
    compute floor = FLOPs / constants.yaml 的 achievable 峰值
    binding       = max(两者)——理论最短时间由更慢的那条腿决定

只算一个会犯两类错：算力绑定卡（GEMM）用带宽地板会得到 remaining=8 的假空间；
带宽绑定卡用算力地板同理。卡片的 ceiling.bound 与之一致性由 validate 守。

用法
----
    python3 -I scripts/floor.py --card rmsnorm-fused   # 单卡（读卡片的 floor_inputs）
    python3 -I scripts/floor.py --all                  # 全仓（缺 floor_inputs 的会 SKIP）
    python3 -I scripts/floor.py --all --write-card     # 把算出的 remaining 写回卡片
    python3 -I scripts/floor.py --selftest
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"
CARDS = REPO / "Plugins/ops/op-atlas/cards"
CONSTANTS = REPO / "Plugins/ops/skills/op-roofline/constants.yaml"

# 模式 → bw-ladder.yaml 的层键
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

# 算力峰值：只认 achievable 口径（见 constants.yaml 三层口径）。
# impl_measured 不许当分母——那是某一条实现的值。
COMPUTE_PEAK = {
    "bf16": ("compute", "bf16_gemm_peak_tf"),
    "fp8": ("compute", "fp8_peak_layers"),
}


def load_ladder():
    import yaml
    if not LADDER.is_file():
        raise SystemExit("缺少 %s —— 先跑 python3 -I scripts/bw_calibrate.py" % LADDER)
    return yaml.safe_load(LADDER.read_text(encoding="utf-8")) or {}


def load_peak(precision):
    """读 constants.yaml 的 achievable 算力峰值。取不到就报错，不猜。"""
    import yaml
    if not CONSTANTS.is_file():
        raise SystemExit("缺少 %s" % CONSTANTS)
    c = yaml.safe_load(CONSTANTS.read_text(encoding="utf-8")) or {}
    node = (c.get("compute") or {}).get(COMPUTE_PEAK[precision][1])
    if not isinstance(node, dict):
        raise SystemExit("constants.yaml 缺 compute.%s" % COMPUTE_PEAK[precision][1])
    if precision == "fp8":
        v = node.get("achievable_tf")
    else:
        v = node.get("value")
    if not v:
        raise SystemExit("constants.yaml 的 %s 没有 achievable 值——"
                         "impl_measured 不许当分母" % COMPUTE_PEAK[precision][1])
    return float(v)


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


def memory_floor(bytes_mb, pattern, ladder):
    peak, note = ladder_value(ladder, pattern)
    if not peak:
        return None, None, note
    return bytes_mb / peak * 1000.0, peak, "%s 档 %.0f GB/s（%s）" % (pattern, peak, note)


def compute_floor(flops, peak_tf):
    return flops / (peak_tf * 1e12) * 1e6, "%.1f TF achievable" % peak_tf


def analyze(spec, ladder):
    """spec 字段：bytes_mb / pattern / flops / peak_tf(或 precision) / t_actual_us。

    **两个地板都算，取更慢的那个当 binding。** 只算一个会给出方向相反的结论。
    """
    t = spec.get("t_actual_us")
    floors = []          # (时间 µs, 名称, 依据)
    if spec.get("bytes_mb") and spec.get("pattern"):
        f, peak, why = memory_floor(spec["bytes_mb"], spec["pattern"], ladder)
        if f is None:
            return {"error": why}
        floors.append((f, "memory", why, peak))
    if spec.get("flops"):
        peak_tf = spec.get("peak_tf") or load_peak(spec.get("precision") or "bf16")
        f, why = compute_floor(spec["flops"], peak_tf)
        floors.append((f, "compute", why, peak_tf))
    if not floors:
        return {"error": "既没有 bytes_mb+pattern，也没有 flops —— 给不出地板"}

    bind = max(floors, key=lambda x: x[0])
    t_floor, bind_kind, bind_why, bind_den = bind
    remaining = (t / t_floor - 1.0) if t else None
    g, advice = gate(remaining)
    if g == "headroom" and bind_kind == "memory" and \
            spec.get("pattern") in ("rw_1to1", "rw_2to1"):
        advice += "（注意：若该算子含行内归约，这个 remaining 只是上界——" \
                  "归约惩罚尚未标定，见 bw-ladder 的 3_reduction_penalty）"
    if len(floors) == 2:
        other = min(floors, key=lambda x: x[0])
        note_other = "另一条腿 %s 地板 %.1f us（%.2fx 更松）" % (
            other[1], other[0], t_floor / other[0])
    else:
        note_other = ""
    parts = ["%s 地板 %.1f us（%s）" % (k, f, w) for f, k, w, _ in floors]
    basis = "；".join(parts) + "；binding=%s" % bind_kind
    if t:
        basis += "；实测 %.1f us" % t
    if note_other:
        basis += "；" + note_other
    return {
        "pattern": spec.get("pattern"),
        "denominator_gbs": bind_den if bind_kind == "memory" else None,
        "denominator_tf": bind_den if bind_kind == "compute" else None,
        "denominator_source": ("Plugins/ops/op-atlas/bw-ladder.yaml" if bind_kind == "memory"
                               else "Plugins/ops/skills/op-roofline/constants.yaml"),
        "bytes_mb": spec.get("bytes_mb"),
        "flops": spec.get("flops"),
        "binding": bind_kind,
        "basis": basis,
        "t_floor_us": round(t_floor, 1),
        "t_actual_us": t,
        "remaining": round(remaining, 4) if remaining is not None else None,
        "utilization": round(t_floor / t, 4) if t else None,
        "gate": g,
        "advice": advice,
    }


def card_spec(card_id):
    """从卡片的 floor_inputs 里取输入。缺字段就明确说缺什么，不猜。"""
    import yaml
    p = CARDS / ("%s.yaml" % card_id)
    if not p.is_file():
        raise SystemExit("找不到卡 %s" % p)
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    fi = d.get("floor_inputs")
    if not isinstance(fi, dict):
        return None, "缺 floor_inputs（要有 bytes_mb/pattern 或 flops，以及 t_actual_us/basis）"
    spec = {k: fi.get(k) for k in
            ("bytes_mb", "pattern", "flops", "peak_tf", "precision", "t_actual_us")}
    if not fi.get("basis"):
        return None, "floor_inputs 缺 basis —— 只有数字没有依据，拒算"
    if spec["t_actual_us"] is None:
        return None, "floor_inputs 缺 t_actual_us —— 没有实测耗时就没有 remaining"
    spec["basis"] = fi["basis"]
    spec["card"] = card_id
    return spec, None


def selftest():
    fake = {"layers": {
        "2_rw_1to1": {"achievable": 1200, "confidence": "high"},
        "2_read_only": {"achievable": 2000, "confidence": "high"},
        "2_rw_2to1": {"achievable": 1730, "confidence": "high"},
        "3_cross_row": {"achievable": None, "reason": "未标定"},
        "3_scatter_gather": {"achievable": None, "reason": "未标定"},
    }}
    # 刚好在地板：268.4MB / 1200 GB/s = 223.7 µs
    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7}, fake)
    assert abs(r["remaining"]) < 0.001, r
    assert r["gate"] == "at_floor", r
    print("[ok] 贴地板 → at_floor（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7 * 1.15}, fake)
    assert r["gate"] == "structural", r
    print("[ok] 15%% 空间 → 结构变换（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7 * 1.40}, fake)
    assert r["gate"] == "headroom", r
    assert "先怀疑" in r["advice"], r
    print("[ok] 40%% 空间 → 先怀疑基线/天花板（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 200.0}, fake)
    assert r["gate"] == "below_floor", r
    print("[ok] 快过地板 → below_floor（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "scatter_gather", "t_actual_us": 100.0}, fake)
    assert "error" in r, r
    print("[ok] 未标定模式 → 拒绝出数：%s" % r["error"])

    # 算力绑定：GEMM 4096³ 理想 2*4096^3/279e12 = 492 µs；
    # 只用带宽地板会得到 58 µs（假空间 8x），binding 必须选算力那条。
    r = analyze({"bytes_mb": 100.7, "pattern": "rw_2to1", "flops": 2 * 4096 ** 3,
                 "peak_tf": 279.0, "t_actual_us": 505.0}, fake)
    assert r["binding"] == "compute", r
    assert r["gate"] == "at_floor", r
    print("[ok] 算力绑定 → binding=compute（remaining=%.3f，带宽腿松 %.1fx）"
          % (r["remaining"], r["t_floor_us"] / 58.2))

    # 反过来：带宽绑定的算子不能因为算了 flops 就变成算力绑定
    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1",
                 "flops": 1.0e9, "peak_tf": 279.0, "t_actual_us": 242.2}, fake)
    assert r["binding"] == "memory", r
    print("[ok] 带宽绑定 → binding=memory（remaining=%.3f）" % r["remaining"])
    print("\nselftest PASS")
    return 0


def write_card_floor(card_path, result, spec):
    """把 floor 结果写进卡片。**没有 basis 就不写** —— 这是反作弊线。

    只有数字没有依据的，等于把「算出来的」和「拍脑袋的」混在一起。
    """
    import yaml
    if not result.get("basis"):
        return False, "缺 basis，拒写"
    if result.get("remaining") is None:
        return False, "remaining 为 None，拒写"
    d = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
    lv = d.setdefault("last_verdict", {})
    lv["floor"] = {
        "remaining": result["remaining"],
        "t_floor_us": result["t_floor_us"],
        "t_actual_us": result["t_actual_us"],
        "binding": result["binding"],
        "basis": spec["basis"] + " ⇒ " + result["basis"],
        "verdict": result["gate"],
        "denominator_source": result["denominator_source"],
        "updated": time.strftime("%Y-%m-%d"),
    }
    card_path.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False, width=200),
                         encoding="utf-8")
    return True, "ok"


def _print(r, as_json):
    if "error" in r:
        print("ERROR: %s" % r["error"], file=sys.stderr)
        return
    if as_json:
        import json
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print("binding    : %s（%s）" % (r["binding"], r["denominator_source"]))
    print("basis      : %s" % r["basis"])
    print("t_floor    : %.1f us" % r["t_floor_us"])
    print("t_actual   : %s us" % (r["t_actual_us"] if r["t_actual_us"] else "?"))
    if r["remaining"] is None:
        print("remaining  : ?")
    else:
        print("remaining  : %+.1f%%  → %s" % (r["remaining"] * 100, r["gate"]))


def measured_cards():
    import yaml
    out = []
    for p in sorted(CARDS.glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if d.get("status") in ("measured", "tuned"):
            out.append(p.stem)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="算本卡地板与剩余空间")
    ap.add_argument("--card")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-card", action="store_true", dest="write_card",
                    help="把 remaining/basis/t_floor_us 写进卡片的 last_verdict.floor")
    ap.add_argument("--all", action="store_true",
                    help="对所有 measured/tuned 卡跑一遍（配 --write-card 批量写）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ladder = load_ladder()
    cards = measured_cards() if a.all else ([a.card] if a.card else [])
    if not cards:
        ap.error("要么 --card，要么 --all")

    ok, failed = [], []
    for cid in cards:
        spec, why = card_spec(cid)
        if spec is None:
            failed.append((cid, why))
            print("%-26s SKIP  %s" % (cid, why))
            continue
        r = analyze(spec, ladder)
        if "error" in r:
            failed.append((cid, r["error"]))
            print("%-26s FAIL  %s" % (cid, r["error"]))
            continue
        print("%-26s %+7.1f%%  %-10s floor=%.1fus actual=%sus  %s"
              % (cid, r["remaining"] * 100, r["gate"], r["t_floor_us"],
                 r["t_actual_us"], r["binding"]))
        if a.write_card:
            wrote, msg = write_card_floor(CARDS / ("%s.yaml" % cid), r, spec)
            if wrote:
                ok.append(cid)
            else:
                failed.append((cid, msg))
                print("%-26s FAIL  %s" % (cid, msg))
        if a.json:
            _print(r, True)
    print("\n写卡 %d / 尝试 %d；未写 %d" % (len(ok), len(cards), len(failed)))
    for cid, why in failed:
        print("  - %s: %s" % (cid, why))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
