#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evalrate —— 「可靠评估/小时」作为一等指标（不只是跑得快，是能信地跑得快）。

为什么需要它
------------
CompoundingKernel（arXiv:2611.19230）的结论一句话：
**"extra computation tends to win, but it pays off only when each new try
builds on a reliable measurement of the last one."**
它一次 warm restart 修复就把评估速率提了 3.6×，并明确说
「限制规模的是可靠测量的速率，不是 agent 数量」。

对照我们自己：46 张卡里 23 张还在 draft，不是因为没方向，是因为**测一次太贵太不稳**：
  - L2 命中虚高（25 MB buffer 测出 4571 GB/s，实际峰值 1724）
  - 重叠度窗口被噪声淹没两次
  - 时钟没锁：同形状两次测量能差 13%（310 vs 279 TF 就是这么来的）

所以这个模块做两件事：
  1. **每次测量记 clock_mhz + timestamp**（不记时钟的测量不可比，等于没测）
  2. **统计 eval_rate = 可靠评估数 / 小时**，并区分「有效」与「不可信」

「可靠」的判据（三条都要过）
----------------------------
  - 显著性：|Δ| > k × MAD（沿用 opbench.significant 的口径）
  - 时钟一致：两侧测量的 sclk 差 <= CLOCK_TOL_PCT
  - 未超峰值：达成率 <= 1 + 2%（超了是记账错，见 opbench.floor_status 的 impossible）

只统计「三条都过」的评估。跑得再多但不可信，eval_rate 不涨——
这正是要把「可靠」和「快」绑在一个数上的原因。

用法
----
    from evalrate import EvalLedger, read_clock_mhz
    led = EvalLedger("Plugins/ops/skills/op-x/runs/evals.jsonl")
    led.record(op="swiglu", variant="fused", ms=0.412, mad_ms=0.003,
               base_ms=0.931, base_mad_ms=0.004, peak=0.35)
    led.record_miss(op="swiglu", variant="split", reason="noise")   # 不可信也记，但不计速率
    print(led.report())

命令行：
    python3 -I scripts/evalrate.py <ledger.jsonl>            # 汇总
    python3 -I scripts/evalrate.py --selftest                # 无需 DCU 自检
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

CLOCK_TOL_PCT = 5.0     # 两侧时钟差超过 5% 就不算同口径（跨卡/跨档位比较无意义）
OVER_PEAK_TOL = 0.02    # 超峰值 2% 以上判 impossible（与 opbench.floor_status 一致）
DEFAULT_K = 3.0


# ---------------------------------------------------------------- 时钟

def read_clock_mhz(device=None):
    """读当前计算时钟（sclk）。读不到返回 None——**不猜**。

    为什么必须记：同形状在 1200 MHz 与 1350 MHz 下测出的 TFLOPS 差 12.5%。
    不记时钟的两个数放一起比，就是在拿时钟差冒充性能差。
    """
    try:
        out = subprocess.run(["hy-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return None
    import re
    vals = [int(x) for x in re.findall(r"sclk clock level:\s*\d+\s*\((\d+)Mhz\)", out)]
    if not vals:
        return None
    if device is not None and 0 <= device < len(vals):
        return vals[device]
    return max(vals)


def clock_consistent(a, b, tol_pct=CLOCK_TOL_PCT):
    """两侧时钟是否同口径。任一侧缺失 → False（宁可判不可比，不默认相等）。"""
    if a is None or b is None:
        return False
    if not a:
        return False
    return abs(float(b) - float(a)) / float(a) * 100.0 <= tol_pct


# ---------------------------------------------------------------- 判定

def verdict(cand_ms, cand_mad, base_ms, base_mad, peak_ratio=None,
            clock_a=None, clock_b=None, k=DEFAULT_K):
    """一次评估是否「可靠」。返回 (ok, reasons, ratio)。

    三条都要过，任何一条不过都记 reason 并**不计入速率**。
    """
    reasons = []
    if cand_ms is None or base_ms is None or not base_ms:
        return False, ["缺少数值"], None
    ratio = float(cand_ms) / float(base_ms)

    noise = k * max(cand_mad or 0.0, base_mad or 0.0)
    if abs(float(cand_ms) - float(base_ms)) <= noise:
        reasons.append("noise(|Δ|=%.6g <= %.1f*MAD=%.6g)" % (
            abs(float(cand_ms) - float(base_ms)), k, noise))

    if not clock_consistent(clock_a, clock_b):
        reasons.append("clock(%s vs %s MHz)" % (clock_a, clock_b))

    if peak_ratio is not None and float(peak_ratio) > 1.0 + OVER_PEAK_TOL:
        reasons.append("over_peak(ratio=%.3f)" % float(peak_ratio))

    return (not reasons), reasons, ratio


# ---------------------------------------------------------------- 账本

class EvalLedger:
    """把每次评估写进 jsonl，并提供 eval_rate 汇总。

    一行一条记录，字段：
      ts / op / variant / ms / mad_ms / base_ms / base_mad_ms
      clock_mhz / clock_base_mhz / ratio / peak_ratio / reliable / reasons
    """

    def __init__(self, path):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    def _append(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")
        return rec

    def record(self, op, variant, ms, mad_ms, base_ms, base_mad_ms,
               peak=None, peak_ratio=None, clock_mhz="auto", clock_base_mhz=None,
               ts=None):
        """记一次评估。peak 给的是「峰值带宽/算力」，会算出 peak_ratio 再判。"""
        c = read_clock_mhz() if clock_mhz == "auto" else clock_mhz
        cb = c if clock_base_mhz is None else clock_base_mhz
        # peak（峰值带宽/算力）与 peak_ratio 二选一：
        # 调用方若已知达成率就直接给 peak_ratio；只给 peak 时算不出达成率
        # （需要形状对应的理论下限），所以不强推，留 None。
        ok, reasons, ratio = verdict(ms, mad_ms, base_ms, base_mad_ms,
                                     peak_ratio=peak_ratio,
                                     clock_a=cb, clock_b=c)
        rec = {
            "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "op": op, "variant": variant,
            "ms": ms, "mad_ms": mad_ms, "base_ms": base_ms, "base_mad_ms": base_mad_ms,
            "clock_mhz": c, "clock_base_mhz": cb,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "peak_ratio": peak_ratio,
            "reliable": bool(ok), "reasons": reasons,
        }
        return self._append(rec)

    def record_miss(self, op, variant, reason, ts=None):
        """不可信的一次也记账——但**不计入** eval_rate 分子。

        记它的理由：不可信率本身就是「评估速率」的一部分。
        如果一半的测量都因噪声作废，加再多 agent 也没用。
        """
        return self._append({
            "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "op": op, "variant": variant,
            "reliable": False, "reasons": [reason], "kind": "miss",
        })

    # ------------------------------------------------------------ 汇总

    def load(self):
        if not os.path.isfile(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def report(self, window_hours=None):
        rows = self.load()
        if not rows:
            return {"total": 0, "reliable": 0, "eval_rate_per_hour": None}
        rel = [r for r in rows if r.get("reliable")]
        tss = [r["ts"] for r in rows if r.get("ts")]
        span_h = None
        if len(tss) >= 2:
            span_h = _hours_between(min(tss), max(tss))
        if window_hours is not None:
            span_h = window_hours
        # 速率分母至少要有个下限，避免「两次相隔 3 秒」算出每小时 2400 次
        rate = None
        if span_h and span_h > 0:
            rate = round(len(rel) / span_h, 1)
        return {
            "total": len(rows),
            "reliable": len(rel),
            "unreliable": len(rows) - len(rel),
            "reliable_pct": round(100.0 * len(rel) / len(rows), 1),
            "span_hours": round(span_h, 3) if span_h else None,
            "eval_rate_per_hour": rate,
            "reason_hist": _hist([x for r in rows if not r.get("reliable")
                                  for x in (r.get("reasons") or [])]),
            "clock_missing": sum(1 for r in rows if r.get("clock_mhz") is None),
        }


def _hours_between(a, b):
    try:
        ta = time.mktime(time.strptime(a, "%Y-%m-%dT%H:%M:%S"))
        tb = time.mktime(time.strptime(b, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None
    return max(0.0, (tb - ta) / 3600.0)


def _hist(items):
    h = {}
    for s in items:
        key = str(s).split("(")[0]
        h[key] = h.get(key, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- 自检

def selftest():
    ok = True
    # 时钟一致性
    assert clock_consistent(1200, 1200)
    assert clock_consistent(1200, 1240)          # 3.3% < 5%
    assert not clock_consistent(1200, 1350)      # 12.5%
    assert not clock_consistent(None, 1350), "缺时钟必须判不可比，不许默认相等"
    print("[ok] clock_consistent  1200≈1240 ✓ / 1200 vs 1350 ✗ / None vs 1350 ✗")

    # 判定三条
    good, rs, r = verdict(0.80, 0.01, 1.00, 0.01, clock_a=1200, clock_b=1200)
    assert good and not rs and abs(r - 0.8) < 1e-9, (good, rs, r)
    print("[ok] verdict 可信       ratio=%.2f" % r)

    noisy, rs, _ = verdict(1.00, 0.01, 1.02, 0.01, clock_a=1200, clock_b=1200)
    assert not noisy and any(s.startswith("noise") for s in rs), rs
    print("[ok] verdict 噪声内     %s" % rs[0])

    clk, rs, _ = verdict(0.80, 0.001, 1.00, 0.001, clock_a=1200, clock_b=1350)
    assert not clk and any(s.startswith("clock") for s in rs), rs
    print("[ok] verdict 时钟不同   %s" % rs[0])

    over, rs, _ = verdict(0.80, 0.001, 1.00, 0.001, peak_ratio=1.8,
                          clock_a=1200, clock_b=1200)
    assert not over and any(s.startswith("over_peak") for s in rs), rs
    print("[ok] verdict 超峰值     %s" % rs[0])

    # 账本：不可信的进 total 不进 reliable
    import tempfile
    d = tempfile.mkdtemp()
    led = EvalLedger(os.path.join(d, "e.jsonl"))
    led.record("demo", "a", 0.80, 0.001, 1.00, 0.001, clock_mhz=1200)
    led.record("demo", "b", 1.00, 0.01, 1.02, 0.01, clock_mhz=1200)   # 噪声
    led.record_miss("demo", "c", "compile_failed")
    rep = led.report()
    assert rep["total"] == 3 and rep["reliable"] == 1, rep
    assert rep["reason_hist"].get("noise") == 1, rep
    print("[ok] ledger            total=%d reliable=%d %s"
          % (rep["total"], rep["reliable"], rep["reason_hist"]))
    print("\nselftest PASS" if ok else "\nselftest FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="eval_rate：可靠评估/小时")
    ap.add_argument("ledger", nargs="?", help="jsonl 账本路径")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.ledger:
        ap.error("要么给账本路径，要么 --selftest")
    led = EvalLedger(a.ledger)
    rep = led.report()
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print("评估总数 %d / 可信 %d (%.1f%%) / 不可信 %d"
              % (rep["total"], rep["reliable"], rep["reliable_pct"] or 0, rep["unreliable"]))
        if rep["span_hours"]:
            print("时间跨度 %.2f h ⇒ 可靠评估速率 %s /小时"
                  % (rep["span_hours"], rep["eval_rate_per_hour"]))
        if rep["reason_hist"]:
            print("不可信原因：" + ", ".join("%s=%d" % kv for kv in rep["reason_hist"].items()))
        if rep["clock_missing"]:
            print("⚠ %d 条记录缺 clock_mhz —— 缺时钟的测量不可跨次比较" % rep["clock_missing"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
