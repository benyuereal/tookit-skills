#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bw_calibrate —— 标定本机可达带宽，给「达成率」提供可复跑的分母。

为什么需要它
------------
`constants.yaml:memory.stream_read_peak = 1724 GB/s` 长期被当分母用，
但它在一段时间里**只有断言、没有测量记录**，全仓也没有标定脚本。
2026-11-23 实测四项：

    copy  y=x      (R+W)   1126 GB/s
    mul   y=x*2    (R+W)   1351 GB/s
    add   y=x+r    (2R+W)  1498 GB/s   ← 观测最高
    sum   x        (R)      811 GB/s

**没有一个到 1724。** 所以 1724 只能标 `unverified`，或者在补上测量前
与「实测 elementwise 峰值」并列双报。

口径（必须随数字一起给）
------------------------
  * copy/mul 是 R+W，分母要按 2×字节；add 是 2R+W，按 3×；sum 是纯读，按 1×。
    分母口径不同，**不同行的 GB/s 不可直接互比**。
  * 计时用 CUDA event（异步 kernel 用 perf_counter 会量到 launch）。
  * 每项 30 次取中位，前 5 次预热。

用法
----
    python3 -I scripts/bw_calibrate.py
    python3 -I scripts/bw_calibrate.py --json results/bw.json
    python3 -I scripts/bw_calibrate.py --check    # 与 constants.yaml 对账
"""
import argparse
import json
import statistics
import sys
import time

# 实测参考值（2026-11-23，BW1100，gfx938，sclk 1350 MHz）
REFERENCE = {"copy": 1126, "mul": 1351, "add": 1498, "sum": 811}
CONSTANTS_PEAK = 1724     # constants.yaml:memory.stream_read_peak


def _bench(fn, torch, rounds=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        e1.record()
        fn()
        e2.record()
        torch.cuda.synchronize()
        ts.append(e1.elapsed_time(e2))
    ts.sort()
    return ts[len(ts) // 2], statistics.median([abs(t - ts[len(ts) // 2]) for t in ts])


def _sclk():
    try:
        import re
        import subprocess
        out = subprocess.run(["hy-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=15).stdout
        v = [int(x) for x in re.findall(r"sclk clock level:\s*\d+\s*\((\d+)Mhz\)", out)]
        return max(v) if v else None
    except Exception:                                            # noqa: BLE001
        return None


def run(torch):
    n = 8192 * 4096
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    mb = n * 2 / 1e6
    cases = [
        ("copy", "y = x        (R+W)", lambda: y.copy_(x), 2 * mb),
        ("mul", "y = x*2      (R+W)", lambda: torch.mul(x, 2.0, out=y), 2 * mb),
        ("add", "y = x+r      (2R+W)", lambda: torch.add(x, r, out=y), 3 * mb),
        ("sum", "sum(x)       (R)  ", lambda: torch.sum(x), 1 * mb),
    ]
    rows = []
    for key, label, fn, mbytes in cases:
        ms, mad = _bench(fn, torch)
        rows.append({"key": key, "label": label, "ms": round(ms, 4),
                     "mad_ms": round(mad, 5), "mb": round(mbytes, 1),
                     # ms 是毫秒：MB/ms = GB/s（1e6 B / 1e-3 s = 1e9 B/s）
                     "gbs": round(mbytes / ms)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="本机带宽标定")
    ap.add_argument("--json", default=None)
    ap.add_argument("--check", action="store_true",
                    help="与 constants.yaml 的 stream_read_peak 对账")
    a = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch 不可用", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2

    rows = run(torch)
    print("%-28s %9s %8s %8s" % ("test", "time", "MB", "GB/s"))
    for r in rows:
        print("%-28s %8.1fus %8.1f %8d" % (r["label"], r["ms"] * 1000, r["mb"], r["gbs"]))
    best = max(rows, key=lambda r: r["gbs"])
    print("\n观测最高 %.0f GB/s（%s）  vs constants 的 %d GB/s"
          % (best["gbs"], best["key"], CONSTANTS_PEAK))
    print("⇒ 没有一个到 %d。该常数应标 unverified，或与观测峰值并列双报。"
          % CONSTANTS_PEAK)

    if a.json:
        payload = {"sclk_mhz": _sclk(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "rows": rows, "observed_max_gbs": best["gbs"],
                   "constants_peak_gbs": CONSTANTS_PEAK,
                   "caliber": "CUDA event median of 30, warmup 5"}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("落盘 -> %s" % a.json)

    if a.check and best["gbs"] < CONSTANTS_PEAK * 0.9:
        print("\n!! constants 的 %d 比观测最高 %.0f 高 %.0f%%，该常数不可作分母"
              % (CONSTANTS_PEAK, best["gbs"],
                 (CONSTANTS_PEAK / best["gbs"] - 1) * 100))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
