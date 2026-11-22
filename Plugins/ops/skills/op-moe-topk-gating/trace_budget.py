#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 torch-profiler trace 提取 MoE top-k gating（第 2 段） 的子算子预算。

这是把「线上 profile 里的热点」映射到「算子卡」的唯一通道：
trace 里看到的时间是 kernel 级的，卡片里说的是算子级的，中间差一层。

只读 trace，不跑任何 kernel。缺 trace 时明确退出，不给空结果。

用法:
    python3 trace_budget.py --trace profile.json[.gz] [--top 10]
"""
import argparse
import collections
import gzip
import json
import sys

# 本段在 trace 里可能出现的 kernel 名片段（大小写不敏感匹配）
KERNEL_PATTERNS = ["topk", "gate", "softmax", "moe_gate"]


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def main(argv=None):
    p = argparse.ArgumentParser(description="MoE top-k gating（第 2 段） 子算子预算提取")
    p.add_argument("--trace", required=True, help="torch-profiler trace (.json / .json.gz)")
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args(argv)

    try:
        events = load(a.trace)
    except OSError as e:
        print("无法读取 trace: %s" % e, file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print("trace 格式不对（需 torch-profiler JSON）: %s" % e, file=sys.stderr)
        return 2

    hit = collections.Counter()
    total_us = 0.0
    for ev in events:
        if ev.get("ph") != "X" or not ev.get("dur"):
            continue
        dur = float(ev["dur"])
        total_us += dur
        name = str(ev.get("name", "")).lower()
        for pat in KERNEL_PATTERNS:
            if pat in name:
                hit[ev.get("name")] += dur / 1000.0      # µs -> ms
                break

    if not hit:
        print("trace 里没有匹配到本段的 kernel（模式: %s）" % sorted(KERNEL_PATTERNS),
              file=sys.stderr)
        print("可能原因：① 该段被融合进了别的 kernel ② trace 没录到 kernel 级事件",
              file=sys.stderr)
        return 1

    total_hit = sum(hit.values())
    print("%-58s %10s %8s" % ("kernel", "ms", "占比"))
    for name, ms in hit.most_common(a.top):
        print("%-58s %10.3f %7.1f%%" % (name[:58], ms, 100.0 * ms / total_hit))
    print("-" * 78)
    print("%-58s %10.3f" % ("本段合计", total_hit))
    print("%-58s %10.3f" % ("trace 全部 kernel", total_us / 1000.0))
    print("\n注意：这是 kernel 级时间，不是算子级。融合后可能落在别的 kernel 名下。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
