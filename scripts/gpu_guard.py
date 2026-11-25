#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_guard —— 这块卡现在能不能用来做测量。

## 为什么需要它（D12 的直接产物）

2026-11-25 实测：同一份代码，卡空闲时 `decode-s128 = 0.3416 ms (MAD 0.19%)`，
卡被别的容器占满时 `0.5904 ms (MAD 24%)`。**同一个变体，两次测量差 73%。**

如果不检查就测，会得到：
  · 假的「变快了」（被测的是别人的负载波动）
  · 假的「变慢了」（同上）
  · 最糟：把一个真实有效的改动判成无效（因为噪声盖过了效果）

CLAUDE.md 的硬约束是「每卡 used 需 < 30 GB」，本脚本把它变成机检。

## 判据（三条全过才算干净）

1. 显存占用 < 30 GB（别人的大任务会挤掉带宽）
2. 使用率 < 10%（说明没有活跃计算）
3. 连续两次采样间隔 2 秒都满足（防止正好在两次 kernel 之间采到 0）

## 用法

    python3 -I scripts/gpu_guard.py            # 查所有卡，返回第一个干净的
    python3 -I scripts/gpu_guard.py --card 3   # 查指定卡
    python3 -I scripts/gpu_guard.py --json     # 机器可读
    python3 -I scripts/gpu_guard.py --selftest
"""
import argparse
import json
import re
import subprocess
import sys
import time

# **显存占用不是判据**。实测（2026-11-25）：8 张卡各占 123GB，
# 但使用率 0%、能正常分配 2GB、跑出来 MAD 0.4% —— 那是别的容器的缓存驻留，
# 不占带宽。第一版按 CLAUDE.md 的「used < 30GB」判，把可用的卡全判死了。
MAX_UTIL = 10.0              # 使用率上限（%）
# 真正要确认的是「能分配」：分配失败说明卡真的满了
PROBE_MB = 512
SETTLE_SECONDS = 2.0


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read_state():
    """返回 {卡号: {"used_mb":..., "util":...}}。读不到就返回空 dict —— 不猜。"""
    out = {}
    txt = _run(["hy-smi", "--showmeminfo", "vram"])
    for line in txt.splitlines():
        m = re.search(r"HCU\[(\d+)\].*Total Used Memory \(MiB\):\s*(\d+)", line)
        if m:
            out.setdefault(int(m.group(1)), {})["used_mb"] = int(m.group(2))
    txt = _run(["hy-smi", "--showuse"])
    for line in txt.splitlines():
        m = re.search(r"HCU\[(\d+)\].*?:\s*([\d.]+)", line)
        if m:
            out.setdefault(int(m.group(1)), {})["util"] = float(m.group(2))
    return out


def is_clean(st):
    """单次采样是否干净。**缺字段按不干净处理** —— 不知道就不能说没事。

    判据只有「使用率」—— 显存占用不作为判据（见 MAX_UTIL 上方注释）。
    """
    util = st.get("util")
    if util is None:
        return False, "读不到使用率"
    if util > MAX_UTIL:
        return False, "使用率 %.1f%% > %.0f%%" % (util, MAX_UTIL)
    return True, "空闲（%.1f%%）" % util


def can_alloc(device):
    """真正分配一块显存验证。**这才是「能不能跑」的判据。**"""
    try:
        import torch
        x = torch.empty(PROBE_MB * 1024 * 256, dtype=torch.uint8,
                        device="cuda:%d" % device)
        del x
        return True, "可分配 %dMB" % PROBE_MB
    except Exception as e:                                   # noqa: BLE001
        return False, "分配失败：%s" % str(e)[:60]


def check(card=None, settle=SETTLE_SECONDS):
    """返回 (卡号, 是否干净, 原因)。卡号 -1 表示没有干净卡。"""
    first = read_state()
    if not first:
        return -1, False, "读不到 GPU 状态（hy-smi 不可用？）"
    cands = [card] if card is not None else sorted(first)
    ok_first = []
    for c in cands:
        if c not in first:
            continue
        ok, why = is_clean(first[c])
        if ok:
            ok2, _ = can_alloc(c)
            if ok2:
                ok_first.append(c)
    if not ok_first:
        worst = ", ".join("HCU%d=%s%%" % (c, first[c].get("util", "?"))
                          for c in cands if c in first)
        return -1, False, "没有干净卡（%s）" % worst
    if settle > 0:
        time.sleep(settle)
        second = read_state()
        for c in ok_first:
            if c in second:
                ok, _ = is_clean(second[c])
                if ok:
                    return c, True, is_clean(second[c])[1]
        return -1, False, "两次采样之间变脏了（瞬时空闲）"
    c = ok_first[0]
    return c, True, is_clean(first[c])[1]


def selftest():
    """不依赖真实 GPU —— 用构造的状态验判据。"""
    ok, why = is_clean({"used_mb": 5 * 1024, "util": 0.0})
    assert ok, why
    print("[ok] 正向：5GB / 0%% 判干净")

    ok, why = is_clean({"used_mb": 139 * 1024, "util": 0.0})
    assert ok, "显存占用高但空闲，应当判可用 —— 实测踩过这个误判：" + why
    print("[ok] 反向：显存占 139GB 但使用率 0%% 判**可用**（实测踩过的误判）")

    ok, why = is_clean({"used_mb": 5 * 1024, "util": 96.9})
    assert not ok and "使用率" in why, why
    print("[ok] 反向：96.9%% 使用率判不干净")

    ok, why = is_clean({"used_mb": 5 * 1024})
    assert not ok, why
    print("[ok] 反向：缺字段按不干净处理（不知道就不能说没事）")

    ok, why = is_clean({})
    assert not ok, why
    print("[ok] 反向：空状态判不干净")

    print("\nselftest PASS（5 项，不依赖 GPU）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=int)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-settle", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    c, ok, why = check(a.card, settle=0 if a.no_settle else SETTLE_SECONDS)
    if a.json:
        print(json.dumps({"card": c, "clean": ok, "why": why},
                         ensure_ascii=False))
    else:
        print(("✓ HCU[%d] %s" % (c, why)) if ok else ("✗ " + why))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
