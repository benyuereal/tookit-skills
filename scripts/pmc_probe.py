#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pmc_probe —— 逐个数一遍：这个 PMC 字段在本机到底采不采得到。

为什么必须有这个工具
--------------------
`scripts/pmc.py` 里有一个 COUNTERS 元组，但那个元组只说明**解析器知道这些名字**，
不说明**本机采得到**。这两件事被混为一谈过：
`surfaces.yaml` 的 check 字符串里写着 `pmc: GRBM_COUNT / SQ_WAVES`，
`op-profiling/references/cards.md` 说 occupancy 用 `SQ_WAVES ÷ GRBM_COUNT`，
而 `SQ_WAVES` 在这台机器的 PMC CSV 里**根本不存在**。

后果是决策树里写了采不到的指标，agent 拿 0 当「没问题」——这是最难发现的一类错。

本工具的判据
------------
对一个真跑出来的 PMC CSV 逐列核对，输出三态：
  present   列存在且有非零值           → 可以写进判据
  all_zero  列存在但全为 0             → **不一定可用**：可能真没发生，也可能没启用
  missing   列不存在                   → 本机采不到，判据里不许出现

`all_zero` 单列出来是刻意的：把它当 0 直接下结论，就是把「没采到」当成「没发生」。

顺带发现（2026-11-24 实测）
--------------------------
CSV 自带 `grd`（网格）、`wgr`（每工作组线程数）、`lds`、`arch_vgpr`、
`accum_vgpr`、`sgpr`、`wave_size` 这几列——**G5 的放置条件可以直接从 PMC CSV 取真值**，
不必依赖编译产物 `.so` 还在不在。这比 `inspect_so.py` 更贴近「这次测量实际跑的是什么」。

用法
----
    # 采一份（注意：本版 hipprof 不认 `--` 分隔符，直接跟命令）
    HIP_VISIBLE_DEVICES=6 hipprof --pmc --pmc-type 3 -o /tmp/pmcout/x python3 my.py

    python3 -I scripts/pmc_probe.py /tmp/pmcout/x.csv
    python3 -I scripts/pmc_probe.py /tmp/pmcout/x.csv --json
    python3 -I scripts/pmc_probe.py --selftest
"""
import argparse
import csv
import io
import json
import sys

# 判据里可能用到的字段（含 pmc.py 的 COUNTERS 全部 + 常被误以为存在的）
WANTED = [
    "GRBM_COUNT", "SQ_INSTS_VALU", "SQ_ACTIVE_INST_VALU", "SQ_INSTS_LDS",
    "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_LDS_BANK_CONFLICT",
    "SQ_WAIT_INST_LDS", "TA_TA_BUSY", "TCP_TCP_TA_DATA_STALL_CYCLES",
    "TA_FLAT_READ_WAVEFRONTS", "TCC_HIT", "TCC_MISS",
    # 这些经常被写进判据但其实要单独验证：
    "SQ_WAVES", "SQ_INSTS_BARRIER", "SQ_INSTS_SALU", "SQ_INSTS_VMEM",
]
# 放置条件字段：有就直接能填 G5，不用去翻 .so
PLACEMENT_COLS = ["grd", "wgr", "lds", "arch_vgpr", "accum_vgpr", "sgpr", "wave_size"]


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _cols_present(cols, name):
    """列名可能带 [0..N] 实例后缀（TA_*/TCP_*/TCC_* 都是）。"""
    return [c for c in cols if c == name or c.startswith(name + "[")]


def probe(text):
    """返回 {counters: {name: {status, total, columns}}, placement: {...}}。"""
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r]
    if not rows:
        raise ValueError("CSV 里没有数据行")
    cols = list(rows[0].keys())

    out = {"rows": len(rows), "counters": {}, "placement": {}}
    for name in WANTED:
        hit = _cols_present(cols, name)
        if not hit:
            out["counters"][name] = {"status": "missing", "total": None,
                                     "columns": 0, "why": "CSV 里没有这一列"}
            continue
        total = 0.0
        ok = True
        for r in rows:
            for c in hit:
                v = _num(r.get(c))
                if v is None:
                    ok = False
                else:
                    total += v
        if not ok:
            out["counters"][name] = {"status": "unparsable", "total": None,
                                     "columns": len(hit), "why": "列存在但值不是数"}
        elif total == 0:
            out["counters"][name] = {
                "status": "all_zero", "total": 0.0, "columns": len(hit),
                "why": "列存在但全 0 —— 可能真的没发生，也可能没启用/要权限，"
                       "**不许直接当成 0 下结论**"}
        else:
            out["counters"][name] = {"status": "present", "total": total,
                                     "columns": len(hit), "why": ""}

    for c in PLACEMENT_COLS:
        out["placement"][c] = (_num(rows[0].get(c)) if c in cols else None)
    return out


def verdict(res):
    """整体判定：哪些能进判据、哪些不能。"""
    usable = [k for k, v in res["counters"].items() if v["status"] == "present"]
    banned = [k for k, v in res["counters"].items()
              if v["status"] in ("missing", "unparsable", "all_zero")]
    return usable, banned


def show(res):
    usable, banned = verdict(res)
    print("PMC 字段可用性（%d 行数据）" % res["rows"])
    print("%-32s %-10s %6s %10s  %s"
          % ("字段", "状态", "列数", "合计", "说明"))
    print("-" * 96)
    for name, v in res["counters"].items():
        tot = "-" if v["total"] is None else "%.0f" % v["total"]
        print("%-32s %-10s %6d %10s  %s"
              % (name, v["status"], v["columns"], tot, v["why"]))
    print("\n可写进判据：%s" % (", ".join(usable) if usable else "（无）"))
    print("**不许写进判据**：%s" % (", ".join(banned) if banned else "（无）"))
    p = res["placement"]
    got = {k: v for k, v in p.items() if v is not None}
    print("\n放置条件（G5 可直接取真值）：%s"
          % (", ".join("%s=%g" % (k, v) for k, v in got.items()) if got else "（CSV 不提供）"))
    miss = [k for k, v in p.items() if v is None]
    if miss:
        print("缺失：%s（要另找来源，不许默认 0）" % ", ".join(miss))


# ------------------------------------------------------------------ selftest

SAMPLE_FULL = """Index,KernelName,grd,wgr,lds,arch_vgpr,sgpr,wave_size,GRBM_COUNT,SQ_INSTS_VALU,SQ_INSTS_LDS,SQ_WAVES
0,k1,1024,256,16384,80,48,64,10000,5000,300,0
1,k1,1024,256,16384,80,48,64,9000,4500,280,0
"""

SAMPLE_MISSING = """Index,KernelName,GRBM_COUNT,SQ_INSTS_VALU
0,k1,10000,5000
"""


def selftest():
    r = probe(SAMPLE_FULL)
    assert r["counters"]["GRBM_COUNT"]["status"] == "present", r
    assert r["counters"]["GRBM_COUNT"]["total"] == 19000, r
    print("[ok] 存在且非零 → present（GRBM 两行求和 19000）")

    assert r["counters"]["SQ_WAVES"]["status"] == "all_zero", r
    print("[ok] 列存在但全 0 → all_zero（**不是 missing，也不是 0**）")

    r2 = probe(SAMPLE_MISSING)
    assert r2["counters"]["SQ_WAVES"]["status"] == "missing", r2
    assert r2["counters"]["SQ_LDS_BANK_CONFLICT"]["status"] == "missing", r2
    print("[ok] 列不存在 → missing（不许当成 0）")

    usable, banned = verdict(r2)
    assert "SQ_WAVES" in banned and "GRBM_COUNT" in usable, (usable, banned)
    print("[ok] 判定：missing/all_zero 一律进 banned，不许进判据")

    assert r["placement"]["arch_vgpr"] == 80, r["placement"]
    assert r["placement"]["wgr"] == 256, r["placement"]
    print("[ok] 放置条件从 CSV 直取：vgpr=80 wgr=256")

    r3 = probe("Index,KernelName,GRBM_COUNT\n0,k1,abc\n")
    assert r3["counters"]["GRBM_COUNT"]["status"] == "unparsable", r3
    print("[ok] 列存在但值不是数 → unparsable（不静默当 0）")

    try:
        probe("Index,KernelName\n")
        raise AssertionError("空 CSV 应报错")
    except ValueError:
        print("[ok] 空 CSV → 明确报错")
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="核对 PMC 字段在本机是否可采")
    ap.add_argument("csv", nargs="?", help="hipprof --pmc --pmc-type 3 产出的 CSV")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.csv:
        ap.error("给一个 PMC CSV 路径，或 --selftest")
    text = open(a.csv, encoding="utf-8", errors="ignore").read()
    res = probe(text)
    if a.json:
        usable, banned = verdict(res)
        print(json.dumps({"result": res, "usable": usable, "banned": banned},
                         ensure_ascii=False, indent=1))
    else:
        show(res)
    usable, _ = verdict(res)
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
