#!/usr/bin/env python3
"""pmc — PMC CSV 的可执行解析器（唯一权威实现；`op-profiling/references/pmc_reference.py.md` 只是它的说明页）。

它只负责**选对行、排掉噪声、按实例求和、打印原值**；不解释、不判定。
判读口径与三级证据纪律见 `Plugins/ops/skills/op-profiling/SKILL.md` §四/§五/§六。

四条硬规则（每一条都对应一次真实读错，见 §五）：
  1. 稳态行 = 单次 launch 中 **GRBM 最小** 的行；PMC 下首次 launch 的 GRBM 虚高 ~40%，绝不取首行。
  2. 自动选主 kernel 先排噪声前缀（`_ZN2at` / `_ZN7rocprim` / `_ZN12_GLOBAL__N_1` / `_ZN2c10` /
     `_ZN3c10`），再按**单次** GRBM 取最大 —— 否则 `randperm` 内部 kernel 会夺冠。
  3. `TA_*` / `TCP_*` / `TCC_*` 的 `[n]` 是按 CU 编号跨 SE 的实例，只能**求和**；
     `max(TA_TA_BUSY)/GRBM` 会得出 352% 的荒谬值，逐编号均值/比大小也无意义（CU 编号只有 0..8 存在）。
  4. CU 数、每 MMA 的 FLOPs 等硬件常数**不许写死默认值**：必须显式传入，否则打印 `unknown(<原因>)`。

输出只给原值；**不下结论**（没有 bound/ratios/facts 字段）。MBU/MFU、非MFMA VALU/MFMA
要不要算、拿什么当分母，由写报告的人自己定。

用法：
    python3 -I scripts/pmc.py /tmp/pmcout/x.csv                      # 自动选主 kernel
    python3 -I scripts/pmc.py x.csv --kernel 'main_kernel' --cu 64   # 显式指定 + 给 CU 数
    python3 -I scripts/pmc.py x.csv --row max-dur --json             # 取最慢行 / 机器可读
    python3 -I scripts/pmc.py --selftest                             # 无 CSV、无 GPU 的自检

退出码：0 = 正常输出；1 = 输入不可解析（找不到数据行 / 指定 kernel 不存在）；2 = 用法错误。
"""

import argparse
import csv
import io
import json
import re
import sys

NOISE = ("_ZN2at", "_ZN7rocprim", "_ZN12_GLOBAL__N_1", "_ZN2c10", "_ZN3c10")
COUNTERS = ("GRBM_COUNT", "SQ_INSTS_VALU", "SQ_ACTIVE_INST_VALU", "SQ_INSTS_LDS",
            "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_LDS_BANK_CONFLICT",
            "SQ_WAIT_INST_LDS", "TA_TA_BUSY", "TCP_TCP_TA_DATA_STALL_CYCLES",
            "TA_FLAT_READ_WAVEFRONTS", "TCC_HIT", "TCC_MISS")
NOTE = {"SQ_INSTS_VALU": "含 MFMA，未拆分（非MFMA = 本值 − FLOPs/8192）",
        "TA_TA_BUSY": "跨 SE 实例求和",
        "TCP_TCP_TA_DATA_STALL_CYCLES": "跨 SE 实例求和",
        "TA_FLAT_READ_WAVEFRONTS": "只统计 flat 寻址；常规 global load 不计",
        "TCC_HIT": "DRAM 字节按 miss 项 × 128 B 折算，hit 不是字节",
        "TCC_MISS": "miss 项 × 128 B = DRAM 读字节（颗粒粒度是推导值，需显式确认）"}


def f(v):
    """CSV 单元格 → float；空/坏值 → None（缺就是缺，不当 0）。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def base(name):
    """去掉 kernel 名的模板参数后缀：`foo<S=128>` → `foo`。"""
    return re.sub(r"\(.*", "", name or "")


def total(row, key):
    """KEY 或 KEY[0..N]：有实例列就跨实例求和（TA_*/TCP_*/TCC_* 都是这种）。"""
    if key in row:
        return f(row[key])
    vals, i = [], 0
    while "%s[%d]" % (key, i) in row:
        vals.append(f(row["%s[%d]" % (key, i)]) or 0.0)
        i += 1
    return sum(vals) if vals else None


def read_rows(text):
    """读 CSV，只保留有 GRBM_COUNT 的数据行（表头/尾注行被剃掉）。"""
    return [r for r in csv.DictReader(io.StringIO(text)) if r.get("GRBM_COUNT")]


def pick_kernel(rows, kernel=None):
    """返回 (kernel, 同 kernel 的数据行)。不给 kernel 时先排噪声，再按单次 GRBM 取最大。"""
    if kernel:
        same = [r for r in rows if base(r["KernelName"]) == base(kernel)]
        return (same[0]["KernelName"] if same else None), same
    cand = [r for r in rows if not base(r["KernelName"]).startswith(NOISE)]
    if not cand:
        return None, []
    top = max(cand, key=lambda r: f(r["GRBM_COUNT"]) or 0.0)
    return top["KernelName"], [r for r in rows if base(r["KernelName"]) == base(top["KernelName"])]


def pick_row(same, mode="steady"):
    """steady = 单次 GRBM 最小（首行虚高）；max-dur = 墙钟最长（只查长尾时用）。"""
    if mode == "max-dur":
        key = lambda r: (f(r.get("EndNs")) or 0) - (f(r.get("BeginNs")) or 0)   # noqa: E731
    else:
        key = lambda r: f(r.get("GRBM_COUNT")) or 0.0                            # noqa: E731
    return min(same, key=key)


def parse(text, kernel=None, row_mode="steady", cu=None, counters=COUNTERS):
    """解析一份 PMC CSV，返回只含原值的 dict。cu 不给就是 unknown（不许写死）。"""
    rows = read_rows(text)
    if not rows:
        return {"error": "CSV 里没有带 GRBM_COUNT 的数据行"}
    kernel, same = pick_kernel(rows, kernel)
    if not same:
        known = sorted({base(r["KernelName"]) for r in rows})
        return {"error": "找不到 kernel %r；CSV 里有 %d 个：%s"
                         % (kernel, len(known), ", ".join(known[:8]))}
    row = pick_row(same, row_mode)
    dur_ns = (f(row.get("EndNs")) or 0) - (f(row.get("BeginNs")) or 0)
    noise = sorted({base(r["KernelName"]) for r in rows
                    if base(r["KernelName"]).startswith(NOISE)})
    values = {}
    for k in counters:
        v = total(row, k)
        values[k] = {"value": v, "note": NOTE.get(k, ""),
                     "missing": v is None}
    geom = {k: row.get(k) for k in ("grd", "wgr", "wave_size", "lds", "arch_vgpr",
                                    "sgpr", "scr") if k in row}
    return {
        "kernel": kernel,
        "launches": len(same),
        "row_index": row.get("Index"),
        "row_mode": row_mode,
        "wall_ns": dur_ns,
        "wall_note": "PMC 采集墙钟，不是 bench 值；计时口径走 op-bench",
        "cu": cu if cu is not None else "unknown(未给 --cu；CU 数每卡不同，不写死)",
        "geometry": geom,
        "excluded_noise": noise,
        "counters": values,
        "counters_note": "原值；跨实例已求和。不在此算 MBU/MFU 或任何比值。",
    }


def print_human(res):
    if "error" in res:
        print("error: %s" % res["error"])
        return
    print("kernel=%s launches=%d row_index=%s wall_ns=%.0f cu=%s"
          % (res["kernel"], res["launches"], res["row_index"], res["wall_ns"], res["cu"]))
    print("geometry:", res["geometry"])
    print("excluded_noise=%d" % len(res["excluded_noise"]))
    for k, e in res["counters"].items():
        shown = "unknown(列不存在)" if e["missing"] else format(e["value"], ",.0f")
        print("%-32s %s  %s" % (k, shown, e["note"]))
    print("note: PMC 墙钟 ≠ bench 耗时；CU 数须显式给，# 缺了就写 unknown。")


SAMPLE = """Index,KernelName,GRBM_COUNT,BeginNs,EndNs,grd,wgr,
0,_ZN2at6native29normal_and_transformEii,500,0,1000,,
1,_ZN7rocprim7detail12radix_sortE,900,0,2000,,
2,main_kernel<S=128>,1400000,0,9000,512,256,
3,main_kernel<S=128>,1000000,0,12000,512,256,
4,main_kernel<S=128>,1010000,0,11000,512,256,
"""


def selftest():
    """用内联 CSV 验四条硬规则；不需要真 CSV、不需要 GPU。"""
    res = parse(SAMPLE, cu=None)
    assert res["kernel"].startswith("main_kernel"), res["kernel"]
    assert res["launches"] == 3, res["launches"]
    assert res["row_index"] == "3", res["row_index"]          # GRBM 最小行，不是首行
    print("[ok] 选行：排掉 at/rocprim 噪声，稳态取 GRBM 最小行（index=%s）" % res["row_index"])

    res2 = parse(SAMPLE, cu=64)
    assert res2["cu"] == 64
    res3 = parse(SAMPLE)                                        # 不给 CU
    assert isinstance(res3["cu"], str) and res3["cu"].startswith("unknown("), res3["cu"]
    print("[ok] CU 数：显式传入才显示数；不给则 unknown(原因)，不写默认值")

    row = {"TA_TA_BUSY[0]": "10", "TA_TA_BUSY[1]": "7", "TA_TA_BUSY[2]": "3"}
    assert total(row, "TA_TA_BUSY") == 20.0
    print("[ok] TA_* 跨实例求和：10+7+3=20（取 max 会得 10，逐编号比大小无意义）")

    assert total({"X": ""}, "X") is None and total({}, "X") is None
    print("[ok] 缺列/空值 → None，不当 0")

    bad = parse("Index,KernelName\n0,foo\n")
    assert "error" in bad and "GRBM_COUNT" in bad["error"]
    miss = parse(SAMPLE, kernel="nope")
    assert "error" in miss
    print("[ok] 输入不可解析时明确报错（不静默输出空表）")
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PMC CSV 解析器：只打印机械原值，不下结论")
    ap.add_argument("csv", nargs="?", help="hipprof --pmc 产出的 CSV 路径")
    ap.add_argument("--kernel", help="指定 kernel（缺省：排噪声后按单次 GRBM 取最大）")
    ap.add_argument("--row", choices=("steady", "max-dur"), default="steady",
                    help="steady=单次 GRBM 最小（默认）；max-dur=墙钟最长（只查长尾）")
    ap.add_argument("--cu", type=int, help="真实 CU 数；不给则打印 unknown(原因)")
    ap.add_argument("--counter", action="append", dest="counters",
                    help="额外/替换要看的计数器（可重复）；缺省用内置清单")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--selftest", action="store_true", help="用内联 CSV 自检")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.csv:
        ap.print_help()
        return 2
    try:
        text = open(a.csv, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        print("error: 读不到 %s：%s" % (a.csv, exc))
        return 1
    counters = COUNTERS + tuple(c for c in (a.counters or []) if c not in COUNTERS)
    res = parse(text, a.kernel, a.row, a.cu, counters)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_human(res)
    return 1 if "error" in res else 0


if __name__ == "__main__":
    sys.exit(main())
