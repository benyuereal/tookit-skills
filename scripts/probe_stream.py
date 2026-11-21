#!/usr/bin/env python3
"""probe_stream — 本机流式带宽实测（只读 / copy / 纯写三种口径分开命名）。

**输出的是「实测流式峰值」，不是型号标称值。** 本机实测只读 ≈1.7 TB/s，型号名 BW1100 标称 1.1 TB/s，
差 20 个百分点以上；拿标称当分母会把「还有 1.8×」判成「到顶」。每次输出都重复这句，防止数字被摘走。
口径（分别命名，不能混）：read_peak = reduce/sum（分母 1× 字节）；copy_peak = copy 读+写（分母 2×，
另给单边 side_value）；write_peak = fill（分母 1×）。小 buffer 跑不进 HBM 稳态，数字偏低是常态，
所以逐尺寸单独报，默认 16/64/256/1024 MiB。

用法：
    python3 -I scripts/probe_stream.py --json | --check | --selftest
    --check 与 constants.yaml 对账（偏差 >2% 报警）；--selftest 无 GPU 也能跑。
依赖标准库 + 可选 torch；没有 torch/GPU 时明确说「无法探测」，不崩、不静默给默认值。
退出码：0 正常；1 对账偏差 >2%；2 一个尺寸都没测成。
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOFLINE = REPO / "Plugins" / "ops" / "skills" / "op-roofline"
DEFAULT_CONSTANTS, PEAK_TOOLS = ROOFLINE / "constants.yaml", ROOFLINE / "tools"
DEFAULT_SIZES, PHASES, TOLERANCE_PCT = (16, 64, 256, 1024), ("read", "copy", "write"), 2.0

DISCLAIMER = ("实测流式峰值（本机当场跑出来的）——不是型号标称值。标称带宽只是型号名，"
              "用它当分母会把「还有空间」判成「到顶」。")


def _torch():
    """惰性导入 torch；不可用时返回 (None, 原因)，由调用方明确降级。"""
    try:
        import torch
    except Exception as exc:
        return None, "没有可用的 torch：%s" % exc
    if not torch.cuda.is_available():
        return None, "torch 在，但 torch.cuda.is_available() = False（无可见 DCU）"
    return torch, None


def _measure(torch, fn, reps=7, warmup=3):
    """CUDA/HIP event 计时，取中位数 + MAD；不用 wall clock（异步队列只量到 launch）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    med = statistics.median(xs)
    return {"median_ms": med, "mad_ms": statistics.median([abs(x - med) for x in xs]),
            "n": len(xs)}


def payload_bytes(phase, size_bytes):
    """每种口径的分母字节数——写死在函数里，而不是注释里。"""
    return 2 * size_bytes if phase == "copy" else size_bytes


def caliber(phase):
    return {"read": "流式只读（reduce/sum），分母 = 1× 字节数",
            "copy": "copy 读+写，分母 = 2× 字节数（R+W 各一份）",
            "write": "纯写（fill），分母 = 1× 字节数"}[phase]


def probe(torch, sizes, reps, phases=PHASES):
    """逐尺寸、逐口径测量；分配在计时之外，分配完后释放再进下一个尺寸。"""
    out = {}
    for size in sizes:
        n = int(size) * 1024 * 1024 // 4
        src = torch.rand(n, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)
        fns = {"read": lambda: src.sum(), "copy": lambda: dst.copy_(src),
               "write": lambda: src.fill_(1.0)}
        nbytes = int(size) * 1024 * 1024
        rec = {}
        for phase in phases:
            s = _measure(torch, fns[phase], reps)
            gbs = payload_bytes(phase, nbytes) / (s["median_ms"] * 1e6)  # bytes/ms → GB/s
            rec[phase] = {"value": round(gbs, 1), "unit": "GB/s",
                          "caliber": caliber(phase), "size_mib": size,
                          "source": "probe_stream（本机实测，%d MiB，n=%d 取中位数）"
                                    % (size, s["n"]),
                          "median_ms": round(s["median_ms"], 4),
                          "mad_ms": round(s["mad_ms"], 4),
                          "stable": bool(s["median_ms"] and s["mad_ms"] / s["median_ms"] < .05)}
        if "copy" in rec:  # 单边值，防止「copy 比 read 慢」的误读
            rec["copy"]["side_value"] = round(rec["copy"]["value"] / 2.0, 1)
            rec["copy"]["side_note"] = "单边(=R 或 W 的一份)GB/s，与 read_peak 同口径可比"
        out[size] = rec
        del src, dst, fns
        torch.cuda.empty_cache()
    return out


def _load_constants(path):
    """复用 op-roofline/tools/probe_peak.py 的极简 YAML 读取（同一套规则只有一份实现）。"""
    mod_path = PEAK_TOOLS / "probe_peak.py"
    if not mod_path.is_file():
        return {}, "找不到 %s" % mod_path
    try:
        spec = importlib.util.spec_from_file_location("_probe_peak_mod", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_constants(path), None
    except Exception as exc:                  # 读不到就降级为「对账不可用」，不崩
        return {}, "constants 读取失败：%s" % exc


def check(results, consts, tol=TOLERANCE_PCT):
    """与 constants.yaml 对账。峰值口径 = 各尺寸最大值（与 constants「多次峰值」一致），
    但每个尺寸的值仍留在人读表里，引用时看得见它量在哪个尺寸上。"""
    items = []
    for phase, key, label in (("read", "memory.stream_read_peak.value", "stream_read_peak"),
                              ("copy", "memory.stream_copy_peak.value", "stream_copy_peak")):
        probed, at = None, None
        for size, rec in results.items():
            v = rec.get(phase, {}).get("value")
            if v is not None and (probed is None or v > probed):
                probed, at = v, size
        hand = consts.get(key)
        try:
            hand = float(hand) if hand not in (None, "") else None
        except ValueError:
            hand = None
        item = {"phase": phase, "key": label, "handwritten": hand, "probed": probed,
                "probed_size_mib": at, "unit": "GB/s", "deviation_pct": None,
                "status": "skip", "note": None}
        if hand is None and probed is None:
            item["note"] = "两边都没有值"
        elif hand is None:
            item["note"] = "constants.yaml 缺这一项（本机实测有值，可考虑补进去）"
        elif probed is None:
            item["note"] = "无法探测，退回手抄值 %s GB/s（未校验）" % hand
        else:
            dev = (probed - hand) / hand * 100.0
            item["deviation_pct"] = round(dev, 2)
            item["status"] = "warn" if abs(dev) > tol else "ok"
        if phase == "copy" and item["status"] != "skip":
            item["note"] = "copy 口径是 R+W（2× 字节）；constants 的 stream_copy_peak 同口径"
        items.append(item)
    return items


def print_human(results, checks, consts_path, err=None):
    print("== 流式带宽实测 ==\n  %s" % DISCLAIMER)
    if err:
        print("\n  ✗ %s\n  ⇒ 无法探测：不输出任何默认值（静默给默认值比不给更危险）。" % err)
        return
    print()
    for size in sorted(results):
        print("  -- %d MiB --" % size)
        for phase in PHASES:
            e = results[size].get(phase)
            if not e:
                continue
            flag = "" if e["stable"] else "  ⚠ 抖动 >5%（卡上可能有别的负载）"
            extra = ("（单边 %s GB/s）" % e["side_value"]
                     if phase == "copy" and "side_value" in e else "")
            print("    %-6s %8.1f GB/s  median=%.4f ms  MAD=%.4f ms%s%s"
                  % (phase, e["value"], e["median_ms"], e["mad_ms"], extra, flag))
    if not checks:
        return
    print("\n== 与 constants.yaml 对账（阈值 %.0f%%，%s）==" % (TOLERANCE_PCT, consts_path))
    for c in checks:
        if c["status"] == "skip":
            print("  ⚠ %-18s %s" % (c["key"], c["note"]))
            continue
        mark = "⚠" if c["status"] == "warn" else "✅"
        rel = "≠" if c["status"] == "warn" else "≈"
        print("  %s %-18s 手抄 %s GB/s %s 实测 %s GB/s（@%s MiB，偏差 %+.1f%%）"
              % (mark, c["key"], c["handwritten"], rel, c["probed"],
                 c["probed_size_mib"], c["deviation_pct"]))
    print("\n注意：小 buffer 跑不进 HBM 稳态，数字偏低是常态；引用峰值请写明尺寸与口径。")
    if any(c["status"] == "warn" for c in checks):
        print("⇒ 手抄常数与实测不符：先确认口径（尺寸/锁频/卡上负载），再改 constants.yaml。")


def selftest():
    """不碰 GPU：验口径算术、对账逻辑、降级路径。"""
    assert payload_bytes("read", 100) == payload_bytes("write", 100) == 100
    assert payload_bytes("copy", 100) == 200
    print("[ok] payload_bytes  copy 按 R+W 计 2×")
    fake = {256: {"read": {"value": 1700.0}, "copy": {"value": 1340.0}},
            1024: {"read": {"value": 1750.0}, "copy": {"value": 1355.0}}}
    consts = {"memory.stream_read_peak.value": "1724",
              "memory.stream_copy_peak.value": "1350"}
    checks = check(fake, consts)
    assert checks[0]["probed"] == 1750.0 and checks[0]["probed_size_mib"] == 1024, checks[0]
    assert checks[0]["status"] == "ok" and checks[1]["status"] == "ok", checks
    print("[ok] 对账取各尺寸最大值（read 1750 @1024 MiB，偏差 %+.2f%%）"
          % checks[0]["deviation_pct"])
    consts["memory.stream_read_peak.value"] = "1100"       # 标称值当分母的反例
    bad = check(fake, consts)[0]
    assert bad["status"] == "warn" and bad["deviation_pct"] > 50, bad
    print("[ok] 标称值当分母会被抓出来（偏差 %+.1f%%）" % bad["deviation_pct"])
    t, err = _torch()
    print("[ok] 环境探测：%s" % ("torch + GPU 可用" if t else "无法探测（%s）" % err))
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="本机流式带宽实测（read/copy/write 三种口径分开）",
                                 epilog=DISCLAIMER)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help="buffer 尺寸（MiB），默认 16 64 256 1024")
    ap.add_argument("--reps", type=int, default=7, help="每个口径的计时轮数（取中位数）")
    ap.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    ap.add_argument("--check", action="store_true", help="与 constants.yaml 对账")
    ap.add_argument("--constants", default=str(DEFAULT_CONSTANTS))
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--selftest", action="store_true", help="无 GPU 的逻辑自检")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    torch, err = _torch()
    results, checks = {}, []
    if torch is not None:
        try:
            results = probe(torch, a.sizes, a.reps, a.phases)
        except Exception as exc:                 # 显存不足等：降级而不是抛栈
            err, results = "探测中断：%s" % exc, {}
    if a.check:
        consts, cerr = _load_constants(a.constants)
        err = (err + "；" if err and cerr else "") + cerr if cerr else err
        checks = check(results, consts) if results else []
    code = 1 if any(c["status"] == "warn" for c in checks) else (2 if err or not results else 0)
    if a.json:
        print(json.dumps({"disclaimer": DISCLAIMER, "sizes_mib": list(a.sizes),
                          "calibers": {p: caliber(p) for p in a.phases},
                          "results": results, "checks": checks,
                          "constants_path": a.constants if a.check else None,
                          "tolerance_pct": TOLERANCE_PCT, "error": err,
                          "ok": code == 0, "exit_code": code}, ensure_ascii=False,
                         indent=2))
        return code
    print_human(results, checks, a.constants, err)
    return code


if __name__ == "__main__":
    sys.exit(main())
