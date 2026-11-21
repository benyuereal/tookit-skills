#!/usr/bin/env python3
"""probe_stream — 本机流式带宽实测探针（只读 / copy / 纯写，三种口径分开命名）。

为什么需要它
------------
`op-roofline/constants.yaml` 里的 `stream_read_peak` / `stream_copy_peak` 是手抄的实测值，
会随机器、驱动、锁频状态漂移。本工具在**本机当场跑**，给一个带口径的实测分母；
`--check` 再与 constants.yaml 对账。

**最重要的口径：输出的是「实测流式峰值」，不是型号标称值。**
本机实测流式只读 ≈ 1.7 TB/s，而型号名 BW1100 的标称是 1.1 TB/s —— 两者差 20 个百分点以上。
拿标称值当分母，会把「还有 1.8× 空间」判成「已经到顶」。本工具只输出实测值，
每次输出都打印这一条，防止数字被摘走之后失去上下文。

三种口径（分别命名，不能混）
----------------------------
  read_peak   ：大 buffer 上做 reduce/sum（只读）。分母 = 1× 字节数。
  copy_peak   ：copy(src→dst)（读 + 写）。分母 = 2× 字节数（R+W 各一份）；
                同时给出单边值 copy_side，避免「copy 1350 比 read 1724 慢」这种误读。
  write_peak  ：fill（纯写，不读）。分母 = 1× 字节数。

尺寸的意义
----------
同一口径在不同 buffer 尺寸下数字不同：小 buffer（≤ L2 的几倍）跑进 L2、且单次耗时太短，
计时被 launch/事件开销吃掉，**不能当 HBM 峰值**。因此默认扫 16/64/256/1024 MiB，
逐尺寸单独报；`--check` 只拿各尺寸中的最大值与 constants 对账（峰值 = 可复现的最好值），
但人读表里每个尺寸都保留，谁想引用都能看见它是在哪个尺寸上量的。

用法
----
    python3 -I scripts/probe_stream.py                      # 人读表格
    python3 -I scripts/probe_stream.py --json               # 机器可读
    python3 -I scripts/probe_stream.py --check              # 与 constants.yaml 对账（偏差 >2% 报警）
    python3 -I scripts/probe_stream.py --sizes 64 256 1024 --reps 9
    python3 -I scripts/probe_stream.py --selftest           # 无 GPU 也能跑的逻辑自检

依赖：标准库 + 可选 torch。没有 torch / 没有 GPU 时**明确说「无法探测」**，
不崩、不静默给一个默认值。退出码：0 = 正常（或已明确说明无法探测）；
1 = 对账偏差 > 2%；2 = 一个尺寸都没测成。
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_CONSTANTS = REPO / "Plugins" / "ops" / "skills" / "op-roofline" / "constants.yaml"
PEAK_TOOLS = REPO / "Plugins" / "ops" / "skills" / "op-roofline" / "tools"
DEFAULT_SIZES = (16, 64, 256, 1024)
PHASES = ("read", "copy", "write")
TOLERANCE_PCT = 2.0

DISCLAIMER = ("实测流式峰值（本机当场跑出来的）——不是型号标称值。"
              "标称带宽只是型号名，用它当分母会把「还有空间」判成「到顶」。")


# ---------------------------------------------------------------- 环境

def _torch():
    """惰性导入 torch；不可用时返回 (None, 原因)，由调用方明确降级。"""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # ImportError / 链接失败都算「无法探测」
        return None, "没有可用的 torch：%s" % exc
    if not torch.cuda.is_available():
        return None, "torch 在，但 torch.cuda.is_available() = False（无可见 DCU）"
    return torch, None


def _event_timer(torch):
    """CUDA/HIP event 计时；不用 wall clock（异步队列只量到 launch）。"""
    def _one(fn, iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def measure(fn, reps=7, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        xs = [_one(fn, 1) for _ in range(reps)]
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        return {"median_ms": med, "mad_ms": mad, "n": len(xs),
                "min_ms": min(xs), "max_ms": max(xs)}
    return measure


# ---------------------------------------------------------------- 口径

def payload_bytes(phase, size_bytes):
    """每种口径的「分母字节数」——一个数一个分母，写死在函数里而不是注释里。"""
    if phase in ("read", "write"):
        return size_bytes
    if phase == "copy":
        return 2 * size_bytes          # 读 1× + 写 1×
    raise ValueError(phase)


def caliber(phase):
    return {
        "read": "流式只读（reduce/sum），分母 = 1× 字节数",
        "copy": "copy 读+写，分母 = 2× 字节数（R+W 各一份）",
        "write": "纯写（fill），分母 = 1× 字节数",
    }[phase]


def make_fns(torch, size_mib):
    """为每个尺寸建 src/dst 和三种口径的闭包。分配不计入计时（分配在计时之外）。"""
    n = int(size_mib) * 1024 * 1024 // 4
    src = torch.rand(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    return {
        "read": lambda: src.sum(),
        "copy": lambda: dst.copy_(src),
        "write": lambda: src.fill_(1.0),
    }


def probe(torch, sizes, reps, phases=PHASES):
    """逐尺寸、逐口径测量，返回 {size_mib: {phase: {...}}}。"""
    measure = _event_timer(torch)
    out = {}
    for size in sizes:
        fns = make_fns(torch, size)
        nbytes = int(size) * 1024 * 1024
        rec = {}
        for phase in phases:
            s = measure(fns[phase], reps=reps)
            gbs = payload_bytes(phase, nbytes) / (s["median_ms"] * 1e6)  # bytes/ms → GB/s
            rec[phase] = {
                "value": round(gbs, 1), "unit": "GB/s",
                "caliber": caliber(phase), "size_mib": size,
                "source": "probe_stream（本机实测，%d MiB buffer，n=%d 取中位数）"
                          % (size, s["n"]),
                "median_ms": round(s["median_ms"], 4),
                "mad_ms": round(s["mad_ms"], 4),
                "stable": (s["mad_ms"] / s["median_ms"] < 0.05) if s["median_ms"] else False,
            }
        if "copy" in rec:  # 单边值，防止「copy 比 read 慢」的误读
            rec["copy"]["side_value"] = round(rec["copy"]["value"] / 2.0, 1)
            rec["copy"]["side_note"] = "单边(=R 或 W 的一份)GB/s；与 read_peak 同口径可比"
        out[size] = rec
        del fns
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- 对账

def _load_constants(path):
    """复用 op-roofline/tools/probe_peak.py 的极简 YAML 读取（同一套规则只有一个权威点）。"""
    mod_path = PEAK_TOOLS / "probe_peak.py"
    if not mod_path.is_file():
        return {}, "找不到 %s" % mod_path
    try:
        spec = importlib.util.spec_from_file_location("_probe_peak_mod", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_constants(path), None
    except Exception as exc:
        return {}, "constants 读取失败：%s" % exc


def check(results, consts, tol=TOLERANCE_PCT):
    """与 constants.yaml 的 stream_read_peak / stream_copy_peak 对账。

    峰值口径 = 各尺寸中的最大值（与 constants 里「多次峰值」的定义一致）；
    但每个尺寸的值都还在人读表里，引用时能看见它量在哪个尺寸上。
    """
    items = []
    pairs = (("read", "memory.stream_read_peak.value", "stream_read_peak"),
             ("copy", "memory.stream_copy_peak.value", "stream_copy_peak"))
    for phase, key, label in pairs:
        probed, at = None, None
        for size, rec in results.items():
            v = rec.get(phase, {}).get("value")
            if v is not None and (probed is None or v > probed):
                probed, at = v, size
        hand = consts.get(key)
        hand = float(hand) if hand not in (None, "") else None
        item = {"phase": phase, "key": label, "handwritten": hand,
                "probed": probed, "probed_size_mib": at, "unit": "GB/s",
                "deviation_pct": None, "status": "skip", "note": None}
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


# ---------------------------------------------------------------- 输出

def print_human(results, checks, consts_path, err=None):
    print("== 流式带宽实测 ==")
    print("  %s" % DISCLAIMER)
    if err:
        print("\n  ✗ %s" % err)
        print("  ⇒ 无法探测：不输出任何默认值（静默给默认值比不给更危险）。")
        return
    print()
    for size in sorted(results):
        rec = results[size]
        print("  -- %d MiB --" % size)
        for phase in PHASES:
            e = rec.get(phase)
            if not e:
                continue
            flag = "" if e["stable"] else "  ⚠ 抖动 >5%%（卡上可能有别的负载）"
            extra = ("（单边 %s GB/s）" % e["side_value"]
                     if phase == "copy" and "side_value" in e else "")
            print("    %-6s %8.1f GB/s  median=%.4f ms  MAD=%.4f ms%s%s"
                  % (phase, e["value"], e["median_ms"], e["mad_ms"], extra, flag))
    if not checks:
        return
    print("\n== 与 constants.yaml 对账（阈值 %.0f%%，%s）==" % (TOLERANCE_PCT, consts_path))
    for c in checks:
        if c["status"] == "warn":
            print("  ⚠ %-18s 手抄 %s GB/s ≠ 实测 %s GB/s（@%s MiB，偏差 %+.1f%%）"
                  % (c["key"], c["handwritten"], c["probed"],
                     c["probed_size_mib"], c["deviation_pct"]))
        elif c["status"] == "ok":
            print("  ✅ %-18s 手抄 %s GB/s ≈ 实测 %s GB/s（@%s MiB，偏差 %+.1f%%）"
                  % (c["key"], c["handwritten"], c["probed"],
                     c["probed_size_mib"], c["deviation_pct"]))
        else:
            print("  ⚠ %-18s %s" % (c["key"], c["note"]))
    print("\n注意：小 buffer（如 16 MiB）跑不进 HBM 稳态，数字偏低是常态；"
          "引用峰值请写明尺寸与口径。")
    if any(c["status"] == "warn" for c in checks):
        print("⇒ 手抄常数与实测不符：先确认口径（尺寸/是否锁频/卡上是否有负载），再改 constants.yaml。")


# ---------------------------------------------------------------- 自检

def selftest():
    """不碰 GPU：验口径算术、对账逻辑、降级路径。"""
    ok = True
    assert payload_bytes("read", 100) == 100
    assert payload_bytes("write", 100) == 100
    assert payload_bytes("copy", 100) == 200
    print("[ok] payload_bytes  copy 按 R+W 计 2×")

    fake = {256: {"read": {"value": 1700.0}, "copy": {"value": 1340.0}},
            1024: {"read": {"value": 1750.0}, "copy": {"value": 1355.0}}}
    consts = {"memory.stream_read_peak.value": "1724",
              "memory.stream_copy_peak.value": "1350"}
    checks = check(fake, consts)
    assert checks[0]["probed"] == 1750.0 and checks[0]["probed_size_mib"] == 1024, checks[0]
    assert checks[0]["status"] == "ok", checks[0]
    assert checks[1]["status"] == "ok", checks[1]
    print("[ok] 对账取各尺寸最大值（read 1750 @1024 MiB，偏差 %+.2f%%）"
          % checks[0]["deviation_pct"])

    consts["memory.stream_read_peak.value"] = "1100"   # 标称值当分母的反例
    bad = check(fake, consts)[0]
    assert bad["status"] == "warn" and bad["deviation_pct"] > 50, bad
    print("[ok] 标称值当分母会被抓出来（偏差 %+.1f%%）" % bad["deviation_pct"])

    t, err = _torch()
    print("[ok] 环境探测：%s" % ("torch + GPU 可用" if t else "无法探测（%s）" % err))
    print("\nselftest %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------- 入口

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="本机流式带宽实测（read/copy/write 三种口径分开）",
        epilog=DISCLAIMER)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help="buffer 尺寸（MiB），默认 16 64 256 1024")
    ap.add_argument("--reps", type=int, default=7, help="每个口径的计时轮数（取中位数）")
    ap.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES),
                    help="只测指定口径")
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
        except Exception as exc:          # 显存不足等：降级而不是抛栈
            err = "探测中断：%s" % exc
            results = {}
    if a.check:
        consts, cerr = _load_constants(a.constants)
        if cerr:
            err = (err + "；" if err else "") + cerr
        checks = check(results, consts) if results else []

    if any(c["status"] == "warn" for c in checks):
        code = 1
    elif err or not results:
        code = 2
    else:
        code = 0

    if a.json:
        payload = {
            "disclaimer": DISCLAIMER,
            "sizes_mib": list(a.sizes),
            "calibers": {p: caliber(p) for p in a.phases},
            "results": results,
            "checks": checks,
            "constants_path": a.constants if a.check else None,
            "tolerance_pct": TOLERANCE_PCT,
            "error": err,
            "ok": code == 0,
            "exit_code": code,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return code

    print_human(results, checks, a.constants, err)
    return code


if __name__ == "__main__":
    sys.exit(main())
