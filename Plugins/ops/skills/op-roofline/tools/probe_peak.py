#!/usr/bin/env python3
"""probe_peak.py —— 从本机反推峰值常数，替代手抄的 constants.yaml。

为什么需要它
------------
constants.yaml 是手抄的，会漂。它自己就记着一个坑：同一份 955 GB/s 的实测数据，
拿「型号标称 1100」做分母是 87%（到顶了），拿「本机实测 1724」做分母是 55%（还有 1.8×）。
**分母换了，结论就反了。** 所以峰值常数必须能由机器反推，并且每个数都带口径。

口径（输出里逐项标明）
----------------------
  * 标称：型号/规格书给的值（如 BW1100 的 1100 GB/s）；
  * 实测：跑负载量出来的值（常量里的 stream_read_peak 1724）；
  * 推导：由探测到的硬件参数按公式算出（CU × SIMD × wave × 2 × sclk）；
  * 频率档：sclk 是计算时钟、mclk 是显存时钟；锁频 / 不锁频（DPM 档位会变）必须写清。

只依赖标准库；外部命令 `rocminfo` / `hy-smi` 缺失、超时或无权限时**优雅降级**：
明确输出「无法探测，退回手抄值」，不崩、不静默取一个。

用法
----
    python3 -I probe_peak.py                          # 人读表格
    python3 -I probe_peak.py --check                  # 与 constants.yaml 对账（偏差 >2% 报警）
    python3 -I probe_peak.py --json                   # 机器可读（含 value/unit/source/basis）
    python3 -I probe_peak.py --check --json
    python3 -I probe_peak.py --check --bus-width 4096 --channels 2 --sclk-mhz 1150
    python3 -I probe_peak.py --rocminfo-output /tmp/rocminfo.txt --hysmi-output /tmp/hysmi_c.txt

退出码：0 = 全部一致（或已明确说明无法探测）；1 = 有偏差 >2%；2 = 一个常数都没能探测。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONSTANTS = HERE.parent / "constants.yaml"
TOLERANCE_PCT = 2.0


# ---------------------------------------------------------------- 常量文件读取


def _strip_comment(line):
    """去掉 # 注释（引号内的不算）。"""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)


def load_constants(path):
    """极简 YAML 子集读取：只认 key: value 的缩进映射，返回 dotted-path → 标量字符串。

    列表项（- id: ...）、块标量（note: >）都跳过；够读 constants.yaml 的数值条目，
    不引入 pyyaml 依赖。
    """
    out, stack, block_indent = {}, [], None
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for raw in lines:
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        s = line.strip()
        if s.startswith("- "):
            continue
        key, sep, val = s.partition(":")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if val in (">", "|", ">-", "|-", ">+", "|+"):
            block_indent = indent
            continue
        if not val:
            stack.append((indent, key))
            continue
        out[".".join([k for _, k in stack] + [key])] = val.strip('"').strip("'")
    return out


def const_num(consts, path, cast=float):
    v = consts.get(path)
    if v is None:
        return None
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 命令探测


def run_command(argv, timeout):
    """跑外部命令；缺失/无权限/超时都返回 None（调用方据此降级）。"""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (p.stdout or "") + (p.stderr or "")
    return text if text.strip() else None


def parse_rocminfo(text):
    """解析 rocminfo：每个 gfx agent 的 CU / wave / SIMD / L2 / LDS，以及通用内存字段。"""
    info = {
        "available": text is not None,
        "gfx": None, "cu": [], "wave": None, "simd_per_cu": None,
        "l2_mb": None, "lds_kb": None,
        "mem_clock_mhz": None, "bus_width_bits": None, "channels": None,
    }
    if not text:
        return info
    for block in re.split(r"\n\s*Agent\s+\d+\s*\n", text):
        m = re.search(r"^\s*Name:\s*(\S+)", block, re.M)
        if not m or not m.group(1).startswith("gfx"):
            continue
        info["gfx"] = info["gfx"] or m.group(1)
        cu = re.search(r"Compute Unit:\s*(\d+)", block)
        if cu:
            info["cu"].append(int(cu.group(1)))
        w = re.search(r"Wavefront Size:\s*(\d+)", block)
        if w and info["wave"] is None:
            info["wave"] = int(w.group(1))
        simd = re.search(r"SIMDs per CU:\s*(\d+)", block)
        if simd and info["simd_per_cu"] is None:
            info["simd_per_cu"] = int(simd.group(1))
        l2 = re.search(r"L2:\s*(\d+)(?:\(0x[0-9a-fA-F]+\))?\s*KB", block)
        if l2 and info["l2_mb"] is None:
            info["l2_mb"] = int(l2.group(1)) / 1024.0
        for pm in re.finditer(
                r"Segment:\s*GROUP\b.*?Size:\s*(\d+)(?:\(0x[0-9a-fA-F]+\))?\s*KB",
                block, re.S):
            if info["lds_kb"] is None:
                info["lds_kb"] = int(pm.group(1))
    # 通用字段：不同版本 rocminfo / hy-smi 可能提供
    mc = re.search(r"(?:Memory Clock|Mem Clock|mclk)\s*[:=]\s*([0-9.]+)\s*(GHz|MHz)?",
                   text, re.I)
    if mc:
        val, unit = float(mc.group(1)), (mc.group(2) or "MHz").lower()
        info["mem_clock_mhz"] = val * 1000.0 if unit == "ghz" else val
    bw = re.search(r"(?:Memory Bus Width|Bus Width)\s*[:=]\s*(\d+)\s*(?:bits?|bit)?",
                   text, re.I)
    if bw:
        info["bus_width_bits"] = int(bw.group(1))
    ch = re.search(r"(?:Channels|Channel Count)\s*[:=]\s*(\d+)", text, re.I)
    if ch:
        info["channels"] = int(ch.group(1))
    return info


def parse_supported_sclk(text):
    """从 hy-smi -s 的 “Supported sclk frequencies” 段取档位列表（含 *: 1600Mhz 这类 max 行）。"""
    vals, section = [], False
    for line in (text or "").splitlines():
        if "Supported sclk frequencies" in line:
            section = True
            continue
        if "Supported " in line:
            section = False
            continue
        if not section:
            continue
        m = re.search(r"(\d+)\s*Mhz", line, re.I)
        if m:
            vals.append(int(m.group(1)))
    return sorted(set(vals))


def parse_hysmi(text):
    """解析 hy-smi -c / --showcunum 输出：逐卡 mclk / sclk / CU，以及通用位宽字段。"""
    info = {"available": text is not None, "mclk_mhz": [], "sclk_mhz": [],
            "sclk_max_mhz": None, "cu": [],
            "bus_width_bits": None, "channels": None}
    if not text:
        return info
    info["mclk_mhz"] = [float(x) for x in
                        re.findall(r"mclk clock level:\s*\d+\s*\((\d+)\s*Mhz\)",
                                   text, re.I)]
    info["sclk_mhz"] = [float(x) for x in
                        re.findall(r"sclk clock level:\s*\d+\s*\((\d+)\s*Mhz\)",
                                   text, re.I)]
    info["cu"] = [int(x) for x in re.findall(r"CU number:\s*(\d+)", text, re.I)]
    bw = re.search(r"(?:Memory Bus Width|Bus Width)\s*[:=]\s*(\d+)\s*(?:bits?|bit)?",
                   text, re.I)
    if bw:
        info["bus_width_bits"] = int(bw.group(1))
    ch = re.search(r"(?:Channels|Channel Count)\s*[:=]\s*(\d+)", text, re.I)
    if ch:
        info["channels"] = int(ch.group(1))
    supported = parse_supported_sclk(text)
    info["sclk_supported_mhz"] = supported
    info["sclk_max_mhz"] = supported[-1] if supported else None
    return info


# ---------------------------------------------------------------- 推导


def _entry(value, unit, source, basis, caliber=None, **extra):
    d = {"value": value, "unit": unit, "source": source, "basis": basis}
    if caliber:
        d["caliber"] = caliber
    d.update(extra)
    return d


def _nearest_dev(hand, lo, hi):
    if hand is None:
        return None
    anchor = hand if lo is None else min(max(hand, lo), hi)
    if anchor == 0:
        return 0.0
    return (hand - anchor) / anchor * 100.0


def build_values(args, roc, hys, consts):
    vals = {}

    # --- CU 数：优先 rocminfo，其次 hy-smi，最后 --cu
    cu_list = list(roc["cu"]) or list(hys["cu"])
    if args.cu:
        cu_list = [args.cu]
    if cu_list:
        vals["cu_count"] = _entry(
            min(cu_list), "CU", "probed",
            "rocminfo 逐卡 Compute Unit: %s；取最小值作保守分母"
            % "/".join(str(x) for x in cu_list),
            caliber="实测（逐卡，取最小）",
            all=cu_list, min=min(cu_list), max=max(cu_list))

    # --- wave size：rocminfo 直接给；否则按 gfx 前缀推
    wave, wave_basis = None, None
    if roc["wave"]:
        wave, wave_basis = roc["wave"], "rocminfo Wavefront Size: %d" % roc["wave"]
    elif roc["gfx"]:
        gfx = roc["gfx"]
        if gfx.startswith(("gfx10", "gfx11")):
            wave, wave_basis = 32, "推导：%s 属 RDNA 系（wave32 为常见档）" % gfx
        elif re.match(r"gfx9\d", gfx):
            wave, wave_basis = 64, "推导：%s 属 GCN/CDNA 系，wave64" % gfx
    if wave:
        vals["wave_size"] = _entry(
            wave, "lane/wave", "probed", wave_basis,
            caliber="实测" if roc["wave"] else "推导（gfx 前缀）")

    # --- 时钟：sclk（计算）/ mclk（显存）
    sclk_list = list(hys["sclk_mhz"])
    if args.sclk_mhz:
        sclk_list = [args.sclk_mhz]
    if sclk_list:
        vals["sclk_ghz"] = _entry(
            max(sclk_list) / 1000.0, "GHz", "probed",
            "hy-smi -c 逐卡 sclk: %s MHz；取最大（当前 DPM 档位，未锁频）"
            % "/".join("%g" % x for x in sclk_list),
            caliber="实测运行档位（未锁频，随 DPM 变化）",
            all_ghz=[x / 1000.0 for x in sclk_list],
            min=min(sclk_list) / 1000.0, max=max(sclk_list) / 1000.0,
            supported_ghz=[x / 1000.0 for x in hys.get("sclk_supported_mhz", [])])

    mclk_list = list(hys["mclk_mhz"])
    if args.mclk_mhz:
        mclk_list = [args.mclk_mhz]
    elif not mclk_list and roc["mem_clock_mhz"]:
        mclk_list = [roc["mem_clock_mhz"]]
    mclk = max(mclk_list) if mclk_list else None
    if mclk:
        vals["mclk_mhz"] = _entry(
            mclk, "MHz", "probed",
            "hy-smi -c 显存时钟: %s MHz" % "/".join("%g" % x for x in mclk_list),
            caliber="实测（显存时钟）")

    # --- L2 / LDS
    if roc["l2_mb"]:
        vals["l2_mb"] = _entry(roc["l2_mb"], "MB", "probed",
                               "rocminfo Cache Info L2: %g KB"
                               % (roc["l2_mb"] * 1024), caliber="实测")
    if roc["lds_kb"]:
        vals["lds_per_cu_kb"] = _entry(
            roc["lds_kb"], "KB/CU", "probed",
            "rocminfo Pool Segment=GROUP Size: %d KB" % roc["lds_kb"], caliber="实测")

    # --- HBM 峰值带宽：mclk × DDR × 位宽
    bus = args.bus_width or roc["bus_width_bits"] or hys["bus_width_bits"]
    channels = args.channels or roc["channels"] or hys["channels"] or 1
    if mclk and bus:
        total_bits = bus * channels
        bw = mclk * 1e6 * 2 * total_bits / 8 / 1e9
        vals["hbm_peak_bw_gbs"] = _entry(
            round(bw, 1), "GB/s", "probed",
            "%g MHz × 2(DDR) × %d bit（%d×%d）/ 8 = %.1f GB/s"
            % (mclk, total_bits, bus, channels, bw),
            caliber="推导（mclk × DDR × 总位宽）")
    else:
        vals["hbm_peak_bw_gbs"] = _entry(
            None, "GB/s", "probed",
            "缺总线位宽/通道数（mclk=%s, bus_width=%s）；"
            "需要 --bus-width [--channels] 才能推导" % (mclk, bus),
            caliber="无法推导")

    # --- 峰值 TFLOPS：用「最高支持 sclk」算能力上界；当前运行档位另列
    sclk_cur = vals.get("sclk_ghz", {}).get("value")
    if args.sclk_mhz:
        sclk_peak = args.sclk_mhz / 1000.0
        sclk_basis = "--sclk-mhz 指定 %g GHz" % sclk_peak
    elif hys.get("sclk_max_mhz"):
        sclk_peak = hys["sclk_max_mhz"] / 1000.0
        sclk_basis = "hy-smi -s 支持的 sclk 最高档 %g MHz" % hys["sclk_max_mhz"]
    else:
        sclk_peak = sclk_cur
        sclk_basis = ("退回当前运行档位 %g GHz（拿不到 hy-smi -s 支持档位）"
                      % sclk_cur) if sclk_cur else "缺 sclk"
    if sclk_peak:
        vals["sclk_peak_ghz"] = _entry(
            sclk_peak, "GHz", "probed", sclk_basis,
            caliber="推导（能力上界；实际运行档位见 sclk）")

    sclk = sclk_peak
    cu = vals.get("cu_count", {}).get("value")
    simd = args.simd_per_cu or roc["simd_per_cu"] or 4
    if cu and sclk and wave:
        fp32 = cu * simd * wave * 2 * sclk * 1e9 / 1e12
        vals["fp32_peak_tflops"] = _entry(
            round(fp32, 1), "TFLOPS", "probed",
            "%d CU × %d SIMD/CU × %d lane × 2 flop/FMA × %g GHz = %.1f TF（%s）"
            % (cu, simd, wave, sclk, fp32, sclk_basis),
            caliber="推导（CU×SIMD×wave×2×sclk，按能力上界时钟）")
        vals["fp16_vector_peak_tflops"] = _entry(
            round(2 * fp32, 1), "TFLOPS", "probed",
            "2 × FP32 峰值 %.1f = %.1f TF（打包 FMA；非矩阵峰值）" % (fp32, 2 * fp32),
            caliber="推导")
        vals["bf16_vector_peak_tflops"] = _entry(
            round(2 * fp32, 1), "TFLOPS", "probed",
            "2 × FP32 峰值 %.1f = %.1f TF（打包 FMA；非矩阵峰值）" % (fp32, 2 * fp32),
            caliber="推导")
    else:
        for k in ("fp32_peak_tflops", "fp16_vector_peak_tflops",
                  "bf16_vector_peak_tflops"):
            vals[k] = _entry(None, "TFLOPS", "probed",
                             "无法推导：缺 CU/sclk/wave（cu=%s sclk=%s wave=%s）"
                             % (cu, sclk, wave), caliber="无法推导")

    # 矩阵峰值：只能由 constants 里的 MFMA 标定发射率推出（ISA 不直接可读）
    rate = const_num(consts, "compute.mma_issue_peak.value")
    flops = const_num(consts, "compute.mma_bf16.flops", int)
    cal_cu = const_num(consts, "compute.cu_count.value", int)
    if rate and flops and cu and sclk and cal_cu:
        per_cu = rate / cal_cu
        mma = per_cu * cu * flops * sclk * 1e9 / 1e12
        basis = ("标定 %g MFMA/cyc @%d CU ÷%d × %d CU × %d flop/MFMA × %g GHz = %.1f TF"
                 "（发射率是实测标定值，不是 ISA 直接可读）"
                 % (rate, cal_cu, cal_cu, cu, flops, sclk, mma))
        vals["bf16_matrix_peak_tflops"] = _entry(
            round(mma, 1), "TFLOPS", "handwritten", basis,
            caliber="推导（依赖 constants 的 MFMA 标定）")
        vals["fp16_matrix_peak_tflops"] = _entry(
            round(mma, 1), "TFLOPS", "handwritten",
            basis + "；假设 fp16 与 bf16 同发射率（未单独标定）",
            caliber="推导（假设）")
    else:
        for k in ("bf16_matrix_peak_tflops", "fp16_matrix_peak_tflops"):
            vals[k] = _entry(None, "TFLOPS", "handwritten",
                             "无法推导：缺 MFMA 发射率标定（constants mma_issue_peak）"
                             "或 CU/sclk", caliber="无法推导")

    return vals


# ---------------------------------------------------------------- 对账


def build_checks(vals, consts):
    checks = []

    def add(key, hand_path, probe_key, unit, mode="point", note=None):
        hand = const_num(consts, hand_path)
        p = vals.get(probe_key, {})
        probed = p.get("value")
        base = hand_path[:-len(".value")] if hand_path.endswith(".value") else hand_path
        item = {"key": key, "handwritten": hand, "probed": probed, "unit": unit,
                "handwritten_status": consts.get(base + ".status"),
                "basis": p.get("basis"), "caliber": p.get("caliber"),
                "status": "skip", "deviation_pct": None, "note": note}
        if hand is None and probed is None:
            item["note"] = note or "两边都没有值"
            checks.append(item)
            return
        if hand is None:
            item["note"] = note or "constants.yaml 缺这一项（探测有值）"
            checks.append(item)
            return
        if probed is None:
            reason = "无法探测，退回手抄值 %s %s（未校验）" % (hand, unit)
            item["note"] = (note + "；" + reason) if note else reason
            checks.append(item)
            return
        if mode == "range" and p.get("min") is not None and p.get("max") is not None:
            lo, hi = p["min"], p["max"]
            dev = _nearest_dev(hand, lo, hi)
            item["probed"] = {"min": lo, "max": hi, "value": probed}
        else:
            dev = _nearest_dev(hand, probed, probed)
        item["deviation_pct"] = None if dev is None else round(dev, 2)
        if dev is not None and abs(dev) > TOLERANCE_PCT:
            item["status"] = "warn"
        else:
            item["status"] = "ok"
        if note:
            item["note"] = note
        checks.append(item)

    add("cu_count", "compute.cu_count.value", "cu_count", "CU", mode="range")
    add("wave_size", "compute.wave_size.value", "wave_size", "lane/wave")
    add("clock_ghz", "compute.clock_ghz.value", "sclk_ghz", "GHz", mode="range",
        note="sclk 是当前 DPM 档位（未锁频）；constants 若写某一档，只在该档内可比")
    # clock 的判据是「是否是机器支持的档位」，不是「此刻恰好跑到哪一档」
    c = checks[-1]
    sup = vals.get("sclk_ghz", {}).get("supported_ghz") or []
    if c["status"] != "skip" and sup:
        lo, hi = min(sup), max(sup)
        dev = _nearest_dev(c["handwritten"], lo, hi)
        c["deviation_pct"] = round(dev, 2)
        c["status"] = "warn" if abs(dev) > TOLERANCE_PCT else "ok"
        c["probed"] = {"min": lo, "max": hi, "value": max(sup)}
        c["note"] = ("支持档位 %s GHz；当前运行档位 %s GHz（未锁频，DPM 会变）"
                     % ("/".join("%g" % x for x in sup),
                        "/".join("%g" % x for x in vals["sclk_ghz"].get("all_ghz", []))))
    add("l2_size", "memory.l2_size.value", "l2_mb", "MB")
    add("lds_per_cu", "compute.lds_per_cu.value", "lds_per_cu_kb", "KB")
    add("stream_read_peak", "memory.stream_read_peak.value", "hbm_peak_bw_gbs", "GB/s",
        note="这是「实测流式只读」，与「mclk×位宽推导的理论峰值」口径不同；"
             "偏差大先怀疑口径/位宽假设，理论值不能当达成率分母")
    add("stream_copy_peak", "memory.stream_copy_peak.value", "hbm_peak_bw_gbs", "GB/s",
        note="copy(R+W) 没有独立探测项（mclk×位宽推的是理论峰值，不是 copy 实测）")
    add("spec_bandwidth", "memory.spec_bandwidth.value", "hbm_peak_bw_gbs", "GB/s",
        note="型号标称值，不是机器能力；不能拿它算达成率")

    def skip(key, hand_path, unit, note):
        base = hand_path[:-len(".value")] if hand_path.endswith(".value") else hand_path
        checks.append({"key": key, "handwritten": const_num(consts, hand_path),
                       "probed": None, "unit": unit,
                       "handwritten_status": consts.get(base + ".status"),
                       "basis": None, "caliber": None,
                       "status": "skip", "deviation_pct": None, "note": note})

    skip("fp8_gemm_peak_tf", "compute.fp8_gemm_peak_tf.value", "TFLOPS",
         "fp8 发射率尚未直接标定，无法由本机反推；退回手抄值（标定方法见 constants.yaml）")
    skip("mma_issue_peak", "compute.mma_issue_peak.value", "MFMA/cyc",
         "MFMA 发射率无法从 rocminfo/hy-smi 直接探测；退回手抄标定值")
    return checks


# ---------------------------------------------------------------- 输出


def _fmt_probed(v):
    if isinstance(v, dict):
        return "%s (%s..%s)" % (v.get("value"), v.get("min"), v.get("max"))
    return "%s" % (v,)


def print_human(sources, vals, checks, consts_path):
    print("== 探测源 ==")
    for name, ok, note in sources:
        print("  %-9s %s  %s" % (name, "可用" if ok else "不可用", note))

    print("\n== 机器峰值（每个数带口径）==")
    labels = [
        ("cu_count", "CU 数"), ("wave_size", "wave size"),
        ("sclk_ghz", "sclk（计算）"), ("mclk_mhz", "mclk（显存）"),
        ("l2_mb", "L2"), ("lds_per_cu_kb", "LDS/CU"),
        ("hbm_peak_bw_gbs", "HBM 峰值带宽"), ("fp32_peak_tflops", "FP32 峰值"),
        ("fp16_vector_peak_tflops", "FP16 峰值(向量)"),
        ("bf16_vector_peak_tflops", "BF16 峰值(向量)"),
        ("fp16_matrix_peak_tflops", "FP16 峰值(矩阵)"),
        ("bf16_matrix_peak_tflops", "BF16 峰值(矩阵)"),
    ]
    for key, label in labels:
        e = vals.get(key)
        if not e:
            continue
        val = e["value"]
        if val is None:
            print("  %-20s 无法探测（%s）" % (label, e["basis"]))
            continue
        print("  %-20s %-10s %s  [%s] %s"
              % (label, ("%s %s" % (val, e["unit"])), e.get("caliber", ""),
                 e["source"], e["basis"]))

    print("\n== 常数对账（阈值 %.0f%%，%s）==" % (TOLERANCE_PCT, consts_path))
    n_ok = n_warn = n_skip = 0
    for c in checks:
        hs = ("[手抄 %s]" % c["handwritten_status"]) if c.get("handwritten_status") else ""
        if c["status"] == "warn":
            n_warn += 1
            print("  ⚠ %-18s 手抄 %s %s %s ≠ 探测 %s %s（偏差 %+.1f%%）"
                  % (c["key"], c["handwritten"], c["unit"], hs,
                     _fmt_probed(c["probed"]), c["unit"], c["deviation_pct"]))
            print("      推导: %s" % c["basis"])
        elif c["status"] == "ok":
            n_ok += 1
            print("  ✅ %-18s 手抄 %s %s %s ≈ 探测 %s %s（偏差 %+.1f%%）"
                  % (c["key"], c["handwritten"], c["unit"], hs,
                     _fmt_probed(c["probed"]), c["unit"], c["deviation_pct"]))
            if c.get("note"):
                print("      口径: %s" % c["note"])
        else:
            n_skip += 1
            print("  ⚠ %-18s %s" % (c["key"], c["note"]))

    print("\n结论：%d 项一致，%d 项偏差 >%.0f%%%s。"
          % (n_ok, n_warn, TOLERANCE_PCT,
             ("，%d 项无法探测（已退回手抄值，未校验）" % n_skip) if n_skip else ""))
    if n_warn:
        print("⇒ **手抄常数已漂移**：不要静默挑一个值；先确认口径（标称/实测/推导 + 频率档）"
              "再改 constants.yaml。")
    elif n_ok == 0:
        if any(v.get("value") is not None for v in vals.values()):
            print("⇒ 已探测到机器参数，但 constants.yaml 没有可对账的项（检查 --constants 路径）。")
        else:
            print("⇒ 一个常数都没能探测：无法校验 constants.yaml（不静默取数）。")
    else:
        print("⇒ 可探测项一致；无法探测项仍是手抄值，引用时必须标明口径。")


def main():
    ap = argparse.ArgumentParser(description="从本机反推峰值常数并与 constants.yaml 对账")
    ap.add_argument("--check", action="store_true", help="与 constants.yaml 逐项对账")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--constants", default=str(DEFAULT_CONSTANTS))
    ap.add_argument("--rocminfo-output", help="读取已捕获的 rocminfo 文本（离线/测试）")
    ap.add_argument("--hysmi-output", help="读取已捕获的 hy-smi -c 文本（离线/测试）")
    ap.add_argument("--bus-width", type=int, help="整卡总位宽 (bit)")
    ap.add_argument("--channels", type=int, help="HBM 通道/栈数；总位宽 = bus-width × channels")
    ap.add_argument("--cu", type=int, help="覆盖探测到的 CU 数")
    ap.add_argument("--sclk-mhz", type=float, help="覆盖 sclk (MHz)")
    ap.add_argument("--mclk-mhz", type=float, help="覆盖 mclk (MHz)")
    ap.add_argument("--simd-per-cu", type=int, help="覆盖每 CU 的 SIMD 数（默认探测或 4）")
    ap.add_argument("--timeout", type=float, default=15.0, help="外部命令超时（秒）")
    args = ap.parse_args()

    # 探测（或读离线文本）
    if args.rocminfo_output:
        try:
            roc_text = Path(args.rocminfo_output).read_text(errors="ignore")
        except OSError as exc:
            roc_text = None
            roc_note = "读取失败: %s" % exc
        else:
            roc_note = "来自 %s" % args.rocminfo_output
    else:
        roc_text = run_command(["rocminfo"], args.timeout)
        roc_note = "实时执行 rocminfo" if roc_text else "找不到命令/无权限/超时"
    if args.hysmi_output:
        try:
            hys_text = Path(args.hysmi_output).read_text(errors="ignore")
        except OSError as exc:
            hys_text = None
            hys_note = "读取失败: %s" % exc
        else:
            hys_note = "来自 %s" % args.hysmi_output
    else:
        hys_text = run_command(["hy-smi", "-c"], args.timeout)
        hys_note = "实时执行 hy-smi -c" if hys_text else "找不到命令/无权限/超时"

    roc = parse_rocminfo(roc_text)
    hys = parse_hysmi(hys_text)
    if not args.hysmi_output:
        # 支持档位（含最高 sclk）只有 -s 给；拿不到就退回当前档位
        sclk_text = run_command(["hy-smi", "-s"], args.timeout)
        if sclk_text:
            sup = parse_supported_sclk(sclk_text)
            hys["sclk_supported_mhz"] = sup
            hys["sclk_max_mhz"] = sup[-1] if sup else None
            hys_note += "；另执行 hy-smi -s 取支持档位"
        # rocminfo 拿不到 CU 时，补一次 hy-smi --showcunum
        if not roc["cu"] and not args.cu:
            cu_text = run_command(["hy-smi", "--showcunum"], args.timeout)
            if cu_text:
                hys["cu"] = parse_hysmi(cu_text)["cu"]

    consts = load_constants(args.constants)
    vals = build_values(args, roc, hys, consts)
    checks = build_checks(vals, consts)

    sources = [
        ("rocminfo", bool(roc_text), roc_note),
        ("hy-smi", bool(hys_text), hys_note),
        ("constants", bool(consts), args.constants if consts else "读不到或为空"),
    ]

    warns = [c for c in checks if c["status"] == "warn"]
    oks = [c for c in checks if c["status"] == "ok"]
    if warns:
        code = 1
    elif not oks:
        code = 2
    else:
        code = 0

    if args.json:
        payload = {
            "sources": [{"name": n, "available": ok, "note": note}
                        for n, ok, note in sources],
            "values": vals,
            "checks": checks,
            "constants_path": args.constants,
            "tolerance_pct": TOLERANCE_PCT,
            "ok": code == 0,
            "exit_code": code,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return code

    print_human(sources, vals, checks, args.constants)
    return code


if __name__ == "__main__":
    sys.exit(main())
