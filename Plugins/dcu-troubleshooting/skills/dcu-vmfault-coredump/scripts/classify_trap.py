#!/usr/bin/env python3
"""Gate: is this crash a VMFault? Only then does the dcu-vmfault-coredump
workflow (args decode + gpucore memory forensics) apply.

VMFault = page fault / OOB memory access = TRAPSTS bit 31 (0x80000000), or
exception text mentioning "memory access fault"/"VMFault".  It is the only
trap where reading the fault-time value from the gpucore memory snapshot is
meaningful.  Other traps have different root causes (instruction/ISA, driver,
hardware) and should NOT be pushed through this pipeline.

Usage:
    python classify_trap.py CRASH.log            # scan a crash log
    python classify_trap.py --trapsts 0x80000000
    python classify_trap.py --exception 'HSA_STATUS_ERROR_EXCEPTION 0x1016'
    python classify_trap.py CRASH.log --json     # machine-readable decision
"""
import argparse
import re
import sys

VMFault_BIT = 0x80000000  # TRAPSTS bit 31
# Heuristic bits (AMD gfx s_trapsts; primary signal is bit 31 / exception text)
BITS = {0: "illegal instruction", 1: "privileged instruction",
        2: "memory violation"}

TRAPSTS_RE = re.compile(r"TRAPSTS\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)")
EXCP_RE = re.compile(r"HSA_STATUS_ERROR_EXCEPTION\s*\(?(0x[0-9a-fA-F]+)?\)?", re.I)
KEYWORDS = {
    "vmfault": "vmfault",
    "memory access fault": "vmfault",
    "memoryaccessfault": "vmfault",
    "page fault": "vmfault",
    "illegal instruction": "illegal_instruction",
    "illegalinstruction": "illegal_instruction",
    "privileged instruction": "privileged_instruction",
    "privilegedinstruction": "privileged_instruction",
    "memory violation": "memory_violation",
    "memoryviolation": "memory_violation",
    "xnack": "xnack",
    "xnwait": "xnack",
    "arithmetic": "arithmetic",
    "denorm": "arithmetic",
    "div by zero": "arithmetic",
}


def classify(trapsts=None, exc_text=""):
    """Return (trap_class, detail, applies) where applies -> use this workflow."""
    low = exc_text.lower()
    reason = []

    # 1) Explicit exception text wins when it names the trap.
    for kw, cls in KEYWORDS.items():
        if kw in low:
            reason.append(f"exception text '{kw}'")

    # 2) TRAPSTS bits.
    if trapsts is not None:
        if trapsts & VMFault_BIT:
            reason.append(f"TRAPSTS bit31 ({VMFault_BIT:#x})")
        else:
            for bit, name in BITS.items():
                if trapsts & (1 << bit):
                    reason.append(f"TRAPSTS bit{bit} ({name})")

    if not reason:
        if trapsts is None and not low:
            return "unknown", "no TRAPSTS / exception info in input", False
        return "unknown", "TRAPSTS/exception not recognized", False

    detail = "; ".join(reason)

    if any("vmfault" in r or "memory access fault" in r
           or "memoryaccessfault" in r or "page fault" in r or "bit31" in r
           for r in reason):
        return "vmfault", detail, True
    if any("memory_violation" in r or "bit2" in r for r in reason):
        return "memory_violation", detail, False
    if any("illegal_instruction" in r or "bit0" in r for r in reason):
        return "illegal_instruction", detail, False
    if any("privileged_instruction" in r or "bit1" in r for r in reason):
        return "privileged_instruction", detail, False
    if any("xnack" in r for r in reason):
        return "xnack", detail, False
    if any("arithmetic" in r for r in reason):
        return "arithmetic", detail, False
    return "other", detail, False


NOT_APPLIES = {
    "illegal_instruction": "查 kernel 编译/ISA 不匹配、gfx arch 编译选项、代码段损坏",
    "privileged_instruction": "查驱动/ROCm 运行时版本不匹配、内核用了特权指令",
    "memory_violation": "内存相关但非页错误（常为二次异常）；先看 faultinfo VA，VMFault 的 gpucore 取证意义有限",
    "xnack": "硬件/驱动问题或二次异常；不是数据越界",
    "arithmetic": "浮点异常（若未屏蔽）；不是访存越界",
    "unknown": "无 TRAPSTS/异常信息；若是 host 侧（segfault/malloc 失败）看 host 栈，不走本流程",
    "other": "未识别的 trap；不要硬套本流程",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="crash log path to scan")
    ap.add_argument("--trapsts", help="TRAPSTS value, e.g. 0x80000000")
    ap.add_argument("--exception", help="exception text, e.g. 'HSA_STATUS_ERROR_EXCEPTION 0x1016'")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    trapsts = None
    exc_text = ""
    if args.trapsts:
        trapsts = int(args.trapsts, 0)
    if args.exception:
        exc_text = args.exception
    if args.input:
        for line in open(args.input):
            m = TRAPSTS_RE.search(line)
            if m and trapsts is None:
                trapsts = int(m.group(1), 0)
            if EXCP_RE.search(line) and "EXCEPTION" in line.upper():
                exc_text = line.strip()

    cls, detail, applies = classify(trapsts, exc_text)

    if args.json:
        import json as _json
        print(_json.dumps({"trap": cls, "detail": detail, "applies": applies,
                           "trapsts": trapsts, "exception": exc_text}, indent=2))
        return 0

    trapsts_str = f"{trapsts:#x}" if trapsts is not None else "N/A"
    print(f"trap     : {cls}")
    print(f"detail   : {detail}")
    print(f"TRAPSTS  : {trapsts_str}")
    print(f"exception: {exc_text or 'N/A'}")
    print()
    if applies:
        print("=> VMFault: 走 dcu-vmfault-coredump 流程（args 解码 + gpucore 取证）")
    else:
        print(f"=> 非 VMFault: 不走本流程。{NOT_APPLIES.get(cls, '')}")
    return 0 if applies else 2


if __name__ == "__main__":
    sys.exit(main())
