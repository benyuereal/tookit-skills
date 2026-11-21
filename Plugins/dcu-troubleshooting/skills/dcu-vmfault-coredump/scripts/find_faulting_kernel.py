#!/usr/bin/env python3
"""Select the FAULTING kernel among the many `FIND MATCH KERNEL DETAILS` blocks
a VMFault queue analysis prints.

Why this exists: the DAS runtime dumps queue analysis for EVERY host queue
(stream) when one faults, and within each queue it prints a sliding window of
packets — so a single VMFault can produce hundreds of `FIND MATCH KERNEL
DETAILS` blocks (1239 in the reference case), most of them innocent context.

Selection rule (semantics, not position):
  1. `Callback: Queue 0x... aborting ... 0x1016` names the faulting queue (anchor).
  2. The faulting queue is the ONLY one whose write index != read index
     ("all command executed" queues are drained -> innocent). Cross-check its
     base_address against the callback address.
  3. Inside it, the faulting packet is the one at the READ index (the packet
     that was executing when the trap fired). The dump starts there.

Usage:
    python find_faulting_kernel.py CRASH.log [--demangle]
"""
import argparse
import re
import sys

CALLBACK_RE = re.compile(r"Callback:\s*Queue\s+(0x[0-9a-fA-F]+)\s+aborting")
QHEADER_RE = re.compile(r"=========>\s*HOSTQUEUE\s+<(0x[0-9a-fA-F]+)>:\s*VMFault HSA QUEUE ANALYSIS")
BASE_RE = re.compile(r"base_address\s+(0x[0-9a-fA-F]+)")
WR_RE = re.compile(r"write index:\s*(\d+),\s*read index:\s*(\d+)")
AQL_RE = re.compile(r"DUMP AQL\s+0x[0-9a-fA-F]+\s+PACKET WITH INDEX\s+(\d+)")
KNAME_RE = re.compile(r"kernel name:\s*(\S+)")
ARGS_RE = re.compile(r"DUMP KERNEL ARGS:\s*size:\s*(\d+)")


def read_lines(path):
    """Split on \\n like grep -n (NOT splitlines: bare \\r progress-carriage
    returns would be miscounted as extra lines). Strip trailing \\r for match."""
    raw = open(path, "rb").read().split(b"\n")
    return [ln.rstrip(b"\r").decode("utf-8", "replace") for ln in raw]


def parse(path):
    lines = read_lines(path)
    callback_addr = None
    queues = []  # list of dicts
    cur = None
    for ln, line in enumerate(lines, 1):
        m = CALLBACK_RE.search(line)
        if m and callback_addr is None:
            callback_addr = m.group(1)
        m = QHEADER_RE.search(line)
        if m:
            cur = {"addr": m.group(1), "start_line": ln, "base": None,
                   "wr": None, "aqls": [], "end_line": None}
            queues.append(cur)
            continue
        if cur is None:
            continue
        m = BASE_RE.search(line)
        if m and cur["base"] is None:
            cur["base"] = m.group(1)
            continue
        m = WR_RE.search(line)
        if m and cur["wr"] is None:
            cur["wr"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = AQL_RE.search(line)
        if m:
            cur["aqls"].append((int(m.group(1)), ln))
            continue
        m = KNAME_RE.search(line)
        if m:
            cur.setdefault("kernels", []).append((ln, m.group(1)))
            continue
        m = ARGS_RE.search(line)
        if m:
            cur.setdefault("args_lines", []).append((ln, int(m.group(1))))
        if line.strip() == "" and cur["aqls"]:
            pass  # keep scanning; blocks end at next QHEADER or EOF
    for q in queues:
        if q["aqls"]:
            q["end_line"] = q["aqls"][-1][1]
    return lines, callback_addr, queues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crash_log")
    ap.add_argument("--demangle", action="store_true", help="demangle kernel name via c++filt")
    args = ap.parse_args()

    lines, callback_addr, queues = parse(args.crash_log)
    if not queues:
        print("no HOSTQUEUE VMFault queue analysis found in log")
        return 1

    print(f"callback aborting queue : {callback_addr or 'N/A'}")
    print(f"queues analyzed         : {len(queues)}")
    pending = [q for q in queues if q["wr"] and q["wr"][0] != q["wr"][1]]
    base_hit = [q for q in queues if callback_addr and q["base"] == callback_addr]

    if len(pending) == 0:
        print("\nno queue has pending packets (write==read everywhere) — "
              "cannot pick faulting queue by W/R; check faultinfo instead")
        return 2
    if len(pending) > 1:
        print(f"\nwarning: {len(pending)} queues with pending packets")
    if base_hit and base_hit[0] not in pending:
        print("warning: callback address matches a drained queue, not a pending one")

    for q in queues:
        w, r = q["wr"] if q["wr"] else (None, None)
        mark = "  <-- FAULTING" if w != r else ""
        cb = "  (callback addr match)" if q in base_hit else ""
        print(f"  queue {q['addr']} stream/line {q['start_line']} "
              f"W/R=({w},{r}) base={q['base']}{mark}{cb}")

    faulting = pending[0]
    w, r = faulting["wr"]
    # the faulting packet is the first AQL dumped at the read index
    aql = next((idx, ln) for idx, ln in faulting["aqls"] if idx == r) \
        if any(idx == r for idx, _ in faulting["aqls"]) else faulting["aqls"][0]
    print(f"\n=== FAULTING KERNEL (queue {faulting['addr']}, "
          f"packet index {aql[0]} at line {aql[1]}) ===")
    # kernel name line within the faulting packet's block: between aql line and next AQL
    klines = [k for k in faulting.get("kernels", []) if aql[1] <= k[0]]
    if klines:
        kln, kname = klines[0]
        print(f"  line {kln}: {kname}")
        if args.demangle:
            try:
                from demangle import cppfilt, fallback
                print(f"  demangled: {cppfilt(kname) or fallback(kname)}")
            except ImportError:
                pass
    alines = [a for a in faulting.get("args_lines", []) if a[0] > aql[1]]
    if alines:
        aln, asize = alines[0]
        print(f"  args dump: line {aln}, size {asize}  "
              f"-> python parse_kernel_args.py {args.crash_log} (default ABI) or --raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
