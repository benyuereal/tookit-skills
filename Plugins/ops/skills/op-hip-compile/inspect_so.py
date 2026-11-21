#!/usr/bin/env python3
"""从编好的 HIP .so 里读设备 ELF，给出每个 kernel 的 VGPR/SGPR/LDS 与占用率。

为什么需要它：occupancy 不能猜。`aicc -shared` 产出的 .so 里，host ELF 之后还内嵌一份
设备 ELF（在 `.hip_fatbin` 段里）；设备 ELF 的 `.note` 段带 AMDGPU metadata，里面有
`.vgpr_count` / `.sgpr_count` / `.group_segment_fixed_size`。本工具把它们读出来，
按「每 CU 能驻几个 CTA」算占用率。

用法：
  python3 inspect_so.py libx.so
  python3 inspect_so.py libx.so --threads 256 --lds 28672
  python3 inspect_so.py libx.so --json

占用率公式：
  CTA/CU = min(VGPR_BUDGET / (vgpr * threads), LDS_BUDGET / lds_bytes)
VGPR_BUDGET 默认 196608（768-VGPR 模式 = 4 SIMD × 64 lane × 768），因为本项目 kernel 都用
`-mllvm=-support-768-vgprs=true`；未开该 flag 的库要传 `--vgpr-budget 65536`。
2026-11-13 校准：hbird-gemm 172 VGPR × 512 线程实测 2 CTA/CU，只有 196608 能复现（65536 算出 0）；
本仓库 LDS 探针（28672→40960 强制 1 CTA）实测 +68%，与基线 2 CTA/CU 一致。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

VGPR_BUDGET = 196608          # 768 模式：4 SIMD × 64 lane × 768（非 768 模式为 65536）
LDS_BUDGET = 64 * 1024        # 每 CU 的 LDS

READELF_CANDIDATES = ["/opt/dtk/llvm/bin/llvm-readelf", "/opt/dtk/bin/llvm-readelf", "llvm-readelf"]


def find_readelf():
    for c in READELF_CANDIDATES:
        if os.path.sep in c:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        else:
            p = shutil.which(c)
            if p:
                return p
    return None


def extract_device_elf(blob):
    """返回内嵌设备 ELF 的字节；没有就返回 None。"""
    offs = [m.start() for m in re.finditer(b"\x7fELF", blob)]
    if len(offs) < 2:
        return None
    return blob[offs[-1]:]


def parse_metadata(text):
    """把 llvm-readelf -n 的 amdhsa.kernels 列表解析成 dict 列表。"""
    records = []
    cur = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"\.([a-z_]+):\s*(.*)$", s)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "group_segment_fixed_size":
            cur = {}
            records.append(cur)
        if cur is None:
            continue
        cur[key] = val
    out = []
    for r in records:
        if "name" in r:  # 只保留真正有名字的 kernel
            out.append({
                "name": r.get("name", ""),
                "vgpr": int(r.get("vgpr_count", 0) or 0),
                "sgpr": int(r.get("sgpr_count", 0) or 0),
                "vgpr_spill": int(r.get("vgpr_spill_count", 0) or 0),
                "sgpr_spill": int(r.get("sgpr_spill_count", 0) or 0),
                "lds_fixed": int(r.get("group_segment_fixed_size", 0) or 0),
                "private": int(r.get("private_segment_fixed_size", 0) or 0),
                "kernarg": int(r.get("kernarg_segment_size", 0) or 0),
                "max_wg": int(r.get("max_flat_workgroup_size", 0) or 0),
            })
    return out


def device_text_size(readelf, elf_path):
    out = subprocess.run([readelf, "-S", elf_path], capture_output=True, text=True).stdout
    total = 0
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)", line)
        if m and m.group(1) == ".text":
            total += int(m.group(5), 16)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so")
    ap.add_argument("--threads", type=int, default=None,
                    help="每 CTA 线程数；不给则用 metadata 里的 max_flat_workgroup_size")
    ap.add_argument("--lds", type=int, default=None, help="每 CTA 动态 LDS 字节数（metadata 不含动态 LDS）")
    ap.add_argument("--vgpr-budget", type=int, default=VGPR_BUDGET,
                    help="每 CU 的 VGPR 文件；768 模式 196608（默认），非 768 模式 65536")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    readelf = find_readelf()
    if not readelf:
        print("ERROR: 找不到 llvm-readelf（试 /opt/dtk/llvm/bin/llvm-readelf）", file=sys.stderr)
        return 2

    blob = open(a.so, "rb").read()
    dev = extract_device_elf(blob)
    if dev is None:
        print("ERROR: .so 里只有一个 ELF —— 这不是 aicc 产出的设备镜像，"
              "或编译时没带 --offload-arch。", file=sys.stderr)
        return 3

    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
        f.write(dev)
        tmp = f.name
    try:
        notes = subprocess.run([readelf, "-n", tmp], capture_output=True, text=True).stdout
        kernels = parse_metadata(notes)
        tsize = device_text_size(readelf, tmp)
    finally:
        os.unlink(tmp)

    rows = []
    for k in kernels:
        threads = a.threads or k["max_wg"] or 256
        lds = a.lds if a.lds is not None else k["lds_fixed"]
        by_vgpr = a.vgpr_budget // (k["vgpr"] * threads) if k["vgpr"] * threads else 0
        by_lds = (LDS_BUDGET // lds) if lds else None
        ctas = min(x for x in (by_vgpr, by_lds) if x is not None)
        rows.append({**k, "threads": threads, "lds_used": lds,
                     "cta_per_cu_vgpr": by_vgpr, "cta_per_cu_lds": by_lds,
                     "cta_per_cu": max(ctas, 0)})

    if a.json:
        print(json.dumps({"device_text_bytes": tsize, "kernels": rows}, indent=2, ensure_ascii=False))
        return 0

    print(f"{a.so}\n  设备 ELF .text = {tsize} B")
    print(f"  {'kernel':<38} {'VGPR':>5} {'SGPR':>5} {'spill':>6} {'LDS(B)':>8} {'thr':>4} {'CTA/CU':>6}")
    for r in rows:
        spill = f"{r['vgpr_spill']}/{r['sgpr_spill']}"
        flag = "  <-- 1 CTA/CU，等内存时无可切换" if r["cta_per_cu"] == 1 else ""
        print(f"  {r['name']:<38} {r['vgpr']:>5} {r['sgpr']:>5} {spill:>6} "
              f"{r['lds_used']:>8} {r['threads']:>4} {r['cta_per_cu']:>6}{flag}")
    print("\n  口径：CTA/CU = min(%d/(VGPR*threads), %d/LDS)。"
          "\n       动态 LDS 不在 metadata 里，用 --lds 显式传。"
          % (a.vgpr_budget, LDS_BUDGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
