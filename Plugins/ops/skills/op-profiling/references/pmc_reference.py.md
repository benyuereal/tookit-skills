# PMC CSV 解析参考实现

复制下面这段（标准库，无第三方依赖），按本次要看的计数器改 `COUNTERS` 即可。它只负责**选对行、排掉噪声、按实例求和、打印原值**，不解释、不判定。

选行/排除噪声/跨 SE 求和的口径与反例见 `op-profiling/SKILL.md §五 口径陷阱`；本文件只是那份口径的可复制实现，**同一套规则只有这一份**，多项目共用，别另起一套。

```python
#!/usr/bin/env python3
# PMC CSV 解析参考实现：只打印机械原值。按需改，不要当黑盒。
import argparse, csv, re

NOISE = ("_ZN2at", "_ZN7rocprim", "_ZN12_GLOBAL__N_1", "_ZN2c10", "_ZN3c10")
COUNTERS = ("GRBM_COUNT", "SQ_INSTS_VALU", "SQ_ACTIVE_INST_VALU", "SQ_INSTS_LDS",
            "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_LDS_BANK_CONFLICT",
            "SQ_WAIT_INST_LDS", "TA_TA_BUSY", "TCP_TCP_TA_DATA_STALL_CYCLES",
            "TA_FLAT_READ_WAVEFRONTS", "TCC_HIT", "TCC_MISS")
NOTE = {"SQ_INSTS_VALU": "含 MFMA，未拆分", "TA_TA_BUSY": "跨 SE 实例求和",
        "TCP_TCP_TA_DATA_STALL_CYCLES": "跨 SE 实例求和",
        "TA_FLAT_READ_WAVEFRONTS": "只统计 flat 寻址"}

def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def base(name):
    return re.sub(r"\(.*", "", name)

def total(row, key):
    """KEY 或 KEY[0..N]：有实例列就跨实例求和（TA_*/TCP_*/TCC_* 都是这种）。"""
    if key in row:
        return f(row[key])
    vals, i = [], 0
    while f"{key}[{i}]" in row:
        vals.append(f(row[f"{key}[{i}]"]) or 0.0); i += 1
    return sum(vals) if vals else None

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--kernel", help="缺省：排除噪声后单次 GRBM 最大")
ap.add_argument("--row", choices=("steady", "max-dur"), default="steady")
ap.add_argument("--cu", type=int, help="真实 CU 数；不给就打印 unknown")
a = ap.parse_args()

with open(a.csv, newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r.get("GRBM_COUNT")]
if not a.kernel:                       # 噪声前缀必须先排，否则 randperm 会夺冠
    cand = [r for r in rows if not base(r["KernelName"]).startswith(NOISE)]
    kernel = max(cand, key=lambda r: f(r["GRBM_COUNT"]) or 0.0)["KernelName"]
else:
    kernel = a.kernel
same = [r for r in rows if base(r["KernelName"]) == base(kernel)]
key = ((lambda r: (f(r["EndNs"]) or 0) - (f(r["BeginNs"]) or 0)) if a.row == "max-dur"
       else (lambda r: f(r["GRBM_COUNT"]) or 0.0))
row = min(same, key=key)               # 稳态 = 单次 GRBM 最小，绝不取首行
dur_ns = (f(row["EndNs"]) or 0) - (f(row["BeginNs"]) or 0)

noise = sorted({base(r["KernelName"]) for r in rows
                if base(r["KernelName"]).startswith(NOISE)})
print(f"kernel={kernel} launches={len(same)} row_index={row.get('Index')} "
      f"wall_ns={dur_ns:.0f} cu={a.cu if a.cu is not None else 'unknown(未给 --cu)'} "
      f"excluded_noise={len(noise)}")
print("geometry:", {k: row.get(k) for k in
                    ("grd", "wgr", "wave_size", "lds", "arch_vgpr", "sgpr", "scr")})
for k in COUNTERS:
    v = total(row, k)
    print(f"{k:32s} {'unknown(列不存在)' if v is None else format(v, ',.0f')}  {NOTE.get(k, '')}")
```
